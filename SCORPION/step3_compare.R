#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(optparse)
  library(data.table)
  library(dplyr)
  library(biomaRt)
})

option_list <- list(
  make_option(
    "--human_root",
    type = "character",
    help = paste0(
      "Human SCORPION root. Supported layouts: ",
      "<root>/networks/<celltype>/results or <root>/<celltype>/results."
    )
  ),
  make_option(
    "--mouse_root",
    type = "character",
    help = paste0(
      "Mouse SCORPION root. Supported layouts: ",
      "<root>/networks/<celltype>/results or <root>/<celltype>/results."
    )
  ),
  make_option(
    "--out_root",
    type = "character",
    help = "Output directory for cross-species comparison."
  ),
  make_option(
    "--celltypes",
    type = "character",
    default = "Immune,Keratinocyte",
    help = "Comma-separated cell compartments [default: %default]."
  ),
  make_option(
    "--tf_support_fdr",
    type = "double",
    default = 0.1,
    help = paste0(
      "Threshold applied to the minimum supporting edge FDR for TF-centred ",
      "candidate classification [default: %default]."
    )
  ),
  make_option(
    "--edge_fdr",
    type = "double",
    default = 0.1,
    help = "Adjusted P-value threshold for edge classification [default: %default]."
  ),
  make_option(
    "--ortholog_file",
    type = "character",
    default = NULL,
    help = paste0(
      "Optional TSV containing mouse_gene and human_gene columns. ",
      "If omitted, one-to-one orthologues are queried from Ensembl BioMart."
    )
  ),
  make_option(
    "--ensembl_version",
    type = "integer",
    default = NULL,
    help = "Optional Ensembl release number used for the BioMart query."
  )
)

opt <- parse_args(OptionParser(option_list = option_list))

required_args <- c("human_root", "mouse_root", "out_root")
missing_args <- required_args[
  vapply(required_args, function(x) is.null(opt[[x]]) || !nzchar(opt[[x]]), logical(1))
]
if (length(missing_args) > 0) {
  stop("Missing required argument(s): ", paste(missing_args, collapse = ", "))
}

celltypes <- trimws(strsplit(opt$celltypes, ",", fixed = TRUE)[[1]])
celltypes <- celltypes[nzchar(celltypes)]
if (length(celltypes) == 0) {
  stop("No valid cell compartments were provided.")
}

dir.create(opt$out_root, recursive = TRUE, showWarnings = FALSE)

assert_columns <- function(x, required, label) {
  missing <- setdiff(required, colnames(x))
  if (length(missing) > 0) {
    stop(
      label,
      " is missing required column(s): ",
      paste(missing, collapse = ", "),
      "\nAvailable columns: ",
      paste(colnames(x), collapse = ", ")
    )
  }
}

resolve_results_dir <- function(root, celltype) {
  candidates <- c(
    file.path(root, "networks", celltype, "results"),
    file.path(root, celltype, "results")
  )
  existing <- candidates[dir.exists(candidates)]
  if (length(existing) == 0) {
    stop(
      "Could not find a results directory for ",
      celltype,
      " under ",
      root,
      ". Tried:\n  ",
      paste(candidates, collapse = "\n  ")
    )
  }
  normalizePath(existing[[1]], mustWork = TRUE)
}

safe_sign <- function(x) {
  out <- sign(x)
  out[!is.finite(x)] <- NA_real_
  out
}

clean_nonfinite <- function(x) {
  x[!is.finite(x)] <- NA_real_
  x
}

load_orthologues <- function() {
  if (!is.null(opt$ortholog_file) && nzchar(opt$ortholog_file)) {
    message("=== Reading one-to-one orthologue map: ", opt$ortholog_file, " ===")
    orth <- fread(opt$ortholog_file)
    assert_columns(orth, c("mouse_gene", "human_gene"), "Orthologue file")
    orth <- orth[, .(
      mouse_gene = as.character(mouse_gene),
      human_gene = as.character(human_gene)
    )]
  } else {
    message("=== Querying one-to-one orthologues from Ensembl BioMart ===")
    if (is.null(opt$ensembl_version)) {
      mouse_mart <- useEnsembl(
        biomart = "genes",
        dataset = "mmusculus_gene_ensembl"
      )
    } else {
      mouse_mart <- useEnsembl(
        biomart = "genes",
        dataset = "mmusculus_gene_ensembl",
        version = opt$ensembl_version
      )
    }
    
    orth <- getBM(
      attributes = c(
        "mgi_symbol",
        "hsapiens_homolog_associated_gene_name",
        "hsapiens_homolog_orthology_type"
      ),
      mart = mouse_mart
    ) %>%
      as_tibble() %>%
      transmute(
        mouse_gene = as.character(mgi_symbol),
        human_gene = as.character(hsapiens_homolog_associated_gene_name),
        orthology_type = as.character(hsapiens_homolog_orthology_type)
      ) %>%
      filter(
        orthology_type == "ortholog_one2one",
        mouse_gene != "",
        human_gene != ""
      ) %>%
      select(mouse_gene, human_gene)
  }
  
  # Remove duplicated rows and enforce a unique symbol-to-symbol mapping.
  orth <- orth %>%
    filter(
      !is.na(mouse_gene),
      !is.na(human_gene),
      mouse_gene != "",
      human_gene != ""
    ) %>%
    distinct(mouse_gene, human_gene) %>%
    group_by(mouse_gene) %>%
    filter(n_distinct(human_gene) == 1) %>%
    ungroup() %>%
    group_by(human_gene) %>%
    filter(n_distinct(mouse_gene) == 1) %>%
    ungroup() %>%
    arrange(human_gene, mouse_gene)
  
  if (nrow(orth) == 0) {
    stop("No valid one-to-one orthologue pairs were available.")
  }
  
  orth
}

orth <- load_orthologues()
fwrite(
  as.data.table(orth),
  file.path(opt$out_root, "one2one_orthologs.tsv"),
  sep = "\t"
)

tf_required <- c(
  "tf",
  "n_targets",
  "mean_diffMean",
  "mean_abs_diffMean",
  "mean_cohensD",
  "min_pAdj",
  "median_pAdj"
)

edge_required <- c(
  "tf",
  "target",
  "meanGroup1",
  "meanGroup2",
  "diffMean",
  "cohensD",
  "log2FoldChange",
  "pValue",
  "pAdj"
)

compare_one_celltype <- function(celltype) {
  message("=== Comparing ", celltype, " ===")
  
  human_results <- resolve_results_dir(opt$human_root, celltype)
  mouse_results <- resolve_results_dir(opt$mouse_root, celltype)
  
  human_tf_path <- file.path(human_results, "tf_stats.tsv.gz")
  mouse_tf_path <- file.path(mouse_results, "tf_stats.tsv.gz")
  human_edge_path <- file.path(human_results, "edge_stats.tsv.gz")
  mouse_edge_path <- file.path(mouse_results, "edge_stats.tsv.gz")
  
  required_files <- c(
    human_tf_path,
    mouse_tf_path,
    human_edge_path,
    mouse_edge_path
  )
  missing_files <- required_files[!file.exists(required_files)]
  if (length(missing_files) > 0) {
    stop("Missing input file(s):\n  ", paste(missing_files, collapse = "\n  "))
  }
  
  human_tf <- fread(human_tf_path)
  mouse_tf <- fread(mouse_tf_path)
  human_edge <- fread(human_edge_path)
  mouse_edge <- fread(mouse_edge_path)
  
  assert_columns(human_tf, tf_required, paste("Human", celltype, "TF table"))
  assert_columns(mouse_tf, tf_required, paste("Mouse", celltype, "TF table"))
  assert_columns(human_edge, edge_required, paste("Human", celltype, "edge table"))
  assert_columns(mouse_edge, edge_required, paste("Mouse", celltype, "edge table"))
  
  out_dir <- file.path(opt$out_root, celltype)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  
  # ---------------------------------------------------------------------------
  # TF-centred comparison
  # min_pAdj is the minimum adjusted P value among a TF's outgoing edges.
  # It is used as supporting-edge evidence, not as an independent TF-level test.
  # ---------------------------------------------------------------------------
  htf <- human_tf %>%
    transmute(
      human_gene = as.character(tf),
      human_n_targets = as.integer(n_targets),
      human_mean_delta = as.numeric(mean_diffMean),
      human_mean_abs_delta = as.numeric(mean_abs_diffMean),
      human_mean_cohensD = as.numeric(mean_cohensD),
      human_min_edge_p_adj = clean_nonfinite(as.numeric(min_pAdj)),
      human_median_edge_p_adj = clean_nonfinite(as.numeric(median_pAdj))
    ) %>%
    inner_join(orth, by = "human_gene")
  
  mtf <- mouse_tf %>%
    transmute(
      mouse_gene = as.character(tf),
      mouse_n_targets = as.integer(n_targets),
      mouse_mean_delta = as.numeric(mean_diffMean),
      mouse_mean_abs_delta = as.numeric(mean_abs_diffMean),
      mouse_mean_cohensD = as.numeric(mean_cohensD),
      mouse_min_edge_p_adj = clean_nonfinite(as.numeric(min_pAdj)),
      mouse_median_edge_p_adj = clean_nonfinite(as.numeric(median_pAdj))
    ) %>%
    inner_join(orth, by = "mouse_gene")
  
  tf_compare <- htf %>%
    inner_join(mtf, by = c("human_gene", "mouse_gene")) %>%
    mutate(
      sign_human = safe_sign(human_mean_delta),
      sign_mouse = safe_sign(mouse_mean_delta),
      direction_class = case_when(
        is.na(sign_human) | is.na(sign_mouse) ~ "not_evaluable",
        sign_human == sign_mouse & sign_human != 0 ~ "same_direction",
        sign_human != sign_mouse ~ "opposite_direction",
        TRUE ~ "weak_or_zero"
      ),
      concordance_subclass = case_when(
        direction_class == "same_direction" & human_mean_delta > 0 ~
          "concordant_increase",
        direction_class == "same_direction" & human_mean_delta < 0 ~
          "concordant_decrease",
        direction_class == "opposite_direction" ~
          "opposite_direction",
        direction_class == "weak_or_zero" ~
          "weak_or_zero",
        TRUE ~
          "not_evaluable"
      ),
      human_supported = !is.na(human_min_edge_p_adj) &
        human_min_edge_p_adj < opt$tf_support_fdr,
      mouse_supported = !is.na(mouse_min_edge_p_adj) &
        mouse_min_edge_p_adj < opt$tf_support_fdr,
      support_class = case_when(
        human_supported & mouse_supported ~ "supported_in_both",
        human_supported & !mouse_supported ~ "human_supported_only",
        !human_supported & mouse_supported ~ "mouse_supported_only",
        TRUE ~ "not_supported"
      ),
      candidate_class = case_when(
        support_class == "supported_in_both" &
          direction_class == "same_direction" ~
          "conserved_same_direction",
        support_class == "supported_in_both" &
          direction_class == "opposite_direction" ~
          "divergent_both_supported",
        support_class == "human_supported_only" ~
          "human_supported_only",
        support_class == "mouse_supported_only" ~
          "mouse_supported_only",
        TRUE ~
          "other"
      ),
      abs_delta_mean = (
        abs(human_mean_delta) + abs(mouse_mean_delta)
      ) / 2
    ) %>%
    arrange(
      factor(
        candidate_class,
        levels = c(
          "conserved_same_direction",
          "divergent_both_supported",
          "human_supported_only",
          "mouse_supported_only",
          "other"
        )
      ),
      desc(abs_delta_mean)
    )
  
  fwrite(
    as.data.table(tf_compare),
    file.path(out_dir, "tf_level_human_mouse_compare.tsv.gz"),
    sep = "\t"
  )
  fwrite(
    as.data.table(filter(tf_compare, candidate_class == "conserved_same_direction")),
    file.path(out_dir, "tf_level_conserved_candidates.tsv.gz"),
    sep = "\t"
  )
  fwrite(
    as.data.table(filter(tf_compare, candidate_class == "divergent_both_supported")),
    file.path(out_dir, "tf_level_divergent_candidates.tsv.gz"),
    sep = "\t"
  )
  fwrite(
    as.data.table(filter(tf_compare, candidate_class == "human_supported_only")),
    file.path(out_dir, "tf_level_human_supported_only.tsv.gz"),
    sep = "\t"
  )
  fwrite(
    as.data.table(filter(tf_compare, candidate_class == "mouse_supported_only")),
    file.path(out_dir, "tf_level_mouse_supported_only.tsv.gz"),
    sep = "\t"
  )
  
  # ---------------------------------------------------------------------------
  # Edge-level comparison
  # meanGroup1 is disease and meanGroup2 is healthy in step2_run.R.
  # ---------------------------------------------------------------------------
  tf_map <- orth %>%
    rename(
      human_tf = human_gene,
      mouse_tf = mouse_gene
    )
  
  target_map <- orth %>%
    rename(
      human_target = human_gene,
      mouse_target = mouse_gene
    )
  
  h_edge_map <- human_edge %>%
    transmute(
      human_tf = as.character(tf),
      human_target = as.character(target),
      human_mean_disease = as.numeric(meanGroup1),
      human_mean_healthy = as.numeric(meanGroup2),
      human_delta = as.numeric(diffMean),
      human_cohensD = as.numeric(cohensD),
      human_log2_fold_change = as.numeric(log2FoldChange),
      human_p = as.numeric(pValue),
      human_p_adj = clean_nonfinite(as.numeric(pAdj))
    ) %>%
    inner_join(tf_map, by = "human_tf") %>%
    inner_join(target_map, by = "human_target")
  
  m_edge_map <- mouse_edge %>%
    transmute(
      mouse_tf = as.character(tf),
      mouse_target = as.character(target),
      mouse_mean_disease = as.numeric(meanGroup1),
      mouse_mean_healthy = as.numeric(meanGroup2),
      mouse_delta = as.numeric(diffMean),
      mouse_cohensD = as.numeric(cohensD),
      mouse_log2_fold_change = as.numeric(log2FoldChange),
      mouse_p = as.numeric(pValue),
      mouse_p_adj = clean_nonfinite(as.numeric(pAdj))
    )
  
  edge_compare <- h_edge_map %>%
    inner_join(m_edge_map, by = c("mouse_tf", "mouse_target")) %>%
    mutate(
      sign_human = safe_sign(human_delta),
      sign_mouse = safe_sign(mouse_delta),
      direction_class = case_when(
        is.na(sign_human) | is.na(sign_mouse) ~ "not_evaluable",
        sign_human == sign_mouse & sign_human != 0 ~ "same_direction",
        sign_human != sign_mouse ~ "opposite_direction",
        TRUE ~ "weak_or_zero"
      ),
      concordance_subclass = case_when(
        direction_class == "same_direction" & human_delta > 0 ~
          "concordant_increase",
        direction_class == "same_direction" & human_delta < 0 ~
          "concordant_decrease",
        direction_class == "opposite_direction" ~
          "opposite_direction",
        direction_class == "weak_or_zero" ~
          "weak_or_zero",
        TRUE ~
          "not_evaluable"
      ),
      human_significant = !is.na(human_p_adj) &
        human_p_adj < opt$edge_fdr,
      mouse_significant = !is.na(mouse_p_adj) &
        mouse_p_adj < opt$edge_fdr,
      significance_class = case_when(
        human_significant & mouse_significant ~ "significant_in_both",
        human_significant & !mouse_significant ~ "human_significant_only",
        !human_significant & mouse_significant ~ "mouse_significant_only",
        TRUE ~ "not_significant"
      ),
      candidate_class = case_when(
        significance_class == "significant_in_both" &
          direction_class == "same_direction" ~
          "conserved_same_direction",
        significance_class == "significant_in_both" &
          direction_class == "opposite_direction" ~
          "divergent_both_significant",
        significance_class == "human_significant_only" ~
          "human_significant_only",
        significance_class == "mouse_significant_only" ~
          "mouse_significant_only",
        TRUE ~
          "other"
      ),
      abs_delta_mean = (
        abs(human_delta) + abs(mouse_delta)
      ) / 2
    ) %>%
    arrange(
      factor(
        candidate_class,
        levels = c(
          "conserved_same_direction",
          "divergent_both_significant",
          "human_significant_only",
          "mouse_significant_only",
          "other"
        )
      ),
      desc(abs_delta_mean)
    )
  
  fwrite(
    as.data.table(edge_compare),
    file.path(out_dir, "edge_level_human_mouse_compare.tsv.gz"),
    sep = "\t"
  )
  fwrite(
    as.data.table(filter(edge_compare, candidate_class == "conserved_same_direction")),
    file.path(out_dir, "edge_level_conserved_candidates.tsv.gz"),
    sep = "\t"
  )
  fwrite(
    as.data.table(filter(edge_compare, candidate_class == "divergent_both_significant")),
    file.path(out_dir, "edge_level_divergent_candidates.tsv.gz"),
    sep = "\t"
  )
  fwrite(
    as.data.table(filter(edge_compare, candidate_class == "human_significant_only")),
    file.path(out_dir, "edge_level_human_significant_only.tsv.gz"),
    sep = "\t"
  )
  fwrite(
    as.data.table(filter(edge_compare, candidate_class == "mouse_significant_only")),
    file.path(out_dir, "edge_level_mouse_significant_only.tsv.gz"),
    sep = "\t"
  )
  
  summary <- bind_rows(
    tf_compare %>%
      count(candidate_class, name = "n_records") %>%
      mutate(level = "TF"),
    edge_compare %>%
      count(candidate_class, name = "n_records") %>%
      mutate(level = "edge")
  ) %>%
    mutate(
      celltype = celltype,
      tf_support_fdr = opt$tf_support_fdr,
      edge_fdr = opt$edge_fdr
    ) %>%
    select(
      celltype,
      level,
      candidate_class,
      n_records,
      tf_support_fdr,
      edge_fdr
    )
  
  fwrite(
    as.data.table(summary),
    file.path(out_dir, "cross_species_summary.tsv"),
    sep = "\t"
  )
  
  message(
    "Completed ",
    celltype,
    ": ",
    nrow(tf_compare),
    " orthologous TF pairs and ",
    nrow(edge_compare),
    " orthologous TF-target edges."
  )
  
  summary
}

summary_list <- lapply(celltypes, compare_one_celltype)
master_summary <- bind_rows(summary_list)

fwrite(
  as.data.table(master_summary),
  file.path(opt$out_root, "cross_species_summary_all.tsv"),
  sep = "\t"
)

message("=== Cross-species comparison finished ===")
