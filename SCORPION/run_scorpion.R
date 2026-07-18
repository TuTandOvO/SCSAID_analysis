#!/usr/bin/env Rscript

options(future.globals.maxSize = 16 * 1024^3)
options(timeout = 1800)

suppressPackageStartupMessages({
  library(optparse)
  library(Matrix)
  library(data.table)
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(future)
  library(SCORPION)
  library(decoupleR)
})

option_list <- list(
  make_option("--organism", type = "character", help = "human or mouse"),
  make_option("--input_dir", type = "character", help = "Input directory from python export"),
  make_option("--out_dir", type = "character", help = "Output directory"),
  make_option("--healthy_label", type = "character", help = "Healthy condition label"),
  make_option("--disease_label", type = "character", help = "Disease condition label"),
  make_option("--ncores", type = "integer", default = 1),
  make_option("--min_cells", type = "integer", default = 30),
  make_option("--gamma", type = "integer", default = 10),
  make_option("--npc", type = "integer", default = 25),
  make_option("--assoc", type = "character", default = "pearson"),
  make_option("--dorothea_levels", type = "character", default = "A,B,C"),
  make_option("--string_root", type = "character", default = "/gpfsdata/home/renyixiang/SkinDB/10X/scorpion"),
  make_option("--string_score_threshold", type = "integer", default = 200),
  make_option("--dorothea_signed", type = "character", default = "TRUE"),
  make_option("--future_max_gb", type = "double", default = 16),
  make_option("--runscorpion_patch_file", type = "character", default = NULL,
              help = "Optional path to a dumped/patched runSCORPION.R file"),
  make_option("--scorpion_source_dir", type = "character", default = NULL,
              help = "Optional path to local SCORPION source tree used to override installed runSCORPION()")
)

opt <- parse_args(OptionParser(option_list = option_list))

future_max_bytes <- opt$future_max_gb * 1024^3
options(future.globals.maxSize = future_max_bytes)

if (opt$ncores > 1) {
  future::plan(
    future::multisession,
    workers = opt$ncores,
    maxSizeOfObjects = future_max_bytes
  )
} else {
  future::plan(
    future::sequential,
    maxSizeOfObjects = future_max_bytes
  )
}

log_normalize_data <- utils::getFromNamespace("log_normalize_data", "SCORPION")
remove_batch <- utils::getFromNamespace("remove_batch", "SCORPION")

runSCORPION <- function(gexMatrix,
                        tfMotifs,
                        ppiNet,
                        cellsMetadata,
                        groupBy,
                        normalizeData = TRUE,
                        removeBatchEffect = FALSE,
                        batch = NULL,
                        minCells = 30,
                        computingEngine = "cpu",
                        nCores = 1,
                        gammaValue = 10,
                        nPC = 25,
                        assocMethod = "pearson",
                        alphaValue = 0.1,
                        hammingValue = 0.001,
                        nIter = Inf,
                        outNet = "regNet",
                        zScaling = TRUE,
                        showProgress = TRUE,
                        randomizationMethod = "None",
                        scaleByPresent = FALSE,
                        filterExpr = FALSE) {
  cellsMetadata <- as.data.frame(cellsMetadata, stringsAsFactors = FALSE)

  if (ncol(gexMatrix) != nrow(cellsMetadata)) {
    cli::cli_abort("gexMatrix must have the same number of columns as cellsMetadata has rows")
  }
  if (!all(groupBy %in% colnames(cellsMetadata))) {
    cli::cli_abort("groupBy columns not found in cellsMetadata: {paste(setdiff(groupBy, colnames(cellsMetadata)), collapse=', ')}")
  }
  if (!is.numeric(minCells) || minCells < 1) {
    cli::cli_abort("minCells must be a positive integer")
  }
  if (removeBatchEffect && is.null(batch)) {
    cli::cli_abort("batch must be provided when removeBatchEffect = TRUE")
  }
  if (!is.null(batch) && length(batch) != ncol(gexMatrix)) {
    cli::cli_abort("batch must have the same length as gexMatrix columns (cells)")
  }
  if (alphaValue < 0 || alphaValue > 1) {
    cli::cli_abort("alphaValue must be a numeric value between 0 and 1")
  }

  if (showProgress) {
    cli::cli_h1("SCORPION")
  }

  if (normalizeData) {
    if (showProgress) {
      cli::cli_alert_success("Normalizing data (log scale)")
    }
    gexMatrix <- log_normalize_data(gexMatrix)
  }

  if (removeBatchEffect) {
    if (showProgress) {
      cli::cli_alert_success("Correcting for batch effects")
    }
    mean_expr <- apply(gexMatrix, 1, median)
    gexMatrix <- remove_batch(X = gexMatrix, batch = batch)
    gexMatrix <- gexMatrix + mean_expr
  }

  min_cells <- max(minCells, 30)
  collapsedGroup <- apply(cellsMetadata[, groupBy, drop = FALSE], 1, function(X) {
    paste0(X, collapse = "--")
  })
  metadata <- data.frame(cell_id = colnames(gexMatrix), network_id = collapsedGroup)
  metadata <- metadata %>%
    dplyr::group_by(.data$network_id) %>%
    dplyr::mutate(n_cells = length(.data$cell_id))

  total_net <- length(unique(metadata$network_id))
  metadata <- metadata %>% dplyr::filter(.data$n_cells >= min_cells)
  filtered_net <- length(unique(metadata$network_id))

  if (showProgress) {
    cli::cli_alert_info(paste0(total_net, " networks requested"))
    cli::cli_alert_success(paste0(filtered_net, " networks meet the minimum cell requirement (", min_cells, ")"))
  }

  compute_network <- function(idx) {
    selected_network <- network_ids[idx]
    selected_cells <- metadata %>% dplyr::filter(.data$network_id %in% selected_network)
    selected_cells <- gexMatrix[, selected_cells$cell_id]

    scorpion(
      gexMatrix = selected_cells,
      tfMotifs = as.data.frame(tfMotifs),
      ppiNet = as.data.frame(ppiNet),
      computingEngine = computingEngine,
      nCores = 1,
      gammaValue = gammaValue,
      nPC = nPC,
      assocMethod = assocMethod,
      alphaValue = alphaValue,
      hammingValue = hammingValue,
      nIter = nIter,
      outNet = outNet,
      zScaling = zScaling,
      showProgress = FALSE,
      randomizationMethod = randomizationMethod,
      scaleByPresent = scaleByPresent,
      filterExpr = filterExpr
    )[[outNet]]
  }

  network_ids <- unique(metadata$network_id)
  future_max_size <- getOption("future.globals.maxSize", NULL)

  if (nCores > 1 && is.null(future_max_size)) {
    cli::cli_abort("future.globals.maxSize must be set when nCores > 1")
  }

  if (!is.null(future_max_size)) {
    options(future.globals.maxSize = future_max_size)
  }

  if (nCores > 1) {
    old_plan <- future::plan(
      future::multisession,
      workers = nCores,
      maxSizeOfObjects = future_max_size
    )
  } else {
    old_plan <- future::plan(
      future::sequential,
      maxSizeOfObjects = future_max_size
    )
  }
  on.exit(future::plan(old_plan), add = TRUE)

  if (showProgress) {
    cli::cli_alert_info("Computing networks")
    if (nCores > 1) {
      cli::cli_alert_info(paste0("Using ", nCores, " cores for parallel processing"))
    }
  }

  network_matrices <- furrr::future_map(
    seq_along(network_ids),
    compute_network,
    .options = furrr::furrr_options(seed = TRUE),
    .progress = TRUE
  )

  if (showProgress) {
    cli::cli_alert_success("Networks successfully constructed")
  }

  first_net <- network_matrices[[1]]
  tf_target_df <- as.data.frame(as.table(first_net))[, 1:2]
  colnames(tf_target_df) <- c("tf", "target")

  weight_matrix <- vapply(
    network_matrices,
    as.vector,
    numeric(length(first_net))
  )
  colnames(weight_matrix) <- network_ids

  networks <- data.frame(
    tf = as.character(tf_target_df$tf),
    target = as.character(tf_target_df$target),
    weight_matrix,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )

  if (showProgress) {
    cli::cli_alert_success("Networks successfully combined")
  }

  networks
}

message("=== Using embedded patched runSCORPION implementation ===")

dir.create(opt$out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(opt$out_dir, "priors"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(opt$out_dir, "results"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(opt$out_dir, "qc"), recursive = TRUE, showWarnings = FALSE)

message("=== Reading exported input ===")
gex <- readMM(file.path(opt$input_dir, "gex_genes_x_cells.mtx"))
genes <- fread(file.path(opt$input_dir, "genes.tsv"), header = FALSE)$V1
cells <- fread(file.path(opt$input_dir, "cells.tsv"), header = FALSE)$V1
meta <- fread(file.path(opt$input_dir, "metadata.tsv"))
meta <- as.data.frame(meta, stringsAsFactors = FALSE)

stopifnot(nrow(gex) == length(genes))
stopifnot(ncol(gex) == length(cells))
stopifnot(nrow(meta) == length(cells))
stopifnot(all(meta$cell_barcode == cells))

rownames(gex) <- genes
colnames(gex) <- cells

meta <- meta %>%
  mutate(
    sample = as.character(sample),
    condition = as.character(condition),
    Gross_Map = as.character(Gross_Map),
    group_id = paste(sample, condition, sep = "__GROUP__")
  )

fwrite(meta, file.path(opt$out_dir, "qc", "metadata_used.tsv"), sep = "\t")

group_counts <- meta %>%
  count(group_id, sample, condition, name = "n_cells") %>%
  arrange(condition, sample)

fwrite(group_counts, file.path(opt$out_dir, "qc", "group_counts.tsv"), sep = "\t")

message("=== Preparing TF prior from DoRothEA ===")
organism_for_doro <- ifelse(opt$organism == "human", "human", "mouse")
levels_vec <- strsplit(opt$dorothea_levels, ",", fixed = TRUE)[[1]]
use_signed <- tolower(opt$dorothea_signed) == "true"

dor <- decoupleR::get_dorothea(
  organism = organism_for_doro,
  levels = levels_vec
)

if (use_signed) {
  tf_prior <- dor %>%
    transmute(
      source_genesymbol = as.character(source),
      target_genesymbol = as.character(target),
      weight = as.numeric(mor)
    )
} else {
  tf_prior <- dor %>%
    transmute(
      source_genesymbol = as.character(source),
      target_genesymbol = as.character(target),
      weight = abs(as.numeric(mor))
    )
}

genes_in_matrix <- rownames(gex)

tf_prior <- tf_prior %>%
  filter(
    source_genesymbol %in% genes_in_matrix,
    target_genesymbol %in% genes_in_matrix
  ) %>%
  distinct()

if (nrow(tf_prior) == 0) {
  stop("TF prior is empty after filtering to matrix genes.")
}

fwrite(
  tf_prior,
  file.path(opt$out_dir, "priors", "tf_prior_dorothea.tsv"),
  sep = "\t"
)

message("=== Preparing TF-TF PPI prior from local STRING files ===")

species_prefix <- ifelse(opt$organism == "human", "9606", "10090")
species_dir <- ifelse(opt$organism == "human", "human", "mouse")

links_file <- file.path(
  opt$string_root,
  species_dir,
  paste0(species_prefix, ".protein.links.v12.0.txt.gz")
)

info_file <- file.path(
  opt$string_root,
  species_dir,
  paste0(species_prefix, ".protein.info.v12.0.txt.gz")
)

if (!file.exists(links_file)) {
  stop("Missing local STRING links file: ", links_file)
}
if (!file.exists(info_file)) {
  stop("Missing local STRING info file: ", info_file)
}

string_info <- fread(info_file)
string_links <- fread(links_file)

required_link_cols <- c("protein1", "protein2", "combined_score")
if (!all(required_link_cols %in% colnames(string_links))) {
  stop(
    "Unexpected STRING links columns. Need: ",
    paste(required_link_cols, collapse = ", "),
    "\nFound: ",
    paste(colnames(string_links), collapse = ", ")
  )
}

string_id_col <- NULL
if ("protein_external_id" %in% colnames(string_info)) {
  string_id_col <- "protein_external_id"
} else if ("#string_protein_id" %in% colnames(string_info)) {
  string_id_col <- "#string_protein_id"
} else if ("string_protein_id" %in% colnames(string_info)) {
  string_id_col <- "string_protein_id"
} else {
  stop(
    "Unexpected STRING info columns. Need one of protein_external_id / #string_protein_id / string_protein_id",
    "\nFound: ",
    paste(colnames(string_info), collapse = ", ")
  )
}

if (!("preferred_name" %in% colnames(string_info))) {
  stop(
    "STRING info file missing preferred_name column.\nFound: ",
    paste(colnames(string_info), collapse = ", ")
  )
}

tfs <- sort(unique(tf_prior$source_genesymbol))

mapped_tfs <- string_info %>%
  transmute(
    gene = as.character(preferred_name),
    STRING_id = as.character(.data[[string_id_col]])
  ) %>%
  filter(gene %in% tfs) %>%
  distinct()

if (nrow(mapped_tfs) == 0) {
  stop("No TFs mapped to local STRING protein IDs.")
}

ppi_prior <- string_links %>%
  filter(combined_score >= opt$string_score_threshold) %>%
  inner_join(
    mapped_tfs %>% select(source_genesymbol = gene, protein1 = STRING_id),
    by = "protein1"
  ) %>%
  inner_join(
    mapped_tfs %>% select(target_genesymbol = gene, protein2 = STRING_id),
    by = "protein2"
  ) %>%
  transmute(
    source_genesymbol = source_genesymbol,
    target_genesymbol = target_genesymbol,
    weight = combined_score / 1000
  ) %>%
  filter(source_genesymbol != target_genesymbol) %>%
  mutate(
    a = pmin(source_genesymbol, target_genesymbol),
    b = pmax(source_genesymbol, target_genesymbol)
  ) %>%
  group_by(a, b) %>%
  summarise(weight = max(weight), .groups = "drop") %>%
  transmute(
    source_genesymbol = a,
    target_genesymbol = b,
    weight = weight
  ) %>%
  distinct()

if (nrow(ppi_prior) == 0) {
  stop("PPI prior is empty after TF-only filtering.")
}

fwrite(
  ppi_prior,
  file.path(opt$out_dir, "priors", "ppi_prior_string_local.tsv"),
  sep = "\t"
)

message("=== Running SCORPION ===")
nets <- runSCORPION(
  gexMatrix = gex,
  tfMotifs = tf_prior,
  ppiNet = ppi_prior,
  cellsMetadata = meta,
  groupBy = "group_id",
  normalizeData = TRUE,
  removeBatchEffect = FALSE,
  batch = NULL,
  minCells = opt$min_cells,
  computingEngine = "cpu",
  nCores = opt$ncores,
  gammaValue = opt$gamma,
  nPC = opt$npc,
  assocMethod = opt$assoc,
  alphaValue = 0.1,
  hammingValue = 0.001,
  nIter = Inf,
  outNet = "regNet",
  zScaling = TRUE,
  showProgress = TRUE,
  randomizationMethod = "None",
  scaleByPresent = FALSE,
  filterExpr = FALSE
)

fwrite(
  as.data.table(nets),
  file.path(opt$out_dir, "results", "networks_wide.tsv.gz"),
  sep = "\t"
)

message("=== Edge-level statistics with official testEdges() ===")
network_cols <- setdiff(colnames(nets), c("tf", "target"))

parse_condition <- function(x) {
  parts <- strsplit(x, "__GROUP__", fixed = TRUE)[[1]]
  if (length(parts) != 2) return(NA_character_)
  parts[2]
}

col_condition <- setNames(
  vapply(network_cols, parse_condition, character(1)),
  network_cols
)

healthy_cols <- names(col_condition)[col_condition == opt$healthy_label]
disease_cols <- names(col_condition)[col_condition == opt$disease_label]

if (length(healthy_cols) == 0 || length(disease_cols) == 0) {
  stop("Could not find both healthy and disease network columns.")
}

edge_stats <- testEdges(
  networksDF = nets,
  testType = "two.sample",
  group1 = disease_cols,
  group2 = healthy_cols,
  moderateVariance = TRUE,
  empiricalNull = TRUE,
  padjustMethod = "BH",
  nCores = opt$ncores
)

fwrite(
  as.data.table(edge_stats),
  file.path(opt$out_dir, "results", "edge_stats.tsv.gz"),
  sep = "\t"
)

message("=== TF-level summary derived from edge_stats ===")

edge_stats_dt <- as.data.table(edge_stats)

tf_col <- if ("tf" %in% names(edge_stats_dt)) {
  "tf"
} else if ("source_genesymbol" %in% names(edge_stats_dt)) {
  "source_genesymbol"
} else {
  stop("Could not identify TF column in testEdges output.")
}

diff_col <- if ("diffMean" %in% names(edge_stats_dt)) {
  "diffMean"
} else if ("log2FoldChange" %in% names(edge_stats_dt)) {
  "log2FoldChange"
} else {
  NA_character_
}

cohens_col <- if ("cohensD" %in% names(edge_stats_dt)) "cohensD" else NA_character_
padj_col <- if ("pAdj" %in% names(edge_stats_dt)) "pAdj" else if ("padj" %in% names(edge_stats_dt)) "padj" else NA_character_

tf_stats <- edge_stats_dt %>%
  as_tibble() %>%
  group_by(.data[[tf_col]]) %>%
  summarise(
    n_targets = n(),
    mean_diffMean = if (!is.na(diff_col)) mean(.data[[diff_col]], na.rm = TRUE) else NA_real_,
    mean_abs_diffMean = if (!is.na(diff_col)) mean(abs(.data[[diff_col]]), na.rm = TRUE) else NA_real_,
    mean_cohensD = if (!is.na(cohens_col)) mean(.data[[cohens_col]], na.rm = TRUE) else NA_real_,
    min_pAdj = if (!is.na(padj_col)) suppressWarnings(min(.data[[padj_col]], na.rm = TRUE)) else NA_real_,
    median_pAdj = if (!is.na(padj_col)) suppressWarnings(median(.data[[padj_col]], na.rm = TRUE)) else NA_real_,
    .groups = "drop"
  ) %>%
  rename(tf = 1) %>%
  arrange(min_pAdj, desc(mean_abs_diffMean))

fwrite(
  tf_stats,
  file.path(opt$out_dir, "results", "tf_stats.tsv.gz"),
  sep = "\t"
)

message("=== Finished ===")
