#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(optparse)
  library(data.table)
  library(dplyr)
  library(tidyr)
  library(biomaRt)
})

option_list <- list(
  make_option("--human_root", type = "character", help = "Human SCORPION root"),
  make_option("--mouse_root", type = "character", help = "Mouse SCORPION root"),
  make_option("--out_root", type = "character", help = "Cross-species output root")
)

opt <- parse_args(OptionParser(option_list = option_list))

dir.create(opt$out_root, recursive = TRUE, showWarnings = FALSE)

message("=== Downloading one-to-one ortholog map from Ensembl via biomaRt ===")
mouse_mart <- useEnsembl("genes", dataset = "mmusculus_gene_ensembl")

orth <- getBM(
  attributes = c(
    "mgi_symbol",
    "hsapiens_homolog_associated_gene_name",
    "hsapiens_homolog_orthology_type"
  ),
  mart = mouse_mart
) %>%
  as_tibble() %>%
  rename(
    mouse_gene = mgi_symbol,
    human_gene = hsapiens_homolog_associated_gene_name,
    orthology_type = hsapiens_homolog_orthology_type
  ) %>%
  filter(
    orthology_type == "ortholog_one2one",
    mouse_gene != "",
    human_gene != ""
  ) %>%
  distinct(mouse_gene, human_gene)

fwrite(orth, file.path(opt$out_root, "one2one_orthologs.tsv"), sep = "\t")

compare_one_celltype <- function(celltype) {
  message("=== Comparing: ", celltype, " ===")

  human_tf <- fread(file.path(opt$human_root, "networks", celltype, "results", "tf_stats.tsv.gz"))
  mouse_tf <- fread(file.path(opt$mouse_root, "networks", celltype, "results", "tf_stats.tsv.gz"))

  human_edge <- fread(file.path(opt$human_root, "networks", celltype, "results", "edge_stats.tsv.gz"))
  mouse_edge <- fread(file.path(opt$mouse_root, "networks", celltype, "results", "edge_stats.tsv.gz"))

  out_dir <- file.path(opt$out_root, celltype)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  # TF-level
  htf <- human_tf %>%
    rename(human_gene = tf) %>%
    inner_join(orth, by = "human_gene") %>%
    rename(
      human_n_targets = n_targets,
      human_mean_healthy = mean_healthy,
      human_mean_disease = mean_disease,
      human_delta = delta_disease_minus_healthy,
      human_p = p_value,
      human_p_adj = p_adj
    )

  mtf <- mouse_tf %>%
    rename(mouse_gene = tf) %>%
    inner_join(orth, by = "mouse_gene") %>%
    rename(
      mouse_n_targets = n_targets,
      mouse_mean_healthy = mean_healthy,
      mouse_mean_disease = mean_disease,
      mouse_delta = delta_disease_minus_healthy,
      mouse_p = p_value,
      mouse_p_adj = p_adj
    )

  tf_compare <- htf %>%
    inner_join(mtf, by = c("human_gene", "mouse_gene")) %>%
    mutate(
      sign_human = sign(human_delta),
      sign_mouse = sign(mouse_delta),
      direction_class = case_when(
        sign_human == sign_mouse & sign_human != 0 ~ "same_direction",
        sign_human != sign_mouse ~ "opposite_direction",
        TRUE ~ "weak_or_zero"
      ),
      abs_delta_mean = (abs(human_delta) + abs(mouse_delta)) / 2
    ) %>%
    arrange(desc(abs_delta_mean))

  fwrite(tf_compare, file.path(out_dir, "tf_level_human_mouse_compare.tsv.gz"), sep = "\t")

  # Edge-level
  h_edge_map <- human_edge %>%
    rename(human_tf = tf, human_target = target) %>%
    inner_join(orth %>% rename(human_tf = human_gene, mouse_tf = mouse_gene), by = "human_tf") %>%
    inner_join(orth %>% rename(human_target = human_gene, mouse_target = mouse_gene), by = "human_target") %>%
    rename(
      human_mean_healthy = mean_healthy,
      human_mean_disease = mean_disease,
      human_delta = delta_disease_minus_healthy,
      human_p = p_value,
      human_p_adj = p_adj
    )

  m_edge_map <- mouse_edge %>%
    rename(mouse_tf = tf, mouse_target = target) %>%
    rename(
      mouse_mean_healthy = mean_healthy,
      mouse_mean_disease = mean_disease,
      mouse_delta = delta_disease_minus_healthy,
      mouse_p = p_value,
      mouse_p_adj = p_adj
    )

  edge_compare <- h_edge_map %>%
    inner_join(m_edge_map, by = c("mouse_tf", "mouse_target")) %>%
    mutate(
      sign_human = sign(human_delta),
      sign_mouse = sign(mouse_delta),
      direction_class = case_when(
        sign_human == sign_mouse & sign_human != 0 ~ "same_direction",
        sign_human != sign_mouse ~ "opposite_direction",
        TRUE ~ "weak_or_zero"
      ),
      abs_delta_mean = (abs(human_delta) + abs(mouse_delta)) / 2
    ) %>%
    arrange(desc(abs_delta_mean))

  fwrite(edge_compare, file.path(out_dir, "edge_level_human_mouse_compare.tsv.gz"), sep = "\t")

  # Strong candidates
  strong_tf <- tf_compare %>%
    filter(
      !is.na(human_p_adj), !is.na(mouse_p_adj),
      human_p_adj < 0.1, mouse_p_adj < 0.1,
      direction_class == "same_direction"
    ) %>%
    arrange(desc(abs_delta_mean))

  strong_edge <- edge_compare %>%
    filter(
      !is.na(human_p_adj), !is.na(mouse_p_adj),
      human_p_adj < 0.1, mouse_p_adj < 0.1,
      direction_class == "same_direction"
    ) %>%
    arrange(desc(abs_delta_mean))

  fwrite(strong_tf, file.path(out_dir, "tf_level_conserved_candidates.tsv.gz"), sep = "\t")
  fwrite(strong_edge, file.path(out_dir, "edge_level_conserved_candidates.tsv.gz"), sep = "\t")
}

compare_one_celltype("Immune")
compare_one_celltype("Keratinocyte")

message("=== Cross-species comparison finished ===")
