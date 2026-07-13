library(CellChat)
library(Matrix)
library(patchwork)
library(ggplot2)
library(future)
library(ComplexHeatmap)
library(NMF)
library(ggalluvial)
library(reticulate)
reticulate::use_python("~/miniconda3/envs/SRTP/bin/python", required = TRUE)

# ============================================================
# 全局设置
# ============================================================
options(future.globals.maxSize = 250 * 1024^3)   # 250GB，充分利用 300G+ 内存
options(future.rng.onMisuse = "ignore")
options(stringsAsFactors = FALSE)

n_workers <- 24  # 28核节点，留4个给系统（CellChat 计算密集，多分配）

set_future_parallel <- function(workers = n_workers) {
  options(future.globals.maxSize = 250 * 1024^3)
  future::plan("multicore", workers = workers)
}

set_future_sequential <- function() {
  options(future.globals.maxSize = 250 * 1024^3)
  future::plan("sequential")
}

# ============================================================
# 配置
# ============================================================
h5ad_paths <- list(
  human = "/gpfsdata/home/renyixiang/SkinDB/10X/human/Annotation/Annotated/human_with_all_leiden_10threshold_annotated_updated.h5ad",
  mouse = "/gpfsdata/home/renyixiang/SkinDB/10X/mouse/Annotation/Annotated/mouse_with_all_leiden_10threshold_annotated_updated.h5ad"
)

datasets <- list(
  human = list(
    healthy_cond  = "Healthy",
    disease_cond  = "psoriasis",
    disease_name  = "Psoriasis",
    db            = "human",
    filter_skin   = TRUE,
    skin_value    = "whole skin"
  ),
  mouse = list(
    healthy_cond  = "Healthy",
    disease_cond  = "IMQ-induced psoriasis",
    disease_name  = "IMQ-induced Psoriasis",
    db            = "mouse",
    filter_skin   = FALSE,
    skin_value    = NULL
  )
)

run_queue <- list(
  list(species = "human", ct_col = "Gross_Map"),
  list(species = "mouse", ct_col = "Gross_Map"),
  list(species = "human", ct_col = "Fine_Map"),
  list(species = "mouse", ct_col = "Fine_Map")
)

output_dir <- "/gpfsdata/home/renyixiang/SkinDB/CellChat/results"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# ============================================================
# 从 h5ad 读取单个条件的数据
# 使用 adata.X（已 normalize + log1p 的数据）
# ============================================================
read_h5ad_condition <- function(h5ad_path, condition_val, celltype_col,
                                filter_skin = FALSE, skin_value = NULL) {
  ad <- reticulate::import("anndata")
  scipy_sparse <- reticulate::import("scipy.sparse")

  cat("    Reading h5ad:", h5ad_path, "\n")
  adata <- ad$read_h5ad(h5ad_path, backed = "r")

  # ---------- 构建细胞过滤掩码 ----------
  mask <- adata$obs[["condition"]] == condition_val
  if (filter_skin) {
    mask <- mask & (adata$obs[["Skin_location"]] == skin_value)
  }
  mask <- as.logical(mask)

  cat("    Filter '", condition_val, "'",
      if (filter_skin) paste0(" & Skin_location='", skin_value, "'") else "",
      ": ", sum(mask), "/", length(mask), " cells\n", sep = "")

  if (sum(mask) == 0) stop("No cells passed filter for condition: ", condition_val)

  meta <- as.data.frame(adata$obs[mask, ])

  # ---------- 取 adata.X（normalized data） ----------
  # CellChat 要求 normalized (NOT count) 数据
  # adata.X 应已经过 sc.pp.normalize_total + sc.pp.log1p
  # backed 模式下需要用 numpy mask 做 subset 再 to_memory
  np <- reticulate::import("numpy")
  mask_np <- np$array(mask)
  cat("    Subsetting adata to memory...\n")
  adata_sub <- adata[mask_np]$to_memory()

  X_sub <- adata_sub$X

  # reticulate 可能自动转为 R 的 dgCMatrix/dgRMatrix，也可能保留为 Python 对象
  if (inherits(X_sub, "sparseMatrix")) {
    # 已经是 R sparse matrix，直接转置为 genes x cells
    X <- Matrix::t(X_sub)
  } else if (inherits(X_sub, "matrix") || inherits(X_sub, "array")) {
    # 稠密矩阵，转为稀疏再转置
    X <- Matrix::t(as(X_sub, "dgCMatrix"))
  } else {
    # 仍然是 Python scipy sparse 对象
    X_sub <- X_sub$tocsr()
    nz <- X_sub$nonzero()
    X <- Matrix::sparseMatrix(
      i    = as.integer(reticulate::py_to_r(nz[[1]])) + 1L,
      j    = as.integer(reticulate::py_to_r(nz[[2]])) + 1L,
      x    = as.numeric(reticulate::py_to_r(X_sub$data)),
      dims = c(as.integer(X_sub$shape[[1]]), as.integer(X_sub$shape[[2]]))
    )
    X <- Matrix::t(X)  # 转为 genes x cells
  }

  gene_names <- as.character(reticulate::py_to_r(adata_sub$var_names$tolist()))
  cell_names <- rownames(meta)
  rownames(X) <- gene_names
  colnames(X) <- cell_names

  # 保留真实的 sample 信息（如果 obs 中有 sample 列则使用，否则设为 sample1）
  if ("sample" %in% colnames(meta)) {
    meta$samples <- factor(meta$sample)
  } else if ("batch" %in% colnames(meta)) {
    meta$samples <- factor(meta$batch)
  } else {
    meta$samples <- factor("sample1")
  }

  cat("    Matrix (genes x cells):", nrow(X), "x", ncol(X), "\n")
  cat("    Cell types in", celltype_col, ":", length(unique(meta[[celltype_col]])), "\n")

  # 去除未使用的 factor levels（不同 condition 可能 cell type 组成不同）
  for (col in colnames(meta)) {
    if (is.factor(meta[[col]])) {
      meta[[col]] <- droplevels(meta[[col]])
    }
  }

  # ---------- 释放 Python 对象 ----------
  tryCatch(adata$file$close(), error = function(e) NULL)
  rm(adata, adata_sub, X_sub, mask_np, np)
  py_gc <- reticulate::import("gc")
  py_gc$collect()
  rm(py_gc)
  gc()

  return(list(X = X, meta = meta))
}

# ============================================================
# 创建 CellChat 对象并完成推断
# ============================================================
create_cellchat <- function(h5ad_path, condition_val, celltype_col, species_db,
                            filter_skin = FALSE, skin_value = NULL) {

  data <- read_h5ad_condition(h5ad_path, condition_val, celltype_col,
                              filter_skin, skin_value)

  cellchat <- createCellChat(object = data$X, meta = data$meta, group.by = celltype_col)

  # 释放原始数据
  rm(data); gc()

  # 去除空的 cell type levels，避免 computeCommunProb 报错
  cellchat@idents <- droplevels(cellchat@idents)

  # ---------- 设置数据库（使用全库） ----------
  if (species_db == "human") {
    cellchat@DB <- CellChatDB.human
  } else {
    cellchat@DB <- CellChatDB.mouse
  }

  cellchat <- subsetData(cellchat)

  # ---------- 推断通信 ----------
  set_future_parallel()
  cellchat <- identifyOverExpressedGenes(cellchat)
  cellchat <- identifyOverExpressedInteractions(cellchat)

  ptm <- Sys.time()
  cellchat <- computeCommunProb(cellchat, type = "triMean")
  cellchat <- filterCommunication(cellchat, min.cells = 10)
  cellchat <- computeCommunProbPathway(cellchat)
  cellchat <- aggregateNet(cellchat)
  cat("    Inference time:", round(as.numeric(Sys.time() - ptm, units = "mins"), 1), "min\n")

  # ---------- Centrality 分析 ----------
  cellchat <- netAnalysis_computeCentrality(cellchat, slot.name = "netP")

  set_future_sequential()
  return(cellchat)
}

# ============================================================
# 辅助函数：获取 top N signaling pathways
# ============================================================
get_top_pathways <- function(cellchat, n = 10) {
  df <- subsetCommunication(cellchat, slot.name = "netP")
  if (nrow(df) == 0) return(character(0))
  pw_strength <- aggregate(prob ~ pathway_name, data = df, FUN = sum)
  pw_strength <- pw_strength[order(-pw_strength$prob), ]
  head(pw_strength$pathway_name, n)
}

# ============================================================
# 单条件可视化（完整版，对照官方 tutorial）
# ============================================================
plot_single_cellchat <- function(cellchat, run_name, cond_name, run_dir) {

  prefix    <- paste0(run_name, "_", gsub(" ", "_", cond_name))
  groupSize <- as.numeric(table(cellchat@idents))

  # ---- 01 总体圆形图（interaction count + strength）----
  pdf(file.path(run_dir, paste0(prefix, "_01_circle_overview.pdf")), width = 14, height = 7)
  par(mfrow = c(1, 2), xpd = TRUE)
  netVisual_circle(cellchat@net$count, vertex.weight = groupSize, weight.scale = TRUE,
                   label.edge = FALSE, title.name = paste0(cond_name, " - Number of interactions"))
  netVisual_circle(cellchat@net$weight, vertex.weight = groupSize, weight.scale = TRUE,
                   label.edge = FALSE, title.name = paste0(cond_name, " - Interaction strength"))
  dev.off()
  cat("      Saved: 01_circle_overview\n")

  # ---- 02 per celltype 圆形图 ----
  mat     <- cellchat@net$weight
  n_ct    <- nrow(mat)
  nc      <- 4
  nr      <- ceiling(n_ct / nc)
  pdf(file.path(run_dir, paste0(prefix, "_02_per_celltype_circle.pdf")), width = nc*4, height = nr*4)
  par(mfrow = c(nr, nc), xpd = TRUE)
  for (i in 1:n_ct) {
    mat2 <- matrix(0, nrow = n_ct, ncol = n_ct, dimnames = dimnames(mat))
    mat2[i, ] <- mat[i, ]
    netVisual_circle(mat2, vertex.weight = groupSize, weight.scale = TRUE,
                     edge.weight.max = max(mat), title.name = rownames(mat)[i])
  }
  dev.off()
  cat("      Saved: 02_per_celltype_circle\n")

  # ---- 03 Top signaling pathway 的多种可视化 ----
  top_pathways <- get_top_pathways(cellchat, n = 10)
  cat("      Top pathways:", paste(top_pathways, collapse = ", "), "\n")

  for (pw in top_pathways) {
    pw_safe <- gsub("[^A-Za-z0-9_-]", "_", pw)

    # Hierarchy plot
    tryCatch({
      vertex.receiver <- seq(1, min(floor(n_ct / 2), 10))
      pdf(file.path(run_dir, paste0(prefix, "_03a_hierarchy_", pw_safe, ".pdf")), width = 14, height = 10)
      netVisual_aggregate(cellchat, signaling = pw, layout = "hierarchy",
                          vertex.receiver = vertex.receiver)
      dev.off()
    }, error = function(e) cat("      Warning [03a-hierarchy", pw, "]:", conditionMessage(e), "\n"))

    # Circle plot per pathway
    tryCatch({
      pdf(file.path(run_dir, paste0(prefix, "_03b_circle_", pw_safe, ".pdf")), width = 8, height = 8)
      netVisual_aggregate(cellchat, signaling = pw, layout = "circle")
      dev.off()
    }, error = function(e) cat("      Warning [03b-circle", pw, "]:", conditionMessage(e), "\n"))

    # Chord diagram per pathway
    tryCatch({
      pdf(file.path(run_dir, paste0(prefix, "_03c_chord_", pw_safe, ".pdf")), width = 10, height = 10)
      netVisual_aggregate(cellchat, signaling = pw, layout = "chord")
      dev.off()
    }, error = function(e) cat("      Warning [03c-chord", pw, "]:", conditionMessage(e), "\n"))

    # Heatmap per pathway
    tryCatch({
      pdf(file.path(run_dir, paste0(prefix, "_03d_heatmap_", pw_safe, ".pdf")), width = 10, height = 8)
      print(netVisual_heatmap(cellchat, signaling = pw, color.heatmap = "Reds"))
      dev.off()
    }, error = function(e) cat("      Warning [03d-heatmap", pw, "]:", conditionMessage(e), "\n"))

    # Centrality network per pathway
    tryCatch({
      pdf(file.path(run_dir, paste0(prefix, "_03e_centrality_", pw_safe, ".pdf")), width = 10, height = 4)
      netAnalysis_signalingRole_network(cellchat, signaling = pw, width = 10, height = 3, font.size = 10)
      dev.off()
    }, error = function(e) cat("      Warning [03e-centrality", pw, "]:", conditionMessage(e), "\n"))

    # Gene expression of signaling genes
    tryCatch({
      pdf(file.path(run_dir, paste0(prefix, "_03f_geneExpr_", pw_safe, ".pdf")), width = 12, height = 6)
      print(plotGeneExpression(cellchat, signaling = pw, enriched.only = TRUE))
      dev.off()
    }, error = function(e) cat("      Warning [03f-geneExpr", pw, "]:", conditionMessage(e), "\n"))
  }
  cat("      Saved: 03 per-pathway visualizations for", length(top_pathways), "pathways\n")

  # ---- 04 L-R pair bubble plot (all pathways) ----
  tryCatch({
    pdf(file.path(run_dir, paste0(prefix, "_04_bubble_all.pdf")), width = 16, height = 12)
    print(netVisual_bubble(cellchat, remove.isolate = FALSE))
    dev.off()
    cat("      Saved: 04_bubble_all\n")
  }, error = function(e) cat("      Warning [04]:", conditionMessage(e), "\n"))

  # ---- 05 signaling role scatter ----
  tryCatch({
    pdf(file.path(run_dir, paste0(prefix, "_05_signaling_role_scatter.pdf")), width = 10, height = 8)
    print(netAnalysis_signalingRole_scatter(cellchat))
    dev.off()
    cat("      Saved: 05_signaling_role_scatter\n")
  }, error = function(e) cat("      Warning [05]:", conditionMessage(e), "\n"))

  # ---- 06 outgoing/incoming heatmap (global) ----
  tryCatch({
    pdf(file.path(run_dir, paste0(prefix, "_06_signaling_role_heatmap.pdf")), width = 14, height = 8)
    ht1 <- netAnalysis_signalingRole_heatmap(cellchat, pattern = "outgoing")
    ht2 <- netAnalysis_signalingRole_heatmap(cellchat, pattern = "incoming")
    print(ht1 + ht2)
    dev.off()
    cat("      Saved: 06_signaling_role_heatmap\n")
  }, error = function(e) cat("      Warning [06]:", conditionMessage(e), "\n"))

  # ---- 07-09 communication patterns（NMF，串行执行）----
  set_future_sequential()
  for (pattern_type in c("outgoing", "incoming")) {
    tryCatch({
      pdf(file.path(run_dir, paste0(prefix, "_07_selectK_", pattern_type, ".pdf")), width = 10, height = 6)
      selectK(cellchat, pattern = pattern_type)
      dev.off()

      nPatterns <- min(5, length(unique(cellchat@idents)) - 1)
      cellchat  <- identifyCommunicationPatterns(cellchat, pattern = pattern_type, k = nPatterns)

      pdf(file.path(run_dir, paste0(prefix, "_08_river_", pattern_type, ".pdf")), width = 14, height = 8)
      netAnalysis_river(cellchat, pattern = pattern_type)
      dev.off()

      pdf(file.path(run_dir, paste0(prefix, "_09_dot_", pattern_type, ".pdf")), width = 14, height = 8)
      netAnalysis_dot(cellchat, pattern = pattern_type)
      dev.off()

      cat("      Saved: 07-09 patterns for", pattern_type, "\n")
    }, error = function(e) cat("      Warning [07-09]", pattern_type, ":", conditionMessage(e), "\n"))
  }

  # ---- 10-11 embedding (functional + structural) ----
  tryCatch({
    cellchat <- computeNetSimilarity(cellchat, type = "functional")
    cellchat <- netEmbedding(cellchat, type = "functional", umap.method = "uwot")
    cellchat <- netClustering(cellchat, type = "functional")
    pdf(file.path(run_dir, paste0(prefix, "_10_functional_embedding.pdf")), width = 10, height = 8)
    netVisual_embedding(cellchat, type = "functional", label.size = 3.5)
    dev.off()
    pdf(file.path(run_dir, paste0(prefix, "_10b_functional_embeddingZoom.pdf")), width = 10, height = 8)
    netVisual_embeddingZoomIn(cellchat, type = "functional", nCol = 2)
    dev.off()
    cat("      Saved: 10_functional_embedding\n")
  }, error = function(e) cat("      Warning [10]:", conditionMessage(e), "\n"))

  tryCatch({
    cellchat <- computeNetSimilarity(cellchat, type = "structural")
    cellchat <- netEmbedding(cellchat, type = "structural", umap.method = "uwot")
    cellchat <- netClustering(cellchat, type = "structural")
    pdf(file.path(run_dir, paste0(prefix, "_11_structural_embedding.pdf")), width = 10, height = 8)
    netVisual_embedding(cellchat, type = "structural", label.size = 3.5)
    dev.off()
    pdf(file.path(run_dir, paste0(prefix, "_11b_structural_embeddingZoom.pdf")), width = 10, height = 8)
    netVisual_embeddingZoomIn(cellchat, type = "structural", nCol = 2)
    dev.off()
    cat("      Saved: 11_structural_embedding\n")
  }, error = function(e) cat("      Warning [11]:", conditionMessage(e), "\n"))

  # ---- 12 导出通讯表格 ----
  tryCatch({
    df_net <- subsetCommunication(cellchat)
    write.csv(df_net, file.path(run_dir, paste0(prefix, "_LR_communication.csv")), row.names = FALSE)
    df_pw  <- subsetCommunication(cellchat, slot.name = "netP")
    write.csv(df_pw, file.path(run_dir, paste0(prefix, "_pathway_communication.csv")), row.names = FALSE)
    cat("      Saved: 12 CSV tables\n")
  }, error = function(e) cat("      Warning [12-CSV]:", conditionMessage(e), "\n"))

  return(cellchat)
}

# ============================================================
# 比较分析（完整版，对照 Comparison_analysis tutorial）
# ============================================================
run_comparison <- function(cc_healthy, cc_disease, run_name, disease_name, out_dir) {

  run_dir <- file.path(out_dir, run_name)

  cells_h   <- levels(cc_healthy@idents)
  cells_d   <- levels(cc_disease@idents)
  all_cells <- union(cells_h, cells_d)

  cat("    Healthy:", length(cells_h), "| Disease:", length(cells_d),
      "| Union:", length(all_cells), "cell types\n")

  # ---------- Lift cell types if needed ----------
  if (!setequal(cells_h, cells_d)) {
    cat("    Lifting cell types to union set...\n")
    cc_healthy <- liftCellChat(cc_healthy, all_cells)
    cc_disease <- liftCellChat(cc_disease, all_cells)
  }

  object.list <- list(Healthy = cc_healthy, Disease = cc_disease)
  cellchat    <- mergeCellChat(object.list, add.names = c("Healthy", disease_name))

  # ======== Part I: 总体比较 ========

  # comp_01: interaction count + strength barplot
  tryCatch({
    gg1 <- compareInteractions(cellchat, show.legend = FALSE, group = c(1, 2))
    gg2 <- compareInteractions(cellchat, show.legend = FALSE, group = c(1, 2), measure = "weight")
    pdf(file.path(run_dir, "comp_01_interaction_comparison.pdf"), width = 12, height = 6)
    print(gg1 + gg2)
    dev.off(); cat("    Saved: comp_01\n")
  }, error = function(e) cat("    Warning [comp_01]:", conditionMessage(e), "\n"))

  # comp_02: differential network (circle)
  tryCatch({
    pdf(file.path(run_dir, "comp_02_diff_network.pdf"), width = 14, height = 6)
    par(mfrow = c(1, 2))
    netVisual_diffInteraction(cellchat, weight.scale = TRUE)
    netVisual_diffInteraction(cellchat, weight.scale = TRUE, measure = "weight")
    dev.off(); cat("    Saved: comp_02\n")
  }, error = function(e) cat("    Warning [comp_02]:", conditionMessage(e), "\n"))

  # comp_03/04: differential heatmap (count + weight)
  tryCatch({
    pdf(file.path(run_dir, "comp_03_diff_heatmap_count.pdf"),  width = 12, height = 10)
    print(netVisual_heatmap(cellchat)); dev.off()
    pdf(file.path(run_dir, "comp_04_diff_heatmap_weight.pdf"), width = 12, height = 10)
    print(netVisual_heatmap(cellchat, measure = "weight")); dev.off()
    cat("    Saved: comp_03/04\n")
  }, error = function(e) cat("    Warning [comp_03/04]:", conditionMessage(e), "\n"))

  # comp_05: side-by-side circle plots (count + weight per condition)
  tryCatch({
    weight.max <- getMaxWeight(object.list, slot.name = c("net"), attribute = c("count"))
    pdf(file.path(run_dir, "comp_05_circle_sidebyside_count.pdf"), width = 14, height = 6)
    par(mfrow = c(1, 2), xpd = TRUE)
    for (i in 1:length(object.list)) {
      netVisual_circle(object.list[[i]]@net$count,
                       weight.scale = TRUE, edge.weight.max = weight.max[1],
                       label.edge = FALSE,
                       title.name = paste0("Number of interactions - ", names(object.list)[i]))
    }
    dev.off()

    weight.max <- getMaxWeight(object.list, slot.name = c("net"), attribute = c("weight"))
    pdf(file.path(run_dir, "comp_05b_circle_sidebyside_weight.pdf"), width = 14, height = 6)
    par(mfrow = c(1, 2), xpd = TRUE)
    for (i in 1:length(object.list)) {
      netVisual_circle(object.list[[i]]@net$weight,
                       weight.scale = TRUE, edge.weight.max = weight.max[1],
                       label.edge = FALSE,
                       title.name = paste0("Interaction strength - ", names(object.list)[i]))
    }
    dev.off()
    cat("    Saved: comp_05\n")
  }, error = function(e) cat("    Warning [comp_05]:", conditionMessage(e), "\n"))

  # ======== Part II: 信号通路水平比较 ========

  # comp_06: information flow (stacked + unstacked)
  tryCatch({
    pdf(file.path(run_dir, "comp_06a_infoflow_stacked.pdf"),   width = 10, height = 10)
    print(rankNet(cellchat, mode = "comparison", stacked = TRUE,  do.stat = TRUE)); dev.off()
    pdf(file.path(run_dir, "comp_06b_infoflow_unstacked.pdf"), width = 10, height = 10)
    print(rankNet(cellchat, mode = "comparison", stacked = FALSE, do.stat = TRUE)); dev.off()
    cat("    Saved: comp_06\n")
  }, error = function(e) cat("    Warning [comp_06]:", conditionMessage(e), "\n"))

  # comp_07: per-pathway comparison (top pathways, circle + hierarchy)
  tryCatch({
    top_pw_h <- get_top_pathways(cc_healthy, n = 5)
    top_pw_d <- get_top_pathways(cc_disease, n = 5)
    top_pw   <- unique(c(top_pw_h, top_pw_d))

    for (pw in top_pw) {
      pw_safe <- gsub("[^A-Za-z0-9_-]", "_", pw)

      # Circle plot side by side
      tryCatch({
        weight.max <- getMaxWeight(object.list, slot.name = c("netP"), attribute = pw)
        pdf(file.path(run_dir, paste0("comp_07_circle_", pw_safe, ".pdf")), width = 14, height = 6)
        par(mfrow = c(1, 2), xpd = TRUE)
        for (i in 1:length(object.list)) {
          netVisual_aggregate(object.list[[i]], signaling = pw, layout = "circle",
                              edge.weight.max = weight.max[1], edge.width.max = 10,
                              signaling.name = paste(pw, names(object.list)[i]))
        }
        dev.off()
      }, error = function(e) NULL)

      # Chord diagram per condition
      tryCatch({
        pdf(file.path(run_dir, paste0("comp_07b_chord_", pw_safe, ".pdf")), width = 14, height = 6)
        par(mfrow = c(1, 2), xpd = TRUE)
        for (i in 1:length(object.list)) {
          netVisual_aggregate(object.list[[i]], signaling = pw, layout = "chord",
                              signaling.name = paste(pw, names(object.list)[i]))
        }
        dev.off()
      }, error = function(e) NULL)
    }
    cat("    Saved: comp_07 per-pathway comparison for", length(top_pw), "pathways\n")
  }, error = function(e) cat("    Warning [comp_07]:", conditionMessage(e), "\n"))

  # ======== Part III: Cell population 水平比较 ========

  # comp_08: bubble plots (up in disease / up in healthy)
  tryCatch({
    n_joint <- nlevels(cellchat@idents$joint)
    src_use <- seq_len(n_joint)
    tgt_use <- seq_len(n_joint)

    pdf(file.path(run_dir, "comp_08a_bubble_up_disease.pdf"), width = 14, height = 12)
    print(netVisual_bubble(cellchat, sources.use = src_use, targets.use = tgt_use,
                           comparison = c(1, 2), max.dataset = 2,
                           title.name = paste0("Up in ", disease_name),
                           angle.x = 45, remove.isolate = TRUE)); dev.off()

    pdf(file.path(run_dir, "comp_08b_bubble_up_healthy.pdf"), width = 14, height = 12)
    print(netVisual_bubble(cellchat, sources.use = src_use, targets.use = tgt_use,
                           comparison = c(1, 2), max.dataset = 1,
                           title.name = "Up in Healthy",
                           angle.x = 45, remove.isolate = TRUE)); dev.off()
    cat("    Saved: comp_08\n")
  }, error = function(e) cat("    Warning [comp_08]:", conditionMessage(e), "\n"))

  # comp_09: L-R chord diagram comparison for top pathways
  tryCatch({
    for (pw in top_pw[1:min(5, length(top_pw))]) {
      pw_safe <- gsub("[^A-Za-z0-9_-]", "_", pw)
      tryCatch({
        pdf(file.path(run_dir, paste0("comp_09_LR_chord_", pw_safe, ".pdf")), width = 14, height = 6)
        par(mfrow = c(1, 2), xpd = TRUE)
        for (i in 1:length(object.list)) {
          netVisual_chord_gene(object.list[[i]], signaling = pw,
                               title.name = paste0(pw, " - ", names(object.list)[i]),
                               legend.pos.x = 10)
        }
        dev.off()
      }, error = function(e) NULL)
    }
    cat("    Saved: comp_09\n")
  }, error = function(e) cat("    Warning [comp_09]:", conditionMessage(e), "\n"))

  # comp_10: signaling role scatter (comparison)
  tryCatch({
    pdf(file.path(run_dir, "comp_10_signaling_role_scatter.pdf"), width = 14, height = 6)
    print(netAnalysis_signalingRole_scatter(cellchat, comparison = c(1, 2))); dev.off()
    cat("    Saved: comp_10\n")
  }, error = function(e) cat("    Warning [comp_10]:", conditionMessage(e), "\n"))

  # comp_11: outgoing/incoming signaling changes per cell type
  tryCatch({
    pdf(file.path(run_dir, "comp_11a_signaling_heatmap_outgoing.pdf"), width = 14, height = 8)
    ht1 <- netAnalysis_signalingRole_heatmap(object.list[[1]], pattern = "outgoing",
                                             signaling = NULL, title = "Healthy - Outgoing", width = 5, height = 10)
    ht2 <- netAnalysis_signalingRole_heatmap(object.list[[2]], pattern = "outgoing",
                                             signaling = NULL, title = paste0(disease_name, " - Outgoing"), width = 5, height = 10)
    print(ht1 + ht2)
    dev.off()

    pdf(file.path(run_dir, "comp_11b_signaling_heatmap_incoming.pdf"), width = 14, height = 8)
    ht1 <- netAnalysis_signalingRole_heatmap(object.list[[1]], pattern = "incoming",
                                             signaling = NULL, title = "Healthy - Incoming", width = 5, height = 10)
    ht2 <- netAnalysis_signalingRole_heatmap(object.list[[2]], pattern = "incoming",
                                             signaling = NULL, title = paste0(disease_name, " - Incoming"), width = 5, height = 10)
    print(ht1 + ht2)
    dev.off()
    cat("    Saved: comp_11\n")
  }, error = function(e) cat("    Warning [comp_11]:", conditionMessage(e), "\n"))

  # ======== Part IV: Manifold learning 比较 ========

  # comp_12: functional similarity
  tryCatch({
    cellchat <- computeNetSimilarityPairwise(cellchat, type = "functional")
    cellchat <- netEmbedding(cellchat, type = "functional", umap.method = "uwot")
    cellchat <- netClustering(cellchat, type = "functional")
    pdf(file.path(run_dir, "comp_12a_functional_similarity.pdf"), width = 10, height = 8)
    print(netVisual_embeddingPairwise(cellchat, type = "functional", label.size = 3)); dev.off()
    pdf(file.path(run_dir, "comp_12b_functional_similarity_zoom.pdf"), width = 12, height = 10)
    print(netVisual_embeddingPairwiseZoomIn(cellchat, type = "functional", nCol = 2)); dev.off()
    cat("    Saved: comp_12\n")
  }, error = function(e) cat("    Warning [comp_12]:", conditionMessage(e), "\n"))

  # comp_13: structural similarity
  tryCatch({
    cellchat <- computeNetSimilarityPairwise(cellchat, type = "structural")
    cellchat <- netEmbedding(cellchat, type = "structural", umap.method = "uwot")
    cellchat <- netClustering(cellchat, type = "structural")
    pdf(file.path(run_dir, "comp_13a_structural_similarity.pdf"), width = 10, height = 8)
    print(netVisual_embeddingPairwise(cellchat, type = "structural", label.size = 3)); dev.off()
    pdf(file.path(run_dir, "comp_13b_structural_similarity_zoom.pdf"), width = 12, height = 10)
    print(netVisual_embeddingPairwiseZoomIn(cellchat, type = "structural", nCol = 2)); dev.off()
    cat("    Saved: comp_13\n")
  }, error = function(e) cat("    Warning [comp_13]:", conditionMessage(e), "\n"))

  # ======== Part V: 识别上调/下调通路 ========

  # comp_14: signaling changes scatter per cell type
  tryCatch({
    n_ct <- nlevels(cellchat@idents$joint)
    ct_names <- levels(cellchat@idents$joint)
    # 为每个 cell type 画 signaling changes
    pdf(file.path(run_dir, "comp_14_signaling_changes_scatter.pdf"), width = 8, height = 6 * n_ct)
    par(mfrow = c(n_ct, 1))
    for (ct in ct_names) {
      tryCatch({
        print(netAnalysis_signalingChanges_scatter(cellchat, idents.use = ct,
                                                   comparison = c(1, 2)))
      }, error = function(e) NULL)
    }
    dev.off()
    cat("    Saved: comp_14\n")
  }, error = function(e) cat("    Warning [comp_14]:", conditionMessage(e), "\n"))

  # ======== Part VI: 导出数据 ========

  tryCatch({
    df1 <- subsetCommunication(cc_healthy, slot.name = "netP"); df1$condition <- "Healthy"
    df2 <- subsetCommunication(cc_disease, slot.name = "netP"); df2$condition <- disease_name
    write.csv(rbind(df1, df2), file.path(run_dir, "comparison_pathway_communication.csv"), row.names = FALSE)

    lr1 <- subsetCommunication(cc_healthy); lr1$condition <- "Healthy"
    lr2 <- subsetCommunication(cc_disease); lr2$condition <- disease_name
    write.csv(rbind(lr1, lr2), file.path(run_dir, "comparison_LR_communication.csv"), row.names = FALSE)

    # 导出 centrality scores
    tryCatch({
      cent1 <- cellchat@netP$Healthy$centr
      cent2 <- cellchat@netP[[disease_name]]$centr
      # 保存为 RDS 方便后续使用
      saveRDS(list(healthy = cent1, disease = cent2),
              file.path(run_dir, "centrality_scores.rds"))
    }, error = function(e) NULL)

    cat("    Saved: CSV tables + centrality RDS\n")
  }, error = function(e) cat("    Warning [CSV]:", conditionMessage(e), "\n"))

  # 保存 merged object
  saveRDS(cellchat, file.path(run_dir, paste0(run_name, "_merged_cellchat.rds")))
  cat("    Saved: merged RDS\n")

  return(cellchat)
}

# ============================================================
# 主流程
# ============================================================
cat("\n", strrep("=", 60), "\n")
cat("CellChat Skin Analysis Pipeline\n")
cat("Memory limit:", 250, "GB | Workers:", n_workers, "\n")
cat(strrep("=", 60), "\n\n")

for (task in run_queue) {
  species  <- task$species
  ct_col   <- task$ct_col
  info     <- datasets[[species]]
  run_name <- paste0(species, "_", ct_col)
  run_dir  <- file.path(output_dir, run_name)
  dir.create(run_dir, recursive = TRUE, showWarnings = FALSE)

  cat("\n", strrep("=", 60), "\n")
  cat("Processing:", run_name, "\n")
  cat(strrep("=", 60), "\n")

  h5ad_path <- h5ad_paths[[species]]

  # --- Healthy ---
  rds_h <- file.path(run_dir, paste0(run_name, "_healthy_cellchat.rds"))
  if (file.exists(rds_h)) {
    cat("  [SKIP] Healthy RDS exists, loading...\n")
    cc_healthy <- readRDS(rds_h)
  } else {
    cat("  Building Healthy CellChat...\n")
    cc_healthy <- create_cellchat(h5ad_path, info$healthy_cond, ct_col, info$db,
                                  info$filter_skin, info$skin_value)
    cc_healthy <- plot_single_cellchat(cc_healthy, run_name, "Healthy", run_dir)
    saveRDS(cc_healthy, rds_h)
    cat("  Saved: healthy RDS\n")
  }
  gc()

  # --- Disease ---
  rds_d <- file.path(run_dir, paste0(run_name, "_disease_cellchat.rds"))
  if (file.exists(rds_d)) {
    cat("  [SKIP] Disease RDS exists, loading...\n")
    cc_disease <- readRDS(rds_d)
  } else {
    cat("  Building", info$disease_name, "CellChat...\n")
    cc_disease <- create_cellchat(h5ad_path, info$disease_cond, ct_col, info$db,
                                  info$filter_skin, info$skin_value)
    cc_disease <- plot_single_cellchat(cc_disease, run_name, info$disease_name, run_dir)
    saveRDS(cc_disease, rds_d)
    cat("  Saved: disease RDS\n")
  }
  gc()

  # --- 比较 ---
  rds_m <- file.path(run_dir, paste0(run_name, "_merged_cellchat.rds"))
  if (file.exists(rds_m)) {
    cat("  [SKIP] Merged RDS exists, skipping comparison\n")
  } else {
    cat("  Running comparison...\n")
    merged <- run_comparison(cc_healthy, cc_disease, run_name, info$disease_name, output_dir)
    rm(merged)
  }

  # --- 释放当前 task 内存 ---
  cat("  Releasing memory for", run_name, "...\n")
  rm(cc_healthy, cc_disease)
  gc()
  cat("  Done:", run_name, "\n")
}

cat("\n", strrep("=", 60), "\n")
cat("All CellChat analyses complete.\n")
cat("Results saved to:", output_dir, "\n")
cat(strrep("=", 60), "\n")
