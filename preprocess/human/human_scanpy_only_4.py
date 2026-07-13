import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
import time
import gc
import traceback
from pathlib import Path
import scanpy as sc
import scrublet as scr
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# 配置参数
overwrite = False  # 如需强制重跑，设为 True
resolution = 0.4   # 设置较低分辨率以获得粗聚类

# 设置路径
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
log_path = project_root / "preprocessing_log.txt"
log_file = open(log_path, "a")

def log(message):
    print(message)
    log_file.write(message + "\n")
    log_file.flush()

start_time = time.time()
log("=== Step 1: Package import done. ===")

# 遍历上级目录下所有 GSE 文件夹
gse_dirs = [d for d in project_root.iterdir() if d.is_dir() and d.name.startswith("GSE")]

for gse_dir in tqdm(gse_dirs, desc="📁 Processing GSE folders"):
    run_dirs = [d for d in gse_dir.iterdir() if d.is_dir() and d.name.startswith("run_count_GSM")]

    for run_dir in run_dirs:
        gsm_id = run_dir.name.replace("run_count_", "")
        h5_path = run_dir / "outs" / "filtered_feature_bc_matrix.h5"
        gsm_dir = gse_dir / gsm_id
        gsm_dir.mkdir(exist_ok=True)

        output_h5 = gsm_dir / "pre-processed_anndata.h5ad"
        fig_path = gsm_dir / f"{gsm_id}_umap_leiden.pdf"
        deg_csv_path = gsm_dir / f"{gsm_id}_annotated_all_DEGs.csv"

        try:
            log("======================================")
            log(f"Now processing sample: {gsm_id}")
            loop_start = time.time()

            if output_h5.exists() and not overwrite:
                log(f"🔄 加载已有 AnnData: {output_h5}")
                adata = sc.read_h5ad(output_h5)
            else:
                if not h5_path.exists():
                    log(f"❌ 跳过 {gsm_id}: 文件不存在 {h5_path}")
                    continue

                log(f"--- Loading data from {h5_path} ...")
                adata = sc.read_10x_h5(h5_path)
                adata.var_names_make_unique()
                adata.obs["sample"] = gsm_id

                adata.var["mt"] = adata.var_names.str.startswith("MT-")
                adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
                adata.var["hb"] = adata.var_names.str.match(r"^HB[^(P)]")
                sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True)

                adata = adata[
                    (adata.obs["total_counts"] > adata.obs["total_counts"].quantile(0.01)) &
                    (adata.obs["total_counts"] < adata.obs["total_counts"].quantile(0.99)) &
                    (adata.obs["log1p_total_counts"] > adata.obs["log1p_total_counts"].quantile(0.01)) &
                    (adata.obs["log1p_total_counts"] < adata.obs["log1p_total_counts"].quantile(0.99)) &
                    (adata.obs["n_genes_by_counts"] > adata.obs["n_genes_by_counts"].quantile(0.01)) &
                    (adata.obs["n_genes_by_counts"] < adata.obs["n_genes_by_counts"].quantile(0.99)) &
                    (adata.obs["pct_counts_mt"] < 15) &
                    (adata.obs["pct_counts_ribo"] < 20) &
                    (adata.obs["pct_counts_hb"] < 5)
                ]

                log(f"✅ QC passed: {adata.n_obs} cells retained")

                n_cells, n_genes = adata.shape
                max_pcs = max(2, min(30, n_cells - 1, n_genes - 1))  # 不超过样本数和基因数

                scrub = scr.Scrublet(adata.X)
                scores, preds = scrub.scrub_doublets(n_prin_comps=max_pcs)

                adata.obs["doublet_scores"] = scores
                adata.obs["predicted_doublets"] = preds
                adata = adata[~adata.obs["predicted_doublets"].astype(bool), :]
                sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True)
                log(f"✅ Doublet rate: {preds.mean() * 100:.2f}%")

                adata.layers["counts"] = adata.X.copy()
                sc.pp.normalize_total(adata)
                sc.pp.log1p(adata)
                sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key="sample")
                n_comps = min(55, adata.shape[0] - 1)
                sc.tl.pca(adata, n_comps=n_comps)
                sc.pp.neighbors(adata, n_neighbors=15, n_pcs=n_comps)
                sc.tl.umap(adata)
                sc.tl.leiden(adata, resolution=resolution)

                adata.write(output_h5)
                log(f"✅ Saved AnnData to {output_h5}")

            # 生成 DEG（带 overwrite 控制）
            if not deg_csv_path.exists() or overwrite:
                sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon")
                deg_df = sc.get.rank_genes_groups_df(adata, group=None)
                deg_df.to_csv(deg_csv_path, index=False)
                log(f"✅ DEG saved to {deg_csv_path}")
            else:
                log(f"⏩ 跳过 DEG: 已存在 {deg_csv_path}")

            # 生成 UMAP 图（带 overwrite 控制）
            if not fig_path.exists() or overwrite:
                plt.figure(figsize=(15, 10))
                sc.pl.umap(adata,
                           color=["leiden"],
                           legend_loc="on data",
                           show=False,
                           size=20,
                           legend_fontsize=7)
                plt.savefig(fig_path, bbox_inches="tight")
                plt.close()
                log(f"✅ UMAP saved to {fig_path}")
            else:
                log(f"⏩ 跳过 UMAP: 已存在 {fig_path}")

            plt.close('all')
            gc.collect()
            log(f"✅ {gsm_id} done in {time.time() - loop_start:.2f} seconds")
            log("********************************")

        except Exception as e:
            error_info = f"""
            ❌ 严重错误: {gsm_id}
            === 错误类型 ===
            {type(e).__name__}
            === 错误详情 ===
            {str(e)}
            === 追踪信息 ===
            {traceback.format_exc()}
            """
            log(error_info)
            gc.collect()
            continue

log("🎉 所有样本处理完成！")
log(f"Total time used: {time.time() - start_time:.2f} seconds")
log_file.close()
