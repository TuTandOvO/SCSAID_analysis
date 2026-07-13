import pertpy as pt
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# ============================================================
# 配置
# ============================================================
output_dir = "/gpfsdata/home/renyixiang/SkinDB/Augur_results"
os.makedirs(output_dir, exist_ok=True)

datasets = {
    "human": {
        "path": "/gpfsdata/home/renyixiang/SkinDB/10X/human/Annotation/Annotated/human_with_all_leiden_10threshold_annotated_updated.h5ad",
        "conditions": ["Healthy", "psoriasis"],
    },
    "mouse": {
        "path": "/gpfsdata/home/renyixiang/SkinDB/10X/mouse/Annotation/Annotated/mouse_with_all_leiden_10threshold_annotated_updated.h5ad",
        "conditions": ["Healthy", "IMQ-induced psoriasis"],
    },
}

celltype_cols = ["Gross_Map", "Fine_Map"]
min_cells_per_condition = 50

# ============================================================
# 运行 Augur
# ============================================================
all_results = {}

for species, info in datasets.items():
    print(f"\n{'='*60}")
    print(f"Loading {species} data...")
    adata = sc.read_h5ad(info["path"])

    print(f"  Original cells: {adata.n_obs}")
    print(f"  Available conditions: {adata.obs['condition'].unique().tolist()}")

    adata_sub = adata[adata.obs["condition"].isin(info["conditions"])].copy()
    print(f"  After filtering ({' vs '.join(info['conditions'])}): {adata_sub.n_obs}")

    for ct_col in celltype_cols:
        run_name = f"{species}_{ct_col}"
        print(f"\n  Running Augur: {run_name}")

        # 过滤掉每个 condition 中细胞数不足的 cell type
        ct_counts = adata_sub.obs.groupby([ct_col, "condition"]).size().unstack(fill_value=0)
        valid_cts = ct_counts[ct_counts.min(axis=1) >= min_cells_per_condition].index.tolist()
        removed = adata_sub.obs[ct_col].nunique() - len(valid_cts)
        adata_filtered = adata_sub[adata_sub.obs[ct_col].isin(valid_cts)].copy()

        print(f"    Cell types ({ct_col}): {len(valid_cts)} (removed {removed} with <{min_cells_per_condition} cells/condition)")
        print(f"    Remaining cells: {adata_filtered.n_obs}")

        # 打印每个 cell type 的细胞数分布
        print(f"    Cell counts per condition:")
        for ct in valid_cts:
            counts = ct_counts.loc[ct]
            print(f"      {ct}: {dict(counts)}")

        aug = pt.tl.Augur("random_forest_classifier")

        loaded_data = aug.load(
            adata_filtered,
            label_col="condition",
            cell_type_col=ct_col,
        )

        adata_aug, results = aug.predict(
            loaded_data,
            n_threads=8,
            select_variance_features=False,
            n_subsamples=50,
        )

        # summary_metrics: 行是指标，列是 cell type，需要转置后保存
        summary_scores = results["summary_metrics"].loc["mean_augur_score"].sort_values(ascending=False)
        summary_df = results["summary_metrics"].T.sort_values("mean_augur_score", ascending=False)
        summary_df.to_csv(f"{output_dir}/{run_name}_augur_auc.csv")
        all_results[run_name] = {"aug": aug, "results": results, "summary": summary_scores}

        print(f"\n    Results ({run_name}) - Augur scores ranked:")
        print(summary_scores.to_string())

        # Lollipop plot
        try:
            fig, ax = plt.subplots(figsize=(6, max(4, len(valid_cts) * 0.3)))
            aug.plot_lollipop(results, ax=ax)
            plt.title(f"{species} - {ct_col}\n(Healthy vs Psoriasis)")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{run_name}_lollipop.pdf", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"    Saved: {run_name}_lollipop.pdf")
        except Exception as e:
            print(f"    Warning: lollipop plot failed for {run_name}: {e}")

        # UMAP with augur scores
        try:
            if "X_umap" in adata_aug.obsm:
                fig, ax = plt.subplots(figsize=(8, 6))
                sc.pl.umap(adata_aug, color="augur_score", ax=ax, show=False)
                plt.title(f"{species} - {ct_col} Augur Score")
                plt.tight_layout()
                plt.savefig(f"{output_dir}/{run_name}_umap_augur.pdf", dpi=150, bbox_inches="tight")
                plt.close()
                print(f"    Saved: {run_name}_umap_augur.pdf")
        except Exception as e:
            print(f"    Warning: UMAP plot failed for {run_name}: {e}")

# ============================================================
# 汇总所有结果
# ============================================================
print(f"\n{'='*60}")
print("Summary of all runs:")
print(f"{'='*60}")

for run_name, data in all_results.items():
    print(f"\n{run_name}:")
    print(data["summary"].to_string())

print(f"\nAll results saved to: {output_dir}")
