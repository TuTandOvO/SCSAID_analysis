#!/usr/bin/env python3
"""
Pseudobulk DEG Analysis
========================
Pairwise comparisons:
  - Healthy vs.  psoriasis
  - Healthy vs. HS lesion
For Gross_Map: Keratinocyte, Immune, Fibroblast
"""

import os
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import decoupler as dc
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# CONFIG
# ============================================================
ADATA_PATH = "/gpfsdata/home/renyixiang/SkinDB/10X/human/Annotation/Annotated/human_with_all_leiden_10threshold_annotated_updated.h5ad"
OUTPUT_DIR = "/gpfsdata/home/renyixiang/SkinDB/10X/human/data4viz/DEG_pseudobulk"

CELLTYPE_COL = "Gross_Map"
SAMPLE_COL = "batch"
CONDITION_COL = "condition"

# 要分析的细胞类型
CELLTYPES = ["Keratinocyte", "Immune", "Fibroblast"]

# 要做的 pairwise 比较 (reference, comparison)
COMPARISONS = [
    ("Healthy", "HS lesion"),
    ("Healthy", "psoriasis"),
]

# DEG 阈值（用于标注，不影响完整结果输出）
LOGFC_THR = 1.0
PVAL_THR = 0.05

os.makedirs(OUTPUT_DIR, exist_ok=True)
sc.settings.set_figure_params(dpi=150, frameon=False)

print("=" * 70)
print("Pseudobulk DEG Analysis")
print("=" * 70)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("\n[Step 1] Loading AnnData...")
adata = sc.read_h5ad(ADATA_PATH)
print(f"  Shape: {adata.shape}")

# 检查是否有 raw counts layer
has_counts = 'counts' in adata.layers
print(f"  Has 'counts' layer: {has_counts}")

# 如果没有 counts layer，尝试反推 raw counts（如果 X 是 log1p normalized）
if not has_counts:
    print("  ⚠️  No raw counts layer found.")
    print("  Will use log-normalized data with Wilcoxon test on pseudobulk means.")

# 打印分组信息
for ct in CELLTYPES:
    ct_mask = adata.obs[CELLTYPE_COL] == ct
    for ref, comp in COMPARISONS:
        n_ref = ((adata.obs[CONDITION_COL] == ref) & ct_mask).sum()
        n_comp = ((adata.obs[CONDITION_COL] == comp) & ct_mask).sum()
        ref_batches = adata.obs.loc[(adata.obs[CONDITION_COL] == ref) & ct_mask, SAMPLE_COL].nunique()
        comp_batches = adata.obs.loc[(adata.obs[CONDITION_COL] == comp) & ct_mask, SAMPLE_COL].nunique()
        print(f"  {ct:15s} | {ref} ({n_ref} cells, {ref_batches} batches) vs "
              f"{comp} ({n_comp} cells, {comp_batches} batches)")

# ============================================================
# 2. PSEUDOBULK + DEG PER CELL TYPE & COMPARISON
# ============================================================
print("\n[Step 2] Running pseudobulk DEG analysis...")

# 尝试导入 pydeseq2
try:
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    USE_DESEQ2 = has_counts  # 只有有 raw counts 才用 DESeq2
    if USE_DESEQ2:
        print("  Using PyDESeq2 (raw counts available)")
    else:
        print("  PyDESeq2 available but no raw counts → using Wilcoxon on pseudobulk")
except ImportError:
    USE_DESEQ2 = False
    print("  PyDESeq2 not installed → using Wilcoxon on pseudobulk")

all_results = []

for ct in CELLTYPES:
    for ref, comp in COMPARISONS:
        tag = f"{ct}__{comp.replace(' ', '_')}_vs_{ref}"
        print(f"\n  --- {ct}: {comp} vs {ref} ---")

        # 筛选相关细胞
        mask = (
            (adata.obs[CELLTYPE_COL] == ct) &
            (adata.obs[CONDITION_COL].isin([ref, comp]))
        )
        adata_sub = adata[mask].copy()
        print(f"    Cells: {adata_sub.shape[0]}")

        if adata_sub.shape[0] < 50:
            print(f"    ⚠️ Too few cells, skipping")
            continue

        # --- 方案 A: DESeq2 (raw counts) ---
        if USE_DESEQ2:
            try:
                # Pseudobulk: sum raw counts per batch
                pdata = dc.pp.pseudobulk(
                    adata_sub,
                    sample_col=SAMPLE_COL,
                    groups_col=None,  # 已经筛选了单个 cell type
                    layer='counts',
                    mode='sum',
                )
                dc.pp.filter_samples(pdata, min_cells=10, min_counts=1000)
                print(f"    Pseudobulk samples: {pdata.shape[0]}")

                if pdata.obs[CONDITION_COL].nunique() < 2:
                    print(f"    ⚠️ Only one condition left after filtering, skipping")
                    continue

                n_ref_samples = (pdata.obs[CONDITION_COL] == ref).sum()
                n_comp_samples = (pdata.obs[CONDITION_COL] == comp).sum()
                print(f"    {ref}: {n_ref_samples} samples, {comp}: {n_comp_samples} samples")

                if n_ref_samples < 2 or n_comp_samples < 2:
                    print(f"    ⚠️ Need ≥2 samples per group for DESeq2, skipping")
                    continue

                # Filter low-expression genes
                sc.pp.filter_genes(pdata, min_cells=1)

                # Run DESeq2
                dds = DeseqDataSet(
                    adata=pdata,
                    design_factors=CONDITION_COL,
                    ref_level=[CONDITION_COL, ref],
                    refit_cooks=True,
                )
                dds.deseq2()

                stat_res = DeseqStats(dds, contrast=[CONDITION_COL, comp, ref])
                stat_res.summary()

                res_df = stat_res.results_df.copy()
                res_df['gene'] = res_df.index
                res_df['cell_type'] = ct
                res_df['comparison'] = f"{comp} vs {ref}"
                res_df['method'] = 'DESeq2'

                # 标准化列名
                res_df = res_df.rename(columns={
                    'log2FoldChange': 'logfoldchanges',
                    'pvalue': 'pvals',
                    'padj': 'pvals_adj',
                    'baseMean': 'base_mean',
                })

                all_results.append(res_df)
                n_sig = ((res_df['pvals_adj'] < PVAL_THR) & (res_df['logfoldchanges'].abs() > LOGFC_THR)).sum()
                print(f"    ✅ DESeq2 done: {res_df.shape[0]} genes, {n_sig} significant (|logFC|>{LOGFC_THR}, padj<{PVAL_THR})")

            except Exception as e:
                print(f"    ⚠️ DESeq2 failed: {e}, falling back to Wilcoxon")
                USE_DESEQ2_THIS = False
            else:
                continue  # 成功了就跳到下一个

        # --- 方案 B: Wilcoxon on pseudobulk means ---
        try:
            pdata = dc.pp.pseudobulk(
                adata_sub,
                sample_col=SAMPLE_COL,
                groups_col=None,
                mode='mean',
                skip_checks=True,  # 因为是 log-normalized data
            )
            dc.pp.filter_samples(pdata, min_cells=10, min_counts=0)
            print(f"    Pseudobulk samples: {pdata.shape[0]}")

            if pdata.obs[CONDITION_COL].nunique() < 2:
                print(f"    ⚠️ Only one condition left after filtering, skipping")
                continue

            n_ref_samples = (pdata.obs[CONDITION_COL] == ref).sum()
            n_comp_samples = (pdata.obs[CONDITION_COL] == comp).sum()
            print(f"    {ref}: {n_ref_samples} samples, {comp}: {n_comp_samples} samples")

            if n_ref_samples < 2 or n_comp_samples < 2:
                print(f"    ⚠️ Need ≥2 samples per group, falling back to single-cell Wilcoxon")
                # 直接用单细胞做（不理想但可用）
                sc.tl.rank_genes_groups(
                    adata_sub, groupby=CONDITION_COL,
                    reference=ref, method='wilcoxon',
                    groups=[comp]
                )
                res_df = sc.get.rank_genes_groups_df(adata_sub, group=comp)
                res_df['cell_type'] = ct
                res_df['comparison'] = f"{comp} vs {ref}"
                res_df['method'] = 'wilcoxon_single_cell'
                res_df['gene'] = res_df['names']
                all_results.append(res_df)
                n_sig = ((res_df['pvals_adj'] < PVAL_THR) & (res_df['logfoldchanges'].abs() > LOGFC_THR)).sum()
                print(f"    ✅ Single-cell Wilcoxon: {res_df.shape[0]} genes, {n_sig} significant")
                continue

            # Wilcoxon on pseudobulk
            sc.tl.rank_genes_groups(
                pdata, groupby=CONDITION_COL,
                reference=ref, method='wilcoxon',
                groups=[comp]
            )
            res_df = sc.get.rank_genes_groups_df(pdata, group=comp)
            res_df['cell_type'] = ct
            res_df['comparison'] = f"{comp} vs {ref}"
            res_df['method'] = 'wilcoxon_pseudobulk'
            res_df['gene'] = res_df['names']

            all_results.append(res_df)
            n_sig = ((res_df['pvals_adj'] < PVAL_THR) & (res_df['logfoldchanges'].abs() > LOGFC_THR)).sum()
            print(f"    ✅ Pseudobulk Wilcoxon: {res_df.shape[0]} genes, {n_sig} significant")

        except Exception as e:
            print(f"    ⚠️ Failed: {e}")
            import traceback
            traceback.print_exc()

# ============================================================
# 3. COMBINE & SAVE RESULTS
# ============================================================
print("\n[Step 3] Saving results...")

if all_results:
    deg_all = pd.concat(all_results, ignore_index=True)

    # 保存完整结果
    deg_all.to_csv(os.path.join(OUTPUT_DIR, "DEG_all_results.csv.gz"), compression='gzip', index=False)
    print(f"  ✅ DEG_all_results.csv.gz — {deg_all.shape}")

    # 按 cell type × comparison 分别保存
    for (ct, comp_label), sub_df in deg_all.groupby(['cell_type', 'comparison']):
        fname = f"DEG_{ct}__{comp_label.replace(' ', '_').replace('.', '')}.csv"
        sub_df.to_csv(os.path.join(OUTPUT_DIR, fname), index=False)
        print(f"  ✅ {fname} — {sub_df.shape[0]} genes")

    # ============================================================
    # 4. VOLCANO PLOTS
    # ============================================================
    print("\n[Step 4] Generating volcano plots...")

    for (ct, comp_label), sub_df in deg_all.groupby(['cell_type', 'comparison']):
        fig, ax = plt.subplots(figsize=(8, 6))

        df_plot = sub_df.copy()
        df_plot['neg_log10_pval'] = -np.log10(df_plot['pvals_adj'].clip(lower=1e-300))

        # 分类标记
        df_plot['category'] = 'NS'
        df_plot.loc[
            (df_plot['logfoldchanges'] > LOGFC_THR) & (df_plot['pvals_adj'] < PVAL_THR),
            'category'
        ] = 'Up'
        df_plot.loc[
            (df_plot['logfoldchanges'] < -LOGFC_THR) & (df_plot['pvals_adj'] < PVAL_THR),
            'category'
        ] = 'Down'

        colors = {'NS': '#BBBBBB', 'Up': '#E74C3C', 'Down': '#3498DB'}
        for cat in ['NS', 'Down', 'Up']:
            mask = df_plot['category'] == cat
            ax.scatter(
                df_plot.loc[mask, 'logfoldchanges'],
                df_plot.loc[mask, 'neg_log10_pval'],
                c=colors[cat], s=3, alpha=0.5, label=f"{cat} ({mask.sum()})"
            )

        # 标注 top genes
        top_genes = df_plot[df_plot['category'] != 'NS'].nlargest(15, 'neg_log10_pval')
        for _, row in top_genes.iterrows():
            gene_name = row.get('gene', row.get('names', ''))
            ax.annotate(
                gene_name,
                (row['logfoldchanges'], row['neg_log10_pval']),
                fontsize=6, alpha=0.8,
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.3),
                textcoords="offset points", xytext=(5, 5)
            )

        ax.axhline(-np.log10(PVAL_THR), ls='--', c='gray', lw=0.5)
        ax.axvline(LOGFC_THR, ls='--', c='gray', lw=0.5)
        ax.axvline(-LOGFC_THR, ls='--', c='gray', lw=0.5)

        n_up = (df_plot['category'] == 'Up').sum()
        n_down = (df_plot['category'] == 'Down').sum()
        ax.set_title(f"{ct}: {comp_label}\n↑Up: {n_up}  ↓Down: {n_down}", fontsize=11)
        ax.set_xlabel("log2 Fold Change")
        ax.set_ylabel("-log10(adjusted p-value)")
        ax.legend(loc='upper right', fontsize=8, markerscale=3)

        fname = f"volcano_{ct}__{comp_label.replace(' ', '_').replace('.', '')}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  ✅ {fname}")

    # ============================================================
    # 5. SUMMARY HEATMAP — Top DEGs across comparisons
    # ============================================================
    print("\n[Step 5] Summary heatmap of top DEGs...")

    for ct in CELLTYPES:
        ct_degs = deg_all[deg_all['cell_type'] == ct]
        if ct_degs.empty:
            continue

        # 取每个 comparison 的 top 15 up + top 15 down
        top_genes_set = set()
        for comp_label, sub in ct_degs.groupby('comparison'):
            sig = sub[(sub['pvals_adj'] < PVAL_THR) & (sub['logfoldchanges'].abs() > LOGFC_THR)]
            top_up = sig.nlargest(15, 'logfoldchanges')['gene'].tolist()
            top_down = sig.nsmallest(15, 'logfoldchanges')['gene'].tolist()
            top_genes_set.update(top_up + top_down)

        if len(top_genes_set) < 3:
            print(f"  {ct}: too few significant DEGs for heatmap, skipping")
            continue

        # Pivot: genes × comparisons (logFC values)
        pivot_data = {}
        for comp_label, sub in ct_degs.groupby('comparison'):
            sub_indexed = sub.set_index('gene')['logfoldchanges']
            pivot_data[comp_label] = sub_indexed

        heatmap_df = pd.DataFrame(pivot_data)
        heatmap_df = heatmap_df.loc[heatmap_df.index.isin(top_genes_set)].dropna()

        if heatmap_df.shape[0] < 3:
            continue

        # Clip for visualization
        heatmap_df = heatmap_df.clip(-5, 5)

        g = sns.clustermap(
            heatmap_df, cmap='RdBu_r', center=0, vmin=-5, vmax=5,
            figsize=(6, max(8, heatmap_df.shape[0] * 0.25)),
            xticklabels=True, yticklabels=True,
            dendrogram_ratio=0.08,
            cbar_kws={'label': 'log2FC'}
        )
        g.ax_heatmap.set_title(f"{ct} — Top DEGs across comparisons", fontsize=11)
        fname = f"heatmap_DEG_{ct}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  ✅ {fname}")

else:
    print("  ⚠️ No results generated!")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("✅ Pipeline Complete!")
print("=" * 70)
print(f"\nOutput: {OUTPUT_DIR}\n")

for f in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, f)
    size_mb = os.path.getsize(fpath) / 1024 / 1024
    print(f"  📄 {f} ({size_mb:.1f} MB)")

print(f"""
=== 输出文件说明 ===

完整结果:
  • DEG_all_results.csv.gz         — 所有比较的完整 DEG 结果

分组结果:
  • DEG_Keratinocyte__*.csv        — Keratinocyte 各比较的 DEG
  • DEG_Immune__*.csv              — Immune 各比较的 DEG
  • DEG_Fibroblast__*.csv          — Fibroblast 各比较的 DEG

可视化:
  • volcano_*.png                  — Volcano plots
  • heatmap_DEG_*.png              — Top DEGs heatmap across comparisons

关键列说明:
  • gene          — 基因名
  • logfoldchanges — log2 fold change (正值 = comparison 中上调)
  • pvals_adj     — adjusted p-value
  • cell_type     — 细胞类型
  • comparison    — 比较组
  • method        — 使用的统计方法
""")
