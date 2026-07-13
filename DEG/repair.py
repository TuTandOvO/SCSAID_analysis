#!/usr/bin/env python3
"""
补算 SkinDB human 中遗漏的 GSM DEG。
三个目标 GSE 都是 multi-GSM，统一走原 per-GSE 1-vs-rest 逻辑。
注意：会重算并覆盖这三个 GSE 下所有 GSM 的 DEG 文件（结果应与已有的一致）。
"""

import os
import sys
import time
import warnings
import scanpy as sc
import numpy as np
import pandas as pd
from multiprocessing import Pool

warnings.filterwarnings('ignore')

# ============================================================
# Config
# ============================================================
BASE_DIR = "/gpfsdata/home/renyixiang/SkinDB"
H5AD_PATH = os.path.join(
    BASE_DIR,
    "10X/human/Annotation/Annotated/human_with_all_leiden_10threshold_annotated_updated.h5ad",
)
SPECIES_DIR = os.path.join(BASE_DIR, "DEG", "human")
MIN_CELLS = 10
N_WORKERS = 3  # 三个 GSE 任务

TARGET_GSES = ["GSE213849", "GSE249622", "GSE249793"]

# 期望补齐的目标 GSM（用于末尾自检）
EXPECTED = {
    "GSE213849": ["GSM6595207", "GSM6595213", "GSM6595214", "GSM6595216"],
    "GSE249622": ["GSM7951929", "GSM7951933"],
    "GSE249793": ["GSM7964421"],
}


# ============================================================
# Core (复用原脚本)
# ============================================================
def _extract_deg(result, group):
    df = pd.DataFrame({
        'gene':          result['names'][group],
        'logfoldchange': result['logfoldchanges'][group],
        'pval':          result['pvals'][group],
        'pval_adj':      result['pvals_adj'][group],
        'score':         result['scores'][group],
    })
    if 'pts' in result and result['pts'] is not None:
        df['pct_group'] = result['pts'][group]
    if 'pts_rest' in result and result['pts_rest'] is not None:
        df['pct_rest'] = result['pts_rest'][group]
    return df.sort_values(['pval_adj', 'score'], ascending=[True, False])


def process_one_gse(args):
    gse_id, h5ad_path, gsm_col, gse_col, species_dir = args
    try:
        import traceback as tb
        t0 = time.time()

        adata_full = sc.read_h5ad(h5ad_path, backed='r')
        gse_values = adata_full.obs[gse_col].astype(str).values
        indices = np.where(gse_values == str(gse_id))[0]
        if len(indices) == 0:
            adata_full.file.close()
            return (gse_id, 0, [], time.time() - t0, "no_cells_found")

        adata = adata_full[indices].to_memory()
        adata_full.file.close()

        if adata.raw is not None:
            adata = adata.raw.to_adata()

        gsm_counts = adata.obs[gsm_col].value_counts()
        valid_gsms = gsm_counts[gsm_counts >= MIN_CELLS].index.tolist()
        if len(valid_gsms) < 2:
            return (gse_id, len(valid_gsms), [], time.time() - t0, "skipped_few_gsms")

        adata = adata[adata.obs[gsm_col].isin(valid_gsms)].copy()
        adata.obs[gsm_col] = adata.obs[gsm_col].astype(str)

        sc.tl.rank_genes_groups(
            adata, groupby=gsm_col, method='wilcoxon',
            n_genes=adata.shape[1], use_raw=False, pts=True,
        )

        result = adata.uns['rank_genes_groups']
        saved = []
        for gsm_id in result['names'].dtype.names:
            df = _extract_deg(result, gsm_id)
            gsm_dir = os.path.join(species_dir, str(gse_id), str(gsm_id))
            os.makedirs(gsm_dir, exist_ok=True)
            df.to_csv(os.path.join(gsm_dir, "DEGs_all.csv"), index=False)
            df[df['pval_adj'] < 0.05].to_csv(
                os.path.join(gsm_dir, "DEGs_significant.csv"), index=False)
            saved.append(gsm_id)

        return (gse_id, len(valid_gsms), saved, time.time() - t0, "ok")
    except Exception as e:
        import traceback as tb
        return (gse_id, 0, [], 0, f"error: {e}\n{tb.format_exc()}")


# ============================================================
# Main
# ============================================================
def main():
    total_t0 = time.time()
    os.makedirs(SPECIES_DIR, exist_ok=True)

    # 列名探测
    print(f"[init] 读取 obs 探测列名 ...")
    adata = sc.read_h5ad(H5AD_PATH, backed='r')
    cols = list(adata.obs.columns)
    adata.file.close()
    gsm_col = next((c for c in ['GSM', 'gsm', 'GSM_ID', 'gsm_id', 'sample', 'Sample'] if c in cols), None)
    gse_col = next((c for c in ['GSE', 'gse', 'GSE_ID', 'gse_id', 'study', 'Study', 'project', 'Project'] if c in cols), None)
    if not gsm_col or not gse_col:
        sys.exit(f"找不到 GSM/GSE 列，现有: {cols}")
    print(f"[init] GSM='{gsm_col}'  GSE='{gse_col}'")
    print(f"[init] 目标 GSEs: {TARGET_GSES}")
    print(f"[init] 注意：将覆盖这三个 GSE 下所有 GSM 的 DEG 文件\n")

    # 并行处理
    tasks = [(gse, H5AD_PATH, gsm_col, gse_col, SPECIES_DIR) for gse in TARGET_GSES]
    all_written = []  # (gse, gsm)

    with Pool(processes=min(N_WORKERS, len(tasks))) as pool:
        for res in pool.imap_unordered(process_one_gse, tasks):
            gse_id, n_gsms, saved, elapsed, status = res
            print(f"  {gse_id}: n_valid_GSMs={n_gsms}, saved={len(saved)}, {elapsed:.1f}s, {status}")
            for g in saved:
                all_written.append((gse_id, g))

    # 写入清单
    print(f"\n{'='*60}\n 写入清单\n{'='*60}")
    for gse, gsm in sorted(all_written):
        path = os.path.join(SPECIES_DIR, gse, gsm)
        print(f"  + {path}")
    print(f"\n共写入 {len(all_written)} 个 GSM 目录")

    # 自检：目标 GSM 是否齐
    print(f"\n[check] 目标 GSM 状态：")
    all_ok = True
    for gse, gsms in EXPECTED.items():
        for gsm in gsms:
            f = os.path.join(SPECIES_DIR, gse, gsm, "DEGs_all.csv")
            ok = os.path.exists(f)
            all_ok &= ok
            print(f"  {'✓' if ok else '✗'} {gse}/{gsm}")
    print(f"\n{'✓ 全部补齐' if all_ok else '✗ 仍有缺失'}")
    print(f"总用时: {time.time()-total_t0:.1f}s")


if __name__ == "__main__":
    main()