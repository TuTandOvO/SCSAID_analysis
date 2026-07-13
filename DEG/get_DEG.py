#!/usr/bin/env python3
"""
Per-GSM DEG analysis for SkinDB — optimized version.
Optimizations:
  1. Backed-mode metadata scan (no X loaded for planning)
  2. Per-GSE parallel Wilcoxon (multiprocessing, each worker reads only its subset)
  3. Orphan GSMs (single-GSM GSEs) pooled into one global 1-vs-rest run
  4. Sparse matrix preserved throughout
"""

import scanpy as sc
import pandas as pd
import numpy as np
import os
import sys
import time
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial

warnings.filterwarnings('ignore')

# ============================================================
# Config — adjust as needed
# ============================================================
BASE_DIR = "/gpfsdata/home/renyixiang/SkinDB"
SPECIES_FILES = {
    "mouse": os.path.join(BASE_DIR, "10X/mouse/Annotation/Annotated/mouse_with_all_leiden_10threshold_annotated_updated.h5ad"),
    "human": os.path.join(BASE_DIR, "10X/human/Annotation/Annotated/human_with_all_leiden_10threshold_annotated_updated.h5ad"),
}
OUTPUT_DIR = os.path.join(BASE_DIR, "DEG")
N_WORKERS = min(8, cpu_count())   # parallel workers
MIN_CELLS = 10                     # minimum cells per GSM


# ============================================================
# Worker: process one GSE (called in subprocess)
# ============================================================
def process_one_gse(args):
    """
    Read only cells for one GSE via backed mode, run Wilcoxon, save CSVs.
    Receives a plain tuple for pickling compatibility.
    """
    gse_id, h5ad_path, gsm_col, gse_col, species_dir = args

    try:
        import traceback as tb
        t0 = time.time()

        # Read backed → subset → to_memory (only loads X for this GSE)
        adata_full = sc.read_h5ad(h5ad_path, backed='r')
        # Convert to str to avoid categorical comparison issues
        gse_values = adata_full.obs[gse_col].astype(str).values
        mask = gse_values == str(gse_id)
        indices = np.where(mask)[0]

        if len(indices) == 0:
            adata_full.file.close()
            return (gse_id, 0, 0, time.time() - t0, "no_cells_found")

        adata = adata_full[indices].to_memory()
        adata_full.file.close()

        if adata.raw is not None:
            adata = adata.raw.to_adata()

        # Filter small GSMs
        gsm_counts = adata.obs[gsm_col].value_counts()
        valid_gsms = gsm_counts[gsm_counts >= MIN_CELLS].index.tolist()
        if len(valid_gsms) < 2:
            return (gse_id, len(valid_gsms), 0, time.time() - t0, "skipped_few_gsms")

        adata = adata[adata.obs[gsm_col].isin(valid_gsms)].copy()

        # Ensure gsm_col is string type (avoid categorical groupby issues)
        adata.obs[gsm_col] = adata.obs[gsm_col].astype(str)

        # Wilcoxon 1-vs-rest
        sc.tl.rank_genes_groups(
            adata, groupby=gsm_col, method='wilcoxon',
            n_genes=adata.shape[1], use_raw=False, pts=True,
        )

        result = adata.uns['rank_genes_groups']
        saved = 0
        for gsm_id in result['names'].dtype.names:
            df = _extract_deg(result, gsm_id)
            gsm_dir = os.path.join(species_dir, str(gse_id), str(gsm_id))
            os.makedirs(gsm_dir, exist_ok=True)
            df.to_csv(os.path.join(gsm_dir, "DEGs_all.csv"), index=False)
            df[df['pval_adj'] < 0.05].to_csv(
                os.path.join(gsm_dir, "DEGs_significant.csv"), index=False)
            saved += 1

        return (gse_id, len(valid_gsms), saved, time.time() - t0, "ok")

    except Exception as e:
        import traceback as tb
        return (gse_id, 0, 0, 0, f"error: {e}\n{tb.format_exc()}")


def _extract_deg(result, group):
    """Extract DEG DataFrame for one group from rank_genes_groups result."""
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


# ============================================================
# Fallback: global 1-vs-rest for single-GSM GSEs
# ============================================================
def process_orphan_gsms(h5ad_path, gsm_col, gse_col,
                        orphan_gsms, gsm_gse_map, species_dir):
    if len(orphan_gsms) < 2:
        print(f"    Only {len(orphan_gsms)} orphan GSM(s), nothing to compare.")
        return 0

    print(f"    Global 1-vs-rest for {len(orphan_gsms)} orphan GSMs...")
    t0 = time.time()

    adata_full = sc.read_h5ad(h5ad_path, backed='r')
    gsm_values = adata_full.obs[gsm_col].astype(str).values
    orphan_set = set(str(g) for g in orphan_gsms)
    mask = np.array([v in orphan_set for v in gsm_values])
    adata = adata_full[np.where(mask)[0]].to_memory()
    adata_full.file.close()

    if adata.raw is not None:
        adata = adata.raw.to_adata()

    adata.obs[gsm_col] = adata.obs[gsm_col].astype(str)

    sc.tl.rank_genes_groups(
        adata, groupby=gsm_col, method='wilcoxon',
        n_genes=adata.shape[1], use_raw=False, pts=True,
    )

    result = adata.uns['rank_genes_groups']
    saved = 0
    for gsm_id in result['names'].dtype.names:
        df = _extract_deg(result, gsm_id)
        gse_id = gsm_gse_map.get(gsm_id, "unknown_GSE")
        gsm_dir = os.path.join(species_dir, str(gse_id), str(gsm_id))
        os.makedirs(gsm_dir, exist_ok=True)
        df.to_csv(os.path.join(gsm_dir, "DEGs_all.csv"), index=False)
        df[df['pval_adj'] < 0.05].to_csv(
            os.path.join(gsm_dir, "DEGs_significant.csv"), index=False)
        saved += 1

    print(f"    Done: {saved} GSMs, {time.time()-t0:.1f}s")
    return saved


# ============================================================
# Column auto-detection
# ============================================================
def find_columns(columns):
    gsm_col, gse_col = None, None
    for c in ['GSM', 'gsm', 'GSM_ID', 'gsm_id', 'sample', 'Sample']:
        if c in columns:
            gsm_col = c; break
    for c in ['GSE', 'gse', 'GSE_ID', 'gse_id', 'study', 'Study', 'project', 'Project']:
        if c in columns:
            gse_col = c; break
    return gsm_col, gse_col


# ============================================================
# Main
# ============================================================
def main():
    total_t0 = time.time()

    for species, h5ad_path in SPECIES_FILES.items():
        print(f"\n{'='*60}")
        print(f" {species.upper()}")
        print(f"{'='*60}")

        try:
            sp_t0 = time.time()

            # ---- 1. Metadata scan (backed, no X in RAM) ----
            print(f"  [1/3] Scanning metadata (backed mode)...")
            adata = sc.read_h5ad(h5ad_path, backed='r')
            obs = adata.obs.copy()
            n_cells, n_genes = adata.shape
            adata.file.close()
            del adata

            print(f"        {n_cells:,} cells × {n_genes:,} genes")
            print(f"        Obs columns: {list(obs.columns)}")

            gsm_col, gse_col = find_columns(obs.columns)
            if gsm_col is None or gse_col is None:
                print(f"  [!] GSM/GSE column not found. Available: {list(obs.columns)}")
                continue
            print(f"        GSM='{gsm_col}'  GSE='{gse_col}'")

            # Cast to str to avoid categorical issues
            obs[gsm_col] = obs[gsm_col].astype(str)
            obs[gse_col] = obs[gse_col].astype(str)

            # ---- 2. Plan tasks ----
            gsm_counts = obs[gsm_col].value_counts()
            valid_gsms = set(gsm_counts[gsm_counts >= MIN_CELLS].index)
            obs_v = obs[obs[gsm_col].isin(valid_gsms)]

            all_gses = sorted(obs_v[gse_col].unique())
            gse_n_gsms = obs_v.groupby(gse_col)[gsm_col].nunique()
            multi_gses  = gse_n_gsms[gse_n_gsms >= 2].index.tolist()
            single_gses = gse_n_gsms[gse_n_gsms < 2].index.tolist()

            orphan_gsms = obs_v[obs_v[gse_col].isin(single_gses)][gsm_col].unique().tolist()
            gsm_gse_map = obs_v.groupby(gsm_col)[gse_col].first().to_dict()

            n_total_gsms = len(valid_gsms)
            print(f"        {n_total_gsms} valid GSMs | "
                  f"{len(multi_gses)} multi-GSM GSEs | "
                  f"{len(orphan_gsms)} orphan GSMs")
            print(f"        All GSEs in data: {all_gses}")
            print(f"        Multi-GSM GSEs:   {sorted(multi_gses)}")
            print(f"        Single-GSM GSEs:  {sorted(single_gses)}")

            species_dir = os.path.join(OUTPUT_DIR, species)
            os.makedirs(species_dir, exist_ok=True)

            # ---- 3. Parallel per-GSE processing ----
            print(f"  [2/3] Parallel DEG ({N_WORKERS} workers, {len(multi_gses)} GSEs)...")
            tasks = [
                (gse, h5ad_path, gsm_col, gse_col, species_dir)
                for gse in multi_gses
            ]

            total_saved = 0
            failed_gses = []
            if tasks:
                with Pool(processes=N_WORKERS) as pool:
                    for i, res in enumerate(pool.imap_unordered(process_one_gse, tasks)):
                        gse_id, n_gsms, saved, elapsed, status = res
                        total_saved += saved
                        # Print EVERY GSE result (only ~20-30 per species)
                        print(f"        [{i+1}/{len(tasks)}] {gse_id}: "
                              f"{saved} GSMs, {elapsed:.1f}s, {status}")
                        if 'error' in status or saved == 0:
                            failed_gses.append((gse_id, status))

            # ---- 4. Orphan GSMs ----
            print(f"  [3/3] Orphan GSMs...")
            orphan_saved = process_orphan_gsms(
                h5ad_path, gsm_col, gse_col,
                orphan_gsms, gsm_gse_map, species_dir,
            )
            total_saved += orphan_saved

            # ---- 5. Final validation ----
            actual_gses = set()
            if os.path.isdir(species_dir):
                actual_gses = set(os.listdir(species_dir))
            expected_gses = set(all_gses)
            missing = expected_gses - actual_gses
            extra   = actual_gses - expected_gses

            elapsed = time.time() - sp_t0
            print(f"\n  ✓ {species}: {total_saved} GSM folders saved in {elapsed:.1f}s")
            if failed_gses:
                print(f"  ⚠ Failed GSEs ({len(failed_gses)}):")
                for gid, st in failed_gses:
                    print(f"      {gid}: {st}")
            if missing:
                print(f"  ⚠ Missing GSE folders: {sorted(missing)}")
            if not missing and not failed_gses:
                print(f"  ✓ All {len(expected_gses)} GSEs accounted for!")

        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f" Total wall time: {time.time() - total_t0:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
