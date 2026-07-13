"""Per-cell-type pseudobulk DEG: psoriasis vs healthy (PyDESeq2).

Used for:
  (a) building the psoriasis/healthy direction used by the current
      Geneformer and L1000 phases;
  (b) building the query signature for `phase3_zero_shot_l1000.py`.

Input:  results/adata/human_pso_healthy.h5ad
Output: results/deg/{cell_type}_pso_vs_healthy.csv   (gene, log2FC, padj, ...)
        results/deg/_all_deg_long.csv   (concatenated with cell_type column)

Run:
    python 03_pseudobulk_deg.py --adata results/adata/human_pso_healthy.h5ad \
        --groupby Gross_Map --min-cells 10 --min-samples 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

sys.path.insert(0, str(Path(__file__).parent))
from config import DEG_OUT


def pseudobulk(adata: ad.AnnData, sample_col: str = "GSM",
               cond_col: str = "disease") -> pd.DataFrame:
    """Sum counts per sample → rows=samples, cols=genes. Returns (counts_df, meta)."""
    samples = adata.obs[sample_col].astype(str).values
    uniq = pd.unique(samples)
    X = adata.X.tocsr()
    mat = np.zeros((len(uniq), adata.n_vars), dtype=np.int64)
    for i, s in enumerate(uniq):
        idx = np.where(samples == s)[0]
        mat[i] = np.asarray(X[idx].sum(axis=0)).flatten()
    counts = pd.DataFrame(mat, index=uniq, columns=adata.var.index)
    meta = (adata.obs[[sample_col, cond_col]]
            .drop_duplicates(sample_col).set_index(sample_col).loc[uniq])
    return counts, meta


def run_deseq(counts: pd.DataFrame, meta: pd.DataFrame,
              cond_col: str = "disease",
              ref: str = "healthy") -> pd.DataFrame:
    # drop all-zero genes
    keep = counts.sum(axis=0) > 0
    counts = counts.loc[:, keep]
    # New pydeseq2 API uses formulaic `design` string; set ref via Categorical order.
    meta = meta.copy()
    other = [c for c in meta[cond_col].unique() if c != ref][0]
    meta[cond_col] = pd.Categorical(meta[cond_col], categories=[ref, other])
    dds = DeseqDataSet(counts=counts, metadata=meta, design=f"~{cond_col}", quiet=True)
    dds.deseq2()
    stats = DeseqStats(dds, contrast=[cond_col, other, ref], quiet=True)
    stats.summary()
    res = stats.results_df.copy()
    res["gene"] = res.index
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adata", required=True)
    ap.add_argument("--groupby", default="Gross_Map",
                    help="Cell type column (Gross_Map or Fine_Map)")
    ap.add_argument("--sample-col", default="GSM")
    ap.add_argument("--cond-col", default="disease")
    ap.add_argument("--min-cells", type=int, default=10,
                    help="Min cells per sample×celltype to keep")
    ap.add_argument("--min-samples", type=int, default=3,
                    help="Min samples per condition per cell type")
    args = ap.parse_args()

    print(f"[load] {args.adata}")
    adata = ad.read_h5ad(args.adata)

    cell_types = adata.obs[args.groupby].dropna().unique()
    all_deg = []

    for ct in cell_types:
        sub = adata[adata.obs[args.groupby] == ct].copy()
        # filter samples with enough cells
        n_per = sub.obs.groupby(args.sample_col).size()
        good = n_per[n_per >= args.min_cells].index
        sub = sub[sub.obs[args.sample_col].isin(good)].copy()
        # need both conditions
        n_cond = sub.obs.groupby(args.cond_col)[args.sample_col].nunique()
        if (n_cond < args.min_samples).any() or len(n_cond) < 2:
            print(f"[skip] {ct}: insufficient samples ({n_cond.to_dict()})")
            continue
        print(f"[deg]  {ct}  n_cells={sub.n_obs}  samples={n_cond.to_dict()}")

        counts, meta = pseudobulk(sub, args.sample_col, args.cond_col)
        try:
            res = run_deseq(counts, meta, args.cond_col)
        except Exception as e:
            print(f"[err]  {ct}: {e}")
            continue
        res["cell_type"] = ct
        safe = str(ct).replace(" ", "_").replace("/", "_")
        res.to_csv(DEG_OUT / f"{safe}_pso_vs_healthy.csv", index=False)
        all_deg.append(res)

    if all_deg:
        pd.concat(all_deg, ignore_index=True).to_csv(
            DEG_OUT / "_all_deg_long.csv", index=False
        )
    print(f"[save] → {DEG_OUT}")


if __name__ == "__main__":
    main()
