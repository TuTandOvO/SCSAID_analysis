"""Load human h5ad (log1p-CP10K normalized, HVG 8000) → subset psoriasis + healthy
→ exactly invert log1p-CP10K to raw counts using per-cell `total_counts`
→ remap var to Ensembl (Geneformer-compatible, dedup) → save h5ad ready for
Geneformer's TranscriptomeTokenizer.

Input  : human_with_clean_obs.h5ad
          - shape (N, 8000); X = log1p(CP10K) float32
          - obs['total_counts'] = per-cell library size (used for EXACT inverse)
          - var.index = gene symbol
Output : results/adata/human_pso_healthy.h5ad
          - X = raw integer counts (csr_matrix, int32)
          - var.index = Ensembl gene ID, unique (dups dropped)
          - var['ensembl_id'] column (Geneformer V2 tokenizer expects this)
          - var['symbol'] original symbol
          - obs['disease'] in {healthy, psoriasis}
          - obs['n_counts'] recomputed post-rounding

Run:
    python 02_prep_adata.py
    python 02_prep_adata.py --strict
    python 02_prep_adata.py --keep-resolved-separate
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    HUMAN_H5AD, PSO_CONDITIONS, HEALTHY_CONDITIONS, ADATA_OUT,
    GF_SYMBOL_TO_ENS,
)


def load_symbol_to_ensembl() -> dict[str, str]:
    """Geneformer's local gene_name_id_dict — offline, ~40k human symbols."""
    with open(GF_SYMBOL_TO_ENS, "rb") as f:
        return pickle.load(f)


def inverse_log1p_cp10k(X_log: sp.spmatrix, total_counts: np.ndarray) -> sp.csr_matrix:
    """Exact inverse of Scanpy's normalize_total(target=1e4) + log1p.

    Per cell i, gene j:  count[i,j] = round( expm1(X[i,j]) * total_counts[i] / 1e4 )
    Zeros stay zero.
    Vectorized — O(nnz), no Python loop.
    """
    X = X_log.tocsr().astype(np.float32, copy=True)
    X.data = np.expm1(X.data)
    scale = (total_counts.astype(np.float32) / 1e4)
    # Row scaling via diag-matrix multiply (stays sparse, fast).
    X = sp.diags(scale) @ X
    X = X.tocsr()
    X.data = np.rint(X.data).astype(np.int32)
    X.eliminate_zeros()
    return X


def remap_var_to_ensembl(adata: ad.AnnData,
                         sym_to_ens: dict[str, str]) -> ad.AnnData:
    """Map var.index (gene symbols) → Ensembl IDs.
    Drops unmapped genes. Drops duplicate Ensembls (keeps first occurrence).
    Adds var['ensembl_id'] column (Geneformer V2 expects this).
    """
    symbols = adata.var.index.astype(str).tolist()
    ens = np.array([sym_to_ens.get(s) for s in symbols], dtype=object)
    keep = ens != None                                            # noqa: E711

    n_mapped = int(keep.sum())
    print(f"[ensembl] mapped {n_mapped}/{len(symbols)} symbols")
    if n_mapped == 0:
        raise RuntimeError("No symbols mapped — check GF_SYMBOL_TO_ENS path")

    # subset
    sub = adata[:, keep].copy()
    sub.var["symbol"] = sub.var.index.astype(str)
    sub.var["ensembl_id"] = ens[keep].astype(str)

    # dedup — keep first occurrence of each Ensembl ID
    dup = pd.Index(sub.var["ensembl_id"]).duplicated(keep="first")
    if dup.any():
        print(f"[ensembl] dropping {int(dup.sum())} duplicate Ensembl IDs (keep first)")
        sub = sub[:, ~dup].copy()

    sub.var.index = pd.Index(sub.var["ensembl_id"].values, name="ensembl_id")
    return sub


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="Psoriasis only; drop 'Resolved Psoriatic Lesion'")
    ap.add_argument("--keep-resolved-separate", action="store_true",
                    help="Tag Resolved Psoriatic Lesion as its own group")
    args = ap.parse_args()

    print(f"[load] {HUMAN_H5AD}")
    adata = ad.read_h5ad(HUMAN_H5AD)
    print(f"[load] shape={adata.shape}  X dtype={adata.X.dtype}")
    print(f"[load] conditions: {adata.obs['condition'].value_counts().to_dict()}")

    # ---- subset psoriasis + healthy --------------------------------------
    # Note: PSO_CONDITIONS in config is now ["Psoriasis"] only, matching
    # upstream math-modelling cohort. --strict kept as no-op for backwards
    # compatibility; --keep-resolved-separate adds Resolved as a 3rd group.
    pso_conds = list(PSO_CONDITIONS)
    if args.keep_resolved_separate:
        pso_conds = pso_conds + ["Resolved Psoriatic Lesion"]
    keep = adata.obs["condition"].isin(pso_conds + HEALTHY_CONDITIONS)
    adata = adata[keep].copy()
    print(f"[subset] n={adata.n_obs}  "
          f"{adata.obs['condition'].value_counts().to_dict()}")
    if adata.n_obs == 0:
        raise RuntimeError("Empty subset — check condition filter values.")

    # ---- disease label ---------------------------------------------------
    if args.keep_resolved_separate:
        adata.obs["disease"] = adata.obs["condition"].astype(str)
    else:
        adata.obs["disease"] = np.where(
            adata.obs["condition"].isin(HEALTHY_CONDITIONS),
            "healthy", "psoriasis",
        )
    adata.obs["disease"] = adata.obs["disease"].astype(str)

    # ---- exact inverse: log1p(CP10K) → raw counts ------------------------
    if "total_counts" not in adata.obs.columns:
        raise RuntimeError("obs['total_counts'] missing — can't do exact inverse.")
    print("[X] inverting log1p(CP10K) → raw counts (vectorized)")
    adata.X = inverse_log1p_cp10k(adata.X, adata.obs["total_counts"].values)
    adata.obs["n_counts"] = np.asarray(adata.X.sum(axis=1)).flatten()

    # ---- symbol → Ensembl (local dict, dedup) ----------------------------
    sym2ens = load_symbol_to_ensembl()
    print(f"[ensembl] Geneformer dict: {len(sym2ens)} entries")
    adata = remap_var_to_ensembl(adata, sym2ens)

    # ---- save ------------------------------------------------------------
    out = ADATA_OUT / "human_pso_healthy.h5ad"
    # compression='lzf' is ~10x faster to write than gzip at ~80% size.
    adata.write_h5ad(out, compression="lzf")
    print(f"[save] → {out}  final shape={adata.shape}  X dtype={adata.X.dtype}")


if __name__ == "__main__":
    main()
