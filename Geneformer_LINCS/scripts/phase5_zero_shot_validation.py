"""Phase 5 — RC validation suite for the Immune-only zero-shot pipeline.

Consolidates the relevant reviewer-concern checks into one script so that
they all read from the new `*_zs*` outputs (not the old 7-CT production data).

Sub-checks COVERED in this script:
    RC#23  per-gene effective N  (Immune)
    RC#3a  NaN bias check        (uses nan_log.csv written natively by
                                  phase2_zero_shot_ko.py centroid extraction)
    RC#7   binomial GLM          (cross_species enrichment of Immune reversal)

Sub-checks DEFERRED (intentional, decided 2026-04-27):
    RC#1   driver sanity         — would need ~8h GPU to redo on 2 CT.
                                   §Limitations cite the 7-CT pre-revision
                                   driver sanity check as already establishing
                                   broad-immune driver enrichment in Immune.
    RC#22  multi-seed bootstrap  — REPLACED by phase2_sampling_stability.py
                                   max_ncells sweep (n=2000/2500/3000), which
                                   answers the same robustness question
                                   without monkey-patching Geneformer's
                                   downsample_and_sort. See manuscript Methods
                                   for justification.
    RC#3b  fp32 vs fp16          — moot. Geneformer V2 default IS fp32
                                   (verified by inspecting perturber_utils.py
                                   on the local Desktop reference). Both
                                   "fp16" and "fp32" runs in the previous
                                   round produced bit-identical pickles
                                   because both were already fp32.
    RC#4   CLS vs mean-pool      — upstream limitation: Geneformer V2
                                   InSilicoPerturber.validate_options
                                   accepts only cell_emb_style='mean_pool'.
    RC#5a  L1000 top-N stability — not retained in this focused pipeline.
                                   It should be rerun only if required for a
                                   specific reviewer response, using the
                                   current Phase 3 output namespace.

Outputs go to results/validation_zs/.

Run (CPU, ~minutes):
    python scripts/phase5_zero_shot_validation.py --ref-n 2000
"""
from __future__ import annotations

import argparse, pickle, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    CANDIDATES_OUT, RESULTS_DIR, FIGURES_DIR,
)


FOCUS_CELL_TYPES = ["Immune"]
LOW_CONF_THR    = 200


# ===========================================================================
def rc23_effective_n(ref_n: int, out_dir: Path):
    """Per-gene N (cell pool size used) per cell type."""
    perturb_root = RESULTS_DIR / f"perturbation_zs_n{ref_n}"
    rows = []
    for ct in FOCUS_CELL_TYPES:
        per = perturb_root / ct / "per_gene"
        if not per.exists():
            print(f"[rc23] {ct}: no per_gene/ at {per}, skip"); continue
        for pkl in per.glob(f"in_silico_delete_{ct}_ko_*_cell_embs_dict_*_raw.pickle"):
            ens = pkl.name.split(f"{ct}_ko_")[1].split("_cell_embs_dict_")[0]
            n = _count_cells_in_pickle(pkl)
            rows.append({"cell_type": ct, "Ensembl_ID": ens, "n_cells": n})
    df = pd.DataFrame(rows)
    if df.empty:
        print("[rc23] empty"); return
    df = df[df["n_cells"] >= 0]
    df["low_confidence"] = (df["n_cells"] < LOW_CONF_THR).astype(int)
    df.to_csv(out_dir / "effective_n_per_gene.csv", index=False)
    summ = (df.groupby("cell_type")["n_cells"]
              .agg(["min", "median", "mean", "max", "count"]).round(1).reset_index())
    pct = (df.groupby("cell_type")["low_confidence"].mean() * 100).round(1)
    summ["pct_low_confidence"] = summ["cell_type"].map(pct)
    summ.to_csv(out_dir / "effective_n_summary.csv", index=False)
    print("\n=== RC#23 effective N ===")
    print(summ.to_string(index=False))


def _count_cells_in_pickle(p: Path) -> int:
    try:
        with open(p, "rb") as f:
            d = pickle.load(f)
    except Exception:
        return -1
    if not isinstance(d, dict): return -1
    n = 0
    for state, inner in d.items():
        if hasattr(inner, "values"):
            for v in inner.values():
                if hasattr(v, "__len__"):
                    n = max(n, len(v))
    return n


# ===========================================================================
def rc3a_nan_bias(ref_n: int, out_dir: Path):
    """Aggregate nan_log.csv (written by phase2_zero_shot_ko.py per CT)."""
    perturb_root = RESULTS_DIR / f"perturbation_zs_n{ref_n}"
    rows = []
    for ct in FOCUS_CELL_TYPES:
        f = perturb_root / ct / "nan_log.csv"
        if not f.exists():
            print(f"[rc3a] {ct}: no nan_log.csv, skip"); continue
        d = pd.read_csv(f); d["cell_type"] = ct
        rows.append(d)
    if not rows:
        print("[rc3a] no nan_log files"); return
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(out_dir / "nan_log_summary.csv", index=False)
    print("\n=== RC#3a NaN cells dropped during centroid extraction ===")
    print(df.to_string(index=False))


# ===========================================================================
def rc7_binomial_glm(ref_n: int, out_dir: Path):
    """Per-gene Immune reversal status modelled by cross-species support."""
    f = RESULTS_DIR / f"shift_scores_zs_n{ref_n}" / "ko_shift_scores.csv"
    if not f.exists():
        print(f"[rc7] missing {f}"); return
    shift = pd.read_csv(f)
    shift = shift[shift["cell_type"].isin(FOCUS_CELL_TYPES)].copy()
    cand = pd.read_csv(CANDIDATES_OUT / "human_targets.csv")
    cs_col = next((c for c in ("cross_species", "cross_species_support")
                   if c in cand.columns), None)
    if cs_col is None:
        print("[rc7] no cross_species column in candidates"); return
    cand_cs = pd.to_numeric(cand[cs_col], errors="coerce").fillna(0).astype(int)
    cs_map = dict(zip(cand["human_ensembl"].astype(str), cand_cs))
    shift["cross_species"] = shift["Ensembl_ID"].astype(str).map(cs_map).fillna(0).astype(int)
    shift["reversed"] = (shift["Shift_to_goal_end"] > 0).astype(int)

    g = (shift.groupby("Ensembl_ID")
              .agg(n_reversed=  ("reversed", "sum"),
                   n_total=     ("reversed", "size"),
                   cross_species=("cross_species", "max"))
              .reset_index())
    g["n_failure"] = g["n_total"] - g["n_reversed"]
    g["frac"]     = g["n_reversed"] / g["n_total"]
    g.to_csv(out_dir / "cross_species_glm_data.csv", index=False)

    print("\n=== RC#7 cross-species GLM (Immune-only) ===")
    print(f"  cross_species=1 : n={(g.cross_species==1).sum()}, "
          f"mean k/N = {g.loc[g.cross_species==1,'frac'].mean():.3f}")
    print(f"  cross_species=0 : n={(g.cross_species==0).sum()}, "
          f"mean k/N = {g.loc[g.cross_species==0,'frac'].mean():.3f}")

    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        glm = smf.glm("frac ~ cross_species",
                      data=g, family=sm.families.Binomial(),
                      freq_weights=g["n_total"]).fit()
        print(glm.summary())
        with open(out_dir / "cross_species_glm.txt", "w") as f:
            f.write(str(glm.summary()) + "\n")
    except Exception as e:
        print(f"[rc7] statsmodels failed: {e}")

    from scipy.stats import fisher_exact
    a = int(g.loc[g.cross_species == 1, "n_reversed"].sum())
    b = int(g.loc[g.cross_species == 1, "n_failure"].sum())
    c = int(g.loc[g.cross_species == 0, "n_reversed"].sum())
    d = int(g.loc[g.cross_species == 0, "n_failure"].sum())
    odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")
    print(f"\nFisher exact (one-sided greater): "
          f"contingency [[{a},{b}],[{c},{d}]]  OR={odds:.3f}  p={p:.3g}")


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-n", type=int, default=2000)
    ap.add_argument("--skip", nargs="*", default=[],
                    help="Sub-check names to skip: rc23 rc3a rc7")
    args = ap.parse_args()

    out_dir = RESULTS_DIR / "validation_zs"
    out_dir.mkdir(parents=True, exist_ok=True)

    if "rc23" not in args.skip:  rc23_effective_n(args.ref_n, out_dir)
    if "rc3a" not in args.skip:  rc3a_nan_bias   (args.ref_n, out_dir)
    if "rc7"  not in args.skip:  rc7_binomial_glm(args.ref_n, out_dir)

    print(f"\n[done] outputs in {out_dir}/")


if __name__ == "__main__":
    main()
