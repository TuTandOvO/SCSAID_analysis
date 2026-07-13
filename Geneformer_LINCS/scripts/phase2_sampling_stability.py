"""Sampling-stability analysis for Phase 2 zero-shot KO.

After phase2_zero_shot_ko.py has been run at multiple max_ncells values
(typically 2000, 2500, 3000), this script:

  1. Loads each run's per-(gene, cell_type) Shift_to_goal_end.
  2. Computes Spearman ρ + Kendall τ between rankings at different max_ncells.
  3. Computes per-gene CV (std / |mean|) across max_ncells.
  4. Reports concrete pass/fail thresholds for sampling robustness.

Manuscript pass thresholds (by reviewer convention):
  • Spearman ρ ≥ 0.95   → "ranking is robust to sample size"
  • Kendall τ  ≥ 0.85   → "rank order convergence is strong"
  • Median |CV| < 0.15  → "per-gene shift score is stable"

Run (CPU, seconds):
    python scripts/phase2_sampling_stability.py --ns 2000 2500 3000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS_DIR, FIGURES_DIR


def load_run(n: int) -> pd.DataFrame | None:
    f = RESULTS_DIR / f"shift_scores_zs_n{n}" / "ko_shift_scores.csv"
    if not f.exists():
        print(f"[warn] no run for n={n} at {f}")
        return None
    d = pd.read_csv(f)
    keep = [c for c in ("Ensembl_ID", "Gene_name", "Shift_to_goal_end",
                        "Goal_end_vs_random_pval", "cell_type")
            if c in d.columns]
    d = d[keep].copy()
    d["max_ncells"] = n
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[2000, 2500, 3000])
    ap.add_argument("--ref", type=int, default=None,
                    help="Reference max_ncells (default = max of --ns)")
    args = ap.parse_args()

    ref = args.ref or max(args.ns)

    runs = {n: load_run(n) for n in args.ns}
    runs = {n: d for n, d in runs.items() if d is not None}
    if ref not in runs:
        print(f"[err] reference n={ref} missing"); return

    out_root = RESULTS_DIR / "shift_scores_zs_stability"
    out_root.mkdir(parents=True, exist_ok=True)

    # per (cell_type, n) → score column
    big = pd.concat(runs.values(), ignore_index=True)
    big.to_csv(out_root / "all_runs_long.csv", index=False)

    # ===== A. ranking correlation per cell type =================
    rows = []
    for ct in big["cell_type"].unique():
        ref_df = runs[ref]
        ref_df = ref_df[ref_df["cell_type"] == ct][["Ensembl_ID",
                                                     "Shift_to_goal_end"]]
        ref_df = ref_df.rename(columns={"Shift_to_goal_end": f"shift_n{ref}"})
        for n, df in runs.items():
            if n == ref:
                rows.append({
                    "cell_type": ct, "n": n, "vs_ref": ref,
                    "n_genes": len(ref_df),
                    "spearman_rho": 1.0, "spearman_p": 0.0,
                    "kendall_tau": 1.0, "kendall_p": 0.0,
                })
                continue
            sub = df[df["cell_type"] == ct][["Ensembl_ID",
                                              "Shift_to_goal_end"]]
            sub = sub.rename(columns={"Shift_to_goal_end": f"shift_n{n}"})
            m = ref_df.merge(sub, on="Ensembl_ID", how="inner").dropna()
            if len(m) < 5:
                continue
            rho, pr = spearmanr(m[f"shift_n{ref}"], m[f"shift_n{n}"])
            tau, pt = kendalltau(m[f"shift_n{ref}"], m[f"shift_n{n}"])
            rows.append({
                "cell_type": ct, "n": n, "vs_ref": ref,
                "n_genes": len(m),
                "spearman_rho": rho, "spearman_p": pr,
                "kendall_tau": tau, "kendall_p": pt,
            })
    rank_df = pd.DataFrame(rows)
    rank_df.to_csv(out_root / "ranking_correlation.csv", index=False)
    print("\n== Ranking correlation across max_ncells ==")
    print(rank_df.round(4).to_string(index=False))

    # ===== B. per-gene CV across max_ncells =====================
    pivot = (big.pivot_table(index=["cell_type", "Ensembl_ID"],
                             columns="max_ncells",
                             values="Shift_to_goal_end")
                .reset_index())
    n_cols = [c for c in pivot.columns if isinstance(c, (int, np.integer))]
    pivot["mean"] = pivot[n_cols].mean(axis=1)
    pivot["std"]  = pivot[n_cols].std(axis=1)
    pivot["cv"]   = pivot["std"] / pivot["mean"].abs().clip(lower=1e-12)
    pivot.to_csv(out_root / "per_gene_cv.csv", index=False)

    print("\n== Per-gene CV (std / |mean|) across max_ncells ==")
    summ = (pivot.groupby("cell_type")
                 .agg(median_abs_cv=("cv", lambda s: float(s.abs().median())),
                      max_abs_cv   =("cv", lambda s: float(s.abs().max())),
                      n_genes      =("Ensembl_ID", "count"))
                 .reset_index())
    print(summ.round(4).to_string(index=False))

    # ===== C. pass/fail summary =================================
    print("\n== Pass / fail vs manuscript thresholds ==")
    nontrivial = rank_df[rank_df["n"] != ref]
    rho_ok = (nontrivial["spearman_rho"] >= 0.95).all()
    tau_ok = (nontrivial["kendall_tau"]  >= 0.85).all()
    cv_ok  = (summ["median_abs_cv"]      <  0.15).all()
    print(f"  Spearman ρ ≥ 0.95 across all (CT, n vs ref):  "
          f"{'PASS' if rho_ok else 'FAIL'}")
    print(f"  Kendall  τ ≥ 0.85 across all (CT, n vs ref):  "
          f"{'PASS' if tau_ok else 'FAIL'}")
    print(f"  Median |CV| < 0.15 per cell type:             "
          f"{'PASS' if cv_ok else 'FAIL'}")
    print()
    print(f"  Overall sampling stability: "
          f"{'✓ ROBUST' if (rho_ok and tau_ok and cv_ok) else '⚠ NOT ROBUST'}")

    # ===== D. plot ===============================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))

        # rank-rank scatter, all CT
        for ct in big["cell_type"].unique():
            ref_df = runs[ref]
            ref_df = ref_df[ref_df["cell_type"] == ct]
            for n, df in runs.items():
                if n == ref: continue
                sub = df[df["cell_type"] == ct]
                m = ref_df.merge(sub, on="Ensembl_ID",
                                 suffixes=("_ref", f"_n{n}"))
                axes[0].scatter(m["Shift_to_goal_end_ref"],
                                m[f"Shift_to_goal_end_n{n}"],
                                s=12, alpha=0.5,
                                label=f"{ct}, n={n} vs ref={ref}")
        axes[0].plot([-0.05, 0.05], [-0.05, 0.05], "--", lw=0.6, c="#888")
        axes[0].set_xlabel(f"Shift_to_goal_end at n={ref} (ref)")
        axes[0].set_ylabel("Shift_to_goal_end at smaller n")
        axes[0].set_title("Per-gene shift convergence")
        axes[0].legend(fontsize=6, loc="upper left")

        # CV histogram
        axes[1].hist(pivot["cv"].abs().clip(0, 1), bins=40,
                     color="#0071BC", alpha=0.8)
        axes[1].axvline(0.15, ls="--", c="#C1272D", lw=0.8,
                        label="threshold 0.15")
        axes[1].set_xlabel("|CV| (std / |mean|) across max_ncells")
        axes[1].set_ylabel("n_genes")
        axes[1].set_title("Per-gene CV distribution")
        axes[1].legend(fontsize=8)

        fig.tight_layout()
        out = FIGURES_DIR / "robustness" / "phase2_sampling_stability.pdf"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"\n[save] {out}")
    except Exception as e:
        print(f"[warn] plot failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
