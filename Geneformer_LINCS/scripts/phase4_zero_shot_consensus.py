"""Phase 4 — RRA consensus + double-tier output for Immune cells.

Reads:
    results/shift_scores_zs_n{REF}/ko_shift_scores.csv        (Geneformer KO)
    results/l1000_zs/Immune_trt_sh_cgs_tau.csv (L1000 cgs)
    results/l1000_zs/Immune_trt_sh_tau.csv     (L1000 raw)
    results/candidates/human_targets.csv                       (cross-species tag)

Per-(gene, cell_type) ranking aggregation across:
    - GF_KO            (Shift_to_goal_end, descending = better)
    - L1000_sh_cgs     (xsum_median ascending = better; reverses disease)
    - L1000_sh         (xsum_median ascending = better)

Tier assignment per gene:
    triple_evidence    has all three rankings present (= L1000-covered candidates, ~18)
    gf_only            only GF_KO ranking present     (~49 GF-only candidates)

Outputs:
    results/consensus_zs/per_celltype_consensus.csv
    results/consensus_zs/final_ranking.csv
        columns: gene, mean_rra_score, n_cell_types_top25pct,
                 supporting_methods, tier, cross_species_support, ...
    results/consensus_zs/final_ranking_triple.csv   (filter to tier=triple_evidence)
    results/consensus_zs/final_ranking_gf_only.csv  (filter to tier=gf_only)

Run (CPU, seconds):
    python scripts/phase4_zero_shot_consensus.py --ref-n 2000
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS_DIR, CANDIDATES_OUT


FOCUS_CELL_TYPES = ["Immune"]
RRA_TOP_QUANTILE = 0.25


# ---------------------------------------------------------------------------
def rra_score(ranks: np.ndarray, n_per_method: list[int]) -> float:
    """Kolde 2012 RRA: beta-CDF of min normalised rank.
    `ranks[k]` = gene's rank in method k; `n_per_method[k]` = total ranked in k."""
    norm = np.array([r / n for r, n in zip(ranks, n_per_method)])
    norm = np.sort(norm)
    K = len(norm)
    pvals = beta.cdf(norm, np.arange(1, K + 1), K - np.arange(1, K + 1) + 1)
    return float(np.min(pvals))


def build_long(args):
    """Collect ranks per (gene, cell_type, method) into long df."""
    long = []

    # ----- Geneformer KO (Shift_to_goal_end DESCENDING = better) -----
    gf = RESULTS_DIR / f"shift_scores_zs_n{args.ref_n}" / "ko_shift_scores.csv"
    if not gf.exists():
        raise FileNotFoundError(f"GF KO scores not found: {gf}")
    df = pd.read_csv(gf)
    df = df[df["cell_type"].isin(FOCUS_CELL_TYPES)]
    df = df.dropna(subset=["Shift_to_goal_end"]).copy()
    df = df.rename(columns={"Ensembl_ID": "ensembl"})
    for ct, grp in df.groupby("cell_type"):
        grp = grp.sort_values("Shift_to_goal_end", ascending=False).reset_index(drop=True)
        grp["rank"] = grp.index + 1
        grp["method"] = "GF_KO"
        long.append(grp[["ensembl", "cell_type", "method", "rank"]])

    # ----- L1000 (xsum_median ASCENDING = better) -----
    cand_path = CANDIDATES_OUT / "human_targets.csv"
    cand = pd.read_csv(cand_path)
    sym2ens = dict(zip(cand["human_gene"].astype(str),
                       cand["human_ensembl"].astype(str)))
    L1000_OUT_ZS = RESULTS_DIR / "l1000_zs"
    for tag, fname_suffix in (("L1000_sh_cgs", "trt_sh_cgs_tau"),
                               ("L1000_sh",     "trt_sh_tau")):
        for ct in FOCUS_CELL_TYPES:
            f = L1000_OUT_ZS / f"{ct}_{fname_suffix}.csv"
            if not f.exists():
                print(f"[warn] missing {f}")
                continue
            d = pd.read_csv(f).dropna(subset=["xsum_median"]).copy()
            d["ensembl"] = d["gene"].astype(str).map(sym2ens)
            d = d.dropna(subset=["ensembl"])
            d = d.sort_values("xsum_median", ascending=True).reset_index(drop=True)
            d["rank"] = d.index + 1
            d["method"] = tag
            d["cell_type"] = ct
            long.append(d[["ensembl", "cell_type", "method", "rank"]])

    return pd.concat(long, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-n", type=int, default=2000,
                    help="Which max_ncells run of Geneformer to use as reference "
                         "(must already be aggregated; 2000 is production)")
    ap.add_argument("--min-effective-n", type=int, default=200,
                    help="Drop (gene, cell_type) pairs whose Geneformer KO used "
                         "fewer than this many cells. Default 200 protects against "
                         "Immune-cohort small-pool noise (some genes have N<10).")
    args = ap.parse_args()

    out_root = RESULTS_DIR / "consensus_zs"
    out_root.mkdir(parents=True, exist_ok=True)

    long = build_long(args)

    # ----- low-effective-N filter for GF_KO rows ---------------------------
    if args.min_effective_n > 0:
        eff_csv = (RESULTS_DIR / f"perturbation_zs_n{args.ref_n}").parent / \
                   "validation_zs" / "effective_n_per_gene.csv"
        # also try the in-place location written by phase5
        if not eff_csv.exists():
            eff_csv = RESULTS_DIR / "validation_zs" / "effective_n_per_gene.csv"
        if eff_csv.exists():
            eff = pd.read_csv(eff_csv)[["cell_type", "Ensembl_ID", "n_cells"]]
            eff = eff.rename(columns={"Ensembl_ID": "ensembl"})
            n_before = len(long)
            long = long.merge(eff, on=["cell_type", "ensembl"], how="left")
            mask_drop = (long["method"] == "GF_KO") & \
                        (long["n_cells"].fillna(0) < args.min_effective_n)
            print(f"[filter] dropping {int(mask_drop.sum())} GF_KO rows with "
                  f"n_cells < {args.min_effective_n}")
            long = long[~mask_drop].drop(columns=["n_cells"])
            print(f"[filter] long: {n_before} → {len(long)} rows")
        else:
            print(f"[warn] effective_n_per_gene.csv not found at {eff_csv} — "
                  f"skipping low-N filter (run phase5 first to enable)")

    print(f"[long] {len(long)} (gene,CT,method) rank rows after filtering")
    print(long.groupby(["cell_type", "method"]).size().reset_index(name="n"))

    # how many distinct genes per method (= n_per_method for RRA)
    n_per_method_per_ct = (long.groupby(["cell_type", "method"])["ensembl"]
                                .nunique().reset_index(name="n_total"))

    # ----- per (gene, cell_type) RRA across available methods --------------
    rra_rows = []
    for (g, ct), grp in long.groupby(["ensembl", "cell_type"]):
        methods = grp["method"].unique().tolist()
        ranks   = grp["rank"].values
        n_per   = []
        for m in grp["method"]:
            n_per.append(int(n_per_method_per_ct[(n_per_method_per_ct["cell_type"] == ct)
                                                  & (n_per_method_per_ct["method"] == m)
                                                  ]["n_total"].iloc[0]))
        rho = rra_score(ranks, n_per)
        rra_rows.append({
            "ensembl": g, "cell_type": ct,
            "rra_score": rho,
            "n_methods": len(methods),
            "methods": ";".join(sorted(methods)),
        })
    rra_df = pd.DataFrame(rra_rows)

    # top-25% flag per (cell_type, method scope)
    rra_df["top25_flag"] = 0
    for ct in rra_df["cell_type"].unique():
        cut = rra_df[rra_df["cell_type"] == ct]["rra_score"].quantile(RRA_TOP_QUANTILE)
        rra_df.loc[(rra_df["cell_type"] == ct)
                    & (rra_df["rra_score"] <= cut), "top25_flag"] = 1
    rra_df.to_csv(out_root / "per_celltype_consensus.csv", index=False)

    # ----- gene-level aggregation ------------------------------------------
    cand = pd.read_csv(CANDIDATES_OUT / "human_targets.csv")
    cs_col = next((c for c in ("cross_species", "cross_species_support")
                   if c in cand.columns), None)
    cand_cs = (pd.to_numeric(cand[cs_col], errors="coerce").fillna(0).astype(int)
               if cs_col else pd.Series([0] * len(cand)))
    sym_map = dict(zip(cand["human_ensembl"].astype(str),
                       cand["human_gene"].astype(str)))
    cs_map  = dict(zip(cand["human_ensembl"].astype(str), cand_cs))

    g = (rra_df.groupby("ensembl")
                .agg(mean_rra_score=        ("rra_score",  "mean"),
                     n_cell_types_top25pct= ("top25_flag", "sum"),
                     cell_types_top=        ("cell_type",
                                              lambda s: ";".join(s[rra_df.loc[s.index,
                                                                              "top25_flag"] == 1])),
                     supporting_methods=    ("methods",
                                              lambda s: ";".join(sorted(set(
                                                  m for entry in s for m in entry.split(";")
                                              )))))
                .reset_index())

    g["human_gene"]            = g["ensembl"].map(sym_map)
    g["cross_species_support"] = g["ensembl"].map(cs_map).fillna(0).astype(int)
    g["tier"] = np.where(
        g["supporting_methods"].str.contains("L1000_sh") &
        g["supporting_methods"].str.contains("L1000_sh_cgs") &
        g["supporting_methods"].str.contains("GF_KO"),
        "triple_evidence", "gf_only")

    g = g.sort_values(["tier", "mean_rra_score"], ascending=[True, True])
    g.to_csv(out_root / "final_ranking.csv", index=False)
    print(f"[save] {out_root/'final_ranking.csv'}  rows={len(g)}")

    triple = g[g["tier"] == "triple_evidence"].sort_values("mean_rra_score")
    gf_only = g[g["tier"] == "gf_only"      ].sort_values("mean_rra_score")
    triple.to_csv(out_root / "final_ranking_triple.csv", index=False)
    gf_only.to_csv(out_root / "final_ranking_gf_only.csv", index=False)
    print(f"[tier triple_evidence] {len(triple)} genes")
    print(f"[tier gf_only        ] {len(gf_only)} genes")
    print()
    print("=== top 10 triple-evidence ===")
    print(triple.head(10)[["human_gene", "ensembl", "mean_rra_score",
                           "n_cell_types_top25pct", "cross_species_support",
                           "supporting_methods"]].to_string(index=False))
    print()
    print("=== top 10 gf-only ===")
    print(gf_only.head(10)[["human_gene", "ensembl", "mean_rra_score",
                            "n_cell_types_top25pct", "cross_species_support"]
                          ].to_string(index=False))


if __name__ == "__main__":
    main()
