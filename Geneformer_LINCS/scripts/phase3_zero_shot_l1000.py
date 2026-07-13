"""Phase 3 — L1000 connectivity for Immune cells (zero-shot pipeline).

Uses XSum / cmapPy connectivity restricted to the final Immune compartment and
writes into `results/l1000_zs/`, parallel to `results/perturbation_zs_n*`.

Three pert_types per run:
    trt_sh.cgs   consensus shRNA (primary KD evidence)
    trt_sh       raw shRNA       (sensitivity / fallback)
    trt_cp       compounds       (drug repurposing — DO NOT truncate top-200,
                                  use --top-compounds 0 so positive controls
                                  remain visible regardless of rank)

Run (CPU):
    python scripts/phase3_zero_shot_l1000.py --pert-type trt_sh.cgs
    python scripts/phase3_zero_shot_l1000.py --pert-type trt_sh
    python scripts/phase3_zero_shot_l1000.py --pert-type trt_cp --top-compounds 0
"""
from __future__ import annotations

import argparse, pickle, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from cmapPy.pandasGEXpress.parse import parse as parse_gctx

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    CANDIDATES_OUT, DEG_OUT, RESULTS_DIR,
    L1000_GCTX, L1000_SIG_INFO, L1000_GENE_INFO,
    GF_SYMBOL_TO_ENS, L1000_TOP_N_GENES_QUERY,
)


FOCUS_CELL_TYPES = ["Immune"]
L1000_OUT_ZS = RESULTS_DIR / "l1000_zs"


# ---------------------------------------------------------------------------
def load_ens2sym() -> dict[str, str]:
    with open(GF_SYMBOL_TO_ENS, "rb") as f:
        sym2ens = pickle.load(f)
    out: dict[str, str] = {}
    for s, e in sym2ens.items():
        if e and e not in out:
            out[e] = s
    return out


def build_query(deg_csv: Path, ens2sym: dict, top_n: int):
    df = pd.read_csv(deg_csv)
    df = df[df["padj"] < 0.05].dropna(subset=["log2FoldChange", "gene"])
    df["sym"] = df["gene"].astype(str).map(ens2sym)
    df = df.dropna(subset=["sym"])
    up   = df.sort_values("log2FoldChange", ascending=False).head(top_n)["sym"].tolist()
    down = df.sort_values("log2FoldChange", ascending=True ).head(top_n)["sym"].tolist()
    return up, down


def xsum(up, down, ref_sig):
    up_in   = [g for g in up   if g in ref_sig.index]
    down_in = [g for g in down if g in ref_sig.index]
    n = len(up_in) + len(down_in)
    if n == 0:
        return np.nan, 0
    return float((ref_sig[up_in].sum() - ref_sig[down_in].sum()) / n), n


def compute_one_celltype(deg_csv: Path, sig_info: pd.DataFrame,
                          gene_info: pd.DataFrame, pert_type: str,
                          top_n_query: int, top_compounds: int,
                          ens2sym: dict) -> pd.DataFrame:
    ct = deg_csv.stem.replace("_pso_vs_healthy", "")
    print(f"[{ct}] building query (top {top_n_query} up/down)")
    up, down = build_query(deg_csv, ens2sym, top_n_query)
    print(f"[{ct}] query: {len(up)} up + {len(down)} down")

    sigs = sig_info[sig_info["pert_type"] == pert_type].copy()
    if pert_type in ("trt_sh", "trt_sh.cgs", "trt_xpr"):
        cand = pd.read_csv(CANDIDATES_OUT / "human_targets.csv")
        cand_syms = set(cand["human_gene"].dropna().astype(str))
        sigs = sigs[sigs["pert_iname"].isin(cand_syms)]
        print(f"[{ct}] {pert_type}: {len(sigs)} sigs covering "
              f"{sigs['pert_iname'].nunique()} candidate genes")
    else:
        print(f"[{ct}] {pert_type}: {len(sigs)} compound sigs")

    if sigs.empty:
        return pd.DataFrame()

    print(f"[{ct}] reading GCTX subset ({len(sigs)} sigs)")
    mat = parse_gctx(str(L1000_GCTX), cid=sigs.index.tolist()).data_df
    sym = gene_info.loc[mat.index, "pr_gene_symbol"]
    mat.index = sym.values
    mat = mat[~mat.index.duplicated(keep="first")]
    mat = mat[mat.index.notna()]

    rows = []
    for sig_id in mat.columns:
        score, n = xsum(up, down, mat[sig_id])
        md = sigs.loc[sig_id]
        rows.append({
            "sig_id": sig_id, "pert_iname": md["pert_iname"],
            "cell_line": md.get("cell_id", ""),
            "time": md.get("pert_itime", ""),
            "dose": md.get("pert_idose", ""),
            "xsum": score, "n_matched": n,
        })
    df = pd.DataFrame(rows)

    agg = (df.groupby("pert_iname")
             .agg(xsum_median=("xsum", "median"),
                  xsum_mean  =("xsum", "mean"),
                  xsum_std   =("xsum", "std"),
                  n_sigs     =("xsum", "count"))
             .reset_index()
             .rename(columns={"pert_iname": "gene"}))
    agg["cell_type"] = ct
    agg["pert_type"] = pert_type
    agg["tau"] = (agg["xsum_median"].rank(pct=True, method="average") * 200) - 100
    agg = agg.sort_values("xsum_median")

    # truncation only when --top-compounds > 0 AND drug-repurposing mode
    if top_compounds and pert_type == "trt_cp":
        agg = agg.head(top_compounds)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pert-type",
                    choices=["trt_sh", "trt_sh.cgs", "trt_xpr", "trt_cp"],
                    required=True)
    ap.add_argument("--top-n-query", type=int, default=L1000_TOP_N_GENES_QUERY)
    ap.add_argument("--top-compounds", type=int, default=0,
                    help="0 = keep all (recommended for positive control). "
                         "200 = production legacy; not advised for zero-shot run.")
    args = ap.parse_args()

    print("[load] sig_info, gene_info, ens2sym")
    sig_info  = pd.read_csv(L1000_SIG_INFO,  sep="\t", low_memory=False).set_index("sig_id")
    gene_info = pd.read_csv(L1000_GENE_INFO, sep="\t", dtype={"pr_gene_id": str}
                             ).set_index("pr_gene_id")
    ens2sym = load_ens2sym()
    print(f"  sig_info {len(sig_info)} sigs, gene_info {len(gene_info)} probes, "
          f"ens2sym {len(ens2sym)} entries")

    deg_files = sorted(DEG_OUT.glob("*_pso_vs_healthy.csv"))
    deg_files = [f for f in deg_files
                 if f.stem.replace("_pso_vs_healthy", "") in FOCUS_CELL_TYPES]
    if not deg_files:
        raise RuntimeError(f"No DEG CSVs for {FOCUS_CELL_TYPES} in {DEG_OUT}")
    print(f"[focus] running on {[f.stem.replace('_pso_vs_healthy','') for f in deg_files]}")

    L1000_OUT_ZS.mkdir(parents=True, exist_ok=True)
    pt_safe = args.pert_type.replace(".", "_")
    suffix = "drug_repurposing" if args.pert_type == "trt_cp" else f"{pt_safe}_tau"

    all_agg = []
    for deg in deg_files:
        agg = compute_one_celltype(
            deg, sig_info, gene_info,
            pert_type=args.pert_type,
            top_n_query=args.top_n_query,
            top_compounds=args.top_compounds,
            ens2sym=ens2sym,
        )
        if agg.empty:
            continue
        ct = deg.stem.replace("_pso_vs_healthy", "")
        out = L1000_OUT_ZS / f"{ct}_{suffix}.csv"
        agg.to_csv(out, index=False)
        print(f"  saved {out}  rows={len(agg)}")
        all_agg.append(agg)

    if all_agg:
        merged = pd.concat(all_agg, ignore_index=True)
        out = L1000_OUT_ZS / f"all_{suffix}.csv"
        merged.to_csv(out, index=False)
        print(f"[save] concat → {out}")


if __name__ == "__main__":
    main()
