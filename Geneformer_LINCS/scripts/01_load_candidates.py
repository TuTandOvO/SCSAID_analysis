"""Load Candidate_gene.csv, map human symbols → Ensembl using Geneformer's LOCAL
gene_name_id_dict (offline, works on compute node without network).

Output (to results/candidates/):
    human_targets.csv
    dropped.csv

Run:
    python 01_load_candidates.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    CANDIDATE_CSV, CANDIDATES_OUT, GF_TOKEN_DICT, GF_SYMBOL_TO_ENS,
)


def load_symbol_to_ensembl() -> dict[str, str]:
    """Geneformer's built-in symbol → ensembl dict (~40k genes)."""
    with open(GF_SYMBOL_TO_ENS, "rb") as f:
        d = pickle.load(f)
    return d


def load_gf_vocab() -> set[str]:
    """Ensembl IDs present in Geneformer token dictionary."""
    with open(GF_TOKEN_DICT, "rb") as f:
        tok = pickle.load(f)
    return {k for k in tok.keys() if isinstance(k, str) and k.startswith("ENS")}


def main() -> None:
    df = pd.read_csv(CANDIDATE_CSV)
    print(f"[load] {len(df)} rows, groups: {df['group'].value_counts().to_dict()}")

    df["cross_species_support"] = df["group"] == "Overlap disease-high"

    sym2ens = load_symbol_to_ensembl()
    print(f"[map] Geneformer gene_name_id_dict: {len(sym2ens)} symbols")

    # human symbols
    df["human_ensembl"] = df["human_gene"].map(sym2ens)
    # mouse symbols — Geneformer dict is human-only; leave mouse as NA
    df["mouse_ensembl"] = pd.NA

    # GF vocab check
    gf_vocab = load_gf_vocab()
    df["in_gf_vocab_human"] = df["human_ensembl"].isin(gf_vocab)
    print(f"[gf] vocab={len(gf_vocab)}  in-vocab: "
          f"{df['in_gf_vocab_human'].sum()}/{len(df)}")

    human_targets = df[[
        "human_gene", "human_ensembl", "human_rank",
        "human_weight_mean_beta", "human_selection_freq",
        "combined_priority_score",
        "cross_species_support",
        "mouse_gene", "mouse_ensembl", "mouse_rank",
        "in_gf_vocab_human",
    ]].copy()

    dropped = df[df["human_ensembl"].isna() | ~df["in_gf_vocab_human"]][
        ["human_gene", "human_ensembl", "in_gf_vocab_human", "group", "note"]
    ]

    human_targets.to_csv(CANDIDATES_OUT / "human_targets.csv", index=False)
    dropped.to_csv(CANDIDATES_OUT / "dropped.csv", index=False)

    print(f"[save] human_targets={len(human_targets)} "
          f"(cross_species={human_targets['cross_species_support'].sum()}, "
          f"gf_usable={human_targets['in_gf_vocab_human'].sum()})  "
          f"dropped={len(dropped)}")
    print(f"[save] → {CANDIDATES_OUT}")


if __name__ == "__main__":
    main()
