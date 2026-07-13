"""Phase 2 — Geneformer V2 zero-shot in silico KO (clean refactor).

Design principles:
  • Focus only on the Immune compartment used in the final manuscript.
  • fp32 throughout. Geneformer V2 default load_model uses HuggingFace
    `from_pretrained` with NO torch_dtype kwarg → fp32 by default. Confirmed
    by inspecting `geneformer/perturber_utils.py` (no .half(), no
    torch.float16). We don't need any fp32 patch — the existing tool is
    already fp32.
  • NaN-robust state centroid extraction. We do NOT use EmbExtractor
    .get_state_embs (which has `embs.mean(dim=0)` that propagates a single
    NaN cell into the entire centroid). Instead we forward all cells
    ourselves with vanilla BertModel, drop NaN rows, then mean.
  • No monkey-patching. We use Geneformer's public API as-is:
      - InSilicoPerturber.perturb_data         (KO loop)
      - InSilicoPerturberStats.get_stats       (aggregation)
  • Sampling-stability via max_ncells sweep (not multi-seed bootstrap, which
    would require monkey-patching downsample_and_sort). Run at multiple
    max_ncells values; verify ranking convergence.
  • Idempotent (done_marker per gene).
  • Output dirs are namespaced by max_ncells so multiple sweeps do not
    overwrite each other.

Inputs:
    results/tokenized/{cell_type}/*.dataset
    results/candidates/human_targets.csv
    models/Geneformer/Geneformer-V2-104M

Outputs (per max_ncells N):
    results/perturbation_zs_n{N}/{cell_type}/state_embs.pkl
    results/perturbation_zs_n{N}/{cell_type}/nan_log.csv
    results/perturbation_zs_n{N}/{cell_type}/per_gene/*.pickle
    results/shift_scores_zs_n{N}/{cell_type}_stats*.csv
    results/shift_scores_zs_n{N}/ko_shift_scores.csv

Run on GPU node:
    python scripts/phase2_zero_shot_ko.py --all --max-ncells 2000
    python scripts/phase2_zero_shot_ko.py --all --max-ncells 3000
"""
from __future__ import annotations

import argparse
import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import BertModel
from geneformer import InSilicoPerturber, InSilicoPerturberStats

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    CANDIDATES_OUT, TOKEN_OUT, RESULTS_DIR,
    GF_V2_MODEL_DIR, GF_TOKEN_DICT,
    GF_BATCH_SIZE, GF_EMB_LAYER,
)


FOCUS_CELL_TYPES = ["Immune"]

CELL_STATES_TO_MODEL = {
    "state_key":   "disease",
    "start_state": "psoriasis",
    "goal_state":  "healthy",
    "alt_states":  [],
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def perturb_root(max_ncells: int) -> Path:
    return RESULTS_DIR / f"perturbation_zs_n{max_ncells}"


def shift_root(max_ncells: int) -> Path:
    return RESULTS_DIR / f"shift_scores_zs_n{max_ncells}"


# ===========================================================================
#                       state centroid extraction
# ===========================================================================
def load_pad_token() -> int:
    with open(GF_TOKEN_DICT, "rb") as f:
        tok = pickle.load(f)
    return tok.get("<pad>", 0)


@torch.no_grad()
def cell_mean_pool(model, input_ids, attn_mask, emb_layer):
    """[B, L] → [B, D] mean pool over non-pad tokens at hidden layer emb_layer.
    All compute in the model's dtype (we'll force fp32)."""
    out = model(input_ids=input_ids.to(DEVICE),
                attention_mask=attn_mask.to(DEVICE))
    h = out.hidden_states[emb_layer]                     # [B, L, D]
    m = attn_mask.to(DEVICE).unsqueeze(-1).to(h.dtype)   # [B, L, 1]
    pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)      # [B, D]
    return pooled.float().cpu()


def compute_state_centroids(cell_type: str, ds_path: Path, out_dir: Path,
                            batch_size: int):
    """Forward all cells (fp32), drop NaN rows, mean → fp32 centroid."""
    print(f"[{cell_type}] loading model (fp32)")
    model = BertModel.from_pretrained(
        str(GF_V2_MODEL_DIR),
        output_hidden_states=True,
        torch_dtype=torch.float32,
    ).to(DEVICE).eval()
    pad = load_pad_token()

    print(f"[{cell_type}] loading tokenized dataset")
    ds = load_from_disk(str(ds_path))

    state_embs: dict = {}
    nan_records = []

    for state in ("psoriasis", "healthy"):
        sub = ds.filter(lambda ex: ex["disease"] == state, num_proc=4)
        n = len(sub)
        if n == 0:
            raise RuntimeError(f"{cell_type}: no cells with disease={state}")
        print(f"[{cell_type}] {state}: forward {n} cells (fp32)")

        embs = []
        for start in tqdm(range(0, n, batch_size), desc=state):
            batch = sub[start:start + batch_size]
            ids = batch["input_ids"]
            L = max(len(s) for s in ids)
            input_ids = torch.full((len(ids), L), pad, dtype=torch.long)
            attn_mask = torch.zeros_like(input_ids)
            for i, seq in enumerate(ids):
                input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
                attn_mask[i, :len(seq)] = 1
            embs.append(cell_mean_pool(model, input_ids, attn_mask, GF_EMB_LAYER))

        E = torch.cat(embs, dim=0)
        nan_rows = torch.isnan(E).any(dim=1)
        n_nan = int(nan_rows.sum())
        nan_records.append({"state": state, "n_total": n, "n_nan": n_nan,
                            "pct_nan": round(100 * n_nan / max(n, 1), 3)})
        print(f"[{cell_type}] {state}: dropped {n_nan}/{n} NaN cells "
              f"({100*n_nan/max(n,1):.2f}%)")

        E_clean = E[~nan_rows]
        if len(E_clean) == 0:
            raise RuntimeError(f"{cell_type} {state}: ALL cells NaN")
        centroid = E_clean.mean(dim=0)                  # fp32 mean
        assert not torch.isnan(centroid).any()
        state_embs[state] = centroid                    # KEEP fp32

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "state_embs.pkl", "wb") as f:
        pickle.dump(state_embs, f)
    pd.DataFrame(nan_records).to_csv(out_dir / "nan_log.csv", index=False)
    print(f"[{cell_type}] saved → {out_dir}")

    del model
    torch.cuda.empty_cache()
    return state_embs


# ===========================================================================
#                              per-gene KO loop
# ===========================================================================
def run_ko(cell_type: str, genes: list[str], ds_path: Path,
           state_embs: dict, out_dir: Path,
           max_ncells: int, batch_size: int):
    safe = cell_type.replace(" ", "_").replace("/", "_")
    per_gene_dir = out_dir / "per_gene"
    per_gene_dir.mkdir(parents=True, exist_ok=True)

    n_total = len(genes)
    for idx, gene in enumerate(genes, start=1):
        marker = per_gene_dir / f"{safe}_ko_{gene}_done"
        if marker.exists():
            print(f"[{cell_type} {idx}/{n_total}] {gene} skip (done)")
            continue
        print(f"[{cell_type} {idx}/{n_total}] {gene}")
        try:
            p = InSilicoPerturber(
                perturb_type="delete",
                perturb_rank_shift=None,
                genes_to_perturb=[gene],
                combos=0,
                anchor_gene=None,
                model_type="Pretrained",
                num_classes=0,
                emb_mode="cell",
                cell_emb_style="mean_pool",
                filter_data={"disease": ["psoriasis"]},
                cell_states_to_model=CELL_STATES_TO_MODEL,
                state_embs_dict=state_embs,
                max_ncells=max_ncells,
                emb_layer=GF_EMB_LAYER,
                forward_batch_size=batch_size,
                nproc=4,
                token_dictionary_file=str(GF_TOKEN_DICT),
            )
            p.perturb_data(
                model_directory=str(GF_V2_MODEL_DIR),
                input_data_file=str(ds_path),
                output_directory=str(per_gene_dir),
                output_prefix=f"{safe}_ko_{gene}",
            )
            marker.touch()
        except (RuntimeError, ValueError, KeyError, IndexError) as e:
            print(f"[{cell_type}] {gene}: skip ({type(e).__name__}: {e})")
        except Exception as e:
            print(f"[{cell_type}] {gene}: UNEXPECTED ({type(e).__name__}: {e})")


# ===========================================================================
#                              stats aggregation
# ===========================================================================
def aggregate_stats(cell_type: str, perturb_dir: Path,
                    shift_dir: Path) -> pd.DataFrame | None:
    safe = cell_type.replace(" ", "_").replace("/", "_")
    per_gene_dir = perturb_dir / "per_gene"
    if not per_gene_dir.exists():
        print(f"[{cell_type}] no per_gene/ — skip stats")
        return None
    pkls = list(per_gene_dir.glob(
        f"in_silico_delete_{safe}_ko_*_cell_embs_dict_*_raw.pickle"))
    if not pkls:
        print(f"[{cell_type}] no pickles — skip stats")
        return None
    shift_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{cell_type}] aggregating {len(pkls)} pickles")

    ips = InSilicoPerturberStats(
        mode="goal_state_shift",
        genes_perturbed="all",
        combos=0,
        anchor_gene=None,
        cell_states_to_model=CELL_STATES_TO_MODEL,
        pickle_suffix="_raw.pickle",
        token_dictionary_file=str(GF_TOKEN_DICT),
    )
    ips.get_stats(
        input_data_directory=str(per_gene_dir),
        null_dist_data_directory=None,
        output_directory=str(shift_dir),
        output_prefix=f"{safe}_stats",
    )
    csvs = list(shift_dir.glob(f"{safe}_stats*.csv"))
    if not csvs:
        return None
    df = pd.read_csv(csvs[0])
    df["cell_type"] = cell_type
    return df


# ===========================================================================
#                                  main
# ===========================================================================
def find_dataset(token_dir: Path) -> Path | None:
    cands = list(token_dir.glob("*.dataset"))
    return cands[0] if cands else None


def load_candidate_genes() -> list[str]:
    cand = pd.read_csv(CANDIDATES_OUT / "human_targets.csv")
    cand = cand[cand["in_gf_vocab_human"].astype(str).str.lower() == "true"]
    genes = cand["human_ensembl"].dropna().astype(str).unique().tolist()
    with open(GF_TOKEN_DICT, "rb") as f:
        vocab = set(pickle.load(f).keys())
    in_vocab = [g for g in genes if g in vocab]
    dropped = [g for g in genes if g not in vocab]
    if dropped:
        print(f"[load] dropped {len(dropped)} not in token dict: {dropped}")
    return in_vocab


def run_one(cell_type: str, max_ncells: int, batch_size: int,
            centroids_only: bool):
    if cell_type not in FOCUS_CELL_TYPES:
        print(f"[skip] {cell_type} (not in FOCUS_CELL_TYPES)")
        return
    safe = cell_type.replace(" ", "_").replace("/", "_")
    ds_path = find_dataset(TOKEN_OUT / safe)
    if ds_path is None:
        raise FileNotFoundError(f"no .dataset in {TOKEN_OUT/safe}")

    out_dir   = perturb_root(max_ncells) / safe
    shift_dir = shift_root(max_ncells)
    out_dir.mkdir(parents=True, exist_ok=True)

    # centroids: reuse if clean, else recompute
    cent_pkl = out_dir / "state_embs.pkl"
    state_embs = None
    if cent_pkl.exists():
        with open(cent_pkl, "rb") as f:
            cached = pickle.load(f)
        try:
            ok = all(not torch.isnan(v).any().item() for v in cached.values())
        except Exception:
            ok = False
        if ok:
            print(f"[{cell_type}] reusing fp32 centroids")
            state_embs = cached
    if state_embs is None:
        state_embs = compute_state_centroids(cell_type, ds_path, out_dir,
                                             batch_size=batch_size)

    if centroids_only:
        return

    genes = load_candidate_genes()
    print(f"[{cell_type}] {len(genes)} candidate genes (max_ncells={max_ncells})")
    run_ko(cell_type, genes, ds_path, state_embs, out_dir,
           max_ncells=max_ncells, batch_size=batch_size)
    aggregate_stats(cell_type, out_dir, shift_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-type", default=None)
    ap.add_argument("--all", action="store_true",
                    help="Run all FOCUS_CELL_TYPES (final analysis: Immune)")
    ap.add_argument("--max-ncells", type=int, required=True,
                    help="2000 / 2500 / 3000 — sampling-stability sweep")
    ap.add_argument("--batch-size", type=int, default=GF_BATCH_SIZE)
    ap.add_argument("--centroids-only", action="store_true")
    args = ap.parse_args()

    if args.all:
        cts = FOCUS_CELL_TYPES
    else:
        assert args.cell_type, "--cell-type or --all required"
        cts = [args.cell_type.replace(" ", "_")]

    for ct in cts:
        run_one(ct, args.max_ncells, args.batch_size, args.centroids_only)

    if args.centroids_only:
        return

    # final concat per max_ncells
    rows = []
    sd = shift_root(args.max_ncells)
    for ct in cts:
        safe = ct.replace(" ", "_")
        for c in sd.glob(f"{safe}_stats*.csv"):
            d = pd.read_csv(c); d["cell_type"] = ct
            rows.append(d)
    if rows:
        merged = pd.concat(rows, ignore_index=True)
        out = sd / "ko_shift_scores.csv"
        merged.to_csv(out, index=False)
        print(f"[save] → {out}  rows={len(merged)}")


if __name__ == "__main__":
    main()
