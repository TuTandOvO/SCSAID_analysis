"""Geneformer tokenization: h5ad → .dataset (HuggingFace Arrow).

Splits by cell type so that downstream perturbation can operate per-context.

Input:  results/adata/human_pso_healthy.h5ad
Output: results/tokenized/{cell_type}/  (HF arrow dataset)
        results/tokenized/_full.dataset/  (single concat dataset with cell_type col)

Run:
    python 04_tokenize.py --adata results/adata/human_pso_healthy.h5ad \
        --groupby Gross_Map
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import anndata as ad
from geneformer import TranscriptomeTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from config import TOKEN_OUT, GF_GENE_MEDIAN, GF_TOKEN_DICT, GF_MAX_LEN


def tokenize_one(h5ad_path: Path, out_dir: Path, nproc: int = 8) -> None:
    """Wrap TranscriptomeTokenizer for one h5ad → arrow dataset."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # TranscriptomeTokenizer reads all .h5ad in a directory — stage via symlink.
    staging = out_dir / "_staging"
    staging.mkdir(exist_ok=True)
    staged = staging / h5ad_path.name
    if staged.exists() or staged.is_symlink():
        staged.unlink()
    staged.symlink_to(h5ad_path.resolve())

    tk = TranscriptomeTokenizer(
        custom_attr_name_dict={
            "disease": "disease",
            "GSM": "sample",
            "Gross_Map": "cell_type_gross",
            "Fine_Map": "cell_type_fine",
        },
        nproc=nproc,
        model_input_size=GF_MAX_LEN,           # 4096 for gc104M V2
        # Disable CLS/EOS: EmbExtractor's "cls" mode is not fully implemented
        # (get_embs has no "cls" branch populating embs_list), while
        # InSilicoPerturber raises when CLS is in data but emb_mode != "cls".
        # Using mean-pool over gene tokens ("cell" mode) is the consistent,
        # science-OK alternative. Re-enable if using Geneformer classifier.
        special_token=False,
        gene_median_file=str(GF_GENE_MEDIAN),
        token_dictionary_file=str(GF_TOKEN_DICT),
    )
    tk.tokenize_data(
        data_directory=str(staging),
        output_directory=str(out_dir),
        output_prefix=h5ad_path.stem,
        file_format="h5ad",
    )
    shutil.rmtree(staging)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adata", required=True)
    ap.add_argument("--groupby", default="Gross_Map")
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--per-celltype", action="store_true",
                    help="Additionally write per-cell-type tokenized datasets")
    args = ap.parse_args()

    adata_path = Path(args.adata)

    # full dataset
    print("[tokenize] full dataset")
    tokenize_one(adata_path, TOKEN_OUT / "_full", nproc=args.nproc)

    # per cell type (write temp h5ad per cell type, tokenize, clean up)
    if args.per_celltype:
        adata = ad.read_h5ad(adata_path)
        for ct in adata.obs[args.groupby].dropna().unique():
            safe = str(ct).replace(" ", "_").replace("/", "_")
            sub = adata[adata.obs[args.groupby] == ct].copy()
            if sub.n_obs < 50:
                print(f"[skip] {ct} n={sub.n_obs}")
                continue
            tmp = TOKEN_OUT / f"_tmp_{safe}.h5ad"
            sub.write_h5ad(tmp)
            print(f"[tokenize] {ct} n={sub.n_obs}")
            tokenize_one(tmp, TOKEN_OUT / safe, nproc=args.nproc)
            tmp.unlink()

    print(f"[save] → {TOKEN_OUT}")


if __name__ == "__main__":
    main()
