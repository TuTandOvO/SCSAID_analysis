#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gzip
import json
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.io import mmwrite


CONFIGS = [
    {
        "species": "human",
        "adata_path": "/gpfsdata/home/renyixiang/SkinDB/10X/human/Annotation/Annotated/human_with_all_leiden_10threshold_annotated_updated.h5ad",
        "out_root": "/gpfsdata/home/renyixiang/SkinDB/10X/human/data4viz/regulon/SCORPION",
        "valid_conditions": ["Healthy", "psoriasis"],
    },
    {
        "species": "mouse",
        "adata_path": "/gpfsdata/home/renyixiang/SkinDB/10X/mouse/Annotation/Annotated/mouse_with_all_leiden_10threshold_annotated_updated.h5ad",
        "out_root": "/gpfsdata/home/renyixiang/SkinDB/10X/mouse/data4viz/regulon/SCORPION",
        "valid_conditions": ["Healthy", "IMQ-induced psoriasis"],
    },
]

TARGET_CELLTYPES = ["Immune", "Keratinocyte"]
REQUIRED_OBS = ["sample", "condition", "Gross_Map"]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_lines(path: str, lines):
    with open(path, "w") as f:
        for x in lines:
            f.write(f"{x}\n")


def main():
    for cfg in CONFIGS:
        species = cfg["species"]
        adata_path = cfg["adata_path"]
        out_root = cfg["out_root"]
        valid_conditions = set(cfg["valid_conditions"])

        print(f"\n===== Processing {species} =====")
        print(f"Reading: {adata_path}")
        adata = sc.read_h5ad(adata_path)

        for col in REQUIRED_OBS:
            if col not in adata.obs.columns:
                raise ValueError(f"[{species}] Missing obs column: {col}")

        if "counts" not in adata.layers.keys():
            raise ValueError(f"[{species}] adata.layers['counts'] not found")

        keep_mask = (
            adata.obs["Gross_Map"].isin(TARGET_CELLTYPES)
            & adata.obs["condition"].isin(valid_conditions)
        )

        adata = adata[keep_mask].copy()
        print(f"[{species}] after global filter: {adata.shape}")

        summary_root = os.path.join(out_root, "input")
        ensure_dir(summary_root)

        # global summary
        global_summary = (
            adata.obs.groupby(["Gross_Map", "condition", "sample"])
            .size()
            .reset_index(name="n_cells")
            .sort_values(["Gross_Map", "condition", "sample"])
        )
        global_summary.to_csv(
            os.path.join(summary_root, "global_cellcount_summary.tsv"),
            sep="\t",
            index=False,
        )

        # write config snapshot
        with open(os.path.join(summary_root, "export_config.json"), "w") as f:
            json.dump(
                {
                    "species": species,
                    "adata_path": adata_path,
                    "target_celltypes": TARGET_CELLTYPES,
                    "valid_conditions": list(valid_conditions),
                    "matrix_source": "adata.layers['counts']",
                },
                f,
                indent=2,
            )

        for celltype in TARGET_CELLTYPES:
            print(f"\n[{species}] Exporting cell type: {celltype}")
            sub = adata[adata.obs["Gross_Map"] == celltype].copy()
            print(f"[{species}][{celltype}] subset shape: {sub.shape}")

            if sub.n_obs == 0:
                print(f"[{species}][{celltype}] skipped: no cells")
                continue

            counts = sub.layers["counts"]
            if not sp.issparse(counts):
                counts = sp.csr_matrix(counts)
            else:
                counts = counts.tocsr()

            # remove zero-expression genes in this subset
            gene_nonzero = np.asarray(counts.sum(axis=0)).ravel() > 0
            sub = sub[:, gene_nonzero].copy()
            counts = sub.layers["counts"]
            if not sp.issparse(counts):
                counts = sp.csr_matrix(counts)
            else:
                counts = counts.tocsr()

            # SCORPION expects genes x cells
            gex = counts.T.tocoo()

            cell_dir = os.path.join(out_root, "input", celltype)
            ensure_dir(cell_dir)

            mtx_path = os.path.join(cell_dir, "gex_genes_x_cells.mtx")
            genes_path = os.path.join(cell_dir, "genes.tsv")
            cells_path = os.path.join(cell_dir, "cells.tsv")
            meta_path = os.path.join(cell_dir, "metadata.tsv")
            summary_path = os.path.join(cell_dir, "cellcount_summary.tsv")

            print(f"[{species}][{celltype}] writing matrix: {mtx_path}")
            mmwrite(mtx_path, gex)

            write_lines(genes_path, sub.var_names.astype(str).tolist())
            write_lines(cells_path, sub.obs_names.astype(str).tolist())

            metadata = sub.obs.copy()
            metadata = metadata.loc[:, ["sample", "condition", "Gross_Map"]].copy()
            metadata.insert(0, "cell_barcode", sub.obs_names.astype(str).tolist())
            metadata.to_csv(meta_path, sep="\t", index=False)

            summary = (
                metadata.groupby(["Gross_Map", "condition", "sample"])
                .size()
                .reset_index(name="n_cells")
                .sort_values(["condition", "sample"])
            )
            summary.to_csv(summary_path, sep="\t", index=False)

            print(f"[{species}][{celltype}] done: {gex.shape[0]} genes x {gex.shape[1]} cells")

    print("\nAll exports finished.")


if __name__ == "__main__":
    main()
