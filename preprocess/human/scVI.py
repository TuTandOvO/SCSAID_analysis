import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import anndata as ad
import scvi
import scanpy as sc
import os
import torch
seed=10
sc.logging.print_versions()

adata = scvi.data.read_h5ad("/gpfsdata/home/renyixiang/SkinDB/10X/human/Integration/Human_merged_before_int8000HVG_V3.h5ad")
adata.obs_names_make_unique

torch.set_float32_matmul_precision("medium")
scvi.settings.dl_num_workers = 8
scvi.model.SCVI.setup_anndata(
    adata, 
    batch_key="batch",
    layer="counts", categorical_covariate_keys=["GSE"], 
    continuous_covariate_keys=["pct_counts_mt"])
model = scvi.model.SCVI(adata)
model
vae = scvi.model.SCVI(adata, n_layers=2, n_latent=30, gene_likelihood="nb", dropout_rate=0.1)
vae.train(max_epochs = 600, plan_kwargs={"lr":0.001}, early_stopping = True, early_stopping_patience = 15)
model = vae

model.save("/gpfsdata/home/renyixiang/SkinDB/10X/human/Integration/Human_Model_V3")
