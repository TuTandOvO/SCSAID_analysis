"""Paths + constants for SkinDB psoriasis target validation pipeline."""

from pathlib import Path

# ---- project root ---------------------------------------------------------
PROJECT_ROOT = Path("/gpfsdata/home/renyixiang/SkinDB_FM")

# ---- input ----------------------------------------------------------------
CANDIDATE_CSV = PROJECT_ROOT / "Candidate_gene.csv"

# Atlas h5ad (raw counts + cleaned obs applied; per CLAUDE.md §5.1.1).
# X is raw counts (int), var.index = gene symbol (full gene set).
HUMAN_H5AD = PROJECT_ROOT / "human_with_clean_obs.h5ad"

# Geneformer weights + dicts (git clone ctheodoris/Geneformer)
GENEFORMER_ROOT  = PROJECT_ROOT / "models" / "Geneformer"
GF_V2_MODEL_DIR  = GENEFORMER_ROOT / "Geneformer-V2-104M"    # V2 standard
GF_TOKEN_DICT    = GENEFORMER_ROOT / "geneformer" / "token_dictionary_gc104M.pkl"
GF_GENE_MEDIAN   = GENEFORMER_ROOT / "geneformer" / "gene_median_dictionary_gc104M.pkl"
GF_ENSEMBL_MAP   = GENEFORMER_ROOT / "geneformer" / "ensembl_mapping_dict_gc104M.pkl"
GF_SYMBOL_TO_ENS = GENEFORMER_ROOT / "geneformer" / "gene_name_id_dict_gc104M.pkl"  # symbol → ensembl

# LINCS L1000 Phase I (GSE92742) — downloaded to data/l1000/
L1000_DATA_DIR = PROJECT_ROOT / "data" / "l1000"
L1000_GCTX     = L1000_DATA_DIR / "GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx"
L1000_SIG_INFO = L1000_DATA_DIR / "GSE92742_Broad_LINCS_sig_info.txt"
L1000_GENE_INFO= L1000_DATA_DIR / "GSE92742_Broad_LINCS_gene_info.txt"

# ---- output ---------------------------------------------------------------
RESULTS_DIR    = PROJECT_ROOT / "results"
CANDIDATES_OUT = RESULTS_DIR / "candidates"          # 01
ADATA_OUT      = RESULTS_DIR / "adata"               # 02
DEG_OUT        = RESULTS_DIR / "deg"                 # 03
TOKEN_OUT      = RESULTS_DIR / "tokenized"           # 04
PERTURB_OUT    = RESULTS_DIR / "perturbation"        # 05/06
SHIFT_OUT      = RESULTS_DIR / "shift_scores"        # 07
L1000_OUT      = RESULTS_DIR / "l1000"               # 08
CONSENSUS_OUT  = RESULTS_DIR / "consensus"           # 09
FIGURES_DIR    = PROJECT_ROOT / "figures"            # 10
LOGS_DIR       = PROJECT_ROOT / "logs"

for d in [RESULTS_DIR, CANDIDATES_OUT, ADATA_OUT, DEG_OUT, TOKEN_OUT,
          PERTURB_OUT, SHIFT_OUT, L1000_OUT, CONSENSUS_OUT, FIGURES_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---- analysis parameters --------------------------------------------------
# Disease cohort: STRICT to match upstream mathematical modelling, which
# uses only active Psoriasis (excluding Resolved Psoriatic Lesion). This
# keeps validation cohort consistent with candidate-derivation cohort.
# condition filters (must match cleaned obs values)
# Strict: Psoriasis only (matches upstream math-modelling cohort)
# (Resolved Psoriatic Lesion was previously included as a sensitivity setting,
#  but is now excluded by default for cohort consistency.)
PSO_CONDITIONS     = ["Psoriasis"]
HEALTHY_CONDITIONS = ["Healthy"]

# Focal cell types (Gross_Map values actually present in atlas with DEG):
# Keratinocyte, Fibroblast, Immune, Vascular_Endothelial_Cell,
# Melanocyte, Smooth_Muscle_Cell, Eccrine_Sweat_Gland_Cell
# No separate T cell / Macrophage / DC — all lumped in 'Immune' at Gross level.
# Scripts auto-discover from results/tokenized/ and results/deg/, no need to set here.

# Geneformer
GF_MAX_LEN     = 4096      # V2 supports up to 4096
GF_BATCH_SIZE  = 8         # V100-32GB fp16; tune after OOM check
GF_PRECISION   = "fp16"    # V100 does NOT support bf16
GF_EMB_LAYER   = -1        # last hidden layer for cell embedding

# Per-gene KO perturbation
GF_MAX_NCELLS   = 2000     # cells per gene for InSilicoPerturber (seed=42 fixed)
RANDOM_SEED     = 42

# L1000 (GSE92742 Phase I has no trt_xpr; we use trt_sh.cgs as primary KD)
L1000_PERT_TYPES         = ["trt_sh.cgs", "trt_sh", "trt_cp"]
L1000_TOP_N_GENES_QUERY  = 150   # top up/down DEGs → query signature

# Consensus
RRA_TOP_QUANTILE = 0.25    # "top 25%" per cell type threshold
