import os
import re
import gc
import json
import time
import logging
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd
import h5py
import zarr
import numcodecs
import scanpy as sc
from scipy import sparse

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    classification_report,
    roc_curve,
    precision_recall_curve,
)

# =========================================================
# USER EDIT BLOCK
# =========================================================

# -------------------------
# HUMAN paths
# -------------------------
HUMAN_H5AD_PATH = r"/gpfsdata/home/renyixiang/SkinDB/10X/human/Annotation/Annotated/human_with_all_leiden_10threshold_annotated_updated.h5ad"
HUMAN_STRIPPED_H5AD_PATH = r"/gpfsdata/home/renyixiang/SkinDB/10X/human/model/human_model.stripped.h5ad"
HUMAN_ZARR_PATH = r"/gpfsdata/home/renyixiang/SkinDB/10X/human/model/human_universal.zarr"

# -------------------------
# MOUSE paths
# -------------------------
MOUSE_H5AD_PATH = r"/gpfsdata/home/renyixiang/SkinDB/10X/mouse/Annotation/Annotated/mouse_with_all_leiden_10threshold_annotated_updated.h5ad"
MOUSE_STRIPPED_H5AD_PATH = r"/gpfsdata/home/renyixiang/SkinDB/10X/mouse/model/mouse_model.stripped.h5ad"
MOUSE_ZARR_PATH = r"/gpfsdata/home/renyixiang/SkinDB/10X/mouse/model/mouse_universal.zarr"

# -------------------------
# ortholog + output
# -------------------------
ORTHOLOG_TSV = r"/gpfsdata/home/renyixiang/SkinDB/cross_species/human_mouse_1to1_ortholog.tsv"
WORKDIR = r"/gpfsdata/home/renyixiang/SkinDB/cross_species/human_like_zarr_ortholog_mouse_validate"

# -------------------------
# HUMAN obs columns
# -------------------------
HUMAN_AGE_COL = "Age"
HUMAN_SEX_COL = "sex"
HUMAN_COND_COL = "condition"

# -------------------------
# MOUSE obs columns
# -------------------------
MOUSE_AGE_COL = "Age"
MOUSE_SEX_COL = "sex"
MOUSE_COND_COL = "condition"

# -------------------------
# condition mapping
# -------------------------
HUMAN_COND_MAP = {"Healthy": 0, "psoriasis": 1}
MOUSE_COND_MAP = {"Healthy": 0, "IMQ-induced psoriasis": 1}

HUMAN_COND_ALIASES = {
}
MOUSE_COND_ALIASES = {
}

# =========================================================
# Env
# =========================================================
os.environ.setdefault("OMP_NUM_THREADS", "10")
os.environ.setdefault("MKL_NUM_THREADS", "10")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "10")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "10")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# CONFIG
# =========================================================

# build behavior
BUILD_MISSING_STRIPPED = True
BUILD_MISSING_ZARR = True
NEVER_OVERWRITE_ZARR = True

EXPORT_ROC_DATA = True
ROC_EXPORT_MODE = "panel_only"   # "panel_only" | "all_pruned"

# base zarr build
TARGET_SUM = 1e4
CHUNK_CELLS = 20_000

MIN_GENE_COUNTS = 1
MIN_GENE_CELLS_FRAC = 0.01

# task training
TEST_SIZE = 0.2
RANDOM_STATE = 42
USE_GROUP_SPLIT = True

GROUP_COL_CANDIDATES_HUMAN = [
    "human_id", "Human", "human",
    "sample", "Sample", "orig.ident",
    "donor", "Donor",
    "batch", "Batch",
    "lane", "Lane",
]

GROUP_COL_CANDIDATES_MOUSE = [
    "mouse_id", "Mouse", "mouse",
    "sample", "Sample", "orig.ident",
    "donor", "Donor",
    "batch", "Batch",
    "lane", "Lane",
]

# corr pruning
MAX_CORR_CELLS = 15000
CORR_BLOCK_ROWS = 3000
PRUNE_ABS_CORR = 0.90
MIN_PRUNED_GENES = 200

# stability selection
N_REPEATS = 30
SUBSAMPLE_FRAC_GROUPS = 0.75
C_GRID = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
L1_RATIO_SEL = 0.9
TARGET_NNZ_FOR_SELECTION = 250

# final panel size
PANEL_K = 50

# final model
FINAL_C = 1.0
FINAL_SOLVER = "saga"
FINAL_L1_RATIO = 0.0
Z_CLIP = 5.0
DROP_MT_RP = True
EXT_CHUNK = 20000

# supervised effect score epsilon
CAND_EPS = 1e-8

# unified cross-species age stages for final model
AGE_STAGE_COLS = ["Age_juvenile", "Age_young_adult", "Age_adult", "Age_aged"]

# outputs
os.makedirs(WORKDIR, exist_ok=True)
OUT_PRUNED_GENES = os.path.join(WORKDIR, "genes_pruned.txt")
OUT_PRUNING_MAP = os.path.join(WORKDIR, "pruning_map.tsv")
OUT_STABILITY = os.path.join(WORKDIR, "stability_selection.tsv")
OUT_PANEL = os.path.join(WORKDIR, "panel_genes.tsv")
OUT_METRICS_HUMAN = os.path.join(WORKDIR, "human_internal_metrics.json")
OUT_METRICS_MOUSE = os.path.join(WORKDIR, "mouse_external_metrics.json")
OUT_REPORT_HUMAN = os.path.join(WORKDIR, "human_internal_classification_report.txt")
OUT_REPORT_MOUSE = os.path.join(WORKDIR, "mouse_external_classification_report.txt")
OUT_BUNDLE = os.path.join(WORKDIR, "cross_species_bundle.json")
OUT_MOUSE_SCORES = os.path.join(WORKDIR, "mouse_external_scores.tsv")
OUT_HUMAN_PANEL_STATS = os.path.join(WORKDIR, "human_panel_stats.tsv")
OUT_C_GRID = os.path.join(WORKDIR, "c_grid_selection.tsv")
OUT_FINAL_COEF = os.path.join(WORKDIR, "final_coefficients.tsv")
OUT_CANDIDATE_SCORES = os.path.join(WORKDIR, "candidate_scores.tsv")
XMM_PATH = os.path.join(WORKDIR, "X_pruned_genes_float32.memmap")


# =========================================================
# logging
# =========================================================
def setup_logging(workdir: str):
    log_path = os.path.join(workdir, "pipeline.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )


@contextmanager
def log_step(name: str):
    t0 = time.time()
    logging.info(f"========== {name} | START ==========")
    yield
    dt = time.time() - t0
    logging.info(f"========== {name} | DONE | {dt:.2f}s ==========")


# =========================================================
# utils
# =========================================================
def to_pystr_array(x):
    if hasattr(x, "astype") and hasattr(x, "map") and hasattr(x, "to_numpy"):
        s = x.astype("object")
        return s.map(
            lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
        ).to_numpy(dtype=object)
    arr = np.asarray(x, dtype="object")
    return np.array(
        ["" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v) for v in arr],
        dtype=object,
    )


def iter_chunks(idx: np.ndarray, chunk_size: int):
    for s in range(0, idx.size, chunk_size):
        yield idx[s:s + chunk_size]


def iter_ranges(n: int, step: int):
    for s in range(0, n, step):
        e = min(n, s + step)
        yield s, e


def print_counts(name, y):
    vc = pd.Series(y).value_counts().to_dict()
    vc = {int(k): int(v) for k, v in vc.items()}
    logging.info(f"[INFO] {name} counts: {vc}")


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_curves(y_true, y_score, prefix):
    fpr, tpr, thr_roc = roc_curve(y_true, y_score)
    precision, recall, thr_pr = precision_recall_curve(y_true, y_score)

    pd.DataFrame({
        "fpr": fpr,
        "tpr": tpr,
        "threshold": np.r_[thr_roc, np.nan][:len(fpr)],
    }).to_csv(prefix + "_roc.tsv", sep="\t", index=False)

    pd.DataFrame({
        "precision": precision,
        "recall": recall,
        "threshold": np.r_[thr_pr, np.nan][:len(precision)],
    }).to_csv(prefix + "_pr.tsv", sep="\t", index=False)


def normalize_condition_series(raw_series: pd.Series, alias_map: dict) -> pd.Series:
    s = raw_series.astype(str).str.strip()
    if alias_map:
        s = s.map(lambda x: alias_map.get(x, x))
    return s


# =========================================================
# HUMAN age parsing
# =========================================================
def parse_human_age(age):
    if age is None:
        return None
    s = str(age).strip()
    if s == "" or s.lower() in ["nan", "none"]:
        return None
    try:
        val = float(s)
        if val < 0:
            return None
        return val
    except Exception:
        return None


def human_age_to_stage(age_years):
    if age_years is None:
        return None
    if age_years < 18:
        return "Age_juvenile"
    if age_years < 40:
        return "Age_young_adult"
    if age_years < 65:
        return "Age_adult"
    return "Age_aged"


# =========================================================
# MOUSE age parsing
# =========================================================
P_RE = re.compile(r"^P\s*([0-9]+)", re.IGNORECASE)
E_RE = re.compile(r"^E", re.IGNORECASE)

def parse_postnatal_days(age):
    if age is None:
        return None
    s = str(age).strip()
    if s == "" or s.lower() in ["nan", "none"]:
        return None
    if E_RE.match(s):
        return None
    m = P_RE.match(s)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def mouse_age_to_stage(d):
    if d is None or d < 0:
        return None
    weeks = d / 7.0
    months = d / 30.4375
    years = d / 365.25
    if weeks < 6:
        return "Age_juvenile"
    if weeks >= 6 and months < 6:
        return "Age_young_adult"
    if months >= 6 and years < 1:
        return "Age_adult"
    return "Age_aged"


def encode_sex(x):
    if x is None:
        return 0.5
    s = str(x).strip().lower()
    if s in ["male", "m", "♂", "1"]:
        return 1.0
    if s in ["female", "f", "♀", "0"]:
        return 0.0
    return 0.5


# =========================================================
# gene filters
# =========================================================
_MT_RE_H = re.compile(r"^MT-", re.IGNORECASE)
_MT_RE_M = re.compile(r"^mt-", re.IGNORECASE)
_RP_RE = re.compile(r"^(RPS|RPL|Mrps|Mrpl|Rps|Rpl)", re.IGNORECASE)

def keep_gene_pair(hg: str, mg: str) -> bool:
    if hg is None or mg is None:
        return False
    hg = str(hg)
    mg = str(mg)
    if hg == "" or mg == "":
        return False
    if DROP_MT_RP:
        if _MT_RE_H.match(hg) or _MT_RE_M.match(mg):
            return False
        if _RP_RE.match(hg) or _RP_RE.match(mg):
            return False
    return True


# =========================================================
# Step0: strip h5ad
# =========================================================
def strip_h5ad_to_x_obs_var(in_h5ad: str, out_h5ad: str):
    if os.path.exists(out_h5ad):
        logging.info(f"[INFO] Detected existing stripped h5ad, skip: {out_h5ad}")
        return
    if not BUILD_MISSING_STRIPPED:
        raise FileNotFoundError(f"Missing stripped h5ad and BUILD_MISSING_STRIPPED=False: {out_h5ad}")

    logging.info(f"[INFO] Creating stripped h5ad (X/obs/var only) -> {out_h5ad}")
    with h5py.File(in_h5ad, "r") as fin, h5py.File(out_h5ad, "w") as fout:
        for k, v in fin.attrs.items():
            fout.attrs[k] = v
        root_keys = list(fin.keys())
        logging.info(f"[INFO] Input h5ad root keys: {root_keys}")
        for key in ("X", "obs", "var"):
            if key not in fin:
                raise KeyError(f"Key '{key}' not found in input h5ad. Root keys: {root_keys}")
            fin.copy(key, fout, name=key)
    logging.info("[INFO] Stripped h5ad created.")


# =========================================================
# base kept cells
# =========================================================
def build_base_kept_from_stripped_human(stripped_h5ad: str):
    adata_b = sc.read_h5ad(stripped_h5ad, backed="r")
    obs = adata_b.obs

    if HUMAN_AGE_COL not in obs.columns:
        raise KeyError(f"Missing '{HUMAN_AGE_COL}' in obs.")
    if HUMAN_SEX_COL not in obs.columns:
        raise KeyError(f"Missing '{HUMAN_SEX_COL}' in obs.")
    if HUMAN_COND_COL not in obs.columns:
        logging.warning(f"Missing '{HUMAN_COND_COL}' in obs. Condition task later may fail.")

    age_raw = pd.Series(to_pystr_array(obs[HUMAN_AGE_COL]))
    age_years = age_raw.apply(parse_human_age)
    age_stage = age_years.apply(human_age_to_stage)
    mask_age = age_stage.notna()

    base_orig_idx = np.where(mask_age.to_numpy())[0].astype(np.int64)
    age_stage_base = age_stage[mask_age].reset_index(drop=True)
    sex_base = obs.iloc[base_orig_idx][HUMAN_SEX_COL].astype("object").reset_index(drop=True)

    group_col = None
    for c in GROUP_COL_CANDIDATES_HUMAN:
        if c in obs.columns:
            group_col = c
            break

    if group_col is not None:
        groups_base = obs.iloc[base_orig_idx][group_col].astype("object").reset_index(drop=True)
        logging.info(f"[INFO] Human base group column: {group_col} (unique={groups_base.nunique()})")
    else:
        groups_base = None
        logging.warning("[WARN] No human group column found in obs.")

    logging.info(f"[INFO] Human base kept cells (age-parsed): {base_orig_idx.size} / {adata_b.n_obs}")

    try:
        adata_b.file.close()
    except Exception:
        pass

    return base_orig_idx, age_stage_base, sex_base, groups_base, group_col


def build_base_kept_from_stripped_mouse(stripped_h5ad: str):
    adata_b = sc.read_h5ad(stripped_h5ad, backed="r")
    obs = adata_b.obs

    if MOUSE_AGE_COL not in obs.columns:
        raise KeyError(f"Missing '{MOUSE_AGE_COL}' in obs.")
    if MOUSE_SEX_COL not in obs.columns:
        raise KeyError(f"Missing '{MOUSE_SEX_COL}' in obs.")
    if MOUSE_COND_COL not in obs.columns:
        logging.warning(f"Missing '{MOUSE_COND_COL}' in obs. Condition task later may fail.")

    age_raw = pd.Series(to_pystr_array(obs[MOUSE_AGE_COL]))
    p_days = age_raw.apply(parse_postnatal_days)
    age_stage = p_days.apply(mouse_age_to_stage)
    mask_age = age_stage.notna()

    base_orig_idx = np.where(mask_age.to_numpy())[0].astype(np.int64)
    age_stage_base = age_stage[mask_age].reset_index(drop=True)
    sex_base = obs.iloc[base_orig_idx][MOUSE_SEX_COL].astype("object").reset_index(drop=True)

    group_col = None
    for c in GROUP_COL_CANDIDATES_MOUSE:
        if c in obs.columns:
            group_col = c
            break

    if group_col is not None:
        groups_base = obs.iloc[base_orig_idx][group_col].astype("object").reset_index(drop=True)
        logging.info(f"[INFO] Mouse base group column: {group_col} (unique={groups_base.nunique()})")
    else:
        groups_base = None
        logging.warning("[WARN] No mouse group column found in obs.")

    logging.info(f"[INFO] Mouse base kept cells (age-parsed): {base_orig_idx.size} / {adata_b.n_obs}")

    try:
        adata_b.file.close()
    except Exception:
        pass

    return base_orig_idx, age_stage_base, sex_base, groups_base, group_col


# =========================================================
# gene stats on base cells
# =========================================================
def compute_gene_stats_on_rows(stripped_h5ad: str, orig_rows: np.ndarray, chunk_cells: int):
    adata_b = sc.read_h5ad(stripped_h5ad, backed="r")
    n_genes = adata_b.n_vars
    gene_counts = np.zeros(n_genes, dtype=np.float64)
    gene_ncells = np.zeros(n_genes, dtype=np.int64)

    for chunk_idx in iter_chunks(orig_rows, chunk_cells):
        X = adata_b.X[chunk_idx, :]
        if not sparse.issparse(X):
            X = sparse.csr_matrix(X)
        X = X.tocsr()
        gene_counts += np.asarray(X.sum(axis=0)).ravel()
        gene_ncells += np.asarray((X > 0).sum(axis=0)).ravel().astype(np.int64)

    try:
        adata_b.file.close()
    except Exception:
        pass

    return gene_counts, gene_ncells


# =========================================================
# Build universal zarr/log1p CSR if missing
# =========================================================
def _normalize_log1p_csr(X_csr: sparse.csr_matrix, target_sum: float) -> sparse.csr_matrix:
    X_csr = X_csr.tocsr()
    lib = np.asarray(X_csr.sum(axis=1)).ravel().astype(np.float64)
    lib[lib <= 0] = 1.0
    scale = (float(target_sum) / lib).astype(np.float64)
    X_norm = X_csr.multiply(scale[:, None]).tocsr()
    X_norm.data = np.log1p(X_norm.data).astype(np.float32, copy=False)
    return X_norm


def build_universal_zarr_if_missing(
    stripped_h5ad: str,
    zarr_path: str,
    base_orig_idx: np.ndarray,
    gene_mask: np.ndarray,
    target_sum: float,
    chunk_cells: int,
):
    if os.path.exists(zarr_path):
        root = zarr.open_group(zarr_path, mode="r")
        if "layers" in root and "log1p" in root["layers"]:
            logging.info("[INFO] Detected existing zarr/layers/log1p, reuse read-only.")
            return
        if NEVER_OVERWRITE_ZARR:
            raise RuntimeError(
                f"Zarr exists but layers/log1p missing and NEVER_OVERWRITE_ZARR=True: {zarr_path}"
            )
        raise RuntimeError("Overwriting existing zarr is not supported in this script for safety.")

    if not BUILD_MISSING_ZARR:
        raise FileNotFoundError(f"Missing zarr and BUILD_MISSING_ZARR=False: {zarr_path}")

    logging.info(f"[INFO] Building universal zarr -> {zarr_path}")
    logging.info(f"[INFO] Base rows: {base_orig_idx.size}, gene_mask kept genes: {int(gene_mask.sum())}")

    adata_b = sc.read_h5ad(stripped_h5ad, backed="r")
    var_names_full = to_pystr_array(adata_b.var_names)
    var_names_kept = var_names_full[gene_mask]
    n_cells = int(base_orig_idx.size)
    n_genes_kept = int(gene_mask.sum())
    try:
        adata_b.file.close()
    except Exception:
        pass

    indptr = np.zeros(n_cells + 1, dtype=np.int64)
    nnz_total = 0

    adata_b = sc.read_h5ad(stripped_h5ad, backed="r")
    for bi, chunk_orig in enumerate(iter_chunks(base_orig_idx, chunk_cells)):
        X = adata_b.X[chunk_orig, :]
        if not sparse.issparse(X):
            X = sparse.csr_matrix(X)
        X = X.tocsr()

        Xn = _normalize_log1p_csr(X, target_sum=target_sum)
        Xk = Xn[:, gene_mask].tocsr()
        nnz_row = np.diff(Xk.indptr).astype(np.int64)

        start = bi * chunk_cells
        end = start + nnz_row.size
        indptr[start + 1:end + 1] = nnz_row
        nnz_total += int(nnz_row.sum())

        del X, Xn, Xk
        gc.collect()

    try:
        adata_b.file.close()
    except Exception:
        pass

    indptr = np.cumsum(indptr, dtype=np.int64)
    if int(indptr[-1]) != int(nnz_total):
        raise RuntimeError("NNZ mismatch in pass1.")

    logging.info(f"[INFO] Universal log1p CSR nnz_total={nnz_total}")

    root = zarr.open_group(zarr_path, mode="w-")
    root.create_dataset(
        "obs_orig_index",
        data=base_orig_idx.astype(np.int64),
        chunks=(min(n_cells, 200000),),
        dtype="i8",
    )

    vg = root.create_group("var")
    vg.create_dataset(
        "_index",
        data=var_names_kept.astype(object),
        chunks=(min(n_genes_kept, 200000),),
        dtype=object,
        object_codec=numcodecs.VLenUTF8(),
    )

    layers = root.create_group("layers")
    lg = layers.create_group("log1p")
    lg.attrs["shape"] = [n_cells, n_genes_kept]

    lg.create_dataset("indptr", data=indptr, chunks=(min(indptr.size, 200000),), dtype="i8")
    lg.create_dataset("indices", shape=(nnz_total,), chunks=(min(nnz_total, 5_000_000),), dtype="i4")
    lg.create_dataset("data", shape=(nnz_total,), chunks=(min(nnz_total, 5_000_000),), dtype="f4")

    write_ptr = 0
    adata_b = sc.read_h5ad(stripped_h5ad, backed="r")
    for chunk_orig in iter_chunks(base_orig_idx, chunk_cells):
        X = adata_b.X[chunk_orig, :]
        if not sparse.issparse(X):
            X = sparse.csr_matrix(X)
        X = X.tocsr()
        Xn = _normalize_log1p_csr(X, target_sum=target_sum)
        Xk = Xn[:, gene_mask].tocsr()

        nnz = int(Xk.nnz)
        if nnz > 0:
            lg["indices"][write_ptr:write_ptr + nnz] = Xk.indices.astype(np.int32, copy=False)
            lg["data"][write_ptr:write_ptr + nnz] = Xk.data.astype(np.float32, copy=False)
            write_ptr += nnz

        del X, Xn, Xk
        gc.collect()

    try:
        adata_b.file.close()
    except Exception:
        pass

    if write_ptr != nnz_total:
        raise RuntimeError(f"Pass2 write_ptr mismatch: {write_ptr} vs {nnz_total}")

    root.attrs["target_sum"] = float(target_sum)
    root.attrs["base_keep_rule"] = "age_parsed_only"
    root.attrs["min_gene_counts"] = int(MIN_GENE_COUNTS)
    root.attrs["min_gene_cells_frac"] = float(MIN_GENE_CELLS_FRAC)

    logging.info("[INFO] Universal zarr build complete.")


# =========================================================
# open universal CSR + var names
# =========================================================
def open_log1p_csr(zarr_path: str):
    root = zarr.open_group(zarr_path, mode="r")
    if "layers" not in root or "log1p" not in root["layers"]:
        raise RuntimeError("Missing zarr/layers/log1p.")
    lg = root["layers"]["log1p"]
    shp = lg.attrs.get("shape", None)
    if shp is None:
        raise RuntimeError("Cannot read log1p shape.")
    n_cells, n_genes = int(shp[0]), int(shp[1])
    return root, lg, n_cells, n_genes


def load_var_names(zroot):
    if "var" in zroot and "_index" in zroot["var"]:
        return to_pystr_array(zroot["var"]["_index"][:])
    raise KeyError("Cannot find var/_index in zarr.")


def load_obs_orig_index(zroot):
    if "obs_orig_index" in zroot:
        return np.asarray(zroot["obs_orig_index"][:], dtype=np.int64)
    raise KeyError("Cannot find obs_orig_index in zarr.")


# =========================================================
# task rows from universal base
# =========================================================
def build_task_from_universal_base_human(
    stripped_h5ad: str,
    base_orig_idx: np.ndarray,
    age_stage_base,
    sex_base,
    groups_base,
):
    adata_b = sc.read_h5ad(stripped_h5ad, backed="r")
    obs = adata_b.obs

    if HUMAN_COND_COL not in obs.columns:
        raise KeyError(f"Missing '{HUMAN_COND_COL}' in obs for task.")

    cond = pd.Series(to_pystr_array(obs.iloc[base_orig_idx][HUMAN_COND_COL]))
    cond = normalize_condition_series(cond, HUMAN_COND_ALIASES)
    y_series = cond.map(HUMAN_COND_MAP)
    mask_cond = ~y_series.isna()

    task_rows = np.where(mask_cond.to_numpy())[0].astype(np.int64)
    y = y_series[mask_cond].astype(np.int8).to_numpy()

    age_stage_task = age_stage_base[mask_cond].reset_index(drop=True)
    sex_task = sex_base[mask_cond].reset_index(drop=True)
    groups_task = groups_base[mask_cond].reset_index(drop=True) if groups_base is not None else None

    try:
        adata_b.file.close()
    except Exception:
        pass

    logging.info(f"[INFO] Human task cells: {task_rows.size} / {base_orig_idx.size}")
    print_counts("human_y_task", y)

    return task_rows, y, age_stage_task, sex_task, groups_task


def build_task_from_universal_base_mouse(
    stripped_h5ad: str,
    base_orig_idx: np.ndarray,
    age_stage_base,
    sex_base,
    groups_base,
):
    adata_b = sc.read_h5ad(stripped_h5ad, backed="r")
    obs = adata_b.obs

    if MOUSE_COND_COL not in obs.columns:
        raise KeyError(f"Missing '{MOUSE_COND_COL}' in obs for task.")

    cond = pd.Series(to_pystr_array(obs.iloc[base_orig_idx][MOUSE_COND_COL]))
    cond = normalize_condition_series(cond, MOUSE_COND_ALIASES)
    y_series = cond.map(MOUSE_COND_MAP)
    mask_cond = ~y_series.isna()

    task_rows = np.where(mask_cond.to_numpy())[0].astype(np.int64)
    y = y_series[mask_cond].astype(np.int8).to_numpy()

    age_stage_task = age_stage_base[mask_cond].reset_index(drop=True)
    sex_task = sex_base[mask_cond].reset_index(drop=True)
    groups_task = groups_base[mask_cond].reset_index(drop=True) if groups_base is not None else None

    try:
        adata_b.file.close()
    except Exception:
        pass

    logging.info(f"[INFO] Mouse task cells: {task_rows.size} / {base_orig_idx.size}")
    print_counts("mouse_y_task", y)

    return task_rows, y, age_stage_task, sex_task, groups_task


def build_covariates(age_stage: pd.Series, sex_series: pd.Series):
    age_dum = pd.get_dummies(age_stage, prefix="", prefix_sep="")
    for c in AGE_STAGE_COLS:
        if c not in age_dum.columns:
            age_dum[c] = 0
    age_dum = age_dum[AGE_STAGE_COLS]
    age_mat = age_dum.values.astype(np.float32, copy=False)
    sex_num = sex_series.apply(encode_sex).to_numpy(dtype=np.float32)
    return age_mat, sex_num


# =========================================================
# split helpers
# =========================================================
def stratified_group_split_local(y, groups, test_size=0.2, random_state=42, max_tries=300):
    y = np.asarray(y, dtype=np.int8)
    g = np.asarray(groups, dtype=object)
    idx_all = np.arange(y.size, dtype=np.int64)

    df = pd.DataFrame({"i": idx_all, "y": y, "g": g})

    nu = df.groupby("g")["y"].nunique()
    bad = nu[nu > 1]
    if len(bad) > 0:
        raise RuntimeError(f"Found mixed-label groups (nunique>1). Examples: {bad.index.tolist()[:10]}")

    g2y = df.groupby("g")["y"].first()
    g_list = g2y.index.to_numpy(dtype=object)
    g_y = g2y.to_numpy(dtype=np.int8)

    g0 = g_list[g_y == 0]
    g1 = g_list[g_y == 1]
    if g0.size == 0 or g1.size == 0:
        raise RuntimeError("Only one class exists at group level.")

    n_test0 = max(1, int(round(test_size * g0.size)))
    n_test1 = max(1, int(round(test_size * g1.size)))
    if g0.size > 1:
        n_test0 = min(n_test0, g0.size - 1)
    if g1.size > 1:
        n_test1 = min(n_test1, g1.size - 1)

    rng = np.random.RandomState(random_state)
    for t in range(1, max_tries + 1):
        te_g0 = rng.choice(g0, size=n_test0, replace=False)
        te_g1 = rng.choice(g1, size=n_test1, replace=False)
        te_groups = set(te_g0.tolist() + te_g1.tolist())

        te_mask = df["g"].isin(te_groups).to_numpy()
        te = df.loc[te_mask, "i"].to_numpy(dtype=np.int64)
        tr = df.loc[~te_mask, "i"].to_numpy(dtype=np.int64)

        if (np.unique(y[tr]).size == 2) and (np.unique(y[te]).size == 2):
            logging.info(f"[INFO] StratifiedGroupSplit success on try {t}")
            print_counts("y_train", y[tr])
            print_counts("y_test", y[te])
            return tr, te

    raise RuntimeError("Failed to find valid stratified group split.")


def stratified_random_split_local(y, test_size=0.2, random_state=42):
    y = np.asarray(y, dtype=np.int8)
    idx = np.arange(y.size, dtype=np.int64)
    rng = np.random.RandomState(random_state)

    i0 = idx[y == 0]
    i1 = idx[y == 1]
    rng.shuffle(i0)
    rng.shuffle(i1)

    nte0 = max(1, int(round(test_size * i0.size)))
    nte1 = max(1, int(round(test_size * i1.size)))
    te = np.concatenate([i0[:nte0], i1[:nte1]])
    tr = np.concatenate([i0[nte0:], i1[nte1:]])
    rng.shuffle(tr)
    rng.shuffle(te)

    logging.info("[INFO] Stratified random split.")
    print_counts("y_train", y[tr])
    print_counts("y_test", y[te])
    return tr, te


# =========================================================
# fast CSR read helpers
# =========================================================
_INDPTR_CACHE = {}

def get_cached_indptr(lg) -> np.ndarray:
    key = id(lg)
    if key not in _INDPTR_CACHE:
        logging.info("[INFO] Caching full indptr array in memory ...")
        _INDPTR_CACHE[key] = np.asarray(lg["indptr"][:], dtype=np.int64)
        logging.info(
            f"[INFO] indptr cached: {_INDPTR_CACHE[key].shape[0]} entries, "
            f"{_INDPTR_CACHE[key].nbytes / 1e6:.1f} MB"
        )
    return _INDPTR_CACHE[key]


def fill_dense_block_for_row_list(lg, rows_u: np.ndarray, G: int) -> np.ndarray:
    indptr_cached = get_cached_indptr(lg)
    indices_arr = lg["indices"]
    data_arr = lg["data"]

    rows_u = np.asarray(rows_u, dtype=np.int64)
    n = rows_u.size
    if n == 0:
        return np.zeros((0, G), dtype=np.float32)

    Xout = np.zeros((n, G), dtype=np.float32)

    seg_starts = [0]
    for i in range(1, n):
        if rows_u[i] != rows_u[i - 1] + 1:
            seg_starts.append(i)
    seg_starts.append(n)

    for si in range(len(seg_starts) - 1):
        a = seg_starts[si]
        b = seg_starts[si + 1]
        seg_rows = rows_u[a:b]
        seg_len = b - a
        rs = int(seg_rows[0])
        re = int(seg_rows[-1]) + 1

        ip = indptr_cached[rs:re + 1]
        data_start = int(ip[0])
        data_end = int(ip[-1])

        if data_end <= data_start:
            continue

        all_idx = np.asarray(indices_arr[data_start:data_end], dtype=np.int32)
        all_dat = np.asarray(data_arr[data_start:data_end], dtype=np.float32)

        ip_local = ip - data_start
        csr = sparse.csr_matrix((all_dat, all_idx, ip_local), shape=(seg_len, G))
        Xout[a:b, :] = csr.toarray()

    return Xout


def fill_dense_for_scattered_rows(lg, rows_sorted: np.ndarray, G: int, superblock_size: int = 50000) -> np.ndarray:
    indptr_cached = get_cached_indptr(lg)
    indices_arr = lg["indices"]
    data_arr = lg["data"]

    rows_sorted = np.asarray(rows_sorted, dtype=np.int64)
    n = rows_sorted.size
    if n == 0:
        return np.zeros((0, G), dtype=np.float32)

    Xout = np.zeros((n, G), dtype=np.float32)

    sb_start = 0
    while sb_start < n:
        first_row = int(rows_sorted[sb_start])
        upper_row = first_row + superblock_size

        sb_end = sb_start
        while sb_end < n and int(rows_sorted[sb_end]) < upper_row:
            sb_end += 1

        sb_rows = rows_sorted[sb_start:sb_end]
        sb_n = sb_end - sb_start

        row_starts = indptr_cached[sb_rows]
        row_ends = indptr_cached[sb_rows + 1]
        data_start = int(row_starts.min())
        data_end = int(row_ends.max())

        if data_end > data_start:
            all_idx = np.asarray(indices_arr[data_start:data_end], dtype=np.int32)
            all_dat = np.asarray(data_arr[data_start:data_end], dtype=np.float32)

            new_indptr = np.zeros(sb_n + 1, dtype=np.int64)
            row_lengths = row_ends - row_starts
            new_indptr[1:] = np.cumsum(row_lengths)
            total_nnz = int(new_indptr[-1])

            new_data = np.empty(total_nnz, dtype=np.float32)
            new_indices = np.empty(total_nnz, dtype=np.int32)

            for i in range(sb_n):
                src_s = int(row_starts[i]) - data_start
                src_e = int(row_ends[i]) - data_start
                dst_s = int(new_indptr[i])
                dst_e = int(new_indptr[i + 1])
                if dst_e > dst_s:
                    new_data[dst_s:dst_e] = all_dat[src_s:src_e]
                    new_indices[dst_s:dst_e] = all_idx[src_s:src_e]

            csr = sparse.csr_matrix((new_data, new_indices, new_indptr), shape=(sb_n, G))
            Xout[sb_start:sb_end, :] = csr.toarray()

            del all_idx, all_dat, new_data, new_indices, csr

        sb_start = sb_end

    return Xout


# =========================================================
# CSR ops on contiguous rows
# =========================================================
def gene_stats_on_sorted_rows_contiguous(lg, rows_sorted: np.ndarray, n_genes: int, block_rows: int):
    rows_sorted = np.asarray(rows_sorted, dtype=np.int64)
    gene_sum = np.zeros(n_genes, dtype=np.float64)
    gene_sum2 = np.zeros(n_genes, dtype=np.float64)

    if rows_sorted.size == 0:
        return gene_sum, gene_sum2

    seg_starts = [0]
    for i in range(1, rows_sorted.size):
        if rows_sorted[i] != rows_sorted[i - 1] + 1:
            seg_starts.append(i)
    seg_starts.append(rows_sorted.size)

    indptr_cached = get_cached_indptr(lg)
    indices_arr = lg["indices"]
    data_arr = lg["data"]

    for si in range(len(seg_starts) - 1):
        a = seg_starts[si]
        b = seg_starts[si + 1]
        seg = rows_sorted[a:b]
        if seg.size == 0:
            continue

        start_row = int(seg[0])
        end_row = int(seg[-1]) + 1

        for rs in range(start_row, end_row, block_rows):
            re = min(end_row, rs + block_rows)

            ip = indptr_cached[rs:re + 1]
            nnz_s = int(ip[0])
            nnz_e = int(ip[-1])
            if nnz_e <= nnz_s:
                continue

            idx = np.asarray(indices_arr[nnz_s:nnz_e], dtype=np.int32)
            dat = np.asarray(data_arr[nnz_s:nnz_e], dtype=np.float32)

            gene_sum += np.bincount(idx, weights=dat.astype(np.float64), minlength=n_genes)
            gene_sum2 += np.bincount(idx, weights=(dat.astype(np.float64) ** 2), minlength=n_genes)

        gc.collect()

    return gene_sum, gene_sum2


# =========================================================
# effect score
# =========================================================
def supervised_effect_score_all_genes(lg, n_genes: int, tr_rows: np.ndarray, y_tr: np.ndarray):
    tr_rows = np.asarray(tr_rows, dtype=np.int64)
    y_tr = np.asarray(y_tr, dtype=np.int8)

    tr0 = np.asarray(tr_rows[y_tr == 0], dtype=np.int64)
    tr1 = np.asarray(tr_rows[y_tr == 1], dtype=np.int64)
    tr0.sort()
    tr1.sort()

    if tr0.size < 50 or tr1.size < 50:
        raise RuntimeError("Too few train cells in one class for effect score.")

    s0, s02 = gene_stats_on_sorted_rows_contiguous(lg, tr0, n_genes, block_rows=20000)
    s1, s12 = gene_stats_on_sorted_rows_contiguous(lg, tr1, n_genes, block_rows=20000)

    mean0 = s0 / float(tr0.size)
    mean1 = s1 / float(tr1.size)
    var0 = (s02 / float(tr0.size)) - mean0 * mean0
    var1 = (s12 / float(tr1.size)) - mean1 * mean1
    var0[var0 < 0] = 0.0
    var1[var1 < 0] = 0.0

    denom = np.sqrt(0.5 * (var0 + var1) + float(CAND_EPS))
    score = np.abs(mean1 - mean0) / denom
    return score.astype(np.float64)


# =========================================================
# corr pruning
# =========================================================
def corr_on_train_subsample_subset(lg, tr_rows: np.ndarray, gene_idx: np.ndarray, max_cells: int, block_rows: int):
    rng = np.random.RandomState(RANDOM_STATE)

    tr_rows = np.asarray(tr_rows, dtype=np.int64)
    if tr_rows.size > max_cells:
        rows = rng.choice(tr_rows, size=max_cells, replace=False)
        logging.info(f"[INFO] corr rows subsample: {max_cells}/{tr_rows.size}")
    else:
        rows = tr_rows
        logging.info(f"[INFO] corr rows subsample: all {tr_rows.size}")

    rows = np.asarray(rows, dtype=np.int64)
    rows.sort()

    gene_idx = np.asarray(gene_idx, dtype=np.int32)
    Gsub = int(gene_idx.size)
    Gall = int(lg.attrs["shape"][1])

    sum_x = np.zeros(Gsub, dtype=np.float64)
    sum_x2 = np.zeros(Gsub, dtype=np.float64)
    XtX = np.zeros((Gsub, Gsub), dtype=np.float64)

    n_rows = rows.size
    processed = 0

    for s in range(0, n_rows, block_rows):
        e = min(n_rows, s + block_rows)
        chunk_rows = rows[s:e]

        Xb_full = fill_dense_for_scattered_rows(lg, chunk_rows, Gall, superblock_size=50000)
        Xb = Xb_full[:, gene_idx]
        xb64 = Xb.astype(np.float64, copy=False)

        sum_x += xb64.sum(axis=0)
        sum_x2 += (xb64 * xb64).sum(axis=0)
        XtX += xb64.T @ xb64

        processed += (e - s)
        if processed % (block_rows * 5) == 0 or processed == n_rows:
            logging.info(f"[INFO] corr processed rows: {processed}/{n_rows}")

        del Xb_full, Xb, xb64
        gc.collect()

    n = float(n_rows)
    mean = sum_x / n
    var = (sum_x2 / n) - mean * mean
    var[var < 1e-12] = 1e-12
    sd = np.sqrt(var)

    cov = (XtX / n) - np.outer(mean, mean)
    corr = cov / np.outer(sd, sd)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)

    return corr.astype(np.float32)


def prune_by_abs_corr_greedy(gene_names: np.ndarray, scores_full: np.ndarray, corr: np.ndarray, thr: float):
    order = np.argsort(-scores_full)
    keep = []
    suppressed = np.zeros(order.size, dtype=bool)

    for i, idx in enumerate(order):
        if suppressed[i]:
            continue
        keep.append(idx)
        bad = np.where(np.abs(corr[idx, order]) >= thr)[0]
        suppressed[bad] = True
        suppressed[i] = True

    keep = np.array(sorted(set(keep)), dtype=np.int32)
    keep_names = np.asarray(gene_names[keep], dtype=object)
    return keep, keep_names


# =========================================================
# stability selection
# =========================================================
def pick_C_for_target_sparsity(Xmm, tr_rows, cov_age, cov_sex, y, groups):
    rng = np.random.RandomState(RANDOM_STATE + 13)
    tr_rows = np.asarray(tr_rows, dtype=np.int64)

    if groups is not None:
        g_tr = groups[tr_rows]
        ug = pd.Series(g_tr).unique().astype(object)
        n_take = max(2, int(round(SUBSAMPLE_FRAC_GROUPS * ug.size)))
        n_take = min(n_take, ug.size)
        take = rng.choice(ug, size=n_take, replace=False)
        mask = np.isin(g_tr, take)
        rows_sub = tr_rows[mask]
    else:
        n_take = max(50, int(round(0.75 * tr_rows.size)))
        rows_sub = rng.choice(tr_rows, size=n_take, replace=False)

    y_sub = y[rows_sub]
    if np.unique(y_sub).size < 2:
        raise RuntimeError("Calibration subsample has only one class; check split.")

    Xg = np.asarray(Xmm[rows_sub, :], dtype=np.float32)
    Xa = cov_age[rows_sub, :].astype(np.float32, copy=False)
    Xs = cov_sex[rows_sub].reshape(-1, 1).astype(np.float32, copy=False)
    X = np.concatenate([Xg, Xa, Xs], axis=1)

    best_C = C_GRID[-1]
    best_diff = 1e18
    rows_out = []

    for C in C_GRID:
        clf = LogisticRegression(
            max_iter=8000,
            solver="saga",
            penalty="elasticnet",
            l1_ratio=float(L1_RATIO_SEL),
            C=float(C),
            class_weight="balanced",
            tol=1e-3,
            random_state=RANDOM_STATE,
        )
        clf.fit(X, y_sub)
        beta = clf.coef_.ravel()[:Xg.shape[1]]
        nnz = int(np.sum(np.abs(beta) > 1e-8))
        diff = abs(nnz - int(TARGET_NNZ_FOR_SELECTION))
        rows_out.append({"C": float(C), "observed_nonzero": nnz, "target_nonzero": int(TARGET_NNZ_FOR_SELECTION)})
        logging.info(f"[INFO] C={C:.2e}, nnz={nnz}, target={TARGET_NNZ_FOR_SELECTION}")
        if diff < best_diff:
            best_diff = diff
            best_C = float(C)

    pd.DataFrame(rows_out).to_csv(OUT_C_GRID, sep="\t", index=False)
    logging.info(f"[INFO] Picked C for stability selection: {best_C}")
    return best_C


def stability_selection(Xmm, tr_rows, cov_age, cov_sex, y, gene_names, groups, C_sel):
    rng = np.random.RandomState(RANDOM_STATE + 7)

    G = Xmm.shape[1]
    freq = np.zeros(G, dtype=np.int32)
    sign_sum = np.zeros(G, dtype=np.int32)
    beta_sum = np.zeros(G, dtype=np.float64)
    beta_abs_sum = np.zeros(G, dtype=np.float64)

    tr_rows = np.asarray(tr_rows, dtype=np.int64)

    if groups is not None:
        g_tr = groups[tr_rows]
        ug = pd.Series(g_tr).unique().astype(object)
        if ug.size < 4:
            logging.warning("[WARN] Too few groups in TRAIN for group-aware subsampling; fallback to cell subsampling.")
            ug = None
    else:
        ug = None

    for t in range(1, int(N_REPEATS) + 1):
        if ug is not None:
            n_take = max(2, int(round(SUBSAMPLE_FRAC_GROUPS * ug.size)))
            n_take = min(n_take, ug.size)
            take = rng.choice(ug, size=n_take, replace=False)
            mask = np.isin(g_tr, take)
            rows_sub = tr_rows[mask]
        else:
            n_take = max(50, int(round(0.75 * tr_rows.size)))
            rows_sub = rng.choice(tr_rows, size=n_take, replace=False)

        y_sub = y[rows_sub]
        if np.unique(y_sub).size < 2:
            continue

        Xg = np.asarray(Xmm[rows_sub, :], dtype=np.float32)
        Xa = cov_age[rows_sub, :].astype(np.float32, copy=False)
        Xs = cov_sex[rows_sub].reshape(-1, 1).astype(np.float32, copy=False)
        X = np.concatenate([Xg, Xa, Xs], axis=1)

        clf = LogisticRegression(
            max_iter=8000,
            solver="saga",
            penalty="elasticnet",
            l1_ratio=float(L1_RATIO_SEL),
            C=float(C_sel),
            class_weight="balanced",
            tol=1e-3,
            random_state=RANDOM_STATE,
        )
        clf.fit(X, y_sub)
        beta = clf.coef_.ravel()[:G]

        sel = np.abs(beta) > 1e-8
        freq[sel] += 1
        sign_sum[sel] += np.where(beta[sel] > 0, 1, -1)
        beta_sum[sel] += beta[sel].astype(np.float64)
        beta_abs_sum[sel] += np.abs(beta[sel]).astype(np.float64)

        if t % 10 == 0:
            logging.info(f"[INFO] stability repeat {t}/{N_REPEATS}, nnz_genes={int(sel.sum())}")

    df = pd.DataFrame({
        "gene": gene_names,
        "selection_freq": freq / float(N_REPEATS),
        "sign_consistency": np.where(freq > 0, np.abs(sign_sum) / freq, 0.0),
        "mean_beta": np.where(freq > 0, beta_sum / freq, 0.0),
        "mean_abs_beta": np.where(freq > 0, beta_abs_sum / freq, 0.0),
    }).sort_values(
        ["selection_freq", "sign_consistency", "mean_abs_beta"],
        ascending=[False, False, False],
    )

    return df.reset_index(drop=True)


# =========================================================
# final model
# =========================================================
def fit_logistic_with_fallback(X, y):
    try:
        clf = LogisticRegression(
            max_iter=8000,
            solver=FINAL_SOLVER,
            penalty="elasticnet",
            l1_ratio=float(FINAL_L1_RATIO),
            C=float(FINAL_C),
            class_weight="balanced",
            tol=1e-4,
            random_state=RANDOM_STATE,
        )
        clf.fit(X, y)
        logging.info("[INFO] final model fit with elasticnet")
        return clf
    except Exception as e:
        logging.warning(f"[WARN] final elasticnet failed: {e}")

    clf = LogisticRegression(
        max_iter=8000,
        solver="liblinear",
        penalty="l2",
        C=float(FINAL_C),
        class_weight="balanced",
        tol=1e-4,
        random_state=RANDOM_STATE,
    )
    clf.fit(X, y)
    logging.info("[INFO] final model fallback to liblinear+l2")
    return clf


def export_final_coefficients(clf, panel_human):
    feat_names = list(panel_human.astype(str)) + AGE_STAGE_COLS + ["sex"]
    beta = clf.coef_.ravel()
    pd.DataFrame({
        "feature": feat_names,
        "beta": beta,
        "abs_beta": np.abs(beta),
    }).sort_values("abs_beta", ascending=False).to_csv(OUT_FINAL_COEF, sep="\t", index=False)


# =========================================================
# ortholog helpers
# =========================================================
def load_human_mouse_ortholog_candidates(human_var_names, mouse_var_names, ortholog_tsv):
    orth = pd.read_csv(ortholog_tsv, sep="\t")
    if not {"human_gene", "mouse_gene"}.issubset(orth.columns):
        raise ValueError("Ortholog table must contain columns: human_gene, mouse_gene")

    orth = orth.copy()
    orth["human_gene"] = orth["human_gene"].astype(str)
    orth["mouse_gene"] = orth["mouse_gene"].astype(str)

    if "orthology_type" in orth.columns:
        orth = orth[orth["orthology_type"] == "ortholog_one2one"].copy()

    orth = orth[orth.apply(lambda r: keep_gene_pair(r["human_gene"], r["mouse_gene"]), axis=1)].copy()
    orth = orth[orth["human_gene"].isin(set(human_var_names.tolist()))].copy()
    orth = orth[orth["mouse_gene"].isin(set(mouse_var_names.tolist()))].copy()
    orth = orth.drop_duplicates(subset=["human_gene", "mouse_gene"]).copy()

    human2mouse = dict(zip(orth["human_gene"], orth["mouse_gene"]))
    candidate_mask = np.isin(human_var_names, orth["human_gene"].to_numpy(dtype=object))
    candidate_pos = np.where(candidate_mask)[0].astype(np.int32)

    return orth, human2mouse, candidate_pos


# =========================================================
# species setup
# =========================================================
def setup_human():
    strip_h5ad_to_x_obs_var(HUMAN_H5AD_PATH, HUMAN_STRIPPED_H5AD_PATH)
    base_orig_idx, age_stage_base, sex_base, groups_base, group_col = build_base_kept_from_stripped_human(HUMAN_STRIPPED_H5AD_PATH)

    gene_counts, gene_ncells = compute_gene_stats_on_rows(HUMAN_STRIPPED_H5AD_PATH, base_orig_idx, CHUNK_CELLS)
    min_cells = max(1, int(base_orig_idx.size * MIN_GENE_CELLS_FRAC))
    gene_mask = (gene_counts >= MIN_GENE_COUNTS) & (gene_ncells >= min_cells)

    build_universal_zarr_if_missing(
        HUMAN_STRIPPED_H5AD_PATH,
        HUMAN_ZARR_PATH,
        base_orig_idx,
        gene_mask,
        TARGET_SUM,
        CHUNK_CELLS,
    )

    zroot, lg, n_cells_z, n_genes_z = open_log1p_csr(HUMAN_ZARR_PATH)
    var_names = load_var_names(zroot)
    obs_orig_index = load_obs_orig_index(zroot)

    return {
        "base_orig_idx": base_orig_idx,
        "age_stage_base": age_stage_base,
        "sex_base": sex_base,
        "groups_base": groups_base,
        "group_col": group_col,
        "zroot": zroot,
        "lg": lg,
        "n_cells_z": n_cells_z,
        "n_genes_z": n_genes_z,
        "var_names": var_names,
        "obs_orig_index": obs_orig_index,
    }


def setup_mouse():
    strip_h5ad_to_x_obs_var(MOUSE_H5AD_PATH, MOUSE_STRIPPED_H5AD_PATH)
    base_orig_idx, age_stage_base, sex_base, groups_base, group_col = build_base_kept_from_stripped_mouse(MOUSE_STRIPPED_H5AD_PATH)

    gene_counts, gene_ncells = compute_gene_stats_on_rows(MOUSE_STRIPPED_H5AD_PATH, base_orig_idx, CHUNK_CELLS)
    min_cells = max(1, int(base_orig_idx.size * MIN_GENE_CELLS_FRAC))
    gene_mask = (gene_counts >= MIN_GENE_COUNTS) & (gene_ncells >= min_cells)

    build_universal_zarr_if_missing(
        MOUSE_STRIPPED_H5AD_PATH,
        MOUSE_ZARR_PATH,
        base_orig_idx,
        gene_mask,
        TARGET_SUM,
        CHUNK_CELLS,
    )

    zroot, lg, n_cells_z, n_genes_z = open_log1p_csr(MOUSE_ZARR_PATH)
    var_names = load_var_names(zroot)
    obs_orig_index = load_obs_orig_index(zroot)

    return {
        "base_orig_idx": base_orig_idx,
        "age_stage_base": age_stage_base,
        "sex_base": sex_base,
        "groups_base": groups_base,
        "group_col": group_col,
        "zroot": zroot,
        "lg": lg,
        "n_cells_z": n_cells_z,
        "n_genes_z": n_genes_z,
        "var_names": var_names,
        "obs_orig_index": obs_orig_index,
    }


# =========================================================
# main
# =========================================================
def main():
    setup_logging(WORKDIR)

    with log_step("SETUP HUMAN"):
        H = setup_human()

    with log_step("SETUP MOUSE"):
        M = setup_mouse()

    var_h = H["var_names"].astype(object)
    var_m = M["var_names"].astype(object)

    with log_step("LOAD ORTHOLOG"):
        orth, human2mouse, candidate_pos = load_human_mouse_ortholog_candidates(
            var_h,
            var_m,
            ORTHOLOG_TSV,
        )
        logging.info(f"[INFO] usable 1:1 ortholog genes: {candidate_pos.size}")
        if candidate_pos.size < MIN_PRUNED_GENES:
            raise RuntimeError("Too few ortholog genes remain after filtering.")

    with log_step("BUILD HUMAN TASK"):
        task_rows_h, y_h, age_stage_h, sex_h, groups_h = build_task_from_universal_base_human(
            HUMAN_STRIPPED_H5AD_PATH,
            H["base_orig_idx"],
            H["age_stage_base"],
            H["sex_base"],
            H["groups_base"],
        )

        age_mat_h, sex_num_h = build_covariates(age_stage_h, sex_h)

        if groups_h is not None and USE_GROUP_SPLIT:
            tr_i, te_i = stratified_group_split_local(
                y_h,
                groups_h.to_numpy(dtype=object),
                TEST_SIZE,
                RANDOM_STATE,
            )
            groups_arr_h = groups_h.to_numpy(dtype=object)
        else:
            tr_i, te_i = stratified_random_split_local(
                y_h,
                TEST_SIZE,
                RANDOM_STATE,
            )
            groups_arr_h = None

        tr_rows_u = task_rows_h[tr_i].astype(np.int64)
        te_rows_u = task_rows_h[te_i].astype(np.int64)
        y_tr, y_te = y_h[tr_i], y_h[te_i]

        print_counts("human_train", y_tr)
        print_counts("human_test", y_te)

    with log_step("EFFECT SCORE ON ORTHOLOG GENE SPACE"):
        score_all = supervised_effect_score_all_genes(H["lg"], H["n_genes_z"], tr_rows_u, y_tr)
        cand_scores = score_all[candidate_pos]

        pd.DataFrame({
            "human_gene": var_h[candidate_pos].astype(str),
            "effect_score": cand_scores,
            "mouse_gene": [human2mouse[g] for g in var_h[candidate_pos].astype(str)],
        }).sort_values("effect_score", ascending=False).to_csv(
            OUT_CANDIDATE_SCORES, sep="\t", index=False
        )

    with log_step("CORR PRUNING ON ORTHOLOG GENE SPACE"):
        corr = corr_on_train_subsample_subset(
            H["lg"],
            tr_rows_u,
            candidate_pos,
            max_cells=MAX_CORR_CELLS,
            block_rows=CORR_BLOCK_ROWS,
        )

        pruned_local, pruned_names = prune_by_abs_corr_greedy(
            var_h[candidate_pos].astype(object),
            cand_scores,
            corr,
            thr=PRUNE_ABS_CORR,
        )

        del corr
        gc.collect()

        pruned_global = candidate_pos[pruned_local]
        pruned_names_arr = pruned_names.astype(object)

        if pruned_global.size < MIN_PRUNED_GENES:
            raise RuntimeError(f"Too few genes after pruning: {pruned_global.size} < {MIN_PRUNED_GENES}")

        pd.Series(pruned_names_arr).to_csv(OUT_PRUNED_GENES, sep="\t", index=False, header=False)
        pd.DataFrame({
            "human_gene": pruned_names_arr,
            "human_gene_index": pruned_global,
            "mouse_gene": [human2mouse[g] for g in pruned_names_arr.astype(str)],
        }).to_csv(OUT_PRUNING_MAP, sep="\t", index=False)

    with log_step("BUILD HUMAN X MEMMAP ON ORTHOLOG-PRUNED GENES"):
        if os.path.exists(XMM_PATH):
            os.remove(XMM_PATH)

        n_task_h = int(task_rows_h.size)
        Xmm = np.memmap(XMM_PATH, dtype=np.float32, mode="w+", shape=(n_task_h, int(pruned_global.size)))

        sum_x = np.zeros(pruned_global.size, dtype=np.float64)
        sum_x2 = np.zeros(pruned_global.size, dtype=np.float64)

        for s, e in iter_ranges(tr_i.size, 2000):
            rows_u = task_rows_h[tr_i[s:e]]
            Xb_full = fill_dense_block_for_row_list(H["lg"], rows_u, H["n_genes_z"])
            Xb = Xb_full[:, pruned_global].astype(np.float64, copy=False)

            sum_x += Xb.sum(axis=0)
            sum_x2 += (Xb * Xb).sum(axis=0)

            del Xb_full, Xb
            gc.collect()

        mean = (sum_x / float(tr_i.size)).astype(np.float32)
        var = (sum_x2 / float(tr_i.size)) - (mean.astype(np.float64) ** 2)
        var[var < 1e-12] = 1e-12
        std = np.sqrt(var).astype(np.float32)

        for s, e in iter_ranges(n_task_h, 2000):
            rows_u = task_rows_h[s:e]
            Xb_full = fill_dense_block_for_row_list(H["lg"], rows_u, H["n_genes_z"])
            Xb = Xb_full[:, pruned_global]
            Xb = (Xb - mean) / std
            if Z_CLIP is not None and float(Z_CLIP) > 0:
                np.clip(Xb, -float(Z_CLIP), float(Z_CLIP), out=Xb)
            Xmm[s:e, :] = Xb.astype(np.float32, copy=False)
            Xmm.flush()

            del Xb_full, Xb
            gc.collect()

    with log_step("STABILITY SELECTION ON HUMAN / ORTHOLOG-PRUNED SPACE"):
        C_sel = pick_C_for_target_sparsity(
            Xmm,
            tr_i,
            age_mat_h,
            sex_num_h,
            y_h,
            groups_arr_h,
        )

        df_stab = stability_selection(
            Xmm,
            tr_i,
            age_mat_h,
            sex_num_h,
            y_h,
            pruned_names_arr,
            groups_arr_h,
            C_sel=C_sel,
        )
        df_stab.to_csv(OUT_STABILITY, sep="\t", index=False)

    with log_step("BUILD TOP PANEL"):
        panel_human = df_stab.head(PANEL_K)["gene"].to_numpy(dtype=object)
        panel_mask = np.isin(pruned_names_arr, panel_human)
        panel_cols = np.where(panel_mask)[0].astype(np.int32)

        if panel_cols.size == 0:
            raise RuntimeError("Panel selection failed: no genes selected.")

        panel_mouse = np.array([human2mouse[g] for g in panel_human], dtype=object)
        pd.DataFrame({
            "human_gene": panel_human,
            "mouse_gene": panel_mouse,
        }).to_csv(OUT_PANEL, sep="\t", index=False)

        df_stab.head(PANEL_K).to_csv(OUT_HUMAN_PANEL_STATS, sep="\t", index=False)

    with log_step("FINAL HUMAN MODEL"):
        Xg_tr = np.asarray(Xmm[tr_i, :][:, panel_cols], dtype=np.float32)
        Xg_te = np.asarray(Xmm[te_i, :][:, panel_cols], dtype=np.float32)

        X_tr = np.concatenate(
            [
                Xg_tr,
                age_mat_h[tr_i, :].astype(np.float32),
                sex_num_h[tr_i].reshape(-1, 1).astype(np.float32),
            ],
            axis=1,
        )
        X_te = np.concatenate(
            [
                Xg_te,
                age_mat_h[te_i, :].astype(np.float32),
                sex_num_h[te_i].reshape(-1, 1).astype(np.float32),
            ],
            axis=1,
        )

        clf = fit_logistic_with_fallback(X_tr, y_tr)

        y_pred_in = clf.predict(X_te)
        y_proba_in = clf.predict_proba(X_te)[:, 1]

        internal_acc = float(accuracy_score(y_te, y_pred_in))
        internal_auc = float(roc_auc_score(y_te, y_proba_in))
        rep_h = classification_report(y_te, y_pred_in, target_names=["Class0", "Class1"], digits=6)

        logging.info("\n===== HUMAN INTERNAL TEST =====")
        logging.info(f"Accuracy: {internal_acc:.4f}")
        logging.info(f"AUC     : {internal_auc:.4f}")
        logging.info("\n" + rep_h)

        save_json(
            {
                "acc": internal_acc,
                "auc": internal_auc,
                "panel_k": int(panel_cols.size),
                "n_train": int(tr_i.size),
                "n_test": int(te_i.size),
                "n_repeats": N_REPEATS,
                "threads": os.environ.get("OMP_NUM_THREADS", None),
            },
            OUT_METRICS_HUMAN,
        )
        with open(OUT_REPORT_HUMAN, "w", encoding="utf-8") as f:
            f.write(rep_h)

        export_final_coefficients(clf, panel_human)

        if EXPORT_ROC_DATA:
            if ROC_EXPORT_MODE == "panel_only":
                save_curves(y_te, y_proba_in, os.path.join(WORKDIR, "human_internal_panel"))
            else:
                save_curves(y_te, y_proba_in, os.path.join(WORKDIR, "human_internal_all_pruned"))

    with log_step("BUILD MOUSE TASK"):
        task_rows_m, y_m, age_stage_m, sex_m, groups_m = build_task_from_universal_base_mouse(
            MOUSE_STRIPPED_H5AD_PATH,
            M["base_orig_idx"],
            M["age_stage_base"],
            M["sex_base"],
            M["groups_base"],
        )

        age_mat_m, sex_num_m = build_covariates(age_stage_m, sex_m)

        var_m_to_pos = {g: i for i, g in enumerate(var_m.tolist())}
        panel_mouse_pos = np.array([var_m_to_pos[g] for g in panel_mouse], dtype=np.int32)

        panel_mean = mean[panel_cols]
        panel_std = std[panel_cols]

    with log_step("MOUSE EXTERNAL VALIDATION"):
        y_proba_ext = np.zeros(task_rows_m.size, dtype=np.float32)

        for s, e in iter_ranges(task_rows_m.size, EXT_CHUNK):
            rows_u = task_rows_m[s:e]
            Xb_full = fill_dense_block_for_row_list(M["lg"], rows_u, M["n_genes_z"])

            Xg = Xb_full[:, panel_mouse_pos]
            Xg = (Xg - panel_mean) / panel_std
            if Z_CLIP is not None and float(Z_CLIP) > 0:
                np.clip(Xg, -float(Z_CLIP), float(Z_CLIP), out=Xg)

            X_ext = np.concatenate(
                [
                    Xg.astype(np.float32),
                    age_mat_m[s:e, :].astype(np.float32),
                    sex_num_m[s:e].reshape(-1, 1).astype(np.float32),
                ],
                axis=1,
            )
            y_proba_ext[s:e] = clf.predict_proba(X_ext)[:, 1]

            del Xb_full, Xg, X_ext
            gc.collect()

        y_pred_ext = (y_proba_ext >= 0.5).astype(np.int8)
        external_acc = float(accuracy_score(y_m, y_pred_ext))
        external_auc = float(roc_auc_score(y_m, y_proba_ext))
        rep_m = classification_report(y_m, y_pred_ext, target_names=["Class0", "Class1"], digits=6)

        logging.info("\n===== MOUSE EXTERNAL TEST =====")
        logging.info(f"Accuracy: {external_acc:.4f}")
        logging.info(f"AUC     : {external_auc:.4f}")
        logging.info("\n" + rep_m)

        save_json(
            {
                "acc": external_acc,
                "auc": external_auc,
                "panel_k": int(panel_cols.size),
                "n_mouse_test": int(task_rows_m.size),
                "n_repeats": N_REPEATS,
                "threads": os.environ.get("OMP_NUM_THREADS", None),
            },
            OUT_METRICS_MOUSE,
        )
        with open(OUT_REPORT_MOUSE, "w", encoding="utf-8") as f:
            f.write(rep_m)

        if EXPORT_ROC_DATA:
            save_curves(y_m, y_proba_ext, os.path.join(WORKDIR, "mouse_external"))

    pd.DataFrame({
        "mouse_task_row": np.arange(task_rows_m.size, dtype=np.int64),
        "mouse_zarr_row": task_rows_m,
        "mouse_orig_obs_index": M["obs_orig_index"][task_rows_m],
        "y_true": y_m,
        "y_proba": y_proba_ext,
        "y_pred": y_pred_ext,
    }).to_csv(OUT_MOUSE_SCORES, sep="\t", index=False)

    bundle = {
        "train_species": "human",
        "test_species": "mouse",
        "panel_human": list(panel_human.astype(str)),
        "panel_mouse": list(panel_mouse.astype(str)),
        "age_stage_cols": AGE_STAGE_COLS,
        "beta": clf.coef_.ravel().astype(float).tolist(),
        "intercept": float(clf.intercept_.ravel()[0]),
        "panel_mean": panel_mean.astype(float).tolist(),
        "panel_std": panel_std.astype(float).tolist(),
        "human_cond_map": HUMAN_COND_MAP,
        "mouse_cond_map": MOUSE_COND_MAP,
        "threads": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", None),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", None),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", None),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS", None),
        },
        "n_repeats": N_REPEATS,
    }
    save_json(bundle, OUT_BUNDLE)

    logging.info(f"[INFO] Done. Outputs written to: {WORKDIR}")


if __name__ == "__main__":
    main()
