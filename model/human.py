import os
import re
import gc
import json
import numpy as np
import pandas as pd
import h5py
import zarr
import scanpy as sc
import numcodecs
from scipy import sparse

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold


# =========================================================
# USER EDIT BLOCK (only change here)
# =========================================================

# paths
H5AD_PATH = r"/gpfsdata/home/renyixiang/SkinDB/10X/human/Annotation/Annotated/human_with_all_leiden_10threshold_annotated_updated.h5ad"
STRIPPED_H5AD_PATH = r"/gpfsdata/home/renyixiang/SkinDB/10X/human/model/human_model.stripped.h5ad"
ZARR_PATH = r"/gpfsdata/home/renyixiang/SkinDB/10X/human/model/human_universal.zarr"  # one universal zarr

# outputs
WORKDIR = r"/gpfsdata/home/renyixiang/SkinDB/10X/human/model/_tmp_panel_one_zarr"

# obs columns
AGE_COL = "Age"
SEX_COL = "sex"
COND_COL = "condition"

# condition mapping for THIS run/task (you can change and rerun WITHOUT rebuilding zarr)
COND_MAP = {"Healthy": 0, "psoriasis": 1}

# >>> HUMAN AGE BINS (numeric age in years)
AGE_BIN_COLS = ["Age_0_20", "Age_20_40", "Age_40_60", "Age_gt_60"]

# build behavior
BUILD_MISSING_STRIPPED = True
BUILD_MISSING_ZARR = True   # if zarr/log1p missing, build once
NEVER_OVERWRITE_ZARR = True # safety: refuse if zarr path exists but needs rebuild

# export model ROC data files only (no single-gene export)
EXPORT_ROC_DATA = True

# additional evaluation
PRINT_TRAIN_METRICS = True
RUN_SAMPLE_LEVEL_CV = True
SAMPLE_LEVEL_CV_MODE = "auto"   # "auto" | "loso" | "kfold"
SAMPLE_LEVEL_CV_N_SPLITS = 5

# =========================================================


# ======================
# Env: reduce hidden spikes
# ======================
os.environ.setdefault("OMP_NUM_THREADS", "10")
os.environ.setdefault("MKL_NUM_THREADS", "10")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "10")


# ======================
# CONFIG (base zarr build)
# ======================
TARGET_SUM = 1e4
CHUNK_CELLS = 20_000

MIN_GENE_COUNTS = 1
MIN_GENE_CELLS_FRAC = 0.01

# ======================
# CONFIG (task training)
# ======================
TEST_SIZE = 0.2
RANDOM_STATE = 42

USE_GROUP_SPLIT = True
GROUP_COL_CANDIDATES = [
    "human_id", "Human", "human",
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
N_REPEATS = 50
SUBSAMPLE_FRAC_GROUPS = 0.75
C_GRID = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
L1_RATIO_SEL = 0.9
TARGET_NNZ_FOR_SELECTION = 250

# final panel size
PANEL_K = 50

# final model (L2-like via elasticnet l1_ratio=0)
FINAL_C = 1.0
FINAL_SOLVER = "saga"
FINAL_L1_RATIO = 0.0
Z_CLIP = 5.0

# supervised effect score epsilon
CAND_EPS = 1e-8

# outputs
os.makedirs(WORKDIR, exist_ok=True)
OUT_PRUNED_GENES = os.path.join(WORKDIR, "genes_pruned.txt")
OUT_PRUNING_MAP = os.path.join(WORKDIR, "pruning_map.tsv")
OUT_STABILITY = os.path.join(WORKDIR, "stability_selection.tsv")
OUT_PANEL = os.path.join(WORKDIR, "panel_genes.tsv")
OUT_METRICS = os.path.join(WORKDIR, "final_metrics.json")
OUT_REPORT = os.path.join(WORKDIR, "final_classification_report.txt")
OUT_SAMPLE_CV = os.path.join(WORKDIR, "sample_level_cv_predictions.tsv")
XMM_PATH = os.path.join(WORKDIR, "X_pruned_genes_float32.memmap")


# ======================
# Human Age parsing & binning (numeric age)
# ======================
def parse_human_age(age):
    """
    Expect age column to be numeric or numeric string (years).
    Return float age (years) or None if invalid.
    """
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


def age_bin_from_years(age_years):
    """
    Age bins:
      0-20
      20-40
      40-60
      60+
    """
    if age_years is None:
        return None
    if age_years < 20:
        return "Age_0_20"
    if age_years < 40:
        return "Age_20_40"
    if age_years < 60:
        return "Age_40_60"
    return "Age_gt_60"


def encode_sex(x):
    if x is None:
        return 0.5
    s = str(x).strip().lower()
    if s in ["male", "m", "♂", "1"]:
        return 1.0
    if s in ["female", "f", "♀", "0"]:
        return 0.0
    return 0.5


# ======================
# Utils
# ======================
def to_pystr_array(x):
    if hasattr(x, "astype") and hasattr(x, "map") and hasattr(x, "to_numpy"):
        s = x.astype("object")
        return s.map(lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)).to_numpy(dtype=object)
    arr = np.asarray(x, dtype="object")
    return np.array(["" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v) for v in arr], dtype=object)


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
    print(f"[INFO] {name} counts: {vc}")


def safe_binary_auc(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.int8)
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def make_classification_report(y_true, y_pred):
    return classification_report(
        y_true,
        y_pred,
        target_names=["Class0", "Class1"],
        digits=6,
        zero_division=0,
    )


def aggregate_sample_predictions(groups, y_true, y_score):
    df = pd.DataFrame({
        "group": np.asarray(groups, dtype=object),
        "y_true": np.asarray(y_true, dtype=np.int8),
        "cell_score": np.asarray(y_score, dtype=np.float64),
    })
    nu = df.groupby("group")["y_true"].nunique()
    bad = nu[nu > 1]
    if len(bad) > 0:
        raise RuntimeError(f"Found mixed-label groups during sample aggregation. Examples: {bad.index.tolist()[:10]}")

    df_sample = (
        df.groupby("group", sort=False)
          .agg(
              y_true=("y_true", "first"),
              sample_score=("cell_score", "mean"),
              n_cells=("cell_score", "size"),
          )
          .reset_index()
    )
    df_sample["y_pred"] = (df_sample["sample_score"] >= 0.5).astype(np.int8)
    return df_sample


def choose_sample_cv_splitter(groups, y):
    df = pd.DataFrame({
        "group": np.asarray(groups, dtype=object),
        "y": np.asarray(y, dtype=np.int8),
    })
    g2y = df.groupby("group")["y"].first()
    n_groups = int(g2y.shape[0])
    n_pos = int(np.sum(g2y.to_numpy(dtype=np.int8) == 1))
    n_neg = int(np.sum(g2y.to_numpy(dtype=np.int8) == 0))
    min_class_groups = min(n_pos, n_neg)

    mode = str(SAMPLE_LEVEL_CV_MODE).lower()
    if min_class_groups < 2:
        return None, {
            "reason": f"Need at least 2 groups in each class for sample-level CV, got class0={n_neg}, class1={n_pos}."
        }

    if mode == "loso" or (mode == "auto" and n_groups < 30):
        return ("loso", LeaveOneGroupOut()), {
            "n_groups": n_groups,
            "n_pos_groups": n_pos,
            "n_neg_groups": n_neg,
            "n_splits": n_groups,
        }

    n_splits = min(int(SAMPLE_LEVEL_CV_N_SPLITS), min_class_groups)
    if n_splits < 2:
        return None, {"reason": f"Invalid sample-level CV n_splits={n_splits}."}

    return (
        "kfold",
        StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE),
    ), {
        "n_groups": n_groups,
        "n_pos_groups": n_pos,
        "n_neg_groups": n_neg,
        "n_splits": n_splits,
    }


def run_sample_level_cv_fixed_panel(X_all, y_all, groups, group_col_name: str):
    choice, meta = choose_sample_cv_splitter(groups, y_all)
    if choice is None:
        print(f"[WARN] Sample-level CV skipped: {meta['reason']}")
        return None

    mode_name, splitter = choice
    y_all = np.asarray(y_all, dtype=np.int8)
    groups = np.asarray(groups, dtype=object)
    records = []

    if mode_name == "loso":
        split_iter = splitter.split(X_all, y_all, groups)
    else:
        split_iter = splitter.split(X_all, y_all, groups=groups)

    for fold_id, (tr_idx, te_idx) in enumerate(split_iter, start=1):
        y_tr_fold = y_all[tr_idx]
        if np.unique(y_tr_fold).size < 2:
            print(f"[WARN] Sample-level CV fold {fold_id} skipped: training fold has one class only.")
            continue

        clf_fold = LogisticRegression(
            max_iter=8000,
            solver=FINAL_SOLVER,
            penalty="elasticnet",
            l1_ratio=float(FINAL_L1_RATIO),
            C=float(FINAL_C),
            class_weight="balanced",
            tol=1e-4,
        )
        clf_fold.fit(X_all[tr_idx], y_tr_fold)
        y_score_fold = clf_fold.predict_proba(X_all[te_idx])[:, 1]

        records.append(pd.DataFrame({
            "fold": fold_id,
            "group": groups[te_idx].astype(object),
            "y_true": y_all[te_idx].astype(np.int8),
            "cell_score": y_score_fold.astype(np.float64),
        }))

        if fold_id % 5 == 0 or fold_id == 1:
            print(f"[INFO] Sample-level CV fold {fold_id}/{meta['n_splits']} done.")

        del clf_fold
        gc.collect()

    if not records:
        print("[WARN] Sample-level CV skipped: no valid folds were completed.")
        return None

    df_cell = pd.concat(records, ignore_index=True)
    df_sample = aggregate_sample_predictions(
        groups=df_cell["group"].to_numpy(dtype=object),
        y_true=df_cell["y_true"].to_numpy(dtype=np.int8),
        y_score=df_cell["cell_score"].to_numpy(dtype=np.float64),
    )
    fold_map = df_cell.groupby("group", sort=False)["fold"].first().reset_index()
    df_sample = df_sample.merge(fold_map, on="group", how="left")
    df_sample.rename(columns={"group": group_col_name}, inplace=True)

    sample_acc = float(accuracy_score(df_sample["y_true"], df_sample["y_pred"]))
    sample_auc = safe_binary_auc(df_sample["y_true"], df_sample["sample_score"])
    sample_report = make_classification_report(df_sample["y_true"], df_sample["y_pred"])

    df_sample.to_csv(OUT_SAMPLE_CV, sep="\t", index=False)
    print(f"[INFO] Wrote sample-level CV predictions: {OUT_SAMPLE_CV}")

    return {
        "mode": mode_name,
        "n_splits": int(meta["n_splits"]),
        "n_groups": int(meta["n_groups"]),
        "n_pos_groups": int(meta["n_pos_groups"]),
        "n_neg_groups": int(meta["n_neg_groups"]),
        "acc": sample_acc,
        "auc": sample_auc,
        "report": sample_report,
    }


# ======================
# Step0: strip h5ad -> keep only X/obs/var (skip if exists)
# ======================
def strip_h5ad_to_x_obs_var(in_h5ad: str, out_h5ad: str):
    if os.path.exists(out_h5ad):
        print(f"[INFO] Detected existing stripped h5ad, skip: {out_h5ad}")
        return
    if not BUILD_MISSING_STRIPPED:
        raise FileNotFoundError(f"Missing stripped h5ad and BUILD_MISSING_STRIPPED=False: {out_h5ad}")

    print(f"[INFO] Creating stripped h5ad (X/obs/var only) -> {out_h5ad}")
    with h5py.File(in_h5ad, "r") as fin, h5py.File(out_h5ad, "w") as fout:
        for k, v in fin.attrs.items():
            fout.attrs[k] = v
        root_keys = list(fin.keys())
        print(f"[INFO] Input h5ad root keys: {root_keys}")
        for key in ("X", "obs", "var"):
            if key not in fin:
                raise KeyError(f"Key '{key}' not found in input h5ad. Root keys: {root_keys}")
            fin.copy(key, fout, name=key)
    print("[INFO] Stripped h5ad created.")


# ======================
# Base kept cells: ONLY depends on AGE parsing (universal zarr rows)
# ======================
def build_base_kept_from_stripped(stripped_h5ad: str):
    adata_b = sc.read_h5ad(stripped_h5ad, backed="r")
    obs = adata_b.obs

    if AGE_COL not in obs.columns:
        raise KeyError(f"Missing '{AGE_COL}' in obs.")
    if SEX_COL not in obs.columns:
        raise KeyError(f"Missing '{SEX_COL}' in obs.")
    if COND_COL not in obs.columns:
        print(f"[WARN] Missing '{COND_COL}' in obs. Condition task later may fail.")

    # >>> HUMAN AGE: numeric years
    age_raw = pd.Series(to_pystr_array(obs[AGE_COL]))
    age_years = age_raw.apply(parse_human_age)
    age_bin = age_years.apply(age_bin_from_years)
    mask_age = age_bin.notna()

    base_orig_idx = np.where(mask_age.to_numpy())[0].astype(np.int64)
    age_bin_base = age_bin[mask_age].reset_index(drop=True)
    sex_base = obs.iloc[base_orig_idx][SEX_COL].astype("object").reset_index(drop=True)

    # groups (base)
    group_col = None
    for c in GROUP_COL_CANDIDATES:
        if c in obs.columns:
            group_col = c
            break
    if group_col is not None:
        groups_base = obs.iloc[base_orig_idx][group_col].astype("object").reset_index(drop=True)
        print(f"[INFO] Base group column: {group_col} (unique={groups_base.nunique()})")
    else:
        groups_base = None
        print("[WARN] No group column found in obs for base data.")

    print(f"[INFO] Base kept cells (age-parsed): {base_orig_idx.size} / {adata_b.n_obs}")
    print("[INFO] Age bin counts (base):")
    print(age_bin_base.value_counts())

    try:
        adata_b.file.close()
    except Exception:
        pass

    return base_orig_idx, age_bin_base, sex_base, groups_base, group_col


# ======================
# Gene stats on base cells (for universal gene_mask)
# ======================
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


# ======================
# Build universal zarr/log1p CSR if missing
# ======================
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
    # If exists and has log1p: reuse
    if os.path.exists(zarr_path):
        root = zarr.open_group(zarr_path, mode="r")
        if "layers" in root and "log1p" in root["layers"]:
            print("[INFO] Detected existing zarr/layers/log1p, reuse read-only.")
            return
        # exists but missing log1p
        if NEVER_OVERWRITE_ZARR:
            raise RuntimeError(
                f"Zarr exists but layers/log1p missing and NEVER_OVERWRITE_ZARR=True: {zarr_path}"
            )
        else:
            raise RuntimeError("Overwriting existing zarr is not supported in this script for safety.")

    if not BUILD_MISSING_ZARR:
        raise FileNotFoundError(f"Missing zarr and BUILD_MISSING_ZARR=False: {zarr_path}")

    print(f"[INFO] Building universal zarr (no condition dependency) -> {zarr_path}")
    print(f"[INFO] Base rows: {base_orig_idx.size}, gene_mask kept genes: {int(gene_mask.sum())}")

    # Load var names from stripped (full var, then mask)
    adata_b = sc.read_h5ad(stripped_h5ad, backed="r")
    var_names_full = to_pystr_array(adata_b.var_names)
    var_names_kept = var_names_full[gene_mask]
    n_cells = int(base_orig_idx.size)
    n_genes_kept = int(gene_mask.sum())
    try:
        adata_b.file.close()
    except Exception:
        pass

    # Pass 1: count nnz per row after normalize+log1p and gene_mask, build indptr
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
        indptr[start + 1:end + 1] = nnz_row  # store counts temporarily
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

    print(f"[INFO] Universal log1p CSR nnz_total={nnz_total}")

    # Create zarr (mode="w-" prevents overwrite)
    root = zarr.open_group(zarr_path, mode="w-")

    root.create_dataset(
        "obs_orig_index",
        data=base_orig_idx.astype(np.int64),
        chunks=(min(n_cells, 200000),),
        dtype="i8",
    )

    vg = root.create_group("var")
    vg.create_dataset("_index", data=var_names_kept.astype(object), chunks=(min(n_genes_kept, 200000),), dtype=object, object_codec=numcodecs.VLenUTF8())

    layers = root.create_group("layers")
    lg = layers.create_group("log1p")
    lg.attrs["shape"] = [n_cells, n_genes_kept]

    lg.create_dataset("indptr", data=indptr, chunks=(min(indptr.size, 200000),), dtype="i8")
    lg.create_dataset("indices", shape=(nnz_total,), chunks=(min(nnz_total, 5_000_000),), dtype="i4")
    lg.create_dataset("data", shape=(nnz_total,), chunks=(min(nnz_total, 5_000_000),), dtype="f4")

    # Pass 2: fill indices/data sequentially
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

    print("[INFO] Universal zarr build complete.")


# ======================
# Open universal CSR + var names
# ======================
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


# ======================
# Build task rows/y/covariates from universal base (no rebuilding zarr)
# ======================
def build_task_from_universal_base(stripped_h5ad: str, base_orig_idx: np.ndarray, age_bin_base, sex_base, groups_base):
    adata_b = sc.read_h5ad(stripped_h5ad, backed="r")
    obs = adata_b.obs

    if COND_COL not in obs.columns:
        raise KeyError(f"Missing '{COND_COL}' in obs for task.")
    cond = pd.Series(to_pystr_array(obs.iloc[base_orig_idx][COND_COL]))
    y_series = cond.map(COND_MAP)
    mask_cond = ~y_series.isna()

    task_rows = np.where(mask_cond.to_numpy())[0].astype(np.int64)  # zarr row positions
    y = y_series[mask_cond].astype(np.int8).to_numpy()

    age_bin_task = age_bin_base[mask_cond].reset_index(drop=True)
    sex_task = sex_base[mask_cond].reset_index(drop=True)

    if groups_base is not None:
        groups_task = groups_base[mask_cond].reset_index(drop=True)
    else:
        groups_task = None

    try:
        adata_b.file.close()
    except Exception:
        pass

    print(f"[INFO] Task cells (condition-mapped) from universal base: {task_rows.size} / {base_orig_idx.size}")
    print_counts("y_task", y)

    return task_rows, y, age_bin_task, sex_task, groups_task


def build_covariates(age_bin: pd.Series, sex_series: pd.Series):
    age_dum = pd.get_dummies(age_bin, prefix="", prefix_sep="")
    for c in AGE_BIN_COLS:
        if c not in age_dum.columns:
            age_dum[c] = 0
    age_dum = age_dum[AGE_BIN_COLS]
    age_mat = age_dum.values.astype(np.float32, copy=False)
    sex_num = sex_series.apply(encode_sex).to_numpy(dtype=np.float32)
    return age_mat, sex_num


# ======================
# Split helpers (on task rows)
# ======================
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
            print(f"[INFO] StratifiedGroupSplit success on try {t}")
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

    print("[INFO] Stratified random split.")
    print_counts("y_train", y[tr])
    print_counts("y_test", y[te])
    return tr, te


# ======================
# Fast CSR read helpers (OPTIMIZED: cached indptr + batch I/O)
# ======================

# Global indptr cache to avoid repeated zarr reads of the same array
_INDPTR_CACHE = {}


def get_cached_indptr(lg) -> np.ndarray:
    """
    Cache the full indptr array in memory.
    This turns all subsequent row lookups into numpy array indexing (zero I/O).
    """
    key = id(lg)
    if key not in _INDPTR_CACHE:
        print("[INFO] Caching full indptr array in memory ...")
        _INDPTR_CACHE[key] = np.asarray(lg["indptr"][:], dtype=np.int64)
        print(f"[INFO] indptr cached: {_INDPTR_CACHE[key].shape[0]} entries, "
              f"{_INDPTR_CACHE[key].nbytes / 1e6:.1f} MB")
    return _INDPTR_CACHE[key]


def fill_dense_block_contiguous(lg, rs: int, re: int, G: int) -> np.ndarray:
    """
    Read contiguous rows [rs, re) from zarr CSR and return dense (re-rs, G) matrix.
    Uses cached indptr + 2 zarr reads (indices, data).
    """
    indptr_cached = get_cached_indptr(lg)
    indices_arr = lg["indices"]
    data_arr = lg["data"]

    B = re - rs
    ip = indptr_cached[rs:re + 1]
    data_start = int(ip[0])
    data_end = int(ip[-1])

    if data_end <= data_start:
        return np.zeros((B, G), dtype=np.float32)

    all_idx = np.asarray(indices_arr[data_start:data_end], dtype=np.int32)
    all_dat = np.asarray(data_arr[data_start:data_end], dtype=np.float32)

    ip_local = ip - data_start
    csr = sparse.csr_matrix((all_dat, all_idx, ip_local), shape=(B, G))
    return csr.toarray()


def fill_dense_block_for_row_list(lg, rows_u: np.ndarray, G: int) -> np.ndarray:
    """
    Read arbitrary (possibly non-contiguous) rows from zarr CSR.
    Detects contiguous segments and uses batch I/O for each segment.
    Returns dense (len(rows_u), G) matrix in the ORIGINAL row order.
    """
    indptr_cached = get_cached_indptr(lg)
    indices_arr = lg["indices"]
    data_arr = lg["data"]

    rows_u = np.asarray(rows_u, dtype=np.int64)
    n = rows_u.size
    if n == 0:
        return np.zeros((0, G), dtype=np.float32)

    Xout = np.zeros((n, G), dtype=np.float32)

    # Find contiguous segments
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


def fill_dense_for_scattered_rows(lg, rows_sorted: np.ndarray, G: int,
                                  superblock_size: int = 50000) -> np.ndarray:
    """
    Optimized for SCATTERED rows (e.g. 15k random rows across many cells).
    Instead of one zarr read per row, groups nearby rows into superblocks
    and does ONE large data/indices read per superblock.
    """
    indptr_cached = get_cached_indptr(lg)
    indices_arr = lg["indices"]
    data_arr = lg["data"]

    rows_sorted = np.asarray(rows_sorted, dtype=np.int64)
    n = rows_sorted.size
    if n == 0:
        return np.zeros((0, G), dtype=np.float32)

    Xout = np.zeros((n, G), dtype=np.float32)

    # Group rows into superblocks by row index proximity
    sb_start = 0
    while sb_start < n:
        first_row = int(rows_sorted[sb_start])
        upper_row = first_row + superblock_size

        # Find all requested rows within this superblock range
        sb_end = sb_start
        while sb_end < n and int(rows_sorted[sb_end]) < upper_row:
            sb_end += 1

        sb_rows = rows_sorted[sb_start:sb_end]
        sb_n = sb_end - sb_start

        # Get data range for the entire superblock (from cached indptr)
        row_starts = indptr_cached[sb_rows]
        row_ends = indptr_cached[sb_rows + 1]
        data_start = int(row_starts.min())
        data_end = int(row_ends.max())

        if data_end > data_start:
            # ONE read for all data/indices in this superblock
            all_idx = np.asarray(indices_arr[data_start:data_end], dtype=np.int32)
            all_dat = np.asarray(data_arr[data_start:data_end], dtype=np.float32)

            # Build CSR for only the requested rows
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


# ======================
# CSR ops on contiguous rows (zarr CSR) -- gene-level stats
# ======================
def gene_stats_on_sorted_rows_contiguous(lg, rows_sorted: np.ndarray, n_genes: int, block_rows: int):
    rows_sorted = np.asarray(rows_sorted, dtype=np.int64)
    gene_sum = np.zeros(n_genes, dtype=np.float64)
    gene_sum2 = np.zeros(n_genes, dtype=np.float64)

    if rows_sorted.size == 0:
        return gene_sum, gene_sum2

    # Find contiguous segments
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

            # Use cached indptr (no zarr I/O)
            ip = indptr_cached[rs:re + 1]
            nnz_s = int(ip[0])
            nnz_e = int(ip[-1])
            if nnz_e <= nnz_s:
                continue

            # Batch read data/indices
            idx = np.asarray(indices_arr[nnz_s:nnz_e], dtype=np.int32)
            dat = np.asarray(data_arr[nnz_s:nnz_e], dtype=np.float32)

            gene_sum += np.bincount(idx, weights=dat.astype(np.float64), minlength=n_genes)
            gene_sum2 += np.bincount(idx, weights=(dat.astype(np.float64) ** 2), minlength=n_genes)

        gc.collect()

    return gene_sum, gene_sum2


# ======================
# Effect score for all genes (task train only)
# ======================
def supervised_effect_score_all_genes(lg, n_genes: int, tr_rows: np.ndarray, y_tr: np.ndarray):
    tr_rows = np.asarray(tr_rows, dtype=np.int64)
    y_tr = np.asarray(y_tr, dtype=np.int8)

    tr0 = tr_rows[y_tr == 0]
    tr1 = tr_rows[y_tr == 1]
    tr0 = np.asarray(tr0, dtype=np.int64)
    tr1 = np.asarray(tr1, dtype=np.int64)
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


# ======================
# Corr pruning (OPTIMIZED: superblock-based scattered reads)
# ======================
def corr_on_train_subsample_subset(lg, tr_rows: np.ndarray, gene_idx: np.ndarray, max_cells: int, block_rows: int):
    rng = np.random.RandomState(RANDOM_STATE)

    tr_rows = np.asarray(tr_rows, dtype=np.int64)
    if tr_rows.size > max_cells:
        rows = rng.choice(tr_rows, size=max_cells, replace=False)
        print(f"[INFO] corr rows subsample: {max_cells}/{tr_rows.size}")
    else:
        rows = tr_rows
        print(f"[INFO] corr rows subsample: all {tr_rows.size}")

    rows = np.asarray(rows, dtype=np.int64)
    rows.sort()

    gene_idx = np.asarray(gene_idx, dtype=np.int32)
    Gsub = int(gene_idx.size)
    Gall = int(lg.attrs["shape"][1])

    sum_x = np.zeros(Gsub, dtype=np.float64)
    sum_x2 = np.zeros(Gsub, dtype=np.float64)
    XtX = np.zeros((Gsub, Gsub), dtype=np.float64)

    # ---- OPTIMIZED: superblock-based scattered reads ----
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
        print(f"[INFO] corr processed rows {processed}/{n_rows}")

        del Xb_full, Xb, xb64
        gc.collect()

    n = float(rows.size)
    mean = sum_x / n
    mean2 = sum_x2 / n
    var = mean2 - mean * mean
    var[var < 1e-12] = 1e-12
    std = np.sqrt(var)

    Exy = XtX / n
    cov = Exy - np.outer(mean, mean)
    corr = cov / np.outer(std, std)
    corr = np.clip(corr, -1.0, 1.0)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)

    return corr.astype(np.float32)


def prune_by_abs_corr_greedy(gene_names: np.ndarray, scores_full: np.ndarray, corr: np.ndarray, thr: float):
    n = scores_full.size
    order = np.argsort(scores_full)[::-1]
    covered = np.zeros(n, dtype=bool)

    kept_pos = []
    map_rows = []
    abs_corr = np.abs(corr)
    thr = float(thr)

    for pos in order:
        if covered[pos]:
            continue
        kept_pos.append(pos)

        m = abs_corr[pos, :] >= thr
        m[pos] = True
        covered[m] = True

        removed = np.where(m)[0].astype(np.int32)
        removed = removed[removed != pos]
        rep_name = str(gene_names[pos])
        removed_names = [str(gene_names[j]) for j in removed.tolist()]
        map_rows.append({
            "representative": rep_name,
            "rep_score": float(scores_full[pos]),
            "cluster_size": int(1 + removed.size),
            "removed_genes": ",".join(removed_names),
        })

    kept_pos = np.array(kept_pos, dtype=np.int32)
    if kept_pos.size < MIN_PRUNED_GENES:
        top2 = np.argsort(scores_full)[::-1][:min(MIN_PRUNED_GENES, n)]
        kept_pos = top2.astype(np.int32)
        print(f"[WARN] Pruning too aggressive, force keep top {kept_pos.size} by score.")

    pruned_names = gene_names[kept_pos].astype(object)

    pd.DataFrame(map_rows).to_csv(OUT_PRUNING_MAP, sep="\t", index=False)
    with open(OUT_PRUNED_GENES, "w", encoding="utf-8") as f:
        for g in pruned_names.tolist():
            f.write(str(g) + "\n")

    print(f"[INFO] Pruned genes kept: {kept_pos.size} (thr={thr})")
    return kept_pos.astype(np.int32), pruned_names


# ======================
# Sparsity calibration + stability selection
# ======================
def pick_C_for_target_sparsity(Xmm, tr_rows: np.ndarray, cov_age: np.ndarray, cov_sex: np.ndarray, y: np.ndarray, groups: np.ndarray | None):
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
    nnz_by_C = []

    for C in C_GRID:
        clf = LogisticRegression(
            max_iter=8000, solver="saga",
            penalty="elasticnet", l1_ratio=float(L1_RATIO_SEL),
            C=float(C), class_weight="balanced", tol=1e-3,
        )
        clf.fit(X, y_sub)
        beta = clf.coef_.ravel()[:Xg.shape[1]]
        nnz = int(np.sum(np.abs(beta) > 1e-8))
        nnz_by_C.append((float(C), nnz))
        diff = abs(nnz - int(TARGET_NNZ_FOR_SELECTION))
        if diff < best_diff:
            best_diff = diff
            best_C = float(C)

    print("[INFO] Sparsity calibration (C -> nnz_genes):")
    print(nnz_by_C)
    print(f"[INFO] Picked C for stability selection: {best_C} (target nnz={TARGET_NNZ_FOR_SELECTION})")
    return float(best_C)


def stability_selection(Xmm, tr_rows: np.ndarray, cov_age: np.ndarray, cov_sex: np.ndarray, y: np.ndarray,
                        gene_names: np.ndarray, groups: np.ndarray | None, C_sel: float):
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
            print("[WARN] Too few groups in TRAIN for group-aware subsampling; fallback to cell subsampling.")
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
            max_iter=8000, solver="saga",
            penalty="elasticnet", l1_ratio=float(L1_RATIO_SEL),
            C=float(C_sel), class_weight="balanced", tol=1e-3,
        )
        clf.fit(X, y_sub)
        beta = clf.coef_.ravel()[:G]

        sel = np.abs(beta) > 1e-8
        freq[sel] += 1
        sign_sum[sel] += np.where(beta[sel] > 0, 1, -1)
        beta_sum[sel] += beta[sel].astype(np.float64)
        beta_abs_sum[sel] += np.abs(beta[sel]).astype(np.float64)

        if t % 10 == 0:
            print(f"[INFO] stability repeat {t}/{N_REPEATS}, nnz_genes={int(sel.sum())}")

        del Xg, Xa, Xs, X, clf
        gc.collect()

    freq_f = freq.astype(np.float64) / float(N_REPEATS)
    mean_beta = np.where(freq > 0, beta_sum / np.maximum(freq, 1), 0.0)
    mean_abs_beta = np.where(freq > 0, beta_abs_sum / np.maximum(freq, 1), 0.0)
    sign_consistency = np.where(freq > 0, np.abs(sign_sum) / np.maximum(freq, 1), 0.0)

    df = pd.DataFrame({
        "gene": gene_names.astype(object),
        "selection_freq": freq_f,
        "sign_consistency": sign_consistency,
        "mean_beta": mean_beta,
        "mean_abs_beta": mean_abs_beta,
    }).sort_values(["selection_freq", "mean_abs_beta"], ascending=[False, False])

    df.to_csv(OUT_STABILITY, sep="\t", index=False)
    print(f"[INFO] Wrote stability table: {OUT_STABILITY}")
    return df


# ======================
# Export ROC-needed data files
# ======================
def export_roc_data(workdir: str, te_rows_task: np.ndarray, te_orig_idx: np.ndarray, y_te: np.ndarray,
                    y_proba: np.ndarray, meta: dict):
    roc_dir = os.path.join(workdir, "roc_data")
    os.makedirs(roc_dir, exist_ok=True)

    df_labels = pd.DataFrame({
        "task_row_index": te_rows_task.astype(np.int64),
        "orig_cell_index": te_orig_idx.astype(np.int64),
        "y_true": y_te.astype(int),
    })
    df_labels.to_csv(os.path.join(roc_dir, "test_labels.tsv"), sep="\t", index=False)

    df_model = pd.DataFrame({
        "task_row_index": te_rows_task.astype(np.int64),
        "orig_cell_index": te_orig_idx.astype(np.int64),
        "model_score": y_proba.astype(float),
    })
    df_model.to_csv(os.path.join(roc_dir, "model_scores.tsv"), sep="\t", index=False)

    with open(os.path.join(roc_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Exported ROC data to: {roc_dir}")


# ======================
# MAIN
# ======================
def main():
    strip_h5ad_to_x_obs_var(H5AD_PATH, STRIPPED_H5AD_PATH)

    base_orig_idx, age_bin_base, sex_base, groups_base, group_col = build_base_kept_from_stripped(STRIPPED_H5AD_PATH)

    gene_counts, gene_ncells = compute_gene_stats_on_rows(STRIPPED_H5AD_PATH, base_orig_idx, CHUNK_CELLS)
    min_cells = max(1, int(base_orig_idx.size * float(MIN_GENE_CELLS_FRAC)))
    gene_mask = (gene_counts >= float(MIN_GENE_COUNTS)) & (gene_ncells >= min_cells)
    print(f"[INFO] Universal gene_mask kept genes={int(gene_mask.sum())} / {gene_mask.size} (min_cells={min_cells})")

    build_universal_zarr_if_missing(
        stripped_h5ad=STRIPPED_H5AD_PATH,
        zarr_path=ZARR_PATH,
        base_orig_idx=base_orig_idx,
        gene_mask=gene_mask,
        target_sum=TARGET_SUM,
        chunk_cells=CHUNK_CELLS,
    )

    zroot, lg, n_cells_z, n_genes_z = open_log1p_csr(ZARR_PATH)
    var_names = load_var_names(zroot)
    obs_orig_index_in_zarr = load_obs_orig_index(zroot)

    print(f"[INFO] Universal log1p shape: cells={n_cells_z}, genes={n_genes_z}")
    if n_cells_z != int(base_orig_idx.size):
        raise RuntimeError("Zarr cells mismatch base kept cells.")
    if n_genes_z != int(gene_mask.sum()):
        raise RuntimeError("Zarr genes mismatch universal gene_mask.")

    task_rows, y, age_bin_task, sex_task, groups_task = build_task_from_universal_base(
        stripped_h5ad=STRIPPED_H5AD_PATH,
        base_orig_idx=base_orig_idx,
        age_bin_base=age_bin_base,
        sex_base=sex_base,
        groups_base=groups_base,
    )

    n_task = int(task_rows.size)
    if n_task < 200:
        raise RuntimeError("Too few task cells after condition mapping.")

    age_mat, sex_num = build_covariates(age_bin_task, sex_task)

    if groups_task is not None and USE_GROUP_SPLIT:
        groups_arr = groups_task.to_numpy(dtype=object)
        tr_i, te_i = stratified_group_split_local(y=y, groups=groups_arr, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    else:
        groups_arr = None
        tr_i, te_i = stratified_random_split_local(y=y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    tr_rows_u = task_rows[tr_i].astype(np.int64)
    te_rows_u = task_rows[te_i].astype(np.int64)

    y_tr, y_te = y[tr_i], y[te_i]
    print_counts("y_train", y_tr)
    print_counts("y_test", y_te)

    score_all = supervised_effect_score_all_genes(lg=lg, n_genes=n_genes_z, tr_rows=tr_rows_u, y_tr=y_tr)

    all_gene_idx = np.arange(n_genes_z, dtype=np.int32)
    corr = corr_on_train_subsample_subset(
        lg=lg,
        tr_rows=tr_rows_u,
        gene_idx=all_gene_idx,
        max_cells=MAX_CORR_CELLS,
        block_rows=CORR_BLOCK_ROWS,
    )

    pruned_pos, pruned_names = prune_by_abs_corr_greedy(
        gene_names=var_names.astype(object),
        scores_full=score_all,
        corr=corr,
        thr=PRUNE_ABS_CORR,
    )
    del corr
    gc.collect()

    # 8) Build task-local Xmm (n_task x n_pruned) with OPTIMIZED batch I/O
    if os.path.exists(XMM_PATH):
        os.remove(XMM_PATH)
    Xmm = np.memmap(XMM_PATH, dtype=np.float32, mode="w+", shape=(n_task, int(pruned_pos.size)))

    # mean/std on TRAIN (task-local) for pruned genes
    print("[INFO] Compute mean/std on TRAIN rows (task-local) for pruned genes ...")
    sum_x = np.zeros(pruned_pos.size, dtype=np.float64)
    sum_x2 = np.zeros(pruned_pos.size, dtype=np.float64)
    n_tr = int(tr_i.size)
    chunk = 20000

    for s in range(0, n_tr, chunk):
        e = min(n_tr, s + chunk)
        rows_u = task_rows[tr_i[s:e]]
        # ---- OPTIMIZED: batch I/O ----
        Xb_full = fill_dense_block_for_row_list(lg, rows_u, n_genes_z)
        Xb = Xb_full[:, pruned_pos].astype(np.float64, copy=False)
        sum_x += Xb.sum(axis=0)
        sum_x2 += (Xb * Xb).sum(axis=0)
        del Xb_full, Xb
        gc.collect()

    mean = (sum_x / float(n_tr)).astype(np.float32)
    var = (sum_x2 / float(n_tr)) - (mean.astype(np.float64) ** 2)
    var[var < 1e-12] = 1e-12
    std = np.sqrt(var).astype(np.float32)

    print("[INFO] Writing task-local Xmm memmap (z-scored) ...")
    for s in range(0, n_task, chunk):
        e = min(n_task, s + chunk)
        rows_u = task_rows[s:e]
        # ---- OPTIMIZED: batch I/O ----
        Xb_full = fill_dense_block_for_row_list(lg, rows_u, n_genes_z)
        Xb = Xb_full[:, pruned_pos]
        Xb = (Xb - mean) / std
        if Z_CLIP is not None and float(Z_CLIP) > 0:
            np.clip(Xb, -float(Z_CLIP), float(Z_CLIP), out=Xb)
        Xmm[s:e, :] = Xb.astype(np.float32, copy=False)
        Xmm.flush()
        print(f"[INFO] write task rows {e - s} ({e}/{n_task})")
        del Xb_full, Xb
        gc.collect()

    C_sel = pick_C_for_target_sparsity(
        Xmm=Xmm,
        tr_rows=tr_i,
        cov_age=age_mat,
        cov_sex=sex_num,
        y=y,
        groups=(groups_arr if groups_arr is not None else None),
    )

    df_stab = stability_selection(
        Xmm=Xmm,
        tr_rows=tr_i,
        cov_age=age_mat,
        cov_sex=sex_num,
        y=y,
        gene_names=pruned_names.astype(object),
        groups=(groups_arr if groups_arr is not None else None),
        C_sel=C_sel,
    )

    df_panel = df_stab.sort_values(["selection_freq", "mean_abs_beta"], ascending=[False, False])
    panel_genes = df_panel.head(int(PANEL_K))["gene"].to_numpy(dtype=object)

    pruned_names_arr = pruned_names.astype(object)
    panel_mask = np.isin(pruned_names_arr, panel_genes)
    panel_cols = np.where(panel_mask)[0].astype(np.int32)
    if panel_cols.size == 0:
        raise RuntimeError("Panel selection failed: no genes selected.")
    print(f"[INFO] Final panel genes: K={panel_cols.size} (target={PANEL_K})")

    Xg_tr = np.asarray(Xmm[tr_i, :][:, panel_cols], dtype=np.float32)
    Xg_te = np.asarray(Xmm[te_i, :][:, panel_cols], dtype=np.float32)
    Xa_tr = age_mat[tr_i, :].astype(np.float32, copy=False)
    Xa_te = age_mat[te_i, :].astype(np.float32, copy=False)
    Xs_tr = sex_num[tr_i].reshape(-1, 1).astype(np.float32, copy=False)
    Xs_te = sex_num[te_i].reshape(-1, 1).astype(np.float32, copy=False)

    X_tr = np.concatenate([Xg_tr, Xa_tr, Xs_tr], axis=1)
    X_te = np.concatenate([Xg_te, Xa_te, Xs_te], axis=1)

    clf = LogisticRegression(
        max_iter=8000,
        solver=FINAL_SOLVER,
        penalty="elasticnet",
        l1_ratio=float(FINAL_L1_RATIO),
        C=float(FINAL_C),
        class_weight="balanced",
        tol=1e-4,
    )
    clf.fit(X_tr, y_tr)

    y_pred_train = clf.predict(X_tr)
    y_proba_train = clf.predict_proba(X_tr)[:, 1]
    y_pred = clf.predict(X_te)
    y_proba = clf.predict_proba(X_te)[:, 1]

    acc_train = float(accuracy_score(y_tr, y_pred_train))
    auc_train = safe_binary_auc(y_tr, y_proba_train)
    acc = float(accuracy_score(y_te, y_pred))
    auc_cell = safe_binary_auc(y_te, y_proba)
    train_report = make_classification_report(y_tr, y_pred_train)
    test_report = make_classification_report(y_te, y_pred)

    if PRINT_TRAIN_METRICS:
        print("\n===== TRAIN METRICS (REAL MODEL) =====")
        print(f"Cell-level Accuracy (train): {acc_train:.4f}")
        print(f"Cell-level ROC-AUC  (train): {auc_train:.4f}")
        print(train_report)

    print("\n===== TEST METRICS (REAL MODEL) =====")
    print(f"Cell-level Accuracy (test): {acc:.4f}")
    print(f"Cell-level ROC-AUC  (test): {auc_cell:.4f}")
    print(test_report)

    sample_cv_result = None
    if RUN_SAMPLE_LEVEL_CV:
        if groups_arr is None:
            print("[WARN] Sample-level CV skipped: no group column available.")
        else:
            print("\n===== SAMPLE-LEVEL CV (FIXED FINAL PANEL) =====")
            X_all_final = np.concatenate([
                np.asarray(Xmm[:, panel_cols], dtype=np.float32),
                age_mat.astype(np.float32, copy=False),
                sex_num.reshape(-1, 1).astype(np.float32, copy=False),
            ], axis=1)
            sample_cv_result = run_sample_level_cv_fixed_panel(
                X_all=X_all_final,
                y_all=y,
                groups=groups_arr,
                group_col_name=(group_col if group_col is not None else "group"),
            )
            del X_all_final
            gc.collect()

            if sample_cv_result is not None:
                print(f"Mode: {sample_cv_result['mode']}")
                print(f"Groups: {sample_cv_result['n_groups']} (Class0={sample_cv_result['n_neg_groups']}, Class1={sample_cv_result['n_pos_groups']})")
                print(f"Sample-level Accuracy: {sample_cv_result['acc']:.4f}")
                print(f"Sample-level ROC-AUC : {sample_cv_result['auc']:.4f}")
                print(sample_cv_result["report"])

    if EXPORT_ROC_DATA:
        te_rows_task = te_i.astype(np.int64)
        te_rows_u = task_rows[te_rows_task]
        te_orig_idx = obs_orig_index_in_zarr[te_rows_u]

        export_roc_data(
            workdir=WORKDIR,
            te_rows_task=te_rows_task,
            te_orig_idx=te_orig_idx,
            y_te=y_te,
            y_proba=y_proba,
            meta={
                "workdir": WORKDIR,
                "zarr_path": ZARR_PATH,
                "stripped_h5ad": STRIPPED_H5AD_PATH,
                "cond_col": COND_COL,
                "cond_map": COND_MAP,
                "base_keep_rule": "age_parsed_only (universal)",
                "universal_n_cells": int(n_cells_z),
                "universal_n_genes": int(n_genes_z),
                "task_n_cells": int(n_task),
                "n_pruned": int(pruned_pos.size),
                "panel_k": int(panel_cols.size),
                "stability_C_selected": float(C_sel),
                "selection_l1_ratio": float(L1_RATIO_SEL),
                "target_nnz_for_selection": int(TARGET_NNZ_FOR_SELECTION),
                "prune_abs_corr": float(PRUNE_ABS_CORR),
                "auc_cell": float(auc_cell),
                "roc_export_mode": "model_only",
                "age_bins": AGE_BIN_COLS,
            }
        )

    beta = clf.coef_.ravel().astype(float)
    intercept = float(clf.intercept_.ravel()[0])

    feat_names = list(panel_genes.astype(str)) + AGE_BIN_COLS + ["sex"]
    df_coef = pd.DataFrame({"feature": feat_names, "beta": beta})
    df_coef.loc[len(df_coef)] = {"feature": "Intercept", "beta": intercept}

    df_coef.to_csv(OUT_PANEL, sep="\t", index=False)
    print(f"[INFO] Wrote final panel table: {OUT_PANEL}")

    metrics = {
        "train_acc_cell": acc_train,
        "train_auc_cell": auc_train,
        "acc": acc,
        "auc_cell": auc_cell,
        "universal_zarr": ZARR_PATH,
        "task_cond_map": COND_MAP,
        "universal_n_cells": int(n_cells_z),
        "universal_n_genes": int(n_genes_z),
        "task_n_cells": int(n_task),
        "n_pruned": int(pruned_pos.size),
        "panel_k": int(panel_cols.size),
        "workdir": WORKDIR,
        "age_bins": AGE_BIN_COLS,
        "group_col": group_col,
    }
    if sample_cv_result is not None:
        metrics["sample_level_cv"] = {
            "mode": sample_cv_result["mode"],
            "n_splits": int(sample_cv_result["n_splits"]),
            "n_groups": int(sample_cv_result["n_groups"]),
            "n_pos_groups": int(sample_cv_result["n_pos_groups"]),
            "n_neg_groups": int(sample_cv_result["n_neg_groups"]),
            "acc": float(sample_cv_result["acc"]),
            "auc": float(sample_cv_result["auc"]),
            "predictions_tsv": OUT_SAMPLE_CV,
        }
    with open(OUT_METRICS, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("===== TRAIN METRICS (REAL MODEL) =====\n")
        f.write(f"Accuracy: {acc_train:.6f}\n")
        f.write(f"AUC: {auc_train:.6f}\n\n")
        f.write(train_report)
        f.write("\n\n")
        f.write("===== TEST METRICS (REAL MODEL) =====\n")
        f.write(f"Accuracy: {acc:.6f}\n")
        f.write(f"AUC: {auc_cell:.6f}\n\n")
        f.write(test_report)
        f.write("\n")
        if sample_cv_result is not None:
            f.write("\n===== SAMPLE-LEVEL CV (FIXED FINAL PANEL) =====\n")
            f.write(f"Mode: {sample_cv_result['mode']}\n")
            f.write(f"Groups: {sample_cv_result['n_groups']} (Class0={sample_cv_result['n_neg_groups']}, Class1={sample_cv_result['n_pos_groups']})\n")
            f.write(f"Accuracy: {sample_cv_result['acc']:.6f}\n")
            f.write(f"AUC: {sample_cv_result['auc']:.6f}\n\n")
            f.write(sample_cv_result["report"])
            f.write("\n")

    del Xmm
    gc.collect()


if __name__ == "__main__":
    main()
