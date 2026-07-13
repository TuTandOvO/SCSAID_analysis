import os
import numpy as np
import scanpy as sc

# ==== 输入 / 输出路径 ====
in_path = "/gpfsdata/home/renyixiang/SkinDB/10X/mouse/Annotation/mouse_with_all_leiden.h5ad"
out_path = "/gpfsdata/home/renyixiang/SkinDB/10X/mouse/Annotation/mouse_with_all_leiden_10threshold.h5ad"

# ==== 读入 AnnData ====
print(f"[INFO] Reading AnnData from: {in_path}")
adata = sc.read_h5ad(in_path)
print(f"[INFO] Original adata shape: {adata.shape}")  # (n_cells, n_genes)

# ==== 需要检查的分辨率列表（跟你之前 DE 脚本保持一致）====
res_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# 与你之前脚本一致的命名方式：0.6 -> 'leiden_scVI_0_6'；1.0 -> 'leiden_scVI_1'
def res_to_key(res: float) -> str:
    res_str = str(res).replace('.', '_')
    if float(res).is_integer():
        res_str = str(int(res))
    return f"leiden_scVI_{res_str}"

# 全局保留 mask：只要在任一 resolution 中属于小 cluster（<10），就会被标记为 False
keep_mask = np.ones(adata.n_obs, dtype=bool)

for res in res_list:
    key = res_to_key(res)
    if key not in adata.obs.columns:
        print(f"[WARN] resolution={res} -> {key} not in adata.obs, skip this resolution.")
        continue

    print(f"\n[INFO] Processing resolution {res} ({key})")

    # 当前 resolution 的 cluster 标签
    labels = adata.obs[key].astype("category")
    sizes = labels.value_counts()  # Series: index=cluster label, value=cell count

    # 找出细胞数 < 10 的 cluster
    small_clusters = sizes[sizes < 10].index.tolist()
    n_small_clusters = len(small_clusters)

    if n_small_clusters == 0:
        print("[INFO] No clusters with <10 cells at this resolution.")
        continue

    print(f"[INFO] Found {n_small_clusters} clusters with <10 cells at resolution {res}.")
    print(f"[INFO] Small clusters: {small_clusters}")

    # 标记在这些 small clusters 中的细胞
    is_small = labels.isin(small_clusters)  # True 表示在小 cluster 中
    n_cells_to_drop_here = int(is_small.sum())
    print(f"[INFO] Cells in small clusters at this resolution: {n_cells_to_drop_here}")

    # 在全局 keep_mask 中，把这些细胞标记为 False
    keep_mask &= ~is_small

# ==== 应用过滤 ====
n_kept = int(keep_mask.sum())
n_total = adata.n_obs
n_removed = n_total - n_kept

print(f"\n[INFO] Total cells before filtering: {n_total}")
print(f"[INFO] Total cells after  filtering: {n_kept}")
print(f"[INFO] Total cells removed          : {n_removed}")

adata_filtered = adata[keep_mask].copy()
print(f"[INFO] Filtered adata shape: {adata_filtered.shape}")

# ==== 写出新的 h5ad ====
os.makedirs(os.path.dirname(out_path), exist_ok=True)
adata_filtered.write_h5ad(out_path)
print(f"[INFO] Saved filtered AnnData to: {out_path}")
