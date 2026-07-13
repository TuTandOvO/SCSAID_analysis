#!/usr/bin/env python3
"""
Leiden Clustering + Filtering Pipeline
=======================================
1. 读取已整合好的 AnnData（需要有 neighbors graph 或 scVI embedding）
2. 对多个 resolution 计算 leiden clustering
3. 过滤掉在任一 resolution 中属于小 cluster（<阈值）的细胞
4. 保存过滤后的 AnnData（包含所有 resolution 的聚类结果）
"""

import os
import numpy as np
import scanpy as sc

# ==============================================================================
# 参数配置
# ==============================================================================

# 输入 / 输出路径
IN_PATH = "/gpfsdata/home/renyixiang/SkinDB/10X/human/Annotation/human_int_V3.h5ad"
OUT_PATH = "/gpfsdata/home/renyixiang/SkinDB/10X/human/Annotation/human_with_all_leiden_10threshold.h5ad"

# Leiden 聚类参数
RES_LIST = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
MIN_CLUSTER_SIZE = 10  # 小于此阈值的 cluster 中的细胞将被过滤

# 如果使用 scVI latent space，需要先计算 neighbors
# 设置 scVI latent representation 的 key（如果有的话）
SCVI_LATENT_KEY = "X_scVI"  # 如果没有使用 scVI，设为 None
N_NEIGHBORS = 15  # neighbors 数量
N_PCS = None  # 如果使用 scVI latent，设为 None；如果使用 PCA，设为具体数字如 50


# ==============================================================================
# 辅助函数
# ==============================================================================

def res_to_key(res: float) -> str:
    """
    将 resolution 转换为列名
    例如：0.6 -> 'leiden_scVI_0_6'；1.0 -> 'leiden_scVI_1'
    """
    res_str = str(res).replace('.', '_')
    if float(res).is_integer():
        res_str = str(int(res))
    return f"leiden_scVI_{res_str}"


def compute_neighbors_if_needed(adata, use_rep=None, n_neighbors=15, n_pcs=None):
    """
    检查是否已有 neighbors graph，如果没有则计算
    """
    if 'neighbors' in adata.uns and 'connectivities' in adata.obsp:
        print("[INFO] Neighbors graph already exists, skipping computation.")
        return
    
    print("[INFO] Computing neighbors graph...")
    if use_rep is not None:
        print(f"[INFO] Using representation: {use_rep}")
        sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=n_neighbors)
    else:
        print(f"[INFO] Using PCA with n_pcs={n_pcs}")
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    print("[INFO] Neighbors graph computed.")


def compute_leiden_all_resolutions(adata, res_list):
    """
    对所有指定的 resolution 计算 leiden 聚类
    """
    print(f"\n[INFO] Computing Leiden clustering for {len(res_list)} resolutions...")
    
    for res in res_list:
        key = res_to_key(res)
        print(f"[INFO] Computing Leiden at resolution {res} -> {key}")
        
        sc.tl.leiden(
            adata,
            resolution=res,
            key_added=key,
            flavor="igraph",  # 或 "leidenalg"，取决于你的环境
            n_iterations=2,   # -1 表示直到收敛
            directed=False
        )
        
        n_clusters = adata.obs[key].nunique()
        print(f"[INFO]   -> Found {n_clusters} clusters")
    
    print("[INFO] All Leiden clusterings computed.\n")


def filter_small_clusters(adata, res_list, min_size=10):
    """
    过滤掉在任一 resolution 中属于小 cluster 的细胞
    返回过滤后的 adata
    """
    print(f"[INFO] Filtering cells in clusters with < {min_size} cells...")
    
    # 全局保留 mask
    keep_mask = np.ones(adata.n_obs, dtype=bool)
    
    for res in res_list:
        key = res_to_key(res)
        
        if key not in adata.obs.columns:
            print(f"[WARN] {key} not found in adata.obs, skipping...")
            continue
        
        labels = adata.obs[key].astype("category")
        sizes = labels.value_counts()
        
        # 找出小 cluster
        small_clusters = sizes[sizes < min_size].index.tolist()
        
        if len(small_clusters) == 0:
            print(f"[INFO] Resolution {res}: No small clusters found.")
            continue
        
        # 统计信息
        n_small = len(small_clusters)
        is_small = labels.isin(small_clusters)
        n_cells_small = int(is_small.sum())
        
        print(f"[INFO] Resolution {res}: {n_small} small clusters, {n_cells_small} cells to mark")
        
        # 更新全局 mask
        keep_mask &= ~is_small
    
    # 统计过滤结果
    n_total = adata.n_obs
    n_kept = int(keep_mask.sum())
    n_removed = n_total - n_kept
    
    print(f"\n[INFO] Filtering Summary:")
    print(f"[INFO]   Total cells before: {n_total:,}")
    print(f"[INFO]   Cells kept:         {n_kept:,}")
    print(f"[INFO]   Cells removed:      {n_removed:,} ({n_removed/n_total*100:.2f}%)")
    
    # 应用过滤
    adata_filtered = adata[keep_mask].copy()
    
    return adata_filtered


def print_cluster_summary(adata, res_list):
    """
    打印每个 resolution 的聚类统计信息
    """
    print("\n" + "="*60)
    print("CLUSTER SUMMARY (after filtering)")
    print("="*60)
    
    for res in res_list:
        key = res_to_key(res)
        if key not in adata.obs.columns:
            continue
        
        labels = adata.obs[key]
        n_clusters = labels.nunique()
        sizes = labels.value_counts()
        
        print(f"\nResolution {res} ({key}):")
        print(f"  Number of clusters: {n_clusters}")
        print(f"  Cluster sizes: min={sizes.min()}, max={sizes.max()}, median={sizes.median():.0f}")


# ==============================================================================
# 主流程
# ==============================================================================

def main():
    # 1. 读取数据
    print("="*60)
    print("STEP 1: Loading Data")
    print("="*60)
    print(f"[INFO] Reading AnnData from: {IN_PATH}")
    adata = sc.read_h5ad(IN_PATH)
    print(f"[INFO] AnnData shape: {adata.shape} (cells × genes)")
    
    # 2. 计算 neighbors（如果需要）
    print("\n" + "="*60)
    print("STEP 2: Computing Neighbors Graph")
    print("="*60)
    
    use_rep = SCVI_LATENT_KEY if SCVI_LATENT_KEY and SCVI_LATENT_KEY in adata.obsm else None
    if use_rep:
        print(f"[INFO] Found scVI latent representation: {use_rep}")
    else:
        print("[INFO] No scVI latent found, will use PCA")
    
    compute_neighbors_if_needed(adata, use_rep=use_rep, n_neighbors=N_NEIGHBORS, n_pcs=N_PCS)
    
    # 3. 计算所有 resolution 的 Leiden 聚类
    print("\n" + "="*60)
    print("STEP 3: Computing Leiden Clustering")
    print("="*60)
    compute_leiden_all_resolutions(adata, RES_LIST)
    
    # 4. 过滤小 cluster 中的细胞
    print("\n" + "="*60)
    print("STEP 4: Filtering Small Clusters")
    print("="*60)
    adata_filtered = filter_small_clusters(adata, RES_LIST, min_size=MIN_CLUSTER_SIZE)
    
    # 5. 打印聚类统计
    print_cluster_summary(adata_filtered, RES_LIST)
    
    # 6. 保存结果
    print("\n" + "="*60)
    print("STEP 5: Saving Results")
    print("="*60)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    adata_filtered.write_h5ad(OUT_PATH)
    print(f"[INFO] Saved filtered AnnData to: {OUT_PATH}")
    print(f"[INFO] Final shape: {adata_filtered.shape}")
    
    # 列出保存的 leiden 列
    leiden_cols = [col for col in adata_filtered.obs.columns if col.startswith("leiden_")]
    print(f"[INFO] Saved Leiden columns: {leiden_cols}")
    
    print("\n[INFO] Pipeline completed successfully!")


if __name__ == "__main__":
    main()
