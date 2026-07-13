import os
import scanpy as sc
import pandas as pd

#========================
# 基本路径
#========================
in_file = "/gpfsdata/home/renyixiang/SkinDB/10X/human/Annotation/human_with_all_leiden_10threshold.h5ad"
out_dir = "/gpfsdata/home/renyixiang/SkinDB/10X/human/Annotation/Annotated"
os.makedirs(out_dir, exist_ok=True)

out_h5ad = os.path.join(
    out_dir,
    "human_with_all_leiden_10threshold_annotated.h5ad"
)

#========================
# 1. 读入数据
#========================
adata = sc.read_h5ad(in_file)

#========================
# 2. 删除 “leiden_scVI_0_8” 中的 cluster 4
#========================
col = "leiden_scVI_0_8"

# 确保是字符串，避免类型问题
adata.obs[col] = adata.obs[col].astype(str)

# 删除 cluster "4"
adata = adata[adata.obs[col] != "4", :].copy()

#========================
# 3. 按照映射新建 Fine_Map / Gross_Map
#========================
# 映射字典（key 一律用字符串）
cluster_map = {
    "0": ("cDC2 Dendritic Cell", "Immune"),
    "1": ("FCN1+ Monocyte", "Immune"),
    "2": ("cDC1 Dendritic Cell", "Immune"),
    "3": ("SPP1+ Inflammatory Macrophage", "Immune"),
    "5": ("Langerhans Cell", "Immune"),
    "6": ("Suprabasal Keratinocyte", "Keratinocyte"),
    "7": ("Hair Follicle Keratinocyte", "Keratinocyte"),
    "8": ("Stress Keratinocyte", "Keratinocyte"),
    "9": ("KRT1 KRT10+ Spinous Keratinocyte", "Keratinocyte"),
    "10": ("ACTA2+ Myoepithelial-like Keratinocyte", "Keratinocyte"),
    "11": ("KRTDAP+ DMKN+ Spinous Keratinocyte", "Keratinocyte"),
    "12": ("COL17A1+ KRT14+ Basal Keratinocyte", "Keratinocyte"),
    "13": ("Smooth Muscle Cell", "Smooth Muscle Cell"),
    "14": ("IL7R+ Naive CD4 T Cell", "Immune"),
    "15": ("CD8+ Cytotoxic T/NK Cells", "Immune"),
    "16": ("CD69+ IL7R+ Tissue-resident memory T Cell", "Immune"),
    "17": ("Resident T Cell", "Immune"),
    "18": ("Regulatory T Cell", "Immune"),
    "19": ("Cycling T Cell", "Immune"),
    "20": ("Matrix Remodeling Fibroblasts", "Fibroblast"),
    "21": ("KIR3DL2+ NK/T Cell", "Immune"),
    "22": ("Plasma Cell", "Immune"),
    "23": ("Erythrocyte", "Erythrocyte"),
    "24": ("CFD+ Secretory fibroblasts", "Fibroblast"),
    "25": ("ACKR1+ Post-Capillary Venule Endothelial Cell", "Vascular Endothelial Cell"),
    "26": ("Hair Follicle Outer Root Sheath Keratinocyte", "Keratinocyte"),
    "27": ("Mast Cell", "Immune"),
    "29": ("PECAM1+ Blood vascular Endothelial Cell", "Vascular Endothelial Cell"),
    "30": ("B Cell", "Immune"),
    "31": ("MKI67+ Granular Keratinocyte", "Keratinocyte"),
    "32": ("KRT2+ SLURP1+ Granular Keratinocyte", "Keratinocyte"),
    "34": ("LOR+ Cornified Keratinocyte", "Keratinocyte"),
    "36": ("S100A2+ Proliferating Keratinocyte", "Keratinocyte"),
    "37": ("Melanocyte", "Melanocyte"),
    "38": ("Eccrine Sweat Gland Cell", "Eccrine Sweat Gland Cell"),
    "39": ("MHC-II+ Stressed Suprabasal Keratinocyte", "Keratinocyte"),
    "40": ("Lymphatic Endothelial Cell", "Lymphatic Endothelial Cell"),
    "41": ("SPP1+ Inflammatory Macrophages", "Immune"),
    "42": ("Schwann Cell", "Schwann cell"),
    "43": ("IFN-activated Basal Keratinocyte", "Keratinocyte"),
    "44": ("Sebocyte", "Sebaceous Gland Cell"),
}

fine_map_dict  = {k: v[0] for k, v in cluster_map.items()}
gross_map_dict = {k: v[1] for k, v in cluster_map.items()}

adata.obs["Fine_Map"]  = adata.obs[col].map(fine_map_dict)
adata.obs["Gross_Map"] = adata.obs[col].map(gross_map_dict)

# 对于没有在表里的 cluster，用 “Unknown” 标记（如果你不想要可以删掉这几行）
adata.obs["Fine_Map"]  = adata.obs["Fine_Map"].fillna("Unknown")
adata.obs["Gross_Map"] = adata.obs["Gross_Map"].fillna("Unknown")

# 设成分类变量（方便画图排图例）
adata.obs["Fine_Map"]  = adata.obs["Fine_Map"].astype("category")
adata.obs["Gross_Map"] = adata.obs["Gross_Map"].astype("category")

#========================
# 4. 画 UMAP 图并保存（legend 在右侧）
#========================
sc.settings.figdir = out_dir  # 所有图都写到 Annotated 目录
sc.set_figure_params(figsize=(6, 5))

# Fine_Map
sc.pl.umap(
    adata,
    color="Fine_Map",
    legend_loc="right margin",  # 图例在右侧
    frameon=False,
    save="_Fine_Map.png"        # 会生成 umap_Fine_Map.png
)

# Gross_Map
sc.pl.umap(
    adata,
    color="Gross_Map",
    legend_loc="right margin",
    frameon=False,
    save="_Gross_Map.png"       # 会生成 umap_Gross_Map.png
)

#========================
# 5. 写出注释后的 adata
#========================
adata.write_h5ad(out_h5ad)

print("Done.")
print("Annotated h5ad:", out_h5ad)
print("Figures saved in:", out_dir)
