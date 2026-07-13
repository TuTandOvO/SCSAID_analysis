import os
import scanpy as sc
import pandas as pd

#========================
# 基本路径
#========================
in_file = "/gpfsdata/home/renyixiang/SkinDB/10X/mouse/Annotation/mouse_with_all_leiden_10threshold.h5ad"
out_dir = "/gpfsdata/home/renyixiang/SkinDB/10X/mouse/Annotation/Annotated"
os.makedirs(out_dir, exist_ok=True)

out_h5ad = os.path.join(
    out_dir,
    "mouse_with_all_leiden_10threshold_annotated.h5ad"
)

#========================
# 1. 读入数据
#========================
adata = sc.read_h5ad(in_file)

#========================
# 2. 删除 “leiden_scVI_0_7” 中的 cluster 10
#========================
col = "leiden_scVI_0_7"

# 确保是字符串，避免类型问题
adata.obs[col] = adata.obs[col].astype(str)

# 删除 cluster "10"
adata = adata[adata.obs[col] != "10", :].copy()

#========================
# 3. 按照映射新建 Fine_Map / Gross_Map
#========================
# 映射字典（key 一律用字符串）
cluster_map = {
    "0":  ("Ctss+ Lyz2+ Macrophage",                             "Immune"),
    "31": ("Adgre1+ Itgam+ Dermal Inf Macrophage",               "Immune"),
    "32": ("B Cell",                                             "Immune"),
    "13": ("cDC2 Dendritic Cell",                                "Immune"),
    "33": ("cDC1 Dendritic Cell",                                "Immune"),
    "42": ("Lst1+ Fcer1g+ Monocyte",                             "Immune"),
    "24": ("Langerhans Cell",                                    "Immune"),
    "23": ("Mast Cell",                                          "Immune"),
    "9":  ("Skap1+ Itk+ T Cell",                                 "Immune"),
    "28": ("Gamma-delta T Cell",                                 "Immune"),
    "25": ("Schwann cells",                                      "Schwann cells"),
    "45": ("Neutrophils (Erythroid marker mix)",                 "Immune"),
    "7":  ("S100a9+ S100a8+ Neutrophil",                         "Immune"),

    "1":  ("Krt5+ Krt14+ Basal Keratinocyte",                    "Keratinocyte"),
    "3":  ("Sprr1a+ Krt17+ Prolif Suprabasal Keratinocyte",      "Keratinocyte"),
    "26": ("Proliferating Cell",                                 "Proliferating Cell"),
    "2":  ("Ccl27a+ Pre-granular Suprabasal Keratinocyte",       "Keratinocyte"),
    "5":  ("Krt1+ Krt10+ Spinous Suprabasal Keratinocyte",       "Keratinocyte"),
    "20": ("Sprr1a+ Krt16+ Granular Suprabasal Keratinocyte",    "Keratinocyte"),
    "8":  ("Krt6a+ Krt14+ Metabolically Active Basal Keratinocyte", "Keratinocyte"),
    "34": ("Krt17+ Krt6a+ Migratory Keratinocyte",               "Keratinocyte"),
    "43": ("Lce1+ Lor+ Cornified Keratinocyte_1",                "Keratinocyte"),
    "41": ("Krt16+ Sprr1b+ Activated Keratinocyte",              "Keratinocyte"),
    "44": ("Sprr+ Rptn+ Cornified IFE",                          "Keratinocyte"),

    "35": ("Sensory Neuron",                                     "Sensory Neuron"),
    "6":  ("Bulge Hair-Follicle Stem Cell",                      "Bulge Hair-Follicle Stem Cell"),
    "14": ("Krt25/28+ Dlx3+ Lef1+ Inner Root Sheath Cell",       "Inner Root Sheath Cell"),

    "19": ("Vascular Endothelial Cell",                          "Vascular Endothelial Cell"),
    "21": ("Vascular Smooth Muscle Cell",                        "Vascular Smooth Muscle Cell"),
    "22": ("Melanocyte",                                         "Melanocyte"),
    "27": ("Sebaceous Gland Cell",                               "Sebaceous Gland Cell"),
    "29": ("Lymphatic Endothelial Cell",                         "Lymphatic Endothelial Cell"),

    "30": ("Skeletal Muscle Progenitor",                         "Skeletal Muscle Progenitor"),
    "37": ("Skeletal Muscle Cell",                               "Skeletal Muscle Cell"),
    "39": ("Erythrocyte",                                        "Erythrocyte"),
    "40": ("Lce1+ Lor+ Cornified Keratinocyte_2",                "Keratinocyte"),
    "38": ("Basophil",                                           "Immune"),

    "18": ("Hair Follicle Progenitor",                           "Hair Follicle Progenitor"),

    "4":  ("Dcn+ Fstl1+ Col1a2+ Dermal Fibroblast",              "Fibroblast"),
    "11": ("Dcn+ Lum+ Dermal Fibroblast",                        "Fibroblast"),
    "12": ("Fbn2+ Mfap2+ Papillary Dermal Fibroblast",           "Fibroblast"),
    "15": ("Crabp1+ Hmga2+ Prrx1+ Wound-Activated Fibroblast",   "Fibroblast"),
    "16": ("Prrx1+ Ebf1+ Fibroblast Progenitor",                 "Fibroblast"),
    "17": ("Dcn+ Pi16+ Fibroblast",                              "Fibroblast"),
    "36": ("Apod+ Ebf2+ Igfbp6+ Preadipocyte-like Dermal Fibroblast", "Fibroblast"),
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
