import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

adata = sc.read_h5ad('/gpfsdata/home/renyixiang/SkinDB/10X/human/Annotation/human_with_all_leiden_10threshold.h5ad')

tabledir = "/gpfsdata/home/renyixiang/SkinDB/10X/human/Annotation/DE_tables"
os.makedirs(tabledir, exist_ok=True)

########################################
# 关键：确保有 log1p 数据可用于 DE
########################################

# 如果已经有 log1p，就直接用
if "log1p" in adata.layers:
    print("[INFO] Found existing layer 'log1p', will use it for DE.")
else:
    # 优先从 counts 生成 log1p
    if "counts" in adata.layers:
        print("[INFO] No 'log1p' layer found. Creating 'log1p' from 'counts'...")
        # 注意 copy()，防止 in-place 修改原 counts
        adata.layers["log1p"] = np.log1p(adata.layers["counts"].copy())
    else:
        # 没有 counts，就直接对 X 做 log1p
        print("[INFO] No 'log1p' or 'counts' layer found. Creating 'log1p' from adata.X...")
        adata.layers["log1p"] = np.log1p(adata.X.copy())

# 统一：DE 全程使用 log1p layer，不用 raw
layer_for_de = "log1p"
use_raw_for_de = False

print(f"[INFO] DE will use layer={layer_for_de}, use_raw={use_raw_for_de}")

########################################
# 后面的逻辑基本不变
########################################

res_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

n_top_export = 20
all_res_summary_rows = []

def res_to_key(res: float) -> str:
    """0.6 -> 'leiden_scVI_0_6'；1.0 -> 'leiden_scVI_1'"""
    res_str = str(res).replace('.', '_')
    if float(res).is_integer():
        res_str = str(int(res))
    return f"leiden_scVI_{res_str}"

# 外层：按分辨率显示进度
for res in tqdm(res_list, desc="Scanpy DE by resolution", unit="res"):
    key = res_to_key(res)
    adata.obs["leiden"] = adata.obs[key].astype("category").cat.remove_unused_categories()
    cats = list(adata.obs["leiden"].cat.categories)

    de_key = f"rank_genes_{key}"
    # 这一步最花时间：对每个簇做 vs rest DE
    sc.tl.rank_genes_groups(
        adata,
        groupby="leiden",
        method="wilcoxon",
        use_raw=use_raw_for_de,
        layer=layer_for_de,  # <--- 用 log1p
        n_genes=adata.shape[1],
        key_added=de_key,
        pts=True
    )

    df_all = sc.get.rank_genes_groups_df(adata, key=de_key, group=None)

    if "pvals_adj" in df_all.columns:
        df_all = df_all[df_all["pvals_adj"] < 0.05]
    if "logfoldchanges" in df_all.columns:
        df_all = df_all[df_all["logfoldchanges"] > 0.5]
    if "pts" in df_all.columns:
        df_all = df_all[df_all["pts"] > 0.1]

    top_by_grp = (
        df_all.sort_values(["group", "scores"], ascending=[True, False])
             .groupby("group", as_index=False, sort=False)
             .head(n_top_export)
             .copy()
    )
    top_by_grp["resolution"] = res

    out_csv = os.path.join(tabledir, f"scanpy_DE_wilcoxon_{key}_top{n_top_export}.csv")
    top_by_grp.to_csv(out_csv, index=False)

    all_res_summary_rows.append(top_by_grp)

    # 内层：为 dotplot 准备每个簇的 Top10 基因，显示进度
    top10_dict = {}
    grp_list = list(top_by_grp.groupby("group"))
    for g, grp in tqdm(
        grp_list,
        total=len(grp_list),
        desc=f"Build Top10 for clusters @res={res}",
        unit="cluster",
        leave=True
    ):
        top10_dict[g] = grp.sort_values("scores", ascending=False)["names"].head(10).tolist()

    if len(top10_dict) > 0:
        sc.pl.dotplot(
            adata,
            var_names=top10_dict,
            groupby="leiden",
            use_raw=use_raw_for_de,
            layer=layer_for_de,  # <--- dotplot 也用 log1p
            show=False,
            save=f"_scanpy_DE_dotplot_{key}.pdf"
        )

if all_res_summary_rows:
    summary_df = pd.concat(all_res_summary_rows, ignore_index=True)
    summary_df.to_csv(os.path.join(tabledir, "scanpy_DE_summary_all_resolutions.csv"), index=False)
