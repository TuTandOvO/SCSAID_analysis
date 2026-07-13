"""
Cross-species Keratinocyte log2FC comparison for key psoriasis DEGs.
All 10 genes are computed in BOTH Human and Mouse via BioMart ortholog mapping.

Run on HPC:
    python compute_immune_lfc.py
"""

import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import sparse

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "font.family": "Arial",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ══════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════
HUMAN_PATH = "/gpfsdata/home/renyixiang/SkinDB/10X/human/Annotation/Annotated/human_with_all_leiden_10threshold_annotated_updated.h5ad"
MOUSE_PATH = "/gpfsdata/home/renyixiang/SkinDB/10X/mouse/Annotation/Annotated/mouse_with_all_leiden_10threshold_annotated_updated.h5ad"

CELL_TYPE_COL = "Gross_Map"
CELL_TYPE_LABEL = "Immune"
CONDITION_COL = "condition"

HUMAN_PSO = "psoriasis"
HUMAN_HLT = "Healthy"
MOUSE_PSO = "IMQ-induced psoriasis"
MOUSE_HLT = "Healthy"

# Seed genes (as given by user, in original species naming)
HUMAN_SEED_GENES = ["TPSB2", "IGHA1", "IGLC2", "CPA3", "HDC"]
MOUSE_SEED_GENES = ["Cxcl14", "Ifi202b", "Il20ra"]

OUTDIR = "."


# ══════════════════════════════════════════════════════════════════════
#  STEP 1: BioMart Ortholog Mapping
# ══════════════════════════════════════════════════════════════════════
def query_biomart_orthologs(human_genes, mouse_genes):
    """
    Query Ensembl BioMart for human<->mouse ortholog gene names.
    Returns dict: { display_name: {"human": gene_name, "mouse": gene_name} }
    """
    from pybiomart import Dataset

    print("\n[BioMart] Querying human → mouse orthologs...")
    hs = Dataset(name="hsapiens_gene_ensembl", host="http://www.ensembl.org")
    h2m = hs.query(
        attributes=["external_gene_name", "mmusculus_homolog_associated_gene_name"],
        filters={"external_gene_name": human_genes},
    )
    h2m.columns = ["human_gene", "mouse_gene"]
    h2m = h2m.dropna().drop_duplicates()
    h2m = h2m[h2m["mouse_gene"] != ""]
    print(h2m.to_string(index=False))

    print("\n[BioMart] Querying mouse → human orthologs...")
    mm = Dataset(name="mmusculus_gene_ensembl", host="http://www.ensembl.org")
    m2h = mm.query(
        attributes=["external_gene_name", "hsapiens_homolog_associated_gene_name"],
        filters={"external_gene_name": mouse_genes},
    )
    m2h.columns = ["mouse_gene", "human_gene"]
    m2h = m2h.dropna().drop_duplicates()
    m2h = m2h[m2h["human_gene"] != ""]
    print(m2h.to_string(index=False))

    # Build unified mapping table
    mapping = {}

    # Human-seed genes → get mouse orthologs
    for _, row in h2m.iterrows():
        hg, mg = row["human_gene"], row["mouse_gene"]
        if hg in human_genes:
            mapping[hg] = {"human": hg, "mouse": mg, "origin": "human_specific"}

    # Mouse-seed genes → get human orthologs
    for _, row in m2h.iterrows():
        mg, hg = row["mouse_gene"], row["human_gene"]
        if mg in mouse_genes:
            mapping[mg] = {"human": hg, "mouse": mg, "origin": "mouse_specific"}

    return mapping


def get_hardcoded_mapping():
    """Fallback: well-known ortholog pairs for the 8 selected genes."""
    mapping = {
        # Human-specific (5)
        "TPSB2":   {"human": "TPSB2",   "mouse": "Tpsb2",   "origin": "human_specific"},
        "CPA3":    {"human": "CPA3",    "mouse": "Cpa3",    "origin": "human_specific"},
        "IL1RL1":  {"human": "IL1RL1",  "mouse": "Il1rl1",  "origin": "human_specific"},
        "IGHA1":   {"human": "IGHA1",   "mouse": "Igha",    "origin": "human_specific"},
        "CTSG":    {"human": "CTSG",    "mouse": "Ctsg",    "origin": "human_specific"},
        # Mouse-specific (3)
        "Cxcl14":  {"human": "CXCL14",  "mouse": "Cxcl14",  "origin": "mouse_specific"},
        "Ifi202b": {"human": "MNDA",    "mouse": "Ifi202b", "origin": "mouse_specific"},
        "Il20ra":  {"human": "IL20RA",  "mouse": "Il20ra",  "origin": "mouse_specific"},
    }
    return mapping


def get_ortholog_mapping(human_genes, mouse_genes):
    """Try BioMart first, fall back to hardcoded if unavailable."""
    try:
        mapping = query_biomart_orthologs(human_genes, mouse_genes)
        if len(mapping) >= 6:  # sanity check: expect ~10 genes
            print(f"\n✓ BioMart mapping obtained: {len(mapping)} gene pairs")
            return mapping
        else:
            print(f"\n⚠ BioMart returned only {len(mapping)} pairs, using hardcoded fallback")
    except Exception as e:
        print(f"\n⚠ BioMart query failed: {e}")
        print("  Using hardcoded ortholog mapping as fallback")

    mapping = get_hardcoded_mapping()
    print(f"✓ Hardcoded mapping loaded: {len(mapping)} gene pairs")
    return mapping


# ══════════════════════════════════════════════════════════════════════
#  STEP 2: Compute LFC
# ══════════════════════════════════════════════════════════════════════
def compute_lfc_for_genes(adata, gene_list, pso_label, healthy_label, species_name):
    """
    Compute pseudobulk log2FC for a list of genes in Keratinocytes.
    Returns DataFrame with one row per gene.
    """
    print(f"\n  [{species_name}] Full: {adata.shape[0]} cells x {adata.shape[1]} genes")

    # Subset to Keratinocyte
    mask_kc = adata.obs[CELL_TYPE_COL] == CELL_TYPE_LABEL
    adata_kc = adata[mask_kc]
    print(f"  [{species_name}] Keratinocytes: {adata_kc.n_obs}")

    # Split conditions
    mask_pso = adata_kc.obs[CONDITION_COL] == pso_label
    mask_hlt = adata_kc.obs[CONDITION_COL] == healthy_label
    n_pso, n_hlt = int(mask_pso.sum()), int(mask_hlt.sum())
    print(f"  [{species_name}] Pso: {n_pso} | Healthy: {n_hlt}")

    # Check which genes exist
    available = [g for g in gene_list if g in adata_kc.var_names]
    missing = [g for g in gene_list if g not in adata_kc.var_names]
    if missing:
        print(f"  [{species_name}] ⚠ Not found in var_names: {missing}")

    if not available:
        # Return empty df with NaN
        return pd.DataFrame({
            "gene_in_species": gene_list,
            "mean_pso": np.nan, "mean_hlt": np.nan,
            "log2FC": np.nan, "frac_pso": np.nan, "frac_hlt": np.nan,
            "n_pso": n_pso, "n_hlt": n_hlt,
        })

    # Extract from adata.X (normalized counts)
    X_pso = adata_kc[mask_pso, available].X
    X_hlt = adata_kc[mask_hlt, available].X
    if sparse.issparse(X_pso):
        X_pso = X_pso.toarray()
    if sparse.issparse(X_hlt):
        X_hlt = X_hlt.toarray()

    mean_pso = np.mean(X_pso, axis=0)
    mean_hlt = np.mean(X_hlt, axis=0)
    frac_pso = np.mean(X_pso > 0, axis=0)
    frac_hlt = np.mean(X_hlt > 0, axis=0)

    pseudocount = 1e-4
    log2fc = np.log2(mean_pso + pseudocount) - np.log2(mean_hlt + pseudocount)

    df = pd.DataFrame({
        "gene_in_species": available,
        "mean_pso": np.round(mean_pso, 5),
        "mean_hlt": np.round(mean_hlt, 5),
        "log2FC": np.round(log2fc, 4),
        "frac_pso": np.round(frac_pso, 4),
        "frac_hlt": np.round(frac_hlt, 4),
        "n_pso": n_pso,
        "n_hlt": n_hlt,
    })

    # Add rows for missing genes (as NaN)
    if missing:
        df_miss = pd.DataFrame({
            "gene_in_species": missing,
            "mean_pso": np.nan, "mean_hlt": np.nan,
            "log2FC": np.nan, "frac_pso": np.nan, "frac_hlt": np.nan,
            "n_pso": n_pso, "n_hlt": n_hlt,
        })
        df = pd.concat([df, df_miss], ignore_index=True)

    return df


# ══════════════════════════════════════════════════════════════════════
#  STEP 3: Visualization
# ══════════════════════════════════════════════════════════════════════
def plot_cross_species_lfc(df_plot, outpath="Immune_LFC_cross_species.pdf"):
    """
    Grouped bar chart: for each gene, Human LFC vs Mouse LFC side by side.
    Split into two panels: human-specific origin genes, mouse-specific origin genes.
    """
    origins = ["human_specific", "mouse_specific"]
    panel_titles = [
        "Human-specific DEGs",
        "Mouse-specific DEGs",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"wspace": 0.4})

    bar_h = 0.32  # bar half-height
    colors = {"Human": "#C0392B", "Mouse": "#2E86C1"}

    for ax, origin, ptitle in zip(axes, origins, panel_titles):
        sub = df_plot[df_plot["origin"] == origin].copy()
        genes = sub["display_name"].unique()
        n = len(genes)

        for i, gene in enumerate(genes):
            row = sub[sub["display_name"] == gene].iloc[0]
            lfc_h = row["log2FC_human"]
            lfc_m = row["log2FC_mouse"]

            # Human bar (top)
            if not np.isnan(lfc_h):
                ax.barh(i + bar_h/2 + 0.02, lfc_h, height=bar_h,
                        color=colors["Human"], edgecolor="white", linewidth=0.5,
                        zorder=3, label="Human" if i == 0 else "")
                offset = 0.06 * (1 if lfc_h >= 0 else -1)
                ax.text(lfc_h + offset, i + bar_h/2 + 0.02, f"{lfc_h:.2f}",
                        va="center", ha="left" if lfc_h >= 0 else "right",
                        fontsize=8, color=colors["Human"], fontweight="600")
            else:
                ax.text(0.05, i + bar_h/2 + 0.02, "N/A", va="center", ha="left",
                        fontsize=8, color="#999", fontstyle="italic")

            # Mouse bar (bottom)
            if not np.isnan(lfc_m):
                ax.barh(i - bar_h/2 - 0.02, lfc_m, height=bar_h,
                        color=colors["Mouse"], edgecolor="white", linewidth=0.5,
                        zorder=3, label="Mouse" if i == 0 else "")
                offset = 0.06 * (1 if lfc_m >= 0 else -1)
                ax.text(lfc_m + offset, i - bar_h/2 - 0.02, f"{lfc_m:.2f}",
                        va="center", ha="left" if lfc_m >= 0 else "right",
                        fontsize=8, color=colors["Mouse"], fontweight="600")
            else:
                ax.text(0.05, i - bar_h/2 - 0.02, "N/A", va="center", ha="left",
                        fontsize=8, color="#999", fontstyle="italic")

        ax.set_yticks(range(n))
        ax.set_yticklabels(genes, fontsize=11, fontstyle="italic", fontweight="600")
        ax.set_xlabel("log₂FC (Psoriasis / Healthy)", fontsize=10)
        ax.set_title(ptitle, fontsize=12, fontweight="bold", pad=10)
        ax.axvline(0, color="#333", linewidth=0.8, zorder=2)
        ax.grid(axis="x", alpha=0.15, linewidth=0.5, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.invert_yaxis()

        if origin == origins[0]:
            ax.legend(loc="lower right", fontsize=9, framealpha=0.9,
                      edgecolor="#ddd", fancybox=True)

    fig.suptitle(
        "Cross-species Immune log₂FC Comparison\n(Psoriasis vs Healthy, BioMart orthologs)",
        fontsize=14, fontweight="bold", y=1.03,
    )

    plt.savefig(outpath, bbox_inches="tight", dpi=200)
    plt.savefig(outpath.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    print(f"\n✓ Figure saved: {outpath}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── 1. Ortholog mapping ──
    mapping = get_ortholog_mapping(HUMAN_SEED_GENES, MOUSE_SEED_GENES)

    print("\n" + "="*60)
    print("  Ortholog Mapping")
    print("="*60)
    print(f"  {'Display':<12} {'Human':<12} {'Mouse':<12} {'Origin'}")
    print(f"  {'-'*48}")
    for name, info in mapping.items():
        print(f"  {name:<12} {info['human']:<12} {info['mouse']:<12} {info['origin']}")

    # Collect all genes needed per species
    human_genes_needed = list(set(info["human"] for info in mapping.values()))
    mouse_genes_needed = list(set(info["mouse"] for info in mapping.values()))
    print(f"\n  Human genes to query: {human_genes_needed}")
    print(f"  Mouse genes to query: {mouse_genes_needed}")

    # ── 2. Load data & compute LFC ──
    print(f"\nLoading Human: {HUMAN_PATH}")
    adata_h = sc.read_h5ad(HUMAN_PATH)
    df_human = compute_lfc_for_genes(adata_h, human_genes_needed, HUMAN_PSO, HUMAN_HLT, "Human")
    del adata_h

    print(f"\nLoading Mouse: {MOUSE_PATH}")
    adata_m = sc.read_h5ad(MOUSE_PATH)
    df_mouse = compute_lfc_for_genes(adata_m, mouse_genes_needed, MOUSE_PSO, MOUSE_HLT, "Mouse")
    del adata_m

    # ── 3. Merge into unified table ──
    rows = []
    for display_name, info in mapping.items():
        hg, mg = info["human"], info["mouse"]

        # Human LFC
        h_row = df_human[df_human["gene_in_species"] == hg]
        lfc_h = h_row["log2FC"].values[0] if len(h_row) else np.nan
        mp_h  = h_row["mean_pso"].values[0] if len(h_row) else np.nan
        mh_h  = h_row["mean_hlt"].values[0] if len(h_row) else np.nan
        fp_h  = h_row["frac_pso"].values[0] if len(h_row) else np.nan
        fh_h  = h_row["frac_hlt"].values[0] if len(h_row) else np.nan
        np_h  = h_row["n_pso"].values[0] if len(h_row) else 0
        nh_h  = h_row["n_hlt"].values[0] if len(h_row) else 0

        # Mouse LFC
        m_row = df_mouse[df_mouse["gene_in_species"] == mg]
        lfc_m = m_row["log2FC"].values[0] if len(m_row) else np.nan
        mp_m  = m_row["mean_pso"].values[0] if len(m_row) else np.nan
        mh_m  = m_row["mean_hlt"].values[0] if len(m_row) else np.nan
        fp_m  = m_row["frac_pso"].values[0] if len(m_row) else np.nan
        fh_m  = m_row["frac_hlt"].values[0] if len(m_row) else np.nan
        np_m  = m_row["n_pso"].values[0] if len(m_row) else 0
        nh_m  = m_row["n_hlt"].values[0] if len(m_row) else 0

        rows.append({
            "display_name": display_name,
            "origin": info["origin"],
            "human_gene": hg,
            "mouse_gene": mg,
            "log2FC_human": lfc_h,
            "log2FC_mouse": lfc_m,
            "mean_pso_human": mp_h, "mean_hlt_human": mh_h,
            "mean_pso_mouse": mp_m, "mean_hlt_mouse": mh_m,
            "frac_pso_human": fp_h, "frac_hlt_human": fh_h,
            "frac_pso_mouse": fp_m, "frac_hlt_mouse": fh_m,
            "n_pso_human": int(np_h), "n_hlt_human": int(nh_h),
            "n_pso_mouse": int(np_m), "n_hlt_mouse": int(nh_m),
        })

    df_all = pd.DataFrame(rows)

    # Save CSV
    csv_path = f"{OUTDIR}/Immune_LFC_cross_species.csv"
    df_all.to_csv(csv_path, index=False)
    print(f"\n✓ Table saved: {csv_path}")

    # ── 4. Plot ──
    plot_cross_species_lfc(df_all, f"{OUTDIR}/Immune_LFC_cross_species.pdf")

    # ── 5. Summary ──
    print("\n" + "="*70)
    print("  CROSS-SPECIES LFC SUMMARY  (Immune, Psoriasis vs Healthy)")
    print("="*70)
    summary_cols = ["display_name", "origin", "human_gene", "mouse_gene",
                    "log2FC_human", "log2FC_mouse", "frac_pso_human", "frac_pso_mouse"]
    print(df_all[summary_cols].to_string(index=False))
    print()

    # Highlight concordance
    print("  Concordance check (same direction of change):")
    for _, r in df_all.iterrows():
        lh, lm = r["log2FC_human"], r["log2FC_mouse"]
        if np.isnan(lh) or np.isnan(lm):
            status = "⚠ missing in one species"
        elif (lh > 0 and lm > 0) or (lh < 0 and lm < 0):
            status = "✓ concordant"
        else:
            status = "✗ discordant"
        print(f"    {r['display_name']:<12}  H={lh:>7.3f}  M={lm:>7.3f}  → {status}")
    print()
