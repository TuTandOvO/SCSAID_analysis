"""
Species-specific DEG cross-species LFC comparison
==================================================
For each species-specific gene:
  - Its own LFC (significant)
  - Its homolog's LFC in the other species (not significant / not passing threshold)
  - Delta = |own_LFC| - |homolog_LFC|  (or you could do raw difference)
Sorted by absolute delta descending.
"""

import os
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── CONFIG ──
MOUSE_DEG = "/gpfsdata/home/renyixiang/SkinDB/10X/mouse/data4viz/DEG_pseudobulk/DEG_Immune__IMQ-induced_psoriasis_vs_Healthy.csv"
HUMAN_DEG = "/gpfsdata/home/renyixiang/SkinDB/10X/human/data4viz/DEG_pseudobulk/DEG_Immune__psoriasis_vs_Healthy.csv"
HOMOLOG_CSV = "/gpfsdata/home/renyixiang/SkinDB/final_enrichment_immune/biomart_mouse_human_homologs.csv"
OUTDIR = "/gpfsdata/home/renyixiang/SkinDB/final_enrichment_immune"

PADJ_THRESH  = 0.05
LOG2FC_THRESH = 0.5

def load_full_deg(path, species):
    """Load full DEG table, return (sig_genes_set, full_lfc_dict, full_padj_dict)."""
    df = pd.read_csv(path)
    gene_col = "gene"
    lfc_col  = "logfoldchanges"
    padj_col = "pvals_adj"

    # full LFC lookup: gene -> LFC (keep all genes)
    full_lfc  = df.set_index(gene_col)[lfc_col].to_dict()
    full_padj = df.set_index(gene_col)[padj_col].to_dict()

    # significant set
    mask = (df[padj_col] < PADJ_THRESH) & (df[lfc_col].abs() > LOG2FC_THRESH)
    sig_genes = set(df.loc[mask, gene_col].dropna().unique())

    log.info(f"[{species}] Total genes: {len(full_lfc)}, Significant DEGs: {len(sig_genes)}")
    return sig_genes, full_lfc, full_padj


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # 1. Load DEG tables (full)
    mouse_sig, mouse_lfc, mouse_padj = load_full_deg(MOUSE_DEG, "mouse")
    human_sig, human_lfc, human_padj = load_full_deg(HUMAN_DEG, "human")

    # 2. Homolog mapping
    hom = pd.read_csv(HOMOLOG_CSV)
    m2h = hom.groupby("mouse_symbol")["human_symbol"].apply(set).to_dict()
    h2m = hom.groupby("human_symbol")["mouse_symbol"].apply(set).to_dict()

    # 3. Mouse-specific genes: mouse DEG, has human homolog, homolog NOT human DEG
    rows = []
    for g in mouse_sig:
        h_homologs = m2h.get(g, set())
        if not h_homologs:
            continue  # no homolog
        if h_homologs & human_sig:
            continue  # shared, not specific
        # mouse-specific: pick the best-matching human homolog (the one present in human DEG table)
        m_lfc = mouse_lfc.get(g)
        for hg in h_homologs:
            h_lfc = human_lfc.get(hg)  # might be None if gene not in human table at all
            rows.append({
                "species_of_origin": "mouse",
                "gene_mouse": g,
                "gene_human": hg,
                "LFC_mouse": round(m_lfc, 4) if m_lfc is not None else None,
                "LFC_human": round(h_lfc, 4) if h_lfc is not None else None,
                "padj_mouse": mouse_padj.get(g),
                "padj_human": human_padj.get(hg),
            })

    # 4. Human-specific genes: human DEG, has mouse homolog, homolog NOT mouse DEG
    for g in human_sig:
        m_homologs = h2m.get(g, set())
        if not m_homologs:
            continue
        if m_homologs & mouse_sig:
            continue
        h_lfc = human_lfc.get(g)
        for mg in m_homologs:
            m_lfc = mouse_lfc.get(mg)
            rows.append({
                "species_of_origin": "human",
                "gene_human": g,
                "gene_mouse": mg,
                "LFC_human": round(h_lfc, 4) if h_lfc is not None else None,
                "LFC_mouse": round(m_lfc, 4) if m_lfc is not None else None,
                "padj_human": human_padj.get(g),
                "padj_mouse": mouse_padj.get(mg),
            })

    df = pd.DataFrame(rows)

    # 5. Compute delta
    # delta_abs = |LFC_significant_species| - |LFC_other_species|
    # Also add a simple absolute difference
    df["abs_LFC_mouse"] = df["LFC_mouse"].abs()
    df["abs_LFC_human"] = df["LFC_human"].abs()
    df["delta_absLFC"]   = (df["abs_LFC_mouse"] - df["abs_LFC_human"]).abs()  # symmetric measure

    # Sort by delta descending
    df = df.sort_values("delta_absLFC", ascending=False).reset_index(drop=True)

    # Reorder columns
    df = df[["species_of_origin",
             "gene_mouse", "gene_human",
             "LFC_mouse", "LFC_human",
             "padj_mouse", "padj_human",
             "abs_LFC_mouse", "abs_LFC_human",
             "delta_absLFC"]]

    out_path = os.path.join(OUTDIR, "specific_genes_LFC_comparison.csv")
    df.to_csv(out_path, index=False)
    log.info(f"Saved {len(df)} rows to {out_path}")
    log.info(f"\nTop 20 by delta:\n{df.head(20).to_string(index=False)}")


if __name__ == "__main__":
    main()
