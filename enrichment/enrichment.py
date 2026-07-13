"""
ORA Enrichment Analysis: Shared vs Species-Specific DEGs
=========================================================
Species-specific definition (refined):
  - shared:    DEG whose homolog is ALSO a DEG in the other species
  - specific:  DEG that HAS a homolog in the other species, but homolog is NOT a DEG
  - no_homolog: DEG with no homolog found — EXCLUDED from all analysis
"""

import os
import pandas as pd
import gseapy as gp
import requests
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MOUSE_DEG = "/gpfsdata/home/renyixiang/SkinDB/10X/mouse/data4viz/DEG_pseudobulk/DEG_Immune__IMQ-induced_psoriasis_vs_Healthy.csv"
HUMAN_DEG = "/gpfsdata/home/renyixiang/SkinDB/10X/human/data4viz/DEG_pseudobulk/DEG_Immune__psoriasis_vs_Healthy.csv"

MOUSE_GMT_DIR = "/gpfsdata/home/renyixiang/SkinDB/10X/mouse/data4viz/enrichment/gmt"
HUMAN_GMT_DIR = "/gpfsdata/home/renyixiang/SkinDB/10X/human/data4viz/enrichment/gmt"

OUTDIR = "/gpfsdata/home/renyixiang/SkinDB/final_entichment_immune"

PADJ_THRESH   = 0.05
LOG2FC_THRESH = 0.5


# ─────────────────────────────────────────────
# 1. LOAD DEGs
# ─────────────────────────────────────────────
def load_deg(path: str, species: str) -> list:
    df = pd.read_csv(path)
    log.info(f"[{species}] Raw DEG shape: {df.shape} | Columns: {df.columns.tolist()}")

    gene_col = next(
        (c for c in ["gene", "Gene", "gene_name", "symbol", "Gene_Symbol", "SYMBOL"]
         if c in df.columns), df.columns[0])
    padj_col = next(
        (c for c in ["padj", "p_adj", "pvals_adj", "FDR", "adj.P.Val", "p.adjust", "BH"]
         if c in df.columns), None)
    lfc_col  = next(
        (c for c in ["log2FoldChange", "logfoldchanges", "logFC", "log2FC", "avg_log2FC", "LFC"]
         if c in df.columns), None)

    log.info(f"[{species}] gene={gene_col} | padj={padj_col} | lfc={lfc_col}")

    mask = pd.Series([True] * len(df), index=df.index)
    if padj_col:
        mask &= df[padj_col] < PADJ_THRESH
    if lfc_col and LOG2FC_THRESH > 0:
        mask &= df[lfc_col].abs() > LOG2FC_THRESH

    genes = df.loc[mask, gene_col].dropna().unique().tolist()
    log.info(f"[{species}] Significant DEGs: {len(genes)}")
    return genes


# ─────────────────────────────────────────────
# 2. QUERY BIOMART
# ─────────────────────────────────────────────
def query_homologs_biomart() -> pd.DataFrame:
    log.info("Querying BioMart for mouse<->human homologs...")
    xml_query = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="0" uniqueRows="1" count="" datasetConfigVersion="0.6">
  <Dataset name="mmusculus_gene_ensembl" interface="default">
    <Attribute name="external_gene_name" />
    <Attribute name="hsapiens_homolog_associated_gene_name" />
  </Dataset>
</Query>"""
    for url in ["https://mart.ensembl.org/biomart/martservice",
                "https://useast.ensembl.org/biomart/martservice"]:
        try:
            resp = requests.get(url, params={"query": xml_query}, timeout=300)
            resp.raise_for_status()
            lines = [l.strip().split("\t") for l in resp.text.strip().split("\n") if l.strip()]
            df = pd.DataFrame(lines, columns=["mouse_symbol", "human_symbol"])
            df = df[(df["mouse_symbol"] != "") & (df["human_symbol"] != "")].dropna()
            log.info(f"BioMart: {len(df)} homolog pairs from {url}")
            return df
        except Exception as e:
            log.warning(f"BioMart failed ({url}): {e}")
    raise RuntimeError("All BioMart mirrors failed.")


# ─────────────────────────────────────────────
# 3. SPLIT: SHARED / SPECIFIC / NO-HOMOLOG
# ─────────────────────────────────────────────
def split_shared_specific(mouse_genes: list, human_genes: list,
                           homologs: pd.DataFrame):
    """
    Three-way classification per species:
      shared:     has homolog AND homolog is a DEG in the other species
      specific:   has homolog BUT homolog is NOT a DEG in the other species
      no_homolog: no homolog found in BioMart → EXCLUDED from ORA

    Returns:
      mouse_shared, mouse_specific, human_shared, human_specific
      (no_homolog genes are logged but not returned)
    """
    mouse_set = set(mouse_genes)
    human_set = set(human_genes)

    # gene → set of its homologs in the other species (empty set = no homolog)
    m2h = homologs.groupby("mouse_symbol")["human_symbol"].apply(set).to_dict()
    h2m = homologs.groupby("human_symbol")["mouse_symbol"].apply(set).to_dict()

    mouse_shared, mouse_specific, mouse_no_homolog = [], [], []
    for g in mouse_genes:
        h_homologs = m2h.get(g, set())
        if not h_homologs:
            mouse_no_homolog.append(g)          # no homolog → exclude
        elif h_homologs & human_set:
            mouse_shared.append(g)              # homolog is a human DEG
        else:
            mouse_specific.append(g)            # has homolog, not a human DEG

    human_shared, human_specific, human_no_homolog = [], [], []
    for g in human_genes:
        m_homologs = h2m.get(g, set())
        if not m_homologs:
            human_no_homolog.append(g)
        elif m_homologs & mouse_set:
            human_shared.append(g)
        else:
            human_specific.append(g)

    log.info("\n=== Gene classification ===")
    log.info(f"Mouse  shared     (homolog also DEG in human):      {len(mouse_shared)}")
    log.info(f"Mouse  specific   (has homolog, NOT DEG in human):  {len(mouse_specific)}")
    log.info(f"Mouse  no_homolog (excluded):                        {len(mouse_no_homolog)}")
    log.info(f"Human  shared     (homolog also DEG in mouse):      {len(human_shared)}")
    log.info(f"Human  specific   (has homolog, NOT DEG in mouse):  {len(human_specific)}")
    log.info(f"Human  no_homolog (excluded):                        {len(human_no_homolog)}")

    return (mouse_shared, mouse_specific, mouse_no_homolog,
            human_shared, human_specific, human_no_homolog)


# ─────────────────────────────────────────────
# 4. ORA
# ─────────────────────────────────────────────
def run_ora(gene_list: list, gmt_dir: str, label: str) -> pd.DataFrame:
    gmt_files = list(Path(gmt_dir).glob("*.gmt"))
    if not gmt_files:
        log.error(f"No GMT files found in {gmt_dir}")
        return pd.DataFrame()
    if not gene_list:
        log.warning(f"[{label}] Gene list empty, skipping ORA.")
        return pd.DataFrame()

    log.info(f"[{label}] ORA: {len(gene_list)} genes x {len(gmt_files)} GMT files")
    all_results = []
    for gmt_path in gmt_files:
        gmt_name = gmt_path.stem
        log.info(f"  -> {gmt_name}")
        try:
            enr = gp.enrichr(gene_list=gene_list, gene_sets=str(gmt_path), outdir=None)
            if enr.results is not None and len(enr.results) > 0:
                res = enr.results.copy()
                res.insert(0, "GMT", gmt_name)
                all_results.append(res)
                log.info(f"     {len(res)} terms before filter")
        except Exception as e:
            log.warning(f"  ORA failed for {gmt_name}: {e}")

    if not all_results:
        log.warning(f"[{label}] No ORA results.")
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)
    combined.insert(0, "Group", label)
    combined.rename(columns={
        "P-value":          "Pvalue",
        "Adjusted P-value": "Padj",
        "Odds Ratio":       "OddsRatio",
        "Combined Score":   "CombinedScore",
    }, inplace=True)

    if "Padj" in combined.columns:
        combined = combined[combined["Padj"] < PADJ_THRESH]
    log.info(f"[{label}] {len(combined)} significant terms (Padj < {PADJ_THRESH})")
    return combined


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    Path(OUTDIR).mkdir(parents=True, exist_ok=True)

    # 1. Load DEGs
    mouse_genes = load_deg(MOUSE_DEG, "mouse")
    human_genes = load_deg(HUMAN_DEG, "human")

    # 2. BioMart homologs (cached)
    homolog_cache = os.path.join(OUTDIR, "biomart_mouse_human_homologs.csv")
    if os.path.exists(homolog_cache):
        log.info(f"Loading cached homolog table: {homolog_cache}")
        homologs = pd.read_csv(homolog_cache)
    else:
        homologs = query_homologs_biomart()
        homologs.to_csv(homolog_cache, index=False)
        log.info("Homolog table cached.")

    # 3. Classify genes (three-way)
    (mouse_shared, mouse_specific, mouse_no_homolog,
     human_shared, human_specific, human_no_homolog) = \
        split_shared_specific(mouse_genes, human_genes, homologs)

    # Save gene lists (only shared + specific; no_homolog saved separately for reference)
    for label, lst in [
        ("mouse_shared",      mouse_shared),
        ("mouse_specific",    mouse_specific),
        ("human_shared",      human_shared),
        ("human_specific",    human_specific),
        ("mouse_no_homolog",  mouse_no_homolog),   # reference only, not used in ORA
        ("human_no_homolog",  human_no_homolog),
    ]:
        pd.DataFrame({"gene": lst, "group": label}).to_csv(
            os.path.join(OUTDIR, f"genelist_{label}.csv"), index=False
        )

    # Summary — only shared + specific (no_homolog excluded)
    summary = pd.DataFrame([
        {"group": "mouse_shared",   "n_genes": len(mouse_shared),
         "note": "has homolog, homolog is human DEG"},
        {"group": "mouse_specific", "n_genes": len(mouse_specific),
         "note": "has homolog, homolog NOT human DEG"},
        {"group": "human_shared",   "n_genes": len(human_shared),
         "note": "has homolog, homolog is mouse DEG"},
        {"group": "human_specific", "n_genes": len(human_specific),
         "note": "has homolog, homolog NOT mouse DEG"},
    ])
    summary.to_csv(os.path.join(OUTDIR, "gene_classification_summary.csv"), index=False)
    log.info(f"\n{summary.to_string(index=False)}")
    log.info(f"\nExcluded (no homolog): mouse={len(mouse_no_homolog)}, human={len(human_no_homolog)}")

    # 4. ORA — 4 groups only (no_homolog excluded)
    tasks = [
        ("mouse_shared",   mouse_shared,   MOUSE_GMT_DIR),
        ("mouse_specific", mouse_specific, MOUSE_GMT_DIR),
        ("human_shared",   human_shared,   HUMAN_GMT_DIR),
        ("human_specific", human_specific, HUMAN_GMT_DIR),
    ]

    for label, genes, gmt_dir in tasks:
        df = run_ora(genes, gmt_dir, label)
        out_path = os.path.join(OUTDIR, f"ORA_{label}.csv")
        if len(df) > 0:
            df.to_csv(out_path, index=False)
            log.info(f"Saved: {out_path}  ({len(df)} rows)")
        else:
            log.warning(f"No significant results for {label}.")

    log.info("\n=== Done! ===")
    log.info(f"Output: {OUTDIR}")
    log.info("  ORA_mouse_shared.csv / ORA_mouse_specific.csv")
    log.info("  ORA_human_shared.csv / ORA_human_specific.csv")
    log.info("  gene_classification_summary.csv  (shared+specific only)")
    log.info("  genelist_*_no_homolog.csv         (excluded genes, reference)")


if __name__ == "__main__":
    main()
