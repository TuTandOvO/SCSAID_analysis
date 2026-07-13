# SCSAID_analysis

This repository stores analysis scripts used for the SCSAID project, a human and mouse skin single-cell RNA-seq database resource. The code covers preprocessing, quality control, doublet assessment, integration, clustering, cell-type annotation, downstream analyses, and selected scripts used to prepare database- and figure-related results.

The repository is intended to support reproducibility of the SCSAID analyses. Large raw data files, processed single-cell objects, intermediate results, and local audit notes are not included in this repository.

## Repository Contents

- `preprocess/`: human and mouse single-cell preprocessing, QC, integration, clustering, and annotation scripts.
- `enrichment/`, `DEG/`, `Augur/`, `CellChat/`, `SCORPION/`: downstream analysis scripts.
- `Geneformer_LINCS/`: scripts for the Geneformer and LINCS/L1000-related perturbation analyses.
- `model/` and `model_cross_species/`: classifier training and cross-species model analysis scripts.
- `manuscript.tex` and `reference.bib`: manuscript source and bibliography files used during manuscript preparation.

## Notes

The code was organized to document the analysis workflow used for SCSAID. Paths may need to be adjusted for local computing environments, and large input/output files should be downloaded or generated separately according to the data availability statement in the manuscript.
