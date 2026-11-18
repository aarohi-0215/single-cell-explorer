# 🧬 Single-Cell RNA-seq Explorer

An interactive dashboard for **quality control, dimensionality reduction, clustering, and
marker-gene exploration** of UMI-based (droplet) single-cell RNA-seq data. Upload your own
`.h5ad` count matrix or use the built-in 3k PBMC demo. Built with **Scanpy** and **Streamlit**.

**Live app:** https://single-cell-explorer-pdmkf4omg2bjyixcytoqwm.streamlit.app/

## Features
- **Upload or demo:** bring your own AnnData `.h5ad`, or explore the 10x 3k PBMC demo.
- **Quality Control:** per-cell distributions (genes/cell, total counts, % mitochondrial, species-agnostic) with illustrative QC thresholds.
- **UMAP Exploration:** UMAP colored by cluster or by any gene's expression.
- **Marker Genes:** searchable top marker genes per cluster (Wilcoxon).
- **Exposed parameters:** highly variable genes, number of PCs, and clustering resolution.

## Design & scope (by design, not limitation)
There is no universal single-cell pipeline — normalization is data-type dependent
(Luecken & Theis, *Current best practices in single-cell RNA-seq analysis*, Mol Syst Biol 2019).
So this tool is **scoped to UMI-based droplet scRNA-seq count data** and uses the standard
workflow: `normalize_total` + `log1p` → highly variable genes → PCA → neighbors → UMAP →
Leiden clustering → `rank_genes_groups`.
- Input must be an `.h5ad` with a **raw count** matrix (the app also looks in `layers`/`raw`).
- Full-length (Smart-seq2) data needs TPM normalization and is **not** supported.
- Upload **clean (QC'd)** data; the app does not remove doublets or outliers.
- Clustering falls back Leiden → Louvain → KMeans so it never hard-fails.

## Scope & limitations (dataset size)
The **hosted (free Streamlit Community Cloud) version** runs with ~1 GB of memory, so it is
scoped to datasets up to **~20,000 cells**. Larger datasets are **randomly subsampled to
20,000 cells** for responsiveness, with a visible banner in the app so results are never
silently degraded. This is a hosting limit, not a code limit.

**Have a large cohort (hundreds of thousands of cells)?** Run the app **locally** — there is
**no size cap** other than your own machine's memory. This is the recommended path for
full-scale datasets.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
The 3k PBMC demo downloads automatically on first run via Scanpy. Running locally removes the
hosted memory ceiling, so you can work with your full dataset.

## Test datasets
Run `python make_test_data.py` to generate two ready `.h5ad` files to upload
(mouse myeloid vs. human PBMC). For more variety (e.g., ovarian cancer, Drosophila), download
`.h5ad` files from CELLxGENE Discover (cellxgene.cziscience.com) or the Fly Cell Atlas
(flycellatlas.org); choose datasets that include raw counts. Large files are best run locally.

## Stack
Python · Scanpy · Streamlit · pandas · numpy · matplotlib · scikit-learn

Built by Aarohi Deshpande.
