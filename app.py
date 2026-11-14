"""
Single-Cell RNA-seq Explorer
Interactive QC, dimensionality reduction, clustering, and marker-gene exploration
for UMI-based (droplet) single-cell RNA-seq data.

Upload your own .h5ad (AnnData) count matrix, or use the built-in 3k PBMC demo.
Scope note: this tool assumes UMI-based droplet scRNA-seq COUNT data. Full-length
(e.g., Smart-seq2) data needs gene-length normalization (TPM) and is not supported.

Author: Aarohi Deshpande
"""

import hashlib
import os
import tempfile

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import scanpy as sc
import streamlit as st

sc.settings.verbosity = 0
st.set_page_config(page_title="Single-Cell RNA-seq Explorer", page_icon="🧬", layout="wide")

MAX_CELLS = 20000  # subsample cap to stay within free-tier memory/time


# ----------------------------- Helpers -----------------------------
def _looks_like_counts(X) -> bool:
    """Heuristic: non-negative, integer-valued -> raw counts."""
    if sp.issparse(X):
        d = X.data[:2000] if X.data.size else np.array([0.0])
    else:
        d = np.asarray(X).ravel()[:2000]
    if d.size == 0:
        return True
    if np.nanmin(d) < 0:
        return False
    return np.allclose(d, np.round(d))


def find_counts(adata):
    """Return an AnnData whose .X is raw counts, searching X, layers, and raw.
    Returns (adata_counts, source_str) or (None, None)."""
    if _looks_like_counts(adata.X):
        return adata, "X"
    for key in ("counts", "raw_counts", "umi_counts", "spliced"):
        if key in adata.layers and _looks_like_counts(adata.layers[key]):
            ad = adata.copy()
            ad.X = adata.layers[key].copy()
            return ad, f"layers['{key}']"
    if adata.raw is not None:
        try:
            if _looks_like_counts(adata.raw.X):
                return adata.raw.to_adata(), "raw"
        except Exception:
            pass
    return None, None


def mito_mask(var_names):
    """Species-agnostic mitochondrial gene detection. Returns a plain numpy bool array."""
    up = pd.Index(var_names).str.upper()
    mask = up.str.startswith("MT-") | up.str.startswith("MT.") | up.str.startswith("MT:")
    return np.asarray(mask, dtype=bool)


@st.cache_resource(show_spinner="Loading demo (3k PBMC) data...")
def load_demo():
    ad = sc.datasets.pbmc3k()
    ad.var_names_make_unique()
    return ad


def load_uploaded(uploaded_file):
    """Load an uploaded .h5ad into AnnData via a temp file (robust)."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5ad") as tmp:
        tmp.write(uploaded_file.getbuffer())
        path = tmp.name
    try:
        ad = sc.read_h5ad(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    return ad


@st.cache_data(show_spinner="Computing QC metrics...")
def qc_frame(cache_key, _adata):
    ad = _adata
    mt = mito_mask(ad.var_names)
    X = ad.X
    counts = np.asarray(X.sum(axis=1)).ravel() if sp.issparse(X) else np.asarray(X).sum(axis=1)
    genes = np.asarray((X > 0).sum(axis=1)).ravel() if sp.issparse(X) else (np.asarray(X) > 0).sum(axis=1)
    if mt.sum() > 0:
        idx = np.where(mt)[0]
        Xmt = X[:, idx]
        mt_counts = np.asarray(Xmt.sum(axis=1)).ravel() if sp.issparse(Xmt) else np.asarray(Xmt).sum(axis=1)
        pct_mt = 100.0 * mt_counts / np.maximum(counts, 1)
    else:
        pct_mt = np.zeros(ad.n_obs)
    return pd.DataFrame(
        {"genes_per_cell": genes, "total_counts": counts, "pct_mito": pct_mt}
    ), int(mt.sum())


def do_clustering(ad, resolution):
    """Leiden -> Louvain -> KMeans fallback. Always returns an obs key."""
    try:
        sc.tl.leiden(ad, resolution=resolution, flavor="igraph", n_iterations=2, directed=False)
        return "leiden"
    except Exception:
        pass
    try:
        sc.tl.leiden(ad, resolution=resolution)
        return "leiden"
    except Exception:
        pass
    try:
        sc.tl.louvain(ad, resolution=resolution)
        return "louvain"
    except Exception:
        pass
    from sklearn.cluster import KMeans
    k = int(max(2, min(12, round(resolution * 8))))
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(ad.obsm["X_pca"])
    ad.obs["kmeans"] = pd.Categorical([str(x) for x in labels])
    return "kmeans"


@st.cache_resource(show_spinner="Running single-cell pipeline...")
def run_pipeline(cache_key, _adata, n_hvg, n_pcs, resolution):
    ad = _adata.copy()
    ad.var_names_make_unique()
    ad.obs_names_make_unique()
    sc.pp.filter_genes(ad, min_cells=3)   # drop never-detected genes (hygiene, not QC)
    sc.pp.filter_cells(ad, min_genes=1)
    ad.layers["counts"] = ad.X.copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    ad.raw = ad                            # log-norm, all genes -> for markers & gene plots
    n_hvg = int(min(n_hvg, ad.n_vars - 1))
    sc.pp.highly_variable_genes(ad, n_top_genes=n_hvg, flavor="seurat")
    ad = ad[:, ad.var.highly_variable].copy()
    sc.pp.scale(ad, max_value=10)
    n_pcs = int(min(n_pcs, ad.n_obs - 1, ad.n_vars - 1))
    sc.tl.pca(ad, n_comps=n_pcs)
    sc.pp.neighbors(ad, n_neighbors=15, n_pcs=n_pcs)
    sc.tl.umap(ad)
    cluster_key = do_clustering(ad, resolution)
    sc.tl.rank_genes_groups(ad, cluster_key, method="wilcoxon", use_raw=True)
    res = ad.uns["rank_genes_groups"]
    groups = res["names"].dtype.names
    rows = []
    for g in groups:
        for i in range(min(25, len(res["names"][g]))):
            rows.append({
                "cluster": g, "rank": i + 1, "gene": res["names"][g][i],
                "score": round(float(res["scores"][g][i]), 2),
                "log2FC": round(float(res["logfoldchanges"][g][i]), 2),
                "pval_adj": float(res["pvals_adj"][g][i]),
            })
    markers = pd.DataFrame(rows)
    return ad, cluster_key, markers


# ----------------------------- Sidebar -----------------------------
st.sidebar.header("Data")
source = st.sidebar.radio("Choose data source:", ["Demo: 3k PBMC", "Upload my own (.h5ad)"])
uploaded = None
if source == "Upload my own (.h5ad)":
    uploaded = st.sidebar.file_uploader("Upload an AnnData .h5ad file", type=["h5ad"])

st.sidebar.header("Pipeline parameters")
with st.sidebar.expander("Adjust parameters", expanded=True):
    n_hvg = st.slider("Highly variable genes", 500, 5000, 2000, 250,
                      help="How many top variable genes to keep for dimensionality reduction.")
    n_pcs = st.slider("Principal components", 10, 50, 30, 5,
                      help="Number of PCs used for the neighbor graph and UMAP.")
    resolution = st.slider("Clustering resolution", 0.1, 2.0, 1.0, 0.1,
                           help="Higher = more, smaller clusters (Leiden).")

st.sidebar.caption(
    "Assumes UMI-based droplet scRNA-seq **count** data. Full-length (Smart-seq2) "
    "data needs TPM normalization and is not supported. Upload clean (QC'd) data; "
    "this tool does not remove doublets or outliers."
)
st.sidebar.caption("Built by Aarohi Deshpande · Scanpy + Streamlit")
with st.sidebar.expander("Scope & limitations"):
    st.markdown(
        f"- **Hosted (free tier):** handles up to ~{MAX_CELLS:,} cells; larger datasets are "
        f"randomly subsampled to stay within memory.\n"
        f"- **Large cohorts:** run the app locally (no size cap except your own machine) — "
        f"see the README.\n"
        f"- **Data type:** UMI-based droplet scRNA-seq counts (.h5ad); not full-length or spatial."
    )


# ----------------------------- Resolve data -----------------------------
st.title("🧬 Single-Cell RNA-seq Explorer")
st.caption(
    "Upload a UMI-based scRNA-seq count matrix (.h5ad) or use the 3k PBMC demo, then "
    "explore QC, UMAP, and marker genes. Adjust the pipeline parameters in the sidebar."
)

adata_raw = None
data_id = None

if source == "Upload my own (.h5ad)":
    if uploaded is None:
        st.info("⬅️ Upload an `.h5ad` file in the sidebar, or switch to the demo dataset.")
        st.stop()
    try:
        raw = load_uploaded(uploaded)
    except Exception as e:
        st.error(f"Could not read that file as an AnnData `.h5ad`: {e}")
        st.stop()
    counts_ad, src = find_counts(raw)
    if counts_ad is None:
        st.error(
            "Couldn't find a raw **count** matrix in this file (checked `.X`, common "
            "`layers`, and `.raw`). This tool needs counts, not normalized data. "
            "Please upload a count matrix."
        )
        st.stop()
    if counts_ad.n_obs < 50 or counts_ad.n_vars < 100:
        st.error(f"Dataset looks too small ({counts_ad.n_obs} cells x {counts_ad.n_vars} genes).")
        st.stop()
    orig_cells = counts_ad.n_obs
    if counts_ad.n_obs > MAX_CELLS:
        sc.pp.subsample(counts_ad, n_obs=MAX_CELLS, random_state=0)
        st.session_state["subsampled"] = (orig_cells, MAX_CELLS)
    else:
        st.session_state["subsampled"] = None
    adata_raw = counts_ad
    data_id = hashlib.md5(uploaded.getvalue()).hexdigest()[:10]
    st.success(f"Loaded **{uploaded.name}** — {adata_raw.n_obs:,} cells x {adata_raw.n_vars:,} genes (counts from `{src}`).")
else:
    adata_raw = load_demo()
    data_id = "pbmc3k_demo"
    st.session_state["subsampled"] = None

cache_key = f"{data_id}-{n_hvg}-{n_pcs}-{resolution}"

_sub = st.session_state.get("subsampled")
if _sub:
    st.warning(
        f"**Large dataset ({_sub[0]:,} cells).** Showing a representative random sample of "
        f"{_sub[1]:,} cells so this free-hosted app stays responsive. "
        f"For the full dataset with no size cap, run the app locally (see the README)."
    )

# Run pipeline (guarded)
try:
    pdata, cluster_key, markers = run_pipeline(cache_key, adata_raw, n_hvg, n_pcs, resolution)
except Exception as e:
    st.error(
        "The pipeline hit an error on this dataset. This usually means the data isn't "
        f"UMI count data or has an unusual structure.\n\nDetails: {e}"
    )
    st.stop()

qc_key = f"{data_id}-qc"
qc_df, n_mito = qc_frame(qc_key, adata_raw)


# ----------------------------- Tabs -----------------------------
tab_qc, tab_umap, tab_markers = st.tabs(["① Quality Control", "② UMAP Exploration", "③ Marker Genes"])

with tab_qc:
    st.subheader("Per-cell quality metrics (raw counts)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cells", f"{adata_raw.n_obs:,}")
    c2.metric("Genes", f"{adata_raw.n_vars:,}")
    c3.metric("Median genes / cell", f"{int(np.median(qc_df['genes_per_cell'])):,}")
    c4.metric("Mito genes found", f"{n_mito}")
    if n_mito == 0:
        st.caption("No mitochondrial genes detected (gene names may not use an MT- prefix for this species).")

    f1, f2, f3 = st.columns(3)
    for col, metric, label in [
        (f1, "genes_per_cell", "Genes per cell"),
        (f2, "total_counts", "Total counts per cell"),
        (f3, "pct_mito", "% mitochondrial"),
    ]:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.hist(qc_df[metric], bins=60, color="#1A3E5C")
        ax.set_title(label, fontsize=10); ax.set_xlabel(label, fontsize=8); ax.set_ylabel("Cells", fontsize=8)
        col.pyplot(fig); plt.close(fig)

    st.markdown("#### Illustrative QC thresholds")
    s1, s2, s3 = st.columns(3)
    g_max = int(qc_df["genes_per_cell"].max())
    min_genes = s1.slider("Min genes per cell", 0, g_max, min(200, g_max), 50)
    max_mt = s2.slider("Max % mitochondrial", 0.0, 100.0, 5.0, 0.5)
    min_counts = s3.slider("Min total counts", 0, int(qc_df["total_counts"].max()), 0, 100)
    keep = ((qc_df["genes_per_cell"] >= min_genes) & (qc_df["pct_mito"] <= max_mt) & (qc_df["total_counts"] >= min_counts))
    nk, nt = int(keep.sum()), len(qc_df)
    st.info(f"**{nk:,} of {nt:,} cells** would pass these thresholds ({100*nk/nt:.1f}%).")
    st.caption("Illustrative only — the pipeline uses your data as uploaded (assumed pre-QC'd).")

with tab_umap:
    st.subheader(f"UMAP — clustered with {cluster_key}")
    left, right = st.columns(2)
    with left:
        st.markdown(f"**Colored by cluster ({cluster_key})**")
        fig = sc.pl.umap(pdata, color=cluster_key, return_fig=True, show=False)
        st.pyplot(fig); plt.close(fig)
    with right:
        genes = sorted(pdata.raw.var_names.tolist()) if pdata.raw is not None else sorted(pdata.var_names.tolist())
        default = "CD3D" if "CD3D" in genes else genes[0]
        gene = st.selectbox("Color UMAP by a gene:", genes, index=genes.index(default))
        fig2 = sc.pl.umap(pdata, color=gene, use_raw=(pdata.raw is not None),
                          return_fig=True, show=False, color_map="viridis")
        st.pyplot(fig2); plt.close(fig2)

with tab_markers:
    st.subheader("Top marker genes per cluster (Wilcoxon)")
    clusters = ["All"] + sorted(markers["cluster"].unique().tolist())
    m1, m2 = st.columns([1, 2])
    chosen = m1.selectbox("Filter by cluster:", clusters)
    search = m2.text_input("Search for a gene:", "")
    view = markers
    if chosen != "All":
        view = view[view["cluster"] == chosen]
    if search:
        view = view[view["gene"].str.contains(search, case=False, na=False)]
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.caption(f"{len(view):,} rows.")
