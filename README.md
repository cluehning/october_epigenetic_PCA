# Epigenetic PCA: Extracting Latent Dimensions of Human Epigenetic Variation Using PCA

This project asks whether a collection of Roadmap Epigenomics signal tracks can
be summarized by a smaller set of orthogonal patterns. It connects the geometry
of PCA to genomic measurements, tests three numerical implementations, and
produces figures plus a report from the data that were actually downloaded.

## Mathematical background

The mathematical appendix explains the PCA concepts and derives the formulas
used in the project, including covariance eigenvectors, the SVD, explained
variance, projection, and reconstruction error. See
[E_PCA_Math_Notes](https://github.com/cluehning/october_epigenetic_PCA/blob/main/E_PCA%20I%20Math%20Notes.pdf).

## Biological context

Epigenetics concerns changes in gene regulation that do not alter the DNA
sequence. DNA methylation is the addition of methyl groups to DNA, often at
cytosines; its association with transcription depends on genomic context.

Histones are proteins around which DNA is packaged. Chemical modifications to
histones, including H3K4me3 and H3K27ac, are associated with regulatory states
such as active promoters or enhancers. These associations are useful signals,
not direct measurements of gene expression or proof of function.

Each cell type uses different regulatory programs across millions of genomic
positions. A signal track samples those positions, so even a single epigenetic
mark yields tens of thousands of measurements per sample. PCA asks whether
many correlated measurements can be approximated by a few dominant patterns.
Those patterns can reveal similarity among the measured epigenomes, while
necessarily discarding some information.

## Project layout

```text
src/epigenomics_pca/   Roadmap acquisition, matrix processing, PCA, plots, CLI
tests/                 Offline tests for PCA, processing, and analysis
notebooks/             Guided undergraduate research notebook
data/                  Downloaded tracks and processed matrices (not versioned)
outputs/               Catalog, tables, figures, diagnostics, and generated report
```

## Setup and run

Python 3.10 or newer is required. Install the package and its test extra:

```powershell
python -m pip install -e ".[test]"
```

Inspect the live Roadmap catalog without downloading signal tracks:

```powershell
python -m epigenomics_pca discover
```

Run the complete workflow using the curated default EID subset, H3K4me3, and
100 kb genomic windows:

```powershell
python -m epigenomics_pca all
```

**Download size:** a live HTTP-header check of the 17 default H3K4me3 tracks
reported about 8.86 GB (8.25 GiB), before processed matrices and figures. Sizes
may change. Discovery does not download signal files. For a smaller exploratory
run, use four samples spanning stem, blood, brain, and heart biosamples (about
2.13 GB in the same size check):

```powershell
python -m epigenomics_pca all --eids E003,E029,E073,E083 --marks H3K4me3 --bin-size 100000
```

To query approximate 100 kb bins from Roadmap's precomputed zoom summaries
directly over HTTP instead of staging
those full files, run:

```powershell
python -m epigenomics_pca remote --eids E003,E029,E073,E083 --marks H3K4me3 --bin-size 100000 --output-dir outputs/remote-four
```

This streamed mode is faster but uses approximate zoom-level means; the local
download/process mode uses exact summaries. The streamed analysis writes its
selected URLs and query settings to
`outputs/remote-four/source_manifest.csv` and generates the same PCA tables,
figures, and report. It requires a stable connection to the Roadmap server.

The default selection contains 17 Roadmap EIDs across 11 metadata `GROUP`
labels. The live catalog is checked before download; unavailable sample/mark
combinations are not silently substituted. Override the selection with
comma-separated IDs and marks as needed. A four-sample run has at most three
nonzero PCs, so it is suitable for testing the workflow, not for broad
population-level conclusions.

The workflow has separate `discover`, `download`, `process`, and `analyze`
commands. `analyze` can be run on an existing `data/processed/X.parquet` matrix.
The notebook at `notebooks/epigenomics_pca.ipynb` follows the same analysis
after data preparation.

## Data and provenance

The pipeline inspects the Roadmap by-file-type index and the consolidated
`pval.signal.bigwig` track directory. It fetches the official EID metadata,
records available track filenames, and filters selected samples to tracks that
exist for every requested mark.

- Roadmap index: <https://egg2.wustl.edu/roadmap/data/byFileType/>
- Consolidated signal tracks:
  <https://egg2.wustl.edu/roadmap/data/byFileType/signal/consolidated/macs2signal/pval/>
- Roadmap EID metadata:
  <https://egg2.wustl.edu/roadmap/data/byFileType/metadata/EID_metadata.tab>

Each download is recorded in `data/raw/download_manifest.csv` with source URL,
retrieval timestamp, byte count, and SHA-256. The catalog and selected-sample
table are written under `outputs/` and `data/samples/` respectively. These
external data are not redistributed with the code; Roadmap source terms and
attribution apply.

## Processing choices

Each bigWig is summarized on chromosome-specific windows with an approximately
100 kb width (customizable). The resulting matrix has one row per EID and one
column per mark/window pair. Unreported bigWig intervals are represented as
zero during summarization. Remaining non-finite values are median-imputed,
signal is compressed with `log1p`, constant columns are dropped, and each
feature is z-scored across samples. Both raw and normalized matrices are saved
as Parquet files.

This normalization emphasizes relative cross-sample differences at each
window. It can amplify low-signal noise, so interpretation should be compared
with raw-signal summaries and alternative preprocessing choices in a larger
study.

## Analysis outputs

`outputs/` contains explained variance, all retained loadings and scores, top
absolute loadings, reconstruction error by retained rank, metadata
associations, diagnostics, and `REPORT.md`. The figures directory contains:

- Scree and cumulative variance plot
- PC1/PC2, PC1/PC3, and PC2/PC3 scatter plots
- Interactive 3D Plotly embedding
- Top genomic-window loadings for the first three PCs
- Principal-component score correlation heatmap
- K-means clusters displayed in PCA space, with assignments

The analysis reports categorical eta-squared as a descriptive measure of
component-score separation by the Roadmap `GROUP`, `ANATOMY`, and `STD_NAME`
labels. It is not a significance test or a causal biological interpretation.
No rose diagram is included: genomic windows and PCA scores have no natural
circular ordering or angle, so a polar display would imply meaning that is not
present in the data.

The number of retained dimensions and participation-ratio effective dimension
are distinct quantities. The latter is an eigenvalue-spectrum summary, not a
claim that the data have exactly that many independent biological mechanisms.
See `MATHEMATICAL_APPENDIX.md` for derivations and `REPORT.md` for the research
protocol and interpretation boundaries. The generated results report replaces
`outputs/REPORT.md` only after an analysis has run.

## Tests

```powershell
pytest -q
```

The test suite uses synthetic or mocked data and does not download Roadmap
tracks.
