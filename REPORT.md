# Research Report: Extracting Latent Dimensions of Human Epigenetic Variation Using Principal Component Analysis

## Executive summary

This project provides a reproducible analysis of Roadmap Epigenomics histone
signal tracks. It asks whether measured genomic-window profiles can be
approximated by a small number of orthogonal directions, and whether those
directions align with recorded biosample labels. The code produces numerical
results only after it retrieves and processes the selected data; this checked-in
protocol intentionally makes no unobserved biological claims.

## Background and research questions

Epigenomic measurements describe molecular marks associated with regulation,
not changes to the underlying DNA sequence. This project begins with
H3K4me3, a histone modification commonly associated with active promoters.
Roadmap signal tracks represent a genome-wide assay-derived signal and are not
direct measurements of transcription. Thousands of genomic windows across
many samples create a high-dimensional matrix.

The analysis asks: how many features are measured; how quickly variance
accumulates across principal components; whether biosample groups separate in
score space; which genomic windows contribute strongly to each axis; and how
large a rank-k reconstruction error remains. Findings apply to this selected
subset, marker, genomic resolution, and preprocessing pipeline.

## Data and methods

The acquisition stage inspects the public Roadmap consolidated
`pval.signal.bigwig` directory and joins available EIDs to Roadmap metadata.
The default selection is a curated set of 17 EIDs and H3K4me3, filtered at run
time against the live index. A live header check measured about 8.86 GB total
for those 17 tracks (sizes may change); the four-sample exploratory option is
about 2.13 GB. Each track is summarized over approximately 100 kb windows. The
matrix rows are epigenomes; columns are mark/window pairs.

Unreported bigWig intervals are encoded as zero. Non-finite values are imputed
by feature median; signals are transformed using $\log(1+x)$, constant
features removed, then features centered and scaled to unit sample standard
deviation. A reproducible manifest records URL, retrieval time, size, and
SHA-256.

PCA is independently computed by scikit-learn, the dual sample covariance
eigendecomposition, and thin SVD. Eigenvalues and rank-k reconstructions are
checked numerically. Outputs include scores, loadings, variance ratios,
reconstruction error, participation-ratio effective dimension, descriptive
categorical eta-squared, and figures.

## Results

Run `python -m epigenomics_pca all` to create the data-dependent report at
`outputs/REPORT.md`. It reports actual sample and feature counts, variance
captured by the first three PCs, effective dimension, reconstruction error,
metadata associations, and leading feature loadings. No empirical values are
filled into this protocol before the Roadmap data are downloaded and analyzed.

The intended plain-language summary is calculated from the output: “Although
we measured thousands of epigenetic signals, the first few patterns summarize
the proportion of variation reported for this selected group of samples.”
Whether that proportion is large enough to call the selected matrix
approximately low-dimensional is an empirical result, not an assumption.

## Interpretation boundaries

PC score separation by `GROUP`, `ANATOMY`, or `STD_NAME` is summarized with
eta-squared, the fraction of score variation lying between metadata categories.
This is descriptive; it does not prove that tissue identity caused a pattern.
Prominent windows are associations in the measured signal and require genomic
annotation and independent biological validation before functional claims.

PCA is linear, sensitive to scaling and outliers, and ordered by total
variance rather than biological relevance. Batch effects, donor variation,
cell-type mixtures, assay quality, and biological heterogeneity may contribute
to dominant axes. Standardization can give noisy low-signal windows high
influence. Retaining only a few components necessarily loses information.

No rose diagram is used: genomic features and principal components have no
meaningful circular order or angle, and polar encoding would be misleading.

## Reproducibility

Install with `python -m pip install -e ".[test]"`; run `python -m
epigenomics_pca discover` to inspect the current catalog, or `python -m
epigenomics_pca all` for the complete workflow. Exact inputs and settings are
recorded by the output catalog, selected-sample file, manifest, and analysis
tables. The notebook and `MATHEMATICAL_APPENDIX.md` give the learning narrative
and derivations.