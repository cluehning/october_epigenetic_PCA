# Project Workflow

```mermaid
flowchart TD
    A[Roadmap file index and metadata] --> B[Discover available signal tracks]
    B --> C[Select EIDs and histone marks]
    C --> D[Download bigWig tracks]
    D --> E[SHA-256 provenance manifest]
    D --> F[Summarize tracks on shared genomic windows]
    F --> G[Sample by feature raw matrix]
    G --> H[Impute, log1p, drop constants, z-score]
    H --> I[Normalized matrix X]
    I --> J[scikit-learn PCA]
    I --> K[Covariance eigendecomposition]
    I --> L[Thin SVD]
    J --> M[Compare spectra and reconstructions]
    K --> M
    L --> M
    M --> N[Variance, scores, loadings, error, effective dimension]
    N --> O[Figures, clusters, metadata associations]
    O --> P[Data-generated report]
```

The end-to-end command is `python -m epigenomics_pca all`. Each intermediate
matrix and report is written to `data/` or `outputs/`; external tracks are not
included in version control.