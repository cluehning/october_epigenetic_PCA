"""Run and summarize the three PCA implementations on a processed matrix."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from epigenomics_pca.pca import (
    effective_dimension,
    pca_covariance,
    pca_sklearn,
    pca_svd,
    relative_reconstruction_error,
)


def categorical_associations(
    scores: pd.DataFrame, metadata: pd.DataFrame
) -> pd.DataFrame:
    """Measure PC score variation associated with metadata labels."""
    aligned = metadata.set_index("EID", drop=False).reindex(scores.index)
    records: list[dict[str, object]] = []
    for label in ("GROUP", "ANATOMY", "STD_NAME"):
        if label not in aligned:
            continue
        groups = aligned[label].fillna("").astype(str)
        valid = groups.ne("")
        for component in scores.columns:
            values = scores.loc[valid, component].to_numpy(dtype=float)
            categories = groups.loc[valid].to_numpy()
            if len(values) < 2:
                continue
            overall_mean = float(values.mean())
            total = float(np.sum((values - overall_mean) ** 2))
            between = 0.0
            for category in np.unique(categories):
                group_values = values[categories == category]
                mean_difference = group_values.mean() - overall_mean
                between += len(group_values) * float(mean_difference**2)
            category_count = int(np.unique(categories).size)
            saturated = category_count >= len(values)
            records.append({
                "metadata_field": label,
                "component": component,
                "eta_squared": (
                    np.nan if saturated else between / total if total else 0.0
                ),
                "category_count": category_count,
                "sample_count": int(len(values)),
                "saturated": saturated,
            })
    return pd.DataFrame(records)


def run_pca_analysis(
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    output_dir: Path,
    n_components: int | None = None,
    top_loadings: int = 20,
) -> dict[str, object]:
    """Compute PCA three ways and write comparable scores and diagnostics."""
    if not matrix.index.is_unique:
        raise ValueError("Sample identifiers in the matrix must be unique")
    if matrix.shape[0] < 3:
        raise ValueError("At least three samples are required for PCA.")
    values = matrix.to_numpy(dtype=np.float64)
    component_count = n_components or min(values.shape[0] - 1, values.shape[1])
    results = {
        "sklearn": pca_sklearn(values, component_count),
        "covariance": pca_covariance(values, component_count),
        "svd": pca_svd(values, component_count),
    }
    reference = results["sklearn"]
    reference_reconstruction = reference.scores @ reference.loadings.T
    reconstruction_differences: dict[str, float] = {}
    for method, result in results.items():
        eigenvalues_match = np.allclose(
            result.eigenvalues, reference.eigenvalues, rtol=1e-8, atol=1e-10
        )
        if not eigenvalues_match:
            raise ArithmeticError(
                f"{method} PCA eigenvalues differ from sklearn PCA"
            )
        candidate_reconstruction = result.scores @ result.loadings.T
        reconstruction_difference = float(
            np.max(np.abs(candidate_reconstruction - reference_reconstruction))
        )
        reconstruction_differences[method] = reconstruction_difference
        if not np.allclose(
            candidate_reconstruction,
            reference_reconstruction,
            rtol=1e-7,
            atol=1e-9,
        ):
            raise ArithmeticError(
                f"{method} PCA reconstruction differs from sklearn PCA"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    component_names = [
        f"PC{index + 1}" for index in range(len(reference.eigenvalues))
    ]
    scores = pd.DataFrame(
        reference.scores, index=matrix.index, columns=component_names
    )
    scores.index.name = matrix.index.name or "EID"
    annotated_scores = scores.join(
        metadata.set_index("EID", drop=False), how="left"
    )
    annotated_scores.to_csv(output_dir / "pca_scores.csv")

    loadings = pd.DataFrame(
        reference.loadings, index=matrix.columns, columns=component_names
    )
    loadings.index.name = "feature"
    loadings.to_csv(output_dir / "pca_loadings.csv")
    top_rows: list[dict[str, object]] = []
    for component in component_names:
        ordered = loadings[component].abs().nlargest(top_loadings).index
        for rank, feature in enumerate(ordered, start=1):
            top_rows.append({
                "component": component,
                "rank": rank,
                "feature": feature,
                "loading": float(loadings.at[feature, component]),
            })
    pd.DataFrame(top_rows).to_csv(output_dir / "top_loadings.csv", index=False)

    variance = pd.DataFrame({
        "component": component_names,
        "eigenvalue": reference.eigenvalues,
        "explained_variance_ratio": reference.explained_variance_ratio,
    })
    variance["cumulative_variance_ratio"] = variance[
        "explained_variance_ratio"
    ].cumsum()
    variance.to_csv(output_dir / "explained_variance.csv", index=False)

    centered = values - values.mean(axis=0, keepdims=True)
    centered_norm = np.linalg.norm(centered)
    reconstruction_rows = []
    for retained_count in range(1, len(reference.eigenvalues) + 1):
        reconstruction = (
            reference.scores[:, :retained_count]
            @ reference.loadings[:, :retained_count].T
        )
        error = (
            float(np.linalg.norm(centered - reconstruction) / centered_norm)
            if centered_norm
            else 0.0
        )
        reconstruction_rows.append({
            "retained_components": retained_count,
            "cumulative_variance_ratio": float(
                variance["cumulative_variance_ratio"].iloc[retained_count - 1]
            ),
            "relative_reconstruction_error": error,
        })
    reconstruction_errors = pd.DataFrame(reconstruction_rows)
    reconstruction_errors.to_csv(
        output_dir / "reconstruction_errors.csv", index=False
    )

    association = categorical_associations(scores, metadata)
    association.to_csv(output_dir / "metadata_associations.csv", index=False)
    diagnostics = {
        "sample_count": int(matrix.shape[0]),
        "feature_count": int(matrix.shape[1]),
        "component_count": int(len(reference.eigenvalues)),
        "effective_dimension": effective_dimension(reference.eigenvalues),
        "relative_reconstruction_error": relative_reconstruction_error(
            values, reference
        ),
        "rank_1_reconstruction_error": float(
            reconstruction_errors["relative_reconstruction_error"].iloc[0]
        ),
        "rank_3_reconstruction_error": float(
            reconstruction_errors["relative_reconstruction_error"].iloc[
                min(2, len(reconstruction_errors) - 1)
            ]
        ),
        "method_max_eigenvalue_difference": {
            method: float(
                np.max(np.abs(result.eigenvalues - reference.eigenvalues))
            )
            for method, result in results.items()
        },
        "method_max_reconstruction_difference": reconstruction_differences,
        "sample_ids": [str(sample) for sample in matrix.index],
        "features": [str(feature) for feature in matrix.columns],
    }
    diagnostics_path = output_dir / "diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    return {
        "scores": scores,
        "loadings": loadings,
        "variance": variance,
        "reconstruction_errors": reconstruction_errors,
        "associations": association,
        "diagnostics": diagnostics,
        "methods": results,
    }
