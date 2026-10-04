"""PCA implementations used to teach and verify the linear algebra."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA


FloatArray = NDArray[np.float64]


@dataclass
class PCAResult:
    scores: FloatArray
    loadings: FloatArray
    eigenvalues: FloatArray
    explained_variance_ratio: FloatArray


def _validate_matrix(matrix: FloatArray) -> FloatArray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError("PCA requires a 2D matrix with at least two rows and columns")
    if not np.isfinite(values).all():
        raise ValueError("PCA input must contain only finite values")
    return values


def _result(centered: FloatArray, loadings: FloatArray, eigenvalues: FloatArray) -> PCAResult:
    scores = centered @ loadings
    total_variance = float(np.sum(eigenvalues))
    ratios = eigenvalues / total_variance if total_variance else np.zeros_like(eigenvalues)
    return PCAResult(scores, loadings, eigenvalues, ratios)


def pca_sklearn(matrix: FloatArray, n_components: int | None = None) -> PCAResult:
    """Fit sklearn PCA, using the full SVD for a directly comparable result."""
    values = _validate_matrix(matrix)
    component_count = n_components or min(values.shape[0] - 1, values.shape[1])
    model = PCA(n_components=component_count, svd_solver="full")
    scores = model.fit_transform(values)
    return PCAResult(
        scores=scores,
        loadings=model.components_.T,
        eigenvalues=model.explained_variance_,
        explained_variance_ratio=model.explained_variance_ratio_,
    )


def pca_covariance(matrix: FloatArray, n_components: int | None = None) -> PCAResult:
    """Use the dual sample covariance XX^T/(n-1), then recover feature loadings."""
    values = _validate_matrix(matrix)
    centered = values - values.mean(axis=0, keepdims=True)
    component_count = n_components or min(values.shape[0] - 1, values.shape[1])
    gram = centered @ centered.T / (values.shape[0] - 1)
    eigenvalues, sample_vectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1][:component_count]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    sample_vectors = sample_vectors[:, order]
    denominators = np.sqrt((values.shape[0] - 1) * eigenvalues)
    loadings = np.zeros((values.shape[1], component_count), dtype=np.float64)
    nonzero = denominators > np.finfo(np.float64).eps
    loadings[:, nonzero] = centered.T @ sample_vectors[:, nonzero] / denominators[nonzero]
    return _result(centered, loadings, eigenvalues)


def pca_svd(matrix: FloatArray, n_components: int | None = None) -> PCAResult:
    """Compute PCA directly from the thin SVD of the centered data matrix."""
    values = _validate_matrix(matrix)
    centered = values - values.mean(axis=0, keepdims=True)
    component_count = n_components or min(values.shape[0] - 1, values.shape[1])
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = singular_values[:component_count] ** 2 / (values.shape[0] - 1)
    loadings = right_vectors[:component_count].T
    return _result(centered, loadings, eigenvalues)


def relative_reconstruction_error(matrix: FloatArray, result: PCAResult) -> float:
    """Return ||X - X_k||_F / ||X||_F after centering X by feature."""
    values = _validate_matrix(matrix)
    centered = values - values.mean(axis=0, keepdims=True)
    reconstruction = result.scores @ result.loadings.T
    denominator = np.linalg.norm(centered)
    return float(np.linalg.norm(centered - reconstruction) / denominator) if denominator else 0.0


def effective_dimension(eigenvalues: FloatArray) -> float:
    """Participation ratio: (sum lambda)^2 / sum(lambda^2)."""
    values = np.asarray(eigenvalues, dtype=np.float64)
    squared_sum = float(np.sum(values**2))
    return float(np.sum(values) ** 2 / squared_sum) if squared_sum else 0.0
