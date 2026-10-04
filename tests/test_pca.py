import numpy as np

from epigenomics_pca.pca import (
    effective_dimension,
    pca_covariance,
    pca_sklearn,
    pca_svd,
    relative_reconstruction_error,
)


def test_three_pca_methods_agree_on_spectrum_and_reconstruction():
    rng = np.random.default_rng(2026)
    matrix = rng.normal(size=(9, 14)) @ rng.normal(size=(14, 24))
    results = [pca_sklearn(matrix), pca_covariance(matrix), pca_svd(matrix)]

    for candidate in results[1:]:
        np.testing.assert_allclose(
            candidate.explained_variance_ratio,
            results[0].explained_variance_ratio,
            rtol=1e-10,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            relative_reconstruction_error(matrix, candidate),
            relative_reconstruction_error(matrix, results[0]),
            atol=1e-10,
        )

    assert 1.0 <= effective_dimension(results[0].eigenvalues) <= len(results[0].eigenvalues)
