from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pybigtools

from epigenomics_pca.processing import (
    build_raw_matrix,
    build_remote_matrix,
    extract_track_features,
    normalize_features,
)


def test_extract_track_features_uses_exact_bins_and_zero_fills():
    path = Path(f"_pybigtools_{uuid4().hex}.bigwig")
    try:
        writer = pybigtools.open(path.name, "w")
        writer.write(
            {"chr1": 300},
            [("chr1", 0, 100, 1.0), ("chr1", 200, 300, 3.0)],
        )

        names, values = extract_track_features(path, bin_size=100)

        assert names == ["chr1:0-100", "chr1:100-200", "chr1:200-300"]
        np.testing.assert_allclose(values, [1.0, 0.0, 3.0])
    finally:
        path.unlink(missing_ok=True)


def test_matrix_preserves_mark_and_window_alignment(tmp_path, monkeypatch):
    samples = pd.DataFrame({"EID": ["E001", "E002"]})
    for eid in samples["EID"]:
        for mark in ("H3K4me3", "H3K27ac"):
            (tmp_path / f"{eid}-{mark}.pval.signal.bigwig").touch()

    def fake_extract(path, bin_size):
        mark_values = [1.0, 2.0] if "H3K4me3" in path.name else [3.0, 4.0]
        sample_offset = 0.0 if "E001" in path.name else 10.0
        windows = ["chr1:0-100", "chr1:100-200"]
        return windows, np.array(mark_values) + sample_offset

    monkeypatch.setattr(
        "epigenomics_pca.processing.extract_track_features", fake_extract
    )
    matrix = build_raw_matrix(
        samples, tmp_path, ["H3K4me3", "H3K27ac"], bin_size=100
    )

    assert matrix.shape == (2, 4)
    assert matrix.columns[0] == "H3K4me3|chr1:0-100"
    assert matrix.columns[2] == "H3K27ac|chr1:0-100"
    assert matrix.loc["E002"].tolist() == [11.0, 12.0, 13.0, 14.0]


def test_remote_matrix_uses_available_urls_and_shared_features(monkeypatch):
    samples = pd.DataFrame({"EID": ["E001", "E002"]})
    tracks = pd.DataFrame({
        "EID": ["E001", "E002"],
        "mark": ["H3K4me3", "H3K4me3"],
        "filename": [
            "E001-H3K4me3.pval.signal.bigwig",
            "E002-H3K4me3.pval.signal.bigwig",
        ],
    })
    seen_urls = []
    exact_options = []

    def fake_extract(path, bin_size, exact=True):
        seen_urls.append(path)
        exact_options.append(exact)
        offset = 0.0 if "E001" in path else 10.0
        windows = ["chr1:0-100", "chr1:100-200"]
        return windows, np.array([1.0, 2.0]) + offset

    monkeypatch.setattr(
        "epigenomics_pca.processing.extract_track_features", fake_extract
    )
    matrix = build_remote_matrix(samples, tracks, ["H3K4me3"], bin_size=100)

    assert matrix.shape == (2, 2)
    assert matrix.index.tolist() == ["E001", "E002"]
    assert all(
        path.startswith("https://egg2.wustl.edu/roadmap/")
        for path in seen_urls
    )
    assert exact_options == [False, False]
    assert matrix.loc["E002"].tolist() == [11.0, 12.0]


def test_normalize_features_drops_constant_and_nonfinite_columns():
    matrix = pd.DataFrame({
        "variable": [0.0, 1.0, 3.0],
        "constant": [2.0] * 3,
        "missing": [np.nan] * 3,
    })

    normalized = normalize_features(matrix)

    assert list(normalized.columns) == ["variable"]
    assert np.isfinite(normalized.to_numpy()).all()
    np.testing.assert_allclose(
        normalized.mean(axis=0).to_numpy(), [0.0], atol=1e-12
    )
