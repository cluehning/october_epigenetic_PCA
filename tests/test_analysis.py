import json

import numpy as np
import pandas as pd

from epigenomics_pca.analysis import run_pca_analysis
from epigenomics_pca.visualization import _label_palette, write_report


def test_label_palette_supports_more_categories_than_fixed_color_sets():
    labels = pd.Series([f"cell-type-{index}" for index in range(17)])

    palette = _label_palette(labels)

    assert len(palette) == 17
    assert len(set(palette.values())) == 17


def test_analysis_writes_outputs_and_associates_metadata_by_eid(tmp_path):
    rng = np.random.default_rng(2026)
    matrix = pd.DataFrame(
        rng.normal(size=(8, 12)),
        index=[f"E{index:03}" for index in range(8)],
        columns=[
            f"mark|chr1:{index * 100}-{(index + 1) * 100}"
            for index in range(12)
        ],
    )
    matrix.index.name = "EID"
    metadata = pd.DataFrame({
        "EID": matrix.index[::-1],
        "GROUP": ["group-a"] * 4 + ["group-b"] * 4,
        "ANATOMY": ["tissue-a"] * 4 + ["tissue-b"] * 4,
        "STD_NAME": [f"sample-{index}" for index in range(8)],
    })

    outputs = run_pca_analysis(matrix, metadata, tmp_path)

    assert outputs["scores"].shape == (8, 7)
    assert outputs["associations"]["metadata_field"].eq("GROUP").any()
    unique_label_rows = outputs["associations"].query(
        "metadata_field == 'STD_NAME'"
    )
    assert unique_label_rows["saturated"].all()
    assert unique_label_rows["eta_squared"].isna().all()
    assert (tmp_path / "pca_scores.csv").exists()
    assert (tmp_path / "pca_loadings.csv").exists()
    assert (tmp_path / "explained_variance.csv").exists()
    assert (tmp_path / "reconstruction_errors.csv").exists()
    reconstruction_errors = outputs["reconstruction_errors"][
        "relative_reconstruction_error"
    ]
    assert reconstruction_errors.is_monotonic_decreasing
    assert (
        outputs["diagnostics"]["rank_3_reconstruction_error"]
        < outputs["diagnostics"]["rank_1_reconstruction_error"]
    )
    diagnostics = json.loads((tmp_path / "diagnostics.json").read_text())
    reconstruction_difference = diagnostics[
        "method_max_reconstruction_difference"
    ]["svd"]
    assert reconstruction_difference < 1e-8
    report_path = tmp_path / "REPORT.md"
    write_report(outputs, metadata, report_path)
    report = report_path.read_text(encoding="utf-8")
    assert "at rank 1" in report
    assert "at rank 2" in report
    assert "at rank 3" in report
