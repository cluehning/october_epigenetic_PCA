"""Build sample-by-genomic-window matrices from Roadmap bigWig tracks."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


CANONICAL_CHROMOSOME = re.compile(r"^chr(?:[1-9][0-9]*|X|Y)$")


def extract_track_features(
    bigwig_path: Path | str,
    bin_size: int = 100_000,
    exact: bool = True,
) -> tuple[list[str], np.ndarray]:
    """Summarize a local or remote bigWig on exact, shared-width windows."""
    try:
        import pybigtools
    except ImportError as error:
        raise RuntimeError(
            "Install project dependencies, including pybigtools, first"
        ) from error

    names: list[str] = []
    values: list[float] = []
    with pybigtools.open(str(bigwig_path)) as bigwig:
        for chromosome, length in bigwig.chroms().items():
            if not CANONICAL_CHROMOSOME.match(chromosome):
                continue
            bin_count = max(1, int(np.ceil(length / bin_size)))
            edges = np.linspace(0, length, bin_count + 1, dtype=np.int64)
            starts, ends = edges[:-1], edges[1:]
            summary = bigwig.values(
                chromosome,
                0,
                length,
                bins=bin_count,
                summary="mean",
                exact=exact,
                fillna=0.0,
            )
            if summary is None or len(summary) != bin_count:
                raise RuntimeError(
                    f"Could not summarize {chromosome} in {bigwig_path}"
                )
            names.extend(
                f"{chromosome}:{start}-{end}"
                for start, end in zip(starts, ends, strict=True)
            )
            values.extend(float(value) for value in summary)
    if not names:
        raise ValueError(f"No canonical chromosomes found in {bigwig_path}")
    return names, np.asarray(values, dtype=np.float64)


def build_raw_matrix(
    selected_samples: pd.DataFrame,
    signal_dir: Path,
    marks: list[str],
    bin_size: int = 100_000,
) -> pd.DataFrame:
    """Create a sample-by-feature matrix with zero for unreported signal."""
    rows: list[np.ndarray] = []
    features_by_mark: list[list[str]] = []
    for sample_index, sample in enumerate(
        selected_samples.itertuples(index=False)
    ):
        sample_features: list[float] = []
        for mark_index, mark in enumerate(marks):
            path = signal_dir / f"{sample.EID}-{mark}.pval.signal.bigwig"
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing required Roadmap track: {path}"
                )
            names, values = extract_track_features(path, bin_size)
            current_names = [f"{mark}|{name}" for name in names]
            if sample_index == 0:
                features_by_mark.append(current_names)
            elif current_names != features_by_mark[mark_index]:
                raise ValueError(
                    f"Genome windows differ between tracks; mismatch in {path}"
                )
            sample_features.extend(values.tolist())
        rows.append(np.asarray(sample_features, dtype=np.float64))
    if not rows:
        raise ValueError("No samples or marks were selected")
    feature_names = [
        name for mark_features in features_by_mark for name in mark_features
    ]
    matrix = pd.DataFrame(
        rows,
        index=selected_samples["EID"].tolist(),
        columns=feature_names,
    )
    matrix.index.name = "EID"
    return matrix


def build_remote_matrix(
    selected_samples: pd.DataFrame,
    tracks: pd.DataFrame,
    marks: list[str],
    bin_size: int = 100_000,
) -> pd.DataFrame:
    """Build the same matrix by querying Roadmap bigWigs over HTTP ranges."""
    from epigenomics_pca.roadmap import SIGNAL_ROOT

    track_filenames = tracks.set_index(["EID", "mark"])["filename"]
    rows: list[np.ndarray] = []
    features_by_mark: list[list[str]] = []
    for sample_index, sample in enumerate(
        selected_samples.itertuples(index=False)
    ):
        sample_features: list[float] = []
        for mark_index, mark in enumerate(marks):
            filename = track_filenames.loc[(sample.EID, mark)]
            url = SIGNAL_ROOT + filename
            names, values = extract_track_features(
                url, bin_size, exact=False
            )
            current_names = [f"{mark}|{name}" for name in names]
            if sample_index == 0:
                features_by_mark.append(current_names)
            elif current_names != features_by_mark[mark_index]:
                raise ValueError(
                    f"Genome windows differ between remote tracks: {url}"
                )
            sample_features.extend(values.tolist())
        rows.append(np.asarray(sample_features, dtype=np.float64))
    if not rows:
        raise ValueError("No samples or marks were selected")
    feature_names = [
        name for mark_features in features_by_mark for name in mark_features
    ]
    matrix = pd.DataFrame(
        rows, index=selected_samples["EID"].tolist(), columns=feature_names
    )
    matrix.index.name = "EID"
    return matrix


def normalize_features(matrix: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p compression, imputation, and feature-wise z-scoring."""
    values = matrix.astype(np.float64).copy()
    values = values.replace([np.inf, -np.inf], np.nan)
    values = values.fillna(values.median(axis=0)).fillna(0.0)
    values = np.log1p(values.clip(lower=0.0))
    standard_deviation = values.std(axis=0, ddof=1)
    keep = standard_deviation > 0
    values = values.loc[:, keep]
    centered = values - values.mean(axis=0)
    return centered.divide(standard_deviation[keep], axis=1)


def save_processed_dataset(
    raw_matrix: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "X_raw.parquet"
    normalized_path = output_dir / "X.parquet"
    raw_matrix.to_parquet(raw_path)
    normalize_features(raw_matrix).to_parquet(normalized_path)
    sample_metadata.to_csv(output_dir / "sample_metadata.csv", index=False)
    return raw_path, normalized_path
