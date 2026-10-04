"""Command-line entry point for the Roadmap PCA research workflow."""

from __future__ import annotations

import argparse
import importlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from epigenomics_pca.analysis import run_pca_analysis
from epigenomics_pca.processing import (
    build_raw_matrix,
    build_remote_matrix,
    save_processed_dataset,
)
from epigenomics_pca.roadmap import (
    DEFAULT_EIDS,
    DEFAULT_MARKS,
    SIGNAL_ROOT,
    build_catalog,
    choose_samples,
    download_tracks,
    fetch_metadata,
    list_signal_tracks,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="epigenomics-pca",
        description="Reproducible Roadmap Epigenomics PCA workflow",
    )
    parser.add_argument(
        "command",
        choices=(
            "discover",
            "download",
            "process",
            "analyze",
            "remote",
            "all",
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--eids", default=",".join(DEFAULT_EIDS))
    parser.add_argument("--marks", default=",".join(DEFAULT_MARKS))
    parser.add_argument("--bin-size", type=int, default=100_000)
    parser.add_argument("--components", type=int)
    return parser


def _discover(
    data_dir: Path, output_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata_path = data_dir / "metadata" / "EID_metadata.tab"
    metadata = fetch_metadata(metadata_path)
    tracks = list_signal_tracks()
    catalog = build_catalog(metadata, tracks)
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(output_dir / "roadmap_catalog.csv", index=False)
    tracks.to_csv(output_dir / "available_signal_tracks.csv", index=False)
    print(f"Catalog written to {output_dir / 'roadmap_catalog.csv'}")
    print(f"Discovered {len(tracks):,} available EID/mark tracks.")
    return catalog, tracks


def _select(
    catalog: pd.DataFrame, tracks: pd.DataFrame, eids: str, marks: str
) -> pd.DataFrame:
    requested_eids = [
        value.strip() for value in eids.split(",") if value.strip()
    ]
    requested_marks = [
        value.strip() for value in marks.split(",") if value.strip()
    ]
    selected = choose_samples(catalog, tracks, requested_eids, requested_marks)
    if selected.empty:
        raise RuntimeError(
            "No requested samples have every selected mark. Check the catalog "
            "and --eids/--marks arguments."
        )
    return selected


def _download_and_save_selection(
    data_dir: Path,
    catalog: pd.DataFrame,
    tracks: pd.DataFrame,
    eids: str,
    marks: str,
) -> pd.DataFrame:
    selected = _select(catalog, tracks, eids, marks)
    selection_path = data_dir / "samples" / "selected_samples.csv"
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(selection_path, index=False)
    manifest = download_tracks(selected, tracks, data_dir / "raw")
    print(f"Downloaded {len(manifest)} signal tracks to {data_dir / 'raw'}")
    return selected


def _process(data_dir: Path, bin_size: int) -> None:
    selected_path = data_dir / "samples" / "selected_samples.csv"
    if not selected_path.exists():
        raise FileNotFoundError(
            "Run the download stage before processing tracks."
        )
    selected = pd.read_csv(selected_path, dtype=str).fillna("")
    marks = sorted(
        {mark for value in selected["marks"] for mark in value.split(",")}
    )
    raw = build_raw_matrix(
        selected, data_dir / "raw", marks, bin_size=bin_size
    )
    raw_path, normalized_path = save_processed_dataset(
        raw, selected, data_dir / "processed"
    )
    print(f"Raw matrix: {raw_path}; normalized matrix: {normalized_path}")


def _analyze(
    data_dir: Path, output_dir: Path, component_count: int | None
) -> None:
    from epigenomics_pca.visualization import create_figures, write_report

    matrix_path = data_dir / "processed" / "X.parquet"
    metadata_path = data_dir / "processed" / "sample_metadata.csv"
    if not matrix_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Run the process stage before analyzing the matrix."
        )
    matrix = pd.read_parquet(matrix_path)
    metadata = pd.read_csv(metadata_path, dtype=str).fillna("")
    results = run_pca_analysis(
        matrix, metadata, output_dir, n_components=component_count
    )
    create_figures(results, metadata, output_dir / "figures")
    write_report(results, metadata, output_dir / "REPORT.md")
    print(f"Analysis outputs written to {output_dir}")


def _remote_analysis(
    data_dir: Path,
    output_dir: Path,
    catalog: pd.DataFrame,
    tracks: pd.DataFrame,
    eids: str,
    requested_marks: str,
    bin_size: int,
    component_count: int | None,
) -> None:
    selected = _select(catalog, tracks, eids, requested_marks)
    selection_path = data_dir / "samples" / "selected_samples.csv"
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(selection_path, index=False)
    marks = [mark for mark in requested_marks.split(",") if mark]
    query_started = datetime.now(timezone.utc).isoformat()
    raw = build_remote_matrix(selected, tracks, marks, bin_size=bin_size)
    save_processed_dataset(raw, selected, data_dir / "processed")

    sources = [
        {
            "EID": sample.EID,
            "mark": mark,
            "filename": tracks.set_index(["EID", "mark"])
            .loc[(sample.EID, mark), "filename"],
            "url": SIGNAL_ROOT
            + tracks.set_index(["EID", "mark"])
            .loc[(sample.EID, mark), "filename"],
            "query_started_utc": query_started,
            "bin_size": bin_size,
            "summary": "zoom-level mean (approximate)",
            "exact": False,
        }
        for sample in selected.itertuples(index=False)
        for mark in marks
    ]
    manifest = pd.DataFrame(sources)
    processed_manifest = data_dir / "processed" / "remote_manifest.csv"
    processed_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(processed_manifest, index=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_dir / "source_manifest.csv", index=False)
    _analyze(data_dir, output_dir, component_count)


def main() -> None:
    args = _parser().parse_args()
    if args.command in ("analyze", "remote", "all"):
        try:
            importlib.import_module("plotly.express")
            importlib.import_module("epigenomics_pca.visualization")
        except ImportError as error:
            raise SystemExit(
                "Analysis requires the project dependencies. Install with "
                "python -m pip install -e ."
            ) from error
    if args.command in ("discover", "download", "remote", "all"):
        catalog, tracks = _discover(args.data_dir, args.output_dir)
    if args.command == "download":
        _download_and_save_selection(
            args.data_dir, catalog, tracks, args.eids, args.marks
        )
    elif args.command == "process":
        _process(args.data_dir, args.bin_size)
    elif args.command == "analyze":
        _analyze(args.data_dir, args.output_dir, args.components)
    elif args.command == "remote":
        _remote_analysis(
            args.data_dir,
            args.output_dir,
            catalog,
            tracks,
            args.eids,
            args.marks,
            args.bin_size,
            args.components,
        )
    elif args.command == "all":
        _download_and_save_selection(
            args.data_dir, catalog, tracks, args.eids, args.marks
        )
        _process(args.data_dir, args.bin_size)
        _analyze(args.data_dir, args.output_dir, args.components)


if __name__ == "__main__":
    main()
