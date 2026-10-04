"""Roadmap Epigenomics catalog discovery and reproducible track downloads."""

from __future__ import annotations

import hashlib
import html.parser
import json
import re
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


DATA_ROOT = "https://egg2.wustl.edu/roadmap/data/byFileType/"
SIGNAL_ROOT = DATA_ROOT + "signal/consolidated/macs2signal/pval/"
METADATA_URL = DATA_ROOT + "metadata/EID_metadata.tab"
DEFAULT_EIDS = [
    "E003", "E007", "E017", "E029", "E032", "E034", "E046", "E057",
    "E066", "E073", "E083", "E096", "E107", "E116", "E122", "E125", "E128",
]
DEFAULT_MARKS = ["H3K4me3"]
FILE_PATTERN = re.compile(r"^(E\d{3})-(.+)\.pval\.signal\.bigwig$")


class _LinkParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)


def _read_url(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "epigenomics-pca/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def fetch_metadata(destination: Path | None = None) -> pd.DataFrame:
    """Fetch and optionally cache the official EID-to-biosample metadata table."""
    content = _read_url(METADATA_URL)
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    from io import BytesIO

    return pd.read_csv(BytesIO(content), sep="\t", dtype=str).fillna("")


def list_signal_tracks() -> pd.DataFrame:
    """Inspect the live signal directory and parse EID / histone-mark filenames."""
    parser = _LinkParser()
    parser.feed(_read_url(SIGNAL_ROOT).decode("utf-8", errors="replace"))
    records = []
    for link in parser.links:
        match = FILE_PATTERN.match(link.rsplit("/", 1)[-1])
        if match:
            eid, mark = match.groups()
            records.append({"EID": eid, "mark": mark, "filename": link.rsplit("/", 1)[-1]})
    if not records:
        raise RuntimeError(f"No signal files found at {SIGNAL_ROOT}")
    return pd.DataFrame(records).drop_duplicates(["EID", "mark"]).sort_values(["EID", "mark"])


def build_catalog(metadata: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
    """Join available tracks with the authoritative Roadmap biosample labels."""
    grouped = tracks.groupby("EID")["mark"].agg(lambda values: ",".join(sorted(set(values))))
    catalog = metadata.merge(grouped.rename("available_marks"), left_on="EID", right_index=True, how="left")
    catalog["available_marks"] = catalog["available_marks"].fillna("")
    return catalog


def choose_samples(catalog: pd.DataFrame, tracks: pd.DataFrame, eids: list[str], marks: list[str]) -> pd.DataFrame:
    """Keep curated epigenomes only when every requested signal track exists."""
    available = set(zip(tracks["EID"], tracks["mark"], strict=False))
    rows = catalog.set_index("EID", drop=False)
    selected = []
    for eid in eids:
        if eid not in rows.index:
            continue
        missing = [mark for mark in marks if (eid, mark) not in available]
        if not missing:
            row = rows.loc[eid]
            selected.append({
                "EID": eid,
                "GROUP": row.get("GROUP", ""),
                "ANATOMY": row.get("ANATOMY", ""),
                "STD_NAME": row.get("STD_NAME", ""),
                "marks": ",".join(marks),
            })
    return pd.DataFrame(selected)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_tracks(
    selected: pd.DataFrame,
    tracks: pd.DataFrame,
    output_dir: Path,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Download chosen bigWigs with temporary-file safety and SHA-256 provenance."""
    output_dir.mkdir(parents=True, exist_ok=True)
    track_names = tracks.set_index(["EID", "mark"])["filename"]
    manifest = []
    for row in selected.itertuples(index=False):
        for mark in row.marks.split(","):
            filename = track_names.loc[(row.EID, mark)]
            destination = output_dir / filename
            url = SIGNAL_ROOT + filename
            if overwrite or not destination.exists():
                partial = destination.with_suffix(destination.suffix + ".partial")
                request = urllib.request.Request(url, headers={"User-Agent": "epigenomics-pca/0.1"})
                try:
                    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as target:
                        shutil.copyfileobj(response, target, length=1024 * 1024)
                    partial.replace(destination)
                finally:
                    partial.unlink(missing_ok=True)
            manifest.append({
                "EID": row.EID,
                "mark": mark,
                "filename": filename,
                "url": url,
                "bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
                "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            })
    result = pd.DataFrame(manifest)
    result.to_csv(output_dir / "download_manifest.csv", index=False)
    (output_dir / "source_metadata.json").write_text(
        json.dumps({"metadata_url": METADATA_URL, "signal_index_url": SIGNAL_ROOT}, indent=2),
        encoding="utf-8",
    )
    return result