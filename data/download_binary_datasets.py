from __future__ import annotations

import argparse
import csv
import urllib.request
from collections.abc import Iterable
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent


def _download_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8")


def _write_csv(path: Path, header: list[str], rows: Iterable[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def download_wdbc(output_dir: Path) -> Path:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    text = _download_text(url)
    header = [f"x{i}" for i in range(1, 31)] + ["y"]
    rows: list[list[object]] = []
    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        parts = raw_line.split(",")
        label = 1 if parts[1] == "M" else 0
        rows.append([*parts[2:], label])

    output = output_dir / "breast_cancer_wdbc.csv"
    _write_csv(output, header, rows)
    return output


def download_banknote(output_dir: Path) -> Path:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
    text = _download_text(url)
    header = [f"x{i}" for i in range(1, 5)] + ["y"]
    rows: list[list[object]] = []
    for raw_line in text.splitlines():
        if raw_line.strip():
            rows.append(raw_line.split(","))

    output = output_dir / "banknote_authentication.csv"
    _write_csv(output, header, rows)
    return output


def download_spambase(output_dir: Path) -> Path:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    text = _download_text(url)
    header = [f"x{i}" for i in range(1, 58)] + ["y"]
    rows: list[list[object]] = []
    for raw_line in text.splitlines():
        if raw_line.strip():
            rows.append(raw_line.split(","))

    output = output_dir / "spambase.csv"
    _write_csv(output, header, rows)
    return output


def download_iris(output_dir: Path) -> Path:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    text = _download_text(url)
    header = [f"x{i}" for i in range(1, 5)] + ["y"]
    rows: list[list[object]] = []
    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        parts = raw_line.split(",")
        label = 1 if parts[4] == "Iris-setosa" else 0
        rows.append([*parts[:4], label])

    output = output_dir / "iris.csv"
    _write_csv(output, header, rows)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and normalize public binary-classification datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR,
        help=f"Directory for normalized CSV files. Default: {DATA_DIR}",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir.resolve()
    outputs = [
        download_iris(output_dir),
        download_wdbc(output_dir),
        download_banknote(output_dir),
        download_spambase(output_dir),
    ]
    for output in outputs:
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
