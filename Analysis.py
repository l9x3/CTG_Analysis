#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "Missing dependencies. Install them with: pip install -r requirements.txt"
    ) from exc


DEFAULT_INPUT = "CTG_Dataset.csv"
DEFAULT_OUTPUT_DIR = "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze CTG_Dataset.csv and store summary results and plots to an output folder."
        )
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Path to the CTG dataset CSV (default: CTG_Dataset.csv).",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store analysis outputs (default: ./output).",
    )
    return parser.parse_args()


def sanitize_filename(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return sanitized or "column"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_summary_statistics(df: pd.DataFrame, output_dir: Path) -> None:
    summary = df.describe(include="all").T
    summary.to_csv(output_dir / "summary_statistics.csv")
    summary.to_json(output_dir / "summary_statistics.json", orient="index", indent=2)


def save_missing_values(df: pd.DataFrame, output_dir: Path) -> None:
    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = (
        pd.DataFrame({"missing_count": missing, "missing_percent": missing_pct})
        .reset_index()
        .rename(columns={"index": "column"})
    )
    missing_df.to_csv(output_dir / "missing_values.csv", index=False)


def save_correlations(df: pd.DataFrame, output_dir: Path, plots_dir: Path) -> None:
    correlation = df.corr(numeric_only=True)
    correlation.to_csv(output_dir / "correlation.csv")
    if correlation.empty:
        return
    size = max(10, int(len(correlation.columns) * 0.4))
    fig, ax = plt.subplots(figsize=(size, size))
    cax = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(correlation.columns)))
    ax.set_yticks(range(len(correlation.columns)))
    ax.set_xticklabels(correlation.columns, rotation=90)
    ax.set_yticklabels(correlation.columns)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(plots_dir / "correlation_heatmap.png", dpi=200)
    plt.close(fig)


def save_class_distributions(df: pd.DataFrame, output_dir: Path, plots_dir: Path) -> None:
    for column in ("CLASS", "NSP"):
        if column not in df.columns:
            continue
        counts = df[column].value_counts(dropna=False).sort_index()
        counts.to_csv(output_dir / f"{column.lower()}_distribution.csv", header=["count"])

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(counts.index.astype(str), counts.values, color="#4C78A8")
        ax.set_title(f"{column} distribution")
        ax.set_xlabel(column)
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(plots_dir / f"{column.lower()}_distribution.png", dpi=150)
        plt.close(fig)


def save_histograms(df: pd.DataFrame, plots_dir: Path) -> None:
    hist_dir = plots_dir / "histograms"
    ensure_dir(hist_dir)
    for column in df.columns:
        series = df[column].dropna()
        if series.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(series, bins=30, color="#72B7B2", edgecolor="black")
        ax.set_title(f"{column} histogram")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
        fig.tight_layout()
        filename = sanitize_filename(column)
        fig.savefig(hist_dir / f"{filename}.png", dpi=150)
        plt.close(fig)


def write_metadata(
    input_path: Path, output_dir: Path, df: pd.DataFrame, plots_dir: Path
) -> None:
    metadata = {
        "input_path": str(input_path.resolve()),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "output_dir": str(output_dir.resolve()),
        "plots_dir": str(plots_dir.resolve()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_file():
        raise SystemExit(f"Input file not found: {input_path}")

    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    ensure_dir(output_dir)
    ensure_dir(plots_dir)

    df = pd.read_csv(input_path, encoding="utf-8-sig")
    df = df.apply(pd.to_numeric, errors="coerce")

    save_summary_statistics(df, output_dir)
    save_missing_values(df, output_dir)
    save_correlations(df, output_dir, plots_dir)
    save_class_distributions(df, output_dir, plots_dir)
    save_histograms(df, plots_dir)
    write_metadata(input_path, output_dir, df, plots_dir)

    print(f"Analysis complete. Outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
