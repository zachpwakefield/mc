"""Compare multiple MLP-Cox model runs side-by-side.

The script expects each run directory to contain a ``models`` folder with
``metrics.json`` files arranged as ``models/<cancer>/<modality>/metrics.json``.
It loads the ``test_cidx`` (and optionally ``test_roc_auc`` and
``test_avg_precision``) for each run, produces quantitative summaries, and
exports comparison plots.

Example usage:
    python compare_runs.py \
        --run-dirs \
            /projectnb2/evolution/zwakefield/tcga/cancer_learning/single_input/model_outputs_12_1_mad10000 \
            /projectnb2/evolution/zwakefield/tcga/cancer_learning/single_input/model_outputs_12_1_unique10 \
            /projectnb2/evolution/zwakefield/tcga/cancer_learning/single_input/model_outputs_12_1_nonzero10 \
        --run-labels mad10000 unique10 nonzero10 \
        --plot-dir ./plots/comparisons
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _float_or_none(metrics: dict, key: str) -> Optional[float]:
    val = metrics.get(key)
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def load_run_metrics(run_dir: Path, label: str) -> pd.DataFrame:
    """Load metrics from a single run directory."""

    models_dir = run_dir / "models"
    if not models_dir.is_dir():
        raise RuntimeError(f"Expected models directory under run dir: {models_dir}")

    records: List[dict] = []
    for metrics_file in models_dir.glob("*/*/metrics.json"):
        cancer = metrics_file.parents[1].name
        modality = metrics_file.parent.name

        try:
            with metrics_file.open() as fp:
                metrics = json.load(fp)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"⚠️  Skipping unreadable file {metrics_file}: {exc}")
            continue

        test_cidx = _float_or_none(metrics, "test_cidx")
        test_roc_auc = _float_or_none(metrics, "test_roc_auc")
        test_avg_precision = _float_or_none(metrics, "test_avg_precision")

        if test_cidx is None:
            print(f"⚠️  No usable test_cidx in {metrics_file}")
            continue

        records.append(
            {
                "cancer": cancer,
                "modality": modality,
                "test_cidx": test_cidx,
                "test_roc_auc": test_roc_auc,
                "test_avg_precision": test_avg_precision,
                "run": label,
                "path": str(metrics_file),
            }
        )

    if not records:
        raise RuntimeError(f"No metrics.json files were successfully read for {run_dir}")

    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _set_style() -> None:
    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=0.9)


def _save_fig(fig: plt.Figure, plot_dir: Path, name: str) -> Path:
    plot_dir.mkdir(parents=True, exist_ok=True)
    path = plot_dir / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"[saved plot] {path}")
    return path


def plot_overall_distribution(df: pd.DataFrame, plot_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="run", y="test_cidx", ax=ax, linewidth=1)
    ax.set(title="Overall test c-index distribution", xlabel="Run", ylabel="Test c-index")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return _save_fig(fig, plot_dir, "overall_test_cidx")


def plot_per_cancer(df: pd.DataFrame, plot_dir: Path) -> Path:
    order = (
        df.groupby("cancer")["test_cidx"]
        .median()
        .sort_values(ascending=False)
        .index
    )

    fig, ax = plt.subplots(figsize=(16, 6))
    sns.boxplot(data=df, x="cancer", y="test_cidx", hue="run", order=order, ax=ax, linewidth=1)
    ax.set(title="Test c-index across cancers (per run)", xlabel="Cancer type", ylabel="Test c-index")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    return _save_fig(fig, plot_dir, "per_cancer_test_cidx")


def plot_winner_counts(df: pd.DataFrame, plot_dir: Path) -> Path:
    best_rows = df.loc[df.groupby("cancer")["test_cidx"].idxmax()]
    best_count = best_rows["run"].value_counts().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=best_count.index, y=best_count.values, ax=ax)
    ax.set(title="Which run wins per cancer?", xlabel="Run", ylabel="Number of cancers")
    ax.bar_label(ax.containers[0], padding=3)
    fig.tight_layout()
    return _save_fig(fig, plot_dir, "cancer_winners")


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def print_overall_summary(df: pd.DataFrame) -> None:
    summary = (
        df.groupby("run")["test_cidx"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .sort_values("median", ascending=False)
    )
    print("\n=== Overall test_cidx summary ===")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))


def print_per_cancer_winners(df: pd.DataFrame) -> None:
    best_rows = df.loc[df.groupby("cancer")["test_cidx"].idxmax()]
    counts = best_rows["run"].value_counts().sort_values(ascending=False)
    print("\n=== Per-cancer winners (by test_cidx) ===")
    print(counts.to_string())


# ---------------------------------------------------------------------------
# CLI & entry point
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multiple mlp_cox_job runs")
    parser.add_argument(
        "--run-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="List of run directories to compare (each must contain a models folder)",
    )
    parser.add_argument(
        "--run-labels",
        type=str,
        nargs="+",
        help="Optional labels for each run; defaults to directory names",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("./comparison_plots"),
        help="Directory where comparison plots will be written",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    labels: List[str]
    if args.run_labels:
        if len(args.run_labels) != len(args.run_dirs):
            raise SystemExit("--run-labels must match the number of --run-dirs")
        labels = list(args.run_labels)
    else:
        labels = [p.name for p in args.run_dirs]

    frames: List[pd.DataFrame] = []
    for run_dir, label in zip(args.run_dirs, labels):
        print(f"Loading metrics from {run_dir} (label={label})")
        frames.append(load_run_metrics(run_dir, label))

    df = pd.concat(frames, ignore_index=True)
    print(
        f"Loaded {len(df):,} metrics entries "
        f"({df['cancer'].nunique()} cancers, {df['modality'].nunique()} modalities, {df['run'].nunique()} runs)"
    )
    print(f"Runs: {sorted(df['run'].unique())}")
    print(f"Cancers: {sorted(df['cancer'].unique())}")

    _set_style()
    plot_overall_distribution(df, args.plot_dir)
    plot_per_cancer(df, args.plot_dir)
    plot_winner_counts(df, args.plot_dir)

    print_overall_summary(df)
    print_per_cancer_winners(df)


if __name__ == "__main__":
    main()
