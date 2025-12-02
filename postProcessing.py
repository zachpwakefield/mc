"""Post-process MLP-Cox model outputs.

This script consumes the run layout produced by ``mlp_cox_job_11_25.py``
(`ROOT_DIR` created at runtime) and generates summary plots for the
``metrics.json`` files saved under ``ROOT_DIR/models/<cancer>/<modality>``.

It summarises test concordance index distributions across cancers and
modalities, identifies the best modality per cancer, and exports plots
next to the run outputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_metrics(models_dir: Path) -> pd.DataFrame:
    """Load ``metrics.json`` files from ``models_dir``.

    The ``mlp_cox_job`` script writes metrics to
    ``ROOT_DIR/models/<cancer>/<modality>/metrics.json``. We capture the
    ``test_cidx`` value from each file and retain its origin for plotting.
    """

    records: List[dict] = []
    for metrics_file in models_dir.glob("*/*/metrics.json"):
        # metrics_file = models/<cancer>/<modality>/metrics.json
        cancer = metrics_file.parents[1].name
        modality = metrics_file.parent.name

        try:
            with metrics_file.open() as fp:
                metrics = json.load(fp)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"⚠️  Skipping unreadable file {metrics_file}: {exc}")
            continue

        test_cidx = metrics.get("test_cidx")
        if test_cidx is None:
            print(f"⚠️  No test_cidx in {metrics_file}")
            continue

        records.append(
            {
                "cancer": cancer,
                "modality": modality,
                "test_cidx": float(test_cidx),
                "path": str(metrics_file),
            }
        )

    if not records:
        raise RuntimeError("No metrics.json files were successfully read!")

    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _save_fig(fig: plt.Figure, plot_dir: Path, name: str) -> Path:
    """Save a Matplotlib figure into ``plot_dir`` with a timestamped name."""

    plot_dir.mkdir(parents=True, exist_ok=True)
    path = plot_dir / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"[saved plot] {path}")
    return path


def _set_style() -> None:
    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=0.9)


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------


def plot_by_modality(df: pd.DataFrame, plot_dir: Path) -> Path:
    order = (
        df.groupby("modality")["test_cidx"]
        .median()
        .sort_values(ascending=False)
        .index
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="modality", y="test_cidx", order=order, ax=ax, linewidth=1)
    ax.set(title="Test c-index across modalities (ordered by median)", xlabel="Modality", ylabel="Test c-index")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return _save_fig(fig, plot_dir, "modality_summary")


def plot_by_cancer(df: pd.DataFrame, plot_dir: Path) -> Path:
    order = (
        df.groupby("cancer")["test_cidx"]
        .median()
        .sort_values(ascending=False)
        .index
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=df, x="cancer", y="test_cidx", order=order, ax=ax, linewidth=1)

    # annotate modality with highest c-index for each cancer
    y_offset = 0.01
    for xpos, cancer in enumerate(order):
        cancer_slice = df.loc[df["cancer"] == cancer]
        idx_best = cancer_slice["test_cidx"].idxmax()
        best_val = cancer_slice.loc[idx_best, "test_cidx"]
        best_modal = cancer_slice.loc[idx_best, "modality"]
        ax.text(xpos, best_val + y_offset, best_modal, ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set(title="Test c-index across cancer types (ordered by median)", xlabel="Cancer type", ylabel="Test c-index")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    return _save_fig(fig, plot_dir, "cancer_summary")


def plot_winner_counts(df: pd.DataFrame, plot_dir: Path) -> Path:
    best_rows = df.loc[df.groupby("cancer")["test_cidx"].idxmax()]
    best_count = best_rows["modality"].value_counts().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=best_count.index, y=best_count.values, ax=ax)
    ax.set(
        title="Modality that achieves the highest c-index (count of cancers)",
        xlabel="Modality",
        ylabel="Number of cancers",
    )
    ax.bar_label(ax.containers[0], padding=3)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return _save_fig(fig, plot_dir, "highest_cindex")


def plot_rank_summaries(df: pd.DataFrame, plot_dir: Path) -> List[Path]:
    outputs: List[Path] = []

    # per-cancer ranks (1 = best)
    df = df.copy()
    df["rank"] = df.groupby("cancer")["test_cidx"].rank(method="min", ascending=False)

    rank_sum = df.groupby("modality")["rank"].sum().sort_values()
    fig_sum, ax_sum = plt.subplots(figsize=(8, 4))
    sns.barplot(x=rank_sum.index, y=rank_sum.values, ax=ax_sum, order=rank_sum.index)
    ax_sum.set(
        title="Sum of modality ranks across cancers (lower = better)",
        xlabel="Modality",
        ylabel="Sum of per-cancer ranks",
    )
    ax_sum.bar_label(ax_sum.containers[0], padding=3)
    ax_sum.tick_params(axis="x", rotation=45)
    fig_sum.tight_layout()
    outputs.append(_save_fig(fig_sum, plot_dir, "summed_ranks"))

    median_rank = df.groupby("modality")["rank"].median().sort_values()
    order = median_rank.index

    fig_rank, ax_rank = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="modality", y="rank", order=order, linewidth=1, ax=ax_rank)
    ax_rank.axhline(1, ls="--", c="grey", lw=0.8)
    ax_rank.set(
        title="Distribution of modality ranks across cancers (sorted by median rank)",
        xlabel="Modality",
        ylabel="Per-cancer rank (1 = best c-index)",
    )
    ax_rank.tick_params(axis="x", rotation=45)
    fig_rank.tight_layout()
    outputs.append(_save_fig(fig_rank, plot_dir, "median_ranks"))

    return outputs


# ---------------------------------------------------------------------------
# CLI & entry point
# ---------------------------------------------------------------------------


def _find_latest_run(root: Path) -> Path:
    runs = [p for p in root.iterdir() if p.is_dir()]
    if not runs:
        raise RuntimeError(f"No runs found under {root}")
    return max(runs, key=lambda p: p.stat().st_mtime)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise mlp_cox_job results")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("/projectnb2/evolution/zwakefield/tcga/cancer_learning/single_input"),
        help="Root directory that contains timestamped mlp_cox_job runs",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Specific run directory to analyse; if omitted, the latest run under --runs-root is used",
    )
    parser.add_argument(
        "--plots-subdir",
        type=str,
        default="plots",
        help="Subdirectory (relative to the run dir) where plots are written",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir if args.run_dir else _find_latest_run(args.runs_root)
    if not run_dir.is_dir():
        raise RuntimeError(f"Run directory does not exist: {run_dir}")

    models_dir = run_dir / "models"
    if not models_dir.is_dir():
        raise RuntimeError(f"Expected models directory under run dir: {models_dir}")

    plot_dir = run_dir / args.plots_subdir

    df = _load_metrics(models_dir)
    print(
        f"Loaded {len(df):,} result files "
        f"({df['cancer'].nunique()} cancers, {df['modality'].nunique()} modalities)"
    )
    print(f"Cancers included: {sorted(df['cancer'].unique())}")

    _set_style()
    plot_by_modality(df, plot_dir)
    plot_by_cancer(df, plot_dir)
    plot_winner_counts(df, plot_dir)
    plot_rank_summaries(df, plot_dir)


if __name__ == "__main__":
    main()
