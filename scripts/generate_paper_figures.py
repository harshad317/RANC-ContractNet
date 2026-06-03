"""Generate paper-facing figures from existing aggregate outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path("paper/neurips2027/figures")


def _read_csv(relative: str) -> pd.DataFrame:
    path = REPO_ROOT / relative
    if not path.exists():
        raise FileNotFoundError(f"Missing required figure input: {relative}")
    return pd.read_csv(path)


def _draw_pipeline_panel(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.set_title("A. compilation path", loc="left", fontsize=8, fontweight="bold", pad=2)
    labels = [
        "contract",
        "cards",
        "ledger",
        "compiler",
        "tests",
        "audit",
    ]
    positions = [
        (0.07, 0.62),
        (0.37, 0.62),
        (0.67, 0.62),
        (0.07, 0.25),
        (0.37, 0.25),
        (0.67, 0.25),
    ]
    width = 0.20
    height = 0.24
    for idx, ((x, y), label) in enumerate(zip(positions, labels)):
        face = "#e8f1f2" if idx % 2 == 0 else "#f5efe6"
        box = FancyBboxPatch(
            (x, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.018,rounding_size=0.018",
            linewidth=0.8,
            edgecolor="#263238",
            facecolor=face,
            transform=ax.transAxes,
        )
        ax.add_patch(box)
        ax.text(
            x + width / 2,
            y,
            label,
            ha="center",
            va="center",
            fontsize=6.8,
            color="#1b1f23",
            transform=ax.transAxes,
        )
    arrows = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    for src, dst in arrows:
        sx, sy = positions[src]
        dx, dy = positions[dst]
        start = (sx + width + 0.015, sy) if sy == dy else (sx + width / 2, sy - height / 2)
        end = (dx - 0.015, dy) if sy == dy else (dx + width / 2, dy + height / 2)
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            xycoords=ax.transAxes,
            arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#455a64"},
        )


def _draw_outlier_panel(ax: plt.Axes, deltas: pd.DataFrame) -> None:
    rows = deltas[
        deltas["metric_label"].isin(["Corruption AUROC", "Rare recall"])
    ].copy()
    rows["label"] = rows["scenario"].map(
        {"noise": "noise: corruption AUROC", "signal": "signal: rare recall"}
    )
    rows = rows.sort_values("scenario", ascending=True)

    y = np.arange(len(rows))
    means = rows["delta_mean"].to_numpy(float)
    low = rows["ci95_low"].to_numpy(float)
    high = rows["ci95_high"].to_numpy(float)
    xerr = np.vstack([means - low, high - means])

    ax.barh(y, means, color=["#2a9d8f", "#b56576"], height=0.52)
    ax.errorbar(means, y, xerr=xerr, fmt="none", ecolor="#1b1f23", elinewidth=0.8, capsize=2)
    ax.axvline(0, color="#5f6368", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(rows["label"], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("correct - wrong contract", fontsize=6.5)
    ax.set_title("B. paired semantic deltas", loc="left", fontsize=8, fontweight="bold", pad=2)
    ax.tick_params(axis="x", labelsize=6.3)
    ax.grid(axis="x", color="#d0d7de", linewidth=0.5, alpha=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)


def _draw_openml_panel(ax: plt.Axes, summary: pd.DataFrame) -> None:
    order = ["standard", "robust", "selector"]
    rows = summary.set_index("baseline").loc[order].reset_index()
    y = np.arange(len(rows))
    means = rows["mean_delta"].to_numpy(float)
    low = rows["ci95_low"].to_numpy(float)
    high = rows["ci95_high"].to_numpy(float)
    xerr = np.vstack([means - low, high - means])
    colors = ["#457b9d", "#6d597a", "#d07a2d"]

    ax.barh(y, means, color=colors, height=0.52)
    ax.errorbar(means, y, xerr=xerr, fmt="none", ecolor="#1b1f23", elinewidth=0.8, capsize=2)
    ax.axvline(0, color="#5f6368", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(rows["baseline"], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("RANC - baseline, signed", fontsize=6.5)
    ax.set_title("C. public OpenML/UCI deltas", loc="left", fontsize=8, fontweight="bold", pad=2)
    ax.tick_params(axis="x", labelsize=6.3)
    ax.grid(axis="x", color="#d0d7de", linewidth=0.5, alpha=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)


def generate_visual_summary(output_dir: Path) -> tuple[Path, Path]:
    deltas = _read_csv("outputs/outlier_pair/contract_delta_stats.csv")
    openml = _read_csv("outputs/openml/openml_win_loss_summary.csv")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.0, 1.75),
        gridspec_kw={"width_ratios": [1.75, 1.0, 1.05], "wspace": 0.55},
    )
    _draw_pipeline_panel(axes[0])
    _draw_outlier_panel(axes[1], deltas)
    _draw_openml_panel(axes[2], openml)
    pdf_path = output_dir / "ranc_visual_summary.pdf"
    png_path = output_dir / "ranc_visual_summary.png"
    metadata = {
        "Creator": "RANC-ContractNet figure generator",
        "Producer": "matplotlib",
        "Title": "RANC-ContractNet visual summary",
    }
    fig.savefig(pdf_path, bbox_inches="tight", metadata=metadata)
    fig.savefig(png_path, dpi=220, bbox_inches="tight", metadata={"Software": "matplotlib"})
    plt.close(fig)
    return pdf_path, png_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    pdf_path, png_path = generate_visual_summary(output_dir)
    print(pdf_path.relative_to(REPO_ROOT))
    print(png_path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
