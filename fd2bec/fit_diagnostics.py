"""Diagnostics and visual reports for grouped least-squares residuals."""

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

MODIFIED_Z_SCORE_THRESHOLD = 3.5


@dataclass(frozen=True)
class FrameResidualDiagnostics:
    """Per-frame residual statistics for a grouped linear-system fit."""

    residual_norms: np.ndarray
    component_spreads: np.ndarray
    squared_residual_contributions: np.ndarray
    modified_z_scores: np.ndarray
    outliers: np.ndarray
    median_residual_norm: float
    median_absolute_deviation: float
    total_squared_residual: float


def frame_residual_diagnostics(
    residual_vector: np.ndarray,
    group_size: int,
    outlier_threshold: float = MODIFIED_Z_SCORE_THRESHOLD,
) -> FrameResidualDiagnostics:
    """Compute per-frame residual diagnostics and robust high-residual outliers.

    Outliers are frames whose residual norm has a modified z-score greater
    than ``outlier_threshold``.  The default threshold of 3.5 is the common
    robust criterion based on the median absolute deviation (MAD).
    """
    residual_vector = np.asarray(residual_vector, dtype=float)
    if residual_vector.ndim == 2:
        if residual_vector.shape[1] != 1:
            raise ValueError("Frame residual diagnostics require one right-hand side.")
        residual_vector = residual_vector[:, 0]
    if residual_vector.ndim != 1:
        raise ValueError("Residuals must be a one-dimensional vector.")
    if group_size <= 0:
        raise ValueError("The group size must be positive.")
    if len(residual_vector) % group_size:
        raise ValueError(
            f"Cannot split {len(residual_vector)} residuals into groups of {group_size}."
        )
    if outlier_threshold <= 0:
        raise ValueError("The outlier threshold must be positive.")

    residuals = residual_vector.reshape((-1, group_size))
    residual_norms = np.linalg.norm(residuals, axis=1)
    component_spreads = np.std(residuals, axis=1)
    squared_residuals = residual_norms**2
    total_squared_residual = float(np.sum(squared_residuals))
    contributions = np.divide(
        squared_residuals,
        total_squared_residual,
        out=np.full_like(squared_residuals, np.nan, dtype=float),
        where=total_squared_residual > 0,
    )

    median = float(np.median(residual_norms))
    deviations = np.abs(residual_norms - median)
    mad = float(np.median(deviations))
    if mad > np.finfo(float).eps:
        modified_z_scores = 0.67448975 * (residual_norms - median) / mad
    else:
        modified_z_scores = np.where(residual_norms > median + np.finfo(float).eps, np.inf, 0.0)
    outliers = modified_z_scores > outlier_threshold

    return FrameResidualDiagnostics(
        residual_norms=residual_norms,
        component_spreads=component_spreads,
        squared_residual_contributions=contributions,
        modified_z_scores=modified_z_scores,
        outliers=outliers,
        median_residual_norm=median,
        median_absolute_deviation=mad,
        total_squared_residual=total_squared_residual,
    )


def retained_equation_mask(outliers: np.ndarray, group_size: int) -> np.ndarray:
    """Return the equation mask that removes every component of outlier frames."""
    outliers = np.asarray(outliers, dtype=bool)
    if outliers.ndim != 1:
        raise ValueError("Outlier flags must be a one-dimensional array.")
    if group_size <= 0:
        raise ValueError("The group size must be positive.")
    return np.repeat(~outliers, group_size)


def _format_float(value: float) -> str:
    """Format a finite diagnostic value compactly for a Markdown report."""
    return f"{value:.6e}" if np.isfinite(value) else "n/a"


def _plot_residual_report(
    path: Path,
    diagnostics: FrameResidualDiagnostics,
    directions: np.ndarray,
    frame_indices: np.ndarray,
    n_equations: int,
    n_unknowns: int,
    rank: int,
    condition_number: float,
    relative_residual: float,
) -> None:
    """Create a two-panel PNG residual report without requiring a display."""
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fd2bec-matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = frame_indices
    colors = np.where(diagnostics.outliers, "tab:red", "tab:blue")
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    figure.suptitle("Dipole/polarization fit residuals", fontsize=16)

    axes[0].errorbar(
        frames,
        diagnostics.residual_norms,
        yerr=diagnostics.component_spreads,
        fmt="none",
        ecolor="0.45",
        capsize=3,
        zorder=1,
        label="±1σ component spread",
    )
    axes[0].scatter(frames, diagnostics.residual_norms, c=colors, zorder=2, label="residual norm")
    axes[0].axhline(
        diagnostics.median_residual_norm,
        color="tab:green",
        linestyle="--",
        label="median residual norm",
    )
    for frame in frames[diagnostics.outliers]:
        axes[0].annotate(
            "*",
            (frame, diagnostics.residual_norms[frame]),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            color="tab:red",
            fontsize=15,
            fontweight="bold",
        )
    axes[0].set_ylabel("Residual norm\n(stored dipole units)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper left")

    contributions = 100.0 * diagnostics.squared_residual_contributions
    axes[1].bar(frames, contributions, color=colors)
    axes[1].set_ylabel("Squared-residual\ncontribution [%]")
    axes[1].set_xlabel("Dataset frame")
    axes[1].grid(axis="y", alpha=0.25)
    for frame in frames[diagnostics.outliers]:
        axes[1].annotate(
            "*",
            (frame, contributions[frame]),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            color="tab:red",
            fontsize=15,
            fontweight="bold",
        )

    outlier_frames = ", ".join(map(str, frames[diagnostics.outliers])) or "none"
    figure.subplots_adjust(bottom=0.15, top=0.93, hspace=0.06)
    figure.text(
        0.5,
        0.025,
        "\n".join(
            (
                f"equations/unknowns/rank: {n_equations}/{n_unknowns}/{rank}"
                f" | condition number: {_format_float(condition_number)}",
                f"relative residual: {_format_float(relative_residual)}"
                f" | outlier frames: {outlier_frames}",
            )
        ),
        ha="center",
        fontsize=9,
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_dipole_fit_residual_report(
    output: Path,
    diagnostics: FrameResidualDiagnostics,
    directions: np.ndarray,
    displacement_norms: np.ndarray,
    maximum_atomic_displacements: np.ndarray,
    n_equations: int,
    n_unknowns: int,
    rank: int,
    condition_number: float,
    relative_residual: float,
    stem: str = "dipole-fit",
    frame_indices: Optional[np.ndarray] = None,
) -> Tuple[Path, Path]:
    """Write PNG and Markdown reports for a dipole-fit residual analysis."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not stem or Path(stem).name != stem:
        raise ValueError("The report stem must be a simple filename.")
    directions = np.asarray(directions, dtype=object)
    displacement_norms = np.asarray(displacement_norms, dtype=float)
    maximum_atomic_displacements = np.asarray(maximum_atomic_displacements, dtype=float)
    expected_shape = diagnostics.residual_norms.shape
    if frame_indices is None:
        frame_indices = np.arange(len(diagnostics.residual_norms))
    frame_indices = np.asarray(frame_indices)
    if (
        directions.shape != expected_shape
        or displacement_norms.shape != expected_shape
        or maximum_atomic_displacements.shape != expected_shape
        or frame_indices.shape != expected_shape
    ):
        raise ValueError(
            "There must be one index, direction, and displacement magnitude per frame."
        )

    image = output / f"{stem}-residuals.png"
    markdown = output / f"{stem}-report.md"
    _plot_residual_report(
        image,
        diagnostics,
        directions,
        frame_indices,
        n_equations,
        n_unknowns,
        rank,
        condition_number,
        relative_residual,
    )

    outlier_frames = frame_indices[diagnostics.outliers]
    lines = [
        "# Dipole/polarization fit residual report",
        "",
        f"![Per-frame residual plot]({image.name})",
        "",
        "## Summary",
        "",
        f"- Equations / unknowns / rank: `{n_equations} / {n_unknowns} / {rank}`",
        f"- Condition number: `{_format_float(condition_number)}`",
        f"- Relative residual: `{_format_float(relative_residual)}`",
        f"- Total squared residual: `{_format_float(diagnostics.total_squared_residual)}`",
        f"- Median frame residual norm: `{_format_float(diagnostics.median_residual_norm)}`",
        f"- Median absolute deviation: `{_format_float(diagnostics.median_absolute_deviation)}`",
        f"- Outlier frames: `{', '.join(map(str, outlier_frames)) or 'none'}`",
        "",
        "An asterisk marks a high-residual outlier: modified z-score greater than 3.5, "
        "computed from frame residual norms and their median absolute deviation (MAD). "
        "The plotted error bars are the standard deviation of the three Cartesian residual "
        "components, not an uncertainty estimate from the electronic-structure calculation.",
        "",
        "## Per-frame residuals",
        "",
        "| Frame | Kind | Direction | Displacement norm | Max. atomic displacement | "
        "Residual norm | Contribution | Modified z-score | Flag |",
        "| ---: | :--- | :---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for frame, (
        norm,
        contribution,
        score,
        direction,
        displacement_norm,
        maximum_atomic_displacement,
        outlier,
    ) in zip(
        frame_indices,
        zip(
            diagnostics.residual_norms,
            diagnostics.squared_residual_contributions,
            diagnostics.modified_z_scores,
            directions,
            displacement_norms,
            maximum_atomic_displacements,
            diagnostics.outliers,
        ),
    ):
        kind = "reference" if frame == 0 else "displacement"
        contribution_text = (
            "n/a" if not np.isfinite(contribution) else f"{100.0 * contribution:.6f}%"
        )
        score_text = _format_float(score)
        flag = "*" if outlier else ""
        lines.append(
            f"| {frame} | {kind} | {direction} | {displacement_norm:.6e} | "
            f"{maximum_atomic_displacement:.6e} | {norm:.6e} | {contribution_text} | "
            f"{score_text} | {flag} |"
        )
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return image, markdown
