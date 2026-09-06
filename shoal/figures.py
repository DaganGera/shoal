"""Static figure rendering for the auto-report and the run directory.

The Streamlit dashboard builds its own interactive Plotly versions; these are the
headless PNGs that go in ``outputs/<run_id>/`` and (Phase 2) the PDF report. Kept
separate so neither matplotlib nor Streamlit is a hard import of the other path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def residency_png(residency: dict, background_frame: np.ndarray | None, out_path: str | Path,
                  title: str = "Residency (time-weighted occupancy)") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hist = np.asarray(residency["histogram"], dtype=float)
    extent = residency["extent"]
    fig, ax = plt.subplots(figsize=(8, 8 * hist.shape[0] / max(hist.shape[1], 1)))
    if background_frame is not None:
        ax.imshow(background_frame[..., ::-1], extent=extent, alpha=0.55)
    floor = np.percentile(hist[hist > 0], 8) if np.any(hist > 0) else 0
    masked = np.ma.masked_where(hist <= floor, hist)
    im = ax.imshow(masked, extent=extent, cmap="inferno", alpha=0.8,
                   interpolation="gaussian")
    ax.set_title(title)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(residency.get("units", "fish-seconds per bin"))
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def visibility_png(visibility_df, gate_threshold: float, out_path: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(visibility_df["timestamp"], visibility_df["visibility_score"], lw=1.2)
    ax.axhline(gate_threshold, ls="--", c="crimson", lw=1,
               label=f"restoration gate = {gate_threshold:.2f}")
    below = visibility_df[visibility_df["visibility_score"] < gate_threshold]
    ax.scatter(below["timestamp"], below["visibility_score"], s=8, c="crimson",
               label=f"{len(below)} frames restored")
    ax.set_ylim(0, 1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("visibility score (0-1)")
    ax.set_title("Per-frame visibility index")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def ablation_png(ablation: dict, out_path: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3 if ablation.get("gt_available") else 2, figsize=(11, 3.2))
    axes = np.atleast_1d(axes)

    axes[0].bar(["raw", "restored"], [ablation["raw_detections"], ablation["restored_detections"]],
                color=["#8a8a8a", "#2a7fff"])
    axes[0].set_title("fish detections (sampled frames)")

    axes[1].bar(["raw", "restored"], [ablation["raw_mean_conf"], ablation["restored_mean_conf"]],
                color=["#8a8a8a", "#2a7fff"])
    axes[1].set_title("mean detection confidence")
    axes[1].set_ylim(0, 1)

    if ablation.get("gt_available"):
        axes[2].bar(["raw", "restored"], [ablation["map50_raw"], ablation["map50_restored"]],
                    color=["#8a8a8a", "#2a7fff"])
        axes[2].set_title("mAP@0.5 vs ground truth")
        axes[2].set_ylim(0, 1)

    fig.suptitle(ablation.get("verdict", ""), fontsize=8, y=1.02)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path
