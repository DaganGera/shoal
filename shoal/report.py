"""Module [11b] — auto-generated one-page PDF report.  **Phase 2** (plan §11.3, §13 priority 9).

Design, for when this is built:

    One page (`reportlab`), generated from a completed `outputs/<run_id>/` directory:
      * clip metadata + the camera-motion verdict
      * visibility profile (the `visibility.png` figure)
      * abundance bounds (MaxN vs unique tracks)
      * diversity with CIs, or the honesty-guard notice
      * behavioural state breakdown
      * flagged events (Phase 2 event detection)
      * method summary: library versions + model checkpoints, straight from `run.json`

    Reproducibility metadata is included because a scientific tool that doesn't
    record its own configuration isn't a scientific tool.

Phase 1 status: NOT IMPLEMENTED. Everything the report needs is already produced by
`shoal.pipeline` — `run.json` (full provenance), `results.json` (every headline
number), and the PNG figures. Dashboard tab 7 shows that provenance today.
"""

from __future__ import annotations

from pathlib import Path


def build_report(run_dir: str | Path, out_pdf: str | Path | None = None) -> Path:
    raise NotImplementedError(
        "PDF report is Phase 2 (plan §11.3 / §13 priority 9). The inputs already "
        "exist in the run directory: run.json, results.json, visibility.png, "
        "residency.png, ablation.png. Dashboard tab 7 renders the provenance now."
    )
