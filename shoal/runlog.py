"""Run context: output directory, logging, and provenance capture.

Plan §16 requires that every run logs "which model, which tracker config, and which
code path was actually taken". `RunContext.provenance` is the structured home for
that; it is written to ``outputs/<run_id>/run.json`` at the end of every run and is
read back verbatim by the dashboard's method summary and the Phase-2 PDF report. A
scientific tool that does not record its own configuration is not a scientific tool.
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from shoal.config import REPO_ROOT

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)-18s %(message)s"


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _package_versions() -> dict[str, str]:
    """Record the versions of every library whose output the plan puts on a slide."""
    names = [
        "numpy", "pandas", "scipy", "opencv-python-headless", "opencv-python",
        "ultralytics", "torch", "torchvision", "shapely", "geopandas",
        "movingpandas", "scikit-learn", "hmmlearn", "motmetrics",
    ]
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str] = {}
    for name in names:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            continue
    return out


@dataclass
class RunContext:
    """One pipeline invocation: where artefacts go, and what was actually done."""

    run_id: str
    out_dir: Path
    clip_path: Path
    device: str
    config: dict[str, Any]
    log: logging.Logger
    started: float = field(default_factory=time.time)
    provenance: dict[str, Any] = field(default_factory=dict)

    # -- artefact path helpers -------------------------------------------------
    def path(self, *parts: str) -> Path:
        p = self.out_dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def note(self, key: str, value: Any) -> None:
        """Record a provenance fact (model landed on, code path taken, ...)."""
        self.provenance[key] = value
        self.log.info("provenance: %s = %s", key, _short(value))

    # -- lifecycle ----------------------------------------------------------
    def write_manifest(self, extra: dict[str, Any] | None = None) -> Path:
        manifest = {
            "run_id": self.run_id,
            "clip": str(self.clip_path),
            "device": self.device,
            "started_utc": datetime.fromtimestamp(self.started, UTC).isoformat(),
            "finished_utc": datetime.now(UTC).isoformat(),
            "wall_seconds": round(time.time() - self.started, 2),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "git_commit": _git_commit(),
            "package_versions": _package_versions(),
            "config": self.config,
            "provenance": self.provenance,
        }
        if extra:
            manifest.update(extra)
        out = self.path("run.json")
        out.write_text(json.dumps(manifest, indent=2, default=str))
        self.log.info("wrote run manifest -> %s", out)
        return out


def _short(value: Any, limit: int = 120) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def new_run(
    clip_path: str | Path,
    device: str,
    config: dict[str, Any],
    outputs_root: str | Path | None = None,
    run_id: str | None = None,
) -> RunContext:
    """Create ``outputs/<run_id>/`` and a logger that tees to ``run.log``."""
    clip_path = Path(clip_path)
    outputs_root = Path(outputs_root) if outputs_root else REPO_ROOT / "outputs"
    if run_id is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_id = f"{stamp}_{clip_path.stem}"
    out_dir = outputs_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"shoal.run.{run_id}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(_LOG_FORMAT)
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    file_handler = logging.FileHandler(out_dir / "run.log")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    ctx = RunContext(
        run_id=run_id,
        out_dir=out_dir,
        clip_path=clip_path,
        device=device,
        config=config,
        log=logger,
    )
    logger.info("run %s  clip=%s  device=%s", run_id, clip_path, device)
    return ctx


def basic_logging(level: int = logging.INFO) -> None:
    """Console logging for the non-pipeline entry points (check, fetch, calibrate)."""
    logging.basicConfig(level=level, format=_LOG_FORMAT, stream=sys.stderr)
