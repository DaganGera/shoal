"""Module [4] — DETECTION, plus the fallback chain (plan §5, §16).

The detector has ONE job: localise fish with high recall. Species identification is
a separate, later, track-level step (plan §2). We keep the detector single-job in
two complementary ways:

  * ``classes=fish_class_ids`` is passed to every predict/track call, so NMS itself
    only ever emits fish-family detections (and, importantly, the ReID feature rows
    stay aligned with the boxes);
  * a post-process callback rewrites the surviving class ids to 0 and renames the
    class map to ``{0: "fish"}``, so the tracker's association, the annotated video,
    and every downstream table see exactly one class.

Fallback chain, tried in order, with the landing logged to the run manifest:

    1. weights/shoal_fish.pt      fine-tuned single-class YOLO26  (Phase 2 output; slot is real now)
    2. weights/FishInv.pt         marine-detect YOLOv8, fish families collapsed  (Phase 1 primary)
    3. yolov8s-worldv2.pt         YOLO-World open-vocab, prompted with "fish"  (zero-training emergency)

Every rung is a real, exercised code path — not a note in a README.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from shoal.config import REPO_ROOT

WEIGHTS_DIR = REPO_ROOT / "weights"

# Open-vocab prompts for the YOLO-World emergency rung.
_WORLD_PROMPTS = ["fish", "a fish", "school of fish"]


@dataclass
class Detector:
    """A loaded detector + the metadata needed to run it single-class."""

    name: str                       # human label for logs / dashboard
    kind: str                       # 'finetuned' | 'marine-detect' | 'yolo-world'
    source: str                     # file path or pretrained id actually loaded
    model: Any                      # ultralytics YOLO / YOLOWorld
    fish_class_ids: list[int]       # indices in model.names treated as fish
    class_names: dict[int, str]     # the underlying model's full name map (for the dashboard)
    device: str
    conf: float
    iou: float
    imgsz: int
    max_det: int
    _hook_installed: bool = field(default=False, repr=False)

    # -- single-class collapse ----------------------------------------------
    def install_single_class_hook(self, log: logging.Logger | None = None) -> None:
        """Add an Ultralytics callback that, after NMS, rewrites every (already
        fish-filtered) class id to 0 and presents a ``{0: 'fish'}`` name map.

        No rows are removed here — the ``classes=`` filter on the predict/track call
        already did that, which keeps the ReID feature rows aligned with the boxes.
        """
        if self._hook_installed:
            return

        def _collapse(predictor) -> None:
            for res in predictor.results:
                boxes = res.boxes
                if boxes is not None and boxes.data.numel():
                    data = boxes.data.clone()
                    data[:, -1] = 0.0
                    res.boxes = boxes.__class__(data, boxes.orig_shape)
                res.names = {0: "fish"}
            predictor.model.names = {0: "fish"}

        self.model.add_callback("on_predict_postprocess_end", _collapse)
        self._hook_installed = True
        if log:
            log.info("single-class hook installed on %s (classes=%s -> 0)",
                     self.name, self.fish_class_ids)

    # -- per-frame prediction (used by ablation + calibration, no tracking) --
    def predict_frame(self, bgr: np.ndarray) -> np.ndarray:
        """Return an (N, 5) array [x1, y1, x2, y2, conf] of fish detections only."""
        res = self.model.predict(
            bgr, conf=self.conf, iou=self.iou, imgsz=self.imgsz, max_det=self.max_det,
            classes=self.fish_class_ids, device=self.device, verbose=False,
        )[0]
        if res.boxes is None or len(res.boxes) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        return np.column_stack([xyxy, conf]).astype(np.float32)


# --------------------------------------------------------------------------- #
# fish-class discovery
# --------------------------------------------------------------------------- #
def _fish_class_ids(names: dict[int, str], aliases: list[str]) -> list[int]:
    alias_l = {a.strip().lower() for a in aliases}
    ids = [i for i, n in names.items() if str(n).strip().lower() in alias_l]
    ids += [i for i, n in names.items() if "fish" in str(n).lower() and i not in ids]
    return sorted(set(ids))


# --------------------------------------------------------------------------- #
# the chain
# --------------------------------------------------------------------------- #
def _load_ultralytics(path_or_id: str):
    from ultralytics import YOLO

    return YOLO(path_or_id)


def _try_finetuned(cfg, device, log) -> Detector | None:
    path = Path(cfg.get("finetuned_weights") or (WEIGHTS_DIR / "shoal_fish.pt"))
    if not path.exists():
        log.info("fallback chain [1/3]: fine-tuned weights absent (%s) — skipping", path.name)
        return None
    model = _load_ultralytics(str(path))
    ids = _fish_class_ids(model.names, cfg["fish_class_aliases"]) or list(model.names)
    log.info("fallback chain [1/3]: loaded fine-tuned detector %s", path.name)
    return _mk(model, "SHOAL fine-tuned YOLO26 (single-class)", "finetuned", str(path), ids, cfg, device)


def _try_marine_detect(cfg, device, log) -> Detector | None:
    path = Path(cfg.get("marine_detect_weights") or (WEIGHTS_DIR / "FishInv.pt"))
    if not path.exists():
        log.info("fallback chain [2/3]: FishInv.pt absent (%s) — run `shoal-fetch` — skipping", path)
        return None
    model = _load_ultralytics(str(path))
    ids = _fish_class_ids(model.names, cfg["fish_class_aliases"])
    if not ids:
        log.warning("fallback chain [2/3]: FishInv.pt exposes no fish-like class in %s", model.names)
        return None
    log.info("fallback chain [2/3]: loaded marine-detect FishInv.pt; fish class ids %s of %d classes",
             ids, len(model.names))
    return _mk(model, "marine-detect FishInv (YOLOv8, fish families collapsed)",
               "marine-detect", str(path), ids, cfg, device)


def _try_yolo_world(cfg, device, log) -> Detector | None:
    try:
        from ultralytics import YOLOWorld

        weights = cfg.get("world_weights") or "yolov8s-worldv2.pt"
        model = YOLOWorld(weights)
        model.set_classes(_WORLD_PROMPTS)
        log.info("fallback chain [3/3]: loaded YOLO-World (%s), prompts=%s", weights, _WORLD_PROMPTS)
        names = {i: p for i, p in enumerate(_WORLD_PROMPTS)}
        return _mk(model, "YOLO-World open-vocab ('fish')", "yolo-world",
                   weights, list(names), cfg, device, names=names)
    except Exception as exc:
        log.error("fallback chain [3/3]: YOLO-World failed: %s", exc)
        return None


def _mk(model, name, kind, source, ids, cfg, device, names=None) -> Detector:
    return Detector(
        name=name, kind=kind, source=source, model=model, fish_class_ids=list(ids),
        class_names=dict(names or model.names), device=device,
        conf=float(cfg.get("conf", 0.20)), iou=float(cfg.get("iou", 0.6)),
        imgsz=int(cfg.get("imgsz", 640)), max_det=int(cfg.get("max_det", 300)),
    )


_CHAIN: list[Callable[..., Detector | None]] = [_try_finetuned, _try_marine_detect, _try_yolo_world]


def build_detector(cfg: dict, device: str, log: logging.Logger) -> Detector:
    """Walk the fallback chain; return the first detector that loads, or raise."""
    for rung in _CHAIN:
        det = rung(cfg, device, log)
        if det is not None:
            try:
                det.model.to(device)
            except Exception as exc:
                log.warning("could not move %s to %s (%s); staying on cpu", det.name, device, exc)
                det.device = "cpu"
            log.info("DETECTOR SELECTED: %s  (source=%s, device=%s, conf=%.2f)",
                     det.name, det.source, det.device, det.conf)
            return det
    raise RuntimeError(
        "no detector could be loaded. Run `uv run shoal-fetch` to download FishInv.pt, "
        "or check network access for the YOLO-World fallback."
    )
