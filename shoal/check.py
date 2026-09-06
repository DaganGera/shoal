"""Preflight (plan §13, 0:00-0:20 acceptance test: "prints all green").

Verifies the environment can run the pipeline end to end BEFORE a demo:
dependencies, GPU, ffmpeg, config files, weights, clips, calibration status, output
writability, and a real 2-frame detect+track smoke test on the actual detector.

Exit code 0 iff nothing is RED. YELLOW items are runnable-but-degraded (e.g. no
calibration yet, CPU fallback).
"""

from __future__ import annotations

import importlib
import shutil
import sys
import traceback

import numpy as np

from shoal.config import REPO_ROOT, load_config, resolve_device

GREEN, YELLOW, RED = "\033[92m", "\033[93m", "\033[91m",
RESET, BOLD = "\033[0m", "\033[1m"

_results: list[tuple[str, str]] = []  # (level, message)


def ok(msg: str) -> None:
    _results.append(("green", msg))
    print(f"  {GREEN}✓{RESET} {msg}")


def warn(msg: str) -> None:
    _results.append(("yellow", msg))
    print(f"  {YELLOW}!{RESET} {msg}")


def bad(msg: str) -> None:
    _results.append(("red", msg))
    print(f"  {RED}✗{RESET} {msg}")


def section(title: str) -> None:
    print(f"\n{BOLD}{title}{RESET}")


# --------------------------------------------------------------------------- #
def check_python() -> None:
    section("python")
    v = sys.version_info
    (ok if (3, 11) <= (v.major, v.minor) < (3, 13) else warn)(
        f"python {v.major}.{v.minor}.{v.micro}"
    )


def check_deps() -> None:
    section("dependencies")
    core = {
        "numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "cv2": "opencv",
        "yaml": "pyyaml", "ultralytics": "ultralytics", "torch": "torch",
        "torchvision": "torchvision", "shapely": "shapely", "geopandas": "geopandas",
        "movingpandas": "movingpandas", "sklearn": "scikit-learn", "lap": "lap",
        "streamlit": "streamlit", "plotly": "plotly", "matplotlib": "matplotlib",
    }
    for mod, label in core.items():
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            ok(f"{label} {ver}")
        except Exception as exc:
            bad(f"{label}: {exc}")

    section("dependencies (Phase 2)")
    for mod, label in {"hmmlearn": "hmmlearn", "motmetrics": "motmetrics",
                       "pydeck": "pydeck", "reportlab": "reportlab"}.items():
        try:
            m = importlib.import_module(mod)
            ok(f"{label} {getattr(m, '__version__', '?')}")
        except Exception:
            warn(f"{label} not installed (Phase 2 feature) — `uv sync --group phase2`")


def check_gpu() -> str:
    section("compute")
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            ok(f"CUDA available: {name}  ({vram:.1f} GB VRAM)")
            if vram < 5.5:
                warn(f"VRAM {vram:.1f} GB is tight for imgsz=640 batch>1; pipeline runs batch=1")
            return "cuda"
        warn("no CUDA — pipeline will run on CPU (slower, still correct)")
        return "cpu"
    except Exception as exc:
        bad(f"torch check failed: {exc}")
        return "cpu"


def check_ffmpeg() -> None:
    section("ffmpeg")
    for tool in ("ffmpeg", "ffprobe"):
        (ok if shutil.which(tool) else bad)(
            f"{tool} {'found' if shutil.which(tool) else 'MISSING (needed for video IO / turbid fixture)'}"
        )


def check_configs() -> None:
    section("config")
    try:
        cfg = load_config()
        ok(f"pipeline.yaml parsed ({len(cfg.as_dict())} sections)")
        for key in ("detect.conf", "restore.gate_threshold", "repair.savgol_window",
                    "ecology.species_conf_floor"):
            v = cfg.get(key)
            (ok if v is not None else bad)(f"config[{key}] = {v}")
    except Exception as exc:
        bad(f"pipeline.yaml: {exc}")

    tracker = REPO_ROOT / "config" / "botsort_underwater.yaml"
    (ok if tracker.exists() else bad)(f"tracker config {'present' if tracker.exists() else 'MISSING'}")

    calib_path = REPO_ROOT / "config" / "visibility_calibration.json"
    try:
        import json

        c = json.loads(calib_path.read_text())
        if c.get("calibrated"):
            s = c.get("_sanity", {})
            ok(f"visibility calibration: calibrated ({c.get('method', '?')[:40]}...); "
               f"anchors ordered={s.get('ordered_correctly')}")
        else:
            warn("visibility calibration: PROVISIONAL — run `uv run shoal-calibrate` before the demo")
    except Exception as exc:
        bad(f"visibility_calibration.json: {exc}")


def check_weights() -> None:
    section("weights")
    fishinv = REPO_ROOT / "weights" / "FishInv.pt"
    if fishinv.exists() and fishinv.stat().st_size > 5e6:
        ok(f"FishInv.pt present ({fishinv.stat().st_size/1e6:.0f} MB)")
    else:
        warn("FishInv.pt absent — `uv run shoal-fetch`. "
             "Pipeline will fall back to YOLO-World (needs network on first use).")
    ft = REPO_ROOT / "weights" / "shoal_fish.pt"
    (ok if ft.exists() else warn)(
        "fine-tuned shoal_fish.pt present" if ft.exists()
        else "fine-tuned shoal_fish.pt absent (Phase 2 output — expected in Phase 1)"
    )


def check_clips() -> None:
    section("data / clips")
    clips_dir = REPO_ROOT / "data" / "clips"
    mp4s = sorted(clips_dir.glob("*.mp4")) if clips_dir.exists() else []
    if not mp4s:
        bad("no clips in data/clips — `uv run shoal-fetch`")
        return
    import subprocess

    for clip in mp4s:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=width,height,nb_frames,r_frame_rate",
                            "-of", "csv=p=0", str(clip)], capture_output=True, text=True)
        (ok if r.returncode == 0 else bad)(f"{clip.name}: {r.stdout.strip() or r.stderr.strip()[:60]}")
    turbid = clips_dir / "reef_turbid_matched.mp4"
    (ok if turbid.exists() else warn)(
        "turbid matched-pair fixture present" if turbid.exists()
        else "turbid fixture absent — `uv run shoal-fetch` (needed for matched-pair ablation)"
    )


def check_outputs_writable() -> None:
    section("outputs")
    out = REPO_ROOT / "outputs"
    try:
        out.mkdir(exist_ok=True)
        probe = out / ".write_test"
        probe.write_text("ok")
        probe.unlink()
        ok(f"{out} writable")
    except Exception as exc:
        bad(f"cannot write to {out}: {exc}")


def smoke_detect_track(device: str) -> None:
    section("smoke test: detect + track (2 frames)")
    try:
        import logging

        log = logging.getLogger("shoal.check.smoke")
        log.addHandler(logging.NullHandler())
        from shoal.detect import build_detector

        cfg = load_config()
        det = build_detector(cfg["detect"], resolve_device(device), log)
        ok(f"detector loaded: {det.name} on {det.device}")

        # two synthetic frames with a moving bright blob (won't necessarily detect a
        # fish, but must run the full inference + tracking path without error)
        frame = np.full((360, 640, 3), 40, np.uint8)
        cv2 = importlib.import_module("cv2")
        for i in range(2):
            f = frame.copy()
            cv2.circle(f, (200 + i * 40, 180), 30, (180, 200, 210), -1)
            det.model.track(f, persist=True,
                            tracker=str(REPO_ROOT / "config" / "botsort_underwater.yaml"),
                            classes=det.fish_class_ids, device=det.device, verbose=False)
        ok("detect + BoT-SORT track ran end to end on 2 frames")

        boxes = det.predict_frame(frame)
        ok(f"predict_frame path OK (returned {len(boxes)} boxes on a blank frame)")
    except Exception as exc:
        bad(f"smoke test failed: {exc}")
        traceback.print_exc()


def main(argv: list[str] | None = None) -> int:
    print(f"{BOLD}SHOAL preflight{RESET}  ({REPO_ROOT})")
    check_python()
    check_deps()
    device = check_gpu()
    check_ffmpeg()
    check_configs()
    check_weights()
    check_clips()
    check_outputs_writable()
    smoke_detect_track(device)

    n_red = sum(1 for lvl, _ in _results if lvl == "red")
    n_yellow = sum(1 for lvl, _ in _results if lvl == "yellow")
    n_green = sum(1 for lvl, _ in _results if lvl == "green")
    print(f"\n{BOLD}summary{RESET}: {GREEN}{n_green} green{RESET}, "
          f"{YELLOW}{n_yellow} yellow{RESET}, {RED}{n_red} red{RESET}")
    if n_red:
        print(f"{RED}preflight FAILED — {n_red} blocking issue(s) above.{RESET}")
        return 1
    if n_yellow:
        print(f"{YELLOW}preflight OK with warnings — runnable, some features degraded.{RESET}")
        return 0
    print(f"{GREEN}{BOLD}all green.{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
