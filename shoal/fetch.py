"""Data + weight acquisition (build-plan §3).

Not one of the numbered pipeline modules — this is the "acquire before anything
else, cache to data/, never re-download" step from §3, made reproducible.

What it fetches for Phase 1:

  * weights/FishInv.pt  — marine-detect (Orange Business x Tenaka), a real YOLOv8
    fish + invertebrate detector. Used single-class as the Phase 1 detector and,
    in Phase 2, as the regional species head.
  * data/clips/*.mp4    — CC0 underwater clips (Mixkit) used as the demo spine and
    as the visibility-calibration anchors. These are STOCK PLACEHOLDERS. The plan's
    primary sources (Brackish, DeepFish) require Kaggle credentials; ``--brackish``
    / ``--deepfish`` pull them via kagglehub the moment a token is present.
  * data/clips/reef_turbid_matched.mp4 — a turbidity-augmented copy of the clear
    reef clip, clearly labelled. It exists to drive a MATCHED-PAIR restoration
    ablation (identical scene, only visibility changed) and to anchor the
    low-visibility end of the calibration until real turbid footage is available.

Everything is content-checked (size / ffprobe) and skipped if already present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import requests

from shoal.config import REPO_ROOT
from shoal.runlog import basic_logging

log = logging.getLogger("shoal.fetch")

WEIGHTS_DIR = REPO_ROOT / "weights"
CLIPS_DIR = REPO_ROOT / "data" / "clips"
RAW_DIR = REPO_ROOT / "data" / "raw"

# marine-detect model files (Azure blob, SAS token valid to 2099, no auth).
MARINE_DETECT = {
    "FishInv.pt": "https://stpubtenakanclyw.blob.core.windows.net/marine-detect/models2025/FishInv.pt",
    "MegaFauna.pt": "https://stpubtenakanclyw.blob.core.windows.net/marine-detect/models2025/MegaFauna.pt",
}


@dataclass
class ClipSpec:
    name: str            # destination filename in data/clips/
    url: str
    role: str            # what the pipeline / calibration uses it for
    note: str


# Mixkit CC0 clips (https://mixkit.co/license/). Direct CDN URLs, 720p.
DEMO_CLIPS = [
    ClipSpec("reef_clear.mp4",
             "https://assets.mixkit.co/videos/47402/47402-720.mp4",
             "primary demo + 'clear' calibration anchor",
             "tropical reef, several small fish in frame, clear blue water"),
    ClipSpec("anemone_static.mp4",
             "https://assets.mixkit.co/videos/8544/8544-720.mp4",
             "camera-motion-guard PASS example + clean two-fish tracking",
             "fixed camera on an anemone, two clownfish"),
    ClipSpec("reef_snapper.mp4",
             "https://assets.mixkit.co/videos/44973/44973-720.mp4",
             "single-subject trajectory / residency demo",
             "one snapper cruising over a reef, mild camera drift"),
]

TURBID_MATCHED = "reef_turbid_matched.mp4"
TURBID_SOURCE = "reef_clear.mp4"


# --------------------------------------------------------------------------- #
def _download(url: str, dest: Path, min_bytes: int, force: bool) -> bool:
    if dest.exists() and dest.stat().st_size >= min_bytes and not force:
        log.info("cached: %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("downloading %s -> %s", url, dest.name)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        got = 0
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                got += len(chunk)
                if total:
                    print(f"\r  {dest.name}: {got/1e6:6.1f} / {total/1e6:6.1f} MB", end="", file=sys.stderr)
    print(file=sys.stderr)
    if got < min_bytes:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{dest.name}: got {got} bytes, expected >= {min_bytes}")
    tmp.rename(dest)
    log.info("saved %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    return True


def _ffprobe_ok(path: Path) -> bool:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,nb_frames", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        return out.returncode == 0 and bool(out.stdout.strip())
    except Exception:
        return False


def fetch_weights(force: bool = False, all_models: bool = False) -> dict[str, str]:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    wanted = MARINE_DETECT if all_models else {"FishInv.pt": MARINE_DETECT["FishInv.pt"]}
    out: dict[str, str] = {}
    for name, url in wanted.items():
        dest = WEIGHTS_DIR / name
        _download(url, dest, min_bytes=5_000_000, force=force)
        out[name] = str(dest)
    return out


def fetch_demo_clips(force: bool = False) -> dict[str, str]:
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    for spec in DEMO_CLIPS:
        raw = RAW_DIR / Path(spec.url).name
        dest = CLIPS_DIR / spec.name
        if dest.exists() and _ffprobe_ok(dest) and not force:
            log.info("cached clip: %s (%s)", spec.name, spec.role)
            out[spec.name] = str(dest)
            continue
        try:
            _download(spec.url, raw, min_bytes=200_000, force=force)
            shutil.copyfile(raw, dest)
        except Exception as exc:
            log.error("could not fetch %s (%s): %s", spec.name, spec.url, exc)
            continue
        if not _ffprobe_ok(dest):
            log.error("clip %s failed ffprobe after download", spec.name)
            dest.unlink(missing_ok=True)
            continue
        out[spec.name] = str(dest)
    _write_manifest(out)
    return out


def make_turbid_matched(force: bool = False) -> str | None:
    """Build the labelled turbidity-augmented copy of the clear reef clip.

    ffmpeg chain: green-blue colour push + luminance lift + contrast cut + mild blur
    + light additive noise. This is a CONTROLLED FIXTURE, not footage — its only
    jobs are (a) a matched-pair restoration ablation where the only variable is
    visibility, and (b) the turbid calibration anchor until real turbid video with
    ground truth (Brackish) is fetched.
    """
    src = CLIPS_DIR / TURBID_SOURCE
    dest = CLIPS_DIR / TURBID_MATCHED
    if not src.exists():
        log.warning("turbid fixture: source %s missing — skipping", src.name)
        return None
    if dest.exists() and _ffprobe_ok(dest) and not force:
        log.info("cached clip: %s (matched-pair turbid fixture)", dest.name)
        return str(dest)
    vf = (
        "colorchannelmixer="
        "rr=0.72:rg=0.10:rb=0.02:"          # attenuate red, the fast-lost channel
        "gr=0.04:gg=0.94:gb=0.06:"
        "br=0.03:bg=0.14:bb=1.06,"          # push blue-green cast
        "eq=brightness=0.06:contrast=0.74:saturation=0.88,"
        "gblur=sigma=1.6,"
        "noise=alls=7:allf=t"
    )
    log.info("building turbid matched-pair fixture -> %s", dest.name)
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src),
           "-vf", vf, "-c:v", "libx264", "-preset", "medium", "-crf", "20",
           "-an", str(dest)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not _ffprobe_ok(dest):
        log.error("ffmpeg turbid fixture failed: %s", r.stderr[-500:])
        return None
    log.info("saved %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    return str(dest)


def fetch_kaggle(dataset: str, subdir: str) -> str | None:
    """Pull a Kaggle dataset via kagglehub if a token is available.

    Brackish:  aalborguniversity/brackish-dataset
    BrackishMOT: maltepedersen/brackishmot
    DeepFish is not on Kaggle; see docs/PHASE1_STATUS.md.
    """
    try:
        import kagglehub
    except ImportError:
        log.error("kagglehub not installed. `uv add kagglehub` then retry.")
        return None
    token = Path.home() / ".kaggle" / "kaggle.json"
    if not token.exists() and "KAGGLE_USERNAME" not in __import__("os").environ:
        log.error("no Kaggle credentials (~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY). "
                  "Brackish/BrackishMOT need an account; see docs/PHASE1_STATUS.md.")
        return None
    import kagglehub

    log.info("downloading Kaggle dataset %s (this is large) ...", dataset)
    path = kagglehub.dataset_download(dataset)
    target = REPO_ROOT / "data" / subdir
    target.mkdir(parents=True, exist_ok=True)
    log.info("kaggle dataset cached at %s ; linking -> %s", path, target)
    (target / "SOURCE").write_text(str(path))
    return str(path)


def _write_manifest(clips: dict[str, str]) -> None:
    manifest = {
        "weights": {n: str(WEIGHTS_DIR / n) for n in MARINE_DETECT},
        "demo_clips": [
            {"name": s.name, "url": s.url, "role": s.role, "note": s.note,
             "sha256_url": hashlib.sha256(s.url.encode()).hexdigest()[:12]}
            for s in DEMO_CLIPS
        ],
        "turbid_matched": {"name": TURBID_MATCHED, "derived_from": TURBID_SOURCE,
                           "kind": "labelled turbidity augmentation (ffmpeg), not footage"},
        "present": clips,
    }
    (REPO_ROOT / "data" / "clips" / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))


def main(argv: list[str] | None = None) -> int:
    basic_logging()
    ap = argparse.ArgumentParser(description="Fetch SHOAL demo clips + detector weights (cached).")
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    ap.add_argument("--all-models", action="store_true", help="also fetch MegaFauna.pt (Phase 2 species)")
    ap.add_argument("--brackish", action="store_true", help="pull Brackish via kagglehub (needs a Kaggle token)")
    ap.add_argument("--brackishmot", action="store_true", help="pull BrackishMOT via kagglehub")
    args = ap.parse_args(argv)

    log.info("== weights ==")
    fetch_weights(force=args.force, all_models=args.all_models)
    log.info("== demo clips ==")
    clips = fetch_demo_clips(force=args.force)
    log.info("== turbid matched-pair fixture ==")
    make_turbid_matched(force=args.force)

    if args.brackish:
        fetch_kaggle("aalborguniversity/brackish-dataset", "brackish")
    if args.brackishmot:
        fetch_kaggle("maltepedersen/brackishmot", "brackishmot")

    ok = (WEIGHTS_DIR / "FishInv.pt").exists() and len(clips) >= 1
    log.info("fetch %s: %d clips, weights %s",
             "OK" if ok else "INCOMPLETE", len(clips),
             "present" if (WEIGHTS_DIR / "FishInv.pt").exists() else "MISSING")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
