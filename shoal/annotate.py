"""Annotated-video renderer (dashboard tab 1, plan §11.1).

Boxes with stable track IDs, fading trajectory tails, a live on-frame counter, and
— when a visibility trace is supplied — the per-frame visibility score with a
"RESTORED" flag when the gate fired. Deliberately cv2-only so it runs anywhere the
pipeline runs.

The trajectory tail is drawn from the REPAIRED, smoothed positions, so what the
viewer sees on screen is the same signal every downstream metric is computed from.
"""

from __future__ import annotations

import colorsys
import logging
from pathlib import Path

import cv2
import pandas as pd

from shoal.ingest import iter_frames, probe


def _id_colour(track_id: int) -> tuple[int, int, int]:
    """Deterministic, well-separated BGR colour per track id (golden-ratio hue hop)."""
    h = (track_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def render(
    clip_path: str | Path,
    tracks_df: pd.DataFrame,
    out_path: str | Path,
    log: logging.Logger,
    visibility_df: pd.DataFrame | None = None,
    restored_frames: set[int] | None = None,
    tail_seconds: float = 1.5,
    max_frames: int | None = None,
) -> Path:
    meta = probe(clip_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # cv2 writes MPEG-4 Part 2 (mp4v), which Chrome / Streamlit's <video> cannot
    # decode. Write to a temp file, then transcode to H.264 yuv420p so the dashboard
    # actually plays it.
    tmp_path = out_path.with_name(out_path.stem + "_mp4v.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_path), fourcc, meta.fps, (meta.width, meta.height))

    by_frame = {f: g for f, g in tracks_df.groupby("frame")}
    tail_frames = int(tail_seconds * meta.fps)
    vis_by_frame = (visibility_df.set_index("frame")["visibility_score"].to_dict()
                    if visibility_df is not None else {})
    restored_frames = restored_frames or set()

    # pre-index each track's smoothed path for fast tail slicing
    paths = {
        int(tid): g.sort_values("frame")[["frame", "x", "y"]].to_numpy()
        for tid, g in tracks_df.groupby("track_id")
    }

    n = 0
    for frame in iter_frames(clip_path, max_frames=max_frames):
        img = frame.bgr
        fi = frame.index
        present = by_frame.get(fi)
        live_count = 0

        if present is not None:
            for row in present.itertuples():
                tid = int(row.track_id)
                colour = _id_colour(tid)
                x, y, w, h = row.x, row.y, row.w, row.h
                p1 = (int(x - w / 2), int(y - h / 2))
                p2 = (int(x + w / 2), int(y + h / 2))
                interp = bool(getattr(row, "is_interpolated", False))
                cv2.rectangle(img, p1, p2, colour, 1 if interp else 2,
                              lineType=cv2.LINE_AA)
                label = f"fish #{tid}"
                state = getattr(row, "behaviour_state", "unknown")
                if isinstance(state, str) and state not in ("unknown", "nan"):
                    label += f"  {state}"
                cv2.putText(img, label, (p1[0], max(12, p1[1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
                if not interp:
                    live_count += 1

                # fading tail
                path = paths.get(tid)
                if path is not None:
                    seg = path[(path[:, 0] <= fi) & (path[:, 0] > fi - tail_frames)]
                    for k in range(1, len(seg)):
                        a = (int(seg[k - 1, 1]), int(seg[k - 1, 2]))
                        b = (int(seg[k, 1]), int(seg[k, 2]))
                        alpha = k / len(seg)
                        faded = tuple(int(c * alpha) for c in colour)
                        cv2.line(img, a, b, faded, 1, cv2.LINE_AA)

        # HUD
        cv2.rectangle(img, (0, 0), (meta.width, 28), (0, 0, 0), -1)
        hud = f"t={frame.timestamp:5.1f}s   frame {fi}   fish in frame: {live_count}"
        if fi in vis_by_frame:
            hud += f"   visibility={vis_by_frame[fi]:.2f}"
            if fi in restored_frames:
                hud += "  [RESTORED]"
        cv2.putText(img, hud, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(img)
        n += 1

    writer.release()

    ok = _transcode_h264(tmp_path, out_path, meta.fps, log)
    if ok:
        tmp_path.unlink(missing_ok=True)
    else:                                   # ffmpeg missing/failed: keep the mp4v file
        tmp_path.replace(out_path)
        log.warning("annotate: H.264 transcode failed; kept mp4v (may not play in-browser)")

    log.info("annotate: wrote %s (%d frames, %dx%d @ %.1f fps)",
             out_path, n, meta.width, meta.height, meta.fps)
    return out_path


def _transcode_h264(src: Path, dst: Path, fps: float, log: logging.Logger) -> bool:
    import shutil
    import subprocess

    if not shutil.which("ffmpeg"):
        return False
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-r", f"{fps:.6f}", "-i", str(src),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
           "-preset", "veryfast", "-crf", "23", str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log.warning("ffmpeg transcode: %s", r.stderr[-300:])
        return False
    return dst.exists() and dst.stat().st_size > 0
