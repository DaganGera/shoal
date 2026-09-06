"""Render clean, dark-themed site assets from real SHOAL run directories.
Every figure here is generated from outputs/<run>/ - no invented data.
"""
import json, subprocess
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/marcben/fish")
OUT = ROOT / "web" / "assets"
OUT.mkdir(parents=True, exist_ok=True)
INK = "#0b0e13"; AMBER = "#ffb54a"; TEXT = "#e8ecf1"

def frame_at(clip, idx):
    cap = cv2.VideoCapture(str(clip)); cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx))
    ok, f = cap.read(); cap.release()
    return f if ok else None

def residency(run, clip_name, frame_idx, name):
    from scipy.ndimage import zoom, gaussian_filter
    d = ROOT / "outputs" / run
    hist = np.load(d / "residency_hist.npy").astype(float)
    meta = json.load(open(d / "residency_meta.json"))
    W, H = json.load(open(d / "results.json"))["clip_meta"]["width"], json.load(open(d / "results.json"))["clip_meta"]["height"]
    bg = frame_at(ROOT / "data" / "clips" / clip_name, frame_idx)
    # the stored histogram is coarse (96 bins); upsample + smooth for display only
    z = gaussian_filter(zoom(hist, 6, order=3), 4)
    z[z < 0] = 0
    fig, ax = plt.subplots(figsize=(12, 12 * H / W), dpi=120)
    fig.patch.set_facecolor(INK); ax.set_facecolor(INK)
    if bg is not None:
        ax.imshow(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB), extent=[0, W, H, 0])
        ax.imshow(np.zeros((H, W, 4)) + [0, 0, 0, 0.5], extent=[0, W, H, 0])
    floor = np.percentile(z[z > 0], 20) if np.any(z > 0) else 0
    zm = np.ma.masked_where(z <= floor, z)
    ax.imshow(zm, extent=[0, W, H, 0], cmap="inferno", alpha=0.92, interpolation="bilinear")
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(OUT / name, facecolor=INK, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return meta.get("total_dwell_s", 0)

def trajectories(run, clip_name, frame_idx, name):
    import matplotlib
    d = ROOT / "outputs" / run
    gj = json.load(open(d / "trajectories.geojson"))
    res = json.load(open(d / "results.json"))
    W, H = res["clip_meta"]["width"], res["clip_meta"]["height"]
    bg = frame_at(ROOT / "data" / "clips" / clip_name, frame_idx)
    fig, ax = plt.subplots(figsize=(12, 12 * H / W), dpi=120)
    fig.patch.set_facecolor(INK); ax.set_facecolor(INK)
    if bg is not None:
        ax.imshow(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB), extent=[0, W, H, 0])
        ax.imshow(np.zeros((H, W, 4)) + [0.043, 0.055, 0.075, 0.62], extent=[0, W, H, 0])
    cmap = matplotlib.colormaps["inferno"]
    feats = [f for f in gj["features"] if len(f["geometry"]["coordinates"]) > 8]
    for i, f in enumerate(feats):
        c = np.array(f["geometry"]["coordinates"], float)
        c[:, 1] = H - c[:, 1]
        ax.plot(c[:, 0], c[:, 1], "-", lw=1.6, color=cmap(0.25 + 0.6 * (i / max(len(feats) - 1, 1))), alpha=0.9, solid_capstyle="round")
        ax.plot(c[-1, 0], c[-1, 1], "o", ms=3.5, color=AMBER)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(OUT / name, facecolor=INK, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return len(feats)

def ablation_chart(run, name):
    d = ROOT / "outputs" / run
    a = json.load(open(d / "ablation.json"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), dpi=120)
    fig.patch.set_facecolor(INK)
    for ax, (title, raw, res, ymax, fmt) in zip(axes, [
        ("fish detections, sampled frames", a["raw_detections"], a["restored_detections"], None, "{:.0f}"),
        ("mean detection confidence", a["raw_mean_conf"], a["restored_mean_conf"], 1.0, "{:.2f}"),
    ]):
        ax.set_facecolor(INK)
        bars = ax.bar(["raw", "restored"], [raw, res], color=["#3a424f", AMBER], width=0.55)
        for b, v in zip(bars, [raw, res]):
            ax.text(b.get_x() + b.get_width()/2, v, " " + fmt.format(v), ha="center", va="bottom", color=TEXT, fontsize=12, fontfamily="monospace")
        ax.set_title(title, color="#8b95a4", fontsize=10, pad=12, loc="left")
        if ymax: ax.set_ylim(0, ymax)
        for s in ax.spines.values(): s.set_visible(False)
        ax.tick_params(colors="#8b95a4"); ax.set_yticks([])
        for lbl in ax.get_xticklabels(): lbl.set_color("#8b95a4"); lbl.set_fontfamily("monospace")
    fig.tight_layout(pad=1.5)
    fig.savefig(OUT / name, facecolor=INK, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return a

def visibility_chart(run, name):
    import pandas as pd
    d = ROOT / "outputs" / run
    v = pd.read_csv(d / "visibility.csv")
    gate = json.load(open(d / "run.json"))["config"]["restore"]["gate_threshold"]
    fig, ax = plt.subplots(figsize=(11, 2.8), dpi=120)
    fig.patch.set_facecolor(INK); ax.set_facecolor(INK)
    ax.plot(v["timestamp"], v["visibility_score"], color=TEXT, lw=1.4)
    ax.axhline(gate, color=AMBER, ls=(0, (4, 3)), lw=1)
    below = v[v["visibility_score"] < gate]
    ax.fill_between(v["timestamp"], 0, v["visibility_score"], where=v["visibility_score"] < gate, color=AMBER, alpha=0.18)
    ax.set_ylim(0, 1); ax.set_xlim(v["timestamp"].min(), v["timestamp"].max())
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(colors="#8b95a4", labelsize=9)
    ax.set_xlabel("seconds", color="#8b95a4", fontsize=9)
    ax.text(v["timestamp"].iloc[-1], gate + 0.03, f" restoration gate {gate:.2f}", color=AMBER, fontsize=9, ha="right", fontfamily="monospace")
    fig.tight_layout()
    fig.savefig(OUT / name, facecolor=INK, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

# --- video: tight ~9s 720p loop from the reef_clear annotated output ---
def clip_loop(run, name, start, dur):
    src = ROOT / "outputs" / run / "annotated.mp4"
    subprocess.run(["ffmpeg","-nostdin","-v","error","-y","-ss",str(start),"-t",str(dur),"-i",str(src),
        "-vf","scale=960:-2","-c:v","libx264","-pix_fmt","yuv420p","-movflags","+faststart","-preset","veryfast","-crf","26","-an",
        str(OUT / name)], check=True)

# anemone_static has the steadiest camera, so its residency map is the most legible
dwell_an = residency("anemone_static", "anemone_static.mp4", 300, "residency.png")
dwell_rc = json.load(open(ROOT / "outputs" / "reef_clear" / "residency_meta.json")).get("total_dwell_s", 0)
ntraj = trajectories("reef_clear", "reef_clear.mp4", 516, "trajectories.png")
abl = ablation_chart("reef_snapper", "ablation.png")
visibility_chart("reef_turbid_matched", "visibility.png")
clip_loop("reef_clear", "tracking.mp4", 3.0, 9.0)

# raw / restored pair (turbid)
d = ROOT / "outputs" / "reef_turbid_matched"
for k in ("raw", "restored"):
    im = cv2.imread(str(d / f"ab_example_{k}.png"))
    cv2.imwrite(str(OUT / f"ab_{k}.jpg"), cv2.resize(im, (960, int(960*im.shape[0]/im.shape[1]))), [cv2.IMWRITE_JPEG_QUALITY, 88])

# real numbers bundle for the page
data = {}
for rid in ["reef_clear","reef_turbid_matched","anemone_static","reef_snapper"]:
    r = json.load(open(ROOT / "outputs" / rid / "results.json"))
    a = r["ecology"]["abundance"]
    data[rid] = {
        "frames": r["clip_meta"]["n_frames"], "fps": round(r["clip_meta"]["fps"]),
        "duration_s": round(r["clip_meta"]["duration_s"], 1),
        "camera_static": r["camera_motion"]["is_static"],
        "camera_drift_pct": round(r["camera_motion"]["cumulative_drift_frac"]*100),
        "visibility_mean": round(r["visibility"]["score_mean"], 2),
        "maxn": a["maxn_total"], "maxn_t": a["maxn_timestamp"],
        "tracks": a["unique_tracks_total"], "accepted": r["n_tracks_accepted"],
        "rejected_short": r["repair"].get("n_tracks_rejected_short"),
        "stitched": len(r["repair"].get("stitched_track_ids", [])),
        "interpolated": r["repair"].get("n_points_interpolated"),
        "behaviour": r["behaviour"].get("state_counts"),
        "ablation": r["ablation"]["verdict"],
        "enhance_ms": round(r["ablation"]["enhance_ms_mean"]),
    }
data["_derived"] = {"dwell_reef_clear_s": round(dwell_rc), "dwell_anemone_s": round(dwell_an),
                    "n_trajectories": ntraj}
rj = json.load(open(ROOT / "outputs" / "reef_clear" / "run.json"))
data["_env"] = {"device": "NVIDIA RTX 3050 6GB", "python": rj["python"],
                "torch": rj["package_versions"].get("torch"), "ultralytics": rj["package_versions"].get("ultralytics"),
                "movingpandas": rj["package_versions"].get("movingpandas")}
json.dump(data, open(OUT / "data.json", "w"), indent=2)
print("assets:", sorted(p.name for p in OUT.iterdir()))
print("dwell reef_clear:", dwell_rc, "| trajectories:", ntraj, "| ablation enhance ms:", abl["enhance_ms_mean"])
