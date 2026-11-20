import argparse, json, os, re, subprocess, sys
from collections import defaultdict
import cv2
import numpy as np

def get_metadata(url, cookies=None):
    cmd = ["py" if os.name == "nt" else "python3", "-m", "yt_dlp", "--dump-single-json", "--skip-download"]
    if cookies and os.path.exists(cookies):
        cmd += ["--cookies", cookies]
    cmd.append(url)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"yt-dlp metadata failed:\n{res.stderr}")
    return json.loads(res.stdout)

def score_chat_api(url, duration, bucket_sec=10):
    print("[1/4] Checking YouTube/Twitch live chat replay API...")
    chat_file = "temp_chat.json"
    if os.path.exists(chat_file):
        os.remove(chat_file)

    py_bin = "py" if os.name == "nt" else "python3"
    cmd = [py_bin, "-m", "chat_downloader", url, "--output", chat_file, "--quiet"]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    buckets = defaultdict(int)
    if not os.path.exists(chat_file):
        print("  -> Live chat API unavailable; relying on on-screen chat motion + audio.")
        return buckets

    try:
        with open(chat_file, "r", encoding="utf-8") as f:
            messages = json.load(f)
        for msg in messages:
            t = msg.get("time_in_seconds", -1)
            if 0 <= t <= duration:
                buckets[int(t // bucket_sec)] += 1
        print(f"  -> Parsed {len(messages)} chat messages across {len(buckets)} windows.")
    except Exception:
        print("  -> Chat parse skipped.")
    finally:
        if os.path.exists(chat_file):
            os.remove(chat_file)
    return buckets

def score_onscreen_chat(url, chat_side="right", cookies=None, bucket_sec=10):
    """Streams 360p video via FFmpeg pipe and measures pixel scroll velocity in the on-screen chat box."""
    if chat_side == "none":
        return {}
    print(f"[2/4] Scanning on-screen chat box motion ({chat_side} side) at 2 FPS...")

    py_bin = "py" if os.name == "nt" else "python3"
    get_url_cmd = [py_bin, "-m", "yt_dlp", "-f", "best[height<=360]/worst", "-g"]
    if cookies and os.path.exists(cookies):
        get_url_cmd += ["--cookies", cookies]
    get_url_cmd.append(url)
    res = subprocess.run(get_url_cmd, capture_output=True, text=True)
    stream_url = res.stdout.strip().splitlines()[0] if res.returncode == 0 and res.stdout.strip() else None

    if not stream_url:
        print("  -> Could not open low-res stream for visual chat scan; skipping.")
        return {}

    # Decode at 2 FPS, scale to 640x360 raw grayscale frames directly in RAM
    w, h = 640, 360
    ff_cmd = [
        "ffmpeg", "-loglevel", "error", "-i", stream_url,
        "-vf", f"fps=2,scale={w}:{h},format=gray",
        "-f", "rawvideo", "-"
    ]
    proc = subprocess.Popen(ff_cmd, stdout=subprocess.PIPE, bufsize=w*h*4)

    # Crop region for typical on-screen chat overlay (outer 26% width, middle-bottom 65% height)
    y1, y2 = int(h * 0.25), int(h * 0.90)
    if chat_side == "left":
        x1, x2 = 0, int(w * 0.26)
    else:
        x1, x2 = int(w * 0.74), w

    prev_crop = None
    frame_idx = 0
    bucket_motion = defaultdict(list)

    while True:
        raw = proc.stdout.read(w * h)
        if len(raw) < w * h:
            break
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w))
        crop = frame[y1:y2, x1:x2]

        if prev_crop is not None:
            # Measure mean absolute pixel change (text scrolling up causes sharp delta spikes)
            diff = cv2.absdiff(crop, prev_crop)
            # Threshold out minor compression noise, count moving text pixels
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            motion_score = float(np.mean(thresh))
            t_sec = frame_idx * 0.5  # 2 FPS = 0.5s per frame
            bucket_motion[int(t_sec // bucket_sec)].append(motion_score)

        prev_crop = crop
        frame_idx += 1

    proc.wait()
    averaged = {b: sum(vals) / len(vals) for b, vals in bucket_motion.items() if vals}
    print(f"  -> Scanned {frame_idx} frames ({int(frame_idx * 0.5 // 60)} mins) for visual chat scroll spikes.")
    return averaged

def score_audio_variation(url, cookies=None, bucket_sec=10):
    print("[3/4] Scanning audio variation (sudden volume jumps)...")
    audio_file = "temp_audio.m4a"
    if os.path.exists(audio_file):
        os.remove(audio_file)

    py_bin = "py" if os.name == "nt" else "python3"
    cmd = [py_bin, "-m", "yt_dlp", "-f", "bestaudio[ext=m4a]/bestaudio", "-o", audio_file]
    if cookies and os.path.exists(cookies):
        cmd += ["--cookies", cookies]
    cmd.append(url)
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    deltas = defaultdict(float)
    if not os.path.exists(audio_file):
        print("  -> Audio download failed; skipping audio scoring.")
        return deltas

    ff_cmd = [
        "ffmpeg", "-i", audio_file,
        "-af", "astats=metadata=1:reset=100,ametadata=print:key=lavfi.astats.Overall.RMS_level",
        "-f", "null", "-"
    ]
    res = subprocess.run(ff_cmd, capture_output=True, text=True)
    if os.path.exists(audio_file):
        os.remove(audio_file)

    times = re.findall(r"pts_time:([0-9.]+)", res.stderr)
    levels = re.findall(r"RMS_level=([-0-9.]+)", res.stderr)

    bucket_rms = defaultdict(list)
    for t, lvl in zip(times, levels):
        try:
            val = float(lvl)
            if val > -60.0:
                bucket_rms[int(float(t) // bucket_sec)].append(val)
        except ValueError:
            pass

    sorted_b = sorted(bucket_rms.keys())
    for i, b in enumerate(sorted_b):
        curr_max = max(bucket_rms[b])
        prev_avg = sum(bucket_rms[sorted_b[i-1]]) / len(bucket_rms[sorted_b[i-1]]) if i > 0 else curr_max
        jump = max(0.0, curr_max - prev_avg)
        deltas[b] = (jump * 2.0) + max(0.0, curr_max + 30.0)

    print(f"  -> Scored {len(deltas)} audio windows.")
    return deltas

def normalize(score_dict):
    if not score_dict:
        return {}
    max_val = max(score_dict.values()) or 1.0
    return {k: (v / max_val) * 100.0 for k, v in score_dict.items()}

def render_clip(url, idx, start, end, out_dir, cookies=None, split_cam=False):
    raw_path = os.path.join(out_dir, "raw", f"raw_{idx:03d}.mp4")
    final_path = os.path.join(out_dir, "final_shorts", f"short_{idx:03d}.mp4")
    section = f"*{int(start)}-{int(end)}"

    py_bin = "py" if os.name == "nt" else "python3"
    dl_cmd = [
        py_bin, "-m", "yt_dlp", "--download-sections", section, "--force-keyframes-at-cuts",
        "-f", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
        "-o", raw_path
    ]
    if cookies and os.path.exists(cookies):
        dl_cmd += ["--cookies", cookies]
    dl_cmd.append(url)
    subprocess.run(dl_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if not os.path.exists(raw_path):
        return False

    if split_cam:
        vf = (
            "split[cam][game];"
            "[cam]crop=iw*0.28:ih*0.28:0:0,scale=1080:670:force_original_aspect_ratio=increase,crop=1080:670[top];"
            "[game]scale=1080:1250:force_original_aspect_ratio=increase,crop=1080:1250[bot];"
            "[top][bot]vstack"
        )
    else:
        vf = (
            "split[orig][bg];"
            "[bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,gblur=sigma=20[blurred];"
            "[orig]scale=1080:-1[fg];"
            "[blurred][fg]overlay=(W-w)/2:(H-h)/2"
        )

    for encoder, extra in [
        ("h264_qsv", ["-b:v", "5M"]),
        ("h264_nvenc", ["-preset", "p4", "-cq", "23"]),
        ("libx264", ["-preset", "ultrafast", "-crf", "23"]),
    ]:
        ff_cmd = ["ffmpeg", "-y", "-i", raw_path, "-vf", vf, "-c:v", encoder, *extra, "-c:a", "copy", final_path]
        r = subprocess.run(ff_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode == 0 and os.path.exists(final_path):
            return True
    return False

def main():
    parser = argparse.ArgumentParser(description="Multi-Signal Hype Clipper (API Chat + Visual Chat Motion + Audio Delta + Heatmap)")
    parser.add_argument("url", help="YouTube or Twitch stream/video URL")
    parser.add_argument("--clips", type=int, default=40, help="Number of clips (default: 40)")
    parser.add_argument("--len", type=int, default=30, help="Clip length in seconds (default: 30)")
    parser.add_argument("--out", default=".", help="Output folder")
    parser.add_argument("--cookies", default=None, help="Path to cookies.txt")
    parser.add_argument("--chat-box", choices=["left", "right", "none"], default="right", help="Side of screen where chat overlay sits (default: right)")
    parser.add_argument("--split-cam", action="store_true", help="Stack top-left webcam over gameplay")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out, "raw"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "final_shorts"), exist_ok=True)

    meta = get_metadata(args.url, args.cookies)
    duration = meta.get("duration") or 36000
    print(f"Loaded: {meta.get('title')} ({int(duration // 60)} mins)")

    chat_api_norm = normalize(score_chat_api(args.url, duration))
    chat_vis_norm = normalize(score_onscreen_chat(args.url, args.chat_box, args.cookies))
    audio_norm = normalize(score_audio_variation(args.url, args.cookies))

    print("[4/4] Fusing all signals...")
    heat_raw = defaultdict(float)
    for item in (meta.get("heatmap") or []):
        b = int(((item["start_time"] + item["end_time"]) / 2) // 10)
        heat_raw[b] = max(heat_raw[b], item.get("value", 0))
    heat_norm = normalize(heat_raw)

    all_buckets = set(chat_api_norm.keys()) | set(chat_vis_norm.keys()) | set(audio_norm.keys()) | set(heat_norm.keys())
    fused = []
    for b in all_buckets:
        # Use API chat if available, otherwise rely heavily on visual on-screen chat scroll velocity
        chat_score = chat_api_norm.get(b, 0) if chat_api_norm else chat_vis_norm.get(b, 0)
        score = (0.45 * chat_score) + (0.35 * audio_norm.get(b, 0)) + (0.20 * heat_norm.get(b, 0))
        fused.append((score, b * 10))

    fused.sort(reverse=True)
    selected = []
    for score, center in fused:
        start = max(0, center - (args.len // 2))
        end = min(duration, start + args.len)
        if any(abs(start - s) < (args.len - 5) for s, _, _ in selected):
            continue
        selected.append((start, end, round(score, 1)))
        if len(selected) >= args.clips:
            break

    selected.sort()
    print(f"\nRendering top {len(selected)} multi-signal clips into {os.path.abspath(os.path.join(args.out, 'final_shorts'))} ...\n")
    for i, (s, e, sc) in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] {int(s)}s -> {int(e)}s (Hype Score: {sc}/100)")
        render_clip(args.url, i, s, e, args.out, args.cookies, args.split_cam)

    print("\nDONE! All vertical Shorts are ready.")

if __name__ == "__main__":
    main()
