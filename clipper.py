import argparse, json, os, re, subprocess, sys
from collections import defaultdict

def get_metadata(url, cookies=None):
    cmd = ["py", "-m", "yt_dlp", "--dump-single-json", "--skip-download"]
    if cookies and os.path.exists(cookies):
        cmd += ["--cookies", cookies]
    cmd.append(url)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"yt-dlp metadata failed:\n{res.stderr}")
    return json.loads(res.stdout)

def score_chat_spikes(url, duration, bucket_sec=10):
    print("[1/3] Scanning live chat velocity spikes...")
    chat_file = "temp_chat.json"
    if os.path.exists(chat_file):
        os.remove(chat_file)

    cmd = ["py", "-m", "chat_downloader", url, "--output", chat_file, "--quiet"]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    buckets = defaultdict(int)
    if not os.path.exists(chat_file):
        print("  -> No live chat replay found; using audio + heatmap.")
        return buckets

    try:
        with open(chat_file, "r", encoding="utf-8") as f:
            messages = json.load(f)
        for msg in messages:
            t = msg.get("time_in_seconds", -1)
            if 0 <= t <= duration:
                buckets[int(t // bucket_sec)] += 1
        print(f"  -> Found {len(messages)} chat messages across {len(buckets)} time windows.")
    except Exception:
        print("  -> Chat parse skipped.")
    finally:
        if os.path.exists(chat_file):
            os.remove(chat_file)
    return buckets

def score_audio_variation(url, cookies=None, bucket_sec=10):
    print("[2/3] Scanning audio variation (sudden volume jumps)...")
    audio_file = "temp_audio.m4a"
    if os.path.exists(audio_file):
        os.remove(audio_file)

    cmd = ["py", "-m", "yt_dlp", "-f", "bestaudio[ext=m4a]/bestaudio", "-o", audio_file]
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

    dl_cmd = [
        "py", "-m", "yt_dlp", "--download-sections", section, "--force-keyframes-at-cuts",
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
    parser = argparse.ArgumentParser(description="Multi-Signal Hype Clipper")
    parser.add_argument("url", help="YouTube or Twitch stream/video URL")
    parser.add_argument("--clips", type=int, default=40, help="Number of clips (default: 40)")
    parser.add_argument("--len", type=int, default=30, help="Clip length in seconds (default: 30)")
    parser.add_argument("--out", default=".", help="Output folder")
    parser.add_argument("--cookies", default=None, help="Path to cookies.txt")
    parser.add_argument("--split-cam", action="store_true", help="Stack top-left webcam over gameplay")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out, "raw"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "final_shorts"), exist_ok=True)

    meta = get_metadata(args.url, args.cookies)
    duration = meta.get("duration") or 36000
    print(f"Loaded: {meta.get('title')} ({int(duration // 60)} mins)")

    chat_norm = normalize(score_chat_spikes(args.url, duration))
    audio_norm = normalize(score_audio_variation(args.url, args.cookies))

    print("[3/3] Checking YouTube heatmap & fusing signals...")
    heat_raw = defaultdict(float)
    for item in (meta.get("heatmap") or []):
        b = int(((item["start_time"] + item["end_time"]) / 2) // 10)
        heat_raw[b] = max(heat_raw[b], item.get("value", 0))
    heat_norm = normalize(heat_raw)

    all_buckets = set(chat_norm.keys()) | set(audio_norm.keys()) | set(heat_norm.keys())
    fused = []
    for b in all_buckets:
        score = (0.45 * chat_norm.get(b, 0)) + (0.35 * audio_norm.get(b, 0)) + (0.20 * heat_norm.get(b, 0))
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
    print(f"\nRendering top {len(selected)} clips into {os.path.abspath(os.path.join(args.out, 'final_shorts'))} ...\n")
    for i, (s, e, sc) in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] {int(s)}s -> {int(e)}s (Hype Score: {sc}/100)")
        render_clip(args.url, i, s, e, args.out, args.cookies, args.split_cam)

    print("\nDONE! All vertical Shorts are ready.")

if __name__ == "__main__":
    main()
