import argparse, json, os, re, subprocess, sys
from collections import defaultdict
import cv2
import numpy as np
import requests

def py_cmd():
    return "py" if os.name == "nt" else "python3"

def get_metadata(url, cookies=None):
    cmd = [py_cmd(), "-m", "yt_dlp", "--dump-single-json", "--skip-download"]
    if cookies and os.path.exists(cookies):
        cmd += ["--cookies", cookies]
    cmd.append(url)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"yt-dlp metadata failed:\n{res.stderr}")
    return json.loads(res.stdout)

def download_audio(url, out_path="temp_audio.m4a", cookies=None):
    if os.path.exists(out_path):
        os.remove(out_path)
    cmd = [py_cmd(), "-m", "yt_dlp", "-f", "bestaudio[ext=m4a]/bestaudio", "-o", out_path]
    if cookies and os.path.exists(cookies):
        cmd += ["--cookies", cookies]
    cmd.append(url)
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path if os.path.exists(out_path) else None

def transcribe_segments(audio_path):
    """Fast Whisper transcription with timestamps so we can snap clips to hooks & endings."""
    print("[1/4] Transcribing speech with faster-whisper...")
    try:
        from faster_whisper import WhisperModel
        device = "cuda" if subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0 else "cpu"
        compute = "float16" if device == "cuda" else "int8"
        model = WhisperModel("small", device=device, compute_type=compute)
        segs, _ = model.transcribe(audio_path, beam_size=3, vad_filter=True)
        out = []
        for s in segs:
            txt = s.text.strip()
            if txt:
                out.append({"start": float(s.start), "end": float(s.end), "text": txt})
        print(f"  -> Captured {len(out)} speech segments on {device.upper()}.")
        return out
    except Exception as e:
        print(f"  -> Whisper skipped ({e}); using audio + chat scoring.")
        return []

def score_audio_variation(audio_path, bucket_sec=10):
    print("[2/4] Measuring audio excitement & volume jumps...")
    deltas = defaultdict(float)
    if not audio_path or not os.path.exists(audio_path):
        return deltas
    ff_cmd = [
        "ffmpeg", "-i", audio_path,
        "-af", "astats=metadata=1:reset=100,ametadata=print:key=lavfi.astats.Overall.RMS_level",
        "-f", "null", "-"
    ]
    res = subprocess.run(ff_cmd, capture_output=True, text=True)
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
    return deltas

def score_chat_and_motion(url, duration, chat_side="right", cookies=None, bucket_sec=10):
    print("[3/4] Checking chat velocity & on-screen chat motion...")
    chat_file = "temp_chat.json"
    if os.path.exists(chat_file):
        os.remove(chat_file)
    subprocess.run([py_cmd(), "-m", "chat_downloader", url, "--output", chat_file, "--quiet"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    chat_buckets = defaultdict(int)
    if os.path.exists(chat_file):
        try:
            with open(chat_file, "r", encoding="utf-8") as f:
                for msg in json.load(f):
                    t = msg.get("time_in_seconds", -1)
                    if 0 <= t <= duration:
                        chat_buckets[int(t // bucket_sec)] += 1
        except Exception:
            pass
        finally:
            os.remove(chat_file)

    motion_buckets = defaultdict(float)
    if chat_side != "none":
        get_cmd = [py_cmd(), "-m", "yt_dlp", "-f", "best[height<=360]/worst", "-g"]
        if cookies and os.path.exists(cookies):
            get_cmd += ["--cookies", cookies]
        get_cmd.append(url)
        r = subprocess.run(get_cmd, capture_output=True, text=True)
        stream_url = r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else None
        if stream_url:
            w, h = 640, 360
            proc = subprocess.Popen(
                ["ffmpeg", "-loglevel", "error", "-i", stream_url, "-vf", f"fps=2,scale={w}:{h},format=gray", "-f", "rawvideo", "-"],
                stdout=subprocess.PIPE, bufsize=w*h*4
            )
            y1, y2 = int(h * 0.25), int(h * 0.90)
            x1, x2 = (0, int(w * 0.26)) if chat_side == "left" else (int(w * 0.74), w)
            prev_crop = None
            idx = 0
            b_vals = defaultdict(list)
            while True:
                raw = proc.stdout.read(w * h)
                if len(raw) < w * h:
                    break
                crop = np.frombuffer(raw, dtype=np.uint8).reshape((h, w))[y1:y2, x1:x2]
                if prev_crop is not None:
                    diff = cv2.absdiff(crop, prev_crop)
                    _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                    b_vals[int((idx * 0.5) // bucket_sec)].append(float(np.mean(th)))
                prev_crop = crop
                idx += 1
            proc.wait()
            for b, vals in b_vals.items():
                if vals:
                    motion_buckets[b] = sum(vals) / len(vals)

    return chat_buckets, motion_buckets

def ai_select_hook_to_payoff_clips(segments, num_clips, min_len, max_len):
    """Uses free Groq or Gemini API if key is set; otherwise builds complete hook-to-payoff windows."""
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()

    if segments and (groq_key or gemini_key):
        compact = "\n".join(
            f"[{int(s['start'])}s-{int(s['end'])}s] {s['text']}" for s in segments[:600]
        )
        prompt = (
            f"Select up to {num_clips} engaging Short-form video clips from this transcript.\n"
            f"Rules:\n"
            f"1. Each clip MUST start on a strong hook/setup line and end on a complete payoff/punchline/resolution (never cut mid-sentence).\n"
            f"2. Each clip duration MUST be between {min_len} and {max_len} seconds (strictly <= {max_len}s).\n"
            f"3. Return ONLY a JSON list of objects: [{\"start\": 120, \"end\": 162, \"reason\": \"...\"}].\n\n"
            f"Transcript:\n{compact}"
        )
        try:
            if groq_key:
                resp = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
                    json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "temperature": 0.3},
                    timeout=45
                )
                content = resp.json()["choices"][0]["message"]["content"]
            else:
                resp = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}",
                    json={"contents": [{"parts": [{"text": prompt}]}]},
                    timeout=45
                )
                content = resp.json()["candidates"][0]["content"]["parts"][0]["text"]

            match = re.search(r"\[[\s\S]*\]", content)
            if match:
                raw_clips = json.loads(match.group(0))
                valid = []
                for c in raw_clips:
                    st = float(c.get("start", 0))
                    en = float(c.get("end", 0))
                    dur = en - st
                    if dur < min_len:
                        en = st + min_len
                    elif dur > max_len:
                        en = st + max_len
                    valid.append((st, en, c.get("reason", "AI hook + payoff")))
                if valid:
                    print(f"  -> AI selected {len(valid)} hook-to-payoff clips ({min_len}s-{max_len}s).")
                    return valid[:num_clips]
        except Exception as e:
            print(f"  -> AI API fallback ({e}); using transcript + energy window builder.")

    # Structured hook-to-payoff window builder (no API key needed)
    if segments:
        hook_words = re.compile(r"\b(what|why|how|wait|bro|no way|crazy|secret|insane|look|watch|never|mistake|truth|caught)\b", re.I)
        candidates = []
        for i, seg in enumerate(segments):
            hook_bonus = 25.0 if hook_words.search(seg["text"]) else 0.0
            start = max(0.0, seg["start"] - 1.5)
            end = start
            text_acc = []
            for j in range(i, len(segments)):
                nxt_end = min(segments[j]["end"] + 1.0, start + max_len)
                if nxt_end - start > max_len:
                    break
                end = nxt_end
                text_acc.append(segments[j]["text"])
                # Stop at a natural sentence boundary once minimum duration is reached
                if (end - start) >= min_len and re.search(r"[.!?]$", segments[j]["text"].strip()):
                    break
            if (end - start) < min_len:
                end = start + min_len
            score = hook_bonus + min(30.0, len(" ".join(text_acc).split()) * 0.5)
            candidates.append((score, start, end))

        candidates.sort(reverse=True)
        picked = []
        for sc, st, en in candidates:
            if any(abs(st - p_st) < max_len for p_st, _, _ in picked):
                continue
            picked.append((st, en, f"Score {sc:.0f}"))
            if len(picked) >= num_clips:
                break
        picked.sort()
        return picked

    return []

def render_clip(url, idx, start, end, out_dir, cookies=None, split_cam=False):
    raw_path = os.path.join(out_dir, "raw", f"raw_{idx:03d}.mp4")
    final_path = os.path.join(out_dir, "final_shorts", f"short_{idx:03d}.mp4")
    section = f"*{int(start)}-{int(end)}"

    dl_cmd = [
        py_cmd(), "-m", "yt_dlp", "--download-sections", section, "--force-keyframes-at-cuts",
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
    parser = argparse.ArgumentParser(description="Hook + Payoff Vertical Shorts Clipper (Max 50s)")
    parser.add_argument("url", help="YouTube or Twitch stream/video URL")
    parser.add_argument("--clips", type=int, default=40, help="Number of clips (default: 40)")
    parser.add_argument("--min-len", type=int, default=25, help="Minimum clip seconds (default: 25)")
    parser.add_argument("--max-len", type=int, default=50, help="Maximum clip seconds (default: 50)")
    parser.add_argument("--out", default=".", help="Output folder")
    parser.add_argument("--cookies", default=None, help="Path to cookies.txt")
    parser.add_argument("--chat-box", choices=["left", "right", "none"], default="right")
    parser.add_argument("--split-cam", action="store_true")
    args = parser.parse_args()

    max_len = min(args.max_len, 50)
    min_len = min(args.min_len, max_len)

    os.makedirs(os.path.join(args.out, "raw"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "final_shorts"), exist_ok=True)

    meta = get_metadata(args.url, args.cookies)
    duration = float(meta.get("duration") or 36000)
    print(f"Loaded: {meta.get('title')} ({int(duration // 60)} mins) | Target length: {min_len}s-{max_len}s")

    audio_path = download_audio(args.url, "temp_audio.m4a", args.cookies)
    segments = transcribe_segments(audio_path) if audio_path else []
    audio_scores = score_audio_variation(audio_path)
    if audio_path and os.path.exists(audio_path):
        os.remove(audio_path)

    chat_scores, motion_scores = score_chat_and_motion(args.url, duration, args.chat_box, args.cookies)

    print("[4/4] Selecting hook-to-payoff clips...")
    clips = ai_select_hook_to_payoff_clips(segments, args.clips, min_len, max_len)

    # Fallback if transcript was empty (e.g., pure gameplay/music)
    if not clips:
        combined = defaultdict(float)
        for b, v in audio_scores.items():
            combined[b] += v * 0.45
        for b, v in motion_scores.items():
            combined[b] += v * 0.35
        for b, v in chat_scores.items():
            combined[b] += v * 0.20
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        for b, sc in ranked:
            st = max(0.0, (b * 10) - 5.0)
            en = min(duration, st + min(max_len, max(min_len, 40)))
            if any(abs(st - p_st) < max_len for p_st, _, _ in clips):
                continue
            clips.append((st, en, f"Energy {sc:.1f}"))
            if len(clips) >= args.clips:
                break
        clips.sort()

    for idx, (st, en, info) in enumerate(clips, 1):
        en = min(duration, st + min(max_len, max(min_len, en - st)))
        print(f"[{idx}/{len(clips)}] {int(st)}s -> {int(en)}s ({int(en - st)}s) | {info}")
        render_clip(args.url, idx, st, en, args.out, args.cookies, args.split_cam)

    print("\nDONE! Check final_shorts for complete hook-to-payoff vertical videos (<= 50s).")

if __name__ == "__main__":
    main()
