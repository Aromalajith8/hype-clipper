import argparse, csv, hashlib, json, os, re, subprocess, sys
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

def download_audio(url, out_path, cookies=None):
    if os.path.exists(out_path):
        os.remove(out_path)
    cmd = [py_cmd(), "-m", "yt_dlp", "-f", "bestaudio[ext=m4a]/bestaudio", "-o", out_path]
    if cookies and os.path.exists(cookies):
        cmd += ["--cookies", cookies]
    cmd.append(url)
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path if os.path.exists(out_path) else None

def transcribe_segments(audio_path, cache_path):
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached:
                print(f"[1/4] Loaded {len(cached)} cached transcript segments.")
                return cached
        except Exception:
            pass

    print("[1/4] Transcribing speech with faster-whisper...")
    try:
        from faster_whisper import WhisperModel
        has_gpu = subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        device = "cuda" if has_gpu else "cpu"
        compute = "float16" if has_gpu else "int8"
        model_size = "small" if has_gpu else "base"
        model = WhisperModel(model_size, device=device, compute_type=compute)
        segs, _ = model.transcribe(audio_path, beam_size=3, vad_filter=True)
        out = []
        for s in segs:
            txt = s.text.strip()
            if txt:
                out.append({"start": float(s.start), "end": float(s.end), "text": txt})
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"  -> Transcribed {len(out)} segments on {device.upper()} ({model_size}).")
        return out
    except Exception as e:
        print(f"  -> Whisper skipped ({e}); falling back to audio/chat signals.")
        return []

def score_audio_variation(audio_path, bucket_sec=10):
    print("[2/4] Scoring audio energy & sudden volume jumps...")
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
    print("[3/4] Checking live chat velocity & on-screen chat scroll motion...")
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

def _snap_to_hook_and_payoff(segments, start_sec, end_sec, min_len, max_len, duration):
    start = max(0.0, float(start_sec))
    end = min(float(duration), float(end_sec))
    if end - start < min_len:
        end = min(float(duration), start + min_len)
    if end - start > max_len:
        end = start + max_len

    if not segments:
        return round(start, 2), round(min(float(duration), end), 2)

    hook_re = re.compile(r"\b(wait|what|why|how|no way|bro|crazy|insane|look|watch|secret|never|caught|mistake|truth)\b", re.I)

    for s in reversed(segments):
        if start - 8.0 <= s["start"] <= start + 2.0 and hook_re.search(s["text"]):
            start = max(0.0, s["start"] - 1.0)
            break

    best_end = min(float(duration), start + max_len, max(end, start + min_len))
    for s in segments:
        if s["end"] < start + min_len:
            continue
        if s["end"] - start > max_len:
            break
        if re.search(r"[.!?]$", s["text"].strip()):
            best_end = min(float(duration), s["end"] + 0.8)
            break
        best_end = min(float(duration), s["end"] + 0.5)

    if best_end - start < min_len:
        best_end = min(float(duration), start + min_len)
    if best_end - start > max_len:
        best_end = start + max_len

    return round(start, 2), round(best_end, 2)

def _parse_ai_clips(content, num_clips, min_len, max_len, duration, segments):
    match = re.search(r"\[[\s\S]*\]", content)
    if not match:
        return []
    items = json.loads(match.group(0))
    results = []
    for item in items:
        st, en = _snap_to_hook_and_payoff(
            segments, item.get("start", 0), item.get("end", 0), min_len, max_len, duration
        )
        if en - st >= min(min_len, 15):
            results.append({
                "start": st,
                "end": en,
                "duration": round(en - st, 1),
                "title": str(item.get("title", "Hook + Payoff Clip"))[:80],
                "reason": str(item.get("reason", "Strong hook and complete payoff"))[:140],
            })
    return results[:num_clips]

def select_clips(segments, audio_scores, chat_scores, motion_scores, duration, num_clips, min_len, max_len, provider, ollama_model):
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()

    prompt = None
    if segments:
        transcript_lines = "\n".join(
            f"[{int(s['start'])}s-{int(s['end'])}s] {s['text']}" for s in segments[:650]
        )
        prompt = (
            f"You are an expert short-form video editor. Pick up to {num_clips} viral-worthy clips from this stream transcript.\n"
            f"Strict rules:\n"
            f"1. Every clip MUST open with a clear HOOK (curiosity, shock, bold statement, question, or intense moment) and stay through the PAYOFF/ENDING (punchline, result, or resolution—never cut mid-thought).\n"
            f"2. Each clip duration MUST be between {min_len} and {max_len} seconds (NEVER exceed {max_len}s, and avoid short fragmented cuts).\n"
            f"3. Return ONLY a valid JSON array: [{{\"start\": 120, \"end\": 164, \"title\": \"Short title\", \"reason\": \"Why hook + ending works\"}}].\n\n"
            f"Transcript:\n{transcript_lines}"
        )

    order = []
    if provider == "auto":
        if groq_key:
            order.append("groq")
        if gemini_key:
            order.append("gemini")
        order.append("ollama")
    elif provider != "heuristic":
        order.append(provider)

    if prompt:
        for p in order:
            try:
                if p == "groq":
                    if not groq_key:
                        continue
                    print("  -> Selecting hook+payoff clips via Groq free API (llama-3.3-70b-versatile)...")
                    r = requests.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
                        json={
                            "model": "llama-3.3-70b-versatile",
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0.35,
                        },
                        timeout=60,
                    )
                    r.raise_for_status()
                    txt = r.json()["choices"][0]["message"]["content"]
                elif p == "gemini":
                    if not gemini_key:
                        continue
                    print("  -> Selecting hook+payoff clips via Google AI Studio free API (gemini-3.8-flash)...")
                    r = requests.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent?key={gemini_key}",
                        json={"contents": [{"parts": [{"text": prompt}]}]},
                        timeout=60,
                    )
                    r.raise_for_status()
                    txt = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                elif p == "ollama":
                    print(f"  -> Selecting hook+payoff clips via Ollama ({ollama_model})...")
                    r = requests.post(
                        "http://127.0.0.1:11434/api/generate",
                        json={"model": ollama_model, "prompt": prompt, "stream": False, "options": {"temperature": 0.35}},
                        timeout=240,
                    )
                    r.raise_for_status()
                    txt = r.json().get("response", "")
                else:
                    continue

                parsed = _parse_ai_clips(txt, num_clips, min_len, max_len, duration, segments)
                if parsed:
                    return parsed
            except Exception as e:
                print(f"  -> {p} skipped ({e})")

    candidates = []
    if segments:
        hook_re = re.compile(r"\b(wait|what|why|how|no way|bro|crazy|insane|look|watch|secret|never|caught|mistake|truth)\b", re.I)
        for seg in segments:
            b_idx = int(seg["start"] // 10)
            signal_boost = (
                (0.45 * min(100.0, chat_scores.get(b_idx, 0) * 5.0))
                + (0.35 * min(100.0, motion_scores.get(b_idx, 0) * 4.0))
                + (0.20 * min(100.0, audio_scores.get(b_idx, 0)))
            )
            hook_score = 35.0 if hook_re.search(seg["text"]) else 5.0
            st, en = _snap_to_hook_and_payoff(segments, seg["start"] - 1.0, seg["start"] + max_len, min_len, max_len, duration)
            candidates.append((hook_score + signal_boost, st, en, seg["text"][:60]))
    else:
        combined = defaultdict(float)
        for b, v in audio_scores.items():
            combined[b] += v * 0.45
        for b, v in motion_scores.items():
            combined[b] += v * 0.35
        for b, v in chat_scores.items():
            combined[b] += v * 0.20
        for b, sc in sorted(combined.items(), key=lambda x: x[1], reverse=True):
            st, en = _snap_to_hook_and_payoff([], max(0.0, (b * 10) - 5.0), (b * 10) + max_len, min_len, max_len, duration)
            candidates.append((sc, st, en, f"Peak moment at {int(st)}s"))

    candidates.sort(key=lambda x: x[0], reverse=True)
    picked = []
    for sc, st, en, label in candidates:
        if any(abs(st - p["start"]) < max_len for p in picked):
            continue
        picked.append({
            "start": st,
            "end": en,
            "duration": round(en - st, 1),
            "title": label,
            "reason": f"Hook + payoff window (score {sc:.1f})",
        })
        if len(picked) >= num_clips:
            break
    picked.sort(key=lambda x: x["start"])
    return picked

def _fmt_ass_time(sec):
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h}:{m:02d}:{s:05.2f}"

def _escape_ass_text(txt):
    txt = txt.replace("{", "(").replace("}", ")")
    return re.sub(r"\s+", " ", txt).strip()

def build_ass_subtitles(segments, clip_start, clip_end, ass_path, style_mode):
    """Generates a styled .ass subtitle file aligned to [clip_start, clip_end]."""
    clip_segs = [
        s for s in (segments or [])
        if s["end"] > clip_start and s["start"] < clip_end and s["text"].strip()
    ]
    if not clip_segs:
        return None

    if style_mode == "bold_shorts":
        # Centered bold Shorts captions (Alignment=5 = middle-center, above bottom UI)
        font_name, font_size, bold, alignment, margin_v = "Arial", 64, -1, 5, 120
        primary, outline, back = "&H0000FFFF", "&H00000000", "&H80000000"
    else:
        # Clean lower-third subtitles (Alignment=2 = bottom-center)
        font_name, font_size, bold, alignment, margin_v = "Arial", 48, 0, 2, 180
        primary, outline, back = "&H00FFFFFF", "&H00000000", "&H64000000"

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1080\n"
        "PlayResY: 1920\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{font_name},{font_size},{primary},&H000000FF,{outline},{back},"
        f"{bold},0,0,0,100,100,0,0,1,3,1,{alignment},60,60,{margin_v},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    events = []
    for s in clip_segs:
        rel_st = max(0.0, s["start"] - clip_start)
        rel_en = min(clip_end - clip_start, s["end"] - clip_start)
        if rel_en - rel_st < 0.25:
            continue
        words = _escape_ass_text(s["text"]).split()
        # Chunk into short readable phrases (up to 5 words)
        chunk_size = 4 if style_mode == "bold_shorts" else 6
        chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
        chunk_dur = (rel_en - rel_st) / max(1, len(chunks))
        for idx_c, ch in enumerate(chunks):
            c_st = rel_st + (idx_c * chunk_dur)
            c_en = min(rel_en, c_st + chunk_dur)
            events.append(f"Dialogue: 0,{_fmt_ass_time(c_st)},{_fmt_ass_time(c_en)},Default,,0,0,0,,{ch}")

    if not events:
        return None

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(events) + "\n")
    return ass_path

def render_clip(url, idx, start, end, raw_dir, shorts_dir, segments=None, subtitles="none", cookies=None, split_cam=False):
    raw_path = os.path.join(raw_dir, f"raw_{idx:03d}.mp4")
    final_path = os.path.join(shorts_dir, f"short_{idx:03d}.mp4")
    ass_path = os.path.join(raw_dir, f"sub_{idx:03d}.ass")

    if os.path.exists(final_path) and os.path.getsize(final_path) > 0:
        return final_path

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
        return None

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

    if subtitles != "none":
        built_ass = build_ass_subtitles(segments, start, end, ass_path, subtitles)
        if built_ass:
            escaped_path = os.path.abspath(built_ass).replace("\\", "/").replace(":", "\\:")
            vf = f"{vf},ass='{escaped_path}'"

    for encoder, extra in [
        ("h264_nvenc", ["-preset", "p4", "-cq", "23"]),
        ("h264_qsv", ["-b:v", "5M"]),
        ("libx264", ["-preset", "ultrafast", "-crf", "23"]),
    ]:
        ff_cmd = ["ffmpeg", "-y", "-i", raw_path, "-vf", vf, "-c:v", encoder, *extra, "-c:a", "copy", final_path]
        r = subprocess.run(ff_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode == 0 and os.path.exists(final_path) and os.path.getsize(final_path) > 0:
            return final_path
    return None

def write_reports(session_dir, title, url, rows):
    trans_dir = os.path.join(session_dir, "transcript")
    rep_dir = os.path.join(session_dir, "report")
    os.makedirs(trans_dir, exist_ok=True)
    os.makedirs(rep_dir, exist_ok=True)

    json_path = os.path.join(trans_dir, "clips.json")
    csv_path = os.path.join(trans_dir, "clips.csv")
    html_path = os.path.join(rep_dir, "index.html")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["clip", "start", "end", "duration", "title", "reason", "file"])
        writer.writeheader()
        writer.writerows(rows)

    trs = "\n".join(
        f"<tr><td>{r['clip']}</td><td>{r['start']}s - {r['end']}s</td><td>{r['duration']}s</td><td>{r['title']}</td><td>{r['reason']}</td><td>{r['file']}</td></tr>"
        for r in rows
    )
    html = (
        f"<!doctype html><html><head><meta charset='utf-8'><title>StreamClipper Report - {title}</title>"
        f"<style>body{{font-family:sans-serif;background:#0f1115;color:#eee;padding:24px}}table{{width:100%;border-collapse:collapse}}th,td{{border:1px solid #2b303c;padding:8px;text-align:left}}th{{background:#1b202b}}</style></head>"
        f"<body><h1>{title}</h1><p>Source: {url}</p><table><thead><tr><th>#</th><th>Range</th><th>Duration</th><th>Hook/Title</th><th>Payoff Reason</th><th>Output File</th></tr></thead><tbody>{trs}</tbody></table></body></html>"
    )
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

def main():
    parser = argparse.ArgumentParser(description="StreamClipper Hook + Payoff Vertical Shorts Generator (Max 50s)")
    parser.add_argument("url", help="YouTube, Twitch, or Kick stream/VOD URL")
    parser.add_argument("--clips", type=int, default=40)
    parser.add_argument("--min-len", type=int, default=25)
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--out", default="StreamClipper")
    parser.add_argument("--cookies", default=None)
    parser.add_argument("--chat-box", choices=["left", "right", "none"], default="right")
    parser.add_argument("--split-cam", action="store_true")
    parser.add_argument("--subtitles", choices=["none", "clean", "bold_shorts"], default="bold_shorts")
    parser.add_argument("--ai-provider", choices=["auto", "groq", "gemini", "ollama", "heuristic"], default="auto")
    parser.add_argument("--ollama-model", default="qwen3.5:9b")
    args = parser.parse_args()

    max_len = min(args.max_len, 50)
    min_len = min(args.min_len, max_len)

    meta = get_metadata(args.url, args.cookies)
    title = meta.get("title") or "Stream"
    duration = float(meta.get("duration") or 36000)
    session_id = hashlib.md5(f"{args.url}|{min_len}|{max_len}|{args.subtitles}".encode()).hexdigest()[:10]
    session_dir = os.path.join(args.out, session_id)

    raw_dir = os.path.join(session_dir, "raw")
    shorts_dir = os.path.join(session_dir, "shorts")
    trans_dir = os.path.join(session_dir, "transcript")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(shorts_dir, exist_ok=True)
    os.makedirs(trans_dir, exist_ok=True)

    print(f"Stream: {title} ({int(duration // 60)}m) | Subtitles: {args.subtitles} | Length: {min_len}s-{max_len}s")

    audio_path = download_audio(args.url, os.path.join(raw_dir, "source_audio.m4a"), args.cookies)
    cache_json = os.path.join(trans_dir, "transcript.json")
    segments = transcribe_segments(audio_path, cache_json) if audio_path else []
    audio_scores = score_audio_variation(audio_path)
    chat_scores, motion_scores = score_chat_and_motion(args.url, duration, args.chat_box, args.cookies)

    print("[4/4] Selecting hook-to-payoff clips (<= 50s)...")
    selected = select_clips(
        segments, audio_scores, chat_scores, motion_scores, duration,
        args.clips, min_len, max_len, args.ai_provider, args.ollama_model
    )

    rows = []
    for i, c in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] {c['start']}s -> {c['end']}s ({c['duration']}s) | {c['title']}")
        out_mp4 = render_clip(
            args.url, i, c["start"], c["end"], raw_dir, shorts_dir,
            segments=segments, subtitles=args.subtitles, cookies=args.cookies, split_cam=args.split_cam
        )
        rows.append({
            "clip": i,
            "start": c["start"],
            "end": c["end"],
            "duration": c["duration"],
            "title": c["title"],
            "reason": c["reason"],
            "file": os.path.basename(out_mp4) if out_mp4 else "FAILED",
        })

    write_reports(session_dir, title, args.url, rows)
    print(f"\nDONE! Clips & report saved in: {session_dir}")

if __name__ == "__main__":
    main()
