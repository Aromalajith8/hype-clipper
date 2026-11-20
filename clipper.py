import argparse, csv, hashlib, json, os, re, shutil, subprocess, time
from collections import defaultdict
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

def download_source_once(url, dest_mp4, cookies=None, max_height=1080):
    if os.path.exists(dest_mp4) and os.path.getsize(dest_mp4) > 1024 * 1024:
        print(f"[1/4] Using cached local source video: {dest_mp4}")
        return dest_mp4
    print(f"[1/4] Downloading source video once (up to {max_height}p) into fast local storage...")
    fmt = f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/best[height<={max_height}][ext=mp4]/best"
    cmd = [py_cmd(), "-m", "yt_dlp", "-f", fmt, "--merge-output-format", "mp4", "-o", dest_mp4]
    if cookies and os.path.exists(cookies):
        cmd += ["--cookies", cookies]
    cmd.append(url)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not os.path.exists(dest_mp4):
        raise RuntimeError(f"Failed to download video:\n{res.stderr}")
    return dest_mp4

def extract_audio_wav(video_path, wav_path):
    if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
        return wav_path
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return wav_path if os.path.exists(wav_path) else None

def transcribe_segments(wav_path, cache_json):
    if os.path.exists(cache_json):
        try:
            with open(cache_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data:
                print(f"[2/4] Loaded {len(data)} cached transcript segments.")
                return data
        except Exception:
            pass

    print("[2/4] Transcribing speech with faster-whisper...")
    try:
        from faster_whisper import WhisperModel
        has_gpu = subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        device = "cuda" if has_gpu else "cpu"
        compute = "float16" if has_gpu else "int8"
        model_name = "small" if has_gpu else "base"
        model = WhisperModel(model_name, device=device, compute_type=compute)
        segs, _ = model.transcribe(wav_path, beam_size=1, vad_filter=True)
        out = [{"start": float(s.start), "end": float(s.end), "text": s.text.strip()} for s in segs if s.text.strip()]
        with open(cache_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"  -> Transcribed {len(out)} segments ({device.upper()}, {model_name}).")
        return out
    except Exception as e:
        print(f"  -> Whisper skipped ({e}); using heuristic windows.")
        return []

def score_audio_peaks(wav_path, bucket_sec=10):
    deltas = defaultdict(float)
    if not wav_path or not os.path.exists(wav_path):
        return deltas
    cmd = [
        "ffmpeg", "-i", wav_path,
        "-af", "astats=metadata=1:reset=100,ametadata=print:key=lavfi.astats.Overall.RMS_level",
        "-f", "null", "-"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    times = re.findall(r"pts_time:([0-9.]+)", res.stderr)
    levels = re.findall(r"RMS_level=([-0-9.]+)", res.stderr)
    buckets = defaultdict(list)
    for t, lvl in zip(times, levels):
        try:
            v = float(lvl)
            if v > -60.0:
                buckets[int(float(t) // bucket_sec)].append(v)
        except ValueError:
            pass
    ordered = sorted(buckets.keys())
    for i, b in enumerate(ordered):
        mx = max(buckets[b])
        prev = sum(buckets[ordered[i - 1]]) / len(buckets[ordered[i - 1]]) if i > 0 else mx
        deltas[b] = max(0.0, mx - prev) * 2.0 + max(0.0, mx + 30.0)
    return deltas

def snap_hook_payoff(segments, start_s, end_s, min_len, max_len, duration):
    start = max(0.0, float(start_s))
    end = min(float(duration), float(end_s))
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

def parse_ai_json(text, num_clips, min_len, max_len, duration, segments):
    m = re.search(r"\[[\s\S]*\]", text or "")
    if not m:
        return []
    items = json.loads(m.group(0))
    out = []
    for item in items:
        st, en = snap_hook_payoff(segments, item.get("start", 0), item.get("end", 0), min_len, max_len, duration)
        if en - st >= min(min_len, 15):
            out.append({
                "start": st,
                "end": en,
                "duration": round(en - st, 1),
                "title": str(item.get("title", "Hook + Payoff Clip"))[:80],
                "reason": str(item.get("reason", "Hook + complete payoff"))[:140],
            })
    return out[:num_clips]

def select_clips(segments, audio_scores, duration, num_clips, min_len, max_len, provider, ollama_model):
    print("[3/4] Selecting hook-to-payoff segments (<= 50s)...")
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()

    if segments:
        lines = "\n".join(f"[{int(s['start'])}s-{int(s['end'])}s] {s['text']}" for s in segments[:600])
        prompt = (
            f"Select up to {num_clips} engaging Short-form clips from this transcript.\n"
            f"Rules:\n"
            f"1. Start on a strong HOOK and end on a complete PAYOFF/PUNCHLINE/ENDING (never cut mid-thought).\n"
            f"2. Duration MUST be between {min_len} and {max_len} seconds (strictly <= {max_len}s).\n"
            f"3. Return ONLY a JSON list: [{{\"start\": 120, \"end\": 162, \"title\": \"Title\", \"reason\": \"Hook + payoff\"}}].\n\n"
            f"Transcript:\n{lines}"
        )

        order = []
        if provider == "auto":
            if groq_key:
                order.append("groq")
            if gemini_key:
                order.append("gemini")
            order.append("heuristic")
        elif provider != "heuristic":
            order.append(provider)

        for p in order:
            try:
                if p == "groq" and groq_key:
                    r = requests.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
                        json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "temperature": 0.35},
                        timeout=45,
                    )
                    r.raise_for_status()
                    parsed = parse_ai_json(r.json()["choices"][0]["message"]["content"], num_clips, min_len, max_len, duration, segments)
                    if parsed:
                        return parsed
                elif p == "gemini" and gemini_key:
                    r = requests.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}",
                        json={"contents": [{"parts": [{"text": prompt}]}]},
                        timeout=45,
                    )
                    r.raise_for_status()
                    parsed = parse_ai_json(r.json()["candidates"][0]["content"]["parts"][0]["text"], num_clips, min_len, max_len, duration, segments)
                    if parsed:
                        return parsed
                elif p == "ollama":
                    r = requests.post(
                        "http://127.0.0.1:11434/api/generate",
                        json={"model": ollama_model, "prompt": prompt, "stream": False, "options": {"temperature": 0.35}},
                        timeout=120,
                    )
                    r.raise_for_status()
                    parsed = parse_ai_json(r.json().get("response", ""), num_clips, min_len, max_len, duration, segments)
                    if parsed:
                        return parsed
            except Exception as e:
                print(f"  -> {p} skipped ({e}); using fast hook+payoff scorer.")

    # Fast local hook + payoff + audio scorer
    hook_re = re.compile(r"\b(wait|what|why|how|no way|bro|crazy|insane|look|watch|secret|never|caught|mistake|truth)\b", re.I)
    scored = []
    if segments:
        for seg in segments:
            b = int(seg["start"] // 10)
            sc = (30.0 if hook_re.search(seg["text"]) else 5.0) + min(50.0, audio_scores.get(b, 0.0))
            st, en = snap_hook_payoff(segments, seg["start"] - 1.0, seg["start"] + max_len, min_len, max_len, duration)
            scored.append((sc, st, en, seg["text"][:60]))
    else:
        for b, sc in sorted(audio_scores.items(), key=lambda x: x[1], reverse=True):
            st, en = snap_hook_payoff([], max(0.0, b * 10 - 5.0), b * 10 + max_len, min_len, max_len, duration)
            scored.append((sc, st, en, f"High-energy moment at {int(st)}s"))

    scored.sort(key=lambda x: x[0], reverse=True)
    picked = []
    for sc, st, en, title in scored:
        if any(abs(st - p["start"]) < max_len for p in picked):
            continue
        picked.append({"start": st, "end": en, "duration": round(en - st, 1), "title": title, "reason": f"Hook+payoff score {sc:.1f}"})
        if len(picked) >= num_clips:
            break
    picked.sort(key=lambda x: x["start"])
    return picked

def _fmt_ass(sec):
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h}:{m:02d}:{s:05.2f}"

def build_ass(segments, start_s, end_s, ass_path, style_mode):
    segs = [s for s in (segments or []) if s["end"] > start_s and s["start"] < end_s and s["text"].strip()]
    if not segs or style_mode == "none":
        return None
    if style_mode == "bold_shorts":
        font_sz, bold, align, margin_v, color = 62, -1, 5, 120, "&H0000FFFF"
    else:
        font_sz, bold, align, margin_v, color = 46, 0, 2, 170, "&H00FFFFFF"

    header = (
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n\n"
        "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,Arial,{font_sz},{color},&H000000FF,&H00000000,&H80000000,{bold},0,0,0,100,100,0,0,1,3,1,{align},60,60,{margin_v},1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    lines = []
    for s in segs:
        rel_s = max(0.0, s["start"] - start_s)
        rel_e = min(end_s - start_s, s["end"] - start_s)
        if rel_e - rel_s < 0.25:
            continue
        words = re.sub(r"\s+", " ", s["text"].replace("{", "(").replace("}", ")")).strip().split()
        step = 4 if style_mode == "bold_shorts" else 6
        chunks = [" ".join(words[i:i + step]) for i in range(0, len(words), step)]
        dur = (rel_e - rel_s) / max(1, len(chunks))
        for idx_c, ch in enumerate(chunks):
            cs = rel_s + idx_c * dur
            ce = min(rel_e, cs + dur)
            lines.append(f"Dialogue: 0,{_fmt_ass(cs)},{_fmt_ass(ce)},Default,,0,0,0,,{ch}")
    if not lines:
        return None
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(lines) + "\n")
    return ass_path

def detect_fast_encoder():
    res = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True)
    enc_out = res.stdout
    if "h264_nvenc" in enc_out and subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        return ("h264_nvenc", ["-preset", "p2", "-cq", "24"])
    if "h264_qsv" in enc_out:
        return ("h264_qsv", ["-preset", "veryfast", "-b:v", "4.5M"])
    return ("libx264", ["-preset", "veryfast", "-crf", "24"])

def fast_render_clip(src_mp4, idx, start_s, end_s, tmp_dir, final_dir, encoder_cfg, segments=None, subtitles="none", split_cam=False):
    os.makedirs(final_dir, exist_ok=True)
    out_mp4 = os.path.join(final_dir, f"short_{idx:03d}.mp4")
    if os.path.exists(out_mp4) and os.path.getsize(out_mp4) > 0:
        return out_mp4

    dur = max(1.0, float(end_s) - float(start_s))
    if split_cam:
        vf = (
            "split[cam][game];"
            "[cam]crop=iw*0.28:ih*0.28:0:0,scale=1080:670:force_original_aspect_ratio=increase,crop=1080:670[top];"
            "[game]scale=1080:1250:force_original_aspect_ratio=increase,crop=1080:1250[bot];"
            "[top][bot]vstack"
        )
    else:
        # Fast vertical framing: scaled foreground centered over dark/dimmed backdrop
        vf = (
            "split[orig][bg];"
            "[bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,eq=brightness=-0.25[bgd];"
            "[orig]scale=1080:-2[fg];"
            "[bgd][fg]overlay=(W-w)/2:(H-h)/2"
        )

    if subtitles != "none":
        ass_file = os.path.join(tmp_dir, f"sub_{idx:03d}.ass")
        built = build_ass(segments, start_s, end_s, ass_file, subtitles)
        if built:
            esc = os.path.abspath(built).replace("\\", "/").replace(":", "\\:")
            vf = f"{vf},ass='{esc}'"

    enc_name, enc_args = encoder_cfg
    # Fast input seeking (-ss before -i) so FFmpeg jumps straight to start_s
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_s:.2f}", "-t", f"{dur:.2f}", "-i", src_mp4,
        "-vf", vf,
        "-c:v", enc_name, *enc_args,
        "-c:a", "aac", "-b:a", "128k",
        out_mp4
    ]
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if r.returncode != 0 and enc_name != "libx264":
        fallback_cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start_s:.2f}", "-t", f"{dur:.2f}", "-i", src_mp4,
            "-vf", vf,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
            "-c:a", "aac", "-b:a", "128k",
            out_mp4
        ]
        subprocess.run(fallback_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return out_mp4 if (os.path.exists(out_mp4) and os.path.getsize(out_mp4) > 0) else None

def write_reports(session_dir, title, url, rows):
    trans_dir = os.path.join(session_dir, "transcript")
    rep_dir = os.path.join(session_dir, "report")
    os.makedirs(trans_dir, exist_ok=True)
    os.makedirs(rep_dir, exist_ok=True)

    with open(os.path.join(trans_dir, "clips.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    with open(os.path.join(trans_dir, "clips.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "start", "end", "duration", "title", "reason", "file"])
        w.writeheader()
        w.writerows(rows)

    trs = "\n".join(
        f"<tr><td>{r['clip']}</td><td>{r['start']}s - {r['end']}s</td><td>{r['duration']}s</td><td>{r['title']}</td><td>{r['reason']}</td><td>{r['file']}</td></tr>"
        for r in rows
    )
    html = (
        f"<!doctype html><html><head><meta charset='utf-8'><title>StreamClipper - {title}</title>"
        f"<style>body{{font-family:sans-serif;background:#0f1115;color:#eee;padding:24px}}table{{width:100%;border-collapse:collapse}}th,td{{border:1px solid #2b303c;padding:8px;text-align:left}}th{{background:#1b202b}}</style></head>"
        f"<body><h1>{title}</h1><p>Source: {url}</p><table><thead><tr><th>#</th><th>Range</th><th>Duration</th><th>Hook/Title</th><th>Payoff Reason</th><th>File</th></tr></thead><tbody>{trs}</tbody></table></body></html>"
    )
    with open(os.path.join(rep_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

def main():
    t0 = time.time()
    parser = argparse.ArgumentParser(description="Fast OpusClip-Style Hook + Payoff Shorts Clipper (Max 50s)")
    parser.add_argument("url")
    parser.add_argument("--clips", type=int, default=30)
    parser.add_argument("--min-len", type=int, default=25)
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--out", default="StreamClipper")
    parser.add_argument("--work-dir", default="/tmp/streamclipper_fast" if os.name != "nt" else r"D:\hype-clipper\.fast_work")
    parser.add_argument("--max-height", type=int, default=1080)
    parser.add_argument("--cookies", default=None)
    parser.add_argument("--split-cam", action="store_true")
    parser.add_argument("--subtitles", choices=["none", "clean", "bold_shorts"], default="bold_shorts")
    parser.add_argument("--ai-provider", choices=["auto", "groq", "gemini", "ollama", "heuristic"], default="auto")
    parser.add_argument("--ollama-model", default="qwen3.5:9b")
    args = parser.parse_args()

    max_len = min(args.max_len, 50)
    min_len = min(args.min_len, max_len)

    meta = get_metadata(args.url, args.cookies)
    title = meta.get("title") or "Stream"
    duration = float(meta.get("duration") or 3600)
    sid = hashlib.md5(f"{args.url}|{min_len}|{max_len}|{args.subtitles}".encode()).hexdigest()[:10]

    local_work = os.path.join(args.work_dir, sid)
    local_shorts = os.path.join(local_work, "shorts")
    os.makedirs(local_work, exist_ok=True)
    os.makedirs(local_shorts, exist_ok=True)

    final_session_dir = os.path.join(args.out, sid)
    final_shorts_dir = os.path.join(final_session_dir, "shorts")
    os.makedirs(final_shorts_dir, exist_ok=True)

    src_mp4 = download_source_once(args.url, os.path.join(local_work, "source.mp4"), args.cookies, args.max_height)
    wav_path = extract_audio_wav(src_mp4, os.path.join(local_work, "audio.wav"))
    segments = transcribe_segments(wav_path, os.path.join(local_work, "transcript.json")) if wav_path else []
    audio_scores = score_audio_peaks(wav_path)

    selected = select_clips(
        segments, audio_scores, duration, args.clips, min_len, max_len, args.ai_provider, args.ollama_model
    )

    enc_cfg = detect_fast_encoder()
    print(f"[4/4] Fast-rendering {len(selected)} clips ({min_len}s-{max_len}s) using {enc_cfg[0]}...")

    rows = []
    for i, c in enumerate(selected, 1):
        c_start = time.time()
        out_local = fast_render_clip(
            src_mp4, i, c["start"], c["end"], local_work, local_shorts,
            enc_cfg, segments=segments, subtitles=args.subtitles, split_cam=args.split_cam
        )
        fname = f"short_{i:03d}.mp4"
        dest_file = os.path.join(final_shorts_dir, fname)
        if out_local and os.path.exists(out_local):
            shutil.copy2(out_local, dest_file)
            status_file = fname
        else:
            status_file = "FAILED"

        print(f"  [{i}/{len(selected)}] {c['start']}s->{c['end']}s ({c['duration']}s) in {time.time() - c_start:.1f}s | {c['title']}")
        rows.append({
            "clip": i,
            "start": c["start"],
            "end": c["end"],
            "duration": c["duration"],
            "title": c["title"],
            "reason": c["reason"],
            "file": status_file,
        })

    write_reports(final_session_dir, title, args.url, rows)
    print(f"\nDONE in {time.time() - t0:.1f}s total! Output: {final_session_dir}")

if __name__ == "__main__":
    main()
