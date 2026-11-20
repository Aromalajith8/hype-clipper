import argparse, csv, hashlib, json, os, re, shutil, subprocess, sys, time, wave
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import requests

CAPTION_STYLES = [
    "auto", "bold-pop", "beast-bold", "karaoke-highlight",
    "word-by-word", "two-line-stack", "highlight-box",
    "neon-glow", "impact-shake", "emoji-pop",
    "typewriter", "bounce", "gradient-stroke",
    "comic-burst", "minimal-clean", "gaming-glitch", "none"
]
HOOK_STYLES = ["bold-box", "outline-glow", "top-bar", "none"]

def py_cmd():
    return "py" if os.name == "nt" else "python3"

def get_metadata(url, cookies=None):
    if os.path.isfile(url):
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", url],
            capture_output=True, text=True
        )
        dur = float(probe.stdout.strip()) if probe.returncode == 0 and probe.stdout.strip() else 3600.0
        return {"title": os.path.splitext(os.path.basename(url))[0], "duration": dur}

    cmd = [py_cmd(), "-m", "yt_dlp", "--dump-single-json", "--skip-download"]
    if cookies and os.path.exists(cookies):
        cmd += ["--cookies", cookies]
    cmd.append(url)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"yt-dlp metadata failed:\n{res.stderr[-1200:]}")
    return json.loads(res.stdout)

def download_source_1080p(url, dest_mp4, cookies=None, min_height=1080):
    if os.path.isfile(url):
        print(f"[1/4] Using local/Drive video: {url}")
        return url
    if os.path.exists(dest_mp4) and os.path.getsize(dest_mp4) > 1024 * 1024:
        print(f"[1/4] Reusing cached source video: {dest_mp4}")
        return dest_mp4

    print(f"[1/4] Downloading source (>= {min_height}p) with 12 parallel fragments...")
    fmt = (
        f"bestvideo[height>={min_height}][ext=mp4]+bestaudio[ext=m4a]/"
        f"bestvideo[height>={min_height}]+bestaudio/"
        f"bestvideo[ext=mp4]+bestaudio[ext=m4a]/best"
    )
    cmd = [
        py_cmd(), "-m", "yt_dlp",
        "-N", "12",
        "--concurrent-fragments", "12",
        "-f", fmt,
        "--merge-output-format", "mp4",
        "-o", dest_mp4
    ]
    if cookies and os.path.exists(cookies):
        cmd += ["--cookies", cookies]
    cmd.append(url)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not os.path.exists(dest_mp4):
        raise RuntimeError(f"Download failed:\n{res.stderr[-1500:]}")
    return dest_mp4

def extract_audio_wav(video_path, wav_path):
    if os.path.exists(wav_path) and os.path.getsize(wav_path) > 1024:
        return wav_path
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav_path]
    subprocess.run(cmd, capture_output=True, text=True)
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
            json.dump(out, f)
        print(f"  -> Transcribed {len(out)} segments on {device.upper()} ({model_name}).")
        return out
    except Exception as e:
        print(f"  -> Whisper skipped ({e}); falling back to audio peaks.")
        return []

def score_audio_peaks_fast(wav_path, bucket_sec=10):
    deltas = defaultdict(float)
    if not wav_path or not os.path.exists(wav_path):
        return deltas
    try:
        with wave.open(wav_path, "rb") as wf:
            sr = wf.getframerate()
            chunk_frames = sr * bucket_sec
            rms_vals = []
            while True:
                raw = wf.readframes(chunk_frames)
                if not raw:
                    break
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                if samples.size == 0:
                    continue
                db = 20.0 * np.log10(float(np.sqrt(np.mean(samples * samples)) + 1e-8))
                rms_vals.append(db)
        for i, db in enumerate(rms_vals):
            prev = rms_vals[i - 1] if i > 0 else db
            deltas[i] = max(0.0, db - prev) * 2.5 + max(0.0, db + 35.0)
    except Exception as e:
        print(f"  -> Audio scoring warning: {e}")
    return deltas

def snap_hook_payoff(segments, start_s, end_s, min_len, max_len, duration):
    """Guarantees min_len <= duration <= max_len, starts on hook/setup, and holds through payoff/ending."""
    start = max(0.0, float(start_s))
    end = min(float(duration), float(end_s))
    if end - start < min_len:
        end = min(float(duration), start + min_len)
    if end - start > max_len:
        end = start + max_len

    if not segments:
        return round(start, 2), round(min(float(duration), end), 2)

    hook_re = re.compile(r"\b(wait|what|why|how|no way|bro|crazy|insane|look|watch|secret|never|caught|mistake|truth|stop|listen|holy|check this)\b", re.I)
    for s in reversed(segments):
        if start - 8.0 <= s["start"] <= start + 2.0 and hook_re.search(s["text"]):
            start = max(0.0, s["start"] - 1.0)
            break

    # Hold through complete sentences/payoff up to max_len (never cut off early like a 13s fragment)
    best_end = min(float(duration), start + max_len)
    sentence_ends = [s["end"] + 0.7 for s in segments if s["end"] >= start + min_len and s["end"] - start <= max_len and re.search(r"[.!?]$", s["text"].strip())]
    if sentence_ends:
        best_end = min(float(duration), max(sentence_ends))
    else:
        seg_ends = [s["end"] + 0.4 for s in segments if s["end"] >= start + min_len and s["end"] - start <= max_len]
        if seg_ends:
            best_end = min(float(duration), max(seg_ends))

    if best_end - start < min_len:
        best_end = min(float(duration), start + min_len)
    if best_end - start > max_len:
        best_end = start + max_len

    return round(start, 2), round(best_end, 2)

def _parse_ai_json(text):
    m = re.search(r"\[[\s\S]*\]", text or "")
    if not m:
        return []
    try:
        return json.loads(m.group(0))
    except Exception:
        return []

def _call_free_ai_chunk(prompt, provider, ollama_model):
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()

    order = []
    if provider == "auto":
        if groq_key:
            order.append("groq")
        if gemini_key:
            order.append("gemini")
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
                return r.json()["choices"][0]["message"]["content"]
            elif p == "gemini" and gemini_key:
                r = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}",
                    json={"contents": [{"parts": [{"text": prompt}]}]},
                    timeout=45,
                )
                r.raise_for_status()
                return r.json()["candidates"][0]["content"]["parts"][0]["text"]
            elif p == "ollama":
                r = requests.post(
                    "http://127.0.0.1:11434/api/generate",
                    json={"model": ollama_model, "prompt": prompt, "stream": False, "options": {"temperature": 0.35}},
                    timeout=120,
                )
                r.raise_for_status()
                return r.json().get("response", "")
        except Exception as e:
            print(f"  -> AI ({p}) skipped: {e}")
    return ""

def select_clips_full_stream(segments, audio_scores, duration, num_clips, min_len, max_len, provider, ollama_model):
    print(f"[3/4] Selecting {num_clips} hook-to-payoff clips ({min_len}s-{max_len}s) across full stream...")
    picked = []

    if segments and provider != "heuristic":
        chunk_size = 320
        num_chunks = max(1, (len(segments) + chunk_size - 1) // chunk_size)
        per_chunk = max(4, (num_clips + num_chunks - 1) // num_chunks + 3)

        for c_idx in range(num_chunks):
            chunk = segments[c_idx * chunk_size : (c_idx + 1) * chunk_size]
            if not chunk:
                continue
            lines = "\n".join(f"[{int(s['start'])}s-{int(s['end'])}s] {s['text']}" for s in chunk)
            prompt = (
                f"Pick up to {per_chunk} high-retention Short-form clips from this transcript chunk.\n"
                f"Rules:\n"
                f"1. Every clip MUST open with a strong HOOK and continue through a full PAYOFF/PUNCHLINE/ENDING.\n"
                f"2. Duration MUST be between {min_len} and {max_len} seconds (strictly <= {max_len}s).\n"
                f"3. Return ONLY JSON: [{{\"start\": 120, \"end\": 165, \"title\": \"Hook Title\", \"reason\": \"Hook + payoff\"}}].\n\n"
                f"Transcript:\n{lines}"
            )
            raw = _call_free_ai_chunk(prompt, provider, ollama_model)
            for item in _parse_ai_json(raw):
                st, en = snap_hook_payoff(segments, item.get("start", 0), item.get("end", 0), min_len, max_len, duration)
                if en - st >= min(min_len, 20) and not any(abs(st - p["start"]) < max_len * 0.8 for p in picked):
                    picked.append({
                        "start": st, "end": en, "duration": round(en - st, 1),
                        "title": str(item.get("title", "Hook + Payoff"))[:80],
                        "reason": str(item.get("reason", "AI hook + payoff"))[:140],
                    })
            if len(picked) >= num_clips:
                break

    if len(picked) < num_clips:
        hook_re = re.compile(r"\b(wait|what|why|how|no way|bro|crazy|insane|look|watch|secret|never|caught|mistake|truth|stop|listen|holy|check this)\b", re.I)
        scored = []
        if segments:
            for seg in segments:
                b = int(seg["start"] // 10)
                sc = (35.0 if hook_re.search(seg["text"]) else 5.0) + min(55.0, audio_scores.get(b, 0.0))
                st, en = snap_hook_payoff(segments, seg["start"] - 1.0, seg["start"] + max_len, min_len, max_len, duration)
                scored.append((sc, st, en, seg["text"][:60]))
        else:
            for b, sc in sorted(audio_scores.items(), key=lambda x: x[1], reverse=True):
                st, en = snap_hook_payoff([], max(0.0, b * 10 - 5.0), b * 10 + max_len, min_len, max_len, duration)
                scored.append((sc, st, en, f"Peak moment at {int(st)}s"))

        scored.sort(key=lambda x: x[0], reverse=True)
        for sc, st, en, title in scored:
            if any(abs(st - p["start"]) < max_len * 0.8 for p in picked):
                continue
            picked.append({
                "start": st, "end": en, "duration": round(en - st, 1),
                "title": title, "reason": f"Hook + energy score {sc:.1f}",
            })
            if len(picked) >= num_clips:
                break

    picked = picked[:num_clips]
    picked.sort(key=lambda x: x["start"])
    return picked

def _fmt_ass(sec):
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h}:{m:02d}:{s:05.2f}"

def _clean_text(txt):
    return re.sub(r"\s+", " ", txt.replace("{", "(").replace("}", ")")).strip()

def _style_params(style_name):
    table = {
        "bold-pop":          ("Arial", 66, "&H0000FFFF", "&H00000000", "&H80000000", -1, 1, 4, 2, 5, 110, 3, "phrase"),
        "beast-bold":        ("Impact", 74, "&H00FFFFFF", "&H00000000", "&H90000000", -1, 1, 5, 3, 5, 90,  2, "phrase"),
        "karaoke-highlight": ("Arial", 62, "&H0000FF88", "&H00000000", "&H80000000", -1, 1, 3, 1, 5, 120, 4, "phrase"),
        "word-by-word":      ("Impact", 82, "&H0000FFFF", "&H00000000", "&H00000000", -1, 1, 5, 2, 5, 100, 1, "word"),
        "two-line-stack":    ("Arial", 56, "&H00FFFFFF", "&H00000000", "&H88000000", -1, 1, 3, 2, 5, 120, 6, "phrase"),
        "highlight-box":     ("Arial", 58, "&H00000000", "&H0000FFFF", "&HFF00FFFF", -1, 3, 2, 0, 5, 120, 3, "phrase"),
        "neon-glow":         ("Arial", 64, "&H00FFFF00", "&H00FFAA00", "&H80000000", -1, 1, 4, 3, 5, 120, 3, "phrase"),
        "impact-shake":      ("Impact", 72, "&H003366FF", "&H00000000", "&H80000000", -1, 1, 4, 2, 5, 100, 2, "phrase"),
        "emoji-pop":         ("Arial", 62, "&H00FFFFFF", "&H00000000", "&H80000000", -1, 1, 3, 2, 5, 120, 3, "emoji"),
        "typewriter":        ("Courier New", 54, "&H00FFFFFF", "&H00000000", "&H88000000", -1, 1, 2, 1, 5, 130, 1, "word"),
        "bounce":            ("Arial", 66, "&H0066FFFF", "&H00000000", "&H80000000", -1, 1, 4, 2, 5, 110, 2, "phrase"),
        "gradient-stroke":   ("Arial", 66, "&H00FF66FF", "&H00FFCC00", "&H80000000", -1, 1, 4, 2, 5, 115, 3, "phrase"),
        "comic-burst":       ("Impact", 70, "&H003355FF", "&H0000FFFF", "&H80000000", -1, 1, 4, 2, 5, 115, 2, "phrase"),
        "minimal-clean":     ("Arial", 46, "&H00FFFFFF", "&H00000000", "&H64000000",  0, 1, 2, 1, 2, 180, 6, "phrase"),
        "gaming-glitch":     ("Courier New", 64, "&H00FFFF00", "&H00FF00FF", "&H80000000", -1, 1, 3, 2, 5, 115, 2, "phrase"),
    }
    return table.get(style_name, table["bold-pop"])

def build_styled_ass(segments, start_s, end_s, ass_path, caption_style, hook_style, hook_title):
    if caption_style == "none" and hook_style == "none":
        return None
    chosen = caption_style
    if chosen == "auto":
        pool = [s for s in CAPTION_STYLES if s not in ("auto", "none", "minimal-clean")]
        chosen = pool[int(start_s) % len(pool)]

    font, sz, pri, out_c, back_c, bold, bstyle, out_w, shd, align, mv, step, mode = _style_params(chosen)
    header = (
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n\n"
        "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{font},{sz},{pri},&H000000FF,{out_c},{back_c},{bold},0,0,0,100,100,0,0,{bstyle},{out_w},{shd},{align},60,60,{mv},1\n"
        f"Style: HookBanner,Arial,44,&H00000000,&H000000FF,&H0000FFFF,&HFF00FFFF,-1,0,0,0,100,100,0,0,3,6,0,8,50,50,150,1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    events = []
    if hook_style != "none" and hook_title:
        banner = _clean_text(hook_title).upper()[:42]
        if banner:
            events.append(f"Dialogue: 2,0:00:00.20,0:00:03.60,HookBanner,,0,0,0,,{banner}")

    if chosen != "none" and segments:
        clip_segs = [s for s in segments if s["end"] > start_s and s["start"] < end_s and s["text"].strip()]
        for s in clip_segs:
            rel_s = max(0.0, s["start"] - start_s)
            rel_e = min(end_s - start_s, s["end"] - start_s)
            if rel_e - rel_s < 0.2:
                continue
            words = _clean_text(s["text"]).split()
            if not words:
                continue
            chunks = words if mode == "word" else [" ".join(words[i:i + step]) for i in range(0, len(words), step)]
            dur = (rel_e - rel_s) / max(1, len(chunks))
            for idx_c, ch in enumerate(chunks):
                cs = rel_s + idx_c * dur
                ce = min(rel_e, cs + dur)
                label = f"{ch} 🔥" if (mode == "emoji" and idx_c % 2 == 0) else ch
                events.append(f"Dialogue: 1,{_fmt_ass(cs)},{_fmt_ass(ce)},Default,,0,0,0,,{label}")

    if not events:
        return None
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(events) + "\n")
    return ass_path

def detect_fast_1080p_encoder():
    res = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True)
    enc = res.stdout
    has_nv = subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    if "h264_nvenc" in enc and has_nv:
        return ("h264_nvenc", ["-preset", "p1", "-b:v", "8M", "-maxrate", "12M", "-bufsize", "16M"])
    if "h264_qsv" in enc:
        return ("h264_qsv", ["-preset", "veryfast", "-b:v", "8M", "-maxrate", "12M"])
    return ("libx264", ["-preset", "ultrafast", "-crf", "21"])

def _render_worker(args_tuple):
    src_mp4, idx, clip, tmp_dir, local_shorts, final_shorts, enc_name, enc_args, segments, caption_style, hook_style, split_cam = args_tuple
    t0 = time.time()
    fname = f"short_{idx:03d}.mp4"
    final_path = os.path.join(final_shorts, fname)
    if os.path.exists(final_path) and os.path.getsize(final_path) > 10240:
        return idx, fname, 0.0, None

    out_local = os.path.join(local_shorts, fname)
    start_s, end_s = clip["start"], clip["end"]
    dur = max(1.0, float(end_s) - float(start_s))

    if split_cam:
        vf = (
            "split[cam][game];"
            "[cam]crop=iw*0.28:ih*0.28:0:0,scale=1080:670:force_original_aspect_ratio=increase:flags=lanczos,crop=1080:670[top];"
            "[game]scale=1080:1250:force_original_aspect_ratio=increase:flags=lanczos,crop=1080:1250[bot];"
            "[top][bot]vstack"
        )
    else:
        vf = (
            "split[orig][bg];"
            "[bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,eq=brightness=-0.25[bgd];"
            "[orig]scale=1080:-2:flags=lanczos[fg];"
            "[bgd][fg]overlay=(W-w)/2:(H-h)/2"
        )

    ass_file = os.path.join(tmp_dir, f"sub_{idx:03d}.ass")
    built = build_styled_ass(segments, start_s, end_s, ass_file, caption_style, hook_style, clip["title"])
    if built:
        esc = os.path.abspath(built).replace("\\", "/").replace(":", "\\:")
        vf = f"{vf},ass='{esc}'"

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start_s:.2f}", "-t", f"{dur:.2f}", "-i", src_mp4,
        "-vf", vf,
        "-c:v", enc_name, *enc_args,
        "-c:a", "aac", "-b:a", "192k",
        out_local
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 and enc_name != "libx264":
        fb_cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start_s:.2f}", "-t", f"{dur:.2f}", "-i", src_mp4,
            "-vf", vf,
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "21",
            "-c:a", "aac", "-b:a", "192k",
            out_local
        ]
        r = subprocess.run(fb_cmd, capture_output=True, text=True)

    if os.path.exists(out_local) and os.path.getsize(out_local) > 10240:
        shutil.copy2(out_local, final_path)
        return idx, fname, time.time() - t0, None
    return idx, "FAILED", time.time() - t0, (r.stderr or "").strip()[-300:]

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
        f"<!doctype html><html><head><meta charset='utf-8'><title>HypeClipper 1080p - {title}</title>"
        f"<style>body{{font-family:sans-serif;background:#0f1115;color:#eee;padding:24px}}table{{width:100%;border-collapse:collapse}}th,td{{border:1px solid #2b303c;padding:8px;text-align:left}}th{{background:#1b202b}}</style></head>"
        f"<body><h1>{title}</h1><p>Source: {url}</p><table><thead><tr><th>#</th><th>Range</th><th>Duration</th><th>Hook/Title</th><th>Payoff Reason</th><th>File</th></tr></thead><tbody>{trs}</tbody></table></body></html>"
    )
    with open(os.path.join(rep_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

def main():
    t0 = time.time()
    parser = argparse.ArgumentParser(description="HypeClipper 1080p+ Hook & Payoff Shorts Engine (40+ clips, Multi-Process)")
    parser.add_argument("url")
    parser.add_argument("--clips", type=int, default=45)
    parser.add_argument("--min-len", type=int, default=25)
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--min-height", type=int, default=1080)
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 4))
    parser.add_argument("--out", default="HypeClipper")
    parser.add_argument("--work-dir", default="/tmp/hypeclipper_fast" if os.name != "nt" else r"D:\hype-clipper\.fast_work")
    parser.add_argument("--cookies", default=None)
    parser.add_argument("--split-cam", action="store_true")
    parser.add_argument("--caption-style", choices=CAPTION_STYLES, default="auto")
    parser.add_argument("--hook-style", choices=HOOK_STYLES, default="bold-box")
    parser.add_argument("--ai-provider", choices=["auto", "groq", "gemini", "ollama", "heuristic"], default="auto")
    parser.add_argument("--ollama-model", default="qwen3.5:9b")
    args = parser.parse_args()

    max_len = min(args.max_len, 50)
    min_len = min(args.min_len, max_len)
    min_height = max(1080, args.min_height)

    meta = get_metadata(args.url, args.cookies)
    title = meta.get("title") or "Stream"
    duration = float(meta.get("duration") or 3600)
    sid = hashlib.md5(f"{args.url}|{min_len}|{max_len}|{min_height}|{args.caption_style}|{args.hook_style}".encode()).hexdigest()[:10]

    local_work = os.path.join(args.work_dir, sid)
    local_shorts = os.path.join(local_work, "shorts")
    os.makedirs(local_work, exist_ok=True)
    os.makedirs(local_shorts, exist_ok=True)

    final_session_dir = os.path.join(args.out, sid)
    final_shorts_dir = os.path.join(final_session_dir, "shorts")
    os.makedirs(final_shorts_dir, exist_ok=True)

    src_mp4 = download_source_1080p(args.url, os.path.join(local_work, "source_1080p.mp4"), args.cookies, min_height)
    wav_path = extract_audio_wav(src_mp4, os.path.join(local_work, "audio.wav"))
    segments = transcribe_segments(wav_path, os.path.join(local_work, "transcript.json")) if wav_path else []
    audio_scores = score_audio_peaks_fast(wav_path)

    selected = select_clips_full_stream(
        segments, audio_scores, duration, args.clips, min_len, max_len, args.ai_provider, args.ollama_model
    )

    enc_name, enc_args = detect_fast_1080p_encoder()
    workers = 2 if enc_name == "h264_nvenc" else max(1, min(args.workers, os.cpu_count() or 4))
    print(f"[4/4] Multi-process rendering {len(selected)} 1080p Shorts ({workers} workers, {enc_name})...")

    tasks = [
        (
            src_mp4, i, c, local_work, local_shorts, final_shorts_dir,
            enc_name, enc_args, segments, args.caption_style, args.hook_style, args.split_cam
        )
        for i, c in enumerate(selected, 1)
    ]

    results_map = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_render_worker, t): (t[1], t[2]) for t in tasks}
        for fut in as_completed(futures):
            i, c = futures[fut]
            idx, fname, elapsed, err = fut.result()
            results_map[idx] = {
                "clip": idx, "start": c["start"], "end": c["end"],
                "duration": c["duration"], "title": c["title"], "reason": c["reason"], "file": fname
            }
            if err:
                print(f"  [{idx}/{len(selected)}] FAILED ({elapsed:.1f}s): {err}")
            else:
                print(f"  [{idx}/{len(selected)}] {c['start']}s->{c['end']}s ({c['duration']}s) in {elapsed:.1f}s | {c['title']}")

    rows = [results_map[i] for i in sorted(results_map.keys())]
    write_reports(final_session_dir, title, args.url, rows)
    print(f"\nDONE in {time.time() - t0:.1f}s total! Output: {final_session_dir}")

if __name__ == "__main__":
    main()
