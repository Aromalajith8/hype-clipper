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
HOOK_STYLES = ["bold-box", "outline-glow", "top-bar", "mid-card", "none"]
LAYOUTS = ["cam-top", "cam-bottom", "crop-focus", "blur-bg"]
CAM_POSITIONS = ["mid-left", "mid-right", "top-left", "top-right", "bottom-left", "bottom-right", "custom"]

GROQ_FREE_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
GEMINI_FREE_MODELS = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-2.5-flash"]

def py_cmd():
    return "py" if os.name == "nt" else "python3"

def find_font_file():
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\impact.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibrib.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def escape_drawtext(txt):
    s = re.sub(r"\s+", " ", (txt or "").replace("{", "(").replace("}", ")")).strip()
    s = s.replace("\\", "").replace(":", "\\:").replace("'", "\u2019").replace("%", "\\%")
    return s[:70]

def probe_video_info(video_path):
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type", "-of", "json", video_path],
        capture_output=True, text=True
    )
    dur, has_audio = 3600.0, True
    if res.returncode == 0 and res.stdout.strip():
        try:
            data = json.loads(res.stdout)
            if data.get("format", {}).get("duration"):
                dur = float(data["format"]["duration"])
            streams = data.get("streams", [])
            has_audio = any(s.get("codec_type") == "audio" for s in streams)
        except Exception:
            pass
    return dur, has_audio

def detect_content_crop(video_path):
    cmd = [
        "ffmpeg", "-hide_banner", "-ss", "15", "-t", "6",
        "-i", os.path.abspath(video_path),
        "-vf", "cropdetect=24:16:0", "-f", "null", "-"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    crops = re.findall(r"crop=(\d+:\d+:\d+:\d+)", res.stderr or "")
    if not crops:
        return None
    most_common = max(set(crops), key=crops.count)
    w, h, x, y = [int(v) for v in most_common.split(":")]
    if w >= 200 and h >= 200:
        return f"crop={w}:{h}:{x}:{y}"
    return None

def get_metadata(url, cookies=None):
    if os.path.isfile(url):
        dur, _ = probe_video_info(url)
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
        print(f"[1/5] Using local/Drive video: {url}")
        return os.path.abspath(url)
    if os.path.exists(dest_mp4) and os.path.getsize(dest_mp4) > 1024 * 1024:
        print(f"[1/5] Reusing cached source video: {dest_mp4}")
        return os.path.abspath(dest_mp4)

    print(f"[1/5] Downloading source (>= {min_height}p, 12 parallel fragments)...")
    fmt = (
        f"bestvideo[height>={min_height}][ext=mp4]+bestaudio[ext=m4a]/"
        f"bestvideo[height>={min_height}]+bestaudio/"
        f"bestvideo[ext=mp4]+bestaudio[ext=m4a]/best"
    )
    cmd = [
        py_cmd(), "-m", "yt_dlp",
        "-N", "12", "--concurrent-fragments", "12",
        "-f", fmt, "--merge-output-format", "mp4", "-o", dest_mp4
    ]
    if cookies and os.path.exists(cookies):
        cmd += ["--cookies", cookies]
    cmd.append(url)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not os.path.exists(dest_mp4):
        raise RuntimeError(f"Download failed:\n{res.stderr[-1500:]}")
    return os.path.abspath(dest_mp4)

def extract_audio_wav(video_path, wav_path):
    if os.path.exists(wav_path) and os.path.getsize(wav_path) > 1024:
        return wav_path
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav_path]
    subprocess.run(cmd, capture_output=True, text=True)
    return wav_path if os.path.exists(wav_path) else None

def generate_hook_sfx(sfx_path):
    if os.path.exists(sfx_path) and os.path.getsize(sfx_path) > 1024:
        return os.path.abspath(sfx_path)
    sr = 44100
    dur = 0.55
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    rng = np.random.default_rng(42)
    noise = rng.uniform(-1.0, 1.0, t.size)
    env_whoosh = np.sin(np.pi * (t / dur)) ** 2
    whoosh = np.convolve(noise * env_whoosh, np.ones(18) / 18.0, mode="same") * 0.42
    pop_t = np.maximum(0.0, t - 0.12)
    pop_env = np.exp(-pop_t * 28.0) * (t >= 0.12)
    freq = 160.0 * np.exp(-pop_t * 10.0) + 45.0
    pop = np.sin(2.0 * np.pi * np.cumsum(freq) / sr) * pop_env * 0.62
    mono = np.clip(whoosh + pop, -0.95, 0.95)
    stereo = np.column_stack((mono, mono))
    pcm = (stereo * 32767.0).astype(np.int16)
    with wave.open(sfx_path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return os.path.abspath(sfx_path)

def transcribe_segments(wav_path, cache_json):
    if os.path.exists(cache_json):
        try:
            with open(cache_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data and isinstance(data, list) and "words" in data[0]:
                print(f"[2/5] Loaded {len(data)} cached word-synced transcript segments.")
                return data
        except Exception:
            pass

    print("[2/5] Transcribing with faster-whisper (word-level sync)...")
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        print(f"  -> WARNING: faster-whisper not installed ({e}). Falling back to hook captions.")
        return []

    has_gpu = subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    attempts = [("cuda", "float16", "small"), ("cpu", "int8", "base")] if has_gpu else [("cpu", "int8", "base")]

    for device, compute, model_name in attempts:
        try:
            model = WhisperModel(model_name, device=device, compute_type=compute)
            segs, _ = model.transcribe(wav_path, beam_size=1, vad_filter=True, word_timestamps=True)
            out = []
            for s in segs:
                txt = s.text.strip()
                if not txt:
                    continue
                words = [
                    {"start": round(float(w.start), 2), "end": round(float(w.end), 2), "word": w.word.strip()}
                    for w in (s.words or []) if w.word and w.word.strip()
                ]
                out.append({"start": round(float(s.start), 2), "end": round(float(s.end), 2), "text": txt, "words": words})
            with open(cache_json, "w", encoding="utf-8") as f:
                json.dump(out, f)
            print(f"  -> Transcribed {len(out)} segments on {device.upper()} ({model_name}).")
            return out
        except Exception as e:
            print(f"  -> Whisper ({device}/{model_name}) failed: {e}")
            if device == "cuda":
                print("  -> Retrying Whisper on CPU...")
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
    eff_min = min(float(min_len), max(5.0, float(duration) * 0.8))
    eff_max = min(float(max_len), float(duration))
    start = max(0.0, min(float(start_s), max(0.0, float(duration) - eff_min)))
    end = min(float(duration), float(end_s))

    if end - start < eff_min:
        end = min(float(duration), start + eff_min)
    if end - start > eff_max:
        end = start + eff_max

    if not segments:
        return round(start, 2), round(end, 2)

    hook_re = re.compile(r"\b(wait|what|why|how|no way|bro|crazy|insane|look|watch|secret|never|caught|mistake|truth|stop|listen|holy|check this)\b", re.I)
    for s in reversed(segments):
        if start - 8.0 <= s["start"] <= start + 2.0 and hook_re.search(s["text"]):
            start = max(0.0, s["start"] - 0.8)
            break

    best_end = min(float(duration), start + eff_max)
    sentence_ends = [
        s["end"] + 0.6 for s in segments
        if s["end"] >= start + eff_min and s["end"] - start <= eff_max and re.search(r"[.!?]$", s["text"].strip())
    ]
    if sentence_ends:
        best_end = min(float(duration), max(sentence_ends))
    else:
        seg_ends = [s["end"] + 0.4 for s in segments if s["end"] >= start + eff_min and s["end"] - start <= eff_max]
        if seg_ends:
            best_end = min(float(duration), max(seg_ends))

    if best_end - start < eff_min:
        best_end = min(float(duration), start + eff_min)
    if best_end - start > eff_max:
        best_end = start + eff_max

    return round(start, 2), round(best_end, 2)

def _parse_ai_json(text):
    m = re.search(r"\[[\s\S]*\]", text or "")
    if not m:
        return []
    try:
        return json.loads(m.group(0))
    except Exception:
        return []

def _ollama_is_alive():
    try:
        return requests.get("http://127.0.0.1:11434/api/version", timeout=1.5).status_code == 200
    except Exception:
        return False

def _call_free_ai_chunk(prompt, provider, ollama_model):
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()

    order = []
    if provider == "auto":
        if groq_key:
            order.append("groq")
        if gemini_key:
            order.append("gemini")
        if not order and _ollama_is_alive():
            order.append("ollama")
    elif provider != "heuristic":
        order.append(provider)

    for p in order:
        if p == "groq" and groq_key:
            for model_id in GROQ_FREE_MODELS:
                try:
                    r = requests.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
                        json={"model": model_id, "messages": [{"role": "user", "content": prompt}], "temperature": 0.35},
                        timeout=45,
                    )
                    if r.status_code == 429:
                        time.sleep(2.0)
                        continue
                    r.raise_for_status()
                    return r.json()["choices"][0]["message"]["content"]
                except Exception as e:
                    print(f"  -> Groq ({model_id}) skipped: {e}")
        elif p == "gemini" and gemini_key:
            for model_id in GEMINI_FREE_MODELS:
                try:
                    r = requests.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={gemini_key}",
                        json={"contents": [{"parts": [{"text": prompt}]}]},
                        timeout=45,
                    )
                    if r.status_code in (429, 503):
                        time.sleep(1.5)
                        continue
                    r.raise_for_status()
                    return r.json()["candidates"][0]["content"]["parts"][0]["text"]
                except Exception as e:
                    print(f"  -> Gemini ({model_id}) skipped: {e}")
        elif p == "ollama":
            try:
                r = requests.post(
                    "http://127.0.0.1:11434/api/generate",
                    json={"model": ollama_model, "prompt": prompt, "stream": False, "options": {"temperature": 0.35}},
                    timeout=120,
                )
                r.raise_for_status()
                return r.json().get("response", "")
            except Exception as e:
                print(f"  -> Ollama ({ollama_model}) skipped: {e}")
    return ""

def select_clips_full_stream(segments, audio_scores, duration, num_clips, min_len, max_len, provider, ollama_model):
    print(f"[3/5] Selecting up to {num_clips} hook-to-payoff clips ({min_len}s-{max_len}s)...")
    eff_min = min(float(min_len), max(5.0, float(duration) * 0.8))
    picked = []

    has_llm = (
        (provider == "groq" and os.environ.get("GROQ_API_KEY", "").strip())
        or (provider == "gemini" and os.environ.get("GEMINI_API_KEY", "").strip())
        or (provider == "ollama" and _ollama_is_alive())
        or (provider == "auto" and (os.environ.get("GROQ_API_KEY", "").strip() or os.environ.get("GEMINI_API_KEY", "").strip() or _ollama_is_alive()))
    )

    if segments and has_llm:
        chunk_size = 240
        num_chunks = max(1, (len(segments) + chunk_size - 1) // chunk_size)
        per_chunk = max(4, (num_clips + num_chunks - 1) // num_chunks + 2)

        for c_idx in range(num_chunks):
            chunk = segments[c_idx * chunk_size : (c_idx + 1) * chunk_size]
            if not chunk:
                continue
            lines = "\n".join(f"[{int(s['start'])}s-{int(s['end'])}s] {s['text']}" for s in chunk)
            prompt = (
                f"Pick up to {per_chunk} high-retention Short-form clips from this transcript section.\n"
                f"Rules:\n"
                f"1. Open with a strong HOOK and hold through a complete PAYOFF/PUNCHLINE/ENDING.\n"
                f"2. Duration MUST be between {int(eff_min)} and {max_len} seconds (strictly <= {max_len}s).\n"
                f"3. Rate viral potential 1-100.\n"
                f"4. Return ONLY valid JSON: [{{\"start\": 120, \"end\": 165, \"score\": 92, \"title\": \"Hook Title\", \"reason\": \"Hook + payoff\"}}].\n\n"
                f"Transcript:\n{lines}"
            )
            raw = _call_free_ai_chunk(prompt, provider, ollama_model)
            for item in _parse_ai_json(raw):
                st, en = snap_hook_payoff(segments, item.get("start", 0), item.get("end", 0), min_len, max_len, duration)
                if en - st >= eff_min and not any(abs(st - p["start"]) < max_len * 0.75 for p in picked):
                    b = int(st // 10)
                    ai_sc = float(item.get("score", 85))
                    energy_bonus = min(15.0, audio_scores.get(b, 0.0) * 0.3)
                    picked.append({
                        "start": st, "end": en, "duration": round(en - st, 1),
                        "score": round(min(100.0, ai_sc + energy_bonus), 1),
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
                st, en = snap_hook_payoff(segments, seg["start"] - 0.8, seg["start"] + max_len, min_len, max_len, duration)
                scored.append((sc, st, en, seg["text"][:60]))
        else:
            for b, sc in sorted(audio_scores.items(), key=lambda x: x[1], reverse=True):
                st, en = snap_hook_payoff([], max(0.0, b * 10 - 5.0), b * 10 + max_len, min_len, max_len, duration)
                scored.append((sc, st, en, f"Peak moment at {int(st)}s"))

        scored.sort(key=lambda x: x[0], reverse=True)
        for sc, st, en, title in scored:
            if any(abs(st - p["start"]) < max_len * 0.75 for p in picked):
                continue
            picked.append({
                "start": st, "end": en, "duration": round(en - st, 1),
                "score": round(min(99.0, sc), 1),
                "title": title, "reason": f"Hook + energy score {sc:.1f}",
            })
            if len(picked) >= num_clips:
                break

    picked = picked[:num_clips]
    picked.sort(key=lambda x: x["start"])
    return picked

def _cam_crop_expr(cam_pos, cam_w, cam_h, cam_x, cam_y):
    cw = max(0.08, min(0.95, float(cam_w)))
    ch = max(0.08, min(0.95, float(cam_h)))
    if cam_pos == "custom":
        rx = max(0.0, min(1.0 - cw, float(cam_x)))
        ry = max(0.0, min(1.0 - ch, float(cam_y)))
    elif cam_pos == "mid-left":
        rx, ry = 0.01, max(0.0, min(1.0 - ch, 0.36))
    elif cam_pos == "mid-right":
        rx, ry = max(0.0, min(1.0 - cw, 0.99 - cw)), max(0.0, min(1.0 - ch, 0.36))
    elif cam_pos == "top-right":
        rx, ry = max(0.0, min(1.0 - cw, 1.0 - cw)), 0.0
    elif cam_pos == "bottom-left":
        rx, ry = 0.0, max(0.0, min(1.0 - ch, 1.0 - ch))
    elif cam_pos == "bottom-right":
        rx, ry = max(0.0, min(1.0 - cw, 1.0 - cw)), max(0.0, min(1.0 - ch, 1.0 - ch))
    else:
        rx, ry = 0.0, 0.0
    return f"crop=iw*{cw:.3f}:ih*{ch:.3f}:iw*{rx:.3f}:ih*{ry:.3f}"

def _build_base_filter(layout, cam_pos, cam_w, cam_h, cam_x, cam_y, content_crop=None):
    src_chain = f"[0:v]{content_crop}[vsrc]" if content_crop else "[0:v]null[vsrc]"
    cam_crop = _cam_crop_expr(cam_pos, cam_w, cam_h, cam_x, cam_y)

    if layout == "crop-focus":
        return f"{src_chain};[vsrc]scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos,crop=1080:1920[vbase]"
    if layout == "cam-top":
        return (
            f"{src_chain};[vsrc]split[cam][game];"
            f"[cam]{cam_crop},scale=1080:680:force_original_aspect_ratio=increase:flags=lanczos,crop=1080:680[top];"
            "[game]scale=1080:1240:force_original_aspect_ratio=increase:flags=lanczos,crop=1080:1240[bot];"
            "[top][bot]vstack[vbase]"
        )
    if layout == "cam-bottom":
        return (
            f"{src_chain};[vsrc]split[game][cam];"
            "[game]scale=1080:1240:force_original_aspect_ratio=increase:flags=lanczos,crop=1080:1240[top];"
            f"[cam]{cam_crop},scale=1080:680:force_original_aspect_ratio=increase:flags=lanczos,crop=1080:680[bot];"
            "[top][bot]vstack[vbase]"
        )
    return (
        f"{src_chain};[vsrc]split[orig][bg];"
        "[bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=18:2,eq=brightness=-0.22[bgd];"
        "[orig]scale=1080:1920:force_original_aspect_ratio=decrease:flags=lanczos[fg];"
        "[bgd][fg]overlay=(W-w)/2:(H-h)/2[vbase]"
    )

def _build_drawtext_chain(clip_segs, start_s, end_s, caption_style, hook_style, hook_title):
    """Builds reliable FFmpeg drawtext filters so captions & hook text render even without libass."""
    fontfile = find_font_file()
    font_opt = f":fontfile='{fontfile.replace(chr(92), '/').replace(':', chr(92)+':')}'" if fontfile else ""
    filters = []

    if hook_style != "none" and hook_title:
        htxt = escape_drawtext(hook_title.upper())
        if htxt:
            y_expr = "680-th/2" if hook_style == "mid-card" else "110"
            box_col = "white@0.92" if hook_style == "mid-card" else ("red@0.90" if hook_style == "top-bar" else "yellow@0.95")
            txt_col = "black" if hook_style == "mid-card" else ("white" if hook_style == "top-bar" else "black")
            filters.append(
                f"drawtext=text='{htxt}'{font_opt}:fontsize=42:fontcolor={txt_col}:box=1:boxcolor={box_col}:boxborderw=14:"
                f"x=(w-tw)/2:y={y_expr}:enable='between(t,0.15,3.80)'"
            )

    if caption_style != "none" and clip_segs:
        # Build short timed caption phrases in the lower/center safe zone
        step = 1 if caption_style in ("word-by-word", "typewriter") else (2 if caption_style in ("beast-bold", "impact-shake", "comic-burst", "bounce", "gaming-glitch") else 4)
        for s in clip_segs:
            words = s.get("words") or []
            if words:
                for i in range(0, len(words), step):
                    grp = words[i : i + step]
                    if not grp:
                        continue
                    rs = max(0.0, grp[0]["start"] - start_s)
                    re_ = min(end_s - start_s, grp[-1]["end"] - start_s)
                    if re_ - rs < 0.12:
                        continue
                    ctxt = escape_drawtext(" ".join(w["word"].upper() for w in grp))
                    if ctxt:
                        fsize = 76 if caption_style == "word-by-word" else (50 if caption_style == "minimal-clean" else 62)
                        tcol = "yellow" if caption_style in ("bold-pop", "bounce", "impact-shake", "word-by-word") else ("cyan" if caption_style in ("neon-glow", "gaming-glitch") else "white")
                        y_pos = "h-260" if caption_style == "minimal-clean" else "h*0.62"
                        filters.append(
                            f"drawtext=text='{ctxt}'{font_opt}:fontsize={fsize}:fontcolor={tcol}:borderw=4:bordercolor=black:"
                            f"x=(w-tw)/2:y={y_pos}:enable='between(t,{rs:.2f},{re_:.2f})'"
                        )
            else:
                rs = max(0.0, s["start"] - start_s)
                re_ = min(end_s - start_s, s["end"] - start_s)
                if re_ - rs < 0.15:
                    continue
                words_list = re.sub(r"\s+", " ", s["text"]).strip().split()
                for i in range(0, len(words_list), step):
                    grp = words_list[i : i + step]
                    if not grp:
                        continue
                    dur_w = (re_ - rs) / max(1, (len(words_list) + step - 1) // step)
                    cs = rs + (i // step) * dur_w
                    ce = min(re_, cs + dur_w)
                    ctxt = escape_drawtext(" ".join(w.upper() for w in grp))
                    if ctxt:
                        filters.append(
                            f"drawtext=text='{ctxt}'{font_opt}:fontsize=62:fontcolor=yellow:borderw=4:bordercolor=black:"
                            f"x=(w-tw)/2:y=h*0.62:enable='between(t,{cs:.2f},{ce:.2f})'"
                        )

    return ",".join(filters) if filters else None

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
    (
        src_mp4, idx, clip, local_shorts, final_shorts,
        enc_name, enc_args, clip_segs, caption_style, hook_style,
        layout, cam_pos, cam_w, cam_h, cam_x, cam_y, content_crop, sfx_path, has_audio
    ) = args_tuple
    t0 = time.time()
    fname = f"short_{idx:03d}.mp4"
    final_path = os.path.abspath(os.path.join(final_shorts, fname))
    if os.path.exists(final_path) and os.path.getsize(final_path) > 10240:
        return idx, fname, 0.0, None

    out_local = os.path.abspath(os.path.join(local_shorts, fname))
    start_s, end_s = clip["start"], clip["end"]
    dur = max(1.0, float(end_s) - float(start_s))

    base_f = _build_base_filter(layout, cam_pos, cam_w, cam_h, cam_x, cam_y, content_crop)
    dt_chain = _build_drawtext_chain(clip_segs, start_s, end_s, caption_style, hook_style, clip["title"])
    vfilter = f"{base_f};[vbase]{dt_chain}[vout]" if dt_chain else f"{base_f};[vbase]null[vout]"

    inputs = ["-ss", f"{start_s:.2f}", "-t", f"{dur:.2f}", "-i", os.path.abspath(src_mp4)]
    if has_audio and sfx_path and os.path.exists(sfx_path):
        inputs += ["-i", os.path.abspath(sfx_path)]
        fc = (
            f"{vfilter};"
            "[1:a]adelay=120|120,volume=0.45[sfx];"
            "[0:a]aformat=sample_rates=44100:channel_layouts=stereo[main_a];"
            "[main_a][sfx]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]"
        )
        maps = ["-map", "[vout]", "-map", "[aout]"]
    else:
        fc = vfilter
        maps = ["-map", "[vout]", "-map", "0:a?"]

    def _run_enc(e_name, e_args, filter_str):
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            *inputs,
            "-filter_complex", filter_str,
            *maps,
            "-pix_fmt", "yuv420p",
            "-c:v", e_name, *e_args,
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            out_local
        ]
        return subprocess.run(cmd, capture_output=True, text=True)

    r = _run_enc(enc_name, enc_args, fc)
    if r.returncode != 0 and dt_chain:
        # Retry without fontfile in case fontconfig/freetype dislikes fontfile path
        no_font_dt = re.sub(r":fontfile='[^']*'", "", dt_chain)
        fc2 = fc.replace(dt_chain, no_font_dt)
        r = _run_enc(enc_name, enc_args, fc2)
    if r.returncode != 0 and enc_name != "libx264":
        r = _run_enc("libx264", ["-preset", "ultrafast", "-crf", "21"], fc)

    if os.path.exists(out_local) and os.path.getsize(out_local) > 10240:
        shutil.copy2(out_local, final_path)
        return idx, fname, time.time() - t0, None
    return idx, "FAILED", time.time() - t0, (r.stderr or "").strip()[-350:]

def build_top_10_viral(session_dir, rows):
    print("[5/5] Building clean top_10_viral/ folder + 1080x1920 thumbnails...")
    top_dir = os.path.join(session_dir, "top_10_viral")
    shorts_dir = os.path.join(session_dir, "shorts")
    if os.path.exists(top_dir):
        shutil.rmtree(top_dir, ignore_errors=True)
    os.makedirs(top_dir, exist_ok=True)

    valid = [r for r in rows if r["file"] != "FAILED" and os.path.exists(os.path.join(shorts_dir, r["file"]))]
    ranked = sorted(valid, key=lambda x: x.get("score", 0), reverse=True)[:10]

    top_meta = []
    for rank, r in enumerate(ranked, 1):
        src_mp4 = os.path.join(shorts_dir, r["file"])
        dst_name = f"rank_{rank:02d}_{r['file']}"
        thumb_name = f"rank_{rank:02d}_{os.path.splitext(r['file'])[0]}_thumb.jpg"
        dst_mp4 = os.path.join(top_dir, dst_name)
        thumb_jpg = os.path.join(top_dir, thumb_name)
        shutil.copy2(src_mp4, dst_mp4)
        snap_t = min(1.5, max(0.3, float(r.get("duration", 10)) * 0.2))
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{snap_t:.2f}", "-i", dst_mp4, "-vframes", "1", "-q:v", "2", thumb_jpg],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        top_meta.append({**r, "rank": rank, "top_file": dst_name, "thumb": thumb_name})
    print(f"  -> Saved {len(top_meta)} top viral clips + thumbnails in {top_dir}")
    return top_meta

def write_reports(session_dir, title, url, rows, top_meta):
    trans_dir = os.path.join(session_dir, "transcript")
    rep_dir = os.path.join(session_dir, "report")
    os.makedirs(trans_dir, exist_ok=True)
    os.makedirs(rep_dir, exist_ok=True)

    with open(os.path.join(trans_dir, "clips.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with open(os.path.join(trans_dir, "clips.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "score", "start", "end", "duration", "title", "reason", "file"])
        w.writeheader()
        w.writerows(rows)

    cards = "\n".join(
        f"<div class='card'><img src='../top_10_viral/{t['thumb']}'/><div><b>#{t['rank']} (Score {t['score']})</b><br>{t['title']}<br><small>{t['duration']}s | {t['top_file']}</small></div></div>"
        for t in top_meta
    )
    trs = "\n".join(
        f"<tr><td>{r['clip']}</td><td><b>{r['score']}</b></td><td>{r['start']}s - {r['end']}s</td><td>{r['duration']}s</td><td>{r['title']}</td><td>{r['reason']}</td><td>{r['file']}</td></tr>"
        for r in rows
    )
    html = (
        f"<!doctype html><html><head><meta charset='utf-8'><title>HypeClipper v5 - {title}</title>"
        f"<style>body{{font-family:sans-serif;background:#0f1115;color:#eee;padding:24px}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:14px;margin-bottom:28px}}.card{{background:#171b24;border:1px solid #2b303c;border-radius:8px;overflow:hidden;padding:8px}}.card img{{width:100%;border-radius:6px;margin-bottom:6px}}table{{width:100%;border-collapse:collapse}}th,td{{border:1px solid #2b303c;padding:8px;text-align:left}}th{{background:#1b202b}}</style></head>"
        f"<body><h1>HypeClipper Report: {title}</h1><p>Source: {url}</p><h2>Top 10 Viral Picks</h2><div class='grid'>{cards}</div><h2>All Clips</h2><table><thead><tr><th>#</th><th>Score</th><th>Range</th><th>Duration</th><th>Hook/Title</th><th>Payoff Reason</th><th>File</th></tr></thead><tbody>{trs}</tbody></table></body></html>"
    )
    with open(os.path.join(rep_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

def main():
    t0 = time.time()
    parser = argparse.ArgumentParser(description="HypeClipper v5: Drawtext + Subtitle Burn-in, Top 10 Viral, SFX, Custom Cam Crop")
    parser.add_argument("url")
    parser.add_argument("--clips", type=int, default=40)
    parser.add_argument("--min-len", type=int, default=25)
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--min-height", type=int, default=1080)
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 4))
    parser.add_argument("--out", default="HypeClipper")
    parser.add_argument("--work-dir", default="/tmp/hypeclipper_fast" if os.name != "nt" else r"D:\hype-clipper\.fast_work")
    parser.add_argument("--cookies", default=None)
    parser.add_argument("--layout", choices=LAYOUTS, default="cam-top")
    parser.add_argument("--cam-pos", choices=CAM_POSITIONS, default="mid-left")
    parser.add_argument("--cam-w", type=float, default=0.26)
    parser.add_argument("--cam-h", type=float, default=0.34)
    parser.add_argument("--cam-x", type=float, default=0.01)
    parser.add_argument("--cam-y", type=float, default=0.36)
    parser.add_argument("--no-autocrop", action="store_true")
    parser.add_argument("--sfx", action="store_true")
    parser.add_argument("--caption-style", choices=CAPTION_STYLES, default="beast-bold")
    parser.add_argument("--hook-style", choices=HOOK_STYLES, default="mid-card")
    parser.add_argument("--ai-provider", choices=["auto", "groq", "gemini", "ollama", "heuristic"], default="auto")
    parser.add_argument("--ollama-model", default="qwen3.5:9b")
    args = parser.parse_args()

    max_len = min(args.max_len, 50)
    min_len = min(args.min_len, max_len)
    min_height = max(1080, args.min_height)

    meta = get_metadata(args.url, args.cookies)
    title = meta.get("title") or "Stream"
    sid = hashlib.md5(f"{args.url}|{min_len}|{max_len}|{min_height}|{args.layout}|{args.cam_pos}|{args.cam_w}|{args.cam_h}|{args.cam_x}|{args.cam_y}|{args.caption_style}|{args.hook_style}|{args.sfx}".encode()).hexdigest()[:10]

    local_work = os.path.abspath(os.path.join(args.work_dir, sid))
    local_shorts = os.path.join(local_work, "shorts")
    os.makedirs(local_work, exist_ok=True)
    os.makedirs(local_shorts, exist_ok=True)

    final_session_dir = os.path.abspath(os.path.join(args.out, sid))
    final_shorts_dir = os.path.join(final_session_dir, "shorts")
    os.makedirs(final_shorts_dir, exist_ok=True)

    src_mp4 = download_source_1080p(args.url, os.path.join(local_work, "source_1080p.mp4"), args.cookies, min_height)
    duration, has_audio = probe_video_info(src_mp4)
    content_crop = None if args.no_autocrop else detect_content_crop(src_mp4)
    if content_crop:
        print(f"  -> Detected borders; trimming source with {content_crop}")

    wav_path = extract_audio_wav(src_mp4, os.path.join(local_work, "audio.wav")) if has_audio else None
    sfx_path = generate_hook_sfx(os.path.join(local_work, "hook_sfx.wav")) if (args.sfx and has_audio) else None
    segments = transcribe_segments(wav_path, os.path.join(local_work, "transcript.json")) if wav_path else []
    audio_scores = score_audio_peaks_fast(wav_path)

    selected = select_clips_full_stream(
        segments, audio_scores, duration, args.clips, min_len, max_len, args.ai_provider, args.ollama_model
    )

    enc_name, enc_args = detect_fast_1080p_encoder()
    workers = 2 if enc_name == "h264_nvenc" else max(1, min(args.workers, os.cpu_count() or 4))
    print(f"[4/5] Rendering {len(selected)} 1080p Shorts ({workers} workers, {enc_name}, layout={args.layout}, cam={args.cam_pos})...")

    tasks = []
    for i, c in enumerate(selected, 1):
        c_segs = [s for s in segments if s["end"] > c["start"] and s["start"] < c["end"]]
        tasks.append((
            src_mp4, i, c, local_shorts, final_shorts_dir,
            enc_name, enc_args, c_segs, args.caption_style, args.hook_style,
            args.layout, args.cam_pos, args.cam_w, args.cam_h, args.cam_x, args.cam_y,
            content_crop, sfx_path, has_audio
        ))

    results_map = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_render_worker, t): (t[1], t[2]) for t in tasks}
        for fut in as_completed(futures):
            i, c = futures[fut]
            idx, fname, elapsed, err = fut.result()
            results_map[idx] = {
                "clip": idx, "score": c.get("score", 0), "start": c["start"], "end": c["end"],
                "duration": c["duration"], "title": c["title"], "reason": c["reason"], "file": fname
            }
            if err:
                print(f"  [{idx}/{len(selected)}] FAILED ({elapsed:.1f}s): {err}")
            else:
                print(f"  [{idx}/{len(selected)}] Score {c.get('score',0)} | {c['start']}s->{c['end']}s ({c['duration']}s) in {elapsed:.1f}s | {c['title']}")

    rows = [results_map[i] for i in sorted(results_map.keys())]
    top_meta = build_top_10_viral(final_session_dir, rows)
    write_reports(final_session_dir, title, args.url, rows, top_meta)
    print(f"\nDONE in {time.time() - t0:.1f}s total! Output: {final_session_dir}")

if __name__ == "__main__":
    main()
