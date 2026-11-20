import argparse, csv, hashlib, json, math, os, re, shutil, subprocess, sys, time, wave
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
HOOK_STYLES = ["mid-card", "bold-box", "outline-glow", "top-bar", "none"]
LAYOUTS = ["cam-top", "cam-bottom", "pip-top-right", "pip-top-left", "crop-focus", "blur-bg"]
CAM_POSITIONS = ["mid-left", "mid-right", "top-left", "top-right", "bottom-left", "bottom-right", "custom"]

GROQ_FREE_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]
GEMINI_FREE_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash",
]

def py_cmd():
    return "py" if os.name == "nt" else "python3"

def has_ffmpeg_ass_filter():
    r = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True)
    return bool(re.search(r"\bass\b", r.stdout or ""))

def prepare_local_font(tmp_dir):
    """Copies one guaranteed bold TTF into tmp_dir/fonts so libass & drawtext never miss fonts."""
    fonts_dir = os.path.join(tmp_dir, "fonts")
    os.makedirs(fonts_dir, exist_ok=True)
    candidates = [
        (r"C:\Windows\Fonts\arialbd.ttf", "Arial"),
        (r"C:\Windows\Fonts\impact.ttf", "Impact"),
        (r"C:\Windows\Fonts\arial.ttf", "Arial"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVu Sans"),
        ("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", "Liberation Sans"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVu Sans"),
    ]
    for src, fam in candidates:
        if os.path.exists(src):
            dst = os.path.join(fonts_dir, os.path.basename(src))
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            return fam, dst
    return ("Arial" if os.name == "nt" else "DejaVu Sans"), None

def probe_video_info(video_path):
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height", "-of", "json", video_path],
        capture_output=True, text=True
    )
    dur, has_audio, w, h = 3600.0, True, 1920, 1080
    if res.returncode == 0 and res.stdout.strip():
        try:
            data = json.loads(res.stdout)
            if data.get("format", {}).get("duration"):
                dur = float(data["format"]["duration"])
            streams = data.get("streams", [])
            has_audio = any(s.get("codec_type") == "audio" for s in streams)
            for s in streams:
                if s.get("codec_type") == "video" and s.get("width") and s.get("height"):
                    w, h = int(s["width"]), int(s["height"])
                    break
        except Exception:
            pass
    return dur, has_audio, w, h

def detect_content_crop(video_path):
    cmd = [
        "ffmpeg", "-hide_banner", "-ss", "30", "-t", "5",
        "-i", os.path.abspath(video_path),
        "-vf", "cropdetect=24:16:0", "-f", "null", "-"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    crops = re.findall(r"crop=(\d+:\d+:\d+:\d+)", res.stderr or "")
    if not crops:
        return None
    most_common = max(set(crops), key=crops.count)
    w, h, x, y = [int(v) for v in most_common.split(":")]
    if w >= 640 and h >= 360:
        return f"crop={w}:{h}:{x}:{y}"
    return None

def get_metadata(url, cookies=None):
    if os.path.isfile(url):
        dur, _, _, _ = probe_video_info(url)
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

    print(f"[1/5] Downloading source (>= {min_height}p, 16 parallel fragments)...")
    fmt = (
        f"bestvideo[height>={min_height}][ext=mp4]+bestaudio[ext=m4a]/"
        f"bestvideo[height>={min_height}]+bestaudio/"
        f"bestvideo[ext=mp4]+bestaudio[ext=m4a]/best"
    )
    cmd = [
        py_cmd(), "-m", "yt_dlp",
        "-N", "16", "--concurrent-fragments", "16",
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
    if os.path.exists(wav_path) and os.path.getsize(wav_path) > 4096:
        return wav_path
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav_path]
    subprocess.run(cmd, capture_output=True, text=True)
    return wav_path if os.path.exists(wav_path) else None

def generate_hook_sfx(sfx_path):
    """Synthesizes a stereo 44.1kHz copyright-free whoosh + punchy sub-pop WAV."""
    if os.path.exists(sfx_path) and os.path.getsize(sfx_path) > 1024:
        return os.path.abspath(sfx_path)
    sr, dur = 44100, 0.55
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    rng = np.random.default_rng(42)
    noise = rng.uniform(-1.0, 1.0, t.size)
    whoosh = np.convolve(noise * (np.sin(np.pi * (t / dur)) ** 2), np.ones(18) / 18.0, mode="same") * 0.40
    pop_t = np.maximum(0.0, t - 0.12)
    pop_env = np.exp(-pop_t * 28.0) * (t >= 0.12)
    freq = 165.0 * np.exp(-pop_t * 10.0) + 45.0
    pop = np.sin(2.0 * np.pi * np.cumsum(freq) / sr) * pop_env * 0.60
    mono = np.clip(whoosh + pop, -0.95, 0.95)
    pcm = (np.column_stack((mono, mono)) * 32767.0).astype(np.int16)
    with wave.open(sfx_path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return os.path.abspath(sfx_path)

def _transcribe_groq_chunked(wav_path, tmp_dir, duration):
    """Ultra-fast free cloud transcription via Groq whisper-large-v3-turbo (15-min compressed chunks)."""
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not groq_key:
        return []
    print("  -> Trying Groq whisper-large-v3-turbo (word-level sync)...")
    chunk_sec = 900
    n_chunks = max(1, int(math.ceil(duration / chunk_sec)))
    out = []
    for idx in range(n_chunks):
        offset = idx * chunk_sec
        chunk_mp3 = os.path.join(tmp_dir, f"groq_chunk_{idx:02d}.mp3")
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", str(offset), "-t", str(chunk_sec),
             "-i", wav_path, "-ac", "1", "-ar", "16000", "-b:a", "32k", chunk_mp3],
            capture_output=True, text=True
        )
        if not os.path.exists(chunk_mp3) or os.path.getsize(chunk_mp3) < 1024:
            continue
        try:
            with open(chunk_mp3, "rb") as f:
                r = requests.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {groq_key}"},
                    files={"file": (os.path.basename(chunk_mp3), f, "audio/mpeg")},
                    data=[
                        ("model", "whisper-large-v3-turbo"),
                        ("response_format", "verbose_json"),
                        ("timestamp_granularities[]", "word"),
                        ("timestamp_granularities[]", "segment"),
                    ],
                    timeout=90,
                )
            if r.status_code == 429:
                time.sleep(3.0)
                continue
            r.raise_for_status()
            payload = r.json()
            all_words = [
                {"start": round(float(w["start"]) + offset, 2), "end": round(float(w["end"]) + offset, 2), "word": w["word"].strip()}
                for w in (payload.get("words") or []) if w.get("word", "").strip()
            ]
            segs = payload.get("segments") or []
            if segs:
                for s in segs:
                    st = round(float(s["start"]) + offset, 2)
                    en = round(float(s["end"]) + offset, 2)
                    txt = (s.get("text") or "").strip()
                    if not txt:
                        continue
                    w_slice = [w for w in all_words if w["end"] > st - 0.1 and w["start"] < en + 0.1]
                    out.append({"start": st, "end": en, "text": txt, "words": w_slice})
            elif all_words:
                for i in range(0, len(all_words), 10):
                    g = all_words[i : i + 10]
                    out.append({"start": g[0]["start"], "end": g[-1]["end"], "text": " ".join(x["word"] for x in g), "words": g})
        except Exception as e:
            print(f"  -> Groq Whisper chunk {idx} skipped ({e}); will fall back to local Whisper.")
            return []
    return out

def transcribe_segments(wav_path, cache_json, tmp_dir, duration, transcribe_mode="auto"):
    if os.path.exists(cache_json):
        try:
            with open(cache_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) > 0 and "words" in data[0]:
                print(f"[2/5] Loaded {len(data)} cached word-synced segments.")
                return data
        except Exception:
            pass

    print(f"[2/5] Transcribing speech (mode={transcribe_mode})...")
    has_gpu = subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0

    # Use Groq Whisper when explicitly requested OR when on CPU with a Groq key (10x faster than CPU Whisper)
    if transcribe_mode == "groq" or (transcribe_mode == "auto" and not has_gpu and os.environ.get("GROQ_API_KEY", "").strip()):
        out = _transcribe_groq_chunked(wav_path, tmp_dir, duration)
        if out:
            with open(cache_json, "w", encoding="utf-8") as f:
                json.dump(out, f)
            print(f"  -> Groq transcribed {len(out)} word-synced segments.")
            return out

    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        print(f"  -> WARNING: faster-whisper not installed ({e}).")
        return []

    attempts = [("cuda", "float16", "small"), ("cpu", "int8", "base")] if has_gpu else [("cpu", "int8", "base")]
    for device, compute, model_name in attempts:
        try:
            model = WhisperModel(model_name, device=device, compute_type=compute)
            for use_vad in (True, False):
                segs, _ = model.transcribe(wav_path, beam_size=1, vad_filter=use_vad, word_timestamps=True)
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
                if out:
                    with open(cache_json, "w", encoding="utf-8") as f:
                        json.dump(out, f)
                    print(f"  -> Transcribed {len(out)} segments on {device.upper()} ({model_name}, vad={use_vad}).")
                    return out
                if use_vad:
                    print("  -> VAD returned 0 segments; retrying without VAD filter...")
        except Exception as e:
            print(f"  -> Whisper ({device}/{model_name}) failed: {e}")
            if device == "cuda":
                print("  -> Retrying Whisper on CPU so subtitles still render...")
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
                rms_vals.append(20.0 * np.log10(float(np.sqrt(np.mean(samples * samples)) + 1e-8)))
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

    hook_re = re.compile(r"\b(wait|what|why|how|no way|bro|crazy|insane|look|watch|secret|never|caught|mistake|truth|stop|listen|holy|check this|oh my)\b", re.I)
    for s in reversed(segments):
        if start - 7.0 <= s["start"] <= start + 2.5 and hook_re.search(s["text"]):
            start = max(0.0, s["start"] - 0.6)
            break

    best_end = min(float(duration), start + eff_max)
    sentence_ends = [
        s["end"] + 0.55 for s in segments
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
                f"Pick up to {per_chunk} viral Short-form clips from this transcript section.\n"
                f"Rules:\n"
                f"1. Open with a strong HOOK and hold through a complete PAYOFF/PUNCHLINE/RESOLUTION.\n"
                f"2. Duration MUST be between {int(eff_min)} and {max_len} seconds (strictly <= {max_len}s).\n"
                f"3. Write a short punchy viral title (max 8 words) and rate viral score 1-100.\n"
                f"4. Return ONLY valid JSON: [{{\"start\": 120, \"end\": 165, \"score\": 92, \"title\": \"He Didn't Expect This\", \"reason\": \"Hook + payoff\"}}].\n\n"
                f"Transcript:\n{lines}"
            )
            raw = _call_free_ai_chunk(prompt, provider, ollama_model)
            for item in _parse_ai_json(raw):
                st, en = snap_hook_payoff(segments, item.get("start", 0), item.get("end", 0), min_len, max_len, duration)
                if en - st >= eff_min and not any(abs(st - p["start"]) < max_len * 0.70 for p in picked):
                    b = int(st // 10)
                    ai_sc = float(item.get("score", 85))
                    energy_bonus = min(15.0, audio_scores.get(b, 0.0) * 0.3)
                    picked.append({
                        "start": st, "end": en, "duration": round(en - st, 1),
                        "score": round(min(100.0, ai_sc + energy_bonus), 1),
                        "title": str(item.get("title", "Wait For The Ending"))[:64],
                        "reason": str(item.get("reason", "AI hook + payoff"))[:140],
                    })
            if len(picked) >= num_clips:
                break

    if len(picked) < num_clips:
        hook_re = re.compile(r"\b(wait|what|why|how|no way|bro|crazy|insane|look|watch|secret|never|caught|mistake|truth|stop|listen|holy|check this|oh my)\b", re.I)
        scored = []
        if segments:
            for seg in segments:
                b = int(seg["start"] // 10)
                wcount = len(seg["text"].split())
                sc = (35.0 if hook_re.search(seg["text"]) else 6.0) + min(45.0, audio_scores.get(b, 0.0)) + min(15.0, wcount * 1.2)
                st, en = snap_hook_payoff(segments, seg["start"] - 0.6, seg["start"] + max_len, min_len, max_len, duration)
                scored.append((sc, st, en, seg["text"][:56]))
        else:
            for b, sc in sorted(audio_scores.items(), key=lambda x: x[1], reverse=True):
                st, en = snap_hook_payoff([], max(0.0, b * 10 - 4.0), b * 10 + max_len, min_len, max_len, duration)
                scored.append((sc, st, en, f"Peak Moment At {int(st)}s"))

        scored.sort(key=lambda x: x[0], reverse=True)
        for sc, st, en, title in scored:
            if any(abs(st - p["start"]) < max_len * 0.65 for p in picked):
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

def _fmt_ass(sec):
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h}:{m:02d}:{s:05.2f}"

def _clean_text(txt):
    return re.sub(r"\s+", " ", (txt or "").replace("{", "(").replace("}", ")").replace("\\", "")).strip()

def _wrap_banner(txt, max_line=24):
    words = _clean_text(txt).split()
    lines, cur = [], []
    for w in words:
        if sum(len(x) for x in cur) + len(cur) + len(w) > max_line and cur:
            lines.append(" ".join(cur))
            cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(" ".join(cur))
    return "\\N".join(lines[:3])

def _style_params(style_name, font_family):
    # ASS colors: &HAABBGGRR (AA=00 opaque, AA=FF transparent)
    # (font, size, pri, out, back, bold, bstyle, out_w, shd, step, mode, accent_bgr)
    f_std = font_family
    f_hvy = "Impact" if os.name == "nt" else font_family
    f_mon = "Courier New" if os.name == "nt" else "DejaVu Sans Mono"
    table = {
        "bold-pop":          (f_std, 72, "&H00FFFFFF", "&H00000000", "&H80000000", -1, 1, 6, 3, 3, "active", "00FFFF"),
        "beast-bold":        (f_hvy, 80, "&H00FFFFFF", "&H00000000", "&H80000000", -1, 1, 7, 3, 2, "active", "44FF44"),
        "karaoke-highlight": (f_std, 68, "&H00FFFFFF", "&H00000000", "&H80000000", -1, 1, 5, 2, 4, "active", "00FFFF"),
        "word-by-word":      (f_hvy, 88, "&H0000FFFF", "&H00000000", "&H80000000", -1, 1, 7, 3, 1, "word",   "00FFFF"),
        "two-line-stack":    (f_std, 62, "&H00FFFFFF", "&H00000000", "&H80000000", -1, 1, 5, 2, 6, "active", "00FFFF"),
        "highlight-box":     (f_std, 64, "&H00000000", "&H0000FFFF", "&H0000FFFF", -1, 3, 12, 0, 3, "phrase", "00FFFF"),
        "neon-glow":         (f_std, 70, "&H00FFFFFF", "&H00FFFF00", "&H80000000", -1, 1, 6, 3, 3, "active", "FFFF00"),
        "impact-shake":      (f_hvy, 78, "&H0000FFFF", "&H00000000", "&H80000000", -1, 1, 6, 3, 2, "phrase", "00FFFF"),
        "emoji-pop":         (f_std, 68, "&H00FFFFFF", "&H00000000", "&H80000000", -1, 1, 5, 2, 3, "active", "00FFFF"),
        "typewriter":        (f_mon, 60, "&H00FFFFFF", "&H00000000", "&H80000000", -1, 1, 4, 2, 1, "word",   "00FFFF"),
        "bounce":            (f_std, 72, "&H0000FFFF", "&H00000000", "&H80000000", -1, 1, 6, 3, 2, "phrase", "00FFFF"),
        "gradient-stroke":   (f_std, 72, "&H0000FFFF", "&H00FF33CC", "&H80000000", -1, 1, 7, 3, 2, "phrase", "00FFFF"),
        "comic-burst":       (f_hvy, 76, "&H004444FF", "&H00000000", "&H80000000", -1, 1, 6, 3, 2, "phrase", "4444FF"),
        "minimal-clean":     (f_std, 52, "&H00FFFFFF", "&H00000000", "&H80000000",  0, 1, 4, 2, 5, "active", "00FFFF"),
        "gaming-glitch":     (f_mon, 70, "&H00FFFF00", "&H00FF00FF", "&H80000000", -1, 1, 5, 2, 2, "phrase", "FFFF00"),
    }
    return table.get(style_name, table["bold-pop"])

def _layout_text_coords(layout, hook_style):
    """Returns (hook_y, sub_y) inside 1080x1920 so hook cards & captions never clash with facecam or Shorts UI."""
    if layout == "cam-top":
        hook_y = 720 if hook_style == "mid-card" else 130
        sub_y = 1320
    elif layout == "cam-bottom":
        hook_y = 1200 if hook_style == "mid-card" else 140
        sub_y = 880
    else:
        hook_y = 260 if hook_style != "mid-card" else 560
        sub_y = 1340
    return hook_y, sub_y

def _extract_word_tokens(clip_segs, start_s, end_s):
    tokens = []
    for s in clip_segs:
        if s.get("words"):
            for w in s["words"]:
                ws = max(0.0, float(w["start"]) - start_s)
                we = min(end_s - start_s, float(w["end"]) - start_s)
                wt = _clean_text(w["word"])
                if wt and we > ws:
                    tokens.append((ws, we, wt))
        else:
            rs = max(0.0, float(s["start"]) - start_s)
            re_ = min(end_s - start_s, float(s["end"]) - start_s)
            words = _clean_text(s["text"]).split()
            if words and re_ > rs:
                wd = (re_ - rs) / len(words)
                for i, wt in enumerate(words):
                    tokens.append((rs + i * wd, min(re_, rs + (i + 1) * wd), wt))
    return tokens

def build_styled_ass(clip_segs, start_s, end_s, ass_path, caption_style, hook_style, hook_title, layout, font_family):
    if caption_style == "none" and hook_style == "none":
        return None, 0
    chosen = caption_style
    if chosen == "auto":
        pool = [s for s in CAPTION_STYLES if s not in ("auto", "none", "minimal-clean")]
        chosen = pool[int(start_s) % len(pool)]

    font, sz, pri, out_c, back_c, bold, bstyle, out_w, shd, step, mode, accent = _style_params(chosen, font_family)
    hook_y, sub_y = _layout_text_coords(layout, hook_style)

    # BorderStyle=3 uses OutlineColour/BackColour for the solid opaque card box
    if hook_style == "mid-card":
        hook_line = f"Style: HookBanner,{font_family},46,&H00000000,&H000000FF,&H00FFFFFF,&H00FFFFFF,-1,0,0,0,100,100,0,0,3,18,0,5,60,60,0,1"
    elif hook_style == "top-bar":
        hook_line = f"Style: HookBanner,{font_family},46,&H00FFFFFF,&H000000FF,&H002626E6,&H002626E6,-1,0,0,0,100,100,0,0,3,16,0,5,50,50,0,1"
    elif hook_style == "outline-glow":
        hook_line = f"Style: HookBanner,{font_family},54,&H00FFFFFF,&H000000FF,&H0000FFFF,&H80000000,-1,0,0,0,100,100,0,0,1,8,3,5,50,50,0,1"
    else:
        hook_line = f"Style: HookBanner,{font_family},46,&H00000000,&H000000FF,&H0000FFFF,&H0000FFFF,-1,0,0,0,100,100,0,0,3,16,0,5,50,50,0,1"

    header = (
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\nScaledBorderAndShadow: yes\nWrapStyle: 2\n\n"
        "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{font},{sz},{pri},&H000000FF,{out_c},{back_c},{bold},0,0,0,100,100,0,0,{bstyle},{out_w},{shd},5,60,60,0,1\n"
        f"{hook_line}\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    events = []
    if hook_style != "none" and hook_title:
        banner = _wrap_banner(hook_title if hook_style == "mid-card" else hook_title.upper(), 26)
        if banner:
            hook_end = "0:00:05.00" if hook_style == "mid-card" else "0:00:03.80"
            events.append(f"Dialogue: 2,0:00:00.10,{hook_end},HookBanner,,0,0,0,,{{\\an5\\pos(540,{hook_y})}}{banner}")

    word_count = 0
    if chosen != "none" and clip_segs:
        tokens = _extract_word_tokens(clip_segs, start_s, end_s)
        word_count = len(tokens)
        for i in range(0, len(tokens), step):
            group = tokens[i : i + step]
            if not group:
                continue
            if mode == "active" and len(group) > 1:
                for active_idx, (ws, we, _) in enumerate(group):
                    nxt = group[active_idx + 1][0] if active_idx + 1 < len(group) else we
                    seg_end = max(we, min(nxt, ws + 1.5))
                    if seg_end - ws < 0.04:
                        continue
                    parts = []
                    for j, (_, _, wt) in enumerate(group):
                        word_up = wt.upper() if chosen != "minimal-clean" else wt
                        if j == active_idx:
                            parts.append(f"{{\\c&H{accent}&}}{word_up}{{\\c&HFFFFFF&}}")
                        else:
                            parts.append(word_up)
                    line = " ".join(parts)
                    if chosen == "two-line-stack" and len(parts) >= 4:
                        mid = len(parts) // 2
                        line = " ".join(parts[:mid]) + "\\N" + " ".join(parts[mid:])
                    events.append(f"Dialogue: 1,{_fmt_ass(ws)},{_fmt_ass(seg_end)},Default,,0,0,0,,{{\\an5\\pos(540,{sub_y})}}{line}")
            else:
                gs, ge = group[0][0], group[-1][1]
                if ge - gs < 0.06:
                    continue
                text_out = " ".join(w[2].upper() if chosen != "minimal-clean" else w[2] for w in group)
                events.append(f"Dialogue: 1,{_fmt_ass(gs)},{_fmt_ass(ge)},Default,,0,0,0,,{{\\an5\\pos(540,{sub_y})}}{text_out}")

    if not events:
        return None, 0
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(events) + "\n")
    return os.path.basename(ass_path), word_count

def _cam_box_norm(cam_pos, cam_w, cam_h, cam_x, cam_y):
    cw = max(0.08, min(0.95, float(cam_w)))
    ch = max(0.08, min(0.95, float(cam_h)))
    if cam_pos == "custom":
        rx, ry = max(0.0, min(1.0 - cw, float(cam_x))), max(0.0, min(1.0 - ch, float(cam_y)))
    elif cam_pos == "mid-left":
        rx, ry = 0.0, max(0.0, min(1.0 - ch, 0.38))
    elif cam_pos == "mid-right":
        rx, ry = max(0.0, min(1.0 - cw, 1.0 - cw)), max(0.0, min(1.0 - ch, 0.38))
    elif cam_pos == "top-right":
        rx, ry = max(0.0, min(1.0 - cw, 1.0 - cw)), 0.0
    elif cam_pos == "bottom-left":
        rx, ry = 0.0, max(0.0, min(1.0 - ch, 1.0 - ch))
    elif cam_pos == "bottom-right":
        rx, ry = max(0.0, min(1.0 - cw, 1.0 - cw)), max(0.0, min(1.0 - ch, 1.0 - ch))
    else:
        rx, ry = 0.0, 0.0
    return cw, ch, rx, ry

def _build_video_filter(layout, cam_pos, cam_w, cam_h, cam_x, cam_y, content_crop=None, ass_filename=None):
    # CRITICAL: setpts=PTS-STARTPTS guarantees t=0.000s at frame 0 so libass captions ALWAYS sync & display!
    src_chain = f"[0:v]setpts=PTS-STARTPTS,{content_crop}[vsrc]" if content_crop else "[0:v]setpts=PTS-STARTPTS[vsrc]"
    cw, ch, rx, ry = _cam_box_norm(cam_pos, cam_w, cam_h, cam_x, cam_y)
    cam_crop = f"crop=iw*{cw:.3f}:ih*{ch:.3f}:iw*{rx:.3f}:ih*{ry:.3f}"

    if layout == "crop-focus":
        base = f"{src_chain};[vsrc]scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos,crop=1080:1920[vbase]"
    elif layout == "cam-top":
        base = (
            f"{src_chain};[vsrc]split[cam][game];"
            f"[cam]{cam_crop},scale=1080:720:force_original_aspect_ratio=increase:flags=lanczos,crop=1080:720[top];"
            "[game]scale=1080:1200:force_original_aspect_ratio=increase:flags=lanczos,crop=1080:1200[bot];"
            "[top][bot]vstack[vbase]"
        )
    elif layout == "cam-bottom":
        base = (
            f"{src_chain};[vsrc]split[game][cam];"
            "[game]scale=1080:1200:force_original_aspect_ratio=increase:flags=lanczos,crop=1080:1200[top];"
            f"[cam]{cam_crop},scale=1080:720:force_original_aspect_ratio=increase:flags=lanczos,crop=1080:720[bot];"
            "[top][bot]vstack[vbase]"
        )
    elif layout in ("pip-top-right", "pip-top-left"):
        ox = "W-w-36" if layout == "pip-top-right" else "36"
        base = (
            f"{src_chain};[vsrc]split[cam][game];"
            "[game]scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos,crop=1080:1920[bg];"
            f"[cam]{cam_crop},scale=360:360:force_original_aspect_ratio=increase:flags=lanczos,crop=360:360,pad=372:372:6:6:white[pip];"
            f"[bg][pip]overlay={ox}:90[vbase]"
        )
    else:
        base = (
            f"{src_chain};[vsrc]split[orig][bg];"
            "[bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=18:2,eq=brightness=-0.22[bgd];"
            "[orig]scale=1080:1920:force_original_aspect_ratio=decrease:flags=lanczos[fg];"
            "[bgd][fg]overlay=(W-w)/2:(H-h)/2[vbase]"
        )

    if ass_filename:
        return f"{base};[vbase]ass={ass_filename}:fontsdir=fonts[vout]"
    return f"{base};[vbase]null[vout]"

def generate_layout_preview(src_mp4, tmp_dir, out_dir, layout, cam_pos, cam_w, cam_h, cam_x, cam_y, content_crop, caption_style, hook_style, font_family, seek_sec=60.0):
    """Generates a 2-second visual check: source frame with green facecam box + rendered 1080x1920 Short frame with burned-in hook & subtitle."""
    dur, _, _, _ = probe_video_info(src_mp4)
    st = min(max(5.0, float(seek_sec)), max(1.0, dur - 5.0))
    cw, ch, rx, ry = _cam_box_norm(cam_pos, cam_w, cam_h, cam_x, cam_y)

    src_box_jpg = os.path.join(out_dir, "preview_source_cambox.jpg")
    rendered_jpg = os.path.join(out_dir, "preview_rendered_short.jpg")

    box_vf = f"drawbox=x=iw*{rx:.3f}:y=ih*{ry:.3f}:w=iw*{cw:.3f}:h=ih*{ch:.3f}:color=lime@0.95:t=8"
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{st:.2f}", "-i", src_mp4, "-vf", box_vf, "-vframes", "1", "-q:v", "2", src_box_jpg],
        capture_output=True, text=True
    )

    sample_segs = [{
        "start": st, "end": st + 3.0, "text": "NO WAY BRO CLIP THAT RIGHT NOW",
        "words": [
            {"start": st + 0.0, "end": st + 0.4, "word": "NO"},
            {"start": st + 0.4, "end": st + 0.9, "word": "WAY"},
            {"start": st + 0.9, "end": st + 1.5, "word": "BRO"},
            {"start": st + 1.5, "end": st + 2.2, "word": "CLIP"},
            {"start": st + 2.2, "end": st + 3.0, "word": "THAT"},
        ]
    }]
    ass_path = os.path.join(tmp_dir, "preview_sub.ass")
    ass_name, _ = build_styled_ass(sample_segs, st, st + 3.0, ass_path, caption_style, hook_style, "He Didn't Expect This Moment", layout, font_family)
    vfilter = _build_video_filter(layout, cam_pos, cam_w, cam_h, cam_x, cam_y, content_crop, ass_name)
    fc_file = os.path.join(tmp_dir, "preview_fc.txt")
    with open(fc_file, "w", encoding="utf-8") as f:
        f.write(vfilter)

    r = subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{st:.2f}", "-t", "1.0", "-i", os.path.abspath(src_mp4),
         "-filter_complex_script", os.path.basename(fc_file), "-map", "[vout]", "-ss", "0.55", "-vframes", "1", "-q:v", "2", os.path.abspath(rendered_jpg)],
        cwd=tmp_dir, capture_output=True, text=True
    )
    if r.returncode != 0:
        print(f"  -> Preview warning: {r.stderr.strip()[-300:]}")
    else:
        print(f"Preview saved:\n  1) Source Cam Box: {src_box_jpg}\n  2) Rendered Short: {rendered_jpg}")
    return src_box_jpg, rendered_jpg

def detect_fast_1080p_encoder():
    res = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True)
    enc = res.stdout or ""
    has_nv = subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    if "h264_nvenc" in enc and has_nv:
        return ("h264_nvenc", ["-preset", "p4", "-cq", "20", "-b:v", "0", "-maxrate", "12M", "-bufsize", "16M"])
    if "h264_qsv" in enc:
        return ("h264_qsv", ["-preset", "veryfast", "-global_quality", "21", "-maxrate", "12M", "-bufsize", "16M"])
    return ("libx264", ["-preset", "veryfast", "-crf", "20"])

def _render_worker(args_tuple):
    (
        src_mp4, idx, clip, tmp_dir, local_shorts, final_shorts,
        enc_name, enc_args, clip_segs, caption_style, hook_style,
        layout, cam_pos, cam_w, cam_h, cam_x, cam_y, content_crop, sfx_path, has_audio, font_family
    ) = args_tuple
    t0 = time.time()
    fname = f"short_{idx:03d}.mp4"
    final_path = os.path.abspath(os.path.join(final_shorts, fname))
    if os.path.exists(final_path) and os.path.getsize(final_path) > 10240:
        return idx, fname, 0.0, 0, None

    out_local = os.path.abspath(os.path.join(local_shorts, fname))
    start_s, end_s = clip["start"], clip["end"]
    dur = max(1.0, float(end_s) - float(start_s))

    ass_path = os.path.join(tmp_dir, f"sub_{idx:03d}.ass")
    ass_name, wcount = build_styled_ass(clip_segs, start_s, end_s, ass_path, caption_style, hook_style, clip["title"], layout, font_family)
    vfilter = _build_video_filter(layout, cam_pos, cam_w, cam_h, cam_x, cam_y, content_crop, ass_name)

    inputs = ["-ss", f"{start_s:.2f}", "-t", f"{dur:.2f}", "-i", os.path.abspath(src_mp4)]
    if has_audio and sfx_path and os.path.exists(sfx_path):
        inputs += ["-i", os.path.abspath(sfx_path)]
        fc = (
            f"{vfilter};\n"
            "[1:a]adelay=100|100,volume=0.45[sfx];\n"
            "[0:a]asetpts=PTS-STARTPTS,aformat=sample_rates=44100:channel_layouts=stereo[main_a];\n"
            "[main_a][sfx]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]"
        )
        maps = ["-map", "[vout]", "-map", "[aout]"]
    else:
        fc = f"{vfilter};\n[0:a]asetpts=PTS-STARTPTS[aout]" if has_audio else vfilter
        maps = ["-map", "[vout]", "-map", "[aout]"] if has_audio else ["-map", "[vout]"]

    fc_file = os.path.join(tmp_dir, f"fc_{idx:03d}.txt")
    with open(fc_file, "w", encoding="utf-8") as f:
        f.write(fc)

    def _run_enc(e_name, e_args):
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            *inputs,
            "-filter_complex_script", os.path.basename(fc_file),
            *maps,
            "-pix_fmt", "yuv420p",
            "-c:v", e_name, *e_args,
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            out_local
        ]
        return subprocess.run(cmd, cwd=tmp_dir, capture_output=True, text=True)

    r = _run_enc(enc_name, enc_args)
    if r.returncode != 0 and enc_name != "libx264":
        r = _run_enc("libx264", ["-preset", "veryfast", "-crf", "20"])

    if os.path.exists(out_local) and os.path.getsize(out_local) > 10240:
        shutil.copy2(out_local, final_path)
        return idx, fname, time.time() - t0, wcount, None
    return idx, "FAILED", time.time() - t0, wcount, (r.stderr or "").strip()[-350:]

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
        snap_t = min(1.6, max(0.4, float(r.get("duration", 10)) * 0.15))
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
        w = csv.DictWriter(f, fieldnames=["clip", "score", "start", "end", "duration", "words", "title", "reason", "file"])
        w.writeheader()
        w.writerows(rows)

    cards = "\n".join(
        f"<div class='card'><img src='../top_10_viral/{t['thumb']}'/><div><b>#{t['rank']} (Score {t['score']})</b><br>{t['title']}<br><small>{t['duration']}s | {t['top_file']}</small></div></div>"
        for t in top_meta
    )
    trs = "\n".join(
        f"<tr><td>{r['clip']}</td><td><b>{r['score']}</b></td><td>{r['start']}s - {r['end']}s</td><td>{r['duration']}s</td><td>{r['words']}</td><td>{r['title']}</td><td>{r['reason']}</td><td>{r['file']}</td></tr>"
        for r in rows
    )
    html = (
        f"<!doctype html><html><head><meta charset='utf-8'><title>HypeClipper v6 - {title}</title>"
        f"<style>body{{font-family:sans-serif;background:#0f1115;color:#eee;padding:24px}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:14px;margin-bottom:28px}}.card{{background:#171b24;border:1px solid #2b303c;border-radius:8px;overflow:hidden;padding:8px}}.card img{{width:100%;border-radius:6px;margin-bottom:6px}}table{{width:100%;border-collapse:collapse}}th,td{{border:1px solid #2b303c;padding:8px;text-align:left}}th{{background:#1b202b}}</style></head>"
        f"<body><h1>HypeClipper v6 Report: {title}</h1><p>Source: {url}</p><h2>Top 10 Viral Picks</h2><div class='grid'>{cards}</div><h2>All Clips</h2><table><thead><tr><th>#</th><th>Score</th><th>Range</th><th>Duration</th><th>Sub Words</th><th>Hook/Title</th><th>Payoff Reason</th><th>File</th></tr></thead><tbody>{trs}</tbody></table></body></html>"
    )
    with open(os.path.join(rep_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

def main():
    t0 = time.time()
    parser = argparse.ArgumentParser(description="HypeClipper v6: Production 1080p+ Shorts Engine (PTS-Synced libass Subtitles, Custom Facecam, Top 10 Viral, SFX)")
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
    parser.add_argument("--cam-h", type=float, default=0.38)
    parser.add_argument("--cam-x", type=float, default=0.0)
    parser.add_argument("--cam-y", type=float, default=0.38)
    parser.add_argument("--autocrop", action="store_true", help="Opt-in border detection for letterboxed/pillarboxed videos")
    parser.add_argument("--sfx", action="store_true")
    parser.add_argument("--caption-style", choices=CAPTION_STYLES, default="beast-bold")
    parser.add_argument("--hook-style", choices=HOOK_STYLES, default="mid-card")
    parser.add_argument("--ai-provider", choices=["auto", "groq", "gemini", "ollama", "heuristic"], default="auto")
    parser.add_argument("--transcribe", choices=["auto", "local", "groq"], default="auto")
    parser.add_argument("--ollama-model", default="qwen3.5:9b")
    parser.add_argument("--preview-layout", action="store_true", help="Render 1-frame facecam + subtitle verification preview and exit")
    parser.add_argument("--preview-sec", type=float, default=120.0)
    args = parser.parse_args()

    max_len = min(args.max_len, 50)
    min_len = min(args.min_len, max_len)
    min_height = max(1080, args.min_height)

    meta = get_metadata(args.url, args.cookies)
    title = meta.get("title") or "Stream"
    # Source cache key depends only on URL + resolution so changing layout/captions reuses downloaded video & transcript!
    src_sid = hashlib.md5(f"{args.url}|{min_height}".encode()).hexdigest()[:10]
    run_sid = hashlib.md5(f"{args.url}|{min_len}|{max_len}|{min_height}|{args.layout}|{args.cam_pos}|{args.cam_w}|{args.cam_h}|{args.cam_x}|{args.cam_y}|{args.caption_style}|{args.hook_style}|{args.sfx}".encode()).hexdigest()[:10]

    shared_cache = os.path.abspath(os.path.join(args.work_dir, f"src_{src_sid}"))
    local_work = os.path.abspath(os.path.join(args.work_dir, f"run_{run_sid}"))
    local_shorts = os.path.join(local_work, "shorts")
    os.makedirs(shared_cache, exist_ok=True)
    os.makedirs(local_work, exist_ok=True)
    os.makedirs(local_shorts, exist_ok=True)

    final_session_dir = os.path.abspath(os.path.join(args.out, run_sid))
    final_shorts_dir = os.path.join(final_session_dir, "shorts")
    os.makedirs(final_shorts_dir, exist_ok=True)

    font_family, _ = prepare_local_font(local_work)
    if not has_ffmpeg_ass_filter():
        print("WARNING: FFmpeg 'ass' filter (libass) not detected in this build!")

    src_mp4 = download_source_1080p(args.url, os.path.join(shared_cache, "source_1080p.mp4"), args.cookies, min_height)
    duration, has_audio, sw, sh = probe_video_info(src_mp4)
    print(f"  -> Source ready: {sw}x{sh}, {duration/60:.1f} min, audio={has_audio}, font={font_family}")
    content_crop = detect_content_crop(src_mp4) if args.autocrop else None

    if args.preview_layout:
        generate_layout_preview(
            src_mp4, local_work, final_session_dir, args.layout, args.cam_pos,
            args.cam_w, args.cam_h, args.cam_x, args.cam_y, content_crop,
            args.caption_style, args.hook_style, font_family, args.preview_sec
        )
        return

    wav_path = extract_audio_wav(src_mp4, os.path.join(shared_cache, "audio.wav")) if has_audio else None
    sfx_path = generate_hook_sfx(os.path.join(shared_cache, "hook_sfx.wav")) if (args.sfx and has_audio) else None
    segments = transcribe_segments(wav_path, os.path.join(shared_cache, "transcript.json"), shared_cache, duration, args.transcribe) if wav_path else []
    audio_scores = score_audio_peaks_fast(wav_path)

    selected = select_clips_full_stream(
        segments, audio_scores, duration, args.clips, min_len, max_len, args.ai_provider, args.ollama_model
    )

    enc_name, enc_args = detect_fast_1080p_encoder()
    workers = 2 if enc_name == "h264_nvenc" else max(1, min(args.workers, os.cpu_count() or 4))
    print(f"[4/5] Rendering {len(selected)} 1080p Shorts ({workers} workers, {enc_name}, layout={args.layout}, cam={args.cam_pos}, subs={args.caption_style})...")

    tasks = []
    for i, c in enumerate(selected, 1):
        c_segs = [s for s in segments if s["end"] > c["start"] and s["start"] < c["end"]]
        tasks.append((
            src_mp4, i, c, local_work, local_shorts, final_shorts_dir,
            enc_name, enc_args, c_segs, args.caption_style, args.hook_style,
            args.layout, args.cam_pos, args.cam_w, args.cam_h, args.cam_x, args.cam_y,
            content_crop, sfx_path, has_audio, font_family
        ))

    results_map = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_render_worker, t): (t[1], t[2]) for t in tasks}
        for fut in as_completed(futures):
            i, c = futures[fut]
            idx, fname, elapsed, wcount, err = fut.result()
            results_map[idx] = {
                "clip": idx, "score": c.get("score", 0), "start": c["start"], "end": c["end"],
                "duration": c["duration"], "words": wcount, "title": c["title"], "reason": c["reason"], "file": fname
            }
            if err:
                print(f"  [{idx}/{len(selected)}] FAILED ({elapsed:.1f}s): {err}")
            else:
                print(f"  [{idx}/{len(selected)}] Score {c.get('score',0)} | words={wcount} | {c['start']}s->{c['end']}s ({c['duration']}s) in {elapsed:.1f}s | {c['title']}")

    rows = [results_map[i] for i in sorted(results_map.keys())]
    top_meta = build_top_10_viral(final_session_dir, rows)
    write_reports(final_session_dir, title, args.url, rows, top_meta)
    print(f"\nDONE in {time.time() - t0:.1f}s total! Output: {final_session_dir}")

if __name__ == "__main__":
    main()
