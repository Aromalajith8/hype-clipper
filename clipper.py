import argparse, glob, hashlib, json, math, os, re, shutil, subprocess, sys, time, wave
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
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

def log(msg):
    print(msg, flush=True)

def py_cmd():
    return "py" if os.name == "nt" else "python3"

def has_ffmpeg_ass_filter():
    r = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True)
    return bool(re.search(r"\bass\b", r.stdout or ""))

def prepare_local_fonts(tmp_dir):
    fonts_dir = os.path.join(tmp_dir, "fonts")
    os.makedirs(fonts_dir, exist_ok=True)
    candidates = [
        (r"C:\Windows\Fonts\arialbd.ttf", "Arial"),
        (r"C:\Windows\Fonts\impact.ttf", "Impact"),
        (r"C:\Windows\Fonts\courbd.ttf", "Courier New"),
        (r"C:\Windows\Fonts\arial.ttf", "Arial"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVu Sans"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", "DejaVu Sans Mono"),
        ("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", "Liberation Sans"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVu Sans"),
    ]
    primary_family = None
    for src, fam in candidates:
        if os.path.exists(src):
            dst = os.path.join(fonts_dir, os.path.basename(src))
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            if primary_family is None:
                primary_family = fam
    return primary_family or ("Arial" if os.name == "nt" else "DejaVu Sans")

def probe_video_info(video_path):
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height", "-of", "json", video_path],
        capture_output=True, text=True, timeout=30
    )
    dur, has_audio, w, h = 0.0, True, 0, 0
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

def get_metadata(url, cookies=None):
    if os.path.isfile(url):
        dur, _, _, _ = probe_video_info(url)
        return {"title": os.path.splitext(os.path.basename(url))[0], "duration": dur or 3600.0, "heatmap": []}
    cmd = [py_cmd(), "-m", "yt_dlp", "--socket-timeout", "20", "--retries", "3", "--dump-single-json", "--skip-download"]
    if cookies and os.path.exists(cookies):
        cmd += ["--cookies", cookies]
    cmd.append(url)
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if res.returncode != 0:
        raise RuntimeError(f"yt-dlp metadata failed:\n{res.stderr[-1000:]}")
    return json.loads(res.stdout)

def extract_heatmap_scores(meta, bucket_sec=10):
    scores = defaultdict(float)
    for item in (meta.get("heatmap") or []):
        try:
            b = int(float(item["start_time"]) // bucket_sec)
            val = float(item.get("value", 0.0)) * 85.0
            if val > scores[b]:
                scores[b] = val
        except Exception:
            continue
    return scores

def fetch_youtube_captions(url, cache_dir, cookies=None):
    cache_json = os.path.join(cache_dir, "yt_captions.json")
    if os.path.exists(cache_json):
        try:
            with open(cache_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) > 20:
                log(f"  -> Loaded {len(data)} cached YouTube transcript blocks.")
                return data
        except Exception:
            pass

    if os.path.isfile(url):
        return []

    log("  -> Checking YouTube captions/auto-subs (fast scout)...")
    out_tpl = os.path.join(cache_dir, "yt_sub_raw.%(ext)s")
    cmd = [
        py_cmd(), "-m", "yt_dlp", "--socket-timeout", "20", "--retries", "2",
        "--skip-download", "--write-sub", "--write-auto-sub",
        "--sub-lang", "en,en-orig,en-US,en-GB", "--sub-format", "json3", "-o", out_tpl
    ]
    if cookies and os.path.exists(cookies):
        cmd += ["--cookies", cookies]
    cmd.append(url)
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        log("  -> Caption fetch timed out; continuing with audio/heatmap scout.")
        return []

    json3_files = sorted(glob.glob(os.path.join(cache_dir, "yt_sub_raw*.json3")))
    if not json3_files:
        return []

    raw_segs = []
    try:
        with open(json3_files[0], "r", encoding="utf-8") as f:
            payload = json.load(f)
        for ev in payload.get("events", []):
            segs = ev.get("segs")
            if not segs:
                continue
            txt = re.sub(r"\s+", " ", "".join(s.get("utf8", "") for s in segs).replace("\n", " ")).strip()
            if not txt:
                continue
            st = float(ev.get("tStartMs", 0)) / 1000.0
            dur = float(ev.get("dDurationMs", 2500)) / 1000.0
            raw_segs.append({"start": round(st, 2), "end": round(st + max(0.4, dur), 2), "text": txt})
    except Exception:
        return []

    merged = []
    for s in raw_segs:
        if merged and (s["start"] - merged[-1]["end"] < 1.2) and (s["end"] - merged[-1]["start"] < 6.5) and len(merged[-1]["text"]) < 95:
            merged[-1]["end"] = max(merged[-1]["end"], s["end"])
            merged[-1]["text"] = (merged[-1]["text"] + " " + s["text"]).strip()
        else:
            merged.append(dict(s))

    if merged:
        with open(cache_json, "w", encoding="utf-8") as f:
            json.dump(merged, f)
        log(f"  -> Fetched {len(merged)} transcript blocks from YouTube captions.")
    return merged

def download_audio_only(url, dest_m4a, cookies=None):
    if os.path.exists(dest_m4a) and os.path.getsize(dest_m4a) > 65536:
        return os.path.abspath(dest_m4a)
    log("  -> Downloading lightweight audio-only track for hype peak scanning...")
    cmd = [
        py_cmd(), "-m", "yt_dlp", "--socket-timeout", "20", "--retries", "3", "--fragment-retries", "3",
        "-N", "16", "--concurrent-fragments", "16",
        "-f", "bestaudio[ext=m4a]/bestaudio", "-o", dest_m4a
    ]
    if cookies and os.path.exists(cookies):
        cmd += ["--cookies", cookies]
    cmd.append(url)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        log("  -> Audio-only download timed out.")
        return None
    if os.path.exists(dest_m4a) and os.path.getsize(dest_m4a) > 10240:
        return os.path.abspath(dest_m4a)
    for m in sorted(glob.glob(dest_m4a + "*")):
        if os.path.getsize(m) > 10240:
            return os.path.abspath(m)
    log(f"  -> Audio-only warning: {(r.stderr or '')[-250:]}")
    return None

def stream_audio_peaks_zero_disk(media_path, cache_json, bucket_sec=10):
    if cache_json and os.path.exists(cache_json):
        try:
            with open(cache_json, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if raw:
                return defaultdict(float, {int(k): float(v) for k, v in raw.items()})
        except Exception:
            pass

    deltas = defaultdict(float)
    if not media_path or not os.path.exists(media_path):
        return deltas

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", os.path.abspath(media_path),
        "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "pipe:1"
    ]
    bytes_per_bucket = max(320, int(16000 * bucket_sec * 2))
    rms_vals = []
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        while True:
            chunk = proc.stdout.read(bytes_per_bucket)
            if not chunk:
                break
            usable = len(chunk) - (len(chunk) % 2)
            if usable <= 0:
                continue
            samples = np.frombuffer(chunk[:usable], dtype=np.int16).astype(np.float32) / 32768.0
            rms_vals.append(20.0 * np.log10(float(np.sqrt(np.mean(samples * samples)) + 1e-8)))
        proc.wait(timeout=30)
        for i, db in enumerate(rms_vals):
            prev = rms_vals[i - 1] if i > 0 else db
            deltas[i] = round(max(0.0, db - prev) * 2.5 + max(0.0, db + 35.0), 2)
        if deltas and cache_json:
            with open(cache_json, "w", encoding="utf-8") as f:
                json.dump({str(k): v for k, v in deltas.items()}, f)
    except Exception as e:
        log(f"  -> Audio scan warning: {e}")
    return deltas

def download_source_1080p(url, dest_mp4, cookies=None, min_height=1080):
    if os.path.isfile(url):
        log(f"[1/5] Using local/Drive video directly: {url}")
        return os.path.abspath(url)
    if os.path.exists(dest_mp4) and os.path.getsize(dest_mp4) > 1024 * 1024:
        _, _, _, h = probe_video_info(dest_mp4)
        if h >= 720:
            log(f"[1/5] Reusing cached HD source ({h}p): {dest_mp4}")
            return os.path.abspath(dest_mp4)
        try:
            os.remove(dest_mp4)
        except Exception:
            pass

    log(f"[1/5] Downloading full HD source (>= {min_height}p, strictly no 360p fallback)...")
    fmt = f"bestvideo[height>={min_height}]+bestaudio/bestvideo[height>=720]+bestaudio"
    cmd = [
        py_cmd(), "-m", "yt_dlp", "--socket-timeout", "20", "--retries", "3", "--fragment-retries", "3",
        "-N", "16", "--concurrent-fragments", "16",
        "-f", fmt, "--merge-output-format", "mp4", "-o", dest_mp4
    ]
    if cookies and os.path.exists(cookies):
        cmd += ["--cookies", cookies]
    cmd.append(url)
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if res.returncode != 0 or not os.path.exists(dest_mp4):
        raise RuntimeError(f"HD download failed:\n{res.stderr[-1200:]}")
    return os.path.abspath(dest_mp4)

def download_surgical_section(url, start_s, end_s, out_mp4, cookies=None, min_height=1080):
    """Fast keyframe-range HD download (no slow CPU re-encode, hard 75s timeout, rejects <720p)."""
    if os.path.exists(out_mp4) and os.path.getsize(out_mp4) > 50000:
        _, _, _, h = probe_video_info(out_mp4)
        if h >= 720:
            return os.path.abspath(out_mp4), None
        try:
            os.remove(out_mp4)
        except Exception:
            pass

    s0 = max(0.0, float(start_s))
    s1 = max(s0 + 3.0, float(end_s))

    attempts = [
        (f"bestvideo[height>={min_height}]+bestaudio/bestvideo[height>=720]+bestaudio", []),
        (f"bestvideo[height>={min_height}]+bestaudio/bestvideo[height>=720]+bestaudio", ["--extractor-args", "youtube:player_client=tv,web"]),
        ("bestvideo[height>=720]+bestaudio", ["--extractor-args", "youtube:player_client=android,web"]),
    ]
    last_err = ""
    for fmt, extra_args in attempts:
        if os.path.exists(out_mp4):
            try:
                os.remove(out_mp4)
            except Exception:
                pass
        cmd = [
            py_cmd(), "-m", "yt_dlp", "--no-part",
            "--socket-timeout", "15", "--retries", "2", "--fragment-retries", "2",
            "--download-sections", f"*{s0:.1f}-{s1:.1f}",
            "-f", fmt, "--merge-output-format", "mp4",
            *extra_args, "-o", out_mp4
        ]
        if cookies and os.path.exists(cookies):
            cmd += ["--cookies", cookies]
        cmd.append(url)
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=75)
        except subprocess.TimeoutExpired:
            last_err = "Section download timed out (75s)"
            continue
        if os.path.exists(out_mp4) and os.path.getsize(out_mp4) > 50000:
            _, _, _, h = probe_video_info(out_mp4)
            if h >= 720:
                return os.path.abspath(out_mp4), None
            last_err = f"Low-res {h}p rejected"
            try:
                os.remove(out_mp4)
            except Exception:
                pass
        else:
            last_err = (r.stderr or "").strip()[-240:]
    return None, last_err

def _write_stereo_wav(path, mono_arr, sr=44100):
    mono = np.clip(mono_arr, -0.96, 0.96)
    pcm = (np.column_stack((mono, mono)) * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return os.path.abspath(path)

def generate_sfx_pack(sfx_dir):
    """Synthesizes a 5-sound copyright-free viral Shorts SFX pack (44.1kHz stereo)."""
    os.makedirs(sfx_dir, exist_ok=True)
    sr = 44100
    rng = np.random.default_rng(77)
    pack = {}

    p = os.path.join(sfx_dir, "whoosh_in.wav")
    if not os.path.exists(p):
        dur = 0.58
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        noise = rng.uniform(-1.0, 1.0, t.size)
        whoosh = np.convolve(noise * (np.sin(np.pi * (t / dur)) ** 2), np.ones(16) / 16.0, mode="same") * 0.55
        pt = np.maximum(0.0, t - 0.12)
        pop = np.sin(2 * np.pi * np.cumsum(175 * np.exp(-pt * 11.0) + 42) / sr) * np.exp(-pt * 22.0) * (t >= 0.12) * 0.85
        _write_stereo_wav(p, whoosh + pop, sr)
    pack["whoosh_in"] = os.path.abspath(p)

    p = os.path.join(sfx_dir, "vine_boom.wav")
    if not os.path.exists(p):
        dur = 0.85
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        freq = 115.0 * np.exp(-t * 9.5) + 34.0
        sub = np.sin(2 * np.pi * np.cumsum(freq) / sr) * np.exp(-t * 4.2) * 0.92
        punch = np.tanh(sub * 1.9) * 0.85
        click = rng.uniform(-1.0, 1.0, t.size) * np.exp(-t * 90.0) * 0.45
        _write_stereo_wav(p, punch + click, sr)
    pack["vine_boom"] = os.path.abspath(p)

    p = os.path.join(sfx_dir, "pop_ding.wav")
    if not os.path.exists(p):
        dur = 0.30
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        freq = 520.0 + 680.0 * np.exp(-t * 35.0)
        tone = np.sin(2 * np.pi * np.cumsum(freq) / sr) * np.exp(-t * 24.0) * 0.72
        harm = np.sin(2 * np.pi * 1560.0 * t) * np.exp(-t * 30.0) * 0.30
        _write_stereo_wav(p, tone + harm, sr)
    pack["pop_ding"] = os.path.abspath(p)

    p = os.path.join(sfx_dir, "riser_hit.wav")
    if not os.path.exists(p):
        dur = 0.95
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        rise_mask = t < 0.72
        rf = 180.0 + (t / 0.72) ** 2 * 900.0
        riser = np.sin(2 * np.pi * np.cumsum(rf) / sr) * ((t / 0.72) ** 2) * rise_mask * 0.38
        sw = np.convolve(rng.uniform(-1.0, 1.0, t.size) * ((t / 0.72) ** 2) * rise_mask, np.ones(12) / 12.0, mode="same") * 0.35
        ht = np.maximum(0.0, t - 0.72)
        hit = np.sin(2 * np.pi * np.cumsum(140 * np.exp(-ht * 14) + 40) / sr) * np.exp(-ht * 14.0) * (t >= 0.72) * 0.88
        _write_stereo_wav(p, riser + sw + hit, sr)
    pack["riser_hit"] = os.path.abspath(p)

    p = os.path.join(sfx_dir, "glitch_zap.wav")
    if not os.path.exists(p):
        dur = 0.28
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        freq = 1200.0 * np.exp(-t * 16.0) + 140.0
        zap = np.sign(np.sin(2 * np.pi * np.cumsum(freq) / sr)) * np.exp(-t * 16.0) * 0.42
        _write_stereo_wav(p, zap, sr)
    pack["glitch_zap"] = os.path.abspath(p)

    return pack

def plan_clip_sfx_stage1(stage1_mp4, render_dur, rel_segs, sfx_pack):
    """Analyzes the 0.000s-locked stage1 clip's audio spikes + hype words to place context-aware SFX."""
    if not sfx_pack or not os.path.exists(stage1_mp4):
        return []
    events = [(sfx_pack["whoosh_in"], 30, 0.78)]

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", os.path.abspath(stage1_mp4),
        "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "pipe:1"
    ]
    rms_list = []
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        step_bytes = int(16000 * 0.25 * 2)
        while True:
            ch = proc.stdout.read(step_bytes)
            if not ch:
                break
            u = len(ch) - (len(ch) % 2)
            if u <= 0:
                continue
            arr = np.frombuffer(ch[:u], dtype=np.int16).astype(np.float32) / 32768.0
            rms_list.append(20.0 * np.log10(float(np.sqrt(np.mean(arr * arr)) + 1e-8)))
        proc.wait(timeout=15)
    except Exception:
        rms_list = []

    used_times = [0.0]
    if len(rms_list) > 16:
        jumps = []
        for idx_r in range(10, len(rms_list) - 4):
            t_sec = idx_r * 0.25
            delta = max(0.0, rms_list[idx_r] - rms_list[idx_r - 2])
            score = delta * 2.2 + max(0.0, rms_list[idx_r] + 28.0)
            jumps.append((score, t_sec))
        jumps.sort(key=lambda x: x[0], reverse=True)
        if jumps:
            climax_t = jumps[0][1]
            if climax_t >= 3.2:
                events.append((sfx_pack["riser_hit"], int((climax_t - 0.72) * 1000), 0.80))
            else:
                events.append((sfx_pack["vine_boom"], int(climax_t * 1000), 0.90))
            used_times.append(climax_t)
            for _, t2 in jumps[1:]:
                if all(abs(t2 - ut) >= 6.0 for ut in used_times):
                    events.append((sfx_pack["vine_boom"], int(t2 * 1000), 0.86))
                    used_times.append(t2)
                    break

    shock_re = re.compile(r"\b(no way|what the|wtf|bro|insane|crazy|dead|cooked|holy|oh my god|clip that)\b", re.I)
    callout_re = re.compile(r"\b(wait|look|watch|listen|secret|actually|why|how)\b", re.I)
    for s in (rel_segs or []):
        if len(events) >= 5:
            break
        st = float(s.get("start", 0.0))
        if st < 2.8 or st > render_dur - 1.5:
            continue
        if any(abs(st - ut) < 5.0 for ut in used_times):
            continue
        txt = s.get("text", "")
        if shock_re.search(txt):
            events.append((sfx_pack["glitch_zap"], int(st * 1000), 0.70))
            used_times.append(st)
        elif callout_re.search(txt):
            events.append((sfx_pack["pop_ding"], int(st * 1000), 0.75))
            used_times.append(st)

    if len(events) < 2 and render_dur >= 16.0:
        events.append((sfx_pack["pop_ding"], int(render_dur * 0.50 * 1000), 0.75))

    return events[:5]

def load_whisper_once():
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        log(f"  -> WARNING: faster-whisper not installed ({e}).")
        return None
    has_gpu = subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    attempts = [("cuda", "float16", "small"), ("cpu", "int8", "base")] if has_gpu else [("cpu", "int8", "base")]
    for device, compute, model_name in attempts:
        try:
            m = WhisperModel(model_name, device=device, compute_type=compute)
            log(f"  -> Loaded faster-whisper once on {device.upper()} ({model_name}).")
            return m
        except Exception as e:
            log(f"  -> Whisper load ({device}/{model_name}) failed: {e}")
    return None

def transcribe_stage1_exact(whisper_model, stage1_mp4, tmp_wav):
    """Zero-seek transcription directly on the 0.000s-locked stage1 clip -> 100% millisecond subtitle sync!"""
    if whisper_model is None or not stage1_mp4 or not os.path.exists(stage1_mp4):
        return []
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", os.path.abspath(stage1_mp4),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", tmp_wav
    ]
    subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if not os.path.exists(tmp_wav) or os.path.getsize(tmp_wav) < 2048:
        return []
    try:
        for use_vad in (True, False):
            kwargs = dict(beam_size=1, word_timestamps=True, condition_on_previous_text=False, vad_filter=use_vad)
            if use_vad:
                kwargs["vad_parameters"] = dict(min_silence_duration_ms=250, speech_pad_ms=80)
            segs, _ = whisper_model.transcribe(tmp_wav, **kwargs)
            out = []
            for s in segs:
                txt = s.text.strip()
                if not txt:
                    continue
                words = [
                    {"start": round(float(w.start), 3), "end": round(float(w.end), 3), "word": w.word.strip()}
                    for w in (s.words or []) if w.word and w.word.strip()
                ]
                out.append({"start": round(float(s.start), 3), "end": round(float(s.end), 3), "text": txt, "words": words})
            if out:
                return out
        return []
    except Exception as e:
        log(f"  -> Clip Whisper warning: {e}")
        return []
    finally:
        try:
            os.remove(tmp_wav)
        except Exception:
            pass

def build_stratified_candidates(scout_segs, energy_scores, heatmap_scores, duration, num_clips, min_len, max_len):
    hook_re = re.compile(r"\b(wait|what|why|how|no way|bro|crazy|insane|look|watch|secret|never|caught|mistake|truth|stop|listen|holy|check this|oh my|clip that|chat)\b", re.I)
    total_buckets = max(1, int(duration // 10) + 1)
    target_pool = max(num_clips * 4, 60)
    strata = max(3, min(20, int(duration // 1800) + 1))
    per_stratum = max(5, int(math.ceil(target_pool / strata)) + 2)

    seg_by_bucket = defaultdict(list)
    for s in scout_segs:
        seg_by_bucket[int(s["start"] // 10)].append(s)

    candidates = []
    stratum_len = duration / strata
    for st_idx in range(strata):
        t_lo = st_idx * stratum_len
        t_hi = min(duration, (st_idx + 1) * stratum_len)
        b_lo, b_hi = int(t_lo // 10), min(total_buckets, int(t_hi // 10) + 1)
        local_scored = []
        for b in range(b_lo, b_hi):
            t_start = b * 10.0
            if t_start + min_len > duration:
                continue
            segs_here = seg_by_bucket.get(b, []) + seg_by_bucket.get(b + 1, [])
            txt = " ".join(s["text"] for s in segs_here)
            h_bonus = 30.0 if (txt and hook_re.search(txt)) else 0.0
            sc = heatmap_scores.get(b, 0.0) * 1.1 + energy_scores.get(b, 0.0) * 0.9 + h_bonus + min(15.0, len(txt.split()) * 0.5)
            local_scored.append((sc, t_start))
        local_scored.sort(key=lambda x: x[0], reverse=True)
        chosen_here = []
        for sc, ts in local_scored:
            if any(abs(ts - x[1]) < max(min_len, max_len * 0.75) for x in chosen_here):
                continue
            chosen_here.append((sc, ts))
            if len(chosen_here) >= per_stratum:
                break
        candidates.extend(chosen_here)

    candidates.sort(key=lambda x: x[1])
    return candidates[: max(num_clips * 5, 120)]

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
        if start - 5.0 <= s["start"] <= start + 3.0 and hook_re.search(s["text"]):
            start = max(0.0, s["start"] - 0.5)
            break

    best_end = min(float(duration), start + eff_max)
    sentence_ends = [
        s["end"] + 0.45 for s in segments
        if s["end"] >= start + eff_min and s["end"] - start <= eff_max and re.search(r"[.!?]$", s["text"].strip())
    ]
    if sentence_ends:
        best_end = min(float(duration), max(sentence_ends))
    else:
        seg_ends = [s["end"] + 0.35 for s in segments if s["end"] >= start + eff_min and s["end"] - start <= eff_max]
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

def _call_free_ai(prompt, provider, ollama_model):
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
                        timeout=25,
                    )
                    if r.status_code == 429:
                        time.sleep(1.5)
                        continue
                    r.raise_for_status()
                    return r.json()["choices"][0]["message"]["content"]
                except Exception as e:
                    log(f"  -> Groq ({model_id}) skipped: {e}")
        elif p == "gemini" and gemini_key:
            for model_id in GEMINI_FREE_MODELS:
                try:
                    r = requests.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={gemini_key}",
                        json={"contents": [{"parts": [{"text": prompt}]}]},
                        timeout=25,
                    )
                    if r.status_code in (429, 503):
                        time.sleep(1.5)
                        continue
                    r.raise_for_status()
                    return r.json()["candidates"][0]["content"]["parts"][0]["text"]
                except Exception as e:
                    log(f"  -> Gemini ({model_id}) skipped: {e}")
        elif p == "ollama":
            try:
                r = requests.post(
                    "http://127.0.0.1:11434/api/generate",
                    json={"model": ollama_model, "prompt": prompt, "stream": False, "options": {"temperature": 0.35}},
                    timeout=90,
                )
                r.raise_for_status()
                return r.json().get("response", "")
            except Exception as e:
                log(f"  -> Ollama ({ollama_model}) skipped: {e}")
    return ""

def _overlaps_any(st, en, picked, min_gap_ratio=0.55):
    for p in picked:
        inter = max(0.0, min(en, p["end"]) - max(st, p["start"]))
        shortest = max(1.0, min(en - st, p["end"] - p["start"]))
        if inter / shortest > (1.0 - min_gap_ratio):
            return True
    return False

def select_clips_scalable(scout_segs, energy_scores, heatmap_scores, duration, num_clips, min_len, max_len, provider, ollama_model):
    eff_min = min(float(min_len), max(5.0, float(duration) * 0.8))
    max_possible = max(1, int(duration // max(eff_min * 0.85, 5.0)))
    target_clips = min(num_clips, max_possible)
    log(f"[2/5] Selecting {target_clips} hook-to-payoff clips ({min_len}s-{max_len}s) across {duration/60:.1f} min...")

    candidates = build_stratified_candidates(scout_segs, energy_scores, heatmap_scores, duration, target_clips, min_len, max_len)
    picked = []

    has_llm = (
        (provider == "groq" and os.environ.get("GROQ_API_KEY", "").strip())
        or (provider == "gemini" and os.environ.get("GEMINI_API_KEY", "").strip())
        or (provider == "ollama" and _ollama_is_alive())
        or (provider == "auto" and (os.environ.get("GROQ_API_KEY", "").strip() or os.environ.get("GEMINI_API_KEY", "").strip() or _ollama_is_alive()))
    )

    if scout_segs and has_llm and candidates:
        batch_sz = 24
        n_batches = max(1, (len(candidates) + batch_sz - 1) // batch_sz)
        per_batch = max(5, int(math.ceil(target_clips / n_batches)) + 3)
        for b_idx in range(n_batches):
            batch = candidates[b_idx * batch_sz : (b_idx + 1) * batch_sz]
            blocks = []
            for sc, ts in batch:
                win_segs = [s for s in scout_segs if s["end"] > ts - 4 and s["start"] < ts + max_len + 8]
                if not win_segs:
                    continue
                snippet = " ".join(f"[{int(s['start'])}s] {s['text']}" for s in win_segs[:14])
                blocks.append(f"WINDOW {int(ts)}s-{int(min(duration, ts+max_len+6))}s (hype={sc:.0f}): {snippet}")
            if not blocks:
                continue
            prompt = (
                f"Pick up to {per_batch} highest-retention Short-form clips from these hype windows.\n"
                f"Rules:\n"
                f"1. Start on a strong HOOK and hold through a complete PAYOFF/PUNCHLINE/REACTION.\n"
                f"2. Duration MUST be between {int(eff_min)} and {max_len} seconds (strictly <= {max_len}s).\n"
                f"3. Write a punchy clickbait viral title (3-6 words) and score viral potential 1-100.\n"
                f"4. Return ONLY valid JSON: [{{\"start\": 1200, \"end\": 1240, \"score\": 94, \"title\": \"HE DIDN'T EXPECT THIS\", \"reason\": \"Hook + payoff\"}}].\n\n"
                + "\n".join(blocks)
            )
            raw = _call_free_ai(prompt, provider, ollama_model)
            for item in _parse_ai_json(raw):
                st, en = snap_hook_payoff(scout_segs, item.get("start", 0), item.get("end", 0), min_len, max_len, duration)
                if en - st >= eff_min and not _overlaps_any(st, en, picked, min_gap_ratio=0.65):
                    b = int(st // 10)
                    ai_sc = float(item.get("score", 85))
                    bonus = min(15.0, (heatmap_scores.get(b, 0.0) + energy_scores.get(b, 0.0)) * 0.15)
                    picked.append({
                        "start": st, "end": en, "duration": round(en - st, 1),
                        "score": round(min(100.0, ai_sc + bonus), 1),
                        "title": str(item.get("title", "WAIT FOR THE END"))[:56],
                        "reason": str(item.get("reason", "AI hook + payoff"))[:120],
                    })

    ranked_cand = sorted(candidates, key=lambda x: x[0], reverse=True)
    for gap_ratio in (0.70, 0.45):
        if len(picked) >= target_clips:
            break
        for sc, ts in ranked_cand:
            st, en = snap_hook_payoff(scout_segs, ts, ts + max_len, min_len, max_len, duration)
            if en - st < eff_min or _overlaps_any(st, en, picked, min_gap_ratio=gap_ratio):
                continue
            win = [s["text"] for s in scout_segs if s["start"] >= st - 1 and s["start"] <= st + 8]
            title = (win[0][:50] if win else f"INSANE MOMENT AT {int(st//60)}:{int(st%60):02d}")
            picked.append({
                "start": st, "end": en, "duration": round(en - st, 1),
                "score": round(min(98.0, 55.0 + min(43.0, sc * 0.45)), 1),
                "title": title, "reason": f"Hype score {sc:.1f}",
            })
            if len(picked) >= target_clips:
                break

    if len(picked) < target_clips:
        step = max(eff_min, (duration - eff_min) / max(1, target_clips + 1))
        for k in range(1, target_clips * 3):
            if len(picked) >= target_clips:
                break
            ts = min(max(0.0, duration - eff_min), k * (step * 0.65))
            st, en = snap_hook_payoff(scout_segs, ts, ts + max_len, min_len, max_len, duration)
            if en - st < eff_min or _overlaps_any(st, en, picked, min_gap_ratio=0.50):
                continue
            win = [s["text"] for s in scout_segs if s["start"] >= st - 1 and s["start"] <= st + 8]
            title = (win[0][:50] if win else f"MUST WATCH AT {int(st//60)}:{int(st%60):02d}")
            picked.append({
                "start": st, "end": en, "duration": round(en - st, 1),
                "score": 52.0, "title": title, "reason": "Even-coverage highlight",
            })

    picked.sort(key=lambda x: x["score"], reverse=True)
    picked = picked[:target_clips]
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

def _wrap_banner(txt, max_line=22):
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
    if layout == "cam-top":
        return (720 if hook_style == "mid-card" else 140), 1320
    if layout == "cam-bottom":
        return (1200 if hook_style == "mid-card" else 140), 860
    return (540 if hook_style == "mid-card" else 240), 1340

def _extract_clean_word_tokens(rel_segs, max_dur):
    raw = []
    for s in rel_segs:
        if s.get("words"):
            for w in s["words"]:
                ws = max(0.0, float(w["start"]))
                we = min(max_dur, float(w["end"]))
                wt = _clean_text(w["word"])
                if wt and we > ws:
                    raw.append([ws, we, wt])
        else:
            rs = max(0.0, float(s["start"]))
            re_ = min(max_dur, float(s["end"]))
            words = _clean_text(s["text"]).split()
            if words and re_ > rs:
                wd = (re_ - rs) / len(words)
                for i, wt in enumerate(words):
                    raw.append([rs + i * wd, min(re_, rs + (i + 1) * wd), wt])

    raw.sort(key=lambda x: (x[0], x[1]))
    clean = []
    for ws, we, wt in raw:
        if clean and ws < clean[-1][1]:
            clean[-1][1] = max(clean[-1][0] + 0.02, ws - 0.01)
            ws = max(ws, clean[-1][1] + 0.005)
        if we - ws >= 0.03:
            clean.append([ws, we, wt])
    return clean

def build_styled_ass_relative(rel_segs, clip_dur, ass_path, caption_style, hook_style, hook_title, layout, font_family, seed_val=0):
    if caption_style == "none" and hook_style == "none":
        return None, 0
    chosen = caption_style
    if chosen == "auto":
        pool = [s for s in CAPTION_STYLES if s not in ("auto", "none", "minimal-clean")]
        chosen = pool[int(seed_val) % len(pool)]

    font, sz, pri, out_c, back_c, bold, bstyle, out_w, shd, step, mode, accent = _style_params(chosen, font_family)
    hook_y, sub_y = _layout_text_coords(layout, hook_style)

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
        banner = _wrap_banner(hook_title if hook_style == "mid-card" else hook_title.upper(), 24)
        if banner:
            hook_end = _fmt_ass(min(clip_dur, 4.2 if hook_style == "mid-card" else 3.2))
            events.append(f"Dialogue: 2,0:00:00.08,{hook_end},HookBanner,,0,0,0,,{{\\an5\\pos(540,{hook_y})}}{banner}")

    sub_rows = []
    word_count = 0
    if chosen != "none" and rel_segs:
        tokens = _extract_clean_word_tokens(rel_segs, clip_dur)
        word_count = len(tokens)
        for i in range(0, len(tokens), step):
            group = tokens[i : i + step]
            if not group:
                continue
            if mode == "active" and len(group) > 1:
                for active_idx, (ws, we, _) in enumerate(group):
                    nxt_start = group[active_idx + 1][0] if active_idx + 1 < len(group) else we
                    ev_end = min(nxt_start, we + 0.25) if active_idx + 1 < len(group) else we
                    parts = []
                    for j, (_, _, wt) in enumerate(group):
                        word_up = wt.upper() if chosen != "minimal-clean" else wt
                        parts.append(f"{{\\c&H{accent}&}}{word_up}{{\\c&HFFFFFF&}}" if j == active_idx else word_up)
                    line = " ".join(parts)
                    if chosen == "two-line-stack" and len(parts) >= 4:
                        mid = len(parts) // 2
                        line = " ".join(parts[:mid]) + "\\N" + " ".join(parts[mid:])
                    sub_rows.append([ws, ev_end, line])
            else:
                gs, ge = group[0][0], group[-1][1]
                text_out = " ".join(w[2].upper() if chosen != "minimal-clean" else w[2] for w in group)
                sub_rows.append([gs, ge, text_out])

    sub_rows.sort(key=lambda x: x[0])
    for idx_r in range(len(sub_rows)):
        s0, e0, txt = sub_rows[idx_r]
        if idx_r + 1 < len(sub_rows):
            e0 = min(e0, sub_rows[idx_r + 1][0] - 0.015)
        if e0 - s0 >= 0.04:
            events.append(f"Dialogue: 1,{_fmt_ass(s0)},{_fmt_ass(e0)},Default,,0,0,0,,{{\\an5\\pos(540,{sub_y})}}{txt}")

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

def _build_layout_filter(layout, cam_pos, cam_w, cam_h, cam_x, cam_y):
    src_chain = "[0:v]setpts=PTS-STARTPTS[vsrc]"
    cw, ch, rx, ry = _cam_box_norm(cam_pos, cam_w, cam_h, cam_x, cam_y)
    cam_crop = f"crop=iw*{cw:.3f}:ih*{ch:.3f}:iw*{rx:.3f}:ih*{ry:.3f}"

    if layout == "crop-focus":
        return f"{src_chain};[vsrc]scale=1080:1920:force_original_aspect_ratio=increase:flags=bilinear,crop=1080:1920[vout]"
    if layout == "cam-top":
        return (
            f"{src_chain};[vsrc]split[cam][game];"
            f"[cam]{cam_crop},scale=1080:720:force_original_aspect_ratio=increase:flags=bilinear,crop=1080:720[top];"
            "[game]scale=1080:1200:force_original_aspect_ratio=increase:flags=bilinear,crop=1080:1200[bot];"
            "[top][bot]vstack[vout]"
        )
    if layout == "cam-bottom":
        return (
            f"{src_chain};[vsrc]split[game][cam];"
            "[game]scale=1080:1200:force_original_aspect_ratio=increase:flags=bilinear,crop=1080:1200[top];"
            f"[cam]{cam_crop},scale=1080:720:force_original_aspect_ratio=increase:flags=bilinear,crop=1080:720[bot];"
            "[top][bot]vstack[vout]"
        )
    if layout in ("pip-top-right", "pip-top-left"):
        ox = "W-w-36" if layout == "pip-top-right" else "36"
        return (
            f"{src_chain};[vsrc]split[cam][game];"
            "[game]scale=1080:1920:force_original_aspect_ratio=increase:flags=bilinear,crop=1080:1920[bg];"
            f"[cam]{cam_crop},scale=360:360:force_original_aspect_ratio=increase:flags=bilinear,crop=360:360,pad=372:372:6:6:white[pip];"
            f"[bg][pip]overlay={ox}:90[vout]"
        )
    return (
        f"{src_chain};[vsrc]split[orig][bg];"
        "[bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=18:2,eq=brightness=-0.22[bgd];"
        "[orig]scale=1080:1920:force_original_aspect_ratio=decrease:flags=bilinear[fg];"
        "[bgd][fg]overlay=(W-w)/2:(H-h)/2[vout]"
    )

def run_instant_preview(url, cookies, min_height, tmp_dir, out_dir, layout, cam_pos, cam_w, cam_h, cam_x, cam_y, caption_style, hook_style, font_family, preview_sec):
    if os.path.isfile(url):
        prev_src = os.path.abspath(url)
        seek_in = max(1.0, float(preview_sec))
    else:
        prev_mp4 = os.path.join(tmp_dir, f"prev_{int(preview_sec)}.mp4")
        log(f"Downloading 4-second HD preview slice at {int(preview_sec)}s...")
        prev_src, err = download_surgical_section(url, preview_sec, preview_sec + 4.0, prev_mp4, cookies, min_height)
        if not prev_src:
            raise RuntimeError(f"Preview slice download failed: {err}")
        seek_in = 0.5

    cw, ch, rx, ry = _cam_box_norm(cam_pos, cam_w, cam_h, cam_x, cam_y)
    src_box_jpg = os.path.join(out_dir, "preview_source_cambox.jpg")
    rendered_jpg = os.path.join(out_dir, "preview_rendered_short.jpg")

    box_vf = f"drawbox=x=iw*{rx:.3f}:y=ih*{ry:.3f}:w=iw*{cw:.3f}:h=ih*{ch:.3f}:color=lime@0.95:t=8"
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{seek_in:.2f}", "-i", prev_src, "-vf", box_vf, "-vframes", "1", "-q:v", "2", src_box_jpg],
        capture_output=True, text=True, timeout=30
    )

    sample_segs = [{
        "start": 0.0, "end": 3.0, "text": "NO WAY BRO CLIP THAT",
        "words": [
            {"start": 0.0, "end": 0.4, "word": "NO"},
            {"start": 0.4, "end": 0.9, "word": "WAY"},
            {"start": 0.9, "end": 1.5, "word": "BRO"},
            {"start": 1.5, "end": 2.2, "word": "CLIP"},
            {"start": 2.2, "end": 3.0, "word": "THAT"},
        ]
    }]
    ass_path = os.path.join(tmp_dir, "preview_sub.ass")
    ass_name, _ = build_styled_ass_relative(sample_segs, 3.0, ass_path, caption_style, hook_style, "HE DIDN'T EXPECT THIS", layout, font_family)
    base_vf = _build_layout_filter(layout, cam_pos, cam_w, cam_h, cam_x, cam_y)
    vfilter = base_vf.replace("[vout]", "[vbase]") + (f";[vbase]ass={ass_name}:fontsdir=fonts[vout]" if ass_name else ";[vbase]null[vout]")
    fc_file = os.path.join(tmp_dir, "preview_fc.txt")
    with open(fc_file, "w", encoding="utf-8") as f:
        f.write(vfilter)

    r = subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{seek_in:.2f}", "-t", "1.2", "-i", prev_src,
         "-filter_complex_script", os.path.basename(fc_file), "-map", "[vout]", "-ss", "0.55", "-vframes", "1", "-q:v", "2", os.path.abspath(rendered_jpg)],
        cwd=tmp_dir, capture_output=True, text=True, timeout=30
    )
    if r.returncode != 0:
        log(f"Preview render warning: {(r.stderr or '')[-300:]}")
    else:
        log(f"Preview ready!\n  1) {src_box_jpg}\n  2) {rendered_jpg}")

def detect_fast_1080p_encoder():
    res = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True)
    enc = res.stdout or ""
    has_nv = subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    if "h264_nvenc" in enc and has_nv:
        return ("h264_nvenc", ["-preset", "p2", "-rc", "vbr", "-cq", "19", "-b:v", "8M", "-maxrate", "14M", "-bufsize", "20M"])
    if "h264_qsv" in enc:
        return ("h264_qsv", ["-preset", "veryfast", "-global_quality", "19", "-maxrate", "14M", "-bufsize", "20M"])
    return ("libx264", ["-preset", "veryfast", "-crf", "19"])

def _stage1_trim_worker(args_tuple):
    """Stage 1: Creates a clean 0.000s-locked 1080x1920 vertical clip with synced audio."""
    (src_mp4, seek_s, render_dur, idx, tmp_dir, enc_name, enc_args, layout, cam_pos, cam_w, cam_h, cam_x, cam_y, has_audio) = args_tuple
    t0 = time.time()
    out_stage1 = os.path.abspath(os.path.join(tmp_dir, f"stage1_{idx:03d}.mp4"))
    if os.path.exists(out_stage1) and os.path.getsize(out_stage1) > 10240:
        return idx, out_stage1, time.time() - t0, None

    vf = _build_layout_filter(layout, cam_pos, cam_w, cam_h, cam_x, cam_y)
    if has_audio:
        fc = f"{vf};\n[0:a]asetpts=PTS-STARTPTS,aresample=async=1:first_pts=0,aformat=sample_rates=44100:channel_layouts=stereo[aout]"
        maps = ["-map", "[vout]", "-map", "[aout]"]
    else:
        fc = vf
        maps = ["-map", "[vout]"]

    fc_file = os.path.join(tmp_dir, f"fc1_{idx:03d}.txt")
    with open(fc_file, "w", encoding="utf-8") as f:
        f.write(fc)

    inputs = []
    if seek_s is not None:
        fast_seek = max(0.0, float(seek_s) - 4.0)
        fine_seek = float(seek_s) - fast_seek
        inputs += ["-ss", f"{fast_seek:.3f}", "-i", os.path.abspath(src_mp4), "-ss", f"{fine_seek:.3f}", "-t", f"{render_dur:.3f}"]
    else:
        inputs += ["-i", os.path.abspath(src_mp4), "-t", f"{render_dur:.3f}"]

    def _run(e_name, e_args):
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            *inputs,
            "-filter_complex_script", os.path.basename(fc_file),
            *maps,
            "-pix_fmt", "yuv420p",
            "-c:v", e_name, *e_args,
            "-c:a", "aac", "-b:a", "192k",
            out_stage1
        ]
        return subprocess.run(cmd, cwd=tmp_dir, capture_output=True, text=True, timeout=420)

    try:
        r = _run(enc_name, enc_args)
        if r.returncode != 0 and enc_name != "libx264":
            r = _run("libx264", ["-preset", "veryfast", "-crf", "19"])
    except subprocess.TimeoutExpired:
        return idx, None, time.time() - t0, "Stage 1 encode timed out"

    if os.path.exists(out_stage1) and os.path.getsize(out_stage1) > 10240:
        return idx, out_stage1, time.time() - t0, None
    return idx, None, time.time() - t0, (r.stderr or "").strip()[-300:]

def _stage3_finish_worker(args_tuple):
    """Stage 3: Burns 0.000s-locked captions/hook + mixes dynamic SFX onto stage1 clip and saves to Drive."""
    (
        stage1_mp4, render_dur, idx, clip, tmp_dir, local_shorts, final_shorts,
        enc_name, enc_args, rel_segs, caption_style, hook_style,
        layout, sfx_events, has_audio, font_family
    ) = args_tuple
    t0 = time.time()
    fname = f"short_{idx:03d}.mp4"
    final_path = os.path.abspath(os.path.join(final_shorts, fname))
    out_local = os.path.abspath(os.path.join(local_shorts, fname))

    ass_path = os.path.join(tmp_dir, f"sub_{idx:03d}.ass")
    ass_name, wcount = build_styled_ass_relative(
        rel_segs, render_dur, ass_path, caption_style, hook_style, clip["title"], layout, font_family, seed_val=idx
    )
    valid_sfx = [(p, d, v) for (p, d, v) in (sfx_events or []) if p and os.path.exists(p)]

    # Fast path: if no captions/hook AND no SFX, zero extra re-encoding needed!
    if not ass_name and not valid_sfx:
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", stage1_mp4, "-c", "copy", "-movflags", "+faststart", out_local],
            capture_output=True, text=True, timeout=60
        )
        if os.path.exists(out_local) and os.path.getsize(out_local) > 10240:
            shutil.copy2(out_local, final_path)
            try:
                os.remove(stage1_mp4)
            except Exception:
                pass
            return idx, fname, time.time() - t0, wcount, 0, None

    inputs = ["-i", os.path.abspath(stage1_mp4)]
    vf_line = f"[0:v]ass={ass_name}:fontsdir=fonts[vout]" if ass_name else "[0:v]null[vout]"

    if has_audio and valid_sfx:
        sfx_chains = []
        mix_labels = ["[main_a]"]
        for s_idx, (sp, delay_ms, vol) in enumerate(valid_sfx, 1):
            inputs += ["-i", os.path.abspath(sp)]
            sfx_chains.append(f"[{s_idx}:a]adelay={int(delay_ms)}|{int(delay_ms)},volume={vol:.2f}[sfx{s_idx}]")
            mix_labels.append(f"[sfx{s_idx}]")
        fc = (
            f"{vf_line};\n"
            "[0:a]aformat=sample_rates=44100:channel_layouts=stereo[main_a];\n"
            + ";\n".join(sfx_chains) + ";\n"
            + f"{''.join(mix_labels)}amix=inputs={len(mix_labels)}:duration=first:dropout_transition=0:normalize=0,alimiter=limit=0.95[aout]"
        )
        maps = ["-map", "[vout]", "-map", "[aout]"]
        audio_args = ["-c:a", "aac", "-b:a", "192k"]
    else:
        fc = vf_line
        maps = ["-map", "[vout]", "-map", "0:a?"]
        audio_args = ["-c:a", "copy"]

    fc_file = os.path.join(tmp_dir, f"fc3_{idx:03d}.txt")
    with open(fc_file, "w", encoding="utf-8") as f:
        f.write(fc)

    def _run(e_name, e_args):
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            *inputs,
            "-filter_complex_script", os.path.basename(fc_file),
            *maps,
            "-pix_fmt", "yuv420p",
            "-c:v", e_name, *e_args,
            *audio_args,
            "-movflags", "+faststart",
            out_local
        ]
        return subprocess.run(cmd, cwd=tmp_dir, capture_output=True, text=True, timeout=420)

    try:
        r = _run(enc_name, enc_args)
        if r.returncode != 0 and enc_name != "libx264":
            r = _run("libx264", ["-preset", "veryfast", "-crf", "19"])
    except subprocess.TimeoutExpired:
        return idx, "FAILED", time.time() - t0, wcount, len(valid_sfx), "Stage 3 burn-in timed out"

    if os.path.exists(out_local) and os.path.getsize(out_local) > 10240:
        shutil.copy2(out_local, final_path)
        try:
            os.remove(stage1_mp4)
        except Exception:
            pass
        return idx, fname, time.time() - t0, wcount, len(valid_sfx), None
    return idx, "FAILED", time.time() - t0, wcount, len(valid_sfx), (r.stderr or "").strip()[-300:]

def _score_frame_clickbait(mp4_path, ts):
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", f"{ts:.2f}", "-i", mp4_path,
        "-vf", "scale=180:320", "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=15)
    if r.returncode != 0 or len(r.stdout) < 180 * 320 * 3:
        return -1.0
    arr = np.frombuffer(r.stdout[: 180 * 320 * 3], dtype=np.uint8).reshape((320, 180, 3)).astype(np.float32)
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    sharpness = float(np.abs(np.diff(gray, axis=1)).mean() + np.abs(np.diff(gray, axis=0)).mean())
    contrast = float(gray.std())
    mean_luma = float(gray.mean())
    saturation = float(arr.std(axis=2).mean())
    luma_penalty = 0.45 if (mean_luma < 35.0 or mean_luma > 225.0) else 1.0
    return (sharpness * 2.2 + contrast * 1.1 + saturation * 0.9) * luma_penalty

def _find_clickbait_timestamp(mp4_path, dur):
    dur = max(4.0, float(dur))
    lo = min(2.2, dur * 0.15)
    hi = max(lo + 1.5, dur - 1.5)

    peaks = stream_audio_peaks_zero_disk(mp4_path, cache_json=None, bucket_sec=0.5)
    cand_times = []
    if peaks:
        valid_b = [(sc, b * 0.5 + 0.25) for b, sc in peaks.items() if lo <= (b * 0.5 + 0.25) <= hi]
        valid_b.sort(key=lambda x: x[0], reverse=True)
        for _, t_pk in valid_b[:6]:
            if all(abs(t_pk - existing) >= 1.2 for existing in cand_times):
                cand_times.append(t_pk)
                if len(cand_times) >= 4:
                    break

    for frac in (0.38, 0.55, 0.72):
        t_f = lo + (hi - lo) * frac
        if all(abs(t_f - x) >= 1.0 for x in cand_times):
            cand_times.append(t_f)

    best_t, best_sc = cand_times[0] if cand_times else dur * 0.5, -1.0
    for t_c in cand_times[:7]:
        try:
            sc = _score_frame_clickbait(mp4_path, t_c)
            if sc > best_sc:
                best_sc, best_t = sc, t_c
        except Exception:
            continue
    return best_t

def _render_clickbait_thumbnail(mp4_path, thumb_jpg, title, dur, tmp_dir, font_family):
    peak_t = _find_clickbait_timestamp(mp4_path, dur)
    words = _clean_text(title).upper().split()[:5]
    headline = _wrap_banner(" ".join(words) if words else "WAIT FOR IT", max_line=15)

    ass_file = os.path.join(tmp_dir, "thumb_title.ass")
    ass_text = (
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Clickbait,{font_family},92,&H0000FFFF,&H000000FF,&H00000000,&HC0000000,-1,0,0,0,100,100,0,0,1,10,5,5,50,50,0,1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        f"Dialogue: 1,0:00:00.00,0:00:10.00,Clickbait,,0,0,0,,{{\\an5\\pos(540,420)}}{headline}\n"
    )
    with open(ass_file, "w", encoding="utf-8") as f:
        f.write(ass_text)

    vf = "setpts=PTS-STARTPTS,eq=contrast=1.16:saturation=1.32:brightness=0.03,unsharp=5:5:1.1,ass=thumb_title.ass:fontsdir=fonts"
    r = subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{peak_t:.2f}", "-i", os.path.abspath(mp4_path),
         "-vf", vf, "-vframes", "1", "-q:v", "2", os.path.abspath(thumb_jpg)],
        cwd=tmp_dir, capture_output=True, text=True, timeout=30
    )
    if r.returncode != 0 or not os.path.exists(thumb_jpg):
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{peak_t:.2f}", "-i", os.path.abspath(mp4_path),
             "-vframes", "1", "-q:v", "2", os.path.abspath(thumb_jpg)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30
        )
    return peak_t

def build_top_10_viral(session_dir, rows, tmp_dir, font_family):
    log("[5/5] Building top_10_viral/ folder + 100% clickbait core-moment thumbnails...")
    top_dir = os.path.join(session_dir, "top_10_viral")
    shorts_dir = os.path.join(session_dir, "shorts")
    if os.path.exists(top_dir):
        shutil.rmtree(top_dir, ignore_errors=True)
    os.makedirs(top_dir, exist_ok=True)

    valid = [r for r in rows if r["file"] != "FAILED" and os.path.exists(os.path.join(shorts_dir, r["file"]))]
    ranked = sorted(valid, key=lambda x: x.get("score", 0), reverse=True)[:10]

    for rank, r in enumerate(ranked, 1):
        src_mp4 = os.path.join(shorts_dir, r["file"])
        dst_mp4 = os.path.join(top_dir, f"rank_{rank:02d}_{r['file']}")
        thumb_jpg = os.path.join(top_dir, f"rank_{rank:02d}_{os.path.splitext(r['file'])[0]}_thumb.jpg")
        shutil.copy2(src_mp4, dst_mp4)
        pk = _render_clickbait_thumbnail(dst_mp4, thumb_jpg, r["title"], r.get("duration", 30.0), tmp_dir, font_family)
        log(f"  -> Top #{rank:02d}: {os.path.basename(dst_mp4)} | clickbait thumb @ {pk:.1f}s")
    log(f"  -> Saved {len(ranked)} top viral clips + clickbait thumbnails in {top_dir}")

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    t0 = time.time()
    parser = argparse.ArgumentParser(description="HypeClipper v10: Zero-Drift 3-Stage Sync, Anti-Hang Timeouts, Strict HD, Dynamic Multi-SFX, Clickbait Thumbnails")
    parser.add_argument("url")
    parser.add_argument("--clips", type=int, default=20)
    parser.add_argument("--min-len", type=int, default=25)
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--min-height", type=int, default=1080)
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 4))
    parser.add_argument("--out", default="HypeClipper")
    parser.add_argument("--work-dir", default="/tmp/hypeclipper_fast" if os.name != "nt" else r"D:\hype-clipper\.fast_work")
    parser.add_argument("--cookies", default=None)
    parser.add_argument("--fetch-mode", choices=["auto", "surgical", "full"], default="auto")
    parser.add_argument("--layout", choices=LAYOUTS, default="cam-top")
    parser.add_argument("--cam-pos", choices=CAM_POSITIONS, default="mid-left")
    parser.add_argument("--cam-w", type=float, default=0.26)
    parser.add_argument("--cam-h", type=float, default=0.38)
    parser.add_argument("--cam-x", type=float, default=0.0)
    parser.add_argument("--cam-y", type=float, default=0.38)
    parser.add_argument("--sfx", action="store_true")
    parser.add_argument("--no-subtitles", action="store_true")
    parser.add_argument("--caption-style", choices=CAPTION_STYLES, default="beast-bold")
    parser.add_argument("--hook-style", choices=HOOK_STYLES, default="mid-card")
    parser.add_argument("--ai-provider", choices=["auto", "groq", "gemini", "ollama", "heuristic"], default="auto")
    parser.add_argument("--ollama-model", default="qwen3.5:9b")
    parser.add_argument("--preview-layout", action="store_true")
    parser.add_argument("--preview-sec", type=float, default=300.0)
    args = parser.parse_args()

    effective_caption_style = "none" if args.no_subtitles else args.caption_style
    max_len = min(args.max_len, 50)
    min_len = min(args.min_len, max_len)
    min_height = max(720, int(args.min_height))

    meta = get_metadata(args.url, args.cookies)
    title = meta.get("title") or "Stream"
    duration = float(meta.get("duration") or 3600.0)
    is_local = os.path.isfile(args.url)
    use_surgical = (not is_local) and (args.fetch_mode == "surgical" or (args.fetch_mode == "auto" and duration >= 2700.0))

    src_sid = hashlib.md5(f"v10|{args.url}|{min_height}".encode()).hexdigest()[:10]
    run_sid = hashlib.md5(f"v10|{args.url}|{args.clips}|{min_len}|{max_len}|{min_height}|{args.layout}|{args.cam_pos}|{args.cam_w}|{args.cam_h}|{args.cam_x}|{args.cam_y}|{effective_caption_style}|{args.hook_style}|{args.sfx}".encode()).hexdigest()[:10]

    shared_cache = os.path.abspath(os.path.join(args.work_dir, f"src_{src_sid}"))
    local_work = os.path.abspath(os.path.join(args.work_dir, f"run_{run_sid}"))
    local_shorts = os.path.join(local_work, "shorts")
    raw_clips_dir = os.path.join(shared_cache, "surgical_clips")
    os.makedirs(shared_cache, exist_ok=True)
    os.makedirs(local_work, exist_ok=True)
    os.makedirs(local_shorts, exist_ok=True)
    os.makedirs(raw_clips_dir, exist_ok=True)

    final_session_dir = os.path.abspath(os.path.join(args.out, run_sid))
    final_shorts_dir = os.path.join(final_session_dir, "shorts")
    os.makedirs(final_shorts_dir, exist_ok=True)

    font_family = prepare_local_fonts(local_work)
    if not has_ffmpeg_ass_filter():
        log("WARNING: FFmpeg 'ass' filter (libass) not detected in this build!")

    if args.preview_layout:
        run_instant_preview(
            args.url, args.cookies, min_height, local_work, final_session_dir,
            args.layout, args.cam_pos, args.cam_w, args.cam_h, args.cam_x, args.cam_y,
            effective_caption_style, args.hook_style, font_family, args.preview_sec
        )
        return

    log(f"[1/5] Stream: '{title}' | {duration/60:.1f} min | mode={'SURGICAL HD' if use_surgical else 'FULL/LOCAL HD'} | subs={effective_caption_style} | sfx={args.sfx}")
    heatmap_scores = extract_heatmap_scores(meta)
    scout_segs = []
    energy_scores = defaultdict(float)
    full_src_mp4 = None

    sel_cache = os.path.join(final_session_dir, "selected_clips.json")
    selected = []
    if os.path.exists(sel_cache):
        try:
            with open(sel_cache, "r", encoding="utf-8") as f:
                cached_sel = json.load(f)
            if len(cached_sel) >= args.clips:
                selected = cached_sel[: args.clips]
                log(f"[2/5] Resuming checkpoint: loaded {len(selected)} clips from Drive.")
        except Exception:
            selected = []

    if not use_surgical:
        full_src_mp4 = download_source_1080p(args.url, os.path.join(shared_cache, "source_1080p.mp4"), args.cookies, min_height)
        duration, _, _, _ = probe_video_info(full_src_mp4)

    if not selected:
        if not is_local:
            scout_segs = fetch_youtube_captions(args.url, shared_cache, args.cookies)

        if not scout_segs or not heatmap_scores:
            scan_media = full_src_mp4 if full_src_mp4 else download_audio_only(args.url, os.path.join(shared_cache, "scout_audio.m4a"), args.cookies)
            if scan_media:
                energy_scores = stream_audio_peaks_zero_disk(scan_media, os.path.join(shared_cache, "audio_energy.json"))

        selected = select_clips_scalable(
            scout_segs, energy_scores, heatmap_scores, duration, args.clips, min_len, max_len, args.ai_provider, args.ollama_model
        )
        with open(sel_cache, "w", encoding="utf-8") as f:
            json.dump(selected, f, indent=2)

    sfx_pack = generate_sfx_pack(os.path.join(shared_cache, "sfx_pack")) if args.sfx else None
    enc_name, enc_args = detect_fast_1080p_encoder()
    workers = 2 if enc_name == "h264_nvenc" else max(1, min(args.workers, os.cpu_count() or 4))

    # --- STAGE 1: Fetch HD clips & encode 0.000s-locked 1080x1920 intermediates ---
    log(f"[3/5] Stage 1: Fetching HD + locking {len(selected)} clips to 0.000s ({workers} workers, {enc_name})...")
    clip_inputs = {}
    if use_surgical:
        def _dl_one(item):
            i, c = item
            final_mp4 = os.path.join(final_shorts_dir, f"short_{i:03d}.mp4")
            if os.path.exists(final_mp4) and os.path.getsize(final_mp4) > 10240:
                return i, final_mp4, None, True
            raw_mp4 = os.path.join(raw_clips_dir, f"raw_{i:03d}_{int(c['start'])}_{int(c['end'])}.mp4")
            path, err = download_surgical_section(args.url, c["start"], c["end"], raw_mp4, args.cookies, min_height)
            return i, path, err, False

        with ThreadPoolExecutor(max_workers=4) as dl_pool:
            for i, path, err, already_done in dl_pool.map(_dl_one, enumerate(selected, 1)):
                if already_done:
                    clip_inputs[i] = ("DONE", None, None, True)
                elif path:
                    cdur, haud, _, h = probe_video_info(path)
                    log(f"  -> Clip {i}/{len(selected)} fetched at {h}p ({cdur:.1f}s)")
                    clip_inputs[i] = (path, None, min(max_len, max(5.0, cdur or selected[i - 1]["duration"])), haud)
                else:
                    log(f"  -> Clip {i}/{len(selected)} HD fetch failed: {err}")
                    clip_inputs[i] = (None, None, 0.0, False)
    else:
        _, haud, _, h = probe_video_info(full_src_mp4)
        log(f"  -> Local/Full source verified at {h}p")
        for i, c in enumerate(selected, 1):
            final_mp4 = os.path.join(final_shorts_dir, f"short_{i:03d}.mp4")
            if os.path.exists(final_mp4) and os.path.getsize(final_mp4) > 10240:
                clip_inputs[i] = ("DONE", None, None, True)
            else:
                clip_inputs[i] = (full_src_mp4, c["start"], max(5.0, c["end"] - c["start"]), haud)

    stage1_tasks = []
    skipped_rows = {}
    for i, c in enumerate(selected, 1):
        info = clip_inputs.get(i)
        if not info or info[0] is None:
            skipped_rows[i] = {"clip": i, "score": c.get("score", 0), "start": c["start"], "end": c["end"], "duration": c["duration"], "words": 0, "title": c["title"], "file": "FAILED"}
            continue
        if info[0] == "DONE":
            skipped_rows[i] = {"clip": i, "score": c.get("score", 0), "start": c["start"], "end": c["end"], "duration": c["duration"], "words": -1, "title": c["title"], "file": f"short_{i:03d}.mp4"}
            continue
        src_path, seek_s, rdur, haud = info
        stage1_tasks.append((src_path, seek_s, rdur, i, local_work, enc_name, enc_args, args.layout, args.cam_pos, args.cam_w, args.cam_h, args.cam_x, args.cam_y, haud))

    stage1_ready = {}
    if stage1_tasks:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_stage1_trim_worker, t): t for t in stage1_tasks}
            for fut in as_completed(futs):
                idx, s1_path, el, err = fut.result()
                if s1_path:
                    stage1_ready[idx] = s1_path
                    log(f"  -> Stage 1 locked clip {idx}/{len(selected)} in {el:.1f}s")
                else:
                    c = selected[idx - 1]
                    log(f"  -> Stage 1 FAILED clip {idx}: {err}")
                    skipped_rows[idx] = {"clip": idx, "score": c.get("score", 0), "start": c["start"], "end": c["end"], "duration": c["duration"], "words": 0, "title": c["title"], "file": "FAILED"}

    # --- STAGE 2: Zero-seek Whisper + Dynamic SFX on locked stage1 clips ---
    need_subs = (effective_caption_style != "none")
    whisper_model = load_whisper_once() if (need_subs and stage1_ready) else None
    stage3_tasks = []

    for idx in sorted(stage1_ready.keys()):
        s1_path = stage1_ready[idx]
        c = selected[idx - 1]
        s1_dur, haud, _, _ = probe_video_info(s1_path)
        rdur = s1_dur or c["duration"]

        rel_segs = []
        if need_subs and haud:
            sub_cache = os.path.join(local_work, f"words_{idx:03d}.json")
            if os.path.exists(sub_cache):
                try:
                    with open(sub_cache, "r", encoding="utf-8") as f:
                        rel_segs = json.load(f)
                except Exception:
                    rel_segs = []
            if not rel_segs and whisper_model is not None:
                rel_segs = transcribe_stage1_exact(whisper_model, s1_path, os.path.join(local_work, f"w_{idx:03d}.wav"))
                if rel_segs:
                    with open(sub_cache, "w", encoding="utf-8") as f:
                        json.dump(rel_segs, f)
            if not rel_segs and scout_segs:
                rel_segs = [
                    {"start": max(0.0, s["start"] - c["start"]), "end": min(rdur, s["end"] - c["start"]), "text": s["text"]}
                    for s in scout_segs if s["end"] > c["start"] and s["start"] < c["end"]
                ]

        sfx_events = plan_clip_sfx_stage1(s1_path, rdur, rel_segs, sfx_pack) if (args.sfx and haud) else []
        stage3_tasks.append((
            s1_path, rdur, idx, c, local_work, local_shorts, final_shorts_dir,
            enc_name, enc_args, rel_segs, effective_caption_style, args.hook_style,
            args.layout, sfx_events, haud, font_family
        ))

    del whisper_model

    # --- STAGE 3: Burn synced captions/hook + mix dynamic SFX -> copy to Drive ---
    log(f"[4/5] Stage 3: Finalizing {len(stage3_tasks)} HD Shorts ({len(skipped_rows)} cached/skipped, {workers} workers, {enc_name})...")
    results_map = dict(skipped_rows)
    if stage3_tasks:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_stage3_finish_worker, t): (t[2], t[3]) for t in stage3_tasks}
            for fut in as_completed(futs):
                i, c = futs[fut]
                idx, fname, elapsed, wcount, sfx_n, err = fut.result()
                results_map[idx] = {
                    "clip": idx, "score": c.get("score", 0), "start": c["start"], "end": c["end"],
                    "duration": c["duration"], "words": wcount, "title": c["title"], "file": fname
                }
                if err:
                    log(f"  [{idx}/{len(selected)}] FAILED ({elapsed:.1f}s): {err}")
                else:
                    log(f"  [{idx}/{len(selected)}] SAVED -> {fname} | score={c.get('score',0)} | words={wcount} | sfx={sfx_n} | {elapsed:.1f}s | {c['title']}")

    rows = [results_map[i] for i in sorted(results_map.keys())]
    build_top_10_viral(final_session_dir, rows, local_work, font_family)
    log(f"\nDONE in {time.time() - t0:.1f}s total! Output: {final_session_dir}")

if __name__ == "__main__":
    main()

