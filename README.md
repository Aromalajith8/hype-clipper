# HypeClipper

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aromalajith8/hype-clipper/blob/main/Colab_Runner.ipynb)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FFmpeg](https://img.shields.io/badge/FFmpeg-1080x1920%20NVENC%20%7C%20QSV%20%7C%20x264-brightgreen)
![Cost](https://img.shields.io/badge/Cost-100%25%20Free-success)

**HypeClipper** converts 40-minute to **10+ hour streams and VODs** into upload-ready **1080x1920 (9:16) vertical Shorts** with hook-to-payoff clipping (`25s-50s`), word-synced burned-in captions, customizable facecam framing, stereo intro SFX, and an automatic **Top 10 Viral** picks folder with thumbnails.

---
<img width="2560" height="1440" alt="Logo 16 9@2x" src="https://github.com/user-attachments/assets/d859df14-d7d2-4426-88f4-0eade189deb9" />
---
## Key Features

- **10+ Hour VOD Ready (Surgical Range Fetch):** Automatically scouts long YouTube VODs using Replay Heatmaps, YouTube captions, and a zero-disk streaming NumPy audio scan—then downloads **only** the selected clips in 1080p instead of a 30 GB full stream.
- **0.00s Frame-Locked Subtitles:** Runs `faster-whisper` with `word_timestamps=True` directly on each clip and forces `setpts=PTS-STARTPTS` + local bundled fonts via `libass` (`-filter_complex_script`) so subtitles never drift or disappear.
- **8-Second Visual Calibration Preview:** Generates `preview_source_cambox.jpg` (green overlay showing your exact facecam crop) and `preview_rendered_short.jpg` (actual 1080x1920 Short frame with hook + subtitle) before rendering a full batch.
- **6 Vertical 9:16 Layouts + Custom Facecam Positioning:** Supports stacked facecam (`cam-top`, `cam-bottom`), floating PiP (`pip-top-right`, `pip-top-left`), full-screen `crop-focus`, and centered `blur-bg` with presets (`mid-left`, `top-left`, etc.) or custom `X/Y/W/H` coordinates.
- **15+ Viral Caption Styles & 4 Hook Banners:** Includes active-word highlighting (`beast-bold`, `bold-pop`, `karaoke-highlight`, `word-by-word`, `two-line-stack`, `highlight-box`, `neon-glow`, `comic-burst`, etc.) plus split-seam `mid-card`, `bold-box`, `outline-glow`, and `top-bar` hook banners.
- **Top 10 Viral Folder + Thumbnails:** Ranks all clips by AI viral score + audio/replay energy, copies the top 10 into `top_10_viral/`, and exports crisp `1080x1920` `.jpg` thumbnails.
- **Built-in Copyright-Free Hook SFX:** Synthesizes a stereo 44.1 kHz whoosh + sub-pop intro sound locally via NumPy and mixes it with `normalize=0` (no voice ducking) to boost retention and audio uniqueness.
- **Auto-Resume & Parallel Rendering:** Hardware-accelerated (`h264_nvenc` on Colab T4/NVIDIA, `h264_qsv` on Intel, `libx264` fallback) across `ProcessPoolExecutor` workers, with automatic Drive checkpointing (`selected_clips.json`) if a Colab session disconnects.

---

## Quick Start: Google Colab (Recommended — Free T4 GPU)

1. Click **[Open in Colab](https://colab.research.google.com/github/Aromalajith8/hype-clipper/blob/main/Colab_Runner.ipynb)**.
2. Go to **Runtime → Change runtime type → T4 GPU**.
3. Run **Step 1** (mount Google Drive) and **Step 2** (install FFmpeg, fonts, and dependencies).
4. In **Step 3**, paste your YouTube stream URL (or local Drive `.mp4` path) and optionally add a free **Groq** (`console.groq.com`) or **Gemini** API key—or leave blank for the zero-key Hook + Energy scorer.
5. In **Step 4**, pick your layout, facecam position, caption style, and hook style using the built-in visual gallery.
6. Run **Step 4.5** to verify your facecam box and burned-in subtitle in ~8 seconds.
7. Run **Step 5** to generate 40+ Shorts into `MyDrive/HypeClipper/`, then **Step 6** to preview the top viral clips.

---

## Local Installation (Windows / Linux / macOS)

### Prerequisites
- **Python 3.10+**
- **FFmpeg & FFprobe** installed and available in `PATH` (with `libass` support)
  - Windows: `winget install Gyan.FFmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg fonts-dejavu-core fontconfig`

### Setup
```bash
git clone [https://github.com/Aromalajith8/hype-clipper.git](https://github.com/Aromalajith8/hype-clipper.git)
cd hype-clipper
pip install -U yt-dlp faster-whisper numpy requests
```

### Optional Free AI Keys
```powershell
# Windows PowerShell
$env:GROQ_API_KEY="gsk_your_key_here"
# or
$env:GEMINI_API_KEY="AIza_your_key_here"
```

```bash
# Linux / macOS
export GROQ_API_KEY="gsk_your_key_here"
# or
export GEMINI_API_KEY="AIza_your_key_here"
```

---

## CLI Usage

### 1. Run an 8-Second Facecam & Subtitle Preview First
```bash
python clipper.py "[https://youtu.be/VIDEO_ID](https://youtu.be/VIDEO_ID)" \
  --preview-layout --preview-sec 300 \
  --layout cam-top --cam-pos mid-left \
  --caption-style beast-bold --hook-style mid-card
```

### 2. Generate 40+ Viral 1080p Shorts (Works on 10+ Hour VODs)
```bash
python clipper.py "[https://youtu.be/VIDEO_ID](https://youtu.be/VIDEO_ID)" \
  --clips 45 --min-len 25 --max-len 50 --min-height 1080 \
  --fetch-mode auto \
  --layout cam-top --cam-pos mid-left --cam-w 0.26 --cam-h 0.38 \
  --caption-style beast-bold --hook-style mid-card --sfx \
  --out ./HypeClipper
```

### 3. Local Video File
```bash
python clipper.py "D:\Streams\my_10hr_vod.mp4" \
  --clips 40 --layout cam-top --cam-pos custom \
  --cam-x 0.00 --cam-y 0.38 --cam-w 0.26 --cam-h 0.38 \
  --caption-style bold-pop --hook-style mid-card --sfx
```

---

## Configuration Reference

### Layouts (`--layout`)
| Layout | Description |
|---|---|
| `cam-top` | Cropped facecam (`1080x720`) stacked above vertical gameplay (`1080x1200`) |
| `cam-bottom` | Vertical gameplay (`1080x1200`) stacked above cropped facecam (`1080x720`) |
| `pip-top-right` | Full-screen `1080x1920` gameplay with framed `360x360` facecam box top-right |
| `pip-top-left` | Full-screen `1080x1920` gameplay with framed `360x360` facecam box top-left |
| `crop-focus` | Full-screen `1080x1920` center-focused vertical crop |
| `blur-bg` | Full 16:9 frame centered over a blurred, dimmed 9:16 background |

### Facecam Positioning (`--cam-pos`)
- **Presets:** `mid-left` (default), `mid-right`, `top-left`, `top-right`, `bottom-left`, `bottom-right`
- **Fine Control:** Use `--cam-pos custom` with normalized `0.0-1.0` coordinates: `--cam-x`, `--cam-y`, `--cam-w`, `--cam-h`

### Caption Styles (`--caption-style`)
`auto` · `bold-pop` · `beast-bold` · `karaoke-highlight` · `word-by-word` · `two-line-stack` · `highlight-box` · `neon-glow` · `impact-shake` · `emoji-pop` · `typewriter` · `bounce` · `gradient-stroke` · `comic-burst` · `minimal-clean` · `gaming-glitch` · `none`

### Hook Styles (`--hook-style`)
- `mid-card` — White card banner anchored across the split seam (`y=720` on `cam-top`)
- `bold-box` — High-contrast yellow banner box
- `outline-glow` — Bold white text with glowing yellow outline
- `top-bar` — Breaking-news style red bar banner
- `none` — Disable hook title banner

### Fetch & AI Flags
| Flag | Values | Purpose |
|---|---|---|
| `--fetch-mode` | `auto` (default), `surgical`, `full` | Auto switches to surgical clip fetching for YouTube streams `>= 45 min` |
| `--ai-provider` | `auto`, `groq`, `gemini`, `ollama`, `heuristic` | Free AI hook/payoff ranker with automatic model fallback |
| `--cookies` | `/path/to/cookies.txt` | Optional YouTube cookies if Colab hits bot/age-gate checks |
| `--sfx` | flag | Mixes stereo whoosh + sub-pop intro SFX into every clip |

---

## Output Folder Structure

```text
HypeClipper/<session_id>/
├── shorts/                        # All rendered 1080x1920 Shorts (short_001.mp4 ...)
├── top_10_viral/                  # Top 10 highest-scoring clips + 1080x1920 thumbnails
│   ├── rank_01_short_012.mp4
│   ├── rank_01_short_012_thumb.jpg
│   └── ...
├── transcript/
│   ├── clips.json                 # Full clip metadata, scores, timestamps, word counts
│   └── clips.csv                  # Spreadsheet-ready clip schedule
├── report/
│   └── index.html                 # Visual HTML dashboard with Top 10 thumbnails & clip table
└── selected_clips.json            # Auto-resume checkpoint
```

---

## Tips for 10+ Hour Streams

- **Prefer YouTube URLs over uploading raw 25 GB files to Drive:** Free Google Drive caps at 15 GB. If you recorded locally, either run `clipper.py` directly on your PC or upload the VOD as an **Unlisted** YouTube video and paste that URL into Colab with `--fetch-mode auto`.
- **Always run Step 4.5 / `--preview-layout` first:** Checking one frame takes ~8 seconds and guarantees your facecam framing and caption placement are locked in before rendering 40+ clips.
- **YouTube bot checks on Colab:** Export a `cookies.txt` file to `MyDrive/cookies.txt` and pass it via `cookies_path` / `--cookies`.

---

## License

MIT License — free to use, modify, and build upon.
