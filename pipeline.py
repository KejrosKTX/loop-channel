import argparse
import datetime as dt
import json
import os
import pathlib
import random
import re
import shutil
import subprocess
import sys
import time

import requests

import blender_loop as BL

ROOT = pathlib.Path(__file__).resolve().parent
WORK = ROOT / "work"
STATE_FILE = ROOT / "state.json"
GEMINI = "https://generativelanguage.googleapis.com/v1beta"
MAX_FILE = 1_800_000_000  # GitHub release assets must stay under 2 GiB


def log(*a):
    print(*a, flush=True)


def sh(cmd, **kw):
    cmd = [str(c) for c in cmd]
    log("$", " ".join(cmd)[:300])
    return subprocess.run(cmd, check=True, **kw)


def ffmpeg(*args):
    sh(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args])


def load_config():
    return json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"videos": [], "last_update_id": 0}


def save_state(s):
    STATE_FILE.write_text(json.dumps(s, indent=1, ensure_ascii=False), encoding="utf-8")


def now():
    return dt.datetime.now(dt.timezone.utc)


# ============================================================================ Telegram


def tg(method, data=None, files=None, params=None):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        return None
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/{method}", data=data, files=files,
                          params=params, timeout=300)
        if not r.ok:
            log("Telegram error", method, r.status_code, r.text[:200])
            return None
        return r.json()
    except requests.RequestException as e:
        log("Telegram request failed:", e)
        return None


def say(text):
    log("TG>", text[:200].replace("\n", " "))
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not chat:
        return
    for i in range(0, len(text), 3800):
        tg("sendMessage", data={"chat_id": chat, "text": text[i:i + 3800], "disable_web_page_preview": "true"})


def send_file(kind, path, caption=""):
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not chat or not pathlib.Path(path).exists():
        return
    with open(path, "rb") as f:
        tg({"photo": "sendPhoto", "video": "sendVideo"}[kind],
           data={"chat_id": chat, "caption": caption[:900]}, files={kind: f})


def tg_download(file_id, dest):
    info = tg("getFile", data={"file_id": file_id})
    if not info:
        raise RuntimeError("Telegram getFile failed (file over 20 MB? send it as a normal video, not a file)")
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    url = f"https://api.telegram.org/file/bot{token}/{info['result']['file_path']}"
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)


# ============================================================================ Gemini (the AI "team")


def gemini_headers():
    return {"x-goog-api-key": os.environ["GEMINI_API_KEY"], "Content-Type": "application/json"}


def list_models():
    out, token = [], None
    while True:
        r = requests.get(f"{GEMINI}/models", headers=gemini_headers(),
                         params={"pageSize": 200, **({"pageToken": token} if token else {})}, timeout=60)
        r.raise_for_status()
        j = r.json()
        out += [m for m in j.get("models", []) if "generateContent" in m.get("supportedGenerationMethods", [])]
        token = j.get("nextPageToken")
        if not token:
            return out


def pick_models(cfg):
    """Newest text 'flash' model your key can use (Gemini 2.5 is being shut down, so don't hardcode)."""
    pinned = cfg.get("ai", {}).get("gemini_model", "auto")
    if pinned != "auto":
        return [pinned]
    cands = []
    try:
        for m in list_models():
            n = m["name"].split("/")[-1]
            if "flash" not in n or re.search(r"lite|image|tts|live|audio|embed|robot|computer|native|learnlm|gemma|vision|exp|omni|veo", n):
                continue
            v = re.search(r"gemini-([\d.]+)-flash", n)
            ver = tuple(int(x) for x in v.group(1).split(".") if x) if v else (0,)
            cands.append(((ver, "preview" not in n), n))
        cands.sort(reverse=True)
    except Exception as e:
        log("model listing failed:", e)
    names = [n for _, n in cands[:3]]
    names.append("gemini-flash-latest")
    return names


def gemini_json(prompt, models, temperature=1.0):
    last = "no model tried"
    for m in models:
        for attempt in range(3):
            r = requests.post(
                f"{GEMINI}/models/{m}:generateContent", headers=gemini_headers(), timeout=180,
                json={"contents": [{"parts": [{"text": prompt}]}],
                      "generationConfig": {"temperature": temperature, "responseMimeType": "application/json"}})
            if r.status_code == 200:
                try:
                    parts = r.json()["candidates"][0]["content"]["parts"]
                    text = "".join(p.get("text", "") for p in parts)
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        return json.loads(re.search(r"\{.*\}", text, re.S).group(0))
                except Exception as e:
                    last = f"{m}: bad JSON ({e})"
                    break
            if r.status_code in (429, 500, 503):
                last = f"{m}: HTTP {r.status_code}"
                time.sleep(15 * (attempt + 1))
                continue
            last = f"{m}: HTTP {r.status_code} {r.text[:200]}"
            break  # model unavailable for this key -> try next model
    raise RuntimeError("Gemini failed: " + last)


def fill(tpl, **kw):
    for k, v in kw.items():
        tpl = tpl.replace(f"<<{k}>>", str(v))
    return tpl


PRODUCER = """You are the creative director of the YouTube channel "<<channel>>".
Niche: <<niche>>
Create ONE new video concept. The video is a seamless, hypnotic 3D-style loop that will be repeated for about <<minutes>> minutes (people leave it running on a TV or monitor).
<<source_rules>>
Recent videos (do NOT repeat their concept or colour mood): <<recent>>
Return ONLY a JSON object with exactly these keys:
<<schema>>
Rules for the text fields:
- title: max 60 characters, evokes a feeling or curiosity, honest about what the video shows, no emojis, no ALL CAPS, no duration, no brand/artist names.
- description: 2-3 natural sentences using words like satisfying, 3D loop, relaxing, hypnotic. No false health claims.
- tags: 12-15 short search phrases people would actually type.
- hook_text: 1-3 words in capitals for the thumbnail."""

SRC_PROCEDURAL = """The animation is generated by a procedural script, so choose parameters from these options only:
- type: wave_grid (rippling grid of pillars/spheres), cube_flip_wave (cubes flipping in a diagonal wave), pendulum_wave (row of swinging balls forming wave patterns), ring_gyro (nested spinning rings with beads), spiral_ripple (sunflower spiral of spheres with outward ripples)
- shape (wave_grid only): cube | sphere | cylinder
- material: glossy | matte | metallic | neon
- camera: front_iso | top_down | low_angle | close
- grid: integer 6-21 (density); speed: integer 1-3
- palette: 3 hex colours that look great together; background: a very dark hex colour"""

SCHEMA_PROCEDURAL = ('{"scene": {"type": "...", "shape": "...", "material": "...", "camera": "...", "grid": 13, '
                     '"speed": 1, "palette": ["#..", "#..", "#.."], "background": "#.."}, "title": "...", '
                     '"description": "...", "tags": ["..."], "hook_text": "..."}')

SRC_MANUAL = """The footage comes from an AI text-to-video model (6-8 second clip, 16:9, static camera). It will be made seamless by playing it forward then backward, so the motion must look natural in reverse: oscillating, rotating, pulsing, rippling or swinging motion - NOT falling, pouring or one-directional flow.
Write a video_prompt of 60-100 words: one clear subject (e.g. glossy cubes, liquid-metal rings, glass spheres, pendulum balls), exact material and colours, soft studio lighting with reflections on a dark glossy floor, smooth continuous motion, static camera, 3D render look, no text, no logos, no people."""

SCHEMA_MANUAL = ('{"video_prompt": "...", "title": "...", "description": "...", "tags": ["..."], '
                 '"hook_text": "..."}')

CRITIC = """You are a strict YouTube packaging and policy reviewer. Review this video package.
What the video actually shows: <<shows>>
Check: the title is honest about that content and creates curiosity or emotion; no clickbait lies; no copyrighted names, artists or brands; no misleading claims (e.g. medical or sleep cures); tags are relevant, not spam; description reads naturally.
Return ONLY JSON: {"score": <0-10>, "issues": ["..."], "package": <the improved package, same keys as the input, same keys only>}.
Package:
<<package>>"""

FALLBACK_SCENES = {
    "wave_grid": ("Hypnotic Ripple Wave", ["ripple", "wave grid", "pillars"]),
    "cube_flip_wave": ("Cubes Flipping in Perfect Sync", ["cube flip", "domino wave", "cubes"]),
    "pendulum_wave": ("Pendulum Wave Pattern", ["pendulum wave", "pendulum", "physics"]),
    "ring_gyro": ("Endless Spinning Rings", ["spinning rings", "gyroscope", "rings"]),
    "spiral_ripple": ("Golden Spiral Ripple", ["spiral", "golden ratio", "ripple"]),
}
FALLBACK_PALETTES = [
    (["#ff5ea8", "#ffb347", "#4cc9f0"], "#0b0d14"), (["#7cf5c8", "#3a86ff", "#c77dff"], "#080b12"),
    (["#ff9f1c", "#ff4d6d", "#ffd6a5"], "#120a0a"), (["#2ec4b6", "#e9ff70", "#ffffff"], "#06100f"),
    (["#9d4edd", "#ff7aa2", "#5eead4"], "#0d0814"), (["#f72585", "#4361ee", "#4cc9f0"], "#070812"),
]
FALLBACK_PROMPTS = {
    "wave_grid": "Soft clay pillars in {c1} and {c2} rising and falling in an outward ripple, matte material, dark background, gentle lighting, static camera, calm satisfying 3D render, no text",
    "cube_flip_wave": "A grid of glossy {c1} cubes on a dark reflective floor, flipping over one after another in a smooth diagonal wave, soft studio lighting, sharp reflections, static camera, calm hypnotic 3D render, no text",
    "pendulum_wave": "A long row of shiny {c1} and {c2} pendulum balls swinging in a mesmerizing wave pattern, dark glossy studio background, soft rim light, static camera, smooth continuous motion, 3D render, no text",
    "ring_gyro": "Concentric rings of polished {c1} metal rotating at different speeds like a gyroscope, dark background with soft reflections, slow smooth motion, static camera, satisfying 3D render, no text",
    "spiral_ripple": "A field of glass-like {c1} spheres rippling up and down in a spiral wave pattern, dark reflective floor, soft studio light, static camera, hypnotic smooth motion, 3D render, no text",
}
COLOR_NAMES = ["pink", "orange", "cyan", "mint", "blue", "violet", "coral", "lime", "gold", "teal"]


def fallback_package(state, source):
    """No AI available (key missing / quota hit): still produce something so the channel never stalls."""
    used = [v.get("scene_type") for v in state["videos"][-8:]]
    types = sorted(BL.SCENE_TYPES, key=lambda t: (used.count(t), random.random()))
    t = types[0]
    pal, bg = random.choice(FALLBACK_PALETTES)
    base, kw = FALLBACK_SCENES[t]
    pkg = {
        "title": f"{base} | Satisfying 3D Loop",
        "description": f"{base} in a seamless 3D loop. Put it on a big screen to relax, focus or unwind.",
        "tags": ["satisfying", "3d loop", "relaxing", "hypnotic", "seamless loop", "oddly satisfying",
                 "relaxing visuals", "background video", "focus", "calming", "abstract 3d"] + kw,
        "hook_text": "SO SATISFYING",
        "scene": {"type": t, "palette": pal, "background": bg, "material": random.choice(BL.MATERIALS),
                  "shape": random.choice(BL.SHAPES), "camera": random.choice(BL.CAMERAS),
                  "grid": random.randint(9, 17), "speed": random.randint(1, 2)},
    }
    if source == "manual_ai":
        c = random.sample(COLOR_NAMES, 2)
        pkg["video_prompt"] = FALLBACK_PROMPTS[t].format(c1=c[0], c2=c[1])
        pkg["scene"] = {"type": "ai_clip"}
    return pkg


def sanitize(pkg, source):
    title = re.sub(r"\s+", " ", str(pkg.get("title", "")).replace("<", "").replace(">", "")).strip()
    if len(title) > 60:
        title = title[:60].rsplit(" ", 1)[0]
    tags, total = [], 0
    for t in pkg.get("tags") or []:
        t = re.sub(r"[<>\"]", "", str(t)).strip().lower()
        if t and t not in tags and total + len(t) + 3 <= 450:
            tags.append(t)
            total += len(t) + 3
    out = {
        "title": title or "Satisfying 3D Loop",
        "description": str(pkg.get("description", "")).replace("<", "").replace(">", "").strip()[:1500],
        "tags": tags,
        "hook_text": " ".join(str(pkg.get("hook_text", "SATISFYING")).upper().split()[:3])[:24] or "SATISFYING",
    }
    if source == "manual_ai":
        out["video_prompt"] = str(pkg.get("video_prompt", "")).strip()[:1200]
        out["scene"] = {"type": "ai_clip"}
    else:
        out["scene"] = BL.clean_scene_params(pkg.get("scene"))
    return out


def make_package(cfg, state, source, offline=False):
    if offline or not os.environ.get("GEMINI_API_KEY"):
        log("No Gemini key (or --offline): using the built-in fallback generator")
        return sanitize(fallback_package(state, source), source), "fallback"
    try:
        models = pick_models(cfg)
        log("Gemini models:", models)
        recent = "; ".join(v["title"] for v in state["videos"][-25:]) or "(none yet)"
        prompt = fill(PRODUCER, channel=cfg["channel_name"], niche=cfg["niche"],
                      minutes=cfg["video"]["target_minutes"], recent=recent,
                      source_rules=SRC_MANUAL if source == "manual_ai" else SRC_PROCEDURAL,
                      schema=SCHEMA_MANUAL if source == "manual_ai" else SCHEMA_PROCEDURAL)
        pkg = sanitize(gemini_json(prompt, models), source)
        for rnd in range(int(cfg.get("ai", {}).get("critic_rounds", 2))):
            shows = pkg.get("video_prompt") or json.dumps(pkg["scene"])
            review_in = {k: v for k, v in pkg.items() if k != "scene"}
            review = gemini_json(fill(CRITIC, shows=shows, package=json.dumps(review_in)), models, 0.4)
            score = review.get("score", 0)
            log(f"Critic round {rnd + 1}: score {score}; issues {review.get('issues')}")
            better = review.get("package")
            if isinstance(better, dict):
                merged = {**pkg, **{k: better[k] for k in ("title", "description", "tags", "hook_text", "video_prompt")
                                    if k in better}}
                pkg = sanitize(merged, source)
                if source != "manual_ai":
                    pkg["scene"] = BL.clean_scene_params(pkg["scene"])
            if isinstance(score, (int, float)) and score >= 8:
                break
        return pkg, "gemini"
    except Exception as e:
        log("AI step failed, using fallback:", e)
        say(f"AI step failed ({str(e)[:150]}). Used the built-in fallback instead.")
        return sanitize(fallback_package(state, source), source), "fallback"


# ============================================================================ video building


def render_blender(pkg, work, v):
    frames = int(round(v["loop_seconds"] * v["fps"]))
    fdir = work / "frames"
    shutil.rmtree(fdir, ignore_errors=True)
    (work / "package.json").write_text(json.dumps(pkg), encoding="utf-8")
    sh([os.environ.get("BLENDER_BIN", "blender"), "-b", "--factory-startup", "-noaudio", "--python",
        ROOT / "blender_loop.py", "--", work / "package.json", fdir, v["width"], v["height"], v["fps"],
        frames, v["samples"]])
    got = len(list(fdir.glob("frame_*.png")))
    if got != frames:
        raise RuntimeError(f"Blender produced {got}/{frames} frames")
    master = work / "master.mp4"
    ffmpeg("-framerate", v["fps"], "-i", fdir / "frame_%04d.png", "-c:v", "libx264", "-preset", "slow",
           "-crf", v["crf"], "-pix_fmt", "yuv420p", "-r", v["fps"], "-an", "-movflags", "+faststart", master)
    shutil.rmtree(fdir, ignore_errors=True)
    return master


def probe(path, entry):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                        f"stream={entry}", "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True)
    return r.stdout.strip()


def clip_to_loop(clip, work, v):
    """AI clip -> seamless loop: normalise size/fps, then play forward + backward (boomerang)."""
    norm = work / "norm.mp4"
    w, h, fps = v["width"], v["height"], v["fps"]
    ffmpeg("-i", clip, "-t", 12, "-an", "-vf",
           f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},fps={fps},format=yuv420p",
           "-c:v", "libx264", "-crf", 14, "-preset", "fast", norm)
    n = int(probe(norm, "nb_read_frames") or 0)
    if n < 12:
        raise RuntimeError(f"clip too short ({n} frames)")
    master = work / "master.mp4"
    # forward F0..F(n-1), then backward F(n-2)..F1 -> loops back to F0 without a repeated frame
    ffmpeg("-i", norm, "-filter_complex",
           f"[0:v]split[a][b];[b]trim=start_frame=1:end_frame={n - 1},setpts=PTS-STARTPTS,reverse[r];"
           f"[a][r]concat=n=2:v=1:a=0,fps={fps}[v]",
           "-map", "[v]", "-c:v", "libx264", "-preset", "slow", "-crf", v["crf"], "-pix_fmt", "yuv420p",
           "-an", "-movflags", "+faststart", master)
    return master, (2 * n - 2) / fps


def make_audio(out, seconds, seed):
    """Soft synthesized ambient drone (no copyrighted music, no download)."""
    f0 = random.Random(seed).choice([110.0, 123.47, 130.81, 146.83])
    lav = ["-f", "lavfi", "-i"]
    ffmpeg(*lav, f"sine=f={f0}:r=44100", *lav, f"sine=f={f0 * 1.5}:r=44100", *lav, f"sine=f={f0 * 2}:r=44100",
           *lav, f"sine=f={f0 * 3}:r=44100", *lav, "anoisesrc=color=brown:amplitude=0.25:r=44100",
           "-filter_complex",
           "[0]volume=0.25,tremolo=f=0.1:d=0.5[a];[1]volume=0.18,tremolo=f=0.13:d=0.6[b];"
           "[2]volume=0.12,tremolo=f=0.17:d=0.5[c];[3]volume=0.06,tremolo=f=0.11:d=0.7[d];"
           "[4]lowpass=f=500,volume=0.5[n];[a][b][c][d][n]amix=inputs=5:normalize=0,"
           "aecho=0.8:0.7:60|120:0.3|0.2,lowpass=f=2400[o]",
           "-map", "[o]", "-t", seconds, "-c:a", "aac", "-b:a", "128k", out)


def make_thumbnail(master, loop_seconds, text, out):
    from PIL import Image, ImageDraw, ImageFont
    raw = out.with_suffix(".png")
    ffmpeg("-ss", loop_seconds * 0.3, "-i", master, "-frames:v", 1, raw)
    img = Image.open(raw).convert("RGB").resize((1280, 720))
    shade = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(shade)
    for y in range(360, 720):
        d.line([(0, y), (1280, y)], fill=(0, 0, 0, int(190 * (y - 360) / 360)))
    img = Image.alpha_composite(img.convert("RGBA"), shade).convert("RGB")
    font = None
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVuSans-Bold.ttf"):
        try:
            font = ImageFont.truetype(p, 120)
            break
        except OSError:
            continue
    font = font or ImageFont.load_default()
    ImageDraw.Draw(img).text((60, 560), text, font=font, fill="white", stroke_width=6, stroke_fill="black")
    img.save(out, quality=90)
    raw.unlink(missing_ok=True)


def finalize(entry, master, loop_seconds, cfg, work, no_release=False):
    v = cfg["video"]
    per_sec = master.stat().st_size / loop_seconds + 16000
    total = min(v["target_minutes"] * 60, MAX_FILE / per_sec)
    loops = max(1, int(total // loop_seconds))
    total = loops * loop_seconds
    log(f"Long video: {loops} loops x {loop_seconds:.1f}s = {total / 60:.1f} min")
    audio, final = work / "audio.m4a", work / "final.mp4"
    make_audio(audio, total, entry["id"])
    ffmpeg("-stream_loop", loops - 1, "-i", master, "-i", audio, "-map", "0:v", "-map", "1:a", "-c", "copy",
           "-t", total, "-movflags", "+faststart", final)
    preview = work / "preview.mp4"
    ffmpeg("-i", master, "-vf", "scale=-2:720", "-c:v", "libx264", "-crf", 28, "-preset", "veryfast", "-an",
           "-t", 20, preview)
    thumb = work / "thumbnail.jpg"
    make_thumbnail(master, loop_seconds, entry["hook_text"], thumb)

    minutes = max(1, round(total / 60))
    suffix = f" | {minutes} Min"
    entry["minutes"] = minutes
    if len(entry["title"]) + len(suffix) <= 100:
        entry["title"] += suffix
    entry["description"] = (
        f"{entry['description']}\n\nA {minutes}-minute seamless 3D loop - play it on a TV or monitor to relax, "
        "focus or unwind. The ambient sound is a simple synthesized drone made for this video.\n\n"
        "New loops every few days - subscribe so you don't miss them.\n\n#satisfying #3dloop #relaxing")
    entry["status"] = "pending"

    if not no_release and os.environ.get("GITHUB_REPOSITORY"):
        sh(["gh", "release", "create", entry["tag"], final, thumb, preview, "--title", entry["title"],
            "--notes", f"{entry['title']}\n\n{entry['description']}\n\nTags: {', '.join(entry['tags'])}"])
        entry["release_url"] = f"https://github.com/{os.environ['GITHUB_REPOSITORY']}/releases/tag/{entry['tag']}"
    return preview, thumb, final


def announce_ready(entry, preview, thumb, final):
    send_file("photo", thumb, "Thumbnail")
    send_file("video", preview, f"Preview of the loop ({entry['minutes']} min version is ready)")
    say(f"NEW VIDEO READY: {entry['id']}\n\nTITLE:\n{entry['title']}")
    say(f"DESCRIPTION:\n{entry['description']}")
    say(f"TAGS:\n{', '.join(entry['tags'])}")
    where = entry.get("release_url") or str(final)
    say(f"FULL VIDEO ({entry['minutes']} min) - download from the GitHub release, then upload in YouTube/Studio and "
        f"schedule it:\n{where}\n\nIf YouTube API publishing is set up, reply /approve to schedule it automatically, "
        "or /reject to discard.")


# ============================================================================ commands


def cmd_create(a):
    cfg = load_config()
    quick = a.quick or os.environ.get("QUICK", "").lower() == "true"
    if quick:
        cfg["video"].update(width=640, height=360, loop_seconds=4, samples=8, target_minutes=1)
        log("QUICK TEST MODE: small and fast")
    state = load_state()
    source = cfg.get("video_source", "procedural")
    vid = now().strftime("%Y%m%d-%H%M")
    work = WORK / vid
    work.mkdir(parents=True, exist_ok=True)

    pkg, how = make_package(cfg, state, source, a.offline)
    entry = {"id": vid, "tag": f"video-{vid}", "created": now().isoformat(timespec="seconds"), "source": source,
             "scene_type": pkg["scene"].get("type"), "ideas_by": how, **{k: pkg[k] for k in (
                 "title", "description", "tags", "hook_text")}, "status": "awaiting_clip"}
    log("TITLE:", entry["title"])

    if source == "manual_ai":
        entry["video_prompt"] = pkg["video_prompt"]
        state["videos"].append(entry)
        save_state(state)
        say(f"NEW VIDEO IDEA {vid}: {entry['title']}\n\nGenerate a 6-8 second 16:9 clip with this prompt in any AI video "
            "tool, then send the clip to this chat as a normal VIDEO (not as a file, max 20 MB). I will loop it, "
            "build the long video and send it back for approval.")
        say(entry["video_prompt"])
        return

    master = render_blender(pkg, work, cfg["video"])
    preview, thumb, final = finalize(entry, master, cfg["video"]["loop_seconds"], cfg, work, a.no_release)
    state["videos"].append(entry)
    save_state(state)
    announce_ready(entry, preview, thumb, final)


def cmd_models(_):
    cfg = load_config()
    log("Models your key can use:", [m["name"] for m in list_models()])
    log("Would pick:", pick_models(cfg))


# ---------------------------------------------------------------------------- YouTube (Phase 2)


def youtube_client():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    creds = Credentials(None, refresh_token=os.environ["YT_REFRESH_TOKEN"], token_uri="https://oauth2.googleapis.com/token",
                        client_id=os.environ["YT_CLIENT_ID"], client_secret=os.environ["YT_CLIENT_SECRET"],
                        scopes=["https://www.googleapis.com/auth/youtube"])
    creds.refresh(Request())
    return build("youtube", "v3", credentials=creds, cache_discovery=False)


def next_slot(cfg, state):
    times = cfg["upload"].get("publish_times_utc", ["16:00"])
    taken = [dt.datetime.fromisoformat(v["publish_at"].replace("Z", "+00:00")) for v in state["videos"]
             if v.get("publish_at")]
    earliest = max([now() + dt.timedelta(hours=1)] + taken)
    for day in range(0, 60):
        d = (now() + dt.timedelta(days=day)).date()
        for t in sorted(times):
            h, m = map(int, t.split(":"))
            cand = dt.datetime(d.year, d.month, d.day, h, m, tzinfo=dt.timezone.utc)
            if cand > earliest:
                return cand
    return earliest + dt.timedelta(days=1)


def youtube_upload(entry, work, cfg, publish_at):
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload
    yt = youtube_client()
    body = {"snippet": {"title": entry["title"], "description": entry["description"], "tags": entry["tags"],
                        "categoryId": cfg["upload"].get("category_id", "24"),
                        "defaultLanguage": cfg.get("language", "en"), "defaultAudioLanguage": cfg.get("language", "en")},
            "status": {"privacyStatus": "private", "publishAt": publish_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                       "selfDeclaredMadeForKids": bool(cfg["upload"].get("made_for_kids", False))}}
    media = MediaFileUpload(str(work / "final.mp4"), mimetype="video/mp4", chunksize=16 * 1024 * 1024, resumable=True)
    req = yt.videos().insert(part="snippet,status", body=body, media_body=media)
    resp, tries = None, 0
    while resp is None:
        try:
            status, resp = req.next_chunk()
            if status:
                log(f"upload {int(status.progress() * 100)}%")
        except HttpError as e:
            tries += 1
            if e.resp.status not in (500, 502, 503, 504) or tries > 6:
                raise
            time.sleep(5 * tries)
    try:
        yt.thumbnails().set(videoId=resp["id"], media_body=MediaFileUpload(str(work / "thumbnail.jpg"))).execute()
    except Exception as e:  # custom thumbnails need a phone-verified channel
        log("thumbnail not set:", e)
    return resp["id"]


def approve(entry, cfg, state):
    if not os.environ.get("YT_REFRESH_TOKEN"):
        say("YouTube API publishing is not set up. Download the video from the release link, upload it in "
            "YouTube/Studio and schedule it there.")
        return
    work = WORK / entry["id"]
    work.mkdir(parents=True, exist_ok=True)
    sh(["gh", "release", "download", entry["tag"], "-p", "final.mp4", "-p", "thumbnail.jpg", "-D", work, "--clobber"])
    slot = next_slot(cfg, state)
    vid = youtube_upload(entry, work, cfg, slot)
    entry.update(status="published", youtube_id=vid, publish_at=slot.strftime("%Y-%m-%dT%H:%M:%SZ"))
    say(f"Uploaded and scheduled for {slot:%Y-%m-%d %H:%M} UTC\nhttps://youtu.be/{vid}\n"
        "(If the link shows the video as locked/private after the time, your API project has not passed Google's audit.)")


def pick_entry(state, status, arg=None):
    pool = [v for v in state["videos"] if v["status"] == status and (not arg or v["id"] == arg)]
    return pool[-1] if pool else None


def handle_clip(msg, cfg, state):
    media = msg.get("video") or msg.get("animation") or (
        msg.get("document") if str(msg.get("document", {}).get("mime_type", "")).startswith("video") else None)
    entry = pick_entry(state, "awaiting_clip", (msg.get("caption") or "").strip() or None) or pick_entry(
        state, "awaiting_clip")
    if not entry:
        say("I received a video but no idea is waiting for a clip.")
        return
    work = WORK / entry["id"]
    work.mkdir(parents=True, exist_ok=True)
    say(f"Got the clip for {entry['id']}. Building the loop and the long video...")
    tg_download(media["file_id"], work / "clip.mp4")
    master, loop_seconds = clip_to_loop(work / "clip.mp4", work, cfg["video"])
    preview, thumb, final = finalize(entry, master, loop_seconds, cfg, work)
    announce_ready(entry, preview, thumb, final)


def cmd_poll(_):
    cfg = load_config()
    state = load_state()
    chat = str(os.environ["TELEGRAM_CHAT_ID"])
    res = tg("getUpdates", params={"offset": state["last_update_id"] + 1, "timeout": 0})
    changed = False
    for u in (res or {}).get("result", []):
        state["last_update_id"] = max(state["last_update_id"], u["update_id"])
        msg = u.get("message") or {}
        if str(msg.get("chat", {}).get("id")) != chat:
            continue  # ignore everyone except you
        text = (msg.get("text") or "").strip()
        try:
            if msg.get("video") or msg.get("animation") or msg.get("document"):
                changed = True
                handle_clip(msg, cfg, state)
            elif text.startswith("/approve"):
                changed = True
                arg = text.split()[1] if len(text.split()) > 1 else None
                e = pick_entry(state, "pending", arg)
                approve(e, cfg, state) if e else say("Nothing is waiting for approval.")
            elif text.startswith("/reject"):
                changed = True
                arg = text.split()[1] if len(text.split()) > 1 else None
                e = pick_entry(state, "pending", arg)
                if e:
                    e["status"] = "rejected"
                    subprocess.run(["gh", "release", "delete", e["tag"], "--yes", "--cleanup-tag"])
                    say(f"Discarded {e['id']}.")
                else:
                    say("Nothing is waiting.")
            elif text.startswith("/status"):
                changed = True
                say("\n".join(f"{v['id']} [{v['status']}] {v['title']}" for v in state["videos"][-6:]) or "No videos yet.")
        except Exception as e:  # report instead of crashing, so one failure doesn't block the queue
            changed = True
            say(f"Error: {str(e)[:300]}")
    if changed:
        save_state(state)


def cmd_token(_):
    cid = os.environ.get("YT_CLIENT_ID") or input("OAuth client ID: ").strip()
    secret = os.environ.get("YT_CLIENT_SECRET") or input("OAuth client secret: ").strip()
    r = requests.post("https://oauth2.googleapis.com/device/code",
                      data={"client_id": cid, "scope": "https://www.googleapis.com/auth/youtube"}).json()
    if "user_code" not in r:
        sys.exit(f"Google refused: {r}")
    print(f"\nOpen {r['verification_url']} and enter the code: {r['user_code']}\n(sign in with the Google account that owns the channel)\n")
    interval, end = r.get("interval", 5), time.time() + r.get("expires_in", 1800)
    while time.time() < end:
        time.sleep(interval)
        t = requests.post("https://oauth2.googleapis.com/token", data={
            "client_id": cid, "client_secret": secret, "device_code": r["device_code"],
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code"}).json()
        if "refresh_token" in t:
            print("\nYOUR REFRESH TOKEN (store it as the GitHub secret YT_REFRESH_TOKEN, show it to nobody):\n")
            print(t["refresh_token"])
            return
        if t.get("error") == "slow_down":
            interval += 5
        elif t.get("error") != "authorization_pending":
            sys.exit(f"Failed: {t}")
    sys.exit("Timed out - run again.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("--quick", action="store_true")
    c.add_argument("--offline", action="store_true")
    c.add_argument("--no-release", action="store_true")
    c.set_defaults(fn=cmd_create)
    sub.add_parser("poll").set_defaults(fn=cmd_poll)
    sub.add_parser("models").set_defaults(fn=cmd_models)
    sub.add_parser("token").set_defaults(fn=cmd_token)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
