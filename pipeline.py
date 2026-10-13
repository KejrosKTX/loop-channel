import argparse
import datetime as dt
import json
import os
import pathlib
import random
import re
import subprocess
import time

import requests

ROOT = pathlib.Path(__file__).resolve().parent
WORK = ROOT / "work"
STATE_FILE = ROOT / "bot_state.json"
GEMINI = "https://generativelanguage.googleapis.com/v1beta"
W, H, FPS = 1080, 1920, 30
BOT_LIMIT = 49_000_000  # Telegram bots can send files up to 50 MB


def log(*a):
    print(*a, flush=True)


def now():
    return dt.datetime.now(dt.timezone.utc)


def ffmpeg(*args):
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *[str(a) for a in args]]
    log("$", " ".join(cmd)[:250])
    subprocess.run(cmd, check=True)


def probe(path, entry):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                        f"stream={entry}", "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True)
    return r.stdout.strip()


def load_config():
    return json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"entries": [], "last_update_id": 0, "last_slot": ""}


def save_state(s):
    STATE_FILE.write_text(json.dumps(s, indent=1, ensure_ascii=False), encoding="utf-8")


# ============================================================================ Telegram


def tg(method, data=None, params=None, files=None):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        return None
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/{method}", data=data, params=params, files=files,
                          timeout=300)
        if not r.ok:
            log("Telegram error", method, r.status_code, r.text[:200])
            return None
        return r.json()
    except requests.RequestException as e:
        log("Telegram request failed:", e)
        return None


def say(text, buttons=None):
    log("TG>", text[:160].replace("\n", " "))
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not chat:
        return
    chunks = [text[i:i + 3800] for i in range(0, len(text), 3800)] or [""]
    for i, c in enumerate(chunks):
        data = {"chat_id": chat, "text": c, "disable_web_page_preview": "true"}
        if buttons and i == len(chunks) - 1:
            data["reply_markup"] = json.dumps({"inline_keyboard": [buttons]})
        tg("sendMessage", data=data)


def send_video(path, caption):
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    with open(path, "rb") as f:
        return tg("sendVideo", data={"chat_id": chat, "caption": caption[:900], "supports_streaming": "true"},
                  files={"video": f})


def tg_download(file_id, dest):
    info = tg("getFile", data={"file_id": file_id})
    if not info:
        raise RuntimeError("Telegram would not give me that file (bots can only download up to 20 MB). "
                           "Send the clip as a normal video, and keep it under 20 MB.")
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    with requests.get(f"https://api.telegram.org/file/bot{token}/{info['result']['file_path']}", stream=True,
                      timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)


# ============================================================================ Gemini


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
    return [n for _, n in cands[:3]] + ["gemini-flash-latest"]


def gemini_json(prompt, models, temperature=1.0):
    last = "no model tried"
    for m in models:
        for attempt in range(3):
            t0 = time.time()
            try:
                r = requests.post(
                    f"{GEMINI}/models/{m}:generateContent", headers=gemini_headers(), timeout=300,
                    json={"contents": [{"parts": [{"text": prompt}]}],
                          "generationConfig": {"temperature": temperature, "responseMimeType": "application/json"}})
            except requests.RequestException as e:
                last = f"{m}: {type(e).__name__}"
                log("Gemini call failed:", last)
                break
            log(f"Gemini {m}: HTTP {r.status_code} in {time.time() - t0:.0f}s")
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
            break
    raise RuntimeError("Gemini failed: " + last)


def trend_notes(models):
    q = ("What kinds of satisfying, oddly satisfying and 3D-render-style looping visuals are trending right now on "
         "YouTube Shorts, TikTok and Instagram Reels? List 5 specific visual concepts popular this month "
         "(subject, material, motion), one line each. Plain text.")
    for m in models[:2]:
        try:
            r = requests.post(f"{GEMINI}/models/{m}:generateContent", headers=gemini_headers(), timeout=180,
                              json={"contents": [{"parts": [{"text": q}]}], "tools": [{"google_search": {}}]})
        except requests.RequestException as e:
            log("trend search failed:", type(e).__name__)
            continue
        if r.status_code == 200:
            try:
                parts = r.json()["candidates"][0]["content"]["parts"]
                text = "".join(p.get("text", "") for p in parts).strip()
                if text:
                    return text[:2500]
            except Exception:
                pass
        else:
            log("trend search unavailable:", m, r.status_code, r.text[:150])
    return ""


def fill(tpl, **kw):
    for k, v in kw.items():
        tpl = tpl.replace(f"<<{k}>>", str(v))
    return tpl


PRODUCER = """You are the creative director of the YouTube Shorts channel "<<channel>>". Niche: <<niche>>
Invent ONE new Short: a seamless looping clip that looks like a satisfying, polished 3D render and will be made by an AI text-to-video model.
Live trend notes (may be empty, use them as inspiration, do not copy brands): <<trends>>
Already used or DECLINED by the owner (do not repeat or imitate these): <<recent>>
Rules for video_prompt (English, 60-110 words): one clear subject; exact materials, colours and lighting; smooth continuous OSCILLATING, rotating, pulsing or rippling motion that looks natural when played backwards (no falling, pouring, melting or one-way flow); static camera; vertical 9:16; 5-8 seconds; no cuts; no text, logos, people or copyrighted characters.
Return ONLY a JSON object with exactly these keys:
{"concept": "one line: the trend angle", "video_prompt": "...", "title": "...", "description": "...", "tags": ["..."]}
title: max 60 characters, evokes a feeling or curiosity, honest about what the video shows, no emojis, no ALL CAPS.
description: 1-2 natural sentences.
tags: 10-15 short search phrases people would really type."""

CRITIC = """You are a strict reviewer of YouTube Shorts packages. Check: the video_prompt describes motion that still looks natural when played backwards, has no brands, people, text or copyrighted characters, and is clear enough for an AI video model; the title is honest and creates curiosity; tags are relevant and not spam.
Return ONLY JSON: {"score": <0-10>, "issues": ["..."], "package": <the improved package with the same keys: concept, video_prompt, title, description, tags>}.
Package:
<<package>>"""

FALLBACK_PROMPTS = [
    "A grid of glossy {c1} cubes on a dark reflective floor, flipping over one after another in a smooth diagonal wave, soft studio lighting, sharp reflections, static camera, vertical 9:16, calm hypnotic 3D render, no text",
    "A row of shiny {c1} and {c2} pendulum balls swinging in a mesmerizing wave pattern, dark glossy studio background, soft rim light, static camera, vertical 9:16, smooth continuous motion, 3D render, no text",
    "Concentric rings of polished {c1} metal rotating at different speeds like a gyroscope, dark background with soft reflections, slow smooth motion, static camera, vertical 9:16, satisfying 3D render, no text",
    "A field of glass-like {c1} spheres rippling up and down in a spiral wave, dark reflective floor, soft studio light, static camera, vertical 9:16, hypnotic smooth motion, 3D render, no text",
    "Soft clay pillars in {c1} and {c2} rising and falling in an outward ripple, matte material, dark background, gentle lighting, static camera, vertical 9:16, calm satisfying 3D render, no text",
]
COLORS = ["pink", "orange", "cyan", "mint", "blue", "violet", "coral", "lime", "gold", "teal"]


def sanitize(pkg):
    clean = lambda s: re.sub(r"[<>]", "", str(s or "")).strip()
    title = re.sub(r"\s+", " ", clean(pkg.get("title")))
    if len(title) > 60:
        title = title[:60].rsplit(" ", 1)[0]
    tags, total = [], 0
    for t in pkg.get("tags") or []:
        t = re.sub(r"[<>\"]", "", str(t)).strip().lower()
        if t and t not in tags and total + len(t) + 3 <= 450:
            tags.append(t)
            total += len(t) + 3
    return {"title": title or "Satisfying 3D Loop", "description": clean(pkg.get("description"))[:1200],
            "tags": tags, "concept": clean(pkg.get("concept"))[:300], "video_prompt": clean(pkg.get("video_prompt"))[:1500]}


def make_package(cfg, state):
    if not os.environ.get("GEMINI_API_KEY"):
        return None, "no key"
    try:
        models = pick_models(cfg)
        log("Gemini models:", models)
        trends = trend_notes(models) if cfg["ai"].get("use_trends", True) else ""
        log("trend notes:", "found" if trends else "none")
        recent = "; ".join(
            f"{e['title']} ({e['video_prompt'][:70]}...){' [DECLINED]' if e['status'] == 'declined' else ''}"
            for e in state["entries"][-30:]) or "(none yet)"
        pkg = sanitize(gemini_json(fill(PRODUCER, channel=cfg["channel_name"], niche=cfg["niche"],
                                        trends=trends or "(not available)", recent=recent), models))
        for rnd in range(int(cfg["ai"].get("critic_rounds", 1))):
            review = gemini_json(fill(CRITIC, package=json.dumps(pkg)), models, 0.4)
            log(f"Critic round {rnd + 1}: score {review.get('score')}; issues {review.get('issues')}")
            if isinstance(review.get("package"), dict):
                better = sanitize(review["package"])
                if better["video_prompt"]:
                    pkg = better
            if isinstance(review.get("score"), (int, float)) and review["score"] >= 8:
                break
        if not pkg["video_prompt"]:
            raise RuntimeError("Gemini returned no video_prompt")
        return pkg, "Gemini + live trends" if trends else "Gemini (no live trend data)"
    except Exception as e:
        log("AI step failed:", e)
        say(f"The AI step failed ({str(e)[:150]}), so this prompt comes from a built-in list.")
        return None, "fallback"


def fallback_package(state):
    used = [e["video_prompt"] for e in state["entries"][-10:]]
    options = [p for p in FALLBACK_PROMPTS if not any(p[:30] in u for u in used)] or FALLBACK_PROMPTS
    c = random.sample(COLORS, 2)
    return sanitize({
        "title": "Hypnotic 3D Loop You Can't Stop Watching", "concept": "built-in prompt",
        "description": "A calm, satisfying 3D loop.", "video_prompt": random.choice(options).format(c1=c[0], c2=c[1]),
        "tags": ["satisfying", "3d loop", "oddly satisfying", "relaxing", "hypnotic", "seamless loop", "shorts",
                 "satisfying video", "3d animation", "asmr visuals"]})


# ============================================================================ the Accept / Decline flow


def active(state, *statuses):
    return [e for e in state["entries"] if e["status"] in statuses]


def find(state, eid, *statuses):
    pool = [e for e in state["entries"] if e["status"] in statuses and (not eid or e["id"] == eid)]
    return pool[-1] if pool else None


def propose(cfg, state):
    pkg, how = make_package(cfg, state)
    if pkg is None:
        pkg, how = fallback_package(state), "built-in list"
    entry = {"id": now().strftime("%Y%m%d-%H%M%S"), "created": now().isoformat(timespec="seconds"),
             "status": "proposed", "source": how, **pkg}
    state["entries"].append(entry)
    save_state(state)
    say(f"NEW PROMPT\nTitle: {entry['title']}\nTrend angle: {entry['concept']}\nSource: {how}\n\n"
        "Prompt (press and hold to copy):")
    say(entry["video_prompt"], buttons=[{"text": "Accept", "callback_data": f"acc:{entry['id']}"},
                                        {"text": "Decline", "callback_data": f"dec:{entry['id']}"}])
    return entry


def generate_clip(entry, cfg):
    provider = cfg.get("video_generator", "manual")
    if provider == "manual":
        return None  # no free AI-video API exists; a provider with an API would be connected here
    raise RuntimeError(f"Unknown video_generator '{provider}' in config.json")


def do_accept(e, cfg, state):
    e["status"] = "accepted"
    say(f"Accepted: {e['title']}\nStarting the video...")
    clip = generate_clip(e, cfg)
    if clip is None:
        e["status"] = "awaiting_clip"
        say("No AI video generator is connected, so this part is yours. Paste the prompt below into your AI video "
            "tool (vertical 9:16, 5 to 8 seconds), then send the finished clip to this chat as a normal video, "
            "under 20 MB. I will turn it into a seamless looping Short and send it back.")
        say(e["video_prompt"])
    else:
        finish_short(e, clip, cfg)


def do_decline(e, cfg, state):
    e["status"] = "declined"
    say("Declined. Generating the next prompt...")
    propose(cfg, state)


def force_new(cfg, state):
    if active(state, "awaiting_clip"):
        say("You already accepted a prompt and I am waiting for its clip. Send the clip, or send /cancel first.")
        return
    for e in active(state, "proposed"):
        e["status"] = "skipped"
    propose(cfg, state)


def build_short(entry, clip, cfg):
    work = WORK / entry["id"]
    work.mkdir(parents=True, exist_ok=True)
    norm, master, audio, final = work / "norm.mp4", work / "master.mp4", work / "audio.m4a", work / "short.mp4"
    ffmpeg("-i", clip, "-t", 10, "-an", "-vf",
           f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps={FPS},format=yuv420p",
           "-c:v", "libx264", "-crf", 14, "-preset", "fast", norm)
    n = int(probe(norm, "nb_read_frames") or 0)
    if n < 12:
        raise RuntimeError(f"The clip is too short ({n} frames).")
    ffmpeg("-i", norm, "-filter_complex",
           f"[0:v]split[a][b];[b]trim=start_frame=1:end_frame={n - 1},setpts=PTS-STARTPTS,reverse[r];"
           f"[a][r]concat=n=2:v=1:a=0,fps={FPS}[v]",
           "-map", "[v]", "-c:v", "libx264", "-preset", "medium", "-crf", 22, "-maxrate", "6M", "-bufsize", "12M",
           "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", master)
    loop_s = (2 * n - 2) / FPS
    loops = max(1, round(cfg["short_seconds"] / loop_s))
    total = loops * loop_s
    make_audio(audio, total, entry["id"])
    ffmpeg("-stream_loop", loops - 1, "-i", master, "-i", audio, "-map", "0:v", "-map", "1:a", "-c", "copy",
           "-t", total, "-movflags", "+faststart", final)
    if final.stat().st_size > BOT_LIMIT:
        raise RuntimeError("The finished video is over Telegram's 50 MB limit. Lower short_seconds in config.json.")
    return final, total


def make_audio(out, seconds, seed):
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


def finish_short(e, clip, cfg):
    say("Got the clip. Building the looping Short...")
    final, total = build_short(e, clip, cfg)
    e["status"] = "ready"
    desc = f"{e['description']}\n\n#Shorts #satisfying #3d #loop".strip()
    send_video(final, f"{e['title']} ({total:.0f} s loop)")
    say(f"READY\n\nTitle:\n{e['title']}\n\nDescription:\n{desc}\n\nTags:\n{', '.join(e['tags'])}")
    say("To publish: press and hold the video above, save it to your phone, then upload it in the YouTube app "
        "(+, then Upload). Paste the title, description and tags. Send /new when you want the next prompt.")


def handle_clip(msg, cfg, state):
    media = msg.get("video") or msg.get("animation") or msg.get("document")
    e = find(state, None, "awaiting_clip")
    if not e:
        say("I received a video, but no accepted prompt is waiting for one. Send /new for a new prompt.")
        return
    work = WORK / e["id"]
    work.mkdir(parents=True, exist_ok=True)
    tg_download(media["file_id"], work / "clip.mp4")
    finish_short(e, work / "clip.mp4", cfg)


HELP = ("Commands: /accept, /decline, /new (new prompt now), /cancel (drop the accepted prompt), /status. "
        "Or use the Accept and Decline buttons under each prompt.")


def act(action, eid, cfg, state):
    if action == "acc":
        e = find(state, eid, "proposed")
        do_accept(e, cfg, state) if e else say("There is no prompt waiting for an answer. Send /new.")
    elif action == "dec":
        e = find(state, eid, "proposed")
        do_decline(e, cfg, state) if e else say("There is no prompt waiting for an answer. Send /new.")
    elif action == "new":
        force_new(cfg, state)
    elif action == "cancel":
        e = find(state, None, "awaiting_clip", "proposed")
        if e:
            e["status"] = "cancelled"
            say("Cancelled. Send /new for a new prompt.")
        else:
            say("Nothing to cancel.")
    elif action == "status":
        say("\n".join(f"{e['id']} [{e['status']}] {e['title']}" for e in state["entries"][-6:]) or "No prompts yet.")
    else:
        say(HELP)


WORDS = {"accept": "acc", "yes": "acc", "ok": "acc", "decline": "dec", "no": "dec", "next": "dec", "skip": "dec",
         "new": "new", "cancel": "cancel", "status": "status"}


def handle_update(u, cfg, state, chat):
    if "callback_query" in u:
        cq = u["callback_query"]
        if str(cq.get("message", {}).get("chat", {}).get("id")) != chat:
            return
        tg("answerCallbackQuery", data={"callback_query_id": cq["id"]})
        action, _, eid = (cq.get("data") or "").partition(":")
        return act(action, eid, cfg, state)
    msg = u.get("message") or {}
    if str(msg.get("chat", {}).get("id")) != chat:
        return
    doc = msg.get("document") or {}
    if msg.get("video") or msg.get("animation") or str(doc.get("mime_type", "")).startswith("video"):
        return handle_clip(msg, cfg, state)
    word = (msg.get("text") or "").strip().lower().split(" ")[0].lstrip("/").split("@")[0]
    act(WORDS.get(word, "help"), "", cfg, state)


def due_slot(cfg, t):
    s = cfg["schedule"]
    if t.weekday() not in s["weekdays"]:
        return None
    slots = []
    for x in s["times_utc"]:
        hh, mm = map(int, x.split(":"))
        slot = t.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if slot <= t:
            slots.append(slot)
    return max(slots).strftime("%Y-%m-%dT%H:%M") if slots else None


def cmd_tick(a):
    cfg, state = load_config(), load_state()
    chat = str(os.environ["TELEGRAM_CHAT_ID"])
    res = tg("getUpdates", params={"offset": state["last_update_id"] + 1, "timeout": 0})
    for u in (res or {}).get("result", []):
        state["last_update_id"] = max(state["last_update_id"], u["update_id"])
        save_state(state)
        try:
            handle_update(u, cfg, state, chat)
        except Exception as e:
            log("update failed:", repr(e))
            say(f"Error: {str(e)[:300]}")
        save_state(state)
    if a.new or os.environ.get("FORCE_NEW", "").lower() == "true":
        force_new(cfg, state)
    elif not active(state, "proposed", "awaiting_clip"):
        slot = due_slot(cfg, now())
        if slot and slot != state.get("last_slot"):
            state["last_slot"] = slot
            propose(cfg, state)
    save_state(state)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tick")
    t.add_argument("--new", action="store_true")
    t.set_defaults(fn=cmd_tick)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
