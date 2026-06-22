"""Loom video ingestion for the DH Tech Issue Tracker.

Tracker rows frequently carry a Loom share link. Empirically these are almost
always SILENT screen recordings (sampled 5/5: empty Loom transcript + no
narration), so the meaning is VISUAL. This module extracts BOTH signals and
fuses them into a troubleshooting-oriented understanding for a ticket:

  - AUDIO  : Loom's own transcript if present; else, if the audio track has
             speech, transcribe it in-boundary. (Narration is rare on this
             tracker but fully supported + verified.)
  - VISUAL : evenly-spaced keyframes -> in-boundary multimodal description of
             the on-screen actions / UI / errors.

COMPLIANCE: every LLM call goes to the regional, in-boundary Vertex Gemini
(kb/gemini.py: us-central1, gemini-2.5-flash-lite). NEVER an external LLM. In
prod this runs in-process on samurai-bot (in-boundary). Media is downloaded to a
temp dir and deleted after; only the derived TEXT is kept.

Requires `yt-dlp` and `ffmpeg`/`ffprobe` on PATH.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field

from kb.gemini import KB_COMPILE_MODEL, KB_VERTEX_LOCATION, kb_engine_info

logger = logging.getLogger(__name__)

# Below this mean loudness we treat the track as having no transcribable speech
# (real narration sits ~ -20 to -30 dB; the silent tracker demos sit < -38 dB).
_SPEECH_MEAN_DB = -38.0
_MAX_FRAMES = 8
_SHARE_RE = re.compile(r"loom\.com/share/([0-9a-fA-F]{16,})")


@dataclass
class LoomAnalysis:
    loom_id: str
    url: str
    title: str = ""
    duration: float = 0.0
    uploader: str = ""
    upload_date: str = ""
    narration_source: str = "none"   # loom_transcript | transcribed | none
    narration: str = ""
    visual_summary: str = ""
    understanding: str = ""           # fused, troubleshooting-oriented summary
    frames_used: int = 0
    engine: dict = field(default_factory=kb_engine_info)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# ── shell helpers ─────────────────────────────────────────────────────────


def _run(cmd: list[str], timeout: int = 240) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def loom_id_from_url(url: str) -> str | None:
    m = _SHARE_RE.search(url or "")
    return m.group(1) if m else None


def _default_project() -> str:
    p = os.environ.get("GCP_PROJECT_ID")
    if p:
        return p
    try:
        return _run(["gcloud", "config", "get-value", "project"]).stdout.strip()
    except Exception:
        return ""


# ── in-boundary multimodal call (regional Vertex, same guardrail as kb/gemini) ──


def _access_token() -> str:
    """ADC in prod (the bot's SA); fall back to the gcloud user token locally."""
    try:
        import google.auth
        import google.auth.transport.requests

        creds, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        creds.refresh(google.auth.transport.requests.Request())
        if creds.token:
            return creds.token
    except Exception:
        pass
    return _run(["gcloud", "auth", "print-access-token"]).stdout.strip()


def _vertex_multimodal(parts: list[dict], max_tokens: int = 700, temperature: float = 0.2) -> str:
    """POST mixed text/image/audio parts to the regional in-boundary Gemini."""
    if KB_VERTEX_LOCATION == "global":
        raise RuntimeError(
            "Loom ingest must use a regional Vertex endpoint (FedRAMP residency); "
            "KB_VERTEX_LOCATION='global' is not allowed."
        )
    import httpx

    project = _default_project()
    url = (
        f"https://{KB_VERTEX_LOCATION}-aiplatform.googleapis.com/v1/projects/"
        f"{project}/locations/{KB_VERTEX_LOCATION}/publishers/google/models/"
        f"{KB_COMPILE_MODEL}:generateContent"
    )
    body = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
    }
    r = httpx.post(url, headers={"Authorization": f"Bearer {_access_token()}"},
                   json=body, timeout=240)
    r.raise_for_status()
    d = r.json()
    cand = (d.get("candidates") or [{}])[0]
    return "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", [])).strip()


def _text(s: str) -> dict:
    return {"text": s}


def _inline(path: str, mime: str) -> dict:
    with open(path, "rb") as f:
        return {"inlineData": {"mimeType": mime, "data": base64.b64encode(f.read()).decode()}}


# ── download + media extraction ────────────────────────────────────────────


def _yt(args: list[str], url: str, timeout: int = 240) -> subprocess.CompletedProcess:
    return _run(["yt-dlp", "--no-warnings", *args, url], timeout=timeout)


def download_loom(url: str, workdir: str) -> LoomAnalysis:
    """Resolve metadata, Loom transcript, video, and audio for a share URL."""
    vid = loom_id_from_url(url) or "loom"
    a = LoomAnalysis(loom_id=vid, url=url)

    meta = _yt(["--print", "%(title)s|||%(duration)s|||%(uploader)s|||%(upload_date)s"], url)
    if meta.returncode == 0 and "|||" in meta.stdout:
        t, d, up, ud = (meta.stdout.strip().split("|||") + ["", "", "", ""])[:4]
        a.title, a.uploader, a.upload_date = t, up, ud
        try:
            a.duration = float(d)
        except ValueError:
            a.duration = 0.0

    # Loom's own transcript (free when there's narration)
    _yt(["--skip-download", "--write-subs", "--sub-langs", "en", "--sub-format", "json",
         "-o", os.path.join(workdir, f"{vid}.%(ext)s")], url)
    jp = os.path.join(workdir, f"{vid}.en.json")
    if os.path.exists(jp):
        try:
            phrases = json.load(open(jp)).get("phrases", [])
            text = " ".join(p.get("text", "") for p in phrases).strip()
            if text:
                a.narration, a.narration_source = text, "loom_transcript"
        except Exception:
            pass

    # Video (<=720p) for keyframes
    _yt(["-f", "bestvideo[height<=720]/bestvideo/best",
         "-o", os.path.join(workdir, f"{vid}_v.%(ext)s")], url)
    a._video = next((os.path.join(workdir, f) for f in os.listdir(workdir)
                     if f.startswith(f"{vid}_v.")), None)  # type: ignore[attr-defined]

    # Audio (only needed if Loom gave us no transcript)
    a._audio = None  # type: ignore[attr-defined]
    if a.narration_source == "none":
        _yt(["-f", "bestaudio", "-o", os.path.join(workdir, f"{vid}_a.%(ext)s")], url)
        a._audio = next((os.path.join(workdir, f) for f in os.listdir(workdir)  # type: ignore[attr-defined]
                         if f.startswith(f"{vid}_a.")), None)
    return a


def _mean_volume_db(audio_path: str) -> float | None:
    v = _run(["ffmpeg", "-hide_banner", "-i", audio_path, "-af", "volumedetect", "-f", "null", "/dev/null"])
    m = re.search(r"mean_volume:\s*(-?[\d.]+) dB", v.stderr)
    return float(m.group(1)) if m else None


def extract_keyframes(video_path: str, workdir: str, duration: float) -> list[str]:
    n = max(4, min(_MAX_FRAMES, int(duration // 5) or 4))
    fps = n / max(duration, 1.0)
    out = os.path.join(workdir, "frame_%02d.jpg")
    _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", video_path,
          "-vf", f"fps={fps:.4f},scale=1280:-2", out])
    return sorted(os.path.join(workdir, f) for f in os.listdir(workdir) if f.startswith("frame_"))


# ── analysis ───────────────────────────────────────────────────────────────


def transcribe_audio(audio_path: str) -> str:
    parts = [_text("Transcribe this audio verbatim. If there is no speech, reply exactly '(no speech)'."),
             _inline(audio_path, "audio/mpeg")]
    try:
        out = _vertex_multimodal(parts, max_tokens=500, temperature=0.0)
    except Exception as e:
        logger.warning("[loom] transcribe failed: %s", e)
        return ""
    return "" if out.strip().lower().startswith("(no speech") else out.strip()


def analyze_frames(frames: list[str], a: LoomAnalysis, ticket_context: str) -> str:
    head = (
        f"These are {len(frames)} chronological keyframes from a {a.duration:.0f}s screen "
        f"recording titled '{a.title}', posted on the DH Tech Issue Tracker (the VirtualDojo CRM). "
        "Describe concretely what the user does, which screen/record/feature is shown, any error "
        "toasts or unexpected state changes, and what behaviour the clip demonstrates."
    )
    if ticket_context:
        head += f"\nTicket context: {ticket_context}"
    parts = [_text(head)] + [_inline(f, "image/jpeg") for f in frames]
    return _vertex_multimodal(parts, max_tokens=700, temperature=0.2)


def _fuse(a: LoomAnalysis, ticket_context: str) -> str:
    prompt = (
        "You are SamurAI turning a Loom video into a troubleshooting note for a developer.\n"
        f"Title: {a.title}\nDuration: {a.duration:.0f}s\n"
        f"Narration ({a.narration_source}): {a.narration or '(none — silent screen recording)'}\n"
        f"On-screen (from frames): {a.visual_summary}\n"
        f"Ticket context: {ticket_context or '(none)'}\n\n"
        "Write a concise note with exactly these sections:\n"
        "WHAT THE VIDEO SHOWS: 2-3 sentences.\n"
        "LIKELY ISSUE: 1-2 sentences on the bug/behaviour being reported.\n"
        "SUGGESTED AREA: the feature/screen a developer should look at first."
    )
    return _vertex_multimodal([_text(prompt)], max_tokens=500, temperature=0.2)


def ingest_loom(url: str, ticket_context: str = "", keep_media: bool = False) -> LoomAnalysis:
    """Full pipeline: download -> audio (transcript|transcribe) + visual -> fuse."""
    workdir = tempfile.mkdtemp(prefix="loom-")
    try:
        a = download_loom(url, workdir)

        # AUDIO: only transcribe if Loom gave no transcript AND the track has speech.
        if a.narration_source == "none" and getattr(a, "_audio", None):
            mean = _mean_volume_db(a._audio)  # type: ignore[attr-defined]
            if mean is not None and mean > _SPEECH_MEAN_DB:
                txt = transcribe_audio(a._audio)  # type: ignore[attr-defined]
                if txt:
                    a.narration, a.narration_source = txt, "transcribed"

        # VISUAL: always — this is what carries the meaning for silent demos.
        if getattr(a, "_video", None):
            frames = extract_keyframes(a._video, workdir, a.duration)  # type: ignore[attr-defined]
            a.frames_used = len(frames)
            if frames:
                a.visual_summary = analyze_frames(frames, a, ticket_context)

        a.understanding = _fuse(a, ticket_context)
        return a
    finally:
        if not keep_media:
            shutil.rmtree(workdir, ignore_errors=True)


# ── CLI for local testing: python -m kb.ingest_loom <url> [ticket context] ──

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("usage: python -m kb.ingest_loom <loom-share-url> [ticket context]")
        raise SystemExit(2)
    res = ingest_loom(sys.argv[1], ticket_context=" ".join(sys.argv[2:]))
    d = res.to_dict()
    print(json.dumps({k: v for k, v in d.items() if not k.startswith("_")}, indent=2, default=str))
