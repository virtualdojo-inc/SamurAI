"""Agent tool: analyze a Loom video (audio + visual) for troubleshooting / ticket
understanding.

Thin async wrapper over kb.ingest_loom, which downloads the Loom, reads BOTH its
audio (Loom transcript, else in-boundary transcription of any narration) and its
on-screen visuals (keyframes -> multimodal), and fuses them into a dev-oriented
note. All LLM work runs on the regional in-boundary Vertex Gemini (kb/gemini.py).

Requires yt-dlp + ffmpeg in the container (see Dockerfile). ingest_loom is
blocking (subprocess), so it runs in a thread to avoid stalling the event loop.
"""
from __future__ import annotations

import asyncio
import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
async def analyze_loom_video(url: str, ticket_context: str = "") -> str:
    """Analyze a Loom video and return what it shows — for troubleshooting or ticket understanding.

    Downloads the Loom and reads BOTH its audio (Loom's transcript, or
    in-boundary transcription if there's narration) and its on-screen visuals
    (keyframes via multimodal). Returns a concise note: what the video shows, the
    likely issue, and the area a developer should look at first. DH Tech Tracker
    Looms are usually silent screen recordings, so this works even with no
    narration. Use it when a tracker row or message links a Loom and you need to
    understand the reported issue.

    Args:
        url: The Loom share URL (https://www.loom.com/share/<id>).
        ticket_context: Optional context (e.g. the tracker row text) to focus the analysis.
    """
    from kb.ingest_loom import ingest_loom, loom_id_from_url

    if not loom_id_from_url(url):
        return f"That doesn't look like a Loom share URL: {url!r}"
    try:
        res = await asyncio.to_thread(ingest_loom, url, ticket_context)
    except Exception as e:
        logger.exception("[loom] analyze failed")
        return f"Could not analyze the Loom video: {type(e).__name__}: {e}"

    parts = [
        f"**{res.title or 'Loom video'}** — {res.duration:.0f}s"
        + (f", by {res.uploader}" if res.uploader else "")
    ]
    if res.narration_source != "none" and res.narration:
        parts.append(f"Narration ({res.narration_source}): {res.narration}")
    else:
        parts.append("Narration: none (silent screen recording).")
    parts.append(res.understanding or res.visual_summary or "(no analysis produced)")
    return "\n\n".join(parts)


LOOM_TOOLS = [analyze_loom_video]
