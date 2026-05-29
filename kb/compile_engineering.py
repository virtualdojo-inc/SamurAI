"""Compile engineering article STUBS → final wiki/troubleshooting articles.

The structural reasoning already happened mechanically in ``kb.ingest_repo``,
which wrote one scaffolded **stub** per target to ``engineering/raw/stubs/``
(headings + pre-grouped grounded material + code pointers + embedded source docs
+ inline ``<!-- WRITE: ... -->`` instructions). This module runs a single uniform
"fill the stub" pass with in-boundary Gemini: one stub → one article. No grouping
logic here — the stub carries it.

Principle ("map, don't mirror"): the model fleshes out each heading into prose +
invariants grounded ONLY in the stub, keeps the code-path pointers (so SamurAI
can target ``repo_sync``), and never pastes code or asserts a structural map as a
guarantee of current runtime behavior.

Bounded + resumable: a per-article manifest hashes each stub, so unchanged stubs
skip and an interrupted tick resumes. Single-flight is handled by the caller
(``kb.run.run_engineering_pipeline``), same as the support pipeline.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date

import yaml

from kb import storage
from kb.gemini import get_kb_llm, kb_engine_info

STUBS_PREFIX = "engineering/raw/stubs/"
WIKI_PREFIX = "engineering/wiki/"
TROUBLE_PREFIX = "engineering/troubleshooting/"
INDEX_PATH = "engineering/wiki/index.md"
MANIFEST_PATH = "engineering/.eng_manifest.json"

_FILL_SYS = (
    "You are a senior engineer documenting the VirtualDojo CRM for an AI "
    "teammate that troubleshoots issues and writes GitHub bug reports. You are "
    "given ONE article STUB: a skeleton with headings, pre-grouped grounded "
    "material (real service/model/view filenames, the API router map, embedded "
    "source docs), and inline `<!-- WRITE: ... -->` instructions. Produce the "
    "final article in markdown. RULES: (1) MAP, DON'T MIRROR — describe structure "
    "and durable invariants; reference code PATHS as pointers (so the reader can "
    "open them) but NEVER paste code. (2) Ground every statement in the stub's "
    "material; do not invent files, endpoints, or behavior. (3) Resolve every "
    "`<!-- WRITE: ... -->` and remove the comment; drop a heading the stub has no "
    "material for. (4) Frame everything as a structural map / troubleshooting "
    "pointer, NOT a guarantee of current runtime behavior (the code is the source "
    "of truth and may have changed). (5) No secrets, PII, or financial values. "
    "Output ONLY the markdown body — do NOT emit YAML frontmatter (it is added "
    "programmatically)."
)


def _llm_text(llm, system: str, user: str) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage

    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return resp.content if isinstance(resp.content, str) else str(resp.content)


def _parse_stub(text: str) -> tuple[dict, str]:
    """Return (frontmatter_dict, body) for a stub."""
    meta: dict = {}
    body = text
    if text.lstrip().startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            try:
                loaded = yaml.safe_load(parts[1])
                meta = loaded if isinstance(loaded, dict) else {}
            except yaml.YAMLError:
                meta = {}
            body = parts[2].strip()
    return meta, body


def _yaml_str(s: str) -> str:
    """Emit a YAML-safe double-quoted scalar."""
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _frontmatter(title: str, summary: str, kind: str) -> str:
    return (
        "---\n"
        f"title: {_yaml_str(title)}\n"
        f"summary: {_yaml_str(summary)}\n"
        "kind: engineering-knowledge\n"
        f"category: {kind}\n"
        f"source: virtualdojo repo (orientation map — verify against live code)\n"
        f"compiled: {date.today().isoformat()}\n"
        "---\n\n"
    )


def _load_manifest() -> dict:
    raw = storage.read_text(MANIFEST_PATH)
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _refresh_index() -> int:
    articles = []
    for prefix in (WIKI_PREFIX, TROUBLE_PREFIX):
        for p in storage.list_paths(prefix):
            if p.endswith(".md") and not p.endswith("index.md"):
                articles.append(p)
    articles.sort()
    lines = [
        "# VirtualDojo Engineering Knowledge",
        "",
        "Orientation map of the VirtualDojo CRM (frontend + backend), auto-compiled "
        "from the repo's docs + structure by in-boundary Gemini. These are "
        "structural pointers and durable invariants — the live code is the source "
        "of truth. Do not edit by hand.",
        "",
    ]
    for p in articles:
        lines.append(f"- [[{p.rsplit('/', 1)[-1][:-3]}]]")
    storage.write_text(INDEX_PATH, "\n".join(lines) + "\n")
    return len(articles)


def compile_engineering(llm=None, max_docs: int | None = None) -> dict:
    """Fill up to ``max_docs`` new/changed stubs into final articles. Resumable."""
    llm = llm or get_kb_llm()
    manifest = _load_manifest()

    stubs = [
        (p, t) for p, t in storage.list_text(STUBS_PREFIX)
        if p.rsplit("/", 1)[-1] not in ("index.md",)
    ]
    pending = [(p, t, hashlib.sha256(t.encode("utf-8")).hexdigest()) for p, t in stubs]
    pending = [(p, t, h) for p, t, h in pending if manifest.get(p) != h]
    batch = pending[:max_docs] if max_docs else pending

    written = 0
    for path, text, h in batch:
        meta, body = _parse_stub(text)
        title = str(meta.get("title") or path.rsplit("/", 1)[-1][:-3])
        summary = str(meta.get("summary") or title)
        kind = str(meta.get("kind") or "wiki").lower()
        out_prefix = TROUBLE_PREFIX if kind == "troubleshooting" else WIKI_PREFIX
        slug = path.rsplit("/", 1)[-1][:-3]

        filled = _llm_text(llm, _FILL_SYS, f"STUB ({slug}):\n\n{body}").strip()
        storage.write_text(
            f"{out_prefix}{slug}.md",
            _frontmatter(title, summary, kind) + filled + "\n",
        )
        manifest[path] = h
        storage.write_text(MANIFEST_PATH, json.dumps(manifest), content_type="application/json")
        written += 1

    total = _refresh_index()
    stats = {
        "engine": kb_engine_info(),
        "stubs_seen": len(stubs),
        "articles_written": written,
        "articles_total": total,
        "stubs_remaining": max(0, len(pending) - written),
    }
    print(
        f"[kb.compile_eng] batch: written={written} "
        f"remaining={stats['stubs_remaining']} total_articles={total}",
        flush=True,
    )
    return stats
