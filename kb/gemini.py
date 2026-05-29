"""In-boundary Vertex AI Gemini client for KB ingest/compile/lint.

COMPLIANCE: this is the ONLY LLM the knowledge-base pipeline may use. It runs on
Vertex AI inside the SamurAI Assured Workloads boundary. Two rules enforced here:

1. **Regional endpoint, not global.** FedRAMP-Moderate data residency requires the
   regional ``us-central1`` Vertex endpoint so the bucket's data never leaves the
   boundary region. (The chat agent uses ``global`` for gemini-3.5-flash; the KB
   pipeline must NOT — it pins a regionally-available model.)
2. **No external LLM, ever.** Anthropic/OpenAI/etc. are never called from this
   pipeline. There is deliberately no Anthropic client here.

Model is configurable via ``KB_COMPILE_MODEL`` (default a model confirmed
available at us-central1). Upgrade it only to another regionally-available,
in-boundary model.
"""

from __future__ import annotations

import os

from langchain_google_genai import ChatGoogleGenerativeAI

# Regional, in-boundary endpoint — do not change to "global" for KB data.
KB_VERTEX_LOCATION = os.environ.get("KB_VERTEX_LOCATION", "us-central1")
# gemini-2.5-flash-lite is confirmed available at us-central1 for this project.
KB_COMPILE_MODEL = os.environ.get("KB_COMPILE_MODEL", "gemini-2.5-flash-lite")

_llm: ChatGoogleGenerativeAI | None = None


def get_kb_llm() -> ChatGoogleGenerativeAI:
    """Return the singleton in-boundary, regional Gemini client for the KB."""
    global _llm
    if _llm is None:
        if KB_VERTEX_LOCATION == "global":
            # Guardrail: refuse the global endpoint for in-boundary KB data.
            raise RuntimeError(
                "KB pipeline must use a regional Vertex endpoint (FedRAMP "
                "residency); KB_VERTEX_LOCATION='global' is not allowed."
            )
        _llm = ChatGoogleGenerativeAI(
            model=KB_COMPILE_MODEL,
            vertexai=True,
            project=os.environ.get("GCP_PROJECT_ID"),
            location=KB_VERTEX_LOCATION,
            temperature=0,
        )
    return _llm


def kb_engine_info() -> dict:
    """Provenance for logs/verification — proves which in-boundary engine ran."""
    return {
        "engine": "vertex-gemini",
        "model": KB_COMPILE_MODEL,
        "location": KB_VERTEX_LOCATION,
        "external_llm": False,
    }
