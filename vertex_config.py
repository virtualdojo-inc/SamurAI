"""Central Vertex AI endpoint + model configuration (single source of truth).

The chat / serving path historically ran on the Vertex ``global`` endpoint because
the flash line was only served there for some projects. This project
(``virtualdojo-samurai``) can instead serve from the **US data-residency REP
endpoint** ``aiplatform.us.rep.googleapis.com`` (``locations/us``), which keeps
inference in-US (FedRAMP data-residency; VPC-SC compatible).

The serving tiers were bumped to the Gemini 3.6 flash generation (released
2026-07-21):
  - ``gemini-3.6-flash``        -> the SERVE tier (agent, verifier, judge stage-2, …)
  - ``gemini-3.5-flash-lite``   -> the LITE tier (synth, judge stage-1)

⚠️ REP ``locations/us`` availability of these two ids is NOT yet verified for this
project. As of the last live check (2026-07-12) the REP endpoint served exactly
``gemini-3.5-flash`` (SERVE) + ``gemini-3.1-flash-lite`` (LITE); the 3.6 flash
generation is one day old and new models typically reach REP/regional residency
endpoints after ``global``. BEFORE deploying, confirm both ids serve on
``aiplatform.us.rep.googleapis.com`` (``locations/us``) for ``virtualdojo-samurai``
— e.g. ``curl`` a ``:generateContent`` probe or check the project's model garden.
If they do not serve yet, roll the defaults back to the last known-good pair
(see below); do NOT fall back to ``global`` for customer data — that breaks
data-residency.

Everything here is env-overridable so a rollback is a config change, not a code
change / redeploy:
  ``SAMURAI_VERTEX_LOCATION``  default ``us``       (set ``global`` to roll back)
  ``SAMURAI_VERTEX_ENDPOINT``  default the REP url  (set ``""`` for the default frontend)
  ``SAMURAI_SERVE_MODEL``      default ``gemini-3.6-flash``
  ``SAMURAI_LITE_MODEL``       default ``gemini-3.5-flash-lite``

To roll the SERVING TIERS back to the last REP-verified pair (stay in-boundary):
  SAMURAI_SERVE_MODEL=gemini-3.5-flash  SAMURAI_LITE_MODEL=gemini-3.1-flash-lite

To roll the whole serving path back to global (last resort — NOT data-resident):
  SAMURAI_VERTEX_LOCATION=global  SAMURAI_VERTEX_ENDPOINT=  SAMURAI_SERVE_MODEL=gemini-3.5-flash  SAMURAI_LITE_MODEL=gemini-2.5-flash-lite

NOTE — deliberately NOT covered here: embeddings + the KB/memory pipeline. They
already run at ``us-central1`` (in-US) on their own env vars (``GCP_LOCATION`` /
``KB_VERTEX_LOCATION``), and the REP endpoint serves no embedding model, so moving
them here would 404. Social image generation (``tools/social_media.py``) also stays
on its current endpoint — its model isn't served on REP and it handles no customer
data. Only the text serving path is migrated.
"""
import os

VERTEX_LOCATION = os.environ.get("SAMURAI_VERTEX_LOCATION", "us")
VERTEX_ENDPOINT = os.environ.get(
    "SAMURAI_VERTEX_ENDPOINT", "https://aiplatform.us.rep.googleapis.com"
).strip()

# Model ids for the two serving tiers. Must exist at the configured endpoint.
# ⚠️ REP locations/us availability of the 3.6 flash generation is unverified —
# see the module docstring before deploying.
SERVE_MODEL = os.environ.get("SAMURAI_SERVE_MODEL", "gemini-3.6-flash")
LITE_MODEL = os.environ.get("SAMURAI_LITE_MODEL", "gemini-3.5-flash-lite")


def vertex_kwargs(**extra) -> dict:
    """kwargs for ``ChatGoogleGenerativeAI(vertexai=True)`` targeting the configured
    endpoint/region. Merge in per-call extras (e.g. ``temperature=0``). When an
    endpoint is set it is passed as ``client_options={"api_endpoint": <https url>}``
    — the ``https://`` prefix is REQUIRED (host-only raises UnsupportedProtocol)."""
    kw = dict(
        project=os.environ.get("GCP_PROJECT_ID"),
        location=VERTEX_LOCATION,
        vertexai=True,
    )
    if VERTEX_ENDPOINT:
        kw["client_options"] = {"api_endpoint": VERTEX_ENDPOINT}
    kw.update(extra)
    return kw


def genai_http_options():
    """For the raw google-genai SDK path (``tools/google_search.py``): an
    ``HttpOptions`` with ``base_url`` set to the configured endpoint, or ``None``
    to use the SDK default. Imported lazily so importing this module stays cheap."""
    if not VERTEX_ENDPOINT:
        return None
    from google.genai.types import HttpOptions
    return HttpOptions(base_url=VERTEX_ENDPOINT)
