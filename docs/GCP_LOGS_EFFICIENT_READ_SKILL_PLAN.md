# Plan: `reading-cloud-logs` skill — read GCP logs fast and cheap

## Goal

Cut both the **wall-clock time** and the **token cost** of pulling Google Cloud
logs during troubleshooting, without losing the signal needed for root-cause.
Two levers, applied in order:

1. **Narrow server-side** with a tight Logging Query Language (LQL) filter so the
   API returns fewer rows.
2. **Compress client-side** — project only the fields that matter, drop known
   junk with regex, collapse repeated lines, and truncate — so what reaches the
   model is small.

The current path fails on lever 2 entirely.

## The problem, concretely

`tools/gcp_logging.py::query_cloud_logs` (the tool the agent actually calls):

- Returns `entry.payload` **verbatim** — for a Cloud Run structured log that is
  the whole `jsonPayload` dict; for an audit log it's the entire `protoPayload`.
  A single entry can be hundreds of tokens of `insertId`/`labels`/`trace`/`spanId`
  noise wrapped around a one-line message.
- `max_results=50` with **no `order_by`** → `list_entries` does not default to
  newest-first, so on a busy window it can hand back the *oldest* 50 entries and
  miss the errors you're chasing. (`admin.py:114` already passes
  `order_by=gcl.DESCENDING` — the logging tool does not.)
- No regex include/exclude, no dedup, no per-message truncation. The known-noise
  lines the `troubleshooting-cloud-run` skill tells us to ignore (gcsfuse
  `OutOfOrderError`, the `gsa-scraper` service, drain/shutdown) are still fully
  serialized into context before anyone filters them.

So the model pays tokens to read junk it was told to throw away, and may not even
get the relevant rows. That's the thing to fix.

## Deliverable 1 — the skill (`skills/reading-cloud-logs/SKILL.md`)

A focused how-to that any log pull runs through. It is *distinct from*
`troubleshooting-cloud-run` (which owns the investigation flow: timeline,
regression-vs-drain, failure signatures). This skill owns only **"how to get the
bytes out efficiently"** and cross-links to the other. Contents:

### A. Filter tightly (server-side, indexed fields first)
Order every filter to hit indexed fields, which scan fastest:
`resource.type`, `resource.labels.*`, `severity`, `logName`, and a timestamp
bound. Canonical SamurAI base filter:

```
resource.type="cloud_run_revision"
resource.labels.service_name="samurai-bot"
severity>=ERROR
```

Rules:
- **Always** bound time (`--freshness=1h`/`2h`, widen only if empty). Never run
  unbounded.
- **Always** pin `resource.labels.service_name` — excludes the co-located
  `gsa-scraper` service at the source instead of filtering it later.
- Prefer the `SEARCH()` function or a label match over a bare substring scan of
  `textPayload`/`jsonPayload` (content fields aren't indexed).
- Start at `severity>=ERROR`; drop to `>=WARNING` only when ERROR is empty.

### B. Project only what matters (the big token win)
Never dump whole payloads. Pull four fields — timestamp, revision, severity, and
the single best message string:

```
gcloud logging read '<filter>' --project virtualdojo-samurai --freshness=1h \
  --order=desc --limit=50 \
  --format='value(timestamp, resource.labels.revision_name, severity, json_payload.message, text_payload)'
```

`--order=desc` gets newest-first; read bottom-up for chronological order.

### C. Strip junk with regex (client-side)
Pipe through `grep -vE` to drop documented noise, then optionally `grep -E` to
keep only the signal:

```
... | grep -vE 'OutOfOrderError|tasks\.sqlite-journal|langmem_memories\.sqlite|Shutdown|SIGTERM|draining|/healthz?'
```

Maintain the exclusion regex as a named constant in the skill so it stays in sync
with the "known operational noise" list in CLAUDE.md and the
`troubleshooting-cloud-run` skill.

### D. Collapse and cap
- Collapse identical repeated lines (`sort | uniq -c | sort -rn`, or read the
  distinct set) — a crash loop is one fact, not 200 rows.
- If still large, `--limit=20` and widen only if the tail is truncated.
- For a split multi-line traceback, widen the window just enough to capture the
  final exception line (which names the real error) — per
  `troubleshooting-cloud-run`.

### E. Decision flow (one screen)
`errors in last hour? → base filter + projection + noise grep → if empty widen
freshness → if empty drop to WARNING → collapse repeats → read tail`.

## Deliverable 2 — make the in-bot tool compress by default

A skill only helps the CLI path. The **agent** calls `query_cloud_logs`, so the
compression has to live in the tool too, or in-bot log reads stay expensive.
Minimal, backward-compatible change to `tools/gcp_logging.py`:

- Add `order_by=DESCENDING` and keep the newest N (fixes the ordering bug).
- Replace `entry.payload` dump with a **compact message extractor**: prefer
  `jsonPayload.message` → `textPayload` → `protoPayload.status.message` → a short
  `repr` fallback; emit `[ts] REVISION SEVERITY: message` only.
- New optional params (all default to sensible values, no caller changes needed):
  - `exclude_regex: str | None` — drop matching lines (default = the SamurAI
    noise pattern above).
  - `include_regex: str | None` — keep only matching lines.
  - `max_chars: int = 300` per message, and a total-output cap with an explicit
    `…(truncated N more)` marker.
  - `collapse_repeats: bool = True` — fold identical messages to `(×N)`.

This is a pure compression layer over the same query — it changes formatting, not
which rows are fetched, so it can't hide data the caller didn't already filter
out server-side. Keep the raw-filter escape hatch (a caller can pass
`exclude_regex=""` to disable stripping).

## Why not just edit the existing skill?
`troubleshooting-cloud-run` is about *what to conclude*; this is about *how to
fetch*. Splitting keeps each short (the model loads only what it needs) and lets
the read-efficiency guidance be reused by non-troubleshooting log pulls (billing
audit, FedRAMP evidence in `tools/fedramp.py`, which has the same
`list_entries(... max_results=500)` full-payload pattern and would benefit from
the same extractor).

## Scope / non-goals
- No Log Router sink or exclusion-filter changes (that drops ingestion at the
  source but is an infra + cost-model decision, out of scope for a read skill).
- No caching layer. KISS — the win is projection + regex + dedup, not
  infrastructure.

## Test / verify
- Unit-test the message extractor and regex/dedup layer in `tests/` against
  sample `jsonPayload`, `textPayload`, and `protoPayload` entries (no network).
- End-to-end: run one real `gcloud logging read` with the projected format vs the
  old full-payload dump on the same window and record the token delta in the PR.

## Sources
- [Logging query language — indexed fields & SEARCH()](https://cloud.google.com/logging/docs/view/logging-query-language)
- [gcloud logging read reference](https://cloud.google.com/sdk/gcloud/reference/logging/read)
- [Logging & viewing logs in Cloud Run](https://cloud.google.com/run/docs/logging)
- [Optimise Cloud Logging with LQL (indexed fields, narrow scope)](https://medium.com/google-cloud/optimise-cloud-logging-in-google-cloud-with-logging-query-language-6f4e4ba417b0)
