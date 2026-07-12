---
name: reading-cloud-logs
description: Read Google Cloud logs fast and cheap — narrow server-side with a tight filter, project only the fields that matter, strip known noise with regex, collapse repeats. Use before any log pull to cut wall-clock time and token cost. Pairs with troubleshooting-cloud-run (which owns what to conclude); this owns how to fetch.
---

# Reading Cloud Logs efficiently

Goal: get the signal out of Google Cloud Logging with the fewest rows and tokens.
Two levers, in order — (1) **narrow server-side** so the API returns less, then
(2) **compress client-side** so little reaches the model. For *what to conclude*
from the logs (timeline, regression-vs-drain, failure signatures), load
`troubleshooting-cloud-run`.

## From the bot: use `query_cloud_logs`
It already compresses by default — newest-first, one compact
`[ts] revision SEVERITY: message` line per entry (not the payload blob), known
noise stripped, identical repeats collapsed. Give it a tight filter and let it do
the rest:

```
query_cloud_logs(
  'resource.type="cloud_run_revision" '
  'resource.labels.service_name="samurai-bot" severity>=ERROR',
  max_results=30,
)
```

- `exclude_regex` defaults to the operational-noise pattern (OutOfOrderError,
  sqlite-journal, Shutdown/SIGTERM/draining, /healthz). Pass `exclude_regex=""`
  to see everything; pass a custom pattern to override.
- `include_regex="..."` keeps only matching lines (e.g. a specific error class).
- Filtered-out counts are reported so nothing is hidden silently.

## From the CLI: filter tight, project narrow, grep the rest

### 1. Filter tightly — indexed fields first
Indexed fields scan fastest: `resource.type`, `resource.labels.*`, `severity`,
`logName`, and a **time bound**. Canonical SamurAI base filter:

```
resource.type="cloud_run_revision"
resource.labels.service_name="samurai-bot"
severity>=ERROR
```

- **Always** bound time (`--freshness=1h`); widen only if empty.
- **Always** pin `service_name` — excludes the co-located `gsa-scraper` service
  at the source, not after the fact.
- Prefer `SEARCH()` / a label match over a bare substring scan of
  `textPayload`/`jsonPayload` (content fields aren't indexed → slow).
- Start at `severity>=ERROR`; drop to `>=WARNING` only when ERROR is empty.

### 2. Project only what matters (the big token win)
Never dump whole payloads. Pull timestamp, revision, severity, message:

```bash
gcloud logging read \
  'resource.type="cloud_run_revision" resource.labels.service_name="samurai-bot" severity>=ERROR' \
  --project virtualdojo-samurai --freshness=1h --order=desc --limit=50 \
  --format='value(timestamp, resource.labels.revision_name, severity, json_payload.message, text_payload)'
```

`--order=desc` = newest first; read bottom-up for chronological order.

### 3. Strip junk with regex, then keep signal
```bash
... | grep -vE 'OutOfOrderError|tasks\.sqlite-journal|langmem_memories\.sqlite|Shutdown|SIGTERM|draining|/healthz?'
```
Add a `grep -E '<pattern>'` to keep only the error class you're chasing.

### 4. Collapse and cap
- Fold identical repeats: `... | sort | uniq -c | sort -rn`. A crash loop is one
  fact, not 200 rows.
- Still large? `--limit=20`, widen only if the tail looks truncated.
- A split multi-line traceback: widen the window just enough to capture the final
  exception line (which names the real error).

## Decision flow
errors last hour? → base filter + projection + noise grep → empty? widen
`--freshness` → still empty? drop to `>=WARNING` → collapse repeats → read the
tail → hand off to `troubleshooting-cloud-run` for root cause.

## Keep the noise pattern in sync
The exclusion regex here, in `tools/gcp_logging.py::DEFAULT_NOISE`, and CLAUDE.md's
"Known operational notes" must stay aligned. Update all three together.
