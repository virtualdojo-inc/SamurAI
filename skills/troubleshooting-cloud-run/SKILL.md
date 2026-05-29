---
name: troubleshooting-cloud-run
description: Diagnose SamurAI / Cloud Run production issues from Google Cloud logs — correlate errors with deploys, separate real regressions from drain/shutdown noise, and verify root cause end-to-end before reporting. Use when investigating errors, crashes, failed deploys, alerts, latency, or unexpected bot behavior in Google Cloud.
---

# Troubleshooting Cloud Run

SamurAI runs on Cloud Run service `samurai-bot` (project `virtualdojo-samurai`,
region `us-central1`). Apply the project's **facts-only** rule: every claim must
be grounded in a live query, the code, or fresh docs — never "should be".

## 1. Establish the timeline first
- Get the currently-serving revision and recent revisions:
  - `gcloud run services describe samurai-bot --region us-central1 --format='value(status.traffic[].revisionName,status.traffic[].percent)'`
  - `gcloud run revisions list --service samurai-bot --region us-central1 --sort-by=~metadata.creationTimestamp --limit 5`
- A spike of errors right after a new revision points at a regression; a spike
  right before/at shutdown of an old revision is usually **drain noise**, not a
  bug. Check the revision name on each error line.

## 2. Read the actual logs (don't infer from one warning)
- `gcloud logging read 'resource.type="cloud_run_revision" AND resource.labels.service_name="samurai-bot" AND severity>=ERROR' --project virtualdojo-samurai --freshness=1d --format='value(timestamp,resource.labels.revision_name,textPayload)'`
- Filter out known non-SamurAI noise: the separate `gsa-scraper` service, and
  gcsfuse `OutOfOrderError` on `tasks.sqlite-journal` / `langmem_memories.sqlite`
  (expected, documented).
- A multi-line traceback may be split across log entries — widen the timestamp
  window to capture the final exception line, which names the real error.

## 3. Known failure signatures
- `on_turn_error ... INVALID_ARGUMENT ... number of function response parts is
  equal to the number of function call parts` → a tool-call turn has an
  unmatched function_call (e.g. the judge blocked one call in a multi-call batch
  without pairing siblings). Check `judge.py` sibling-skip handling.
- `NOT_FOUND ... publishers/google/models/<model>` → a decommissioned/unavailable
  Vertex model. Verify availability by probing `generateContent` directly, per
  the vertex-model-availability notes. `gemini-3.5-flash` is global-only;
  `gemini-2.0-flash-lite` is gone.
- `GraphRecursionError` / `recursion_limit_hit` → agent hit the 75 tool-call
  limit. One occurrence on a heavy task is tolerable; a recurring background task
  hitting it needs its scope narrowed.

## 4. Verify end-to-end, then report
- Prefer an end-to-end test over inference from a warning. Reproduce the failing
  path (hit `/health`, re-run the tool, query the live resource) before
  concluding.
- When a deploy is suspect, the CI pipeline already health-gates and
  auto-rolls-back; check the GitHub Actions run and the serving revision before
  assuming an outage.
