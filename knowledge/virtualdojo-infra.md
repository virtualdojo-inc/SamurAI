---
title: VirtualDojo Infrastructure
summary: Where SamurAI and the VirtualDojo services run on GCP — projects, the Cloud Run service, persistent storage, and the model backend.
tags: [infra, gcp, cloud-run]
updated: 2026-05-29
---

# VirtualDojo Infrastructure

SamurAI runs on **Cloud Run** service `samurai-bot` in project
`virtualdojo-samurai` (project number `1019610148219`), region `us-central1`.

## Projects
- `virtualdojo-samurai` — this bot's infrastructure.
- `virtualdojo-fedramp-dev` / `virtualdojo-fedramp-prod` — FedRAMP environments.

## Runtime
- Min instances 1 (always warm), max 20; 2Gi memory, CPU 1, gen2.
- Persistent storage: GCS FUSE bucket `samurai-bot-data` mounted at `/data`
  (SQLite for tasks + LangMem; the `raw/` conversation log lives here too).
- Conversation checkpoints are ephemeral (`/tmp/checkpoints.sqlite`).

## Models (Vertex AI)
- `gemini-3.5-flash` is served on the **global** endpoint only (404 at
  us-central1); `gemini-2.5-flash-lite` works at both. Probe `generateContent`
  to verify availability — do not assume. See [[deploy-pipeline]] for how
  changes reach production.

## Related
- [[deploy-pipeline]]
