# Code Sandbox — plan & state

SamurAI can generate-and-run Python in an isolated executor for computation,
data analysis, transforms, parsing, and codegen-and-test over data it has
**already fetched** with other tools, and reuse vetted prior scripts.

Decisions (locked 2026-06-21):
- **Dedicated sandbox service** (not a graph node, not in-process exec).
- **Pure compute over passed-in inputs** — no credentials, no network.
- **Hybrid approval** — the safety judge auto-reviews each script; anything that
  must persist to prod goes through the existing Approve/Revise/Reject card.

## Done (merged code — `feature/code-sandbox`, commit cc44c3a)

| Piece | File |
|-------|------|
| Executor service (isolated child, rlimits, wall-clock kill, scrubbed env, net seatbelt, bearer auth) | `sandbox/app.py`, `sandbox/Dockerfile`, `sandbox/requirements.txt` |
| `run_code` (judge-gated) + `find_prior_script` (pgvector reuse) | `tools/code_sandbox.py` |
| Keyword-gated `sandbox` tool group | `agent.py` |
| `run_code` classified as a write tool (judge + selftune) | `judge.py`, `selftune/evalset.py` |
| Real-execution isolation tests + tool tests | `tests/test_sandbox_service.py`, `tests/test_code_sandbox_tool.py` |
| Reuse-library schema | `db/models.py::CodeRun` (already modeled) |

Kill switch: `SAMURAI_SANDBOX_ENABLED` (off by default).

## Security audit (2026-06-21)

A pre-deployment adversarial red-team audit ran before go-live — see
`docs/sandbox_security_audit.md`. Verdict: **conditional GO**. The in-process
seatbelt is bypassable (by design — infra is the boundary). The two HIGH code
fixes (CODE-1 parent-memory balloon, CODE-2 post-kill hang) + CODE-3 hardening
are **DONE** (`feature/sandbox-hardening`). Go-live is gated on the infra
verifications **INFRA-1…6** below — especially blocking the metadata server
(INFRA-3) and PGA=False on a dedicated sandbox subnet (INFRA-4), which the
original plan did not specify.

## Pending — prod infra (needs approval; not yet run)

**1. Apply Alembic to prod** so `code_runs` + `pending_approvals` exist in prod
Cloud SQL (currently only on the local test container). Run in-VPC (private DB)
as a one-shot Cloud Run Job built from source, command `alembic upgrade head`
(reuse the `samurai-migrate` Job pattern from the data migration). Verify via
the admin `db_query` op: `select count(*) from code_runs`.

**2. Deploy `samurai-sandbox`** (project `virtualdojo-samurai`, us-central1):
- Service account `samurai-sandbox@…` with **zero IAM roles**.
- `gcloud run deploy samurai-sandbox --source sandbox/ --service-account=samurai-sandbox@… --ingress=internal --no-allow-unauthenticated --cpu=1 --memory=512Mi --concurrency=1 --min-instances=0 --max-instances=5`.
- **Egress denied:** route egress through the VPC (`--network`/`--subnet` +
  `--vpc-egress=all-traffic`) onto a subnet with no Cloud NAT / no default
  internet route, so the container cannot reach the internet. The in-process
  socket block in `app.py` is the defense-in-depth seatbelt, not the boundary.
- `SANDBOX_TOKEN` from a region-pinned (user-managed replication, us-central1 —
  org policy `constraints/gcp.resourceLocations`) Secret Manager secret, mounted
  on the sandbox service.

**3. Wire the bot:** grant `samurai-bot@…` read on the `SANDBOX_TOKEN` secret;
set `SANDBOX_URL` (sandbox internal URL), `SANDBOX_TOKEN`, and
`SAMURAI_SANDBOX_ENABLED=on` on `samurai-bot`. Smoke-test by calling the sandbox
directly from the VPC, then via the agent `run_code`.

## Future work

- **Loom video ingestion for the DH Tech Issue Tracker.** Tracker rows often
  carry a Loom video link. We want SamurAI to download/transcribe the Loom and
  feed a text/structured representation into the in-boundary Gemini agent so it
  can understand the reported issue. Approach to scope:
  - Detect Loom URLs on tracker rows (extend `kb/ingest_smartsheet.py` /
    `tracker_diagnostics.py`).
  - Resolve the Loom media: Loom's `https://www.loom.com/share/<id>` →
    transcript via Loom API if available, else the downloadable mp4
    (`/api/campaigns/sessions/<id>/transcoded-url`), captured in-boundary.
  - Transcribe in-boundary (Vertex Gemini multimodal can take audio/video, or
    Speech-to-Text) — **never** send tracker media to an external LLM (FedRAMP
    boundary).
  - Store the transcript alongside the row in `support/raw/` so the playbook
    compile + retrieval can use it; cite it as a log, not product fact.
  - Open question: Loom auth/download mechanics + whether the org's Looms are
    public/share-linked or workspace-private (affects the download path).
