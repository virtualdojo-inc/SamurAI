# SamurAI Tech-Issue Triage (pre-computed diagnostics) — Plan

**Status:** Draft for review · **Last updated:** 2026-06-20 · **Owner:** Devin

## Purpose

Diagnosing a DH Tech Issue Tracker item takes SamurAI a while — it has to read the
row, cross-reference Cloud Logging, read code, form a hypothesis, and pressure-test
it. Today that latency lands on whoever asks. The fix: **do the diagnostic work
ahead of time, as rows come in, and park the results** so that when Devin, Vedant,
Jason, or anyone else opens SamurAI, the fact-grounded analysis and recommendation
are already done and instant to engage with.

**Hard constraint (from Devin):** SamurAI does **fact-finding and plan-making
only**. It takes **no action** — files nothing, changes nothing in the tenant or
GitHub — without a human present who approves it. The background pipeline is
read-only. Every claim it parks must be **grounded in a fact it can cite**
(a log query + entries, a code location, an issue ref) — never an assumption.

This plan is grounded in (1) the actual SamurAI codebase (verified by reading it),
(2) the FedRAMP in-boundary + human-approval rules in `CLAUDE.md`, and (3) the live
Smartsheet webhook API docs (see **References**).

---

## What it does (end state)

A background worker, running **frequently during business hours (M–F, ~8am–5pm)**,
watches the DH Tech Issue Tracker (sheet `1146352141553540`). For each **new or
changed** row it:

1. **Diagnoses** (read-only) — cross-references the symptom against Cloud Logging
   and repo code, forms a likely cause and a candidate fix.
2. **Adversarially checks** the candidate fix — spawns `investigate()` sub-agents
   *prompted to refute it* against the same evidence; revises or discards if the
   refutation holds. (This is the in-runtime analog of an ultracode refute-loop —
   SamurAI is a Gemini LangGraph agent, not Claude Code, so it emulates the pattern;
   it does not literally run ultracode.)
3. **Categorizes** the item into exactly one of:
   - **(A) Tenant config tweak** — a setting change in the tenant.
   - **(B) Config change needing Devin's review** — higher-risk; needs sign-off.
   - **(C) Future feature requirement** — would map to a GitHub `Feature`.
   - **(D) Backend code bug** — would map to a GitHub `Bug`; candidate for the
     existing `controlled-issue-fix` path.
4. **Grounds + stores** — the finished diagnosis runs through the existing
   verification node (`verification.py`) in enforce mode; **only fact-grounded
   diagnoses are stored.** Stored to a SQLite table on `/data`, keyed by row id +
   a content hash, with the suggested GitHub issue type + priority.

Then, when **any team member** engages SamurAI about the tracker (or a specific
item), SamurAI serves the parked diagnosis **instantly** via a read-only tool, and
the human decides what to do. **Any consequential action — filing the issue,
applying a config change — happens only then, with the human present, through the
normal judge-gated approval path.** The standing "confirm before filing" rule is
preserved structurally: the unattended pipeline *cannot* file or change anything.

---

## Trigger: poll during business hours (webhook is a clean phase-2 swap)

**Decision (v1): poll via the existing scheduler.** A recurring background task
(`tools/background_tasks.py` → `scheduler.py`) ticks every ~10 minutes during
business hours and processes only changed rows. This reuses proven machinery and
adds **no public attack surface**.

Why not webhook first — and why it's an easy later swap:

- **Smartsheet does support webhooks** (`POST /webhooks`, API v2.0.0): they attach
  to a **sheet**, fire on row/cell change, are **inactive until enabled**, and
  enabling triggers a **verification handshake** (Smartsheet calls the endpoint with
  a challenge the callback must echo). They require a **public HTTPS callback** and
  are capped at **100 webhooks per plan**. (See References.)
- SamurAI already runs a public HTTPS aiohttp service on Cloud Run (`app.py`,
  min-instances=1, always warm), so a `/smartsheet-webhook` route + handshake +
  shared-secret check is feasible — but it's more surface area and security to get
  right up front, for a latency win that 10-minute polling largely already gets.
- The webhook payload only says *"row X changed"* — you still call
  `smartsheet_get_sheet` to read it. So **the diagnostic worker is identical** for
  poll or webhook; phase 2 just changes what *calls* it. No rework.

**Timezone caveat (verified gap):** APScheduler cron in `scheduler.py` is
**UTC-only** — there's no per-task timezone. "M–F 8–5pm" must be written in UTC
(e.g. for US-Central, roughly `*/10 13-23 * * 1-5`). The exact offset + DST handling
is an open item to pin at build time (the scheduler stores a per-task timezone but
does not apply it to cron — confirm before relying on it).

---

## Where the diagnostics are stored

**A new SQLite table on the existing `/data` GCS-FUSE mount** — the same database
`task_store.py` already owns (`/data/tasks.sqlite`) — keyed by `tracker_row_id` +
a **content hash of the row**. The hash is the dedup/watermark key: recompute a row
only when its content actually changes, so ticks are cheap and idempotent.

Proposed columns (mirroring the `task_store` schema conventions, lines ~31–51):

| column | purpose |
|---|---|
| `row_id` (PK) | Smartsheet `_row_id` (stored as **string** — 16-digit IDs lose precision as numbers; this is why `tools/smartsheet.py` stringifies them) |
| `row_hash` | content hash of the row; mismatch ⇒ stale ⇒ recompute |
| `status` | `pending` / `diagnosed` / `stale` / `skipped` |
| `category` | A / B / C / D (see above) |
| `diagnosis_json` | the grounded payload: symptom, **log evidence (the exact `query_cloud_logs` filter + entries)**, code refs, candidate fix, adversarial-check result, suggested issue type + priority |
| `model` | model used (for the in-boundary audit trail) |
| `computed_at` | timestamp |

**Why `/data` SQLite and not the knowledge bucket:**

- These are **ephemeral operational work-products**, recomputed as rows change —
  not curated durable knowledge. The KB bucket's `support/playbooks/` is for
  distilled *patterns*, and its own rule (`CLAUDE.md`) is that a resolved ticket is
  *a historical record, not a fact*. Live per-row diagnostics don't belong in that
  tier.
- **Compliance:** the diagnosis cross-references Cloud Logging (in-boundary data),
  so it **must** stay in-boundary. `/data` (GCS-FUSE `samurai-bot-data`) is
  in-boundary and is already where `conversation_log.py` writes turn logs
  (`/data/raw/<date>/*.json`). Same boundary, established pattern.
- Reuses `task_store`'s proven SQLite-on-FUSE patterns, including the documented
  `OutOfOrderError` tolerance and the atomic `try_lock`/`unlock` for safe
  cross-instance access.

**Serving:** a new **read-only** tool `get_tracker_diagnostics` (no judge needed —
it's a read), plus a one-line prompt index ("N tracker items have prepared diagnoses
ready") injected exactly the way `wiki.py:knowledge_index_text()` is injected via
`agent.py:_select_prompt_sections()`. So a person asking about the tracker gets the
parked analysis with no diagnostic latency.

---

## The diagnostic flow (read-only, fact-grounded)

Lives in a `SKILL.md` (`tech-issue-triage`) so it is observable and editable, and is
invoked both by the background worker's prompt and on-demand by a person. Reuses
tools that already exist — nothing new is needed to *read*:

1. **Read the row** — `smartsheet_get_sheet(sheet_id="1146352141553540")`. Discover
   columns at runtime (no hard-coded schema); identify the symptom text, any
   existing `Github Issue No`, priority/status columns.
2. **Cross-reference logs** — `query_cloud_logs(filter, project_id)` across the
   relevant projects; correlate with code via `sync_repo` / `search_repo_code` /
   `read_repo_file_range`; orient with the engineering wiki (`search_wiki`).
3. **Form cause + candidate fix** — optionally fan out 2–4 `investigate()` calls
   (parallel, read-only) for competing hypotheses.
4. **Adversarial refute pass** — spawn `investigate()` sub-agents prompted to
   *disprove* the candidate fix against the evidence; revise or discard.
5. **Categorize** — A / B / C / D; for C/D suggest the GitHub issue **type**
   (`Feature` / `Bug`) and a **priority** (P0–P3), per the existing
   `github-issue-triage` skill's rules. (`github_create_issue` already sets native
   type + priority + Project #2 — used only later, with a human.)
6. **Grounding gate** — run the finished diagnosis through `verification.py`
   (enforce mode: `verify_response(draft, messages)` / `verification_node`); store
   **only** if grounded. Anything with ungrounded specifics is dropped/flagged, not
   parked. This is the cheapest enforcement of "everything grounded in facts."

> **Compliance note:** generation uses `investigate()` (Gemini). Confirm at build
> time that the log-touching diagnostic calls run on an **in-boundary** region, not
> the global endpoint. This is the same open data-residency caveat already noted in
> `CLAUDE.md` for the KB-serving chat model — flag, don't silently assume.

---

## Safety & compliance model

- **The pipeline is read-only.** It calls only read tools
  (`smartsheet_get_sheet`, `query_cloud_logs`, repo reads, `investigate`) plus the
  one write to its own `/data` diagnostics table. It never calls a tenant-write,
  GitHub-write, or CRM-write tool. There is therefore **no autonomous action** to
  gate — by construction.
- **Action happens only with a human present**, when they engage with the parked
  diagnosis and approve filing/applying. That action flows through the **existing
  fail-closed judge** (`judge.py`) like any other write — no new approval apparatus
  needed for v1.
- **In-boundary throughout:** in-boundary log data → in-boundary diagnosis →
  in-boundary `/data` store. Nothing crosses to an external LLM or out-of-boundary
  runner. (Contrast with `controlled-issue-fix`, which deliberately *does* cross to
  CI and is therefore field-allowlisted; this pipeline does not cross, so it can use
  log/code freely **as long as it stays in-boundary**.)
- **Untrusted input:** Smartsheet cell values are attacker-influenceable free text.
  The diagnostic worker treats row content as data to analyze, never as
  instructions. (The repo-wide inbound prompt-injection probe is a known deferred
  item in `docs/judge-design.md`; this pipeline's read-only nature limits the blast
  radius — the worst case is a wasted/incorrect parked diagnosis a human reviews,
  not an action.)

---

## Build sequence — Phase 1

Smallest useful piece first; each ships value and is verifiable on its own.

### Build 1 — `tech-issue-triage` skill (read-only; ships value immediately)
A `skills/tech-issue-triage/SKILL.md` encoding the diagnostic flow above (steps
1–6), the A/B/C/D categorization rules, and the honest capability boundary
("diagnose + recommend only; never files or changes anything"). Reuses the existing
read tools. Add a keyword trigger to `_select_tool_groups` / `_select_prompt_sections`
if needed.
- **Done when:** asked about a real tracker row interactively, SamurAI produces a
  correctly-categorized, log-cited diagnosis with a candidate fix that survives the
  refute pass; declines to assert anything it can't ground.

### Build 2 — `/data` diagnostics store
A small store (new table in `task_store.py`, or a sibling module reusing its
connection/locking) with: upsert-by-`row_id`, `row_hash` dedup, `get_pending`,
`mark_stale`, and a read for serving. Plus the read-only `get_tracker_diagnostics`
tool and the `knowledge_index_text`-style prompt line.
- **Done when:** diagnoses persist across a restart; a changed row hash marks the
  row stale; the serving tool returns a parked diagnosis with its citations; a
  person querying the tracker is served it with no diagnostic latency.

### Build 3 — the background worker + schedule
The worker: list tracker rows → diff hashes → for each new/changed row, run the
skill → grounding gate → store. Registered as a recurring background task
(`create_background_task`) on the business-hours UTC cron, single-flight via
`task_store` locking so overlapping ticks don't double-diagnose.
- **Baked in:** bounded batch per tick (cap N rows) so a tick is cheap and a
  Cloud Run drain costs ≤ one small batch — mirrors the KB compile's
  interruption-tolerant batching.
- **Done when:** a new tracker row gets a parked diagnosis within one cadence window;
  unchanged rows are skipped; the worker survives a redeploy and resumes; nothing is
  filed or changed.

### Build 4 — (optional, measure first) tighten serving + freshness
Mark a parked diagnosis **stale** when its row changes (so a person is never served
outdated analysis), and surface staleness in the served result. Add a digest
("3 new tracker items diagnosed since yesterday") a person can pull on demand.

---

## Phase 2 — deferred

- **Smartsheet webhook** — `/smartsheet-webhook` route in `app.py` + verification
  handshake + shared-secret/HMAC check; flip the trigger from poll to push. Same
  worker; near-real-time. (100-webhook/plan limit is not a constraint at one sheet.)
- **>100-row handling** — `smartsheet_get_sheet` defaults to `max_rows=100` with **no
  pagination** (verified gap). If the tracker exceeds 100 active rows, add a
  status/open filter or paginate. Pin this when we see the live row count.
- **Action-from-card** — if a team member should be able to file the issue / apply a
  (B)-class change directly from a Teams card, add an Approve card with **clicker
  re-verification** (`TeamsInfo.get_member` against a hardcoded approver set) — the
  same hardening deferred in the controlled-issue-fixer Phase 2. Until then, action
  is taken conversationally with the human present and judge-gated.

---

## Open items / prerequisites

- [ ] Confirm `investigate()` / log-touching diagnostic calls run **in-boundary**
      (region), not on the global Vertex endpoint (data-residency).
- [ ] Pin the business-hours cron in **UTC** + decide DST handling (scheduler cron is
      UTC-only; per-task timezone is stored but not applied to cron — verify).
- [ ] Read the DH Tech Issue Tracker live to capture its **actual columns**
      (symptom, status, priority, `Github Issue No`) and live **row count** (>100?).
- [ ] Decide the per-tick **batch cap** and the cadence (start `*/10`, relax later).
- [ ] Decide content-hash inputs (which columns count toward "changed").

---

## References

Smartsheet webhooks (verified against live docs, 2026-06-20):
- Webhooks API (support, scope=sheet, inactive-by-default, 100/plan) —
  https://developers.smartsheet.com/api/smartsheet/openapi/webhooks
- Create Webhook (`POST /webhooks`, v2.0.0, APIToken/OAuth2) —
  https://developers.smartsheet.com/api/smartsheet/openapi/webhooks/createwebhook

---

## Appendix — codebase anchors

- **Skill mechanism:** `skills.py` (`_parse_skill_md`, `load_skill_catalog`,
  `skills_catalog_text`, `get_skill`); skills live in `skills/<name>/SKILL.md`.
  Model the new skill on `skills/github-issue-triage/SKILL.md` (type/priority rules)
  and `skills/controlled-issue-fix/SKILL.md` (honest capability boundary).
- **Read tools (reuse, no new writes):** `tools/smartsheet.py`
  (`smartsheet_get_sheet`, sheet id `1146352141553540`, string IDs, fuzzy columns,
  `max_rows=100` no-pagination); `tools/gcp_logging.py` (`query_cloud_logs`);
  `tools/repo_sync.py` (`sync_repo`/`search_repo_code`/`read_repo_file_range`);
  `tools/investigate.py` (`investigate` — parallel, read-only); `wiki.py`
  (`search_wiki`/`read_knowledge`, `knowledge_index_text` injection pattern).
- **Grounding gate:** `verification.py` (`get_verify_mode`, `verify_response`,
  `verification_node`; modes off/shadow/enforce via `SAMURAI_VERIFY_MODE`).
- **Storage:** `task_store.py` (`/data/tasks.sqlite`, schema ~31–51, `try_lock`/
  `unlock`, error/auto-pause patterns); `conversation_log.py` (`/data/raw/...`
  precedent for in-boundary `/data` persistence).
- **Schedule:** `scheduler.py` (APScheduler, `CronTrigger.from_crontab`, **UTC-only
  cron**, single-flight locking, drain-tolerant); `tools/background_tasks.py`
  (`create_background_task` + cron validation).
- **Later, with a human:** `tools/github.py` (`github_search_issues` dup-check;
  `github_create_issue` sets native Bug/Feature/Task type + P0–P3 priority +
  Project #2); `judge.py` (`WRITE_TOOL_NAMES`, fail-closed) gates any such write.
- **Sibling feature (compose, don't duplicate):** `docs/controlled_issue_fixer_plan.md`
  + `skills/controlled-issue-fix/` — the (D) backend-code-bug path hands off here.
