# SamurAI

SamurAI is the VirtualDojo team's AI-powered assistant, embedded directly in Microsoft Teams. Its purpose is to be a helpful, autonomous member of the team -- handling DevOps troubleshooting, GitHub workflow management, CRM queries, FedRAMP compliance, social media, and proactive follow-ups so the team can focus on building the product.

## Working principle: facts only, no assumptions

**Every claim you make must be grounded in something you can cite right now.** Approved sources:

- **Up-to-date online research** — fetch live docs (Microsoft Learn, vendor pages, RFCs, Microsoft Graph reference, etc.). Do NOT rely on training-time knowledge for anything that could have changed since cutoff — APIs, cmdlets, regional availability, license SKUs, UI navigation, default-on toggles.
- **This codebase** — read the file before claiming what it does. `git blame` / `git log` beat "should be" inferences.
- **Live system state** — query the actual tenant or environment with `az`, `gcloud`, `gh`, Exchange Online / Security & Compliance cmdlets, Microsoft Graph, Cloud Logging, BigQuery. Don't infer from documented defaults; check the running state.

If you can't verify, say *"I don't know — here's what I'd check"* instead of guessing. An unverified recommendation that fails costs more than a one-line admission of uncertainty. The user has explicitly asked for facts-based answers; "I'm not sure" is a valid answer when followed by a verification step.

**Red-flag phrases in your own drafts — stop and verify before sending:**
- "by default" — defaults vary by tenant, region, license tier, version
- "should be" / "usually" / "typically" — actual configurations often diverge from norms
- "should work" — that's a hypothesis, not a fact; run it or fetch the doc that confirms it
- "based on training" — training is not authoritative for current state

**For diagnoses, prefer end-to-end tests over inference from warnings or partial output.** A warning message is a clue, not a conclusion. Examples from this project's session history:
- A `WARNING: Encountered WebException while getting UDP policy` during `Set-Label` was initially diagnosed as "AIP/RMS service not activated." `Test-IRMConfiguration` then showed AIP was fully healthy in the GCC region; only two specific templates were archived. Inference from the warning was wrong; the end-to-end test was right.
- Azure Cloud Shell's PowerShell mode was recommended as a Windows-PowerShell environment to run the AIPService module. It actually runs PowerShell Core on a Linux backend and can't load Windows-only .NET Framework assemblies. Verifying the runtime beats assuming based on the product name.

When the user pushes back, or behavior surprises you, your prior chain of reasoning was probably built on an unverified link. Restart the diagnosis from a live observation, not from the prior chain.

## What SamurAI does

SamurAI is not just a chatbot. It is an autonomous agent that can investigate issues end-to-end, take action on behalf of the team, and follow up without being prompted.

### Troubleshooting and infrastructure
- Query Google Cloud logs, metrics, and Cloud Run service status across all environments
- View GCP billing/cost breakdown by service (read-only, via BigQuery billing export)
- Correlate errors with deployments by tracking revision names and timestamps
- Distinguish real regressions from draining/shutdown noise after deploys
- Sync and read source code from GitHub repos to trace bugs back to the code
- Cross-reference logs, code, and service status to deliver root cause analysis

### GitHub workflow
- Review PRs, list issues, check recent commits, view commit diffs across all virtualdojo-inc repos
- Create GitHub issues (always checking for duplicates first)
- Manage GitHub Projects V2 (create items, update Status/Priority fields)
- Suggest the `autofix` label on virtualdojo-inc/virtualdojo bugs when appropriate (with user approval)
- Close duplicate or erroneous issues (with a reason)

### CRM and business data
- Query VirtualDojo CRM data (contacts, accounts, opportunities, quotes, compliance records)
- Handle OAuth flow for user authentication to the CRM
- Read-only by default; creating/updating/deleting records requires human approval

### Communication and team coordination
- Send 1:1 Teams messages to team members
- Create scheduled/recurring background tasks that run autonomously
- Follow up on sent messages (e.g., check if someone reviewed a PR after being reminded)
- Escalate when things haven't been addressed after multiple attempts

### FedRAMP compliance
- Collect automated evidence from GCP (IAM, Cloud Run configs, KMS, audit logs, SCC findings)
- Generate and update OSCAL packages (SSP, POA&M)
- Review code against FedRAMP control families (SC-7, SC-12, CM-6, SC-18, AC-8)
- Track remediation SLAs and flag overdue items

### Social media
- Draft, preview, schedule, and publish posts to LinkedIn, X/Twitter, and other platforms via Ayrshare
- Generate images with VirtualDojo brand colors
- Enforces preview-before-publish flow; only Cyrus and Devin can approve posts

### File handling
- Process uploaded spreadsheets (Excel/CSV) -- fill columns, edit specific cells, return modified files
- Always verifies changes with read-back before reporting success

## Autonomy rules

SamurAI acts independently on read-only operations, communications, and scheduling. It requires explicit human approval (Devin or Cyrus) before:
- Modifying production infrastructure or deploying services
- Creating/closing/merging GitHub PRs
- Modifying CRM records
- Publishing social media posts
- Deleting persistent data

**Exception — autonomous self-improvement pipeline (CI, not the runtime bot):**
The nightly self-improvement GitHub Actions pipeline is authorized to open,
review, auto-merge, and deploy changes **without human approval**, but ONLY
within a tightly bounded scope: changes limited to `skills/**` and `tests/**`,
enforced by a hard path-allowlist CI gate, with the full test suite as a
non-negotiable gate and a Claude reviewer approval required before merge. Merges
deploy via the blue/green, health-gated, auto-rollback deploy workflow. Core code
(`agent.py`, `app.py`, `tools/`, infra) is out of scope and still requires a
human. The pipeline is gated by the `SELF_IMPROVE_ENABLED` repo variable (kill
switch) and the `ANTHROPIC_API_KEY` secret. See "Skills and self-improvement".

## Tech stack

- **Runtime**: Python 3.12, aiohttp, Microsoft Bot Framework SDK
- **AI**: LangGraph agent with Google Gemini (`gemini-3.5-flash`), LangChain tools
- **Scheduling**: APScheduler (AsyncIOScheduler) for background tasks
- **Persistence**: SQLite on GCS FUSE mount (`/data`) for tasks, conversation refs, team roster
- **Memory**: LangMem three-tier memory (core/team/user) with background extraction
- **Hosting**: Google Cloud Run (project: `virtualdojo-samurai`, region: `us-central1`)

## Key architecture

- `app.py` -- Bot entrypoint, message routing, Teams integration, error handling
- `agent.py` -- LangGraph agent graph, system prompt, tool binding, `run_agent()` entry point
- `scheduler.py` -- APScheduler background task execution, conversation ref resolution, retry logic
- `task_store.py` -- SQLite persistence for tasks, conversation refs, team roster
- `tools/` -- All agent tools:
  - `gcp_logging.py`, `gcp_cloudrun.py`, `gcp_monitoring.py` -- GCP infrastructure + billing
  - `github.py` -- GitHub issues, PRs, commits, commit diffs, projects
  - `repo_sync.py` -- Sync and read source code from GitHub repos
  - `background_tasks.py` -- Create/manage scheduled tasks
  - `teams_messaging.py` -- Send 1:1 Teams messages
  - `virtualdojo_mcp.py` -- VirtualDojo CRM integration
  - `social_media.py` -- Ayrshare social media publishing
  - `fedramp.py`, `fedramp_oscal.py`, `fedramp_docs.py` -- FedRAMP compliance
  - `file_handler.py` -- Spreadsheet processing
  - `google_search.py` -- Web search
  - `database.py` -- Database tools
- `cards/` -- Adaptive Card builders and action handlers (social media previews, etc.)
- `memory.py` -- LangGraph checkpointing and LangMem three-tier memory store

## GitHub repos SamurAI can access

- `virtualdojo-inc/virtualdojo` -- Main data service (FastAPI + Vue.js CRM)
- `virtualdojo-inc/virtualdojo_cli` -- VirtualDojo CLI tool
- `virtualdojo-inc/SamurAI` -- This bot
- `virtualdojo-inc/Fedramp` -- FedRAMP compliance documentation and OSCAL packages

## Autofix label (virtualdojo-inc/virtualdojo)

The `autofix` label on virtualdojo-inc/virtualdojo issues triggers an automated Claude-based TDD bug fix attempt (via the `claude_automation/bugfix/` workflow in that repo). SamurAI may suggest applying the label but must never apply it without explicit user approval.

Good candidates for autofix:
- Backend data/logic bugs with a clear error trace (NOT NULL violations, type mismatches, missing defaults, query filter bugs, wrong field references)
- API endpoint bugs where the error and expected behavior are unambiguous
- Regex/pattern matching fixes (error sanitization, input parsing)
- Missing or incorrect DB column defaults, constraints, or migrations
- Off-by-one errors, wrong status codes, missing null checks
- Test gaps where the fix is adding coverage for an existing behavior

Bad candidates for autofix:
- Frontend/UI bugs (Vue components, CSS, layout) -- requires visual verification
- Multi-tenant authorization or access control changes -- too security-sensitive
- Alembic migrations on production data -- need manual review and rollback planning
- Business logic changes that require product/UX decisions
- Performance issues -- profiling needed, not just code changes
- Anything touching payment, compliance, or PII handling

## Three-tier memory system

SamurAI learns from every interaction through three memory tiers:

| Tier | Namespace | Who reads it | What's stored |
|------|-----------|-------------|---------------|
| **Core** | `("core",)` | Everyone (including future external users) | Successful tool patterns, troubleshooting recipes, error resolutions |
| **Team** | `("team", "virtualdojo")` | VirtualDojo team only | Project decisions, infrastructure facts, internal processes |
| **User** | `("memories", "{user_id}")` | That user only | Personal preferences, communication style, role context |

After each conversation, three background extractors run automatically to populate each tier. The bot also has explicit memory tools (`manage_core_memory`, `manage_team_memory`, `manage_memory`) to save knowledge during conversations. All tiers are searched and injected into the system prompt on every message.

Tool calls and their outcomes are logged and included in the extraction payload, so the core extractor can learn successful multi-tool patterns over time.

## GCP projects

- `virtualdojo-samurai` -- This bot's infrastructure
- `virtualdojo-fedramp-dev` -- FedRAMP dev environment
- `virtualdojo-fedramp-prod` -- FedRAMP production environment

## Deployment

Deploy to Cloud Run using source-based deployment (builds via Dockerfile):

```bash
gcloud run deploy samurai-bot --source . --region=us-central1 --project=virtualdojo-samurai
```

This preserves all existing config (env vars, secrets, volume mounts, scaling). Only the container image is updated.

If auth has expired: `gcloud auth login`

### Cloud Run configuration

- Min instances: 1 (always warm), Max instances: 20
- Memory: 2Gi, CPU: 1, CPU throttling: disabled
- Persistent storage: GCS FUSE bucket `samurai-bot-data` mounted at `/data`
- Execution environment: gen2, startup CPU boost enabled

### Required secrets and env vars for Bot Framework auth

The bot uses a **SingleTenant** Azure Bot Service registration. All three of these must be set correctly or the bot will silently fail to respond on Teams (`Unauthorized` on all outbound calls):

| Env var | Source | Notes |
|---------|--------|-------|
| `MICROSOFT_APP_ID` | GCP Secret Manager (`ms-app-id`) | Azure AD app registration client ID: `35e1851a-0377-47f3-8b47-09110fec743c` |
| `MICROSOFT_APP_PASSWORD` | GCP Secret Manager (`ms-app-password`) | Azure AD app client secret — must match a valid credential on the app registration |
| `MICROSOFT_APP_TENANT_ID` | Cloud Run env var (not a secret) | Must be set to `a0a6af2b-e398-4029-94c7-5fbae193405f` — without this, the SDK authenticates against the wrong (multi-tenant) endpoint and all outbound calls fail |

**Troubleshooting "Unauthorized" errors**: If the bot receives messages but can't reply, check these in order:
1. `MICROSOFT_APP_TENANT_ID` is set on the Cloud Run service (this was missing once and caused a full outage)
2. The client secret in GCP Secret Manager matches a valid credential on the Azure app registration (`az ad app credential list --id 35e1851a-0377-47f3-8b47-09110fec743c`)
3. The Azure Bot Service app type (`az bot show --name samurai-dojo-bot --resource-group samurai-rg --query 'properties.msaAppType'`) is `SingleTenant`

## Skills and self-improvement

### Skills (`skills/` + `skills.py`)
SamurAI has an Agent-Skills capability modeled on Anthropic's Agent Skills.
A skill is a directory `skills/<name>/SKILL.md` with YAML frontmatter
(`name` + `description`) and a markdown body of procedural knowledge.
Progressive disclosure:
- **Level 1 (always in prompt):** every skill's `name` + `description`, injected
  via `skills.skills_catalog_text()` into the system prompt in
  `agent._select_prompt_sections`.
- **Level 2 (on demand):** the full `SKILL.md` body, fetched by the `get_skill`
  tool (a core, always-available tool) when the agent judges a skill relevant.

`skills.py` validates frontmatter (name `[a-z0-9-]` ≤64 chars, no reserved words
`anthropic`/`claude`; non-empty description ≤1024 chars) and skips malformed
skills with a warning rather than crashing. Skills are plain files so they can be
tuned via ordinary PRs — this is the surface the nightly self-improvement loop
edits.

### CI/CD (`.github/workflows/`)
- `deploy.yml` — on push to `main`: full test suite gates a **blue/green** deploy
  (deploy candidate with `--no-traffic --tag`, health-check `/health` on the
  candidate's tagged URL, promote to 100% only if healthy, re-verify prod, and
  **auto-rollback** to the previous revision if the post-promote check fails).
  Auth is keyless via Workload Identity Federation. A bad build cannot take the
  bot down. (Infra details in the memory note `project_samurai_cicd`.)
- `nightly-self-improve.yml` — cron nightly: reads last-24h GCP logs + current
  skills, makes ONE scoped improvement to `skills/**` (+ tests), runs tests, and
  opens a PR labeled `self-improve`.
- `claude-pr-review.yml` — on `self-improve` PRs: hard path-allowlist gate
  (`skills/**`, `tests/**` only) → full tests → Claude review → auto-merge
  (squash) on approval marker. Human PRs are never auto-merged here.
- `deploy-troubleshoot.yml` — on a failed deploy: Claude reads the failed run
  logs, and either opens a scoped fix PR (`self-improve`) or files an issue if
  the fix is risky.

**Enabling the loop:** set repo variable `SELF_IMPROVE_ENABLED=true` (kill
switch) and add the `ANTHROPIC_API_KEY` repo secret. Scope is enforced in CI,
not just by prompt. The Claude-in-CI agents use `anthropics/claude-code-action@v1`
pinned to `claude-sonnet-4-6`.

## Running tests

```bash
python -m pytest tests/ -v
```

## Hallucination mitigation (Chain-of-Verification)

SamurAI now has a CoVe-style verification node between the agent loop and
the final response. Purpose: catch fabricated specifics (line numbers,
counts, API names, file paths) before they reach the user. Motivated by
observed accuracy gap vs Claude Code on code-analysis tasks —
see the research log below and the thread in SamurAI session notes for
Apr 2026.

### Files
- `verification.py` -- the verifier node. Runs a separate Flash call in a
  fresh context with only the draft + tool trace. Returns JSON grading
  each specific claim as grounded / ungrounded / unverifiable.
- `agent.py` -- wires the node into the LangGraph as a conditional step
  between `agent` (when it produces a non-tool response) and `END`.

### Configuration (env var `SAMURAI_VERIFY_MODE`)
- `off` (default): verification skipped entirely. Zero overhead.
- `shadow`: verification runs, logs what it *would* have rejected, but
  passes the draft through unchanged. **Use this first** to collect data
  on what the verifier catches before flipping to enforce.
- `enforce`: verification runs; ungrounded claims route the graph back
  to the agent with a structured correction asking it to verify or drop
  the claim.

### Rollout plan
1. Deploy with `SAMURAI_VERIFY_MODE=shadow` first.
2. Watch Cloud Logging for `[verification.shadow]` entries for a week.
   Grep: `resource.type="cloud_run_revision" jsonPayload.message=~"verification.shadow"`
3. Spot-check the flagged claims. If the verifier is accurate, flip to
   `enforce`. If it's over-flagging (false positives), tune the
   verifier prompt in `verification.py:VERIFIER_SYSTEM_PROMPT` before
   flipping.

### What was deferred (for a future session to pick up)

1. **System prompt pruning.** `agent.py:SYSTEM_PROMPT` is 340+ lines.
   Research (RAG-MCP, arXiv 2505.03275) shows 3.2x accuracy improvement
   from pruning bloated prompts. Suggested approach: mirror the existing
   `_select_tool_groups()` pattern with `_select_prompt_sections()` that
   loads FedRAMP / OSCAL / Social / Background Tasks sections only on
   keyword match. Keep ~60-100 lines of always-on core rules. Deferred
   to a separate PR to keep this change focused.

2. **Explicit "evidence before claims" rules in the system prompt.**
   Complementary to the verifier node. Add a short block at the top of
   `SYSTEM_PROMPT` forbidding specific-number / line-number claims
   without a supporting tool call in the same turn. Keeps the verifier
   from having to catch as many issues on the back end.

3. **Verifier prompt tuning.** After shadow data comes in, expect the
   verifier prompt (`VERIFIER_SYSTEM_PROMPT`) to need iteration on what
   counts as "a claim" worth checking.

### Research basis (published, Apr 2026)
- **Chain-of-Verification** (Dhuliawala et al., arXiv:2309.11495, ACL
  Findings 2024): +23% F1 when verification runs in a *fresh* context,
  not as self-critique. Implemented here as a separate Flash call with
  only the draft + tool log in scope.
- **RAG-MCP prompt bloat study** (arXiv:2505.03275): tool-selection
  accuracy 13.6% -> 43.1% from pruning a bloated prompt. Motivates the
  deferred prompt-prune work.
- **Gemini function-calling hallucination bug** (googleapis/python-genai
  #813): model can claim tool output that was never returned. The
  verifier's independent-context design is what catches this.
- **Artificial Analysis Omniscience Index**: Gemini 3 Pro ~88% hallucination
  rate on knowledge tasks vs Claude 4.5 tier materially lower. Gap is
  partly trained-in (Anthropic's Constitutional AI explicitly trains
  epistemic humility) -- the verifier narrows but does not close it.

### Cost
One extra Flash call per agent turn that produces a non-tool response.
Typical latency +200-400ms. Token cost ~1-2k tokens per verification.
Skips verification on turns with no tool calls (greetings, clarifying
questions) since there's nothing to ground against.

### If you are a future Claude session asked to adjust this
- The three deferred items above are the natural next increments.
- Do NOT extend verification to rewrite the draft -- the verifier can
  only flag. Only the main agent produces user-facing text. This is a
  deliberate design choice to keep failure modes scoped.
- Before making changes, check Cloud Logging for shadow-mode data.
  Tune based on evidence, not vibes.

## Known operational notes

- APScheduler runs in-process; jobs are in-memory and rebuilt from SQLite on restart
- Recursion limit is 75 for both interactive and background tasks
- One-shot tasks get 1 automatic retry on failure (60s delay)
- Conversation refs are resolved through `bg_task_` parent chains for sub-task delivery
- Background tasks are tagged with `is_background_task=True` so the agent executes directly without conversational back-and-forth
- Tool calls are logged to stdout (`[agent] tool_calls` and `[agent] tool_result`) for observability in Cloud Logging
- If the bot hits the recursion limit, it asks the user what to focus on instead of failing silently
- SQLite over GCS FUSE shows occasional `OutOfOrderError` on journal files -- this is expected
- GCP billing export is configured on `virtualdojo-samurai.billing_export` (table populates daily, env var: `GCP_BILLING_TABLE`)
