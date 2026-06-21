# SamurAI Controlled Issue-Fixer — Plan

**Status:** Draft for review · **Last updated:** 2026-06-05 · **Owner:** Devin

## Purpose

Give SamurAI the ability to do what the `autofix` label was *supposed* to do —
attempt an automated, test-driven bug fix that yields a reviewable PR — but in a
**controlled, step-gated, observable** way that SamurAI owns, rather than a
fire-and-forget label that kicks off an opaque black-box run.

This plan is grounded in three things: (1) the actual SamurAI + `virtualdojo`
codebase (verified by reading the code and the live GitHub App / branch-protection
state), (2) the FedRAMP in-boundary and human-approval rules in `CLAUDE.md`, and
(3) a fact-checked survey of how the proven production systems and the program-repair
research actually do this (see **References**).

---

## What it does (end state)

In Teams, a developer says **"fix issue #123"** (or SamurAI proactively flags a
good candidate — deferred to Phase 2). SamurAI then:

1. **Localizes + classifies** — finds the root-cause files and decides whether the
   issue is an eligible candidate.
2. **Proposes** — posts a short fix plan (root-cause files, approach, eligibility).
3. **Builds + tests the fix** in a bounded, sandboxed CI run (the developer's
   request is the authorization in v1).
4. **Opens a draft PR** to `development` — **only if the fix is green**.
5. **Reports back** — PR link + diff summary + test result (or a clean "couldn't
   fix it, here's why").
6. **Developers review and merge** through GitHub's standard protected-branch flow.
   **SamurAI never merges.**

Same capability as the autolabel, but a human gate sits *before* any code is
written, the codegen is pinned/bounded instead of a 50-turn free-for-all, and
every step is observable.

---

## Why this shape is correct (it's the proven production pattern)

The exact flow — *assign an issue → agent produces a tested PR → human merges* —
is what the two shipping reference systems do, and both enforce the no-merge gate
**structurally, not by policy**:

- **GitHub Copilot coding agent**: works in an ephemeral GitHub-Actions sandbox,
  runs tests/linters, pushes to its *own* branch, opens a **draft PR**, and
  *cannot merge or self-approve*. It can only push to branches it created, opens
  one PR per task, and **the requester cannot be the approver**. Merge power lives
  in branch protection / required reviews / CODEOWNERS.
  ([GitHub blog][copilot-blog], [GitHub docs][copilot-docs])
- **Anthropic Claude Code Action**: by default commits to a new branch and returns
  a prefilled PR-creation link; **it cannot approve PRs**, and tool access is
  restricted via `--allowedTools`. ([repo][cca], [security docs][cca-sec])

So "bounded run → branch → PR to a protected branch → developer merges" isn't us
being timid; it's the enforced design of both production systems. We adopt two of
their specifics directly: **open the PR as a draft**, and rely on the structural
**requester ≠ approver** gate.

---

## Where the code gets written: dispatched CI, not in-process

**Decision: the code-writing run executes in a dispatched GitHub Actions job in
`virtualdojo-inc/virtualdojo` — never in-process on `samurai-bot`.** Two reasons:

1. **Compliance.** `CLAUDE.md`'s in-boundary rule scopes to the
   `gs://virtualdojo-knowledge` *bucket data* (that's why the KB compile runs
   in-process on Vertex). Ordinary product source is not bucket data, and it is
   *already* sent to Claude on a GitHub runner by both `autofix-bug.yml` and our
   own `claude-pr-review.yml`. Running codegen there introduces **no new boundary
   crossing**.
2. **Blast radius.** A dispatched run is sandboxed — ephemeral runner, scoped
   `GITHUB_TOKEN`, writes confined to a `bugfix/issue-N` branch off a protected
   base. Running a 25-turn `Edit/Write/Bash` agent *in-process* would couple it to
   the live bot's secrets, the `/data` GCS-FUSE mount, and the production Cloud Run
   identity — the opposite of controlled.

`samurai-bot` stays **read-only + exactly one new write (a dispatch)**: it
localizes, proposes, fires one bounded run, then observes.

> **Hard rule (compliance):** the localization brief SamurAI sends into CI must be
> built **only** from repo source + the issue number/title + a scrubbed issue body
> — **never** from Cloud Logging, CRM, or KB content. It crosses from a
> (potentially in-boundary-touching) bot into an out-of-boundary runner, so the
> brief is field-allowlisted at the source. (Bonus: precise localization keeps the
> brief small, which shrinks this surface — see best practice #4.)

---

## Safety model

The autolabel's weakness was a fail-open judge that couldn't even see the diff. We
close that two ways:

- **The judge goes fail-closed** (separate change, in progress). That turns it from
  best-effort theater into a real gate on the agent's dispatch call. Because of
  this, the MVP can **defer** the heavier approval apparatus (durable approval rows,
  hash-binding, the approve-card) to Phase 2.
- **The dispatch tool is deny-by-default**: it can fire *only* the one allowed
  workflow in the one allowed repo. The App token's `actions: write` is repo-wide,
  so this allowlist is **security-critical** — it's what stops the bot from firing
  `reset-production-database.yml` or anything else.

**Why Phase 1 is safe without the approve-card:** four independent layers — a
developer must ask → the **fail-closed judge** clears the dispatch → the tool can
only fire the one allowed workflow → the run is bounded (pinned model, scoped tools,
capped turns) → and the output is just a **draft PR** that a developer merges
through `development`'s existing protection (required `test`+`security` checks,
code-owner review, conversation resolution). Nothing reaches `development` without
a human approving it in GitHub.

---

## Recovery & feedback model

Recovery is real, and happens in three layers — only one is a true "retry":

- **Layer 1 — in-run recovery (the real engine).** Inside the single `claude -p`
  run, the agent runs the tests itself, reads the failures, and iterates
  (edit → re-run pytest → edit) until green or it exhausts the turn budget. The
  failing-test output **is** the feedback; `--max-turns` is literally the recovery
  budget. **Nothing is committed unless it reaches green** — no broken PR. A
  **baseline** snapshot of the suite is captured first so a *pre-existing* failure
  isn't blamed on the fix.
- **Layer 2 — post-run gate (backstop, not a loop).** After the agent stops, the
  workflow independently re-runs the full suite + the red→green check. Green → PR.
  Not green → no PR.
- **Layer 3 — dispatch-level retry.** Two failure types, two policies:
  - **Transient / infra failure** (runner died, dep install hiccup, timeout):
    SamurAI **auto-retries once** (matches the one-shot-task retry culture in
    `scheduler.py`).
  - **Genuine can't-fix** (still red after the cap, or the agent bailed): **no
    blind re-dispatch** — a fresh run usually hits the same wall and burns tokens.
    SamurAI reports the failure detail and offers a **guided retry** (human supplies
    a hint; attempt #2 re-dispatches with the prior failure summary in the brief).

**Three terminal outcomes, each reported to Teams** (read via the run details
SamurAI already polls, plus `AUTOFIX_RESULT.md` on the branch for the failure
narrative):

- ✅ *"PR #456 opened to `development` (draft) — diff + tests passing. Your review."*
- ⚠️ *"Attempted #123, couldn't get tests green. What it tried / why it stopped: …
  Reply with guidance to retry, or I'll leave it."*
- 🔁 *"The fix run errored (infra). Retried once — [succeeded / still failing, log]."*

**Critical refinement from the research (best practice #2): a hard cap is not
enough — add futility detection.** Failed attempts burn ~4–5x the compute of
successes because stuck agents loop until the budget cap (SWE-Effi, 2025). Kill the
run early when it's looping (same test failing repeatedly, or a no-progress diff
between iterations), rather than letting it grind to turn 25.

---

## Best practices baked in (from the research)

These are folded into the builds below; collected here with citations.

1. **Architecture is validated** — issue → tested fix → *draft* PR → human merges,
   with a structural no-merge gate, is exactly Copilot coding agent + Claude Code
   Action. ([copilot-blog], [copilot-docs], [cca])

2. **Recovery = hard cap + futility detection.** Failures cost ~4–5x successes via
   repetitive loops; cap retries *and* detect no-progress. ([SWE-Effi][swe-effi])

3. **Simpler pipeline tends to win.** The non-autonomous **Agentless**
   localize → sample-several-patches → validate(run tests + independent reproduction
   test) → re-rank pipeline beat *every* open-source agent on SWE-bench Lite (32%,
   $0.70/issue) and hit **50.8% on Verified with Claude 3.5 Sonnet**; same-model
   head-to-head it beat SWE-agent **48% vs 28%**. The "the model is what matters"
   claim was *refuted* — scaffold design matters as much as the model.
   ([Agentless FSE 2025][agentless], [Agentless repo][agentless-repo],
   [SWE-Effi][swe-effi], [arch-diversity]) — **so: MVP on the Claude Code Action
   (least build, itself proven); graduate to localize→sample→re-rank only if hit
   rate disappoints.**

4. **Localize precisely; do NOT dump context.** File-level localization is the
   dominant success factor (**15–17x**), but indiscriminate context expansion
   *degrades* repair via noise amplification.
   ([fault-localization-context study][floc]) *(medium confidence — one repair model
   tested)*

5. **Anti-gaming guards on the test gate — the real teeth.** Reward-hacking on tests
   is documented and *no* surveyed system had a measured anti-tampering guard, so we
   design our own: (a) the reproduction test must **fail on the pre-fix commit and
   pass after**; (b) **diff the test files** — block deletions/weakening (add-only);
   (c) run regression tests the agent **didn't author**.
   ([ImpossibleBench][impossiblebench], [Agentless][agentless])

6. **Scope tools tightly AND verify the allowlist binds.** Restrict `--allowedTools`
   to the minimum; but there are known bugs where it doesn't bind — **test in a dry
   run that it actually restricts.** ([cca-sec])

7. **Realistic expectations.** Even SOTA is ~40–50% on real bugs, and leaderboard
   numbers are inflated by weak tests / solution leakage — real-world is lower.
   **"No fix produced" is the expected outcome roughly half the time, not an
   error.** Measure the real metric — **PR merge/acceptance rate**, not "did it open
   a PR." ([Agentless][agentless], report caveats)

---

## Build sequence — Phase 1 (the baby step)

Smallest useful piece first; the scariest piece (the CI rail) is proven by hand
*before* the bot can fire it.

### Build 0 — Prerequisite (~15 min)
Grant **`actions: write`** on the SamurAI GitHub App installation (currently
**unverified** — the App key is runtime-only). Without it the dispatch 403s.

### Build 1 — Fix-plan skill (read-only; ships value on its own)
A prompt section + tool group keyed on "fix issue," reusing the existing
`_select_tool_groups` / `_select_prompt_sections` pattern. SamurAI localizes with
the tools it already has — `investigate`, `sync_repo`, `read_repo_file_range`,
`search_repo_code`, the engineering wiki — and emits a structured brief:
`{repo, issue, root_cause_files, approach, eligible, reason, caps}`.
- **Baked in (#4):** localize to **precise files/locations**; keep surrounding
  context tight — do *not* paste large code chunks.
- **Baked in (compliance):** the brief is field-allowlisted — repo source + issue
  number/title + scrubbed issue body only.
- Eligibility rule baked from the `CLAUDE.md` good/bad lists: allow clear backend
  bugs; **deny** frontend/Vue/CSS, multi-tenant authz, Alembic-on-prod,
  payments/compliance/PII, perf.
- **Done when:** on a real issue it names the right files (verified by reading
  them) and correctly allows a backend bug / denies a frontend one. Nothing is
  written or dispatched.

### Build 2 — `controlled-fix.yml` engine in `virtualdojo` (human-triggered first)
A new workflow, `on: workflow_dispatch` with typed inputs (`issue_number`, `brief`,
`max_files`, `max_diff_lines`) — copy the typed-inputs shape from the repo's
existing `reset-production-database.yml`. Built and proven by hand-dispatch *before*
SamurAI can fire it.
- Branch `bugfix/issue-N` off `development`; run `claude -p` with **decisions made**:
  model **pinned** to `claude-sonnet-4-6`, tools scoped to
  `Edit,Write,Read,Glob,Grep,Bash(git:*),Bash(python -m pytest:*)`, `--max-turns 25`.
- **Baked in (#2):** **futility detection** — abort early on repeated identical test
  failures / no-progress diff, rather than grinding to the cap.
- **Baked in (#5):** capture a **baseline** suite run; the agent's reproduction test
  must **fail on the pre-fix commit and pass after**; **diff the test files** and
  fail the run on test deletion/weakening; run regression tests the agent didn't
  author.
- Gates lifted from `claude-pr-review.yml`: path scope-guard, added-lines
  secret-scan, full pytest. Plus a **diff-size cap** (files/lines → `exit 1`).
- **Open a draft PR** to `development` **only if green**; else write
  `AUTOFIX_RESULT.md` (what it tried / why it stopped) and **no PR**. Distinct exit
  code for infra errors so Layer-3 knows to auto-retry vs. report.
- **No merge step.** Gated by a `CONTROLLED_FIX_ENABLED` kill-switch repo var.
- **Baked in (#6):** a dry run must confirm the `--allowedTools` allowlist actually
  restricts.
- **Done when:** a manual dispatch opens a green draft PR; an oversized change is
  rejected by the diff-size cap; a frontend-only change is rejected by scope-guard;
  a run that deletes a test fails closed; the kill switch blocks the run.

### Build 3 — `github_dispatch_workflow` tool (the one new write)
One `@tool` in `tools/github.py` reusing the existing App-token plumbing
(`repo.get_workflow(...).create_dispatch(ref, inputs)`).
- **Deny-by-default repo + workflow-name allowlist** as the first thing in the tool
  body (security-critical — see Safety model).
- Added to `judge.py` `WRITE_TOOL_NAMES` so the **fail-closed** judge gates every
  dispatch the agent attempts.
- Fired when a developer asks SamurAI to attempt the fix; passes the brief + bounded
  params as typed inputs, `ref=development`.
- **Done when:** an asked-for fix dispatches; an injected/unauthorized request is
  blocked by the judge; a dispatch to any other workflow/repo is refused **even with
  the judge forced off** (proving the allowlist, not the judge, is the hard gate).

### Build 4 — Close the loop
After dispatch, poll the run (`github_list_workflow_runs` /
`github_get_workflow_run_details`); on completion fetch the diff
(`github_get_commit_diff`) and post the appropriate one of the **three terminal
outcomes** above to Teams. Developers review and merge through `development`'s
protection.
- **Baked in (#7):** log the outcome so we can compute the real metric — **PR
  merge/acceptance rate** over time.
- **Done when:** end-to-end on a real eligible issue: ask → plan → bounded run →
  green draft PR opened (not merged) → link posted; a no-fix run reports gracefully;
  an infra error auto-retries once.

---

## Phase 2 — deferred (turn "developer asks" into "SamurAI proposes")

Add this when you want SamurAI to proactively flag candidates nobody asked about, or
to require an explicit pre-codegen approve click:

- A durable **`pending_fixes`** table in `task_store.py` (the tasks table has no
  payload column).
- A **fix-plan Adaptive Card** (modeled on the social-preview card) with
  Approve/Reject, and `fix_*` handlers in `cards/actions.py`.
- **Clicker re-verification**: the approve handler re-resolves who clicked via
  `TeamsInfo.get_member` and checks a hardcoded `{devin, cyrus}` set — closing the
  spoofing hole in the current social flow (which trusts an LLM-stamped email).
- **Brief-hash binding**: approval is bound to a hash of
  `(repo, issue, brief, caps, ref)`, and the dispatch tool refuses if the dispatched
  inputs don't match — so an approved small fix can't be mutated into a large one.

This layers on top of Phase 1 with no rework: Phase 1's brief and dispatch tool are
exactly what it gates.

### Higher-ceiling option (only if hit rate disappoints)
Replace the autonomous `claude -p` run with the **Agentless procedural pipeline**:
hierarchical localize → **sample multiple candidate patches** → validate each
(regression tests + independent reproduction test) → **re-rank by test results** →
submit the best. The research shows this beats autonomous agents on accuracy *and*
cost — but it's more orchestration to build, so defer until the simple MVP's
measured merge rate justifies it. ([agentless], [agentless-repo])

---

## Realistic expectations & metrics

- Plan for **"no fix produced" ~half the time** — a clean failure report is the
  pipeline working, not failing.
- The metric that proves it works is **PR merge/acceptance rate**, logged from day
  one — not "did it open a PR."
- Benchmark resolve rates (e.g. Agentless 50.8% Verified) are **point-in-time and
  inflated**; treat them as "simple pipelines reach competitive rates," not as a
  promise. Re-measure cost/latency on our own Claude pipeline rather than assuming
  the 2024 `$0.34–$0.70/issue` figures.

---

## Open items / prerequisites

- [ ] **`actions: write`** on the GitHub App installation — confirm/grant (Build 0).
- [ ] **Judge fail-closed** change landed (the MVP's safety leans on it).
- [ ] Confirm the `--allowedTools` allowlist binds in a dry run (known upstream bugs).
- [ ] Anti-test-gaming guards (#5) are the least-precedented piece — **validate
      empirically** (e.g. deliberately try to make a run pass by weakening a test and
      confirm the gate catches it).
- [ ] Decide the fixer caps to freeze: pinned model id, exact `--allowedTools`,
      `--max-turns`, `max_files` / `max_diff_lines`.

---

## References

Production / reference systems:
- [copilot-blog]: GitHub Copilot coding agent — https://github.blog/news-insights/product-news/github-copilot-meet-the-new-coding-agent/
- [copilot-docs]: About Copilot coding agent — https://docs.github.com/copilot/concepts/agents/coding-agent/about-coding-agent
- Copilot risks & mitigations — https://docs.github.com/en/copilot/concepts/agents/cloud-agent/risks-and-mitigations
- Copilot build guardrails — https://docs.github.com/en/copilot/tutorials/cloud-agent/build-guardrails
- [cca]: Anthropic Claude Code Action — https://github.com/anthropics/claude-code-action
- [cca-sec]: Claude Code Action security docs — https://github.com/anthropics/claude-code-action/blob/main/docs/security.md

Program-repair research:
- [agentless]: Agentless (FSE 2025, doi:10.1145/3715754) — https://arxiv.org/abs/2407.01489
- [agentless-repo]: Agentless repo — https://github.com/OpenAutoCoder/Agentless
- AutoCodeRover — https://github.com/AutoCodeRoverSG/auto-code-rover · https://arxiv.org/abs/2404.05427
- [arch-diversity]: SWE-bench architecture analysis — https://arxiv.org/html/2506.17208v2
- [swe-effi]: SWE-Effi (futility / 4–5x failure cost; same-model head-to-head) — https://arxiv.org/html/2509.09853v2
- [floc]: On the Role of Fault Localization Context for LLM-Based Program Repair — https://arxiv.org/pdf/2604.05481
- [impossiblebench]: ImpossibleBench — measuring reward hacking in LLM coding — https://www.lesswrong.com/posts/qJYMbrabcQqCZ7iqm/impossiblebench-measuring-reward-hacking-in-llm-coding-1
- Aider repo map (localization) — https://aider.chat/docs/repomap.html

---

## Appendix — codebase anchors

- **Localization (reuse):** `tools/investigate.py`; `tools/repo_sync.py`
  (`sync_repo`, `read_repo_file_range`, `search_repo_code`, `list_repo_files`);
  `wiki.py` engineering scope + `search_wiki`/`read_knowledge`.
- **Agent wiring:** `agent.py` `_select_tool_groups` / `_select_prompt_sections`;
  autonomy rules; recursion limit 75.
- **Safety gate:** `judge.py` `WRITE_TOOL_NAMES` (add `github_dispatch_workflow`);
  fail-closed change.
- **GitHub plumbing:** `tools/github.py` `_github()` / `_github_token()`
  (App installation token); `github_list_workflow_runs` /
  `github_get_workflow_run_details`; `github_get_commit_diff`.
- **Sibling repo `virtualdojo-inc/virtualdojo`:** `autofix-bug.yml` (label-only
  today, the uncontrolled baseline); `reset-production-database.yml` /
  `alloydb-snapshot-reaper.yml` (the `workflow_dispatch` + typed-inputs shape to
  copy); `development` branch protection (required `test`+`security` checks,
  code-owner review, dismiss-stale, conversation-resolution) — the terminal human
  gate.
- **Reusable CI rails (SamurAI repo):** `.github/workflows/claude-pr-review.yml`
  (scope-guard, secret-scan, pytest gates); `.github/workflows/deploy.yml`
  (blue/green + `/health` + auto-rollback — the merged PR still deploys through it).
- **Phase 2:** `task_store.py` (`pending_fixes` table); `cards/` +
  `cards/actions.py` (fix-plan card + handlers); `app.py` `TeamsInfo.get_member`
  (clicker re-verification).
