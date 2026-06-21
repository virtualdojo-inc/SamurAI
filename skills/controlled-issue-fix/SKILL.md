---
name: controlled-issue-fix
description: Produce a controlled, test-driven fix PLAN for a virtualdojo-inc/virtualdojo backend bug — localize the root cause in code, judge eligibility, and emit a structured fix brief for human review. Use when asked to fix, attempt a fix for, or plan a fix for a GitHub issue. SamurAI produces the plan only; it does not yet write code, trigger a fix run, or open a PR.
---

# Controlled Issue Fix (plan + brief)

This is the **controlled** path for fixing a bug — distinct from the legacy
`autofix` label, which is the uncontrolled, fire-and-forget one (see
`github-issue-triage`). Here SamurAI owns the work and is observable at every step.

**Scope:** `virtualdojo-inc/virtualdojo` only, **backend bugs only**. Decline
anything in the DENY list below.

**Current capability (be honest):** SamurAI produces and presents the fix
**brief** for human review. It does **not** yet write code, dispatch a fix run, or
open a PR — those are gated steps added in later builds. Never claim to have opened
a PR or fixed the bug; you are producing a plan.

## Step 1 — Localize the root cause (read-only)
1. `github_get_issue_details` for the issue (number, title, body, error trace).
2. Orient with the engineering wiki: `search_wiki` / `read_knowledge` on the
   system-map and symptom-to-subsystem index to find the owning subsystem and the
   code paths to read first.
3. Read the live code: `sync_repo` (branch `development`), then
   `read_repo_file_range` / `search_repo_code`, and dispatch `investigate` for
   parallel root-cause analysis. Confirm against the actual code — the wiki is an
   orientation map, not ground truth.
4. **Localize precisely.** Name the exact file(s) + function/line range. Do NOT
   paste large code blocks into the brief — over-broad context measurably *hurts*
   fix quality. Point to locations; don't dump them.

## Step 2 — Judge eligibility
**Good candidates (ALLOW):** backend data/logic bugs with a clear trace (NOT NULL,
type mismatch, missing default, wrong field/filter), unambiguous API endpoint bugs,
regex/parsing fixes, missing DB defaults/constraints, off-by-one / wrong status
code / missing null check, test-coverage gaps for existing behavior.

**Bad candidates (DENY — refuse with a one-line reason, do not produce a brief):**
frontend/UI (Vue/CSS/layout), multi-tenant authz / access control, Alembic
migrations on prod data, business-logic/UX decisions, performance (needs
profiling), anything touching payments, compliance, or PII.

## Step 3 — Emit the fix brief (the contract later steps consume)
Present a brief with exactly these fields:
- **repo**: `virtualdojo-inc/virtualdojo`
- **issue_number**
- **root_cause_files**: each as `path` + one line on why it's the cause (with the
  function / line range)
- **approach**: the minimal change, in 1–3 sentences
- **reproduction**: how the bug manifests / the failing regression test to write
- **eligible**: yes (+ why it's a good candidate) — only reach this for ALLOW cases
- **caps** (proposed bounds for the eventual run): `max_files`, `max_diff_lines`,
  `target_branch: development`

## Compliance — what may go in the brief
The brief is destined to cross into an out-of-boundary CI runner in a later build,
so build it **only** from: repo source code + the issue number/title + a scrubbed
issue body. **NEVER** put Cloud Logging output, CRM data, customer or PII content,
or knowledge-base content into the brief. If the root cause is only legible from
logs/CRM, say so and stop — do not copy that content into the brief.
