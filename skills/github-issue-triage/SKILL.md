---
name: github-issue-triage
description: Triage bugs and issues for virtualdojo-inc/virtualdojo — check for duplicates first, file with the correct native issue type, and judge whether a bug is a good `autofix` candidate. Use when handling a bug report, filing/closing a GitHub issue, or deciding whether to suggest the autofix label.
---

# GitHub Issue Triage

Org is `virtualdojo-inc`; main repo is `virtualdojo-inc/virtualdojo`. Never act
on the old `Quote-ly/*` names (GitHub redirects them, but use the canonical
names).

## Always do first: check for duplicates
- Search before filing: `github_search_issues` on the key error string / symptom.
- If a matching open issue exists, comment/link instead of creating a new one.

## Filing correctly
- Use the native issue **type** (Bug / Feature / Task). The old project-2
  `Status=Bug` option is deprecated — do not use it.
- Title = the symptom; body = repro steps, the exact error/trace, expected vs
  actual, and the source location if known.
- Filing is an action: confirm findings with the user conversationally first
  unless they explicitly said "file an issue". "Look into" / "investigate" is not
  "file it".

## Autofix candidacy (the `autofix` label triggers an automated TDD bug-fix)
Suggest the label only with user approval, and only for **good candidates**:
- Backend data/logic bugs with a clear trace (NOT NULL, type mismatch, missing
  default, wrong field/filter), unambiguous API endpoint bugs, regex/parsing
  fixes, missing DB defaults/constraints, off-by-one / wrong status code / missing
  null check, or test-coverage gaps for existing behavior.

**Bad candidates — do NOT suggest autofix:**
- Frontend/UI (Vue/CSS/layout, needs visual check), multi-tenant authz / access
  control, Alembic migrations on prod data, business-logic/UX decisions,
  performance (needs profiling), or anything touching payments, compliance, or
  PII.

## Closing
- Close duplicates/erroneous issues with a clear reason and a link to the
  canonical issue. Closing/merging PRs and applying `autofix` require explicit
  Devin/Cyrus approval.
