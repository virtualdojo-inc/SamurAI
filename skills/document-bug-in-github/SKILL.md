---
name: document-bug-in-github
description: Turn a VERIFIED bug reproduction into a properly documented GitHub issue on virtualdojo-inc/virtualdojo — grounded root cause, the sandbox reproduction as evidence, code-vs-data classification, and the repo's issue conventions. Draft-and-confirm — never file or edit an issue without explicit human (Devin/Cyrus) approval. Use after reproduce-bug-in-sandbox produces a verdict.
---

# Document a reproduced bug in GitHub (draft-and-confirm)

You convert a reproduction into a clear, evidence-grounded GitHub issue. You
**draft** it and present it for approval — you do **not** create or edit any issue
until a human (Devin or Cyrus) says to. "Look into it" / "reproduce it" is NOT
"file it."

**Every claim in the issue must cite evidence you have right now:** a Cloud Logging
query + entry (with timestamp + project), a code location as `file:line @ <sha>`,
and — for a code bug — a sandbox `code_run` id from `reproduce-bug-in-sandbox`. No
"by default" / "should be" / unverified line numbers. If you can't ground it, write
"unverified" — never assert it.

## Step 1 — Require a real verdict first
Do not write a bug issue from a symptom alone. You must have run
`reproduce-bug-in-sandbox` and have:
- a reproduction (`code_run` id, exact expected-vs-actual), and
- its classification: **CODE** (real product function inlined at the deployed SHA
  and still misbehaves) vs **DATA** (a None/missing/malformed field) vs
  **behavior-only** (library reproduced, product handling not verified).

If it's DATA or behavior-only, say so plainly — do **not** file it as a code bug.
A DATA issue may still warrant an issue (e.g. "add coercion/guard"), but framed as
a hardening request, not a crash.

**"Handled gracefully" is not P0/P1.** If the reproduction shows the product
already **catches** the condition (try/except → logs a WARNING → returns a
default/None/blank) rather than crashing or returning a wrong number, then:
- it is **not** a crash and almost never P0/P1 — severity is driven by user impact
  (a silently-blank field is typically P2/P3), not by the scary-looking log line;
- whether the *default* behavior should change (e.g. "numeric formulas should treat
  blank as 0") is a **product/UX decision**, so frame it as a
  **behavior/enhancement** request, not "BUG: crash", and set the type/severity to
  match. Do not inflate a logged-and-handled warning into a P1 bug.

## Step 2 — Dedup before drafting
`github_search_issues` on virtualdojo-inc/virtualdojo for the error string / symptom
/ file. If an open issue covers it, propose commenting on that one (with the new
evidence) instead of a duplicate.

## Step 3 — Draft the issue body (grounded template)
```
### Symptom
<what the user/system sees; 1–2 lines>

### Evidence (observed)
- Log: `<exact error string>` — <project>, <service>, <timestamp> (via query_cloud_logs)
- Frequency/scope: <how often / which tenants or records, if known>

### Reproduction
- Reproduced in the run_code sandbox: code_run `<id>`
- Inlined: `<file>:<start>-<end> @ <sha>` (the real product function, verbatim)  ← required for a CODE bug
- Input (scenario, no CUI values): <shape only>
- Expected vs actual: `<expected>` vs `<actual>`

### Root cause  ← only if CODE-verified; else write "DATA issue — see classification"
- `<file>:<line> @ <sha>` — <the specific mechanism, grounded in the read source>

### Classification
CODE bug | DATA issue | behavior-only (product handling unverified) — <one line why>

### Suggested fix
<concrete, tied to the file:line; note if it's an autofix candidate per CLAUDE.md>

### Regression test
<the scrubbed pytest that should lock this — synthetic fixtures, never a raw CUI record>
```

## Step 4 — Apply the repo's conventions (only when filing, with approval)
- Issue **type**: native `Bug` (via `github_set_issue_type`) — not the deprecated
  Status=Bug option.
- **Priority**: P0–P3 as a plain label + the Project 2 Priority field; set Area/Effort
  fields where known (`github_update_item_field`, add to project with
  `github_add_item_to_project`).
- **autofix label**: only suggest it (never apply without approval), and only for a
  good candidate per CLAUDE.md (clear backend logic bug with an **unambiguous**,
  single correct fix — NOT NULL, wrong default, off-by-one, missing null check that
  causes a wrong result). **Do NOT suggest autofix when the "fix" is a
  product/UX/default-behavior decision** (e.g. "should blank coerce to 0?"), or for
  frontend, auth/multi-tenant, migrations, payments/PII, or perf. A condition the
  product already catches-and-logs is a design question, not an autofix bug — if you
  find yourself proposing a change to *default behavior*, that disqualifies autofix.

## Step 5 — Confirm, then act
Present the drafted body + proposed type/priority/labels and ask for approval. Only
after an explicit go-ahead from Devin/Cyrus: `github_create_issue` (or comment on the
existing one), then set type/priority/fields. Report the issue URL. If approval isn't
given, leave it as a draft — that is a complete, correct outcome.
