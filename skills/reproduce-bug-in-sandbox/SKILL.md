---
name: reproduce-bug-in-sandbox
description: Turn a real user bug into a deterministic, in-boundary reproduction using the run_code sandbox — read the actual product source at the DEPLOYED revision, drive it with the real record's inputs, and assert the exact expected-vs-actual. Use ONLY for pure, stdlib-only, self-contained logic (formula/pricing/parsing math). For anything importing pydantic/sqlalchemy/app modules or needing the DB/ORM, STOP and use a CI pytest instead. Produces a reproduced/not verdict with explicit confounds — never a fix.
---

# Reproduce a bug in the sandbox (repro only, never fix)

You take a real user scenario and try to reproduce it deterministically in the
`run_code` sandbox. You produce a **reproduced / not-reproduced verdict with its
confounds** — you do not fix anything, commit anything, or claim a root cause you
did not execute.

**Be honest about what the sandbox is.** The child runs `python -I -B -S`:
**stdlib only, no third-party packages, no network, no credentials, no DB, no repo
on disk.** It computes over the `inputs` global and returns text. It therefore
canNOT `import` product code — you must bring the logic in as text, and you can
only reproduce logic that a **clean stdlib-only seam** can express. Most bugs do
NOT fit here. When one doesn't, say so and route to CI (Step 2 gate).

**Grounding rule.** Every part of the repro must trace to something you fetched:
the source (`file:line` at a specific SHA), the record's data (a CRM/log read), and
the observed failure (a log line or the record's stored value). If you can't ground
it, say "I couldn't verify X" — do not invent it.

---

## Step 0 — Preconditions
- Confirm the sandbox is enabled (`SAMURAI_SANDBOX_ENABLED`). If it isn't, stop:
  "The sandbox isn't deployed yet — I can draft the repro but can't run it."
- You need: (a) the failing behavior (error string or wrong value), and (b) a
  concrete record/scenario that exhibits it.

## Step 1 — Locate the seam and the deployed revision
- Find the suspect function with `search_repo_code` / `read_repo_file_range`.
- **Pin the DEPLOYED SHA, not HEAD.** The bug lives in what prod is running. Get
  the deployed revision/image for the affected service (e.g. `virtualdojo-backend`
  in `virtualdojo-fedramp-prod`) and read the source **at that revision**. If you
  read HEAD/main, you may be testing code that isn't in prod — call that out.
- Record the file path, line range, and SHA. You will put them in the script
  header and the run description.

## Step 2 — GO/NO-GO gate (do this before writing anything)
The sandbox is stdlib-only. Inspect the seam's imports and dependencies:
- **NO-GO → use CI (a scrubbed pytest in `virtualdojo/tests/`), not the sandbox, if
  the seam (or anything it must call) imports** `pydantic`, `sqlalchemy`, `app.*`,
  or any third-party package; needs the ORM/session, DB rows, HTTP, or framework
  context; or the buggy value is produced by a relationship/DB traversal you cannot
  reconstruct from plain data.
- **GO only if** the buggy logic can be expressed with the **standard library**
  (`decimal`, `re`, `datetime`, `math`, `json`, etc.) **plus the curated deps
  vendored into the sandbox** (see `sandbox/vendor/`) — currently **`simpleeval`**,
  which is what the product's formula/pricing evaluator (`_safe_eval_formula`) runs
  on. So a formula-runtime repro that inlines the evaluator and imports `simpleeval`
  is GO; anything needing `sqlalchemy`/`pydantic`/`app.*`/the DB is still NO-GO.
  If unsure, treat it as NO-GO.
State the decision and why. Do not force a NO-GO case into the sandbox. If a repro
needs one more pure-Python dep, propose vendoring it (add a file to
`sandbox/vendor/`) rather than hand-porting the logic.

## Step 3 — Bring the source in VERBATIM (do not retype from memory)
- Copy the exact source from the `read_repo_file_range` output **byte-for-byte**
  into the script. Do **not** paraphrase, reformat, "clean up", or fix anything —
  **especially not the line that looks buggy.** Silently correcting the bug while
  inlining produces a false "not reproduced," which is the worst failure of this
  skill.
- Inline the **smallest faithful unit** that carries the behavior — less transcription
  is less room for drift.
- After composing, re-read the same range and confirm the inlined text matches. Note
  in the script header: `# inlined from <path>:<start>-<end> @ <sha> — verbatim`.

## Step 4 — Reproduce the real derivation, don't hand-fake it
- Build the input state the way prod builds it, as far as stdlib allows. Do **not**
  simply hard-set the suspected bad value (e.g. `contract_fee = None`) — that
  reproduces a *symptom* against a synthetic state and can pass/fail for the wrong
  reason. Derive it from the record's real fields.
- If you must stub a boundary, state **exactly** what you stubbed and treat every
  stub as a confound in Step 7. If the buggy value comes from something you had to
  stub away, the repro is invalid — go back to Step 2 (NO-GO).

## Step 5 — Data goes in `inputs`, never in the script
- The script text is persisted to `code_runs.script` **and embedded**. Real record
  data (CUI-adjacent) must therefore go **only** in `inputs` (only a hash of
  `inputs` is stored). Never paste customer field values into the script body.
- The script may contain product **source** (that gets persisted + embedded
  in-boundary, which is acceptable) — but **no data values**.

## Step 6 — Assert EXACT expected-vs-actual
- For a value bug: assert the exact expected value (use `decimal.Decimal` / rounding
  exactly as prod does — float drift causes false verdicts).
- For an error bug: assert the exact **exception type** and a **message substring**
  from the real log line — not merely "an error happened."
- Return structured JSON. Template:

```python
# inlined from app/services/<module>.py:<start>-<end> @ <sha> — verbatim, DO NOT EDIT
def <fn>(...):
    ...  # byte-for-byte from the repo

def run_repro(inputs):
    exp = inputs["expected"]
    try:
        got = {"value": <fn>(**inputs["args"])}
    except Exception as e:
        got = {"error_type": type(e).__name__, "error_msg": str(e)}
    if "error_type" in exp:
        reproduced = got.get("error_type") == exp["error_type"] and \
                     exp["error_msg"] in got.get("error_msg", "")
    else:
        reproduced = got.get("value") == exp["value"]
    return {"reproduced": reproduced, "expected": exp, "actual": got}

import json
print(json.dumps(run_repro(inputs), default=str))
```

Pass the real scenario as `inputs`, e.g.
`{"args": {...from the record...}, "expected": {"error_type": "TypeError", "error_msg": "unsupported operand type(s) for -: 'NoneType'"}}`.

## Step 7 — Report with confidence AND confounds
A sandbox result is **not** ground truth. Report:
- The verdict (reproduced / not), the SHA, and what was inlined vs stubbed.
- If **not reproduced**, enumerate the confounds — it does NOT prove "no bug":
  transcription drift, wrong SHA (read HEAD not deployed), wrong/hand-faked input
  state, over-stubbing, or the logic exceeding stdlib-only. Say which you can rule out.
- If **reproduced**, confirm it's for the *right reason* (the failure path matches the
  real log line), not an incidental error from your stubbing.
- Never upgrade a sandbox repro to "root cause confirmed in prod" — it confirms the
  behavior of the inlined logic at that SHA over that input, nothing more.

## When to escalate to CI (Path B)
Any NO-GO from Step 2, or any repro that needed heavy stubbing, becomes a **scrubbed
pytest** in `virtualdojo/tests/` running against the real code in CI — using
**synthetic/anonymized fixtures**, never a raw CUI record committed to git. That test
can then carry the `autofix` label (with the user's approval) to drive the fix.
