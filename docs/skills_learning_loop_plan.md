# Autonomous Skill-Induction Loop for SamurAI — Recommended Plan

**Status:** Proposal for review · **NOT** for implementation · **Owner:** Devin · **Updated:** 2026-06-21
**Supersedes** the earlier human-gated / eval-tuned draft (rejected: Devin wants the AI to author
skills autonomously from tool-call patterns, with no training loop; humans curate by prompt).
Verified against the codebase at `feature/controlled-issue-fixer`.

## Thesis (the loop)

**OBSERVE** the team's tool-call patterns from the per-turn conversation log → **INDUCE** a skill in a
background, in-boundary job that freezes a recurring, verified-successful multi-tool sequence into a
`support/skills/<name>.md` file → **ADMIT** it only after a cheap verification gate (the existing
objective `label_turn == "good"` signal + frequency ≥ N distinct turns — never an LLM self-critique) →
**GOVERN** it by observable usage signals (refine-on-growth, hash-dedup, retire-on-low-utility / 👎) →
humans **CURATE** reactively by prompt only (`save_skill`/`delete_skill`). No human authoring gate, no
approve-card, no training/eval loop.

Authoring runs **in-process on `samurai-bot` via in-boundary regional Vertex Gemini** — the retired
`skills/**` auto-write path was retired only because a GitHub *runner* is out-of-boundary; in-process
in-boundary authoring does not cross the boundary. Honest framing throughout: the in-boundary model
(`gemini-2.5-flash-lite`, `kb/gemini.py:27`) is a **weak grounding-critic**, so the design constrains it
to *describe the observed tool order* (invent no product facts) and leans on cheap append-only writes
backstopped by **reactive** human delete-by-prompt and 👎-driven auto-retire.

---

## Phase 0 — Plumb the induction corpus (prerequisite, tiny)

**Goal.** Make the log carry which skill `get_skill` loaded, and keep raw tool-result content out of the
inducer's LLM payload — closing two red-team channels at the source.

- **Verified carrier.** `agent.py:1516` appends `f"{msg.name}: {status} -> {content_preview[:150]}"`.
  That `[:150]` tail carries both stale specifics and laundered attacker text. `selftune.evalset.parse_tools`
  (`evalset.py:96`, regex `^([A-Za-z0-9_]+):\s+(ok|error)\b`) already discards the `-> …` tail — so the
  inducer is safe **as long as it never passes the tail to the LLM**. No hot-path change needed.
- **`get_skill` telemetry (for Phase 3).** The skill *name* arg is known when the tool_call is seen
  (`agent.py:~1448`) but dropped by the ToolMessage log at 1516. Emit `get_skill[<skill-name>]: ok -> …`
  instead of `get_skill: ok`. One-token change; `parse_tools` won't match the `[..]` suffix, so add a
  tolerant secondary parse in the new governance module — **do not** change `parse_tools` (the self-tune
  gate depends on it).

**Guardrail (red-team #2, prompt-injection laundering — HIGH).** Injection reaches a skill body only if the
authoring LLM sees attacker content. Feeding the LLM **only `(name, outcome)`** — never `user_message`,
never the `-> content` tail — closes that channel while fingerprinting stays intact.

**Done when.** `get_skill[<name>]` appears in new turn records; a test asserts the inducer payload contains
no `user_message` substring and no `->` tail.

---

## Phase 1 — Induction: in-boundary background distillation (`skills_induce.py`, NEW)

**Goal.** Mine recurring, verified-successful multi-tool sequences and freeze each into one
`support/skills/<name>.md` — the prefix `skills.py` already loads and `save_skill` already writes, so the
**serving path needs zero changes**.

`skills_induce.run_induction(force=False)`:
1. **Gate** — return early unless `SKILL_INDUCE_ENABLED` (mirror `kb.run.pipeline_enabled`); `force=True`
   (human trigger) bypasses the kill switch only.
2. **Single-flight** — `storage.acquire_lock("support/skills/.induce.lock", ttl=KB_LOCK_TTL)`; `release_lock`
   in `finally`.
3. **Read turns** — `evalset.read_raw_turns(days=SKILL_INDUCE_DAYS)` (default 30), in-boundary on `/data`.
4. **Verification gate (Voyager admit)** — keep only turns where `evalset.label_turn(t) == "good"` (all
   task-tools `ok`, no give-up, not 👎, not background; a 👍 trumps heuristics). Reuses the *objective*
   success signal, not an LLM self-critique.
5. **Build each procedure** — from `parse_tools` take `ok` names in order, drop `NOISE_TOOLS` + `META_TOOLS`
   (incl. `get_skill`); fingerprint = ordered tuple; **skip if len < 2**.
6. **Aggregate frequency** — group by fingerprint; dedup contributors by `(conversation_id,
   user_message.lower())` so one thread can't inflate; record distinct-turn count + distinct user_ids.
7. **Select inducible** — distinct-turn count ≥ `SKILL_INDUCE_MIN_FREQ` (default 3); `seq_hash = sha256(...)`.
8. **Dedup vs catalog + manifest** — `support/skills/.induce_manifest.json` (`seq_hash → {name, last_freq,
   human_touched}`) + `skills.load_skill_catalog()`. Skip seen seq_hashes (unless frequency grew → refine),
   repo-skill names (immutable to the loop), and `human_touched` names.
9. **Bound batch** — ≤ `SKILL_INDUCE_MAX_PER_RUN` (default 3) per tick; converges; a kill costs ≤ one batch.
10. **Author** — Phase 2.
11. **Compose + validate** — `skill_authoring._compose_skill_md` + body banner (Phase 2) + `provenance:
    induced` / `induced_from: <seq_hash>` frontmatter; reject if `skills._parse_skill_text` is None, name
    mismatches/invalid, or collides with a repo skill.
12. **Write + read-back** — `storage.write_text` then `read_text` + re-parse to confirm (write-then-verify,
    as `save_skill` does); on failure log+skip.
13. **Checkpoint** — update the manifest after EACH skill so interruption resumes with no re-author.
14. **Refresh + report** — `skills.load_skill_catalog(force=True)`; emit content-free
    `[skills.induce] candidates=.. authored=.. refined=.. skipped_dup=..` (no PII).

**Reuses.** `evalset`: `read_raw_turns`, `parse_tools`, `label_turn`, `NOISE_TOOLS`, `META_TOOLS`.
`kb/storage.py`: lock + read/write. `kb/run.py`: kill-switch + bounded-batch + finally-release.
`kb/compile.py`: `_json_from`/`_llm_text`, manifest checkpoint, "historical work-log, not a fact" framing.
`skills.py`: `load_skill_catalog(force=True)`, `_parse_skill_text`, name rules, `SKILLS_BUCKET_PREFIX`.
`tools/skill_authoring.py`: `_compose_skill_md`, `_valid_name`, write-then-read-back.

**Done when.** A fixture `/data/raw` with a sequence in ≥3 good turns → `run_induction` (fake storage + fake
LLM) writes one valid skill + manifest entry; a second run is a no-op (dedup); lock-held and kill-switch-off
short-circuit; malformed LLM output is rejected, not written.

---

## Phase 2 — Admission authoring with the weak model (constrained)

**Goal.** Turn an admitted fingerprint into name+description+body **without inventing product facts** — the
canonical autonomous-write failure (ChatGPT "Dreaming").

**Mechanism.** Call `kb.gemini.get_kb_llm()` (regional us-central1, `temperature=0`, no Anthropic client) via
`_llm_text`/`_json_from`. A strict `_INDUCE_SYS` prompt (mirroring `compile.py`'s "HISTORICAL record, NOT
current behavior" framing): *describe the OBSERVED tool ORDER and WHEN it applies; invent NO product facts,
field names, IDs, endpoints, or UI steps; output ONLY JSON `{name, description, body}`.* Payload = **tool-NAME
sequence only** (no `user_message`, no tail — Phase 0).

**Guardrail (red-team #1, staleness-laundering — HIGH).** `_parse_skill_text` keeps only
name/description/body and **drops all other frontmatter** (verified), and `get_skill` returns the body
verbatim — so a `last_verified` frontmatter would die before reaching the agent. The staleness warning must
live in the **body**: prepend a fixed banner before composing —
> *Induced from observed tool-call patterns on {date}; this is a HISTORICAL procedure, not verified current
> product behavior. Re-verify any API/cmdlet/field/UI step against live state or docs before asserting it.*

This reuses `compile.py`'s framing (the mitigation CLAUDE.md already endorses) and rides through `get_skill`.

**Guardrail (red-team #2 defense-in-depth).** Refuse to write an induced body containing imperative
directives toward `evalset.WRITE_TOOLS` (e.g. "always call `github_create_issue`", "`send_teams_message` to
all admins"). An induced skill may *describe* a sequence; it must never *instruct* the agent to fire a write.

**Correction (do NOT bend `judge.py`).** `judge.py` is a LangGraph node keyed on a live `AIMessage.tool_calls`
(`should_judge_writes` / `judge_writes_node`), taking `(user_messages, tool_call)` — **not** a callable
`judge(body) → verdict`. Reusing it for static body validation would require faking a tool_call or refactoring
the stages; that is not minimum-viable. The WRITE_TOOLS-directive regex above replaces it.
(`save_skill`/`delete_skill` stay in `WRITE_TOOL_NAMES` for the *human* path.)

**Done when.** `_INDUCE_SYS` is pinned; a test asserts the written body starts with the banner, contains no
WRITE_TOOLS imperative, and round-trips through `_parse_skill_text`.

---

## Phase 3 — Governance by observable signals (`kb/skill_governance.py`, NEW)

**Goal.** Refine, dedup, retire induced skills from usage+outcome telemetry only — no eval loop, no training;
a background consolidator (Letta/LangMem/ChatGPT pattern), never hot-path self-critique.

**Mechanism.** Mirror `kb/compile.py`: lease lock, JSON manifest, bounded batch, `KB_SKILLS_GOV_ENABLED` kill
switch, scheduled via `asyncio.to_thread`. Read recent `/data/raw` with `read_raw_turns`; reuse `label_turn`
as the outcome; join to each turn's `get_skill[<name>]` uses (Phase 0) → per-skill
`uses/wins/losses/👍/👎/last_used` in `support/skills/.telemetry.json` (sidecar — `.json` is never served:
loader skips non-`.md`).
- **Refine** — an induced skill whose frequency keeps growing gets its description regenerated (Phase 2 path).
  Append-only; never auto-deletes another author's edit.
- **Dedup** — observable, not semantic: intent-hash (`sha256` of normalized name+description) + body-hash flag
  near-duplicates; keep the higher-utility one, `storage.delete` the other. (Red-team #4: dedup on the
  **set** of tool names, not the ordered seq_hash, so reordered duplicates collapse.)
- **Retire** — zero `get_skill` uses over N days, OR ≥K repeated 👎 on turns that loaded it → auto-delete from
  `support/skills/` (**bucket skills only**; repo skills immutable). This is the reactive control that
  compensates for the heuristic admission gate.

**Reuses.** `kb/compile.py` scaffold; `kb/storage.py` lock/read/write/delete; `evalset.label_turn`;
`memory.py` background-extractor precedent.

**Guardrail (append-only growth / prompt bloat — RAG-MCP).** Every skill's name+description is injected at
`agent.py:819`. `SKILL_INDUCE_MAX_PER_RUN` + retire-on-zero-usage keep the catalog bounded; growth past a
threshold is the trigger to add retrieval (deferred, out of scope).

**Done when.** A fixture with `get_skill[foo]` on 👎 turns retires `foo`; a duplicate pair collapses to the
higher-utility one; telemetry persists to the sidecar and is never returned by `load_skill_catalog`.

---

## Phase 4 — Scheduling + reactive human curation

- `scheduler.py` — in `init_scheduler`, if `SKILL_INDUCE_ENABLED`: `add_job(_run_skill_induction,
  CronTrigger.from_crontab(os.environ.get("SKILL_INDUCE_CRON","30 8 * * *")), id="skill_induction")`;
  `_run_skill_induction` = `await asyncio.to_thread(run_induction)` in try/except (never crash the scheduler),
  mirroring `_run_kb_pipeline`. Same for the governance tick (`KB_SKILLS_GOV_ENABLED` + its cron).
- `tools/self_improve.py` — add `trigger_skill_induction(reason)` (daemon thread → `run_induction(force=True)`),
  modeled on `trigger_wiki_compile`; Devin/Cyrus gated; append to `SELF_IMPROVE_TOOLS`.
- **Human curation stays prompt-only** — `save_skill`/`delete_skill` unchanged ("change skill X to …" /
  "delete skill X"). A `save_skill` write marks the name `human_touched` in the manifest so refinement never
  clobbers a human edit.

**Done when.** `SKILL_INDUCE_ENABLED=on` registers the job; the force-trigger runs in-boundary; a `save_skill`
on an induced name flips `human_touched` and the next tick skips it.

---

## Top risks & guardrails

| # | Risk | Sev | Guardrail (minimum-viable, reuses existing) | Residual |
|---|------|-----|---------------------------------------------|----------|
| 1 | **Staleness-laundering** — once-correct procedure canonized as current; frontmatter dropped; body served verbatim; CoVe off-by-default | HIGH | Dated **body banner** ("HISTORICAL… re-verify"); `_INDUCE_SYS` forbids product specifics; payload = tool names only | Soft — relies on the same weak model to re-verify; no hard expiry beyond the banner |
| 2 | **Prompt-injection laundering** — attacker text via `content_preview[:150]` into the body; no human/judge on the auto path | HIGH | LLM sees **only `(name, outcome)`**; refuse bodies with imperative WRITE_TOOLS directives | The tool-NAME sequence itself can encode a deprecated workflow; judge-node reuse rejected as non-minimal |
| 3 | **Verifier single-point-of-trust** — `label_turn`-good is heuristic; tool-ok-but-wrong, popular procedure qualifies | Med | Freq ≥ 3 distinct turns + dedup by `(conv,msg)`; weight 👍; mark `provenance: induced` | Frequency ≠ correctness — compensated reactively (below) |
| 4 | **Weak model invents facts** (`gemini-2.5-flash-lite`) | Med | `temperature=0`, strict `_INDUCE_SYS`, round-trip validation, CoVe still guards serve-time claims | A confident model can over-describe |
| 5 | **Append-only growth / prompt bloat** | Med | `SKILL_INDUCE_MAX_PER_RUN` cap + retire-on-zero-usage | Past a threshold → add retrieval (deferred) |
| 6 | **Name collision / clobber a human edit** | Med | Skip repo names; `human_touched` flag; bucket-overrides-repo is existing tested semantics | None significant |
| 7 | **PII in the LLM payload** | Med | In-boundary Vertex only; payload = tool names + outcomes; body must not embed raw messages | In-boundary by construction |
| 8 | **Interruption mid-author during a Cloud Run drain** | Low | Per-skill manifest checkpoint + single-flight lock → resume, no re-author | ≤ one small batch |

**Reactive controls that ARE the real safety net** (stated plainly because the admission gate is weak):
**(a) 👎 auto-retire** on turns that loaded a skill; **(b) human delete-by-prompt** ("delete skill X"); **(c)
zero-usage retire**. Because `gemini-2.5-flash-lite` cannot reliably self-critique grounding, **these signals —
not the LLM — enforce correctness after the fact.** The admission gate's only job is to keep the *write* cheap
and rare.

---

## What we are NOT building
- **No human-authoring gate / approve-card** for creation or refinement — autonomous by design.
- **No training/eval loop for skills** — we do NOT reuse the selftune propose→score→promote gate. Admission =
  `label_turn`-good + frequency; governance = observable signals.
- **No reuse of the retired GitHub-Actions path** — authoring is in-process on `samurai-bot`.
- **No external LLM on bucket/log data, ever** — only `kb.gemini.get_kb_llm` (regional Vertex).
- **No serve-time fresh-verification gate** — the staleness banner is soft; a hard re-verify gate is the
  explicit residual on risk #1 (out of scope for MVP).
- **No semantic dedup / embedding store** — hash-based dedup only.
- **No retrieval layer yet** — full catalog still injected; retrieval is the deferred response if it outgrows
  the prompt.
- **No change to `parse_tools` or the self-tune gate** — `get_skill[<name>]` parsing lives in the new module.
- **No bending `judge.py` into a body-validator** — the WRITE_TOOLS-directive regex replaces it.

## New / touched files
- **NEW** `skills_induce.py` — mine → fingerprint → freq+dedup → in-boundary author → validate+write+checkpoint.
  Manifest `support/skills/.induce_manifest.json`, lock `support/skills/.induce.lock`.
- **NEW** `kb/skill_governance.py` — telemetry, refine, hash-dedup, retire. Sidecar `support/skills/.telemetry.json`.
- `agent.py` — emit `get_skill[<name>]` in the tool_result log (Phase 0); no other hot-path change.
- `scheduler.py` — two `add_job`s + `_run_skill_induction` / `_run_skill_governance` `to_thread` wrappers.
- `tools/self_improve.py` — add `trigger_skill_induction`; append to `SELF_IMPROVE_TOOLS`.
- `tests/test_skills.py` (+ a new test module) — fingerprint/freq, `label_turn`-good gating, dedup, banner,
  WRITE_TOOLS-directive rejection, malformed rejection, lock/kill-switch, `human_touched` skip, retire/dedup —
  all with fake storage + fake LLM.

## Env vars (default-off / bounded)
`SKILL_INDUCE_ENABLED` (off) · `SKILL_INDUCE_CRON` (`30 8 * * *`) · `SKILL_INDUCE_DAYS` (30) ·
`SKILL_INDUCE_MIN_FREQ` (3) · `SKILL_INDUCE_MAX_PER_RUN` (3) · `KB_SKILLS_GOV_ENABLED` (off) + its cron ·
reuses `KB_LOCK_TTL`, `SKILLS_BUCKET_ENABLED` (must be on for induced skills to *serve*), `KB_COMPILE_MODEL`.

## Rollout
1. Land Phase 0 (`get_skill[<name>]` logging) + tests; deploy; let telemetry accumulate.
2. Land `skills_induce.py` with `SKILL_INDUCE_ENABLED=off`; tests green; deploy dormant.
3. Force-trigger once in a quiet window; inspect the written `support/skills/*.md` + manifest by hand (banner,
   no WRITE_TOOLS directives, valid round-trip).
4. Flip `SKILL_INDUCE_ENABLED=on` with a daily cron; watch `[skills.induce]` stats.
5. Land governance (`KB_SKILLS_GOV_ENABLED`) after ~a week of induction data so retire/dedup has signal.

## Research basis (high-confidence, sourced)
- **Voyager** (arXiv:2305.16291) — admit a skill only after success verification; freeze the observed sequence;
  cheap model writes the description. → Phases 1–2 (verification = `label_turn`-good).
- **Agent Workflow Memory** (arXiv:2409.07429) / **ExpeL** (arXiv:2308.10144) — induce the repetitive action
  subset across tasks, online + supervision-free. → Phase 1.
- **Letta/MemGPT, LangMem, ChatGPT memory** — separate **background** consolidator, never hot-path self-critique;
  ChatGPT "Dreaming" over-trust is the autonomous-write failure to avoid. → Phases 2–3 + the banner guardrail.
- **RAG-MCP** (arXiv:2505.03275) / SkillOps — govern by observable signals; retrieve-don't-dump as the catalog
  grows. → Phase 3 + deferred-retrieval note.
- **Reflexion / Self-Refine** (2303.11366 / 2303.17651) — a weak base model is a poor self-critic of grounding;
  bound the loop. → why correctness is enforced reactively, not by the LLM.
- **Cursor / Devin playbooks** — description-matched lazy selection + prompt-invoked rules → confirms current
  recall design + human curate-by-prompt.
