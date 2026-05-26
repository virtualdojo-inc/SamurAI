# Write-action judge: design

A two-stage LLM-as-judge node that intercepts every write-action tool call
before execution. Modeled directly on Anthropic Claude Code's Auto Mode
(launched 2026-03-24); adapted where SamurAI's threat model differs.

## What this solves

Looking at the last week of incidents:

- `smartsheet_update_row` called with `row_id=1146352141553540` — that's the
  sheet ID being passed as a row ID by accident.
- `smartsheet_update_row` called with `row_id=101` — the user-facing "Row
  ID" column display value, not the API `_row_id`.
- A docstring + fuzzy-match defense caught some of these, but neither
  catches the general class: **the model picks the wrong target ID for a
  mutating call.**

A pre-execution judge that looks at the call args against the user's
request catches this class systematically.

## What the judge sees (the prompt-injection boundary)

**Exactly two inputs per tool call:**

1. **All `HumanMessage` text in the conversation**, concatenated in order
   and separated by `---`. This is the user-intent anchor.
2. **The tool name and args** from the current `AIMessage.tool_calls[i]`.

**Nothing else.** Not the agent's text content alongside the tool call,
not earlier `AIMessage`s, not any `ToolMessage` results, not the agent's
system prompt, not memory retrieval output.

This matches Anthropic's design exactly:
> "Input: User messages + tool call payloads only (assistant text and
> tool results stripped). Including assistant messages and tool results
> caused a drop in performance."

The agent's text content sometimes contains its reasoning for the call.
We deliberately exclude it because: (a) a poisoned tool result could
have contaminated that reasoning earlier in the loop, and (b) Anthropic
measured that excluding it produces a lower false-negative rate.

## Two-stage classifier

| Stage | Model | Output | When it runs |
|---|---|---|---|
| 1 | `gemini-2.5-flash-lite` | Single token (`safe` or `review`) | Every write tool call |
| 2 | `gemini-3-flash-preview` | JSON `{verdict, reason}` | Only when Stage 1 returns `review` |

Anthropic uses Sonnet 4.6 for both stages. We start with Flash-Lite + Flash
(~100x cheaper at our volume) and use shadow-mode data to decide if we need
to escalate the model choice. Their reported FPR numbers (8.5% → 0.4%
after Stage 2) are our target; if we don't get close, the model choice is
the first knob.

Stage 1 is tuned to err toward `review` — false positives in Stage 1 are
fine, they just cause an extra Stage 2 call. False negatives in Stage 1
are bad (they ship without review).

Stage 2 returns three possible verdicts:
- `approve` — let the call through
- `block` — synthetic tool-failure message back to the agent, agent revises
- `pass` — ambiguous, let it through with a logged note (rare)

## Policy: two tiers (not three)

```python
READ_ONLY_TOOL_NAMES = frozenset({...})  # never reach the judge
WRITE_TOOL_NAMES     = frozenset({...})  # always reach the judge
```

Anthropic has three tiers. Their Tier 2 ("in-project file writes skip
the classifier because git provides review") has no SamurAI equivalent
— our writes hit GitHub, Smartsheet, Teams, CRM with no rollback. So
the middle tier collapses on purpose.

An import-time sanity check asserts every tool in `agent.ALL_TOOLS` is
in exactly one of the two sets — no tool can be accidentally unclassified.

### Read-only (skip judge)

- All `query_*`, `list_*`, `get_*`, `search_*`, `read_*` tools
- `smartsheet_list_sheets`, `smartsheet_get_sheet`
- `update_progress` (writes the conversation's plan, no external state)
- `investigate` (read-only sub-agent)
- `sync_repo` (idempotent local clone, no external mutation)

### Write (always judge)

- `github_create_issue`, `github_close_issue`, `github_set_issue_type`,
  `github_create_draft_issue`, `github_add_item_to_project`,
  `github_update_item_field`
- `smartsheet_update_row`
- `social_publish_post`, `social_schedule_post`, `social_update_post`,
  `social_delete_post`, `social_preview_post`
- `fedramp_commit_document`, `fedramp_propose_edit`
- `send_teams_message`
- `manage_memory`, `manage_core_memory`, `manage_team_memory`
- `create_background_task`, `pause_background_task`,
  `resume_background_task`, `cancel_background_task`
- All VirtualDojo CRM write operations (per CLAUDE.md autonomy rules)

## Block flow: deny-and-continue

When Stage 2 returns `block`, the judge node emits a synthetic
`ToolMessage` paired with the blocked `tool_call_id`:

```
name="_judge_block"
status="error"
content="BLOCKED by safety judge.

Reason: <Stage 2's 1-2 sentence reason>

Do not retry the same call. Either pick a different target, verify the
IDs by calling a read tool (smartsheet_get_sheet,
github_get_issue_details, etc.), or ask the user to confirm."
```

The agent's next turn sees this as a normal tool-failure result and
revises. No special block-aware logic in the agent loop. This is
Anthropic's *"deny-and-continue"* pattern verbatim.

## Backstop: accumulation limits

Same thresholds as Anthropic:

- **3 consecutive denials** since the last successful tool call, OR
- **20 total denials** in the conversation

→ judge emits an `AIMessage` with content `ESCALATED: ...` instead of
blocking individual calls. The route function sends that to `END`. The
user gets the message in Teams; the agent loop terminates.

Counts are derived by walking the conversation message history for
prior `_judge_block` ToolMessages — no extra state to manage.

## Routing graph

```
START → agent → has_tool_calls?
                  no  → has_text? → verification (existing) → END
                  yes → any_writes?
                          no  → tools (read-only fast path, no judge)
                          yes → judge → verdict?
                                          approve  → tools
                                          block    → inject_block_msg → agent
                                          escalate → END
```

One new node (`judge`), one new conditional predicate (`should_judge_writes`
on the `agent → ...` edge), one new conditional predicate (`route_after_judge`
on the `judge → ...` edge).

## Env gates and rollout

`SAMURAI_JUDGE_WRITES`:
- `off` (default): routing predicate sends every write to `tools` directly. Zero overhead. Safe-by-default until we're ready.
- `shadow`: judge runs end-to-end, logs `[judge.shadow] tool=X verdict=Y reason="..."`, **never blocks**. Use for ~1 week to measure FPR and tune prompts.
- `enforce`: judge blocks on `block` verdict, passes on `approve` and `pass`.

Mirrors `verification.py:VERIFICATION_MODE` exactly so the operational
muscle memory transfers.

Rollout sequence:
1. Ship in `off` mode. Verify no graph regression.
2. Flip to `shadow` for ~1 week. Grep Cloud Logging for `[judge.shadow]`.
   Spot-check verdicts against ground truth.
3. Tune Stage 1 / Stage 2 prompts based on real false positives.
4. Flip to `enforce`. Monitor backstop triggers as the leading indicator
   of either real safety wins or judge over-eagerness.

## Cost and latency

- Stage 1: ~400 input + 1 output token Flash-Lite call ≈ $0.00004 per
  write tool call, ~50ms.
- Stage 2 (~10% of writes, post-tuning): ~700 input + 100 output token
  Flash call ≈ $0.0005 per call, ~300ms.
- Expected total: ~$0.01/day at current volume.
- Read-only turns unaffected — they skip the judge entirely.

## File layout

```
judge.py                            (new, ~180 lines)
docs/judge-design.md                (this file)
tests/test_judge.py                 (new, ~150 lines)
agent.py                            (~40 line edit: imports + graph wiring)
```

### `judge.py` contents

- `WRITE_TOOL_NAMES`, `READ_ONLY_TOOL_NAMES` registries
- `_BLOCK_TOOL_NAME = "_judge_block"` constant
- `_extract_user_messages(messages)` — concatenates all `HumanMessage`
  contents in order
- `_count_prior_denials(messages)` — returns `(consecutive, total)`
- `_make_block_tool_message(tool_call_id, reason)` — synthetic
  `ToolMessage` for deny-and-continue
- `_make_escalation_ai_message(consecutive, total)` — `AIMessage` with
  `ESCALATED` content
- `_stage_1(user_messages, tool_call)` — async, returns
  `Literal["safe", "review"]`
- `_stage_2(user_messages, tool_call, denial_count)` — async, returns
  `(verdict, reason)`
- `judge_writes_node(state)` — async LangGraph node
- `should_judge_writes(state)` — routing predicate after `agent`
- `route_after_judge(state)` — routing predicate after `judge`

## Tests (`tests/test_judge.py`)

| Test | What it guards |
|---|---|
| `test_judge_skips_read_only_tools` | Read-only tools never reach the judge |
| `test_judge_skips_when_env_off` | `SAMURAI_JUDGE_WRITES=off` short-circuits the routing predicate |
| `test_stage1_safe_skips_stage2` | Stage 2 doesn't fire when Stage 1 returns `safe` |
| `test_stage1_review_triggers_stage2` | Stage 2 does fire when Stage 1 returns `review` |
| `test_stage2_approve_passes_through` | Empty messages returned, agent proceeds to tools |
| `test_stage2_block_returns_synthetic_tool_message` | Block creates the right `_judge_block` ToolMessage |
| `test_shadow_mode_logs_but_does_not_block` | Verdicts logged, return is always empty messages |
| `test_three_consecutive_denials_escalates` | Backstop fires on 3 consecutive |
| `test_twenty_total_denials_escalates` | Backstop fires on 20 total |
| `test_judge_prompts_isolate_inputs` | **The critical guard**: judge prompts contain user messages + tool args ONLY. No AIMessage content, no ToolMessage content, no system prompts. Tested by inspecting the actual prompt strings sent to mocked LLM clients with a state full of decoy contamination attempts. |
| `test_concatenates_all_human_messages` | Multi-turn user intent preserved |
| `test_block_message_uses_tool_call_id` | LangGraph correctly threads the block back |
| `test_route_after_judge_sends_block_to_agent` | Routing predicate dispatches correctly |
| `test_route_after_judge_sends_escalation_to_end` | Backstop terminates loop |

## Deferred (mentioned in research, out of scope for this commit)

1. **Inbound prompt-injection probe** on tool results — Anthropic's
   second classifier that warns the agent about suspicious content in
   `ToolMessage` payloads before they enter the agent's context. Real
   gap for us (GitHub issue bodies, Smartsheet cells are attacker-
   controllable text). Separate ~150 line piece of work, separate
   design pass.

2. **Subagent dispatch judging** — Anthropic's outbound/return checks
   on sub-agent delegation. SamurAI's `investigate()` sub-agent is
   read-only at the sub-agent level so the write-judge concern doesn't
   apply directly. An "intent alignment" check before dispatching is a
   different judge — not write-related.

3. **Plan-mode approval integration** — using `update_progress` plans
   as a basis to relax the judge for writes inside an already-approved
   plan. Anthropic's *"approve the strategy, not each step"* idea. Best
   done after shadow data shows what the false-positive rate is.

4. **Tiered judge by tool risk** (low / medium / high writes) — start
   uniform; split later if FPR data shows certain tools deserve lighter
   scrutiny.

5. **Synchronous human approval** for high-blast-radius operations —
   Anthropic doesn't have this either; they rely on the accumulation
   backstop. Add only if the backstop proves insufficient.
