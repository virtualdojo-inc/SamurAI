# SamurAI Reusable Approve/Revise/Reject Teams Card — MVP Design

**Status:** Proposal for review · **Owner:** Devin · **Updated:** 2026-06-21
Verified against the live tree + Microsoft Learn (Adaptive Cards / Bot Framework). Goal: **one**
reusable Teams Adaptive Card approval mechanism (Approve / Revise / Reject), everything in Teams, no
external dashboard. Shared by skill-promotion, the controlled-fix CI dispatch, social publish, FedRAMP
commit, and GCP/config changes.

## The key simplification
Authorize off the **server-verified clicker** (`TeamsInfo.get_member` / the JWT-attested
`activity.from_property`), never off an email in card data. That makes the card payload a **non-secret
`request_id` lookup key**, not a capability token — so **no HMAC / signed card data is needed for MVP**.
This also *fixes the existing social-flow spoofing hole* (which trusts an LLM-stamped `draft.user_email`)
as a side benefit.

## 1. Card UX — one builder, three verbs, in-place decision
`build_approval_card(request_id, title, summary, facts, sensitive)` in a new `cards/approvals.py`, modeled
on `cards/feedback.py:33-78` + `cards/social.py`. Buttons carry **verb + request_id only** (never the
payload, never an email):
- **Approve** → `Action.Submit`, `data={"approval_action":"approve","request_id":<uuid>}`
- **Reject** → `Action.Submit`, `data={"approval_action":"reject","request_id":<uuid>}`
- **Revise** → two-step (Teams only delivers typed `Input.Text` on a Submit press, verified):
  1. `Action.ToggleVisibility` reveals a hidden `Input.Text id="revise_note"`,
  2. a second `Action.Submit` `data={"approval_action":"revise","request_id":<uuid>}` ships the note.

`version: "1.4"`, plain `Action.Submit` (proven here today). **In-place decision:** reuse
`cards/actions.py:296-324` `_update_or_send_card` (update by stored activity id, fallback to fresh send);
on any terminal action, swap to a **buttonless** terminal card ("Approved by devin@… · ts") so it can't
be re-clicked. Store the activity id keyed by `(conversation_id, request_id)`, fixing the current
single-slot collision.

## 2. The one reusable module — `cards/approvals.py` (~120 lines)
```
APPROVERS = {"devin@virtualdojo.com", "cyrus@virtualdojo.com"}
_REGISTRY: dict[str, ApprovalHandler] = {}

def register(action_type, *, on_approve, on_revise=None, sensitive=True): ...
async def request_approval(turn_context, *, action_type, payload, title, summary, facts) -> request_id:
    # uuid → durable row (status=pending, payload_json, payload_sha256, expires_at=now+1h) → send card → store activity_id
async def resume(turn_context, value, clicker_email):  # the dispatcher (see §3)
def parse_approval_submit(value): ...  # defensive nesting parse, modeled on feedback.parse_feedback_submit
```
Each caller `register(...)`s at import; the callback closes over the real tool, so **Approve/Reject call
the tool directly — they do NOT re-run the agent**. **Revise** routes the note back to the agent (§6),
which edits and calls `request_approval` again with a fresh `request_id`.

## 3. Security core (in `approvals.resume`) — fixes the two confirmed HIGH holes
1. **Verified clicker only.** `app.py` resolves `clicker_email` via `TeamsInfo.get_member` before dispatch
   and passes it in; `resume` enforces `clicker_email in APPROVERS` for sensitive actions and **ignores any
   email in card data**. Kills both spoof variants (any-member approve; forged `value` dict) because the
   card no longer carries authority. Keep the coarse `@virtualdojo.com` gate as layer 1.
2. **Single-use atomic claim.** `UPDATE pending_approvals SET status='claimed',… WHERE request_id=? AND
   status='pending'`; proceed only if `rowcount==1`. Closes double-click + error-replay.
3. **Payload-hash binding.** Payload lives only in the DB row keyed by `request_id`; the card carries no
   payload, so there's nothing to mutate. `payload_sha256` is a cheap audit/integrity anchor (logged), not a
   depended-on defense — which is why no HMAC is needed.
4. **Expiry.** `expires_at = now+1h`; reject stale; an APScheduler purge job (reuse the `asyncio.to_thread`
   cron pattern) deletes terminal/expired rows.
5. **Audit.** Log `(request_id, action_type, verified approver, decision, payload_sha256, ts)` to Cloud
   Logging (same `print(..., flush=True)` convention).

## 4. Durable state — `pending_approvals` table in `task_store.py`
In-memory `_pending_posts` is lost on every Cloud Run drain → use durable SQLite on `/data`.
```sql
CREATE TABLE IF NOT EXISTS pending_approvals (
  request_id TEXT PRIMARY KEY, action_type TEXT NOT NULL, conversation_id TEXT NOT NULL,
  payload_json TEXT NOT NULL, payload_sha256 TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',   -- pending|claimed|approved|rejected|revising|expired
  requested_by TEXT NOT NULL DEFAULT '', approver_email TEXT, activity_id TEXT, revise_note TEXT,
  created_at REAL NOT NULL, expires_at REAL NOT NULL, decided_at REAL
);
```
Methods mirror `save_team_member`/`get_team_member`: `create_pending_approval`, `get_pending_approval`,
`claim_pending_approval(request_id, email)->bool` (the atomic UPDATE), `set_activity_id`, `purge_expired`.
`request_id` PK fixes the conversation_id collision by construction.

## 5. Wiring (the crux fix is one line at the dispatch boundary)
- `app.py:96-98`: resolve verified clicker first (lift the message-path `TeamsInfo.get_member` logic at
  `app.py:174-177` into `_resolve_clicker_email`), then `handle_card_action(turn_context, value, clicker_email)`.
- `cards/actions.py:39`: add `clicker_email` param; branch `if value.get("approval_action"): return await
  approvals.resume(...)`. **Delete** the spoofable `draft.get("user_email")` reads at `cards/actions.py:91,
  123, 254` and authorize on `clicker_email` — fixes the legacy social flow immediately.
- Reuse `feedback.py` defensive parsing + the `request_id` round-trip (like `turn_id` at `feedback.py:42`).
- **5-second invoke deadline:** ack fast — do the claim + card swap synchronously; run slow `on_approve`
  (Ayrshare publish, CI dispatch, FedRAMP commit) via `asyncio.create_task` after returning the updated card.
- Plain `Action.Submit` routes through the existing `activity_name is None` branch (`app.py:92-98`) — **no
  new invoke type, no TaskModule** (that path stays for the native 👍/👎 loop).

## 6. All callers use the SAME mechanism
| Caller | action_type | on_approve runs | on_revise |
|---|---|---|---|
| Social publish/schedule | `social_publish` | `social_publish_post.invoke(payload)` | re-prompt agent → new draft → new request_approval |
| Controlled-fix CI dispatch (deferred) | `controlled_fix_dispatch` | `gh workflow run` (verified approver ≠ agent satisfies requester≠approver) | edit scope, re-present |
| Skill promotion (skills plan) | `skill_promote` | promote `candidate/` → live `skills/**` | adjust skill, re-present |
| FedRAMP commit | `fedramp_commit` | commit OSCAL package | revise text, re-present |
| GCP/config change | `gcp_config` | apply the change | revise params, re-present |

A caller adds ~5 lines (`register(...)`) + swaps its bespoke card for `request_approval(...)`. The **Revise**
resume reuses the proven `scheduler.py` `ConversationReference` + `adapter.continue_conversation` + `run_agent`
path; Approve/Reject never touch the agent.

## 7. What to NOT build
- **No Action.Execute / Universal Actions** — Submit is proven; `_update_or_send_card` already solves in-place
  update; Universal adds a version-floor ambiguity + new invoke type for zero MVP benefit.
- **No HMAC / signed card data** — authority is server-side; the card carries only a non-secret `request_id`.
- **No web/admin dashboard, no approval inbox** — everything in the Teams thread.
- **No new invoke type / TaskModule** for the approval card — reuse the `activity_name is None` Submit branch.
- **No generic state-machine engine** — five string statuses in one SQLite column is the whole machine.
- **No auto-retry/queue for on_approve** — on failure, surface the error on the card; re-request fresh.
- **No per-caller card layouts** — one `build_approval_card(title, summary, facts)` for all five.

## Red-team guardrails baked in (all four were HIGH)
spoof → verified-clicker auth · replay/stale → single-use atomic claim + expiry + buttonless terminal card ·
mutated action → payload lives server-side keyed by request_id (nothing to swap) + sha256 audit ·
revise-injection → wrap the revise note as untrusted DATA (fenced) on agent resume + bound revise rounds.

## Files
- **NEW** `cards/approvals.py` (builder + parse + dispatcher).
- `app.py:96-98` (resolve clicker, pass through), `cards/actions.py:39` (signature + branch + delete spoofable
  reads at 91/123/254), `task_store.py:88,340+` (table + CRUD), `scheduler.py` (purge job),
  +5-line `register(...)` per caller.
- Tests: verified-clicker auth (reject non-approver + forged email), single-use claim (double-click no-ops),
  expiry, revise two-step + note-as-data, terminal-card swap, all five callers register.
