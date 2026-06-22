# SamurAI Tenant-Data Read Access via Support Grants — Plan

## ✅ IMPLEMENTED DESIGN (SSO model — supersedes the API-key plan below)

The shipped design (`tools/tenant_data.py`) **authenticates as the signed-in SamurAI user
via their VirtualDojo SSO token, not a service/superadmin API key.** Why: an `sk-` API key
resolves a *tenant* but not a *user*, so `get_current_user` (`deps.py`) 403s it. The SSO
token is a standard JWT minted by `create_access_token(subject=user_id, tenant_id, scopes)`
— and `get_current_user` just decodes any `SECRET_KEY`-signed JWT with `sub`+`exp`, so the
**same SSO token authenticates `/api/v1` REST.** (Verified live: signed in as Superadmin
Tenant → `GET /impersonation/my-grants` returned 6 active grants.)

**Authorization = SSO sign-in + the user's explicit in-chat request.** The user signing in
and asking IS the direct authorization (Devin confirmed SSO is sufficient), and reads run
under *their* identity + `system_administrator` rights. **Never autonomous:** a background
task has no signed-in user → `_signed_in_token()` returns the SSO prompt and never reaches
the backend (covered by a test). No separate per-read approval card — the SSO model makes
the human the load-bearing control. (A per-read card is an available future hardening.)

**Tools (per-user factory `create_tenant_data_tools(user_id)`, gated `SAMURAI_TENANT_DATA_ENABLED`):**
- `list_tenant_support_grants` → `GET /api/v1/impersonation/my-grants` (Phase 1, validated live).
- `describe_tenant_schema(grant_id, object_name?)` → start a 15-min impersonation session as
  the granting user (`POST /impersonation/start/{grant_id}`) → `GET /schema/objects[/{object}/schema]`.
- `read_tenant_records(grant_id, object_name, limit, skip)` → start session → `GET /objects/{object}/records`.

**Read-only by construction:** data + schema are GET-only and `confirm_write` is never sent.
The *only* non-GET verb is impersonation session start/end (`_vdj_post`, asserted to
`/impersonation/` paths) — it mints/ends a read token and mutates no customer data; the
backend audits the session start into the customer tenant. Per-read audit (metadata only,
never row contents): `[samurai.tenant_data_access]`.

**#2 — env consistency (deploy requirement):** the SSO token only validates against
`/api/v1` if the user signed in against the SAME backend that serves `VIRTUALDOJO_API_URL`
(shared `SECRET_KEY`). So to enable in prod: set `VIRTUALDOJO_API_URL=https://app.virtualdojo.com`
**and** point the bot's VirtualDojo SSO (`VIRTUALDOJO_MCP_URL`) at the same host
(`https://app.virtualdojo.com/mcp/v1`). Today the CRM SSO defaults to **dev** — moving it to
prod also moves the `virtualdojo_crm` tool to prod (a product decision to confirm).
`_warn_if_sso_env_mismatch()` logs a loud hint if the hosts diverge.

**⚠ Residency gate before enabling record reads in prod:** record rows may be customer
PII/CUI. Reads are never fed to LangMem/KB and only metadata is logged, but the serving chat
model is on the Vertex **global** endpoint — summarizing raw rows through it is a residency
item to resolve / get ATO sign-off on. `describe_tenant_schema` (names only) is low-risk;
`read_tenant_records` (rows) is what the residency decision gates. Code ships gated **off**.

---

## Original plan (API-key model — kept for reference; superseded above)

Goal: let SamurAI read a customer tenant's data/schema **read-only**, via VirtualDojo's
existing support-grant mechanism, authenticating as the **superadmin/support tenant**
with a superadmin API key — and **only when a SamurAI user explicitly authorizes that
specific access**. Never autonomous.

Design constraints (from Devin):
1. **Build our own read-only tools** — borrow the `virtualdojo_cli` + backend as
   *references* (endpoints, auth-header shape, read patterns); do NOT import/run the CLI.
2. **Auth via a superadmin API key against the superadmin tenant.** Tenants authorize
   support grants *to the superadmin tenant*; SamurAI (as that tenant) uses them to read
   on the tenant's behalf.
3. **Per-access human authorization** — a SamurAI user must approve each specific tenant
   read; the scheduled triage task is hard-barred.

## How the mechanism works (grounded references)

CLI = `~/Code/virtualdojo_cli`, backend = virtualdojo-inc/virtualdojo.

- **Auth header shape** (reference): every request carries `Authorization: Bearer <token>`
  + `X-Tenant-ID: <tenant>` to `<server>` (`client.py:83-84`). Our superadmin credential
  authenticates as the support tenant.
- **Grant model** — tenant-scoped, not user-scoped: `LoginAccessGrant(tenant_id=customer,
  granted_to_tenant_id=support, granting_user_id, expires_at, is_active)`
  (`app/models/login_access.py:19-83`). A customer creates it via `POST /api/v1/login-access/grants`;
  server requires the target be a `TenantType.SUPPORT` tenant with a `system_administrator`
  (`app/api/v1/endpoints/login_access.py:176-318`). **This is the tenant's consent.**
- **List grants:** `GET /api/v1/login-access/grants`.
- **Start a read session:** `POST /api/v1/impersonation/start/{grant_id}` — server validates
  caller's tenant == grant's `granted_to_tenant_id`, caller is `system_administrator`, grant
  active/unexpired; mints a **15-min JWT as the customer user, for the customer tenant**
  (`impersonation.py:358-417`). No refresh token; renewable to an 8h ceiling
  (`impersonation.py:613-760`). With the superadmin key + sysadmin, **SamurAI performs this
  itself** (no human-runs-CLI step).
- **Read endpoints** (read-only by construction): `GET /api/v1/schema/objects`,
  `GET /api/v1/schema/objects/{object}/schema`, `GET /api/v1/objects/{object}/records`.
  `POST /api/v1/sql/query` needs a server `confirm_write` flag for mutations.
- **Read-only is NOT enforced broadly at the impersonation layer** — guards are per-endpoint
  (`deps.py:187-189` sets `_is_impersonated`), not middleware-wide. **So SamurAI enforces
  read-only itself: call only GET endpoints, never send `confirm_write`, reject non-SELECT SQL
  client-side.**
- **Audit:** session ops audit into the *customer* tenant's AuditLog; the backend does **not**
  log per-endpoint impersonated reads → SamurAI's own per-read audit is the fine-grained record.

## The tools (our own; `tools/tenant_data.py`, new `tenant_data` group — never in `core`)

- **`list_tenant_support_grants`** (read-only) — as the support tenant, `GET /login-access/grants`
  → which tenants have an authorized grant + the `grant_id`. Classify read-safe (no card).
- **`describe_tenant_schema`** / **`read_tenant_records`** (read-only, but write-gated, see below) —
  on an approved access: SamurAI starts the impersonation session itself, then GETs schema/records.
- **(v2) `query_tenant_sql`** — SELECT-only; reject non-SELECT before sending; never `confirm_write`.

## Authorization model (the crux) — two gates, both required

- **Gate 1 — tenant consent (backend-enforced):** SamurAI can only `impersonation/start` for a
  tenant that authorized an active grant to the support tenant. The superadmin key cannot
  manufacture access to a tenant that didn't grant it.
- **Gate 2 — SamurAI-user per-access approval (the human gate):** the read tools are placed in
  `judge.WRITE_TOOL_NAMES` so every call routes through the judge → an **Approve/Revise/Reject
  card** (reuse `PendingApproval` + `cards/actions.py:handle_card_action` dispatch — note: there
  is no `cards/approvals.py`/`register()`; mirror the social/task card pattern). The approval:
  - is by the **verified clicker** (`TeamsInfo.get_member`, not the card's claimed email),
    restricted to Devin/Cyrus (`AUTHORIZED_TENANT_USERS`), single-use atomic claim;
  - **binds the exact scope** (`payload_sha256` over `{tenant_id, object/query, grant_id}`); on
    approve, re-validate the minted JWT's `tenant_id` == approved `tenant_id` before the GET;
  - is **time-boxed** (card ~1h; the impersonation token 15 min — whichever expires first fails closed).
  Because SamurAI mints the session itself, **Gate 2 is the load-bearing human control.**
- **Never autonomous / triage barred:** `tenant_data` group is never in `core`; multi-word keywords
  only (`"tenant data"`, `"read customer data"`, `"support grant"`, `"impersonate tenant"`). Plus a
  hard refusal in the tool: if `is_background_task` **or** no `approved` `PendingApproval` matches the
  `payload_sha256`, refuse and never call the backend. A background task can never satisfy Gate 2
  (no human clicker). Add a test asserting no existing task prompt activates the group.

## Security / compliance

- **Superadmin key:** Secret Manager (region-pinned us-central1, mounted only on the bot, never
  logged) — same pattern as `samurai-admin-key`/`SANDBOX_TOKEN`. **Recommend the key be scoped
  read-only at the backend if possible**; if it's a full superadmin key, our GET-only tool surface
  + Gate 2 are what keep it read-only in practice.
- **Read-only:** GET-only, no `confirm_write`, SELECT-only SQL — enforced in our code.
- **Audit:** per approved read, log `[samurai.tenant_data_access]` {ts, requesting_user, verified
  approver, tenant_id, grant_id, object/query, result_count, payload_sha256}.
- **FedRAMP / PII residency (gates production):** tenant rows may be PII/CUI.
  - **Never** feed tenant-read results into LangMem memory extraction or the KB bucket; log only
    metadata (counts/object names), never row contents.
  - The serving chat model is on the Vertex **global** endpoint (`gemini-3.5-flash` is global-only).
    **Summarizing raw tenant PII through it is a residency concern.** Resolve before prod: either
    route tenant-data turns through a regional in-boundary model, or return reads as structured data
    the user inspects and bar the model from echoing rows — or get ATO sign-off.

## Phased rollout
- **Phase 0 (blocking unknowns — confirm first):** (1) **Is SamurAI's tenant a `TenantType.SUPPORT`
  tenant with a `system_administrator`?** If not, no grant can target it + `impersonation/start` 403s.
  (2) Provision the **superadmin API key** + confirm it can list grants + start impersonation.
  (3) **Residency** decision for tenant-data turns through the global serving model.
- **Phase 1:** ship `list_tenant_support_grants` only, behind `SAMURAI_TENANT_DATA_ENABLED` (off).
- **Phase 2:** `describe_tenant_schema` + `read_tenant_records`, write-gated via the `tenant_data_*`
  approval card + per-read audit + background-task refusal + the group-exclusion test.
- **Phase 3:** SELECT-only `query_tenant_sql` + optional `impersonation/renew` for long reads — only
  after the residency question is resolved/ATO-signed.

## Open questions
- Q1 SamurAI support-tenant + sysadmin status (highest priority).
- Q2 Superadmin-key provisioning + scope (read-only if possible).
- Q3 Tenant-data residency through the global serving model (blocks prod).
- Q4 Does the backend expose a superadmin/cross-tenant path, or is "as the support tenant" sufficient
  to list all grants + start any granted impersonation? (Confirm in the backend API.)
- Q5 Session/renew handling for multi-turn reads given 15-min, no-refresh tokens.
