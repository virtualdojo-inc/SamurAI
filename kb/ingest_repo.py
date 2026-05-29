"""Ingest the virtualdojo repo into the KB engineering scope (mechanical, no LLM).

Refreshes ``engineering/raw/`` with:
  - ``docs/``      — allowlisted, secret-scrubbed human-written docs (verbatim source).
  - ``structure/`` — grounded, names-only inventories (the router map + dir listings).
  - ``stubs/``     — one **article stub** per target: a scaffolded skeleton with the
                     relevant pre-grouped material + code pointers + doc refs + inline
                     ``<!-- WRITE: ... -->`` authoring instructions. The compile step
                     (``kb.compile_engineering``) only ever reads ``stubs/`` and fleshes
                     each one into a final ``engineering/wiki|troubleshooting`` article.

Doing the structural reasoning HERE (mechanically) — rather than asking Gemini to
infer it from a flat file list — is what makes it obvious how each piece of the
system maps to each article. The clustering (``DOMAINS``) lives in exactly one
place, is inspectable in the bucket before compile, and regenerates every sync so
it tracks the live repo.

Runs IN-BOUNDARY on samurai-bot; reuses ``tools/github.py`` auth and the
``kb.ingest_github`` secret scrubber. No LLM involved. STRICT allowlist: only the
curated docs + names-only structure are ever fetched — never ``.env``, SQL dumps,
``database_dumps/``, ``credentials/``, fixtures, etc.
"""

from __future__ import annotations

import fnmatch
import re
from datetime import datetime, timezone

from kb import storage
from kb.ingest_github import _scrub
from tools.github import _github

REPO = "virtualdojo-inc/virtualdojo"
REF = "main"

RAW_PREFIX = "engineering/raw/"
DOCS_PREFIX = RAW_PREFIX + "docs/"
STRUCT_PREFIX = RAW_PREFIX + "structure/"
STUBS_PREFIX = RAW_PREFIX + "stubs/"
STATE_PATH = RAW_PREFIX + ".state/repo_last_sync.txt"

# Embedded doc text in a stub is capped so one stub stays a sane compile payload.
_DOC_EMBED_CAP = 24000

# --- Allowlist: ONLY these human-written docs are fetched (anything else, never).
# Single files + directory trees (walked recursively, .md only).
DOC_FILES = [
    "README.md",
    "CLAUDE.md",
    "AUTOFIX_PROCESS.md",
    "LOGIN_TROUBLESHOOTING_GUIDE.md",
    "PRODUCTION_DEPLOYMENT_TROUBLESHOOTING.md",
    "WORKFLOW_GUIDE.md",
    "TENANT_PROVISIONING_ANALYSIS.md",
    "TENANT_ADMINISTRATION_PLAN.md",
]
DOC_TREES = ["docs/project", "docs/features", "docs/setup"]

# Directories we inventory by NAME only (no code bodies). The router map (api.py)
# is the one file whose *content* we parse — and only for the include_router map.
STRUCTURE_DIRS = {
    "services": "app/services",
    "models": "app/models",
    "endpoints": "app/api/v1/endpoints",
    "middleware": "app/middleware",
    "frontend-views": "frontend/src/views",
    "frontend-stores": "frontend/src/stores",
    "frontend-services": "frontend/src/services",
}

# --- Domain clustering. Ordered (most specific first); each backend file is
# assigned to the FIRST domain whose substrings match its name, else "misc".
# Matchers are plain substrings — grounded (the filename literally contains them).
# Each domain: svc/mdl (backend file substrings), tags (router-tag substrings),
# fe (frontend file substrings), docs (doc slugs to surface for this domain).
DOMAINS: list[tuple[str, str, dict]] = [
    ("agents", "Agents, AI execution & MCP-less reasoning", {
        "svc": ["agent_", "react_chat", "langgraph_reasoning", "enhanced_langchain", "langchain_agent"],
        "mdl": ["agent", "agent_facts", "agent_execution", "agent_task"],
        "tags": ["agent", "ai-chat", "ai-component", "natural-language", "ai-token"],
        "fe": ["Agent", "AIChat", "NaturalLanguageAgent"], "docs": []}),
    ("mcp", "Model Context Protocol servers & connections", {
        "svc": ["mcp_"], "mdl": ["mcp_"], "tags": ["mcp"],
        "fe": ["profileMCP", "Connector"], "docs": []}),
    ("packages", "Package authoring, licensing & marketplace", {
        "svc": ["package_", "component_packager", "secure_component", "component_security", "dependency_resolver"],
        "mdl": ["package"], "tags": ["package", "component-management", "marketplace"],
        "fe": ["Package", "ComponentRegistry", "Developer"], "docs": []}),
    ("flows", "Flow automation engine & triggers", {
        "svc": ["flow_", "trigger_execution", "button_executor"],
        "mdl": ["flow_", "object_trigger", "trigger"], "tags": ["flow", "object-trigger"],
        "fe": ["flow", "Flow"], "docs": []}),
    ("quoting-forecasts", "Quotes, formulas, forecasting & products", {
        "svc": ["quote_", "formula_engine", "forecast", "stage_probability", "numbering", "math_validator", "pdf_generator", "pdf_content"],
        "mdl": ["quote", "forecast", "product", "quota", "pdf_template", "picklist"],
        "tags": ["formula", "forecast", "products", "chart-data"],
        "fe": ["quote", "Quote", "Forecast"], "docs": []}),
    ("compliance-fedramp", "Compliance, CUI/PII classification & FedRAMP", {
        "svc": ["compliance", "cui_", "pii_", "catap", "clin", "ites_", "tr_", "trust_scoring", "fact_sanitization", "document_review", "math_validator"],
        "mdl": ["compliance", "clin", "classification", "contract_vehicle", "entity_use", "equipment_failure", "tr_response"],
        "tags": ["compliance", "cui", "pii", "clin", "cmmc", "auditor", "edr", "mdm"],
        "fe": ["compliance", "government-contracting", "Document"], "docs": []}),
    ("email", "Email templates, sending, queue & inbound processing", {
        "svc": ["email_", "sendgrid", "flow_email"],
        "mdl": ["email", "flow_inbound_email"], "tags": ["email", "inbound-email", "agent-email"],
        "fe": ["email", "Email"], "docs": []}),
    ("tenant-provisioning", "Tenant provisioning, data copy & standard objects", {
        "svc": ["tenant_", "provision", "sql_provisioner", "enhanced_sql", "standard_objects", "standard_picklist", "navigation_profile", "business_tenant", "business_sample", "sandbox", "default_layouts", "custom_object_system"],
        "mdl": ["tenant", "sandbox", "registration", "workspace"], "tags": ["tenant", "provisioning"],
        "fe": ["Tenant", "Provision"], "docs": ["tenant-provisioning-analysis", "tenant-administration-plan"]}),
    ("auth-permissions-sharing", "Auth, SSO/MFA, permissions, profiles & sharing", {
        "svc": ["sso_", "mfa_", "idp_mfa", "permission_", "sharing_", "profile_", "record_access", "public_group", "user_hierarchy", "license_service", "credential_encryption", "internal_jwt", "sso_auth"],
        "mdl": ["permission", "profile", "sharing", "login_access", "password_reset", "refresh_token", "user_sso", "public_group", "api_key", "user_role"],
        "tags": ["authentication", "sso", "mfa", "login-access", "impersonation", "roles", "profiles", "sharing", "api-keys", "credentials", "auditor"],
        "fe": ["Admin", "Login", "Mfa", "Auth", "ChangePassword", "ForgotPassword", "AcceptInvitation", "CLILogin"],
        "docs": ["login-troubleshooting-guide"]}),
    ("payments-billing-tax", "Payments, billing, commissions & tax", {
        "svc": ["stripe", "payment", "billing", "avatax", "quickbooks", "commission", "credit_card", "order_management"],
        "mdl": ["billing", "payment", "commission", "order_management", "tax_provider"],
        "tags": ["billing", "commission", "payment"], "fe": ["billing", "Billing", "Tax", "Commission", "OrderManagement"], "docs": []}),
    ("approvals", "Approval processes, requests & delegations", {
        "svc": ["approval"], "mdl": ["approval"], "tags": ["approval"],
        "fe": ["approval", "Approval"], "docs": []}),
    ("rfx-ingestion", "RFx ingestion & response parsing", {
        "svc": ["rfx_", "data_extraction", "extraction_", "document_chunking", "document_single", "excel_import", "embedding_service", "salesforce", "hubspot", "dynamics_oauth", "contact_enrichment"],
        "mdl": ["rfx", "extraction_rule", "manufacturer", "adapter_type", "file_embedding"],
        "tags": ["rfx", "data-import", "contact-enrichment", "document-processor", "crm-connection", "crm-push"],
        "fe": ["rfx", "Rfx", "FileUpload"], "docs": []}),
    ("records-schema-objects", "Dynamic schema, custom objects, layouts & list views", {
        "svc": ["dynamic_", "schema_introspection", "record_type", "related_objects", "list_view", "report", "recycle_bin", "crud_operations", "custom_object", "registration_rollup", "file_storage_rollup", "activity_service"],
        "mdl": ["dynamic_schema", "custom_object", "custom_component", "custom_button", "record_type", "page_layout", "list_view", "report", "relationship", "standard_object", "navigation", "essential_info"],
        "tags": ["schema", "custom-objects", "custom-components", "custom-buttons", "list-views", "reports", "navigation", "essential-info"],
        "fe": ["object-manager", "LayoutEditor", "DatabaseSchema", "Contacts", "Dashboard", "Files"], "docs": []}),
]
MISC_DOMAIN = ("misc", "Other / not yet clustered", {})


def _slug(path: str) -> str:
    """Slugify a repo path into a flat doc filename stem."""
    s = re.sub(r"\.md$", "", path)
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:80]


def _text_of(content_file) -> str:
    try:
        return content_file.decoded_content.decode("utf-8", "replace")
    except Exception:  # pragma: no cover - defensive
        return ""


def _fetch_docs(repo) -> tuple[dict, int]:
    """Return ({slug: scrubbed_text}, total_secrets_redacted) for the allowlist."""
    docs: dict[str, str] = {}
    redacted = 0
    paths: list[str] = list(DOC_FILES)
    for tree in DOC_TREES:
        try:
            stack = [tree]
            while stack:
                cur = stack.pop()
                for c in repo.get_contents(cur, ref=REF):
                    if c.type == "dir":
                        stack.append(c.path)
                    elif c.name.endswith(".md"):
                        paths.append(c.path)
        except Exception:  # tree may not exist on a given ref
            continue
    for path in paths:
        try:
            cf = repo.get_contents(path, ref=REF)
        except Exception:
            continue
        if isinstance(cf, list):  # a tree slipped into DOC_FILES — skip
            continue
        clean, n = _scrub(_text_of(cf))
        redacted += n
        docs[_slug(path)] = clean
        fm = (
            "---\n"
            f"source: github-repo\nrepo: {REPO}\npath: {path}\nref: {REF}\n"
            f"ingested_at: {datetime.now(timezone.utc).isoformat()}\n"
            f"secrets_redacted: {n}\n---\n\n"
        )
        storage.write_text(f"{DOCS_PREFIX}{_slug(path)}.md", fm + clean)
    return docs, redacted


def _list_names(repo, path: str) -> list[str]:
    try:
        return sorted(
            c.name for c in repo.get_contents(path, ref=REF)
            if c.name.endswith(".py") or c.name.endswith(".vue") or c.name.endswith(".js") or c.type == "dir"
        )
    except Exception:
        return []


_ROUTER_CALL_RE = re.compile(r"include_router\((.*?)\)", re.DOTALL)


def _router_map(repo) -> list[tuple[str, str, str]]:
    """Parse app/api/v1/api.py into (router, prefix, tags) rows. Mechanical, no LLM."""
    try:
        text = _text_of(repo.get_contents("app/api/v1/api.py", ref=REF))
    except Exception:
        return []
    rows: list[tuple[str, str, str]] = []
    for m in _ROUTER_CALL_RE.finditer(text):
        arg = m.group(1)
        router = (re.search(r"(\w+)\.router", arg) or [None, ""])[1] if re.search(r"(\w+)\.router", arg) else ""
        prefix = (re.search(r"prefix\s*=\s*[\"']([^\"']*)", arg) or [None, ""])
        prefix = prefix[1] if prefix else ""
        tags = re.search(r"tags\s*=\s*\[([^\]]*)\]", arg)
        tags = re.sub(r"[\"']", "", tags.group(1)).strip() if tags else ""
        if router or prefix or tags:
            rows.append((router, prefix, tags))
    return rows


def _collect_inventory(repo) -> dict:
    """Names-only inventories + the parsed router map. Written to structure/."""
    inv = {name: _list_names(repo, path) for name, path in STRUCTURE_DIRS.items()}
    inv["router"] = _router_map(repo)
    return inv


def _write_structure(inv: dict) -> int:
    n = 0
    for name, items in inv.items():
        if name == "router":
            rows = "\n".join(f"| {r or '?'} | {p or '/'} | {t} |" for r, p, t in items)
            body = (
                "# Backend router map (`app/api/v1/api.py`)\n\n"
                "| router | prefix | tags |\n|---|---|---|\n" + rows + "\n"
            )
        else:
            body = f"# {name} ({len(items)})\n\n" + "\n".join(f"- {i}" for i in items) + "\n"
        storage.write_text(f"{STRUCT_PREFIX}{name}.md", body)
        n += 1
    return n


# --- Domain assignment (first-match) ----------------------------------------

def _assign(names: list[str], key: str) -> dict[str, list[str]]:
    """First-match each backend filename into a domain by `key` ('svc'|'mdl')."""
    out: dict[str, list[str]] = {d[0]: [] for d in DOMAINS}
    out[MISC_DOMAIN[0]] = []
    for nm in names:
        placed = False
        for slug, _title, m in DOMAINS:
            if any(sub in nm for sub in m.get(key, [])):
                out[slug].append(nm)
                placed = True
                break
        if not placed:
            out[MISC_DOMAIN[0]].append(nm)
    return out


def _match_any(names: list[str], subs: list[str]) -> list[str]:
    return [nm for nm in names if any(s in nm for s in subs)]


def _tags_for(router_rows: list[tuple], subs: list[str]) -> list[str]:
    hits = []
    for _r, _p, tags in router_rows:
        for tag in [t.strip() for t in tags.split(",") if t.strip()]:
            if any(fnmatch.fnmatch(tag, f"*{s}*") for s in subs):
                hits.append(tag)
    return sorted(set(hits))


# --- Stub builders ----------------------------------------------------------

def _fm(title: str, summary: str, kind: str) -> str:
    return f"---\ntitle: {title}\nsummary: {summary}\nkind: {kind}\n---\n\n"


def _embed_doc(docs: dict, slug: str) -> str:
    body = docs.get(slug, "")
    if not body:
        return f"<!-- doc '{slug}' not found in this sync -->\n"
    if len(body) > _DOC_EMBED_CAP:
        body = body[:_DOC_EMBED_CAP] + "\n\n<!-- (truncated) -->"
    return f"> Source doc `{slug}` (verbatim, secret-scrubbed):\n\n{body}\n"


def _stub_system_map(docs, inv, svc_by, mdl_by) -> str:
    out = [_fm("VirtualDojo — System Map",
               "Domain map of the VirtualDojo CRM: which backend services/models, API tags, and frontend views own each subsystem.",
               "wiki"),
           "# VirtualDojo System Map\n",
           "<!-- WRITE: 2-3 sentence intro — VirtualDojo is a multi-tenant, "
           "Salesforce-style CRM (FastAPI backend + Vue frontend). Then keep each "
           "domain section below; for each, write ONE paragraph on the domain's role "
           "and how a request flows, citing the code paths listed. Map, don't mirror: "
           "do NOT paste code; these names are pointers for repo_sync. -->\n"]
    views = inv.get("frontend-views", [])
    router = inv.get("router", [])
    for slug, title, m in DOMAINS:
        svcs, mdls = svc_by.get(slug, []), mdl_by.get(slug, [])
        fes = _match_any(views, m.get("fe", []))
        tags = _tags_for(router, m.get("tags", []))
        if not (svcs or mdls or fes or tags):
            continue
        out.append(f"## {title} (`{slug}`)\n")
        if svcs:
            out.append(f"- **Services** (`app/services/`): {', '.join(svcs[:24])}")
        if mdls:
            out.append(f"- **Models** (`app/models/`): {', '.join(mdls[:24])}")
        if tags:
            out.append(f"- **API tags**: {', '.join(tags)}")
        if fes:
            out.append(f"- **Frontend views**: {', '.join(fes[:16])}")
        out.append("\n<!-- WRITE: one paragraph on this domain. -->\n")
    return "\n".join(out)


def _stub_backend_flow(docs, inv) -> str:
    rows = inv.get("router", [])
    table = "\n".join(f"| {r or '?'} | {p or '/'} | {t} |" for r, p, t in rows)
    mw = inv.get("middleware", [])
    return (
        _fm("VirtualDojo — Backend Request Flow",
            "How a request flows through the FastAPI backend: routers (api.py) → middleware → services → models; multi-tenant resolution.",
            "wiki")
        + "# Backend Request Flow\n\n"
        + "<!-- WRITE: describe the path of a request: app/main.py mounts the v1 "
          "router (prefix from settings.API_V1_STR) and the /mcp/v1 router; "
          "middleware runs (list below); endpoints delegate to app/services/*; "
          "services use app/models/* via the DB session. Call out multi-tenant "
          "resolution explicitly. Cite paths; do NOT paste code. -->\n\n"
        + "## Router map (`app/api/v1/api.py`)\n\n"
        + "| router | prefix | tags |\n|---|---|---|\n" + table + "\n\n"
        + "## Middleware (`app/middleware/`)\n\n"
        + ", ".join(mw) + "\n\n<!-- WRITE: one line on each middleware's role if evident from its name. -->\n"
    )


def _stub_frontend_map(docs, inv) -> str:
    return (
        _fm("VirtualDojo — Frontend Map",
            "Vue frontend wiring: views ↔ Pinia stores ↔ service action modules ↔ backend endpoints.",
            "wiki")
        + "# Frontend Map\n\n"
        + "<!-- WRITE: explain the wiring — views (pages) call action modules in "
          "frontend/src/services/* which hit backend endpoints via services/api.js; "
          "shared state lives in Pinia stores. Show how a UI symptom maps to a "
          "store → service → endpoint → backend service. Cite filenames; no code. -->\n\n"
        + "## Views (`frontend/src/views/`)\n\n" + ", ".join(inv.get("frontend-views", [])) + "\n\n"
        + "## Stores (`frontend/src/stores/`)\n\n" + ", ".join(inv.get("frontend-stores", [])) + "\n\n"
        + "## Service action modules (`frontend/src/services/`)\n\n" + ", ".join(inv.get("frontend-services", [])) + "\n"
    )


def _stub_data_model(docs, inv, mdl_by) -> str:
    out = [_fm("VirtualDojo — Data Model Overview",
               "The ~140 SQLAlchemy models grouped by domain, plus the dynamic-schema / custom-object system.",
               "wiki"),
           "# Data Model Overview\n",
           "<!-- WRITE: intro on the multi-tenant data model and the dynamic-schema / "
           "custom-object system (records-schema-objects domain). Then one line per "
           "domain on what its models represent. Cite app/models/*; no code. -->\n"]
    for slug, title, _m in DOMAINS:
        mdls = mdl_by.get(slug, [])
        if mdls:
            out.append(f"## {title}\n- {', '.join(mdls[:30])}\n")
    misc = mdl_by.get(MISC_DOMAIN[0], [])
    if misc:
        out.append(f"## {MISC_DOMAIN[1]}\n- {', '.join(misc[:30])}\n")
    return "\n".join(out)


def _stub_gotchas(docs, inv) -> str:
    return (
        _fm("VirtualDojo — Dev Gotchas & Invariants",
            "Non-obvious rules and pitfalls for the VirtualDojo backend: common mistakes, Alembic migration rules, multi-tenant isolation.",
            "wiki")
        + "# Dev Gotchas & Invariants\n\n"
        + "<!-- WRITE: distill the 'Common Mistakes to Avoid', 'Alembic Migration "
          "Rules', and 'System Architecture / Multi-Tenant Foundation' sections of "
          "the embedded CLAUDE.md below into durable invariants a troubleshooter "
          "must know (flush() vs commit(), Celery sessionmaker, asyncpg CAST, "
          "FastAPI dependency imports, tenant isolation, migration head chain). "
          "Frame as standing rules. Quote rules, not large code blocks. -->\n\n"
        + _embed_doc(docs, "claude")
    )


def _stub_symptom_index(docs, inv, svc_by) -> str:
    def paths(slug, n=2):
        return ", ".join(f"app/services/{s}" for s in svc_by.get(slug, [])[:n]) or "(see system-map)"
    rows = [
        ("Login / SSO / MFA fails", "auth-permissions-sharing", paths("auth-permissions-sharing"), "login-troubleshooting-guide"),
        ("Record not visible to a user", "auth-permissions-sharing", "app/services/permission_resolution_service.py, sharing_recalculation_service.py", ""),
        ("Quote totals / formulas wrong", "quoting-forecasts", paths("quoting-forecasts"), ""),
        ("Flow / trigger not firing", "flows", paths("flows"), "workflow-guide"),
        ("Tenant provisioning errors", "tenant-provisioning", paths("tenant-provisioning"), "tenant-provisioning-analysis"),
        ("Email not sending / inbound not parsed", "email", paths("email"), ""),
        ("Package install / licensing", "packages", paths("packages"), ""),
        ("Agent / AI chat failure", "agents", paths("agents"), ""),
        ("Deploy / production outage", "(infra)", "<!-- WRITE: from production-deployment-troubleshooting -->", "production-deployment-troubleshooting"),
    ]
    table = "\n".join(f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows)
    return (
        _fm("VirtualDojo — Symptom → Subsystem Index",
            "Map a reported symptom to the owning subsystem and the code paths to read first.",
            "troubleshooting")
        + "# Symptom → Subsystem\n\n"
        + "| Symptom area | Likely subsystem | Key code paths (read with repo_sync) | Relevant doc |\n"
        + "|---|---|---|---|\n" + table + "\n\n"
        + "<!-- WRITE: for each row add 1-2 common symptoms + likely-cause notes "
          "grounded in the linked docs / domain. Add rows for any domain in "
          "structure/ not covered. Frame as troubleshooting patterns and pointers, "
          "NOT guarantees of current behavior. -->\n"
    )


def _stub_doc_troubleshooting(slug_out, title, summary, docs, doc_slugs, extra="") -> str:
    body = _fm(title, summary, "troubleshooting") + f"# {title}\n\n"
    body += ("<!-- WRITE: turn the embedded doc(s) below into a troubleshooting "
             "playbook — common symptoms → likely causes → checks/resolution steps "
             "+ the code paths to read. Frame fixes as historical patterns, not "
             "current product guarantees. -->\n\n" + extra)
    for s in doc_slugs:
        body += "\n" + _embed_doc(docs, s)
    return body


def _build_stubs(docs: dict, inv: dict) -> int:
    svc_by = _assign(inv.get("services", []), "svc")
    mdl_by = _assign(inv.get("models", []), "mdl")
    stubs = {
        "system-map": _stub_system_map(docs, inv, svc_by, mdl_by),
        "backend-request-flow": _stub_backend_flow(docs, inv),
        "frontend-map": _stub_frontend_map(docs, inv),
        "data-model-overview": _stub_data_model(docs, inv, mdl_by),
        "dev-gotchas-and-invariants": _stub_gotchas(docs, inv),
        "symptom-to-subsystem": _stub_symptom_index(docs, inv, svc_by),
        "login-and-auth": _stub_doc_troubleshooting(
            "login-and-auth", "VirtualDojo — Login & Auth Troubleshooting",
            "Troubleshooting login / SSO / MFA issues and where the auth code lives.",
            docs, ["login-troubleshooting-guide"],
            extra="Owning subsystem: `auth-permissions-sharing` (see system-map).\n\n"),
        "tenant-provisioning": _stub_doc_troubleshooting(
            "tenant-provisioning", "VirtualDojo — Tenant Provisioning Troubleshooting",
            "Troubleshooting tenant provisioning and where the provisioning code lives.",
            docs, ["tenant-provisioning-analysis", "tenant-administration-plan"],
            extra="Owning subsystem: `tenant-provisioning` (see system-map).\n\n"),
        "deployment": _stub_doc_troubleshooting(
            "deployment", "VirtualDojo — Deployment Troubleshooting",
            "Troubleshooting production deploys and outages for the VirtualDojo service.",
            docs, ["production-deployment-troubleshooting", "workflow-guide"]),
    }
    for slug, body in stubs.items():
        storage.write_text(f"{STUBS_PREFIX}{slug}.md", body)
    return len(stubs)


def refresh_repo_knowledge(force: bool = False) -> dict:
    """Refresh engineering/raw/ from the repo. Gated on main HEAD sha (the
    "any PR merged to main?" check). Returns content-free stats."""
    repo = _github().get_repo(REPO)
    head = repo.get_branch(REF).commit.sha
    last = (storage.read_text(STATE_PATH) or "").strip()
    if head == last and not force:
        return {"source": "repo", "skipped": "no-merges", "head": head}

    docs, redacted = _fetch_docs(repo)
    inv = _collect_inventory(repo)
    struct_written = _write_structure(inv)
    stubs_written = _build_stubs(docs, inv)
    storage.write_text(STATE_PATH, head, content_type="text/plain")

    return {
        "source": "repo",
        "head": head,
        "docs_written": len(docs),
        "structure_written": struct_written,
        "stubs_written": stubs_written,
        "secrets_redacted": redacted,
    }
