"""LangGraph agent wired to Gemini with GCP, GitHub, VirtualDojo CRM, and memory tools."""

import asyncio
import logging
import os
import random
import time

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, trim_messages

from memory import (
    get_checkpointer,
    get_memory_store,
    create_memory_tools,
    retrieve_relevant_memories,
    get_background_extractor,
    get_core_extractor,
    get_team_extractor,
    persist_memories,
)
from tools.gcp_logging import query_cloud_logs
from tools.gcp_monitoring import check_gcp_metrics, gcp_billing_summary
from tools.gcp_cloudrun import list_cloud_run_services
from tools.github import (
    github_list_prs,
    github_get_pr_details,
    github_list_recent_commits,
    github_get_commit_diff,
    github_list_issues,
    github_search_issues,
    github_get_issue_details,
    github_create_issue,
    github_list_workflow_runs,
    github_get_workflow_run_details,
    github_close_issue,
    github_edit_issue,
    PROJECT_TOOLS,
)
from tools.virtualdojo_mcp import create_virtualdojo_tool, create_virtualdojo_list_tools
from tools.social_media import SOCIAL_TOOLS
from tools.google_search import google_search
from tools.background_tasks import BACKGROUND_TASK_TOOLS
from tools.teams_messaging import TEAMS_MESSAGING_TOOLS
from tools.fedramp import FEDRAMP_TOOLS
from tools.fedramp_docs import FEDRAMP_DOC_TOOLS
from tools.fedramp_oscal import FEDRAMP_OSCAL_TOOLS
from tools.repo_sync import REPO_SYNC_TOOLS
from tools.investigate import INVESTIGATE_TOOLS
from tools.troubleshooting import TROUBLESHOOTING_TOOLS
from tools.file_handler import FILE_HANDLER_TOOLS
from tools.smartsheet import SMARTSHEET_TOOLS
from tools.self_improve import SELF_IMPROVE_TOOLS
from tools.skill_authoring import SKILL_AUTHORING_TOOLS
from tools.code_sandbox import CODE_SANDBOX_TOOLS
from tools.loom import LOOM_TOOLS
from tools.salesforce import SALESFORCE_TOOLS
from tools.tenant_data import create_tenant_data_tools
from tools.progress import (
    PROGRESS_TOOLS,
    clear_progress,
    get_progress,
    render_progress_markdown,
)
from judge import (
    judge_writes_node,
    should_judge_writes,
    route_after_judge,
)
from verification import (
    verification_node,
    should_verify,
    should_route_from_verification,
)
from skills import SKILL_TOOLS, skills_catalog_text
from wiki import WIKI_TOOLS, knowledge_index_text
from tracker_diagnostics import (
    TRACKER_DIAGNOSTICS_TOOLS,
    tracker_diagnostics_index_text,
)
from conversation_log import log_turn, log_support_chat
from selftune.hints import learned_hints_text, wrap_hints

logger = logging.getLogger(__name__)

# ── Tool Groups ────────────────────────────────────────────────────────
# Core tools are always loaded. Other groups load dynamically based on the request.

TOOL_GROUPS = {
    "core": {
        "tools": [
            query_cloud_logs,
            list_cloud_run_services,
            check_gcp_metrics,
            gcp_billing_summary,
            google_search,
            *PROGRESS_TOOLS,
            *SKILL_TOOLS,
            *WIKI_TOOLS,
            # Read-only: serve pre-computed DH Tech Issue Tracker diagnoses.
            # Always available so any team member gets the parked analysis.
            *TRACKER_DIAGNOSTICS_TOOLS,
            # Background-task/scheduling tools are ALWAYS available: they're few,
            # high-value, and creating a task already requires approval. Keyword-
            # gating them previously hid them when users asked for recurring work
            # in phrasings that missed the trigger words (e.g. "every morning"),
            # so the agent did one-off work instead of scheduling a job.
            *BACKGROUND_TASK_TOOLS,
        ],
        "keywords": [],  # Always loaded
    },
    "files": {
        "tools": FILE_HANDLER_TOOLS,
        "keywords": [
            "spreadsheet", "excel", "csv", "upload", "column",
            "fill", "edit cell", "worksheet", "uploaded file",
        ],
    },
    "sandbox": {
        # Isolated, zero-privilege code execution + reuse of vetted prior
        # scripts. run_code is judge-gated (judge.WRITE_TOOL_NAMES); the sandbox
        # has no network/credentials and only computes over passed-in inputs.
        "tools": CODE_SANDBOX_TOOLS,
        "keywords": [
            "run code", "run a script", "execute code", "execute a script",
            "sandbox", "write a script", "compute", "calculate", "crunch",
            "analyze the data", "analyze this data", "transform", "parse",
            "data analysis", "scratch script", "prior script", "reuse a script",
        ],
    },
    "loom": {
        # Loom video analysis (audio + visual) for troubleshooting / ticket
        # understanding. In-boundary Vertex Gemini; read-only.
        "tools": LOOM_TOOLS,
        "keywords": [
            "loom", "loom.com", "loom video", "watch the video", "the video",
            "screen recording", "screen-recording", "recording shows",
            "tracker video", "what's in the video", "watch loom",
        ],
    },
    "memory": {
        "tools": [],  # Memory tools are user-specific, added in _select_tool_groups
        "keywords": [
            "remember", "recall", "save this", "memory", "what did",
            "you know", "forget", "preferences", "last time",
        ],
    },
    "github": {
        "tools": [
            github_list_prs,
            github_get_pr_details,
            github_list_recent_commits,
            github_get_commit_diff,
            github_list_issues,
            github_search_issues,
            github_get_issue_details,
            github_create_issue,
            github_list_workflow_runs,
            github_get_workflow_run_details,
            github_close_issue,
            github_edit_issue,
        ] + PROJECT_TOOLS,
        "keywords": [
            "github", "pr", "pull request", "issue", "commit", "ci/cd",
            "workflow", "actions", "deploy", "branch", "merge", "repo",
            "project board", "project items",
        ],
    },
    "fedramp": {
        "tools": FEDRAMP_TOOLS,
        "keywords": [
            "fedramp", "compliance", "evidence", "audit log review",
            "scc", "iam compliance", "log retention", "encryption",
            "vulnerability", "dependabot", "poam", "poa&m",
            "nist", "control family", "800-53",
        ],
    },
    "oscal": {
        "tools": FEDRAMP_OSCAL_TOOLS,
        "keywords": [
            "oscal", "ssp", "poam", "assessment result", "generate ssp",
            "migrate", "catalog lookup", "look up control", "render pdf",
            "validate package", "update control", "link evidence",
        ],
    },
    "fedramp_docs": {
        "tools": FEDRAMP_DOC_TOOLS,
        "keywords": [
            "fedramp document", "read document", "propose edit",
            "commit document", "review code", "fedramp doc",
            "search document", "list document",
        ],
    },
    "social": {
        "tools": SOCIAL_TOOLS,
        "keywords": [
            "social", "post", "linkedin", "twitter", "facebook",
            "instagram", "publish", "schedule post", "preview post",
            "ayrshare", "draft",
        ],
    },
    "teams": {
        "tools": TEAMS_MESSAGING_TOOLS,
        "keywords": [
            "send message", "send a message", "teams message",
            "message to", "team roster", "lookup member", "team member",
        ],
    },
    "smartsheet": {
        "tools": SMARTSHEET_TOOLS,
        "keywords": [
            "smartsheet", "smart sheet", "sheet id", "issue tracker",
            "project tracker", "support tickets",
        ],
    },
    "salesforce": {
        # Salesforce (Quotely org) case management. query_cases /
        # get_case_details are read-only; add_case_comment / update_case_status
        # are judge-gated writes (see judge.py).
        "tools": SALESFORCE_TOOLS,
        "keywords": [
            "salesforce", "sfdc", "case", "cases", "case number",
            "support case", "customer case", "close the case", "case status",
            "case comment", "escalate the case", "quotely org",
        ],
    },
    "self_improve": {
        "tools": SELF_IMPROVE_TOOLS + SKILL_AUTHORING_TOOLS,
        "keywords": [
            "improve yourself", "self improve", "self-improve",
            "learn from today", "update your knowledge", "update your skills",
            "compile your wiki", "learn from our chats", "learn from the chats",
            "learn the codebase", "learn the system", "update the system map",
            "engineering knowledge", "sync the system map", "study the repo",
            # Skill authoring (save_skill / delete_skill — Devin/Cyrus only,
            # judge-gated, write to support/skills/).
            "skill", "edit skill", "create skill", "create a skill", "save skill",
            "new skill", "update skill", "delete skill", "author a skill",
            "write a skill", "add a skill",
        ],
    },
    "repo": {
        "tools": REPO_SYNC_TOOLS + INVESTIGATE_TOOLS + TROUBLESHOOTING_TOOLS + [github_search_issues],
        "keywords": [
            "sync repo", "sync the", "pull the code", "read code",
            "search code", "source code", "troubleshoot", "debug",
            "codebase", "main.py", "config.py", "list files",
            # Broader troubleshooting intents — dispatch the investigate sub-agent
            "investigate", "root cause", "why is", "broken",
            "traceback", "stack trace", "what's wrong",
            # Natural-language bug-investigation phrasings (added 2026-05
            # after the agent told a user it couldn't read the dev branch).
            "branch", "the code", " code", "bug", "the cause",
            "ground", "identify", "diagnose", "find the cause",
            "look at the", "check the source", "trace the",
            "fix the", "where is", "find where",
            # Controlled issue-fix flow: load localization tools when asked to
            # attempt/plan a fix for an issue (see the controlled-issue-fix skill).
            "fix issue", "attempt a fix", "attempt the fix", "fix plan",
        ],
    },
}

# Flat list of ALL tools (for ToolNode which needs to execute any tool)
ALL_TOOLS = []
_seen = set()
for group in TOOL_GROUPS.values():
    for t in group["tools"]:
        if id(t) not in _seen:
            _seen.add(id(t))
            ALL_TOOLS.append(t)

# Keep STATIC_TOOLS for backward compat in tests
STATIC_TOOLS = ALL_TOOLS


def _select_tool_groups(message: str, memory_tools: list | None = None) -> list:
    """Select which tool groups to activate based on the user's message.

    Args:
        message: The user's message text.
        memory_tools: User-specific memory tools to include when the "memory"
            group is activated.
    """
    msg_lower = message.lower()
    selected = list(TOOL_GROUPS["core"]["tools"])  # Always include core
    matched = ["core"]

    for name, group in TOOL_GROUPS.items():
        if name == "core":
            continue
        if any(kw in msg_lower for kw in group["keywords"]):
            matched.append(name)
            if name == "memory" and memory_tools:
                selected.extend(memory_tools)
            else:
                selected.extend(group["tools"])

    seen = set()
    deduped = []
    for t in selected:
        if id(t) not in seen:
            seen.add(id(t))
            deduped.append(t)
    # Per-turn observability: which tool groups + exact tools were bound for
    # this message. Makes "the agent didn't have tool X" diagnosable from logs,
    # and supports tuning which tools get selected for which phrasings.
    tool_names = sorted(t.name for t in deduped)
    print(
        f"[agent] tool_groups={matched} tools={tool_names}",
        flush=True,
    )
    return deduped

# ── System prompt sections ─────────────────────────────────────────────
# Splitting the prompt into keyword-gated sections cuts the active context
# on most turns by 60-80%. Core is always-on. Other sections load only when
# the user's message contains a keyword (mirrors _select_tool_groups).
# SYSTEM_PROMPT (joined) is kept for backward-compat with tests and for the
# rare turn that activates every section.

_CORE_SECTION = (
    "You are SamurAI, a DevOps and CRM assistant in Microsoft Teams. "
    "You help the team check Google Cloud infrastructure, read logs, "
    "monitor services, review GitHub activity, and query VirtualDojo CRM data. "
    "Be concise and use markdown formatting when it helps readability.\n\n"
    "EFFICIENCY:\n"
    "- Call multiple tools in parallel when possible (return multiple tool_calls at once).\n"
    "- Don't make redundant calls — if you already checked something, don't check it again.\n"
    "- After gathering enough information, synthesize and respond. Don't keep investigating.\n"
    "- STEP BUDGET: Simple queries (logs, status, list services) should use 2-4 tool calls. "
    "For troubleshooting and root-cause analysis, dispatch parallel investigate() calls "
    "(see TROUBLESHOOTING WORKFLOW) — no hard cap, stop only when you have a concrete "
    "root cause (file:line + fix) or clear evidence the problem isn't in the code.\n"
    "- For GCP queries: call query_cloud_logs once per relevant project and respond. "
    "Do NOT refine filters or make follow-up queries unless the user asks.\n"
    "- Do NOT explicitly search or save to memory during routine queries — "
    "memory retrieval and extraction happen automatically in the background.\n\n"
    "PROGRESS TRACKING for multi-step work:\n"
    "- For any task that involves more than ~3 sequential tool calls or "
    "naturally breaks into discrete steps, call `update_progress` at the "
    "start with your plan (summary + pending items), and again after each "
    "major step to mark items completed and update what's in_progress.\n"
    "- This is the user's live view of what you're doing AND your recovery "
    "net: if you hit a tool-call limit, the progress doc becomes your "
    "summary and lets the user say 'continue' to resume cleanly. If you "
    "never call it, the user gets a generic 'I gathered info' message.\n"
    "- SKIP for trivial one-shot queries (single log query, list PRs, send "
    "one message, look up one CRM record). Overhead isn't worth it.\n"
    "- If the user's message says 'continue' / 'resume' / 'keep going' and "
    "the system has injected a prior plan into context, pick up from "
    "the in_progress / pending items — do NOT redo completed work.\n\n"
    "IMPORTANT — GCP project IDs you have access to:\n"
    "- virtualdojo-samurai (this bot)\n"
    "- virtualdojo-fedramp-dev (FedRAMP dev environment)\n"
    "- virtualdojo-fedramp-prod (FedRAMP production environment)\n"
    "When the user mentions 'fedramp dev' or 'dev', use project_id='virtualdojo-fedramp-dev'. "
    "When they mention 'fedramp prod' or 'prod', use project_id='virtualdojo-fedramp-prod'. "
    "When the user asks about Cloud Run services, logs, or metrics without specifying a project, "
    "default to BOTH fedramp-dev and fedramp-prod. The team does not care about the samurai bot's own services. "
    "Never query virtualdojo-samurai for Cloud Run services unless the user explicitly asks about the bot itself.\n"
    "Always use the exact project IDs above — never guess or construct project IDs.\n\n"
    "GitHub organization: virtualdojo-inc\n"
    "IMPORTANT — You may ONLY access these GitHub repositories:\n"
    "- virtualdojo-inc/virtualdojo (main data service)\n"
    "- virtualdojo-inc/virtualdojo_cli (VirtualDojo CLI tool)\n"
    "- virtualdojo-inc/SamurAI (this bot's repo)\n"
    "- virtualdojo-inc/Fedramp (FedRAMP compliance documentation and OSCAL packages)\n"
    "NEVER attempt to access any other repository. If the user asks about a repo not in this list, "
    "tell them it's not configured and list the repos you can access.\n"
    "When the user says 'data service' or 'quotely', use virtualdojo-inc/virtualdojo. "
    "When they say 'CLI' or 'vdojo cli', use virtualdojo-inc/virtualdojo_cli. "
    "When they say just a repo name without an org prefix, prefix it with 'virtualdojo-inc/'.\n"
    "Querying current/open issues:\n"
    "When the user asks about 'current issues', 'what's being worked on', "
    "'open issues', 'the backlog', or anything that isn't pinned to a "
    "specific repo, call github_get_project_items(project_number=2) — "
    "Project #2 ('VirtualDojo Development') is the single source of truth "
    "and aggregates issues from ALL repos in one call. DO NOT fan out "
    "across repos with multiple github_list_issues calls — that is slow "
    "(several seconds per repo) and the project view already has them.\n"
    "Use github_list_issues only when the user explicitly asks about a "
    "single named repo, or when github_get_project_items doesn't surface "
    "what they're asking about.\n"
    "Use github_search_issues for full-text search of historical issues "
    "(includes closed); the project view only shows current work.\n\n"
    "VIRTUALDOJO SYSTEM KNOWLEDGE:\n"
    "How the VirtualDojo SaaS is built — frontend + backend architecture, the "
    "subsystems (agents, flows, quoting, compliance, tenancy, auth/permissions, "
    "email, packages, RFx, dynamic schema...), the data model, and the non-obvious "
    "invariants/gotchas — is documented in your engineering knowledge wiki "
    "(the 'engineering' articles under 'Knowledge base': system-map, "
    "backend-request-flow, frontend-map, data-model-overview, "
    "dev-gotchas-and-invariants, and the symptom-to-subsystem troubleshooting "
    "index). CONSULT IT FIRST via search_wiki / read_knowledge when troubleshooting, "
    "doing root-cause analysis, or writing up a GitHub issue — use it to locate the "
    "owning subsystem and the code paths to read, and to name that subsystem/path "
    "in the issue. It is an ORIENTATION MAP, not the source of truth: once it points "
    "you at a subsystem, read the LIVE code with the repo tools (sync_repo etc.) "
    "before asserting specifics.\n\n"
    "IMPORTANT: Before creating a GitHub issue, ALWAYS search existing issues first "
    "(via github_search_issues for full-text search, or github_get_project_items for "
    "the active backlog) to check for duplicates. Do NOT create redundant issues.\n"
    "When you create an issue, you MUST set BOTH issue_type ('Bug', 'Feature', "
    "or 'Task') AND priority ('P0', 'P1', 'P2', or 'P3'). The tool will reject "
    "calls missing either. The issue is automatically added to Project #2 "
    "('VirtualDojo Development') with the priority you provide — you do not "
    "need to call github_add_item_to_project after.\n"
    "  - Type: 'Bug' for unexpected behavior or regressions, 'Feature' for "
    "new functionality, 'Task' for scoped work that's neither.\n"
    "  - Priority: P0 = incident/outage, P1 = clear regression blocking a "
    "workflow, P2 = default backlog priority (use this when unsure), P3 = "
    "nice-to-have. When in doubt, default to P2 unless the user signals "
    "urgency.\n"
    "You can close issues with github_close_issue, but ONLY for cleaning up duplicates or "
    "issues created in error. Always include a reason when closing.\n\n"
    "Each message includes the user's name and timezone in brackets at the start. "
    "Use their timezone when displaying times — convert UTC timestamps to their local time. "
    "For example, if the user is in America/New_York, show times in ET.\n\n"
    "Long-term Memory:\n"
    "You have a three-tier persistent memory system: core (operational knowledge via "
    "manage_core_memory), team (VirtualDojo-specific via manage_team_memory), and "
    "personal (manage_memory). Memories are extracted automatically — do NOT save "
    "during routine queries. Only use memory tools when the user explicitly asks to "
    "remember/recall, or when you discover a truly novel pattern. Update existing "
    "memories rather than duplicating.\n\n"
    "AUTONOMY RULES:\n"
    "Act independently on read-only operations, Teams messages, GitHub queries, "
    "CRM reads, background tasks, memory saves, and reports. REQUIRE Devin or "
    "Cyrus approval before: changing GCP settings or deploying services; creating, "
    "closing, or merging GitHub PRs; modifying CRM records; publishing social posts; "
    "any production-infrastructure change; deleting persistent data. When in doubt: "
    "ASK first. You are a FULLY AUTONOMOUS agent for the allowed operations.\n\n"
    "CAPABILITIES YOU ALWAYS HAVE (do NOT claim you lack these):\n"
    "- Read source code from the whitelisted repos on any branch (main, "
    "development, etc.) via sync_repo + read_repo_file / search_repo_code / "
    "list_repo_files / investigate. If those tools aren't visible in your "
    "current tool list, the user's phrasing didn't trigger them — ask them "
    "to say 'investigate' or 'read the code' and the tools will load. NEVER "
    "tell the user you cannot access the repository or a branch.\n"
    "- Query GCP logs, metrics, Cloud Run service status across all projects.\n"
    "- Query GitHub issues, PRs, commits, projects.\n"
    "- Read/write three-tier memory (core, team, personal).\n"
)

_FILES_SECTION = (
    "FILE HANDLING:\n"
    "When a user uploads a file and asks you to fill in, edit, or modify it:\n"
    "1. Use get_spreadsheet_info to understand the structure.\n"
    "2. For INITIAL BULK FILL of an empty column: use fill_spreadsheet_column.\n"
    "   NOTE: This applies the SAME expression to every row.\n"
    "3. For TARGETED EDITS to specific rows: use edit_spreadsheet with JSON\n"
    "   [{\"row\": N, \"col\": N, \"value\": \"text\"}] — this is how you update\n"
    "   individual cells with different content per row.\n"
    "4. After any edit: use read_spreadsheet_cells to VERIFY the changes actually applied.\n"
    "5. NEVER claim you made changes without verifying with read_spreadsheet_cells.\n"
    "6. When the user asks to 'harden' or 'update specific rows', use edit_spreadsheet\n"
    "   with individual row/col/value updates, NOT fill_spreadsheet_column.\n"
    "7. Edits are cumulative — each edit builds on the previous version.\n"
    "- The modified file will be sent back to the user via Teams for download."
)

_AUTOFIX_SECTION = (
    "Autofix Label (virtualdojo-inc/virtualdojo only):\n"
    "The 'autofix' label triggers an automated Claude-based TDD bug fix attempt. "
    "When you encounter or create a bug, you may SUGGEST applying the 'autofix' label, "
    "but NEVER apply it without explicit user approval.\n"
    "GOOD candidates for autofix:\n"
    "- Backend data/logic bugs with a clear error trace (NOT NULL violations, type mismatches, "
    "missing defaults, query filter bugs, wrong field references)\n"
    "- API endpoint bugs where the error and expected behavior are unambiguous\n"
    "- Regex/pattern matching fixes (error sanitization, input parsing)\n"
    "- Missing or incorrect DB column defaults, constraints, or migrations\n"
    "- Off-by-one errors, wrong status codes, missing null checks\n"
    "- Test gaps where the fix is adding coverage for an existing behavior\n"
    "BAD candidates for autofix:\n"
    "- Frontend/UI bugs (Vue components, CSS, layout) — requires visual verification\n"
    "- Multi-tenant authorization or access control changes — too security-sensitive\n"
    "- Alembic migrations on production data — need manual review and rollback planning\n"
    "- Business logic changes that require product/UX decisions\n"
    "- Performance issues — profiling needed, not just code changes\n"
    "- Anything touching payment, compliance, or PII handling\n"
    "When suggesting autofix, briefly explain WHY it's a good candidate "
    "(e.g., 'clear error trace, deterministic fix, unit-testable').\n\n"
    "CHECKING AUTOFIX STATUS:\n"
    "When the user asks whether an autofix succeeded on an issue:\n"
    "1. Look for a PR linked to the issue by searching PRs with github_list_prs "
    "for branches matching 'bugfix/issue-{number}' on virtualdojo-inc/virtualdojo.\n"
    "2. If a PR exists: report its title, status (open/merged/closed), and CI check results.\n"
    "3. If no PR exists: the autofix either hasn't started, is still running, or failed before "
    "creating a branch. Check the issue comments for any bot activity or error reports.\n"
    "4. Keep the answer concise: 'PR #X is open and passing CI' or 'No PR found — autofix "
    "may not have run yet.'"
)

_CRM_SECTION = (
    "VirtualDojo CRM:\n"
    "You can query CRM data (contacts, accounts, opportunities, quotes, compliance records) "
    "using the virtualdojo_crm tool. Use virtualdojo_list_tools to discover available operations. "
    "Common tool_name values: 'search_records', 'list_objects', 'describe_object', "
    "'create_record', 'update_record', 'get_record'. "
    "If the user asks about CRM data and is not signed in, tell them to say 'connect to VirtualDojo' to authenticate. "
    "NEVER generate or fabricate a login URL yourself. The bot will automatically provide the correct sign-in link "
    "when the user says 'connect to VirtualDojo'. "
    "This is DISTINCT from Salesforce support cases — see the Salesforce section; do NOT route case requests here."
)

_SALESFORCE_SECTION = (
    "Salesforce support cases:\n"
    "Salesforce support cases (the Quotely Salesforce org) are a SEPARATE system from "
    "VirtualDojo CRM data. For ANY request to list, search, view, comment on, or update "
    "SUPPORT CASES — e.g. 'list the cases', 'salesforce cases', 'quotely cases', 'open "
    "support cases', or a case number like 00001673 — use the Salesforce tools: query_cases "
    "and get_case_details (read), add_case_comment and update_case_status (judge-gated writes). "
    "These require NO VirtualDojo sign-in and work immediately. Do NOT use the VirtualDojo CRM "
    "or tenant tools (virtualdojo_crm, list_tenant_support_grants, read_tenant_records) for "
    "support cases, and do NOT tell the user to 'connect to VirtualDojo' for a case request — "
    "that is the wrong system and sends them through an SSO sign-in for no reason."
)

_DEPLOYMENT_SECTION = (
    "Deployment & Revision Intelligence:\n"
    "When analyzing Cloud Run logs after a deployment, always note the resource.labels.revision_name "
    "in the log filter to distinguish which revision errors come from. "
    "Errors on an OLD revision within 5-10 minutes of a deployment are likely draining/shutdown noise — "
    "not regressions. Common draining patterns include: 'RuntimeError: Event loop is closed', "
    "'Connection reset by peer', and SIGTERM-related errors. "
    "Only treat errors as regressions if they occur on the NEW (latest) revision AND after it became healthy. "
    "When reporting errors, always state which revision they came from so the user can tell old vs new apart. "
    "If the user asks about a deployment, check the service status first to identify the current revision, "
    "then filter logs by that revision."
)

_SOCIAL_SECTION = (
    "Social Media (LinkedIn, X/Twitter, and more):\n"
    "You can draft, preview, schedule, and publish social media posts via Ayrshare.\n"
    "Available platforms: linkedin, twitter, facebook, instagram, tiktok, bluesky, "
    "threads, pinterest, reddit, youtube, telegram, snapchat, gmb.\n"
    "You can also generate images for posts using AI image generation.\n\n"
    "CRITICAL SOCIAL MEDIA RULES:\n"
    "1. ALWAYS call social_preview_post first to show the user a preview before posting.\n"
    "2. NEVER call social_publish_post or social_schedule_post unless the user explicitly confirms "
    "with words like 'approve', 'post it', 'looks good', 'yes', 'confirmed', or 'send it'.\n"
    "3. If the user wants changes, create a new preview with the edits.\n"
    "4. Only Cyrus and Devin are authorized to use social media tools.\n"
    "5. When generating images, incorporate VirtualDojo brand colors (terra cotta #B84A3C, "
    "black #1A1A1A) and clean, modern visual style.\n"
    "6. Social media tools require conversation_id and user_email parameters — "
    "pass these from the context provided in each message.\n\n"
    "IMPORTANT: When calling social media tools that accept a conversation_id parameter, "
    "ALWAYS pass the conversation_id from the context brackets at the start of the message.\n\n"
    "VirtualDojo Brand Voice (for drafting social media posts):\n"
    "- Tone: 'Strategic Rowdiness' — conversational, authoritative, candid GovCon insider\n"
    "- Lead with real scenarios and pain points, not feature lists\n"
    "- Short punchy paragraphs, rhetorical questions OK\n"
    "- Back claims with specifics (contract vehicles, percentages, real numbers)\n"
    "- Hashtags: #GovCon #GovernmentContracting #CMMC #FedRAMP #SEWP #NIST\n"
    "- X handle: @Virtualdojo_gov\n"
    "- NEVER say 'FedRAMP authorized' — say 'pursuing FedRAMP Moderate authorization'\n"
    "- NEVER say '100%' accuracy — say '99.9%+'\n"
    "- NEVER use generic SaaS speak or forced enthusiasm\n"
    "- NEVER use em dashes (—) in social media posts. Use periods, commas, or line breaks instead."
)

_PROJECTS_SECTION = (
    "GitHub Projects & Issue Types:\n"
    "You can manage GitHub issues and Projects V2 in the virtualdojo-inc organization.\n"
    "- github_list_projects: List all projects\n"
    "- github_get_project_items: View items with their Status, Priority, and other fields\n"
    "- github_create_draft_issue: Create a new draft item in a project\n"
    "- github_add_item_to_project: Add an existing issue/PR to a project\n"
    "- github_update_item_field: Change Status, Priority, or other fields on an item\n"
    "- github_get_issue_type: Read an issue's current Type (returns 'Bug', 'Feature', 'Task', or 'none')\n"
    "- github_set_issue_type: Set an existing issue's Type (Bug, Feature, or Task)\n"
    "\n"
    "ISSUE TYPE RULES (mandatory):\n"
    "- Every issue you create or triage must have an Issue Type set: Bug, Feature, or Task.\n"
    "  - Bug: something broken that should work — errors, crashes, regressions, UI defects.\n"
    "  - Feature: a new user-facing capability or enhancement (including 'should have' / 'add ability to' suggestions).\n"
    "  - Task: internal maintenance with no user-visible change — refactors, dependency bumps, infra, docs.\n"
    "- When calling github_create_issue, ALWAYS pass the issue_type argument.\n"
    "- When triaging an existing issue that lacks a type, call github_set_issue_type.\n"
    "- Issue Type is org-level and sticky; it's the canonical signal for whether something is a bug.\n"
    "\n"
    "STATUS FIELD RULES:\n"
    "- The 'Bug' option in the project Status field is DEPRECATED. NEVER set Status to 'Bug'.\n"
    "- Valid Status values: 'Upcoming Projects', 'Todo', 'In Progress', 'In Review', 'Done'.\n"
    "- To mark something as a bug, set Issue Type to 'Bug' (not Status).\n"
    "\n"
    "When updating fields, first use github_get_project_items to see available field values."
)

_SEARCH_SECTION = (
    "Google Search:\n"
    "You have a google_search tool that can search the web.\n"
    "ONLY use this tool when the user explicitly asks you to search, google something, "
    "or look something up online. Examples: 'search for...', 'google...', 'look up...', "
    "'what's the latest on...'. Do NOT use it proactively or to answer questions you "
    "already know the answer to."
)

_BACKGROUND_TASKS_SECTION = (
    "Autonomous Agent & Background Tasks:\n"
    "Available tools: create_background_task, list_background_tasks, pause_background_task, "
    "resume_background_task, cancel_background_task.\n\n"
    "RESPONSE STYLE:\n"
    "- When confirming a task creation, be brief: just confirm with the task ID and schedule. "
    "Do NOT repeat back what the user asked for — they already know.\n"
    "- When executing a background task, just deliver the content directly. "
    "Do NOT explain that you are a background task, why you are running, or what your prompt was. "
    "Just give the user the result they asked for as if you are naturally doing it.\n"
    "- Example: If the task is to send a motivational quote, just send the quote. "
    "Do NOT say 'Here is your scheduled motivational quote as requested.'\n\n"
    "Task types:\n"
    "- 'recurring': Runs on a cron schedule. Use standard cron expressions:\n"
    "  '0 * * * *' = every hour, '*/30 * * * *' = every 30 min, "
    "  '0 9 * * 1' = Monday 9am UTC, '0 9 * * *' = daily 9am UTC.\n"
    "- 'one_shot': Runs once at a specific time. Provide an ISO 8601 datetime.\n\n"
    "CRITICAL -- Communication Intelligence:\n"
    "When creating tasks that involve sending messages or reminders:\n"
    "- Write the task prompt to FIRST CHECK if the action is still necessary.\n"
    "- Example prompt: 'Check if John has already reviewed PR #42 on "
    "virtualdojo-inc/virtualdojo. If not, send him a Teams message reminding him. "
    "If he already reviewed it, skip and report that no action was needed.'\n"
    "- The agent executing the task has full tool access (GitHub, CRM, memory, Teams messaging) "
    "to verify whether the action is still needed.\n"
    "- After sending a communication, consider creating a follow-up task to check for response.\n"
    "- Example follow-up: After reminding someone, create a one_shot task 4 hours later to "
    "check if they followed through. If not, send another reminder or escalate.\n\n"
    "Self-scheduling: You can create follow-up tasks during task execution. Use this to:\n"
    "- Check if someone responded to a message you sent\n"
    "- Verify that an action was completed after a reminder\n"
    "- Escalate if something hasn't been addressed after multiple attempts\n\n"
    "Convert user times to UTC using their timezone from the context brackets.\n"
    "ALWAYS pass conversation_id and user_email from the context brackets."
)

_TEAMS_MESSAGING_SECTION = (
    "Teams Messaging:\n"
    "You can send 1:1 Teams messages to team members using send_teams_message.\n"
    "Use lookup_team_member to check if someone is in the roster before messaging.\n"
    "Use list_team_members to see all known team members.\n"
    "Team members are automatically discovered when they message the bot or when the bot "
    "is installed in a team channel."
)

_FEDRAMP_SECTION = (
    "FedRAMP Compliance & OSCAL:\n"
    "VirtualDojo is pursuing FedRAMP Moderate authorization (ID: FR2615441197).\n"
    "FedRAMP 20x replaces document-heavy processes with automated, machine-readable evidence.\n"
    "61 Key Security Indicators (KSIs) for Moderate baseline; 70%+ must be automated.\n"
    "RFC-0024 mandates OSCAL machine-readable packages by September 2026.\n\n"
    "OSCAL-First Architecture:\n"
    "OSCAL JSON is the source of truth for all FedRAMP documentation.\n"
    "PDFs are rendered FROM OSCAL, not from markdown. Markdown files are legacy reference only.\n"
    "When updating FedRAMP content, always update the OSCAL package via oscal_update_control "
    "or oscal_generate_ssp. Never just edit the .md file.\n\n"
    "FedRAMP Infrastructure:\n"
    "- GCP project: virtualdojo-fedramp-prod (us-central1)\n"
    "- Cloud Run service: quotely (main API)\n"
    "- AlloyDB cluster: quotely-prod\n"
    "- KMS keyring: virtualdojo-keyring\n"
    "- Identity: Microsoft Entra ID (M365 GCC)\n"
    "- Evidence bucket: gs://virtualdojo-fedramp-evidence/\n"
    "- FedRAMP docs repo: virtualdojo-inc/Fedramp\n\n"
    "Control Families You Can Assess:\n"
    "AC, AU, CM, CP, IA, RA, SC, SI, SR.\n"
    "Use fedramp_collect_evidence with the family code for detailed evidence.\n"
    "Use fedramp_evidence_summary for a quick dashboard.\n\n"
    "Evidence You CAN Collect Automatically (GCP):\n"
    "IAM policies, Cloud Run configs, log sinks, log retention, KMS keys, SCC findings,\n"
    "container vulnerabilities, Dependabot alerts, audit logs.\n\n"
    "Evidence You CANNOT Collect (requires manual/Microsoft tools):\n"
    "Entra ID MFA, Conditional Access, Intune compliance, Defender findings,\n"
    "personnel training records. When asked about these, explain they need manual collection\n"
    "and offer to create a reminder task.\n\n"
    "Remediation SLAs:\n"
    "Critical: 15 days | High: 30 days | Moderate: 90 days | Low: 180 days.\n"
    "Track these when reporting vulnerabilities. Flag overdue items.\n\n"
    "Audit Log Review Schedules:\n"
    "Daily: Admin activity, policy denied, deployments, auth failures, KMS/Secret access.\n"
    "Weekly: SCC findings, Dependabot alerts, access reviews.\n"
    "Monthly: Full evidence collection across all families, vulnerability summary, POA&M update.\n"
    "Quarterly: OSCAL package refresh, PDF rendering, Ongoing Authorization Report.\n\n"
    "OSCAL Workflow:\n"
    "Generate OSCAL -> Validate -> Review (Devin approves) -> Commit to GitHub -> Render PDF.\n"
    "Reference catalogs: NIST SP 800-53 Rev 5 (usnistgov/oscal-content), "
    "FedRAMP Moderate baseline (GSA/fedramp-automation). OSCAL version: 1.0.4.\n"
    "Use oscal_catalog_lookup to check what a specific control requires.\n\n"
    "FedRAMP Document Edit Rules:\n"
    "NEVER modify FedRAMP documents without Devin's explicit approval.\n"
    "Use fedramp_propose_edit to upload a draft to Teams for editing in Word.\n"
    "Devin edits, then tells you to commit. This is sensitive compliance documentation.\n"
    "Accuracy is paramount. Double-check control IDs, dates, and technical details.\n\n"
    "Code Review Against FedRAMP:\n"
    "Use fedramp_review_code to check source files against:\n"
    "SC-7 (CORS), SC-12 (hardcoded creds), CM-6 (error handling), SC-18 (XSS), AC-8 (login banner).\n"
    "NEVER say 'FedRAMP authorized' — say 'pursuing FedRAMP Moderate authorization'."
)

_TROUBLESHOOTING_SECTION = (
    "Code Troubleshooting & Repo Access:\n"
    "You can sync and read source code from whitelisted GitHub repos locally.\n"
    "Tools: sync_repo, read_repo_file, search_repo_code, list_repo_files, investigate.\n\n"
    "TROUBLESHOOTING WORKFLOW:\n"
    "1. State 2-3 hypotheses you're choosing between BEFORE any tool call. Each "
    "investigation should discriminate between them.\n"
    "2. ISSUE SEARCH FIRST: call github_search_issues with keywords from the symptom "
    "on the relevant repo (default virtualdojo-inc/virtualdojo) BEFORE investigate() "
    "or code reads. If a closed bug issue already matches, prior investigation likely "
    "solved it — cite the issue and synthesize from there instead of re-investigating. "
    "Include this call in your first parallel batch.\n"
    "2a. ARCHITECTURE CONTEXT: in the same first batch, call search_wiki / "
    "read_knowledge on the engineering wiki (system-map, symptom-to-subsystem, "
    "frontend-map, backend-request-flow) to identify the OWNING subsystem and the "
    "service/model/view/endpoint that likely holds the bug BEFORE sync_repo / "
    "investigate — so code reads and investigate() calls target the right files "
    "instead of searching blind. (It's an orientation map; confirm against live "
    "code.) When you later write up the issue, name that subsystem + code path.\n"
    "3. Determine which environment has the issue: production (main branch) or dev "
    "(development branch). Call sync_repo to ensure the latest code is available.\n"
    "3a. CRITICAL: sync_repo must complete BEFORE search_repo_code / read_repo_file / "
    "read_repo_file_range / list_repo_files for the same repo. Do NOT put sync_repo "
    "in the same parallel tool batch as those readers — the readers will execute "
    "before the sync finishes and return 'Repo not synced yet'. Either wait for "
    "sync_repo to return in a prior turn, or skip the top-level sync_repo and let "
    "investigate() sync internally. (If you do omit the branch kwarg on a reader, "
    "it falls back to the most-recently-synced branch for that repo, but only AFTER "
    "sync_repo has returned — never assume it has.)\n"
    "4. PARALLEL INVESTIGATION (the speed lever): for any non-trivial bug, dispatch "
    "2-4 investigate() calls IN THE SAME TURN. LangGraph runs them concurrently — "
    "3 parallel calls take the same wall time as 1. Each investigator is a focused "
    "Flash-powered sub-agent that returns a written summary so your main context "
    "stays clean.\n"
    "   Example for 'why does endpoint X reject API keys':\n"
    "     investigate('Find the auth dependency used by endpoint X. Cite file:line.')\n"
    "     investigate('Find the auth dependency used by known-working endpoint Y. Cite file:line.')\n"
    "     investigate('Does the auth function support API keys? Trace the logic.')\n"
    "5. For simple single-file questions, skip investigate() and call search_repo_code "
    "or read_repo_file directly.\n"
    "6. DUPLICATE IMPLEMENTATION CHECK: when wiring-layer bugs are suspected (auth, "
    "middleware, DI, dependencies), explicitly search for duplicate definitions of "
    "the same function/symbol across the repo. Two get_current_user definitions in "
    "different modules is a common class of bug — don't stop at the first match.\n"
    "7. Cross-reference findings with logs (query_cloud_logs) and service status "
    "(list_cloud_run_services).\n"
    "8. SYNTHESIZE: state which hypothesis the evidence supports, why the others are "
    "ruled out, and the one-line fix (file:line + change). Don't delegate the "
    "conclusion to a sub-agent — you own the answer.\n"
    "9. SAVE THE PATTERN: if you reached a concrete root cause with file:line "
    "evidence AND the pattern could plausibly recur, call save_troubleshooting_step "
    "ONCE at the end. Populate hypotheses_ruled_out with the dead-ends you "
    "investigated — dead ends are as valuable as wins. Skip the save for one-off "
    "typos, trivial bugs, or cases where you only speculated. Prior saved patterns "
    "for similar symptoms are retrieved automatically and appear in the system "
    "prompt under 'Prior troubleshooting patterns' — read them first.\n\n"
    "Branch mapping:\n"
    "- Production issues: sync_repo(repo='virtualdojo-inc/virtualdojo', branch='main')\n"
    "- Development issues: sync_repo(repo='virtualdojo-inc/virtualdojo', branch='development')\n"
    "- Bot issues: sync_repo(repo='virtualdojo-inc/SamurAI', branch='main')\n"
    "Always sync before reading code — the local copy may be stale.\n"
    "When troubleshooting, read the actual code, don't guess at what it does."
)


PROMPT_SECTIONS = {
    "core": {"content": _CORE_SECTION, "keywords": []},
    "files": {
        "content": _FILES_SECTION,
        "keywords": [
            "spreadsheet", "excel", "csv", "upload", "column",
            "fill", "edit cell", "worksheet", "uploaded file",
        ],
    },
    "autofix": {
        "content": _AUTOFIX_SECTION,
        "keywords": ["autofix", "auto-fix", "auto fix", "label", "fix the bug"],
    },
    "crm": {
        "content": _CRM_SECTION,
        "keywords": [
            "crm", "contact", "account", "opportunity", "quote",
            "virtualdojo_crm", "connect to virtualdojo", "sign in", "signed in",
        ],
    },
    "salesforce": {
        "content": _SALESFORCE_SECTION,
        "keywords": [
            "salesforce", "sfdc", "case", "cases", "case number",
            "support case", "customer case", "close the case", "case status",
            "case comment", "escalate the case", "quotely org", "quotely case",
        ],
    },
    "deployment": {
        "content": _DEPLOYMENT_SECTION,
        "keywords": [
            "deploy", "deployment", "revision", "draining", "rollout",
            "after deploy", "cloud run", "regression",
        ],
    },
    "social": {
        "content": _SOCIAL_SECTION,
        "keywords": [
            "social", "post", "linkedin", "twitter", "facebook",
            "instagram", "publish", "schedule post", "preview post",
            "ayrshare", "draft post", "brand voice",
        ],
    },
    "projects": {
        "content": _PROJECTS_SECTION,
        "keywords": [
            "project board", "project items", "status field", "issue type",
            "set type", "update field", "draft issue", "add to project",
        ],
    },
    "search": {
        "content": _SEARCH_SECTION,
        "keywords": ["search for", "google", "look up", "what's the latest"],
    },
    "background_tasks": {
        "content": _BACKGROUND_TASKS_SECTION,
        "keywords": [
            "background task", "schedule", "recurring", "cron", "remind",
            "follow up", "check back", "one shot", "autonomous", "every hour",
            "every day", "daily", "weekly",
        ],
    },
    "teams_messaging": {
        "content": _TEAMS_MESSAGING_SECTION,
        "keywords": [
            "send message", "send a message", "teams message",
            "message to", "team roster", "lookup member", "team member",
        ],
    },
    "fedramp": {
        "content": _FEDRAMP_SECTION,
        "keywords": [
            "fedramp", "compliance", "evidence", "audit log review",
            "scc", "iam compliance", "log retention", "encryption",
            "vulnerability", "dependabot", "poam", "poa&m",
            "nist", "control family", "800-53", "oscal", "ssp",
        ],
    },
    "troubleshooting": {
        "content": _TROUBLESHOOTING_SECTION,
        "keywords": [
            "troubleshoot", "debug", "investigate", "root cause",
            "why is", "broken", "traceback", "stack trace", "what's wrong",
            "regression", "error in", "fix the", "failing",
            "sync repo", "read code", "search code", "source code",
            "codebase",
            # Natural-language bug-investigation phrasings (added 2026-05).
            "branch", "the code", " code", "bug", "the cause",
            "ground", "identify", "diagnose", "find the cause",
            "look at the", "check the source", "trace the",
            "where is", "find where",
        ],
    },
}


def _select_prompt_sections(message: str, hints_override: str | None = None) -> str:
    """Build the system prompt by selecting only relevant sections.

    Core section is always included. Other sections load when their keywords
    appear in the user's message. Mirrors _select_tool_groups.
    """
    # Cache per message (production path only — hints_override is the selftune
    # eval harness and must see a fresh assembly). The loaders below are TTL-
    # cached individually, but this also skips re-running them on every hop.
    if hints_override is None:
        hit = _prompt_cache.get(message)
        if hit is not None and (time.time() - hit[0]) < _PROMPT_CACHE_TTL:
            return hit[1]

    msg_lower = message.lower()
    # Section order is tuned for Gemini IMPLICIT prompt caching: the large,
    # request-invariant content goes FIRST so it forms a byte-stable prefix the
    # model can cache across turns and users (a cache read is ~75-90% cheaper than
    # fresh input). Per-message and volatile content goes AFTER, so a miss on it
    # never invalidates the shared prefix. Order:
    #   core -> skills catalog -> knowledge index   (stable prefix, ~every request)
    #   -> keyword-matched sections                 (per-message)
    #   -> tracker index -> learned hints           (volatile / self-tuned)
    #   -> retrieved memories                       (appended per-user in call_model)
    parts = [PROMPT_SECTIONS["core"]["content"]]
    # Skills catalog + knowledge index (level-1 disclosure): always advertise
    # available skills/articles so the agent can pull their full bodies via
    # get_skill / read_knowledge / search_wiki when relevant. Present on every
    # request and slow-changing, so they belong in the cacheable prefix.
    catalog = skills_catalog_text()
    if catalog:
        parts.append(catalog)
    index = knowledge_index_text()
    if index:
        parts.append(index)
    # Per-message domain sections (loaded by keyword match) — placed after the
    # stable prefix, since which ones apply varies with the user's message.
    for name, section in PROMPT_SECTIONS.items():
        if name == "core":
            continue
        if any(kw in msg_lower for kw in section["keywords"]):
            parts.append(section["content"])
    # Volatile: the tracker-diagnostics index updates as diagnoses are computed,
    # so keep it out of the cacheable prefix.
    tracker_index = tracker_diagnostics_index_text()
    if tracker_index:
        parts.append(tracker_index)
    # Learned operational guidance — the single mutable, self-tuned prompt layer
    # (selftune). Still injected AFTER the core so it refines, never overrides, the
    # frozen core; keeping it late also keeps it out of the cached prefix.
    # hints_override lets the self-tuning loop evaluate a candidate doc against
    # the exact production prompt; None = use the live doc.
    hints = wrap_hints(hints_override) if hints_override is not None else learned_hints_text()
    if hints:
        parts.append(hints)
    assembled = "\n\n".join(parts)
    if hints_override is None:
        if len(_prompt_cache) >= _PROMPT_CACHE_MAX:
            _prompt_cache.clear()
        _prompt_cache[message] = (time.time(), assembled)
    return assembled


# Joined for backward compatibility (tests reference SYSTEM_PROMPT as a string).
SYSTEM_PROMPT = "\n\n".join(s["content"] for s in PROMPT_SECTIONS.values())


# Keywords that trigger the Pro model for complex reasoning
PRO_MODEL_KEYWORDS = [
    # OSCAL & FedRAMP document work
    "oscal", "generate ssp", "generate poam", "assessment results",
    "migrate", "update control", "link evidence", "validate package",
    "render pdf", "review code", "fedramp_review_code",
    "propose edit", "commit document", "fedramp document",
    "update ssp", "update the ssp", "control implementation",
    "catalog lookup", "look up control",
    # Code troubleshooting & analysis
    "troubleshoot", "debug", "why is", "root cause", "stack trace",
    "traceback", "exception", "bug", "broken", "not working",
    "investigate", "diagnose", "analyze code", "code review",
    "what's wrong", "error in", "fix the", "failing",
    "fix issue", "attempt a fix", "attempt the fix",
    # Operations & log analysis (added 2026-05 — Flash was being chosen
    # for the most common troubleshooting phrasings, hurting answer quality)
    "logs", "errors", "regression", "outage", "alert", "alerts",
    "metrics", "monitoring", "crashed", "crash", "crashing",
    "downtime", "cloud run", "deployment", "deploy ", "revision",
    "review the ", "check the logs", "look at logs", "any errors",
    "what happened",
]


def _text_of(content) -> str:
    """Plain text of a message whose content may be a string or a multimodal list
    (text + image blocks, when images are attached — see _build_human_content).
    Everything that keyword-matches or caches on the user's text must go through
    this, or it hits `'list' object has no attribute 'lower'` on image turns."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            b.get("text", "") if isinstance(b, dict) and b.get("type") == "text"
            else b if isinstance(b, str) else ""
            for b in content
        ]
        return " ".join(p for p in parts if p)
    return str(content or "")


def _needs_pro_model(messages) -> bool:
    """Check if the conversation needs the Pro model for complex reasoning."""
    last_human = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
    )
    if not last_human:
        return False
    content = _text_of(last_human.content).lower()
    return any(kw in content for kw in PRO_MODEL_KEYWORDS)


SOFT_TOOL_LIMIT = 15  # Send a "still working" notice after this many unique tool calls

# Populated at the end of each run_agent() call so the scheduler can tell whether
# the agent already delivered content via send_teams_message — in which case the
# scheduler should suppress the proactive post of the agent's final text to the
# task creator, which otherwise shows up as a meta "Sent a message to..." echo.
# Keyed by conversation_id; consumers should .pop() after reading.
_last_run_metadata: dict[str, dict] = {}

# Vertex endpoint/region + serve/lite model ids — single source of truth in
# vertex_config (defaults to the US data-residency REP endpoint; env-overridable
# back to global). Kept as a module-level dict so `**agent._GCP_KWARGS` spreads
# (e.g. selftune/loop.py) keep working.
import vertex_config
_GCP_KWARGS = vertex_config.vertex_kwargs()

# Per-message prompt-assembly cache. _select_prompt_sections runs on EVERY graph
# hop (3-15 per turn) and its loaders (skills catalog, knowledge index, tracker
# index, hints) are pure functions of the message within a turn. Same bounded
# full-eviction pattern as _memory_cache below.
_prompt_cache: dict[str, tuple[float, str]] = {}
_PROMPT_CACHE_MAX = 100
_PROMPT_CACHE_TTL = 60.0


# Per-turn memory cache. retrieve_relevant_memories was running on every
# call_model entry (3+ times per turn, ~1s each). Within a turn the user
# message content is identical, so cache by (user_id, content). Bounded
# size with simple full-eviction on overflow — memory blobs are small.
_memory_cache: dict[tuple[str, str], str] = {}
_MEMORY_CACHE_MAX = 200


# Fire-and-forget tasks (post-reply logging). Strong refs so tasks aren't GC'd
# mid-flight; sync callables run in a worker thread to keep the loop free.
_background_tasks: set[asyncio.Task] = set()


def _spawn_background(fn, /, **kwargs) -> None:
    async def _run():
        try:
            await asyncio.to_thread(fn, **kwargs)
        except Exception as e:  # best-effort by contract — never surface
            logger.warning("[background] %s failed: %s", getattr(fn, "__name__", fn), e)

    task = asyncio.create_task(_run())
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def _retrieve_memories_cached(user_id: str, last_human) -> str:
    text = _text_of(last_human.content)
    key = (user_id, text)
    cached = _memory_cache.get(key)
    if cached is not None:
        return cached
    result = await retrieve_relevant_memories(user_id, text) or ""
    if len(_memory_cache) >= _MEMORY_CACHE_MAX:
        _memory_cache.clear()
    _memory_cache[key] = result
    return result


# gemini-3.5-flash has a 1,048,576-token context window. Long, tool-heavy
# conversations (history + tool traces + memory injections + checkpoint state)
# can blow past it, hard-crashing the turn with GoogleContextOverflowError.
# Trim oldest history down to this budget before each call.
#
# Budget is deliberately conservative (300k, not ~700k) because:
#   1. This bot's content (GCP logs, JSON, tracebacks, code) tokenizes denser
#      than plain prose — closer to ~3 chars/token than 4 — so a char-based
#      estimate undercounts real tokens.
#   2. The tool schemas bound via bind_tools ALSO count toward the 1M limit but
#      are NOT in the message list we trim — tens of thousands of extra tokens.
#   3. The model still needs room to generate its response.
# 300k estimated tokens leaves ~700k of headroom for all of the above.
_MAX_INPUT_TOKENS = 300_000

# Hard cap for any SINGLE message's content. A lone giant tool result or pasted
# log can exceed the whole budget by itself, and trim_messages (strategy="last")
# always keeps the most recent message — so we truncate oversized content before
# trimming. ~150k chars ≈ 50k tokens; generous for a single message.
_MAX_MSG_CHARS = 150_000


def _approx_tokens(msg_or_msgs) -> int:
    """Cheap, dependency-free token estimate for trim_messages.

    Uses ~3 chars/token (conservative for this bot's log/JSON/code-heavy
    content) so we lean toward over-counting and trim sooner rather than risk
    the hard 1M limit. trim_messages may call this with a single message OR a
    list (e.g. measuring the system message separately), so handle both.
    """
    if isinstance(msg_or_msgs, (list, tuple)):
        return sum(_approx_tokens(m) for m in msg_or_msgs)
    content = msg_or_msgs.content
    n = len(content) if isinstance(content, str) else len(str(content))
    tool_calls = getattr(msg_or_msgs, "tool_calls", None)
    if tool_calls:
        n += len(str(tool_calls))
    return n // 3 + 8


def _cap_message_content(messages):
    """Truncate any single message whose string content exceeds _MAX_MSG_CHARS.

    Returns a new list; oversized messages are replaced with copies holding
    head+tail of the content and a truncation marker (originals/ checkpoint
    state are left untouched). Non-string content (multimodal parts) is left
    as-is. This stops one huge tool output/paste from blowing the context limit
    on its own, which plain trimming can't fix since the latest message is kept.
    """
    out = []
    for m in messages:
        content = m.content
        if isinstance(content, str) and len(content) > _MAX_MSG_CHARS:
            head = content[: _MAX_MSG_CHARS // 2]
            tail = content[-_MAX_MSG_CHARS // 2 :]
            dropped = len(content) - _MAX_MSG_CHARS
            new_content = (
                f"{head}\n\n... [truncated {dropped:,} chars to fit the context "
                f"limit] ...\n\n{tail}"
            )
            out.append(m.model_copy(update={"content": new_content}))
        else:
            out.append(m)
    return out


def _content_has_parts(content, has_tool_calls: bool) -> bool:
    """True if a message would serialize to at least one non-empty Gemini part.

    Handles every content shape LangChain can carry, not just strings:
      - tool_calls present            -> function_call part(s) -> True
      - str                           -> True iff non-whitespace
      - list (multimodal)             -> True iff any part is a non-text part
                                         (image_url/media/…) OR a text part with
                                         non-empty text; [] or all-empty-text -> False
      - None                          -> False
      - anything else                 -> True (unknown; keep to be safe)
    """
    if has_tool_calls:
        return True
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        for part in content:
            if isinstance(part, str):
                if part.strip():
                    return True
            elif isinstance(part, dict):
                ptype = part.get("type")
                if ptype in (None, "text"):
                    if str(part.get("text", "")).strip():
                        return True
                else:
                    return True  # image_url / media / other non-text part
            else:
                return True  # unknown part object; assume renderable
        return False
    if content is None:
        return False
    return True


def _drop_empty_messages(messages):
    """Drop messages that would serialize to a Gemini request Content with ZERO
    parts. Gemini rejects such a request with 400 INVALID_ARGUMENT ("must include
    at least one parts field"), killing the whole turn ("Sorry, something went
    wrong").

    Zero-parts messages accumulate in long histories — a persisted empty AIMessage,
    or a multimodal turn whose image parts went missing leaving ``content=[]`` /
    ``[{"type":"text","text":""}]`` — and then poison EVERY subsequent turn once
    they fall inside the trimmed window, so this runs right before the model call.
    Always kept (they carry their own parts): SystemMessage, ToolMessage
    (function_response), and any AIMessage with tool_calls (function_call).
    """
    out = []
    dropped = []
    for m in messages:
        has_tool_calls = bool(getattr(m, "tool_calls", None))
        if isinstance(m, (SystemMessage, ToolMessage)) or _content_has_parts(
            m.content, has_tool_calls
        ):
            out.append(m)
        else:
            dropped.append(f"{type(m).__name__}({type(m.content).__name__})")
    if dropped:
        logger.warning(
            "[drop_empty] removed %d zero-parts message(s) before model call: %s",
            len(dropped), dropped,
        )
    return out


async def _ainvoke_with_backoff(llm_with_tools, messages, max_attempts: int = 6):
    """Invoke the model, retrying on Gemini 429 RESOURCE_EXHAUSTED with backoff.

    The agent call path has no quota guard, so a saturated per-minute quota
    surfaces to the user as a dead turn. Exponential backoff (1s, 2s, 4s, ...
    capped at 30s, +jitter) rides out QPM spikes; non-quota errors propagate
    immediately. 429 responses aren't billed, so extra attempts cost latency
    only — and the 2026-07 logs showed background tasks dying after exhausting
    the previous 3 retries during a quota storm.
    """
    delay = 1.0
    for attempt in range(max_attempts):
        try:
            return await llm_with_tools.ainvoke(messages)
        except ChatGoogleGenerativeAIError as e:
            msg = str(e)
            # Definitive telemetry for the empty-parts 400: if a zero-parts message
            # still slips past _drop_empty_messages, name each message's shape so we
            # can see exactly what produced it (instead of guessing).
            if "parts field" in msg or "INVALID_ARGUMENT" in msg:
                shapes = []
                for i, m in enumerate(messages):
                    c = m.content
                    desc = (f"len{len(c)}" if isinstance(c, (str, list)) else "None"
                            if c is None else type(c).__name__)
                    tc = "+tc" if getattr(m, "tool_calls", None) else ""
                    shapes.append(f"{i}:{type(m).__name__}[{type(c).__name__}:{desc}]{tc}")
                logger.error("[invoke.invalid_argument] %d msgs -> %s", len(messages), shapes)
            is_quota = "RESOURCE_EXHAUSTED" in msg or "429" in msg
            if not is_quota or attempt == max_attempts - 1:
                raise
            sleep_s = min(delay, 30.0) + random.uniform(0, 0.5)
            logger.warning(
                "Gemini quota (429); retry %d/%d after %.1fs",
                attempt + 1, max_attempts - 1, sleep_s,
            )
            await asyncio.sleep(sleep_s)
            delay *= 2


def _log_cache_stats(response) -> None:
    """Log implicit-cache hit rate from a model response (content-free).

    langchain-google-genai reports Gemini's cached-prefix reads in
    ``usage_metadata.input_token_details['cache_read']``. Logging input vs
    cache_read tokens lets us measure how much of each request hit the cached
    prefix — the signal for whether the prompt-ordering above is paying off.

    Emitted via print() like the rest of the [agent]/[investigate] telemetry:
    the app never configures a logging handler, so logger.info records are
    dropped (this line was invisible in Cloud Logging when it shipped as
    logger.info), and logger.warning would land on stderr, which Cloud Logging
    ingests at error severity and would pollute severity>=WARNING filters.
    Best-effort: never let telemetry break a turn.
    """
    try:
        um = getattr(response, "usage_metadata", None)
        if not um:
            return
        inp = int(um.get("input_tokens", 0) or 0)
        details = um.get("input_token_details") or {}
        cache_read = int(details.get("cache_read", 0) or 0)
        if inp:
            print(
                f"[cache] input_tokens={inp} cache_read={cache_read} "
                f"({100.0 * cache_read / inp:.0f}% cached) "
                f"output_tokens={int(um.get('output_tokens', 0) or 0)}",
                flush=True,
            )
    except Exception:  # telemetry must never crash the turn
        pass


async def _build_graph(user_id: str = "default"):
    """Build a LangGraph agent with user-specific CRM and memory tools."""
    llm_flash = ChatGoogleGenerativeAI(model=vertex_config.SERVE_MODEL, **_GCP_KWARGS)
    llm_pro = ChatGoogleGenerativeAI(model=vertex_config.SERVE_MODEL, **_GCP_KWARGS)
    # Fast synthesis model — used for the final-draft hop (tool results in,
    # producing prose out). Reasoning load is low; the bottleneck is just
    # generating text over a fat context. Env-gated for safe rollback.
    llm_synth = ChatGoogleGenerativeAI(model=vertex_config.LITE_MODEL, **_GCP_KWARGS)

    # User-specific tools
    memory_tools = await create_memory_tools(user_id)
    crm_tools = [
        create_virtualdojo_tool(user_id),
        create_virtualdojo_list_tools(user_id),
    ]
    # Per-user read-only tenant-data tools (support grants), authenticated via the
    # user's SSO session — runs as the signed-in user; prompts SSO sign-in if not.
    tenant_tools = create_tenant_data_tools(user_id)
    # Always-available user tools (CRM + tenant-data). Memory tools are keyword-gated.
    always_user_tools = crm_tools + tenant_tools

    # ToolNode needs ALL tools so it can execute whatever the LLM selected
    all_tools = ALL_TOOLS + crm_tools + memory_tools + tenant_tools
    tool_node = ToolNode(all_tools, handle_tool_errors=True)

    async def call_model(state: MessagesState):
        messages = state["messages"]

        # Select model: Pro for OSCAL/FedRAMP doc/code review, Flash for everything else
        if _needs_pro_model(messages):
            llm = llm_pro
        else:
            llm = llm_flash

        # Fast-synthesis path: when the agent re-enters with tool results
        # already in scope, the next step is usually "summarize and respond"
        # — minimal reasoning, just prose generation over fat context.
        # Route to Flash-Lite. Env-gated for safe rollback.
        # SAMURAI_FAST_SYNTH=off disables, anything else enables (default on).
        if (
            os.environ.get("SAMURAI_FAST_SYNTH", "on").lower() != "off"
            and messages
            and isinstance(messages[-1], ToolMessage)
            and not _needs_pro_model(messages)
        ):
            llm = llm_synth

        # Dynamically select tools based on the user's message
        last_human = next(
            (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
        )
        last_human_text = _text_of(last_human.content) if last_human else ""
        if last_human:
            selected_tools = _select_tool_groups(
                last_human_text, memory_tools=memory_tools
            ) + always_user_tools
        else:
            selected_tools = all_tools

        llm_with_tools = llm.bind_tools(selected_tools)

        # Build system prompt by selecting only relevant sections (mirrors
        # _select_tool_groups). Core is always-on; other sections load on
        # keyword match. Cuts active context 60-80% on the common case.
        if last_human:
            system_content = _select_prompt_sections(last_human_text)
            memory_context = await _retrieve_memories_cached(
                user_id, last_human
            )
            if memory_context:
                system_content += f"\n\n{memory_context}"
        else:
            system_content = SYSTEM_PROMPT

        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=system_content)] + messages

        # Guard the 1M-token context limit. First cap any single oversized
        # message (a lone giant tool output/paste that trimming alone can't
        # fix), then trim oldest history — keeping the system message and a
        # valid human-led tail. Both are no-ops for normal-length turns.
        messages = _cap_message_content(messages)
        before = len(messages)
        est_before = _approx_tokens(messages)
        messages = trim_messages(
            messages,
            max_tokens=_MAX_INPUT_TOKENS,
            token_counter=_approx_tokens,
            strategy="last",
            include_system=True,
            start_on="human",
        )
        if len(messages) < before:
            logger.warning(
                "Trimmed conversation history %d -> %d messages (~%d -> ~%d est "
                "tokens) to stay under the %d-token budget",
                before, len(messages), est_before, _approx_tokens(messages),
                _MAX_INPUT_TOKENS,
            )

        # Final guard: strip any zero-parts message (e.g. a persisted empty
        # AIMessage surfaced by trimming) that would 400 the whole turn.
        messages = _drop_empty_messages(messages)

        response = await _ainvoke_with_backoff(llm_with_tools, messages)
        _log_cache_stats(response)
        return {"messages": [response]}

    def should_continue(state: MessagesState):
        """Route after the agent node.

        - If the agent produced write tool calls AND the judge is enabled,
          route to the judge for evaluation.
        - If the agent produced only read tool calls (or the judge is off),
          run them directly.
        - Else if verification is enabled, route to the verification node.
        - Else end the graph.
        """
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return should_judge_writes(state)
        # No tool calls — draft is ready. Route to verification or END.
        return should_verify(state)

    def should_continue_after_verification(state: MessagesState):
        """Route after the verification node.

        If the verifier appended a correction message (ungrounded claims
        in enforce mode), route back to the agent for another turn.
        Otherwise end.
        """
        return should_route_from_verification(state)

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)
    graph.add_node("verify", verification_node)
    graph.add_node("judge", judge_writes_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "judge": "judge",
            "verify": "verify",
            "end": END,
        },
    )
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges(
        "judge",
        route_after_judge,
        {"tools": "tools", "agent": "agent", END: END},
    )
    graph.add_conditional_edges(
        "verify",
        should_continue_after_verification,
        {"agent": "agent", "end": END},
    )

    checkpointer = await get_checkpointer()
    store = await get_memory_store()
    return graph.compile(checkpointer=checkpointer, store=store)


# Cache of per-user graphs to avoid rebuilding on every message
_user_graphs: dict[str, object] = {}


async def _get_graph(user_id: str):
    """Get or create a LangGraph agent for a specific user."""
    if user_id not in _user_graphs:
        _user_graphs[user_id] = await _build_graph(user_id)
    return _user_graphs[user_id]


def reset_user_graph(user_id: str):
    """Reset a user's graph to pick up new tools (e.g. after OAuth)."""
    _user_graphs.pop(user_id, None)


async def inject_auth_message(user_id: str, conversation_id: str):
    """Inject a message into the conversation history confirming CRM auth succeeded."""
    graph = await _get_graph(user_id)
    config = {"configurable": {"thread_id": conversation_id, "user_id": user_id}}
    await graph.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="[SYSTEM: The user has successfully authenticated with VirtualDojo CRM. "
                    "The connection is now active. You can now use virtualdojo_crm and "
                    "virtualdojo_list_tools to access their CRM data. "
                    "Do NOT ask the user to connect again.]"
                )
            ]
        },
        config=config,
    )


def _extract_text(content) -> str:
    """Extract plain text from Gemini's content blocks."""
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return content


_CONTINUE_KEYWORDS = (
    "continue", "resume", "keep going", "pick up", "carry on",
    "where were you", "where were we", "go on",
)


def _is_continue_intent(message: str) -> bool:
    """Detect whether the user's message is a request to resume prior work.

    Uses an exact-token match against a short keyword list. We intentionally
    do NOT use substring matching ("continue" would match "continuous"); the
    user has to actually be asking to continue. Empty messages and long
    messages (>40 chars) never match — those are real requests, not resumes.
    """
    if not message:
        return False
    stripped = message.strip().lower().rstrip(".!?")
    if len(stripped) > 40:
        return False
    return stripped in _CONTINUE_KEYWORDS


async def _synthesize_partial_findings(
    user_message: str,
    tool_log: list[str],
    progress: dict | None,
    reason: str,
) -> str:
    """Generate a real recovery message when the agent ran out of tool calls.

    Fires a single Flash-Lite call in a fresh context (no tools, no history)
    with the original user question, the agent's self-reported progress
    doc (if any), and the tool log as supporting evidence. Returns a
    natural-language summary citing what was found, what's pending, and
    asking if the user wants to continue.

    Mirrors the verification node's design (verification.py): separate
    client, fresh context, single shot. The verifier catches fabricated
    claims; this one synthesizes ungenerated ones.
    """
    synth_llm = ChatGoogleGenerativeAI(
        model=vertex_config.LITE_MODEL, **_GCP_KWARGS
    )

    log_tail = "\n".join(tool_log[-30:]) if tool_log else "(no tools called)"
    progress_block = (
        f"\nThe agent's self-reported plan at the time of stopping:\n"
        f"{render_progress_markdown(progress)}\n"
        if progress
        else ""
    )

    reason_line = {
        "recursion_limit": "Ran out of tool-call budget before finishing.",
        "empty_response": "Finished the tool work but didn't generate a reply.",
    }.get(reason, "Stopped without a reply.")

    prompt = (
        "You are summarizing a partial agent investigation for a user on "
        "Microsoft Teams. The agent stopped before producing a response — "
        f"{reason_line}\n\n"
        "Write a reply under 200 words that:\n"
        "1. Names the concrete things that were found (cite specifics from "
        "the tool log — issue numbers, file paths, row IDs).\n"
        "2. Names what's still uncertain or unfinished.\n"
        "3. Asks if the user wants to continue, or suggests a more focused "
        "follow-up question they could ask.\n\n"
        "Ground every specific claim in the tool log or the plan below. "
        "Do not invent details. If the data is too thin for a real summary, "
        "say so plainly.\n\n"
        f"User's original question:\n{user_message}\n"
        f"{progress_block}\n"
        f"Tool calls made (most recent {min(len(tool_log), 30)} of "
        f"{len(tool_log)}):\n{log_tail}"
    )

    try:
        # Vertex Gemini requires a user-role message in contents; a
        # system-only call returns "contents are required".
        response = await synth_llm.ainvoke([HumanMessage(content=prompt)])
        text = _extract_text(response.content)
        return text.strip() if text else ""
    except Exception as e:
        logger.warning("[run_agent] synthesizer call failed: %s", e)
        return ""


def _generic_recovery_message(progress: dict | None) -> str:
    """Fallback recovery message when the synthesizer can't be reached.

    Better than the legacy generic line because it at least surfaces the
    agent's self-reported plan if one exists.
    """
    if progress:
        rendered = render_progress_markdown(progress)
        return (
            "I hit my tool-call limit before finishing. Here's where I was:\n\n"
            f"{rendered}\n\n"
            "Say **continue** to resume from here."
        )
    return (
        "I've gathered a lot of information but hit my tool call limit. "
        "Here's what I have so far — ask me to continue if you need more detail."
    )


def _build_human_content(message: str, images: list[dict] | None):
    """HumanMessage content for the turn. Plain string normally; a multimodal list
    (text block + image blocks) when images are attached so the model can SEE them.
    langchain-google-genai requires the data-content-block form
    {"type":"image","base64":..,"mime_type":..} — image_url forms are NOT accepted
    by its parser. Malformed image entries are dropped."""
    if not images:
        return message
    blocks: list = [{"type": "text", "text": message}]
    for im in images:
        if im.get("data") and im.get("mime_type"):
            blocks.append({"type": "image", "base64": im["data"], "mime_type": im["mime_type"]})
    return blocks


async def run_agent(
    user_message: str,
    conversation_id: str = "default",
    user_id: str = "default",
    user_name: str = "",
    user_timezone: str = "",
    user_email: str = "",
    status_callback=None,
    recursion_limit: int = 75,
    is_background_task: bool = False,
    images: list[dict] | None = None,
) -> str:
    start = time.time()

    # Build context prefix so the LLM knows who it's talking to
    context_parts = []
    if user_name:
        context_parts.append(f"User: {user_name}")
    if user_email:
        context_parts.append(f"Email: {user_email}")
    context_parts.append(f"conversation_id: {conversation_id}")
    if user_timezone:
        from datetime import datetime
        import zoneinfo

        try:
            tz = zoneinfo.ZoneInfo(user_timezone)
            local_time = datetime.now(tz).strftime("%Y-%m-%d %H:%M %Z")
            context_parts.append(
                f"Timezone: {user_timezone} (current time: {local_time})"
            )
        except Exception:
            context_parts.append(f"Timezone: {user_timezone}")

    # If this is a continue/resume request AND we have a prior plan for this
    # conversation, inject the plan into the message so the agent picks up
    # from where it stopped instead of asking "continue what?"
    resumed_from_plan = False
    if _is_continue_intent(user_message):
        prior = get_progress(conversation_id)
        if prior:
            user_message = (
                f"{user_message}\n\n"
                f"[Resuming the prior plan — pick up from in_progress / pending "
                f"items, do NOT redo completed work]\n"
                f"{render_progress_markdown(prior)}"
            )
            resumed_from_plan = True

    message = user_message
    if is_background_task:
        message = (
            "[BACKGROUND TASK — This is an autonomous scheduled task, not a live "
            "user message. Execute the instruction below directly and return the "
            "result. Do NOT ask clarifying questions, suggest creating tasks, or "
            "check for duplicate tasks. Just do the work and report your findings.]\n"
            + message
        )
    if context_parts:
        message = f"[{' | '.join(context_parts)}]\n{message}"

    # Multimodal: a content list (text + image blocks) when images are attached,
    # plain string otherwise. See _build_human_content.
    human_content = _build_human_content(message, images)

    # Instant canned acknowledgment. This was an extra Vertex call (~0.5-1s of
    # serial latency + one billed request per message) purchased for a one-line
    # nicety — a static ack plus the typing indicator reads the same and is free.
    if status_callback:
        try:
            await status_callback("On it — working on that now.")
        except Exception:
            pass  # Don't block the main agent if ack fails

    graph = await _get_graph(user_id)
    config = {
        "configurable": {"thread_id": conversation_id, "user_id": user_id},
        "recursion_limit": recursion_limit,
    }

    # Fallback for any tool missing from _tool_labels: turn a raw snake_case
    # name into a human label so we never surface "search_wiki" to the user.
    def _humanize_tool_name(n: str) -> str:
        for prefix in ("github_", "fedramp_", "oscal_", "social_", "smartsheet_",
                       "virtualdojo_", "db_", "gcp_"):
            if n.startswith(prefix):
                n = n[len(prefix):]
                break
        return n.replace("_", " ").strip().capitalize() or "Working"

    # Friendly tool names for status updates
    _tool_labels = {
        "sync_repo": "Syncing repository",
        "read_repo_file": "Reading source code",
        "read_repo_file_range": "Reading code range",
        "search_repo_code": "Searching codebase",
        "list_repo_files": "Browsing files",
        "investigate": "Dispatching investigator",
        "save_troubleshooting_step": "Saving troubleshooting pattern",
        "search_troubleshooting": "Searching troubleshooting patterns",
        "delete_troubleshooting_step": "Removing troubleshooting pattern",
        "query_cloud_logs": "Querying Cloud Logging",
        "list_cloud_run_services": "Checking Cloud Run services",
        "check_gcp_metrics": "Checking metrics",
        "gcp_billing_summary": "Checking billing costs",
        "github_list_prs": "Checking pull requests",
        "github_get_pr_details": "Reading PR details",
        "github_list_recent_commits": "Checking recent commits",
        "github_get_commit_diff": "Reading commit diff",
        "github_list_issues": "Checking GitHub issues",
        "github_search_issues": "Searching GitHub issues",
        "github_get_issue_details": "Reading issue details",
        "github_create_issue": "Creating GitHub issue",
        "github_get_issue_type": "Reading issue type",
        "github_set_issue_type": "Setting issue type",
        "github_close_issue": "Closing GitHub issue",
        "github_edit_issue": "Editing GitHub issue",
        "github_list_workflow_runs": "Checking CI/CD workflows",
        "github_get_workflow_run_details": "Reading workflow details",
        "fedramp_collect_evidence": "Collecting FedRAMP evidence",
        "fedramp_daily_log_review": "Running audit log review",
        "fedramp_check_scc_findings": "Checking Security Command Center",
        "fedramp_evidence_summary": "Building compliance dashboard",
        "oscal_generate_ssp": "Generating OSCAL SSP",
        "oscal_validate_package": "Validating OSCAL package",
        "oscal_render_pdf": "Rendering PDF",
        "google_search": "Searching the web",
        "send_teams_message": "Sending Teams message",
        "create_background_task": "Creating background task",
        "manage_memory": "Saving to memory",
        "search_memory": "Searching memory",
        "manage_core_memory": "Saving operational knowledge",
        "search_core_memory": "Searching operational knowledge",
        "manage_team_memory": "Saving team knowledge",
        "search_team_memory": "Searching team knowledge",
        "get_uploaded_file_content": "Reading uploaded file",
        "get_spreadsheet_info": "Analyzing spreadsheet structure",
        "read_spreadsheet_cells": "Verifying spreadsheet changes",
        "edit_document": "Editing document",
        "edit_spreadsheet": "Editing spreadsheet",
        "fill_spreadsheet_column": "Filling spreadsheet column",
        "fedramp_read_document": "Reading FedRAMP document",
        "fedramp_list_documents": "Browsing FedRAMP docs",
        "fedramp_search_documents": "Searching FedRAMP docs",
        "fedramp_propose_edit": "Proposing document edit",
        "fedramp_commit_document": "Committing to GitHub",
        "fedramp_review_code": "Reviewing code for compliance",
        "fedramp_check_log_retention": "Checking log retention",
        "fedramp_check_encryption": "Checking encryption",
        "fedramp_check_iam_compliance": "Checking IAM compliance",
        "fedramp_check_failed_logins": "Checking failed logins",
        "fedramp_check_dependabot_alerts": "Checking Dependabot alerts",
        "fedramp_poam_status": "Checking POA&M status",
        "oscal_generate_poam": "Generating OSCAL POA&M",
        "oscal_catalog_lookup": "Looking up NIST control",
        "oscal_update_control": "Updating control implementation",
        "oscal_migrate_from_markdown": "Migrating to OSCAL",
        "oscal_link_evidence": "Linking evidence",
        "oscal_generate_assessment_results": "Generating assessment results",
        # Background tasks
        "list_background_tasks": "Checking background tasks",
        "pause_background_task": "Pausing task",
        "resume_background_task": "Resuming task",
        "cancel_background_task": "Cancelling task",
        # Teams
        "lookup_team_member": "Looking up team member",
        "list_team_members": "Listing team members",
        # GitHub Projects
        "github_list_projects": "Listing GitHub projects",
        "github_get_project_items": "Reading project items",
        "github_create_draft_issue": "Creating draft issue",
        "github_add_item_to_project": "Adding to project",
        "github_update_item_field": "Updating project field",
        # Social media
        "social_preview_post": "Drafting social post",
        "social_publish_post": "Publishing post",
        "social_schedule_post": "Scheduling post",
        "social_list_scheduled": "Checking scheduled posts",
        "social_get_post": "Reading post details",
        "social_update_post": "Updating post",
        "social_delete_post": "Deleting post",
        "social_generate_image": "Generating image",
        # FedRAMP docs
        "fedramp_discard_draft": "Discarding draft",
        "fedramp_scan_container_vulnerabilities": "Scanning container vulnerabilities",
        # Smartsheet
        "smartsheet_list_sheets": "Listing Smartsheets",
        "smartsheet_get_sheet": "Reading Smartsheet rows",
        "smartsheet_update_row": "Updating Smartsheet row",
        # Salesforce case management
        "query_cases": "Querying Salesforce cases",
        "get_case_details": "Reading Salesforce case details",
        "add_case_comment": "Adding a case comment",
        "update_case_status": "Updating case status",
        # Knowledge base & skills
        "get_skill": "Looking up a skill",
        "read_knowledge": "Reading knowledge article",
        "search_wiki": "Searching the knowledge base",
        # CRM
        "virtualdojo_crm": "Querying VirtualDojo CRM",
        "virtualdojo_list_tools": "Checking CRM capabilities",
        # Database
        "db_query": "Querying the database",
        "db_list_tables": "Listing database tables",
        "db_describe_table": "Inspecting table schema",
        "db_check_user": "Looking up user record",
        "db_recent_audit_logs": "Checking audit logs",
        # Self-improvement
        "trigger_wiki_compile": "Updating knowledge base",
        "trigger_engineering_compile": "Updating engineering knowledge",
        # Progress tracking
        "update_progress": "Updating plan",
        # Skills authoring + tracker diagnostics
        "save_skill": "Saving a skill",
        "delete_skill": "Deleting a skill",
        "get_tracker_diagnostics": "Checking tracker diagnostics",
        # Code sandbox
        "run_code": "Running code in the sandbox",
        "find_prior_script": "Finding a past script",
        # Loom video analysis
        "analyze_loom_video": "Watching the Loom video",
        # Tenant data (support grants)
        "list_tenant_support_grants": "Listing tenant support grants",
        "describe_tenant_schema": "Reading the tenant's schema",
        "read_tenant_records": "Reading the tenant's records",
    }

    final_messages = []
    _sent_statuses: set[str] = set()  # Track sent labels to avoid duplicates
    _sent_midstream_summary = False
    _tool_call_log: list[str] = []  # Track tool calls for memory extraction
    _tools_invoked: list[str] = []  # Tool names the agent asked to run this call
    _teams_recipients: list[str] = []  # Emails send_teams_message targeted
    _hit_recursion_limit = False  # Set if GraphRecursionError fires

    from langgraph.errors import GraphRecursionError

    try:
      async for event in graph.astream(
        {"messages": [HumanMessage(content=human_content)]},
        config=config,
        stream_mode="updates",
      ):
        # event is a dict like {"agent": {"messages": [...]}} or {"tools": {"messages": [...]}}
        if "agent" in event:
            final_messages = event["agent"].get("messages", [])
            # Send status updates for tool calls (ack already sent above)
            if final_messages:
                last_msg = final_messages[-1]
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    tool_names = [tc.get("name", "") for tc in last_msg.tool_calls]
                    _tools_invoked.extend(tool_names)
                    # Capture send_teams_message recipients so the scheduler
                    # can skip a redundant proactive post when the content has
                    # already been delivered to the task creator.
                    for tc in last_msg.tool_calls:
                        if tc.get("name") == "send_teams_message":
                            recipient = (tc.get("args") or {}).get("recipient_email", "")
                            if recipient:
                                _teams_recipients.append(recipient.lower())
                    # Include args in the log line so 4xx/5xx tool errors can be
                    # diagnosed without instrumenting each tool. Truncated to
                    # keep log entries bounded; redact obvious secret-like keys.
                    _SECRET_ARG_KEYS = {"token", "api_key", "password", "secret", "private_key"}
                    call_summaries = []
                    for tc in last_msg.tool_calls:
                        raw_args = tc.get("args") or {}
                        safe_args = {
                            k: ("***" if any(s in k.lower() for s in _SECRET_ARG_KEYS) else v)
                            for k, v in raw_args.items()
                        }
                        args_str = str(safe_args)
                        if len(args_str) > 500:
                            args_str = args_str[:500] + "...(truncated)"
                        call_summaries.append(f"{tc.get('name','')}({args_str})")
                    print(
                        f"[agent] tool_calls: {call_summaries} conv={conversation_id}",
                        flush=True,
                    )
                    if status_callback:
                        new_labels = []
                        for n in tool_names:
                            label = _tool_labels.get(n) or _humanize_tool_name(n)
                            if label not in _sent_statuses:
                                _sent_statuses.add(label)
                                new_labels.append(label)
                        if new_labels:
                            status = "_" + ", ".join(new_labels) + "..._"
                            try:
                                await status_callback(status)
                            except Exception:
                                pass

                        # Surface the plan to the user the moment the agent
                        # commits to it, not after update_progress returns.
                        # The args ARE the plan — we don't need the result.
                        for tc in last_msg.tool_calls:
                            if tc.get("name") != "update_progress":
                                continue
                            try:
                                rendered = render_progress_markdown(tc.get("args") or {})
                                if rendered:
                                    await status_callback(rendered)
                            except Exception:
                                pass

        elif "tools" in event:
            final_messages = event["tools"].get("messages", [])

            # Log tool results and track for memory extraction
            from langchain_core.messages import ToolMessage
            for msg in final_messages:
                if isinstance(msg, ToolMessage):
                    content_str = str(msg.content) if msg.content is not None else ""
                    content_preview = content_str[:200]
                    size = len(content_str)
                    status = "error" if msg.status == "error" else "ok"
                    print(f"[agent] tool_result: {msg.name} ({status}) size={size} conv={conversation_id} -> {content_preview}", flush=True)
                    _tool_call_log.append(f"{msg.name}: {status} -> {content_preview[:150]}")

            # Mid-stream summary: when we hit the soft limit, notify user but keep going
            if status_callback and not _sent_midstream_summary:
                from langchain_core.messages import ToolMessage
                tool_count = sum(1 for m in final_messages if isinstance(m, ToolMessage))
                # Check total tool messages across all events
                total_tools = len(_sent_statuses)
                if total_tools >= SOFT_TOOL_LIMIT:
                    _sent_midstream_summary = True
                    try:
                        await status_callback(
                            "This is taking longer than expected — still working. "
                            "Say **stop** if you'd like me to wrap up."
                        )
                    except Exception:
                        pass
    except GraphRecursionError as e:
        # The agent hit the hard tool-call limit. Don't propagate — we have
        # _tool_call_log and the progress doc in scope and can produce a real
        # recovery message. Without this catch the error escapes to app.py
        # where the log is no longer accessible.
        _hit_recursion_limit = True
        logger.warning(
            "[run_agent] GraphRecursionError after %d tool calls — synthesizing recovery",
            len(_tool_call_log),
        )
        print(
            f"[run_agent] recursion_limit_hit conv={conversation_id} "
            f"tools={len(_tool_call_log)} err={type(e).__name__}",
            flush=True,
        )

    elapsed = time.time() - start
    logger.info("[run_agent] user=%s elapsed=%.2fs", user_name or user_id, elapsed)

    response_text = ""
    if final_messages:
        response_text = _extract_text(final_messages[-1].content) or ""

    # If the agent hit the recursion limit OR ended with an empty AIMessage
    # (tool calls but no text), synthesize a real recovery reply from the
    # tool log + progress doc instead of returning the legacy generic line.
    # Env-gated for safe rollback: SAMURAI_SYNTHESIZE_ON_LIMIT=off skips
    # the synth call and uses the generic fallback.
    needs_recovery = _hit_recursion_limit or not (response_text or "").strip()
    if needs_recovery:
        progress_entry = get_progress(conversation_id)
        synth_enabled = (
            os.environ.get("SAMURAI_SYNTHESIZE_ON_LIMIT", "on").lower() != "off"
        )
        if synth_enabled and (progress_entry or _tool_call_log):
            response_text = await _synthesize_partial_findings(
                user_message=user_message,
                tool_log=_tool_call_log,
                progress=progress_entry,
                reason="recursion_limit" if _hit_recursion_limit else "empty_response",
            )
        if not (response_text or "").strip():
            response_text = _generic_recovery_message(progress_entry)
    else:
        # The agent completed successfully — clear the conversation's
        # progress so a stale plan doesn't haunt the next turn.
        clear_progress(conversation_id)

    if not response_text:
        logger.error("[run_agent] empty messages in result for thread=%s", conversation_id)
        return "I wasn't able to generate a response. Please try again."

    # Background memory extraction — auto-saves facts from conversation.
    # Each step is observable via print() because the root logger is at
    # WARNING and logger.info gets dropped on Cloud Run. The team and user
    # tiers are populated, but core has been empty for weeks — instrument
    # each step so we can identify which one fails.
    extraction_content = response_text
    if _tool_call_log:
        tools_used = "\n".join(_tool_call_log[:10])
        extraction_content += f"\n\n[Tools used in this interaction:\n{tools_used}]"

    msg_payload = {"messages": [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": extraction_content},
    ]}
    user_config = {"configurable": {"user_id": user_id}}

    # Durable raw-conversation capture (the wiki's nightly ingest). Best-effort:
    # log_turn swallows its own errors so it can never break the turn. Runs in
    # the background — it writes to the GCS FUSE mount, and the reply must not
    # wait on a network filesystem.
    _spawn_background(
        log_turn,
        conversation_id=conversation_id,
        user_id=user_id,
        user_name=user_name,
        user_email=user_email,
        user_message=user_message,
        assistant_response=response_text,
        tools=list(_tool_call_log[:25]),
        is_background_task=is_background_task,
    )

    # Support-scope chat → in-boundary knowledge bucket (log only, env-gated).
    # Heuristic: a turn counts as "support" if it used a support-related tool or
    # the user message mentions support topics. Refine classification later.
    _tools_blob = " ".join(_tool_call_log).lower()
    _is_support = any(
        k in _tools_blob for k in ("smartsheet", "github_", "troubleshoot", "investigate")
    ) or any(
        k in user_message.lower()
        for k in ("support", "ticket", "troubleshoot", "bug", "not working", "error", "issue")
    )
    if _is_support:
        # Background: a synchronous GCS upload that fires on most troubleshooting
        # turns — must not sit between the agent finishing and the reply sending.
        _spawn_background(
            log_support_chat,
            conversation_id=conversation_id,
            user_id=user_id,
            user_name=user_name,
            user_message=user_message,
            assistant_response=response_text,
            tools=list(_tool_call_log[:25]),
        )

    # 1. Always print this BEFORE any submit so we know the block was reached.
    print(
        f"[memory.extract] start user_id={user_id!r} conv={conversation_id!r} "
        f"tool_count={len(_tool_call_log)} content_chars={len(extraction_content)}",
        flush=True,
    )

    # 2. Submit to each tier in its own try/except so one failure doesn't
    # cascade. Each result is logged so we can tell which tier broke.
    for tier_name, getter, delay in (
        ("user", get_background_extractor, 1.0),
        ("core", get_core_extractor, 2.0),
        ("team", get_team_extractor, 3.0),
    ):
        try:
            executor = await getter()
            executor.submit(msg_payload, config=user_config, after_seconds=delay)
            print(f"[memory.extract] {tier_name} submit OK", flush=True)
        except Exception as e:
            print(
                f"[memory.extract] {tier_name} submit FAILED: "
                f"{type(e).__name__}: {e}",
                flush=True,
            )

    # Periodic persistence — flush memories to SQLite every call
    # (lightweight no-op if nothing changed)
    try:
        persist_memories()
    except Exception:
        pass

    # Expose per-run metadata so the scheduler can decide whether to suppress
    # the proactive post (e.g. the agent already delivered content via Teams).
    _last_run_metadata[conversation_id] = {
        "tools_invoked": list(_tools_invoked),
        "teams_recipients": list(_teams_recipients),
        "finished_at": time.time(),
    }

    return response_text
