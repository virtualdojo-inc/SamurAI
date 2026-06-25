"""Memory system using LangMem + LangGraph InMemoryStore with SQLite persistence.

Provides:
1. LangGraph InMemoryStore — fast in-RAM memory with semantic search
2. LangMem tools — manage_memory + search_memory for the agent
3. Background extraction — auto-extracts memories after conversations
4. SQLite persistence — periodic flush to GCS for survival across restarts
5. AsyncSqliteSaver — LangGraph checkpointer for conversation history
"""

import json
import logging
import os
import time

logger = logging.getLogger(__name__)

DATA_DIR = os.environ.get("SAMURAI_DATA_DIR", "/data")
MEMORY_DB_PATH = os.path.join(DATA_DIR, "langmem_memories.sqlite")
# Checkpoints on local SSD — too write-heavy for GCS FUSE
CHECKPOINT_DB_PATH = "/tmp/checkpoints.sqlite"

# Three-tier memory namespaces
CORE_NAMESPACE = ("core",)  # Operational knowledge — available to all users
TEAM_NAMESPACE = ("team", "virtualdojo")  # Internal team knowledge — team only
USER_NAMESPACE = ("memories", "{user_id}")  # Personal preferences — per user

# Singletons
_store = None
_store_pool = None
_checkpointer = None
_checkpoint_conn = None
_checkpoint_pool = None
_background_executor = None
_core_executor = None
_team_executor = None


# ── Vertex AI Embedding Function ──────────────────────────────────────


def _create_embed_fn():
    """Create an embedding function using Vertex AI (service-account auth).

    Uses langchain_google_genai.GoogleGenerativeAIEmbeddings with vertexai=True,
    the same pattern as ChatGoogleGenerativeAI elsewhere in the bot. Without
    the vertexai flag this class defaults to the Gemini Developer API and
    requires GOOGLE_API_KEY — which the Cloud Run service account cannot
    provide.
    """
    _embeddings = None

    def embed(texts: list[str]) -> list[list[float]]:
        nonlocal _embeddings
        if _embeddings is None:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            _embeddings = GoogleGenerativeAIEmbeddings(
                model="text-embedding-005",
                vertexai=True,
                project=os.environ.get("GCP_PROJECT_ID"),
                location=os.environ.get("GCP_LOCATION", "us-central1"),
            )
        return _embeddings.embed_documents(texts)

    return embed


def _create_extractor_llm():
    """Build the ChatGoogleGenerativeAI client used by the LangMem extractors.

    Same Vertex-auth caveat as _create_embed_fn: passing the model as a string
    to create_memory_store_manager() resolves via init_chat_model, which
    defaults to the Gemini Developer API path and requires GOOGLE_API_KEY —
    which Cloud Run doesn't have (it authenticates via service account).
    Construct the client explicitly with vertexai=True so it routes through
    Vertex AI like the rest of the bot.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        vertexai=True,
        project=os.environ.get("GCP_PROJECT_ID"),
        location=os.environ.get("GCP_LOCATION", "us-central1"),
    )


# ── Memory Store (InMemoryStore + SQLite persistence) ─────────────────


async def get_memory_store():
    """Get the singleton memory store with embedding search.

    Postgres (pgvector via AsyncPostgresStore) when DATABASE_URL is set — memories
    + embeddings live IN the DB, shared across instances, with NO startup load and
    NO re-embedding (this removes the multi-minute cold start the InMemoryStore had
    from re-embedding thousands of memories off GCS-FUSE SQLite). Falls back to the
    InMemoryStore + SQLite backup (tests/local, no DATABASE_URL).

    Async because the Postgres store needs an async connection pool + setup; all
    callers use ``await store.asearch(...)`` which works on both backends.
    """
    global _store, _store_pool
    if _store is not None:
        return _store

    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        try:
            from langgraph.store.postgres.aio import AsyncPostgresStore
            from psycopg.rows import dict_row
            from psycopg_pool import AsyncConnectionPool

            conninfo = database_url.replace("postgresql+asyncpg://", "postgresql://")
            _store_pool = AsyncConnectionPool(
                conninfo=conninfo,
                min_size=0,
                # max_size 10 pairs with the checkpoint pool (10) for a ~20-conn
                # per-instance ceiling under Cloud SQL samurai-db's
                # max_connections=50 (tier default, db-g1-small — verified live).
                # min_size=0 + max_idle is deliberate: a Teams bot is idle most of
                # the time, and min_size>0 made warm-but-idle instances squat one
                # connection per pool indefinitely (observed: candidate revisions
                # holding connections idle for 3+ hours). With min_size=0 an idle
                # instance drains to ZERO connections; the first request after a
                # lull pays a ~50ms reconnect, which is fine here.
                max_size=10,
                max_idle=180.0,
                open=False,
                kwargs={"autocommit": True, "row_factory": dict_row, "prepare_threshold": 0},
            )
            await _store_pool.open()
            _store = AsyncPostgresStore(
                _store_pool,
                index={"dims": 768, "embed": _create_embed_fn()},
            )
            await _store.setup()
            logger.info("Postgres memory store ready (pgvector, no startup load)")
            print("[memory] postgres store ready", flush=True)
            return _store
        except Exception as e:
            logger.warning("Postgres store unavailable, falling back to InMemoryStore: %s", e)
            print(f"[memory] postgres store failed: {type(e).__name__}: {e}", flush=True)
            _store = None

    from langgraph.store.memory import InMemoryStore

    _store = InMemoryStore(
        index={
            # text-embedding-005 on Vertex returns 768-dim vectors by default.
            "dims": 768,
            "embed": _create_embed_fn(),
        }
    )
    # Load persisted memories from SQLite (fallback path only).
    _load_persisted_memories(_store)
    logger.info("LangMem memory store ready (InMemoryStore + SQLite backup)")
    # print() mirrors the logger line so it's guaranteed to surface in
    # Cloud Run's captured stdout (some logger handlers silently drop).
    print("[memory] store ready", flush=True)
    return _store


def _load_persisted_memories(store):
    """Load memories from SQLite into the InMemoryStore on startup."""
    import sqlite3

    if not os.path.exists(MEMORY_DB_PATH):
        return

    try:
        conn = sqlite3.connect(MEMORY_DB_PATH)
        cursor = conn.execute(
            "SELECT namespace, key, value_json, created_at, updated_at FROM memories"
        )
        count = 0
        for row in cursor:
            namespace = tuple(json.loads(row[0]))
            key = row[1]
            value = json.loads(row[2])
            store.put(namespace, key, value)
            count += 1
        conn.close()
        if count:
            logger.info("Loaded %d persisted memories from SQLite", count)
            print(f"[memory] loaded {count} persisted memories from SQLite", flush=True)
    except Exception as e:
        logger.warning("Failed to load persisted memories: %s", e)
        print(f"[memory] load failed: {type(e).__name__}: {e}", flush=True)


def persist_memories():
    """Flush the InMemoryStore to SQLite for persistence across restarts.

    No-op for the Postgres store (writes land in the DB immediately) and when
    no store has been created yet.
    """
    import sqlite3

    if _store is None:
        return
    from langgraph.store.memory import InMemoryStore

    if not isinstance(_store, InMemoryStore):
        return  # Postgres-backed store persists on write; nothing to flush.

    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        conn = sqlite3.connect(MEMORY_DB_PATH)
        conn.execute(
            """CREATE TABLE IF NOT EXISTS memories (
                namespace TEXT NOT NULL,
                key TEXT NOT NULL,
                value_json TEXT NOT NULL,
                created_at TEXT,
                updated_at TEXT,
                PRIMARY KEY (namespace, key)
            )"""
        )

        # Get all items from the store by searching known namespaces
        # InMemoryStore stores items internally — we iterate via _data
        items_saved = 0
        if hasattr(_store, '_data'):
            for namespace_tuple, keys in _store._data.items():
                ns_json = json.dumps(list(namespace_tuple))
                for key, item in keys.items():
                    value_json = json.dumps(item.value)
                    created = getattr(item, 'created_at', '')
                    updated = getattr(item, 'updated_at', '')
                    conn.execute(
                        """INSERT OR REPLACE INTO memories
                           (namespace, key, value_json, created_at, updated_at)
                           VALUES (?, ?, ?, ?, ?)""",
                        (ns_json, key, value_json, str(created), str(updated)),
                    )
                    items_saved += 1

        conn.commit()
        conn.close()
        if items_saved:
            logger.info("Persisted %d memories to SQLite", items_saved)
    except Exception as e:
        logger.warning("Failed to persist memories: %s", e)


# ── Checkpointer ─────────────────────────────────────────────────────


async def get_checkpointer():
    """Get or create the singleton LangGraph checkpointer.

    Postgres (durable, shared across Cloud Run instances) when DATABASE_URL is
    set — replacing the per-instance, ephemeral /tmp SQLite checkpointer that
    couldn't survive instance recycling or load-balanced approval clicks. Falls
    back to SQLite (tests/local, no DATABASE_URL), then in-memory as a last resort.
    """
    global _checkpointer, _checkpoint_conn, _checkpoint_pool
    if _checkpointer is not None:
        return _checkpointer

    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            from psycopg.rows import dict_row
            from psycopg_pool import AsyncConnectionPool

            # psycopg uses the bare postgresql:// scheme (not the SQLAlchemy
            # +asyncpg form). AsyncPostgresSaver requires autocommit + dict rows.
            conninfo = database_url.replace("postgresql+asyncpg://", "postgresql://")
            _checkpoint_pool = AsyncConnectionPool(
                conninfo=conninfo,
                min_size=0,
                # Mirrors the store pool: max_size 10 (~20-conn/instance ceiling),
                # min_size=0 + max_idle so an idle instance releases all of its
                # connections instead of squatting them. See note on the store pool.
                max_size=10,
                max_idle=180.0,
                open=False,
                kwargs={"autocommit": True, "row_factory": dict_row, "prepare_threshold": 0},
            )
            await _checkpoint_pool.open()
            _checkpointer = AsyncPostgresSaver(_checkpoint_pool)
            await _checkpointer.setup()
            logger.info("Postgres checkpointer ready (durable, shared)")
            print("[memory] postgres checkpointer ready", flush=True)
            return _checkpointer
        except Exception as e:
            logger.warning("Postgres checkpointer unavailable, falling back: %s", e)
            print(f"[memory] postgres checkpointer failed: {type(e).__name__}: {e}", flush=True)

    try:
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        _checkpoint_conn = await aiosqlite.connect(CHECKPOINT_DB_PATH)
        _checkpointer = AsyncSqliteSaver(_checkpoint_conn)
        await _checkpointer.setup()
        logger.info("SQLite checkpointer ready: %s", CHECKPOINT_DB_PATH)
    except Exception as e:
        logger.warning(
            "SQLite checkpointer unavailable, falling back to in-memory: %s", e
        )
        from langgraph.checkpoint.memory import MemorySaver

        _checkpointer = MemorySaver()
    return _checkpointer


# ── LangMem Memory Tools ─────────────────────────────────────────────


async def create_memory_tools(user_id: str) -> list:
    """Create LangMem memory tools for all three memory tiers."""
    from langmem import create_manage_memory_tool, create_search_memory_tool

    store = await get_memory_store()
    return [
        # User memory — personal preferences, per-individual
        create_manage_memory_tool(
            namespace=USER_NAMESPACE,
            name="manage_memory",
            instructions=(
                "Save personal facts about this specific user: preferences, "
                "communication style, role, and individual context. "
                "Update existing memories when information changes rather than "
                "creating duplicates. Delete outdated memories."
            ),
            store=store,
        ),
        create_search_memory_tool(
            namespace=USER_NAMESPACE,
            name="search_memory",
            store=store,
        ),
        # Core memory — operational knowledge, shared with ALL users
        create_manage_memory_tool(
            namespace=CORE_NAMESPACE,
            name="manage_core_memory",
            instructions=(
                "Save operational knowledge about how this bot works effectively: "
                "successful tool call patterns, troubleshooting recipes, workflow tips, "
                "and error resolution strategies. These are available to ALL users. "
                "Do NOT save user-specific preferences here."
            ),
            store=store,
        ),
        create_search_memory_tool(
            namespace=CORE_NAMESPACE,
            name="search_core_memory",
            store=store,
        ),
        # Team memory — VirtualDojo internal knowledge, team only
        create_manage_memory_tool(
            namespace=TEAM_NAMESPACE,
            name="manage_team_memory",
            instructions=(
                "Save VirtualDojo team-specific knowledge: project decisions, "
                "infrastructure facts, internal processes, team conventions. "
                "Do NOT save personal preferences (use manage_memory) or generic "
                "operational knowledge (use manage_core_memory)."
            ),
            store=store,
        ),
        create_search_memory_tool(
            namespace=TEAM_NAMESPACE,
            name="search_team_memory",
            store=store,
        ),
    ]


# ── Background Memory Extraction ─────────────────────────────────────


async def get_background_extractor():
    """Get the singleton background memory extractor.

    Automatically extracts and consolidates memories from conversations
    without the agent needing to explicitly call save_memory.
    """
    global _background_executor
    if _background_executor is None:
        from langmem import create_memory_store_manager, ReflectionExecutor

        store = await get_memory_store()
        manager = create_memory_store_manager(
            _create_extractor_llm(),
            namespace=USER_NAMESPACE,
            store=store,
            enable_inserts=True,
            enable_deletes=True,
            instructions=(
                "Extract PERSONAL facts about this specific user: preferences, "
                "communication style, role-specific context, and individual work patterns. "
                "Update existing memories if information has changed. "
                "Delete memories that are contradicted by new information. "
                "Do NOT save team-level knowledge or general operational patterns here — "
                "those are handled by separate extractors. "
                "Do NOT save trivial information like greetings or routine status checks."
            ),
        )
        # LangMem requires store= on ReflectionExecutor explicitly even when
        # the underlying manager already has one — without this every
        # [memory.extract] call fails with "ReflectionExecutor could not
        # resolve store to persist memories to".
        _background_executor = ReflectionExecutor(manager, store=store)
        logger.info("Background memory extractor ready")
    return _background_executor


async def get_core_extractor():
    """Get the singleton background extractor for core operational knowledge.

    Core memories are shared across ALL users (including future external users).
    They capture reusable tool patterns, troubleshooting recipes, and workflow knowledge.
    """
    global _core_executor
    if _core_executor is None:
        from langmem import create_memory_store_manager, ReflectionExecutor

        store = await get_memory_store()
        manager = create_memory_store_manager(
            _create_extractor_llm(),
            namespace=CORE_NAMESPACE,
            store=store,
            enable_inserts=True,
            enable_deletes=True,
            instructions=(
                "Extract OPERATIONAL KNOWLEDGE from this conversation that would help "
                "any user of this bot work more effectively. Focus on:\n"
                "- Successful tool call patterns and sequences (e.g., 'To check autofix "
                "status, search PRs for branches matching bugfix/issue-{N}')\n"
                "- Troubleshooting recipes (e.g., 'Gemini 429 errors are transient, "
                "retry usually succeeds')\n"
                "- Workflow knowledge (e.g., 'After deploying, check new revision logs "
                "not old revision')\n"
                "- Error patterns and their solutions\n"
                "- API behaviors and quirks discovered through tool usage\n\n"
                "Do NOT extract:\n"
                "- User preferences or personal details\n"
                "- Transient information (greetings, routine status checks)\n"
                "- Information that is only relevant to a specific user\n"
                "- Raw data or logs (extract the LESSON, not the data)\n\n"
                "Write memories as reusable recipes or facts that would help a future "
                "agent session solve similar problems faster."
            ),
        )
        _core_executor = ReflectionExecutor(manager, store=store)
        logger.info("Core memory extractor ready")
    return _core_executor


async def get_team_extractor():
    """Get the singleton background extractor for team knowledge.

    Team memories are shared within VirtualDojo but NOT with external users.
    They capture project decisions, infrastructure facts, and internal processes.
    """
    global _team_executor
    if _team_executor is None:
        from langmem import create_memory_store_manager, ReflectionExecutor

        store = await get_memory_store()
        manager = create_memory_store_manager(
            _create_extractor_llm(),
            namespace=TEAM_NAMESPACE,
            store=store,
            enable_inserts=True,
            enable_deletes=True,
            instructions=(
                "Extract TEAM KNOWLEDGE from this conversation that is specific to "
                "the VirtualDojo team and its projects. Focus on:\n"
                "- Project decisions and architecture choices\n"
                "- Infrastructure facts (e.g., 'virtualdojo-inc/virtualdojo uses "
                "claude_automation/bugfix for autofix')\n"
                "- Internal processes (e.g., 'FedRAMP evidence collection runs monthly')\n"
                "- Team conventions and workflows\n"
                "- Service configurations and deployment patterns\n"
                "- Repository structure and branch strategies\n\n"
                "Do NOT extract:\n"
                "- Individual user preferences (those go to user memory)\n"
                "- Generic operational knowledge not specific to VirtualDojo\n"
                "- Transient information (greetings, routine status checks)\n\n"
                "Write memories as team-level facts that any VirtualDojo team member "
                "would benefit from knowing."
            ),
        )
        _team_executor = ReflectionExecutor(manager, store=store)
        logger.info("Team memory extractor ready")
    return _team_executor


# ── Auto-Retrieval for System Prompt Injection ────────────────────────


def _format_memory(r) -> str:
    """Format a single memory search result as a string."""
    val = r.value
    if isinstance(val, dict):
        content = val.get("content", json.dumps(val))
    else:
        content = str(val)
    return f"- {content}"


async def retrieve_relevant_memories(user_id: str, query: str) -> str | None:
    """Auto-retrieve relevant memories from all three tiers for system prompt injection.

    Searches core (operational), team (VirtualDojo), and user (personal) namespaces.
    Returns a formatted string with labeled sections, or None if nothing found.
    """
    try:
        store = await get_memory_store()

        core_results = await store.asearch(CORE_NAMESPACE, query=query, limit=3)
        team_results = await store.asearch(TEAM_NAMESPACE, query=query, limit=3)
        user_results = await store.asearch(("memories", user_id), query=query, limit=3)

        sections = []

        if core_results:
            lines = [_format_memory(r) for r in core_results]
            sections.append("Operational knowledge:\n" + "\n".join(lines))

        if team_results:
            lines = [_format_memory(r) for r in team_results]
            sections.append("Team knowledge:\n" + "\n".join(lines))

        if user_results:
            lines = [_format_memory(r) for r in user_results]
            sections.append("Personal context:\n" + "\n".join(lines))

        # Troubleshooting patterns live in their own namespace with structured
        # fields; dedicated retriever handles formatting + retrieval counting.
        ts_count = 0
        try:
            from tools.troubleshooting import retrieve_troubleshooting_patterns

            ts = await retrieve_troubleshooting_patterns(query, limit=3)
            if ts:
                sections.append(ts)
                # The formatter emits one leading "- " bullet per match.
                ts_count = ts.count("\n- ") + (1 if ts.startswith("- ") else 0)
                # Fallback: count lines that start with "- " after the header.
                if ts_count == 0:
                    ts_count = sum(1 for line in ts.splitlines() if line.startswith("- "))
        except Exception as e:
            logger.debug("Troubleshooting retrieval failed: %s", e)

        # Single-line observability: what landed in the system prompt this turn.
        # Visible in Cloud Logging so we can see retrieval quality in real time.
        print(
            f"[memory] retrieved core={len(core_results)} team={len(team_results)} "
            f"user={len(user_results)} troubleshooting={ts_count} "
            f"query={query[:80]!r} user_id={user_id!r}",
            flush=True,
        )

        if not sections:
            return None

        return "Relevant context from memory:\n\n" + "\n\n".join(sections)
    except Exception as e:
        logger.debug("Memory retrieval failed: %s", e)
        return None
