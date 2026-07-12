"""VirtualDojo MCP client — OAuth SSO + dynamic CRM tool calling."""

import asyncio
import hashlib
import base64
import secrets
import os
import time
import logging

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token store — per-user tokens keyed by Teams user ID
# In production, swap this for Redis or a database.
# ---------------------------------------------------------------------------
_token_store: dict[str, dict] = {}
# Pending OAuth flows keyed by state param
_pending_auth: dict[str, dict] = {}
# Registered client credentials (populated on first use)
_client_creds: dict | None = None

MCP_URL = os.environ.get("VIRTUALDOJO_MCP_URL", "https://dev.virtualdojo.com/mcp/v1")
BOT_CALLBACK_URL = os.environ.get(
    "BOT_CALLBACK_URL",
    "https://samurai-bot-1019610148219.us-central1.run.app/api/oauth/callback",
)

MAX_RETRIES = 2
RETRY_DELAY = 1.0  # seconds
CIRCUIT_BREAKER_COOLDOWN = 60  # seconds before retrying after repeated failures
CIRCUIT_BREAKER_THRESHOLD = 3  # consecutive failures to trip the breaker


# ---------------------------------------------------------------------------
# Circuit breaker — stop calling MCP if it's down, reset after cooldown
# ---------------------------------------------------------------------------

class _CircuitBreaker:
    def __init__(self, threshold: int = CIRCUIT_BREAKER_THRESHOLD, cooldown: float = CIRCUIT_BREAKER_COOLDOWN):
        self.threshold = threshold
        self.cooldown = cooldown
        self.failure_count = 0
        self.last_failure_time: float = 0
        self.open = False

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.threshold:
            self.open = True
            logger.warning(
                f"Circuit breaker OPEN — VirtualDojo MCP server failed "
                f"{self.failure_count} times. Will retry after {self.cooldown}s."
            )

    def record_success(self):
        if self.failure_count > 0:
            logger.info("Circuit breaker RESET — VirtualDojo MCP server recovered.")
        self.failure_count = 0
        self.open = False

    def is_available(self) -> bool:
        if not self.open:
            return True
        # Check if cooldown has elapsed
        elapsed = time.time() - self.last_failure_time
        if elapsed >= self.cooldown:
            logger.info(
                f"Circuit breaker HALF-OPEN — {elapsed:.0f}s since last failure, "
                f"allowing one request through."
            )
            return True
        return False

    def time_until_retry(self) -> float:
        if not self.open:
            return 0
        remaining = self.cooldown - (time.time() - self.last_failure_time)
        return max(0, remaining)


_circuit = _CircuitBreaker()


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

async def _request_with_retry(
    method: str,
    url: str,
    retries: int = MAX_RETRIES,
    timeout: float = 15,
    **kwargs,
) -> httpx.Response:
    """Make an HTTP request with retries and circuit breaker protection."""
    if not _circuit.is_available():
        remaining = _circuit.time_until_retry()
        raise ConnectionError(
            f"VirtualDojo CRM is temporarily unavailable. Will retry in {remaining:.0f}s."
        )

    last_error = None
    for attempt in range(retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.request(method, url, **kwargs)
                # Retry on 5xx server errors and 429 rate limits
                if resp.status_code >= 500 or resp.status_code == 429:
                    _circuit.record_failure()
                    if attempt < retries:
                        delay = RETRY_DELAY * (2 ** attempt)
                        logger.warning(
                            f"MCP request {method} {url} returned {resp.status_code}, "
                            f"retrying in {delay}s (attempt {attempt + 1}/{retries})"
                        )
                        await asyncio.sleep(delay)
                        continue
                    return resp
                # Success — reset circuit breaker
                _circuit.record_success()
                return resp
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
            last_error = e
            _circuit.record_failure()
            if attempt < retries:
                delay = RETRY_DELAY * (2 ** attempt)
                logger.warning(
                    f"MCP request {method} {url} failed with {type(e).__name__}, "
                    f"retrying in {delay}s (attempt {attempt + 1}/{retries})"
                )
                await asyncio.sleep(delay)
            else:
                raise
    raise last_error


# ---------------------------------------------------------------------------
# OAuth helpers
# ---------------------------------------------------------------------------

def _generate_pkce() -> tuple[str, str]:
    """Generate PKCE code_verifier and code_challenge (S256)."""
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge


async def _ensure_client_registered() -> dict:
    """Register this bot as an OAuth client with the MCP server (once)."""
    global _client_creds
    if _client_creds:
        return _client_creds

    try:
        resp = await _request_with_retry(
            "POST",
            f"{MCP_URL}/oauth/register",
            json={
                "client_name": "SamurAI Teams Bot",
                "redirect_uris": [BOT_CALLBACK_URL],
            },
        )
        resp.raise_for_status()
        _client_creds = resp.json()
        return _client_creds
    except Exception as e:
        logger.error(f"Failed to register with VirtualDojo MCP server: {e}")
        raise


async def start_oauth_flow(user_id: str) -> tuple[str, str]:
    """Start the OAuth flow and return (authorize_url, state)."""
    creds = await _ensure_client_registered()
    verifier, challenge = _generate_pkce()
    state = secrets.token_urlsafe(32)

    _pending_auth[state] = {
        "user_id": user_id,
        "code_verifier": verifier,
        "client_id": creds["client_id"],
    }

    params = {
        "client_id": creds["client_id"],
        "redirect_uri": BOT_CALLBACK_URL,
        "response_type": "code",
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "scope": "tools",
        "state": state,
    }
    query = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{MCP_URL}/oauth/authorize?{query}", state


async def exchange_code(code: str, state: str) -> dict | None:
    """Exchange an OAuth authorization code for tokens."""
    flow = _pending_auth.pop(state, None)
    if not flow:
        print(f"[oauth] exchange_code: NO pending auth for state={state[:8]}..., pending_keys={list(_pending_auth.keys())[:5]}", flush=True)
        return None
    print(f"[oauth] exchange_code: found flow for user_id={flow['user_id']}", flush=True)

    creds = await _ensure_client_registered()
    try:
        resp = await _request_with_retry(
            "POST",
            f"{MCP_URL}/oauth/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": BOT_CALLBACK_URL,
                "client_id": creds["client_id"],
                "code_verifier": flow["code_verifier"],
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        resp.raise_for_status()
        tokens = resp.json()
    except Exception as e:
        logger.error(f"Failed to exchange OAuth code: {e}")
        return None

    _token_store[flow["user_id"]] = {
        "access_token": tokens["access_token"],
        "refresh_token": tokens.get("refresh_token"),
        "expires_at": time.time() + tokens.get("expires_in", 1800),
    }
    print(f"[oauth] Token stored for user_id={flow['user_id']}, store_keys={list(_token_store.keys())}", flush=True)
    return tokens


def _is_token_valid(user_id: str) -> bool:
    """Check if a user's token is still valid (with 60s buffer)."""
    t = _token_store.get(user_id)
    if not t:
        return False
    return t["expires_at"] > time.time() + 60


async def _get_access_token(user_id: str) -> str | None:
    """Get a valid access token for the user, refreshing if needed."""
    t = _token_store.get(user_id)
    if not t:
        return None

    if _is_token_valid(user_id):
        return t["access_token"]

    # Try refresh
    if not t.get("refresh_token"):
        del _token_store[user_id]
        return None

    creds = await _ensure_client_registered()
    try:
        resp = await _request_with_retry(
            "POST",
            f"{MCP_URL}/oauth/token",
            data={
                "grant_type": "refresh_token",
                "refresh_token": t["refresh_token"],
                "client_id": creds["client_id"],
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        resp.raise_for_status()
        tokens = resp.json()

        _token_store[user_id] = {
            "access_token": tokens["access_token"],
            "refresh_token": tokens.get("refresh_token", t["refresh_token"]),
            "expires_at": time.time() + tokens.get("expires_in", 1800),
        }
        return tokens["access_token"]
    except Exception as e:
        logger.error(f"Failed to refresh token for user {user_id}: {e}")
        del _token_store[user_id]
        return None


def is_user_authenticated(user_id: str) -> bool:
    """Check if a user has stored tokens."""
    result = user_id in _token_store
    print(f"[oauth] is_user_authenticated({user_id[:20]}...)={result}, store_keys={list(_token_store.keys())[:3]}", flush=True)
    return result


# ---------------------------------------------------------------------------
# MCP tool discovery and execution
# ---------------------------------------------------------------------------

async def list_mcp_tools(user_id: str) -> list[dict]:
    """List all MCP tools available to the authenticated user."""
    token = await _get_access_token(user_id)
    if not token:
        return []

    try:
        resp = await _request_with_retry(
            "POST",
            f"{MCP_URL}/tools/list",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("tools", [])
    except Exception as e:
        logger.error(f"Failed to list MCP tools: {e}")
        return []


async def call_mcp_tool(user_id: str, tool_name: str, arguments: dict) -> dict:
    """Call an MCP tool on behalf of the authenticated user."""
    token = await _get_access_token(user_id)
    if not token:
        return {"error": "Not authenticated. Please sign in to VirtualDojo first."}

    try:
        resp = await _request_with_retry(
            "POST",
            f"{MCP_URL}/tools/call",
            json={"name": tool_name, "arguments": arguments},
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            timeout=60,
        )
        if resp.status_code == 401:
            # Token may be invalid — clear it so user re-authenticates
            _token_store.pop(user_id, None)
            return {"error": "Your VirtualDojo session expired. Please sign in again."}
        if resp.status_code == 403:
            return {"error": "You don't have permission to use this CRM tool. Check your VirtualDojo profile permissions."}
        resp.raise_for_status()
        return resp.json()
    except ConnectionError as e:
        # Circuit breaker is open
        return {"error": str(e)}
    except httpx.TimeoutException:
        return {"error": "The CRM request timed out. The VirtualDojo server may be slow — please try again."}
    except httpx.ConnectError:
        return {"error": "Could not connect to VirtualDojo CRM. The service may be temporarily unavailable."}
    except Exception as e:
        logger.error(f"MCP tool call '{tool_name}' failed: {e}")
        return {"error": f"CRM request failed: {str(e)}"}


# ---------------------------------------------------------------------------
# LangGraph tool wrappers
# ---------------------------------------------------------------------------

class VirtualDojoQueryInput(BaseModel):
    tool_name: str = Field(description="The MCP tool name to call (e.g., 'search_records', 'list_objects', 'describe_object', 'create_record')")
    arguments: str = Field(description='JSON string of arguments to pass to the tool (e.g., \'{"object_type": "contacts", "limit": 10}\')')


def create_virtualdojo_tool(user_id: str) -> StructuredTool:
    """Create a LangGraph-compatible tool that calls VirtualDojo MCP tools for a specific user."""

    async def _call_virtualdojo(tool_name: str, arguments: str) -> str:
        import json
        try:
            args = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            return f"Invalid JSON arguments: {arguments}"

        if not is_user_authenticated(user_id):
            return (
                "You are not signed in to VirtualDojo. "
                "Please click the sign-in link I'll provide to connect your CRM account."
            )

        result = await call_mcp_tool(user_id, tool_name, args)
        if "error" in result:
            return f"Error: {result['error']}"

        # MCP returns content as a list of content blocks
        content = result.get("content", [])
        if isinstance(content, list):
            texts = [c.get("text", str(c)) for c in content if isinstance(c, dict)]
            return "\n".join(texts) if texts else str(result)
        return str(content)

    return StructuredTool.from_function(
        coroutine=_call_virtualdojo,
        name="virtualdojo_crm",
        description=(
            "Query the VirtualDojo CRM system. Use this for anything related to "
            "CRM data: contacts, accounts, opportunities, quotes, compliance records, etc. "
            "Common tool_name values: 'search_records', 'list_objects', 'describe_object', "
            "'create_record', 'update_record', 'get_record'. "
            "Pass arguments as a JSON string."
        ),
        args_schema=VirtualDojoQueryInput,
    )


class VirtualDojoListToolsInput(BaseModel):
    pass


def create_virtualdojo_list_tools(user_id: str) -> StructuredTool:
    """Create a tool that lists available VirtualDojo CRM tools for the user."""

    async def _list_tools() -> str:
        if not is_user_authenticated(user_id):
            return "You are not signed in to VirtualDojo. Please sign in first."

        tools = await list_mcp_tools(user_id)
        if not tools:
            return "No CRM tools available. Either VirtualDojo is unreachable or you may not have the right permissions."

        lines = []
        for t in tools:
            lines.append(f"- **{t['name']}**: {t.get('description', 'No description')}")
        return f"Available VirtualDojo CRM tools ({len(tools)}):\n" + "\n".join(lines)

    return StructuredTool.from_function(
        coroutine=_list_tools,
        name="virtualdojo_list_tools",
        description="List all available VirtualDojo CRM tools for the authenticated user.",
        args_schema=VirtualDojoListToolsInput,
    )
