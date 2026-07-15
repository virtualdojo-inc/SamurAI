"""Tests for app.py — aiohttp server and Bot Framework handlers."""

import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web


@pytest.fixture
def patched_app():
    """Import app.py with Bot Framework adapter and agent mocked."""
    with (
        patch("langchain_google_genai.ChatGoogleGenerativeAI", MagicMock()),
        patch("memory.get_checkpointer", new_callable=AsyncMock),
        patch("memory.retrieve_relevant_memories", new_callable=AsyncMock, return_value=None),
        patch("memory.create_memory_tools", return_value=[]),
        patch("botbuilder.core.BotFrameworkAdapter") as mock_adapter_cls,
    ):
        mock_adapter = MagicMock()
        # Real BotFrameworkAdapter.process_activity returns None for message
        # activities (only invoke activities yield an InvokeResponse).
        mock_adapter.process_activity = AsyncMock(return_value=None)
        mock_adapter_cls.return_value = mock_adapter

        import app as app_module
        importlib.reload(app_module)
        app_module.adapter = mock_adapter
        yield app_module


@pytest.fixture
async def client(patched_app, aiohttp_client):
    return await aiohttp_client(patched_app.app)


@pytest.mark.asyncio
async def test_health_endpoint(client):
    resp = await client.get("/health")
    assert resp.status == 200
    text = await resp.text()
    assert text == "ok"


@pytest.mark.asyncio
async def test_messages_returns_415_for_non_json(client):
    resp = await client.post(
        "/api/messages",
        data="not json",
        headers={"Content-Type": "text/plain"},
    )
    assert resp.status == 415


@pytest.mark.asyncio
async def test_messages_returns_200_for_valid_json(client, patched_app):
    resp = await client.post(
        "/api/messages",
        json={"type": "message", "text": "hello"},
        headers={"Content-Type": "application/json"},
    )
    assert resp.status == 200
    patched_app.adapter.process_activity.assert_called_once()


# --- Unit tests for handler functions ---


def _make_turn_context(text="show logs", email="devin@virtualdojo.com"):
    """Create a mock TurnContext with a virtualdojo.com user."""
    ctx = MagicMock()
    ctx.activity.text = text
    ctx.activity.from_property.id = "user-123"
    ctx.activity.from_property.name = "Test User"
    ctx.activity.conversation.id = "conv-123"
    ctx.activity.local_timezone = None
    ctx.activity.service_url = "https://smba.trafficmanager.net/teams/"
    ctx.activity.channel_data = None
    ctx.activity.value = None
    ctx.send_activity = AsyncMock()

    # Mock TeamsInfo to return the email
    member = MagicMock()
    member.email = email
    member.user_principal_name = email
    member.id = "user-123"
    member.name = "Test User"
    return ctx, member


@pytest.mark.asyncio
async def test_on_message_calls_run_agent(patched_app):
    ctx, member = _make_turn_context("show logs")

    with (
        patch.object(patched_app, "run_agent", new_callable=AsyncMock, return_value="here are logs"),
        patch("botbuilder.core.teams.TeamsInfo.get_member", new_callable=AsyncMock, return_value=member),
        patch("task_store.get_task_store", new_callable=AsyncMock),
    ):
        await patched_app.on_message(ctx)

    # Should have sent at least typing + response
    sent_types = [call[0][0].type if hasattr(call[0][0], 'type') else 'text' for call in ctx.send_activity.call_args_list]
    assert "message" in sent_types


@pytest.mark.asyncio
async def test_on_message_sends_typing_indicator(patched_app):
    import asyncio

    ctx, member = _make_turn_context("hi")

    async def slow_agent(*args, **kwargs):
        await asyncio.sleep(0.1)  # Give typing task time to fire
        return "hey"

    with (
        patch.object(patched_app, "run_agent", side_effect=slow_agent),
        patch("botbuilder.core.teams.TeamsInfo.get_member", new_callable=AsyncMock, return_value=member),
        patch("task_store.get_task_store", new_callable=AsyncMock),
    ):
        await patched_app.on_message(ctx)

    # At least one typing indicator should have been sent
    sent_types = [call[0][0].type if hasattr(call[0][0], 'type') else 'text' for call in ctx.send_activity.call_args_list]
    assert "typing" in sent_types


@pytest.mark.asyncio
async def test_on_message_blocks_non_virtualdojo_user(patched_app):
    ctx, member = _make_turn_context("show logs", email="outsider@gmail.com")

    with (
        patch.object(patched_app, "run_agent", new_callable=AsyncMock) as mock_agent,
        patch("botbuilder.core.teams.TeamsInfo.get_member", new_callable=AsyncMock, return_value=member),
    ):
        await patched_app.on_message(ctx)

    mock_agent.assert_not_called()
    sent_texts = [
        str(call[0][0].text) if hasattr(call[0][0], 'text') else str(call[0][0])
        for call in ctx.send_activity.call_args_list
        if hasattr(call[0][0], 'type') and call[0][0].type == "message"
    ]
    assert any("VirtualDojo" in t for t in sent_texts)


@pytest.mark.asyncio
async def test_reset_command_clears_thread_and_skips_agent(patched_app):
    """'reset' must actually wipe the conversation checkpoint and NOT run the
    agent (the bug was it only claimed to reset)."""
    ctx, member = _make_turn_context("reset")

    with (
        patch.object(patched_app, "run_agent", new_callable=AsyncMock) as mock_agent,
        patch("botbuilder.core.teams.TeamsInfo.get_member", new_callable=AsyncMock, return_value=member),
        patch("task_store.get_task_store", new_callable=AsyncMock),
        patch("memory.clear_thread", new_callable=AsyncMock, return_value=True) as mock_clear,
    ):
        await patched_app.on_message(ctx)

    mock_clear.assert_awaited_once_with("conv-123")
    mock_agent.assert_not_called()
    sent_texts = [
        str(call[0][0]) for call in ctx.send_activity.call_args_list
    ]
    assert any("cleared" in t.lower() for t in sent_texts)


@pytest.mark.asyncio
async def test_normal_message_does_not_trigger_reset(patched_app):
    """A message that merely mentions reset must NOT wipe the thread."""
    ctx, member = _make_turn_context("can you reset the staging database counter?")

    with (
        patch.object(patched_app, "run_agent", new_callable=AsyncMock, return_value="ok"),
        patch("botbuilder.core.teams.TeamsInfo.get_member", new_callable=AsyncMock, return_value=member),
        patch("task_store.get_task_store", new_callable=AsyncMock),
        patch("memory.clear_thread", new_callable=AsyncMock) as mock_clear,
    ):
        await patched_app.on_message(ctx)

    mock_clear.assert_not_called()


# --- Group-chat support ---


def test_clean_user_text_strips_bot_mention(patched_app):
    from botbuilder.schema import Activity

    # Built the way Teams actually sends a group-chat @mention (deserialized JSON).
    act = Activity().deserialize({
        "type": "message",
        "text": "<at>SamurAI</at> show logs",
        "recipient": {"id": "bot-1", "name": "SamurAI"},
        "entities": [{
            "type": "mention",
            "text": "<at>SamurAI</at>",
            "mentioned": {"id": "bot-1", "name": "SamurAI"},
        }],
    })
    assert patched_app._clean_user_text(act) == "show logs"


def test_clean_user_text_falls_back_on_plain_text(patched_app):
    act = MagicMock()  # mock entities aren't iterable -> falls back to .text
    act.text = "show logs"
    assert patched_app._clean_user_text(act) == "show logs"


def test_is_group_scope(patched_app):
    def _act(ct):
        a = MagicMock()
        a.conversation.conversation_type = ct
        return a
    assert patched_app._is_group_scope(_act("groupChat")) is True
    assert patched_app._is_group_scope(_act("channel")) is True
    assert patched_app._is_group_scope(_act("personal")) is False


@pytest.mark.asyncio
async def test_on_message_silent_in_group_for_unauthorized(patched_app):
    """In a group chat, an unauthorized sender gets no reply (no noise)."""
    ctx, member = _make_turn_context("<at>SamurAI</at> show logs", email="guest@gmail.com")
    ctx.activity.conversation.conversation_type = "groupChat"
    with (
        patch.object(patched_app, "run_agent", new_callable=AsyncMock) as mock_agent,
        patch("botbuilder.core.teams.TeamsInfo.get_member", new_callable=AsyncMock, return_value=member),
    ):
        await patched_app.on_message(ctx)
    mock_agent.assert_not_called()
    msg_texts = [
        call[0][0].text for call in ctx.send_activity.call_args_list
        if hasattr(call[0][0], "type") and call[0][0].type == "message"
    ]
    assert msg_texts == []  # silent in a group — no rejection posted


# --- Image / screenshot intake (gated by SAMURAI_VISION_ENABLED) ---


class _FakeResp:
    def __init__(self, content=b""):
        self.content = content

    def raise_for_status(self):
        pass


class _FakeClient:
    def __init__(self, resp_or_exc):
        self._r = resp_or_exc

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, headers=None):
        if isinstance(self._r, Exception):
            raise self._r
        return self._r


def _img_att(content_type="image/png", url="https://cdn.teams/img.png", name="shot.png"):
    att = MagicMock()
    att.name = name
    att.content_type = content_type
    att.content_url = url
    att.content = None
    return att


@pytest.mark.asyncio
async def test_ingest_image_success(patched_app, monkeypatch):
    import base64
    monkeypatch.setattr("httpx.AsyncClient", lambda *a, **k: _FakeClient(_FakeResp(b"PNGBYTES")))
    parts = []
    note = await patched_app._ingest_image_attachment(_img_att(), parts)
    assert note == ""
    assert len(parts) == 1
    assert parts[0]["mime_type"] == "image/png"
    assert base64.b64decode(parts[0]["data"]) == b"PNGBYTES"


@pytest.mark.asyncio
async def test_ingest_image_wildcard_sniffs_png(patched_app, monkeypatch):
    """Teams pasted/inline images arrive as content_type 'image/*'; the real type
    is sniffed from the bytes. Regression test for the paste-a-screenshot bug."""
    import base64
    png = b"\x89PNG\r\n\x1a\n" + b"rest-of-the-file"
    monkeypatch.setattr("httpx.AsyncClient", lambda *a, **k: _FakeClient(_FakeResp(png)))
    parts = []
    note = await patched_app._ingest_image_attachment(_img_att(content_type="image/*"), parts)
    assert note == ""
    assert len(parts) == 1
    assert parts[0]["mime_type"] == "image/png"
    assert base64.b64decode(parts[0]["data"]) == png


@pytest.mark.asyncio
async def test_ingest_image_wildcard_unrecognized_bytes_skipped(patched_app, monkeypatch):
    monkeypatch.setattr("httpx.AsyncClient", lambda *a, **k: _FakeClient(_FakeResp(b"not-an-image")))
    parts = []
    note = await patched_app._ingest_image_attachment(_img_att(content_type="image/*"), parts)
    assert "unsupported" in note.lower()
    assert parts == []


@pytest.mark.asyncio
async def test_ingest_image_unsupported_skipped(patched_app):
    parts = []
    note = await patched_app._ingest_image_attachment(_img_att(content_type="image/heic"), parts)
    assert "unsupported" in note.lower()
    assert parts == []


@pytest.mark.asyncio
async def test_ingest_image_respects_cap(patched_app):
    parts = [{"mime_type": "image/png", "data": "x"}] * patched_app._MAX_IMAGES
    note = await patched_app._ingest_image_attachment(_img_att(), parts)
    assert "only the first" in note.lower()
    assert len(parts) == patched_app._MAX_IMAGES  # unchanged


@pytest.mark.asyncio
async def test_ingest_image_download_error_is_graceful(patched_app, monkeypatch):
    monkeypatch.setattr("httpx.AsyncClient", lambda *a, **k: _FakeClient(RuntimeError("403 Forbidden")))
    parts = []
    note = await patched_app._ingest_image_attachment(_img_att(), parts)
    assert "couldn't fetch" in note.lower()
    assert parts == []  # no crash, nothing appended


@pytest.mark.asyncio
async def test_ingest_image_too_large_skipped(patched_app, monkeypatch):
    big = b"x" * (patched_app._MAX_IMAGE_BYTES + 1)
    monkeypatch.setattr("httpx.AsyncClient", lambda *a, **k: _FakeClient(_FakeResp(big)))
    parts = []
    note = await patched_app._ingest_image_attachment(_img_att(), parts)
    assert "too large" in note.lower()
    assert parts == []


@pytest.mark.asyncio
async def test_on_message_blocks_user_with_no_email(patched_app):
    ctx, member = _make_turn_context("show logs", email="")

    with (
        patch.object(patched_app, "run_agent", new_callable=AsyncMock) as mock_agent,
        patch("botbuilder.core.teams.TeamsInfo.get_member", new_callable=AsyncMock, side_effect=Exception("no email")),
    ):
        await patched_app.on_message(ctx)

    mock_agent.assert_not_called()


@pytest.mark.asyncio
async def test_on_message_ignores_empty_text(patched_app):
    ctx = MagicMock()
    ctx.activity.text = None
    ctx.send_activity = AsyncMock()

    with patch.object(patched_app, "run_agent", new_callable=AsyncMock) as mock_agent:
        await patched_app.on_message(ctx)

    mock_agent.assert_not_called()
    ctx.send_activity.assert_not_called()


@pytest.mark.asyncio
async def test_on_error_sends_apology(patched_app):
    ctx = MagicMock()
    ctx.send_activity = AsyncMock()

    await patched_app.on_error(ctx, Exception("boom"))
    ctx.send_activity.assert_called_once()
    msg = ctx.send_activity.call_args[0][0]
    assert "something went wrong" in msg.lower() or "something went wrong" in str(msg).lower()


# --- Error-handler resilience + turn timeout (2026-07 log fixes) ---


@pytest.mark.asyncio
async def test_on_error_survives_dead_connection(patched_app):
    """If the gateway already canceled the request, the fallback reply itself
    throws — on_error must swallow that instead of turning it into a 500."""
    ctx = MagicMock()
    ctx.send_activity = AsyncMock(side_effect=Exception("A task was canceled."))
    await patched_app.on_error(ctx, Exception("boom"))  # must not raise


@pytest.mark.asyncio
async def test_on_message_times_out_gracefully(patched_app, monkeypatch):
    import asyncio

    ctx, member = _make_turn_context("investigate everything")
    monkeypatch.setattr(patched_app, "_TURN_TIMEOUT_S", 0.05)

    async def never_finishes(*args, **kwargs):
        await asyncio.sleep(5)
        return "too late"

    with (
        patch.object(patched_app, "run_agent", side_effect=never_finishes),
        patch("botbuilder.core.teams.TeamsInfo.get_member", new_callable=AsyncMock, return_value=member),
        patch("task_store.get_task_store", new_callable=AsyncMock),
    ):
        await patched_app.on_message(ctx)

    # The timeout path sends a plain string (not an Activity).
    sent_texts = [
        str(getattr(call[0][0], "text", None) or call[0][0])
        for call in ctx.send_activity.call_args_list
    ]
    assert any("taking longer" in t for t in sent_texts)
    assert not any("too late" in t for t in sent_texts)


@pytest.mark.asyncio
async def test_roster_persist_does_not_block_reply(patched_app):
    """The conversation-ref/roster persistence runs as a background task; a
    hanging store must not delay (or fail) the user's reply."""
    import asyncio

    ctx, member = _make_turn_context("hello")
    store_started = asyncio.Event()

    async def _hanging_store(*args, **kwargs):
        store_started.set()
        await asyncio.sleep(30)

    with (
        patch.object(patched_app, "run_agent", new_callable=AsyncMock, return_value="done"),
        patch("botbuilder.core.teams.TeamsInfo.get_member", new_callable=AsyncMock, return_value=member),
        patch("task_store.get_task_store", side_effect=_hanging_store),
    ):
        await asyncio.wait_for(patched_app.on_message(ctx), timeout=5)

    sent_texts = [
        str(call[0][0].text) if hasattr(call[0][0], "text") else ""
        for call in ctx.send_activity.call_args_list
    ]
    assert any("done" in t for t in sent_texts)  # reply arrived despite hung store
    for t in list(patched_app._background_tasks):
        t.cancel()
