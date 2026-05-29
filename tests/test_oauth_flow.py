"""Tests for OAuth flow — proactive messaging and auth injection into conversation history."""

import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def patched_app():
    """Import app.py with Bot Framework adapter and agent mocked."""
    with (
        patch("botbuilder.core.BotFrameworkAdapter") as mock_adapter_cls,
    ):
        mock_adapter = MagicMock()
        mock_adapter.process_activity = AsyncMock()
        mock_adapter.continue_conversation = AsyncMock()
        mock_adapter_cls.return_value = mock_adapter

        import app as app_module

        importlib.reload(app_module)
        app_module.adapter = mock_adapter
        yield app_module


@pytest.fixture
async def client(patched_app, aiohttp_client):
    return await aiohttp_client(patched_app.app)


# --- OAuth callback proactive messaging ---


@pytest.mark.asyncio
async def test_oauth_callback_sends_proactive_message(patched_app, client):
    """After OAuth callback, bot should proactively message the user in Teams."""
    from botbuilder.schema import ConversationReference

    conv_ref = ConversationReference(
        channel_id="msteams",
        service_url="https://smba.trafficmanager.net/teams/",
        conversation=MagicMock(id="conv-123"),
    )

    # Simulate storing a conversation reference for this OAuth state
    patched_app._oauth_conversation_refs["test-state-123"] = {
        "conv_ref": conv_ref,
        "user_id": "user-abc",
        "conversation_id": "conv-123",
    }

    with (
        patch.object(
            patched_app,
            "exchange_code",
            new_callable=AsyncMock,
            return_value={"access_token": "tok", "expires_in": 1800},
        ),
        patch.object(
            patched_app,
            "inject_auth_message",
            new_callable=AsyncMock,
        ) as mock_inject,
    ):
        resp = await client.get(
            "/api/oauth/callback?code=test-code&state=test-state-123"
        )

    assert resp.status == 200
    html = await resp.text()
    assert "Connected to VirtualDojo" in html

    # inject_auth_message should have been called
    mock_inject.assert_called_once_with("user-abc", "conv-123")

    # Proactive message should have been sent
    patched_app.adapter.continue_conversation.assert_called_once()


@pytest.mark.asyncio
async def test_oauth_callback_no_conv_ref_still_succeeds(patched_app, client):
    """If no conversation ref stored (e.g. expired), callback still succeeds."""
    with patch.object(
        patched_app,
        "exchange_code",
        new_callable=AsyncMock,
        return_value={"access_token": "tok", "expires_in": 1800},
    ):
        resp = await client.get(
            "/api/oauth/callback?code=test-code&state=unknown-state"
        )

    assert resp.status == 200
    html = await resp.text()
    assert "Connected to VirtualDojo" in html
    # No proactive message since no conv ref
    patched_app.adapter.continue_conversation.assert_not_called()


@pytest.mark.asyncio
async def test_oauth_callback_exchange_fails(patched_app, client):
    """If token exchange fails, show error page."""
    with patch.object(
        patched_app,
        "exchange_code",
        new_callable=AsyncMock,
        return_value=None,
    ):
        resp = await client.get("/api/oauth/callback?code=bad-code&state=bad-state")

    assert resp.status == 200
    html = await resp.text()
    assert "Authentication failed" in html


@pytest.mark.asyncio
async def test_oauth_callback_error_param(client):
    """If error query param present, show error page."""
    resp = await client.get("/api/oauth/callback?error=access_denied")
    assert resp.status == 200
    html = await resp.text()
    assert "Authentication failed" in html


# --- Connect phrase detection stores conversation ref ---


@pytest.mark.asyncio
async def test_connect_phrase_stores_conv_ref(patched_app):
    """When user says 'connect to crm', conversation ref should be stored."""
    ctx = MagicMock()
    ctx.activity.text = "connect to crm"
    ctx.activity.value = None
    ctx.activity.from_property.id = "user-123"
    ctx.activity.from_property.name = "Test User"
    ctx.activity.conversation.id = "conv-123"
    ctx.activity.local_timezone = None
    ctx.send_activity = AsyncMock()
    ctx.turn_state = {}

    mock_store = AsyncMock()

    with (
        patch.object(
            patched_app,
            "start_oauth_flow",
            new_callable=AsyncMock,
            return_value=(
                "https://dev.virtualdojo.com/mcp/v1/oauth/authorize?...",
                "state-xyz",
            ),
        ),
        patch(
            "botbuilder.core.teams.TeamsInfo.get_member",
            new_callable=AsyncMock,
            return_value=MagicMock(
                email="test@virtualdojo.com", user_principal_name="",
                id="user-123", name="Test User",
            ),
        ),
        patch(
            "botbuilder.core.teams.TeamsInfo.get_members",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch("task_store.get_task_store", new_callable=AsyncMock, return_value=mock_store),
    ):
        await patched_app.on_message(ctx)

    # Conversation reference should be stored under the OAuth state
    assert "state-xyz" in patched_app._oauth_conversation_refs
    stored = patched_app._oauth_conversation_refs["state-xyz"]
    assert stored["user_id"] == "user-123"
    assert stored["conversation_id"] == "conv-123"
    assert "conv_ref" in stored

    # Clean up
    patched_app._oauth_conversation_refs.clear()


# --- inject_auth_message ---


@pytest.mark.asyncio
async def test_inject_auth_message_calls_graph():
    """inject_auth_message should invoke the graph with an auth confirmation message."""
    import agent

    importlib.reload(agent)

    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(
        return_value={"messages": [MagicMock(content="ok")]}
    )
    agent._user_graphs["user-abc"] = mock_graph

    await agent.inject_auth_message("user-abc", "conv-123")

    mock_graph.ainvoke.assert_called_once()
    call_args = mock_graph.ainvoke.call_args[0][0]
    messages = call_args["messages"]
    assert len(messages) == 1
    assert "authenticated" in messages[0].content.lower()
    assert "VirtualDojo CRM" in messages[0].content

    # Clean up
    agent._user_graphs.pop("user-abc", None)


# --- Card action handlers ignore empty activity.value ---


@pytest.mark.asyncio
async def test_on_message_handles_card_action(patched_app):
    """When activity.value is set (card button click), route to card handler."""
    ctx = MagicMock()
    ctx.activity.text = None
    ctx.activity.value = {"action": "social_approve", "conversation_id": "conv-1"}
    # activity.name must be None/falsy so on_message doesn't fall through
    ctx.activity.name = None
    ctx.send_activity = AsyncMock()

    # Patch the imported reference that on_message actually calls
    with patch(
        "app.handle_card_action",
        new_callable=AsyncMock,
    ) as mock_handler:
        await patched_app.on_message(ctx)

    mock_handler.assert_called_once_with(ctx, ctx.activity.value)
