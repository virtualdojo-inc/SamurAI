"""Microsoft Teams bot entrypoint — runs on Cloud Run via aiohttp."""

import asyncio
import json
import os

from aiohttp import web
from botbuilder.core import (
    BotFrameworkAdapter,
    BotFrameworkAdapterSettings,
    CardFactory,
    TurnContext,
)
from botbuilder.schema import Activity

from agent import run_agent, inject_auth_message
from tools.virtualdojo_mcp import exchange_code, start_oauth_flow
from tools.social_media import _pending_cards
from tools.background_tasks import _pending_task_context
from tools.fedramp_docs import _pending_fedramp_cards, _pending_file_uploads, _uploaded_files
from cards.social import (
    build_social_preview_card,
    build_scheduled_posts_cards,
)
from cards.actions import (
    handle_card_action,
    handle_schedule_date_reply,
    is_awaiting_schedule_date,
    store_card_activity_id,
)

# Store conversation references for proactive messaging after OAuth
# Key: OAuth state parameter, Value: ConversationReference
_oauth_conversation_refs: dict[str, object] = {}

# Track running agent tasks so they can be cancelled with "stop"
# Key: conversation_id, Value: asyncio.Task
_running_tasks: dict[str, asyncio.Task] = {}

settings = BotFrameworkAdapterSettings(
    app_id=os.environ.get("MICROSOFT_APP_ID", ""),
    app_password=os.environ.get("MICROSOFT_APP_PASSWORD", ""),
    channel_auth_tenant=os.environ.get("MICROSOFT_APP_TENANT_ID", ""),
)
adapter = BotFrameworkAdapter(settings)


async def on_error(context: TurnContext, error: Exception):
    import traceback
    traceback.print_exc()
    print(f"[on_turn_error] {error}", flush=True)
    await context.send_activity("Sorry, something went wrong. Please try again.")


adapter.on_turn_error = on_error


async def _keep_typing(turn_context: TurnContext, stop_event: asyncio.Event):
    """Send typing indicators every 2 seconds until the stop event is set."""
    while not stop_event.is_set():
        try:
            await turn_context.send_activity(Activity(type="typing"))
        except Exception:
            break
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pass


def _clean_user_text(activity) -> str:
    """Message text with the bot's own @mention removed (group chats / channels).
    In 1:1 there is no mention so the text is unchanged; defensive fallback for
    non-Teams / mock activities where mention parsing isn't available."""
    try:
        cleaned = TurnContext.remove_recipient_mention(activity)
    except Exception:
        cleaned = None
    return (cleaned or getattr(activity, "text", "") or "").strip()


def _is_group_scope(activity) -> bool:
    """True for Teams group chats and channels (vs a personal 1:1 chat)."""
    ct = getattr(getattr(activity, "conversation", None), "conversation_type", "") or ""
    return ct in ("groupChat", "channel")


async def on_message(turn_context: TurnContext):
    activity_type = turn_context.activity.type
    activity_name = getattr(turn_context.activity, "name", None)
    print(f"[on_message] type={activity_type} name={activity_name}", flush=True)

    # Handle FileConsentCard invoke (user accepted/declined file upload)
    if activity_name == "fileConsent/invoke":
        print("[on_message] Routing to file consent handler", flush=True)
        await _handle_file_consent(turn_context)
        return

    # Teams native feedback-loop (👍/👎) — custom dialog flow.
    # Thumb click → fetch our validation card; form submit → persist the feedback.
    if activity_name == "message/fetchTask":
        await _handle_feedback_fetch(turn_context)
        return
    if activity_name == "message/submitAction":
        await _handle_feedback_submit(turn_context)
        return

    # Handle Adaptive Card Action.Submit callbacks (buttons clicked)
    if turn_context.activity.value and isinstance(turn_context.activity.value, dict):
        # Don't catch file consent invokes here
        if activity_name:
            pass  # Let it fall through to normal message handling
        else:
            await handle_card_action(turn_context, turn_context.activity.value)
            return

    user_message = _clean_user_text(turn_context.activity)
    conversation_id = turn_context.activity.conversation.id
    is_group = _is_group_scope(turn_context.activity)

    # Handle file attachments — download, parse, and append content to message
    file_context = ""
    attachments = turn_context.activity.attachments or []
    for att in attachments:
        if att.content_type == "application/vnd.microsoft.teams.file.download.info":
            try:
                import httpx
                from tools.file_handler import parse_file, _uploaded_files

                download_url = att.content.get("downloadUrl", "")
                filename = att.name or "unknown"
                print(f"[on_message] File received: {filename}", flush=True)

                async with httpx.AsyncClient() as client:
                    resp = await client.get(download_url)
                    resp.raise_for_status()
                    content_bytes = resp.content

                text_content, file_type = parse_file(filename, content_bytes)

                # Store for agent tools (edit_document, get_uploaded_file_content)
                _uploaded_files[conversation_id] = {
                    "filename": filename,
                    "content_bytes": content_bytes,
                    "file_type": file_type,
                    "text_content": text_content,
                }

                preview = text_content[:5000] if len(text_content) > 5000 else text_content
                file_context += f"\n\n[Attached file: {filename} ({file_type})]\n{preview}"
                if len(text_content) > 5000:
                    file_context += f"\n... [truncated — use get_uploaded_file_content for full content]"
            except Exception as e:
                print(f"[on_message] File download/parse failed: {e}", flush=True)
                file_context += f"\n\n[Attached file: {att.name} — failed to process: {e}]"

    if file_context:
        user_message = (user_message or "The user uploaded a file. Please review it.") + file_context

    if not user_message:
        return

    # Handle "stop" command — cancel the running agent task
    if user_message.strip().lower() == "stop":
        task = _running_tasks.pop(conversation_id, None)
        if task and not task.done():
            task.cancel()
            await turn_context.send_activity(
                Activity(type="message", text="Stopped.")
            )
        else:
            await turn_context.send_activity(
                Activity(type="message", text="Nothing running to stop.")
            )
        return

    # If we're waiting for a schedule date, intercept the reply
    if is_awaiting_schedule_date(conversation_id):
        await handle_schedule_date_reply(turn_context, conversation_id, user_message)
        return

    # Start continuous typing indicator
    stop_typing = asyncio.Event()
    typing_task = asyncio.create_task(_keep_typing(turn_context, stop_typing))

    user_id = turn_context.activity.from_property.id
    user_name = turn_context.activity.from_property.name or "unknown"
    local_tz = getattr(turn_context.activity, "local_timezone", None) or ""
    # Extract user email via Teams-specific roster API
    user_email = ""
    try:
        from botbuilder.core.teams import TeamsInfo

        member = await TeamsInfo.get_member(turn_context, user_id)
        user_email = member.email or member.user_principal_name or ""
    except Exception as e:
        print(f"[on_message] TeamsInfo.get_member failed: {e}", flush=True)
    if not user_email and user_name and "@" in user_name:
        user_email = user_name
    print(f"[on_message] user={user_name} email={user_email} id={user_id}", flush=True)

    # Only allow virtualdojo.com users
    if not user_email or not user_email.lower().endswith("@virtualdojo.com"):
        stop_typing.set()
        await typing_task
        # In a group chat / channel, stay silent for a non-team sender — posting a
        # rejection into a shared conversation is noise. Only reply in a 1:1 chat.
        if not is_group:
            await turn_context.send_activity(
                Activity(
                    type="message",
                    text="Sorry, SamurAI is only available to VirtualDojo team members.",
                )
            )
        print(f"[on_message] BLOCKED unauthorized user: {user_email or user_id} group={is_group}", flush=True)
        return

    # Persist conversation reference for proactive messaging (background tasks)
    try:
        from task_store import get_task_store

        _store = await get_task_store()
        conv_ref = TurnContext.get_conversation_reference(turn_context.activity)
        await _store.save_conversation_ref(
            conversation_id=conversation_id,
            user_id=user_id,
            ref_json=json.dumps(conv_ref.serialize()),
        )
        # Auto-populate team roster with current user
        if user_email:
            service_url = turn_context.activity.service_url or ""
            tenant_id = ""
            if hasattr(turn_context.activity, "channel_data") and turn_context.activity.channel_data:
                tenant_id = turn_context.activity.channel_data.get("tenant", {}).get("id", "")
            await _store.save_team_member(
                email=user_email,
                teams_id=user_id,
                display_name=user_name,
                service_url=service_url,
                tenant_id=tenant_id,
            )
        # Discover other team members if in a team context
        try:
            from botbuilder.core.teams import TeamsInfo as _TeamsInfo

            members = await _TeamsInfo.get_members(turn_context)
            for m in members:
                m_email = m.email or m.user_principal_name or ""
                if m_email:
                    await _store.save_team_member(
                        email=m_email,
                        teams_id=m.id,
                        display_name=m.name or "",
                        service_url=turn_context.activity.service_url or "",
                        tenant_id=tenant_id,
                    )
        except Exception:
            pass  # Not in a team context or roster fetch failed
    except Exception as e:
        print(f"[on_message] persist conversation ref failed: {e}", flush=True)

    try:
        # Check if user is asking to connect to VirtualDojo CRM
        msg_lower = user_message.lower()
        if any(
            phrase in msg_lower
            for phrase in [
                "connect to virtualdojo",
                "connect to the virtualdojo",
                "connect virtualdojo",
                "sign in to virtualdojo",
                "sign into virtualdojo",
                "login to virtualdojo",
                "login virtualdojo",
                "connect crm",
                "login crm",
                "sign in crm",
                "connect to crm",
                "virtualdojo login",
                "virtualdojo connect",
                "virtualdojo sign in",
            ]
        ):
            login_url, oauth_state = await start_oauth_flow(user_id)
            # Save conversation reference so we can notify after OAuth completes
            conv_ref = TurnContext.get_conversation_reference(
                turn_context.activity
            )
            _oauth_conversation_refs[oauth_state] = {
                "conv_ref": conv_ref,
                "user_id": user_id,
                "conversation_id": conversation_id,
            }
            stop_typing.set()
            await typing_task
            await turn_context.send_activity(
                Activity(
                    type="message",
                    text=f"[Sign in to VirtualDojo CRM]({login_url})\n\n"
                    f"Click the link above to authenticate. Once done, "
                    f"I'll be able to access your CRM data.",
                )
            )
            return

        # Provide user context for background task tools
        _pending_task_context[conversation_id] = {
            "user_id": user_id,
            "user_name": user_name,
            "user_timezone": local_tz,
        }
        async def _send_status(status_text: str):
            """Send an intermediate status message to Teams."""
            await turn_context.send_activity(
                Activity(type="message", text=status_text)
            )

        agent_task = asyncio.create_task(run_agent(
            user_message,
            conversation_id=conversation_id,
            user_id=user_id,
            user_name=user_name,
            user_timezone=local_tz,
            user_email=user_email,
            status_callback=_send_status,
        ))
        _running_tasks[conversation_id] = agent_task
        try:
            response = await agent_task
        except asyncio.CancelledError:
            return  # User said "stop" — already handled
        except Exception as e:
            if "recursion limit" in str(e).lower() or "GraphRecursionError" in type(e).__name__:
                await turn_context.send_activity(
                    "I ran out of steps before finishing — this one's more complex than I expected. "
                    "Want me to keep going from where I left off, or should I focus on a specific part?"
                )
                print(f"[on_message] GraphRecursionError for {user_name}: {e}", flush=True)
                return
            raise
        finally:
            _running_tasks.pop(conversation_id, None)
            _pending_task_context.pop(conversation_id, None)
    finally:
        stop_typing.set()
        await typing_task

    # If the agent says user needs to sign in, generate a login link
    if "not signed in to VirtualDojo" in response or "sign-in link" in response:
        login_url, oauth_state = await start_oauth_flow(user_id)
        conv_ref = TurnContext.get_conversation_reference(
            turn_context.activity
        )
        _oauth_conversation_refs[oauth_state] = {
            "conv_ref": conv_ref,
            "user_id": user_id,
            "conversation_id": conversation_id,
        }
        response = (
            f"You need to sign in to VirtualDojo to access CRM data.\n\n"
            f"[Sign in to VirtualDojo]({login_url})\n\n"
            f"Click the link above, then try your request again."
        )

    # Check if any tool stored card data for this conversation
    card_data = _pending_cards.pop(conversation_id, None)
    fedramp_card = _pending_fedramp_cards.pop(conversation_id, None)

    # Check for edited files to send back
    from tools.file_handler import _pending_edited_files
    edited_file = _pending_edited_files.pop(conversation_id, None)

    if card_data:
        card_type = card_data.get("card_type")
        await _send_card_response(
            turn_context, card_type, card_data, response, conversation_id
        )
    elif fedramp_card:
        await _send_card_response(
            turn_context, "fedramp_file_upload", fedramp_card, response, conversation_id
        )
    else:
        # Native Teams 👍/👎 footer buttons (passive, not a card). type:"custom"
        # routes a click to _handle_feedback_fetch so we can show our own
        # validation card. The thumbs render on the plain markdown message.
        await turn_context.send_activity(
            Activity(
                type="message",
                text=response,
                channel_data={"feedbackLoop": {"type": "custom"}},
            )
        )

    # Send edited file via FileConsentCard if one was created
    if edited_file:
        from botbuilder.schema import Attachment

        file_name = edited_file["filename"]
        file_size = len(edited_file["content_bytes"])
        file_consent_card = {
            "description": edited_file.get("summary", "Edited document"),
            "sizeInBytes": file_size,
            "acceptContext": {
                "conversation_id": conversation_id,
                "file_source": "edited_document",
            },
            "declineContext": {"conversation_id": conversation_id},
        }
        # Store the bytes for the upload callback
        from tools.file_handler import _pending_edited_files as _pef_store
        _pef_store[f"_upload_{conversation_id}"] = edited_file
        print(f"[edited_file] Stored {file_size} bytes for consent upload, conv={conversation_id}", flush=True)

        await turn_context.send_activity(
            Activity(
                type="message",
                text=f"Here's the edited file: **{file_name}**",
                attachments=[
                    Attachment(
                        content_type="application/vnd.microsoft.teams.card.file.consent",
                        name=file_name,
                        content=file_consent_card,
                    )
                ],
            )
        )


async def _send_card_response(
    turn_context: TurnContext,
    card_type: str,
    card_data: dict,
    text_fallback: str,
    conversation_id: str,
):
    """Send an Adaptive Card based on tool card data."""
    if card_type == "social_preview":
        card = build_social_preview_card(
            text=card_data["text"],
            platforms=card_data["platforms"],
            conversation_id=card_data["conversation_id"],
            image_url=card_data.get("image_url", ""),
            scheduled_date=card_data.get("scheduled_date", ""),
        )
        attachment = CardFactory.adaptive_card(card)
        resource = await turn_context.send_activity(
            Activity(type="message", attachments=[attachment])
        )
        # Store activity ID so we can update the card on approve/reject
        if resource and resource.id:
            store_card_activity_id(conversation_id, resource.id)

    elif card_type == "scheduled_posts":
        cards = build_scheduled_posts_cards(card_data.get("posts", []))
        if cards:
            attachments = [CardFactory.adaptive_card(c) for c in cards]
            await turn_context.send_activity(
                Activity(
                    type="message",
                    attachment_layout="carousel",
                    attachments=attachments,
                )
            )
        else:
            await turn_context.send_activity(
                Activity(type="message", text=text_fallback)
            )

    elif card_type == "fedramp_file_upload":
        # Send FileConsentCard for FedRAMP document editing
        from botbuilder.schema import Attachment

        file_name = card_data.get("file_name", "document.md")
        file_size = card_data.get("file_size", 0)
        file_consent_card = {
            "description": card_data.get("summary", "FedRAMP document for review"),
            "sizeInBytes": file_size,
            "acceptContext": {"conversation_id": conversation_id, "file_path": card_data.get("file_path", "")},
            "declineContext": {"conversation_id": conversation_id},
        }
        await turn_context.send_activity(
            Activity(
                type="message",
                text=f"I'd like to upload **{file_name}** for your review.",
                attachments=[
                    Attachment(
                        content_type="application/vnd.microsoft.teams.card.file.consent",
                        name=file_name,
                        content=file_consent_card,
                    )
                ],
            )
        )

    else:
        # Unknown card type — fall back to text
        await turn_context.send_activity(Activity(type="message", text=text_fallback))


async def _handle_feedback_fetch(turn_context: TurnContext):
    """A 👍/👎 was clicked → return our feedback-validation card as a dialog.

    Teams sends message/fetchTask with value.actionValue.reaction (like|dislike).
    We correlate to the latest turn (reply_to_id is unreliable) and embed that
    turn_id in the card so the submit can attach feedback to the exact turn.
    """
    from botbuilder.core import CardFactory
    from botbuilder.schema import ActivityTypes, InvokeResponse
    from botbuilder.schema.teams import (
        TaskModuleContinueResponse,
        TaskModuleResponse,
        TaskModuleTaskInfo,
    )

    from cards.feedback import build_feedback_card, extract_reaction
    from conversation_log import find_latest_turn_id

    value = turn_context.activity.value or {}
    reaction = extract_reaction(value) or "like"
    conversation_id = turn_context.activity.conversation.id
    turn_id = find_latest_turn_id(conversation_id) or ""

    card = build_feedback_card(reaction=reaction, turn_id=turn_id)
    task_info = TaskModuleTaskInfo(
        title="Send feedback",
        card=CardFactory.adaptive_card(card),
        width="medium",
        height="small",
    )
    body = TaskModuleResponse(task=TaskModuleContinueResponse(value=task_info))
    await turn_context.send_activity(
        Activity(
            type=ActivityTypes.invoke_response,
            value=InvokeResponse(status=200, body=body.serialize()),
        )
    )


async def _handle_feedback_submit(turn_context: TurnContext):
    """The feedback form was submitted → persist onto the turn record.

    Teams sends message/submitAction. value.actionValue.feedback is a JSON-encoded
    STRING; our card's Action.Submit data (turn_id, reaction, category) may surface
    inside it, directly under actionValue, or at the top level — parse defensively.
    MUST return 200 with an empty body, or Teams shows error 400.
    """
    from botbuilder.schema import ActivityTypes, InvokeResponse

    from cards.feedback import parse_feedback_submit
    from conversation_log import record_feedback

    parsed = parse_feedback_submit(turn_context.activity.value)
    conversation_id = turn_context.activity.conversation.id

    record_feedback(
        conversation_id=conversation_id,
        turn_id=parsed["turn_id"],
        reaction=parsed["reaction"],
        category=parsed["category"],
        text=parsed["text"],
    )
    print(
        f"[feedback] reaction={parsed['reaction']} category={parsed['category'] or '-'} conv={conversation_id}",
        flush=True,
    )
    await turn_context.send_activity(
        Activity(
            type=ActivityTypes.invoke_response,
            value=InvokeResponse(status=200, body={}),
        )
    )


async def _handle_file_consent(turn_context: TurnContext):
    """Handle FileConsentCard accept/decline from Teams."""
    value = turn_context.activity.value or {}
    action = value.get("action", "")
    context = value.get("context", {})
    conversation_id = context.get("conversation_id", "")

    if action == "accept":
        upload_info = value.get("uploadInfo", {})
        upload_url = upload_info.get("uploadUrl", "")
        content_url = upload_info.get("contentUrl", "")
        file_path = context.get("file_path", "")

        if not upload_url or not conversation_id:
            await turn_context.send_activity(
                Activity(type="message", text="File upload failed: missing upload info.")
            )
            return

        # Get the pending file content — check edited files first, then FedRAMP docs
        from tools.file_handler import _pending_edited_files as _edit_store
        edited = _edit_store.pop(f"_upload_{conversation_id}", None)
        pending = _pending_file_uploads.get(conversation_id) if not edited else None

        print(f"[file_consent] edited={bool(edited)} pending={bool(pending)} conv={conversation_id}", flush=True)

        if not pending and not edited:
            await turn_context.send_activity(
                Activity(type="message", text="No pending file found for this conversation.")
            )
            return

        # Upload file to OneDrive via the provided URL
        try:
            import httpx

            if edited:
                file_bytes = edited["content_bytes"]
                print(f"[file_consent] Using edited file: {len(file_bytes)} bytes", flush=True)
            else:
                content = pending.get("content", "")
                file_bytes = content.encode("utf-8") if isinstance(content, str) else content
                print(f"[file_consent] Using pending file: {len(file_bytes)} bytes", flush=True)
            file_size = len(file_bytes)
            async with httpx.AsyncClient() as client:
                resp = await client.put(
                    upload_url,
                    content=file_bytes,
                    headers={
                        "Content-Type": "application/octet-stream",
                        "Content-Range": f"bytes 0-{file_size - 1}/{file_size}",
                        "Content-Length": str(file_size),
                    },
                )
                resp.raise_for_status()

            # Store the content URL for later retrieval when committing
            _uploaded_files[conversation_id] = {
                "file_path": file_path,
                "content_url": content_url,
            }

            file_name = upload_info.get("name", "file")
            unique_id = upload_info.get("uniqueId", "")
            file_type = upload_info.get("fileType", "")

            # Send a Teams file info card — renders as a clickable file in chat
            from botbuilder.schema import Attachment

            file_info_card = {
                "uniqueId": unique_id,
                "fileType": file_type,
            }
            await turn_context.send_activity(
                Activity(
                    type="message",
                    attachments=[
                        Attachment(
                            content_type="application/vnd.microsoft.teams.card.file.info",
                            name=file_name,
                            content=file_info_card,
                            content_url=content_url,
                        )
                    ],
                )
            )
        except Exception as e:
            print(f"[file_consent] Upload failed: {e}", flush=True)
            await turn_context.send_activity(
                Activity(type="message", text=f"File upload failed: {e}")
            )

    elif action == "decline":
        _pending_file_uploads.pop(conversation_id, None)
        await turn_context.send_activity(
            Activity(type="message", text="File upload declined. The draft has been discarded.")
        )


async def messages(req: web.Request) -> web.Response:
    if "application/json" not in req.headers.get("Content-Type", ""):
        return web.Response(status=415)

    body = await req.json()
    activity = Activity().deserialize(body)
    auth_header = req.headers.get("Authorization", "")

    # process_activity returns an InvokeResponse for invoke activities (e.g. the
    # message/fetchTask feedback dialog). The previous code discarded it, so the
    # dialog body never reached Teams. Emit it when present; otherwise plain 200
    # (preserves existing fileConsent/message behavior, which sets no invoke body).
    invoke_response = await adapter.process_activity(activity, auth_header, on_message)
    if invoke_response is not None:
        return web.json_response(data=invoke_response.body, status=invoke_response.status)
    return web.Response(status=200)


async def oauth_callback(req: web.Request) -> web.Response:
    """Handle OAuth redirect from VirtualDojo after user authenticates."""
    code = req.query.get("code")
    state = req.query.get("state")
    error = req.query.get("error")

    if error:
        return web.Response(
            text=f"<html><body><h2>Authentication failed</h2><p>{error}</p>"
            f"<p>You can close this window and try again in Teams.</p></body></html>",
            content_type="text/html",
        )

    if not code or not state:
        return web.Response(
            text="<html><body><h2>Missing parameters</h2>"
            "<p>Invalid callback. Please try signing in again from Teams.</p></body></html>",
            content_type="text/html",
        )

    tokens = await exchange_code(code, state)
    if tokens:
        # Send proactive message to Teams and inject auth into conversation history
        oauth_ctx = _oauth_conversation_refs.pop(state, None)
        if oauth_ctx:
            conv_ref = oauth_ctx["conv_ref"]
            oauth_user_id = oauth_ctx["user_id"]
            oauth_conv_id = oauth_ctx["conversation_id"]

            # Inject auth confirmation into LangGraph conversation history
            try:
                await inject_auth_message(oauth_user_id, oauth_conv_id)
            except Exception as e:
                print(f"[oauth] inject_auth_message failed: {e}", flush=True)

            # Send proactive message to Teams
            try:

                async def _notify(turn_context: TurnContext):
                    await turn_context.send_activity(
                        Activity(
                            type="message",
                            text="Connected to VirtualDojo CRM! "
                            "I can now access your contacts, accounts, "
                            "opportunities, and compliance records. "
                            "What would you like to look up?",
                        )
                    )

                await adapter.continue_conversation(
                    conv_ref, _notify, settings.app_id
                )
            except Exception as e:
                print(f"[oauth] Proactive message failed: {e}", flush=True)

        return web.Response(
            text="<html><body><h2>Connected to VirtualDojo!</h2>"
            "<p>You can close this window and return to Teams. "
            "SamurAI can now access your CRM data.</p></body></html>",
            content_type="text/html",
        )
    else:
        _oauth_conversation_refs.pop(state, None)
        return web.Response(
            text="<html><body><h2>Authentication failed</h2>"
            "<p>Could not exchange the authorization code. "
            "Please try signing in again from Teams.</p></body></html>",
            content_type="text/html",
        )


async def health(req: web.Request) -> web.Response:
    return web.Response(text="ok")


from admin import handle_admin  # secured admin endpoint (fail-closed; allowlisted ops)

app = web.Application()
app.router.add_post("/api/messages", messages)
app.router.add_get("/api/oauth/callback", oauth_callback)
app.router.add_get("/health", health)
app.router.add_post("/admin", handle_admin)


async def on_startup(app_instance):
    from scheduler import init_scheduler

    await init_scheduler(adapter, settings.app_id)
    print("[startup] Background task scheduler started", flush=True)


async def on_cleanup(app_instance):
    from scheduler import shutdown_scheduler

    await shutdown_scheduler()
    print("[cleanup] Background task scheduler stopped", flush=True)


app.on_startup.append(on_startup)
app.on_cleanup.append(on_cleanup)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting bot on port {port}", flush=True)
    web.run_app(app, host="0.0.0.0", port=port)
