"""Feedback-validation card shown when a user clicks the native Teams 👍/👎.

Teams renders the thumbs itself (via ``channelData.feedbackLoop`` on the outbound
message — NOT an Adaptive Card). When a thumb is clicked with ``type:"custom"``,
Teams sends a ``message/fetchTask`` invoke and we return THIS small card as a
dialog. On submit, Teams sends ``message/submitAction`` carrying the card's
inputs + ``data``, which we persist onto the turn record (see
``conversation_log.record_feedback``) so the self-tuning eval gets a real,
human-verified signal instead of grading the model against its own past choices.

The ``turn_id`` round-trips through the card's ``Action.Submit`` data so the
submission correlates to the exact turn (``reply_to_id`` is unreliable for this —
msteams-docs #11870). Categories map directly to the eval buckets:
``gave_up`` is the anti-stall negative the gate currently can't measure.
"""

from __future__ import annotations

import json

# 👎 categories — each maps to ONE downstream action (see selftune.evalset.failure_route):
#   wrong_tool / gave_up  -> the self-tuning loop CAN fix these (routing + anti-stall)
#   incorrect             -> a grounding error; routes to the verifier / human, NOT the tuner
#   other                 -> triage only (the free-text note carries the signal)
DISLIKE_CATEGORIES = [
    {"title": "Looked in the wrong place / used the wrong source", "value": "wrong_tool"},
    {"title": "Stopped before finishing", "value": "gave_up"},
    {"title": "Wrong or made-up answer", "value": "incorrect"},
    {"title": "Other", "value": "other"},
]


def build_feedback_card(reaction: str, turn_id: str = "") -> dict:
    """Build the feedback dialog card for a 👍 (``like``) or 👎 (``dislike``).

    ``turn_id`` is embedded in the submit ``data`` so the response correlates to
    the originating turn. Returns a plain Adaptive Card dict (wrapped by the
    caller via ``CardFactory.adaptive_card``).
    """
    is_dislike = reaction == "dislike"
    # Carried back verbatim in the submit payload so we know which turn + reaction.
    submit_data = {"action": "feedback", "turn_id": turn_id, "reaction": reaction}

    body: list[dict] = []
    if is_dislike:
        body.append({
            "type": "TextBlock", "text": "Thanks — what was off?",
            "weight": "Bolder", "wrap": True,
        })
        body.append({
            "type": "Input.ChoiceSet", "id": "category", "style": "expanded",
            "isRequired": False, "choices": list(DISLIKE_CATEGORIES),
        })
        body.append({
            "type": "Input.Text", "id": "feedbackText", "isMultiline": True,
            "placeholder": "Anything to add? (optional)",
        })
        submit_title = "Send feedback"
    else:
        body.append({
            "type": "TextBlock", "text": "Glad that helped! 🙌",
            "weight": "Bolder", "wrap": True,
        })
        body.append({
            "type": "Input.Text", "id": "feedbackText", "isMultiline": True,
            "placeholder": "What worked well? (optional)",
        })
        submit_title = "Send"

    return {
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "type": "AdaptiveCard",
        "version": "1.5",
        "body": body,
        "actions": [
            {"type": "Action.Submit", "title": submit_title, "data": submit_data}
        ],
    }


def extract_reaction(value: dict | None) -> str:
    """Best-effort pull of 'like'/'dislike' from a feedback invoke value.

    The docs say it's at ``actionValue.reaction``, but the live Teams payload
    differs (observed: our code fell through to the default on a real 👎). So we
    check the documented spot, a stringified ``actionValue``, and the top level.
    Returns '' if we genuinely can't find it (caller logs the raw payload).
    """
    value = value or {}
    # message/fetchTask wraps the payload under "data"; message/submitAction does
    # not (confirmed from live Teams logs — the docs omit the "data" wrapper).
    # Search both scopes.
    scopes = [value]
    if isinstance(value.get("data"), dict):
        scopes.append(value["data"])
    for scope in scopes:
        av = scope.get("actionValue")
        if isinstance(av, str):
            try:
                av = json.loads(av)
            except (json.JSONDecodeError, TypeError):
                av = {}
        candidates = []
        if isinstance(av, dict):
            candidates.append(av.get("reaction"))
        candidates.append(scope.get("reaction"))
        raw_av = scope.get("actionValue")
        if isinstance(raw_av, str):
            candidates.append(raw_av)
        for c in candidates:
            if c in ("like", "dislike"):
                return c
    return ""


def parse_feedback_submit(value: dict | None) -> dict:
    """Extract {reaction, turn_id, category, text} from a message/submitAction value.

    Teams nests the submitted form under ``actionValue.feedback`` as a JSON-encoded
    STRING; our card's ``Action.Submit`` data (turn_id/reaction/category) may land
    inside it, directly under ``actionValue``, or at the top level depending on the
    client. The exact shape isn't verified against a live tenant, so we parse
    defensively and check all three places.
    """
    value = value or {}
    # Symmetry with the fetch payload: unwrap a "data" wrapper if present.
    if "actionValue" not in value and isinstance(value.get("data"), dict):
        value = value["data"]
    action_value = value.get("actionValue") or {}
    reaction = action_value.get("reaction", "")

    raw = action_value.get("feedback")
    form: dict = {}
    if isinstance(raw, str):
        try:
            form = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            form = {"feedbackText": raw}
    elif isinstance(raw, dict):
        form = raw

    def _pick(key: str) -> str:
        return form.get(key) or action_value.get(key) or value.get(key) or ""

    return {
        "reaction": reaction or _pick("reaction"),
        "turn_id": _pick("turn_id"),
        "category": _pick("category"),
        "text": form.get("feedbackText") or form.get("text") or "",
    }
