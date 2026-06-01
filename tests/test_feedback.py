"""Tests for the Teams 👍/👎 feedback-validation flow + its eval-signal wiring.

Covers the deterministic pieces: card building, the defensive submit-payload
parser, turn correlation + persistence in conversation_log, and that human
feedback overrides the self-referential heuristic in evalset. The live-Teams
rendering / invoke round-trip is NOT covered here (needs a real client).
"""

from datetime import datetime, timezone

import conversation_log
from cards.feedback import build_feedback_card, extract_reaction, parse_feedback_submit
from selftune import evalset


# ---- card building ----------------------------------------------------------

def test_card_dislike_has_categories_and_correlation():
    card = build_feedback_card(reaction="dislike", turn_id="2026-06-01/120000-abcd1234.json")
    # category chooser present
    ids = [b.get("id") for b in card["body"]]
    assert "category" in ids
    assert any(b.get("type") == "Input.ChoiceSet" for b in card["body"])
    # the gave_up category exists (the anti-stall signal)
    choiceset = next(b for b in card["body"] if b.get("type") == "Input.ChoiceSet")
    assert "gave_up" in [c["value"] for c in choiceset["choices"]]
    # Action.Submit round-trips turn_id + reaction
    data = card["actions"][0]["data"]
    assert data == {"action": "feedback", "turn_id": "2026-06-01/120000-abcd1234.json", "reaction": "dislike"}


def test_card_like_is_minimal():
    card = build_feedback_card(reaction="like", turn_id="t9")
    assert not any(b.get("type") == "Input.ChoiceSet" for b in card["body"])  # no categories on 👍
    assert any(b.get("id") == "feedbackText" for b in card["body"])  # optional note
    assert card["actions"][0]["data"]["reaction"] == "like"


# ---- reaction extraction (shape varies by client) ---------------------------

def test_extract_reaction_documented_shape():
    assert extract_reaction({"actionName": "feedback", "actionValue": {"reaction": "dislike"}}) == "dislike"
    assert extract_reaction({"actionValue": {"reaction": "like"}}) == "like"


def test_extract_reaction_alternate_locations():
    assert extract_reaction({"reaction": "dislike"}) == "dislike"  # top level
    assert extract_reaction({"actionValue": '{"reaction": "dislike"}'}) == "dislike"  # stringified
    assert extract_reaction({"actionValue": "dislike"}) == "dislike"  # actionValue IS the reaction


def test_extract_reaction_fetchtask_data_wrapper():
    # The REAL message/fetchTask shape observed in live Teams logs.
    value = {"data": {"actionName": "feedback", "actionValue": {"reaction": "dislike"}}}
    assert extract_reaction(value) == "dislike"
    assert extract_reaction({"data": {"actionValue": {"reaction": "like"}}}) == "like"


def test_extract_reaction_missing_returns_empty():
    assert extract_reaction({"actionName": "feedback"}) == ""
    assert extract_reaction(None) == ""


# ---- defensive submit-payload parsing ---------------------------------------

def test_parse_default_form_shape():
    # Teams' default form: feedback is a JSON STRING with feedbackText.
    value = {"actionName": "feedback",
             "actionValue": {"reaction": "like", "feedback": '{"feedbackText": "great job"}'}}
    out = parse_feedback_submit(value)
    assert out["reaction"] == "like"
    assert out["text"] == "great job"


def test_parse_custom_form_roundtrips_our_data():
    # Our card's data nested into the JSON-string feedback field.
    value = {"actionName": "feedback", "actionValue": {
        "reaction": "dislike",
        "feedback": '{"turn_id": "2026-06-01/120000-abcd.json", "category": "gave_up", "feedbackText": "stopped early"}',
    }}
    out = parse_feedback_submit(value)
    assert out["turn_id"] == "2026-06-01/120000-abcd.json"
    assert out["category"] == "gave_up"
    assert out["text"] == "stopped early"
    assert out["reaction"] == "dislike"


def test_parse_data_directly_under_action_value():
    # Defensive: some clients surface card data directly, not inside the string.
    value = {"actionValue": {"reaction": "dislike", "turn_id": "t1", "category": "wrong_tool"}}
    out = parse_feedback_submit(value)
    assert out["turn_id"] == "t1"
    assert out["category"] == "wrong_tool"


def test_parse_handles_dict_feedback_and_empty():
    assert parse_feedback_submit({"actionValue": {"reaction": "like", "feedback": {"feedbackText": "ok"}}})["text"] == "ok"
    blank = parse_feedback_submit(None)
    assert blank == {"reaction": "", "turn_id": "", "category": "", "text": ""}


# ---- correlation + persistence (conversation_log) ---------------------------

def _seed_turn(tmp_raw, ts, conv="c1", msg="hello"):
    conversation_log.RAW_DIR = tmp_raw
    return conversation_log.log_turn(
        conversation_id=conv, user_id="u1", user_message=msg,
        assistant_response="hi", tools=["query_cloud_logs: ok -> ..."], ts=ts,
    )


def test_log_turn_writes_turn_id(tmp_path, monkeypatch):
    monkeypatch.setattr(conversation_log, "RAW_DIR", tmp_path / "raw")
    import json
    path = _seed_turn(tmp_path / "raw", datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc))
    rec = json.loads(open(path).read())
    assert rec["turn_id"] == "2026-06-01/" + path.split("/")[-1]


def test_find_latest_turn_id_picks_newest(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    monkeypatch.setattr(conversation_log, "RAW_DIR", raw)
    _seed_turn(raw, datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc), conv="c1")
    _seed_turn(raw, datetime(2026, 6, 1, 12, 5, 0, tzinfo=timezone.utc), conv="c1")
    _seed_turn(raw, datetime(2026, 6, 1, 12, 9, 0, tzinfo=timezone.utc), conv="other")
    latest = conversation_log.find_latest_turn_id("c1")
    assert latest.endswith(".json") and latest.startswith("2026-06-01/")
    # newest c1 turn is the 12:05 one, not the 12:09 'other' conversation
    assert "120500" in latest


def test_record_feedback_attaches_to_turn(tmp_path, monkeypatch):
    import json
    raw = tmp_path / "raw"
    monkeypatch.setattr(conversation_log, "RAW_DIR", raw)
    path = _seed_turn(raw, datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc), conv="c1")
    turn_id = json.loads(open(path).read())["turn_id"]
    out = conversation_log.record_feedback(
        conversation_id="c1", turn_id=turn_id, reaction="dislike",
        category="gave_up", text="stopped early",
    )
    assert out is not None
    rec = json.loads(open(path).read())
    assert rec["feedback"]["reaction"] == "dislike"
    assert rec["feedback"]["category"] == "gave_up"
    assert rec["feedback"]["text"] == "stopped early"


def test_record_feedback_falls_back_to_latest(tmp_path, monkeypatch):
    import json
    raw = tmp_path / "raw"
    monkeypatch.setattr(conversation_log, "RAW_DIR", raw)
    path = _seed_turn(raw, datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc), conv="c1")
    # No turn_id passed → resolves to the conversation's latest turn.
    out = conversation_log.record_feedback(conversation_id="c1", reaction="like")
    assert out is not None
    assert json.loads(open(path).read())["feedback"]["reaction"] == "like"


# ---- category-based routing (only tunable failures reach the proposer) -------

def _dislike(category):
    return {"user_message": "x", "assistant_response": "done",
            "tools": ["query_cloud_logs: ok -> ..."],
            "feedback": {"reaction": "dislike", "category": category, "text": "note here"}}


def test_card_dislike_categories_are_the_four_actionable_ones():
    card = build_feedback_card(reaction="dislike", turn_id="t")
    choiceset = next(b for b in card["body"] if b.get("type") == "Input.ChoiceSet")
    values = [c["value"] for c in choiceset["choices"]]
    assert values == ["wrong_tool", "gave_up", "incorrect", "other"]  # 'style' dropped


def test_failure_route_tunable_categories():
    assert evalset.failure_route(_dislike("wrong_tool")) == "tune"
    assert evalset.failure_route(_dislike("gave_up")) == "tune"


def test_failure_route_quarantines_grounding_and_unknown():
    assert evalset.failure_route(_dislike("incorrect")) == "quarantine"
    assert evalset.failure_route(_dislike("other")) == "quarantine"
    assert evalset.failure_route(_dislike("")) == "quarantine"  # 👎 with no category


def test_failure_route_heuristic_failures_are_tunable():
    tool_err = {"user_message": "x", "assistant_response": "done", "tools": ["sync_repo: error -> ..."]}
    gave_up = {"user_message": "x", "assistant_response": "Would you like me to continue?",
               "tools": ["query_cloud_logs: ok -> ..."]}
    assert evalset.failure_route(tool_err) == "tune"
    assert evalset.failure_route(gave_up) == "tune"


def test_failure_route_none_for_good():
    good = {"user_message": "x", "assistant_response": "done", "tools": ["query_cloud_logs: ok -> ..."]}
    assert evalset.failure_route(good) is None


def test_mine_failures_quarantines_incorrect_keeps_actionable():
    turns = [
        dict(_dislike("wrong_tool"), ts="2026-06-01T12:03:00", user_message="routing miss"),
        dict(_dislike("incorrect"), ts="2026-06-01T12:02:00", user_message="grounding miss"),
        dict(_dislike("gave_up"), ts="2026-06-01T12:01:00", user_message="stalled"),
    ]
    failures = evalset.mine_failures(turns)
    msgs = [f["message"] for f in failures]
    assert "routing miss" in msgs and "stalled" in msgs
    assert "grounding miss" not in msgs  # quarantined — not the tuner's job
    # category + note surfaced to the (in-boundary) proposer
    routing = next(f for f in failures if f["message"] == "routing miss")
    assert routing["category"] == "wrong_tool"
    assert routing["note"] == "note here"


# ---- recovery cases (anti-stall becomes measurable) --------------------------

def test_mine_recovery_cases_from_stalls():
    turns = [
        # heuristic give-up (no feedback) → recovery case
        {"user_message": "fix the build", "assistant_response": "Would you like me to continue?",
         "tools": ["sync_repo: ok -> ..."], "ts": "2026-06-01T12:03:00"},
        # 👎 gave_up → recovery case
        dict(_dislike("gave_up"), ts="2026-06-01T12:02:00", user_message="deploy it"),
        # 👎 incorrect → NOT a recovery case (quarantined, grounding issue)
        dict(_dislike("incorrect"), ts="2026-06-01T12:01:00", user_message="what is X"),
        # a good turn → not a stall
        {"user_message": "check logs", "assistant_response": "done",
         "tools": ["query_cloud_logs: ok -> ..."], "ts": "2026-06-01T12:00:00"},
    ]
    cases = evalset.mine_recovery_cases(turns)
    msgs = {c["message"] for c in cases}
    assert msgs == {"fix the build", "deploy it"}
    assert all(c["require_any_tool"] and c["source"] == "recovery" for c in cases)


# ---- evalset: human feedback overrides the self-referential heuristic --------

def test_dislike_labels_bad_even_if_tools_ok():
    turn = {"user_message": "x", "assistant_response": "done",
            "tools": ["query_cloud_logs: ok -> ..."], "feedback": {"reaction": "dislike"}}
    assert evalset.label_turn(turn) == "bad"


def test_like_overrides_giveup_false_positive():
    # Response matches a give-up phrase but the human said 👍 — trust the human.
    turn = {"user_message": "x", "assistant_response": "Would you like me to continue?",
            "tools": ["query_cloud_logs: ok -> ..."], "feedback": {"reaction": "like"}}
    assert evalset.label_turn(turn) == "good"


def test_turn_to_case_tags_human_verified():
    turn = {"user_message": "check logs", "assistant_response": "done",
            "tools": ["query_cloud_logs: ok -> ..."], "feedback": {"reaction": "like"}}
    case = evalset.turn_to_case(turn)
    assert case["source"] == "human-verified"
    assert case["human_verified"] is True
    assert case["expected_tools"] == ["query_cloud_logs"]


def test_no_feedback_behaves_as_before():
    turn = {"user_message": "check logs", "assistant_response": "done",
            "tools": ["query_cloud_logs: ok -> ..."]}
    assert evalset.label_turn(turn) == "good"
    assert evalset.turn_to_case(turn)["source"] == "mined"
