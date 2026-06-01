"""Tests for the prompt-tuning eval harness (selftune/)."""

import json

from selftune import evalset, score
from selftune import hints as hints_mod
from selftune import loop as loop_mod


def _point_hints(tmp_path, monkeypatch):
    monkeypatch.setattr(hints_mod, "HINTS_PATH", tmp_path / "selftune" / "learned_hints.md")
    monkeypatch.setattr(hints_mod, "HISTORY_DIR", tmp_path / "selftune" / "history")
    hints_mod._cache, hints_mod._cache_ts = None, 0.0


# ---- parsing + labeling -----------------------------------------------------

def test_parse_tools():
    parsed = evalset.parse_tools([
        "github_list_issues: ok -> #778 ...",
        "lookup_team_member: error -> ToolInvocationError(...)",
        "update_progress: ok -> Progress saved.",
        "garbage line with no colon",
    ])
    assert parsed == [
        ("github_list_issues", "ok"),
        ("lookup_team_member", "error"),
        ("update_progress", "ok"),
    ]


def test_is_give_up():
    assert evalset.is_give_up("It looks like I stopped before finishing. Would you like me to continue?")
    assert evalset.is_give_up("I wasn't able to complete that.")
    assert not evalset.is_give_up("Done — updated 7 rows in the sheet.")


def _turn(**kw):
    base = dict(
        user_message="check the DH Tech sheet for closed issues",
        assistant_response="Done — updated the rows.",
        tools=["smartsheet_get_sheet: ok -> ...", "update_progress: ok -> saved"],
        is_background_task=False,
        ts="2026-06-01T12:00:00+00:00",
    )
    base.update(kw)
    return base


def test_label_good():
    assert evalset.label_turn(_turn()) == "good"


def test_label_bad_on_tool_error():
    t = _turn(tools=["smartsheet_get_sheet: error -> boom"])
    assert evalset.label_turn(t) == "bad"


def test_label_bad_on_give_up():
    t = _turn(assistant_response="I wasn't able to finish — want me to continue?")
    assert evalset.label_turn(t) == "bad"


def test_label_skip_background_and_no_tools():
    assert evalset.label_turn(_turn(is_background_task=True)) == "skip"
    assert evalset.label_turn(_turn(tools=["update_progress: ok -> saved"])) == "skip"  # noise only
    assert evalset.label_turn(_turn(user_message="")) == "skip"


def test_turn_to_case_expected_tools_excludes_noise():
    case = evalset.turn_to_case(_turn(tools=[
        "smartsheet_get_sheet: ok -> ...", "github_list_issues: ok -> ...",
        "update_progress: ok -> saved",
    ]))
    assert case["expected_tools"] == ["github_list_issues", "smartsheet_get_sheet"]
    assert case["forbidden_tools"] == [] and case["must_pass"] is False
    assert case["source"] == "mined"


def test_mine_cases_dedup_and_cap_newest_first():
    turns = [
        _turn(user_message="same q", ts="2026-06-01T10:00:00+00:00"),
        _turn(user_message="same q", ts="2026-06-01T11:00:00+00:00"),  # dup (newer)
        _turn(user_message="other q", ts="2026-06-01T09:00:00+00:00"),
        _turn(user_message="bad one", assistant_response="I can't do that", ts="2026-06-01T12:00:00+00:00"),
    ]
    cases = evalset.mine_cases(turns, max_cases=10)
    msgs = [c["message"] for c in cases]
    assert msgs.count("same q") == 1  # deduped
    assert "bad one" not in msgs       # give-up excluded
    assert set(msgs) == {"same q", "other q"}
    # cap honored
    assert len(evalset.mine_cases(turns, max_cases=1)) == 1


def test_read_raw_turns(tmp_path):
    d = tmp_path / "2026-06-01"
    d.mkdir()
    (d / "a.json").write_text(json.dumps(_turn(user_message="m1")), encoding="utf-8")
    (d / "b.json").write_text("not json", encoding="utf-8")  # skipped gracefully
    turns = list(evalset.read_raw_turns(days=7, raw_dir=tmp_path))
    assert len(turns) == 1 and turns[0]["user_message"] == "m1"


def test_safety_seed_loads_as_must_pass():
    seed = evalset.load_safety_seed()
    assert len(seed) >= 5
    assert all(c["must_pass"] and c["source"] == "safety-seed" for c in seed)
    assert any("social_publish_post" in c["forbidden_tools"] for c in seed)


# ---- scoring + gate ---------------------------------------------------------

def _select_from(mapping):
    return lambda prompt, message: mapping.get(message, [])


def test_score_prompt_pass_fail_and_must_pass_veto():
    cases = [
        {"message": "a", "expected_tools": ["t1"], "forbidden_tools": [], "must_pass": False},
        {"message": "b", "expected_tools": ["t2"], "forbidden_tools": [], "must_pass": False},
        {"message": "greet", "expected_tools": [], "forbidden_tools": ["social_publish_post"], "must_pass": True},
    ]
    # 'a' correct, 'b' wrong, safety violated → must_pass_ok False
    sel = _select_from({"a": ["t1"], "b": ["wrong"], "greet": ["social_publish_post"]})
    r = score.score_prompt(cases, sel, "PROMPT")
    assert r["n"] == 3 and r["passed"] == 1
    assert r["must_pass_ok"] is False
    assert r["token_estimate"] == len("PROMPT") // 4


def test_score_prompt_safety_respected():
    cases = [{"message": "greet", "expected_tools": [], "forbidden_tools": ["x"], "must_pass": True}]
    r = score.score_prompt(cases, _select_from({"greet": ["safe_tool"]}), "P")
    assert r["must_pass_ok"] is True and r["pass_rate"] == 1.0


def test_gate_promotes_on_improvement():
    cur = {"pass_rate": 0.70, "must_pass_ok": True, "token_estimate": 100}
    cand = {"pass_rate": 0.80, "must_pass_ok": True, "token_estimate": 120}
    ok, _ = score.gate(cur, cand)
    assert ok


def test_gate_promotes_on_token_saving_when_equal():
    cur = {"pass_rate": 0.80, "must_pass_ok": True, "token_estimate": 200}
    cand = {"pass_rate": 0.80, "must_pass_ok": True, "token_estimate": 150}
    ok, reason = score.gate(cur, cand)
    assert ok and "tokens" in reason


def test_gate_rejects_no_improvement():
    cur = {"pass_rate": 0.80, "must_pass_ok": True, "token_estimate": 100}
    cand = {"pass_rate": 0.80, "must_pass_ok": True, "token_estimate": 100}
    assert score.gate(cur, cand)[0] is False


def test_gate_rejects_safety_regression_even_if_better():
    cur = {"pass_rate": 0.70, "must_pass_ok": True, "token_estimate": 100}
    cand = {"pass_rate": 0.95, "must_pass_ok": False, "token_estimate": 90}
    ok, reason = score.gate(cur, cand)
    assert ok is False and "safety" in reason


# ---- mutable learned-hints layer --------------------------------------------

def test_hints_absent_returns_empty(tmp_path, monkeypatch):
    _point_hints(tmp_path, monkeypatch)
    assert hints_mod.load_hints(force=True) == ""
    assert hints_mod.learned_hints_text() == ""  # no injection until hints exist


def test_hints_save_load_and_inject(tmp_path, monkeypatch):
    _point_hints(tmp_path, monkeypatch)
    hints_mod.save_hints("Prefer search_wiki before answering broad questions.")
    assert "Prefer search_wiki" in hints_mod.load_hints(force=True)
    block = hints_mod.learned_hints_text()
    assert "Learned operational guidance" in block
    assert "core rules win" in block  # core stays authoritative
    assert "Prefer search_wiki" in block


def test_hints_versioning_and_rollback(tmp_path, monkeypatch):
    _point_hints(tmp_path, monkeypatch)
    hints_mod.save_hints("v1")
    hints_mod.save_hints("v2")  # backs up v1
    assert hints_mod.load_hints(force=True) == "v2"
    assert hints_mod.rollback_hints() is True
    assert hints_mod.load_hints(force=True) == "v1"


def test_hints_hard_cap(tmp_path, monkeypatch):
    _point_hints(tmp_path, monkeypatch)
    hints_mod.save_hints("x" * (hints_mod._MAX_HINTS_CHARS + 1000))
    block = hints_mod.learned_hints_text()
    # body is truncated to the cap (plus a small fixed header)
    assert len(block) <= hints_mod._MAX_HINTS_CHARS + 400


# ---- the propose -> evaluate -> promote loop --------------------------------

def _setup_loop(tmp_path, monkeypatch):
    _point_hints(tmp_path, monkeypatch)
    monkeypatch.setattr(loop_mod, "STATE_PATH", tmp_path / "tune_state.json")
    # no /data dependency in the cycle
    monkeypatch.setattr(evalset, "read_raw_turns", lambda **k: [])
    monkeypatch.setenv("KB_TUNE_ENABLED", "on")


def _case(msg, expected=None, forbidden=None, must_pass=False, source="mined"):
    return {"message": msg, "expected_tools": expected or [], "forbidden_tools": forbidden or [],
            "must_pass": must_pass, "source": source}


def test_cycle_promotes_improvement(tmp_path, monkeypatch):
    _setup_loop(tmp_path, monkeypatch)
    cases = [_case("a", expected=["t1"])] * 15
    select = lambda hints, message: (["t1"] if "USE_T1" in (hints or "") else [])
    propose = lambda cur, good, fails: "USE_T1: prefer t1 for 'a'"
    out = loop_mod.run_tuning_cycle(force=True, propose_fn=propose, select_fn=select, cases=cases)
    assert out["result"] == "promoted"
    assert "USE_T1" in hints_mod.load_hints(force=True)


def test_cycle_rejects_no_improvement(tmp_path, monkeypatch):
    _setup_loop(tmp_path, monkeypatch)
    cases = [_case("a", expected=["t1"])] * 15
    select = lambda hints, message: []  # neither current nor candidate selects t1
    propose = lambda cur, good, fails: "some unhelpful change"
    out = loop_mod.run_tuning_cycle(force=True, propose_fn=propose, select_fn=select, cases=cases)
    assert out["result"] == "rejected"
    assert hints_mod.load_hints(force=True) == ""  # not promoted


def test_cycle_rejects_safety_regression(tmp_path, monkeypatch):
    _setup_loop(tmp_path, monkeypatch)
    cases = [_case("greet", forbidden=["social_publish_post"], must_pass=True, source="safety-seed")] * 15
    # candidate makes it pick the forbidden tool; current does not
    select = lambda hints, message: (["social_publish_post"] if "BAD" in (hints or "") else [])
    propose = lambda cur, good, fails: "BAD change that publishes on greetings"
    out = loop_mod.run_tuning_cycle(force=True, propose_fn=propose, select_fn=select, cases=cases)
    assert out["result"] == "rejected"
    assert "safety" in out["reason"]
    assert hints_mod.load_hints(force=True) == ""


def test_cycle_no_change(tmp_path, monkeypatch):
    _setup_loop(tmp_path, monkeypatch)
    cases = [_case("a", expected=["t1"])] * 15
    out = loop_mod.run_tuning_cycle(force=True, propose_fn=lambda *a: "", select_fn=lambda *a: [], cases=cases)
    assert out["result"] == "no_change"


def test_cycle_skips_insufficient_cases(tmp_path, monkeypatch):
    _setup_loop(tmp_path, monkeypatch)
    out = loop_mod.run_tuning_cycle(force=True, propose_fn=lambda *a: "x", select_fn=lambda *a: [],
                                    cases=[_case("a", expected=["t1"])])
    assert out["skipped"] == "insufficient_cases"


def test_cycle_disabled_without_force(tmp_path, monkeypatch):
    _setup_loop(tmp_path, monkeypatch)
    monkeypatch.setenv("KB_TUNE_ENABLED", "off")
    assert loop_mod.run_tuning_cycle(force=False)["skipped"] == "disabled"


def test_should_run_force_and_not_converged():
    assert loop_mod._should_run({"streak": 99}, force=True) is True
    assert loop_mod._should_run({"streak": 0}, force=False) is True  # still improving → run


def test_streak_resets_on_accept_and_grows_on_reject(tmp_path, monkeypatch):
    _setup_loop(tmp_path, monkeypatch)
    cases = [_case("a", expected=["t1"])] * 15
    # reject → streak grows
    loop_mod.run_tuning_cycle(force=True, propose_fn=lambda *a: "noop", select_fn=lambda *a: [], cases=cases)
    assert loop_mod._load_state()["streak"] == 1
    # accept → streak resets
    loop_mod.run_tuning_cycle(force=True, propose_fn=lambda *a: "USE_T1",
                              select_fn=lambda h, m: (["t1"] if "USE_T1" in h else []), cases=cases)
    assert loop_mod._load_state()["streak"] == 0
