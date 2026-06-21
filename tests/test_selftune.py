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


def test_score_recovery_pass_requires_any_real_tool():
    case = {"message": "m", "expected_tools": [], "forbidden_tools": [], "require_any_tool": True}
    assert score.score_prompt([case], _select_from({"m": ["query_cloud_logs"]}), "P")["buckets"]["recovery"]["pass_rate"] == 1.0
    assert score.score_prompt([case], _select_from({"m": []}), "P")["buckets"]["recovery"]["pass_rate"] == 0.0
    # a noise-only selection is still a stall (no real task tool committed)
    assert score.score_prompt([case], _select_from({"m": ["update_progress"]}), "P")["buckets"]["recovery"]["pass_rate"] == 0.0


def test_score_buckets_classify_cases():
    cases = [
        {"message": "h", "expected_tools": ["t"], "human_verified": True},
        {"message": "r", "require_any_tool": True},
        {"message": "m", "expected_tools": ["t"]},
        {"message": "s", "forbidden_tools": ["x"], "must_pass": True},
    ]
    r = score.score_prompt(cases, _select_from({"h": ["t"], "r": ["t"], "m": ["t"], "s": []}), "P")
    assert set(r["buckets"]) == {"human", "recovery", "mined", "safety"}


def _stats(must_pass_ok=True, tokens=100, mean_tools=1.0, **buckets):
    """Build a score_prompt-shaped result from {bucket: pass_rate} kwargs (n=10)."""
    return {
        "must_pass_ok": must_pass_ok,
        "token_estimate": tokens,
        "mean_tools": mean_tools,
        "buckets": {b: {"n": 10, "passed": int(round(r * 10)), "pass_rate": r}
                    for b, r in buckets.items()},
    }


def test_gate_promotes_on_human_verified_improvement():
    cur = _stats(human=0.60, mined=0.80)
    cand = _stats(human=0.80, mined=0.80, tokens=120)  # human up, mined flat → promote
    ok, reason = score.gate(cur, cand)
    assert ok and "human" in reason


def test_gate_promotes_on_recovery_improvement():
    cur = _stats(recovery=0.40, mined=0.80)
    cand = _stats(recovery=0.70, mined=0.80)  # anti-stall improved
    ok, reason = score.gate(cur, cand)
    assert ok and "recovery" in reason


def test_gate_mined_alone_cannot_promote():
    # Self-labeled mined improves but no trusted driver moves → must NOT promote.
    cur = _stats(mined=0.40, human=0.80)
    cand = _stats(mined=0.90, human=0.80)
    assert score.gate(cur, cand)[0] is False


def test_gate_rejects_token_saving_without_driver():
    # Old code promoted equal-rate-fewer-tokens; new code must reject it.
    cur = _stats(human=0.80, mined=0.80, tokens=200)
    cand = _stats(human=0.80, mined=0.80, tokens=150)
    assert score.gate(cur, cand)[0] is False


def test_gate_rejects_floor_regression_even_if_driver_improves():
    # human improves but mined (a floor) regresses → reject.
    cur = _stats(human=0.50, mined=0.80)
    cand = _stats(human=0.90, mined=0.60)
    ok, reason = score.gate(cur, cand)
    assert ok is False and "mined" in reason


def test_gate_rejects_safety_regression_even_if_better():
    cur = _stats(human=0.70)
    cand = _stats(human=0.95, must_pass_ok=False, tokens=90)
    ok, reason = score.gate(cur, cand)
    assert ok is False and "safety" in reason


def test_gate_rejects_single_case_flip():
    # One flipped case can be temp-0 sampling noise — require >= MIN_IMPROVE_CASES.
    cur = _stats(human=0.70)   # 7/10
    cand = _stats(human=0.80)  # 8/10 → +1 case only
    assert score.gate(cur, cand)[0] is False


def test_quality_gain_prefers_precise_over_greedy():
    cur = _stats(recovery=0.20, human=0.80, mean_tools=1.0)
    greedy = _stats(recovery=1.00, human=0.80, mean_tools=3.0)   # saturates recovery by over-calling
    precise = _stats(recovery=0.60, human=1.00, mean_tools=1.0)  # real gains, no tool bloat
    assert score.quality_gain(cur, precise) > score.quality_gain(cur, greedy)


# ---- anti-reward-hacking: write-deny, over-call, recovery safety -------------

def test_case_passes_safety_denies_any_write_tool():
    greeting = {"message": "hi", "expected_tools": [], "forbidden_tools": [], "must_pass": True}
    assert score._case_passes(greeting, {"fedramp_commit_document"}) is False  # write → veto
    assert score._case_passes(greeting, {"send_teams_message"}) is False
    assert score._case_passes(greeting, {"query_cloud_logs"}) is True          # read tool ok


def test_case_passes_rejects_overcalling():
    case = {"message": "m", "expected_tools": ["query_cloud_logs"], "forbidden_tools": []}
    assert score._case_passes(case, {"query_cloud_logs"}) is True
    assert score._case_passes(case, {"query_cloud_logs", "x"}) is True          # 1 extra tolerated
    assert score._case_passes(case, {"query_cloud_logs", "x", "y"}) is False    # sprayed extras


def test_case_passes_recovery_requires_safe_small_set():
    rec = {"message": "m", "require_any_tool": True, "forbidden_tools": []}
    assert score._case_passes(rec, {"query_cloud_logs"}) is True
    assert score._case_passes(rec, {"social_publish_post"}) is False  # escaping a stall via a WRITE
    assert score._case_passes(rec, {"a", "b", "c"}) is False          # tool spray (> MAX_RECOVERY_TOOLS)
    assert score._case_passes(rec, set()) is False                    # still stalled


def test_case_passes_recovery_rejects_meta_only_commitment():
    # A generic "when unsure, call search_wiki" hint must NOT count as recovery —
    # a meta lookup isn't a task commitment, and search_wiki is always bound (core).
    rec = {"message": "m", "require_any_tool": True, "forbidden_tools": []}
    assert score._case_passes(rec, {"search_wiki"}) is False
    assert score._case_passes(rec, {"read_knowledge"}) is False
    assert score._case_passes(rec, {"search_wiki", "query_cloud_logs"}) is True  # has a real task tool


def test_write_tools_cover_known_mutators():
    from selftune.evalset import WRITE_TOOLS
    for t in ("social_publish_post", "fedramp_commit_document", "send_teams_message",
              "github_close_issue", "oscal_update_control", "create_background_task",
              "smartsheet_update_row", "oscal_generate_ssp", "oscal_generate_poam"):
        assert t in WRITE_TOOLS
    assert "query_cloud_logs" not in WRITE_TOOLS  # read tools stay allowed


# Every bound tool must be explicitly classified as a WRITE, META, NOISE, or
# READ-safe tool. A NEW/renamed tool falls into none → this FAILS, forcing a
# human to decide write-vs-read instead of a mutator silently defaulting to
# "safe" (exactly how oscal_generate_ssp/poam slipped past WRITE_TOOLS).
_READ_SAFE_TOOLS = {
    "check_gcp_metrics", "gcp_billing_summary", "query_cloud_logs", "list_cloud_run_services",
    "investigate", "google_search",
    "sync_repo", "read_repo_file", "read_repo_file_range", "search_repo_code", "list_repo_files",
    "search_troubleshooting",
    "github_get_commit_diff", "github_get_issue_details", "github_get_issue_type",
    "github_get_pr_details", "github_get_project_items", "github_get_workflow_run_details",
    "github_list_issues", "github_list_projects", "github_list_prs",
    "github_list_recent_commits", "github_list_workflow_runs", "github_search_issues",
    "fedramp_check_dependabot_alerts", "fedramp_check_encryption", "fedramp_check_failed_logins",
    "fedramp_check_iam_compliance", "fedramp_check_log_retention", "fedramp_check_scc_findings",
    "fedramp_collect_evidence", "fedramp_daily_log_review", "fedramp_evidence_summary",
    "fedramp_list_documents", "fedramp_poam_status", "fedramp_read_document", "fedramp_review_code",
    "fedramp_scan_container_vulnerabilities", "fedramp_search_documents", "fedramp_discard_draft",
    "oscal_catalog_lookup", "oscal_generate_assessment_results", "oscal_render_pdf",
    "oscal_validate_package",
    "smartsheet_get_sheet", "smartsheet_list_sheets",
    "get_tracker_diagnostics",  # read-only: serves pre-computed tracker diagnoses
    "social_generate_image", "social_get_post", "social_list_scheduled", "social_preview_post",
    "list_background_tasks", "list_team_members", "lookup_team_member",
    # consent-gated file mutators (staged behind a Teams FileConsentCard — not an
    # autonomous outward write; promote to WRITE_TOOLS if the consent step is removed)
    "edit_document", "edit_spreadsheet", "fill_spreadsheet_column",
    "get_spreadsheet_info", "get_uploaded_file_content", "read_spreadsheet_cells",
}


def test_every_bound_tool_is_classified():
    import agent
    from selftune.evalset import META_TOOLS, NOISE_TOOLS, WRITE_TOOLS

    bound = {t.name for t in agent.ALL_TOOLS if getattr(t, "name", None)}
    classified = WRITE_TOOLS | META_TOOLS | NOISE_TOOLS | _READ_SAFE_TOOLS
    unclassified = bound - classified
    assert not unclassified, (
        "Unclassified bound tools — add each to WRITE_TOOLS (if it mutates external "
        f"state) or _READ_SAFE_TOOLS (if read-only): {sorted(unclassified)}"
    )


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


def _case(msg, expected=None, forbidden=None, must_pass=False, source="mined",
          human_verified=False, require_any_tool=False):
    return {"message": msg, "expected_tools": expected or [], "forbidden_tools": forbidden or [],
            "must_pass": must_pass, "source": source, "human_verified": human_verified,
            "require_any_tool": require_any_tool}


def test_cycle_promotes_improvement(tmp_path, monkeypatch):
    _setup_loop(tmp_path, monkeypatch)
    # A human-verified driver case — only a trusted bucket can justify promotion.
    cases = [_case("a", expected=["t1"], source="human-verified", human_verified=True)] * 15
    select = lambda hints, message: (["t1"] if "USE_T1" in (hints or "") else [])
    propose = lambda cur, good, fails: "USE_T1: prefer t1 for 'a'"
    out = loop_mod.run_tuning_cycle(force=True, propose_fn=propose, select_fn=select, cases=cases)
    assert out["result"] == "promoted"
    assert "USE_T1" in hints_mod.load_hints(force=True)


def test_cycle_mined_only_does_not_promote(tmp_path, monkeypatch):
    # Even though the candidate fixes the mined cases, mined is floor-only.
    _setup_loop(tmp_path, monkeypatch)
    cases = [_case("a", expected=["t1"])] * 15  # source defaults to 'mined'
    select = lambda hints, message: (["t1"] if "USE_T1" in (hints or "") else [])
    propose = lambda cur, good, fails: "USE_T1: prefer t1"
    out = loop_mod.run_tuning_cycle(force=True, propose_fn=propose, select_fn=select, cases=cases)
    assert out["result"] == "rejected"
    assert hints_mod.load_hints(force=True) == ""


def test_cycle_promotes_on_recovery(tmp_path, monkeypatch):
    # An anti-stall recovery case: pass = selects ANY tool. A hint that makes the
    # model commit to a tool (instead of stalling) drives the promotion.
    _setup_loop(tmp_path, monkeypatch)
    cases = [_case("stalled msg", source="recovery", require_any_tool=True)] * 15
    select = lambda hints, message: (["t1"] if "ACT" in (hints or "") else [])
    propose = lambda cur, good, fails: "ACT: commit to a tool, don't ask permission"
    out = loop_mod.run_tuning_cycle(force=True, propose_fn=propose, select_fn=select, cases=cases)
    assert out["result"] == "promoted"


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
    cases = [_case("a", expected=["t1"], source="human-verified", human_verified=True)] * 15
    # reject → streak grows
    loop_mod.run_tuning_cycle(force=True, propose_fn=lambda *a: "noop", select_fn=lambda *a: [], cases=cases)
    assert loop_mod._load_state()["streak"] == 1
    # accept → streak resets
    loop_mod.run_tuning_cycle(force=True, propose_fn=lambda *a: "USE_T1",
                              select_fn=lambda h, m: (["t1"] if "USE_T1" in h else []), cases=cases)
    assert loop_mod._load_state()["streak"] == 0
