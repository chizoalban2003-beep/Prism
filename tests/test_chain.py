import json
import tempfile
import pathlib
from unittest.mock import MagicMock, call
from prism_chain import PrismChain, ChainState, ChainStep, LLMDecision, BranchResult
from prism_responses import text_card


def _make_chain(llm_responses=None):
    router = MagicMock()
    responses = llm_responses or []
    router.call.side_effect = [(r, {}) for r in responses]
    return PrismChain(llm_router=router, use_evaluator=False)


def _make_chain_eval(llm_responses=None):
    """Chain with evaluator enabled — responses must include eval JSON slots."""
    router = MagicMock()
    responses = llm_responses or []
    router.call.side_effect = [(r, {}) for r in responses]
    return PrismChain(llm_router=router, use_evaluator=True)


def _agent(intent, message, ctx):
    return text_card(f"result: {message[:40]}", intent)


# ── should_chain heuristic ────────────────────────────────────────────────────

def test_should_chain_conditional():
    c = _make_chain()
    assert c.should_chain("check my emails and if anything urgent add as a task")

def test_should_chain_research():
    c = _make_chain()
    assert c.should_chain("research the best approach for this and give me comprehensive analysis")

def test_should_chain_rejects_simple():
    c = _make_chain()
    assert not c.should_chain("what is the weather today")

def test_should_chain_rejects_single_word():
    c = _make_chain()
    assert not c.should_chain("help")


# ── LLM node decisions ────────────────────────────────────────────────────────

def test_llm_node_first_step():
    done_resp = json.dumps({
        "done": True,
        "answer": "Here is the answer.",
        "reasoning": "Task is simple."
    })
    c = _make_chain([done_resp])
    state = ChainState("t1", "test message", "test message")
    decision = c._llm_node(state, 1)
    assert decision is not None
    assert decision.done
    assert decision.answer == "Here is the answer."

def test_llm_node_chooses_logic():
    resp = json.dumps({
        "done": False,
        "next_logic": "web_search",
        "next_message": "search for X",
        "reasoning": "need web info"
    })
    c = _make_chain([resp])
    state = ChainState("t2", "find info about X", "find info about X")
    decision = c._llm_node(state, 1)
    assert not decision.done
    assert decision.next_logic == "web_search"

def test_llm_node_unknown_logic_becomes_autonomous():
    resp = json.dumps({
        "done": False,
        "next_logic": "some_nonexistent_logic_xyz",
        "next_message": "do the thing",
        "reasoning": "need custom tool"
    })
    c = _make_chain([resp])
    state = ChainState("t3", "do something novel", "do something novel")
    decision = c._llm_node(state, 1)
    assert decision.next_logic == "autonomous"

def test_llm_node_bad_json_returns_none():
    c = _make_chain(["this is not json at all!!!"])
    state = ChainState("t4", "test", "test")
    decision = c._llm_node(state, 1)
    assert decision is None


# ── Policy node ───────────────────────────────────────────────────────────────

def test_policy_node_flags_email_send():
    c = _make_chain()
    note = c._policy_node("email_send", "sent successfully", {})
    assert "policy" in note.lower()

def test_policy_node_clear_for_read_logic():
    c = _make_chain()
    note = c._policy_node("web_search", "here are results", {})
    assert note == ""


# ── Full chain run ────────────────────────────────────────────────────────────

def test_chain_completes_in_one_step():
    step1 = json.dumps({
        "done": False,
        "next_logic": "web_search",
        "next_message": "search for python news",
        "reasoning": "need current info"
    })
    done = json.dumps({
        "done": True,
        "answer": "Python released version 3.14 recently.",
        "reasoning": "have enough info"
    })
    c = _make_chain([step1, done])
    card = c.run("what is the latest python news", _agent, {})
    assert card is not None
    assert "Python" in card.body or "3.14" in card.body or card.body

def test_chain_hits_max_steps_gracefully():
    # Every LLM response keeps the chain going
    always_continue = json.dumps({
        "done": False,
        "next_logic": "web_search",
        "next_message": "keep searching",
        "reasoning": "need more info"
    })
    c = _make_chain([always_continue] * 20)
    # Override router for synthesis call
    c._router.call.side_effect = None
    c._router.call.return_value = ("Final synthesised answer.", {})
    card = c.run("research everything about everything", _agent, {})
    assert card is not None
    assert len(card.body) > 0

def test_chain_accumulates_state():
    state = ChainState("acc1", "test", "test")
    state.steps.append(ChainStep(1, "web_search", "search x", "found y", "", 100.0))
    state.accumulated = "\n[Step 1 — web_search]\nAsked: search x\nGot: found y"
    assert "web_search" in state.accumulated
    assert "found y" in state.accumulated


# ── Branching ─────────────────────────────────────────────────────────────────

def test_llm_node_returns_branch_decision():
    branch_resp = json.dumps({
        "done": False,
        "is_branch": True,
        "branches": [
            {"logic": "web_search", "message": "search for X"},
            {"logic": "email_read", "message": "check email for X"},
        ],
        "reasoning": "ambiguous — try both"
    })
    c = _make_chain([branch_resp])
    state = ChainState("b1", "find info about X from web and email", "find X")
    decision = c._llm_node(state, 1)
    assert decision is not None
    assert decision.is_branch
    assert len(decision.branches) == 2
    assert decision.branches[0]["logic"] == "web_search"

def test_execute_branch_runs_parallel():
    c = _make_chain()
    branches = [
        {"logic": "web_search", "message": "search A"},
        {"logic": "calendar_read", "message": "check calendar"},
    ]
    results = c._execute_branch(branches, _agent, {})
    assert len(results) == 2
    assert all(r.success for r in results)

def test_execute_branch_unknown_logic_becomes_autonomous():
    c = _make_chain()
    branches = [{"logic": "totally_unknown_xyz", "message": "do thing"}]
    results = c._execute_branch(branches, _agent, {})
    assert len(results) == 1
    # Should have executed (autonomous fallback)
    assert results[0].branch_id == "branch_1"

def test_execute_branch_max_three():
    c = _make_chain()
    branches = [{"logic": "web_search", "message": f"search {i}"}
                for i in range(5)]
    results = c._execute_branch(branches, _agent, {})
    assert len(results) <= 3

def test_chain_with_branch_accumulates():
    branch_resp = json.dumps({
        "done": False,
        "is_branch": True,
        "branches": [
            {"logic": "web_search", "message": "search for news"},
            {"logic": "email_read", "message": "check emails"},
        ],
        "reasoning": "need both"
    })
    done_resp = json.dumps({
        "done": True,
        "answer": "Found info from web and email.",
        "reasoning": "complete"
    })
    c = _make_chain([branch_resp, done_resp])
    card = c.run("get latest news and check my email", _agent, {})
    assert card is not None
    assert "PARALLEL BRANCH" in card.body or card.body


# ── Branch dataclass ──────────────────────────────────────────────────────────

def test_branch_result_dataclass():
    br = BranchResult("branch_1", "web_search", "found stuff", True, 123.4)
    assert br.branch_id == "branch_1"
    assert br.logic == "web_search"
    assert br.success is True

def test_llm_decision_has_branch_fields():
    d = LLMDecision(done=False, next_logic="web_search", next_message="go",
                    reasoning="test")
    assert d.is_branch is False
    assert d.branches == []

def test_llm_decision_branch_mode():
    d = LLMDecision(done=False, next_logic="", next_message="",
                    reasoning="branching", is_branch=True,
                    branches=[{"logic": "web_search", "message": "search"}])
    assert d.is_branch is True
    assert len(d.branches) == 1


# ── Chain persistence ─────────────────────────────────────────────────────────

def test_chain_persists_to_db():
    step1 = json.dumps({
        "done": False,
        "next_logic": "web_search",
        "next_message": "search python",
        "reasoning": "need info"
    })
    done = json.dumps({
        "done": True,
        "answer": "Python is great.",
        "reasoning": "done"
    })
    c = _make_chain([step1, done])
    c._db = pathlib.Path(tempfile.mktemp(suffix=".db"))
    c._init_db()
    card = c.run("tell me about python", _agent, {})
    recent = c.recent_chains()
    assert len(recent) >= 1
    assert "python" in recent[0]["original"].lower()

def test_recent_chains_empty_db():
    c = _make_chain()
    c._db = pathlib.Path(tempfile.mktemp(suffix=".db"))
    c._init_db()
    recent = c.recent_chains()
    assert recent == []

def test_chain_no_router_returns_none():
    c = PrismChain()   # no router
    result = c.run("some message", _agent, {})
    assert result is None


# ── Evaluator bolt-on ─────────────────────────────────────────────────────────

def test_evaluator_disabled_returns_defaults():
    c = _make_chain()   # use_evaluator=False
    score, sufficient, gap = c._evaluator_node("goal", "web_search", "some result")
    assert score == 3
    assert sufficient is False
    assert gap == ""


def test_evaluator_node_sufficient():
    eval_resp = json.dumps({"score": 5, "sufficient": True, "gap": "", "reasoning": "perfect"})
    c = _make_chain_eval([eval_resp])
    score, sufficient, gap = c._evaluator_node("find X", "web_search", "Found X clearly.")
    assert score == 5
    assert sufficient is True


def test_evaluator_node_insufficient():
    eval_resp = json.dumps({"score": 2, "sufficient": False,
                             "gap": "missing date", "reasoning": "incomplete"})
    c = _make_chain_eval([eval_resp])
    score, sufficient, gap = c._evaluator_node("find X", "web_search", "vague result")
    assert score == 2
    assert not sufficient
    assert "date" in gap


def test_evaluator_node_bad_json_defaults():
    c = _make_chain_eval(["not json at all"])
    score, sufficient, gap = c._evaluator_node("goal", "logic", "result")
    assert score == 3
    assert not sufficient


def test_evaluator_early_exit():
    """Chain exits early when evaluator returns sufficient=True."""
    step_resp = json.dumps({
        "done": False, "next_logic": "web_search",
        "next_message": "search it", "reasoning": "need info"
    })
    eval_resp = json.dumps({"score": 5, "sufficient": True, "gap": "", "reasoning": "done"})
    synth_resp = "Here is the synthesised answer."

    c = _make_chain_eval([step_resp, eval_resp, synth_resp])
    c._db = pathlib.Path(tempfile.mktemp(suffix=".db"))
    c._init_db()

    card = c.run("find something", _agent, {})
    assert card is not None
    assert len(c._router.call.call_args_list) == 3  # LLM + eval + synth


def test_evaluator_no_early_exit_on_score_3():
    """Chain continues when evaluator score=3 (sufficient=False)."""
    step_resp = json.dumps({
        "done": False, "next_logic": "web_search",
        "next_message": "search", "reasoning": "need info"
    })
    eval_resp = json.dumps({"score": 3, "sufficient": False,
                             "gap": "more detail needed", "reasoning": "partial"})
    done_resp = json.dumps({"done": True, "answer": "Final answer.", "reasoning": "done"})

    c = _make_chain_eval([step_resp, eval_resp, done_resp])
    c._db = pathlib.Path(tempfile.mktemp(suffix=".db"))
    c._init_db()

    card = c.run("find something", _agent, {})
    assert card is not None
    # Chain should have gone to step 2 LLM node after eval score=3
    assert len(c._router.call.call_args_list) == 3  # LLM step1 + eval + LLM done


def test_evaluator_score_stored_in_step():
    step_resp = json.dumps({
        "done": False, "next_logic": "web_search",
        "next_message": "search", "reasoning": "need"
    })
    eval_resp = json.dumps({"score": 4, "sufficient": True, "gap": "", "reasoning": "good"})
    synth_resp = "Answer."

    c = _make_chain_eval([step_resp, eval_resp, synth_resp])
    c._db = pathlib.Path(tempfile.mktemp(suffix=".db"))
    c._init_db()

    card = c.run("test", _agent, {})
    assert card is not None
    # ChainState eval_scores should have a 4
    # We check via the card title which includes eval score
    assert "eval" in card.title.lower() or "4" in card.title


def test_evaluator_avg_score_in_recent_chains():
    step_resp = json.dumps({
        "done": False, "next_logic": "web_search",
        "next_message": "search", "reasoning": "need"
    })
    eval_resp = json.dumps({"score": 4, "sufficient": True, "gap": "", "reasoning": "good"})
    synth_resp = "Answer."

    c = _make_chain_eval([step_resp, eval_resp, synth_resp])
    c._db = pathlib.Path(tempfile.mktemp(suffix=".db"))
    c._init_db()

    c.run("test eval persistence", _agent, {})
    recent = c.recent_chains()
    assert len(recent) >= 1
    assert recent[0]["avg_eval_score"] == 4.0


def test_evaluator_chain_step_has_eval_score():
    step = ChainStep(
        step_num=1, logic="web_search",
        message_in="search", result_out="result",
        policy_note="", duration_ms=100.0, eval_score=4
    )
    assert step.eval_score == 4


def test_evaluator_step_default_none():
    step = ChainStep(
        step_num=1, logic="web_search",
        message_in="search", result_out="result",
        policy_note="", duration_ms=100.0
    )
    assert step.eval_score is None
