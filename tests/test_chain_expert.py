import json
from unittest.mock import MagicMock
from prism_chain_expert import (
    PrismChainExpert, ExpertChainState, ROUTER_PROMPT, EVALUATOR_PROMPT, BRANCH_JUDGE_PROMPT, SYNTHESISER_PROMPT,
)
from prism_responses import text_card
import tempfile, pathlib


def _agent(intent, message, ctx):
    return text_card(f"result of {intent}", intent)


def _make_expert(responses=None):
    router = MagicMock()
    responses = responses or []
    router.call.side_effect = [(r, {}) for r in responses]
    e = PrismChainExpert(llm_router=router)
    e._db = pathlib.Path(tempfile.mktemp(suffix=".db"))
    e._init_db()
    return e


def test_branch_judge_no_branch():
    resp = json.dumps({"branch": False, "reasoning": "clear single path"})
    e = _make_expert([resp])
    state = ExpertChainState("t1","find weather","find weather")
    result = e._branch_judge(state, 1)
    assert result["branch"] == False


def test_branch_judge_branches():
    resp = json.dumps({
        "branch": True,
        "paths": [
            {"logic": "web_search", "message": "search news"},
            {"logic": "email_read", "message": "check email"},
        ],
        "reasoning": "two independent sources"
    })
    e = _make_expert([resp])
    state = ExpertChainState("t2","news and email","news and email")
    result = e._branch_judge(state, 1)
    assert result["branch"] == True
    assert len(result["paths"]) == 2


def test_router_node_returns_logic():
    resp = json.dumps({"logic":"web_search","message":"search X","reasoning":"need web"})
    e = _make_expert([resp])
    state = ExpertChainState("t3","find X","find X")
    result = e._router_node(state, 1)
    assert result["logic"] == "web_search"


def test_router_node_bad_json_returns_none():
    e = _make_expert(["not valid json!!!"])
    state = ExpertChainState("t4","test","test")
    result = e._router_node(state, 1)
    assert result is None


def test_evaluator_scores_result():
    resp = json.dumps({"score":4,"sufficient":True,"gap":"","reasoning":"good"})
    e = _make_expert([resp])
    state = ExpertChainState("t5","test","test")
    result = e._evaluator_node(state, "web_search", "some result", 1)
    assert result["score"] == 4
    assert result["sufficient"] == True


def test_evaluator_identifies_gap():
    resp = json.dumps({"score":2,"sufficient":False,"gap":"missing date info","reasoning":"incomplete"})
    e = _make_expert([resp])
    state = ExpertChainState("t6","test","test")
    result = e._evaluator_node(state, "web_search", "partial result", 1)
    assert result["sufficient"] == False
    assert "date" in result["gap"]


def test_synthesiser_returns_string():
    e = _make_expert(["Final answer: the task is done."])
    state = ExpertChainState("t7","what is X","what is X")
    state.accumulated = "\n[Step 1] Got: X is Y"
    result = e._synthesiser_node(state)
    assert isinstance(result, str)
    assert len(result) > 0


def test_execute_branches_parallel():
    e = _make_expert()
    paths = [
        {"logic":"web_search","message":"search A"},
        {"logic":"calendar_read","message":"check calendar"},
    ]
    results = e._execute_branches(paths, _agent, {})
    assert len(results) == 2
    assert all(isinstance(r, str) for r in results)


def test_full_run_single_path():
    # branch_judge → no branch; router → web_search; evaluator → sufficient; synthesiser
    responses = [
        json.dumps({"branch": False, "reasoning": "single"}),
        json.dumps({"logic":"web_search","message":"search","reasoning":"need web"}),
        json.dumps({"score":5,"sufficient":True,"gap":"","reasoning":"perfect"}),
        "The answer is: Python is great.",
    ]
    e = _make_expert(responses)
    card = e.run("tell me about python", _agent, {})
    assert card is not None
    assert "Python" in card.body or card.body


def test_full_run_branch_path():
    # Step 1: branch_judge → branch (no evaluator for branch paths)
    # Step 2: branch_judge → single; router → web_search; evaluator → sufficient
    # Then: synthesiser
    responses = [
        # step 1: branch judge → branch
        json.dumps({"branch":True,"paths":[
            {"logic":"web_search","message":"search news"},
            {"logic":"email_read","message":"check email"},
        ],"reasoning":"two sources"}),
        # step 2: branch judge → no branch (have enough context now)
        json.dumps({"branch":False,"reasoning":"have results already"}),
        # step 2: router
        json.dumps({"logic":"web_search","message":"summarise","reasoning":"wrap up"}),
        # step 2: evaluator → sufficient
        json.dumps({"score":4,"sufficient":True,"gap":"","reasoning":"enough"}),
        # synthesiser
        "Gathered info from web and email successfully.",
    ]
    e = _make_expert(responses)
    card = e.run("get news and check email", _agent, {})
    assert card is not None
    assert card.body


def test_node_traces_recorded():
    responses = [
        json.dumps({"branch": False, "reasoning": "single"}),
        json.dumps({"logic":"web_search","message":"search","reasoning":"need web"}),
        json.dumps({"score":4,"sufficient":True,"gap":"","reasoning":"good"}),
        "Final answer here.",
    ]
    e = _make_expert(responses)
    state = ExpertChainState("trace_test","test","test")
    # Run through nodes manually
    e._branch_judge(state, 1)
    e._router_node(state, 1)
    assert len(state.traces) == 2
    assert state.traces[0].node == "branch_judge"
    assert state.traces[1].node == "router"


def test_prompts_have_required_sections():
    assert "ROUTER" in ROUTER_PROMPT
    assert "EVALUATOR" in EVALUATOR_PROMPT
    assert "BRANCH JUDGE" in BRANCH_JUDGE_PROMPT
    assert "SYNTHESISER" in SYNTHESISER_PROMPT


def test_pyproject_has_chain_expert():
    content = (pathlib.Path(__file__).parent.parent / "pyproject.toml").read_text()
    assert "prism_chain_expert" in content
