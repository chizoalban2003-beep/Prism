"""/organs/intents route shadow fix for issue #28 bug 45.

Live test: ``GET /organs/intents`` returned ``{"error":"organ 'intents'
not found"}`` even when organs were loaded. The cause was a router
ordering issue:

* ``/organs/intents`` was declared in ``prism_routes_chain``.
* ``/organs/{name}`` was declared in ``prism_routes_infra``.
* ``prism_asgi`` registered ``infra_router`` before ``chain_router``.

FastAPI matches routes in registration order, so the dynamic
``/organs/{name}`` swallowed every request to ``/organs/intents``
before the static route ever got a chance.

Fix: move ``/organs/intents`` into ``prism_routes_infra`` and declare
it **before** ``/organs/{name}`` in the same file. Static-before-dynamic
ordering inside a single router is the durable pattern: it survives
any future reshuffling of ``include_router`` calls.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import prism_state
from prism_routes_chain import router as chain_router
from prism_routes_infra import router as infra_router


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    prism_state._state.clear()


def _agent_with_organs() -> MagicMock:
    organ_loader = MagicMock()
    organ_loader.known_intents.return_value = {"email_send": "Send an email"}
    organ_loader.list_organs.return_value = ["email_organ"]
    organ_loader.organ_details.side_effect = lambda name: (
        {"intent": name} if name == "email_send" else None
    )
    agent = MagicMock()
    agent._organ_loader = organ_loader
    return agent


class TestStaticRouteReachable:
    """The static route must respond, not the dynamic one."""

    def test_organs_intents_hits_static_route(self):
        # Replay the production registration order: infra first, then chain.
        app = FastAPI()
        app.include_router(infra_router)
        app.include_router(chain_router)
        prism_state._state.clear()
        prism_state._state["agent"] = _agent_with_organs()

        resp = TestClient(app).get("/organs/intents")
        # The reported bug: this used to be a 404 with "organ 'intents' not found".
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "organs" in body and "count" in body
        assert body["organs"] == {"email_send": "Send an email"}
        assert body["count"] == 1

    def test_organs_intents_no_agent_still_static(self):
        app = FastAPI()
        app.include_router(infra_router)
        app.include_router(chain_router)
        prism_state._state.clear()

        resp = TestClient(app).get("/organs/intents")
        # Even without an agent, we want the static handler — empty map,
        # not a 404 from the dynamic catch-all.
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"organs": {}}

    def test_dynamic_route_still_works(self):
        # Sanity: the dynamic route is still reachable for other names.
        app = FastAPI()
        app.include_router(infra_router)
        prism_state._state.clear()
        prism_state._state["agent"] = _agent_with_organs()

        ok = TestClient(app).get("/organs/email_send")
        assert ok.status_code == 200
        assert ok.json()["intent"] == "email_send"

        miss = TestClient(app).get("/organs/does_not_exist")
        assert miss.status_code == 404


class TestRouteDeclarationOrder:
    """Pin the file-level invariant: static must appear before dynamic.

    Reordering inside the file is what makes the fix robust against
    future include_router() shuffles. If somebody moves the static
    declaration below the dynamic one we'd be back to the original bug.
    """

    def test_static_declared_before_dynamic_in_infra(self):
        from pathlib import Path

        source = (Path(__file__).resolve().parent.parent
                  / "prism_routes_infra.py").read_text()
        static_idx = source.find('@router.get("/organs/intents")')
        dynamic_idx = source.find('@router.get("/organs/{name}")')
        assert static_idx > 0, "/organs/intents must live in prism_routes_infra"
        assert dynamic_idx > 0, "/organs/{name} must live in prism_routes_infra"
        assert static_idx < dynamic_idx, (
            "/organs/intents must be declared before /organs/{name} — "
            "otherwise the dynamic route shadows it (issue #28-45)."
        )

    def test_chain_router_no_longer_owns_intents(self):
        # If somebody re-adds it in chain_router AND keeps it in infra,
        # we want to know — duplicate routes are a bug too.
        from pathlib import Path

        source = (Path(__file__).resolve().parent.parent
                  / "prism_routes_chain.py").read_text()
        assert '@router.get("/organs/intents")' not in source, (
            "/organs/intents must only be declared in prism_routes_infra; "
            "duplicating it in chain_router re-introduces issue #28-45."
        )
