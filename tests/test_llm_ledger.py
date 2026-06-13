"""
test_llm_ledger.py
==================
Tests for the LLM cost ledger (prism_llm_ledger.py) and
the /analytics/tokens/* API routes.
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prism_asgi import app
from prism_state import _set_state  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ledger(tmp_path: Path):
    from prism_llm_ledger import LLMLedger
    return LLMLedger(db_path=str(tmp_path / "test_ledger.db"))


@pytest.fixture()
def tmp_path():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture()
def ledger(tmp_path):
    return _make_ledger(tmp_path)


@pytest.fixture()
def client(tmp_path):
    from prism_llm_ledger import reset_ledger
    reset_ledger(db_path=str(tmp_path / "api_ledger.db"))
    _set_state(agent=MagicMock())
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# LLMLedger unit tests
# ---------------------------------------------------------------------------

class TestLLMLedgerUnit:

    def test_record_call_returns_record(self, ledger):
        rec = ledger.record_call(
            provider="claude", model="claude-sonnet-4",
            input_tokens=500, output_tokens=200,
            latency_ms=1200, source="chain",
        )
        assert rec.provider == "claude"
        assert rec.model == "claude-sonnet-4"
        assert rec.input_tokens == 500
        assert rec.output_tokens == 200
        assert rec.latency_ms == 1200
        assert rec.cost_usd > 0  # sonnet has a positive price
        assert rec.call_id
        assert rec.timestamp > 0

    def test_ollama_zero_cost(self, ledger):
        rec = ledger.record_call(
            provider="ollama", model="mistral",
            input_tokens=1000, output_tokens=500, latency_ms=300,
        )
        assert rec.cost_usd == 0.0

    def test_stdlib_zero_cost(self, ledger):
        rec = ledger.record_call(
            provider="stdlib", model="stdlib",
            input_tokens=100, output_tokens=50, latency_ms=10,
        )
        assert rec.cost_usd == 0.0

    def test_summary_empty(self, ledger):
        s = ledger.summary()
        assert s["total_calls"] == 0
        assert s["total_cost_usd"] == 0.0
        assert s["total_tokens"] == 0

    def test_summary_after_records(self, ledger):
        ledger.record_call("claude", "claude-sonnet-4", 100, 50, 500)
        ledger.record_call("ollama", "mistral", 200, 100, 200)
        s = ledger.summary()
        assert s["total_calls"] == 2
        assert s["total_input_tokens"] == 300
        assert s["total_output_tokens"] == 150
        assert s["total_tokens"] == 450

    def test_summary_since_ts_filters(self, ledger):
        ledger.record_call("claude", "claude-haiku-4", 100, 50, 300)
        future_ts = time.time() + 3600
        s = ledger.summary(since_ts=future_ts)
        assert s["total_calls"] == 0

    def test_by_model_groups_correctly(self, ledger):
        ledger.record_call("claude", "claude-sonnet-4", 100, 50, 500)
        ledger.record_call("claude", "claude-sonnet-4", 200, 80, 600)
        ledger.record_call("ollama", "mistral", 50, 25, 100)
        rows = ledger.by_model(days=1)
        models = {r["model"] for r in rows}
        assert "claude-sonnet-4" in models
        assert "mistral" in models
        sonnet = next(r for r in rows if r["model"] == "claude-sonnet-4")
        assert sonnet["calls"] == 2
        assert sonnet["input_tokens"] == 300

    def test_by_day_returns_dates(self, ledger):
        ledger.record_call("ollama", "llama3", 100, 50, 200)
        rows = ledger.by_day(days=1)
        assert len(rows) >= 1
        assert "date" in rows[0]
        assert "cost_usd" in rows[0]

    def test_by_source_groups(self, ledger):
        ledger.record_call("ollama", "mistral", 100, 50, 200, source="chain")
        ledger.record_call("ollama", "mistral", 100, 50, 200, source="agent")
        ledger.record_call("ollama", "mistral", 100, 50, 200, source="chain")
        rows = ledger.by_source(days=1)
        sources = {r["source"] for r in rows}
        assert "chain" in sources
        assert "agent" in sources
        chain = next(r for r in rows if r["source"] == "chain")
        assert chain["calls"] == 2

    def test_recent_returns_n_records(self, ledger):
        for _ in range(5):
            ledger.record_call("ollama", "mistral", 10, 5, 100)
        rows = ledger.recent(n=3)
        assert len(rows) == 3
        assert "call_id" in rows[0]

    def test_recent_newest_first(self, ledger):
        ledger.record_call("ollama", "mistral", 10, 5, 100)
        time.sleep(0.01)
        ledger.record_call("ollama", "llama3", 20, 10, 150)
        rows = ledger.recent(n=2)
        assert rows[0]["model"] == "llama3"

    def test_clear_deletes_all(self, ledger):
        ledger.record_call("ollama", "mistral", 10, 5, 100)
        ledger.record_call("ollama", "mistral", 10, 5, 100)
        count = ledger.clear()
        assert count == 2
        assert ledger.summary()["total_calls"] == 0

    def test_price_table_returned(self, ledger):
        pt = ledger.price_table()
        assert "claude-sonnet" in pt
        assert "ollama" in pt
        assert isinstance(pt["claude-sonnet"], list)
        assert len(pt["claude-sonnet"]) == 2

    def test_haiku_cheaper_than_sonnet(self, ledger):
        r_haiku = ledger.record_call("claude", "claude-haiku-4", 1000, 500, 100)
        r_sonnet = ledger.record_call("claude", "claude-sonnet-4", 1000, 500, 100)
        assert r_haiku.cost_usd < r_sonnet.cost_usd

    def test_custom_prices_override(self, tmp_path):
        from prism_llm_ledger import LLMLedger
        custom = {"mymodel": (1.00, 2.00)}
        ledger2 = LLMLedger(db_path=str(tmp_path / "custom.db"), custom_prices=custom)
        rec = ledger2.record_call("custom", "mymodel", 1_000_000, 1_000_000, 0)
        assert rec.cost_usd == pytest.approx(3.0, rel=0.01)

    def test_unknown_model_zero_cost(self, ledger):
        rec = ledger.record_call("mystery", "unknown-model-xyz", 100, 50, 100)
        assert rec.cost_usd == 0.0

    def test_get_ledger_singleton(self, tmp_path):
        from prism_llm_ledger import get_ledger, reset_ledger
        led1 = reset_ledger(db_path=str(tmp_path / "s.db"))
        led2 = get_ledger()
        assert led1 is led2


# ---------------------------------------------------------------------------
# API route tests
# ---------------------------------------------------------------------------

class TestTokensAPI:

    def test_summary_endpoint_ok(self, client):
        r = client.get("/analytics/tokens")
        assert r.status_code == 200
        data = r.json()
        assert "summary" in data
        assert "by_model" in data
        assert "by_source" in data

    def test_summary_counts_after_record(self, client):
        client.post("/analytics/tokens/record", json={
            "provider": "ollama", "model": "mistral",
            "input_tokens": 100, "output_tokens": 50, "latency_ms": 200,
        })
        r = client.get("/analytics/tokens")
        assert r.json()["summary"]["total_calls"] == 1

    def test_daily_endpoint(self, client):
        r = client.get("/analytics/tokens/daily?days=7")
        assert r.status_code == 200
        data = r.json()
        assert "daily" in data
        assert data["days"] == 7

    def test_by_model_endpoint(self, client):
        client.post("/analytics/tokens/record", json={
            "provider": "claude", "model": "claude-sonnet-4",
            "input_tokens": 200, "output_tokens": 100, "latency_ms": 800,
        })
        r = client.get("/analytics/tokens/by-model")
        assert r.status_code == 200
        models = [row["model"] for row in r.json()["by_model"]]
        assert "claude-sonnet-4" in models

    def test_by_source_endpoint(self, client):
        client.post("/analytics/tokens/record", json={
            "provider": "ollama", "model": "mistral",
            "input_tokens": 50, "output_tokens": 25, "latency_ms": 100,
            "source": "chain",
        })
        r = client.get("/analytics/tokens/by-source")
        assert r.status_code == 200
        sources = [row["source"] for row in r.json()["by_source"]]
        assert "chain" in sources

    def test_record_endpoint_returns_call_id(self, client):
        r = client.post("/analytics/tokens/record", json={
            "provider": "ollama", "model": "llama3",
            "input_tokens": 80, "output_tokens": 40, "latency_ms": 150,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True
        assert "call_id" in data
        assert "cost_usd" in data

    def test_record_missing_provider_returns_400(self, client):
        r = client.post("/analytics/tokens/record", json={
            "model": "mistral", "input_tokens": 100, "output_tokens": 50,
        })
        assert r.status_code == 400

    def test_record_missing_model_returns_400(self, client):
        r = client.post("/analytics/tokens/record", json={
            "provider": "ollama", "input_tokens": 100, "output_tokens": 50,
        })
        assert r.status_code == 400

    def test_record_invalid_json_returns_400(self, client):
        r = client.post("/analytics/tokens/record",
                        content=b"not-json",
                        headers={"Content-Type": "application/json"})
        assert r.status_code == 400

    def test_clear_endpoint(self, client):
        client.post("/analytics/tokens/record", json={
            "provider": "ollama", "model": "mistral",
            "input_tokens": 10, "output_tokens": 5, "latency_ms": 50,
        })
        r = client.delete("/analytics/tokens")
        assert r.status_code == 200
        assert r.json()["ok"] is True
        assert r.json()["deleted"] >= 1

        # Verify cleared
        r2 = client.get("/analytics/tokens")
        assert r2.json()["summary"]["total_calls"] == 0

    def test_ollama_call_has_zero_cost(self, client):
        client.post("/analytics/tokens/record", json={
            "provider": "ollama", "model": "mistral",
            "input_tokens": 500, "output_tokens": 250, "latency_ms": 200,
        })
        r = client.get("/analytics/tokens/by-model")
        row = next(x for x in r.json()["by_model"] if x["model"] == "mistral")
        assert row["cost_usd"] == 0.0

    def test_days_parameter_respected(self, client):
        r = client.get("/analytics/tokens?days=7")
        assert r.json()["days"] == 7

    def test_recent_calls_in_summary(self, client):
        for _ in range(3):
            client.post("/analytics/tokens/record", json={
                "provider": "ollama", "model": "llama3",
                "input_tokens": 50, "output_tokens": 25, "latency_ms": 100,
            })
        r = client.get("/analytics/tokens")
        assert r.json()["summary"]["total_calls"] == 3
