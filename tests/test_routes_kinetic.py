"""
Tests for prism_routes_kinetic — /kinetic/* REST endpoints.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from prism_kinetic_engine import KineticEngine
from prism_routes_kinetic import get_or_set_engine, router


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(router)
    eng = KineticEngine.for_prism()
    get_or_set_engine(eng)
    return TestClient(app)


class TestKineticStatus:
    def test_status_200(self, client):
        r = client.get("/kinetic/status")
        assert r.status_code == 200

    def test_status_has_levers(self, client):
        data = client.get("/kinetic/status").json()
        assert "levers" in data
        assert len(data["levers"]) == 3

    def test_status_has_links(self, client):
        data = client.get("/kinetic/status").json()
        assert "links" in data
        assert len(data["links"]) == 6

    def test_status_has_compound_phi(self, client):
        data = client.get("/kinetic/status").json()
        assert "compound_phi_delta" in data
        assert isinstance(data["compound_phi_delta"], float)

    def test_status_windows_1h_is_int(self, client):
        data = client.get("/kinetic/status").json()
        assert isinstance(data["windows_1h"], int)


class TestKineticWindows:
    def test_windows_200(self, client):
        r = client.get("/kinetic/windows")
        assert r.status_code == 200

    def test_windows_has_count(self, client):
        data = client.get("/kinetic/windows").json()
        assert "count" in data
        assert "windows" in data
        assert isinstance(data["windows"], list)

    def test_windows_max_age_param(self, client):
        r = client.get("/kinetic/windows?max_age=0")
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 0


class TestKineticSignalIngest:
    def test_ingest_valid_signal(self, client):
        payload = {
            "domain": "health",
            "signal_type": "hrv_drop",
            "raw_value": 42.0,
            "mu": 65.0,
            "sigma": 12.0,
            "impact": 0.8,
            "confidence": 0.9,
        }
        r = client.post("/kinetic/signal", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert "windows" in data

    def test_ingest_missing_field_422(self, client):
        payload = {"domain": "health"}  # missing required fields
        r = client.post("/kinetic/signal", json=payload)
        assert r.status_code == 422

    def test_ingest_invalid_type_422(self, client):
        payload = {
            "domain": "health", "signal_type": "hrv",
            "raw_value": "not_a_number",
            "mu": 60.0, "sigma": 10.0,
        }
        r = client.post("/kinetic/signal", json=payload)
        assert r.status_code == 422

    def test_ingest_crisis_signal_fires_window(self, client):
        payload = {
            "domain": "health", "signal_type": "hrv_drop",
            "raw_value": 90.0, "mu": 0.0, "sigma": 10.0,
        }
        r = client.post("/kinetic/signal", json=payload)
        data = r.json()
        # Crisis (Z=9) should generate windows
        if data["windows"]:
            assert data["windows"][0]["is_crisis"] is True
