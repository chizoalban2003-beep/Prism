from __future__ import annotations

from pathlib import Path

import pytest

from device_hub import DeviceHub


@pytest.fixture
def hub(tmp_path):
    return DeviceHub(db_path=str(tmp_path / "devices.db"), poll_interval=1)


def _write_xml(tmp_path: Path, body: str) -> str:
    path = tmp_path / "export.xml"
    path.write_text(f'<?xml version="1.0" encoding="UTF-8"?><HealthData>{body}</HealthData>', encoding="utf-8")
    return str(path)


def test_parse_apple_health_hrv(hub, tmp_path):
    xml_path = _write_xml(
        tmp_path,
        '<Record type="HKQuantityTypeIdentifierHeartRateVariabilitySDNN" startDate="2024-01-15 07:30:00 +0000" value="52.3" unit="ms" />',
    )
    result = hub.parse_apple_health(xml_path)
    assert result["hrv"]


def test_parse_apple_health_sleep_hours(hub, tmp_path):
    xml_path = _write_xml(
        tmp_path,
        '<Record type="HKCategoryTypeIdentifierSleepAnalysis" startDate="2024-01-14 23:00:00 +0000" endDate="2024-01-15 07:00:00 +0000" value="1" />',
    )
    result = hub.parse_apple_health(xml_path)
    assert result["sleep"][0]["hours"] == pytest.approx(8.0)


def test_parse_apple_health_empty(hub, tmp_path):
    result = hub.parse_apple_health(_write_xml(tmp_path, ""))
    assert result == {"hrv": [], "sleep": [], "steps": [], "heart_rate": []}


def test_parse_apple_health_quality_capped(hub, tmp_path):
    xml_path = _write_xml(
        tmp_path,
        '<Record type="HKCategoryTypeIdentifierSleepAnalysis" startDate="2024-01-14 20:00:00 +0000" endDate="2024-01-15 08:30:00 +0000" value="1" />',
    )
    result = hub.parse_apple_health(xml_path)
    assert result["sleep"][0]["quality"] == 1.0
