"""
tests/test_device_hub.py
========================
Tests for device_hub.py

Covers:
  - register_device
  - file deduplication by sha256
  - media type classification by extension
  - Apple Health XML parse
  - watch callback fires on new file
"""

from __future__ import annotations

import json
import os
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from device_hub import (
    Device,
    DeviceHub,
    DeviceType,
    IngestedFile,
    MediaType,
    _classify,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def hub(tmp_path):
    db = tmp_path / "devices.db"
    return DeviceHub(db_path=str(db), poll_interval=1)


@pytest.fixture
def gopro_device(tmp_path):
    watch = tmp_path / "gopro_watch"
    watch.mkdir()
    return Device(
        device_id="dev-001",
        name="GoPro Hero 12",
        device_type=DeviceType.GOPRO,
        watch_path=str(watch),
    )


@pytest.fixture
def sample_video_file(tmp_path):
    f = tmp_path / "session.mp4"
    f.write_bytes(b"\x00" * 1024)
    return f


@pytest.fixture
def sample_data_file(tmp_path):
    f = tmp_path / "training.csv"
    f.write_text("heart_rate,speed\n150,3.2\n155,3.5\n")
    return f


# ---------------------------------------------------------------------------
# register_device
# ---------------------------------------------------------------------------

class TestRegisterDevice:
    def test_register_returns_device_id(self, hub, gopro_device):
        device_id = hub.register_device(gopro_device)
        assert device_id == "dev-001"

    def test_list_devices_empty(self, hub):
        assert hub.list_devices() == []

    def test_list_devices_after_register(self, hub, gopro_device):
        hub.register_device(gopro_device)
        devices = hub.list_devices()
        assert len(devices) == 1
        d = devices[0]
        assert d.device_id == "dev-001"
        assert d.name == "GoPro Hero 12"
        assert d.device_type == DeviceType.GOPRO
        assert d.enabled is True

    def test_register_multiple_devices(self, hub, tmp_path):
        for i in range(3):
            hub.register_device(
                Device(
                    device_id=f"dev-{i:03d}",
                    name=f"Device {i}",
                    device_type=DeviceType.MANUAL,
                    watch_path=str(tmp_path),
                )
            )
        assert len(hub.list_devices()) == 3

    def test_register_replace_existing(self, hub, gopro_device):
        hub.register_device(gopro_device)
        updated = Device(
            device_id="dev-001",
            name="GoPro Hero 12 Updated",
            device_type=DeviceType.GOPRO,
            watch_path=gopro_device.watch_path,
        )
        hub.register_device(updated)
        devices = hub.list_devices()
        assert len(devices) == 1
        assert devices[0].name == "GoPro Hero 12 Updated"

    def test_enable_disable(self, hub, gopro_device):
        hub.register_device(gopro_device)
        hub.disable("dev-001")
        assert hub.list_devices()[0].enabled is False
        hub.enable("dev-001")
        assert hub.list_devices()[0].enabled is True


# ---------------------------------------------------------------------------
# Media type classification
# ---------------------------------------------------------------------------

class TestMediaTypeClassification:
    @pytest.mark.parametrize("filename,expected", [
        ("clip.mp4",   MediaType.VIDEO),
        ("clip.MOV",   MediaType.VIDEO),
        ("clip.avi",   MediaType.VIDEO),
        ("clip.mkv",   MediaType.VIDEO),
        ("photo.jpg",  MediaType.IMAGE),
        ("photo.jpeg", MediaType.IMAGE),
        ("photo.png",  MediaType.IMAGE),
        ("photo.heic", MediaType.IMAGE),
        ("note.m4a",   MediaType.AUDIO),
        ("note.wav",   MediaType.AUDIO),
        ("note.mp3",   MediaType.AUDIO),
        ("run.gpx",    MediaType.GPS),
        ("data.csv",   MediaType.DATA),
        ("data.json",  MediaType.DATA),
        ("data.fit",   MediaType.DATA),
        ("unknown.xyz",MediaType.DATA),
    ])
    def test_classify(self, filename, expected):
        assert _classify(filename) == expected


# ---------------------------------------------------------------------------
# File ingestion & deduplication
# ---------------------------------------------------------------------------

class TestFileIngestion:
    def test_ingest_file_returns_ingested_file(
        self, hub, gopro_device, sample_video_file
    ):
        hub.register_device(gopro_device)
        ingested = hub.ingest_file(str(sample_video_file), gopro_device.device_id)
        assert isinstance(ingested, IngestedFile)
        assert ingested.filename == "session.mp4"
        assert ingested.media_type == MediaType.VIDEO
        assert len(ingested.sha256) == 64

    def test_ingest_sets_correct_media_type(
        self, hub, gopro_device, sample_data_file
    ):
        hub.register_device(gopro_device)
        ingested = hub.ingest_file(str(sample_data_file), gopro_device.device_id)
        assert ingested.media_type == MediaType.DATA

    def test_deduplication_by_sha256(
        self, hub, gopro_device, sample_video_file
    ):
        hub.register_device(gopro_device)
        f1 = hub.ingest_file(str(sample_video_file), gopro_device.device_id)
        f2 = hub.ingest_file(str(sample_video_file), gopro_device.device_id)
        assert f1.file_id == f2.file_id
        assert f1.sha256 == f2.sha256

    def test_same_content_different_name_deduplicates(
        self, hub, gopro_device, tmp_path
    ):
        hub.register_device(gopro_device)
        content = b"\xDE\xAD\xBE\xEF" * 256
        p1 = tmp_path / "file_a.mp4"
        p2 = tmp_path / "file_b.mp4"
        p1.write_bytes(content)
        p2.write_bytes(content)
        f1 = hub.ingest_file(str(p1), gopro_device.device_id)
        f2 = hub.ingest_file(str(p2), gopro_device.device_id)
        assert f1.file_id == f2.file_id

    def test_different_content_not_deduplicated(
        self, hub, gopro_device, tmp_path
    ):
        hub.register_device(gopro_device)
        p1 = tmp_path / "a.mp4"
        p2 = tmp_path / "b.mp4"
        p1.write_bytes(b"\x01" * 512)
        p2.write_bytes(b"\x02" * 512)
        f1 = hub.ingest_file(str(p1), gopro_device.device_id)
        f2 = hub.ingest_file(str(p2), gopro_device.device_id)
        assert f1.file_id != f2.file_id

    def test_ingest_folder_recursive(self, hub, gopro_device, tmp_path):
        hub.register_device(gopro_device)
        watch = tmp_path / "folder"
        sub   = watch / "sub"
        sub.mkdir(parents=True)
        (watch / "vid.mp4").write_bytes(b"\x01" * 100)
        (sub   / "data.csv").write_bytes(b"a,b\n1,2\n")

        results = hub.ingest_folder(str(watch), gopro_device.device_id)
        assert len(results) == 2
        types = {r.media_type for r in results}
        assert MediaType.VIDEO in types
        assert MediaType.DATA in types

    def test_mark_processed(
        self, hub, gopro_device, sample_video_file
    ):
        hub.register_device(gopro_device)
        ingested = hub.ingest_file(str(sample_video_file), gopro_device.device_id)
        hub.mark_processed(ingested.file_id, "vision_analysis_complete")
        # Verify via list_files: file still listed (processed flag internal)
        files = hub.list_files(device_id=gopro_device.device_id, since_days=1)
        assert any(f.file_id == ingested.file_id for f in files)

    def test_list_files_filter_by_media_type(
        self, hub, gopro_device, tmp_path
    ):
        hub.register_device(gopro_device)
        (tmp_path / "v.mp4").write_bytes(b"\xAA" * 200)
        (tmp_path / "d.csv").write_bytes(b"x,y\n1,2\n")
        hub.ingest_folder(str(tmp_path), gopro_device.device_id)
        videos = hub.list_files(media_type=MediaType.VIDEO, since_days=1)
        assert all(f.media_type == MediaType.VIDEO for f in videos)

    def test_list_files_filter_by_device_id(
        self, hub, tmp_path
    ):
        d1 = Device("d1", "D1", DeviceType.MANUAL, str(tmp_path))
        d2 = Device("d2", "D2", DeviceType.MANUAL, str(tmp_path))
        hub.register_device(d1)
        hub.register_device(d2)
        f = tmp_path / "unique.csv"
        f.write_bytes(b"data")
        hub.ingest_file(str(f), "d1")
        assert hub.list_files(device_id="d1", since_days=1)
        assert not hub.list_files(device_id="d2", since_days=1)


# ---------------------------------------------------------------------------
# Watch callback
# ---------------------------------------------------------------------------

class TestWatchCallback:
    def test_watch_callback_fires_on_new_file(self, hub, gopro_device, tmp_path):
        watch_dir = tmp_path / "watch"
        watch_dir.mkdir()
        device = Device(
            device_id="watch-dev",
            name="Watch Device",
            device_type=DeviceType.MANUAL,
            watch_path=str(watch_dir),
        )
        hub.register_device(device)

        received: list[IngestedFile] = []
        event = threading.Event()

        def on_file(f: IngestedFile) -> None:
            received.append(f)
            event.set()

        hub = DeviceHub(
            db_path=str(tmp_path / "watch_hub.db"),
            poll_interval=1,
        )
        hub.register_device(device)
        hub.start_watching(on_file)

        # Drop a file
        (watch_dir / "new_clip.mp4").write_bytes(b"\xFF" * 512)

        fired = event.wait(timeout=5)
        hub.stop_watching()

        assert fired, "Watch callback did not fire within 5 seconds"
        assert len(received) >= 1
        assert received[0].media_type == MediaType.VIDEO


# ---------------------------------------------------------------------------
# Apple Health XML parser
# ---------------------------------------------------------------------------

class TestAppleHealthParse:
    def _make_xml(self, tmp_path: Path) -> str:
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<HealthData locale="en_US">
  <Record type="HKQuantityTypeIdentifierHeartRateVariabilitySDNN"
          startDate="2024-01-15 08:00:00 +0000"
          endDate="2024-01-15 08:05:00 +0000"
          value="45.3"/>
  <Record type="HKQuantityTypeIdentifierStepCount"
          startDate="2024-01-15 09:00:00 +0000"
          endDate="2024-01-15 10:00:00 +0000"
          value="3200"/>
  <Record type="HKQuantityTypeIdentifierHeartRate"
          startDate="2024-01-15 10:30:00 +0000"
          endDate="2024-01-15 10:30:01 +0000"
          value="72"/>
  <Record type="HKCategoryTypeIdentifierSleepAnalysis"
          startDate="2024-01-15 23:00:00 +0000"
          endDate="2024-01-16 06:30:00 +0000"
          value="HKCategoryValueSleepAnalysisAsleep"/>
</HealthData>"""
        p = tmp_path / "export.xml"
        p.write_text(xml_content)
        return str(p)

    def test_parse_hrv(self, hub, tmp_path):
        xml_path = self._make_xml(tmp_path)
        result   = hub.parse_apple_health(xml_path)
        assert len(result["hrv"]) == 1
        assert result["hrv"][0]["value"] == 45.3
        assert result["hrv"][0]["date"] == "2024-01-15"

    def test_parse_steps(self, hub, tmp_path):
        xml_path = self._make_xml(tmp_path)
        result   = hub.parse_apple_health(xml_path)
        assert len(result["steps"]) == 1
        assert result["steps"][0]["count"] == 3200

    def test_parse_heart_rate(self, hub, tmp_path):
        xml_path = self._make_xml(tmp_path)
        result   = hub.parse_apple_health(xml_path)
        assert len(result["heart_rate"]) == 1
        assert result["heart_rate"][0]["bpm"] == 72.0

    def test_parse_sleep(self, hub, tmp_path):
        xml_path = self._make_xml(tmp_path)
        result   = hub.parse_apple_health(xml_path)
        assert len(result["sleep"]) == 1
        assert result["sleep"][0]["hrs"] == pytest.approx(7.5, abs=0.1)

    def test_parse_returns_all_keys(self, hub, tmp_path):
        xml_path = self._make_xml(tmp_path)
        result   = hub.parse_apple_health(xml_path)
        assert set(result.keys()) == {"hrv", "sleep", "steps", "heart_rate"}


# ---------------------------------------------------------------------------
# Garmin CSV parser
# ---------------------------------------------------------------------------

class TestGarminCsvParse:
    def _make_csv(self, tmp_path: Path) -> str:
        content = (
            "Date,Avg HR,Max HR,Distance,Calories,Body Battery (Max)\n"
            "2024-01-15,145,185,5.2,420,88\n"
        )
        p = tmp_path / "garmin.csv"
        p.write_text(content)
        return str(p)

    def test_parse_garmin_csv(self, hub, tmp_path):
        csv_path = self._make_csv(tmp_path)
        result   = hub.parse_garmin_csv(csv_path)
        assert result["avg_hr"] == 145.0
        assert result["max_hr"] == 185.0
        assert result["calories"] == 420.0
        assert result["body_battery"] == 88.0

    def test_parse_garmin_csv_empty(self, hub, tmp_path):
        p = tmp_path / "empty.csv"
        p.write_text("Date,Avg HR\n")
        result = hub.parse_garmin_csv(str(p))
        assert result["avg_hr"] is None

    def test_parse_garmin_csv_raw_included(self, hub, tmp_path):
        csv_path = self._make_csv(tmp_path)
        result   = hub.parse_garmin_csv(csv_path)
        assert isinstance(result["raw"], dict)
        assert "Avg HR" in result["raw"]
