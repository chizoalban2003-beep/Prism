"""
device_hub.py
=============
KDE Sports Agent — Device Registry & Data Ingestion

Discovers, registers, and reads data from all connected devices.
File-based ingestion (watch folders) + HTTP API for networked devices.

SQLite tables:
    devices        — registered hardware devices
    ingested_files — deduplicated file records (sha256 UNIQUE)
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import sqlite3
import threading
import urllib.request
import uuid
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DeviceType(str, Enum):
    GOPRO           = "gopro"
    PHONE_CAMERA    = "phone_camera"
    DRONE           = "drone"
    WEARABLE_WHOOP  = "whoop"
    WEARABLE_GARMIN = "garmin"
    WEARABLE_APPLE  = "apple_watch"
    WEARABLE_OURA   = "oura"
    GPS_TRACKER     = "gps"
    HRM             = "hrm"
    SMART_BALL      = "smart_ball"
    TRACKING_CSV    = "tracking_csv"
    MANUAL          = "manual"


class MediaType(str, Enum):
    VIDEO = "video"   # .mp4 .mov .avi .mkv
    IMAGE = "image"   # .jpg .jpeg .png .heic
    AUDIO = "audio"   # .m4a .wav .mp3
    GPS   = "gps"     # .gpx
    DATA  = "data"    # .csv .json .fit


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Device:
    device_id:   str
    name:        str
    device_type: DeviceType
    watch_path:  str
    api_url:     str  = ""
    api_key:     str  = ""
    enabled:     bool = True
    last_sync:   str  = ""


@dataclass
class IngestedFile:
    file_id:     str
    device_id:   str
    device_type: DeviceType
    media_type:  MediaType
    path:        str
    filename:    str
    size_bytes:  int
    sha256:      str
    ingested_at: str
    metadata:    dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Extension → MediaType mapping
# ---------------------------------------------------------------------------

_EXT_MAP: dict[str, MediaType] = {
    ".mp4": MediaType.VIDEO, ".mov": MediaType.VIDEO,
    ".avi": MediaType.VIDEO, ".mkv": MediaType.VIDEO,
    ".jpg": MediaType.IMAGE, ".jpeg": MediaType.IMAGE,
    ".png": MediaType.IMAGE, ".heic": MediaType.IMAGE,
    ".m4a": MediaType.AUDIO, ".wav": MediaType.AUDIO, ".mp3": MediaType.AUDIO,
    ".gpx": MediaType.GPS,
}


def _classify(path: str) -> MediaType:
    return _EXT_MAP.get(Path(path).suffix.lower(), MediaType.DATA)


# ---------------------------------------------------------------------------
# DeviceHub
# ---------------------------------------------------------------------------

class DeviceHub:
    """
    Manages all connected devices and their incoming data.

    Watch mode: runs a background thread that polls all registered
    device watch_paths every `poll_interval` seconds.  New files
    trigger the registered on_file callback.
    """

    def __init__(
        self,
        db_path:       str = "~/.kde/devices.db",
        poll_interval: int = 30,
    ) -> None:
        self._db_path       = Path(db_path).expanduser()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._poll_interval = poll_interval
        self._stop_event    = threading.Event()
        self._watch_thread: Optional[threading.Thread] = None
        self._init_db()

    # ── DB plumbing ─────────────────────────────────────────────────────────

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS devices (
                    id          TEXT PRIMARY KEY,
                    name        TEXT NOT NULL,
                    device_type TEXT NOT NULL,
                    watch_path  TEXT NOT NULL,
                    api_url     TEXT DEFAULT '',
                    api_key     TEXT DEFAULT '',
                    enabled     INTEGER DEFAULT 1,
                    last_sync   TEXT DEFAULT ''
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ingested_files (
                    id             TEXT PRIMARY KEY,
                    device_id      TEXT NOT NULL,
                    device_type    TEXT NOT NULL,
                    media_type     TEXT NOT NULL,
                    path           TEXT NOT NULL,
                    filename       TEXT NOT NULL,
                    size_bytes     INTEGER NOT NULL,
                    sha256         TEXT UNIQUE NOT NULL,
                    ingested_at    TEXT NOT NULL,
                    processed      INTEGER DEFAULT 0,
                    result_summary TEXT DEFAULT '',
                    metadata_json  TEXT DEFAULT '{}'
                )
            """)

    # ── Device management ───────────────────────────────────────────────────

    def register_device(self, device: Device) -> str:
        """Insert or replace device record.  Returns device_id."""
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO devices "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    device.device_id, device.name,
                    device.device_type.value, device.watch_path,
                    device.api_url, device.api_key,
                    int(device.enabled), device.last_sync,
                ),
            )
        return device.device_id

    def list_devices(self) -> list[Device]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM devices").fetchall()
        return [self._row_to_device(r) for r in rows]

    def enable(self, device_id: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE devices SET enabled=1 WHERE id=?", (device_id,)
            )

    def disable(self, device_id: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE devices SET enabled=0 WHERE id=?", (device_id,)
            )

    def _get_device(self, device_id: str) -> Optional[Device]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM devices WHERE id=?", (device_id,)
            ).fetchone()
        return self._row_to_device(row) if row else None

    @staticmethod
    def _row_to_device(row) -> Device:
        return Device(
            device_id=row["id"],
            name=row["name"],
            device_type=DeviceType(row["device_type"]),
            watch_path=row["watch_path"],
            api_url=row["api_url"],
            api_key=row["api_key"],
            enabled=bool(row["enabled"]),
            last_sync=row["last_sync"],
        )

    # ── Watch mode ──────────────────────────────────────────────────────────

    def start_watching(
        self,
        on_file: Callable[[IngestedFile], None],
    ) -> None:
        """
        Start background polling thread.  Calls on_file for every new file
        found in each enabled device's watch_path.
        """
        if self._watch_thread and self._watch_thread.is_alive():
            logger.warning("DeviceHub: watch thread already running")
            return
        self._stop_event.clear()
        self._watch_thread = threading.Thread(
            target=self._watch_loop,
            args=(on_file,),
            daemon=True,
            name="DeviceHub-Watcher",
        )
        self._watch_thread.start()
        logger.info("DeviceHub: started watching (interval=%ds)", self._poll_interval)

    def stop_watching(self) -> None:
        self._stop_event.set()
        if self._watch_thread:
            self._watch_thread.join(timeout=self._poll_interval + 5)
        logger.info("DeviceHub: watch thread stopped")

    def _watch_loop(self, on_file: Callable[[IngestedFile], None]) -> None:
        while not self._stop_event.wait(self._poll_interval):
            for device in self.list_devices():
                if not device.enabled:
                    continue
                watch_path = Path(device.watch_path)
                if not watch_path.exists():
                    continue
                try:
                    new_files = self.ingest_folder(str(watch_path), device.device_id)
                    for f in new_files:
                        try:
                            on_file(f)
                        except Exception:
                            logger.exception("on_file callback failed for %s", f.file_id)
                except Exception:
                    logger.exception(
                        "watch_loop error for device %s", device.device_id
                    )

    # ── File ingestion ──────────────────────────────────────────────────────

    @staticmethod
    def _sha256(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def ingest_file(self, path: str, device_id: str) -> IngestedFile:
        """
        Manually ingest one file.  Computes SHA256, classifies media type by
        extension, stores record to DB.  Returns existing record if already
        ingested (deduplication by sha256).
        """
        p = Path(path).resolve()
        sha = self._sha256(str(p))

        # Deduplication check
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM ingested_files WHERE sha256=?", (sha,)
            ).fetchone()
        if row:
            return self._row_to_ingested(row)

        device   = self._get_device(device_id)
        dtype    = device.device_type if device else DeviceType.MANUAL
        mtype    = _classify(str(p))
        now      = datetime.now(UTC).isoformat()
        file_id  = uuid.uuid4().hex

        ingested = IngestedFile(
            file_id=file_id,
            device_id=device_id,
            device_type=dtype,
            media_type=mtype,
            path=str(p),
            filename=p.name,
            size_bytes=p.stat().st_size,
            sha256=sha,
            ingested_at=now,
            metadata={},
        )

        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO ingested_files "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    ingested.file_id, ingested.device_id,
                    ingested.device_type.value, ingested.media_type.value,
                    ingested.path, ingested.filename,
                    ingested.size_bytes, ingested.sha256,
                    ingested.ingested_at, 0, "",
                    json.dumps(ingested.metadata),
                ),
            )
        return ingested

    def ingest_folder(self, path: str, device_id: str) -> list[IngestedFile]:
        """Ingest all unprocessed files in a folder recursively."""
        results: list[IngestedFile] = []
        for root, _dirs, files in os.walk(path):
            for fname in files:
                fpath = os.path.join(root, fname)
                try:
                    results.append(self.ingest_file(fpath, device_id))
                except Exception:
                    logger.warning("Failed to ingest %s", fpath, exc_info=True)
        return results

    def list_files(
        self,
        device_id:  Optional[str]       = None,
        media_type: Optional[MediaType]  = None,
        since_days: int                  = 7,
    ) -> list[IngestedFile]:
        clauses: list[str] = []
        params:  list      = []

        if device_id:
            clauses.append("device_id = ?")
            params.append(device_id)
        if media_type:
            clauses.append("media_type = ?")
            params.append(media_type.value)

        cutoff = (datetime.now(UTC) - timedelta(days=since_days)).isoformat()
        clauses.append("ingested_at >= ?")
        params.append(cutoff)

        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        sql   = f"SELECT * FROM ingested_files {where} ORDER BY ingested_at DESC"

        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_ingested(r) for r in rows]

    def mark_processed(self, file_id: str, result_summary: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE ingested_files "
                "SET processed=1, result_summary=? WHERE id=?",
                (result_summary, file_id),
            )

    @staticmethod
    def _row_to_ingested(row) -> IngestedFile:
        return IngestedFile(
            file_id=row["id"],
            device_id=row["device_id"],
            device_type=DeviceType(row["device_type"]),
            media_type=MediaType(row["media_type"]),
            path=row["path"],
            filename=row["filename"],
            size_bytes=row["size_bytes"],
            sha256=row["sha256"],
            ingested_at=row["ingested_at"],
            metadata=json.loads(row["metadata_json"] or "{}"),
        )

    # ── GoPro OpenGoPro HTTP API ─────────────────────────────────────────────

    def sync_gopro(self, device: Device) -> list[IngestedFile]:
        """
        Fetch media list from GoPro via OpenGoPro HTTP API.
        Endpoint: GET http://{device.api_url}/gopro/media/list
        Downloads new files to device.watch_path.
        Ref: https://gopro.github.io/OpenGoPro/http
        """
        url = f"http://{device.api_url}/gopro/media/list"
        try:
            req = urllib.request.Request(
                url, headers={"Accept": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
        except Exception:
            logger.warning("GoPro sync failed for device '%s'", device.name, exc_info=True)
            return []

        watch_path = Path(device.watch_path)
        watch_path.mkdir(parents=True, exist_ok=True)

        ingested: list[IngestedFile] = []
        for media_dir in data.get("media", []):
            directory = media_dir.get("d", "")
            for media_file in media_dir.get("fs", []):
                fname    = media_file.get("n", "")
                file_url = (
                    f"http://{device.api_url}/videos/DCIM/{directory}/{fname}"
                )
                dest = watch_path / fname
                if not dest.exists():
                    try:
                        urllib.request.urlretrieve(file_url, str(dest))
                        logger.info("GoPro: downloaded %s", fname)
                    except Exception:
                        logger.warning(
                            "GoPro: failed to download %s", fname, exc_info=True
                        )
                        continue
                if dest.exists():
                    try:
                        ingested.append(
                            self.ingest_file(str(dest), device.device_id)
                        )
                    except Exception:
                        logger.warning(
                            "GoPro: failed to ingest %s", dest, exc_info=True
                        )
        return ingested

    # ── Apple Health export parser ───────────────────────────────────────────

    def parse_apple_health(self, export_xml_path: str) -> dict:
        """
        Parse Apple Health export.xml using streaming stdlib XML parsing.
        """
        result: dict = {"hrv": [], "sleep": [], "steps": [], "heart_rate": []}
        hrv_by_day: dict[str, dict] = {}
        sleep_by_day: dict[str, dict] = {}
        steps_by_day: dict[str, list[float]] = {}
        heart_by_day: dict[str, list[float]] = {}

        def _parse_date(raw: str) -> Optional[datetime]:
            if not raw:
                return None
            try:
                return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S %z")
            except ValueError:
                return None

        def _mean(values: list[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        for _, elem in ET.iterparse(export_xml_path, events=("end",)):
            if elem.tag != "Record":
                continue

            record_type = elem.get("type", "")
            start_date = elem.get("startDate", "")
            day = start_date[:10]
            value_text = elem.get("value", "")

            try:
                if record_type == "HKQuantityTypeIdentifierHeartRateVariabilitySDNN" and day and value_text:
                    bucket = hrv_by_day.setdefault(day, {"values": [], "unit": elem.get("unit", "ms") or "ms"})
                    bucket["values"].append(float(value_text))

                elif record_type == "HKQuantityTypeIdentifierStepCount" and day and value_text:
                    steps_by_day.setdefault(day, []).append(float(value_text))

                elif record_type == "HKQuantityTypeIdentifierHeartRate" and day and value_text:
                    heart_by_day.setdefault(day, []).append(float(value_text))

                elif record_type == "HKCategoryTypeIdentifierSleepAnalysis":
                    start_dt = _parse_date(start_date)
                    end_dt = _parse_date(elem.get("endDate", ""))
                    sleep_values = {"1", "HKCategoryValueSleepAnalysisAsleep", "HKCategoryValueSleepAnalysisInBed"}
                    is_sleep = value_text in sleep_values
                    if start_dt and end_dt and is_sleep:
                        hours = max(0.0, (end_dt - start_dt).total_seconds() / 3600.0)
                        sleep_day = end_dt.date().isoformat()
                        bucket = sleep_by_day.setdefault(sleep_day, {"hours": [], "quality": []})
                        bucket["hours"].append(hours)
                        bucket["quality"].append(min(hours / 10.0, 1.0))
            except (TypeError, ValueError):
                logger.debug("Skipping Apple Health record", exc_info=True)
            finally:
                elem.clear()

        for day, data in sorted(hrv_by_day.items()):
            result["hrv"].append({
                "date": day,
                "value": round(_mean(data["values"]), 2),
                "unit": data["unit"],
            })
        for day, data in sorted(sleep_by_day.items()):
            hours = round(_mean(data["hours"]), 2)
            result["sleep"].append({
                "date": day,
                "hours": hours,
                "hrs": hours,
                "quality": round(_mean(data["quality"]), 2),
            })
        for day, values in sorted(steps_by_day.items()):
            result["steps"].append({"date": day, "count": int(round(_mean(values)))})
        for day, values in sorted(heart_by_day.items()):
            result["heart_rate"].append({"date": day, "bpm": round(_mean(values), 2)})
        return result

    # ── Garmin Connect CSV parser ────────────────────────────────────────────

    def parse_garmin_csv(self, csv_path: str) -> dict:
        """
        Parse Garmin Connect activity CSV export.
        Returns normalised dict with keys matching DailyContext fields.
        """
        result: dict = {
            "hrv_ms":      None, "sleep_hrs":    None,
            "sleep_score": None, "body_battery": None,
            "resting_hr":  None, "distance_km":  None,
            "calories":    None, "avg_hr":        None,
            "max_hr":      None, "avg_speed_ms":  None,
            "max_speed_ms":None,
            "raw": {},
        }

        with open(csv_path, newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            rows   = list(reader)

        if not rows:
            return result

        # Column-name candidates for each normalised field
        _candidates: dict[str, list[str]] = {
            "hrv_ms":       ["HRV Status", "Avg HRV", "hrv"],
            "sleep_hrs":    ["Sleep Time", "Total Sleep Time", "sleep_hrs"],
            "sleep_score":  ["Sleep Score", "sleep_score"],
            "body_battery": ["Body Battery (Max)", "Body Battery Max", "body_battery"],
            "resting_hr":   ["Resting Heart Rate", "resting_hr"],
            "distance_km":  ["Distance", "Total Distance", "distance"],
            "calories":     ["Calories", "Active Calories", "calories"],
            "avg_hr":       ["Avg HR", "Average HR", "avg_hr"],
            "max_hr":       ["Max HR", "Maximum HR", "max_hr"],
        }

        for row in rows:
            result["raw"].update(dict(row))
            for field_name, candidates in _candidates.items():
                if result[field_name] is None:
                    for col in candidates:
                        val = row.get(col, "")
                        if val and val.strip():
                            try:
                                result[field_name] = float(
                                    val.replace(",", "").strip()
                                )
                                break
                            except ValueError:
                                pass

        return result
