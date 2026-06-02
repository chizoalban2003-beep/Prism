from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CalendarEvent:
    event_id:   str
    title:      str
    start:      datetime
    end:        datetime
    location:   str = ""
    description:str = ""
    attendees:  list[str] = field(default_factory=list)
    calendar:   str = "default"

    @property
    def duration_mins(self) -> int:
        return int((self.end - self.start).total_seconds() / 60)

    @property
    def starts_in_mins(self) -> int:
        delta = self.start - datetime.now()
        return int(delta.total_seconds() / 60)

    def __str__(self) -> str:
        return (f"{self.start.strftime('%a %d %b %H:%M')} — "
                f"{self.title} ({self.duration_mins}min)")


class PrismCalendar:
    """
    Calendar integration via CalDAV (standard) or Google Calendar API.

    Config in prism_config.toml:
      [calendar]
      provider     = "caldav"        # "caldav" | "google" | "ical_url"
      caldav_url   = "https://caldav.example.com/user/calendar/"
      username     = "user@example.com"
      password     = "password"
      # For iCal URL (read-only):
      ical_url     = "webcal://..."
      # For Google Calendar: set up OAuth2 credentials
      google_creds = "~/.prism/google_creds.json"
    """

    def __init__(
        self,
        provider:    str = "",
        caldav_url:  str = "",
        username:    str = "",
        password:    str = "",
        ical_url:    str = "",
        google_creds:str = "",
        google_token:str = "",
    ):
        self._provider     = provider
        self._caldav_url   = caldav_url
        self._username     = username
        self._password     = password
        self._ical_url     = ical_url
        self._google_creds = google_creds
        self._google_token = google_token
        # Auto-detect Google provider when token is provided
        if google_token and not provider:
            self._provider = "google"

    @classmethod
    def from_config(cls, config: dict) -> "PrismCalendar":
        cal = config.get("calendar", {})
        return cls(
            provider     = cal.get("provider", ""),
            caldav_url   = cal.get("caldav_url", ""),
            username     = cal.get("username", ""),
            password     = cal.get("password", ""),
            ical_url     = cal.get("ical_url", ""),
            google_creds = cal.get("google_creds", ""),
            google_token = cal.get("google_token", ""),
        )

    @property
    def configured(self) -> bool:
        return bool(self._provider)

    # ── OAuth2 token management ────────────────────────────────────────────

    def _get_access_token(self) -> str:
        """Return a valid Google access token, refreshing it if expired.

        Reads ``google_creds`` JSON file (contains access_token, refresh_token,
        client_id, client_secret, expiry).  If the access token is expired or
        missing, exchanges the refresh token for a new one via the Google
        OAuth2 token endpoint and writes the updated credentials back to disk.

        Falls back to the bare ``google_token`` config value if no creds file
        is configured.
        """
        if not self._google_creds:
            return self._google_token

        creds_path = Path(self._google_creds).expanduser()
        if not creds_path.exists():
            return self._google_token

        try:
            import json as _json
            creds = _json.loads(creds_path.read_text())
        except Exception as exc:
            logger.warning("PrismCalendar: cannot read creds file: %s", exc)
            return self._google_token

        access_token  = creds.get("access_token", "")
        refresh_token = creds.get("refresh_token", "")
        expiry_str    = creds.get("expiry", "")     # ISO-8601 string
        client_id     = creds.get("client_id", "")
        client_secret = creds.get("client_secret", "")

        # Check whether the current token is still valid (with 60-s margin)
        token_valid = False
        if access_token and expiry_str:
            try:
                from datetime import timezone
                expiry = datetime.fromisoformat(expiry_str.replace("Z", "+00:00"))
                token_valid = expiry > datetime.now(timezone.utc) + timedelta(seconds=60)
            except Exception:
                token_valid = False

        if token_valid:
            self._google_token = access_token
            return access_token

        # Token expired or missing — use refresh_token to get a new one
        if not refresh_token or not client_id or not client_secret:
            logger.warning(
                "PrismCalendar: token expired and no refresh_token/client credentials "
                "available — re-authorisation required"
            )
            return access_token or self._google_token

        try:
            import json as _json
            payload = urllib.parse.urlencode({
                "grant_type":    "refresh_token",
                "refresh_token": refresh_token,
                "client_id":     client_id,
                "client_secret": client_secret,
            }).encode()
            req = urllib.request.Request(
                "https://oauth2.googleapis.com/token",
                data=payload,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = _json.loads(resp.read().decode())

            new_token   = data["access_token"]
            expires_in  = int(data.get("expires_in", 3600))
            from datetime import timezone
            new_expiry  = (datetime.now(timezone.utc) + timedelta(seconds=expires_in)
                           ).isoformat()

            creds["access_token"] = new_token
            creds["expiry"]       = new_expiry
            # Google may rotate the refresh token; persist the new one if provided
            if "refresh_token" in data:
                creds["refresh_token"] = data["refresh_token"]

            creds_path.write_text(_json.dumps(creds, indent=2))
            self._google_token = new_token
            logger.info("PrismCalendar: Google access token refreshed, expires %s",
                        new_expiry)
            return new_token

        except Exception as exc:
            logger.warning("PrismCalendar: token refresh failed: %s", exc)
            return access_token or self._google_token

    # ── Reading events ─────────────────────────────────────────────────────

    def upcoming(self, hours: int = 24) -> list[CalendarEvent]:
        """Return events starting within the next N hours."""
        all_events = self._fetch_events(days_ahead=max(1, hours // 24 + 1))
        cutoff     = datetime.now() + timedelta(hours=hours)
        return [e for e in all_events
                if datetime.now() <= e.start <= cutoff]

    def today(self) -> list[CalendarEvent]:
        """Return all events today."""
        if self._google_token:
            return self._google_today()
        now    = datetime.now()
        events = self._fetch_events(days_ahead=1)
        return [e for e in events if e.start.date() == now.date()]

    def next_event(self) -> Optional[CalendarEvent]:
        """Return the next upcoming event."""
        upcoming = self.upcoming(hours=48)
        return upcoming[0] if upcoming else None

    def find_free_slot(
        self, duration_mins: int = 60,
        within_hours: int = 48
    ) -> Optional[datetime]:
        """Find the next available free slot of given duration."""
        events   = self.upcoming(hours=within_hours)
        now      = datetime.now()
        # Round up to next 30-min boundary
        mins     = now.minute
        start    = now.replace(second=0, microsecond=0)
        if mins % 30 != 0:
            start = start + timedelta(minutes=30 - mins % 30)

        # Walk forward in 30-min increments
        check    = start
        for _ in range(within_hours * 2):
            end  = check + timedelta(minutes=duration_mins)
            busy = any(
                not (end <= e.start or check >= e.end)
                for e in events
            )
            if not busy and check.hour >= 8 and check.hour <= 19:
                return check
            check = check + timedelta(minutes=30)
        return None

    # ── Creating events ────────────────────────────────────────────────────

    def create_event(
        self,
        title:       str,
        start:       datetime,
        duration_mins:int = 60,
        location:    str = "",
        description: str = "",
        attendees:   list[str] = None,
    ) -> Optional[CalendarEvent]:
        """Create a new calendar event."""
        if not self.configured:
            return None
        if self._google_token:
            gcal_id = self._google_create(
                title, start.isoformat(), duration_mins,
                location, attendees)
            if gcal_id:
                return CalendarEvent(
                    event_id    = gcal_id,
                    title       = title,
                    start       = start,
                    end         = start + timedelta(minutes=duration_mins),
                    location    = location,
                    description = description,
                    attendees   = attendees or [],
                )
            return None
        event = CalendarEvent(
            event_id    = f"prism-{int(time.time())}",
            title       = title,
            start       = start,
            end         = start + timedelta(minutes=duration_mins),
            location    = location,
            description = description,
            attendees   = attendees or [],
        )
        ok = self._write_event(event)
        return event if ok else None

    # ── Natural language parsing ───────────────────────────────────────────

    def parse_event_from_text(
        self, text: str, llm_router=None
    ) -> Optional[dict]:
        """
        Parse event details from natural language.
        "Schedule a meeting with Sarah tomorrow at 3pm for 1 hour"
        Returns dict: {title, start_iso, duration_mins, attendees}
        """
        if llm_router is None:
            return None
        prompt = (
            f"Extract calendar event details from this text.\n"
            f"Text: {text}\n"
            f"Today is {datetime.now().strftime('%A %d %B %Y %H:%M')}.\n"
            f"Return ONLY valid JSON:\n"
            f'{{"title":"...","start_iso":"YYYY-MM-DDTHH:MM:00",'
            f'"duration_mins":60,"attendees":[],"location":""}}\n'
            f"If you cannot determine start time, use null for start_iso."
        )
        raw, _ = llm_router.call(prompt, min_capability=1, max_tokens=200,
                                  json_mode=True)
        try:
            clean = raw.strip().lstrip("```json").rstrip("```").strip()
            return json.loads(clean)
        except Exception:
            return None

    def status_summary(self) -> dict:
        if not self.configured:
            return {"configured": False,
                    "message": "Add calendar config to prism_config.toml"}
        today_events = self.today()
        next_ev      = self.next_event()
        return {
            "configured":   True,
            "provider":     self._provider,
            "today_count":  len(today_events),
            "next_event":   str(next_ev) if next_ev else "nothing scheduled",
            "today":        [str(e) for e in today_events],
        }

    # ── Backend implementations ────────────────────────────────────────────

    def _google_today(self) -> list:
        """Fetch today's events from Google Calendar API."""
        from datetime import timezone as _tz
        now   = datetime.now(_tz.utc)
        start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        end   = now.replace(hour=23, minute=59, second=59, microsecond=0).isoformat()
        url   = (f"https://www.googleapis.com/calendar/v3/calendars/primary/events"
                 f"?timeMin={urllib.parse.quote(start)}"
                 f"&timeMax={urllib.parse.quote(end)}"
                 f"&singleEvents=true&orderBy=startTime")
        req = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {self._get_access_token()}"})
        try:
            resp  = urllib.request.urlopen(req, timeout=8)
            items = json.loads(resp.read()).get("items", [])
            return [self._google_item_to_event(i) for i in items]
        except Exception as e:
            logger.warning("Google Calendar fetch failed: %s", e)
            return []

    def _google_create(self, title: str, start_iso: str,
                        duration_mins: int = 60,
                        location: str = "",
                        attendees=None) -> Optional[str]:
        from datetime import timedelta as _td
        start  = datetime.fromisoformat(start_iso)
        end    = start + _td(minutes=duration_mins)
        body   = {
            "summary":  title,
            "location": location,
            "start":    {"dateTime": start.isoformat(), "timeZone": "UTC"},
            "end":      {"dateTime": end.isoformat(),   "timeZone": "UTC"},
            "attendees":[{"email": a} for a in (attendees or [])],
        }
        payload = json.dumps(body).encode()
        req = urllib.request.Request(
            "https://www.googleapis.com/calendar/v3/calendars/primary/events",
            data=payload,
            headers={"Authorization": f"Bearer {self._get_access_token()}",
                     "Content-Type": "application/json"},
            method="POST")
        try:
            resp = urllib.request.urlopen(req, timeout=8)
            return json.loads(resp.read()).get("id")
        except Exception as e:
            logger.warning("Google Calendar create failed: %s", e)
            return None

    def _google_item_to_event(self, item: dict) -> CalendarEvent:
        start_str = item.get("start",{}).get("dateTime",
                    item.get("start",{}).get("date",""))
        end_str   = item.get("end",{}).get("dateTime",
                    item.get("end",{}).get("date",""))
        try:
            start = datetime.fromisoformat(start_str.replace("Z","+00:00"))
        except Exception:
            start = datetime.now()
        try:
            end = datetime.fromisoformat(end_str.replace("Z","+00:00"))
        except Exception:
            end = start + timedelta(hours=1)
        return CalendarEvent(
            event_id    = item.get("id",""),
            title       = item.get("summary","Untitled"),
            start       = start,
            end         = end,
            location    = item.get("location",""),
            description = item.get("description",""),
            attendees   = [a.get("email","")
                           for a in item.get("attendees",[])],
        )

    def _fetch_events(self, days_ahead: int = 7) -> list[CalendarEvent]:
        if self._provider == "ical_url" and self._ical_url:
            return self._fetch_ical(self._ical_url, days_ahead)
        if self._provider == "caldav" and self._caldav_url:
            return self._fetch_caldav(days_ahead)
        return []

    def _fetch_ical(self, url: str, days_ahead: int) -> list[CalendarEvent]:
        """Parse a .ics URL. No dependencies beyond stdlib."""
        try:
            resp = urllib.request.urlopen(url.replace("webcal://", "https://"),
                                          timeout=5)
            text = resp.read().decode(errors="replace")
            return self._parse_ics(text)
        except Exception as e:
            logger.debug("iCal fetch failed: %s", e)
            return []

    def _parse_ics(self, text: str) -> list[CalendarEvent]:
        """Minimal ICS parser — no dependencies."""
        events: list[CalendarEvent] = []
        current: dict = {}
        for line in text.splitlines():
            line = line.strip()
            if line == "BEGIN:VEVENT":
                current = {}
            elif line == "END:VEVENT":
                try:
                    start = self._parse_dt(current.get("DTSTART", ""))
                    end   = self._parse_dt(current.get("DTEND", ""))
                    if start and end:
                        events.append(CalendarEvent(
                            event_id    = current.get("UID", str(time.time())),
                            title       = current.get("SUMMARY", "(untitled)"),
                            start       = start,
                            end         = end,
                            location    = current.get("LOCATION", ""),
                            description = current.get("DESCRIPTION", ""),
                        ))
                except Exception:
                    pass
            elif ":" in line:
                key, _, val = line.partition(":")
                current[key.split(";")[0]] = val
        return events

    @staticmethod
    def _parse_dt(dt_str: str) -> Optional[datetime]:
        for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%dT%H%M%S", "%Y%m%d"):
            try:
                return datetime.strptime(dt_str, fmt)
            except ValueError:
                pass
        return None

    def _fetch_caldav(self, days_ahead: int) -> list[CalendarEvent]:
        """Basic CalDAV REPORT request."""
        try:
            import base64
            creds   = base64.b64encode(
                f"{self._username}:{self._password}".encode()).decode()
            start   = datetime.now().strftime("%Y%m%dT000000Z")
            end_dt  = (datetime.now() + timedelta(days=days_ahead)
                       ).strftime("%Y%m%dT235959Z")
            body    = (
                f'<?xml version="1.0" encoding="utf-8" ?>'
                f'<C:calendar-query xmlns:D="DAV:" xmlns:C="urn:ietf:params:xml:ns:caldav">'
                f'<D:prop><D:getetag/><C:calendar-data/></D:prop>'
                f'<C:filter><C:comp-filter name="VCALENDAR">'
                f'<C:comp-filter name="VEVENT">'
                f'<C:time-range start="{start}" end="{end_dt}"/>'
                f'</C:comp-filter></C:comp-filter></C:filter>'
                f'</C:calendar-query>'
            )
            req = urllib.request.Request(
                self._caldav_url, data=body.encode(),
                headers={"Authorization": f"Basic {creds}",
                         "Content-Type": "application/xml",
                         "Depth": "1"},
                method="REPORT")
            resp = urllib.request.urlopen(req, timeout=10)
            text = resp.read().decode(errors="replace")
            # Extract VCALENDAR blocks and parse
            all_events = []
            for block in text.split("BEGIN:VCALENDAR"):
                if "VEVENT" in block:
                    all_events.extend(self._parse_ics(
                        "BEGIN:VCALENDAR" + block))
            return all_events
        except Exception as e:
            logger.debug("CalDAV fetch failed: %s", e)
            return []

    def _write_event(self, event: CalendarEvent) -> bool:
        """Write event via CalDAV PUT."""
        if self._provider != "caldav":
            logger.info("Calendar write only supported for CalDAV provider")
            return False
        try:
            import base64
            creds = base64.b64encode(
                f"{self._username}:{self._password}".encode()).decode()
            ics = (
                f"BEGIN:VCALENDAR\r\nVERSION:2.0\r\n"
                f"PRODID:-//PRISM//EN\r\n"
                f"BEGIN:VEVENT\r\n"
                f"UID:{event.event_id}\r\n"
                f"SUMMARY:{event.title}\r\n"
                f"DTSTART:{event.start.strftime('%Y%m%dT%H%M%S')}\r\n"
                f"DTEND:{event.end.strftime('%Y%m%dT%H%M%S')}\r\n"
                f"LOCATION:{event.location}\r\n"
                f"DESCRIPTION:{event.description}\r\n"
                f"END:VEVENT\r\nEND:VCALENDAR\r\n"
            )
            url = f"{self._caldav_url.rstrip('/')}/{event.event_id}.ics"
            req = urllib.request.Request(
                url, data=ics.encode(),
                headers={"Authorization": f"Basic {creds}",
                         "Content-Type": "text/calendar"},
                method="PUT")
            urllib.request.urlopen(req, timeout=10)
            return True
        except Exception as e:
            logger.debug("CalDAV write failed: %s", e)
            return False
