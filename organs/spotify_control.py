"""Bundled organ: spotify_control — control Spotify playback via spotipy."""
ORGAN_META = {
    "intent":      "spotify_control",
    "description": "control Spotify playback: play, pause, next, previous, or volume",
    "version":     "1.0",
}

ORGAN_POLICY = {
    "risk_level":        "low",
    "requires_approval": False,
    "irreversible":      False,
    "max_per_session":   None,
}


def _get_config(ctx: dict) -> dict:
    cfg = ctx.get("spotify_config") or {}
    if not cfg:
        try:
            import re
            from pathlib import Path
            env = Path("/proc/self/environ").read_text(errors="replace")
            for key, env_key in [
                ("client_id", "SPOTIFY_CLIENT_ID"),
                ("client_secret", "SPOTIFY_CLIENT_SECRET"),
                ("redirect_uri", "SPOTIFY_REDIRECT_URI"),
            ]:
                m = re.search(rf'{env_key}=([^\x00]+)', env)
                if m:
                    cfg[key] = m.group(1).strip()
        except Exception:
            pass
    return cfg


def _parse_command(message: str) -> tuple:
    """Return (action, arg) where action in {play,pause,next,prev,volume,search}."""
    import re
    msg = message.lower()

    vol_m = re.search(r'volume\s+(?:to\s+)?(\d+)', msg)
    if vol_m:
        return "volume", int(vol_m.group(1))

    search_m = re.search(r'play\s+(.+?)(?:\s+on\s+spotify)?$', message, re.IGNORECASE)
    if "pause" in msg or "stop" in msg:
        return "pause", None
    if "next" in msg or "skip" in msg:
        return "next", None
    if "prev" in msg or "back" in msg or "previous" in msg:
        return "prev", None
    if search_m:
        return "play", search_m.group(1).strip()
    if "play" in msg or "resume" in msg or "start" in msg:
        return "resume", None
    return "status", None


def execute(intent: str, message: str, ctx: dict):
    from prism_responses import text_card

    try:
        import spotipy  # type: ignore[import]
        from spotipy.oauth2 import SpotifyOAuth  # type: ignore[import]
    except ImportError:
        return text_card(
            "spotipy library not installed. Run: pip install spotipy\n"
            "Then set up Spotify credentials in ctx['spotify_config'] with "
            "client_id, client_secret, and redirect_uri.",
            "Spotify",
        )

    cfg = _get_config(ctx)
    client_id = cfg.get("client_id", "")
    client_secret = cfg.get("client_secret", "")
    redirect_uri = cfg.get("redirect_uri", "http://localhost:8888/callback")

    if not client_id or not client_secret:
        return text_card(
            "Spotify credentials not configured.\n"
            "Add spotify_config with client_id and client_secret to ctx,\n"
            "or set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET env vars.",
            "Spotify",
        )

    action, arg = _parse_command(message)

    try:
        sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope=(
                "user-modify-playback-state user-read-playback-state "
                "user-read-currently-playing"
            ),
        ))
    except Exception as exc:
        return text_card(f"Spotify authentication failed: {exc}", "Spotify")

    try:
        if action == "pause":
            sp.pause_playback()
            return text_card("Spotify paused.", "Spotify")
        if action == "resume":
            sp.start_playback()
            return text_card("Spotify resumed.", "Spotify")
        if action == "next":
            sp.next_track()
            return text_card("Skipped to next track.", "Spotify")
        if action == "prev":
            sp.previous_track()
            return text_card("Went back to previous track.", "Spotify")
        if action == "volume":
            vol = max(0, min(100, int(arg or 50)))
            sp.volume(vol)
            return text_card(f"Spotify volume set to {vol}%.", "Spotify")
        if action == "play" and arg:
            results = sp.search(q=arg, limit=1, type="track")
            tracks = results.get("tracks", {}).get("items", [])
            if not tracks:
                return text_card(f"No tracks found for: {arg}", "Spotify")
            track = tracks[0]
            sp.start_playback(uris=[track["uri"]])
            name = track["name"]
            artist = track["artists"][0]["name"]
            return text_card(f"Now playing: {name} — {artist}", "Spotify")
        # Status
        current = sp.current_playback()
        if not current or not current.get("item"):
            return text_card("Spotify: nothing currently playing.", "Spotify")
        item = current["item"]
        name = item.get("name", "Unknown")
        artist = item.get("artists", [{}])[0].get("name", "Unknown")
        is_playing = current.get("is_playing", False)
        status = "Playing" if is_playing else "Paused"
        return text_card(f"Spotify {status}: {name} — {artist}", "Spotify")
    except Exception as exc:
        return text_card(f"Spotify command failed: {exc}", "Spotify")
