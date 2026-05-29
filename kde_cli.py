"""
kde_cli.py
==========
KDE Sports Agent — Unified CLI Entry Point

Usage:
    python kde_cli.py <command> [options]
    kde <command> [options]

Commands:
    setup                    First-time setup wizard
    server | prism           Start local PRISM server + open browser
    morning                  Run morning briefing
    ask "<prompt>"           Natural language task
    session                  Log a session
    evening                  Evening review
    device add               Register a device
    device list              List connected devices
    device sync              Sync all devices
    report                   Generate weekly report
    analyse                  Analyse specific footage
    highlight                Create highlight reel
    reflect                  Show agent's learned state
    status                   System status

Global flags:
    --profile NAME           Use a specific profile (default: last used)
    --config PATH            Config file path (default: ~/.kde/config.toml)
    --verbose                Debug logging
    --dry-run                Describe action without executing
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import webbrowser
from pathlib import Path

# Ensure the package directory is on the path when run as a script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kde_agent import KDEAgent, KDEConfig
from kde_server import KDEServer, DEFAULT_PORT
from kde_config import load_config, build_agent_from_config
from kde_profiles import UserProfile, UserRole, setup_wizard, write_toml
from sports_pro import Role
from device_hub import DeviceType

logger = logging.getLogger(__name__)

_PROFILE_CACHE = Path("~/.kde/.last_profile").expanduser()

_DEVICE_TYPE_ALIASES: dict[str, DeviceType] = {
    "gopro":       DeviceType.GOPRO,
    "phone":       DeviceType.PHONE_CAMERA,
    "drone":       DeviceType.DRONE,
    "whoop":       DeviceType.WEARABLE_WHOOP,
    "garmin":      DeviceType.WEARABLE_GARMIN,
    "apple_watch": DeviceType.WEARABLE_APPLE,
    "apple":       DeviceType.WEARABLE_APPLE,
    "oura":        DeviceType.WEARABLE_OURA,
    "gps":         DeviceType.GPS_TRACKER,
    "hrm":         DeviceType.HRM,
    "csv":         DeviceType.TRACKING_CSV,
    "manual":      DeviceType.MANUAL,
}


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog        = "kde",
        description = "PRISM — local decision intelligence chat and sports platform",
        formatter_class = argparse.RawDescriptionHelpFormatter,
    )

    # Global flags
    parser.add_argument("--profile", metavar="NAME",  help="Profile name to use")
    parser.add_argument("--config",  metavar="PATH",  help="Config file path")
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run",
                        help="Describe action without executing")

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # setup
    sp_setup = sub.add_parser("setup", help="First-time setup wizard")
    sp_setup.add_argument("--name",  required=False, help="Practitioner name")
    sp_setup.add_argument("--role",  required=False, help="Role (athlete|coach|...)")
    sp_setup.add_argument("--sport", required=False, help="Sport")
    sp_setup.add_argument("--team",  required=False, default="", help="Team")

    sp_server = sub.add_parser("server", help="Start local PRISM server", aliases=["prism"])
    sp_server.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind (default 8742)")

    # morning
    sp_morning = sub.add_parser("morning", help="Morning briefing")
    sp_morning.add_argument("--hrv",      type=float, help="HRV in ms")
    sp_morning.add_argument("--sleep",    type=float, help="Sleep hours")
    sp_morning.add_argument("--soreness", type=int,   help="Muscle soreness 1-10")
    sp_morning.add_argument("--energy",   type=int,   help="Energy level 1-10")

    # ask
    sp_ask = sub.add_parser("ask", help="Natural language task")
    sp_ask.add_argument("prompt", nargs="+", help="Your request")

    # session
    sp_session = sub.add_parser("session", help="Log a training session")
    sp_session.add_argument("--rpe",   type=int, required=True, help="Rate of perceived exertion 1-10")
    sp_session.add_argument("--type",  default="training", dest="session_type",
                            help="Session type (training|match|recovery|gym)")
    sp_session.add_argument("--video", metavar="PATH", help="Video folder path")
    sp_session.add_argument("--gps",   metavar="PATH", help="GPS file path")
    sp_session.add_argument("--notes", default="", help="Session notes")

    # evening
    sp_evening = sub.add_parser("evening", help="Evening review")
    sp_evening.add_argument("--rating", type=float, help="Day rating 1-5")
    sp_evening.add_argument("--notes",  default="", help="Notes")

    # device
    sp_device = sub.add_parser("device", help="Device management")
    dev_sub   = sp_device.add_subparsers(dest="device_cmd", metavar="SUBCOMMAND")

    dev_add = dev_sub.add_parser("add", help="Register a device")
    dev_add.add_argument("--name",  required=True, help="Device name")
    dev_add.add_argument("--type",  required=True, help="Device type (gopro|garmin|...)")
    dev_add.add_argument("--path",  required=True, dest="watch_path", help="Watch folder path")
    dev_add.add_argument("--api",   default="",    dest="api_url",    help="Device API URL")

    dev_sub.add_parser("list", help="List connected devices")
    dev_sub.add_parser("sync", help="Sync all devices")

    # report
    sp_report = sub.add_parser("report", help="Generate performance report")
    sp_report.add_argument("--week", type=int, default=0, help="Week offset (0=this week)")
    sp_report.add_argument("--output", metavar="PATH", help="Output file path")

    # analyse
    sp_analyse = sub.add_parser("analyse", help="Analyse specific footage", aliases=["analyze"])
    sp_analyse.add_argument("--video", required=True, metavar="PATH", help="Video file path")
    sp_analyse.add_argument("--no-vision", action="store_true", dest="no_vision",
                            help="Skip Ollama vision analysis")

    # highlight
    sp_hl = sub.add_parser("highlight", help="Create highlight reel")
    sp_hl.add_argument("--week", type=int, default=7, help="Include last N days (default 7)")

    # reflect
    sub.add_parser("reflect", help="Show agent's learned state")

    # status
    sub.add_parser("status", help="System status")

    return parser


# ---------------------------------------------------------------------------
# Agent loader
# ---------------------------------------------------------------------------

def _load_agent(args) -> KDEAgent:
    """Load or create a KDEAgent from config / cache / wizard."""
    config_path = getattr(args, "config", None)

    # If a config file is available, use it
    cfg_file = None
    if config_path:
        cfg_file = config_path
    else:
        for candidate in [
            Path("~/.kde/config.toml").expanduser(),
            Path("~/.kde/kde.toml").expanduser(),
            Path("./prism_config.toml"),
            Path("./kde_config.toml"),
        ]:
            if candidate.exists():
                cfg_file = str(candidate)
                break

    if cfg_file:
        try:
            return build_agent_from_config(cfg_file)
        except Exception as exc:
            logger.warning("Could not build agent from config: %s", exc)

    # Fall back to cached profile
    profile_name = getattr(args, "profile", None) or _read_cached_profile()
    if profile_name:
        cfg = load_config(config_path)
        try:
            agent = KDEAgent.setup(
                name   = profile_name,
                role   = Role.ATHLETE,
                sport  = "general",
                config = cfg,
            )
            # Attempt to reuse existing profile from DB
            status, existing = agent._assistant.get_profile(profile_name)
            if status == "ok":
                agent._profile = existing
            return agent
        except Exception as exc:
            logger.warning("Could not load profile '%s': %s", profile_name, exc)

    # Run wizard
    return _run_setup_wizard(config_path)


def _read_cached_profile() -> str:
    try:
        if _PROFILE_CACHE.exists():
            return _PROFILE_CACHE.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return ""


def _cache_profile(name: str) -> None:
    try:
        _PROFILE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        _PROFILE_CACHE.write_text(name, encoding="utf-8")
    except Exception:
        pass


def _run_setup_wizard(config_path: str = None) -> KDEAgent:
    """Interactive first-time setup."""
    original = os.environ.get("KDE_CONFIG")
    if config_path:
        os.environ["KDE_CONFIG"] = config_path
    try:
        profile = setup_wizard()
    finally:
        if config_path:
            if original is None:
                os.environ.pop("KDE_CONFIG", None)
            else:
                os.environ["KDE_CONFIG"] = original
    agent = KDEAgent.setup(profile=profile)
    _cache_profile(profile.name)
    return agent


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_setup(agent: KDEAgent | None, args) -> None:
    if args.name and args.role and args.sport:
        role_aliases = {
            "developer": UserRole.DEVELOPER,
            "athlete": UserRole.ATHLETE,
            "coach": UserRole.COACH,
            "analyst": UserRole.ANALYST,
            "physio": UserRole.PHYSIO,
            "physiotherapist": UserRole.PHYSIO,
            "agent": UserRole.AGENT,
            "universal": UserRole.UNIVERSAL,
        }
        role = role_aliases.get(args.role.lower(), UserRole.ATHLETE)
        profile = UserProfile(
            name=args.name,
            role=role,
            sport=args.sport,
            team=getattr(args, "team", ""),
        )
        write_toml(profile, getattr(args, "config", None))
        KDEAgent.setup(profile=profile)
        _cache_profile(args.name)
        print(f"✓ Profile created: {args.name} ({role.value}, {args.sport})")
    else:
        _run_setup_wizard(getattr(args, "config", None))


def cmd_morning(agent: KDEAgent, args) -> None:
    if args.dry_run:
        print("[dry-run] Would run morning briefing")
        return
    brief = agent.morning_briefing(
        hrv_ms    = getattr(args, "hrv",      None),
        sleep_hrs = getattr(args, "sleep",    None),
        soreness  = getattr(args, "soreness", None),
        energy    = getattr(args, "energy",   None),
    )
    print(f"\n🌅  MORNING BRIEFING — {brief.time[:10]}")
    print(f"   Focus:       {brief.plan.primary_focus}")
    print(f"   Activation:  {brief.plan.activation:.0%}")
    print(f"   Wearables:   {brief.wearable_summary}")
    if brief.device_status:
        print("   Devices:")
        for d in brief.device_status:
            print(f"     {d}")
    print("   Priority tasks:")
    for t in brief.priority_tasks:
        print(f"     • {t}")
    if brief.alerts:
        print("   ⚠ Alerts:")
        for a in brief.alerts:
            print(f"     {a}")
    print(f"\n   Rationale: {brief.plan.rationale}\n")


def cmd_ask(agent: KDEAgent, args) -> None:
    prompt = " ".join(args.prompt)
    if args.dry_run:
        print(f"[dry-run] Would ask: {prompt}")
        return
    result = agent.ask(prompt)
    print(f"\n Task:    {result.task} (via {result.method})")
    print(f" Success: {result.success}")
    print(f" Time:    {result.elapsed_ms:.0f}ms\n")
    if isinstance(result.output, dict):
        print(json.dumps(result.output, indent=2, default=str))
    elif isinstance(result.output, str):
        print(result.output)
    else:
        print(str(result.output))


def cmd_session(agent: KDEAgent, args) -> None:
    if args.dry_run:
        print(f"[dry-run] Would log session: rpe={args.rpe} type={args.session_type}")
        return
    log = agent.log_session(
        rpe          = args.rpe,
        session_type = args.session_type,
        notes        = args.notes,
        video_folder = getattr(args, "video", None),
        gps_file     = getattr(args, "gps",   None),
    )
    print(f"\n✓ Session logged [{log.session_id[:8]}]")
    print(f"  Type: {log.session_type}  RPE: {log.rpe}")
    if log.metrics:
        print(f"  Distance: {log.metrics.get('distance_m', 0):.0f}m")
        print(f"  Avg HR:   {log.metrics.get('avg_hr', 0):.0f} bpm")
    if log.vision_summary:
        print(f"  Vision:   {log.vision_summary[:80]}…")
    print()


def cmd_evening(agent: KDEAgent, args) -> None:
    if args.dry_run:
        print("[dry-run] Would run evening review")
        return
    review = agent.evening_review(
        day_rating = getattr(args, "rating", None),
        notes      = getattr(args, "notes", ""),
    )
    print(f"\n🌙  EVENING REVIEW — {review.date_str}")
    print(f"   Sessions today: {len(review.session_logs)}")
    print(f"\n   {review.day_rating_prompt}")
    print(f"\n   Recovery protocol:")
    for step in review.recovery_protocol:
        print(f"     • {step}")
    print(f"\n   Sleep target: {review.sleep_target_hrs}h")
    print(f"   Tomorrow:     {review.tomorrow_preview}\n")


def cmd_device(agent: KDEAgent, args) -> None:
    sub = getattr(args, "device_cmd", None)
    if sub == "add":
        dtype = _DEVICE_TYPE_ALIASES.get(args.type.lower(), DeviceType.MANUAL)
        if args.dry_run:
            print(f"[dry-run] Would register device: {args.name} ({dtype.value}) → {args.watch_path}")
            return
        device_id = agent.add_device(
            name        = args.name,
            device_type = dtype,
            watch_path  = args.watch_path,
            api_url     = getattr(args, "api_url", ""),
        )
        print(f"✓ Device registered: {args.name} [{device_id[:8]}]")

    elif sub == "list":
        devices = agent._hub.list_devices()
        if not devices:
            print("No devices registered.")
            return
        print(f"\n{'Name':<25} {'Type':<20} {'Path':<40} Enabled")
        print("-" * 90)
        for d in devices:
            print(f"{d.name:<25} {d.device_type.value:<20} {d.watch_path:<40} {'✓' if d.enabled else '✗'}")
        print()

    elif sub == "sync":
        if args.dry_run:
            print("[dry-run] Would sync all devices")
            return
        result = agent.sync_devices()
        for name, count in result.items():
            status = f"{count} file(s)" if count >= 0 else "error"
            print(f"  {name}: {status}")

    else:
        print("device subcommands: add | list | sync")


def cmd_report(agent: KDEAgent, args) -> None:
    if args.dry_run:
        print(f"[dry-run] Would generate weekly report (offset={args.week})")
        return
    report = agent._workflow.generate_weekly_report(week_offset=args.week)
    if getattr(args, "output", None):
        Path(args.output).expanduser().write_text(report, encoding="utf-8")
        print(f"✓ Report saved to: {args.output}")
    else:
        print(report)


def cmd_analyse(agent: KDEAgent, args) -> None:
    if args.dry_run:
        print(f"[dry-run] Would analyse: {args.video}")
        return
    result = agent.analyze_footage(
        path       = args.video,
        run_vision = not getattr(args, "no_vision", False),
    )
    print(json.dumps(result, indent=2, default=str))


def cmd_highlight(agent: KDEAgent, args) -> None:
    if args.dry_run:
        print(f"[dry-run] Would create highlight reel (last {args.week} days)")
        return
    result = agent.ask(f"highlight reel from last {args.week} days")
    if isinstance(result.output, dict):
        print(json.dumps(result.output, indent=2, default=str))
    else:
        print(result.output)


def cmd_reflect(agent: KDEAgent, args) -> None:
    data = agent.reflect()
    print(json.dumps(data, indent=2, default=str))


def cmd_status(agent: KDEAgent, args) -> None:
    data = agent.status()
    print(f"\n── KDE Agent Status ─────────────────────")
    print(f"   Profile  : {data['profile']} ({data['role']}, {data['sport']})")
    print(f"   Ollama   : {'✓' if data['ollama_available'] else '✗'}")
    print(f"   ffmpeg   : {'✓' if data['ffmpeg_available'] else '✗'}")
    print(f"   Devices  : {len(data['devices'])}")
    for d in data['devices']:
        print(f"             {'✓' if d['enabled'] else '○'}  {d['name']}")
    print(f"   Plans    : {data['plans_this_month']} this month")
    print(f"   Sessions : {data['sessions_this_month']} logged")
    print(f"   Artifacts: {data['artifacts_stored']}")
    print(f"   Fulcrum  : {data['fixed_fulcrum']} ({data['fulcrum_trend']})")
    print()


def cmd_server(agent: KDEAgent, args) -> None:
    server = KDEServer(agent=agent, port=args.port, verbose=getattr(args, "verbose", False))
    if args.dry_run:
        print(f"[dry-run] Would start server at http://localhost:{args.port}")
        return
    try:
        webbrowser.open(server.url)
    except Exception:
        logger.debug("Could not open browser automatically", exc_info=True)
    server.start(blocking=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    # Logging
    level = logging.DEBUG if getattr(args, "verbose", False) else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    command = getattr(args, "command", None)

    if command == "setup":
        cmd_setup(None, args)
        return 0

    if command is None:
        parser.print_help()
        return 0

    try:
        agent = _load_agent(args)
    except KeyboardInterrupt:
        print("\nAborted.")
        return 1
    except Exception as exc:
        print(f"Error loading agent: {exc}", file=sys.stderr)
        logger.exception("Agent load failed")
        return 1

    dispatch = {
        "morning":   cmd_morning,
        "ask":       cmd_ask,
        "session":   cmd_session,
        "evening":   cmd_evening,
        "device":    cmd_device,
        "report":    cmd_report,
        "analyse":   cmd_analyse,
        "analyze":   cmd_analyse,
        "highlight": cmd_highlight,
        "reflect":   cmd_reflect,
        "status":    cmd_status,
        "server":    cmd_server,
        "prism":     cmd_server,
    }

    handler = dispatch.get(command)
    if handler is None:
        print(f"Unknown command: {command}", file=sys.stderr)
        parser.print_help()
        return 1

    try:
        handler(agent, args)
    except KeyboardInterrupt:
        print("\nAborted.")
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        logger.exception("Command '%s' failed", command)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
