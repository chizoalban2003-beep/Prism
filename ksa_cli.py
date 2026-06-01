"""
ksa_cli.py
==========
Kinetic State Agent — Command-Line Interface

Entry point: ``python ksa_cli.py`` or ``python -m ksa``

Default database: ~/.ksa/state.db  (directory created if missing)

Commands
--------
run   <prompt>       Route a prompt through the full pipeline and print the outcome.
status               Show all registered tasks and their snapshot versions.
history <task_name>  Print the version history for a task.
rollback <task_name> Revert the task's current snapshot to the previous version.
promote  <task_name> <version>
                     Promote a specific snapshot version to current.
prune    <task_name> [--keep N]
                     Delete old snapshot versions, keeping the N most recent.
delete   <task_name> Remove ALL snapshots for a task.
snapshot load <file> Load a ThreeBarSystem state from a JSON snapshot file
                     and save it to the registry under an inferred task name.

Global flags
------------
--db <path>          Path to the SQLite database (default: ~/.ksa/state.db).
--config <path>      Path to a TOML or JSON config file.
--dry-run            Execute without touching the filesystem or shell.
--verbose            Enable DEBUG-level logging.
--quiet              Suppress all output except errors.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure the package directory is on the path when run as a script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ksa_config import KSAConfig
from ksa_executor import (
    FileIndexExecutor,
    LocalSearchExecutor,
    ShellExecutor,
)
from ksa_agent import KSAgent
from ksa_lever import ThreeBarSystem
from ksa_registry import SnapshotRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool, quiet: bool) -> None:
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.WARNING
    logging.basicConfig(
        level  = level,
        format = "%(levelname)-8s %(name)s: %(message)s",
    )


def _build_agent(args: argparse.Namespace, cfg: KSAConfig) -> KSAgent:
    """Construct and pre-register a KSAgent from config + CLI args."""
    db_path = args.db if args.db else cfg.resolved_db_path
    agent   = KSAgent(
        db_path       = db_path,
        working_dir   = cfg.resolved_working_dir,
        ollama_model  = cfg.ollama_model,
        ollama_host   = cfg.ollama_host,
        auto_optimise = cfg.auto_optimise,
        dry_run       = getattr(args, "dry_run", False) or cfg.dry_run,
    )

    # Register default built-in executors
    _register_builtins(agent)

    # Register any tasks declared in the config file
    executor_map = {
        "FileIndexExecutor":   FileIndexExecutor,
        "LocalSearchExecutor": LocalSearchExecutor,
        "ShellExecutor":       ShellExecutor,
    }
    for tc in cfg.tasks:
        cls = executor_map.get(tc.executor)
        if cls is None:
            logging.getLogger(__name__).warning(
                "Unknown executor class '%s' for task '%s'; skipping.",
                tc.executor,
                tc.task_name,
            )
            continue
        agent.register(
            task_name   = tc.task_name,
            keywords    = tc.keywords,
            executor    = cls(),
            aliases     = tc.aliases,
            description = tc.description,
        )

    return agent


def _register_builtins(agent: KSAgent) -> None:
    """Register the three built-in task executors with default keyword sets."""
    agent.register(
        task_name   = "file_index_stealth",
        keywords    = ["index", "scan", "files", "directory", "folder",
                       "stealth", "background", "quiet", "silently"],
        executor    = FileIndexExecutor(),
        aliases     = ["index"],
        description = "Background file indexing without UI interference",
    )
    agent.register(
        task_name   = "local_search",
        keywords    = ["search", "find", "locate", "grep", "query", "lookup"],
        executor    = LocalSearchExecutor(),
        aliases     = ["search", "find"],
        description = "Low-priority local file/content search",
    )
    agent.register(
        task_name   = "shell_generic",
        keywords    = ["run", "exec", "execute", "shell", "command", "bash"],
        executor    = ShellExecutor(),
        aliases     = ["shell"],
        description = "Generic shell command execution",
    )


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace, cfg: KSAConfig) -> int:
    agent   = _build_agent(args, cfg)
    outcome = agent.run(args.prompt)
    print(outcome)
    if outcome.stdout:
        print(outcome.stdout, end="")
    if outcome.stderr:
        print(outcome.stderr, end="", file=sys.stderr)
    return outcome.return_code


def cmd_status(args: argparse.Namespace, cfg: KSAConfig) -> int:
    db_path  = args.db if args.db else cfg.resolved_db_path
    registry = SnapshotRegistry(db_path)
    tasks    = registry.list_tasks()

    if not tasks:
        print("No tasks registered yet.")
        return 0

    print(f"{'Task':<35} {'Current Version':>15} {'Total Versions':>15}")
    print("-" * 68)
    for t in tasks:
        print(
            f"{t['task_name']:<35} "
            f"{t['current_version']:>15} "
            f"{t['total_versions']:>15}"
        )
    return 0


def cmd_history(args: argparse.Namespace, cfg: KSAConfig) -> int:
    db_path  = args.db if args.db else cfg.resolved_db_path
    registry = SnapshotRegistry(db_path)

    try:
        records = registry.history(args.task_name)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not records:
        print(f"No history for task '{args.task_name}'.")
        return 0

    print(f"History for '{args.task_name}':")
    print(f"  {'Version':>7}  {'Current':>7}  {'Score':>8}  Created")
    print("  " + "-" * 55)
    for rec in records:
        current_mark = "  ✓  " if rec.is_current else "     "
        score_str    = f"{rec.score:.4f}" if rec.score is not None else "  n/a "
        print(f"  v{rec.version:>6}  {current_mark}  {score_str:>8}  {rec.created_at}")
    return 0


def cmd_rollback(args: argparse.Namespace, cfg: KSAConfig) -> int:
    db_path  = args.db if args.db else cfg.resolved_db_path
    registry = SnapshotRegistry(db_path)

    try:
        prev = registry.rollback(args.task_name)
        print(f"Rolled back '{args.task_name}' to v{prev}.")
    except (KeyError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_promote(args: argparse.Namespace, cfg: KSAConfig) -> int:
    db_path  = args.db if args.db else cfg.resolved_db_path
    registry = SnapshotRegistry(db_path)

    try:
        registry.promote(args.task_name, args.version)
        print(f"Promoted '{args.task_name}' to v{args.version}.")
    except KeyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_prune(args: argparse.Namespace, cfg: KSAConfig) -> int:
    db_path  = args.db if args.db else cfg.resolved_db_path
    registry = SnapshotRegistry(db_path)
    keep     = args.keep

    try:
        removed = registry.prune(args.task_name, keep=keep)
        print(f"Pruned {removed} old snapshot(s) for '{args.task_name}' (kept {keep}).")
    except KeyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_delete(args: argparse.Namespace, cfg: KSAConfig) -> int:
    db_path  = args.db if args.db else cfg.resolved_db_path
    registry = SnapshotRegistry(db_path)

    try:
        registry.delete_task(args.task_name)
        print(f"Deleted all snapshots for '{args.task_name}'.")
    except KeyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_snapshot_load(args: argparse.Namespace, cfg: KSAConfig) -> int:
    """Load a ThreeBarSystem JSON snapshot and save it to the registry."""
    db_path  = args.db if args.db else cfg.resolved_db_path
    registry = SnapshotRegistry(db_path)

    snap_path = Path(os.path.expanduser(args.file))
    if not snap_path.exists():
        print(f"Error: file not found: {snap_path}", file=sys.stderr)
        return 1

    task_name = args.task_name if args.task_name else snap_path.stem

    try:
        system  = ThreeBarSystem.load_snapshot(str(snap_path))
        version = registry.save(task_name, system)
        print(f"Loaded snapshot from '{snap_path}' → saved as '{task_name}' v{version}.")
    except Exception as exc:
        print(f"Error loading snapshot: {exc}", file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog        = "ksa",
        description = "Kinetic State Agent — physics-metaphor local AI agent.",
        formatter_class = argparse.RawDescriptionHelpFormatter,
    )

    # Global flags
    parser.add_argument(
        "--db",
        metavar = "PATH",
        default = None,
        help    = "SQLite database path (default: ~/.ksa/state.db)",
    )
    parser.add_argument(
        "--config",
        metavar = "PATH",
        default = None,
        help    = "TOML or JSON config file path",
    )
    parser.add_argument(
        "--dry-run",
        action = "store_true",
        help   = "Run without executing OS commands",
    )
    parser.add_argument(
        "--verbose", "-v",
        action = "store_true",
        help   = "Enable DEBUG logging",
    )
    parser.add_argument(
        "--quiet", "-q",
        action = "store_true",
        help   = "Suppress all output except errors",
    )

    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # run
    p_run = sub.add_parser("run", help="Route a prompt through the KSA pipeline")
    p_run.add_argument("prompt", help="Natural-language prompt to route and execute")

    # status
    sub.add_parser("status", help="Show all tasks and their snapshot versions")

    # history
    p_hist = sub.add_parser("history", help="Show version history for a task")
    p_hist.add_argument("task_name", help="Task name to inspect")

    # rollback
    p_rb = sub.add_parser("rollback", help="Revert a task to the previous snapshot")
    p_rb.add_argument("task_name", help="Task name to roll back")

    # promote
    p_prom = sub.add_parser("promote", help="Promote a snapshot version to current")
    p_prom.add_argument("task_name", help="Task name")
    p_prom.add_argument("version",   type=int, help="Version number to promote")

    # prune
    p_prune = sub.add_parser("prune", help="Remove old snapshot versions")
    p_prune.add_argument("task_name", help="Task name to prune")
    p_prune.add_argument(
        "--keep", type=int, default=5, metavar="N",
        help="Number of recent versions to keep (default: 5)",
    )

    # delete
    p_del = sub.add_parser("delete", help="Remove all snapshots for a task")
    p_del.add_argument("task_name", help="Task name to delete")

    # snapshot sub-group
    p_snap = sub.add_parser("snapshot", help="Snapshot management commands")
    snap_sub = p_snap.add_subparsers(dest="snap_command", metavar="<subcommand>")
    snap_sub.required = True

    p_snap_load = snap_sub.add_parser(
        "load", help="Load a ThreeBarSystem state from a JSON file"
    )
    p_snap_load.add_argument("file", help="Path to the JSON snapshot file")
    p_snap_load.add_argument(
        "--task-name", dest="task_name", default=None,
        help="Task name to save under (default: filename stem)",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    # Require a command (handles all Python versions explicitly)
    if not args.command:
        parser.print_help()
        return 1

    _setup_logging(args.verbose, args.quiet)

    cfg = KSAConfig.load(args.config)

    handlers = {
        "run":      cmd_run,
        "status":   cmd_status,
        "history":  cmd_history,
        "rollback": cmd_rollback,
        "promote":  cmd_promote,
        "prune":    cmd_prune,
        "delete":   cmd_delete,
    }

    if args.command == "snapshot":
        if args.snap_command == "load":
            return cmd_snapshot_load(args, cfg)
        parser.error(f"Unknown snapshot subcommand: {args.snap_command!r}")

    handler = handlers.get(args.command)
    if handler is None:
        parser.error(f"Unknown command: {args.command!r}")

    return handler(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
