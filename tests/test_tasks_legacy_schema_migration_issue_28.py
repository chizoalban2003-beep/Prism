"""Tasks-table migration for issue #28 bug 10 — legacy 3-column schema crashed add/list.

Live test: ``add task: buy milk`` returned ``Error: table tasks has 3
columns but 10 values were supplied`` and ``list my tasks`` returned
``no such column: done``. The user's ~/.prism/tasks.db was created by
an earlier PRISM version that stored the background-task queue in the
same file under a ``tasks`` table with the schema
``(id TEXT, status TEXT, data_json TEXT)``. The task-queue store has
since moved to ~/.prism/task_queue.db, but PrismTasks (the to-do store)
still uses ~/.prism/tasks.db with the new 10-column schema.

The CREATE IF NOT EXISTS in _init_db is a no-op when the old table is
already there, so every INSERT crashed. Fix: on init, inspect the
current column list and ALTER ... RENAME TO tasks_legacy_task_queue if
the schema doesn't match the expected to-do columns. The new table is
then created cleanly. Legacy data is preserved under the renamed table
in case the user wants to inspect it.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from prism_tasks import PrismTasks


@pytest.fixture()
def legacy_db(tmp_path: Path) -> Path:
    """A tasks.db pre-populated with the old 3-column schema."""
    db = tmp_path / "tasks.db"
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE tasks(id TEXT PRIMARY KEY, status TEXT, data_json TEXT)")
        c.execute("INSERT INTO tasks VALUES (?,?,?)",
                  ("legacy-1", "completed", '{"task_id":"legacy-1","title":"old"}'))
    return db


class TestMigrationRenamesOldTable:
    def test_init_rewrites_schema(self, legacy_db: Path):
        # Constructing PrismTasks against the legacy DB must migrate.
        PrismTasks(db_path=str(legacy_db))
        with sqlite3.connect(legacy_db) as c:
            cols = [r[1] for r in c.execute("PRAGMA table_info(tasks)")]
        assert cols == [
            "id", "title", "notes", "due_date", "priority",
            "done", "project", "tags_json", "source", "url",
        ]

    def test_legacy_data_preserved_under_backup_name(self, legacy_db: Path):
        PrismTasks(db_path=str(legacy_db))
        with sqlite3.connect(legacy_db) as c:
            rows = c.execute(
                "SELECT id FROM tasks_legacy_task_queue"
            ).fetchall()
        assert rows == [("legacy-1",)], "legacy data should be preserved"


class TestAddAndListAfterMigration:
    """Headline: add and list both work end-to-end after migration."""

    def test_add_succeeds(self, legacy_db: Path):
        tasks = PrismTasks(db_path=str(legacy_db))
        task = tasks.add("buy milk")
        assert task.task_id
        assert task.title == "buy milk"

    def test_list_returns_added_task(self, legacy_db: Path):
        tasks = PrismTasks(db_path=str(legacy_db))
        tasks.add("buy milk")
        tasks.add("walk the dog")
        items = tasks.list_tasks()
        titles = sorted(t.title for t in items)
        assert titles == ["buy milk", "walk the dog"]


class TestFreshDbStillWorks:
    """No legacy table → no migration → add/list still work."""

    def test_fresh_db_no_backup_table(self, tmp_path: Path):
        db = tmp_path / "fresh.db"
        tasks = PrismTasks(db_path=str(db))
        tasks.add("first task")
        with sqlite3.connect(db) as c:
            backup = c.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='tasks_legacy_task_queue'"
            ).fetchone()
        assert backup is None, "fresh DB must not create the backup table"
        assert tasks.list_tasks()[0].title == "first task"

    def test_existing_correct_schema_not_renamed(self, tmp_path: Path):
        db = tmp_path / "current.db"
        # Seed with the *current* schema, not the legacy one.
        PrismTasks(db_path=str(db)).add("seeded")
        # Re-open — must not re-migrate or wipe the table.
        tasks2 = PrismTasks(db_path=str(db))
        assert [t.title for t in tasks2.list_tasks()] == ["seeded"]
        with sqlite3.connect(db) as c:
            backup = c.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='tasks_legacy_task_queue'"
            ).fetchone()
        assert backup is None
