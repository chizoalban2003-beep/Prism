"""Tests for prism_tasks.py — Gap Prompt 14b-ii."""
import pytest

from prism_tasks import Task, PrismTasks


@pytest.fixture
def tasks(tmp_path):
    return PrismTasks(db_path=str(tmp_path / "tasks.db"))


def test_add_returns_task(tasks):
    """add('test task') returns a Task with the correct title."""
    t = tasks.add("test task")
    assert isinstance(t, Task)
    assert t.title == "test task"


def test_list_empty_initially(tasks):
    """list_tasks() returns a list (may be empty) without raising."""
    result = tasks.list_tasks()
    assert isinstance(result, list)


def test_complete_marks_done(tasks):
    """add then complete → task no longer appears in list_tasks()."""
    t = tasks.add("finish report")
    tasks.complete(t.task_id)
    open_tasks = tasks.list_tasks(done=False)
    assert not any(x.task_id == t.task_id for x in open_tasks)


def test_search_finds_by_title(tasks):
    """add 'write report' → search('report') finds it."""
    tasks.add("write report")
    results = tasks.search("report")
    assert any(t.title == "write report" for t in results)


def test_provider_auto_local(tasks):
    """No tokens configured → _resolve_provider() returns 'local'."""
    assert tasks._resolve_provider() == "local"
