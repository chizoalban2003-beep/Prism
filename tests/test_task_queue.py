from __future__ import annotations

import time

from prism_task_queue import TaskProgress, TaskQueue, TaskStatus


def test_submit_returns_string(tmp_path):
    q = TaskQueue(db_path=str(tmp_path / "tasks.db"))
    task_id = q.submit_single("test", lambda: 1)
    assert isinstance(task_id, str)
    assert len(task_id) == 8


def test_get_after_submit(tmp_path):
    q = TaskQueue(db_path=str(tmp_path / "tasks.db"))
    task_id = q.submit_single("fetch me", lambda: 42)
    deadline = time.time() + 0.1
    progress = None
    while time.time() < deadline:
        progress = q.get(task_id)
        if progress is not None:
            break
        time.sleep(0.005)
    assert progress is not None
    assert isinstance(progress, TaskProgress)


def test_completes_fast_fn(tmp_path):
    q = TaskQueue(db_path=str(tmp_path / "tasks.db"))
    task_id = q.submit_single("quick", lambda: 1)
    deadline = time.time() + 3
    while time.time() < deadline:
        p = q.get(task_id)
        if p and p.status == TaskStatus.COMPLETED:
            break
        time.sleep(0.05)
    p = q.get(task_id)
    assert p is not None
    assert p.status == TaskStatus.COMPLETED


def test_cancel_returns_bool(tmp_path):
    q = TaskQueue(db_path=str(tmp_path / "tasks.db"))
    task_id = q.submit_single("slow", lambda: time.sleep(10))
    result = q.cancel(task_id)
    assert isinstance(result, bool)
