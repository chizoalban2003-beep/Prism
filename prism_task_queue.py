from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

class TaskStatus(str, Enum):
    QUEUED    = "queued"
    RUNNING   = "running"
    PAUSED    = "paused"
    COMPLETED = "completed"
    FAILED    = "failed"
    CANCELLED = "cancelled"

@dataclass
class TaskProgress:
    task_id:     str
    status:      TaskStatus
    title:       str
    progress:    float         # 0.0 to 1.0
    current_step:str = ""
    result:      dict = field(default_factory=dict)
    error:       str  = ""
    started_at:  float = 0.0
    completed_at:float = 0.0
    steps_done:  int   = 0
    steps_total: int   = 0

@dataclass
class QueuedTask:
    task_id:   str
    title:     str
    fn:        Callable        # the function to run
    steps:     list[dict]      # [{title, fn, params}] for multi-step tasks
    on_update: Callable        # called with TaskProgress on each step

class TaskQueue:
    """
    Persistent background task queue.
    Long tasks run in daemon threads, reporting progress to SQLite.
    The UI polls GET /tasks/{id}/status for live updates.
    """

    def __init__(self, db_path: str = "~/.prism/task_queue.db"):
        # Own DB file: ~/.prism/tasks.db is the PA to-do store (prism_tasks),
        # which uses a different `tasks` schema. Sharing the file + table name
        # caused a "table tasks has N columns but M values" collision.
        self._db  = Path(db_path).expanduser()
        self._db.parent.mkdir(parents=True, exist_ok=True)
        self._threads: dict[str, threading.Thread] = {}
        self._cancel:  dict[str, threading.Event]  = {}
        self._init_db()

    def submit(
        self,
        title: str,
        steps: list[dict],      # [{"title":str, "fn":callable, "params":dict}]
        on_complete: Optional[Callable] = None,
    ) -> str:
        """Submit a multi-step task. Returns task_id immediately."""
        task_id = str(uuid.uuid4())[:8]
        cancel  = threading.Event()
        self._cancel[task_id] = cancel
        self._write(TaskProgress(
            task_id=task_id, status=TaskStatus.QUEUED,
            title=title, progress=0.0,
            steps_total=len(steps), started_at=time.time()))

        def _run():
            self._write(TaskProgress(task_id=task_id,
                status=TaskStatus.RUNNING,title=title,
                progress=0.0,steps_total=len(steps),started_at=time.time()))
            results = []
            for i, step in enumerate(steps):
                if cancel.is_set():
                    self._write(TaskProgress(task_id=task_id,
                        status=TaskStatus.CANCELLED,title=title,
                        progress=i/len(steps),steps_done=i,
                        steps_total=len(steps)))
                    return
                self._write(TaskProgress(task_id=task_id,
                    status=TaskStatus.RUNNING,title=title,
                    progress=i/len(steps),current_step=step.get("title",""),
                    steps_done=i,steps_total=len(steps),
                    started_at=time.time()))
                try:
                    fn     = step["fn"]
                    params = step.get("params",{})
                    result = fn(**params) if params else fn()
                    results.append({"step":step.get("title",""),
                                    "result":str(result)[:500],"ok":True})
                except Exception as e:
                    logger.warning("Step %d failed: %s", i, e)
                    results.append({"step":step.get("title",""),
                                    "error":str(e),"ok":False})
                    self._write(TaskProgress(task_id=task_id,
                        status=TaskStatus.FAILED,title=title,
                        progress=i/len(steps),error=str(e),
                        steps_done=i,steps_total=len(steps)))
                    return

            final = TaskProgress(task_id=task_id,
                status=TaskStatus.COMPLETED,title=title,
                progress=1.0,steps_done=len(steps),
                steps_total=len(steps),result={"steps":results},
                completed_at=time.time())
            self._write(final)
            if on_complete:
                on_complete(final)

        t = threading.Thread(target=_run, daemon=True, name=f"prism-{task_id}")
        self._threads[task_id] = t
        t.start()
        return task_id

    def submit_single(self, title: str, fn: Callable,
                       params: Optional[dict] = None) -> str:
        """Convenience: submit a single-step task."""
        return self.submit(title, [{"title":title,"fn":fn,"params":params or {}}])

    def cancel(self, task_id: str) -> bool:
        if task_id in self._cancel:
            self._cancel[task_id].set()
            return True
        return False

    def get(self, task_id: str) -> Optional[TaskProgress]:
        with sqlite3.connect(self._db, timeout=30.0) as c:
            row = c.execute(
                "SELECT data_json FROM tasks WHERE id=?", (task_id,)
            ).fetchone()
        if not row:
            return None
        d = json.loads(row[0])
        return TaskProgress(**{k:v for k,v in d.items()
                               if k in TaskProgress.__dataclass_fields__})

    def list_recent(self, n: int = 10) -> list[TaskProgress]:
        with sqlite3.connect(self._db, timeout=30.0) as c:
            # started_at is stored inside data_json, not a real column; rowid
            # auto-increments on INSERT/REPLACE so highest rowid = most recent.
            rows = c.execute(
                "SELECT data_json FROM tasks ORDER BY rowid DESC LIMIT ?",
                (n,)).fetchall()
        out = []
        for row in rows:
            try:
                d = json.loads(row[0])
                out.append(TaskProgress(**{k:v for k,v in d.items()
                    if k in TaskProgress.__dataclass_fields__}))
            except Exception:
                pass
        return out

    def _write(self, p: TaskProgress) -> None:
        serializable = {k for k in TaskProgress.__dataclass_fields__}
        data = {k: (v.value if isinstance(v, TaskStatus) else v)
                for k, v in p.__dict__.items() if k in serializable}
        with sqlite3.connect(self._db, timeout=30.0) as c:
            c.execute("INSERT OR REPLACE INTO tasks VALUES(?,?,?)",
                      (p.task_id, p.status.value if hasattr(p.status,'value')
                       else p.status, json.dumps(data)))

    def _init_db(self) -> None:
        with sqlite3.connect(self._db, timeout=30.0) as c:
            c.execute("CREATE TABLE IF NOT EXISTS tasks("
                      "id TEXT PRIMARY KEY, status TEXT, data_json TEXT)")
            self._migrate(c)

    def _migrate(self, c) -> None:
        ver = c.execute("PRAGMA user_version").fetchone()[0]
        if ver < 1:
            c.execute("PRAGMA user_version = 1")
