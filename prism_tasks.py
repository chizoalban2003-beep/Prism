from __future__ import annotations

import json
import logging
import sqlite3
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

@dataclass
class Task:
    task_id:   str
    title:     str
    notes:     str = ""
    due_date:  str = ""
    priority:  int = 1
    done:      bool = False
    project:   str = ""
    tags:      list[str] = field(default_factory=list)
    source:    str = "local"
    url:       str = ""

class PrismTasks:
    """
    Task management integration.

    Config:
      [tasks]
      provider       = "auto"
      todoist_token  = ""
      github_token   = ""
      github_repo    = ""
      linear_api_key = ""
    """

    def __init__(self, db_path="~/.prism/tasks.db",
                  todoist_token="", github_token="",
                  github_repo="", linear_key="",
                  provider="auto"):
        self._db       = Path(db_path).expanduser()
        self._db.parent.mkdir(parents=True, exist_ok=True)
        self._todoist  = todoist_token
        self._github   = github_token
        self._repo     = github_repo
        self._linear   = linear_key
        self._provider = provider
        self._init_db()

    @classmethod
    def from_config(cls, config: dict) -> PrismTasks:
        t = config.get("tasks", {})
        return cls(
            todoist_token = t.get("todoist_token",""),
            github_token  = t.get("github_token",""),
            github_repo   = t.get("github_repo",""),
            linear_key    = t.get("linear_api_key",""),
            provider      = t.get("provider","auto"),
        )

    def add(self, title: str, notes="", due_date="",
             priority=1, project="", tags=None) -> Task:
        import uuid
        task = Task(str(uuid.uuid4())[:8], title, notes,
                    due_date, priority, False, project,
                    tags or [], "local")
        provider = self._resolve_provider()
        if provider == "todoist":
            todoist_id = self._todoist_add(task)
            if todoist_id:
                task.task_id = todoist_id
                task.source  = "todoist"
        elif provider == "github":
            gh_num = self._github_add(task)
            if gh_num:
                task.task_id = str(gh_num)
                task.source  = "github"
                task.url     = (f"https://github.com/{self._repo}"
                                f"/issues/{gh_num}")
        elif provider == "linear":
            linear_id = self._linear_add(task)
            if linear_id:
                task.task_id = linear_id
                task.source  = "linear"
        self._store(task)
        return task

    def list_tasks(self, done=False, project="") -> list[Task]:
        provider = self._resolve_provider()
        if provider == "todoist" and not done:
            remote = self._todoist_list()
            if remote:
                return remote
        if provider == "github" and not done:
            remote = self._github_list()
            if remote:
                return remote
        if provider == "linear" and not done:
            remote = self._linear_list()
            if remote:
                return remote
        with sqlite3.connect(self._db, timeout=30.0) as c:
            if project:
                rows = c.execute(
                    "SELECT * FROM tasks WHERE done=? AND project=? "
                    "ORDER BY priority DESC",
                    (int(done), project)).fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM tasks WHERE done=? ORDER BY priority DESC",
                    (int(done),)).fetchall()
        return [self._row_to_task(r) for r in rows]

    def complete(self, task_id: str) -> bool:
        provider = self._resolve_provider()
        if provider == "todoist":
            self._todoist_complete(task_id)
        elif provider == "github":
            self._github_close(task_id)
        elif provider == "linear":
            self._linear_complete(task_id)
        with sqlite3.connect(self._db, timeout=30.0) as c:
            c.execute("UPDATE tasks SET done=1 WHERE id=?", (task_id,))
        return True

    def search(self, query: str) -> list[Task]:
        q = f"%{query.lower()}%"
        with sqlite3.connect(self._db, timeout=30.0) as c:
            rows = c.execute(
                "SELECT * FROM tasks WHERE done=0 AND "
                "(lower(title) LIKE ? OR lower(notes) LIKE ? "
                "OR lower(project) LIKE ?)",
                (q, q, q)).fetchall()
        return [self._row_to_task(r) for r in rows]

    def _todoist_add(self, task: Task) -> Optional[str]:
        payload = json.dumps({
            "content":  task.title,
            "description": task.notes,
            "priority": task.priority,
            "due_string": task.due_date or None,
        }).encode()
        req = urllib.request.Request(
            "https://api.todoist.com/rest/v2/tasks",
            data=payload,
            headers={"Authorization": f"Bearer {self._todoist}",
                     "Content-Type": "application/json"})
        try:
            resp = urllib.request.urlopen(req, timeout=8)
            return json.loads(resp.read()).get("id","")
        except Exception as e:
            logger.debug("Todoist add failed: %s", e)
            return None

    def _todoist_list(self) -> list[Task]:
        req = urllib.request.Request(
            "https://api.todoist.com/rest/v2/tasks",
            headers={"Authorization": f"Bearer {self._todoist}"})
        try:
            resp = urllib.request.urlopen(req, timeout=8)
            data = json.loads(resp.read())
            return [Task(
                task_id  = str(t["id"]),
                title    = t["content"],
                notes    = t.get("description",""),
                due_date = t.get("due",{}).get("date","") if t.get("due") else "",
                priority = t.get("priority",1),
                project  = t.get("project_id",""),
                source   = "todoist",
                url      = t.get("url",""),
            ) for t in data]
        except Exception as e:
            logger.debug("Todoist list failed: %s", e)
            return []

    def _todoist_complete(self, task_id: str) -> None:
        req = urllib.request.Request(
            f"https://api.todoist.com/rest/v2/tasks/{task_id}/close",
            data=b"",
            headers={"Authorization": f"Bearer {self._todoist}"},
            method="POST")
        try:
            urllib.request.urlopen(req, timeout=8)
        except Exception:
            pass

    def _github_add(self, task: Task) -> Optional[int]:
        payload = json.dumps({
            "title": task.title, "body": task.notes,
            "labels": ["prism"] + task.tags,
        }).encode()
        req = urllib.request.Request(
            f"https://api.github.com/repos/{self._repo}/issues",
            data=payload,
            headers={"Authorization": f"Bearer {self._github}",
                     "Content-Type": "application/json",
                     "Accept": "application/vnd.github.v3+json"})
        try:
            resp = urllib.request.urlopen(req, timeout=8)
            return json.loads(resp.read()).get("number")
        except Exception as e:
            logger.debug("GitHub issue create failed: %s", e)
            return None

    def _github_list(self) -> list[Task]:
        url = (f"https://api.github.com/repos/{self._repo}/issues"
               f"?state=open&labels=prism&per_page=30")
        req = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {self._github}",
            "Accept": "application/vnd.github.v3+json"})
        try:
            resp = urllib.request.urlopen(req, timeout=8)
            return [Task(
                task_id = str(i["number"]),
                title   = i["title"],
                notes   = i.get("body",""),
                source  = "github",
                url     = i.get("html_url",""),
            ) for i in json.loads(resp.read())]
        except Exception as e:
            logger.debug("GitHub list failed: %s", e)
            return []

    def _github_close(self, task_id: str) -> None:
        payload = json.dumps({"state":"closed"}).encode()
        req = urllib.request.Request(
            f"https://api.github.com/repos/{self._repo}/issues/{task_id}",
            data=payload,
            headers={"Authorization": f"Bearer {self._github}",
                     "Content-Type":"application/json",
                     "Accept":"application/vnd.github.v3+json"},
            method="PATCH")
        try:
            urllib.request.urlopen(req, timeout=8)
        except Exception:
            pass

    def _linear_add(self, task: Task) -> Optional[str]:
        """Create a Linear issue via GraphQL API."""
        if not self._linear:
            return None
        query = """
        mutation CreateIssue($title: String!, $description: String) {
          issueCreate(input: {title: $title, description: $description}) {
            issue { id identifier url }
          }
        }
        """
        payload = json.dumps({
            "query": query,
            "variables": {"title": task.title, "description": task.notes or ""}
        }).encode()
        req = urllib.request.Request(
            "https://api.linear.app/graphql",
            data=payload,
            headers={"Authorization": self._linear,
                     "Content-Type": "application/json"})
        try:
            resp = urllib.request.urlopen(req, timeout=8)
            data = json.loads(resp.read())
            issue = data.get("data",{}).get("issueCreate",{}).get("issue",{})
            if issue.get("id"):
                task.url = issue.get("url","")
                return issue["id"]
        except Exception as e:
            logger.debug("Linear add failed: %s", e)
        return None

    def _linear_list(self) -> list[Task]:
        query = """
        query { issues(filter: {state: {type: {nin: ["completed","cancelled"]}}},
                       first: 30) {
          nodes { id title description url state { name } dueDate priority }
        }}
        """
        payload = json.dumps({"query": query}).encode()
        req = urllib.request.Request(
            "https://api.linear.app/graphql",
            data=payload,
            headers={"Authorization": self._linear,
                     "Content-Type": "application/json"})
        try:
            resp  = urllib.request.urlopen(req, timeout=8)
            nodes = json.loads(resp.read()).get("data",{}).get("issues",{}).get("nodes",[])
            return [Task(
                task_id  = n["id"],
                title    = n["title"],
                notes    = n.get("description","") or "",
                due_date = n.get("dueDate","") or "",
                priority = n.get("priority",1),
                source   = "linear",
                url      = n.get("url",""),
            ) for n in nodes]
        except Exception as e:
            logger.debug("Linear list failed: %s", e)
            return []

    def _linear_complete(self, task_id: str) -> None:
        cancel_query = f"""
        mutation {{ issueUpdate(id: "{task_id}", input: {{stateType: "completed"}}) {{ success }} }}
        """
        payload = json.dumps({"query": cancel_query}).encode()
        req = urllib.request.Request(
            "https://api.linear.app/graphql",
            data=payload,
            headers={"Authorization": self._linear,
                     "Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=8)
        except Exception:
            pass

    def _resolve_provider(self) -> str:
        if self._provider != "auto":
            return self._provider
        if self._todoist:
            return "todoist"
        if self._github and self._repo:
            return "github"
        if self._linear:
            return "linear"
        return "local"

    def _store(self, task: Task) -> None:
        with sqlite3.connect(self._db, timeout=30.0) as c:
            c.execute("INSERT OR REPLACE INTO tasks VALUES(?,?,?,?,?,?,?,?,?,?)",
                      (task.task_id, task.title, task.notes, task.due_date,
                       task.priority, int(task.done), task.project,
                       json.dumps(task.tags), task.source, task.url))

    def _row_to_task(self, row) -> Task:
        return Task(row[0],row[1],row[2],row[3],row[4],
                    bool(row[5]),row[6],
                    json.loads(row[7]),row[8],row[9])

    def _init_db(self) -> None:
        with sqlite3.connect(self._db, timeout=30.0) as c:
            c.execute("""CREATE TABLE IF NOT EXISTS tasks(
                id TEXT PRIMARY KEY, title TEXT, notes TEXT,
                due_date TEXT, priority INTEGER, done INTEGER,
                project TEXT, tags_json TEXT, source TEXT, url TEXT)""")
