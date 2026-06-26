"""Empty-title task bug for issue #28 bug 59.

Live probe ("list my tasks") returned::

    Tasks (4) · local
    · buy milk
    · Test Notification
    · do laundry
    ·

That trailing bullet is a real task row with ``title=''`` sitting in
``~/.prism/tasks.db``.  Two paths produced it:

* ``handle_pa_intent`` add_task branch did
  ``parsed.get("title", message[:80])`` — but ``.get()`` only falls back
  on a *missing* key, not on an LLM that returned ``{"title": ""}``.
* ``PrismTasks.add`` happily inserted an empty title with no guard.

Surgical fix:

1. ``PrismTasks.add`` strips the title and coerces empty → ``"Untitled
   task"`` so the store can never hold a blank row again.
2. ``handle_pa_intent`` falls back to ``message[:80]`` whenever the
   parsed ``title`` is empty/whitespace.
3. ``list_tasks`` handler skips any blank-title rows defensively, so
   the existing bad row in the user's DB stops rendering immediately
   without a destructive migration.
"""
from __future__ import annotations

import tempfile

from prism_pa_intents import handle_pa_intent
from prism_tasks import PrismTasks, Task

# ---------------------------------------------------------------------
# 1. PrismTasks.add never persists an empty title.
# ---------------------------------------------------------------------

class TestAddRejectsEmptyTitle:

    def _mgr(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        return PrismTasks(db_path=tmp.name)

    def test_empty_string_becomes_untitled(self):
        mgr = self._mgr()
        task = mgr.add(title="")
        assert task.title == "Untitled task"

    def test_whitespace_only_becomes_untitled(self):
        mgr = self._mgr()
        task = mgr.add(title="   \n  ")
        assert task.title == "Untitled task"

    def test_normal_title_unchanged(self):
        mgr = self._mgr()
        task = mgr.add(title="buy milk")
        assert task.title == "buy milk"

    def test_title_stripped(self):
        mgr = self._mgr()
        task = mgr.add(title="  buy milk  ")
        assert task.title == "buy milk"


# ---------------------------------------------------------------------
# 2. add_task handler falls back when LLM returns ``{"title": ""}``.
# ---------------------------------------------------------------------

class _StubRouter:
    def __init__(self, raw):
        self._raw = raw

    def call(self, *_a, **_kw):
        return self._raw, None


class _StubAgent:
    def __init__(self, raw, mgr):
        self._router = _StubRouter(raw)
        self._task_mgr = mgr


class TestAddTaskHandlerFallback:

    def _mgr(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        return PrismTasks(db_path=tmp.name)

    def test_empty_llm_title_falls_back_to_message(self):
        mgr = self._mgr()
        agent = _StubAgent(
            raw='{"title": "", "notes": "", "due_date": "", "priority": 1}',
            mgr=mgr,
        )
        card = handle_pa_intent(agent, "add_task", "remember to buy bread", {})
        assert card is not None
        assert "remember to buy bread" in card.body
        # Confirm DB row matches.
        stored = mgr.list_tasks(done=False)
        assert any(t.title.strip() == "remember to buy bread" for t in stored)

    def test_whitespace_llm_title_falls_back(self):
        mgr = self._mgr()
        agent = _StubAgent(
            raw='{"title": "   ", "notes": "", "due_date": ""}',
            mgr=mgr,
        )
        card = handle_pa_intent(agent, "add_task", "wash the car", {})
        assert card is not None
        assert "wash the car" in card.body


# ---------------------------------------------------------------------
# 3. list_tasks handler skips legacy blank rows.
# ---------------------------------------------------------------------

class TestListTasksSkipsBlank:

    def _mgr_with_blank(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        mgr = PrismTasks(db_path=tmp.name)
        # Force a legacy blank row past the guard.
        mgr._store(Task(task_id="legacy01", title=""))
        mgr.add(title="real task")
        return mgr

    def test_blank_row_does_not_render(self):
        mgr = self._mgr_with_blank()
        # Sanity: the row is still in the DB.
        raw = mgr.list_tasks(done=False)
        titles = [t.title for t in raw]
        assert "" in titles
        # But the handler filters it out.

        class _A:
            pass
        agent = _A()
        agent._task_mgr = mgr
        card = handle_pa_intent(agent, "list_tasks", "list my tasks", {})
        assert card is not None
        assert "real task" in card.body
        # No empty bullet line.
        for line in card.body.splitlines():
            stripped = line.strip()
            assert stripped not in {"·", "⚡"}
            assert not stripped.endswith("· ")

    def test_count_in_title_reflects_filtered_total(self):
        mgr = self._mgr_with_blank()

        class _A:
            pass
        agent = _A()
        agent._task_mgr = mgr
        card = handle_pa_intent(agent, "list_tasks", "list my tasks", {})
        # Only 1 visible task, not 2.
        assert "(1)" in card.title
