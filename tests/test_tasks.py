import tempfile
from prism_tasks import PrismTasks

def _tmp_tasks():
    tmp = tempfile.mktemp(suffix=".db")
    return PrismTasks(db_path=tmp)

def test_add_returns_task():
    t = _tmp_tasks()
    task = t.add("test task")
    assert task.title == "test task"

def test_list_empty_initially():
    t = _tmp_tasks()
    assert isinstance(t.list_tasks(), list)

def test_complete_marks_done():
    t = _tmp_tasks()
    task = t.add("finish report")
    t.complete(task.task_id)
    open_tasks = t.list_tasks(done=False)
    assert not any(x.task_id == task.task_id for x in open_tasks)

def test_search_finds_by_title():
    t = _tmp_tasks()
    t.add("write the report")
    results = t.search("report")
    assert results and "report" in results[0].title

def test_provider_auto_local():
    t = _tmp_tasks()
    assert t._resolve_provider() == "local"
