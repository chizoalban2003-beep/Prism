"""device_task substring-match fix for issue #28 bug 50.

Live test: "remove the never mind instruction" returned an "Approval
required" card with task = "remove the never mind instruction" — a
medium-risk wrapper around a no-op. The real intent was
``remove_instruction`` (the user wants to delete a standing rule),
but it never got a chance because device_task's regex contained the
bare keyword ``move`` with no word boundary, so it matched as a
substring of "re**move**".

Same class of bug for any file-op verb embedded in another word:
* ``"deleted"``  matched ``delete``
* ``"copied"``   matched ``copy``
* ``"renamed"``  matched ``rename``
* ``"executed"`` matched ``execute``

Fix: tighten the bare verbs to require word boundaries AND a following
object (file/folder/directory/the/a/my). This kills the substring
collisions without losing the genuine "move a file" / "delete the
folder" use cases that device_task was meant to catch.

Pin the user-facing surface (instruction-management) AND the verb-form
collisions so future tweaks to device_task's regex don't reintroduce
either bug.
"""
from __future__ import annotations

import re

from prism_intents import INTENTS


def _route(text: str) -> str:
    lowered = text.lower()
    for pattern, intent in INTENTS:
        if re.search(pattern, lowered):
            return intent
    return ""


class TestInstructionManagementReachable:
    """The reported surface: removing a stored instruction must work."""

    def test_remove_named_instruction(self):
        assert _route("remove the never mind instruction") == "remove_instruction"

    def test_forget_an_instruction(self):
        assert _route("forget the dark mode instruction") == "remove_instruction"

    def test_delete_a_rule(self):
        assert _route("delete the rule about uber") == "remove_instruction"


class TestSubstringCollisionsKilled:
    """The verbs must no longer match inside larger words."""

    def test_removed_does_not_hit_device_task(self):
        # 'remove' as a substring of 'removed'.
        assert _route("I removed the file") != "device_task"

    def test_remove_alone_does_not_hit_device_task(self):
        # Bare 'remove' without a file-op object should not claim
        # device_task. (It may fall through to LLM classifier; we only
        # assert it doesn't get pulled by device_task.)
        assert _route("remove that") != "device_task"


class TestDeviceTaskStillWorks:
    """Sanity: genuine file-op phrases still route to device_task."""

    def test_move_a_file(self):
        assert _route("move a file") == "device_task"

    def test_delete_the_folder(self):
        assert _route("delete the folder") == "device_task"

    def test_copy_my_documents(self):
        # "copy my X" — bare verb + my noun. The regex requires file/
        # folder/directory/the/a/my following the verb.
        assert _route("copy my photos to a new folder") == "device_task"

    def test_rename_the_file(self):
        assert _route("rename the file") == "device_task"

    def test_my_documents_still_routes(self):
        # The "\bmy (?:downloads|documents|...)\b" branch is unchanged.
        assert _route("show me my documents") == "device_task"

    def test_resize_image(self):
        # The "resize" / "convert file" branches are unchanged.
        assert _route("resize the image") == "device_task"
