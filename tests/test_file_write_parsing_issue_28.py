"""Path-inference fix for issue #28 bug 5 — file_write missed natural phrasings.

Live test: PRISM was asked things like

  * "create a file at ~/Documents/notes.txt with the content: Hello"
  * "write the file ~/Documents/hello.txt with Hello World"
  * "make a file called hello.txt with Hello inside it"
  * "create ~/Documents/foo.md and put 'bar' in it"
  * "write 'hello' to file ~/Documents/hello.txt"

Every one of these either dropped the tilde (writing to ``/Documents/...``,
which fails the allow-list), captured the literal word ``"file"`` as the
path, or treated noise like ``"a file"`` as the content body. The
underlying ``_parse_message`` was three regexes that all assumed
content-before-path with an absolute leading slash.

The rewrite locates the path token first (handling ``~/``, ``/``, ``./``,
quoted, and ``file NAME.ext`` forms), splits the message around it, and
runs a small ladder of content patterns either side. These tests pin
the new behaviour so the next refactor can't silently regress it.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "_test_file_write_organ",
    Path(__file__).resolve().parent.parent / "organs" / "file_write.py",
)
_organ = importlib.util.module_from_spec(_SPEC)  # type: ignore[arg-type]
_SPEC.loader.exec_module(_organ)  # type: ignore[union-attr]

_parse = _organ._parse_message


class TestPathDetection:
    """Path must be located correctly regardless of where it sits in the
    sentence and which scheme (tilde / absolute / bare-with-keyword) is
    used."""

    @pytest.mark.parametrize("message,expected", [
        ("write Hello World to ~/Documents/hello.txt",       "~/Documents/hello.txt"),
        ("save my notes to ~/Documents/notes.txt",           "~/Documents/notes.txt"),
        ("write 'Test content' to ~/Documents/test.txt",     "~/Documents/test.txt"),
        ("create a file at ~/Documents/notes.txt with the content: Hello",
                                                             "~/Documents/notes.txt"),
        ("write the file ~/Documents/hello.txt with Hello World",
                                                             "~/Documents/hello.txt"),
        ("save the following to ~/Documents/notes.txt:\nMy notes",
                                                             "~/Documents/notes.txt"),
        ("make a file called hello.txt with Hello inside it", "hello.txt"),
        ("create ~/Documents/foo.md and put 'bar' in it",    "~/Documents/foo.md"),
        ("write 'hello' to file ~/Documents/hello.txt",      "~/Documents/hello.txt"),
        # Absolute-path inputs still work.
        ("write hello to /tmp/prism/x.txt",                  "/tmp/prism/x.txt"),
    ])
    def test_path_extracted(self, message, expected):
        path, _ = _parse(message)
        assert path == expected, f"got {path!r} from {message!r}"


class TestContentDetection:
    """Content extraction handles 'with', 'with the content:', ':', 'and put X in it',
    and the canonical 'CONTENT to PATH' phrasing."""

    @pytest.mark.parametrize("message,expected", [
        ("write Hello World to ~/Documents/hello.txt",       "Hello World"),
        ("save my notes to ~/Documents/notes.txt",           "my notes"),
        ("write 'Test content' to ~/Documents/test.txt",     "Test content"),
        ("create a file at ~/Documents/notes.txt with the content: Hello",
                                                             "Hello"),
        ("write the file ~/Documents/hello.txt with Hello World",
                                                             "Hello World"),
        ("save the following to ~/Documents/notes.txt:\nMy notes",
                                                             "My notes"),
        ("make a file called hello.txt with Hello inside it", "Hello"),
        ("create ~/Documents/foo.md and put 'bar' in it",    "bar"),
        ("write 'hello' to file ~/Documents/hello.txt",      "hello"),
    ])
    def test_content_extracted(self, message, expected):
        _, content = _parse(message)
        assert content == expected, f"got {content!r} from {message!r}"


class TestPreviouslyBuggyCases:
    """Anti-regression: the exact symptoms that bug 5 documented."""

    def test_word_file_is_never_the_path(self):
        # Before the fix: path captured as "file" (whatever followed
        # the first "to" preposition).
        path, _ = _parse("write 'hello' to file ~/Documents/hello.txt")
        assert path != "file"

    def test_tilde_is_preserved(self):
        # Before the fix: path captured as "/Documents/hello.txt" (the
        # leading "~" was dropped), which then failed _is_path_allowed.
        path, _ = _parse("write the file ~/Documents/hello.txt with Hello World")
        assert path.startswith("~/"), f"tilde lost — got {path!r}"

    def test_a_file_is_never_the_content(self):
        # Before the fix: content captured as "a file" because the regex
        # greedily grabbed the first token after the verb.
        _, content = _parse(
            "create a file at ~/Documents/notes.txt with the content: Hello"
        )
        assert content == "Hello"
        assert "a file" not in content

    def test_make_a_file_called_phrasing_recognised(self):
        # Before the fix: returned (None, None) because no absolute path
        # was present.
        path, content = _parse("make a file called hello.txt with Hello inside it")
        assert path == "hello.txt"
        assert content == "Hello"


class TestGracefulFallback:
    def test_no_recognisable_path_returns_none(self):
        path, content = _parse("write something somewhere")
        assert path is None
        assert content is None
