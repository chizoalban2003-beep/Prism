from __future__ import annotations
import importlib.util
import logging
from dataclasses import dataclass
from typing import Optional
from prism_device_scanner import CapabilityMap

logger = logging.getLogger(__name__)


@dataclass
class ToolResolution:
    resolved:         bool
    method:           str    # stdlib|installed_cli|py_package|synthesised|install_suggested|manual
    implementation:   str
    requires_install: bool = False
    install_command:  str  = ""
    user_message:     str  = ""


class ToolResolver:
    """
    When a tool is not installed, finds the next best option.

    Resolution chain (in order):
      1. Python stdlib  — always available, no install
      2. Installed CLI  — checked via CapabilityMap
      3. LLM synthesis  — Claude/Ollama writes pure-Python implementation
      4. pip install    — suggests package install (requires user approval)
      5. System install — suggests brew/apt (requires user approval)
      6. Manual steps   — explains what the user needs to do
    """

    STDLIB_TASKS: dict[str, str] = {
        "zip":    "import zipfile",
        "csv":    "import csv",
        "json":   "import json",
        "http":   "import urllib.request",
        "hash":   "import hashlib",
        "base64": "import base64",
        "email":  "import smtplib, email",
        "sqlite": "import sqlite3",
        "xml":    "import xml.etree.ElementTree",
    }

    PIP_PACKAGES: dict[str, tuple[str, str]] = {
        "image_resize":  ("Pillow",     "pip install Pillow"),
        "pdf_read":      ("pdfplumber", "pip install pdfplumber"),
        "pdf_write":     ("reportlab",  "pip install reportlab"),
        "excel":         ("openpyxl",   "pip install openpyxl"),
        "browser_auto":  ("playwright", "pip install playwright"),
        "audio":         ("pydub",      "pip install pydub"),
        "http_async":    ("aiohttp",    "pip install aiohttp"),
        "data_analysis": ("pandas",     "pip install pandas"),
    }

    SYSTEM_PACKAGES: dict[str, tuple[str, str]] = {
        "image_convert": ("ImageMagick", "brew install imagemagick"),
        "video_convert": ("ffmpeg",      "brew install ffmpeg"),
        "audio_convert": ("ffmpeg",      "brew install ffmpeg"),
        "pdf_convert":   ("wkhtmltopdf", "brew install wkhtmltopdf"),
        "doc_convert":   ("pandoc",      "brew install pandoc"),
        "git":           ("git",         "brew install git"),
    }

    # Maps pip package name → actual importable module name
    _IMPORT_NAMES: dict[str, str] = {
        "Pillow":     "PIL",
        "pdfplumber": "pdfplumber",
        "reportlab":  "reportlab",
        "openpyxl":   "openpyxl",
        "playwright": "playwright",
        "pydub":      "pydub",
        "aiohttp":    "aiohttp",
        "pandas":     "pandas",
    }

    def __init__(self, collaborator=None):
        self.collaborator = collaborator

    def resolve(
        self,
        task_type:   str,
        description: str,
        caps:        CapabilityMap,
    ) -> ToolResolution:

        # 1. stdlib
        for key, lib in self.STDLIB_TASKS.items():
            if key in task_type:
                lib_parts = lib.split()
                lib_name = lib_parts[1] if len(lib_parts) > 1 else lib
                return ToolResolution(
                    True, "stdlib", lib,
                    user_message=f"Using Python built-in ({lib_name}).")

        # 2. pip package already installed
        pip_entry = self.PIP_PACKAGES.get(task_type)
        if pip_entry:
            pkg, cmd = pip_entry
            import_name = self._IMPORT_NAMES.get(pkg, pkg.lower().replace("-", "_"))
            if importlib.util.find_spec(import_name) is not None:
                return ToolResolution(
                    True, "py_package", f"import {import_name}",
                    user_message=f"Using {pkg} (already installed).")

        # 3. LLM synthesis
        if self.collaborator:
            code = self._synthesise(description, caps)
            if code:
                return ToolResolution(
                    True, "synthesised", code,
                    user_message="Generated a Python solution for this task.")

        # 4. Suggest pip install
        if pip_entry:
            pkg, cmd = pip_entry
            return ToolResolution(
                False, "install_suggested", "",
                requires_install=True, install_command=cmd,
                user_message=(
                    f"This task needs {pkg}. "
                    f"I can install it with `{cmd}`. Allow?"))

        # 5. Suggest system install
        sys_entry = self.SYSTEM_PACKAGES.get(task_type)
        if sys_entry:
            pkg, cmd = sys_entry
            return ToolResolution(
                False, "install_suggested", "",
                requires_install=True, install_command=cmd,
                user_message=(
                    f"This task needs {pkg}. "
                    f"I can install it with `{cmd}`. Allow?"))

        # 6. Manual fallback
        return ToolResolution(
            False, "manual", "",
            user_message=(
                f"I can't do '{description}' automatically — "
                f"no suitable tool found. "
                f"Here are the manual steps: ..."))

    def _synthesise(self, description: str, caps: CapabilityMap) -> Optional[str]:
        if not self.collaborator:
            return None
        prompt = (
            f"Write a Python function using ONLY Python stdlib to: {description}. "
            f"Available CLIs: {caps.summary()}. "
            f"Return ONLY the function code, no explanation."
        )
        try:
            code = self.collaborator._call_llm(prompt)
            return code if code and "def " in code else None
        except Exception:
            return None
