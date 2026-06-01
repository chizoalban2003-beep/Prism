"""Installed-app and system capability scanner for the local device."""
from __future__ import annotations
import shutil
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

TOOL_CATALOGUE: dict[str, list[str]] = {
    "image":    ["convert", "ffmpeg", "sips", "magick"],
    "video":    ["ffmpeg", "avconv", "HandBrakeCLI"],
    "audio":    ["ffmpeg", "sox", "lame"],
    "pdf":      ["gs", "pdftk", "wkhtmltopdf", "mutool"],
    "document": ["pandoc", "libreoffice", "soffice"],
    "code":     ["git", "docker", "python3", "node", "npm", "pip3"],
    "network":  ["curl", "wget", "jq"],
    "compress": ["zip", "tar", "gzip", "bzip2"],
    "browser":  ["chromium", "google-chrome", "firefox"],
    "system":   ["df", "du", "ps"],
}

PYTHON_PACKAGES: list[str] = [
    "PIL", "cv2", "requests", "playwright", "selenium",
    "aiohttp", "pandas", "openpyxl", "pdfplumber", "reportlab",
    "pydub", "paramiko",
]


@dataclass
class CapabilityMap:
    cli_tools:   dict[str, list[str]]
    py_packages: list[str]
    platform:    str
    has_browser: bool
    scanned_at:  float = field(default_factory=time.time)

    def can_do(self, category: str) -> bool:
        return bool(self.cli_tools.get(category))

    def best_tool(self, category: str) -> Optional[str]:
        tools = self.cli_tools.get(category, [])
        return tools[0] if tools else None

    def summary(self) -> str:
        cats  = [c for c, t in self.cli_tools.items() if t]
        total = sum(len(t) for t in self.cli_tools.values())
        return f"{total} tools in {len(cats)} categories: {', '.join(cats)}"


class DeviceCapabilityScanner:
    """Scans installed tools on the device. Completes in under 100ms."""

    def scan(self) -> CapabilityMap:
        cli: dict[str, list[str]] = {}
        for category, tools in TOOL_CATALOGUE.items():
            found = [t for t in tools if shutil.which(t)]
            if found:
                cli[category] = found

        pkgs: list[str] = []
        for pkg in PYTHON_PACKAGES:
            try:
                __import__(pkg.split("-")[0].replace("-", "_"))
                pkgs.append(pkg)
            except ImportError:
                pass

        has_browser = (
            bool(shutil.which("chromium"))
            or bool(shutil.which("google-chrome"))
            or bool(shutil.which("firefox"))
            or "playwright" in pkgs
        )

        return CapabilityMap(
            cli_tools   = cli,
            py_packages = pkgs,
            platform    = sys.platform,
            has_browser = has_browser,
        )
