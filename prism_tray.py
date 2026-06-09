"""
prism_tray.py
=============
Native system tray icon for PRISM. Provides a frameless browser window
(pywebview) toggled by tray icon click, and a context menu with quick actions.

Usage:
    python prism_tray.py           # starts tray (daemon must be running on 8742)
    python prism_tray.py --port N  # override port
"""
from __future__ import annotations
import argparse
import logging
import threading

logger = logging.getLogger(__name__)

PRISM_URL = "http://127.0.0.1:{port}"
ICON_COLOR = (99, 102, 241)  # indigo — matches UI palette


def _load_icon():
    """Generate a simple 64x64 indigo square icon (no file dependency)."""
    try:
        from PIL import Image
        img = Image.new("RGB", (64, 64), color=ICON_COLOR)
        return img
    except ImportError:
        return None


def _build_menu(window_ref: list, port: int):
    """Build pystray menu. window_ref is a mutable list holding the webview window."""
    import pystray

    def on_open(icon, item):
        if window_ref:
            try:
                window_ref[0].show()
            except Exception:
                pass
        else:
            _open_window(window_ref, port)

    def on_quit(icon, item):
        if window_ref:
            try:
                window_ref[0].destroy()
            except Exception:
                pass
        icon.stop()

    return pystray.Menu(
        pystray.MenuItem("Open PRISM", on_open, default=True),
        pystray.MenuItem("Quit", on_quit),
    )


def _open_window(window_ref: list, port: int):
    """Create and show a pywebview window in a background thread."""
    try:
        import webview
        url = PRISM_URL.format(port=port)
        window = webview.create_window(
            "PRISM",
            url,
            width=420,
            height=640,
            frameless=False,
            on_top=False,
        )
        window_ref.clear()
        window_ref.append(window)
        # webview.start() blocks — run in thread
        t = threading.Thread(target=webview.start, daemon=True)
        t.start()
    except ImportError:
        logger.warning("[tray] pywebview not installed — open http://127.0.0.1:%d manually", port)
    except Exception as exc:
        logger.warning("[tray] webview error: %s", exc)


def run_tray(port: int = 8742):
    """Start the system tray icon. Blocks until user quits."""
    try:
        import pystray
    except ImportError:
        logger.error("[tray] pystray not installed. Run: pip install pystray pywebview Pillow")
        return

    icon_image = _load_icon()
    if icon_image is None:
        logger.error("[tray] Pillow not installed — cannot render tray icon")
        return

    window_ref: list = []
    menu = _build_menu(window_ref, port)
    icon = pystray.Icon("prism", icon_image, "PRISM", menu)
    icon.run()


def main():
    parser = argparse.ArgumentParser(description="PRISM system tray")
    parser.add_argument("--port", type=int, default=8742)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run_tray(port=args.port)


if __name__ == "__main__":
    main()
