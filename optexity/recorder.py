"""
Lightweight GUI Action Recorder (client for remote post-processing).

Same capture behavior as ``opcloud.recording_processor.recorder``; after a **graceful** stop,
uploads the session as a zip to ``{SERVER_URL}/process_gui_recording_file`` instead of running
``process_gui_recording_file`` locally.

**Stopping:** ESC, **F12**, or **Ctrl+Shift+Q** end recording and trigger post-processing when
``SERVER_URL`` is set. In Docker or over ``docker exec``, the terminal often consumes ESC (you
see ``^[`` echoed) before pynput sees it—use **F12** or **Ctrl+Shift+Q** instead. Optional:
set ``RECORDER_POST_ON_CTRL_C=1`` to run post-processing after Ctrl+C as well.

Set ``SERVER_URL`` (e.g. ``http://localhost:8000``). The server must expose the FastAPI route
from ``opcloud.api.gui_recording``.

Dependencies: pip install pynput pillow httpx
Optional: python-dotenv (for ENV_PATH / .env)

Output structure:
  recordings/
    session_<timestamp>/
      events.jsonl
      frames/
        <YYYYMMDD_HHMMSS_microseconds>_<event_id>_before.png
      processed/            # written after successful API response
        automation.json
        ...
"""

from __future__ import annotations

import io
import json
import os
import sys
import threading
import time
import uuid
import zipfile
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from queue import Queue
from urllib.parse import urljoin

import httpx
import mss
from PIL import Image
from pynput import keyboard, mouse

from optexity.utils.settings import settings


def _macos_is_secure_event_input_enabled() -> bool | None:
    """
    True if macOS Secure Input is active (password fields, Terminal secure mode, etc.).

    When True, pynput will not see keystrokes globally; None if the check could not run.
    """
    if sys.platform != "darwin":
        return None
    try:
        import ctypes

        lib = ctypes.CDLL(
            "/System/Library/Frameworks/Carbon.framework/Frameworks/HIToolbox.framework/HIToolbox"
        )
        lib.IsSecureEventInputEnabled.argtypes = []
        lib.IsSecureEventInputEnabled.restype = ctypes.c_bool
        return bool(lib.IsSecureEventInputEnabled())
    except Exception:
        return None


def _zip_session_dir(session_dir: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in session_dir.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(session_dir))
    return buf.getvalue()


@dataclass
class Event:
    id: str
    timestamp: float
    event_type: (
        str  # click, double_click, right_click, scroll, keypress, key_combo, drag
    )
    x: int
    y: int
    button: str = ""
    key: str = ""
    scroll_dx: int = 0
    scroll_dy: int = 0
    drag_start_x: int = 0
    drag_start_y: int = 0
    has_before: bool = False
    has_after: bool = False  # unused; kept for JSON compatibility


class Recorder:
    def __init__(self, output_dir: str = "recordings"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path(output_dir) / f"session_{ts}"
        self.frames_dir = self.session_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.events_file = self.session_dir / "events.jsonl"

        self.event_queue: Queue = Queue()
        self.running = False
        self._frame_lock = threading.Lock()
        self._frame_history = deque(maxlen=120)  # (timestamp, PIL.Image)
        self._screenshot_interval_s = 0.5
        self.held_keys: set = set()
        self.mouse_pressed = False
        self.mouse_press_pos = (0, 0)
        self.last_click_time = 0.0
        self.double_click_threshold = 0.4
        self.last_mouse_x = 0
        self.last_mouse_y = 0

        self._mouse_listener = None
        self._key_listener = None
        self._stop_event = threading.Event()
        self._sct = mss.mss()

        self.key_buffer: list = []
        self.key_buffer_start_time = 0.0
        self.key_buffer_pos = (0, 0)
        self.key_flush_timer: threading.Timer = None
        self.key_flush_delay = 0.5

        # True after ESC / F12 / Ctrl+Shift+Q (or Ctrl+C if RECORDER_POST_ON_CTRL_C=1)
        self._stop_requested_by_esc = False
        self._warned_secure_input = False

    def _secure_input_poll_loop(self) -> None:
        while self.running:
            time.sleep(1.5)
            if not self.running:
                break
            active = _macos_is_secure_event_input_enabled()
            if active is True and not self._warned_secure_input:
                self._warned_secure_input = True
                print(
                    "\n[NOTE] macOS Secure Input is ON — keystrokes may not be recorded "
                    "(common in password fields). Clicks still work. "
                    "Tab out or use a non-secure field if you need typed text in events.jsonl.\n",
                    file=sys.stderr,
                )

    def _capture_screen(self) -> Image.Image:
        shot = self._sct.grab(self._sct.monitors[0])
        return Image.frombytes("RGB", shot.size, shot.rgb)

    def _is_escape_key(self, key) -> bool:
        try:
            if key == keyboard.Key.esc:
                return True
        except Exception:
            pass
        try:
            if hasattr(key, "char") and key.char == "\x1b":
                return True
        except Exception:
            pass
        try:
            ks = self._key_to_str(key).lower()
            return ks in ("esc", "escape")
        except Exception:
            return False

    def _request_graceful_stop(self) -> None:
        """Stop recording and allow post-processing (ESC / F12 / Ctrl+Shift+Q)."""
        if self._stop_event.is_set():
            return
        if self.key_flush_timer:
            self.key_flush_timer.cancel()
        self._flush_key_buffer()
        self._stop_requested_by_esc = True
        self.stop()

    _CTRL_KEYS = frozenset({"ctrl", "ctrl_l", "ctrl_r"})
    _SHIFT_KEYS = frozenset({"shift", "shift_l", "shift_r"})

    def _is_ctrl_shift_q_press(self, key) -> bool:
        """Docker-friendly stop: terminal often eats ESC; this combo usually reaches pynput."""
        key_str = self._key_to_str(key).lower()
        if key_str != "q":
            return False
        mods = self.held_keys & self._MODIFIER_NAMES
        return bool(mods & self._CTRL_KEYS) and bool(mods & self._SHIFT_KEYS)

    def _save_screenshot(self, img, event_id: str, suffix: str, action_ts: float):
        ts_part = datetime.fromtimestamp(action_ts).strftime("%Y%m%d_%H%M%S_%f")
        path = self.frames_dir / f"{ts_part}_{event_id}_{suffix}.png"
        img.save(str(path), optimize=True)

    def _get_frame_before_action(self, action_time: float):
        with self._frame_lock:
            if not self._frame_history:
                return None
            best_img = None
            best_ts = None
            for ts, img in self._frame_history:
                if ts < action_time and (best_ts is None or ts > best_ts):
                    best_ts = ts
                    best_img = img
            if best_img is not None:
                return best_img.copy()
            return self._frame_history[0][1].copy()

    def _screenshot_loop(self):
        while self.running:
            try:
                img = self._capture_screen()
                ts = time.time()
                with self._frame_lock:
                    self._frame_history.append((ts, img))
            except Exception:
                pass
            time.sleep(self._screenshot_interval_s)

    def _process_events(self):
        with open(self.events_file, "a") as f:
            while True:
                try:
                    item = self.event_queue.get(timeout=1.0)
                except Exception:
                    continue

                if item is None:
                    break

                event = item
                before_img = self._get_frame_before_action(event.timestamp)
                if before_img is not None:
                    self._save_screenshot(
                        before_img, event.id, "before", event.timestamp
                    )
                    event.has_before = True

                f.write(json.dumps(asdict(event)) + "\n")
                f.flush()

    def _emit(self, event: Event):
        self.event_queue.put(event)

    def _on_click(self, x, y, button, pressed):
        if pressed:
            self.mouse_pressed = True
            self.mouse_press_pos = (x, y)
            return

        print(f"Click: {x}, {y}")

        self.mouse_pressed = False
        self._flush_typing_before_pointer_action()
        dx = abs(x - self.mouse_press_pos[0])
        dy = abs(y - self.mouse_press_pos[1])

        if dx > 10 or dy > 10:
            evt = Event(
                id=uuid.uuid4().hex[:10],
                timestamp=time.time(),
                event_type="drag",
                x=x,
                y=y,
                button=button.name,
                drag_start_x=self.mouse_press_pos[0],
                drag_start_y=self.mouse_press_pos[1],
            )
            self._emit(evt)
            return

        now = time.time()
        if (
            button == mouse.Button.left
            and (now - self.last_click_time) < self.double_click_threshold
        ):
            evt = Event(
                id=uuid.uuid4().hex[:10],
                timestamp=now,
                event_type="double_click",
                x=x,
                y=y,
                button=button.name,
            )
            self._emit(evt)
            self.last_click_time = 0
            return

        self.last_click_time = now

        event_type = "click" if button == mouse.Button.left else "right_click"
        evt = Event(
            id=uuid.uuid4().hex[:10],
            timestamp=now,
            event_type=event_type,
            x=x,
            y=y,
            button=button.name,
        )
        self._emit(evt)

    def _on_scroll(self, x, y, dx, dy):
        self._flush_typing_before_pointer_action()
        evt = Event(
            id=uuid.uuid4().hex[:10],
            timestamp=time.time(),
            event_type="scroll",
            x=x,
            y=y,
            scroll_dx=dx,
            scroll_dy=dy,
        )
        self._emit(evt)

    def _on_move(self, x, y):
        self.last_mouse_x = x
        self.last_mouse_y = y

    def _key_to_str(self, key) -> str:
        if hasattr(key, "char") and key.char:
            return key.char
        return key.name if hasattr(key, "name") else str(key)

    def _flush_typing_before_pointer_action(self) -> None:
        if self.key_flush_timer:
            self.key_flush_timer.cancel()
            self.key_flush_timer = None
        self._flush_key_buffer()

    def _flush_key_buffer(self):
        if not self.key_buffer:
            return
        typed = "".join(self.key_buffer)
        evt = Event(
            id=uuid.uuid4().hex[:10],
            timestamp=self.key_buffer_start_time,
            event_type="type",
            x=self.key_buffer_pos[0],
            y=self.key_buffer_pos[1],
            key=typed,
        )
        self._emit(evt)
        self.key_buffer.clear()

    def _buffer_key(self, char: str):
        if self.key_flush_timer:
            self.key_flush_timer.cancel()

        if not self.key_buffer:
            self.key_buffer_start_time = time.time()
            self.key_buffer_pos = (self.last_mouse_x, self.last_mouse_y)

        self.key_buffer.append(char)

        self.key_flush_timer = threading.Timer(
            self.key_flush_delay, self._flush_key_buffer
        )
        self.key_flush_timer.daemon = True
        self.key_flush_timer.start()

    _MODIFIER_NAMES = frozenset(
        {
            "ctrl",
            "ctrl_l",
            "ctrl_r",
            "alt",
            "alt_l",
            "alt_r",
            "shift",
            "shift_l",
            "shift_r",
            "cmd",
            "cmd_l",
            "cmd_r",
        }
    )
    _SHIFT_ONLY_MODIFIERS = frozenset({"shift", "shift_l", "shift_r"})
    _SPECIAL_KEY_NAMES = frozenset(
        {
            "enter",
            "return",
            "tab",
            "backspace",
            "delete",
            "up",
            "down",
            "left",
            "right",
            "home",
            "end",
            "page_up",
            "page_down",
            "f1",
            "f2",
            "f3",
            "f4",
            "f5",
            "f6",
            "f7",
            "f8",
            "f9",
            "f10",
            "f11",
            "f12",
        }
    )

    def _on_key_press(self, key):
        key_str = self._key_to_str(key)

        # Graceful stops (before mutating held_keys). Prefer F12 / Ctrl+Shift+Q in Docker.
        if self._is_ctrl_shift_q_press(key):
            self._request_graceful_stop()
            return False
        if self._is_escape_key(key):
            self._request_graceful_stop()
            return False
        if key_str.lower() == "f12":
            self._request_graceful_stop()
            return False

        self.held_keys.add(key_str)

        if key_str in self._MODIFIER_NAMES:
            return

        if key_str == "space":
            self._buffer_key(" ")
            return

        if key_str in self._SPECIAL_KEY_NAMES:
            if self.key_flush_timer:
                self.key_flush_timer.cancel()
                self.key_flush_timer = None
            self._flush_key_buffer()
            evt = Event(
                id=uuid.uuid4().hex[:10],
                timestamp=time.time(),
                event_type="keypress",
                x=self.last_mouse_x,
                y=self.last_mouse_y,
                key=key_str,
            )
            self._emit(evt)
            return

        if hasattr(key, "char") and key.char:
            c = key.char
            if c.isprintable() or c in "\n\r\t":
                self._buffer_key(c)

    def _on_key_release(self, key):
        key_str = self._key_to_str(key)

        if self._is_escape_key(key):
            self.held_keys.discard(key_str)
            if not self._stop_event.is_set():
                self._request_graceful_stop()
            return False

        held_modifiers = self.held_keys & self._MODIFIER_NAMES
        non_modifiers = self.held_keys - self._MODIFIER_NAMES

        if (
            held_modifiers
            and non_modifiers
            and not (held_modifiers <= self._SHIFT_ONLY_MODIFIERS)
        ):
            self._flush_key_buffer()
            combo = "+".join(sorted(held_modifiers) + sorted(non_modifiers))
            evt = Event(
                id=uuid.uuid4().hex[:10],
                timestamp=time.time(),
                event_type="key_combo",
                x=self.last_mouse_x,
                y=self.last_mouse_y,
                key=combo,
            )
            self._emit(evt)
        elif key_str not in self._MODIFIER_NAMES:
            if (
                not (hasattr(key, "char") and key.char)
                and len(key_str) == 1
                and key_str.isprintable()
            ):
                self._buffer_key(key_str)

        self.held_keys.discard(key_str)

    def _check_macos_permissions(self) -> bool:
        if sys.platform != "darwin":
            return True
        import ctypes

        appservices = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices"
        )
        appservices.AXIsProcessTrusted.restype = ctypes.c_bool
        trusted = appservices.AXIsProcessTrusted()
        if not trusted:
            print(
                "\n[ERROR] macOS Accessibility permission not granted.\n"
                "  1. Open System Settings → Privacy & Security → Accessibility\n"
                "  2. Add and enable your terminal app (iTerm2, Terminal, Cursor, etc.)\n"
                "  3. Do the same under Input Monitoring\n"
                "  4. Restart the terminal and rerun this script.\n",
                file=sys.stderr,
            )
            return False
        return True

    def start(self):
        if not self._check_macos_permissions():
            return

        print(f"Recording to: {self.session_dir}")
        print(
            "Stop: ESC, or F12, or Ctrl+Shift+Q (recommended in Docker—terminal may swallow ESC).\n"
            "Ctrl+C stops without post-processing unless RECORDER_POST_ON_CTRL_C=1.\n"
            f"RECORDER_POST_ON_CTRL_C={os.environ.get('RECORDER_POST_ON_CTRL_C', '').strip().lower()}"
        )
        self.running = True

        self._screenshot_thread = threading.Thread(
            target=self._screenshot_loop, daemon=True
        )
        self._screenshot_thread.start()

        if sys.platform == "darwin":
            threading.Thread(target=self._secure_input_poll_loop, daemon=True).start()

        processor = threading.Thread(target=self._process_events, daemon=False)
        processor.start()

        self._mouse_listener = mouse.Listener(
            on_click=self._on_click,
            on_scroll=self._on_scroll,
            on_move=self._on_move,
        )
        self._key_listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        )

        self._mouse_listener.start()
        self._key_listener.start()

        try:
            self._stop_event.wait()
        except KeyboardInterrupt:
            self._stop_requested_by_esc = True

        if self.key_flush_timer:
            self.key_flush_timer.cancel()
        self._flush_key_buffer()

        self.running = False
        self._mouse_listener.stop()
        self._key_listener.stop()

        if hasattr(self, "_screenshot_thread"):
            self._screenshot_thread.join(timeout=5.0)

        self.event_queue.put(None)
        processor.join(timeout=60)

        event_count = sum(1 for _ in open(self.events_file))
        frame_count = len(list(self.frames_dir.glob("*.png")))
        print(f"\nDone. Captured {event_count} events, {frame_count} frames.")
        print(f"Session: {self.session_dir}")

        # if self._stop_requested_by_esc:
        #     self._run_post_process_gui_recording()

    def stop(self):
        if self._stop_event.is_set():
            return
        self.running = False
        self._stop_event.set()

    def _run_post_process_gui_recording(self) -> None:
        """POST session zip to ``{SERVER_URL}/process_gui_recording_file`` and save JSON locally."""
        base = settings.SERVER_URL.strip().rstrip("/")
        if not base:
            print(
                "\n[WARNING] SERVER_URL is not set; skipping remote post-processing.",
                file=sys.stderr,
            )
            return

        recording_id = self.session_dir.name
        url = urljoin(base, "process_gui_recording_file")
        save_dir = self.session_dir / "processed"
        save_dir.mkdir(parents=True, exist_ok=True)

        try:
            zip_bytes = _zip_session_dir(self.session_dir)
        except Exception as e:
            print(f"\n[ERROR] Could not zip session: {e}", file=sys.stderr)
            return

        try:
            with httpx.Client(timeout=600.0) as client:
                r = client.post(
                    url,
                    files={
                        "session_zip": (
                            "session.zip",
                            zip_bytes,
                            "application/zip",
                        )
                    },
                    data={"recording_id": recording_id},
                )
            r.raise_for_status()
            payload = r.json()
        except httpx.HTTPStatusError as e:
            detail = ""
            try:
                detail = e.response.text[:500]
            except Exception:
                pass
            print(
                f"\n[ERROR] Remote post-processing HTTP {e.response.status_code}: {detail}",
                file=sys.stderr,
            )
            return
        except Exception as e:
            print(f"\n[ERROR] Remote post-processing failed: {e}", file=sys.stderr)
            return

        if not payload.get("ok"):
            print(f"\n[ERROR] Unexpected response: {payload}", file=sys.stderr)
            return

        def _write_json(name: str, obj: object | None) -> None:
            if obj is None:
                return
            p = save_dir / name
            p.write_text(
                json.dumps(obj, indent=4, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

        _write_json("automation.json", payload.get("automation"))
        _write_json("initial_automation.json", payload.get("initial_automation"))
        _write_json("extra.json", payload.get("extra"))
        _write_json(
            "automation_parameterized.json", payload.get("automation_parameterized")
        )

        print(
            f"\nProcessed automation saved under:\n  {save_dir}\n"
            f"  - automation.json\n"
            f"  - initial_automation.json\n"
            f"  - automation_parameterized.json (if server returned)\n"
            f"  - extra.json"
        )


def inspect_session(session_path: str):
    """Print all events from a recorded session."""
    events_file = Path(session_path) / "events.jsonl"
    if not events_file.exists():
        print(f"No events.jsonl in {session_path}")
        return

    for line in open(events_file):
        evt = json.loads(line)
        ts = datetime.fromtimestamp(evt["timestamp"]).strftime("%H:%M:%S.%f")[:-3]
        etype = evt["event_type"]
        x, y = evt["x"], evt["y"]

        if etype == "type":
            print(f"[{ts}] {etype:14s} typed=\"{evt['key']}\"")
        elif etype == "keypress":
            print(f"[{ts}] {etype:14s} key={evt['key']}")
        elif etype == "key_combo":
            print(f"[{ts}] {etype:14s} combo={evt['key']}")
        elif etype == "scroll":
            print(f"[{ts}] {etype:14s} ({x}, {y}) dy={evt['scroll_dy']}")
        elif etype == "drag":
            print(
                f"[{ts}] {etype:14s} ({evt['drag_start_x']},{evt['drag_start_y']}) -> ({x},{y})"
            )
        else:
            print(f"[{ts}] {etype:14s} ({x}, {y}) btn={evt['button']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="GUI Action Recorder (remote processing)"
    )
    parser.add_argument("--inspect", type=str, help="Path to session dir to inspect")
    args = parser.parse_args()

    if args.inspect:
        inspect_session(args.inspect)
    else:
        recorder = Recorder()
        recorder.start()
