#!/usr/bin/env python3
"""Add focused-terminal n/r/q controls to the older LeRobot recorder.

Usage: python fix_lerobot_keyboard.py
       python fix_lerobot_keyboard.py --test
The default target is ~/soarm101/lerobot/src/lerobot/utils/control_utils.py.
Run only after the current recorder has finished. The original is backed up.
"""
import argparse
import ast
from datetime import datetime
import os
from pathlib import Path
import shutil
import tempfile
import time

LISTENER_SOURCE = r'''def init_keyboard_listener():
    """Read recording controls from the focused terminal."""
    import atexit
    import os
    import select
    import sys
    import termios
    import threading
    import tty

    if not sys.stdin.isatty():
        raise RuntimeError("Run lerobot-record directly in an interactive terminal.")
    events = dict(exit_early=False, rerecord_episode=False, stop_recording=False)

    class TerminalListener:
        def __init__(self):
            self.fd = sys.stdin.fileno()
            self.saved = termios.tcgetattr(self.fd)
            self.done = threading.Event()
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.lock = threading.Lock()
            self.restored = False

        def restore(self):
            with self.lock:
                if not self.restored:
                    termios.tcsetattr(self.fd, termios.TCSANOW, self.saved)
                    self.restored = True

        def start(self):
            tty.setcbreak(self.fd)
            atexit.register(self.stop)
            try:
                self.thread.start()
            except BaseException:
                self.restore()
                raise

        def run(self):
            try:
                while not self.done.is_set():
                    ready, _, _ = select.select([self.fd], [], [], 0.1)
                    if not ready:
                        continue
                    data = os.read(self.fd, 64)
                    if not data:
                        break
                    for key in data:
                        if key == ord('n'):
                            events['exit_early'] = True
                            print('\n[KEY] next', flush=True)
                        elif key == ord('r'):
                            events['rerecord_episode'] = True
                            events['exit_early'] = True
                            print('\n[KEY] re-record', flush=True)
                        elif key == ord('q'):
                            events['stop_recording'] = True
                            events['exit_early'] = True
                            print('\n[KEY] finish', flush=True)
            finally:
                self.restore()

        def stop(self):
            self.done.set()
            if self.thread.is_alive() and threading.current_thread() is not self.thread:
                self.thread.join(timeout=0.3)
            self.restore()

    listener = TerminalListener()
    listener.start()
    print('Terminal controls: n=next, r=re-record, q=finish (no Enter).', flush=True)
    return listener, events
'''

def patch(target):
    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source)
    matches = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "init_keyboard_listener"]
    if len(matches) != 1:
        raise RuntimeError("Cannot uniquely locate init_keyboard_listener; no change made.")
    node = matches[0]
    lines = source.splitlines(keepends=True)
    old = "".join(lines[node.lineno - 1:node.end_lineno])
    if "Read recording controls from the focused terminal." in old:
        print("Terminal controls are already installed.")
        return
    if not all(s in old for s in ("keyboard.Listener", "exit_early", "rerecord_episode", "stop_recording")):
        raise RuntimeError("Unrecognized listener version; no change made.")
    changed = "".join(lines[:node.lineno - 1]) + LISTENER_SOURCE.rstrip() + "\n" + "".join(lines[node.end_lineno:])
    compile(changed, str(target), "exec")
    backup = target.with_name(target.name + ".keyboard-backup-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    shutil.copy2(target, backup)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as out:
            temporary = Path(out.name)
            out.write(changed)
        shutil.copymode(target, temporary)
        if target.read_text(encoding="utf-8") != source:
            raise RuntimeError("Source changed during patching; no change made.")
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print("Patched:", target)
    print("Original backup:", backup)
    print("Next: run this script with --test; press n, r, then q.")


def test_listener():
    namespace = {}
    exec(LISTENER_SOURCE, namespace)
    listener, events = namespace["init_keyboard_listener"]()
    try:
        while not events["stop_recording"]:
            time.sleep(0.05)
    finally:
        listener.stop()
    print("Test finished. This test did not connect to the robot.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=Path.home() / "soarm101/lerobot/src/lerobot/utils/control_utils.py")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    if args.test:
        test_listener()
    else:
        patch(args.target.expanduser().resolve())


if __name__ == "__main__":
    main()
