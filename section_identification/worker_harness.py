"""Generic background-worker harness: run a STIM_* streaming subprocess and
dispatch its messages to handlers.

Generalises the QProcess+stdout pattern that detect_worker/interface use, so the
QC and reorder stages share one launcher. Each emitted ``STIM_<TAG> <json>`` line
is parsed (worker_protocol) and routed to ``handlers[tag](payload)``; non-STIM
stdout is forwarded to ``on_log`` so the footer log mirrors the worker. Runs the
child in the home dir (avoids the in-repo ``sam2/`` shadowing, as detect_worker
does) and streams line-buffered.
"""

from __future__ import annotations

import os
import sys

from qtpy.QtCore import QObject, QProcess, Signal

from . import worker_protocol as wp


class StreamWorker(QObject):
    """Launch ``python -m section_identification.<module> <args>`` and stream."""

    finished = Signal(int)        # exit code
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.proc = None
        self._buf = ""
        self._handlers = {}
        self._on_log = None

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.state() != QProcess.NotRunning

    def start(self, module: str, args, handlers: dict, on_log=None) -> bool:
        if self.is_running():
            return False
        self._handlers = handlers or {}
        self._on_log = on_log
        self._buf = ""
        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        try:
            self.proc.setWorkingDirectory(os.path.expanduser("~"))
        except Exception:
            pass
        self.proc.readyReadStandardOutput.connect(self._on_output)
        self.proc.finished.connect(self._on_finished)
        argv = ["-m", f"section_identification.{module}"] + [str(a) for a in args]
        self.proc.start(sys.executable, argv)
        return True

    def stop(self):
        if self.is_running():
            try:
                self.proc.kill()
            except Exception:
                pass

    # -- internals --
    def _on_output(self):
        try:
            chunk = bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        except Exception:
            return
        self._buf += chunk
        *lines, self._buf = self._buf.split("\n")
        for line in lines:
            msg = wp.parse_line(line)
            if msg is None:
                if self._on_log and line.strip():
                    self._on_log(line.rstrip())
                continue
            tag, payload = msg
            fn = self._handlers.get(tag)
            if fn is not None:
                try:
                    fn(payload)
                except Exception as e:                # a handler bug must not kill the stream
                    if self._on_log:
                        self._on_log(f"[handler:{tag}] {e}")

    def _on_finished(self, code, _status=None):
        # flush any trailing partial line
        if self._buf.strip():
            msg = wp.parse_line(self._buf)
            if msg is not None:
                fn = self._handlers.get(msg[0])
                if fn:
                    try:
                        fn(msg[1])
                    except Exception:
                        pass
            self._buf = ""
        self.finished.emit(int(code))
