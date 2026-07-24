"""Atomic, corruption-safe file writes.

An **atomic** write means a reader of the destination path only ever sees either
the complete *previous* file or the complete *new* file — never a half-written
one. We get this by writing to a temporary sibling ``<path>.part`` and then
calling :func:`os.replace`, which swaps it into place with a single, indivisible
filesystem rename (on the same volume). If anything fails mid-write — a crash, a
raised exception, a full disk, a failed verification — the destination is left
exactly as it was and the partial ``.part`` is discarded.

Because the original file is untouched until that final atomic swap, it is
effectively its own backup: there is never a moment where a good old file has
been destroyed but the new one is not yet complete. An optional ``verify_fn``
runs on the freshly written temp file *before* the swap, so a file that is
complete-but-invalid (e.g. truncated JSON) also can't replace a good one.
"""

from __future__ import annotations

import json
import os
from typing import Callable, Optional


def _remove_quietly(path: str) -> None:
    try:
        if os.path.lexists(path):
            os.remove(path)
    except OSError:
        pass


def atomic_write(path: str,
                 write_fn: Callable[[str], None],
                 *,
                 verify_fn: Optional[Callable[[str], bool]] = None) -> str:
    """Write ``path`` atomically.

    ``write_fn(tmp_path)`` must write the *entire* content to ``tmp_path``.
    ``verify_fn(tmp_path) -> bool`` (optional) validates that temp file before it
    is published; returning False aborts the write with the destination
    untouched. Returns ``path`` on success; re-raises on any failure (after
    removing the temp file).
    """
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    tmp = f"{path}.part"
    _remove_quietly(tmp)                       # never trust a stale partial
    try:
        write_fn(tmp)
        if verify_fn is not None and not verify_fn(tmp):
            raise ValueError(f"atomic_write: verification failed for {path!r}")
        os.replace(tmp, path)                  # atomic swap on the same filesystem
        return path
    except BaseException:
        _remove_quietly(tmp)
        raise


def _json_reparses(tmp: str) -> bool:
    """Default verifier for JSON: the temp file must parse back (catches a
    truncated / half-flushed dump before it can replace a good file)."""
    try:
        with open(tmp) as f:
            json.load(f)
        return True
    except Exception:
        return False


def atomic_write_json(path: str, obj, *, indent=None,
                      verify_fn: Optional[Callable[[str], bool]] = None) -> str:
    """``json.dump`` ``obj`` to ``path`` atomically; verifies it re-parses."""
    def _w(tmp: str) -> None:
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=indent)
    return atomic_write(path, _w, verify_fn=verify_fn or _json_reparses)


def atomic_write_text(path: str, text: str, *, encoding: str = "utf-8",
                      newline: Optional[str] = None) -> str:
    """Write ``text`` to ``path`` atomically."""
    def _w(tmp: str) -> None:
        with open(tmp, "w", encoding=encoding, newline=newline) as f:
            f.write(text)
    return atomic_write(path, _w)
