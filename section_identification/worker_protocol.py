"""The ``STIM_*`` line protocol shared by all background workers (pure).

Heavy stages run in a child process and stream newline-delimited ``STIM_<TAG>``
messages on stdout; the GUI parses them to update layers/log live. detect_worker
already uses ``STIM_TILES``/``STIM_TILE``/``STIM_DONE`` etc.; this centralises the
encode/decode so the new qc_worker and reorder_worker speak the same language and
the GUI has one parser.

A message is ``STIM_<TAG> <json>``; a tag with no payload is just ``STIM_<TAG>``.
Parsing tolerates the legacy ``STIM_DONE: 5 sections`` colon form (payload kept
as a raw string when it isn't JSON), so existing detect_worker output still
parses during the migration.
"""

from __future__ import annotations

import json
import re

PREFIX = "STIM_"

# canonical tags
TILES = "TILES"            # plan: all tile boxes (detection)
TILESTART = "TILESTART"    # a tile is starting
TILE = "TILE"              # a tile's results
PROGRESS = "PROGRESS"      # {done, total[, label]}
RESULT = "RESULT"          # a generic per-item result
DONE = "DONE"              # finished
ERROR = "ERROR"            # failure
QC = "QC"                  # {section_id, scores, flags}
QC_DONE = "QC_DONE"
REORDER_PROGRESS = "REORDER_PROGRESS"   # {done, total}
REORDER_DONE = "REORDER_DONE"           # {order, edges, ...}

_LINE_RE = re.compile(r"^([A-Z_]+)[:\s]+(.*)$")


def emit(tag: str, payload=None) -> str:
    """Encode one protocol line (no trailing newline)."""
    if payload is None:
        return f"{PREFIX}{tag}"
    return f"{PREFIX}{tag} {json.dumps(payload, separators=(',', ':'))}"


def parse_line(line: str):
    """Decode one line -> ``(tag, payload)`` or None if it isn't a STIM message.

    ``payload`` is the parsed JSON value, a raw string if the remainder isn't
    JSON (legacy ``DONE: 5 sections``), or None if there's no payload.
    """
    line = (line or "").strip()
    if not line.startswith(PREFIX):
        return None
    rest = line[len(PREFIX):]
    m = _LINE_RE.match(rest)
    if not m:
        return rest.strip(), None
    tag, blob = m.group(1), m.group(2).strip()
    if not blob:
        return tag, None
    try:
        return tag, json.loads(blob)
    except Exception:
        return tag, blob


def iter_messages(text: str):
    """Yield ``(tag, payload)`` for every STIM line in a block of text."""
    for line in text.splitlines():
        msg = parse_line(line)
        if msg is not None:
            yield msg
