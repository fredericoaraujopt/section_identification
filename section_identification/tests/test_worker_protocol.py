"""Headless tests for the STIM_* worker line protocol.

Run:  python -m section_identification.tests.test_worker_protocol
"""

from __future__ import annotations

from section_identification import worker_protocol as wp


def test_emit_parse_roundtrip_payload():
    line = wp.emit(wp.TILE, {"k": 1, "n": 9, "sections": [{"poly": [[0, 0]], "area": 3}]})
    tag, payload = wp.parse_line(line)
    assert tag == "TILE"
    assert payload["k"] == 1 and payload["sections"][0]["area"] == 3


def test_emit_parse_no_payload():
    assert wp.parse_line(wp.emit(wp.DONE)) == ("DONE", None)


def test_non_stim_line_ignored():
    assert wp.parse_line("just some log text") is None
    assert wp.parse_line("") is None


def test_legacy_colon_form():
    tag, payload = wp.parse_line("STIM_DONE: 5 sections")
    assert tag == "DONE" and payload == "5 sections"


def test_iter_messages_filters():
    text = "\n".join([
        "loading image...",
        wp.emit(wp.PROGRESS, {"done": 1, "total": 3}),
        "noise",
        wp.emit(wp.QC, {"section_id": "section_2", "scores": {"overall": 0.8}}),
        wp.emit(wp.DONE),
    ])
    msgs = list(wp.iter_messages(text))
    assert [t for t, _ in msgs] == ["PROGRESS", "QC", "DONE"]
    assert msgs[1][1]["section_id"] == "section_2"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} worker_protocol tests passed.")


if __name__ == "__main__":
    _run_all()
