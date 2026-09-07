"""Offline harness for the best-of-N calibration draw selection
(sn_gamestate/calibration/broadtrack_api.py) with a STUBBED binary.

The module itself imports torch/tracklab, so the harness extracts the two
methods under test (``_attempt_quality``, ``_run_binary_best_of``) from the
INSTALLED source text and executes that exact text against a dummy stage whose
``_run_binary`` writes canned per-frame score JSONs (or fails). Four cases:

1. best-of-3 keeps the attempt with the highest mean accepted score, deletes
   the losing candidates, and writes an exact ``<seq>.selection.json`` sidecar;
2. ``calib_attempts: 1`` makes a single direct call and writes no sidecar;
3. a failed attempt is tolerated and the best surviving attempt wins;
4. every attempt failing returns failure (and leaves no candidate files).

Run directly (``python tests/test_broadtrack_bestof.py``) or under pytest.
"""
import json
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent.parent / \
      "sn_gamestate" / "calibration" / "broadtrack_api.py"


def _extract(name, source):
    m = re.search(rf"(    def {name}\(self.*?)(?=\n    def )", source, re.S)
    assert m, f"method {name} not found in {SRC}"
    return m.group(1)


class _Log:
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass
    def error(self, *a, **k): pass


def make_stage(tmp, calib_attempts, script):
    """A dummy stage running the extracted methods. ``script`` is a list of
    per-attempt actions: a list of frame scores (the stub writes that JSON and
    succeeds) or None (the attempt fails, nothing written)."""
    source = SRC.read_text()
    body = ("class Stage:\n"
            + _extract("_attempt_quality", source)
            + "\n" + _extract("_run_binary_best_of", source) + "\n")
    ns = {"json": json, "np": np, "Path": Path, "log": _Log()}
    exec(body, ns)
    st = ns["Stage"]()
    st.min_score = 0.3
    st.calib_attempts = calib_attempts
    st.calls = []

    def _run_binary(frames_dir, out_json, tripod_file=None):
        act = script[len(st.calls)]
        st.calls.append(str(out_json))
        if act is None:
            return False
        payload = {f"/f/{i:06d}.jpg": {"score": s, "cp": {}}
                   for i, s in enumerate(act, start=1)}
        Path(out_json).write_text(json.dumps(payload))
        return True

    st._run_binary = _run_binary
    return st


def run_case(calib_attempts, script):
    tmp = Path(tempfile.mkdtemp())
    try:
        out = tmp / "SNGS-116.json"
        st = make_stage(tmp, calib_attempts, script)
        ok = st._run_binary_best_of(Path("/frames"), out, None)
        files = sorted(p.name for p in tmp.iterdir())
        sel_p = tmp / "SNGS-116.selection.json"
        sel = json.loads(sel_p.read_text()) if sel_p.is_file() else None
        data = json.loads(out.read_text()) if out.is_file() else None
        return ok, files, sel, data, st.calls
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_best_of_3_keeps_highest_mean_accepted():
    # attempt means over accepted (>= 0.3): 1: 0.594, 2: 0.590, 3: 0.613
    script = [[0.594, 0.594, 0.1], [0.59, 0.59, 0.2], [0.613, 0.613, 0.25]]
    ok, files, sel, data, calls = run_case(3, script)
    assert ok and len(calls) == 3
    assert files == ["SNGS-116.json", "SNGS-116.selection.json"]  # losers deleted
    assert sel["winner"] == 3 and sel["calib_attempts"] == 3
    assert sel["min_score"] == 0.3 and sel["sequence"] == "SNGS-116"
    a = sel["attempts"]
    assert [x["attempt"] for x in a] == [1, 2, 3]
    assert all(x["ok"] for x in a)
    assert abs(a[0]["mean_accepted_score"] - 0.594) < 1e-12
    assert abs(a[2]["mean_accepted_score"] - 0.613) < 1e-12
    assert [x["accepted_frames"] for x in a] == [2, 2, 2]
    assert [x["frames"] for x in a] == [3, 3, 3]
    # the kept JSON is attempt 3's payload
    assert abs(list(data.values())[0]["score"] - 0.613) < 1e-12


def test_tiebreak_on_accepted_frame_count():
    # equal means (0.5); attempt 2 has MORE accepted frames -> wins
    script = [[0.5, 0.5, 0.1], [0.5, 0.5, 0.5]]
    st_ok, files, sel, data, _ = run_case(2, script)
    assert st_ok and sel["winner"] == 2
    assert sel["attempts"][1]["accepted_frames"] == 3


def test_single_attempt_direct_no_sidecar():
    script = [[0.9, 0.9]]
    ok, files, sel, data, calls = run_case(1, script)
    assert ok and sel is None                      # no selection sidecar
    assert files == ["SNGS-116.json"]
    assert len(calls) == 1 and calls[0].endswith("SNGS-116.json")  # direct target


def test_failed_attempt_tolerated_best_survivor_wins():
    script = [[0.4, 0.4], None, [0.8, 0.8]]
    ok, files, sel, data, calls = run_case(3, script)
    assert ok and len(calls) == 3
    assert sel["winner"] == 3
    assert sel["attempts"][1]["ok"] is False
    assert "mean_accepted_score" not in sel["attempts"][1]
    assert abs(list(data.values())[0]["score"] - 0.8) < 1e-12
    assert files == ["SNGS-116.json", "SNGS-116.selection.json"]


def test_all_attempts_fail_returns_failure():
    ok, files, sel, data, calls = run_case(3, [None, None, None])
    assert ok is False and len(calls) == 3
    assert files == [] and sel is None and data is None  # nothing left behind


def _main():
    import sys
    mod = sys.modules[__name__]
    names = sorted(n for n in dir(mod) if n.startswith("test_"))
    for n in names:
        getattr(mod, n)()
        print(f"ok {n}")
    print(f"{len(names)} tests passed")


if __name__ == "__main__":
    _main()
