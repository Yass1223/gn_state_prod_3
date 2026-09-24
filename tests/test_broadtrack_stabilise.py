"""Offline harness for the temporal stabilisation of the BroadTrack cameras
(sn_gamestate/calibration/broadtrack_api.py: ``stabilise_sequence`` and its
helpers). The module imports torch/tracklab, so the harness extracts the
module-level functions under test from the INSTALLED source text and executes
that exact text against a toy camera model: a "camera" is ``{"pan_degrees": p,
"shift_m": [sx, sy]}`` and the projection of an image point is the point plus
the shift (metres), so a camera error shows up as a rigid displacement of every
projected player -- exactly what a wrong homography does.

Cases:
1.  carry-then-snap: a rejected run between two confident frames is
    interpolated, not carried, and the residual jump count is zero;
2.  drift tail: frames whose scores decay towards a re-initialisation are
    demoted / replaced by interpolation ending exactly on the re-init camera;
3.  a single wrong-but-scored frame (spike) between two confident frames is
    demoted and interpolated away;
4.  a continuous run of weak scores with no rejected frame is left untouched;
5.  a jump between two confident frames is NOT hidden: it is reported as a
    residual jump;
6.  gaps longer than max_interp_frames are not bridged (carry-forward, capped
    by max_carry_frames) and the last, never-calibrated frame is carried;
7.  stabilise=False reproduces the score-gate + carry-forward behaviour;
8.  no shared tracks (untracked detections) disables the jump test but the
    interpolation of rejected runs still applies;
9.  angle interpolation takes the shortest arc; parameter dicts interpolate
    element-wise.

Run directly (``python tests/test_broadtrack_stabilise.py``) or under pytest.
"""
import re
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent.parent / \
      "sn_gamestate" / "calibration" / "broadtrack_api.py"

FUNCS = ["_lerp_angle", "interpolate_parameters", "pitch_jump", "_fill_runs",
         "stabilise_sequence"]


def _extract_top(name, source):
    m = re.search(rf"(^def {name}\(.*?)(?=^(?:def |class |# ---))", source, re.S | re.M)
    assert m, f"function {name} not found in {SRC}"
    return m.group(1)


def load():
    source = SRC.read_text()
    body = "\n".join(_extract_top(n, source) for n in FUNCS)
    ns = {"np": np}
    exec(body, ns)
    return ns


NS = load()
stabilise_sequence = NS["stabilise_sequence"]
interpolate_parameters = NS["interpolate_parameters"]
_lerp_angle = NS["_lerp_angle"]


# ---------------------------------------------------------------- toy camera
def cam(shift, pan=0.0):
    return {"pan_degrees": float(pan), "shift_m": [float(shift[0]), float(shift[1])]}


def project(params, pts):
    return np.asarray(pts, float) + np.asarray(params["shift_m"], float)


def frames(n, n_players=6):
    """Static players (image points) with stable track ids on every frame."""
    pts = [np.column_stack([np.linspace(100, 700, n_players),
                            np.full(n_players, 500.0)]) for _ in range(n)]
    ids = [np.arange(n_players, dtype=float) for _ in range(n)]
    return pts, ids


DEFAULTS = dict(min_score=0.3, anchor_score=0.5, max_jump_m=2.0, jump_min_tracks=3,
                max_interp_frames=50, use_prev_parameters=True, max_carry_frames=0)


def run(params, scores, n_players=6, **over):
    kw = dict(DEFAULTS)
    kw.update(over)
    pts, ids = frames(len(params), n_players)
    return stabilise_sequence(params, scores, pts, ids, project, **kw)


def shifts(final):
    return [None if p is None else tuple(round(v, 6) for v in p["shift_m"]) for p in final]


# ---------------------------------------------------------------- cases
def test_carry_then_snap_is_interpolated():
    # good camera at (0,0) for 5 frames, 4 rejected frames, good camera at (4,0)
    params = [cam((0, 0))] * 5 + [cam((0, 0))] * 4 + [cam((4, 0))] * 5
    scores = [0.7] * 5 + [0.1] * 4 + [0.7] * 5
    final, source, rep = run(params, scores)
    assert source[:5] == ["binary"] * 5 and source[9:] == ["binary"] * 5
    assert source[5:9] == ["interp"] * 4
    # linear ramp from frame 4 (0) to frame 9 (4): 0.8 m per frame
    assert shifts(final)[4:10] == [(0, 0), (0.8, 0), (1.6, 0), (2.4, 0), (3.2, 0), (4, 0)]
    assert rep["jumps_raw"] == [] and rep["jumps_final"] == []
    assert rep["sources"]["carry"] == 0 and rep["sources"]["none"] == 0


def test_drift_tail_replaced_up_to_reinit():
    # confident frames at (0,0); a drift of 5 frames with decaying scores that
    # wanders to (-3,0); 2 lost frames; re-init back at (0.2,0) with high score
    drift = [cam((-0.6 * k, 0)) for k in range(1, 6)]
    params = [cam((0, 0))] * 4 + drift + [cam((-3, 0))] * 2 + [cam((0.2, 0))] * 4
    scores = [0.7] * 4 + [0.48, 0.45, 0.42, 0.38, 0.33] + [0.15, 0.1] + [0.7] * 4
    final, source, rep = run(params, scores)
    # the whole drift tail + lost run is one non-anchor run with bad frames inside
    assert source[4:11] == ["interp"] * 7
    s = shifts(final)
    # monotone ramp from (0,0) at frame 3 to (0.2,0) at frame 11, no excursion to -3
    xs = [v[0] for v in s[3:12]]
    assert all(b >= a for a, b in zip(xs, xs[1:]))
    assert min(xs) == 0.0 and max(xs) == 0.2
    assert rep["jumps_final"] == []


def test_spike_frame_demoted_and_bridged():
    params = [cam((0, 0))] * 4 + [cam((5, 5))] + [cam((0, 0))] * 4
    scores = [0.7] * 4 + [0.55] + [0.7] * 4     # the spike still scores above anchor
    final, source, rep = run(params, scores)
    assert [i for i, _ in rep["jumps_raw"]] == [4, 5]
    assert rep["demoted"] == [4]
    assert source[4] == "interp" and shifts(final)[4] == (0, 0)
    assert rep["jumps_final"] == []


def test_weak_but_continuous_run_untouched():
    params = [cam((0, 0))] * 3 + [cam((0.1 * k, 0)) for k in range(1, 8)] + [cam((0.7, 0))] * 3
    scores = [0.7] * 3 + [0.4] * 7 + [0.7] * 3
    final, source, rep = run(params, scores)
    assert source == ["binary"] * 13
    assert [p["shift_m"] for p in final] == [p["shift_m"] for p in params]
    assert rep["interp_runs"] == [] and rep["jumps_raw"] == []


def test_confident_jump_is_reported_not_hidden():
    params = [cam((0, 0))] * 6 + [cam((6, 0))] * 6
    scores = [0.7] * 12
    final, source, rep = run(params, scores)
    assert rep["jumps_raw"] == [[6, 6.0]]
    assert rep["demoted"] == [5]           # tie -> the earlier frame loses
    assert source[5] == "interp"           # bridged between 4 and 6: halves the step
    assert shifts(final)[5] == (3, 0)
    # both remaining steps are 3 m > max_jump_m: residual jumps reported at 5 and 6
    assert [i for i, _ in rep["jumps_final"]] == [5, 6]
    assert all(abs(d - 3.0) < 1e-9 for _, d in rep["jumps_final"])


def test_long_gap_not_bridged_and_last_frame_carried():
    params = [cam((0, 0))] * 3 + [cam((0, 0))] * 8 + [cam((2, 0))] * 3 + [None]
    scores = [0.7] * 3 + [0.1] * 8 + [0.7] * 3 + [None]
    final, source, rep = run(params, scores, max_interp_frames=4, max_carry_frames=5)
    assert source[3:8] == ["carry"] * 5 and source[8:11] == ["none"] * 3
    assert final[8] is None and final[10] is None
    assert source[14] == "carry" and shifts(final)[14] == (2, 0)   # last frame
    assert rep["interp_runs"] == [] and rep["interp_weak_runs"] == []


def test_stabilise_false_is_previous_behaviour():
    params = [cam((0, 0))] * 3 + [cam((9, 9))] + [cam((0, 0))] * 2 + [cam((0, 0))] * 2 + [None]
    scores = [0.7] * 3 + [0.6] + [0.7] * 2 + [0.1] * 2 + [None]
    final, source, rep = run(params, scores, stabilise=False)
    assert source == ["binary"] * 6 + ["carry"] * 3
    assert shifts(final)[3] == (9, 9)                  # the spike is kept
    assert rep["jumps_raw"] == [] and rep["jumps_final"] == [] and rep["demoted"] == []


def test_no_tracks_disables_jump_test_only():
    params = [cam((0, 0))] * 3 + [cam((9, 9))] + [cam((0, 0))] * 2 + [cam((0, 0))] * 2 + [cam((1, 0))] * 3
    scores = [0.7] * 3 + [0.6] + [0.7] * 2 + [0.1] * 2 + [0.7] * 3
    pts, ids = frames(len(params))
    ids = [np.full(len(i), np.nan) for i in ids]          # untracked detections
    final, source, rep = stabilise_sequence(params, scores, pts, ids, project, **DEFAULTS)
    assert rep["jumps_raw"] == [] and rep["demoted"] == []
    assert shifts(final)[3] == (9, 9)                      # undetectable without tracks
    assert source[6:8] == ["interp"] * 2                   # rejected run still bridged
    assert np.allclose([p["shift_m"] for p in final[6:8]], [(1 / 3, 0), (2 / 3, 0)])


def test_angle_and_parameter_interpolation():
    assert abs(_lerp_angle(170.0, -170.0, 0.5) - 180.0) < 1e-12
    assert abs(_lerp_angle(-170.0, 170.0, 0.25) - (-175.0)) < 1e-12
    a = {"pan_degrees": 10.0, "x_focal_length": 1000.0, "position_meters": [0, 0, -10],
         "radial_distortion": [0.1, 0, 0, 0, 0, 0]}
    b = {"pan_degrees": 20.0, "x_focal_length": 2000.0, "position_meters": [2, 4, -12],
         "radial_distortion": [0.3, 0, 0, 0, 0, 0]}
    m = interpolate_parameters(a, b, 0.5)
    assert m == {"pan_degrees": 15.0, "x_focal_length": 1500.0,
                 "position_meters": [1.0, 2.0, -11.0],
                 "radial_distortion": [0.2, 0.0, 0.0, 0.0, 0.0, 0.0]}
    assert interpolate_parameters(a, b, 0.0) == {k: (list(map(float, v)) if isinstance(v, list) else float(v))
                                                 for k, v in a.items()}


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
