"""Unit tests for the visualization colour contract
(sn_gamestate/visualization/players.py and pitch.py).

Under test: the role_team stage decides role/team per trajectory from its
single detections and writes them on ALL of its rows, so multi-crop detections
carry their trajectory's labels and both display layers draw them like any
other row -- same box colour and radar disc, no special handling; rows without
labels (untracked, or a labelling failure) stay undrawn.

Pure logic, no GPU and no frames. When tracklab (and cv2/distinctipy) are not
installed -- e.g. a bare sandbox -- minimal stubs are injected BEFORE the
visualizer modules import them; in the pipeline environment the real packages
are used. Run directly (``python tests/test_visualization.py``) or under pytest.
"""
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


try:                                     # real environment: nothing to do
    import tracklab.visualization        # noqa: F401
    import cv2                           # noqa: F401
    import distinctipy                   # noqa: F401
except ImportError:                      # sandbox: minimal import-time stubs
    class _Visualizer:
        def draw_frame(self, *a, **k):
            pass

        def post_init(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    _stub("tracklab")
    _stub("tracklab.visualization", Visualizer=_Visualizer,
          ImageVisualizer=_Visualizer, DefaultDetection=_Visualizer,
          EllipseDetection=_Visualizer,
          get_fixed_colors=lambda n: [(0.5, 0.5, 0.5)] * n)
    _stub("tracklab.utils")
    _stub("tracklab.utils.cv2", draw_text=lambda *a, **k: None)
    if "cv2" not in sys.modules:
        _stub("cv2", rectangle=lambda *a, **k: None,
              imread=lambda *a, **k: None, resize=lambda *a, **k: None,
              line=lambda *a, **k: None, circle=lambda *a, **k: None,
              addWeighted=lambda *a, **k: None,
              cvtColor=lambda img, *a, **k: img, COLOR_BGR2RGB=0,
              LINE_AA=16, FONT_HERSHEY_SIMPLEX=0)
    if "distinctipy" not in sys.modules:
        _stub("distinctipy", get_rgb256=lambda c: tuple(int(255 * x) for x in c))

from sn_gamestate.visualization.players import TeamVisualizer        # noqa: E402
from sn_gamestate.visualization.pitch import (COLOR_LEFT, COLOR_REFEREE,  # noqa: E402
                                              radar_color)

BP = {"x_bottom_middle": 0.0, "y_bottom_middle": 0.0}


class AD(dict):
    """Dict with attribute access, as the pipeline's OmegaConf colors give."""
    __getattr__ = dict.__getitem__


COLORS = {"cmap": 4,
          "default": AD(no_id=None, prediction="team", ground_truth=None),
          "team": {"no_team": None,
                   "prediction": {"left": [0, 0, 255], "right": [255, 0, 0],
                                  "referee": [238, 210, 2]},
                   "ground_truth": {"left": [0, 255, 0], "right": [0, 255, 0],
                                    "referee": [255, 255, 0]}}}


class Det:
    """Attribute view of one row, as the bbox visualizer receives it."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def bbox_vis():
    v = TeamVisualizer.__new__(TeamVisualizer)   # skip abstract draw_frame
    v.colors = COLORS
    v.cmap = [(128, 128, 128)] * 4
    return v


def test_bbox_multi_row_draws_like_its_trajectory():
    # a multi-crop row of a left-team player carries the trajectory's labels
    # (written by role_team on all rows) and gets the same colour as a single
    v = bbox_vis()
    single = v.color(Det(track_id=1.0, role="player", team="left"), True)
    multi = v.color(Det(track_id=1.0, role="player", team="left"), True)
    assert single == multi == COLORS["team"]["prediction"]["left"]
    ref = v.color(Det(track_id=2.0, role="referee", team=np.nan), True)
    assert ref == COLORS["team"]["prediction"]["referee"]


def test_bbox_unlabelled_and_untracked_stay_undrawn():
    v = bbox_vis()
    assert v.color(Det(track_id=3.0, role=np.nan, team=np.nan), True) is None
    assert v.color(Det(track_id=np.nan, role=None, team=None), True) is None


def test_radar_multi_row_draws_like_its_trajectory():
    multi = dict(track_id=1.0, role="player", team="left", bbox_pitch=BP)
    assert radar_color(multi) == COLOR_LEFT
    ref_multi = dict(track_id=2.0, role="referee", team=None, bbox_pitch=BP)
    assert radar_color(ref_multi) == COLOR_REFEREE


def test_radar_guards():
    assert radar_color(dict(track_id=1.0, role="player", team="left",
                            bbox_pitch=None)) is None       # no pitch position
    assert radar_color(dict(track_id=3.0, role=None, team=None,
                            bbox_pitch=BP)) is None         # no labels
    assert radar_color(dict(track_id=np.nan, role="player", team="left",
                            bbox_pitch=BP)) is None         # untracked
    assert radar_color(dict(track_id=1.0, role="ball", team=None,
                            bbox_pitch=BP)) is None         # ball


if __name__ == "__main__":
    mod = sys.modules[__name__]
    names = [n for n in dir(mod) if n.startswith("test_")]
    for n in names:
        getattr(mod, n)()
        print(f"ok {n}")
    print(f"{len(names)} tests passed")
