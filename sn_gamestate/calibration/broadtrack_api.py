"""BroadTrack (EVS, WACV'25) as a TrackLab ``VideoLevelModule``.

Replaces the NBJW pair (``pitch`` = ``NBJW_Calib_Keypoints`` + ``calibration`` =
``NBJW_Calib``) with BroadTrack's temporal camera tracking, and performs the
image-to-pitch projection with the *same* code path the repo already uses
(``sn_calibration_baseline.Camera.unproject_point_on_planeZ0``), so the ``bbox_pitch``
contract is byte-compatible with the previous stages.

Engine contract (verified against tracklab 1.3.24 sources)
----------------------------------------------------------
* ``OfflineTrackingEngine.video_loop`` dispatches video-level modules as
  ``detections = model.process(detections, image_pred)`` — the return value MUST be a
  DataFrame (returning a ``(detections, metadatas)`` tuple would corrupt the loop).
* ``image_pred`` is passed by reference and later persisted via
  ``TrackerState.on_video_loop_end -> update -> save``, so the ``parameters`` image
  column is written **in place** on ``metadatas`` here and does persist.
* ``Pipeline.validate`` starts from empty column sets on a fresh run and accumulates
  declared outputs, so inputs below are producible upstream (``bbox_ltwh``/``image_id``
  come from the detector wrapper) and image-level ``parameters`` is declared as an
  output (dict form) so bookkeeping and state save/load stay coherent.

Verified behaviours of the BroadTrack binary this wrapper compensates for
-------------------------------------------------------------------------
* Frames must be ``%06d.jpg`` starting at ``000001.jpg``; ``frames_number`` counts the
  *regular files* in ``-f`` and the loop is ``for (i = 1; i < frames_number; i++)`` so
  the **last frame is never calibrated** -> carry-forward (``use_prev_parameters``).
* Output JSON keys are full path strings built from ``-f`` -> matched by basename.
* Player masking is read from ``<frames_dir>/human-bboxes/%06d.json`` (path hardcoded
  in ``main.cpp``; the ``-b`` flag is parsed but unused) -> ``write_human_bboxes``.
* ``-t <file>`` existing switches the binary to tripod ("soft") mode; otherwise "free"
  mode with the ``--X/--Y/--Z`` prior. The binary's built-in defaults are ``0/90/-18``
  (Bundesliga), *not* the SoccerNet prior, so priors are always passed explicitly.
* The binary is NONDETERMINISTIC across runs: identical inputs yield a different
  homography each session (measured ~9 GS-HOTA points of spread on SNGS-116 from
  the calibration draw alone). On a cache miss the stage therefore runs it
  ``calib_attempts`` times and keeps the draw whose accepted frames have the
  highest mean line-IoU score (the binary's own confidence; no ground truth),
  recording every attempt in ``<seq>.selection.json`` -- see
  ``_run_binary_best_of``. The cache (``use_cached_json``) then freezes the
  kept draw for every later run.
* Every frame is written with a line-IoU ``score`` and a ``reinit`` flag, INCLUDING
  frames on which tracking was lost. In ``main.cpp`` ``score < 0.3`` is the lost state
  (keypoint re-initialisation is attempted) and after more than 5 consecutive lost
  frames with ``score < 0.2`` the camera is RESET to the position prior (pan 0, tilt
  80 deg, focal = image diagonal); that camera is still recorded. Projecting boxes
  with it scatters every player of the frame across the pitch, which is why
  ``min_score`` must not be 0: rejected frames fall back to carry-forward
  (``use_prev_parameters``), optionally capped by ``max_carry_frames``. Upstream's own
  ``scripts/compute_tripod.py`` keeps only frames with ``score > 0.6``.
* The score gate alone does not remove SUDDEN JUMPS of the projected positions: a
  keypoint re-initialisation snaps the camera away from the drifted frames before
  it, and a carried-forward (frozen) camera snaps to the next accepted one. The
  per-frame cameras are therefore post-processed by :func:`stabilise_sequence`
  (``stabilise``): jumps are detected on PLAYER continuity (median pitch-plane
  displacement of the tracks shared by two consecutive frames > ``max_jump_m``),
  the lower-scoring side of a jump and every rejected run are replaced by linear
  interpolation of the camera between the surrounding confident frames
  (``anchor_score``, ``max_interp_frames``), and the residual jumps are reported in
  ``<seq>.stabilise.json``. Deterministic given the JSON; ``stabilise: false``
  restores the score-gate-plus-carry-forward behaviour.

Schema conversion
-----------------
See :func:`broadtrack_cp_to_sncalib`. Validated numerically against an independent
reimplementation of the C++ projection model driven only by raw JSON values:
max projection disagreement ~3e-5 px, plane-unprojection roundtrip < 1 mm across
zoomed/wide/rolled configurations; omitting the distortion rescale diverges by
>1000 px (negative control). Run ``scripts/verify_broadtrack_conversion.py`` on real
output before benchmarking.

No Docker: build the binary natively with ``scripts/setup_broadtrack.sh``. Nothing from
EVS is vendored in this repo (licence: noncommercial research, no redistribution).
"""
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from sn_calibration_baseline.camera import Camera
from tracklab.pipeline.videolevel_module import VideoLevelModule

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Schema conversion : BroadTrack "cp" object  ->  sn-calibration camera parameters
# --------------------------------------------------------------------------------------
def broadtrack_cp_to_sncalib(cp: dict) -> dict:
    """Convert one BroadTrack ``cp`` object to the sn-calibration parameter dict.

    BroadTrack (``Camera::toJSONString``) emits::

        sensorResolutionWidthPixels, sensorResolutionHeightPixels,
        horizontalFieldOfViewDegrees, panDegrees, tiltDegrees, rollDegrees,
        positionXMeters, positionYMeters, positionZMeters,
        normalizedRadialDistortionCoefficients

    ``sn_calibration_baseline.Camera.from_json_parameters`` requires::

        principal_point, x_focal_length, y_focal_length,
        pan_degrees, tilt_degrees, roll_degrees, position_meters,
        radial_distortion (6), tangential_distortion (2), thin_prism_distortion (4)

    Mapping rationale (verified against the C++ and BroadTrack's own
    ``scripts/compute_tripod.py``):

    * **Angles / position** are identical conventions: ``compute_tripod.py`` parses this
      JSON with the same ``pan_tilt_roll_to_orientation`` + transpose used by
      ``sn_calibration_baseline``. No sign flips, no reordering.
    * **Focal length**: ``getHorizontalFieldOfView() = 2*atan2(W/2, f)`` so
      ``f = (W/2) / tan(hFoV/2)``; square pixels => ``fx = fy``; principal point at the
      image centre (the paper reduces the intrinsics to a single ``f``).
    * **Radial distortion**: ``Camera::distort`` applies ``1 + sum k_i r^(2i)`` with
      ``r`` on the normalized image plane (OpenCV / sn-calibration convention), but
      ``getNormalizedRadialDistortion()`` exports
      ``k_i_json = k_i_internal * (H/f)^(2(i+1))`` (radius rescaled by image *height*).
      Inverted here: ``k_i_sncalib = k_i_json * (f/H)^(2(i+1))``. Only ``k1`` is
      modelled by BroadTrack; the remaining slots stay zero. This scaling is covered by
      a machine-precision test in ``scripts/verify_broadtrack_conversion.py``.
    """
    w = float(cp["sensorResolutionWidthPixels"])
    h = float(cp["sensorResolutionHeightPixels"])
    hfov = float(cp["horizontalFieldOfViewDegrees"]) * np.pi / 180.0
    focal = (w / 2.0) / np.tan(hfov / 2.0)

    radial = [0.0] * 6
    for i, k_json in enumerate(cp.get("normalizedRadialDistortionCoefficients", []) or []):
        if i >= 3:  # sn-calibration numerator terms are k1..k3 (slots 3..5 = k4..k6)
            break
        radial[i] = float(k_json) * (focal / h) ** (2.0 * (i + 1))

    return {
        "principal_point": [w / 2.0, h / 2.0],
        "x_focal_length": focal,
        "y_focal_length": focal,
        "pan_degrees": float(cp["panDegrees"]),
        "tilt_degrees": float(cp["tiltDegrees"]),
        "roll_degrees": float(cp["rollDegrees"]),
        "position_meters": [
            float(cp["positionXMeters"]),
            float(cp["positionYMeters"]),
            float(cp["positionZMeters"]),
        ],
        "radial_distortion": radial,
        "tangential_distortion": [0.0, 0.0],
        "thin_prism_distortion": [0.0, 0.0, 0.0, 0.0],
    }


def get_bbox_pitch(cam):
    """Bottom-left / bottom-right / bottom-middle unprojection onto the Z=0 plane.

    Identical to ``sn_gamestate.calibration.bbox2pitch.get_bbox_pitch`` so the
    ``bbox_pitch`` schema and numerics match the rest of the pipeline exactly.
    ``unproject_point_on_planeZ0`` undistorts by default, so BroadTrack's k1 is honoured.
    """
    def _get_bbox(bbox_ltrb):
        l, t, r, b = bbox_ltrb
        bl = np.array([l, b, 1])
        br = np.array([r, b, 1])
        bm = np.array([l + (r - l) / 2, b, 1])
        pbl_x, pbl_y, _ = cam.unproject_point_on_planeZ0(bl)
        pbr_x, pbr_y, _ = cam.unproject_point_on_planeZ0(br)
        pbm_x, pbm_y, _ = cam.unproject_point_on_planeZ0(bm)
        if np.any(np.isnan([pbl_x, pbl_y, pbr_x, pbr_y, pbm_x, pbm_y])):
            return None
        return {
            "x_bottom_left": pbl_x, "y_bottom_left": pbl_y,
            "x_bottom_right": pbr_x, "y_bottom_right": pbr_y,
            "x_bottom_middle": pbm_x, "y_bottom_middle": pbm_y,
        }
    return _get_bbox


# --------------------------------------------------------------------------------------
# Temporal stabilisation of the per-frame cameras (deterministic post-processing)
# --------------------------------------------------------------------------------------
# The binary tracks the camera frame to frame and re-initialises it from keypoints
# when the line-IoU score drops. Two artefacts reach pitch space as SUDDEN JUMPS of
# every player of a frame: (a) a re-initialisation lands on a camera that disagrees
# with the (drifted) frames before it, and (b) a rejected run (score < min_score,
# or the never-calibrated last frame) is served by carry-forward of a frozen camera
# and then snaps to the next accepted one. Both are visible as a discontinuity of
# the projected positions between two consecutive frames while the players
# themselves cannot have moved that far (25 fps: even a sprint is < 0.5 m/frame).
#
# Method (pure functions, no ground truth, deterministic given the JSON):
#   1. JUMP DETECTION on player continuity: for two consecutive frames that both
#      have an accepted camera, project the bottom-middle point of every detection
#      with its own frame's camera and take the median displacement over the
#      track_ids present in both frames (>= jump_min_tracks). Above max_jump_m the
#      pair is a jump and the lower-scoring frame (tie: the earlier, propagated one)
#      loses its anchor status.
#   2. ANCHORS are accepted frames with score >= anchor_score that lost no jump.
#   3. A run of consecutive non-anchor frames that contains at least one "bad"
#      frame (rejected, absent from the JSON, or demoted by a jump) is replaced by
#      LINEAR INTERPOLATION of the camera parameters between the anchors on either
#      side when the run is at most max_interp_frames long (angles on the shortest
#      arc). This removes the carry-then-snap pattern and the drift tail whose
#      scores decay towards a re-initialisation. Runs of merely weak but continuous
#      frames (no bad frame inside) keep the binary's cameras untouched.
#   4. Bad frames that cannot be bracketed by anchors within max_interp_frames are
#      interpolated between the nearest accepted frames instead, when those are
#      within range; whatever is still empty falls back to carry-forward
#      (use_prev_parameters / max_carry_frames), exactly as before.
#   5. The jump test is re-run on the FINAL camera sequence; residual jumps (both
#      sides confident, or too long to bridge) are reported, never hidden.
# stabilise: false reproduces the previous behaviour (score gate + carry-forward).


def _lerp_angle(a: float, b: float, f: float) -> float:
    """Linear interpolation of an angle in degrees along the shortest arc."""
    d = ((float(b) - float(a) + 180.0) % 360.0) - 180.0
    return float(a) + f * d


def interpolate_parameters(a: dict, b: dict, f: float) -> dict:
    """Element-wise linear interpolation of two sn-calibration parameter dicts at
    fraction ``f`` in [0, 1] (0 -> ``a``, 1 -> ``b``). Keys ending in ``_degrees``
    are interpolated on the shortest arc; lists element-wise; scalars linearly."""
    out = {}
    for k, va in a.items():
        vb = b[k]
        if isinstance(va, (list, tuple)):
            out[k] = [float(x) + f * (float(y) - float(x)) for x, y in zip(va, vb)]
        elif k.endswith("_degrees"):
            out[k] = _lerp_angle(va, vb, f)
        else:
            out[k] = float(va) + f * (float(vb) - float(va))
    return out


def pitch_jump(project, params_a, pts_a, ids_a, params_b, pts_b, ids_b, min_tracks):
    """Median pitch-plane displacement (metres) of the tracks shared by two frames.

    ``project(params, pts) -> (N, 2)`` maps image points to the pitch plane with the
    given camera. ``pts_*`` are the ``(N, 2)`` bottom-middle image points of the
    frame's detections and ``ids_*`` their track ids (NaN = untracked, ignored).
    Returns ``None`` when fewer than ``min_tracks`` shared tracks have a finite
    projection under both cameras -- the pair is then simply not checked."""
    if params_a is None or params_b is None:
        return None
    ids_a = np.asarray(ids_a, dtype=float).ravel()
    ids_b = np.asarray(ids_b, dtype=float).ravel()
    ok_a = np.flatnonzero(np.isfinite(ids_a))
    ok_b = np.flatnonzero(np.isfinite(ids_b))
    if len(ok_a) < min_tracks or len(ok_b) < min_tracks:
        return None
    _, ia, ib = np.intersect1d(ids_a[ok_a], ids_b[ok_b], return_indices=True)
    if len(ia) < min_tracks:
        return None
    sel_a = ok_a[ia]
    sel_b = ok_b[ib]
    pa = np.asarray(project(params_a, np.asarray(pts_a, dtype=float)[sel_a]), dtype=float)
    pb = np.asarray(project(params_b, np.asarray(pts_b, dtype=float)[sel_b]), dtype=float)
    d = np.linalg.norm(pa - pb, axis=1)
    d = d[np.isfinite(d)]
    if len(d) < min_tracks:
        return None
    return float(np.median(d))


def _fill_runs(final, source, fill_mask, bound_mask, max_len, tag):
    """Interpolate every maximal run of frames flagged in ``fill_mask`` between the
    nearest ``bound_mask`` frames on both sides, when the run is at most ``max_len``
    long and bracketed on both sides. Frames of a run that is filled are written in
    ``final`` (camera) and ``source`` (``tag``). Returns the list of filled runs as
    ``(first, last)`` index pairs."""
    n = len(final)
    runs = []
    i = 0
    while i < n:
        if not fill_mask[i]:
            i += 1
            continue
        j = i
        while j < n and fill_mask[j]:
            j += 1
        left = i - 1
        right = j
        if (left >= 0 and right < n and bound_mask[left] and bound_mask[right]
                and (j - i) <= max_len and final[left] is not None
                and final[right] is not None):
            gap = float(right - left)
            for k in range(i, j):
                final[k] = interpolate_parameters(final[left], final[right],
                                                  (k - left) / gap)
                source[k] = tag
            runs.append((i, j - 1))
        i = j
    return runs


def stabilise_sequence(params, scores, pts, ids, project, *, min_score, anchor_score,
                       max_jump_m, jump_min_tracks, max_interp_frames,
                       use_prev_parameters, max_carry_frames, stabilise=True):
    """Turn the binary's per-frame output into a jump-free camera sequence.

    Inputs are aligned lists in TEMPORAL order: ``params[i]`` the converted camera
    of frame ``i`` (``None`` when absent from the JSON), ``scores[i]`` its line-IoU
    score (``None`` when absent), ``pts[i]`` the ``(N_i, 2)`` bottom-middle image
    points of the frame's detections, ``ids[i]`` their track ids. ``project`` as in
    :func:`pitch_jump`.

    Returns ``(final, source, report)``: ``final[i]`` the camera to use (``None`` =
    no parameters), ``source[i]`` one of ``binary`` / ``interp`` / ``interp_weak`` /
    ``carry`` / ``none``, and ``report`` the per-frame decisions and the jump
    statistics before and after (``jumps_raw`` / ``jumps_final`` as
    ``[frame_index, displacement_m]`` pairs, ``frame_index`` being the later frame
    of the pair).

    With ``stabilise=False`` steps 1-4 are skipped and the result is the previous
    behaviour exactly: accepted frames keep the binary's camera, every other frame
    is carried forward (subject to ``max_carry_frames``).
    """
    n = len(params)
    scores_f = [float(s) if s is not None else None for s in scores]
    accepted = [params[i] is not None and scores_f[i] is not None
                and scores_f[i] >= min_score for i in range(n)]
    final = [params[i] if accepted[i] else None for i in range(n)]
    source = ["binary" if accepted[i] else "none" for i in range(n)]
    demoted = [False] * n
    jumps_raw, interp_runs, weak_runs = [], [], []

    def _jump(i, cams):
        return pitch_jump(project, cams[i - 1], pts[i - 1], ids[i - 1],
                          cams[i], pts[i], ids[i], jump_min_tracks)

    if stabilise:
        # 1. player-continuity jump detection between consecutive accepted frames
        for i in range(1, n):
            if not (accepted[i - 1] and accepted[i]):
                continue
            d = _jump(i, params)
            if d is not None and d > max_jump_m:
                jumps_raw.append([i, d])
                loser = i - 1 if scores_f[i - 1] <= scores_f[i] else i
                demoted[loser] = True

        # 2. anchors
        anchor = [accepted[i] and not demoted[i] and scores_f[i] >= anchor_score
                  for i in range(n)]
        bad = [(not accepted[i]) or demoted[i] for i in range(n)]

        # 3. non-anchor runs containing a bad frame -> interpolate between anchors
        non_anchor = [not a for a in anchor]
        # only runs that contain at least one bad frame are candidates
        fill = [False] * n
        i = 0
        while i < n:
            if not non_anchor[i]:
                i += 1
                continue
            j = i
            while j < n and non_anchor[j]:
                j += 1
            if any(bad[k] for k in range(i, j)):
                for k in range(i, j):
                    fill[k] = True
            i = j
        interp_runs = _fill_runs(final, source, fill, anchor, max_interp_frames, "interp")

        # 4. bad frames still empty -> interpolate between the nearest accepted frames
        still_bad = [bad[i] and source[i] != "interp" for i in range(n)]
        good = [accepted[i] and not demoted[i] for i in range(n)]
        weak_runs = _fill_runs(final, source, still_bad, good, max_interp_frames,
                               "interp_weak")
        # a demoted frame that could not be bridged keeps the binary's camera
        # (better than freezing the previous one for a frame the binary did score)
        for i in range(n):
            if demoted[i] and source[i] == "none":
                final[i] = params[i]
                source[i] = "binary"

    # carry-forward for whatever is still empty (previous behaviour)
    last, carry_run = None, 0
    for i in range(n):
        if final[i] is not None:
            last, carry_run = final[i], 0
            continue
        if (use_prev_parameters and last is not None
                and (max_carry_frames == 0 or carry_run < max_carry_frames)):
            final[i] = last
            source[i] = "carry"
            carry_run += 1

    # 5. residual jumps on the final sequence
    jumps_final = []
    if stabilise:
        for i in range(1, n):
            if final[i - 1] is None or final[i] is None:
                continue
            d = _jump(i, final)
            if d is not None and d > max_jump_m:
                jumps_final.append([i, d])

    report = {
        "stabilise": bool(stabilise),
        "frames": n,
        "accepted": int(sum(accepted)),
        "demoted": [i for i in range(n) if demoted[i]],
        "jumps_raw": jumps_raw,
        "jumps_final": jumps_final,
        "interp_runs": interp_runs,
        "interp_weak_runs": weak_runs,
        "sources": {tag: int(sum(s == tag for s in source))
                    for tag in ("binary", "interp", "interp_weak", "carry", "none")},
    }
    return final, source, report


# --------------------------------------------------------------------------------------
# Module
# --------------------------------------------------------------------------------------
class BroadTrackCalibration(VideoLevelModule):
    """Camera calibration + image-to-pitch projection driven by the BroadTrack binary."""

    # Dict form so image-level "parameters" is declared (Pipeline.validate and the
    # TrackerState load/save column bookkeeping both use get_*_columns, which handles
    # dicts; the deprecated validate_input/validate_output are never called in 1.3.24).
    input_columns = {"detection": ["bbox_ltwh", "image_id"], "image": []}
    output_columns = {"detection": ["bbox_pitch"], "image": ["parameters"]}

    def __init__(self, cfg, device, tracking_dataset=None):
        self.cfg = cfg
        self.device = device

        self.binary = Path(str(cfg.binary))
        self.kp_model = Path(str(cfg.keypoint_model))
        self.line_model = Path(str(cfg.line_model))
        self.libtorch_lib = str(getattr(cfg, "libtorch_lib", "") or "")

        self.calib_dir = Path(str(cfg.calib_dir))
        self.calib_dir.mkdir(parents=True, exist_ok=True)
        self.use_cached_json = bool(getattr(cfg, "use_cached_json", True))
        self.timeout = int(getattr(cfg, "timeout", 7200))
        # Best-of-N draw selection (the binary is nondeterministic across runs):
        # on a cache miss run it calib_attempts times and keep the attempt whose
        # accepted frames (score >= min_score) have the highest mean line-IoU
        # score, tiebreak on the accepted-frame count. 1 = the previous
        # single-run behaviour, byte for byte. Selection uses only the binary's
        # own per-frame scores -- no ground truth -- so it is valid on test.
        self.calib_attempts = int(getattr(cfg, "calib_attempts", 1) or 1)
        if self.calib_attempts < 1:
            raise ValueError(f"[BroadTrack] calib_attempts must be >= 1, "
                             f"got {self.calib_attempts}")

        self.prior_xyz = [float(v) for v in getattr(cfg, "prior_xyz", [0.0, 55.0, -12.0])]
        self.tripod_mode = str(getattr(cfg, "tripod_mode", "none"))  # none|per_sequence|per_game
        self.tripod_dir = Path(str(getattr(cfg, "tripod_dir", self.calib_dir / "tripod")))

        self.write_human_bboxes = bool(getattr(cfg, "write_human_bboxes", True))
        self.staging_dir = str(getattr(cfg, "staging_dir", "") or "")
        # Stage even when the frames dir IS writable. main.cpp hardcodes reading
        # player masks from <frames_dir>/human-bboxes/, so writing them in place
        # leaves a SUBDIRECTORY inside img1/. That is harmless for SoccerNet
        # sequences (the loader takes nframes from Labels-GameState.json's
        # seq_length) but breaks CUSTOM/UNLABELED video: with no labels file the
        # loader falls back to len(os.listdir(img1)), which counts the extra
        # directory and then requests a frame index one past the end ->
        # cv2.error: (-215:Assertion failed) !_src.empty(). Set true whenever
        # running inference on video that has no Labels-GameState.json.
        self.always_stage = bool(getattr(cfg, "always_stage", False))
        # Frames whose binary score is below min_score are treated as uncalibrated
        # (see the module docstring: 0.3 is the binary's own tracking-lost threshold).
        self.min_score = float(getattr(cfg, "min_score", 0.3))
        self.use_prev_parameters = bool(getattr(cfg, "use_prev_parameters", True))
        # Longest run of consecutive frames allowed to reuse the last accepted camera;
        # 0 = unlimited. Beyond the cap the frame gets no parameters and its
        # detections no bbox_pitch (same outcome as an uncalibrated NBJW frame).
        self.max_carry_frames = int(getattr(cfg, "max_carry_frames", 0) or 0)
        if self.max_carry_frames < 0:
            raise ValueError(f"[BroadTrack] max_carry_frames must be >= 0, got {self.max_carry_frames}")
        # Temporal stabilisation (see stabilise_sequence): jump detection on player
        # continuity + interpolation of the camera across rejected / demoted runs.
        self.stabilise = bool(getattr(cfg, "stabilise", True))
        self.anchor_score = float(getattr(cfg, "anchor_score", 0.5))
        self.max_jump_m = float(getattr(cfg, "max_jump_m", 2.0))
        self.jump_min_tracks = int(getattr(cfg, "jump_min_tracks", 3) or 1)
        self.max_interp_frames = int(getattr(cfg, "max_interp_frames", 50) or 0)
        if self.anchor_score < self.min_score:
            raise ValueError(f"[BroadTrack] anchor_score ({self.anchor_score}) must be >= "
                             f"min_score ({self.min_score})")
        if self.max_jump_m <= 0 or self.max_interp_frames < 0 or self.jump_min_tracks < 1:
            raise ValueError("[BroadTrack] max_jump_m must be > 0, max_interp_frames >= 0 "
                             "and jump_min_tracks >= 1")

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _frames_dir(metadatas: pd.DataFrame) -> Path:
        """Folder holding the %06d.jpg frames of this video (SNGS ``.../img1``)."""
        return Path(str(metadatas["file_path"].iloc[0])).parent

    @staticmethod
    def _sequence_name(frames_dir: Path) -> str:
        """`.../SNGS-116/img1` -> `SNGS-116` (falls back to the folder name)."""
        return frames_dir.parent.name or frames_dir.name

    def _prepare_frames_dir(self, frames_dir: Path, detections: pd.DataFrame,
                            metadatas: pd.DataFrame) -> Path:
        """Return the directory to pass to ``-f``, writing human-bboxes when possible.

        The binary reads masks from ``<frames_dir>/human-bboxes/%06d.json`` (hardcoded).
        If the dataset is mounted read-only (Kaggle), a directory of symlinks is staged
        so the masks can still be written; symlinks count as regular files, so the
        binary's ``frames_number`` stays correct. The ``human-bboxes`` *subdirectory*
        itself is not a regular file, so it does not shift ``frames_number`` either.
        """
        if not self.write_human_bboxes:
            return frames_dir

        target = frames_dir
        if self.always_stage or not os.access(frames_dir, os.W_OK):
            if not self.staging_dir:
                log.warning(
                    "[BroadTrack] frames dir is read-only and no staging_dir configured; "
                    "running without player masking."
                )
                return frames_dir
            target = Path(self.staging_dir) / self._sequence_name(frames_dir)
            target.mkdir(parents=True, exist_ok=True)
            for src in sorted(frames_dir.glob("*.jpg")):
                dst = target / src.name
                if not dst.exists():
                    try:
                        dst.symlink_to(src)
                    except OSError:  # no symlink permission -> copy
                        shutil.copy2(src, dst)

        try:
            bbox_dir = target / "human-bboxes"
            bbox_dir.mkdir(parents=True, exist_ok=True)
            # image_id -> frame stem, via the frame file name (%06d.jpg).
            id2stem = {idx: Path(str(p)).stem for idx, p in metadatas["file_path"].items()}
            for image_id, group in detections.groupby("image_id"):
                stem = id2stem.get(image_id)
                if stem is None:
                    continue
                boxes = []
                for ltwh in group["bbox_ltwh"]:
                    l, t, w, h = [float(v) for v in ltwh]
                    boxes.append([l, t, l + w, t + h])  # binary expects [x1, y1, x2, y2]
                (bbox_dir / f"{stem}.json").write_text(json.dumps({"bboxes": boxes}))
        except OSError as e:
            log.warning(f"[BroadTrack] could not write human-bboxes ({e}); "
                        f"running without player masking.")
        return target

    def _run_binary(self, frames_dir: Path, out_json: Path, tripod_file=None) -> bool:
        cmd = [
            str(self.binary),
            # Boost.ProgramOptions declares these as LONG option names ('f', 'o',
            # 'k', 'l', 't'); only 'help' has a short form ("help,h"). Single-dash
            # '-f' is therefore rejected with "unrecognised option '-f'".
            "--f", str(frames_dir),
            "--o", str(out_json),
            "--k", str(self.kp_model),
            "--l", str(self.line_model),
            "--X", str(self.prior_xyz[0]),
            "--Y", str(self.prior_xyz[1]),
            "--Z", str(self.prior_xyz[2]),
        ]
        if tripod_file is not None and Path(tripod_file).is_file():
            cmd += ["--t", str(tripod_file)]

        env = os.environ.copy()
        if self.libtorch_lib:
            env["LD_LIBRARY_PATH"] = f"{self.libtorch_lib}:{env.get('LD_LIBRARY_PATH', '')}"

        log.info(f"[BroadTrack] running: {' '.join(cmd)}")
        try:
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                                  timeout=self.timeout)
        except subprocess.TimeoutExpired:
            log.error(f"[BroadTrack] timed out after {self.timeout}s on {frames_dir}")
            return False
        except OSError as e:
            log.error(f"[BroadTrack] could not execute '{self.binary}': {e}")
            return False
        if proc.returncode != 0:
            log.error(
                f"[BroadTrack] binary failed (rc={proc.returncode}).\n"
                f"stdout tail: {proc.stdout[-2000:]}\nstderr tail: {proc.stderr[-2000:]}"
            )
            return False
        return out_json.is_file()

    def _attempt_quality(self, candidate_json: Path):
        """Quality of one calibration attempt, from the binary's OWN per-frame
        line-IoU scores: ``(mean score over accepted frames, accepted-frame
        count, total frames)`` with accepted = ``score >= min_score``. An
        attempt with no accepted frame scores ``(0.0, 0, total)``. ``None``
        when the JSON cannot be read or parsed (the attempt is then treated
        as failed). No ground truth is involved."""
        try:
            data = json.loads(candidate_json.read_text())
            scores = [float(v.get("score", 0.0)) for v in data.values()]
        except (OSError, json.JSONDecodeError, AttributeError, TypeError, ValueError):
            return None
        accepted = [x for x in scores if x >= self.min_score]
        mean_acc = float(np.mean(accepted)) if accepted else 0.0
        return mean_acc, len(accepted), len(scores)

    def _run_binary_best_of(self, frames_dir: Path, out_json: Path,
                            tripod_file=None) -> bool:
        """Run the binary up to ``calib_attempts`` times and keep the best draw.

        The binary is nondeterministic across runs (same frames, different
        homography); the draw feeds straight into pitch-space localisation, so
        the low tail of the draw distribution costs several GS-HOTA points.
        Each attempt writes its own candidate JSON; quality is
        ``_attempt_quality`` (mean accepted line-IoU, tiebreak accepted-frame
        count -- the binary's own confidence signal, valid on test). The best
        candidate becomes ``out_json``, the losers are deleted, and a
        ``<seq>.selection.json`` sidecar records every attempt's statistics
        and the winner. ``calib_attempts == 1`` reproduces the previous
        single-run behaviour byte for byte (direct call, no sidecar). A failed
        attempt is tolerated; if every attempt fails, the stage reports
        failure exactly as before."""
        n = self.calib_attempts
        if n == 1:
            return self._run_binary(frames_dir, out_json, tripod_file)

        stem = out_json.stem
        cand_path = lambda i: out_json.parent / f"{stem}.attempt{i}.json"
        attempts, best = [], None            # best = ((mean, n_acc), i, path)
        for i in range(1, n + 1):
            cand = cand_path(i)
            cand.unlink(missing_ok=True)
            ok = self._run_binary(frames_dir, cand, tripod_file)
            stat = {"attempt": i, "ok": bool(ok)}
            if ok:
                q = self._attempt_quality(cand)
                if q is None:
                    stat["ok"] = False
                    stat["error"] = "output JSON unreadable"
                else:
                    mean_acc, n_acc, n_frames = q
                    stat.update(mean_accepted_score=mean_acc,
                                accepted_frames=n_acc, frames=n_frames)
                    key = (mean_acc, n_acc)
                    if best is None or key > best[0]:
                        best = (key, i, cand)
            attempts.append(stat)
            log.info(f"[BroadTrack] attempt {i}/{n}: "
                     + (f"mean accepted score {stat.get('mean_accepted_score'):.3f} "
                        f"over {stat.get('accepted_frames')} frame(s)"
                        if stat["ok"] else "FAILED"))

        if best is None:                     # every attempt failed
            for i in range(1, n + 1):
                cand_path(i).unlink(missing_ok=True)
            return False

        (_, w, wpath) = best
        out_json.unlink(missing_ok=True)
        wpath.replace(out_json)
        for i in range(1, n + 1):
            cand_path(i).unlink(missing_ok=True)
        selection = {"sequence": stem, "calib_attempts": n,
                     "min_score": self.min_score,
                     "attempts": attempts, "winner": w}
        try:
            (out_json.parent / f"{stem}.selection.json").write_text(
                json.dumps(selection, indent=2))
        except OSError as e:                 # never fail the run over telemetry
            log.warning(f"[BroadTrack] could not write the selection sidecar: {e}")
        log.info(f"[BroadTrack] best-of-{n}: kept attempt {w} "
                 f"(mean accepted score {best[0][0]:.3f}, "
                 f"{best[0][1]} accepted frame(s))")
        return True

    def _tripod_file(self, seq_name: str, frames_dir: Path):
        """Two-pass tripod estimation (free run -> compute_tripod.py -> soft run).

        ``per_game`` shares one tripod file between every clip mapped to the same game
        key via ``cfg.game_of`` (the file is estimated from the first clip of that game
        that gets processed, then reused). SNGS clip names do not encode the game, so
        without a ``game_of`` mapping this degrades to per-sequence behaviour.
        """
        if self.tripod_mode == "none":
            return None
        self.tripod_dir.mkdir(parents=True, exist_ok=True)
        key = seq_name
        if self.tripod_mode == "per_game":
            mapping = dict(getattr(self.cfg, "game_of", {}) or {})
            key = mapping.get(seq_name, seq_name)
        tripod = self.tripod_dir / f"{key}.tripod.json"
        if tripod.is_file():
            return tripod

        free_json = self.calib_dir / f"{seq_name}.free.json"
        if not free_json.is_file():
            if not self._run_binary(frames_dir, free_json, tripod_file=None):
                return None
        script = Path(str(getattr(self.cfg, "compute_tripod_script", "")))
        if not script.is_file():
            log.warning(f"[BroadTrack] compute_tripod.py not found at '{script}'; "
                        f"continuing in free mode.")
            return None
        try:
            proc = subprocess.run(
                ["python", str(script), "-i", str(free_json), "-o", str(tripod)],
                capture_output=True, text=True, timeout=600,
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            log.warning(f"[BroadTrack] compute_tripod.py failed ({e}); "
                        f"continuing in free mode.")
            return None
        if proc.returncode != 0 or not tripod.is_file():
            log.warning(f"[BroadTrack] compute_tripod.py failed; continuing in free "
                        f"mode. stderr: {proc.stderr[-1000:]}")
            return None
        return tripod

    # ---------------------------------------------------------------- main
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        # NOTE (engine contract): must return a DataFrame (detections) only; the image
        # 'parameters' column is written on `metadatas` in place, which the offline
        # engine persists (image_pred is the same object across the video loop).
        if len(metadatas) == 0:
            return detections

        frames_dir = self._frames_dir(metadatas)
        seq_name = self._sequence_name(frames_dir)
        out_json = self.calib_dir / f"{seq_name}.json"

        calibs = None
        if self.use_cached_json and out_json.is_file():
            log.info(f"[BroadTrack] using cached calibration: {out_json}")
        else:
            if not self.binary.is_file():
                log.error(
                    f"[BroadTrack] binary not found at '{self.binary}'. Build it first "
                    f"with scripts/setup_broadtrack.sh. Skipping {seq_name}."
                )
                return self._empty_outputs(detections, metadatas)
            run_dir = self._prepare_frames_dir(frames_dir, detections, metadatas)
            tripod = self._tripod_file(seq_name, run_dir)
            if not self._run_binary_best_of(run_dir, out_json, tripod):
                return self._empty_outputs(detections, metadatas)

        try:
            calibs = json.loads(out_json.read_text())
        except (OSError, json.JSONDecodeError) as e:
            log.error(f"[BroadTrack] could not read '{out_json}': {e}")
            return self._empty_outputs(detections, metadatas)

        # Keys are full path strings built from -f -> index by basename.
        by_name = {Path(k).name: v for k, v in calibs.items()}
        log.info(f"[BroadTrack] {seq_name}: {len(by_name)} calibrated frames "
                 f"for {len(metadatas)} images")

        # Temporal order matters for carry-forward and interpolation: iterate frames
        # sorted by file name (%06d.jpg sorts chronologically); do not rely on the
        # incoming row order.
        order = metadatas["file_path"].astype(str).map(lambda p: Path(p).name)
        frame_ids = list(order.sort_values().index)
        frame_names = [order[i] for i in frame_ids]

        # per-frame binary output
        raw_params, raw_scores = [], []
        n_lowscore = n_reinit = n_missing = 0
        for name in frame_names:
            entry = by_name.get(name)
            if entry is None:
                n_missing += 1
                raw_params.append(None)
                raw_scores.append(None)
                continue
            score = float(entry.get("score", 0.0))
            n_reinit += int(bool(entry.get("reinit", False)))
            n_lowscore += int(score < self.min_score)
            try:
                raw_params.append(broadtrack_cp_to_sncalib(entry["cp"]))
            except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
                log.warning(f"[BroadTrack] {seq_name}/{name}: unusable cp ({e}); "
                            f"treated as uncalibrated")
                raw_params.append(None)
            raw_scores.append(score)

        # per-frame detections: bottom-middle image points + track ids, for the
        # player-continuity jump test (no track_id column -> every id NaN -> the
        # test is skipped and only the score gate + interpolation of rejected
        # runs remain active)
        has_tracks = "track_id" in detections.columns
        if self.stabilise and not has_tracks:
            log.warning(f"[BroadTrack] {seq_name}: no track_id column; jump detection "
                        f"on player continuity is disabled for this sequence")
        dets_by_image = {k: g for k, g in detections.groupby("image_id")}
        pts, ids = [], []
        for image_id in frame_ids:
            g = dets_by_image.get(image_id)
            if g is None or len(g) == 0:
                pts.append(np.zeros((0, 2)))
                ids.append(np.zeros((0,)))
                continue
            ltwh = np.asarray([[float(v) for v in b] for b in g["bbox_ltwh"]], dtype=float)
            pts.append(np.column_stack([ltwh[:, 0] + ltwh[:, 2] / 2.0,
                                        ltwh[:, 1] + ltwh[:, 3]]))
            ids.append(pd.to_numeric(g["track_id"], errors="coerce").to_numpy(dtype=float)
                       if has_tracks else np.full(len(g), np.nan))

        final, source, report = stabilise_sequence(
            raw_params, raw_scores, pts, ids, self._project_to_pitch,
            min_score=self.min_score, anchor_score=self.anchor_score,
            max_jump_m=self.max_jump_m, jump_min_tracks=self.jump_min_tracks,
            max_interp_frames=self.max_interp_frames,
            use_prev_parameters=self.use_prev_parameters,
            max_carry_frames=self.max_carry_frames, stabilise=self.stabilise)

        params_rows, bbox_pitch = {}, {}
        for image_id, params in zip(frame_ids, final):
            if params is None:
                params_rows[image_id] = {}
                continue
            params_rows[image_id] = params
            cam = Camera()
            cam.from_json_parameters(params)
            image_dets = detections[detections.image_id == image_id]
            if len(image_dets):
                projected = image_dets.bbox.ltrb().apply(get_bbox_pitch(cam))
                bbox_pitch.update(projected.to_dict())

        # logs + sidecar
        scores_arr = np.asarray([s for s in raw_scores if s is not None], dtype=float)
        if len(scores_arr):
            log.info(f"[BroadTrack] {seq_name}: score min/median/max "
                     f"{scores_arr.min():.3f}/{np.median(scores_arr):.3f}/"
                     f"{scores_arr.max():.3f}, {n_reinit} reinit frame(s)")
        src = report["sources"]
        n_dropped = src["none"]
        msg = (f"[BroadTrack] {seq_name}: {n_lowscore} frame(s) below "
               f"min_score={self.min_score}, {n_missing} absent from the JSON, "
               f"{src['interp'] + src['interp_weak']} interpolated, "
               f"{src['carry']} carried-forward, {n_dropped} left without parameters")
        (log.warning if n_dropped else log.info)(msg)
        if self.stabilise:
            n_raw, n_fin = len(report["jumps_raw"]), len(report["jumps_final"])
            msg = (f"[BroadTrack] {seq_name}: {n_raw} jump(s) > {self.max_jump_m} m in the "
                   f"binary's cameras, {n_fin} residual after stabilisation "
                   f"({len(report['demoted'])} frame(s) demoted, "
                   f"{len(report['interp_runs'])} run(s) interpolated between anchors, "
                   f"{len(report['interp_weak_runs'])} between nearest accepted frames)")
            (log.warning if n_fin else log.info)(msg)
        self._write_stabilise_sidecar(seq_name, frame_names, raw_scores, source, report)

        detections = detections.copy()
        col = pd.Series(bbox_pitch, dtype=object).reindex(detections.index)
        detections["bbox_pitch"] = col.where(col.notna(), None)  # NaN -> None (NBJW parity)

        params_col = pd.Series(params_rows, dtype=object).reindex(metadatas.index)
        metadatas["parameters"] = params_col.apply(
            lambda v: v if isinstance(v, dict) else {}
        )
        return detections

    @staticmethod
    def _project_to_pitch(params: dict, pts: np.ndarray) -> np.ndarray:
        """Image points ``(N, 2)`` -> pitch plane ``(N, 2)`` through the given camera
        (same ``unproject_point_on_planeZ0`` path as ``get_bbox_pitch``; NaN where
        the ray misses the plane)."""
        cam = Camera()
        cam.from_json_parameters(params)
        out = np.full((len(pts), 2), np.nan, dtype=float)
        for i, (x, y) in enumerate(np.asarray(pts, dtype=float)):
            px, py, _ = cam.unproject_point_on_planeZ0(np.array([x, y, 1.0]))
            out[i] = (px, py)
        return out

    def _write_stabilise_sidecar(self, seq_name, frame_names, raw_scores, source, report):
        """``<calib_dir>/<seq>.stabilise.json``: settings, jump statistics before and
        after, and the per-frame decision (score, source). Telemetry only."""
        payload = {
            "sequence": seq_name,
            "settings": {
                "stabilise": self.stabilise, "min_score": self.min_score,
                "anchor_score": self.anchor_score, "max_jump_m": self.max_jump_m,
                "jump_min_tracks": self.jump_min_tracks,
                "max_interp_frames": self.max_interp_frames,
                "use_prev_parameters": self.use_prev_parameters,
                "max_carry_frames": self.max_carry_frames,
            },
            "summary": {k: v for k, v in report.items()
                        if k in ("frames", "accepted", "sources", "demoted",
                                 "interp_runs", "interp_weak_runs")},
            "jumps_raw": [{"frame": frame_names[i], "displacement_m": d}
                          for i, d in report["jumps_raw"]],
            "jumps_final": [{"frame": frame_names[i], "displacement_m": d}
                            for i, d in report["jumps_final"]],
            "frames": [{"frame": n, "score": s, "source": src}
                       for n, s, src in zip(frame_names, raw_scores, source)],
        }
        try:
            (self.calib_dir / f"{seq_name}.stabilise.json").write_text(
                json.dumps(payload, indent=1))
        except OSError as e:
            log.warning(f"[BroadTrack] could not write the stabilise sidecar: {e}")

    @staticmethod
    def _empty_outputs(detections: pd.DataFrame, metadatas: pd.DataFrame):
        detections = detections.copy()
        detections["bbox_pitch"] = None
        metadatas["parameters"] = pd.Series(
            [{} for _ in range(len(metadatas))], index=metadatas.index, dtype=object
        )
        return detections
