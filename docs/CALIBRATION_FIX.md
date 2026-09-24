# Calibration fix — BroadTrack draw variance

## Problem

The camera calibration is produced by the native BroadTrack binary. It is
nondeterministic across runs: the same sequence, with byte-identical tracking
intermediates, yields a different homography each session. Because GS-HOTA is
scored in pitch space (positions are projected through the homography and matched
with a distance tolerance), the calibration draw feeds straight into the
localisation and detection components of the metric.

Measured effect on SNGS-116, across four runs with identical tracking: GS-HOTA
62.4 / 55.7 / 62.1 / 53.1, driven only by the calibration draw (the binary's own
minimum accepted line-IoU score for those runs was 0.494 / 0.370 / high / 0.301).
The spread is about nine GS-HOTA points and is attributable to calibration alone.

## Fix

Two independent layers.

### 1. Best-of-N draw selection (`sn_gamestate/calibration/broadtrack_api.py`)

On a cache miss the stage runs the binary `calib_attempts` times (default 3) and
keeps the attempt with the highest quality, where quality is the mean of the
binary's own per-frame line-IoU scores over accepted frames (score
>= `min_score`), with the accepted-frame count as a tiebreak. This is the
binary's own confidence signal; no ground truth is used, so the selection is
valid on the test split. The losing attempts are deleted and a
`<seq>.selection.json` sidecar records each attempt's statistics and the winner.

`calib_attempts: 1` reproduces the previous single-run behaviour byte for byte.

This is inference-time sample selection, not tuning: no parameter is fitted to
labels, and the same selection would be made on an unlabelled sequence. It cuts
off the low tail of the draw distribution.

Method (`_run_binary_best_of`, `_attempt_quality`):

- run the binary up to `calib_attempts` times, each to its own candidate JSON;
- for each candidate, read its per-frame scores; quality =
  `mean(scores >= min_score)`, tiebreak = number of accepted frames;
- keep the best candidate as `<seq>.json`, delete the rest;
- write `<seq>.selection.json` (per-attempt stats + winner);
- a failed attempt is tolerated; if every attempt fails, the stage reports
  failure as before.

### 2. Freeze (`docs/kaggle_one_sequence_test.ipynb`)

A `CALIB_DATASET` variable mounts a Kaggle dataset of `broadtrack_calib/*.json`.
When set, the run cell copies those files into place before the pipeline runs and
`use_cached_json` short-circuits the binary entirely, so the calibration is
bit-identical across sessions. When empty, the stage computes the calibration
(best-of-N above).

Workflow: one committed best-of-N run generates and persists a good calibration;
upload it as a dataset; set `CALIB_DATASET`. Layer 1 mitigates the variance,
layer 2 removes it.

## Configuration

`configs/modules/calibration/broadtrack.yaml`:

- `calib_attempts: 3`
- `use_cached_json: true`

## Verification

Offline, a four-case harness with a stubbed binary:

- best-of-3 keeps the highest mean-accepted attempt, deletes losers, writes an
  exact selection sidecar;
- `calib_attempts: 1` makes a single call and writes no sidecar;
- a failed attempt is tolerated and the best survivor wins;
- all attempts failing returns failure.

The installed `broadtrack_api.py` is byte-identical to the harness-verified copy;
the notebook parses and its modified cells pass shell/py-compile checks.

## Result on SNGS-116 (first run under the fix)

`CALIB_DATASET` empty (compute path). Best-of-3 attempts scored 0.594 / 0.590 /
0.613; attempt 3 (0.613) was kept and persisted as a freeze candidate. Audit
13 PASS / 0 WARN / 0 FAIL, run integrity clean. GS-HOTA 61.031 (EVAL_SPACE=pitch;
LocA 53.85) — a good draw, versus the 53.1 obtained earlier from a 0.301 draw on
the same tracking.

## Detector A/B note

When comparing two detector variants in one session, the calibration cache
(`broadtrack_calib/`) is reused across both runs so the calibration is held
constant; otherwise each run would redraw a different calibration and the
~9-point variance would mask the detector difference.

## Temporal stabilisation — sudden jumps of the projected positions

### Problem

Independently of the draw variance above, a single draw is not temporally
stable. The binary tracks the camera frame to frame and, when its line-IoU score
drops, re-initialises it from keypoints. Two things reach pitch space as a SUDDEN
JUMP of every player of a frame, followed by a snap back:

- a re-initialisation lands on a camera that disagrees with the drifted frames
  before it (the drift itself is gradual and keeps a score above `min_score`);
- a rejected run (`score < min_score`, or the last frame, which the binary
  never calibrates) was served by carry-forward of a frozen camera, which then
  snaps to the next accepted camera.

The score gate cannot see either: the jump is between two frames the binary
scored acceptably, or between a carried frame and an accepted one.

### Fix (`stabilise_sequence`, `sn_gamestate/calibration/broadtrack_api.py`)

Deterministic post-processing of the per-frame cameras, no ground truth:

1. Jump detection on player continuity, independent of the camera: for two
   consecutive frames that both have an accepted camera, the bottom-middle
   point of every detection is projected with its own frame's camera and the
   median displacement over the track ids present in both frames
   (`jump_min_tracks`) is compared with `max_jump_m`. Players cannot move that
   far in one frame (25 fps: a sprint is under 0.5 m/frame), so a larger
   displacement is a camera discontinuity. The lower-scoring frame of the pair
   (tie: the earlier, propagated one) loses its anchor status.
2. Anchors are accepted frames with `score >= anchor_score` that lost no jump.
3. A run of non-anchor frames that contains a rejected, absent or demoted frame
   is replaced by linear interpolation of the camera parameters (angles on the
   shortest arc) between the anchors on both sides, when the run is at most
   `max_interp_frames` long. This removes the carry-then-snap pattern and the
   drift tail whose scores decay towards a re-initialisation. Continuous runs
   of weak frames are never touched.
4. Rejected frames that no anchor pair can bridge are interpolated between the
   nearest accepted frames; whatever is left falls back to carry-forward
   (`use_prev_parameters`, `max_carry_frames`) as before.
5. The jump test is re-run on the final cameras. Residual jumps (both sides
   confident, or a gap too long to bridge) are reported, not hidden.

Every frame's source (`binary` / `interp` / `interp_weak` / `carry` / `none`),
the raw and residual jumps and the settings are written to
`<calib_dir>/<seq>.stabilise.json`. The stage logs a warning when residual
jumps remain. `stabilise: false` reproduces the previous behaviour exactly.

A sequence without `track_id` (calibration run before tracking, or untracked
detections) disables the jump test only; the interpolation of rejected runs
still applies.

### Configuration

`configs/modules/calibration/broadtrack.yaml`:

- `stabilise: true`
- `anchor_score: 0.5` (must be `>= min_score`)
- `max_jump_m: 2.0`
- `jump_min_tracks: 3`
- `max_interp_frames: 50` (2 s at 25 fps)

The thresholds are not tuned on labels: `max_jump_m` is a physical bound with
margin for projection noise between two slightly different cameras, and
`anchor_score` sits between the binary's lost threshold (0.3) and the threshold
upstream uses to build a tripod (0.6).

### Verification

`tests/test_broadtrack_stabilise.py` runs the installed source text against a
toy camera (a rigid pitch-plane shift, which is what a wrong homography does to
every player of a frame): carry-then-snap and drift tails are bridged with no
residual jump; a scored spike is demoted and bridged; a continuous weak run is
untouched; a jump between two confident frames is reported as residual; long
gaps and the last frame fall back to carry-forward; `stabilise: false` is the
previous behaviour; untracked detections disable the jump test only; angle
interpolation takes the shortest arc. The `process` path was exercised end to
end with a stubbed camera on a synthetic sequence (spike + lost run + absent
last frame): the spike is removed, the lost run is bridged, the sidecar and the
`bbox_pitch` / `parameters` columns are written with the expected shape.

Not yet measured: the effect on GS-HOTA on real sequences. The `.stabilise.json`
sidecar makes the before/after jump count of every run inspectable; compare a
run with `stabilise: false` against the default on the same frozen calibration
JSON (the cache holds the draw constant, so the difference is the stabilisation
alone).

## Residual (unverified)

The root cause of the binary's nondeterminism is not instrumented; the fix
removes its effect (selection + freeze), not its source. Best-of-3's realised
draw-quality gain over many live sessions is not yet characterised beyond the
runs above.
