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

## Residual (unverified)

The root cause of the binary's nondeterminism is not instrumented; the fix
removes its effect (selection + freeze), not its source. Best-of-3's realised
draw-quality gain over many live sessions is not yet characterised beyond the
runs above.
