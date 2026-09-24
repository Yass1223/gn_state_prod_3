# sn_gamestate — Project Reference Guide

## 1. Overview

`sn_gamestate` is a SoccerNet Game State Recognition (GSR) pipeline built on the TrackLab framework. From a SoccerNet-GS video clip (image frames per `SNGS-*` sequence) it produces, for every player/referee, a persistent identity tracked across frames, a pitch-space position, a team side and role, and a jersey number — the full "game state." The pipeline runs as an ordered sequence of TrackLab modules: it detects people (YOLO11-L), tracks them (BoT-SORT + OSNet-AIN with camera-motion compensation), labels crop occlusion, calibrates the camera and projects boxes to the pitch (BroadTrack), gates off-pitch tracklets, splits tracklets into per-identity fragments (DBSCAN), embeds team appearance and clusters into two teams (osnet_team), recognizes jersey numbers (the vendored `jn_gsr` GSR plugin: legibility → DBNet++ → PARSeq+SATRN → vote pool), merges fragments into final trajectories (label- and appearance-aware `traj_refine`), assigns role/team side by geometry and appearance rules, votes a final per-tracklet jersey number, and audits every component. The final per-detection dataframe is saved to `states/sn-gamestate.pklz`, rendered as broadcast overlays plus a radar minimap, and scored with GS-HOTA and reference metrics.

## 2. Project layout

```
repo_prod_v3/
├── README.md                              # top-level project reference: stages, install, run, metrics
├── pyproject.toml                         # sn-gamestate 1.0.0; Python 3.9, torch 1.13.1, tracklab 1.3.24; registers TrackLab plugin
├── preflight_cpu.sh                       # 3-phase CPU preflight: audit → download → re-audit artifacts
├── LICENSE                                # project license
├── uv.lock                                # pinned dependency lockfile (uv)
├── .gitattributes, .gitignore             # git line-ending / ignore rules
├── sn_gamestate/
│   ├── __init__.py                        # exposes __version__ via importlib.metadata
│   ├── config_finder.py                   # TrackLab plugin entry point; registers ${hf:...} resolver
│   ├── configs/
│   │   ├── __init__.py                     # package marker for pkg://sn_gamestate.configs
│   │   ├── soccernet.yaml                  # MASTER Hydra entry config; defaults + ordered pipeline
│   │   ├── modules/…                       # one YAML per stage (see §4, §7)
│   │   ├── eval/gs_hota.yaml               # TrackEval (CLEAR/HOTA/Identity) evaluator
│   │   └── visualization/{gamestate,colors_gs}.yaml
│   ├── bbox_detector/yolo_snft_api.py     # YOLO11-L SoccerNet fine-tune detector
│   ├── track/
│   │   ├── bot_sort.py                     # BoT-SORT + OSNet-AIN + SOF tracker stage
│   │   ├── hf_resolver.py                  # ${hf:...} OmegaConf resolver
│   │   ├── tracklet_split.py               # split-only DBSCAN algorithm (numpy)
│   │   ├── tracklet_split_api.py           # TrackLab wrapper for tracklet split
│   │   └── interpolation.py               # DTI linear interpolation (disabled)
│   ├── crop_filter/crop_filter_api.py     # single/multi overlap labels
│   ├── calibration/broadtrack_api.py      # BroadTrack camera calibration + pitch projection
│   ├── pitch_gate/pitch_gate_api.py       # off-pitch tracklet gate
│   ├── reid/
│   │   ├── osnet_ain.py                    # OSNet-AIN appearance embedder + backbone factory
│   │   └── osnet_team.py                   # osnet_team appearance embedder
│   ├── team/
│   │   ├── team_embed_api.py               # team embedding + 2-means clusters
│   │   ├── role_team_api.py                # role & team-side assignment
│   │   └── rules.py                        # shared rule/helper library (notebook port)
│   ├── refine/
│   │   ├── traj_refine.py                  # 3-phase merge algorithm (numpy)
│   │   └── traj_refine_api.py             # TrackLab wrapper for trajectory refinement
│   ├── jersey/jn_gsr_api.py               # jersey stage driver (subprocess host)
│   ├── audit/run_audit_api.py            # final per-component PASS/WARN/FAIL audit
│   └── visualization/
│       ├── players.py                     # per-detection box/ellipse overlays
│       ├── pitch.py                       # radar minimap
│       └── Radar.png                      # pitch background asset
├── plugins/
│   ├── jn_gsr/                            # vendored jersey-number GSR pipeline (own venv)
│   │   ├── jn_recognizer.py, legibility.py, dbnet_infer.py, roi_dbnet.py,
│   │   │   mmocr_reader.py, fuse_jn.py, evaluate_jn.py, gsr_adapter.py, common.py,
│   │   │   dual_gpu.py, predict_tracklets.py, run_eval.py, crop_classifier.py,
│   │   │   fetch_weights.py, stage_weights.py, audit_parseq.py, setup_env.py,
│   │   │   setup_kaggle.py, stage_data.py, stage_utils.py
│   │   ├── mmocr_cfg/dbnetpp_infer.py     # DBNet++ inference config
│   │   ├── str/parseq                     # vendored PARSeq scene-text recognition library (strhub)
│   │   ├── kaggle_gsr_maxconf.ipynb, README.md, MANIFEST.sha256
│   └── calibration/                       # tracklab-calibration 2.0.0 baseline plugin
│       ├── pyproject.toml
│       └── sn_calibration_baseline/{camera,evaluate_camera,evaluate_extremities,soccerpitch}.py
├── scripts/
│   ├── preflight_imports.py, verify_run_integrity.py, audit_pipeline_columns.py,
│   ├── verify_broadtrack_conversion.py, reference_metrics.py, build_trt_engines.py,
│   ├── inspect_ain_checkpoint.py, setup_broadtrack.sh, setup_jn_gsr.sh, lightning_eval.sh
├── tests/                                 # pytest suite (see §9)
└── docs/                                  # guides & notebooks (see §10)
```

## 3. Pipeline end-to-end

Execution order is defined by `pipeline:` in `sn_gamestate/configs/soccernet.yaml` and is load-bearing.

| # | Stage | Module file | Consumes | Produces |
|---|-------|-------------|----------|----------|
| 1 | `bbox_detector` | `sn_gamestate/bbox_detector/yolo_snft_api.py` | RGB frames, metadatas | `bbox_ltwh`, `bbox_conf`, `image_id`, `video_id`, `category_id` |
| 2 | `track` | `sn_gamestate/track/bot_sort.py` | detections (bbox+conf), frames | `track_id`, `track_bbox_ltwh`, `track_bbox_conf` |
| 3 | `crop_filter` | `sn_gamestate/crop_filter/crop_filter_api.py` | `bbox_ltwh`, `image_id`, `track_id` | `crop_single`, `crop_rT`, `crop_rB`, `crop_trigger` |
| 4 | `calibration` | `sn_gamestate/calibration/broadtrack_api.py` | `bbox_ltwh`, `image_id`, frames | `bbox_pitch`, per-frame `parameters` |
| 5 | `pitch_gate` | `sn_gamestate/pitch_gate/pitch_gate_api.py` | `track_id`, `bbox_pitch` | `track_id` (nulled off-pitch), `track_id_pregate`, `pitch_gate_offpitch`, `pitch_mean_x`, `pitch_mean_y` |
| 6 | `tracklet_split` | `sn_gamestate/track/tracklet_split_api.py` | `track_id`, `bbox_ltwh`, `image_id`, `crop_single` | `track_id` (fragments), `track_id_presplit` |
| 7 | `team_embed` | `sn_gamestate/team/team_embed_api.py` | `track_id`, `bbox_ltwh`, `image_id`, `crop_single` | `team_embedding`, `team_cluster`, `team_cluster_nearest` |
| 8 | `jersey_number_detect` | `sn_gamestate/jersey/jn_gsr_api.py` | `track_id`, `bbox_ltwh`, `image_id`, `crop_single`, `file_path` | `jersey_number_detection`, `jersey_number_confidence`, `jersey_number_candidates`, `jersey_number_maxconf` |
| 9 | `traj_refine` | `sn_gamestate/refine/traj_refine_api.py` | fragments + `team_cluster` + jersey evidence + embeddings | `track_id` (final), `track_id_prerefine`, unified jersey + `team_cluster` |
| 10 | `role_team` | `sn_gamestate/team/role_team_api.py` | `track_id`, `image_id`, `bbox_ltwh`, `bbox_pitch`, `crop_single`, `jersey_number_detection` | `role`, `team` |
| 11 | `tracklet_agg` | `tracklab.wrappers.MajorityVoteTracklet` | `jersey_number` per tracklet | voted `jersey_number` |
| 12 | `audit` | `sn_gamestate/audit/run_audit_api.py` | all columns + all sidecars | `audit/<seq>.json` verdicts (read-only) |

## 4. Stage reference

### 4.1 Detection — `bbox_detector`

- **Files:** `sn_gamestate/bbox_detector/yolo_snft_api.py` — YOLO (ultralytics) detector wrapper.
- **Key classes/functions:**
  - `class YOLOUltralyticsSNFT(YOLOUltralytics)` — YOLO11-L single-class person detector; class attr `level = "image"`.
  - `__init__(cfg, device, batch_size, **kwargs)` — engine-aware loading (imgsz 1280 / iou 0.7 / max_det 300 / half on CUDA); resolves weights from `cfg.path_to_checkpoint` or a `.engine` when TensorRT is enabled.
  - `process(batch, detections, metadatas)` — `@torch.no_grad` inference: RGB→BGR convert, keep boxes where `cls==0` and `conf>=min_confidence`, emit rows.
- **Config:** `sn_gamestate/configs/modules/bbox_detector/yolo_ultralytics_snft_hm.yaml` (selected by the master config); baseline variant `yolo_ultralytics_snft.yaml`.
  - `_target_ = sn_gamestate.bbox_detector.yolo_snft_api.YOLOUltralyticsSNFT`
  - `batch_size = 4` — images per inference batch.
  - `cfg.path_to_checkpoint = ${hf:Ynniss/YOLOv11L_HM,best.zip,yolov11l_hm_best.pt}` — HM fine-tune weights (baseline uses `${hf:${hf_weights_repo},yolov11_sn_best.pt}`).
  - `cfg.min_confidence = 0.35` — confidence floor for the ultralytics `conf` arg and the post-filter (above the tracker's `track_high_thresh` 0.3, so every detection reaches BoT-SORT in the BYTE high band).
  - `cfg.imgsz = 1280` — inference image size.
  - `cfg.iou = 0.7` — NMS IoU threshold.
  - `cfg.max_det = 300` — maximum detections per image.
  - `cfg.use_tensorrt = ${use_tensorrt}` — toggle TensorRT engine vs PyTorch checkpoint.
  - `cfg.engine_path = ${trt_dir}/yolov11l_hm_best.engine` — TensorRT engine path.
- **Data columns:** in = frames/metadatas; out = `image_id`, `bbox_ltwh`, `bbox_conf`, `video_id`, `category_id (=1)`.

### 4.2 Tracking — `track`

- **Files:**
  - `sn_gamestate/track/bot_sort.py` — TrackLab stage wrapping boxmot BoT-SORT with externally-injected OSNet-AIN appearance and externally-computed SOF camera motion.
  - `sn_gamestate/track/hf_resolver.py` — registers the `${hf:repo_id,filename[,local_name]}` OmegaConf resolver (downloads + disk-caches a HuggingFace checkpoint at config-resolution time).
- **Key classes/functions:** `BotSortSOF` with `.reset`, `.process`; helpers `_WarpFeed`, `_NoFeatures`, `_Diagnostics` (`.begin/.record/.flush`), constant `IDENTITY`; `hf_path` + `OmegaConf.register_new_resolver("hf", ...)`.
- **Config:** `sn_gamestate/configs/modules/track/botsort_ain.yaml`
  - `_target_ = sn_gamestate.track.bot_sort.BotSortSOF`
  - `cfg.ain_repo = Ynniss/osnet_ain` — OSNet-AIN checkpoint repo.
  - `cfg.ain_file = best_ain_full.zip` — packed checkpoint.
  - `cfg.ain_revision = d78f65ded828f0cbd8dff2a06dcbff4fc6835dfe` — pinned revision.
  - `cfg.ain_sha256 = a0a7e42676edad0cbc3a4ba0f7d0f8ded75612443ef7a9eabb57cb9e5e245293` — enforced digest.
  - `cfg.ain_local_path = null` — optional on-disk checkpoint (sha256 still enforced).
  - `cfg.embed_batch_size = 64` — crops per appearance forward pass.
  - `cfg.sof_scale = 0.15` — SOF frame downsample fraction before affine warp estimation.
  - `cfg.audit_dir = ${project_dir}/audit/track` — tracker audit sidecar directory.
  - `cfg.hyperparams.min_hits = 1`, `max_age = 50`, `max_obs = 60` — track confirmation/survival/retention.
  - `cfg.hyperparams.track_high_thresh = 0.3` — high-confidence floor (only these are embedded).
  - `cfg.hyperparams.track_low_thresh = 0.05` — BYTE low-band floor.
  - `cfg.hyperparams.new_track_thresh = 0.4` — spawn a new track.
  - `cfg.hyperparams.track_buffer = 60` — base track buffer (→ 50 frames lifetime at 25 fps).
  - `cfg.hyperparams.match_thresh = 0.85` — IoU association threshold.
  - `cfg.hyperparams.proximity_thresh = 0.5` — IoU proximity gate for appearance.
  - `cfg.hyperparams.appearance_thresh = 0.35` — appearance cosine-distance gate.
  - `cfg.hyperparams.cmc_method = sof` — camera motion compensation (sparse optical flow).
  - `cfg.hyperparams.frame_rate = 25` — clip frame rate.
- **Data columns:** in = detections `[l,t,r,b,conf,class,id]`, metadatas (`file_path`, `video_id`), frame; out = `track_bbox_ltwh`, `track_bbox_conf`, `track_id`; sidecar `audit/track/<video_id>.json`.

### 4.3 Crop filter — `crop_filter`

- **Files:** `sn_gamestate/crop_filter/crop_filter_api.py` — per-detection single/multi labels and overlap ratios from bbox overlap, tracked-only contaminator rule.
- **Key classes/functions:**
  - `MODES = ("all","conf","tracked","tracked_or_conf")` — contaminator-selection modes.
  - `label_single_frame(boxes_ltwh, tracked, conf, thr_target, thr_other, contam_mode, conf_thr_other)` — per-frame core: pairwise intersection matrix, `rt=inter/area(T)` and `rb=inter/area(B)` row-maxes, `single=(rt<=thr_target)&(rb<thr_other)`, returns `(single, r_t, r_b, trigger_pos)`.
  - `CropFilter(VideoLevelModule)` — `.__init__` validates mode; `.process` groups by `image_id`, maps trigger positions back to `detections.index`.
- **Config:** `sn_gamestate/configs/modules/crop_filter/overlap_tracked.yaml`
  - `_target_ = sn_gamestate.crop_filter.CropFilter`
  - `cfg.thr_target = 0.25` — max share of the crop covered by a contaminator while still single (`rT <= 0.25`).
  - `cfg.thr_other = 0.40` — max share of a contaminator inside the crop while still single (`rB < 0.40`).
  - `cfg.contam_mode = tracked` — only boxes with a `track_id` may make another box multi.
  - `cfg.conf_thr_other = 0.0` — confidence floor used only by `conf`/`tracked_or_conf` modes.
- **Data columns:** in = `bbox_ltwh`, `image_id`, `track_id`, (`bbox_conf`); out = `crop_single`, `crop_rT`, `crop_rB`, `crop_trigger`.

### 4.4 Calibration — `calibration`

- **Files:**
  - `sn_gamestate/calibration/broadtrack_api.py` — TrackLab VideoLevelModule running the native BroadTrack binary for temporal camera calibration and image-to-pitch projection.
  - `docs/CALIBRATION_FIX.md` — reference on best-of-N draw selection and the CALIB_DATASET freeze.
  - `scripts/verify_broadtrack_conversion.py` — validates the BroadTrack→sn-calibration conversion (`project_direct_from_json`, `check_frame`, `corner_distortion_px`, `main`).
- **Key classes/functions:** `broadtrack_cp_to_sncalib` (cp → sn-calibration params), `get_bbox_pitch` (bottom-left/right/middle → Z=0 plane), `BroadTrackCalibration` with `_frames_dir`, `_sequence_name`, `_prepare_frames_dir`, `_run_binary`, `_attempt_quality`, `_run_binary_best_of`, `_tripod_file`, `process`, `_empty_outputs`.
- **Config:** `sn_gamestate/configs/modules/calibration/broadtrack.yaml`
  - `_target_ = sn_gamestate.calibration.broadtrack_api.BroadTrackCalibration`
  - `binary = ${model_dir}/broadtrack/bin/broadtrack` — native executable.
  - `keypoint_model = …/nbjw_keypoint_model.pt`, `line_model = …/tvcalib_model.pt` — passed via `--k` / `--l`.
  - `libtorch_lib = …/broadtrack/libtorch/lib` — prepended to `LD_LIBRARY_PATH`.
  - `compute_tripod_script = …/src/scripts/compute_tripod.py` — two-pass tripod estimation.
  - `calib_dir = ${project_dir}/broadtrack_calib` — per-sequence calibration JSON cache.
  - `use_cached_json = true` — reuse existing `<seq>.json` on cache hit.
  - `timeout = 7200` — per-sequence binary timeout (s).
  - `calib_attempts = 3` — best-of-N runs on a cache miss; keep highest-quality draw.
  - `prior_xyz = [0.0, 55.0, -12.0]` — camera position prior (m), via `--X/--Y/--Z`.
  - `tripod_mode = none` — `none | per_sequence | per_game`.
  - `tripod_dir = ${project_dir}/broadtrack_calib/tripod`; `game_of = {}` — per-game sharing map.
  - `write_human_bboxes = true` — write player masks to `<frames_dir>/human-bboxes/%06d.json`.
  - `staging_dir = ${project_dir}/broadtrack_staging`; `always_stage = false` — symlink frames when dataset dir is read-only.
  - `use_prev_parameters = true` — carry forward last accepted camera on rejected/missing frames.
  - `min_score = 0.3` — minimum per-frame line-IoU score to accept a frame.
  - `max_carry_frames = 0` — cap on consecutive carry-forward frames (0 = unlimited).
- **Data columns:** in = `bbox_ltwh`, `image_id`, `file_path`, frame images; out = `bbox_pitch` (dict), per-frame `parameters`; artifacts `broadtrack_calib/<seq>.json`, `<seq>.selection.json`, `<seq>.free.json`, `tripod/<key>.tripod.json`, `human-bboxes/%06d.json`.

### 4.5 Pitch gate — `pitch_gate`

- **Files:** `sn_gamestate/pitch_gate/pitch_gate_api.py` — off-pitch tracklet gate.
- **Key classes/functions:** `PitchGate(VideoLevelModule)`; `gate_tracklets`, `tracklet_mean_position`, `is_off_pitch(mean_x,mean_y,margin_m,half_len,half_wid)`, `pitch_xy`, `sequence_name`, `_write`. Rule: off-pitch iff `|mean_x| > 52.5 + margin` or `|mean_y| > 34 + margin` over each tracklet's finite bottom-middle projections; off-pitch rows have `track_id` set to NaN.
- **Config:** `sn_gamestate/configs/modules/pitch_gate/pitch_gate.yaml`
  - `_target_ = sn_gamestate.pitch_gate.PitchGate`
  - `cfg.enabled = true` — when true, null off-pitch `track_id`; when false, no-op on `track_id` but still writes columns + sidecar.
  - `cfg.margin_m = 3.5` — metres the 105×68 m pitch is expanded before the test.
  - `cfg.audit_dir = ${project_dir}/audit/pitch_gate` — per-sequence sidecar directory.
- **Data columns:** in = `track_id`, `bbox_pitch`; out = `track_id`, `track_id_pregate`, `pitch_gate_offpitch`, `pitch_mean_x`, `pitch_mean_y`; sidecar `<audit_dir>/<sequence>.json`.

### 4.6 Tracklet split — `tracklet_split` (SPLITTER, stage 1)

- **Files:**
  - `sn_gamestate/track/tracklet_split.py` — pure NumPy/scikit-learn split-only algorithm.
  - `sn_gamestate/track/tracklet_split_api.py` — TrackLab VideoLevelModule wrapper (OSNet-AIN embedding, relabel, self-checks, audit).
- **Key classes/functions:** `FRAG_BASE (=10000)`, `_unit`, `_centroid`, `_centers`, `split_tracklet(U,single,frames,boxes,eps,min_samples)`, `split_video(...)`; wrapper `TrackletSplit` with `_extract_features`, `process`, `_write`, and `sequence_name`. Splits each tracklet's SINGLE crops with DBSCAN into fragments; single NOISE → nearest centroid; multi GHOSTS attach by time→space→label; all-multi tracklets dissolved video-wide; fragment id `tid*FRAG_BASE+label`, renumbered 1..T.
- **Config:** `sn_gamestate/configs/modules/tracklet_split/tracklet_split.yaml`
  - `_target_ = sn_gamestate.track.tracklet_split_api.TrackletSplit`
  - `cfg.ain_repo = Ynniss/osnet_ain`, `cfg.ain_file = best_ain_full.zip`, `cfg.ain_revision = d78f65d…`, `cfg.ain_sha256 = a0a7e42…`, `cfg.ain_local_path = null` — OSNet-AIN pin (matches tracker).
  - `cfg.eps = 0.2` — DBSCAN eps on cosine distance, per tracklet over SINGLE detections.
  - `cfg.min_samples = 5` — DBSCAN min_samples; fewer singles → one fragment.
  - `cfg.batch_size = 64` — crops per forward pass.
  - `cfg.audit_dir = ${project_dir}/audit/tracklet_split` — sidecar directory.
- **Data columns:** in = `track_id`, `bbox_ltwh`, `image_id`, `crop_single`; out = `track_id` (fragments), `track_id_presplit`; sidecar `<audit_dir>/<sequence>.json`.

### 4.7 Team embedding & clusters — `team_embed`

- **Files:** `sn_gamestate/team/team_embed_api.py` — per-fragment team descriptor from sampled single crops and 2-means clusters. Shared embedders live in `sn_gamestate/reid/osnet_ain.py` and `sn_gamestate/reid/osnet_team.py` (see §5).
- **Key classes/functions:** `TeamEmbedding(VideoLevelModule)` with `.process` (sample single crops on the `pos_stride` grid, embed with osnet_team, L2-normalised median descriptor per fragment, 2-means), `._model`, `._write`; helpers `sequence_name`, `frame_index`.
- **Config:** `sn_gamestate/configs/modules/team_embed/osnet_team.yaml`
  - `_target_ = sn_gamestate.team.TeamEmbedding`
  - `team_repo = Ynniss/osnet_team`, `team_file = osnet_team_best.pt`, `team_revision = null`, `team_sha256 = null`, `team_local_path = null` — checkpoint coordinates.
  - `pos_stride = 5` — frames between sampled crops.
  - `crops_per_track = 16` — max single crops embedded per fragment, evenly spaced.
  - `batch_size = 128` — crops per forward pass.
  - `cluster_method = kmeans2_nearest` — 2-means, every embedded fragment → nearest centroid (alt: `kmeans2_threshold` adds a MAD outlier rule).
  - `outlier_k = 3.25` — `kmeans2_threshold` multiplier (`d > median + k*MAD` → unclustered).
  - `audit_dir = ${project_dir}/audit/team_embed` — sidecar directory.
- **Data columns:** in = `track_id`, `bbox_ltwh`, `image_id`, `crop_single`; out = `team_embedding`, `team_cluster` (0/1), `team_cluster_nearest`; sidecar `<audit_dir>/<sequence>.json`.

### 4.8 Jersey number — `jersey_number_detect`

- **Files:** `sn_gamestate/jersey/jn_gsr_api.py` — TrackLab video-level driver that runs the vendored `jn_gsr` plugin as a subprocess (see §6 for the plugin files).
- **Key classes/functions:** `JNGsrTrackletRecognizer(VideoLevelModule)` with `.process`, `._build_manifest`, `._manifest_hash`, `._ckpt_id`, `._launch_workers`; module `detect_gpus`; constants `RULE='vote_pool'`, `SCHEMA=2`, `PARSEQ_CKPT`, `SATRN_CKPT`, `UNNUMBERED='-1'`.
- **Config:** `sn_gamestate/configs/modules/jersey_number_detect/jn_gsr.yaml`
  - `_target_ = sn_gamestate.jersey.jn_gsr_api.JNGsrTrackletRecognizer`
  - `cfg.pipeline_dir = ${project_dir}/plugins/jn_gsr` — plugin/worker cwd.
  - `cfg.venv_python = …/plugins/jn_gsr/.venv_jn/bin/python` — plugin venv interpreter.
  - `cfg.models_dir = …/plugins/jn_gsr/models` — staged checkpoints, base for content hashing.
  - `cfg.worker` (default `<pipeline_dir>/predict_tracklets.py`) — per-GPU worker script.
  - `cfg.cache_dir = ${project_dir}/jn_cache`; `cfg.use_cache = true` — content-hash result cache.
  - `cfg.stride = 5` — read every 5th single crop of each tracklet.
  - `cfg.fp16 = true` — CUDA autocast float16.
  - `cfg.legibility_thr = 0.72` — strictly-greater `p_legible` cut; part of cache key.
  - `cfg.parseq_ckpt = parseq_gsr_ft_s1.ckpt` — recogniser A checkpoint.
  - `cfg.satrn_ckpt = recog2/best_recog_word_acc_epoch_10.pth` — recogniser B checkpoint.
  - `cfg.single_crops_only = true` — only `crop_single` detections enter the manifest.
  - `cfg.gpus = auto` — one worker per detected GPU.
  - `cfg.timeout = 10800` — seconds per video across workers.
  - `cfg.worker_extra_args = []` — extra CLI args (test hook).
  - `cfg.rule` — fixed `vote_pool` (setting it makes the stage refuse to construct).
- **Data columns:** in = `track_id`, `bbox_ltwh`, `image_id`, `crop_single`, `file_path`; out = `jersey_number_detection`, `jersey_number_confidence`, `jersey_number_candidates`, `jersey_number_maxconf`; artifacts `jn_cache/<seq>.<mhash12>.json`, per-worker `shard_<i>.json`.

### 4.9 Trajectory refinement — `traj_refine` (MERGER, stage 2 — the one merge)

- **Files:**
  - `sn_gamestate/refine/traj_refine.py` — pure NumPy three-phase merger + number-conflict resolution + stage-3 duplicate-frame resolution.
  - `sn_gamestate/refine/traj_refine_api.py` — TrackLab VideoLevelModule wrapper.
- **Key classes/functions:** `DIGITS_1_99`, `score_of`, `pair_maxconf`, `combine_cand`, `ranked_labels`, `_Cluster` (`.key`, `.centroid()` mean/Stage-3, `.median_centroid()` merge, `.absorb()`), `_dist`, `_in_s1`/`_in_s2`, `_phase_within_tracklet` (phase 0), `_phase_s1`, `_resolve_conflict`, `_agglomerate`, `_phase_s2`, `_phase_final`, `_stage3`, `refine_video`. Wrapper: `TrajRefine(VideoLevelModule)`, `_is_null`, `_model`, `_extract_features`, `_track_info`, `process`, `_write`. **Phase 0 (within-tracklet)** runs first: re-merges fragments of the SAME source tracklet (`track_id_presplit`) — same cluster+same number (no threshold), or same cluster+number-unknown at `dist<=tau`; different known numbers never merge. Then the cross-tracklet phases partition fragments into S1 (cluster+number known) and S2 (cluster known, number unknown); S1 label-driven merges (no threshold), S2 median-centroid merging at `dist<=tau`, FINAL agglomerative over survivors under same-cluster + disjoint clean frames + no contradicting numbers at `dist<=tau`; stage 3 resolves (frame,traj) collisions.
- **Config:** `sn_gamestate/configs/modules/traj_refine/traj_refine.yaml`
  - `_target_ = sn_gamestate.refine.traj_refine_api.TrajRefine`
  - `cfg.ain_repo = Ynniss/osnet_ain`, `cfg.ain_file = best_ain_full.zip`, `cfg.ain_revision = d78f65d…`, `cfg.ain_sha256 = a0a7e42…`, `cfg.ain_local_path = null` — OSNet-AIN pin (must match tracklet_split).
  - `cfg.enabled = true` — master switch; false writes only snapshots + sidecar.
  - `cfg.tau = 0.70` — appearance merge threshold (`1 - dot` of the two clusters' clean-crop **median** centroids, each recomputed over its full membership after every merge) for the within-tracklet rule B, S2 and FINAL; the pipeline's only merge threshold. Stage 3 keeps the mean centroid.
  - `cfg.batch_size = 64` — crops per forward pass.
  - `cfg.audit_dir = ${project_dir}/audit/traj_refine` — sidecar directory.
- **Data columns:** in = `track_id`, `bbox_ltwh`, `image_id`, `crop_single`, `team_cluster`, `jersey_number_detection`, `jersey_number_confidence`, `jersey_number_candidates`, `team_embedding`, `file_path`; out = `track_id` (final), `track_id_prerefine`, unified `jersey_number_detection`/`_confidence`/`_maxconf`, `*_prerefine` snapshots, unified `team_cluster` (+ `team_cluster_prerefine`), `team_embedding = None`; sidecar `<audit_dir>/<sequence>.json`.

### 4.10 Role & team-side assignment — `role_team`

- **Files:**
  - `sn_gamestate/team/role_team_api.py` — active role_team stage.
  - `sn_gamestate/team/rules.py` — shared rule/helper library (see §5).
- **Key classes/functions:** `RoleTeamAssignment(VideoLevelModule)` with `.process` (7-step chain: geometry, descriptors, appearance-outlier rule, assistants, goalkeepers, main referee, sides), `._descriptors`, `._model`, `._write`; helpers `appearance_outliers` (mutual-reachability outlier rule), `_pick_candidate`, `_cues`, `_pitch_xy`, `_f`; `TEAM_NAMES={0:left,1:right}`, `DEFAULTS`. Roles ∈ {player, goalkeeper, referee}; team ∈ {left, right} (None for referees); everything computed from SINGLE crops and written to all rows.
- **Config:** `sn_gamestate/configs/modules/role_team/rules.yaml`
  - `_target_ = sn_gamestate.team.RoleTeamAssignment`
  - `cfg.team_repo = Ynniss/osnet_team`, `cfg.team_file = osnet_team_best.pt`, `cfg.team_revision = null`, `cfg.team_sha256 = null`, `cfg.team_local_path = null` — osnet_team checkpoint.
  - `cfg.pos_stride = 5`, `cfg.crops_per_track = 16`, `cfg.batch_size = 128`, `cfg.audit_dir = ${project_dir}/audit/role_team`.
  - `cfg.params.link_k = 4` — neighbour count of the mutual-reachability appearance-outlier rule (`appearance_outliers_plain_v2`, its ONE parameter): team cores = members with `d ≤ median + MAD` of centroid distances; per-tracklet radius `R_i = median + MAD` of its `link_k` NN distances; outlier ⟺ no mutually-compatible chain (`δᵢⱼ ≤ min(Rᵢ, Rⱼ)`) reaches either core; skipped when `n_desc ≤ link_k + 1`.
  - `cfg.params.tau_a = 0.9` — assistant candidate: `|mean y| >= tau_a * max|mean y|`.
  - `cfg.params.tau_a_sy = 3.0` — assistant candidate max y-std (m).
  - `cfg.params.side_rule = keeper` — side-naming cue chain.
  - `cfg.params.band = 0.9` — symmetric y-band factor for the main-referee 2.14 rule.
  - `cfg.params.gk_depth_m = 2.0` — goalkeeper depth tie margin `|mean x|` (m).
  - `cfg.params.a_tie_m = 1.5` — assistant tie margin `|mean y|` (m).
  - `cfg.params.min_n = 10` — min sampled positions for GK / main-referee candidacy.
  - `cfg.params.gk_rel = 0.85` — GK relative-depth gate (`|mean x| > gk_rel * max|mean x|`).
- **Data columns:** in = `track_id`, `image_id`, `bbox_ltwh`, `bbox_pitch`, `crop_single`, `jersey_number_detection`; out = `role`, `team`; sidecar `<audit_dir>/<sequence>.json`.

### 4.11 Tracklet aggregation — `tracklet_agg`

- **Files:** module class `tracklab.wrappers.MajorityVoteTracklet` (framework-provided).
- **Config:** `sn_gamestate/configs/modules/tracklet_agg/voting_jn.yaml`
  - `_target_ = tracklab.wrappers.MajorityVoteTracklet`
  - `cfg.attributes = ["jersey_number"]` — majority-vote the jersey number per tracklet; winner written to every row.
- **Data columns:** in/out = `jersey_number` (voted per track).

### 4.12 Audit — `audit`

- **Files:** `sn_gamestate/audit/run_audit_api.py` — final, read-only per-component verdict stage.
- **Key classes/functions:** `RunAudit(VideoLevelModule)` with `.process` (13 checks, writes `<seq>.json`); `Check` (severity order INFO<PASS<WARN<FAIL); `_check_detector`, `_check_track`, `_check_tracker_internals`, `_check_tracklet_split`, `_check_crop_filter`, `_check_calibration`, `_check_pitch_gate`, `_check_team_embed`, `_check_role_team`, `_check_jersey`, `_check_traj_refine`, `_check_tracklet_agg`, `_check_visualization`; helpers `_eligible_tids`, `_has_single`, `_fragments_without_single`, `_find_track_sidecar`, `_read_sidecar`, `_ckpt_id`, `_sha256`, `_per_track_constant`, `_tid`, `_is_nan`, `_is_number`, `_is_float`, `_share`.
- **Config:** `sn_gamestate/configs/modules/audit/run_audit.yaml`
  - `_target_ = sn_gamestate.audit.RunAudit`
  - `cfg.out_dir = ${project_dir}/audit` — verdict JSON output.
  - `cfg.jn_cache_dir = ${project_dir}/jn_cache`, `cfg.calib_dir = ${project_dir}/broadtrack_calib`, `cfg.models_dir = …/plugins/jn_gsr/models` — artifact locations.
  - `cfg.calib_min_score = ${modules.calibration.cfg.min_score}` — lost-frame threshold.
  - `cfg.parseq_ckpt = parseq_gsr_ft_s1.ckpt`, `cfg.satrn_ckpt = recog2/best_recog_word_acc_epoch_10.pth` — checkpoints whose sha256 must match the jersey blob.
  - `cfg.jn_single_crops_only = ${modules.jersey_number_detect.cfg.single_crops_only}`.
  - `cfg.track_sidecar_dir`, `cfg.tracklet_split_sidecar_dir`, `cfg.team_embed_sidecar_dir`, `cfg.role_team_sidecar_dir`, `cfg.pitch_gate_sidecar_dir`, `cfg.traj_refine_sidecar_dir` — each stage's `audit_dir`.
  - `cfg.expected_tracker.{ain_sha256_track, ain_sha256_tracklet_split, ain_file_track, ain_file_tracklet_split, ain_revision_track, ain_revision_tracklet_split, appearance_thresh, sof_scale}` — declared-vs-ran tracker values; enforces track-vs-`tracklet_split` OSNet-AIN equality on file, revision **and** sha256.
  - `cfg.expected_tracklet_split.{eps, min_samples, ain_sha256}` — declared splitter values.
  - `cfg.expected_crop_filter.{thr_target, thr_other, contam_mode}` — used to recompute labels.
  - `cfg.expected_team_embed.{sha256, pos_stride, crops_per_track, cluster_method, outlier_k}`.
  - `cfg.expected_role_team.params`, `cfg.expected_pitch_gate.{enabled, margin_m}`.
  - `cfg.expected_traj_refine.{enabled, tau, ain_sha256, ain_sha256_tracklet_split}`.
  - Thresholds: `empty_frames_warn=0.20`, `tracked_warn=0.50`, `single_share_warn=0.30`, `pitch_missing_warn=0.05`, `embed_missing_warn=0.01`, `off_grid_warn=0.05`, `jn_min_eligible_for_zero_fail=10`, `radar_skipped_tracked_warn=0.05`, `cmc_identity_warn=0.02`, `crop_clipped_warn=0.001`, `zero_emb_warn=0.01`, `tracklet_split_zero_emb_warn=0.01`, `pitch_gate_gated_warn=0.50`, `pitch_gate_no_position_warn=0.05`, `calib_lost_frames_warn=0.10`.
- **Data columns:** in = all detection columns + all sidecars; out = `audit/<seq>.json` (never modifies detections).

## 5. Shared components

### OSNet embedders (`sn_gamestate/reid/`)

| Embedder | File | Input | Output | Used by |
|----------|------|-------|--------|---------|
| **OSNet-AIN** | `sn_gamestate/reid/osnet_ain.py` | BGR crops, 256×128, ImageNet norm, fp16 autocast on CUDA | (N,dim) L2-normalised float32 | tracker (`bot_sort.py`), `tracklet_split`, `traj_refine` |
| **osnet_team** | `sn_gamestate/reid/osnet_team.py` | RGB crops, 128×64 on grey-110 letterbox, flip TTA, fp32, 256-d | (N,dim) L2-normalised float32 | `team_embed`, `role_team` |

- **`osnet_ain.py`** key symbols: `OsnetAin.embed`, `from_config`, `resolve_checkpoint`/`load_checkpoint`/`_validate_checkpoint`, `build_backbone` (shared backbone factory used by osnet_team too), `_Net` (backbone→GAP→fc→BNNeck→{classifier, role_head}), `letterbox`/`crop_ltrb`/`sha256`; constants `REPO_ID=Ynniss/osnet_ain`, `FILENAME=best_ain_full.zip`, `REVISION=d78f65d…`, `SHA256=a0a7e42…`, `TARGET_ASPECT=2.0`.
- **`osnet_team.py`** key symbols: `OsnetTeam.embed`, `OsnetTeam.letterbox`, `TeamNet` (backbone→GAP→fc→BNNeck→bias-free proj→L2 normalize), `from_config`, `resolve_checkpoint`/`load_checkpoint`, `crop_rgb`, `sha256`; constants `REPO_ID=Ynniss/osnet_team`, `FILENAME=osnet_team_best.pt`, `BACKBONE=osnet_x1_0`, `GREY=110`.

### `sn_gamestate/team/rules.py`

Shared rule/helper library (notebook port) used by `role_team_api.py`. Key symbols: `sample_tracklet_rows(frames, stride=POS_STRIDE, crops=CROPS_PER_TRK)`, `kmeans2(E)` (KMeans(2, n_init=10, random_state=SEED=42)), `knee_eps(E, kth=3)`, `tracklet_row(...)`, `run_sequence(D,P)` (the notebook's full per-sequence rule chain), `check_params(P)`. Constants: `POS_STRIDE=5`, `CROPS_PER_TRK=16`, `MIN_CROPS=3`, `SEED=42`, `PITCH_HALF_LEN=52.5`, `PITCH_HALF_WID=34.0`, `PEN_X=36.0`, `PEN_Y=20.16`, `ROLE_OUT/ROLE_GK/ROLE_REF`, `ROLE_NAMES`, `FROZEN_PARAMS`, `PARAM_CHOICES`. `PITCH_HALF_LEN`/`PITCH_HALF_WID` are also consumed by the pitch gate.

## 6. Plugins

### 6.1 `plugins/jn_gsr` — jersey-number GSR pipeline

Recognizes each tracklet's shirt number in its own Python 3.10 / torch 2.0.1 venv. Per-tracklet flow: subsample by stride → ResNet-34 legibility (`p_legible > 0.72`) AND DBNet++ text detection (top-quad score `> 0.52`) as conjunctive gates → highest-score quad → AABB + glyph-height padding → ROI crop → PARSeq (`parseq_gsr_ft_s1.ckpt`) and SATRN (`recog2/best_recog_word_acc_epoch_10.pth`) each decode 11-way per-position (tens, units) log-likelihoods over `0123456789E` → `vote_pool` consolidation.

| File | Role |
|------|------|
| `plugins/jn_gsr/jn_recognizer.py` | Production per-tracklet API: `JerseyNumberRecognizer` (`predict`, `predict_full`, `consolidate`, `consolidate_full`, `_norm`); `LEGIBILITY_THR=0.72`, `DET_THR=0.52`, `RULE='vote_pool'` |
| `plugins/jn_gsr/legibility.py` | Koshkina ResNet-34 sigmoid legibility classifier: `LegibilityClassifier.score`, `build_model`, `detect_arch`, `resolve_weights`, `frame_verdicts`, `tracklet_is_legible`; `DEFAULT_SIZE=256` |
| `plugins/jn_gsr/dbnet_infer.py` | DBNet++ `TextDetInferencer` wrapper + frame gate/ROI provider: `DBNetDetector.detect`, `DetectorGate` (`is_legible`, `roi_crop`, `roi_crop_scored`), `padded_player_crop`, `make_pose_fn`, `resolve_ckpt`; `PLAYER_PAD=0.18` |
| `plugins/jn_gsr/roi_dbnet.py` | Polygon→ROI crop conversion: `roi_from_detections`, `_aabb`, `_glyph_height`; `DET_THR=0.52`, `PAD_FRAC=0.12`, `MIN_SIDE=4` |
| `plugins/jn_gsr/mmocr_reader.py` | SATRN recogniser behind the `read_many([crops])` contract: `MMOCRRecogniser` (`read_many`, `_scores`), `build_recog2_reader`, `config_from_checkpoint`, `classifier_out_dim`, `resolve_dict_file` |
| `plugins/jn_gsr/fuse_jn.py` | Two-recogniser consolidation: `fuse`, `fuse_stats`, `label_stats`, `ranked_candidates`, `pooled_label_stats`, `maxconf_of`, `votes_of`, `strength_of`; `RULES` (12 incl. `vote_pool`) |
| `plugins/jn_gsr/evaluate_jn.py` | Single-model evaluation harness + shared per-frame decode: `consolidate_tracklet*`, `_decode_frame`, `tracklet_accuracy`, `minus_one_prf`, `detection_metrics`, `sweep_min_conf` |
| `plugins/jn_gsr/gsr_adapter.py` | GSR-2024 dataset adapter: `build_pool`, `find_sequences`, `pad_box`, `iou`, `match_to_gt`, `jersey_label`; `CROP_PAD_FRAC=0.18`, `ROLES={'player','goalkeeper'}` |
| `plugins/jn_gsr/common.py` | Shared helpers: `subsample`, `build_parseq_batch_reader`, `build_parseq_reader`, `seed_everything`; `FRAME_STRIDE=5`, `CHARSET='0123456789'` |
| `plugins/jn_gsr/dual_gpu.py` | Stdlib-only per-GPU subprocess driver: `main`, `stream` |
| `plugins/jn_gsr/predict_tracklets.py` | GSR-pipeline subprocess worker run in `.venv_jn`: reads manifest, runs recognizer per shard, writes shard JSON; `RULE='vote_pool'`, `SCHEMA=2` |
| `plugins/jn_gsr/run_eval.py` | Single-model PARSeq eval worker + merge; `build_models` reused by `JerseyNumberRecognizer` |
| `plugins/jn_gsr/crop_classifier.py` | ResNet-18 single/multi frame filter: `crop_box`, `numpy2_pickle_compat` |
| `plugins/jn_gsr/fetch_weights.py` | Fetch/hash-check DBNet++, ResNet-34 legibility, SATRN; writes `fetch_weights_provenance.json` |
| `plugins/jn_gsr/stage_weights.py` | Resolve PARSeq checkpoint → `models/parseq_gsr_ft_s1.ckpt`; writes `weights_provenance.json` |
| `plugins/jn_gsr/audit_parseq.py` | In-venv audit (stages D1–D6) verifying the PARSeq checkpoint swap takes effect |
| `plugins/jn_gsr/setup_env.py` | Builds the Python 3.10 / torch 2.0.1+cu118 venv (mmcv/mmengine/mmdet/mmocr + PARSeq) |
| `plugins/jn_gsr/setup_kaggle.py` | Kaggle entry point relocating the venv onto ephemeral scratch |
| `plugins/jn_gsr/stage_data.py` | Resumable GSR-2024 dataset downloader/extractor (`--splits`, `--delete-zip`) |
| `plugins/jn_gsr/stage_utils.py` | Shared Kaggle plumbing: `JN_*` env conventions, on-disk state |
| `plugins/jn_gsr/str/parseq` | Vendored PARSeq scene-text recognition library (strhub) — recogniser A |
| `plugins/jn_gsr/mmocr_cfg/dbnetpp_infer.py` | DBNet++ inference config consumed by `DBNetDetector` |
| `plugins/jn_gsr/kaggle_gsr_maxconf.ipynb` | Stand-alone Kaggle validation notebook |
| `plugins/jn_gsr/README.md` | Plugin documentation: config, rules, provenance, integration, file inventory |
| `plugins/jn_gsr/MANIFEST.sha256` | File-integrity manifest: sha256 provenance for the plugin's tracked files |

### 6.2 `plugins/calibration` — sn_calibration_baseline (tracklab-calibration 2.0.0)

Geometry library used by the BroadTrack stage and the reference-metrics harness.

| File | Role |
|------|------|
| `plugins/calibration/sn_calibration_baseline/camera.py` | Pinhole camera with distortion: `Camera`, `to_homography`/`from_homography`, `solve_pnp`, `refine_camera`, `project_point`, `distort`/`undistort_point`, `estimate_calibration_matrix_from_plane_homography`, `to_json_parameters`/`from_json_parameters`, `draw_pitch`, `pan_tilt_roll_to_orientation`, `rotation_matrix_to_pan_tilt_roll`, `unproject_image_point` |
| `plugins/calibration/sn_calibration_baseline/evaluate_camera.py` | Whole-camera evaluator (mirror-aware): `get_polylines`, `distance_to_polyline`, `evaluate_camera_prediction`, `completeness_score`, `final_score` |
| `plugins/calibration/sn_calibration_baseline/evaluate_extremities.py` | Line-extremity evaluator: `distance`, `mirror_labels`, `evaluate_detection_prediction`, `scale_points` |
| `plugins/calibration/sn_calibration_baseline/soccerpitch.py` | Metric 3D pitch model: `SoccerPitch`, `lines_classes`, `symetric_classes`, `palette`, `point_dict`, `line_extremities`, `points`, `sample_field_points`, `get_2d_homogeneous_line`, `CENTER_CIRCLE_RADIUS`, `PENALTY_AREA_WIDTH/LENGTH` |
| `plugins/calibration/pyproject.toml` | Packaging: `name=tracklab-calibration`, `version=2.0.0`, `requires-python >=3.9,<3.10`, `dependencies=[]`, `packages.find.include=["sn_calibration_baseline"]` |

## 7. Configuration system

- **Packaging & plugin registration.** `pyproject.toml` pins the runtime (Python 3.9, torch 1.13.1, tracklab 1.3.24) and registers `[project.entry-points.tracklab_plugin] sn_gamestate = sn_gamestate.config_finder:ConfigFinder`. `sn_gamestate/config_finder.py` (`ConfigFinder`, `config_package = 'pkg://sn_gamestate.configs'`) tells Hydra where the configs live and imports `sn_gamestate.track.hf_resolver` to register the `${hf:...}` resolver. `sn_gamestate/configs/__init__.py` makes the config package importable; `sn_gamestate/__init__.py` exposes `__version__`. Per-package `__init__.py` shims re-export stage classes (`crop_filter.CropFilter`, `pitch_gate.PitchGate`, `team.TeamEmbedding`/`RoleTeamAssignment`, `audit.RunAudit`, `visualization.Radar`/`CompletePlayerBBox`, …) so the short `_target_` strings in the module configs resolve; `refine/` and `jersey/` keep empty `__init__`, so their `_target_` uses the full submodule path (e.g. `sn_gamestate.refine.traj_refine_api.TrajRefine`).
- **Master config.** `sn_gamestate/configs/soccernet.yaml` is the single Hydra entry config. Its `defaults` list composes `dataset=soccernet_gs`, `eval=gs_hota`, `engine=offline`, `visualization=gamestate`, and one file per module (`bbox_detector=yolo_ultralytics_snft_hm`, `track=botsort_ain`, `crop_filter=overlap_tracked`, `tracklet_split=tracklet_split`, `interpolation=dti`, `calibration=broadtrack`, `pitch_gate=pitch_gate`, `team_embed=osnet_team`, `role_team=rules`, `jersey_number_detect=jn_gsr`, `traj_refine=traj_refine`, `tracklet_agg=voting_jn`, `audit=run_audit`). The `pipeline:` list fixes the executed order (`interpolation` is composed but excluded).
- **Key master-config knobs:**
  - `hf_weights_repo = Ynniss/sn-gamestate-weights` — repo for `${hf:...}` weight fetches.
  - `pipeline = [bbox_detector, track, crop_filter, calibration, pitch_gate, tracklet_split, team_embed, jersey_number_detect, traj_refine, role_team, tracklet_agg, audit]`.
  - `experiment_name = sn-gamestate`; `home_dir = ${oc.env:HOME}`; `data_dir = ${project_dir}/data`; `model_dir = ${project_dir}/pretrained_models`.
  - `use_tensorrt = false`; `trt_dir = ${model_dir}/trt`; `num_cores = 4`; `use_wandb = False`; `use_rich = True`.
  - `modules.bbox_detector.batch_size = 4`; `modules.track.batch_size = 64`.
  - `test_tracking = True`; `eval_tracking = True`; `print_config = False`.
  - `dataset.nvid = 1` (`-1` = whole split); `dataset.eval_set = test`; `dataset.dataset_path = ${data_dir}/SoccerNetGS`; `dataset.vids_dict = {valid: [], test: []}`.
  - `state.save_file = states/${experiment_name}.pklz`; `state.load_file = null`; `visualization.cfg.save_videos = True`.
  - `project_dir = ${hydra:runtime.cwd}`; `hydra.output_subdir = configs`; `hydra.job.chdir = True`; `hydra.run.dir = outputs/${experiment_name}/${now:%Y-%m-%d}/${now:%H-%M-%S}`; `hydra.sweep.dir = multirun_outputs/${experiment_name}/${now:%Y-%m-%d}/${now:%H-%M-%S}`.
- **Launching a run.** `tracklab -cn soccernet` composes `soccernet.yaml`, dumps the composed config to `outputs/sn-gamestate/<date>/<time>/configs`, runs the ordered pipeline, and saves state to `states/sn-gamestate.pklz`.

## 8. Scripts & tooling

| Script | Purpose |
|--------|---------|
| `scripts/preflight_imports.py` | Import-checks every config `_target_` plus runtime-only imports and builds backbones in one GPU-free pass; exit code = number of broken stages |
| `scripts/setup_broadtrack.sh` | Docker-free native build of the EVS BroadTrack binary + TorchScript weights + libtorch, with toolchain patches and smoke test |
| `scripts/setup_jn_gsr.sh` | Provisions the jersey pipeline in `.venv_jn`, fetches hash-checked checkpoints, runs self-tests, audits the PARSeq checkpoint |
| `scripts/build_trt_engines.py` | Builds a TensorRT FP16 engine for the YOLO11-L detector via ultralytics export; reports a status table |
| `scripts/inspect_ain_checkpoint.py` | Reports OSNet-AIN checkpoint contents (backbone, dims, role classes, sha256) and probes buildable factories; CPU-only |
| `scripts/lightning_eval.sh` | End-to-end runner: creates the 3.9 venv, installs project + boxmot, patches TrackLab dataset name, downloads splits, runs BroadTrack/jersey provisioning, runs tracklab per split |
| `preflight_cpu.sh` | Three-phase CPU preflight: audit expected artifacts → download flagged items → re-audit (`CHECK_ONLY=1` audits only) |
| `scripts/verify_broadtrack_conversion.py` | Validates the BroadTrack→sn-calibration conversion via model-equivalence, plane-roundtrip, overlay, coverage checks |
| `scripts/reference_metrics.py` | Standalone CLI computing tracking (HOTA/DetA/AssA/MOTA/IDF1/IDSW), GSR (GS-HOTA/GS-DetA/GS-AssA/GS-IDF1), jersey-number, and calibration metrics from a saved state; writes per-label JSON + `summary.md` |
| `scripts/verify_run_integrity.py` | Read-only post-run gate: scans the newest eval log for failure signatures, checks calibration/jersey artifacts and audit JSONs; non-zero exit only on execution failures |
| `scripts/audit_pipeline_columns.py` | Read-only column-level and per-tracklet stage-I/O audit against a saved `.pklz`; four report sections + `crop_check.png`; non-zero exit on any FAIL |

## 9. Tests

| Test file | Verifies |
|-----------|----------|
| `tests/README.md` | How to run the suite (repo root, 3.9 venv) and what each test covers |
| `tests/notebook_reference.py` | Verbatim original notebook code kept as the reference for rules equivalence (not used by the pipeline) |
| `tests/test_audit.py` | `crop_filter → pitch_gate → team_embed → role_team` on a synthetic run; asserts audit per-stage PASS/FAIL with negative controls |
| `tests/test_audit_pipeline_columns.py` | `scripts/audit_pipeline_columns.py` on a synthetic state + sidecars + blob; healthy run passes and four broken variants each flip one check to FAIL |
| `tests/test_broadtrack_bestof.py` | Best-of-N draw selection (stubbed binary): keeps highest-mean-score attempt, writes selection sidecar, tolerates failures, all-failing returns failure |
| `tests/test_jersey_single_crops.py` | Single-crop manifest selection and cache-key folding; audit jersey check with negative controls; no torch/GPU |
| `tests/test_pitch_gate.py` | Pitch-gate rule, stage contract, enable switch, sidecar, audit check on synthetic tracklets |
| `tests/test_rules_equivalence.py` | Ported `team/rules.py` produces identical roles/teams/reasons/outliers/tables as `notebook_reference` |
| `tests/test_stages.py` | End-to-end `crop_filter → team_embed → role_team` on 60 synthetic frames with a synthetic osnet_team checkpoint; load, preprocessing, flip TTA, column contracts |
| `tests/test_tracklet_split.py` | 16 pure-numpy tests for the split-only algorithm (DBSCAN, ghost attachment, all-multi dissolution, invariants, determinism) |
| `tests/test_traj_refine.py` | 27 pure-numpy tests for the merger — within-tracklet phase 0, the three cross-tracklet phases, and stage-3 duplicate-frame resolution |
| `tests/test_visualization.py` | Visualization colour contract: trajectory-wide role/team labels drawn on every row, same box colour and radar disc; unlabelled rows undrawn |

## 10. Docs & notebooks

| Path | Contents |
|------|----------|
| `README.md` | Top-level project reference: pipeline stage table, `tracklet_split`/`pitch_gate` sections, install, run, reference metrics, layout |
| `docs/PIPELINE_REFERENCE.md` | Canonical pipeline description: stage order/rationale, three execution environments, per-stage `_target_` and parameters, artifact sources/fallbacks/integrity, run/verify commands, change log |
| `docs/KAGGLE_GUIDE.md` | Verified Kaggle (GPU T4×2) procedure: session requirements, disk layout, environment rules, network fallback, one-sequence recipe, timings and reference results |
| `docs/CALIBRATION_FIX.md` | Best-of-N draw selection and the CALIB_DATASET freeze for BroadTrack draw variance |
| `docs/kaggle_one_sequence_test.ipynb` | Runnable Kaggle notebook: clone/environment gates, single-sequence extraction, full run, verification chain, `traj_refine` sidecar summary, detector/refine A/B cells |

## 11. Visualization & evaluation

- **Overlays.** `sn_gamestate/visualization/players.py` — per-detection boxes/ellipses with compact labels: `TeamVisualizer` (`color`), `Player`, `PlayerEllipse`, `CompletePlayerEllipse`, `CompletePlayerBBox` (single-line `JN | ID <n> | L/R` tag), `side_letter`, `pprint`. `sn_gamestate/visualization/pitch.py` — radar minimap: `Radar` (`_panel_template`, `draw_frame`), `radar_color` (left=blue, right=red, referee=yellow), `radar_label`; constants `PITCH_LENGTH=105`, `PITCH_WIDTH=68`, `COLOR_LEFT=(0,0,255)`, `COLOR_RIGHT=(255,0,0)`, `COLOR_REFEREE=(238,210,2)`. `sn_gamestate/visualization/Radar.png` is the pitch background asset.
- **Visualization config.** `sn_gamestate/configs/visualization/gamestate.yaml` (`_target_ = tracklab.visualization.VisualizationEngine`, `save_videos=True`, `visualizers.players._target_ = sn_gamestate.visualization.CompletePlayerBBox` with `display_track_id/display_jersey=true`, `display_role/display_team=false`, `visualizers.radar._target_ = sn_gamestate.visualization.Radar` with `scale=4`, `alpha=0.8`, `margin_bottom=12`, `colors.default.prediction=team`) and `colors_gs.yaml` (`colors.cmap=22`; prediction colors left `[0,0,255]`, right `[255,0,0]`, referee `[238,210,2]`; GT colors left/right `[0,255,0]`, referee `[255,255,0]`).
- **Evaluation config.** `sn_gamestate/configs/eval/gs_hota.yaml` (`_target_ = tracklab.wrappers.TrackEvalEvaluator`, `eval_set=${dataset.eval_set}`, `dataset_path=${dataset.dataset_path}`, `cfg.bbox_column_for_eval=bbox_ltwh`, `cfg.metrics=[CLEAR, HOTA, Identity]`, `cfg.eval.USE_PARALLEL=True`, `NUM_PARALLEL_CORES=${num_cores}`, `PRINT_RESULTS=True`, `OUTPUT_SUMMARY/OUTPUT_DETAILED/PLOT_CURVES=True`, `cfg.dataset=${dataset.track_eval}`). `scripts/reference_metrics.py` (`main`, `load_state`, `run_trackeval_variant`, `tracking_block`, `gsr_block`, `jersey_block`, `calibration_block`, `write_summary`) is the standalone metrics CLI writing `reference_metrics/<label>/reference_metrics.json` and `reference_metrics/summary.md`.

## 12. Aggregation & interpolation (support modules)

- **`tracklet_agg`** — `tracklab.wrappers.MajorityVoteTracklet` via `sn_gamestate/configs/modules/tracklet_agg/voting_jn.yaml` (`cfg.attributes=["jersey_number"]`): majority-vote the final per-track jersey number (see §4.11).
- **`interpolation` (DTI, disabled)** — `sn_gamestate/track/interpolation.py`: `LinearInterpolation(VideoLevelModule)` (`input=[track_id, bbox_ltwh, image_id]`, `output=[interpolated]`), `interpolate_detections`, `frame_ranks`, `_CARRIED=(track_id, video_id, category_id)`. Config `sn_gamestate/configs/modules/interpolation/dti.yaml`: `_target_ = sn_gamestate.track.interpolation.LinearInterpolation`, `cfg.enabled=false`, `cfg.n_dti=25` (fill only when `1 < dt < n_dti`), `cfg.n_min=5` (only tracklets with `>= n_min` real detections). Composed in `defaults` but not in the executed `pipeline`.

## 13. Data-column glossary

| Column | Set by (stage) | Meaning |
|--------|----------------|---------|
| `bbox_ltwh` | bbox_detector | Bounding box as left, top, width, height |
| `bbox_conf` | bbox_detector | Detection confidence score |
| `image_id` | bbox_detector | Source image id (from `metadata.name`) |
| `video_id` | bbox_detector | Source video id |
| `category_id` | bbox_detector | Fixed value 1 (person) |
| `track_bbox_ltwh` | track | Tracked bounding box (ltwh) |
| `track_bbox_conf` | track | Tracker output confidence |
| `track_id` | track (rewritten by pitch_gate, tracklet_split, traj_refine) | Persistent object identity; NaN off-pitch after pitch_gate; fragment id after split; final trajectory id after refine |
| `crop_single` | crop_filter | True iff `crop_rT <= thr_target` and `crop_rB < thr_other` (clean single-person crop) |
| `crop_rT` | crop_filter | Max over contaminators of `inter(T,B)/area(T)` |
| `crop_rB` | crop_filter | Max over contaminators of `inter(T,B)/area(B)` |
| `crop_trigger` | crop_filter | `detections.index` of the box that made T multi; NaN when single |
| `bbox_pitch` | calibration | Dict of bottom-left/right/middle pitch coords on Z=0 plane (m); None when uncalibrated |
| `parameters` | calibration | Per-frame sn-calibration camera parameter dict (on metadatas) |
| `track_id_pregate` | pitch_gate | Snapshot of `track_id` the gate received |
| `pitch_gate_offpitch` | pitch_gate | True on rows of an off-pitch tracklet |
| `pitch_mean_x` | pitch_gate | Tracklet mean x (m); NaN when untracked / no projection |
| `pitch_mean_y` | pitch_gate | Tracklet mean y (m); NaN when untracked / no projection |
| `track_id_presplit` | tracklet_split | Per-row copy of the incoming (pre-split) track id |
| `team_embedding` | team_embed | osnet_team descriptor on each sampled single row; cleared (None) by traj_refine |
| `team_cluster` | team_embed (unified by traj_refine) | 0/1 team cluster id per fragment; NaN when unclustered |
| `team_cluster_nearest` | team_embed | 0/1 nearest-centroid id per embedded fragment (diagnostic) |
| `team_cluster_prerefine` | traj_refine | Snapshot of `team_cluster` before the merge |
| `jersey_number_detection` | jersey_number_detect (unified by traj_refine) | Assigned jersey digit string, or None when unnumbered |
| `jersey_number_confidence` | jersey_number_detect (updated by traj_refine) | Winner's pooled vote share |
| `jersey_number_candidates` | jersey_number_detect | Per-tracklet pooled labels `[label, mx, conf_sum, votes]`, ranked by maxconf |
| `jersey_number_maxconf` | jersey_number_detect (updated by traj_refine) | Assigned number's maxconf score `exp(mx)*conf_sum` |
| `jersey_number_detection_prerefine` | traj_refine | Snapshot of the pre-refine jersey number |
| `jersey_number_confidence_prerefine` | traj_refine | Snapshot of the pre-refine jersey confidence |
| `track_id_prerefine` | traj_refine | Snapshot of the incoming (pre-refine) track id |
| `jersey_number` | tracklet_agg | Final per-tracklet voted jersey number (written to every row) |
| `role` | role_team | Player / goalkeeper / referee (on every tracked row of a trajectory) |
| `team` | role_team | Team side left / right; None for referees |
| `interpolated` | interpolation (disabled) | Boolean flag; True on synthesized gap rows (all False in production) |
