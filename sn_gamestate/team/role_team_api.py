"""Role and team SIDE per finished trajectory (``role_team`` stage).

Runs AFTER ``traj_refine``, on the pipeline's final trajectories -- roles and
sides are decided once, where the temporal and positional evidence is
strongest, and never gate the merge. The fragment-level team embeddings and
clusters are gone by this point (``traj_refine`` drops ``team_embedding``;
``team_cluster``/``team_cluster_nearest`` remain as inert snapshots and are
NOT read here): this stage recomputes appearance itself, from the CLEAN
(single) crops of each finished trajectory.

Per sequence (GHOST RULE: every statistic, every candidate condition and the
written labels use SINGLE (``crop_single``) detections only; multi-crop rows
contribute nothing and receive nothing):

1. Geometry statistics per trajectory over its SINGLE detections: mean/std of
   x and y, the 75th percentile of |x| (goal-depth cue), the sampled y-range,
   and the sampled position count ``n``, on the stride grid
   (``rules.sample_tracklet_rows``). The trajectory's jersey number (written
   by ``traj_refine`` on every row) is read as the first non-null value.
2. Appearance descriptor per trajectory: its SINGLE crops (``crop_single``)
   on the stride grid, at most ``crops_per_track`` evenly spaced, embedded
   with ``osnet_team``; descriptor = L2-normalised median of the embedded
   crops (float32). A trajectory with no single crop has no descriptor.
3. Outlier channels over the descriptors (this stage's own machinery,
   unrelated to the splitter's per-tracklet DBSCAN):
     * 2-means (the notebook's seeded k-means); ``d`` = Euclidean distance
       to the nearer centroid; the robust rule flags ``d > m + k*s``
       (m = median, s = MAD of d; disabled when ``s < 0.05*m``). ROBUST
       REFIT: the first fit's centroids are pulled by the very outliers the
       rule must catch, so after the first pass the two centroids are refit
       on the UNFLAGGED descriptors only, every trajectory is re-measured
       against them, and the rule is applied once more; the final rule
       flags are the UNION of the two passes (the refit can only add,
       never remove -- the recall objective). One deterministic iteration;
       skipped when the first pass flags nothing or fewer than two
       descriptors would remain.
     * DBSCAN on cosine distance over ALL descriptors together (eps = knee
       of the kth-neighbour curve times ``dbscan_scale``, ``min_samples`` =
       ``dbscan_min``); a trajectory DBSCAN labels noise is flagged. The
       channel is independent of the refit.
   A trajectory flagged by EITHER channel is an OUTLIER; the flagged
   trajectories are grouped (``outlier_group`` in the sidecar).
4. ASSISTANT referees FIRST, GEOMETRY FIRST: candidates carry NO jersey
   number, have at least ``min_n`` sampled positions, hug a touchline
   (|mean y| >= tau_a * max|mean y|, y-std <= tau_a_sy) AND carry a
   descriptor; one per side, no outlier condition on acceptance. The
   geometric best (largest |mean y|) wins outright when it leads the
   runner-up by at least ``a_tie_m``; otherwise, over the tied set (every
   candidate within ``a_tie_m`` of the best): with no MAD/DBSCAN-flagged
   candidate, the one farthest from both 2-means centroids; with flagged
   candidates, the geometric best among the flagged, and the farthest
   among them when they are again tied. A flagged winner is REMOVED from
   the outlier pool (geometry wins).
5. GOALKEEPERS, GEOMETRY FIRST: candidates satisfy q75(|x|) >= PEN_X and
   |mean y| <= PEN_Y, have at least ``min_n`` sampled positions, AND
   |mean x| > ``gk_rel`` * max|mean x| over all trajectories EXCEPT the
   assistant referees (assistants are assigned first, which is why), AND
   carry a descriptor; EXACTLY ONE per half, chosen by the same tie rule on
   depth |mean x| with margin ``gk_depth_m`` (there is no second-keeper
   confirmation channel). A flagged winner leaves the outlier pool; the
   other candidates are rejected.
6. MAIN referee (one per sequence): among the remaining OUTLIER
   trajectories with a descriptor, NO jersey number and at least ``min_n``
   sampled positions, the candidates whose sampled y-range
   stays inside the symmetric band of the trajectory means (2.14):

       max(y_ref) <= band * max_i(mean_y_i)   and
       min(y_ref) >= band * min_i(mean_y_i)

   (max/min over the means of ALL trajectories with y positions). Among the
   candidates, the one CLOSEST IN APPEARANCE (cosine distance of the
   descriptors) to the assistant referees; with no assistant found (or none
   with a descriptor), the outlier FARTHEST from both k-means centroids
   (largest distance to its nearer centroid) -- goalkeepers are already out
   of the pool.
7. Everything else is a PLAYER (a leftover outlier stays a player, tagged
   ``player_outlier``). Side: 2-means is refit on the NON-outlier player
   descriptors (all player descriptors when fewer than two clean ones);
   every player with a descriptor takes its nearest refit centroid's
   cluster; the two clusters are named left and right by the cue chain over
   the player trajectories' mean x (sign / quantile / mean vote, keeper cue
   available via ``side_rule``; the quantile cue averages the 20% and 80%
   percentiles). A player with no descriptor takes the side of its mean-x
   half (a flagged fallback). Goalkeepers take the side of their half;
   referees have no side.

Columns written on every tracked SINGLE row: ``role`` in {player,
goalkeeper, referee}; ``team`` in {left, right} (None for referees).
Multi-crop rows keep ``role``/``team`` None: labels are applied to single
crops only. Sidecar
``<audit_dir>/<sequence>.json``: per-trajectory role/why/team/outlier
flags, the outlier group, sequence-level naming cues, the (2.14) band, the
embedder provenance, and counts; the run audit reads it.
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from tracklab.pipeline.videolevel_module import VideoLevelModule
from tracklab.utils.cv2 import cv2_load_image

from sn_gamestate.reid import osnet_team
from sn_gamestate.team import rules
from sn_gamestate.team.team_embed_api import frame_index, sequence_name

log = logging.getLogger(__name__)

TEAM_NAMES = {0: "left", 1: "right"}
DEFAULTS = dict(k=3.25, dbscan_min=4, dbscan_scale=1.5,
                tau_a=0.9, tau_a_sy=3.0,
                side_rule="keeper", band=0.9, gk_depth_m=2.0, a_tie_m=1.5,
                min_n=10, gk_rel=0.85)


def _pitch_xy(bp):
    if isinstance(bp, dict):
        return float(bp.get("x_bottom_middle", np.nan)), float(bp.get("y_bottom_middle", np.nan))
    return np.nan, np.nan


def _cues(mxp, lab, myp=None, gk=None):
    """The side-naming cues over player trajectories: which cluster (0/1) is
    the LEFT team. ``mxp``/``myp`` mean x/y per player trajectory, ``lab`` its
    cluster id (0/1), ``gk`` the (mx, my) of the single goalkeeper when
    exactly one exists (the positional keeper cue). The quantile cue averages
    the 20 % and 80 % percentiles of each cluster's mean x."""
    def sign():
        v = [np.nansum(np.sign(mxp[lab == c])) for c in (0, 1)]
        return None if (not np.isfinite(v).all() or abs(v[0] - v[1]) < 1) else int(np.argmin(v))

    def mean():
        c = [np.nanmean(mxp[lab == k]) if (lab == k).any() else np.nan for k in (0, 1)]
        return None if (not np.isfinite(c).all() or c[0] == c[1]) else int(np.argmin(c))

    def quant():
        q = []
        for k in (0, 1):
            v = mxp[lab == k]
            v = v[np.isfinite(v)]
            q.append(np.mean([np.percentile(v, 20), np.percentile(v, 80)]) if len(v) >= 2 else np.nan)
        return None if (not np.isfinite(q).all() or q[0] == q[1]) else int(np.argmin(q))

    def keeper():
        if gk is None or not np.isfinite(gk[0]):
            return None
        kx = gk[0]
        ky = gk[1] if np.isfinite(gk[1]) else 0.0
        yy = myp if myp is not None else np.zeros_like(mxp)
        dist = []
        for k in (0, 1):
            sel = lab == k
            dd = np.hypot(mxp[sel] - kx, np.nan_to_num(yy[sel], nan=0.0) - ky)
            dist.append(np.nanmean(dd) if sel.any() else np.nan)
        if not np.isfinite(dist).all() or dist[0] == dist[1]:
            return None
        own = int(np.argmin(dist))
        return own if kx < 0 else 1 - own

    return dict(sign=sign(), mean=mean(), quantile=quant(), keeper=keeper())


def _pick_candidate(cand, fitness, d, flagged, tie_margin):
    """One winner among the geometry candidates of one half/side.

    ``cand``: candidate indices; ``fitness``: geometric fitness per index
    (larger = better; goalkeeper depth |mean x|, assistant |mean y|);
    ``d``: distance to the nearer 2-means centroid; ``flagged``: MAD-or-DBSCAN
    outlier flag per index; ``tie_margin``: geometric indistinguishability (m).

    The geometric best wins outright when it leads the runner-up by at least
    ``tie_margin``. Otherwise, over the TIED SET (every candidate within
    ``tie_margin`` of the best): with no flagged candidate, the one farthest
    from both centroids (largest ``d``); with flagged candidates, the
    geometric best among the flagged, and the farthest among them when the
    flagged are again within ``tie_margin`` of each other. When ``d`` is
    undefined (fewer than two descriptors in the sequence), the geometric
    best of the set. Returns (winner index, record for the sidecar)."""
    cand = np.asarray(cand, dtype=int)
    fit = np.asarray([fitness[j] for j in cand], dtype=float)
    order = np.argsort(-fit, kind="stable")
    cand, fit = cand[order], fit[order]

    def farthest(js):
        dd = np.asarray([d[j] for j in js], dtype=float)
        return int(js[int(np.nanargmax(dd))]) if np.isfinite(dd).any() else int(js[0])

    rec = dict(tie_margin=float(tie_margin), candidates=[int(j) for j in cand],
               tied=[], flagged=[], branch="geometry_clear")
    if len(cand) == 1 or fit[0] - fit[1] >= tie_margin:
        return int(cand[0]), rec
    tied = cand[fit >= fit[0] - tie_margin]
    rec["tied"] = [int(j) for j in tied]
    fl = np.asarray([j for j in tied if flagged[j]], dtype=int)
    rec["flagged"] = [int(j) for j in fl]
    if len(fl) == 0:
        rec["branch"] = "no_flag_farthest"
        return farthest(tied), rec
    fitf = np.asarray([fitness[j] for j in fl], dtype=float)
    top = fl[fitf >= fitf.max() - tie_margin]
    if len(top) == 1:
        rec["branch"] = "flag_geometry"
        return int(top[0]), rec
    rec["branch"] = "flag_tied_farthest"
    return farthest(top), rec


class RoleTeamAssignment(VideoLevelModule):
    input_columns = ["track_id", "image_id", "bbox_ltwh", "bbox_pitch",
                     "crop_single", "jersey_number_detection"]
    output_columns = ["role", "team"]

    def __init__(self, cfg, device=None, tracking_dataset=None, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.device = device if device is not None else "cpu"
        p = dict(DEFAULTS)
        p.update(dict(cfg.params) if getattr(cfg, "params", None) is not None else {})
        self.params = p
        self.stride = int(getattr(cfg, "pos_stride", rules.POS_STRIDE))
        self.crops_per_track = int(getattr(cfg, "crops_per_track", rules.CROPS_PER_TRK))
        self.batch_size = int(getattr(cfg, "batch_size", 128))
        self.audit_dir = Path(str(cfg.audit_dir)) if getattr(cfg, "audit_dir", None) else None
        if self.audit_dir:
            self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.model = None      # osnet_team, built on first use (weights at run time)
        log.info(f"[role_team] per-trajectory roles and sides AFTER traj_refine, "
                 f"recomputed from clean crops (osnet_team + 2-means/MAD + DBSCAN); "
                 f"assistants first (no jersey number), main referee no-number, "
                 f"goalkeeper depth > gk_rel * max non-assistant |mean x|; labels "
                 f"written on single rows only; params {self.params}")

    def _model(self):
        if self.model is None:
            self.model = osnet_team.from_config(self.cfg, self.device,
                                                batch_size=self.batch_size)
        return self.model

    # -------------------------------------------------------- descriptors --
    def _descriptors(self, tracked, metadatas, fidx, record):
        """One appearance descriptor per trajectory from its sampled single
        crops (the team_embed stage's convention: stride grid, at most
        ``crops_per_track`` evenly spaced, L2-normalised median, float32).
        Returns {track_id: descriptor}."""
        model = self._model()
        record["embedder"] = dict(model.info)
        id2path = {idx: str(p) for idx, p in metadatas["file_path"].items()}
        rows_by_image, row_tid = {}, {}
        for tid, grp in tracked.groupby("track_id"):
            sgrp = grp[grp["crop_single"].astype(bool)]
            if len(sgrp) == 0:
                record["trajectories_no_single"] += 1
                continue                       # no single crop -> no descriptor
            frames = fidx.reindex(sgrp["image_id"].to_numpy()).to_numpy()
            if np.isnan(frames.astype(float)).any():
                raise RuntimeError(f"[role_team] detection image_id without frame metadata")
            _, crop_rows, off_grid = rules.sample_tracklet_rows(
                frames, self.stride, self.crops_per_track)
            record["trajectories_off_grid"] += int(off_grid)
            for r in sgrp.index[crop_rows]:
                rows_by_image.setdefault(sgrp.at[r, "image_id"], []).append(r)
                row_tid[r] = tid
        record["crops_sampled"] = sum(len(v) for v in rows_by_image.values())
        emb_row = {}
        pending, pending_rows = [], []
        for image_id, rows in rows_by_image.items():
            path = id2path.get(image_id)
            if path is None:
                raise RuntimeError(f"[role_team] image_id {image_id} has no file_path")
            img = cv2_load_image(path)        # RGB, as osnet_team expects
            record["frames_read"] += 1
            for r in rows:
                crop = osnet_team.crop_rgb(img, tracked.at[r, "bbox_ltwh"])
                if crop.size == 0 or crop.shape[0] < 1 or crop.shape[1] < 1:
                    record["crops_empty"] += 1
                    continue
                pending.append(np.ascontiguousarray(crop))
                pending_rows.append(r)
            if len(pending) >= self.batch_size:
                for r, e in zip(pending_rows, model.embed(pending)):
                    emb_row[r] = e
                pending, pending_rows = [], []
        if pending:
            for r, e in zip(pending_rows, model.embed(pending)):
                emb_row[r] = e
        record["crops_embedded"] = len(emb_row)
        by_tid = {}
        for r, e in emb_row.items():
            if isinstance(e, np.ndarray) and e.size and np.isfinite(e).all() and np.any(e):
                by_tid.setdefault(row_tid[r], []).append(e)
        desc = {}
        for tid, es in by_tid.items():
            e = np.median(np.asarray(es, dtype=np.float32), axis=0)
            n = float(np.linalg.norm(e))
            if n < 1e-9:
                continue
            desc[tid] = (e / n).astype(np.float32)
        return desc

    # ------------------------------------------------------------------ main --
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        seq = sequence_name(metadatas)
        P = self.params
        out = detections.copy()
        out["role"] = None
        out["team"] = None
        record = dict(sequence=seq, params=dict(P), stride=self.stride,
                      crops_per_track=self.crops_per_track,
                      trajectories=0, trajectories_no_single=0,
                      trajectories_off_grid=0, crops_sampled=0,
                      crops_embedded=0, crops_empty=0, frames_read=0,
                      embedder=None, per_trajectory=[], sequence_level={})
        tracked = out.dropna(subset=["track_id"])
        if len(tracked) == 0:
            log.warning(f"[role_team] {seq}: no tracked detection")
            self._write(record)
            return out
        fidx = frame_index(metadatas)

        # --- 1. per-trajectory geometry (SINGLE rows only) + jersey number -
        T = []
        for tid, grp in tracked.groupby("track_id"):
            num = None
            if "jersey_number_detection" in grp.columns:
                nums = [v for v in grp["jersey_number_detection"]
                        if v is not None and not (isinstance(v, float) and np.isnan(v))
                        and str(v) not in ("", "-1", "None", "nan")]
                num = str(nums[0]) if nums else None
            sgrp = grp[grp["crop_single"].astype(bool)]
            if len(sgrp) == 0:
                T.append(dict(tid=float(tid), n=0, number=num,
                              mx=np.nan, sx=np.nan, q75=np.nan, my=np.nan,
                              sy=np.nan, ymax=np.nan, ymin=np.nan))
                continue
            frames = fidx.reindex(sgrp["image_id"].to_numpy()).to_numpy().astype(float)
            if np.isnan(frames).any():
                raise RuntimeError(f"[role_team] {seq}: detection image_id without frame metadata")
            pos_rows, _, _ = rules.sample_tracklet_rows(frames, self.stride, 1)
            pos = sgrp.iloc[pos_rows]
            xy = np.array([_pitch_xy(b) for b in pos["bbox_pitch"]], dtype=float).reshape(-1, 2)
            px, py = xy[:, 0], xy[:, 1]
            px, py = px[np.isfinite(px)], py[np.isfinite(py)]
            T.append(dict(
                tid=float(tid), n=int(len(pos_rows)), number=num,
                mx=px.mean() if len(px) else np.nan,
                sx=px.std() if len(px) else np.nan,
                q75=np.percentile(np.abs(px), 75) if len(px) else np.nan,
                my=py.mean() if len(py) >= 3 else np.nan,
                sy=py.std() if len(py) >= 3 else np.nan,
                ymax=py.max() if len(py) else np.nan,
                ymin=py.min() if len(py) else np.nan))
        record["trajectories"] = len(T)
        n = len(T)
        tids = [t["tid"] for t in T]
        mx = np.array([t["mx"] for t in T])
        my = np.array([t["my"] for t in T])
        sy = np.array([t["sy"] for t in T])
        q75 = np.array([t["q75"] for t in T])
        n_pos = np.array([t["n"] for t in T], dtype=int)
        no_num = np.array([t["number"] is None for t in T], dtype=bool)
        role = np.array(["player"] * n, dtype=object)
        why = np.array(["player"] * n, dtype=object)

        # --- 2. per-trajectory appearance descriptors (clean crops only) ---
        desc = self._descriptors(tracked, metadatas, fidx, record)
        has_e = np.array([tids[j] in desc for j in range(n)])
        e_idx = np.where(has_e)[0]
        E = (np.stack([desc[tids[j]] for j in e_idx])
             if len(e_idx) else np.zeros((0, 0), np.float32))

        # --- 3. outlier channels: 2-means + MAD rule, and global DBSCAN ----
        d = np.full(n, np.nan)                 # distance to the nearer centroid
        out_rule = np.zeros(n, bool)
        out_db = np.zeros(n, bool)
        m = s = None
        s_ok = False
        eps = None
        km = None
        dbscan_rec = dict(ran=False, n_desc=int(len(e_idx)), n_noise=0)
        mad_refit = dict(refit_ran=False, flags_first_pass=0, flags_refit=None,
                         flags_final=0, m_first=None, s_first=None,
                         s_ok_first=False, m_refit=None, s_refit=None,
                         s_ok_refit=None)
        if len(e_idx) >= 2:
            km = rules.kmeans2(E)
            d_all = np.linalg.norm(E[:, None] - km.cluster_centers_[None], axis=2)
            d[e_idx] = d_all.min(1)
            m = float(np.median(d[e_idx]))
            s = float(np.median(np.abs(d[e_idx] - m)))
            s_ok = s >= 0.05 * m
            if s_ok:
                out_rule[e_idx] = d[e_idx] > m + P["k"] * s
            mad_refit.update(flags_first_pass=int(out_rule.sum()),
                             m_first=_f(m), s_first=_f(s), s_ok_first=bool(s_ok))
            # Robust refit: the first fit's centroids are pulled by the very
            # outliers the rule must catch (their d shrinks, the MAD
            # inflates, the threshold rises). Refit the two centroids on the
            # UNFLAGGED descriptors only, re-measure EVERY trajectory
            # against them, apply the rule once more, and take the UNION of
            # the two passes (monotone: the refit can only add flags). One
            # deterministic iteration; skipped when nothing was flagged or
            # fewer than two descriptors would remain. The DBSCAN channel
            # below is independent of the refit.
            mask_keep = ~out_rule[e_idx]
            if out_rule.any() and int(mask_keep.sum()) >= 2:
                km_r = rules.kmeans2(E[mask_keep])
                d_all = np.linalg.norm(E[:, None] - km_r.cluster_centers_[None], axis=2)
                d[e_idx] = d_all.min(1)
                m = float(np.median(d[e_idx]))
                s = float(np.median(np.abs(d[e_idx] - m)))
                s_ok = s >= 0.05 * m
                refit_flags = np.zeros(n, bool)
                if s_ok:
                    refit_flags[e_idx] = d[e_idx] > m + P["k"] * s
                out_rule = out_rule | refit_flags
                mad_refit.update(refit_ran=True,
                                 flags_refit=int(refit_flags.sum()),
                                 m_refit=_f(m), s_refit=_f(s),
                                 s_ok_refit=bool(s_ok))
            mad_refit["flags_final"] = int(out_rule.sum())
        if len(e_idx) >= 4:
            # one DBSCAN over ALL descriptors together (the notebook's
            # channel): eps = knee of the kth-neighbour curve * dbscan_scale;
            # any trajectory labelled noise is flagged.
            eps = max(1e-3, rules.knee_eps(E) * P["dbscan_scale"])
            lab_db = DBSCAN(eps=eps, min_samples=int(P["dbscan_min"]),
                            metric="cosine").fit_predict(E)
            out_db[e_idx[lab_db == -1]] = True
            dbscan_rec.update(ran=True, n_noise=int((lab_db == -1).sum()))
        outlier = out_rule | out_db            # flagged by either channel
        in_pool = outlier.copy()               # the outlier group; geometry roles leave it

        # --- 4. assistants: geometry candidates + descriptor, one per side -
        absmy = np.abs(my)
        maxmy = np.nanmax(absmy) if np.isfinite(absmy).any() else np.nan
        assist_c = has_e & no_num & (n_pos >= int(P["min_n"])) \
            & np.isfinite(my) & (absmy >= P["tau_a"] * maxmy) \
            & (sy <= P["tau_a_sy"])
        n_geom_kept = 0
        assistant_selection = []
        for sgn in (-1, 1):
            idx = np.where(assist_c & (np.sign(my) == sgn))[0]
            if len(idx) == 0:
                continue
            j, sel = _pick_candidate(idx, absmy, d, outlier, P["a_tie_m"])
            role[j], why[j] = "referee", "assistant"
            if in_pool[j]:
                in_pool[j] = False             # geometry wins over the flag
                n_geom_kept += 1
            sel = {k: ([tids[i] for i in v] if isinstance(v, list) else v)
                   for k, v in sel.items()}
            assistant_selection.append(dict(side=int(sgn), winner=tids[j], **sel))

        # --- 5. goalkeepers: geometry candidates + descriptor, one per half.
        # Depth reference: max |mean x| over every trajectory EXCEPT the
        # assistant referees (assigned first, which is why); the candidate
        # must exceed gk_rel of it.
        absmx = np.abs(mx)
        nonass = why != "assistant"
        ref_pool = absmx[nonass & np.isfinite(mx)]
        gk_depth_ref = float(np.max(ref_pool)) if len(ref_pool) else np.nan
        depth_ok = (absmx > P["gk_rel"] * gk_depth_ref) if np.isfinite(gk_depth_ref) \
            else np.zeros(n, bool)
        gk_c = (role == "player") & has_e & (n_pos >= int(P["min_n"])) & \
            np.isfinite(mx) & (q75 >= rules.PEN_X) & (np.abs(my) <= rules.PEN_Y) & \
            depth_ok
        gk_selection = []
        for sgn in (-1, 1):
            side = np.where(gk_c & (np.sign(mx) == sgn))[0]
            if len(side) == 0:
                continue
            best, sel = _pick_candidate(side, absmx, d, outlier, P["gk_depth_m"])
            role[best], why[best] = "goalkeeper", "gk_extreme"
            if in_pool[best]:
                in_pool[best] = False
                n_geom_kept += 1
            for j in side:
                if j != best:
                    why[j] = "gk_rejected"
            sel = {k: ([tids[i] for i in v] if isinstance(v, list) else v)
                   for k, v in sel.items()}
            gk_selection.append(dict(half=int(sgn), winner=tids[best], **sel))

        # --- 6. MAIN referee: outlier + band (2.14) + appearance -----------
        means_y = my[np.isfinite(my)]
        band_hi = P["band"] * float(np.max(means_y)) if len(means_y) else np.nan
        band_lo = P["band"] * float(np.min(means_y)) if len(means_y) else np.nan
        ymax = np.array([t["ymax"] for t in T])
        ymin = np.array([t["ymin"] for t in T])
        cand = np.where(in_pool & (role == "player") & has_e & no_num
                        & (n_pos >= int(P["min_n"])) & np.isfinite(ymax)
                        & np.isfinite(ymin) & (ymax <= band_hi)
                        & (ymin >= band_lo))[0] if np.isfinite(band_hi) else np.array([], int)
        main_ref = None
        main_rule = None
        if len(cand):
            a_idx = np.where((why == "assistant") & has_e)[0]
            if len(a_idx):
                # cosine distance of the L2-normalised descriptors
                A = np.stack([desc[tids[j]] for j in a_idx])
                da = [float(np.mean(1.0 - A @ desc[tids[j]])) for j in cand]
                main_ref = int(cand[int(np.argmin(da))])
                main_rule = "closest_to_assistants"
            else:
                # no assistant: the outlier farthest from both centroids
                main_ref = int(cand[int(np.nanargmax(d[cand]))])
                main_rule = "farthest_from_clusters"
            role[main_ref], why[main_ref] = "referee", "main_2.14"
            in_pool[main_ref] = False

        # --- 7. sides: refit 2-means on clean players, name the clusters ---
        players = np.where(role == "player")[0]
        for j in players:
            if outlier[j]:
                why[j] = "player_outlier"      # leftover outlier stays a player
        lab = np.full(n, np.nan)
        left = None
        cues = dict(sign=None, mean=None, quantile=None, keeper=None)
        p_emb = players[has_e[players]]
        fit = p_emb[~outlier[p_emb]]
        if len(fit) < 2:
            fit = p_emb
        if len(fit) >= 2:
            km2 = rules.kmeans2(np.stack([desc[tids[j]] for j in fit]))
            for j in p_emb:
                dd = np.linalg.norm(km2.cluster_centers_ - desc[tids[j]], axis=1)
                lab[j] = float(int(np.argmin(dd)))
            gk_idx = np.where(role == "goalkeeper")[0]
            gk_pos = ((mx[gk_idx[0]], my[gk_idx[0]]) if len(gk_idx) == 1 else None)
            cues = _cues(mx[p_emb], lab[p_emb].astype(int), my[p_emb], gk_pos)
            if P["side_rule"] == "keeper":
                left = cues["keeper"]
                left = cues["quantile"] if left is None else left
                left = cues["mean"] if left is None else left
            else:                              # vote (default for other values)
                votes = [c for c in (cues["sign"], cues["quantile"], cues["mean"])
                         if c is not None]
                left = (0 if votes.count(0) > votes.count(1)
                        else 1 if votes.count(1) > votes.count(0)
                        else cues["mean"]) if votes else None
        if left is None:
            left = 0
        team = np.array([None] * n, dtype=object)
        n_fallback_half = 0
        for j in range(n):
            if role[j] == "referee":
                continue
            if role[j] == "goalkeeper":
                team[j] = "left" if (np.isfinite(mx[j]) and mx[j] < 0) else "right"
                continue
            if np.isfinite(lab[j]):
                team[j] = "left" if int(lab[j]) == left else "right"
            else:
                n_fallback_half += 1
                if why[j] == "player":
                    why[j] = "player_half_fallback"
                team[j] = "left" if (np.isfinite(mx[j]) and mx[j] < 0) else "right"

        # --- apply + sidecar (labels on SINGLE rows only: the ghost rule) --
        single_col = out["crop_single"].astype(bool) if "crop_single" in out.columns \
            else pd.Series(True, index=out.index)
        n_multi_unlabelled = 0
        for j, t in enumerate(T):
            sel = (out["track_id"] == t["tid"]) & single_col
            n_multi_unlabelled += int(((out["track_id"] == t["tid"]) & ~single_col).sum())
            out.loc[sel, "role"] = role[j]
            out.loc[sel, "team"] = team[j]
            record["per_trajectory"].append(dict(
                track_id=t["tid"], role=str(role[j]), why=str(why[j]),
                team=team[j], number=t["number"], has_embedding=bool(has_e[j]),
                cluster=(_f(lab[j]) if np.isfinite(lab[j]) else None),
                out_rule=bool(out_rule[j]), out_db=bool(out_db[j]),
                outlier=bool(outlier[j]), d=_f(d[j]),
                n=t["n"], mx=_f(t["mx"]), my=_f(t["my"]), sy=_f(t["sy"]),
                q75=_f(t["q75"]), ymax=_f(t["ymax"]), ymin=_f(t["ymin"])))
        roles = [r["role"] for r in record["per_trajectory"]]
        outlier_group = [tids[j] for j in np.where(outlier)[0]]
        record["sequence_level"] = dict(
            named_left_cluster=int(left), cues=cues,
            band=[_f(band_lo), _f(band_hi)],
            gk_depth_ref=_f(gk_depth_ref),
            n_multi_rows_unlabelled=int(n_multi_unlabelled),
            main_referee=(T[main_ref]["tid"] if main_ref is not None else None),
            main_referee_rule=main_rule,
            dbscan_eps=_f(eps), dbscan=dbscan_rec,
            gk_selection=gk_selection, assistant_selection=assistant_selection,
            mad_refit=mad_refit,
            distance_median=_f(m), distance_mad=_f(s),
            s_ok=bool(s_ok),
            outlier_group=outlier_group, n_outlier=len(outlier_group),
            n_outlier_geometry_kept=int(n_geom_kept),
            n_outlier_players=int(sum(1 for j in players if outlier[j])),
            n_player=roles.count("player"), n_goalkeeper=roles.count("goalkeeper"),
            n_referee=roles.count("referee"),
            n_no_embedding=int((~has_e).sum()),
            n_fallback_half=n_fallback_half,
            n_left=sum(1 for r in record["per_trajectory"] if r["team"] == "left"),
            n_right=sum(1 for r in record["per_trajectory"] if r["team"] == "right"))
        log.info(f"[role_team] {seq}: {n} trajectories -> "
                 f"{roles.count('player')} players, {roles.count('goalkeeper')} "
                 f"goalkeepers, {roles.count('referee')} referees "
                 f"(main {'found (' + str(main_rule) + ')' if main_ref is not None else 'none'}); "
                 f"{len(outlier_group)} outlier(s) ({n_geom_kept} kept by geometry); "
                 f"left cluster {left}, cues {cues}; {n_fallback_half} half fallback(s)")
        self._write(record)
        return out

    def _write(self, record):
        if self.audit_dir:
            (self.audit_dir / f"{record['sequence']}.json").write_text(
                json.dumps(record, indent=2, default=str))


def _f(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None
