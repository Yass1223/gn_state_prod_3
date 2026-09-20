"""Tracklet splitting -- Stage 1 of the refinement method, and nothing else.

The pipeline's ONE merge lives in ``traj_refine`` (Stage 2); this module only
breaks tracklets that hold more than one identity into fragments. It performs
no fragment-to-fragment merging of any kind.

GHOST RULE. Multi-player (non-``crop_single``) detections are ghosts: they are
never embedded, they take no part in the clustering or in any centroid, and
they enter no condition anywhere in the pipeline until the final-trajectory
duplicate resolution inside ``traj_refine``. Here they are only ATTACHED to a
fragment, by geometry and time alone, so that they follow that fragment's
cluster through every later merge.

Per tracklet (inputs: unit appearance embeddings ``u = e/||e||`` of its SINGLE
detections -- multi detections carry zero features and are never used -- plus
the crop filter's ``crop_single`` label, the chronological frame index and the
image-space box of every detection):

1. DBSCAN(eps, min_samples) on the precomputed cosine-distance matrix of the
   SINGLE detections only, yielding fragments and noise points. A tracklet
   with fewer than ``max(2, min_samples)`` single detections is ONE fragment
   (of its single detections); a tracklet whose DBSCAN result is all noise is
   ONE fragment.
2. Every single noise detection is assigned to the fragment with the nearest
   centroid (centroids = mean unit embedding over the fragment's single
   non-zero detections; a fragment with no non-zero embedding has no centroid
   and cannot attract noise; a zero-embedding noise detection is at cosine
   distance 1 from everything and falls to the deterministic tie rule --
   lowest fragment label).
3. GHOST ATTACHMENT (appearance-free): every multi detection of the tracklet
   is attached to the fragment of the SAME tracklet whose nearest single
   detection is closest IN TIME to the ghost's frame; ties break on the
   image-space centre distance between the ghost's box and that nearest
   single detection's box, then on the lowest fragment label.
4. By construction every fragment of a mixed tracklet holds at least one
   single detection; an all-multi fragment cannot arise, because fragments
   are formed from single detections only.

ALL-MULTI TRACKLETS (zero single detections) are dissolved at video level,
after every other tracklet has been split: each of their detections is
assigned, individually, to the fragment -- of ANY tracklet -- whose nearest-
in-time single detection is closest in image space (box-centre distance) to
the detection's box; ties break on the time gap, then on the lowest fragment
id. These cross-assigned rows are the only way a fragment can hold rows from
more than one source tracklet, and the only way two detections of one
fragment can share a frame -- always multi rows, counted, never hidden. When
the video holds NO fragment at all (every tracklet all-multi), there is no
dissolution target: each all-multi tracklet is kept as ONE fragment of its
own (reported as ``allmulti_kept``).

Fragment ids follow the ``split_merge`` convention ``tid * FRAG_BASE + label``
so a fragment's source tracklet is recoverable by integer division
(cross-assigned ghost rows carry the TARGET fragment's id, which is the point).

Input invariant (the tracker's): one detection per (tracklet, frame). The
driver validates it and raises on violation; the one-per-frame guarantee of
the OUTPUT holds for single rows (fragments partition tracklets) and can be
broken only by cross-assigned multi rows, which stage 3 of ``traj_refine``
resolves on the final trajectories.
"""
import numpy as np
from sklearn.cluster import DBSCAN

FRAG_BASE = 10000


def _unit(E):
    n = np.linalg.norm(E, axis=1, keepdims=True)
    return np.where(n > 1e-9, E / np.maximum(n, 1e-9), 0.0)


def _centroid(U, rows):
    """Mean unit embedding over the non-zero rows of ``rows`` (single rows by
    construction -- ghosts are never handed to this function); None when no
    row has a non-zero embedding."""
    rows = np.asarray(rows, dtype=np.int64)
    if len(rows) == 0:
        return None
    nz = np.linalg.norm(U[rows], axis=1) > 1e-6
    if nz.any():
        return U[rows[nz]].mean(axis=0)
    return None


def _centers(boxes):
    """Image-space box centres of ``bbox_ltwh`` rows."""
    b = np.asarray(boxes, dtype=np.float64)
    return np.stack([b[:, 0] + b[:, 2] * 0.5, b[:, 1] + b[:, 3] * 0.5], axis=1)


def split_tracklet(U, single, frames, boxes, eps, min_samples):
    """One tracklet -> fragment label per detection (0..k-1), plus counts.

    ``U`` (n, d) unit embeddings (zero rows on the multi detections -- they are
    never read), ``single`` (n,) bool, ``frames`` (n,) int chronological frame
    index, ``boxes`` (n, 4) float ``bbox_ltwh``. Returns
    ``(labels, k, n_noise, n_ghosts)``. Must not be called on an all-multi
    tracklet (no single detection): those are dissolved at video level --
    ``split_video`` handles them.
    """
    n = len(U)
    single = np.asarray(single, dtype=bool)
    frames = np.asarray(frames, dtype=np.int64)
    s_idx = np.where(single)[0]
    if len(s_idx) == 0:
        raise ValueError("split_tracklet must not receive an all-multi "
                         "tracklet; split_video dissolves those")
    lab = np.full(n, -1, dtype=np.int64)

    # 1. DBSCAN over the single detections only
    if len(s_idx) < max(2, int(min_samples)):
        lab[s_idx] = 0
        n_noise = 0
    else:
        Us = U[s_idx]
        D = np.clip(1.0 - Us @ Us.T, 0.0, 2.0)
        ls = DBSCAN(eps=eps, min_samples=int(min_samples),
                    metric="precomputed").fit_predict(D).astype(np.int64)
        if not (ls >= 0).any():                      # all noise: one fragment
            ls[:] = 0
            n_noise = 0
        else:
            n_noise = int((ls == -1).sum())
        lab[s_idx] = ls

    clusters = sorted(int(c) for c in np.unique(lab[s_idx]) if c >= 0)
    cents = {c: _centroid(U, s_idx[lab[s_idx] == c]) for c in clusters}

    # 2. single noise -> nearest fragment centroid (tie: lowest label)
    for i in s_idx[lab[s_idx] == -1]:
        best, best_d = None, None
        for c in clusters:
            mu = cents[c]
            d = 1.0 if mu is None else float(1.0 - U[i] @ mu)
            if best is None or d < best_d - 1e-12:
                best, best_d = c, d
        lab[i] = best

    # 3. ghost attachment: appearance-free, time first, then space, then label
    ctr = _centers(boxes)
    m_idx = np.where(~single)[0]
    frag_sf = {c: np.sort(s_idx[lab[s_idx] == c]) for c in clusters}
    for i in m_idx:
        f = int(frames[i])
        best = None                       # (dt, dist, label)
        for c in clusters:
            rows = frag_sf[c]
            dts = np.abs(frames[rows] - f)
            j = int(np.argmin(dts))       # earliest row on a time-gap tie
            dt = int(dts[j])
            dist = float(np.hypot(*(ctr[i] - ctr[rows[j]])))
            key = (dt, dist, c)
            if best is None or key < best:
                best = key
        lab[i] = best[2]
    n_ghosts = int(len(m_idx))

    # compact relabel 0..k-1 preserving cluster order
    order = {c: k for k, c in enumerate(sorted(set(int(x) for x in lab)))}
    lab = np.array([order[int(x)] for x in lab], dtype=np.int64)
    return lab, len(order), n_noise, n_ghosts


def split_video(E, single, frames, track_ids, boxes, eps, min_samples):
    """All tracklets of one video.

    Aligned arrays over TRACKED detections: ``E`` (n, d) embeddings (zero rows
    on every multi detection and on failed single crops), ``single`` (n,)
    bool, ``frames`` (n,) int (equality == same frame), ``track_ids`` (n,)
    int, ``boxes`` (n, 4) float ``bbox_ltwh``. Raises on a duplicated
    (tracklet, frame) pair -- the tracker invariant.

    Returns ``(frag, per_tracklet, video_report)``: ``frag[i]`` the fragment
    id of row i (``tid * FRAG_BASE + label``; cross-assigned rows carry their
    TARGET fragment's id), one report entry per tracklet, and the video-level
    report of the all-multi dissolution
    (``allmulti_tracklets``, ``rows_cross_assigned``, ``allmulti_kept``).
    """
    E = np.asarray(E, dtype=np.float32)
    single = np.asarray(single, dtype=bool)
    frames = np.asarray(frames, dtype=np.int64)
    track_ids = np.asarray(track_ids, dtype=np.int64)
    boxes = np.asarray(boxes, dtype=np.float64)
    n = len(E)
    if not (len(single) == len(frames) == len(track_ids) == n
            and boxes.shape == (n, 4)):
        raise ValueError("E, single, frames, track_ids and boxes must have "
                         "one entry per detection")
    pairs = set()
    for t, f in zip(track_ids, frames):
        key = (int(t), int(f))
        if key in pairs:
            raise ValueError(f"tracklet {t} holds two detections in frame {f}; "
                             f"the tracker invariant is broken")
        pairs.add(key)

    U = _unit(E)
    ctr = _centers(boxes)
    frag = np.full(n, -1, dtype=np.int64)
    per_tracklet = []
    allmulti = []
    for tid in np.unique(track_ids):
        idx = np.where(track_ids == tid)[0]
        entry = dict(track_id=int(tid), n=int(len(idx)),
                     n_single=int(single[idx].sum()),
                     n_multi=int((~single[idx]).sum()),
                     k=0, noise=0, ghosts=0, allmulti=False)
        if not single[idx].any():
            entry["allmulti"] = True
            allmulti.append((int(tid), idx))
            per_tracklet.append(entry)
            continue
        lab, k, n_noise, n_ghosts = split_tracklet(
            U[idx], single[idx], frames[idx], boxes[idx], eps, min_samples)
        frag[idx] = int(tid) * FRAG_BASE + lab
        entry.update(k=int(k), noise=int(n_noise), ghosts=int(n_ghosts))
        per_tracklet.append(entry)

    # all-multi tracklets: dissolve across the video (space first at the
    # nearest-in-time single detection, then time gap, then fragment id)
    video_report = dict(allmulti_tracklets=[int(t) for t, _ in allmulti],
                        rows_cross_assigned=0, allmulti_kept=0)
    frag_ids = sorted(int(f) for f in np.unique(frag) if f >= 0)
    if allmulti and frag_ids:
        # single rows per fragment, sorted by frame, for the nearest-in-time
        # lookup (fragments hold their own tracklet's singles only here)
        by_frag = {}
        for fid in frag_ids:
            rows = np.where((frag == fid) & single)[0]
            order = np.argsort(frames[rows], kind="stable")
            by_frag[fid] = rows[order]
        for tid, idx in allmulti:
            for i in idx:
                f = int(frames[i])
                best = None               # (dist, dt, fid)
                for fid in frag_ids:
                    rows = by_frag[fid]
                    dts = np.abs(frames[rows] - f)
                    j = int(np.argmin(dts))
                    dt = int(dts[j])
                    dist = float(np.hypot(*(ctr[i] - ctr[rows[j]])))
                    key = (dist, dt, fid)
                    if best is None or key < best:
                        best = key
                frag[i] = best[2]
                video_report["rows_cross_assigned"] += 1
    elif allmulti:
        # no dissolution target anywhere: keep each all-multi tracklet as one
        # fragment of its own (deterministic, reported)
        for tid, idx in allmulti:
            frag[idx] = int(tid) * FRAG_BASE
            video_report["allmulti_kept"] += 1

    if (frag < 0).any():
        raise RuntimeError("a tracked row was left without a fragment; the "
                           "bookkeeping is broken")
    return frag, per_tracklet, video_report
