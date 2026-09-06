"""Tracklet splitting -- Stage 1 of the refinement method, and nothing else.

The pipeline's ONE merge lives in ``traj_refine`` (Stage 2); this module only
breaks tracklets that hold more than one identity into fragments. It performs
no fragment-to-fragment merging of any kind.

EMBEDDING SCOPE AND THE GHOST RULE. EVERY tracked detection is embedded --
single (``crop_single``) and multi alike -- and the per-tracklet DBSCAN runs
over ALL of them, so multi crops take part in shaping the fragments they sit
in. The ghost rule is about CONDITIONS, and it is unchanged: fragment
centroids are means over SINGLE detections only, and multi detections enter
no merge condition anywhere in the pipeline until the final-trajectory
duplicate resolution inside ``traj_refine`` (stage 3, after team and role are
decided per trajectory).

Per tracklet with at least one single detection (inputs: unit appearance
embeddings ``u = e/||e||`` of ALL its detections -- zero rows only where a
crop failed to embed -- plus the crop filter's ``crop_single`` label):

1. DBSCAN(eps, min_samples) on the precomputed cosine-distance matrix over
   ALL detections (single and multi), yielding raw fragments and noise
   points. A tracklet with fewer than ``max(2, min_samples)`` detections is
   ONE fragment; a tracklet whose DBSCAN result is all noise is ONE fragment.
2. A raw fragment holding at least one single detection is a SINGLE-FRAGMENT;
   its centroid is the mean unit embedding over its single non-zero
   detections (multi members never enter a centroid). A raw fragment with NO
   single detection is an ALL-MULTI FRAGMENT and is DISSOLVED: each of its
   detections is assigned, individually, to the single-fragment of the SAME
   tracklet with the nearest centroid (cosine; a centroid-less fragment
   competes at distance 1; ties break on the lowest fragment label). When
   DBSCAN yields clusters but not one of them holds a single detection, the
   tracklet cannot anchor a centroid and stays ONE fragment (deterministic
   degenerate, like the all-noise case).
3. Every noise detection -- single or multi -- is assigned to the
   single-fragment with the nearest centroid, by the same rule.

By construction every fragment therefore holds at least one single detection.

ALL-MULTI TRACKLETS (zero single detections) are dissolved at video level,
after every other tracklet has been split: each of their detections is
assigned, individually, to the fragment -- of ANY tracklet -- with the
nearest centroid, where the video-level centroids are recomputed from the
FINAL fragment membership (mean unit embedding over the fragment's single
non-zero rows; ties break on the lowest fragment id). These cross-assigned
rows are the only way a fragment can hold rows from more than one source
tracklet, and the only way two detections of one fragment can share a frame
-- always multi rows, counted, never hidden. When the video holds NO fragment
at all (every tracklet all-multi), there is no dissolution target: each
all-multi tracklet is kept as ONE fragment of its own (reported as
``allmulti_kept``).

Fragment ids follow the ``split_merge`` convention ``tid * FRAG_BASE + label``
so a fragment's source tracklet is recoverable by integer division
(cross-assigned rows carry the TARGET fragment's id, which is the point).

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
    """Mean unit embedding over the non-zero rows of ``rows``; None when no
    row has a non-zero embedding. Callers pass SINGLE rows only -- the ghost
    rule keeps every centroid single-only."""
    rows = np.asarray(rows, dtype=np.int64)
    if len(rows) == 0:
        return None
    nz = np.linalg.norm(U[rows], axis=1) > 1e-6
    if nz.any():
        return U[rows[nz]].mean(axis=0)
    return None


def _nearest(U, i, labels, cents):
    """The label among ``labels`` whose centroid is nearest to row ``i``
    (cosine distance; a None centroid competes at distance 1.0; ties break
    on the lowest label -- iteration order plus a strict comparison)."""
    best, best_d = None, None
    for c in labels:
        mu = cents[c]
        d = 1.0 if mu is None else float(1.0 - U[i] @ mu)
        if best is None or d < best_d - 1e-12:
            best, best_d = c, d
    return best


def split_tracklet(U, single, eps, min_samples):
    """One tracklet -> fragment label per detection (0..k-1), plus counts.

    ``U`` (n, d) unit embeddings of ALL the tracklet's detections (zero rows
    only on failed crops), ``single`` (n,) bool. Returns ``(labels, k,
    n_noise, mf_rows, mf_frags)``: the final label per detection, the final
    fragment count, the number of noise detections attached (single and
    multi), the number of rows re-assigned out of dissolved all-multi
    fragments, and the number of all-multi fragments dissolved. Must not be
    called on an all-multi tracklet (no single detection): those are
    dissolved at video level -- ``split_video`` handles them.
    """
    n = len(U)
    single = np.asarray(single, dtype=bool)
    if not single.any():
        raise ValueError("split_tracklet must not receive an all-multi "
                         "tracklet; split_video dissolves those")
    lab = np.full(n, -1, dtype=np.int64)

    # 1. DBSCAN over ALL detections (single and multi)
    if n < max(2, int(min_samples)):
        lab[:] = 0
        return lab, 1, 0, 0, 0
    D = np.clip(1.0 - U @ U.T, 0.0, 2.0)
    ls = DBSCAN(eps=eps, min_samples=int(min_samples),
                metric="precomputed").fit_predict(D).astype(np.int64)
    clusters = sorted(int(c) for c in np.unique(ls) if c >= 0)
    sf = [c for c in clusters if single[ls == c].any()]
    if not sf:              # all noise, or no cluster holds a single detection
        lab[:] = 0
        return lab, 1, 0, 0, 0
    lab[:] = ls

    # 2. centroids: SINGLE non-zero members only (the ghost rule); all-multi
    #    fragments dissolve per detection to the nearest single-fragment
    cents = {c: _centroid(U, np.where((ls == c) & single)[0]) for c in sf}
    mf = [c for c in clusters if c not in sf]
    mf_rows = 0
    for c in mf:
        for i in np.where(ls == c)[0]:
            lab[i] = _nearest(U, i, sf, cents)
            mf_rows += 1

    # 3. noise (single or multi) -> nearest single-fragment centroid
    noise_rows = np.where(ls == -1)[0]
    for i in noise_rows:
        lab[i] = _nearest(U, i, sf, cents)

    # compact relabel 0..k-1 preserving cluster order
    order = {c: k for k, c in enumerate(sorted(set(int(x) for x in lab)))}
    lab = np.array([order[int(x)] for x in lab], dtype=np.int64)
    return lab, len(order), int(len(noise_rows)), int(mf_rows), int(len(mf))


def split_video(E, single, frames, track_ids, eps, min_samples):
    """All tracklets of one video.

    Aligned arrays over TRACKED detections: ``E`` (n, d) appearance
    embeddings of ALL detections (zero rows only on failed crops), ``single``
    (n,) bool, ``frames`` (n,) int (equality == same frame), ``track_ids``
    (n,) int. Raises on a duplicated (tracklet, frame) pair -- the tracker
    invariant.

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
    n = len(E)
    if not (len(single) == len(frames) == len(track_ids) == n):
        raise ValueError("E, single, frames and track_ids must have one "
                         "entry per detection")
    pairs = set()
    for t, f in zip(track_ids, frames):
        key = (int(t), int(f))
        if key in pairs:
            raise ValueError(f"tracklet {t} holds two detections in frame {f}; "
                             f"the tracker invariant is broken")
        pairs.add(key)

    U = _unit(E)
    frag = np.full(n, -1, dtype=np.int64)
    per_tracklet = []
    allmulti = []
    for tid in np.unique(track_ids):
        idx = np.where(track_ids == tid)[0]
        entry = dict(track_id=int(tid), n=int(len(idx)),
                     n_single=int(single[idx].sum()),
                     n_multi=int((~single[idx]).sum()),
                     k=0, noise=0, multifrag_rows=0, multifrag_dissolved=0,
                     allmulti=False)
        if not single[idx].any():
            entry["allmulti"] = True
            allmulti.append((int(tid), idx))
            per_tracklet.append(entry)
            continue
        lab, k, n_noise, mf_rows, mf_frags = split_tracklet(
            U[idx], single[idx], eps, min_samples)
        frag[idx] = int(tid) * FRAG_BASE + lab
        entry.update(k=int(k), noise=int(n_noise),
                     multifrag_rows=int(mf_rows),
                     multifrag_dissolved=int(mf_frags))
        per_tracklet.append(entry)

    # all-multi tracklets: dissolve across the video by APPEARANCE -- each
    # detection to the fragment with the nearest single-only centroid,
    # recomputed from the final fragment membership
    video_report = dict(allmulti_tracklets=[int(t) for t, _ in allmulti],
                        rows_cross_assigned=0, allmulti_kept=0)
    frag_ids = sorted(int(f) for f in np.unique(frag) if f >= 0)
    if allmulti and frag_ids:
        cents = {fid: _centroid(U, np.where((frag == fid) & single)[0])
                 for fid in frag_ids}
        for tid, idx in allmulti:
            for i in idx:
                frag[i] = _nearest(U, i, frag_ids, cents)
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
