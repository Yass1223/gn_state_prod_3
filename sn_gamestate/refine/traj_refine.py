"""Label-aware trajectory refinement -- the algorithm (``traj_refine`` stage).

Runs AFTER team clustering (``team_embed``) and jersey recognition and BEFORE
``role_team``, on the fragments the ``tracklet_split`` stage produced (and the
pitch gate kept): it merges fragments on team-cluster and jersey-number
evidence and resolves contradictory jersey-number claims.

Scope: every fragment (roles do not exist yet at this point of the pipeline;
``scope`` arrives per fragment and out-of-scope fragments never merge).

Inputs, per video (aligned arrays, one entry per TRACKED detection):

    E        (n, d) OSNet-AIN embeddings (unit rows; a zero row carries no
             signal), same checkpoint pin as the tracker and tracklet_split so
             the cosine-distance scale of ``tau`` transfers
    single   (n,)  bool, the crop filter's ``crop_single`` label
    frames   (n,)  int, CHRONOLOGICAL frame index (the dataset's ``frame``
             column; equality == same frame, order == time order)
    boxes    (n, 4) float, ``bbox_ltwh`` in image space (for the re-enter test)
    tids     (n,)  int, the fragment id each row carries when the stage runs

    tracks   {tid: dict(cluster, number, cand, scope)} per-fragment labels:
             ``cluster`` the team CLUSTER id (float) or None; ``number`` a
             digit string or None; ``cand`` the jersey stage's pooled
             candidate list ``[[label, mx, conf_sum, votes], ...]``; ``scope``
             bool
    img_w    image width in pixels, or None (re-enter checks become vacuous)

Jersey confidence model (the maxconf consolidation rule): a fragment's score
for label L is ``exp(mx(L)) * conf_sum(L)`` over the pooled frame decodes of
the two recognisers. When two fragments merge, the pooled statistics combine
exactly per that rule -- ``mx = max``, ``conf_sum``/``votes`` add -- which is
why the stage consumes the stats, not precomputed scores.

Merge conditions, identical in every phase.  Compatible(F, G) holds iff ALL of:

    C2  time overlap: the CLEAN frame sets are disjoint (multi-player
        detections are ignored until Stage 3, so they take no part in this
        test; collisions among them are expected and resolved there);
    C3  re-enter: when one cluster ends before the other begins, the frame
        HALF (whole-width rule, no margin) of the earlier cluster's last box
        and of the later cluster's first box must be equal; the condition is
        vacuous only for interleaved intervals or unknown image width;
    C4  labels, vacuous-when-unknown: team CLUSTER ids must agree when both
        are known, numbers must agree when both are known (two fragments with
        two different known numbers never merge).

The merging procedure, identical in every phase, is agglomerative (average
linkage, the splitter's distance convention: group distance = 1 minus the dot
product of the two mean unit vectors over clean detections): repeatedly the
minimum-distance compatible pair of the pool merges while its distance <=
tau; the merged cluster inherits the union of the known labels (compatibility
guarantees no contradiction), combines the candidate statistics, recomputes
its centroid sums, and only its row/column of the distance matrix is
re-evaluated.

Phases.  Fragments are pooled by the labels they ARRIVE with:

    S1  cluster known, number known
    S2  cluster known, number unknown
    S3  cluster unknown, number known
    S4  cluster unknown, number unknown

Phase 1 merges within S1, phase 2 within S2, phase 3 within S3, phase 4
within S4 -- each pool in isolation (no cross-pool pair is examined before
phase 5).  Phase 5 pools every cluster from phases 1-4, merged or not, and
runs the same procedure once more.  With all labels known inside S1, the C4
condition itself enforces same cluster id and same number in phase 1; inside
S2 it enforces same cluster id; inside S3 same number.

Number-conflict resolution (phases 1, 3 and 5, where same-number pairs are
examined): two clusters with time-overlapping CLEAN frames claiming one
number (cluster ids not contradicting) are two fragments claiming one shirt
at the same time; the pair with the highest JOINT pooled maxconf -- the
maxconf the merged cluster would carry, ``exp(max(mx_F, mx_G)) *
(conf_sum_F + conf_sum_G)`` -- is resolved first: the side with the LOWER
maxconf for that number is reassigned to its best-ranked candidate not yet
lost in a conflict (labels it lost on are banned, so a cascade of conflicts
walks strictly down its candidate list); a non-digit best candidate ("-1", or
nothing left) leaves it unnumbered.  Within a phase, conflicts are resolved
to a fixpoint, then the agglomerative pass runs; because a merge pools
candidate statistics and can surface a new conflict, the two steps repeat
until the pass merges nothing.  Termination: every action either removes a
cluster (merge) or shrinks a candidate list's unbanned prefix (conflict).

A cluster with no valid centroid (no clean detection with a non-zero
embedding, and no non-zero embedding at all) never merges, in any phase; it
can still lose a number conflict, which needs no appearance.

Determinism: every choice breaks ties on explicit keys ending in the cluster
key (the smallest source fragment id), so the output is a function of the
inputs alone.

Stage 3 -- duplicate-frame resolution, after the merger, in-scope clusters
only:

    3a  per trajectory, any frame holding more than one detection keeps the
        clean one when present (a second clean in the same frame is an anomaly
        -- counted, first kept, rest held), otherwise the multi-player
        detection closest to the trajectory's clean-first centroid; the rest
        go to a holding set.  Each trajectory's centroid is then recomputed
        over ALL its remaining detections, single and multi (this stage's
        placement metric only -- merger centroids stay clean-only).
    3b  held detections are processed in ascending distance to their nearest
        admissible trajectory (an in-scope trajectory whose frame is
        unoccupied): each is assigned there and the slot marked occupied;
        centroids stay fixed during 3b; a detection with no admissible
        trajectory is unassigned (it loses its trajectory id).

Output invariant: at most one detection per (frame, cluster) over ALL
detections -- clean disjointness guarantees it for clean rows, Stage 3
enforces it for the rest.
"""
import math

import numpy as np

DIGITS_1_99 = frozenset(str(i) for i in range(1, 100))


# ------------------------------------------------------------------ helpers

def score_of(cand, label):
    """maxconf score exp(mx) * conf_sum of ``label`` in a stats dict
    {label: [mx, conf_sum, votes]}; 0.0 for an unseen label."""
    s = cand.get(label)
    return math.exp(s[0]) * s[1] if s else 0.0


def pair_maxconf(ca, cb, label):
    """Joint pooled maxconf of ``label`` over two stats dicts, as if merged:
    ``exp(max(mx_a, mx_b)) * (conf_sum_a + conf_sum_b)`` -- identical to
    ``score_of(combine_cand(ca, cb), label)`` without building the merge.
    0.0 when neither side holds the label."""
    a, b = ca.get(label), cb.get(label)
    if a is None and b is None:
        return 0.0
    mx = max(a[0] if a else float("-inf"), b[0] if b else float("-inf"))
    cs = (a[1] if a else 0.0) + (b[1] if b else 0.0)
    return math.exp(mx) * cs


def combine_cand(a, b):
    """Combine two per-label stats dicts per the maxconf rule:
    mx = max, conf_sum and votes add."""
    out = {l: list(v) for l, v in a.items()}
    for l, v in b.items():
        if l in out:
            out[l] = [max(out[l][0], v[0]), out[l][1] + v[1], out[l][2] + v[2]]
        else:
            out[l] = list(v)
    return out


def ranked_labels(cand):
    """Labels of a stats dict, best first: (score, votes, label) descending --
    the emission-time tie rule minus per-label strength, which is not carried
    (score ties across distinct real trajectories are not expected; votes and
    the label string keep the order deterministic regardless)."""
    return sorted(cand, key=lambda l: (score_of(cand, l), cand[l][2], l),
                  reverse=True)


def edge_side(box, img_w):
    """'left' / 'right' by which HALF of the frame the box center lies in
    (whole-frame-width rule; no margin -- a side is always defined)."""
    l, _, w, _ = (float(v) for v in box)
    return "left" if (l + w / 2.0) < img_w / 2.0 else "right"


class _Cluster:
    """Mutable merge state of one (possibly merged) trajectory."""

    __slots__ = ("tids", "rows", "sum_clean", "n_clean", "sum_any", "n_any",
                 "clean_frames", "cluster", "number", "cand", "scope",
                 "first_row", "last_row", "banned")

    def __init__(self, tid, rows, E, single, frames, info):
        self.tids = [int(tid)]
        self.rows = np.asarray(rows, dtype=np.int64)
        nz = np.linalg.norm(E[self.rows], axis=1) > 1e-6
        clean = np.asarray(single, dtype=bool)[self.rows] & nz
        self.sum_clean = E[self.rows[clean]].sum(axis=0).astype(np.float64) \
            if clean.any() else np.zeros(E.shape[1], dtype=np.float64)
        self.n_clean = int(clean.sum())
        self.sum_any = E[self.rows[nz]].sum(axis=0).astype(np.float64) \
            if nz.any() else np.zeros(E.shape[1], dtype=np.float64)
        self.n_any = int(nz.sum())
        fr = np.asarray(frames, dtype=np.int64)[self.rows]
        sr = np.asarray(single, dtype=bool)[self.rows]
        self.clean_frames = set(int(x) for x in fr[sr])
        anchor = self.rows[sr] if sr.any() else self.rows
        afr = np.asarray(frames, dtype=np.int64)[anchor]
        self.first_row = int(anchor[int(np.argmin(afr))])
        self.last_row = int(anchor[int(np.argmax(afr))])
        cl = info.get("cluster")
        self.cluster = None if cl is None or (isinstance(cl, float) and np.isnan(cl)) else float(cl)
        number = info.get("number")
        self.number = str(number) if number not in (None, "", "-1") else None
        self.cand = {str(c[0]): [float(c[1]), float(c[2]), int(c[3])]
                     for c in (info.get("cand") or [])}
        self.scope = bool(info.get("scope"))
        self.banned = set()

    @property
    def key(self):
        return min(self.tids)

    def centroid(self):
        """Mean unit vector over clean rows, else over any non-zero rows, else
        None (the cluster then never merges)."""
        if self.n_clean:
            return self.sum_clean / self.n_clean
        if self.n_any:
            return self.sum_any / self.n_any
        return None

    def first_last(self, frames):
        return int(frames[self.first_row]), int(frames[self.last_row])

    def absorb(self, other, frames):
        self.tids += other.tids
        self.rows = np.concatenate([self.rows, other.rows])
        self.sum_clean += other.sum_clean
        self.n_clean += other.n_clean
        self.sum_any += other.sum_any
        self.n_any += other.n_any
        self.clean_frames |= other.clean_frames
        if int(frames[other.first_row]) < int(frames[self.first_row]):
            self.first_row = other.first_row
        if int(frames[other.last_row]) > int(frames[self.last_row]):
            self.last_row = other.last_row
        self.cluster = self.cluster if self.cluster is not None else other.cluster
        self.number = self.number or other.number
        self.cand = combine_cand(self.cand, other.cand)
        self.banned |= other.banned


# ------------------------------------------------------------------ conditions

def _dist(a, b):
    ca, cb = a.centroid(), b.centroid()
    if ca is None or cb is None:
        return float("inf")
    return 1.0 - float(ca @ cb)


def _reenter_ok(a, b, frames, boxes, img_w, record=None):
    """C3. Vacuous unless one cluster ends strictly before the other begins;
    then the exit and entry sides (frame HALVES, whole-width rule) must be
    equal."""
    if img_w is None:
        return True
    fa, la = a.first_last(frames)
    fb, lb = b.first_last(frames)
    if la < fb:
        earlier, later = a, b
    elif lb < fa:
        earlier, later = b, a
    else:
        return True                     # interleaved intervals: vacuous
    exit_side = edge_side(boxes[earlier.last_row], img_w)
    entry_side = edge_side(boxes[later.first_row], img_w)
    if record is not None:
        record.update(exit_side=exit_side, entry_side=entry_side)
    return exit_side == entry_side


def _labels_ok(a, b):
    """Label agreement for a merge: the team CLUSTER ids must not contradict
    (equal when both known; one or both unknown imposes no condition) and the
    jersey numbers must not contradict (equal when both known; unknown imposes
    no condition). Two same-cluster fragments with two different known numbers
    are two different players and never merge."""
    if a.cluster is not None and b.cluster is not None and a.cluster != b.cluster:
        return False
    if a.number is not None and b.number is not None and a.number != b.number:
        return False
    return True


# ------------------------------------------------------------------ conflicts

def _resolve_conflicts(clusters, pool, report, phase):
    """Number-conflict resolution over ``pool``: two in-scope clusters with
    time-overlapping CLEAN frames claiming one number (cluster ids not
    contradicting) are a conflict; the pair with the highest JOINT pooled
    maxconf is resolved first, and the scan repeats until no such pair
    remains. Mutates ``clusters``."""
    while True:
        live = sorted(k for k in pool if k in clusters)
        pairs = []
        for i, ka in enumerate(live):
            a = clusters[ka]
            if not a.scope or a.number is None:
                continue
            for kb in live[i + 1:]:
                b = clusters[kb]
                if not b.scope or b.number != a.number:
                    continue
                if (a.cluster is not None and b.cluster is not None
                        and a.cluster != b.cluster):
                    continue
                if not (a.clean_frames & b.clean_frames):
                    continue        # disjoint pairs belong to the merger
                pairs.append((pair_maxconf(a.cand, b.cand, a.number), ka, kb))
        if not pairs:
            return
        # descending JOINT pooled maxconf; key pair keeps equal scores deterministic
        sc, ka, kb = max(pairs, key=lambda p: (p[0], -p[1], -p[2]))
        _resolve_conflict(clusters[ka], clusters[kb], sc, report, phase)


def _resolve_conflict(a, b, pair_mc, report, phase):
    """Two overlapping clusters claim one number: the lower maxconf side walks
    to its best-ranked candidate it has not lost a conflict on."""
    n = a.number
    sa, sb = score_of(a.cand, n), score_of(b.cand, n)
    # deterministic loser on a perfect tie: the larger key (the smaller keeps)
    loser = b if (sb < sa or (sb == sa and b.key > a.key)) else a
    winner = a if loser is b else b
    loser.banned.add(n)
    new = None
    for lab in ranked_labels(loser.cand):
        if lab in loser.banned:
            continue
        if lab in DIGITS_1_99:
            new = lab
        break                       # the best unbanned label decides either way
    loser.number = new
    report["conflicts"].append(dict(
        phase=phase, number=n, pair=[a.key, b.key],
        winner=winner.key, winner_score=round(score_of(winner.cand, n), 6),
        loser=loser.key, loser_score=round(min(sa, sb), 6),
        reassigned_to=new, pair_maxconf=round(pair_mc, 6)))


# ------------------------------------------------------------------ merging

def _agglomerative(clusters, pool, frames, boxes, img_w, tau, use_reenter,
                   report, phase):
    """One agglomerative average-linkage pass over ``pool``: repeatedly merge
    the minimum-distance compatible pair (C2 clean-frame disjointness, C3
    re-enter, C4 vacuous-when-unknown label agreement) while its distance <=
    tau; every merge recomputes the surviving cluster's centroid sums and
    re-evaluates only its row/column of the distance matrix. Mutates
    ``clusters``; returns the number of merges."""
    keys = sorted(k for k in pool if k in clusters)
    k = len(keys)
    alive = np.ones(k, dtype=bool)
    n_merges = 0

    def compat(a, b):
        if not (a.scope and b.scope):
            return False
        if a.clean_frames & b.clean_frames:
            return False
        if not _labels_ok(a, b):
            return False
        if use_reenter and not _reenter_ok(a, b, frames, boxes, img_w):
            return False
        return True

    D = np.full((k, k), np.inf)
    for i in range(k):
        for j in range(i + 1, k):
            a, b = clusters[keys[i]], clusters[keys[j]]
            if compat(a, b):
                D[i, j] = D[j, i] = _dist(a, b)
    while alive.sum() > 1:
        sub = D.copy()
        sub[~alive, :] = np.inf
        sub[:, ~alive] = np.inf
        i, j = np.unravel_index(np.argmin(sub), sub.shape)
        if not np.isfinite(sub[i, j]) or sub[i, j] > tau:
            break
        ka, kb = keys[i], keys[j]
        if kb < ka:                       # keep the smaller key
            i, j, ka, kb = j, i, kb, ka
        a, b = clusters[ka], clusters[kb]
        report["merges"].append(dict(
            phase=phase, pair=[ka, kb],
            distance=round(float(sub[min(i, j), max(i, j)]), 4),
            cluster=(a.cluster if a.cluster is not None else b.cluster),
            number=a.number or b.number))
        a.absorb(b, frames)
        del clusters[kb]
        alive[j] = False
        D[j, :] = np.inf
        D[:, j] = np.inf
        n_merges += 1
        for m in range(k):
            if m != i and alive[m]:
                other = clusters[keys[m]]
                v = _dist(a, other) if compat(a, other) else np.inf
                D[i, m] = D[m, i] = v
    return n_merges


def _run_phase(clusters, pool, frames, boxes, img_w, tau, use_reenter,
               report, phase, with_conflicts):
    """One phase over ``pool``. Phases whose pool can hold known numbers (1, 3
    and 5) alternate number-conflict resolution to a fixpoint with an
    agglomerative pass, repeating while the pass still merges (a merge pools
    candidate statistics and can surface a new conflict); the other phases
    run the agglomerative pass once. Mutates ``clusters``."""
    if not with_conflicts:
        _agglomerative(clusters, pool, frames, boxes, img_w, tau,
                       use_reenter, report, phase)
        return
    while True:
        _resolve_conflicts(clusters, pool, report, phase)
        if not _agglomerative(clusters, pool, frames, boxes, img_w, tau,
                              use_reenter, report, phase):
            return


# ------------------------------------------------------------------ stage 3

def _stage3(clusters, E, single, frames, new_tid, report):
    """Duplicate-frame resolution (3a + 3b) over the in-scope clusters.
    Mutates ``new_tid`` in place: held detections keep their cluster only if
    re-placed there is impossible by construction, move to another in-scope
    cluster with a free slot, or become unassigned (-1). Returns the stage-3
    report dict. Centroids used: 3a keeps/holds against the cluster's
    clean-first centroid; 3b places against centroids over ALL of a
    trajectory's detections, recomputed every time the trajectory receives a
    held detection."""
    E = np.asarray(E, dtype=np.float32)
    single = np.asarray(single, dtype=bool)
    frames = np.asarray(frames, dtype=np.int64)
    in_scope = {k: c for k, c in clusters.items() if c.scope}

    held = []
    n_coll = 0
    clean_anomaly = 0
    kept_rows = {}                       # key -> list of remaining row indices
    for key, c in in_scope.items():
        by_frame = {}
        for r in c.rows:
            by_frame.setdefault(int(frames[r]), []).append(int(r))
        mu = c.centroid()
        remaining = []
        for f in sorted(by_frame):
            rr = sorted(by_frame[f])
            if len(rr) == 1:
                remaining.append(rr[0])
                continue
            n_coll += 1
            cleans = [r for r in rr if single[r]]
            if cleans:
                keep = cleans[0]
                clean_anomaly += len(cleans) - 1
            elif mu is None:
                keep = rr[0]
            else:
                keep = min(rr, key=lambda r: (1.0 - float(E[r] @ mu), r))
            remaining.append(keep)
            held.extend(r for r in rr if r != keep)
        kept_rows[key] = remaining

    # centroid over ALL remaining detections (single and multi), 3b's metric.
    # Recomputed for a trajectory EVERY time it receives a held detection, so
    # each assignment uses centroids that include everything assigned so far.
    mu_all = {}
    occupied = {}

    def _mu(rows):
        rows = np.asarray(rows, dtype=np.int64)
        nz = np.linalg.norm(E[rows], axis=1) > 1e-6
        return E[rows[nz]].mean(axis=0) if nz.any() else None

    for key, rows in kept_rows.items():
        occupied[key] = set(int(frames[r]) for r in rows)
        mu_all[key] = _mu(rows)

    placed = 0
    unassigned = []
    pending = sorted(held)
    while pending:
        best = None                      # (d, frame, row, key)
        dead = []
        for r in pending:
            f = int(frames[r])
            cand = None
            for key in sorted(occupied):
                if f in occupied[key]:
                    continue
                mu = mu_all[key]
                d = 1.0 if mu is None else 1.0 - float(E[r] @ mu)
                if cand is None or (d, key) < (cand[0], cand[3]):
                    cand = (d, f, r, key)
            if cand is None:
                dead.append(r)
            elif best is None or (cand[0], cand[1], cand[2]) < (best[0], best[1], best[2]):
                best = cand
        for r in dead:                   # occupancy only grows: never admissible again
            pending.remove(r)
            new_tid[r] = -1
            unassigned.append(int(r))
        if best is None:
            break
        d, f, r, key = best
        pending.remove(r)
        new_tid[r] = key
        occupied[key].add(f)
        kept_rows[key].append(int(r))
        mu_all[key] = _mu(kept_rows[key])   # dynamic: the trajectory grew
        placed += 1

    report["stage3"] = dict(collided_frames=n_coll, held=len(held),
                            clean_anomaly=clean_anomaly, placed=placed,
                            unassigned=len(unassigned),
                            unassigned_rows=unassigned)
    return report["stage3"]


# ------------------------------------------------------------------ driver

def refine_video(E, single, frames, boxes, tids, tracks, img_w,
                 tau, use_reenter=True):
    """Whole method for one video.

    Returns ``(new_tid_of_row, resolved, report)``:

    * ``new_tid_of_row`` (n,) int64 -- the cluster id (the smallest source
      trajectory id) each row belongs to after refinement;
    * ``resolved`` {cluster id: dict(tids, team, number, confidence, maxconf)}
      -- the label state of every final cluster.  ``confidence`` is the
      number's share of the cluster's pooled frame votes (the jersey stage's
      definition, extended to combined clusters), ``maxconf`` its combined
      maxconf score; both 0.0 with no number;
    * ``report`` -- the merge and conflict log of the five phases, pool sizes
      and counts for the audit sidecar.
    """
    E = np.asarray(E, dtype=np.float32)
    single = np.asarray(single, dtype=bool)
    frames = np.asarray(frames, dtype=np.int64)
    boxes = np.asarray(boxes, dtype=np.float64)
    tids = np.asarray(tids, dtype=np.int64)
    n = len(E)
    if not (len(single) == len(frames) == len(tids) == n and boxes.shape == (n, 4)):
        raise ValueError("E, single, frames, boxes and tids must have one entry "
                         "per detection")
    tau = float(tau)
    if not (0.0 <= tau <= 2.0):
        raise ValueError(f"tau must be in [0, 2], got {tau}")

    report = dict(merges=[], conflicts=[],
                  clusters_in=0, clusters_out=0, out_of_scope=0,
                  no_centroid=[], img_w=img_w)
    clusters = {}
    for tid in np.unique(tids):
        rows = np.where(tids == tid)[0]
        info = tracks.get(int(tid), {})
        c = _Cluster(tid, rows, E, single, frames, info)
        clusters[c.key] = c
        if not c.scope:
            report["out_of_scope"] += 1
        elif c.centroid() is None:
            report["no_centroid"].append(int(tid))
    report["clusters_in"] = len(clusters)

    # Pools by the labels the fragments ARRIVE with: S1 cluster+number known,
    # S2 cluster only, S3 number only, S4 neither. Phases 1-4 merge each pool
    # in isolation; phase 5 pools every cluster, merged or not.
    pools = {1: [], 2: [], 3: [], 4: []}
    for key, c in clusters.items():
        s = (1 if (c.cluster is not None and c.number is not None) else
             2 if c.cluster is not None else
             3 if c.number is not None else 4)
        pools[s].append(key)
    report["pools"] = {s: len(p) for s, p in pools.items()}
    report["clusters_after_phase"] = {}
    for phase in (1, 2, 3, 4):
        _run_phase(clusters, pools[phase], frames, boxes, img_w, tau,
                   use_reenter, report, phase, with_conflicts=phase in (1, 3))
        report["clusters_after_phase"][phase] = len(clusters)
    _run_phase(clusters, sorted(clusters), frames, boxes, img_w, tau,
               use_reenter, report, 5, with_conflicts=True)
    report["clusters_after_phase"][5] = len(clusters)
    report["clusters_out"] = len(clusters)

    new_tid = np.full(n, -2, dtype=np.int64)
    resolved = {}
    for key, c in clusters.items():
        new_tid[c.rows] = key
    if (new_tid == -2).any():
        raise RuntimeError("a tracked row was left without a cluster; the "
                           "bookkeeping is broken")
    _stage3(clusters, E, single, frames, new_tid, report)
    for key, c in clusters.items():
        total_votes = sum(v[2] for v in c.cand.values())
        number = c.number
        conf = (c.cand[number][2] / total_votes
                if number and number in c.cand and total_votes else 0.0)
        resolved[key] = dict(
            tids=sorted(c.tids), cluster=c.cluster, number=number,
            confidence=float(conf),
            maxconf=float(score_of(c.cand, number)) if number else 0.0)
    # Output invariant: one detection per (frame, cluster) over ALL detections.
    seen = set()
    for r in range(n):
        t = int(new_tid[r])
        if t < 0:
            continue
        key = (int(frames[r]), t)
        if key in seen:
            raise RuntimeError(f"frame collision after stage 3 at frame "
                               f"{key[0]}, cluster {key[1]}; the resolution "
                               f"is broken")
        seen.add(key)
    return new_tid, resolved, report
