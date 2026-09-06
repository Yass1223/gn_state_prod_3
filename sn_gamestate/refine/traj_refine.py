"""Label-aware trajectory refinement -- the algorithm (``traj_refine`` stage).

Runs AFTER ``team_embed`` and ``jersey_number_detect`` and BEFORE ``role_team``,
on the splitter's fragments: the pipeline's ONE merge, deciding on the team
CLUSTER id (``team_embed``), the jersey number with its pooled maxconf
candidate statistics (``jn_gsr_api``) and exit/entry geometry. Team sides and
roles do not exist yet -- they are assigned after this stage, on the finished
trajectories. Every fragment is in scope: there is no role to exempt anyone.

GHOST RULE. Multi-player (non-``crop_single``) detections take no part in any
merge condition: they are absent from the clean-frame disjointness test, from
the re-enter anchors, and from every centroid (centroids are means over CLEAN
non-zero detections ONLY -- there is no fallback to multi rows; a cluster
without such a detection has no centroid and never merges on appearance).
Ghost rows simply follow their fragment through every merge. The ONE place
they are acted on is stage 3, the final-trajectory duplicate resolution.

Inputs, per video (aligned arrays, one entry per TRACKED detection):

    E        (n, d) OSNet-AIN embeddings (unit rows; a zero row carries no
             signal), same checkpoint pin as the tracker and tracklet_split so
             the cosine-distance scale of ``tau`` transfers
    single   (n,)  bool, the crop filter's ``crop_single`` label
    frames   (n,)  int, CHRONOLOGICAL frame index (the dataset's ``frame``
             column; equality == same frame, order == time order)
    boxes    (n, 4) float, ``bbox_ltwh`` in image space (for the re-enter test)
    tids     (n,)  int, the trajectory id each row carries when the stage runs

    tracks   {tid: dict(cluster, number, cand, scope)} per-fragment labels:
             ``cluster`` the team_embed stage's TEAM CLUSTER id (float) or
             None; ``number`` a digit string or None; ``cand`` the jersey
             stage's pooled candidate list ``[[label, mx, conf_sum, votes],
             ...]``; ``scope`` bool (always True in this pipeline)
    img_w    image width in pixels, or None (re-enter checks become vacuous)

Jersey confidence model (the maxconf consolidation rule): a trajectory's score
for label L is ``exp(mx(L)) * conf_sum(L)`` over the pooled frame decodes of
the two recognisers. When two trajectories merge, the pooled statistics
combine exactly per that rule -- ``mx = max``, ``conf_sum``/``votes`` add.

THE MERGE RUNS IN THREE PHASES over a partition of the fragments by label
knowledge (a fragment with NO cluster id -- possible only for the splitter's
kept all-multi degenerates and for fragments whose sampled crops all failed to
embed -- belongs to no partition and NEVER merges):

    S1  cluster known AND number known
    S2  cluster known AND number unknown

Phase S1 -- within S1, label-driven, NO distance threshold.  All pairs with
EQUAL cluster and EQUAL number are processed in descending order of the pair's
JOINT pooled maxconf (``exp(max(mx_F, mx_G)) * (conf_sum_F + conf_sum_G)``):

  * CLEAN frame sets disjoint AND re-enter consistent -> MERGE (appearance
    plays no role: the two labels together already identify one player);
  * clean frame sets disjoint but re-enter fails -> the pair is set aside
    (re-examined if either side later changes through a merge);
  * clean frame sets overlap -> two trajectories claiming one shirt at the
    same time: the one with the LOWER maxconf for that number is reassigned to
    its best-ranked candidate not yet lost in a conflict (labels it lost on
    are banned, so a cascade of conflicts walks strictly down its candidate
    list); a non-digit best candidate ("-1", or nothing left) leaves it
    unnumbered -- it then LEAVES S1 and joins S2 for the next phase.
    The loop re-derives the pair set until no eligible pair remains, and
    terminates because every action either removes a cluster (merge), shrinks
    a candidate list's unbanned prefix (conflict), or grows the set-aside set
    (reject).

Phase S2 -- within S2 (including any fragment demoted from S1), agglomerative
average-linkage merging (group distance = 1 minus the dot product of the two
mean unit vectors over clean detections). Compatible(F, G) holds iff the
clusters are EQUAL, the CLEAN frame sets are disjoint and the re-enter
condition holds; the closest compatible pair merges while its distance <= tau.

Phase FINAL -- over ALL S1 and S2 survivors together, merged and unmerged
alike, agglomerative as in S2. Compatible(F, G) holds iff ALL of:

    C2  time overlap: the CLEAN frame sets are disjoint;
    C3  re-enter (nearest-side rule): when one cluster ends before the other
        begins, the earlier cluster's EXIT side is the lateral image edge its
        last single box is CLOSER to, and the later cluster's ENTRY side is
        the edge its first single box is closer to; the two sides must be
        equal. With interleaved intervals (or no image width) the condition
        is vacuous. There is no touch threshold: every ordered pair has a
        defined side (``edge_margin`` is accepted for compatibility, unused);
    C4  SAME cluster id (both are known by construction of the partition);
    C5  numbers: two DIFFERENT known numbers never merge (a numbered and an
        unnumbered cluster may).

The closest compatible pair merges while its distance <= tau; the merged
cluster inherits the union of the known labels, combines the candidate
statistics, and only its row/column of the distance matrix is re-evaluated.

Determinism: every choice breaks ties on explicit keys ending in the cluster
key (the smallest source trajectory id), so the output is a function of the
inputs alone.

Stage 3 -- duplicate-frame resolution, after the merger, in-scope clusters
only. This is the ONE exception to the ghost rule: final-trajectory multi
assignment IS appearance-based, no-overlap, with dynamic centroids:

    3a  per trajectory, any frame holding more than one detection keeps the
        clean one when present (a second clean in the same frame is an anomaly
        -- counted, first kept, rest held), otherwise the multi-player
        detection closest to the trajectory's clean-first centroid; the rest
        go to a holding set.  Each trajectory's centroid is then recomputed
        over ALL its remaining detections, single and multi (this stage's
        placement metric only -- merger centroids stay clean-only).
    3b  held detections are processed in ascending distance to their nearest
        admissible trajectory (an in-scope trajectory whose frame is
        unoccupied -- the NO-OVERLAP condition): each is assigned there and
        the slot marked occupied; the receiving trajectory's centroid is
        RECOMPUTED over all its detections after every assignment, so each
        placement sees everything assigned so far; a detection with no
        admissible trajectory is unassigned (it loses its trajectory id).

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


def edge_side(box, img_w, margin_px=None):
    """The lateral image edge the box is CLOSER to: 'left' when its centre
    sits in the left half of the image (ties to 'left'), else 'right'.
    ``margin_px`` is accepted for compatibility and unused -- the
    nearest-side rule has no touch threshold, so a side is always defined."""
    l, _, w, _ = (float(v) for v in box)
    return "left" if l + w * 0.5 <= img_w * 0.5 else "right"


class _Cluster:
    """Mutable merge state of one (possibly merged) trajectory."""

    __slots__ = ("tids", "rows", "sum_clean", "n_clean", "clean_frames",
                 "cluster", "number", "cand", "scope", "first_row", "last_row",
                 "banned")

    def __init__(self, tid, rows, E, single, frames, info):
        self.tids = [int(tid)]
        self.rows = np.asarray(rows, dtype=np.int64)
        nz = np.linalg.norm(E[self.rows], axis=1) > 1e-6
        clean = np.asarray(single, dtype=bool)[self.rows] & nz
        self.sum_clean = E[self.rows[clean]].sum(axis=0).astype(np.float64) \
            if clean.any() else np.zeros(E.shape[1], dtype=np.float64)
        self.n_clean = int(clean.sum())
        fr = np.asarray(frames, dtype=np.int64)[self.rows]
        sr = np.asarray(single, dtype=bool)[self.rows]
        self.clean_frames = set(int(x) for x in fr[sr])
        # re-enter anchors: SINGLE rows only (ghosts enter no condition); the
        # all-rows fallback exists only for clusters with no single row, which
        # carry no cluster id and never merge -- the anchor is then unused.
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
        """Mean unit vector over CLEAN non-zero rows; None otherwise (ghost
        rule: no fallback to multi rows -- such a cluster never merges on
        appearance)."""
        if self.n_clean:
            return self.sum_clean / self.n_clean
        return None

    def first_last(self, frames):
        return int(frames[self.first_row]), int(frames[self.last_row])

    def absorb(self, other, frames):
        self.tids += other.tids
        self.rows = np.concatenate([self.rows, other.rows])
        self.sum_clean += other.sum_clean
        self.n_clean += other.n_clean
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


def _reenter_ok(a, b, frames, boxes, img_w, margin_frac, record=None):
    """C3, nearest-side rule. Vacuous only for interleaved intervals (or no
    image width): when one cluster ends strictly before the other begins, the
    earlier cluster's exit side (the lateral edge its last single box is
    closer to) must equal the later cluster's entry side (the edge its first
    single box is closer to). ``margin_frac`` is unused (kept for the call
    signature)."""
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


def _in_s1(c):
    return c.scope and c.cluster is not None and c.number is not None


def _in_s2(c):
    return c.scope and c.cluster is not None and c.number is None


# ------------------------------------------------------------------ phase S1

def _phase_s1(clusters, frames, boxes, img_w, use_reenter, edge_margin,
              report):
    """Same-cluster same-number merges and overlap conflict resolution within
    S1; NO distance threshold. Mutates ``clusters`` (dict key -> cluster).
    See the module docstring."""
    aside = set()               # pair keys set aside on a re-enter reject
    while True:
        live = sorted(clusters)
        pairs = []
        for i, ka in enumerate(live):
            a = clusters[ka]
            if not _in_s1(a):
                continue
            for kb in live[i + 1:]:
                b = clusters[kb]
                if not _in_s1(b) or b.number != a.number:
                    continue
                if b.cluster != a.cluster:
                    continue
                if (ka, kb) in aside:
                    continue
                sc = pair_maxconf(a.cand, b.cand, a.number)
                pairs.append((sc, ka, kb))
        if not pairs:
            return
        # descending JOINT pooled maxconf; key pair keeps equal scores deterministic
        sc, ka, kb = max(pairs, key=lambda p: (p[0], -p[1], -p[2]))
        a, b = clusters[ka], clusters[kb]
        if a.clean_frames & b.clean_frames:
            _resolve_conflict(a, b, sc, report)
            continue
        entry = dict(phase="s1", pair=[ka, kb], number=a.number,
                     cluster=a.cluster, pair_maxconf=round(sc, 6))
        d = _dist(a, b)
        entry["distance"] = None if not np.isfinite(d) else round(d, 4)
        ok_re = (not use_reenter) or _reenter_ok(a, b, frames, boxes, img_w,
                                                 edge_margin, entry)
        if ok_re:
            a.absorb(b, frames)
            del clusters[kb]
            aside = {p for p in aside if ka not in p and kb not in p}
            report["merges"].append(entry)
        else:
            entry["rejected"] = "reenter"
            report["rejected_s1"].append(entry)
            aside.add((ka, kb))


def _resolve_conflict(a, b, pair_mc, report):
    """Two overlapping clusters claim one number: the lower maxconf side walks
    to its best-ranked candidate it has not lost a conflict on. An unnumbered
    loser leaves S1 and joins S2 for the next phase."""
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
        phase="s1", number=n, pair=[a.key, b.key],
        winner=winner.key, winner_score=round(score_of(winner.cand, n), 6),
        loser=loser.key, loser_score=round(min(sa, sb), 6),
        reassigned_to=new, pair_maxconf=round(pair_mc, 6)))


# ---------------------------------------------------- agglomerative phases

def _agglomerate(clusters, member, compat, frames, tau, phase, report):
    """Closest-pair average-linkage merging among ``member(c)`` clusters under
    ``compat`` and ``distance <= tau``. Mutates ``clusters``."""
    keys = sorted(k for k, c in clusters.items() if member(c))
    k = len(keys)
    alive = np.ones(k, dtype=bool)

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
        for m in range(k):
            if m != i and alive[m]:
                other = clusters[keys[m]]
                v = _dist(a, other) if compat(a, other) else np.inf
                D[i, m] = D[m, i] = v


def _phase_s2(clusters, frames, boxes, img_w, tau, use_reenter, edge_margin,
              report):
    """Agglomerative merging within S2 (cluster known, number unknown --
    including fragments demoted from S1): same cluster + C2 + C3 and tau."""
    def compat(a, b):
        if a.cluster != b.cluster:            # both known (S2 membership)
            return False
        if a.clean_frames & b.clean_frames:
            return False
        if use_reenter and not _reenter_ok(a, b, frames, boxes, img_w,
                                           edge_margin):
            return False
        return True

    _agglomerate(clusters, _in_s2, compat, frames, tau, "s2", report)


def _phase_final(clusters, frames, boxes, img_w, tau, use_reenter,
                 edge_margin, report):
    """Agglomerative merging over ALL S1 and S2 survivors together:
    C2 + C3 + same cluster (C4) + no contradicting numbers (C5) and tau."""
    def member(c):
        return c.scope and c.cluster is not None

    def compat(a, b):
        if a.cluster != b.cluster:            # C4: both known by construction
            return False
        if a.number is not None and b.number is not None \
                and a.number != b.number:     # C5: different numbers never merge
            return False
        if a.clean_frames & b.clean_frames:   # C2
            return False
        if use_reenter and not _reenter_ok(a, b, frames, boxes, img_w,
                                           edge_margin):
            return False                      # C3
        return True

    _agglomerate(clusters, member, compat, frames, tau, "final", report)


# ------------------------------------------------------------------ stage 3

def _stage3(clusters, E, single, frames, new_tid, report):
    """Duplicate-frame resolution (3a + 3b) over the in-scope clusters -- the
    ghost rule's one exception: final-trajectory multi assignment is
    appearance-based, no-overlap, with dynamic centroids. Mutates ``new_tid``
    in place: held detections move to another in-scope cluster with a free
    frame slot or become unassigned (-1). Returns the stage-3 report dict.
    Centroids used: 3a keeps/holds against the cluster's clean-first centroid;
    3b places against centroids over ALL of a trajectory's detections,
    recomputed every time the trajectory receives a held detection."""
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
                 tau, use_reenter=True, edge_margin=0.02):
    """Whole method for one video.

    Returns ``(new_tid_of_row, resolved, report)``:

    * ``new_tid_of_row`` (n,) int64 -- the cluster id (the smallest source
      trajectory id) each row belongs to after refinement;
    * ``resolved`` {cluster id: dict(tids, cluster, number, confidence,
      maxconf)} -- the label state of every final cluster.  ``confidence`` is
      the number's share of the cluster's pooled frame votes (the jersey
      stage's definition, extended to combined clusters), ``maxconf`` its
      combined maxconf score; both 0.0 with no number;
    * ``report`` -- merges (with their phase), conflicts, S1 rejections,
      partition counts and stage-3 counts for the audit sidecar.
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
    edge_margin = float(edge_margin)
    if not (0.0 <= edge_margin < 0.5):
        raise ValueError(f"edge_margin must be in [0, 0.5), got {edge_margin}")

    report = dict(merges=[], conflicts=[], rejected_s1=[],
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
    report["partition"] = dict(
        s1=sum(1 for c in clusters.values() if _in_s1(c)),
        s2=sum(1 for c in clusters.values() if _in_s2(c)),
        unclustered=sum(1 for c in clusters.values()
                        if c.scope and c.cluster is None))

    _phase_s1(clusters, frames, boxes, img_w, use_reenter, edge_margin,
              report)
    report["clusters_after_s1"] = len(clusters)
    _phase_s2(clusters, frames, boxes, img_w, tau, use_reenter, edge_margin,
              report)
    report["clusters_after_s2"] = len(clusters)
    _phase_final(clusters, frames, boxes, img_w, tau, use_reenter,
                 edge_margin, report)
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
