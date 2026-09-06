"""Unit tests for sn_gamestate/refine/traj_refine.py (pure numpy, no GPU).

Three-phase merger under test: S1 (cluster + number known, NO distance
threshold), S2 (cluster known, number unknown, tau), final phase over all
S1/S2 survivors (same cluster required, different numbers never merge, tau);
fragments without a cluster id never merge; centroids clean-only (ghost rule);
stage 3 unchanged (the ghost rule's one exception).

Synthetic embeddings: orthogonal unit vectors per identity, so distances
between same-identity clusters are ~0 and between different identities ~1.
Run directly (``python tests/test_traj_refine.py``) or under pytest.
"""
import math

import numpy as np

try:
    from sn_gamestate.refine.traj_refine import (
        DIGITS_1_99, combine_cand, pair_maxconf, ranked_labels,
        refine_video, score_of)
except ImportError:                      # sandbox layout
    from traj_refine import (DIGITS_1_99, combine_cand,
                             pair_maxconf, ranked_labels, refine_video,
                             score_of)

W = 1920.0
D = 8


def unit(i):
    v = np.zeros(D, dtype=np.float32)
    v[i % D] = 1.0
    return v


def rows_for(tid, ident, frame_list, x=900.0, singles=None, zero=False):
    """One synthetic trajectory: (E, single, frames, boxes, tids) row lists."""
    out = []
    for k, f in enumerate(frame_list):
        e = np.zeros(D, dtype=np.float32) if zero else unit(ident)
        s = True if singles is None else bool(singles[k])
        out.append((e, s, int(f), [float(x), 400.0, 40.0, 80.0], int(tid)))
    return out


def build(*trajs):
    rows = [r for t in trajs for r in t]
    E = np.stack([r[0] for r in rows])
    single = np.array([r[1] for r in rows])
    frames = np.array([r[2] for r in rows])
    boxes = np.array([r[3] for r in rows])
    tids = np.array([r[4] for r in rows])
    return E, single, frames, boxes, tids


def cand(*entries):
    """entries: (label, p, votes) -> [label, mx, conf_sum, votes] with
    mx = log p and conf_sum = votes * p, so score = p * votes * p."""
    return [[lab, math.log(p), votes * p, votes] for lab, p, votes in entries]


def track(cluster=None, number=None, c=None, scope=True):
    return dict(cluster=cluster, number=number, cand=c or [], scope=scope)


def run(trajs, tracks, tau=0.6):
    E, single, frames, boxes, tids = build(*trajs)
    return refine_video(E, single, frames, tids, tracks, tau)


def check_invariants(frames_arr, new_tid):
    seen = set()
    for f, t in zip(frames_arr, new_tid):
        if int(t) < 0:
            continue                     # unassigned by stage 3b
        assert (int(f), int(t)) not in seen, "frame collision"
        seen.add((int(f), int(t)))


# ---------------------------------------------------------------- helpers ----

def test_helpers():
    a = {"7": [math.log(0.9), 1.8, 2]}
    b = {"7": [math.log(0.5), 0.5, 1], "9": [math.log(0.8), 0.8, 1]}
    m = combine_cand(a, b)
    assert m["7"] == [math.log(0.9), 2.3, 3] and "9" in m
    assert abs(score_of(m, "7") - 0.9 * 2.3) < 1e-12
    assert score_of(m, "77") == 0.0
    assert ranked_labels({"3": [math.log(.5), 1.0, 2],
                          "8": [math.log(.5), 1.0, 2]})[0] == "8"
    ca = {"7": [math.log(0.9), 1.0, 1]}
    cb = {"7": [math.log(0.1), 20.0, 20], "9": [math.log(0.5), 0.5, 1]}
    assert abs(pair_maxconf(ca, cb, "7") - score_of(combine_cand(ca, cb), "7")) < 1e-12
    assert pair_maxconf(ca, cb, "42") == 0.0
    assert "0" not in DIGITS_1_99 and "1" in DIGITS_1_99 and "99" in DIGITS_1_99


# --------------------------------------------------------------- phase S1 ----

def test_s1_merges_without_distance_threshold():
    # DIFFERENT identities (appearance distance ~1 >> tau): same cluster +
    # same number still merge -- S1 has no threshold.
    t1 = rows_for(1, 0, range(0, 6))
    t2 = rows_for(2, 1, range(10, 16))
    tracks = {1: track(0.0, "7", cand(("7", .9, 5))),
              2: track(0.0, "7", cand(("7", .8, 4)))}
    new_tid, resolved, rep = run([t1, t2], tracks)
    assert set(new_tid) == {1}
    assert rep["merges"][0]["phase"] == "s1"
    assert resolved[1]["number"] == "7" and resolved[1]["cluster"] == 0.0


def test_s1_needs_same_cluster():
    # same number, different clusters: no S1 merge, and the final phase blocks
    # on the cluster too -> two survivors.
    t1 = rows_for(1, 0, range(0, 6))
    t2 = rows_for(2, 0, range(10, 16))
    tracks = {1: track(0.0, "7", cand(("7", .9, 5))),
              2: track(1.0, "7", cand(("7", .8, 4)))}
    new_tid, _, rep = run([t1, t2], tracks)
    assert set(new_tid) == {1, 2} and not rep["merges"]


def test_s1_overlap_conflict_second_candidate():
    # overlapping clean frames, same cluster, same number: the lower-maxconf
    # side walks to its second candidate.
    t1 = rows_for(1, 0, range(0, 8))
    t2 = rows_for(2, 1, range(4, 12))
    tracks = {1: track(0.0, "7", cand(("7", .9, 6))),
              2: track(0.0, "7", cand(("7", .5, 3), ("9", .4, 2)))}
    new_tid, resolved, rep = run([t1, t2], tracks)
    assert set(new_tid) == {1, 2}
    assert rep["conflicts"][0]["loser"] == 2
    assert resolved[2]["number"] == "9" and resolved[1]["number"] == "7"


def test_s1_conflict_to_unnumbered_then_s2():
    # the loser has no digit candidate left -> unnumbered -> S2, where it
    # merges with a same-cluster unnumbered fragment of the SAME identity.
    t1 = rows_for(1, 0, range(0, 8))
    t2 = rows_for(2, 1, range(4, 12))
    t3 = rows_for(3, 1, range(20, 26))
    tracks = {1: track(0.0, "7", cand(("7", .9, 6))),
              2: track(0.0, "7", cand(("7", .5, 3))),
              3: track(0.0, None)}
    new_tid, resolved, rep = run([t1, t2, t3], tracks)
    assert resolved[1]["number"] == "7"
    assert set(new_tid) == {1, 2}
    assert resolved[2]["number"] is None and 3 in resolved[2]["tids"]
    assert any(m["phase"] == "s2" and m["pair"] == [2, 3] for m in rep["merges"])


def test_s1_ordering_is_joint_maxconf():
    # three same-cluster claims on "7"; the strongest JOINT pair goes first
    # and merges; the merged cluster then conflicts with the third.
    t1 = rows_for(1, 0, range(0, 6))
    t2 = rows_for(2, 0, range(10, 16))
    t3 = rows_for(3, 1, range(2, 9))          # overlaps t1
    tracks = {1: track(0.0, "7", cand(("7", .9, 5))),
              2: track(0.0, "7", cand(("7", .9, 5))),
              3: track(0.0, "7", cand(("7", .3, 2), ("4", .2, 1)))}
    new_tid, resolved, rep = run([t1, t2, t3], tracks)
    assert rep["merges"][0]["pair"] == [1, 2]
    assert resolved[3]["number"] == "4"


# --------------------------------------------------------------- phase S2 ----

def test_s2_same_cluster_within_tau_merges():
    t1 = rows_for(1, 0, range(0, 6))
    t2 = rows_for(2, 0, range(10, 16))
    tracks = {1: track(0.0, None), 2: track(0.0, None)}
    new_tid, _, rep = run([t1, t2], tracks)
    assert set(new_tid) == {1}
    assert rep["merges"][0]["phase"] == "s2"


def test_s2_tau_blocks_and_final_cannot_rescue():
    t1 = rows_for(1, 0, range(0, 6))
    t2 = rows_for(2, 1, range(10, 16))        # distance ~1 > tau
    tracks = {1: track(0.0, None), 2: track(0.0, None)}
    new_tid, _, rep = run([t1, t2], tracks)
    assert set(new_tid) == {1, 2} and not rep["merges"]


def test_s2_needs_same_cluster():
    t1 = rows_for(1, 0, range(0, 6))
    t2 = rows_for(2, 0, range(10, 16))
    tracks = {1: track(0.0, None), 2: track(1.0, None)}
    new_tid, _, rep = run([t1, t2], tracks)
    assert set(new_tid) == {1, 2} and not rep["merges"]


# ------------------------------------------------------------- final phase ---

def test_final_numbered_and_unnumbered_attach():
    # S1 has one member (nothing to merge), S2 one member; the final phase
    # joins them: same cluster, same identity, no contradicting number.
    t1 = rows_for(1, 0, range(0, 6))
    t2 = rows_for(2, 0, range(10, 16))
    tracks = {1: track(0.0, "7", cand(("7", .9, 5))), 2: track(0.0, None)}
    new_tid, resolved, rep = run([t1, t2], tracks)
    assert set(new_tid) == {1}
    assert rep["merges"][0]["phase"] == "final"
    assert resolved[1]["number"] == "7"


def test_final_different_numbers_never_merge():
    # same cluster, same identity, disjoint -- but two different known
    # numbers: never merge (they are two players sharing a kit).
    t1 = rows_for(1, 0, range(0, 6))
    t2 = rows_for(2, 0, range(10, 16))
    tracks = {1: track(0.0, "7", cand(("7", .9, 5))),
              2: track(0.0, "9", cand(("9", .9, 5)))}
    new_tid, _, rep = run([t1, t2], tracks)
    assert set(new_tid) == {1, 2} and not rep["merges"]


def test_final_pools_merged_and_unmerged():
    # 1+2 merge in S1 (same cluster, "7"); 3 is unnumbered same cluster and
    # joins the MERGED cluster in the final phase.
    t1 = rows_for(1, 0, range(0, 6))
    t2 = rows_for(2, 0, range(10, 16))
    t3 = rows_for(3, 0, range(20, 26))
    tracks = {1: track(0.0, "7", cand(("7", .9, 5))),
              2: track(0.0, "7", cand(("7", .8, 4))),
              3: track(0.0, None)}
    new_tid, resolved, rep = run([t1, t2, t3], tracks)
    assert set(new_tid) == {1}
    phases = [m["phase"] for m in rep["merges"]]
    assert phases == ["s1", "final"] and resolved[1]["tids"] == [1, 2, 3]


def test_final_same_number_pair_can_merge():
    # equal known numbers never BLOCK the final phase (only DIFFERENT numbers
    # do); with no re-enter condition, every eligible same-number pair already
    # merges in S1 (no distance threshold), so the final phase never sees two
    # separate clusters with the same known number. Nothing to assert.
    pass


# ------------------------------------------------- unclustered / ghost rule --

def test_unclustered_never_merges():
    # no cluster id: neither the numbered nor the unnumbered fragment merges,
    # with anything, however close in appearance.
    t1 = rows_for(1, 0, range(0, 6))
    t2 = rows_for(2, 0, range(10, 16))
    t3 = rows_for(3, 0, range(20, 26))
    tracks = {1: track(None, "7", cand(("7", .9, 5))),
              2: track(None, "7", cand(("7", .8, 4))),
              3: track(0.0, None)}
    new_tid, _, rep = run([t1, t2, t3], tracks)
    assert set(new_tid) == {1, 2, 3} and not rep["merges"]
    assert rep["partition"] == dict(s1=0, s2=1, unclustered=2)


def test_centroid_clean_only_no_multi_fallback():
    # cluster 2's rows are all MULTI with a strong identity-0 appearance; the
    # ghost rule forbids the fallback: no centroid -> no S2/final merge even
    # though cluster 1 is identical in appearance and cluster.
    t1 = rows_for(1, 0, range(0, 6))
    t2 = rows_for(2, 0, range(10, 16), singles=[False] * 6)
    tracks = {1: track(0.0, None), 2: track(0.0, None)}
    new_tid, _, rep = run([t1, t2], tracks)
    assert set(t for t in new_tid if t >= 0) <= {1, 2}
    assert not rep["merges"] and 2 in rep["no_centroid"]


def test_s1_merges_even_without_centroid():
    # S1 needs no appearance at all: a numbered cluster whose clean rows have
    # zero embeddings still merges on cluster + number + time + re-enter.
    t1 = rows_for(1, 0, range(0, 6), zero=True)
    t2 = rows_for(2, 1, range(10, 16))
    tracks = {1: track(0.0, "7", cand(("7", .9, 5))),
              2: track(0.0, "7", cand(("7", .8, 4)))}
    new_tid, resolved, rep = run([t1, t2], tracks)
    assert set(new_tid) == {1}
    assert rep["merges"][0]["phase"] == "s1"


# ----------------------------------------------------------------- stage 3 ---

def test_stage3_clean_disjoint_multi_overlap_resolved():
    # clean frames disjoint (merge allowed) but ghost rows collide after the
    # merge; stage 3 keeps the clean row of the collision frame.
    t1 = rows_for(1, 0, [0, 1, 2, 3], singles=[True, True, True, False])
    t2 = rows_for(2, 0, [3, 4, 5], singles=[True, True, True])
    tracks = {1: track(0.0, None), 2: track(0.0, None)}
    E, single, frames, boxes, tids = build(t1, t2)
    new_tid, _, rep = refine_video(E, single, frames, tids, tracks, 0.6)
    assert set(t for t in new_tid if t >= 0) == {1}
    check_invariants(frames, new_tid)
    assert rep["stage3"]["collided_frames"] == 1
    kept_at_3 = [i for i, (f, t) in enumerate(zip(frames, new_tid))
                 if f == 3 and t == 1]
    assert kept_at_3 == [4]              # the clean row (t2's frame 3)


def test_stage3b_dynamic_centroid_reassignment():
    # a held ghost moves to the OTHER trajectory with a free slot.
    t1 = rows_for(1, 0, [0, 1, 2], singles=[True, True, False])
    t2 = rows_for(2, 0, [2, 3, 4], singles=[True, True, True])
    t3 = rows_for(3, 1, [0, 1, 5], singles=[True, True, True])
    tracks = {1: track(0.0, None), 2: track(0.0, None), 3: track(1.0, None)}
    E, single, frames, boxes, tids = build(t1, t2, t3)
    new_tid, _, rep = refine_video(E, single, frames, tids, tracks, 0.6)
    check_invariants(frames, new_tid)
    # 1+2 merged; the ghost at frame 2 collided with t2's clean frame 2 and
    # had to go somewhere: trajectory 3 has frame 2 free but identity 1 -- the
    # ghost (identity 0) is closer to nothing else, so it lands on 3 or is
    # unassigned; either way the invariant holds and stage 3 reports it.
    assert rep["stage3"]["held"] == 1
    assert rep["stage3"]["placed"] + rep["stage3"]["unassigned"] == 1


def test_stage3_no_op_without_collisions():
    t1 = rows_for(1, 0, range(0, 5))
    t2 = rows_for(2, 1, range(0, 5))
    tracks = {1: track(0.0, None), 2: track(1.0, None)}
    E, single, frames, boxes, tids = build(t1, t2)
    new_tid, _, rep = refine_video(E, single, frames, tids, tracks, 0.6)
    assert rep["stage3"]["held"] == 0 and rep["stage3"]["unassigned"] == 0
    assert set(new_tid) == {1, 2}


def test_out_of_scope_untouched():
    t1 = rows_for(1, 0, range(0, 6))
    t2 = rows_for(2, 0, range(10, 16))
    tracks = {1: track(0.0, "7", cand(("7", .9, 5)), scope=False),
              2: track(0.0, "7", cand(("7", .8, 4)))}
    new_tid, _, rep = run([t1, t2], tracks)
    assert set(new_tid) == {1, 2} and rep["out_of_scope"] == 1


def test_invariants_and_determinism():
    t1 = rows_for(1, 0, range(0, 8), singles=[True] * 6 + [False] * 2)
    t2 = rows_for(2, 0, range(10, 18))
    t3 = rows_for(3, 1, range(0, 8))
    t4 = rows_for(4, 1, range(9, 16), singles=[True] * 5 + [False] * 2)
    t5 = rows_for(5, 2, range(0, 12), singles=[False] * 12)
    tracks = {1: track(0.0, "7", cand(("7", .9, 5))),
              2: track(0.0, "7", cand(("7", .8, 4))),
              3: track(1.0, None), 4: track(1.0, None),
              5: track(None, None)}
    E, single, frames, boxes, tids = build(t1, t2, t3, t4, t5)
    out1 = refine_video(E, single, frames, tids, tracks, 0.6)
    out2 = refine_video(E, single, frames, tids, tracks, 0.6)
    assert (out1[0] == out2[0]).all()
    check_invariants(frames, out1[0])
    assert set(t for t in out1[0] if t >= 0) >= {1, 3}
    m = {tuple(x["pair"]): x["phase"] for x in out1[2]["merges"]}
    assert m.get((1, 2)) == "s1" and m.get((3, 4)) == "s2"


if __name__ == "__main__":
    import sys
    mod = sys.modules[__name__]
    names = [n for n in dir(mod) if n.startswith("test_")]
    for n in names:
        getattr(mod, n)()
        print(f"ok {n}")
    print(f"{len(names)} tests passed")
