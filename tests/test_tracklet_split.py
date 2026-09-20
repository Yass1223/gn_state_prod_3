"""Unit tests for sn_gamestate/track/tracklet_split.py (pure numpy, no GPU).

New semantics under test: DBSCAN over SINGLE detections only; multi detections
are unembedded ghosts attached by time (within a tracklet) or space (all-multi
tracklets, dissolved across the video); every fragment of a mixed tracklet
holds a single detection.

Run directly (``python tests/test_tracklet_split.py``) or under pytest.
"""
import numpy as np

try:
    from sn_gamestate.track.tracklet_split import (FRAG_BASE, split_tracklet,
                                                   split_video, _unit)
except ImportError:                      # sandbox layout
    from tracklet_split import FRAG_BASE, split_tracklet, split_video, _unit

D = 8
EPS, MINS = 0.2, 5


def unit(i):
    v = np.zeros(D, dtype=np.float32)
    v[i % D] = 1.0
    return v


def near(i, wobble=0.05, seed=0):
    rng = np.random.RandomState(seed)
    v = unit(i) + wobble * rng.randn(D).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def block(ident, count, single=True, zero=False, wobble=0.03, seed0=0,
          f0=0, x=100.0):
    """rows: (embedding, single, frame, box). Ghost rows get ZERO embeddings
    (the stage never embeds them)."""
    rows = []
    for k in range(count):
        e = np.zeros(D, np.float32) if (zero or not single) \
            else near(ident, wobble, seed0 + k)
        rows.append((e, single, f0 + k, [x, 400.0, 40.0, 80.0]))
    return rows


def run_split(rows):
    E = np.stack([r[0] for r in rows])
    single = np.array([r[1] for r in rows])
    frames = np.array([r[2] for r in rows])
    boxes = np.array([r[3] for r in rows])
    return split_tracklet(_unit(E), single, frames, boxes, EPS, MINS)


def test_two_identities_split():
    rows = block(0, 8, f0=0) + block(1, 8, f0=8)
    lab, k, n_noise, n_ghosts = run_split(rows)
    assert k == 2 and n_ghosts == 0
    assert len(set(lab[:8])) == 1 and len(set(lab[8:])) == 1
    assert lab[0] != lab[8]


def test_single_identity_one_fragment():
    lab, k, _, _ = run_split(block(0, 10))
    assert k == 1 and set(lab) == {0}


def test_small_tracklet_one_fragment():
    lab, k, _, _ = run_split(block(0, 2) + block(1, 2, f0=2))
    assert k == 1 and set(lab) == {0}


def test_few_singles_one_fragment_ghosts_follow():
    # 3 singles (< max(2, MINS)) + 4 ghosts: one fragment holds everything
    rows = block(0, 3, f0=0) + block(1, 4, single=False, f0=3)
    lab, k, _, n_ghosts = run_split(rows)
    assert k == 1 and set(lab) == {0} and n_ghosts == 4


def test_all_noise_one_fragment():
    rng = np.random.RandomState(0)
    E = rng.randn(8, D).astype(np.float32)
    rows = [(e, True, i, [100.0, 400.0, 40.0, 80.0]) for i, e in enumerate(E)]
    lab, k, _, _ = run_split(rows)
    assert k == 1 and set(lab) == {0}


def test_noise_single_attaches_by_appearance():
    rows = block(0, 8, f0=0) + block(1, 8, f0=8)
    rows.append((near(1, 0.4, 99), True, 16, [100.0, 400.0, 40.0, 80.0]))
    lab, k, n_noise, _ = run_split(rows)
    assert k == 2
    assert lab[-1] == lab[8]             # appearance decides for single noise


def test_zero_embedding_noise_deterministic():
    rows = block(0, 8, f0=0) + block(1, 8, f0=8)
    rows.append((np.zeros(D, np.float32), True, 16, [100.0, 400.0, 40.0, 80.0]))
    lab, k, _, _ = run_split(rows)
    assert k == 2 and lab[-1] == min(lab[0], lab[8])   # lowest label tie rule


def test_ghost_attaches_by_time_not_appearance():
    # fragment A singles at frames 0..7, fragment B singles at frames 20..27;
    # the ghost sits at frame 8 (time-adjacent to A). Its embedding is zero by
    # construction -- appearance CANNOT decide; time must place it with A.
    rows = block(0, 8, f0=0) + block(1, 8, f0=20)
    rows.append((np.zeros(D, np.float32), False, 8, [100.0, 400.0, 40.0, 80.0]))
    lab, k, _, n_ghosts = run_split(rows)
    assert k == 2 and n_ghosts == 1
    assert lab[-1] == lab[0]


def test_ghost_time_tie_breaks_on_space():
    # equal time gap to both fragments (frame 10 between singles at 8 and 12):
    # the spatially closer boundary single decides.
    rows = block(0, 8, f0=1, x=100.0) + block(1, 8, f0=12, x=800.0)
    # ghost at frame 10, dt=2 to A's last (frame 8) and to B's first (frame 12);
    # box centre near B's x -> B wins.
    rows.append((np.zeros(D, np.float32), False, 10, [795.0, 400.0, 40.0, 80.0]))
    lab, k, _, _ = run_split(rows)
    assert k == 2 and lab[-1] == lab[8]


def test_mixed_tracklet_every_fragment_has_single():
    rows = block(0, 8, f0=0) + block(1, 8, f0=8) \
        + block(2, 5, single=False, f0=16)
    lab, k, _, _ = run_split(rows)
    single = np.array([r[1] for r in rows])
    for c in set(lab):
        assert single[lab == c].any()


def test_split_tracklet_rejects_all_multi():
    rows = block(0, 6, single=False)
    try:
        run_split(rows)
    except ValueError:
        return
    raise AssertionError("all-multi tracklet must be rejected at tracklet level")


# ------------------------------------------------------------- split_video --

def vid(*tracklets):
    """tracklets: list of (tid, rows) with rows = (e, single, frame, box)."""
    rows = [(t, *r) for t, rs in tracklets for r in rs]
    tids = np.array([r[0] for r in rows])
    E = np.stack([r[1] for r in rows])
    single = np.array([r[2] for r in rows])
    frames = np.array([r[3] for r in rows])
    boxes = np.array([r[4] for r in rows])
    return E, single, frames, tids, boxes


def test_split_video_invariant_and_frag_ids():
    E, single, frames, tids, boxes = vid(
        (3, block(0, 8, f0=0) + block(1, 8, f0=8)),
        (7, block(2, 6, f0=0)))
    frag, per, vrep = split_video(E, single, frames, tids, boxes, EPS, MINS)
    assert set(frag // FRAG_BASE) == {3, 7}
    assert len(set(frag[:16])) == 2 and len(set(frag[16:])) == 1
    assert vrep["rows_cross_assigned"] == 0 and vrep["allmulti_kept"] == 0
    ks = {p["track_id"]: p["k"] for p in per}
    assert ks == {3: 2, 7: 1}


def test_split_video_rejects_duplicate_frame():
    E, single, frames, tids, boxes = vid((3, block(0, 4, f0=0)))
    frames = frames.copy()
    frames[1] = frames[0]
    try:
        split_video(E, single, frames, tids, boxes, EPS, MINS)
    except ValueError:
        return
    raise AssertionError("duplicate (tracklet, frame) must raise")


def test_allmulti_tracklet_dissolved_across_video_by_space():
    # tracklet 9 is all-multi at x ~ 800; tracklet 3 splits into a fragment at
    # x 100 and one at x 800. Every row of 9 must join the x-800 fragment of 3.
    E, single, frames, tids, boxes = vid(
        (3, block(0, 8, f0=0, x=100.0) + block(1, 8, f0=8, x=800.0)),
        (9, block(5, 4, single=False, f0=4, x=805.0)))
    frag, per, vrep = split_video(E, single, frames, tids, boxes, EPS, MINS)
    assert vrep["allmulti_tracklets"] == [9]
    assert vrep["rows_cross_assigned"] == 4 and vrep["allmulti_kept"] == 0
    target = frag[8]                     # the x-800 fragment of tracklet 3
    assert all(frag[16 + i] == target for i in range(4))
    p9 = next(p for p in per if p["track_id"] == 9)
    assert p9["allmulti"] and p9["k"] == 0


def test_allmulti_everywhere_kept():
    E, single, frames, tids, boxes = vid(
        (4, block(0, 6, single=False, f0=0)),
        (5, block(1, 6, single=False, f0=0)))
    frag, per, vrep = split_video(E, single, frames, tids, boxes, EPS, MINS)
    assert vrep["allmulti_kept"] == 2 and vrep["rows_cross_assigned"] == 0
    assert set(frag[:6]) == {4 * FRAG_BASE} and set(frag[6:]) == {5 * FRAG_BASE}


def test_single_rows_never_cross_tracklets_and_determinism():
    E, single, frames, tids, boxes = vid(
        (3, block(0, 8, f0=0) + block(1, 8, f0=8)),
        (7, block(2, 6, f0=0) + block(3, 3, single=False, f0=6)),
        (9, block(5, 4, single=False, f0=4, x=805.0)))
    frag1, _, _ = split_video(E, single, frames, tids, boxes, EPS, MINS)
    frag2, _, _ = split_video(E, single, frames, tids, boxes, EPS, MINS)
    assert (frag1 == frag2).all()
    for f in np.unique(frag1):
        sel = (frag1 == f) & single
        assert len(set(tids[sel])) <= 1  # a fragment's singles: one tracklet


if __name__ == "__main__":
    import sys
    mod = sys.modules[__name__]
    names = [n for n in dir(mod) if n.startswith("test_")]
    for n in names:
        getattr(mod, n)()
        print(f"ok {n}")
    print(f"{len(names)} tests passed")
