"""Unit tests for sn_gamestate/track/tracklet_split.py (pure numpy, no GPU).

Semantics under test: ALL detections are embedded and DBSCAN runs over single
AND multi crops; fragment centroids are single-only (the ghost rule); DBSCAN
noise -- single or multi -- attaches to the nearest single-fragment centroid
by appearance; an all-multi fragment of a mixed tracklet dissolves per
detection to the nearest single-fragment of the same tracklet; an all-multi
tracklet dissolves across the video to the nearest fragment centroid; with no
fragment anywhere it is kept. Every fragment holds a single detection except
the kept all-multi degenerates.

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


def mix(i, j, wi, wj):
    v = wi * unit(i) + wj * unit(j)
    return (v / np.linalg.norm(v)).astype(np.float32)


def block(ident, count, single=True, zero=False, wobble=0.03, seed0=0, f0=0):
    """rows: (embedding, single, frame). Multi rows are embedded too -- only
    ``zero=True`` (a failed crop) leaves a zero feature."""
    rows = []
    for k in range(count):
        e = np.zeros(D, np.float32) if zero else near(ident, wobble, seed0 + k)
        rows.append((e, single, f0 + k))
    return rows


def emb_block(vec, count, single, f0):
    """rows with one exact embedding for all ``count`` detections."""
    return [(vec.copy(), single, f0 + k) for k in range(count)]


def run_split(rows):
    E = np.stack([r[0] for r in rows])
    single = np.array([r[1] for r in rows])
    return split_tracklet(_unit(E), single, EPS, MINS)


def test_two_identities_split():
    rows = block(0, 8, f0=0) + block(1, 8, f0=8)
    lab, k, n_noise, mf_rows, mf_frags = run_split(rows)
    assert k == 2 and mf_rows == 0 and mf_frags == 0
    assert len(set(lab[:8])) == 1 and len(set(lab[8:])) == 1
    assert lab[0] != lab[8]


def test_single_identity_one_fragment():
    lab, k, *_ = run_split(block(0, 10))
    assert k == 1 and len(set(lab)) == 1


def test_small_tracklet_one_fragment():
    # fewer than max(2, min_samples) detections in total: never split
    rows = block(0, 2, f0=0) + block(1, 2, single=False, f0=2)
    lab, k, n_noise, mf_rows, mf_frags = run_split(rows)
    assert k == 1 and len(set(lab)) == 1 and (n_noise, mf_rows, mf_frags) == (0, 0, 0)


def test_all_noise_one_fragment():
    # pairwise-distant detections: DBSCAN all noise -> one fragment
    rows = [(unit(i), True, i) for i in range(4)] \
         + [(unit(i + 4), False, i + 4) for i in range(2)]
    lab, k, n_noise, *_ = run_split(rows)
    assert k == 1 and len(set(lab)) == 1 and n_noise == 0


def test_multi_crops_cluster_with_their_identity():
    # multi crops carry embeddings and join the DBSCAN cluster they resemble
    rows = block(0, 6, f0=0) + block(1, 6, f0=6) \
         + block(1, 3, single=False, f0=12, seed0=50)
    lab, k, n_noise, mf_rows, mf_frags = run_split(rows)
    assert k == 2 and mf_rows == 0 and mf_frags == 0
    assert set(lab[12:]) == {lab[6]}          # multis sit in identity-1 fragment


def test_noise_single_attaches_by_appearance():
    rows = block(0, 6, f0=0) + block(1, 6, f0=6)
    rows.append((mix(1, 2, 0.75, 0.66), True, 12))   # closer to identity 1
    lab, k, n_noise, *_ = run_split(rows)
    assert k == 2 and n_noise == 1
    assert lab[12] == lab[6]


def test_noise_multi_attaches_by_appearance():
    # a multi noise detection attaches by appearance too (no time/space rule)
    rows = block(0, 6, f0=0) + block(1, 6, f0=6)
    rows.append((mix(1, 2, 0.75, 0.66), False, 12))  # closer to identity 1
    lab, k, n_noise, *_ = run_split(rows)
    assert k == 2 and n_noise == 1
    assert lab[12] == lab[6]


def test_zero_embedding_noise_deterministic():
    # a failed crop is at distance 1 from everything -> lowest fragment label
    rows = block(0, 6, f0=0) + block(1, 6, f0=6)
    rows.append((np.zeros(D, np.float32), True, 12))
    lab, k, n_noise, *_ = run_split(rows)
    assert n_noise == 1 and lab[12] == min(lab[:6].min(), lab[6:12].min())


def test_centroids_are_single_only():
    # identity-1 fragment holds multis whose embeddings lean toward unit(2);
    # a noise single equidistant-ish decides by the SINGLE-only centroid.
    rows = block(0, 6, f0=0)                                   # fragment A
    rows += emb_block(unit(1), 6, True, 6)                     # fragment B singles
    rows += emb_block(mix(1, 2, 0.9, 0.436), 3, False, 12)     # B multis, pulled to 2
    probe = mix(1, 2, 0.98, 0.199)      # very close to unit(1): d(B_single)~0.02
    rows.append((probe, True, 15))
    lab, k, n_noise, mf_rows, mf_frags = run_split(rows)
    # the pulled multis join B (d = 1-0.9 = 0.1 < eps to its singles)
    assert k == 2 and mf_frags == 0
    assert set(lab[12:15]) == {lab[6]}
    # probe: nearest SINGLE-only centroid is B's unit(1); had the multis
    # entered the centroid it would drift toward unit(2) but the assignment
    # must follow the single-only mean
    assert lab[15] == lab[6]


def test_allmulti_fragment_dissolved_to_nearest_single_fragment():
    # a DBSCAN cluster made only of multi crops dissolves per detection
    rows = emb_block(unit(0), 6, True, 0)                      # fragment A
    rows += emb_block(unit(1), 6, True, 6)                     # fragment B
    rows += emb_block(mix(1, 2, 0.6, 0.8), 5, False, 12)       # all-multi cluster
    lab, k, n_noise, mf_rows, mf_frags = run_split(rows)
    assert mf_frags == 1 and mf_rows == 5 and n_noise == 0
    assert k == 2                                              # dissolved, not kept
    # cos to B = 0.6 (d 0.4) vs cos to A = 0 (d 1.0) -> all five to B
    assert set(lab[12:]) == {lab[6]}


def test_no_single_fragment_degenerate_one_fragment():
    # clusters exist but none holds a single (singles are noise): one fragment
    rows = emb_block(mix(1, 2, 0.6, 0.8), 6, False, 0)         # multi-only cluster
    rows.append((unit(0), True, 6))                            # single, noise
    rows.append((unit(3), True, 7))                            # single, noise
    lab, k, n_noise, mf_rows, mf_frags = run_split(rows)
    assert k == 1 and len(set(lab)) == 1
    assert (n_noise, mf_rows, mf_frags) == (0, 0, 0)


def test_split_tracklet_rejects_all_multi():
    rows = block(0, 6, single=False)
    try:
        run_split(rows)
    except ValueError:
        return
    raise AssertionError("an all-multi tracklet must be rejected")


def vid(*tracklets):
    """tracklets: lists of (embedding, single, frame). Returns aligned arrays
    plus the per-row source tracklet ids (1-based)."""
    E, single, frames, tids = [], [], [], []
    for t, rows in enumerate(tracklets, start=1):
        for e, s, f in rows:
            E.append(e); single.append(s); frames.append(f); tids.append(t)
    return (np.stack(E), np.array(single), np.array(frames, dtype=np.int64),
            np.array(tids, dtype=np.int64))


def test_split_video_invariant_and_frag_ids():
    E, s, f, t = vid(block(0, 8, f0=0) + block(1, 8, f0=8),
                     block(2, 6, f0=0))
    frag, per, vrep = split_video(E, s, f, t, EPS, MINS)
    assert len(per) == 2 and per[0]["k"] == 2 and per[1]["k"] == 1
    assert set(frag[:16]) == {1 * FRAG_BASE + 0, 1 * FRAG_BASE + 1}
    assert set(frag[16:]) == {2 * FRAG_BASE + 0}
    assert vrep["rows_cross_assigned"] == 0 and vrep["allmulti_kept"] == 0


def test_split_video_rejects_duplicate_frame():
    E, s, f, t = vid(block(0, 6, f0=0))
    f[1] = f[0]                                  # break the tracker invariant
    try:
        split_video(E, s, f, t, EPS, MINS)
    except ValueError:
        return
    raise AssertionError("a duplicated (tracklet, frame) pair must raise")


def test_allmulti_tracklet_dissolved_across_video_by_appearance():
    # tracklet 3 is all multi near identity 1 -> every row joins tracklet 2's
    # fragment (nearest single-only centroid), regardless of frames
    E, s, f, t = vid(block(0, 8, f0=0),
                     block(1, 8, f0=0),
                     block(1, 4, single=False, f0=100, seed0=77))
    frag, per, vrep = split_video(E, s, f, t, EPS, MINS)
    assert per[2]["allmulti"] is True
    assert vrep["allmulti_tracklets"] == [3]
    assert vrep["rows_cross_assigned"] == 4 and vrep["allmulti_kept"] == 0
    assert set(frag[16:]) == {2 * FRAG_BASE + 0}


def test_allmulti_everywhere_kept():
    E, s, f, t = vid(block(0, 5, single=False, f0=0),
                     block(1, 5, single=False, f0=0))
    frag, per, vrep = split_video(E, s, f, t, EPS, MINS)
    assert vrep["allmulti_kept"] == 2 and vrep["rows_cross_assigned"] == 0
    assert set(frag[:5]) == {1 * FRAG_BASE} and set(frag[5:]) == {2 * FRAG_BASE}


def test_single_rows_never_cross_tracklets_and_determinism():
    E, s, f, t = vid(block(0, 8, f0=0) + block(1, 8, f0=8),
                     block(2, 8, f0=0),
                     block(0, 3, single=False, f0=50, seed0=9))
    out1 = split_video(E, s, f, t, EPS, MINS)
    out2 = split_video(E, s, f, t, EPS, MINS)
    assert np.array_equal(out1[0], out2[0])                    # deterministic
    frag = out1[0]
    # single rows keep their source tracklet (frag // FRAG_BASE)
    for i in np.where(s)[0]:
        assert int(frag[i]) // FRAG_BASE == int(t[i])
    # the all-multi tracklet's rows went to identity-0's fragment of tracklet 1
    assert set(frag[24:]) == {1 * FRAG_BASE + 0} or set(frag[24:]) == {1 * FRAG_BASE + 1}
    # ... specifically the fragment whose singles are identity 0
    target = int(list(set(frag[24:]))[0])
    rows_target_single = np.where((frag == target) & s)[0]
    assert all(int(t[i]) == 1 for i in rows_target_single)


def _main():
    import sys
    mod = sys.modules[__name__]
    names = [n for n in dir(mod) if n.startswith("test_")]
    for n in sorted(names):
        getattr(mod, n)()
        print(f"ok {n}")
    print(f"{len(names)} tests passed")


if __name__ == "__main__":
    _main()
