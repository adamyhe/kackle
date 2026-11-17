import time

import numba
import numpy as np
import pytest

from kackle.computation import (
    artifact_mask_parallel,
    artifact_mask_serial,
    correct_kmers,
    resample_dirmult,
    resample_empirical,
)


def make_artifact_fixture(length=2_000_000, step=4, source=10):
    starts = np.arange(source, length - source, step, dtype=np.int32)
    signal = np.ones(length, dtype=np.int32)
    signal[starts[::50] - 1] = 100
    signal[starts[25::50]] = 120
    return signal, starts


def best_time(fn, *args, repeats=5):
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn(*args)
        timings.append(time.perf_counter() - started)
    return min(timings)


def test_artifact_mask_parallel_matches_serial_for_both_strands():
    signal, starts = make_artifact_fixture(length=200_000)

    plus_parallel = artifact_mask_parallel(signal, starts, 10, 10, "+")
    plus_serial = artifact_mask_serial(signal, starts, 10, 10, "+")
    minus_parallel = artifact_mask_parallel(signal, starts, 10, 10, "-")
    minus_serial = artifact_mask_serial(signal, starts, 10, 10, "-")

    assert np.array_equal(plus_parallel, plus_serial)
    assert np.array_equal(minus_parallel, minus_serial)
    assert plus_parallel.sum() > 0
    assert minus_parallel.sum() > 0


def test_numba_kernels_compile_in_nopython_mode_and_use_cache():
    signal, starts = make_artifact_fixture(length=10_000)
    windows = np.ones((4, 20), dtype=np.int32)

    artifact_mask_parallel(signal, starts, 10, 10, "+")
    artifact_mask_serial(signal, starts, 10, 10, "+")
    correct_kmers(signal, starts[:4], 10, 5, 10, "+")
    resample_dirmult(windows, 10, 5)
    resample_empirical(windows, 10, 5)

    dispatchers = [
        artifact_mask_parallel,
        artifact_mask_serial,
        correct_kmers,
        resample_dirmult,
        resample_empirical,
    ]
    for dispatcher in dispatchers:
        assert dispatcher.signatures
        assert dispatcher.nopython_signatures
        assert dispatcher.targetoptions["nopython"] is True
        assert dispatcher._cache is not None

    assert artifact_mask_parallel.targetoptions["parallel"] is True
    assert correct_kmers.targetoptions["parallel"] is True


def test_correct_kmers_preserves_signal_when_no_candidates_pass_threshold():
    signal = np.ones(200, dtype=np.int32)
    starts = np.array([50, 100, 150], dtype=np.int32)

    corrected = correct_kmers(signal, starts, 10, 5, 1000, "+")

    assert np.array_equal(corrected, signal)


def test_correct_kmers_preserves_signal_with_no_candidates():
    signal = np.ones(200, dtype=np.int32)
    starts = np.array([], dtype=np.int32)

    corrected = correct_kmers(signal, starts, 10, 5, 10, "+")

    assert np.array_equal(corrected, signal)


def test_correct_kmers_fused_kernel_only_updates_target_window():
    signal = np.ones(80, dtype=np.int32)
    signal[39] = 100
    starts = np.array([40], dtype=np.int32)

    corrected = correct_kmers(signal, starts, 10, 3, 10, "+")
    changed = np.flatnonzero(corrected != signal)

    assert set(changed.tolist()) <= set(range(37, 43))
    assert changed.size > 0


def test_parallel_artifact_mask_is_faster_than_serial_baseline():
    if numba.get_num_threads() < 2:
        pytest.skip("parallel speedup requires at least two numba threads")

    signal, starts = make_artifact_fixture()
    artifact_mask_parallel(signal, starts, 10, 10, "+")
    artifact_mask_serial(signal, starts, 10, 10, "+")

    parallel_time = best_time(artifact_mask_parallel, signal, starts, 10, 10, "+")
    serial_time = best_time(artifact_mask_serial, signal, starts, 10, 10, "+")

    assert parallel_time < serial_time


def _flank_sum(signal, first, last, source, target):
    return int(
        signal[first - source : first - target].sum()
        + signal[last + target : last + source].sum()
    )


def _spiked_signal(length, anchors, background=2, spike=500):
    signal = np.full(length, background, dtype=np.int32)
    signal[np.asarray(anchors) - 1] = spike  # plus-strand artifact position
    return signal


def test_correct_kmers_isolated_site_redraws_flank_total_in_target_window():
    signal = _spiked_signal(200, [100])
    signal[85:95] = np.arange(10)  # make the flanks distinguishable

    corrected = correct_kmers(signal, np.array([100], dtype=np.int32), 10, 5, 10, "+")

    changed = np.flatnonzero(corrected != signal)
    assert set(changed.tolist()) <= set(range(95, 105))
    assert corrected[95:105].sum() == _flank_sum(signal, 100, 100, 10, 5)


@pytest.mark.parametrize("gap", [0, 3, 8, 12, 14])
def test_correct_kmers_merges_interfering_sites_into_one_span(gap):
    source, target = 10, 5
    first, last = 100, 100 + gap
    signal = _spiked_signal(300, [first, last])

    corrected = correct_kmers(
        signal, np.array([first, last], dtype=np.int32), source, target, 10, "+"
    )

    span = slice(first - target, last + target)
    width = (last + target) - (first - target)
    expected = (_flank_sum(signal, first, last, source, target) * width + target) // (
        2 * target
    )
    # Neither spike leaks into the redrawn total, and nothing outside the span moves.
    assert corrected[span].sum() == expected
    outside = np.ones(signal.shape[0], dtype=bool)
    outside[span] = False
    assert np.array_equal(corrected[outside], signal[outside])
    assert corrected.max() < 500


def test_correct_kmers_resamples_distant_sites_independently():
    source, target = 10, 5
    anchors = [100, 100 + source + target]  # just far enough apart not to merge
    signal = _spiked_signal(300, anchors)

    corrected = correct_kmers(signal, np.array(anchors, dtype=np.int32), source, target, 10, "+")

    for anchor in anchors:
        window = corrected[anchor - target : anchor + target]
        assert window.sum() == _flank_sum(signal, anchor, anchor, source, target)


def test_correct_kmers_unflagged_neighbor_does_not_join_cluster():
    source, target = 10, 5
    signal = _spiked_signal(300, [100])
    starts = np.array([100, 104], dtype=np.int32)  # 104 is a candidate but not a spike

    corrected = correct_kmers(signal, starts, source, target, 10, "+")

    assert corrected[95:105].sum() == _flank_sum(signal, 100, 100, source, target)
    assert np.array_equal(corrected[105:], signal[105:])


def test_correct_kmers_conserves_cluster_totals_under_parallel_dense_overlaps():
    # Many spiked sites at random spacings, unsorted and with duplicates. Every
    # cluster's span total must equal its scaled flank total, which fails if
    # parallel workers race on overlapping windows. Dense neighbors can hide a
    # spike from the screen, so the expected flagged set comes from the
    # reference screen rather than from where spikes were planted.
    source, target = 10, 5
    rng = np.random.default_rng(0)
    gaps = rng.integers(0, 40, 5_000)
    anchors = 50 + np.cumsum(gaps)
    signal = _spiked_signal(int(anchors[-1]) + 100, anchors, background=3)
    starts = rng.permutation(np.concatenate([anchors, anchors[::7]])).astype(np.int32)

    corrected = correct_kmers(signal, starts, source, target, 10, "+")

    flagged = np.unique(starts[artifact_mask_serial(signal, starts, source, 10, "+")])
    assert flagged.size > 1000
    breaks = np.flatnonzero(np.diff(flagged) >= source + target) + 1
    for cluster in np.split(flagged, breaks):
        first, last = int(cluster[0]), int(cluster[-1])
        width = (last + target) - (first - target)
        flank = _flank_sum(signal, first, last, source, target)
        assert corrected[first - target : last + target].sum() == (
            (flank * width + target) // (2 * target)
        )


def test_correct_kmers_minus_strand_uses_anchor_as_artifact_position():
    signal = np.full(200, 2, dtype=np.int32)
    signal[100] = 500  # minus-strand artifact position is the anchor itself

    corrected = correct_kmers(signal, np.array([100], dtype=np.int32), 10, 5, 10, "-")

    assert corrected[95:105].sum() == _flank_sum(signal, 100, 100, 10, 5)
    assert corrected.max() < 500
