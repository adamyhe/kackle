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


def test_parallel_artifact_mask_is_faster_than_serial_baseline():
    if numba.get_num_threads() < 2:
        pytest.skip("parallel speedup requires at least two numba threads")

    signal, starts = make_artifact_fixture()
    artifact_mask_parallel(signal, starts, 10, 10, "+")
    artifact_mask_serial(signal, starts, 10, 10, "+")

    parallel_time = best_time(artifact_mask_parallel, signal, starts, 10, 10, "+")
    serial_time = best_time(artifact_mask_serial, signal, starts, 10, 10, "+")

    assert parallel_time < serial_time
