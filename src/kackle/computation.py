"""Numerical routines for correcting kmer artifacts.

These functions are compiled with numba and deliberately avoid pandas, Python
objects, and file I/O. They operate only on NumPy arrays prepared by the CLI and
backend layers.
"""

import numpy as np
from numba import njit, prange


@njit(cache=True)
def artifact_mask_serial(
    chromosome_signal_vector: np.array,
    starts: np.array,
    source: np.uint = 10,
    threshold: np.uint = 10,
    strand: str = "+",
) -> np.array:
    """Return a boolean mask of candidate starts that exceed local background.

    This serial implementation is primarily a correctness and performance
    baseline for the parallel kernel used by ``correct_kmers``.
    """
    pass_threshold = np.full(starts.shape[0], False)
    for i in range(len(starts)):
        start = starts[i]
        if start - source < 0 or start + source > len(chromosome_signal_vector):
            continue
        suspicious_signal_idx = start - 1 if strand == "+" else start
        suspicious_signal = chromosome_signal_vector[suspicious_signal_idx]
        neighborhood_sum = 0
        for j in range(start - source, start + source):
            neighborhood_sum += chromosome_signal_vector[j]
        suspicious_background = (neighborhood_sum - suspicious_signal) / (source * 2 - 1)
        if suspicious_signal > threshold * suspicious_background:
            pass_threshold[i] = True
    return pass_threshold


@njit(parallel=True, cache=True)
def artifact_mask_parallel(
    chromosome_signal_vector: np.array,
    starts: np.array,
    source: np.uint = 10,
    threshold: np.uint = 10,
    strand: str = "+",
) -> np.array:
    """Parallel artifact-threshold screen used by ``correct_kmers``."""
    pass_threshold = np.full(starts.shape[0], False)
    for i in prange(len(starts)):
        start = starts[i]
        if start - source < 0 or start + source > len(chromosome_signal_vector):
            continue
        suspicious_signal_idx = start - 1 if strand == "+" else start
        suspicious_signal = chromosome_signal_vector[suspicious_signal_idx]
        neighborhood_sum = 0
        for j in range(start - source, start + source):
            neighborhood_sum += chromosome_signal_vector[j]
        suspicious_background = (neighborhood_sum - suspicious_signal) / (source * 2 - 1)
        if suspicious_signal > threshold * suspicious_background:
            pass_threshold[i] = True
    return pass_threshold


@njit(parallel=True, cache=True)
def resample_dirmult(
    arr: np.array, source: np.uint = 10, target: np.uint = 5
) -> np.array:
    """Resample the central target window using a Dirichlet-multinomial draw.

    The total count is drawn from the flanking source windows and redistributed
    across the target region. The input array is not modified.
    """
    arr_ = np.zeros(arr.shape)
    a = np.ones(target * 2)
    mid = arr.shape[-1] // 2
    for i in prange(arr.shape[0]):
        arr_[i] = arr[i]
        probs = np.random.dirichlet(a)
        z = np.random.multinomial(
            np.sum(arr[i, mid - source : mid - target])
            + np.sum(arr[i, mid + target : mid + source]),
            probs,
        )
        arr_[i, mid - target : mid + target] = z
    return arr_


@njit(parallel=True, cache=True)
def resample_empirical(
    arr: np.array, source: np.uint = 50, target: np.uint = 5
) -> np.array:
    """Resample the central target window from the local empirical background.

    This alternate strategy samples from the interquartile range of flanking
    source positions. It is retained for experimentation and is not currently
    used by ``correct_kmers``.
    """
    arr_ = np.zeros(arr.shape)
    mid = arr.shape[-1] // 2
    for i in prange(arr.shape[0]):
        arr_[i] = arr[i]
        empirical_dist = np.zeros(source * 2 - target * 2)
        empirical_dist[: empirical_dist.shape[0] // 2] = arr[
            i, mid - source : mid - target
        ]
        empirical_dist[empirical_dist.shape[0] // 2 :] = arr[
            i, mid + target : mid + source
        ]
        q1 = np.percentile(empirical_dist, 25)
        q3 = np.percentile(empirical_dist, 75)
        empirical_iqr = empirical_dist[(empirical_dist >= q1) & (empirical_dist <= q3)]
        z = np.random.choice(empirical_iqr, size=target * 2)
        arr_[i, mid - target : mid + target] = z
    return arr_


@njit(parallel=True, cache=True)
def correct_kmers(
    chromosome_signal_vector: np.array,
    starts: np.array,
    source: np.uint = 10,
    target: np.uint = 5,
    threshold: np.uint = 10,
    strand: str = "+",
) -> np.array:
    """Correct suspicious kmer-centered signal spikes in a dense vector.

    Candidate starts are screened, resampled, and written back in one fused
    parallel pass. Thresholds and flanking totals are read from the input vector
    for this motif pass, while only the central target window is updated in the
    returned copy.
    """
    out_vec = chromosome_signal_vector.copy()
    if starts.shape[0] == 0:
        return out_vec

    alpha = np.ones(target * 2)
    signal_len = len(chromosome_signal_vector)
    for i in prange(starts.shape[0]):
        start = starts[i]
        if start - source < 0 or start + source > signal_len:
            continue
        suspicious_signal_idx = start - 1 if strand == "+" else start
        suspicious_signal = chromosome_signal_vector[suspicious_signal_idx]
        neighborhood_sum = 0
        left_flank_sum = 0
        right_flank_sum = 0
        for j in range(start - source, start + source):
            value = chromosome_signal_vector[j]
            neighborhood_sum += value
            if j < start - target:
                left_flank_sum += value
            elif j >= start + target:
                right_flank_sum += value
        suspicious_background = (neighborhood_sum - suspicious_signal) / (source * 2 - 1)
        if suspicious_signal <= threshold * suspicious_background:
            continue

        probs = np.random.dirichlet(alpha)
        resampled = np.random.multinomial(left_flank_sum + right_flank_sum, probs)
        for j in range(target * 2):
            out_vec[start - target + j] = resampled[j]
    return out_vec
