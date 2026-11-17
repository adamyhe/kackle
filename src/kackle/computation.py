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


@njit(cache=True)
def _flagged_clusters(flagged_starts, link_distance):
    """Split sorted flagged anchors into clusters of mutually interfering sites.

    Consecutive anchors closer than ``link_distance`` belong to the same
    cluster. Returns an array of cluster boundaries into ``flagged_starts``:
    cluster ``c`` is ``flagged_starts[bounds[c] : bounds[c + 1]]``.
    """
    n = flagged_starts.shape[0]
    bounds = np.empty(n + 1, dtype=np.int64)
    n_clusters = 0
    if n == 0:
        bounds[0] = 0
        return bounds[:1]
    bounds[0] = 0
    for k in range(1, n):
        if flagged_starts[k] - flagged_starts[k - 1] >= link_distance:
            n_clusters += 1
            bounds[n_clusters] = k
    n_clusters += 1
    bounds[n_clusters] = n
    return bounds[: n_clusters + 1]


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

    1. Every candidate anchor is screened against the input vector: it is
       flagged when its artifact position exceeds ``threshold`` times the mean
       of the rest of its ``[anchor - source, anchor + source)`` neighborhood.
    2. Flagged anchors closer than ``source + target`` are merged into one
       cluster, because at that distance one site's flank would overlap
       another's target window (and closer than ``2 * target`` their target
       windows overlap outright).
    3. Each cluster is resampled once. Its target span is
       ``[first - target, last + target)`` and its flanks are the
       ``source - target`` bases just outside that span on each side, so
       neither can contain another cluster's spike. The span is redrawn with
       a Dirichlet-multinomial whose total is the flank sum scaled by
       ``span_width / (2 * target)``.

    An isolated flagged site is a cluster of one, whose total is exactly its
    flank sum, as in earlier releases. Clusters are disjoint, so the parallel
    resampling has no write races, and only target spans are modified.
    """
    out_vec = chromosome_signal_vector.copy()
    if starts.shape[0] == 0:
        return out_vec

    flagged = artifact_mask_parallel(
        chromosome_signal_vector, starts, source, threshold, strand
    )
    flagged_starts = np.sort(starts[flagged].astype(np.int64))
    if flagged_starts.shape[0] == 0:
        return out_vec
    bounds = _flagged_clusters(flagged_starts, source + target)

    for c in prange(bounds.shape[0] - 1):
        start = flagged_starts[bounds[c]]
        last = flagged_starts[bounds[c + 1] - 1]
        span_lo = start - target
        span_hi = last + target
        flank_sum = 0
        for j in range(start - source, span_lo):
            flank_sum += chromosome_signal_vector[j]
        for j in range(span_hi, last + source):
            flank_sum += chromosome_signal_vector[j]
        width = span_hi - span_lo
        # Width-scaled total, rounded half up; equals flank_sum when isolated.
        total = (flank_sum * width + target) // (2 * target)
        probs = np.random.dirichlet(np.ones(width))
        resampled = np.random.multinomial(total, probs)
        for j in range(width):
            out_vec[span_lo + j] = resampled[j]
    return out_vec


@njit(parallel=True, cache=True)
def correct_artifact_bases(
    chromosome_signal_vector: np.array,
    starts: np.array,
    source: np.uint = 10,
    threshold: np.uint = 10,
    strand: str = "+",
) -> np.array:
    """Redraw only the artifact base of each flagged site.

    Screening is the same as in ``correct_kmers``. For each flagged anchor,
    the strand-appropriate artifact base (plus: ``anchor - 1``; minus:
    ``anchor``) is replaced with a Poisson draw whose mean is the mean of the
    rest of its ``[anchor - source, anchor + source)`` neighborhood, leaving
    out the artifact bases of other flagged sites. Every other base keeps
    its input value, so natural structure around the motif is untouched.
    Each flagged site writes only its own artifact base and reads only the
    input vector, so the parallel loop has no races.
    """
    out_vec = chromosome_signal_vector.copy()
    if starts.shape[0] == 0:
        return out_vec
    flagged = artifact_mask_parallel(
        chromosome_signal_vector, starts, source, threshold, strand
    )
    offset = 1 if strand == "+" else 0
    art = np.sort(starts[flagged].astype(np.int64)) - offset
    n = art.shape[0]
    for i in prange(n):
        idx = art[i]
        anchor = idx + offset
        lo = anchor - source
        hi = anchor + source
        total = 0
        count = 0
        k_lo = np.searchsorted(art, lo)
        k = k_lo
        for j in range(lo, hi):
            # Skip this and other flagged sites' artifact bases.
            while k < n and art[k] < j:
                k += 1
            if k < n and art[k] == j:
                continue
            total += chromosome_signal_vector[j]
            count += 1
        mean = total / count if count > 0 else 0.0
        out_vec[idx] = np.random.poisson(mean)
    return out_vec


# ---------------------------------------------------------------------------
# Motif matching
# ---------------------------------------------------------------------------

_MOTIF_CHUNK = 1 << 16


@njit(cache=True, inline="always")
def _window_matches(sequence, start, pattern, max_mismatches):
    """Whether ``sequence[start:start + k]`` is all A/C/G/T within Hamming distance.

    Bytes are uppercased with ``& 0xDF`` (ASCII letters only). A window with
    any other base (N, IUPAC codes, gaps) never matches, which is what an
    exact search over the A/C/G/T mismatch variants of ``pattern`` gives.
    """
    mismatches = 0
    for j in range(pattern.shape[0]):
        base = sequence[start + j] & 0xDF
        if base != 65 and base != 67 and base != 71 and base != 84:  # A C G T
            return False
        if base != pattern[j]:
            mismatches += 1
            if mismatches > max_mismatches:
                return False
    return True


@njit(parallel=True, cache=True)
def motif_window_starts(sequence, pattern, max_mismatches):
    """Return sorted starts of windows that match ``pattern`` (see ``_window_matches``).

    ``kackle.motifs._python_window_starts`` is the independent reference this
    is tested against, and the fallback when numba JIT is disabled.

    Parameters
    ----------
    sequence : numpy.ndarray of uint8
        ASCII sequence bytes, any case.
    pattern : numpy.ndarray of uint8
        Uppercase A/C/G/T motif bytes.
    max_mismatches : int
        Maximum substitutions allowed.

    Two parallel passes over fixed-size chunks: count matches per chunk, then
    fill each chunk's slice of the output, so starts come out sorted and
    unique without a sort.
    """
    k = pattern.shape[0]
    n_windows = sequence.shape[0] - k + 1
    if k == 0 or n_windows <= 0:
        return np.empty(0, dtype=np.int64)
    n_chunks = (n_windows + _MOTIF_CHUNK - 1) // _MOTIF_CHUNK
    counts = np.zeros(n_chunks, dtype=np.int64)
    for chunk in prange(n_chunks):
        lo = chunk * _MOTIF_CHUNK
        hi = min(lo + _MOTIF_CHUNK, n_windows)
        found = 0
        for start in range(lo, hi):
            if _window_matches(sequence, start, pattern, max_mismatches):
                found += 1
        counts[chunk] = found
    offsets = np.zeros(n_chunks + 1, dtype=np.int64)
    for chunk in range(n_chunks):
        offsets[chunk + 1] = offsets[chunk] + counts[chunk]
    starts = np.empty(offsets[n_chunks], dtype=np.int64)
    for chunk in prange(n_chunks):
        lo = chunk * _MOTIF_CHUNK
        hi = min(lo + _MOTIF_CHUNK, n_windows)
        out = offsets[chunk]
        for start in range(lo, hi):
            if _window_matches(sequence, start, pattern, max_mismatches):
                starts[out] = start
                out += 1
    return starts
