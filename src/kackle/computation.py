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
    min_reads: np.int64 = 1,
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
        if (
            suspicious_signal >= min_reads
            and suspicious_signal > threshold * suspicious_background
        ):
            pass_threshold[i] = True
    return pass_threshold


@njit(parallel=True, cache=True)
def artifact_mask_parallel(
    chromosome_signal_vector: np.array,
    starts: np.array,
    source: np.uint = 10,
    threshold: np.uint = 10,
    strand: str = "+",
    min_reads: np.int64 = 1,
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
        if (
            suspicious_signal >= min_reads
            and suspicious_signal > threshold * suspicious_background
        ):
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
    min_reads: np.int64 = 1,
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
        chromosome_signal_vector, starts, source, threshold, strand, min_reads
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
    min_reads: np.int64 = 1,
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
        chromosome_signal_vector, starts, source, threshold, strand, min_reads
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


@njit(parallel=True, cache=True)
def correct_adapter_sites(
    chromosome_signal_vector: np.array,
    starts: np.array,
    thresholds: np.array,
    source: np.uint = 10,
    strand: str = "+",
    min_reads=None,
) -> np.array:
    """Screen and correct all adapter look-alike sites of one strand in one pass.

    Each anchor's artifact base (plus: ``anchor - 1``; minus: ``anchor``) is
    compared with the mean of its ``[anchor - source, anchor + source)``
    neighborhood in two stages, both on the input signal:

    1. a plain screen: the background is the rest of the neighborhood, and a
       site is flagged when its artifact base exceeds ``thresholds[i]`` times
       that mean;
    2. a rescreen whose background also leaves out the artifact bases of the
       sites flagged in stage 1, so a nearby spike no longer hides a site.
       Candidates that stage 1 did not flag stay in the background.

    A site flagged in either stage has its artifact base replaced with a
    Poisson draw with the stage-2 mean. Every other base keeps its input value.
    Each site writes only its own artifact base and reads only the input, so
    the result does not depend on site order and the loops have no races.
    ``starts`` must not repeat an anchor. ``min_reads``, if given, is a
    per-site int64 array: in both stages a site is flagged only if its
    artifact base also has at least ``min_reads[i]`` reads.
    """
    out_vec = chromosome_signal_vector.copy()
    n = starts.shape[0]
    if n == 0:
        return out_vec
    length = chromosome_signal_vector.shape[0]
    offset = 1 if strand == "+" else 0
    # Stage 1: plain screen.
    flag1 = np.zeros(n, dtype=np.bool_)
    for i in prange(n):
        anchor = np.int64(starts[i])
        idx = anchor - offset
        lo = anchor - source
        hi = anchor + source
        if lo < 0 or hi > length:
            continue
        total = 0.0
        for j in range(lo, hi):
            total += chromosome_signal_vector[j]
        mean = (total - chromosome_signal_vector[idx]) / (hi - lo - 1)
        need = 1 if min_reads is None else min_reads[i]
        if (
            chromosome_signal_vector[idx] >= need
            and chromosome_signal_vector[idx] > thresholds[i] * mean
        ):
            flag1[i] = True
    art = np.sort(starts[flag1].astype(np.int64)) - offset
    m = art.shape[0]
    # Stage 2: rescreen without stage-1 flagged artifact bases.
    for i in prange(n):
        anchor = np.int64(starts[i])
        idx = anchor - offset
        lo = anchor - source
        hi = anchor + source
        if lo < 0 or hi > length:
            continue
        total = 0.0
        count = 0
        k = np.searchsorted(art, lo)
        for j in range(lo, hi):
            while k < m and art[k] < j:
                k += 1
            if j == idx or (k < m and art[k] == j):
                continue
            total += chromosome_signal_vector[j]
            count += 1
        if count == 0:
            continue
        mean = total / count
        need = 1 if min_reads is None else min_reads[i]
        if flag1[i] or (
            chromosome_signal_vector[idx] >= need
            and chromosome_signal_vector[idx] > thresholds[i] * mean
        ):
            out_vec[idx] = np.random.poisson(mean)
    return out_vec


@njit(parallel=True, cache=True)
def adapter_match_lengths(sequence, adapter, min_match, skip=0):
    """Adapter look-alike match lengths, by mismatch budget, at every window start.

    ``sequence`` and ``adapter`` are IUPAC bitmasks (A=1, C=2, G=4, T=8; 0 for
    anything that is not A/C/G/T in the sequence), so a base matches when the
    masks share a bit. ``min_match[m]`` is the shortest prefix of the adapter
    that makes a candidate when it contains ``m`` mismatches; its length minus
    one is the most mismatches allowed. Returns an int8 array of shape
    ``(n_windows, len(min_match))`` over window starts ``p`` in
    ``[0, len(sequence) - len(adapter)]``: entry ``[p, m]`` is the length of
    the longest adapter prefix (ending on a match) with at most ``m``
    mismatches, so column 0 is the exact-match length. A row is all 0 when the
    window is not a candidate.

    This is the plain reference; the pipeline uses :func:`adapter_candidates`.

    The first ``skip`` adapter bases (leading ``N``s such as a UMI) are part
    of the window but are not compared and don't count toward any length:
    lengths and ``min_match`` are measured from adapter base ``skip``.
    """
    n = sequence.shape[0]
    k_len = adapter.shape[0]
    max_mm = min_match.shape[0] - 1
    n_windows = n - k_len + 1
    if n_windows < 0:
        n_windows = 0
    lengths = np.zeros((n_windows, max_mm + 1), dtype=np.int8)
    for p in prange(n_windows):
        mm = 0
        candidate = False
        for k in range(skip, k_len):
            if sequence[p + k] & adapter[k]:
                # Longest prefix ending on a match with exactly ``mm`` so far;
                # fill every budget >= mm (later matches are longer).
                for m in range(mm, max_mm + 1):
                    lengths[p, m] = k + 1 - skip
                if k + 1 - skip >= min_match[mm]:
                    candidate = True
            else:
                mm += 1
                if mm > max_mm:
                    break
        if not candidate:
            for m in range(max_mm + 1):
                lengths[p, m] = 0
    return lengths


_ADAPTER_CHUNK = 1 << 16


@njit(cache=True, inline="always")
def _adapter_maybe(sequence, adapter, min_match, skip, reverse, lo, width, maybe, mm):
    """Vectorized pre-check: ``maybe[i]`` is 0 where window ``lo + i`` can't be a candidate.

    A candidate has, for some budget ``m``, a prefix of at least
    ``min_match[m]`` bases with at most ``m`` mismatches, so its first
    ``min_match[m]`` compared bases hold at most ``m`` mismatches.
    """
    k_len = adapter.shape[0]
    n_budgets = min_match.shape[0]
    for i in range(width):
        mm[i] = 0
        maybe[i] = 0
    depth = min(min_match[n_budgets - 1], k_len - skip)
    for j in range(depth):
        k = skip + j
        a = adapter[k]
        off = lo + (k_len - 1 - k if reverse else k)
        column = sequence[off : off + width]
        for i in range(width):
            mm[i] += np.uint8((column[i] & a) == 0)
        for b in range(n_budgets):
            if min_match[b] == j + 1:
                bb = np.uint8(b)
                for i in range(width):
                    maybe[i] |= np.uint8(mm[i] <= bb)


@njit(cache=True, inline="always")
def _adapter_scan(sequence, adapter, min_match, skip, reverse, p, lengths):
    """The reference prefix scan for one window; fills ``lengths``, returns candidacy."""
    k_len = adapter.shape[0]
    max_mm = min_match.shape[0] - 1
    for m in range(max_mm + 1):
        lengths[m] = 0
    mm = 0
    candidate = False
    for k in range(skip, k_len):
        if reverse:
            base = sequence[p + k_len - 1 - k]
        else:
            base = sequence[p + k]
        if base & adapter[k]:
            for m in range(mm, max_mm + 1):
                lengths[m] = k + 1 - skip
            if k + 1 - skip >= min_match[mm]:
                candidate = True
        else:
            mm += 1
            if mm > max_mm:
                break
    return candidate


@njit(parallel=True, cache=True)
def adapter_candidates(sequence, adapter, min_match, skip=0, reverse=False):
    """Candidate windows and their match lengths: a fast ``adapter_match_lengths``.

    Same arguments and rules as :func:`adapter_match_lengths` (which is the
    reference this is tested against), but returns only the candidates, as
    ``(starts, lengths)``: sorted int64 window starts and their int8
    ``(n, len(min_match))`` lengths by mismatch budget.

    With ``reverse``, windows are matched on the reverse complement in place:
    window ``p`` is ``sequence[p:p + len(adapter)]`` read backwards and
    complemented, and ``adapter`` must then hold the *complemented* adapter
    masks. ``p`` is the window's leftmost base either way, so minus-strand
    starts come out sorted without reversing the sequence.

    A vectorized pre-check (``_adapter_maybe``) rejects nearly every
    non-candidate; only the rest get the per-window prefix scan. Two passes
    over chunks (count, then fill at each chunk's offset) keep the output
    compact without whole-sequence scratch arrays.
    """
    n = sequence.shape[0]
    k_len = adapter.shape[0]
    n_budgets = min_match.shape[0]
    n_windows = n - k_len + 1
    if n_windows < 0:
        n_windows = 0
    n_chunks = (n_windows + _ADAPTER_CHUNK - 1) // _ADAPTER_CHUNK
    counts = np.zeros(n_chunks, dtype=np.int64)
    for chunk in prange(n_chunks):
        lo = chunk * _ADAPTER_CHUNK
        width = min(lo + _ADAPTER_CHUNK, n_windows) - lo
        maybe = np.empty(width, dtype=np.uint8)
        mm = np.empty(width, dtype=np.uint8)
        scratch = np.empty(n_budgets, dtype=np.int8)
        _adapter_maybe(sequence, adapter, min_match, skip, reverse, lo, width, maybe, mm)
        found = 0
        for i in range(width):
            if maybe[i] and _adapter_scan(sequence, adapter, min_match, skip, reverse, lo + i, scratch):
                found += 1
        counts[chunk] = found
    offsets = np.zeros(n_chunks + 1, dtype=np.int64)
    for chunk in range(n_chunks):
        offsets[chunk + 1] = offsets[chunk] + counts[chunk]
    starts = np.empty(offsets[n_chunks], dtype=np.int64)
    lengths = np.empty((offsets[n_chunks], n_budgets), dtype=np.int8)
    for chunk in prange(n_chunks):
        lo = chunk * _ADAPTER_CHUNK
        width = min(lo + _ADAPTER_CHUNK, n_windows) - lo
        maybe = np.empty(width, dtype=np.uint8)
        mm = np.empty(width, dtype=np.uint8)
        scratch = np.empty(n_budgets, dtype=np.int8)
        _adapter_maybe(sequence, adapter, min_match, skip, reverse, lo, width, maybe, mm)
        out = offsets[chunk]
        for i in range(width):
            if maybe[i] and _adapter_scan(sequence, adapter, min_match, skip, reverse, lo + i, scratch):
                starts[out] = lo + i
                for b in range(n_budgets):
                    lengths[out, b] = scratch[b]
                out += 1
    return starts, lengths


# ---------------------------------------------------------------------------
# Motif matching
# ---------------------------------------------------------------------------

_MOTIF_CHUNK = 1 << 16
#: Longest motif ``motif_window_mask`` handles: counts live in the low 7 bits
#: of a uint8, with the top bit marking a non-A/C/G/T base.
MAX_MASK_MOTIF_LENGTH = 127


@njit(parallel=True, cache=True)
def sequence_codes(sequence, lut):
    """``lut[sequence]`` (byte -> base code), in parallel."""
    out = np.empty(sequence.shape[0], dtype=np.uint8)
    for i in prange(sequence.shape[0]):
        out[i] = lut[sequence[i]]
    return out


@njit(parallel=True, cache=True)
def motif_window_mask(codes, pattern, max_mismatches):
    """Whether each window of ``codes`` matches ``pattern`` within Hamming distance.

    ``codes`` holds one base code per position (A=0, C=1, G=2, T=3, any case;
    4 for anything else; see ``kackle.motifs._sequence_codes``) and
    ``pattern`` the motif's codes (0-3), at most ``MAX_MASK_MOTIF_LENGTH``
    long. Entry ``p`` is true when ``codes[p:p + k]`` is all A/C/G/T and at
    most ``max_mismatches`` bases differ from the pattern, for ``p`` in
    ``[0, len(codes) - k]``. ``kackle.motifs._python_window_starts`` is the
    independent reference this is tested against, and the fallback when numba
    JIT is disabled.

    Each chunk of windows counts its mismatches one pattern column at a time
    over contiguous slices, so the inner loop is branch-free and vectorizes.
    The count stays in the low 7 bits of a uint8; a non-A/C/G/T base sets the
    top bit, which no later addition can clear, so that window never passes.
    """
    k = pattern.shape[0]
    n_windows = codes.shape[0] - k + 1
    if k == 0 or n_windows <= 0:
        return np.zeros(0, dtype=np.bool_)
    out = np.empty(n_windows, dtype=np.bool_)
    n_chunks = (n_windows + _MOTIF_CHUNK - 1) // _MOTIF_CHUNK
    for chunk in prange(n_chunks):
        lo = chunk * _MOTIF_CHUNK
        width = min(lo + _MOTIF_CHUNK, n_windows) - lo
        counts = np.zeros(width, dtype=np.uint8)
        for j in range(k):
            column = codes[lo + j : lo + j + width]
            pj = pattern[j]
            for i in range(width):
                c = column[i]
                counts[i] = (counts[i] + (c != pj)) | ((c >> 2) << 7)
        window = out[lo : lo + width]
        for i in range(width):
            window[i] = counts[i] <= max_mismatches
    return out
