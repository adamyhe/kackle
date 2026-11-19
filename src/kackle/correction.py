"""Correction workflow and command-line entry point for kackle."""

import argparse
import contextlib
import multiprocessing
import os
import re
import textwrap
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import tqdm

from kackle.bigwig import (
    bigwig_chrom_sizes,
    open_bigwig_writer,
    read_chrom_values,
    write_chrom_intervals,
)
from kackle.cli_common import (
    DEFAULT_MOTIFS,
    KackleArgumentFormatter,
    bigwig_chrom_names,
    intersect_ordered,
    read_bed6,
    read_chrom_sizes,
)
from kackle.computation import (
    correct_adapter_sites,
    correct_artifact_bases,
    correct_kmers,
)
from kackle.adapter import AdapterSiteProvider, parse_min_match, parse_tiers
from kackle.motifs import FastaMotifSiteProvider
from kackle.motifs import fasta_chrom_names
from kackle.motifs import parse_motif_spec


def parse_positive_int_or_auto(value):
    """Parse a positive integer CLI value, allowing ``auto``."""
    if value == "auto":
        return value
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer or 'auto'") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def parse_non_negative_int(value):
    """Parse a non-negative integer CLI value."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def resolve_chrom_workers(chrom_workers, chrom_count):
    """Resolve ``auto`` chromosome workers to a bounded integer."""
    if chrom_workers != "auto":
        if chrom_workers < 1:
            raise ValueError("chrom_workers must be >= 1")
        return min(chrom_workers, chrom_count)
    return max(1, min(chrom_count, thread_budget(), 4))


def thread_budget():
    """Return the available thread budget, honoring NUMBA_NUM_THREADS."""
    numba_threads = os.environ.get("NUMBA_NUM_THREADS")
    if numba_threads is not None:
        try:
            parsed = int(numba_threads)
        except ValueError:
            parsed = 0
        if parsed > 0:
            return parsed
    return os.cpu_count() or 1


def resolve_numba_threads(numba_threads, chrom_workers):
    """Resolve numba threads so chrom workers do not oversubscribe by default."""
    if numba_threads != "auto":
        if numba_threads < 1:
            raise ValueError("numba_threads must be >= 1")
        return numba_threads
    return max(1, thread_budget() // chrom_workers)


def motif_bed6_fnames(out_prefix, motif_specs):
    """Return deterministic BED6 output names for ordered motif specs."""
    bed6_fnames = []
    for idx, spec in enumerate(motif_specs, start=1):
        motif, max_mismatches = parse_motif_spec(spec)
        safe_motif = re.sub(r"[^A-Za-z0-9]+", "_", motif).strip("_") or "motif"
        bed6_fnames.append(f"{out_prefix}.{idx}.{safe_motif}.m{max_mismatches}.bed6")
    return bed6_fnames


def _isolated_provider(motif_site_provider, isolate_fasta):
    """A fresh FASTA provider (own pyfaidx handle, no BED6 output) when isolating."""
    if isolate_fasta and isinstance(motif_site_provider, FastaMotifSiteProvider):
        return FastaMotifSiteProvider(
            motif_site_provider.fasta_fname,
            motif_site_provider.motif_specs,
            both_strands=motif_site_provider.both_strands,
            match_backend=motif_site_provider.match_backend,
        )
    return motif_site_provider


def _anchors_from_starts(motif_specs, spec_starts, strand):
    """Ordered int32 anchor arrays for one strand from ``starts_for_chrom`` output.

    Minus anchors are ``start + len(motif)``, the BED ``end`` column.
    """
    anchors = []
    for spec, starts in zip(motif_specs, spec_starts):
        strand_starts = starts[strand]
        if strand == "-":
            strand_starts = strand_starts + len(spec.motif)
        anchors.append(strand_starts.astype(np.int32))
    return anchors


def _provider_anchor_arrays(
    motif_site_provider, chrom, strand, keep_sites, isolate_fasta, strands=None
):
    """Ordered per-pass anchor arrays for one chromosome from a motif provider.

    ``strand`` is one strand (``"+"``/``"-"``) or a tuple of strands; with a
    tuple, a ``{strand: anchor_arrays}`` dict is returned instead of a list.
    Providers with ``starts_for_chrom`` (``FastaMotifSiteProvider``) return
    start arrays directly, and only the strands asked for are located unless
    ``keep_sites`` asks for both (for ``--out-bed6-prefix``). Other providers
    go through their ``for_chrom`` BED tables.

    Returns ``(anchors, sites)``, where ``sites`` is the ``starts_for_chrom``
    output when ``keep_sites`` is true, else ``None``.
    """
    wanted = (strand,) if isinstance(strand, str) else tuple(strand)
    provider = _isolated_provider(motif_site_provider, isolate_fasta)
    if not hasattr(provider, "starts_for_chrom"):
        beds = provider.for_chrom(chrom)
        by_strand = {s: motif_anchor_arrays(beds, s) for s in wanted}
        sites = None
    else:
        located = ("+", "-") if keep_sites else wanted
        spec_starts = provider.starts_for_chrom(chrom, strands=located)
        by_strand = {
            s: _anchors_from_starts(provider.motif_specs, spec_starts, s) for s in wanted
        }
        sites = spec_starts if keep_sites else None
    if isinstance(strand, str):
        return by_strand[strand], sites
    return by_strand, sites


def motif_anchor_arrays(motif_beds, strand, chrom=None):
    """Return ordered motif anchor arrays for one strand and optional chromosome."""
    if motif_beds is None:
        return []
    anchor_col = "start" if strand == "+" else "end"
    arrays = []
    for bed in motif_beds:
        keep = bed.strand == strand
        if chrom is not None:
            keep &= bed.chrom == chrom
        arrays.append(bed.loc[keep, anchor_col].to_numpy(dtype=np.int32, copy=True))
    return arrays


def bed_anchor_index(motif_beds, chroms, strands=("+", "-")):
    """``{chrom: {strand: [anchors per BED]}}`` from BED6 tables, grouped once.

    Equivalent to calling :func:`motif_anchor_arrays` for every chromosome
    and strand, without re-filtering each table per chromosome.
    """
    empty = np.empty(0, dtype=np.int32)
    index = {chrom: {strand: [] for strand in strands} for chrom in chroms}
    for bed in motif_beds:
        for strand in strands:
            anchor_col = "start" if strand == "+" else "end"
            rows = bed.loc[bed.strand == strand, ["chrom", anchor_col]]
            groups = {
                chrom: group[anchor_col].to_numpy(dtype=np.int32, copy=True)
                for chrom, group in rows.groupby("chrom", sort=False, observed=True)
            }
            for chrom in chroms:
                index[chrom][strand].append(groups.get(chrom, empty))
    return index


def _needs_bed6_sites(motif_site_provider, chrom):
    """Whether this chromosome's sites still have to be written to BED6 files."""
    return (
        isinstance(motif_site_provider, FastaMotifSiteProvider)
        and motif_site_provider.bed6_fnames is not None
        and chrom not in motif_site_provider._written_chroms
    )


def _append_motif_sites_for_chrom(motif_site_provider, chrom, sites):
    """Append generated FASTA motif sites once, preserving chromosome order."""
    if sites is None or not isinstance(motif_site_provider, FastaMotifSiteProvider):
        return
    motif_site_provider.write_bed6(chrom, sites)


def _set_numba_threads(numba_threads):
    if numba_threads is not None:
        from numba import set_num_threads

        set_num_threads(numba_threads)


def _correct_strand_signal(
    bw_fname,
    chrom,
    strand,
    anchor_arrays,
    source,
    target,
    threshold,
    n_jobs,
    start=0,
    end=None,
    resample="window",
    pass_thresholds=None,
):
    """Read, correct and sparsify one strand of a chromosome or of ``[start, end)``.

    Anchors are relative to ``start``. Returns ``(positions, values)``: the
    int32 positions (relative to ``start``) of the corrected signal's nonzero
    bases and their (integer) values. The dense arrays are freed before
    returning.
    """
    region = {} if end is None and not start else {"start": start, "end": end}
    values = read_chrom_values(bw_fname, chrom, n_jobs=n_jobs, **region).astype(
        np.int32
    )
    if strand == "-":
        np.abs(values, out=values)
    if pass_thresholds is not None:
        # One combined pass (adapter mode): every array is a threshold tier.
        starts = np.concatenate([np.asarray(a, dtype=np.int32) for a in anchor_arrays])
        thresholds = np.concatenate(
            [np.full(len(a), t, dtype=np.float64) for a, t in zip(anchor_arrays, pass_thresholds)]
        )
        if starts.shape[0]:
            values = correct_adapter_sites(values, starts, thresholds, source, strand)
        anchor_arrays = []
    for starts in anchor_arrays:
        if starts.shape[0] == 0:
            continue
        if resample == "base":
            values = correct_artifact_bases(values, starts, source, threshold, strand)
        else:
            values = correct_kmers(values, starts, source, target, threshold, strand)
    positions = np.flatnonzero(values > 0).astype(np.int32)
    return positions, values[positions]


def _sparse_frame(chrom, positions, values):
    """The ``chrom``/``start``/``end``/``value`` rows ``kmer_resample`` returns."""
    return pd.DataFrame(
        {
            "chrom": np.full(positions.shape[0], chrom, dtype=object),
            "start": positions,
            "end": positions + 1,
            "value": values.astype(np.float64),
        }
    )


def _correct_chromosome(
    bw_fname,
    chrom,
    motif_beds,
    motif_start_arrays,
    motif_site_provider,
    strand,
    source,
    target,
    threshold,
    isolate_fasta_provider=False,
    numba_threads=None,
    keep_sites=False,
):
    """Correct one strand of one chromosome.

    Returns ``(chrom, positions, values, sites or None)``; ``sites`` is the
    provider's ``starts_for_chrom`` output when ``keep_sites`` is set (so the
    parent can write ``--out-bed6-prefix``).
    """
    _set_numba_threads(numba_threads)
    chrom_sites = None
    if motif_site_provider is not None:
        motif_start_arrays, chrom_sites = _provider_anchor_arrays(
            motif_site_provider,
            chrom,
            strand,
            keep_sites=keep_sites,
            isolate_fasta=isolate_fasta_provider,
        )
    positions, values = _correct_strand_signal(
        bw_fname,
        chrom,
        strand,
        motif_start_arrays or [],
        source,
        target,
        threshold,
        numba_threads if numba_threads is not None else 1,
    )
    return chrom, positions, values, chrom_sites


def _provider_region_anchor_arrays(
    motif_site_provider, chrom, start, end, strands, keep_sites, isolate_fasta
):
    """``{strand: anchor_arrays}`` for anchors in ``[start, end)``, relative to ``start``.

    The provider must have ``starts_for_region``. A minus anchor is
    ``site start + len(motif)``, so sites are located from
    ``start - max_motif_length``. ``sites`` (when ``keep_sites``) holds the
    sites that *start* in ``[start, end)``, so chunks' sites tile the
    chromosome for ``--out-bed6-prefix``.
    """
    provider = _isolated_provider(motif_site_provider, isolate_fasta)
    located = ("+", "-") if keep_sites else tuple(strands)
    spec_starts = provider.starts_for_region(
        chrom, start - provider.max_motif_length, end, strands=located
    )
    anchors = {}
    for strand in strands:
        arrays = []
        for spec, starts in zip(provider.motif_specs, spec_starts):
            anchor = starts[strand] + (len(spec.motif) if strand == "-" else 0)
            anchor = anchor[(anchor >= start) & (anchor < end)] - start
            arrays.append(anchor.astype(np.int32))
        anchors[strand] = arrays
    sites = None
    if keep_sites:
        sites = [
            {strand: arr[arr >= start] for strand, arr in starts.items()}
            for starts in spec_starts
        ]
    return anchors, sites


def _correct_region_both_strands(
    region,
    bw_fnames,
    strand_anchors,
    motif_site_provider,
    source,
    target,
    threshold,
    isolate_fasta_provider=False,
    numba_threads=None,
    keep_sites=False,
    resample="window",
    pass_thresholds=None,
):
    """Correct both strands of one chromosome region, one strand at a time.

    ``region`` is ``(chrom, start, end)``; ``end=None`` (with ``start=0``)
    means the whole chromosome. ``bw_fnames`` maps ``"+"``/``"-"`` to input
    bigWigs. Anchors, relative to ``start``, come from ``strand_anchors``
    (``{strand: anchor_arrays}``) or, when it is ``None``, from
    ``motif_site_provider``, which locates both strands in one call.
    Returns ``(region, {strand: (positions, values)}, sites or None)``, with
    absolute positions.
    """
    _set_numba_threads(numba_threads)
    chrom, start, end = region
    strands = tuple(bw_fnames)
    chrom_sites = None
    if strand_anchors is None:
        if end is None:
            strand_anchors, chrom_sites = _provider_anchor_arrays(
                motif_site_provider,
                chrom,
                strands,
                keep_sites=keep_sites,
                isolate_fasta=isolate_fasta_provider,
            )
        else:
            strand_anchors, chrom_sites = _provider_region_anchor_arrays(
                motif_site_provider,
                chrom,
                start,
                end,
                strands,
                keep_sites=keep_sites,
                isolate_fasta=isolate_fasta_provider,
            )
    n_jobs = numba_threads if numba_threads is not None else 1
    corrected = {}
    for strand in strands:
        positions, values = _correct_strand_signal(
            bw_fnames[strand],
            chrom,
            strand,
            strand_anchors[strand],
            source,
            target,
            threshold,
            n_jobs,
            start=start,
            end=end,
            resample=resample,
            pass_thresholds=pass_thresholds,
        )
        if start:
            positions += start
        corrected[strand] = (positions, values)
        strand_anchors[strand] = None
    return region, corrected, chrom_sites


def _safe_cut(anchors, near, reach, lo, hi):
    """The cut ``x`` closest to ``near`` with no anchor in ``(x - reach, x + reach)``.

    ``anchors`` must hold every anchor in ``[lo, hi)`` (sorted); anchors
    outside that range are unknown, so ``x`` is kept within
    ``[lo + reach, hi - reach]``. Returns ``None`` when there is no such cut.
    """
    pts = np.concatenate(([lo], anchors, [hi])).astype(np.int64)
    left = pts[:-1] + reach
    right = pts[1:] - reach
    ok = left <= right
    if not ok.any():
        return None
    x = np.clip(near, left[ok], right[ok])
    return int(x[np.argmin(np.abs(x - near))])


def plan_chunks(length, chunk_size, anchors_near, reach, window=1 << 15):
    """Split ``[0, length)`` into chunks that can be corrected independently.

    A site reads and writes only ``[anchor - reach, anchor + reach)``, with
    ``reach = max(source, target)``, and sites closer than
    ``source + target <= 2 * reach`` are merged into one cluster. So a cut
    ``x`` with no anchor of any pass or strand in ``(x - reach, x + reach)``
    separates the two sides completely, and correcting each chunk on its own
    gives the same result as correcting the whole chromosome (up to the
    random draws). Cuts are placed near ``chunk_size`` spacing;
    ``anchors_near(lo, hi)`` returns the sorted anchors in ``[lo, hi)``. A
    boundary with no safe cut within about 4 Mb is dropped (its chunks merge).
    Returns ``[(start, end), ...]``; ``chunk_size`` of 0 or ``None`` gives one
    chunk.
    """
    if not chunk_size or length <= chunk_size:
        return [(0, length)]
    n_chunks = -(-length // chunk_size)
    cuts = [0]
    for i in range(1, n_chunks):
        near = (i * length) // n_chunks
        cut = None
        w = window
        while cut is None and w <= 1 << 22:
            lo, hi = max(0, near - w), min(length, near + w)
            cut = _safe_cut(anchors_near(lo, hi), near, reach, lo, hi)
            w *= 4
        if cut is not None and cuts[-1] < cut < length:
            cuts.append(cut)
    cuts.append(length)
    return list(zip(cuts[:-1], cuts[1:]))


def _fasta_anchors_near(provider, chrom):
    """``anchors_near`` for :func:`plan_chunks`: all strands and specs, from the FASTA."""

    def anchors_near(lo, hi):
        spec_starts = provider.starts_for_region(
            chrom, lo - provider.max_motif_length, hi, strands=("+", "-")
        )
        parts = []
        for spec, starts in zip(provider.motif_specs, spec_starts):
            parts.append(starts["+"])
            parts.append(starts["-"] + len(spec.motif))
        anchors = np.unique(np.concatenate(parts))
        return anchors[(anchors >= lo) & (anchors < hi)]

    return anchors_near


def _sorted_anchors_near(strand_anchors):
    """``anchors_near`` for :func:`plan_chunks` from in-memory per-pass anchor arrays."""
    anchors = np.unique(
        np.concatenate(
            [np.empty(0, dtype=np.int64)]
            + [a.astype(np.int64) for arrays in strand_anchors.values() for a in arrays]
        )
    )

    def anchors_near(lo, hi):
        return anchors[np.searchsorted(anchors, lo) : np.searchsorted(anchors, hi)]

    return anchors_near


def _slice_anchors(strand_anchors, start, end):
    """Per-pass anchors in ``[start, end)``, relative to ``start``."""
    return {
        strand: [
            (a[(a >= start) & (a < end)] - start).astype(np.int32) for a in arrays
        ]
        for strand, arrays in strand_anchors.items()
    }


def _concat_sites(chunk_sites):
    """Join per-chunk ``starts_for_region`` sites (in order) into one chromosome's."""
    if any(sites is None for sites in chunk_sites):
        return None
    return [
        {
            strand: np.concatenate([sites[i][strand] for sites in chunk_sites])
            for strand in chunk_sites[0][i]
        }
        for i in range(len(chunk_sites[0]))
    ]


def _resolve_motif_beds(bed6_fname, motif_beds, motif_site_provider):
    if motif_beds is None and motif_site_provider is None:
        if bed6_fname is None:
            raise ValueError(
                "Either bed6_fname, motif_beds, or motif_site_provider must be provided"
            )
        motif_beds = [read_bed6(bed6_fname)]
    return motif_beds


def _run_chromosomes(
    task, task_args, chroms, chrom_workers, worker_backend, verbose, desc, on_result
):
    """Run ``task(key, *task_args[key])`` per key; hand results over in ``chroms`` order.

    Keys are chromosomes or ``(chrom, start, end)`` regions. ``on_result(result)``
    is called for each key's result as soon as it and every key before it have
    finished, so output can be streamed in order while later keys still run.
    """
    if chrom_workers == 1:
        for chrom in tqdm.tqdm(chroms, disable=not verbose, desc=desc):
            on_result(task(chrom, *task_args[chrom]))
        return

    if worker_backend == "process":
        # spawn, not fork: the parent runs figwig writer threads (and may have
        # run numba threads), and forking a multi-threaded process can crash
        # or deadlock the child.
        executor = ProcessPoolExecutor(
            max_workers=chrom_workers, mp_context=multiprocessing.get_context("spawn")
        )
    else:
        executor = ThreadPoolExecutor(max_workers=chrom_workers)
    pending = {}
    next_idx = 0
    with executor:
        futures = {
            executor.submit(task, chrom, *task_args[chrom]): chrom for chrom in chroms
        }
        for future in tqdm.tqdm(
            as_completed(futures),
            total=len(futures),
            disable=not verbose,
            desc=f"{desc} ({chrom_workers} chrom workers)",
        ):
            pending[futures[future]] = future.result()
            del futures[future]
            while next_idx < len(chroms) and chroms[next_idx] in pending:
                on_result(pending.pop(chroms[next_idx]))
                next_idx += 1


def kmer_resample(
    bw_fname,
    bed6_fname=None,
    motif_beds=None,
    motif_site_provider=None,
    strand="+",
    chroms=None,
    source=10,
    target=5,
    threshold=10,
    chrom_workers=1,
    worker_backend="thread",
    numba_threads=None,
    verbose=True,
):
    """Correct one strand-specific bigWig and return nonzero output intervals.

    The ``kackle`` CLI uses :func:`correct_bigwigs` instead, which corrects
    both strands in one pass and streams them to bigWig files.

    Parameters
    ----------
    bw_fname : str or path-like
        Input bigWig path for one strand.
    bed6_fname : str or path-like, optional
        Precomputed BED6 motif locations. Used when ``motif_beds`` is omitted.
    motif_beds : list[pandas.DataFrame], optional
        Ordered BED6-style motif location tables. Each table is applied as a
        separate correction pass to preserve sequential correction semantics.
    motif_site_provider : object, optional
        Locates motif sites one chromosome at a time. ``FastaMotifSiteProvider``
        (FASTA mode) supplies sorted start arrays via ``starts_for_chrom``;
        any other object with ``for_chrom(chrom)`` returning ordered BED6-style
        tables also works.
    strand : {"+", "-"}
        Signal/motif strand to correct.
    chroms : sequence[str], optional
        Chromosomes to process. Defaults to human autosomes plus chrX/chrY.
    source, target, threshold : int
        Parameters forwarded to ``correct_kmers``.
    chrom_workers : int
        Number of chromosomes to correct concurrently. Each worker opens its
        own bigWig handle. Use ``1`` to disable cross-chromosome parallelism.
    worker_backend : {"thread", "process"}
        Executor backend for chromosome workers. ``process`` can improve CPU
        use for mixed Python/native FASTA workflows; ``thread`` avoids
        pickling large in-memory BED tables for library callers.
    numba_threads : int, optional
        Threads available to numba inside each chromosome worker.
    verbose : bool
        Whether to show a progress bar.

    Returns
    -------
    pandas.DataFrame
        Sparse ``chrom``, ``start``, ``end``, ``value`` rows for writing to
        bigWig.
    """
    if strand not in ["+", "-"]:
        raise ValueError("Strand must be '+' or '-'")
    if worker_backend not in {"thread", "process"}:
        raise ValueError("worker_backend must be 'thread' or 'process'")
    if getattr(motif_site_provider, "pass_thresholds", None) is not None:
        raise ValueError("adapter site providers are only supported by correct_bigwigs")
    motif_beds = _resolve_motif_beds(bed6_fname, motif_beds, motif_site_provider)

    if chroms is None:
        chroms = ["chr" + str(i) for i in range(1, 23)] + ["chrX", "chrY"]
    chrom_workers = resolve_chrom_workers(chrom_workers, len(chroms))
    parallel = chrom_workers > 1
    if motif_site_provider is None:
        index = bed_anchor_index(motif_beds, chroms, strands=(strand,))
    task_args = {
        chrom: (
            None,
            index[chrom][strand] if motif_site_provider is None else None,
            motif_site_provider,
            strand,
            source,
            target,
            threshold,
            parallel,
            numba_threads,
            _needs_bed6_sites(motif_site_provider, chrom),
        )
        for chrom in chroms
    }
    frames = []

    def collect(result):
        chrom, positions, values, chrom_sites = result
        _append_motif_sites_for_chrom(motif_site_provider, chrom, chrom_sites)
        frames.append(_sparse_frame(chrom, positions, values))

    _run_chromosomes(
        _correct_chromosome_task,
        {chrom: (bw_fname, *args) for chrom, args in task_args.items()},
        chroms,
        chrom_workers,
        worker_backend,
        verbose,
        f"Correcting {'plus' if strand == '+' else 'minus'} strand kmers",
        collect,
    )
    return pd.concat(frames, ignore_index=True)


def _correct_chromosome_task(chrom, bw_fname, *args):
    """``_correct_chromosome`` with ``chrom`` first, for :func:`_run_chromosomes`."""
    return _correct_chromosome(bw_fname, chrom, *args)


DEFAULT_CHUNK_SIZE = 10_000_000


def _signal_lengths(in_bw_fnames, chroms):
    """Per-chromosome signal length shared by both input bigWigs, or None if they differ."""
    sizes = [bigwig_chrom_sizes(fname) for fname in in_bw_fnames.values()]
    lengths = {}
    for chrom in chroms:
        found = {int(s[chrom]) for s in sizes if chrom in s}
        lengths[chrom] = found.pop() if len(found) == 1 else None
    return lengths


def correct_bigwigs(
    in_bw_fnames,
    out_bw_fnames,
    chrom_sizes,
    bed6_fname=None,
    motif_beds=None,
    motif_site_provider=None,
    source=10,
    target=5,
    threshold=10,
    chrom_workers=1,
    worker_backend="thread",
    numba_threads=None,
    write_jobs=None,
    chunk_size=DEFAULT_CHUNK_SIZE,
    verbose=True,
    resample="window",
):
    """Correct both strands in one pass and stream them to bigWigs.

    Chromosomes longer than ``chunk_size`` are split into chunks at cuts that
    no site's reads or writes cross (see :func:`plan_chunks`), so chunks are
    corrected independently and large chromosomes spread across workers.
    Each task locates (or looks up) its motif sites for both strands, then
    reads, corrects and sparsifies each strand in turn. Finished chromosomes
    are written to both output bigWigs in ``chrom_sizes`` order as soon as
    every earlier chunk is done, so only the sparse results of chunks waiting
    their turn are held.

    Parameters
    ----------
    in_bw_fnames, out_bw_fnames : dict
        ``{"+": path, "-": path}`` input and output bigWigs. Minus-strand
        input values are converted to absolute counts.
    chrom_sizes : sequence of (str, int)
        Chromosomes to correct, in output order, and their lengths.
    bed6_fname, motif_beds, motif_site_provider
        Motif sites, as in :func:`kmer_resample`. Chunking needs BED tables or
        a provider with ``starts_for_region`` (``FastaMotifSiteProvider``);
        other providers get whole chromosomes.
    source, target, threshold : int
        Parameters forwarded to ``correct_kmers``.
    resample : {"window", "base"}
        ``"window"`` resamples whole target windows (``correct_kmers``);
        ``"base"`` redraws only each flagged site's artifact base
        (``correct_artifact_bases``; experimental, ignores ``target``).
    chrom_workers, worker_backend, numba_threads, verbose
        As in :func:`kmer_resample`.
    write_jobs : int, optional
        Compression threads per output bigWig. Defaults to the thread budget.
    chunk_size : int, optional
        Target chunk length in bases. ``0`` or ``None`` corrects whole
        chromosomes.
    """
    if worker_backend not in {"thread", "process"}:
        raise ValueError("worker_backend must be 'thread' or 'process'")
    strands = ("+", "-")
    in_bw_fnames = {strand: in_bw_fnames[strand] for strand in strands}
    motif_beds = _resolve_motif_beds(bed6_fname, motif_beds, motif_site_provider)
    chrom_sizes = [(chrom, int(size)) for chrom, size in chrom_sizes]
    chroms = [chrom for chrom, _ in chrom_sizes]
    chrom_workers = resolve_chrom_workers(chrom_workers, len(chroms))
    parallel = chrom_workers > 1
    if write_jobs is None:
        write_jobs = thread_budget()
    index = (
        bed_anchor_index(motif_beds, chroms) if motif_site_provider is None else None
    )
    can_split = index is not None or hasattr(motif_site_provider, "starts_for_region")
    lengths = (
        _signal_lengths(in_bw_fnames, chroms)
        if chunk_size and can_split
        else dict.fromkeys(chroms)
    )
    reach = max(source, target)
    pass_thresholds = getattr(motif_site_provider, "pass_thresholds", None)

    regions = []
    task_args = {}
    last_region = {}
    for chrom in chroms:
        length = lengths[chrom]
        chunks = [(0, None)]
        if length is not None and length > chunk_size:
            anchors_near = (
                _fasta_anchors_near(motif_site_provider, chrom)
                if index is None
                else _sorted_anchors_near(index[chrom])
            )
            planned = plan_chunks(length, chunk_size, anchors_near, reach)
            if len(planned) > 1:
                chunks = planned
        keep_sites = _needs_bed6_sites(motif_site_provider, chrom)
        chrom_anchors = None if index is None else index.pop(chrom)
        for start, end in chunks:
            region = (chrom, start, end)
            anchors = chrom_anchors
            if anchors is not None and end is not None:
                anchors = _slice_anchors(chrom_anchors, start, end)
            regions.append(region)
            task_args[region] = (
                in_bw_fnames,
                anchors,
                motif_site_provider,
                source,
                target,
                threshold,
                parallel,
                numba_threads,
                keep_sites,
                resample,
                pass_thresholds,
            )
        last_region[chrom] = regions[-1]
        del chrom_anchors

    if verbose and len(regions) > len(chroms):
        print(f"Split {len(chroms)} chromosomes into {len(regions)} chunks.")

    with contextlib.ExitStack() as stack:
        writers = {
            strand: stack.enter_context(
                open_bigwig_writer(out_bw_fnames[strand], chrom_sizes, n_jobs=write_jobs)
            )
            for strand in strands
        }
        buffered = {strand: [] for strand in strands}
        buffered_sites = []

        def write(result):
            region, corrected, sites = result
            chrom = region[0]
            for strand in strands:
                buffered[strand].append(corrected[strand])
            buffered_sites.append(sites)
            if region != last_region[chrom]:
                return
            chrom_sites = (
                buffered_sites[0] if len(buffered_sites) == 1 else _concat_sites(buffered_sites)
            )
            _append_motif_sites_for_chrom(motif_site_provider, chrom, chrom_sites)
            buffered_sites.clear()
            for strand in strands:
                parts = buffered[strand]
                positions = np.concatenate([p for p, _ in parts])
                values = np.concatenate([v for _, v in parts])
                parts.clear()
                write_chrom_intervals(writers[strand], chrom, positions, values)

        _run_chromosomes(
            _correct_region_both_strands,
            task_args,
            regions,
            chrom_workers,
            worker_backend,
            verbose,
            "Correcting kmers on both strands",
            write,
        )


def parse_args():
    """Parse command-line arguments for the ``kackle`` console script."""
    parser = argparse.ArgumentParser(
        description=(
            "Correct strand-specific bigWig signal tracks by resampling coverage "
            "around recurrent short k-mer artifact sites."
        ),
        epilog=textwrap.dedent(
            f"""\
            Examples:
              kackle -i plus.bw -I minus.bw -b sites.bed6 -o corrected.plus.bw -O corrected.minus.bw -c chrom.sizes

              kackle -i plus.bw -I minus.bw -f genome.fa -o corrected.plus.bw -O corrected.minus.bw -c chrom.sizes

            FASTA mode defaults to motif passes {", ".join(DEFAULT_MOTIFS)} and locates sites one chromosome at a time using pyfaidx and a numba motif matcher.
            """
        ),
        formatter_class=KackleArgumentFormatter,
    )
    parser.add_argument(
        "-i",
        "--in_pl_bw",
        metavar="BIGWIG",
        type=str,
        required=True,
        help="Input plus-strand bigWig signal to correct.",
    )
    parser.add_argument(
        "-I",
        "--in_mn_bw",
        metavar="BIGWIG",
        type=str,
        required=True,
        help=(
            "Input minus-strand bigWig signal to correct. Values are converted "
            "to absolute counts before correction."
        ),
    )
    sites = parser.add_mutually_exclusive_group(required=True)
    sites.add_argument(
        "-b",
        "--bed6",
        metavar="BED6",
        type=str,
        help=(
            "Precomputed BED6 motif locations. Plus-strand rows use start "
            "anchors; minus-strand rows use end anchors."
        ),
    )
    sites.add_argument(
        "-f",
        "--fasta",
        metavar="FASTA",
        type=str,
        help=(
            "Reference FASTA used to locate motif sites directly. This avoids "
            "a separate seqkit locate preprocessing step."
        ),
    )
    parser.add_argument(
        "-o",
        "--out_pl_bw",
        metavar="BIGWIG",
        type=str,
        required=True,
        help="Output path for the corrected plus-strand bigWig.",
    )
    parser.add_argument(
        "-O",
        "--out_mn_bw",
        metavar="BIGWIG",
        type=str,
        required=True,
        help="Output path for the corrected minus-strand bigWig.",
    )
    parser.add_argument(
        "-c",
        "--chrom_sizes",
        metavar="TSV",
        type=str,
        required=True,
        help=(
            "Two-column chromosome sizes file used to order chromosomes and "
            "write the output bigWigs."
        ),
    )
    parser.add_argument(
        "-k",
        "--motif",
        metavar="KMER[:MISMATCHES]",
        action="append",
        default=None,
        help=(
            "Motif pass for FASTA mode. Use KMER for exact matching or "
            "KMER:N for up to N mismatches. May be repeated; order is the "
            "correction order. Defaults to TGG:0 then TGGAA:1."
        ),
    )
    parser.add_argument(
        "--adapter",
        metavar="SEQ",
        default=None,
        help=(
            "Experimental, FASTA mode: derive candidate sites from this 3' "
            "adapter sequence (IUPAC; e.g. TGGAATTCTCGG) instead of -k motifs, "
            "and correct them in one pass with base-only resampling. A site is "
            "a position whose sequence matches the adapter's 5' end for at "
            "least --adapter-min-match bases."
        ),
    )
    parser.add_argument(
        "--adapter-min-match",
        metavar="L0[,L1,...]",
        default="3,5",
        help=(
            "Shortest adapter prefix that makes a candidate, with 0, 1, ... "
            "mismatches in it; the list length minus one is the most "
            "mismatches allowed."
        ),
    )
    parser.add_argument(
        "--adapter-tiers",
        metavar="LEN[/MAXMM]:FOLD[,...]",
        default="4/0:7,8:5,10:3",
        help=(
            "Other fold thresholds by match: a site matching at least LEN "
            "adapter bases (mismatched bases count toward LEN), with at most "
            "MAXMM mismatches if given, qualifies for FOLD. A site is flagged "
            "at the lowest FOLD it qualifies for, or --threshold if none. "
            "Empty for none."
        ),
    )
    parser.add_argument(
        "-P",
        "--only-positive-strand",
        action="store_true",
        help=(
            "In FASTA mode, locate only forward-strand motif matches instead "
            "of searching both motif and reverse-complement matches."
        ),
    )
    parser.add_argument(
        "--out-bed6-prefix",
        metavar="PREFIX",
        help=(
            "When using --fasta, write generated motif sites to ordered BED6 "
            "files named PREFIX.N.MOTIF.mMISMATCHES.bed6."
        ),
    )
    parser.add_argument(
        "--motif-match-backend",
        choices=["auto", "numba", "python", "ahocorasick"],
        default="auto",
        help=(
            "Motif matcher for --fasta mode; both give identical sites. auto "
            "uses numba, or python when numba JIT is disabled. python runs "
            "str.find per mismatch variant (a slower reference). ahocorasick "
            "is a deprecated alias for auto."
        ),
    )
    parser.add_argument(
        "-s",
        "--source",
        metavar="BP",
        type=int,
        default=10,
        help=(
            "Half-window size around a candidate motif used to estimate local "
            "source coverage."
        ),
    )
    parser.add_argument(
        "-t",
        "--target",
        metavar="BP",
        type=int,
        default=5,
        help=("Half-window size around a candidate motif resampled during correction."),
    )
    parser.add_argument(
        "-r",
        "--threshold",
        metavar="COUNT",
        type=int,
        default=10,
        help=(
            "Minimum suspicious signal at the strand-specific artifact anchor "
            "before a motif site is corrected."
        ),
    )
    parser.add_argument(
        "--resample",
        choices=("window", "base"),
        default=None,
        help=(
            "What to redraw at a flagged site: 'window' (the default, except "
            "with --adapter) resamples the whole target window from the "
            "flanks; 'base' (experimental) redraws only the artifact base "
            "from its local mean."
        ),
    )
    parser.add_argument(
        "-w",
        "--chrom-workers",
        metavar="N|auto",
        type=parse_positive_int_or_auto,
        default="auto",
        help=(
            "Number of chromosomes to correct concurrently, or auto to use up "
            "to four chromosome workers. Each worker opens its own bigWig handle."
        ),
    )
    parser.add_argument(
        "--numba-threads",
        metavar="N|auto",
        type=parse_positive_int_or_auto,
        default="auto",
        help=(
            "Threads used by numba inside each chromosome worker. auto divides "
            "available CPUs across chromosome workers."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        metavar="BP",
        type=parse_non_negative_int,
        default=DEFAULT_CHUNK_SIZE,
        help=(
            "Split chromosomes longer than this into chunks corrected in "
            "parallel. Cuts are placed where no motif site's window crosses "
            "them, so output is unchanged. 0 corrects whole chromosomes."
        ),
    )
    parser.add_argument(
        "--worker-backend",
        choices=["process", "thread"],
        default="process",
        help=(
            "Chromosome parallelism backend. process uses separate Python "
            "interpreters for better CPU use in FASTA mode; thread avoids "
            "pickling large in-memory BED tables."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show per-chromosome progress bars while correcting.",
    )
    args = parser.parse_args()
    if args.adapter is not None:
        if not args.fasta:
            parser.error("--adapter needs -f/--fasta")
        if args.motif or args.out_bed6_prefix:
            parser.error("--adapter can't be combined with -k or --out-bed6-prefix")
        if args.resample == "window":
            parser.error("--adapter uses base-only resampling")
        args.resample = "base"
    elif args.resample is None:
        args.resample = "window"
    if args.bed6 and args.motif:
        parser.error("--motif can only be used with --fasta")
    if args.bed6 and args.only_positive_strand:
        parser.error("--only-positive-strand can only be used with --fasta")
    if args.bed6 and args.out_bed6_prefix:
        parser.error("--out-bed6-prefix can only be used with --fasta")
    return args


def run_correction():
    """Run the command-line correction workflow."""
    args = parse_args()
    chrom_sizes = read_chrom_sizes(args.chrom_sizes)
    chrom_sources = [
        bigwig_chrom_names(args.in_pl_bw),
        bigwig_chrom_names(args.in_mn_bw),
    ]
    if args.fasta:
        chrom_sources.append(fasta_chrom_names(args.fasta))
    chrom_sizes = intersect_ordered(
        chrom_sizes, *chrom_sources, key=lambda chrom_size: chrom_size[0]
    )
    if not chrom_sizes:
        raise ValueError(
            "No chromosomes are present in the intersection of chrom.sizes, "
            "input bigWigs, and FASTA when provided"
        )
    chroms = [chrom for chrom, _ in chrom_sizes]
    chrom_workers = resolve_chrom_workers(args.chrom_workers, len(chroms))
    numba_threads = resolve_numba_threads(args.numba_threads, chrom_workers)
    if chrom_workers == 1 or args.worker_backend == "thread":
        from numba import set_num_threads

        set_num_threads(numba_threads)
    if args.verbose:
        print(
            "Using "
            f"{chrom_workers} chromosome worker(s) and "
            f"{numba_threads} numba thread(s) per worker "
            f"with the {args.worker_backend} backend."
        )
    motif_site_provider = None
    if args.adapter is not None:
        motif_site_provider = AdapterSiteProvider(
            args.fasta,
            args.adapter,
            min_match=parse_min_match(args.adapter_min_match),
            tiers=parse_tiers(args.adapter_tiers),
            threshold=args.threshold,
            both_strands=not args.only_positive_strand,
        )
    elif args.fasta:
        motif_specs = args.motif or DEFAULT_MOTIFS
        bed6_fnames = (
            motif_bed6_fnames(args.out_bed6_prefix, motif_specs)
            if args.out_bed6_prefix
            else None
        )
        motif_site_provider = FastaMotifSiteProvider(
            args.fasta,
            motif_specs,
            both_strands=not args.only_positive_strand,
            match_backend=args.motif_match_backend,
            bed6_fnames=bed6_fnames,
        )

    correct_bigwigs(
        {"+": args.in_pl_bw, "-": args.in_mn_bw},
        {"+": args.out_pl_bw, "-": args.out_mn_bw},
        chrom_sizes,
        bed6_fname=args.bed6,
        motif_site_provider=motif_site_provider,
        source=args.source,
        target=args.target,
        threshold=args.threshold,
        chrom_workers=chrom_workers,
        worker_backend=args.worker_backend,
        numba_threads=numba_threads,
        write_jobs=thread_budget(),
        chunk_size=args.chunk_size,
        verbose=args.verbose,
        resample=args.resample,
    )
