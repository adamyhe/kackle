"""Metaplot utilities for before/after k-mer correction diagnostics."""

import argparse
import multiprocessing
import textwrap
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pandas as pd

from kackle.bigwig import bigwig_chrom_sizes, read_chrom_values
from kackle.cli_common import (
    DEFAULT_MOTIFS,
    KackleArgumentFormatter,
    bigwig_chrom_names,
    intersect_ordered,
    read_bed6,
    read_chrom_sizes,
)
from kackle.motifs import FastaMotifSiteProvider
from kackle.motifs import fasta_chrom_names


def motif_beds_from_fasta(
    fasta_fname,
    motif_specs,
    chroms,
    both_strands=True,
    match_backend="auto",
):
    """Locate motif BED rows chromosome-by-chromosome from FASTA."""
    provider = FastaMotifSiteProvider(
        fasta_fname,
        motif_specs,
        both_strands=both_strands,
        match_backend=match_backend,
    )
    beds = []
    for chrom in chroms:
        beds.extend(provider.for_chrom(chrom))
    if not beds:
        return pd.DataFrame(
            [], columns=["chrom", "start", "end", "name", "score", "strand"]
        )
    return pd.concat(beds, ignore_index=True)


def motif_sites_by_strand(bed):
    """Return motif start sites split into plus and minus BED strands."""
    return {
        "+": bed.loc[bed.strand == "+", ["chrom", "start"]],
        "-": bed.loc[bed.strand == "-", ["chrom", "start"]],
    }


def _read_centered_windows(bw, chroms, centers, flank):
    """Read ``2 * flank + 1`` bases around each center, NaN outside the chromosome."""
    width = flank * 2 + 1
    starts = centers - flank
    read_starts = np.maximum(starts, 0)
    values = bw.read(chroms, read_starts, width, missing=0.0, oob=np.nan)
    shift = read_starts - starts
    for row in np.flatnonzero(shift > 0):
        offset = min(int(shift[row]), width)
        values[row, offset:] = values[row, : width - offset].copy()
        values[row, :offset] = np.nan
    return values.astype(np.float64)


METAPLOT_BATCH_SITES = 100_000


def metaplot_profile(
    before_bw, after_bw, sites, flank, absolute=False, batch_size=METAPLOT_BATCH_SITES
):
    """Return mean before/after signal centered on motif starts.

    Parameters
    ----------
    before_bw, after_bw : figwig.BigWigReader
        Open bigWig readers (anything with ``chrom_sizes`` and a figwig-style
        ``read(chroms, starts, width, missing=..., oob=...)``).
    sites : pandas.DataFrame
        Rows with ``chrom`` and ``start`` columns. ``start`` is the centered
        motif coordinate.
    flank : int
        Number of bases shown on each side of the motif start.
    absolute : bool
        Whether to aggregate absolute signal values. This is useful for
        negative-strand bigWigs whose input values are negative.
    batch_size : int
        Sites read per batch. Each batch holds a few ``(batch_size, 2 * flank
        + 1)`` arrays, so memory stays bounded for genome-wide site sets
        (~100M TGG sites per strand on hg38) at the cost of more read calls.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    width = flank * 2 + 1
    x = np.arange(-flank, flank + 1)
    available = set(before_bw.chrom_sizes).intersection(after_bw.chrom_sizes)
    sites = sites.loc[sites["chrom"].isin(available)]

    before_sum = np.zeros(width, dtype=np.float64)
    after_sum = np.zeros(width, dtype=np.float64)
    counts = np.zeros(width, dtype=np.int64)
    all_chroms = sites["chrom"].to_numpy(dtype=object)
    all_centers = sites["start"].to_numpy(dtype=np.int64)
    for lo in range(0, len(all_centers), batch_size):
        chroms = all_chroms[lo : lo + batch_size].astype(str)
        centers = all_centers[lo : lo + batch_size]
        before_values = _read_centered_windows(before_bw, chroms, centers, flank)
        after_values = _read_centered_windows(after_bw, chroms, centers, flank)
        if absolute:
            before_values = np.abs(before_values)
            after_values = np.abs(after_values)
        valid = np.isfinite(before_values) & np.isfinite(after_values)
        before_sum += np.where(valid, before_values, 0.0).sum(axis=0)
        after_sum += np.where(valid, after_values, 0.0).sum(axis=0)
        counts += valid.sum(axis=0)

    before_mean = np.full(width, np.nan, dtype=np.float64)
    after_mean = np.full(width, np.nan, dtype=np.float64)
    np.divide(before_sum, counts, out=before_mean, where=counts > 0)
    np.divide(after_sum, counts, out=after_mean, where=counts > 0)
    return x, before_mean, after_mean, counts


# ---------------------------------------------------------------------------
# Genome-scale path: one dense read per chromosome and track, sparse sums.
#
# For sites c_i on a chromosome of length L and a track sig,
#     sum_i sig[c_i + k] = sum_{p : sig[p] != 0} sig[p] * n_sites[p - k],
# where n_sites[q] counts sites at q. Nascent-RNA signal is sparse, so this
# costs O(nnz * width) per track instead of O(n_sites * width) window reads.
# It gives the same sums and counts as ``metaplot_profile``: bases with no
# data count as 0, and offsets falling off the chromosome are excluded.
# ---------------------------------------------------------------------------


def _dense_signal(values, length, absolute, in_place=False):
    """Trim to ``length``, optionally take ``abs``, and zero non-finite values.

    With ``in_place``, ``values`` (a writable float array the caller owns) is
    modified instead of copied, which avoids a second chromosome-sized array.
    """
    values = np.asarray(values)[:length]
    in_place = (
        in_place and values.flags.writeable and np.issubdtype(values.dtype, np.floating)
    )
    if absolute:
        values = np.abs(values, out=values) if in_place else np.abs(values)
    finite = np.isfinite(values)
    if not finite.all():
        if in_place:
            values[~finite] = 0.0
        else:
            values = np.where(finite, values, 0.0)
    return values


def _site_count_vector(centers, length, flank):
    """Per-base site multiplicity, shifted by ``flank`` so index ``q + flank`` is site ``q``.

    Only sites in ``[-flank, length + flank)`` can reach a base on the
    chromosome, so others are dropped. Uses the smallest unsigned dtype that
    holds the largest multiplicity: uint8 in practice (FASTA-mode duplicates
    come only from pooling motif specs), a quarter of int32 on long
    chromosomes.
    """
    keep = centers[(centers >= -flank) & (centers < length + flank)] + flank
    positions, multiplicity = np.unique(keep, return_counts=True)
    top = int(multiplicity.max()) if multiplicity.size else 0
    dtype = np.uint8 if top <= 0xFF else np.uint16 if top <= 0xFFFF else np.uint32
    site_count = np.zeros(length + 2 * flank, dtype=dtype)
    site_count[positions] = multiplicity
    return site_count


def _window_counts(sorted_centers, length, flank):
    """Sites whose ``c + k`` lands in ``[0, length)``, for each offset ``k``."""
    offsets = np.arange(-flank, flank + 1)
    counts = np.searchsorted(sorted_centers, length - offsets) - np.searchsorted(
        sorted_centers, -offsets
    )
    return np.maximum(counts, 0).astype(np.int64)


def _track_window_sum(signal, site_count, flank):
    """``sum_i signal[c_i + k]`` for each offset, via the nonzero bases of ``signal``."""
    pos = np.flatnonzero(signal)
    vals = signal[pos].astype(np.float64)
    track_sum = np.empty(2 * flank + 1, dtype=np.float64)
    for j, k in enumerate(range(-flank, flank + 1)):
        # pos - k + flank is always within [0, length + 2 * flank).
        track_sum[j] = np.dot(vals, site_count[pos - k + flank])
    return track_sum


def chrom_window_sums(before_values, after_values, centers, flank, absolute=False):
    """Return per-offset ``(before_sum, after_sum, counts)`` for one chromosome.

    Parameters
    ----------
    before_values, after_values : numpy.ndarray
        Dense per-base signal for the whole chromosome (not modified). If the
        lengths differ, both are truncated to the shorter one, matching
        ``metaplot_profile``, which only counts offsets that are
        on-chromosome in both tracks.
    centers : array-like of int
        Motif start coordinates (duplicates count once per occurrence).
    flank : int
        Bases on each side of the motif start.
    absolute : bool
        Aggregate absolute values (for negative minus-strand bigWigs).
    """
    length = min(len(before_values), len(after_values))
    centers = np.sort(np.asarray(centers, dtype=np.int64))
    counts = _window_counts(centers, length, flank)
    if centers.size == 0 or length == 0:
        zeros = np.zeros(2 * flank + 1, dtype=np.float64)
        return zeros, zeros.copy(), counts
    site_count = _site_count_vector(centers, length, flank)
    before_sum, after_sum = (
        _track_window_sum(_dense_signal(values, length, absolute), site_count, flank)
        for values in (before_values, after_values)
    )
    return before_sum, after_sum, counts


STRAND_ABSOLUTE = {"+": False, "-": True}


def _provider_sites(provider, chrom, region=None):
    """``{strand: starts}`` for one chromosome, pooled over motif specs.

    With ``region=(start, end)``, only sites starting in ``[start, end)``
    (needs ``starts_for_region``). Uses the provider's start arrays directly,
    so no BED6 tables are built. Providers with only ``for_chrom`` go
    through their tables.
    """
    if region is not None:
        spec_starts = provider.starts_for_region(chrom, *region)
    elif not hasattr(provider, "starts_for_chrom"):
        beds = provider.for_chrom(chrom)
        spec_starts = [
            {s: bed.loc[bed.strand == s, "start"].to_numpy(np.int64) for s in STRAND_ABSOLUTE}
            for bed in beds
        ]
        del beds
    else:
        spec_starts = provider.starts_for_chrom(chrom)
    return {
        strand: (
            np.concatenate([starts[strand] for starts in spec_starts])
            if spec_starts
            else np.empty(0, dtype=np.int64)
        )
        for strand in STRAND_ABSOLUTE
    }


def _chrom_profile_task(chrom, bigwigs, chrom_sites, provider, flank, region=None):
    """Sum one chromosome, or one chunk of it, for both strands. Runs in a worker.

    ``chrom_sites`` maps strand to motif-start arrays (BED mode); otherwise
    ``provider`` locates the sites here, so FASTA-mode motif location runs in
    parallel and no genome-wide site table is built. To keep the peak low,
    only one dense track is in memory at a time.

    ``region=(start, end)`` sums only the sites that start in
    ``[start, end)`` (``chrom_sites`` must already be limited to those), and
    reads only ``[start - flank, end + flank)`` of each track, clipped to the
    chromosome. Every window of such a site lies inside that range, so
    windows are clipped at the chromosome ends exactly as for a whole
    chromosome, and chunk sums add up to the chromosome's.
    """
    if chrom_sites is None:
        chrom_sites = _provider_sites(provider, chrom, region)
    out = {}
    for strand, absolute in STRAND_ABSOLUTE.items():
        before_fname, after_fname = bigwigs[strand]
        centers = chrom_sites.get(strand)
        if centers is None or len(centers) == 0:
            continue
        before_sizes = bigwig_chrom_sizes(before_fname)
        after_sizes = bigwig_chrom_sizes(after_fname)
        if chrom not in before_sizes or chrom not in after_sizes:
            continue  # chromosome missing from this strand's bigWig pair
        length = min(int(before_sizes[chrom]), int(after_sizes[chrom]))
        read_from, read_to = 0, length
        if region is not None:
            read_from = min(length, max(0, region[0] - flank))
            read_to = max(read_from, min(length, region[1] + flank))
        centers = np.sort(np.asarray(centers, dtype=np.int64)) - read_from
        span = read_to - read_from
        counts = _window_counts(centers, span, flank)
        site_count = _site_count_vector(centers, span, flank)
        sums = []
        for fname in (before_fname, after_fname):
            if region is None:
                values = read_chrom_values(fname, chrom)
            else:
                values = read_chrom_values(fname, chrom, start=read_from, end=read_to)
            signal = _dense_signal(values, span, absolute, in_place=True)
            sums.append(_track_window_sum(signal, site_count, flank))
            del values, signal
        del site_count
        out[strand] = (sums[0], sums[1], counts)
    return out


DEFAULT_CHUNK_SIZE = 10_000_000


def _chunk_bounds(length, chunk_size):
    """Even ``(start, end)`` chunks of at most about ``chunk_size``; ends are open.

    The first chunk starts at ``-inf`` and the last ends at ``+inf`` (as
    ``None``), so every site, even one outside ``[0, length)``, has a chunk.
    """
    if not chunk_size or length <= chunk_size:
        return [None]
    n_chunks = -(-length // chunk_size)
    cuts = [(i * length) // n_chunks for i in range(n_chunks + 1)]
    bounds = list(zip(cuts[:-1], cuts[1:]))
    bounds[0] = (None, bounds[0][1])
    bounds[-1] = (bounds[-1][0], None)
    return bounds


def _owned_sites(chrom_sites, bound):
    lo, hi = bound
    out = {}
    for strand, starts in chrom_sites.items():
        starts = np.asarray(starts, dtype=np.int64)
        keep = np.ones(starts.shape[0], dtype=bool)
        if lo is not None:
            keep &= starts >= lo
        if hi is not None:
            keep &= starts < hi
        out[strand] = starts[keep]
    return out


def _chrom_length(bigwigs, chrom):
    lengths = [
        int(sizes[chrom])
        for pair in bigwigs.values()
        for sizes in map(bigwig_chrom_sizes, pair)
        if chrom in sizes
    ]
    return max(lengths, default=0)


def metaplot_profiles(
    bigwigs,
    chroms,
    flank,
    sites=None,
    provider=None,
    chrom_workers=1,
    worker_backend="process",
    chunk_size=DEFAULT_CHUNK_SIZE,
):
    """Return ``{strand: (x, before_mean, after_mean, counts)}`` over ``chroms``.

    Parameters
    ----------
    bigwigs : dict
        ``{"+": (before_pl, after_pl), "-": (before_mn, after_mn)}`` paths.
    chroms : list of str
        Chromosomes to include.
    flank : int
        Bases on each side of the motif start.
    sites : dict, optional
        ``{chrom: {strand: starts}}``. Exactly one of ``sites`` and
        ``provider`` must be given.
    provider : FastaMotifSiteProvider, optional
        Locates sites per chromosome (or chunk) inside each worker.
    chrom_workers : int
        Chunks processed concurrently. Each worker holds one chunk's
        sequence, motif sites, two dense tracks and a site-count vector.
    worker_backend : {"process", "thread"}
        Executor used when ``chrom_workers > 1``.
    chunk_size : int, optional
        Split chromosomes longer than this many bases into chunks, so large
        chromosomes spread across workers. ``0`` or ``None`` sums whole
        chromosomes. Providers without ``starts_for_region`` get whole
        chromosomes.
    """
    if (sites is None) == (provider is None):
        raise ValueError("pass exactly one of sites or provider")
    width = 2 * flank + 1
    totals = {
        strand: [np.zeros(width), np.zeros(width), np.zeros(width, dtype=np.int64)]
        for strand in STRAND_ABSOLUTE
    }
    can_split = sites is not None or hasattr(provider, "starts_for_region")

    def tasks():
        for chrom in chroms:
            chrom_sites = None if sites is None else sites.get(chrom)
            if sites is not None and chrom_sites is None:
                continue
            bounds = [None]
            if chunk_size and can_split:
                bounds = _chunk_bounds(_chrom_length(bigwigs, chrom), chunk_size)
            for bound in bounds:
                if bound is None:
                    yield chrom, chrom_sites, None
                    continue
                region = (
                    -flank if bound[0] is None else bound[0],
                    np.iinfo(np.int64).max // 2 if bound[1] is None else bound[1],
                )
                part = None if chrom_sites is None else _owned_sites(chrom_sites, bound)
                yield chrom, part, region

    def add(result):
        for strand, parts in result.items():
            for total, part in zip(totals[strand], parts):
                total += part

    if chrom_workers <= 1:
        for chrom, chrom_sites, region in tasks():
            add(_chrom_profile_task(chrom, bigwigs, chrom_sites, provider, flank, region))
    else:
        if worker_backend == "process":
            # spawn, not fork: the parent may already run figwig/numba threads,
            # and forking a multi-threaded process can deadlock the child.
            executor = ProcessPoolExecutor(
                max_workers=chrom_workers,
                mp_context=multiprocessing.get_context("spawn"),
            )
        else:
            executor = ThreadPoolExecutor(max_workers=chrom_workers)
        with executor:
            futures = [
                executor.submit(
                    _chrom_profile_task,
                    chrom,
                    bigwigs,
                    chrom_sites,
                    provider,
                    flank,
                    region,
                )
                for chrom, chrom_sites, region in tasks()
            ]
            for future in futures:
                add(future.result())

    x = np.arange(-flank, flank + 1)
    profiles = {}
    for strand, (before_sum, after_sum, counts) in totals.items():
        before_mean = np.full(width, np.nan, dtype=np.float64)
        after_mean = np.full(width, np.nan, dtype=np.float64)
        np.divide(before_sum, counts, out=before_mean, where=counts > 0)
        np.divide(after_sum, counts, out=after_mean, where=counts > 0)
        profiles[strand] = (x, before_mean, after_mean, counts)
    return profiles


def sites_by_chrom(bed):
    """Split a BED6 table into ``{chrom: {strand: starts}}``, in first-seen chrom order."""
    sites = {}
    for (chrom, strand), rows in bed.groupby(["chrom", "strand"], sort=False):
        if strand in STRAND_ABSOLUTE:
            sites.setdefault(chrom, {})[strand] = rows["start"].to_numpy(np.int64)
    return sites


def write_metaplot_png(x, before, after, counts, strand, out_fname):
    """Write a before/after metaplot PNG for one strand."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    if np.any(counts > 0):
        ax.plot(x, before, label="before", color="#4C78A8", linewidth=2)
        ax.plot(x, after, label="after", color="#F58518", linewidth=2)
        ax.axvline(0, color="0.2", linestyle="--", linewidth=1)
        ax.legend(frameon=False)
    else:
        ax.text(
            0.5,
            0.5,
            "No motif matches",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.set_title(f"{strand} strand motif-start metaplot")
    ax.set_xlabel("Position relative to motif start (bp)")
    ax.set_ylabel("Mean signal")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out_fname, dpi=150)
    plt.close(fig)


def write_strand_metaplots(
    before_pl_bw_fname,
    before_mn_bw_fname,
    after_pl_bw_fname,
    after_mn_bw_fname,
    bed,
    out_prefix,
    flank=50,
    chrom_workers=1,
    worker_backend="process",
):
    """Write plus and minus before/after metaplots for BED sites and return output paths."""
    sites = sites_by_chrom(bed)
    profiles = metaplot_profiles(
        _bigwig_pairs(
            before_pl_bw_fname, before_mn_bw_fname, after_pl_bw_fname, after_mn_bw_fname
        ),
        list(sites),
        flank,
        sites=sites,
        chrom_workers=chrom_workers,
        worker_backend=worker_backend,
    )
    return write_profile_pngs(profiles, out_prefix)


def _bigwig_pairs(before_pl, before_mn, after_pl, after_mn):
    return {"+": (str(before_pl), str(after_pl)), "-": (str(before_mn), str(after_mn))}


def write_profile_tsv(profiles, out_fname):
    """Write ``metaplot_profiles`` output as a long-format TSV.

    Columns: ``strand``, ``offset``, ``before_mean``, ``after_mean``, ``n_sites``
    (sites contributing at that offset).
    """
    with open(out_fname, "w") as handle:
        handle.write("strand\toffset\tbefore_mean\tafter_mean\tn_sites\n")
        for strand in ("+", "-"):
            for x, before, after, count in zip(*profiles[strand]):
                handle.write(f"{strand}\t{x}\t{before:.9g}\t{after:.9g}\t{count}\n")
    return out_fname


def write_profile_pngs(profiles, out_prefix):
    """Write ``PREFIX.plus.png`` and ``PREFIX.minus.png`` from ``metaplot_profiles`` output."""
    outputs = {}
    for strand, suffix in (("+", "plus"), ("-", "minus")):
        out_fname = f"{out_prefix}.{suffix}.png"
        write_metaplot_png(*profiles[strand], strand, out_fname)
        outputs[strand] = out_fname
    return outputs


def _non_negative_int(value):
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def _positive_int_or_auto(value):
    from kackle.correction import parse_positive_int_or_auto

    return parse_positive_int_or_auto(value)


def parse_metaplot_args():
    """Parse command-line arguments for ``kackle-metaplot``."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate before/after metaplots centered on motif match starts, "
            "with separate output images for plus and minus strand matches."
        ),
        epilog=textwrap.dedent(
            """\
            Examples:
              kackle-metaplot --before-pl-bw plus.bw --before-mn-bw minus.bw \\
                --after-pl-bw corrected.plus.bw --after-mn-bw corrected.minus.bw \\
                -b sites.bed6 -o correction.metaplot

              kackle-metaplot --before-pl-bw plus.bw --before-mn-bw minus.bw \\
                --after-pl-bw corrected.plus.bw --after-mn-bw corrected.minus.bw \\
                -f genome.fa -c chrom.sizes -o correction.metaplot
            """
        ),
        formatter_class=KackleArgumentFormatter,
    )
    parser.add_argument(
        "--before-pl-bw",
        metavar="BIGWIG",
        required=True,
        help="Plus-strand bigWig before correction.",
    )
    parser.add_argument(
        "--before-mn-bw",
        metavar="BIGWIG",
        required=True,
        help="Minus-strand bigWig before correction. Values are plotted as absolute signal.",
    )
    parser.add_argument(
        "--after-pl-bw",
        metavar="BIGWIG",
        required=True,
        help="Plus-strand bigWig after correction.",
    )
    parser.add_argument(
        "--after-mn-bw",
        metavar="BIGWIG",
        required=True,
        help="Minus-strand bigWig after correction. Values are plotted as absolute signal.",
    )
    sites = parser.add_mutually_exclusive_group(required=True)
    sites.add_argument(
        "-b",
        "--bed6",
        metavar="BED6",
        help="BED6 motif matches. Metaplots are centered on the BED start column.",
    )
    sites.add_argument(
        "-f",
        "--fasta",
        metavar="FASTA",
        help="Reference FASTA used to locate motif matches before plotting.",
    )
    parser.add_argument(
        "-c",
        "--chrom-sizes",
        metavar="TSV",
        help="Chromosome sizes file. Required with --fasta.",
    )
    parser.add_argument(
        "-k",
        "--motif",
        metavar="KMER[:MISMATCHES]",
        action="append",
        default=None,
        help="Motif spec for FASTA location. May be repeated.",
    )
    parser.add_argument(
        "--motif-match-backend",
        choices=["auto", "numba", "python", "ahocorasick"],
        default="auto",
        help="Motif matcher for --fasta mode (see kackle --help); all give identical sites.",
    )
    parser.add_argument(
        "--flank",
        metavar="BP",
        type=int,
        default=50,
        help="Bases to plot on each side of the motif start.",
    )
    parser.add_argument(
        "-w",
        "--chrom-workers",
        metavar="N|auto",
        type=_positive_int_or_auto,
        default="auto",
        help=(
            "Chromosome chunks processed concurrently (motif location and "
            "summing). auto uses min(chromosomes, CPUs, 4)."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        metavar="BP",
        type=_non_negative_int,
        default=DEFAULT_CHUNK_SIZE,
        help=(
            "Split chromosomes longer than this into chunks summed in "
            "parallel. 0 sums whole chromosomes."
        ),
    )
    parser.add_argument(
        "--worker-backend",
        choices=["process", "thread"],
        default="process",
        help="Executor for chromosome workers.",
    )
    parser.add_argument(
        "-o",
        "--out-prefix",
        metavar="PREFIX",
        required=True,
        help=(
            "Output prefix. Writes PREFIX.plus.png, PREFIX.minus.png, and the "
            "plotted values as PREFIX.tsv."
        ),
    )
    args = parser.parse_args()
    if args.fasta and args.chrom_sizes is None:
        parser.error("--chrom-sizes is required with --fasta")
    return args


def run_metaplot():
    """Run the before/after metaplot workflow."""
    from kackle.correction import resolve_chrom_workers

    args = parse_metaplot_args()
    bigwigs = _bigwig_pairs(
        args.before_pl_bw, args.before_mn_bw, args.after_pl_bw, args.after_mn_bw
    )
    if args.bed6:
        sites = sites_by_chrom(read_bed6(args.bed6))
        chroms, provider = list(sites), None
    else:
        sites = None
        chroms = [chrom for chrom, _ in read_chrom_sizes(args.chrom_sizes)]
        chroms = intersect_ordered(
            chroms,
            bigwig_chrom_names(args.before_pl_bw),
            bigwig_chrom_names(args.before_mn_bw),
            bigwig_chrom_names(args.after_pl_bw),
            bigwig_chrom_names(args.after_mn_bw),
            fasta_chrom_names(args.fasta),
        )
        # Builds the .fai once here, before any worker reads the FASTA.
        provider = FastaMotifSiteProvider(
            args.fasta,
            args.motif or list(DEFAULT_MOTIFS),
            match_backend=args.motif_match_backend,
        )
    profiles = metaplot_profiles(
        bigwigs,
        chroms,
        args.flank,
        sites=sites,
        provider=provider,
        chrom_workers=resolve_chrom_workers(args.chrom_workers, max(1, len(chroms))),
        worker_backend=args.worker_backend,
        chunk_size=args.chunk_size,
    )
    outputs = write_profile_pngs(profiles, args.out_prefix)
    tsv_fname = write_profile_tsv(profiles, f"{args.out_prefix}.tsv")
    print(f"Wrote plus-strand metaplot: {outputs['+']}")
    print(f"Wrote minus-strand metaplot: {outputs['-']}")
    print(f"Wrote metaplot profiles: {tsv_fname}")
