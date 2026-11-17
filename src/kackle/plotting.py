"""Metaplot utilities for before/after k-mer correction diagnostics."""

import argparse
import multiprocessing
import textwrap
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pandas as pd

from kackle.bigwig import read_chrom_values
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
    match_backend="ahocorasick",
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


def _dense_signal(values, length, absolute):
    """Trim to ``length``, optionally take ``abs``, and zero non-finite values."""
    values = np.asarray(values[:length])
    if absolute:
        values = np.abs(values)
    if not np.isfinite(values).all():
        values = np.where(np.isfinite(values), values, 0.0)
    return values


def _site_count_vector(centers, length, flank):
    """Per-base site multiplicity, shifted by ``flank`` so index ``q + flank`` is site ``q``.

    Only sites in ``[-flank, length + flank)`` can reach a base on the
    chromosome, so others are dropped. Stored as int32 to halve memory on
    long chromosomes.
    """
    keep = centers[(centers >= -flank) & (centers < length + flank)] + flank
    site_count = np.zeros(length + 2 * flank, dtype=np.int32)
    positions, multiplicity = np.unique(keep, return_counts=True)
    site_count[positions] = multiplicity
    return site_count


def chrom_window_sums(before_values, after_values, centers, flank, absolute=False):
    """Return per-offset ``(before_sum, after_sum, counts)`` for one chromosome.

    Parameters
    ----------
    before_values, after_values : numpy.ndarray
        Dense per-base signal for the whole chromosome. If the lengths differ,
        both are truncated to the shorter one, matching ``metaplot_profile``,
        which only counts offsets that are on-chromosome in both tracks.
    centers : array-like of int
        Motif start coordinates (duplicates count once per occurrence).
    flank : int
        Bases on each side of the motif start.
    absolute : bool
        Aggregate absolute values (for negative minus-strand bigWigs).
    """
    width = 2 * flank + 1
    length = min(len(before_values), len(after_values))
    centers = np.sort(np.asarray(centers, dtype=np.int64))
    offsets = np.arange(-flank, flank + 1)
    # counts[j]: sites whose c + k lands in [0, length).
    counts = np.searchsorted(centers, length - offsets) - np.searchsorted(
        centers, -offsets
    )
    sums = []
    if centers.size == 0 or length == 0:
        zeros = np.zeros(width, dtype=np.float64)
        return zeros, zeros.copy(), np.maximum(counts, 0).astype(np.int64)
    site_count = _site_count_vector(centers, length, flank)
    for values in (before_values, after_values):
        signal = _dense_signal(values, length, absolute)
        pos = np.flatnonzero(signal)
        vals = signal[pos].astype(np.float64)
        track_sum = np.empty(width, dtype=np.float64)
        for j, k in enumerate(offsets):
            # pos - k + flank is always within [0, length + 2 * flank).
            track_sum[j] = np.dot(vals, site_count[pos - k + flank])
        sums.append(track_sum)
    return sums[0], sums[1], counts.astype(np.int64)


STRAND_ABSOLUTE = {"+": False, "-": True}


def _chrom_profile_task(chrom, bigwigs, chrom_sites, provider, flank):
    """Sum one chromosome for both strands. Runs in a worker.

    ``chrom_sites`` maps strand to motif-start arrays (BED mode); otherwise
    ``provider`` locates this chromosome's sites here, so FASTA-mode motif
    location runs in parallel and no genome-wide site table is built.
    """
    if chrom_sites is None:
        beds = [bed for bed in provider.for_chrom(chrom) if len(bed)]
        bed = pd.concat(beds, ignore_index=True) if beds else None
        chrom_sites = {
            strand: (
                np.empty(0, dtype=np.int64)
                if bed is None
                else bed.loc[bed.strand == strand, "start"].to_numpy(np.int64)
            )
            for strand in STRAND_ABSOLUTE
        }
    out = {}
    for strand, absolute in STRAND_ABSOLUTE.items():
        before_fname, after_fname = bigwigs[strand]
        centers = chrom_sites.get(strand)
        if centers is None or len(centers) == 0:
            continue
        try:
            before = read_chrom_values(before_fname, chrom)
            after = read_chrom_values(after_fname, chrom)
        except KeyError:
            continue  # chromosome missing from this strand's bigWig pair
        out[strand] = chrom_window_sums(before, after, centers, flank, absolute)
    return out


def metaplot_profiles(
    bigwigs,
    chroms,
    flank,
    sites=None,
    provider=None,
    chrom_workers=1,
    worker_backend="process",
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
        Locates sites per chromosome inside each worker.
    chrom_workers : int
        Chromosomes processed concurrently. Each worker holds one
        chromosome's sequence, motif sites, two dense tracks and a site-count
        vector: peak about 11 GB in FASTA mode on hg38 (chr1).
    worker_backend : {"process", "thread"}
        Executor used when ``chrom_workers > 1``.
    """
    if (sites is None) == (provider is None):
        raise ValueError("pass exactly one of sites or provider")
    width = 2 * flank + 1
    totals = {
        strand: [np.zeros(width), np.zeros(width), np.zeros(width, dtype=np.int64)]
        for strand in STRAND_ABSOLUTE
    }

    def tasks():
        for chrom in chroms:
            chrom_sites = None if sites is None else sites.get(chrom)
            if sites is not None and chrom_sites is None:
                continue
            yield chrom, chrom_sites

    def add(result):
        for strand, parts in result.items():
            for total, part in zip(totals[strand], parts):
                total += part

    if chrom_workers <= 1:
        for chrom, chrom_sites in tasks():
            add(_chrom_profile_task(chrom, bigwigs, chrom_sites, provider, flank))
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
                    _chrom_profile_task, chrom, bigwigs, chrom_sites, provider, flank
                )
                for chrom, chrom_sites in tasks()
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
        choices=["auto", "python", "ahocorasick"],
        default="ahocorasick",
        help="Pattern matcher for --fasta mode.",
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
            "Chromosomes processed concurrently (motif location and summing). "
            "FASTA mode on hg38 peaks at about 11 GB per worker (chr1). "
            "auto uses min(chromosomes, CPUs, 4)."
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
    )
    outputs = write_profile_pngs(profiles, args.out_prefix)
    tsv_fname = write_profile_tsv(profiles, f"{args.out_prefix}.tsv")
    print(f"Wrote plus-strand metaplot: {outputs['+']}")
    print(f"Wrote minus-strand metaplot: {outputs['-']}")
    print(f"Wrote metaplot profiles: {tsv_fname}")
