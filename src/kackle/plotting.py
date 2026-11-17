"""Metaplot utilities for before/after k-mer correction diagnostics."""

import argparse
import textwrap

import numpy as np
import pandas as pd

from kackle.bigwig import open_bigwig
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
):
    """Write plus and minus before/after metaplots and return output paths."""
    sites = motif_sites_by_strand(bed)
    outputs = {}
    plot_specs = {
        "+": (before_pl_bw_fname, after_pl_bw_fname, False, f"{out_prefix}.plus.png"),
        "-": (before_mn_bw_fname, after_mn_bw_fname, True, f"{out_prefix}.minus.png"),
    }

    for strand, (before_fname, after_fname, absolute, out_fname) in plot_specs.items():
        x, before, after, counts = metaplot_profile(
            open_bigwig(before_fname),
            open_bigwig(after_fname),
            sites[strand],
            flank,
            absolute=absolute,
        )
        write_metaplot_png(x, before, after, counts, strand, out_fname)
        outputs[strand] = out_fname
    return outputs


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
        "-o",
        "--out-prefix",
        metavar="PREFIX",
        required=True,
        help="Output prefix. Writes PREFIX.plus.png and PREFIX.minus.png.",
    )
    args = parser.parse_args()
    if args.fasta and args.chrom_sizes is None:
        parser.error("--chrom-sizes is required with --fasta")
    return args


def run_metaplot():
    """Run the before/after metaplot workflow."""
    args = parse_metaplot_args()
    if args.bed6:
        bed = read_bed6(args.bed6)
    else:
        chroms = [chrom for chrom, _ in read_chrom_sizes(args.chrom_sizes)]
        chroms = intersect_ordered(
            chroms,
            bigwig_chrom_names(args.before_pl_bw),
            bigwig_chrom_names(args.before_mn_bw),
            bigwig_chrom_names(args.after_pl_bw),
            bigwig_chrom_names(args.after_mn_bw),
            fasta_chrom_names(args.fasta),
        )
        bed = motif_beds_from_fasta(
            args.fasta,
            args.motif or DEFAULT_MOTIFS,
            chroms,
            match_backend=args.motif_match_backend,
        )
    outputs = write_strand_metaplots(
        args.before_pl_bw,
        args.before_mn_bw,
        args.after_pl_bw,
        args.after_mn_bw,
        bed,
        args.out_prefix,
        flank=args.flank,
    )
    print(f"Wrote plus-strand metaplot: {outputs['+']}")
    print(f"Wrote minus-strand metaplot: {outputs['-']}")
