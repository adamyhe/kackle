"""Metaplot utilities for before/after k-mer correction diagnostics."""

import argparse
import textwrap

import numpy as np
import pandas as pd
import pybigtools

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
    fasta_backend="pyfastx",
    match_backend="ahocorasick",
):
    """Locate motif BED rows chromosome-by-chromosome from FASTA."""
    provider = FastaMotifSiteProvider(
        fasta_fname,
        motif_specs,
        both_strands=both_strands,
        fasta_backend=fasta_backend,
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


def metaplot_profile(before_bw, after_bw, sites, flank, absolute=False):
    """Return mean before/after signal centered on motif starts.

    Parameters
    ----------
    before_bw, after_bw : object
        Open pybigtools bigWig readers.
    sites : pandas.DataFrame
        Rows with ``chrom`` and ``start`` columns. ``start`` is the centered
        motif coordinate.
    flank : int
        Number of bases shown on each side of the motif start.
    absolute : bool
        Whether to aggregate absolute signal values. This is useful for
        negative-strand bigWigs whose input values are negative.
    """
    width = flank * 2 + 1
    before_sum = np.zeros(width, dtype=np.float64)
    after_sum = np.zeros(width, dtype=np.float64)
    counts = np.zeros(width, dtype=np.int64)

    for chrom, center in sites.itertuples(index=False, name=None):
        center = int(center)
        try:
            chrom_size = min(before_bw.chroms(chrom), after_bw.chroms(chrom))
        except KeyError:
            continue
        left = max(0, center - flank)
        right = min(chrom_size, center + flank + 1)
        if left >= right:
            continue

        dest_start = left - (center - flank)
        dest_end = dest_start + (right - left)
        before_values = before_bw.values(chrom, left, right, fillna=0.0)
        after_values = after_bw.values(chrom, left, right, fillna=0.0)
        if absolute:
            before_values = np.abs(before_values)
            after_values = np.abs(after_values)

        valid = np.isfinite(before_values) & np.isfinite(after_values)
        if not np.any(valid):
            continue

        target = np.arange(dest_start, dest_end)[valid]
        before_sum[target] += before_values[valid]
        after_sum[target] += after_values[valid]
        counts[target] += 1

    before_mean = np.full(width, np.nan, dtype=np.float64)
    after_mean = np.full(width, np.nan, dtype=np.float64)
    np.divide(before_sum, counts, out=before_mean, where=counts > 0)
    np.divide(after_sum, counts, out=after_mean, where=counts > 0)
    return np.arange(-flank, flank + 1), before_mean, after_mean, counts


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
        before_bw = pybigtools.open(before_fname)
        after_bw = pybigtools.open(after_fname)
        try:
            available_chroms = set(before_bw.chroms()).intersection(after_bw.chroms())
            strand_sites = sites[strand].loc[sites[strand].chrom.isin(available_chroms)]
            x, before, after, counts = metaplot_profile(
                before_bw, after_bw, strand_sites, flank, absolute=absolute
            )
        finally:
            before_bw.close()
            after_bw.close()
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
        "--fasta-backend",
        choices=["auto", "python", "pyfastx"],
        default="pyfastx",
        help="FASTA reader for --fasta mode.",
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
            fasta_backend=args.fasta_backend,
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
