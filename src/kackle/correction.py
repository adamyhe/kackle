"""Correction workflow and command-line entry point for kackle."""

import argparse
import gc
import textwrap

import numpy as np
import pandas as pd
import pybigtools
import tqdm

from kackle.bigwig import write_bigwig
from kackle.cli_common import (
    DEFAULT_MOTIFS,
    KackleArgumentFormatter,
    read_bed6,
    read_chrom_sizes,
)
from kackle.computation import correct_kmers
from kackle.motifs import FastaMotifSiteProvider


def bed_starts_for_strand(bed, chrom, strand):
    """Return strand-specific correction anchors for one chromosome.

    For plus-strand BED rows, kackle corrects around the motif start. For
    minus-strand rows, it corrects around the motif end, matching the original
    BED6 preprocessing convention.
    """
    if strand == "+":
        starts = bed.loc[(bed.chrom == chrom) & (bed.strand == strand), "start"]
    else:
        starts = bed.loc[(bed.chrom == chrom) & (bed.strand == strand), "end"]
    return starts.to_numpy(dtype=np.int32, copy=False)


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
    verbose=True,
):
    """Correct one strand-specific bigWig and return nonzero output intervals.

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
        Provider with ``for_chrom(chrom)`` that returns ordered BED6-style
        motif tables for a chromosome. FASTA mode uses this to avoid
        materializing whole-genome motif locations.
    strand : {"+", "-"}
        Signal/motif strand to correct.
    chroms : sequence[str], optional
        Chromosomes to process. Defaults to human autosomes plus chrX/chrY.
    source, target, threshold : int
        Parameters forwarded to ``correct_kmers``.
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

    if motif_beds is None and motif_site_provider is None:
        if bed6_fname is None:
            raise ValueError(
                "Either bed6_fname, motif_beds, or motif_site_provider must be provided"
            )
        motif_beds = [read_bed6(bed6_fname)]

    if chroms is None:
        chroms = ["chr" + str(i) for i in range(1, 23)] + ["chrX", "chrY"]

    dfs = []

    bw = pybigtools.open(bw_fname)
    try:
        for chrom in tqdm.tqdm(
            chroms,
            disable=not verbose,
            desc=f"Correcting {'plus' if strand == '+' else 'minus'} strand kmers",
        ):
            values = bw.values(chrom, 0, bw.chroms(chrom), fillna=0.0).astype(np.int32)
            if strand == "-":
                values = np.abs(values)
            corrected_signal = values
            chrom_motif_beds = (
                motif_site_provider.for_chrom(chrom)
                if motif_site_provider is not None
                else motif_beds
            )
            for bed in chrom_motif_beds:
                corrected_signal = correct_kmers(
                    corrected_signal,
                    bed_starts_for_strand(bed, chrom, strand),
                    source,
                    target,
                    threshold,
                    strand,
                )
            nonzero_starts = np.flatnonzero(corrected_signal > 0).astype(np.int32)
            dfs.append(
                pd.DataFrame(
                    {
                        "chrom": np.full(nonzero_starts.shape[0], chrom, dtype=object),
                        "start": nonzero_starts,
                        "end": nonzero_starts + 1,
                        "value": corrected_signal[nonzero_starts].astype(np.float64),
                    }
                )
            )
    finally:
        bw.close()

    return pd.concat(dfs, ignore_index=True)


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

            FASTA mode defaults to motif passes {", ".join(DEFAULT_MOTIFS)} and locates sites one chromosome at a time using pyfastx plus ahocorasick-rs.
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
        "-P",
        "--only-positive-strand",
        action="store_true",
        help=(
            "In FASTA mode, locate only forward-strand motif matches instead "
            "of searching both motif and reverse-complement matches."
        ),
    )
    parser.add_argument(
        "--fasta-backend",
        choices=["auto", "python", "pyfastx"],
        default="pyfastx",
        help=(
            "FASTA reader for --fasta mode. pyfastx is the default indexed "
            "reader; python is a simple fallback useful for debugging."
        ),
    )
    parser.add_argument(
        "--motif-match-backend",
        choices=["auto", "python", "ahocorasick"],
        default="ahocorasick",
        help=(
            "Pattern matcher for --fasta mode. ahocorasick scans all expanded "
            "mismatch variants together; python repeats str.find per variant."
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
        "-v",
        "--verbose",
        action="store_true",
        help="Show per-chromosome progress bars while correcting.",
    )
    args = parser.parse_args()
    if args.bed6 and args.motif:
        parser.error("--motif can only be used with --fasta")
    if args.bed6 and args.only_positive_strand:
        parser.error("--only-positive-strand can only be used with --fasta")
    return args


def run_correction():
    """Run the command-line correction workflow."""
    args = parse_args()
    chrom_sizes = read_chrom_sizes(args.chrom_sizes)
    chroms = [chrom for chrom, _ in chrom_sizes]
    motif_site_provider = None
    if args.fasta:
        motif_site_provider = FastaMotifSiteProvider(
            args.fasta,
            args.motif or DEFAULT_MOTIFS,
            both_strands=not args.only_positive_strand,
            fasta_backend=args.fasta_backend,
            match_backend=args.motif_match_backend,
        )

    pl_bg = kmer_resample(
        bw_fname=args.in_pl_bw,
        bed6_fname=args.bed6,
        motif_site_provider=motif_site_provider,
        strand="+",
        chroms=chroms,
        source=args.source,
        target=args.target,
        threshold=args.threshold,
        verbose=args.verbose,
    )
    print("Writing plus strand bigWig ...")
    write_bigwig(pl_bg, args.out_pl_bw, chrom_sizes)
    del pl_bg
    gc.collect()

    mn_bg = kmer_resample(
        bw_fname=args.in_mn_bw,
        bed6_fname=args.bed6,
        motif_site_provider=motif_site_provider,
        strand="-",
        chroms=chroms,
        source=args.source,
        target=args.target,
        threshold=args.threshold,
        verbose=args.verbose,
    )
    print("Writing minus strand bigWig ...")
    write_bigwig(mn_bg, args.out_mn_bw, chrom_sizes)
