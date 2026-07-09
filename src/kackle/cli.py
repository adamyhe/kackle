"""Command-line entry points and orchestration for kackle.

The CLI can consume precomputed BED6 motif locations or locate motifs directly
from FASTA. When multiple motif beds are supplied, corrections are applied in
order to the same in-memory chromosome signal before writing output.
"""

import argparse
import gc

import numpy as np
import pandas as pd
import tqdm

from kackle.bigwig import open_bigwig, read_chrom_values, write_bigwig
from kackle.computation import correct_kmers
from kackle.motifs import FastaMotifSiteProvider


DEFAULT_MOTIFS = ("TGG:0", "TGGAA:1")


def read_bed6(bed6_fname):
    """Read a BED6 file with kmer candidate sites."""
    bed = pd.read_csv(bed6_fname, sep="\t", header=None)
    bed.columns = ["chrom", "start", "end", "name", "score", "strand"]
    return bed


def bed_starts_for_strand(bed, chrom, strand):
    """Return strand-specific correction anchors for one chromosome.

    For plus-strand BED rows, kackle corrects around the motif start. For
    minus-strand rows, it corrects around the motif end, matching the original
    BED6 preprocessing convention.
    """
    bed = bed[(bed.chrom == chrom) & (bed.strand == strand)].copy()
    if strand == "+":
        bed["end"] = bed["start"] + 1
    else:
        bed["start"] = bed["end"]
        bed["end"] = bed["start"] + 1
    return bed.start.to_numpy().astype(np.int32)


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

    bw = open_bigwig(bw_fname)
    try:
        for chrom in tqdm.tqdm(
            chroms,
            disable=not verbose,
            desc=f"Correcting {'plus' if strand == '+' else 'minus'} strand kmers",
        ):
            values = read_chrom_values(bw, chrom).astype(np.int32)
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
            corrected_df = pd.DataFrame(
                {
                    "chrom": chrom,
                    "start": np.arange(corrected_signal.shape[0]),
                    "end": np.arange(corrected_signal.shape[0]) + 1,
                    "value": corrected_signal.astype(np.float64),
                }
            )
            dfs.append(corrected_df[corrected_df.value > 0].reset_index(drop=True))
    finally:
        bw.close()

    return pd.concat(dfs, ignore_index=True)


def parse_args():
    """Parse command-line arguments for the ``kackle`` console script."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in_pl_bw", type=str, required=True)
    parser.add_argument("-I", "--in_mn_bw", type=str, required=True)
    sites = parser.add_mutually_exclusive_group(required=True)
    sites.add_argument("-b", "--bed6", type=str)
    sites.add_argument("-f", "--fasta", type=str)
    parser.add_argument("-o", "--out_pl_bw", type=str, required=True)
    parser.add_argument("-O", "--out_mn_bw", type=str, required=True)
    parser.add_argument("-c", "--chrom_sizes", type=str, required=True)
    parser.add_argument(
        "-k",
        "--motif",
        action="append",
        default=None,
        help="Motif spec for FASTA location as KMER[:MISMATCHES]. May be repeated.",
    )
    parser.add_argument(
        "-P",
        "--only-positive-strand",
        action="store_true",
        help="Only locate motifs on the positive strand when using --fasta.",
    )
    parser.add_argument(
        "--fasta-backend",
        choices=["auto", "python", "pyfastx"],
        default="python",
        help="FASTA reader for --fasta mode.",
    )
    parser.add_argument(
        "--motif-match-backend",
        choices=["auto", "python", "ahocorasick"],
        default="python",
        help="Pattern matcher for --fasta mode.",
    )
    parser.add_argument("-s", "--source", type=int, default=10)
    parser.add_argument("-t", "--target", type=int, default=5)
    parser.add_argument("-r", "--threshold", type=int, default=10)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    if args.bed6 and args.motif:
        parser.error("--motif can only be used with --fasta")
    if args.bed6 and args.only_positive_strand:
        parser.error("--only-positive-strand can only be used with --fasta")
    if args.bed6 and args.fasta_backend != "python":
        parser.error("--fasta-backend can only be used with --fasta")
    if args.bed6 and args.motif_match_backend != "python":
        parser.error("--motif-match-backend can only be used with --fasta")
    return args


def wrapper():
    """Run the command-line correction workflow."""
    args = parse_args()
    chrom_sizes = pd.read_csv(
        args.chrom_sizes, sep="\t", header=None, names=["chrom", "size"]
    )
    chrom_sizes = [
        (chrom, int(size))
        for chrom, size in zip(chrom_sizes["chrom"], chrom_sizes["size"])
    ]
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


if __name__ == "__main__":
    wrapper()
