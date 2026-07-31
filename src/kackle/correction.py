"""Correction workflow and command-line entry point for kackle."""

import argparse
import gc
import os
import re
import textwrap
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pybigtools
import tqdm

from kackle.bigwig import write_bigwig
from kackle.cli_common import (
    DEFAULT_MOTIFS,
    KackleArgumentFormatter,
    bigwig_chrom_names,
    intersect_ordered,
    read_bed6,
    read_chrom_sizes,
)
from kackle.computation import correct_kmers
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


def _chrom_motif_beds(motif_site_provider, chrom, isolate_fasta=False):
    """Return motif beds for one chromosome, optionally isolating FASTA state."""
    if isolate_fasta and isinstance(motif_site_provider, FastaMotifSiteProvider):
        provider = FastaMotifSiteProvider(
            motif_site_provider.fasta_fname,
            motif_site_provider.motif_specs,
            both_strands=motif_site_provider.both_strands,
            fasta_backend=motif_site_provider.fasta_backend,
            match_backend=motif_site_provider.match_backend,
        )
        return provider.for_chrom(chrom)
    return motif_site_provider.for_chrom(chrom)


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


def _append_motif_beds_for_chrom(motif_site_provider, chrom, beds):
    """Append generated FASTA motif beds once, preserving chromosome order."""
    if not isinstance(motif_site_provider, FastaMotifSiteProvider):
        return
    if motif_site_provider.bed6_fnames is None:
        return
    if chrom in motif_site_provider._written_chroms:
        return
    for bed, bed6_fname in zip(beds, motif_site_provider.bed6_fnames):
        bed.to_csv(bed6_fname, sep="\t", header=False, index=False, mode="a")
    motif_site_provider._written_chroms.add(chrom)


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
):
    """Correct one chromosome and return sparse output plus motif beds used."""
    if numba_threads is not None:
        from numba import set_num_threads

        set_num_threads(numba_threads)

    bw = pybigtools.open(bw_fname)
    try:
        values = bw.values(chrom, 0, bw.chroms(chrom), fillna=0.0).astype(np.int32)
    finally:
        bw.close()

    if strand == "-":
        values = np.abs(values)
    corrected_signal = values
    if motif_site_provider is not None:
        chrom_motif_beds = _chrom_motif_beds(
            motif_site_provider,
            chrom,
            isolate_fasta=isolate_fasta_provider,
        )
        motif_start_arrays = motif_anchor_arrays(chrom_motif_beds, strand)
    else:
        chrom_motif_beds = motif_beds or []
        motif_start_arrays = motif_start_arrays or []

    for starts in motif_start_arrays:
        if starts.shape[0] == 0:
            continue
        corrected_signal = correct_kmers(
            corrected_signal,
            starts,
            source,
            target,
            threshold,
            strand,
        )

    nonzero_starts = np.flatnonzero(corrected_signal > 0).astype(np.int32)
    corrected_df = pd.DataFrame(
        {
            "chrom": np.full(nonzero_starts.shape[0], chrom, dtype=object),
            "start": nonzero_starts,
            "end": nonzero_starts + 1,
            "value": corrected_signal[nonzero_starts].astype(np.float64),
        }
    )
    return chrom, corrected_df, chrom_motif_beds


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
    if motif_beds is None and motif_site_provider is None:
        if bed6_fname is None:
            raise ValueError(
                "Either bed6_fname, motif_beds, or motif_site_provider must be provided"
            )
        motif_beds = [read_bed6(bed6_fname)]

    if chroms is None:
        chroms = ["chr" + str(i) for i in range(1, 23)] + ["chrX", "chrY"]
    chrom_workers = resolve_chrom_workers(chrom_workers, len(chroms))

    if chrom_workers == 1:
        dfs = []
        for chrom in tqdm.tqdm(
            chroms,
            disable=not verbose,
            desc=f"Correcting {'plus' if strand == '+' else 'minus'} strand kmers",
        ):
            chrom_motif_beds = None
            chrom_start_arrays = None
            if motif_site_provider is None:
                chrom_start_arrays = motif_anchor_arrays(motif_beds, strand, chrom)
            _, corrected_df, chrom_motif_beds = _correct_chromosome(
                bw_fname,
                chrom,
                None,
                chrom_start_arrays,
                motif_site_provider,
                strand,
                source,
                target,
                threshold,
                False,
                numba_threads,
            )
            _append_motif_beds_for_chrom(motif_site_provider, chrom, chrom_motif_beds)
            dfs.append(corrected_df)
        return pd.concat(dfs, ignore_index=True)

    results = {}
    executor_class = (
        ProcessPoolExecutor if worker_backend == "process" else ThreadPoolExecutor
    )
    task_args = []
    for chrom in chroms:
        chrom_start_arrays = None
        if motif_site_provider is None:
            chrom_start_arrays = motif_anchor_arrays(motif_beds, strand, chrom)
        task_args.append((chrom, chrom_start_arrays))

    with executor_class(max_workers=chrom_workers) as executor:
        futures = {
            executor.submit(
                _correct_chromosome,
                bw_fname,
                chrom,
                None,
                chrom_start_arrays,
                motif_site_provider,
                strand,
                source,
                target,
                threshold,
                True,
                numba_threads,
            ): chrom
            for chrom, chrom_start_arrays in task_args
        }
        for future in tqdm.tqdm(
            as_completed(futures),
            total=len(futures),
            disable=not verbose,
            desc=(
                f"Correcting {'plus' if strand == '+' else 'minus'} strand "
                f"kmers ({chrom_workers} chrom workers)"
            ),
        ):
            chrom, corrected_df, chrom_motif_beds = future.result()
            results[chrom] = (corrected_df, chrom_motif_beds)

    dfs = []
    for chrom in chroms:
        corrected_df, chrom_motif_beds = results[chrom]
        _append_motif_beds_for_chrom(motif_site_provider, chrom, chrom_motif_beds)
        dfs.append(corrected_df)
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
        "--out-bed6-prefix",
        metavar="PREFIX",
        help=(
            "When using --fasta, write generated motif sites to ordered BED6 "
            "files named PREFIX.N.MOTIF.mMISMATCHES.bed6."
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
    if args.fasta:
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
            fasta_backend=args.fasta_backend,
            match_backend=args.motif_match_backend,
            bed6_fnames=bed6_fnames,
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
        chrom_workers=chrom_workers,
        worker_backend=args.worker_backend,
        numba_threads=numba_threads,
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
        chrom_workers=chrom_workers,
        worker_backend=args.worker_backend,
        numba_threads=numba_threads,
        verbose=args.verbose,
    )
    print("Writing minus strand bigWig ...")
    write_bigwig(mn_bg, args.out_mn_bw, chrom_sizes)
