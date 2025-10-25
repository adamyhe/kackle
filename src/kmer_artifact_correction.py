# kmer_artifact_correction.py
# Author: Adam He <adamyhe@gmail.com>

"""
A python utility for identifying and correcting kmer artifacts from PCR amplification
and/or RT bias.
"""

import argparse
import gc

import numpy as np
import pandas as pd
import pyBigWig
import tqdm
from numba import njit, prange


def _get_chrom_values(bw, chrom):
    """
    Very slow version of the _fast_values method. DO NOT USE (THIS IS RETAINED
    FOR ILLUSTRATIVE & UNIT TESTING PURPOSES).
    """
    return np.nan_to_num(np.array(bw.values(chrom, 0, bw.chroms(chrom))))


@njit(parallel=True, cache=False)
def fast_values(intervals: np.array, L: np.uint) -> np.array:
    """
    A internal function that serves as a performant replacement for the pyBigWig.values()
    method, which is very slow for extracting values across whole chromosomes. Instead, given
    all the intervals from a chromosome in a bigWig object (which are very fast to retrieve
    with pyBigWig.intervals()), map these to a 1-D values array.

    IMPORTANT: This function maps positions with no values to 0 rather than nan. This is
    generally the correct interpretation for read coverage bigWigs, but may be inappropriate
    for other types of signal data.

    Parameters
    ----------
    intervals : np.array
        An array containing all the intervals and their associated signal on a given
        chromosome. Obtained via np.array(bw.intervals(chrom))
    L : int
        The length of the chromosome.

    Returns
    -------
    values : np.array, shape (L,)
        An array containing per base values for a given chromosome (equivalent output to
        _get_chrom_values).
    """
    values = np.zeros(L, dtype=intervals.dtype)
    for i in prange(intervals.shape[0]):
        for j in range(intervals[i, 0], intervals[i, 1]):
            values[j] = intervals[i, 2]
    return values


@njit(parallel=True)
def resample_dirmult(
    arr: np.array, source: np.uint = 10, target: np.uint = 5
) -> np.array:
    """
    Replaces a target region around the center of an array with values sampled from
    a Dirichlet + multinomial distribution. Essentially, given a target region, randomly
    generate read counts that sum to the total read count in the source region.

    Does not modify the original array but returns a new array.

    Parameters
    ----------
    arr : np.array
        The array to be resampled.
    source : int, optional
        The size of the source region to sample from, by default 10.
    target : int, optional
        The size of the target region to replace, by default 5.

    Returns
    -------
    np.array
        The resampled array.
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


@njit(parallel=True)
def resample_empirical(
    arr: np.array, source: np.uint = 50, target: np.uint = 5
) -> np.array:
    """
    Replaces a target region around the center of an array with values sampled from
    a local empirical distribution (typically from a larger source region).

    Does not modify the original array but returns a new array.

    Parameters
    ----------
    arr : np.array, shape (n loci, length >= 2 * source)
        The array to be resampled.
    source : int, optional
        The size of the source region to sample from, by default 50.
    target : int, optional
        The size of the target region to replace, by default 5.

    Returns
    -------
    np.array
        The resampled array.
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


@njit(parallel=True, cache=False)
def correct_kmers(
    chromosome_signal_vector: np.array,
    starts: np.array,
    source: np.uint = 10,
    target: np.uint = 5,
    threshold: np.uint = 10,
    strand: str = "+",
) -> np.array:
    out_vec = chromosome_signal_vector.copy()
    # Get kmers with the artifactual signal spike
    pass_threshold = np.full(starts.shape[0], False)
    for i in prange(len(starts)):
        start = starts[i]
        if start - source < 0 or start + source > len(out_vec):
            continue
        suspicious_neighborhood = out_vec[start - source : start + source]
        if strand == "-":
            suspicious_neighborhood = suspicious_neighborhood[::-1]
        suspicious_signal = suspicious_neighborhood[source - 1]
        suspicious_background = (suspicious_neighborhood.sum() - suspicious_signal) / (
            source * 2 - 1
        )
        if suspicious_signal > threshold * suspicious_background:
            pass_threshold[i] = True
    starts = starts[pass_threshold]
    # Pull out neighborhood around each kmer artifact
    suspicious_signal_windows = np.zeros(
        (starts.shape[0], source * 2), dtype=chromosome_signal_vector.dtype
    )
    for i in prange(starts.shape[0]):
        start = starts[i]
        suspicious_signal_windows[i] = out_vec[start - source : start + source]
    # Resample
    corrected_signal_windows = resample_dirmult(
        suspicious_signal_windows, source=source, target=target
    )
    # Apply correction to original array
    for i in prange(starts.shape[0]):
        start = starts[i]
        out_vec[start - source : start + source] = corrected_signal_windows[i]
    return out_vec


def kmer_resample(
    bw_fname,
    bed6_fname,
    strand="+",
    source=10,
    target=5,
    threshold=10,
    verbose=True,
):
    """
    Identifies and resamples positions of kmer artifacts arising from PCR
    amplification/RT bias.

    Given a bed6 file of kmer positions, corrects the signal at 1bp upstream of
    the start of each kmer (strand-specific). The correction is done by sampling
    from a uniform distribution of read depth in a neighborhood around the 1bp
    position.

    Additionally, we only correct kmers if the artifactual peak signal is >
    threshold times the local background signal.

    Parameters
    ----------
    pl_bw_fname : str
        The filename of the plus strand bigwig file.
    mn_bw_fname : str
        The filename of the minus strand bigwig file.
    bed6_fname : str
        The filename of the bed6 file containing kmer coordinates.
    source : int, optional
        The size of the source region to sample from, by default 50.
    target : int, optional
        The size of the target region to replace, by default 5.
    threshold : int, optional
        The threshold for the artifactual signal, by default 10.
    verbose : bool
        Whether to print a progress bar (default = True)

    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        A tuple containing two pandas dataframes, one for the plus strand and
        one for the minus strand.
    """
    if strand not in ["+", "-"]:
        raise ValueError("Strand must be '+' or '-'")
    bed = pd.read_csv(bed6_fname, sep="\t", header=None)
    bed.columns = ["chrom", "start", "end", "name", "score", "strand"]
    chroms = ["chr" + str(i) for i in range(1, 23)] + ["chrX", "chrY"]

    # Filter bed file by strand
    bed = bed[bed.strand == strand]
    if strand == "+":
        bed["end"] = bed["start"] + 1
    else:
        bed["start"] = bed["end"]
        bed["end"] = bed["start"] + 1
    dfs = []

    with pyBigWig.open(bw_fname) as bw:
        for chrom in tqdm.tqdm(
            chroms,
            disable=not verbose,
            desc=f"Correcting {'plus' if strand == '+' else 'minus'} strand kmers",
        ):
            # extract intervals to array
            intervals = np.array(bw.intervals(chrom))
            # fast_values is much faster than pyBigWig.values()
            values = fast_values(intervals, bw.chroms(chrom)).astype(np.int32)
            if strand == "-":
                values = np.abs(values)
            # correct kmers
            corrected_signal = correct_kmers(
                values,
                bed[bed.chrom == chrom].start.to_numpy().astype(np.int32),
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
            # convert to dataframe:
            dfs.append(corrected_df[corrected_df.value > 0].reset_index(drop=True))

    return pd.concat(dfs, ignore_index=True)


def write_bigWig(bed_df, bw_fname, chrom_sizes):
    with pyBigWig.open(bw_fname, "w") as bw:
        bw.addHeader(chrom_sizes)
        bw.addEntries(
            chroms=bed_df.chrom.tolist(),
            starts=bed_df.start.tolist(),
            ends=bed_df.end.tolist(),
            values=bed_df.value.tolist(),
        )


def wrapper():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in_pl_bw", type=str, required=True)
    parser.add_argument("-I", "--in_mn_bw", type=str, required=True)
    parser.add_argument("-b", "--bed6", type=str, required=True)
    parser.add_argument("-o", "--out_pl_bw", type=str, required=True)
    parser.add_argument("-O", "--out_mn_bw", type=str, required=True)
    parser.add_argument("-c", "--chrom_sizes", type=str, required=True)
    parser.add_argument("-s", "--source", type=int, default=10)
    parser.add_argument("-t", "--target", type=int, default=5)
    parser.add_argument("-r", "--threshold", type=int, default=10)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    chrom_sizes = pd.read_csv(
        args.chrom_sizes, sep="\t", header=None, names=["chrom", "size"]
    )
    chrom_sizes = [
        (chrom, int(size))
        for chrom, size in zip(chrom_sizes["chrom"], chrom_sizes["size"])
    ]

    pl_bg = kmer_resample(
        args.in_pl_bw,
        args.bed6,
        "+",
        args.source,
        args.target,
        args.threshold,
        args.verbose,
    )
    print("Writing plus strand bigWig ...")
    write_bigWig(pl_bg, args.out_pl_bw, chrom_sizes)
    del pl_bg
    gc.collect()

    mn_bg = kmer_resample(
        args.in_mn_bw,
        args.bed6,
        "-",
        args.source,
        args.target,
        args.threshold,
        args.verbose,
    )
    print("Writing minus strand bigWig ...")
    write_bigWig(mn_bg, args.out_mn_bw, chrom_sizes)


if __name__ == "__main__":
    wrapper()
