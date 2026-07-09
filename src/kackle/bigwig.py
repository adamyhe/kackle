"""BigWig I/O helpers backed by :mod:`pybigtools`.

The rest of kackle works with dense per-chromosome NumPy arrays and sparse
BED-like dataframes. This module is the boundary between those in-memory
representations and on-disk bigWig files.
"""

import pybigtools


def open_bigwig(path, mode="r"):
    """Open a bigWig file with ``pybigtools``.

    Parameters
    ----------
    path : str or path-like
        Path to the bigWig file.
    mode : {"r", "w"}
        Open mode passed through to ``pybigtools.open``.

    Returns
    -------
    object
        A ``pybigtools`` bigWig reader or writer.
    """
    return pybigtools.open(path, mode)


def read_chrom_values(bw, chrom):
    """Return dense per-base values for one chromosome.

    Missing bigWig intervals are filled with ``0.0``, which matches the read
    coverage interpretation used throughout kackle.

    Parameters
    ----------
    bw : object
        Open ``pybigtools`` bigWig reader.
    chrom : str
        Chromosome or contig name.

    Returns
    -------
    np.ndarray
        Dense signal vector with length equal to the chromosome size.
    """
    return bw.values(chrom, 0, bw.chroms(chrom), fillna=0.0)


def write_bigwig(bed_df, bw_fname, chrom_sizes):
    """Write nonzero per-base signal intervals to a bigWig.

    Parameters
    ----------
    bed_df : pandas.DataFrame
        DataFrame with ``chrom``, ``start``, ``end``, and ``value`` columns.
        Coordinates are expected to be 0-based half-open intervals.
    bw_fname : str or path-like
        Output bigWig path.
    chrom_sizes : iterable
        Iterable of ``(chrom, size)`` pairs.
    """
    chrom_sizes = dict(chrom_sizes)
    entries = (
        (chrom, int(start), int(end), float(value))
        for chrom, start, end, value in bed_df[
            ["chrom", "start", "end", "value"]
        ].itertuples(index=False, name=None)
    )
    bw = open_bigwig(bw_fname, "w")
    try:
        bw.write(chrom_sizes, entries)
    except Exception:
        bw.close()
        raise
