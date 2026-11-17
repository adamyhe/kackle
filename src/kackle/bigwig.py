"""BigWig I/O helpers backed by :mod:`figwig`.

The rest of kackle works with dense per-chromosome NumPy arrays and sparse
BED-like dataframes. This module is the boundary between those in-memory
representations and on-disk bigWig files.
"""

import figwig
import numpy as np


def open_bigwig(bw_fname):
    """Open a bigWig for reading.

    figwig readers hold no open file handle between reads, so no ``close`` is
    needed and one reader can be shared across threads.
    """
    return figwig.BigWigReader(bw_fname)


def bigwig_chrom_sizes(bw_fname):
    """Return ``{chrom: size}`` for a bigWig, in the file's chromosome order."""
    return dict(open_bigwig(bw_fname).chrom_sizes)


def read_chrom_values(bw_fname, chrom, n_jobs=1, start=0, end=None):
    """Return dense per-base values for a chromosome, or ``[start, end)`` of it.

    Bases not covered by any interval are 0. ``end`` defaults to the
    chromosome's length. Raises ``KeyError`` if the chromosome is absent from
    the bigWig.
    """
    bw = open_bigwig(bw_fname)
    if end is None:
        end = int(bw.chrom_sizes[chrom])
    return bw.read([chrom], [int(start)], int(end) - int(start), n_jobs=n_jobs, missing=0.0)[0]


def open_bigwig_writer(bw_fname, chrom_sizes, n_jobs=8):
    """Open a streaming bigWig writer; use it as a context manager.

    Chromosomes must then be written with :func:`write_chrom_intervals` in
    ``chrom_sizes`` order.
    """
    chrom_sizes = [(chrom, int(size)) for chrom, size in dict(chrom_sizes).items()]
    return figwig.BigWigWriter(str(bw_fname), chrom_sizes, n_jobs=n_jobs)


def write_chrom_intervals(writer, chrom, starts, values, ends=None):
    """Write one chromosome's intervals (per-base when ``ends`` is omitted)."""
    starts = np.asarray(starts, dtype=np.int64)
    if starts.shape[0] == 0:
        return
    ends = starts + 1 if ends is None else np.asarray(ends, dtype=np.int64)
    writer.write(chrom, starts, np.asarray(values, dtype=np.float64), ends=ends)


def write_bigwig(bed_df, bw_fname, chrom_sizes, n_jobs=8):
    """Write nonzero per-base signal intervals to a bigWig.

    Intervals are handed to figwig one chromosome at a time, in
    ``chrom_sizes`` order, so each call carries a single chromosome name
    rather than a per-interval string array.

    Parameters
    ----------
    bed_df : pandas.DataFrame
        DataFrame with ``chrom``, ``start``, ``end``, and ``value`` columns.
        Coordinates are expected to be 0-based half-open intervals.
    bw_fname : str or path-like
        Output bigWig path.
    chrom_sizes : iterable
        Iterable of ``(chrom, size)`` pairs.
    n_jobs : int
        Maximum compression threads used by figwig.
    """
    groups = {
        chrom: rows for chrom, rows in bed_df.groupby("chrom", sort=False, observed=True)
    }
    with open_bigwig_writer(bw_fname, chrom_sizes, n_jobs=n_jobs) as writer:
        for chrom in dict(chrom_sizes):
            rows = groups.get(chrom)
            if rows is None or rows.empty:
                continue
            write_chrom_intervals(
                writer, chrom, rows["start"], rows["value"], ends=rows["end"]
            )
