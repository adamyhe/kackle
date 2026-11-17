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


def read_chrom_values(bw_fname, chrom, n_jobs=1):
    """Return dense per-base values for an entire chromosome.

    Bases not covered by any interval are 0. Raises ``KeyError`` if the
    chromosome is absent from the bigWig.
    """
    bw = open_bigwig(bw_fname)
    chrom_size = int(bw.chrom_sizes[chrom])
    return bw.read([chrom], [0], chrom_size, n_jobs=n_jobs, missing=0.0)[0]


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
    chrom_sizes = [(chrom, int(size)) for chrom, size in dict(chrom_sizes).items()]
    groups = {
        chrom: rows for chrom, rows in bed_df.groupby("chrom", sort=False, observed=True)
    }
    with figwig.BigWigWriter(str(bw_fname), chrom_sizes, n_jobs=n_jobs) as writer:
        for chrom, _ in chrom_sizes:
            rows = groups.get(chrom)
            if rows is None or rows.empty:
                continue
            writer.write(
                chrom,
                rows["start"].to_numpy(dtype=np.int64),
                rows["value"].to_numpy(dtype=np.float64),
                ends=rows["end"].to_numpy(dtype=np.int64),
            )
