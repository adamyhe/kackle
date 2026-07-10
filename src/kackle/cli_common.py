"""Shared command-line helpers for kackle entry points."""

import argparse

import numpy as np
import pandas as pd
import pybigtools


DEFAULT_MOTIFS = ("TGG:0", "TGGAA:1")


class KackleArgumentFormatter(
    argparse.RawDescriptionHelpFormatter,
):
    """Format CLI help with preserved examples and useful defaults only."""

    def _get_help_string(self, action):
        """Append defaults for meaningful optional values."""
        help_text = action.help
        if help_text is argparse.SUPPRESS:
            return help_text
        if "%(default)" in help_text:
            return help_text
        if action.default not in (argparse.SUPPRESS, None, False):
            return f"{help_text} (default: %(default)s)"
        return help_text


def read_bed6(bed6_fname):
    """Read a BED6 file with kmer candidate sites."""
    return pd.read_csv(
        bed6_fname,
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "name", "score", "strand"],
        dtype={
            "chrom": "string",
            "start": np.int32,
            "end": np.int32,
            "name": "string",
            "score": np.int32,
            "strand": "string",
        },
    )


def read_chrom_sizes(chrom_sizes_fname):
    """Read a two-column chromosome sizes file."""
    chrom_sizes = pd.read_csv(
        chrom_sizes_fname, sep="\t", header=None, names=["chrom", "size"]
    )
    return [(chrom, int(size)) for chrom, size in chrom_sizes.itertuples(index=False)]


def bigwig_chrom_names(bw_fname):
    """Return chromosome names present in a bigWig."""
    bw = pybigtools.open(bw_fname)
    try:
        return list(bw.chroms().keys())
    finally:
        bw.close()


def intersect_ordered(items, *chrom_name_sets, key=None):
    """Filter ordered items to chromosomes present in every supplied source."""
    if key is None:
        key = lambda item: item
    common = {key(item) for item in items}
    for chrom_names in chrom_name_sets:
        common &= set(chrom_names)
    return [item for item in items if key(item) in common]
