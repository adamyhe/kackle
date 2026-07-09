"""Kmer artifact correction utilities.

The public API exposes the correction kernels, motif locating helpers, bigWig
I/O helpers, and the CLI workflow entry point.
"""

from kackle.bigwig import write_bigwig
from kackle.cli import kmer_resample, wrapper
from kackle.computation import (
    artifact_mask_parallel,
    artifact_mask_serial,
    correct_kmers,
    resample_dirmult,
    resample_empirical,
)
from kackle.motifs import (
    FastaMotifSiteProvider,
    locate_motif,
    locate_motif_in_sequence,
    locate_motif_specs,
    mismatch_variants,
)

__all__ = [
    "artifact_mask_parallel",
    "artifact_mask_serial",
    "correct_kmers",
    "FastaMotifSiteProvider",
    "kmer_resample",
    "locate_motif",
    "locate_motif_in_sequence",
    "locate_motif_specs",
    "mismatch_variants",
    "resample_dirmult",
    "resample_empirical",
    "wrapper",
    "write_bigwig",
]
