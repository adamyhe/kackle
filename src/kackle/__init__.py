"""Kmer artifact correction utilities.

The public API exposes the correction kernels, motif locating helpers, bigWig
I/O helpers, and the CLI workflow entry point.
"""

from kackle.bigwig import write_bigwig
from kackle.correction import kmer_resample, run_correction
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
from kackle.plotting import metaplot_profile, write_strand_metaplots

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
    "metaplot_profile",
    "resample_dirmult",
    "resample_empirical",
    "run_correction",
    "write_bigwig",
    "write_strand_metaplots",
]
