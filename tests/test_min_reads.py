"""``--min-reads``: a minimum count at the artifact base, on top of the fold screen."""

import numpy as np
import pytest

import kackle.correction as cli
from kackle.bigwig import read_chrom_values
from kackle.computation import (
    artifact_mask_parallel,
    artifact_mask_serial,
    correct_adapter_sites,
    correct_artifact_bases,
    correct_kmers,
)
from kackle.motifs import FastaMotifSiteProvider

from test_base_resample import spikes_on_zero
from test_chunking import SPECS, random_genome


def lone_reads(strand):
    """Counts 1, 2 and 5 at artifact bases on an otherwise empty background."""
    off = 1 if strand == "+" else 0
    signal = np.zeros(200, dtype=np.int32)
    starts = np.array([40, 100, 160], dtype=np.int32)
    signal[starts - off] = [1, 2, 5]
    return signal, starts, starts - off


@pytest.mark.parametrize("strand", ["+", "-"])
@pytest.mark.parametrize("min_reads,expected", [(1, [1, 1, 1]), (2, [0, 1, 1]), (3, [0, 0, 1])])
def test_screens_require_min_reads(strand, min_reads, expected):
    signal, starts, _ = lone_reads(strand)
    for mask in (artifact_mask_serial, artifact_mask_parallel):
        got = mask(signal, starts, 10, 10, strand, min_reads)
        assert got.astype(int).tolist() == expected


@pytest.mark.parametrize("strand", ["+", "-"])
def test_default_min_reads_matches_old_screen(strand):
    rng = np.random.default_rng(1)
    signal = rng.poisson(0.3, 20_000).astype(np.int32)
    starts = np.unique(rng.integers(20, 19_980, 3_000)).astype(np.int32)
    default = artifact_mask_parallel(signal, starts, 10, 10, strand)
    assert np.array_equal(default, artifact_mask_parallel(signal, starts, 10, 10, strand, 1))
    assert np.array_equal(default, artifact_mask_serial(signal, starts, 10, 10, strand))
    for m in (2, 4):
        assert np.array_equal(
            artifact_mask_serial(signal, starts, 10, 10, strand, m),
            artifact_mask_parallel(signal, starts, 10, 10, strand, m),
        )
    # min_reads only removes flags, and only those below it.
    off = 1 if strand == "+" else 0
    two = artifact_mask_parallel(signal, starts, 10, 10, strand, 2)
    assert np.array_equal(two, default & (signal[starts - off] >= 2))


@pytest.mark.parametrize("strand", ["+", "-"])
def test_correction_kernels_keep_reads_below_min_reads(strand):
    signal, starts, art = lone_reads(strand)
    thr = np.full(3, 10.0)
    outs = {
        "base": correct_artifact_bases(signal, starts, 10, 10, strand, 2),
        "adapter": correct_adapter_sites(signal, starts, thr, 10, strand, 2),
        # Window mode redraws [anchor - 5, anchor + 5) from an empty flank.
        "window": correct_kmers(signal, starts, 10, 5, 10, strand, 2),
    }
    for out in outs.values():
        assert out[art].tolist() == [1, 0, 0]
        assert out.sum() == 1
    assert correct_artifact_bases(signal, starts, 10, 10, strand).sum() == 0


def test_adapter_rescreen_also_requires_min_reads():
    # The 1 fails the plain screen next to the 500 but would pass the rescreen
    # once the 500 is left out; min_reads=2 keeps it in both stages.
    signal = np.zeros(200, dtype=np.int32)
    starts = np.array([60, 63], dtype=np.int32)
    signal[starts - 1] = [500, 1]
    thr = np.full(2, 10.0)
    assert correct_adapter_sites(signal, starts, thr, 10, "+")[62] == 0
    assert correct_adapter_sites(signal, starts, thr, 10, "+", 2)[62] == 1


def test_pipeline_min_reads(tmp_path):
    chrom_sizes = [("chr1", 20_000), ("chr2", 3_000)]
    fasta = random_genome(tmp_path, chrom_sizes)
    inputs = spikes_on_zero(tmp_path, chrom_sizes, fasta)  # 50 at lone sites, 1s far away
    outs = {}
    for label, min_reads in (("default", 1), ("high", 51)):
        outs[label] = {s: tmp_path / f"{label}{s}.bw" for s in "+-"}
        cli.correct_bigwigs(
            inputs, outs[label], chrom_sizes,
            motif_site_provider=FastaMotifSiteProvider(fasta, SPECS),
            chunk_size=1_500, verbose=False, min_reads=min_reads,
        )
    for s in "+-":
        for chrom, _ in chrom_sizes:
            inp = np.abs(read_chrom_values(inputs[s], chrom))
            assert np.array_equal(read_chrom_values(outs["default"][s], chrom), np.where(inp == 50, 0, inp))
            assert np.array_equal(read_chrom_values(outs["high"][s], chrom), inp)


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_min_reads_must_be_a_positive_integer(bad):
    with pytest.raises(ValueError):
        cli.kmer_resample("x.bw", motif_beds=[], chroms=["chr1"], min_reads=bad, verbose=False)
    with pytest.raises(ValueError):
        cli.correct_bigwigs({"+": "a", "-": "b"}, {"+": "c", "-": "d"}, [], motif_beds=[], min_reads=bad)
