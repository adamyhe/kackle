"""``--resample base``: redraw only the artifact base of flagged sites."""

import numpy as np
import pandas as pd
import pytest

import kackle.correction as cli
from kackle.bigwig import read_chrom_values, write_bigwig
from kackle.computation import artifact_mask_parallel, correct_artifact_bases
from kackle.motifs import FastaMotifSiteProvider

from test_chunking import SPECS, random_genome


@pytest.mark.parametrize("strand", ["+", "-"])
def test_only_flagged_artifact_bases_change(strand):
    rng = np.random.default_rng(0)
    signal = rng.poisson(2, 50_000).astype(np.int32)
    starts = np.unique(rng.integers(20, 49_980, 2_000)).astype(np.int32)
    art = starts - 1 if strand == "+" else starts
    spiked = rng.random(starts.size) < 0.5
    signal[art[spiked]] += 500
    flagged = artifact_mask_parallel(signal, starts, 10, 10, strand)
    assert flagged.sum() > 500
    out = correct_artifact_bases(signal, starts, 10, 10, strand)
    changed = np.flatnonzero(out != signal)
    assert set(changed) <= set(art[flagged])
    assert out[art[flagged]].max() < 50  # spikes replaced by local-level draws
    keep = np.ones(signal.size, bool)
    keep[art[flagged]] = False
    assert np.array_equal(out[keep], signal[keep])


def test_background_leaves_out_other_flagged_artifact_bases():
    # Two adjacent flagged sites on a flat background of 1: each one's mean
    # must ignore the other's spike, so draws average 1, not ~27.
    signal = np.ones(100, dtype=np.int32)
    starts = np.array([50, 52], dtype=np.int32)
    signal[[49, 51]] = 500
    draws = np.array(
        [correct_artifact_bases(signal, starts, 10, 10, "+")[[49, 51]] for _ in range(2000)]
    )
    assert abs(draws.mean() - 1.0) < 0.1


def test_unflagged_sites_and_empty_inputs_are_untouched():
    signal = np.full(100, 3, dtype=np.int32)
    assert np.array_equal(
        correct_artifact_bases(signal, np.array([50], dtype=np.int32), 10, 10, "+"), signal
    )
    assert np.array_equal(
        correct_artifact_bases(signal, np.array([], dtype=np.int32), 10, 10, "+"), signal
    )


def spikes_on_zero(tmp_path, chrom_sizes, fasta):
    """Spikes at every artifact base on an empty background: base mode must zero them."""
    from kackle.motifs import locate_motif_specs

    beds = locate_motif_specs(fasta, SPECS)
    paths = {}
    for strand, sign in (("+", 1.0), ("-", -1.0)):
        rows = []
        for chrom, size in chrom_sizes:
            vals = np.zeros(size)
            anchors = np.sort(np.concatenate([
                (bed.start if strand == "+" else bed.end)[
                    (bed.chrom == chrom) & (bed.strand == strand)
                ].to_numpy()
                for bed in beds
            ]))
            gaps = np.diff(anchors)
            # Spike only sites with no other anchor within 2 * source, so every
            # spike's neighborhood is empty and its draw is exactly 0.
            lone = anchors[1:-1][(gaps[:-1] > 20) & (gaps[1:] > 20)]
            spike = lone - 1 if strand == "+" else lone
            vals[spike] = 50
            far = np.ones(size, bool)
            for a in anchors:
                far[max(a - 25, 0) : a + 25] = False
            vals[np.flatnonzero(far)[::7]] = 1  # untouched signal away from sites
            pos = np.flatnonzero(vals)
            rows.append(pd.DataFrame({"chrom": chrom, "start": pos, "end": pos + 1, "value": sign * vals[pos]}))
        paths[strand] = tmp_path / f"in{strand}.bw"
        write_bigwig(pd.concat(rows, ignore_index=True), paths[strand], chrom_sizes)
    return paths


@pytest.mark.parametrize("backend,workers", [("thread", 1), ("process", 2)])
def test_base_mode_chunked_matches_whole(tmp_path, backend, workers):
    chrom_sizes = [("chr1", 20_000), ("chr2", 3_000)]
    fasta = random_genome(tmp_path, chrom_sizes)
    inputs = spikes_on_zero(tmp_path, chrom_sizes, fasta)
    outs = {}
    for label, chunk_size in (("whole", 0), ("chunked", 1_500)):
        outs[label] = {s: tmp_path / f"{label}{s}.bw" for s in "+-"}
        cli.correct_bigwigs(
            inputs, outs[label], chrom_sizes,
            motif_site_provider=FastaMotifSiteProvider(fasta, SPECS),
            chrom_workers=workers, worker_backend=backend, chunk_size=chunk_size,
            verbose=False, resample="base",
        )
    for s in "+-":
        assert outs["chunked"][s].read_bytes() == outs["whole"][s].read_bytes()
        for chrom, _ in chrom_sizes:
            got = read_chrom_values(outs["whole"][s], chrom)
            inp = np.abs(read_chrom_values(inputs[s], chrom))
            assert inp.max() == 50 and got.max() == 1
            assert np.array_equal(got, np.where(inp == 50, 0, inp))


def test_base_is_the_default_resampling():
    import inspect

    for fn in (cli.kmer_resample, cli.correct_bigwigs):
        assert inspect.signature(fn).parameters["resample"].default == "base"
    for bad in ("Window", "", None):
        with pytest.raises(ValueError):
            cli.kmer_resample("x.bw", motif_beds=[], chroms=["chr1"], resample=bad, verbose=False)
