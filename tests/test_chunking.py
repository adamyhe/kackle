"""Splitting chromosomes into independently corrected chunks."""

import numpy as np
import pandas as pd
import pytest

import kackle.correction as cli
from kackle.bigwig import read_chrom_values, write_bigwig
from kackle.computation import _flagged_clusters, artifact_mask_parallel
from kackle.motifs import FastaMotifSiteProvider, locate_motif_specs


@pytest.fixture(autouse=True)
def window_resampling(monkeypatch):
    """Pin the library calls to window resampling.

    These tests swap ``cli.correct_kmers`` (the window kernel) for
    deterministic stand-ins to check pipeline mechanics; base-only
    resampling, the default, has its own tests in test_base_resample.py.
    """
    for name in ("kmer_resample", "correct_bigwigs"):
        original = getattr(cli, name)

        def pinned(*args, _original=original, **kwargs):
            kwargs.setdefault("resample", "window")
            return _original(*args, **kwargs)

        monkeypatch.setattr(cli, name, pinned)

SPECS = ["TGG:0", "TGGAA:1"]


def deterministic_correct_kmers(values, starts, source, target, threshold, strand, min_reads=1):
    """``correct_kmers`` with the real screen and clustering, minus the randomness.

    Each cluster's span gets its width-scaled flank total spread evenly
    (remainder to the first bases), so chunked and whole-chromosome runs can
    be compared byte for byte.
    """
    out = values.copy()
    if starts.shape[0] == 0:
        return out
    flagged = artifact_mask_parallel(values, starts, source, threshold, strand, min_reads)
    flagged_starts = np.sort(starts[flagged].astype(np.int64))
    if flagged_starts.shape[0] == 0:
        return out
    bounds = _flagged_clusters(flagged_starts, source + target)
    for c in range(bounds.shape[0] - 1):
        first = flagged_starts[bounds[c]]
        last = flagged_starts[bounds[c + 1] - 1]
        lo, hi = first - target, last + target
        flank = int(values[first - source : lo].sum() + values[hi : last + source].sum())
        width = hi - lo
        total = (flank * width + target) // (2 * target)
        fill = np.full(width, total // width, dtype=values.dtype)
        fill[: total % width] += 1
        out[lo:hi] = fill
    return out


def random_genome(tmp_path, chrom_sizes, seed=0):
    """A TGG-rich FASTA (many sites, short gaps) with some lowercase and N runs."""
    rng = np.random.default_rng(seed)
    records = []
    for chrom, size in chrom_sizes:
        seq = rng.choice(list("ACGTTGG"), size)
        seq[rng.random(size) < 0.1] = "N"
        seq = "".join(seq)
        seq = seq[: size // 3] + seq[size // 3 : size // 2].lower() + seq[size // 2 :]
        records.append(f">{chrom}\n{seq}\n")
    fasta = tmp_path / "genome.fa"
    fasta.write_text("".join(records))
    return fasta


def spiky_bigwigs(tmp_path, chrom_sizes, fasta, seed=0):
    """Signal with spikes at artifact positions of real motif sites."""
    rng = np.random.default_rng(seed)
    beds = locate_motif_specs(fasta, SPECS)
    paths = {}
    for strand, sign in (("+", 1.0), ("-", -1.0)):
        rows = []
        for chrom, size in chrom_sizes:
            vals = rng.integers(0, 4, size).astype(float)
            for bed in beds:
                sites = bed[(bed.chrom == chrom) & (bed.strand == strand)]
                spike = (sites.start - 1) if strand == "+" else sites.end
                spike = spike[(spike >= 0) & (spike < size)].to_numpy()
                keep = rng.random(spike.size) < 0.5
                vals[spike[keep]] += 200
            pos = np.flatnonzero(vals)
            rows.append(
                pd.DataFrame(
                    {"chrom": chrom, "start": pos, "end": pos + 1, "value": sign * vals[pos]}
                )
            )
        paths[strand] = tmp_path / f"in{strand}.bw"
        write_bigwig(pd.concat(rows, ignore_index=True), paths[strand], chrom_sizes)
    return paths


@pytest.mark.parametrize("both_strands", [True, False])
def test_starts_for_region_matches_whole_chromosome(tmp_path, both_strands):
    chrom_sizes = [("chr1", 3000)]
    fasta = random_genome(tmp_path, chrom_sizes)
    provider = FastaMotifSiteProvider(fasta, SPECS, both_strands=both_strands)
    whole = provider.starts_for_chrom("chr1")
    rng = np.random.default_rng(3)
    bounds = [(0, 3000), (-50, 10), (2990, 4000), (5, 5), (1000, 900)]
    bounds += [tuple(sorted(rng.integers(-10, 3010, 2))) for _ in range(30)]
    for lo, hi in bounds:
        got = provider.starts_for_region("chr1", lo, hi)
        for spec_whole, spec_got in zip(whole, got):
            for strand in ("+", "-"):
                want = spec_whole[strand]
                want = want[(want >= lo) & (want < hi)]
                assert spec_got[strand].tolist() == want.tolist(), (lo, hi, strand)


def test_safe_cut_is_nearest_cut_clear_of_anchors():
    rng = np.random.default_rng(0)
    for _ in range(300):
        reach = int(rng.integers(1, 12))
        lo, hi = 0, int(rng.integers(5, 200))
        anchors = np.unique(rng.integers(lo, hi, rng.integers(0, 40)))
        near = int(rng.integers(lo, hi))
        valid = [
            x
            for x in range(lo + reach, hi - reach + 1)
            if not ((anchors > x - reach) & (anchors < x + reach)).any()
        ]
        got = cli._safe_cut(anchors, near, reach, lo, hi)
        if not valid:
            assert got is None
        else:
            assert got in valid
            assert abs(got - near) == min(abs(x - near) for x in valid)


def test_plan_chunks_tiles_and_never_cuts_near_an_anchor():
    rng = np.random.default_rng(1)
    length = 200_000
    anchors = np.unique(rng.integers(0, length, 12_000))
    near = cli._sorted_anchors_near({"+": [anchors]})
    for chunk_size, reach in [(10_000, 10), (7_777, 4), (50_000, 25)]:
        chunks = cli.plan_chunks(length, chunk_size, near, reach, window=256)
        assert chunks[0][0] == 0 and chunks[-1][1] == length
        assert all(a[1] == b[0] for a, b in zip(chunks, chunks[1:]))
        assert len(chunks) > 1
        for _, cut in chunks[:-1]:
            assert not ((anchors > cut - reach) & (anchors < cut + reach)).any()
    assert cli.plan_chunks(length, 0, near, 10) == [(0, length)]
    assert cli.plan_chunks(length, length, near, 10) == [(0, length)]


def test_plan_chunks_drops_boundaries_without_a_safe_cut():
    anchors = np.arange(0, 100_000, 5)  # every gap is narrower than 2 * reach
    near = cli._sorted_anchors_near({"+": [anchors]})
    assert cli.plan_chunks(100_000, 10_000, near, 10, window=128) == [(0, 100_000)]


@pytest.mark.parametrize("chrom_workers", [1, 3])
@pytest.mark.parametrize("mode", ["fasta", "bed"])
@pytest.mark.parametrize("source,target", [(10, 5), (6, 4), (3, 5)])
def test_chunked_correction_matches_whole_chromosomes(
    tmp_path, monkeypatch, chrom_workers, mode, source, target
):
    chrom_sizes = [("chr1", 20_000), ("chr2", 3_000), ("chr3", 12_000)]
    fasta = random_genome(tmp_path, chrom_sizes)
    inputs = spiky_bigwigs(tmp_path, chrom_sizes, fasta)
    monkeypatch.setattr(cli, "correct_kmers", deterministic_correct_kmers)

    def run(label, chunk_size):
        out = {strand: tmp_path / f"{label}{strand}.bw" for strand in ("+", "-")}
        prefix = tmp_path / label
        if mode == "bed":
            sites = {"motif_beds": locate_motif_specs(fasta, SPECS)}
        else:
            sites = {
                "motif_site_provider": FastaMotifSiteProvider(
                    fasta, SPECS, bed6_fnames=cli.motif_bed6_fnames(prefix, SPECS)
                )
            }
        cli.correct_bigwigs(
            inputs, out, chrom_sizes, source=source, target=target, threshold=5,
            chrom_workers=chrom_workers, worker_backend="thread",
            chunk_size=chunk_size, verbose=False, **sites,
        )
        return out, prefix

    whole, whole_prefix = run("whole", 0)
    chunked, chunked_prefix = run("chunked", 1_500)

    changed = 0
    for strand in ("+", "-"):
        assert chunked[strand].read_bytes() == whole[strand].read_bytes()
        for chrom, _ in chrom_sizes:
            changed += int(
                (read_chrom_values(whole[strand], chrom)
                 != np.abs(read_chrom_values(inputs[strand], chrom))).sum()
            )
    assert changed > 100  # the comparison is not vacuous
    if mode == "fasta":
        for a, b in zip(
            cli.motif_bed6_fnames(whole_prefix, SPECS),
            cli.motif_bed6_fnames(chunked_prefix, SPECS),
        ):
            assert open(a).read() == open(b).read()
            assert open(a).read()


def test_chunk_plans_actually_split(tmp_path):
    chrom_sizes = [("chr1", 20_000)]
    fasta = random_genome(tmp_path, chrom_sizes)
    provider = FastaMotifSiteProvider(fasta, SPECS)
    near = cli._fasta_anchors_near(provider, "chr1")
    chunks = cli.plan_chunks(20_000, 1_500, near, 10)
    assert len(chunks) >= 10


def test_chunked_process_workers_match_whole_chromosomes(tmp_path):
    # Process workers can't see a monkeypatched kernel, so this checks the
    # plumbing (region reads, offsets, ordering) with a threshold nothing
    # reaches: output must equal the input.
    chrom_sizes = [("chr1", 20_000), ("chr2", 3_000)]
    fasta = random_genome(tmp_path, chrom_sizes)
    inputs = spiky_bigwigs(tmp_path, chrom_sizes, fasta)
    out = {strand: tmp_path / f"out{strand}.bw" for strand in ("+", "-")}
    cli.correct_bigwigs(
        inputs, out, chrom_sizes,
        motif_site_provider=FastaMotifSiteProvider(fasta, SPECS),
        threshold=10**9, chrom_workers=2, worker_backend="process",
        chunk_size=2_000, verbose=False,
    )
    for strand in ("+", "-"):
        for chrom, _ in chrom_sizes:
            assert np.array_equal(
                read_chrom_values(out[strand], chrom),
                np.abs(read_chrom_values(inputs[strand], chrom)),
            )
