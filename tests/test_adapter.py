"""Adapter look-alike sites and the single-pass screen."""

import numpy as np
import pandas as pd
import pytest

import kackle.correction as cli
from kackle.adapter import (
    AdapterSiteProvider,
    IUPAC,
    adapter_masks,
    parse_min_match,
    parse_tiers,
)
from kackle.bigwig import read_chrom_values, write_bigwig
from kackle.computation import correct_adapter_sites

ADAPTER = "TGGAATTCTCGG"
COMP = str.maketrans("ACGTacgtN", "TGCAtgcaN")


def reference_length(window, adapter, min_match):
    """Pure-Python match length (0 if not a candidate) for one window."""
    mm, last, cand = 0, 0, False
    for k, a in enumerate(adapter):
        b = window[k].upper() if k < len(window) else "N"
        if b in "ACGT" and b in IUPAC[a]:
            last = k + 1
            if last >= min_match[mm]:
                cand = True
        else:
            mm += 1
            if mm > len(min_match) - 1:
                break
    return last if cand else 0


def reference_sites(seq, adapter, min_match, tiers):
    """{strand: [(start, tier)]} by brute force over every full window."""
    k = len(adapter)
    out = {"+": [], "-": []}
    for p in range(len(seq) - k + 1):
        for strand, window in (("+", seq[p : p + k]), ("-", seq[p : p + k].translate(COMP)[::-1])):
            length = reference_length(window, adapter, min_match)
            if length:
                tier = sum(length >= l for l, _ in tiers)
                out[strand].append((p, tier))
    return out


def adapter_rich_genome(tmp_path, size=6000, seed=0, name="chr1"):
    rng = np.random.default_rng(seed)
    seq = list(rng.choice(list("ACGT"), size))
    for pos in rng.integers(0, size - 12, size // 40):
        piece = list(ADAPTER[: rng.integers(3, 13)])
        if rng.random() < 0.5:
            piece[rng.integers(0, len(piece))] = rng.choice(list("ACGT"))
        if rng.random() < 0.5:
            piece = list("".join(piece).translate(COMP)[::-1])
        seq[pos : pos + len(piece)] = piece
    seq = "".join(seq)
    seq = seq[:1000] + "NNNNN" + seq[1005:2000].lower() + seq[2000:]
    fasta = tmp_path / "genome.fa"
    fasta.write_text(f">{name}\n{seq}\n")
    return fasta, seq


@pytest.mark.parametrize("adapter,min_match", [(ADAPTER, (3, 5)), (ADAPTER, (4,)), ("TGGNATTC", (3, 4, 7))])
def test_provider_matches_brute_force(tmp_path, adapter, min_match):
    fasta, seq = adapter_rich_genome(tmp_path)
    tiers = ((6, 5.0), (8, 3.0))
    provider = AdapterSiteProvider(fasta, adapter, min_match=min_match, tiers=tiers)
    want = reference_sites(seq, adapter, min_match, tiers)
    got = provider.starts_for_chrom("chr1")
    assert len(got) == 3
    for strand in "+-":
        expected = sorted(want[strand])
        found = sorted((int(p), t) for t, d in enumerate(got) for p in d[strand])
        assert found == expected
        assert len(expected) > 50
        for d in got:
            assert np.all(np.diff(d[strand]) > 0)
    rng = np.random.default_rng(1)
    for lo, hi in [(0, 10), (995, 1010), (5990, 7000)] + [tuple(sorted(rng.integers(-5, 6005, 2))) for _ in range(20)]:
        region = provider.starts_for_region("chr1", lo, hi)
        for d_all, d_reg in zip(got, region):
            for strand in "+-":
                w = d_all[strand]
                assert d_reg[strand].tolist() == w[(w >= lo) & (w < hi)].tolist()


def test_parsers_and_validation(tmp_path):
    assert parse_min_match("3,5") == (3, 5)
    assert parse_tiers("10:3,8:5") == ((8, 5.0), (10, 3.0))
    assert parse_tiers("") == ()
    for bad in ("", "5,3", "0"):
        with pytest.raises(ValueError):
            parse_min_match(bad)
    with pytest.raises(ValueError):
        adapter_masks("TGGX")
    fasta, _ = adapter_rich_genome(tmp_path)
    p = AdapterSiteProvider(fasta, tiers=((8, 5), (10, 3)), threshold=10)
    assert p.pass_thresholds == (10.0, 5.0, 3.0)
    assert p.max_motif_length == 12


@pytest.mark.parametrize("strand", ["+", "-"])
def test_single_pass_screen_rescreens_without_flagged_sites(strand):
    off = 1 if strand == "+" else 0
    signal = np.ones(200, dtype=np.int32)
    starts = np.array([60, 63, 66, 120], dtype=np.int32)
    art = starts - off
    # 100 fails the plain screen next to 500 (mean ~27) but passes once the
    # 500 is left out; 9 is a candidate that is never flagged.
    signal[art] = [500, 100, 9, 4]
    thr = np.full(4, 10.0)
    draws = np.array([correct_adapter_sites(signal, starts, thr, 10, strand) for _ in range(2000)])
    changed = np.flatnonzero((draws != signal).any(0))
    assert set(changed) == set(art[:2])
    # Only the 500 is flagged in stage 1, so the 100's stage-2 background
    # leaves it out but keeps the unflagged 9: (9 + 17 ones) / 18. The 500's
    # background still holds the 100 (as after a TGG pass): (100 + 9 + 17) / 19.
    assert abs(draws[:, art[1]].mean() - 26 / 18) < 0.1
    assert abs(draws[:, art[0]].mean() - 126 / 19) < 0.3
    # Per-site thresholds: the lone 4 passes at 3x but not at 10x.
    thr[3] = 3.0
    flagged = np.array([correct_adapter_sites(signal, starts, thr, 10, strand)[art[3]] for _ in range(200)])
    assert flagged.mean() < 2


def spiky_inputs(tmp_path, provider, chrom_sizes):
    """Spikes on an empty background at sites with no other site within 2 * source."""
    paths = {}
    for strand, sign in (("+", 1.0), ("-", -1.0)):
        rows = []
        for chrom, size in chrom_sizes:
            anchors = np.sort(np.concatenate([
                d[strand] + (12 if strand == "-" else 0) for d in provider.starts_for_chrom(chrom)
            ]))
            gaps = np.diff(anchors)
            lone = anchors[1:-1][(gaps[:-1] > 20) & (gaps[1:] > 20)]
            vals = np.zeros(size)
            vals[lone - 1 if strand == "+" else lone] = 50
            far = np.ones(size, bool)
            for a in anchors:
                far[max(a - 25, 0) : a + 25] = False
            vals[np.flatnonzero(far)[::5]] = 1
            pos = np.flatnonzero(vals)
            rows.append(pd.DataFrame({"chrom": chrom, "start": pos, "end": pos + 1, "value": sign * vals[pos]}))
        paths[strand] = tmp_path / f"in{strand}.bw"
        write_bigwig(pd.concat(rows, ignore_index=True), paths[strand], chrom_sizes)
    return paths


@pytest.mark.parametrize("backend,workers", [("thread", 1), ("thread", 3), ("process", 2)])
def test_adapter_mode_chunked_matches_whole(tmp_path, backend, workers):
    fasta, seq = adapter_rich_genome(tmp_path, size=20_000)
    chrom_sizes = [("chr1", len(seq))]
    provider = AdapterSiteProvider(fasta)
    inputs = spiky_inputs(tmp_path, provider, chrom_sizes)
    outs = {}
    for label, chunk_size in (("whole", 0), ("chunked", 1_500)):
        outs[label] = {s: tmp_path / f"{label}{s}.bw" for s in "+-"}
        cli.correct_bigwigs(
            inputs, outs[label], chrom_sizes, motif_site_provider=provider,
            chrom_workers=workers, worker_backend=backend, chunk_size=chunk_size,
            verbose=False, resample="base",
        )
    for s in "+-":
        assert outs["chunked"][s].read_bytes() == outs["whole"][s].read_bytes()
        got = read_chrom_values(outs["whole"][s], "chr1")
        inp = np.abs(read_chrom_values(inputs[s], "chr1"))
        assert (inp == 50).sum() > 20
        assert np.array_equal(got, np.where(inp == 50, 0, inp))


def test_cli_rejects_bad_adapter_combinations(monkeypatch, tmp_path):
    base = ["kackle", "-i", "a", "-I", "b", "-o", "c", "-O", "d", "-c", "e"]
    for extra in (["--adapter", ADAPTER], ["-f", "g", "--adapter", ADAPTER, "-k", "TGG"],
                  ["-f", "g", "--adapter", ADAPTER, "--resample", "window"]):
        monkeypatch.setattr("sys.argv", base + extra)
        with pytest.raises(SystemExit):
            cli.parse_args()
    monkeypatch.setattr("sys.argv", base + ["-f", "g", "--adapter", ADAPTER])
    assert cli.parse_args().resample == "base"
    monkeypatch.setattr("sys.argv", base + ["-f", "g"])
    assert cli.parse_args().resample == "window"
