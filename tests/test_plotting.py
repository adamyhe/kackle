import sys

import numpy as np
import pandas as pd
import pytest

from kackle.bigwig import write_bigwig
from kackle.motifs import FastaMotifSiteProvider
from kackle.plotting import (
    chrom_window_sums,
    motif_beds_from_fasta,
    metaplot_profile,
    metaplot_profiles,
    parse_metaplot_args,
    run_metaplot,
    sites_by_chrom,
    write_strand_metaplots,
)


class FakeBigWig:
    def __init__(self, values):
        self._values = np.asarray(values, dtype=np.float64)

        self.chrom_sizes = {"chr1": self._values.shape[0]}

    def read(self, chroms, starts, width, missing=0.0, oob=np.nan):
        out = np.full((len(starts), width), oob, dtype=np.float32)
        for row, start in enumerate(starts):
            stop = min(start + width, self._values.shape[0])
            out[row, : stop - start] = self._values[start:stop]
        return out


def test_metaplot_profile_averages_windows_centered_on_motif_start():
    before = FakeBigWig([0, 1, 2, 3])
    after = FakeBigWig([10, 11, 12, 13])
    sites = pd.DataFrame({"chrom": ["chr1", "chr1"], "start": [2, 0]})

    x, before_mean, after_mean, counts = metaplot_profile(before, after, sites, flank=1)

    assert x.tolist() == [-1, 0, 1]
    assert before_mean.tolist() == [1.0, 1.0, 2.0]
    assert after_mean.tolist() == [11.0, 11.0, 12.0]
    assert counts.tolist() == [1, 2, 2]


class RecordingBigWig(FakeBigWig):
    """FakeBigWig that records how many windows each read call asked for."""

    def __init__(self, values):
        super().__init__(values)
        self.read_rows = []

    def read(self, chroms, starts, width, missing=0.0, oob=np.nan):
        self.read_rows.append(len(starts))
        return super().read(chroms, starts, width, missing=missing, oob=oob)


@pytest.mark.parametrize("batch_size", [1, 3, 7, 50, 10_000])
def test_metaplot_profile_batches_match_single_batch(batch_size):
    rng = np.random.default_rng(0)
    length = 200
    before_values = rng.integers(-5, 20, length).astype(np.float64)
    after_values = rng.integers(0, 5, length).astype(np.float64)
    # Include windows that run off both chromosome ends.
    starts = np.concatenate([[0, 1, length - 1, length - 2], rng.integers(0, length, 40)])
    sites = pd.DataFrame({"chrom": "chr1", "start": starts})

    reference = metaplot_profile(
        FakeBigWig(before_values), FakeBigWig(after_values), sites, flank=5,
        absolute=True, batch_size=len(sites),
    )
    before_bw = RecordingBigWig(before_values)
    batched = metaplot_profile(
        before_bw, FakeBigWig(after_values), sites, flank=5,
        absolute=True, batch_size=batch_size,
    )

    for ref, got in zip(reference, batched):
        np.testing.assert_array_equal(ref, got)
    assert max(before_bw.read_rows) <= batch_size
    assert sum(before_bw.read_rows) == len(sites)


def test_metaplot_profile_with_no_sites_returns_nan_means():
    sites = pd.DataFrame({"chrom": pd.Series([], dtype=str), "start": pd.Series([], dtype=int)})
    _, before_mean, after_mean, counts = metaplot_profile(
        FakeBigWig([1, 2, 3]), FakeBigWig([1, 2, 3]), sites, flank=1
    )
    assert np.isnan(before_mean).all() and np.isnan(after_mean).all()
    assert counts.tolist() == [0, 0, 0]


def test_metaplot_profile_rejects_nonpositive_batch_size():
    sites = pd.DataFrame({"chrom": ["chr1"], "start": [1]})
    with pytest.raises(ValueError):
        metaplot_profile(FakeBigWig([1, 2, 3]), FakeBigWig([1, 2, 3]), sites, flank=1,
                         batch_size=0)


def test_metaplot_profile_can_use_absolute_signal_for_minus_strand():
    before = FakeBigWig([-1, -2, -3])
    after = FakeBigWig([1, 4, 9])
    sites = pd.DataFrame({"chrom": ["chr1"], "start": [1]})

    _, before_mean, after_mean, _ = metaplot_profile(
        before, after, sites, flank=1, absolute=True
    )

    assert before_mean.tolist() == [1.0, 2.0, 3.0]
    assert after_mean.tolist() == [1.0, 4.0, 9.0]


def test_write_strand_metaplots_writes_separate_pngs(tmp_path):
    chrom_sizes = [("chr1", 10)]
    before_pl = tmp_path / "before.plus.bw"
    before_mn = tmp_path / "before.minus.bw"
    after_pl = tmp_path / "after.plus.bw"
    after_mn = tmp_path / "after.minus.bw"
    plus_signal = pd.DataFrame(
        {
            "chrom": ["chr1"] * 3,
            "start": [1, 2, 3],
            "end": [2, 3, 4],
            "value": [1.0, 5.0, 1.0],
        }
    )
    minus_signal = plus_signal.assign(value=[-1.0, -5.0, -1.0])

    write_bigwig(plus_signal, before_pl, chrom_sizes)
    write_bigwig(minus_signal, before_mn, chrom_sizes)
    write_bigwig(plus_signal.assign(value=[1.0, 2.0, 1.0]), after_pl, chrom_sizes)
    write_bigwig(minus_signal.assign(value=[-1.0, -2.0, -1.0]), after_mn, chrom_sizes)

    bed = pd.DataFrame(
        [
            ("chr1", 2, 5, "motif", 0, "+"),
            ("chr1", 2, 5, "motif", 0, "-"),
        ],
        columns=["chrom", "start", "end", "name", "score", "strand"],
    )

    outputs = write_strand_metaplots(
        before_pl, before_mn, after_pl, after_mn, bed, tmp_path / "metaplot", flank=2
    )

    for out_fname in outputs.values():
        out_path = tmp_path / out_fname if not str(out_fname).startswith("/") else out_fname
        with open(out_path, "rb") as handle:
            assert handle.read(8) == b"\x89PNG\r\n\x1a\n"


def test_write_strand_metaplots_skips_sites_missing_from_bigwig_pair(tmp_path):
    chrom_sizes = [("chr1", 10)]
    before_pl = tmp_path / "before.plus.bw"
    before_mn = tmp_path / "before.minus.bw"
    after_pl = tmp_path / "after.plus.bw"
    after_mn = tmp_path / "after.minus.bw"
    signal = pd.DataFrame(
        {"chrom": ["chr1"], "start": [2], "end": [3], "value": [1.0]}
    )
    write_bigwig(signal, before_pl, chrom_sizes)
    write_bigwig(signal.assign(value=[-1.0]), before_mn, chrom_sizes)
    write_bigwig(signal, after_pl, chrom_sizes)
    write_bigwig(signal.assign(value=[-1.0]), after_mn, chrom_sizes)
    bed = pd.DataFrame(
        [
            ("chr1", 2, 5, "motif", 0, "+"),
            ("chr2", 2, 5, "motif", 0, "+"),
            ("chr2", 2, 5, "motif", 0, "-"),
        ],
        columns=["chrom", "start", "end", "name", "score", "strand"],
    )

    outputs = write_strand_metaplots(
        before_pl, before_mn, after_pl, after_mn, bed, tmp_path / "metaplot", flank=2
    )

    assert set(outputs) == {"+", "-"}


def test_metaplot_fasta_mode_skips_chromosomes_missing_from_bigwigs(
    tmp_path, monkeypatch
):
    chrom_sizes = [("chr1", 10)]
    before_pl = tmp_path / "before.plus.bw"
    before_mn = tmp_path / "before.minus.bw"
    after_pl = tmp_path / "after.plus.bw"
    after_mn = tmp_path / "after.minus.bw"
    fasta = tmp_path / "genome.fa"
    chrom_sizes_file = tmp_path / "chrom.sizes"
    signal = pd.DataFrame(
        {"chrom": ["chr1"], "start": [2], "end": [3], "value": [1.0]}
    )
    write_bigwig(signal, before_pl, chrom_sizes)
    write_bigwig(signal.assign(value=[-1.0]), before_mn, chrom_sizes)
    write_bigwig(signal, after_pl, chrom_sizes)
    write_bigwig(signal.assign(value=[-1.0]), after_mn, chrom_sizes)
    fasta.write_text(">chr1\nAATGGAAAAA\n>chr2\nAATGGA\n")
    chrom_sizes_file.write_text("chr1\t10\nchr2\t6\n")
    argv = [
        "kackle-metaplot",
        "--before-pl-bw",
        str(before_pl),
        "--before-mn-bw",
        str(before_mn),
        "--after-pl-bw",
        str(after_pl),
        "--after-mn-bw",
        str(after_mn),
        "-f",
        str(fasta),
        "-c",
        str(chrom_sizes_file),
        "--motif",
        "TGG:0",
        "--motif-match-backend",
        "python",
        "-o",
        str(tmp_path / "metaplot"),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    run_metaplot()

    assert (tmp_path / "metaplot.plus.png").exists()
    assert (tmp_path / "metaplot.minus.png").exists()
    tsv = pd.read_csv(tmp_path / "metaplot.tsv", sep="\t")
    assert list(tsv.columns) == ["strand", "offset", "before_mean", "after_mean", "n_sites"]
    assert len(tsv) == 2 * (2 * 50 + 1)


def test_metaplot_cli_requires_chrom_sizes_with_fasta(monkeypatch):
    argv = [
        "kackle-metaplot",
        "--before-pl-bw",
        "before.plus.bw",
        "--before-mn-bw",
        "before.minus.bw",
        "--after-pl-bw",
        "after.plus.bw",
        "--after-mn-bw",
        "after.minus.bw",
        "-f",
        "genome.fa",
        "-o",
        "out",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit):
        parse_metaplot_args()


# --- genome-scale path: chrom_window_sums / metaplot_profiles -----------------


def _reference_sums(before_values, after_values, centers, flank, absolute):
    """Per-offset sums and counts from the windowed reference implementation."""
    sites = pd.DataFrame({"chrom": "chr1", "start": np.asarray(centers, dtype=np.int64)})
    _, before_mean, after_mean, counts = metaplot_profile(
        FakeBigWig(before_values), FakeBigWig(after_values), sites, flank,
        absolute=absolute,
    )
    with np.errstate(invalid="ignore"):
        return (
            np.nan_to_num(before_mean * counts),
            np.nan_to_num(after_mean * counts),
            counts,
        )


@pytest.mark.parametrize("absolute", [False, True])
@pytest.mark.parametrize("flank", [0, 1, 5, 25])
def test_chrom_window_sums_match_windowed_reference(absolute, flank):
    rng = np.random.default_rng(flank)
    length = 300
    # Sparse integer signal, negative when testing absolute (minus strand).
    sign = -1.0 if absolute else 1.0
    before = sign * rng.poisson(0.3, length).astype(np.float64)
    after = sign * rng.poisson(0.2, length).astype(np.float64)
    centers = np.concatenate(
        [
            [0, 1, length - 1, length - 2],  # windows off both chromosome ends
            [7, 7, 7],  # duplicate sites count once per occurrence
            rng.integers(0, length, 60),
        ]
    )

    got = chrom_window_sums(before, after, centers, flank, absolute=absolute)
    want = _reference_sums(before, after, centers, flank, absolute)

    for g, w in zip(got, want):
        np.testing.assert_allclose(g, w, rtol=0, atol=1e-9)


def test_chrom_window_sums_counts_sites_just_off_the_chromosome():
    # A site 2 bp past either end still reaches the chromosome with flank 3.
    values = np.arange(1, 11, dtype=np.float64)
    centers = np.array([-2, 11])
    before_sum, _, counts = chrom_window_sums(values, values, centers, flank=3)

    # offsets -3..3: site -2 covers offsets 2,3 (bases 0,1); site 11 covers
    # offsets -3,-2 (bases 8,9).
    assert counts.tolist() == [1, 1, 0, 0, 0, 1, 1]
    assert before_sum.tolist() == [9.0, 10.0, 0, 0, 0, 1.0, 2.0]


def test_chrom_window_sums_uses_shorter_track_length():
    before = np.ones(10)
    after = np.ones(8)
    _, _, counts = chrom_window_sums(before, after, [7], flank=1)
    # Base 8 exists only in `before`, so offset +1 isn't counted.
    assert counts.tolist() == [1, 1, 0]


def test_chrom_window_sums_with_no_sites():
    before_sum, after_sum, counts = chrom_window_sums(
        np.ones(5), np.ones(5), np.array([], dtype=np.int64), flank=2
    )
    assert before_sum.tolist() == after_sum.tolist() == [0.0] * 5
    assert counts.tolist() == [0] * 5


def _write_stranded_pair(tmp_path, prefix, chrom_sizes, rng):
    """Write random sparse plus/minus bigWigs; return paths and dense values."""
    dense = {}
    paths = {}
    for strand, sign, suffix in (("+", 1.0, "plus"), ("-", -1.0, "minus")):
        rows = []
        dense[strand] = {}
        for chrom, size in chrom_sizes:
            values = sign * rng.poisson(0.4, size).astype(np.float64)
            dense[strand][chrom] = values
            pos = np.flatnonzero(values)
            rows.append(
                pd.DataFrame(
                    {"chrom": chrom, "start": pos, "end": pos + 1, "value": values[pos]}
                )
            )
        paths[strand] = tmp_path / f"{prefix}.{suffix}.bw"
        write_bigwig(pd.concat(rows, ignore_index=True), paths[strand], chrom_sizes)
    return paths, dense


@pytest.mark.parametrize(
    "chrom_workers,backend", [(1, "process"), (2, "thread"), (2, "process")]
)
def test_metaplot_profiles_match_windowed_reference_across_chromosomes(
    tmp_path, chrom_workers, backend
):
    rng = np.random.default_rng(1)
    chrom_sizes = [("chr1", 400), ("chr2", 250)]
    before, before_dense = _write_stranded_pair(tmp_path, "before", chrom_sizes, rng)
    after, after_dense = _write_stranded_pair(tmp_path, "after", chrom_sizes, rng)
    rows = []
    for chrom, size in chrom_sizes:
        for strand in "+-":
            for start in rng.integers(0, size, 40):
                rows.append((chrom, int(start), int(start) + 3, "m", 0, strand))
    bed = pd.DataFrame(rows, columns=["chrom", "start", "end", "name", "score", "strand"])
    bigwigs = {"+": (str(before["+"]), str(after["+"])), "-": (str(before["-"]), str(after["-"]))}

    profiles = metaplot_profiles(
        bigwigs, ["chr1", "chr2"], 10, sites=sites_by_chrom(bed),
        chrom_workers=chrom_workers, worker_backend=backend,
    )

    for strand, absolute in (("+", False), ("-", True)):
        want = [np.zeros(21), np.zeros(21), np.zeros(21, dtype=np.int64)]
        for chrom, _ in chrom_sizes:
            centers = bed.loc[(bed.chrom == chrom) & (bed.strand == strand), "start"]
            for total, part in zip(
                want,
                _reference_sums(
                    before_dense[strand][chrom], after_dense[strand][chrom],
                    centers, 10, absolute,
                ),
            ):
                total += part
        x, before_mean, after_mean, counts = profiles[strand]
        assert x.tolist() == list(range(-10, 11))
        np.testing.assert_array_equal(counts, want[2])
        np.testing.assert_allclose(before_mean * counts, want[0], atol=1e-9)
        np.testing.assert_allclose(after_mean * counts, want[1], atol=1e-9)


def test_metaplot_profiles_fasta_provider_matches_bed_sites(tmp_path):
    rng = np.random.default_rng(2)
    chrom_sizes = [("chr1", 300), ("chr2", 200)]
    seqs = {c: "".join(rng.choice(list("ACGT"), n)) for c, n in chrom_sizes}
    fasta = tmp_path / "genome.fa"
    fasta.write_text("".join(f">{c}\n{s}\n" for c, s in seqs.items()))
    before, _ = _write_stranded_pair(tmp_path, "before", chrom_sizes, rng)
    after, _ = _write_stranded_pair(tmp_path, "after", chrom_sizes, rng)
    bigwigs = {"+": (str(before["+"]), str(after["+"])), "-": (str(before["-"]), str(after["-"]))}
    motifs = ["TGG:0", "TGGAA:1"]
    chroms = ["chr1", "chr2"]

    bed = motif_beds_from_fasta(fasta, motifs, chroms, match_backend="python")
    from_bed = metaplot_profiles(bigwigs, chroms, 8, sites=sites_by_chrom(bed))
    provider = FastaMotifSiteProvider(fasta, motifs, match_backend="python")
    from_fasta = metaplot_profiles(
        bigwigs, chroms, 8, provider=provider, chrom_workers=2, worker_backend="process"
    )

    for strand in "+-":
        for a, b in zip(from_bed[strand], from_fasta[strand]):
            np.testing.assert_array_equal(a, b)
    assert from_bed["+"][3].max() > 0 and from_bed["-"][3].max() > 0


def test_metaplot_profiles_requires_exactly_one_site_source():
    with pytest.raises(ValueError):
        metaplot_profiles({}, [], 5)


def test_chrom_window_sums_does_not_modify_inputs():
    before = np.array([-1.0, -2.0, np.nan, -4.0, 0.0, -6.0])
    after = before.copy()
    snapshot = before.copy()
    chrom_window_sums(before, after, [2, 3], flank=1, absolute=True)
    np.testing.assert_array_equal(before, snapshot)
    np.testing.assert_array_equal(after, snapshot)


@pytest.mark.parametrize("duplicates,dtype", [(3, np.uint8), (300, np.uint16)])
def test_site_count_vector_widens_dtype_for_many_duplicates(duplicates, dtype):
    from kackle.plotting import _site_count_vector

    centers = np.array([5] * duplicates + [9], dtype=np.int64)
    site_count = _site_count_vector(centers, length=20, flank=2)
    assert site_count.dtype == dtype
    assert site_count[5 + 2] == duplicates and site_count[9 + 2] == 1

    values = np.arange(20, dtype=np.float64)
    before_sum, _, counts = chrom_window_sums(values, values, centers, flank=2)
    want = _reference_sums(values, values, centers, 2, False)
    np.testing.assert_allclose(before_sum, want[0])
    np.testing.assert_array_equal(counts, want[2])


def test_dense_signal_in_place_handles_abs_and_non_finite():
    from kackle.plotting import _dense_signal

    values = np.array([-1.0, np.inf, -3.0, np.nan, 5.0], dtype=np.float32)
    out = _dense_signal(values, 4, absolute=True, in_place=True)
    assert np.shares_memory(out, values)
    assert out.tolist() == [1.0, 0.0, 3.0, 0.0]
