import sys

import numpy as np
import pandas as pd
import pytest

from kackle.bigwig import write_bigwig
from kackle.plotting import (
    metaplot_profile,
    parse_metaplot_args,
    run_metaplot,
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
