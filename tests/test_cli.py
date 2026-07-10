import numpy as np
import pandas as pd
import pytest

import kackle.correction as cli


def test_bed_starts_for_strand_uses_kmer_start_for_plus_and_end_for_minus():
    bed = pd.DataFrame(
        [
            ("chr1", 2, 5, "motif", 0, "+"),
            ("chr1", 7, 10, "motif", 0, "-"),
            ("chr2", 11, 14, "motif", 0, "+"),
        ],
        columns=["chrom", "start", "end", "name", "score", "strand"],
    )

    assert cli.bed_starts_for_strand(bed, "chr1", "+").tolist() == [2]
    assert cli.bed_starts_for_strand(bed, "chr1", "-").tolist() == [10]


def test_kmer_resample_applies_motif_beds_sequentially(monkeypatch):
    class FakeBigWig:
        def chroms(self, chrom):
            return 5

        def values(self, chrom, start, end, fillna=0.0):
            return np.array([1, 2, 3, 4, 5])

        def close(self):
            pass

    bed1 = pd.DataFrame(
        [("chr1", 2, 5, "short", 0, "+")],
        columns=["chrom", "start", "end", "name", "score", "strand"],
    )
    bed2 = pd.DataFrame(
        [("chr1", 4, 9, "long", 0, "+")],
        columns=["chrom", "start", "end", "name", "score", "strand"],
    )
    calls = []

    monkeypatch.setattr(cli.pybigtools, "open", lambda path: FakeBigWig())

    def fake_correct_kmers(values, starts, source, target, threshold, strand):
        calls.append((starts.tolist(), strand, values.tolist()))
        return values + len(calls) * 10

    monkeypatch.setattr(cli, "correct_kmers", fake_correct_kmers)

    out = cli.kmer_resample(
        "signal.bw",
        motif_beds=[bed1, bed2],
        strand="+",
        chroms=["chr1"],
        verbose=False,
    )

    assert calls == [
        ([2], "+", [1, 2, 3, 4, 5]),
        ([4], "+", [11, 12, 13, 14, 15]),
    ]
    assert out.value.tolist() == [31.0, 32.0, 33.0, 34.0, 35.0]


def test_kmer_resample_uses_motif_site_provider_per_chromosome(monkeypatch):
    class FakeBigWig:
        def chroms(self, chrom):
            return 3

        def values(self, chrom, start, end, fillna=0.0):
            return np.array([1, 2, 3])

        def close(self):
            pass

    class Provider:
        def __init__(self):
            self.chroms = []

        def for_chrom(self, chrom):
            self.chroms.append(chrom)
            return [
                pd.DataFrame(
                    [(chrom, 1, 4, "motif", 0, "+")],
                    columns=["chrom", "start", "end", "name", "score", "strand"],
                )
            ]

    provider = Provider()
    calls = []

    monkeypatch.setattr(cli.pybigtools, "open", lambda path: FakeBigWig())

    def fake_correct_kmers(values, starts, source, target, threshold, strand):
        calls.append((starts.tolist(), strand))
        return values

    monkeypatch.setattr(cli, "correct_kmers", fake_correct_kmers)

    cli.kmer_resample(
        "signal.bw",
        motif_site_provider=provider,
        strand="+",
        chroms=["chr1", "chr2"],
        verbose=False,
    )

    assert provider.chroms == ["chr1", "chr2"]
    assert calls == [([1], "+"), ([1], "+")]


def test_kmer_resample_returns_sparse_rows_without_dense_dataframe(monkeypatch):
    class FakeBigWig:
        def chroms(self, chrom):
            return 6

        def values(self, chrom, start, end, fillna=0.0):
            return np.array([0, 2, 0, 4, 0, 0])

        def close(self):
            pass

    monkeypatch.setattr(cli.pybigtools, "open", lambda path: FakeBigWig())
    monkeypatch.setattr(cli, "correct_kmers", lambda values, *args: values)

    out = cli.kmer_resample(
        "signal.bw",
        motif_beds=[
            pd.DataFrame(
                [],
                columns=["chrom", "start", "end", "name", "score", "strand"],
            )
        ],
        strand="+",
        chroms=["chr1"],
        verbose=False,
    )

    assert out.to_dict("records") == [
        {"chrom": "chr1", "start": 1, "end": 2, "value": 2.0},
        {"chrom": "chr1", "start": 3, "end": 4, "value": 4.0},
    ]


def test_kmer_resample_requires_bed_or_motif_beds():
    with pytest.raises(ValueError, match="Either bed6_fname, motif_beds"):
        cli.kmer_resample("signal.bw", chroms=["chr1"], verbose=False)


def test_parse_args_rejects_fasta_only_options_with_bed(monkeypatch):
    argv = [
        "kackle",
        "-i",
        "plus.bw",
        "-I",
        "minus.bw",
        "-b",
        "sites.bed",
        "-k",
        "TGG:0",
        "-o",
        "out.plus.bw",
        "-O",
        "out.minus.bw",
        "-c",
        "chrom.sizes",
    ]
    monkeypatch.setattr("sys.argv", argv)

    with pytest.raises(SystemExit):
        cli.parse_args()


def test_parse_args_defaults_to_fast_fasta_backends(monkeypatch):
    argv = [
        "kackle",
        "-i",
        "plus.bw",
        "-I",
        "minus.bw",
        "-f",
        "genome.fa",
        "-o",
        "out.plus.bw",
        "-O",
        "out.minus.bw",
        "-c",
        "chrom.sizes",
    ]
    monkeypatch.setattr("sys.argv", argv)

    args = cli.parse_args()

    assert args.fasta_backend == "pyfastx"
    assert args.motif_match_backend == "ahocorasick"


def test_parse_args_help_describes_modes_and_defaults(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["kackle", "--help"])

    with pytest.raises(SystemExit) as exc:
        cli.parse_args()

    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "Correct strand-specific bigWig signal tracks" in help_text
    assert "Precomputed BED6 motif locations" in help_text
    assert "Reference FASTA used to locate motif sites directly" in help_text
    assert "defaults to motif passes TGG:0, TGGAA:1" in help_text
    assert "pyfastx plus" in help_text
    assert "(default: pyfastx)" in help_text
    assert "(default: ahocorasick)" in help_text
