import numpy as np
import pandas as pd
import pytest

import kackle.correction as cli
from kackle.motifs import locate_motif_specs
from kackle.cli_common import intersect_ordered


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


def test_motif_bed6_fnames_are_ordered_and_descriptive():
    assert cli.motif_bed6_fnames("sites/kackle", ["TGG:0", "TGGAA:1"]) == [
        "sites/kackle.1.TGG.m0.bed6",
        "sites/kackle.2.TGGAA.m1.bed6",
    ]


def test_auto_worker_resolution_respects_numba_thread_budget(monkeypatch):
    monkeypatch.setenv("NUMBA_NUM_THREADS", "16")

    chrom_workers = cli.resolve_chrom_workers("auto", 24)
    numba_threads = cli.resolve_numba_threads("auto", chrom_workers)

    assert chrom_workers == 4
    assert numba_threads == 4


def test_thread_budget_follows_cpu_affinity_not_machine_size(monkeypatch):
    # A Slurm job on a big node: os.cpu_count() is the machine, the job gets 3.
    from numba import config

    monkeypatch.delenv("NUMBA_NUM_THREADS", raising=False)
    monkeypatch.setattr(cli.os, "cpu_count", lambda: 256)
    monkeypatch.setattr(cli.os, "sched_getaffinity", lambda pid: {0, 1, 2}, raising=False)
    assert cli.thread_budget() == min(3, config.NUMBA_NUM_THREADS)
    # Never more than numba allows.
    monkeypatch.setattr(cli.os, "sched_getaffinity", lambda pid: set(range(10_000)), raising=False)
    assert cli.thread_budget() == config.NUMBA_NUM_THREADS


def test_intersect_chrom_sizes_preserves_chrom_size_order():
    chrom_sizes = [("chr1", 10), ("chr2", 20), ("chr3", 30)]

    assert intersect_ordered(
        chrom_sizes,
        ["chr2", "chr1"],
        ["chr1", "chr3", "chr2"],
        key=lambda chrom_size: chrom_size[0],
    ) == [("chr1", 10), ("chr2", 20)]


def test_motif_anchor_arrays_use_chrom_and_strand_specific_anchor():
    bed = pd.DataFrame(
        [
            ("chr1", 1, 4, "motif", 0, "+"),
            ("chr1", 5, 8, "motif", 0, "-"),
            ("chr2", 9, 12, "motif", 0, "+"),
        ],
        columns=["chrom", "start", "end", "name", "score", "strand"],
    )

    plus_starts = cli.motif_anchor_arrays([bed], "+", "chr1")
    minus_starts = cli.motif_anchor_arrays([bed], "-", "chr1")

    assert plus_starts[0].tolist() == [1]
    assert minus_starts[0].tolist() == [8]


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

    monkeypatch.setattr(
        cli,
        "read_chrom_values",
        lambda path, chrom, n_jobs=1: FakeBigWig().values(
            chrom, 0, FakeBigWig().chroms(chrom)
        ),
    )

    def fake_correct_kmers(values, starts, source, target, threshold, strand, min_reads=1):
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

    monkeypatch.setattr(
        cli,
        "read_chrom_values",
        lambda path, chrom, n_jobs=1: FakeBigWig().values(
            chrom, 0, FakeBigWig().chroms(chrom)
        ),
    )

    def fake_correct_kmers(values, starts, source, target, threshold, strand, min_reads=1):
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


def test_kmer_resample_parallel_chrom_workers_match_serial(monkeypatch):
    class FakeBigWig:
        def chroms(self, chrom):
            return 4

        def values(self, chrom, start, end, fillna=0.0):
            if chrom == "chr1":
                return np.array([0, 1, 0, 2])
            return np.array([3, 0, 4, 0])

        def close(self):
            pass

    empty_bed = pd.DataFrame(
        [], columns=["chrom", "start", "end", "name", "score", "strand"]
    )
    monkeypatch.setattr(
        cli,
        "read_chrom_values",
        lambda path, chrom, n_jobs=1: FakeBigWig().values(
            chrom, 0, FakeBigWig().chroms(chrom)
        ),
    )
    monkeypatch.setattr(cli, "correct_kmers", lambda values, *args: values)

    serial = cli.kmer_resample(
        "signal.bw",
        motif_beds=[empty_bed],
        strand="+",
        chroms=["chr1", "chr2"],
        chrom_workers=1,
        verbose=False,
    )
    parallel = cli.kmer_resample(
        "signal.bw",
        motif_beds=[empty_bed],
        strand="+",
        chroms=["chr1", "chr2"],
        chrom_workers=2,
        verbose=False,
    )

    assert parallel.to_dict("records") == serial.to_dict("records")


def test_kmer_resample_uses_precomputed_start_arrays(monkeypatch):
    class FakeBigWig:
        def chroms(self, chrom):
            return 4

        def values(self, chrom, start, end, fillna=0.0):
            return np.array([0, 1, 0, 2])

        def close(self):
            pass

    bed = pd.DataFrame(
        [("chr1", 1, 4, "motif", 0, "+")],
        columns=["chrom", "start", "end", "name", "score", "strand"],
    )
    calls = []
    monkeypatch.setattr(
        cli,
        "read_chrom_values",
        lambda path, chrom, n_jobs=1: FakeBigWig().values(
            chrom, 0, FakeBigWig().chroms(chrom)
        ),
    )

    def fake_correct_kmers(values, starts, source, target, threshold, strand, min_reads=1):
        calls.append(starts.tolist())
        return values

    monkeypatch.setattr(cli, "correct_kmers", fake_correct_kmers)

    cli.kmer_resample(
        "signal.bw",
        motif_beds=[bed],
        strand="+",
        chroms=["chr1"],
        verbose=False,
    )

    assert calls == [[1]]


def test_kmer_resample_auto_chrom_workers_match_serial(monkeypatch):
    class FakeBigWig:
        def chroms(self, chrom):
            return 4

        def values(self, chrom, start, end, fillna=0.0):
            if chrom == "chr1":
                return np.array([0, 1, 0, 2])
            return np.array([3, 0, 4, 0])

        def close(self):
            pass

    empty_bed = pd.DataFrame(
        [], columns=["chrom", "start", "end", "name", "score", "strand"]
    )
    monkeypatch.setenv("NUMBA_NUM_THREADS", "16")
    monkeypatch.setattr(
        cli,
        "read_chrom_values",
        lambda path, chrom, n_jobs=1: FakeBigWig().values(
            chrom, 0, FakeBigWig().chroms(chrom)
        ),
    )
    monkeypatch.setattr(cli, "correct_kmers", lambda values, *args: values)

    serial = cli.kmer_resample(
        "signal.bw",
        motif_beds=[empty_bed],
        strand="+",
        chroms=["chr1", "chr2"],
        chrom_workers=1,
        verbose=False,
    )
    auto = cli.kmer_resample(
        "signal.bw",
        motif_beds=[empty_bed],
        strand="+",
        chroms=["chr1", "chr2"],
        chrom_workers="auto",
        verbose=False,
    )

    assert auto.to_dict("records") == serial.to_dict("records")


def test_kmer_resample_parallel_fasta_provider_writes_bed6_once(tmp_path, monkeypatch):
    class FakeBigWig:
        def chroms(self, chrom):
            return 12

        def values(self, chrom, start, end, fillna=0.0):
            return np.ones(12, dtype=np.int32)

        def close(self):
            pass

    fasta = tmp_path / "genome.fa"
    bed6 = tmp_path / "sites.bed6"
    fasta.write_text(">chr1\nTGGATGGAA\n>chr2\nACCATGG\n")
    provider = cli.FastaMotifSiteProvider(
        fasta,
        ["TGG:0"],
        match_backend="python",
        bed6_fnames=[bed6],
    )
    monkeypatch.setattr(
        cli,
        "read_chrom_values",
        lambda path, chrom, n_jobs=1: FakeBigWig().values(
            chrom, 0, FakeBigWig().chroms(chrom)
        ),
    )
    monkeypatch.setattr(cli, "correct_kmers", lambda values, *args: values)

    cli.kmer_resample(
        "signal.bw",
        motif_site_provider=provider,
        strand="+",
        chroms=["chr1", "chr2"],
        chrom_workers=2,
        verbose=False,
    )
    cli.kmer_resample(
        "signal.bw",
        motif_site_provider=provider,
        strand="-",
        chroms=["chr1", "chr2"],
        chrom_workers=2,
        verbose=False,
    )

    assert bed6.read_text().splitlines() == [
        "chr1\t0\t3\tTGG\t0\t+",
        "chr1\t4\t7\tTGG\t0\t+",
        "chr2\t4\t7\tTGG\t0\t+",
        "chr2\t1\t4\tTGG\t0\t-",
    ]


def test_kmer_resample_returns_sparse_rows_without_dense_dataframe(monkeypatch):
    class FakeBigWig:
        def chroms(self, chrom):
            return 6

        def values(self, chrom, start, end, fillna=0.0):
            return np.array([0, 2, 0, 4, 0, 0])

        def close(self):
            pass

    monkeypatch.setattr(
        cli,
        "read_chrom_values",
        lambda path, chrom, n_jobs=1: FakeBigWig().values(
            chrom, 0, FakeBigWig().chroms(chrom)
        ),
    )
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


def test_kmer_resample_rejects_invalid_chrom_workers():
    with pytest.raises(ValueError, match="chrom_workers"):
        cli.kmer_resample(
            "signal.bw",
            motif_beds=[
                pd.DataFrame(
                    [],
                    columns=["chrom", "start", "end", "name", "score", "strand"],
                )
            ],
            chrom_workers=0,
            verbose=False,
        )


def test_kmer_resample_rejects_invalid_worker_backend():
    with pytest.raises(ValueError, match="worker_backend"):
        cli.kmer_resample(
            "signal.bw",
            motif_beds=[
                pd.DataFrame(
                    [],
                    columns=["chrom", "start", "end", "name", "score", "strand"],
                )
            ],
            worker_backend="bogus",
            verbose=False,
        )


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


def test_parse_args_rejects_bed_output_prefix_with_bed_input(monkeypatch):
    argv = [
        "kackle",
        "-i",
        "plus.bw",
        "-I",
        "minus.bw",
        "-b",
        "sites.bed",
        "--out-bed6-prefix",
        "generated",
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


def test_parse_args_rejects_invalid_chrom_workers(monkeypatch):
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
        "--chrom-workers",
        "0",
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

    assert args.motif_match_backend == "auto"
    assert args.chrom_workers == "auto"
    assert args.numba_threads == "auto"
    assert args.worker_backend == "process"


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
    assert "pyfaidx and a numba motif matcher" in help_text
    assert "(default: auto)" in help_text
    assert "--chrom-workers" in help_text
    assert "--numba-threads" in help_text
    assert "--worker-backend" in help_text


@pytest.mark.parametrize("strand", ["+", "-"])
@pytest.mark.parametrize("keep_sites", [False, True])
def test_provider_anchor_arrays_match_bed_mode_anchors(tmp_path, strand, keep_sites):
    fasta = tmp_path / "genome.fa"
    fasta.write_text(">chr1\nTGGATGGAAnnttccaTGGcTTCCAtggaa\n>chr2\nACCATGG\n")
    specs = ["TGG:0", "TGGAA:1"]
    provider = cli.FastaMotifSiteProvider(fasta, specs)
    beds = locate_motif_specs(fasta, specs)
    for chrom in ("chr1", "chr2"):
        anchors, sites = cli._provider_anchor_arrays(
            provider, chrom, strand, keep_sites=keep_sites, isolate_fasta=False
        )
        want = cli.motif_anchor_arrays(beds, strand, chrom)
        assert [a.tolist() for a in anchors] == [w.tolist() for w in want]
        assert all(a.dtype == np.int32 for a in anchors)
        assert (sites is not None) == keep_sites


def test_bed_anchor_index_matches_per_chrom_filtering(tmp_path):
    fasta = tmp_path / "genome.fa"
    fasta.write_text(">chr1\nTGGATGGAAnnttccaTGGcTTCCAtggaa\n>chr2\nACCATGG\n")
    beds = locate_motif_specs(fasta, ["TGG:0", "TGGAA:1"])
    chroms = ["chr1", "chr2", "chr3"]
    index = cli.bed_anchor_index(beds, chroms)
    for chrom in chroms:
        for strand in ("+", "-"):
            got = index[chrom][strand]
            want = cli.motif_anchor_arrays(beds, strand, chrom)
            assert [a.tolist() for a in got] == [w.tolist() for w in want]
            assert all(a.dtype == np.int32 for a in got)


def _random_signal_bigwigs(tmp_path, chrom_sizes, seed=0):
    from kackle.bigwig import write_bigwig

    rng = np.random.default_rng(seed)
    paths = {}
    for strand, sign in (("+", 1.0), ("-", -1.0)):
        rows = []
        for chrom, size in chrom_sizes:
            pos = np.sort(rng.choice(size, size // 3, replace=False))
            vals = rng.integers(1, 6, pos.size).astype(float)
            spikes = rng.random(pos.size) < 0.05
            vals[spikes] *= 40
            rows.append(
                pd.DataFrame(
                    {"chrom": chrom, "start": pos, "end": pos + 1, "value": sign * vals}
                )
            )
        paths[strand] = tmp_path / f"in{strand}.bw"
        write_bigwig(pd.concat(rows, ignore_index=True), paths[strand], chrom_sizes)
    return paths


@pytest.mark.parametrize(
    "chrom_workers,worker_backend",
    [(1, "thread"), (3, "thread"), (2, "process")],
)
@pytest.mark.parametrize("mode", ["fasta", "bed"])
def test_correct_bigwigs_matches_per_strand_kmer_resample(
    tmp_path, monkeypatch, chrom_workers, worker_backend, mode
):
    from kackle.bigwig import read_chrom_values, write_bigwig

    rng = np.random.default_rng(1)
    chrom_sizes = [("chr1", 3000), ("chr2", 500), ("chr3", 2000), ("chrM", 40)]
    fasta = tmp_path / "genome.fa"
    fasta.write_text(
        "".join(
            f">{chrom}\n{''.join(rng.choice(list('ACGTN'), size, p=[.24, .24, .24, .24, .04]))}\n"
            for chrom, size in chrom_sizes
        )
    )
    inputs = _random_signal_bigwigs(tmp_path, chrom_sizes)
    specs = ["TGG:0", "TGGAA:1"]
    # Deterministic stand-in for the resampler: zero the target window of
    # every anchor, so outputs are comparable without RNG. Spawned/forkserver
    # worker processes would not see the patch, so the process case keeps the
    # real kernel with a threshold nothing reaches (plumbing and order only).
    threshold = 10
    if worker_backend == "process":
        threshold = 10**9
    else:
        monkeypatch.setattr(cli, "correct_kmers", _zeroing_correct_kmers)

    def site_args(prefix):
        if mode == "bed":
            return {"motif_beds": locate_motif_specs(fasta, specs)}
        return {
            "motif_site_provider": cli.FastaMotifSiteProvider(
                fasta, specs, bed6_fnames=cli.motif_bed6_fnames(prefix, specs)
            )
        }

    want = {}
    old_args = site_args(tmp_path / "old")
    for strand in ("+", "-"):
        rows = cli.kmer_resample(
            inputs[strand], strand=strand, chroms=[c for c, _ in chrom_sizes],
            source=4, target=2, threshold=threshold, verbose=False, **old_args,
        )
        want[strand] = tmp_path / f"want{strand}.bw"
        write_bigwig(rows, want[strand], chrom_sizes)

    got = {strand: tmp_path / f"got{strand}.bw" for strand in ("+", "-")}
    cli.correct_bigwigs(
        inputs, got, chrom_sizes, source=4, target=2, threshold=threshold,
        chrom_workers=chrom_workers,
        worker_backend=worker_backend, verbose=False, **site_args(tmp_path / "new"),
    )

    for strand in ("+", "-"):
        assert got[strand].read_bytes() == want[strand].read_bytes()
        for chrom, _ in chrom_sizes:
            before = np.abs(read_chrom_values(inputs[strand], chrom))
            after = read_chrom_values(got[strand], chrom)
            assert (after <= before).all()
    if mode == "fasta":
        for old, new in zip(
            cli.motif_bed6_fnames(tmp_path / "old", specs),
            cli.motif_bed6_fnames(tmp_path / "new", specs),
        ):
            assert open(old).read() == open(new).read()
            assert open(new).read()


def _zeroing_correct_kmers(values, starts, source, target, threshold, strand, min_reads=1):
    out = values.copy()
    for anchor in starts:
        out[max(anchor - target, 0) : anchor + target] = 0
    return out


def test_correct_bigwigs_requires_sites(tmp_path):
    with pytest.raises(ValueError, match="Either bed6_fname, motif_beds"):
        cli.correct_bigwigs({"+": "a", "-": "b"}, {"+": "c", "-": "d"}, [("chr1", 5)])
