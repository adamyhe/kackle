import pytest

from kackle.motifs import (
    FastaMotifSiteProvider,
    fasta_chrom_names,
    locate_motif,
    locate_motif_in_sequence,
    locate_motif_specs,
    mismatch_variants,
    parse_motif_spec,
)


def write_fasta(path, records):
    with open(path, "w") as handle:
        for name, sequence in records:
            handle.write(f">{name}\n{sequence}\n")


def test_parse_motif_spec_defaults_to_exact_matching():
    assert parse_motif_spec("TGG") == ("TGG", 0)
    assert parse_motif_spec("TGGAA:1") == ("TGGAA", 1)


def test_parse_motif_spec_rejects_negative_mismatches():
    with pytest.raises(ValueError, match=">= 0"):
        parse_motif_spec("TGG:-1")


def test_mismatch_variants_expands_hamming_neighborhood():
    variants = mismatch_variants("TGGAA", 1)

    assert len(variants) == 16
    assert "TGGAA" in variants
    assert "AGGAA" in variants
    assert "TGGAT" in variants
    assert "AGGAT" not in variants


def test_mismatch_variants_validates_motif_bases():
    with pytest.raises(ValueError, match="Motif must not be empty"):
        mismatch_variants("", 0)
    with pytest.raises(ValueError, match="Unsupported motif bases"):
        mismatch_variants("TGN", 0)


def test_locate_motif_emits_bed6_style_overlapping_and_reverse_hits(tmp_path):
    fasta = tmp_path / "genome.fa"
    write_fasta(fasta, [("chr1", "TGGATGGAAACCA")])

    bed = locate_motif(fasta, "TGG", both_strands=True)

    assert bed.to_dict("records") == [
        {"chrom": "chr1", "start": 0, "end": 3, "name": "TGG", "score": 0, "strand": "+"},
        {"chrom": "chr1", "start": 4, "end": 7, "name": "TGG", "score": 0, "strand": "+"},
        {"chrom": "chr1", "start": 10, "end": 13, "name": "TGG", "score": 0, "strand": "-"},
    ]


def test_locate_motif_supports_case_insensitive_one_mismatch_search(tmp_path):
    fasta = tmp_path / "genome.fa"
    write_fasta(fasta, [("chr1", "tggatggaaacCa")])

    bed = locate_motif(fasta, "TGGAA", max_mismatches=1, both_strands=True)

    assert bed[["chrom", "start", "end", "strand"]].to_records(index=False).tolist() == [
        ("chr1", 0, 5, "+"),
        ("chr1", 4, 9, "+"),
    ]


def test_locate_motif_can_skip_reverse_strand(tmp_path):
    fasta = tmp_path / "genome.fa"
    write_fasta(fasta, [("chr1", "TGGACCA")])

    bed = locate_motif(fasta, "TGG", both_strands=False)

    assert bed[["start", "strand"]].to_records(index=False).tolist() == [(0, "+")]


@pytest.mark.parametrize("backend", ["numba", "auto"])
def test_locate_motif_in_sequence_backends_match_python(backend):
    sequence = "TGGATGGAAACCATGGAT"

    python_bed = locate_motif_in_sequence(
        "chr1",
        sequence,
        "TGGAA",
        max_mismatches=1,
        both_strands=True,
        match_backend="python",
    )
    other_bed = locate_motif_in_sequence(
        "chr1",
        sequence,
        "TGGAA",
        max_mismatches=1,
        both_strands=True,
        match_backend=backend,
    )

    assert other_bed.to_dict("records") == python_bed.to_dict("records")


def test_fasta_motif_site_provider_locates_one_chromosome_at_a_time(tmp_path):
    fasta = tmp_path / "genome.fa"
    write_fasta(fasta, [("chr1", "TGGATGGAA"), ("chr2", "ACCATGG")])
    provider = FastaMotifSiteProvider(
        fasta,
        ["TGG:0", "TGGAA:1"],
        both_strands=True,
    )

    tgg_bed, tggaa_bed = provider.for_chrom("chr2")

    assert tgg_bed.chrom.unique().tolist() == ["chr2"]
    assert tgg_bed[["start", "strand"]].to_records(index=False).tolist() == [
        (4, "+"),
        (1, "-"),
    ]
    assert tggaa_bed.empty


@pytest.mark.parametrize("backend", ["numba", "auto"])
def test_fasta_motif_site_provider_backends_match_python(tmp_path, backend):
    fasta = tmp_path / "genome.fa"
    write_fasta(fasta, [("chr1", "TGGATGGAA"), ("chr2", "ACCATGG")])
    python_provider = FastaMotifSiteProvider(
        fasta,
        ["TGG:0", "TGGAA:1"],
        both_strands=True,
        match_backend="python",
    )
    other_provider = FastaMotifSiteProvider(
        fasta,
        ["TGG:0", "TGGAA:1"],
        both_strands=True,
        match_backend=backend,
    )

    python_beds = python_provider.for_chrom("chr1")
    other_beds = other_provider.for_chrom("chr1")

    assert [bed.to_dict("records") for bed in other_beds] == [
        bed.to_dict("records") for bed in python_beds
    ]


def test_fasta_chrom_names_and_sequences_use_first_header_token(tmp_path):
    fasta = tmp_path / "genome.fa"
    fasta.write_text(">chr1 assembled chromosome\ntggaTGG\nAA\n>chr2\nACCA\n")

    provider = FastaMotifSiteProvider(fasta, ["TGG:0"], match_backend="python")

    assert fasta_chrom_names(fasta) == ["chr1", "chr2"]
    assert provider.sequence("chr1") == "tggaTGGAA"
    with pytest.raises(KeyError):
        provider.sequence("chr3")


def test_fasta_motif_site_provider_streams_bed6_outputs_once_per_chromosome(tmp_path):
    fasta = tmp_path / "genome.fa"
    tgg_bed = tmp_path / "tgg.bed6"
    tggaa_bed = tmp_path / "tggaa.bed6"
    write_fasta(fasta, [("chr1", "TGGATGGAA"), ("chr2", "ACCATGG")])
    provider = FastaMotifSiteProvider(
        fasta,
        ["TGG:0", "TGGAA:1"],
        both_strands=True,
        match_backend="python",
        bed6_fnames=[tgg_bed, tggaa_bed],
    )

    provider.for_chrom("chr1")
    provider.for_chrom("chr1")
    provider.for_chrom("chr2")

    assert tgg_bed.read_text().splitlines() == [
        "chr1\t0\t3\tTGG\t0\t+",
        "chr1\t4\t7\tTGG\t0\t+",
        "chr2\t4\t7\tTGG\t0\t+",
        "chr2\t1\t4\tTGG\t0\t-",
    ]
    assert tggaa_bed.read_text().splitlines() == [
        "chr1\t0\t5\tTGGAA\t0\t+",
        "chr1\t4\t9\tTGGAA\t0\t+",
    ]


def test_locate_motif_specs_preserves_correction_order(tmp_path):
    fasta = tmp_path / "genome.fa"
    write_fasta(fasta, [("chr1", "TGGATGGAA")])

    tgg_bed, tggaa_bed = locate_motif_specs(fasta, ["TGG:0", "TGGAA:1"])

    assert tgg_bed.name.unique().tolist() == ["TGG"]
    assert tggaa_bed.name.unique().tolist() == ["TGGAA"]


# --- start-array matchers ------------------------------------------------------

import numpy as np  # noqa: E402

from kackle.motifs import (  # noqa: E402
    RC_TABLE,
    _sort_bed,
    bed_from_starts,
    locate_motif_starts,
    resolve_match_backend,
)

BACKENDS = ["numba", "python"]


def brute_force_starts(sequence, pattern, max_mismatches):
    """Window-by-window definition: all A/C/G/T (any case), <= N mismatches."""
    upper = sequence.upper()
    k = len(pattern)
    hits = []
    for start in range(len(upper) - k + 1):
        window = upper[start : start + k]
        if set(window) <= set("ACGT"):
            if sum(a != b for a, b in zip(window, pattern)) <= max_mismatches:
                hits.append(start)
    return hits


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "motif,max_mismatches",
    [("TGG", 0), ("TGGAA", 1), ("TGGAA", 2), ("ACGT", 1), ("A", 0), ("GATTACA", 7)],
)
def test_locate_motif_starts_matches_brute_force(backend, motif, max_mismatches):
    rng = np.random.default_rng(len(motif) * 10 + max_mismatches)
    # Mostly ACGT in both cases, with N, IUPAC codes, a gap character and a
    # non-ASCII character mixed in. ACGT is palindromic: + and - hits coincide.
    alphabet = list("ACGTACGTacgtNnMR-é")
    for length in (0, 1, len(motif) - 1, len(motif), 50, 400):
        sequence = "".join(rng.choice(alphabet, max(length, 0)))
        got = locate_motif_starts(
            sequence, motif, max_mismatches, match_backend=backend
        )
        rc = motif.translate(RC_TABLE)[::-1]
        assert got["+"].tolist() == brute_force_starts(sequence, motif, max_mismatches)
        assert got["-"].tolist() == brute_force_starts(sequence, rc, max_mismatches)
        for strand in "+-":
            assert got[strand].dtype == np.int64


def test_numba_kernel_spans_chunk_boundaries():
    # Long enough for several 65,536-window chunks, with hits planted across
    # each boundary.
    rng = np.random.default_rng(3)
    sequence = np.array(list(rng.choice(list("ACGTN"), 200_000)))
    for boundary in (65_536, 131_072, 196_608):
        sequence[boundary - 2 : boundary + 3] = list("TGGAA")
    sequence = "".join(sequence)
    numba = locate_motif_starts(sequence, "TGGAA", 1, match_backend="numba")
    python = locate_motif_starts(sequence, "TGGAA", 1, match_backend="python")
    for strand in "+-":
        np.testing.assert_array_equal(numba[strand], python[strand])
    assert {65_534, 131_070, 196_606} <= set(numba["+"].tolist())


def test_locate_motif_starts_strand_selection():
    sequence = "TGGATGGAAACCATGGAT"
    both = locate_motif_starts(sequence, "TGG")
    plus_only = locate_motif_starts(sequence, "TGG", both_strands=False)
    minus_only = locate_motif_starts(sequence, "TGG", strands=("-",))
    assert plus_only["+"].tolist() == both["+"].tolist()
    assert plus_only["-"].size == 0
    assert minus_only["-"].tolist() == both["-"].tolist()
    assert minus_only["+"].size == 0


def test_locate_motif_starts_validates_inputs():
    with pytest.raises(ValueError):
        locate_motif_starts("ACGT", "TGN")
    with pytest.raises(ValueError):
        locate_motif_starts("ACGT", "")
    with pytest.raises(ValueError):
        locate_motif_starts("ACGT", "TGG", max_mismatches=-1)
    with pytest.raises(ValueError):
        resolve_match_backend("regex")


def test_ahocorasick_backend_is_a_deprecated_alias_for_auto():
    with pytest.warns(DeprecationWarning):
        assert resolve_match_backend("ahocorasick") in {"numba", "python"}


def test_auto_backend_falls_back_to_python_when_jit_is_disabled(monkeypatch):
    from numba import config

    monkeypatch.setattr(config, "DISABLE_JIT", True)
    assert resolve_match_backend("auto") == "python"
    monkeypatch.setattr(config, "DISABLE_JIT", False)
    assert resolve_match_backend("auto") == "numba"


def test_bed_from_starts_matches_sorted_row_construction():
    sequence = "TGGATGGAAACCATGGATccatgg"
    starts = locate_motif_starts(sequence, "tgg")
    rows = [("chr9", int(s), int(s) + 3, "TGG", 0, strand)
            for strand in "+-" for s in starts[strand]]
    want = _sort_bed(rows)
    got = bed_from_starts("chr9", "tgg", starts)
    assert got.to_dict("records") == want.to_dict("records")
    assert got.to_csv(sep="\t", index=False) == want.to_csv(sep="\t", index=False)


def test_provider_starts_for_chrom_matches_for_chrom_tables(tmp_path):
    fasta = tmp_path / "genome.fa"
    write_fasta(fasta, [("chr1", "TGGATGGAAnnttccaTGGc"), ("chr2", "ACCATGG")])
    provider = FastaMotifSiteProvider(fasta, ["TGG:0", "TGGAA:1"])
    for chrom in ("chr1", "chr2"):
        spec_starts = provider.starts_for_chrom(chrom)
        beds = provider.for_chrom(chrom)
        for starts, bed in zip(spec_starts, beds):
            for strand in "+-":
                assert starts[strand].tolist() == bed.loc[
                    bed.strand == strand, "start"
                ].tolist()
    minus = provider.starts_for_chrom("chr1", strands=("-",))
    assert all(s["+"].size == 0 for s in minus)

    plus_only = FastaMotifSiteProvider(fasta, ["TGG:0"], both_strands=False)
    assert plus_only.starts_for_chrom("chr2", strands=("-",))[0]["-"].size == 0
