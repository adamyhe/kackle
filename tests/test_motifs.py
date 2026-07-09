import pytest

from kackle.motifs import (
    FastaMotifSiteProvider,
    locate_motif,
    locate_motif_in_sequence,
    locate_motif_specs,
    mismatch_variants,
    parse_motif_spec,
    reverse_complement,
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


def test_reverse_complement():
    assert reverse_complement("TGGAA") == "TTCCA"


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


def test_locate_motif_in_sequence_ahocorasick_matches_python():
    sequence = "TGGATGGAAACCATGGAT"

    python_bed = locate_motif_in_sequence(
        "chr1",
        sequence,
        "TGGAA",
        max_mismatches=1,
        both_strands=True,
        match_backend="python",
    )
    aho_bed = locate_motif_in_sequence(
        "chr1",
        sequence,
        "TGGAA",
        max_mismatches=1,
        both_strands=True,
        match_backend="ahocorasick",
    )

    assert aho_bed.to_dict("records") == python_bed.to_dict("records")


def test_fasta_motif_site_provider_locates_one_chromosome_at_a_time(tmp_path):
    fasta = tmp_path / "genome.fa"
    write_fasta(fasta, [("chr1", "TGGATGGAA"), ("chr2", "ACCATGG")])
    provider = FastaMotifSiteProvider(
        fasta,
        ["TGG:0", "TGGAA:1"],
        both_strands=True,
        fasta_backend="python",
        match_backend="python",
    )

    tgg_bed, tggaa_bed = provider.for_chrom("chr2")

    assert tgg_bed.chrom.unique().tolist() == ["chr2"]
    assert tgg_bed[["start", "strand"]].to_records(index=False).tolist() == [
        (4, "+"),
        (1, "-"),
    ]
    assert tggaa_bed.empty


def test_fasta_motif_site_provider_pyfastx_matches_python(tmp_path):
    fasta = tmp_path / "genome.fa"
    write_fasta(fasta, [("chr1", "TGGATGGAA"), ("chr2", "ACCATGG")])
    python_provider = FastaMotifSiteProvider(
        fasta,
        ["TGG:0", "TGGAA:1"],
        both_strands=True,
        fasta_backend="python",
        match_backend="python",
    )
    pyfastx_provider = FastaMotifSiteProvider(
        fasta,
        ["TGG:0", "TGGAA:1"],
        both_strands=True,
        fasta_backend="pyfastx",
        match_backend="ahocorasick",
    )

    python_beds = python_provider.for_chrom("chr1")
    pyfastx_beds = pyfastx_provider.for_chrom("chr1")

    assert [bed.to_dict("records") for bed in pyfastx_beds] == [
        bed.to_dict("records") for bed in python_beds
    ]


def test_locate_motif_specs_preserves_correction_order(tmp_path):
    fasta = tmp_path / "genome.fa"
    write_fasta(fasta, [("chr1", "TGGATGGAA")])

    tgg_bed, tggaa_bed = locate_motif_specs(fasta, ["TGG:0", "TGGAA:1"])

    assert tgg_bed.name.unique().tolist() == ["TGG"]
    assert tggaa_bed.name.unique().tolist() == ["TGGAA"]
