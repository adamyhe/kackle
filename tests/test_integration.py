import sys

import pandas as pd
import pybigtools

from kackle.bigwig import write_bigwig
from kackle.correction import run_correction


def write_text(path, text):
    path.write_text(text)
    return path


def read_values(path, chrom, length):
    bw = pybigtools.open(path)
    try:
        return bw.values(chrom, 0, length, fillna=0.0).tolist()
    finally:
        bw.close()


def test_cli_fasta_mode_generates_motifs_and_writes_bigwigs(tmp_path, monkeypatch):
    chrom_sizes = [("chr1", 40)]
    plus_in = tmp_path / "plus.bw"
    minus_in = tmp_path / "minus.bw"
    plus_out = tmp_path / "out.plus.bw"
    minus_out = tmp_path / "out.minus.bw"
    bed_prefix = tmp_path / "generated"
    fasta = write_text(tmp_path / "genome.fa", ">chr1\nAAAAATGGATGGAAAAATTCCATGGAAAAAAA\n")
    chrom_sizes_file = write_text(tmp_path / "chrom.sizes", "chr1\t40\n")
    signal = pd.DataFrame(
        {
            "chrom": ["chr1"] * 6,
            "start": [5, 6, 7, 10, 11, 12],
            "end": [6, 7, 8, 11, 12, 13],
            "value": [1.0, 1.0, 1.0, 100.0, 1.0, 1.0],
        }
    )
    write_bigwig(signal, plus_in, chrom_sizes)
    write_bigwig(signal, minus_in, chrom_sizes)
    argv = [
        "kackle",
        "-i",
        str(plus_in),
        "-I",
        str(minus_in),
        "-f",
        str(fasta),
        "-o",
        str(plus_out),
        "-O",
        str(minus_out),
        "-c",
        str(chrom_sizes_file),
        "--motif",
        "TGG:0",
        "--motif",
        "TGGAA:1",
        "--source",
        "2",
        "--target",
        "1",
        "--threshold",
        "1000",
        "--out-bed6-prefix",
        str(bed_prefix),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    run_correction()

    assert plus_out.exists()
    assert minus_out.exists()
    assert (tmp_path / "generated.1.TGG.m0.bed6").exists()
    assert (tmp_path / "generated.2.TGGAA.m1.bed6").exists()
    assert (tmp_path / "generated.1.TGG.m0.bed6").read_text().splitlines() == [
        "chr1\t5\t8\tTGG\t0\t+",
        "chr1\t9\t12\tTGG\t0\t+",
        "chr1\t22\t25\tTGG\t0\t+",
        "chr1\t19\t22\tTGG\t0\t-",
    ]
    assert read_values(plus_out, "chr1", 40) == read_values(plus_in, "chr1", 40)
    assert read_values(minus_out, "chr1", 40) == read_values(minus_in, "chr1", 40)
