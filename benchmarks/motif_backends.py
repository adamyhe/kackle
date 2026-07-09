"""Benchmark FASTA and motif matching backends.

The benchmark creates a synthetic multi-record FASTA, locates the default
kackle motifs on the last chromosome, and compares the pure-Python scanner
against the default ``pyfastx`` and ``ahocorasick-rs`` backends.
"""

from __future__ import annotations

import argparse
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

from kackle.motifs import FastaMotifSiteProvider


MOTIFS = ("TGG:0", "TGGAA:1")


def write_fasta(path: Path, records: int, length: int) -> None:
    """Write a deterministic synthetic FASTA with common short motif hits."""
    rng = np.random.default_rng(20260709)
    alphabet = np.frombuffer(b"ACGT", dtype=np.uint8)
    with path.open("wb") as handle:
        for idx in range(records):
            handle.write(f">chr{idx + 1}\n".encode())
            sequence = rng.choice(alphabet, size=length)
            for start in range(100 + idx, length - 5, 997):
                sequence[start : start + 5] = np.frombuffer(b"TGGAA", dtype=np.uint8)
            for offset in range(0, length, 80):
                handle.write(sequence[offset : offset + 80].tobytes())
                handle.write(b"\n")


def time_provider(fasta, chrom, fasta_backend, match_backend, repeats):
    """Return median runtime and hit counts for one backend combination."""
    provider = FastaMotifSiteProvider(
        fasta,
        MOTIFS,
        both_strands=True,
        fasta_backend=fasta_backend,
        match_backend=match_backend,
    )
    timings = []
    counts = None
    for _ in range(repeats):
        start = time.perf_counter()
        beds = provider.for_chrom(chrom)
        timings.append(time.perf_counter() - start)
        current_counts = tuple(len(bed) for bed in beds)
        if counts is None:
            counts = current_counts
        elif counts != current_counts:
            raise RuntimeError(
                f"Backend {fasta_backend}/{match_backend} produced inconsistent counts"
            )
    return statistics.median(timings), counts


def main() -> None:
    """Run the benchmark and print a compact timing table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=int, default=8)
    parser.add_argument("--length", type=int, default=1_000_000)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="kackle-motif-bench-") as tmpdir:
        fasta = Path(tmpdir) / "synthetic.fa"
        write_fasta(fasta, args.records, args.length)
        chrom = f"chr{args.records}"
        combinations = [
            ("pyfastx", "ahocorasick"),
            ("python", "python"),
            ("python", "ahocorasick"),
            ("pyfastx", "python"),
        ]
        results = []
        for fasta_backend, match_backend in combinations:
            elapsed, counts = time_provider(
                fasta, chrom, fasta_backend, match_backend, args.repeats
            )
            results.append((fasta_backend, match_backend, elapsed, counts))

    default_time = results[0][2]
    print("fasta_backend\tmatch_backend\tseconds\tvs_default\thits")
    for fasta_backend, match_backend, elapsed, counts in results:
        relative = default_time / elapsed
        print(
            f"{fasta_backend}\t{match_backend}\t"
            f"{elapsed:.6f}\t{relative:.2f}x\t{counts}"
        )


if __name__ == "__main__":
    main()
