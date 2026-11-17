"""Benchmark motif matching backends.

The benchmark creates a synthetic multi-record FASTA, locates the default
kackle motifs on the last chromosome with ``FastaMotifSiteProvider.starts_for_chrom``,
and compares the default numba matcher with the pure-Python reference (both
read FASTA via pyfaidx). Each backend gets one
untimed warm-up call, so numba compilation isn't counted, and every backend
must return identical starts.
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


def time_provider(fasta, chrom, match_backend, repeats):
    """Return median runtime and hit counts for one backend combination."""
    provider = FastaMotifSiteProvider(
        fasta,
        MOTIFS,
        both_strands=True,
        match_backend=match_backend,
    )
    reference = provider.starts_for_chrom(chrom)  # warm-up (numba compile)
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        spec_starts = provider.starts_for_chrom(chrom)
        timings.append(time.perf_counter() - start)
        for got, want in zip(spec_starts, reference):
            for strand in ("+", "-"):
                if not np.array_equal(got[strand], want[strand]):
                    raise RuntimeError(f"Backend {match_backend} is not deterministic")
    return statistics.median(timings), reference


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
        results = []
        for match_backend in ("numba", "python"):
            elapsed, spec_starts = time_provider(
                fasta, chrom, match_backend, args.repeats
            )
            results.append((match_backend, elapsed, spec_starts))

    default_time = results[0][1]
    reference = results[0][2]
    print("match_backend\tseconds\tvs_default\thits\tidentical")
    for match_backend, elapsed, spec_starts in results:
        relative = default_time / elapsed
        hits = tuple(int(s["+"].size + s["-"].size) for s in spec_starts)
        identical = all(
            np.array_equal(a[strand], b[strand])
            for a, b in zip(spec_starts, reference)
            for strand in ("+", "-")
        )
        print(f"{match_backend}\t{elapsed:.6f}\t{relative:.2f}x\t{hits}\t{identical}")


if __name__ == "__main__":
    main()
