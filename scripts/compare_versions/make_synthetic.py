# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "figwig>=0.3.1"]
# ///
"""Generate a small synthetic genome and stranded bigWigs with planted artifacts.

Spikes are planted at the strand-specific artifact position of TGG motif
sites (plus: base before the motif start; minus: the motif end). Besides
isolated spikes, "hot" regions spike every motif site, which produces
flagged windows that overlap one another. Overlapping windows are where
v0.1.0 and v0.2.0 are expected to diverge.
"""

import argparse
import re
from pathlib import Path

import figwig
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("outdir", type=Path)
    parser.add_argument(
        "--chrom-lengths",
        default="2000000,1000003",
        help="Comma-separated lengths; keep them distinct (default: %(default)s).",
    )
    parser.add_argument("--background", type=float, default=0.3, help="Poisson mean.")
    parser.add_argument("--spike-rate", type=float, default=0.03)
    parser.add_argument("--hot-regions", type=int, default=20)
    parser.add_argument("--hot-width", type=int, default=400)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    lengths = [int(x) for x in args.chrom_lengths.split(",")]
    if len(set(lengths)) != len(lengths):
        parser.error("chromosome lengths must be distinct")
    chroms = [(f"chr{i + 1}", length) for i, length in enumerate(lengths)]
    args.outdir.mkdir(parents=True, exist_ok=True)

    plus = {}
    minus = {}
    with open(args.outdir / "genome.fa", "w") as fasta:
        for chrom, length in chroms:
            seq = rng.choice(np.frombuffer(b"ACGT", np.uint8), length).tobytes().decode()
            fasta.write(f">{chrom}\n")
            fasta.writelines(seq[i : i + 60] + "\n" for i in range(0, length, 60))

            hot = np.zeros(length, dtype=bool)
            for center in rng.integers(args.hot_width, length - args.hot_width, args.hot_regions):
                hot[center - args.hot_width // 2 : center + args.hot_width // 2] = True

            for strand, pattern, store in (("+", "TGG", plus), ("-", "CCA", minus)):
                signal = rng.poisson(args.background, length).astype(np.int64)
                starts = np.array([m.start() for m in re.finditer(f"(?={pattern})", seq)])
                anchor = starts - 1 if strand == "+" else starts + len(pattern)
                anchor = anchor[(anchor >= 0) & (anchor < length)]
                spike = (rng.random(anchor.size) < args.spike_rate) | hot[anchor]
                signal[anchor[spike]] += rng.integers(20, 200, int(spike.sum()))
                store[chrom] = signal

    (args.outdir / "chrom.sizes").write_text(
        "".join(f"{chrom}\t{length}\n" for chrom, length in chroms)
    )
    for name, store, sign in (("plus.bw", plus, 1.0), ("minus.bw", minus, -1.0)):
        with figwig.BigWigWriter(str(args.outdir / name), chroms) as writer:
            for chrom, _ in chroms:
                pos = np.flatnonzero(store[chrom])
                writer.write(
                    chrom, pos, sign * store[chrom][pos].astype(np.float64), ends=pos + 1
                )
    print(f"Wrote synthetic inputs to {args.outdir}")


if __name__ == "__main__":
    main()
