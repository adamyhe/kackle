# kackle

[![PyPI](https://img.shields.io/pypi/v/kackle)](https://pypi.org/project/kackle/)
[![Tests](https://github.com/adamyhe/touche/actions/workflows/ci.yml/badge.svg)](https://github.com/adamyhe/kackle/actions/workflows/ci.yml)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/kackle?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/kackle)

Kmer Artifact Correction (KACkle) -- A small python utility for eliminating k-mer artifacts caused by primer mismatching.

## Installation

Install with `pip`:

```bash
pip install kackle
kackle --help
```

Or with `uv`:

```bash
uv add kackle
uv run kackle --help
```

## Execution

Run with precomputed BED6 motif sites:

```bash
kackle -i plus.bw -I minus.bw -b sites.bed6 -o out.plus.bw -O out.minus.bw -c chrom.sizes
```

Or let kackle locate the original artifact motifs directly from FASTA. By default,
FASTA mode corrects exact `TGG` matches first, then `TGGAA` matches with up to one
mismatch:

```bash
kackle -i plus.bw -I minus.bw -f genome.fa -o out.plus.bw -O out.minus.bw -c chrom.sizes
```

FASTA mode locates motif sites one chromosome at a time to avoid materializing
whole-genome BED tables for common short motifs. It uses indexed `pyfastx`
FASTA access and `ahocorasick-rs` multi-pattern matching by default:

```bash
kackle -i plus.bw -I minus.bw -f genome.fa \
  --fasta-backend pyfastx --motif-match-backend ahocorasick \
  -o out.plus.bw -O out.minus.bw -c chrom.sizes
```

Custom motif order can be supplied with repeated `--motif KMER[:MISMATCHES]` flags.
To save the generated motif sites while using FASTA mode, add
`--out-bed6-prefix PREFIX`. This writes one BED6 file per motif pass, such as
`PREFIX.1.TGG.m0.bed6` and `PREFIX.2.TGGAA.m1.bed6`.

By default, kackle uses `--chrom-workers auto` to process multiple chromosomes
concurrently. Auto mode uses up to four chromosome workers, honors
`NUMBA_NUM_THREADS` as the total thread budget, and divides numba threads across
the chromosome workers. Set `--chrom-workers 1` for serial chromosome
processing, or tune `--numba-threads N` explicitly.

kackle processes chromosomes present in the intersection of `chrom.sizes`, both
input bigWigs, and the FASTA file when FASTA mode is used. Chromosomes absent
from any required source are skipped.

Generate before/after metaplots centered on motif starts:

```bash
kackle-metaplot \
  --before-pl-bw plus.bw --before-mn-bw minus.bw \
  --after-pl-bw out.plus.bw --after-mn-bw out.minus.bw \
  -f genome.fa -c chrom.sizes -o correction.metaplot
```

This writes `correction.metaplot.plus.png` and `correction.metaplot.minus.png`.
BED6 motif sites can be supplied with `-b sites.bed6` instead of FASTA.

## Benchmarking

To reproduce motif-backend timings:

```bash
uv run python benchmarks/motif_backends.py
```
