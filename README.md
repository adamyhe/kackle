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
uv tool install kackle
kackle --help
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

## Benchmarking

To reproduce motif-backend timings:

```bash
uv run python benchmarks/motif_backends.py
```
