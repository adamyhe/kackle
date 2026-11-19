# kackle

[![PyPI](https://img.shields.io/pypi/v/kackle)](https://pypi.org/project/kackle/)
[![Tests](https://github.com/adamyhe/kackle/actions/workflows/ci.yml/badge.svg)](https://github.com/adamyhe/kackle/actions/workflows/ci.yml)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/kackle?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/kackle)

Kmer Artifact Correction (KACkle) -- A small python utility for eliminating k-mer artifacts caused by primer mismatching.

## Installation

Requires Python 3.10 or newer (tested on 3.10-3.14).

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
whole-genome BED tables for common short motifs. It reads the FASTA through
`pyfaidx`, which writes a samtools-compatible `.fai` index next to the FASTA
on first use (uncompressed or bgzip-compressed FASTA only). Motifs are
matched with a numba kernel. A window matches `KMER:N` when every base is
A/C/G/T (any case) and at most `N` differ from the motif, so windows
containing `N` or IUPAC codes never match. `--motif-match-backend` selects
`numba` (the default, via `auto`) or `python`, a much slower reference (one
`str.find` pass per mismatch variant) that `auto` uses only when numba JIT is
disabled (`NUMBA_DISABLE_JIT=1`). Both return identical sites.

Custom motif order can be supplied with repeated `--motif KMER[:MISMATCHES]` flags.
To save the generated motif sites while using FASTA mode, add
`--out-bed6-prefix PREFIX`. This writes one BED6 file per motif pass, such as
`PREFIX.1.TGG.m0.bed6` and `PREFIX.2.TGGAA.m1.bed6`.

By default, kackle uses `--chrom-workers auto` to process multiple chromosomes
concurrently. Auto mode uses up to four chromosome workers, honors
`NUMBA_NUM_THREADS` as the total thread budget, and divides numba threads across
the chromosome workers. The CLI uses `--worker-backend process` by default so
FASTA matching, pandas filtering, and numba kernels can run in separate Python
interpreters. Set `--worker-backend thread` for lower process overhead, set
`--chrom-workers 1` for serial chromosome processing, or tune
`--numba-threads N` explicitly.

Chromosomes longer than `--chunk-size` (default 10 Mb) are split into chunks
that are corrected in parallel, so large chromosomes no longer bound the run.
Cuts are placed only where no motif site's window reaches across them, so the
output is the same as correcting whole chromosomes. `--chunk-size 0` turns
splitting off. On hg38, a run takes about 20–30 seconds with 4–16 workers and
under 1 GB per process.

A site is flagged when the signal at its artifact base (the base just before
the motif in RNA orientation) is more than `-r` (default 10) times the mean of
its neighborhood (`-s`, default 10 bp on each side). By default only that
base is replaced, with a Poisson draw from the neighborhood mean, so the
signal around the motif keeps its shape (`--resample base`). `--resample window` restores the
0.2.0 behavior: the whole window of `-t` (default 5) bp on each side is
redrawn from the flanks, which also fills in natural dips at the motif.

A site with no other reads in its neighborhood is flagged even when its
artifact base holds a single read. `--min-reads N` also requires at least `N`
reads at the artifact base (default 1, which adds nothing; it applies in
`--adapter` mode too). Whether single reads should be left alone is still
being evaluated (see the ROADMAP).

kackle processes chromosomes present in the intersection of `chrom.sizes`, both
input bigWigs, and the FASTA file when FASTA mode is used. Chromosomes absent
from any required source are skipped.

### Adapter look-alike sites (`--adapter`, experimental)

The TGG artifact comes from RT primer mispriming on stretches of real
fragments that resemble the 3' adapter: the read ends on the base just before
the look-alike, and the spike grows with how much of the adapter matches.
Instead of fixed motif passes, `--adapter` derives the sites from the
adapter sequence and screens them all in one pass:

```bash
kackle -i plus.bw -I minus.bw -f genome.fa -o out.plus.bw -O out.minus.bw -c chrom.sizes \
  --adapter TGGAATTCTCGG
```

- A position is a candidate when the RNA-sense sequence that follows matches
  the adapter's 5' end for at least 3 bases with no mismatches, or 5 with one
  (`--adapter-min-match 3,5`). The adapter may use IUPAC codes. Leading
  `N`s (such as a UMI at the start of the adapter) aren't handled yet: they
  match every base, so nearly every position would become a candidate.
- Stronger look-alikes are flagged at lower fold thresholds
  (`--adapter-tiers`, `LEN[/MAXMM]:FOLD`, default `4/0:7,8:5,10:3`): exact
  matches of 4 or more bases at 7x, any match of 8 or more at 5x, 10 or more
  at 3x, everything else at `-r`. Mismatched bases count toward the length; a
  site takes the lowest fold it qualifies for.
- The screen has two stages, both on the input: a plain screen, then a
  rescreen whose background also leaves out the artifact bases flagged in the
  first stage, so a nearby spike can't hide a site. The result doesn't depend
  on site order.
- As with motif passes, only the artifact base of a flagged site is replaced.

`--adapter` can't be combined with `-k`, `--out-bed6-prefix` or
`--resample window`. It assumes 3'-end tracks where the read end sits just
before the look-alike in RNA orientation.

### Metaplots

Generate before/after metaplots centered on motif starts:

```bash
kackle-metaplot \
  --before-pl-bw plus.bw --before-mn-bw minus.bw \
  --after-pl-bw out.plus.bw --after-mn-bw out.minus.bw \
  -f genome.fa -c chrom.sizes -o correction.metaplot
```

This writes `correction.metaplot.plus.png`, `correction.metaplot.minus.png`, and
the plotted values as `correction.metaplot.tsv`. BED6 motif sites can be
supplied with `-b sites.bed6` instead of FASTA. Chromosomes are processed in
parallel, in chunks of `--chunk-size` (default 10 Mb), with
`-w/--chrom-workers` workers (default `auto` = up to 4). A whole-genome hg38 plot
in FASTA mode takes about 50 seconds with 8 workers or 25 with 16, at about
0.25 GB per process.

## Benchmarking

To reproduce motif-backend timings:

```bash
uv run python benchmarks/motif_backends.py
```
