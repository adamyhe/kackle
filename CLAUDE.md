# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-module Python utility that identifies and corrects k-mer artifacts in strand-specific
bigWig signal (e.g. from PCR amplification or reverse-transcription bias) at positions given by a
BED6 file. The entire implementation lives in `src/kackle.py`; there is no
package `__init__.py`, no test suite, and no linter/formatter config in this repo.

## Install / run

```bash
pip install git+https://github.com/adamyhe/kackle.git
```

This installs a console script `kackle` (defined in `pyproject.toml` under
`[project.scripts]`, pointing at `src.kackle:wrapper`). Run it directly to test
changes:

```bash
kackle -i plus.bw -I minus.bw -b sites.bed6 -o out_plus.bw -O out_minus.bw -c chrom.sizes
```

Key CLI flags (see `wrapper()`): `-s/--source` (background sampling window, default 10),
`-t/--target` (correction window, default 5), `-r/--threshold` (fold-over-background cutoff to
flag an artifact, default 10), `-v/--verbose` (progress bars).

There is no build/lint/test command configured — verify changes by running the script against a
small bigWig + BED6 fixture and inspecting the output bigWig (e.g. with `pyBigWig` or `bigWigToBedGraph`).

## Architecture

The pipeline processes plus- and minus-strand bigWigs **independently**, each through the same
per-chromosome flow in `kmer_resample()`:

1. **Load the BED6** of candidate artifact positions once, filter to the current strand, and
   collapse each interval to the single strand-specific 1bp position of interest (kmer start for
   `+`, kmer end for `-`).
2. **Per chromosome** (iterating `chr1`...`chr22`, `chrX`, `chrY`):
   - Extract per-base signal via `fast_values()` — a numba-parallel replacement for
     `pyBigWig.values()`, which is used because `pyBigWig.intervals()` + `fast_values` is
     dramatically faster than calling `.values()` directly across a whole chromosome. Positions
     with no signal are mapped to `0`, which is the read-coverage convention this tool assumes.
   - Call `correct_kmers()` (numba `njit(parallel=True)`) which, for each candidate position:
     a. builds a `source`-sized neighborhood around the position (reversed for `-` strand so the
        "artifact" position is always locally at the same index),
     b. flags it as an artifact if the signal there exceeds `threshold * local_background`,
     c. for flagged positions only, replaces the central `target`-sized window using
        `resample_dirmult()` — draws a Dirichlet mixing distribution, then a multinomial split of
        the *combined source-region read count* across the target window, so total local depth is
        conserved but the artifactual spike is smoothed away.
   - Results are converted to a long-format `(chrom, start, end, value)` DataFrame, dropping
     zero-value rows (bigWig only stores nonzero intervals).
3. Per-chromosome DataFrames are concatenated and returned to `wrapper()`, which writes them back
   out via `write_bigWig()` (`pyBigWig.addEntries`).

`resample_empirical()` is an alternate resampling strategy (samples from the local empirical IQR
distribution instead of a Dirichlet-multinomial draw) that exists in the module but is **not**
currently wired into `correct_kmers()` — if changing the resampling strategy, that's the function
to swap in.

`_get_chrom_values()` is intentionally unused/slow — kept only as a readable reference
implementation of what `fast_values()` optimizes, and for unit testing equivalence.

All hot-path functions (`fast_values`, `resample_dirmult`, `resample_empirical`, `correct_kmers`)
are numba `@njit(parallel=True)`; edits to them must remain numba-compatible (no pandas, limited
stdlib, explicit dtypes) since numba compiles them ahead of use.
