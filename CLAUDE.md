# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Python utility (package name `kackle`, on PyPI) that identifies and corrects k-mer artifacts in
strand-specific bigWig signal (e.g. from PCR amplification or reverse-transcription bias) at
positions given either by a precomputed BED6 file or located directly from a reference FASTA.
Source lives in `src/kackle/` as a small multi-module package; there is a real pytest suite under
`tests/` and CI in `.github/workflows/ci.yml`.

## Install / run / test

Managed with `uv`:

```bash
uv sync --all-groups        # install runtime + dev deps (pytest)
uv run kackle --help
uv run kackle-metaplot --help
uv run pytest               # full test suite
uv run pytest tests/test_computation.py::test_correct_kmers_fused_kernel_only_updates_target_window  # single test
uv run python -m compileall -q src/kackle tests benchmarks  # what CI runs before tests
uv build                    # build wheel/sdist (matches CI's build step)
```

Two console scripts are defined in `pyproject.toml`: `kackle` → `kackle.correction:run_correction`
and `kackle-metaplot` → `kackle.plotting:run_metaplot`.

```bash
# BED6 mode (precomputed motif sites)
kackle -i plus.bw -I minus.bw -b sites.bed6 -o out.plus.bw -O out.minus.bw -c chrom.sizes

# FASTA mode (locate TGG:0 then TGGAA:1 motifs directly, one chromosome at a time)
kackle -i plus.bw -I minus.bw -f genome.fa -o out.plus.bw -O out.minus.bw -c chrom.sizes

# Before/after diagnostic plots centered on motif starts
kackle-metaplot --before-pl-bw plus.bw --before-mn-bw minus.bw \
  --after-pl-bw out.plus.bw --after-mn-bw out.minus.bw -f genome.fa -c chrom.sizes -o correction.metaplot
```

Key CLI flags (`kackle`, see `parse_args()` in `correction.py`): `-s/--source` (background
sampling half-window, default 10), `-t/--target` (correction half-window, default 5),
`-r/--threshold` (fold-over-background cutoff to flag an artifact, default 10), `-w/--chrom-workers`
(concurrent chromosomes, default `auto`), `--worker-backend` (`process` default, or `thread`),
`--numba-threads` (default `auto`), `-k/--motif KMER[:MISMATCHES]` (repeatable, FASTA mode only,
defaults to `TGG:0` then `TGGAA:1`), `--out-bed6-prefix` (dump generated FASTA-mode motif sites),
`-v/--verbose` (progress bars).

`uv run python benchmarks/motif_backends.py` reproduces FASTA motif-matching backend timings.

## Architecture

### Module layout (`src/kackle/`)

- `computation.py` — numba hot-path kernels only (`artifact_mask_serial/parallel`,
  `resample_dirmult`, `resample_empirical`, `correct_kmers`). No pandas, no file I/O, no Python
  objects beyond NumPy arrays — these are `@njit` compiled and must stay numba-compatible.
- `motifs.py` — motif spec parsing (`KMER[:MISMATCHES]`), Hamming-neighborhood expansion
  (`mismatch_variants`), sequence matching (`ahocorasick-rs` by default, plain-Python fallback),
  and `FastaMotifSiteProvider`, which locates motif BED6 rows one chromosome at a time via indexed
  `pyfastx` access instead of materializing whole-genome motif tables.
- `bigwig.py` — the only boundary between in-memory dense arrays/sparse DataFrames and on-disk
  bigWig files, via `pybigtools`.
- `cli_common.py` — shared argparse formatting, BED6/chrom-sizes readers, and
  `intersect_ordered()` for reconciling chromosome sets across inputs.
- `correction.py` — the `kmer_resample()` pipeline and the `kackle` CLI entry point
  (`run_correction`).
- `plotting.py` — before/after metaplot generation and the `kackle-metaplot` CLI entry point
  (`run_metaplot`).

### Correction pipeline (`kmer_resample()` in `correction.py`)

Plus- and minus-strand bigWigs are corrected **independently** (two full passes from
`run_correction`). For each strand:

1. Resolve motif sites either from a precomputed BED6 (`read_bed6`, loaded once, filtered to
   strand and split into `(chrom, start-or-end)` anchor arrays via `motif_anchor_arrays()`) or from
   a `FastaMotifSiteProvider`, which locates sites per chromosome on demand. Plus-strand anchors use
   the BED `start` column; minus-strand anchors use `end`.
2. Chromosomes are corrected via `_correct_chromosome()`, run either serially or fanned out across
   `chrom_workers` (thread or process pool, chosen by `--worker-backend`). Each worker:
   - Opens its own `pybigtools` handle and pulls dense per-base values for the whole chromosome
     (`bw.values(...)`), converting minus-strand values to absolute counts.
   - Applies motif passes **sequentially** — each entry in `motif_specs`/`motif_beds` is one full
     `correct_kmers()` call over the array, so e.g. the default `TGG:0` pass runs and its output
     feeds the subsequent `TGGAA:1` pass. This ordering is deliberate (short exact matches corrected
     before longer fuzzy ones) and must be preserved.
   - `correct_kmers()` (`computation.py`, `njit(parallel=True)`) is a fused screen+resample pass:
     for each candidate anchor it builds a `source`-sized neighborhood (indexed so the "artifact"
     position is consistently at the strand-appropriate offset), flags it if signal exceeds
     `threshold * local_background`, and for flagged positions only, redraws the central
     `target`-sized window with a Dirichlet-multinomial resample of the combined flanking counts —
     conserving local depth while smoothing the spike.
   - Output is converted to a sparse `(chrom, start, end, value)` DataFrame, dropping zero rows.
3. Chromosome results are concatenated and passed to `write_bigwig()`.

`resample_empirical()` (IQR-based resampling from the local empirical distribution) exists as an
alternate strategy but is **not** wired into `correct_kmers()` — swap it in there if changing the
resampling method. `artifact_mask_serial()` is a readable non-parallel reference kept for
correctness comparisons against `artifact_mask_parallel()`, not used in the main pipeline.

### Chromosome list resolution

`run_correction()` intersects chromosomes across chrom.sizes, both input bigWigs, and (in FASTA
mode) the FASTA file via `intersect_ordered()` — chromosomes missing from any required source are
silently skipped, not errored on.

### Parallelism knobs

Two independent levels of parallelism interact and must not oversubscribe CPUs:
- **Cross-chromosome**: `chrom_workers` (thread- or process-pool). `auto` resolves to
  `min(chrom_count, thread_budget(), 4)`.
- **Within-chromosome (numba)**: `numba_threads`, set per worker via `numba.set_num_threads()`.
  `auto` divides `thread_budget()` (honors `NUMBA_NUM_THREADS` env var, else `os.cpu_count()`)
  across `chrom_workers` so the two levels don't multiply out past available CPUs.

`--worker-backend process` is the CLI default (separate interpreters help with FASTA/pandas/numba
mixed workloads); `thread` avoids pickling large in-memory BED tables and is used automatically for
library callers passing `motif_beds` directly.

### Numba constraints

All `computation.py` functions are `@njit` (mostly `parallel=True`, all `cache=True`); edits there
must remain numba-compatible — no pandas, limited stdlib, explicit dtypes, no Python objects beyond
NumPy arrays/scalars.
