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
(concurrent chromosome chunks, default `auto`), `--chunk-size` (split chromosomes longer than this
many bases, default 10 Mb; `0` = whole chromosomes; `kackle-metaplot` has it too),
`--worker-backend` (`process` default, spawned, or `thread`),
`--numba-threads` (default `auto`), `-k/--motif KMER[:MISMATCHES]` (repeatable, FASTA mode only,
defaults to `TGG:0` then `TGGAA:1`), `--out-bed6-prefix` (dump generated FASTA-mode motif sites),
`--resample` (`base`, the default since 0.3.0, or `window`), `--adapter SEQ` (FASTA mode, replaces
`-k`; needs `--resample base`) with `--adapter-min-match` (default `3,5`) and `--adapter-tiers`
(`LEN[/MAXMM]:FOLD`, default `4/0:7,8:5,10:3`), `-v/--verbose` (progress bars).

`uv run python benchmarks/motif_backends.py` reproduces FASTA motif-matching backend timings.

`scripts/compare_versions/` compares two git refs' outputs on the same inputs (`compare.sh -a REF -b
REF`, `-x` for exact mode: no JIT, RNG keyed per anchor, small inputs only); see its README. It's the
main regression check for changes to the correction pipeline: anchors must match, nothing may change
outside motif windows, and isolated windows must keep their sums. It compares in window mode
(`--resample window`), the only mode it instruments.

## Architecture

### Module layout (`src/kackle/`)

- `computation.py` — numba hot-path kernels only (`artifact_mask_serial/parallel`,
  `resample_dirmult`, `resample_empirical`, `correct_kmers`, the base-only kernels
  `correct_artifact_bases` and `correct_adapter_sites`, the motif matcher `motif_window_starts`, and
  the adapter matcher `adapter_match_lengths`). No pandas, no file I/O, no Python
  objects beyond NumPy arrays — these are `@njit` compiled and must stay numba-compatible.
- `motifs.py` — motif spec parsing (`KMER[:MISMATCHES]`), `locate_motif_starts` (sorted int64
  start arrays per strand; backends `numba`, the default, and `python`, which runs `str.find`
  over `mismatch_variants`, is the test reference, and is what `auto` picks when numba JIT is
  disabled), and
  `FastaMotifSiteProvider`, which locates sites one chromosome at a time via indexed `pyfaidx`
  access. `starts_for_chrom()` returns arrays and is what both CLIs use. `for_chrom()` and the
  `locate_motif*` helpers build BED6 tables from them (`bed_from_starts`) only when a table is
  wanted, e.g. `--out-bed6-prefix`. A window matches when it is all A/C/G/T (any case) and within
  the Hamming distance. Keep the backends identical; that's what the tests check. All FASTA reading goes
  through `open_fasta()` (pyfaidx, `.fai` index); there is no hand-rolled FASTA parser.
- `adapter.py` — `--adapter` mode: IUPAC adapter bitmasks, `parse_min_match` / `parse_tiers`, and
  `AdapterSiteProvider`. It has the same `starts_for_chrom` / `starts_for_region` interface as
  `FastaMotifSiteProvider` and returns one `{strand: starts}` dict per threshold tier, with one
  pseudo-`MotifSpec` per tier (the full adapter, so minus anchors are `start + len(adapter)`), so
  chunk planning and region reads work unchanged. Its `pass_thresholds` attribute tells the pipeline
  to screen all tiers together in one pass. A site goes to the qualifying tier with the lowest fold.
  Not supported by `kmer_resample`.
- `bigwig.py` — the only boundary between in-memory dense arrays/sparse DataFrames and on-disk
  bigWig files, via `figwig` (`BigWigReader`, and `open_bigwig_writer` / `write_chrom_intervals`
  for streaming per-chromosome writes, or `write_bigwig` for a whole DataFrame).
- `cli_common.py` — shared argparse formatting, BED6/chrom-sizes readers, and
  `intersect_ordered()` for reconciling chromosome sets across inputs.
- `correction.py` — the `correct_bigwigs()` pipeline (both strands, streamed to bigWig) used by the
  `kackle` CLI entry point (`run_correction`), and `kmer_resample()`, the one-strand library API
  that returns a sparse DataFrame. Both share `_correct_strand_signal` and `_run_chromosomes`.
- `plotting.py` — before/after metaplot generation and the `kackle-metaplot` CLI entry point
  (`run_metaplot`). The CLI path (`metaplot_profiles` → `chrom_window_sums`) works one
  chromosome, or `--chunk-size` chunk, at a time. A chunk owns the sites that start in it and
  reads its tracks with `flank` bases of margin, so chunk sums add up exactly. It locates sites
  per chunk inside the workers, reads each track
  densely once, and sums `sig[p] * n_sites[p - k]` over the track's *nonzero* bases instead of
  reading a window per site. It never builds a genome-wide site table. `metaplot_profile` (windowed
  reads, batched) is kept as the reference implementation the tests compare against.

### Correction pipeline (`correct_bigwigs()` in `correction.py`)

One pass over chromosomes corrects both strands: each chromosome task locates (or looks up) its
motif sites for both strands once, then corrects plus and minus **independently**, one after the
other, so only one dense strand array is alive at a time.

**Chunking.** Chromosomes longer than `chunk_size` are split by `plan_chunks()` at cuts with no
anchor (any pass, either strand) within `reach = max(source, target)`. Every site reads and writes
only `[anchor - reach, anchor + reach)`, and clusters link only below `source + target <= 2 *
reach`, so a chunk's correction is exactly the restriction of the whole-chromosome correction (up to
random draws). Each chunk task locates only its own sites (`starts_for_region`) or gets sliced BED
anchors, reads only its own signal, and the parent reassembles chunks per chromosome before writing.
Don't weaken the cut rule: `tests/test_chunking.py` checks chunked == whole byte for byte with a
deterministic stand-in for the resampler. The comparison harness keys records on signal length, so
it needs `--chunk-size 0`.

For each strand (of each chunk):

1. Resolve motif sites either from a precomputed BED6 (`read_bed6`, loaded once and grouped once
   into per-chromosome, per-strand anchor arrays by `bed_anchor_index()`) or from
   a `FastaMotifSiteProvider`, which locates sites per chromosome on demand. Plus-strand anchors use
   the BED `start` column; minus-strand anchors use `end`.
2. Chromosomes are corrected via `_correct_chromosome_both_strands()` (`_correct_chromosome()` for
   `kmer_resample`), run either serially or fanned out across
   `chrom_workers` (thread or process pool, chosen by `--worker-backend`). Each worker:
   - Opens a `figwig.BigWigReader` and reads the whole chromosome as one dense window
     (`read_chrom_values()` in `bigwig.py`), converting minus-strand values to absolute counts.
   - Applies motif passes **sequentially** — each entry in `motif_specs`/`motif_beds` is one full
     kernel call over the array (`correct_artifact_bases()` by default, `correct_kmers()` with
     `resample="window"`), so e.g. the default `TGG:0` pass runs and its output
     feeds the subsequent `TGGAA:1` pass. This ordering is deliberate (short exact matches corrected
     before longer fuzzy ones) and must be preserved.
   - Default (`resample="base"`): `correct_artifact_bases()` screens every anchor with
     `artifact_mask_parallel` (below), then replaces only each flagged artifact base with a Poisson
     draw from the mean of its `[anchor - source, anchor + source)` neighborhood without any
     flagged artifact base. Each site writes one base, so there are no races; `target` is unused.
   - `resample="window"` (0.2.0 behavior): `correct_kmers()` (`computation.py`,
     `njit(parallel=True)`) runs in three steps:
     1. **Screen** every candidate anchor against the pass's input (`artifact_mask_parallel`):
        flag it if the strand-appropriate artifact position exceeds
        `threshold * local_background` over its `[anchor - source, anchor + source)` neighborhood.
     2. **Cluster** sorted flagged anchors closer than `source + target` (`_flagged_clusters`).
        At that distance one site's flank would overlap another's target window.
     3. **Resample** each cluster once (prange over clusters): redraw
        `[first - target, last + target)` with a Dirichlet-multinomial whose total is the sum of
        the `source - target` flank bases just outside the span, scaled by
        `span_width / (2 * target)`. An isolated site is a cluster of one (scale factor 1).
        Clusters are disjoint, so there are no write races and no spike leaks between neighbors.
     Only target spans are modified. Don't reintroduce per-site write-back of overlapping
     windows (v0.1.0 wrote full source windows and v0.2.0 wrote overlapping target windows, and
     both raced and leaked spikes).
   - With `--adapter` (the provider has `pass_thresholds`), the tiers' anchor arrays are concatenated
     with per-site thresholds and corrected in **one** `correct_adapter_sites()` call: a plain screen,
     then a rescreen whose background also leaves out stage-1-flagged artifact bases; a site flagged
     in either stage gets a Poisson draw from the stage-2 mean at its artifact base only. Both stages
     read the input, so the result doesn't depend on site order. Every read stays within `source`
     of an anchor, so the chunk cut rule still makes chunked output exact.
   - Output is converted to a sparse `(chrom, start, end, value)` DataFrame, dropping zero rows.
3. Each finished chromosome's sparse `(positions, values)` arrays (int32, nonzero bases only) are
   written to both output bigWigs as soon as every earlier chromosome in `chrom_sizes` order is done
   (`_run_chromosomes` buffers out-of-order results). No genome-wide table is built.
   `kmer_resample()` instead collects one strand into a DataFrame.

Pipeline tests that swap `cli.correct_kmers` for deterministic stand-ins pin `resample="window"`
with an autouse fixture (`tests/test_chunking.py`, `tests/test_cli.py`); base mode has its own
chunked == whole tests. The comparison harness adds `--resample window` for releases that have the
flag, because it only instruments `correct_kmers`.

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
- **Cross-chromosome**: `chrom_workers` (thread- or process-pool) over chromosome chunks.
  `auto` resolves to `min(chrom_count, thread_budget(), 4)`. Process pools use the `spawn` context:
  the parent holds figwig writer threads, and forking a multi-threaded process crashes workers.
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
