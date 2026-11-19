# Changelog

Notable changes to kackle. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/) (pre-1.0, so minor versions may
change behavior). Planned work and known issues live in [ROADMAP.md](ROADMAP.md).

## [Unreleased]

### Added

- Support for Python 3.14. No compiled bigWig or FASTA dependencies remain
  ([#4](https://github.com/adamyhe/kackle/pull/4)).
- `figwig[fast]` extra, which adds the libdeflate compressor for bigWig
  writing ([#4](https://github.com/adamyhe/kackle/pull/4)).
- `kackle-metaplot -w/--chrom-workers` and `--worker-backend`, and a
  `PREFIX.tsv` of the plotted values next to the PNGs.
- **`--resample base`** (experimental): replace only a flagged site's
  artifact base, with a Poisson draw from the mean of its neighborhood
  without the other flagged artifact bases, instead of redrawing the whole
  target window. It keeps the natural signal dips around the motif that
  window mode flattens. The default stays `window`
  ([#14](https://github.com/adamyhe/kackle/pull/14)).
- **`--adapter SEQ`** (experimental, FASTA mode): derive candidate sites from
  the 3' adapter sequence instead of fixed motif passes, and screen them all
  in one pass with base-only resampling. Candidates match the adapter's 5'
  end (`--adapter-min-match`, default `3,5`). Stronger matches get lower
  thresholds (`--adapter-tiers LEN[/MAXMM]:FOLD`, default `4/0:7,8:5,10:3`).
  The screen rescreens each site without the artifact bases flagged in a
  first, plain stage, so it is order-independent. On 6 PRO-seq samples it
  matched the two-pass `TGG:0` → `TGGAA:1` base-mode correction in one pass
  (about 15–60 s per sample instead of 1:35–1:54), and it removed about 60% of
  the artifact left at exact TGGA/TGGAA sites without changing other site
  classes ([#14](https://github.com/adamyhe/kackle/pull/14)).

### Changed

- **Large chromosomes are split across workers** (`--chunk-size`, default
  10 Mb, in `kackle` and `kackle-metaplot`; `0` turns it off). `kackle` cuts
  only where no anchor of any motif pass or strand lies within
  `max(source, target)` bases, so chunks never interact and the output is the
  same as for whole chromosomes. Each chunk locates its own sites and reads
  only its own signal. On hg38 (PROseq_2), peak memory per process fell from
  2.1–4.2 GB to about 0.8 GB for `kackle`, and from 1.8 to 0.23 GB for
  `kackle-metaplot`, whose 16-worker run now takes 25 s instead of 51.
  `FastaMotifSiteProvider` gains `starts_for_region`.
- **`kackle` corrects both strands in one pass and streams the output.** Each
  chromosome task locates its motif sites for both strands once, then
  corrects plus and minus in turn, and each finished chromosome is written to
  both bigWigs as soon as the chromosomes before it are done. There is no
  longer a genome-wide per-base DataFrame per strand, nor a second worker pool
  for the minus strand. BED6 mode groups the BED once instead of re-filtering
  it for every chromosome. Output is unchanged. The new `correct_bigwigs`
  function is the CLI's pipeline; `kmer_resample` (one strand, returns a
  DataFrame) is unchanged for library callers. On hg38 with 4 workers, a
  run fell from 38.6 to 29.9 s and its peak from 3.4 to 2.1 GB.
- **Motif location returns start arrays and uses a numba kernel.**
  `FastaMotifSiteProvider.starts_for_chrom` and the new `locate_motif_starts`
  return sorted per-strand start arrays from a numba Hamming-distance kernel,
  instead of building a sorted pandas BED6 table for every motif. `kackle`
  locates only the strand each pass corrects. BED6 tables are built only for
  `--out-bed6-prefix` and the `locate_motif*` / `for_chrom` helpers, with the
  same contents as before. Sites are identical to 0.2.0. On hg38, locating
  all sites fell from 22 minutes and 9.1 GB to 39 s and 0.8 GB; a `kackle`
  run with 4 workers from 40 to 4 minutes (37.7 to 10.4 GB peak), and a
  whole-genome `kackle-metaplot` from 5 minutes to 1 with 8 workers (6.9 to
  1.8 GB per worker).
  `--motif-match-backend` now takes `auto` (default: numba, or `python`
  when numba JIT is disabled), `numba` or `python`.
- **`kackle-metaplot` runs at genome scale.** It now works one chromosome at
  a time: sites are located per chromosome (in parallel), each track is read
  densely once, and the profile is summed over the track's nonzero bases
  instead of reading a 101-base window per site. No genome-wide site table is
  built. Output is unchanged (matches the windowed implementation to
  floating-point rounding). A whole-genome hg38 plot (about 115M sites per
  strand, pooled over both motifs) takes about 8 minutes with 4 workers or 5
  with 8, with about 7 GB peak per worker. 0.2.0 took 40–60 minutes.
- **Lower `kackle-metaplot` memory per worker.** Motif tables are dropped as
  soon as the start arrays are extracted, tracks are read, summed and freed
  one at a time, and the per-base site-count vector is uint8 instead of
  int32. On hg38, peak RSS per worker fell from 11.2 GB to 6.9 GB (34.8 GB
  for the whole run at `-w 8`), with byte-identical output.
- **bigWig I/O uses figwig instead of pybigtools**
  ([#4](https://github.com/adamyhe/kackle/pull/4)). figwig's reader threads are
  capped at the per-worker numba thread count. `write_bigwig` writes one chromosome at a
  time. Corrected output is unchanged: on 6 real PRO-seq samples it matched
  0.2.0 within resampling noise.
- **FASTA access uses pyfaidx instead of pyfastx**
  ([#4](https://github.com/adamyhe/kackle/pull/4)). Chromosome names come
  from the `.fai` index, which is built once in the parent process before
  chromosome workers start. FASTA input must be uncompressed or
  bgzip-compressed (plain gzip isn't supported), and pyfaidx writes a `.fai`
  next to the FASTA on first use.
- **Nearby flagged sites are now corrected together** (`correct_kmers`).
  Flagged anchors closer than `source + target` are merged into a cluster, and
  each cluster's `[first - target, last + target)` span is redrawn once from
  the flanks just outside it. The redrawn total is the flank sum scaled by
  `span_width / (2 * target)`. Isolated sites are unchanged. Outputs differ
  from 0.2.0 only where flagged sites are close together. On 6 real PRO-seq
  samples, clustered-window totals fell 0.8–1.5% and genome-wide signal fell
  0.16–0.30 percentage points relative to 0.2.0.

### Removed

- `--fasta-backend` on `kackle` and `kackle-metaplot`, the hand-rolled FASTA
  parser, and the `fasta_backend` argument of `FastaMotifSiteProvider`
  ([#4](https://github.com/adamyhe/kackle/pull/4)). Passing
  `--fasta-backend` is now an argparse error. The `python` motif-match
  backend is kept as a reference implementation.
- The ahocorasick-rs dependency. `--motif-match-backend ahocorasick` (and
  `match_backend="ahocorasick"`) still work as a deprecated alias for `auto`.

### Fixed

- `kackle-metaplot` reads motif windows in batches of 100,000 sites instead
  of all at once, so memory no longer grows with the site count (genome-wide
  hg38 TGG sites would have needed about 80 GB per array)
  ([#4](https://github.com/adamyhe/kackle/pull/4)).
- `--worker-backend thread` no longer crashes on a FASTA without an index.
  Under pyfastx, chromosome workers raced to build the `.fxi`; the pyfaidx
  `.fai` is built once before workers start
  ([#4](https://github.com/adamyhe/kackle/pull/4)).

- Overlapping target windows of flagged sites less than `2 * target` apart no
  longer race under parallel execution. Previously the last writer won, so
  output was non-deterministic and depth wasn't conserved.
- A neighboring site's spike inside a flagged site's flank no longer leaks
  into that site's redrawn total.

## [0.2.0] - 2026-07-31

### Added

- `kackle-metaplot`: before/after metaplots centered on motif starts.
- Cross-chromosome parallelism (`-w/--chrom-workers`, `--worker-backend`)
  alongside numba threads (`--numba-threads`), sized to avoid oversubscription.
- `--out-bed6-prefix` to save generated FASTA-mode motif sites.
- Support for Python 3.10–3.13.

### Changed

- `correct_kmers` became a fused screen-and-resample kernel that writes back
  only the `target` window of each flagged site instead of the whole `source`
  window. This changes output wherever flagged sites are within `2 * source`
  of each other: 0.1.0 could overwrite a neighbor's correction with original
  flank values, sometimes putting a spike back.
- The CLI moved from `kackle.cli` to `kackle.correction` (`run_correction`).

### Fixed

- Chromosomes missing from any input are skipped instead of crashing.
- Plotting bug in metaplot generation.

## [0.1.0] - 2026-07-09

### Added

- First packaged release: the `kackle` CLI for BED6 and FASTA modes,
  sequential motif passes (`TGG:0` then `TGGAA:1` by default), and numba
  correction kernels.

[Unreleased]: https://github.com/adamyhe/kackle/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/adamyhe/kackle/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/adamyhe/kackle/releases/tag/v0.1.0
