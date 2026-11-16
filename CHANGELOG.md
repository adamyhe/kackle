# Changelog

Notable changes to kackle. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/) (pre-1.0, so minor versions may
change behavior). Planned work and known issues live in [ROADMAP.md](ROADMAP.md).

## [Unreleased]

### Changed

- **Nearby flagged sites are now corrected together** (`correct_kmers`).
  Flagged anchors closer than `source + target` are merged into a cluster, and
  each cluster's `[first - target, last + target)` span is redrawn once from
  the flanks just outside it. The redrawn total is the flank sum scaled by
  `span_width / (2 * target)`. Isolated sites are unchanged. Outputs differ
  from 0.2.0 only where flagged sites are close together.

### Fixed

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
