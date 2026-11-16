# Roadmap

Known issues and planned work. Shipped changes go in
[CHANGELOG.md](CHANGELOG.md).

## Known issues

### Screening contamination from nearby spikes

The screen compares a site's artifact position with the mean of the rest of
its `[anchor - source, anchor + source)` neighborhood. When another spike sits
inside that neighborhood, it inflates the background, and a real artifact can
fail the `threshold` test and go uncorrected. Dense artifact regions are
therefore under-corrected. The error is in the conservative direction (signal
is left alone, not corrupted), but it is common wherever motifs cluster.
`test_correct_kmers_conserves_cluster_totals_under_parallel_dense_overlaps`
shows it: some planted spikes there are never flagged.

Possible fixes, each of which changes which sites get flagged:

- a robust background, such as the median of the neighborhood, or a mean that
  excludes other candidate artifact positions;
- iterating: screen, correct, then re-screen the remaining candidates against
  the corrected signal until nothing new is flagged.

### Resampled total isn't density-preserving when `source - target != target`

Each flagged span is redrawn with a total equal to its flank sum, scaled by
`span_width / (2 * target)`. Isolated sites have `2 * (source - target)` flank
bases and `2 * target` target bases, so the per-base level is only preserved
when `source = 2 * target`, as with the defaults (10 and 5). With `-s 20 -t 5`,
for example, a corrected window gets three times the local density. Scaling by
`span_width / flank_width` instead would fix this, but it would change output
for every non-default `source`/`target` setting.

## Performance

Profiled on a synthetic 100 Mb genome in FASTA mode (2 chromosomes, serial):
25.4 s total, of which motif location was 17.5 s (69%). The correction kernel
itself was negligible.

1. **Motifs are located twice per chromosome.** The plus and minus passes each
   call `FastaMotifSiteProvider.for_chrom`, which scans both strands, and each
   pass discards half the result. Locate once per chromosome, or only the
   strand needed.
2. **Per-hit Python objects in motif location.** The per-hit tuple loop and
   duplicate-tracking set (5.5 s) and pandas BED construction and sorting
   (6.1 s) dwarf the Aho-Corasick scan (3.6 s). Return NumPy anchor arrays and
   build BED tables only for `--out-bed6-prefix`. A numba Hamming-match kernel
   over the sequence bytes could replace the variant expansion entirely.
3. **Per-base output DataFrames** (not measured separately). Each chromosome
   becomes one row per nonzero base with an object `chrom` column, concatenated
   genome-wide before writing and pickled back to the parent in process mode.
   Return arrays, or stream each corrected chromosome straight to the writer.
4. **Two full strand passes.** Each strand gets its own worker pool and reloads
   every chromosome. One per-chromosome pass over both strands would share the
   FASTA load, the motif scan, and worker start-up.
5. **BED6 mode re-filters the whole BED per chromosome** in
   `motif_anchor_arrays`, and reads `chrom`/`name` with the pandas `string`
   dtype. Group once and read `chrom` as a category.

## In progress elsewhere

- [#4](https://github.com/adamyhe/kackle/pull/4): replace pybigtools with
  figwig and pyfastx with pyfaidx, and support Python 3.14.
- `compare-versions` branch: scripts to compare outputs across releases,
  with an exact (seeded, un-jitted) mode and a statistical mode.
