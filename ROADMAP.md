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
   `kackle-metaplot` needs only start arrays too, so this is also the main
   remaining metaplot cost (see item 6).
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

6. **`kackle-metaplot` after genome-scale streaming.** Measured on hg38 in
   FASTA mode, one sample (about 115M sites per strand): 8:09 with `-w 4`
   and 5:03 with `-w 8`, peaking at 6.9 GB per worker (11.2 GB in #6,
   before the memory changes below). On chr21 alone
   (58 s), motif location was about half the runtime. The whole-genome run
   hasn't been profiled by stage, so **profile chr1 before changing
   anything**, to confirm where the time goes.

   **Speed**, in likely order of payoff:
   1. **Return start arrays from motif location** (item 2). Each worker
      builds a sorted pandas BED6 table for its chromosome, with object
      string columns (about 18M hits on chr1), only to pull out `start` per
      strand. This is also the largest remaining memory cost (below), and it
      speeds up `kackle` too.
   2. **Cache sites across invocations.** Sites depend only on the genome
      and motif specs, but every plot repeats location (about 33 CPU-minutes
      for hg38). A per-chromosome `.npz` cache keyed by FASTA and motif specs
      would leave later plots, e.g. one per sample, with only reading and
      summing. A standalone prototype that did this located hg38 sites in
      4 minutes with 16 processes and reused them across 6 samples.
   3. **Split large chromosomes across workers.** At `-w 8` the run is
      bounded by chr1. Splitting a chromosome into chunks (with `flank`
      bases of overlap, assigning each site to the chunk that holds it)
      would let more workers help, but only pays off beyond about 8
      workers.
   4. **A numba kernel for the sparse sum.** `chrom_window_sums` does
      `2 * flank + 1` NumPy gathers over each track's nonzero bases. A fused
      numba loop would be faster but is a small share of the runtime now.

   **Memory.** Peak memory scales with `-w`, one chromosome per worker. On
   chr21 (46.7 Mb, 3M sites, `-w 1`), peak RSS was 2.04 GB, of which motif
   location alone reached 1.13 GB, and the returned motif tables held 0.50 GB.
   - **Done:** drop the motif tables as soon as the start arrays are pulled
     out (they used to be concatenated into another copy and kept alive
     through the track phase); read, sum and free one track at a time, with
     `abs()` in place; store the site-count vector as uint8 instead of int32.
     Peak per worker fell from 2.04 to 1.19 GB on chr21 and from 11.2 to
     6.9 GB on the whole genome (34.8 GB total at `-w 8`), with byte-identical
     output.
   - **Next: return start arrays from motif location** (speed item 1). The
     peak is now essentially motif location's own (1.13 of 1.19 GB on
     chr21), from per-hit tuples, the duplicate-tracking set and the BED6
     table. Plain int64 start arrays would be about 70 MB per strand on
     chr1.
   - **Read covered intervals instead of dense chromosomes.** Each track is
     read as a dense float array over the whole chromosome (about 1 GB on
     chr1), but PRO-seq covers only a few percent of bases. Reading
     intervals would make track memory proportional to coverage. This
     depends on whether figwig can return intervals rather than dense
     values (not yet checked).

   **Plot features** (not performance): overlay several "after" tracks in
   one figure, and show each motif spec separately instead of pooling them.

## Open questions

### Corrected TGGAA sites sit slightly below their flanks

On 6 real PRO-seq samples, the signal at the TGGAA:1 artifact position after
correction is 0.88–0.90 of the mean at ±20–50 bp, compared with 0.99–1.01
at TGG sites and about 14× in the input. This holds for every version,
v0.1.0 included. It could be mild over-correction: TGGAA sites are almost
all TGG sites too, so they have already been corrected once by the first
pass. Or the artifact position could have genuinely lower signal. A check:
compare against a library without the primer artifact, or against TGGAA
sites that the TGG pass did not flag.

## In progress elsewhere

- `compare-versions` branch: scripts to compare outputs across releases,
  with an exact (seeded, un-jitted) mode and a statistical mode.
