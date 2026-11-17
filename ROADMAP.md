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

**Done: start arrays and a numba motif matcher.** Motif location now returns
sorted int64 start arrays from a numba Hamming kernel; BED6 tables are built
only for `--out-bed6-prefix`. Each correction pass locates only the strand it
needs. On hg38 (PROseq_2, 230M sites over both strands and specs), with
identical sites and must-match checks passing:

| | before | after |
|---|---|---|
| locating all sites, one process | 1328 s, 9.1 GB | 39 s, 0.8 GB |
| `kackle`, `-w 4` threads | 39:33, 37.7 GB | 4:16, 10.4 GB |
| `kackle-metaplot`, `-w 4` | 8:09, 6.9 GB | 2:20, 1.8 GB |
| `kackle-metaplot`, `-w 8` | 5:03, 6.9 GB | 1:03, 1.8 GB |

Metaplot memory is the largest single process. Metaplot TSVs are
byte-identical to before.

**Done: one pass over both strands, streamed output.** Each chromosome task
locates sites for both strands once and corrects plus then minus; finished
chromosomes go straight to both bigWigs in order, with no genome-wide per-base
DataFrame. On hg38 (PROseq_2), with byte-identical bigWigs in exact mode
(chr21 and chr22) and must-match checks passing genome-wide:

| | before | after |
|---|---|---|
| `kackle -w 4` (process, default) | 38.6 s, 3.4 GB | 29.9 s, 2.1 GB |
| `kackle -w 8` (process) | | 28.6 s, 2.1 GB |
| harness, `-w 4` threads | 3:15, 10.4 GB | 2:51, 7.7 GB |

Memory is the largest single process. The harness numbers include its own
logging of every anchor array. `-w 8` barely helps: the run is bounded by
chr1.

**Done: large chromosomes split across workers** (`--chunk-size`, default
10 Mb; hg38 becomes 322 chunks). Cuts avoid every anchor by
`max(source, target)`, so chunked output is identical to whole-chromosome
output: byte-identical genome-wide with a deterministic stand-in for the
resampler. On hg38 (PROseq_2), same node, process backend:

| | main | chunked |
|---|---|---|
| `kackle -w 4` | 45.6 s, 4.2 GB | 29.6 s, 0.78 GB |
| `kackle -w 8` | 22.0 s, 2.1 GB | 20.0 s, 0.79 GB |
| `kackle -w 16` | 21.9 s, 2.2 GB | 17.3 s, 0.81 GB |
| `kackle-metaplot -w 8` | 50.8 s, 1.8 GB | 50.5 s, 0.23 GB |
| `kackle-metaplot -w 16` | | 24.7 s, 0.22 GB |

Chunk sizes of 5–25 Mb at `-w 8` were within noise (18.9–20.8 s). Main's
`-w 4` time varies between nodes (29.9 s in the #9 job). `kackle` now
levels off around 17–20 s, which is likely fixed costs: spawning workers,
cut planning in the parent, and writing.

Remaining, roughly in order of payoff:

1. **Profile the remaining `kackle` floor** (about 17 s at `-w 16`) before
   optimizing further: worker start-up, cut planning, bigWig reading and
   writing.
2. **Dense float64 reads.** Each chunk's strand is read as a dense float64
   array and converted to int32 (about 120 MB per 10 Mb chunk). Reading as
   int32 directly, or reading covered intervals, would cut the per-worker
   peak.
3. **BED6 mode reads `chrom`/`name` with the pandas `string` dtype.** The BED
   is now grouped once (`bed_anchor_index`); reading `chrom` as a category
   would also shrink the loaded table.
4. **`kackle-metaplot`:**
   - **A numba kernel for the sparse sum.** `chrom_window_sums` does
     `2 * flank + 1` NumPy gathers over each track's nonzero bases; a fused
     loop would be faster.
   - **Read covered intervals instead of dense chromosomes.** Each track is
     read as a dense float array (about 80 MB per 10 Mb chunk), but PRO-seq covers
     only a few percent of bases. Depends on whether figwig can return
     intervals (not yet checked).
   - Caching sites across invocations is no longer worth it: locating all
     hg38 sites takes about 40 CPU-seconds.

**Plot features** (not performance): overlay several "after" tracks in one
figure, and show each motif spec separately instead of pooling them.

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
