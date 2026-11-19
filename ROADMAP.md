# Roadmap

Known issues and planned work. Shipped changes go in
[CHANGELOG.md](CHANGELOG.md).

## Known issues

### Screening contamination from nearby spikes (investigated; no fix adopted)

The screen compares a site's artifact position with the mean of the rest of
its `[anchor - source, anchor + source)` neighborhood, so a spike at a nearby
candidate inflates that background and can hide a real artifact.
`test_correct_kmers_conserves_cluster_totals_under_parallel_dense_overlaps`
shows it on planted spikes.

On the 6 PRO-seq samples (hg38, TGG pass, 1.9–2.8M sites flagged per sample),
the candidate fixes would add:

| fix | extra flags | notes |
|---|---|---|
| mean excluding other candidates' artifact bases | +2.0–2.6% | drops non-spike candidates from the background too |
| re-screen the corrected signal | +1.2–1.6% | 5% of additions are resampled draws flagged again |
| re-screen: original value vs neighborhood with flagged target windows left out | +1.4–1.9% | deterministic |
| median background | +15% (TGG), +100% (TGGAA) | the median is usually 0 in sparse PRO-seq |

The added sites are weak (71–78% have 1–2 reads). The deterministic re-screen
was implemented and validated (exactly equal to 0.2.0 when off; nothing
changed outside motif windows; chunking still exact), but it made per-motif
metaplots worse. The artifact-base / flank ratio fell in every sample, on
both strands and motifs: TGG 0.92–1.06 → 0.89–1.00 (mean 0.99 → 0.94),
TGGAA:1 0.82–0.92 → 0.80–0.88. It also removed 0.2–0.45% more of the
total signal. Replacing extra 1–2-read bases with flank-level values pulls
the average below the flank. It also doesn't make the TGGAA:1 pass
unnecessary: with the TGG pass alone, TGGAA:1 sites stay at 1.5–2.0× their
flanks either way. So the contamination is real but small, and fixing it
this way costs more than it gains. Revisit only together with the
low-count issue below.

### Single reads on an empty background count as artifacts

A site is flagged when its artifact base exceeds `threshold` times the
neighborhood mean, so with no other reads within ±`source`, one read is
enough (1 > 10 × 0). On the 6 samples, 36–41% of flagged TGG sites have a
single read at the artifact base, and about 70% have a background of 0.
Given the ~11–18× enrichment at artifact bases, most such reads are
probably artifacts, so removing them is likely right on average. But it is
what made re-screening over-correct. Any change to the screen (a minimum
read count, or a pseudocount in the background) should be judged by the
same per-motif metaplots on real samples, not by flag counts.

### Resampling flattens natural structure inside the target window

The whole `[anchor - target, anchor + target)` window of a flagged site is
redrawn from its flanks, not only the artifact base. Metaplots show the
input's own dips near TGG sites (about 0.7 of the flank at offsets −6 to −3
on the plus strand) raised to about 1.0 after correction, in every version.
Resampling only the artifact base, or preserving the window's shape, would
change every corrected site's output.

The dips themselves are real: the raw input shows the same depletion at the
G bases of AGG, CGG and GGG, which carry no artifact. A base-only prototype
(`--resample base`, #14) redraws just the artifact base
with a Poisson draw from its neighborhood mean. On the 6 samples it changed
nothing outside artifact bases. It kept the motif dip (TGG offset +1: input
0.13, window mode 0.33, base only 0.18 of the flank). The artifact-base ratio
stayed close to window mode (TGG 0.91–1.05 vs 0.92–1.06; TGGAA:1 0.80–0.89 vs
0.82–0.92). It removed about 1% more of the total signal, mostly reads that
window mode adds back into the dip.

### Resampled total isn't density-preserving when `source - target != target`

Each flagged span is redrawn with a total equal to its flank sum, scaled by
`span_width / (2 * target)`. Isolated sites have `2 * (source - target)` flank
bases and `2 * target` target bases, so the per-base level is only preserved
when `source = 2 * target`, as with the defaults (10 and 5). With `-s 20 -t 5`,
for example, a corrected window gets three times the local density. Scaling by
`span_width / flank_width` instead would fix this, but it would change output
for every non-default `source`/`target` setting.

## Planned

### Adapter-based, single-pass site search

The artifact is primer mispriming on 3' adapter look-alikes inside real
fragments. Reads end at the base just before a stretch of genome that
resembles the adapter's 5' end (`TGGAATTCTCGG` for the TruSeq small-RNA
adapter). On the 6 samples, the raw spike grows with how much of the adapter
matches:

| match | raw spike |
|---|---|
| TGG, then off-adapter | 3.3× |
| TGGA | 14× |
| TGGAA | 73× |
| TGGAAT | 232× |
| 8–12 adapter bases | 560–1350× |

A mismatch inside TGG is tolerated when the rest matches: one mismatch at the
T followed by 7 more adapter bases gives 40×. Of the 16 TGGAA:1 variants, four
(TCGAA, GGGAA, TGGTA, TGGCA) carry almost no artifact. After the TGG pass,
most of what remains is below the 10× threshold: TGGA–TGGAATT sites sit at
1.6–2.4× their flanks with only about 0.15% flagged.

**Shipped as experimental in #14** (`--adapter SEQ`, with base-only
resampling; usage in the README):

- A position is a candidate when the RNA-sense sequence that follows matches
  the adapter's 5' end: at least 3 bases with no mismatches, or 5 with one.
- Tiers lower the fold threshold for stronger matches, by match length and
  optionally mismatch count (`LEN[/MAXMM]:FOLD`, default `4/0:7,8:5,10:3`).
- One pass, in two stages: a plain screen, then a rescreen whose background
  also leaves out the artifact bases flagged in the first stage. A first
  version left every candidate's artifact base out of the background; that
  flagged more weak sites and pulled the corrected level down by about 0.05.
- `-k` / `-b` multi-pass stay for other artifact types.

On the 6 samples, ratio = artifact base / mean at 20–50 bp from the motif:

| | two-pass base (TGG:0 → TGGAA:1) | two-stage, `8:5,10:3` | two-stage, default `4/0:7,8:5,10:3` |
|---|---|---|---|
| TGG metaplot | 0.91–1.05 | 0.90–1.03 | 0.83–0.94 |
| TGGAA metaplot | 0.80–0.89 | 0.81–0.90 | 0.72–0.81 |
| exact TGG (3 bases) | 0.82 | 0.78 | 0.78 |
| exact TGGA / TGGAA | 1.53 / 1.58 | 1.56 / 1.68 | 1.19 / 1.30 |
| 1-mismatch, 8–10 bases | 0.65–1.29 | 0.32–0.58 | 0.32–0.58 |
| 1-mismatch 5–7 bases, non-TGG 5-mers | 0.32–0.67 | 0.32–0.66 | 0.32–0.66 |
| signal removed | 22.30–35.14% | +0.02–0.05 points | +0.13–0.15 points |
| time per sample | 1:35–1:54 | | 15–60 s |

The exact-match tier changes only the classes it targets; every other class
is the same to two decimals. A plain `4:7` length tier, by contrast, also
lowered the threshold for one-mismatch 5–7-base matches, which carry almost
no artifact, and pulled them down to 0.22–0.43.

**The metaplot ratio is not a good target on its own.** The TGG metaplot
pools exact TGG-only sites, which sit naturally at about 0.78, with TGGA+
sites that still carried leftover artifact. Its "about 1.0" after two-pass
correction was partly that leftover cancelling a natural dip. Judge changes
by match class instead. `4/0:5` goes further (exact TGGA 1.00, TGGAA 1.14;
TGG metaplot 0.79–0.90); without an artifact-free library we can't tell
whether that over-corrects, so the default stops at 7×.

Open:

- About 36k artifact reads per sample remain at exact TGGA/TGGAA sites. Some
  of it is probably spread over many sites only a few times above
  background, which no per-site threshold can remove cleanly.
- Calibrate the tiers per library from the input itself, and report artifact
  signal by match class.
- The artifact base sits just before the look-alike in RNA orientation,
  which holds for 3'-end tracks like these. Other library layouts would need
  an option for which side the adapter is on.
- Decide whether `--adapter` (or `--resample base`) should become the default.

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

### Corrected TGGAA sites sit slightly below their flanks (answered)

After correction, TGGAA:1 sites sit at 0.82–0.92 of their 20–50 bp flanks,
compared with about 1.0 at TGG sites. It is not over-correction from being
corrected twice. 57% of TGGAA:1 sites have no TGG at the same anchor (the
mismatch is in the first three bases), and those sites have low signal at
that base already in the uncorrected input (0.30–0.37 of the flank). Averaging
them with the TGG-containing sites gives about 0.9. TGGAA sites that the TGG
pass flagged look low (about 0.75) only through selection: those whose
resampled value came out high are the ones the second pass flags again.
(6 PRO-seq samples, hg38.)
