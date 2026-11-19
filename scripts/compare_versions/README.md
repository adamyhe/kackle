# Comparing kackle versions

Scripts for checking how two kackle releases (default `v0.1.0` and `v0.2.0`)
differ on the same inputs. Each ref is checked out into its own git worktree
and run in its own `uv` environment, so the versions' dependencies never mix.

```bash
# Small synthetic dataset, exact mode (deterministic, ~30 s)
scripts/compare_versions/compare.sh -x -o compare-out

# Your own data, statistical mode (normal numba-compiled runs)
scripts/compare_versions/compare.sh -o compare-out \
  -i plus.bw -I minus.bw -f genome.fa -c chrom.sizes

# Any two refs, extra kackle args after --
scripts/compare_versions/compare.sh -a v0.2.0 -b figwig-pyfaidx -- -r 8
```

## Why outputs can't be compared directly

`correct_kmers` resamples each flagged window with a Dirichlet-multinomial
draw. The draws come from numba's per-thread RNG and can't be seeded, so two
runs of the *same* version already differ at every corrected window. The
scripts separate that noise from real behavior changes in two ways:

- **Exact mode (`-x`)** disables numba JIT and seeds each resampling draw from
  (strand, pass, anchor position). The same flagged window gets the same draw
  in both versions, so outputs must match wherever the algorithms agree. The
  kernels run as plain Python, so keep inputs small (a few Mb, or a
  `chrom.sizes` restricted to one or two chromosomes).
- **Statistical mode** (the default) runs the real compiled kernels and
  checks properties that don't depend on the random draws (see below).

## What gets checked

`run_version.py` wraps each version's `correct_kmers` to log the motif anchors
of every pass. `compare_outputs.py` then reports, per strand and chromosome:

| Column | Meaning | Must be |
|---|---|---|
| `anchors` | Both versions corrected the same motif anchors in every pass | `same` |
| `outside(A/B)` | Positions changed outside every `[anchor - source, anchor + source)` window | `0/0` |
| `summis` | Isolated windows (no other anchor within `2*source`) whose corrected target-window sum differs. Resampling preserves the flank total, so this holds even with different draws | `0` |
| `iso posdiff` | Per-base differences inside isolated windows | `0` in exact mode |
| `clu posdiff` | Per-base differences inside clustered windows | reported only |

The script exits with status 1 if any must-match check fails. `report.json`
in the output directory has the full per-chromosome numbers.

Logged passes are matched to chromosomes by signal length. Releases that
split chromosomes into chunks (`--chunk-size`) call `correct_kmers` once per
chunk, so `run_version.py` adds `--chunk-size 0` for them unless you pass it.
Chunked and whole-chromosome output are identical anyway (see
`tests/test_chunking.py`). It also adds `-w 1 --worker-backend thread` unless
you pass `-w`, so every pass runs in-process where the wrapper can see it.

## Known differences between releases

Since v0.3.0 the default is base-only resampling (`--resample base`), which
the harness doesn't instrument. `run_version.py` therefore adds
`--resample window` for releases that have the flag, unless the arguments
already choose a mode, so comparisons against older releases stay
like-for-like.

Since v0.2.0, nearby flagged sites are resampled together as one cluster
instead of one window each, so clustered windows also differ between v0.2.0
and later releases. Isolated windows still match.

### v0.1.0 and v0.2.0

All differences fall inside **clustered** windows, where flagged anchors are
less than `2*source` apart:

- **v0.1.0** copies each flagged window's full `[anchor - source, anchor + source)`
  out of the input, resamples the center, and writes the whole window back,
  original flanks included. When two flagged windows overlap, one window's
  original flank values overwrite the other's corrected center, which can put
  the original spike back. Which window wins depends on thread scheduling.
- **v0.2.0** writes back only the `[anchor - target, anchor + target)` target
  window, so neighboring corrections are kept.

On the synthetic dataset in exact mode, isolated windows match base for base
between the versions, and every remaining difference is in a clustered window.
Comparing a ref against itself in exact mode gives zero differences.

## Files

- `compare.sh`: orchestrates worktrees, runs, and the comparison.
- `run_version.py`: runs one version's CLI with anchor logging and exact-mode
  RNG keying. Needs only numpy and the stdlib, so it works with any release.
- `make_synthetic.py`: writes a small genome, chrom.sizes, and stranded
  bigWigs with spikes planted at TGG sites, including dense "hot" regions
  that produce overlapping windows.
- `compare_outputs.py`: the comparison and report.

Worktrees live under `<out>/worktrees/`. Remove them with
`git worktree remove <out>/worktrees/a` (and `b`) when done.
