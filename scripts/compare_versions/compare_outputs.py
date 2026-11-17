# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "figwig>=0.3.1"]
# ///
"""Compare corrected bigWigs from two kackle versions run on the same inputs.

Inputs are the original plus/minus bigWigs, each version's corrected outputs,
and the anchor logs written by ``run_version.py``. For every strand and
chromosome it reports:

* anchors: whether both versions corrected the same motif anchors per pass;
* outside: positions that changed outside every candidate window
  ``[anchor - source, anchor + source)``. Neither version may touch these,
  so any count here is a bug;
* isolated windows: candidate windows with no other anchor (any pass) within
  ``2 * source``. Resampling preserves the flank total, so even with different
  random draws the corrected target-window sums must agree. In ``--exact``
  runs the values themselves must agree;
* clustered windows: windows near other anchors, where v0.1.0 (which writes
  back the full source window, flanks included) and v0.2.0 (which writes only
  the target window) are expected to differ. Reported, not failed.

Exit status is 1 if any must-match check fails.
"""

import argparse
import json
import sys

import figwig
import numpy as np


def load_starts(path):
    """Return {(strand, length, pass): anchors} and the run metadata."""
    data = np.load(path)
    meta = json.loads(str(data["meta"]))
    starts = {}
    for i, record in enumerate(meta["records"]):
        key = (record["strand"], record["length"], record["pass"])
        starts[key] = data[f"starts_{i}"]
    return starts, meta


def read_dense(path, chrom, length):
    reader = figwig.BigWigReader(path)
    if chrom not in reader.chrom_sizes:
        return np.zeros(length, dtype=np.int64)
    values = reader.read([chrom], [0], length, missing=0.0, oob=0.0)[0]
    return np.rint(np.abs(values)).astype(np.int64)


def window_sums(signal, anchors, half):
    """Sum of signal[a - half : a + half] for each anchor, via prefix sums."""
    csum = np.concatenate([[0], np.cumsum(signal)])
    lo = np.clip(anchors - half, 0, signal.size)
    hi = np.clip(anchors + half, 0, signal.size)
    return csum[hi] - csum[lo]


def compare_strand_chrom(strand, chrom, length, paths, logs, source, target, exact):
    inp = read_dense(paths["input"], chrom, length)
    a = read_dense(paths["a"], chrom, length)
    b = read_dense(paths["b"], chrom, length)

    passes = {}
    for label, log in logs.items():
        passes[label] = {
            p: s for (st, ln, p), s in log.items() if st == strand and ln == length
        }
    anchors_match = passes["a"].keys() == passes["b"].keys() and all(
        np.array_equal(passes["a"][p], passes["b"][p]) for p in passes["a"]
    )
    all_anchors = np.unique(
        np.concatenate(
            [s for per in passes.values() for s in per.values()] or [np.array([], int)]
        )
    )
    valid = (all_anchors - source >= 0) & (all_anchors + source <= length)
    all_anchors = all_anchors[valid]

    candidate = np.zeros(length + 1, dtype=np.int64)
    np.add.at(candidate, all_anchors - source, 1)
    np.add.at(candidate, all_anchors + source, -1)
    in_window = np.cumsum(candidate)[:length] > 0

    gaps = np.diff(all_anchors)
    close = gaps < 2 * source
    near_prev = np.zeros(all_anchors.size, dtype=bool)
    near_next = np.zeros(all_anchors.size, dtype=bool)
    near_prev[1:] = close  # safe when there are no anchors
    near_next[:-1] = close
    isolated = all_anchors[~(near_prev | near_next)]
    clustered = all_anchors[near_prev | near_next]

    iso_in = window_sums(inp, isolated, target)
    iso_a = window_sums(a, isolated, target)
    iso_b = window_sums(b, isolated, target)
    corrected_a = iso_a != iso_in
    corrected_b = iso_b != iso_in

    # Exact per-base equality inside isolated windows (meaningful in exact mode).
    iso_mask = np.zeros(length + 1, dtype=np.int64)
    np.add.at(iso_mask, isolated - source, 1)
    np.add.at(iso_mask, isolated + source, -1)
    iso_mask = np.cumsum(iso_mask)[:length] > 0
    clu_mask = in_window & ~iso_mask

    diff = a != b
    result = {
        "strand": strand,
        "chrom": chrom,
        "anchors_match": bool(anchors_match),
        "passes": {k: {p: int(s.size) for p, s in v.items()} for k, v in passes.items()},
        "total_input": int(inp.sum()),
        "total_a": int(a.sum()),
        "total_b": int(b.sum()),
        "positions_differing": int(diff.sum()),
        "outside_changed_a": int(((a != inp) & ~in_window).sum()),
        "outside_changed_b": int(((b != inp) & ~in_window).sum()),
        "outside_differing": int((diff & ~in_window).sum()),
        "isolated_windows": int(isolated.size),
        "isolated_corrected_a": int(corrected_a.sum()),
        "isolated_corrected_b": int(corrected_b.sum()),
        "isolated_sum_mismatch": int((iso_a != iso_b).sum()),
        "isolated_positions_differing": int((diff & iso_mask).sum()),
        "clustered_windows": int(clustered.size),
        "clustered_positions_differing": int((diff & clu_mask).sum()),
        "clustered_total_a": int(a[clu_mask].sum()),
        "clustered_total_b": int(b[clu_mask].sum()),
    }
    failures = []
    if not anchors_match:
        failures.append("motif anchors differ")
    for key in ("outside_changed_a", "outside_changed_b", "isolated_sum_mismatch"):
        if result[key]:
            failures.append(f"{key}={result[key]}")
    if exact and result["isolated_positions_differing"]:
        failures.append(
            f"isolated_positions_differing={result['isolated_positions_differing']}"
        )
    result["failures"] = failures
    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[2:]),
    )
    parser.add_argument("--chrom-sizes", required=True)
    parser.add_argument("--input-pl", required=True)
    parser.add_argument("--input-mn", required=True)
    for label in ("a", "b"):
        parser.add_argument(f"--{label}-pl", required=True)
        parser.add_argument(f"--{label}-mn", required=True)
        parser.add_argument(f"--{label}-starts", required=True)
    parser.add_argument("--a-label", default="A")
    parser.add_argument("--b-label", default="B")
    parser.add_argument("-s", "--source", type=int, default=10)
    parser.add_argument("-t", "--target", type=int, default=5)
    parser.add_argument("--json", help="Also write the full report as JSON here.")
    args = parser.parse_args()

    logs, metas = {}, {}
    for label in ("a", "b"):
        logs[label], metas[label] = load_starts(getattr(args, f"{label}_starts"))
    exact = metas["a"]["exact"] and metas["b"]["exact"]
    if metas["a"]["exact"] != metas["b"]["exact"]:
        print("warning: only one run used --exact; comparing statistically", file=sys.stderr)

    chrom_sizes = []
    with open(args.chrom_sizes) as handle:
        for line in handle:
            if line.strip():
                chrom, size = line.split()[:2]
                chrom_sizes.append((chrom, int(size)))
    lengths = [size for _, size in chrom_sizes]
    if len(set(lengths)) != len(lengths):
        print(
            "warning: duplicate chromosome lengths; anchor logs are keyed by "
            "length, so anchors may be attributed to the wrong chromosome",
            file=sys.stderr,
        )

    results = []
    for strand, suffix in (("+", "pl"), ("-", "mn")):
        paths = {
            "input": getattr(args, f"input_{suffix}"),
            "a": getattr(args, f"a_{suffix}"),
            "b": getattr(args, f"b_{suffix}"),
        }
        for chrom, length in chrom_sizes:
            results.append(
                compare_strand_chrom(
                    strand, chrom, length, paths, logs, args.source, args.target, exact
                )
            )

    a_name = f"{args.a_label} ({metas['a']['version']})"
    b_name = f"{args.b_label} ({metas['b']['version']})"
    print(f"A = {a_name}\nB = {b_name}\nmode = {'exact' if exact else 'statistical'}\n")
    header = (
        "strand chrom      anchors  differ  outside(A/B)  iso:win corrA corrB summis "
        "posdiff  clu:win posdiff   totalA     totalB"
    )
    print(header)
    for r in results:
        print(
            f"{r['strand']:6} {r['chrom']:10} {'same' if r['anchors_match'] else 'DIFF':7} "
            f"{r['positions_differing']:7} {r['outside_changed_a']:6}/{r['outside_changed_b']:<6} "
            f"{r['isolated_windows']:8} {r['isolated_corrected_a']:5} {r['isolated_corrected_b']:5} "
            f"{r['isolated_sum_mismatch']:6} {r['isolated_positions_differing']:7} "
            f"{r['clustered_windows']:8} {r['clustered_positions_differing']:7} "
            f"{r['total_a']:10} {r['total_b']:10}"
        )
    failures = [(r["strand"], r["chrom"], f) for r in results for f in r["failures"]]
    print()
    if failures:
        print("MUST-MATCH CHECKS FAILED:")
        for strand, chrom, failure in failures:
            print(f"  {strand} {chrom}: {failure}")
    else:
        print("All must-match checks passed.")
    clustered_diff = sum(r["clustered_positions_differing"] for r in results)
    if clustered_diff:
        print(
            f"{clustered_diff} differing positions are inside clustered windows "
            "(nearby anchors within 2*source), where v0.1.0's full-window write-back "
            "and v0.2.0's target-only write-back are expected to diverge."
        )
    if args.json:
        with open(args.json, "w") as handle:
            json.dump({"a": metas["a"], "b": metas["b"], "results": results}, handle, indent=2)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
