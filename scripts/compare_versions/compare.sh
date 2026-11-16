#!/usr/bin/env bash
# Compare kackle outputs between two git refs (default: v0.1.0 vs v0.2.0).
#
# Usage:
#   scripts/compare_versions/compare.sh [options] [-- extra kackle args]
#
# Options:
#   -a REF        first ref  (default: v0.1.0)
#   -b REF        second ref (default: v0.2.0)
#   -o DIR        work/output directory (default: ./compare-out)
#   -i PLUS.bw    input plus-strand bigWig   } omit all four to generate a
#   -I MINUS.bw   input minus-strand bigWig  } small synthetic dataset with
#   -f GENOME.fa  reference FASTA            } make_synthetic.py
#   -c SIZES      chrom.sizes                }
#   -x            exact mode: disable numba JIT and seed RNG identically, so
#                 outputs must match wherever the algorithms agree (slow; use
#                 small inputs or a chrom.sizes restricted to a few chroms)
#   -p PYTHON     Python version for the version environments (default: 3.12)
#
# Extra args after `--` go to both kackle runs (e.g. -- -s 10 -t 5 -r 10).
# If you pass -s/-t, pass the same values to compare_outputs.py via
# COMPARE_ARGS="-s N -t M".
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(git -C "$here" rev-parse --show-toplevel)"
ref_a=v0.1.0
ref_b=v0.2.0
out=./compare-out
python_version=3.12
exact=""
plus="" minus="" fasta="" sizes=""

while getopts "a:b:o:i:I:f:c:xp:h" opt; do
  case "$opt" in
    a) ref_a="$OPTARG" ;;
    b) ref_b="$OPTARG" ;;
    o) out="$OPTARG" ;;
    i) plus="$OPTARG" ;;
    I) minus="$OPTARG" ;;
    f) fasta="$OPTARG" ;;
    c) sizes="$OPTARG" ;;
    x) exact="--exact" ;;
    p) python_version="$OPTARG" ;;
    h) sed -n '2,24p' "$0"; exit 0 ;;
    *) exit 2 ;;
  esac
done
shift $((OPTIND - 1))
[[ "${1:-}" == "--" ]] && shift
extra=("$@")

mkdir -p "$out"
out="$(cd "$out" && pwd)"

if [[ -z "$plus$minus$fasta$sizes" ]]; then
  echo "== Generating synthetic inputs in $out/inputs"
  uv run --script "$here/make_synthetic.py" "$out/inputs"
  plus="$out/inputs/plus.bw" minus="$out/inputs/minus.bw"
  fasta="$out/inputs/genome.fa" sizes="$out/inputs/chrom.sizes"
elif [[ -z "$plus" || -z "$minus" || -z "$fasta" || -z "$sizes" ]]; then
  echo "error: pass all of -i -I -f -c, or none" >&2
  exit 2
fi
abspath() { (cd "$(dirname "$1")" && echo "$(pwd)/$(basename "$1")"); }
plus="$(abspath "$plus")" minus="$(abspath "$minus")"
fasta="$(abspath "$fasta")" sizes="$(abspath "$sizes")"

run_ref() {
  local label="$1" ref="$2"
  local tree="$out/worktrees/$label"
  if [[ ! -d "$tree" ]]; then
    git -C "$repo" worktree add --detach "$tree" "$ref" >/dev/null
  else
    git -C "$tree" checkout --quiet --detach "$ref"
  fi
  echo "== Running $label ($ref, $(git -C "$tree" rev-parse --short HEAD))"
  # Run from inside $out so any stray files land there (FASTA indexes are
  # written next to the FASTA itself).
  (cd "$out" && uv run --quiet --project "$tree" --python "$python_version" \
    python "$here/run_version.py" $exact --log-starts "$out/$label.starts.npz" -- \
    -i "$plus" -I "$minus" -f "$fasta" -c "$sizes" \
    -o "$out/$label.plus.bw" -O "$out/$label.minus.bw" ${extra[@]+"${extra[@]}"})
}

run_ref a "$ref_a"
run_ref b "$ref_b"

echo "== Comparing"
# shellcheck disable=SC2086
uv run --script "$here/compare_outputs.py" \
  --chrom-sizes "$sizes" --input-pl "$plus" --input-mn "$minus" \
  --a-pl "$out/a.plus.bw" --a-mn "$out/a.minus.bw" --a-starts "$out/a.starts.npz" \
  --b-pl "$out/b.plus.bw" --b-mn "$out/b.minus.bw" --b-starts "$out/b.starts.npz" \
  --a-label "$ref_a" --b-label "$ref_b" --json "$out/report.json" ${COMPARE_ARGS:-}
