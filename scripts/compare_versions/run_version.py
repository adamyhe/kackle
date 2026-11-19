"""Run one installed kackle version with instrumentation, for version comparisons.

Run this inside the environment of the kackle version under test (e.g.
``uv run --project <worktree> python run_version.py ...``). It depends only on
numpy and the stdlib, so it works against any kackle release.

It wraps the ``correct_kmers`` reference held by the CLI module to:

* log every non-empty batch of motif anchors (strand, chromosome length, pass
  index, anchors) to ``--log-starts`` so the comparison can tell which windows
  each version was allowed to modify; and
* with ``--exact``, disable numba JIT and key every resampling draw to
  (strand, pass, anchor). Each flagged window then gets the same Dirichlet and
  multinomial draw in both versions regardless of how many other windows were
  flagged before it, so any remaining output difference is a logic
  difference, not noise. Exact mode runs the kernels as plain Python: use
  small inputs.

Everything after ``--`` is passed to the kackle CLI. For releases with
``--resample`` (0.3.0 and later), ``--resample window`` is added unless the
arguments choose a mode, because only window-mode passes are instrumented.
"""

import argparse
import importlib
import json
import os
import sys
import zlib


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log-starts", required=True, help="Output .npz of anchors")
    parser.add_argument(
        "--exact",
        action="store_true",
        help="Disable numba JIT and seed NumPy RNG per pass from the anchors.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed (exact mode).")
    parser.add_argument("kackle_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.kackle_args and args.kackle_args[0] == "--":
        args.kackle_args = args.kackle_args[1:]
    return args


def resolve_cli():
    """Return (module holding correct_kmers, CLI entry point, version string)."""
    from importlib.metadata import version

    kackle_version = version("kackle")
    for module_name, entry_name in (
        ("kackle.correction", "run_correction"),  # >= 0.2.0
        ("kackle.cli", "wrapper"),  # 0.1.x
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        if hasattr(module, "correct_kmers") and hasattr(module, entry_name):
            return module, getattr(module, entry_name), kackle_version
    raise RuntimeError(f"Don't know how to drive kackle {kackle_version}")


class AnchorKeyedRandom:
    """Stand-in for ``numpy.random`` inside the (un-jitted) kackle kernels.

    ``dirichlet`` starts a fresh RandomState seeded from the current pass key
    and the anchor of the window being resampled; the ``multinomial`` that
    follows draws from the same state. The anchor is read from the calling
    kernel's frame: ``start`` in v0.2's fused ``correct_kmers``, or
    ``starts[i]`` across v0.1's ``correct_kmers`` -> ``resample_dirmult``.
    """

    def __init__(self, np_module):
        self._np = np_module
        self.pass_key = 0
        self._state = None

    def __getattr__(self, name):
        return getattr(self._np.random, name)

    def _current_anchor(self):
        frame = sys._getframe(2)
        resample_i = None
        while frame is not None:
            name = frame.f_code.co_name
            local = frame.f_locals
            if name == "resample_dirmult":
                resample_i = int(local["i"])
            elif name == "correct_kmers":
                if resample_i is not None:
                    return int(local["starts"][resample_i])
                return int(local["start"])
            frame = frame.f_back
        raise RuntimeError("could not find the anchor of the window being resampled")

    def dirichlet(self, alpha):
        anchor = self._current_anchor()
        seed = zlib.crc32(anchor.to_bytes(8, "little", signed=True), self.pass_key)
        self._state = self._np.random.RandomState(seed)
        return self._state.dirichlet(alpha)

    def multinomial(self, n, pvals):
        return self._state.multinomial(n, pvals)


class NumpyProxy:
    """``numpy`` module proxy whose ``random`` attribute is anchor-keyed."""

    def __init__(self, np_module, random_proxy):
        self._np = np_module
        self.random = random_proxy

    def __getattr__(self, name):
        return getattr(self._np, name)


def main():
    args = parse_args()
    if args.exact:
        # Must be set before numba is imported anywhere.
        os.environ["NUMBA_DISABLE_JIT"] = "1"

    import numpy as np

    module, entry_point, kackle_version = resolve_cli()
    keyed_random = None
    if args.exact:
        import kackle.computation as computation

        keyed_random = AnchorKeyedRandom(np)
        computation.np = NumpyProxy(np, keyed_random)
    original = module.correct_kmers
    records = []
    pass_counter = {}

    def instrumented(signal, starts, source, target, threshold, strand):
        starts_arr = np.asarray(starts)
        if starts_arr.shape[0] > 0:
            key = (strand, int(signal.shape[0]))
            pass_idx = pass_counter.get(key, 0)
            pass_counter[key] = pass_idx + 1
            records.append(
                {
                    "strand": strand,
                    "length": int(signal.shape[0]),
                    "pass": pass_idx,
                    "starts": starts_arr.astype(np.int64),
                }
            )
            if keyed_random is not None:
                keyed_random.pass_key = zlib.crc32(
                    f"{args.seed}|{strand}|{signal.shape[0]}|{pass_idx}".encode()
                )
        return original(signal, starts, source, target, threshold, strand)

    module.correct_kmers = instrumented

    cli_args = list(args.kackle_args)
    if module.__name__ == "kackle.correction" and not {"-w", "--chrom-workers"} & set(
        cli_args
    ):
        # Keep chromosomes serial and in-process so the wrapper sees every pass
        # and exact-mode RNG draws are not interleaved across workers.
        cli_args += ["-w", "1", "--worker-backend", "thread"]
    if hasattr(module, "DEFAULT_CHUNK_SIZE") and "--chunk-size" not in cli_args:
        # Releases that split chromosomes into chunks (--chunk-size) call
        # correct_kmers once per chunk, with chunk-relative anchors. Records
        # are matched to chromosomes by signal length, so correct whole
        # chromosomes. Chunked output equals whole-chromosome output anyway.
        cli_args += ["--chunk-size", "0"]
    resample = cli_args[cli_args.index("--resample") + 1] if "--resample" in cli_args[:-1] else None
    if hasattr(module, "correct_artifact_bases") and resample is None and "--adapter" not in cli_args:
        # Since 0.3.0 the default is base-only resampling, which doesn't go
        # through correct_kmers. The checks here (window sums, keyed draws)
        # are for window mode, so compare releases in window mode.
        cli_args += ["--resample", "window"]
    elif resample == "base" or "--adapter" in cli_args:
        print(
            "[run_version] warning: only window-mode passes are logged and keyed; "
            "--resample base and --adapter runs are not",
            file=sys.stderr,
        )
    sys.argv = ["kackle", *cli_args]
    print(f"[run_version] kackle {kackle_version} exact={args.exact}: {' '.join(cli_args)}")
    entry_point()

    meta = [{k: r[k] for k in ("strand", "length", "pass")} for r in records]
    np.savez_compressed(
        args.log_starts,
        meta=json.dumps(
            {"version": kackle_version, "exact": args.exact, "records": meta}
        ),
        **{f"starts_{i}": r["starts"] for i, r in enumerate(records)},
    )
    print(f"[run_version] logged {len(records)} motif passes to {args.log_starts}")


if __name__ == "__main__":
    main()
