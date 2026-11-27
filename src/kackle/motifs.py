"""Motif location helpers for generating BED6-style kmer sites.

This module replaces the previous ``seqkit locate`` preprocessing step for the
short artifact motifs kackle corrects. It performs case-insensitive exact
matching after expanding a motif to all sequences within a Hamming-distance
neighborhood, then emits BED6-style rows with 0-based half-open coordinates.

The command-line workflow uses :class:`FastaMotifSiteProvider` with pyfaidx to
locate motifs one chromosome at a time, as sorted NumPy start arrays per strand
(:meth:`FastaMotifSiteProvider.starts_for_chrom`). BED6 tables are built only
when asked for (``for_chrom``, ``--out-bed6-prefix``, and the ``locate_motif*``
helpers).

A window matches a motif with up to ``N`` mismatches when every base is A/C/G/T
(any case) and at most ``N`` of them differ from the motif. That is the same
set an exact search over all A/C/G/T Hamming variants of the motif finds, so
windows containing N or IUPAC codes never match.
"""

import os
import threading
import warnings
from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Literal

import numpy as np
import pandas as pd


DNA_ALPHABET = ("A", "C", "G", "T")
RC_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")
BED6_COLUMNS = ["chrom", "start", "end", "name", "score", "strand"]
MatchBackend = Literal["auto", "numba", "python"]
MATCH_BACKENDS = ("auto", "numba", "python")
STRANDS = ("+", "-")


def parse_motif_spec(spec):
    """Parse a CLI motif specification.

    Motifs are written as ``KMER`` for exact matching or ``KMER:N`` for up to
    ``N`` mismatches. For example, ``TGG`` becomes ``("TGG", 0)`` and
    ``TGGAA:1`` becomes ``("TGGAA", 1)``.
    """
    if ":" not in spec:
        return spec, 0
    motif, mismatches = spec.rsplit(":", 1)
    max_mismatches = int(mismatches)
    if max_mismatches < 0:
        raise ValueError("Motif mismatches must be >= 0")
    return motif, max_mismatches


def mismatch_variants(pattern, max_mismatches):
    """Generate all concrete DNA strings within a Hamming distance.

    Parameters
    ----------
    pattern : str
        DNA motif containing only A/C/G/T characters. Matching is
        case-insensitive, and returned variants are uppercase.
    max_mismatches : int
        Maximum number of substitutions to allow.

    Returns
    -------
    set[str]
        All motif variants with ``0..max_mismatches`` substitutions.
    """
    pattern = pattern.upper()
    if not pattern:
        raise ValueError("Motif must not be empty")
    invalid = set(pattern) - set(DNA_ALPHABET)
    if invalid:
        raise ValueError(f"Unsupported motif bases: {''.join(sorted(invalid))}")

    variants = set()
    for replacement_count in range(max_mismatches + 1):
        for positions in combinations(range(len(pattern)), replacement_count):
            choices = []
            for pos in positions:
                choices.append([base for base in DNA_ALPHABET if base != pattern[pos]])
            for replacements in product(*choices):
                chars = list(pattern)
                for pos, base in zip(positions, replacements):
                    chars[pos] = base
                variants.add("".join(chars))
    return variants


def open_fasta(path):
    """Open an indexed FASTA with pyfaidx, building or refreshing its ``.fai``.

    Record names are the first whitespace-delimited token after ``>``.
    Sequences are returned as plain strings, with case preserved.
    """
    import pyfaidx

    return pyfaidx.Fasta(str(path), as_raw=True)


_THREAD_FASTA = threading.local()


def thread_fasta(path):
    """This thread's open :func:`open_fasta` handle for ``path``, reused across calls.

    pyfaidx handles aren't thread-safe and opening one parses the ``.fai``
    (tens of ms for a large genome), which used to happen for every chunk.
    One handle per thread and file is kept open; it is replaced when the
    FASTA's size or modification time changes.
    """
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_size, st.st_mtime_ns)
    handles = getattr(_THREAD_FASTA, "handles", None)
    if handles is None:
        handles = _THREAD_FASTA.handles = {}
    fasta = handles.get(key)
    if fasta is None:
        for old in [k for k in handles if k[0] == key[0]]:
            handles.pop(old).close()
        fasta = handles[key] = open_fasta(path)
    return fasta


def fasta_chrom_names(path):
    """Return FASTA record names in file order, read from the ``.fai`` index."""
    fasta = open_fasta(path)
    try:
        return list(fasta.keys())
    finally:
        fasta.close()


def _sort_bed(rows):
    """Return BED6 rows as a deterministically sorted DataFrame."""
    if not rows:
        return pd.DataFrame([], columns=BED6_COLUMNS)
    bed = pd.DataFrame(rows, columns=BED6_COLUMNS)
    return bed.sort_values(["chrom", "strand", "start", "end"]).reset_index(drop=True)


def resolve_match_backend(match_backend):
    """Resolve a motif matching backend name.

    ``auto`` (the default) uses the numba kernel. When numba JIT is disabled
    (``NUMBA_DISABLE_JIT=1``, e.g. the compare-versions exact mode or
    debugging), it uses the ``python`` matcher instead: the uncompiled kernel
    runs as interpreted loops, about 6 s/Mb per motif and strand. The python
    matcher is still much slower than compiled numba, and its cost grows with
    the number of mismatch variants (on 5 Mb, one strand: TGGAA:1 0.25 s vs
    numba 0.011 s; an 8-mer with 2 mismatches 2.5 s vs 0.017 s), so it is
    meant for small inputs. ``ahocorasick`` is accepted as a deprecated alias
    for ``auto``; ahocorasick-rs is no longer used.
    """
    if match_backend == "ahocorasick":
        warnings.warn(
            "match backend 'ahocorasick' is deprecated and now means 'auto' "
            "(ahocorasick-rs is no longer used)",
            DeprecationWarning,
            stacklevel=3,
        )
        match_backend = "auto"
    if match_backend not in MATCH_BACKENDS:
        raise ValueError(
            "Match backend must be one of: " + ", ".join(repr(b) for b in MATCH_BACKENDS)
        )
    if match_backend == "auto":
        from numba import config

        return "python" if config.DISABLE_JIT else "numba"
    return match_backend


def _validate_motif(motif):
    motif = motif.upper()
    if not motif:
        raise ValueError("Motif must not be empty")
    invalid = set(motif) - set(DNA_ALPHABET)
    if invalid:
        raise ValueError(f"Unsupported motif bases: {''.join(sorted(invalid))}")
    return motif


#: Byte -> base code for the numba matcher: A=0, C=1, G=2, T=3 (any case), 4 otherwise.
_BASE_CODES = np.full(256, 4, dtype=np.uint8)
for _code, _base in enumerate("ACGT"):
    _BASE_CODES[ord(_base)] = _BASE_CODES[ord(_base.lower())] = _code


def _sequence_codes(sequence):
    """Base codes of ``sequence`` (see ``_BASE_CODES``; non-ASCII characters give 4)."""
    from kackle.computation import sequence_codes

    raw = np.frombuffer(sequence.encode("ascii", "replace"), dtype=np.uint8)
    return sequence_codes(raw, _BASE_CODES)


def _python_window_starts(sequence_upper, pattern, max_mismatches):
    """Reference matcher: exact ``str.find`` over every A/C/G/T mismatch variant.

    One pass per variant, so cost grows with ``mismatch_variants`` (1 for
    TGG:0, 16 for TGGAA:1, 277 for an 8-mer with 2 mismatches). Used as the
    independent check on the numba kernel and as the no-JIT fallback.
    """
    found = set()
    for variant in mismatch_variants(pattern, max_mismatches):
        start = sequence_upper.find(variant)
        while start != -1:
            found.add(start)
            start = sequence_upper.find(variant, start + 1)
    return np.array(sorted(found), dtype=np.int64)


def _window_starts(encoded, pattern, max_mismatches, match_backend):
    """Dispatch to a matcher. ``encoded`` is base codes, or an uppercase str for python."""
    if match_backend == "python":
        return _python_window_starts(encoded, pattern, max_mismatches)
    from kackle.computation import MAX_MASK_MOTIF_LENGTH, motif_window_mask

    if len(pattern) > MAX_MASK_MOTIF_LENGTH:
        raise ValueError(f"motifs longer than {MAX_MASK_MOTIF_LENGTH} bases are not supported")
    codes = _BASE_CODES[np.frombuffer(pattern.encode("ascii"), dtype=np.uint8)]
    # More mismatches than bases allows any all-A/C/G/T window; clamp so the
    # kernel's invalid-base bit (128) still exceeds the limit.
    mask = motif_window_mask(encoded, codes, min(max_mismatches, len(pattern)))
    return np.flatnonzero(mask).astype(np.int64)


def _encode_for(sequence, match_backend):
    return sequence.upper() if match_backend == "python" else _sequence_codes(sequence)


def locate_motif_starts(
    sequence,
    motif,
    max_mismatches=0,
    both_strands=True,
    strands=None,
    match_backend="auto",
    _encoded=None,
):
    """Return ``{"+": starts, "-": starts}`` for a motif in one sequence.

    Starts are sorted, unique, 0-based int64 positions of the window's
    leftmost base on the input sequence. Minus-strand hits are windows that
    match the motif's reverse complement. A strand not searched gets an
    empty array.

    Parameters
    ----------
    sequence : str
        Sequence to scan, any case.
    motif : str
        A/C/G/T motif.
    max_mismatches : int
        Maximum substitutions allowed.
    both_strands : bool
        Search the reverse complement too. Ignored when ``strands`` is given.
    strands : iterable of {"+", "-"}, optional
        Strands to search, e.g. ``("+",)`` when only one is needed.
    match_backend : {"auto", "numba", "python"}
        Matching engine. Both give identical results.
    """
    motif = _validate_motif(motif)
    if max_mismatches < 0:
        raise ValueError("Motif mismatches must be >= 0")
    match_backend = resolve_match_backend(match_backend)
    if strands is None:
        strands = STRANDS if both_strands else ("+",)
    encoded = _encode_for(sequence, match_backend) if _encoded is None else _encoded
    patterns = {"+": motif, "-": motif.translate(RC_TABLE)[::-1]}
    return {
        strand: (
            _window_starts(encoded, patterns[strand], max_mismatches, match_backend)
            if strand in strands
            else np.empty(0, dtype=np.int64)
        )
        for strand in STRANDS
    }


def bed_from_starts(chrom, motif, starts):
    """Build a sorted BED6 table from ``{strand: starts}`` (plus rows first)."""
    motif = motif.upper()
    parts = []
    for strand in STRANDS:
        strand_starts = np.asarray(starts.get(strand, ()), dtype=np.int64)
        if strand_starts.size == 0:
            continue
        parts.append(
            pd.DataFrame(
                {
                    "chrom": np.full(strand_starts.size, chrom, dtype=object),
                    "start": strand_starts,
                    "end": strand_starts + len(motif),
                    "name": np.full(strand_starts.size, motif, dtype=object),
                    "score": np.zeros(strand_starts.size, dtype=np.int64),
                    "strand": np.full(strand_starts.size, strand, dtype=object),
                }
            )
        )
    if not parts:
        return pd.DataFrame([], columns=BED6_COLUMNS)
    return pd.concat(parts, ignore_index=True)


def locate_motif_in_sequence(
    chrom,
    sequence,
    motif,
    max_mismatches=0,
    both_strands=True,
    match_backend="auto",
):
    """Locate a motif in one sequence and return BED6-style hits.

    A thin wrapper over :func:`locate_motif_starts`. Rows are sorted by
    strand (``+`` first) and start; ``end = start + len(motif)``, ``name`` is
    the uppercase motif, and ``score`` is 0.

    Parameters
    ----------
    chrom : str
        Chromosome or record name used in returned BED rows.
    sequence : str
        Sequence to scan. Matching is case-insensitive.
    motif : str
        DNA motif to locate.
    max_mismatches : int
        Maximum Hamming substitutions allowed.
    both_strands : bool
        If true, also report reverse-strand hits (``"-"`` rows).
    match_backend : {"auto", "numba", "python"}
        Matching engine; see :func:`locate_motif_starts`.

    Returns
    -------
    pandas.DataFrame
        BED6-style rows with columns ``chrom``, ``start``, ``end``, ``name``,
        ``score``, and ``strand``.
    """
    starts = locate_motif_starts(
        sequence,
        motif,
        max_mismatches=max_mismatches,
        both_strands=both_strands,
        match_backend=match_backend,
    )
    return bed_from_starts(chrom, motif, starts)


def locate_motif(
    fasta_fname,
    motif,
    max_mismatches=0,
    both_strands=True,
    match_backend="auto",
):
    """Locate a motif in a FASTA file and return BED6-style hits.

    This compatibility helper scans every FASTA record and materializes all
    hits. Prefer :class:`FastaMotifSiteProvider` for whole-genome correction,
    because it uses indexed FASTA access and emits one chromosome's hits at a
    time.
    """
    rows = []
    fasta = open_fasta(fasta_fname)
    for chrom in fasta.keys():
        sequence = fasta[chrom][:]
        bed = locate_motif_in_sequence(
            chrom,
            sequence,
            motif,
            max_mismatches=max_mismatches,
            both_strands=both_strands,
            match_backend=match_backend,
        )
        rows.extend(bed.to_records(index=False).tolist())
    return _sort_bed(rows)


@dataclass(frozen=True)
class MotifSpec:
    """Parsed motif configuration for one sequential correction pass."""

    motif: str
    max_mismatches: int = 0


@dataclass
class FastaMotifSiteProvider:
    """Locate motif BED rows for one chromosome at a time.

    The provider reads only the requested FASTA record with pyfaidx and returns
    an ordered list of BED6 DataFrames, one per motif spec. That preserves
    kackle's sequential short-then-long correction behavior without holding
    whole-genome motif locations in memory.
    """

    fasta_fname: str
    motif_specs: list[MotifSpec | str]
    both_strands: bool = True
    match_backend: MatchBackend = "auto"
    bed6_fnames: list[str] | None = None
    _written_chroms: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self):
        """Normalize backend choices and motif specs after construction."""
        self.match_backend = resolve_match_backend(self.match_backend)
        # Build (or refresh) the .fai up front, in the calling process, so
        # concurrent chromosome workers never race to write it.
        thread_fasta(self.fasta_fname)
        parsed_specs = []
        for spec in self.motif_specs:
            if isinstance(spec, MotifSpec):
                parsed_specs.append(spec)
            else:
                motif, max_mismatches = parse_motif_spec(spec)
                parsed_specs.append(MotifSpec(motif.upper(), max_mismatches))
        self.motif_specs = parsed_specs
        if self.bed6_fnames is not None:
            if len(self.bed6_fnames) != len(self.motif_specs):
                raise ValueError("bed6_fnames must match motif_specs length")
            for bed6_fname in self.bed6_fnames:
                open(bed6_fname, "w").close()

    def _record(self, chrom):
        fasta = thread_fasta(self.fasta_fname)
        if chrom not in fasta:
            raise KeyError(f"Chromosome {chrom!r} not found in {self.fasta_fname}")
        return fasta[chrom]

    def sequence(self, chrom, start=None, end=None):
        """Return the requested chromosome sequence, or ``[start, end)`` of it."""
        record = self._record(chrom)
        if start is None and end is None:
            return record[:]
        return record[start:end]

    @property
    def max_motif_length(self):
        """Length of the longest motif, i.e. the farthest a minus anchor sits from its start."""
        return max(len(spec.motif) for spec in self.motif_specs)

    def starts_for_region(self, chrom, start, end, strands=None):
        """Like :meth:`starts_for_chrom`, for the windows starting in ``[start, end)``.

        Starts are absolute chromosome positions. The sequence is read from
        ``start`` to ``end + max_motif_length - 1`` (clipped to the record),
        so the result equals ``starts_for_chrom`` filtered to that range.
        """
        length = len(self._record(chrom))
        start = max(0, int(start))
        end = min(int(end), length)
        if end <= start:
            empty = np.empty(0, dtype=np.int64)
            return [{s: empty for s in STRANDS} for _ in self.motif_specs]
        sequence = self.sequence(
            chrom, start, min(length, end + self.max_motif_length - 1)
        )
        spec_starts = self._locate(sequence, strands)
        del sequence
        for starts in spec_starts:
            for strand, arr in starts.items():
                starts[strand] = arr[arr < end - start] + start
        return spec_starts

    def _locate(self, sequence, strands):
        if strands is None:
            strands = STRANDS if self.both_strands else ("+",)
        elif not self.both_strands:
            strands = tuple(s for s in strands if s == "+")
        encoded = _encode_for(sequence, self.match_backend)
        del sequence  # free the str copy; only the encoded bytes are scanned
        return [
            locate_motif_starts(
                None,
                spec.motif,
                max_mismatches=spec.max_mismatches,
                strands=strands,
                match_backend=self.match_backend,
                _encoded=encoded,
            )
            for spec in self.motif_specs
        ]

    def starts_for_chrom(self, chrom, strands=None):
        """Return one ``{strand: starts}`` dict per motif spec, in spec order.

        Starts are sorted int64 arrays (see :func:`locate_motif_starts`). Pass
        ``strands=("+",)`` or ``("-",)`` to search only one strand; a strand
        that isn't searched, or any minus strand when ``both_strands`` is
        false, gets an empty array.
        """
        return self._locate(self.sequence(chrom), strands)

    def beds_from_starts(self, chrom, spec_starts):
        """BED6 tables (one per motif spec) from ``starts_for_chrom`` output."""
        return [
            bed_from_starts(chrom, spec.motif, starts)
            for spec, starts in zip(self.motif_specs, spec_starts)
        ]

    def write_bed6(self, chrom, spec_starts):
        """Append this chromosome's sites to ``bed6_fnames`` once (no-op otherwise)."""
        if self.bed6_fnames is None or chrom in self._written_chroms:
            return
        for bed, bed6_fname in zip(
            self.beds_from_starts(chrom, spec_starts), self.bed6_fnames
        ):
            bed.to_csv(bed6_fname, sep="\t", header=False, index=False, mode="a")
        self._written_chroms.add(chrom)

    def for_chrom(self, chrom):
        """Return ordered motif BED tables for a single chromosome.

        Also appends them to ``bed6_fnames`` the first time a chromosome is
        seen. Callers that only need coordinates should use
        :meth:`starts_for_chrom`, which skips building the tables.
        """
        spec_starts = self.starts_for_chrom(chrom)
        self.write_bed6(chrom, spec_starts)
        return self.beds_from_starts(chrom, spec_starts)


def locate_motif_specs(
    fasta_fname,
    motif_specs,
    both_strands=True,
    match_backend="auto",
):
    """Locate an ordered list of motif specs in a FASTA file.

    The returned list preserves ``motif_specs`` order so callers can apply
    correction passes sequentially, matching the original pipeline semantics.
    This materializes all hits and is intended mostly for tests and library
    callers that explicitly want BED-like DataFrames.
    """
    motif_beds = []
    for spec in motif_specs:
        motif, max_mismatches = parse_motif_spec(spec)
        motif_beds.append(
            locate_motif(
                fasta_fname,
                motif,
                max_mismatches=max_mismatches,
                both_strands=both_strands,
                match_backend=match_backend,
            )
        )
    return motif_beds
