"""Motif location helpers for generating BED6-style kmer sites.

This module replaces the previous ``seqkit locate`` preprocessing step for the
short artifact motifs kackle corrects. It performs case-insensitive exact
matching after expanding a motif to all sequences within a Hamming-distance
neighborhood, then emits BED6-style rows with 0-based half-open coordinates.

The command-line workflow uses :class:`FastaMotifSiteProvider` with pyfastx and
ahocorasick-rs to locate motifs one chromosome at a time. That avoids
materializing whole-genome motif BEDs for the very common short artifact motifs.
"""

from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Literal

import pandas as pd


DNA_ALPHABET = ("A", "C", "G", "T")
RC_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")
BED6_COLUMNS = ["chrom", "start", "end", "name", "score", "strand"]
FastaBackend = Literal["auto", "python", "pyfastx"]
MatchBackend = Literal["auto", "python", "ahocorasick"]


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


def iter_fasta_records(path):
    """Yield ``(name, sequence)`` records from a FASTA file.

    The record name is the first non-whitespace token after ``>``. Sequence
    lines are concatenated as-is; callers are responsible for uppercasing or
    validating alphabets.
    """
    name = None
    chunks = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(chunks)
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if name is not None:
        yield name, "".join(chunks)


def _sort_bed(rows):
    """Return BED6 rows as a deterministically sorted DataFrame."""
    if not rows:
        return pd.DataFrame([], columns=BED6_COLUMNS)
    bed = pd.DataFrame(rows, columns=BED6_COLUMNS)
    return bed.sort_values(["chrom", "strand", "start", "end"]).reset_index(drop=True)


def resolve_match_backend(match_backend):
    """Resolve a motif matching backend name."""
    if match_backend == "auto":
        return "ahocorasick"
    if match_backend not in {"python", "ahocorasick"}:
        raise ValueError("Match backend must be 'auto', 'python', or 'ahocorasick'")
    if match_backend == "ahocorasick":
        try:
            import ahocorasick_rs  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "ahocorasick-rs is required for the default matching backend"
            ) from exc
    return match_backend


def resolve_fasta_backend(fasta_backend):
    """Resolve a FASTA sequence backend name."""
    if fasta_backend == "auto":
        return "pyfastx"
    if fasta_backend not in {"python", "pyfastx"}:
        raise ValueError("FASTA backend must be 'auto', 'python', or 'pyfastx'")
    if fasta_backend == "pyfastx":
        try:
            import pyfastx  # noqa: F401
        except ImportError as exc:
            raise ImportError("pyfastx is required for the default FASTA backend") from exc
    return fasta_backend


def _python_fasta_sequence(fasta_fname, chrom):
    """Read one FASTA record by name using the built-in parser."""
    for name, sequence in iter_fasta_records(fasta_fname):
        if name == chrom:
            return sequence
    raise KeyError(f"Chromosome {chrom!r} not found in {fasta_fname}")


def _ahocorasick_hits(sequence, patterns):
    """Yield overlapping starts for any pattern using ahocorasick-rs."""
    import ahocorasick_rs

    automaton = ahocorasick_rs.AhoCorasick(
        sorted(patterns),
        matchkind=ahocorasick_rs.MATCHKIND_STANDARD,
        store_patterns=False,
    )
    for _, start, _ in automaton.find_matches_as_indexes(sequence, overlapping=True):
        yield start


def _python_hits(sequence, patterns):
    """Yield overlapping starts for any pattern with repeated C-backed finds."""
    for pattern in patterns:
        start = sequence.find(pattern)
        while start != -1:
            yield start
            start = sequence.find(pattern, start + 1)


def locate_motif_in_sequence(
    chrom,
    sequence,
    motif,
    max_mismatches=0,
    both_strands=True,
    match_backend="ahocorasick",
):
    """Locate a motif in one sequence and return BED6-style hits.

    Forward-strand hits are reported by searching motif variants directly.
    Reverse-strand hits are reported by searching the reverse complements of
    those variants on the input sequence and labeling the row with ``"-"``.

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
        If true, search both motif variants and their reverse complements.
    match_backend : {"auto", "python", "ahocorasick"}
        Pattern matching engine. ``ahocorasick`` is the default and scans all
        expanded variants in one pass per strand. ``python`` is retained as a
        simple comparison backend using repeated C-backed ``str.find`` calls.

    Returns
    -------
    pandas.DataFrame
        BED6-style rows with columns ``chrom``, ``start``, ``end``, ``name``,
        ``score``, and ``strand``.
    """
    motif = motif.upper()
    sequence = sequence.upper()
    match_backend = resolve_match_backend(match_backend)
    variants = mismatch_variants(motif, max_mismatches)
    rc_variants = {variant.translate(RC_TABLE)[::-1] for variant in variants}
    rows = []
    seen = set()
    hit_fn = _ahocorasick_hits if match_backend == "ahocorasick" else _python_hits

    for start in hit_fn(sequence, variants):
        row_key = (chrom, start, start + len(motif), "+")
        if row_key in seen:
            continue
        seen.add(row_key)
        rows.append((chrom, start, start + len(motif), motif, 0, "+"))

    if both_strands:
        for start in hit_fn(sequence, rc_variants):
            row_key = (chrom, start, start + len(motif), "-")
            if row_key in seen:
                continue
            seen.add(row_key)
            rows.append((chrom, start, start + len(motif), motif, 0, "-"))

    return _sort_bed(rows)


def locate_motif(
    fasta_fname,
    motif,
    max_mismatches=0,
    both_strands=True,
    match_backend="ahocorasick",
):
    """Locate a motif in a FASTA file and return BED6-style hits.

    This compatibility helper scans every FASTA record and materializes all
    hits. Prefer :class:`FastaMotifSiteProvider` for whole-genome correction,
    because it uses indexed FASTA access and emits one chromosome's hits at a
    time.
    """
    rows = []
    for chrom, sequence in iter_fasta_records(fasta_fname):
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

    The provider reads only the requested FASTA record with pyfastx and returns
    an ordered list of BED6 DataFrames, one per motif spec. That preserves
    kackle's sequential short-then-long correction behavior without holding
    whole-genome motif locations in memory.
    """

    fasta_fname: str
    motif_specs: list[MotifSpec | str]
    both_strands: bool = True
    fasta_backend: FastaBackend = "pyfastx"
    match_backend: MatchBackend = "ahocorasick"
    bed6_fnames: list[str] | None = None
    _pyfastx_fasta: object | None = field(default=None, init=False, repr=False)
    _written_chroms: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self):
        """Normalize backend choices and motif specs after construction."""
        self.fasta_backend = resolve_fasta_backend(self.fasta_backend)
        self.match_backend = resolve_match_backend(self.match_backend)
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

    def sequence(self, chrom):
        """Return the requested chromosome sequence."""
        if self.fasta_backend == "pyfastx":
            if self._pyfastx_fasta is None:
                import pyfastx

                self._pyfastx_fasta = pyfastx.Fasta(str(self.fasta_fname))
            if chrom not in self._pyfastx_fasta:
                raise KeyError(f"Chromosome {chrom!r} not found in {self.fasta_fname}")
            return str(self._pyfastx_fasta[chrom])
        return _python_fasta_sequence(self.fasta_fname, chrom)

    def for_chrom(self, chrom):
        """Return ordered motif BED tables for a single chromosome."""
        sequence = self.sequence(chrom)
        beds = [
            locate_motif_in_sequence(
                chrom,
                sequence,
                spec.motif,
                max_mismatches=spec.max_mismatches,
                both_strands=self.both_strands,
                match_backend=self.match_backend,
            )
            for spec in self.motif_specs
        ]
        if self.bed6_fnames is not None and chrom not in self._written_chroms:
            for bed, bed6_fname in zip(beds, self.bed6_fnames):
                bed.to_csv(bed6_fname, sep="\t", header=False, index=False, mode="a")
            self._written_chroms.add(chrom)
        return beds


def locate_motif_specs(
    fasta_fname,
    motif_specs,
    both_strands=True,
    match_backend="ahocorasick",
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
