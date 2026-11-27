"""Adapter look-alike sites: candidate artifact sites derived from an adapter sequence.

The artifact kackle corrects comes from primer mispriming on stretches of a
real fragment that resemble the 3' adapter. The read then ends on the base
just before the look-alike (the artifact base). A position is a candidate
when the RNA-sense sequence starting there matches the adapter's 5' end for
long enough (see :func:`kackle.computation.adapter_match_lengths`; the
pipeline runs the faster :func:`kackle.computation.adapter_candidates`). Leading
``N``s (a UMI at the adapter's 5' end) are skipped: they stay part of the
look-alike window, so the artifact base is still the base before it, but
they are not compared and don't count toward match length. The match
length, and optionally its number of mismatches, set the site's screening
threshold, since a longer or cleaner match is a stronger artifact.

:class:`AdapterSiteProvider` exposes the same ``starts_for_chrom`` /
``starts_for_region`` interface as :class:`kackle.motifs.FastaMotifSiteProvider`,
with one pseudo-spec per threshold tier, so chunk planning and region reads
work unchanged. ``pass_thresholds`` tells the correction pipeline to screen
all tiers together in one pass (:func:`kackle.computation.correct_adapter_sites`).
"""

from dataclasses import dataclass, field

import numpy as np

from kackle.computation import adapter_candidates, sequence_codes
from kackle.motifs import MotifSpec, thread_fasta

STRANDS = ("+", "-")

#: TruSeq small-RNA 3' adapter (5' end), the usual PRO-seq 3' adapter.
DEFAULT_ADAPTER = "TGGAATTCTCGG"
#: Shortest adapter prefix that makes a candidate, by number of mismatches in it.
DEFAULT_MIN_MATCH = (3, 5)
#: ``(min match length, max mismatches or None for any, fold threshold or
#: None for the provider's ``threshold``, min reads or None for the run's
#: ``min_reads``)`` tiers. Exact matches of 3 bases (bare TGG) at
#: ``threshold``; exact 4+ (TGGA...) at 7x; any match of 8+ at 5x, 10+ at 3x.
#: The exact-3 tier keeps bare TGG out of the base tier, whose sites (short
#: matches with a mismatch) need ``DEFAULT_BASE_MIN_READS`` reads.
DEFAULT_TIERS = ((3, 0, None, None), (4, 0, 7.0, None), (8, None, 5.0, None), (10, None, 3.0, None))
#: Minimum artifact-base count for sites that qualify for no tier.
DEFAULT_BASE_MIN_READS = 2

IUPAC = {
    "A": "A", "C": "C", "G": "G", "T": "T", "U": "T",
    "R": "AG", "Y": "CT", "S": "CG", "W": "AT", "K": "GT", "M": "AC",
    "B": "CGT", "D": "AGT", "H": "ACT", "V": "ACG", "N": "ACGT",
}
_BIT = {"A": 1, "C": 2, "G": 4, "T": 8}

_SEQ_MASK = np.zeros(256, dtype=np.uint8)
for _base, _bit in _BIT.items():
    _SEQ_MASK[ord(_base)] = _SEQ_MASK[ord(_base.lower())] = _bit
# Complement of a 4-bit mask: A<->T (1<->8), C<->G (2<->4).
_COMPLEMENT = np.array(
    [sum(1 << (3 - b) for b in range(4) if m >> b & 1) for m in range(16)], dtype=np.uint8
)


def adapter_masks(adapter):
    """IUPAC adapter string -> uint8 bitmask array."""
    adapter = adapter.upper()
    bad = sorted(set(adapter) - set(IUPAC))
    if not adapter or bad:
        raise ValueError(f"Adapter must be non-empty IUPAC nucleotides; got {bad or 'empty'}")
    return np.array([sum(_BIT[b] for b in IUPAC[c]) for c in adapter], dtype=np.uint8)


def sequence_masks(sequence):
    """Sequence string -> uint8 bitmasks (0 for anything but A/C/G/T, any case)."""
    raw = np.frombuffer(sequence.encode("ascii", "replace"), dtype=np.uint8)
    return sequence_codes(raw, _SEQ_MASK)


def parse_min_match(text):
    """``"3,5"`` -> ``(3, 5)``: shortest candidate prefix with 0, 1, ... mismatches."""
    values = tuple(int(v) for v in str(text).split(",") if v.strip())
    if not values or values[0] < 1 or any(b < a for a, b in zip(values, values[1:])):
        raise ValueError("min match lengths must be positive and non-decreasing")
    return values


def _tier_key(tier):
    return (tier[0], float("inf") if tier[1] is None else tier[1])


def normalize_tiers(tiers):
    """Sorted ``(length, max_mm or None, fold, min_reads or None)`` tiers.

    ``(length, fold)`` means any mismatches; a 3-tuple leaves ``min_reads``
    to the run's ``min_reads``. A ``fold`` of ``None`` means the provider's
    ``threshold``.
    """
    out = []
    for tier in tiers:
        if len(tier) == 2:
            tier = (tier[0], None, tier[1])
        if len(tier) == 3:
            tier = (*tier, None)
        length, max_mm, fold, min_reads = tier
        if (
            int(length) < 1
            or (max_mm is not None and int(max_mm) < 0)
            or (fold is not None and float(fold) <= 0)
            or (min_reads is not None and int(min_reads) < 1)
        ):
            raise ValueError(f"bad tier {tier!r}")
        out.append((
            int(length),
            None if max_mm is None else int(max_mm),
            None if fold is None else float(fold),
            None if min_reads is None else int(min_reads),
        ))
    out.sort(key=_tier_key)
    if any(_tier_key(a) == _tier_key(b) for a, b in zip(out, out[1:])):
        raise ValueError("tiers must differ in match length or mismatch limit")
    return tuple(out)


def parse_tiers(text):
    """``"3/0,4/0:7,8:5:2"`` -> ``((3, 0, None, None), (4, 0, 7.0, None), (8, None, 5.0, 2))``.

    ``LEN:FOLD`` applies to matches of at least LEN bases with any number of
    mismatches; ``LEN/MAXMM:FOLD`` only to those with at most MAXMM. A
    missing or empty FOLD (``3/0``, ``3/0::1``) means the threshold
    (``-r``). An optional ``:MINREADS`` sets the tier's minimum
    artifact-base count (otherwise the run's ``min_reads``). An empty string
    gives no tiers.
    """
    tiers = []
    for item in str(text).split(","):
        if not item.strip():
            continue
        try:
            cond, *rest = item.split(":")
            if len(rest) > 2:
                raise ValueError
            fold, min_reads = (rest + ["", ""])[:2]
            length, _, max_mm = cond.partition("/")
            tiers.append((
                int(length),
                int(max_mm) if max_mm.strip() else None,
                float(fold) if fold.strip() else None,
                int(min_reads) if min_reads.strip() else None,
            ))
        except ValueError:
            raise ValueError(f"tier {item!r} is not LEN[/MAXMM][:FOLD[:MINREADS]]") from None
    return normalize_tiers(tiers)


@dataclass
class AdapterSiteProvider:
    """Locate adapter look-alike sites one chromosome (or region) at a time.

    Sites come back as one ``{strand: starts}`` dict per threshold tier
    (``motif_specs`` holds one pseudo-spec per tier, each with the full
    adapter as its motif). A site's window is the ``len(adapter)`` bases of
    the look-alike, so a minus anchor is ``start + len(adapter)``, as for
    motif specs. Leading ``N``s in the adapter are in the window but are
    skipped by the matcher, so lengths (``min_match``, tier lengths) count
    from the first fixed base. A site qualifies for a tier when some adapter prefix of at
    least the tier's length matches with at most its mismatches (so an exact
    TGGA followed by a mismatch still qualifies for ``4/0``), and goes to the qualifying
    tier with the lowest fold (the first one on a tie). Tier 0 holds sites
    that qualify for none and uses ``threshold`` (as does a tier whose fold
    is ``None``) and ``base_min_reads``. A tier's minimum read count
    (``pass_min_reads``; ``None`` means the run's ``min_reads``) does not
    affect which tier a site goes to. Only windows that fit
    entirely inside the chromosome are reported.
    """

    fasta_fname: str
    adapter: str = DEFAULT_ADAPTER
    min_match: tuple = DEFAULT_MIN_MATCH
    tiers: tuple = DEFAULT_TIERS
    threshold: float = 10.0
    base_min_reads: object = DEFAULT_BASE_MIN_READS
    both_strands: bool = True
    bed6_fnames: None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.adapter = self.adapter.upper()
        self._masks = adapter_masks(self.adapter)
        self._rc_masks = _COMPLEMENT[self._masks]
        #: Leading ``N``s (a UMI): in the window, but not compared or counted.
        self.skip = len(self.adapter) - len(self.adapter.lstrip("N"))
        if self.skip == len(self.adapter):
            raise ValueError("adapter is all N; it needs fixed bases to match")
        self.min_match = tuple(int(v) for v in self.min_match)
        if self.min_match[0] > len(self.adapter) - self.skip:
            raise ValueError("min match length is longer than the adapter (after leading Ns)")
        self.tiers = normalize_tiers(self.tiers)
        if self.base_min_reads is not None:
            if isinstance(self.base_min_reads, bool) or int(self.base_min_reads) != self.base_min_reads \
                    or self.base_min_reads < 1:
                raise ValueError("base_min_reads must be an integer >= 1 or None")
            self.base_min_reads = int(self.base_min_reads)
        thread_fasta(self.fasta_fname)  # build the .fai in the parent
        spec = MotifSpec(self.adapter, len(self.min_match) - 1)
        self.motif_specs = [spec] * (1 + len(self.tiers))

    @property
    def pass_thresholds(self):
        """Fold threshold per tier, in ``motif_specs`` order; one combined pass."""
        return (float(self.threshold),) + tuple(self._fold(t) for t in self.tiers)

    def _fold(self, tier):
        return float(self.threshold) if tier[2] is None else tier[2]

    @property
    def pass_min_reads(self):
        """Minimum artifact-base count per tier (``None``: the run's ``min_reads``)."""
        return (self.base_min_reads,) + tuple(t[3] for t in self.tiers)

    @property
    def max_motif_length(self):
        return len(self.adapter)

    def _tier_of(self, lengths):
        """Tier per site from its ``(n, budgets)`` match lengths by mismatch budget."""
        tier = np.zeros(lengths.shape[0], dtype=np.int64)
        best = np.full(lengths.shape[0], np.inf)
        top = lengths.shape[1] - 1
        for i, spec in enumerate(self.tiers, start=1):
            min_len, max_mm = spec[0], spec[1]
            fold = self._fold(spec)
            budget = top if max_mm is None else min(max_mm, top)
            ok = lengths[:, budget] >= min_len
            ok &= fold < best
            tier[ok] = i
            best[ok] = fold
        return tier

    def _split(self, starts, lengths):
        tier = self._tier_of(lengths)
        return [starts[tier == i] for i in range(len(self.motif_specs))]

    def starts_for_region(self, chrom, start, end, strands=None):
        """Sites whose window starts in ``[start, end)``, one dict per tier."""
        wanted = STRANDS if strands is None else tuple(strands)
        k = len(self.adapter)
        fasta = thread_fasta(self.fasta_fname)  # per-thread handle: safe across threads
        if chrom not in fasta:
            raise KeyError(f"Chromosome {chrom!r} not found in {self.fasta_fname}")
        length = len(fasta[chrom])
        start = max(0, int(start))
        end = min(int(end), length - k + 1)
        empty = np.empty(0, dtype=np.int64)
        out = [{s: empty for s in STRANDS} for _ in self.motif_specs]
        if end <= start:
            return out
        seq = sequence_masks(str(fasta[chrom][start : end + k - 1]))
        min_match = np.array(self.min_match, dtype=np.int64)
        by_strand = {}
        if "+" in wanted:
            pos, lengths = adapter_candidates(seq, self._masks, min_match, self.skip, False)
            by_strand["+"] = self._split(pos + start, lengths)
        if "-" in wanted and self.both_strands:
            # Matched on the reverse complement in place, with complemented
            # adapter masks; starts are genomic window starts, already sorted.
            pos, lengths = adapter_candidates(seq, self._rc_masks, min_match, self.skip, True)
            by_strand["-"] = self._split(pos + start, lengths)
        for strand, arrays in by_strand.items():
            for tier, arr in enumerate(arrays):
                out[tier][strand] = arr
        return out

    def starts_for_chrom(self, chrom, strands=None):
        """Sites of the whole chromosome, one ``{strand: starts}`` dict per tier."""
        return self.starts_for_region(chrom, 0, 1 << 62, strands)
