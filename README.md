# kackle

Kmer Artifact Correction (KACkle) -- A small python utility for eliminating k-mer artifacts caused by primer mismatching.

```bash
uv tool install git+https://github.com/adamyhe/kackle.git
```

Install optional accelerated motif-location backends with:

```bash
uv tool install 'git+https://github.com/adamyhe/kackle.git[fast-motifs]'
```

Run with precomputed BED6 motif sites:

```bash
kackle -i plus.bw -I minus.bw -b sites.bed6 -o out.plus.bw -O out.minus.bw -c chrom.sizes
```

Or let kackle locate the original artifact motifs directly from FASTA. By default,
FASTA mode corrects exact `TGG` matches first, then `TGGAA` matches with up to one
mismatch:

```bash
kackle -i plus.bw -I minus.bw -f genome.fa -o out.plus.bw -O out.minus.bw -c chrom.sizes
```

FASTA mode locates motif sites one chromosome at a time to avoid materializing
whole-genome BED tables for common short motifs. With the `fast-motifs` extra,
you can use indexed FASTA access and Aho-Corasick multi-pattern matching:

```bash
kackle -i plus.bw -I minus.bw -f genome.fa \
  --fasta-backend pyfastx --motif-match-backend ahocorasick \
  -o out.plus.bw -O out.minus.bw -c chrom.sizes
```

Custom motif order can be supplied with repeated `--motif KMER[:MISMATCHES]` flags.

To reproduce motif-backend timings:

```bash
uv run --extra fast-motifs python benchmarks/motif_backends.py
```
