# R13 step 0(iii) — the distinct-bytes probe: cache dedup is real, but only past a depth crossover

`docs/tasks/red-october.md` R13, step 0(iii): "the real GQA layout (nKV heads shared by nH query
heads) against an MHA-expanded layout with nH distinct KV heads — identical issued work, nH/nKV×
the distinct bytes. If the two read the same, the caches are not deduping the group... if GQA is
markedly faster, the hardware already dedups... It says which end of the band to expect before a
line of assembly exists."

**Bottom line: the answer is depth-dependent, not a single yes/no. At K∈{128,512} the two layouts
run within 1-2% of each other — the CPU cache already dedups the group's repeated reads at these
depths, and there is no bandwidth win left for an explicit K/V-staging kernel to recover there. At
K∈{2048,3900} — the depths R2/R13's own decision cells are actually registered on — GQA is 2.1-3.3×
faster (0.5B) / 3.0-4.9× faster (1.5B) than the expanded layout. Past a crossover somewhere between
K=512 and K=2048, the working set stops fitting in cache and every query head genuinely pays real
memory traffic for bytes its group-mates already touched. "The top of the band is live" at exactly
the depths this brief cares about — an explicit-sharing kernel should expect its largest win where
the µop-sharing argument (§2.2, `docs/measurements/r13-attn-category-split-2026-09-19.md`) also
concentrates, not a competing mechanism but a reinforcing one.**

## Method

New, permanent, standalone benchmark: `decoder.BenchmarkAttnDistinctBytes`
(`decoder/attn_distinct_bytes_bench_test.go`) — no correctness claim, no golden, touches no
production file (only `_test.go`, so no parity-manifest interaction at all). Two arms, same per-head
QKᵀ (`linalg.MatmulQKAcc64`) + scores·V (`linalg.MatmulAVAcc64`) calls, same `nH` heads, same `hd`,
same `nKeys` — only the keys/vals buffer layout and addressing differ:

- **`gqa_real`**: buffer row width `nKV*hd` (the real, shipped layout); heads sharing a KV head
  (`head/group`) read the identical `hd`-wide slice of every row.
- **`mha_expanded`**: buffer row width `nH*hd`; every head reads its own private, distinct `hd`-wide
  slice — same issued work, `nH/nKV×` the *distinct* bytes touched.

Synthetic random data (seeded, via the existing `randF32` test helper) — this is a memory-access-
pattern probe, not a model. Real head shapes (`NumHeads`/`NumKVHeads`/`HeadDim`, so the real group
size `G`) come from `loadBenchModel()`'s own config. `-benchtime=200x` per cell, both models, same
machine/session as the category-split record (same thermal-drift caveat on absolute ns applies —
see that record's Finding 2 — but a same-run ratio between two arms measured back-to-back is far
more robust to a uniform slowdown than an absolute number).

## Data

Ratio = expanded / gqa (>1 means GQA is faster — cache is NOT fully absorbing the redundant reads):

| K | 0.5B gqa (ns) | 0.5B expanded (ns) | 0.5B ratio | 1.5B gqa (ns) | 1.5B expanded (ns) | 1.5B ratio |
|---|---|---|---|---|---|---|
| 128 | 22,583 | 22,705 | 1.005× | 38,584 | 38,514 | 0.998× |
| 512 | 89,278 | 90,475 | 1.013× | 153,942 | 156,599 | 1.017× |
| 2048 | 355,090 | 743,200 | **2.093×** | 609,912 | 1,817,242 | **2.980×** |
| 3900 | 675,625 | 2,259,326 | **3.344×** | 1,156,667 | 5,676,586 | **4.908×** |

## Reading

**K=128/512: ratio ≈ 1.00-1.02×, within measurement noise.** The cache dedups the group's reads at
these depths on both models — an explicit K/V-staging kernel would buy nothing here beyond the
µop-sharing win the category-split record already quantifies. This matches §2.2's own reading 3
("a cache can dedup a group's reads in hardware") — true, but only up to a point this probe locates
for the first time.

**K=2048/3900: ratio climbs sharply, and further on the larger model.** 1.5B's ratio at K=3900
(4.91×) exceeds 0.5B's (3.34×) — consistent with a genuine cache-capacity crossover: 1.5B's `hd`
is larger (more bytes per key per head), so its expanded-layout working set exceeds cache capacity
at a lower K, and by K=3900 both models' expanded arm is well past whatever cache level holds the
real (smaller, `nKV*hd`-wide) GQA working set. This is a plausible mechanism consistent with the
data, not confirmed against a direct cache-miss counter (no `perf`/cache-counter access used here —
a future pass could tighten this with `powermetrics` or equivalent, same instrument gap as the
peer-depth record's Finding 2).

**Direction for the Build.** R13's Build section as currently scoped (`MatmulQKAcc64Group`/
`MatmulAVAcc64Group`) shares the K/V *load* across a group implicitly, by having one kernel call
walk all `G` heads' dots/folds per loaded key/dim rather than `G` separate calls each re-reading the
same bytes — which is exactly the mechanism this probe says pays off at depth. The two probes this
step 0 ran (this one and the category split) point the same way for the same reason: **both softmax's
fixed per-key cost and the QK/AV bandwidth-vs-cache crossover are properties of what a query head
pays PER KEY, not of the group structure itself — grouping shares the group-common part of that
per-key cost (bytes here, µops there) but leaves softmax and the caches'-already-doing-it-at-shallow-
K regime untouched.** Concretely: expect the Build's win to be small-to-nil below roughly K=512-1024
and to grow with depth past whatever the real crossover point turns out to be under the real kernel
(this probe used isolated `MatmulQKAcc64`/`MatmulAVAcc64` calls, not the full tiled/pooled
`attendBatchedHeads` path, so the exact crossover K may shift some once measured in situ).

## What is and isn't established

**Established:** a real, reproducible, depth-dependent crossover in whether GQA's shared reads are
already cache-absorbed, on both models measured, in the same direction and of increasing magnitude
with `hd`. **Not established:** the crossover's exact location (only 4 depths were sampled; the
transition happens somewhere in (512, 2048], not pinned tighter than that); whether it holds inside
the real `attendBatchedHeads` tiled/pooled path (this probe isolates the kernel calls only, per the
brief's own "kernel A/B bench" framing for this step); a direct cache-miss-counter confirmation of
the cache-capacity mechanism proposed above.

## Next step

R13's step 0 is now complete: (i) the peer depth row (not yet benchmarks.md-quality — re-run
needed), (ii) the category split (softmax caps the Build's ceiling to ~1.39-1.54×, not 1.8-1.85×),
(iii) this probe (the top of the band is live, but only past roughly K=1024-2048). All three point
toward the same registered decision depth (K=3900, the 1.5B shape) being exactly where this Build
would pay off most — and where softmax, left untouched, would cap it. Both are now real inputs to
the Build's own decision rule, not open questions step 0 was supposed to answer.
