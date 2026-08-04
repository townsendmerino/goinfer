# Task: bit-identical high-occupancy decode attention (Campaign A, split-KV)

> Scoping doc. Opened 2026-08-04 from the A1-reprofile (`ollama-chase.md` §A2) + D4 (§D4).
> Status: **design + build in progress.** Bit-identity argument settled; kernels being written.

## The problem (measured, not inferred)

A1 coalesced the decode attention K-read (`ollama-chase.md` §A1) — 232.7 → 134.2 µs, L1TEX 71 →
38%. The A1-reprofile then showed the bound **moved to occupancy**: the decode `attn_batched` at
M=1 launches **one block per query head = 12 blocks** on a 40-SM card. Waves/SM 0.04, achieved
occupancy **11.9%** (theoretical 87.5% — *not* register/shared limited), 93% no-eligible-warp,
scoreboard-stall dominated. Nothing is throughput-saturated (DRAM 9.5%, Compute 5.4%). It is
latency-bound purely because there are too few blocks to hide memory latency.

D4 confirmed Ollama holds ~188 tok/s at 2048 by running **flash attention** (`flash_attn_ext`,
parallel over KV, online-softmax). FA fills the SMs by splitting the key dimension. We want the
same occupancy — **without** FA's online rescale, which is not bit-exact.

## The bit-identity constraint (this is the whole design)

The current kernel's numerically-sensitive step is the V-weighted sum, computed **per output dim
`d` by a single thread as a strict sequential left-fold over keys**:

```
acc = 0;  for (s = winStart; s < nKeys; s++)  acc += sc[s] * v[s][d];   ctx[d] = acc * inv;
```

Float addition is **not associative**. The sequential fold is the specific left-leaning tree
`(((t0+t1)+t2)+t3)…`. Any *contiguous key-split with per-chunk partials* combines as a different
tree — e.g. split `[t0,t1][t2,t3]` gives `(t0+t1)+(t2+t3)` ≠ `((t0+t1)+t2)+t3`. So **splitting the
reduction across the key dimension is not bit-identical** (this is exactly why FA, which does split
+ online-rescale, cannot be bit-exact). Growing the block size is also out: it changes the strided
thread→key partition and the reduction-tree depth, so the softmax denominator sum changes in the
last ULP → the byte-identical-decode gate fails.

**The split that IS bit-identical: split the *independent* axes, never the reduction.**

The kernel has two O(nKeys·hd) passes (the expensive ones, both occupancy-starved) and two
O(nKeys) passes (cheap). Their parallelism-vs-bit-identity properties:

| pass | cost | over what axis is it independent? | bit-identical to split? |
|---|---|---|---|
| 1. scores `sc[s] = scale·Σ_d q[d]·k[s][d]` | O(nKeys·hd) | **per key `s`** (each dot is independent) | **yes** — no reduction across `s` |
| 2. max `mx = max_s sc[s]` | O(nKeys) | reduction, but **max is reorder-safe** | yes (assoc+comm, exact) |
| 3. denom `Σ_s exp(sc[s]-mx)` | O(nKeys) | order-dependent **sum** | only if the partition+tree is byte-identical to today's |
| 4. V-sum `acc_d = Σ_s sc[s]·v[s][d]` | O(nKeys·hd) | **per output dim `d`** (each `d` is its own sequential fold) | **yes** — keep each `d`'s fold whole, distribute `d`s across blocks |

The insight: the two **expensive** passes are each independent along an axis that is *not* the
reduction axis — scores over `s`, V-sum over `d`. Distributing *those* axes across many blocks adds
occupancy while every order-dependent fold stays whole and in-order. The two cheap O(nKeys) passes
keep the reduction; we simply replicate today's exact `blockDim=128` strided partition + tree so
their sums are byte-for-byte what they are now.

## Design — a 3-kernel split (implicit global barrier between launches)

Per (head, token), replacing the single 12-block `attn_batched(M=1)` launch:

1. **`splitkv_scores`** — grid `nH × ceil(nWin/Stile)`, block 128. Each block computes `sc[s] =
   scale · Σ_d q·k` for its key-tile, writes raw scores to a global `sc[nH][nWin]` scratch. No
   reduction ⇒ trivially bit-identical. **High occupancy** (12 → 12·⌈nWin/Stile⌉ blocks).
2. **`splitkv_softmax`** — grid `nH`, block **128** (must match today), reads `sc[]`, does the
   max-reduce + `exp` + denom-sum with the **exact** strided partition and tree of the current
   kernel, writes back `sc[s] ← exp(sc[s]-mx)` and `inv = 1/Σ`. O(nKeys), cheap — 12 blocks is fine
   here (it's not the bottleneck). Byte-identical max + denom by construction.
3. **`splitkv_vsum`** — grid `nH × ceil(hd/Dtile)`, block `Dtile`. Each thread owns one output dim
   `d` and does the **whole** sequential fold `Σ_s sc[s]·v[s][d]` in ascending `s` (identical to
   today), then `·inv`. **High occupancy** (12 → 12·⌈hd/Dtile⌉ blocks, e.g. Dtile=32 → 48, Dtile=16
   → 96).

Both expensive passes now fill the GPU; both cheap passes and every order-dependent fold are
untouched → **decode stays byte-identical** (the same contract A1 held).

### Shared with B1 (prefill query-tiling)

The score-tile and V-tile kernels are the same primitive B1 needs — prefill's redundant-re-read
problem is served by tiling K/V across a query *block*, and these kernels already tile K/V. Decode
(M=1) stresses occupancy (split over keys/dims); prefill (M>1) stresses re-reads (share a K/V tile
across query rows). Build the tiles here; let B1 reuse them. Hence "shared tiled/split-KV kernel."

## Cost / risk

- **Launch overhead:** 3 launches/attention/layer/token instead of 1. Decode already dispatches
  ~13/layer; +2 is small, and the occupancy win is the whole point. Measure total, not the sub-bucket.
- **Global scratch:** `sc[nH][nWin]` f32 ≈ 12·2048·4 = 98 KB/token, reused across layers. Trivial.
- **New kernels, own file** (`decode_splitkv.cu`) — audited `glue.ptx` and `moe.ptx` untouched, per
  the repo rule. Guarded by `prefillReady` like A1; glue `attention` stays the fallback.

## Gate plan

- **Bit-identity micro-gate:** `splitkv` composition == the current `attention`/`attn_batched(M=1)`
  bit-for-bit, swept over nKeys {1, 128, 999, 2048}, windowed and non-windowed, hd {128, 256},
  GQA and non-GQA — the same matrix A1's `TestAttnBatched_bitIdentical` used.
- **E2E:** wire into the decode launch behind a flag, run `TestPrefillLast_e2e` / real qwen3 /
  gemma3 decode → KV + logits + 64-token decode **byte-identical**.
- **Perf:** re-`ncu` occupancy (target: Waves/SM ≫ 0.04, No-Eligible ≪ 93%) + `TestDecodeDepth
  Throughput` at 2048 (target: 133.5 → toward the arithmetic's ~214 tok/s / Ollama's ~188).

## Results — landed, opt-in (`GOINFER_SPLITKV_ATTN`), bit-identical (2026-08-04)

**Bit-identity gate green** (`TestSplitKV_bitIdentical`): same prefill(2048)+24-token greedy decode
on the real 1.5B, split-KV off vs on → decode stream + final logits (151936) **byte-identical**.

**Throughput A/B** (`TestDecodeDepthThroughput`, 1.5B):

| depth | attn_batched (A1) | split-KV | |
|---|---|---|---|
| 128 | 226.7 | 219.8 | −3% (3-launch overhead where attention is cheap) |
| **2048** | **133.4** | **160.1** | **1.20×** |

Long-context decode is now **99.5 → 160.1 tok/s = 1.61×** over the original glue (A1 ×1.34, split-KV
×1.20). Gap to Ollama's ~188: **1.41× → 1.17× behind.** Not yet the arithmetic's ~214 — the V-sum
ceiling below is why.

**ncu (the mechanism, confirmed):**

| kernel | grid | dur | occupancy |
|---|---|---|---|
| `splitkv_scores` | 204 | 28.7 µs | **44.3%** (was ~12%) — parallelized fully |
| `splitkv_softmax` | 12 | 5.0 µs | 12% — cheap by design, fine |
| `splitkv_vsum` | 48 | **50.0 µs** | **3.8%** — the new ceiling |

Total kernel 83.7 µs vs attn_batched 134.2 µs. The scores parallelized as designed; **the V-sum is
now the bottleneck** — its bit-identity ceiling is nH·hd parallelism (1536 threads = 48 warps),
latency-starved at 3.8% occupancy.

### Next increment (toward parity): bit-identical V-sum ILP unroll

The V-sum's `for s: acc += sc[s]·v[s][d]` is a dependent chain on `acc`, but the **loads** of
`v[s][d]` for different `s` are independent. **Unrolling the s-loop** (issue N loads, then N adds in
ascending-s order) exposes memory-level parallelism per thread → hides latency at the unavoidable low
occupancy, while the adds stay in the exact fold order ⇒ **still bit-identical.** This is the lever to
push 160 → toward ~188+. (Also: gate split-KV behind a context threshold before default-on, so the
−3% shallow case takes the attn_batched path.)

## Why this one is worth it (recap from §4)

It is the only deficit where the arithmetic says parity is reachable, and occupancy is a *solvable*
bound (unlike the prefill tensor-core ceiling, §7). Flipping "parity at short context" to "parity
across context lengths" is a qualitative win on the axis where users actually spend time.
