# Metal W4A8 GEMV — Fable's optimization pass (response)

> Companion to `metal-gemv-optimization-ask.md`. This is Fable's returned analysis for
> breaking the ALU-bound int4-unpack wall on the two dominant kernels (gate/up ~42%,
> lm-head ~12%), both at ~3× the bandwidth floor. Rig: M1 Pro, 200 GB/s, 32-wide
> simdgroups. Baseline 58.8 tok/s; GO bar ~71.
>
> **⚠️ Transmission note:** the compilable **core-kernel preamble** — the `DOT8` macro,
> the full **Stage A** (no-repack) kernel, and the **Stage B `gemv_w4a8_t32`** top
> (signature + threadgroup activation staging + `Sg` group-sum precompute) — was
> truncated in transit. What is captured below is complete and verbatim: the Stage B
> **dot-loop core**, the repack layout, the per-matrix geometry, the **fused-argmax
> lm-head kernels**, the levers, estimates, priorities, and the certain/speculative
> split. The truncated preamble is being re-fetched / assembled — marked `‹FETCHING›`.

## Headline

- By Fable's numbers the rewrite **clears the GO bar even on the pessimistic edges:**
  **Stage A ~80 tok/s** (floor ~73), **Stage B ~88** (floor ~80), **+f16 scales ~90+**.
- It independently **validated the lm-head-argmax swing**: orthogonal, works on Stage A
  or B, tie-break = strict `>` / lower-index-wins (matches CPU first-max-wins). Complete
  MSL below.

## Recommended sequence (revised — do NOT jump to the repack)

1. **Stage A body swap (no repack):** `uint4` vector loads + stage the int8 activation
   into threadgroup memory (broadcast reads) + 8 simdgroups/threadgroup. **+12–18 tok/s**,
   hours of work, and it *validates the ALU-bound diagnosis* on the first measurement.
   Staging is the single biggest lever — deletes ~90% of the LSU activation-gather work.
2. **Fused block-argmax lm-head** (works on Stage A or B): **+4–7 tok/s**, low risk,
   readback 608 KB → 4 bytes.
3. **Stage B repack** (lane-per-row, one `uint4` = one 32-element scale group,
   `−8·S_g` pre-factored — integer-exact): **+5–8** on top of A, ~a day incl. Go packer.
4. **Indirect Command Buffers** for the 2.4 ms encode/commit gap (the 68→58.8 delta):
   bindings are static per token → encode the token once, replay per token. **+6–9**,
   orthogonal, purego-reachable, medium-confidence — verify compute-ICB barrier semantics.

## Stage B — GEMV core (lane-per-row, `uint4` = one scale group)

Dispatch: one **lane per output row**; a threadgroup owns `SG_PER_TG` simdgroups.

The hot loop (verbatim), one `uint4` load = one full 32-element scale group:

```metal
        uint4 w = wp[g << 5];                       // one full scale group: 4 words, 32 nibbles
        const threadgroup short4* a = As4 + (g << 3);
        short4 a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
        short4 a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];   // broadcast reads: all lanes, same addr
        int gi = -8 * Sg[g];                        // exact: sum((n-8)*a) == sum(n*a) - 8*S_g
        gi += DOT8(w.x, a0, a1);
        gi += DOT8(w.y, a2, a3);
        gi += DOT8(w.z, a4, a5);
        gi += DOT8(w.w, a6, a7);
        acc = fma(float(gi), sp[g << 5], acc);      // ONE float MAC per 32-element group
    }
    outv[row] = acc * asc[0];                       // coalesced 128B store per simdgroup
    // _bias:  outv[row] = acc*asc[0] + bias[row];   _resid: outv[row] += acc*asc[0];
```

Rationales (each tied to a bottleneck):
- **Lane-per-row** → 48 sequential group-iterations/lane, zero `simd_sum`, zero cross-lane
  traffic: fixes the thin-work/latency half of the diagnosis.
- **`uint4` = one scale group** → the load width *is* the parity structure: 1 load, 1 int
  group-sum, 1 scale, 1 `fma`; 512 B contiguous/simdgroup instruction.
- **Broadcast activations from threadgroup memory** → all 32 lanes read the same `short4`
  address (no bank conflicts, no device gather): kills the ~90%-of-LSU activation gather.
- **`−8·S_g` pre-factoring** → removes 32 integer subtracts per group per lane, *exactly*
  (integer identity). Range: `sum(n·a) ≤ 60,960`; `|8·S_g| ≤ 32,512`; `|gi| < 2¹⁷`, so
  `float(gi)` is exact, int32 never overflows.
- **Coalesced per-group scales** → removes the 4-way-redundant `srow[wi>>2]` gather.
- **`SG_PER_TG` compile-time knob** → gate/up: 560 tiles → 70 threadgroups of 256; lm-head:
  4748 → 594; qkv/o at `SG_PER_TG=2` → 32/24 threadgroups so all 16 cores stay fed.

Parity envelope (honest): per-group *terms* `float(gi)*scale` are bit-identical in value to
the current per-word math regrouped within a group (strictly *fewer* float roundings — 1
MAC/group vs 4). What changes is the *summation order across groups* (sequential per lane
vs strided + `simd_sum` tree) — the same class of reorder already accepted by using
`simd_sum`. Expect cosine unchanged, argmax parity 21/24 ±1; **measure**. Keep the epilogue
expressions character-identical to today so fast-math contraction matches.

Per-matrix assignment:

| matrix | kernel | SG_PER_TG | threadgroups | note |
|---|---|---|---|---|
| qkv 2048×1536 | `t32` (+_bias) | 2 | 32 | 64 tiles |
| gate/up 17920×1536 | `t32` | 8 | 70 | the 42% kernel |
| down 1536×8960 | Stage A `_nostage` | 8 rows/tg | 192 | t32 gives only 48 tiles + 18 KB tg mem — wrong shape |
| o 1536×1536 | `t32` (+_resid) | 2 | 24 | |
| lm-head 151936×1536 | `t32_amax` (below) | 8 | 594 | |

### ‹FETCHING› — the truncated compilable preamble

Needed to make the above drop-in: (1) the **`DOT8(word, aLo, aHi)`** macro — 8-nibble
extract-and-multiply into an int, (2) the full **Stage A** (`_nostage`, no-repack) kernel,
(3) the **`gemv_w4a8_t32`** signature + threadgroup activation-staging block (`aq → As`
`short4`, with barrier) + the **`Sg`** per-group activation-sum precompute + `As4`/`sp`
setup. Design is fully specified above; source assembly/refetch pending.

## Fused block-argmax lm-head (complete — the parallel swing)

Readback per token drops from 608 KB + CPU scan to **4 bytes**. Merge operator is max over
the lexicographic key `(v, −idx)` — associative + commutative ⇒ order-independent, tie-broken
identically to a CPU first-max-wins left-to-right scan (values match: each lane computes
`acc*asc[0]` with the store-variant's exact expression, just never materialized).

Kernel 1 — same body as `t32` (staging, `Sg`, dot loop), argmax epilogue. `SG_PER_TG=8`:

```metal
struct AmaxPart { float v; uint i; };   // 8B; Go allocates ceil((N/32)/8) = 594 for lm

kernel void gemv_w4a8_t32_amax(
    device const uint4* wq  [[buffer(0)]],
    device const float* sct [[buffer(1)]],
    device const char*  aq  [[buffer(2)]],
    device const float* asc [[buffer(3)]],
    device AmaxPart*    part[[buffer(4)]],
    constant uint&      K   [[buffer(5)]],
    constant uint&      N   [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    // ... identical staging + Sg blocks and barriers as gemv_w4a8_t32 ...
    const uint G = K >> 5;
    uint tile = tgid * 8u + sgid;
    float v = -INFINITY; uint idx = 0xFFFFFFFFu;
    if (tile * 32u < N) {                          // guard via if, NOT return: barriers below
        uint row = tile * 32u + lane;
        device const uint4* wp = wq  + (ulong)tile * G * 32u + lane;
        device const float* sp = sct + (ulong)tile * G * 32u + lane;
        const threadgroup short4* As4 = (const threadgroup short4*)As;
        float acc = 0.0f;
        for (uint g = 0u; g < G; ++g) { /* identical dot loop */ }
        v = acc * asc[0]; idx = row;               // the logit, exactly as the store variant
    }
    for (uint off = 16u; off > 0u; off >>= 1u) {   // simdgroup argmax, first-max-wins
        float ov = simd_shuffle_down(v, off);
        uint  oi = simd_shuffle_down(idx, off);
        if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
    }
    threadgroup float tv[8]; threadgroup uint ti[8];
    if (lane == 0u) { tv[sgid] = v; ti[sgid] = idx; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float bv = tv[0]; uint bi = ti[0];
        for (uint s = 1u; s < 8u; ++s)
            if (tv[s] > bv || (tv[s] == bv && ti[s] < bi)) { bv = tv[s]; bi = ti[s]; }
        part[tgid].v = bv; part[tgid].i = bi;
    }
}
```

Kernel 2 — one threadgroup of 256, encoded right after in the same command buffer:

```metal
kernel void argmax_finish(
    device const AmaxPart* part [[buffer(0)]],
    device uint*           tok  [[buffer(1)]],
    constant uint&         P    [[buffer(2)]],    // 594
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    float v = -INFINITY; uint idx = 0xFFFFFFFFu;
    for (uint p = tid; p < P; p += 256u) {
        float cv = part[p].v; uint ci = part[p].i;
        if (cv > v || (cv == v && ci < idx)) { v = cv; idx = ci; }
    }
    for (uint off = 16u; off > 0u; off >>= 1u) {
        float ov = simd_shuffle_down(v, off);
        uint  oi = simd_shuffle_down(idx, off);
        if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
    }
    threadgroup float tv[8]; threadgroup uint ti[8];
    if (lane == 0u) { tv[sgid] = v; ti[sgid] = idx; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float bv = tv[0]; uint bi = ti[0];
        for (uint s = 1u; s < 8u; ++s)
            if (tv[s] > bv || (tv[s] == bv && ti[s] < bi)) { bv = tv[s]; bi = ti[s]; }
        tok[0] = bi;                               // host reads exactly these 4 bytes
    }
}
```

Notes: idle tail simdgroups contribute `(−INF, 0xFFFFFFFF)`, which lose every merge.
Lane→row mapping (lower lane = lower row, lower tile = lower rows) keeps index ordering
globally consistent. **Pre-repack variant:** the same epilogue bolts onto Stage A — after
`simd_sum`, lane 0 holds `(v=acc*asc[0], idx=row)`; reduce the 8 simdgroups through
`tv/ti[8]`; `P = N/8 = 18,992` partials (152 KB); finish pass still costs a few µs.

## Other levers — do / don't

- **int8 SIMD dot / dp4a on Apple GPUs: does NOT exist — don't hunt.** MSL `dot()` is
  float-only; no integer-dot builtin at any version; `simdgroup_matrix` is f16/f32 GEMM
  (breaks int-accumulate parity — wrong tool for batch-1 GEMV). Honest int path on M1 is
  `extract_bits` + 32-bit imad, ~2 ops/element after the rewrite — no longer the limiter.
  `char4` loads help as instruction-count, not packed math.
- **f16 group scales: do it *after* Stage B, parity-gated.** Pure bandwidth (scales are
  ~20% of both kernels' bytes → ~10% total cut). No-op while issue-bound; ~8–10% once
  bandwidth-bound. Only value-changing lever — first check if the int8→int4 requant scales
  are already f16-exact (then free/lossless); else measure argmax 21/24 before keeping.
- **Threadgroup-staged activations: yes — the single biggest lever** (inside Stages A/B).
  Payoff isn't the 1.5 KB bandwidth (L1 covers it) — it's deleting ~90% of LSU issue work
  (byte gathers) + pre-widening to `short` once. Staging alone ≈ −30–40% on gate/up; wide
  loads alone ≈ −20–30%; together ≈ −45–55% (superlinear, same port).
- **`−8·S_g` factoring: do (Stage B), free + exact.**
- **int16 accumulate: speculative, try last** (without `−8` factoring, per-group sums fit
  int16 exactly: 32·8·127 = 32,512 ≤ 32,767; with `−8` folded they overflow — pick one).
- **gate+up+swiglu fusion: skip for now** — swiglu needs a full-vector absmax (global
  reduction → two-phase anyway); 22 µs isn't worth entangling the hot kernel.
- **Sidebar — the 2.4 ms encode/commit gap:** host `objc_msgSend` encode + per-dispatch
  serial-encoder barriers. Bindings are static per token → **Indirect Command Buffer
  encoded once, replayed per token** (pure Metal, purego-reachable). +6–9 tok/s, medium
  confidence; verify compute-ICB inter-command barrier semantics first. Argmax fusion
  independently shrinks this gap (removes the 608 KB readback + CPU scan, ~0.2–0.4 ms/tok).

## Estimated µs/dispatch + tok/s roll-up

Floors at 200 GB/s incl. scale traffic; "central" assumes 80–85% of peak; ranges are ±.

| kernel | today | floor | Stage A | Stage B | B + f16 scales |
|---|---|---|---|---|---|
| gate/up 17920×1536 | 220.5 | 86 | 120–145 | **95–115** | 88–105 |
| lm-head 151936×1536 | 1813.5 | 730 | 1000–1250 | **800–950** (+~10 finish) | 700–850 |
| down 1536×8960 | 117.7 | 43 | 65–85 (`_nostage`) | — (keep Stage A) | 60–78 |
| qkv 2048×1536 | 28.8 | ~10 | 18–24 | 14–18 | — |
| o 1536×1536 | 24.0 | ~7 | 15–20 | 11–15 | — |

Token roll-up (28 layers + lm + finish + final norm; keeping the measured 66.2 µs of
norm/swiglu/attention per layer):
- **Stage A + argmax:** ≈ 28×(21+132+75+17+66.2) + 1100 + 10 ≈ 9.8 ms kernels + ~2.1 ms
  overhead ≈ 11.9 ms → **~80 tok/s** (slippage floor ~73).
- **Stage B + argmax:** ≈ 8.6 ms + ~2.1 ms ≈ 10.7 ms → **~88 tok/s** (floor ~80);
  +f16 scales → **~90+**.

Even the pessimistic edges clear 71. Treat central numbers as ±20–25% until the first
Stage A measurement calibrates the model.

## Ranked priorities (expected decode-tok/s gain)

1. **Stage A body swap** (vector loads + staged `short` activations + 8 sg/tg; no repack):
   **+12–18**, hours, also validates the diagnosis.
2. **Fused block-argmax lm-head** (Stage A or B): **+4–7**, low risk.
3. **Stage B repack** (t32 lane-per-row, `uint4`=group): **+5–8** on top of A, ~a day.
4. **ICB / encode-overhead attack** on the 2.4 ms gap: **+6–9**, medium confidence.
5. **f16 group scales**: **+2–4**, parity-gated, ~30 min once B is in.
6. Micro (int16 accumulate, g-loop unroll×2, swiglu fusion): ≤ +2, last or skip.

## Certain vs speculative

**Certain:** no dp4a/int8-dot on M1 via MSL; the coalescing/indexing math (512 B + 128 B
contiguous transactions; all five dims ÷32); `−8·S_g` factoring + within-group int
accumulation exact (`float(gi)` exact below 2¹⁷); the argmax merge is a commutative monoid
⇒ order-independent, tie-broken identically to first-max-wins CPU; both hot kernels run at
the same ~79 GB/s ⇒ not bandwidth-bound; activation byte gathers dominate LSU issue by
~an order of magnitude.

**Speculative:** exact LSU beat-rate/absolute cycles (ratios solid — Instruments limiter
counters settle it in ~5 min); every µs estimate (±20–25%); how much of the 2.4 ms gap ICBs
recover + compute-ICB barrier semantics; f16-scale argmax parity (check requant-scale
f16-exactness first); the int16-accumulate win; whether Stage A's strided threadgroup reads
cost conflict cycles (if A undershoots, first suspect — Stage B's broadcast reads are the cure).

## Measurement protocol

Land **Stage A for gate/up only** first: best-of-40 + per-kernel µs + 21/24 argmax + cosine.
If it lands ≤145 µs the model is calibrated → roll to lm-head/qkv/o/down, then argmax fusion,
then the repack. Keep the current kernel compiled alongside for A/B and the logits debug path.
