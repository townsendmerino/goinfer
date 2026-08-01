# goinfer GPU backends — Engineering Audit

**Date:** 2026-07-25 · **Scope:** `gpu/` (WebGPU, ~19.5k LOC), `metal/` (~8.5k), `cuda/` (~6.5k), plus all WGSL/MSL/CUDA kernel sources
**Categories:** bugs & correctness, performance, architecture & maintainability

**Method.** Every non-test file in `gpu/`, `metal/`, `cuda/` plus all shader/kernel sources (the shaders are Go string constants, not separate `.metal`/`.wgsl` files). `go vet` could not run — `go.mod` pins `go 1.26.5`, the container had 1.24.7, and the toolchain download was blocked — so everything below is from reading, with callers traced to distinguish real bugs from guarded invariants.

---

## Critical & High

### 1. WebGPU MoE router **silently clamps** `nE` to 256 and has an unvalidated 32-group cap — CUDA and Metal both decline

**Severity: High** · `gpu/moe.go:34`, `gpu/moe.go:55`, `gpu/moe.go:65`

```wgsl
fn main() {
    let nE = min(p.nE, MAXE);        // MAXE = 256u
    var score: array<f32, 256>;
    ...
    if (p.nGroup > 1u) {
        let gsz = nE / p.nGroup;
        var gscore: array<f32, 32>;     // <-- 32, not 64
        ...
        var keep:   array<bool, 32>;
```

The other two backends refuse these shapes at load:

```go
// cuda/backend.go:110
case nE > 256:
    return declined(fmt.Errorf("MoE nE=%d exceeds moe_route's MOE_MAX_E=256", nE))
case nGroup > 64:
    return declined(fmt.Errorf("MoE nGroup=%d exceeds moe_route's MOE_MAX_G=64", nGroup))
```
```go
// metal/moe.go:191
if nE > 256 { return nil, fmt.Errorf("metal MoE: nE=%d exceeds router cap 256", nE) }
if nGroup > 64 { return nil, fmt.Errorf("metal MoE: nGroup=%d exceeds cap 64", nGroup) }
```

`gpu/residency.go:100-141` (`BuildResident`) applies only the feature-taxonomy check; there is no numeric cap anywhere on the WebGPU path, and `decoder.MoEResidentParams()` (`decoder/residency.go:194`) passes `NumExperts`/`NGroup` straight through.

**Failure scenario.** A `>256`-expert MoE — Kimi-K2 (`n_routed_experts: 384`), which the README explicitly lists as supported — loads on `--backend webgpu`, `min(384, 256)` truncates the router to experts 0–255, and the top-k is selected from ⅔ of the experts. Logits are plausible and wrong; nothing errors. Separately, `nGroup > 32` writes `gscore[g]`/`keep[g]` past a fixed function-scope array; WGSL robustness clamps rather than faults, so group-limited routing silently collapses onto group 31.

This is precisely the bug class `decoder/features.go:174-184` says the shared taxonomy exists to prevent ("Overclaiming here is exactly the lie the gate exists to catch") — but the taxonomy has no expressive slot for a *numeric* cap, so it can't catch it.

**Fix.** In `gpu/residency.go`'s `BuildResident`, decline when `nE > 256` or `nGroup > 32` (mirroring `cuda/backend.go:110-113`), and change `let nE = min(p.nE, MAXE)` to plain `let nE = p.nE`, so a future cap violation is a bounds bug the tests catch rather than a silent clamp. Raise `gscore`/`keep` to 64 to match the other two backends.

---

### 2. Neither Metal nor WebGPU ever observes a GPU-side error

**Severity: High** · `metal/metal.go:491-497`, `metal/metal.go:353-371`, `gpu/gpu.go:293-300`

Metal commits and waits but never reads the command buffer's outcome:

```go
func (e *Encoder) end() {
	e.enc.Send(selEndEncoding)
	e.cb.Send(selCommit)
	e.cb.Send(selWaitCompleted)
	e.readTimes()
	e.pool.Send(selDrain)
}
```

There is no `MTLCommandBufferStatus` / `.error` selector registered anywhere in the package (`metal/metal.go:164-186` registers only `commit`, `waitUntilCompleted`, and the four timestamp selectors). WebGPU is the same story: `New()` requests the device at `gpu/gpu.go:293` with no `SetUncapturedErrorCallback`, and no call site anywhere in `gpu/` uses `PushErrorScope`/`PopErrorScope`. `DecodeRunner.Run` (`gpu/decoderunner.go:~995`) checks only the *map* status, not device validity.

**Failure scenario.** A dispatch that exceeds `maxThreadgroupMemoryLength` (see #10), a UMA page fault from the OOB write in #4/#13, or a mid-decode allocation failure aborts the command buffer with `MTLCommandBufferStatusError`. `waitUntilCompleted` returns normally, `copy(r.logitsHost, r.logits.Floats())` reads the *previous* token's logits, and generation continues producing fluent, wrong text. Only CUDA gets this right (`cuda/resident.go:388-397` sticky `launchErr` + `stream.Synchronize` checked in `step`).

**Fix.** Register `selStatus`/`selError` and fail loudly in `Encoder.end()`/`waitDone()`; on the WebGPU side install an uncaptured-error callback at device creation that latches into a sticky `Context.err` which `DecodeRunner.Run` returns.

---

### 3. Metal Stage-A GEMV row mapping is wrong in the tail threadgroup of a non-uniform dispatch — and one of the seven variants proves it

**Severity: High** · `metal/kernels.go:131`, `:134`, `:167-168`, `metal/moe.go:100`, `:128`

Every Stage-A kernel derives its output row from the *runtime* threadgroup size:

```c
#define SA_BODY \
    ...
    uint row = tgid*(tgs>>5u) + sgid; \
```

where `tgs` is `[[threads_per_threadgroup]]`. Under `dispatchThreads:threadsPerThreadgroup:` (`metal/metal.go:460`) Metal uses **non-uniform** threadgroups, and MSL reports the *reduced* edge-group size in `threads_per_threadgroup` — that is exactly why `dispatch_threads_per_threadgroup` exists as a separate attribute. So for `total = N*32, perTG = 256` with `N % 8 != 0`, the final threadgroup has `tgs = (N%8)*32` and computes `row = floor(N/8)*(N%8) + sgid` instead of `floor(N/8)*8 + sgid`: it recomputes an already-written row and leaves the true tail rows **never written** (uninitialised scratch).

The author clearly knew: the batch-k twin is the only one with a guard —

```c
// metal/kernels.go:167
uint row = tgid*(tgs>>5u) + sgid;
if (row >= N) return;
```

while `gemv_w4a8_sa`, `_sa_bias`, `_sa_resid`, `_sa_amax`, `gemv_w8a8_amax`, `gemv_w4a8_moe` and `gemv_w4a8_moe_wacc` have none.

**Failure scenario.** `ForwardArgmax` (`metal/model.go:532`) dispatches `gemv_w8a8_amax` at `(r.V)*32, tg=256`. A vocab that is not a multiple of 8 — GPT-2's 50257 — makes threadgroup 6282 write `part[6282]` for row 6282 (a duplicate) while row 50256's logit never reaches any tile, so the greedy token can differ from `argmax(Forward())`. Today's dense/MoE shapes (`qkvRows`, `H`, `2*I`, `2*inter`) all happen to be multiples of 8, so this is latent — but it is one odd `nH*hd + 2*kvDim` away from being live, with no assert and no error.

**Fix.** Add `if (row >= N) return;` (and pass `N`) to all seven Stage-A variants, or switch the row derivation to `[[dispatch_threads_per_threadgroup]]`. Cheapest immediate hardening: assert `N % 8 == 0` in `Encoder.dispatchTG` for the SA pipelines.

---

### 4. WebGPU decode attention pays ~9 workgroup barriers **per cached key** — the dominant long-context cost

**Severity: High (performance)** · `gpu/attention.go:75-96` (and the identical `attnF16` `:248-275`, `attnI8` `:375-397`, `mlaAttn` `gpu/mla.go:52-81`)

```wgsl
for (var s: u32 = p.start; s < p.nKeys; s = s + 1u) {
    ...
    red[d] = prod;
    workgroupBarrier();
    var stride: u32 = 64u;
    loop {
        if (stride == 0u) { break; }
        if (d < stride) { red[d] = red[d] + red[d + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let x = red[0] * p.scale;
    ...
    workgroupBarrier();
}
```

One lane computes **one** multiply per key, then the whole 128-lane workgroup performs a 7-level tree reduce: 1 + 7 + 1 = 9 barriers per key. The arithmetic per lane per key is a single FMA, so the kernel is ~100 % barrier-latency bound.

**Failure scenario (measurable slowdown).** Qwen2.5-7B at a 2048-token context, 28 layers: the *critical path* per layer is 2048 × 9 ≈ 18 400 workgroup barriers, ×28 layers ≈ 516 000 per token. At even 20 ns per barrier that is ~10 ms/token of pure synchronisation, independent of memory bandwidth — and it grows linearly with context while the actual FLOPs are trivial.

**Fix.** Invert the parallel axis: give each lane a *stripe of keys* (`s = start + lane; s += 128`), have it compute the full `hd`-wide dot serially, maintain a per-lane online-softmax `(m, l, acc[])`, and do **one** merge reduction at the end — O(1) barriers per head instead of O(nKeys). This is the standard decode-attention shape and needs no new device features.

---

### 5. The production activation quantizer does its max-abs scan **entirely on lane 0**

**Severity: High (performance)** · `gpu/device.go:38-49`, dispatched from `gpu/decoderunner.go:328-338`

```wgsl
@compute @workgroup_size(64)
fn main(...) {
    let m = wid.x;
    if (m >= d.m) { return; }
    let base = m * d.n;
    if (lid.x == 0u) {
        var mx: f32 = 0.0;
        for (var i: u32 = 0u; i < d.n; i = i + 1u) {   // serial, ONE lane
            let v = abs(src[base + i]);
            if (v > mx) { mx = v; }
        }
        ...
    }
    workgroupBarrier();
```

The header comment excuses it as "naive serial reduce on lane 0 — trivial at decode; the rows run in parallel" — but at decode there is exactly **one** row:

```go
// gpu/decoderunner.go:334
p := uni([]uint32{1, uint32(K), uint32(kp), 0})
add(c.quantizePipeline, bind(c.quantizeLayout, in, q, s, p), 1, 1)
```

So `M = 1` ⇒ one workgroup ⇒ **one thread** doing an O(K) dependent-load scan while 63 lanes wait at the barrier. This runs once per layer (`cq, cs := quant(ctxv, nH*hd)` at `gpu/decoderunner.go:823`; also for MLA `nH*vHead` and mamba `dInner`).

**Failure scenario.** Qwen2.5-7B: `nH*hd = 4096`, 28 layers ⇒ 114 688 serial iterations per token on a single lane, on the critical path. Sibling kernels in the same package already do this correctly with a parallel tree reduce (`rmsnormQuantWGSL`, `gpu/decodefuse.go:50-64`), so this is a straight ~40–60× reduction in that kernel's latency for a copy-paste.

**Fix.** Replace the lane-0 loop with the `rmsnormQuant` reduction pattern (`sh[t] = mx; workgroupBarrier();` + tree reduce) and raise `workgroup_size` to 256.

---

### 6. CUDA's `f32→f16` **truncates** while documenting round-to-nearest-even, and the three backends have three different converters

**Severity: High** · `cuda/kernels.go:67-80` vs `metal/pack.go:53-80` vs `gpu/gemv_w4a8.go:118-131`

```go
// cuda/kernels.go:67
// f32tof16 encodes an IEEE float32 into a float16 bit pattern (round-to-nearest-even, simple).
func f32tof16(f float32) uint16 {
	...
	return s | uint16(e<<10) | uint16(m>>13)      // <-- pure truncation, no rounding at all
}
```

Compare Metal, which rounds and handles subnormals:

```go
// metal/pack.go:74
half := sign | uint16(e<<10) | uint16(m>>13)
if m&0x1000 != 0 { half++ }
```

and WebGPU, which rounds but flushes everything below 2⁻¹⁴ to zero:

```go
// gpu/gemv_w4a8.go:127
if exp <= 0 { return sign }          // no f16 subnormals
half := sign | uint16(exp<<10) | uint16(mant>>13)
if mant&0x1000 != 0 { half++ }
```

**Failure scenario.** Every W4A8 group scale on CUDA is biased **downward** by a mean of 0.5 ULP (~0.024 % relative), systematically and in one direction, across every group of every weight of every layer — unlike a rounding converter, whose error is zero-mean. This shows up as a reproducible cross-backend logit-cosine gap that no amount of re-running will explain, and it burns parity headroom for nothing: the fix is one line. The doc comment actively misdirects whoever debugs it.

**Fix.** Make `cuda/kernels.go:f32tof16` round (`if m&0x1000 != 0 { half++ }`), and better: hoist one shared converter (Metal's is the most complete — it handles inf/NaN, subnormals, and rounding) into a package all three backends import, deleting the other two.

---

### 7. CUDA weight uploads discard `CopyHtoD` errors

**Severity: High** · `cuda/resident.go:151-181`

```go
func (r *cudaResident) up32(v []float32) *gc.Buffer[float32] {
	b := r.af(len(v))
	if b != nil {
		_ = gc.CopyHtoD(context.Background(), b, v)   // <-- error dropped
	}
	return b
}
func (r *cudaResident) upu32(v []uint32) *gc.Buffer[uint32] {
	b, e := gc.Alloc[uint32](r.cx, len(v))
	if e != nil && r.setupErr == nil { r.setupErr = e }
	if b != nil {
		_ = gc.CopyHtoD(context.Background(), b, v)   // <-- error dropped
	}
	return b
}
```

`setupErr` deliberately latches *allocation* failures so `BuildResident` can decline gracefully — but the H2D copy that actually fills the buffer is unchecked in all three helpers (`up32`, `upu32`, `upu16`), and `upW` (`cuda/resident.go:196`) funnels every projection weight through them.

**Failure scenario.** A VRAM-pressure or `CUDA_ERROR_INVALID_VALUE` failure on one layer's `expGU` upload leaves that buffer holding whatever `cuMemAlloc` returned. `setupErr` is nil, so `BuildResident` reports success and the model decodes with one garbage expert stack — exactly the "OOM wearing a parity bug's clothes" failure `metal/metal.go:230-235` was written to avoid on the Metal side.

**Fix.** Latch the copy error into `setupErr` in all three helpers, same as the `Alloc` error.

---

## Medium

### 8. Metal's int4 path silently truncates a non-multiple-of-32 `K`; CUDA declines and WebGPU pads

**Severity: Medium** · `metal/model.go:121-131`, `:154-163`, `metal/pack.go:11-14`, `metal/kernels.go:129`

```go
// metal/model.go:121 — the only shape check is the group size
func int4DirectWords(w *linalg.WeightMat) (words []uint32, scales []uint16, ok bool) {
	q4, q4s, group, ok := w.Int4()
	if !ok || group != 32 { return nil, nil, false }
```
```go
// metal/model.go:155
words := make([]uint32, N*(K/8))
scales := make([]uint16, N*(K/32))
```

and in-kernel: `uint G = K>>5u;` (`metal/kernels.go:130`), `uint wpr = K/8u;` (`:78`).

CUDA refuses the shape outright (`cuda/resident.go`'s `packWeight`: `if K%32 != 0 { return hostW{}, fmt.Errorf("cuda: int4 K=%d not a multiple of 32", K) }`) and WebGPU pads (`padK32`, `gpu/gemv_w4a8.go:113`, `packNibbles`).

**Failure scenario.** A projection whose `K` is 32·q + r (r ≠ 0) — e.g. a future arch with `nH*hd` not 32-aligned for the o-projection — drops the trailing r elements of *every* dot product on Metal, with no error and no test signal beyond a slightly-off cosine. Same code, three different contracts.

**Fix.** Add the CUDA check to `metal/model.go`'s `int4Buf`/`int4Concat`/`int4DirectWords` and let `BuildResident` decline.

### 9. WebGPU activation padding (mult-16) contradicts the W4A8 kernel's requirement (mult-32)

**Severity: Medium** · `gpu/quant.go:114`, `gpu/decoderunner.go:~298`, `gpu/gemv_w4a8.go:67-69`

Activations are padded with `padK` (mult of **16**): `func padK(k int) int { return (k + 15) &^ 15 }`. But `decodeWeight.kPad()` for `*ResidentW4A8` is `padK32(K)` (mult of **32**), and that value is what lands in the uniform (`gpu/decoderunner.go:342`). The W4A8 GEMV then indexes the activation off it:

```wgsl
let ng = dims.kp / 32u;
for (var v: u32 = t; v < ng; v = v + 64u) {
    let a0 = aq[v * 2u];
    let a1 = aq[v * 2u + 1u];       // reads up to kp32/16 vec4s
```

When `K ≡ 16 (mod 32)`, `padK32(K) = padK(K) + 16`, so the final iteration reads one `vec4<u32>` past the activation buffer. WGSL robustness clamps the index rather than faulting, so it re-reads a *live* activation word — and `packNibbles` (`gpu/gemv_w4a8.go:174-188`) leaves pad nibbles at 0, which the kernel decodes as value **−8**, not 0. The product is nonzero garbage.

**Failure scenario.** Any admitted W4A8 projection with `K/16` odd yields a silently wrong 32-element group contribution per output row. Every shipped `K` today (896, 1536, 2048, 3584, 4096, 4864, 8960, 11008, 18944…) is a multiple of 32, so this is guarded only by luck.

**Fix.** Pad activations to `padK32` whenever any consumer is W4A8 (or make `padK` unconditionally mult-32 — the extra 16 bytes are free), and zero the weight pad nibbles to 8 so a residual mismatch contributes zero instead of −8.

### 10. Neither Metal nor WebGPU queries any device limit; threadgroup sizes and shared-memory budgets are hard-coded

**Severity: Medium** · `metal/metal.go` (no limit selectors anywhere), `metal/kernels.go:313`, `metal/moe.go:292`

A repo-wide grep for `maxTotalThreadsPerThreadgroup`, `maxThreadsPerThreadgroup`, `threadExecutionWidth` and `maxThreadgroupMemoryLength` returns **nothing**. The code instead hard-codes:

- `threadgroup float sc[4096]` + `red[128]` in `attention` (`metal/kernels.go:313-314`) = 16.9 KB static, on a 32 KB budget;
- dynamic threadgroup memory sized from model dims, unchecked: `e.dispatchTG(mo.pDownWacc, r.H*32, 256, mo.inter*2, …)` (`metal/moe.go:292`) requests `2·inter` bytes — 28 672 B for Mixtral's `inter = 14336`, i.e. 87 % of the budget, and >32 KB for any `inter > 16384`;
- `tg = 256` and `tg = 128` everywhere, with `Encoder.dispatch` clamping only `tg > n` (`metal/metal.go:441-443`), never against the pipeline's `maxTotalThreadsPerThreadgroup`;
- `simd_sum` / `simd_shuffle_down(v, 16)` assume a 32-wide simdgroup (`metal/kernels.go:266`).

**Failure scenario.** An MoE with `inter ≥ 16384` makes `setThreadgroupMemoryLength:` exceed the device limit; Metal aborts the command buffer, and because of #2 nobody notices — the host reads the previous token's logits.

**Fix.** Query `MTLDevice.maxThreadgroupMemoryLength` and each pipeline's `maxTotalThreadsPerThreadgroup` once in `BuildResident`, clamp `tg`, and decline the model when `2·max(I, inter, nH·hd) > limit`.

### 11. The MoE router is one single-threaded GPU invocation with 2 KB of dynamically-indexed private arrays — in all three backends

**Severity: Medium (performance)** · `metal/moe.go:47-48`, `cuda/moe.cu:37-39`, `gpu/moe.go:32-36`

```c
// metal/moe.go:47
if (tid != 0u) return;
float score[256];
float sel[256];
```

Dispatched as one thread (`e.dispatch(mo.pRoute, 1, 1, …)`, `metal/moe.go:286`; `r.launch(r.fRoute, onecfg(1, 0), …)`; `add(c.moeRoutePipeline, …, 1, 1)`, `gpu/decoderunner.go:~610`). Two 256-entry dynamically-indexed arrays cannot live in registers, so they spill to thread-local (device-backed) memory, and the top-k loop is `k × nE` dependent local-memory loads executed by one lane while the rest of the GPU idles.

**Failure scenario.** DeepSeek-V3-class routing (`nE = 256`, `k = 8`) is 2048 serial spilled-memory compares per layer. Across 60 MoE layers that is ~123 000 serial dependent loads per token on a single lane, on the critical path between the router GEMV and the first expert dispatch — a fixed per-token tax that no expert-GEMV optimisation can hide.

**Fix.** One workgroup, `nE` lanes: sigmoid/softmax in parallel, then `k` iterations of a parallel argmax tree reduce over `sel[]` in shared memory (masking the winner each round). The arrays disappear, the loop becomes `k·log₂(nE)` steps, and tie-breaking stays "lowest index wins" if the reduce keeps the lower index on equality.

### 12. Metal's norm / quantize / SwiGLU kernels each run in **one** threadgroup, on the critical path

**Severity: Medium (performance)** · `metal/model.go:765`, `:774`, `:789`, `:791`, `:801`; `metal/moe.go:282`, `:291`

```go
e.dispatch(r.pRms, 256, 256, r.x, L.preNorm, r.aq, r.aSc, r.uH, r.uEps, r.uAddOne)
...
e.dispatch(r.pQv,  256, 256, r.ctx, r.cq, r.cSc, r.uHH)
...
e.dispatch(r.pSw,  256, 256, r.gu, r.gu.At(r.I*4), r.dq, r.dSc, r.uI, r.uAct)
```

`n = tg = 256` means exactly one threadgroup — one GPU core out of 10–40 — for the whole vector. `swiglu_quant` is the worst: it evaluates `glu_act(g[i], act)*u[i]` **twice** per element (once for the max-abs reduce at `metal/kernels.go:384`, once for the quantize at `:388`), so for `I = 11008` each of the 256 lanes does 86 GELU/SiLU evaluations serially on a single core. Three to four such dispatches per layer × 28 layers ≈ 100 single-core serialisation points per token. WebGPU has the identical shape at `workgroup_size(64)` (`gpu/decodefuse.go:32`, `:104`; `gpu/relu2.go:20`; `gpu/layer.go:29`), dispatched `1,1` at `gpu/decoderunner.go:301,317,325`.

**Fix.** Two-stage reduce: N workgroups emit partial max-abs, a tiny second kernel finalises the scale, a third quantizes in parallel. That is 3 dispatches instead of 1 but ~N× the parallelism, and the encode-ahead executor already amortises dispatch cost. Cheapest partial win: cache `glu_act(g[i])*u[i]` in threadgroup memory to halve the transcendental count.

### 13. `nTiles := V / 8` with no `V % 8` check → out-of-bounds device write in the fused-argmax path

**Severity: Medium** · `metal/model.go:362-363`

```go
nTiles := V / 8 // one (maxLogit,rowIdx) partial per threadgroup (8 rows) — V divisible by 8
r.part, r.tok, r.uP = d.NewBufferLen(nTiles*2), d.NewBufferLen(1), d.NewBufferU32(uint32(nTiles))
```

The comment states the invariant; nothing enforces it. `gemv_w8a8_amax` writes `part[tgid]` (`metal/kernels.go:258`) and `ForwardArgmax` dispatches `(r.V)*32, tg=256` (`metal/model.go:532`), so `tgid` ranges over `ceil(V/8)`. For `V % 8 != 0` the last threadgroup writes one `AmaxPart` (8 bytes) past a shared/UMA `MTLBuffer` — on unified memory that corrupts whatever buffer the allocator placed next, which per `metal/backend.go:96-101` may be another resident model's weights.

**Failure scenario.** GPT-2 (`V = 50257`) → an 8-byte write past `r.part`, plus `uP = 6282` so `argmax_finish` never sees the last tile. `ForwardArgmax` is currently reached only from `metal/model_test.go`, which is why this hasn't bitten — but it is an exported method with no guard, and `metalResident` deliberately does *not* implement `decoder.ResidentGreedy`, so the fast greedy path is also dead weight (see #19).

**Fix.** `nTiles := (V + 7) / 8`, and either wire `ForwardArgmax` into `decoder.ResidentGreedy` (as CUDA does at `cuda/resident.go:725`) or delete it.

### 14. Metal decode attention: uncoalesced per-key reads plus a fixed 16 KB threadgroup allocation

**Severity: Medium (performance)** · `metal/kernels.go:313-319`

```c
threadgroup float sc[4096];
threadgroup float red[128];
for (uint s=winStart+tid; s<nKeys; s+=tgs) {
    float a=0; device const half* k=kb+s*kvDim;
    for (uint d=0; d<hd; d++) a += qr[d]*float(k[d]);
    sc[s]=a*scale;
}
```

Two independent problems. (a) Thread `tid` streams key `s`'s entire row while its neighbour streams row `s+1`, `kvDim·2` bytes away — 32 divergent streams per simdgroup instead of one contiguous burst. (b) `sc[4096]` is *statically* 16 KB regardless of `nKeys`, so a 10-token context reserves the same threadgroup memory as a 4096-token one, capping residency at 2 threadgroups/core (32 KB budget) even at short context.

Note the WebGPU twin has the *opposite* pathology — perfectly coalesced (adjacent lanes read adjacent `d`) but barrier-bound (#4). Neither backend has both properties.

**Fix.** Vectorise the key dot with `half4`/`float4` loads and give each simdgroup a key (rather than each thread), so adjacent lanes read adjacent `d` *and* the reduce is a single `simd_sum`. Size `sc` from `nKeys` via `setThreadgroupMemoryLength:` (the dispatch path for that already exists — `dispatchTG`, `metal/metal.go:468`).

### 15. `PrefillLast` creates and destroys ~24 `MTLBuffer`s (100–150 MB) on every call

**Severity: Medium (performance)** · `metal/prefill.go:316-364`

```go
xF := d.NewBufferU16s(xh)
normF := d.NewBufferU16s(make([]uint16, Mpad*H))
qkvF := d.NewBufferU16s(make([]uint16, Mpad*qkvDim))
ctxF := d.NewBufferU16s(make([]uint16, Mpad*qDim))
guF := d.NewBufferU16s(make([]uint16, Mpad*2*I))
dqF := d.NewBufferU16s(make([]uint16, Mpad*I))
...
scratch := []Buffer{ xF, normF, qkvF, ctxF, guF, dqF, posB, uM, uI, u2I, /* …24 total */ }
defer func() { for _, b := range scratch { d.releaseBuf(b) } }()
```

The comment at `:348-354` documents the leak this replaced, but the fix went from "leak forever" to "allocate and free 150 MB per request" rather than to pooling. Each `releaseBuf` is also an O(n) linear scan of the whole device ledger under `d.mu` (`metal/metal.go:264-278`) — 24 scans over the ~350-entry ledger of a 28-layer model.

Additionally, sixteen of those 24 buffers are 4-byte *uniforms* (`uM`, `uI`, `u2I`, `uQkv`, `uQDim`, `uStride`, `uKOff`, `uVOff`, `uStartPos`, `uTotalQ`, `uTotalK`, `uBase0`, `uBaseK`, `m0`, `m1`, `m2`) whose values depend only on model dims and `startPos` — all but `uM`/`uStartPos` are constant for the lifetime of the `Resident`.

**Failure scenario.** A serving loop (`cmd/serve`, one prefill per request) pays ~150 MB of `newBufferWithLength:` page-table work per request, on the TTFT path, plus 24 ledger scans — measurable at high request rates and entirely avoidable.

**Fix.** Hoist the 14 constant uniforms into `Resident` (next to `r.uH`/`r.uKvDim`, which are already reused) and keep an `Mpad`-keyed scratch pool that only reallocates when the prompt grows.

### 16. Mamba-2 SSM state layout makes every state access uncoalesced

**Severity: Medium (performance)** · `gpu/mamba.go:57`, `:61-65`

```wgsl
let sBase = (head * p.hp + pi) * p.dn;
...
for (var n: u32 = 0u; n < p.dn; n = n + 1u) {
    let v = ssm[sBase + n] * dA + dx * conv[bBase + n];
    ssm[sBase + n] = v;
    acc = acc + v * conv[cBase + n];
}
```

Thread `t = head*hp + pi` owns `ssm[t*dn … t*dn+dn)`. At any fixed `n`, adjacent threads are `dn·4` bytes apart — 512 B for `dn = 128` — so each 32-lane wave touches 32 distinct cache lines per iteration, for both the read and the write.

**Failure scenario.** Granite-4.0-H / Nemotron-H: `dInner = 4096` threads × `dn = 128` = 512 K f32 read **and** written per mamba layer per token = 4 MB of traffic, at roughly ⅛ of achievable bandwidth because of the stride. This is the dominant cost of the SSM step.

**Fix.** Transpose the state to `ssm[n * (nHeads*hp) + t]`. Adjacent threads then read adjacent addresses at every `n`, the access becomes fully coalesced, and only `Reset()` (which writes zeros) and the `dS` uniform need to change.

### 17. int8-KV store kernels run one thread per KV head and recompute RoPE twice per element

**Severity: Medium (performance)** · `gpu/attention.go:404-439`, `:443-465`, dispatched at `gpu/decoderunner.go:~558`, `~578`

```wgsl
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let h = gid.x;
    if (h >= p.heads) { return; }        // heads = nKV
    var amax: f32 = 0.0;
    for (var d: u32 = 0u; d < p.headDim; d = d + 1u) { amax = max(amax, abs(rot(off, d))); }
    ...
    for (var d: u32 = 0u; d < p.headDim; d = d + 4u) {
        dst[wbase + d/4u] = qb(rot(off, d), inv) | (qb(rot(off, d+1u), inv) << 8u) | ...
```
```go
add(c.ropeStoreI8Pipeline, bind(...), uint32(nKV+63)/64, 1)
```

With `nKV = 2..8`, that dispatch is **one** workgroup in which 2–8 of 64 lanes are active, and each active lane serially walks `headDim` elements *twice*, calling `rot()` — two `cos` + two `sin` per element — on both passes. The comment ("recompute the cheap rope rather than spill a local array") is right that a local array would spill, but the parallelisation choice is what costs.

**Fix.** One workgroup per KV head, `headDim` lanes: each lane computes its own `rot()` once into threadgroup memory, tree-reduce the absmax, then pack. That is 2 dispatches' worth of parallelism instead of 8 lanes, and halves the trig.

### 18. `gemmRowW8A8`'s `M ≤ 16` invariant is checked in one caller and not the other

**Severity: Medium** · `gpu/gemm_rows.go:52`, `:~108` vs `gpu/decodetoken_batched.go:126-127`

```wgsl
var acc: array<i32, 16>;          // gemmRowMaxM accumulators
for (var m: u32 = 0u; m < M; m = m + 1u) { acc[m] = 0; }
```

The parity-gated entry point guards it (`MatmulW8A8GemmRow`: `if M <= 0 || M > gemmRowMaxM { return nil, fmt.Errorf(...) }`). `DecodeTokenFusedBatched` does not — it builds the uniform from `M := len(xs)`, unbounded (`gpu/decodetoken_batched.go:26`).

**Failure scenario.** A 20-token speculative block through this exported API writes `acc[16..19]` out of a 16-entry function array; WGSL clamps, so `acc[15]` is corrupted and rows 15–19 return garbage logits, silently. The same function also `panic`s inside its `bind` closure (`gpu/decodetoken_batched.go:73`) rather than returning an error — inconsistent with every other exported call in the package.

**Fix.** Add the `M > gemmRowMaxM` check at the top of `DecodeTokenFusedBatched`, or delete the function (see #19).

### 19. Dead code carrying real invariants

**Severity: Medium (maintainability)**

- **`cuda/megakernel.cu`** — three kernels with empty bodies (`{ // scaffold }`, lines 30-46) and a 24-line spec comment. `build_ptx.sh:74` special-cases it (`if [ "$k" = megakernel ]; then continue; fi`) and no `.ptx` is embedded. It is a spec document masquerading as a source file.
- **Seven `cuda/gemv_w4a8_*.cu` variants** (`_coal`, `_coal2`, `_coal3`, `_coal4`, `_fast`, `_v4`, plain) — all embedded only from `cuda/w4a8_fast_test.go` / `w4a8_test.go` bench files. Production uses `gemv_fwd.cu` alone.
- **`gpu/decodetoken_batched.go` (242 lines) + `gpu/decodetoken_fused.go` (180 lines)** — exported, referenced only from `_test.go` / `_bench_test.go`. `DecodeTokenFusedBatched` also opens a **new compute pass per dispatch** (`:78-84`), forcing a full barrier between every op — the exact thing `DecodeRunner.Run` was written to avoid.
- **`metal/prefill.go`**: `pGemm` (`gemm_w4f16`) and `pRes` (`residual_f16`) are compiled and tracked (`:285-286`) but never dispatched — `PrefillLast` uses only `pGemmStore` and fuses the residual into `mode == 2`.
- **`metal/kernels.go:291-295` / `:333-361`**: `kv_store_f32` and `attention_f32` are compiled behind `if r.kvF32 { … }` with `r.kvF32 = false` hard-coded at `metal/model.go:248` and a comment explaining they were a red herring. Two full kernels kept alive by a constant-false branch.
- **`metal.ForwardArgmax`** (`metal/model.go:524`) — the "fastest greedy decode path", but `metalResident` never implements `decoder.ResidentGreedy` (signature `ForwardArgmax(embedding []float32, pos int) (int, error)`, `decoder/residency.go:58`), so `decoder/model.go:801` can never reach it. Also the site of #13.

**Fix.** Delete, or move under `testdata/`. The `_bk`-vs-`_sa` guard asymmetry in #3 and the `M ≤ 16` gap in #18 both exist because these parallel implementations drifted.

### 20. A mid-chain CUDA launch failure returns without synchronising, and `Close()` then frees buffers under in-flight kernels

**Severity: Medium** · `cuda/resident.go:~700` (`step`), `:~556` (`launchToken`), `:~300` (`Close`)

```go
func (r *cudaResident) step(emb []float32, pos int) ([]float32, error) {
	if e := r.launchToken(emb, pos); e != nil {
		return nil, e                       // <-- no stream.Synchronize on this path
	}
	if e := r.stream.Synchronize(bg); e != nil { return nil, e }
```

`launchToken` returns early from several points after kernels for earlier layers are already enqueued. `Close()` then runs its teardown job on the executor thread and calls `.Close()` on every weight/KV/scratch buffer with no `stream.Synchronize` first — the comment at `:~295` reasons that "all of it runs ON the executor thread … and therefore before reqCh closes", which orders the *host* work but not the *device* work.

**Failure scenario.** A launch-config error mid-token (a bad shared-mem size on an unusual `hidden`, say) → `step` returns the error → the caller aborts the request and calls `Close()` → `cuMemFree` on buffers that queued kernels still reference → device-side use-after-free / `CUDA_ERROR_ILLEGAL_ADDRESS` poisoning the context for every other model in the zoo.

**Fix.** `defer r.stream.Synchronize(bg)` in `launchToken` (or synchronise on the error path in `step`), and add an unconditional `r.stream.Synchronize` as the first statement of `Close`'s teardown job.

### 21. The naive f32 matmul wastes 15/16 of every workgroup at M=1 — and M=1 is the staged decode path

**Severity: Medium (performance)** · `gpu/gpu.go:29-43`, `gpu/gpu.go:559`

```wgsl
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;                    // M
    let col = gid.y;                    // N
    if (row >= dims.m || col >= dims.n) { return; }
```
```go
pass.DispatchWorkgroups((uint32(M)+15)/16, (uint32(N)+15)/16, 1)
```

At `M = 1` — which is what `webgpuBackend.MatmulBT` → `matmulLocked` → `MatmulBTResident` (`gpu/backend.go:141-168`) does for every f32 weight on every decode step — only `gid.x == 0` passes the bounds check, so **16 of 256 lanes** in each workgroup do work. The surviving lanes also read `b[col*K + i]` at stride `K`, uncoalesced.

The W8A8 path already solved this: `MatmulW8A8` dispatches `NewGEMVRunner` at `M == 1` (`gpu/backend.go:115-128`). The f32 path has no such twin.

**Fix.** Add an f32 GEMV shader shaped like `gemvW8A8ShaderWGSL` (one workgroup per output column, 64 lanes striding the contiguous weight row, tree reduce) and route `MatmulBTResident` to it when `M == 1`. Failing that, at minimum swap the shader's axes to `row = gid.y; col = gid.x` so the writes and `dst` accesses coalesce.

### 22. WebGPU decode attention hard-wires `hd ≤ 128` with no admission check

**Severity: Medium** · `gpu/attention.go:45-46`, `:59`, `:69`

```wgsl
var<workgroup> red: array<f32, 128>;
@compute @workgroup_size(128)
fn main(...) {
    let d = lid.x;
    let lane = d < hd;                  // dims >= 128 are simply never touched
```

The header states the assumption — "workgroup_size = WG (one lane per dim, hd ≤ WG)" — but `decodeRunnerEligible` (`decoder/residency.go:127-170`) never checks `HeadDim`, `ResidentBackendFeatures["webgpu"]` has no slot for it, and `gpu/residency.go:100-141` doesn't either. A grep for any `hd > 128` / `HeadDim >` guard across `decoder/` and `gpu/` returns nothing.

**Failure scenario.** An arch with `head_dim = 256` — Gemma 3's shape — computes each score over only the first 128 dims and writes only the first 128 output dims, leaving the rest of `ctx` as stale scratch. Gemma 3 happens to decline today because WebGPU lacks `FeatSandwichNorm`/`FeatGatedGELU`/`FeatEmbedScale`, so the landmine is armed but not tripped; the moment sandwich norms land on WebGPU, a 256-head-dim model runs silently wrong.

**Fix.** Either strip-loop the score dot and the value accumulate over `d = t; d < hd; d += 128` (a 4-line change matching what `mlaAttn` already does at `gpu/mla.go:56`), or decline `hd > 128` in `gpu/residency.go`.

### 23. Metal prefill binds the **model-level** RoPE table and window while decode binds the **per-layer** ones

**Severity: Medium** · `metal/prefill.go:386-391` vs `metal/model.go:770-773`

Prefill binds `r.invf` / `r.uWindow` (model-level); decode binds `L.invf` / `L.uWindow` (per-layer, where `L.uWindow` is `win` only for `m.LayerIsLocalResident(l)` and 0 otherwise — `metal/model.go:314-319`). Prefill would therefore apply the *global* window to a local layer, and one RoPE base to a mixed local/global model.

The guard that saves it is indirect: `prefillFeatures` (`metal/model.go:25-29`) omits `FeatPerLayerRoPE`, so a model with per-layer RoPE bases declines prefill. But `FeatSlidingWindow` **is** claimed as implemented, and the only reason mixed windows can't reach prefill is that every mixed-window arch today also happens to have per-layer RoPE. That's a coincidence, not a contract.

**Fix.** Bind `L.invf` / `L.uWindow` in `PrefillLast` (they already exist per layer, so this is a two-token change), which also lets `FeatPerLayerRoPE` join `prefillFeatures` and gives Gemma a fast TTFT.

### 24. `glu_quant` round-trips the whole intermediate through global memory

**Severity: Medium (performance)** · `cuda/glue.cu:164-188`

```c
float d = a * u[uOff + k]; dscratch[k] = d; ma = fmaxf(ma, fabsf(d));
...
for (int j = t; j < I / 4; j += nt) {
    for (int b = 0; b < 4; b++) { int v = __float2int_rn(dscratch[4 * j + b] * inv); ... }
```

`dscratch` is a device buffer (`r.dScr`, `r.moeScr`, `r.shScr`), so every element is written to and re-read from global memory purely to avoid recomputing `silu(g)*u`. For `I = 11008` that is 88 KB written + 88 KB read per FFN, per layer, on a single-block kernel (`onecfg(256, 256*4)`) — while the Metal and WebGPU twins recompute the activation instead (`metal/kernels.go:388`, `gpu/decodefuse.go:131`) and pay only ALU.

**Fix.** Stage `d` in dynamic shared memory when `I·4 ≤ 48 KB`, else recompute like the other two backends. Either way, drop the `dscratch` binding and the three scratch allocations.

---

## Low

**25. `nsString` autoreleases with no pool in `NewComputePipeline`.** `metal/metal.go:215-217` calls `nsString(fn)` outside any `NSAutoreleasePool`; `BuildResident` and `ensurePrefill` wrap themselves in one (`metal/model.go:203`, `metal/prefill.go:271`) but `NewComputePipeline` is exported and reachable from tests without a pool, which on macOS logs "autoreleased with no pool in place — just leaking".

**26. `moe_route` group partitioning drops orphan experts.** `gsz = nE / nGroup` truncates in all three backends (`metal/moe.go:61`, `cuda/moe.cu:53`, `gpu/moe.go:54`); when `nE % nGroup != 0` the trailing `nE mod nGroup` experts belong to no group, so they are never masked even when their region loses group selection. Also `gscore[g] = t1 + t2` is `-inf` for `gsz == 1`. No shipped config hits either.

**27. `mambaConv` needs `dConv ≤ 9` and hangs at `dConv == 0`.** `gpu/mamba.go:93` declares `var w: array<f32, 8>` indexed `j ∈ [0, K-1)`, and `K - 1u` is unsigned — `K = 0` underflows to `0xFFFFFFFF` and the loop never terminates. `mambaRunParams.dConv` comes from the checkpoint with no validation. Real Mamba-2 uses `d_conv = 4`.

**28. The MLA rank cap is checked in the test API but not the builder.** `gpu/mla.go:405` guards `rank > 8*128` in `MLAAttn`, but `mlaAttnOp` (`gpu/decoderunner.go:~690`) records the same pipeline with no check, and `mlaAttnWGSL:48`'s `var acc: array<f32, 8>` overflows past `rank = 1024`. Every DeepSeek/Kimi variant uses `kv_lora_rank = 512`.

**29. `residentDecoder.Reset()` discards `WriteBuffer` errors.** `gpu/residency.go:~858`: if the zeroing of `lw.mambaWin` fails, the previous sequence's recurrent state carries into the next generation with no signal. `Reset()` returns nothing, so the signature needs changing too.

**30. `ForwardEmbPipe` after `Close` blocks forever.** `metal/model.go:422-429`: `execOnce` has already fired and `stopExec` set `r.execReq = nil` (`:490`), so `r.execReq <- execJob{…}` is a send on a nil channel — a permanent goroutine hang rather than an error. A concurrent `Close` during a `ForwardEmbPipe` would instead panic on a closed channel. `cmd/serve`'s per-model mutex prevents both today.

---

## What's solid

- **The feature-taxonomy admission gate is genuinely well built.** `decoder/features.go:174-247` centralises what each backend implements, `ResidentEligible` combines it with the arch predicate, and all three backends call `MissingResidentFeatures` before building (`metal/backend.go:64`, `cuda/backend.go:98`, `gpu/residency.go:117`). The per-backend comments are honest about what is *not* wired. Findings #1 and #22 are gaps in *expressiveness* (numeric caps, `head_dim`), not in the design.
- **The KV context cap is correctly enforced in all three backends, pre-write.** `metal/backend.go:102-127`, `cuda/resident.go:211-233`, `gpu/residency.go:754-770` each check before enqueueing anything, and each implements `ContextCap()` so the decode loop clamps up front. The Metal comment even explains the two distinct failure modes (`kc[p*kvDim]` overrun *and* `threadgroup float sc[4096]` overflow) it prevents.
- **Speculative decoding is correctly refused for recurrent-state models.** The resident `TruncateTo` no-op looked like a rollback hazard for Mamba state, since `ForwardN` advances `mambaWin`/`mambaSSM` in place and rejected drafts cannot be rolled back. It is properly guarded: `validateNgramSpec` (`decoder/spec_ngram.go:171`) calls `specRollbackSafe`, which returns false for `granite`/`nemotron`/`qwen35` (`decoder/forwardn.go:36-40`) with exactly this reasoning spelled out. A genuinely subtle invariant, correctly located.
- **The Metal lifetime discipline is thorough for a no-ARC binding.** The `Device.allocs`/`Device.objs` ledgers plus `ReleaseAll`/`releaseObjects` (`metal/metal.go:82-127`, `:246-278`), the `ok = false` defer that frees everything on a declined build (`metal/model.go:213-219`), the `stopExec` that *blocks* on `execDone` before freeing (`:488-494`), and the `runtime.LockOSThread` around every pool push/drain (`:386`) all address real, hard bugs — and the comments name the commit each one fixed.
- **CUDA's teardown reasoning about refcounted primary contexts** (`cuda/resident.go:~285-300`) is exactly right: it explains why releasing the context reference does *not* reclaim VRAM when another holder exists, and frees every buffer explicitly instead. The correct answer, and an unusual one to get right.
- **The GEMV kernels are the right shape and correctly guarded.** `gemvW8A8ShaderWGSL` / `gemvW4A8ShaderWGSL` / `gemv_w4a8_fwd` all use one workgroup-or-warp per output row striding a contiguous weight row, power-of-two tree reduces, `vec4`/128-bit loads, and correct tail handling (`gemv_fwd.cu:34-43`'s per-lane-guarded 32-stride remainder is exactly right, and the `if (n >= N) return` is warp-uniform so `__shfl_down_sync(0xffffffff, …)` stays legal).
- **`gemvGrid`'s 2D grid decomposition** (`gpu/gemv.go:428-433`) pairs correctly with the kernels' `n = wid.x + wid.y * 32768u`, is bijective, and the excess workgroups are bounds-guarded. The tiled GEMM's `As`/`Bs` indexing and its `(ceil(N/16), ceil(M/16))` dispatch (`gpu/gemm.go:47-70`, `:276`) are also correct.
- **`rope_kv`'s three-way thread decomposition** (`cuda/gemv_fwd.cu:99-137`) is genuinely subtle and correct: pair-per-thread to avoid the read-write race on `base[d+rhalf]`, plus the third thread group that carries the un-rotated partial-rotary tail into the cache. The comment explains *why* a thread-per-element layout would race. The WebGPU `qkvFinalize` equivalent (`gpu/attention.go:164-211`) is also correct, and its dispatch count `max(nH·half, kvDim)` provably covers all four index ranges.
- **The Gemma GELU-tanh overflow fix** (`metal/kernels.go:368-378`) — clamping the `tanh` argument to ±15 — is a real bug found and fixed with the right reasoning (saturation by |arg|~9 makes the clamp numerically exact), and the comment records the symptom it cured.
- **Numerical hygiene generally.** Every softmax subtracts a max; every int8 dot accumulates in `i32` and only then multiplies by f32 scales; every quantizer guards `sc == 0`; `qk_norm` in CUDA deliberately reduces in `double` to match the CPU's f64 (`cuda/fused_qkv.cu:202-213`). The `f16` KV word-alignment argument (`gpu/attention.go:214-220`: `base` even ⇒ each thread owns a whole word ⇒ no atomics) is correct.
- **WGSL barrier uniformity is right everywhere checked.** Every `workgroupBarrier()` sits in uniform control flow, and the reduce loops break *before* the barrier so the final value is always visible. `mambaGNorm`'s read-back of `gated[base+i]` after the reduce is safe because each lane re-reads only its own `i` stride.

---

## Suggested priority order

| # | Finding | Why first |
|---|---|---|
| 1 | **#1** — WebGPU router `min(nE, 256)` clamp + missing `nGroup` cap | Only *live* silent-wrong-output path on a documented model (Kimi-K2). One `if` in `gpu/residency.go` + one line in the shader. |
| 2 | **#2** — no GPU error checked on Metal/WebGPU | Cheap, and it converts findings #3/#10/#13 (and every future one) from silent to loud. Do it before anything else invasive. |
| 3 | **#7** — CUDA `CopyHtoD` errors dropped | Three lines. Turns a garbage-weights model into a clean CPU decline. |
| 4 | **#6** — CUDA `f32tof16` truncation | One line for the truncation; then hoist one shared converter and delete the other two. Removes a permanent cross-backend parity offset. |
| 5 | **#5** — lane-0 activation quantizer | Highest ratio of speedup to diff size in the whole audit; the correct pattern already exists two files away. |
| 6 | **#4** — per-key barrier storm in WebGPU attention | Biggest single perf win, dominant at long context, but a real kernel rewrite. Land after #5 so the profile is clean. |
| 7 | **#3** + **#13** — Metal Stage-A tail guard and `nTiles` rounding | Same root cause (unguarded divisibility invariants). Add `row >= N` to all seven variants and `(V+7)/8`. |
| 8 | **#8**, **#9**, **#23**, **#22** — divergent `K`/padding/window/`head_dim` contracts | Batch as one "make the three backends agree on shape admission" pass; each is small and each closes a latent silent-wrong-output door. |
| 9 | **#12**, **#11**, **#14**, **#16**, **#17**, **#21**, **#24** — parallelisation and coalescing | Independent perf work, orderable by profile. #16 (SSM transpose) and #21 (f32 GEMV) are the largest wins per line changed. |
| 10 | **#15**, **#20** — prefill buffer pooling, CUDA teardown sync | Serving-path robustness and TTFT; matters once multi-model serve is under load. |
| 11 | **#19**, **#18** — delete dead code, add the missing `M` guard | Do this *last* but do it: the `_bk`-vs-`_sa` and `MatmulW8A8GemmRow`-vs-`DecodeTokenFusedBatched` guard asymmetries both exist because these duplicates drifted, so removing them prevents the next instance of #3 and #18. |
| 12 | **#25–#30** — Low | Opportunistic. #27 (`dConv` underflow → hang) and #30 (nil-channel hang) are the two worth a defensive line each. |
