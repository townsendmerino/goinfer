# Decode-perf via kernel fusion — status & next steps

The decode path's throughput ceiling on WebGPU is the long, serially-dependent
dispatch chain, where per-dispatch barrier cost dominates over kernel work. Cutting
dispatch/barrier count is the lever for cogentcore today.

> **Correction (2026-09-02):** this line read "~535 dispatches/token: ~197 GEMV + ~338
> glue" — the pre-Increment-1/2 count. **Measured now: 366/token = 12 per layer** (28
> layers) + 30, across 7 pipeline classes, on the resident 1.5B. See G35 in
> `docs/QUEUE.md`. The correction matters because "cut the dispatch count" was sized
> against a chain roughly twice as long as the one that exists.
>
> **And the premise is now measured, not assumed.** G35's ablation profile puts the
> whole remaining elementwise-glue budget at ~28% of the token at pos 64 and ~12% at
> pos 512, against **attention at 65% by pos 512**. Dispatch/barrier count across the PLAN is no
> longer the lever it was when this doc was written.
>
> **Barrier count INSIDE one kernel still was, though — G36 (2026-09-02).** The attention kernel
> reduced once per key (9 `workgroupBarrier()` × nKeys per layer) where `cuda/attn_block.cu`
> reduces twice per layer. Splitting the workgroup over KEYS instead of head dims measured
> **9.2× on the kernel, 2.39× on the token at 1k context**, and flattened decode's fall-off with
> context from −54% to −12% (128→1024 tokens). Note what that says about this doc's framing: the
> plan-level dispatch chain was the wrong altitude to look at, twice — G35 found a bad reduction
> inside `quantize`, G36 a bad decomposition inside `attn`.

> **Correction (2026-06-19):** an earlier draft claimed a "wgpu-native v29 decode
> penalty" that this fusion would also shrink. That penalty was **measured and does
> NOT exist** — via the `oliverbestmann/webgpu` CGO v29 fork on the box, the
> per-dispatch cgo record cost is **identical** to cogentcore/v22 (1.1 µs/dispatch
> both) and gemv compute is ~equal (v29 0.64 ms vs v22 0.62 ms). So fusion's value is
> the dispatch-chain throughput on the *current* binding, not a v29 hedge.

## Done — Increment 1 (shipped, `e8add57`)

Fused `rope(q) + rope-store(k) + store(v)` → one `qkvFinalize` dispatch (f32 KV;
f16/int8 KV keep their packed-store kernels). Bit-exact (full parity suite green).
**−56 dispatches/token; decode 89.8 → 91.1 tok/s (+1.5%)** on the synthetic 1.5B-int8
bench, identical across runs.

### What it taught us (recalibration)
The gain was smaller than the naive ceiling because **q/k/v-finalize are INDEPENDENT**
(different buffers) — the GPU already overlapped them, so fusing saved per-dispatch
*launch/record* cost, not *barrier* cost. The expensive barriers are between
**dependent** dispatches. So the rule for future increments: **fuse dependent links,
not independent ones.**

## The real-model bench (shipped, `901b8aa`)

`TestDecodeRealModel_throughput` loads a real GGUF model GPU-resident (int8) and
measures steady-state inter-token rate (best-of-rounds, prefill excluded). Defaults to
Qwen2.5-coder-1.5B; `GOINFER_DECODE_GGUF` overrides. This is the gate the family-specific
folds (2 & 3) needed: the synthetic `TestDecodeToken_throughput` dispatches no bias /
qk-norm, so it can't see them. **Baseline (Increment 1 in): 102.2 tok/s.**

## Done — Increment 2 (shipped, `38babc4`)

q/k/v bias → GEMV epilogue: `gemvW8A8Bias` adds a 7th binding and computes
`dst[n] = scale·acc + bias[n]`, deleting the three standalone `biasAdd` dispatches/layer
(84/token on the 28-layer 1.5B) for **Qwen2/Qwen2.5** bias models. A *dependent*-chain
fusion (biasAdd reads the GEMV output) → removes barrier cost. W8A8 only; W4A8+bias falls
back to gemv+biasAdd. Bit-exact: `TestResidentForwardN_parity` (Qwen2.5-coder-0.5B)
cosine=1.0, maxAbsDiff=0. **Measured: 102.2 → 104.5 tok/s (+2.3%)** — beats Increment 1's
+1.5%, confirming dependent folds pay more than independent ones.

## Tried + rejected — Increment 3 (QK-norm fold REGRESSES, −2.7%)

Measured on Qwen3-1.7B-Q8_0 (downloaded for exactly this; resident int8, 16 q / 8 kv
heads, hd 128). Built `qkNormFinalize`: one **workgroup-per-head** dispatch doing
per-head qk-norm + rope(q) + rope-store(k) + store(v), replacing the two `qkNorm`
dispatches *and* `qkvFinalize`. Bit-exact (`TestResidentForwardN_parity` cosine=1.0,
maxAbsDiff=0; qknorm-vs-CPU parity green). But decode went **104.6 → 101.8 tok/s
(−2.7%)**, consistent across runs — so it was reverted (commit never landed).

**Why it loses (the lesson the bench bought):** qk-norm is a per-head reduction over
headDim, so it *must* be workgroup-per-head — only nH=16 workgroups on a 40-SM card,
low occupancy. rope/store want the opposite: a flat, high-occupancy grid (that's what
`qkvFinalize` is). Fusing them drags rope into the low-occupancy per-head geometry AND
serializes two reductions (q then k) behind barriers inside each workgroup. The
dispatch-count cut (3→1) didn't pay for the occupancy/serialization loss. This is the
mirror image of Increment 2, which folded into the GEMV — already workgroup-per-output
high-occupancy, with a trivial `+ bias[n]` epilogue.

**Corollary to the Increment-1 rule:** "fuse dependent links" holds *only if the fused
kernel keeps the better launch geometry*. Folding a reduction-shaped op (qk-norm,
softmax, any cross-element reduce) into a flat elementwise kernel — or vice-versa —
trades dispatch count for occupancy and can net out negative. Dispatch-count reduction
is necessary, not sufficient. The real-model bench is what caught this; the
"dependent-fold = win" heuristic alone would have shipped a regression.

GLM-4.5 / Mellum share the qk-norm geometry, so the same fold would regress them too —
not worth retrying without a fundamentally different approach (e.g. a higher-occupancy
norm that doesn't need per-head workgroups).

## Hard ceiling

WGSL cannot fuse across the **matmuls** (workgroup-per-output geometry ≠ the
elementwise glue geometry) — the "single megakernel per layer" that CUDA/Metal can
express. So decode-fusion headroom is bounded to the glue/elementwise chain; the GEMV
stream (~4.3 ms/token, ~98% of the bandwidth roofline) is already optimal and unfusable
further here.

### Confirmed against the CUDA spike's stage-grouping (2026-09-02)

The obvious next question — can `docs/cuda-megakernel-spec.md` §5.2's K1/K2/K3
super-kernel grouping be built as 3 WGSL dispatches, since it was designed specifically
to avoid needing grid-wide cooperative launch? — is **NO**, and the ceiling above is why.
K2 (attention ⊕ quant ⊕ O-proj) needs a grid-wide sync: attention is one workgroup per
head, O-proj needs every head. `storageBarrier()` orders memory *within* a workgroup and
is not an execution barrier across them, so WGSL cannot express it; the redundant-recompute
escape multiplies KV traffic by the O-proj block count (~24×) on the term that already
dominates. K3's SwiGLU fold is expressible but bandwidth-fatal (CUDA measured ~6.9 MB/layer
against a 4 MB L2). The reachable floor is **8 dispatches/layer — exactly where CUDA landed**,
which also shipped K1+K3a and **reverted K2 at a measured ~0%**.

That makes the fusion arithmetic decisive on its own: K1's whole target here, `rmsQuant`,
is **5.2% of the token at pos 64 and 2.2% at pos 512**. CUDA's K1+K3a bought **+1.5% on the
1.5B**. NO-GO, and the reason is the size of the prize, not just the expressiveness wall.

**What the profile found instead** — `quantize` was doing its row max-abs as a serial scan
on lane 0 in a single-workgroup dispatch (37 µs/dispatch vs `rmsQuant`'s 7.9 for more work).
The comment justified it as "trivial at decode; the rows run in parallel"; decode is M=1, so
there is exactly one row and nothing runs in parallel. A 64-lane tree reduce — the idiom
already in `rmsnormQuantWGSL` — is **bit-identical** (f32 `max` is exact and order-independent)
and measured **104.8 → 118.4 tok/s (+13.0%)**, same-session interleaved A/B. Full numbers and limits: G35 in `docs/QUEUE.md`.

**The lesson, which is this doc's own rule turned one level inward:** "fuse dependent links,
keeping the better launch geometry" treats each dispatch as a fixed unit with a geometry.
The largest available win was a *bad geometry inside one kernel* — invisible to dispatch
counting, and it survived precisely because it sat next to a kernel that already did it right.
