# goinfer Metal audit — 2026-09-12

**Tree:** goinfer `da1e461` (2026-09-12, `main`) + aikit `d295ba5` (2026-09-11, v1.41.0 +2). Every
`path:line` is keyed to those commits. The 53 commits touching `metal/` and `decoder/` since the
2026-09-10 whole-repo audit (`c7ef16a`) were read with `--stat`.

**Ask:** a focused, *performance-led* audit of the Metal backend — the cgo-free Apple-GPU path
(`metal/` module: 14 non-test Go files, 6,796 lines of which ~3,400 are MSL embedded as Go strings,
69 kernels; 135 test files; plus the aikit `gpu/` Metal bindings and shaders goinfer dispatches).
Metal is Francis's main machine (M1 Pro, 16 GB, 16-core GPU, ~200 GB/s) and the backend furthest
behind its peer. Correctness findings stay in scope, reported second.

**Method:** six independent reviewers (decode kernels; prefill + attention; MoE/streaming/DeltaNet;
executor + runtime; the aikit Metal seam; docs vs code + gates), each required to read the
recorded Metal negatives (`metal-verdict.md`, the cgo-free spike, the autoresearch rounds in the
kernel comments, `pagecost_*` harnesses) *before* the code so no recorded dead end is re-proposed.
Every finding cites lines the reviewer read; every cost is bounded from the code's shape or from a
number the repo itself recorded. The consolidator re-read the source for every Major and every
correctness item below.

**Limitations:** static only — no macOS, no Metal device, no GPU here. Nothing was built, run or
timed; every "×" and "ms" is either the repo's own measurement (cited) or a count from the code
(marked *counted*). The FIRST-RUN gate ledger, `.github/workflows`, `decoder/weightbytes.go`,
`scripts/autoresearch_rmsnorm_results.tsv` and goinfer's own CHANGELOG were not in the snapshot;
findings that depend on them say so.

---

## 0. The shape of it

**The recorded TTFT gap is one kernel generation stale, and what remains is a different term than
the docs say.** `docs/benchmarks.md` §A (2026-09-09) shows 3.3× behind Ollama at K=512 and 8.8× at
K=3900, and attributes the growth to O(K²) attention. That row was measured with the *exact*
prefill-attention kernel; the fused `simdgroup_matrix` kernel became the default the next day and
the tree's own record (`prefill-l2-metal-fused-attn-2026-09-09.md`) puts P=3900 at 17.9 s, ≈4.3×
behind. At this tree attention is ~18% of TTFT at K=3900 and ~1.5% at K=256. **The term that keeps
goinfer 3–4× behind Ollama at every depth is flat: the f16-MMA GEMM runs at ~0.73 TFLOPS
(`gemm_w4f16_store`: 8 of 32 lanes dequant each weight tile, 4 MMAs per barrier pair, no operand
staging) against Ollama's ~1.0 ms/token on the same weights.** That is M-03, and it is the largest
single lever on the Mac.

**Below the 512-token floor — the common chat turn — the fast path never runs at all, and the floor
is set on a premise the repo's own gate record contradicts.** `metal/backend.go:329-333` says "K=256
is expected to fail §3.2"; `prefill-gate-l1-ref-b-2026-09-09.md:23` records K=256 **PASS** as a
decision cell, and the fused-kernel gate passed it again. Every prompt in [8, 512) runs the decode
kernel per token at 74–77 tok/s where the batched path's own record is 272 tok/s at P=256 (M-02).
And on that sequential path — which is also every adapter prompt, every declined family including
all of Gemma 3 (M-06), and every embedding request — **each prompt token runs the full int8 LM head
and a 608 KB logits readback for logits the decoder discards**, because `metalResident` never
implemented `ResidentPrefillKV` although the head-less trunk (`forwardHiddenNoHead`) already exists
(M-01; four of six reviewers found it independently). 233 MB of head weight per prompt token on
the 1.5B, 671 MB on a Gemma-class vocabulary.

**Decode is at the launch/sync ceiling the design allows.** One command buffer, one commit, one
`waitUntilCompleted`, 310 dispatches and ~1,030 purego transitions per token; encode-ahead holds on
the production path; the W4A8 GEMVs are byte-minimal and issue-bound in the peer's own band. Two
plausible levers remain (decode attention's strided K gather, M-09; the down-proj on the unstaged
kernel, M-10), each gated behind a 30-line discriminating probe because the recorded Stage-B
precedent says isolated GEMV wins can vanish end to end.

**Memory: a Metal-backed int4 load keeps three copies of every dense projection** — host canonical,
host row4 repack (read by nothing once resident), and the Metal buffer — 889.6 MB of dead host
memory on the 1.5B by commit `3931ae1`'s own measurement, ~4 GB at 7B, on the box whose 26B cell
misses the budget by 3.75 GB (M-07). This is L4 of `docs/task-int4-layout-2026-09.md`, now with
its number.

**The paged path (26B/35B/gpt-oss on the Mac) is opt-in, serialised, and measured against the wrong
shape.** The Metal expert pager engages only with `--moe-cache-slots N` (no auto-sizing; the
recorded N=64 recommendation is not a default), so the peer-matrix row that "parked" M35/M26 on the
Mac ran the CPU-staged fallback for 2h10 with zero completions (M-12). When it does run, the token
is 2L+1 synchronous command buffers at a recorded ~14 ms boundary each, staging is queue-depth 1,
and the one design that removes the boundary — a single command buffer with a shared event — was
rejected on a measurement taken on a dense 1.5B where the boundary costs 0.2 ms (M-08, M-11, G-05).

**Correctness held on the dense decode path** (C-07/C-08 verified fixed; encode-ahead/KV ordering,
tails, alignment all checked). Six items remain, none on the default dense shape: the fused prefill
kernel reads past the cache end when `ctxCap % 8 ≠ 0` (C-01); `HiddenLast`, `Forward` and
`ForwardArgmax` on a paged MoE bind zero-value expert buffers — C-08's defect on three more entry
points (C-02); adapter `In`/`Out` still unchecked on Metal (C-03); a binding helper without the
G22 thread pin (C-05).

**Gates:** `go test ./metal/` has been red since the floor landed (`TestMoE_declinesPrefill`, three
commits note it as "pre-existing") — a permanently red suite gates nothing (G-01). The §3.2 pooled
gate still drops missing cells and SKIPs a model that fails to build (G-02); the Gemma fixture is
shaped so it cannot see the decline it exists to catch (G-03); the absolute snapshot golden was
re-baked by the code it checks (G-04).

---

## 1. Performance — Major

### A. Prefill: the whole-ladder gap and the short-prompt band

#### M-01 · `ResidentPrefillKV` is not implemented on Metal — every sequential prompt token runs the full int8 LM head and a 608 KB readback for logits nobody reads
- **Where:** `decoder/model.go:1111` (`kvOnly, hasKV := m.resident.(ResidentPrefillKV)`),
  `decoder/residency.go:100-109`; `metal/backend.go:277-560` (the complete `metalResident` method
  set — no `ForwardNoLogits`); `metal/model.go:1378-1384` (`encodeLogitsCB`, the only executor job
  shape, always appends `pGemvW8`); `metal/model.go:1262-1281` (`forwardHiddenNoHead` — the
  trunk-only encode already exists, used only by `HiddenLast`); `metal/model.go:1304`
  (`finalizeLogits` memcpy).
- **Mechanism and bound (counted + record):** `hasKV` is false for `*metalResident`, so
  `residentPrefillSeed` takes `m.resident.Forward(emb, i)` for every prompt token. Which prompts
  are sequential on Metal: every prompt below the 512 floor (M-02), every adapter prompt at any
  length (`decoder/model.go:1086`, C-01 of the prior audit), every family `prefillOK` rejects
  (Gemma 3 — M-06 — every DeltaNet family, gpt-oss, GPT-2, Cohere, Olmo, SmolLM3, Ministral 3,
  Mellum, dense Gemma 4, paged MoE), every `HiddenLast` embedding token. Per prompt token: 1.5B
  V×H int8 = 233 MB (≈24% of the ~0.97 GB the token moves) + 608 KB copy; 0.5B 136 MB of ≈420 MB
  (≈⅓); Gemma-class 262k vocab 671 MB + 1 MB + a 262k-element `tanh` softcap on the host. At the
  record's 84 GB/s head rate that is ≈2.8 ms of the 12.9 ms sub-floor token (§A K=256: 77.3
  tok/s) — ≈22% of sub-floor TTFT. A 256-token prompt: ≈3.3 s → ≈2.6 s from this alone.
- **Fix:** `ForwardNoLogits` on `metalResident` as a `noHead` bit on `execJob`, so `encodeLogitsCB`
  skips the head dispatch and `execLoop` skips `finalizeLogits` for that job — this keeps
  encode-ahead (a synchronous `forwardHiddenNoHead` wrapper works but gives back ~0.9 ms/token of
  un-overlapped encode). Pre-encode the next buffer in the same mode; re-encode on a mode switch
  (one bubble per prompt). **Constraint found in verification:** `forwardHiddenNoHead` →
  `encodeTrunkInto` → `encodeLayer` never takes the paged branch that `Forward` →
  `forwardLogitsPaged` does (C-02), so the paged families need "the paged forward minus its
  tail", not the trunk encoder. Second step at the same seam: a prefill job's token is known, so
  the executor can commit t+1 without waiting for t (wait only on the last) — the "`ForwardN` is a
  loop" defect `theta-per-backend-2026-09-01.md` records.
- **Confidence:** confirmed (four reviewers; interface absence and dispatch traced end to end; the
  2.8 ms is the record's head rate, not a measurement here).
- **Prior:** new as a mechanism; audit-2026-09-10 N-49 and the Metal TTFT Major name the floor, not
  the per-token head. CUDA has the twin (`TestKVOnlyPrefill_byteIdentical_tiny`).

#### M-02 · The 512-token floor keeps every prompt under 512 tokens sequential, on a stated reason the repo's own gate record contradicts
- **Where:** `metal/backend.go:329-333` (`const metalFastPrefillFloor = 512` — "K=256 is expected to
  fail §3.2"), `:426-431` (the decline), `:340,358` (the same file: "gate passed 2026-09-09 (S
  model, K=256/512/1024)"); `docs/measurements/prefill-gate-l1-ref-b-2026-09-09.md:23` ("S K=256 …
  94.2% / 4 / 0.0328 vs exact 93.0% / 4 / 0.0347 **PASS**"), `prefill-l2-metal-fused-attn-2026-09-09.md:138`
  (fused gate, K=256 pass again); `docs/task-prefill-gap.md:226-227` ("Moving the floor DOWN
  requires a passing gate cell at the new depth").
- **Mechanism and bound (record):** the K=256 failure the comment cites is CUDA's combined L2+L3
  run; the floor was "set to 512 to match CUDA's measured floor". On Metal the cell passed twice.
  What the floor costs: every prompt in [8, 512) runs the decode kernel per token — 310 dispatches,
  the full head (M-01), one wait — at the recorded 74–77 tok/s, against the batched path's own
  measured 272 tok/s at P=256 (941.5 ms). K=256 is the worst cell on the ladder (10.2× behind
  Ollama); a 500-token prompt pays ≈6.7 s instead of ≈1.8 s.
- **Fix:** `metalFastPrefillFloor = 256` — the depth with a passing decision cell on both gates —
  and correct the `:329-333`/`:426` comments. Going lower needs one more gate cell, which the
  harness already parameterises (`FLOOR=0`).
- **Confidence:** confirmed.
- **Prior:** new; prefill-gap §3 "Floor vs decision set" is the governing rule and it is satisfied.

#### M-03 · `gemm_w4f16_store` dequants each weight tile with 8 of 32 lanes, runs 4 MMAs per barrier pair, and stages neither operand — the flat 3.3–3.6× GEMM term at every K
- **Where:** `metal/prefill.go:66-119` (kernel; `if (lane < 8u)` dequant at `:87-92`, `RPS 4` at
  `:23`, per-k-step barrier pair), `:590-596` (grid), `docs/task-prefill-gap.md:159-162` ("Metal's
  GEMM is already simdgroup_matrix f16 MMA" — and stops there).
- **Mechanism and bound (counted + record):** one simdgroup owns a 32(M)×8(N) block and walks K in
  steps of 8. Per k-step: 8 lanes do 8 nibble extracts + 8 f16 converts each while 24 lanes idle →
  barrier → one threadgroup load + four 8×8 A-tile loads straight from device memory (8 rows × 16 B
  at stride 2K — 8 cache lines per tile, no staging, no double-buffering) → 4 MMAs (2,048 MACs) →
  barrier. The int4 matrix is re-dequanted ⌈Mpad/32⌉ times per GEMM (122× at M=3900; ~160 G
  nibble-dequants per prefill on 25%-occupied SIMDs); the A panel is re-streamed N/8 times (2,240×
  for gate/up) from L1/L2. Barriers per MMA: 0.5. Recorded outcome: the tree's own post-flip TTFT
  (P=256 941.5 ms, flat 3.6–3.7 ms/token) is 2.62 GFLOP/token → **≈0.73 TFLOPS**; Ollama's Metal
  `mul_mm` on the same q4 weights: ~1.0 ms/token (`benchmarks.md:536`, 953 tok/s) → ≥2.4 TFLOPS.
  At K=512 this kernel is ≥95% of TTFT, so the recorded 3.33× at K=512 *is* this kernel.
- **Fix (at the seam, not a new kernel):** widen the per-simdgroup block to 32×32 (RPS=4 rows × 4
  N-tiles): all 32 lanes dequant four weight tiles per k-step into scratch, 16 MMAs per barrier
  pair, A tiles reused across four N-tiles (dequant lanes 4× busier, barriers/MMA 4× fewer); stage
  the 32-row A panel for a 32-wide K slab in threadgroup memory once per threadgroup (its 8
  simdgroups already share those rows). Gate: raw-bit vs the current kernel is not required (f16
  storage rounding already differs from exact) — the §3.2 pooled gate is the oracle, and the
  parity fixture must stay inside it.
- **Confidence:** confirmed (shape traced; the TFLOPS figure is arithmetic on the repo's own
  record; the ceiling is the repo's own peer row).
- **Prior:** new. No autoresearch round touched this kernel (`theta-*`/`uploadbatch-*` are decode
  GEMV/upload). `benchmarks.md:546-550` attributes the whole gap to attention — stale (N-01).

#### M-04 · `attention_prefill_fused` keeps O in threadgroup memory and rescales it with 8 scalar lanes — ~34 barrier-separated phases per 8-key tile, the residual O(K²) term
- **Where:** `metal/prefill.go:324-329` (`threadgroup float oScr[ATTN_SGPT][8*ATTN_MAXHD]`),
  `:342-400` (tile loop; `if (lane < 8u)` softmax at `:352-380`, per-`cc` `pTile` reload + store +
  barrier + 8-lane rescale at `:383-399`), `:601-604` (grid = nH × ⌈M/8⌉ simdgroups).
- **Mechanism and bound (counted + record):** Σ tiles ≈ nH·M²/128 iterations per layer (1.43 M at
  S/M=3900). Per iteration at hd=128: 16 QKᵀ MMAs, store+barrier, 8-lane softmax (8 exp per lane,
  24 lanes idle), barrier, then 16 × {reload `pTile` (redundant), load `vTile`, MMA, store,
  barrier, 8-lane rescale of 8 floats through `oScr` (128 threadgroup RMW per lane per tile),
  barrier} — **34 barriers and ~136 dependent scalar threadgroup-memory ops per lane per 8 keys
  against 32 MMAs of useful work.** `oScr` alone is 16 KB per 128-thread threadgroup, bounding
  concurrent threadgroups per core and the latency hiding that would mask the chain. K/V are read
  (nH/nKV)=6× redundantly across the GQA group because tiles are keyed by query head. Recorded
  isolated: 118.76 ms/layer at M=3900 → 3.3 s of the 17.9 s TTFT (18%); it is what makes the
  1024→3900 marginal 4.86 ms/token vs 3.95 at 256→1024.
- **Fix:** hold O as `simdgroup_float8x8 oAcc[hdTiles]` in registers and apply α via one MMA with an
  8×8 diagonal-α tile (deletes `oScr`, 16 stores and 32 barriers per tile); widen the key tile to 32
  (4 S tiles per softmax phase) so the scalar phase is amortised 4×; group the 6 query heads of a
  kv head into one threadgroup so K/V tiles load once per group. The record already flags
  reduction-order as a fidelity decision — re-run the §3.2 gate.
- **Confidence:** confirmed (shape); the per-iteration cycle estimate is plausible (no Metal
  profile exists; the CUDA twin's "1.72% of tensor peak" is an `ncu` figure).
- **Prior:** prefill-gap §4 L1 last bullet; `…fused-attn…md` (built, 5.45× over exact at the
  kernel); `benchmarks.md` §A not re-run since the flip.

#### M-05 · MoE batched prefill runs the FFN half as M sequential rows; paged/DeltaNet families prefill as M decode tokens — bounded by M × active-expert bytes, undocumented
- **Where:** `metal/prefill.go:651-674` (`for m := 0; m < M; m++ { … r.encodeMoEExperts(e, L, moeDst) }`),
  `metal/moe.go:643-663`; `metal/model.go:681-682` (paged/g4moe/DeltaNet → `prefillOK=false`);
  `metal/backend.go:396-408` (`PrefillPath` reports "batched f16-MMA" for it);
  `docs/task-gpu-paths-2026-09.md:1184-1191` (G8: "Mirrors CUDA's own established shape exactly").
- **Mechanism and bound (counted):** non-paged: per row per MoE layer (5 + 3k [+3–5 shared])
  dispatches and a full read of the k routed experts — bytes ≈ M × L × k·3·H·I/2: Qwen1.5-MoE-class
  (24 L, k=4, H=2048, I=1408) ≈ 0.83 GB per row → 1.7 TB at M=2048 (≥8.5 s at 200 GB/s) plus ~0.84 M
  dispatches (≈6 s at the recorded ~7 µs GPU-side floor); expert-major would read each expert once
  per prefill ≈ 6 GB. Paged (26B/35B): TTFT = M × the paged token (512 tokens ≈ 3.9 min on the 35B)
  with M·k·(1−hit) stages per layer vs ≤ nE if rows were grouped by expert, plus the full head per
  row (M-01). Deliberate and documented for parity; the bandwidth term — which dominates the
  dispatch term for k ≥ 4 — is not.
- **Fix:** for the paged families a layer-major prefill loop (mixer/attention per row inside the
  layer; route all M rows, group by expert, stage each distinct expert once, run its rows with the
  batch-K GEMM); the same grouping replaces the row loop for the non-paged case. Chunk M to bound
  the residual buffer. Until then, record the bound beside G8 so "batched" is not read as a TTFT
  promise.
- **Confidence:** confirmed (shape); bytes are arithmetic on config, unmeasured (P-15 is un-run —
  G-09).
- **Prior:** audit-2026-09-10 P-15 (dispatch count only); queue-performance P20/P18 (CPU
  expert-major 4.36×, bit-identical); CUDA `prefill-moe-m26` (batching worth 8%, expert DMA 59.5%).

#### M-06 · Gemma 3 never reaches the batched prefill — `prefillFeatures` still lacks `FeatPerLayerRoPE` (prior audit M-23, open)
- **Where:** `metal/model.go:111-122` (the map: no `FeatPerLayerRoPE`), `decoder/features.go:152`
  (`add(!a.ropeUniform(), FeatPerLayerRoPE)` — every shipped Gemma 3, 5:1 local/global, derives it),
  `metal/prefill.go:620-627` (the dispatch already binds `L.invf`/`L.uWindow` per layer; the comment
  at `:624-625` says the feature "is not claimed").
- **Mechanism and bound:** every real Gemma 3 prompt is sequential: M-01's 671 MB head per token on
  4B, at 13.5 ms/token. Note admitting it would still route Gemma 3 (hd=256) to the *exact*
  `attention_prefill` (`ATTN_MAXHD 128`, `prefill.go:606`) — the 46 GB/layer re-read shape — so the
  fused kernel needs an hd=256 variant for the full win.
- **Fix:** `decoder.FeatPerLayerRoPE: true` in `prefillFeatures` (safe: `FeatRopeMscale` stays
  undeclared so per-layer *mscale* families still decline); give `testdata/gemma3-vl-tiny` a global
  layer so `TestPrefillParityGemma` covers the real shape (G-03); then an hd=256 fused variant.
- **Confidence:** confirmed. **Prior:** audit-2026-09-10 M-23 — unchanged by the 53 commits.

### B. Decode: two probes, one adapter kernel, one memory item

#### M-07 · A GGUF/safetensors int4 load on `--backend metal` keeps THREE copies of every dense projection: host canonical + host row4 repack (read by nothing once resident) + the Metal buffer
- **Where:** `decoder/weightmat.go:414-425` (`wantsCanonicalInt4`: `if backendName != "cpu" { return
  true }`), `:440-444` (`repackedOnlyOrCanonical` → `repackW4A8IfEligible(canon)` — both kept),
  `:251-257` ("both ALLOCATE A SECOND BUFFER and keep the canonical nibbles alongside"),
  `metal/model.go:443-444,475-478` (`int4DirectWords` → `NewBufferUint32s` = `newBufferWithBytes`,
  a third copy); `decoder/fitguard.go:251-260` (the guard prices int4 at ~2× on arm64 because of
  row4); commit `3931ae1` (log: "2365.1 MB (Backend:"cpu") vs 3254.7 MB (unspecified) — 889.6 MB
  saved").
- **Mechanism and bound (record):** once resident the GPU reads only the MTLBuffer; the host row4
  copy is read only by the CPU fallback (the `resBusy` loser, or a declined build); the host
  canonical only by `Embed.Row`. Measured on this Mac's 1.5B: **3,254.7 MB host** for a model whose
  int4 weights are ≈0.78 GB, plus ≈0.87 GB device → ≈4.1 GB resident for a 1.5B. Scaled to the 7B
  (D7, 4.4 GB int4): ≈13 GB before KV — the cell the §3.2 decision skipped as "fit-guard on 16 GB"
  (G-02), and why no 7B-class Metal number exists. Gemma-4 26B-A4B misses the 11.2 GB budget by
  3.75 GB (`benchmarks.md:1687-1688`); whether dropping row4 alone brings it under depends on its
  dense share (`weightbytes.go` not in tree).
- **Fix:** for a resident GPU backend return canonical-only from `repackedOnlyOrCanonical` (skip
  the row4 repack — the CPU fallback then runs the canonical kernel, ~1.1–1.3× slower on a path that
  is already the slow one); after `BuildResident` succeeds, release the layer projections' host
  `WeightMat`s (keep `Embed`/`LMHead` for the host lookup). This is **L4 of
  `docs/task-int4-layout-2026-09.md`**, now with its number and a second half (drop after upload).
- **Confidence:** confirmed (three allocation sites traced; the 889.6 MB is the commit's own
  measurement). **Prior:** audit-2026-09-10 M-24 (the guard's 2× double-count — the real footprint
  is 3×); aikit M-22; task-int4-layout L4 (filed, not scheduled — schedule it).

#### M-08 · `lora_delta` runs each projection's whole adapter delta in ONE threadgroup — the design CUDA's P-11 measured at +124%/token; Metal's P-11 fused two dispatches and kept the serial block
- **Where:** `metal/kernels.go:834-873` (`// ONE THREADGROUP ONLY … looping over ranks serially …
  is cheap`; `for (uint r = 0; r < R; r++)` with a 256-wide tree reduce + barrier per rank; up
  stage `Out/256` rows per thread), `metal/lora.go:198-203` (`e.Dispatch(r.pLoraDelta, tgReduceNorm,
  tgReduceNorm, …)` — n == tg == 256, one threadgroup); `docs/audit-2026-09-10.md:1130-1176` (CUDA:
  "ONE block that looped over every rank, pulling ~61 MB of f32 A matrices per token through a
  single SM … 20.06 ms vs 8.89 ms"; fix "strides ranks by `blockIdx`"; Metal disposition "the same
  class of fix").
- **Mechanism and bound (counted; CUDA record for scale):** an adapter on all seven projections
  adds 196 single-threadgroup dispatches per token (+63% over 310), each streaming f32 A and B
  through one of 16 cores with 9 barriers per rank: ≈4.6 MB·rank per token → **74 MB/token at
  r=16, 295 MB at r=64** (a third of the base traffic) serially. CUDA measured the same shape at
  +11 ms on an 8.9 ms token. Since C-01, adapter prompts are also sequential, so every prompt token
  pays this on top of M-01. No Mac adapter-decode cost measurement exists (the P-11 record is
  parity only).
- **Fix:** launch `ceil(Out/256)` threadgroups, each recomputing `t[R]` (R·K MACs — negligible; A is
  L2-resident after the first group) and owning 256 output rows: no scratch, no second dispatch,
  up stage parallel over Out. Store A/B as f16 (the delta feeds an int8-quantised activation).
- **Confidence:** mechanism confirmed (two reviewers); Metal magnitude plausible (unmeasured on the
  Mac). **Prior:** audit-2026-09-10 P-11 — disagree with its Metal disposition; task-gpu-paths G3.

#### M-09 · Decode attention's K read is a 32-lane 512 B-strided gather (32 load instructions per 256 B row) — every recorded probe fits an L1/LSU-transaction wall as well as the "DRAM latency" reading; the one untested corner is bit-identical cooperative staging
- **Where:** `metal/kernels.go:628-636` (thread `tid` owns keys `tid, tid+128, …`; per key 32 `half4`
  loads at stride `kvDim*2` = 512 B across lanes), `:661-677` (V: thread `d` walks all keys
  serially, 2 B per load), `metal/model.go:1989` (12 threadgroups × 128 threads per layer);
  `docs/completed/metal-verdict.md:95-99,136-145,175-178`; `metal/attn_m3_probe_test.go`,
  `attn_kvwidth_probe_test.go`.
- **Mechanism and bound (counted + record):** each K-row load instruction touches 32 distinct cache
  lines for 256 useful bytes; if the lines do not survive in L1 across the 4 co-resident
  simdgroups, L2 traffic is ~16× the K bytes — ~100 MB/layer at depth 2048, ~2.8 GB/token, enough
  alone for the ~10 ms scores pass the record shows. Every record is consistent with this reading
  *and* with the DRAM-latency one: the half4 rewrite (same bytes, 8 B loads) won 1.79× — a purely
  latency-bound loop would not; the q8 probe (same 32 lines/instruction) moved 12% — both predict
  that null; the width probe (more simdgroups on the same core) was flat — an L1 wall predicts that
  too. The verdict's refuted "threadgroup-staged K/V (0.23×)" entry is the *deduped* variant (2
  threadgroups per KV head); staging at the shipped 12-threadgroup grid without dedup is not on the
  ledger. Bit-identity holds by construction: only the load pattern changes; the per-key serial
  f32 dot and the per-dim serial V sum stay. At depth 2048 attention is 21.5 of 36.8 ms/token.
- **Fix:** **a 30-line discriminating probe first** (next to `attn_pin0`): the shipped K loop vs a
  cooperative-staged K loop reading the same bytes at nKeys=2048, 12 threadgroups. If not
  materially faster, the DRAM-latency reading stands and this closes; if faster, port into
  `attention` with the reduction code untouched and gate byte-exact against the snapshot golden
  at context > 128. Tile budget: 16 KB `sc` + 8 KB staging + 0.5 KB `red` < 32 KB.
- **Confidence:** plausible — the probe decides. **Prior:** metal-verdict §2b/§4 split-KV, grouped
  dedup, staged dedup — not re-proposed; this is the fourth corner.
- **CLOSED 2026-09-13, NEGATIVE — materially SLOWER, not faster.** Built the probe next to
  `attn_kvwidth_probe_test.go` (`attn_kread_staged_probe_test.go`): a coalesced device->threadgroup
  copy of a 64-key tile (16 KB — a different split of the same <32 KB budget than the fix text's
  16+8+0.5, still well inside it) per round, each thread then computing its own keys' dot products
  from the staged copy instead of device memory; reduction/PV code untouched. All-28-layer,
  nKeys=2048, min-of-20 GPU-busy: shipped 17.024 ms vs staged 39.629 ms — **0.43x, i.e. staged is
  2.3x SLOWER**, not faster. The probe also flagged a correctness mismatch (maxAbs 1.7e38 between
  the two kernels' outputs) that was NOT root-caused — hand-traced the staging/compute index
  algebra twice (thread tid's iterations over a tile write dimension tid of every key in it; the
  compute read at tid*hd recovers key tileBase+tid from exactly those writes) without finding the
  discrepancy — but the timing result alone settles the port-or-not question regardless of whether
  that bug is real or a probe-harness artifact: a design already 2.3x slower than shipped has
  nothing to gain from being fixed. Plausible cause, not confirmed: only 64 of 128 threads compute
  per tile round (idle during compute, busy only during staging) against the shipped kernel's full
  128-thread occupancy throughout, plus 64 barrier-pairs (32 tile rounds x 2) against the shipped
  loop's ~4 for the whole K-read+softmax phase — either alone could plausibly account for a 2x+
  regression without DRAM bandwidth being the bottleneck at all, consistent with the L1-issue-wall
  reading M-09's own mechanism section already favored over pure DRAM latency. Not re-proposed.

#### M-10 · The dense down-projection (22% of per-token weight bytes) is the only decode GEMV still on the unstaged byte-gather kernel the tree's own Stage-A comment calls LSU-issue-dominated
- **Where:** `metal/kernels.go:220-233` (`W4A8_BODY`: 8 scalar byte loads of the activation + one
  half scale per 32-bit word), `:272-274` ("int8 activation staged once into threadgroup short —
  replaces the per-row device byte-gather (17920× re-reads) that dominates LSU issue"),
  `metal/model.go:1903` (`e.Dispatch(r.pGemvResid, r.H*32, 32, L.dW, …)` — one simdgroup per row,
  no staging), `:1005-1009` (the M-06 comment records the choice as accounting, not a
  measurement); `docs/task-metal-batched-verify-kernel.md:143-144` (isolated: down-proj 68 GB/s vs
  gate/up 96 GB/s, same weight format).
- **Mechanism and bound (counted + record):** per token the down-proj re-gathers 28 × 1536 × 8960 =
  385 MB of activation bytes from L1/L2 for 217 MB of weights and re-loads each group scale 4×.
  Closing the isolated gap is ≈0.9 ms/token ≈ 7% of a 13.6 ms token. **Caveat the record insists
  on:** those isolated numbers are cache-warm (7–15 MB working sets fit the system cache), and
  Stage B's isolated 164→118 µs "bought zero end-to-end". The swap is not byte-identical (the SA
  body accumulates one f32 FMA per 32 weights instead of per 8) — snapshot golden and the
  batched-verify twin move together.
- **Fix:** dispatch `pSAResid` with `DispatchTG(…, 2*r.I, …)` at this site, add `I` to
  `maxThreadgroupStageBytes` (17.9 KB at I=8960), A/B on the depth bench before keeping.
- **Confidence:** plausible (end-to-end unmeasured; Stage-B precedent may apply). **Prior:**
  metal-verdict §4 "do not let an agent optimize the GEMV again" — this is a dispatch-site swap
  onto the kernel that did ship.
- **CLOSED 2026-09-13, NEGATIVE — Stage-B precedent held.** Built exactly as scoped above
  (correctness verified: full `go test ./metal/...` and `-tags goinfer_testhooks` both green,
  including every resident-parity test; `TestMaxThreadgroupStageBytes` extended for the new
  `denseInter` term). `GOINFER_METAL_DEPTH_BENCH=1` on qwen2.5-coder-1.5b, M1 Pro, three runs
  (`-count=1` each, not cached): baseline 72.3/67.9/51.4/40.4 tok/s at depth 128/512/2048/4000;
  with the fix, 70.6/66.8/49.6/38.4 and 71.8/64.0/49.8/39.1 — **slower at every depth in both
  post-fix runs (8/8), by roughly 2–6%**, run-to-run noise between the two post-fix runs
  themselves notwithstanding. Reverted (`metal/model.go`'s down-proj dispatch, `metal/kernels.go`
  untouched since no kernel code needed to change for this probe — only the dispatch site and
  `maxThreadgroupStageBytes`). metal-verdict §4's caution was right twice now: the isolated
  68 GB/s vs 96 GB/s gap this finding started from does not survive contact with the actual
  decode token, same as Stage B's own 164→118 µs. Not re-proposed.

### C. The paged path (26B / 35B / gpt-oss-20b on the Mac)

#### M-11 · Paged decode pays a ~14 ms command-buffer boundary 61–81× per token; the one design that removes it was rejected on a measurement taken where the boundary costs 0.2 ms
- **Where:** `metal/gemma4_moe.go:470-538` (`begin()`/`end()` per phase; `end` = commit +
  `waitUntilCompleted`; two per MoE layer), `metal/moe.go:747-783` (same, generic);
  `metal/residency_probe_test.go:11-12` ("~15 ms/boundary of GPU-idle-in-wait, 72× Step-0's 0.213
  ms"); `metal/pagecost_sharedevent_test.go:47-64` (verdict "recovers ~0%" — measured on
  qwen2.5-coder-1.5b, dense); `metal/model.go:1138-1144` (residency-set comment: p1 still carries
  the pinned set, +2.07 ms/CB); `metal/moe.go:135` (expert kernels already index
  `idx[slot]*rowsPerExpert`).
- **Mechanism and bound (record + counted):** the token is 2L+1 synchronous command buffers (61 on
  the 26B, 81 on the 35B). GPU-busy per token is small (≈2.1 GB weights ≈ 10 ms + ≈1,500 dispatches
  ≈ 6 ms on the 26B), yet the repo's own decomposition puts "compute+coord" at ~860 ms/token after
  the residency set — ≥13 ms of non-GPU time per command buffer, attributed to per-submit residency
  re-validation of a referenced set that changes every submit. On the 35B the same bucket is the
  193 ms/token "residual" the sweep calls the 5.2 tok/s ceiling. The single-command-buffer +
  `MTLSharedEvent` design keeps the referenced set identical every token (the case the probe found
  cached at ~0.4 ms) and was rejected on the dense 1.5B, where the 14 ms mechanism does not exist
  (G-05). What blocks it: phase-2 binds *which slot buffer* at encode time — but the kernels already
  take a slot index, so a per-layer pool allocated as one contiguous buffer + a CPU-written
  `slotIdx[k]` makes the whole token's dispatch sequence static. Bound: ≤1.37× (26B), ≤1.6× (35B) —
  upper bounds; the real number is the shared-event boundary's cost on *this* shape.
- **Fix:** contiguous per-layer slot pool (`newExpertPool`), `slotIdx` buffer instead of `idxZeros`,
  re-run `forwardLogitsSharedEvent`'s regime on the paged 26B/35B with the pread stage at the ack
  point; ship if the boundary measures ≪ 14 ms. Do M-13 at the same time.
- **Confidence:** plausible (14 ms/CB is the repo's record; the event regime on the paged shape is
  unmeasured; the pool change is traced but unbuilt). **Prior:** `pagecost_sharedevent_test.go`
  verdict — disagree (wrong shape); audit-2026-09-10 P-22; queue P21.

#### M-12 · Expert staging is queue-depth 1: k misses × 3 preads, sequential on one thread, GPU idle
- **Where:** `metal/moe.go:764-767` (serial `ensureResident` loop), `:535-553` (three sequential
  `preadRangeIntoU32Buf` per expert + `int4DirectBytes` scale narrowing on the host),
  `metal/expertpool.go:164-200`; `docs/task-metal-expert-streaming-at-scale.md:206-212`.
- **Mechanism and bound (record-derived):** per-miss cost from the sweep = staging share ×
  s/token ÷ misses/token = 1.6 ms (N=8), 3.0 ms (N=32), 3.6 ms (N=64) for ~1.57 MB — 440–980 MB/s
  effective against the same file's measured 3,687 MB/s sequential pread. Per-miss cost *rising*
  with N is the signature of latency-bound cold reads with one outstanding request. Staging is 58%
  of the 35B token (265 ms). Concurrent preads (k goroutines; the 3 spans of one expert
  concurrently — `pread` on a shared fd into disjoint buffers is safe) bound the staging wall by the
  slowest read: 265 → ~90–130 ms, 456 → ~280–320 ms/token (≤1.6×). This is *not* the declined
  `GOINFER_MOE_WILLNEED` (darwin `MADV_WILLNEED` "pays the read synchronously").
- **Fix:** resolve misses first, run the miss set's `stagePread` closures under a `WaitGroup`
  (counters under a mutex); inside `stagePread` issue the three preads concurrently. Cache the f16
  scales per expert at build (N-20).
- **Confidence:** plausible (derived from recorded aggregates; unmeasured). **Prior:** new; the
  sweep names the bucket ("a different lever, and not obviously a cheap one") but not the
  mechanism.

#### M-13 · The Metal expert pager engages only with an explicit `--moe-cache-slots N`; the default declines the 26B/35B/gpt-oss-20b to the CPU-staged path, and the peer-matrix row that "parked" them on the Mac measured that fallback
- **Where:** `metal/backend.go:154-159` (`metalMoESlotsRequest`: flag or env only; 0 ⇒ unpaged),
  `metal/moe.go:425-434`, `metal/backend.go:224-255` (guard prices the *unpaged* set when slots are
  unset, declines to CPU; the message names `GOINFER_NO_RESIDENT_MEM_GUARD` but not
  `--moe-cache-slots`); `internal/serveapp/main.go:483` (`--moe-cache-experts` … "CUDA only"),
  `:488` ("Metal: every expert resident, unpaged"); `docs/benchmarks.md:1686-1696` ("falls back
  automatically to a CPU-staged … path … killed after 2h10min with zero completions");
  `docs/task-metal-expert-streaming-at-scale.md:258-261` (recommendation: default N=64).
- **Mechanism and bound (confirmed):** `MoECacheExperts()` has no reader in `metal/`; a zero slot
  request means "unpaged", not CUDA's "ask for all, auto-cap". The recorded Metal-paged numbers
  (2.19 tok/s at N=64 on the 35B, 0.67 on the 26B) exist only through test harnesses that set the
  env var themselves; the N=64 recommendation is implemented nowhere. gpt-oss-20b (13.8 GB) is the
  same story: 16 slots/layer ≈ 4.8 GB would fit the 11.2 GB budget through the gpt-oss branch, but
  the row is "declined". A user with the flag the docs advertise gets the CPU pager.
- **Fix:** when `MoECacheExperts()` is set and no slot count is given, derive N =
  clamp((0.7·RAM − dense − KV) / (L × perExpertBytes), k, 64) from the byte accessors
  `residentNeedBytes` already calls; name `--moe-cache-slots` in the decline line. Then the
  M35/M26/G20 Mac cells measure the pager.
- **Confidence:** confirmed. **Prior:** audit-2026-09-10 M-02 (guard prices the paged set only when
  slots are given); task-gpu-paths Phase 2 delivered the flag, not the default.

#### M-14 · The residency set rides on every command buffer because aikit's binding has `Queue.AddResidencySet` but not `MTLCommandBuffer.useResidencySet:` — a recorded +62 ms/token waiting on one selector (aikit item)
- **Where:** aikit `gpu/residencyset.go:107-109` (only the queue-level attach; `:22-29` lists five
  selectors, none per-command-buffer); `metal/model.go:1138-1144` ("+2.07 ms/CB → +62 ms/tok …
  FIX (fold into the next aikit release …): a PHASE-SCOPED residency set attached only to phase-2's
  command buffers (per-encoder useResidencySet)").
- **Mechanism and bound (record):** phase 1 never touches the slot pool but carries the ~3 GB
  pinned set in its referenced list on every commit: ~4% of the paged 26B token, growing with N.
  aikit v1.40/v1.41 did not carry the selector.
- **Fix:** `func (e *Encoder) UseResidencySet(rs ResidencySet)` — one `objc.RegisterName
  ("useResidencySet:")` + one send; `gemma4_moe.go`/`moe.go` call it on the phase-2 encoder only and
  drop the queue attach. Ships with M-11.
- **Confidence:** confirmed. **Prior:** new at the seam; the cost is goinfer's own record.

### D. Cross-repo and unassessed

#### M-15 · On a Metal box every image turn runs the vision tower on the CPU, and aikit's Metal tower cannot be wired as a win until three shapes change (aikit M-14/M-09/M-10 Metal halves)
- **Where:** `internal/serveapp/main.go:988-992` (`EnableResident` only for `webgpu`; nothing imports
  `visionmetal`/`qwenmetal`); aikit `gpu/metal_vit.go:168-221` (attention: one threadgroup per
  (head, query), re-streams K and V per query — no query tile; score lanes 4,608 B apart; PV keeps
  hd=72 of 256 lanes busy), `:397-420` (`gemm_w8a8_tiled`: one output per thread, byte-granular
  staging, scalar int8 — the shape CUDA's M-14 retired), `gpu/qwenmetal/encoder.go:188-213,311-334`
  (per-op `Run1D`/`Run2D`, each a commit + `waitUntilCompleted` + pool drain: 544–704 synchronous
  submits per image at 32 blocks); aikit `CHANGELOG.md:292-293` (batched SigLIP tower 0.46×/0.33× of
  CPU by its own crossover), `:316-318` (M-10 "NOT DONE: the Metal half"); `docs/multimodal.md:172`
  ("Metal — still not started"), `docs/benchmarks.md:554-557` (CPU SigLIP 31.3 s/image).
- **Mechanism and bound (counted):** at so400m (np=4096, nH=16, hd=72, 27 layers) the attention
  re-reads K+V 4,096× → **≈4.2 TB per image** of L2/DRAM traffic against a 37.7 MB/layer minimum;
  the int8 GEMM amplifies fc1's operands ~260× (2.5 GB of tile traffic per GEMM vs 9.7 MB); the
  qwenmetal submit floor alone (~250 µs × 544–704) is 136–176 ms/image, ≥ the CPU Qwen tower's
  recorded 157 ms. goinfer not wiring the tower is therefore currently correct — and neither doc
  says why.
- **Fix (aikit):** query-tiled attention with online softmax (re-baseline the ViT parity gate);
  int8 projections through a half-input `simdgroup_matrix` GEMM or a register-blocked int8 kernel
  (the CUDA `gemm_w8a8_reg` shape); port visionmetal's `runBatch` + scalar ring to qwenmetal. Then
  (goinfer) `EnableResident` for `cfg.backend == "metal"` and a crossover row in `multimodal.md`.
- **Confidence:** confirmed for shapes and submit counts; absolute image times on the M1 unrecorded.
- **Prior:** aikit audit M-14/M-09/M-10 (CUDA halves done); goinfer task-gpu-paths G2 / multimodal P6.

#### M-16 · Every buffer is hazard-tracked and every encoder serial; the binding exposes neither the untracked option bit nor `computeCommandEncoderWithDispatchType:`, and the record calls the resulting per-dispatch floor "unassessed"
- **Where:** aikit `gpu/metal.go:424,432,443,491,500` (every `newBuffer*` passes `options = 0` =
  Shared + DefaultCache + HazardTrackingModeDefault), `:682,701,906` (`selComputeEncoder`, serial);
  `docs/completed/metal-verdict.md:131-133` ("~14 cores + serial hazard-tracked encoding ⇒ a ~3.8
  µs/dispatch GPU-side floor"), `:248-249` (watch item, "unassessed").
- **Mechanism and bound (record):** ~3.8 µs × ~310 dispatches ≈ 1.2 ms of an ~18.5 ms token (~6%),
  the term that sets the "~70 tok/s shallow dispatch floor". Untracked buffers under a serial
  encoder are safe by construction (the encoder already orders every dispatch; one queue executes
  in order), so the change is the allocation flag (`1 << 8`). What fraction of the 3.8 µs is
  tracking vs raw drain is exactly what is unmeasured.
- **Fix:** an `options` variant (`NewBufferLenUntracked` or a package flag) and an A/B on the depth
  bench; only if it moves, follow with the concurrent dispatch type + explicit barriers where the
  graph has independent siblings (the decode graph is nearly a chain, so expect less there).
- **Confidence:** plausible. **Prior:** metal-verdict §3 Wall 1 / §5 watch item — not a recorded
  negative.

---

## 2. Correctness

#### C-01 · `attention_prefill_fused` reads K/V rows past `nKeysMax` on a ragged last tile — past the cache end when `ctxCap % 8 ≠ 0` (prior audit N-46, open)
- **Where:** `metal/prefill.go:337-346` (`for (uint j0=j0start; j0<nKeysMax; j0+=8u)` with
  `simdgroup_load(kT, kBase + j0*kvDim …)` — full 8-row tiles, mask applied after the load at
  `:352-363`), `metal/model.go:914-919` (`kc/vc` sized `ctxCap*kvDim*2`), `:57-74` (`ctxCap` = the
  user's request, unrounded), `metal/attention_prefill_fused_test.go:39,63-65` (M=37, exactly
  `cacheLen` rows — passes because MTLBuffers are page-rounded).
- **Failure:** `--resident-context 1000` with a prompt that fills it → up to 7·kvDim·2 bytes read
  past `kc[l]`/`vc[l]`; a non-finite value in an unspecified tail gives 0×Inf = NaN in that row's O.
  OOB reads are tolerated on this hardware, so no fault — a wrong row instead.
- **Fix:** round `ctxCap` up to a multiple of 8 rows when sizing `kc/vc` (one line), or clamp the
  tile and zero-fill in-kernel. Make the unit test allocate exactly `cacheLen` rows on a
  non-page-rounded size so it can see the read.
- **Confidence:** confirmed (read pattern); NaN outcome plausible. **Prior:** N-46.

#### C-02 · `HiddenLast` (serve `/v1/embeddings`), `Forward(id,pos)` and `ForwardArgmax` on a paged MoE bind the zero-value stacked-expert buffers — C-08's defect on three more entry points
- **Where:** `metal/backend.go:481-513` (`HiddenLast` → `forwardHiddenNoHead` per position),
  `metal/model.go:1262-1271` (→ `encodeTrunkInto` → `encodeLayer`, `:1808-1813` — no paged branch;
  paging lives only in `Forward`'s dispatch to `forwardLogitsPaged`, `:1241`), `metal/moe.go:289-291`
  ("expGuW/expGuS/expDW/expDS stay zero-value when paged"), `:651-659` (bound unconditionally);
  `gpu/metal.go:745-748` (OOB/unmapped reads are silently tolerated).
- **Failure:** on a `--moe-cache-slots` MoE (generic or Gemma 4), an embeddings request returns a
  finite garbage vector with no error; the snapshot golden's `Forward` path likewise.
- **Fix:** decline in `HiddenLast`/`Forward`/`ForwardArgmax` when `r.g4moe.paged || r.moe.paged`,
  or make `encodeLayer` refuse a layer whose `pool != nil`. Note this also constrains M-01's fix.
- **Confidence:** confirmed (structurally; not reproduced). **Prior:** audit-2026-09-10 C-08 (fix
  covered `PrefillLast` only).

#### C-03 · Adapter `In`/`Out` are never checked against the base projection on Metal (prior audit M-05, open) — a same-family different-size adapter writes past the Q slot into K/V
- **Where:** `metal/lora.go:147-149` (rank-only check), `metal/kernels.go:859-861,868-872` (`Ar = A +
  r*K`, `out[row]` for `row < Out` — both from the adapter's own uniforms).
- **Fix:** in `conv`, refuse when `p.In`/`p.Out` differ from the projection's `K`/`N` (seven
  comparisons; `SetAdapter` has `r.H`, `r.I`, `L.geom`). The decoder-side chokepoint the prior
  audit proposed is the better home.
- **Confidence:** confirmed. **Prior:** M-05.

#### C-04 · `SetAdapter`'s error path leaks the partially built bind until `Close`
- **Where:** `metal/lora.go:157-181` (`if out[i].q, err = conv(l.Q); err != nil { return err }` —
  buffers for layers `0..i` stay on the ledger; `releaseLoRALayers` not called).
- **Fix:** `defer` a release of `out` on error. **Confidence:** confirmed. **Prior:** new (minor
  impact, ratchets across failed binds).

#### C-05 · aikit `Queue.Run1DBatchTG` / `Run1DTG` own an NSAutoreleasePool without the G22 OS-thread pin every sibling helper has
- **Where:** aikit `gpu/metal.go:953-982` (`pool := … Send(selInit); defer pool.Send(selDrain)` with
  no `runtime.LockOSThread`) vs `:585-589,621-625,925-929` (siblings pin, citing "intermittent
  SIGSEGV (fault 0x10) inside objc_msgSend"). Only production caller (`qwenmetal.ForwardViT`) pins
  for the whole forward; goinfer's batch-k harnesses call it unpinned.
- **Fix:** the same two lines as `Run1DBatch`. **Confidence:** confirmed (shape). **Prior:** aikit
  G22 (missed one helper).

#### C-06 · `visionmetal` has no threadgroup-memory budget guard (qwenmetal's C-02 fix was not mirrored), and the status latch it relies on cannot fire on Apple silicon
- **Where:** aikit `gpu/visionmetal/encoder.go:225-228` (`DispatchTG(…, np*4, …)` — over the 32 KiB
  limit above np=7,680) vs `gpu/qwenmetal/encoder.go:247-250,368` (`attnThreadgroupBytes` guard);
  `gpu/metal.go:743-748` ("silently tolerates … over-budget threadgroup memory … status Completed").
- **Failure:** a SigLIP tower with >7,680 patches returns a plausible wrong hidden state. Not a
  shipped shape today (so400m/896 = 4,096).
- **Fix:** the qwenmetal check with `np` in `newEncoder`. **Confidence:** plausible. **Prior:** aikit
  C-02 (qwenmetal only).

---

## 3. Gates that cannot fail

#### G-01 · `TestMoE_declinesPrefill` has been red on every Metal box since the 512 floor landed; three commits carry it as "pre-existing, unrelated"
- **Where:** `metal/moe_model_test.go:302,321-327` (8 embeddings; sets `GOINFER_METAL_BATCHED_PREFILL=1`
  only), `metal/backend.go:428-431` (floor check precedes `prefillOK`); log lines 859, 909, 1213.
- **Mechanism:** 8 < 512 ⇒ decline ⇒ `Fatalf` before any MoE code runs; the admit-side MoE-vs-dense
  check C-08's fix relies on has not run green since the floor. A `go test ./metal/` that is always
  red trains everyone to ignore it.
- **Fix:** `t.Setenv("GOINFER_METAL_FAST_PREFILL_FLOOR", "0")` as `prefill_ttft_test.go:45` does.
- **Confidence:** confirmed (three reviewers).

#### G-02 · The §3.2 pooled gate still drops missing cells silently and turns a fit-guard decline into a SKIP that "SHIPS" (prior audit G-08, open)
- **Where:** `metal/prefill_gate_ref_test.go:186-202` (`if cs != nil { … }` — a missing reference
  file is dropped; only zero cells fails; the header prints the full K set), `:114-117` (D7 that
  fails to build → `Skipf`). Both records say D7 was decided-around by fit-guard: the floor and both
  default-ON flips that govern 7B-class Mac users rest on the 1.5B alone (and M-07 is why D7 does
  not fit).
- **Fix:** `len(decisionCells) != len(decisionKs) ⇒ Fatalf`; a decision model that does not build is a
  FAIL when deciding. **Confidence:** confirmed. **Prior:** G-08.

#### G-03 · `TestPrefillParityGemma` gates the "Gemma set" on a fixture shaped so `prefillOK` is true — the shape every real Gemma 3 lacks
- **Where:** `metal/prefill_gemma_test.go:44-47`; fixture `[sliding, sliding]` shares one RoPE base
  → `ropeUniform()` true; real Gemma 3 derives `FeatPerLayerRoPE` and is declined before the kernel
  runs (M-06). **Fix:** a global layer in the fixture. **Confidence:** confirmed. **Prior:** M-23.

#### G-04 · The absolute snapshot golden is re-baked by the code it checks after each accepted kernel round; the autoresearch ledger is outside the tree
- **Where:** `metal/snapshot_golden_test.go:20,177-184` ("the ABSOLUTE STORED REFERENCE";
  `GOINFER_UPDATE_GOLDENS` writes whatever current code produces), `metal/kernels.go:26-29` ("only
  deep-mantissa sha bits move — see the round's own commit"); `docs/task-autoresearch-loop.md:3`
  ("not started") and §3 ("Do NOT point it at Metal") vs `kernels.go:102,147,729-754` (Metal rounds
  ran on the norm-class kernels); `scripts/autoresearch_rmsnorm_results.tsv` not in
  `docs/measurements/`.
- **Mechanism:** the one gate that sees reduction-order/fast-math drift was refreshed to the new
  bits on an argmax-unchanged argument; which rounds re-baked and which reverted cannot be audited
  from the repo. **Fix:** a re-bake in the same commit as a kernel change needs its own gate line
  (argmax-at-every-checkpoint + cosine vs the previous golden, in `docs/measurements/`), or the
  golden is declared an OS-drift detector only; move the tsv into `docs/measurements/`; fix the
  autoresearch doc's status. **Confidence:** plausible (the tsv is not here).

#### G-05 · The shared-event verdict was measured on a shape without the cost it targets
- **Where:** `metal/pagecost_sharedevent_test.go:47-64` (qwen2.5-1.5b dense int8int8; "recovers ~0%"),
  `metal/pagecost_measure_test.go:47-52` ("There is NO such checkpoint on this Mac … this measures
  the SUBMISSION-STRUCTURE cost on a DENSE model"), `metal/residency_probe_test.go:11-12` (paged
  26B: ~15 ms/boundary). Carried into production comments as settled (`gemma4_moe.go:434-436`,
  `model.go:1360`). **Fix:** M-11's re-run. **Confidence:** confirmed.

#### G-06 · The device-ledger "did Close/ReleaseBuf free it" assertions pass by construction
- **Where:** `metal/close_leak_test.go:160-169,224-248` vs aikit `gpu/metal.go:382-390`
  (`ids := d.allocs; d.allocs = nil; … for … Send(selRelease)`): `LedgerLen()` is emptied
  regardless of whether `release` is sent; the test's own comment (`:229-231`) records RSS "DID NOT
  ratchet" under a neutered `ReleaseAll` because macOS compressed the pages. **Fix:** assert on
  `MTLDevice.currentAllocatedSize` (exact, compression-immune). **Confidence:** confirmed.

#### G-07 · `TestPrefillGate` (superseded §3 form) would `Fatalf` on its first cell; three files cite `TestMetalPrefillDivergenceRate`, which does not exist
- **Where:** `metal/prefill_gate_test.go:65,75,254-257` (K=256 without the floor override);
  `spec_prefill_regression_test.go:46`, `spec_verify_curve_test.go:22`,
  `docs/measurements/prefill-gate-l1-2026-09-05.md:66` cite a test `grep` cannot find — the
  most-quoted Metal fidelity number ("54% stream divergence") has no test behind it. **Fix:** delete
  `TestPrefillGate` or set the override; re-add the divergence test or stop citing it.

#### G-08 · The §3.2 gate never exercises `startPos > 0`, which every resident-prefix-reuse turn uses
- **Where:** `metal/prefill_gate_ref_test.go:429` (`PrefillLast(ctx, embs, 0)`) vs
  `decoder/model.go:1094` (`from`); the fused kernel's `startPos`/`uMReal` masking is covered only by
  a synthetic hd=64 case. The agent-turn shape the peer matrix calls the headline workload is not
  a fidelity cell. **Fix:** one decision cell with `from = K/2` on S. **Confidence:** plausible
  (coverage gap, no defect shown).
- **CLOSED 2026-09-13, coverage added — no defect found.** Implemented as a focused correctness
  test (`metal/prefill_startpos_test.go`) on the tiny synthetic fixture instead of a new decision
  cell in the pooled §3.2 gate: `decisionKs`/`confirmKs` and the critA/B/C pooling formulas in
  `prefill_gate_ref_test.go` are a carefully pre-registered statistical methodology (this audit's
  own M-03/M-04 SHIPS verdicts depend on it), and this finding's own confidence — "plausible,
  coverage gap, no defect shown" — did not justify the risk of modifying that machinery to add a
  second dimension (K, from) it was not designed around. The new test builds a shared KV prefix
  [0,from) on two residents via Forward (bit-identical by construction), then diverges: one
  continues via Forward through [from,K), the other via `PrefillLast(embs[from:], from)` — the
  exact code path (startPos/uMReal masking) this finding flagged as uncovered — and compares
  final logits. K=48, from=24: argmax match, cosine 0.999941. No defect found; the gap is closed,
  not a bug fixed.

#### G-09 · P-15's MoE-prefill measurement is written, never run, and refuses the paged shape that actually runs on the Mac
- **Where:** `metal/moe_prefill_measure_test.go:14-27,52-56` (`GOINFER_MOE_PREFILL_CKPT`; declines a
  paged resident). The "default ON above 512 for MoE" claim (`model.go:106-108`) rests on dense
  cells. **Fix:** let it run on the paged 35B, sequential vs expert-major once M-05 exists.

#### G-10 · The C-09 status latch (`mustCmdBufOK` / `Encoder.Err()`) is, by its own comment, inert on Apple silicon — host-side pre-checks are the real gate
- **Where:** aikit `gpu/metal.go:743-748,770-775`, `metal/cmdbuf_status_test.go:19-22` (both repos'
  tests inject the error). Documentation, not a lever: every "Err() will catch it" reliance needs a
  pre-check (C-06 is the bare one).

---

## 4. Minor

**Docs stale (fix with the next benchmarks pass):**
- N-01 `docs/benchmarks.md:40,502-550` — §A Metal prefill row (goinfer `82b7b8a`) predates the fused
  attention default; "8.81× at K=3900" is ≈4.3× by the tree's own L2 record; "the remaining gap is
  the attention kernel" describes the pre-fused kernel and is now false (M-03).
- N-02 `docs/benchmarks.md:985-991` — §B3 "declines batched prefill by default … 54% stream
  divergence": default ON since 09-09; `:972` "`//go:build darwin && metal`": every file is
  `//go:build darwin`.
- N-03 `docs/benchmarks.md:998-1022` — the §B3 depth curve is `TestZZ_metalDepthBench`, a tight
  `ForwardArgmax` loop; `metalResident` does not implement `ResidentGreedy`, so production greedy
  runs `ForwardEmbPipe` → full head + host argmax. Labelling defect (fused argmax is a recorded
  speed-neutral on UMA), but the curve measures a path serve never takes.
- N-04 `docs/gpu-residency-coverage.md:19-25` — "in int8 W8A8": Metal has no int8 GEMV and
  re-quantises to W4A8 (`model.go:435-461`). Qwen2.5-VL/Qwen3-VL "✅ resident" Metal cells are
  text-only (no `ForwardMRoPE` in `metal/`); Gemma 4 E2B/E4B CPU-only under a "✅" row — no footnote.
- N-05 `docs/audit-2026-09-10.md:153` lists C-08 open; `:1216-1224` records it fixed (d9139bc).
- N-06 `docs/task-autoresearch-loop.md:3` "not started" / §3 "do NOT point it at Metal" — it ran on
  Metal (norm-class kernels); ledger tsv outside the tree (G-04).
- N-07 `docs/task-metal-batched-verify-kernel.md:3` header says "CONFIRMED"; §Go/no-go is NO-GO
  (code honours the NO-GO).
- N-08 `metal/backend.go:337-340` ("CURRENTLY OPT-IN … pending Phase B" then "Default ON"), `:426`
  ("K=256 cell failed"), `:475-480` (`HiddenLast`: "declined by default") — N-44 of the prior audit,
  still present; `:355-356` accepts only `"1"` (N-45). `metal/spec_prefill_regression_test.go:43-51`
  same stale precondition.
- N-09 `metal/model.go:1323-1324` "greedy decode skips [softcap]" — false in production
  (`finalizeLogits` runs every executor token; Gemma 4 greedy pays a 262k `tanh` per token, ~1%).
- N-10 `metal/model.go:1921` "11 dispatches vs 19" is stale (block is 7); `metal-verdict.md:75`
  "337 dispatches/token" is 310 at this tree. `:1538-1541` "fastest greedy path" stale.
- N-11 `metal/cmd/serve/main.go:7-8` "Dense residency only … int8": MoE is resident; int8 is the
  re-quantised case. `decoder/features.go:317` cites `moe.go:206-211`; it is `:375`.
- N-12 `docs/measurements/prefill-gate-l1-ref-b-2026-09-09.md:14` names `qwen2.5-1.5b-instruct`;
  test default and L2 record say `qwen2.5-coder-1.5b-instruct` — methodology wants the exact file.
- N-13 `metal/snapshot_golden_test.go:127` names `attention_f32` as covered; N-28 says it is dead.
- N-14 `metal/lora_resident_parity_test.go:104,191` — cosine floor 0.95 vs measured 0.99996 (N-52).

**Cold-path waste and small levers:**
- N-15 `metal/prefill.go:529-542` — 26 per-request scratch buffers built from `make`d, zero-filled Go
  slices then copied (`guF` alone 140 MB at M=3900; ≈262 MB memset + ≈262 MB memcpy per long
  prompt); `gpu.NewBufferLenOf` exists and goinfer never calls it; only `xF` needs zeroed pad rows.
  A high-water-mark cache across calls removes the allocation entirely. Tens of ms vs a 36 s TTFT.
- N-16 `metal/prefill.go:438-468` — the 13-kernel prefill library + 12 pipelines compile lazily inside
  the first `PrefillLast` (the first request's TTFT); `buildResident` could do it. Also N-47: a failed
  `ensurePrefill` re-panics per call (no latch).
- N-17 `metal/prefill.go:24-61,426-427` — `gemm_w4f16` "Removed" but still compiled; `allKernels`
  compiles seven kernels no pipeline uses.
- N-18 `metal/prefill.go:385` — `pTile` reloaded from `pScr` per `cc` (16×/tile); subsumed by M-04.
- N-19 `metal/prefill.go:204-216` — `rope_f16` computes cos/sin per (row, pair) with no table; a
  second reason to fuse RoPE-K into `kv_store_f16`.
- N-20 `metal/model.go:378-399` + `moe.go:546-552` — per stage the f16 scales are re-derived from an
  f32 heap copy that is 2× the bytes the GPU consumes (≈2.85 GB on the 26B, ≈4 GB on the 35B, on
  the box whose N=128 cliff was memory pressure); cache f16 per expert at build.
- N-21 `metal/expertpool.go:150-155` — each slot built via `NewBufferUint32s(d, make([]uint32, n))`:
  ≈4.5 GB of transient Go allocation at N=64 on the 35B to zero-initialise; `NewBufferBytes(n)`.
- N-22 `metal/moe.go:768-771`, `gemma4_moe.go:520-524` — phase 2 of layer l and phase 1 of l+1 have no
  host dependency and could share one command buffer (2L+1 → L+1); superseded by M-11.
- N-23 `metal/moe.go:774-777` — a hybrid's dense layers each get their own `Begin/End` in
  `forwardLogitsMoEPaged`.
- N-24 `metal/moe.go:29-37,622` — f32 router weight: 84 MB/token on the 35B (deliberate, ≤0.4 ms).
  `moe_route` on one GPU thread (deliberate, value-independent dispatch; ~10% of a fitting ~5 ms
  MoE token).
- N-25 `metal/backend.go:481-514` — `HiddenLast` is one synchronous command buffer per position
  (≈K × 13–18 ms; ~7–9 s for 512 tokens) where the batched trunk would take ~1.8 s; the stated
  rationale ("declined by default") is stale. Fix is `PrefillLast` minus its last two dispatches.
- N-26 `metal/backend.go:518-533` — `ForwardN` is a per-token loop allocating 608 KB per row; cold
  (Theta ≈ 1.02 declines speculation, P21) but the interface doc's "K tokens in ONE command
  buffer" is not what Metal does.
- N-27 `metal/model.go:1304` — pipe path memcpys 608 KB into `logitsHost` before the ack; the
  zero-copy `r.logits.Floats()` view could be returned (the contract already says "consume before
  the next call"). ≈30–60 µs/token.
- N-28 `metal/model.go:1372-1373,1412-1424` — at temperature > 0.2 (serve default 1.0) nothing
  overlaps the CPU sampler with the GPU: the full-vocab softmax/filter sits in the GPU-idle gap
  (order 0.5–1.5 ms of ~13.6 ms; the served 54 vs decode-only 73.6 tok/s is where it shows). The
  0.2 cap is a measured decision; the only structural way past it is device-side sampling.
- N-29 `metal/model.go:348-368,475-478` — load path rebuilds every word byte-by-byte from nibbles
  that are already the word bytes; `int4Concat` grows by `append` (two reallocs per fused tensor);
  scales through the serial loop. ≈0.5 s at 1.5B.
- N-30 `metal/kernels.go:737-777` — `swiglu_quant` evaluates `glu_act_pinned` twice per element;
  <1%. `:276` "K<=1536" stale. `:563-567` rope2_kv recorded null (0.6%) — not re-proposed.
- N-31 `gpu/metal.go:713-717` — `WaitDone` reads four GPU timestamps per production token for
  `LastGPUTimes` (tests only); µs.
- N-32 `gpu/metal_vit.go:576-578` — "64×64 tile" stale (32×32). `:665-666` — the ViT library compiles
  fast-math OFF library-wide for one exact divide; `precise::divide` per op would free the rest.
- N-33 `metal/expertpool.go:41-48` — `copyBytesToU32Buf` duplicates `gpu.Upload` minus its bounds check.
- N-34 `gpu/metal_copy.go`, `metal_upload_batch.go` — unused by goinfer (correct on UMA); note they
  are host-side and unfenced, so a `CopyDevice` during an in-flight command buffer would race.
- N-35 `decoder/model.go:1063-1068,1104` — `warnPrefillDeclined` is process-lifetime `sync.Once`; on
  Metal the first sub-floor prompt consumes it, so a later real decline (cap, OOM) is silent (N-49).
- N-36 `metal/backend.go:199-202` — `residentKVBytes` charges KV for DeltaNet layers that allocate none.
- N-37 `metal/gemv_w4a8_coal_bench_test.go:29-30` — the GEMV micro-bench shape (512×4096, 1 MB) is
  cache-resident: a "micro-win" here is an issue/latency result, not a bytes result (the production
  down-proj reads 7–14 MB per dispatch from DRAM). `metal/profile_test.go:78` per-kernel µs are
  warm-cache for the same reason.
- N-38 `metal/prefill_ttft_test.go:81` — the first `PrefillLast` (P=256) includes the one-time compile;
  the L2 record's P=256 row carries it in both arms.
- N-39 `internal/serveapp/openai.go:1081-1086` — comment says adapter requests "drop to the staged
  path"; since G3 they reach the resident path on a `prefillFrom == 0` turn. Later-turn behaviour
  (`decoder/session.go`) not in tree.
- N-40 `docs/benchmarks.md:975` — §B3 "4-bit both sides": the tied LM head (24% of per-token bytes)
  runs int8 by a deliberate fidelity pin; up to ~117 MB/token (≈1.4 ms) of the 1.5B decode deficit
  is a chosen precision trade, not kernel quality. Labelling, not a defect.

---

## 5. Checked and found correct / at ceiling

- **Command-buffer and sync structure (decode):** one command buffer, one commit, one
  `waitUntilCompleted`, one serial encoder, 310 dispatches, ~1,030 purego transitions, one heap
  allocation per token; no per-token buffer creation; `setBuffers` batching already cut binds from
  ~2,600 to ~337 msgSends/token; ICB and unretained references are recorded nulls
  (`model.go:1379-1412`, `gpu/metal.go:700-812`, `metal-verdict.md:171-172`).
- **Encode-ahead on the production path:** `metalResident.Forward` → `ForwardEmbPipe` → `execLoop`;
  t+1 encoded between `Commit(t)` and `WaitDone(t)`; value-independent (pos/nKeys/`r.x` written at
  commit); only 1 token in 64 (pool drain), the first token after `SetAdapter`, paged models and
  `HiddenLast` are un-overlapped — all documented. `TestEncodeAhead` gates parity.
- **KV ordering across command buffers:** cb(t+1) committed only after cb(t) completes; `PrefillLast`
  runs on the caller's pinned thread with its own commit+wait while the executor holds only an
  un-committed buffer; `resBusy` serialises requests. No race.
- **C-07 fixed** (`SetAdapter` → `stopExec()` first; nil-check re-arm; mutation-tested to cosine 0.36
  with the call removed). **C-08 fixed** (`prefillOK` excludes paged generic MoE and Gemma-4 MoE;
  decline test). **G-06 Metal half, C-01 (`!hasAdapter`), P-10, P-11 (dispatch half), P-14
  (`parallelEmbedsF32ToF16`) fixed** in 3054ede, e23bbb4, b9aa35d. **qwenmetal C-02 fixed** with the
  corrected 7,680 bound.
- **W4A8 SA GEMVs (qkv, o, gate/up):** bytes per output row = K×0.5625 (format minimum); activation
  staged once per 8 rows; one `uint4` + one FMA per 32 weights; int32 group sums exact; the record's
  84–105 GB/s against a ~120–145 GB/s no-DP4A issue ceiling and the peer's own 75–83 GB/s band —
  cannot reach 200 GB/s with nibble unpack at ~10 issue slots/weight. Stage B is a recorded
  negative. At ceiling.
- **LM head `gemv_w8a8_coal`:** byte-minimal for int8, uint-vectorised (+11% recorded), 151,936
  threadgroups.
- **Fusion state:** norm+quant, swiglu+quant, act+quant, QKV bias, o-proj/down residual, Q+K RoPE
  fused; rope2+kv_store (~0.6%) and quant_vec-into-GEMV (~0.97×) measured and left unwired;
  19→12→11 dispatches/layer recorded ~0. The 113 single-threadgroup norm/quant dispatches per token
  bound at ≈0.3–0.5 ms by shape; widening costs a second dispatch each. At ceiling within the trade.
- **Attention reduction widths and orders** are the bit-identity contract; split-KV and the three
  dedup variants are recorded negatives (0.23–0.98×) — M-09 is the one corner that leaves every
  reduction untouched.
- **KV cache:** f16 store, f32 read-side accumulation, contiguous 512 B per layer per token,
  allocated once at build.
- **Tails:** M%8 via `Mpad` zero rows and masks; hd>128 or hd%8≠0 → exact kernel; N%8 refused at
  build; K%32 pack invariant; `sc[4096]` bound enforced by `ctxCap ≤ 4096`; `idScr[16]` widest
  bind is 12; alignment rides on H%8/I%8 + page-aligned buffers; `rope2` partial rotary matches the
  NeoX split.
- **Prefill batched path launch structure:** one command buffer, one wait, ~310 dispatches, ≈2 ms
  of launch per prompt; `kv_store_f16` O(M), one launch, coalesced; causal/window tightness of the
  fused loop (no wasted tiles beyond the ragged last one, which is C-01); scratch leak (C5) fixed;
  final softcap applied; floor semantics with prefix reuse correct (`promptLen = startPos + M`).
- **f16 storage vs f32 accumulation:** accumulation is f32 in every prefill kernel; what is f16 is
  storage (the residual stream rounded 56×, dequanted weights, inter-op activations, the P tile).
  The set-B gate measured the batched path *closer* to the f32 reference than the exact path's
  int8-activation quantisation. One cheap fidelity lever with a mechanism: keep `xF` in f32 (+24 MB
  at M=3900) — fidelity, not speed; unmeasured.
- **Paged path internals:** paged ≡ stacked byte-identity trick (`idxZeros`, `rWgt[uSlot[j]]`,
  `rIdx`) correct; LRU cannot evict the current token's top-k (N ≥ k enforced, touch-to-MRU); pread
  offset resolution all-or-nothing per layer, group-32 spans word-aligned, short-read loop, range
  check; the pread A/B (3.23×, faults 98.5→0/stage) is real and saturated; `WILLNEED`/`NOCACHE`
  measured and declined with the confound named; residency-set scope = slots-only is a five-arm
  bisect; expert GEMV launches are at the ~3.8 µs floor not the bandwidth roof — batching across k
  saves <1% of the paged token; DeltaNet 12 launches/layer, f32 state 2 MB/layer, no readback,
  `delta_rule` ≈3 ms/token pinned to the CPU's accumulation order — at ceiling under the
  bit-identity contract; abort discipline and `Close` correct.
- **aikit binding:** library/pipeline compile once per load (no `MTLBinaryArchive` — cgo-free forces
  runtime MSL); every buffer `StorageModeShared` (right on UMA; `NewBufferNoCopy` inapplicable to
  goinfer's fused/narrowed layouts); no status-polling loop anywhere in production; `SharedEvent`
  test-only; fast-math ON with precise opt-in measured 4–7% for no parity gain; both compile paths
  read back `languageVersion`/`mathMode`; goinfer reimplements no binding; `Encoder.Dispatch` is
  2–3 sends. `UploadBatch`/`CopyDevice` correctly unused on UMA.
- **`ResidentGreedy` absent on Metal:** recorded speed-neutral on UMA ("~30 µs zero-copy view";
  "kept as API"). Not a lever — only a labelling issue (N-03).
- **Speculative verify:** `ForwardN` is a loop; bvk kernels never compiled into production (NO-GO
  honoured); Theta ≈ 1.02 declines — not re-proposed.
- **Docs that verify line for line:** every `features.go`/`residency.go` citation in
  `gpu-residency-coverage.md` resolves; every Metal-declared feature has its kernel/wiring (qk_norm,
  window, partial rotary, MoE route/experts/shared, sandwich, GeGLU clamp, addOne, embed scale,
  per-layer invf, softcap/logit-scale, GPT-2's four, sinks + gpt-oss route, seven delta kernels, NoPE,
  qTempScale, postOnly, qkNormWhole, parallel block); hardware-matrix Metal column spot-checked
  against `ResidentEligible` — consistent; the 256-expert cap enforced at build.
- **Kernel micro-benchmarks** do not hoist work out of the timed region (caveat N-37 on shape).

---

## 6. Landscape — what this means against the peer

Counted from the tree's own records, with the same caveat prefill-gap §6.2 recorded (the last
peer extrapolation was ~25% optimistic; internal projections held within 1.5%):

- **Short prompts (K < 512), today:** 74–77 tok/s TTFT, 10.2× behind at K=256. After M-02 (floor
  256): the batched path's own 272 tok/s at P=256 → ≈3.5× behind at K=256, ≈3.3× at 512. After
  M-01 on what remains sequential (8–255 tokens, adapters, declined families): −22% per prompt
  token. These two are a day's work and need no new kernel.
- **The ladder:** the flat term is the GEMM at ≈3.6 ms/token (M-03) vs Ollama's ≈1.0. If the wider
  tile with A staging reaches 2× (1.8 ms/token — well inside what the same primitive delivers in
  Ollama's `mul_mm`), TTFT at K=3900 goes ≈17.9 s → ≈10.9 s (≈2.6× behind); with M-04 halving the
  attention term, ≈9.3 s (≈2.2×). At K=512 the same 2× on the GEMM alone takes 3.3× → ≈1.8×.
  Reaching parity needs the GEMM at ≈3.5× — the CUDA tensor-core lever (L3) got 4.52× on its
  category from a worse starting shape.
- **Decode:** at the launch/sync ceiling; the two probes (M-09, M-10) are worth ≈7% each if they
  land, and the int8 head (N-40) is a chosen ≈10% precision trade. The 1.5B decode gap to Ollama
  (0.74×) is mostly the W4A8 issue ceiling the record already priced; do not expect a decode
  campaign to close it.
- **Memory:** M-07 is ≈0.9 GB on the 1.5B and ≈4 GB on the 7B — the difference between D7 fitting
  the §3.2 decision set or not, and part of the 26B's 3.75 GB miss.
- **Paged MoE on the Mac:** exists only through test harnesses today (M-13). With M-13 + M-14 + M-11
  + M-12 the recorded 2.19 tok/s (35B) has a counted ceiling around 3–4 tok/s; still a
  "runs-at-all" cell, not a peer cell.

---

## 7. Program

Ordered by TTFT-on-the-Mac per hour of work; each lands with its own gate line and a
`benchmarks.md` touch.

1. **M-02 + M-01 + G-01** (a day; `metal/backend.go`, `metal/model.go`, `moe_model_test.go`): floor
   → 256, `noHead` executor job with the paged-aware constraint from C-02, the red test made green.
   Gate: §3.2 pooled at K=256 (already passing), byte-identical decode after a no-head prefill
   (CUDA's twin test), TTFT ladder re-run at K∈{64,128,256,512}.
2. **M-03** (the GEMM tile; days): 32×32 per-simdgroup block, all-lane dequant, A staged per
   threadgroup. Gate: §3.2 pooled + the L2 record's shapes; the number that matters is P=256 TTFT.
3. **M-04** (fused attention, register O + diagonal α + 32-key tile + GQA grouping): re-run §3.2.
4. **M-06 + G-03 + C-01** (three one-line changes plus a fixture): Gemma 3 onto the batched path,
   the fixture that can see it, the cache rounding.
5. **M-07** (task-int4-layout L4, scheduled): canonical-only for resident GPU backends; release
   host projections after upload. Gate: RSS after load on the 1.5B and the D7 fit.
6. **M-08** (LoRA grid): with a Mac adapter-decode measurement recorded for the first time.
7. **Paged:** M-14 (aikit selector, one release) → M-13 (auto-sized slots; the M35/M26/G20 rows
   re-run against the pager) → M-11 + G-05 (the shared-event re-run on the paged shape) → M-12.
8. **Probes:** both CLOSED 2026-09-13, NEGATIVE (see their own entries above). M-09 — the staged
   K-read probe measured 2.3x SLOWER than shipped, not faster; not ported. M-10 — built and A/B'd
   on the depth bench, slower at every depth; reverted.
9. **Docs and gates batch:** N-01…N-14, G-02, G-04, G-07, G-08, G-09; the §A Metal row re-measured
   after 1–3 with the same protocol and Ollama in the same session.
10. **M-15** (cross-repo, aikit first): the three Metal tower shapes; then `EnableResident` on Metal.
11. **M-16** (A/B only): untracked buffers on the depth bench.

Not proposed, because the record already closed them: split-KV / dedup attention variants, Stage-B
GEMV, ICB, unretained references, megakernel, dispatch-count fusions, rope2+kv_store, sa_qv,
batched small-M verify, fused argmax, `WILLNEED`, sampler overlap above temperature 0.2.
