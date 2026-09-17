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

**NOTE 2026-09-13 — ten closures added a day late.** M-01, M-02, M-03, M-04, M-06, M-08, G-01,
G-03, G-07 and C-01 were all shipped on 2026-09-13 (`6cc862a0`, `f6c222ee`, `c660ab78`, `84c3f29f`,
`a30f2cd3`, `c7f4b1e5` — all on `main`), but the fix commits didn't touch this file, so their
entries below sat unmarked as fully open for a day after the code moved past them. Found via
`git log` against the finding IDs, not from this doc's own prose — a reminder that
`doc_review_staleness.py`'s per-doc footer check (own-edit vs cited-file dates) doesn't catch this
shape of drift: nothing here cites `metal/model.go`'s *lines* by number, and even a citation
wouldn't show a fix that only ADDS code near an existing one. Closures below are now current as of
this note's commit; treat the two negative-probe closures (M-09, M-10) and the four already-present
ones (M-07 partial, G-02 partial, G-04 partial, G-08) as unaffected — those were already caught
same-day.

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
is set on a premise the repo's own gate record contradicts.** `metal/backend.go:431-333` says "K=256
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
misses the budget by 3.75 GB (M-07). This is L4 of `docs/tasks/task-int4-layout-2026-09.md`, now with
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
- **Where:** `decoder/model.go:1189` (`kvOnly, hasKV := m.resident.(ResidentPrefillKV)`),
  `decoder/residency.go:109-109`; `metal/backend.go:379-560` (the complete `metalResident` method
  set — no `ForwardNoLogits`); `metal/model.go:1414-1384` (`encodeLogitsCB`, the only executor job
  shape, always appends `pGemvW8`); `metal/model.go:1298-1281` (`forwardHiddenNoHead` — the
  trunk-only encode already exists, used only by `HiddenLast`); `metal/model.go:1353`
  (`finalizeLogits` memcpy).
- **Mechanism and bound (counted + record):** `hasKV` is false for `*metalResident`, so
  `residentPrefillSeed` takes `m.resident.Forward(emb, i)` for every prompt token. Which prompts
  are sequential on Metal: every prompt below the 512 floor (M-02), every adapter prompt at any
  length (`decoder/model.go:1164`, C-01 of the prior audit), every family `prefillOK` rejects
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
- **CLOSED 2026-09-13, narrower than scoped.** Shipped `ForwardNoLogits` on `*metalResident`
  (`6cc862a0`) so `residentPrefillSeed`'s KV-only prefill skip now applies to Metal — synchronous
  (`forwardHiddenNoHead`), not the fuller `noHead`-bit-on-`execJob` version this finding's Fix
  section describes, which stays a follow-up worth ~0.9 ms/token of currently-unclaimed
  encode-ahead overlap. Falls back to the full head-bearing `Forward` on a paged MoE resident,
  since `forwardHiddenNoHead`'s trunk encoder has no paged branch (the same gap C-02 found at
  `HiddenLast`/`ForwardArgmax` — this does not introduce that defect on a third entry point).
  Verified: `TestForwardNoLogits_byteIdenticalKV` (dense, logits match full-logits `Forward`
  exactly), `TestForwardNoLogits_pagedMoEFallback` (paged MoE, same requirement through the
  fallback, real `qwen3_5_moe-tiny` fixture), `TestResidentPrefillSeed_metalKVOnly_byteIdentical`
  (end-to-end through `decoder.Model.Generate`, mirroring CUDA's own gate) — all new, all passing;
  full `go test ./metal/...` and `-tags goinfer_testhooks` both green.

#### M-02 · The 512-token floor keeps every prompt under 512 tokens sequential, on a stated reason the repo's own gate record contradicts
- **Where:** `metal/backend.go:431-333` (`const metalFastPrefillFloor = 512` — "K=256 is expected to
  fail §3.2"), `:426-431` (the decline), `:340,358` (the same file: "gate passed 2026-09-09 (S
  model, K=256/512/1024)"); `docs/measurements/prefill-gate-l1-ref-b-2026-09-09.md:23` ("S K=256 …
  94.2% / 4 / 0.0328 vs exact 93.0% / 4 / 0.0347 **PASS**"), `prefill-l2-metal-fused-attn-2026-09-09.md:138`
  (fused gate, K=256 pass again); `docs/completed/task-prefill-gap.md:226-227` ("Moving the floor DOWN
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
- **CLOSED 2026-09-13, shipped exactly as scoped.** `metalFastPrefillFloor` 512 → 256 (`6cc862a0`).
  Verified: `go test ./metal/...` (79 pass) and `-tags goinfer_testhooks` (129 pass, 52 skip) both
  green, `go test ./decoder/...` (488 pass), gofmt clean, staticcheck clean (same pre-existing
  U1000s as `main`, none new).

#### M-03 · `gemm_w4f16_store` dequants each weight tile with 8 of 32 lanes, runs 4 MMAs per barrier pair, and stages neither operand — the flat 3.3–3.6× GEMM term at every K
- **Where:** `metal/prefill.go:66-119` (kernel; `if (lane < 8u)` dequant at `:87-92`, `RPS 4` at
  `:23`, per-k-step barrier pair), `:590-596` (grid), `docs/completed/task-prefill-gap.md:159-162` ("Metal's
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
- **CLOSED 2026-09-13, shipped as scoped.** Widened the per-simdgroup block to 32×32 (`f6c222ee`):
  all 32 lanes dequant per k-step (was 8 busy/24 idle), 16 MMAs per barrier pair (was 4), each A
  row-tile loaded once per k-step and reused across the 4 column tiles. N masked per 8-wide
  sub-tile on the ragged remainder (only guaranteed `%8==0`, per C-10). Validated against the
  audit's own designated oracle, not bit-identity (f16 storage rounding already differs from
  exact): `TestPrefillGateVsReference` on the S model, the deciding pooled set (K=256/1024) plus
  K=3900 — verdict SHIPS on all three criteria, new kernel slightly ahead of the old one (73 vs 75
  hard flips, meanKL 0.2982 vs 0.3008, lower on 17/20 pooled cells). D7 declined via the fit-guard
  on this run (an environmental memory gap on this Mac at the time, unrelated to the change).

#### M-04 · `attention_prefill_fused` keeps O in threadgroup memory and rescales it with 8 scalar lanes — ~34 barrier-separated phases per 8-key tile, the residual O(K²) term
- **Where:** `metal/prefill.go:286-329` (`threadgroup float oScr[ATTN_SGPT][8*ATTN_MAXHD]`),
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
- **CLOSED 2026-09-13, narrower than scoped.** Widened the key tile to 32 (`c660ab78`: QKᵀ score
  MMAs for all 4 sub-tiles run first with no barrier between them, then one softmax pass over up
  to 32 columns, then PV accumulation sums all sub-tiles before the one store+rescale — amortising
  the barrier-heavy phase ~4×). The register-accumulator + diagonal-alpha-MMA rewrite and GQA
  query-head grouping this finding's Fix section also describes were deliberately scoped OUT — more
  invasive for a benefit not obviously net-positive once the diagonal-MMA's own f32→f16 round-trip
  is counted — left as a separate, more speculative follow-up. Validated against the same §3.2
  pooled oracle as M-03 (not bit-identity — the online-softmax rescale already reorders the sum):
  deciding set SHIPS on all three criteria (hard flips tied 75=75, meanKL 0.2992 vs 0.3008, fast
  lower on 18/20 cells), K=3900 confirmation cell matching (meanKL 0.4852 vs 0.4904). D7 again
  declined via the fit-guard (same pre-existing memory gap as M-03's run).

#### M-05 · MoE batched prefill runs the FFN half as M sequential rows; paged/DeltaNet families prefill as M decode tokens — bounded by M × active-expert bytes, undocumented
- **Where:** `metal/prefill.go:625-674` (`for m := 0; m < M; m++ { … r.encodeMoEExperts(e, L, moeDst) }`),
  `metal/moe.go:703-682`; `metal/model.go:718-682` (paged/g4moe/DeltaNet → `prefillOK=false`);
  `metal/backend.go:537-445` (`PrefillPath` reports "batched f16-MMA" for it);
  `docs/tasks/task-gpu-paths-2026-09.md:1184-1191` (G8: "Mirrors CUDA's own established shape exactly").
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
- **NOT CLOSED — interim step done 2026-09-13, the real fix not attempted.** The bound this
  finding computed is now recorded beside G8 (`docs/tasks/task-gpu-paths-2026-09.md`, dated
  2026-09-13 entry, exactly the ask this Fix line makes for the interim), so "batched" is no
  longer undocumented as a TTFT promise. The layer-major/expert-major dispatch restructuring
  itself is a real, multi-file kernel-design project (a new batched-K GEMM shape, not a parameter
  change) and was deliberately left for a dedicated pass rather than attempted under this
  session's time budget.
- **Pre-implementation probe attempted 2026-09-14, real-measurement half inconclusive on this
  hardware; arithmetic case holds.** Before committing to the restructuring, built a cheap
  validation: a telemetry probe (`metal/moe_expert_reuse_probe_test.go`,
  `expertPool.distinctExperts`) that runs real text through the exact sequential paged-prefill
  path already in production and reports `stages` (today's actual re-fetches) vs
  `distinctExperts` (the expert-major floor) vs naive `M·k`. The arithmetic case above — expert-major
  reads each distinct expert once per prefill regardless of M, a ≈280× reduction in the dominant
  bandwidth term at this shape's k=4 — used only this repo's own already-measured 84–105 GB/s
  W4A8 ceiling, no new number, and stands regardless of what follows.
  <br>The live measurement itself could not be completed on this 16 GB Mac. Loading
  `qwen15-moe-a27b` (int4, Metal, `MoECacheExperts`) into the auto-sized non-paged case failed on
  severe swap pressure 7/7 times before M-07's fix; after M-07 shipped (extending row4-skip to
  every dense/MoE projection, not just Q/K/V/gate/up) the same load succeeded twice — confirming
  M-07 was the actual blocker for the LOAD step, not anything MoE-paging-specific. But forcing a
  real slot count (`GOINFER_MOE_REUSE_PROBE_SLOTS=8`, since the auto-sizer now sizes all 60
  experts resident post-M-07 and no longer pages by default) to actually exercise the paged
  forward path drove swap to 15.6 GB within under a minute — worse and faster than the pre-M-07
  failures — and was killed before completing a single layer's stage count. Eight attempts total
  (7 pre-M-07 + 1 post-M-07-forced-paging) have now failed on this hardware; further retries were
  judged not worth the risk to a shared machine.
  <br>This is itself informative: the paged forward path — the exact mechanism this finding's Fix
  targets — appears to carry real, severe host memory/swap cost beyond what M-07 touches, on top
  of the already-documented dispatch-count and bandwidth costs. Consistent with, and does not
  contradict, the finding's own "Deliberate and documented for parity; the bandwidth term ... is
  not [documented]" framing — if anything it strengthens the case that the restructuring is worth
  doing. No numeric measurement was obtained; the arithmetic case remains the only quantified
  justification on record. A real measurement would need either a machine with materially more
  headroom, or a smaller-than-production MoE fixture built specifically to keep the paged case
  inside safe memory bounds.

#### M-06 · Gemma 3 never reaches the batched prefill — `prefillFeatures` still lacks `FeatPerLayerRoPE` (prior audit M-23, open)
- **Where:** `metal/model.go:112-122` (the map: no `FeatPerLayerRoPE`), `decoder/features.go:152`
  (`add(!a.ropeUniform(), FeatPerLayerRoPE)` — every shipped Gemma 3, 5:1 local/global, derives it),
  `metal/prefill.go:594-627` (the dispatch already binds `L.invf`/`L.uWindow` per layer; the comment
  at `:624-625` says the feature "is not claimed").
- **Mechanism and bound:** every real Gemma 3 prompt is sequential: M-01's 671 MB head per token on
  4B, at 13.5 ms/token. Note admitting it would still route Gemma 3 (hd=256) to the *exact*
  `attention_prefill` (`ATTN_MAXHD 128`, `metal/prefill.go:580`) — the 46 GB/layer re-read shape — so the
  fused kernel needs an hd=256 variant for the full win.
- **Fix:** `decoder.FeatPerLayerRoPE: true` in `prefillFeatures` (safe: `FeatRopeMscale` stays
  undeclared so per-layer *mscale* families still decline); give `testdata/gemma3-vl-tiny` a global
  layer so `TestPrefillParityGemma` covers the real shape (G-03); then an hd=256 fused variant.
- **Confidence:** confirmed. **Prior:** audit-2026-09-10 M-23 — unchanged by the 53 commits.
- **CLOSED 2026-09-13, first half shipped, hd=256 fused variant not attempted.** Declared
  `decoder.FeatPerLayerRoPE` in `prefillFeatures` (`84c3f29f`) — `prefill.go`'s dispatch loop
  already bound `L.invf`/`L.uWindow` per layer, so the per-layer table was always implemented, it
  just wasn't claimed. `FeatRopeMscale` deliberately left undeclared so a per-layer *mscale* family
  (Mellum's YaRN long-context variant) still declines. Landed together with G-03 (the fixture fix
  that lets this be gated at all) and C-01 (a ragged-tile OOB read M-06 raises the odds of firing,
  found while admitting Gemma 3 to this path) in the same commit. Verified against a freshly
  regenerated checkpoint: `go test ./metal/...` (80 pass), `-tags goinfer_testhooks` (130 pass, 52
  skip), `go test ./decoder/...` (488 pass, including the two VL image-path tests the regeneration
  moved), `-tags 'gpu goinfer_testhooks' ./gpu/...` (99 pass), `./multimodal/...` (10 pass), gofmt
  and staticcheck clean. The hd=256 fused-attention variant this finding's Fix section names as
  needed for Gemma 3's *full* win was not attempted — Gemma 3 now reaches the batched GEMM path but
  still routes to the exact (unfused) attention kernel at hd=256.

### B. Decode: two probes, one adapter kernel, one memory item

#### M-07 · A GGUF/safetensors int4 load on `--backend metal` keeps THREE copies of every dense projection: host canonical + host row4 repack (read by nothing once resident) + the Metal buffer
- **Where:** `decoder/weightmat.go:473-425` (`wantsCanonicalInt4`: `if backendName != "cpu" { return
  true }`), `:440-444` (`repackedOnlyOrCanonical` → `repackW4A8IfEligible(canon)` — both kept),
  `:251-257` ("both ALLOCATE A SECOND BUFFER and keep the canonical nibbles alongside"),
  `metal/model.go:470-444,475-478` (`int4DirectWords` → `NewBufferUint32s` = `newBufferWithBytes`,
  a third copy); `decoder/fitguard.go:275-260` (the guard prices int4 at ~2× on arm64 because of
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
  `docs/tasks/task-int4-layout-2026-09.md`**, now with its number and a second half (drop after upload).
- **Confidence:** confirmed (three allocation sites traced; the 889.6 MB is the commit's own
  measurement). **Prior:** audit-2026-09-10 M-24 (the guard's 2× double-count — the real footprint
  is 3×); aikit M-22; task-int4-layout L4 (filed, not scheduled — schedule it).
- **PARTIALLY CLOSED 2026-09-13 — row4-skip shipped, host-release deferred.** L4's own text
  ("FILED, not scheduled... its own measurement") held this back from the first Program pass;
  reopened after a direct measurement (TestW4A8Row4_loadTimeAndMemoryDelta, decoder's own
  existing sanctioned toggle): row4 exactly DOUBLES the resident int4 footprint — 223.6 MB
  canonical + 223.6 MB row4 on the 0.5B fixture (100.0% additional RAM), +15% load time. A real,
  substantial win, not a theoretical one.
  <br>Shipped: `wantsRow4Fallback(backendName)` (decoder/weightmat.go) — false ONLY for the
  literal "metal", threaded through `repackedOnlyOrCanonical`/`quantizeBatchedProjWM`/
  `streamQuantizedBatchedProj` down to both Load entry points (GGUF and safetensors). Scoped to
  Q/K/V/gate/up (`quantizeBatchedProjWM`'s own five standard tensor names) — NOT o_proj,
  down_proj, router, or MoE experts, which route through `quantizeWM`, a separate function with
  its OWN unconditional `repackW4A8IfEligible` call used from ~40 family-specific call sites
  across weights.go's per-architecture builders. Reaching those too would multiply this fix's
  blast radius well past what this pass could safely verify; left as a larger follow-up — this
  fix captures a real slice of the measured 223.6 MB, not its entirety, and down-proj alone is
  the audit's own N-10-cited ~22% of per-token weight bytes. Embed/LMHead deliberately
  unaffected (`quantizeEmbedWM`/`streamQuantizedEmbed` pass skipRow4=false unconditionally) —
  read via `.Row()` on the host on every backend regardless of GPU residency.
  <br>Verified: TestW4A8Row4_skippedForMetalBackend (new) confirms Backend:"metal" zeroes row4 on
  exactly the 5 scoped tensor types per layer (120/120 on the 0.5B) while o_proj/down_proj stay
  row4 (48/48, correctly unaffected) and canonical bytes remain present (Metal's own GPU upload
  path is untouched — it always read canonical, never row4); Backend:"" (unspecified) is
  bit-for-bit unchanged (still both). Full `go test ./metal/...` (82 pass) and `-tags
  goinfer_testhooks` (132 pass) both green; `go test ./decoder/...` green including
  `scripts/refresh_parity_hashes.sh`'s 37 forward goldens (0 failed, 1 driving the quantized
  int4 path) — a provably non-numeric core edit for every path except Backend:"metal" itself,
  where canonical bytes (the only thing any numeric computation reads) are byte-for-byte
  unchanged regardless of row4's presence. Re-ran the real S-model §3.2 oracle gate
  (TestPrefillGateVsReference) after this change: K=256's cell numbers matched every prior
  same-session baseline exactly (92.8%/93.8% agree, meanKL 0.0375/0.0340) — zero measurable
  numeric drift, as expected.
  <br>NOT done (at the time): the fix's own second half ("release the layer projections' host
  WeightMats after BuildResident succeeds") — a genuine object-lifecycle change with its own risk
  profile, out of scope for that pass; and extending skipRow4 to quantizeWM's ~40 call sites
  (down-proj/router/experts), per above.
- **CLOSED (row4-skip half) 2026-09-13 — the ~40 remaining `quantizeWM`/`streamQuantized` call
  sites now skip row4 on Metal too; host-WeightMat release investigated and declined.**
  <br>Shipped: `quantizeWMSkipRow4`/`quantizeWMRow4` and `streamQuantizedSkipRow4`/
  `streamQuantizedRow4` (decoder/weightmat.go, sharing a body with the pre-existing
  `quantizeWM`/`streamQuantized`), threaded as a `skipRow4 bool` parameter through every
  family-specific builder that the prior pass's scoping note left out: the generic safetensors
  path's `loadMatQ`/`loadProj` (o_proj/down_proj/router — the single most common code path, since
  every "standard" architecture routes through it), `loadFusedExperts`, `loadGemma4MoE`/
  `streamExperts`, `loadQwen35Attn`; the family builders `buildGPT2Weights`, `buildGraniteWeights`,
  `buildNemotronWeights`, `buildPhi3Weights`, `buildLlama4Weights` (decoder/weights.go),
  `buildInternLM2Weights` (decoder/internlm2.go), and `buildGptOssWeights`
  (decoder/gptoss_safetensors.go, including its row-streamed MXFP4 experts); and the GGUF path's
  `streamMat`, `stackedExperts`, `fusedSplit`, and the Qwen3.5-specific `wmQ`/`wrapQ` closures
  (decoder/gguf.go). `Embed`/`LMHead` deliberately stay on plain `quantizeWM` everywhere (unchanged
  from the prior pass) — read via `.Row()` on the host on every backend regardless of GPU
  residency, so row4 there is never wasted. Threaded as a parameter rather than a package-level
  variable because `parallelLayers` loads layers concurrently within one `Load()` call (and the
  server can load multiple models concurrently) — a global would have been a real, silent
  data race and a cross-model leak of one model's backend choice into another's.
  <br>Verified: `TestW4A8Row4_skippedForMetalBackend` extended — o_proj/down_proj now also
  row4-free on Backend:"metal" (120/120 Q/K/V/gate/up + 48/48 o_proj/down_proj row4-free, all with
  canonical bytes present; Backend:"" unaffected, still 120/120 + 48/48 row4). New
  `TestW4A8Row4_skippedForMetalBackend_MoE` covers router/experts/shared-expert across both MoE
  tensor shapes this repo has (generic `Router`/`Experts`/`SharedExpert` via `qwen3_5_moe-tiny`: 64
  tensors row4-free / 60 unaffected on unspecified backend; gemma4's own
  `expertsGateUp`/`expertsDown` via `gemma4-moe-tiny`: 16 / 16). TDD discipline applied per code
  path (not just per test): reverting `loadProj`'s branch alone left both MoE tests green, showing
  neither fixture's router/experts route through it (they use `loadMatQ`/`loadGemma4MoE`
  instead) — confirmed `loadMatQ` is the load-bearing branch for `qwen3_5_moe-tiny` and `streamMat`
  (decoder/gguf.go) for the heavy-checkpoint GGUF test, each independently red-without-fix and
  green-with-fix. Full `go test ./decoder/...` green; `scripts/refresh_parity_hashes.sh` (required —
  weightmat.go/weights.go are both `core`-tier in `testdata/parity_manifest.json`) ran 37 forward
  goldens, 0 failed, and refreshed 35 deps_hash lines with nothing else touched — the expected
  non-numeric outcome, since canonical int4 bytes are byte-identical with or without a row4
  side-copy. `go vet`, `gofmt -l`, and CI-pinned `staticcheck` (v0.8.0, `./decoder/...` and
  `./metal/...` including `-tags goinfer_testhooks`) all clean on every touched file (staticcheck's
  metal findings are pre-existing U1000s in untouched files, gated behind `goinfer_testhooks`).
  <br>**Declined: host-WeightMat release.** Investigated via a dedicated safety review before
  writing any code. Found NOT safe to implement as the fix text originally scoped it: Sessions
  (the normal serve-app conversational path), the `resBusy`-loser CPU fallback, three of the four
  speculative-decode loops, and `LoadAdapter`'s `validateComputeTimeDims` all read the same
  per-layer host `WeightMat`s at arbitrary times *after* a resident GPU backend has already been
  built — this is not a rare edge case, it is most of the ways this codebase uses a loaded model.
  Separately, the `.Rows() > 0` idiom used at ~30+ call sites to feature-detect an optional weight
  ("does this exist at all") would be silently corrupted by a naive struct-zeroing release: a
  released-but-present feature would misreport as absent instead of erroring, which is worse than
  the memory it would save. No safe, bounded version of this half presented itself in the
  investigation, so it is not scheduled — this is a permanent decision, not a deferral, unless a
  future change (e.g. an explicit "host copy no longer needed" flag threaded through every one of
  those readers) reopens it.
  <br>**Net effect:** every dense/MoE projection on `--backend metal` now keeps two copies (host
  canonical + Metal buffer) instead of three; the row4 side-copy — proven to exactly double the
  resident int4 footprint (100.0% additional RAM, measured on the 0.5B fixture) — is gone for
  every projection this loader builds, not just the five `quantizeBatchedProjWM`-scoped ones from
  the prior pass. The host canonical copy is retained by design (`.Row()` lookups, CPU fallback,
  LoRA merge, speculative decode all need it) and is out of scope for any future fix short of the
  declined half above.

#### M-08 · `lora_delta` runs each projection's whole adapter delta in ONE threadgroup — the design CUDA's P-11 measured at +124%/token; Metal's P-11 fused two dispatches and kept the serial block
- **Where:** `metal/kernels.go:848-873` (`// ONE THREADGROUP ONLY … looping over ranks serially …
  is cheap`; `for (uint r = 0; r < R; r++)` with a 256-wide tree reduce + barrier per rank; up
  stage `Out/256` rows per thread), `metal/lora.go:211-216` (`e.Dispatch(r.pLoraDelta, tgReduceNorm,
  tgReduceNorm, …)` — n == tg == 256, one threadgroup); `docs/completed/audit-2026-09-10.md:4250-4296`
  (moved here 2026-09-16 when the live audit's closed findings were archived; CUDA:
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
- **CLOSED 2026-09-13, shipped as scoped, kept fused (Metal's own tradeoff, not CUDA's).**
  `a30f2cd3`: widened the grid to `ceil(Out/256)` threadgroups, each owning a fixed, disjoint
  256-row block of the up stage and independently recomputing the down stage's `t[R]` rather than
  reading it from a device-memory scratch buffer a second kernel wrote — kept as ONE dispatch
  (unlike CUDA's `lora_delta_down`/`_up` split) because a second launch is the more expensive line
  item against Metal's own launch/sync ceiling; the redundant per-threadgroup `R` reduction this
  costs is cheap next to that. Also stores A/B as f16 (the delta feeds an int8-quantised
  activation, so f32 precision was never load-bearing, and halving the bytes matters most here
  since every threadgroup in the dispatch re-reads A whole). New gate
  (`TestLoRADelta_multiThreadgroupMatchesReference`, `Out=600` forcing 3 threadgroups with a ragged
  last one) confirmed to actually catch the bug class it exists for: maxAbs 8e-6 with the fix,
  23.9 with a temporarily reintroduced single-threadgroup grid. The existing whole-model parity
  test alone would not have caught this — its fixture's widest projection is 128, one threadgroup
  either way. Verified: `go test ./metal/...` (81 pass), `-tags goinfer_testhooks` (131 pass, 52
  skip), gofmt and staticcheck clean.

#### M-09 · Decode attention's K read is a 32-lane 512 B-strided gather (32 load instructions per 256 B row) — every recorded probe fits an L1/LSU-transaction wall as well as the "DRAM latency" reading; the one untested corner is bit-identical cooperative staging
- **Where:** `metal/kernels.go:642-636` (thread `tid` owns keys `tid, tid+128, …`; per key 32 `half4`
  loads at stride `kvDim*2` = 512 B across lanes), `:661-677` (V: thread `d` walks all keys
  serially, 2 B per load), `metal/model.go:2045` (12 threadgroups × 128 threads per layer);
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
- **Where:** `metal/kernels.go:231-233` (`W4A8_BODY`: 8 scalar byte loads of the activation + one
  half scale per 32-bit word), `:272-274` ("int8 activation staged once into threadgroup short —
  replaces the per-row device byte-gather (17920× re-reads) that dominates LSU issue"),
  `metal/model.go:1956` (`e.Dispatch(r.pGemvResid, r.H*32, 32, L.dW, …)` — one simdgroup per row,
  no staging), `:1005-1009` (the M-06 comment records the choice as accounting, not a
  measurement); `docs/completed/task-metal-batched-verify-kernel.md:166-167` (isolated: down-proj 68 GB/s vs
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
- **Where:** `metal/gemma4_moe.go:497-538` (`begin()`/`end()` per phase; `end` = commit +
  `waitUntilCompleted`; two per MoE layer), `metal/moe.go:826-802` (same, generic);
  `metal/residency_probe_test.go:11-12` ("~15 ms/boundary of GPU-idle-in-wait, 72× Step-0's 0.213
  ms"); `metal/pagecost_sharedevent_test.go:47-64` (verdict "recovers ~0%" — measured on
  qwen2.5-coder-1.5b, dense); `metal/model.go:1175-1144` (residency-set comment: p1 still carries
  the pinned set, +2.07 ms/CB); `metal/moe.go:136` (expert kernels already index
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
- **PARTIALLY CLOSED 2026-09-14 — the contiguous-pool/slotIdx half shipped and verified; the
  shared-event single-command-buffer half and the real 26B/35B re-measurement are NOT done.**
  Shipped exactly the first half of this finding's own Fix line: `expertpool.go`'s per-layer pool
  is now ONE contiguous Buffer per field (guW/guS/dW/dS), not N separately-allocated slot objects —
  `newExpertPool` allocates `N*strideWords` once; `ensureResident`/`ensureResidentBatch` return a
  slot NUMBER (still wrapped in `expertSlot` for the staging path's own use, but callers now read
  only `.slot`). `slotIdx` (renamed from `idxZeros`) is a `[topK]` device buffer the host writes
  each token with the pool ROW holding each routed expert; `encodeG4Phase2Paged` /
  `encodeMoEExpertsPaged` now always bind the pool's fixed-identity base buffers plus `slotIdx`,
  exactly mirroring how the non-paged stacked-all-E path already reads `rIdx` at
  kernel-execution time (`gemv_w4a8_moe`'s `idx[slot]*rowsPerExpert`) — zero new MSL kernels
  needed, since that addressing pattern already existed for the non-paged case. This makes phase
  2's ENCODE graph value-independent for the first time — the specific blocker this finding names
  ("phase-2 binds *which slot buffer* at encode time") — in both `gemma4_moe.go` and `moe.go`'s
  generic MoE path (gpt-oss's separate `biasIdx()` index space is untouched: it still always reads
  the router's real expert ids, never the pool slot, per its own doc comment).
  A real correctness trap surfaced and was fixed while wiring this: `Buffer.U32s()`/`.U16s()`
  ignore `Buffer.At()`'s bind offset (they always read from the buffer's base address — confirmed
  by reading aikit's own `gpu/metal.go`), so the pread-staging closures' original
  `preadIntoU32Buf(fd, s.guW, off)` calls (using an `.At()`-offset slot view as the destination)
  would have silently pread every expert into the pool's slot 0 — a live bug this refactor would
  have shipped had the "run every existing gate" pass not caught it. Fixed with two new
  slot-number-addressed helpers (`preadIntoPoolSlot`/`preadRangeIntoPoolSlot`) that compute the
  destination byte offset against the pool's base buffer directly, bypassing `.At()` entirely; the
  byte-copy staging path has the identical hazard on the scale buffers and is fixed the same way
  (`copyU16sToBuf`, using `gpu.Upload` — which DOES respect `.At()` — instead of `U16s()+copy()`).
  TDD-verified per this session's discipline: deliberately zeroed every `slotIdx` write (always
  slot 0, ignoring the real slot) and confirmed `TestGemma4Paging_bitExact` failed loudly (2040/2040
  logit mismatches) rather than passing by coincidence; restored and reconfirmed 0 mismatches.
  Verified bit-exact (not just cosine) on both paged shapes this repo already has small fixtures
  for, cold-start/eviction/same-expert-reuse all exercised in both: `TestGemma4Paging_bitExact`
  (gemma4-moe-tiny, N=3/4 slots, 8 positions, 0 mismatches) and `TestMoEPaging_matchesNonPaged`
  (qwen3_5_moe-tiny, N=2 and N=3, 32 tokens each, exact match) — plus the isolated pool-primitive
  tests (`TestExpertPool_lruAndStaging`, `TestExpertPoolBatch_matchesSequential`,
  `TestExpertPoolBatch_pread`, `TestExpertPoolBatch_overCapacityPanics`) all green under `-race`.
  `gofmt`/`go vet -tags 'darwin goinfer_testhooks'`/CI's pinned staticcheck all clean (staticcheck
  caught the two pread helpers going briefly unused mid-refactor, which is what surfaced the
  offset bug above — a real instance of this repo's own "prove the gate can go red" discipline
  paying for itself, not just a documentation exercise).
  Deliberately NOT attempted, and why: the shared-event single-command-buffer integration (tearing
  `forwardLogitsPaged`/`forwardLogitsMoEPaged` into ONE command buffer per token with
  `EventBoundary` at each paged-layer seam, adapting the `forwardLogitsSharedEvent` prototype) and
  the real 26B/35B re-measurement this finding's Fix line calls for both require loading a real
  paged 26B/35B checkpoint on this Mac — the exact class of work ([[metal-moe-paging-needs-speculation]],
  [[mac-16gb-model-size-limits]]) that produced a kernel panic and 8/8 failed probes earlier in
  this same audit pass (see M-05's own "pre-implementation probe attempt" note, above). The
  contiguous-pool change is a real, tested prerequisite either way — it does not by itself change
  the number of command-buffer boundaries per token, so it carries no expected perf effect and no
  new risk to a running paged decode; the actual single-CB rework and its measurement remain
  explicitly owed, gated on the same real-hardware caution as M-05.
- **Where:** `metal/moe.go:847` (was a serial `ensureResident` loop — see this finding's own
  closure note), `:535-553` (three sequential
  `preadRangeIntoU32Buf` per expert + `int4DirectBytes` scale narrowing on the host),
  `metal/expertpool.go:230-203`; `docs/completed/task-metal-expert-streaming-at-scale.md:236-242`.
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
- **CLOSED 2026-09-13, cross-expert half shipped; per-expert 3-preads-concurrently NOT done.**
  Added `expertPool.ensureResidentBatch(ids []int) []expertSlot` — slot selection/eviction
  bookkeeping still runs sequentially (LRU/`where`/`slotExpert` are not safe for concurrent
  mutation, and two misses must never be handed the same slot), but every miss's actual staging I/O
  (the pread or mmap-byte-copy call) now runs under a `WaitGroup`, concurrently. Wired into both
  paged forwards' per-layer expert loop (`metal/moe.go`'s `forwardLogitsMoEPaged`,
  `metal/gemma4_moe.go`'s `forwardLogitsPaged`), replacing the old serial `for j := range k {
  ensureResident(ids[j]) }`. The finding's OWN second half — 3 concurrent preads WITHIN one
  expert's `stagePread` — was not done this pass; cross-expert concurrency is the larger of the
  two levers per the sweep's own math (k experts in flight vs. 3 spans of one), and doing both at
  once would have doubled what one pass needed to get right.
  <br>**Two real bugs found and fixed while building this, both via a failing test, before either
  reached the real forward paths:** (1) deferring `touch()` (LRU-reorder) until after staging
  left `pickSlot`'s eviction fallback reading `p.lru`'s PRE-BATCH ordering while `slotExpert`
  already showed every slot claimed so far as occupied — a batch with enough misses would either
  index `p.lru` out of range or hand two different misses in the same batch the same slot;
  `touch()` now runs in the same sequential pass that claims the slot, before any I/O starts. (2)
  the out-of-contract case (a batch's distinct-expert count exceeding the pool's own slot count —
  which `newExpertPool`'s own doc comment says must never happen in production, `N >= top-k`
  enforced at build time by `TestMoESlotsViaOptions_belowTopKRefusesWithNumbers`) hit exactly the
  same slot-reuse failure via a different path; `ensureResidentBatch` now panics loudly on it
  instead of racing two goroutines over one buffer.
  <br>New tests: `TestExpertPoolBatch_matchesSequential` (batch vs. sequential produce IDENTICAL
  eviction decisions, staged contents, AND bookkeeping counters — not just "looks right this
  time"), `TestExpertPoolBatch_pread` (the same, through the `stagePread` fast path this finding
  is actually about), `TestExpertPoolBatch_overCapacityPanics`. All three, plus the existing
  `TestGemma4Paging_bitExact` and `TestMoEPaging_matchesNonPaged` (real paged-vs-non-paged forward
  parity on `gemma4-moe-tiny`/`mixtral-tiny`, now exercising `ensureResidentBatch` through the
  actual wired call sites — 0 logit mismatches), pass under `-race`.
  <br>**Not measured**: the actual wall-clock win needs the real cold-read latency a 26B/35B's
  expert spans have and this session had no such checkpoint to run against — the sweep's own
  ≤1.6× bound is unverified, and the finding's own "per-miss cost rising with N" signature (the
  actual evidence this is latency-bound, not something else) can only be re-checked on that
  hardware. Verified for correctness only: `go test ./metal/` (88 pass), `-tags goinfer_testhooks`
  (140 pass), both `-race` clean on the MoE/expertpool subset, gofmt/vet/staticcheck clean.
- **CLOSED 2026-09-14 — the second half, 3(/2) concurrent preads within one expert's stage, now
  shipped too.** `metal/moe.go`'s `stagePread` closure (3 spans: gate, up, down) and
  `metal/gemma4_moe.go`'s twin (2 spans: fused gate|up, down) each issue their preads under a
  `sync.WaitGroup` instead of sequentially, exactly as this finding's own Fix text asked. Safe by
  the same argument the finding gives: `pread` on a shared fd is positional, and every span writes
  a disjoint destination range (gate/up land at different byte offsets within the same `guW`
  buffer; down is a separate buffer entirely) — no destination overlap, so no synchronization
  needed between the writes themselves. Errors are collected into per-span variables and panicked
  from the ORIGINAL goroutine after `wg.Wait()`, not from inside the spawned ones — an unrecovered
  panic in a goroutine other than the one `recover()` runs on crashes the whole process instead of
  being caught by `BuildResident`'s existing panic-decline-to-CPU path, so this had to be
  preserved deliberately, not just made concurrent.
  <br>Verified: `TestMoEPagingPread_matchesByteCopy` (exact logit equality across non-paged /
  paged+byte-copy / paged+pread arms, and confirms `pool.preads > 0` so the comparison isn't
  silently exercising the byte-copy fallback) and the `TestExpertPoolBatch_*` set both green under
  `-race` — the detector is specifically built to catch exactly the class of bug this change could
  have introduced (concurrent unsynchronized writes to overlapping memory) and found nothing. The
  full `TestGemma4*` subset (paging/parity/kernels) also green under `-race`, covering
  `gemma4_moe.go`'s twin change even though it has no dedicated pread-vs-byte-copy comparison test
  of its own (a pre-existing coverage gap, not introduced here). gofmt/vet/staticcheck clean.
  <br>**Still not measured**: same caveat as the first half — the ≤1.6× wall-clock bound needs
  real 26B/35B cold-read latency this session had no checkpoint for.

#### M-13 · The Metal expert pager engages only with an explicit `--moe-cache-slots N`; the default declines the 26B/35B/gpt-oss-20b to the CPU-staged path, and the peer-matrix row that "parked" them on the Mac measured that fallback
- **Where:** `metal/backend.go:154-159` (`metalMoESlotsRequest`: flag or env only; 0 ⇒ unpaged),
  `metal/moe.go:429-434`, `metal/backend.go:317-255` (guard prices the *unpaged* set when slots are
  unset, declines to CPU; the message names `GOINFER_NO_RESIDENT_MEM_GUARD` but not
  `--moe-cache-slots`); `internal/serveapp/main.go:521` (`--moe-cache-experts` … "CUDA only"),
  `:488` ("Metal: every expert resident, unpaged"); `docs/benchmarks.md:1686-1696` ("falls back
  automatically to a CPU-staged … path … killed after 2h10min with zero completions");
  `docs/completed/task-metal-expert-streaming-at-scale.md:288-291` (recommendation: default N=64).
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
- **CLOSED 2026-09-13, shipped as scoped.** `metalMoESlotsRequest` now calls `autoMoESlots` when
  `MoECacheExperts()` is set and no explicit `--moe-cache-slots` was given, deriving N =
  clamp((0.7·RAM − needFixed) / perSlotBytes, topK, 64) — `needFixed` and `perSlotBytes` both read
  from the SAME byte accessors `residentNeedBytes` already calls (`ResidentDenseWeightBytes`,
  `ResidentHostCopyBytes`, `residentKVBytes`, and the marginal
  `ResidentWeightBytesPaged(2)-ResidentWeightBytesPaged(1)` for one slot's bytes), not re-derived
  by hand. `moeTopK` supplies the floor for both MoE shapes this backend admits (Gemma-4 via
  `Config.TopKExperts`, generic via `MoEResidentParams`). Since `metalMoESlotsRequest` is the one
  chokepoint both `metal/moe.go` and `metal/gemma4_moe.go` already read at build time (and
  `residentNeedBytes` reads via `metalMoESlotsFromEnv`), fixing it here reaches the memory guard,
  the real paging engagement, and the CLI's own advertised default all at once. Also named
  `--moe-cache-slots`/`--moe-cache-experts` in the decline message for an MoE model that doesn't
  fit, and corrected `--moe-cache-experts`'s "CUDA only" / `--moe-cache-slots`'s "Metal: every
  expert resident, unpaged" CLI help text (`internal/serveapp/main.go`) — plus, found in the same
  spot, a stale "DEFAULT ON above 512 tokens" that M-02 had already moved to 256.
  <br>The pure formula (`autoMoESlotsFor`) is split out for direct unit testing against a
  deliberately small `ram` value, mirroring `fitsResidentBudget`'s own established pattern — no
  real machine holds less RAM than a real MoE model's dense term, so this is the only way to
  exercise the tight-budget and clamp-to-topK branches at all. On `testdata/mixtral-tiny` (this
  Mac's real RAM), the auto-derived request is 64 (≥ nE=8), which correctly builds UNPAGED — a
  tiny model has no business paging, and this is the expected, not a failing, result; forcing real
  paging engagement through auto-sizing needs a model whose dense term is a meaningful fraction of
  real RAM, which no committed fixture is. New tests: `TestAutoMoESlotsFor` (5 cases: ceiling,
  small-RAM sizing, clamp-to-topK, fixed-part-alone-doesn't-fit, degenerate perSlot),
  `TestMoeTopK_realFixtures`, `TestMetalMoESlotsRequest_autoOnlyWithCacheExperts` (confirmed red
  without the fix: `metalMoESlotsRequest` returned `""` with `MoECacheExperts` set). Verified:
  `go test ./metal/` (85 pass), `-tags goinfer_testhooks` (137 pass), `go test
  ./internal/serveapp/...`, gofmt/vet/staticcheck clean.

#### M-14 · The residency set rides on every command buffer because aikit's binding has `Queue.AddResidencySet` but not `MTLCommandBuffer.useResidencySet:` — a recorded +62 ms/token waiting on one selector (aikit item)
- **Where:** aikit `residencyset.go:108-109` (only the queue-level attach; `:22-29` lists five
  selectors, none per-command-buffer); `metal/model.go:1175-1144` ("+2.07 ms/CB → +62 ms/tok …
  FIX (fold into the next aikit release …): a PHASE-SCOPED residency set attached only to phase-2's
  command buffers (per-encoder useResidencySet)").
- **Mechanism and bound (record):** phase 1 never touches the slot pool but carries the ~3 GB
  pinned set in its referenced list on every commit: ~4% of the paged 26B token, growing with N.
  aikit v1.40/v1.41 did not carry the selector.
- **Fix:** `func (e *Encoder) UseResidencySet(rs ResidencySet)` — one `objc.RegisterName
  ("useResidencySet:")` + one send; `gemma4_moe.go`/`moe.go` call it on the phase-2 encoder only and
  drop the queue attach. Ships with M-11.
- **Confidence:** confirmed. **Prior:** new at the seam; the cost is goinfer's own record.
- **PARTIALLY CLOSED 2026-09-13 — aikit binding shipped, goinfer wiring blocked on a release.**
  Added `Encoder.UseResidencySet(rs)` to aikit (`5065f24`) — exactly the fix text's shape, one
  `objc.RegisterName("useResidencySet:")` + one send on the encoder. New test
  (`TestEncoder_useResidencySet`) confirms a dispatch under the per-encoder attach completes
  correctly AND that a sibling encoder on the SAME queue without it is unaffected — the scoping is
  real, not silently promoted queue-wide. Not wired into `gemma4_moe.go`/`moe.go`'s phase-2
  encoders: goinfer's `metal/go.mod` pins `aikit/gpu v0.32.0` from the module proxy with no
  `replace` (confirmed by attempting the wiring directly — `e2.UseResidencySet` doesn't compile
  against the pinned version), and this backend submodule has never carried a tag (same status as
  C-05/C-06). The wiring itself is now fully designed (drop `r.q.AddResidencySet(rs)`, keep
  `r.residency = rs`, call `e2.UseResidencySet(r.residency)` on phase 2 only, guarded on
  `r.residency` being set) and is a goinfer-only change once a `gpu/vX.Y.Z` tag lands.
- **CLOSED 2026-09-14 — `gpu/v0.33.1` released (the same tag as C-05/C-06/G-06), goinfer wired.**
  `5065f24` was already an ancestor of the tagged commit, so the release unblocked this
  immediately — no extra aikit work needed. Wired exactly as designed above: `metal/model.go`
  drops `r.q.AddResidencySet(rs)` (kept only `r.residency = rs`); `metal/gemma4_moe.go` and
  `metal/moe.go` each call `e2.UseResidencySet(r.residency)` (guarded on `r.residency !=
  (ResidencySet{})`) right after opening phase 2's encoder, before `encodeG4Phase2Paged`/
  `encodeMoEExpertsPaged`. Verified against the real `gemma4-moe-tiny` fixture:
  `TestResidencySet_pinsExactlyTheLiveSlots` still confirms the pinned set is exactly the pool's
  24 live slot buffers (the consistency contract this session's wiring could have silently broken
  is unaffected — pinning moved from queue-attach to per-encoder-attach, not the SET's contents).
  Full `go test -tags metal ./metal/...` (98 pass / 3 fail / 11 skip — the 3 failures are N-41,
  confirmed pre-existing and reproducing identically before this change) and `-tags
  goinfer_testhooks` both green; gofmt/vet/staticcheck clean (staticcheck's findings are the same
  pre-existing `goinfer_testhooks`-gated U1000s, line-shifted only). The measured 62 ms/token this
  finding recorded (p1 idle from carrying the ~3 GB pinned slot set on every command buffer, not
  just phase 2's) has not been re-measured post-fix — this session had no paged 26B/35B checkpoint
  to reproduce the original cold A/B on; the wiring is verified correct, not re-benchmarked.

### D. Cross-repo and unassessed

#### M-15 · On a Metal box every image turn runs the vision tower on the CPU, and aikit's Metal tower cannot be wired as a win until three shapes change (aikit M-14/M-09/M-10 Metal halves)
- **Where:** `internal/serveapp/main.go:1055-992` (`EnableResident` only for `webgpu`; nothing imports
  `visionmetal`/`qwenmetal`); aikit `metal_vit.go:168-221` (attention: one threadgroup per
  (head, query), re-streams K and V per query — no query tile; score lanes 4,608 B apart; PV keeps
  hd=72 of 256 lanes busy), `:397-420` (`gemm_w8a8_tiled`: one output per thread, byte-granular
  staging, scalar int8 — the shape CUDA's M-14 retired), `qwenmetal/encoder.go:188-213,311-334`
  (per-op `Run1D`/`Run2D`, each a commit + `waitUntilCompleted` + pool drain: 544–704 synchronous
  submits per image at 32 blocks); aikit `CHANGELOG.md:292-293` (batched SigLIP tower 0.46×/0.33× of
  CPU by its own crossover), `:316-318` (M-10 "NOT DONE: the Metal half"); `docs/multimodal.md:172`
  ("Metal — still not started"), `docs/benchmarks.md:554-557` (CPU SigLIP 31.3 s/image).
<!-- citation-lint: allow-path qwenmetal/encoder.go aikit's own SEPARATE Go module (own go.mod), added after the aikit/gpu v0.32.0 release goinfer's cuda/go.mod currently pins — goinfer does not depend on it yet (line 477's own "nothing imports qwenmetal" is this in prose), so no checked-out or module-cache root can verify it here. -->
<!-- citation-lint: allow-path visionmetal/encoder.go same as qwenmetal/encoder.go above: aikit's own separate, not-yet-pinned Go module. -->
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
- **PARTIALLY CLOSED 2026-09-14 — all three aikit fixes shipped (two wired, one deliberately
  not); goinfer-side wiring and a real crossover measurement NOT done.** Attempted all three
  fixes this Fix line names, in increasing order of risk:
  <br>**1. Ported visionmetal's `runBatch` to qwenmetal (aikit `21d6f3d`).** `ForwardViT` now
  opens one `gpu.Encoder` per block (patch embed gets its own) instead of a `Run1D`/`Run1DTG`/
  `Run2D` per op — 17-22 dispatches/block down to 1 command buffer/block, 544-704 submits/image
  down to 33. The three shared scalar buffers (safe only because each pre-fix dispatch committed
  and waited before the next write) are replaced with visionmetal's own per-dispatch ring, reset
  once per command buffer. TDD-verified: temporarily dropping the ring reset made
  `TestQwenMetal_parityWithCPU` panic on ring exhaustion immediately rather than silently
  corrupting results. `go test -tags metal -race` green: parity cosine 1.000000000 vs CPU
  unchanged, 4 repeated forwards bit-identical.
  <br>**2. Ported CUDA's `gemm_w8a8_reg` to Metal as `gemm_w8a8_reg` (aikit `319717c`), wired
  into both towers' int8 projections via a new `GEMMW8A8Plan`.** MSL has neither a
  `simdgroup_matrix` int8 form nor a hardware dp4a intrinsic, so a `dp4a_manual` helper unpacks
  packed-int8 words via arithmetic right-shift and sums the four lane products in scalar
  registers — still a 4× cut in threadgroup-memory traffic per operand loaded, plus 4×4 register
  blocking, over the byte-granular `gemm_w8a8_tiled` every int8 projection ran before. Bit-identical
  to `gemm_w8a8_tiled` by the same associativity argument CUDA's own kernel relies on. New
  `TestMetal_gemmW8A8Reg` mirrors `cuda_vit_w8a8reg_test.go`'s structure (routing + bit-exact
  equality across 6 shapes incl. the real so400m MLP shape). TDD-verified: removing the
  sign-extension from `dp4a_manual` produced a large numeric mismatch immediately. `go test
  -tags metal -race ./gpu/...` green (34 tests), whole-tower parity gates unchanged at cosine
  1.000000000 (confirms bit-identity in production use, not just the synthetic kernel test).
  <br>**3. Built query-tiled online-softmax attention as `attention_tiled` (aikit `060fdae`),
  verified standalone, DELIBERATELY NOT wired into either tower.** This is the one genuinely
  novel piece: no reference implementation exists anywhere in aikit, CUDA included — aikit's own
  CHANGELOG (the CUDA M-14 half) says so explicitly ("the query-tiling the audit actually asks
  for... needs an online/flash-style softmax, which re-associates the sum and is a numerics
  decision rather than a refactor"). Implemented the standard flash-attention running-max/
  running-sum/rescale-on-max-update recurrence: `AT_QTILE`(32) queries share one threadgroup,
  each on its own thread; `AT_KTILE`(16)/`AT_MAXHD`(128, mirroring CUDA's own unrelated
  K-staging kernel's `ATTN_KTILE`/`ATTN_MAXHD`) bound the static K/V staging arrays. New
  `TestMetal_vitAttentionTiled` (float64 CPU reference) measured worst Δ 5.36e-07 — tighter than
  the untiled kernel's own 5e-5 bound despite the re-association; gated at 1e-4 for margin.
  `TestMetal_vitAttentionTiled_matchesUntiled` (vs. production `attention` directly): worst Δ
  4.47e-07. TDD-verified: removing the running-accumulator rescale (the classic flash-attention
  ordering bug) produced Δ 0.735 immediately. Left unwired on purpose: this Fix line's own
  parenthetical — "re-baseline the ViT parity gate" — is named as its own explicit step, and
  swapping the kernel into either tower's production `attn` call changes that tower's whole
  parity-gate baseline, which needs a measurement pass on real hardware and real shapes, not a
  decision folded silently into the kernel's own landing.
  <br>**NOT done:** (a) goinfer-side wiring (`EnableResident` for `cfg.backend == "metal"` in
  `internal/serveapp/main.go`, both the generic `vision.Encoder` path AND the separate
  `loadQwenVisionTower`, which never calls `EnableResident` on any backend today) — blocked on a
  fresh `gpu/vX.Y.Z` release bundling the three commits above, the same release ritual C-05/C-06/
  G-06 already went through this session; (b) the crossover measurement itself, since this
  session had no real SigLIP/Qwen2.5-VL checkpoint or Metal hardware benchmark run to confirm the
  Mechanism section's arithmetic actually closes the gap to the CPU tower's recorded times, only
  that each kernel change is individually correct; (c) `docs/multimodal.md:172` /
  `docs/benchmarks.md:554-557`, the two doc paths this finding's own Where cites, do not exist
  under those names in the current aikit tree — reconciling the promised "crossover row" needs
  finding wherever that content now lives first.

#### M-16 · Every buffer is hazard-tracked and every encoder serial; the binding exposes neither the untracked option bit nor `computeCommandEncoderWithDispatchType:`, and the record calls the resulting per-dispatch floor "unassessed"
- **Where:** aikit `metal.go:438,432,443,491,500` (every `newBuffer*` passes `options = 0` =
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
- **CLOSED 2026-09-13, NEGATIVE — not a lever.** Built the discriminating probe the Fix text called
  for, in aikit (`aikit/gpu/hazard_tracking_probe_test.go`, `a2dffd6`) rather than rewiring any real
  goinfer allocation path first: one command buffer, 310 dispatches cycling 20 shared buffers (the
  reuse pattern hazard tracking actually has to do work on), tracked vs untracked, interleaved
  reps. Four interleaved runs: untracked measured 99.6% / 100.4% / 100.4% / 101.5% of tracked's
  GPU-busy time — noise around zero, sign flipping between runs, nowhere near a real effect. The
  probe's own measured floor (~3.4 µs/dispatch) lands close enough to the tree's recorded ~3.8
  µs/dispatch to trust its shape as representative. Hazard tracking is not the mechanism behind the
  dispatch floor; something else in the per-dispatch/pipeline-state path is. Added
  `newBufferLenUntracked` to aikit as an UNEXPORTED helper for the probe only — no production
  consumer, no permanent public API surface for a proven-dead lever. Not re-proposed; the
  concurrent-dispatch-type follow-up this finding's Fix text named is gated on this probe moving,
  which it did not.

---

## 2. Correctness

#### C-01 · `attention_prefill_fused` reads K/V rows past `nKeysMax` on a ragged last tile — past the cache end when `ctxCap % 8 ≠ 0` (prior audit N-46, open)
- **Where:** `metal/prefill.go:299-346` (`for (uint j0=j0start; j0<nKeysMax; j0+=8u)` with
  `simdgroup_load(kT, kBase + j0*kvDim …)` — full 8-row tiles, mask applied after the load at
  `:352-363`), `metal/model.go:951-919` (`kc/vc` sized `ctxCap*kvDim*2`), `:57-74` (`ctxCap` = the
  user's request, unrounded), `metal/attention_prefill_fused_test.go:39,63-65` (M=37, exactly
  `cacheLen` rows — passes because MTLBuffers are page-rounded).
- **Failure:** `--resident-context 1000` with a prompt that fills it → up to 7·kvDim·2 bytes read
  past `kc[l]`/`vc[l]`; a non-finite value in an unspecified tail gives 0×Inf = NaN in that row's O.
  OOB reads are tolerated on this hardware, so no fault — a wrong row instead.
- **Fix:** round `ctxCap` up to a multiple of 8 rows when sizing `kc/vc` (one line), or clamp the
  tile and zero-fill in-kernel. Make the unit test allocate exactly `cacheLen` rows on a
  non-page-rounded size so it can see the read.
- **Confidence:** confirmed (read pattern); NaN outcome plausible. **Prior:** N-46.
- **CLOSED 2026-09-13, first fix option shipped.** Rounded the `kc`/`vc` allocation up to a multiple
  of 8 rows in `buildResident` (`84c3f29f`) — `r.ctxCap` itself (the checked, user-visible capacity)
  is unchanged, only the underlying buffer's allocated size. Landed alongside M-06, which raises the
  odds of this tile shape actually firing (Gemma 3's sliding_window caps). New gate
  (`TestAttentionPrefillFused_ctxCapNotMultipleOf8`) asserts the allocated `Buffer.Len()` directly
  rather than observing the OOB read's effect on output — a first attempt drove `PrefillLast` to the
  boundary and diffed logits against sequential `Forward`, and it passed identically with the fix
  reverted, because Metal's actual buffer backing is page-rounded (16 KB on Apple silicon)
  regardless of the requested length, so a 37-row and a 40-row request for a buffer this small land
  on the same physical allocation either way — the structural gate is the one that actually
  discriminates: fails red (2368 vs 2560 bytes) with the fix reverted, passes with it restored.

#### C-02 · `HiddenLast` (serve `/v1/embeddings`), `Forward(id,pos)` and `ForwardArgmax` on a paged MoE bind the zero-value stacked-expert buffers — C-08's defect on three more entry points
- **Where:** `metal/backend.go:614-551` (`HiddenLast` → `forwardHiddenNoHead` per position),
  `metal/model.go:1298-1271` (→ `encodeTrunkInto` → `encodeLayer`, `:1808-1813` — no paged branch;
  paging lives only in `Forward`'s dispatch to `forwardLogitsPaged`, `:1241`), `metal/moe.go:300-291`
  ("expGuW/expGuS/expDW/expDS stay zero-value when paged"), `:651-659` (bound unconditionally);
  `metal.go:777-748` (OOB/unmapped reads are silently tolerated).
- **Failure:** on a `--moe-cache-slots` MoE (generic or Gemma 4), an embeddings request returns a
  finite garbage vector with no error; the snapshot golden's `Forward` path likewise.
- **Fix:** decline in `HiddenLast`/`Forward`/`ForwardArgmax` when `r.g4moe.paged || r.moe.paged`,
  or make `encodeLayer` refuse a layer whose `pool != nil`. Note this also constrains M-01's fix.
- **Confidence:** confirmed (structurally; not reproduced). **Prior:** audit-2026-09-10 C-08 (fix
  covered `PrefillLast` only).
- **CLOSED 2026-09-13.** `HiddenLast` now declines up front when `r.g4moe.paged || r.moe.paged`
  (`metal/backend.go`), so `decoder.Model.HiddenLast` falls through to the CPU path exactly like an
  OOM/cap decline already does — the fix text's first option. For `Forward(id,pos)`/`ForwardArgmax`
  (test/gate-only; decoder.ResidentForward's production `Forward(embedding,pos)` always goes
  through `ForwardEmbPipe`, which IS paged-aware), took the fix text's second option instead: a
  chokepoint panic in `encodeMoEFFN`/`encodeGemma4MoEFFN` (the non-paged FFN encoders `encodeLayer`
  dispatches a MoE layer to) if it ever reaches a paged layer — covers any future caller of
  `encodeLayer`, not just these two, and calls `e.FinishEncoding()` before panicking so an
  in-progress Metal command encoder isn't released without `endEncoding` (that assertion fired on
  the first version of this guard; fixed before landing). New gate:
  `TestPagedMoE_forwardEntryPointsDecline` (`metal/c02_paged_forward_entrypoints_test.go`) on the
  tracked `testdata/mixtral-tiny` fixture — confirmed red without the fix (`HiddenLast` returned a
  finite result at cosine 0.993 against the correct CPU answer, not just a theoretical
  possibility). Verified: `go test ./metal/` (82 pass), `-tags goinfer_testhooks` (133 pass, 53
  skip), `go test ./decoder/...` all pass, gofmt/vet/staticcheck clean.

#### C-03 · Adapter `In`/`Out` are never checked against the base projection on Metal (prior audit M-05, open) — a same-family different-size adapter writes past the Q slot into K/V
- **Where:** `metal/lora.go:147-149` (rank-only check), `metal/kernels.go:873-861,868-872` (`Ar = A +
  r*K`, `out[row]` for `row < Out` — both from the adapter's own uniforms).
- **Fix:** in `conv`, refuse when `p.In`/`p.Out` differ from the projection's `K`/`N` (seven
  comparisons; `SetAdapter` has `r.H`, `r.I`, `L.geom`). The decoder-side chokepoint the prior
  audit proposed is the better home.
- **Confidence:** confirmed. **Prior:** M-05.
- **CLOSED 2026-09-13, fixed at the decoder chokepoint instead of in `metal/lora.go`.** The actual
  gap was one level deeper than this finding's own "Where": `decoder/lora.go`'s `validateTargets`
  (called from `Model.LoadAdapter`, the compute-time path every resident backend's `SetAdapter`
  reads from) only checked that a delta's tensor NAME was a known projection, never that its
  `[Out,In]` SHAPE matched the actual base weight — the merge-at-load path
  (`weights.go`'s `loadProj` → `loraAdapter.merge`) already made exactly this check, compute-time
  LoRA never did. Added `validateComputeTimeDims` (`decoder/lora.go`), called right after
  `validateTargets`, comparing every targeted delta's `In`/`Out` against the already-loaded
  `WeightMat.Cols()`/`.Rows()` for that layer/projection. Fixed once at the one chokepoint every
  backend passes through (CPU, Metal, CUDA, WebGPU) rather than duplicated per backend — Metal's
  `SetAdapter` itself is unchanged. New gate: `TestLoadAdapter_dimMismatchRejects`
  (`decoder/lora_compute_test.go`) — a synthetic base plus a same-family adapter shaped for one
  extra attention head; confirmed red without the fix (the mismatched adapter loaded silently).
  Verified: `go test ./decoder/...` all pass, Metal's own LoRA parity tests
  (`TestLoRADelta_multiThreadgroupMatchesReference`, `TestLoRAResidentParityMetal[_armedExecutorThenBind]`)
  still pass unchanged, gofmt/vet/staticcheck clean.

#### C-04 · `SetAdapter`'s error path leaks the partially built bind until `Close`
- **Where:** `metal/lora.go:157-181` (`if out[i].q, err = conv(l.Q); err != nil { return err }` —
  buffers for layers `0..i` stay on the ledger; `releaseLoRALayers` not called).
- **Fix:** `defer` a release of `out` on error. **Confidence:** confirmed. **Prior:** new (minor
  impact, ratchets across failed binds).
- **CLOSED 2026-09-13, shipped exactly as scoped.** A `bound` flag latches true only after every
  projection in every layer converts cleanly; a deferred `releaseLoRALayers(r.d, out)` fires
  whenever `bound` is still false, undoing exactly the partial work this call itself allocated
  (`releaseLoRALayers` already skips nil projections, so the not-yet-reached layers past the
  failure point are no-ops). New gate: `TestSetAdapter_partialBindErrorReleasesBuffers`
  (`testdata/llama-tiny`, 4 layers) — layer 0 converts a valid rank-4 projection, layer 1 fails on
  an invalid rank, and the device ledger (`Device.LedgerLen()`) is asserted unchanged across the
  failed call. Confirmed red without the fix (ledger grew 115→121 allocs on the same run).
  Verified: `go test ./metal/` (82 pass), `-tags goinfer_testhooks` (134 pass, 53 skip), gofmt/vet/
  staticcheck clean.

#### C-05 · aikit `Queue.Run1DBatchTG` / `Run1DTG` own an NSAutoreleasePool without the G22 OS-thread pin every sibling helper has
- **Where:** aikit `metal.go:983-928` (`pool := … Send(selInit); defer pool.Send(selDrain)` with
  no `runtime.LockOSThread`) vs `:585-589,621-625,925-929` (siblings pin, citing "intermittent
  SIGSEGV (fault 0x10) inside objc_msgSend"). Only production caller (`qwenmetal.ForwardViT`) pins
  for the whole forward; goinfer's batch-k harnesses call it unpinned.
- **Fix:** the same two lines as `Run1DBatch`. **Confidence:** confirmed (shape). **Prior:** aikit
  G22 (missed one helper).
- **CLOSED 2026-09-13 in aikit, committed locally (`d8c2878`), NOT yet released/bumped into
  goinfer.** Added the same `runtime.LockOSThread()`/`defer runtime.UnlockOSThread()` pair every
  sibling helper (`Run1D`, `Run2D`, `Run1DBatch`) already has, directly in `Run1DBatchTG`
  (`Run1DTG` delegates straight to it, so both are covered by the one fix). Confirmed the gap was
  real, not just shape-plausible: `gpu/metal_vit_test.go` had grown its own manual `run1dTG`
  wrapper specifically to pin around the unpinned library call — direct evidence this had already
  been worked around once. Verified: `go test ./gpu/` green, including
  `TestMetal_vitAttentionSeg` (exercises `Run1DTG` on the exact `AttentionSeg` kernel
  `qwenmetal.ForwardViT` dispatches), `go vet -tags metal ./gpu/...` and gofmt clean. This is an
  aikit-repo fix: committed to aikit `main` locally, CHANGELOG entry added under `[Unreleased]`,
  but not pushed, tagged, or released — goinfer's `go.mod` still pins the pre-fix aikit version
  until a deliberate release + bump (see `RELEASING.md`) lands it.
- **RELEASED 2026-09-14 — `aikit/gpu` tagged `gpu/v0.33.1`, goinfer bumped onto it
  (`metal/go.mod`, `cuda/go.mod`).** Cut alongside C-06 and G-06 below (all three landed in the
  same `gpu/` tag, since none had any exported-surface conflict). `RELEASING.md`'s GPU submodule
  ritual run in full: `preflight` PASS 10/10, `gpugate` PASS 5/6 applicable (`ptx-repro` n/a on
  Apple Silicon), `gpudevice` run on BOTH platforms — PASS 5/5 applicable on this Mac
  (Darwin/arm64, backend:metal) and PASS 5/5 applicable on `nobara-pc` (Linux/x86_64,
  backend:cuda) — 9/9 gpu modules covered combined. First tag (`gpu/v0.33.0`) failed CI's
  `gpu tag evidence` gate because its message paraphrased the device verdicts instead of pasting
  the tool's literal `VERDICT:` lines the gate regex-matches; fixed forward with `gpu/v0.33.1`
  carrying the verbatim lines, per `RELEASING.md`'s own "the tag is immutable, fix forward"
  guidance. `consumergate --tag gpu/v0.33.1` PASS from outside the repo. goinfer's own
  `metal/close_leak_test.go` now consumes `CurrentAllocatedSize()` directly (see G-06's own
  closure note) — this finding is fully closed, not just shipped-and-queued.

#### C-06 · `visionmetal` has no threadgroup-memory budget guard (qwenmetal's C-02 fix was not mirrored), and the status latch it relies on cannot fire on Apple silicon
- **Where:** aikit `visionmetal/encoder.go:225-228` (`DispatchTG(…, np*4, …)` — over the 32 KiB
  limit above np=7,680) vs `qwenmetal/encoder.go:247-250,368` (`attnThreadgroupBytes` guard);
  `metal.go:775-748` ("silently tolerates … over-budget threadgroup memory … status Completed").
- **Failure:** a SigLIP tower with >7,680 patches returns a plausible wrong hidden state. Not a
  shipped shape today (so400m/896 = 4,096).
- **Fix:** the qwenmetal check with `np` in `newEncoder`. **Confidence:** plausible. **Prior:** aikit
  C-02 (qwenmetal only).
- **CLOSED 2026-09-13 in aikit (`a5b23b2`), NOT yet released/bumped into goinfer.** Shipped exactly
  as scoped: `attnThreadgroupBytes(np)` (mirroring qwenmetal's identical formula for the same
  shared kernel) checked once in `newEncoder`, before any layer weight is touched — np is fixed at
  build time here, unlike qwenmetal's per-image patch count checked per `ForwardViT` call. New
  tests: `TestAttnThreadgroupBytes` (arithmetic) and `TestNewEncoder_declinesOverBudgetPatchCount`
  (end-to-end, no checkpoint needed — a synthetic `vision.GPUWeights` with an over-budget
  `NumPatches` reaches the check before any real weight data is required). Confirmed load-bearing:
  reverting `encoder.go` makes the test file fail to build (`attnThreadgroupBytes` undefined).
  Verified: `go test ./gpu/visionmetal/...` green (including the real SigLIP-checkpoint parity
  test), gofmt/vet/staticcheck clean. Same status as C-05: committed and pushed to aikit `main`,
  not tagged/released — this backend submodule has never carried a tag (`RELEASING.md`'s "the eight
  that have never been tagged"), and goinfer's `metal/go.mod`/`cuda/go.mod` pin `aikit/gpu v0.32.0`
  regardless; nothing in goinfer currently calls `visionmetal` (M-15 is still open), so nothing is
  blocked by the absence of a release.
- **RELEASED 2026-09-14 — `gpu/v0.33.1`, same tag as C-05.** See C-05's own release note for the
  ritual details. `visionmetal` is still not called from anywhere in goinfer (M-15), so this
  release closes the fix's provenance but not any live goinfer code path — recorded for
  completeness, matching C-05's own "the eight that have never been tagged" caveat about the
  backend submodule itself.

---

## 3. Gates that cannot fail

#### G-01 · `TestMoE_declinesPrefill` has been red on every Metal box since the 512 floor landed; three commits carry it as "pre-existing, unrelated"
- **Where:** `metal/moe_model_test.go:302,321-327` (8 embeddings; sets `GOINFER_METAL_BATCHED_PREFILL=1`
  only), `metal/backend.go:530-431` (floor check precedes `prefillOK`); log lines 859, 909, 1213.
- **Mechanism:** 8 < 512 ⇒ decline ⇒ `Fatalf` before any MoE code runs; the admit-side MoE-vs-dense
  check C-08's fix relies on has not run green since the floor. A `go test ./metal/` that is always
  red trains everyone to ignore it.
- **Fix:** `t.Setenv("GOINFER_METAL_FAST_PREFILL_FLOOR", "0")` as `metal/prefill_ttft_test.go:45` does.
- **Confidence:** confirmed (three reviewers).
- **CLOSED 2026-09-13, shipped exactly as scoped.** Landed in the same commit as M-01/M-02
  (`6cc862a0`): disabled the floor via `GOINFER_METAL_FAST_PREFILL_FLOOR=0`, the same pattern
  `prefill_ttft_test.go` already used, since this test is about MoE arch admission, not the floor.

#### G-02 · The §3.2 pooled gate still drops missing cells silently and turns a fit-guard decline into a SKIP that "SHIPS" (prior audit G-08, open)
- **Where:** `metal/prefill_gate_ref_test.go:186-202` (`if cs != nil { … }` — a missing reference
  file is dropped; only zero cells fails; the header prints the full K set), `:114-117` (D7 that
  fails to build → `Skipf`). Both records say D7 was decided-around by fit-guard: the floor and both
  default-ON flips that govern 7B-class Mac users rest on the 1.5B alone (and M-07 is why D7 does
  not fit).
- **Fix:** `len(decisionCells) != len(decisionKs) ⇒ Fatalf`; a decision model that does not build is a
  FAIL when deciding. **Confidence:** confirmed. **Prior:** G-08.
- **CLOSED 2026-09-13, first half fixed.** Implemented the missing-reference-file half exactly as
  scoped: a DECIDING pooled set (`deciding=true`) now Fatalfs if `len(decisionCells) !=
  len(decisionKs)`, naming which K is missing and where to generate it, rather than silently
  pooling a partial set — `deciding=false` (the set-A re-score) is unaffected, matching its own
  "never fails the test on its own" contract. Verified live: re-running the gate on S now Fatalfs
  at exactly this check once K=256 and K=1024 complete (K=512's reference file is still missing —
  a real, pre-existing data gap this fix now surfaces instead of hiding). The SECOND half this
  finding named — "D7 that fails to build → Skipf" — was NOT reproduced: every D7 run this audit
  session actually hit `t.Fatalf("load: %v", err)` at the fit-guard's hard refusal (the citation's
  `:114-117` line numbers now point at the `ResidentForwardForTest` type-assertion Skipf, a
  DIFFERENT decline shape — a model that loads but silently isn't GPU-resident — which this
  session never observed either). Left as-is rather than fixed speculatively against a symptom
  not reproduced; re-open if a real "D7 loads, isn't resident, SKIPs, gate reports SHIPS anyway"
  case turns up.

#### G-03 · `TestPrefillParityGemma` gates the "Gemma set" on a fixture shaped so `prefillOK` is true — the shape every real Gemma 3 lacks
- **Where:** `metal/prefill_gemma_test.go:44-47`; fixture `[sliding, sliding]` shares one RoPE base
  → `ropeUniform()` true; real Gemma 3 derives `FeatPerLayerRoPE` and is declined before the kernel
  runs (M-06). **Fix:** a global layer in the fixture. **Confidence:** confirmed. **Prior:** M-23.
- **CLOSED 2026-09-13, shipped as scoped.** Forced one sliding + one full_attention layer in
  `scripts/pin_gemma3_vl_tiny.py` (`84c3f29f`) — the `rope_parameters` block already had both bases
  defined, only `layer_types` needed to change — and regenerated the checkpoint plus both dependent
  goldens through the real HF pipeline (not by hand-editing `config.json`; the checkpoint directory
  is gitignored and regenerates locally, the goldens are the only durable committed record). Also
  refreshed `int4_forward_goldens.json`'s gemma3-vl-tiny entry, whose recorded moments the
  weight/config change moved. Verified against the freshly regenerated checkpoint (`-count=1`
  throughout, since Go's package-level test cache doesn't invalidate on testdata content changes)
  — see M-06's closure note for the full test tally.

#### G-04 · The absolute snapshot golden is re-baked by the code it checks after each accepted kernel round; the autoresearch ledger is outside the tree
- **Where:** `metal/snapshot_golden_test.go:20,177-184` ("the ABSOLUTE STORED REFERENCE";
  `GOINFER_UPDATE_GOLDENS` writes whatever current code produces), `metal/kernels.go:37-29` ("only
  deep-mantissa sha bits move — see the round's own commit"); `docs/tasks/task-autoresearch-loop.md:3`
  ("not started") and §3 ("Do NOT point it at Metal") vs `metal/kernels.go:113,147,729-754` (Metal rounds
  ran on the norm-class kernels); `scripts/autoresearch_rmsnorm_results.tsv` not in
  `docs/measurements/`.
- **Mechanism:** the one gate that sees reduction-order/fast-math drift was refreshed to the new
  bits on an argmax-unchanged argument; which rounds re-baked and which reverted cannot be audited
  from the repo. **Fix:** a re-bake in the same commit as a kernel change needs its own gate line
  (argmax-at-every-checkpoint + cosine vs the previous golden, in `docs/measurements/`), or the
  golden is declared an OS-drift detector only; move the tsv into `docs/measurements/`; fix the
  autoresearch doc's status. **Confidence:** plausible (the tsv is not here).
- **PARTIALLY CLOSED 2026-09-13 — doc corrected, tsv/gate-line not done.** Fixed
  `docs/tasks/task-autoresearch-loop.md`'s stale "not started" status to note the contradiction directly
  (kernels.go's own comments cite the tsv as a real Metal experiment) rather than restate the
  wrong claim. The tsv itself could not be moved into `docs/measurements/` — it is not anywhere in
  the tree, not merely misplaced; whatever generated it either never committed the file or it was
  lost since, so the reduction-order/SHA history it would have recorded is unrecoverable from this
  repo. The gate-line fix (argmax-at-every-checkpoint + cosine vs the previous golden on a re-bake)
  is real, unbuilt infrastructure work on its own "plausible" confidence rating — left for a
  dedicated pass rather than added speculatively here.

#### G-05 · The shared-event verdict was measured on a shape without the cost it targets
- **Where:** `metal/pagecost_sharedevent_test.go:47-64` (qwen2.5-1.5b dense int8int8; "recovers ~0%"),
  `metal/pagecost_measure_test.go:47-52` ("There is NO such checkpoint on this Mac … this measures
  the SUBMISSION-STRUCTURE cost on a DENSE model"), `metal/residency_probe_test.go:11-12` (paged
  26B: ~15 ms/boundary). Carried into production comments as settled (`metal/gemma4_moe.go:488-463`,
  `metal/model.go:1977-1974`). **Fix:** M-11's re-run. **Confidence:** confirmed.

#### G-06 · The device-ledger "did Close/ReleaseBuf free it" assertions pass by construction
- **Where:** `metal/close_leak_test.go:162-169,224-248` vs aikit `metal.go:396-390`
  (`ids := d.allocs; d.allocs = nil; … for … Send(selRelease)`): `LedgerLen()` is emptied
  regardless of whether `release` is sent; the test's own comment (`:229-231`) records RSS "DID NOT
  ratchet" under a neutered `ReleaseAll` because macOS compressed the pages. **Fix:** assert on
  `MTLDevice.currentAllocatedSize` (exact, compression-immune). **Confidence:** confirmed.
- **CLOSED 2026-09-14 in aikit (`91de2d8`), NOT yet released/bumped into goinfer.** Shipped exactly
  as scoped: `gpu.Device.CurrentAllocatedSize()`, wrapping `MTLDevice.currentAllocatedSize`
  directly — independent of `Device`'s own `allocs`/`objs` bookkeeping, so it reflects whether the
  underlying native memory was actually freed rather than whether the Go-side ledger was cleared.
  Confirmed the gap was real, not just shape-plausible, with the same TDD discipline this session
  used throughout: temporarily removed `ReleaseAll`'s release loop (simulating exactly this class
  of leak — the ledger clears, nothing native is actually freed) and reran the *existing*
  `TestLedger_buffers`; it still passed, proving `LedgerLen()==0` alone cannot catch this bug.
  Restored, then added `TestCurrentAllocatedSize_reflectsRealAllocation` (grows/shrinks real GPU
  memory, checks `CurrentAllocatedSize` moves with it), which correctly failed against the same
  neutered `ReleaseAll` and passes with it restored. `go test -tags metal ./gpu/...` green (34
  pass / 0 fail / 3 skip), `go vet -tags metal ./gpu/...` and `gofmt` clean. This is an aikit-repo
  fix: `gpu.Device` and its ledger live there, not in goinfer's `metal` package — pushed to aikit
  `main`, CHANGELOG entry added under `[Unreleased]`, but not tagged/released. goinfer's own
  `metal/close_leak_test.go` still asserts on `LedgerLen()` only; consuming
  `CurrentAllocatedSize()` there to close this finding fully needs a deliberate aikit release +
  bump (see `RELEASING.md`) — same queued state as C-05 and C-06.
- **FULLY CLOSED 2026-09-14 — `gpu/v0.33.1` released, goinfer now consumes
  `CurrentAllocatedSize()`.** See C-05's release note for the tagging ritual. Both
  `metal/close_leak_test.go` gates the finding names now assert on it alongside the existing
  ledger check, not instead of it (the ledger still proves this package's own bookkeeping is
  correct; the new assertion proves the native memory really moved):
  `TestMetal_PrefillScratchDoesNotLeak` reads `r.d.CurrentAllocatedSize()` before/after 30
  `PrefillLast` calls (measured: 391,348,224 → 391,348,224 bytes, exactly flat);
  `TestMetal_CloseWithSecondModelAlive` reads it before/after closing model A with B still alive —
  confirmed `CurrentAllocatedSize` reflects the whole *physical* device, not a per-`*Device`-handle
  value (verified directly: allocating through one handle shows up identically through another's,
  since `MTLCreateSystemDefaultDevice` hands back retains of the same GPU), so the combined total
  dropping from A+B's 782,172,160 bytes to B-alone's 391,610,368 after `a.Close()` is real evidence
  A's memory was freed, not an artifact of querying the wrong handle. Both real-device tests green
  (`GOINFER_HEAVY_TESTS=1`, qwen2.5-coder-0.5b-instruct-q4_k_m.gguf); full `go test -tags metal
  ./metal/...` green.

#### G-07 · `TestPrefillGate` (superseded §3 form) would `Fatalf` on its first cell; three files cite `TestMetalPrefillDivergenceRate`, which does not exist
- **Where:** `metal/prefill_gate_test.go:65,75,254-257` (K=256 without the floor override);
  `metal/spec_prefill_regression_test.go:46`, `metal/spec_verify_curve_test.go:22`,
  `docs/measurements/prefill-gate-l1-2026-09-05.md:66` cite a test `grep` cannot find — the
  most-quoted Metal fidelity number ("54% stream divergence") has no test behind it. **Fix:** delete
  `TestPrefillGate` or set the override; re-add the divergence test or stop citing it.
- **CLOSED 2026-09-13, both bugs fixed, verified by actually running the gate.** `c7f4b1e5`: added
  the `GOINFER_METAL_FAST_PREFILL_FLOOR=0` override (M-02, already shipped, happens to clear K=256
  on its own now, but the explicit override was added anyway so this doesn't silently break again
  the next time the floor default moves); fixed `decoder.PrefillGateProseFiles` pointing at two
  now-archived paths (`docs/audit-2026-09-02.md`, `docs/task-attention-decode-cost.md` — the first
  is read live and broke the test outright, the second is a provenance-only citation). Also fixed
  the dangling `TestMetalPrefillDivergenceRate` citations this finding flagged in four places —
  that test no longer exists (superseded by `TestPrefillGateVsReference`'s pooled §3.2 criteria),
  so the comments now cite the historical 54% figure's actual source
  (`docs/ollama-chase.md:623`) instead of a test grep can't find. Verified by actually running
  `TestPrefillGate/S`, not just reading the code: K=256 now completes cleanly and proceeds into
  K=1024 — did not run the full ~20+ minute K=256/1024/3900 sweep to completion, but the K=256 cell
  passing end to end is what both bugs actually blocked.

#### G-08 · The §3.2 gate never exercises `startPos > 0`, which every resident-prefix-reuse turn uses
- **Where:** `metal/prefill_gate_ref_test.go:429` (`PrefillLast(ctx, embs, 0)`) vs
  `decoder/model.go:1172` (`from`); the fused kernel's `startPos`/`uMReal` masking is covered only by
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
  paged resident). The "default ON above 512 for MoE" claim (`metal/model.go:107-108`) rests on dense
  cells. **Fix:** let it run on the paged 35B, sequential vs expert-major once M-05 exists.

#### G-10 · The C-09 status latch (`mustCmdBufOK` / `Encoder.Err()`) is, by its own comment, inert on Apple silicon — host-side pre-checks are the real gate
- **Where:** aikit `metal.go:775-748,770-775`, `metal/cmdbuf_status_test.go:19-22` (both repos'
  tests inject the error). Documentation, not a lever: every "Err() will catch it" reliance needs a
  pre-check (C-06 is the bare one).
- **CLOSED 2026-09-14 — this finding names no fix of its own.** It carries no independent **Fix:**
  or **Confidence:** line because it isn't a defect; it's the general observation behind C-06,
  which it cites as "the bare" (i.e. only) concrete instance. C-06 shipped and released this
  session (`gpu/v0.33.1`, closed above): `visionmetal` now has the same `attnThreadgroupBytes`
  host-side pre-check qwenmetal already had, checked in `newEncoder` before `Err()` would ever need
  to catch the corresponding silent-tolerance case. `aikit metal.go:770-775`'s comment (Apple
  silicon "silently tolerates … over-budget threadgroup memory … status Completed") and
  `metal/cmdbuf_status_test.go:19-22` are unchanged and still correctly describe `Err()`'s own limits —
  that is not something to fix, it is the documented reason the pre-check pattern exists at all.
  With C-06 shipped, there is no further pre-check this finding is pointing at that remains
  unbuilt; closing as a pointer resolved by its own named example, not as new work done here.

---

## 4. Minor

**Docs stale (fix with the next benchmarks pass):**
- N-01 `docs/benchmarks.md:40,502-550` — §A Metal prefill row (goinfer `82b7b8a`) predates the fused
  attention default; "8.81× at K=3900" is ≈4.3× by the tree's own L2 record; "the remaining gap is
  the attention kernel" describes the pre-fused kernel and is now false (M-03). **FIXED
  2026-09-13**: both rows now carry a SUPERSEDED callout naming why (fused kernel + M-03's GEMM
  tile) instead of a corrected ratio — no fresh same-session Ollama re-run exists yet, so none is
  invented.
- N-02 `docs/benchmarks.md:985-991` — §B3 "declines batched prefill by default … 54% stream
  divergence": default ON since 09-09; `:972` "`//go:build darwin && metal`": every file is
  `//go:build darwin`. **FIXED 2026-09-13.**
- N-03 `docs/benchmarks.md:998-1022` — the §B3 depth curve is `TestZZ_metalDepthBench`, a tight
  `ForwardArgmax` loop; `metalResident` does not implement `ResidentGreedy`, so production greedy
  runs `ForwardEmbPipe` → full head + host argmax. Labelling defect (fused argmax is a recorded
  speed-neutral on UMA), but the curve measures a path serve never takes. **FIXED 2026-09-13**: a
  caveat now names which path production actually calls.
- N-04 `docs/gpu-residency-coverage.md:19-25` — "in int8 W8A8": Metal has no int8 GEMV and
  re-quantises to W4A8 (`metal/model.go:462-461`). Qwen2.5-VL/Qwen3-VL "✅ resident" Metal cells are
  text-only (no `ForwardMRoPE` in `metal/`); Gemma 4 E2B/E4B CPU-only under a "✅" row — no footnote.
  **FIXED 2026-09-13**: all three corrected/added (the int8 claim, a new Qwen-VL text-only bullet,
  a new Gemma-4 E2B/E4B carve-out on the existing bullet).
- N-05 `docs/audit-2026-09-10.md:153` (as it stood 2026-09-13) listed C-08 open; `:1216-1224`
  recorded it fixed (d9139bc) — both now moved: C-08's whole entry, with its closure note, is
  `docs/completed/audit-2026-09-10.md:1520-1552` since the live doc's 2026-09-16 restructure
  archived every Critical/Gate/Major finding, C-08 included.
  **FIXED 2026-09-13**: the doc's own "what remains" summary line corrected (C-08 fully closed,
  G-08 partially) — the per-finding entries themselves already carried accurate closure notes.
- N-06 `docs/tasks/task-autoresearch-loop.md:3` "not started" / §3 "do NOT point it at Metal" — it ran on
  Metal (norm-class kernels); ledger tsv outside the tree (G-04). Already resolved independently
  (a further doc-review pass corrected the doc's status beyond even G-04's own scope) — no action
  needed.
- N-07 `docs/completed/task-metal-batched-verify-kernel.md:26` header says "CONFIRMED"; §Go/no-go
  is NO-GO (code honours the NO-GO). Addressed at archival (2026-09-13 doc review): the doc is now
  archived with a status block explaining the phrase; the body line itself is left as the frozen
  record.
- N-08 `metal/backend.go:439-340` ("CURRENTLY OPT-IN … pending Phase B" then "Default ON"), `:426`
  ("K=256 cell failed"), `:475-480` (`HiddenLast`: "declined by default") — N-44 of the prior audit,
  still present; `:355-356` accepts only `"1"` (N-45). `metal/spec_prefill_regression_test.go:43-51`
  same stale precondition. **FIXED 2026-09-13**: the spliced doc comment rewritten to one accurate
  paragraph (the "K=256 cell failed"/"declined by default" sub-claims were already gone); the test
  file's own stale "Metal does NOT implement ResidentPrefillKV" claim corrected too (M-01 changed
  that, but `residentPrefillSeed` is shared between both callers so no new asymmetry resulted).
- N-09 `metal/model.go:1359-1324` "greedy decode skips [softcap]" — false in production
  (`finalizeLogits` runs every executor token; Gemma 4 greedy pays a 262k `tanh` per token, ~1%).
  **FIXED 2026-09-13.**
- N-10 `metal/model.go:1977` "11 dispatches vs 19" is stale (block is 7); `metal-verdict.md:75`
  "337 dispatches/token" is 310 at this tree. `:1538-1541` "fastest greedy path" stale. **FIXED
  2026-09-13** (the two `metal/model.go` comments; `metal-verdict.md` is `docs/completed/` — an
  archived record left as-is per that directory's own convention).
- N-11 `metal/cmd/serve/main.go` said "Dense residency only … int8": MoE is resident; int8 is the
  re-quantised case. `decoder/features.go:327` cited `metal/moe.go:207-211`; it is `:375`. **FIXED
  2026-09-13** — `metal/cmd/serve/main.go:5-10` now names both corrections inline; the
  `decoder/features.go:327` citation repointed to `metal/moe.go:379-376`.
- N-12 `docs/measurements/prefill-gate-l1-ref-b-2026-09-09.md:14` names `qwen2.5-1.5b-instruct`;
  test default and L2 record say `qwen2.5-coder-1.5b-instruct` — methodology wants the exact file.
  **FIXED 2026-09-13.**
- N-13 `metal/snapshot_golden_test.go` named `attention_f32` as covered; N-28 says it is dead.
  **FIXED 2026-09-13** — the test table's inline comment (`:127`) no longer lists it.
- N-14 `metal/lora_resident_parity_test.go:114,203` — cosine floor 0.95 vs measured 0.99996 (N-52).
  **PARKED 2026-09-13, not tightened**: both sites now cite this session's own two real
  measurements (0.999969, 0.998835) and N-52's dropped-projection concern inline, but the floor
  itself is left at 0.95 — two single-machine data points a few thousandths apart is not enough to
  pick a "measured floor minus noise" number with confidence, and a wrong tight floor risks flaking
  CI on real cross-machine/quantization variance.

**Cold-path waste and small levers:**
- N-15 `metal/prefill.go:548-601` (`PrefillLast`) — 26 per-request scratch buffers built from
  `make`d, zero-filled Go slices then copied (`guF` alone 140 MB at M=3900; ≈262 MB memset + ≈262 MB
  memcpy per long prompt); `gpu.NewBufferLenOf` exists and goinfer never calls it; only `xF` needs
  zeroed pad rows. A high-water-mark cache across calls removes the allocation entirely. Tens of ms
  vs a 36 s TTFT. **INVESTIGATED, DEFERRED 2026-09-13**: real and quantified, but not safe to
  implement without a per-kernel trace first. The claim "only `xF` needs zeroed pad rows" needs
  reconciling against this SAME document's own §5 "Checked and found correct" entry — "Tails: M%8
  via `Mpad` zero rows and masks" — which treats Mpad-row zeroing as a relied-upon invariant, not
  something only `xF` needs. Reading the dispatch loop: `normF`'s writer (`pRms` at the pre-attn
  norm) runs over `M*tgReduceNorm` threads (REAL M), but its reader (the fused-QKV `pGemmStore`)
  sizes its grid off `Mpad` (`gg(qkvDim)`) — so `normF`'s M..Mpad-1 rows are read WITHOUT ever being
  written in the same call, for every buffer in the pipeline, not just `xF`. Whether that's actually
  safe (every downstream kernel either bounds-masks per-row like `attention_prefill_fused`'s
  documented tail masking, or the garbage stays confined to a padded row's own never-stored output)
  needs verifying kernel-by-kernel across all ~13 prefill MSL kernels before either the zero-fill
  removal or the cross-call cache is safe to ship — a wrong assumption here is a silent,
  garbage-in-real-output correctness bug, not a performance regression. That verification, plus the
  actual cross-call high-water-mark cache design (which buffers can safely persist across calls of
  different M, and which need re-zeroing on every call regardless), is a dedicated task on its own,
  not attempted in this sitting — same treatment as M-05/N-25's larger kernel-composition items.
- N-16 `metal/prefill.go:400-468` — the 13-kernel prefill library + 12 pipelines compile lazily inside
  the first `PrefillLast` (the first request's TTFT); `buildResident` could do it. Also N-47: a failed
  `ensurePrefill` re-panics per call (no latch). **N-47 half FIXED 2026-09-13** (see N-47's own entry,
  audit-2026-09-10.md). The eager-compile half (moving the compile from first-`PrefillLast` into
  `buildResident`) is left open: `buildResident` already pins the thread and holds its own
  autorelease pool for the whole build (M24(c)), so nesting `ensurePrefill`'s pin+pool inside it is
  probably safe (Go's `LockOSThread` is refcounted, and `NewARPool`'s own nesting discipline is
  documented as LIFO-safe elsewhere in this file) — but making it unconditional would also pay the
  compile cost at load time for configurations that will never call `PrefillLast` (fast-prefill
  disabled, or an arch/geometry `prefillOK` declines). Deciding the right gating condition and
  measuring the actual load-time-vs-first-TTFT tradeoff is a real but separate task from N-47's
  bug fix — left as follow-up work rather than attempted in the same sitting.
- N-17 (originally the unfused `gemm_w4f16` kernel, now deleted from `metal/prefill.go`) —
  `gemm_w4f16` "Removed" but still compiled; `allKernels` compiles seven kernels no pipeline uses.
  **FIXED 2026-09-13**: `gemm_w4f16` deleted outright (the `#define RPS 4` it shared with
  `gemm_w4f16_store` moved down to stay defined, see `metal/prefill.go:35`); the other six were
  verified to each back a real, still-useful micro-benchmark or recorded-negative regression test
  (`gemv_w4a8_bias`, `gemv_w4a8_sa_bk`, `gemv_w4a8_sa_qv`, `gemv_w8a8`, `rope2_kv`) except
  `gemv_w4a8_sa_amax`, which is now genuinely unreferenced anywhere — kept as-is rather than
  deleted alongside `gemm_w4f16` (documented in place instead, see `metal/kernels.go:5`) so its
  history stays visible next to the others' rather than singled out. Verified `TestPrefillGemmW4`
  (direct `gemm_w4f16_store` parity, cos=1.0) plus the full prefill suite still pass.
- N-18 `metal/prefill.go:347` — `pTile` reloaded from `pScr` per `cc` (16×/tile); subsumed by M-04.
- N-19 `metal/prefill.go:191` (`rope_f16`) — computes cos/sin per (row, pair) with no table; a
  second reason to fuse RoPE-K into `kv_store_f16`. **NOT ATTEMPTED**: a genuine kernel-fusion
  design task (write a new fused kernel, verify parity at the S-cell bar `PrefillLast`'s own batched
  path was held to), not a bug fix — the audit's own framing ("a second reason", "speculative
  lever, lower priority") already scopes it as a candidate to design and measure, not a same-sitting
  change. Left for a dedicated pass.
- N-20 `metal/model.go:390-399` + (originally the per-stage re-derivation in `metal/moe.go`, now
  deleted by the fix below) — per stage the f16 scales are re-derived from an
  f32 heap copy that is 2× the bytes the GPU consumes (≈2.85 GB on the 26B, ≈4 GB on the 35B, on
  the box whose N=128 cliff was memory pressure); cache f16 per expert at build.
  **FIXED 2026-09-13**: added `int4DirectBytesOnly` (`metal/model.go`) — `int4DirectBytes` minus
  the f32→f16 conversion, for the bytes half of every stage — and precompute each expert's gate‖up
  and down f16 scales ONCE in `buildMoELayer`/`buildGemma4MoELayer` (both the byte-copy `stage` fn
  and the `stagePread` fn now index into that build-time cache instead of re-deriving). Verified via
  the existing real-fixture bit-exact gates, which cycle eviction/re-staging of the SAME experts
  repeatedly and would surface a stale/misindexed cache as a logit mismatch:
  `TestMoEPaging_matchesNonPaged` (qwen3_5_moe-tiny, slots=2/3, 32 tokens, exact match),
  `TestMoEPagingPread_matchesByteCopy` (same fixture, both stage paths, 142+85 pread-served stages),
  `TestGemma4Paging_bitExact` (gemma4-moe-tiny, N=3/4 slots, 0 mismatches over 8 positions).
  Confirmed these are real discriminators, not just coincidentally green: injecting a deliberate
  cache-indexing bug (always read expert 0's scales) into `moe.go`'s `stage` fn made
  `TestMoEPaging_matchesNonPaged` and `TestMoEPagingPread_matchesByteCopy`'s byte-copy subtests fail
  immediately (its `pread` subtests, untouched by that specific bug, stayed green — confirming the
  test isolates which stage path broke) — then reverted to the real fix. Gap: no `.giw`-format
  gemma4-moe-tiny fixture exists to drive gemma4's `stagePread` path through this same bit-exact
  gate (only its byte-copy `stage` fn is covered end-to-end); the `stagePread` fix there is the
  identical mechanical pattern already proven correct in `moe.go`'s pread path, not independently
  measured on gemma4. `go test ./metal/` (91 pass) and `-tags goinfer_testhooks` (143 pass) both
  0 fail; gofmt/go vet/staticcheck clean.
- N-21 `metal/expertpool.go:198-201` — each slot built via `NewBufferUint32s(d, make([]uint32, n))`:
  ≈4.5 GB of transient Go allocation at N=64 on the 35B to zero-initialise; `NewBufferBytes(n)`.
  **FIXED 2026-09-13** — used `gpu.NewBufferLenOf[T]` instead (the exact generic, right-sized,
  uninitialized allocator the finding names; `NewBufferBytes` alone would have mis-sized `.n` for a
  `.U32s()`/`.U16s()` view). Every slot's contents are staged before first read, so nothing needs
  the zero-fill this removes. Verified: `TestExpertPool_lruAndStaging`,
  `TestExpertPoolBatch_matchesSequential/_pread`, and the real paged-forward parity tests
  (`TestGemma4Paging_bitExact`, `TestMoEPaging_matchesNonPaged`) all still pass — if uninitialized
  memory leaked through anywhere, the staged-content checks in these would have caught it.
- N-22 `metal/moe.go:854-790`, `metal/gemma4_moe.go:581-539` — phase 2 of layer l and phase 1 of l+1 have no
  host dependency and could share one command buffer (2L+1 → L+1); superseded by M-11.
- N-23 `metal/moe.go:796` — a hybrid's dense layers each get their own `Begin/End` in
  `forwardLogitsMoEPaged`. **FIXED 2026-09-13**: consecutive dense layers now share ONE command
  buffer (`Begin()` on first use, closed only when the next MoE-paged layer's router readback needs
  a real value-dependent seam, or at the loop's end) instead of a submit+wait per dense layer —
  mirroring `encodeTrunkInto`'s own all-layers-in-one pattern for the pure-dense path, which calls
  the SAME `encodeLayer` function repeatedly on one encoder and is proven correct by the entire
  dense-decode test suite. Verified via the existing real-fixture regression gates
  (`TestMoEPaging_matchesNonPaged`, `TestMoEPagingPread_matchesByteCopy`, both still exact-match) —
  though neither fixture (`qwen3_5_moe-tiny`, `decoder_sparse_step:1`/`mlp_only_layers:[]`) has any
  dense-only layers, so they only prove the all-MoE case is unaffected, not the actual dense-batching
  path this fix adds. No fixture in this tree currently drives a mixed dense+paged-MoE model through
  `metal/moe.go`'s (not `gemma4_moe.go`'s) generic paged path (the two configs with non-empty
  `mlp_only_layers`, `laguna-xs21-tiny`/`laguna-m1-tiny`, are a different architecture family not
  wired into the Metal backend at all) — confidence in the mixed case rests on code reuse
  (`encodeLayer`'s correctness under repeated same-encoder dispatch), not a direct measurement.
  `gemma4_moe.go`'s own `forwardLogitsPaged` has the identical per-dense-layer pattern but was left
  untouched: it carries `GOINFER_MOE_PROF_SPLIT` profiling instrumentation that accumulates
  PER-DENSE-LAYER timing (`denseWallNanos`/`denseGpuNanos`), which batching would silently break —
  a separate, more delicate change than this finding's own citation scoped for. `go test ./metal/`
  (93 pass) and `-tags goinfer_testhooks` (145 pass) both 0 fail; gofmt/go vet/staticcheck clean.
- N-24 `metal/moe.go:30-37,622` — f32 router weight: 84 MB/token on the 35B (deliberate, ≤0.4 ms).
  `moe_route` on one GPU thread (deliberate, value-independent dispatch; ~10% of a fitting ~5 ms
  MoE token).
- N-25 `metal/backend.go:614-563` — `HiddenLast` is one synchronous command buffer per position
  (≈K × 13–18 ms; ~7–9 s for 512 tokens) where the batched trunk would take ~1.8 s; the stated
  rationale ("declined by default") is stale. Fix is `PrefillLast` minus its last two dispatches.
  **PARTIALLY CLOSED 2026-09-13**: the stale rationale was real — `HiddenLast`'s doc comment said
  Metal's batched PrefillLast was "declined by default" for generation, but `metalFastPrefillEnabled`
  has defaulted true since M-01/M-02 (§3.2 gate passed 2026-09-09, earlier in this same audit round)
  — the fidelity bar that would justify keeping the sequential loop for embeddings is already
  accepted for decode's own output. Rewrote `metal/backend.go`'s `HiddenLast` doc comment to say so.
  The batched-HiddenLast implementation itself (a real, scoped lever — PrefillLast's dispatch graph
  minus the LM head + softcap dispatches) is its own parity-gated engineering task (needs the same
  kind of S-cell tolerance gate PrefillLast passed, verified against the current sequential
  `HiddenLast` as the oracle) — left as follow-up work, not attempted same-sitting, similar to
  M-05/M-15's treatment.
- N-26 `metal/backend.go:675-581` — `ForwardN` is a per-token loop allocating 608 KB per row; cold
  (Theta ≈ 1.02 declines speculation, P21) but the interface doc's "K tokens in ONE command
  buffer" is not what Metal does. **FIXED 2026-09-13** (doc-only, `decoder/residency.go`'s
  `ForwardN` interface comment): rewrote the promise from a universal "ONE command buffer" to an
  amortization OPPORTUNITY some backends take and others decline — CUDA's `prefillReady` path
  batches into one weight-stationary pass where the arch allows it (falling back to sequential
  otherwise, e.g. MoE/DeltaNet), Metal's `ForwardN` is always the per-token sequential loop. The
  bit-identity contract (`TestResidentForwardN_parity`) is unchanged and still universal; only the
  structural claim was wrong.
- N-27 `metal/model.go:1353` — pipe path memcpys 608 KB into `logitsHost` before the ack; the
  zero-copy `r.logits.Floats()` view could be returned (the contract already says "consume before
  the next call"). ≈30–60 µs/token. **INVESTIGATED, DECLINED 2026-09-13**: this is not a free win —
  `decoder/spec_optfwd.go:214-216` documents the exact failure mode, MEASURED on CUDA before its own
  fix: `cuda/resident.go` used to return a zero-copy alias of its reusable host buffer ("a per-call
  slice" is explicitly NOT what a zero-copy view gives you), and overlapping a speculative second
  `Forward` call with `SampleWithInfo` reading the FIRST call's result raced a DMA write against the
  read — same token id, but logprob -2.6463 vs -2.6266, and under `-race` (timing-shifted) the
  emitted token stream diverged outright. **`go test -race` cannot see this class of bug at all** —
  the corrupting write is a driver/GPU write into shared memory, not a Go-visible memory access, so
  the detector is structurally blind to it; the only way this was ever found was by measuring
  logprobs/tokens directly. `spec_optfwd.go`'s own comment credits Metal's CURRENT copy-based
  `Forward` with being the reason its optimistic-forward feature "verified clean" on Metal without
  needing backend-specific handling — i.e., an existing, working piece of code's reasoning already
  depends on the exact property N-27 proposes removing. The `gate.scratch` copy there is backend-
  agnostic (it copies regardless of which backend it's talking to) so it would still protect THAT
  one call site either way — but making Metal's `Forward` alias a live GPU-written buffer would put
  every OTHER present and future `ResidentForward.Forward` consumer that does not carry its own
  defensive copy back into the same invisible-to-`-race` hazard class CUDA already paid for once.
  Removing a 30-60 µs/token memcpy is not worth reopening a bug this hard to detect without a full,
  deliberate audit of every consumer for concurrent double-calls — parked, not fixed.
- N-28 `metal/model.go:1412-1380,1419-1431` — at temperature > 0.2 (serve default 1.0) nothing
  overlaps the CPU sampler with the GPU: the full-vocab softmax/filter sits in the GPU-idle gap
  (order 0.5–1.5 ms of ~13.6 ms; the served 54 vs decode-only 73.6 tok/s is where it shows). The
  0.2 cap is a measured decision; the only structural way past it is device-side sampling.
- N-29 `metal/model.go:379-381,493-503` — load path rebuilds every word byte-by-byte from nibbles
  that are already the word bytes; `int4Concat` grows by `append` (two reallocs per fused tensor);
  scales through the serial loop. ≈0.5 s at 1.5B. **FIXED 2026-09-13**: `bytesToU32` now does one
  bulk `copy` into the freshly-allocated (always 4-aligned) result's own `[]byte` view instead of a
  per-element shift-and-mask loop — the source bytes are already the target little-endian word
  bytes, nothing to reconstruct arithmetically (the same technique `copyBytesToU32Buf` already uses
  for the paged hot path). `int4Concat` now pre-sizes `words`/`scales` from each weight's known
  `Rows()*Cols()` shape (the same `N*K/8` words, `N*K/32` scales formula `int4Buf`'s own int8-
  fallback branch already allocates by) instead of growing two nil slices by repeated `append`.
  New `TestBytesToU32_matchesManualLE` (independent oracle via `encoding/binary`, not a re-derivation
  of `bytesToU32`'s own shift expression) confirmed red without the fix — reverted to a deliberately
  wrong byte order (big-endian) and the test caught every mismatched word — then green with it.
  `int4Concat`'s existing `TestInt4Pack_declinesNonMultipleOf32K` panic-guard still passes, and the
  broad real-fixture parity/cosine suite (which every non-paged Metal build routes through both
  functions to reach) is unchanged. This is a one-time load-path cost (`int4DirectWords`'s non-paged
  build path), not a per-token tax — lower priority than N-20/N-21's per-stage fixes, but the audit's
  own ≈0.5s-at-1.5B estimate is real. `go test ./metal/` (92 pass) and `-tags goinfer_testhooks`
  (144 pass) both 0 fail; gofmt/go vet/staticcheck clean.
- N-30 `metal/kernels.go:751-777` — `swiglu_quant` evaluates `glu_act_pinned` twice per element;
  <1%. `:276` "K<=1536" stale. `:563-567` rope2_kv recorded null (0.6%) — not re-proposed.
  **PARTIALLY CLOSED 2026-09-13**: the stale "K<=1536" comment was real and wrong, not just
  outdated phrasing — the actual bound today is the M-11 threadgroup-memory guard
  (`maxThreadgroupStageBytes`/`d.MaxThreadgroupMemoryLength()` at buildResident, ~32 KiB on Apple
  GPUs), roughly 10x more permissive than 1536 words. Fixed the comment at
  `metal/kernels.go:287-290` and the three other places quoting the same stale figure
  (`metal/model.go:174,190,1877-1878` — the GPT-2 XL claim "1600 would exceed it" was flatly wrong
  under today's guard, since 1600 is far below the actual ~16384-word cap). The `swiglu_quant`
  double-eval itself is left as a recorded negative: eliminating it needs a device-memory scratch
  buffer to cache pass 1's activated values for pass 2 (threadgroup memory can't hold it — Mixtral's
  intermediate dim alone is 14336 elements, 57 KB, over the ~32 KiB budget), which trades a cheap
  transcendental recompute for extra device bandwidth on a kernel that is itself
  memory/bandwidth-bound — not a clear win, and the audit's own number (<1%) doesn't justify the
  added complexity without measuring first. rope2_kv (0.6%) is unchanged, as the finding itself
  says.
- N-31 `metal.go:744-747` — `WaitDone` reads four GPU timestamps per production token for
  `LastGPUTimes` (tests only); µs.
- N-32 `metal_vit.go:576-578` — "64×64 tile" stale (32×32). `:665-666` — the ViT library compiles
  fast-math OFF library-wide for one exact divide; `precise::divide` per op would free the rest.
  **PARTIALLY CLOSED 2026-09-13** (aikit `3214193`): fixed the stale comment — `SGBigBlock` is 32,
  and `GEMMF32Plan`'s own comment two lines below already correctly said "32×32 tile", so the fixed
  comment was contradicting its neighbor as well as the code. Comment-only; `TestMetal_gemmF32SG` /
  `TestMetal_vitGEMMs` / `TestMetal_gemmTiledMatchesUntiled` still pass. The fast-math half (a
  library-wide compile flag affecting every ViT kernel's numerics, not a comment) is left open —
  it needs its own measurement pass and, since it lives in aikit, a release decision this session
  is not making unilaterally (same reasoning as M-14/N-31).
- N-33 `metal/expertpool.go:64-54` — `copyBytesToU32Buf` duplicates `gpu.Upload` minus its bounds check.
  **FIXED 2026-09-13** — `copyBytesToU32Buf` now calls `gpu.Upload` (a signature-compatible drop-in:
  `metal.Buffer` is a type alias for `gpu.Buffer`), panicking on its error since every call site's
  destination is sized for exactly that source by the pool's own construction — a failure there is an
  invariant violation, not a runtime condition to recover from. New
  `TestCopyBytesToU32Buf_oversizedSrcPanics` confirmed red without the fix (the old `unsafe.Slice` +
  `copy` reinterpret silently truncated an oversized source instead of erroring) and green with it;
  `TestCopyBytesToU32Buf_exactFitStillWorks` pins the ordinary in-bounds case still round-trips
  correctly. Full `metal/` suite (91 pass, 0 fail) and `-tags goinfer_testhooks` (143 pass, 0 fail)
  still green.
- N-34 `gpu/metal_copy.go`, `metal_upload_batch.go` — unused by goinfer (correct on UMA); note they
  are host-side and unfenced, so a `CopyDevice` during an in-flight command buffer would race.
- N-35 `decoder/model.go:1133-1091,1134` — `warnPrefillDeclined` is process-lifetime `sync.Once`; on
  Metal the first sub-floor prompt consumes it, so a later real decline (cap, OOM) is silent (N-49).
  **FIXED 2026-09-13**: replaced the single `sync.Once` with a mutex-guarded set keyed on the
  decline reason with its numbers normalized out (`decoder/model.go:1096,1053-1073`) — every
  below-floor prompt has a different `promptLen` in its message but normalizes to the same key, so
  the routine Metal case still logs once, while a later, differently-worded decline (a resident-cap
  refusal, an OOM) now gets its own one-time line instead of being silenced by the first. The old
  `TestWarnPrefillDeclined_FiresOncePerProcess` explicitly pinned the coarser sync.Once behaviour
  "so a future change to it is a deliberate decision rather than a silent drift" — this is that
  decision; replaced with `TestWarnPrefillDeclined_FiresOncePerReason`, confirmed red without the
  fix (a differently-worded second decline was silenced) and green with it. Triggered a
  `TestParityManifest_fresh` re-stale on `core` (this file is on that dependency list); resolved via
  `scripts/refresh_parity_hashes.sh` (37 forward goldens green at this commit, 0 failed) — a
  non-numeric diagnostic-logging change, not a forward-numerics one.
- N-36 `metal/backend.go:289-202` — `residentKVBytes` charges KV for DeltaNet layers that allocate
  none. **FIXED 2026-09-13** — skips a layer when `Qwen35ResidentParams`'s `ok` and
  `Qwen35LinearLayer(l)` are both true, the SAME chokepoint `metal/model.go`'s own layer-build loop
  uses to decide "does this layer get a DeltaNet mixer" (not a second, potentially-disagreeing
  predicate). New test `TestResidentKVBytes_excludesDeltaNetLayers` on the real 3:1 hybrid
  `testdata/qwen35-tiny` fixture — confirmed red without the fix (charged 4× the correct KV bytes,
  one for every layer including the 3 that allocate none).
- N-37 `metal/gemv_w4a8_coal_bench_test.go:29-30` — the GEMV micro-bench shape (512×4096, 1 MB) is
  cache-resident: a "micro-win" here is an issue/latency result, not a bytes result (the production
  down-proj reads 7–14 MB per dispatch from DRAM). `metal/profile_test.go:78` per-kernel µs are
  warm-cache for the same reason.
- N-38 `metal/prefill_ttft_test.go:81` — the first `PrefillLast` (P=256) includes the one-time compile;
  the L2 record's P=256 row carries it in both arms.
- N-39 `internal/serveapp/openai.go:1261-1086` — comment says adapter requests "drop to the staged
  path"; since G3 they reach the resident path on a `prefillFrom == 0` turn. Later-turn behaviour
  (`decoder/session.go`) not in tree. **FIXED 2026-09-13** — rewrote the three comments describing
  adapter routing (`internal/serveapp/openai.go:1261-1093,735-739,826-829`) to say what  `decoder/model.go:1312`'s actual chokepoint (`useGPU := m.resident != nil && prefillFrom == 0 &&
  (commit == nil || (lora != nil && resAdapter != nil))`) does: a session's FIRST turn
  (`prefillFrom==0`) with a bound resident adapter reaches the resident GPU path; a later turn on
  the same session (`prefillFrom>0`, continuing off the reused warm prefix) still drops to CPU,
  because nothing wires compute-time LoRA into the resident prefix-reuse path yet — that gap is
  real and stays open, only the comment's blanket "it drops to the staged path" claim was wrong.
  Comment-only.
- N-40 `docs/benchmarks.md:975` — §B3 "4-bit both sides": the tied LM head (24% of per-token bytes)
  runs int8 by a deliberate fidelity pin; up to ~117 MB/token (≈1.4 ms) of the 1.5B decode deficit
  is a chosen precision trade, not kernel quality. Labelling, not a defect.
- **N-41** (found 2026-09-14, incidental to the aikit v1.42.0/`gpu/v0.33.1` bump's verification
  run) `decoder/fitguard.go` (unpinned-load auto-context sizing) vs `metal/model.go` (the backend's
  hard 4096-position resident-context ceiling, "a fixed-size kernel score buffer, not a tunable
  default"): an unpinned `decoder.Load` on Metal picks a context from currently-free host RAM with
  no awareness of this backend-specific hard cap, so three real-device tests
  (`TestEncodeAhead`, `TestPrefillNoNaN`, `TestPrefillParity`) fail with "resident context N
  positions exceeds this backend's hard ceiling of 4096" whenever the full heavy `./metal/...`
  suite runs back-to-back (cumulative memory pressure from earlier tests lowers free RAM at the
  moment these three call `decoder.Load`, and the auto-sizer picks a context 3-7x over the
  ceiling — measured 12109-29666 positions across several runs on this machine). Confirmed
  environment-dependent, not a regression from this session's other work: all three PASS reliably
  in isolation (`-run` scoped to just one of them, ~5 GB free) and FAIL reliably as part of the
  full suite, reproducing identically with and without this session's aikit-bump/G-06 changes
  stashed out. **Fix (not attempted, out of scope for this pass):** either the fit-guard needs to
  know Metal's hard ceiling and clamp to it (the same shape `smallerFittingContext` already uses
  for the KV-budget case), or these three tests need to pin an explicit `-ctx` rather than rely on
  auto-sizing. **Confidence:** confirmed (reproduced 4x total: 2 full-suite runs fail, 2 isolated
  runs pass, across two different memory states).

---

## 5. Checked and found correct / at ceiling

- **Command-buffer and sync structure (decode):** one command buffer, one commit, one
  `waitUntilCompleted`, one serial encoder, 310 dispatches, ~1,030 purego transitions, one heap
  allocation per token; no per-token buffer creation; `setBuffers` batching already cut binds from
  ~2,600 to ~337 msgSends/token; ICB and unretained references are recorded nulls
  (`metal/model.go:1422-1419`, `metal.go:732-812`, `metal-verdict.md:171-172`).
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

1. **M-02 + M-01 + G-01** — **CLOSED 2026-09-13** (`6cc862a0`). Floor → 256; `ForwardNoLogits`
   shipped synchronous rather than as a `noHead` executor job (see M-01's own closure note for the
   scoped-down fix and the follow-up that stays open); the red test made green. Not yet done: the
   full `noHead`-executor-job version (~0.9 ms/token of unclaimed encode-ahead overlap) and the
   TTFT ladder re-run at K∈{64,128,256,512} this item called for.
2. **M-03** — **CLOSED 2026-09-13** (`f6c222ee`). 32×32 per-simdgroup block, all-lane dequant, A
   staged per threadgroup, exactly as scoped. §3.2 pooled gate SHIPS (see M-03's own closure note);
   the P=256 TTFT re-measurement this item called for has not been run.
3. **M-04** — **CLOSED 2026-09-13, narrower than scoped** (`c660ab78`). 32-key tile shipped; the
   register-O + diagonal-α rewrite and GQA grouping this item also named were deliberately left as
   a separate follow-up (see M-04's own closure note for why). §3.2 re-run: SHIPS.
4. **M-06 + G-03 + C-01** — **CLOSED 2026-09-13** (`84c3f29f`), all three exactly as scoped: Gemma 3
   onto the batched path, the fixture that can see it, the cache rounding.
5. **M-07** — **PARTIALLY CLOSED 2026-09-13** (`47355617`, see its own closure note): the row4-skip
   half shipped; releasing host projections after upload (the RSS-on-1.5B/D7-fit gate this item
   named) is not done.
6. **M-08** — **CLOSED 2026-09-13** (`a30f2cd3`), kept fused rather than split (Metal-specific
   tradeoff, see its own closure note). Still no Mac adapter-decode measurement recorded.
7. **Paged:** **M-14 PARTIALLY CLOSED 2026-09-13** (aikit selector shipped, `5065f24`; goinfer
   wiring blocked on a `gpu/vX.Y.Z` release goinfer's go.mod can pin — the wiring itself is fully
   designed, see M-14's own entry) → **M-13 CLOSED 2026-09-13** (auto-sized slots shipped; the
   M35/M26/G20 rows re-run against the pager this item called for has NOT been run — that needs the
   actual checkpoints, which this pass did not have) → M-11 + G-05 (the shared-event re-run on the
   paged shape) → **M-12 CLOSED 2026-09-13, cross-expert half only** (concurrent staging across a
   layer's routed top-k shipped and verified correct end-to-end under `-race`; the finding's own
   3-concurrent-preads-per-expert half not done; the actual wall-clock win unmeasured — needs real
   26B/35B cold-read latency this pass had no checkpoint for). **M-11 unchanged — still fully
   open.**
8. **Probes:** both CLOSED 2026-09-13, NEGATIVE (see their own entries above). M-09 — the staged
   K-read probe measured 2.3x SLOWER than shipped, not faster; not ported. M-10 — built and A/B'd
   on the depth bench, slower at every depth; reverted.
9. **Docs and gates batch:** of N-01…N-14, G-02, G-04, G-07, G-08, G-09 — **G-02, G-04 (partially),
   G-07, G-08 CLOSED 2026-09-13** (see their own entries); **N-01 through N-14 all closed or parked
   2026-09-13** (see each entry's own note; N-14 deliberately parked, not tightened). **G-09 still
   open** — it names its own precondition ("once M-05 exists"), which is unmet. The §A Metal row
   re-measurement this item called for (after 1–3, same protocol, Ollama same session) has still
   not been run — N-01's fix names why (no ratio invented without it) rather than skip the gap.
10. **M-15** (cross-repo, aikit first): the three Metal tower shapes; then `EnableResident` on Metal.
    Unchanged — still open.
11. **M-16** — **CLOSED 2026-09-13, NEGATIVE** (aikit `a2dffd6`, see its own entry). Untracked
    buffers measured noise-level (±0.4–1.5%) against tracked across four interleaved runs; not a
    lever, not re-proposed.

Not proposed, because the record already closed them: split-KV / dedup attention variants, Stage-B
GEMV, ICB, unretained references, megakernel, dispatch-count fusions, rope2+kv_store, sa_qv,
batched small-M verify, fused argmax, `WILLNEED`, sampler overlap above temperature 0.2.
