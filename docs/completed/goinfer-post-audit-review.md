# goinfer post-audit review — 2026-08-07

> **ARCHIVED — a record, not instructions.** This file is closed work kept for its reasoning and
> its numbers. Checkboxes record the state at the moment it was archived: an unticked box means
> "not ticked when this closed", **not** "still to do", and nothing in `docs/completed/` is
> actionable. If you need a task, use the live docs; if something here reads as an instruction to
> a future reader, it was missed at archival — see the doc-closeout rule in
> `docs/parity-coverage-policy.md`, and move it to live policy or strike it.


**Scope:** the whole repo at the current tree (working copy synced from the Mac today; HEAD `53c67eb`), reviewed against the closed `docs/completed/audit-2026-08-05.md`. The question asked: do the audit fixes hold, and are there remaining bugs?

**Method:** six parallel subsystem reviews (serve/HTTP, decoder state & concurrency, loaders/tokenizer/constrain/multimodal, gpu/, cuda/, metal/), each instructed to verify the audit's fix claims in code and then hunt beyond them. Every major finding below was then re-verified directly by the lead pass (file:line, mechanism traced end-to-end); minors were verified by the subsystem pass and spot-checked. Tooling ran under a real **go1.26.5** toolchain (built from source in the sandbox): `go build` clean for root / gpu / cuda / metal(darwin-arm64 cross) incl. `-tags gpu` serve; `go vet` clean for all four modules (one exception → R-12); `gofmt -l` clean; **test suites green** for root, gpu, cuda (the only failures are the two tests needing the 5.5 GB `testdata/` that wasn't copied to the sandbox). No GPU device here, so nothing device-side was executed — device-dependent findings are static, and say so.

**Verdict:** the audit work broadly holds — the large majority of the 78 dispositions checked out exactly as described, including the hairy ones (C-01/02/03 recurrent+spec guards, C-08 setupErr, C-24/25/26/27/28, C-09 execErr ordering, C-11, C-14, C-15 bit-for-bit, M-01..M-08, M-10..M-18, B-08..B-14). What remains is: **8 confirmed findings that meet the audit's own critical bar** (silent wrong output, process death, or a gate that cannot fail) — most of them residuals or regressions of closed findings — plus ~20 confirmed minors and a short unconfirmed-leads list.

**Box verification — the R-01…R-30 fixes are device-verified (2026-08-07).** The Mac ran metal + gpu(webgpu) + the pure-Go modules on-device; the cuda module (code-complete but unbuilt on the Mac) is now verified on the Linux RTX-2070 box:
- **cuda** `-tags cuda` build + vet + full test suite green — **incl. `-tags 'cuda goinfer_testhooks'`** (22.2 s, 0 fail / 0 race), which is what actually exercises **R-03**'s ordered `g4x2` clear: all g4moe parity gates (`TestGemma4{MoE_localize,Router_residentIdxParity,DenseScaled_residentParity,_perExpertScaleFold}`, `TestMoEResidentParity`, `TestGemma4Graphs_bitExact_*`) pass bit-exact — no regression from the added `Sync`. R-20 `TestRunJob_recoversPanic` ✓, R-25 `TestSlotBytesPerLayer_outOfRange` ✓.
- **R-04** FMA lint green (`TestKernelFMALint` + `_coversEmbeddedPTX`); regenerating `router_f32.ptx` at the box's NVRTC 12.9.86 (its recorded version) is **byte-identical** (sha unchanged) — `__fmaf_rn` lowers to the `fma.rn.f32` NVRTC already chose, so no numeric change and nothing to commit.
- **R-12** confirmed on Linux: with the `//go:build darwin` tag now on `snapshot_golden_test.go`, native-GOOS `go vet` over the metal module matches **no packages** (no `undefined: resident`).
- **R-05** verified on the box: real **DeepSeek-V2-Lite** (`~/models/deepseek-v2-lite-gguf`, qk head dim = 128+64 = **192**) loaded on the webgpu backend goes **fully GPU-resident** (`ResidentActive()` true) and decodes on the resident runner — the head-dim guard no longer rejects MLA's >128 qk head. New regression gate `gpu/mla_real_resident_test.go::TestMLAResidency_realDeepSeek_R05` (asserts resident, or — for a card too small to fit the 16B — a non-`head_dim` decline reason; a `head_dim` decline is the R-05 regression).

---

## Confirmed findings — critical bar (ranked)

### R-01 · Adapter requests are served with base weights when the base is resident
`internal/serveapp/openai.go:749` (drive's resident fast path) · **silent wrong output** · residual of the C-29/#7 multi-LoRA feature area

Compute-time LoRA is applied only via the session binding: `sessionLRU.acquire` → `Session.UseAdapter` → `cache.lora` (sessions.go:73, decoder/session.go:54), and the forward reads it per layer at `decoder/model.go:515`. But `drive()` branches on `lm.model.ResidentActive()` **before** touching sessions and calls the stateless `lm.model.Generate(...)` / `GenerateNgramSpeculativeAdaptive(...)`, whose fresh cache has `lora == nil`. There is no `lm.adapter` check in that branch, and nothing rejects the combination at startup — `loadAdapters` rejects only `--stream-weights`, and `LoadAdapter` (decoder/lora.go:260) rejects gguf/MoE/gemma4/non-gated but not a resident base. The dense safetensors class it accepts is exactly the resident-eligible class, and `loadAdapters`' own comment says the adapter "shares the base's **resident** decoder.Model".

Failure: `serve --backend webgpu|cuda|metal` with a resident dense base + `--adapter fine=base=dir` → every request with `model:"fine"` returns **base-model output, HTTP 200**, on all endpoints. The branch comment's premise ("sessions are purely a TTFT optimisation, not correctness") is false for adapter models — the session is the correctness carrier.

Fix shape: in `drive()`, route `lm.adapter != ""` down the session path (correct, slower), or refuse resident+adapter at startup like `--stream-weights`; longer-term, plumb the adapter into stateless Generate.

### R-02 · Metal paged gemma-4: pread staging panics at decode time → process death
`metal/gemma4_moe.go:333,336` · **process kill on the flagship path** · residual of B-10

The `stagePread` closure panics on any `preadIntoU32Buf` error. It runs at **forward** time — `expertPool.ensureResident` (expertpool.go:153) ← `forwardLogitsPaged` ← adapter `Forward` — and pread staging is **default-on** for paged runs (`GOINFER_MOE_PREAD != "0"`, gemma4_moe.go:223). The only recovers in the module are build-scoped (backend.go:52, model.go:356); the generation goroutine has none. B-10's site list included this line, but the fix (recover in `buildResident`) can never cover a closure that only executes at decode.

Failure: paged 26B decode (the documented 16 GB MacBook config) hits a transient read error on the `.giw` — external-volume hiccup, file replaced/truncated under a running serve — and the whole server dies mid-token instead of erroring that request.

Fix shape: return an error from the staging path (plumb through `ensureResident`), or add a recover→`execErr` boundary around the paged forward.

### R-03 · CUDA gemma-4 MoE: accumulator zero-fill is unordered against the previous layer's still-running kernels
`cuda/resident.go:1296` (`gpu.Upload(r.g4x2, r.g4zero)` in launchToken) · **silently wrong logits, timing-dependent** · audit-missed

`r.stream` is non-blocking; aikit's `Upload` runs the copy on the context (legacy null stream) and does a full sync only **after** enqueuing — aikit's own doc-comment (gpu/cuda.go:308-321) spells out that these two streams have "no ordering guarantee at all" (that sync fixes upload→next-launch, not pending-launch→upload). At the moment layer *l*'s zero-fill is issued, segC(*l−1*) — the kernels that **write and then read `g4x2`** (fMoEWacc accumulate, the fRes join) — may still be executing on `r.stream`. If the DMA lands mid-segC, layer *l−1*'s expert contribution is partially or wholly zeroed before the join consumes it.

Per the CUDA model this is a straight data race on every g4moe layer after the first, every token, graphs on or off. Your RTX box's parity gates pass, so current hardware/timing masks it (the window is small: segC(l−1) launched a few ops earlier), but nothing in the model guarantees that — and it's exactly the "intermittent, clustered, hides behind small fixtures" failure the aikit comment describes for the sibling race it fixed.

Fix shape: `r.stream.Sync()` before the Upload (cheap — Upload already full-syncs after), or better, enqueue the clear as `cuMemsetD32Async` on `r.stream` so it's ordered and graph-capturable.

### R-04 · The C-16 FMA lint cannot see the very lines it was added for
`cuda/kernel_fma_lint_test.go:56-58` + `cuda/router_f32.cu:26,59` · **gate that cannot fail** · defective fix of C-16

The lint's `isDeclOrIndex` pre-filter includes `\bfor\s*\(` — any line containing a for-header is skipped before the MAC regexes run. `router_f32.cu`'s only two float MACs are written **on** for-header lines (`for (int k = t; k < K; k += nt) acc += wr[k] * a[k];` and the rmsnorm `ss += src[k]*src[k]`), so the test passes vacuously. The disposition's "it passes, so nothing was actually wrong with the kernel" is the filter talking: the committed `router_f32.ptx` shows NVRTC **chose** contraction (10× `fma.rn.f32`), i.e. codegen on the file the repo calls "the one DISCRETE-failure path" is unpinned. Numerically fine today; the next `build_ptx.sh` regen with a different NVRTC may emit mul+add and shift router logits ~1 ULP — expert flips near top-k ties — with the gate green. Class-wide: any future MAC on a for-header line is invisible to the lint.

Fix shape: don't skip the whole line for `for (` — strip the header up to the closing paren and lint the body remainder (or split single-line for-bodies before filtering). Then either add `__fmaf_rn` to router_f32.cu and regen, or whitelist with a reason.

### R-05 · The M-12 fix regressed MLA: every DeepSeek/Kimi model is silently declined from webgpu residency
`gpu/decoderunner.go:255` · **feature-killing regression** (output stays correct via CPU fallback) · regression of M-12

`attnHeadDimSupported(hd, m.layers)` is the first, unconditional check in `newDecodeRunner`. DeepSeek arches set `HeadDim = qk_nope + qk_rope` (decoder/registry.go:1267) = **192** for real V2-Lite/V3/Kimi — but the MLA plan (`m.mla != nil`, decoderunner.go:285) never dispatches the 128-wide GQA attention kernels the guard protects; it uses the mlaAttn family with its own rank-bounded accumulator. Net: `BuildResident` errors, `withResidency` silently falls back, and the entire MLA residency lever (gpu/mla.go) is dead for every real checkpoint. The gates stay green because `mla_test.go:355` passes `vHead` (=32) as hd instead of the qk head dim. (Related residual, R-24: the resident MLA path also lacks the `kvLoRARank ≤ 1024` bound that `Context.MLAAttn` enforces — worth adding when un-regressing this.)

Fix shape: exempt `m.mla != nil` from the GQA head-dim guard (and add the rank bound to the resident admission); fix the test to use real qk geometry.

### R-06 · gpu staged-path constructors discard allocation errors → nil-deref panic on VRAM exhaustion
`gpu/gemv.go:339-341` (NewGEMVRunner: `asBuf, _` / `dstBuf, _` / `stag, _`), `gpu/gemv.go:242,255-256` (BatchGEMV `dims, _` + unconditional `dims.Release()`), `gpu/gemm.go:150,157-158` (BatchTiled, same) · **process kill** · same class as closed C-27, in the production staged path

On a failed `CreateBuffer` the nil buffer flows into `CreateBindGroup`, and the error-cleanup path then calls `Release()` on the nil buffers — nil-pointer panic. These are reached from `webgpuBackend.MatmulW8A8`/`MatmulW8A8Batch` (backend.go:117,206,208) — the staged decode/prefill fallback whose contract is "return false on GPU error, fall back to CPU". C-27 fixed exactly this pattern in the fused/batched entry points and `newDecodeRunner`; these three neighbors kept the old shape (note `aBuf` and `dimsBuf` in the same function *do* check).

Fix shape: port the same buildErr/checked-alloc pattern; nil-guard the cleanup.

### R-07 · `.giw` shape validation still misses the gemma-4 MoE sub-block, PLE, and all bias/norm vectors
`decoder/serialize.go:302-382` (validateShapes) vs reader `:893-935` · **crafted/corrupt file → silent wrong routing or decode-goroutine panic** · residual of C-06

The C-06 disposition says the per-layer accessors "cover gemma-4 too", but that covers the standard projections only. For a gemma-4 MoE layer the standard `Router`/`Experts` are empty (writer comment, serialize.go:622-624) and the real weights live in `l.gemma4moe` — `routerProj`, `expertsGateUp[]`/`expertsDown[]` (count `ne` taken from the blob, only an "implausible" sanity bound at :926), `routerScale`, `perExpertScale` — plus `PLEGate`/`PLEProj`. None are cross-checked against the arch. Forward consequence (forward_gemma4_moe.go:81): `matmul(&mo.routerProj, …)` writes `Rows()` floats into `scores := make([]float32, mo.nE)` where `mo.nE` comes from the **config** (:916) — short rows leave zero-score experts (silent mis-routing), long rows / short scale tables / `ne < NumExperts` panic in the decode goroutine. Separately, **all** per-layer f32 vectors (Q/K/V/O bias, QK-norm, pre/post norms) are read at blob-controlled lengths and consumed at arch-derived lengths (`addBias`, `rmsNorm` — attention.go:11,95) with no length check, for every family. Same exported entry point (`LoadSerializedWeights`), same threat model C-06 was filed under.

Fix shape: extend validateShapes to the gemma4moe sub-block (cross-check `ne == arch.NumExperts`, router rows, scale lengths) and add a vector-length pass for biases/norms.

### R-08 · Anthropic `/v1/messages` silently drops `tools` when the request carries an image
`internal/serveapp/anthropic.go:404` · **silent capability loss** · residual of N-16 (fixed on the OpenAI surface only)

`handleMessages` dispatches to `serveVisionMessages` the moment image blocks exist — before any tools check — and the vision path never renders or parses tools (`vision_serve.go` has zero tool handling). The OpenAI path got exactly this guard as the N-16 fix (openai.go:351: "tools are not supported together with image inputs"). The named consumer of this endpoint (Claude Code) sends its toolset on every request; against a vision model, a screenshot turn silently loses all tools and comes back as prose with `stop_reason:"end_turn"`.

Fix shape: mirror the OpenAI 400 (or implement tools-in-vision) on the Anthropic path.

---

## Confirmed findings — minor

**serve (internal/serveapp)**
- **R-09** `openai.go:888` + `decoder/model.go:858` — M-04 residual: `Budget == 0` means both "no clamp" and "clamped to zero", so a prompt that **exactly** fills the resident cap returns empty content with `finish_reason:"stop"` (the M-04 text's own listed case). The gate test codifies the wrong branch (`openai_test.go:593`). Publish a `-1`/pointer sentinel or a separate bool.
- **R-10** `openai.go:591` — prompts in `(ctxCap, MaxPositions)` on a resident backend pass the C-20 check, then die mid-prefill with a 500 whose body leaks the internal "use the staged path" hint. Docs promise a staged fallback for long prompts; stateless Generate has none. Either fall back or 400 with `context_length_exceeded`.
- **R-11** `helpers.go:94` — `decodeJSON` returns `json.Unmarshal` errors verbatim: any type-mismatched field leaks Go struct/field/type names (`completionReq.logprobs`, …). Defeats M-06's "no leaked Go field names" for the bool case it was about (reproduced empirically).
- **R-16** `responses.go:241` — N-15 residual: the `/v1/responses` **tools** branch discards finish and hardcodes `status:"completed"`, so a budget-truncated (broken) tool call reads as complete.

**decoder**
- **R-13** `session.go:204` — N-01 half-fix: `genSpec` guards the **rewind** on empty prompt but still runs `reconcile(seq)` in the goroutine; a rejected empty-prompt call wipes the warm session KV via `TruncateTo(0)` (empirically demonstrated: pos 5→0). `Session.Generate` guards both — mirror it.
- **R-14** `session.go:91` — `reconcile` ignores `TruncateTo`'s `exact` and never resets on a mid-sweep forward error, so a recurrent-family session can warm-reuse partially-advanced Mamba/DeltaNet state after a transient error (C-01-class seam; needs a rare mid-sweep failure). Cheap fix: reset/consume `exact` on the error path.
- **R-15** `kvsnapshot.go:233-236` — LoadSession accepts `count>0, stride=0, nLive=0` (a state the writer never produces; the `continue` runs before the geometry check) → nil `k/v` ring → panic on first decode from a crafted CRC-valid snapshot. Only matters if `--session-dir` is attacker-writable — the same threat model the file's other hardening targets.

**gpu**
- **R-17** `gpu/gpu.go` — C-26 residual: 25 `GetBindGroupLayout` call sites, zero `Release`s (grep-verified); `c.layout` also missing from Close's nil-out set. ~40 native BGL objects (each holding a device ref) leak per Context — the C-26 checkpoint-swap scenario, smaller objects.
- **R-18** `gpu/decoderunner.go:378` — N-05 residual: `rmsQuant`'s int8 scratch is sized `padK` (mult-16) while its W4A8 consumers read to `padK32`; for `K%32 ∈ [1,16]` the kernel reads past the activation buffer, and int4 zero-pad nibbles decode to −8, so a clamped-nonzero read is a numerics error, not just formal OOB. Latent at today's dims (all %32), same status the audit gave N-05 — but the fix covered the three siblings and missed this one.
- **R-19** `gpu/decodetoken_batched.go:134` — N-07 residual: the batched path's `rms()` still dispatches 64 workgroups of the single-workgroup rmsnorm kernel (fixed only in decodetoken_fused.go). 64× redundant same-address writes; currently no production caller.
- **R-30** `gpu/layer.go:79`, `attention.go:475`, `decodefuse.go:138`, `vision.go:115` — multi-pipeline `ensure*` builders guard on the first field only; a mid-build failure leaves the guard satisfied and a later call dispatches nil pipelines. `ensureGEMVW8A16` shows the correct shape. Exotic trigger (retry after transient OOM), cheap fix.

**cuda**
- **R-20** `cuda/prefill.go:402` — the C-24 decline classifier is `strings.Contains(err.Error(), "panicked")`: **every** recovered executor panic (including future programming bugs) is relabeled "out of device memory … declining" and silently absorbed into the ~9×-slower sequential path. Match a sentinel from the OOM site instead of any panic.
- **R-21** `cuda/resident.go:596` + `prefill.go:169` — `ForwardN(nil, pos)` returns "empty prompt" error when `prefillReady` instead of the interface-documented no-op (cpu/webgpu no-op). Latent; contract drift.
- **R-25** `cuda/resident.go:344` — C-25 residual: the `perLayer <= 0` decline leaves `r.cacheExperts` true with no slots/`expCache` allocated → first decode would nil-deref (recovered into a persistent "executor job panicked"). Unreachable today; the branch's comment promises a fallback that doesn't exist. Clear `cacheExperts` in the guard.
- **R-26** docs drift, worth fixing because both misdescribe the exact C-24 invariant: `cuda/resident.go:233` still claims BuildResident's defer catches executor panics (it cannot — that was the C-24 finding; runJob does); `cuda/backend.go:389` claims oversized `GOINFER_MOE_CACHE_SLOTS` "crashes rather than declining" (false post-C-24). Also the "audited 12.6 glue.ptx" provenance underpinning the C-14 split rationale is wrong: only `moe.ptx` (+bench kernels) are 12.6.85; `glue.ptx` and all other production PTX are NVRTC **12.9.86**.

**metal**
- **R-12** `metal/snapshot_golden_test.go:1` — the one metal test file with no `//go:build darwin` tag (touched by N-10). Native-GOOS `go vet ./metal/` / `go test ./metal/` on the Linux box fails with `undefined: resident` (reproduced here). One-line tag.
- **R-22** `metal/prefill.go:261,287` — dead `pGemm` ("gemm_w4f16") pipeline: created, never dispatched (LM head moved to `pRmsQ`+`pGemvW8`). Exactly the N-09 class the N-09 fix cleaned elsewhere.
- **R-23** `metal/prefill.go` (batched path, opt-in env) — B-10 residual, self-documented at :356 ("that panic is recovered only on the BuildResident path, not here"): `ensurePrefill`'s compile panic and the ~24 per-call `MustBuf` OOM panics fire at request time with no recover. Only with `GOINFER_METAL_BATCHED_PREFILL=1`.
- **R-27** `metal/backend.go:154` — adapter `PrefillLast` checks `startPos+len > cap` but not `startPos < 0`; a negative position wraps to `uint32` and `kv_store_f16` writes far out of bounds on UMA. Unreachable today (decoder always passes 0); cheap belt.
- **R-28** `metal/model.go:596` — C-10's decline set (H, 2I, V, MoE inters) doesn't include the fused-QKV width `(nH+2nKV)·hd` dispatched at width 256 via `pSABias` (model.go:1266); an admitted arch with `hd%8 ≠ 0` would corrupt rather than decline. All shipped admitted arches are safe (hd ∈ {64,80,96,128,…}); add `qkvRows` to `bad8` for the guarantee C-10 promised.

**tokenizer**
- **R-29** `tokenizer/sentencepiece.go:674` (new in b548449) — the SPM-scores encode breaks equal-score merge ties by **token id**; llama.cpp breaks by **leftmost position**. On a vocab with equal-score competing merges this diverges from the reference tokenization silently. Medium confidence (mechanism certain, real-vocab frequency unknown; the parity test's 6 fixture cases don't exercise a tie). Worth matching llama.cpp's comparator since byte-identity with llama.cpp is the stated bar elsewhere.

---

## What held up (verified clean)

The following audit dispositions were checked end-to-end and are correct as described — listing the load-bearing ones: C-01 (all three facets, incl. gpu pos==0 re-zero both entries), C-02/C-03 (guards + panic-safe resBusy CAS pairing, target and draft), C-04 (unconditional windowed refusal), C-05 (snapshot refusal + stride restore both quant modes, ring round-trip symmetric), C-06 (standard-block coverage incl. per-expert and layer count — see R-07 for what's still outside it), C-07, C-08 (every setup path latches via recordUpload or panics into runJob→setupErr), C-09 (all five completion sites; the execAck happens-before claim checks out; paged-abort readback is benign — `moe_route` writes only idx<nE), C-11 (ceil consistent across sizing/dispatch/reduce), C-13, C-14 (kernel + dispatch + strided-scan lowest-index for any V), C-15 (cuda `f32tof16` token-identical to `decoder.f32ToF16bits`; launchGluSplit is the sole single-buffer split), C-17 (decline at the owner, both gate tests device-free), C-20/C-21/C-22 (shutdown: BaseContext cancel + tryLockUntil + second-signal force-exit — no deadlock), C-23, C-24 (recover boundary + scratch-defer-before-first-alloc + errPrefillDeclined fallback), C-25 (guard correct — see R-25 for the state it leaves), C-26 (all 21 inline creation sites tracked, LIFO drain, idempotent, no double-release vs runner-owned buffers — see R-17 for the BGL gap), C-27 (fused/batched/runner constructors complete — see R-06 for the neighbors), C-28 (both sites, consumed[] prevents double-unmap; no other persistent-buffer MapAsync user), C-29 (every map access locked; retire-not-close; growth bounded), C-30 (both pagers mutexed; cross-stream madvise is advisory-only), C-31 + N-17/N-18 (bounds + depth caps), B-08..B-14 (incl. B-14 auth: localhost default, constant-time compare, admin gated), M-01 (semaphore leak-free on all paths incl. panic; SSE unaffected by ReadTimeout), M-02/M-03 (consistent across all three surfaces), M-05 (all three call sites; named-only 400 by design), M-06 (integer path), M-07 (canBatchN superset), M-08 (close-on-error, no double-close), M-10 (all three pack entry points), M-11 (every DispatchTG tgBytes covered by the staging max), M-12-as-scoped (see R-05 for the MLA collateral), M-13/M-14/M-15, M-16 (all named leak paths), M-17 (caps pinned to kernel sources by test), M-21 (cancellation actually checked per-write), M-22/M-23, M-25 (reset outside the recurrent guard), G-02 (single-site scale, no double-scale via adapter), G-04, N-02, N-05/N-06/N-07/N-08-as-scoped, N-13, N-20. Full WGSL↔Go layout re-verification of the kernels added since the audit found no mismatches; cuda launch-geometry sweep (every production dispatch vs its .cu) found none either; metal sizing/geometry sweep likewise (attention `sc[4096]` bounded by checkCap, moe_route bounds enforced at build).

Tooling: `go vet` (go1.26.5) clean on root+gpu (workspace), cuda, metal (darwin/arm64) — the single failure is R-12; `gofmt -l` clean; tests: root module green except the two `testdata/`-dependent tests absent in this sandbox; gpu and cuda modules fully green, including the C-24/C-25/C-08/C-15 gates and the serveapp chaos/backpressure suites.

## Suspicious but unconfirmed (leads, not findings)

- Gemma-3 vision path: M-15's decode-bomb cap covers `QwenPreprocess` only; the gemma3 branch delegates to aikit `vision.Preprocess` — couldn't confirm an input-pixel bound there.
- Anthropic `count_tokens` ignores image blocks → undercounts vs what `/v1/messages` prefills.
- `tool_choice:"required"`/`"any"` with multiple tools is accepted but unenforced (no grammar, may answer in prose) on both surfaces.
- Speculative **draft** models are exempt from `specRollbackSafe` — a windowed/recurrent draft degrades acceptance silently after its window wraps (lossless, perf-only).
- metal `gemma4_moe.go:276` — mmap byte-copy staging ignores `int4DirectBytes` ok beyond expert 0; a mixed-format expert set would tag a slot with a new id while holding the previous expert's weights. No known writer produces one.
- metal `prefill.go:425` — PrefillLast bypasses `finalizeLogits` (softcap); protected only by admission, not locally.
- gpu `backend.go:220` — backend `Close` races a concurrent in-flight matmul (drops the lock before `ctx.Close()`; post-Close use panics on a nil map). Caller-misuse territory.
- cuda `prefill.go:197` — scratch freed on the mid-loop launch-error path without a stream sync while kernels may reference it (aikit doc puts that on the caller). Panic paths are safe.

## Environment caveats + cleanup

Static analysis and tests only — no GPU/device execution here; metal was cross-compiled, not run. `testdata/`, `demo/`, `dist/`, `.git` were not copied to the sandbox. Sandbox-only edits (never part of any finding): dependency `replace` blocks in `go.work`/`cuda/go.mod` and dir-cloned deps, since the sandbox has no module-proxy egress.

One artifact to delete on the Mac: `~/tmcode/goinfer/_to_delete/cowork-stage-src.tgz` (the source snapshot staged for this review; the sandbox can't delete files on your machine).
