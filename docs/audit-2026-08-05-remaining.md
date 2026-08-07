# goinfer audit — remaining (OPEN) items

**Derived from** [`audit-2026-08-05.md`](audit-2026-08-05.md), which holds the full record with
the closed findings and their dispositions. This file carries ONLY the items still open as of
`08fc4ea` (2026-08-06). Verify each against the tree before acting — some `metal/` fixes have
landed in code and only await a device run.

Ownership tags: **[mac]** needs a Metal box (the parallel MacBook owns these); **[linux]** is
ownable on the CUDA box; **[decision]** is a breaking post-tag API change that needs a maintainer
call, not a unilateral fix.

## Status at a glance

| severity | closed | remaining |
|---|---|---|
| Blockers (B) | 14 / 14 | 0 (all resolved; release shipped) |
| Critical (C) | 31 / 31 | **0** (both owed regression tests now device-verified: C-01-resident on the Linux/webgpu box — real granite, maxAbs=0; C-03 on the Mac and cross-confirmed on the box under `-race`) |
| Gate (G) | 6 / 6 | **0** (G-03/G-05 on Linux; G-04/G-06-residual device-verified on the Mac / macOS 26.6) |
| Major (M) | 23 / 23 | **0** of the original 23 — M-21 (ctx on transcodes) + M-22 (GPUSpecStats rename) now FIXED. Only **M-25** remains (new, latent, deferred — not in the original count) |
| Minor (N) | 0 / 24 | **24** |

**Critical batch fixed on the Linux box (2026-08-06):** C-05, C-13, C-23, C-27, C-28, C-29 — each
with a gate test — plus **C-11** (metal, safe Go one-liner, cross-compiled; device run owed on the
Mac). **C-15 is FIXED** (2026-08-06, option (b) — canonical f32tof16 + a dedicated gate/up-wiring
test `TestMoeSwigluWiring_C15`, since the e2e cosine gate's bug-B power was a truncation artifact; the
PTX-regen blocker was also shown surmountable via a pip NVRTC 12.6.85 venv — see below). **C-12 is now CLOSED** (2026-08-06, Mac) — the B-11 unexport already
routes every external caller through the cap-checked `metalResident` adapter, so no unchecked path
remains (details below). **C-10 is now MITIGATED in code** (2026-08-06, Mac, device-verified on macOS
26.6) — `buildResident` declines any non-multiple-of-8 SA-GEMV output width → CPU fallback, closing
the silent-corruption half; the kernel-side `N`-guard stays deferred but is no longer a correctness
gate. **C-09 is now FIXED** (2026-08-06, Mac, device-verified on macOS 26.6) — `gpu.Encoder.Err()`
(aikit `gpu/v0.26.1`) latches the command buffer's status/error at `WaitDone`/`End`, and every metal
forward path records it into `resident.execErr`, which the `metalResident` adapter surfaces (and
clears) after each forward instead of returning the stale logits a faulted buffer left behind.
**C-11 is now CLOSED** (2026-08-06, Mac, device-verified on macOS 26.6) — the ceil-tiled fused
argmax is confirmed equal to argmax(full-logits Forward) by a committed-fixture device gate, and the
non-%8-V failure it targeted is now unreachable (C-10 declines that shape from the resident path).
The 2 owed regression tests are now device-verified: **C-03** on the Mac (webgpu) and cross-confirmed
on the Linux box under `-race`, and **C-01-resident** on the Linux/webgpu box against real granite
(maxAbs=0; this Mac carries no SSM checkpoint so it skips there).

**Mac OS update (2026-08-06) — RESOLVED.** This MacBook moved macOS 26.5.2 (25F84) → 26.6 (25G72),
which redded the OS-pinned `TestMetalSnapshotGolden` (expected — MSL recompiles per-OS toolchain). The
golden has now been **re-baked on 26.6** (see G-02 below), folding in the embed-scale fix; it is green
again on this machine. Notably `mixtral-tiny` was byte-identical across the two OS versions (only
`gemma4-dense-scaled` moved, via the embed-scale fix), so the OS bump alone changed no bits here.

---

## §1 — Blockers

All 14 release blockers are resolved (module-graph mechanics closed during the v0.9.0/v0.9.1
release; B-12 teardown-naming fixed in `1a4d4bd`). Nothing open. See the source doc's Disposition
log for detail.

---

## §2 — Critical (open)

**C-05 — FIXED** (2026-08-06, Linux box). `Snapshot` now refuses a cache whose global layers have
non-uniform KV widths (a global `stride[l] != 0 && != kvDim`) — gemma-4's per-layer geometry — the
same policy as the recurrent/latent families, so the caller cold-prefills instead of round-tripping
a snapshot the format can't restore. Gate `TestSnapshot_refusesNonUniformKVWidth_C05`. Original
finding below.
**C-05 (was PARTIAL)** [linux] | `decoder/kvsnapshot.go` — session KV snapshot mis-slices gemma-4.
`LoadSession` restores `stride[l] = kvDim` for uniform geometry, but gemma-4 global layers use a
different KV width than local ones, and `Snapshot` refuses only delta/mamba/mlaLatent — so a
gemma-4 session round-trips through `--session-dir` and then mis-slices on the first `TruncateTo`.
Reachable via the serve session path.
*Fix:* refuse gemma-4 (per-layer KV width) in `Snapshot`, or restore per-layer strides on load.

**C-09 — FIXED** (2026-08-06, Mac, device-verified on macOS 26.6). Two layers:
- **aikit `gpu/v0.26.1`** adds `Encoder.Err()`. `WaitDone`/`End` now read the command buffer's
  `status` (via `objc.Send[uintptr]`, the arm64 integer-return path) and, on
  `MTLCommandBufferStatusError`, format the `NSError.localizedDescription` — latched into `Encoder.err`
  *before* the autorelease pool drains (reading a drained cb is a use-after-free), so `Err()` is a safe
  getter. (`gpu/v0.26.0` first added `Err()` reading the cb directly; `v0.26.1` moved the read ahead of
  the drain.)
- **goinfer `metal/`** records `enc.Err()` at every command-buffer completion site (`execLoop` pipelined
  decode, `forwardLogits`, `forwardLogitsPaged`'s per-layer submits + final head, `PrefillLast`,
  `ForwardArgmax`) into `resident.execErr` (plain field: the pipelined executor writes it before the
  `execAck` send, which happens-before the adapter's read). The `metalResident` adapter's `Forward` /
  `PrefillLast` call `takeExecErr()` after each forward and return the error **instead of** the stale
  logits an aborted buffer left behind, clearing it so the next token is unaffected.

Device validation: `gpu.TestCmdBufStatus_C09` (aikit) proves the objc status read (completed→4, `Err()`
nil) and the `NSError`→string abort formatting on device; `metal.TestMetalResident_C09_execErrSurfaces`
(goinfer) proves a clean decode never false-positives and that a recorded abort surfaces through the
adapter and then clears. The abort is injected rather than provoked by a real GPU fault because **this
machine's GPU silently tolerates every abort trigger tried** (OOB and 32-GiB-unmapped stores, 64-MiB
threadgroup memory vs a 32-KiB limit, 4096 threads/tg vs a 1024 max — all report status Completed);
that permissiveness is exactly the hazard this check guards against on stricter OS/GPUs. Original
finding below.
**C-09** [mac] | `metal/model.go:651,730` + `gemma4_moe.go:479,503` — no site in `metal/` ever
observes the Metal command-buffer status. A kernel that aborts (see M-11) leaves the host reading
the previous token's logits with no error. Underlies several other metal findings.
*Fix:* read `commandBuffer.status`/`error` after commit and surface a failure.

**C-10 — MITIGATED in code** (2026-08-06, Mac — device-verified on macOS 26.6). Took the audit's
Go-side alternative: `buildResident` (metal/model.go, right after the guDim/MoE-inter resolve) now
declines — returns an error → CPU fallback — any model whose SA-GEMV output width isn't a multiple
of 8 (`hidden`, `intermediate`, `vocab`, and the MoE expert/shared/gemma-4 dense/MoE inters). The
attention widths qDim=nH·hd / kvDim=nKV·hd are structurally %8 (hd is 64/128), so they're exempt.
This closes the *silent-corruption* half: an odd-width future model gets a clean decline, not
uninitialised-scratch logits. **Device-verified NOT to regress any current shape** — the resident
build + parity tests still pass on 26.6 (TestGemma4DenseScaled_metalParity, TestMoE_assemblyVsDense,
TestMetalResidentCheckCap); every shipped metal-eligible arch is %8, so it declines nothing today.
The deeper kernel fix (add an `N` param + `if (row >= N) return;` to the other six SA-GEMV variants —
buffer-binding churn, the "stale binding" class the metal-CI comment names) stays DEFERRED, but is no
longer a correctness gate now that odd widths can't reach those kernels resident. Original finding below.
**C-10** [mac] | `metal/kernels.go:134,250` + `moe.go:100,128` — seven GEMV variants derive the
output row from the *runtime* threadgroup size (reduced in the tail threadgroup of a non-uniform
`dispatchThreads:` launch). Only `gemv_w4a8_sa_bk` carries `if (row >= N) return;`.
*Failure:* any `N % 8 != 0` makes the last threadgroup rewrite an already-written row while the
true tail rows are never written — GEMV output tail is uninitialised scratch. Today's shapes are
multiples of 8 by luck.
*Fix:* add the guard to all seven, or assert `N%8==0` at each dispatch site.

**C-11 — CLOSED** (2026-08-06, Mac, device-verified on macOS 26.6). `nTiles := (V + 7) / 8` (was
floor `V/8`): the strictly-larger `r.part` buffer holds every tile ForwardArgmax dispatches and `uP`
now counts all of them, so the last tile is written in-bounds AND reduced. No kernel/binding change;
correct by construction. Device gate `metal.TestMetalResident_C11_argmaxEqualsFullLogits` confirms
the fused block-argmax equals argmax(full-logits Forward) across a greedy walk on a committed %8-vocab
fixture (tmVocab=64) — a self-contained replacement for the owed heavy-model run. Note the gate can't
exercise the ceil≠floor *difference* directly: that needs a non-%8 V, and the only V where ceil≠floor
is exactly the shape **C-10 now declines from the resident path** (and `ForwardArgmax` has no
production caller — metal decode uses full-logits Forward + host argmax), so the non-%8 failure is
unreachable two ways over. The ceil fix stands as belt (correct sizing by construction) to C-10's
suspenders (decline). Original finding below.
**C-11** [mac] | `metal/model.go:534` — `nTiles := V / 8` (floor) sizes `r.part`/`r.uP`, but
`ForwardArgmax` dispatches `ceil(V/8)` tiles and the kernel writes `part[tgid]` unconditionally.
*Failure:* any `V % 8 != 0` (e.g. 50257) writes 8 bytes past `r.part` — on UMA it lands in the
next buffer, potentially another resident model's weights — and `uP = V/8` leaves the last tile
unreduced, so the greedy token can differ from `argmax(Forward())`.
*Fix (applied):* `nTiles := (V + 7) / 8`. (The audit also mentions a row guard; the ceil buffer
alone fixes both the OOB write and the unreduced last tile, since the amax kernel writes `part[tgid]`
and tgid ∈ [0, ceil).)

**C-12 — CLOSED** (2026-08-06, Mac — by the B-11 unexport, verified against the tree). The premise
("the *exported* `Forward`/`ForwardEmb`/`ForwardArgmax` take `pos` unchecked") no longer holds: B-11
unexported the type (`type resident struct`, `metal/model.go:83`), so those inner methods are not
public API at all. The only external entry is the `metalResident` adapter (registered via
`decoder.RegisterBackend`), and every one of its methods checks the cap — `Forward` and `ForwardN`
call `checkCap` (`backend.go:123,161`) before the inner call, and `PrefillLast` rejects
`startPos+len > metalCtxCap` (`backend.go:154`). Crucially, `metalResident` does **not** implement
`decoder.ResidentGreedy`, so the unchecked inner `ForwardArgmax` fast path is never reached from
outside — the decoder falls back to the checked `Forward()` + argmax. No unchecked external path
remains. Original finding below.
**C-12** [mac] | `metal/model.go:618,633,831` — the context-cap guard lives only in the unexported
adapter (`checkCap`); exported `Forward`/`ForwardEmb`/`ForwardArgmax`/`PrefillLast` take `pos`
unchecked.
*Failure:* `Resident.Forward(id, 4096)` at/past `metalCtxCap` makes `kv_store` write past a KV
buffer (UMA write into an adjacent buffer) and `attention` index `threadgroup float sc[4096]` OOB.
*Fix:* move the check into the exported methods.

**C-13 — FIXED** (2026-08-06, Linux box). `DecodeTokenFusedBatched` rejects `M > gemmRowMaxM` up
front (the one-shot `MatmulW8A8GemmRow` already did), so a block wider than the accumulator can't
alias row 15. Gate `TestDecodeTokenFusedBatched_Mbound_C13`. Original finding below.
**C-13** [linux] | `gpu/decodetoken_batched.go:127` — `DecodeTokenFusedBatched` dispatches with
`M = len(xs)` and no bound, while the kernel's accumulator is `array<i32, 16>`.
*Failure:* 17+ rows (a speculative block of K≥16) index a 16-element private array OOB; WGSL
robustness clamps to 15, so rows ≥16 accumulate into and read back the row-15 accumulator —
silently wrong logits, no error. The one-shot sibling `MatmulW8A8GemmRow` does check.
*Fix:* reject `M > gemmRowMaxM`, or chunk.

**C-15 — FIXED** (2026-08-06, Linux box; chose option (b)). Shipped as three parts:

1. **The fix:** `cuda/f32tof16` is now byte-identical to the canonical `decoder.f32ToF16bits`
   (round-half-up + gradual underflow to subnormals) — NOT the audit's RNE (a lone RNE would
   re-diverge cuda from metal/aikit). The old helper truncated + flushed all subnormals. Gate
   `TestF32ToF16_C15` asserts bit-for-bit canonical agreement over a sweep incl. the subnormal range.

2. **Why a floor recalibration was impossible** (measured): the 0.999 floor's entire bug-B (gate/up
   swap) discrimination was a TRUNCATION artifact. With correct scales, correct dispatch = 0.997833
   and bug B = 0.997846 — indistinguishable (matched f16 scales don't help; the near-tie is int4
   KERNEL arithmetic, device W8A8-GEMV vs CPU int4, and the f32 router makes it a razor-thin
   logit-argmax tie, not an expert flip). On `mixtral-tiny`'s RANDOM experts `silu(up)*gate ≈
   silu(gate)*up`, so no floor separates correct from bug B.

3. **The gate redesign (option b):** the e2e cosine floor was recalibrated to **0.995** — below the
   correct run (0.997833) and above the one STRUCTURAL bug that survives the 3% argmax rule (A,
   down-proj slot pinned, 0.988509; C/D/E have >3% gaps, caught by the rule). Bug B is now caught by a
   NEW dedicated test `TestMoeSwigluWiring_C15`: the gate/up split convention was centralized into
   `cudaResident.launchGluSplit` (one place, shared by routed/shared/gemma-4), and the test drives it
   with crafted gate≠up and asserts the pre-quant SwiGLU output directly — scale/fixture-independent.
   Break-to-verify: swapping gOff/uOff fails the wiring test (maxErr 0.40) while the e2e cosine
   (0.997846) sails past. Re-measured control table is in the `moe_parity_test.go` comment.

*Env note (resolved, not needed in the end):* the "no `moe.ptx` regen on this box" premise was false —
a pip venv with `nvidia-cuda-nvrtc-cu12==12.6.85` (the exact V-number in the frozen PTX header)
regenerates `moe.ptx` byte-identically (`build_ptx.sh` `NVRTC_LIB`/`CUDA_INC`). Not needed: the fix is
a Go-side helper (no PTX) and every control bug was Go-side injectable.
**C-15 (original)** [linux] | `cuda/kernels.go:119-132` — `f32tof16` truncates the mantissa (`m>>13`,
no rounding) despite a doc comment claiming round-to-nearest-even, flushes everything below the f16
normal range to zero, and returns `+Inf` rather than saturating on overflow.
*Failure:* every int4 group scale is biased downward by up to 1 ULP (mean ≈ −0.05%) — a systematic
shrink applied to every int4 projection/expert/LM-head scale. Any group whose max |w| falls under
~4.3e-4 is silently zeroed on device while the CPU reference keeps it.
*Original suggested fix (SUPERSEDED — see DEFERRED note):* implement RNE, emit subnormals, clamp
overflow to `0x7bff`. The real fix is to match `decoder.f32ToF16bits` (round-half-up), not RNE.

**C-23 — FIXED** (2026-08-06, Linux box). `LoadProjector` now rejects `len(normW) != visionHidden`
right after reading it (mirroring the existing `projW` guard), so a short norm tensor is a load
error, not an index-out-of-range panic in `Forward` on the first image request. Gate
`TestLoadProjector_normWLength_C23` (synthetic safetensors). Original finding below.
**C-23** [linux] | `multimodal/projector.go:80` — `mm_soft_emb_norm.weight` is never validated
against `vision_config.hidden_size` (the sibling `projW` length *is* gated two lines below).
*Failure:* a checkpoint whose config says 1152 but whose norm tensor is shorter loads cleanly, then
`Forward` panics `index out of range` inside the HTTP handler goroutine — first image request kills
the process.
*Fix:* reject `len(normW) != p.visionHidden` at load.

**C-27 — FIXED** (2026-08-06, Linux box). Both `DecodeTokenFused` and `DecodeTokenFusedBatched`
now carry the `newDecodeRunner` `buildErr` pattern: `storF`/`uni`/`bind`/`disp` (and a guarded
`cpy`) short-circuit on the first allocation/bind failure and the function returns it before Submit
— VRAM exhaustion is an error the caller falls back on, never a panic or a nil-buffer deref. Error
paths only fire under device-allocation failure (not unit-testable without a fault-injecting mock);
verified by build/vet/staticcheck `-tags gpu`. Original finding below.
**C-27** [linux] | `gpu/decodetoken_fused.go:64` + `decodetoken_batched.go:72` — exported
`DecodeTokenFused`/`DecodeTokenFusedBatched` `panic(e)` on bind-group failure, and their
`storF`/`uni` helpers discard `CreateBuffer` errors and pass nil buffers downstream.
*Failure:* on VRAM exhaustion the exported entry points panic instead of returning an error, taking
down the calling server. `newDecodeRunner` fixed exactly this with a `buildErr` short-circuit;
these two were left behind.
*Fix:* port the `buildErr` pattern.

**C-28 — FIXED** (2026-08-06, Linux box). `runBatch` now records a `mapErr` and settles the Poll,
then a deferred sweep `Unmap`s every successfully-mapped-but-unconsumed `stag` on ALL return paths
(a MapAsync error, a non-Success status, or a clean finish) — the persistent per-runner buffer can
no longer be left mapped and poison every future call. Error path needs device failure to fire;
verified by build/vet/staticcheck `-tags gpu`. Original finding below.
**C-28** [linux] | `gpu/decoderunner.go:1133` — `runBatch` returns on the first non-Success map
status without unmapping the rows it already mapped (`stag` is a per-runner *persistent* buffer).
*Failure:* the next call's `MapAsync` on an already-mapped buffer fails, and every subsequent call
fails identically — the runner is permanently poisoned for the process lifetime. Line 1126 has the
same shape.
*Fix:* deferred cleanup walking all rows whose `MapAsync` succeeded.

**C-29 — FIXED** (2026-08-06, Linux box). A new `adaptersMu` guards `m.adapters` (LoadAdapter write
vs UseAdapter/HasAdapter reads), and re-registration now RETIRES the displaced runtime into
`retiredAdapters` (released only at `Model.Close`) instead of munmapping it — a live Session holding
it via `cache.lora` keeps reading valid memory. Gate `TestRegisterAdapter_retiresNotCloses_C29`
(retire bookkeeping + `-race` concurrent register/read). Original finding below.
**C-29** [linux] | `decoder/lora.go:285` — `LoadAdapter` closes (munmaps) the previously registered
runtime with no check that a live `Session` still references it, and mutates the unsynchronized
`m.adapters` map.
*Failure:* re-registering an adapter name while a session uses it unmaps the safetensors under
`applyLoRA`'s `d.a`/`d.b` → SIGSEGV; concurrently the map write racing a `Session.UseAdapter` read
triggers Go's fatal `concurrent map read and map write`.
*Fix:* mutex the map; refcount/retire runtimes rather than closing on re-registration.

### Owed regression tests (code fixed, device-gated)

- **C-01 (resident half)** [linux/gpu] — **DONE** (2026-08-06, Linux/webgpu box, device-verified on
  real granite-4.0-h-tiny-Q8_0). The gpu resident re-zeroing `{win,ssm}` at `pos==0` is fixed in code;
  `gpu.TestResident_C01_pos0ResetsRecurrent` reproduces the leak directly — run token T at pos 0 on a
  fresh resident, compound `{win,ssm}` by decoding 8 more tokens, then run token T at pos 0 again and
  assert the logits are reproducible (fix re-zeroes; a leak diverges). **Passed on the box: maxAbs = 0**
  (exact reproduction across sequences — state genuinely re-zeroed; well inside the 1e-3 floor). Needs a
  Mamba-2 hybrid resident (`GOINFER_HEAVY_TESTS=1`; path via `GOINFER_SSM_MODEL` or the default granite
  path); skips on the Mac purely for model-absence (WebGPU itself inits there). CPU half covered by
  `TestTruncateTo_resetsRecurrent`.
- **C-03** — **DONE** (2026-08-06, Mac, device-verified on webgpu). `GenerateSpeculative`'s `resBusy`
  CAS is fixed in code; `gpu.TestSpeculative_C03_concurrentResidentClaim` runs a plain resident
  `Generate` and a `GenerateSpeculative` CONCURRENTLY on one `*Model` (webgpu 1.5B target + CPU 0.5B
  draft, the parity-test models) across 4 rounds and asserts every stream is a valid greedy decode on
  EITHER backend — resident, or the staged CPU fallback the CAS loser takes (the two differ by a token
  at a near-tie: the documented resident-vs-CPU gap, not corruption). An interleaved-resident-KV
  corruption (the pre-fix bug) matches neither reference. Passes on this Mac; heavy+webgpu gated
  (`GOINFER_HEAVY_TESTS=1`, models via `GOINFER_SPEC_TARGET`/`GOINFER_SPEC_DRAFT`). **Cross-confirmed
  2026-08-06 on the Linux/webgpu box under `-race`** (PASS, 181s, 4 concurrent rounds, no
  interleaved-KV corruption) — clean on both backends' hardware.

---

## §3 — Gates that cannot fail (open)

**G-02 — DONE** (2026-08-06, Mac, re-baked on macOS 26.6). The embed-scale fix (`resident.embedScale`
applied by `loadEmbedRow`) is baked into `testdata/metal_snapshot_golden.json` on this machine. The
re-bake was gated on evidence, not reflex: before writing, both live-correctness gates passed on 26.6
— `TestMetalEmbedScale_forwardMatchesForwardEmb` (Forward(id) == ForwardEmb(production √hidden-scaled
row)) and `TestGemma4DenseScaled_metalParity` (metal == CPU truth, cosine 1.0) — so the new values are
the *correct* fixed output, not degenerate. The diff is exactly the expected shape: `gemma4-dense-scaled`
moved (argmax 170/123/52 → the embed-scaled 2/2/2 at depths 130/260/320) and **`mixtral-tiny` did NOT
move at all** (byte-identical across 26.5.2→26.6 — the invariant held), plus the env bump to 26.6.
`TestMetalSnapshotGolden` is now green on this Mac. (Re-baking on a different Mac/OS will red again —
expected, MSL recompiles per-OS toolchain — and would re-bake to that machine's reference.)

**G-03 — FIXED** (2026-08-06, Linux box). `buildMatrix` and `buildHardwareRows` now `t.Setenv`
both `GOINFER_GEMMA4_RESIDENT` and `GOINFER_SSM_RESIDENT` to empty up front, so the generated
matrix is a property of the code alone. Verified: `TestCapabilityMatrix`/`TestHardwareMatrix`/
`TestParityManifest` pass even with **both env vars exported** (pre-fix, that reddened the freshness
check for an unrelated reason and `-update` baked an env-on answer). Original finding:
`GPUResident` = `arch.decodeRunnerEligible()` reads those two env vars.

**G-05 — FIXED** (2026-08-06, Linux box). Committed a tiny (~4.3 KB) tokenizer-only ChatML GGUF
fixture at `tokenizer/testdata/chatml-tiny.gguf` (generator: `scripts/chatml_tiny_fixture.py`) — a
real GPT-2 byte-level base (`bytes_to_unicode`, byte-for-byte matching `buildByteLevelTables`) + 15
merges (the loader refuses a zero-merge decode-only vocab) + `<|im_start|>`/`<|im_end|>` CONTROL
tokens, `model=gpt2`/`pre=qwen2`. Both `TestEncodeSegments_parityAndInjection` (tokenizer) and
`TestRenderSegments_chatml` (chat) now default to it and **PASS with no env** — the gate runs
everywhere; `GOINFER_CHATML_GGUF` stays an override. The tests' own naive-encode sanity assertion
(forged `<|im_end|>` must promote under whole-string Encode) confirms the fixture is discriminating.

### Metal G-items — code fixed on the Linux box (cross-compiled darwin/arm64, device run owed)

**G-04 — DONE** (fixed on the Linux box; device-verified on the Mac / macOS 26.6, 2026-08-06). The
residency-scope switch now accumulates the pinned buffers into a local `pinned` slice inside
`addAll()` and assigns `r.residencyBufs = pinned` AFTER the switch — so every slot-scoped arm
(`slots`, `slots+kv`, `slots+scratch`, default) records the true pinned set, not just `default`. The
two whole-device diagnostic arms (`readonly`/`all`) intentionally leave it nil (not a slot-pool
contract; `TestResidencySet_pinsExactlyTheLiveSlots` doesn't run under them). **Device-verified:**
`TestResidencySet_pinsExactlyTheLiveSlots` PASSES ("pins exactly the 24 live slot buffers") under BOTH
the default scope AND the previously-broken `GOINFER_MOE_RESIDENCY_SCOPE=slots` arm (which pre-fix
reported "no buffers pinned"). Original finding: `r.residencyBufs` was assigned only in the `default`
arm.

**G-06 (residual) — DONE** (fixed on the Linux box; device-verified on the Mac / macOS 26.6,
2026-08-06). Added a `metal/modelsdir_test.go` `modelPath` helper (mirrors the tokenizer/chat/decoder
ones: `GOINFER_MODELS_DIR`, default `$HOME/models`) and routed all 6 `gemma4-26b-int4.giw` test sites
through it (`gemma4_26b_{localize,paged,routing,possweep,widthconsistency}_test.go` +
`paging_budget_test.go`) — no more hardcoded `/Users/...`. **Device-verified:** metal module builds +
vets clean on-device; with the `.giw` at `$HOME/models`, all five `gemma4_26b_*` tests resolved the
model and PASSED (`paging_budget` skips cleanly behind `GOINFER_BUDGET_PROBE`), confirming the
path-resolution refactor is behavior-neutral.

---

## §4 — Major (open)

**M-09 — DONE** (2026-08-06, Mac, macOS 26.6). `PrefillLast`'s per-layer loop now binds `L.invf` and
`L.uWindow` (the same per-layer RoPE table + window `encodeTrunkInto` decode binds), not the model-
level `r.invf`/`r.uWindow`. A mixed local/global-window arch now prefills each global layer with
window=0 instead of the local window. **Latent, so no positive on-device trigger exists** (every
shipped mixed-window arch also has per-layer RoPE and is declined by `prefillFeatures`, which does not
claim `FeatPerLayerRoPE`); admitted archs have a uniform RoPE table so `L.invf == r.invf` there and
the change is behaviour-neutral. **Verified correct-by-construction** (prefill now binds identically
to decode) **and regression-clean on device**: `TestPrefillParity` (qwen2.5-1.5B) still matches
sequential decode — last-token argmax equal, cosine 0.9986 — and `TestPrefillNoNaN` passes. Original
finding below.
**M-09** [mac] | `metal/prefill.go:391` — `PrefillLast` binds the model-level RoPE table and window
while decode binds the per-layer `L.invf`/`L.uWindow` (0 for non-local layers). `prefillFeatures`
claims `FeatSlidingWindow` and omits only `FeatPerLayerRoPE`, so a mixed local/global-window arch
without per-layer RoPE prefills its *global* layers with the local window applied. Latent only
because every shipped mixed-window arch also has per-layer RoPE.
*Fix:* bind the per-layer values (both already exist).

**M-25 (new, 2026-08-06)** [any] | `decoder/kvcache.go` — `TruncateTo(0)` (the Session.Reset /
`sessionLRU.fresh` reset path) clears the KV, recurrent (`mamba`/`delta`), and MLA state but
**leaves the multimodal fields `imgBlocks`, `mropePos`, `mropeDelta` populated**. Reported by an
external pass with a unit repro (`TestTempKVCacheStateLeak`, not committed) asserting `imgBlocks`
is empty after `TruncateTo(0)` — the method behaviour is confirmed. **Latent, not currently
reachable:** verified that every VL generation gets a *fresh* cache (`GenerateVL`/`GenerateQwenVL`
→ `m.NewCache`), `SetImageBlocks`/`mropePos` are set only inside the VL prefill on that single-use
cache, and the serve dispatch routes image requests to the fresh-cache VL path while warm-KV
**sessions are text-only** (`openai.go:797/799` vs `sess.Generate`) — so no reused cache ever
carries image state into a later read. It goes **live and silent** the day VL is wired through the
warm-KV session path (image-in-chat with prefix reuse), which is a plausible future feature.
*Fix (cheap defence-in-depth):* nil `imgBlocks`/`mropePos` and zero `mropeDelta` in the `pos == 0`
branch of `TruncateTo`, alongside `resetRecurrent()` — makes the reset complete and the external
repro pass. Touches core, so it carries a goldens-gated parity refresh.

**M-10 — DONE** (2026-08-06, Mac, device-verified on macOS 26.6). The dense W4A8 pack entry points
now reject `K%32 != 0`: `int4Buf` returns an error and `int4Concat` panics (no error return; caught
by `buildResident`'s recover → clean CPU decline), each above their int4-direct and int8-repack
sub-paths. The gemma4-MoE paged path already guarded `H/MoeInter/DenseInter %32` (gemma4_moe.go:173).
Gate `TestInt4Pack_declinesNonMultipleOf32K` (K=40 declines via both error and panic; K=64 still
accepted); gemma4-dense + MoE residents still build and match parity (no false decline). Original
finding below.
**M-10** [mac] | `metal/model.go:251` + `pack.go:11` — int4 pack paths check `group == 32` but never
`K % 32 == 0`, while the kernels hard-assume it. `K=40` fills only the first group and leaves zero
nibbles decoding as −8; the per-row stride disagrees with the kernel's; `K=12` panics. CUDA declines
this shape; WebGPU pads.
*Fix:* reject `K%32 != 0` in the three pack entry points so `BuildResident` declines.

**M-11 — DONE** (2026-08-06, Mac, device-verified on macOS 26.6). aikit `gpu/v0.27.0` adds
`Device.MaxThreadgroupMemoryLength()`, and `buildResident` now computes the widest threadgroup
staging (`maxThreadgroupStageBytes` = 2·max(hidden, q-width, MoE expert intermediate)) and declines →
CPU when it exceeds the device tile limit. The dense down-proj uses the non-staging `pGemv`, so the
dense intermediate is deliberately excluded. Gate `TestMaxThreadgroupStageBytes` (arithmetic: which K
wins, ×2, exactly-at-limit allowed, dense-inter excluded) + aikit `TestMaxThreadgroupMemoryLength`
(device reports 32768 B) + gemma4-dense/MoE residents still build (no false decline). Original
finding below.
**M-11** [mac] | `metal/moe.go:307` + `model.go:1091` + `gemma4_moe.go:411` — dynamic threadgroup
memory is computed from model dims and never validated against `maxThreadgroupMemoryLength`. Mixtral
is already at 87% of a 32 KB budget; `inter ≥ 16384` exceeds it, Metal aborts the command buffer,
and per C-09 nobody notices.
*Fix:* query the limit in `BuildResident` and decline.

**M-19 — FIXED** (2026-08-07, Linux box; idiomatic submodule-entrypoints approach). The tag-gated
blank imports are OUT of the root module. Server/REPL logic moved to importable app packages
(`internal/serveapp`, `internal/chatapp`, `internal/gemmaapp`); the root `cmd/serve`, `demo/chat`,
`demo/gemma` are now pure-Go shims that call `…app.Main()` and import no backend. The opt-in
accelerated builds live in the submodules as their own binaries — `gpu/cmd/{serve,chat}` (`-tags
gpu`), `cuda/cmd/{serve,chat}` (`-tags cuda`), `metal/cmd/{serve,chat,gemma}` (darwin) — each
blank-importing its backend before the shared entrypoint. Root `go.mod` now requires only
`aikit` + `golang.org/x/text`; **`GOWORK=off go list -m all` on the root is clean** — no
`cogentcore/webgpu`, `purego`, `gocudrv`, `aikit/gpu`, or the three submodules. The CI cleanliness
guard now runs BOTH `go list -deps` (compiled set) AND `go list -m all` (module graph a consumer
resolves), so a future root-package blank-import of a submodule fails CI. `build-embed.sh` stages
the `//go:embed` asset into `internal/chatapp`; README's "Running on a GPU" + the M-18 wired-serve
CI steps point at the new `<submodule>/cmd/*` paths. Verified: root pure-Go build + `go vet` +
staticcheck + `-race ./...` green; cuda/gpu entrypoints build under the workspace; metal entrypoints
cross-compile darwin/arm64. Original finding below.
**M-19 (was OPEN)** [linux] | root `go.mod:14-18` — the pure-Go CPU consumer pulled the whole GPU
dependency set. Because `cmd/serve/{gpu,cuda,metal}.go` (and `demo/chat`, `demo/gemma`) blank-imported
the submodules under build tags, the root required all three, and therefore `cogentcore/webgpu`,
`purego`, `gocudrv` and `aikit/gpu` as indirects. The old CI guard checked only `go list -deps`, which
sees imported packages and can't catch a module-graph leak.
*Note:* release-strategy consequence — the submodule `cmd/*` entrypoints import `internal/…app` from
the root, so a published accelerated build needs the goinfer root tagged with those packages (the
existing `replace … => ../` covers local clones; `go install …/cuda/cmd/serve@latest` needs the tag).

**M-21 — FIXED** (2026-08-07, Linux box; maintainer chose the breaking change). `context.Context` is
now the first parameter of `decoder.StreamTranscodeGGUF` (public) and `internal/prequant`'s
`Transcode`/`transcodeDir`/`EnsureCachedGIW` (internal). Cancellation is enforced by wrapping the
sink in a `ctxWriter` that returns `ctx.Err()` before each per-layer write, so a cancelled context
aborts the streaming transcode at the next layer boundary (and the caller removes the partial
`.giw`) rather than running the whole multi-GB pass. Wired: `cmd/prequant` installs a
`signal.NotifyContext` (Ctrl-C / SIGTERM) so the CLI aborts cleanly; serve's admin load passes the
request context (a disconnected `POST /admin/models/load` cancels), while startup load passes
`context.Background()` (a Ctrl-C there already ends the process). Gate
`TestStreamTranscode_ctxCancel_M21` (pre-cancelled ctx → `context.Canceled`, zero bytes written);
existing round-trip/byte-identity tests unchanged. Breaking to the one exported func — pre-1.0
(v0.9.x), so a minor bump with a CHANGELOG note.
*Original finding:* no `context.Context` on the minutes-long quantize-and-write passes.

**M-22 — FIXED** (2026-08-07, Linux box; maintainer chose the rename). `cuda.SpecStats` →
`cuda.GPUSpecStats`, so the GPU batched-verify counters no longer share a name with
`decoder.SpecStats` (the CPU-spec counters) — a telemetry reader can't silently conflate the two
distinct field sets. Both types now carry reciprocal doc comments mapping the (non-1:1) field
correspondence (`Emitted`≈`Generated`, `Evaluated`≈`VerifyToks`, etc.). No external consumer existed,
so the break is contained to the `cuda` module. Breaking to `cuda.GPUSpecStats` users (none today) —
CHANGELOG note.
*Original finding:* `cuda.SpecStats` and `decoder.SpecStats` were different types with the same name.

---

## §5 — Minor (all open)

All 24 minors are unworked. None are release-blocking. `[mac]` = metal.

- **N-01** [linux] `decoder/session.go:168,188` — `genSpec` rewinds unconditionally where `Generate`
  skips an empty prompt, so a rejected empty-prompt speculative call destroys a warm session's KV.
- **N-02** [linux] `decoder/mlp.go:210` — `groupLimit`'s integer truncation leaves trailing experts
  at `0.0` instead of `-Inf` when `NumExperts % NGroup != 0` (latent; real DeepSeek divides evenly).
- **N-03** [linux] `decoder/gguf.go:1108` — backend-fallback note goes to **stdout**, contaminating
  piped token streams; every other load path uses stderr.
- **N-04** [linux] `cuda/prefill.go:334` — `prefillCore` returns `r.launchErr` without clearing it
  first (unlike `launchToken`); stale errors are re-reported and `PrefillLast` is not idempotent
  w.r.t. prior decode state.
- **N-05** [linux] `gpu/decoderunner.go:343` — activation scratch sized with `padK` (mult of 16)
  while W4A8 kernels index to `padK32`; unreachable at today's dims, but `uploadProj` supports
  `K % 32 != 0`.
- **N-06** [linux] `gpu/backend.go:149` — `b.fallbacks++` outside the lock; benign today, trips
  `-race`.
- **N-07** [linux] `gpu/decodetoken_fused.go:80` — dispatches 64 workgroups for a single-workgroup
  rmsnorm kernel; 64× redundant work and 64 unsynchronised writes to the same addresses.
- **N-08** [linux] `gpu/mamba.go:93` — `array<f32, 8>` indexed by `DConv-1` straight from config;
  `conv_kernel > 9` overruns, `== 1` underflows a u32 index.
- **N-09** [mac] `metal/model.go:350,352` — `pGemvBias`/`pSAAmax` created and never dispatched;
  `ForwardArgmax` unreachable from production. The unused `gemv_w4a8_sa_amax` is also the variant
  most likely to be wired next while still missing the `row >= N` guard.
- **N-10** [mac] `metal/snapshot_golden_test.go:56` — no `m.Close()`/`r.Close()`, so the test can't
  catch the Close-path regressions `close_leak_test.go` exists for.
- **N-11** [mac] `metal/model.go:758` — the `Close` doc (incl. idempotency guarantee) is attached to
  the unexported `slotBuffers`; `go doc metal.Resident.Close` renders empty.
- **N-12** [linux] `decoder/features_test.go:147` — the `known` map omits `FeatAttnSink` and
  `FeatGemma4EModel`, so adding either legitimately fails with a misleading "unknown feature" error.
- **N-13** [linux] `decoder/layerpaging.go:47` — a pager is built and a banner printed for
  own-forward families (gemma4, nemotron) whose loops never call `enterLayer`, so the advertised RAM
  bound isn't delivered.
- **N-14** [linux] `cmd/serve/openai.go:467` — the 151936-entry token→bytes table is rebuilt on
  every constrained request, before the queue gate.
- **N-15** [linux] `cmd/serve/responses.go:170` — `/v1/responses` always reports
  `status: "completed"`, even when cut off by `max_output_tokens`. (Related to the M-04 class.)
- **N-16** [linux] `cmd/serve/openai.go:282` — a request with both `tools` and an image silently
  drops the tools; `serveVisionChat` neither renders nor parses them.
- **N-17** [linux] `internal/prequant/ggufmeta.go:44` — the tensor-dimension loop doesn't test
  `c.err`; `n_dims = 0xFFFFFFFF` spins ~4.29e9 no-op reads.
- **N-18** [linux] `internal/prequant/ggufmeta.go:119` — `skipValue` recurses without a depth limit;
  12 bytes per level against a 64 MiB prefix is ~5.6M levels of stack growth.
- **N-19** [linux] `multimodal/projector.go:106` — `kernel := grid / tps` unchecked for zero; a
  config with `mm_tokens_per_image` larger than the grid makes every image embedding NaN, silently.
- **N-20** [linux] `tokenizer/gguf.go:134` — `merges = nil` swallows every `ggufStringArray`
  failure, so a corrupt merges array is indistinguishable from an absent one and surfaces later as a
  misleading "decode-only vocab" error. (This is the class behind the gemma3-4b tokenizer report.)
- **N-21** [linux] `scripts/pin_gemma4_forward.py:13` — writes to `~/tmcode/...` while every sibling
  writes `~/mycode/...`; `makedirs` silently creates the wrong tree and the golden never lands where
  the test looks.
- **N-22** [linux] `scripts/gen_chat_goldens.py:11` + `gen_tool_goldens.py:11` — hardcoded home path
  and five HF repos pulled with no `revision=` pin, so an upstream `chat_template` edit silently
  changes committed byte-exact goldens.
- **N-23** [linux] `demo/agent/go.mod:7` — pins `go 1.26.3` while the other four pin `go 1.26.5`,
  with no `toolchain` line; anyone on 1.26.0–1.26.4 is forced into a toolchain download.
- **N-24** [mac] `metal/model.go:579` — see G-04; also leaves the `slots` arm's pinned set
  unrecorded for teardown accounting.
