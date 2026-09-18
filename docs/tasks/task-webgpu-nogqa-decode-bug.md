# Task: find the actual bug behind WebGPU's no-GQA resident-decode divergence

> **Status: OPEN, filed 2026-09-18.** Interim safety fix shipped (gpu/residency.go declines the
> broken case rather than serving it); the actual kernel bug is NOT found. This doc exists so the
> next pass does not repeat the eliminations already done here.
>
> **Second pass, same day (Cowork, section directly below): no kernel bug reproduced.** The
> shipped kernels and plan match an oracle at cosine 1.000000 on phi3-mini's exact shape, the
> production path on a phi3-shaped synthetic checkpoint reproduces the first pass's numbers, and the
> CPU forward reproduces them against itself under f32-rounding-sized noise. The two real-checkpoint
> confirmation runs are listed there; the guard comes out when they pass. The first pass's record
> (from "What triggered this" on) is kept unchanged below it.

## Second pass, 2026-09-18 (Cowork): no kernel bug reproduced — the evidence is the quantized forward's own sensitivity

> **Status after this pass: the kernel-bug hypothesis is refuted for the SHAPE and explained for the
> VALUES; the guard should come out once the two real-checkpoint confirmation runs below pass.**
> Everything above this section is the first pass's elimination trail and still stands as recorded;
> what changed is the reading of the two numbers it could not explain.

### How this was run

The `gpu` module was built and run in a cloud sandbox with no GPU at all: Mesa lavapipe (llvmpipe,
Vulkan 1.3, the same software adapter CI's headless runner uses), wgpu-native's Vulkan backend,
Go 1.27.1, tree `6bb16fca` plus the instrumentation listed under "Tooling landed". Nothing in
`gpu/` was changed except the additions below; the kernels and the dispatch plan are the shipped
ones. The real phi3-mini checkpoint cannot be fetched from that sandbox, so every real-checkpoint
number in this section is from the first pass; every synthetic number is new and reproducible
with `scripts/mk_phi3_synth.py`.

### Result 1 — the kernels and the plan are correct for phi3-mini's exact shape

`gpu/geom_mha_decoderunner_test.go` (`TestDecodeRunnerW4A8_geometries`) is
`TestDecodeRunnerW4A8_parity` re-run over geometries no synthetic test had ever used (all of them
used qwen2.5-1.5B's `(nH,nKV,hd) = (12,2,128)`), through the production `newDecodeRunner` plan —
`rmsQuant` → W4A8 q/k/v → `qkvFinalize` → the key-split attention kernel → `quant` → o-proj
`gemvAdd` → `rmsQuant` → gate/up → `swigluQuant` → down `gemvAdd` → final norm → LM head —
against the same int4/int8 oracle, at position 0 (one key, attention is the identity on V) and
position 20:

| geometry | hidden | nH | nKV | hd | inter | vocab | pos 0 | pos 20 |
|---|---|---|---|---|---|---|---|---|
| qwen2.5-1.5b (control) | 1536 | 12 | 2 | 128 | 8960 | 4096 | cosine 1.000000, maxAbs 7.6e-6 | 1.000000, 7.6e-6 |
| **phi3-mini** | **3072** | **32** | **32** | **96** | **8192** | **32064** | **1.000000, 1.5e-5** | **1.000000, 7.6e-6** |
| MHA only (group=1, hd 128) | 1536 | 12 | 12 | 128 | 8960 | 4096 | 1.000000, 3.8e-6 | 1.000000, 7.6e-6 |
| hd=96 only (GQA 12/2) | 1536 | 12 | 2 | 96 | 8960 | 4096 | 1.000000, 3.8e-6 | 1.000000, 1.5e-5 |

That closes suspects 1–3 of "where to look next": the int4 GEMV at `[3072,3072]`/`[8192,3072]`/
`[3072,8192]`/`[32064,3072]`, the attention kernel at `group = 1` and `hd = 96`, and the
o-proj/MLP chain all match the oracle to f32 rounding. The argmax agrees in every cell.

### Result 2 — the production path on a phi3-shaped checkpoint reproduces the first pass's NUMBERS, and per-layer capture shows what they are

`scripts/mk_phi3_synth.py` writes a random `Phi3ForCausalLM` safetensors with phi3-mini-4k's exact
per-layer dims (fused `qkv_proj`/`gate_up_proj`, untied head, 4 layers). `gpu/resident_capture_parity_test.go`
(`TestResidentCaptureParityWebGPU`) loads it exactly the way the first pass loaded the real model —
`decoder.Load` with `Backend:"webgpu", Quant:"int4"` (so `BuildResident`, the `UploadW4A8Packed`
fast path and the `fusedSplit` are all the real ones) against `decoder.Load` with `Quant:"int4"` —
and runs the same `[1 7 42 100 5 200 13 88]` prompt plus a greedy continuation:

| comparison | prompt cosines (8 positions) | notes |
|---|---|---|
| resident int4 vs CPU int4, as the first pass ran it | 0.9982 – 0.9991 | argmax flip at position 6 (cpu 8317 / gpu 24969) |
| same, with `GOINFER_INT4_F16_SCALES=1` (both sides carry the f16 group scales the GPU stores) | 0.9993 – 0.9996 | the only single-variable form of the comparison |
| resident W8A8 (`Quant:""` → int8 rows at upload) vs CPU **f32** — the first pass's "f32" run | 0.9963 – 0.9985 | greedy flip at step 2, then cosines 0.25 |
| same, on a copy with a crude planted "massive activation" channel | 0.966 – 0.999 | argmax flip at position 4 |

So a 4-layer *random* phi3-shaped checkpoint with no outliers already sits below this repo's 0.999
bar on the exact test the first pass used, and the "f32" comparison sits below it by more. With
`GOINFER_GPU_CAPTURE=1` the same test differences the two forwards per sublayer (the runner's new
capture buffers vs `decoder.ForwardSubCaptureLogitsForTest`), relative L2 error `‖gpu−cpu‖/‖cpu‖`,
f16 scales matched, position 0:

| layer | attention context (pre o-proj) | attention contribution | MLP contribution |
|---|---|---|---|
| 0 | 3.5e-7 | 2.0e-7 | 4.2e-7 |
| 1 | 1.3e-7 | 5.0e-4 | 1.5e-2 |
| 2 | 1.4e-2 | 1.8e-2 | 3.5e-2 |
| 3 | 2.1e-2 | 2.5e-2 | 4.4e-2 |

Layer 0 — identical input on both sides — agrees to f32 rounding across all three sublayers. At
layer 1 the attention *context* still agrees to 1.3e-7 while the attention *contribution* (the
o-proj output) is off by 5e-4 and the MLP contribution by 1.5e-2. The only operations between
those points are `quant(ctx)` and the o-proj GEMV, then `rmsQuant`, gate/up, `swigluQuant`, down.
That is the int8 activation re-quantization acting on a dense ~1e-7 difference: each quantizer
flips the rounding decisions that sit within the perturbation of a half-step, every flip is a
whole step, the next GEMV turns the flips into a dense perturbation roughly the square root of a
step larger, and the next quantizer sees that. One flip in a 3072-wide int8 row is ~1e-3 of a GEMV
output; a 1e-3 dense perturbation flips ~hundreds of the next 8192-wide row; that is the 1.5e-2.
From there it compounds per layer and saturates near the quantization noise itself.

### Result 3 — the control: the CPU forward against ITSELF shows the same profile

`gpu/cpu_quant_sensitivity_test.go` (`TestCPUQuantSensitivity`) uses no GPU. It runs the CPU int4
forward twice on the same checkpoint, the second time with every f32 norm vector nudged by −1/0/+1
ULP per element (`GOINFER_NORM_ULP_NOISE`, `decoder/normnoise.go`: f32-rounding-sized noise, dense,
which is what two correct implementations that reduce in different orders look like to each other):

| | pos 0 | pos 1 | pos 2 | pos 3 | pos 4 | pos 5 | pos 6 | pos 7 |
|---|---|---|---|---|---|---|---|---|
| CPU int4 vs CPU int4 + ULP noise (phi3-shaped, 4 layers) | 0.99956 | 0.99952 | 0.99950 | 0.99940 | 0.99946 | 0.99946 | 0.99946 | 0.99943 |
| argmax | = | = | = | = | = | = | **8317 / 24969** | = |

Same magnitude as resident-vs-CPU, the same per-layer profile (layer 0 to 1e-7, layer 1's MLP
contribution 2.3e-2, layer 3's 5e-2 — the same columns as the table above), and the argmax flips
at the same position to the same token the GPU produced. Whether the noise is absorbed or amplified
is itself a coin toss on where the activations sit relative to rounding boundaries: a second seed
of the same noise, and a single-element nudge, both came back bit-identical or within one flip —
which is why a comparison that happens to land clean on one checkpoint proves nothing about the
next. A 4-layer GQA control with qwen2.5-1.5B's attention geometry (`(12,2,128)`, separate q/k/v,
llama family) behaves the same way: resident-vs-CPU 0.9997 with one flip seeding layer 1 at
position 1, CPU-vs-CPU-noise 0.9977–0.9984 with an argmax flip at position 2. **This is not a
no-GQA property and not a phi3 property. It is what an int4/int8 forward with per-row int8
activations does on any checkpoint wide and deep enough, and 32 layers of a real model with real
outlier channels is the far end of it.**

### What this does to the first pass's evidence

- **0.94–0.99 prompt cosines, int4 vs int4.** Within what the CPU shows against itself on a
  4-layer synthetic (0.9994) extrapolated over 32 layers and real activations; the f16-vs-f32
  scale representation the comparison never matched costs another ~1e-3 on its own.
- **Greedy generation diverging at step 0–1 and going cosine-negative.** Once either side picks a
  different argmax, the two logit vectors are for different contexts and their cosine measures
  nothing; the synthetic runs reproduce the exact signature (0.999986 one step, 0.28 the next) off
  a near-tie flip. The random-token prompt makes near-ties the norm.
- **The "f32" path diverging "differently", worse at position 0 (0.358).** `Quant:""` on the
  resident path is W8A8 — `uploadProj` quantizes every f32 projection to int8 rows at upload and
  the activations are int8 per row throughout — compared against an unquantized f32 CPU forward.
  That is not a second code path agreeing on a bug, it is the int8 activation quantization error
  itself, and per-token int8 activations are known to be worst exactly at the first token of a
  real LLM (the massive-activation position). The synthetic W8A8-vs-f32 run is at 0.996 with no
  outliers and 0.966 with a crude one.
- **Position 0 diverging "where RoPE and softmax are inert".** Correct observation, and the
  geometry gate now confirms the position-0 chain (norm, V, o-proj, MLP, head) is exact on this
  shape; the position-0 numbers on the real model are the value-dependent effects above.
- **"Bit-for-bit identical" with the separate rope/ropeStore/vStore kernels.** Expected: those
  kernels are documented bit-identical to `qkvFinalize`, so the experiment could not distinguish
  anything either way.

### What still has to be run, on the real checkpoint (RTX box or the Mac)

Both are cheap and both are decisive; neither was possible from the sandbox.

1. **Text.** Lift the guard with `GOINFER_WEBGPU_ALLOW_NOGQA=1`, run `goinfer-chat` on
   `Phi-3-mini-4k-instruct-q4.gguf` with `--backend webgpu` int4, greedy, on a real prompt, and read
   the output beside the CPU run's. Coherent, same-or-near-same text ⇒ there is no bug to find.
   Garbage ⇒ there is, and (2) localises it.
2. **Per-layer capture on the real model.**
   `GOINFER_WEBGPU_ALLOW_NOGQA=1 GOINFER_INT4_F16_SCALES=1 GOINFER_GPU_CAPTURE=1
   GOINFER_PARITY_CKPT=~/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf go test -tags
   'gpu goinfer_testhooks' ./gpu/ -run TestResidentCaptureParityWebGPU -v`, then
   `GOINFER_PARITY_CKPT=… -run TestCPUQuantSensitivity -v` for the floor. Expected if there is no
   bug: layer 0 at ~1e-7 in all three columns, then growth by layer with the CPU-vs-CPU-noise run
   showing the same profile. A bug looks different: one layer/sublayer whose input still agrees to
   ~1e-7 while its own contribution jumps to O(1) — and the column names which kernel.
3. Then delete the `nKV == nH` decline in `gpu/residency.go` (and its escape hatch). As written the
   guard also declines every future MHA family on WebGPU (`decoder/registry.go`'s notes list several
   real ones), for a divergence that the evidence now says is the comparison, not the kernel.

### What a resident parity gate should be for a deep real checkpoint

The 0.999 floor in `gpu/gemma3_resident_parity_test.go` was measured on a tiny fixture where the
cascade has nowhere to compound; it is a real floor for that fixture and no floor at all for a
32-layer model. For a real checkpoint: match the scale representation (`GOINFER_INT4_F16_SCALES=1`
on the CPU side); difference per sublayer with relative error, not final-logit cosine; run
`TestCPUQuantSensitivity` on the same checkpoint and treat its number as the resolving power of
any cosine on it; and judge fidelity the way the prefill gate already does (`docs/completed/task-prefill-gap.md` §3.2 — argmax agreement / KL against the **f32** CPU reference over a real
prompt, pooled), where the resident int4 path and the CPU int4 path are two arms and neither is
the reference. Never read W8A8-vs-f32 at a first token as a code-path signal.

### Tooling landed (uncommitted on the Mac by Cowork; gofmt clean, `go vet` clean under `gpu`, `gpu goinfer_testhooks`, and untagged; staticcheck not run — the sandbox cannot fetch it)

- `gpu/decoderunner.go` — `GOINFER_GPU_CAPTURE=1` records a kv-store copy of the attention context,
  the post-attention residual and the post-MLP residual per layer into build-once capture buffers
  (`DecodeRunner.ReadCapture`). Nothing is allocated or dispatched when unset.
- `decoder/testhooks.go` — `ForwardSubCaptureLogitsForTest`: `ForwardSubCapture` plus the logits
  from the same forward (running the token twice would append it to the KV cache twice).
- `decoder/normnoise.go` + one line in `withResidency` — the `GOINFER_NORM_ULP_NOISE=<seed>`
  diagnostic (copies, never in place: a norm can alias a read-only mmap).
- `gpu/residency.go` — `GOINFER_WEBGPU_ALLOW_NOGQA=1` bypasses the decline for the runs above.
- `gpu/geom_mha_decoderunner_test.go`, `gpu/resident_capture_parity_test.go`,
  `gpu/cpu_quant_sensitivity_test.go` — the three tests above; the last two are env-gated on
  `GOINFER_PARITY_CKPT` and skip otherwise. `scripts/mk_phi3_synth.py` writes the checkpoint they
  were developed on (`python3 scripts/mk_phi3_synth.py /tmp/phi3-synth 4`; `--outlier` for the
  planted channel).
- Full `gpu` suite on the software adapter with these changes: 62 pass / 108 skip (hardware-only
  and asset-gated) / 0 fail.

## What triggered this

Benchmarking phi3-mini on WebGPU (`docs/benchmarks.md`'s peer-matrix re-run) found it running at
CPU-equivalent speed (~8.4 tok/s) despite the `webgpu` backend flag. Its `/health` endpoint showed
why: `BuildResident declined ... residency KV alloc (layer 13): ... Not enough memory left`,
falling back to the staged (host-prefill) path.

Root cause of *that*: `gpu/residency.go`'s `ctxCap` used a fixed per-precision ceiling
(`decoder.WebGPUCtxCeiling`, 16384 positions at f32) with no clamp to the model's own
`max_position_embeddings`. Every WebGPU-resident model tested before this used real GQA
(`head_count_kv < head_count`), so its real KV-cache bytes/position were small enough that 16384
positions always fit an 8 GB card — the ceiling's own comment calls this "the proven 8 GB fit".
Phi-3-mini has **no GQA at all** (`head_count_kv == head_count == 32`, confirmed directly from the
GGUF header, not assumed) — its real per-position KV cost is ~6x a typical GQA model's, and 16384
positions is 4x its own 4096-token native window besides. The allocation loop ran out of VRAM
partway through (layer 13 of 32) trying to reserve KV for positions the model could never even
serve.

**Fixed** (`gpu/residency.go`): clamp `ctxCap` to `m.Config().MaxPositions` before anything else,
mirroring `cuda/resident.go`'s `resolveCtxCap`, which already does exactly this. Verified: at its
own 4096-position ceiling phi3-mini's real KV need is ~3.1 GB, which fits an 8 GB card easily
beside its ~2.3 GB of int4 weights — the model now builds resident (`webgpu:vulkan-resident`)
instead of declining on VRAM.

## The bug this uncovered

Letting phi3-mini actually reach WebGPU residency exposed a **second, unrelated, and more
serious** problem: the resident decode path's output does not match CPU.

Direct per-position logit comparison (`rf.Forward` vs `mcpu.ForwardForTest`, both int4, fixed
arbitrary token prompt `[1 7 42 100 5 200 13 88]`):

- Prompt-position cosine similarity: **0.94–0.99** across all 8 positions. This repo's own bar
  for "the resident kernel is correct" is 0.999+ (`gpu/gemma3_resident_parity_test.go`'s own
  gate). Every other WebGPU-resident model this session re-checked (Gemma3, gpt-oss, Nemotron
  MoE, LoRA) landed at 0.9998–1.0 on the same style of test.
- Greedy continuation (each side fed its own argmax back in) diverges at generation step 0-1 and
  goes **cosine-negative** by step 2 (-0.02) and again at steps 4, 6, 9 (-0.28, -0.43, -0.12) —
  the two logit vectors point in nearly opposite directions. That is not quantization noise
  (which stays close to 1.0); it is a structural computation difference.

## What was ruled out (do not re-check these first)

- **Not the ctxCap fix itself.** The divergence is present with the fix applied and absent only
  in the sense that the model can no longer reach residency at all without it — the fix is
  necessary to even observe the bug, not the cause of it.
- **Not the `qkvFinalize` fused dispatch.** Forcing the separate `rope`/`ropeStore`/`vStore`
  kernels instead of the fused one (`gpu/decoderunner.go`'s `if m.kvF16 || m.kvI8 { ... } else {
  qkvFinalize(...) }` — temporarily changed to always take the separate-kernel branch) reproduced
  **bit-for-bit identical** wrong output (same cosines, same tokens, to the decimal). Whatever is
  wrong is either upstream of both (Q/K/V projection) or in something both paths share (the
  attention kernel, the KV cache itself, or the RoPE frequency table).
- **Not solely an int4-quantization artifact.** Re-run with `Quant: ""` (full f32, `decode_path:
  "webgpu:vulkan-resident (native)"` — a genuinely different code path from int4) still diverges,
  but with a *different* pattern: worse at position 0 (cosine 0.358, vs int4's 0.988) but greedy
  generation stays correct for 4 steps before diverging (vs int4's divergence at step 0-1). Two
  different wrong patterns across two different precision code paths suggests either two
  compounding bugs, or one bug whose visibility depends on precision-path timing/ordering — not a
  single simple off-by-one that a naive "make it match" patch would fix blind.
- **Not the fused-QKV weight split.** `decoder/weights.go`'s `buildPhi3Weights` splits
  `self_attn.qkv_proj.weight` into Q/K/V by fixed, index-based row ranges (`qkv[0:qDim*hidden]`,
  `qkv[qDim*hidden:(qDim+kvDim)*hidden]`, `qkv[(qDim+kvDim)*hidden:(qDim+2*kvDim)*hidden]`) with no
  size-dependent branching. This is **shared** CPU/GPU code, and CPU is correct — if this split
  were wrong, CPU would be wrong too.
- **Not the GQA head-mapping arithmetic.** `group := nH / nKV` and `kvh := qh / p.group`
  (`gpu/decodelayer.go:146`, `gpu/attention.go:118` and its three other copies) both evaluate
  correctly for phi3's `group = 32/32 = 1` (every query head maps to its own KV head — the
  identity case, not a divide-by-zero or off-by-one).
- **Not partial-rotary tail handling.** `qkvFinalizeShaderWGSL`'s `ktail` pass-through block (for
  GLM/some-Phi partial rotary) is a documented no-op for full rotary; phi3-mini-4k's own GGUF
  metadata confirms `rope.dimension_count: 96 == head_count * embedding_length/head_count` — full
  rotary, `ktail = 0`. This code path never fires for this checkpoint.
- **Softmax/multi-key aggregation is not implicated by position 0.** At position 0 there is
  exactly one cached key, so softmax over one element is 1.0 regardless of whether the attention
  weights are computed correctly (`CLAUDE.md`'s own documented minimal-repro trap, in reverse
  here: position 0 does NOT mask this bug — real divergence is present even where softmax is
  inert). Whatever's wrong runs before or independent of the multi-key softmax math.
- **RoPE rotation is inert at position 0 too** (`theta = pos * invFreq = 0`, so `cos=1, sin=0`,
  an identity transform) — yet position-0 divergence is still present in the f32/native path
  (cosine 0.358). So the bug is not purely "RoPE angle is wrong at pos > 0" either, though a
  RoPE-table or `AttnScale`/`RotaryDimResident` mismatch specific to hd=96 has not been fully
  ruled out for the int4 path specifically (only shown to be non-exclusive as the position-0
  cause in f32).

## What's still open — where to look next

Not found: the actual line. The likely remaining suspects, roughly in order of how cheap they'd
be to check with proper instrumentation:

1. **The int4 GEMV/dequant kernel for phi3's specific weight shapes.** `hidden=3072`, Q/K/V/O all
   `[3072, 3072]` (unlike every other tested model, where Q is strictly wider than K/V) — check
   whether the int4 quantization group layout or the dequant kernel has an assumption keyed on
   relative Q/K/V sizes.
2. **The attention kernel (`attnShaderWGSL`) at `group=1` specifically**, even though the
   arithmetic reads correctly on paper — build a *pure* kernel-level unit test (no full model)
   with `nH == nKV` and a known Go-computed reference, the way `attnbatched_test.go` already does
   for `(nH,nKV,hd) = (8,4,64)` and `(4,1,50)` but never `(n,n,h)`.
3. **O-projection / MLP**, downstream of attention — not yet isolated at all.
4. **Build the per-layer hidden-state capture this class of bug needs.** This repo's own
   CLAUDE.md is explicit that this class of divergence should be chased "per layer, not from
   final logits" — that tooling does not exist for the WebGPU backend today (it does, in a
   test-only form, for CUDA's MLA work — `cuda/mla_resident_test.go`'s per-position cosine check
   is the shape to copy). Building it properly (dump Q/K/V/attn-output/o-proj-output per layer,
   not just the final 32-layer-deep logits) is very likely the fastest real path to the actual
   line, not more WGSL reading.

## The interim fix, and its blast radius

`gpu/residency.go`, two independent changes:

1. **`ctxCap` clamps to `m.Config().MaxPositions`** before the ceiling/explicit-request clamps.
   Unconditionally beneficial — stops wasting VRAM reserving positions no model could ever serve.
   Verified inert for models whose native window already exceeds the ceiling (Qwen2.5-1.5B:
   32768-token window, ctxCap stays at the 16384 ceiling as before).
2. **`BuildResident` declines when `nKV == nH` (no GQA at all)**, SCOPED to exclude MLA
   (DeepSeek/Kimi), Qwen3.5/DeltaNet and Nemotron — those take entirely separate attention code
   paths further down this function and their own `nKV`/`nH` can coincide for reasons unrelated
   to GQA grouping (`deepseek-tiny` reports `nKV==nH==4`; an unscoped first version of this guard
   broke `TestMLAResidency_matchesCPU`, caught by the full `gpu` test suite before it shipped).

Verified on real hardware (RTX 2070 SUPER): phi3-mini now declines cleanly with an honest reason
(`no-GQA attention (kv heads == query heads, 32 == 32) hits an unresolved WebGPU resident decode
divergence`) instead of either the old misleading OOM message or silently-wrong fast output.
Qwen2.5-1.5B (real GQA) still goes resident and generates correctly, unaffected. Full `gpu` module
test suite (`go test -tags 'gpu goinfer_testhooks' .`, non-heavy): 123 pass, 0 fail — includes
Gemma3/gpt-oss/Nemotron-MoE/LoRA/QKNorm resident-parity gates at their existing 0.999+ floors and
the MLA parity test.

**What this does NOT fix**: phi3-mini and Phi-4 (and any other future no-GQA checkpoint) stay on
the CPU-staged WebGPU path — slow, but correct — until whoever picks this up finds the real bug
and removes the `nKV == nH` guard. `docs/hardware-matrix.md`'s "✅ resident" for Phi-3/Phi-4 on
WebGPU is unaffected by this change (it reflects feature *admission*, checked with no device
present, not a real `BuildResident` attempt — the same admission/runtime-fit distinction this
repo's MLA nGroup/topkGroup trap already documents at `decoder/features.go`).

<!-- doc-reviewed: 2026-09-18 -->
