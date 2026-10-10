# decoder: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `decoder`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

## ResidentFeature

Moved from `decoder/features.go` (the comment above `ResidentFeature`) on 2026-10-10.

```text
ResidentFeature is one architecture capability a resident (GPU) decode path must implement
in order to run a model CORRECTLY.

The failure mode this taxonomy exists to prevent is SILENT. A backend that admits a model
needing a feature it has not implemented raises no error — it simply drops the feature and
emits wrong logits (docs/metal-model-coverage.md found exactly this: Qwen3 running with
QK-norm ignored, Mistral running full-attention past its window). Admission must therefore
be a subset check: RequiredResidentFeatures(model) ⊆ the backend's implemented set, else
decline to the staged/CPU path.

Requirements are DERIVED from the loaded Architecture's own flags — never a hand-maintained
per-arch list. That is the point: a newly registered arch is classified automatically, and a
backend that has not implemented its features declines by default instead of mis-running.
Adding a field to Architecture that changes the math means adding it here too; the
registry-driven test (features_test.go) is what makes forgetting expensive.

Note this layer sits ABOVE the arch flags, so it cannot catch an arch that fails to CLAIM a
feature it needs — e.g. phi3Architecture once dropped sliding_window entirely, which made
every path (CPU included) silently wrong. That class is a registry bug, caught by
per-family parity, not by admission.
```

## FeatAttnSink

Moved from `decoder/features.go` (the comment above `FeatAttnSink`) on 2026-10-10.

```text
FeatAttnSink bundles gpt-oss's THREE departures: the learned per-head sink in the softmax
denominator, the clamped interleaved-SwiGLU expert, and a router whose bias reaches the
WEIGHT rather than only the selection. No resident backend DECLARES it yet, so CUDA/Metal/
WebGPU all decline (TestGptOss_cudaWebgpuDecline).

Metal HAS all three as end-to-end PROVEN kernels (2026-08-18: kernels.go's `attention`
sink term, moe.go's route_gptoss/swiglu_quant_gptoss/gemv_w4a8_moe_wacc_bias, the moe.go
isGptOss dispatch split — TestGptOssResidentParity, 8/8 argmax-exact, min cosine 0.9989 on
the tiny fixture) but still does not declare this: gpt-oss also needs FeatRopeMscale
(YaRN), which is held pending Mellum's own end-to-end gate — see the FeatRopeMscale note on
metal's residentBackendFeatures entry. Declaring FeatAttnSink alone would still correctly
decline gpt-oss via the missing FeatRopeMscale check, but the two are declared together
once Mellum unblocks it, so a reader sees one coherent "gpt-oss shipped" commit rather than
a half-declared feature set.

CUDA has all three as gated KERNELS (cuda/gptoss_act.cu, plus the sink argument on both
attention kernels) and does not declare this either — but NOT for the reason this comment
used to give. It said the kernels were "LOADED but never DISPATCHED into a forward pass",
which the code contradicts: sinkArg is threaded into BOTH attention launches
(cuda/resident.go: the split-KV launch and the decode launch) and launchGluSplitExpert is dispatched from
the MoE expert loop (cuda/resident.go, both the decode and the prefill MoE paths). The bridge is WRITTEN. What has never happened is a
real gpt-oss forward EXECUTING it, because the 20b MXFP4 checkpoint (~13.8 GB) does not fit
the CUDA box's 8 GB VRAM without the host<->VRAM MoE-streaming path. Written-but-unexercised
is a different, smaller gap than not-yet-attempted, and the distinction is the estimate.

The genuinely missing CUDA piece is FeatOutBias: no o_proj-bias kernel and no wiring exists
there at all (grep OBias/out_bias across cuda/*.go). Kernel-level parity is still not
end-to-end parity — 2224441 declared on kernel evidence and was correctly reverted, which
is why neither flag is set here. WebGPU has none of this.
```

The trailing comment on the `FeatAttnSink` constant read:

```text
see above: CPU-only until FeatOutBias exists on CUDA and ONE real gpt-oss forward has run resident
```

## FeatAttnTemp

Moved from `decoder/features.go` (the comment above `FeatAttnTemp`) on 2026-10-10.

```text
FeatAttnTemp (Ministral 3, batch 2 G3): the Llama4-style attn-temp query scale
(AttnTempBeta/AttnTempOrigMaxPos) was new to the GENERIC forward path (decoder/attention.go,
decoder/forwardn.go) when this feature was added — admitting mistral3 to a resident path
that didn't apply it would silently drop the scale for every position past
original_max_position_embeddings, producing plausible-but-wrong logits at exactly the
context lengths the mechanism exists for. G5 (docs/tasks/task-gpu-paths-2026-09.md): cuda and
metal now apply it (Model.AttnTempScale/AttnTempParams, folded into the existing rope
launch's Q output rather than a new kernel) and declare this. WebGPU still declines —
otherwise a plain GQA+YaRN model needing nothing else any backend lacks, so without this
declaration it would be ADMITTED there too and mis-run rather than correctly declined.
```

## FeatPostOnlyNorm

Moved from `decoder/features.go` (the comment above `FeatPostOnlyNorm`) on 2026-10-10.

```text
FeatPostOnlyNorm (Olmo 3/Olmo Hybrid, batch 2 G2): NormPostOnly — no pre-norm at all, the
sublayer's OUTPUT is normalized before the residual add. Genuinely different from
FeatSandwichNorm (which normalizes BOTH the input and the output). G5
(docs/tasks/task-gpu-paths-2026-09.md): cuda and metal now implement it (quant_vec on the raw
residual in place of the pre-norm dispatch; the sandwich post-norm dispatch already both
backends ship widens to cover this placement too) and declare it.
```

## FeatQKNormWhole

Moved from `decoder/features.go` (the comment above `FeatQKNormWhole`) on 2026-10-10.

```text
FeatQKNormWhole (Olmo 3/Olmo Hybrid): QK-norm computed over the FULL projected q/k vector
(one RMSNorm over num_heads*head_dim) rather than per-head — verified against the real
modeling_olmo3.py (`Olmo3RMSNorm(config.num_attention_heads * self.head_dim, ...)`), not
the standard per-head FeatQKNorm (Qwen3/Gemma3/Mellum). Different statistic AND a
differently-shaped weight tensor, so it is its own feature, not a variant of FeatQKNorm. G5:
cuda and metal now implement it by reusing the existing per-head qk-norm kernel with its
grid collapsed to one whole-vector Q block and one whole-vector K block — valid only for
MHA (nH==nKV, true of both families that need this today); declared with that guard.
```

## FeatKDA

Moved from `decoder/features.go` (the comment above `FeatKDA`) on 2026-10-10.

```text
FeatKDA (Bailing Hybrid / Ling 3.0, batch 2 G5): Kimi Delta Attention's linear-attention
mixer — a delta-rule recurrence structurally identical to Gated DeltaNet (FeatDeltaNet) but
with a PER-CHANNEL decay (one value per row of the state matrix) where Gated DeltaNet's is a
single scalar per head — verified against fla-org/flash-linear-attention's actual source,
not the HF modeling file's opaque Triton-kernel call. A genuinely different recurrence, so
its own feature rather than a FeatDeltaNet variant; no resident backend implements it.
```

## FeatPairwiseRoPE

Moved from `decoder/features.go` (the comment above `FeatPairwiseRoPE`) on 2026-10-10.

```text
FeatPairwiseRoPE: GPT-J pairwise rotation (dims 2d, 2d+1) in the scalar decode/prefill rope —
Cohere/Command-R, Cohere2/Command-R7B, Aya (cohere), and GLM-OCR's text rows. Every GENERIC
resident rope kernel is the NeoX half-split rotation of pairs (d, d+half): cuda/glue.cu `rope`,
gemv_fwd.cu rope_kv, prefill_batched.cu rope_kv_batched, and metal/kernels.go `rope`. A resident
runner on those kernels is exact at position 0 (rotation is the identity there) and wrong at
every later one: no error, fluent-looking logits. MEASURED 2026-10-01 on the CUDA resident
(RTX 2070 SUPER, int4, resident vs CPU at the same quant, 48-token prompt): real Command-R7B
worst per-position cosine -0.075 and Aya-expanse-8B -0.041; Aya's greedy continuation matched
the HF golden 1/8 against the CPU's 8/8. Before this feature existed, cuda and metal declared
cohere/cohere2 anyway (their committed gates used flat 0.02-std fixtures, where a wrong rotation
is invisible: cohere-tiny at 0.25-std weights reads cosine 0.06 with the NeoX kernels against
0.9997 at 0.02). MLA (DeepSeek/Kimi) is NOT this feature: it carries its own interleave flag
into mla.cu and is gated by FeatMLA.

Declared by CUDA only (cuda/rope_pairwise.cu: rope_kv_pw, rope_kv_batched_pw,
rope_kv_mrope_batched_pw, gated by TestPairwiseRoPEResidentParityCUDA and the real-checkpoint
Aya/R7B gate). Metal and WebGPU do not declare it, so cohere/cohere2/glm_ocr DECLINE there to
the CPU path until pairwise kernels exist (docs/measurements/cuda-pairwise-rope-2026-10-01.md
says what to port): a decline that names the cause beats a resident that returns garbage.
```

## FeatPairwiseMRoPE

Moved from `decoder/features.go` (the comment above `FeatPairwiseMRoPE`) on 2026-10-10.

```text
FeatPairwiseMRoPE (GLM-OCR): pairwise rotation on a family that ALSO carries m-RoPE sections
(ropeInterleave with MRopeSection set), so the image-block prefill rotates each frequency by its
own (t,h,w) component through the pairwise pairs: the m-RoPE image-prefill kernels (cuda
rope_kv_mrope_batched) are NeoX too. glm_ocr needs this IN ADDITION to FeatPairwiseRoPE (its
text rows and decode go through the scalar kernel). First measured 2026-10-01 with admission
bypassed: glm-ocr-tiny resident vs CPU (int8int8, 48-token prompt) worst cosine -0.34, against
a NeoX llama control at 1.000000. CUDA declares it from rope_kv_mrope_batched_pw
(cuda/rope_pairwise.cu), gated by TestGlmOcrResidentParityCUDA; Metal and WebGPU do not.
```

## residentFeatures.softcap

Moved from `decoder/features.go` (the comment in `residentFeatures`, above the softcap features) on 2026-10-10.

```text
Split (9a-P2): the old single FeatLogitSoftcap conflated two different capabilities. The
attention-score softcap is a per-layer KERNEL; the final-logit softcap is one host-side
tanh after the LM head (like FeatEmbedScale's √hidden). Gemma 4 needs ONLY the latter, so
declining it for the former it does not use was over-broad. Backends declare each
separately — a backend that ships the host tanh but no attention-softcap kernel gets
FeatFinalLogitSoftcap alone.
```

## residentFeatures.layerNorm

Moved from `decoder/features.go` (the comment in `residentFeatures`, above the LayerNorm feature) on 2026-10-10.

```text
Mean-subtracting LayerNorm (Cohere/GPT-2) is a distinct kernel from RMSNorm;
no resident backend implements it, so carriers decline to CPU. GPT-2 already
carried the other GPT-2 declines (learned-pos, non-gated, out-bias); this
just makes the norm itself explicit.
```

## residentFeatures.gatedShared

Moved from `decoder/features.go` (the comment in `residentFeatures`, above FeatMoEGatedShared) on 2026-10-10.

```text
A sigmoid-GATED always-on shared expert (Qwen2-MoE: out += sigmoid(SharedGate·h)·shared(h))
is a distinct kernel from the ungated add (GLM/DeepSeek: out += shared(h)). CUDA implements
only the ungated combine and DECLINES the gated one (cuda/backend.go); Metal and WebGPU
implement both. Splitting it out moves that decline from a hand-coded backend check into the
shared taxonomy, so the hardware matrix matches admission.
```

## ResidentEligible

Moved from `decoder/features.go` (the comment above `ResidentEligible`) on 2026-10-10.

```text
ResidentEligible reports whether `backend` can run architecture `a` on its resident (GPU)
decode path — the CAPABILITY predicate, model-free (arch flags only). It is exactly the two
gates every resident admission already applies, composed in one place: the arch is a shape the
runner supports (decodeRunnerEligible) AND the backend implements every feature the arch needs
(the shared taxonomy). The runtime and the hardware-matrix generator both derive from these
same pieces, so the published table can never disagree with what a backend can run.

Scope note (deliberate, matches capability_matrix's GPUResident = decodeRunnerEligible): this is
arch-level CAPABILITY, not a runtime admission. The runtime additionally applies load-time
POLICY that a model-free predicate cannot know — the Nemotron int4-only / GOINFER_SSM_RESIDENT
precision gate (Model.DecodeRunnerEligible). Those are precision choices, not "can this backend
run this family", so the matrix shows capability and footnotes the policy.

N-94 (docs/audit-2026-09-10.md): "model-free (arch flags only)" is NOT true of granite. Unlike
Nemotron's policy gate above (which lives in the separate, model-level Model.DecodeRunnerEligible),
granite's admission gate lives INSIDE Architecture.decodeRunnerEligible() itself — the very
function this comment describes as arch-only — and reads GOINFER_SSM_RESIDENT (the model's
snapshot, or the live environment for an Architecture built without one, as the matrix
generator's are; see that function's granite case). The hardware-matrix generator pins this env var
empty (decoder/hardware_matrix_test.go), so the published table shows granite as CPU-only on
every backend, including WebGPU (which already declares FeatSSM) — not because no backend can
run it, but because this specific arch-level gate is still parity-bring-up-guarded off by
default. See docs/hardware-matrix.md's own footnote on this.
```

## residentBackendMoECap

Moved from `decoder/features.go` (the comment above `residentBackendMoECap`) on 2026-10-10.

```text
residentBackendMoECap is the router-kernel capacity of each backend whose MoE scoreboard is a
FIXED-SIZE array — the numeric twin of ResidentBackendFeatures (a feature is "implemented at
all"; a cap is "implemented up to N"). gpu/moe.go scores into array<f32,256> and its group-limited
path into array<f32,32>/array<bool,32>; cuda/backend.go rejects >256 identically. A model past the
cap would route on only the first N experts (or index groups out of bounds) — plausible-looking
WRONG output, no error — so it must decline to the staged/CPU path. This is exactly the runtime
guard M22 added in gpu/backend.go BuildResident; declared here too so the hardware-matrix
generator derives the same answer the runtime gives (the C6 one-source-of-truth discipline — a
feature-only predicate silently over-admitted Kimi K2's 384 experts as WebGPU-resident). Absent
entry = no fixed-size router cap (a backend that declines these archs on features never reaches
this — Metal/CUDA decline Kimi on FeatMLA).
```

## residentBackendMoECap.why

Moved from `decoder/features.go` (the comment after `residentBackendMoECap` (the "WHY cuda is 512" block)) on 2026-10-10.

```text
WHY cuda is 512 and the others are not — this is three shader constants plus this map, and only
one of them is expensive to change:

  - cuda   moe.cu MOE_MAX_E, raised 256→512. The constant bounds moe_route's per-thread scratch
           (score[]/sel[]), and moe_route lives in the AUDITED 12.6.85 cuda/testdata/moe.ptx — so
           raising it required regenerating that artifact. Done at a PINNED, IDENTICAL toolchain
           with a byte-identical rebuild-unchanged control first; the resulting diff touches only
           moe_route's stack depot (2368→4416 B) and every other kernel in the file is
           byte-identical. Procedure: cuda/testdata/REGEN.md. 512 covers Kimi-K2's 384 routed
           experts (its whole point) and DeepSeek-V4-Pro's 384; it deliberately stops short of
           Kimi-K3's 896, which is an unbuilt family that should not set validated limits.
  - metal  DECLARED here for the first time, NOT raised. Its shader really is capped at 256
           (metal/moe.go: the route kernel's score[256]/sel[256] arrays) and rejects above it, so the old "absent entry = no fixed-size
           router cap" claim was FALSE for metal and the hardware-matrix generator derived a
           different answer than the runtime gives — exactly the C6 one-source-of-truth defect
           this map exists to prevent. Raising metal's shader needs Mac validation: future leg.
  - webgpu ALSO raised 256→512, and this is the one that actually unblocks Kimi-K2: webgpu is the
           only backend declaring FeatMLA, so on cuda/metal K2 declines on FEATURES no matter what
           this cap says. Its WGSL compiles at runtime (no frozen artifact), and it was validated
           on this box. Groups stay 32 — array<f32,32> gscore is untouched and no target needs more.
```

## ResidentBackendMoECap

Moved from `decoder/features.go` (the comment above `ResidentBackendMoECap`) on 2026-10-10.

```text
ResidentBackendMoECap returns backend's declared router-kernel capacity. ok is false for a
backend with no fixed-size router.

Exported for the BACKENDS to read (M-31). gpu/residency.go had its own hardcoded 256/32 while
this map said 512, so ResidentEligible admitted a 384-expert Kimi-K2 or DeepSeek-V4-Pro — "✅
resident" in both generated matrices — and BuildResident then declined it to CPU with a
message naming 256, or refused to start under -require-be webgpu. Two pin tests were green
throughout: one greps gpu/moe.go, the other asserts ResidentEligible; neither reads
residency.go. That is exactly the drift this map exists to prevent, happening one file over.
```

## residentPerLayerGeomBackends

Moved from `decoder/features.go` (the comment above `residentPerLayerGeomBackends`) on 2026-10-10.

```text
residentPerLayerGeomBackends declares which resident backends implement PER-LAYER attention
geometry — a layer's own head_dim/KV-head count genuinely differing from another's (Gemma 4's
local/global split: HeadDim 256 vs gemma4.GlobalHeadDim 512), not just a per-layer differing
RoPE table (FeatPerLayerRoPE, which every resident backend already has and which is NOT this).

This is NOT expressed as a ResidentFeature: no other family needs it (Gemma 3's dual-base RoPE
keeps head_dim uniform), so the taxonomy has no flag for it and residentFeatures() cannot name
it as a requirement — Gemma 3 and dense Gemma 4 derive the IDENTICAL feature set otherwise (see
TestZZGemmaFeatureDiff-shaped comparisons). CUDA and Metal implement it via their own per-layer
geometry seam (cuda/resident.go's cudaLayer.hd/nKV, metal/model.go's residLayer.geom); WebGPU's
twin fields (runLayer.ghd/gnKV/ghalf, gpu/decoderunner.go) exist but are never populated by
gpu/residency.go's per-layer builder for any family — confirmed 2026-09-08 when G6's Gemma-set
feature work (docs/tasks/task-gpu-paths-2026-09.md) satisfied every ResidentFeature dense Gemma 4
nominally requires without also covering this, which would have silently admitted it to a path
that crashes on upload ("gpu: residency unsupported projection precision \"\"") rather than
mis-running quietly — still a decline this predicate exists to make deliberate instead of
accidental.
```

## Model.PerLayerGeomOK

Moved from `decoder/features.go` (the comment above `Model.PerLayerGeomOK`) on 2026-10-10.

```text
PerLayerGeomOK is residentPerLayerGeomOK's Model-level twin, exported so a resident backend's
own BuildResident can check it directly — the same pattern ResidentBackendMoECap already
established (a runtime check a backend calls individually, rather than through the combined
ResidentEligible, which is the doc-generation/admission-golden predicate). gpu/residency.go's
BuildResident calls this because its own admission check is hand-rolled from
MissingResidentFeatures, not ResidentEligible, and MissingResidentFeatures alone would have
silently admitted dense Gemma 4 once G6's Gemma-set features landed — see
residentPerLayerGeomBackends' own comment for the incident this predicate exists to prevent.
```

## ResidentBackendFeatures

Moved from `decoder/features.go` (the comment above `ResidentBackendFeatures`) on 2026-10-10.

```text
ResidentBackendFeatures returns a COPY of the feature set a resident backend implements
(nil if the backend is unknown). Returning a copy keeps the source map read-only from
outside the package: an external caller or third-party init() cannot add a feature claim
its kernels don't implement — precisely the silent-wrong-output failure the registry exists
to prevent — nor trigger a fatal concurrent map write during a Load (audit B-09). Callers
look up one backend by name; the package's own admission path reads the unexported map.
```

## residentBackendFeatures

Moved from `decoder/features.go` (the comment above `residentBackendFeatures`) on 2026-10-10.

```text
residentBackendFeatures declares what each resident backend's decode path implements.

These live HERE, not in the backends, for two reasons. First, one source of truth: three
hand-maintained copies of this logic is precisely how the silent-wrong-output bug recurs
(Metal had it; CUDA had it; the audit found them independently). Second, testability — the
backends are build-tagged (`-tags cuda`, `-tags gpu`, darwin-only metal), so a test that
could see their sets could not run in CI. Declared here, the registry-driven admission gate
(features_test.go) checks every (arch × backend) pair with no GPU present.

A backend adds an entry ONLY when it ships the kernel that implements it. Overclaiming here
is exactly the lie the gate exists to catch.
```

## residentBackendFeatures.cuda

Moved from `decoder/features.go` (the comment above the `"cuda"` entry of `residentBackendFeatures`) on 2026-10-10.

```text
cgo-free CUDA (cuda/): the dense Qwen2/Llama block, plus QK-norm, sliding window, the
Gemma set ((1+w) RMS, sandwich norms, GeGLU, embed scale, per-layer RoPE base), partial
rotary, and MoE (routed + ungated shared expert). N-100 (docs/audit-2026-09-10.md,
corrected 2026-09-16): YaRN mscale (FeatRopeMscale) and logit softcap
(FeatFinalLogitSoftcap) are BOTH declared below now — this comment's "no YaRN mscale; no
logit softcap" was stale, from before they landed. What remains genuinely NOT implemented:
per-layer rotary WIDTH (only per-layer base); no SSM.

TRAP, resolved 2026-09 — see cuda/resident.go's own moe_route call site for the full
story: the nGroup/topkGroup argument order was unverified for a real mismatch until this
pass (found live, not assumed: a deliberate transposition passed the existing
TestMLAResidentParityCUDA clean, since its generation check compared only the first
token). Closed by strengthening that test to compare the full sequence, confirmed to
catch the same transposition, restored clean. FeatMLA declared below with real
end-to-end parity against testdata/deepseek-tiny (n_group=2, topk_group=1 — genuinely
mismatched, not a coincidental no-op case).

FeatMoE covers the ROUTED block (router + stacked experts + every routing flavour the
route kernel handles) AND the always-on UNGATED shared expert (GLM/DeepSeek). The GATED
shared expert (Qwen-MoE's sigmoid(SharedGate·h) scaling) is NOT wired — BuildResident
declines it at load, since no committed fixture gates it end to end and FeatMoE is one flag
that cannot express the sub-shape. That decline is the honest "admitted, but this variant
is not wired" in a table whose whole job is to not lie.

FeatPartialRotary and the shared expert land together on purpose: every partial-rotary arch
(glm4_moe) also has a shared expert, so neither is independently reachable — glm-tiny is the
joint end-to-end gate (TestGLMResidentParity), and declaring partial rotary before the
shared expert existed would have admitted glm onto a path no model could exercise.
```

## residentBackendFeatures.cuda.gatedShared

Moved from `decoder/features.go` (the comment above `FeatMoEGatedShared` in the cuda entry) on 2026-10-10.

```text
The SIGMOID-GATED always-on shared expert (Qwen-MoE): out += sigmoid(SharedGate·h)·shared(h).
Declared 2026-08-20. The kernel was always here — moe.cu's shared_gate_combine has an
`ungated` flag and its comment names the gated case "Qwen-MoE" — so what this backend
actually lacked was the [1,hidden] gate weight in the build, not any device code. Worth
recording: the feature table said "CUDA implements only the ungated combine", which was
true of the WIRING and false of the kernel, and nothing reconciled the two.

GATED BY qwen3_5_moe-tiny (cuda.TestQwen35ResidentParityCUDA), whose MoE block IS
Qwen2-MoE's — transformers derives Qwen3_5MoeSparseMoeBlock from it, shared_expert_gate
included. So declaring this ALSO admits qwen2_moe on cuda as a documented side effect,
the same shape as Metal's FeatRopeMscale/Mellum note above: no qwen2_moe fixture exists
in this tree, so that family's CUDA admission rests on the inheritance, not on its own
end-to-end run. Add one if that ever stops being good enough.
```

## residentBackendFeatures.cuda.deltanet

Moved from `decoder/features.go` (the comment above `FeatDeltaNet` in the cuda entry) on 2026-10-10.

```text
Gated-DeltaNet: the deltanet.ptx mixer (conv ring + delta rule + gated norm) plus the
family's fused double-width q_proj and sigmoid output gate. Declared 2026-08-20 with
the end-to-end gate, not ahead of it — the same discipline the GPT-2/gpt-oss entries
record. This admits BOTH siblings: the dense one needs only this feature, and
qwen3_5_moe/qwen3_next additionally need FeatMoEGatedShared (declared above, same
day) — TestQwen35ResidentParityCUDA covers both (N-34 (09-02): a stale comment here
once said CUDA "still does not implement" FeatMoEGatedShared, written before or
alongside the entry two lines up that declares it; the map and the prose disagreed
within the same block).
```

## residentBackendFeatures.cuda.ropeMscale

Moved from `decoder/features.go` (the comment above `FeatRopeMscale` in the cuda entry) on 2026-10-10.

```text
FeatRopeMscale: YaRN's attention_factor, folded into cos/sin by rope / rope_kv /
rope_kv_batched (cuda/glue.cu, gemv_fwd.cu, prefill_batched.cu) and threaded per LAYER
from Model.RopeMscaleLayer via cudaLayer.mscale. Proven in isolation first by
TestRopeMscale (scale=1 reproduces the unscaled rotation to 8.9e-08; scale=0.85 matches
the scaled reference AND is provably different from unscaled), then end-to-end on real
weights by TestMellumResidentParityCUDA — in that order, because the kernels took NO
scale parameter at all until 2026-08-31 and declaring this on them would have admitted
families onto a path that silently drops the factor.

DECLARING THIS ADMITS MELLUM, which is not a side effect but the point of validating it
first: mellumArchitecture needs exactly {FeatMoE, FeatPerLayerRoPE, FeatQKNorm,
FeatRopeMscale, FeatSlidingWindow} and CUDA already declared the other four, so this
single flag is the whole admission. Metal hit the identical coupling (G10) and resolved
it by an explicit call because no Mellum checkpoint was reachable there; here one is, so
it was measured instead.
```

## residentBackendFeatures.cuda.gptoss

Moved from `decoder/features.go` (the comment above `FeatAttnSink` in the cuda entry) on 2026-10-10.

```text
gpt-oss's two remaining departures, declared TOGETHER on 2026-08-31 because the family
needs both and neither admits anything on its own:

  FeatAttnSink  the learned per-head softmax sink, the clamped interleaved-SwiGLU
                expert, and the router whose bias reaches the selection WEIGHT. Kernels
                in cuda/gptoss_act.cu; sinkArg threaded into BOTH attention launches
                (decode + prefill) and launchGluSplitExpert dispatched from the MoE
                expert loop.
  FeatOutBias   the o_proj bias. NO new kernel was needed: aikit's gemv_quant.cu and
                goinfer's batched gemv_w4a8_rn already fold bias into the value BEFORE
                the accumulate select, so bias-plus-residual is one instruction here.
                It was pure wiring, at four launch sites (two decode, two prefill).

DECLARED ONLY AFTER A REAL gpt-oss-20b FORWARD RAN ON THIS PATH — the thing G7 had been
blocked on since 2026-08-18, and the reason 2224441's earlier declaration was reverted:
kernel-level parity is not end-to-end parity. TestGptOssResidentParityCUDA on the real
20B, resident on an 8 GB card via --moe-cache-experts: 7/8 argmax-exact, min cosine
0.996392. For scale, the same harness measures 0.982 on a 40-layer qwen3.6-35b-a3b and
0.974 on a 24-layer dense 0.5B, so this is at the top of the range, not scraping a bar.

Getting there took THREE silent defects, none of which any kernel test could see,
because each was a term the wiring dropped rather than a kernel computing it wrongly:
  d9829ce  the gate‖up bias table indexed by SLOT id under expert caching
  610ce7f  the per-expert DOWN bias never applied at all (0.750 -> 0.9993 on the tiny)
  this     route_gptoss never LOADED, so the router fell back to moe.cu's moe_route,
           which takes the mixing weight from the UNBIASED score. Same experts
           selected, different weights (0.895 -> 0.9964 on the real 20B).
```

## residentBackendFeatures.cuda.nope

Moved from `decoder/features.go` (the comment above `FeatNoPE` in the cuda entry) on 2026-10-10.

```text
G5 (docs/tasks/task-gpu-paths-2026-09.md): SmolLM3's NoPE layers get an all-zero per-layer
invFreq table instead of a new kernel path — RopeInvFreqLayer folds this in for every
backend that reads it (decoder/residency.go), so this line and Metal's twin are the
whole change. Identity rotation at invFreq==0 holds only when mscale==1 on those
layers, true of every layerNoPE family admitted so far — see that function's comment.
```

## residentBackendFeatures.cuda.attnTemp

Moved from `decoder/features.go` (the comment above `FeatAttnTemp` in the cuda entry) on 2026-10-10.

```text
G5 (docs/tasks/task-gpu-paths-2026-09.md): Ministral 3's post-RoPE query scale, folded into
rope_kv's existing launch (a new qTempScale parameter, applied to Q only, after the
rotation) rather than a new kernel — Model.AttnTempScale/AttnTempParams
(decoder/residency.go) supply the value; rope_kv_batched's twin recomputes it per row
device-side (position varies within one batched launch). beta==0 makes qTempScale==1
(exact no-op) for every other family. PTX regenerated (cuda/build_ptx.sh, NVRTC) and
verified end-to-end on real CUDA hardware — TestMinistral3ResidentParityCUDA.
```

## residentBackendFeatures.cuda.postOnly

Moved from `decoder/features.go` (the comment above `FeatPostOnlyNorm` in the cuda entry) on 2026-10-10.

```text
G5 (docs/tasks/task-gpu-paths-2026-09.md): Olmo 3 / Olmo Hybrid's no-pre-norm placement
(Model.PostOnlyNormResident) — segA quantizes the RAW residual (quant_vec) instead of
running rmsnorm_quant, and the pre-existing sandwich post-norm dispatch widens from
`sandwich` to `sandwich || postOnly`. Requires the fused QKV path off (it bakes a real
pre-norm weight in) — postOnly forces the unfused segA chain.
```

## residentBackendFeatures.cuda.cohere

Moved from `decoder/features.go` (the comment above `FeatLayerNorm` in the cuda entry) on 2026-10-10.

```text
G5 (docs/tasks/task-gpu-paths-2026-09.md), the last row: Cohere/Command-R + Cohere2/Command-
R7B. FeatLayerNorm is a genuinely NEW kernel (layernorm_quant, cuda/glue.cu) — this
backend had NO mean-centered norm before, only RMSNorm variants — but bias-free only
(Cohere's LayerNorm carries no learned bias term; a future bias-bearing LayerNorm family
would need its own kernel, not a flag on this one). FeatParallelBlock reuses segA's
existing pre-attn norm+quant (r.aq/r.aSc) as the MLP's input too, instead of segBFFN
re-normalizing the post-attention residual — no new kernel, a sequencing change (see
Model.ParallelBlockResident's comment). FeatLogitScale is a host-side multiply after
readback, the same shape as FeatFinalLogitSoftcap, via Model.LogitScaleResident (NOT
GraniteResidentParams' own copy of the same arch field — Granite's SSM path isn't CUDA-
resident, so this is the first consumer here).
```

## residentBackendFeatures.cuda.pairwise

Moved from `decoder/features.go` (the comment above `FeatPairwiseRoPE` in the cuda entry) on 2026-10-10.

```text
2026-10-01: GPT-J PAIRWISE rotation, via cuda/rope_pairwise.cu (rope_kv_pw,
rope_kv_batched_pw, rope_kv_mrope_batched_pw), bound in place of the NeoX rope pipelines
when Model.PairwiseRoPEResident(). Cohere/Cohere2/Aya were ADMITTED here before this on the
NeoX kernels and ran wrong from position 1 (real R7B/Aya at int4, worst cosine -0.075/
-0.041); the declaration is now backed by a peaked-attention gate that goes red on the NeoX
kernels (TestPairwiseRoPEResidentParityCUDA), the glm_ocr gate (TestGlmOcrResidentParityCUDA)
and the real-checkpoint gate (TestCohereRealResidentParityCUDA, heavy).
```

## residentBackendFeatures.webgpu.gatedGELU

Moved from `decoder/features.go` (the comment above `FeatGatedGELU` in the webgpu entry) on 2026-10-10.

```text
FeatGatedGELU: a genuinely new kernel pair (gegluShaderWGSL/gegluQuantWGSL,
gpu/layer.go + gpu/decodefuse.go) — this backend had no GELU-tanh-gated activation
before, only SiLU. Clamps the tanh argument to ±15 before calling tanh, matching
Metal's own fix for the exact overflow that cost it a real cosine regression
(0.818→0.994) the first time this math shipped there.
```

## residentBackendFeatures.webgpu.attnSink

Moved from `decoder/features.go` (the comment above `FeatAttnSink` in the webgpu entry) on 2026-10-10.

```text
FeatAttnSink: gpt-oss's three departures — the learned per-head softmax sink
(threaded through every attention kernel: attn, attn-keys, attn-f16, attn-i8, and
all three wide variants — 7 pipelines total), the clamped interleaved-SwiGLU expert
(new gptossGluQuantWGSL kernel), and a router whose bias reaches the mixing WEIGHT,
not just selection (new routeGptOssWGSL kernel, mirroring cuda/gptoss_act.cu's
route_gptoss and metal/moe.go's twin exactly — moeRouteWGSL's own contract is wrong
for this family). The sink itself needed a genuinely separate per-layer uniform
(gpu/attention.go's HS struct) rather than folding into the existing geometry-cached
P uniform, since hasSink is a per-LAYER property that P's geomFor dedup cache cannot
safely carry. Declared only after a real gpt-oss forward ran resident end-to-end
(TestGptOssResidentParityWebGPU) — see CUDA's/Metal's own FeatAttnSink comments for
why kernel-level parity alone is not enough evidence for this family.
```

## residentBackendFeatures.metal

Moved from `decoder/features.go` (the comment above the `"metal"` entry of `residentBackendFeatures`) on 2026-10-10.

```text
cgo-free Metal (metal/): dense Qwen2/Llama plus qk-norm, sliding-window, partial-rotary,
MoE (router + stacked experts + shared expert; metal/moe.go), the full Gemma set —
sandwich norms, GeGLU, (1+w) RMS, √hidden embed scale, per-layer RoPE base — GPT-2's
LayerNorm/non-gated-MLP/learned-pos/out-bias (2026-08-18: layernorm_quant, act_quant,
gemv_w4a8_resid_bias/gemv_w4a8_sa_bias_resid, encodeLayer/encodeAttention wiring, the
ForwardArgmax V%8!=0 fallback for GPT-2's 50257 vocab — TestGPT2ResidentParity, min cosine
0.999) — and gpt-oss's attention sink + clamped-SwiGLU MoE + custom router (2026-08-18:
kernels.go's `attention` sink term, moe.go's route_gptoss/swiglu_quant_gptoss/
gemv_w4a8_moe_wacc_bias, the moe.go isGptOss dispatch split, gpt-oss's YaRN rope mscale
riding the already-wired-everywhere rope kernel param — TestGptOssResidentParity, 8/8
argmax-exact, min cosine 0.9989 on the tiny fixture). Gemma parity was gated on the
GELU-tanh overflow fix (glu_act clamp, 38a2b7c): logit cosine 0.818→0.994. Still declines
MLA and SSM.

FeatRopeMscale is SHARED with Mellum (same required-feature set minus
FeatAttnSink/FeatOutBias) — declaring it for gpt-oss's YaRN ALSO admits Mellum on Metal, a
path with ZERO end-to-end validation here: no real Mellum checkpoint on this box (~24GB),
and a synthetic-random-weight structural test was tried and abandoned as inconclusive (even
a plain dense qwen2 with NO QK-norm fails the same cosine bar against fully-random untrained
weights at realistic dims — the methodology can't discriminate a real bug from quantization
noise on unstructured weights, so it proves nothing either way). Declared anyway (explicit
user call, 2026-08-18): Mellum already has a trusted GPU path on WebGPU, this only adds an
unvalidated SECOND path on Metal, and the flag is one boolean — trivially reversible if a
real Mellum checkpoint later surfaces a problem. If you're chasing a Mellum-on-Metal bug,
start here.
```

## residentBackendFeatures.metal.cohere

Moved from `decoder/features.go` (the comment above `FeatParallelBlock` in the metal entry) on 2026-10-10.

```text
G5 (docs/tasks/task-gpu-paths-2026-09.md), the last row: Cohere/Command-R + Cohere2/Command-R7B.
FeatLayerNorm was already true (GPT-2's layernorm_quant, bias-capable via r.uLNHasBias —
Cohere just needs hasBias=0, no kernel change). FeatParallelBlock reuses encodeAttention's
pre-attn r.aq/r.aSc as the MLP's gate|up input in encodeLayer instead of re-normalizing
r.x post-attention-add — no new kernel, a sequencing change (see
Model.ParallelBlockResident's comment). FeatLogitScale is a host-side multiply in
finalizeLogits, the same shape as FeatFinalLogitSoftcap's softcapParallel, via
Model.LogitScaleResident.
```

## residentBackendFeatures.metal.pairwise

Moved from `decoder/features.go` (the comment above `FeatPairwiseRoPE` in the metal entry) on 2026-10-10.

```text
2026-10-09 (docs/tasks/task-metal-pairwise-rope-2026-10.md): GPT-J PAIRWISE rotation, via the twins rope_pw,
rope2_pw, rope_f16_pw and rope_mrope_f16_pw, bound in place of the NeoX pipelines when
Model.PairwiseRoPEResident() (metal/model.go, metal/prefill.go); the MC3 batched step declines a pairwise model.
Backed by the kernel gate (TestRopePairwise_*), the peaked-attention resident gates that go red on the NeoX
kernels (TestPairwiseRoPEResidentParityMetal, TestGlmOcrResidentParityMetal) and the real checkpoints.
```

## FeatAttnOutputGate

Moved from `decoder/features.go` (the trailing comment on the `FeatAttnOutputGate` constant) on 2026-10-10.

```text
Laguna: ctx *= softplus(g_proj·h) applied BEFORE o_proj, plus a per-layer QUERY head count. CPU-only — no resident backend implements either, so CUDA/Metal/WebGPU all decline. Without this the family needs nothing CUDA lacks and would be ADMITTED-but-mis-run: the resident path would skip the gate entirely and still produce plausible logits.
```

## FeatShortConv

Moved from `decoder/features.go` (the trailing comment on the `FeatShortConv` constant) on 2026-10-10.

```text
LFM2/LFM2.5: the gated short-convolution mixer that replaces attention on 22 of 30 layers (B,C,x = in_proj(h); conv = depthwise_causal_conv(B*x), no activation; out_proj(C*conv)), carrying a per-layer rolling window of the last K-1 inputs. CPU-only — no resident backend implements the conv OR its recurrent state. Declared for the SAME reason as FeatAttnOutputGate above: LFM2 is otherwise a plain GQA+QK-norm+SwiGLU model that needs nothing CUDA lacks, so without this it would be ADMITTED and then mis-run, with the resident path treating every conv layer as attention. The window also makes it stateful, so a resident runner would need the state plumbing FeatSSM/FeatDeltaNet have and this has not.
```

## FeatGemma4EModel

Moved from `decoder/features.go` (the trailing comment on the `FeatGemma4EModel` constant) on 2026-10-10.

```text
Gemma-4 E2B/E4B shape: per-layer embeddings (PLE, hidden_size_per_layer_input>0) + cross-layer shared-KV + variable per-layer FFN. runLayersGemma4 injects PLE per layer. Metal declares it (S1, docs/tasks/task-multimodal-support-2026-10.md): the PLE inputs ride the embedding row (embedResidentInto), shared layers alias their source's KV, FFN width is per layer. CUDA and WebGPU implement none of it and must keep declining, or they would SKIP the PLE branch and silently mis-run.
```

## residentBackendFeatures.cuda.gemma4EModel

Moved from `decoder/features.go` (the trailing comment on `FeatGemma4EModel` in the cuda entry of `residentBackendFeatures`) on 2026-10-10.

```text
Gemma 4 E2B/E4B on CUDA (S1 on CUDA, docs/tasks/task-multimodal-support-2026-10.md): the PLE branch in segBFFN, KV-shared layers aliasing their source cache, per-layer FFN widths
```

## errBlockSpecResidentBusy

Moved from `decoder/blockspec.go` (the comment above `errBlockSpecResidentBusy`) on 2026-10-10.

```text
errBlockSpecResidentBusy is returned when another generation already holds the model's single
shared resident KV (audit R-00). Returned before any device write, so a caller's existing
fallback to plain Generate (internal/serveapp/openai.go) is exact — Generate makes its own CAS
attempt and, finding the same claim held, drops to the staged CPU path itself (M9).
```

## ResidentBlockDrafter.ExtendContext

Moved from `decoder/blockspec.go` (the comment above `ExtendContext` in `ResidentBlockDrafter`) on 2026-10-10.

```text
ExtendContext appends fused rows to the drafter's own K/V at the current context end.
Incremental by contract: rebuilding the whole context per block measured 2.4x the cost
at a 1024-token context (docs/spec/08), widening with length.
```

## ResidentDrafterHost.SetBatchedCapture

Moved from `decoder/blockspec.go` (the comment above `SetBatchedCapture` in `ResidentDrafterHost`) on 2026-10-10.

```text
SetBatchedCapture arms the hidden-state seam for the whole batch: the next
PrefillLastNArgmax records the residual at each named layer for ALL its rows. The
per-token seam costs a sync and a download per tap PER TOKEN; this pays one per tap for
the block (1.09 ms vs 2.79 ms at M=6, measured).
```

## ResidentSeedArgmax

Moved from `decoder/blockspec.go` (the comment above `ResidentSeedArgmax`) on 2026-10-10.

```text
ResidentSeedArgmax is an OPTIONAL narrowing of PrefillLastNArgmax for the prompt seed: the one
place the loop asks for M rows of argmax and reads exactly one of them.

The seed calls PrefillLastNArgmax over the whole prompt and uses `ids[len(ids)-1]`. On a
vocab-151,936 target a 2048-token prompt therefore allocates 1.24 GB of VRAM for the batched
logits, a 1.24 GB host slice and a 1.24 GB device-to-host copy, runs the head GEMV over 2048 rows
and a single-threaded host argmax over 311M floats — to obtain ONE token id. The capture the seed
actually needs comes from the LAYER LOOP, not the head, so heading one row loses nothing
(audit-2026-09-02 C-12).

Optional rather than added to ResidentDrafterHost, following ResidentCapped/ResidentGreedy: a
backend that has not implemented it keeps working through the wide path.
```

## BlockSpecOptions.VerifyWidth

Moved from `decoder/blockspec.go` (the comment above `VerifyWidth` in `BlockSpecOptions`) on 2026-10-10.

```text
VerifyWidth is how many block positions the TARGET verifies per round (anchor plus
VerifyWidth-1 drafts). 0 selects the default below.

It is NOT the drafter's trained block width, and that distinction is the single biggest
lever measured: the drafter drafts its full block either way, but verifying all 16
positions makes code a 0.89x LOSS, while verifying 7 makes it 1.60x. The tail positions
rarely land and cost full batched-verify price — positions 12-15 gain 0.09 accepted
tokens BETWEEN THEM while costing 9.4 ms of verify per round (docs/spec/08).
```

## BlockSpecOptions.StopIDs

Moved from `decoder/blockspec.go` (the comment above `StopIDs` in `BlockSpecOptions`) on 2026-10-10.

```text
StopIDs are the caller's extra stop tokens — SamplingParams.StopIDs, which for a served
request carries the CHAT TEMPLATE's stops on top of the model's own.

It exists because this loop rebuilt its stop set from Cfg.EOSIDs() alone while every other
speculative loop asks target.isStop(tok, sp). For the pairing this ships for (Qwen3-4B +
DFlash) that is {151645} against m.eosIDs' {151645, 151643}, so a <|endoftext|> was emitted
as ordinary content and generation ran on to <|im_end|> or max_tokens — with streamTokens
decoding the stop token into the response. The "lossless by construction" contract was
broken by the STOP SET, not by the verify (audit-2026-09-02 C-11).
```

## defaultVerifyWidth

Moved from `decoder/blockspec.go` (the comment above `defaultVerifyWidth`) on 2026-10-10.

```text
defaultVerifyWidth is 8 — measured as the optimum for math (1.79x), within 2% of code's
optimum of 7 (1.60x), and serving both pairings tested. Per-traffic-class tuning is worth
a few percent (code 7, chat 4) and needs a router; 8 is the one number that works everywhere.
```

## BlockSpec

Moved from `decoder/blockspec.go` (the comment above `BlockSpec`) on 2026-10-10.

```text
BlockSpec is an attached block drafter, ready to serve many generations.

ATTACHING IS SEPARATE FROM GENERATING, and that split is not cosmetic. AttachBlockDrafter
uploads the drafter's weights (~500 MB for the 4B pairing) to the device; doing it per request
made the production path measure 0.17x — a 6x LOSS — while the loop itself was healthy and
lossless at 5.76 tok/round. Acceptance looked fine and the wiring was throwing the speedup
away. Attach once per process, generate per request.
```

## BlockSpec.generate.exactAttention

Moved from `decoder/blockspec.go` (the comment above the `enterExactAttention` defer in `BlockSpec.generate`) on 2026-10-10.

```text
Defense in depth, and NOT load-bearing today: this loop takes no M=1 decode step (it seeds through a batched pass and then only verifies), so with the
multi-row verify lane off its verify rows are exact whether or not this scope is held — removing it does not change TestFlashDecodeBlockSpecLane's outcome
(checked: the mutant survives). It is here so a future edit that adds an M=1 step to this loop cannot silently mix trees. Counted, so GenerateStream's hold nests.
```

## BlockSpec.generate.claim

Moved from `decoder/blockspec.go` (the comment above `var reuseFrom` in `BlockSpec.generate`) on 2026-10-10.

```text
Claim the single shared resident KV before any device write (audit R-00): this path drives
the same positional cache as Model.Generate and the n-gram/speculative paths, none of which
it coordinated with before this fix. A loser returns before touching state, so the caller's
existing fallback to plain Generate is exact (M9's "concurrent distinct sequences still
complete correctly, only resident speed is lost"). Gated on m.resident != nil, mirroring
model.go's useGPU check: a host without a resident keeps its KV in its own CPU cache and
never touches the resident device KV (the removed CPU BlockSpec was one such host), so it
must not contend for resBusy or forget a resIDs commit it never wrote.
```

## BlockSpec.generate.guardStopped

Moved from `decoder/blockspec.go` (the comment above the `seamOff` disarm in `BlockSpec.generate`) on 2026-10-10.

```text
The drafter is not paying for itself on this generation. Finish with plain
resident decoding rather than continuing to lose ~20% invisibly.

DISARM THE CAPTURE SEAM FIRST. Leaving it armed makes every fallback token pay
five tap downloads for hidden states nothing will read — measured turning a
0.98x generation into 0.87x, i.e. the guard made things WORSE than not guarding.
```

## BlockSpec.generate.fallbackStep

Moved from `decoder/blockspec.go` (the comment above the fallback step in `BlockSpec.generate`) on 2026-10-10.

```text
The FAST greedy step where the backend has one: argmax reduced on-device with a
4-byte readback, which is what Model.Generate uses. Falling back through
PrefillLastNArgmax(M=1) instead downloads the full logit row per token — the
same slow primitive that made gate 3's baseline wrong, and it left the guard
converting a 0.82x into 0.82x instead of into plain-decode speed.
```

## BlockSpec.generate.fallbackStop

Moved from `decoder/blockspec.go` (the comment above the fallback stop check in `BlockSpec.generate`) on 2026-10-10.

```text
The stop token is NOT emitted here either. The loop-top check breaks on it, but
only AFTER this append ran — so without this the fallback emits one token past
where plain decoding stops, which is exactly the 259-vs-258 mismatch.
```

## BlockSpec.generate.width

Moved from `decoder/blockspec.go` (the comment above `blockSpecRoundWidth` call in `BlockSpec.generate`) on 2026-10-10.

```text
M-13: the round's width is clamped by what is left of BOTH budgets.

MaxTokens: the loop condition is checked per ROUND while a round commits up to `width`
tokens at once, so max_tokens=2 could return 9 and usage.completion_tokens could exceed
the request's own cap. The losslessness gates could not see it — the (since removed) CPU one
compared only the common prefix, and the CUDA one asks the reference for exactly len(got) tokens.

The context cap: verifying `width` rows at `pos` with no clamp makes checkCap refuse the
WHOLE round near the end of the window, so a nearly complete response ends in a
generation error. Plain Generate and the server both clamp instead, so a max-length turn
finishes cleanly with "length"; this path did not.
```

## BlockSpec.generate.burstStop

Moved from `decoder/blockspec.go` (the comment above the burst stop scan in `BlockSpec.generate`) on 2026-10-10.

```text
TRUNCATE BEFORE EOS INSIDE THE BURST. A round commits several tokens at once, so a
stop token can land in the MIDDLE of one; appending the whole burst emits content
AFTER it, which plain decoding never does. And the stop token itself is EXCLUDED,
because Generate breaks on it without emitting (model.go, isStop) — matching that
exactly is what makes the two paths token-identical.

Caught by a 384-token run where spec emitted 259 tokens against greedy's 258. At 96
tokens neither generation reached EOS, so the bug was invisible — a reminder that a
losslessness gate only covers the lengths it actually runs.
```

## BlockSpec.generate.onRound

Moved from `decoder/blockspec.go` (the comment above the `OnRound` call in `BlockSpec.generate`) on 2026-10-10.

```text
M-13: `width` here is the CLAMPED width this round actually verified, which is what the
telemetry must count — drafted = the positions offered (width-1, the anchor is not a
draft), evaluated = the positions tested. GenerateStream left Drafted and Evaluated at
zero, so AcceptanceRate() was 0 for every block-spec generation and the adaptive
controller's own signal was unreadable.
```

## BlockSpec.generate.commit

Moved from `decoder/blockspec.go` (the comment above the final commit in `BlockSpec.generate`) on 2026-10-10.

```text
P-05 (audit-2026-09-10): the ONLY exit that commits, mirroring generateInto's own rule
(decoder/model.go, "the ONLY place the resident cache's contents are recorded: a
generation that ran to completion"). Every early return above (a device error, or the
caller's emit stopping consumption early) leaves resIDs nil, so the next turn cold-prefills
rather than trusting a state this function cannot vouch for as fully written. This gives the
next PLAIN Generate turn a warm prefix after a --drafter turn. resDrafterSynced = s marks
THIS BlockSpec instance's own drafter context as the one that's actually in sync with the
commit below (the deferred half this fix completes) — the next call into this same
instance's generate() can then reuse both the target's resident KV AND the drafter's own
context; any OTHER writer's commit (plain Generate, n-gram) clears it via
residentForgetIDs, so a drafter turn never trusts a context it never built.
```

## blockSpecStopSet

Moved from `decoder/blockspec.go` (the comment above `blockSpecStopSet`) on 2026-10-10.

```text
blockSpecStopSet is this loop's stop predicate, and it must agree with Model.isStop — the
predicate plain decoding and every OTHER speculative loop use.

It did not. The set was rebuilt from Cfg.EOSIDs() alone: config.json's eos_token_id, without
generation_config.json's additions (which resolveEOSIDs merges into m.eosIDs) and without the
caller's SamplingParams.StopIDs (which for a served request carries the chat template's stops).
On the pairing this ships for, Qwen3-4B + DFlash, that is {151645} against {151645, 151643}: a
<|endoftext|> was emitted as ordinary content and generation ran on to <|im_end|> or max_tokens,
with streamTokens decoding the stop token into the response. "Lossless by construction" was
broken by the STOP SET, not by the verify (audit-2026-09-02 C-11).
```

## blockSpecRoundWidth

Moved from `decoder/blockspec.go` (the comment above `blockSpecRoundWidth`) on 2026-10-10.

```text
blockSpecRoundWidth clamps a round's verify width to what is left of BOTH budgets. Returns < 1
when neither has room, which the caller treats as a clean finish.

TWO SEPARATE M-13 DEFECTS, ONE CLAMP. The loop tests its token budget once per ROUND while a
round commits up to `width` tokens at once, so max_tokens=2 could return 9 and
usage.completion_tokens could exceed the request's own cap. And verifying `width` rows at `pos`
with no context clamp made the backend's checkCap refuse the WHOLE round near the end of the
window, so a nearly complete response ended in a generation error — where plain Generate and the
server both clamp instead, and a max-length turn finishes cleanly with "length".

The losslessness gates could not see either one: the (since removed) CPU gate compared only the
common prefix, and the CUDA gate asks the reference for exactly len(got) tokens.
```

## BlockSpec.GenerateStream.stopIDs

Moved from `decoder/blockspec.go` (the comment above `drafted, evaluated` in `BlockSpec.GenerateStream`) on 2026-10-10.

```text
StopIDs: this loop is the one speculative path that did not consult the caller's stop
set, so a chat-template stop was decoded into the response as content (C-11).

Drafted/Evaluated come from the per-round hook because only the loop knows the CLAMPED
width each round used; computing them out here from len(toks) would assume every round
ran at the configured width, which is exactly what the M-13 clamp makes untrue.
```

## BlockSpec.GenerateStream.cancel

Moved from `decoder/blockspec.go` (the comment above `g.err = ctx.Err()` in `BlockSpec.GenerateStream`) on 2026-10-10.

```text
M-13: a cancelled generation returned with g.err nil, so the caller read a truncated
stream as a clean finish. Same shape as M-23 one tranche earlier, in a different loop.
```

## breakEvenTokensPerRound

Moved from `decoder/blockspec.go` (the comment above `breakEvenTokensPerRound`) on 2026-10-10.

```text
breakEvenTokensPerRound is the acceptance below which block drafting LOSES.

A round costs draft + batched verify + the capture seam whatever it accepts; plain decoding
costs one target forward per token. So the drafter pays only when it commits more tokens per
round than the round costs in decode-equivalents. On the measured 4B/2070S pairing that is
~39 ms per round against an 11.1 ms decode — about 3.5.

2.5, BELOW break-even, not above it. This was 3.8 and that was backwards. The reasoning for a
margin ABOVE break-even was "disabling a drafter that is merely breaking even costs nothing" —
which is false, because acceptance MEASURED OVER THE FIRST FEW ROUNDS is not acceptance over
the generation. Math averages 5.88 tok/round end to end and is a 1.58x workload, but its
opening rounds are slow enough that a 3.8 threshold disabled it: 1.58x became 0.97x, the guard
costing 39% on a workload it was supposed to protect.

So the margin belongs BELOW break-even. A false negative (disabling a paying workload) costs
~40%; a false positive (six unprofitable rounds before tripping) costs ~8%. The guard should
only fire when a workload is CLEARLY losing — chat sits at 1.96 and still trips at 2.5 — and
should leave anything ambiguous alone.
```

## guardWindow

Moved from `decoder/blockspec.go` (the comment above `guardWindow`) on 2026-10-10.

```text
guardWindow is how many rounds to observe before judging.

SIX. Three was TRIED AND REVERTED, and the measurement is worth keeping because it refutes
the reasoning that motivated it:

	case      window 6   window 3
	code        1.57x      1.54x   (kept either way)
	MATH        1.58x      0.91x   <- falsely tripped
	chat A      0.91x      0.90x
	chat B      0.94x      0.96x
	thinking    0.96x      0.92x

Three bought nothing on chat and cost 42% on math by disabling a drafter that was paying.
The argument for shortening was that a response OPENS with boilerplate, so early rounds are
optimistic and judging early errs toward keeping a good drafter. That is false for math,
whose opening is evidently less predictable than its body — the whole-generation average
(5.88 tok/round) hides a slow start.

It also confirms the asymmetry that shapes this whole design: a false negative costs ~42%
(a paying workload disabled) where a false positive costs ~8% (six unprofitable rounds).
The window should err LONG. Six, not twelve, because the rounds spent deciding are pure loss when the answer is "stop":
twelve rounds is a third of a 96-token response, and halving the window halves that. The risk
of judging early is disabling a drafter that would have paid — and that risk is LOW here in a
way worth stating: a response's opening is boilerplate ("Here's a Python function...", a code
fence), which is the part a drafter predicts BEST. Early rounds are optimistic, so a short
window errs toward keeping a good drafter, not dropping one.
```

## acceptanceGuard

Moved from `decoder/blockspec.go` (the comment above `acceptanceGuard`) on 2026-10-10.

```text
acceptanceGuard disables a drafter that is not paying for itself, per generation.

It exists because the failure it catches is SILENT. A mis-paired drafter, a target in a mode
the drafter was not trained for, or an out-of-domain workload all produce correct output at
reduced speed — losslessness guarantees the tokens are right. Without this, `--drafter` on a
thinking-mode Qwen3 serves at 0.83x and nothing anywhere says so.
```

## acceptanceGuard.observe

Moved from `decoder/blockspec.go` (the comment above the window check in `acceptanceGuard.observe`) on 2026-10-10.

```text
CUMULATIVE, not per-window. Resetting the counters after a passing window gave a losing
workload a fresh budget every time: chat survived its first six rounds, reset, and only
tripped on the twelfth — measured 0.79x where tripping at six gives 0.91x. Keeping the
running average means a workload trips as soon as its EVIDENCE says so, while a
slow-starting profitable one (math opens below its own average) recovers as later rounds
pull the average up, instead of being judged on a six-round snapshot.
```

## Placement

Moved from `decoder/fitplan.go` (the comment above `Placement`) on 2026-10-10.

```text
Placement is which of tasks/task-fit-to-hardware.md's five strategies Plan chose for one backend.
The enum exists in full even though this phase only ever returns three of them — Phase 1 is
scoped to the pure decision function ("no behaviour change yet"), and PlacementHostComputedExperts
is reserved so a later phase (L-01) does not need to rewrite the type.
```

## ctxPlanFloor

Moved from `decoder/fitplan.go` (the comment above `ctxPlanFloor`) on 2026-10-10.

```text
ctxPlanFloor is the smallest context Plan will shrink to before giving up — task-fit-to-
hardware.md §2's priority order ("shrink context toward a floor of 4096 before moving anything
[else]"). Same figure decoder/fitguard.go's ctxFloor uses for the CPU staged-load guard, kept
as its own constant here rather than imported: the two guards are independent by design
(fitguard.go predates this file and is not being folded into it this phase), and a future
change to one floor should not silently move the other's.
```

## PlanRequest.ExtraBytes

Moved from `decoder/fitplan.go` (the comment above `ExtraBytes` in `PlanRequest`) on 2026-10-10.

```text
ExtraBytes prices whatever the model itself does not know about but will share its device:
a block drafter's weights + verify/capture buffers, a vision tower. tasks/task-fit-to-hardware.md
§2's "every allocation is a term of the plan, including the ones that attach after load" —
the concrete example that motivated it (a 26B's expert cache sized before a later --drafter
attach grabbed room NewBlockSpec then needed) is exactly what this term exists to prevent
PLAN from repeating. capSlots itself is now fixed too (docs/tasks/task-gpu-paths-2026-09.md,
2026-09-09): no ResidencyBackend interface change was needed after all — an out-of-band
hint on decoder.Model (Options.ExtraResidentBytes / Model.ExtraResidentBytes()) was enough,
since cuda/backend.go's BuildResident already receives *Model and can read it directly.
resolveCtxCapFit passes the SAME value as this field's own ExtraBytes when it asks Plan for
an unpinned ctx, so the two paths (Plan's dry run and the real load) price a drafter
identically.
```

## Model.kvBytesPerPositionAllLayers

Moved from `decoder/fitplan.go` (the comment above `Model.kvBytesPerPositionAllLayers`) on 2026-10-10.

```text
kvBytesPerPositionAllLayers is Plan's own KV-cost formula: per-layer (a per-layer-varying
family like Gemma 4 is not approximated by one model-level figure — the same reason
metal/backend.go's residentKVBytes sums per layer rather than using a single kvDim), summed
for K+V, at the requested precision. Bytes-per-element figures match
decoder/arch.go's kvBytesForCtx exactly (f32 4.0, f16 2, int8 1.125-including-scale).

M-28 (docs/audit-2026-09-10.md): the per-layer WIDTH now uses Architecture.kvDimAt — zero for
a linear/mamba/conv mixer layer (no position-indexed K/V at all), MLA's real compressed
latent width instead of the reconstructed per-head width — fixing the same overpricing bug
decoder/fitguard.go's estimateKVBytes fixes for the load-time host-RAM guard.

DELIBERATELY NOT applying the sliding-window COUNT cap (kvPositionsAt) here, unlike
kvBytesForCtx: this function returns a flat PER-POSITION rate (the caller multiplies by
whatever ctx it is evaluating), and a sliding-window layer's true cost is not linear in ctx —
it flattens at SlidingWindow. Doing that properly needs tryCtx/chooseCtx below to search
rather than multiply, a real restructuring with no sliding-window fixture in this file's own
test suite to verify against; scoped out of this pass rather than guessed at. Every family
this IS fixed for (MLA, DeltaNet/Mamba/conv hybrids) has no sliding window, so this rate stays
exact for them; a family with BOTH (none exists in the registry today) would still be
overpriced by the uncapped ctx term, the same direction the guard already erred in before
M-28, never the unsafe direction.
```

## WebGPUCtxCeiling

Moved from `decoder/fitplan.go` (the comment above `WebGPUCtxCeiling`) on 2026-10-10.

```text
WebGPUCtxCeiling is the WebGPU backend's fixed per-precision KV-capacity ceiling —
gpu/residency.go's own ctxCap before any -ctx request lowers it further via min(): 16384
positions at f32 (the proven 8 GB fit), 32768 at f16 (half the per-token bytes), 65536 at i8
(a quarter). Shared here, and gpu/residency.go calls THIS function instead of repeating the
three literals, so the planner's ctx choice and the backend's actual allocation can never
drift apart — exactly what Phase 3 (tasks/task-fit-to-hardware.md §7) needs before admitting webgpu:
a freeBytes-driven plan alone could pick a context above this fixed ceiling (VRAM allowing),
which BuildResident would then silently NOT honour (min() keeps the ceiling, and the plan's
promise would be wrong).
```

## Model.Plan

Moved from `decoder/fitplan.go` (the comment above `Model.Plan`) on 2026-10-10.

```text
Plan is tasks/task-fit-to-hardware.md §2's pure function ("no behaviour change yet [Phase 1] — the
plan is printed beside today's decision"): given this model, a candidate backend, how many
bytes are free on it, and what the caller asked for, decide a placement — resident,
expert-cached, weight-paged, or decline — following §2's priority order (shrink context toward
ctxPlanFloor first, since it costs no numerics; then cap routed experts into a cache; dense
weights always stay resident; CPU alone falls to weight-paging rather than ever declining).

No I/O, no side effects, and no defaults invented — see PlanRequest's own doc comment. backend
is "cuda", "metal", "cpu", or (Phase 3, tasks/task-fit-to-hardware.md §7) "webgpu" — admitted now that
M-32 is fixed (gpu/residency.go: BuildResident declines the same Nemotron/Qwen3.5/MLA +
KVF16/KVI8 combo Plan declines below, and honours -ctx via the same WebGPUCtxCeiling); an
unrecognised backend name gets the same GPU-shaped feature-eligibility decline a real one would
for an unsupported arch, rather than a panic or a silent wrong answer — enforced by the
ResidentEligible(m.w.arch, backend) check below, not by MissingResidentFeatures alone (M-08:
that check alone passed a feature-free arch on ANY name, including an unregistered one).
```

## Model.Plan.eligible

Moved from `decoder/fitplan.go` (the comment above the `ResidentEligible` check in `Model.Plan`) on 2026-10-10.

```text
M-08 (docs/audit-2026-09-10.md): MissingResidentFeatures alone is not admission —
ResidentEligible additionally checks that the backend is a REGISTERED one at all (an
unrecognised name plus a feature-free arch made the check above vacuously pass, since
missingFeatures(nil-required, nil-implemented) is empty), that the arch's own forward
is bridged to the resident runner (decodeRunnerEligible), that its MoE router fits the
backend's fixed-size scoreboard, and that per-layer attention geometry (Gemma 4's split
head_dim) is implemented. Missing any of those reported "resident" for Llama-4/cuda,
dense Gemma-4/webgpu, Kimi-K2/metal, and any unrecognised backend name.
```

## fitguard.go

Moved from `decoder/fitguard.go` (the comment at the top of the file (the load-time fit guard header)) on 2026-10-10.

```text
The load-time fit guard: refuse a model that cannot fit in RAM BEFORE allocating it, naming
the numbers and the flag that fixes it.

WHY. Cold-user run 2026-09-06 scenario D (docs/measurements/cold-user-2026-09-06.md): a 21 GB
35B-A3B on a 16 GB Mac drove the box +7,819 MB into swap in FIVE SECONDS with no message —
"the tool never told me it would not fit, the machine told me". `serve --help` already names
-stream-weights for exactly that model and that RAM, and with it RSS capped at 8.95 GB with
zero swapouts. The engine does the right thing; nothing told the user it existed.

SCOPE, deliberately. This is Phase 0 of docs/tasks/task-fit-to-hardware.md and nothing else: a
refusal with arithmetic. It does NOT plan a configuration and it does NOT flip -stream-weights
on for you; both are that doc's later phases, and choosing for the user is a bigger change than
telling them.

EVERY UNKNOWN PROCEEDS. An unreadable RAM figure, an unsupported source format, a zero-byte
estimate — each returns "don't know" and the load continues. The guard's failure mode must be
letting a doomed load through (the status quo), never refusing one that would have run.
```

## fitMemFraction

Moved from `decoder/fitguard.go` (the comment above `fitMemFraction`) on 2026-10-10.

```text
fitMemFraction is the share of the fit check's base memory figure the WEIGHTS alone may occupy.
Same figure and same provenance as metal/backend.go's residentMemFraction: ONE measured failure
(11.28 GB of 16 GB = 70.5% thrashed to swap exhaustion), so a threshold rather than a swept
curve. The rest is not slack — KV, scratch, the tokenizer, and the operating system live there
too. Originally fractioned against TOTAL physical RAM; fitCheckFor now fractions it against
CURRENTLY AVAILABLE memory instead (R13-follow-on) — the threshold itself is unchanged, only
what it is a fraction OF.
```

## ctxFloor

Moved from `decoder/fitguard.go` (the comment above `ctxFloor`) on 2026-10-10.

```text
ctxFloor is the smallest context this guard will auto-pin down to when the caller did not pin
one and the model's own maximum does not fit. Below this a context is not useful enough to hand
a user silently — refuse instead, the way R3 already does for the rest of the model. Named,
not measured: R13 (docs/measurements/cold-user-2026-09-07-macbook-arm64.md) did not measure a
real floor, and 2048 is stated as a product choice pending a real one.
```

## fitCheck.srcFileBytes

Moved from `decoder/fitguard.go` (the comment above `srcFileBytes` in `fitCheck`) on 2026-10-10.

```text
srcFileBytes is the on-disk size of a plain (non-streamed) .gguf SOURCE file, priced as an
ADDITIONAL transient term alongside weightBytes+kvBytes.

MEASURED, 2026-09-18, docs/measurements/cold-user-2026-09-18-nobara-pc.md Scenario D,
reproduced directly on nobara-pc (the same box) with a heap profile + /proc RSS sampling
around a real `decoder.Load` of gpt-oss-20b (12.11 GB MXFP4 GGUF, int4 resident weights
12.58 GB): peak RSS reached ~24.5 GB — matching weightBytes+fileSize (12.58+12.11=24.69 GB)
to within 2%, NOT the ~12.58 GB this guard priced before this field existed. Once
decoder.Load returns and the mmap is closed, RSS drops back to ~13.0 GB, confirming the
extra ~12 GB was the mmap'd SOURCE file, not a second copy of the resident weights.

WHY: loadGGUFWeights's own comment ("mmap, not heap-read: the raw quantized bytes stay in
reclaimable page cache") is true but incomplete — those pages are reclaimable in principle,
but the mapping (embed.OpenGGUFMmap) is held open for the ENTIRE build (buildWeightsFromGGUF
runs parallelLayers across every layer before the deferred g.Close() in loadGGUFWeights
finally runs), and RowDequantizer's per-row reads touch essentially every page of the file
by the time the model is fully quantized — so for most of the load, the WHOLE source file is
resident in RAM at the same time as the (also whole, by the end) resident weight set. This is
not double-buffering of the SAME data — it is source-plus-destination coexisting because
nothing releases the source pages incrementally as each tensor is consumed. That release
would need per-tensor madvise inside aikit/embed (a separate module, out of scope here); this
guard fixes what goinfer controls — pricing the real peak instead of only the final size.

Only meaningful for a plain resident `.gguf` load (isGGUF && !streamWeights): a `.giw` load
mmaps its own weight blob directly (no separate dequant-and-copy pass) and a safetensors
directory's loader has its own accounting; StreamWeights (once transcoded to .giw) also
leaves this repo through a different Load branch entirely (see model.go's ".giw" branch,
which never reaches fitCheckFor at all — by design, not a gap: measured 2026-09-20, a 7.8 GB
`.giw` loads as 1.46 GB anonymous + 7.4 GB file-backed pages the kernel can drop under
pressure, so there is no allocation peak for a guard to price). Zero when not applicable, so an existing fitCheck literal built by a test or another
caller is unaffected.
```

## fitCheck.cudaBuildBytes

Moved from `decoder/fitguard.go` (the comment above `cudaBuildBytes` in `fitCheck`) on 2026-10-10.

```text
cudaBuildBytes is the HOST peak of building a CUDA C' expert cache (--backend cuda
--moe-cache-experts) from a plain .gguf: 2*weightBytes + expertBytes. Zero when that path is
not in play. It REPLACES the weights+KV+srcFileBytes total when it is larger rather than adding
to it, because the phases do not overlap: decoder.Load unmaps the source file before
cuda.BuildResident starts, and on the resident path KV lives in VRAM.

MEASURED 2026-09-19 on the real gpt-oss-20b (docs/measurements/cold-user-2026-09-18-nobara-pc.md
follow-up), GC-traced with a per-region /proc breakdown: after Load the canonical weights sit on
the Go heap (~13 GB live); BuildResident then host-packs every layer (a second, packed copy of
the same size) and cacheWQ copies each expert stack into pinned host memory (~10 GB, counted
as neither anon nor file). Peak RSS 39.1 GB against this model's 2*12.2 + 11.1 = 35.5 GB — the
~9% remainder is Go heap slack, not priced here. Before cuda.packWeightStack stopped regrowing
its slices the same load reached 50 GB+ and was killed unfinished.
```

## fitCheck.remedy

Moved from `decoder/fitguard.go` (the comment above `fitCheck.remedy`) on 2026-10-10.

```text
remedy names -stream-weights (a .gguf source) or GOINFER_NO_FIT_GUARD=1 (a safetensors
directory) and says what it will do, because the user who reads this message is by definition
the one who did not know the option existed.

M-30 (docs/audit-2026-09-10.md): this used to return the -stream-weights text unconditionally,
on the stale claim that only a .gguf can reach a refusal. P9(b) (b7715ca) made a safetensors
DIRECTORY reachable here too (it now prices those, correctly — before it they silently always
"fit"), and -stream-weights genuinely does nothing for one: serve's manual and auto-retry gates
are both .gguf-suffix-only, and decoder.Load ignores Options.StreamWeights for a directory
input — so the flag was being recommended as a fix that could not possibly change anything,
producing an identical refusal after the user did what they were told.

cmd/prequant builds a .giw from a directory one layer at a time since 2026-10-09 (StreamTranscodeDir,
docs/tasks/task-prequant-dir-streaming-2026-10.md): it no longer loads the checkpoint whole, so it no longer hits this
guard, and it is the remedy for a directory (goinfer-serve and goinfer-chat already take that route unless --direct-load).
Before that, the directory transcode loaded fully resident and could not help, so GOINFER_NO_FIT_GUARD=1 was named
instead; it stays in the text for the case the 70% margin is being conservative about.
```

## FitDeclineError.DenseStreamable

Moved from `decoder/fitguard.go` (the comment above `DenseStreamable` in `FitDeclineError`) on 2026-10-10.

```text
DenseStreamable is true when a -stream-weights retry after this refusal would engage
decoder/layerpaging.go's windowed dense pager — the mechanism tasks/task-fit-to-hardware.md's
CPU placement piece measured as sound for an AUTOMATIC retry
(docs/tasks/task-gpu-paths-2026-09.md). It is false for MoE models and "own-forward" families
(gemma4, nemotron-h-moe, lfm2): MoE CPU weight streaming is a documented, MEASURED failure
mode instead — docs/benchmarks.md "M35/M26 on the Mac" ran a real 20 GB MoE checkpoint
through the CPU-staged --stream-weights path for 2h10min with ZERO completions (RSS pinned
at ~3.2 GB against a 20 GB model — re-reading weights from disk essentially every token, no
useful cache retention), and that run is very likely what produced a genuine kernel panic on
this machine shortly afterward. An automatic retry into that path would risk repeating the
same incident silently, so it stays a manual, explicit choice (-stream-weights typed by
hand) rather than something the guard does on the caller's behalf.
```

## fitCheck.declineErr

Moved from `decoder/fitguard.go` (the comment above `alt` in `fitCheck.declineErr`) on 2026-10-10.

```text
Suggesting a smaller quant to someone already at int4 is noise, and noise in a refusal is
how the useful line gets skipped.

AND int4 IS NOT ALWAYS THE SMALLER ONE. On arm64-with-dotprod (and AVX2-without-VNNI) the
loader keeps a repacked second copy of the nibbles beside the canonical ones, so int4
measures ~1.25 bytes/element against int8's ~1.02 — MORE resident memory, not less
(measured in CI on darwin/arm64, docs/tasks/task-first-hour.md). Offering "int4, the smallest"
there would send a user who is already out of memory in the wrong direction, so the line is
derived from the same measurement the arithmetic above uses rather than from the nominal
bit width.
```

## guardFit

Moved from `decoder/fitguard.go` (the comment above `guardFit`) on 2026-10-10.

```text
guardFit runs the check. It returns the context to PIN — 0 meaning "leave the caller's request
alone", nonzero meaning "the guard chose this smaller one, apply it" — and the refusal, or
(0, nil) to proceed unchanged. It prints the arithmetic to stderr when the load is within
fitWarnRatio of refusing, or when it auto-pins.

R13 (docs/measurements/cold-user-2026-09-07-macbook-arm64.md): before this, an unpinned load
that did not fit at its own maximum context simply loaded anyway (kvBytes was 0, so `fits()`
only ever saw the weight term) — the guard existed and said nothing, because nothing asked it
the question a real request would ask. Three outcomes now, in order: an explicit pin that does
not fit is REFUSED (G-07: an explicit request that cannot be honoured is refused, not silently
downgraded); an unpinned load that does not fit at the model's maximum but DOES fit at some
smaller context ≥ ctxFloor is auto-pinned to that context, reported, and proceeds; an unpinned
load that does not fit even at ctxFloor is refused, same as a pinned one.
```

## giwMemMargin

Moved from `decoder/fitguard.go` (the comment above `giwMemMargin`) on 2026-10-10.

```text
giwMemMargin is S4's own literal (task-never-swap-2026-09.md, item 1): "HostRAMAvailableBytes
minus a 1 GB margin". A .giw load's WEIGHTS are file-backed (S0's own table: zero-copy aliases,
evictable under memory pressure — decoder.Load's own .giw branch never calls loadWeights at
all), so guardGIWFit deliberately does NOT reuse fitMemFraction's 70%-of-available conservatism
(that number is sized for a load that genuinely commits anonymous memory for its weights); only
KV and scratch are real anonymous cost here, and a flat margin against the live probe is what
the brief registered.
```

## guardGIWFit

Moved from `decoder/fitguard.go` (the comment above `guardGIWFit`) on 2026-10-10.

```text
guardGIWFit is item 1's load-time check for a .giw load: refuse, or auto-pin to a smaller
context, when KV + scratch would exceed CURRENTLY AVAILABLE memory minus giwMemMargin. Mirrors
guardFit's own return shape (0 = no pin needed) so decoder.Load can treat either path the same
way. Every unknown proceeds, same rule as guardFit: no cfg, no live probe reading, or no usable
context all mean "cannot price this, so do not refuse a load that might have been fine."

Scratch is priced at prefillAttnScratchBudget alone (the FIXED attention-scratch cap every
prefill already enforces) — prefillScratchBytes' other term (gate/up MLP activations) scales
with a specific PROMPT's length, which load time does not know; this is the same "state a scope
cut rather than guess" prefillScratchBytes' own doc comment already takes for its fixed half.

NOT priced here: a Metal buffer-copy term (S4's own item 1 also names "(Metal) buffer
projection" — metal/model.go's int4Buf/int4Concat make a real host copy per dense projection on
that backend, unlike CPU's pure mmap alias). decoder cannot import metal (metal imports
decoder), so that term needs its own hook — not built in this pass; CPU is this function's only
backend today.
```

## resolveWeightCacheBudget

Moved from `decoder/fitguard.go` (the comment above `resolveWeightCacheBudget`) on 2026-10-10.

```text
resolveWeightCacheBudget is S4 item 2 (task-never-swap-2026-09.md): when the caller leaves
WeightCacheBytes at its 0 "auto" default, price the streamed-weight budget from THIS platform's
own live probe (hostRAMAvailable — vm_stat on darwin, /proc/meminfo on linux) rather than
leaving it entirely to aikit's mmap.AutoBudget(), which is Linux-only (reads /proc/meminfo
itself) and falls back to a FIXED 8 GB on every other platform including darwin — a number with
no relation to what this machine actually has free right now. Same math AutoBudget itself uses
(half of available), so an "auto" request costs nothing new when the live probe already agrees
with what AutoBudget would have found on Linux; it only fixes the darwin case AutoBudget cannot
see (S0's own finding: the pager's auto budget was Linux-only with an 8 GB darwin fallback).

requested > 0 (an explicit --weight-cache) passes straight through unchanged — this resolves
only the "auto" (0) case. Falls through to 0 (aikit's own AutoBudget, now genuinely the WORST
case rather than the only one) when this platform's own probe is unavailable too, logging that
so the fallback is visible rather than silent.
```

## quantBytesPerElem

Moved from `decoder/fitguard.go` (the comment above `quantBytesPerElem`) on 2026-10-10.

```text
quantBytesPerElem is the resident cost of one weight ELEMENT of a 2-D matmul matrix under each
quant mode — MEASURED by running a probe matrix through the loader's own quantization, not
derived from the nominal bit width.

WHY IT IS MEASURED. The arithmetic answer ("int4 is 0.5 bytes plus a scale per group of 32, so
0.625") is right about the encoding and wrong about the FOOTPRINT, because the loader repacks:
repackW4A8Row4IfEligible on arm64 and repackW4A8SplitHalfIfEligible on AVX2-without-VNNI amd64
both ALLOCATE A SECOND BUFFER and keep the canonical nibbles alongside it, so an int4 weight
really costs about twice its encoding on those hosts. wmBytes counts both, correctly.

Caught by CI, not by reasoning: TestFitEstimate_agreesWithResidentWeightBytes passed on
linux/amd64 (ratio 0.96) and failed on darwin/arm64 at ratio 0.53 — estimate 104256 against
195584 accounted. Apple Silicon is exactly the platform the fit guard exists for
(docs/measurements/cold-user-2026-09-06.md was a 16 GB M1 Pro), so a constant tuned on the
developer's box was ~1.8x low precisely where it mattered. Measuring through the real path
tracks the arch, the CPU features, and any repack added later, none of which a constant can.

THE RESIDUAL, stated rather than hidden. The probe is 256x256, which the repacks accept (rows a
multiple of 4, cols a multiple of int4GroupSize). A model built entirely from matrices the
repack REJECTS would be over-priced by up to that factor — the direction that can refuse a
model which would have fit. Real transformer matrices are multiples of 4 and 32 by
construction, the 70% budget carries slack of its own, and GOINFER_NO_FIT_GUARD is named in the
refusal; that is the trade, taken deliberately, because the alternative was a guard that
under-reports by ~2x on the platform it was written for.
```

## estimateKVBytes

Moved from `decoder/fitguard.go` (the comment above `estimateKVBytes`) on 2026-10-10.

```text
estimateKVBytes is the KV cache at ctx positions.

R13 (docs/measurements/cold-user-2026-09-07-macbook-arm64.md): this used to return 0 whenever
no context was explicitly pinned, reasoning that "the CPU cache grows with the conversation
rather than being allocated up front, so counting a context nobody asked for would refuse
models that run fine for short turns." That reasoning is true about short turns and wrong about
what a user actually sends: a 7B int4 model priced at "79% of budget" (KV priced at 0) reached
14 GB RSS and swapped the machine hard on its first real agent request — an opencode system
prompt plus tool schema, tens of thousands of tokens, well inside the model's own context
window. "No context pinned" does not mean "no KV ever allocated"; it means the ceiling is
whatever the model's own maximum context is, because nothing else bounds the CPU/Metal-staged
KV cache's growth. ctx is now the caller's job to choose correctly (fitCheckFor picks
opts.ResidentContext when pinned, else cfg.MaxPositions) — this function just prices whatever
it is given.
```

## estimateKVBytes.geometry

Moved from `decoder/fitguard.go` (the comment above the `resolveArchitecture` call in `estimateKVBytes`) on 2026-10-10.

```text
M-28 (docs/audit-2026-09-10.md): kvBytesPerPosition's flat formula overpriced hybrid
(DeltaNet/conv/Mamba), sliding-window, and MLA models 3-7x. Resolve the real per-layer
geometry when possible (kvBytesForCtx, decoder/arch.go) and fall back to the flat formula
only when the architecture cannot be resolved at all — not a real load (every real GGUF/
safetensors config that reaches this point already resolved one further up in
fitCheckFor/denseStreamable), but a synthetic Config with no registered model_type, the
shape several of this file's own unit tests construct directly.
```

## fitCheckFor

Moved from `decoder/fitguard.go` (the comment above `fitCheckFor`) on 2026-10-10.

```text
fitCheckFor assembles the check for a load that has not happened yet. It prices both a .gguf
and a safetensors directory (P9b, docs/multimodal.md) the SAME way — shape-only, quant-priced
element counts (estimateGGUFWeightBytes / estimateSafetensorsWeightBytes) — never from on-disk
file size: a safetensors checkpoint is usually f32 or bf16 on disk and shrinks several-fold once
quantized on load, so pricing the on-disk bytes would refuse models that fit comfortably. An
estimate that is wrong in the refusing direction is worse than none, which is why this waited
for the shape-based technique rather than shipping the naive (and wrong) file-size one earlier.
Anything neither format resolves (a bare .giw path, an unreadable config, in-flux directory) is
"unknown ⇒ proceed", same as always.

PRICED AGAINST CURRENTLY-AVAILABLE MEMORY, NOT TOTAL RAM (R13-follow-on,
docs/measurements/cold-user-2026-09-07-macbook-arm64.md's SECOND live re-run). The first
version of this function read hostRAM() — total physical RAM, a fixed number that assumes
nothing else on the machine ever needs more than the 30% fitMemFraction reserves. The live
re-run of R13's own fix (which changed prefill_budget.go's request-time check the same way)
found the load-time guard's version of this bug too: on a real, shared Mac, swap began within
15 SECONDS OF LOAD COMPLETING, with the server sitting idle and no request in flight yet — proof
the "30% of total RAM is always enough for everything else" assumption is what was actually
wrong, not merely a per-request pricing gap. Weights ARE still subtracted here (unlike
prefill_budget.go's request-time check): at LOAD time the weights this call is about to allocate
are NOT YET resident (guardFit runs before loadWeights, decoder/model.go), so the current
availability figure does not yet reflect their cost the way it does for an already-loaded model.
```

## kvPricingFor

Moved from `decoder/fitguard.go` (the comment above `kvPricingFor`) on 2026-10-10.

```text
kvPricingFor is what both host guards (priceCtxAndKV for .gguf/safetensors, guardGIWFit for .giw) price KV at: what
the load will actually allocate (A3, docs/completed/task-audit-followups-2026-10-06.md).

  - A load that will be Metal-resident — Metal compiled in, the resolved backend, and the architecture inside Metal's
    feature gate — holds f16 KV (the only KV Metal ships) for MetalCtxDefault positions unless the caller pinned a
    context, clamped to the model's window. Until 2026-10-07 the guards priced it at Options.KVPrecision (f32 unless
    -kv f16) over the model's whole window: on Gemma 4 E2B, 3.6 GB against the ~75 MB the resident holds, enough to
    refuse or pin down a load that fits.
  - Every other load keeps the old pricing: the CPU allocates KV per request, at Options.KVPrecision, up to the
    window, and that ceiling is what a long request reaches (R13).

A Metal resident that then declines for memory falls back to the CPU, whose per-request KV this no longer prices at
load. The weights term is unchanged by that fallback, and serve's -require-backend refuses rather than fall back.

ok is false when nothing is pinned and the model's own maximum is unknown: nothing to price.
```

## fitCheck.smallerFittingContext

Moved from `decoder/fitguard.go` (the comment above `fitCheck.smallerFittingContext`) on 2026-10-10.

```text
smallerFittingContext solves for the largest context ≤ f.effCtx whose weights+KV fit the
budget, floored at ctxFloor. Only meaningful when the caller did not pin a context — a pin is
an explicit request and is refused outright rather than silently downgraded (see guardFit).

M-28 (docs/audit-2026-09-10.md): this used to divide the budget by a single flat per-position
rate, exact only because the flat formula priced every position identically. estimateKVBytes
is no longer exactly linear in ctx once a sliding-window layer's cost flattens past its own
window — but it IS still monotonic non-decreasing (more context never needs LESS KV), so a
binary search finds the largest fitting ctx exactly, the same guarantee the division used to
give for free.
```

## resolveArchitecture.bounds

Moved from `decoder/registry.go` (the comment above `validateConfigBounds` in `resolveArchitecture`) on 2026-10-10.

```text
M-10(b): BOUND THE CONFIG BEFORE THE ADAPTER RUNS. Several adapters allocate
NumLayers-sized slices with only a `> 0` check (qwen3_next, llama4), and loadConfig has
no bound at all — so a 300-byte .giw or a hostile safetensors config.json declaring
num_hidden_layers: 68719476736 is a FATAL out-of-memory, not the typed error
LoadSerializedWeights' doc promises. Under Go's maxAlloc, so no recover() catches it.

Here rather than at the two JSON chokepoints the audit names: this is the single point
every path reaches — .giw, safetensors, GGUF and whatever is added next — and putting it
at the callers would be the "one predicate, N consumers" shape that produced half the
findings in this audit. The GGUF paths bound some of these already; re-checking costs a
handful of comparisons once per load.
```

## Architecture.validateResolved

Moved from `decoder/registry.go` (the comment above `Architecture.validateResolved`) on 2026-10-10.

```text
validateResolved catches descriptor fields an adapter left at their zero value when zero
is not a legal setting. Every adapter builds an Architecture by hand from a struct
literal, so a field simply omitted is a compile-clean, load-clean, silently-wrong model.

Both fields here were live bugs in lfm2Architecture, found 2026-08-31 against HF:

  - AttnScale 0 makes every q·k score 0, so softmax returns a UNIFORM average over the
    context. Invisible at one token (softmax of a single element is 1.0 whatever the
    scale) and invisible in any greedy smoke test that only reads argmax, which matched
    HF anyway. It showed up as cosine 0.928 at five tokens.
  - NormEps 0 divides by rsqrt(variance) with no floor. Not merely imprecise: on a small
    first-layer variance it scaled the norm output by a uniform 1.0185x.

Checked here rather than in each adapter because the point is to cover the families
nobody has written yet. An arch that genuinely wants no attention scaling sets 1.0.
```

## Architecture.validateResolved.position

Moved from `decoder/registry.go` (the comment above the position-information check in `validateResolved`) on 2026-10-10.

```text
M-06: POSITION INFORMATION MUST COME FROM SOMEWHERE. finalizeRoPE treats
RoPEGlobalBase <= 0 as "no tables", and applyRoPE is a silent no-op on an empty table,
so an adapter that never reads rope_theta loads clean and generates fluent,
POSITION-BLIND text — and drops YaRN with it. gpt-oss and llama4 both read only the
flat rope_theta, and transformers >= 5.10 nests it under rope_parameters; for
llama/mistral/qwen3 that is a loud error, and for these two it was silence.

The four legitimate ways to have no global RoPE table are named explicitly rather
than inferred, so a new family that simply forgot cannot look like one of them:
GPT-2 has learned positions, Nemotron-H encodes NoPE layers as base 0, MLA carries its
own decoupled rope dims, and Olmo Hybrid's released checkpoint genuinely has none at
all (NoPositionEncoding, verified against rope_parameters: {"rope_theta": null}).
```

## qwen3MoeArchitecture

Moved from `decoder/registry.go` (the comment above `qwen3MoeArchitecture`) on 2026-10-10.

```text
qwen3MoeArchitecture expresses Qwen3-MoE (Qwen3-30B-A3B / Qwen3-Coder-30B-A3B-
Instruct, both model_type "qwen3_moe" — confirmed against both real released
config.json files, config-identical apart from max_position_embeddings):
qwen3's dense attention (per-head q_norm/k_norm, GQA, no q/k/v bias,
1/√head_dim scale, single-base RoPE) with the FFN replaced on every layer by a
sparse MoE — qwen2_moe's router shape (top-k of num_experts at
moe_intermediate_size, norm_topk_prob) but with NO always-on shared expert.
Verified against a real GGUF file's header too (unsloth/Qwen3-30B-A3B-GGUF
Q2_K, HTTP-Range-fetched): architecture string "qwen3moe", plain
{arch}.attention.*/{arch}.expert_*/{arch}.rope.freq_base metadata (no
sliding-window or YaRN keys), and a tensor set with attn_q_norm/attn_k_norm +
ffn_gate_inp/ffn_{gate,up,down}_exps but no ffn_*_shexp — so the existing
generic GGUF loadLayer path (gated on arch.QKNorm / arch.MoE /
arch.MoE.SharedIntermediateDim>0) handles this family with no new loader code,
same as the safetensors path. The tensor schema is qwen3MoeTensorSchema.
```

## backfillFlatRope

Moved from `decoder/registry.go` (the comment above `backfillFlatRope`) on 2026-10-10.

```text
backfillFlatRope fills the flat rope_theta / rope_scaling fields from transformers >=5.10's
rope_parameters object, for the SINGLE-BASE architectures (llama, mistral, qwen3).

transformers moved RoPE config out of top-level rope_theta/rope_scaling and into
rope_parameters — {"rope_theta": 1e4, "rope_type": "default"}, with linear/yarn/llama3
scaling carried inside the same object. Archs that only read the flat fields therefore
REJECT any checkpoint saved by a current transformers, with "rope_theta must be >0" — a
hard load failure on freshly re-saved upstream weights, not a niche path. phi3 already
handled this via parseRopeFlat and gemma3/mellum handle the per-layer-type nesting
(full_attention/sliding_attention); llama and mistral did not, and each has its own
architecture func, which is exactly how one got fixed and the other did not. Hence one
helper rather than a third copy.
```

## smollm3Architecture

Moved from `decoder/registry.go` (the comment above `smollm3Architecture`) on 2026-10-10.

```text
smollm3Architecture expresses SmolLM3-3B (HuggingFaceTB/SmolLM3-3B, model_type "smollm3"): a
plain llama-shaped dense GQA model — tensor names byte-identical, `llamaTensorSchema` reused
verbatim — with per-layer NoPE on 9 of 36 layers via `no_rope_layers`, reusing the SAME
Config field and boolean convention llama4_text already established (`NoRopeLayers[i]==1` ⇒
layer i USES RoPE, `==0` ⇒ NoPE), not a new one — and `layerNoPE`, the SAME generic
Architecture hook cohere2Architecture already populates for its own global-layer NoPE. The
`layer_types` field on real released checkpoints is a RED HERRING here: every entry reads
"full_attention" regardless of which layers are actually NoPE (confirmed against the real
config.json, not assumed) — `no_rope_layers` is the only authoritative source, unlike
Gemma/cohere2 where `layer_types` itself carries the split.

THE FIELD NAME IS THE OPPOSITE OF ITS OWN VALUES, verified against the real
modeling_smollm3.py rather than guessed from the name (the exact class of silent-wrong bug
this repo's own culture names repeatedly): `self.use_rope = config.no_rope_layers[layer_idx]`
— a "no_rope_layers" entry of 1 means the layer HAS rope, 0 means NoPE. The real released
config's list is `[1,1,1,0]` repeating (0 at every 4th layer, 0-indexed positions 3,7,11,...) —
checked against `configuration_smollm3.py`'s own generation formula
(`(layer_idx+1) % no_rope_layer_interval != 0`) for when a checkpoint omits the explicit list,
which independently confirms the every-4th-layer pattern the brief itself named. Getting the
polarity backwards would silently flip 27 RoPE layers to NoPE and 9 NoPE layers to RoPE —
correct shapes, plausible logits, wrong model — with no crash to catch it.
```

## olmo3Architecture

Moved from `decoder/registry.go` (the comment above `olmo3Architecture`) on 2026-10-10.

```text
olmo3Architecture expresses Olmo 3 (allenai/Olmo-3-{7B,32B}, model_type "olmo3"): a
softmax-GQA/MHA model with two real departures from every existing family, BOTH verified
against the real `modeling_olmo3.py` rather than assumed from the brief's own framing:

 1. **NormPostOnly**: there is NO pre-norm at all — confirmed by instantiating
    `Olmo3ForCausalLM` and reading its `state_dict()`, which has no `input_layernorm` tensor
    anywhere. `Olmo3DecoderLayer.forward` reads the RAW residual stream directly into both
    `self_attn` and `mlp`, and normalizes each sublayer's OUTPUT (`post_attention_layernorm` /
    `post_feedforward_layernorm`) before the residual add. Genuinely different from
    `NormSandwich4` (which normalizes BOTH input and output) — a new placement, not a variant.
 2. **QKNormWhole**: QK-norm is computed over the FULL projected q/k vector
    (`Olmo3RMSNorm(config.num_attention_heads * self.head_dim, ...)`), one statistic over
    `num_heads*head_dim` elements, not the standard per-head convention (Qwen3/Gemma3/Mellum
    normalize each head independently over `head_dim`). Reuses the SAME `rmsNorm` function
    with rows/dim swapped (`QKNormWhole`'s own comment on `Architecture`), not new math.

Otherwise plain: `num_key_value_heads == num_attention_heads` on the real release (MHA, not
GQA — checked, not assumed), sliding-window on 3 of every 4 layers (`layer_types`, reusing
`Config.IsGlobalLayer` — the same authoritative layer_types-then-pattern rule cohere2 already
uses), `tie_word_embeddings: false`. Tensor names are llama-shaped except the post-only norms
and whole-vector QK-norm weight width — `olmo3TensorSchema`.

A THIRD real finding, on RoPE, corrected 2026-09-12 (a prior revision of this comment had it
backwards — see below): sliding-attention and full-attention layers use GENUINELY DIFFERENT
rotary tables, at the SAME theta but different scaling — full gets YaRN (mscale 1.1 on this
fixture), sliding gets plain unscaled RoPE (mscale 1.0, no NTK-by-parts interpolation). This
is the ORIGINAL family design (the local/global RoPE split Mellum already implements,
`RoPELocalBase`/`RoPEGlobalBase` + `ropeScaling`/`ropeScalingLocal`, dispatched on
`arch.layerIsGlobal`), and `TestOlmo3_forwardParity` scored 0.9999999999997883 under it at
ship time (docs/completed/task-families-2026-09.md G2).

**Verified by calling the real forward, not by reading source.** A prior revision of this
comment ("0b0f5c9") read `modeling_olmo3.py` and concluded `Olmo3Model.__init__` builds ONE
shared `self.rotary_emb` and calls it once per forward with no per-layer distinction — citing
the real-checkpoint T3 gate's cosine 0.992789 as confirmation of a "24 of 32 layers at the
wrong frequency" bug, and switching every layer to `full`'s YaRN table discarding `sliding`
entirely. That reading was WRONG for transformers 5.15.0 (this repo's pinned version, the same
one `pin_olmo3_tiny.py` uses): `Olmo3RotaryEmbedding.forward` takes an explicit `layer_type`
argument and returns a DIFFERENT (cos, sin) pair per call — confirmed by instantiating the real
class and calling `rotary_emb(hidden, pos_ids, "full_attention")` vs `(..., "sliding_attention")`
directly: sliding's cos/sin come back with `attention_scaling == 1.0` and no YaRN
interpolation; full's come back YaRN-scaled. The forward signature REQUIRING a layer_type
argument is itself the tell that a single shared table cannot be what runs — a uniform table
would need no such argument. The "uniform" fix's own T3 finding (cosine 0.992789) was real,
but the fix over-corrected: it likely traded one wrong frequency table (whatever the T3
checkpoint's bug actually was) for a table that is now wrong on 3 of every 4 layers instead of
being right on all 4 — this tiny golden's post-fix cosine (0.98997287, argmax still exact) is
the same magnitude-only-drift signature as the original bug, on the layers this fix touched.
`base` (theta) IS the same value on both layer types on the real release — only the SCALING
differs, which is why `localBase`/`base` collapse to the same number below but
`scalingLocal`/`scaling` do not.

The flat-top-level branch (the real 7B/32B release's on-disk form) is unaffected by this
correction: `PretrainedConfig`'s `standardize_rope_params` expands it into the identical
nested full/sliding split at construction time, so both branches now agree.
```

## olmo3Architecture.rope

Moved from `decoder/registry.go` (the comment above the nested `rope_parameters` branch in `olmo3Architecture`) on 2026-10-10.

```text
NESTED {"full_attention": {...yarn...}, "sliding_attention": {"rope_type": "default",
"rope_theta": <same theta>}} — see the doc comment above (found 2026-09-12):
`full`/`sliding` are BOTH real and DIFFERENT, not "sliding discarded, one uniform
table" — verified by actually calling `Olmo3RotaryEmbedding.forward(hidden, pos,
layer_type)` for both layer types against transformers 5.15.0 (the same install that
generates olmo3_forward_full.json): sliding_attention's cos/sin come back PLAIN
(attention_scaling 1.0, no YaRN interpolation), full_attention's come back YaRN-scaled
(mscale 1.1 on this fixture). The forward signature itself requiring an explicit
layer_type argument is the tell — a single shared table would need none. A prior
revision of this comment ("0b0f5c9") concluded the opposite from reading
modeling_olmo3.py's forward rather than calling it, and that reading was wrong for
this transformers version. base (theta) is the same value on both layer types on the
real release, so only the SCALING differs — sliding gets no scaling at all.
```

## ministral3Architecture

Moved from `decoder/registry.go` (the comment above `ministral3Architecture`) on 2026-10-10.

```text
ministral3Architecture expresses Ministral 3 (mistralai/Ministral-3-{3b,8b,14b}, model_type
"mistral3" — the OUTER Mistral3ForConditionalGeneration wrapper's type, which `loadConfig`'s
generic text_config flattening re-applies LAST over whatever the nested text_config's own
model_type ("ministral3") set; confirmed by reading that flattening code directly rather than
assumed, since it decides which registry key this family actually resolves under): Mistral's
GQA skeleton (reused verbatim: same tensor names, confirmed by instantiating
Ministral3ForCausalLM directly and reading its state_dict) with two real deltas Phase 0 found,
both checked against the released config rather than assumed from the brief's own framing:

 1. `sliding_window: null` on the real release — mistralArchitecture already treats
    SlidingWindow<=0 as full attention (its own "0 ⇒ full attention" comment), so this needs
    no new code; the brief's own caution ("verify... whether it is every layer") was answered
    "there is no window at all", not "yes, every layer".
 2. `rope_parameters` is `rope_type: "yarn"` with a real, load-bearing THIRD field alongside
    the standard YaRN ones: `llama_4_scaling_beta`. Verified against the real
    modular_ministral3.py (not guessed): `get_llama_4_attn_scale` multiplies the QUERY by
    `1 + beta·ln(1 + floor(pos/original_max_position_embeddings))`, AFTER RoPE, on EVERY
    layer — the exact formula llama4Architecture's own attnTemp/floorScale primitive already
    implements (`decoder/forward_llama4.go`), but Llama4 applies it INSTEAD of RoPE on NoPE
    layers only, never combined with RoPE the way this family needs. Generalized to the two
    new Architecture fields AttnTempBeta/AttnTempOrigMaxPos (see their own comment) and wired
    into the GENERIC causalAttention/forwardN paths rather than copied into an own-forward
    function, since every existing family leaves both fields at their zero-value no-op.

Also confirmed: `mscale`/`mscale_all_dim` (both 1.0 on the release) are DeepSeek's own spelling
of the YaRN attention_factor, not the generic `attention_factor` key parseRopeScaling reads —
left unhandled, its own default (0.1·ln(16)+1 ≈ 1.277) would silently override the correct
value (1.0, since mscale == mscale_all_dim here, same reasoning deepseekArchitecture's own
comment gives for V2-Lite). Overridden the same way deepseekArchitecture already does.
```

## gpt2Act

Moved from `decoder/registry.go` (the comment above `gpt2Act`) on 2026-10-10.

```text
gpt2Architecture expresses GPT-2: the GPT-2/NeoX class
that breaks the Llama mold on several axes — LayerNorm (mean-centered, with
bias) instead of RMSNorm, learned absolute position embeddings instead of
RoPE, a non-gated GELU MLP (up→gelu→down) instead of a gated one, fused q/k/v
with bias, an attention output bias, and tied embeddings. The Conv1D weight
layout + fused projections need a dedicated loader (buildGPT2Weights), so
this returns the gpt2TensorSchema as a marker; the schema's field names are
unused.
gpt2Act maps GPT-2's activation_function to the ActKind that actually implements it.
"gelu_new" (and the empty default, which is GPT-2's own) is the TANH approximation;
"gelu" is the exact erf function. validateGPT2 accepts both, and before this they both
ran geluTanh — so a checkpoint declaring the exact function silently got the
approximation. The two differ by up to 4.73e-4, small enough to pass unnoticed and
still wrong. Every shipping GPT-2 config declares gelu_new, so nothing in tree moves.
```

## qwen3_vlArchitecture.mrope

Moved from `decoder/registry.go` (the comment above the m-RoPE section check in `qwen3_vlArchitecture`) on 2026-10-10.

```text
The released Qwen3-VL checkpoints write rope_scaling {mrope_interleaved: true, mrope_section: [24, 20, 20],
rope_type: "default"}: m-RoPE is marked by the section being there, not by the type (S10, found by G-S10a's
load printing MRopeSection=[] on Qwen3-VL-2B-Instruct). A section present is taken whatever the type says.
```

## lfm2Architecture

Moved from `decoder/registry.go` (the comment above `lfm2Architecture`) on 2026-10-10.

```text
lfm2Architecture expresses LFM2 / LFM2.5 (model_type lfm2): a gated-short-convolution +
softmax-attention hybrid. Every layer has a SwiGLU FFN; layer_types decides whether its
mixer is a conv block (22 of 30 on LFM2.5-2.6B) or GQA attention with per-head RMSNorm on
Q and K (8 of 30, at 2/5/9/13/17/21/24/27).

It is EXPERIMENTAL tier: validated against the HF reference on a real checkpoint, not
against a full-model T3.

Three facts here were checked against the released LFM2.5-2.6B rather than inherited from
the original scoping brief, and two of them contradicted it:

  - QK-norm is RMSNorm, not LayerNorm. The brief said LayerNorm; the reference uses
    Lfm2RMSNorm(head_dim) per head, and the checkpoint carries q_layernorm.weight with NO
    bias tensor anywhere in its 266. That is the difference between reusing the existing
    hardcoded QK-norm path and writing a bias-carrying LayerNorm variant.
  - vocab is 128,000 (the brief said 65,536, which is the older LFM2-2.6B tokenizer), and
    rope_theta is 1e7 (was 1e6).
  - intermediate_size is STATED (10752), not computed from block_multiple_of — so the
    block_ffn_dim_multiplier / block_multiple_of machinery is inert here and is not read.
```

## graniteDenseArchitecture

Moved from `decoder/registry.go` (the comment above `graniteDenseArchitecture`) on 2026-10-10.

```text
graniteDenseArchitecture expresses dense Granite 4.2 (ibm-granite/granite-4.2-{3b,8b,30b},
model_type "granite", GraniteForCausalLM): a plain llama skeleton — confirmed byte-identical
tensor names (self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj, input_layernorm/
post_attention_layernorm, no bias, no QK-norm) by instantiating GraniteForCausalLM directly and
reading its state_dict, not assumed from Granite-4.0-H's own tensor names — plus Granite's four
scalar multipliers, THREE of which are already generic on Architecture (embedding_multiplier →
EmbedScale, attention_multiplier → AttnScale in place of 1/√d, logits_scaling → LogitScale;
granitemoehybrid's own comment on EmbedScale notes it applies "the Gemma sqrt path", but the
mechanism itself — multiply the embedding by a constant — is generic regardless of how that
constant is derived). residual_multiplier is the one exception: granitemoehybrid's own-forward
(runLayersGranite) applies it via graniteParams.ResidMul, and the generic uniform-layer forward
this family rides has no such hook. Checked against all three released sizes' real config.json
(3b/8b/30b): every one ships residual_multiplier 1.0 (identity), so validateGraniteDense rejects
anything else rather than silently dropping it — the same discipline validateLlama already
applies to scaled RoPE. Verified against a real GGUF header too (bartowski/granite-4.2-3b-GGUF
Q2_K, HTTP-Range-fetched): architecture string "granite", metadata carries the multipliers
directly (attention.scale/embedding_scale/logit_scale/residual_scale) and the tensor set is
exactly llama's — the tensor schema is llamaTensorSchema, reused rather than duplicated.
```

## deepseekArchitecture.attnScale

Moved from `decoder/registry.go` (the comment above `AttnScale` in `deepseekArchitecture`) on 2026-10-10.

```text
Plain qk_head_dim^-0.5 — NOT a TODO (audit-2026-09-02.md N-34, resolved 2026-09-11):
an older version of this comment claimed the real V2-Lite/V3 fold YaRN's
mscale_all_dim² into this scale and called it unwired, contradicting the OTHER
mscale comment in this same function (parseRopeScaling's caller above, "transformers
5.12 does NOT fold mscale² into it"). The real-model gates settle it:
testdata/parity_manifest.json's deepseek_v2/deepseek_v3 entries are "validated"
against real HF bf16 oracles (DeepSeek-V2-Lite 15.7B, Moonlight-16B-A3B) at
cosine_min 0.999+ with THIS plain scale — a ~2x attention-softmax error from a
missing mscale_all_dim²≈0.5 fold would not read as 0.999. No fold needed; the tiny
golden also uses default RoPE, so no mscale there either.
```

## bailingHybridArchitecture

Moved from `decoder/registry.go` (the comment above `bailingHybridArchitecture`) on 2026-10-10.

```text
bailingHybridArchitecture expresses Bailing Hybrid (inclusionAI, model_type "bailing_hybrid";
Ling 3.0 — Ling-3.0-tiny/flash): DeepSeek-style Multi-head Latent Attention (MLA) alternating
with Kimi Delta Attention (KDA) linear-attention layers every LayerGroupSize-th layer being MLA,
over a DeepSeekMoE FFN — verified field-for-field against the real modeling_bailing_moe_v3.py
(fetched directly, plus a real checkpoint's actual safetensors header via HTTP Range, not
downloaded in full), not the task brief's own paraphrase:

  - layer_types is NOT a config.json field for this family at all (no released checkpoint
    carries it) — the pattern is COMPUTED from layer_group_size, replicated exactly in
    normalizeBailingLayerTypes (including a tail-cleanup clause the brief's own description
    omitted).
  - MLA is reused via forward_deepseek.go's mlaAttention UNCHANGED, parameterized for two real
    departures: both mixers are named self.attention (not self.self_attn — mlaParams.AttnPrefix)
    and the output projection is self.dense (not self_attn.o_proj — mlaParams.DenseSuffix). An
    optional per-head sigmoid output gate (gated_attention_proj_granularity_type) rides the same
    mechanism Laguna's own attention-output gate already ships, activation aside.
  - MoE reuses moeMLP/routeExperts UNCHANGED (sigmoid + expert_bias + group-limited top-k +
    routed_scaling_factor + an ungated shared expert is byte-for-byte DeepSeek-V3's noaux_tc
    shape — confirmed from BailingMoeV3Gate/SparseMoeBlock's own forward, not assumed from field
    names), but this family spells its expert counts num_experts/num_shared_experts (not
    DeepSeek's n_routed_experts/n_shared_experts) and its bias buffer expert_bias (not
    e_score_correction_bias) — real, checked-not-assumed naming departures.
  - KDA (kda.go) is the one genuinely new primitive: a delta-rule recurrence structurally
    identical to Gated DeltaNet but with a PER-CHANNEL decay (fla-org/flash-linear-attention's
    actual source, not HF's opaque Triton-kernel call — see F4's own rehearsal,
    docs/completed/task-families-2026-09.md). Only the no_kda_lora + kda_safe_gate variant Ling-3.0-tiny's
    own config selects is implemented; validateBailingHybrid refuses anything else rather than
    silently mis-running an unimplemented variant.

No YaRN/rope_scaling override is wired (the DeepSeek-style mscale/mscale_all_dim ratio
deepseekArchitecture applies): Ling-3.0-tiny's own released config carries rope_scaling: null,
so nothing exercises it; a future Bailing checkpoint that sets YaRN would need it added, same as
deepseekArchitecture's own.
```

## spark25Architecture

Moved from `decoder/registry.go` (the comment above `spark25Architecture`) on 2026-10-10.

```text
spark25Architecture expresses Spark-X2.5 (model_type spark2_5; XHToken/Spark-X2.5-{1.7B,4B}).
docs/tasks/task-spark-x2-5.md / docs/audit-2026-09-10.md L-06 scoped this from a survey-level
summary; TWO of its claims did not survive reading the real configuration_spark.py /
modeling_spark.py (fetched directly, trust_remote_code) and are corrected here:

  - The audit says Spark-X2.5 reuses "the parallel block (Command-R)". It does not — its
    decoder layer is the STANDARD sequential residual (norm→attn→residual, norm→mlp→residual),
    byte-for-byte the same shape as Llama/Qwen (NormPre2). No parallel-block wiring needed.
  - The audit says "non-gated GELU". Spark2_5MLP IS gated (SwiGLU-shaped:
    down(gelu(gate(x))*up(x))) — the only real departure from Llama's MLP is the activation
    (exact erf GELU, HF's "gelu", NOT gelu_new/gelu_pytorch_tanh) inside that gated shape. See
    gegluExact (mlp.go) — ActGelu previously only reached the non-gated path.

What IS real, confirmed against source:
  - Fused QKV: one q_k_v_proj Linear(hidden, qDim+2*kvDim), split Q‖K‖V by output rows in that
    order — buildSpark25Weights (weights.go), modeled on buildPhi3Weights's split.
  - Head-wise SIGMOID attention-output gate (g_proj: Linear(hidden, numHeads)), applied
    element-wise to attn_output BEFORE out_proj — exactly Laguna's g_proj STRUCTURE with a
    different activation. This is the one genuinely NEW forward-pass wire: the sigmoid gate
    math (applySigmoidGateRow) already existed for MLA (Bailing Hybrid) but had no hook in the
    generic (non-MLA) attention path Spark2.5 uses — see AttnGate/GateSigmoid (arch.go).
  - Per-layer-type RoPE: full_attention layers get partial_rotary_factor=0.25 (rotary_dim =
    head_dim/4) + theta 5e6; sliding_attention layers get partial_rotary_factor=1.0 (full
    rotation) + theta 1e4. Exactly Laguna's RotaryDim/RotaryDimLocal/RoPEGlobalBase/
    RoPELocalBase mechanism (see lagunaArchitecture above) — reused verbatim, no new plumbing.
  - 1:3 sliding:full interleave (sliding_window 512), read generically from layer_types via
    cfg.IsGlobalLayer — owned by no single family, every existing consumer already shares it.
  - No QK-norm, no embed scale, no logit softcap, no attention sink — the plainest attention
    shape this family touches.
  - rotate_half is the standard NeoX half-split (ropeInterleave stays false, the default).
  - inv_freq is recomputed fresh per forward call in the reference (not a cached
    persistent=False buffer) — no internlm2-class fast-init corruption risk on this family.
```
