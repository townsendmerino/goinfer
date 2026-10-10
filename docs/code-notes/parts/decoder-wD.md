# decoder: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `decoder`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

## applyNormULPNoiseDiag

Moved from `decoder/normnoise.go` (the comment above `applyNormULPNoiseDiag`) on 2026-10-09.

```text
GOINFER_NORM_ULP_NOISE=<seed> is a DIAGNOSTIC (default-off, one env read per load): every f32
norm vector the loaded model carries (pre/post-attention, pre/post-MLP, per-head QK norms, the
final norm) gets an independent −1/0/+1 ULP nudge per element, seeded by <seed>. That is noise
of exactly f32-rounding size (~1.2e-7 relative), dense across every activation — the same
magnitude and shape as the difference between two CORRECT implementations of one forward that
sum in different orders (a GPU's tree reductions vs the CPU's sequential ones).

What it measures: the model's OWN sensitivity to that noise. Run the CPU forward twice, once
with this set and once without, and the logit cosine between the two runs is the floor below
which a resident-vs-CPU cosine on that checkpoint carries no information about the kernels —
the quantized (W4A8/W8A8) forward re-rounds activations to int8 at every projection, and a
perturbation far below one int8 step still flips a fraction of rounding decisions, each by a
whole step, compounding per layer. Measured on a 4-layer phi3-mini-shaped checkpoint (docs/
tasks/task-webgpu-nogqa-decode-bug.md): CPU-vs-CPU under this noise 0.9994–0.9996, argmax flip
at the same prompt position the WebGPU resident path flipped at — indistinguishable from the
resident-vs-CPU gap that had been read as a kernel bug. Observe-only: unset, nothing changes.
```

## knobs.go.header

Moved from `decoder/knobs.go` (the comment above `knobs.go.header`) on 2026-10-09.

```text
Per-model operator knobs (docs/tasks/task-env-config-2026-09.md, phase 2a). These were read from the
process environment on EVERY forward or generation, so changing the environment altered a model that was
already loaded, and two models in one process could not differ. Each model now snapshots them ONCE, at
Load (Options.Knobs overrides the environment per model), and every reader consults the model's snapshot.
The environment variables keep working — they are simply read once, at Load.

The values are kept RAW (as the environment would give them) and each reader below keeps the exact parsing
its os.Getenv call had, so a default or an override means precisely what it meant before.
```

## knobMoECacheExperts

Moved from `decoder/knobs.go` (the comment above `knobMoECacheExperts`) on 2026-10-09.

```text
Phase 2b: read at Load or per call on a loaded model.
```

## knobMoEPreadCPU

Moved from `decoder/knobs.go` (the comment above `knobMoEPreadCPU`) on 2026-10-09.

```text
Phase 6: a rollback switch that had been filed as a diagnostic.
```

## knobSpecAdaptiveNeverYield

Moved from `decoder/knobs.go` (the comment above `knobSpecAdaptiveNeverYield`) on 2026-10-09.

```text
MC4 "spec inside a batch" premise (docs/tasks/task-concurrency-2026-09.md): does the just-shipped
switch's per-round yield ever cost more than it buys on copy-heavy traffic, where staying exclusive
beats joining MC3's batch? Set to force genNgramInto's adaptive round loop to never yield — the
measurement this bisects, not a shipped policy.
```

## cudaKnobs

Moved from `decoder/knobs.go` (the comment above `cudaKnobs`) on 2026-10-09.

```text
cudaKnobs are phase 3's: the CUDA backend's operator knobs, snapshotted here with the rest so one mechanism
(Load-time read, Options.Knobs override, the testhooks drift check) covers every backend. The CUDA resident
reads them through Model.Knob; decoder itself never interprets them.
```

## cudaKnobs.rollback

Moved from `decoder/knobs.go` (the comment above `cudaKnobs.rollback`) on 2026-10-09.

```text
Phase 6: rollback switches for default-on paths, filed as diagnostics until then.
```

## metalKnobs

Moved from `decoder/knobs.go` (the comment above `metalKnobs`) on 2026-10-09.

```text
metalKnobs are phase 4's: the Metal backend's operator knobs, same arrangement as cudaKnobs (read through
Model.Knob by the Metal resident). GOINFER_MOE_EXPERT_MAJOR is shared with the CPU path and already listed.
```

## knobSet

Moved from `decoder/knobs.go` (the comment above `knobSet`) on 2026-10-09.

```text
knobSet is one model's snapshot. A nil *knobSet reads the live environment — the old behaviour — for the
structures tests build by hand without a Model (a hand-made Architecture, scratch or worker pool).
```

## Model.Knob

Moved from `decoder/knobs.go` (the comment above `Model.Knob`) on 2026-10-09.

```text
Knob returns this model's value for one per-model knob and whether it is set: the snapshot taken at Load, with
Options.Knobs applied. It is how a backend reads its own operator knobs (phase 3: CUDA), so they are read once
per model and can differ between two models in one process. name must be on knobs.go's list — an unknown name
panics rather than silently reading "unset", which is what a typo would otherwise do.
```

## testhooks.go.header

Moved from `decoder/testhooks.go` (the comment above `testhooks.go.header`) on 2026-10-09.

```text
Code relocated by the B-08 build-tag pass: these are test-only hooks, compiled
only under -tags goinfer_testhooks so they are NOT part of the public API
(audit B-08). See RELEASING.md. Imports are added to satisfy the moved bodies.

HAND-MAINTAINED, NOT MACHINE-GENERATED — despite this file's former name
(testhooks_gen.go, renamed 2026-09-11, audit-2026-09-02.md N-41). There is no
//go:generate directive and nothing regenerates it; edit it directly like any
other file. Mirrors gpu/testhooks.go and cuda/testhooks.go.
```

## Gemma4ResidentMediaRowForTest

Moved from `decoder/testhooks.go` (the comment above `Gemma4ResidentMediaRowForTest`) on 2026-10-09.

```text
Gemma4ResidentMediaRowForTest is gemma4ResidentMediaRow: the resident row GenerateGemma4VL's E-model resident prefill
builds for an image or audio position (S9).
```

## Model.AutoPinResidentContextForTest

Moved from `decoder/testhooks.go` (the comment above `Model.AutoPinResidentContextForTest`) on 2026-10-09.

```text
AutoPinResidentContextForTest makes the model's resident context request ctx with the caller not having chosen it,
the state the load-time fit guard's auto-pin leaves (R13), which only a load under memory pressure produces.
```

## Gemma4MoEExpertForTest

Moved from `decoder/testhooks.go` (the comment above `Gemma4MoEExpertForTest`) on 2026-10-09.

```text
Gemma4MoEExpertForTest computes ONE gemma4 MoE expert's output on a caller-supplied input xe
([hidden]) — the gelu-tanh GeGLU expert function edown = Down · (geluTanh(gate)·up), gate‖up =
GateUp·xe — and returns it alongside the expert's fused gate‖up and down weight matrices. The
cuda single-expert gate (task 2c) packs those weights, runs the resident chain
(gemv_w4a8_moe → glu_quant act=GELU_TANH → down) on the same xe, and compares: the gelu-tanh MoE
epilogue × the indexed-expert GEMV is a combination that ships in neither Gemma-3 (dense, not
indexed) nor Mixtral/GLM (indexed, but SiLU), so it is verified directly, not argued by
composition. ok=false for a non-gemma4 model, a dense layer, or e out of range.
```

## Gemma4MoERouterForTest

Moved from `decoder/testhooks.go` (the comment above `Gemma4MoERouterForTest`) on 2026-10-09.

```text
Gemma4MoERouterForTest exposes a gemma4 MoE layer's f32 router projection (row-major [nE, hidden])
and its selection bias (zeros — gemma4 has no router bias) for the resident-router idx-equality
unit test (cuda/). That test replays captured router inputs (routerRnBuf) through the CUDA
selection kernels and gates resident idx[] against the CPU idx[], isolating a routing FLIP from
any expert-GEMV numeric difference — the "router first" discipline. ok=false for a non-gemma4
model or a dense (non-MoE) layer.
```

## gemma4HiddenBuf

Moved from `decoder/testhooks.go` (the comment above `gemma4HiddenBuf`) on 2026-10-09.

```text
gemma4HiddenBuf accumulates a COPY of the residual stream after each layer (index 0 =
post-embedding, from g4traceHidden's layer -1; index i+1 = after layer i) on every
runLayersGemma4 call, when capture is on. It backs the per-layer LOCALIZATION a resident-vs-CPU
gate uses (metal/cuda Step 4). Test-only: appended only by the SetGemma4HiddenCaptureForTest
closure below (production only sees the nil-by-default g4traceHidden seam), so it lives here under
goinfer_testhooks — off-tag it would be an unused package var (staticcheck U1000). B-08.
```

## RouterMarginForTest

Moved from `decoder/testhooks.go` (the comment above `RouterMarginForTest`) on 2026-10-09.

```text
RouterMarginForTest returns the per-decision top-k boundary margin (smallest selected expert's
softmax prob minus the largest rejected expert's), same order/index as RouterCaptureForTest. It
is the MoE-specific robustness signal a noise-floor check reads: a fixture whose margin sits below
the int4-vs-f32 routing perturbation can flip top-k under quant and cannot gate a resident router,
however correct the port (the reason the CUDA MoE fixture was rebuilt at 9275f94).
```

## KVCache.LayerKVForTest

Moved from `decoder/testhooks.go` (the comment above `KVCache.LayerKVForTest`) on 2026-10-09.

```text
LayerKVForTest is a thin wrapper over the production KVCache.LayerKV (kvcache.go — promoted
there so decoder/generate_vl_resident.go's resident-KV upload bridge can call it untagged).
Kept under this name for the existing cross-package (cuda/metal) test call sites: the
cross-backend attention confirmer injects goinfer's exact K/V into another engine.
```

## SetSSMForceF32

Moved from `decoder/testhooks.go` (the comment above `SetSSMForceF32`) on 2026-10-09.

```text
SetSSMForceF32 / SetSSMQ8CPU toggle the CPU-reference precision-localization seams
at runtime (gpu/ssm_kernel_control_test.go needs the staged webgpu backend, which
lives in a package the decoder can't import — so it drives these via the registry).
```

## EncodeChatForTest

Moved from `decoder/testhooks.go` (the comment above `EncodeChatForTest`) on 2026-10-09.

```text
LoadTokenizerForTest / EncodeChatForTest give a backend package the same chat-templated token
ids the CPU acceptance sweep used, so a GPU wall-clock number is comparable to it. Without
them a backend test would have to re-implement the template, and the non-thinking suffix in
particular has already cost this program one wrong measurement.
```

## DeltaNetStepForTest

Moved from `decoder/testhooks.go` (the comment above `DeltaNetStepForTest`) on 2026-10-09.

```text
DeltaNetStepForTest runs ONE Gated-DeltaNet decode step on the CPU reference and returns the
layer output — the parity target for a backend's DeltaNet kernel (gpu/deltanet_test.go).

Why a hook rather than a reimplementation in the backend's test: the CPU recurrence is already
gated against HF (TestGatedDeltaNet_parity, against transformers'
torch_recurrent_gated_delta_rule), so comparing a kernel to THIS makes the chain
kernel ≡ CPU ≡ HF. A reference re-written inside the gpu package would be a second unvalidated
implementation, which is the shape that let a hand-rolled forward drift for months
(docs/parity-coverage-policy.md).

The caller supplies the weights and the state; `st` is mutated in place across calls, so a
multi-token drift test just calls this in a loop.
A nil backend means the CPU one, which is unexported — the same accommodation
DraftBlockCPUForTest makes, and necessary because matmul dereferences it unconditionally.
```

## SetSSMStopLayerForTest

Moved from `decoder/testhooks.go` (the comment above `SetSSMStopLayerForTest`) on 2026-10-09.

```text
SetSSMStopLayerForTest sets the granite forward's layer truncation (forward_granite.go's
ssmStopLayer) and returns a func that restores the previous value. It replaced the env var
GOINFER_SSM_STOP_LAYER, which the decoder read ONCE at init — so the gpu layer sweeps' os.Setenv
inside their loops changed nothing on the decoder side, and their CPU references ran every layer.
```

## SetMemoryProbeForTest

Moved from `decoder/testhooks.go` (the comment above `SetMemoryProbeForTest`) on 2026-10-09.

```text
SetMemoryProbeForTest makes the named backend's registered memory probe (RegisterMemoryProbe: "cuda", "metal") report freeBytes and ok, and returns the restore
(hardware-coverage H1.5). It reaches the residency decisions that read FreeBytesFor: the resident context a load picks, how many KV slots fit, whether Plan places
or declines the model. A budget BELOW the real card runs end to end on it (the model shrinks its context or declines to the CPU); a budget above it only changes the
decision, and a resident build that then asked the real card for the memory would fail, so a test should not force a figure the model cannot really be given.
Set it BEFORE the model loads. A name with no registered probe is created, so a test can also force a backend that is not linked.
```

## SetGemma4PLEDropTokenForTest

Moved from `decoder/testhooks.go` (the comment above `SetGemma4PLEDropTokenForTest`) on 2026-10-09.

```text
SetGemma4PLEDropTokenForTest plants S1's G2 defect (2): the resident embedding row's PLE inputs lose their
token-identity term (docs/tasks/task-multimodal-support-2026-10.md). The CPU forward is unaffected.
```

## KVCache.SetCaptureLayersForTest

Moved from `decoder/testhooks.go` (the comment above `KVCache.SetCaptureLayersForTest`) on 2026-10-09.

```text
SetCaptureLayersForTest arms the batched hidden-state capture (the seam forwardn.go's layer loop fills after layer l's residual add and DeepStack add) on c for the
next CPU prefill: CapturedForTest then holds one [rows*hidden] float32 copy per requested layer, the residual AFTER that layer, as HF's hidden_states[l+1] is. The
896-pixel investigation's per-layer differencing uses it from the cuda package. nil disarms.
```
