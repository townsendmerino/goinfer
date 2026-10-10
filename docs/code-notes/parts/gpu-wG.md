# gpu: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `gpu`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

## ssmStopLayerForTest

Moved from `gpu/decoderunner.go` (the comment above `ssmStopLayerForTest`) on 2026-10-10.

```text
ssmStopLayerForTest truncates a newly built decode plan after this layer (-1 = every layer): the
resident-SSM bring-up's layer-sweep seam, set only by tests in this package
(mamba_layersweep_test.go). It was the env var GOINFER_SSM_STOP_LAYER until 2026-09-24.
```

## attnGeom

Moved from `gpu/decoderunner.go` (the comment above `attnGeom`) on 2026-10-10.

```text
attnGeom is one distinct per-layer attention shape: head_dim (hd), KV-head count
(nKV), rotary half-width (half = rotaryDim/2), and the attention_k_eq_v flag (kEqV).
Gemma 4 interleaves two shapes (local hd=256/nKV=8; global hd=512/nKV=2, K=V); every
other family has exactly one. The per-token uniforms that carry these dims (v-store,
attn, the windowed-attn variant, and — keyed additionally by rope scale — q-rope /
k-rope-store / the fused qkv-finalize) are deduplicated by value: geomFor caches one
attnGeom per distinct {hd, nKV, half, kEqV} tuple, so a uniform-geometry model
collapses to a single entry with the same buffers, bind groups, and dispatch it had
before this seam existed. Byte-identity for non-Gemma models is thus structural (a
shared *attnGeom), not asserted.

kEqV is in the key because the geom OWNS the v-store uniforms (vStoreUni/vStoreI8Uni):
two layers with equal {hd, nKV, half} but different attention_k_eq_v want different
v-store behaviour — a K=V layer derives V from K instead of storing a projected V — so
they must not share a geom. The K=V forward itself (V = v_norm(k), no v_proj, the
attention V binding aliasing the K cache) lands with Gemma-4 admission, where v_norm
exists and it is testable; keying on it now keeps that future branch sound.

nH (query-head count) is deliberately NOT in the key: it is a model-level constant, and
GQA still tracks per-layer nKV because the group ratio is recomputed as nH/nKV per geom
(see geomFor's attnUni). A family with per-layer QUERY-head counts would have to add nH
to the key too; none on this seam has that (Gemma 4 is 16 query heads in both variants).
```

## runModel.gatedGELU

Moved from `gpu/decoderunner.go` (the comment above `runModel.gatedGELU`) on 2026-10-10.

```text
G6 (docs/tasks/task-gpu-paths-2026-09.md): FeatGatedGELU, model-level like ropeHalf/
slidingWindow above (every gated-MLP family shares one activation). Deliberately a bool
with SiLU as the zero value, NOT decoder.ActKind's own ordinal (where 0=GELU-tanh,
1=SiLU) — a runModel built by hand (every gpu/*_test.go that constructs one directly,
rather than through BuildResident) leaves this at its Go zero value, and that zero value
must be the behavior every one of those tests already assumed (SiLU), not silently
switch them to an activation this row just added. Measured: mirroring decoder's own
ordinal directly broke TestDecodeRunnerW4A8_parity et al. (cosine 0.9995, not ~1.0) the
first time this was wired, because their hand-built runModel left the field at 0 = GELU
under that convention. residency.go sets this to `m.GatedActResident() == 0` — true only
for the family that actually needs it.
```

## attnHeadDimSupported

Moved from `gpu/decoderunner.go` (the comment above `attnHeadDimSupported`) on 2026-10-10.

```text
newDecodeRunner builds the persistent decode plan for either precision.
attnHeadDimSupported declines a resident decode plan whose model-level or any per-layer
head_dim exceeds what the single-query attention kernels can dot. Those kernels run at
@workgroup_size(128) with a fixed 128-entry `red` reduction array (attention.go, one lane per
dim), so a head_dim above attnMaxHeadDim would leave the tail dims un-dotted and the
o-projection would consume half-zero context — plausible-looking WRONG output, no error. The
caller falls back to the staged/CPU path on this error. MLAAttn guards its own analogous rank
limit; this covers the softmax/GQA runners including Gemma 4's per-layer head_dim (audit M-12).
```

## newDecodeRunner.guards

Moved from `gpu/decoderunner.go` (the comment above `newDecodeRunner.guards`) on 2026-10-10.

```text
The single-query attention kernels (attention.go) parallelize over head_dim at
@workgroup_size(128) with a fixed 128-wide workgroup reduction array (`red: array<f32,128>`).
A head_dim > 128 would dot only dims 0..127 and leave ctxv[128..hd) zeroed — the o-projection
then consumes half-zero context: plausible-looking WRONG output, no error. Decline here (like a
VRAM-exhaustion decline) so the caller falls back to the staged/CPU path; MLAAttn guards its own
analogous rank limit (audit M-12). Covers the model-level shape and any per-layer geometry
override (Gemma 4's per-layer head_dim), so an admitted arch can never silently truncate.
MLA (DeepSeek/Kimi) attention runs the mlaAttn kernel family (mla.go) with its own rank-bounded
accumulator, NOT the 128-wide GQA kernels attnHeadDimSupported protects — its qk head dim
(qk_nope+qk_rope) is 192 on real V2-Lite/V3/Kimi and legitimately exceeds 128. Applying the GQA
guard here regressed EVERY MLA checkpoint off residency (audit R-05, collateral of M-12). Exempt
MLA, but enforce the analogous per-lane rank cap MLAAttn itself checks (rank ≤ 1024; audit R-24).
```

## DecodeRunner.mambaOps

Moved from `gpu/decoderunner.go` (the comment above `DecodeRunner.mambaOps`) on 2026-10-10.

```text
Mamba-2 SSM mixer dispatches (P5b): the conv/ssm/gatedNorm kernels read slices of
the in_proj output (z|xBC|dt) via bind-group offsets (256-aligned for granite), so
no extra split kernel. The dispatches are pos-independent — {win, ssm} state carries
the recurrence. mamba* are nil for non-hybrid models (the closures go unused).
```

## ropeKUniFor

Moved from `gpu/decoderunner.go` (the comment above `ropeKUniFor`) on 2026-10-10.

```text
ropeKUniFor feeds ropeStoreShaderWGSL, whose P struct has TWO distinct roles: slot 3
("pos") is the rotation angle, slot 5 ("base") is the KV-cache write offset pos*kvDim —
confirmed directly against the shader source (attention.go). Only slot 3 switches to
ropePos; slot 5 stays keyed on the true sequential pos. Slot 7 ("spos") carries the true pos
too: the int8 variant indexes its per-(position, KV-head) scale by it, and the f32/f16 variants
never read it. The int8 variant used slot 3, the rope angle, until audit-2026-09-10 C-09.
```

## DecodeRunner.ReadMambaCap

Moved from `gpu/decoderunner.go` (the comment above `DecodeRunner.ReadMambaCap`) on 2026-10-10.

```text
ReadMambaCap copies the first mamba layer's captured proj/conv/y/gated buffers (their
values from the most recent Run) back to the host — the resident's actual per-token kernel
I/O, for diffing against mamba2Step (gpu/mamba_resident_capture_test.go). projN/convN/dInner
are the element counts. Test-only; allocates fresh staging per call.

N-15: this is EXPORTED and used to panic on a failed buffer map. "Test-only" is a comment, not
a compiler constraint — an exported method on an exported type is callable by anyone, and a
panic on a device-boundary failure takes the caller's process down. Returns an error now.
```

## DecodeRunner.Run.errchecks

Moved from `gpu/decoderunner.go` (the comment above `DecodeRunner.Run.errchecks`) on 2026-10-10.

```text
N-84 (docs/audit-2026-09-10.md): these two used to discard their error returns, unlike
every other fallible call in this function (and unlike gpu.go's run(), the reference this
mirrors) — a validation error here surfaced only later as an opaque "DecodeRunner map
failed" with no indication which call actually caused it.
```

## uploadProj.slowpath

Moved from `gpu/residency.go` (the comment above `uploadProj.slowpath`) on 2026-10-10.

```text
The int4 resident upload takes the direct byte path whenever K%32==0 (decoder int4 bytes == GPU
layout), else unpack+packNibbles. GOINFER_INT4_SLOWPATH, which forced the latter to isolate the
fast path's delta, was retired 2026-09-24 (docs/tasks/task-env-config-2026-09.md, phase 6); the
measurement is in docs/completed/mellum2-resident.md.
```

## uploadProj.fastpath

Moved from `gpu/residency.go` (the comment above `uploadProj.fastpath`) on 2026-10-10.

```text
Fast path: when K%32==0 the decoder's 2-nibble/byte int4 is byte-identical to the
GPU packed layout (TestInt4LayoutMatch), so upload the bytes straight — no
per-element unpack + packNibbles re-pack (the ~30s/12B-param resident-load tax).
```

## webgpuBackend.BuildResident.perlayergeom

Moved from `gpu/residency.go` (the comment above `webgpuBackend.BuildResident.perlayergeom`) on 2026-10-10.

```text
Per-layer attention geometry (decoder.Model.PerLayerGeomOK): dense Gemma 4's local/global
head_dim split (256 vs gemma4.GlobalHeadDim 512) needs a per-layer geometry seam this
backend's runLayer.ghd/gnKV/ghalf fields exist for but gpu/residency.go's per-layer builder
never actually populates for any family. Not expressible as a ResidentFeature — Gemma 3 and
dense Gemma 4 derive the IDENTICAL required-feature set otherwise, so MissingResidentFeatures
alone cannot catch this; see PerLayerGeomOK's own comment for the admission this predicate
exists to prevent (found 2026-09-08 when G6's Gemma-set work satisfied every OTHER
requirement dense Gemma 4 has).
```

## webgpuBackend.BuildResident.routercap

Moved from `gpu/residency.go` (the comment above `webgpuBackend.BuildResident.routercap`) on 2026-10-10.

```text
Router-kernel capacity (gpu/moe.go): score/sel are array<f32,256> with nE clamped to
min(nE,256), and the group-limited path uses array<f32,32>/array<bool,32> indexed by
nGroup. A model with more experts would route on only the first 256 (Kimi K2 has 384 —
experts 256+ silently never considered); more than 32 groups indexes out of bounds. Either
is plausible-looking wrong output, so decline to the staged path — matching cuda/backend.go's
nE>256 build check (M22). Checked here, before any allocation, so the decline is clean.
M-31: the cap is READ from decoder's declaration, not restated here. This site had its
own 256/32 while the map, gpu/moe.go's MAXE and both published matrices said 512 — so a
384-expert Kimi-K2 or DeepSeek-V4-Pro was admitted by ResidentEligible, advertised as
resident, and then declined to CPU by this line with a message naming a number nothing
else agreed with.
```

## webgpuBackend.BuildResident.kvflag

Moved from `gpu/residency.go` (the comment above `webgpuBackend.BuildResident.kvflag`) on 2026-10-10.

```text
M-32: ONLY the generic GQA branch honours the KV precision flag. The Nemotron, Qwen3.5
and MLA branches always allocate f32 NewKVCache, while ctxCap below is raised by the flag
and the kernel selection later is model-wide. The two failure shapes:

  --kv i8 : rl.kScale stays nil, bind reports "nil buffer for binding 3 (allocation
            failed)", which surfaces as "device allocation failed (VRAM exhausted?)" —
            so the whole model silently runs on CPU for a reason that is not true.
  --kv f16: each cache is ctxCap×kvDim×4 bytes at the RAISED cap, i.e. 2x the intended
            f16 footprint and 2x the f32 default. The "f16 halves KV bytes, so 32k fits"
            premise inverts.

Declining is the honest fix rather than a guess: quantized KV for MLA's rank-space latent
(or Nemotron's mixed mamba/attention layers) is real kernel work, and inventing it here
unvalidated would trade a loud wrong reason for a quiet wrong number. The decline names
the flag, so the operator can drop it and get the resident path back.
```

## webgpuBackend.BuildResident.nogqa

Moved from `gpu/residency.go` (the comment above `webgpuBackend.BuildResident.nogqa`) on 2026-10-10.

```text
TRAP, found live 2026-09-18, CLOSED 2026-09-18 (same day — see
docs/completed/task-webgpu-nogqa-decode-bug.md for the full trail). A decline used to sit here:
letting phi3-mini (head_count_kv == head_count == 32, no GQA grouping) reach residency for
the first time ever (the ctxCap clamp two lines below stopped masking it via a VRAM
failure) showed real divergence from CPU — cosine 0.94-0.99 on the prompt, well under this
repo's usual 0.999 bar, and negative cosine within a few greedy steps. That read as a
kernel bug and the decline was the honest response to it.

It was not one. Three independent lines of evidence, the last two against the REAL
checkpoint on REAL hardware (RTX 2070 SUPER, this box — a third adapter beyond the Mac's
Metal and a no-GPU sandbox's software Vulkan):
 1. `gpu/geom_mha_decoderunner_test.go` runs the exact production dispatch plan (int4 GEMV,
    the attention kernel at group=1/hd=96, o-proj, MLP) against a bit-exact oracle at
    phi3-mini's precise shape: cosine 1.000000 at every geometry tried. The kernels are
    correct for this shape.
 2. `gpu/resident_capture_parity_test.go` differences resident-vs-CPU PER SUBLAYER on the
    real checkpoint (GOINFER_GPU_CAPTURE=1): early layers agree to ~1e-7 (f32 rounding),
    then at some layer — the position varies — one sublayer's contribution jumps to
    O(1e-2) and every later layer inherits and compounds it. That is a int8
    activation-quantizer flipping a rounding decision that sat within a ~1e-7 perturbation,
    not a wrong computation: each flip is a whole quantization step, and the next GEMV
    turns a few flips into a denser perturbation the next quantizer also sees.
 3. `gpu/cpu_quant_sensitivity_test.go` is the control: the CPU forward against ITSELF, one
    side's f32 norms nudged by ±1 ULP (`decoder/normnoise.go` — exactly the magnitude two
    CORRECT implementations differ by from reduction order alone), on the SAME real
    checkpoint. Self-sensitivity floor: min cosine 0.953287, with its own argmax flips at
    specific prompt positions. The resident-vs-CPU comparison (min cosine 0.985860) is
    BETTER than what pure floating-point noise alone produces in a self-comparison — the
    opposite of what an additional kernel bug on top of that noise would look like. Real
    greedy text from the resident path stays coherent and factually correct throughout,
    with isolated single-token anomalies at what the capture data shows are near-tied
    positions, not wholesale breakage.

Kept: the ctxCap-to-modelCtx clamp just below (independently correct and valuable). The
decline itself, its MLA/DeltaNet/Nemotron scoping, and its GOINFER_WEBGPU_ALLOW_NOGQA
escape hatch are gone — the divergence they existed to prevent shipping was the
COMPARISON's sensitivity, not the kernels', so declining every current and future no-GQA
family on WebGPU bought nothing worth what it cost. The gpu module's own 0.999 fixed-floor
convention (gemma3_resident_parity_test.go and friends) does not transfer to a real,
32-layer, int4-quantized checkpoint; docs/completed/task-webgpu-nogqa-decode-bug.md's own
closing section says what a resident parity gate should measure there instead.
```

## webgpuBackend.BuildResident.ctxclamp

Moved from `gpu/residency.go` (the comment above `webgpuBackend.BuildResident.ctxclamp`) on 2026-10-10.

```text
Clamp to the model's own window BEFORE anything else. WebGPUCtxCeiling's "proven 8 GB fit"
was proven against the GQA geometry every previously-tested model shares (nKV well below
nH); it says nothing about a model whose nKV == nH (no GQA at all). Found live 2026-09-18
benchmarking phi3-mini (head_count_kv == head_count == 32, vs. e.g. qwen2.5-7B's 4): its
real per-position KV cost is ~6x a typical GQA model's, so reserving the ceiling's 16384
positions — four times phi3-mini's own 4096-token max_position_embeddings, and positions
the model could never actually serve — ran the per-layer allocation loop below out of VRAM
partway through (layer 13 of 32), and BuildResident declined to the CPU-staged path for a
reason that had nothing to do with whether phi3-mini actually fits this card: at its own
4096-position ceiling the real KV need is ~3.1 GB, which fits an 8 GB card easily beside
phi3-mini's ~2.3 GB of int4 weights. cuda/resident.go's resolveCtxCap already clamps to
modelCtx for exactly this reason; WebGPU never did. No live free-VRAM query exists for this
backend (unlike CUDA's driver MemInfo()) to catch this the way cuda/resident.go's
checkKVFits does, so the model's own window is the one geometry-aware signal available
here, and skipping it lets any zero-GQA model try to over-allocate the same way.
```

## webgpuBackend.BuildResident.ctxreq

Moved from `gpu/residency.go` (the comment above `webgpuBackend.BuildResident.ctxreq`) on 2026-10-10.

```text
M-32, the -ctx half: Options.ResidentContext / `serve -ctx` was read nowhere under gpu/,
so -ctx 32768 silently kept 16k (requests past it fail at checkCap) and -ctx 2048 still
allocated 16k per layer. min(), not the request: the caps above are proven-fit ceilings,
and honouring a LARGER request would trade a clear checkCap refusal for an OOM.
```

## webgpuBackend.BuildResident.stacked

Moved from `gpu/residency.go` (the comment above `webgpuBackend.BuildResident.stacked`) on 2026-10-10.

```text
int4 stacked experts (W4A8): fits VRAM where int8 spills (e.g. Mellum2 12B on
8 GB). K must be a multiple of the 32-wide W4A8 group.

Fast path (K%32==0): the decoder's 2-nibble/byte int4 is byte-identical to the
GPU packed layout, so pass the decoder bytes straight to the Packed upload —
one memcpy/expert to concatenate, no per-element unpack+repack (the ~30s tax).
```

## webgpuBackend.BuildResident.attnSinks

Moved from `gpu/residency.go` (the comment above `webgpuBackend.BuildResident.attnSinks`) on 2026-10-10.

```text
FeatAttnSink (gpt-oss): attnSinks is ALWAYS bound (a real one-element dummy when the
layer has none — WGSL bind groups can't bind a null storage buffer), matching Metal's
exact convention (metal/model.go's L.attnSinks/L.uHasSink) rather than CUDA's
null-pointer sentinel.

N-87 (docs/audit-2026-09-10.md, confirmed 2026-09-16, no action needed — genuinely
cosmetic): this uploads its own tiny one-float dummy per non-gpt-oss layer, and
decoderunner.go's noAttnSinks ALSO allocates one — but noAttnSinks's own doc comment
says why they can't just share one: it exists purely as a RUNNER-local fallback for
hand-built test fixtures that skip BuildResident and leave rl.attnSinks nil, not as a
reuse mechanism for the production path (a real BuildResident load, like this one,
always populates rl.attnSinks for real, so noAttnSinks's fallback never fires for it).
The two buffers have different owners and lifetimes (resident, shared across runners,
vs per-runner) for a real reason; the redundancy is a handful of 4-byte buffers against
weight/KV-cache VRAM, not worth the complexity of unifying them.
```

## webgpuBackend.BuildResident.biasgate

Moved from `gpu/residency.go` (the comment above `webgpuBackend.BuildResident.biasgate`) on 2026-10-10.

```text
q/k/v bias (Qwen2): the fused-epilogue tiled GEMM (gemm.go's
matmulTiledW8A8BiasKernelWGSL, added to fix a real nKeys-dependent
divergence — see runModelToModelW's doc comment) is measured BIT-EXACT
on Vulkan (RTX 2070 SUPER, nKeys 1-50) but STILL diverges on Metal
(M1 Pro) past nKeys~15, cosine ~0.998 — a real, smaller, still-open gap,
not float noise. Gate bias to the backend where it's actually proven
rather than either fully shipping an unverified-on-Metal result or
throwing away a confirmed fix everywhere. Revisit once the Metal gap is
found, or once it's checked against this repo's real ship gate
(docs/completed/task-prefill-gap.md §3.2 pooled fidelity vs the CPU-f32
reference, not cosine-vs-sequential-GPU-decode).
```

## residentDecoder.ForwardN.recurrent

Moved from `gpu/residency.go` (the comment above `residentDecoder.ForwardN.recurrent`) on 2026-10-10.

```text
Batched verify is ONLY sound when every position this runner writes can be rolled back.
The resident KV can (TruncateTo is a no-op precisely because the cache is positional and
Forward re-sets nKeys=pos+1), but the Mamba {win,ssm} state is NOT positional: mamba2Step
mutates it in place, so running K rows advances it K times and a partial accept has no way
to undo the rejected rows' advance. The next round then decodes from over-advanced state —
silently wrong output, not a crash (audit C-17, compounding C-02).

Today decoder.specRollbackSafe already refuses the recurrent families, so this is
unreachable in production. That is exactly why the guard belongs HERE: the invariant is a
property of this runner's own state, and it was previously enforced only by a check in
another package that a future "re-enable recurrent speculation" change would relax without
ever reading this function. Decline (the callers fall back to plain decode); do not
silently advance. n==1 is the ordinary single-step case and stays allowed.
```

## residentDecoder.VerifyPath

Moved from `gpu/residency.go` (the comment above `residentDecoder.VerifyPath`) on 2026-10-10.

```text
VerifyPath (decoder.VerifyPathReporter) reports whether this resident's ForwardN executes a
batched pass whose marginal per-node cost approaches zero, or a sequential one whose marginal
cost approaches a full extra step. P22 (docs/queue-performance.md): before this, WebGPU
implemented neither VerifyPathReporter nor PrefillPathReporter, so decoder.verifyTheta() fell
through to thetaFor("webgpu")'s unmeasured 0.5 default — the same "worst available choice"
defect Metal had (a smaller Theta drafts DEEPER, not shallower; spec_adaptive.go's Depth() is
monotone-decreasing in Theta), just never caught here because nothing had measured it.

MEASURED, not inferred from ForwardN's "one command buffer" doc comment above: despite runBatch
recording every row into a single Submit/Poll, Theta = 0.978-1.028 across four real
configurations (docs/measurements/theta-webgpu-2026-09-23.md; gpu/theta_probe_test.go), i.e.
T(n) is n·T(1) to within noise — indistinguishable from a plain per-token loop. The single
submit removes Go-side dispatch/sync overhead between rows, not GPU-side compute: each runner
still issues its own full set of per-layer dispatches, so the marginal wall-clock cost of an
extra row stays ~1×, which is exactly the quantity this interface's contract is about — a
structurally-batched submission is not what "batched" means here if it doesn't move Theta.
Always sequential (unconditionally, unlike Metal's paged-MoE-only decline): no configuration
measured otherwise, and there is no code path here that behaves differently by model shape.
```

## webgpuBackend.BuildResident.mambaInt8

Moved from `gpu/residency.go` (the int8-projection comment in the Granite Mamba-2 layer build) on 2026-10-10.

```text
int8 projections. An f16 path was measured and did NOT recover quality (63.9% vs
int8 66.2%, ~2× slower): the loss is the SSM f32-vs-f64 exp feeding the 64-expert
router, not projection precision (docs/ssm-int8-quality.md; code removed 2026-09-24).
```

## webgpuBackend.BuildResident.kvgeom

Moved from `gpu/residency.go` (the per-layer KV geometry comment in the generic attention layer build) on 2026-10-10.

```text
Per-layer KV geometry (P1 seam): each layer's cache is sized by its own
nKV·hd, not a single model-level kvDim. Every non-Gemma family leaves the
runLayer tuple unset, so lnKV/lhd/lkvDim fall back to the model values and the
allocation is byte-identical. Gemma 4's builder will set rl.gnKV/rl.ghd (local
8·256=2048, global 2·512=1024) and this picks up the differing widths per layer.
```
