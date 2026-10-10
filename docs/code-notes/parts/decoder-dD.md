# decoder: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `decoder`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

## loraAdapter.validateTargets

Moved from `decoder/lora.go` (the comment above `loraAdapter.validateTargets`) on 2026-10-10.

```text
validateTargets checks every adapter delta maps to a per-layer projection the
loader will merge into — so an adapter targeting an unsupported module (e.g.
embed_tokens) fails loudly rather than silently no-op'ing.

name MUST build the SAME tensor names the corresponding merge path actually looks the delta up
by, or the check lies (M18): the merge-at-load path in weights.go looks up prefixed names (tn,
which adds language_model.* / model.language_model.* on VL checkpoints), while the compute-time
path (buildLoraRuntime) uses the bare tensorName. Passing the wrong builder is exactly how a
VL-prefixed base validated clean and then silently ignored the whole adapter.
```

## loraAdapter.validateComputeTimeDims

Moved from `decoder/lora.go` (the comment above `loraAdapter.validateComputeTimeDims`) on 2026-10-10.

```text
validateComputeTimeDims checks every targeted delta's declared [Out,In] shape against the
ACTUAL base projection it will be added to — the same check merge() already makes for every
tensor on the merge-at-load path (its own `d.out != out || d.in != in` above), which the
compute-time path never made: buildLoraRuntime maps deltas onto projections by NAME only, with
no shape check at all. A same-family adapter trained against a different-size base (a narrower
or wider hidden/head/intermediate dim) then reaches every resident backend's SetAdapter
carrying a rank-only-checked delta whose In/Out the kernel trusts blindly — on Metal this writes
past the Q slot into K/V (audit-metal-2026-09-12.md C-03). Fixed once here, at the one
chokepoint every backend's compute-time LoRA passes through, rather than duplicated per backend.
```

## loraAdapter.checkAllMerged

Moved from `decoder/lora.go` (the comment above `loraAdapter.checkAllMerged`) on 2026-10-10.

```text
checkAllMerged refuses an adapter that has a delta the load never merged (D3). validateTargets checks
names against a list before the load; this checks them against what the load actually did, so it
also covers a family whose tensor names no list holds (qwen35's DeltaNet layouts), a loader that
takes no adapter at all (internlm2, gpt-oss), and a name that is on the list but is not loaded
in that layer (a self_attn projection on a linear-attention layer). nil-safe.
```

## compute-time LoRA section

Moved from `decoder/lora.go` (the comment above `compute-time LoRA section`) on 2026-10-10.

```text
--- Compute-time LoRA (#7): apply the low-rank delta in the forward instead of
merging it into the base weight. Keeping the base immutable + zero-copy lets N
adapters of one base share a single resident transformer (≈ base + N small
deltas) rather than paying the full RAM N times over. Opt-in: merged-LoRA stays
the faster, simpler default — this trades a little decode speed for that density.
Only the generic dense forward (runLayersFromEmbed → causalAttention + gatedMLP)
is wired; Model.LoadAdapter rejects the special-forward / MoE / non-gated archs.
```

## Model.LoadAdapter

Moved from `decoder/lora.go` (the comment above `Model.LoadAdapter`) on 2026-10-10.

```text
LoadAdapter loads a PEFT LoRA adapter for compute-time application (#7) and
registers it under name. Unlike the merge-at-load path (Options.Lora), the base
stays immutable, so many adapters of one base share its resident weights — the
density win. Only the generic dense forward is wired: MoE, non-gated, and every
own-forward family (arch.ownForward — gemma4, qwen3_5_moe, lfm2, granitemoehybrid,
nemotron_h, deepseek_v2/v3, llama4_text, gpt-oss) are rejected, as is a
GGUF/serialized base (the adapter is HF-named — it needs the safetensors schema).

V-12 (docs/review-2026-09-04.md): this used to hand-list gemma4/qwen35 instead of
deriving from arch.ownForward() — the same "one predicate, seven consumers" bug class
canBatchN (decoder/forwardn.go) was fixed for after LFM2 fell out of ITS hand-copied
list (audit-2026-09-02 C-01/C-02). LoRA against LFM2 loaded and validated cleanly —
runLayersLFM2 takes no lora parameter at all — so the adapter silently did nothing.
```

## Model.LoadAdapter.dims

Moved from `decoder/lora.go` (the comment above `Model.LoadAdapter.dims`) on 2026-10-10.

```text
C-03 (audit-metal-2026-09-12.md): validateTargets only checks that every target NAME is a
known projection — it says nothing about SHAPE. Every resident backend's SetAdapter trusts
In/Out from the checkpoint (Metal only range-checks rank); catch a mismatched adapter here,
once, before it reaches any of them.
```

## Model.registerAdapter

Moved from `decoder/lora.go` (the comment above `Model.registerAdapter`) on 2026-10-10.

```text
registerAdapter installs rt under name, retiring any runtime it displaces (audit C-29). A live
Session may still hold the displaced runtime via cache.lora and read its mmap'd deltas
mid-generation, so it is retired (released at Model.Close), never munmap'd here. The registry's
mutex guards this against concurrent UseAdapter/HasAdapter reads from other request goroutines.
The registry is allocated on first use (adapters are loaded at setup, before serving), so the
pointer field is only ever set here.
```

## mmapAliasedBytes

Moved from `decoder/weightbytes.go` (the comment above `mmapAliasedBytes`) on 2026-10-10.

```text
mmapAliasedBytes is wmBytes' quantized-kind sum, but ONLY counting bytes that actually alias
m's .giw mmap region (m.MmapByteOffset) — a genuinely zero-copy load, not a heap allocation
this model's own quantizer produced. Mirrors wmBytes' own int8/int4/int4Row4 cases (f32 and
SplitHalfBytes are never mmap-aliased: f32 is copied at .giw read time — LoadSerializedWeights'
own doc comment says "float arrays are copied" — and SplitHalfBytes is a derived repack, not a
stored tensor). M-24 (docs/audit-2026-09-10.md): unlike aikit's WeightMat.MappedSpan (used for
the expert PAGING registry, metal/moepaging.go), this does NOT page-align — a byte accounting
question ("did this allocation happen at all") is not the same question as a pageable-span
question ("can MADV_DONTNEED release whole pages of it"), and page-rounding would undercount a
small-but-real mmap-backed tensor for no reason that matters here.
```

## Model.ResidentWeightBytes

Moved from `decoder/weightbytes.go` (the comment above `Model.ResidentWeightBytes`) on 2026-10-10.

```text
ResidentWeightBytes is the total byte footprint of this model's weight matrices — what a
resident backend must hold to run the whole model on-device, assuming every routed MoE expert
is resident. See ResidentWeightBytesPaged for the synchronous-paging case
(GOINFER_METAL_MOE_SLOTS), where only N experts per layer are.

WHY IT EXISTS. A resident backend had no way to ask "will this model fit?" before allocating,
and nothing else in the tree answers it: Dims() exposes hidden/layers/heads but NOT the expert
count, so a shape-derived estimate under-reports a sparse MoE by the factor that matters most —
gpt-oss-20b's experts ARE the model. Measured 2026-08-31: loading an 11.28 GB gpt-oss-20b on
Metal's resident path on a 16 GB machine drove swap to 35.98 GB of 36 GB and never completed OR
declined, because the only size guard in the tree caps the KV CONTEXT (metal/backend.go), not
the weights.

It sums the MATRICES, which is where the bytes are; the elementwise norms/biases are [hidden]-
sized and round to nothing beside them. That makes this a LOWER BOUND on the real footprint,
which is the safe direction for a guard: it can fail to refuse a marginal model, but it cannot
refuse one that would have fit.

This is a quantity we COMPUTE, deliberately — not the OS's account of free memory. Darwin's UBC
reclaims under pressure, so "available" reports what survived rather than what can be asked
for; an RSS-keyed ceiling once reported LESS memory at a known failure point than at baseline.
```

## Model.ResidentWeightBytesPaged

Moved from `decoder/weightbytes.go` (the comment above `Model.ResidentWeightBytesPaged`) on 2026-10-10.

```text
ResidentWeightBytesPaged is ResidentWeightBytes under Metal's synchronous MoE paging
(metal/moe.go, metal/gemma4_moe.go): GOINFER_METAL_MOE_SLOTS=N keeps only N of each layer's
ROUTED experts resident, staging the rest per token. slots<=0 means unpaged (identical to
ResidentWeightBytes). SharedExpert is never paged — it is always active, not top-k routed — so
it is counted in full either way, same as every dense matrix.

M-02: this is what the memory-fit guard was missing. It always summed EVERY expert — the
unpaged number — even when the caller had asked to page, so a model that would fit paged (e.g.
Qwen3.5-35B-A3B's 22.1 GB unpaged vs. a few GB at N=64) was declined to CPU on a bound it never
actually needed. A layer's experts are uniform in shape, so "per-expert bytes" is the full
per-layer expert sum divided by the expert count — exact, not an approximation across layers.
```

## Model.ResidentDenseWeightBytesFor

Moved from `decoder/weightbytes.go` (the comment above `Model.ResidentDenseWeightBytesFor`) on 2026-10-10.

```text
ResidentDenseWeightBytesFor is ResidentDenseWeightBytes less the model-level tables backend keeps on the HOST rather
than on its device — the dense figure its fit checks must price against device memory.

CUDA (cuda/backend.go) uploads every per-layer matrix and the LM head, which is the embedding table when the
embeddings are tied. It gathers token embeddings on the host (embedResident), and it never uploads the learned
position table or Gemma 4's per-layer embedding tables. So for an UNTIED model the whole token-embedding table was
priced as device memory it never takes. Measured 2026-09-27 on qwen2.5-7b-instruct q4_k_m (RTX 2070 SUPER): Plan
priced the dense weights at 4930 MB, and the build allocates ~4476 MB before its KV, scratch and modules included.
The ~520 MB embedding table is the difference. That overcount made the MC1 slot-aware context
(cuda ctxForSlots) land on its 4096 floor where ~4,870 positions fit
(docs/measurements/concurrency-mc1-cuda-2026-09-27.md).

Other backends: unchanged, ResidentDenseWeightBytes. Metal's accounting of these tables (unified memory, its own
host-copy term) was not re-measured here.
```

## Model.ResidentHostCopyBytes

Moved from `decoder/weightbytes.go` (the comment above `Model.ResidentHostCopyBytes`) on 2026-10-10.

```text
ResidentHostCopyBytes is the portion of ResidentWeightBytesPaged's footprint that a UNIFIED-
MEMORY backend (Metal — "device" memory IS host RAM) keeps resident in TWO PLACES at once: the
quantized host WeightMat the loader materializes, and a second, freshly re-packed device buffer
built from it (metal/model.go's int4Buf), with nothing released in between. See M-02
(docs/audit-2026-09-02.md): a resident GGUF/safetensors model on Metal was measured landing at
~2x the guard's own estimate for exactly this reason.

Dense weights (every non-expert matrix, including the mixer/MLA/Mamba/shortConv projections and
an UNPAGED model's experts) always double this way for a GGUF/safetensors load — the loader
materializes a fresh heap-allocated quantized copy, and int4Buf/int8Buf build a SEPARATE device
buffer from it, nothing released in between. Genuinely PAGED routed experts (0 < slots <
nExperts) do NOT: they stream via pread straight from the .giw file into their device slot
buffer, or via an mmap the OS can reclaim under pressure (metal/moe.go, metal/gemma4_moe.go) —
an UNSTAGED expert leaves no committed host allocation behind to double.

M-24 (docs/audit-2026-09-10.md): a THIRD case the doc comment above used to miss entirely — a
.giw-loaded model's int8/int4 payloads are themselves mmap-ALIASED (LoadSerializedWeights'
own doc comment: "Big int8/int4 arrays are aliased into data (zero-copy)"), not heap-allocated,
for every tensor the format supports zero-copy for, not just paged experts. That mmap-backed
data is reclaimable page cache, not a second committed host allocation — so it must NOT count
as a "host copy" alongside the (separately, genuinely allocated) device buffer int4Buf builds.
Before this fix, a .giw int4 dense model's real ≈8.75 GB anonymous footprint priced at ≈17.3 GB
(measured shape from the finding): once, correctly, as the device buffer's estimated size
(ResidentWeightBytesPaged, unaffected by this fix), and once again, incorrectly, as if the
mmap-aliased source were ALSO a genuine second host-resident copy.
```

## Model.residentWeightBytesSplit

Moved from `decoder/weightbytes.go` (the comment above `Model.residentWeightBytesSplit`) on 2026-10-10.

```text
residentWeightBytesSplit does the one enumeration pass ResidentWeightBytesPaged and
ResidentHostCopyBytes both need, returning the DENSE (non-expert) sum plus a closure that caps
the routed-expert sum at `slots` experts (see pagedExperts' own doc) — so the two accessors
cannot enumerate the model differently and disagree about what "dense" means. mmapDense and
mmapExpertBytesAt are the M-24 (docs/audit-2026-09-10.md) parallel sums: the portion of dense/
expert bytes that alias this model's .giw mmap rather than a heap allocation — computed in the
SAME pass so the two accounting questions ("how many bytes" and "how many of those are
mmap-backed") can never disagree about which matrices exist either.
```

## Model.residentWeightBytesSplit: qwen3_5_moe mixer

Moved from `decoder/weightbytes.go` (the comment above `Model.residentWeightBytesSplit: qwen3_5_moe mixer`) on 2026-10-10.

```text
M-01: qwen3_5_moe's per-layer mixer (DeltaNet or gated-softmax attention) — the three
dominant projections quantize (WeightMat, 2026-08-19); the rest stay f32 vectors small
enough to fall under the doc's norms/biases exemption. At most one of these is non-nil.
```

## Model.residentWeightBytesSplit: gemma4moe

Moved from `decoder/weightbytes.go` (the comment above `Model.residentWeightBytesSplit: gemma4moe`) on 2026-10-10.

```text
M-01: Gemma 4's MoE sub-block. mlpGate/mlpUp/mlpDown ALIAS l.GateProj/UpProj/DownProj
(serialize.go's gemma4Layer comment) — already counted above; only routerProj and the
fused experts are new tensors here. The fused experts page the same way as l.Experts.
```

## weightmat.go: header

Moved from `decoder/weightmat.go` (the comment above `weightmat.go: header`) on 2026-10-10.

```text
Weight matrices are linalg.WeightMat (aikit): one type that hides f32 / per-row
int8 / group-wise int4 storage behind uniform accessors + the linalg kernels.
goinfer keeps the model POLICY here — which table gets which precision
(quantMode), the int4 group size, and the matmul backend routing (the staged
GPU hooks below) — while the storage wrapper, quantize primitives, and Row
dequant live in linalg. (Consolidates the wrapper formerly open-coded as
decoder.weightMat; see aikit linalg.WeightMat.)
```

## quantInt4Mix

Moved from `decoder/weightmat.go` (the comment above `quantInt4Mix`) on 2026-10-10.

```text
quantInt4Mix is a per-tensor mixed mode (idea #5): attention (+ embed/head/
router) at int8 where a calibration spike found the int4→int8 quality loss
concentrated, the FFN bulk (gate/up/down/experts) at int4. It is a LOAD-TIME
policy only — matmulQuant resolves it to int8/int4 per tensor, so the resident
weights and the .giw never carry quantInt4Mix itself. GGUF load path only.
```

## matmulQuant

Moved from `decoder/weightmat.go` (the comment above `matmulQuant`) on 2026-10-10.

```text
M-27 (docs/audit-2026-09-10.md): the router (ffn_gate_inp) must stay out of the "ffn_"
bulk even though its own name contains that prefix — top-k selection is discrete, so
quantizing it flips which experts win rather than adding rounding noise (the same
reasoning that keeps it out of quantInt4 entirely at every other quant mode). Every
current call site already routes the router through streamMat(..., quantNone, ...)
directly and never reaches matmulQuant at all; this is a guardrail against a future
family adding a router through the generic mat() helper and repeating that mistake —
it would then degrade to int8 (this function's own "attention stays int8" branch)
rather than int4.
```

## quantMode.embedding

Moved from `decoder/weightmat.go` (the comment above `quantMode.embedding`) on 2026-10-10.

```text
embedding returns the precision to use for the token-embedding table (and the
LM head, tied or not). Full W8A8 (int8 weights AND int8 activations), not
weight-only Q8 — changed 2026-08-24 (docs/completed/lmhead-workspace-fix.md Step
2, after the W4A8 plumbing phase found the LM head running weight-only Q8 was
the single largest per-token cost in int4 decode, achieving only 11-13 GB/s
against W8A8's 97.12 GB/s at the same shape — 7.7x). Precision measured before
switching, teacher-forced real-continuation comparison against the old
weight-only-Q8 pin, both real model sizes: 1.5% argmax flip rate, mean cosine
0.9998+ (200 positions each, docs/completed/task-w4a8-neon-bandwidth.md). Small and real,
nowhere near int4-weight quantizing these same tensors ("flips the argmax and
tanks the cosine" — mirrors why GGUF Q4_K_M keeps token_embd/output at Q6_K
while the projections go 4-bit) — kept as the unconditional int4-mode default,
not opt-in, given the size of the win against the size of the cost. int8 and
f32 modes use themselves, unaffected.
```

## quantMode.embeddingWith

Moved from `decoder/weightmat.go` (the comment above `quantMode.embeddingWith`) on 2026-10-10.

```text
embeddingWith resolves the embed/head precision allowing the int8 pin to be
relaxed to int4 (Options.EmbedInt4): in int4 mode the table goes int4 too,
halving what is the single largest resident tensor on a big-vocab small model.
Lossy and opt-in — a 1.5B Q4_K_M spike measured ~2.3 pts top-1 vs the pin (≈0 on
frequent tokens, ~3 on rare). Off (the pin) is the default and the bit-exact path.
```

## int4ParThreshold

Moved from `decoder/weightmat.go` (the comment above `int4ParThreshold`) on 2026-10-10.

```text
int4ParThreshold lowers the fan-out threshold for the int4 (W4A8) matmul below aikit's
default (parThreshold = 1<<24 = 16.78M MACs) so the small int4 DECODE matmuls parallelize.
At decode (M=1) every Gemma-4 int4 matmul is small — expert gate‖up 3.96M, down 1.98M,
dense ~5.9M, attention ~11.5M MACs — so ALL of them fell under aikit's default and ran
SERIAL, while only the int8 LM head (738M) parallelized. That serial fast-path (NOT a
barrier) capped 8-core scaling at 1.61× and decode at ~2.3 tok/s (profiled on the real
gemma4-26b int4 .giw). 1<<20 ≈ 1.05M sits below the 1.98M smallest decode matmul, so all of
them fan out, while truly tiny ops (<1M) stay serial. Byte-identical (aikit partitions
output columns in 8-wide groups — the width-invariant contract), measured ~2.3× decode
(2.3→5.3 tok/s) + TTFT 7.3→3.2s. Only widens fan-out (never narrows it), so prefill's
already-parallel large-M matmuls are unaffected. See docs/task-gemma4-moe.md.

PROVENANCE: 1<<20 was **Ryzen 7 3700X (8-core) measured**. On that rig it does not regress
the small end — a 0.5B int4 decodes 1.9× faster than serial at this value (26.8 vs 13.9
tok/s, 4 cores) because it parallelizes the ~4M-MAC matmuls while leaving truly tiny (<1M)
ops serial (thr=0, which fans out everything, was *slower* there — over-parallelizes).

M1 PRO SWEEP (6P+2E, BenchmarkInt4ParThresholdSweep): the value TRANSFERS — it sits in the
flat-optimal region. All four gemma4-26b decode shapes are ≥1.98M, so 1<<20 (1.05M)
parallelizes every one, capturing 1.46× (down 1.98M), 1.84× (gate_up 3.96M), 1.9× (dense
5.9M), 2.56× (attn 11.5M) vs serial. And UNLIKE the Ryzen, thr=0 shows NO over-parallelize
penalty on M1 Pro (thr=0 ties the low thresholds), so the Ryzen value is if anything slightly
conservative here but lands squarely in the optimum for every real decode op. No per-platform
split warranted. Re-run the benchmark if the core topology or aikit's kernel changes.
```

## streamQuantizedRepackable

Moved from `decoder/weightmat.go` (the comment above `streamQuantizedRepackable`) on 2026-10-10.

```text
streamQuantizedRepackable is streamQuantized's twin for the GGUF streaming path, shared by
streamQuantizedEmbed (Embed/LMHead) and streamQuantizedBatchedProj (attention Q/K/V, MLP
gate/up) — see decoder/weightmat.go's quantizeEmbedWM/quantizeBatchedProjWM/
repackedOnlyOrCanonical for the full policy and each caller's own safety argument. The
row-quantize loop is identical regardless of final layout — the quantized BYTES are the same —
so this only differs from streamQuantized in the last step for quantInt4: the repack decision
routes through repackedOnlyOrCanonical (row4-only when needCanonical is false and the shape/
core qualify; canonical, optionally row4-skipped, when needCanonical is true) instead of
streamQuantized's own hardcoded canonical+row4 "both" — this is why quantInt4 no longer
early-returns into streamQuantized the way it used to (that hardcoding is exactly what made
skipRow4 (M-07, audit-metal-2026-09-12.md) unreachable from the GGUF streaming path).
```

## repackW4A8Row4IfEligible

Moved from `decoder/weightmat.go` (the comment above `repackW4A8Row4IfEligible`) on 2026-10-10.

```text
repackW4A8Row4IfEligible opts wm into the arm64 split-half + 4-row-interleaved
W4A8 layout (docs/completed/task-w4a8-neon-bandwidth.md's item-3+4 harness, GO
2026-08-23/24) by calling linalg.WeightMat.RepackInt4Row4 — a no-op on
non-int4 WeightMats, non-arm64 builds, and any shape the repack rejects
(rows not a multiple of 4, cols not a multiple of the int4 group size), so
always safe to call unconditionally.

ONLY wired into streamQuantized/quantizeWM — the GGUF/safetensors streaming
paths, which always allocate fresh heap-backed q4/q4s. Deliberately NOT
wired into the .giw loader (decoder/serialize.go): .giw tensors zero-copy
mmap-alias their packed bytes, and SOME of those (MoE experts, when
newExpertPager's paging is active) are later released from RAM on demand
via madvise DONTNEED (moepaging.go) — a heap-resident row4 copy sitting
alongside a pageable mmap alias would pin that memory permanently, defeating
paging's whole point for exactly the tensors it exists to bound. Every
GGUF/safetensors-streamed int4 tensor is heap-backed regardless (never
paged — moepaging.go's own MappedSpan check silently skips heap-backed
weights), so repacking here adds no new pageability constraint. Extending
this to non-paged .giw tensors is a real follow-up, out of scope for this
pass: it needs the repack decision sequenced after newExpertPager decides
which specific experts it's managing, not made at load time before that
decision exists.
```

## w4a8Row4RepackEnabled

Moved from `decoder/weightmat.go` (the comment above `w4a8Row4RepackEnabled`) on 2026-10-10.

```text
w4a8Row4RepackEnabled is a load-time-measurement toggle ONLY — production
code never sets it, so it stays true always in a real build. A test
measuring the repack's load-time/resident-memory delta (the two numbers
the parked .giw-kind decision is waiting on, docs/task-w4a8-neon-
bandwidth.md) flips it off to get an apples-to-apples "without repack"
baseline from the exact same load path, rather than comparing against a
differently-built binary.
```

## repackW4A8SplitHalfIfEligible

Moved from `decoder/weightmat.go` (the comment above `repackW4A8SplitHalfIfEligible`) on 2026-10-10.

```text
repackW4A8SplitHalfIfEligible is the amd64 counterpart to
repackW4A8Row4IfEligible: it opts wm into the split-half W4A8 nibble layout
(byte i holds weight i's low nibble and weight i+16's high nibble, so the
AVX2 kernel's two per-group VPUNPCK{L,H}BW disappear — docs/queue-
performance.md P14 item 3, measured 1.12x hot AND cold on Zen 2). A no-op on
non-int4 WeightMats, on non-amd64 builds, on CPUs without AVX2, and on any
shape the repack rejects, so it is always safe to call unconditionally.

ALSO a no-op on hosts WITH AVX-512 VNNI, which is the surprising one: aikit's
canonical W4A8 dot prefers its VNNI tier there and split-half exists only at
AVX2, so the layout would swap a faster kernel for a slower one. aikit
declines rather than pessimize. The consequence here is that this repack —
and the +2.10% below — applies to AVX2-WITHOUT-VNNI hosts only, which is a
narrower audience than "amd64".

Wired into exactly the same two call sites as the row4 repack and for the
same reason — see that function's comment for why the .giw loader is
deliberately excluded. The constraint is identical here: the repack
ALLOCATES a second buffer and never writes through the canonical bytes,
which for a .giw kind=3 tensor are a zero-copy mmap alias of the file.
Rewriting them in place would silently misdecode every existing bundle, with
no error and wrong numbers; aikit's TestWeightMatSplitHalf_canonicalUntouched
pins that it does not.

MEMORY: this is a second copy of every eligible tensor's nibbles, and
canonical is NOT dropped. The cost, the measurement that priced it, and why
it is default-off live on w4a8SplitHalfRepackEnabled below — deliberately in
ONE place, so the figures cannot drift apart from each other.
```

## w4a8SplitHalfRepackEnabled

Moved from `decoder/weightmat.go` (the comment above `w4a8SplitHalfRepackEnabled`) on 2026-10-10.

```text
w4a8SplitHalfRepackEnabled is DEFAULT-OFF, and that is a measured decision,
not caution. Set GOINFER_W4A8_SPLITHALF=1 to opt in.

The A/B is recorded in docs/measurements/w4a8-splithalf-decode-ab-
PREREGISTERED.md: on Qwen2.5-Coder-1.5B at int4, Ryzen 7 3700X, interleaved,
same binary both arms, the repack is worth **+2.10% decode tok/s** — real
(floor 0.75%, and the two arms' sample ranges do not overlap at all), but
short of the +4% that was pre-registered as the bar for accepting its memory
cost. It landed in the band the pre-registration named in advance as
AMBIGUOUS -> PARKED, so it parks, with the code and the wiring kept intact.

The cost it is short against: a second copy of every eligible tensor's
nibbles, +0.5 bytes/weight on top of the 0.625 an int4 tensor already pays,
so int4 weight bytes grow ~80%. MEASURED on that 1.5B model, not estimated:
196 tensors repacked, **+624.8 MiB** of duplicate nibbles, taking its int4
weights from 781 MiB to 1.37 GiB.
Canonical is never dropped — M>1 prefill and every non-AVX2 path read it.

Turning this on is defensible where decode latency outranks resident memory
and the machine is amd64 with AVX2 and NO AVX-512 VNNI (aikit declines on
VNNI hosts — see above). It is not defensible as a default, which is why it
is not one. Re-open the decision if the kernel gets faster than
1.12x, or if canonical can be dropped for a build that only ever decodes.
```

## w4a8BatchEnabled

Moved from `decoder/weightmat.go` (the comment above `w4a8BatchEnabled`) on 2026-10-10.

```text
w4a8BatchEnabled runs a layer's q/k/v (and gate/up, where the fused gate+up does not take them) as
one W4A8 fork/join instead of one per projection (audit R-06, aikit MatmulBTW4A8Batch). Bit-identical:
every output column is the same dot product either way. Default per architecture (w4a8BatchDefault,
cpu_tuning_{arm64,other}.go); GOINFER_W4A8_BATCH=0 opts out, =1 forces it on.

History: parked by audit R-06 at 1.08x on the 1.5B against S-02's ≥1.15x ship bar (ambiguous), and a null on
the M1 Pro's 7B (docs/measurements/w4a8-batch-7b-2026-09-20.md). Turned on for non-arm64 by owner
decision 2026-09-27 under docs/tasks/task-cpu-decode-peer-gap-2026-09.md's L2 gates: logits
bit-identical on the real 0.5B/1.5B/7B, and on top of today's fused gate+up (so only q/k/v changes)
paired ABBA 1.016x / 1.030x / 1.018x on the Ryzen 7 3700X, no size regressing.
```

## wantsCanonicalInt4

Moved from `decoder/weightmat.go` (the comment above `wantsCanonicalInt4`) on 2026-10-10.

```text
wantsCanonicalInt4 reports whether this load might need canonical int4 bytes (packed nibbles +
scales) for a tensor somewhere in its lifetime — the two consumers that read them directly
rather than through WeightMat's own layout-agnostic methods:

  - The staged per-token GPU consult (QuantBackend4.MatmulW4A8 / QuantBatchBackend4.
    MatmulW4A8Batch in matmul()/matmulInto()/matmulW4A8Batch): hands q4/q4s to the backend on
    EVERY call, no persistent upload. A repacked-only tensor has nothing to hand it —
    reconstructing canonical on demand there would cost a canonical-sized allocation per
    token, the wrong shape for a per-call path, not merely a deferred optimization.
  - A resident GPU build (ResidencyBackend.BuildResident — cuda/resident.go, metal/model.go,
    gpu/residency.go all read w.Int4() directly to upload a tensor once at build time).

backendName is Options.Backend AS THE CALLER WROTE IT, not be's resolved capabilities — this
is the load of the decision: repacked-only is a PROMISE ("no GPU backend will ever touch this
*Model, resident or staged") that must be STATED, never INFERRED from an omission. Only the
literal "cpu" states it. Empty/unspecified is not a promise of anything — it is simply what a
caller wrote before deciding, or a generic load a DIFFERENT package's code may later hand to
ANY backend's own resident-build machinery outside decoder.Load's own dispatch entirely (this
is not hypothetical: metal's own test suite does exactly this at ~87 call sites — load
generically, then call metal.buildResident on the result directly, which decoder.Load has no
way to see coming). Getting this wrong previously (keying on be's interfaces alone, which are
only known for the backend the CALLER already named) broke exactly that pattern: a model
loaded with Backend unset resolves to the plain CPU backend, which implements none of the
interfaces below, so the interface-only check said "safe" — and a later, out-of-band Metal
residency attempt on that same model then found no canonical bytes to upload.

The interface assertions stay as a SECOND, belt-and-braces guard for the "cpu" case itself:
if some future build ever registers a "cpu" name whose Backend value also happens to
implement one of these (should never happen — NewBackend("cpu") always returns the plain CPU
backend, in every configuration), this still declines to repacked-only rather than trust the
name alone. Every other backendName (a real GPU name, or empty/unspecified) returns true
unconditionally: canonical(+row4) stays today's default, never worse than before this policy
existed.
```

## wantsRow4Fallback

Moved from `decoder/weightmat.go` (the comment above `wantsRow4Fallback`) on 2026-10-10.

```text
wantsRow4Fallback reports whether the arm64 row4 repack should be built ALONGSIDE canonical, as
a safety net for a CPU fallback (M-07, audit-metal-2026-09-12.md). Row4 is read ONLY by the
CPU's own decode kernel (arm64 SIMD dotprod) — no GPU backend's kernels ever read it, resident
or staged; it exists purely so a model that falls back to CPU decode (a declined residency
build, LoRA/session paths outside a resident backend's coverage) keeps its ~1.1-1.3x speed.

False specifically for backendName == "metal": Options.Backend is the caller's own explicit
commitment to that backend (same "stated, never inferred" discipline wantsCanonicalInt4 above
already applies to "cpu"), and row4 is measured to exactly DOUBLE the resident int4 footprint
of every layer projection it applies to (TestW4A8Row4_loadTimeAndMemoryDelta: 223.6 MB
canonical + 223.6 MB row4 on the 0.5B fixture, 100.0% additional RAM). Accepting a slower CPU
fallback in the rare case Metal residency later declines is the trade this backs out of paying
on every load. True for every other backendName (cpu, cuda, webgpu, empty/unspecified) — this
is intentionally narrower than wantsCanonicalInt4's own "any non-cpu name" rule: CUDA/WebGPU
residency have not been measured against this same trade, so they keep today's default.
```

## repackedOnlyOrCanonical

Moved from `decoder/weightmat.go` (the comment above `repackedOnlyOrCanonical`) on 2026-10-10.

```text
repackedOnlyOrCanonical is the shared decision behind quantizeEmbedWM and
quantizeBatchedProjWM: given an already-quantized CANONICAL int4 WeightMat, build it
repacked-only (aikit audit M-22) when needCanonical is false and this core/shape can build
row4 (linalg.Int4Row4Usable) — closing the double nibble+scale residency the canonical+row4
"both" repackW4A8IfEligible policy otherwise pays (aikit's own audit M-22, cross-referenced
from goinfer's audit-2026-09-10.md's own M-22, an unrelated finding sharing the same label by
coincidence of two repos' independent numbering). Falls back to repackW4A8IfEligible's
existing policy whenever needCanonical is true, this core can't build row4 at all (non-arm64,
no dotprod), or the shape doesn't qualify — identical to what every other int4 tensor already
gets from quantizeWM, so this never produces a WORSE outcome than today's default.

amd64 split-half repacked-only is out of scope for both callers: split-half repacking itself
is already a separate, measured-marginal, parked feature (w4a8SplitHalfRepackEnabled, default
off), so there is no default-on amd64 path this closes yet.

skipRow4 (M-07, audit-metal-2026-09-12.md; wantsRow4Fallback's own doc comment has the
measurement) — only consulted in the needCanonical branch, since the !needCanonical branch
below is the CPU-only repacked-only path row4 exists FOR; skipRow4 there would defeat its own
purpose. Callers that must never skip row4 regardless of the backend (Embed/LMHead, read via
.Row() on the host on every backend) pass false unconditionally rather than threading a real
decision through.
```

## quantizeBatchedProjWM

Moved from `decoder/weightmat.go` (the comment above `quantizeBatchedProjWM`) on 2026-10-10.

```text
quantizeBatchedProjWM is quantizeWM's twin for the attention Q/K/V and MLP gate/up
projections — the tensors reached through the batched W4A8 dispatch (matmulW4A8Batch /
wmW4A8Op / isW4A8), which aikit's own audit M-22 note flags as unsafe for a repacked-only op
in general: MatmulBTW4A8Batch has no row4 TILE (unlike WeightMat.MatmulBTW4A8Into's own M>1
case), so a repacked-only op PANICS if it is ever reached at M>1, or when a fan-out shard
boundary splits one of its quads (N%4 != 0 at the boundary).

Verified safe for THIS codebase's actual two batch call sites (decoder/attention.go's
causalAttention, decoder/mlp.go's gatedMLP): both are decode-only functions that call
matmulW4A8Batch with a hardcoded M=1 literal — never reached from the batched-M prefill path,
which uses matmul()/matmulInto() per projection instead (safe at any M, since those dispatch
through WeightMat.MatmulBTW4A8Into directly, not the batch entry point). wmW4A8Op itself needs
no change: it already builds the correct W4:nil/Row4:.../Row4Scales:... op shape for a
repacked-only tensor (aikit's audit confirms this), and isW4A8's IsInt4() gate (already fixed,
this same audit item) is what makes such a tensor reach it at all. The remaining condition —
N%4==0 — is exactly linalg.Int4Row4Usable's own rows%4==0 check, applied per tensor by
repackedOnlyOrCanonical below, so a shape that would violate the quad-boundary constraint
never gets built repacked-only in the first place.

Down-proj, the router, and MoE expert weights are deliberately NOT covered here: down-proj is
unverified against this same batch-path constraint (it is never one of the two batched
tensors today, but has not been separately audited), and expert/layer weights read through a
read-only mmap span (paged .giw loading) are explicitly excluded by aikit's own note — paging
has no load-time repack step, so they stay canonical-only regardless of this policy.

skipRow4 (M-07, audit-metal-2026-09-12.md) is ALSO scoped to just this function's own tensor
class for the same reason: down-proj/router/MoE-expert weights route through quantizeWM
(weightmat.go's own generic quantizer), a SEPARATE function with its OWN unconditional
repackW4A8IfEligible call that takes no needCanonical/backend signal at all — used from ~40
family-specific call sites across weights.go's per-architecture builders. Reaching those too
would multiply this fix's blast radius well past what this pass measured or verified; left as
a separate, larger follow-up. This function's own scope (Q/K/V/gate/up) is a real, smaller
slice of the measured 223.6 MB/223.6 MB row4 overhead (TestW4A8Row4_loadTimeAndMemoryDelta),
not the whole of it.
```

## GIWTarget

Moved from `decoder/weightmat.go` (the comment above `GIWTarget`) on 2026-10-10.

```text
GIWTarget names the single consumer a .giw bundle (or one cmd/prequant run) is
built for, so the writer can choose the one on-disk int4 layout that consumer
actually reads (docs/tasks/task-int4-layout-2026-09.md's L2 — the .giw analogue of
wantsCanonicalInt4's load-time decision). GIWTargetNone ("") means
unknown/multi-consumer and always keeps every int4 tensor canonical (kind 3) —
the safe default: a bundle nobody has promised to a single reader must stay
portable, mirroring wantsCanonicalInt4's own "never infer from an omission" rule.
```

## GIWTargetForBackend.arm64

Moved from `decoder/weightmat.go` (the comment above `GIWTargetForBackend.arm64`) on 2026-10-10.

```text
A kind-5 (row4-only) file is a promise that THIS core can read it, and aikit's row4 kernels need DotProd (Int4Row4Usable). A core without it (a Raspberry Pi 4, or Windows on ARM,
where aikit assumes none) that wrote a cpu-arm64 sidecar would refuse to load its own file and rebuild it on every start: found 2026-10-04 by the first windows-arm64 CI run.
Such a core builds canonical bundles, which any core loads. A cpu-arm64 bundle can still be built for a DotProd reader by naming the target (-target cpu-arm64).
```

## matmul

Moved from `decoder/weightmat.go` (the comment above `matmul`) on 2026-10-10.

```text
matmul computes dst[M, rows] = a[M, cols] · wᵀ, dispatching on w's precision
with goinfer's backend routing: the f32, W8A8 and W4A8 paths can run on a GPU backend
(be.MatmulBT / QuantBackend.MatmulW8A8 / QuantBackend4.MatmulW4A8, the last a G6
docs/tasks/task-gpu-paths-2026-09.md addition); weight-only int8 (Q8) stays CPU. (The old
weightMat.matmul, now a free function over linalg.WeightMat.)
```

## matmul.int4.consult

Moved from `decoder/weightmat.go` (the comment above `matmul.int4.consult`) on 2026-10-10.

```text
G6 (docs/tasks/task-gpu-paths-2026-09.md): staged int4 backend consult, mirroring the W8A8
branch below — matmulInto's own int4 branch gets the same fix, for the same reason.

Nested under Int4()'s narrower ok (canonical bytes present), not the outer IsInt4():
the staged consult hands q4/q4s to the GPU backend on EVERY call (no persistent
upload), so a repacked-only tensor (no canonical bytes, aikit audit M-22) has nothing
to hand it here — reconstructing canonical on demand would cost a canonical-sized
allocation per token, the wrong shape for a per-call path. It falls through to
w.MatmulBTW4A8Into below instead, which dispatches on whichever layout is actually
present (canonical, row4, or split-half) via aikit's own per-arch method.
A per-group activation weight (ActQuantGroup) skips the staged consult: no staged GPU
kernel reads per-group activation scales.
```

## matmul.int4.kernel

Moved from `decoder/weightmat.go` (the comment above `matmul.int4.kernel`) on 2026-10-10.

```text
int4 weights run the int8-activation W4A8 integer kernel at EVERY M this CPU
path reaches (decode AND prefill): it stays integer (int4 weight × int8
activation) and benchmarks faster than the dequant-to-f32 Q4 path at every M,
and its own per-output result is M-independent so, taken alone, batched
prefill is bit-identical to sequential decode ON THIS KERNEL.

THAT DOES NOT MAKE THE WHOLE matmul() CALL M-INDEPENDENT (M-09,
docs/audit-2026-09-10.md): the QuantBackend4 consult just above intercepts
M=1 for a staged webgpu backend (its own MatmulW4A8 declines any M != 1,
gpu/backend.go) and routes it through a completely different kernel (a WGSL
f32 GEMV with f16 group scales) — only M>1 (prefill, speculative verify)
actually falls through to the bit-identical-with-itself CPU kernel this
comment describes. A staged-int4 model on webgpu therefore decodes and
verifies on two DIFFERENT kernels — decoder.Model.SpecDecodeConflict
(decoder/spec_verify_guard.go) now refuses speculative decoding for exactly
this combination (a webgpu backend + int4/int4mix quant); see that guard
before assuming this comment covers the device-backed path too.

The pooled Workspace lowers the fan-out threshold below aikit's default so the
small int4 DECODE matmuls parallelize instead of running serial — see
int4ParThreshold. Each Get is exclusive to this call (Put deferred until
return), so concurrent decode streams never share one — same race-freedom as
the old per-call ws, just with its buffers surviving between calls.

w.MatmulBTW4A8Into (not the raw linalg.MatmulBTW4A8Into free function) so the
load-time layout repacks actually get used here — the repack alone does
nothing without this call using it. BOTH arches depend on this one line:
arm64's split-half + 4-row-interleave (RepackInt4Row4, docs/task-w4a8-neon-
bandwidth.md) and amd64's split-half (RepackInt4SplitHalf, queue-performance
P14 item 3). Neither has a dispatch of its own — aikit picks the layout
inside this method, at M=1 only, whenever the repack populated it.
```

## matmul.w8a8

Moved from `decoder/weightmat.go` (the comment above `matmul.w8a8`) on 2026-10-10.

```text
Pooled Workspace with the int8 decode threshold — the free-matmul path
(e.g. gemma4's own forward) has no scratch Workspace, so without this its
W8A8 decode matmuls would run at aikit's conservative 16.78M default. Same
mechanism as the int4 branch above. matmulInto() gets this via the
decodeScratch Workspace instead. Threshold differs from int4's (300K vs
1<<20) — the crossover is kernel- and model-specific, measured separately;
see DefaultDecodeParallelThreshold + int4ParThreshold.
```

## matmul.q8

Moved from `decoder/weightmat.go` (the comment above `matmul.q8`) on 2026-10-10.

```text
Pooled Workspace, same reason as the W8A8 case just above — the bare
linalg.MatmulBTQ8 wrapper builds a fresh, non-pooled Workspace every call
(docs/completed/lmhead-workspace-fix.md, Step 1: the LM head's own weight-only
Q8 path, the largest per-token cost measured in the W4A8 plumbing phase).
```

## matmulInto

Moved from `decoder/weightmat.go` (the comment above `matmulInto`) on 2026-10-10.

```text
matmulInto is matmul using the caller's Workspace, so steady-state decode quantizes the activation
once into reusable scratch instead of allocating per call.

P7 — this used to dispatch on `isW8A8(w)` and send EVERYTHING ELSE to matmul, which allocates a
fresh Workspace. So W4A8 never reached the per-stream Workspace its six call sites already hand
in, purely because the dispatch named one quantization instead of asking the question it meant.
That is the DISPATCH form of sibling drift (see parity-coverage-policy.md): a check that names one
member fails to CATCH divergences, a dispatch that names one member CREATES them.

The question it meant is "does this weight have an Into form that takes a Workspace", so that is
what it asks now. Adding a third such quantization needs a case here and nothing else.

Race-freedom is unchanged and does not need a new argument: `ws` is the per-stream Workspace on
decodeScratch, and "a cache is one generation stream, so the buffers are never shared
concurrently" (decoder/scratch.go). The per-call Workspace in matmul stays exactly as it was, for
callers that have no scratch at all.
matmulIntoPre is matmulInto(ws, be, w, a, dst, M), using p's block when it was quantized for an input w can take
(R-13; see w4a8Act).
```

## matmulInto.int4

Moved from `decoder/weightmat.go` (the comment above `matmulInto.int4`) on 2026-10-10.

```text
G6 (docs/tasks/task-gpu-paths-2026-09.md): the staged int4 backend consult this branch
never had, mirroring the isW8A8 branch's QuantBackend check above.

Nested under Int4()'s ok, not the outer IsInt4() — see matmul()'s own comment on
this same shape for why a repacked-only tensor cannot serve the staged consult.
```

## qwenAttnWeights

Moved from `decoder/weights.go` (the comment above `qwenAttnWeights`) on 2026-10-10.

```text
Quantizable as of 2026-08-19 — same reason as deltaNetWeights: these were f32 regardless of
Options.Quant, so the 16 softmax layers of a 27.8B Qwen3.8 streamed 6.7 GB per token that
int4 should have made ~1.7 GB. WeightMat stays f32 when no quant is requested.
```

## repackedOnlyInt4Count

Moved from `decoder/weights.go` (the comment above `repackedOnlyInt4Count`) on 2026-10-10.

```text
repackedOnlyInt4Count reports how many of w's matmulWeights() tensors are
int4-resident with NO canonical bytes at all (IsInt4() true, Int4()'s ok
false) — a kind-5 .giw tensor (docs/tasks/task-int4-layout-2026-09.md's L2), or (in
principle, never produced by any writer today) an in-RAM repacked-only build
that somehow reached a .giw round-trip. Used by decoder.Load's .giw branch to
refuse loading such a file under a backend that needs canonical bytes.
```

## LoadWeights

Moved from `decoder/weights.go` (the comment above `LoadWeights`) on 2026-10-10.

```text
LoadWeights reads config.json + model.safetensors from a real on-disk
directory (the HF snapshot layout). The .safetensors blob is mmapped
(not heap-copied) so the 270M's ~340 MB bf16 checkpoint stays in the OS
page cache — same M8 path as encoder.LoadWeights.

NOTE: this widens bf16/f16 weights to f32 on load (BFloat16sToF32 /
Float16sToF32 allocate), which roughly doubles resident RAM vs keeping
the tensors bf16. That's the M1 correctness-first choice; the
half-the-RAM route is per-tile widen inside matmul.
TODO(M8): bf16-resident matmul tiling to
drop the widen-on-load 2× memory cost for the 1B+ checkpoints.

Use LoadWeightsFromFS for fs.FS-backed (MapFS, embed.FS) paths — that
route stays heap-backed because fs.FS doesn't expose a file descriptor.
```

## loadWeights.weightAllocs

Moved from `decoder/weights.go` (the comment above `loadWeights.weightAllocs`) on 2026-10-10.

```text
One atomic add per model load, so the fit guard's test can OBSERVE that a refused load
allocated nothing rather than infer it from an error string. Inferring is how a guard that
fires after the allocation still looks correct (docs/tasks/task-first-hour.md, R3).
```

## loadWeights.close

Moved from `decoder/weights.go` (the comment above `loadWeights.close`) on 2026-10-10.

```text
buildWeightsFromSafetensors retains st (the WeightMats MAY alias its mmap) ONLY on success —
on any of its ~40 error returns st would otherwise leak the mapping + fd. A serve process
probing candidate dirs, or retrying a load of a checkpoint with one missing tensor,
accumulates GBs of address space — the exact leak Model.Close exists to avoid (audit M-08).
```

## loadWeights.releaseSource

Moved from `decoder/weights.go` (the comment above `loadWeights.releaseSource`) on 2026-10-10.

```text
P13: release the SOURCE mapping now when nothing can alias it, instead of holding it for
the model's whole life. The mapping is the bf16 checkpoint — 55.6 GB for a 27B — and the
quantized weights the decode actually reads are a separate, much smaller allocation. Holding
the source means dead pages compete with hot weights for page cache: measured 46.8 GB RSS
against GGUF's 24.5 GB for an IDENTICAL 17.9 GB Go heap, and 1.69x slower decode.
```

## buildWeightsFromSafetensorsTo

Moved from `decoder/weights.go` (the comment above `buildWeightsFromSafetensorsTo`) on 2026-10-10.

```text
buildWeightsFromSafetensorsTo is buildWeightsFromSafetensors with an optional sink. With sink nil it is the resident build,
unchanged. With a sink (StreamTranscodeDir) the generic builder writes the bundle head once the globals exist, then
loads, writes and frees one layer at a time, and loads the embedding (and an untied head) a row at a time, so peak
memory is about the globals plus one layer rather than the whole model. Six dedicated builders stream their layers the same
way (their embedding still loads whole); gpt2, internlm2 and gpt-oss do not stream yet and return errDirNoStream.
```

## buildWeightsFromSafetensorsTo.lora

Moved from `decoder/weights.go` (the comment above `buildWeightsFromSafetensorsTo.lora`) on 2026-10-10.

```text
LoRA merge-at-load validation (M18). Deferred to here so it sees tn — the SAME prefixed name
(language_model.* / model.language_model.* on VL checkpoints, or model.-stripped) that loadProj
looks the delta up by. Two silent-no-op classes this closes:
  - a VL-prefixed base validated clean against bare names, then merge no-op'd every prefixed
    tensor (deltas are unprefixed) — now it fails loudly instead;
  - qwen35/mla load attention via loadQwen35Attn/loadDeepseekAttn, OUTSIDE loadProj, so merge
    never touches their attention deltas — reject rather than half-merge.

qwen35 (D3, docs/tasks/task-constrained-confidence.md) loads its attention and DeltaNet
projections through loadQwen35Attn, which now merges. Its tensor names live in that function
across four layouts (plain in_proj_qkv, qwen3_next's fused in_proj_qkvz, Olmo Hybrid's separate
q/k/v, Olmo's plain full attention), so a list here would be a second copy of them that could
drift, which is V-12's bug. It skips this name check and is covered by checkAllMerged in
loadWeights, which compares the adapter with what the load merged.
```

## buildWeightsFromSafetensorsTo.lmhead

Moved from `decoder/weights.go` (the comment above `buildWeightsFromSafetensorsTo.lmhead`) on 2026-10-10.

```text
LM head: separate tensor when the family/checkpoint is untied, else the
tied embedding serves as the head. Determined by tensor presence so a
checkpoint that ties despite its family default still loads.
The head sits under the checkpoint's top-level prefix like every other tensor (Voxtral: language_model.lm_head.weight, untied): looked up bare it is never found and the embedding silently
becomes the head (found by G-S14e2's text gate: logit cosine 0.15 against transformers; the families with a prefix and a head, Gemma 3 VL and Qwen3-ASR, both tie it).
```

## buildWeightsFromSafetensorsTo.kvshared

Moved from `decoder/weights.go` (the comment above `buildWeightsFromSafetensorsTo.kvshared`) on 2026-10-10.

```text
Cross-layer KV sharing (num_kv_shared_layers, P7): the last N layers carry
NO k_proj/k_norm/v_proj tensors at all in the checkpoint — they reuse an
earlier layer's KV at forward time (runLayersGemma4FromEmbed's kvSrc; see
its own comment). Missing entirely from this branch until P7's real-checkpoint
vision gate actually tried to load a real E2B safetensors checkpoint through
the full decoder (num_kv_shared_layers=20 of 35 layers there) — the GGUF
loader (decoder/gguf.go's loadG4) already had this right; this mirrors it.
l.KVShared here is the PER-LAYER "reuses an earlier layer's KV" flag
(decoder/weights.go's own LayerWeights.KVShared doc comment) — NOT the same
thing as arch.gemma4.KVShared below, which is attention_k_eq_v (a config-level
"V reuses K's projection on global layers" flag; same field name, different
struct, different meaning — see decoder/arch.go's two separate doc comments).
```

## buildWeightsFromSafetensorsTo.lqDim

Moved from `decoder/weights.go` (the comment above `buildWeightsFromSafetensorsTo.lqDim`) on 2026-10-10.

```text
Attention projections ([out, in] row-major). qDim is per-LAYER: Laguna's XS
generations vary the query head count by layer type (48 on full-attention,
64 on sliding), which the real checkpoint shows as q_proj [6144,2048] on
layer 0 and [8192,2048] on layer 1. headsAt collapses to NumHeads for every
other family, leaving lqDim == qDim.
```

## buildWeightsFromSafetensorsTo.ffnPerLayer

Moved from `decoder/weights.go` (the comment above `buildWeightsFromSafetensorsTo.ffnPerLayer`) on 2026-10-10.

```text
Gemma 4 per-layer FFN width (P7): a real safetensors checkpoint can vary
intermediate_size by layer (confirmed on google/gemma-4-E2B-it: layers
0-14 are 6144-wide, layers 15-34 — exactly the cross-layer-KV-shared tail —
are 12288-wide), but config.json carries only ONE scalar intermediate_size,
unlike GGUF, which stores an explicit per-layer array (gguf.go's
ggufIntArray -> cfg.FFNPerLayer). Missing entirely until P7's real-checkpoint
vision gate actually tried to load this checkpoint's full text decoder — the
forward pass already calls arch.ffnAt(i) per layer (forward_gemma4.go) and
silently fell back to the uniform width for every safetensors-loaded gemma4
model. Seeded HERE, sequentially, before the parallel loop above reads it via
arch.ffnAt(i): each layer's real width comes from its own GateProj tensor's
on-disk shape (a cheap header lookup — Tensor() does not decode data), not a
rule guessed from the KV-shared boundary, which is correlated on this one
checkpoint but not something the config asserts holds in general.
```

## buildWeightsFromSafetensorsTo.ple

Moved from `decoder/weights.go` (the comment above `buildWeightsFromSafetensorsTo.ple`) on 2026-10-10.

```text
Per-Layer-Embedding (PLE) model-level inputs (S1.1, docs/tasks/task-multimodal-support-2026-10.md).
Until 2026-10-06 this loader refused a PLE checkpoint outright: only GGUF loaded them, and a
safetensors E-model left them nil and crashed in runLayersGemma4FromEmbed's rmsNorm. The same
three tensors GGUF loads, at the same precisions: the token table with the embedding policy
(streamed a row at a time — E2B's is [262144, 8960], 9.4 GB as f32), the projection as a
plain matmul weight, the norm as-is (safetensors norms carry the HF convention already;
GGUF's vnorm has to undo llama.cpp's baked +1, this path does not).
```

## buildWeightsFromSafetensorsTo.varies

Moved from `decoder/weights.go` (the comment above `buildWeightsFromSafetensorsTo.varies`) on 2026-10-10.

```text
Only record it when it genuinely varies: decoder/features.go's FeatGemma4EModel
gate reads "FFNPerLayer is non-empty" as "this checkpoint declared real per-layer
FFN metadata" (GGUF's own ggufIntArray only returns non-empty for a checkpoint
that actually carries that metadata key at all). Setting it unconditionally here
made every safetensors gemma4 checkpoint — including plain dense/MoE fixtures with
a uniform width — report FeatGemma4EModel, which CUDA/Metal residency does not
implement; found by TestPlan_tableDriven/TestPlan_extraBytesReservedAheadOfExperts
(gemma4-moe-tiny) regressing to "decline" in the full suite, not assumed safe.
```

## loadQwen35Attn

Moved from `decoder/weights.go` (the comment above `loadQwen35Attn`) on 2026-10-10.

```text
loadQwen35Attn loads one qwen3_5_moe layer's attention tensors as f32 (the
parity-first forward uses plain matvec): the Gated DeltaNet set on linear
layers (linear_attn.*), the gated-softmax set on the rest (self_attn.*, with a
double-width q_proj — query ‖ gate per head). The MoE FFN is loaded by the
shared path. See docs/qwen3_5_moe.md.
mkQ builds a quantized-if-requested WeightMat, so this loader honours Options.Quant like every
other family. Passed in rather than rebuilt here because the quant resolution lives in
buildWeights with the rest of the load.

mkQ and f32 are the loader's merge-aware readers (loadMatMerged / loadF32Merged): every tensor here
is read through one of them, so a LoRA delta on any of its projections is merged on the on-disk
tensor, before qwen3_next's fused split or Olmo's q/k/v concatenation rearranges it (D3).
```

## loadFusedExperts

Moved from `decoder/weights.go` (the comment above `loadFusedExperts`) on 2026-10-10.

```text
loadFusedExperts unpacks the real qwen3_5_moe MoE FFN, which stores all experts
as two stacked 3-D tensors instead of per-expert weights: gate_up_proj
[nExpert, 2*inter, hidden] (gate ‖ up concatenated on the output/row axis) and
down_proj [nExpert, hidden, inter]. Splits the fused gate_up and de-stacks per
expert (the safetensors analogue of the GGUF stackedExperts path), then
quantizes each into the resident format.

P-11: streamed via Tensor.SubF32, one expert at a time — the same win
streamExperts already banks for gemma4's fused experts (the bf16-26B transient
fix). The two whole-tensor TensorF32 reads this used to do materialized
nExpert*2*inter*hidden + nExpert*hidden*inter f32 floats before touching a
single expert (~3 GB transient at Qwen3.6-35B-A3B shapes, per layer, times
however many layers parallelLayers has in flight at once) — exactly the
per-layer materialization streamExperts exists to avoid, just not routed
through it because this loader predates the split of that helper out. Same
external behavior (same expertWeights per index, same quantization), smaller
peak.
```

## loadGemma4MoE

Moved from `decoder/weights.go` (the comment above `loadGemma4MoE`) on 2026-10-10.

```text
loadGemma4MoE loads one gemma4 layer's parallel dense+MoE FFN sub-block
(enable_moe_block) into a gemma4MoEWeights, consumed by gemma4MoEFFN. The dense
branch MLP (l.GateProj/UpProj/DownProj) and its sandwich norms (l.PreMLPNorm =
pre_feedforward_layernorm, l.PostMLPNorm = the JOINT post_feedforward_layernorm)
are already loaded by the caller — this aliases them and loads the MoE-specific
tensors: the three parallel-branch norms, the weightless-norm/learned-scale
router + per-expert scale, and the fused gelu-tanh experts (gate_up ‖ down).

Router weights stay f32 (loadMat, no quant — the router is logit-critical); the
experts quantize at load through the layer's quant mode like every other family
(router-f32 / experts-int4). The experts stream one at a time via Tensor.SubF32
(§4): each expert's slice is widened/quantized on its own, so a bf16 26B-A4B never
materializes the whole [128, 2*inter, hidden] gate_up (a ~2 GB/layer transient) —
only one expert's f32 at a time.
```

## tensorSchema.linearNorms

Moved from `decoder/weights.go` (the comment above `tensorSchema.linearNorms`) on 2026-10-10.

```text
PreAttnNormLinear/PostAttnNormLinear/PreMLPNormLinear/PostMLPNormLinear
override the corresponding suffix above on layers where isLinearLayer(i) is
true — but AS A GROUP, not per-field (N-62, docs/audit-2026-09-10.md: this
used to describe a per-field fallback that isn't what the code does).
buildWeightsFromSafetensors ORs the four together into one
hasLinearNormOverride flag; if ANY is non-empty, ALL FOUR replace the
generic suffixes above for isLinearLayer(i) rows — including a field left
"" within that group, which then means "no norm at this position on linear
layers" (Olmo Hybrid's own PostAttnNormLinear/PostMLPNormLinear below), NOT
"fall back to the generic PostAttnNorm/PostMLPNorm". Leaving exactly one of
the four set and the rest "" expecting per-field fallback would silently
drop the other three's generic norms on linear layers instead. Every family
so far (Olmo Hybrid, the only user) sets all four together, which is why
this has never mattered in practice — but the contract is group-or-nothing.
Olmo Hybrid needs this: its two decoder-layer CLASSES each independently
define an attribute literally NAMED "post_attention_layernorm", but in
DIFFERENT POSITIONAL ROLES — a post-attn norm on full-attention layers
(NormPostOnly), a pre-MLP norm on linear/DeltaNet layers (NormPre2) — and
only its linear layers carry an input_layernorm tensor at all (full-attention
layers have none, per NormPostOnly). One static per-family schema can't route
one on-disk name to two different LayerWeights fields depending on layer
kind; these four give linear layers their own suffix set instead.
```

## lfm2TensorSchema

Moved from `decoder/weights.go` (the comment above `lfm2TensorSchema`) on 2026-10-10.

```text
lfm2TensorSchema: LFM2/LFM2.5. Tied head, Pre2 norms under LFM2's own names
(operator_norm before the mixer, ffn_norm before the FFN), per-head RMSNorm on Q and K,
SwiGLU under llama's w1/w2/w3 naming, and attention output as out_proj rather than o_proj.

The attention entries apply to the 8 attention layers only; the 22 conv layers have none of
them and instead carry conv.{in_proj,conv,out_proj}, which this schema cannot express (it has
no conv roles) and buildLFM2Weights loads directly — the same division Granite uses for its
Mamba tensors.

FinalNorm is embedding_norm, not model.norm: LFM2 normalises before the tied LM head under a
name no other family here uses, so a copy-paste of "model.norm.weight" would fail to load
rather than load the wrong thing — which is the better failure, but worth naming.
```

## qwen35TensorSchema

Moved from `decoder/weights.go` (the comment above `qwen35TensorSchema`) on 2026-10-10.

```text
qwen2MoeTensorSchema: qwen2 attention (q/k/v bias) with the FFN replaced by a
sparse MoE (router mlp.gate + per-expert mlp.experts.%d.*) plus an always-on
shared expert (mlp.shared_expert.* + the mlp.shared_expert_gate sigmoid gate).
qwen35TensorSchema covers the qwen3_5_moe SOFTMAX layers (QK-norm, no bias) +
the routed/shared MoE common to every layer. The Gated DeltaNet (linear) layers
carry an entirely different tensor set (in_proj_qkv/z/a/b, conv1d, A_log,
dt_bias, norm, out_proj) that the current tensorSchema can't express; loading
those — and pinning the exact fused-expert tensor names against a real
checkpoint — is Phase 4 (see docs/qwen3_5_moe.md). Used today only for
descriptor resolution.
qwen35DenseTensorSchema is Qwen3.8's (model_type qwen3_5): identical to the MoE sibling's
except the router/expert names give way to a plain SwiGLU. Kept as its own value rather
than mutating the MoE schema, so the MoE families are untouched by this addition.
```

## buildGPT2Weights.maybeQuant

Moved from `decoder/weights.go` (the comment above `buildGPT2Weights.maybeQuant`) on 2026-10-10.

```text
maybeQuant streams a matmul weight to per-row int8 when quant is set,
freeing its f32 (see loadWeights). The Conv1D projections are built with
newWeightMat (post-transpose), so the quantization is applied here rather
than in a loader closure. skipRow4 (M-07, audit-metal-2026-09-12.md): this
covers every layer projection quantizeBatchedProjWM's own dispatch never
reaches (GPT-2 doesn't route through buildWeightsFromSafetensors at all).
```

## lagunaTensorSchema

Moved from `decoder/weights.go` (the comment above `lagunaTensorSchema`) on 2026-10-10.

```text
lagunaTensorSchema: Laguna (poolside). Read from the REAL Laguna-XS.2 checkpoint
index rather than inferred from modeling_laguna.py, because the two disagree in
two places that matter:

 1. The module allocates FUSED 3D expert parameters (LagunaExperts holds
    gate_up_proj [E, 2*inter, hidden] and down_proj [E, hidden, inter]), but the
    shipped checkpoint stores PER-EXPERT 2D tensors — 9984 = 39 MoE layers × 256
    experts of each. HF re-packs them at load via its conversion mapping. The
    per-expert form is what goinfer already reads, so the Expert* templates apply
    unchanged and no stacked-expert handling is needed.

 2. The module names the shared expert self.shared_experts (PLURAL, as GLM and
    DeepSeek do), but the checkpoint keys are mlp.shared_expert.* (SINGULAR).

RouterBias is likewise the SHIPPED spelling: the bias lives under mlp.experts.*
on disk and HF's _checkpoint_conversion_mapping rewrites it to mlp.gate.* at
load, so reading the checkpoint directly means taking the experts spelling.

The dense prefix layers (mlp_only_layers) use the plain GateProj/UpProj/DownProj
names at the model's intermediate_size; the MoE layers use the Expert*/Shared*
names at moe_intermediate_size. See docs/task-laguna.md.
```

## gguf.go: header

Moved from `decoder/gguf.go` (the comment above `gguf.go: header`) on 2026-10-10.

```text
GGUF loading — read a quantized llama.cpp checkpoint and
run it through the generic forward. The GGUF file carries both the
architecture config (metadata) and the weights (dequantized from the mmap,
then optionally re-quantized to resident int8/int4 per the quant mode), so no
separate config.json/safetensors is needed. Layout quirks vs the HF safetensors
path are normalized at load: tensors use llama.cpp's blk.N.* names; the q/k
projections of NORM-rope archs (llama/mellum) are in llama.cpp's interleaved-
RoPE permutation, inverted here to this package's rotate_half order (NEOX-rope
archs — qwen/gemma — are left as-is, see ggufQKPermuted); and Gemma's (1+w)
norm offset, baked into the stored weights by llama.cpp, is subtracted back out.

Architectures: whatever decoder/registry.go's `registry` map carries (34 as of 2026-09-02 —
llama/mistral/mixtral/phi3, the qwen2/qwen3/qwen3_5 family, gemma3/gemma4, gpt2, gpt_oss,
cohere, deepseek_v2/v3, kimi_k2, glm4_moe, granitemoehybrid, nemotron_h, internlm2/3, laguna,
lfm2, llama4_text, mellum). NOT restated as a fixed list here: this comment carried five names
for long enough that it read as a limit rather than an example (N-34). Read the map.
Quant types: F32/F16,
Q8_0/Q4_0/Q5_0, the K-quants Q2_K/Q3_K/Q4_K/Q5_K/Q6_K, and IQ4_NL/IQ4_XS.
```

## ggufGraniteDenseConfig

Moved from `decoder/gguf.go` (the comment above `ggufGraniteDenseConfig`) on 2026-10-10.

```text
ggufGraniteDenseConfig builds a dense Granite 4.2 Config from the granite.* metadata.
Verified against a real file's header (HTTP-Range-fetched, bartowski/granite-4.2-3b-GGUF
Q2_K): architecture string "granite" (distinct from the hybrid's "granitehybrid"), and
llama.cpp bakes Granite's scalar multipliers directly into metadata — attention.scale (the
resolved attention_multiplier, not 1/√d), embedding_scale, logit_scale, residual_scale — so
no separate multiplier-vs-default resolution is needed the way the safetensors path does.
The tensor set is exactly llama's (no separate ggufLlamaConfig reuse possible here since the
metadata NAMESPACE differs, but the loader dispatch for actually reading tensors is identical
once resolveArchitecture returns llamaTensorSchema).
```

## ggufQwen3MoeConfig

Moved from `decoder/gguf.go` (the comment above `ggufQwen3MoeConfig`) on 2026-10-10.

```text
ggufQwen3MoeConfig builds a Qwen3-MoE Config from the qwen3moe.* metadata.
Verified against a real file's header (HTTP-Range-fetched, first 30MB of
unsloth/Qwen3-30B-A3B-GGUF's Q2_K — well under the 15GB+ full file), not
assumed: architecture string is literally "qwen3moe", the metadata is the
plain {arch}.attention.head_count/head_count_kv/key_length/
layer_norm_rms_epsilon, {arch}.expert_count/expert_used_count/
expert_feed_forward_length, {arch}.rope.freq_base — no sliding-window or YaRN
keys at all (single global RoPE, unlike Mellum's per-layer-type split) — and
the tensor set carries attn_q_norm/attn_k_norm and
ffn_gate_inp/ffn_{gate,up,down}_exps but NO ffn_*_shexp (no shared expert,
matching the real config.json's absent shared_expert_intermediate_size).
norm_topk_prob isn't carried as GGUF metadata; hardcoded true to match the
real config (HF Qwen3MoeConfig's own default, same convention ggufMellumConfig
uses for its own missing metadata).
```

## ggufLagunaConfig

Moved from `decoder/gguf.go` (the comment above `ggufLagunaConfig`) on 2026-10-10.

```text
ggufLagunaConfig builds a Laguna Config from the laguna.* metadata. llama.cpp
has FIRST-CLASS laguna support (general.architecture == "laguna"), and the GGUF
carries the family's two awkward parts more cleanly than the safetensors config
does: the per-layer QUERY head counts arrive as an ARRAY
(laguna.attention.head_count), and the two layer types' rotary widths as separate
rope.dimension_count / rope.dimension_count_swa scalars.

THREE THINGS THE GGUF DOES NOT SAY, each handled explicitly:

 1. WHICH LAYERS ARE FULL. There is no layer_types and no sliding-window PATTERN
    key, so it must be derived. The head-count array encodes it (48 on
    full_attention, 64 on sliding on the XS line) and layer 0 is full in every
    released config, so layers matching heads[0] are the full ones. That is an
    INFERENCE, so it is validated: at most two distinct counts, and the derived
    split is reported to the caller through LayerTypes for the gate to assert.

 2. THE GATE'S GRANULARITY. There is no gating key at all — matching the
    safetensors path, where the declared value is unreliable anyway (XS.2 says
    `gating: true` and ships a per-HEAD tensor). The loader reads it from
    blk.0.attn_gate.weight's shape, which is the authority in both formats.

 3. YARN'S attention_factor. llama.cpp writes rope.scaling.yarn_attn_factor = 1.0
    as its "unset" sentinel and computes the mscale itself. Passing 1.0 through
    would REPLACE YaRN's mscale with a no-op: goinfer's attention_factor is a
    *float64 whose nil means "compute get_mscale(factor) = 0.1·ln(factor)+1", which
    for factor 32 is 1.3465735902799727 — exactly what the safetensors config
    states. So the field is OMITTED at the sentinel and passed through otherwise.
```

## ggufGraniteConfig.nope

Moved from `decoder/gguf.go` (the comment above `ggufGraniteConfig.nope`) on 2026-10-10.

```text
NoPE. The released granite-4.0-h models set position_embedding_type "nope" and the
converter carries that across as rope.scaling.finetuned — the rope.dimension_count /
rope.freq_base keys are written regardless and are vestigial here (this file has
dimension_count 128 on a model HF ropes not at all). Only an explicitly present key
flips the behaviour; absent leaves the roped path, so an older GGUF cannot silently
lose its RoPE. Verified against the bf16 oracle: roped ⇒ cosine 0.9936 + a wrong
continuation, NoPE ⇒ 0.9995 + exact.
```

## ggufNemotronConfig.moe

Moved from `decoder/gguf.go` (the comment above `ggufNemotronConfig.moe`) on 2026-10-10.

```text
Nemotron 3 Nano's MoE fields — key names verified against a real GGUF file's
metadata (bartowski/nvidia_Nemotron-3-Nano-30B-A3B-GGUF), not assumed from the
safetensors config's field names or llama.cpp's convert script alone:
expert_count/expert_used_count/expert_feed_forward_length/
expert_shared_feed_forward_length/expert_shared_count/expert_weights_norm/
expert_weights_scale/expert_group_count/expert_group_used_count — all present,
including expert_group_used_count (the topk_group equivalent), which is easy to
assume absent since the safetensors config's own topk_group has no direct GGUF
key of the same name.
```

## StreamTranscodeGGUF.eos

Moved from `decoder/gguf.go` (the comment above `StreamTranscodeGGUF.eos`) on 2026-10-10.

```text
M-04 (docs/audit-2026-09-10.md): this streaming path never goes through decoder.Load, so it
never picked up Load's own resolveEOSIDs write-back — a .giw built by StreamTranscodeGGUF
carried only cfg.EOSTokenID's raw GGUF-metadata value, dropping any extra stop id a sibling
generation_config.json next to the .gguf would have added, and (found only by
TestStreamTranscodeMatchesResident, which byte-diffs this path against decoder.Load's own
resident-then-serialize output) making the two bundles diverge before the quant label even
on a fixture with no such file to lose. Mirrors decoder/model.go's Load fix exactly: resolve
against the GGUF's own parent directory (os.DirFS(path) can't open anything inside a FILE),
then write the resolved set back into cfg.EOSTokenID so it round-trips through the bundle.
```

## StreamTranscodeGGUF.canSerialize

Moved from `decoder/gguf.go` (the comment above `StreamTranscodeGGUF.canSerialize`) on 2026-10-10.

```text
canSerialize once refused MLA / Mamba-2 / Gemma-4 PLE / Llama-4 here; since v6 the writer
expresses all of them and it returns nil unconditionally. The comment that used to sit
here still described the refusal, which is how M-09 stayed invisible: the families it
named are exactly the ones whose GGUF branch never drives the sink, and the reader was
told they could not get this far.
```

## StreamTranscodeGGUF.stream

Moved from `decoder/gguf.go` (the comment above `StreamTranscodeGGUF.stream`) on 2026-10-10.

```text
Every family streams (S2, task-never-swap-2026-09.md): buildWeightsFromGGUF's sink != nil branch
builds, writes and releases one layer at a time, so peak RSS is ~one layer rather than the whole
model. qwen35 got there first (2026-08-24, docs/completed/task-zeno-compare.md: a 35B-A3B MoE had
OOM'd at 40.5 GB resident on a 16 GB Mac under the old resident-then-serialize path); gpt-oss,
laguna, granite, nemotron and llama4 on 2026-09-23; gemma4, the last, on 2026-09-24. The
needsResidentSerialize list that routed families around this path is gone with its last entry.

needCanonical=true unconditionally, regardless of target: the writer needs canonical bytes IN RAM
to choose what to write (repackRow4ForEmit computes row4 from canonical), even on a cpu-arm64
target that will write kind 5 (canonical-absent) to DISK — per layer, that is one layer's
canonical bytes. See docs/tasks/task-int4-layout-2026-09.md's L2. abort=nil: this is the
cmd/prequant transcode path, which observes ctx per layer via M-21's ctxWriter (S3 scopes the
swap tripwire's load-time consumer to decoder.Load's resident build only — see Options.LoadAbort).
```

## buildWeightsFromGGUF

Moved from `decoder/gguf.go` (the comment above `buildWeightsFromGGUF`) on 2026-10-10.

```text
buildWeightsFromGGUF dequantizes the GGUF tensors into the weight bundle.
When quant is set, each matmul tensor is re-quantized (per-row int8 or
group-wise int4) right after it is dequantized (and un-permuted) and its f32
is freed — so a Q4/Q8 GGUF lands resident as int8/int4 (~¼ / ~⅛ f32) without
ever materializing the whole model in f32 (see loadWeights). The GGUF's own
quant is lossy and so is the re-quant, but it captures nearly all of what a
Q4_K_M file carries.
When sink is non-nil, the weights are STREAMED to it (a .giw body) instead of
retained: the header + globals are written, then each layer is loaded, written,
and freed in turn, so peak RAM is ~one layer rather than the whole model — this is
what lets a model larger than RAM be transcoded. The returned *Weights then holds
no layer tensors (they were freed); the caller writes the trailing CRC. Streaming
is supported for the generic per-layer loader (llama/qwen2/qwen3/mellum/glm4_moe) and for
qwen35's own dedicated branch, which streams per layer too (N-64, docs/audit-2026-09-10.md:
this used to say qwen35 rejected streaming — loadQ35 builds each layer independently, so a
sequential build-then-write-then-release loop bounds peak RSS the same way). Every other family's
dedicated branch streams the same way (S2, task-never-swap-2026-09.md: gpt-oss, laguna, granite,
nemotron and llama4 on 2026-09-23, gemma4 on 2026-09-24). A branch that returns without driving
the sink would write a header declaring N layers and then none — the M-09 failure — so any new
family branch must stream too; decoder/gguf_streaming_shape_test.go checks its closure reads only
per-layer tensors, and TestGemma4GGUF_streamedMatchesResident-style byte identity is the gate.
```

## stackedExperts.parallel

Moved from `decoder/gguf.go` (the comment above `stackedExperts.parallel`) on 2026-10-10.

```text
A streaming transcode builds one layer at a time (S2), so the across-layer parallelism the
resident build gets from parallelLayers is gone — and the experts ARE the layer (the 26B-A4B:
~95% of each layer's bytes). Build them in parallel instead: the dequantizer is a pure function
of the mapped tensor bytes and streamQuantized allocates its own scratch, so concurrent experts
share nothing, and each lands in its own res[e] — byte-identical to the sequential order.
Measured on the 26B-A4B (nobara, 2026-09-24): sequential streaming ran at 93% CPU and took
4.9x the resident build's wall time. The resident path (sink == nil) is left exactly as it was;
it already runs one layer per core.
```

## loadQ35.wmQ

Moved from `decoder/gguf.go` (the comment above `loadQ35.wmQ`) on 2026-10-10.

```text
wmQ is f32mat + Options.Quant, for the projections that dominate decode bandwidth. Until
2026-08-19 this path (like its safetensors twin) kept them f32 regardless of the requested
quant — "parity-first" from the bring-up — which on a 27.8B Qwen3.8 meant ~29 GB of f32
weights streamed per token while the FFN was int4. Transform-then-quantize: the untile
below has to see f32.
M-07 (audit-metal-2026-09-12.md): these are exactly "the projections that dominate decode
bandwidth" per this function's own comment above — real candidates for the row4 side-copy
waste, so both route through the skip-aware wrapper like every other family's layer
projections.
```

## loadQ35.stream

Moved from `decoder/gguf.go` (the comment above `loadQ35.stream`) on 2026-10-10.

```text
Streaming (sink != nil, e.g. cmd/prequant): loadQ35 already builds one
layer's data independently of every other layer — parallelLayers's own
concurrency is an unrelated speed choice, not a correctness dependency —
so a sequential build-then-write-then-release loop produces bit-identical
per-layer output while bounding peak RSS to ~one layer instead of all of
them (docs/completed/task-zeno-compare.md, 2026-08-24: the old always-parallel,
always-resident path OOM'd a 35B-A3B MoE at 40.5GB on a 16GB Mac).
Non-streaming (sink == nil, regular resident load): unchanged, still
parallel — every qwen35-family model tried before now fits resident, and
there is no reason to slow that path down.
```

## loadGptOss.router

Moved from `decoder/gguf.go` (the comment above `loadGptOss.router`) on 2026-10-10.

```text
MoE: router (+ logit bias) + stacked routed experts (+ per-expert biases). Router
stays f32 regardless of the ambient quant mode (matching gptoss_safetensors.go and
qwen35's streamMat(..., quantNone, ...) below) — top-k selection is discrete, so
quantizing it flips which experts win rather than adding rounding noise. Plain mat()
here was a latent bug: it had never been exercised at a non-f32 quant until Metal
residency's f32Mat(router) panic caught it.
```

## loadGptOss.stream

Moved from `decoder/gguf.go` (the comment above `loadGptOss.stream`) on 2026-10-10.

```text
S2 (task-never-swap-2026-09.md): streaming (sink != nil, e.g. cmd/prequant) —
loadGptOss already builds one layer's data independently of every other layer
(every tensor it reads is named blk.{i}.*, including the per-layer AttnSinks and
RouterBias this family keeps per-layer rather than as a model-level tail; stackedExperts
and stackedExpertBias are per-tensor RowDequantizer reads keyed by the same blk.{i}.*
name, not a whole-file or cross-layer dependency) — the loadQ35 shape (2026-08-24,
docs/completed/task-zeno-compare.md) applies unchanged: a sequential build-then-write-
then-release loop bounds peak RSS to ~one layer instead of the whole resident model,
which is what the historical gpt-oss-20b 22.9 GB swap incident this whole task is
written against actually built. Non-streaming (sink == nil): unchanged, still parallel.
```

## loadGranite.doc

Moved from `decoder/gguf.go` (the comment above `loadGranite.doc`) on 2026-10-10.

```text
Granite-4.0-H (granitehybrid): per-layer Mamba-2 mixer or GQA attention, MoE on
every layer. The Mamba-2 ssm_* tensors use llama.cpp's conventions (shared with
qwen35): ssm_a stores −exp(A_log) directly (reversed to A_log via log(−a) so the
shared mamba2Step works), conv1d is [convDim, K], ssm_norm raw. Mixer stays f32
(parity-first); experts/attention/embeddings quantize. NEOX rope ⇒ no q/k permute.
Laguna (poolside). llama.cpp names its tensors the way every other MoE family
here is named, so the only genuinely new one is blk.N.attn_gate.weight — the
softplus output gate. Two shape details drive the rest:

  * qDim is PER LAYER (arch.headsAt(i)): full-attention layers project 48 heads
    and sliding layers 64 on the XS line, so attn_q/attn_output differ by layer.
  * the gate's granularity is read from attn_gate's ROW COUNT, exactly as the
    safetensors loader does, because neither format carries a trustworthy
    declaration of it (XS.2's config says per-element and ships per-head).

Experts are FUSED+STACKED per projection (ffn_*_exps), the shared expert is
*_shexp, and the router bias is exp_probs_b — all shapes goinfer already reads
for GLM/DeepSeek/Granite. Layers below FirstKDense are plain dense FFNs.
```

## loadNemo.moe

Moved from `decoder/gguf.go` (the comment above `loadNemo.moe`) on 2026-10-10.

```text
Nemotron 3 Nano's MoE FFN. Tensor names verified against a real GGUF
file's tensor list (bartowski/nvidia_Nemotron-3-Nano-30B-A3B-GGUF,
fetched directly and parsed with this package's own GGUF reader — not
assumed from llama.cpp's conversion-script source mapping, which names
the SOURCE safetensors tensor, not the output GGUF tensor). Experts are
FUSED per projection (one 3-D [in,out,nExpert] tensor each), unlike the
safetensors path's one-tensor-per-expert layout — stackedExperts is the
existing helper other GGUF MoE families already use for this shape.
exp_probs_b.bias is llama.cpp's name for e_score_correction_bias.
```

## loadG4.layerScalar

Moved from `decoder/gguf.go` (the comment above `loadG4.layerScalar`) on 2026-10-10.

```text
The per-layer output scalar is read BEFORE the MoE branch below, which copies it into
gemma4MoEWeights.layerScalar — the value the MoE forward multiplies the whole layer's output by
(forward_gemma4_moe.go). Until 2026-09-24 this assignment sat at the end of loadG4, after that
copy, so every MoE layer of a DIRECTLY loaded gemma4-26B GGUF was scaled by 0 and the model
emitted only <pad> (token 0); the .giw reader and the safetensors loader already read it first,
which is why the sidecar was fine (TestGemma4GGUF_moeLayerScalarMatchesLayer).
```

## loadG4.stream

Moved from `decoder/gguf.go` (the comment above `loadG4.stream`) on 2026-10-10.

```text
S2 (task-never-swap-2026-09.md), 2026-09-24: gemma4 streams like every other family. The
"fused PLE/MoE tail" that kept it on a resident build turned out not to exist in this format:
the model-level PLE inputs (per_layer_token_embd / per_layer_model_proj / per_layer_proj_norm,
plus FFNPerLayer) are written in the HEAD (writeHeadGlobals), before any layer, and loaded above
before loadG4 runs; everything else gemma4-specific — PLE gate/proj/norm, the layer scalar,
KV-shared / K=V flags and the whole 26B-A4B MoE branch — lives in each layer's own record
(giwWriter.gemma4Layer) and is read from blk.{i}.* only (decoder/gguf_streaming_shape_test.go
keeps that true). So: head once the PLE globals exist, then build → write → release per layer.
The 26B-A4B's resident transcode peaked at 34.7 GB RSS (nobara, 2026-09-24), which no 16 GB
Mac can build; this bounds it to ~one layer. Non-streaming (sink == nil): unchanged, parallel.
```

## loadG4.releaseHead

Moved from `decoder/gguf.go` (the comment above `loadG4.releaseHead`) on 2026-10-10.

```text
The head is on disk; nothing below reads these again (the returned *Weights of a
streaming build is discarded — see this function's doc). Holding them was the flat
~1.4 GB under the whole 26B-A4B stream (its 262k-vocab int8 embedding table alone is
~0.74 GB), so let them go before the layers.
```

## loadG4.kEqV

Moved from `decoder/gguf.go` (the comment above `loadG4.kEqV`) on 2026-10-10.

```text
K=V (attention_k_eq_v) on the 12B global layers: V reuses K's projection
(v_norm(k_proj) — see loadG4/runLayersGemma4). Parity-gated against the HF
bf16 oracle by TestGemma4_12B_logitParity (argmax exact, cosine 0.990).
```

## buildWeightsFromGGUF.parallel

Moved from `decoder/gguf.go` (the comment above `buildWeightsFromGGUF.parallel`) on 2026-10-10.

```text
Load the layers in parallel: each is independent (its own linalg.WeightMat slots
over the read-only mmap), and the per-tensor dequant + re-quant is the load's
cost — fanning it out across cores turns a 12B GGUF's ~2 min load into seconds.
```

## serialize.go: header

Moved from `decoder/serialize.go` (the comment above `serialize.go: header`) on 2026-10-10.

```text
This file defines a versioned binary format for an already-quantized *Weights
bundle (a ".giw" — goinfer weights), so the resident weights can be produced
once at build time and embedded, skipping the GGUF dequant+requant on every
launch. The big int8/int4 weight arrays are ALIASED directly over the input
slice at load (zero-copy — this is the speed/RAM win); the small per-row scale
floats and the norm/bias vectors are COPIED (the input isn't guaranteed
4-byte aligned, and unaligned float reads are UB).

Discipline mirrors ken's index_serialize.go: magic + version + a config/quant
guard + CRC — any mismatch returns a typed error and never panics. There is no automatic
fallback to the GGUF: a sidecar that fails its freshness check is rebuilt (internal/prequant),
and an embedded or explicitly named bundle refuses to load.

Format (little-endian throughout):

	magic   [5]byte = "GINFW"
	version uint32
	quant   uint32   (quantMode enum: first-weight kind — the legacy tag, validated on read)
	id      str      (model identity — the source's basename; not validated on read)
	config  str      (Config as JSON; arch is re-derived from it on load)
	quantLabel str   (v5+: the resolved quant label — int4|int4mix|int8int8|int8|native — or "" to
	                  fall back to inference; the reader PREFERS this over re-deriving from kinds)
	Embed, LMHead, PosEmbed     weightMat
	FinalNorm, FinalNormBias    f32
	numLayers uint32
	  per layer: the LayerWeights fields, in declaration order, then a v2 hybrid
	  tail (uint8 kind: 0 none | 1 DeltaNet | 2 gated-softmax) with the
	  qwen3_5_moe per-layer delta / qattn f32 tensors when set.
	crc     uint32   (CRC32-IEEE over every preceding byte)

str  = uint32 len + len bytes
f32  = uint32 len + len*4 LE-float32 bytes   (len 0 ⇒ nil on load)
i8   = uint32 len + len bytes                (aliased on load)
raw  = uint32 len + len bytes                (aliased on load)
weightMat = uint8 kind (0 empty|1 f32|2 q8|3 q4|4 q4-row4|5 q4-row4-only); if
            non-empty: int32 rows, cols, group; uint8 w8a8; then the kind's arrays.
            kind 4 (v7+, legacy — no longer emitted, still read) is kind 3's arrays
            (q4s, q4 — canonical) followed by q4Row4Scales, q4Row4 (the arm64
            split-half + 4-row-interleaved layout, docs/completed/task-w4a8-neon-bandwidth.md
            "Format follow-on") — both layouts, so any reader could use the file.
            Opt-in via SerializeWeightsRow4/SerializeWeightsToRow4, for shapes
            RepackW4A8Row4/RepackW4A8Row4Scales accept; every other int4 tensor
            still writes kind 3.

            kind 5 (v11+) is q4Row4Scales, q4Row4 ALONE — no canonical arrays at
            all, the on-disk form of aikit audit M-22's repacked-only WeightMat
            (docs/tasks/task-int4-layout-2026-09.md's L2). Chosen per tensor by
            giwWriter.target: only on a cpu-arm64 target, only for a tensor whose
            call site opted into kind-5 eligibility (weightMat, not
            weightMatKind3Only — see that function's doc for which tensors are
            excluded and why), and only when repackRow4ForEmit succeeds for this
            shape/core; everything else stays kind 3. Loaded with
            linalg.WrapInt4Row4Only, which declines (a named *SerializeError, not a
            panic or silent fallback) when Int4Row4Usable is false for the
            reader's own core — a kind-5 file is a promise to ONE target, unlike
            kind 4's "usable anywhere" portability. Bit-identical dispatch either
            way (TestDotW4A8SplitHalf4Row_bitIdenticalToCanonical) — this is a
            storage choice, not a numerics one, so no golden depends on which kind
            a tensor took.

v2 added the per-layer hybrid tail so the qwen3_5_moe (DeltaNet + gated-softmax)
family round-trips through .giw; v1 blobs (no tail) are rejected by the version
guard and rebuilt from the source GGUF.

v8 added the per-layer LFM2 short-conv mixer (presence byte + inProj/convW/outProj), the same
shape as the v6 Mamba-2 block. Before it, `grep shortConv decoder/serialize.go` returned nothing:
cmd/prequant wrote a CRC-valid bundle with every conv layer's mixer missing, selfCheck passed
because it only Loads, and the first forward nil-dereferenced in the decode goroutine
(audit-2026-09-02 C-03, a regression of R3).
```

## maxSnapshotCacheBytes

Moved from `decoder/serialize.go` (the comment above `maxSnapshotCacheBytes`) on 2026-10-10.

```text
maxSnapshotCacheBytes caps the KV cache a SESSION SNAPSHOT may ask LoadSession to allocate
(kvsnapshot.go). It bounds the ALLOCATION rather than the blob, because the allocation is
what OOM'd (M-04) and because a well-formed snapshot body can be SMALL while pos is LARGE:
a never-written ring and a KV-shared layer each serialise zero KV bytes, and a ring layer
stores only min(count, W) rows. 16 GiB is far above any legitimate session (a 7B at
pos 32768 wants ~3.8 GB) and far below the TBs the unbounded path could reach.
```

## canSerialize.doc

Moved from `decoder/serialize.go` (the comment above `canSerialize.doc`) on 2026-10-10.

```text
canSerialize reports why a model's per-layer state cannot round-trip through the .giw format,
or nil if it can. The writer expresses the standard attention+MLP(+MoE) block plus
qwen3_5_moe's DeltaNet/gated-softmax extras and (v4) the gemma4 PLE / layer_scalar /
KV-share / MoE tail; it does NOT write MLA latent projections (DeepSeek/Kimi) or Mamba-2
SSM weights (Granite/Nemotron) — so serializing those yields a CRC-valid bundle that
nil-derefs at the first forward. Refuse those families up front rather than emit silent
garbage (C2).
```

## canSerialize

Moved from `decoder/serialize.go` (the comment above `canSerialize`) on 2026-10-10.

```text
EMPTY AS OF v6 (2026-08-19), and that is the point: every registered family is representable.

This used to be a hand-maintained blocklist of families the writer could not express — and it
DRIFTED, twice, silently: gpt-oss rode it while dropping its attention sinks (bundles loaded
clean and generated wrong text) and Laguna rode it while producing bundles the reader refused.
The v6 completeness tail writes GProj, AttnSinks, per-expert biases, and the MLA / Mamba-2
sub-structs, so there is nothing left to list.

KEEP THE FUNCTION. A future family may genuinely be unrepresentable (a new per-layer state
with no field here), and refusing is the correct answer for it — an empty list is today's
truth, not a reason to delete the mechanism. What guards against the drift returning is
TestSerializeCensus_noSilentFieldDrop, which asks the STRUCT whether a round-trip lost
anything rather than asking a human whether they remembered to update this.
```

## SerializeWeightsRow4

Moved from `decoder/serialize.go` (the comment above `SerializeWeightsRow4`) on 2026-10-10.

```text
SerializeWeightsRow4 is SerializeWeights, but ALSO opts every eligible int4
tensor into weightMat kind 4 — the on-disk arm64 split-half + 4-row-
interleaved layout (docs/completed/task-w4a8-neon-bandwidth.md's "Format follow-on"),
so the paged-MoE path can use the faster kernel without an in-RAM repack.
Never the default: SerializeWeights (kind 3 only) is what every existing
caller gets and stays unaffected by this function's existence. A tensor
whose shape RepackW4A8Row4/RepackW4A8Row4Scales reject (the router, or any
int4 tensor not a multiple of 4 rows / group cols), or a run on a non-arm64
build, falls back to kind 3 automatically — this is always safe to call.
```

## SerializeWeightsForTarget

Moved from `decoder/serialize.go` (the comment above `SerializeWeightsForTarget`) on 2026-10-10.

```text
SerializeWeightsForTarget is SerializeWeights for a bundle promised to ONE
consumer (docs/tasks/task-int4-layout-2026-09.md's L2): on a cpu-arm64 target, every
eligible int4 tensor (see weightMat vs weightMatKind3Only) writes kind 5
(row4-only — no canonical arrays at all) instead of kind 3; every other target,
including GIWTargetNone, writes kind 3 for every int4 tensor exactly like
SerializeWeights. This is what internal/prequant.Transcode/EnsureCachedGIW and
cmd/prequant drive now — SerializeWeightsRow4/kind 4 is legacy, kept for its own
"usable on any core" contract, not for this one.
```

## giwWriter.writeHeadGlobals.label

Moved from `decoder/serialize.go` (the comment above `giwWriter.writeHeadGlobals.label`) on 2026-10-10.

```text
v5: the resolved quant label, so the reader need not re-infer it (the source of truth is
recorded, not reconstructed).

GATED ON DATA AVAILABILITY, NOT ON WHICH WRITER IS IN USE (B11). The condition used to be
`wr.sink == nil` — "are we the buffered writer" — on the theory that only the buffered path
has full weights in hand. That conflated two different questions: which io.Writer the bytes
go to, and whether w.Layers is actually populated yet. They agree for the true incremental
GGUF transcode (gguf.go's per-family streaming path calls writeHeadGlobals on a freshly
make()'d, all-zero Layers slice BEFORE streaming any layer in — quantLabel() truly cannot see
real data there, and its default case returns "native", a REAL quant mode, not an empty
string, so calling it unconditionally would have baked a FALSE "native" label into every
genuinely-streamed bundle). But they disagree for a caller that already has a fully-loaded
*Weights and simply chooses the streaming API for its I/O shape — internal/prequant and the
qwen35 GGUF branch (a dedicated loader that fully materializes w, THEN calls
SerializeWeightsTo) both do exactly this — and there the label WAS resolvable, just skipped
because the wrong signal was being tested. That mismatch is B11: a buffered and a streamed
call on the SAME fully-loaded model produced non-identical bytes for no reason tied to the
data itself, differing by exactly len("int8int8") = 8 bytes in the length-prefixed field.
```

## LoadSerializedWeights

Moved from `decoder/serialize.go` (the comment above `LoadSerializedWeights`) on 2026-10-10.

```text
LoadSerializedWeights reconstructs a *Weights from a SerializeWeights blob
WITHOUT any dequant/requant. Big int8/int4 arrays are aliased into data
(zero-copy); float arrays are copied. data MUST stay alive for the returned
model's lifetime (the aliased slices point into it). On any magic/version/
quant/arch/CRC mismatch it returns a *SerializeError.

N-34: that used to end "so the caller can fall back to the GGUF". No caller does —
decoder/model.go and internal/chatapp both return the error, the latter telling the operator
to rebuild the bundle or pass --model <gguf> themselves. The typed error is still worth having
(it distinguishes a corrupt bundle from an I/O failure); the fallback it promised was never
built.
```

## validateShapes.headsAt

Moved from `decoder/serialize.go` (the comment above `validateShapes.headsAt`) on 2026-10-10.

```text
headsAt(i), not NumHeads: Laguna varies the QUERY head count per layer (48 on its
full-attention layers, 64 on the sliding ones), and this line was the last uniform-geometry
assumption in the reader — it rejected a correctly-written Laguna bundle with
"layer 1 QProj: 128 rows, arch expects 64". headDimAt/kvHeadsAt/ffnAt were already
per-layer here; this one was missed because no serializable family had needed it yet.
```

## validateShapes.PerLayerModelProj

Moved from `decoder/serialize.go` (the comment above `validateShapes.PerLayerModelProj`) on 2026-10-10.

```text
M-11: the model-level PLE tail. gemma4's forward reads all three unconditionally when
the arch declares PLE, and a bundle from before v4 has none of them.

PerLayerModelProj is [NumLayers*HiddenSizePerLayerInput, HiddenDim] — gguf.go builds it as
mat("per_layer_model_proj.weight", pleTotal, hidden) and forward_gemma4.go's
matmul(&PerLayerModelProj, h /*[hidden]*/, ctxAware /*[pleTotal]*/, 1) both agree Rows() is
pleTotal, not HiddenDim. This check originally asserted Rows()==HiddenDim — the wrong axis —
so it failed on every real gemma4-E2B round-trip (observed: 8960 rows, "expects" 1536) even
though the loader, forward pass, and serializer all round-trip the tensor correctly. Caught
2026-09-03 by TestSerializeGemma4E2B_roundTrip on an overnight parity sweep.
```

## validateShapes.PerLayerProjNorm

Moved from `decoder/serialize.go` (the comment above `validateShapes.PerLayerProjNorm`) on 2026-10-10.

```text
Same wrong-axis bug as PerLayerModelProj above, on a second field: gguf.go's
vnorm("per_layer_proj_norm.weight", g4.HiddenSizePerLayerInput) and forward_gemma4.go's
normalize(arch, row, m.w.PerLayerProjNorm, nil, pleDim) both size it at
HiddenSizePerLayerInput (the RMSNorm runs over one pleDim-wide row of ctxAware), not
HiddenDim. Unreached by the real-model round-trip until PerLayerModelProj's check above
was fixed, since validateShapes returns on the first failure.
```

## Model.CheckGiwQuantMatch

Moved from `decoder/serialize.go` (the comment above `Model.CheckGiwQuantMatch`) on 2026-10-10.

```text
CheckGiwQuantMatch returns a startup error when an EXPLICIT weight-quant request cannot take
effect because the model is an already-baked prequant .giw whose quant differs. A .giw is
serialized at a fixed precision, so --quant is inert for it (Load ignores it); today it is
silently dropped, which the T1-7 report flagged. This surfaces the mismatch instead.

`requested` is the quant the user EXPLICITLY asked for — pass "" when they did not (relied on the
default). A bare default must NOT conflict: a .giw carries its own quant, and running it with
process defaults is the normal cross-format case (the caller, which alone knows whether the flag
was set, is responsible for passing "" then). For a non-.giw model (GGUF/safetensors), where
--quant IS honored at load, this is a no-op.

The comparison uses the corrected Quant() label (commit 9020160) — the resident weight kinds —
not the raw .giw header field. Message shape mirrors the safetensors int4mix decline
(weights.go): the constraint, the requested value, the baked value, and the file.
```

## Weights.quantLabel

Moved from `decoder/serialize.go` (the comment above `Weights.quantLabel`) on 2026-10-10.

```text
quantLabel names the precision of the resident matmul weights for display + the
KV-snapshot fingerprint, accounting for MIXED bundles. It scans the BODY matmuls — the
per-layer attention/FFN projections, experts, and routers, i.e. exactly what
`-quant int4|int4mix|int8int8` selects and what the batched-prefill gate inspects — and
returns "int4mix" only when int4 coexists with a higher-precision BODY weight. Pure
bundles collapse to int4 / int8int8 / int8 / native.

The token embedding and LM head are EXCLUDED. int4 mode pins them to int8 by DEFAULT
(logit-critical; the EmbedInt4 knob relaxes them), so their precision is orthogonal to
the int4-vs-int4mix distinction — a plain `-quant int4` bundle keeps an int8 head.
Including them made such a bundle mislabel as "int4mix" (audit/T1-6) even though every
projection is int4 and the prefill gate correctly batched it: the label contradicted the
path. Excluding them also sidesteps the older quantMode failure this comment used to cite
(that .giw header field derives from the FIRST weight — the int8 embed — so it reports
plain "int8" for the same all-int4-body bundle). Both mislabels have the same root: the
int8-pinned logit tables are not part of the quant the user chose.
```

## giwWriter.weightMatKind3Only

Moved from `decoder/serialize.go` (the comment above `giwWriter.weightMatKind3Only`) on 2026-10-10.

```text
weightMatKind3Only is weightMat for a tensor that must never take kind 5
regardless of target — it always writes kind 3 (or, under the legacy row4
opt-in, kind 4) for an int4 tensor. Two independent reasons land a call site
here, per docs/tasks/task-int4-layout-2026-09.md's L2:

  - MoE-paged experts (l.Experts[*], gemma4's mo.expertsGateUp/expertsDown): the
    doc's ground rule — decoder/moepaging.go reads these off the mmap with no
    load-time repack step, so the file must carry whatever layout the pager
    needs, chosen once at write time, not per-reader. (decoder/layerpaging.go's
    DENSE per-layer pager pages QProj/KProj/VProj/OProj/GateProj/UpProj/DownProj
    too, and already prefers WeightMat.MappedSpanRow4 over MappedSpan — so on
    inspection it does NOT need this exclusion; those stay kind-5-eligible via
    plain weightMat. Flagged as a finding in the L2 status line rather than
    silently narrowing the ground rule to MoE alone.)
  - Not yet scoped: KDA/DeltaNet/qattn mixer projections and gemma4's fused-MoE
    router (mo.routerProj). These are absent from Weights.matmulWeights(), which
    decoder.Load's post-load kind-5-vs-backend check walks (see repackedOnlyInt4Count)
    — routing them through plain weightMat would let a kind-5 instance of one of
    these slip past that check, relying solely on the (soft, logged-not-fatal)
    residency decline downstream. Kept kind-3-only until they get their own
    entry in that census, rather than widening the census for this cut.
```

## giwWriter.v6Layer

Moved from `decoder/serialize.go` (the comment above `giwWriter.v6Layer`) on 2026-10-10.

```text
v6Layer writes the state that made five families unrepresentable, in one unconditional tail.

WHY UNCONDITIONAL RATHER THAN ARCH-GATED like the gemma4 tail: every field here is empty on the
families that do not use it, so the cost is a handful of zero lengths per layer, and an
arch-gated tail is precisely how gpt-oss's sinks went missing — the gate is another place to
remember. A tail that always writes what the struct holds cannot be forgotten for the next
family, and TestSerializeCensus_noSilentFieldDrop checks that claim against the struct itself.
```

## giwWriter.v8Layer

Moved from `decoder/serialize.go` (the comment above `giwWriter.v8Layer`) on 2026-10-10.

```text
v8Layer writes the LFM2 gated short-convolution mixer: presence byte then the three tensors.

UNCONDITIONAL, LIKE THE v6 TAIL AND FOR THE SAME REASON. An arch-gated tail is how gpt-oss's
attention sinks went missing, and the cost here is one zero byte per layer on every other family.

This field existed for a whole family and serialize.go did not mention it once — `grep shortConv
decoder/serialize.go` returned zero matches. cmd/prequant loaded an LFM2 checkpoint, wrote every
field EXCEPT this one, appended a valid CRC, and selfCheck passed because selfCheck only Loads.
Serving the bundle, the first token reached conv layer 0 with lw.shortConv == nil and
nil-dereferenced in the decode goroutine (audit-2026-09-02 C-03, a regression of R3).

As with mamba, only the WEIGHTS are here — the rolling conv window is per-sequence state that
lives in the KVCache and is rebuilt at load.
```

## giwWriter.v9Layer

Moved from `decoder/serialize.go` (the comment above `giwWriter.v9Layer`) on 2026-10-10.

```text
v9Layer writes Bailing Hybrid's (Ling 3.0) per-layer v9 tail: MLA's optional attention-output
gate (l.mla.gProj — added to mlaWeights after v6Layer's MLA block already shipped, so it rides a
new version rather than retrofitting v6Layer's fixed byte layout, which would corrupt every
existing v6/v7/v8 file's read), then the KDA mixer (presence byte + the thirteen tensors,
kdaWeights' own field count — N-63, docs/audit-2026-09-10.md: this used to say nine) —
caught by TestSerializeCensus_noSilentFieldDrop the same way v8Layer's LFM2 gap was (R3/C-03):
l.kda existed and this file did not mention it once, so a round-tripped bailing_hybrid bundle
nil-dereferenced in kdaMixerStep on the first KDA layer.
```

## giwVersion

Moved from `decoder/serialize.go` (the trailing comments on `giwVersion` and `giwMinReadV`, and the stray `// v3: ...` line above `maxSerializedLayers`) on 2026-10-10.

```text
v15: every target — int4 kinds 3/4/5 store their group scales as binary16 (a u32 count + little-endian uint16 payload, 16-aligned like every array) instead of f32: aikit v1.50.0's in-RAM representation, so the reader aliases them as the WeightMat's storage (v12's scale aliasing, which converting at load had lost); kinds 6/7 keep their v14 layout and the reader takes their f16 block as the storage; a pre-v15 reader refuses the file via the version guard, a v15 reader converts an older file's f32 scales at load (the same rounding fresh quantization applies); v14: metal target only — every canonical group-32 int4 tensor also carries its group scales pre-converted to f16 (decoder.F16Bits, the kernels' own conversion), so a Metal no-copy buffer can alias them too: a kind-6 group gains an f16 block after its nibbles (members' scales back to back), and an eligible SINGLE int4 tensor is written as kind 7 (f32 scales, nibbles, f16 scales, each 16-aligned) — a distinct kind so a group's per-member fallback records can never be mistaken for group members; other targets still emit v12; v13: adds kind 6 — an int4 tensor whose nibbles live in a shared GROUP BLOCK after the group's headers, so the members' nibbles are ADJACENT in the file (a Metal fused QKV / gate|up buffer can then alias them, S6); written only for GIWTargetMetal, which is the only writer that emits version 13 (every other target still emits 12, so a pre-v13 reader keeps reading them); a pre-v13 reader refuses a v13 file via the version guard, a v13 reader still reads every older layout; v12: NO new kind — every weight-matrix payload array (int8 scales+codes, int4 scales+nibbles, row4 scales+row4 nibbles) is now preceded by zero padding so its bytes start 16-aligned relative to the blob start (giwAlignArray), which lets the reader ALIAS the group scales instead of copying them to the heap (docs/measurements/moe-pager-mode-darwin-2026-09-23.md, "Finding"); needs the v3 bundle header (blob at offset 64) to be aligned in the FILE; a pre-v12 reader refuses the file via the version guard, a v12 reader still reads every older layout; v11: no layout change to existing kinds — adds kind 5 (row4-only, docs/tasks/task-int4-layout-2026-09.md L2), gated on version so a pre-v11 reader refuses the file via the version guard rather than hitting an unknown kind byte; v10: a dense-granite bundle below it may hold llama.cpp-permuted q/k and is refused (audit C-05); v9: Bailing Hybrid's KDA mixer + MLA's optional attention-output gate — see the format comment above
read v3/v4 too (each version only ADDS: v4 the gemma4-gated tail, v5 the quant-label field, v7 kind 4, v8 shortConv, v9 KDA/MLA-gate, v11 kind 5; older bundles stay valid and fall back to inference)
```

## gguf.go: router and expert labels

Moved from `decoder/gguf.go` (the one-line comment repeated at four Router streamMat sites, and the one above the MoE experts' skipRow4 branch in stackedExperts) on 2026-10-10.

```text
Router stays f32 (M-27, docs/audit-2026-09-10.md) — see the Laguna site above.
M-07 (audit-metal-2026-09-12.md): MoE experts, same skip-row4-only scope as streamMat.
```
