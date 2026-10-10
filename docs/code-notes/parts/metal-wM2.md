# metal: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `metal`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

## metalBackend.memoryProbe

Moved from `metal/backend.go` (the comment in `init`, above the memory-probe registration) on 2026-10-10.

```text
residentFitsMemory's own budget, exposed to decoder.Model.Plan (docs/tasks/task-fit-to-hardware.md
Phase 1) via the SAME arithmetic — not a live "available" query, deliberately: darwin's UBC
reclaim makes "available" report what survived rather than what can be asked for (the
residentMemFraction comment above this file's own guard). Plan sees exactly the number the
real guard would judge it against, so the two can never disagree.
```

## metalBackend.BuildResident.guard

Moved from `metal/backend.go` (the comment above the memory-guard checks in `BuildResident`) on 2026-10-10.

```text
FITS-IN-MEMORY GUARD. Metal's unified memory IS host RAM, and it WIRES the mmap pages a
command buffer touches, so a model whose weights exceed RAM does not merely run slowly —
it pages until swap is exhausted. Measured 2026-08-31 on a 16 GB MacBook with gpt-oss-20b
(11.28 GB of weights): swap went to 35.98 GB of 36 GB, the process sat in uninterruptible
I/O wait with RSS creeping 1.8 -> 2.0 GB over 12 minutes, and the load NEVER completed and
never declined. Declining is strictly better than that, and there was no check at all: the
only size guard here caps the KV CONTEXT (checkCap, above), not the weights.

Keyed on two quantities WE compute — the model's own weight bytes and the machine's
physical RAM — never on the OS's account of what is free. Darwin's UBC reclaims under
pressure, so "available" reports what survived rather than what can be asked for; an
RSS-keyed ceiling once reported LESS memory at a known failure point than at baseline,
which is a guard that inverts exactly when it is needed.
G6 (docs/tasks/task-gpu-paths-2026-09.md — "honoured or refused with numbers"): an explicit -ctx
above metalCtxCapMax cannot be honoured (a fixed-size kernel score buffer, not a tunable
budget) and must be a NAMED refusal, not folded into the generic "BuildResident declined"
swallow below (buildResident itself also calls resolveMetalCtxCap and would hit the exact
same error, but by then it's just another opaque build failure) — mirrors CUDA's own
errKVWontFit precedent (cuda/backend.go): an operator's explicit request that cannot be
honoured gets a specific, propagated error naming the numbers, not a silent CPU fallback
indistinguishable from "this arch doesn't fit here at all".
```

## residentMemFraction

Moved from `metal/backend.go` (the comment above `residentMemFraction`) on 2026-10-10.

```text
residentMemFraction is the share of physical RAM the WEIGHTS alone may occupy. The remainder
is not slack: the KV cache, per-layer scratch, the command buffers, and the rest of the system
all live in the same unified memory.

0.70 is set from ONE measured failure (11.28 GB of 16 GB = 70.5% thrashed to swap exhaustion)
and is therefore a threshold, not a curve — it is honestly a single point, and a machine that
would in fact have fit can override with GOINFER_NO_RESIDENT_MEM_GUARD=1 rather than be told
no by a number nobody has swept. What it must not do is silently pass the case it was written
for, which is why the bar sits just below that measurement rather than at a rounder 0.75.
```

## metalMemoryCeiling

Moved from `metal/backend.go` (the comment above `metalMemoryCeiling`) on 2026-10-10.

```text
metalMemoryCeiling is S4 item 3 (task-never-swap-2026-09.md): the SINGLE arithmetic both
residentFitsMemory (the real load-time guard) and the registered "metal" memory probe
(decoder.Model.Plan's own view, above) now share, so the two can never disagree — the property
RegisterMemoryProbe's own comment already promised and, before this, did not fully keep: it
recomputed ram*residentMemFraction independently of residentFitsMemory's own copy of the same
formula, two copies that happened to agree only because neither had a second term yet.

Adds a SECOND bound from HostRAMAvailableBytes, min'd against the static 70% ceiling — so the
result can only get STRICTER than the static ceiling alone, never looser. This is deliberate:
darwin's UBC reclaim means "available" can look artificially GENEROUS under memory pressure (a
guard keyed on the live figure ALONE would invert exactly when it's needed — the same class of
mistake an RSS-keyed budget guard made elsewhere in this repo, reporting LESS memory at a known
failure point than at baseline), but min() can never make the COMBINED ceiling look artificially
SCARCE, because it only ever lowers what the static 70% already allowed — the one direction a
stricter-not-looser bound is safe to move in. live<=0 (no probe on this platform, or genuinely
unknown) leaves the static ceiling untouched, same as every other "unknown ⇒ proceed" case here.

ram==0 returns 0: every caller here already treats a zero budget from an unreadable hw.memsize
as "unknown", unchanged.
```

## metalMoESlotsRequest.cacheExperts

Moved from `metal/backend.go` (the comment inside `metalMoESlotsRequest`) on 2026-10-10.

```text
M-13 (audit-metal-2026-09-12.md): --moe-cache-experts alone (no explicit --moe-cache-slots)
used to read as "0 ⇒ unpaged" here, so a user following the CLI's own advice for a
bigger-than-RAM MoE (26B/35B/gpt-oss-20b) got every-expert-resident anyway, which the memory
guard then declined to the CPU-staged path the peer-matrix row actually measured. CUDA
already auto-caps to free VRAM in this exact shape (MoECacheExperts set, slots unset); this
is Metal's twin.
```

## residentKVBytes

Moved from `metal/backend.go` (the comment above `residentKVBytes`) on 2026-10-10.

```text
residentKVBytes is the resident KV cache's footprint: the resolved ctx cap × kvDim × 2 bytes
(f16 KV — the only path this backend ships; the f32 KV kernels exist but are compiled out,
metal/model.go's kvF32 hardcoded false since the Gemma "crater" traced to BOS K/V, not
precision) × 2 buffers (K and V), summed per layer since a per-layer geometry family (Gemma 4)
varies kvDim between local and global layers — a single model-level figure would misprice
whichever shape isn't the majority. This is the dominant "scratch" term M-02 named as missing;
the many small per-model buffers (r.mq/r.gu/r.logits/etc, metal/model.go) are each at most a
few hundred KB — elementwise or vocab/hidden-sized — and round to nothing beside it, the same
exemption ResidentWeightBytes' own doc comment already gives norms/biases.

Runs BEFORE buildResident (this is the pre-build guard), so there is no real *resident yet to
read ctxCap off of — resolveMetalCtxCap(m) is called independently here. An error (an explicit
ctx above metalCtxCapMax) is NOT this function's problem to report: residentFitsMemory's caller
(metalBackend.BuildResident) checks that specific case on its own, earlier and more clearly, so
this just falls back to metalCtxCapMax on error — a safe over-estimate for a guard whose whole
job is "don't under-count", never the thing that actually explains the refusal to the user.
```

## residentNeedBytes

Moved from `metal/backend.go` (the comment above `residentNeedBytes`) on 2026-10-10.

```text
residentNeedBytes is the byte count residentFitsMemory judges against — split out from
residentFitsMemory so it can be unit-tested directly, without real RAM or a checkpoint large
enough to swing the guard's verdict.

M-02: this used to always be m.ResidentWeightBytes() — the UNPAGED number — even when the
caller had set GOINFER_METAL_MOE_SLOTS, so a model that fits fine under paging (a few GB) was
declined on the number it would need with every expert resident (tens of GB for a large MoE).
Reading the same knob buildResident is about to honor and asking for the PAGED estimate instead
fixes the audit's Qwen3.5-35B-A3B example without moving the guard itself — it still runs
before buildResident, on the byte count that will actually apply once paging is resolved.

M-02 (continued, 2026-09-09): the weight term alone under-counted by ~2x on a real GGUF/
safetensors load (docs/audit-2026-09-02.md's "~9 GB Q4_K_M GGUF... lands at ~18 GB anonymous"
example) because Metal's unified memory holds the quantized HOST WeightMat AND a freshly
re-packed device buffer for the same weights (decoder.Model.ResidentHostCopyBytes' own doc
comment). Genuinely paged experts are exempt (they stream, no host copy), which is why this
asks for the host-copy addend at the SAME slot count rather than assuming it doubles the whole
weight term. KV was entirely absent; residentKVBytes above closes that.
```

## metalDefaultKVSlots

Moved from `metal/backend.go` (the comment above `metalDefaultKVSlots`) on 2026-10-10.

```text
metalDefaultKVSlots is how many resident KV slots Metal keeps when the count asked is the caller's default
(decoder.Options.ResidentKVSlotsDefault: serve without -kv-sessions), E-P09 of docs/audit-metal-2026-09-30.md. Each
slot's KV is resident from the first token on (T1.6: 4 slots cost 335 MB more than 1 on the 1.5B after one token,
though only slot 0 was written), so the default 4 cost about 224 MB on the 1.5B and 470 MB on the 7B over 2. Two
keep MC3's batched step (it needs 2 slots) and a second conversation's prefix; a third and fourth concurrent client
lose their own slot and their batching. -kv-sessions N still asks for N.
```

## residentMemoryDecline.moeHint

Moved from `metal/backend.go` (the comment above `moeHint` in `residentMemoryDecline`) on 2026-10-10.

```text
M-13: name the actual escape hatch for an MoE model, not just the guard override — a model
with routed experts that doesn't fit resident may still fit PAGED, and the decline line
used to say nothing about how to reach that path.
```

## metalResident.checkCap

Moved from `metal/backend.go` (the comment above `metalResident.checkCap`) on 2026-10-10.

```text
checkCap guards the resident KV allocation (C3). Every layer's cache is r.kc[l]/r.vc[l], sized
ctxCap()*kvDim, so kv_store writes absolute position p at kc[p*kvDim ...]; valid positions
are [0, ctxCap()). Writing past it is an out-of-bounds device write — on Metal's UNIFIED
memory that silently corrupts adjacent MTLBuffers (other models' resident weights), and once
nKeys > 4096 the attention kernel's `threadgroup float sc[4096]` overflows too. The decode loop
increments pos unbounded (a ≤cap prompt + a large max_tokens is enough), so refuse here; the
decode loop surfaces the error (model.go) and the caller can fall back to the staged path.
```

## metalResident.ForwardNoLogits

Moved from `metal/backend.go` (the comment above `metalResident.ForwardNoLogits`) on 2026-10-10.

```text
ForwardNoLogits (decoder.ResidentPrefillKV) runs the token's forward to build ONLY its
resident K/V — skipping the final norm's LM-head dispatch, the ~1 MB logits readback, and any
softcap. residentPrefillSeed calls this for every prompt token but the last (audit-
metal-2026-09-12.md M-01): before this existed every sequential-prefill token on Metal paid the
full int8 LM head + a 608 KB readback for logits nobody read (233 MB/token on a 1.5B, 671 MB on
a Gemma-class vocabulary). The layer chain — hence the K/V written at pos — is identical to
Forward, so decode from the last prompt token is byte-identical.

Pipelined through the encode-ahead executor (ForwardEmbNoLogitsPipe) via a noHead bit on execJob:
overlaps token t+1's trunk encode with token t's GPU execution while skipping the LM head
dispatch and logits readback (~0.9 ms/token prefill latency recovery).

On a paged MoE (g4moe or generic moe) the executor's trunk encoder has no paged branch, so
ForwardEmbNoLogitsPipe runs the synchronous paged forward with its final norm and LM head skipped
(pagedNoHead): the same layers and K/V, without the head (task-m26-mac-2026-10.md).
```

## metalFastPrefillFloor

Moved from `metal/backend.go` (the comment above `metalFastPrefillFloor`) on 2026-10-10.

```text
metalFastPrefillFloor is the PROMPT-LENGTH floor (whole prompt = startPos+M) below which
the f16-MMA batched pass declines and the sequential per-token loop runs instead. Lowered to
64 (2026-09-20, R3, docs/tasks/red-october.md and docs/measurements/metal-prefill-floor-
2026-09-20.md): the §3.2 pooled gate SHIPS at K=64 alone (critA/B/C all true, S model) and at
K=64+128 pooled together, and the batched arm beats sequential on served TTFT by 3.83x at
K=64 and 4.66x at K=128 (bench_peer_prefill.py, three arms interleaved, n=6) — both well past
the >=2x ships band, not just the fidelity precondition. K=64 is the smaller of the two
registered candidates {64, 128}, so it is the new floor per the brief's own decision rule.
Previously lowered to 256 (2026-09-12, audit-metal-2026-09-12.md M-02).

16 since A-P02 (docs/tasks/task-metal-audit-2026-10.md, graded 2026-10-04 against its pre-registration): the §3.2 pooled
gate SHIPS on the 1.5B with K = 16, 32, 48 and 64 pooled, and on each new cell alone, against CPU f32 references built
together; the pass beats the sequential loop 3.48x at K = 16 in-process (7 of 7 reps above the 1.5 bar; the 7B 2.52x).
GOINFER_METAL_FAST_PREFILL_FLOOR overrides it, 0 = no floor. A resident with the batched step keeps the exact step
below metalStepPrefillCeiling instead (promptStepOK).
```

## metalStepPrefillCeiling

Moved from `metal/backend.go` (the comment above `metalStepPrefillCeiling`) on 2026-10-10.

```text
metalStepPrefillCeiling (A-P02's step bound): a resident with the batched step runs a prompt that ends below this many
tokens as exact decode rows on the step (E-P01) rather than the pass. The step is bit-identical to the sequential
loop and the pass is not, so it keeps every length where it is not slower: the 1.5B measured step / pass 0.755 at
K = 16 (the step faster in 7 of 7 reps) and 1.485 at K = 32 (the pass faster in 7 of 7), the 7B 0.627 and 1.243.
Held at 64 for a morning (2026-10-04) on what was read as an MC3 identity defect and was a warm prompt taking a
different route from the cold one (fixed: decoder.PrefillTailExact); released to the graded 32 the same day.
```

## metalFastPrefillEnabled

Moved from `metal/backend.go` (the comment above `metalFastPrefillEnabled`) on 2026-10-10.

```text
metalFastPrefillEnabled reports whether the batched f16-MMA prefill path is selected.

Default ON above metalFastPrefillFloor (16 tokens since A-P02; 64 from R3) since §3.2 gate (TestPrefillGateVsReference)
passed 2026-09-20 (S model, K=64/128 pooled and K=64 alone; see docs/measurements/metal-
prefill-floor-2026-09-20.md). GOINFER_METAL_FAST_PREFILL=0/false/off or
--exact-prefill to opt out.

	GOINFER_METAL_FAST_PREFILL  1 | true | on   on (even below the floor — for tests)
	                            0 | false | off  off (explicit opt-out; use --exact-prefill on the server)

The old GOINFER_METAL_BATCHED_PREFILL continues to work: =1 forces on, =0 forces off, unset defers to the default.
```

## metalAttnFAEnabled

Moved from `metal/backend.go` (the comment above `metalAttnFAEnabled`) on 2026-10-10.

```text
metalAttnFAEnabled reports whether decode attention defaults to attention_fa (R2,
docs/tasks/red-october.md) — the kvHead×split-gridded kernel, gated per layer by
canUseAttnFA (dense-GQA, hd=128, key count >= attnFADepthFloor). DEFAULT ON since
2026-09-21. The fidelity pass that admitted it (gate (3), docs/measurements/r2-attn-fa-rootcause-2026-09-21.md)
is VOID: its arms were contaminated by the executor's stale pre-encoded buffer and 4 of its 10 set-A references
did not match their prompts. Re-gated 2026-09-25 on set B under the owner's amendment
(docs/measurements/metal-decode-attn-fidelity-setb-PREREGISTERED.md): PASSES, with kernel error vs float64 ~3x below
the shipped kernel's (docs/measurements/metal-decode-attn-r17-2026-09-25.md). Since 2026-09-25 the first pass is the
R17 block kernel attention_fa_blk for GQA group sizes 6 and 7 (buildResident). The divergence that parked the Build
attempt was an end-to-end-logits instrument crossing one int8 rounding boundary, not a kernel defect,
and it is a real, deterministic 1.11-1.19x at depth (docs/measurements/r2-attn-fa-
speed-2026-09-21.md) — KILLED on the brief's own peer-parity band (needs >=60 tok/s at
depth 4000, measured 44.9), shipped anyway by owner decision as an incremental win. NOT
bit-identical to the shipped `attention` kernel (reduction/combine order differs by
design), so this moves argmax at the margin on some inputs (gate (3)'s own hard-flip
counts) and is a SECOND source of decode/ForwardN divergence on top of the pre-existing
one docs/spec/08-dspark-dflash.md already names (PrefillLast's f16-MMA activations) —
Metal spec-decode verify (P10) was already not a legal oracle for that reason and stays
so; nothing here newly breaks a path that was shippable before.

	GOINFER_METAL_ATTN_FA  1 | true | on    on (explicit; harmless, matches the default)
	                       0 | false | off  off (opt out; the shipped `attention` kernel only)
```

## metalFusedAttentionEnabled

Moved from `metal/backend.go` (the comment above `metalFusedAttentionEnabled`) on 2026-10-10.

```text
metalFusedAttentionEnabled reports whether attention_prefill_fused (the simdgroup_matrix
flash-attention twin of attention_prefill, L2-Metal — docs/completed/task-prefill-gap.md §4) runs in
place of the exact scalar kernel. Default ON since §3 gate passed 2026-09-10 (S model, set B
decision cells K=256/512/1024 + K=3900 confirm; fused beat exact on all three pooled criteria —
see docs/measurements/prefill-l2-metal-fused-attn-2026-09-09.md §5). GOINFER_METAL_FUSED_ATTENTION=0
or --exact-prefill (which also covers metalFastPrefillEnabled) opts back to the exact kernel.
attention_prefill_fused also requires hd%8==0 && hd<=128 (ATTN_MAXHD in metal/prefill.go) —
PrefillLast falls back to the exact kernel outside that range regardless of this flag.

	GOINFER_METAL_FUSED_ATTENTION  1 | true | on   on
	                               0 | false | off  off (explicit opt-out; use --exact-prefill on the server)
```

## metalResident.PrefillPath

Moved from `metal/backend.go` (the comment above `ResidentQuant`, where it was glued to the wrong declaration) on 2026-10-10.

```text
PrefillPath (decoder.PrefillPathReporter) reports at load time whether this resident will use
the batched f16-MMA path. Default ON above metalFastPrefillFloor (16 tokens since A-P02, 64 from R3) since §3.2
gate passed 2026-09-20. The floor applies per-call; PrefillPath reports true iff the enabled
state AND arch both allow batching.
ResidentQuant (decoder.ResidentQuantReporter) is the precision this resident runs its weights at, when the decoder
cannot infer it: an int8 model on the native path (r.w8) runs at its own quant, where one on the int4 path is
re-quantized. "" leaves the decoder's label.
```

## metalResident.PrefillLast.cancel

Moved from `metal/backend.go` (the comment inside `PrefillLast`, above its context check) on 2026-10-10.

```text
One pass, so one check: this backend ingests the prompt in a single command buffer and has no
inner loop to interrupt, so checking at entry is the only granularity there is. The decoder's
chunked prefill (serve -prefill-chunk) cuts a prompt only while another conversation is
decoding, so a lone long prompt is still one pass that cannot be cancelled midway; that needs
chunking inside PrefillLast, as cuda's prefillChunked does (A-D02, audit-metal-2026-09-30.md).
```

## metalResident.PrefillLast.default

Moved from `metal/backend.go` (the comment inside `PrefillLast`, above the fast-prefill check) on 2026-10-10.

```text
Default on since the §3.2 gate passed on 2026-09-09 (S cells K=256/512/1024), above metalFastPrefillFloor: 16
tokens since A-P02 (cells K=16..64), 64 from R3, 256 before. GOINFER_METAL_FAST_PREFILL=0 or --exact-prefill opts
out.
```

## resident.prefillUnaligned

Moved from `metal/backend.go` (the comment above `resident.prefillUnaligned`) on 2026-10-10.

```text
prefillUnaligned names the first reduction length the batched pass's GEMMs cannot take, or "" when every one is a multiple
of 32. gemm_w4f16_tile and its int8 twin stage K in slabs of 32 with no partial-slab handling (metal/prefill.go). An int4
resident cannot violate it (int4Buf refuses K%32 != 0 at build), but a native int8 one can: glm-ocr-tiny (hidden 48, FFN
144) was admitted and read cosine 0.22 against the CPU (docs/tasks/task-metal-pairwise-followups-2026-10.md, Part A).
Real checkpoints are 32-aligned; this keeps a model that is not on the sequential path instead of computing garbage.
```

## metalMRoPEPrefillOn

Moved from `metal/backend.go` (the comment above `metalMRoPEPrefillOn`) on 2026-10-10.

```text
metalMRoPEPrefillOn and metalDeepstackPrefillOn turn S16's resident m-RoPE prefill on, for a turn without DeepStack sets
(Qwen2.5-VL; Qwen3.5+ is not claimed) and with them (Qwen3-VL). Off, an image turn takes the CPU prefill and the upload.
  - DeepStack (Qwen3-VL): ON since 2026-10-09 (owner). G-S16c real PASS through its sidecar on all four images over healthy
    text controls (0.87-0.95), the served comparison IDENTICAL, image-turn TTFT 8.25 -> 2.86 s (2.9x).
  - Without DeepStack (Qwen2.5-VL): ON since 2026-10-09 (owner). G-S16c and the served comparison passed (19.46 -> 5.73
    s). It was held for a text control that fell to -0.36, but that control's reference was W4A8, the less precise arm
    (docs/completed/task-metal-prefill-precision-2026-10.md). Against non-W4A8 references the image turns read even
    with the CPU prefill (mean 0.948 against weight-only int8) and closer to the same weights (0.990 against 0.977).

Tests set them.
```

## metalResident.HiddenLast

Moved from `metal/backend.go` (the comment above `metalResident.HiddenLast`) on 2026-10-10.

```text
HiddenLast (decoder.ResidentHiddenLast) ingests a whole sequence starting at startPos and
returns the LAST position's hidden state after the model's final norm — the resident twin of
PrefillLast, but for embedding requests (G4, docs/tasks/task-gpu-paths-2026-09.md) instead of
generation: it never runs the LM head. This runs the SAME per-token sequential kernels decode
uses — one forwardHiddenNoHead call per position. It does not match the CPU reference bit for
bit: the decode kernels round differently and forwardHiddenNoHead returns the int8-dequantized
activation. Measured against the CPU: cosine about 0.9991-0.9993 on gpt2
(hiddenlast_resident_parity_test.go, bar 0.998), and 0.99985 int4 against int4 on qwen3_5-tiny
with its adapter merged, 0.99998 without (prompthidden_resident_parity_test.go; F-D03,
audit-metal-2026-09-30.md). The cost is one
command-buffer submit per token (≈K × 13-18ms) instead of a single batched pass (≈1.8s for K=512).

N-25 (audit-metal-2026-09-12.md): unlike when this doc comment was first written, Metal's
batched (f16-MMA) PrefillLast is NOT declined by default for generation anymore —
metalFastPrefillEnabled() defaults true (§3.2 gate passed 2026-09-09) — so the fidelity bar
that justifies this sequential loop's cost for embeddings is already accepted for decode's own
output. The audit's suggested fix (PrefillLast's dispatch graph minus its last two dispatches —
the LM head + softcap — reused here) is a real, scoped lever, not a design question, but it is
its own parity-gated engineering task (a HiddenLast-batched path needs the same kind of S-cell
tolerance gate PrefillLast itself passed, verified against THIS function as the oracle) rather
than a same-sitting fix; left as follow-up work.
```

## metalResident.PrefillTailExact

Moved from `metal/backend.go` (the comment above `metalResident.PrefillTailExact`) on 2026-10-10.

```text
PrefillTailExact (decoder.PrefillTailExact): a short PrefillLast continuing a prefix the pass prefilled reproduces the
cold pass bit for bit (TestPrefillLast_tailContinuationMatchesCold), and the routing (floor, step range) is on the
whole prompt's length. Proven for the dense and Gated-DeltaNet passes (TestMC5_prefillChunkInvariance,
TestDB01_chunkedPrefillMatchesWhole) and, since 2026-10-04, for the generic resident MoE's expert-major pass
(TestPrefillLast_tailContinuationMatchesCold_MoE: Mixtral, two Qwen3-MoE shapes and the Qwen3.5 MoE hybrid, tails of
1-33 rows). A paged Gemma 4 MoE's layer-major path is decode's kernels, so exact by construction; every other MoE shape
(paged generic, resident Gemma 4, gpt-oss) declines the pass, and its suffix runs the sequential loop as before.
```

## allKernels

Moved from `metal/kernels.go` (the comment above `allKernels`) on 2026-10-10.

```text
allKernels is the full dense-decode-layer MSL kernel set in one library (W8A8 path —
W4A8 is validated separately; this proves ASSEMBLY, not the int4 packing again).

N-17 (audit-metal-2026-09-12.md): six of these kernels have no PRODUCTION pipeline —
gemv_w4a8_bias, gemv_w4a8_sa_amax, gemv_w4a8_sa_bk, gemv_w4a8_sa_qv, gemv_w8a8, rope2_kv — each
backs a dedicated micro-benchmark or recorded-negative regression test instead (profile_test.go,
batchk_test.go, sa_qv_fusion_test.go, gemv_test.go, rope2_kv_test.go respectively;
gemv_w4a8_sa_amax has no reference anywhere and is the one genuinely dead survivor of this
list — kept rather than deleted alongside it so its own history stays visible next to the
others, not because anything still needs it). None of these is "safe to delete because nothing
production calls it" — deleting one breaks the test that keeps its measurement/negative result
honest. A seventh, gemm_w4f16 (metal/prefill.go, a genuinely dead duplicate of
gemm_w4f16_store with no reference anywhere, test included), was deleted outright.
```

## pagedProfile

Moved from `metal/gemma4_moe.go` (the comment above `pagedProfile`) on 2026-10-10.

```text
pagedProfile decomposes the paged forward's per-token cost into ACTUAL GPU-busy time (per phase,
from the command buffer's GPU timestamps) vs wall time (GPU + submit/wait/encode coordination) vs
staging (the ensureResident pread body). It accumulates across calls; snapshot with PagedProfile()
and diff over a timed window. This is the DIRECT decomposition of "compute+coord" — not
total-minus-staging — so the ~70%-of-budget bucket is split into GPU execution vs host coordination.
```

## gemma4MoeKernels

Moved from `metal/gemma4_moe.go` (the comment above `gemma4MoeKernels`) on 2026-10-10.

```text
Gemma-4 enable_moe_block (26B-A4B) MSL kernels — the parallel dense‖MoE FFN that the generic
moe.go path (Mixtral/Qwen/GLM shape) cannot express. Kept in their OWN file/const, concatenated
after moeKernels, so moe.go's audited MoE kernels are not touched — the same discipline as CUDA's
separate router_f32.cu (which exists so adding a kernel doesn't rewrite the audited moe.ptx).

The router-first step (9c Step 5a) added gemv_f32_f32. This block adds the remaining primitives
the parallel dense‖MoE forward needs — each verified in isolation against a CPU oracle before the
composition is wired (gemma4_moekernels_test.go), the same 2b→2c→2d order CUDA followed.
```

## gemma4MoeResident.paged

Moved from `metal/gemma4_moe.go` (the comment above `gemma4MoeResident.paged`) on 2026-10-10.

```text
Synchronous paging (GOINFER_METAL_MOE_SLOTS=N>0): the full expert set doesn't fit resident, so
each layer keeps N experts in a slot pool and stages the routed top-k in per token. Off (all
experts resident) when slots==0. slotIdx is a [topK] device buffer, HOST-WRITTEN each token
with the POOL SLOT NUMBER holding each of the k routed experts (M-11, audit-metal-2026-09-12.md
— was idxZeros, always 0, back when each slot was its own single-expert Buffer object; now the
pool's storage is one contiguous per-field buffer, so which physical row holds a given expert
varies token to token and must be told to the kernel). The reused gemv_w4a8_moe(_wacc) kernels
read row slotIdx[slot] of the pool's contiguous buffer while still indexing rWgt by the
selection slot uSlot[j] — makes paged dispatch byte-identical to the stacked path (a slot's
bytes == the stacked buffer's rows for that expert). Fixed Buffer identity for the resident's
whole lifetime — only its CONTENTS change — which is what lets phase 2's encode be
value-independent (the actual point of this change: see expertpool.go's own doc comment).
```

## buildGemma4MoE.slots

Moved from `metal/gemma4_moe.go` (the comment above the slot request in `buildGemma4MoE`) on 2026-10-10.

```text
Synchronous paging: --moe-cache-slots / GOINFER_METAL_MOE_SLOTS (deprecated fallback,
metalMoESlotsRequest) keeps only N experts/layer resident and stages the routed top-k in
per token (the only way the 26B's 11.96 GB expert set runs on a 16 GB Mac). N must be >=
topK (a token's own top-k must fit). N==0 / unset ⇒ all experts resident (the fitting
path + the paged≡non-paged parity reference). slotIdx (M-11) is host-written each token with
the pool slot holding each routed expert, so the paged expert GEMVs read the RIGHT row of the
pool's contiguous buffer while rWgt is still indexed by the selection slot (byte-identical).
```

## buildGemma4MoE.pread

Moved from `metal/gemma4_moe.go` (the comment above the pread-staging setup in `buildGemma4MoE`) on 2026-10-10.

```text
Stage experts by pread'ing their nibbles straight into the slot buffers instead of a byte-copy
off the mmap. DEFAULT ON (cold A/B verdict: total 1892→1488 ms/tok, 0.53→0.67 tok/s, 1.26×;
major faults 92.8→0.0/stage — the mmap demand-fault page-in is gone). Needs a .giw-mmap'd model
(offsets are into that file); re-open it once, shared across layers. GOINFER_MOE_PREAD=0 opts out
(the mmap byte-copy baseline, kept for the A/B). If open fails or the model isn't .giw-backed,
buildGemma4MoELayer falls back to the byte-copy.
```

## buildGemma4MoE.nocache

Moved from `metal/gemma4_moe.go` (the comment above the `GOINFER_MOE_NOCACHE` check in `buildGemma4MoE`) on 2026-10-10.

```text
GOINFER_MOE_NOCACHE=1: F_NOCACHE on the pread fd — reads bypass the unified buffer
cache. MEASURED AND DECLINED (cold A/B): NO effect — total 1535.3→1514.0 ms/tok
(−1.4%, noise), and RSS-after-build (3306→5095) and compute+coord (1032→1056) both
moved the WRONG way. The motivating cache-pressure hypothesis is refuted, AND its
evidence was a CONFOUND: the ~+2 GB RSS-after-build growth lands on whichever arm runs
SECOND, and RSS-after-build is sampled BEFORE the timed decode, so the pread cache flag
cannot cause it (ordering artifact). The pread win's +365 ms compute+coord displacement
remains UNEXPLAINED. Kept off by default, wired, so it isn't re-proposed.
```

## buildGemma4MoELayer.paged

Moved from `metal/gemma4_moe.go` (the comment above the paged branch of `buildGemma4MoELayer`) on 2026-10-10.

```text
Paged: DON'T stack the experts (that is the 11.96 GB we can't afford). Build a bounded LRU
slot pool + a stage fn that reads expert e's W4A8 bytes straight from the bundle's mmap-backed
WeightMats on demand (int4DirectWords aliases the .giw). Per-expert buffer sizes come from
expert 0. The .giw expert weights MUST be int4-direct (group-32) for this zero-intermediate path.
```

## buildGemma4MoELayer.scales

Moved from `metal/gemma4_moe.go` (the comment above the `stage` closure in `buildGemma4MoELayer`) on 2026-10-10.

```text
C-P01 (audit-metal-2026-09-30.md): each expert's f16 scales are read from its own WeightMat
(Int4ScalesF16), which a v14 metal or v15 .giw aliases from the mapping. This used to be a
build-time heap cache (N-20, which replaced a per-page-in f32→f16 conversion), holding the
same bits again: 1361 MB on the 26B, measured 2026-10-02 (T1.8).
```

## buildGemma4MoELayer.willneed

Moved from `metal/gemma4_moe.go` (the comment above the pread-staging block in `buildGemma4MoELayer`) on 2026-10-10.

```text
MADV_WILLNEED over the routed experts' nibble spans before staging (was GOINFER_MOE_WILLNEED=1;
the switch was REMOVED 2026-09-24 — this record is what stays). MEASURED AND DECLINED (cold A/B, sudo purge between arms, 2026-08-03): it does NOT
help. major faults/stage ROSE 92.5→147.1 (readahead read MORE from disk, did not batch the
serial faults), and the staging-line drop (−775 ms) was an ATTRIBUTION SHIFT into the residual
compute+coord bucket (+700 ms), for a total move of 1851.8→1776.6 ms/tok (~4%, noise band).
WILLNEED targets fault LATENCY/overlap, but the bound here is fault COUNT, and darwin
MADV_WILLNEED on a span you touch immediately pays the read synchronously in the Advise call —
the synchronous per-layer Advise→copy path has no lead time to hide it behind (the routed
experts aren't known until phase-1 completes: the value-dependent seam). The extra faults are
consistent with prefetched pages being evicted under memory pressure before use (RSS fell
3369→2870 MB). It stayed wired, off, for a while so nobody re-proposed it without re-reading this;
the comment now does that job on its own.
Lesson: when you optimize a measured sub-bucket, the TOTAL is the gate — a residual bucket
absorbs displaced cost silently. See [[optimize-sub-bucket-total-is-the-gate]].
pread staging (GOINFER_MOE_PREAD=1): resolve each expert's nibble file offset within the .giw
mmap (pure pointer arithmetic, no page touch), then stage by pread'ing straight into the slot's
UMA words — zero mmap faults, one big sequential read per expert. The f16 scales are pread the same way
when every expert's lie in the mapping (a v14 metal or v15 .giw, C-P01), and copied from the WeightMat
otherwise (an older file's scales are converted onto the heap at load). Falls back to the
byte-copy path if the fd is absent or ANY expert's nibbles aren't .giw-mmap-backed (e.g. a
requantized HF load) — the offsets must all resolve for pread to be correct.
```

## buildGemma4MoELayer.stagePread

Moved from `metal/gemma4_moe.go` (the comment above the `stagePread` closure body in `buildGemma4MoELayer`) on 2026-10-10.

```text
M-12 (audit-metal-2026-09-12.md), second half: gate|up and down are separate
buffers (pool.guW, pool.dW) — disjoint destinations, safe to pread concurrently
(same argument as moe.go's 3-way twin). Errors panic from THIS goroutine, not
the spawned ones, so BuildResident's recover() still sees them.

M-11: preadIntoPoolSlot, not preadIntoU32Buf(fd, s.guW, ...) — s.guW is a
Buffer.At()-offset VIEW, and preadIntoU32Buf's U32s() call ignores that offset
(expertSlot's doc comment), which would silently pread every expert into slot
0. preadIntoPoolSlot addresses the pool's base buffer by slot NUMBER instead.
```

## encodeGemma4MoEFFN

Moved from `metal/gemma4_moe.go` (the comment above `encodeGemma4MoEFFN`) on 2026-10-10.

```text
encodeGemma4MoEFFN records Gemma-4's parallel dense‖MoE FFN for one layer, replacing the dense
gate/up/swiglu/down block. Unlike the generic encodeMoEFFN (one branch, wacc straight into the
residual), TWO branches run off the SAME post-attention residual h through THREE independent
normalizations, then join under a shared post-norm with a per-layer scalar. Mirrors
decoder/forward_gemma4_moe.go and cuda/resident.go gemma4MoeMLP exactly:

	x1 = postFFN1( mlpDown( geluTanh(mlpGate·xd)·(mlpUp·xd) ) )   xd = preFFN(h)    [dense]
	rn = rmsnorm_nw(h); logits = RouterProjScaled·rn; idx,wgt = route; wgt *= perExpertScale[idx]
	x2 = postFFN2( Σ_j wgt[j]·expertDown_j( geluTanh(gu_j)·up_j ) ) xe = preFFN2(h) [MoE]
	h  = (h + postFFN(x1 + x2)) · layerScalar                                        [join]

h (r.x) is read THREE times and written only at the very end. The dispatch sequence is
value-independent (the top-k loop count is the model constant topK; each expert GEMV reads its
own rIdx slot at execution time), so the command buffer is static every token and the encode-
ahead executor still pre-encodes token t+1 while t runs (task-metal-moe.md).
This is the NON-PAGED path: encodeG4Phase2NonPaged reads the stacked all-E buffers
(ml.expGuW/expGuS/expDW/expDS), which stay zero-value once the layer is paged. forwardLogitsPaged
never reaches here for a paged layer (it tears the layer into encodeG4Phase1 + encodeG4Phase2Paged
around a host readback instead) — the panic below is a chokepoint against every OTHER caller of
encodeLayer (Forward, ForwardArgmax, forwardHiddenNoHead's encodeTrunkInto) reaching a paged layer
through the non-paged encoder and silently computing off zero-value weights instead of failing
(audit-metal-2026-09-12.md C-02).

FinishEncoding before the panic — see encodeMoEFFN's twin comment (metal/moe.go): e already has
this layer's attention dispatches recorded, and Metal asserts on a command encoder released
without endEncoding.
```

## encodeG4Phase1

Moved from `metal/gemma4_moe.go` (the comment above `encodeG4Phase1`) on 2026-10-10.

```text
encodeG4Phase1 is the value-INDEPENDENT head of the FFN: the dense branch (→g4x1) and the router
(→rIdx/rWgt device buffers), ending with the expert-branch input quant (preFFN2(h) → mq/mSc). In
the paged forward this is the first command buffer; the host then reads rIdx and stages the routed
experts before phase 2. Byte-identical to the old inline head (same dispatches, same order).
```

## forwardLogitsPaged

Moved from `metal/gemma4_moe.go` (the comment above `forwardLogitsPaged`) on 2026-10-10.

```text
forwardLogitsPaged is the SYNCHRONOUS expert-paging decode: the reference implementation that lets
the 26B (11.96 GB experts) run on a 16 GB Mac. Per layer, dense layers encode in one command
buffer; a paged Gemma-4 MoE layer is torn at the router (the value-dependent seam Step-0 priced at
+43%): [attention + dense + router] → submit+wait → read rIdx → stage the routed top-k into the
layer's LRU slot pool → [experts-from-slots + join] → submit+wait. Assumes the caller filled r.x
with the embedding and holds the OS thread (ForwardEmb does both).
```

## forwardLogitsPaged.recover

Moved from `metal/gemma4_moe.go` (the comment above the deferred recover in `forwardLogitsPaged`) on 2026-10-10.

```text
The pread staging closure (stagePread) and the per-token MustBuf allocations panic on a
transient .giw read error or OOM — deep inside expertPool.ensureResident, at DECODE time, where
no recover otherwise exists (buildResident's is build-scoped). A single external-volume hiccup on
the flagship 26B paged path would kill the whole server mid-token. Convert the panic to execErr
so the metalResident adapter surfaces a failed request and drops the stale logits, rather than
crashing the process (audit R-02; same runtime-abort discipline as C-09).
```

## forwardLogitsPaged.abort

Moved from `metal/gemma4_moe.go` (the comment inside the deferred recover in `forwardLogitsPaged`) on 2026-10-10.

```text
Accepted bounded side-effects on this rare abort path (audit F-06): the in-flight
encoder is left un-ended (a small native cb/pool leak) and a slot's stale slotExpert tag
may cost one spurious re-stage on a later eviction. Both are perf-only and bounded by the
abort count. We deliberately do NOT drain/End the live encoder here: the path mixes
Begin() (own autorelease pool) and BeginNP() (nil pool, shared arp), so a blind
DrainPool/End would nil-panic or commit a half-encoded command buffer — worse than the leak.
```

## forwardLogitsPaged.split

Moved from `metal/gemma4_moe.go` (the comment above the `GOINFER_MOE_PROF_SPLIT` read in `forwardLogitsPaged`) on 2026-10-10.

```text
GOINFER_MOE_PROF_SPLIT: split each End() into commit() vs waitUntilCompleted() to locate the
~15 ms/boundary overhead (vs Step-0's 0.213 ms). Needs BeginNP (no per-call autorelease pool) +
one long-lived pool drained per token, so the sub-steps can be called individually and the drain
isn't per-boundary — which ALSO discriminates mechanism 2 (per-CB autorelease churn): if this
path's per-boundary cost is much lower than the default End() path, the per-CB pool was the cost.
```

## g4PrefetchOn

Moved from `metal/gemma4_moe.go` (the comment above `g4PrefetchOn`) on 2026-10-10.

```text
g4PrefetchOn guesses a paged Gemma 4 MoE layer's experts from the layer before it and reads the misses in the
background (lever 3, docs/tasks/task-m26-mac-2026-10.md): phase 1 of layer l also runs layer l+1's router on layer l's
residual, the host starts reading those experts into layer l+1's pool, and the reads overlap the GPU's phase 2 of l and
phase 1 of l+1. The true router still picks; a wrong guess costs an evicted slot, never a wrong bit. OFF: on M26 at
24 slots it read 0.894x a token guessing all 8 experts (43% of guesses used) and 0.947x guessing the top 2 (60%), 0
of 9 reps above 1 each: the extra reads compete with demand reads on the SSD (docs/tasks/task-m26-mac-2026-10.md,
"Lever 3"). Kept, gated bit-exact, for a better guess.
```

## weightAlias

Moved from `metal/alias.go` (the comment above `weightAlias`) on 2026-10-10.

```text
S6 (docs/tasks/task-never-swap-2026-09.md): a dense int4 weight is copied twice at build — the
.giw's mapping → a heap []uint32 → a StorageModeShared MTLBuffer — and the MTLBuffer copy is the
~2.5 GB IOAccelerator dirty term of a Metal M26 load. weightAlias replaces the second copy for a
canonical int4 array that lives in the mapping: it wraps the array's pages in a no-copy MTLBuffer
(newBufferWithBytesNoCopy) and binds the array at its offset inside that window, so the GPU reads the
nibbles straight out of the page cache. Nothing else changes: same bytes, same kernels, same bind
order — only who owns the memory.

One buffer per tensor, over the page-aligned window that encloses it (Model.MmapAliasWindow), rather
than one over the whole mapping: a 16 GB .giw exceeds a 16 GB Mac's MTLBuffer size limit, and a
window per tensor needs no page padding in the file (v12 already 16-byte-aligns every array, which is
all the kernels' vector loads need). Scales alias too when the file carries them as f16 (weights format
v14, -target metal); an older file stores only f32, so its scales are converted into a small buffer
(~1/8 of the nibble bytes) and the banner says so.

WHAT THE ALIASED PAGES ARE (measured, docs/measurements/s6-alias-2026-09-24.md, "MAP_SHARED"): IOKit wires a
no-copy buffer's pages with write intent. Over a MAP_PRIVATE mapping that makes every GPU-read page a wired
ANONYMOUS copy-on-write copy — the 1.5B served one request with +40,430 COW faults for 625 MB aliased — so
aliasing saved nothing, and fork() then copied the whole mapping (the M26 collapse). decoder.Load therefore
maps a .giw MAP_SHARED on darwin (decoder/giwmap_darwin.go): the GPU reads the file's own page-cache pages
(+132 COW faults, background), wired only while a command buffer uses them, and a fork is cheap.

ON BY DEFAULT since 2026-09-24, on the owner's decision after S6's gates (docs/measurements/s6-alias-2026-09-24.md):
logits byte-identical on the 1.5B/7B and all 23 resident fixtures, decode within 3% at depth 128 and 2048, and
Close ordering. Not every gate read as passed: the footprint gate read NOT MET, and when its dense-term condition
was later met the process was still about 60 MB over the bar read literally; the memory-hog arm ran on the 7B
only, not the M26; and M26 decode was not resolvable at the record's n (F-D02, audit-metal-2026-09-30.md). GOINFER_METAL_ALIAS=0 turns it off: this type is
then nil and int4Buf is byte-for-byte the copy path.
```

## newWeightAlias

Moved from `metal/alias.go` (the comment above `newWeightAlias`) on 2026-10-10.

```text
newWeightAlias returns an aliaser for any model backed by a .giw mapping unless GOINFER_METAL_ALIAS=0.
Any other value — unset, "1", or "force" (from scripts written while it was opt-in, and while a size
guard existed) — leaves it on. A .gguf or safetensors load has no .giw mapping and never aliases.

There is deliberately no size limit. One existed from 2026-09-24 until the same day: on gemma4-26b the
aliased arm paged the whole server out at its first request, 3 of 3 times. The cause was not aliasing's
memory but fork(): a fork while Metal had any page of the MAP_PRIVATE mapping wired copied the entire
15 GB mapping, and the swap guard forked every 2 s. decoder.Load now marks the mapping VM_INHERIT_NONE and
the guard reads swap with a bare sysctl; with both, four interleaved M26 arms (two aliased) ran clean with
swap flat (docs/measurements/m26-alias-fork-collapse-2026-09-24.md).
```

## weightAlias.int8Codes

Moved from `metal/alias.go` (the comment above `weightAlias.int8Codes`) on 2026-10-10.

```text
int8Codes is nibbles for an int8 (W8A8) matrix: its codes are one contiguous array in the .giw, 16-byte
aligned since weights format v12, and the kernels read them as the same signed bytes, so they can be bound
in place exactly like int4 nibbles. On the 7B this is the LM head: 519.8 MB of the 911 MB the Metal build
still copied after int4 aliasing (measured with a per-caller buffer ledger, 2026-09-24).
```

## gumbelMSLKernels

Moved from `metal/gumbel.go` (the comment above `gumbelMSLKernels`) on 2026-10-10.

```text
gumbelMSLKernels — device-side temperature-only sampling by Gumbel-max, the Mac half of R7b
(docs/tasks/red-october.md). Port of cuda/gumbel.cu and gpu/gumbel.go (WGSL) — read either
first; this mirrors their structure kernel-for-kernel over MSL's own constraints, checked
directly against the Metal Shading Language Specification (2026-06-04 rev), not assumed:

  - MSL HAS NO log1p. It does not appear anywhere in Table 8.1 or 8.2's function lists (the
    complete standard math function set), unlike CUDA's log1pf. So the small-w branch below
    uses the SAME 12-term Taylor polynomial gpu/gumbel.go's WGSL kernel already uses (w < 0.25:
    -w*p(w), computed directly from w, never forming 1-w) — for the same reason WGSL needed it:
    computing 1.0-w in f32 for w below ~6e-8 rounds EXACTLY to 1.0 (below f32's representable
    ULP spacing near 1), losing the input entirely regardless of which log variant runs on the
    result afterward. That is a cancellation problem in the subtraction, not a log-accuracy
    problem, so a more-accurate log alone cannot fix it — only avoiding the subtraction does.
  - Unlike WGSL, MSL natively supports 64-bit unsigned integers (ulong, Metal 2.4+; this
    library already targets MSL3_1), so Philox's mulhi is a direct 64-bit multiply + shift —
    the same computation decoder.philox4x32 does in Go — not WGSL's 16-bit-limb reconstruction.
  - precise::log is used for the w >= 0.25 branch and the h >= 2^31 branch. Table 8.1 (accuracy
    with fast math off, i.e. what metal::precise selects) gives log <= 4 ulp for any x > 0;
    Table 8.2 (fast math, the library-wide default here — metal/model.go's preciseMathCompile
    defaults to false / CompileLibrary, not CompileLibraryPrecise) gives only "absolute error <=
    2^-21 near x=1, else <= 3 ulp" — the same weak near-1 guarantee WGSL's own comment already
    flags. Verified against the spec's actual tables, not assumed to match WGSL's.
  - NO FMA IN THE KEY. #pragma METAL fp contract(off) brackets both kernels below. The spec
    states the library's default (fast) contraction fuses a*b+c ACROSS STATEMENTS, not only
    within one expression (a scale-then-add pair of separate statements is exactly the case
    "fast" is documented to still fuse) — so writing the multiply and add as two statements,
    the way decoder.gumbelKey's explicit float32(...) casts do on the host, is NOT on its own
    sufficient here. The pragma is the documented mechanism to forbid contraction at exactly
    this kernel's scope without recompiling the whole allKernels library under
    CompileLibraryPrecise (metal/model.go), which would also change every other kernel's bits
    and timing — not what a two-kernel addition should do. Restored to `fast` immediately after
    the second kernel so nothing else in allKernels is affected.

TWO KERNELS, matching CUDA/WGSL: gumbel_stage1, one thread per group of 4 vocabulary entries
(one Philox call, four keys), threadgroup-level argmax reduction, one (key, index) pair per
threadgroup. gumbel_stage2, one threadgroup reduces those into the final id. Ties go to the
LOWEST index (strict >), as on the host and every other backend.

THREADGROUP MEMORY LAYOUT. DispatchTG takes one byte count for one [[threadgroup(0)]] pointer,
so a (key, index) pair cannot use two separately-typed threadgroup arrays the way a plain CUDA
__shared__ pair can. Each slot t packs its key at shm[2t] and its index bit-reinterpreted via
as_type<float> at shm[2t+1] — the same bit-reinterpretation WGSL's own kernel already relies on
(bitcast<f32>(prm[1]) for invT), not a raw pointer cast between element types.
```

## Removed tracker labels and short history (metal-wM2)

Comment lines the diet cut down to their contract (a tracker id, a date or a one-clause history), from `metal/backend.go`, `metal/gemma4_moe.go` and `metal/alias.go`, on 2026-10-10. Each is the old text at its old line, verbatim; the surviving comment says what the code does.


### metal/backend.go, before `var metalCtxCeiling sync.Map // *decoder.Model -> int`

```text
metalCtxCeiling is S18's context ceiling per model (docs/tasks/task-multimodal-support-2026-10.md, "S18 on the Mac"):
what shrinkCtxToFit lowered an unpinned resident context to, read by resolveMetalCtxCap. A map keyed by the model, so
the decoder keeps no backend state.
```


### metal/backend.go, before `budget := metalMemoryCeiling(ram)`

```text
S4 item 3: the SAME combined ceiling the registered "metal" memory probe reports to
decoder.Model.Plan, so the two can never disagree.
```


### metal/backend.go, before `fastOnce sync.Once`

```text
fast is the batched-prefill decision, made ONCE per resident on first use: the
GOINFER_METAL_FAST_PREFILL env var used to be re-read on every prefill call, so a change to
the process environment mid-serve silently switched a loaded model's kernels. Lazy (not set at
build) so a resident built as a struct literal in a test gets the same default a real one does.
```


### metal/backend.go, before `func (a *metalResident) ctxCap() int {`

```text
ctxCap is this resident's resolved KV capacity — a.r.ctxCap when a real *resident exists, else
metalCtxCapDefault. The fallback matters for TestMetalResidentCheckCap (metal/resident_cap_test.go),
which deliberately constructs a zero-value &metalResident{} (r == nil) to test checkCap/ContextCap as
pure logic with no Metal device — that test predates G6's per-build ctxCap and is meant to keep
working unmodified against "the historical constant" semantics, so a nil/zero r reads as "no
explicit request was ever resolved here", not as 0.
```


### metal/backend.go, before `a.Reset()`

```text
Fresh sequence: re-zero any Gated-DeltaNet state before it carries over from a prior
Generate on this resident (audit C-01's CUDA analogue). No-op for every other family.
```


### metal/backend.go, before `}`

```text
return nil, err // C-09: a command buffer aborted — surface it, do NOT return stale logits
```


### metal/backend.go, before `func (a *metalResident) SampleAvailable() bool { return a.r != nil && a.r.Sample`

```text
SampleAvailable/ForwardSample (decoder.ResidentSample, R7b Mac half — gumbel_sample.go) delegate
straight to *resident: this wrapper adds no logic of its own here, but the methods must exist on
*metalResident (not just *resident) because *decoder.Model.resident holds a *metalResident, and
Go does not promote a NAMED field's (a.r) methods the way an embedded one would — see this
struct's own field comment and every other delegating method below it.
```


### metal/backend.go, before `}`

```text
return a.r.takeExecErr() // C-09: a command buffer aborted — surface it
```


### metal/backend.go, before `}`

```text
return true // §3.2 gate passed 2026-09-09 (S cells K=256/512/1024); floor lowered to 64 2026-09-20 (R3)
```


### metal/backend.go, before `}`

```text
return true // default ON 2026-09-21 (R2) — see the doc comment above
```


### metal/backend.go, before `}`

```text
return true // §3 gate passed 2026-09-10 (S set B decision cells K=256/512/1024 + K=3900 confirm)
```


### metal/backend.go, before `if a.emodelBatched() && a.fastPrefill() && startPos+len(embeddings) >= metalFast`

```text
4b (docs/tasks/task-m26-mac-2026-10.md): a paged Gemma 4 MoE (M26) takes no batched pass; its prompt runs layer by
layer on decode's own kernels, bit-identical to the sequential loop, so neither the floor nor --exact-prefill
applies. Behind g4LayerMajorOn until graded.
S9 step 2: an E-model prompt at or above the floor takes the batched pass when it is on; below the floor, or on any
decline, the layer-major pass below runs it as before.
```


### metal/backend.go, before `promptLen := startPos + len(embeddings)`

```text
FLOOR: below metalFastPrefillFloor no decision cell has passed, so the batched pass declines there; the lowest that
passed is K=16 (the floor's own doc comment). E-P01: a resident with the step kernels runs a prompt that ends below
max(floor, metalStepPrefillCeiling) as decode rows on those kernels instead: the sequential loop's bits, faster
than the pass there (promptStepOK).
```


### metal/backend.go, before `func (a *metalResident) PrefillImageLast(ctx context.Context, embeddings [][]flo`

```text
PrefillImageLast (decoder.ResidentImagePrefill; S17's Metal image prefill, docs/tasks/task-multimodal-support-2026-10.md) is
the batched f16 pass over an image turn's spliced embeddings with the image block [imgStart, imgEnd) attending
bidirectionally (attention_prefill_img), as CUDA's PrefillImageLast and the CPU's prefillLogitsVL. The whole prompt in one
pass, never chunked across the block. It takes the plain uniform pass only: a Gemma 4 E-model (its rows carry per-layer
inputs this route does not build), a layer-major family or a recurrent hybrid declines, and the decoder then takes its
CPU-prefill and upload bridge, unchanged.
```


### metal/backend.go, before `if a.r.kvI8 {`

```text
The f16 MMA prefill kernels implement a dense gated FFN (SiLU or GeGLU, G8) out of
L.guW/L.dW with per-layer rope/window, per-head QK-norm, and Gemma's sandwich norms, and
(G8 MoE half) a generically-shaped gated-SwiGLU MoE FFN — expert-major by default, the
per-token decode MoE chain row by row with GOINFER_MOE_EXPERT_MAJOR=0 (prefill.go) — but
NOT per-layer-varying attention geometry (dense Gemma 4's local/global head_dim split;
prefillOK's own per-layer-geometry guard, metal/model.go) or Gemma 4's enable_moe_block
variant (residLayer.g4moe, a third FFN shape this path never reads; explicitly excluded
via HasGemma4MoEResident regardless of per-layer geometry). Any of these declines here,
and the caller re-runs the prompt through the (correct) sequential Forward loop.
```


### metal/backend.go, before `if startPos < 0 || len(embeddings) == 0 || startPos+len(embeddings) > a.ctxCap()`

```text
startPos < 0 would wrap to a huge uint32 and make kv_store_f16 write far out of bounds — on
UMA that silently corrupts adjacent buffers (audit R-27). The decoder never passes one (it passes 0,
a reused prefix's length, or a chunk's offset; F-G01), but the guard is cheap.
```


### metal/backend.go, before `if a.r.prefillExactAttn() && startPos+len(embeddings) > prefillExactAttnMaxKeys `

```text
F-C02: the exact attention kernel (no fused kernel for this head dim, or fused attention off) holds at most
prefillExactAttnMaxKeys scores; the sequential path's decode kernels tile theirs.
```


### metal/backend.go, before `if startPos == 0 {`

```text
A fresh sequence (startPos 0) starts a Gated-DeltaNet hybrid's recurrent state from zero, as Forward(pos 0) does: the
pass continues whatever conv window and state the resident holds (D-B01), which for a continuation is the point and
for a new prompt would be the previous sequence's. A no-op for every other family.
```


### metal/backend.go, before `var logits []float32`

```text
ensurePrefill's compile panic and the ~24 per-call MustBuf OOM panics fire HERE, at request
time, with no recover of their own (buildResident's is build-scoped). A transient OOM would kill
the server; recover into an error so the request fails and the caller falls back to sequential
decode (audit R-23; B-10 class).
```


### metal/backend.go, before `if a.r.prefillResidNonFinite {`

```text
S17 (docs/tasks/task-multimodal-support-2026-10.md): the f16 residual overflowed. The head reads the NaN row as
finite zeros, so the logit check below cannot see it.
```


### metal/backend.go, before `if i := firstNonFinite(logits); i >= 0 {`

```text
A-C02 (docs/audit-metal-2026-09-30.md): the batched pass carries f16 activations, which a checkpoint the
graded models never exercised could overflow; the sequential decode kernels carry f32. A non-finite logit
declines, so the decoder re-runs the prompt sequentially, which rewrites every K/V row this pass wrote. The
scan is about 0.05 ms at a 152k vocabulary.
```


### metal/backend.go, before `if (a.r.g4moe != nil && a.r.g4moe.paged) || (a.r.moe != nil && a.r.moe.paged) {`

```text
forwardHiddenNoHead's trunk encoder (encodeTrunkInto → encodeLayer) has no paged branch — only
Forward's dispatch to forwardLogitsPaged/forwardLogitsMoEPaged does — so on a paged MoE it would
bind the zero-value stacked expert buffers and return a finite garbage hidden state with no
error (audit-metal-2026-09-12.md C-02). Decline instead: decoder.Model.HiddenLast treats a
resident decline as "no resident backend for this request" and falls through to the CPU path,
exactly like an OOM or cap decline does (decoder/embed.go:Model.HiddenLast).
```


### metal/backend.go, before `if err := ctx.Err(); err != nil {`

```text
G18: an abandoned client otherwise leaves the whole sequence streaming through the
device with nothing watching — same discipline as residentPrefillSeed's sequential loop.
```


### metal/backend.go, before `}`

```text
return nil, err // C-09: a command buffer aborted — surface it, do NOT return a stale/zero vector
```


### metal/backend.go, before `}`

```text
return nil, err // C-09: a command buffer aborted; never return a stale residual
```


### metal/backend.go, before `}`

```text
return nil, err // C-09: a command buffer aborted — surface it
```


### metal/gemma4_moe.go, before `paged   bool`

```text
fence                         *pagedFence // M-11 / C-B03 (paged_fence.go); nil unpaged or where MSL 3.2 does not compile
```


### metal/gemma4_moe.go, before `wg.Add(2)`

```text
if scalesInFile { // C-P01: the scales are four more disjoint destinations' worth of the same argument
```


### metal/gemma4_moe.go, before `rp := pos`

```text
N-48 (docs/audit-2026-09-10.md): setPos, not a direct uPos/uNKeys write — same gap as
metal/moe.go's paged path, see its own comment there.
```


### metal/gemma4_moe.go, before `var arp ARPool`

```text
async := pagedAsyncPhase2On && !split // M-11: phase 2 committed, not waited (pagedAsyncPhase2On)
```


### metal/gemma4_moe.go, before `return`

```text
r.recordExecErr(e.Err()) // C-09: catch an abort in any paged per-layer submit
```


### metal/gemma4_moe.go, before `var seq uint32`

```text
fenced := g.fence != nil && async // M-11 / C-B03: the route mirrored and fenced, the host spins (paged_fence.go)
```


### metal/gemma4_moe.go, before `slots := L.g4moe.pool.ensureResidentBatch(ids)`

```text
M-12 (audit-metal-2026-09-12.md): stage every miss in this token's top-k concurrently
instead of one pread at queue depth 1 per expert — see ensureResidentBatch's own doc
comment for why this is safe.
```


### metal/gemma4_moe.go, before `gIdx := g.slotIdx.U32s()`

```text
M-11: tell the GPU which pool row holds each routed expert — a small host→device write,
not a re-encode. slotIdx replaces the old idxZeros (see gemma4MoeResident's doc comment).
```


### metal/gemma4_moe.go, before `e2.UseResidencySet(r.residency)`

```text
if r.residency != (ResidencySet{}) { // M-14: per-encoder attach, phase 2 only
```


### metal/alias.go, before `return int4Concat(d, wms...)`

```text
if w.Cols()%32 != 0 { // int4Concat owns the K%32 panic (audit M-10)
```


### metal/alias.go, before `func (a *weightAlias) summary() string {`

```text
summary is the banner line S6 asks for: the number a user would otherwise never see — how much of the
weights is served from the file and how much the build still copied into anonymous memory — plus, for a
file whose int4 scales are not in the layout Metal binds, what to do about it. Since weights format v15 every
bundle stores binary16 scales, but only a -target metal bundle (kind 7, and fused groups) lays them out to be bound,
so a v15 file for another target is converted too (F-D02, audit-metal-2026-09-30.md).
```

## TestAttentionPrefillFused_ctxCapNotMultipleOf8

Moved from `metal/attention_prefill_fused_cachepad_test.go` (the comment above `TestAttentionPrefillFused_ctxCapNotMultipleOf8`) on 2026-10-10.

```text
TestAttentionPrefillFused_ctxCapNotMultipleOf8 gates C-01 (audit-metal-2026-09-12.md):
attention_prefill_fused's key loop reads whole 8-row simdgroup tiles and masks the ragged
remainder AFTER the load, so a resident cache built for a ctxCap that is NOT a multiple of 8
can have its last, ragged tile read past kc/vc's actual allocated end when nKeysMax reaches
that cap exactly. The fix (metal/model.go buildResident) rounds the kc/vc ALLOCATION up to a
multiple of 8 rows while leaving r.ctxCap itself (the checked, user-visible capacity) alone.

This asserts the allocated BYTE LENGTH directly (Buffer.Len(), the logical size passed to the
allocator) rather than trying to observe the OOB read's effect on output: a first attempt at
this test drove PrefillLast right up to the ctxCap=37 boundary and compared logits against the
sequential-Forward reference, and it passed identically with the fix reverted — Metal's actual
buffer backing is page-rounded (16 KB on Apple silicon) regardless of the requested length, so
a 37-row and a 40-row request for a buffer this small land on the exact same physical
allocation and the "OOB" read is silently in-bounds either way. Buffer.Len() reports the
LOGICAL length the caller asked for, not the physical rounding, so it is the only reliable way
to see this fix take effect — the same "prove the gate can go red" discipline this repo's other
gates are held to (CLAUDE.md).
```

## TestOlmo3ResidentSmokeMetal

Moved from `metal/olmo3_resident_smoke_test.go` (the comment above `TestOlmo3ResidentSmokeMetal`) on 2026-10-10.

```text
TestOlmo3ResidentSmokeMetal and TestOlmoHybridResidentSmokeMetal are G5's FeatPostOnlyNorm +
FeatQKNormWhole row (docs/tasks/task-gpu-paths-2026-09.md) smoke gates: the model actually goes
resident and produces finite, non-degenerate output.

DELIBERATELY NOT a resident-vs-CPU cosine floor — same finding G5 rows 1-2 already recorded for
their own seeded/synthetic tiny fixtures (testdata/olmo3-tiny, testdata/olmo_hybrid-tiny are
both "seeded" per their own pin_*.py scripts, same pattern). This feature has no new pure-Go
formula the way FeatNoPE/FeatAttnTemp did, so the real correctness gate is one level lower:
TestQKNorm_wholeVector (qknorm_whole_test.go) proves the whole-vector qk_norm DISPATCH GEOMETRY
directly against the real production kernel with an exact per-component comparison (no GPU
quantization noise at all in that path) — the strongest evidence this row has. The postOnly
pre-norm skip (quant_vec instead of rmsnorm_quant) reuses an already-proven, unmodified kernel
(ctx-before-o-proj already dispatches it), so its own correctness rests on that kernel's
existing coverage plus this smoke test's admission-and-no-NaN check.
```

## TestPrefillLast_startPosGreaterThanZero

Moved from `metal/prefill_startpos_test.go` (the comment above `TestPrefillLast_startPosGreaterThanZero`) on 2026-10-10.

```text
TestPrefillLast_startPosGreaterThanZero gates G-08 (audit-metal-2026-09-12.md): the §3.2 pooled
gate (prefill_gate_ref_test.go) only ever calls PrefillLast(embs, 0) — every resident-prefix-
reuse turn (decoder/model.go's residentPrefillSeed, `from` — an agent loop continuing from an
already-resident prefix, the peer matrix's own headline workload) calls it with startPos > 0,
and the fused kernel's startPos/uMReal masking (attention_prefill_fused's nKeysMax computation)
has no coverage at that shape outside one synthetic hd=64 unit case.

A focused correctness check on the tiny synthetic fixture, not a change to the pooled gate's
own carefully pre-registered statistics (decisionKs/confirmKs, the critA/B/C formulas) — G-08's
own confidence is "plausible, coverage gap, no defect shown", and this closes the gap without
risking the established methodology those formulas represent. Builds the SAME shared KV prefix
[0,from) on two residents via Forward (bit-identical by construction — same sequential path),
then diverges: one continues the reference way (Forward, one token at a time) through [from,K);
the other takes the SAME suffix through PrefillLast(embs[from:], from) — the exact code path
G-08 flags as uncovered. Compares the two residents' final logits at position K-1.

The fixture's head dim is 16, so PrefillLast runs attention_prefill_fused here. At head dim 128 production runs
attention_prefill_steel, which TestAttentionPrefillSteelMatchesFloat64 checks at startPos > 0 (F-G01).
```

## TestResidentNeedBytes_honorsPagingSlots

Moved from `metal/resident_memguard_test.go` (the comment above `TestResidentNeedBytes_honorsPagingSlots`) on 2026-10-10.

```text
TestResidentNeedBytes_honorsPagingSlots is M-02's gate for the actual wiring gap: reverting
residentNeedBytes to always call ResidentWeightBytes() (the pre-fix behavior) compiles clean
and TestMetalMoESlotsFromEnv above still passes, because that test only exercises the parsing
function in isolation — it never proves the guard USES what it parses. This does, by loading a
real (tiny) MoE checkpoint and comparing residentNeedBytes' output against
ResidentWeightBytes/ResidentWeightBytesPaged/ResidentHostCopyBytes/residentKVBytes directly,
with no real RAM or a checkpoint large enough to swing residentFitsMemory's verdict required.

2026-09-09 (M-02 continued): residentNeedBytes gained two more additive terms (the host-copy
addend and KV bytes) beside the paged-weight term this test originally gated alone — so "==
unpaged weight bytes" is no longer residentNeedBytes' own contract; the assertions below add
the SAME two terms back in, computed independently via the public accessors, so this still
catches a regression in the paging wiring specifically without needing to be rewritten every
time another additive term is found.
```

## TestSmolLM3ResidentSmokeMetal

Moved from `metal/smollm3_resident_parity_test.go` (the comment above `TestSmolLM3ResidentSmokeMetal`) on 2026-10-10.

```text
TestSmolLM3ResidentSmokeMetal is G5's FeatNoPE row (docs/tasks/task-gpu-paths-2026-09.md) smoke
gate: the model actually goes resident and produces finite, non-degenerate output when its
LAST layer is NoPE (testdata/smollm3-tiny's no_rope_layers=[1,1,1,0], 1=has-rope/0=NoPE).

This is DELIBERATELY NOT a resident-vs-CPU cosine floor, and used to be one — MEASURED (not
assumed) that this fixture cannot discriminate a correct NoPE implementation from a broken
one: with the real fix (zero invFreq on layer 3 only), with the fix reverted (real invFreq on
every layer, i.e. layer 3 wrongly ropes), and with EVERY layer's invFreq forced to zero (layers
0-2 wrongly skip rope too), the worst cosine against the CPU reference over 32 tokens was
0.9619, 0.9617, and 0.9624 respectively — a ~0.0006 spread, all three configurations equally
"passing" or "failing" any threshold that would separate them. The seeded/synthetic weights at
hidden=64 mean int8 quantization noise dominates the comparison regardless of rope correctness
— the SAME "cannot discriminate a real bug from quantization noise on unstructured weights"
finding this codebase already recorded for Mellum-on-Metal (features.go's FeatRopeMscale note).

The actual correctness gate is decoder.TestRopeInvFreqLayer_NoPEIsZero — a pure, backend-
agnostic unit test of RopeInvFreqLayer itself (exact zero/non-zero per layer, no GPU, no
quantization noise), plus the shipped rope2 kernel math being exact identity at invFreq==0 by
construction (metal/kernels.go, metal/rope_test.go's TestRope_mscale family already exercises
that kernel directly). This test's only job is: does declaring FeatNoPE actually let the model
go resident and run without error/NaN.
```

## TestPhiloxGumbelMSL_mutationDetectsAConstantChange

Moved from `metal/gumbel_test.go` (the comment above `TestPhiloxGumbelMSL_mutationDetectsAConstantChange`) on 2026-10-10.

```text
TestPhiloxGumbelMSL_mutationDetectsAConstantChange applies R7b's own registered mutation check
(docs/measurements/sampled-gumbel-2026-09-20.md: "a Philox constant changed on the CUDA kernel →
mismatches with host-key gaps 1.4-5.2") to the MSL kernel: flip gumbel.go's philoxM0 constant
(0xD2511F53 -> 0xD2511F52), rebuild the library, and confirm TestGumbelDeviceAgreesWithHost's own
rule (every mismatch within 5e-5 of the host's own keys) goes red with LARGE gaps — proving the
gate can actually catch a broken kernel, not just pass vacuously on a correct one. Restores the
source afterward and confirms the restoration is byte-identical.
```

## TestSampledDecodeLadder_speed

Moved from `metal/sampled_gumbel_speed_test.go` (the comment above `TestSampledDecodeLadder_speed`) on 2026-10-10.

```text
TestSampledDecodeLadder_speed is R7b Mac's speed measurement (docs/tasks/red-october.md),
following the same protocol docs/measurements/sampled-gumbel-2026-09-20.md's CUDA/WebGPU numbers
used (paired against greedy, same session, interleaved with a rotating start, a discarded
warm-up round, and a do-nothing arm) — CUDA's own TestSampledDecodeLadder was not found
committed anywhere in this tree to port directly (checked: no match repo-wide), so this is a
from-scratch harness built to the SAME protocol description, not a line-for-line port.

THREE ARMS, one seed, one prompt, per round:
  - greedy       (Temperature: 0)
  - host draw    (Temperature: 1.0, GOINFER_NO_SAMPLE_FASTPATH=1) -- the do-nothing arm: proves
    the device path is worth having at all, not just that it beats itself
  - device draw  (Temperature: 1.0, fastpath default)

PAIRED, NOT POOLED (CLAUDE.md measurement discipline, rule 7): each round runs all three arms
back to back before the next round starts, and the ratio is computed PER ROUND, then those
per-round ratios are summarized (median + spread) -- never a ratio of pooled means, which would
carry between-round variance (thermal, scheduler noise) into the comparison.

DECODE-ONLY: timed from the first emitted token to the last (time-to-first-token subtracted),
not from the call start, so prefill cost (paid once, off this measurement's critical path in a
real decode-bound workload) doesn't dilute the per-token rate.

Run: GOINFER_HEAVY_TESTS=1 go test -tags goinfer_testhooks ./metal/ -run TestSampledDecodeLadder_speed -v -timeout 30m
```

### metal/attention_prefill_fused_cachepad_test.go, on `const ctxCap = 37`

```text
const ctxCap = 37                                 // NOT a multiple of 8 — the exact shape C-01 describes
```

## Test files (metal-wM2): tracker labels and short history removed

Comment lines of the test files `metal/resident_memguard_test.go`, `metal/gumbel_test.go` and `metal/moe_expert_major_prefill_test.go` that the diet cut down to what the test pins, verbatim, on 2026-10-10. The test names are unchanged; the larger stories are under their own `## Test...` headings above.


### metal/resident_memguard_test.go, before `{"gptoss20b_on_16gb", 11280 * gb / 1000, 16 * uint64(gb), false},`

```text
THE MEASURED CASE. gpt-oss-20b's 11.28 GB of weights on a 16 GB MacBook drove swap to
35.98 GB of 36 GB and never completed or declined. It must be refused.
```


### metal/resident_memguard_test.go, before `func TestMetalMemoryCeiling_takesTheStricterBound(t *testing.T) {`

```text
TestMetalMemoryCeiling_takesTheStricterBound is S4 item 3's own gate (task-never-swap-2026-09.md):
the combined ceiling must equal the static ram*residentMemFraction figure whenever the live
probe is unknown or looser, and must equal the LIVE figure whenever that is the tighter one —
never the other way around, since a live figure ALLOWED to widen the ceiling would reopen
exactly the darwin-UBC "guard that inverts under pressure" failure this file's own doc comment
on metalMemoryCeiling explains.
```


### metal/resident_memguard_test.go, before `func TestResidentFitsMemory_honorsLiveCeiling(t *testing.T) {`

```text
TestResidentFitsMemory_honorsLiveCeiling is S4 item 3's own WIRING gate:
TestMetalMemoryCeiling_takesTheStricterBound above proves the combined-ceiling arithmetic in
isolation; this proves residentFitsMemory — the REAL load-time guard, not a private copy of the
same formula — actually reads through it. A version of residentFitsMemory that reverted to its
pre-item-3 static-only budget would still pass every other test in this file (they never touch
metalLiveAvailable) while silently ignoring a live probe reporting almost no memory at all —
exactly the wiring gap this guards against.
```


### metal/resident_memguard_test.go, before `func TestMetalMoESlotsFromEnv(t *testing.T) {`

```text
TestMetalMoESlotsFromEnv is M-02's gate for the guard's half of the ordering fix:
residentFitsMemory must ask ResidentWeightBytesPaged for the SAME N that metal/moe.go and
metal/gemma4_moe.go are about to honor, not silently fall back to the unpaged number on
anything it cannot parse cleanly. Mirrors those two files' resolution exactly (metalMoESlotsRequest),
except an invalid/unset value means "assume unpaged" here (safe: buildResident still validates
and declines on a bad value) rather than a hard error.

Phase 2 (docs/tasks/task-gpu-paths-2026-09.md — "Metal slots become an Option and a flag"): the
env-var cases below now go through a model whose Options.MoECacheSlots is 0 (unset), so
metalMoESlotsRequest's fallback to GOINFER_METAL_MOE_SLOTS is what's actually exercised — an
additional case pins the NEW priority order directly (Options wins over the env var when both
are set).
```


### metal/resident_memguard_test.go, before `if hc0, hc1 := m.ResidentHostCopyBytes(0), m.ResidentHostCopyBytes(1); hc1 >= hc`

```text
The host-copy addend must ITSELF shrink under paging (that's the whole point of M-02's
distinction): paged experts stream, so ResidentHostCopyBytes(1) must be strictly smaller
than ResidentHostCopyBytes(0) whenever paging actually caps anything on this fixture.
```


### metal/gumbel_test.go, before `func TestGumbelDeviceAgreesWithHost(t *testing.T) {`

```text
TestGumbelDeviceAgreesWithHost is the kernel gate for gumbel_stage1/2 (R7b Mac half), a direct
port of cuda's TestGumbelDeviceAgreesWithHost (cuda/gumbel_test.go) — read its header first;
this uses the exact same pre-registered rule, rows and seeds.

PRE-REGISTERED RULE (written before the first run, matching the CUDA gate). The host draw
(decoder.gumbelDraw, f64 noise transform) is the reference; the device computes the same argmax
with an f32 transform. Philox is integer-exact, so the two agree except where the best two keys
are within a few f32 ulps. Therefore:

 1. Overall agreement >= 99.99% of draws (mismatch rate <= 1e-4).
 2. EVERY mismatch is a genuine near-tie: the HOST's own keys for the device's token and the
    host's token differ by <= 5e-5. A mismatch with a larger gap would mean the noise itself
    differs — a bug, not rounding.

Rows: the real vocab and sizes that are not multiples of 4 or of the block (GB_THREADS=256, so
the block is 1024 entries), normal / peaked / -inf-masked / flat; temperatures 0.3, 1.0, 2.0;
seeds and draw indices that exercise the high words of both.
```


### metal/moe_expert_major_prefill_test.go, before `func TestMoEExpertMajor_ParityVsRowByRow(t *testing.T) {`

```text
TestMoEExpertMajor_ParityVsRowByRow verifies that the expert-major batched prefill
produces near-identical logits and matching argmax compared to the row-by-row fallback.
```
