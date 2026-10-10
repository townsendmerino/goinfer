# cuda: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `cuda`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

## prefillCore: m-RoPE and image-block uploads

Moved from `cuda/prefill.go` (the comment in the body of `prefillCore`) on 2026-10-09.

```text
Qwen2.5-VL m-RoPE: build and upload this pass's per-row (t,h,w) rotation triples ONCE,
reused unchanged across every layer's rope_kv_mrope_batched launch below.
mropePosWindow's own doc comment explains why the absolute-vs-chunk-relative slicing
lives in a named, separately-tested function rather than an inline expression here.
S11: every image block's [start, end), uploaded once for every layer's attn_img_batched launch.
```

## prefillCore: K=V copy

Moved from `cuda/prefill.go` (the comment in the body of `prefillCore`) on 2026-10-09.

```text
K=V (Gemma-4 global layers): this layer has NO v_proj. V is v_norm(the RAW pre-RoPE k_proj output), so copy the k projection into the V buffer here and normalize it
below, before rope_kv_batched rotates k. Mirrors segA's decode path op for op. It used to project k a SECOND time "because decode does": decode now copies too
(R-23, docs/tasks/task-recompute-audit.md), so the two still agree, and neither reads the k weight twice. kBb and vBb are [M, kvDim] row-major, so one contiguous
copy of M*kvDim floats moves every row.
```

## prefillCore: default attention tile

Moved from `cuda/prefill.go` (the comment in the body of `prefillCore`) on 2026-10-09.

```text
Default tile (attn-fused-tile128-default-PREREGISTERED.md): hd128 layers with no sliding window run the
128-row-tile kernel (bit-identical to the 64x64 one for window == 0, 2.4x faster at K=3900). hd64 stays 64x64
(0.5B measured ~5% slower with it), and so do windowed layers (their key-tile grouping starts at the block's
first row, so a taller block is not bit-identical there). GOINFER_CUDA_ATTN_FUSED_TILE=64x64 forces 64x64.
```

## prefillCore: MoE FFN

Moved from `cuda/prefill.go` (the comment in the body of `prefillCore`) on 2026-10-09.

```text
--- FFN. Dense batches; MoE runs ROW BY ROW off the batched residual. ---

The routed-expert GEMVs are indexed by a DEVICE-side routing decision that differs per
token, so there is no M-wide form of them without an expert-major gather and a new
kernel (queue-performance P20 step 2). What there IS, for free, is the attention half
above: on Gemma-4-26B-A4B the attention projections are ~45% of the per-token weight
traffic and the dense FFN branch another ~25%, all of it re-read once per token on the
sequential path and once per PASS here.

aikit/gpu.Buffer.At gives a zero-copy sub-view that binds as a raw device pointer, so
row m of the batched residual IS a valid single-row residual for the existing per-token
FFN chain — segBFFN → layerTail → segC, the same calls decode makes, in the same order,
including the g4x2 accumulator clear and the C′ routed-expert DMA. Nothing about the
expert path changes; it simply no longer drags the attention weights along with it.

gC=false: prefill never replays captured graphs (a graph bakes r.x, and these rows are
not r.x). The per-token debug seams layerTail also carries — hidCapTaps, layerCap —
would fire M times per layer here, which is why prefillStaticDecline refuses a model
with either armed rather than quietly returning M× the rows they expect.
```

## prefillCore: MoE row cancellation

Moved from `cuda/prefill.go` (the comment in the body of `prefillCore`) on 2026-10-09.

```text
Between ROWS: this loop is the one that made cancellation coarse. A MoE
chunk is M sequential per-token FFNs, so without this a cancelled 512-row
chunk still runs every one of them — measured ~22 s on M26, against the
~46 ms the per-token fallback it replaced would have taken to notice.
Checked per row, so the granularity is back to roughly one token.
```

## batchedHeadArgmax

Moved from `cuda/prefill.go` (the comment above `batchedHeadArgmax`) on 2026-10-09.

```text
batchedHeadArgmax is tailAllArgmax's tail: ONE batched final-norm, ONE batched head GEMV over
all M rows, M argmax reductions, then a 4-bytes-per-row readback.

The win it exists for: the per-row tail issues the head as an M=1 GEMV per row, so the head's
~389 M parameters are re-read from VRAM M times. Measured, that marginal row costs 1.046 ms
against a 0.934 ms single-row head — no amortization whatsoever, in the one place the batched
pass exists to provide it.

Buffers are allocated lazily HERE because af/ai need r.dev's context current, which holds on
the executor thread this runs on. They are sized to M and reused; a wider block reallocates
once. The old buffers are left to the device ledger rather than freed mid-job, which is the
same lifetime the rest of the resident scratch has.
```

## batchedHeadArgmax: release before growing

Moved from `cuda/prefill.go` (the comment in the body of `batchedHeadArgmax`) on 2026-10-09.

```text
RELEASE BEFORE GROWING. This was grow-only: each larger prompt abandoned the previous
buffer to the device ledger, bounded only by 2*ctxCap*vocab*4 B — about 5 GB at the 4096
default, on top of the live one (audit-2026-09-02 C-12).
```

## batchedHeadArgmax: device argmax

Moved from `cuda/prefill.go` (the comment in the body of `batchedHeadArgmax`) on 2026-10-09.

```text
The argmax used to be taken on the HOST after downloading all M×vocab logits ("a batched
argmax kernel is a later, separate ~1 ms"). R14 measured that tail at 15-16% of a spec round
(docs/measurements/r14-drafter-argmax-2026-09-22.md); argmax_rows now reduces on the device
and M ints come back.
```

## batchedHeadFull

Moved from `cuda/prefill.go` (the comment above `batchedHeadFull`) on 2026-10-09.

```text
batchedHeadFull is prefillCore's batched head for tailAllLogits, where every row's FULL logits
are needed (PrefillLastN, speculative verify, MC3's StepBatch) — unlike batchedHeadArgmax, which
only ever returns an id per row. It replaces the M single-row upload+norm+GEMV calls the tail
used to make with ONE batched norm+quant (bNormB) and ONE batched GEMV (bGemvB) over all M rows
— the same primitives batchedHeadArgmax already uses, and bGemvB's own doc comment documents its
int8 kernel as "bit-identical to gemv_w8a8_fwd by construction" (exact int32 accumulation; tiling
M changes no element) — the S0 measurement (concurrency-mc3-cuda-s0-2026-09-27.md) found the same
holds for the int4 kernels used elsewhere in this same batched pass. The GEMV was the only
redundant part of the old loop: it read the SAME lm_head weights M times over.

Everything downstream of the GEMV stays exactly as the loop it replaces did it, per row:
  - a step row with an on-device draw (rows[m].draw != nil) still goes through the unmodified
    stepDraw/gumbelPick kernels, which read the single-row r.logits field, not a batched buffer —
    rather than touching those kernels, this copies that row's slice of the batched logits into
    r.logits first (gpu.CopyDevice, a existing, documented, synchronous device-to-device verb),
    which is cheap for one vocab-sized row and leaves the sampling kernels themselves untouched;
  - every other row is downloaded to the SAME pinned host buffer the per-row loop already used
    (r.logitsPinned/r.logitsHost), one row at a time — the download itself was never the
    redundant part, since M rows always need M separate host slices regardless of how the GEMV
    that produced them was dispatched;
  - softcap and logit scale are applied per row exactly as before (rows != nil only — the same
    condition the loop already used, for the same reason: a step row must equal Forward's own
    return, not only the device buffer's raw contents).
```

## fastPrefillFloor

Moved from `cuda/prefill.go` (the comment above `fastPrefillFloor`) on 2026-10-09.

```text
fastPrefillFloor is the PROMPT-LENGTH floor below which neither fast lever engages.

It is set from measurement, not chosen. Two independent lines of evidence put it here and they
were taken in that order:

  - PERFORMANCE (Phase 1/2, measured before any fidelity run): L2 is 0.94x — SLOWER — at K=128,
    where attention is 5.0% of prefill and a 64-row query tile still stages 64 keys per block.
    The win begins by K=512 (1.13x) and grows. L3 wins at every depth measured (1.85x at K=128).
  - FIDELITY (Phase 3, docs/measurements/prefill-l2l3-phase3-2026-09-05.md): the §3 gate PASSES
    at K>=512 and FAILS at K=256 for the two levers combined, on both S and D7.

THE FLOOR IS AT A DEPTH THAT WAS ACTUALLY MEASURED. The §3 decision cells were 256 and 1024; a
floor placed between them would have been interpolating a fidelity result nobody took, so a
K=512 cell was generated and run specifically to justify this number. Moving it DOWN requires a
passing gate cell at the new depth — not an argument that the curve looks smooth.
```

## useAttnFused

Moved from `cuda/prefill.go` (the comment above `useAttnFused`) on 2026-10-09.

```text
useAttnFused is the ONE place the L2 kernel is chosen, so the fallback cannot drift between call
sites. Every "no" means attn_batched, which is the exact path, is bit-identical to decode, and is
what spec-decode verify and the parity gates run.

THAT LAST CLAIM WAS ASPIRATIONAL UNTIL M-11 (docs/audit-2026-09-10.md): before
forceExactKernels existed, the only gates here were M/floor-based (attnFusedMinRows,
aboveFastPrefillFloor), which verify's own M (often >=16 past the floor) could satisfy by
coincidence of constants, not by construction — "is what spec-decode verify runs" was true of
serve's DEFAULT widths, not guaranteed for every caller. forceExactKernels (set by prefillCore,
cuda/resident.go's own field comment) now makes it a real guarantee: unconditional "no" for any
tail but tailLastLogits, independent of M/K/position.
It returns only the shared-memory size, NOT the pipeline: the launch site names
r.bAttnFused64 / r.bAttnFused128 explicitly. Handing back a Pipeline in a local variable would
hide WHICH kernel runs from every static reader, including
TestPipelineLint_boundKernelsAreLaunched, which flags a field bound at every model load and
launched by nothing — the exact state gemv_w4a8_batched sat in while a benchmark quoted its
throughput as the shipping kernel's.
```

## useGemmMMA

Moved from `cuda/prefill.go` (the comment above `useGemmMMA`) on 2026-10-09.

```text
useGemmMMA is the ONE place the L3 kernel is chosen. Every "no" means gemv_w4a8_rn, which is the
exact path, is bit-identical to the M=1 decode GEMV, and is what the parity gates run.

M-09/M-11 (docs/audit-2026-09-10.md): before forceExactKernels (see its own doc comment,
cuda/resident.go), the only gate here was shape-based (gemmMMAMinRows, aboveFastPrefillFloor) —
speculative verify's own M (often >=16) could cross it by coincidence, not by construction, the
same M-dependent-kernel gap WebGPU's staged int4 path has independently (M-09). forceExactKernels
now makes exactness unconditional for any prefillCore tail but tailLastLogits.

The K constraints are not defensive padding: the kernel contracts 32 elements per group scale and
8 per packed weight word, so a K that is not a multiple of 32 would misalign the group-scale fold.
Every production shape here satisfies it (1536, 3584, 8960, 18944 are all multiples of 32), and a
shape that does not is served correctly by the exact path rather than by a special case.
```

## rnBlockRows

Moved from `cuda/prefill.go` (the comment above `rnBlockRows`) on 2026-10-09.

```text
rnBlockRows must equal RN in gemv_w4a8_rn.cu — each warp computes this many output rows, so the grid
covers ceil(N/rnBlockRows) warps. Bit-identical for any RN; 2 is the profiled knee (halves the L1TEX
load count → halves the scoreboard stall, 4.41→3.38 ms, at the 64-reg / 100%-occupancy limit).
```

## prefillStaticDecline: MoE guard

Moved from `cuda/prefill.go` (the comment in the body of `prefillStaticDecline`) on 2026-10-09.

```text
MoE is no longer a categorical refusal: a MoE layer's FFN runs ROW BY ROW off the batched
residual (prefillCore), so the attention half batches and the routed experts keep the exact
per-token sequence decode uses. What must still decline are the PER-TOKEN DEBUG SEAMS, which
that row loop would fire M times per layer instead of once: hidCapTaps records one residual
per tap per TOKEN for a block drafter, and layerCap appends one snapshot per layer for the
divergence probe. Both would silently return M× the rows their consumers expect. Declining
sends those runs down the sequential path, where their semantics are the ones they were
written against.
SCOPED TO MoE, because that is the only branch that can reach them. layerTail is called from
prefill at exactly ONE site — inside the per-row MoE FFN loop — so on a dense model these
seams are untouched by the batched pass and refusing it gains nothing.

The first version of this guard was NOT scoped, and it broke a real flow: DFlash's block
drafter arms hidCapTaps and verifies through the batched path on a DENSE model, so
TestDFlashRoundComposition and TestDFlashCompositionResidual both failed with "per-token
hidden-state taps are armed". Caught only by the full heavy suite — the targeted prefill
subset does not run the drafter composition tests, which is the second time in this change
that a guard written for one arch refused another.
```

## prefillStaticDecline: DeltaNet and per-layer geometry

Moved from `cuda/prefill.go` (the comment in the body of `prefillStaticDecline`) on 2026-10-09.

```text
RECURRENT STATE (Gated-DeltaNet: qwen3_5 / qwen3_5_moe / qwen3_next). A DeltaNet layer's conv ring and matrix
state must advance strictly one token at a time and in order. The batched path now honours that INSIDE the
pass: prefillCore branches on Ly.isDeltaNet, runs the layer's projections over the M rows, and runs the
recurrence through row-batched twins of decode's kernels that walk the rows in order (prefillDeltaNetRows,
docs/tasks/task-cuda-deltanet-prefill-2026-09.md). It dispatches on the layer's KIND, not on which weights
happen to be absent, so the LFM2 bug class (audit-2026-09-02 C-01: a dense attention stack run over conv
layers that load no q/k/v/o) cannot recur through this path; nonBatchableKind checks the projections each
kind actually binds. ForwardN (spec verify) keeps its sequential path for this family: verify rewinds, and
recurrent state does not.
PER-LAYER geometry, not layer 0's hoisted and asserted uniform. The batched launches bind
each layer's own hd/nKV/qDim/kvDim/rhalf exactly as the decode launches already do, and the
M-sized scratch is sized by the MAX across layers — so a family whose layers differ (Gemma-4:
5 of its 30 are full_attention with a different KV width) strides correctly instead of being
refused. What made the old uniform assertion necessary was hoisting L0 into every launch;
remove the hoist and the assertion has nothing left to protect.
```

## checkPrefillShmem

Moved from `cuda/prefill.go` (the comment above `checkPrefillShmem`) on 2026-10-09.

```text
checkPrefillShmem is prefillStaticDecline's PROMPT-dependent twin (V-05, docs/review-2026-09-04.md):
the single-block batched-prefill attention launch sizes its dynamic shared memory the same way
decode's does, (maxNWin+128)*4 bytes, but — unlike decode — has no split-KV fallback kernel, so
there is no case where exceeding singleBlockAttnShmemLimit is survivable here. Without this check
the launch itself failed at the driver, prefillCore returned an unnamed error, and the caller
(decoder/model.go's PrefillLast handling) silently fell through to the ~9x-slower sequential
per-token path with nothing distinguishing "declined" from "crashed". Checked per layer because a
sliding-window layer's maxNWin is clamped to its own window and may stay under the limit even when
a global layer in the SAME model does not — mirrors the launch site's own per-layer maxNWin
computation in prefillCore exactly, so this can never decline a shape the launch would have run,
or miss one it would have failed.
```

## PrefillPath

Moved from `cuda/prefill.go` (the comment above `PrefillPath`) on 2026-10-09.

```text
PrefillPath (decoder.PrefillPathReporter) answers, at load, whether this model will get the batched
prefill — before a single request has been served. Batched prefill now covers int4 AND int8 bundles
(§C6); it still declines a native/f32 projection or a non-uniform/K=V geometry. Before int8 batched
prefill landed, a dense model loaded at int8int8 built a fully resident decode path (looked healthy,
decoded at 0.7× int4) but every prompt took the sequential per-token prefill — measured 1.73 s vs
0.19 s on a 300-token prompt (9×), 20× the CPU (4.56 vs 0.22 CPU-s), no compute hotspot: the executor
spin-waiting through 300 sequential launches instead of one pass.
```

## PrefillPath: rows per pass

Moved from `cuda/prefill.go` (the comment in the body of `PrefillPath`) on 2026-10-09.

```text
Say ROWS PER PASS, not "one pass". The report is read as a promise about how a long prompt
is ingested, and a prompt past the chunk width is now several weight-stationary passes over
the positional KV rather than one — same numbers (TestPrefillChunked_bitIdentical), bounded
scratch. Claiming "one pass" was what let the O(M·inter) OOM decline hide behind a green
startup line for every prompt long enough to matter.
```

## fastPrefillEnabled

Moved from `cuda/prefill.go` (the comment above `fastPrefillEnabled`) on 2026-10-09.

```text
fastPrefillEnabled reports which fast prefill kernels are selected, PER LEVER.

DEFAULT ON above fastPrefillFloor (512 prompt tokens) as of 2026-09-05, and `=0` is a complete
undo. It became a default only after §3's fidelity gate passed at every cell at or above that
floor, on BOTH bench models, against a CPU reference with f32 weights and f32 activations:

	S  K=512 SHIPS (agree 93.91% vs exact 93.75%, flips 5 v 5, KL 0.03208 v 0.03321)
	S  K=1024, K=3900 SHIP — fast is CLOSER to the reference than exact on all three criteria
	D7 K=512 SHIPS (agree 86.41% v 86.56%, flips 16 v 16), D7 K=1024 SHIPS

It FAILS at K=256, which is why the floor exists and why it is 512 and not lower; that cell
stands on the record and is not withdrawn (docs/completed/task-prefill-gap.md §3, and
measurements/prefill-l2l3-phase3-2026-09-05.md).

The exact path — attn_batched and gemv_w4a8_rn — remains selectable, remains bit-identical to
the M=1 decode kernels, remains what spec-decode verify and the parity gates run, and still
serves every shape the fast kernels decline (hd not in {64,128}, M below the mma row floors,
int8 bundles, K%32 != 0, and every prompt below the floor).

THE LEVERS ARE SEPARATELY SELECTABLE ON PURPOSE. L2 (fused attention) and L3 (tensor-core GEMM)
touch DIFFERENT categories of prefill — attention and the weight term — and §5 requires each to
be measured against the exact path alone before the two are measured together, "so the end-to-end
number has an attribution". §3.1 needs the same split for a different reason: if the combined
fast path ever scores worse than exact under the fidelity gate, the first question is WHICH
kernel, and an all-or-nothing flag cannot answer it.
```

## prefillCore: forceExactKernels, DeltaNet

Moved from `cuda/prefill.go` (the comment in the `forceExactKernels` expression of `prefillCore`) on 2026-10-09.

```text
Gated-DeltaNet: the fast levers' projection error feeds the recurrent state and compounds token after
token — measured on Qwen3.5-9B, 561 tokens: cosine 0.994 and different greedy continuations with them,
bit-identical to decode without (docs/tasks/task-cuda-deltanet-prefill-2026-09.md).
```

## prefillCore: forceExactKernels, E-model

Moved from `cuda/prefill.go` (the comment in the `forceExactKernels` expression of `prefillCore`) on 2026-10-09.

```text
Gemma 4 E-model (S9 on CUDA part A): the fast levers have no fidelity evidence on this family, and the first served read showed it. On the real E2B, a
2,170-token prompt above the 512-row floor took the fast kernels and first differed from the sequential path at generated token 10 (' and' 0.19 against
' issues' 0.19, the latter outside the sequential top 3: not a near-tie under the registered rule), while the same prompt on the exact kernels matched
the sequential path in every logprob (docs/tasks/task-multimodal-support-2026-10.md, G3p). So an E-model batches on the exact kernels, bit-identical to
decode at every length, until the levers pass a fidelity gate of their own.
```

## prefillCore: forceExactKernels, DeepStack

Moved from `cuda/prefill.go` (the comment in the `forceExactKernels` expression of `prefillCore`) on 2026-10-09.

```text
Qwen3-VL's DeepStack prefill (S10 on CUDA, G-S10g): the first real reading failed on table.png (986 rows, past the 512-row floor): the last-row logits read 0.9125 against the CPU prefill with the fast levers
on, 0.9741 with them off (and 0.9711 with them on but no sets added), under a text control minimum of 0.9539. The levers have no fidelity evidence on image rows carrying DeepStack sets, so such a pass runs
the exact kernels, as the E-model does, until they have.
```

## prefillCore: scratch free list

Moved from `cuda/prefill.go` (the comment in the body of `prefillCore`) on 2026-10-09.

```text
--- M-sized scratch (device), freed at the end.

The free list and its defer are registered BEFORE the first allocation, and each buffer
joins the list as it is created (audit C-24). Allocation PANICS on OOM per
gpu.NewBufferLenOf's contract, and at M=3000 this is hundreds of MB, so a partial
allocation is the expected failure on a nearly-full card — not a rare one. Building the
list first and deferring after (the previous shape) freed nothing at all when allocation
#10 of 17 panicked, because the defer had not been registered yet. That leaked only
because the panic used to kill the process anyway; now that runJob recovers it into a
decline, the leak would be real, repeatable, and would push the NEXT prompt closer to OOM.
```

## prefillDefaultChunk

Moved from `cuda/prefill.go` (the comment above `prefillDefaultChunk`) on 2026-10-09.

```text
prefillDefaultChunk is the default number of prompt rows per batched pass.

WHY CHUNKING EXISTS. prefillCore's scratch is O(M·inter): at Qwen2.5-7B's inter=18944 it is
~278 KB per row, so an 8k prompt asks for 2.28 GB on top of the weights and the KV. MEASURED on
this box (RTX 2070 SUPER, 8 GB, qwen2.5-7b-instruct-q4_k_m at int4, ResidentContext=8192, 1.96 GB
free after load — docs/measurements/prefill-chunking-2026-09-04/):

	M=512   batched     2.776 ms/token      M=512   sequential  12.485 ms/token
	M=2048  batched     3.543 ms/token      M=2048  sequential  13.850 ms/token
	M=4096  batched     4.440 ms/token
	M=8012  DECLINED — cuMemAlloc_v2 CUDA_ERROR_OUT_OF_MEMORY on the 607 MB gate buffer

So the batched path passed its LOAD-time report ("batched (one weight-stationary CUDA pass)") and
then declined every prompt long enough to need it, falling back — silently, since
residentPrefillSeed discards the decline — to the ~4.5× slower per-token loop. The failure grows
with the prompt: it is exactly the deep-context cell where TTFT matters most that lost the path.

WHY 512. The per-token cost above is a + b·(average attended keys), and attention is charged per
position against its own prefix whatever the chunking, so chunk size buys nothing there — it only
sets how many times each weight is re-read. Fitting the three batched points gives a ≈ 2.52
ms/token of weight+glue work and b ≈ 1.0 µs/key; at 512 rows each weight is already amortized
512-fold, which is within a hair of the M→∞ limit, and the scratch is ~146 MB rather than 2.3 GB.
A 2048-token prompt therefore costs the same chunked as it did in one pass (predicted 3.54 vs
measured 3.543 ms/token), so this is not a trade against the lengths that already worked.
GOINFER_PREFILL_CHUNK overrides it (0 or unset = this default).
```

## prefillImageDefaultChunk

Moved from `cuda/prefill.go` (the comment above `prefillImageDefaultChunk`) on 2026-10-09.

```text
prefillImageDefaultChunk is PrefillImageLast's own row budget — separate from
prefillDefaultChunk because it prices a BOUNDED, KNOWN quantity (a real image's token budget —
Gemma 3: 256, Gemma 4: up to 1120, docs/multimodal.md — plus whatever chat text accompanies it)
rather than an open-ended prompt, and because PrefillImageLast never chunks at all (a
bidirectional image block split across a chunk boundary is unverified — its own doc comment),
so M > chunk is an outright decline, not a retry at a smaller width. 2048 is not a fresh guess:
prefillDefaultChunk's own measurement table above already measured M=2048 batched on THIS box
(RTX 2070 SUPER) at Qwen2.5-7B — inter=18944, the worst-case width in that table — at 3.543
ms/token with no OOM (the table's own OOM point is M=8012, four times higher). A card where
2048 IS too wide still declines cleanly: prefillCore's own OOM handling wraps a real device OOM
as errPrefillDeclined/errPrefillOOM regardless of caller, and GenerateVL's fallback reuses the
already-computed vision features rather than re-running the tower — an overly optimistic
default costs one wasted allocation attempt, never a crash or a wrong answer.
GOINFER_PREFILL_IMAGE_CHUNK overrides it (0 or unset = this default).
```

## PrefillLast

Moved from `cuda/prefill.go` (the comment above `PrefillLast`) on 2026-10-09.

```text
PrefillLast (decoder.Prefiller) ingests a whole prompt in ONE weight-stationary pass and returns the
logits for the last token — the batched (M=len) counterpart of the sequential ForwardNoLogits loop.
It fixes the ~128-token Ollama crossover: goinfer's sequential prefill reads every weight once PER
PROMPT TOKEN (weight-bandwidth-bound at ~6 ms/token), while this reads each weight once for all M
tokens (the gemv_w4a8_batched amortization). The K/V it writes is BIT-IDENTICAL to the sequential
path row-for-row (every batched kernel is the M=1 kernel with an M dimension, per-row math verbatim),
so decode from the last prompt token stays byte-identical.

It handles the plain dense unfused forward only (Llama/Qwen2/Mistral-class): rmsnorm→Q/K/V→rope+kv→
causal windowed attention→o-proj→rmsnorm→gate/up→swiglu→down. Anything it does not cover — MoE, the
Gemma parallel dense‖MoE, sandwich norms, per-head QK-norm, K=V (Gemma), int8 weights, non-uniform
per-layer geometry, or a prompt past the KV cap — returns an error so decoder/model.go falls back to
the sequential KV-only prefill (which is correct for every family). Uniform-only is enforced against
layer 0; a non-uniform family trips the guard and declines rather than reading a wrong stride.
PrefillLast ingests a whole prompt in one batched pass, returning the last token's logits.
```

## PrefillImageLast

Moved from `cuda/prefill.go` (the comment above `PrefillImageLast`) on 2026-10-09.

```text
PrefillImageLast satisfies decoder.ResidentImagePrefill: PrefillLast's Gemma-3 twin for a turn
whose prompt carries a bidirectional image block [imgStart,imgEnd) (docs/multimodal.md's image
path — decoder/kvcache.go's SetImageBlocks/attendHi is the CPU reference this must match
bit-for-bit). embeddings already carry the spliced vision features at that range — the same
"embed by vector" convention PrefillLast already uses; there is no GPU-side embedding table to
bypass. startPos is always 0 today (GenerateVL's image-prefill branch only ever runs on a fresh
turn) but is not hardcoded, for symmetry with PrefillLast and to not foreclose a future
prefix-reuse combination.

v1 REQUIRES the whole prompt (the image block included) to fit in ONE weight-stationary pass —
prefillChunked's positional-KV chunking is unverified (likely unsafe) for a bidirectional block
split across a chunk boundary — so this never chunks, unlike PrefillLast. The row budget for
that one pass is prefillImageChunkRows (2048 by default), NOT prefillChunkRows's 512: real image
token budgets are bounded and known (see prefillImageDefaultChunk's own doc comment), so this
prices "one image plus its surrounding chat text" rather than an open-ended text prompt. Any
decline (kernel unavailable, invalid range, or M past the chunk width) wraps errPrefillDeclined;
the caller (decoder.GenerateVL) treats that as "fall through to the CPU-prefill+UploadKV bridge,
unchanged".
```

## PrefillImageBlocksLast: OOM budget

Moved from `cuda/prefill.go` (the comment above `PrefillImageBlocksLast: OOM budget`) on 2026-10-09.

```text
N-41 (docs/audit-2026-09-10.md): unlike prefillChunked, this call cannot retry at a
smaller width — a bidirectional image block has to land in one pass, and errPrefillOOM
here is a function of THIS M, not of chunk, so a smaller chunk would not change M or
rescue this attempt. But leaving prefillChunkCap unlearned means the pre-check above
keeps admitting up to prefillImageChunkRows() on every future image turn too, so the
SAME OOM repeats on every one — and per prefillChunkCap's own doc comment
(cuda/resident.go), repeatedly driving the context to CUDA_ERROR_OUT_OF_MEMORY risks the
context afterward launching kernels that "return SUCCESS and execute NOTHING", not just
wasted retries. Halve the budget for the NEXT image call, same floor prefillChunked
already uses, so a smaller image is caught by the cheap pre-check instead of repeating
the same real OOM.
```

## cudaDeepstackPrefillOn

Moved from `cuda/prefill.go` (the comment above `cudaDeepstackPrefillOn`) on 2026-10-09.

```text
cudaDeepstackPrefillOn is the production switch for the resident DeepStack prefill, ON BY THE OWNER'S DECISION of 2026-10-09 ("just turn it on"), over a registered FAIL. The record (docs/tasks/task-multimodal-support-2026-10.md):
G-S10g passed on all four images against a text control; G-S10j (one prompt per image, bars set without a noise floor) read FAIL and left it off; the first pass of the 896-pixel investigation found the chunk boundary exact, no
localized fault in the per-layer residuals, and that one equivalent CPU kernel moves the same statistic as much as G-S10j's "defect"; G-S10k (64 units, the margin taken from an A/A in the same run) read FAIL on its per-image
guard alone: pooled the resident prefill is within the margin (+0.0007 [-0.0044, +0.0060] against 0.0050), and on the 896-pixel image it is farther in logits cosine by about 0.011 while closer in KL and argmax agreement.
So the one known residual is a small cosine difference on that image, not a defect located anywhere, and what it buys is a prefill about 13x faster than the CPU prefill and upload. Off, a Qwen3-VL image turn takes the
CPU prefill and the upload; setting this false restores that path exactly. The gates set it explicitly and put it back.
```

## HiddenLast

Moved from `cuda/prefill.go` (the comment above `HiddenLast`) on 2026-10-09.

```text
HiddenLast (decoder.ResidentHiddenLast) is prefillChunked's twin for G4
(docs/tasks/task-gpu-paths-2026-09.md, embedding requests): the resident batched pass ingests the
whole sequence exactly as PrefillLast does — same chunking, same K/V it writes, bit-identical
per-row math — but the final chunk's tail computes the last row's post-final-norm hidden state
instead of logits, and never dispatches the LM head at all: an embedder never needs it
(decoder/embed.go's HiddenLast doc comment), and the head is the single most expensive matmul
in a forward. startPos is always 0 for every caller today (HiddenLast has no prefix reuse), but
this takes it anyway so it can share prefillChunked's chunk-boundary bookkeeping unchanged.
```

## ResidualAll

Moved from `cuda/prefill.go` (the comment above `ResidualAll`) on 2026-10-09.

```text
ResidualAll (decoder.ResidentResidualAll) returns every row's residual stream after the last layer and before the final norm: HiddenLast's twin for a head that reads
every position (D11's follow-up; the Clef joint head). The passes are the same chunked ones prefillChunked runs, except that every chunk keeps its rows instead of
discarding them, and all of them take the exact kernels (every tail but an ordinary single-row prefill does; a DeltaNet model always does). The rows are the f32 residual
the pass already downloads, so nothing here is quantized and the decoder's own f32 final norm finishes the job.
```

## PrefillLastNArgmax

Moved from `cuda/prefill.go` (the comment above `PrefillLastNArgmax`) on 2026-10-09.

```text
PrefillLastNArgmax is the spec-decode VERIFY primitive: the same batched pass, returning only
each row's argmax token id — which is all the accept decision needs.

It exists because PrefillLastN's per-row tail re-reads the LM head's weights ONCE PER ROW
(~389 M params, ~195 MB at int4), measured at 1.046 ms marginal per row against a 0.934 ms
single-row head — no amortization at all, in the one place the batched pass exists to provide
it. This tail instead runs ONE batched final-norm, ONE batched head GEMV over all M rows, and M
argmax reductions, then reads back 4 bytes per row instead of 608 KB.

LOSSLESSNESS: the accept decision compares the drafted token against the target's argmax, so
only the ARGMAX must match the sequential path — not the logits bit-for-bit. That is a strictly
weaker requirement than PrefillLastN's, and it is what makes batching the head admissible at
all. TestPrefillLastNArgmax_matchesPerRow gates it.
```

## PrefillSeedArgmax

Moved from `cuda/prefill.go` (the comment above `PrefillSeedArgmax`) on 2026-10-09.

```text
PrefillSeedArgmax satisfies decoder.ResidentSeedArgmax: the same batched forward and the same
batched capture, but the head runs over ONE row.

The block-spec prompt seed asked for M rows of argmax and read only the last. At vocab 151,936 a
2048-token prompt therefore allocated 1.24 GB of VRAM for the batched logits, a 1.24 GB host
slice and a 1.24 GB D2H, ran the head GEMV over 2048 rows and a single-threaded host argmax over
311M floats — for one token id. logitsB is grow-only, so each longer prompt also abandoned its
predecessor; a 4096-token prompt on an 8 GB card OOM'd inside the executor, and per backend.go's
A13 a context driven to refusal and kept in use can afterwards launch kernels that "return
SUCCESS and execute NOTHING" (audit-2026-09-02 C-12).

tailLastLogits, not a new kernel path: it ALREADY heads the last row only, and the seed's other
requirement — the batched capture — comes from the layer loop either way. One row of logits is
vocab floats (~0.6 MB) against M x vocab.
```

## PrefillSeedArgmax: cancellation gap

Moved from `cuda/prefill.go` (the comment above `PrefillSeedArgmax: cancellation gap`) on 2026-10-09.

```text
context.Background(), and it is a KNOWN GAP of the same class PrefillLast just closed:
decoder.ResidentSeedArgmax carries no context, and this one DOES ingest a whole prompt, so a
cancelled block-spec seed runs to completion. It is not fixed here because the fix is another
interface change on a different seam, and doing it silently as a side effect of this one is
how a surface changes without anyone deciding to. Filed with the P20 cancellation item.
```

## prefillCore: batched hidden-state capture

Moved from `cuda/prefill.go` (the comment in the body of `prefillCore`) on 2026-10-09.

```text
BATCHED HIDDEN-STATE CAPTURE (P10). The per-token seam (capVec) syncs and
downloads once per TAP PER TOKEN; a block drafter needs the taps for every token
the verify commits, so on this path that is 5 taps x M tokens of stalls. Here the
residual for all M rows is already in xB, so one download per tap covers the whole
block — 5 downloads per verify instead of 5*M.
```

## prefillCore: tailResidualAll download

Moved from `cuda/prefill.go` (the comment in the body of `prefillCore`) on 2026-10-09.

```text
The ONLY tail that reads the whole residual on the host (audit R-24): the argmax and all-logits heads below read xB on the device, and
the last-row tails need one row, copied device-to-device. Downloading all M rows for them was M*hidden*4 bytes (23 MB at M=2048 on a
2816-wide model) of pageable D2H after every pass, for nothing.
```

## prefillCore: OOM as decline

Moved from `cuda/prefill.go` (the comment in the body of `prefillCore`) on 2026-10-09.

```text
An OOM inside the job arrives as a recovered panic (runJob, audit C-24) carrying aikit
MustBuf's "device allocation failed" message. For prefill specifically that is a DECLINE, not
a request failure: the sequential per-token path needs no M-sized scratch and will serve this
prompt. Match that OOM SENTINEL, not any "panicked" (audit R-20): a future programming-bug
panic in the batched path must surface as a real error, not be silently absorbed into the
~9×-slower sequential path. Errors that are already declines (static guards, checkCap) keep
their own wrapping.
```

## prefillCore: final-logit softcap

Moved from `cuda/prefill.go` (the comment in the body of `prefillCore`) on 2026-10-09.

```text
Final-logit softcap (Gemma) — host-side, exactly as step(). No-op (0) for the dense families
this path serves, but kept so the contract matches Forward if a softcapped dense arch appears.
SKIPPED for tailHiddenLast: outs holds a HIDDEN STATE there, not logits, and Gemma's softcap
is a logit-only transform — applying it here would silently corrupt every G4 embedding on a
softcapped family.
```

## prefillCore: qTempRows

Moved from `cuda/prefill.go` (the comment in the body of `prefillCore`) on 2026-10-09.

```text
qTempRows (Ministral 3, FeatAttnTemp): this launch covers M rows at different
positions (startPos+m), so each row's post-RoPE query scale comes from a table
built on the host with launchToken's own float64 expression. That keeps a batched
row bit-identical to decode (audit-2026-09-10 G-11). Not carried into the m-RoPE
branch above: rope_kv_mrope_batched has no attention temperature, and no family
needs both today (Ministral 3 has no m-RoPE, Qwen2.5-VL no temperature).
```

## prefillCore: sinkArg

Moved from `cuda/prefill.go` (the comment in the body of `prefillCore`) on 2026-10-09.

```text
N-10: r.sinkArg(l), not ArgNull(). The decode launches thread the gpt-oss
learned sink through and this one hard-coded null — unreachable today only
because every gpt-oss model is MoE and MoE declines batched prefill, which
is a property of a DIFFERENT check and not something this call site should
depend on.
```

## prefillCore: final norm and head

Moved from `cuda/prefill.go` (the comment in the body of `prefillCore`) on 2026-10-09.

```text
Final norm + LM head, per row — copy xB[m] into the M=1 scratch and reuse the exact Forward
tail, so each row's logits are bit-identical to a sequential Forward at position startPos+m
(given identical residual, which the KV/logits gate checks). allLogits=false heads only the
last row (the crossover-fixing PrefillLast); allLogits=true heads every row (verify). Drain
the layer launches first: they run on r.stream, and the DtoH below is not ordered after it.
```

## bGemvB and gemmMMAMinRows: batched int8 and mma row floor

Moved from `cuda/prefill.go` (the comment in `bGemvB` and above `gemmMMAMinRows`) on 2026-10-09.

```text
Batched W8A8 (§C6). One warp per output row (8 warps/block), same layout as doG's int8
GEMV (wt.ws = per-row f32 scale, K/4 int words). Bit-identical to gemv_w8a8_fwd by
construction — exact int32 accumulation, so tiling M cannot change any element.

gemmMMAMinRows: tensor cores lose below a warp's worth of rows, and §4 L3 pre-registers
gemv_w4a8_rn — the EXACT path — as the M<16 path for that reason. 16 is two m8n8k16 M-tiles.
```

## BuildResident: KV slots partially granted

Moved from `cuda/backend.go` (the comment in the body of `BuildResident`) on 2026-10-09.

```text
S18 (docs/tasks/task-multimodal-support-2026-10.md): checkKVFits sized the slot count against the free VRAM read before the build's own scratch, with
the margin as the only slack (Gemma 3 4B on the 8 GB card: four slots fit with 1.2 MB to spare), so the last slot can miss by the scratch. Keep the
slots that fit instead of dropping the whole resident to the CPU; the first slot is the build's own and still declines.
```

## BuildResident: dense-FFN scratch

Moved from `cuda/backend.go` (the comment in the body of `BuildResident`) on 2026-10-09.

```text
DENSE-FFN scratch, and only if the model HAS a dense FFN. A model whose every layer is
routed reports intermediate_size 0 — Qwen3.6-35B-A3B's config omits the key entirely —
and these become 0-byte allocations, which this driver rejects as "invalid length". The
dense branch of segBFFN is unreachable for such a model, so the buffers are simply never
needed; allocating them anyway was the only thing standing between it and residency.

Nothing hit this before because every previously-resident MoE (Mixtral, GLM, Mellum) has
a real intermediate_size — dense prefix layers or a genuine dense width. Note that the
tiny qwen3_5_moe fixture does NOT reproduce it either: it carries intermediate_size 128
because HF's config defaults one in, so the fixture is less pure-MoE than the model it
stands for. That is a fixture-fidelity gap, recorded rather than silently fixed here.
Scratch is sized to the WIDEST dense layer: an E-model's layers differ and the launches take each layer's own width (cudaLayer.ffnI).
```

## BuildResident: moe_route warm-up

Moved from `cuda/backend.go` (the comment in the body of `BuildResident`) on 2026-10-09.

```text
A9-FIX: pay the DEFERRED first-launch reservation BEFORE the free reading that sizes the
cache, so the cap is correct by construction rather than covered by a margin.

moe_route declares per-thread local scratch (two float[MOE_MAX_E]), and the driver backs
local memory for the device's occupancy the first time that kernel runs — not at module
load, which costs a measured 0 B. On an RTX 2070 SUPER that launch DEMANDS 289,013,760 B
and RETAINS 138,412,032 B. Both were invisible here: allocSlots reads free VRAM before any
kernel has run, so it sized the cache against memory that was about to be taken.

Forcing it here is strictly better than enlarging slotMarginBytes, which is the fix
everyone reaches for first. The peak is 2.09x the residual, and it is TRANSIENT: paying it
now, while ~3.8 GB is still free, means the free reading below sees only the 132 MiB that
is actually retained. A margin bump would have to reserve the 275.6 MiB peak permanently
to cover something needed for microseconds — and it would bury a named consumer inside an
unnamed constant, so the next kernel with per-thread scratch reopens it silently.

WHY BY NAME IS SAFE HERE, given that naming one member of a set is the sibling-drift shape:
the backing store is SHARED and sized by the largest kernel, measured — launching the whole
census gives a threshold and residual identical to moe_route alone, to the byte. So forcing
the maximum forces the pool for every kernel. That moe_route IS the maximum is not assumed:
TestKernelLocalMemoryCensus enumerates every entry point in every embedded module and fails
if any other kernel declares more, naming this site.

One kernel does: route_gptoss, at 4608 B/thread against moe_route's 4416. The census could
not see it until its module list was derived from kernels.go's embeds (audit-2026-09-10
G-13(b)); gptoss_act.ptx was one of the seven it missed. It is bound only on gpt-oss, the
model the expert cache exists for, so on exactly that path the larger pool was left to grow
on the first real token, after allocSlots had read free VRAM — by 8,388,608 B on the RTX
2070 SUPER (TestRouteGptOssGrowsPoolPastMoERoute; 0 B once both are forced). It is forced
below wherever it is bound. Forcing both is right whether the pool is sized by the max or
by the sum.

REGIME: `max` was measured with sequential single-stream launch, which is what goinfer
does. Concurrent streams would reopen whether the bound is max or a sum.
```

## BuildResident: per-32 fusion gate

Moved from `cuda/backend.go` (the comment in the body of `BuildResident`) on 2026-10-09.

```text
Their per-32 twins (fused_rms_qkv_g32 / fused_rms_gu_g32) take any of the per-32 weight kinds,
per projection. Losing fusion measured ~23% of the 1.5B's decode (docs/tasks/
task-int4-weight-quality-2026-09.md, lever 3); the per-32 kernels themselves ~3%. Same
structural exclusions as fuseQKV (no pre-norm, parallel block, MLA, gated q, MoE layers).
Size-gated (fusedG32MaxHidden): every block redoes the rmsnorm + per-32 quant, and that cost
grows with H while the launches it saves matter less as the GEMVs grow. Measured fused ÷
unfused: +10% at H=1536 (qwen2.5-coder-1.5b), −8% at 3072 (phi3-mini), −5% at 3584 (qwen2.5-7b).
```

## cudaKVSlotsRequest

Moved from `cuda/backend.go` (the comment above `cudaKVSlotsRequest`) on 2026-10-09.

```text
cudaKVSlotsRequest is how many resident KV slots a build asks checkKVFits for (MC1,
docs/tasks/task-concurrency-2026-09.md): the model's request (decoder.Model.ResidentKVSlotsRequest, already 1 for a
family with recurrent state), and 1 in two cases the decoder cannot see:
  - a Gated-DeltaNet resident (recurrent): its state is mutated in place and is not part of a slot. The decoder's
    hasRecurrentState already keeps these at one; this refuses to allocate for a second slot anyway;
  - expert streaming (MoECacheExperts): that cache takes whatever VRAM is left after everything pinned, so every
    extra KV slot is expert-cache slots lost — the trade resolveCtxCapFit already declines to make with the context
    (48% decode lost on gemma4-26b when KV grew into the cache's share).
```

## BuildResident: decline reason

Moved from `cuda/backend.go` (the comment in the body of `BuildResident`) on 2026-10-09.

```text
The reason is printed UNCONDITIONALLY, not behind a debug flag. Declining moves the whole
forward to CPU, and v0.10.0's contract is that the runtime names the reason when it is not on
the fast path — a reason nobody can see does not satisfy that. It cost a 307-second 26B run to
learn "the experts do not fit VRAM", which the runtime knew at the moment it declined. One
line at load, not a debug stream: it is the same "zero means either" shape as a skip census
that prints nothing — a silent decline and a successful build look identical from outside.

Since 2026-09-25 the reason travels as a typed decline (decoder.DeclineResident): decoder's load path
prints it once and records it as the model's ResidentDecline, which DecodePath and `serve check`
report — before, it reached stderr only, and the model's recorded reason was a generic string.
```

## BuildResident: DeltaNet gated softmax case

Moved from `cuda/backend.go` (the comment in the body of `BuildResident`) on 2026-10-09.

```text
The same family's GATED softmax layer (qwen3_5/qwen3_5_moe/qwen3_next — NOT every
dnetOK family: Olmo Hybrid's full-attention layer is olmo3's plain scheme instead,
dnAttnGate=false, and falls through to default below — G5,
docs/tasks/task-gpu-paths-2026-09.md. This used to be a bare `dnetOK` case, silently
wrong the moment a non-gated hybrid family reached residency, since
Qwen35ResidentParams hardcoded attnGate=true). Its weights live off lw.QProj (the
family keeps them in its own struct), and q_proj is DOUBLE WIDTH — [query ‖ gate]
per head.
```

## BuildResident: o_proj bias

Moved from `cuda/backend.go` (the comment in the body of `BuildResident`) on 2026-10-09.

```text
Captured INDEPENDENTLY of QBias: the two travel together in Qwen2 but not in general —
GPT-2 carries an o_proj bias with no q/k/v bias, so folding this into the branch above
would silently drop it for exactly the families FeatOutBias exists for.

N-34: this used to name gpt-oss as that example, saying it carries an o_proj bias
"with no q/k/v bias at all". It does carry them — gptoss_safetensors.go loads q/k/v
bias as REQUIRED (a missing one is an error) and decoder/testdata/gptoss_tiny.gguf
holds attn_q/k/v.bias. The code was right and the comment wrong, which is the
dangerous direction: it invited exactly the fold it warns against.
```

## BuildResident: MoE cache slots default

Moved from `cuda/backend.go` (the comment in the body of `BuildResident`) on 2026-10-09.

```text
C′ step 2: device slots per layer — an LRU cache of nSlots experts (clamped [topK, nE]).
VRAM is nLayers·nSlots·perExpert, so more slots trades VRAM for fewer per-token DMAs.
GOINFER_MOE_CACHE_SLOTS=N requests N explicitly.

With caching ON and NO explicit request, ask for ALL experts and let allocSlots cap to
measured free VRAM. This reverses a default of topK, which was the worst possible setting for
the only situation this code runs in: at nSlots=topK the cache degenerates to fresh-loading
every routed expert every token — ~714 MB/token on the 26B, ~5 tok/s against ~17 at the 38
slots that fit. Nobody enables expert streaming to get the slow version of it, so the
conservative default was really deferring a VRAM decision that allocSlots already makes
properly: it measures free VRAM and caps-and-logs. An over-large request was never the hazard
it looked like either — post-C-24 an alloc panic on the executor becomes a DECLINE (→ staged
fallback), not a process kill.

REVERTED to topK (2026-08-11). Defaulting to "ask for all, let allocSlots cap to free VRAM"
was correct in intent and WRONG in practice, because allocSlots' cap is not the safety net it
looks like: its headroom is a flat `marginBytes = 384 MB` described as covering "the
greedy-argmax readback + driver overhead" — per-token costs — while what it must actually
leave room for is everything the forward allocates AFTER it runs, which scales with layers,
context and vocab. On the real 26B it capped 128 slots to 34 (3.4 GB of 3.8 GB free) and the
warm forward then died with cuLaunchKernel: CUDA_ERROR_OUT_OF_MEMORY.

Measured, same test, same box, only this default differing:
    GOINFER_MOE_CACHE_SLOTS=8  (this default) -> PASS 305s
    unset -> 128, capped to 34               -> FAIL 477s (OOM)

RAISED (2026-08-20) to a BOUNDED multiple of topK. This is the "change about defaults" the
note above parked — its safety precondition (fix the margin, prove it on the 26B) was met by
A5/A7, and what remained was only that a default change should not ride along inside an
unrelated commit.

WHY A BOUND AND NOT THE "ask for all, cap to VRAM" THE ACCESSOR DOCUMENTS. Measured on the
real Qwen3.6-35B-A3B (nE=256, topK=8), sweeping slots/layer:

    8 (topK)      6.77 tok/s   ~0% hit   ~630 MB expert DMA/token
    48           10.09 tok/s   71.1%      265 MB
    76           10.32 tok/s   77.7%      205 MB

The knee is around 48: cutting DMA 630→265 MB bought 1.49x, cutting it 265→205 bought 1.02x.
Past the knee, slots consume GB of VRAM for nothing — and every extra GB is more exposure to
the deferred local-memory reservations that made the previous "ask for all" attempt OOM after
allocSlots had already capped. A bound gets the whole win with a fraction of the risk, which
"all" cannot claim.

8*topK sits just above the measured knee. It is a HEURISTIC from one sweep (plus the 26B's
"~17 tok/s at the 38 slots that fit"), not a derived constant: where the knee falls depends on
the model's routing entropy. Erring above it costs VRAM the cap will reclaim if it is short;
erring below it costs throughput nothing reclaims.

The floor stays topK — one token's routed set must be simultaneously resident — and allocSlots
still caps to measured free VRAM, so this can only ever ask for less than the reverted default.
```

## BuildResident: MoE cache slots request

Moved from `cuda/backend.go` (the comment in the body of `BuildResident`) on 2026-10-09.

```text
The request now comes from Options (--moe-cache-slots), and MoECacheSlotsRequest still
honours GOINFER_MOE_CACHE_SLOTS, so nothing that set the env var breaks.

NOTE the accessor documents 0 as "ask for all, auto-cap to VRAM". That default is NOT
taken here: unset leaves topK in place, exactly as before this branch. Raising it is a
SEPARATE decision — main's comment above states its own precondition ("fixing the margin
FIRST and proving it on the 26B"), which A5 (6091e7a) and A7 have now met — and a change
of default belongs in a change about defaults, not in one promoting env vars to flags.
G-07: `> topK` silently floored a request of topK or less. A request BELOW topK cannot
be honoured — one token's own top-k must fit — but it was neither honoured nor
refused: cacheSlots stayed at the 8·topK default, which on a small fixture is nE, so
every expert gets a permanent slot and slot ≠ expert only by first-admit order. A gate
asking for "fewer slots than experts" therefore got the identity mapping and
discriminated by routing luck. Refuse what cannot be honoured; honour the rest exactly.
```

## BuildResident: checkWeightsFit order

Moved from `cuda/backend.go` (the comment in the setup job of `BuildResident`) on 2026-10-09.

```text
M-02 (docs/audit-2026-09-02.md, docs/tasks/task-fit-to-hardware.md): CUDA had no memory-fit
check at all for the FIXED (non-expert) part of a resident load — a dense model whose
weights alone exceed free VRAM previously ran the full kernel-compile + upload sequence
before failing on whatever CUDA allocation happened to be the first one that didn't fit,
with a raw driver error rather than a clean decline. checkWeightsFit runs here, as early
as possible (right after the device exists, before any of that), and is checked BEFORE
this backend's other, more specific declines so an obviously-too-large model gets the
clearest message rather than whichever unrelated shape check happens to run first.
```

## BuildResident: device exhaustion

Moved from `cuda/backend.go` (the comment in the setup job of `BuildResident`) on 2026-10-09.

```text
THESE MODULE AND PIPELINE HANDLES DO NOT SURVIVE DEVICE EXHAUSTION. Read this before
adding any path that recovers residency after memory pressure.

Measured (A13, docs/QUEUE.md): once the device has been drained to exhaustion, a later
launch through a handle cached here returns SUCCESS and executes NOTHING — the output
buffer is left untouched and surfaces downstream as an all-zero result, e.g. a cosine of
exactly 0.000000. No CUDA call reports an error at any point, and free VRAM is back to
~7.3 GB by then, so neither an error check nor a memory check will catch it.

AND cuFuncGetAttribute WILL NOT DETECT IT. Queried across a poisoned and a clean run it
returns byte-identical valid values (maxThreadsPerBlock, numRegs, ptxVersion) because
those come from metadata that outlives the device code. A handle that answers is not a
handle that works.

What restores it, measured 3/3: re-loading the module and re-resolving the function
IMMEDIATELY BEFORE the launch. Re-loading earlier does not — allocations performed
between the load and the launch re-invalidate it.

goinfer does not hit this today only because BuildResident DECLINES on exhaustion
((nil,false,nil)) and cudaBackend.MatmulBT then runs linalg.MatmulBT with no CUDA at all.
That decline is a safety property, not an incidental fallback. Any future residency
recovery must RE-LOAD rather than reuse what is cached here.

THE FALSIFIER, which is the one sentence to carry away from all of this:

	ANY CHANGE THAT DRIVES THE DEVICE TO REFUSAL AND THEN CONTINUES USING THE SAME CONTEXT
	BREAKS THIS.

The whole tag rests on that single property holding across every shipped path — measured
path by path in docs/QUEUE.md A13 (prefill peaks 39.9x clear of the floor; unload frees
rather than exhausts; the cap search allocates nothing; the one path that DOES exhaust,
resident build, then declines and issues no CUDA). It is not a guarantee the type system
or any test can enforce for code that does not exist yet, so it is written here, where
someone adding a retry loop, an eviction-and-rebuild, or a "just try a smaller cache"
will be reading. If your change makes the device refuse and then keeps going on that
context, the failure will be silent zeros, not an error.
```

## BuildResident: rope binding

Moved from `cuda/backend.go` (the comment in the setup job of `BuildResident`) on 2026-10-09.

```text
kv_store and rope are NOT bound: the fused rope_kv below subsumes both (the Incr1
decode-fusion win). They were left bound and dead after that fusion shipped, JIT-compiled
into every model load and launched by nothing — see TestPipelineLint_boundKernelsAreLaunched.
GPT-J pairwise rotation (Cohere/Cohere2/Aya, GLM-OCR): bind the pairwise twin of rope_kv
(cuda/rope_pairwise.cu, same argument list and launch geometry) into the SAME field, so no
launch site changes and every other family still gets exactly the NeoX kernel above.
A load failure DECLINES — running the NeoX kernel on a pairwise model is silent-wrong from
position 1 (real R7B/Aya at int4: worst cosine -0.075/-0.041), so there is no fallback.
```

## BuildResident: PTX modules and NVRTC

Moved from `cuda/backend.go` (the comment in the setup job of `BuildResident`) on 2026-10-09.

```text
argmax_reduce lives in its own module (argmax.ptx), off glue.ptx, so the C-14 index tie-break
fix didn't force a glue.ptx regen. (moe.ptx, glue.ptx and gemv_fwd.ptx are the audited
artifacts, pinned at NVRTC 12.6.85 — audit R-26, restored per M-35/REGEN.md after drifting
to this box's ambient 12.9.86 in 610ce7f/23c46b13/5b443834. Every other production PTX,
including argmax.ptx, is built at whatever NVRTC was on hand when it was added — never
claimed otherwise.) See cuda/argmax.cu.
```

## BuildResident: batched prefill modules

Moved from `cuda/backend.go` (the comment in the setup job of `BuildResident`) on 2026-10-09.

```text
Batched prefill kernels (weight-stationary M=len path). Own module; the audited PTX is
untouched; the kernels come from prefill_batched.ptx. gemv_w4a8_batched's PTX is gone: its
only entry was bound and never launched, so an entire module was JIT-compiled on every model
load to feed a dead field (then left go:embed-ed but unused, until removed 2026-09-24). bGemvB
dispatches int4 to bRN (gemv_w4a8_rn) unconditionally.
```

## BuildResident: L2 default

Moved from `cuda/backend.go` (the comment in the setup job of `BuildResident`) on 2026-10-09.

```text
L2 fused prefill attention (docs/completed/task-prefill-gap.md §4 L2). DEFAULT ON above
fastPrefillFloor (512 prompt tokens) since 2026-09-05, when §3's reference gate passed on
CUDA; GOINFER_CUDA_FAST_PREFILL=0 or Options.ExactPrefill keeps the exact path. Own module — prefill_batched.ptx is untouched, the isolation pattern
attn_block.cu established. A load failure is not fatal: it leaves bAttnFused* zero and
every selection site falls back to attn_batched, which is the exact path anyway.
```

## BuildResident: gpt-oss router

Moved from `cuda/backend.go` (the comment in the setup job of `BuildResident`) on 2026-10-09.

```text
gpt-oss's ROUTER. moe.cu's moe_route means something different by "bias" —
it steers SELECTION only and takes the weight from the UNBIASED score, which
is right for DeepSeek/GLM and wrong here: gpt-oss softmaxes over the SELECTED
BIASED logits. Its own kernel comment calls running gpt-oss through moe_route
"plausible mixing weights that are simply not this model's — a silent quality
loss, not a crash". The kernel shipped 2026-08-18 and was never loaded, so
that is exactly what the resident path has been doing.
```

## BuildResident: router_f32 module

Moved from `cuda/backend.go` (the comment in the setup job of `BuildResident`) on 2026-10-09.

```text
router_f32 module: Gemma-4's own kernels, kept off the audited moe.ptx (this box's 12.9
NVRTC would rewrite every moe.ptx kernel). Pure-f32 router GEMV + per-expert-scale fold +
weightless out-of-place norm + scalar-scale. gemma4Dense (broader than gemma4Moe: ANY
gemma4 checkpoint, not just enable_moe_block ones) compiles the module too — scale_vec
(fScaleVec) is segB's dense-tail per-layer-output-scalar kernel, nothing router-specific
about it, and was previously compiled ONLY under the MoE-only gate even though every
dense gemma4 layer needs it (a real gap: dense resident decode never applied the
checkpoint's per-layer output scalar at all before this). The three genuinely
router-specific kernels stay gemma4Moe-only — a dense-only build has no router to serve.
```

## BuildResident: gpt-oss down bias

Moved from `cuda/backend.go` (the comment in the setup job of `BuildResident`) on 2026-10-09.

```text
The per-expert DOWN bias. Uploaded here with the others because it is a weight,
not an activation; consumed by gemv_w4a8_moe_wacc_bias INSIDE the router-weight
product. Never wired until 2026-08-31, which cost min cosine 0.75 vs 0.997.
```

## gemvFwdPTX

Moved from `cuda/kernels.go` (the comment above `gemvFwdPTX`) on 2026-10-09.

```text
gemvFwdPTX: the LLM-specific forward kernels — kv_store / rope_kv.

The generic quantized GEMVs it used to carry (gemv_w4a8_fwd, gemv_w8a8_fwd) moved to
aikit/gpu >= v0.4.0 in the Phase-1b blob-split and are loaded via
gpu.QuantGEMVPTX / Device.NewQuantGEMV (backend.go). The name is kept for continuity
with the .cu it is built from.
```

## gluePTX

Moved from `cuda/kernels.go` (the comment above `gluePTX`) on 2026-10-09.

```text
gluePTX: the per-token elementwise/attention glue — rmsnorm_quant, quant_vec, rope,
attention (GQA online softmax), swiglu_quant, residual. (argmax_reduce moved to argmaxPTX;
see below. moe.ptx, glue.ptx and gemv_fwd.ptx are the audited artifacts (R-26); M-35 found
all three had drifted to this box's ambient NVRTC 12.9.86 — restored to the pinned 12.6.85
via cuda/testdata/REGEN.md's procedure, per-kernel hash audit recorded there. Only 2 of the
16 kernels across the three files differ at all between the two toolchains (gemv_f32_a8,
glu_quant — both confirmed benign register/scheduling differences, not FMA-contraction: glu_quant
is FMA-linted, so its source already forces explicit ordering regardless of compiler discretion),
and the real MoE/glue resident-parity gates measure byte-identical before and after.)
```

## layernormQuantPTX

Moved from `cuda/kernels.go` (the comment above `layernormQuantPTX`) on 2026-10-09.

```text
layernormQuantPTX: layernorm_quant_batched / layernorm_f32_batched — the resident SigLIP vision
tower's LayerNorm (P6, docs/multimodal.md's "P6's other half"), a genuinely new primitive: no
text family in this codebase uses LayerNorm (mean+variance, weight+bias), only RMSNorm. Own
module for the same isolation reason as every kernel above — prefill_batched.ptx/glue.ptx stay
untouched. See cuda/layernorm_quant.cu / cuda/vision_encoder.go.
```

## f32tof16

Moved from `cuda/kernels.go` (the comment above `f32tof16`) on 2026-10-09.

```text
f32tof16 encodes an IEEE-754 float32 into a float16 bit pattern, byte-for-byte identical to
decoder.f32ToF16bits — the CANONICAL resident-backend f16 scale representation that metal/pack.go,
aikit/linalg, and the GOINFER_INT4_F16_SCALES CPU diagnostic all replicate. The int4 group scales
this encodes (resident.go `ws16`) MUST match every other backend's f16 scales bit-for-bit (audit
C-15). Round-half-up + gradual underflow to subnormals. The OLD version TRUNCATED (m>>13, no
rounding — a systematic downward bias) and flushed the whole e<=0 range to zero, diverging CUDA's
scales from metal/aikit/CPU. NOT RNE/saturate: a lone RNE here would re-introduce that divergence.
```

## ExtendContext

Moved from `cuda/drafter.go` (the comment above `ExtendContext`) on 2026-10-09.

```text
ExtendContext projects the fused context rows into every layer's K/V cache, at positions
[ctxLen, ctxLen+len(fused)), and advances ctxLen.

TWO THINGS HERE ARE NOT WHAT A DECODER LAYER DOES, and both are load-bearing:

	NO INPUT NORM. The context's K/V come from the fused rows RAW. blockTrunk.layer says it
	outright — "input_layernorm normalizes the BLOCK only... the reference passes target_hidden
	straight into k_proj/v_proj while only hidden_states goes through the norm. Norming both
	would be the natural-looking port and would be wrong."

	NO Q. The context supplies keys and values only; queries come from the block. rope_kv_batched
	rotates q and k together, so a q scratch is projected and discarded — cheaper than a second
	kernel, and it keeps the context on the SAME code path as the block, which is what stops the
	two drifting apart.

INCREMENTAL BY CONSTRUCTION: it appends at ctxLen rather than rebuilding. The CPU measurement
(TestDFlashDraftScaling) put the rebuild path at 2.4x the incremental one at a 1024-token
context, widening with length — a full drafter-prefill of the whole context on every block.
```

## ExtendContext: capacity

Moved from `cuda/drafter.go` (the comment in the body of `ExtendContext`) on 2026-10-09.

```text
SIZED TO THE TARGET'S CONTEXT, not need+512. The drafter is a 5-layer trunk, so its
whole K/V is layers*ctxCap*kvDim*2 floats — small. At need+512 the capacity froze at
len(prompt)+512 on the first call and never grew, so any greedy generation past
~500 tokens failed mid-stream, prompt-length independent, and blockspec returned that
as the generation's terminal error. Every committed block-spec test stops at <= 96
tokens, so none of them could reach it.
```

## DraftBlock

Moved from `cuda/drafter.go` (the comment above `DraftBlock`) on 2026-10-09.

```text
DraftBlock runs the trunk over one block and returns its output rows.

The block occupies absolute positions [ctxLen, ctxLen+M), directly after the context
ExtendContext wrote — so the attention sees ctx‖block exactly, with attn_block_full's uniform
nKeys giving every row all of it (the drafter's block is bidirectional; see attn_block.cu).

Every kernel here is the target's own except that one. The drafter is five layers of the same
Qwen3 shape, which is the whole reason this is assembly rather than new numerics.

THE NORM CONSTANTS ARE THE DRAFTER'S, NOT THE TARGET'S, everywhere. bRmsB/bNormF32B/the
prefill qk-norm all read r.eps and r.addOneArg(), and reusing them here would apply the
target's normalization to the drafter's weights — silent, plausible, and wrong. Each is
launched directly with the drafter's own eps and a plain (addOne=0) norm.
N-43 (docs/audit-2026-09-10.md, documented 2026-09-16, not a correctness bug): every d.r.bGemvB
call below (q/k/v/o/gate/up/down) reads r.aboveFastPrefillFloor(), which is r.passPromptLen —
state set once at the TOP of the TARGET model's prefillCore and never cleared afterward (see
its own doc comment on the field, cuda/resident.go). The drafter never calls prefillCore itself,
so a draft block's choice between gemm_w4a8_mma and gemv_w4a8_rn is decided by the LENGTH OF THE
TARGET'S LAST PROMPT, not by anything about this block — an undocumented coupling nothing here
used to explain. It costs no correctness: speculative decoding's own verify step accepts or
rejects each draft token against the target's real logits regardless of which kernel produced
the draft, so a "wrong" kernel choice here can only shift acceptance rate, never the final
output (the same "costs acceptance rather than correctness" framing NewResidentDrafter's own
N-08 comment already uses for a different upload-failure case).
```

## DraftBlock: attention scale

Moved from `cuda/drafter.go` (the comment in the attnBlock launch of `DraftBlock`) on 2026-10-09.

```text
N-11: the DRAFTER's scale, not the target's. d.r.attnScale is the target
model's, and this file's own rule three dozen lines up says the norm
constants are the drafter's everywhere — this launch was the exception.
Equal today only because DFlash drafters share head_dim 128 with Qwen3
targets at the default scale, and lossless verify makes any mismatch
perf-only: the drafter proposes worse tokens, verify rejects them,
acceptance falls, and every correctness gate stays green.
```

## SetBatchedCapture

Moved from `cuda/drafter.go` (the comment above `SetBatchedCapture`) on 2026-10-09.

```text
SetBatchedCapture arms the batched hidden-state seam on the target: the next PrefillLastN /
PrefillLastNArgmax records the residual for ALL its rows at each named layer.

The per-token seam (SetHiddenCapture) costs a sync and a download per tap PER TOKEN — measured
at 0.465 ms/token for five taps, ~2.3 ms per round at four accepted. This pays one download
per tap for the whole block, because the batched forward already has every row's residual in
one buffer.
```

## NewVisionEncoder: fused attention default

Moved from `cuda/vision_encoder.go` (the comment in the body of `NewVisionEncoder`) on 2026-10-09.

```text
DEFAULT since 2026-09-21 (owner override of the pre-registered rule; the served downstream
gate came back f_N=6/8 against the registered <=1 pass line — docs/measurements/
vision-tower-downstream-2026-09-21.md — with no defect found; the owner chose the 6.4x
speedup anyway, the same shape as R2's "shipped anyway" call). GOINFER_CUDA_VISION_ATTN=exact
restores the old kernel; =bm64 selects the other fused arm.
```

## ForwardPatches: scratch free list

Moved from `cuda/vision_encoder.go` (the comment in the body of `ForwardPatches`) on 2026-10-09.

```text
--- M-sized scratch (device), freed at the end. Mirrors cuda/prefill.go's own free-list
(audit C-24 there): the list and its defer are registered BEFORE the first allocation,
and each buffer joins the list as it is created via the af/ai closures below, so a later
allocation panicking (gpu.NewBufferLenOf's OOM contract) still releases every buffer
already made — not just the ones a flat post-hoc list would have caught. Before this fix
(M-18, docs/audit-2026-09-10.md) NONE of these 19 buffers, nor the queue above, were ever
released: ~265 MB per image call, held for the encoder's lifetime.
```

## VisionEncoder.Close

Moved from `cuda/vision_encoder.go` (the comment above `Close`) on 2026-10-09.

```text
Close tears down the device and its executor goroutine. The release runs ON the executor (via
reqCh, not called directly from Close's own caller goroutine) because CUDA contexts are
thread-affine — the context was created on the executor's locked OS thread (NewVisionEncoder),
so it must be released there too, mirroring cudaResident's own Close (cuda/resident.go: send the
teardown job, wait for its ack, THEN close reqCh — not the reverse). Before this fix (M-18,
docs/audit-2026-09-10.md) ReleaseAll ran directly on whatever goroutine called Close.
```

## moe_expert_major.go: file header

Moved from `cuda/moe_expert_major.go` (the comment at the top of the file) on 2026-10-09.

```text
R11/P20 — CUDA expert-major MoE prefill (docs/queue-performance.md P20, docs/tasks/red-october.md R11(b),
docs/measurements/p20-expert-locality-2026-09-21.md). GENERIC MoE PATH ONLY (Ly.isMoE, moeMLPPre/moeMLPPost's
own shape) — gemma4's parallel dense‖MoE FFN (gemma4MoeMLPPre/Post) is NOT covered here; that is a separate,
structurally similar extension, not attempted in this pass (M26, the model the locality measurement used,
therefore does NOT yet benefit from this — a real, named remainder, not an oversight).

MECHANISM: cuda/prefill.go's per-row MoE FFN loop admits/DMAs a routed expert once per (row, routing rank),
which the P20 locality measurement found re-fetches the SAME expert a mean of ~69x per 512-row chunk on M26
(a different model's C′ config, but the mechanism is model-independent) because only cacheSlots (as few as 10
on a tight card) can be resident at once and rows are visited in POSITIONAL order, not grouped by expert. This
reorders PROCESSING (not admission capacity) to group by expert: route every row in the chunk first, bucket
(row, rank) pairs by expert, then for each DISTINCT expert admit/DMA it ONCE and run every row assigned to it
before moving on. No new GEMV/SwiGLU/down-proj kernel: gemv_w4a8_moe(_wacc) and glu_quant are the SAME kernels
the per-row path already uses (moe.ptx is the audited 12.6.85 artifact and is NOT touched), called with a
dedicated one-entry "current slot" index instead of the live per-token r.slotIdx.

BIT-IDENTITY (required, not optional — the brief's own words): float addition is not associative, so folding
rank-major (today: for each row, rank 0..topK-1 in order) must produce the IDENTICAL term sequence as folding
expert-major would if experts happened to be visited in a different order per row (they do — expert iteration
order has nothing to do with any one row's own rank order). The fix, mirrored EXACTLY from the CPU precedent
(decoder/mlp.go's moeMLPBatch, P18): compute every (row, rank) expert output into a SEPARATE, rank-indexed
scratch buffer (moeScratch[rank], written via the UNCHANGED gemv_w4a8_moe_wacc kernel into a buffer that
starts at exactly zero — 0+x is exact in IEEE754, so this is not merely close to a plain write, it IS one),
then fold each row in RANK order at the end via topK sequential residual_batched launches (rank 0, then 1, ...
— same CUDA stream, so launch order IS per-row term order, for every row, regardless of what order the
experts were computed in). This is the CPU's own documented technique, restated for a stream-ordered GPU
instead of a single-threaded Go loop.

REFUSES (falls through to the per-row path, exactly as CPU's moeMLPBatch does) on: a shared expert
(Ly.hasShared — a different combine shape, not attempted here, matching CPU's own refusal), any per-expert
bias table (gpt-oss; its bias lookup is keyed by the LIVE per-token slot index this rewrite does not use in
the same way, and it was never verified against this scheme), and gpt-oss's own route kernel.
```

## prefillExpertMajorEnabled

Moved from `cuda/moe_expert_major.go` (the comment above `prefillExpertMajorEnabled`) on 2026-10-09.

```text
prefillExpertMajorEnabled reports whether the expert-major MoE prefill restructuring is on.
DEFAULT ON since 2026-09-21 (docs/measurements/p20-expert-major-m26-2026-09-21.md — the gemma4
extension, cuda/moe_expert_major_gemma4.go, measured 2.66x/2.50x/2.39x/2.26x at M=512/2048/4096/8012
on the real M26, sequential control unmoved within 0.3% noise), mirroring the CPU precedent this
build mirrors throughout (decoder/mlp.go's moeExpertMajor, P18: "GOINFER_MOE_EXPERT_MAJOR=0
restores the per-row path... an escape hatch and an A/B handle, not a user setting"). The generic
(non-gemma4) path's own measured win (Mellum2, 3.5-4.3%) is real but small — it rides the same
default because it is bit-identical and never measured a regression, not because it was the case
this default was chosen for.
GOINFER_CUDA_MOE_EXPERT_MAJOR=0 restores the per-row path.
```

## PerLayerCacheStatsForTest

Moved from `cuda/testhooks.go` (the comment above `PerLayerCacheStatsForTest`) on 2026-10-09.

```text
PerLayerCacheStatsForTest returns hits/misses for every MoE layer individually, in layer order
(dense layers report 0/0) — the per-layer breakdown CacheStatsForTest's sum discards. This is
the instrument `docs/tasks/task-freetoken-techniques.md` Lead 4 asks for: is C′'s per-layer slot
budget wasted on some layers and starved on others, or is demand actually even across layers?
```

## flashDecodeDefaultMinKeys

Moved from `cuda/flash_decode.go` (the comment above `flashDecodeDefaultMinKeys`) on 2026-10-09.

```text
flashDecodeDefaultMinKeys is the attended-span floor for the lane: below it the exact
path runs. Set from the served forced-on ladder (docs/measurements/attn-decode-fa-served-2026-09-20.md): the 0.5B and
gemma3-1b lose 3-9% up to 1024 keys and win from 2048, so 2048 is the lowest floor with no measured regression.
```

## moe_expert_major_gemma4.go: file header

Moved from `cuda/moe_expert_major_gemma4.go` (the comment at the top of the file) on 2026-10-09.

```text
Gemma-4 extension of cuda/moe_expert_major.go's expert-major restructuring (R11/P20, this is the
item docs/queue-performance.md's P20 entry names as "OPEN, ORPHANED: the gemma4-specific extension
this needs to actually reach M26" — see docs/measurements/p20-expert-major-2026-09-21.md).

gemma4MoeMLPPre/Post (cuda/resident.go) is a PARALLEL dense‖MoE FFN, structurally different from
the generic moeMLPPre/Post the other file covers:
```

## graphs_safe.go: file header

Moved from `cuda/graphs_safe.go` (the comment at the top of the file) on 2026-10-09.

```text
CUDA graph replay (r.graphs) is ~1.4–1.7× faster but is BIT-EXACT to live launch only under
EXCLUSIVE_PROCESS device tenancy or active CUDA MPS. Under DEFAULT compute mode (time-sliced
multi-context sharing) it silently mis-runs on this Turing box — proven by an MPS A/B (MPS-off
diverges, MPS-on bit-exact ×10; docs/cuda-graphs-investigation.md §5.1). The backend must be
"byte-identical or decline, never silently mis-run", so graphs are admitted only under a
driver-enforced safe condition and then confirmed with a startup self-test.
```

## graphsSelfTest: reset first

Moved from `cuda/graphs_safe.go` (the comment in the body of `graphsSelfTest`) on 2026-10-09.

```text
RESET FIRST, or this comparison is invalid for a recurrent model. The test's premise is
"same input twice ⇒ same logits", which holds for attention because writing K/V at
position 0 is idempotent — but a Gated-DeltaNet mixer ADVANCES {conv ring, matrix state}
on every call, so the second run starts from state the first run left behind and the
logits differ for reasons that have nothing to do with graph capture.

Without this the self-test reported "graph replay diverged from live" for every DeltaNet
model and graphs were declined on a false positive. Reset is a no-op for every other
family, so the attention path is unchanged.
```

## admitGraphs: DeltaNet

Moved from `cuda/graphs_safe.go` (the comment in the body of `admitGraphs`) on 2026-10-09.

```text
NOTE: this used to decline outright for Gated-DeltaNet models, on the reasoning that the
mixer's buffers ARE the per-token recurrent state and so could not be captured. That was
wrong, and measurably so: a graph replay reads CURRENT buffer contents — which is precisely
why the MoE routing, which changes every token, already flows through a captured segC. What
matters is that the POINTERS and the launch geometry are fixed, and the mixer's are: it has
no rope, no attention and no positional uniform at all. captureGraphs now takes mixer+FFN-pre
as one segment, making a DeltaNet layer the most graph-friendly kind in the runner.
```

## softcapParallelMin

Moved from `cuda/softcap.go` (the comment above `softcapParallelMin`) on 2026-10-09.

```text
softcapParallelMin is the vocabulary size above which splitting the softcap across cores pays.
MEASURED (this box, GOMAXPROCS 16, RTX 2070 SUPER host, 2026-08-12) — the crossover is real and
the small end is a LOSS, which is why there is a threshold rather than an unconditional fan-out:

	vocab      serial    parallel   speedup
	  8 192   45.7 us    48.2 us     0.95x   <- slower
	 32 768  166.3 us   117.4 us     1.42x
	 65 536  422.9 us   207.9 us     2.03x
	131 072  747.5 us   371.9 us     2.01x
	262 144    1.47 ms   639.8 us    2.30x   <- Gemma 3/4
```

## applySoftcap

Moved from `cuda/softcap.go` (the comment above `applySoftcap`) on 2026-10-09.

```text
applySoftcap applies softcap·tanh(x/softcap) elementwise and in place to a logit vector.

This runs on the SAMPLING path only. ForwardArgmax reduces the argmax on-device and reads back
4 bytes, never materialising the logit vector, so greedy decoding does not pay this at all — the
cost lands exactly on the path that also does the ~1 MB readback.

BIT-IDENTITY IS STRUCTURAL, not argued. Every output element is a pure function of the single
input element at the same index: there is no reduction, no accumulation, and therefore no ordering
or reassociation freedom for the split to exercise differently. Splitting the range cannot change
a bit, whatever the worker count or the split points. That is why this stays float64 `math.Tanh`
rather than becoming a device kernel or a float32 approximation — both would be faster and neither
would be the same number as the CPU path produces (decoder/forwardn.go, decoder/model.go).

SIBLING SET. Six sites carry this identical loop: decoder/forwardn.go, decoder/model.go,
cuda/prefill.go, cuda/resident.go, metal/model.go, and gpu/softcap.go (G6,
docs/tasks/task-gpu-paths-2026-09.md). Both cuda/ callers now share this helper. The other three are
unchanged and deliberately so — decoder/ is under the 6edd1ca numerics freeze and metal/ is on
hold — which is recorded in docs/QUEUE.md B6 so the pair is not left implicit.
```

## package cuda

Moved from `cuda/doc.go` (the comment the package comment) on 2026-10-09.

```text
Package cuda is an OPT-IN, cgo-free native-CUDA backend for goinfer's resident decode path.

The design splits in two, and the split is the whole strategic point:

  - Layer A (driver plumbing) — a 1:1 shim over github.com/eitamring/gocudrv:
    dlopen libcuda.so.1 at runtime, so `CGO_ENABLED=0` and the single-static-
    binary property hold. All context/alloc/memcpy/module-from-PTX/launch/
    stream/event calls are here. gocudrv covers this EXCEPT cooperative launch.

  - Layer B (the compute) — the production kernel set in cuda/*.cu, each compiled to PTX by
    `go generate` (build_ptx.sh: NVRTC, not nvcc — see that script for why; audited modules are
    pinned to NVRTC 12.6.85) and go:embed'd for driver-side JIT (cuModuleLoadDataEx):
    GEMV/GEMM decode (gemv_fwd, gemv_w8a8_batched,
    gemv_w4a8_rn, gemm_w4a8_mma), prefill and attention (prefill_batched, decode_splitkv,
    attn_block, attn_img_prefill, attn_fused, rope_mrope_prefill), MoE routing (moe,
    router_f32), other families (deltanet, gptoss_act), and fused/misc (fused_qkv, glue,
    argmax, lora, layernorm_quant, gelu_quant, and more). kernels.go's //go:embed list is the
    census (28 modules on 2026-09-25; this comment said 22 and drifted — count there, not here).

This backend grew out of a 2026-07 spike into a single fused decode-layer megakernel, scoped
in docs/completed/cuda-megakernel-spec.md; the go/no-go read is
docs/completed/task-cuda-cgofree-spike.md. The megakernel itself never shipped — the spike's
K1/K3a option (per-op kernel fusion, not one all-in-one kernel) did, as cuda/fused_qkv.cu; a
K2 (broader fusion) was built, measured ~0%, and reverted. The spec's scaffold
(cuda/megakernel.cu, never go:embed'd or functional) was deleted in the 2026-09-12 closeout.

Build with `-tags cuda`. BuildResident is live: it builds a resident forward when the driver
and the arch's features allow, and declines (ok=false) otherwise, in which case the decoder
falls back to the staged/CPU path. Blank-importing this package is safe either way.
```

## fusedQKVRowsPerWarp

Moved from `cuda/fused_qkv_rows.go` (the comment above `fusedQKVRowsPerWarp`) on 2026-10-09.

```text
fusedQKVRowsPerWarp chooses how many output rows each warp of fused_rms_qkv_rows walks (1 = the original one-row-per-warp kernel, which the caller keeps for
rpw <= 1). Every warp of every block first redundantly recomputes the layer's rmsnorm + int8 quantisation, so the kernel is fastest when the grid is small enough that
few blocks pay that prologue and large enough to fill the card. Measured on ten real geometries (docs/measurements/fused-rms-qkv-2026-09-21.md, ncu kernel time,
random int4 weights): the best setting lands where the grid is about 64-100 blocks — D7 (4608 rows) rpw 8 = 72 blocks, 0.64x of the original's time; qwen2.5-3b (2560) rpw 4 =
80 blocks, 0.76x; 1.5B (2048) rpw 2-4, 0.72x; gemma3-1b (1536) rpw 2, 0.76x — and the small 0.5B (1152 rows) gains nothing (rpw 2 is 6% WORSE), hence the floor below.

The rule: the largest power of two (<= 16) that still leaves >= 64 blocks of 8*rpw rows, for projections of at least 1536 rows; else 1. The kernel is bit-identical for any
value (TestFusedQKVRowsBitIdentical), so a wrong pick on an unmeasured geometry can only cost speed, never change a logit.
```

## fusedGURowsPerWarp

Moved from `cuda/fused_qkv_rows.go` (the comment above `fusedGURowsPerWarp`) on 2026-10-09.

```text
fusedGURowsPerWarp chooses the rows-per-warp of fused_rms_gu_rows for a (hidden, intermediate) geometry; 8 is the original fused_rms_gu (which the caller keeps for 8).
Unlike the QKV kernel there is no simple rule: measured on nine real geometries (15 launches each, ncu kernel time, random int4 weights, spreads 1-10 us) the win jumps around
at grid-size thresholds (wave quantisation on 40 SMs) — rows-per-warp 16 is -13% on qwen3-1.7b and -8% on gemma3-1b, but +33% on the 0.5B and +4% on qwen2.5-3b, which prefers
32 (-11%). So this is a TABLE of the geometries measured to win, and every other geometry keeps the original kernel. The kernel is bit-identical for any value
(TestFusedGURowsBitIdentical), so a miss can only cost speed. Medians, original -> chosen: D7 (3584,18944) 214.9 -> 208.0 us (rpw 16); qwen2.5-3b (2048,11008) 76.9 -> 68.1 (32);
qwen3-4b (2560,9728) 79.6 -> 76.2 (16); qwen3-1.7b (2048,6144) 46.3 -> 40.3 (16); llama3-8b (4096,14336) 179.6 -> 164.2 (32); gemma3-1b (1152,6912) 35.5 -> 32.7 (16);
phi3-mini (3072,8192) 79.1 -> 75.7 (16). Not in the table (no gain): 1.5B (1536,8960), 0.5B (896,4864).
```

## waveRowsPerWarpFor

Moved from `cuda/fused_qkv_rows.go` (the comment above `waveRowsPerWarpFor`) on 2026-10-09.

```text
waveRowsPerWarpFor is the rows-per-warp that sizes a fused-projection grid (blocks of 8 warps, one prologue per block) to ONE resident wave: blocks = (resident blocks per SM) x (SM count),
where resident blocks per SM is limited by threads (256 per block) and by the block's dynamic shared memory, and rows-per-warp = ceil(rows / (8 x blocks)). 0 means "cannot tell".

Why: every block first recomputes the layer's rmsnorm + int8 quantisation before streaming any weight, which costs ~24 us on D7's gate/up and never overlaps the block's own traffic. More blocks
than one wave means the tail of the grid runs partly-empty waves (wave quantisation on 40 SMs: D7 gate/up measured 214.6 us at 592 blocks, 238 us at 132, and 196.9 us at 119 = one full wave of 3 per SM);
fewer blocks means fewer prologues. The rule replaced a per-geometry table with a formula that is at least as good almost everywhere, and finds wins the table missed
(docs/measurements/fused-rms-qkv-2026-09-21.md, "wave rule"). It assumes registers do not limit occupancy (true for these kernels on the measured card); if they did, the grid would simply be larger than one wave.
The kernels are bit-identical for any rows-per-warp, so a bad pick costs speed, never a logit. Measured on one card (RTX 2070 SUPER, 40 SMs) only.
```

## batchstep.go: multi-sequence step header

Moved from `cuda/batchstep.go` (the comment above the ResidentBatchStepper assertion) on 2026-10-09.

```text
MC3 on CUDA (docs/tasks/task-concurrency-2026-09.md, "MC3 on CUDA"): one decode token for each of several sequences,
each on its own resident KV slot at its own position, in one step. The S0 found a batched pass through the exact
kernels bit-identical to decode per row, and 1.75x (1.5B) / 2.04x (7B) cheaper than 4 decodes
(docs/measurements/concurrency-mc3-cuda-s0-2026-09-27.md).

The step is prefillCore's batched layer stack in its multi-sequence mode (rows != nil): rms+quant, the q/k/v, o,
gate/up and down GEMVs, the batched qk-norm, SwiGLU and residuals run over the B rows through the exact kernels
(forceExactKernels holds for tailAllLogits). Rope, the KV store and attention run per sequence through decode's own
gap (decodeAttnGap) — every row gets exactly the kernel and position-dependent scales its own decode would, the
flash-decode lane past 2048 keys included. The head is per row, decode's, with Forward's host tail; a row whose token
is drawn on-device ends in ForwardSample's pick.
```

## specdecode.go: file header

Moved from `cuda/specdecode.go` (the comment at the top of the file) on 2026-10-09.

```text
Speculative decoding (D1) — n-gram / prompt-lookup drafting + batched verify. Greedy-lossless by
construction: the verify's per-position logits (PrefillLastN) are bit-identical to a sequential
Forward at each position, so the accepted tokens are exactly the sequential greedy tokens. The
drafter is FREE (no draft model — a context lookup), so the whole cost is the batched M=k verify,
which amortizes the weight read across k tokens (measured 2.5–3.6× cheaper than k decodes at k=4–8,
TestSpecVerifyCeiling). See docs/ollama-chase.md §D1.
```

## qwen25_vision.go: tower header

Moved from `cuda/qwen25_vision.go` (the comment at the top of the file) on 2026-10-09.

```text
Qwen2.5-VL's vision tower in float32 on the CUDA tower base (S7 on CUDA's fix, docs/tasks/task-multimodal-support-2026-10.md): the port of metal/vl_towers.go's qwen25VResident, replacing aikit's gpu/qwencuda
(7.97 s on the 896x896 image, with aikit's unfused attention, and an allocation that failed beside the decoder at serve's defaults). Everything positional comes from aikit's own exports, not a reimplementation:
```

## qwen25_vision.go: padded intermediate width

Moved from `cuda/qwen25_vision.go` (the comment in the body of the registered factory) on 2026-10-09.

```text
The intermediate width is padded up to a multiple of 64 with zeros: Qwen2.5-VL-3B's is 3420 (3420 % 16 = 12), which sends the down projection (K = 3420) and the biased epilogue through the tiled GEMM at 1.8 TFLOPS,
about half of what the register-blocked kernel reaches on this base (S17 step 0 on this tower: GEMM 92% of 3.8 s). The extra gate and up rows and bias entries are zero, so silu(0) * 0 = 0 exactly and the padded
columns of down add nothing: the arithmetic is the unpadded tower's.
```

## prefillDeltaNetRows: rule kernel

Moved from `cuda/prefill_deltanet.go` (the comment in the body of `prefillDeltaNetRows`) on 2026-10-09.

```text
Every released model's geometry: the state row stays in registers across the rows (1.23 s → see the task
doc for the measured cost of the generic scan on the 9B).
```

## device.go: file header

Moved from `cuda/device.go` (the comment at the top of the file) on 2026-10-09.

```text
CUDA device layer — now aikit's native-GPU substrate (github.com/townsendmerino/aikit/gpu),
the CUDA analogue of what metal/device.go did for Metal (goinfer 5ec20ff). goinfer keeps its
tuned decode kernels here and builds them on these device types — the GPU analogue of the
linalg relationship. Only the device TYPES moved; nothing about the decode path changed, so
it must stay bit-identical (the CUDA device-parity suite is the tripwire).

The type + non-generic-func aliases below let the tuned code read unqualified (Buffer, Arg,
Grid1D, …), exactly as it did against gocudrv. Go has no generic-method/var aliases, so the
generic verbs (ArgValue, NewBufferOf/LenOf, Upload/Download, NewHostBuffer, ReadToHost) are
called as gpu.X[T](…) at the sites; everything else is unqualified here.
```

## vision_towers.go: imports not taken

Moved from `cuda/vision_towers.go` (the comment at the top of the file) on 2026-10-09.

```text
Qwen2.5-VL's vision tower on CUDA is goinfer's own (qwen25_vision.go, on the tower base with the fused attention), registered through vision.RegisterQwenResident from this package. aikit's gpu/qwencuda is NOT imported
any more: it carried aikit's unfused attention (7.97 s on the 896x896 image against this tower's), and at serve's defaults on the 8 GB card its scratch allocation failed beside the decoder (S7, 2026-10-07). Its
allocation-failure fix (v0.1.1) stays in aikit. aikit's gpu/visioncuda (SigLIP) is likewise not imported: it is wrong at real size (G-S4q), and cuda/vision_register.go owns the one global vision.RegisterResident hook.
```

## addInPlaceHost and residual: host round trip

Moved from `cuda/vision_encoder.go` (the comment above `residual` and `addInPlaceHost`) on 2026-10-09.

```text
residual: x += y (elementwise), same shape as cuda/glue.cu's own `residual` kernel — but that
kernel lives in glue.ptx, a module this tower does not load (it never touches the text decoder's
glue kernels other than the shared quant_vec_batched). Reuses quant_vec_batched's own module
(prefill_batched.ptx) is not possible for a plain add, so this is a tiny host-side loop over a
download+add+upload instead — the residual add is O(hidden) per row, not the bottleneck.

NOT a new kernel: see ForwardPatches's own comment at each residual site for why a device-side
add was skipped for v1 (small, and correctness-first).

addInPlaceHost: x += y (elementwise, full arrays of matched length n — NOT a broadcast; every
caller here, including the posEmb add, needs a same-shape add). Correctness-first v1 via a host
round-trip (small — O(M*hidden) per call, not the per-layer GEMV/attention bottleneck). A
device-side elementwise-add kernel is a natural, cheap follow-on if profiling ever shows this
matters; not attempted here (P6's own scope is the tower's GEMV/attention cost, untouched by
this).
```
