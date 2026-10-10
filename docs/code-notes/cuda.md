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

## packWeight.int4

Moved from `cuda/resident.go` (the comment above `packWeight.int4`) on 2026-10-09.

```text
Kind() reports precision, not layout, and stays "int4" for a repacked-only tensor
(aikit audit M-22) — so this switch alone cannot tell canonical from repacked-only.
The ok this used to discard is that distinction: without it, a repacked-only tensor's
nil q4 indexes out of range below instead of declining cleanly (found by inspection,
unverified on real hardware — no CUDA device to run this on; mirrors the identical bug
metal/model.go's int4Concat had and was fixed for, decoder/weightmat.go's
wantsCanonicalInt4 doc comment has the full policy). Under that gate this case should
be unreachable in practice (repacked-only only activates for Options.Backend=="cpu"
literally, never "cuda"), so this is defense in depth, not a path expected to fire.
```

## packWeightStack.reserve

Moved from `cuda/resident.go` (the comment above `packWeightStack.reserve`) on 2026-10-09.

```text
Reserve the whole stack up front. Without this the appends below regrow the slice
geometrically and leave every outgrown copy as garbage; measured 2026-09-19 on the
real gpt-oss-20b (--moe-cache-experts), the heap reached ~40 GB with ~23 GB live because
the collector let that garbage pile up across all layers' expert stacks.
```

## deltaNetMixer.graphs

Moved from `cuda/resident.go` (the comment above `deltaNetMixer.graphs`) on 2026-10-09.

```text
NOT GRAPH-CAPTURED. The three static segments exist so CUDA graphs can replay them; this path
runs live. That costs nothing real — graphs measured 1.01× on this backend and are off by
default — and it avoids capturing launches over buffers whose contents ARE the per-token state.
```

## layerTail.clear

Moved from `cuda/resident.go` (the comment above `layerTail.clear`) on 2026-10-09.

```text
segC(l-1) writes AND reads g4x2 on r.stream (CU_STREAM_NON_BLOCKING). The clear must land after that read and before segC(l) accumulates
into it. A memset ON r.stream is ordered by the stream itself, in every mode, and drains nothing (audit R-24). The non-overlap path used to
do r.stream.Sync() then gpu.Upload of a host zero slice, because gpu.Upload runs its copy on the context's legacy null stream, which has NO
ordering against r.stream (aikit gpu/cuda.go): the Sync made the clear follow the prior layer's kernels (audit R-03), and the Upload added two
more context syncs and a pageable copy. ZeroAsync keeps the R-03 ordering without any of them. The router readback that follows
(loadRoutedExperts) still syncs r.stream itself, so the wait is not lost, only moved to where it was needed anyway.
```

## fusedG32MaxHidden

Moved from `cuda/resident.go` (the comment above `fusedG32MaxHidden`) on 2026-10-09.

```text
fusedG32MaxHidden is the widest hidden size the fused per-32 kernels are used at: the largest
measured win (qwen2.5-coder-1.5b, H=1536, +10%). Phi-3 (3072) and qwen2.5-7b (3584) measured slower
fused than unfused (docs/tasks/task-int4-weight-quality-2026-09.md, lever 3). Sizes between 1536 and
3072 are unmeasured and stay unfused.
```

## Close

Moved from `cuda/resident.go` (the comment above `Close`) on 2026-10-09.

```text
Close shuts down the executor goroutine (and unpins its OS thread). Device buffers are
freed by primary-context teardown at process exit; a per-buffer free is unnecessary for
the single-model serve lifetime.
Close releases the model's GPU memory and tears down the pinned executor.

Freeing the DEVICE memory is the whole job: a resident model owns the weight buffers and the
per-layer KV cache — gigabytes for a real checkpoint. This once freed only the page-locked
HOST buffer and closed the channel, so every Load(cuda)+Close leaked the entire model until
the process exited: invisible in a one-model run, fatal for a model zoo, an
/admin/models/unload, or a test binary loading models in sequence (it saturated an 8 GB card
mid-suite, after which every Alloc silently returned nil and the zero-filled buffers looked
like a parity bug rather than an OOM).

Every buffer is freed EXPLICITLY rather than by leaning on context destruction. Releasing our
primary-context reference only reclaims memory if the refcount reaches ZERO — and
dev.Primary() hands out a refcounted per-device singleton, so any other holder (a second
model in a zoo, another subsystem, a test's own probe context) keeps the context alive and
the "freed" model's VRAM never comes back. That is precisely the multi-model case unloading
exists for, so the release-the-context shortcut was wrong exactly where it mattered most;
TestResidentCloseFreesVRAM pins it.

All of it runs ON the executor thread — that thread made the context current — and therefore
before reqCh closes. Page-locked host memory goes first: it must be freed before the context.

Returns error to satisfy io.Closer (audit B-12; see the assertion in backend.go); the native
releases are best-effort and can't meaningfully fail, so it always returns nil — like metal's.
```

## glueQuantThreads

Moved from `cuda/resident.go` (the comment above `glueQuantThreads`) on 2026-10-09.

```text
glueQuantThreads is glu_quant's block size. The kernel is ONE block (its int8 scale needs the max over the whole intermediate
vector), so its time is the serial per-thread loop over I elements: at 256 threads it measured 35.7 us per layer on D7 (I = 18944),
6.7-7.3% of a decode token. Its only reduction is a MAX (exact, order-independent) and every element's value and packing are per-element, so
the block size changes nothing but how the work is divided: the output is bit-identical (checked on real logits, docs/measurements/d7-decode-breakdown-2026-09-21.md).
```

## describeLaunchErr

Moved from `cuda/resident.go` (the comment above `describeLaunchErr`) on 2026-10-09.

```text
describeLaunchErr turns a bare driver status into something that names what ran out and what the
operator can do about it.

The motivating case: a 26B at 34 expert-cache slots died with exactly
`cuLaunchKernel: CUDA_ERROR_OUT_OF_MEMORY` — the API call, not the kernel, and no connection to
the setting that caused it. A day of investigation started from that string. The decline floor
added alongside does NOT cover this: it fires below topK, and this failed at 34 slots with topK
of 8, so the path most in need of an honest failure had none.

Message content only — the error is wrapped with %w, so type and classification are unchanged.
```

## moeMLPPre

Moved from `cuda/resident.go` (the comment above `moeMLPPre`) on 2026-10-09.

```text
moeMLP issues one MoE FFN block for layer Ly, accumulating straight into the residual
stream r.x. Mirrors decoder/mlp.go's moeMLP exactly:

	h        = rmsNorm(x, PreMLPNorm)          — the SAME normed activation feeds router AND experts
	logits   = Router · h                       (f32; see cudaResident.moe)
	idx, wgt = route(logits, bias, ...)
	x       += Σ_j wgt[j] · Down_e(silu(Gate_e·h) * Up_e·h)     where e = idx[j]

The k experts are dispatched SEQUENTIALLY, one slot at a time, but every launch has the same
geometry regardless of which experts the router picked — the expert is chosen by ARITHMETIC on
the weight-row index inside the kernel, not by binding a different buffer. That is what lets a
resident runner keep a static dispatch chain: the routing changes per token, the launches do
not.

The final GEMV weight-accumulates into r.x, so the per-expert combine and the residual add are
the same instruction — no scratch, no separate combine pass. This is why the block `continue`s
the layer loop rather than falling through to the dense epilogue.
moeMLPPre issues the pre-readback half of the MoE FFN: the shared normed activation (mq/mSc) and
the router (logits → top-k idx/wgt, left on the device). It ends exactly at the point where the
cacheExperts path must read rIdx back to the host — so this half is graph-static (segB), and the
loadRoutedExperts D2H stays live in the gap between segB and segC.
```

## moeMLPPre.route

Moved from `cuda/resident.go` (the comment above `moeMLPPre.route`) on 2026-10-09.

```text
TRAP, formerly live, now closed with a test that actually catches it (2026-09): CUDA now
declares FeatMLA (DeepSeek/Kimi, decoder/features.go), so the group-routed families this
comment used to say could never reach this line now do, with a REAL mismatch —
testdata/deepseek-tiny sets n_group=2, topk_group=1.

This mapping was found UNVERIFIED reviewing that change: TestMoERoute constructs
nGroup != topkGroup (8/4) and asserts against cpuRoute, but builds its own argument list
and calls the kernel directly, so it validates the kernel's math, not this call's argument
ORDER — and TestMLAResidentParityCUDA's own greedy-generation check compared only
cpuToks[0], not the sequence, which a transposition passed cleanly (forward-logit cosine
unaffected; token 0 unaffected; tokens 4-7 silently different — a discrete-selection bug
does not have to show up in the first few tokens the same way a numerical one would).
Reproduced directly: swapping the two ArgValue calls below, TestMLAResidentParityCUDA
PASSED. Fixed by comparing the full generated sequence, not just its first token
(cuda/mla_resident_test.go) — re-run with the same swap, it now fails at token 4 exactly
where the logged (but previously unchecked) divergence already was. Restored the correct
order and confirmed clean again before either change shipped.

Typed launch wrappers (one generated type per (parameter name, C type)) make a
transposition at this site a COMPILE error. They do not verify that the values are the
right way round — a wrong value of the right kind still compiles. See the reciprocal trap
at decoder/features.go's cuda entry, and docs/parity-coverage-policy.md § "Relevant".
gpt-oss routes through its OWN kernel: moe_route takes the mixing weight from the
UNBIASED score (bias steers selection only), while gpt-oss softmaxes over the SELECTED
BIASED logits. Same selection, different weights — which is why the wrong one produces
plausible output rather than an error.
```

## segA.kEqV

Moved from `cuda/resident.go` (the comment above `segA.kEqV`) on 2026-10-09.

```text
K=V: vB is the RAW (pre-norm) k projection, which v_norm consumes below. It used to be projected a second time from the same inputs (one more kvDim x hidden
weight read per K=V layer per token); a copy of kB is bit-identical, and a launch like every other op here, so it records into the CUDA graph and is ordered on this stream
between the projection and qk_norm (which rewrites kB in place), exactly where the second projection sat.
```

## expertBiasIdx

Moved from `cuda/resident.go` (the comment above `expertBiasIdx`) on 2026-10-09.

```text
expertBiasIdx is the index for PER-EXPERT TABLES that are uploaded ONCE for all experts and
indexed on the device — today only gpt-oss's [nExpert][2*I] gate‖up bias table. It is always
the router's real expert ids, NEVER the slot ids: the table is expert-indexed and does not
move when an expert is streamed into a slot.

Binding expIdx here instead was a live defect (fixed 2026-08-31, never shipped in a run):
glu_quant_gptoss does `biasGU + idx[slot]*2*I`, so with caching on it would have selected the
bias row by SLOT id — the wrong expert's gate/up biases, no error, plausible output. It could
not be caught by any test that exists because gpt-oss has never been admitted on CUDA, and
expert caching is exactly the path gpt-oss needs to fit an 8 GB card at all.
```

## mapBytes.register

Moved from `cuda/resident.go` (the comment above `mapBytes.register`) on 2026-10-09.

```text
Register in place (GOINFER_MOE_PIN_REGISTER, read from the model's knob snapshot in mapBytes below): Lead 3
(docs/tasks/task-freetoken-techniques.md), measured
2026-09-22 (docs/measurements/lead3-pin-order-2026-09-22.md) — populate-then-pin beat
allocate-then-copy 1.33x-4.46x in a standalone microbenchmark at this scale, and never lost a
single real-load trial (5/5) in the follow-up decision measurement. That measurement's own
pre-registered rule (ship >=15% off the real 26B load time) put the result — 1.105x, 10.5% —
in its PARK band, not ship. DEFAULT ON ANYWAY, owner override, same day: a real, direction-
consistent win with zero losing trials, on a path with no numerics risk (RegisterMappedHostBuffer
pins the caller's own already-populated bytes; the DMA source content is byte-for-byte the same
either way, gated correctness tests unaffected). `GOINFER_MOE_PIN_REGISTER=0` restores the
allocate-then-copy order the pre-registered measurement called the do-nothing arm.
```

## runJob

Moved from `cuda/resident.go` (the comment above `runJob`) on 2026-10-09.

```text
runJob is the executor goroutine's PANIC BOUNDARY: it runs one job and converts a panic into
an ordinary error, so the pinned thread survives and `do` returns to its caller (audit C-24).

Why it has to live here and not at a call site. Every job runs on the executor goroutine, but
`do` blocks on a *different* goroutine — so a `defer recover()` in BuildResident, or in any
caller, cannot catch a panic raised inside `j()`. Two comments (resident.go's setup path and
prefill.go's scratch note) asserted this was already handled; it was not, and the gap is
reachable by design rather than by accident: `gpu.NewBufferLenOf` PANICS on allocation failure
per its own contract, and prefillCore allocates M*(2*inter+2*hidden+…) floats — hundreds of MB
on a long prompt. So a long prompt against a nearly-full card killed the serve process, at the
one seam whose entire job is to decline to the sequential path instead.

The recovered error is deliberately NOT wrapped in errPrefillDeclined here: `do` is shared by
every job (setup, decode, prefill), and only the prefill caller knows a decline is the right
response. prefillCore wraps it at its own boundary.
```

## checkWeightsFit

Moved from `cuda/resident.go` (the comment above `checkWeightsFit`) on 2026-10-09.

```text
checkWeightsFit fails the load EARLY — right after the device exists, before any kernel compile
or weight upload — when the model's FIXED weight bytes alone exceed free VRAM.

M-02 (docs/audit-2026-09-02.md): CUDA had NO memory-fit check at all for this. checkKVFits
(below) covers KV, but only after the weights are already uploaded; a dense model whose weights
alone do not fit ran the entire compile+upload sequence and then failed on whichever CUDA
allocation happened to be the first one that didn't fit — a raw driver error, not a clean
decline naming the numbers, the exact failure MODE (not arithmetic) checkKVFits' own doc
comment already called out as the point of a load-time guard.

`need` is decoder.Model.ResidentDenseWeightBytes, deliberately NOT ResidentWeightBytesPaged(0):
routed MoE experts have their own elastic sizing (capSlots, above), a bisection search against
LIVE free VRAM that runs after dense weights are uploaded and correctly shrinks to whatever
still fits. Pricing experts at their full unpaged size here, before capSlots ever runs, would
decline a model whose experts are about to be capped down to something that fits — the same
class of over-eager guard M-02 already fixed once for Metal's paging case.
```

## checkKVFits.trim

Moved from `cuda/resident.go` (the comment above `checkKVFits.trim`) on 2026-10-09.

```text
An unpinned context that even ONE slot of KV overshoots (by the build's own scratch, which ctxForSlots cannot see) gives up positions, never
below cudaCtxCapDefault, instead of declining the whole resident path to the CPU with a log line. R19 (639e646c) raised the candidate to 16384
and says it is "shrunk to what the card holds", but only the multi-slot arm below did that; on the 8 GB card a plain decoder.Load of a 7B
(16000 positions, 1.84 GB of KV against 1.80 GB free) or of Gemma-3 (11800, 3.29 GB against 3.35 GB free, less the 384 MB margin) fell to the
CPU. Found by the 2026-10-07 night gate: five heavy-tier tests failed on it. serve never saw it (it plans its own context).
```

## slotMarginBytes

Moved from `cuda/resident.go` (the comment above `slotMarginBytes`) on 2026-10-09.

```text
slotMarginBytes is the launch-time headroom the cap must leave free. NOTE it is currently the
only unmeasured constant in this path: whether 384 MiB is the right figure is open, and the
per-layer allocation overhead it must absorb has not yet been measured within a single process.
```

## capSlots

Moved from `cuda/resident.go` (the comment above `capSlots`) on 2026-10-09.

```text
capSlots is the sizing arithmetic, and it is a SEARCH rather than a division.

Per-buffer 2 MiB rounding makes the requirement a step function of the slot count, and

	fit := (free - slotMarginBytes) / nLayers / perLayer

cannot invert a step function — it is wrong precisely at the boundaries the failure lives on. On
the real 26B that division returned 34, where the true requirement at 34 exceeds free by
203,816,960 B: at n=34 the ratio n*123904/2MiB crosses 2 and all four buffers tip a quantum AT
ONCE, a 4-quanta step. The forward then generated zero tokens. A division plus a correction term
would reproduce the same class one boundary over, so there is no fudge factor here.

The requirement is monotone non-decreasing in n, so bisect it. Pure function of its inputs, so
synthetic free-VRAM figures exercise a branch that in production binds only on models far larger
than any fixture — the exercised-but-never-triggered shape.

Returns the slot count to use, or decline=true when not even topK fits.
```

## allocSlots.nostride

Moved from `cuda/resident.go` (the comment above `allocSlots.nostride`) on 2026-10-09.

```text
No routed layer reports a per-expert stride (a degenerate blob): nothing to cache and,
more importantly, nothing to divide by. Clear cacheExperts so the decode path can't later
read slots/expCache that were never allocated → nil-deref (audit R-25). NOTE (F-05): this
does NOT yield a working "hold every expert" resident path — with caching off and no stride,
upExperts left the expert stacks host-mapped-only, so the expert GEMVs would bind zero-value
device buffers. This branch is unreachable with any real MoE checkpoint (they all report a
per-expert stride); it exists only to fail safe, not to serve. A blob that trips it should be
declined to the staged/CPU path upstream.
```

## allocSlots.budget

Moved from `cuda/resident.go` (the comment above `allocSlots.budget`) on 2026-10-09.

```text
ONE implementation. This used to be an inline copy of the same arithmetic, with capSlots
existing only for the gate — so the gate corroborated a parallel copy and a change to
either was uncontradicted by the other (the sibling-drift instance in
docs/parity-coverage-policy.md). The gate now points at the shipping path.

budget, not free, goes into capSlots: r.extraBytes reserves room for a companion attach
(--drafter) coming after this build (Model.ExtraResidentBytes's own doc comment) — the
exact scenario tasks/task-fit-to-hardware.md §2 measured (a 26B auto-sized to 31 slots/layer,
then --drafter attached and NewBlockSpec failed with no room left). 0 when nothing is
attaching, so budget == free then and this is unchanged.
```

## allocSlots.floor

Moved from `cuda/resident.go` (the comment above `allocSlots.floor`) on 2026-10-09.

```text
FLOOR. topK slots is the minimum that can work — one token's routed set must be
simultaneously resident — so if even that does not fit, DECLINE naming the shortfall
instead of allocating and discovering it at the first kernel launch.

This used to read `if capped < r.topK { capped = r.topK }`, commented "topK always
fits". That is an assumption written as a check: when false it clamps UP to a figure
it has just computed does not fit, allocates it, and the failure surfaces later as
CUDA_ERROR_OUT_OF_MEMORY from cuLaunchKernel or a generation loop returning nothing —
neither of which points back here.
```

## allocSlots.probe

Moved from `cuda/resident.go` (the comment above `allocSlots.probe`) on 2026-10-09.

```text
A10's per-allocation recording (GOINFER_A10_PROBE) was retired 2026-09-24, phase 6 of
docs/tasks/task-env-config-2026-09.md: its question — capacity or servability — is answered in
the account below.
```

## allocSlots.order

Moved from `cuda/resident.go` (the comment above `allocSlots.order`) on 2026-10-09.

```text
Issue LARGEST FIRST across all layers, rather than group-by-group. Total is identical either
way — capSlots and the granularity form are untouched, only the order moves.

HONEST ACCOUNT OF WHY: this was A10's hypothesised FIX and it was REFUTED as one. The theory
was that group-by-group (30 repetitions of {64.3, 8.0, 32.1, 4.0} MiB) leaves the last layer's
biggest request facing the most carved-up heap. Largest-first was predicted to complete at 34
slots; it did not. It failed on a 4,212,736 B request with 155,385,856 B free — a ratio of
36.88, which no contiguity story survives.

The real constraint is a driver ALLOCATION FLOOR: 151,191,552 B (144.2 MiB) that cuMemGetInfo
reports as free and cuMemAlloc will not hand out at ANY request size down to 1 MiB, measured
directly in TestAllocFloor. Leftover after allocSlots must exceed it, which is what the margin
now provides.

This ordering is KEPT anyway, on its measured merit and not on the refuted theory: it drains
27 MiB further before hitting the floor (155,385,856 vs 182,648,832) and packs more bytes
before failing, at zero cost. It is a packing improvement, not a fix.

Sorting by MEASURED request size rather than by an assumed stride order, because per-layer
geometry is not guaranteed uniform (Gemma 4's KV widths already differ per layer, and
slotBytesPerLayer exists because layer 0 can be dense). A stable sort keeps layer order within
each size class, so the sequence stays deterministic.
```

## appendExpertSlot

Moved from `cuda/resident.go` (the comment above `appendExpertSlot`) on 2026-10-09.

```text
appendExpertSlot QUEUES one expert's weight+scales copy from the pinned host source into device
slot `slot`. It does not upload: loadRoutedExperts submits the layer's whole batch with a single
gpu.UploadBatch, so one synchronize covers every miss in the layer instead of two per miss.

Why the change is the sync count and not the copy. Each gpu.Upload ends in a full device
Synchronize — right for per-request uploads, wrong here: a MoE decode token loads ~120 slots at
two uploads each, so ~240 synchronizes land on one token. Measured on an RTX 2070 SUPER that is
~3.6 ms of a 64 ms token (5.6%), paid for nothing, since the bytes are already in flight and one
sync at the end covers them all.

The correctness property is PRESERVED, not weakened. UploadBatch still synchronizes before it
returns, so the guarantee is identical to Upload's and the race that sync was added for
(non-blocking streams unordered against the null stream) stays covered. Nothing moves into a
caller's hands — that is the whole difference from the declined async variant.
```

## loadRoutedExperts.timing

Moved from `cuda/resident.go` (the comment above `loadRoutedExperts.timing`) on 2026-10-09.

```text
C′ TIMING SEAM (GOINFER_MOE_CACHE_PROF). This function is the only host round trip on the
decode path, and it happens once per MoE layer per token — 40 times for the 35B. It splits
into three costs that call for completely different fixes, and tok/s alone cannot separate
them:

  stall  the stream drain, waiting for the router kernel. Fixing this means removing the
         round trip (device-side slot mapping), not making it faster.
  host   the LRU bookkeeping. Pure CPU; fixing it is ordinary optimization.
  dma    the H2D of missed experts. Fixing this means more slots or fewer bytes — and the
         slot sweep already showed that lever saturating, so if dma is small the knee is
         explained and more slots really are pointless.

Off by default and zero cost when off (one branch); it adds no syncs of its own, because the
stall it measures is a sync that already exists.
```

## cudaCtxCapDefault

Moved from `cuda/resident.go` (the comment above `cudaCtxCapDefault`) on 2026-10-09.

```text
cudaCtxCapDefault is the resident KV capacity in positions when nothing asks for more; the staged
path handles longer. It is a DEFAULT, not a ceiling: decoder.Options.ResidentContext raises it (see
resolveCtxCap). It stays 4096 so that a caller who did not ask never allocates deep-KV VRAM —
raising the default would silently multiply every resident model's KV footprint.

4096 is a round, conservative choice, NOT a value tuned against real VRAM headroom — nothing has
ever measured how much of a real card's free VRAM it leaves unused. Measured 2026-09-06
(docs/tasks/parked/task-kv-cache-streaming.md): on an RTX 2070 SUPER (8 GB) with a dense 7B at int4,
checkKVFits accepts -ctx 20000 (7257/8192 MiB used) and refuses -ctx 24576 (needs 2.82 GB of KV,
2.90 GB free) — a true per-card ceiling roughly 5-6x this default. Whether that ratio holds for
other model sizes/quants/cards is unmeasured; raise -ctx and read checkKVFits' own error to find
the real number for a given deployment rather than assuming this default reflects it.
```

## fitDefaultCtx

Moved from `cuda/resident.go` (the comment above `fitDefaultCtx`) on 2026-10-09.

```text
fitDefaultCtx is the candidate context resolveCtxCapFit tries for an UNPINNED request, before
falling back to cudaCtxCapDefault — tasks/task-fit-to-hardware.md §8's own answer to "what should the
default even be": the size of a coding agent's turn, not the model's full window (which can be far
larger than anyone asked for). It was 8192 (the agent-turn size docs/server.md's dsh section measured)
until 2026-10-01, when a cold-user run's first opencode request was 11,137 tokens and a default of 8192
refused it (R19, docs/tasks/task-first-hour.md); 16384 holds that request with headroom. It is a
CANDIDATE, not a grant: Plan shrinks it to what the card holds, and ctxForSlots shrinks it further until
the requested KV slots all fit (owner decision 2026-09-27: context before conversations), so a card or
model that cannot hold 16384 per slot lands where it always did. goinfer-chat fit's own -ctx default
(internal/fitcmd/fit.go) uses the same figure, so the dry run and the real load agree.
```

## resolveCtxCapFit

Moved from `cuda/resident.go` (the comment above `resolveCtxCapFit`) on 2026-10-09.

```text
resolveCtxCapFit is tasks/task-fit-to-hardware.md Phase 2's "fit by default" for CUDA's context: an
UNPINNED load no longer gets a flat cudaCtxCapDefault regardless of the card — cudaCtxCapDefault's
OWN doc comment records a real measurement (RTX 2070 SUPER, dense 7B int4) where the true ceiling
was 5-6x the default, unused by anyone who did not know to pass -ctx. This asks Plan for a bigger
candidate (fitDefaultCtx, clamped to the model's own window) and uses whatever Plan lands on —
which can only ever be cudaCtxCapDefault or MORE, never less, because of the explicit floor
checks below. A caller with m.FitDisabled() true (--fit=off, or its GOINFER_NO_FIT_DEFAULT env
var precursor), or whose free-VRAM probe is unknown, gets EXACTLY today's resolveCtxCap — this
function can only improve on the historical default, never regress it, so there is no failure
mode where turning fit-by-default off would have helped.

m.MoECacheExperts() gets the same treatment, found live 2026-09-15/16 re-measuring the peer
matrix: growing the default context is a pure win for a model whose KV is the only thing
competing for free VRAM (the commit that introduced this measured exactly that, at fixed decode
depth), but a MoE-cache-experts load has a SECOND, elastic claimant on that same free VRAM — the
host↔VRAM expert-slot cache (docs/benchmarks.md §B4.1's own slots-vs-ctx table) — and every byte
this function hands to KV is a byte the expert cache never sees. Measured on gemma4-26b-int4.giw
(RTX 2070 SUPER): growing ctx 4096→8192 here cut free VRAM after KV from 3.4 GB to 1.5 GB, which
capped the expert cache from 28 slots to 10 and cut decode from 24.6 to 12.9 tok/s — a ~48%
regression this function's own "can only improve, never regress" invariant was supposed to rule
out, just not for this class of load. Unlike the drafter case (M-22), the expert cache isn't a
fixed cost that can be priced into ExtraBytes and left to Plan — its whole design is "however
much VRAM is left after everything pinned", so the fix is to not let ctx grow into that
leftover at all when this mode is on, the same way FitDisabled already opts out.

slots is MC1's requested resident KV slot count (cudaKVSlotsRequest). Above 1, the unpinned choice gives up context
until every slot fits (ctxForSlots), never below cudaCtxCapDefault — or below request, when request is itself a
guard pin under that default (see below).

request > 0 is NOT always a choice. decoder.Model.ResidentContextPinned() is what actually tells a genuine -ctx
apart from the load-time fit guard auto-pinning a smaller context for an UNREQUESTED load (R13,
decoder/model.go): request still reports that pin (ResidentContextRequest's own doc comment), but it is a
ceiling the guard already proved safe against host RAM, not the caller's own decision — MC1's slots rule is
meant to shrink an unpinned load, and a guard pin was never a "the caller chose this" pin in the first place.
Before this fix, m.ResidentContextPinned() didn't exist and this function had only request>0 to go on, so it
treated a guard pin exactly like an explicit -ctx: fit-by-default (and the slots rule) never ran for it, even
though decoder.ctxFloor (2048) can pin BELOW cudaCtxCapDefault (4096) — a case this function's own "never
regress the historical default" comment above did not anticipate, because raising an already-guard-shrunk
context back to cudaCtxCapDefault is exactly the regression the guard pinned it to prevent.
```

## resolveCtxCapFitSlack.margin

Moved from `cuda/resident.go` (the comment above `resolveCtxCapFitSlack.margin`) on 2026-10-09.

```text
M-12 (docs/audit-2026-09-10.md): Plan's own chooseCtx reserves NO margin — it picks the
largest ctx that exactly fills freeBytes-dense-extra, budget down to the last byte. But the
REAL build-time check, checkKVFits below, requires an ADDITIONAL ctxCapMarginBytes (384 MiB)
beyond dense+KV+extra. Asking Plan with the raw free bytes let it choose a ctx that had
already spent that margin, so any interior (non-ceiling, non-floor) choice failed
checkKVFits almost every time — the whole resident build declining to CPU-only on any card
where the default 8192 doesn't fit outright. Subtracting the SAME margin here, before Plan
ever sees freeBytes, makes the two checks agree: whatever ctx Plan picks now already leaves
room for it. floored at 0 rather than going negative on an already-tiny free-bytes probe
(Plan's own tryCtx/chooseCtx already decline cleanly on an unfittable budget).
```

## resolveCtxCapFitSlack.extra

Moved from `cuda/resident.go` (the comment above `resolveCtxCapFitSlack.extra`) on 2026-10-09.

```text
ExtraBytes: tasks/task-fit-to-hardware.md §2's drafter-aware sizing — a --drafter attaching after
BuildResident must not find the context Plan chose here left it no room (m.ExtraResidentBytes's
own doc comment). Zero when nothing is attaching, so this is a no-op for every load without one.

M-22 (docs/audit-2026-09-10.md): a drafter's device K/V is priced here too now, against
CANDIDATE — the widest ctx this very call is asking Plan to fit, before Plan has shrunk it to
whatever the card actually holds. Plan's chooseCtx only ever shrinks from the value it is
asked with, never grows past it (fitplan.go's own "never GROW past what was asked"), so
pricing the drafter's K/V at candidate can only over-estimate the eventual real cost (safe)
or land exactly on it — never under-price the way pricing at a fixed guess could.
```

## ctxForSlots

Moved from `cuda/resident.go` (the comment above `ctxForSlots`) on 2026-10-09.

```text
ctxForSlots is resolveCtxCapFit's answer when MC1's resident KV slots are requested (owner decision 2026-09-27,
docs/tasks/task-concurrency-2026-09.md MC1 on CUDA): the largest context in [floor, oneSlot] at which Plan fits
every requested slot — the build's own KV plus slots-1 more copies of it, priced as ExtraBytes — so an unpinned
load gives up context before conversations. Measured before the decision on the 8 GB card: the 7B at the
one-slot choice (8192) fit 2 of 4 slots and 4 round-robin clients thrashed (1.006x); at 4096 all 4 fit (1.33x).
When not even floor holds them all, it returns floor and checkKVFits clamps the count, as it always has. An
explicit -ctx never reaches here (resolveCtxCapFit's first branch).

floor is cudaCtxCapDefault, UNLESS resolveCtxCapFit's own caller passed a guard-pinned request below it
(decoder.ctxFloor, 2048, can sit under cudaCtxCapDefault's 4096) — the slots rule may shrink a guard-pinned load
same as an unpinned one, but never past what the guard already proved was the real safety floor.

KV is exactly linear in the context and Plan's other terms do not grow with it, so "fits" is monotone and a binary
search finds the edge.

Plan's weight figure is conservative, so this can land lower than the build would allow. On the 7B, Plan prices the
device weights at 4930 MB and the build allocates ~4476 MB before its KV. It returns the 4096 floor where checkKVFits
would have held 4 slots to ~4870 positions: 4 slots at 4096 instead of at ~4870. Measured 2026-09-27; tightening
Plan's weight estimate is its own item.
```

## kvBytesForCap

Moved from `cuda/resident.go` (the comment above `kvBytesForCap`) on 2026-10-09.

```text
kvBytesForCap is the device bytes the resident K+V caches occupy at a given capacity: every layer
holds K and V as f32[cap*kvDim]. Measured against this formula: 24.0 KB/position for
qwen2.5-coder-0.5b (24 layers × 128 kvDim × 2 × 4 B) and 56.0 KB/position for the 1.5B
(28 × 256 × 2 × 4 B), which is what the deep-context sizing in docs/benchmarks.md is derived from.
```

## kvSlotsFit

Moved from `cuda/resident.go` (the comment above `kvSlotsFit`) on 2026-10-09.

```text
kvSlotsFit is MC1's slot arithmetic on CUDA (docs/tasks/task-concurrency-2026-09.md): the largest n in [1, want]
whose n resident KV slots, perSlot bytes each, fit free VRAM beside reserve (a companion attach plus
ctxCapMarginBytes). Never below 1: the first slot is checkKVFits' own and is refused there, not clamped here.

free is read AFTER the weights are on the device, and perSlot counts KV only — the two sides price the same
thing. Metal's first version compared a weights-inclusive base with a live figure the weights had already left,
counting them twice (6807ab95); pricing KV against what is left for KV cannot.
```

## splitkvThreshold

Moved from `cuda/resident.go` (the comment above `splitkvThreshold`) on 2026-10-09.

```text
splitkvThreshold returns the EFFECTIVE attended-key count (nWin — window-clamped, not the raw
position) at/above which the split-KV decode attention beats the single-block attn_batched(M=1)
for this geometry, or splitkvNever to disable it.

WHY A TABLE AND NOT A FORMULA. Split-KV buys occupancy and pays for it in DRAM. attn_batched
launches nH blocks and keeps the whole score row in SHARED memory (SharedMemBytes=(nWin+128)*4);
split-KV materializes an nH×nWin f32 score array in GLOBAL memory and touches it three times
(splitkv_scores writes, splitkv_softmax reads+writes, splitkv_vsum reads) in exchange for filling
the SMs that nH blocks leave idle. So

	net(nWin) ≈ (A−B)·nWin − 2·nLayers·T_launch

where A grows with the occupancy deficit (it needs nH ≪ SM count) and B grows with nH (score
materialization). A > B gives a crossover; A < B means split-KV NEVER wins and the deficit WIDENS
with depth — which is exactly what phi3-mini measures (nH=32 on a 40-SM part: almost no deficit to
recover, and the largest score array of the four). No one-parameter law reproduces all four
geometries — nLayers/(nH·hd) and nLayers/(nKV·hd) both underpredict the 0.5B crossover by ~2×, and
neither can express phi3's "never" at any threshold. An honest lookup beats a false formula.

MEASURED (e2e decode-only tok/s through serve, int4, RTX 2070 SUPER / 40 SMs, ON÷OFF; >1 = split-KV
wins). Full table and method: docs/benchmarks.md §B6.

	geometry              nH  nKV  hd   L   256    512    1024   2048   3900    crossover
	qwen2.5-0.5b          14   2    64  24  0.839  0.819  0.869  0.955  1.197   ~2560 (see below)
	qwen2.5-1.5b          12   2   128  28  0.941  0.939  1.078  1.191  1.280   ( 512, 1024]
	gemma3-1b (win 512)    4   1   256  26  0.890  0.909  0.919  0.941  1.084   windowed — see below
	phi3-mini (MHA)       32  32    96  32  0.993  0.969  0.919  0.815  0.754   NONE (monotone)

qwen2.5-0.5b was localized further inside the (2048, 3900] band: 2560 → 1.019 (break-even), 3072 →
1.061 (first clear win). So splitkvConservative = 3072 is the measured first-clear-win depth for
that geometry, not a guess; it forfeits ~2% at 2560, which is the asymmetric-loss trade taken
deliberately.

The 256/512 columns are where the old constant fired: it cost the 0.5B up to 18%. The old comment
here claimed "break-even 256, clear win from 384+" from TestSplitKVCrossover on the 1.5B — that is
refuted even on its own geometry (the 1.5B loses at 256 AND 512). That test measures a tight
in-process ForwardArgmax loop and takes best-of-3 MINIMUM; both choices flatter split-KV relative
to serving (the loop hides per-token CPU dispatch that e2e exposes, and best-of-min favours the
higher-variance arm — ON's spread is 3.6–6.4 tok/s vs OFF's 0.1–0.6).

ASYMMETRIC LOSS: firing early costs up to 18–25%, firing late costs a few percent (OFF's slope is
mild near the crossover). Every threshold is therefore rounded UP, and unmeasured geometries get
the conservative default rather than an extrapolation.

NOT DEVICE-PORTABLE: the occupancy term scales with SM count and every cell above is one 40-SM
Turing part. On a much wider GPU nH=32 would be starved and phi3's "never" would not hold.
Re-measure per device class before trusting these on other hardware; do not scale them by SM count
on paper.
```

## splitkvNeverKVFloats

Moved from `cuda/resident.go` (the comment above `splitkvNeverKVFloats`) on 2026-10-09.

```text
splitkvNeverKVFloats: at/above this many KV floats per key (nKV*hd, i.e. one of K or V), the
single-block kernel is close enough to the DRAM roof that extra blocks cannot help, and split-KV
is pure cost.

THIS REPLACED A RULE KEYED ON QUERY-HEAD COUNT (splitkvMaxHeads = 24), WHICH WAS REFUTED.
That rule read: "at/above this many query heads the single-block kernel already fills the
device." Both halves were measured false in 2026-09:

  - Two models at the anchor's OWN nH=32 measure OPPOSITE signs — phi3-mini (MHA, 3072 floats/key)
    0.746 at depth 3900, mistral-7b (GQA 4:1, 1024/key) 1.024. Same head count, so nH cannot be
    what decides. A/A floor 0.268%, effect 9.5-37x it, reproduced across two runs.
  - "Already fills the device" is false: three geometries measure 11-13% achieved occupancy at
    nH=28-32, ~4 active warps per SM against a theoretical 50%.

What does order the sign is how close the kernel already runs to the DRAM roof,
f = 2*nKeys*nKV*hd*4B / (t*BW), and f is set by nKV*hd. Measured at depth 3900, monotone across a
6x span of KV traffic:

	 512 floats/key (Qwen2.5-7B, nH=28)  f 13.5%  ratio 1.0496   +9.94% at depth 8000
	1024 floats/key (mistral-7b, nH=32)  f 26.9%  ratio 1.0240
	3072 floats/key (phi3-mini,  nH=32)  f 67.8%  ratio 0.7460

3072 is the lowest MEASURED loss, not a midpoint: everything at or above phi3-mini's traffic is
excluded, everything below keeps whatever depth threshold it already had. The gap between 1024
and 3072 is unmeasured and lands in the default, which stays conservative.

docs/measurements/splitkv-aa-floor-2026-09-12.md, splitkv-d7-fthreshold-2026-09-13.md.
```

## splitkvMin

Moved from `cuda/resident.go` (the comment above `splitkvMin`) on 2026-10-09.

```text
splitkvMin is splitkvThreshold with the runtime override applied. The override exists because the
previous constant could only be re-characterized by rebuilding, which is part of why a refuted
number survived a release: GOINFER_SPLITKV_MIN_KEYS=<n> re-gates a stock binary (0 ⇒ always take
the split path, the force-on A/B arm), GOINFER_SPLITKV_ATTN=0 still force-disables entirely.
nKV is taken PER LAYER, not from layers[0]: gemma4's layers differ in KV width, and a global
nKV would misclassify its narrow layers against its wide ones.
```

## cudaLayer.vNorm

Moved from `cuda/resident.go` (the comment above `cudaLayer.vNorm`) on 2026-10-09.

```text
vNorm: scale-less v_norm on this layer's V before it is stored. HF and the CPU apply it on EVERY Gemma 4 layer
that owns its K/V, K=V or not (S1.0, docs/tasks/task-multimodal-support-2026-10.md); here only kEqV layers did
until 2026-10-07, so every sliding layer of the resident 12B/26B/31B missed it. Implies kEqV or a real v_proj.
```

## cudaResident.expBatch

Moved from `cuda/resident.go` (the comment above `cudaResident.expBatch`) on 2026-10-09.

```text
The batch that replaced them. profWCalls/profSCalls still count logical COPIES (so the
per-token copy count is unchanged and comparable across the change); profSyncCalls counts
the SYNCHRONIZES, which is the quantity batching actually moves — ~240/token to ~40.
Per-kind timing is gone rather than kept at ~0: with the copies merely appended here and
issued together later, a "weight upload took Xµs" figure would name something that no
longer happens.
```

## cudaResident.hidCap

Moved from `cuda/resident.go` (the comment above `cudaResident.hidCap`) on 2026-10-09.

```text
hidCap is the hidden-state seam (P10 / docs/spec/08) a resident CUDA target would need
to feed a hidden-state drafter (DFlash, DSpark) — the resident analogue of
decoder.Model.ForwardCapture, which exists only on the CPU forward. NOT wired into
production yet (audit-2026-09-02.md N-34, checked 2026-09-11): SetHiddenCapture/
HiddenCapture have no non-test caller anywhere in the tree today — internal/serveapp's
--drafter flag attaches through decoder.DFlashDrafter/LoadDFlashDrafter, which does not
call into this seam, and decoder/*.go's only hidden-capture callers
(SetGemma4HiddenCaptureForTest and friends) are a SEPARATE, Gemma4-CPU-specific,
test-only mechanism. Built ahead of the caller that will use it, same shape as
decoder/mtp.go's Gate 1 adapter — a seam, not a claim that anything reaches it yet.

Distinct from layerCap above, deliberately. layerCap is a divergence-localization
probe: EVERY layer, a stream.Sync() and a download each, appended to an unbounded
buffer. At 36 layers that is 36 syncs per token — fine to bisect a bug with, far too
expensive to decode against. hidCap copies only the TAPPED layers into fixed slots,
so a 5-tap drafter costs 5, not 36.
```

## cudaResident.moe

Moved from `cuda/resident.go` (the comment above `cudaResident.moe`) on 2026-10-09.

```text
Sparse MoE. The router projection stays f32 (gemv_f32_a8) while the experts are int4:
the router's output steers a DISCRETE choice, so a quantization error near a tie does not
perturb the result slightly — it runs a DIFFERENT expert and the output is unrelated.
goinfer has already paid for that class once (the Granite SSM work traced a 66%-agreement
wall to discrete expert flips and proved no precision knob recovered it), so the router is
the one place in this backend where the cheap thing is not worth it.
```

## cudaResident.prefillChunkCap

Moved from `cuda/resident.go` (the comment above `cudaResident.prefillChunkCap`) on 2026-10-09.

```text
prefillChunkCap is the LEARNED row budget shared by every batched-prefill caller that has a
row-count knob to shrink: prefillChunked (which retries the SAME pass smaller and stores
whatever width worked) and PrefillImageLast (N-41, docs/audit-2026-09-10.md — it cannot
retry a bidirectional image block at a smaller width, since an image OOM is a function of
that block's own M, but still halves and stores the budget on OOM so the NEXT image call is
caught by its cheap pre-check instead of repeating the same real OOM). 0 until a pass OOMs,
then the width that worked (or, for the image path, the width to try next). It exists so a
card that cannot hold the default chunk is discovered ONCE rather than on every prompt.
Repeatedly driving the context to CUDA_ERROR_OUT_OF_MEMORY is not merely wasteful: per
backend.go's A13 note a context taken to refusal and kept in use can afterwards launch
kernels that "return SUCCESS and execute NOTHING". Atomic because both callers run on their
own CALLER's goroutine (only the per-pass job is serialized through the executor).
```

## cudaResident.passPromptLen

Moved from `cuda/resident.go` (the comment above `cudaResident.passPromptLen`) on 2026-10-09.

```text
passPromptLen is the TOTAL prompt length this batched pass belongs to (startPos+M), set once
at the top of prefillCore. The fast-prefill floor is a property of the PROMPT, not of the
chunk: prefillChunked splits a long prompt into passes of <=512 rows, so gating on M alone
would judge a 3900-token prompt by its 512-row chunk. Per-pass mutable state on the resident,
the same shape as prof above.

N-43 (docs/audit-2026-09-10.md): this is NOT cleared after prefillCore returns, and
residentDrafter.DraftBlock (cuda/drafter.go) reads it too, via aboveFastPrefillFloor/
bGemvB — the drafter never calls prefillCore, so it inherits whatever the TARGET model's
last prefill happened to set this to. See DraftBlock's own doc comment for why that
coupling costs acceptance rate at worst, never correctness.
```

## cudaResident.forceExactKernels

Moved from `cuda/resident.go` (the comment above `cudaResident.forceExactKernels`) on 2026-10-09.

```text
forceExactKernels disables useAttnFused/useGemmMMA (the L2/L3 fast levers) for the
DURATION of one prefillCore pass — the same "per-pass mutable state on the resident" shape
as passPromptLen above, set/cleared once at prefillCore's own r.do(...) boundary. M-09/M-10/
M-11 (docs/audit-2026-09-10.md): those levers are chosen purely on shape (M/K/position),
never on WHO is asking — HiddenLast's own norm-output tail and speculative verify's
tailAllLogits/tailAllArgmax tails both need decode-identical numerics (the whole point of
verifying, or of HiddenLast's bit-identity contract), which the fast levers do not carry
(they are cosine-close to gemv_w4a8_rn/attn_batched, not proven bit-identical — that is
exactly why they are opt-in performance levers rather than the default path). Set true for
every prefillCore tail except tailLastLogits (ordinary single-row-output prefill, where the
existing cosine-gated tolerance already applies and always has); the drafter's own forward
(cuda/drafter.go) never touches this field, so its bGemvB calls are unaffected — a proposal
never needs to be bit-identical to anything, only the target's verify of it does.
```

## cudaResident.logitsPinned

Moved from `cuda/resident.go` (the comment above `cudaResident.logitsPinned`) on 2026-10-09.

```text
logitsPinned is PAGE-LOCKED host memory for the per-token logits readback. A pageable
D2H of 594 KB measured only ~1.26 GB/s (it stages through a driver bounce buffer);
pinned memory DMAs straight out. Slice() is a zero-copy view, so Forward still returns
without an extra copy. Reused across calls (decode consumes each before the next).
```

## allocRoundSlack

Moved from `cuda/resident.go` (the comment above `allocRoundSlack`) on 2026-10-09.

```text
allocRoundSlack is what the driver adds to ONE buffer of n bytes: a buffer of a quantum or more is rounded up to
the next quantum, a smaller one is not (measured 2026-10-08 against the free VRAM at every allocation of a 7B load:
applying this to the exact packed-buffer sizes predicts the gap to the requested bytes within 3-5%, 406 MiB against
420 on the 7B; rounding the small ones too predicts 998 MiB).
```

## cudaResident

Moved from `cuda/resident.go` (the comment the comment above `var (` (it described the type, which has since been given its own doc)) on 2026-10-09.

```text
cudaResident is the production resident decode runner: the parity-green cgo-free forward
(TestRealForwardParity), promoted from the test harness. All CUDA state is owned by a
single LockOSThread-pinned executor goroutine (guardrail #3); Forward routes one channel
round-trip per token. Dense residency only (Qwen2/Llama, DecodeRunnerEligible), mixed
int4/int8/f32 weights as the real q4_k_m checkpoint stores them.
```

## cudaLayer.mscale

Moved from `cuda/resident.go` (the comment the comment above `mscale` in `cudaLayer`) on 2026-10-09.

```text
mscale: YaRN's attention_factor for THIS layer (decoder.Model.RopeMscaleLayer). 1.0 for
every family without YaRN, and 1.0 on a YaRN family's non-scaled layers — Mellum carries
1.2772588722239782 on its full-attention layers and 1.0 on the sliding ones, so this is
per-layer and not per-model. Passed to rope_kv / rope_kv_batched, which fold it into
cos/sin. Metal and WebGPU both carry the same per-layer value for the same reason.
```

## cudaResident.gemma4Dense

Moved from `cuda/resident.go` (the comment the comment above `gemma4Dense`) on 2026-10-09.

```text
gemma4Dense is true for ANY gemma4 checkpoint (dense or MoE) — broader than gemma4Moe,
which is enable_moe_block layers specifically. Gates compiling fScaleVec (the dense
per-layer-output-scalar kernel, segB's dense tail) even when gemma4Moe is false, since that
kernel was previously compiled ONLY as part of the MoE-only router_f32 module build even
though it has nothing router-specific about it.
```

## recordUpload

Moved from `cuda/resident.go` (the comment the comment above `recordUpload`) on 2026-10-09.

```text
recordUpload captures the FIRST alloc/upload error hit during BuildResident's setup job into
r.setupErr (audit C-08). The up* helpers used to discard gpu.Upload's error with `_ =`, so a failed
weight upload left a ZEROED device buffer and the build still returned ok=true — a resident that
decodes garbage. The setup job's last statement returns r.setupErr, which BuildResident turns into a
graceful decline (→ staged/CPU fallback); recording here is what makes that check ever fire. Only the
first error is kept (later uploads in the same doomed job are noise). Called only at load time.
```

## slotBytesPerLayer

Moved from `cuda/resident.go` (the comment the comment above `slotBytesPerLayer`) on 2026-10-09.

```text
slotBytesPerLayer is the device VRAM one slot's worth of BOTH expert projections costs (int4
weight + f16 scales), used to size the cache to free VRAM.

It must be measured from a ROUTED layer, not from layer 0 (audit C-25). Layer 0 is dense on every
family with `first_k_dense_replace` — GLM-4.5/4.6, DeepSeek-V2/V3, Kimi — so its expGU/expDown
strides are zero, and the caller's `budget / len(moeLayers) / perLayer` is then an integer divide
by zero. That panic raised on the EXECUTOR goroutine, which (before C-24) killed the process
rather than declining. Trigger: GOINFER_MOE_CACHE_EXPERTS=1 on any dense-prefix MoE.
```

## allocSlots.prediction

Moved from `cuda/resident.go` (the comment the comment in `allocSlots`) on 2026-10-09.

```text
One prediction now, because there is one implementation. These used to record BOTH the inline
copy's choice and capSlots' choice, precisely because they could disagree.
```

## loadRoutedExperts.batch

Moved from `cuda/resident.go` (the comment the comment in `loadRoutedExperts`) on 2026-10-09.

```text
One synchronize for the WHOLE layer: every expert-slot miss plus the per-token slot-index
upload the GEMV reads this round's routing from, folded into the SAME batch (P-21). This
used to be two separate calls — UploadBatch for the misses, then a lone gpu.Upload for
slotIdx — each paying its own synchronize; slotIdx is always present (hit or miss) and
tiny, so there was nothing to gain from keeping it apart. All destinations are slot buffers
on the one resident context, so UploadBatch's mixed-context refusal should never fire here
— if it does, something about the layer's buffers is not what this code believes.
```

## UploadProfForTest

Moved from `cuda/resident.go` (the comment the comment above `UploadProfForTest`) on 2026-10-09.

```text
UploadProfForTest reports the expert-DMA split: the big weight copies vs the tiny scale copies,
by bytes moved and COPY count. Zero unless GOINFER_MOE_CACHE_PROF is set.

The per-kind ELAPSED TIMES this used to return are gone, deliberately. The copies are now queued
by appendExpertSlot and issued together by one UploadBatch, so there is no longer a per-kind
upload to time; returning the append cost under the old names would have been an accessor whose
name promised a measurement it no longer makes. The transfer+sync time is BatchProfForTest's.
```

## runJob.stack

Moved from `cuda/resident.go` (the comment the comment in `runJob`) on 2026-10-09.

```text
The STACK, not just the value. A recovered executor panic becomes a resident
DECLINE, printed once to stderr — and "device allocation failed (0 bytes)" with no
frame is unactionable: it took four ~9-minute 35B load cycles to localize one to a
dense-FFN scratch buffer that a pure-MoE model has no width for. The stack is a few
KB on a path that runs at most once per model load.
```

## checkKVFits.slots

Moved from `cuda/resident.go` (the comment the comment in `checkKVFits`) on 2026-10-09.

```text
Slots before context (owner decision 2026-09-27): an unpinned context gives up positions, never below
cudaCtxCapDefault, until every requested slot fits. ctxForSlots planned this with Plan, which cannot see
the build's own scratch and kernel modules (34-120 MB measured, TestResidentDenseBytes_matchesCUDADevice),
so its choice can be a few dozen positions long; this trims it against the real free VRAM, before any KV
exists. Nothing allocated so far depends on ctxCap exactly: the split-KV score scratch is indexed by the
attended span, so a smaller cap uses less of it.
```

## launch.probe

Moved from `cuda/resident.go` (the comment the comment in `launch`) on 2026-10-09.

```text
A1 item 2, recording only: free VRAM immediately BEFORE the launch, i.e. on the other side of
the event from describeLaunchErr's reading. That reading is reached only after Launch returns
non-nil, so it cannot distinguish "memory was released before this launch was attempted" from
"the failed attempt released it while unwinding" — the two hypotheses differ by where the
probe sits, not by what happened. Recording the pre-launch value at every launch also yields
the decrement A9 asks for (free at first launch → free at the failing launch) from one run.
```

## describeLaunchErr.nofree

Moved from `cuda/resident.go` (the comment the comment in `describeLaunchErr`) on 2026-10-09.

```text
Deliberately NO free-VRAM reading here. This site is reached only after Launch has
returned non-nil, so any figure taken here is a POST-failure state and cannot be
distinguished by the reader from a pre-launch one — the number's meaning depends on where
the probe sits, which is not something an error string can carry. It was carrying one, and
the 64 MiB apparently "released" between two launches was an artifact of exactly that.
Free VRAM belongs to instrumentation, which can state its own probe position.

Also deliberately NOT suggesting a specific safe slot count: the computation that would
produce one is the thing under suspicion whenever this fires, and printing a number from
it would launder a suspect figure into advice.
```

## splitKVAttnDecode.sink

Moved from `cuda/resident.go` (the comment the comment in `splitKVAttnDecode`) on 2026-10-09.

```text
N-38 (docs/audit-2026-09-10.md, corrected 2026-09-16): gpt-oss is the only family with
an attention sink, and it IS resident-eligible (glu_quant_gptoss/route_gptoss and its
per-expert down bias are all real, wired kernels — see gptOssSw/gptOssRoute/
fMoEWaccBias above). r.sinkArg(l) below returns the real per-layer sink for it and
ArgNull() for every other family, so this is not an always-null argument; the kernel
itself is unchanged either way.
```

## splitKVAttnDecode.spike

Moved from `cuda/resident.go` (the comment the comment in `splitKVAttnDecode`) on 2026-10-09.

```text
3. V-sum → r.cctx (each thread the whole ascending-s fold for one output dim).
SPIKE PATH, opt-in only. Splits the V fold over S key chunks and combines in fixed order —
NOT bit-identical to attn_batched, which is why it is unreachable unless
GOINFER_SPLITKV_VSUM_SPLIT is set. Prices the trade the scoping doc's kill criteria decide on.
```

## decodeAttnGap.mustSplit

Moved from `cuda/resident.go` (the comment the comment in `decodeAttnGap`) on 2026-10-09.

```text
M-16: split-KV is REQUIRED, not merely preferred, once the single-block launch would
exceed the device's shared-memory limit. The `r.splitkvAttn` env gate and the
per-geometry perf threshold both describe when split-KV is FASTER; neither knows when
the alternative cannot run at all. Without this, -ctx 16384 on a model whose geometry
says splitkvNever (nH >= 24: Qwen2.5-7B, Llama-3-8B, phi3-mini) fails at position
12,160 — or silently drops to the sequential prefill.
```

## decodeAttnGap.coalesced

Moved from `cuda/resident.go` (the comment the comment in `decodeAttnGap`) on 2026-10-09.

```text
Coalesced M=1 decode attention: attn_batched with M=1 is BIT-IDENTICAL to the glue
`attention` (TestAttnBatched_bitIdentical) but reads K via float4 — 21.96%→98% bytes/sector.
ncu found the glue decode attention L1TEX-latency-bound at 2048 (~63% of the decode budget,
the 221→97 tok/s long-context deficit vs current Ollama); the coalesced read recovers it.
startPos=pos, M=1 → nKeys = pos+1; same GridX/block/shared/ctx-layout as the glue launch, so
decode stays byte-identical. glue `attention` (audited) is UNTOUCHED and is the fallback below.
```

## singleBlockAttnShmemLimit

Moved from `cuda/resident.go` (the comment the comment above `singleBlockAttnShmemLimit`) on 2026-10-09.

```text
singleBlockAttnShmemLimit is the dynamic shared memory a single-block attention launch may
request. The glue/batched attention kernels size their scratch (nWin+128)*4, with no ceiling.

M-16, MEASURED on the RTX 2070 SUPER (Turing) rather than inferred:

	CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK        49152 (48 KB)  -> nWin <= 12160 keys
	CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN  65536 (64 KB)  -> nWin <= 16256 keys

48 KB is the operative one: nothing here calls cuFuncSetAttribute to raise a kernel into the
opt-in range, so the driver enforces the default. Past 12,160 attended keys the launch is
refused — decode fails outright at that position, and batched prefill errors at layer 0 and
silently falls back to the ~9x slower sequential path.

Conservative by choice: the constant is the DEFAULT limit, not the opt-in one, because raising
it needs a per-kernel SetAttribute call that does not exist yet. If that lands, this becomes a
device query.
```

# Test files

Moved from the comments of the package's `_test.go` files (CC5); same conventions as above.

## TestA13_LoadModuleOnResidentExecutor

Moved from `cuda/a13_cell_test.go` (the comment above `TestA13_LoadModuleOnResidentExecutor`) on 2026-10-09.

```text
TestA13_LoadModuleOnResidentExecutor fills the missing cell of the 1×2 and is the CONTROL that
decides whether the multi-model unload null means anything.

The context factor is already eliminated by reading: CreateSystemDefaultDevice calls dev.Primary(),
gocudrv never binds cuCtxCreate, so the resident's context and a test's context are THE SAME
primary context. Two factors remain:

	A — module route : ctx.LoadModule(prebuilt PTX)   vs  CompileLibrary (NVRTC at runtime)
	B — launch site  : test goroutine                 vs  the resident's pinned executor

	known: (LoadModule, test)          POISONS
	known: (CompileLibrary, resident)  does not, with the stimulus applied through rf.do

This is (LoadModule, resident): a prebuilt PTX module loaded, launched, and stimulated entirely on
the resident's executor thread.

	poisons -> resident-executor launches ARE poisonable, so the multi-model null is real evidence,
	           and the variable is the MODULE ROUTE rather than the thread
	clean   -> nothing has ever poisoned this route; the multi-model null means nothing yet and the
	           harness question is still open
```

## TestA13_MultiModelUnloadPoisons

Moved from `cuda/a13_multimodel_test.go` (the comment above `TestA13_MultiModelUnloadPoisons`) on 2026-10-09.

```text
TestA13_MultiModelUnloadPoisons is the shipped-path question, and the one that decides the tag.

The correction that produced it: "each resident model builds its own context" is FALSE.
CreateSystemDefaultDevice calls dev.Primary() — the device's PRIMARY context, retained by
refcount — and Context.Close calls PrimaryCtxRelease, a decrement. gocudrv does not bind
cuCtxCreate at all. So every model in a process shares ONE context, destroyed only when the LAST
holder releases it.

Which means POST /admin/models/unload, with another model loaded, is a multi-gigabyte free INSIDE
A LIVE CONTEXT — the exact stimulus the A13 sweep showed can leave later launches returning
success and writing nothing. Not synthetic, not a test artifact: a shipped feature reached by
ordinary operation.

A is the 7B at int4 (~4.9 GB, ~67% of this card) so its release is the largest one a shipped path
can make on this device. The ">=25% reliable / 15% ambiguous" sweep bands this comment once cited
to justify that size are WITHDRAWN — that probe proved intermittent, and the real trigger is drain
to refusal, which an unload never does. The size is chosen as the worst realistic case, not as a
point inside a band. B is small, so both fit at once.

NO CONTROL IS NEEDED FOR A POSITIVE. If B's output degrades after A is unloaded, that is a
shipping correctness bug and the diagnosis stops there. A clean result is NOT clean yet — it needs
the (LoadModule, resident-executor) cell to show that this route is poisonable at all.
```

## TestA13_PrefillChurnPoisons

Moved from `cuda/a13_prefillchurn_test.go` (the comment above `TestA13_PrefillChurnPoisons`) on 2026-10-09.

```text
TestA13_PrefillChurnPoisons is the tag-blocking measurement.

A13 established that a large hold-and-release INSIDE A LIVE CONTEXT can leave later launches
returning success and writing nothing. Four production paths were enumerated and are clean by
construction — admin unload destroys the context, capSlots is pure arithmetic, allocSlots discards
the resident on failure, and no mid-life KV/expert resize exists. One is not:

	cuda/prefill.go:cudaResident.prefillCore allocates `scratch` per call and releases it with a deferred
	r.dev.ReleaseBuf loop, inside the live resident context. Its own comment: "at M=3000 this is
	hundreds of MB".

And that release really does return memory to the driver: ReleaseBuf -> Buffer.Close ->
cudaresult.MemFree. No pool, no reuse. So the stimulus occurs on the hot path on every long
prompt, and whether it poisons is a measurement rather than an argument.

SIZE IS NOT AN ARGUMENT HERE, and the percentages this comment used to quote are WITHDRAWN. They
came from a synthetic hold-and-release probe that later proved INTERMITTENT (C C C C P C on a
repeat), so its "reliably clean <=12%" and "poisons at >=25%" bands were reading noise as
structure. The real trigger is DRAIN TO REFUSAL — deterministic, 5/5. What actually closes prefill
is a measurement of its own peak: min free 5752.2 MiB during a real-model prefill, 39.9x the
refusal floor. Only measurements count, and the one that counts here is that one.

TWO SYMPTOMS, reported separately because they mean different things:

	(a) the prefill's own logits degrading across repetitions -> a correctness bug in SHIPPED output
	(b) a probe launch on the same context failing afterwards  -> narrower, still real

POSITIVE CONTROL (GOINFER_A13_CHURN_CONTROL=1): reproduce the known poisoning stimulus in this same
process and code path and confirm it DOES poison. A clean result from a harness that cannot poison
is not evidence — the fourth time in this campaign a null needed its forcing mechanism verified
before it meant anything.
```

## TestA13_PrefillChurnPoisons.control

Moved from `cuda/a13_prefillchurn_test.go` (the comment above `TestA13_PrefillChurnPoisons.control`) on 2026-10-09.

```text
THE CONTROL MUST STIMULATE THE CONTEXT UNDER TEST. A first version allocated through
dev.Primary() and showed nothing — because BuildResident creates its OWN context, so the
primary context is a different one and the control never touched the subject. That is the
"harness that cannot poison" failure, caught by running the control before believing a
clean result rather than after.

So this allocates and frees through the RESIDENT's device, on the resident's pinned
executor thread (r.do), which is the only place its context is current.
```

## TestA13_SingleFailedAllocPoisons

Moved from `cuda/a13_singlefail_test.go` (the comment above `TestA13_SingleFailedAllocPoisons`) on 2026-10-09.

```text
TestA13_SingleFailedAllocPoisons asks the question that decides whether A13 is a test defect or a
production bug: does ONE failed allocation poison the context, or does it take a full drain?

A13 established that after draining the device to exhaustion, a later `attention` launch returns
success and writes nothing. Every draining test allocates until refusal — hundreds of failures and
gigabytes held. Production never does that deliberately. But production DOES hit a failed
allocation: BuildResident sizes the expert cache against free VRAM and can have an allocation
refused, and a multi-model server can have one model's OOM land in a process another model is
using.

So the shape that matters is not "drain" but "how little is enough".

	(c) correct -> one failure does not poison; bisect upward to bound what does
	(c) zeros   -> ANY failed allocation poisons, multi-model servers are exposed, and one model's
	               OOM silently breaks another model's CUDA path in the same process with no error

Three repeats either way: A13's failures vary run to run, so a single negative clears nothing.
```

## TestA13_SingleFailedAllocPoisons.pin

Moved from `cuda/a13_singlefail_test.go` (the comment above `TestA13_SingleFailedAllocPoisons.pin`) on 2026-10-09.

```text
A13 item 1: PIN THE GOROUTINE. Every observed poisoning has been on a test goroutine, which Go
is free to migrate across OS threads; the resident's executor is LockOSThread-pinned and has
never poisoned. If pinning alone makes this clean, the mechanism is unpinned CUDA usage from a
migrating goroutine rather than driver-side module eviction — and the eviction story, the
cache-site comment, and everything downstream of it are wrong.
```

## TestA13_SingleFailedAllocPoisons.keep

Moved from `cuda/a13_singlefail_test.go` (the comment above `TestA13_SingleFailedAllocPoisons.keep`) on 2026-10-09.

```text
A13 item 1: PERSISTENT ALLOCATION (GOINFER_A13_KEEP=<MiB>). Held for the whole test and never
freed, so the context's LIVE SET never collapses to empty.

The hypothesis it tests: the trigger is not memory pressure but the live set going (near)
empty — every poisoning run frees everything it allocated, while a resident context always
holds a model's weights. If holding ~1 GB makes the known-poisoning sequence clean, that one
variable explains the resident executor, prefill churn and multi-model unload together, and
converts four separately-measured nulls into one predicted property.
```

## TestA13_SingleFailedAllocPoisons.nfail

Moved from `cuda/a13_singlefail_test.go` (the comment above `TestA13_SingleFailedAllocPoisons.nfail`) on 2026-10-09.

```text
(b) EXACTLY ONE failed allocation. Not a drain: one request, larger than the whole device, and
nothing retained. If it succeeds the probe is void, so that is checked rather than assumed.
N failed allocations, N=1 by default. GOINFER_A13_NFAIL bisects upward: the point of the
sweep is to bound how little is enough, since "a full drain poisons" and "one refusal does
not" leave the interesting range unmeasured — and the decline path's own attempt count has
to sit inside whatever bound comes out.
```

## TestA13_SingleFailedAllocPoisons.holdpct

Moved from `cuda/a13_singlefail_test.go` (the comment above `TestA13_SingleFailedAllocPoisons.holdpct`) on 2026-10-09.

```text
PARTIAL DRAIN (GOINFER_A13_HOLDPCT): successfully hold a percentage of free VRAM, then
release it. 1000 refusals turned out to be harmless, which points at the SUCCESSFUL
allocation rather than the refusal — the draining tests hold gigabytes before anything is
refused. This is the knob that separates the two.
```

## TestActGroup_phi3ResidentMatchesCPU.bars

Moved from `cuda/actgroup_phi3_resident_test.go` (the comment above `TestActGroup_phi3ResidentMatchesCPU.bars`) on 2026-10-09.

```text
The kernels are responsible for matching the CPU per-32 path, not for the prompt's own
difficulty: on this prompt the CPU per-32 path itself sits at p10 ~0.93 against f32 (measured
2026-09-25; the gate's prompt, without the "\n\n", gave 0.973). So the bars are relative:
median agreement with CPU per-32, and quality no worse than CPU per-32's on the same prompt.
Measured: agreement median 0.99983; quality p10 0.934 (CUDA) vs 0.930 (CPU).
```

## TestAllocFloor

Moved from `cuda/alloc_floor_test.go` (the comment above `TestAllocFloor`) on 2026-10-09.

```text
TestAllocFloor measures how far cuMemAlloc will actually drain the device, against what
cuMemGetInfo reports as free at that moment.

A10's ordering hypothesis predicted that issuing the largest slot buffers first would let the
sequence complete. It did not. It got 27 MiB further (failing with 155,385,856 B free instead of
182,648,832) and allocated more total bytes, but still failed — and the failing request was
4,212,736 B against 155,385,856 B free, a ratio of 36.88. A 4 MiB request refused with 148 MiB
free is not a contiguity story.

Both failures sit in the same band regardless of request size, which reads as a FLOOR: some
quantity cuMemGetInfo counts as free that cuMemAlloc will not hand out. This measures it directly,
with no model and no 26B: drain in shrinking chunks until even a 2 MiB request is refused, then
report what free says.
```

## TestAllocFloor.defer

Moved from `cuda/alloc_floor_test.go` (the comment above `TestAllocFloor.defer`) on 2026-10-09.

```text
FREE THE DRAIN, ON EVERY EXIT PATH. This test deliberately allocates until the device
refuses a 2 MiB request — that is what it measures — and every buffer stays reachable in
`hold` so the GC cannot reclaim one mid-measurement. Without an explicit release the
process then carries an EXHAUSTED device into every later test in the package.

That is not hypothetical: it made TestAllocGranularity fail with CUDA_ERROR_OUT_OF_MEMORY
on a 5 MiB allocation whenever it ran after this test, and the GPU gate reported it as
"a CUDA forward moved" — a numerics-sounding verdict for a bookkeeping leak. Bisected:
TestAllocFloor+TestAllocGranularity fails, TestA10Floor...+TestAllocGranularity passes.

THIS IS A defer RATHER THAN A TAIL BLOCK because the function can now exit early: the
foreign-CUDA-context skip below returns via runtime.Goexit and would jump straight past a
trailing release. Adding that skip reintroduced the exact leak this comment documents —
TestMoERouteDemandThreshold went from a 3.8 s bisection to a 0.02 s failure on a drained
device — which is why the cleanup is now structural instead of positional.
```

## TestAllocFloor.pin

Moved from `cuda/alloc_floor_test.go` (the comment above `TestAllocFloor.pin`) on 2026-10-09.

```text
THE VALUE IS NOW PINNED, AND IT WAS NOT BEFORE. That absence had a consequence worth stating,
because "not a threshold assertion: the number is the finding" was a deliberate and reasonable
choice when nothing depended on the number — and then something did.

TestMoERouteDemandThreshold asserts demand == floor + residual and hardcoded this floor,
describing it as "pinned by its own gate". It was not: this test reported the floor and
asserted only the margin relation, which a SMALLER floor satisfies more easily. So when the
floor halved on 2026-08-21 (151,191,552 -> 54,263,808, no reboot, no driver change, nothing in
the tree), this gate stayed green and the demand gate went red accusing the KERNEL of moving.
The identity in fact closed to the byte against the new floor. An unpinned number that another
gate depends on does not stop being load-bearing; it just stops being watched.

Pinned to a WINDOW, not a byte: this is a machine property, and a byte-exact pin on a machine
property is what made the demand pin brittle in the first place. Re-derive on a new box or a
driver change — and when you do, re-check the demand identity, which is downstream of this.

RE-DERIVED 2026-08-26 for the driver/distro re-anchor (P16): 54,263,808 -> 1,769,472. The box
went Nobara 43 -> 44 overnight, carrying NVIDIA 595.58.03 -> 595.91.07, kernel 7.0.5 -> 7.2.0
and glibc with it, so this is the "driver change" the paragraph above anticipated. Measured
three times in three separate processes, byte-identical each time. The instruction that
paragraph gives was followed rather than skipped: the demand identity was re-checked and it
CLOSES TO THE BYTE against the new floor —

	1,769,472 (floor) + 138,412,032 (residual) = 140,181,504 = the measured demand

— with the residual independently re-measured and UNCHANGED (TestMoERouteFirstLaunchReservation
PASSES at 138,412,032). So a component moved and the downstream pin follows it; the kernel did
not move, and A1/A5/A7/A9 do not need re-deriving. Safety direction is the same one as
2026-08-21 and it is checked, not assumed: a SMALLER floor means less memory is reported free
but unallocatable, so there is MORE headroom than the cap analysis assumed. The floor is now
small enough that the window's lower edge falls below zero — the bottom is guarded by the
`floor <= 0` fatal above, and "moved" can now only mean "grew".
```

## TestAllocFloor.previous

Moved from `cuda/alloc_floor_test.go` (the comment above `TestAllocFloor.previous`) on 2026-10-09.

```text
Kept so a REVERSION is recognisable rather than reading as a move somewhere new. Both prior
values were measured on THIS box: the machine has now produced three different floors without
the tree changing once.
```

## TestAllocFloor.foreign

Moved from `cuda/alloc_floor_test.go` (the comment above `TestAllocFloor.foreign`) on 2026-10-09.

```text
The pin is a property of an EXCLUSIVE device. A foreign CUDA context
raises this floor (measured +16 MiB for KDE's compositor, 2026-09-01),
so asserting the exclusive number against a desktop session reports a
machine change that has not happened -- and this gate's failure text
then sends the reader toward TestMoERouteDemandThreshold and the
A1/A5/A7/A9 pins over a window manager.

So: SKIP rather than fail, and say exactly why. A skip is not a pass --
`gate gpu` lists it as uncovered, which is the honest report -- whereas
a red that everyone learns to expect on a desktop is a gate nobody
reads.
```

## TestAllocFloor.relationship

Moved from `cuda/alloc_floor_test.go` (the comment above `TestAllocFloor.relationship`) on 2026-10-09.

```text
THE RELATIONSHIP, pinned. Leftover after allocSlots must clear the floor, and the margin is
what guarantees it. Every observation fits:

	cap 31 -> leftover 501,415,936  > floor  -> works
	cap 33 -> leftover 312,672,256  > floor  -> works
	cap 34 -> leftover  61,014,016  < floor  -> fails mid-allocation

It also retires a figure A9-MARGIN nearly recommended. A 128 MiB margin (134,217,728) is BELOW
this floor; the cap-33 run under it worked only because that cap's leftover happened to be
312 MiB. That was luck, not safety, and the assertion below is what turns the distinction into
something a test can see.
```

## TestA10ReportingGap

Moved from `cuda/alloc_reporting_gap_test.go` (the comment above `TestA10ReportingGap`) on 2026-10-09.

```text
A10 model under test: cuMemGetInfo reports ~151,191,552 B more free than is allocatable, by
anyone, so usable = reported_free - gap.

TestA10ReportingGap is the cheap cross-check: no launch, no balloon-and-bisect. Allocate directly
in a fresh context until even a 1 MiB request fails, and compare the total actually obtained
against the free figure reported at the start. If the shortfall is ~151 MiB, the reporting gap is
confirmed with no kernel involved at all — which separates "the allocator reserves" from anything
about launches.
```

## TestA10FloorIsPerProcessOrPerDevice

Moved from `cuda/alloc_reporting_gap_test.go` (the comment above `TestA10FloorIsPerProcessOrPerDevice`) on 2026-10-09.

```text
TestA10FloorIsPerProcessOrPerDevice varies the CONTEXT rather than the kernel — the axis the floor
has never been tested against.

A child process drains to the floor and HOLDS. This process, with its own context, then reads free
and tries to allocate.

	parent can still allocate -> the floor is per-process/per-context; N contexts cost N x the
	                             reserve, and the margin is NOT a constant
	parent cannot             -> one device-wide reserve, and the margin CAN be derived from it

RESULT 2026-08-12: NEITHER — the parent cannot create a context at all while the child holds
(cuDevicePrimaryCtxRetain: CUDA_ERROR_OUT_OF_MEMORY at 151,191,552 B reported free). That is a
finding on its own — the floor is not available for context setup either — but it means this arm
cannot measure what it was built to measure. The in-process arm is blocked too: gocudrv exposes
only primary-context retain, not cuCtxCreate, so a second simultaneous context cannot be made.

What IS established: the floor is 151,191,552 B in every separate process measured, so it is a
stable per-device property rather than something accumulating per process. Whether two SIMULTANEOUS
contexts each pay it is untested and untestable with the current API surface.
probeFreeWithoutContext reads free VRAM from nvidia-smi, which needs no CUDA context — so it can
be read BEFORE this process retains one, which cuMemGetInfo cannot.
```

## TestA10FloorIsPerProcessOrPerDevice.marked

Moved from `cuda/alloc_reporting_gap_test.go` (the comment above `TestA10FloorIsPerProcessOrPerDevice.marked`) on 2026-10-09.

```text
Marked AFTER the child branch is checked, not before: the child arm IS the drain, and it is
spawned with GOINFER_A10_DRAIN_CHILD set, so gating it on the group flag as well would be
redundant. The parent is marked because it holds a live context against a device its own child
has taken to ~300 MiB — the near-floor half of the same hazard.
```

## TestA10FloorIsPerProcessOrPerDevice.leave

Moved from `cuda/alloc_reporting_gap_test.go` (the comment above `TestA10FloorIsPerProcessOrPerDevice.leave`) on 2026-10-09.

```text
Leave ~300 MiB reported free rather than draining to the floor. The first attempt drained
completely, and the parent then could not create a context at all — context setup needs
memory the floor does not provide, so the arm could not measure what it was built for.
```

## TestA10FloorIsPerProcessOrPerDevice.instrument

Moved from `cuda/alloc_reporting_gap_test.go` (the comment above `TestA10FloorIsPerProcessOrPerDevice.instrument`) on 2026-10-09.

```text
BOTH readings from nvidia-smi. An earlier version took `pre` from nvidia-smi and `post` from
cuMemGetInfo, so the delta silently carried the disagreement between two instruments (~832 KiB
here) as if it were context cost. Same shape as the measurement-shape class: the number was
real and the comparison was not like-for-like.
```

## TestAllocGranularity

Moved from `cuda/allocgran_test.go` (the comment above `TestAllocGranularity`) on 2026-10-09.

```text
TestAllocGranularity records what this driver actually charges for a device allocation. It exists
because the expert-cache sizing arithmetic must predict device consumption, and summing requested
bytes does not.

MEASURED (RTX 2070 SUPER, driver 595.58.03, 2026-08-11) — the numbers, not just the conclusion:

	request         actual/alloc   overhead
	1 048 576 B ->   1 048 576 B     +0.0%
	1 048 577 B ->   2 097 152 B   +100.0%
	1 052 672 B ->   2 097 152 B    +99.2%
	2 973 696 B ->   4 194 304 B    +41.0%   (int4 weights, one 26B expert)
	3 490 000 B ->   4 194 304 B    +20.2%

Those four large samples ALL sit just above a power of two, where next-power-of-two and 2 MiB
granularity predict identically — so they cannot separate the two hypotheses, and reading them as
"rounds to the next power of two" was a name asserted from data that did not constrain it. The
discriminating requests are 5 / 6 / 9 MiB:

	5 MiB -> actual  6.00 MiB    nextPow2 says  8    2 MiB-granular says  6
	6 MiB -> actual  6.00 MiB    nextPow2 says  8    2 MiB-granular says  6
	9 MiB -> actual 10.00 MiB    nextPow2 says 16    2 MiB-granular says 10

Unanimous for 2 MiB granularity. nextPow2 would over-charge by up to 2x on any buffer that does
not happen to sit just above a power of two, under-granting slots on a future geometry — a new
mis-estimate introduced by the fix for the old one.
```

## TestSmallAllocPool

Moved from `cuda/allocgran_test.go` (the comment above `TestSmallAllocPool`) on 2026-10-09.

```text
TestSmallAllocPool records that sub-granularity allocations are NOT free, which a single
allocation appears to show and cannot: one 371 712-byte request measured ZERO device bytes,
because the pool page had already been charged.

MEASURED, same box, 371 712 B (f16 scales for one 26B expert):

	after   1 alloc  ->   2.00 MiB total   (the page, charged once)
	after 2..4       ->   +0.00 MiB        (drawn down, marginal cost zero)
	after  64        ->  +24.00 MiB
	after 128..512   ->  +26.00 MiB per 64
	512 allocs       -> 206.00 MiB total = 421 888 B/alloc amortised

So the marginal cost is zero until the page exhausts and then it steps. 421 888 B amortised
against a 371 712 B request is the honest figure at the counts that matter (30 layers x N slots).
```

## TestAttnBlockFull_nonCausal.compile

Moved from `cuda/attn_block_test.go` (the comment above `TestAttnBlockFull_nonCausal.compile`) on 2026-10-09.

```text
Compiled HERE, not bound at model load. The kernel has no production consumer yet
(the resident drafter path is not built), and TestPipelineLint_boundKernelsAreLaunched
exists precisely to stop a kernel being NVRTC-compiled into every model load while
nothing launches it — that was gemv_w4a8_batched's exact history. When the drafter
path lands it binds this at load; until then the only launch site is this gate.
```

## TestAttnBlockFull_cost

Moved from `cuda/attn_block_test.go` (the comment above `TestAttnBlockFull_cost`) on 2026-10-09.

```text
TestAttnBlockFull_cost closes the last substitution in the gate-3 composition.

The round-composition measurement stood in for the drafter with the target's stack truncated
to 5 layers — right shape, right weights-per-layer, but its attention is CAUSAL where the
drafter's is non-causal. Every row attending ALL keys is strictly more work than row m
attending m+1 of them, so the stand-in could only have UNDERSTATED the draft.

This times both kernels at the drafter's real geometry, which is the only remaining way the
8.82 ms draft could be wrong in the optimistic direction.

	GOINFER_HEAVY_TESTS=1 GOINFER_CUDA_MODEL=$HOME/models/qwen3-4b \
	  go test -tags 'cuda goinfer_testhooks' -run TestAttnBlockFull_cost -v
```

## TestAttnBatchedBandwidth

Moved from `cuda/attn_bw_test.go` (the comment above `TestAttnBatchedBandwidth`) on 2026-10-09.

```text
TestAttnBatchedBandwidth times attn_batched in isolation at the real qwen2.5-coder-1.5b attention
shape and M=2048 (nH=12, nKV=2, hd=128, full attention), the ncu target for the attention-lever
decision. It exists to be profiled: `ncu --kernel-name attn_batched ... /tmp/attnbench
-test.run TestAttnBatchedBandwidth`. The question it must answer BEFORE any tiling design — is the
33×-off-compute a TRAFFIC bound (K/V re-read from L2, which shared-memory tiling fixes) or a LATENCY
bound (like the GEMV, where the same shared-staging change bought only 1.2×)?
```

## attnFusedTolerance

Moved from `cuda/attn_fused_test.go` (the comment above the tolerance constants) on 2026-10-09.

```text
TOLERANCE — PRE-REGISTERED, THEN MEASURED TO BE MIS-DERIVED, AND CORRECTED IN THE OPEN.

The first version of this file pre-registered "max |delta| <= 1e-3 of THE ROW'S OWN max |ctx|"
plus "cosine >= 0.9999 per row", derived from f16 operand rounding. That bar failed widely --
worst cosine 0.9642, worst row-relative delta 1.6 -- and the failure was NOT the kernel.

What settled it (docs/measurements/prefill-l2l3-phase1-2026-09-05.md records the run): scoring
attn_fused against exact f64 math on inputs FIRST ROUNDED TO f16 -- which is what the kernel
actually receives -- gives cosine 1.00000000, worst 0.99999996 over 64 rows x 4 heads. The kernel
reproduces its own inputs' arithmetic essentially exactly. Meanwhile attn_batched scores
1.00000000 against the f32 reference, so the exact path is sound too, and the gap between them is
the f16 operand precision and nothing else.

THE DERIVATION'S ERROR was the DENOMINATOR, not the numerator. It assumed |ctx| ~ |V|/sqrt(nKeys).
With synthetic V of quasi-random sign the weighted average can cancel far more deeply than that:
measured here, |ctx|/rms|V| ran from 1.02 down to 0.000211, and the cosine gap tracked it
monotonically (1.02 -> 0.99999997; 0.0099 -> 0.99996; 0.00021 -> 0.9642). A row whose context
cancels to one part in 4700 is ill-conditioned by construction: rounding the INPUTS alone rotates
it 27%, so no kernel of any quality can meet a bar scaled by that row's own |ctx|.

THE CORRECTED BARS, and why each is the right shape:

	attnFusedMaxDeltaVsV -- max |delta| relative to max |V| over the head, NOT to |ctx|. |V| is the
	scale the output is drawn from and does not collapse, so this is well-conditioned everywhere
	while still catching any real defect: a kernel that attends the wrong keys, mismaps a fragment
	or drops a seam is wrong by a fraction of |V|, not of |ctx|.

	attnFusedMinCosine -- kept, but applied only to rows that are actually conditioned enough to
	carry a direction (|ctx| >= attnFusedCondFloor * rms|V|). The conditioning of every row is
	REPORTED either way, and the count of rows excluded is reported too, so this cannot silently
	become a bar that tests nothing.

TestAttnFused_vsF16Reference below is the logic gate that does not depend on any of this: it is
immune to conditioning because it compares the kernel against its own inputs' exact arithmetic.
```

## TestAttnFused_vsF16Reference

Moved from `cuda/attn_fused_test.go` (the comment above `TestAttnFused_vsF16Reference`) on 2026-10-09.

```text
TestAttnFused_vsF16Reference is THE logic gate for attn_fused, and the one assertion here that
no amount of input conditioning can distort.

TestAttnFused_vsExact compares two GPU kernels that use different operand precision, so a
disagreement there cannot by itself say which one is wrong — the same trap docs/task-prefill-
gap.md §3.1 corrected at the model level, where a fast path was scored against an exact path that
was itself a quantisation, and the distance was booked against the faster one. This test avoids
it by scoring attn_fused against EXACT f64 arithmetic on ITS OWN INPUTS, rounded to f16 exactly
as the kernel rounds them. Any error left is the kernel's logic: a mismapped mma fragment, a
dropped seam, the wrong keys attended, a botched online rescale. Operand precision is factored
out by construction rather than budgeted for.

Small shapes on purpose: the reference is O(M · nH · nKeys · hd) in Go and this needs to stay a
test, not a benchmark. Seam BREADTH is TestAttnFused_vsExact's job; DEPTH of correctness is this
one's. Both are needed — neither substitutes for the other.

	GOINFER_HEAVY_TESTS=1 GOINFER_CUDA_FAST_PREFILL=1 go test -tags 'cuda goinfer_testhooks' ./cuda/ -run TestAttnFused_vsF16Reference -v
```

## TestAttnFused_vsF16Reference.reference

Moved from `cuda/attn_fused_test.go` (the comment above `TestAttnFused_vsF16Reference.reference`) on 2026-10-09.

```text
Reference: exact f64 over f16-rounded operands, ROUNDED AND SEQUENCED THE WAY THE
KERNEL DOES IT.

A one-pass softmax over the global max is mathematically equal to the online form but
NOT numerically equal, and modelling it that way is a real error rather than a nicety:
the kernel rounds each tile's PROVISIONAL weights exp(s - m_running) to f16 and then
rescales the f32 accumulator by exp(m_old - m_new), so the f16 rounding happens at a
different scale than a global-max reference would apply. Measured, before this was
fixed: a global-max reference put the multi-tile cases at cosine 0.9992-0.9994 while
single-tile cases sat at 0.99999996 — a gap that reads exactly like a rescale defect
and is in fact the reference not modelling the algorithm. So the reference walks the
same BN-key tiles in the same order and carries the same running state.
```

## TestAttnFusedTile_defaultBitIdenticalWholeModel

Moved from `cuda/attn_fused_tile_model_test.go` (the comment above `TestAttnFusedTile_defaultBitIdenticalWholeModel`) on 2026-10-09.

```text
TestAttnFusedTile_defaultBitIdenticalWholeModel is gate 2 of attn-fused-tile128-default-PREREGISTERED.md: a whole chunked
prefill with the DEFAULT selector equals the same prefill with the 64x64 kernel forced (attnTile = -1), bit for bit on
the last-row logits. Every earlier row's K/V reach the last row's attention, so an earlier-chunk difference would surface
there. It also asserts WHICH kernel ran (tile128Launches): the hd128 model must use the 128-row tile on every layer of
every chunk by default and never when forced; the hd64 model must not use it at all (measured ~5% slower there).
```

## TestAttention_TailPoison

Moved from `cuda/attn_headdim_test.go` (the comment above `TestAttention_TailPoison`) on 2026-10-09.

```text
TestAttention_TailPoison is the scratch-to-max gate for the per-layer geometry port
(9a-P2). The runner allocates ONE Q/context scratch sized to the WIDEST layer (maxQDim =
nH*512 for Gemma 4's global head), then runs the NARROW hd=16 local layer into it. If the
attention kernel — or anything sizing off the allocation — reads past nH*hd, it picks up
the residue a previous wide layer left in the tail.

A zeroed scratch cannot catch that: an over-read folds in zeros, contributes nothing to
the dot products / accumulations, and the hd=16 result stays correct — so
TestAttention_HeadDimWidths passing at hd=16 proves nothing about tail-reads. The residue
is non-zero, so the tail is memset to a sentinel and only the live hd=16 region written;
the kernel then either stays byte-identical to the tight run (no over-read) or diverges
(found the bug the zeroed test structurally could not see). It also closes the consumer-
sizes-off-the-buffer gap the compiler removal alone cannot: anything deriving its extent
from the wide allocation processes sentinel and diverges.
```

## TestAttention_HeadDimWidths

Moved from `cuda/attn_headdim_test.go` (the comment above `TestAttention_HeadDimWidths`) on 2026-10-09.

```text
TestAttention_HeadDimWidths guards that the shipped `attention` kernel is correct across the
head-dim widths goinfer's arch set uses — including the ones NO other cuda test exercises. It drives the SHIPPED `attention` kernel at hd 16/64/128/256/512 through
the known-good validateGlue oracle (cosine vs a CPU GQA online-softmax reference). 128 is the
existing green control; 256 is gemma3's width (already resident); 512 is the gemma4 global-head
question the Phase-9a spec gates on; 16 and 64 are the SMALL end — gemma4's local layer is
hd=16, below every previously-tested width. The kernel decomposes each head over a fixed
128-thread block, so the large end (512 = 4 elems/thread) and the small end (16 = 112 of 128
threads idle) stress different assumptions: any hd ≥ blockDim or blockDim % hd == 0 dependence
would break at 16, not 512. Adding these rows keeps a red on the Split-A resident run
attributable to the geometry seam, not to the tiny head. Single variable = hd.
```

## attn_sink_test.header

Moved from `cuda/attn_sink_test.go` (the comment at the top of the file) on 2026-10-09.

```text
Attention-sink kernel gate (gpt-oss residency, step 1 of 2).

The sink is a learned per-head logit with NO key and NO value. It joins the softmax MAX
and the DENOMINATOR but never the numerator:

	m     = max(max_s score_s, sink_h)
	denom = Σ_s exp(score_s − m) + exp(sink_h − m)
	out_d = Σ_s exp(score_s − m)/denom · v[s][d]

WHY IT CANNOT BE A POST-HOC DENOMINATOR FIX, which is the natural-looking shortcut: the
sink competes for the max, so folding it in after the exp pass would need every exponent
recomputed against a new maximum. Getting that wrong is invisible whenever the sink is
below the score maximum — i.e. in most random tests — and wrong exactly when it is not.
This test therefore includes a case where the SINK DOMINATES.

It gates the kernel directly rather than through a model, because gpt-oss cannot be
resident yet (its clamped-SwiGLU expert kernel does not exist). That is the point of doing
the sink first: it is independently checkable.
```

## TestGenerateBlockSpec_production.prompt

Moved from `cuda/blockspec_test.go` (the comment above `TestGenerateBlockSpec_production.prompt`) on 2026-10-09.

```text
GOINFER_TEST_PROMPT selects the workload. Chat is the case the acceptance guard EXISTS
for: unguarded it measures 0.61x (1.96 accepted/round against a ~3.0 break-even), and
"the guard makes that safe" has so far been an inference from a different losing case
(thinking mode) rather than a measurement of this one.
```

## TestGenerateBlockSpec_production.attach

Moved from `cuda/blockspec_test.go` (the comment above `TestGenerateBlockSpec_production.attach`) on 2026-10-09.

```text
Attach ONCE — the weight upload is a per-process cost, not a per-request one. Timing the
attach inside the generation is what made the first version of this path measure 0.17x.
```

## TestGenerateBlockSpec_production.baseline

Moved from `cuda/blockspec_test.go` (the comment above `TestGenerateBlockSpec_production.baseline`) on 2026-10-09.

```text
THE BASELINE IS Model.Generate — the path a server actually takes.

An earlier version used a PrefillLastNArgmax(M=1) loop, which downloads the full 608 KB
logit row per token; Generate uses launchToken with the GPU argmax fast-path (a 4-byte
readback). That made the baseline slower than production and flattered every speedup
measured against it. Comparing against anything but the real path is measuring the wrong
thing.
```

## TestBuildScratchAccounting_plantedDefect

Moved from `cuda/build_scratch_accounting_test.go` (the comment above `TestBuildScratchAccounting_plantedDefect`) on 2026-10-09.

```text
TestBuildScratchAccounting_plantedDefect plans WITHOUT the slack (the plan as it was) and shows what each half of the change buys, against the same build with it:
  - the 7B: without the slack the device is further from Plan than the whole margin (425 MiB against 384, the bound the old TestResidentDenseBytes enforced), with
    it the residual is under 64 MiB;
  - every model: the residual with the slack is smaller than without it;
  - Gemma 3 4B, the one bench model whose plan trims at the build: the trim (planned minus final context) is smaller with the slack than without it.

The registered G-M2 said the 7B and Gemma 3 4B "trimmed before"; the 7B did not (it plans 16384 and keeps 16384 either way), so the 7B's evidence is the
residual, not a trim. Heavy.
```

## TestResidentCloseSettleTime

Moved from `cuda/closesettle_test.go` (the comment above `TestResidentCloseSettleTime`) on 2026-10-09.

```text
TestResidentCloseSettleTime measures how long free VRAM takes to stop rising after Close returns.

WHY THIS EXISTS, and it is the one datum the A12 retraction produced. While disproving the "leak",
the tracer's tail showed free VRAM still climbing as the process exited — 2.4 -> 4.3 -> 6.2 GiB
across three 50 ms samples. Close() had returned; the driver had not finished. If that interval is
long relative to the gap between tests, the next test's Load starts inside it and sees a card that
is still handing memory back, which reproduces every symptom the CUDA tier shows: passes alone,
fails in suite, VRAM signature, and no leak anywhere.

Cross-package parallelism is already excluded — the GPU gate passes -p 1 on every CUDA invocation,
targets the single ./cuda/ package rather than ./cuda/..., and the package contains zero
t.Parallel() calls. So the tests ARE sequential, and "sequential" is exactly what makes a
non-instant teardown matter: nothing else is running, but the previous test may not be finished.

It reports rather than asserts. A threshold pulled out of one machine's timing would be the
stale-constant shape this queue keeps finding; what a stabilisation wait should be, if the tier
needs one at all, follows from the number rather than preceding it.
```

## TestCohereRealResidentParityCUDA

Moved from `cuda/cohere_real_resident_test.go` (the comment above `TestCohereRealResidentParityCUDA`) on 2026-10-09.

```text
WHY IT EXISTS. On 2026-10-01 these two checkpoints, on this resident with the NeoX half-split rope kernels,
read per-position resident-vs-CPU worst cosine -0.075 (R7B) / -0.041 (Aya) on a 48-token prompt, exact at
position 0 and diverging after, and Aya's greedy continuation matched the HF golden 1/8 against the CPU's
8/8. The committed flat-weight gate (TestCohereResidentParityCUDA, 0.02-std) read 0.9997 on the same wrong
kernels. This gate reads the real weights.

BARS. Pre-registered 2026-10-01 BEFORE the first run with the pairwise kernels: every prompt position,
resident vs CPU (same quant), cosine >= 0.995 and relL2 <= 0.15 on the golden prompt and a 48-token prompt;
last-token cosine vs the HF golden within 0.01 of the CPU int4's; the 8-token greedy continuation equal to
the CPU int4's. THE FIRST AND THIRD DID NOT HOLD, and the reason is int4 noise, not the rotation (measured
in docs/measurements/cuda-pairwise-rope-2026-10-01.md against an HF f32 forward over EVERY position of both
prompts): the resident is exactly as far from HF f32 as the CPU int4 is (mean per-position cosine to HF 0.9851
resident / 0.9853 CPU on Aya, 0.9746 / 0.9761 on R7B; both have positions at 0.85-0.93 on the random-token
tail), and the two continuation flips are near-ties (the CPU's own top-2 gap at the flip is 0.17% / 0.07% of
its logit range, against the repo's 3% near-tie rule). A tight resident-vs-CPU bar on an 8B int4 model with
random-token tails therefore measures the quantizer. The bars asserted below are the PROPOSED, noise-referenced
ones; the pre-registered tight tier is still computed and logged ("tight tier") so the gap stays visible:
 1. resident vs CPU int4, golden prompt and 48-token prompt: MEAN per-position cosine >= 0.99 and MIN >= 0.90
    (decode), batched-prefill last-token cosine >= 0.98. The NeoX control reads min -0.04 / -0.075;
 2. last-token cosine vs the HF golden: resident >= CPU int4 - 0.01 (unchanged from the pre-registration);
 3. greedy continuation teacher-forced on the CPU's tokens: the resident's argmax equals the CPU's unless
    the CPU's own gap between the two tokens is under 3% of its logit range (decoder.NearTieHardFailPct, the
    rule every other gate in this tree uses); every flip is logged with its gap.

IT PROVES IT CAN FAIL: after the real measurement it rebinds the NeoX rope pipelines into the SAME resident
and re-measures the 48-token prompt (decode and batched prefill), which must read below bar 1.
```

## TestCohereResidentSmokeCUDA

Moved from `cuda/cohere_resident_smoke_test.go` (the comment above `TestCohereResidentSmokeCUDA`) on 2026-10-09.

```text
This smoke gate passed while every decode token's final norm was the wrong kind (audit-2026-09-10
C-04). The numeric gate is TestCohereResidentParityCUDA (cohere_resident_parity_test.go).

TestCohereResidentSmokeCUDA and TestCohere2ResidentSmokeCUDA are G5's last row
(docs/tasks/task-gpu-paths-2026-09.md) smoke gates on CUDA — the Metal twin
(metal/cohere_resident_smoke_test.go) explains why this is deliberately a smoke check
(admission + no NaN) rather than a resident-vs-CPU cosine floor: testdata/cohere-tiny and
testdata/cohere2-tiny are both "tiny-random" (scripts/pin_cohere_tiny.py), the same
seeded/synthetic class as every other G5 fixture, and FeatParallelBlock is a pure sequencing
change (reuse segA's r.aq/r.aSc as the MLP's input instead of re-normalizing the post-attention
residual) with no new numerical formula to isolate a floor against — quantizing the same shared
input norm once and reusing it is bit-identical to quantizing it twice from the same source.

The real correctness evidence sits one level down: layernorm_quant_test.go's TestLayerNormQuant
proves the NEW kernel (this backend had no mean-centered norm before) against an exact CPU
reference in isolation, and BuildResident's own validation (empty PostNorm/PostAttnNorm/
PostMLPNorm required to stay empty and unbuilt for a parallelBlock arch) is exercised simply by
this fixture reaching resident at all.
```

## TestTrimUnpinnedCtx

Moved from `cuda/ctx_trim_test.go` (the comment above `TestTrimUnpinnedCtx`) on 2026-10-09.

```text
TestTrimUnpinnedCtx pins the arithmetic behind the unpinned-context trim (the 2026-10-07 night gate's five default-context failures): the figures are the
two real misses, a 7B at 16000 positions (1.84 GB of KV, 1.80 GB free) and Gemma-3 at 11800 (3.29 GB, 3.35 GB free).
```

## TestDeclineAdvice

Moved from `cuda/decline_advice_test.go` (the comment above `TestDeclineAdvice`) on 2026-10-09.

```text
R20's gate: a resident decline's reason names a remedy, and a recovered executor panic's stack is not part of it. The two inputs
are the errors the real 26B q4_0 produced on the 8 GB RTX 2070 SUPER (2026-10-01, device OOM through runJob's panic boundary)
and the cold-user run's (an empty weight), not strings written to suit the function.
```

## TestDecodeAttn2048Probe

Moved from `cuda/decode_attn_probe_test.go` (the comment above `TestDecodeAttn2048Probe`) on 2026-10-09.

```text
TestDecodeAttn2048Probe drives the DECODE (M=1) attention kernel at ~2048 KV depth on the real
1.5B, so ncu can profile it: PrefillLast builds the 2048-token cache (using attn_batched), then a
run of Forward calls decode at pos 2048+ — those launch the M=1 `attention` kernel (glue.ptx) at
nKeys≈2048. Prefill used attn_batched, so `--kernel-name attention` targets ONLY decode attention.
Investigating the comparative deficit (goinfer decode 221→97 tok/s from 128→2048 ctx vs current
Ollama holding ~188) — a HYPOTHESIS to test at the hardware, not a diagnosis carried from prefill.

	GOINFER_HEAVY_TESTS=1 go test -tags cuda -c -o /tmp/decattn && \
	  sudo env ... ncu --kernel-name attention --launch-skip N /tmp/decattn -test.run TestDecodeAttn2048Probe
```

## TestDecodeDepthThroughput

Moved from `cuda/decode_depth_throughput_test.go` (the comment above `TestDecodeDepthThroughput`) on 2026-10-09.

```text
TestDecodeDepthThroughput measures real decode tok/s at a shallow (128) and a deep (2048) KV
depth on the 1.5B. §B2 recorded the deep number collapsing to ~97 tok/s vs ~221 shallow — the
long-context deficit ncu traced to the uncoalesced glue decode-attention K read. This is the
A/B instrument for the coalesced (attn_batched M=1) decode swap: run it on the coalesced build,
then `git stash` resident.go and run it on the glue build, to attribute the recovery.

	GOINFER_HEAVY_TESTS=1 go test -tags cuda -run TestDecodeDepthThroughput -v
```

## default_width_sweep_test.header

Moved from `cuda/default_width_sweep_test.go` (the comment at the top of the file) on 2026-10-09.

```text
Does defaultVerifyWidth = 8 hold, or is 7 better?

The ship-gate run (docs/measurements/adaptive-width-shipgates-2026-08-25.md) found static7
beating static8 by +7.1% on code and +5.1% on math -- a free win for every `--drafter` user
from a one-character change. It was NOT acted on, because that evidence was two prompts per
suite, one session, one target quant. This sweep is what clearing that bar looks like on the
box that has a viable pairing:

  - more prompts per suite (6 code / 6 math, not 2)
  - REPEATS, so within-condition spread is visible and the 5-7% claim can be read against it
  - BOTH target quants. This is the substantive addition, not padding: optimal width is set
    by the ratio between a plain decode step and a batched verify, and changing the target's
    quantization moves exactly that ratio. If 7 wins at int4 and 8 wins at int8, the default
    is quant-dependent and neither constant is right.

What it still is NOT: a second PAIRING. This box has one viable one (qwen3-4b dense + DFlash);
the other drafters are absent and their targets are MoE, where batched verify touches ~8x the
expert weight. The cross-pairing cell belongs on the Mac -- docs/prompts/mac-default-verify-width.md.

	GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks' ./cuda/ \
	  -run TestDefaultVerifyWidth -v -timeout 4h
```

## deltanet_snapshot_cuda_test.header

Moved from `cuda/deltanet_snapshot_cuda_test.go` (the comment at the top of the file) on 2026-10-09.

```text
MEASUREMENT ONLY — prices the DeltaNet state snapshot on the RESIDENT CUDA path, which is the
regime that actually decides the narrow-snapshot question. Builds nothing: specRollbackSafe is
untouched and no snapshot is wired into any decode path.

WHY THIS AND NOT THE CPU NUMBER. docs/spec/09-mtp-heads.md priced the copy on CPU and recorded
the direction: the numerator is a fixed 20.2 MiB while the denominator shrinks with every
quantization and backend improvement, so the fraction grows over time by construction. On CPU
f32 -> int8 already spanned most of the "cheap" band. This measures the endpoint that matters —
a resident decode step, where decode is fastest relative to a fixed copy.

AND THE SHAPE OF THE COPY CHANGES HERE, which is the part the CPU figure cannot speak to. On the
resident path both pieces of state are ALREADY on the device (cuda/resident.go:cudaResident — dnWin, the
causal-conv ring, and dnState, the recurrent matrix). A snapshot is therefore a device-side
copy, not a host memcpy, and the 13.9 GB/s host figure has no bearing on it in either direction.

	GOINFER_QWEN35_08B=~/models/qwen3.5-0.8b \
	  go test -tags 'cuda goinfer_testhooks' ./cuda/ -run TestDeltaNetSnapshotCUDA -v -timeout 30m
```

## TestDeltaNetSnapshotCUDA.paths

Moved from `cuda/deltanet_snapshot_cuda_test.go` (the comment above `TestDeltaNetSnapshotCUDA.paths`) on 2026-10-09.

```text
TWO SNAPSHOT PATHS, MEASURED IN THE SAME LOOP so the comparison is paired rather than
cross-session.

(a) PCIe ROUND TRIP — what was implementable before aikit/gpu v0.31.0. With only Upload and
    Download, state that is already on the device has to come back to the host and go out
    again. Kept as the control: it is the number the passthrough was justified against.

(b) DEVICE-TO-DEVICE via CopyDeviceBatch (v0.31.0). This is the in-situ measurement the
    synthetic-buffer probe stood in for — real state, real buffers, real decode between
    rounds. The probe reported ~446 us for snapshot+restore; a figure measured through the
    primitive on synthetic buffers is not an integration cost, which is why this exists.
```

## TestDeltaNetSnapshotCUDA.ratio

Moved from `cuda/deltanet_snapshot_cuda_test.go` (the comment above `TestDeltaNetSnapshotCUDA.ratio`) on 2026-10-09.

```text
THE RATIO IS FORMED PER ROUND AND CARRIES ITS OWN SPREAD — not median(cost)/median(decode).
A ratio of medians hides the round-to-round covariance, and here that matters: the decode
step alone ranges 2.2x within one run, so a single "100.3%" says nothing about whether the
figure is 100 +/- 5 or 100 +/- 60. Both terms wander; only the paired form shows whether
they wander together.
```

## TestDeltaNetKernels_cpuParity.conv

Moved from `cuda/deltanet_test.go` (the comment above `TestDeltaNetKernels_cpuParity.conv`) on 2026-10-09.

```text
WHY THIS STARTS AT delta_conv AND THE WEBGPU TEST DID NOT. WebGPU's causal conv IS the Mamba-2
conv, already gated there. CUDA has no SSM engine — no conv-ring, no persistent state, nothing
recurrent in any of its 24 kernels — so delta_conv is new code and has to be gated from its own
input. That is why the capture hook grew a `mixed` slot.
```

## TestDFlashDispatchAmortization

Moved from `cuda/dflash_dispatch_test.go` (the comment above `TestDFlashDispatchAmortization`) on 2026-10-09.

```text
TestDFlashDispatchAmortization settles the ONE assumption gate 3's draft term still rests on.

The 6.6 ms draft in docs/spec/08 is `5.33 × per-layer(M=16)`, where per-layer comes from
dividing the 36-layer target's batched verify by 36. That silently assumes **a 5-layer model
costs 5/36ths of a 36-layer one** — i.e. that per-layer cost is independent of how many layers
are in the stack. It need not be: a 36-layer forward has 36 dispatches to hide launch latency
behind, and five have far less. If per-layer cost RISES as the stack shortens, the drafter is
more expensive than 6.6 ms and every projected speedup drops.

This is not hypothetical in this repo. It is the mechanism that landed the CUDA-graphs
projection (1.4–1.7×) at a measured 1.01× — CPU dispatch overlaps GPU compute differently at
different scales — and the mechanism behind Lever 2's "the draft was the wall, not the verify".

METHOD: run the resident forward with the layer loop truncated (`r.nLayers`), at both M=1
(`launchToken`) and M=16 (`PrefillLast`, the regime the block draft actually runs in), and
compare per-layer cost across stack depths. The outputs are numerically meaningless — a
truncated stack is not a model — but the TIMING is exactly the quantity in question, and the
weights for every layer are already resident so no reload is involved.
```

## TestDFlashCaptureSeamCost

Moved from `cuda/dflash_dispatch_test.go` (the comment above `TestDFlashCaptureSeamCost`) on 2026-10-09.

```text
TestDFlashCaptureSeamCost measures a composition cost the gate-3 arithmetic omits entirely.

The projection composes draft + verify, where verify is the measured `W + C*k` curve. But that
curve was measured with the HIDDEN-STATE SEAM OFF. In the real loop the drafter needs the
target's residual at 5 tap layers for every token the target commits, and `capVec` implements
that as a full `r.stream.Sync()` followed by a device->host `Download` — **per tap**. Five taps
is five pipeline stalls per token, mid-forward.

That is not a hypothetical cost: it is the difference between the verify the projection prices
and the verify the loop would actually run. If it is large, every speedup figure is optimistic
by that margin, and the fix (capture into a device buffer, download once, or overlap on a
second stream) becomes a prerequisite rather than an optimization.
```

## TestDFlashRoundComposition

Moved from `cuda/dflash_dispatch_test.go` (the comment above `TestDFlashRoundComposition`) on 2026-10-09.

```text
TestDFlashRoundComposition measures the LOOP, not its parts.

Every term in gate 3's projection is now measured — acceptance, the verify curve, decode, the
draft (8.82 ms), the capture seam (0.465 ms/token). What is still arithmetic is the
COMPOSITION: that a round costs draft + verify + seam and nothing else. Real loops have costs
between their operations — host round-trips, stream syncs at the boundaries, the argmax and
accept comparison, cache rollback — that a sum of independently-timed parts cannot show.

The real drafter kernel does not exist yet, so this substitutes the TARGET's stack truncated to
5 layers as a timing stand-in. Its OUTPUT is meaningless — a truncated stack is not the
drafter — but its COST is the right shape: 5 layers at M=16 over the same geometry, which is
exactly what TestDFlashDispatchAmortization measured at 8.273 ms. What this adds is the
sequencing: draft, then verify at M=k with capture live, then the host-side accept, per round.

Reading: if measured/round ≈ predicted/round, the arithmetic composes and the projection's only
remaining risk is the drafter kernel's own efficiency. If it exceeds the prediction, there is
per-round overhead the projection never priced.
```

## TestDFlashCompositionResidual

Moved from `cuda/dflash_dispatch_test.go` (the comment above `TestDFlashCompositionResidual`) on 2026-10-09.

```text
TestDFlashCompositionResidual decomposes the +4.37 ms/round that TestDFlashRoundComposition
found unaccounted, because how much of it is REAL decides two things: whether code clears the
1.3x bar, and whether the optimum verify width shifts.

A fixed per-round cost is amortized better by a WIDER block — so if the residual is genuinely
fixed, the optimum moves away from the k=7 the acceptance sweep found, and increment 4 should
be built for a different width. That is why this belongs before the kernel work.
```

## TestDFlashVerifyHeadCost

Moved from `cuda/dflash_dispatch_test.go` (the comment above `TestDFlashVerifyHeadCost`) on 2026-10-09.

```text
TestDFlashVerifyHeadCost isolates the 3.09 ms the residual decomposition could not explain.

HYPOTHESIS: the verify curve `T(M) = W + C*M` was measured with `PrefillLast`, which applies
the LM head to the LAST row only. The spec-decode loop needs `PrefillLastN` — logits at ALL M
positions, because every drafted token must be compared against the target's own argmax there.
That is M head applications, not one, and the head is 8% of an M=1 decode. The curve therefore
prices a verify the loop cannot use.
```

## TestDFlashDraftCostProbe

Moved from `cuda/dflash_draftcost_test.go` (the comment above `TestDFlashDraftCostProbe`) on 2026-10-09.

```text
TestDFlashDraftCostProbe measures the one term P10's gate-3 projection could not settle:
what a RESIDENT DFlash trunk would cost per block (docs/spec/08).

It does it WITHOUT building the trunk, on a structural fact worth stating plainly: the
DFlash drafter is exactly FIVE LAYERS OF THE TARGET'S OWN LAYER SHAPE plus `fc`. Same
hidden 2560, same 32/8 GQA at head_dim 128, same 9728 SwiGLU — 5 × 100.9 M + 32.8 M =
537.4 M, which matches the checkpoint's tensor count to the digit. So the resident runner
ALREADY executes the drafter's per-layer work 36 times per token; the drafter is 5/36ths
of it, and the cost can be read off the target instead of modelled.

The probe isolates the LM head (launchToken's `head` flag), because the head is 389 M of
the target's 4.02 B and the drafter has none — it borrows the target's, and that cost is
already inside the verify. Attributing head time to the drafter would overstate it.

WHAT THIS IS NOT: a claim that the trunk will hit this number. It excludes the drafter's
non-causal attention over [ctx‖block], which is the one part with no counterpart in the
target's per-token path, and it assumes the same kernels. It is a floor with a named
omission, not a prediction.
```

## dflashLoop.mask

Moved from `cuda/drafter_loop_test.go` (the comment above `dflashLoop.mask`) on 2026-10-09.

```text
THE MASK TOKEN IS TRAINED, not a placeholder. DFlash learned to see this specific
embedding at unfilled block positions; feeding any other id puts the drafter
off-distribution and it drafts badly while everything still runs. Measured cost of
getting this wrong: 1.77 tok/round against the CPU sweep's 4.97 at the same width.
```

## TestDFlashLoop_gate3

Moved from `cuda/drafter_loop_test.go` (the comment above `TestDFlashLoop_gate3`) on 2026-10-09.

```text
TestDFlashLoop_gate3 is GATE 3: the end-to-end wall-clock the whole projection has been
standing in for.

docs/spec/08 projects code 1.52x / math 1.96x at verify widths 7/8, composed from separately
measured terms — draft 8.82 ms, the batched-head verify curve, a 1.09 ms batched seam,
acceptance from a 7-width CPU sweep. Every term is measured; the COMPOSITION was arithmetic.
This runs the real loop against plain greedy on the same resident and divides.

REAL PROMPTS, chat-templated, because acceptance is a property of real text: the lossless gate
above reads 1.56 tok/round on random ids, which says nothing about anything. The suite is the
same one the CPU acceptance sweep used, so the numbers are comparable.

	GOINFER_HEAVY_TESTS=1 GOINFER_CUDA_MODEL=$HOME/models/qwen3-4b \
	  go test -tags 'cuda goinfer_testhooks' -run TestDFlashLoop_gate3 -v -timeout 2h
```

## TestDFlashLoop_gate3.prompts

Moved from `cuda/drafter_loop_test.go` (the comment above `TestDFlashLoop_gate3.prompts`) on 2026-10-09.

```text
The same prompts and the same non-thinking template the CPU acceptance sweep used, so
tok/round here is comparable to the 4.97 that sweep measured at width 7.
```

## TestDFlashLoop_gate3.chat

Moved from `cuda/drafter_loop_test.go` (the comment above `TestDFlashLoop_gate3.chat`) on 2026-10-09.

```text
chat is the class the projection says LOSES (0.78x), and gate 4's router exists for
it. Measured here rather than projected, because a router should be designed against
the real number: if chat is a mild loss the router is an optimization, and if it is a
severe one the router is a correctness-of-economics requirement.
```

## TestDFlashLoop_gate3.baseline

Moved from `cuda/drafter_loop_test.go` (the comment above `TestDFlashLoop_gate3.baseline`) on 2026-10-09.

```text
THE BASELINE IS Model.Generate — what a server actually runs. An earlier version
looped PrefillLastNArgmax(M=1), which downloads the full 608 KB logit row per
token where Generate uses the GPU argmax fast-path (4-byte readback). That made
every ratio here ~10% optimistic.
```

## withDrafterReserve

Moved from `cuda/drafter_reserve_test.go` (the comment above `withDrafterReserve`) on 2026-10-09.

```text
withDrafterReserve is o for a target a block drafter attaches to AFTER the load, priced the way `serve --drafter` prices it (internal/serveapp/main.go loads the
drafter first and sets both fields): the plan then leaves the drafter's int8 weights and its per-position K/V out of the context it chooses. A test that loads the
target plainly and bolts NewBlockSpec on afterwards gets a context sized to every byte beyond the 384 MiB margin and the drafter's allocation runs out of device
memory (the 2026-10-07 night gate's three drafter failures, root-caused 2026-10-08: the loads logged "less 384 MB reserved", the margin alone). Production never
does that, so the tests were the stale half.
```

## TestBatchedCapture_matchesPerToken

Moved from `cuda/drafter_test.go` (the comment above `TestBatchedCapture_matchesPerToken`) on 2026-10-09.

```text
TestBatchedCapture_matchesPerToken gates the batched hidden-state seam against the per-token
one it replaces.

The drafter reads the target's residual at five tap layers for every token the verify commits.
The existing seam does that with a sync and a download per tap PER TOKEN (0.465 ms/token
measured); the batched one does one download per tap for the whole block. That is only a valid
substitution if it records the SAME tensors — and a batched capture taken at the wrong point
in the layer loop, or reading the residual before the MLP's residual add, would still be the
right shape and the right magnitude.

So: run M tokens sequentially with the per-token seam, run the same M as one batched call with
the batched seam, and require BIT EQUALITY. Not cosine — both paths are the same kernels on
the same weights, and the batched layer stack is already bit-identical to sequential
(TestPrefillLast_e2e). Anything less than == here would mean the two seams disagree about
which tensor they are recording.
```

## TestBatchedCapture_matchesPerToken.cost

Moved from `cuda/drafter_test.go` (the comment above `TestBatchedCapture_matchesPerToken.cost`) on 2026-10-09.

```text
What it costs. The per-token seam measured 0.465 ms/token (5 taps), so ~2.3 ms per round
at four accepted -- a term the gate-3 projection carries. This is the batched replacement.
```

## drafter_vs_off_test.header

Moved from `cuda/drafter_vs_off_test.go` (the comment at the top of the file) on 2026-10-09.

```text
Per-suite drafter-vs-OFF comparison, including the mixed-content suite.

Born as the adaptive-width ship-gates; the adaptive arm and its gates came out when Phase 2's
premise died and took the controller with it. What remains is the part that earned its keep:
every static width scored against running NO drafter at all, per traffic class. Runs the REAL BlockSpec path on a CUDA-resident target, not a reimplemented loop:
cuda/drafter_loop_test.go's dflashLoop is a standalone copy of the round loop and would
measure a controller that is not in it.

	GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks' ./cuda/ \
	  -run TestDrafterVsOff -v -timeout 4h
```

## adaptiveSuites

Moved from `cuda/drafter_vs_off_test.go` (the comment above `adaptiveSuites`) on 2026-10-09.

```text
adaptiveSuites: the three existing traffic classes plus MIXED, which is new and is the case
the whole idea exists for.

WHY MIXED HAD TO BE BUILT. code/math/chat each sit in ONE regime for a whole generation, so
a static width can be optimal for the entire run and adaptivity has nothing to win. The
claimed advantage lives at a prose->structured BOUNDARY, where no single static value is
right for both halves. Every prompt here forces the transition mid-generation: explain in
prose first, THEN emit something structured.
```

## TestDrafterVsOff.off

Moved from `cuda/drafter_vs_off_test.go` (the comment above `TestDrafterVsOff.off`) on 2026-10-09.

```text
THE COMPARISON SET INCLUDES OFF. Adaptive beating every static width proves nothing on a
suite where running no drafter at all is faster -- which is precisely the guard's answer
for chat (it disables, measuring 0.92x). Without this column the chat cell cannot answer
the only question that matters there.
```

## TestDrafterVsOff.trace

Moved from `cuda/drafter_vs_off_test.go` (the comment above `TestDrafterVsOff.trace`) on 2026-10-09.

```text
The mixed suite's per-round trace is kept: it is how Finding 2 was found
and how the next drafter change gets checked against it. Recorded at the
widest arm, since that is where a transition shows most.
```

## TestDrafterVsOff.gate

Moved from `cuda/drafter_vs_off_test.go` (the comment above `TestDrafterVsOff.gate`) on 2026-10-09.

```text
THE SURVIVING GATE: does the drafter beat NO drafter at all on this suite?
The adaptive arm and its two gates were removed with the controller (Phase 2's
premise died, so nothing was left to reuse it) — but `off` as a competitor is
the part that earned its keep, and it stays.
```

## drainsDevice

Moved from `cuda/drain_marker_test.go` (the comment above `drainsDevice`) on 2026-10-09.

```text
drainsDevice is the MARKER for the tier partition, and it is a shared helper rather than a naming
convention or a hand-kept -run list for one reason: a list of names is a constant that restates a
property, and it drifts silently. The census-denominator work made that shape visible four times
over — a check reports its numerator and stays green while its universe shrinks. A `-run
'TestAllocFloor|TestA10ReportingGap|…'` in the GPU gate is exactly that constant. This is the
property itself, in the code that has it.

WHAT IT MARKS. A test that deliberately drives the device to REFUSAL — allocates until even a
small request fails — or holds it near the floor while a live context keeps working. That is A13's
only reproducible poisoning stimulus (5/5), and it is a property of the test, not of its name.

IT DOES NOT ONLY LABEL, IT ENFORCES. Deriving a group and trusting everyone to run it is the same
advisory-comment failure as before, so the marker is also the gate: without GOINFER_DRAIN_GROUP a
marked test SKIPS. A drainer can therefore never execute inside the main tier even if the shell's
derivation misses it, and the skip line it prints is what the reconciliation counts.

WHAT THE RECONCILIATION DOES AND DOES NOT CATCH — corrected 2026-08-13, after the first gate run
caught a case this comment had claimed was impossible. It said "the partition cannot silently drop
a test into neither half". That is true only of MARKED tests: a derivation miss shows up as a
main-tier skip with no matching drain-tier run, and the gate fails on the mismatch. It says
NOTHING about a drainer that never calls this helper at all — that test is invisible to both the
derivation and the reconciliation, and runs in the main tier as if it were harmless.

That is not hypothetical. The very first gate run after this marker landed went red on
TestMoERouteDemandThreshold, which balloons the device to as little as 64 MiB free — an unmarked
drainer, exactly the blind spot. It is marked now. The blind spot is not closed by marking it, so:
COVERAGE HERE IS BY INSPECTION, and the honest statement is that this helper enforces the
partition for tests someone remembered to mark, and nothing more.

The child processes spawned by TestA10FloorIsPerProcessOrPerDevice inherit the environment
(cmd.Env = append(os.Environ(), …)), so they inherit the flag with it.

TO ADD A DRAINER: call this first thing in the test. Nothing else. `gate gpu` derives the group
by scanning for calls to this function, so there is no second place to update and no list to
forget.
```

## TestE2EDecodeThroughput_synthetic.argmax

Moved from `cuda/e2e_decode_test.go` (the comment above `TestE2EDecodeThroughput_synthetic.argmax`) on 2026-10-09.

```text
argmax_reduce comes from argmaxPTX, NOT gluePTX. C-14 (c6600fc) split it into its own
module so the index tie-break fix could land without regenerating the audited glue.ptx;
binding it off glmod here exercised the PRE-C-14 kernel, i.e. not the one production ships.
```

## validateGlue.errors

Moved from `cuda/e2e_decode_test.go` (the comment above `validateGlue.errors`) on 2026-10-09.

```text
ERRORS ARE CHECKED, NOT DROPPED. Every call here used to be `_ =`, so a failure left `dc`
UNWRITTEN, `got` all zeros, and the assertion reported "attention cosine 0.000000" with no
error text at all — a resource failure wearing a numerics bug's clothes. That is the exact
history in the GPU gate's own header, now cmd/gate/gpu.go ("the tests DROPPED those errors, and the resulting
zero-filled buffers surfaced as cosine 0.000000"), and it recurred here: four silent zero
cosines in the tier, with zero CUDA errors anywhere in the log.

A zero cosine now means the arithmetic is wrong. Anything else names the call that failed.
```

## validateGlue.handle

Moved from `cuda/e2e_decode_test.go` (the comment above `validateGlue.handle`) on 2026-10-09.

```text
A13 launch diff (GOINFER_A13_LAUNCH=1). Prints everything the launch depends on, so a
poisoned run and a clean one can be diffed field by field: if an argument or a device
pointer differs, something upstream is holding state from the drain and the culprit is
named; if every field is identical and only the result differs, the state is inside the
driver or the context rather than in this call.
A13 step 1: interrogate the CACHED function handle. Hypothesis under test — the drain
causes driver-side module eviction under pressure, and the cached CUfunction outlives what
it names, so a launch through a stale handle returns success and does nothing.
  errors or implausible values in the poisoned run only -> handle is stale, confirmed
  identical valid attributes in both                    -> handle is live, look elsewhere
```

## validateGlue.reload

Moved from `cuda/e2e_decode_test.go` (the comment above `validateGlue.reload`) on 2026-10-09.

```text
A13 step 2, TEMPORARY PROBE (GOINFER_A13_RELOAD=1) — not a fix. Re-load the module and
re-resolve the function immediately before the launch. If the result becomes correct,
eviction is confirmed from the other direction; if it is still zeros, the state lives
BELOW the module layer, in the context. A reload before every launch would mask the
mechanism rather than address it, which is why this is env-gated and disposable.
```

## TestEmbedInt4_residentDecisionDoesNotChange

Moved from `cuda/embed_int4_resident_test.go` (the comment above `TestEmbedInt4_residentDecisionDoesNotChange`) on 2026-10-09.

```text
Since 2026-10-09 Options.EmbedInt4 reaches every family loader (decoder/embed_int4_loaders_test.go), so a family that used to get an int8 head by default
now gets an int4 one. The resident must take that table without changing its decision: where a fixture builds resident with the int8 head it builds resident
with the int4 head (the same decode path, the head reading int4), and where it declines it declines for the same reason either way. A resident that
declined an int4 head would silently move a whole family to the CPU the day the default reached it.
```

## executor_panic_test.header

Moved from `cuda/executor_panic_test.go` (the comment at the top of the test group) on 2026-10-09.

```text
Audit C-24 / C-25 — the executor goroutine must survive a panicking job, and the expert-cache
sizing must not divide by zero on a dense-prefix MoE.

WHY THESE EXIST. Both findings are the same shape: code that the design says should DECLINE
instead kills the process, and both do it on the pinned executor goroutine where no caller's
`defer recover()` can reach. C-24: `gpu.NewBufferLenOf` panics on OOM per its own contract, and
prefillCore allocates hundreds of MB at M=3000 — so a long prompt on a nearly-full card killed
serve at the exact seam whose job is to fall back to the sequential path. Two comments claimed
this was handled; `BuildResident`'s recover runs on the CALLING goroutine and cannot catch it.
C-25 is a reachable trigger for the same crash: GLM/DeepSeek/Kimi put dense layers first, so
layer 0 has no expert strides and `budget / len(moeLayers) / perLayer` divides by zero.

NO DEVICE NEEDED: runJob is a pure function, and slotBytesPerLayer reads struct state only.
```

## TestFlashDecodeSplit_default

Moved from `cuda/flash_decode_default_test.go` (the comment above `TestFlashDecodeSplit_default`) on 2026-10-09.

```text
TestFlashDecodeSplit_default pins how GOINFER_CUDA_FLASH_DECODE resolves now that the lane is default ON (R6, 2026-09-23):
unset or empty is the registered S, an explicit 0/off/false is the exact path, a positive integer picks S, and anything that
is not a positive integer is OFF — a typo must never enable a non-exact attention path at some other S.
```

## flashOracleRealKV

Moved from `cuda/flash_decode_gate_test.go` (the comment above `flashOracleRealKV`) on 2026-10-09.

```text
flashOracleRealKV: strict=true is the registered precondition (lane no worse than exact vs f64, per layer). strict=false
is a sanity bound for geometries outside the gate: on gemma3-1b's 512-key windows the exact fold is already accurate to
~1e-8, so the lane can sit a few ulp above it (measured: 2 of 26 layers, by <= 2.5e-7 relative); there the check is that
the lane's error stays under 1e-5 of max|ref| at every layer, and the comparison is logged.
```

## TestFlashDecodeKernelLadder.topS

Moved from `cuda/flash_decode_ladder_test.go` (the comment above `TestFlashDecodeKernelLadder.topS`) on 2026-10-09.

```text
The ladder's top S. The suite's TestMain pins GOINFER_CUDA_FLASH_DECODE=0 (vramtrace_test.go) so that other tests
never exercise the lane by accident — which left this test, the one that exists to time the lane, unable to run
without a hand-set variable: it failed with faSplit=0 on every clean invocation. A caller's positive S is kept;
"0" (the pin, or an explicit off, meaningless for a lane ladder) becomes the registered default S.
```

## foreign_context_test.header

Moved from `cuda/foreign_context_test.go` (the comment at the top of the file) on 2026-10-09.

```text
Foreign CUDA contexts — asking the real question.

Several gates in this package assert against a PINNED device allocation
floor, and that floor is not a property of the code: it moves when another
process holds a CUDA context on the device. Measured 2026-09-01 on nobara,
with KDE's compositor (`kwin_wayland`) holding one:

	floor with a foreign context   18,546,688 B
	floor with none (2026-08-26)    1,769,472 B
	                    difference 16,777,216 B  = exactly 16 MiB

The demand identity itself was NOT disturbed — it closed to the byte in both
cases (18,546,688 + 138,412,032 = 156,958,720 = the measured demand) — so the
kernel's requirement is unchanged and only the environment-dependent
component moved.

WHY THIS HELPER EXISTS AT ALL. moe_route_demand_test.go discriminated the two
regimes with `warm := freeBefore < pinnedDeviceFloor`, which can only be true
when the floor exceeds the residual. That stopped being true on 2026-08-21,
and the file has carried a KNOWN LATENT DEFECT note ever since saying the warm
branch was unreachable and "a warm run would go red claiming a broken
identity". That is exactly what happened. The heuristic was inferring device
state from a number; this asks the device.

nvidia-smi rather than NVML bindings: this is test-only, runs once, and adding
a library dependency to answer a question a shipped tool already answers would
be a worse trade. If nvidia-smi is absent the caller is told "unknown" and must
decide — never silently "none", which would restore the very failure mode this
replaces.
```

## TestForwardMRoPE_constantShiftInvariance

Moved from `cuda/forwardmrope_parity_test.go` (the comment above `TestForwardMRoPE_constantShiftInvariance`) on 2026-10-09.

```text
TestForwardMRoPE_constantShiftInvariance is ForwardMRoPE's (decoder.ResidentMRoPE) FIRST real
correctness test — before this, the new interface (and the rope_kv kernel's ropePos parameter
it depends on) had never been exercised outside its own compile.
```

## gemmMMAMaxRelDelta

Moved from `cuda/gemm_mma_test.go` (the comment above `gemmMMAMaxRelDelta`) on 2026-10-09.

```text
PRE-REGISTERED BOUND for TestGemmMMA_vsExact, derived before the kernel was first run.

gemm_w4a8_mma and gemv_w4a8_rn compute the SAME int8 products against the SAME per-group f16
scales. Neither the products nor the scales differ; the ONLY difference is the association of the
cross-group float sum, which is exactly what §4 L3 predicts and what the sibling kernel's header
names as the reason bit-identity forecloses tensor cores.

The two differ in how many float roundings they perform. gemv_w4a8_rn folds ONE float FMA PER
WORD — K/8 terms — because dp4a can only accumulate 8 elements exactly. gemm_w4a8_mma folds one
per GROUP — K/32 terms — because the two m8n8k16 MMAs accumulate all 32 elements of a group in
int32 with NO rounding at all. So the new kernel performs 4x FEWER float roundings and is, if
anything, the more accurate of the two; the test does not assume that, it just bounds the gap.

THE NUMBER: f32 eps = 2^-24 = 6.0e-8. Summing G terms in two different orders differs by roughly
eps * sum|partial sums|, which for random-signed terms is ~ eps * sqrt(G) * |result|. The largest
production K here is 18944 (G = 2368 words for the reference), giving ~ 6.0e-8 * 49 = 2.9e-6.
The bar is set at 1e-5 of the OUTPUT SCALE with ~3x margin.

RELATIVE TO max|dst| ACROSS THE OUTPUT, not to each element. That is a deliberate correction
learned in this same task: docs/measurements/prefill-l2l3-phase1-2026-09-05.md §2.1 records an
L2 bar scaled per-element by a quantity that can cancel to near zero, which made it unmeetable
for reasons that had nothing to do with the kernel. A GEMM output element can likewise land near
zero by cancellation; the output scale cannot.
```

## TestGemma3ImgPrefillResidentReal_gate.reuse

Moved from `cuda/gemma3_img_prefill_resident_real_test.go` (the comment above `TestGemma3ImgPrefillResidentReal_gate.reuse`) on 2026-10-09.

```text
Reuses mc rather than loading a second resident instance: two resident int4
gemma-3-4b-it instances (weights + a 4096-position KV cache each) do not fit together on
this 8GB card (measured directly building this session's P9(a) timing driver). Safe to
reuse — ResidentImagePrefillForTest above left mc.resIDs untouched (nil; that bookkeeping
is P9(a)'s, not this primitive's), and GenerateVL's ordinary path is ALWAYS a full,
unconditional prefill overwrite regardless of whatever the resident cache held before.
```

## TestGemma3ResidentReal_gate

Moved from `cuda/gemma3_resident_real_test.go` (the comment above `TestGemma3ResidentReal_gate`) on 2026-10-09.

```text
TestGemma3ResidentReal_gate is gap 0's real-checkpoint gate for GenerateVL (Gemma 3) — the
Gemma-3 twin of qwen25vl_resident_real_test.go's TestQwen25VLResidentReal_gate, closing the
gap that file's own doc comment and docs/multimodal.md's gap-0 entry both name: "no real-image
end-to-end gate yet" for GenerateVL specifically. Simpler than the Qwen twin — Gemma 3 has no
m-RoPE, so plain Forward (not ForwardMRoPE) is the whole story once the CPU prefill's KV is
uploaded.

Same methodology as the Qwen gate, for the same reason (see that file's doc comment for the
full rationale): one forced-trajectory decode step, matched precision (int4 both arms), cosine
on raw logits rather than sampled greedy-token-stream identity — comparing SAMPLED tokens
across a multi-step free-running rollout would conflate this design's own correctness with
ordinary f32-vs-int4 quantization noise compounding through greedy decode, which is an
orthogonal, pre-existing property this repo already understands (measured directly on this
same box's Qwen2.5-VL checkpoint via a throwaway probe before the Qwen gate was rebuilt this
way).
```

## TestGemma3VisionResidentReal_gate

Moved from `cuda/gemma3_vision_resident_real_test.go` (the comment above `TestGemma3VisionResidentReal_gate`) on 2026-10-09.

```text
TestGemma3VisionResidentReal_gate is P6's real-checkpoint gate (docs/multimodal.md's "P6's
other half"): the resident CUDA SigLIP tower vs its own CPU path, on the real gemma-3-4b-it
vision tower, matched precision (both int8 — vision.LoadEncoder(dir, quant=true) puts BOTH the
CPU and resident paths on the same W8A8 weights; this is not an int8-vs-f32 comparison).

THE THRESHOLD IS NOT THE USUAL ≥0.99. Measured directly on this checkpoint (real image pixels,
4096 patches, 27 layers): patch-embed alone already matches the CPU path at cosine 0.999999,
and a matched-precision CPU probe reconstructed from the SAME int8 weight data (linalg.WrapInt8
+ MatmulBTInto) reproduces one layer's raw FC2 GEMV output at cosine 1.000000 — i.e. the KERNELS
are exact. What is NOT bit-identical is the ACCUMULATION ORDER between the CPU's and CUDA's
per-layer reductions (LayerNorm's mean/variance sums, the attention softmax denominator, the
int8 GEMV's own accumulation) — a real but ordinary "not bit-identical, cosine-gated" property
already true of this repo's other batched kernels (rmsnorm_quant_batched's own header makes
the same point). Compounded over 27 layers — SigLIP so400m is unusually deep for a vision tower
this project has resident-ported — that ordinary per-layer rounding difference accumulates to
cosine ~0.91-0.96 end-to-end depending on the input pixel pattern (measured directly, both
arms — not a single lucky run), confirmed via the matched-precision probe above to be genuine
accumulated rounding, NOT a wiring bug. The floor below (0.80) sits with real margin under the
lower end of that measured range — loose enough that ordinary input-dependent variance in the
accumulated rounding doesn't flake the gate, tight enough that an actual wiring regression
(which produced cosine ~0.09-0.18 before the posEmb bug in this file's own history was found
and fixed) still fails it by a wide margin.
```

## TestGemma4_26B_1bBound

Moved from `cuda/gemma4_26b_bound_test.go` (the comment above `TestGemma4_26B_1bBound`) on 2026-10-09.

```text
TestGemma4_26B_1bBound is Lever-1b Step 0: bound the win BEFORE building the aikit event primitive
+ the gemma4MoeMLP reorder. 1b hides the per-layer router-wait drain behind the dense branch, so
the recoverable time is capped by the DENSE-BRANCH GPU time. Measure it (loop the dispatch
sequence N times, one sync, divide — no event support needed) and compare to the ~12 ms/token
drain the skip-readback probe isolated (59→36, minus ~11 ms DMA). If the dense branch is thin, 1b
is not worth the primitive+reorder and the fallback (cross-layer pipelining, a larger design) or
the dispatch-reduction lever should be chosen deliberately.
```

## TestGemma4_26B_1bBound.drain

Moved from `cuda/gemma4_26b_bound_test.go` (the comment above `TestGemma4_26B_1bBound.drain`) on 2026-10-09.

```text
The drain the probe isolated: 59→36 with stale idx removed BOTH the r.stream.Sync() drain AND
~11 ms of miss-DMA, so the drain alone ≈ 12 ms/token. 1b can hide at most densePerTok of it.
```

## TestGemma4_26B_cache_B.prof

Moved from `cuda/gemma4_26b_cache_test.go` (the comment above `TestGemma4_26B_cache_B.prof`) on 2026-10-09.

```text
PRICE THE ROUTING ROUND TRIP. cacheProf has existed and been read by nothing; this wires it
up. It decomposes loadRoutedExperts into the three things it actually does per MoE layer per
token — the pipeline drain, the host-side slot bookkeeping, and the expert DMAs — which is the
number that decides whether speculative prefetch is worth its complexity (G30, spec/10's
standing verdict). Zero unless GOINFER_MOE_CACHE_PROF is set.
```

## TestGemma4_26B_cache_B.upload

Moved from `cuda/gemma4_26b_cache_test.go` (the comment above `TestGemma4_26B_cache_B.upload`) on 2026-10-09.

```text
PHASE 0 ANSWERED IT, AND THE FIX IS IN. The question was whether the expert DMA is
bandwidth-bound or per-call-overhead bound: each miss used to issue FOUR blocking
null-stream uploads, and the tiny scale copy costing comparable per-call time to the big
weight copy would mean fixed per-call cost dominated. It did, so the copies are now
QUEUED and issued per layer by one gpu.UploadBatch. Copy count is therefore unchanged
and sync count is what moved, which is why the two are reported separately below.
```

## TestGemma4DenseScaled_residentParity

Moved from `cuda/gemma4_dense_scaled_test.go` (the comment above `TestGemma4DenseScaled_residentParity`) on 2026-10-09.

```text
TestGemma4DenseScaled_residentParity exercises the CUDA resident bridge on a SCALED dense Gemma 4
(hidden 1024, 12 layers, 5:1 sliding/full, REAL head dims 256 local / 512 global, K=V globals) —
closing the 256-local geometry gap the tiny Split-A fixture (hd=16) left. Gated exactly as Split A
/ the MoE 2c gate: pos-0 kernel correctness + the calibrated per-position curve (568f292),
CUDA-int4-vs-CPU-int4 measured against the fixture's own CPU-int4-vs-f32 quantization curve so a
chaotic int4 floor doesn't masquerade as a kernel bug.

FINDING baked into the gate: random weights over 12 layers are LESS int4-conditioned than the tiny
2-layer fixture (floor ~0.46 vs 0.79) — realistic geometry ≠ realistic conditioning (that needs
trained weights). So the multi-position int4-vs-int4 drift is REPORTED, and the gate is the run
mean vs that curve, not an absolute floor. No coherence gate: random weights → degenerate greedy.
```

## TestGemma4DenseScaled_residentParity.pos0

Moved from `cuda/gemma4_dense_scaled_test.go` (the comment above `TestGemma4DenseScaled_residentParity.pos0`) on 2026-10-09.

```text
pos-0 kernel correctness (no KV accumulation): the 256-local + 512-global geometry must compose
correctly. cuda-vs-cpu-int4 differs only in W4A8 activation rounding, so pos 0 is close even when
the int4-vs-f32 floor is chaotic.
S1.0 amendment 2026-10-07 (docs/tasks/task-multimodal-support-2026-10.md): the bar sits between the before-v_norm-fix and after readings; the fix is the mechanism. Never loosened. pos0 was 0.996182 before the fix (v_norm missing on the sliding layers), 1.000000 after; the run mean 0.910545 -> 0.999634.
```

## gemma4_emodel_image_test.header

Moved from `cuda/gemma4_emodel_image_test.go` (the comment at the top of the file) on 2026-10-09.

```text
S9 on CUDA, part B: the Gemma 4 E-model image turn prefilled on the resident in one batched pass, from rows the decoder builds. The plan, the gates and the
bars are registered in docs/tasks/task-multimodal-support-2026-10.md ("S9 on CUDA, part B") before the code was written.
```

## gemma4_emodel_prefill_test.header

Moved from `cuda/gemma4_emodel_prefill_test.go` (the comment at the top of the file) on 2026-10-09.

```text
S9 on CUDA, part A: the batched text-prompt prefill for Gemma 4 E-models. The plan, the gates and the bars are registered in
docs/tasks/task-multimodal-support-2026-10.md ("S9 on CUDA, part A") before any of this code was written.
```

## TestGemma4EModelPrefill_exactKernelsAboveTheFloor

Moved from `cuda/gemma4_emodel_prefill_test.go` (the comment above `TestGemma4EModelPrefill_exactKernelsAboveTheFloor`) on 2026-10-09.

```text
TestGemma4EModelPrefill_exactKernelsAboveTheFloor is the guard for what the first served G3p read found (docs/tasks/task-multimodal-support-2026-10.md): the
fast prefill levers engage above a 512-row prompt floor and are not bit-identical to decode, and they have no fidelity evidence on an E-model. With the floor
moved to 0 ("fast at any length", GOINFER_CUDA_FAST_PREFILL_FLOOR) an E-model's PrefillLast must therefore still be bit-identical to the sequential path, and
must launch none of the fast kernels. The tiny gates above never saw this: their prompts are far under the floor, which is how the real 2,170-token prompt
was the first to differ.
```

## g3Model.sameOptions

Moved from `cuda/gemma4_emodel_real_test.go` (the comment above `g3Model.sameOptions`) on 2026-10-09.

```text
The CPU loads the SAME file with the SAME options, so both sides hold the same quantization of every table. (G3c run 1, 2026-10-07, let the CPU
read the CUDA e4h sidecar while the CUDA side loaded the GGUF with Options.EmbedInt4 unset: an int4 head against an int8 pin, the confound
Metal's G3 run 1 fell into. It was caught in the log before anything was recorded and is superseded; see the task doc.)
```

## TestGemma4EModel_realE2BNonInferiority

Moved from `cuda/gemma4_emodel_real_test.go` (the comment above `TestGemma4EModel_realE2BNonInferiority`) on 2026-10-09.

```text
TestGemma4EModel_realE2BNonInferiority is G3c (docs/tasks/task-multimodal-support-2026-10.md, "S1 on CUDA", registered 2026-10-07 before any CUDA
run, Metal's re-registered G3 rule unchanged): in one process, g3Run on Qwen2.5-Coder-1.5B on CUDA (the validated reference), then on E2B.
PASS: E2B's teacher-forced agreement >= the reference's - 2.0 points and its free-run passes >= the reference's - 1; 2.0-4.0 points below is
ambiguous (parked for the owner); worse, or free-run passes 2+ short, fails.
```

## TestGemma4EModel_realE4BNonInferiority

Moved from `cuda/gemma4_emodel_real_test.go` (the comment above `TestGemma4EModel_realE4BNonInferiority`) on 2026-10-09.

```text
TestGemma4EModel_realE4BNonInferiority is G-E4B-C1 (docs/tasks/task-multimodal-support-2026-10.md, "S6 on nobara", registered 2026-10-08 before any run): G3c's procedure and rule, unchanged, on Gemma 4 E4B loaded from its
safetensors directory by both sides.
```

## TestGemma4EModel_realE4BAnchorDump

Moved from `cuda/gemma4_emodel_real_test.go` (the comment above `TestGemma4EModel_realE4BAnchorDump`) on 2026-10-09.

```text
TestGemma4EModel_realE4BAnchorDump is an EXPLORATORY dump, not a gate. G-E4B-C1 failed (CUDA against the CPU: 87.59% teacher-forced, 3/8 free-run passes, against the E2B's 94.48% and 7/8), and a G3 comparison
of two int4 implementations cannot say which of them, if either, is wrong. This writes, for the CPU's own greedy sequence on each of G3's prompts, both arms' top-8 (id, logit) at every position, so that an HF
float32 forward over the SAME ids (scripts/anchor_e4b_hf.py) can say, at every position where the arms disagree, which one HF sides with. Output: $E4B_ANCHOR_DIR (default ~/goinfer-logs/e4b-anchor/dump.json).
```

## TestGemma4EModel_realE4BF32Dump

Moved from `cuda/gemma4_emodel_real_test.go` (the comment above `TestGemma4EModel_realE4BF32Dump`) on 2026-10-09.

```text
TestGemma4EModel_realE4BF32Dump is an EXPLORATORY follow-up to TestGemma4EModel_realE4BAnchorDump, not a gate. The anchor read both int4 arms about 77% from HF float32 (CPU 77.70%, CUDA 77.01%) against 87.59% from each other, which a shared
deviation or plain int4 sensitivity could both produce. This removes quantization: goinfer's CPU forward in float32 (Options.Quant "f32") teacher-forced over the SAME ids, written in the dump's shape so scripts/anchor_e4b_hf.py can
compare it with HF float32. If the implementation is right, float32 against float32 is near-identical; if it is not, the gap is the bug.
```

## TestGemma4EModel_realE4BQATNonInferiority

Moved from `cuda/gemma4_emodel_real_test.go` (the comment above `TestGemma4EModel_realE4BQATNonInferiority`) on 2026-10-09.

```text
TestGemma4EModel_realE4BQATNonInferiority is G-E4B-C1b (docs/tasks/task-multimodal-support-2026-10.md, "S6 on nobara", re-registered 2026-10-08 before any run on this file): G3c, unchanged in procedure, rule and reference, on Google's
quantization-aware-trained E4B GGUF, the equivalent of the E2B file G3c used. The plain bf16 checkpoint (TestGemma4EModel_realE4BNonInferiority) failed for the checkpoint's sake, not the implementation's.
```

## requireEModelFixture

Moved from `cuda/gemma4_emodel_test.go` (the comment above `requireEModelFixture`) on 2026-10-09.

```text
requireEModelFixture skips unless the tiny E-model checkpoint is complete. The directory is not gitignored as a whole (only *.safetensors is), so a
pinned worktree that symlinks just the ignored weights (run-gate-gpu.sh) holds a directory with model.safetensors and no config.json: present, and
unloadable. A dir-only check turns that into a failure; the first 2026-10-07 night gate run hit exactly that.
```

## TestGemma4EModel_plantedDefects

Moved from `cuda/gemma4_emodel_test.go` (the comment above `TestGemma4EModel_plantedDefects`) on 2026-10-09.

```text
TestGemma4EModel_plantedDefects is G2c: each planted defect, alone, must turn G1c red. A defect that stays green means the fixture is degenerate
along that axis (fix the fixture, not the bar). Defect (5), the K/V store not skipped on a shared layer, was registered with its outcome open
(on Metal it is a byte-for-byte no-op and cannot go red). On CUDA it DOES go red (2026-10-07: mean cosine 0.763998 against 0.999931 clean), because
the shared layer's K/V scratch holds another layer's data and the store writes it through the alias into the source's cache; so it is a gate here.
```

## graphsBitExact.tautology

Moved from `cuda/gemma4_graphs_test.go` (the comment above `graphsBitExact.tautology`) on 2026-10-09.

```text
TAUTOLOGY GUARD. This gate compares a graphs-ON build against a graphs-OFF build; if the
ON build did not actually capture, both arms run the live path and the comparison is
live-vs-live — it passes having tested nothing, while logging "replay == live launches,
BIT-IDENTICAL". That is exactly what happened: on a DEFAULT-compute-mode box without MPS,
admitGraphs declines, and the scaled variant spent 704 s of a 26B load to print a
meaningless pass. Its sibling (sameModelUnderLoad) had this guard; the shared helper did
not, so all four bit-exact variants were affected, not just the scaled one.

Skip, don't fail: declining under unsafe tenancy is CORRECT production behaviour. What is
not acceptable is reporting it as a pass. Forcing is a separate, labelled gate run.
```

## TestGemma4Graphs_bitExact_scaled.fixture

Moved from `cuda/gemma4_graphs_test.go` (the comment above `TestGemma4Graphs_bitExact_scaled.fixture`) on 2026-10-09.

```text
Defaults to the committed-by-generator scaled fixture, like the C′ cache gates. It used to
require GOINFER_HEAVY_TESTS *and* an explicitly-named fixture, which is why it never ran: the
variable pointed at nothing that existed, and aimed at the real 26B it spent 704 s per arm.
The scaled fixture runs it in ~10 s, so it belongs in the forced-graphs group with the rest.
```

## loadG4MoECache.structural

Moved from `cuda/gemma4_moe_cache_test.go` (the comment above `loadG4MoECache.structural`) on 2026-10-09.

```text
The cache=false arm needs the WHOLE expert stack in VRAM, which is precisely what the
cache exists to avoid. Pointing this gate at a model that does not fit (the real 26B:
~11.4 GB of experts on an 8 GB card) therefore fails STRUCTURALLY, not numerically —
it burned 307 s to reach an OOM the runtime already knew about. Say so here rather
than leaving a bare decline that reads like a parity failure.
```

## TestGemma4MoE_cacheExpertsBitExact_scaled

Moved from `cuda/gemma4_moe_cache_test.go` (the comment above `TestGemma4MoE_cacheExpertsBitExact_scaled`) on 2026-10-09.

```text
TestGemma4MoE_cacheExpertsBitExact_scaled runs the same gate at the WIDTH that broke A′ zero-copy:
the correctness proof that matters for B′, and the one this track never actually had.

It had never run. GOINFER_MOE_SCALED_FIXTURE named no fixture that existed (the only MoE fixtures
were the three tiny ones), so it skipped from the day it was written; and aimed at the real 26B it
fails structurally, because the cache=false control arm cannot be resident on a card the model
does not fit. So C′ — the path the shipped 26B result runs on — was gated only at toy width
(2 layers, hidden 256, 4 experts), which is the SAME class of evidence A′ had when A′ was wrong.

testdata/gemma4-moe-scaled resolves both problems. It keeps hidden=2816 and moe_inter=704 — the
REAL per-expert row geometry, the dimension A′ was actually sensitive to — and shrinks only the
axes the A′ post-mortem excludes (128→32 experts, 30→4 layers). Its full int4 expert stack is
~428 MB, so BOTH arms are resident simultaneously-satisfiable on an 8 GB card with wide margin,
which is what makes the control arm meaningful rather than impossible.

Defaults to that fixture; GOINFER_MOE_SCALED_FIXTURE still overrides for a one-off.
```

## TestGemma4MoE_cacheReuse_scaled

Moved from `cuda/gemma4_moe_cache_test.go` (the comment above `TestGemma4MoE_cacheReuse_scaled`) on 2026-10-09.

```text
TestGemma4MoE_cacheReuse_scaled is the same gate with cross-token slot reuse AND eviction active
at real width: nSlots=12 sits between topK=8 and nE=32, so the LRU both hits and evicts. Step-2's
reuse path is what the 26B actually decodes on (38 slots of 128), and until now it too was only
gated at nE=4.
```

## TestGemma4MoE_localize

Moved from `cuda/gemma4_moe_localize_test.go` (the comment above `TestGemma4MoE_localize`) on 2026-10-09.

```text
TestGemma4MoE_localize is the localization harness the task-2c steer asked for BEFORE the gate:
at pos 0 it diffs the four gemma4-MoE-layer buffers (rn / wgt / x1 / x2) resident-vs-CPU, so a
whole-forward miss points at router vs dense branch vs expert branch (and the join, by elimination)
in one run instead of a cosine that only says "wiring". Debug at pos 0 (smallest error), per the
steer. Diagnostic: logs, never fails (the gate is TestGemma4MoE_residentParity).
```

## TestGemma4MoE_localize.amendment

Moved from `cuda/gemma4_moe_localize_test.go` (the comment above `TestGemma4MoE_localize.amendment`) on 2026-10-09.

```text
S1.0 amendment 2026-10-07 (docs/tasks/task-multimodal-support-2026-10.md): the bar sits between the before-v_norm-fix and after readings; the fix is the mechanism. Never loosened. Worst decision before the fix: wgt 0.999731, x1 0.999125, x2 0.999387; after: 1.000000 on all three.
```

## TestGemma4MoE_residentParity

Moved from `cuda/gemma4_moe_resident_test.go` (the comment above `TestGemma4MoE_residentParity`) on 2026-10-09.

```text
TestGemma4MoE_residentParity is Split-B task 2c's END-TO-END gate. Every primitive underneath is
pinned in isolation and the pos-0 per-branch parity (TestGemma4MoE_localize) proves the wiring, so
this asserts kernel correctness at the logit level (pos 0) and then CHARACTERIZES the multi-position
int4-vs-int4 drift with two calibrated instruments instead of a picked-below-observed floor:

 1. ROUTING AGREEMENT at EVERY position, not just pos 0. rn derives from h derives from the KV
    cache, so as attention drifts a top-k flip at position N becomes possible even though pos 0 is
    clean — and a flipped expert reads IDENTICAL to accumulation in a cosine (same shape, same
    "grows with position"). If resident idx == CPU idx at every position, accumulation is the only
    explanation left; a flip means the 0.87 has a discrete component.
 2. A CALIBRATED curve: CUDA-int4-vs-CPU-int4 vs CPU-int4-vs-CPU-f32 at the same positions. The
    latter is "as well as int4 arithmetic can agree with f32". CUDA-vs-CPU-int4 (same weights, only
    W4A8 activation rounding differs) should track it or sit ABOVE it. If CUDA drops FASTER than the
    fixture's own quantization curve, that's a real divergence no conditioning explains.
```

## TestGemma4MoE_residentParity.positions

Moved from `cuda/gemma4_moe_resident_test.go` (the comment above `TestGemma4MoE_residentParity.positions`) on 2026-10-09.

```text
16 positions: pos 7 was the only inversion AND the endpoint (max accumulation), so run out to
2× to see whether the CUDA-vs-CPUint4 / CPUint4-vs-f32 gap STABILIZES or keeps widening — a
widening gap at the tail would flake the tolerance for reasons unrelated to a bug.
```

## TestGemma4MoE_residentParity.marginGate

Moved from `cuda/gemma4_moe_resident_test.go` (the comment above `TestGemma4MoE_residentParity.marginGate`) on 2026-10-09.

```text
---- MARGIN-GATED routing agreement (the reusable cross-backend MoE instrument) ----

Unconditional resident-idx == CPU-idx is an INVALID gate for MoE, and the 26B established why:
a top-k router is a DISCRETE function of a continuously-drifting input, so where two selected
experts are near-tied in router probability, the tiny W4A8-activation delta between resident and
CPU legitimately FLIPS the selection — a different-but-correct expert, not a bug (the resident
and CPU routers are each bit-exact given their own input; only the input differs by rounding).
Past such a flip the two backends compute different experts, so a hidden-state cosine CLIFFS and
per-position argmax vs CPU goes to noise; neither is a defect. But at WIDE margin a flip is NOT
explainable by rounding — it means the dispatch fed the wrong activation, or the router itself
diverged — a real bug. So gate on the margin: assert index agreement only where the top-k
boundary margin (smallest-selected minus largest-rejected softmax prob) exceeds a threshold;
below it, record the disagreement as expected sensitivity rather than failing.

THRESHOLD (marginGate = 0.01), chosen from the MEASURED margin distribution, which is bimodal by
~2 orders of magnitude (metal/gemma4_moe_noisefloor_test.go + metal/gemma4_26b_routing_test.go):
  - THIS fixture, gemma4-moe-tiny (nE=4, top-2): min margin 0.2679 — every decision well-separated
  - real width, 26B (nE=128, top-8): flips sit at 0.00115, matched at 0.00218 — the near-tie band
  - the degenerate control, gemma4-moe-kv-tiny: 0.0001 — routing is a coin-flip, non-gating
0.01 sits 5x above the near-tie band (0.002) and 27x below this fixture's min (0.268) — an order
of magnitude clear of both regimes, so it is robust to per-arch softmax-scale drift. On moe-tiny
every margin is >> 0.01, so this gate stays FULLY STRICT here (a real nE=4 dispatch bug still
fails); the sensitivity exemption only ever fires at real width, where unconditional agreement is
the wrong bar. To reuse on another MoE family, confirm its well-separated band still clears 0.01
(wider nE compresses margins) and re-pick from that family's distribution if it does not.
```

## TestGemma4MoE_residentParity.amendment

Moved from `cuda/gemma4_moe_resident_test.go` (the comment above `TestGemma4MoE_residentParity.amendment`) on 2026-10-09.

```text
S1.0 amendment 2026-10-07 (docs/tasks/task-multimodal-support-2026-10.md): the bar sits between the before-v_norm-fix and after readings; the fix is the mechanism. Never loosened. pos0 0.999581 -> 1.000000, run mean 0.947512 -> 1.000000.
```

## TestGemma4MoE_residentParity.calibrated

Moved from `cuda/gemma4_moe_resident_test.go` (the comment above `TestGemma4MoE_residentParity.calibrated`) on 2026-10-09.

```text
CALIBRATED, RUN-LEVEL. A per-position CUDA ≥ CPUint4-vs-f32 gate is too literal: the two curves
measure DIFFERENT perturbations (CUDA differs from CPU only in W4A8 activation rounding; the
baseline is the full int4 weight quantization), so they legitimately CROSS position-to-position
(run to 16 and CUDA dips under at pos 7/9/15 — with routing bit-equal at all 32 decisions, i.e.
no flip, those are conditioning, not bugs). The property that survives a prompt/length change is
the run mean: CUDA must agree with CPU-int4 AT LEAST AS WELL, on average, as int4 agrees with
f32 — the activation perturbation is smaller than the weight one, so this holds by construction
and by a wide margin (~0.95 vs ~0.87). A real divergence (CUDA dropping FASTER than the fixture's
own quantization across the run) sinks the mean below the baseline; conditioning cannot.
```

## TestGemma4MoEScaled_residentParity

Moved from `cuda/gemma4_moe_scaled_parity_test.go` (the comment above `TestGemma4MoEScaled_residentParity`) on 2026-10-09.

```text
TestGemma4MoEScaled_residentParity closes audit Check A: until this landed there was NO asserting
parity gate for the Gemma-4 resident forward against the CPU path at real width.

What existed, and why none of it covered this:
  - TestGemma4MoE_residentParity — resident-vs-CPU, but on gemma4-moe-tiny (2 layers, hidden 256,
    4 experts). Real assertions, toy width.
  - TestGemma4DenseScaled_residentParity — real head geometry, but DENSE and random-weight.
  - the real-26B gates (cache/graphs bit-exactness) — compare the resident path against ITSELF
    with one knob moved. Nothing anchored either arm to CPU.
  - TestGemma4_12B_logitParity — CPU vs HF bf16, never touches the resident path.

So every real Gemma-4 checkpoint that reached the resident path was compared only to itself, and
the shipped "26B decodes coherently at ~17 tok/s" rested on a distinct-trigram degeneracy score —
a forward that was numerically wrong but non-repetitive would have passed everything. Until this
gate existed, GOINFER_GEMMA4_RESIDENT could not be defaulted on: the flag had become load-bearing
by accident. This is what let it come off (a5ebb35).

The fixture keeps hidden=2816 / moe_inter=704 / head_dim 256 local, 512 global K=V — the real
26B's per-expert and per-head geometry — and shrinks only expert count and depth. Its per-group
weight scales are TRANSPLANTED from the real 26B (log2std 0.32, 24.1x spread on experts vs 0.27 /
5.0x for random init), because a fused-multiply-add defect that cost 84% stream divergence in
v0.9.0 was invisible on uniform random weights. See scripts/pin_gemma4_moe_scaled.py.
```

## TestGemma4MoEScaled_residentParity.amendment

Moved from `cuda/gemma4_moe_scaled_parity_test.go` (the comment above `TestGemma4MoEScaled_residentParity.amendment`) on 2026-10-09.

```text
S1.0 amendment 2026-10-07 (docs/tasks/task-multimodal-support-2026-10.md): the bar sits between the before-v_norm-fix and after readings; the fix is the mechanism. Never loosened. pos0 0.998601 -> 1.000000, run mean 0.960917 -> 0.996078.
```

## TestGemma4Router_residentIdxParity

Moved from `cuda/gemma4_router_parity_test.go` (the comment above `TestGemma4Router_residentIdxParity`) on 2026-10-09.

```text
TestGemma4Router_residentIdxParity is Split-B task 2a, the ROUTER-FIRST gate: before any expert
GEMV output is compared, prove the resident router selects the SAME experts as the CPU router —
a binary idx[] equality check, the one MoE failure a whole-forward cosine can't localize (a
flipped expert is a different computation, not a small error).

The resident router uses gemv_f32_f32 (cuda/router_f32.cu) — a PURE-f32 projection, NOT the
shared int8-activation gemv_f32_a8. That is a deliberate choice, not a reuse: gemv_f32_a8 would
quantize the router input rn to int8 (~1e-2), which can flip a top-k decision near a tie. An
earlier version of this test ran that int8 path and found no flip — but the gemma4-moe-tiny
fixture's 0.12 routing margin was CONSTRUCTED by least-squares to be wide, so that result is
CIRCULAR for a trained 128-expert/top-8 router whose 8th-vs-9th boundary is far tighter. f32xf32
quantizes NOTHING, so the only residual is f32 reduction order (~1e-6) — routing cannot flip from
activation quant at ANY expert count. This test therefore verifies the kernel we actually ship;
the 128/top-8 re-run is no longer a correctness precondition (there is no quant perturbation to
re-check), only a nice-to-have when a real router is available.
```

## TestGemma4DenseTwoGeom_residentParity.amendment

Moved from `cuda/gemma4_twogeom_parity_test.go` (the comment above `TestGemma4DenseTwoGeom_residentParity.amendment`) on 2026-10-09.

```text
S1.0 amendment 2026-10-07 (docs/tasks/task-multimodal-support-2026-10.md): the bar sits between the before-v_norm-fix and after readings; the fix is the mechanism. Never loosened. minCosine 0.977972 -> 0.999942.
```

## TestBuildScratchAccounting.warmup

Moved from `cuda/build_scratch_accounting_test.go` (the comment first load inside `TestBuildScratchAccounting`) on 2026-10-09.

```text
The first load in a process also pays the context and module cost (~386 MiB on the 0.5B) that a serve process has paid before it plans: a warm-up, not a row.
```

## TestDeltaNetSnapshotCUDA.batch

Moved from `cuda/deltanet_snapshot_cuda_test.go` (the comment comment above the one-batch copy in `TestDeltaNetSnapshotCUDA`) on 2026-10-09.

```text
One batch, one synchronize, for all 36 copies — the form aikit added for exactly this
consumer. A loop over CopyDevice would pay 36 synchronizes instead of one.
```

## TestDFlashDispatchAmortization.verdict

Moved from `cuda/dflash_dispatch_test.go` (the comment verdict line in `TestDFlashDispatchAmortization`) on 2026-10-09.

```text
The verdict: per-layer cost at 5 layers against per-layer cost at the full stack. The
6.6 ms draft assumes this ratio is 1.0.
```

## TestB2DenseFlagship

Moved from `cuda/b2_flagship_test.go` (the comment above `TestB2DenseFlagship`) on 2026-10-09.

```text
TestB2DenseFlagship measures a fitting dense flagship (qwen2.5-7B int4, resident on the 2070 SUPER
with real KV headroom) end-to-end, §B2 method: TTFT at 128/512/2048 + all-in and decode-only tok/s,
best of 3 warm with the first discarded. Pair the goinfer column with pinned Ollama 0.5.7 for the
published §B2 row. The claim here is the honest one: faster decode, prefill within a stated multiple,
crossover at a measured prompt length.

	GOINFER_HEAVY_TESTS=1 go test -tags cuda -run TestB2DenseFlagship -v -timeout 900s
```

## TestResidentDrafter_extendContext

Moved from `cuda/drafter_test.go` (the comment above `TestResidentDrafter_extendContext`) on 2026-10-09.

```text
TestResidentDrafter_extendContext gates the drafter's context K/V on device.

It checks the two things this path does that a decoder layer does NOT, because both are
silent when wrong — the K/V would still be the right shape at the right positions:

	INCREMENTAL: extending by 4 then 4 must land the same K/V as extending by 8 in one call.
	That is the property the serving path depends on (rebuilding costs 2.4x at ctx=1024, per
	TestDFlashDraftScaling), and an off-by-one in the write position breaks it while leaving
	every buffer plausibly populated.

	POSITION-DEPENDENT: rows written at different absolute positions must DIFFER even for
	identical input, because RoPE rotates by position. If they matched, the rope call is being
	handed the wrong start and every drafted token after the first block would be subtly wrong.
```

## TestE2EDecodeThroughput_synthetic

Moved from `cuda/e2e_decode_test.go` (the comment above `TestE2EDecodeThroughput_synthetic`) on 2026-10-09.

```text
TestE2EDecodeThroughput_synthetic is the end-to-end cgo-free CUDA decode THROUGHPUT measurement
(docs/prompts/cuda-measure-e2e-decode.md): the full per-token work — GEMVs PLUS the
glue the 244 projection omitted (RMSNorm+quant, RoPE, GQA attention, SwiGLU+quant,
residual, argmax) — so the tok/s is end-to-end, not a streaming ceiling. Shippable
config: PTX compiled offline (NVRTC) + go:embed'd + DRIVER-JIT'd (no libnvrtc in the
binary), every launch through gocudrv's LockOSThread executor channel (its hop is in
the number), CGO_ENABLED=0. Synthetic weights (bandwidth is value-independent); the
non-trivial kernels are cosine-validated vs a CPU reference here. Run:
CGO_ENABLED=0 go test -tags cuda -run E2EDecode -v
NOT A CORRECTNESS GATE. This measures THROUGHPUT over SYNTHETIC random weights — no model
is loaded, so the "token" it argmaxes is a pick over garbage and there is no CPU reference a
token-identity assertion could be written against. It was previously named TestE2EDecode and
asserted nothing, so it read as e2e correctness evidence for CUDA greedy decode while being a
benchmark. Correctness for that path lives in TestRealE2EDecode (real model, token identity vs
the CPU reference) and in the per-kernel bit-identity gates.
```

## TestCohereRealResidentParityCUDA.prompt

Moved from `cuda/cohere_real_resident_test.go` (the comment above `TestCohereRealResidentParityCUDA.prompt`) on 2026-10-09.

```text
48-token prompt: the golden prompt cycled, then a deterministic tail (the construction the 2026-10-01
measurement used, so the numbers are comparable).
```

## a9ChildEnv

Moved from `cuda/moe_route_demand_test.go` (the comment above `a9ChildEnv`) on 2026-10-09.

```text
A9 reopened here. The reservation is a confirmed COST and was not yet a confirmed CAUSE, because
the arithmetic does not close:

	free immediately before the failing launch   198,836,224 B
	measured moe_route reservation               138,412,032 B
	spare                                         60,424,192 B

The reservation fits, and the launch failed anyway. Worse, free after the failure was 265,945,088
— 67,108,864 B ABOVE the pre-attempt level. An unwind returns to the pre-attempt level; it cannot
exceed it. So something that existed BEFORE the attempt was released, which reads as the driver
trimming a cache to satisfy a request it still could not satisfy. If that is right the true demand
is above 265,945,088 and the 132 MiB reservation is one component of it.

This measures the demand directly instead of inferring it: balloon the device to leave a chosen
number of bytes free, launch moe_route, and binary-search the pass/fail boundary.

PRE-REGISTERED readings:

	threshold ~= 138,412,032           the reservation is the whole demand, and the 34-slot failure
	                                   needs a different explanation entirely
	198,836,224 < threshold <= 265,945,088   consistent with the observed failure; the reservation is
	                                   one component and the remainder needs naming
	threshold > 265,945,088            demand exceeds even the post-trim free, and the trim behaviour
	                                   is part of the mechanism

If the result VARIES run to run at the same balloon size, that is contiguity rather than capacity
and it is a different finding. Contiguity was refuted earlier in this campaign against a different
observation (a fresh heap had worse contiguity than the slot-loaded one at equal free); that
refutation was about slot buffers and does not carry here.
```

## TestMoERouteDemandThresholdChild

Moved from `cuda/moe_route_demand_test.go` (the comment above `dev.CompileLibrary(moePTXOrOverride())`) on 2026-10-09.

```text
Everything the launch needs is allocated BEFORE ballooning: the module, the pipeline, and the
four small buffers. That is both the right shape (in production they exist long before the
launch) and the fix for a real bug in the first version, which ballooned first and then could
not allocate 32 bytes for an argument buffer.
```

## TestMoERouteDemandThresholdChild.balloon

Moved from `cuda/moe_route_demand_test.go` (the comment above the GOINFER_A9_BALLOON=fine branch) on 2026-10-09.

```text
Back off on failure rather than stopping. A failed request does not mean the heap is full —
it means THAT SIZE does not fit, which is a statement about contiguity, not capacity. Halving
until the quantum is reached is what actually drains the pool; the first version gave up on
the first refusal and left 307 MiB unballooned against a 64 MiB target. The bracket check in
the parent caught that, which is the only reason it is not silently in the numbers below.
Balloon SHAPE is a variable, not a detail. A deterministic balloon produces a deterministic
heap layout, so identical repeats do NOT by themselves exclude contiguity — they only exclude
run-to-run noise. Filling with many small blocks instead of a few large ones leaves the same
free BYTES in a very different arrangement; if the threshold is capacity it should barely
move, and if it is contiguity it should.
```

## TestMoERouteDemandThreshold.drainer

Moved from `cuda/moe_route_demand_test.go` (the comment above `drainsDevice`) on 2026-10-09.

```text
MARKED AS A DRAINER, and found by the gate rather than by the derivation — see the note in
cuda/drain_marker_test.go. The bisection deliberately balloons the device to leave as little as
64 MiB (below the 144 MiB floor) and records the resulting refusal as data: `bracket low: leave
67108864 -> ok=false` IS a refusal, driven on purpose. It balloons through child processes, so
each child's memory is returned when it exits, but a child that fails or hangs leaves the
device at the floor for whatever runs next in this process — which is exactly what the log
showed: the following test opened with `free at start 151191552 B`.
```

## TestMoERouteDemandThreshold

Moved from `cuda/moe_route_demand_test.go` (the comment above the pin constants) on 2026-10-09.

```text
PINNED (item 6). The threshold is the number the cap analysis depends on; leaving it unasserted
makes this a report rather than a gate. Both bounds are pinned because the pair brackets the
demand and a one-sided pin would drift.

RE-DERIVED 2026-08-12 (A11), not edited to match a red gate. The pins moved +589,824 B, and
that is the number A9-RESID recorded as "baseline drift" — the amount by which
demand = floor + residual failed to close at MOE_MAX_E=512 while closing EXACTLY at 256:

    256:  151,191,552 + 54,525,952 = 205,717,504   measured 205,717,504   EXACT
    512:  151,191,552 + 138,412,032 = 289,603,584   measured 289,013,760   short by 589,824

The measurement now reads 289,603,584 — the closed form, to the byte. Both components were
re-measured here and BOTH HELD: the floor is 151,191,552 (allocate-until-failure in a fresh
context: 7,665,287,168 reported, 7,514,095,616 obtained) and the residual is 138,412,032.
So nothing about the machine or the kernel moved; the OLD PIN was the outlier, recorded from
the one measurement that did not close, and the 589,824 was misattributed to drift rather
than read as a failure to close.

The new values are therefore the DERIVED ones, and the identity is what justifies them. If
these ever move again, check the identity first: if floor + residual still equals the demand,
the components are what moved and this pin is downstream of them.
RE-DERIVED 2026-08-19, because the old pin (287,506,432 / 289,603,584) failed — and it failed
for a reason worth more than the number it was guarding.

THE OLD PIN WAS A SUM WHOSE VALUE DEPENDS ON A PRECONDITION THIS TEST DOES NOT CONTROL:
whether another CUDA context is alive on the device while the child launches. Measured both
ways on one box, one commit, minutes apart:

	child driven by this test (parent process holds a context):
	  leave 141,819,904 -> freeBefore 141,557,760  ok=TRUE
	child driven straight from a shell (nothing else on the card):
	  leave 141,819,904 -> freeBefore 140,050,432  ok=FALSE
	  leave 289,603,584 -> freeBefore 288,948,224  ok=TRUE

So there are two regimes, and the launch's requirement differs by exactly the device-wide
reserve that the FIRST context on the card pays:

	WARM (another context alive)  demand = residual                     ~= 138.4 MiB
	COLD (this is the first)      demand = deviceFloor + residual        = 289,603,584

The RESIDUAL held bit-for-bit across every measurement (138,412,032), and the floor is the
same 151,191,552 TestAllocFloor reports. Nothing about the kernel or the machine moved; the
pin recorded the COLD sum and the test now runs WARM (it was moved into the drain group's own
process, where the preceding A10 tests leave a context on the device). A pin that flips with
its neighbours is measuring the neighbourhood.

So pin the COMPONENTS, which are stable, and assert the IDENTITY against the regime actually
observed. That is the re-derivation the old assertion demanded ("re-deriving, not editing") —
and it now fails for a moved KERNEL rather than for a moved test-ordering.
RE-DERIVED 2026-08-21, and this time the component that moved was the FLOOR — which the
previous revision could not have discovered, because it asserted the opposite.

The gate failed with "measured 192,675,840, expected 289,603,584..295,895,040" and concluded
"a break here means the KERNEL's launch requirement moved". IT DID NOT. Three measurements
settle it:

	1. The value is bit-stable — 192,675,840 on four consecutive runs, not a wobble.
	2. It is IDENTICAL at c6760d7, the commit that recorded 141,557,760 and pinned against it.
	   Same commit, same box, two days apart, different number: nothing in the tree moved.
	3. TestAllocFloor now measures the floor at 54,263,808, not 151,191,552.

And then the identity closes to the byte:

	54,263,808 (floor) + 138,412,032 (residual) = 192,675,840 = the measured demand

So the model is exactly right and one of its INPUTS changed underneath it. The residual is
unchanged (it is also what the launch actually consumes at the threshold: 192,675,840 ->
54,263,808 leaves precisely 138,412,032). The floor dropped by 96,927,744 B for a reason
outside this repo — same driver, same uptime, no reboot between the two measurements.

WHY THE GATE BLAMED THE KERNEL: its own comment claimed "the residual and the floor are each
pinned by their own gate". That is TRUE of the residual and FALSE of the floor —
TestAllocFloor says in as many words "Not a threshold assertion: the number is the finding",
so a floor move cannot fail there and surfaces here instead, wearing a kernel move's clothes.
The fix is not this constant; it is the missing pin, now added in TestAllocFloor, so the next
component move fails where the component is.

SAFETY DIRECTION, checked rather than assumed: a SMALLER floor means less memory is reported
free but unallocatable, so there is MORE headroom than the cap analysis assumed, not less. The
margin clears it by 332.2 MiB (slotMarginBytes 402,653,184 vs floor 54,263,808). A1/A5/A7/A9's
conclusions are unaffected in the safe direction; the 33-slot cap stays safe and the 34-slot
cap stays unsafe for the residual reason, which did not move.
RE-DERIVED 2026-08-26 (P16, the driver/distro re-anchor): pinnedDeviceFloor 54,263,808 ->
1,769,472. This is the SECOND time this pin has moved for a reason outside the repo, and the
second time the procedure the message below prescribes has been followed to the letter — with
the same answer. Both components were re-measured independently before anything here was
touched: the residual PASSES unchanged at 138,412,032, the floor is 1,769,472 (three separate
processes, byte-identical), and

	1,769,472 + 138,412,032 = 140,181,504 = the measured demand, to the byte.

A COMPONENT moved; the kernel did not. Step (2) below — the branch that would require
re-deriving A1/A5/A7/A9 — is NOT what happened, and must not be read as if it were. The cause
is the 2026-08-25 Nobara 43 -> 44 upgrade (NVIDIA 595.58.03 -> 595.91.07, kernel 7.0.5 ->
7.2.0, glibc, CUDA 13.2). Safety direction unchanged and again in the safe sense: a smaller
floor means MORE headroom, and the margin now clears the worst-regime demand by 250.3 MiB
(slotMarginBytes 402,653,184 vs 140,181,504) where it cleared by 200.2 MiB before.

KNOWN LATENT DEFECT, recorded here rather than fixed in the same change: the `warm`
discriminator below reads `freeBefore < pinnedDeviceFloor`, which can only be true when the
FLOOR EXCEEDS THE RESIDUAL. That held when the floor was 151,191,552 and has been false since
2026-08-21, so the WARM branch is now unreachable and a warm run would go red claiming a
broken identity. The drain group always runs this cold, which is why it has never fired. See
P16 in docs/queue-performance.md.
```

## TestMoERouteDemandThreshold.regime

Moved from `cuda/moe_route_demand_test.go` (the comment above `foreignCUDAContexts()`) on 2026-10-09.

```text
REGIME, asked of the device rather than inferred from a number.

This used to read `warm := firstPass.freeBefore < pinnedDeviceFloor`, which
can only be true when the floor EXCEEDS the residual. That stopped being
true on 2026-08-21 and the comment above has carried the consequence as a
KNOWN LATENT DEFECT since: the warm branch was unreachable, and a warm run
would go red claiming a broken identity. On 2026-09-01 it did exactly that,
on a desktop session where KDE's compositor held a context.

The identity was never the problem. It closed to the byte in that very run
once the floor was measured rather than assumed:

	18,546,688 (measured floor) + 138,412,032 (residual) = 156,958,720 = demand

What is NOT pinned is the floor in the presence of a foreign context (it was
16 MiB higher there). So when one exists this SKIPS rather than asserting a
number nobody has pinned — a skip is not a pass, and the gate reports it as
uncovered, which is the honest outcome. Asserting the cold pin anyway is
what sent a reader toward re-deriving A1/A5/A7/A9 over a compositor.
```

## TestMoERouteDemandThreshold.errmsg

Moved from `cuda/moe_route_demand_test.go` (the comment above the identity t.Errorf) on 2026-10-09.

```text
CHECK THE COMPONENTS BEFORE BLAMING THE KERNEL. An earlier revision of this message
asserted flatly that a break here means the kernel moved, and it was wrong the first time
it fired: the floor had halved and the identity still closed. So the message now says what
is actually known — the SUM disagrees — and names the two ways that happens, in the order
they should be checked.
```

## TestMoERouteDemandThreshold.margin

Moved from `cuda/moe_route_demand_test.go` (the comment above `coldDemand`) on 2026-10-09.

```text
---- the RELATIONSHIP, not just the figures (item 2) ----

The three per-kernel byte pins say "a number changed". This says "the safety property broke",
which is the one that explains why anyone should care. slotMarginBytes exists to leave room
for exactly the costs measured here, and nothing checked that it does.

MAX, not SIGMA. Launching the whole census (moe_route + rope_kv + rope_kv_batched) gives a
threshold and a residual IDENTICAL to moe_route alone, to the byte — the driver shares one
local-memory backing store sized by the largest kernel rather than summing them. Summing would
overstate the requirement, so the assertion is against the maximum, and the census gate is what
guarantees the maximum is taken over every kernel rather than a remembered one.

THE REGIME IS PART OF THE CLAIM. That measurement launched the census SEQUENTIALLY IN ONE
CONTEXT, which is what goinfer does today: batch-1, single stream, one resident model. Under
concurrent residency on separate streams there is no reason the bound stays `max` — two
kernels in flight may each need their own backing store — and this assertion would then be
wrong WITHOUT FAILING, which is the worse of the two ways to be wrong. If goinfer gains
concurrent streams or multi-model residency on one context, re-measure before trusting this.
AGAINST THE WORST REGIME, not the measured one (2026-08-19). The measurement above may be
WARM, where the launch does not pay the device reserve — but the margin's job is to be
sufficient whatever the card's state, and asserting it against the smaller warm figure would
let the cold requirement exceed the margin without failing. So the safety check uses
max(measured, cold), which is regime-independent by construction.
```

## TestMoERouteFirstLaunchReservation

Moved from `cuda/moe_route_reservation_test.go` (the comment above `TestMoERouteFirstLaunchReservation`) on 2026-10-09.

```text
MEASURED (RTX 2070 SUPER, 40 SMs, driver 595.58.03, 2026-08-12), both instruments agreeing:

	step                          cuMemGetInfo free     nvidia-smi process
	start                            7 664 697 344 B          102 MiB
	CompileLibrary(moePTX)           7 664 697 344 B          102 MiB   (cost 0)
	5x NewComputePipeline            7 664 697 344 B          102 MiB   (cost 0)
	first launch, shared_gate_combine  unchanged              unchanged (cost 0)
	first launch, moe_route          7 524 188 160 B          236 MiB   (cost 138 412 032 B = 132 MiB)

132 MiB, paid once, at the first launch of moe_route — long after allocSlots sized the cache.
Two float[MOE_MAX_E] per-thread arrays at MOE_MAX_E=512 is 4 KiB/thread of local memory, and the
driver backs local memory for the device's occupancy on first use regardless of the 1x1 grid
goinfer launches it with. Raising MOE_MAX_E 256 -> 512 therefore doubled a hidden fixed cost from
~66 to 132 MiB; that halving is DERIVED from the form, not measured, and is recorded as the price
of the router cap rather than as an argument to change it.

TestMoERouteFirstLaunchReservation measures when moePTX's device memory is actually taken: at CompileLibrary,
at NewComputePipeline, or deferred to the first launch of one of its kernels.

A9's premise was that the cost is deferred. goinfer compiles moePTX at cuda/backend.go:cudaBackend.BuildResident and
sizes the expert cache at cuda/backend.go:cudaBackend.BuildResident (allocSlots), so under the driver's default
CUDA_MODULE_LOADING=LAZY a deferred module load would be paid AFTER the cap was computed from a
free-VRAM reading that did not include it — invisible to before/after readings around allocSlots,
and invisible to any between-slot-count delta, because it does not scale with slots.

The 26B run under CUDA_MODULE_LOADING=EAGER returned free-before-allocSlots byte-identical to the
LAZY run (3,847,880,704 B) and failed at fRoute identically. That null is NOT an answer on its
own: it is equally consistent with "EAGER took effect and module loading costs nothing" and with
"EAGER was ignored". This test discriminates them directly, with no model and no cache, by
reading free VRAM around each step.

It needs no fixture and takes seconds, which is the point — the mechanism question was never
model-dependent, and answering it inside a five-minute 26B load is what made it look expensive.
```

## TestMoERouteFirstLaunchReservation.discriminate

Moved from `cuda/moe_route_reservation_test.go` (the comment above `base == 0 || afterPipelines == 0`) on 2026-10-09.

```text
The discriminating assertion. If the whole cost lands before any kernel of the module has been
launched, then goinfer's free reading at cuda/backend.go:cudaBackend.BuildResident ALREADY includes it, and the cap
arithmetic is not being deceived by a deferred cost — A9's premise is refuted for a reason
rather than by a null. If instead the cost here is ~0, the memory is genuinely taken later, at
first launch, and A9's premise stands.

Either way this is a recording test, not a threshold: the number is the finding, and it is
logged above with its probe positions. The one thing that WOULD be a defect is measuring
nothing at all.
```

## TestMoERouteFirstLaunchReservation.launch

Moved from `cuda/moe_route_reservation_test.go` (the comment above `dev.NewCommandQueue()`) on 2026-10-09.

```text
---- and now actually launch one of its kernels ----

This is the step CUDA_MODULE_LOADING=EAGER was supposed to make unnecessary. It did not:
the readings above are byte-identical with and without it, so EAGER does not engage on this
driver/path and the 26B run made under it forced nothing. A null from a forcing mechanism
that never fired says nothing about what it was meant to force.

shared_gate_combine is the safe choice: `dst[i] += g*shDown[i]` over N elements, no cache, no
routing, no expert weights. N=1 with three one-float buffers touches nothing else.
```

## TestMoERouteFirstLaunchReservation.moe_route

Moved from `cuda/moe_route_reservation_test.go` (the comment above `nE, k := 8, 2`) on 2026-10-09.

```text
---- the kernel that actually fails: moe_route ----

shared_gate_combine materialises the module but reserves nothing, which is why it read 0.
moe_route declares `float score[MOE_MAX_E]; float sel[MOE_MAX_E]` with MOE_MAX_E = 512 —
4 KB of LOCAL memory per thread. The driver must back local memory for the device's full
occupancy on the first launch of such a kernel, no matter that goinfer launches it with one
block of one thread. That reservation is a deferred fixed cost paid at first launch, which is
A9's shape exactly — but in local memory, not module code, which is why probing the module
found nothing.
A9-RESID: nE and k are variable so the reservation can be tested for launch-configuration
dependence. Local memory is a COMPILE-TIME property, so a dependence here would itself be a
finding — the driver would be sizing the backing store from something other than the kernel's
declared footprint.
```

## TestMoERouteFirstLaunchReservation.verdict

Moved from `cuda/moe_route_reservation_test.go` (the comment above `pinnedReservation`) on 2026-10-09.

```text
VERDICT.

A9 asked whether a deferred fixed cost, invisible to the cap arithmetic, explains the 34-slot
failure. It does — but not through the mechanism A9 named.

Module code:      0 B, by BOTH instruments, at CompileLibrary, at NewComputePipeline, and at
                  the first launch of a module kernel that declares no scratch. So "moePTX's
                  load is charged after the cap is computed" is REFUTED.
Local memory:     moe_route's first launch reserves the measured figure above, because it
                  declares two float[MOE_MAX_E] per-thread arrays and the driver must back
                  local memory for the device's occupancy on first use. Paid at first launch,
                  long after allocSlots sized the cache. A9's SHAPE is confirmed; its named
                  mechanism was the wrong one.

Note what did not work. CUDA_MODULE_LOADING=EAGER was the intended forcing mechanism, and the
readings here are byte-identical with and without it — it does not engage on this driver and
path. The 26B run made under EAGER therefore forced nothing, and its null was uninformative.
A forcing mechanism has to be shown to fire before a null from it means anything.
PINNED (item 6). Asserting only "> 0" would let a MOE_MAX_E change double a hidden cost with
the gate still green. This is the RESIDUAL cost, which is 48% of the launch's PEAK demand —
see TestMoERouteDemandThreshold, which pins the other number.
```

## TestMoERouteFirstLaunchReservation.precondition

Moved from `cuda/moe_route_reservation_test.go` (the comment above `routeCost == 0`) on 2026-10-09.

```text
THE PRECONDITION IS NOW ASSERTED RATHER THAN ASSUMED. This measures a FIRST launch, and the
reservation is a CONTEXT property: once any earlier test in the process has launched
moe_route, the store is already reserved and this reads 0 B — not a changed reservation, but
a measurement that never had its precondition. It failed exactly that way in the full tier
("reservation is 0 B, pinned at 138412032") while passing alone, which is the signature of a
test whose correctness depends on its position in the suite. That is a defect independently
of any gate.

0 B is therefore reported as COULD NOT EVALUATE, not as a moved constant. Any other
unexpected value is still a real finding and still fails.
```

## TestMoERoute

Moved from `cuda/moe_route_test.go` (the comment above `TestMoERoute`) on 2026-10-09.

```text
The router is where MoE goes wrong QUIETLY. Its output steers a DISCRETE choice, so a
disagreement is not a small numeric error — it runs a different expert, and the output is
unrelated rather than slightly off. goinfer has already paid for that class once: the Granite
SSM investigation traced a 66%-agreement wall to discrete expert flips, and proved no
precision knob could recover it. So the bar here is EXACT on the selected indices, not a
cosine.
```

## TestCUDAKVSlots_pricedAgainstWhatIsLeft

Moved from `cuda/kv_slots_test.go` (the comment above `TestCUDAKVSlots_pricedAgainstWhatIsLeft`) on 2026-10-09.

```text
TestCUDAKVSlots_pricedAgainstWhatIsLeft pins how the slot count is priced — the CUDA side of the Metal double-count
bug (6807ab95), where a weights-inclusive base was compared with a live figure the weights had already left. The
free-VRAM probe is stubbed to fall by exactly what the device really allocates from a fictional starting figure,
chosen (from a calibration build) so that precisely `want` slots fit beside everything the build puts on the device
before its KV. One more slot than that is requested, so the test is two-sided:
  - pricing that counts the weights twice (a weights-inclusive base against the post-weights probe) grants fewer
    than `want`;
  - pricing KV alone against a figure read before the weights are on the device grants `want`+1, which on a real
    card would not fit.

A double count is visible only where the build's pre-KV bytes exceed half a slot, so a fixture where they do not
skips with the numbers. GOINFER_CUDA_KVSLOTS_MODEL runs it on a real checkpoint (the 1.5B: ~1 GB before KV
against ~0.46 GB per slot).
```

## TestResidentDenseBytes_matchesCUDADevice

Moved from `cuda/kv_slots_test.go` (the comment above `TestResidentDenseBytes_matchesCUDADevice`) on 2026-10-09.

```text
TestResidentDenseBytes_matchesCUDADevice pins Plan's dense-weight figure for CUDA
(decoder.Model.ResidentDenseWeightBytesFor("cuda")) against what a real build puts on the device before its KV: the
fall in free VRAM from before the load to checkKVFits' probe. That fall is the weights plus the build's scratch and
kernel modules, so the estimate must not exceed it by more than an allocation quantum per matrix-ish slack, and it
must fall short of it by less than ctxCapMarginBytes, the margin Plan and checkKVFits both reserve for exactly that
scratch. Before the fix the untied 7B was priced with its ~520 MB host-side embedding table: 4930 MB against ~4476 MB
on the device (docs/measurements/concurrency-mc1-cuda-2026-09-27.md), so the first bound failed. GOINFER_HEAVY_TESTS
adds the 7B, the one untied model here.
```

## TestResidentDenseBytes_matchesCUDADevice.slack

Moved from `cuda/kv_slots_test.go` (the comment above the slack t.Logf) on 2026-10-09.

```text
Plan prices the requested bytes; the driver rounds each buffer of a quantum or more up (allocRoundSlack), which the build prices into the plan on its own
(res.allocSlackBytes). The margin is for what neither knows, so the bound below is on Plan + that slack, not on Plan alone: on the 7B the rounding alone
is 406 MB, over the whole margin (the 2026-10-07 night gate's failure, root-caused 2026-10-08: docs/tasks/task-multimodal-support-2026-10.md).
```

## TestCUDAKVSlots_oomKeepsTheSlotsThatFit

Moved from `cuda/kv_slots_test.go` (the comment above `TestCUDAKVSlots_oomKeepsTheSlotsThatFit`) on 2026-10-09.

```text
TestCUDAKVSlots_oomKeepsTheSlotsThatFit is S18's CUDA gate (docs/tasks/task-multimodal-support-2026-10.md, "G-S18a"): when the device runs out of memory allocating a later KV slot (checkKVFits sizes the
count against the free VRAM read before the build's own scratch, with the margin the only slack, so on the 8 GB card Gemma 3 4B's four slots fit with 1.2 MB to spare and the last one misses by the scratch),
the resident keeps the slots that fit instead of declining to the CPU. Three slots requested, the seam fails slot 2: the model stays CUDA-resident with two, no device buffer leaks (the ledger equals a
plain two-slot build's), and both granted slots decode the same tokens as a plain two-slot build's. The control, no seam: all three slots. Without the recover the build declines and this goes red.
```

## TestPrefillGateVsReferenceCUDA

Moved from `cuda/prefill_gate_ref_test.go` (the comment above `TestPrefillGateVsReferenceCUDA`) on 2026-10-09.

```text
TestPrefillGateVsReferenceCUDA is docs/completed/task-prefill-gap.md §3's fidelity gate, run on CUDA for
the L2 (attn_fused) + L3 (gemm_w4a8_mma) fast prefill. It is the ONLY thing that can justify
changing the default, and until it passes both levers stay opt-in however fast they are.

THE ORACLE IS NOT THE EXACT PATH. §3.1 records why at length: the first version of this gate
scored a fast path against an exact path and called the exact path truth, but on Metal the exact
path is itself a quantisation of the activations, so the two are guaranteed to disagree for
reasons that have nothing to do with a defect, and the distance was booked against the faster
arm. Here BOTH arms are scored against a THIRD thing — the CPU backend's own forward with f32
activations and the exact f64-accumulating attention (decoder/prefill_ref_gen_test.go, Phase A,
run in its own process). A missing reference file SKIPS the cell with a message; it never falls
back to exact-as-oracle, because that fallback is precisely the mistake §3.1 corrected.

WHAT THE TWO ARMS ARE ON CUDA, and why the exact arm is the batched path rather than a
sequential decode loop. On Metal the shipped default IS sequential decode, so that was the arm to
beat. On CUDA the shipped default is already the BATCHED prefill with the exact kernels
(attn_batched + gemv_w4a8_rn), which are bit-identical to the M=1 decode kernels by construction
and measured at 0/50 diverged greedy streams. §3 says "the exact arm sets the bar because it is
what ships today", so the exact arm here is PrefillLast with both levers off.

Both arms are teacher-forced on the SAME reference-supplied tokens — not on their own greedy
output and not on each other's — so a per-position difference is attributable to the arm.

GATE, pre-registered in §3, per (model, K) DECISION cell (K in {256, 1024}; S at K=3900 is a
confirmation cell, reported the same way but not part of the decision):

	(a) fast's hard-flip count vs the reference <= exact's, over the same continuation positions
	(b) fast's mean teacher-forced agreement >= exact's mean - 1.0 pt AND fast >= exact on >= half
	    the prompts (the PAIRED comparison, not just the cell mean)
	(c) fast's mean continuation KL(reference || arm) <= 1.1 x exact's mean

PREDICTION ON RECORD, written before the run so it can be wrong in public: fast ~= exact within
noise. L3 is a pure reassociation of a float sum that performs FEWER roundings than the exact
kernel; L2's f16 K/V operands and online rescale are the only real precision change, and the L2
unit gate already puts the kernel at cosine >= 0.99999707 against exact math on those operands.
If fast is measurably WORSE, the per-lever env split (=attn / =gemm) exists to say which.

	GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks' ./cuda/ -run TestPrefillGateVsReferenceCUDA -v -timeout 4h
```

## TestPrefillGateVsReferenceCUDA.decisionset

Moved from `cuda/prefill_gate_ref_test.go` (the comment above `decisionKs`) on 2026-10-09.

```text
The pre-registered §3 decision set. GOINFER_CUDA_GATE_KS overrides it for a run that needs a
depth the standing set does not cover — which is how the production floor was justified: §3's
cells are 256 and 1024, so placing a floor between them without measuring there would be
interpolating a fidelity result nobody took. An override run labels itself in the log and
does not silently become the decision set.

THE DECISION SET IS {512, 1024}, NOT §3's ORIGINAL {256, 1024}. The fast path ships behind
fastPrefillFloor (512), and docs/measurements/prefill-l2l3-phase3-2026-09-05.md §2.6 records the
consequence the day the floor landed: "With the floor at 512 the decision set becomes {512, 1024},
and every cell in it ships on both models." This test kept grading 256 as a decision cell, so from
2026-09-05 it reported the floor's own evidence as a failed gate on every heavy run. K=256 still
runs (the floor is disabled above), as a REPORTED sub-floor cell: its DOES NOT SHIP is why the floor
is at 512, and a pass there is what moving the floor down would need (fastPrefillFloor's comment).
```

## TestPrefillGateVsReferenceCUDA.snapshots

Moved from `cuda/prefill_gate_ref_test.go` (the comment above `decoder.PrefillGatePromptSet()`) on 2026-10-09.

```text
THE SNAPSHOTS, NOT THE LIVE DOCS. PrefillGateProseFiles names live repo documents
(../docs/QUEUE.md and friends) that are edited constantly; PrefillGatePromptSet
returns the frozen copies under testdata/prefill-gate-prose-<set>/ that Phase A
actually built the reference from. b0bdf43d (2026-09-09) introduced the snapshots and
moved Phase A and Metal's Phase B onto them but did not touch this file, so from that
day until this fix a CUDA run prefixed a LIVE document and teacher-forced it against a
reference built from the SNAPSHOT of an older revision of that document — scoring both
arms on an input the reference never saw. It fails silently and plausibly: both arms
get the same wrong prompt, so the run still prints agreement figures and a verdict.
The Sep-5 CUDA cells quoted in prefill.go predate b0bdf43d and are unaffected.
```

## runCUDARefGateCell.identity

Moved from `cuda/prefill_gate_ref_test.go` (the comment above `len(identityFail) > 0`) on 2026-10-09.

```text
Prompt identity (2026-09-26, docs/measurements/prefill-ref-identity-2026-09-26.md): the reference files carry no
prompt ids, and set A's 2026-09-05 files at K = 512/1024/3900 predate the 2026-09-09 prompt snapshot — 1 to 4 of
their 10 prompts are scored against logits for different text. A cell whose exact-arm prompt-final logits sit
above KL 1.0 from the reference's (valid prompts: <= 0.8) is VOID, not a verdict.
```

## ptxModules

Moved from `cuda/kernel_local_memory_test.go` (the comment above `ptxModules`) on 2026-10-09.

```text
ptxModules is every PTX blob goinfer embeds, DERIVED from kernels.go's //go:embed list (audit
2026-09-10 G-13(b)): the hand-written list covered 15 of 22 modules, so the census never saw
gptoss_act.ptx, the expert-cache path its moe_route precondition exists for. Reading
testdata/<name>.ptx is byte-for-byte what go:embed embeds. TestPTXModules_coverEveryEmbed holds
the result to the embed list, as TestKernelFMALint_coversEmbeddedPTX does for the FMA lint.
```

## TestKernelLocalMemoryCensus

Moved from `cuda/kernel_local_memory_test.go` (the comment above `TestKernelLocalMemoryCensus`) on 2026-10-09.

```text
TestKernelLocalMemoryCensus reports CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES for every entry point in
every embedded module, and checks the backing-store multiplier against a measured reservation.

A9 established that moe_route's first launch reserves 138,412,032 B, and did so by measuring two
kernels. Two kernels is a sample. Local memory per thread is a per-kernel compile-time property,
so any kernel with per-thread arrays carries its own deferred reservation, and nothing in the tree
reported them. This is the loop.

It also settles the multiplier. "MOE_MAX_E 256 -> 512 doubled the cost from ~66 to 132 MiB"
assumes the backing store is linear in per-thread bytes with a constant occupancy factor. That is
an assumption about the driver, not an observation, and it is checked here against
multiProcessorCount x maxThreadsPerMultiProcessor rather than asserted.
```

## TestKernelLocalMemoryCensus.embedaudit

Moved from `cuda/kernel_local_memory_test.go` (the comment above the SUM t.Logf) on 2026-10-09.

```text
AUDITED 2026-09-12 against the embeds (docs/completed/cuda-megakernel-closeout.md): 31 .ptx
blobs are go:embed-ed across cuda/*.go today, 22 of them in kernels.go — exactly what
ptxModules() reports, since G-13(b) made it read that same file. The other 9
(gemv_w4a8{,_coal,_coal2,_coal3,_coal4,_fast,_v4}.ptx, gemv_w8a8.ptx, addone.ptx) were
referenced only from _test.go — variant-comparison blobs, no production path; the seven
gemv_w4a8 variants were removed 2026-09-24 (last at 8f452a7e), leaving two; megakernel.ptx
(the tenth such blob as of the prior audit) was deleted in this closeout along with the rest
of the dead scaffold. Re-run this count (`grep -n go:embed cuda/*.go`, then which vars
non-test files use) if kernels.go's own embed list ever needs independent confirmation —
the derivation above means it can no longer drift silently, only the source file can move.
```

## TestKernelLocalMemoryCensus.firstreading

Moved from `cuda/kernel_local_memory_test.go` (the comment above `route_gptoss`) on 2026-10-09.

```text
The next two were invisible until ptxModules() was derived from kernels.go's embeds
(audit-2026-09-10 G-13(b)). Both figures are this census's first reading of them.
```

## TestKernelLocalMemoryCensus.deltanet

Moved from `cuda/kernel_local_memory_test.go` (the comment above `delta_rule_rows_128`) on 2026-10-09.

```text
deltanet.ptx, the batched Gated-DeltaNet prefill (docs/tasks/task-cuda-deltanet-prefill-2026-09.md): the
scan holds its 128-float state row in registers across the rows and spills 24 floats of it at 255
registers (launch_bounds(128, 1) measured the same 96 B). 3.8 MiB at full occupancy, against the 1.23 s →
38 ms it bought on a 621-token Qwen3.5-9B prefill over the no-spill generic scan.
```

## TestKernelLocalMemoryCensus.multiplier

Moved from `cuda/kernel_local_memory_test.go` (the comment above `measuredMoERoute`) on 2026-10-09.

```text
---- the multiplier, checked rather than assumed ----

moe_route's reservation was MEASURED at 138,412,032 B (RTX 2070 SUPER, driver 595.58.03,
2026-08-12, both cuMemGetInfo and nvidia-smi agreeing). If the naive form
local x SMs x maxThreadsPerSM reproduces it, the "256 -> 512 halves it" derivation is sound.
If it does not, the multiplier carries something else and the derivation was doing more work
than it looked.
```

## lintedKernels

Moved from `cuda/kernel_fma_lint_test.go` (the comment above `lintedKernels`) on 2026-10-09.

```text
lintedKernels is the contracted-kernel list TestKernelFMALint checks, and the list
TestKernelFMALint_coversEmbeddedPTX holds to the set of kernels actually shipped as PTX.

moe.cu is exempt because the shipped moe.ptx is a FROZEN artifact, audited at NVRTC 12.6.85.

M-35 (CLOSED 2026-09-10, option (a)): the shipped moe.ptx (and glue.ptx, gemv_fwd.ptx —
cuda/kernels.go names all three as the audited set) had drifted to this box's ambient NVRTC
12.9.86 across three separate regens, none at the pinned toolchain. Re-pinned at genuine
12.6.85 per cuda/testdata/REGEN.md's own "re-pin at 12.6.85" record — moe.cu's bare MACs are
still exempt (they were never converted to intrinsics; that is still-open option (b)), but the
artifact behind the "frozen, audited" claim is real again. Per-kernel hash audit + the real
MoE-resident-parity gate (identical min cosine 0.997829 before/after) found no measurable
numeric drift from the three ambient-NVRTC regens either — see REGEN.md for the full record.

moe.cu's bare MACs remain unconverted (option (b) from the original M-35 finding), so this
exemption stays. (TestMoEPTX_versionMatchesItsDocumentation, which held REGEN.md and this
exemption to moe.ptx's banner, was deleted 2026-09-25; the pin is now kept by reading the banner.)

router_f32.cu was added AFTER this lint and never joined the list (audit C-16), so the pure-f32
Gemma-4 router projection — on the production decode path, and the one path the repo calls "the
discrete-failure path" because a near-tie flips which expert runs — sat unguarded. A kernel is
not covered by being contracted; it is covered by being in THIS list.
```

## TestKernelFMALint

Moved from `cuda/kernel_fma_lint_test.go` (the comment above `TestKernelFMALint`) on 2026-10-09.

```text
TestKernelFMALint enforces the bit-identity rule at BUILD TIME: no bare float multiply-accumulate
in any kernel under a bit-identity contract. A bare MAC (`x += a*b`, `= a*b + c`) lets the compiler
CHOOSE fma vs mul+add, and separately-compiled kernels that share a contract (a decode kernel and
its batched counterpart) then compile to ~1 ULP-different, DATA-DEPENDENT numerics — invisible on
uniform fixtures, an 84% token-stream divergence on real weights (docs/task-batched-prefill-
bitidentity.md). Every MAC must be an explicit intrinsic (__fmaf_rn / __fmul_rn / __fadd_rn) so no
compiler discretion remains. This catches the CAUSE at compile time — before any numerical test,
and independent of the NVRTC version that JITs the PTX. A new bare MAC fails the build. (aikit's
gemv_quant.cu carries the same rule in its own repo — see its header.)
```

## TestKernelFMALint.macrev

Moved from `cuda/kernel_fma_lint_test.go` (the comment above `macExprRev`) on 2026-10-09.

```text
... and the same MAC with the multiply to the RIGHT of the add (c + a * b). Its absence let
prefill_batched.cu's `1.0f + attnTempBeta * log1pf(...)` ship in a linted kernel
(audit-2026-09-10 G-11).
```

## TestKernelFMALint.forheader

Moved from `cuda/kernel_fma_lint_test.go` (the comment above `forHeader`) on 2026-10-09.

```text
A for-header is INTEGER arithmetic (the init/cond/incr), but the loop BODY on the same line is
not — a single-line `for (…) acc += a[k] * b[k];` carries a real float MAC. Skipping the whole
line (the old `\bfor\s*\(` in isDeclOrIndex) let router_f32.cu's two MACs pass unseen (audit
R-04). Strip only the `for (…)` header, then lint the remaining body.
```

## TestGemma4VLResident_bidirParity.pos0

Moved from `cuda/gemma4_vl_resident_parity_test.go` (the comment above `meanCuda < meanCpu`) on 2026-10-09.

```text
NOTE: unlike TestGemma4DenseScaled_residentParity's own pos-0 (truly zero attention
history, the least int4-noisy point it can measure), THIS pos0 is decode-step-0 AFTER a
full bidirectional-block prefill (19-31 positions of history) — already deep in this
fixture's own documented int4 chaos regime (that sibling test's own late positions, e.g.
pos 15, land at cosine 0.68 with zero bridge involvement at all: self-consistent resident
decode alone). An absolute bar at this depth would fail on ALREADY-ACCEPTED int4 noise, not
on a bridge defect — confirmed by a same-session A/B diagnostic: resident computing every
position itself vs CPU-prefill+upload for an equivalent prefix land EQUALLY far from the
CPU reference (0.68 vs 0.72), and disagree with EACH OTHER by exactly that same margin
(0.70) — two independent equally-noisy realizations, not a systematic upload defect. So
the calibrated-mean check below (which already accounts for this fixture's own chaos,
exactly TestGemma4DenseScaled_residentParity's reasoning) is the only valid bar here.
```

## TestGemmaBOSBuild

Moved from `cuda/gemma_bos_build_test.go` (the comment above `TestGemmaBOSBuild`) on 2026-10-09.

```text
TestGemmaBOSBuild looks at how L0 builds the BOS token's massive activation — the Metal box
found Metal under-builds it (goinfer |resid into L1| = 12491, Metal = 1461, 8.5x too weak),
corrupting Gemma's attention sink and craterng every downstream context. This traces the
build on the CUDA side: does the amplification live in L0's ATTENTION contribution or its
MLP contribution, and does CUDA-int4 reach f32's magnitude (so the bug is Metal-specific) or
under-build like Metal (an int4 issue CUDA tolerates)?
```

## TestGemmaOutlierTrace.trunk

Moved from `cuda/gemma_outlier_trace_test.go` (the comment above the handshake channel loop) on 2026-10-09.

```text
The channels the Metal bisect fingered as the ACTUAL failure — mid-magnitude trunk
channels Metal sign-flips or zeroes, which the final (1+w) norm (~16-20x) then amplifies
into the head. The massive channel 443 both backends track; the catastrophe is here. This
is the definitive oracle question: does CUDA keep these channels' SIGN, where Metal loses
it? Neighbors +/-1 printed too, to catch any index-offset between the two harnesses (my
drift table had 1697 with the opposite CPU sign the Mac reports at 1698).
HANDSHAKE ANCHORS: the values the Mac must match to prove we run the same input. If f32@443
differs across boxes on identical code, the inputs differ (checkpoint / aikit) and no oracle
is valid yet. If f32 matches but their "CPU" == my f32, the earlier int4-vs-int4 mismatch was
a precision-label confusion, not a bug.
```

## residentCosineParity

Moved from `cuda/gemma_parity_test.go` (the comment above `residentCosineParity`) on 2026-10-09.

```text
residentCosineParity drives BOTH paths from the SAME text, tokenized by the model's OWN
tokenizer.

It used to take hardcoded token ids, and the Gemma gate's were ones I invented and never
decoded: they were not valid Gemma tokens at all, so every Gemma parity number reported from
here (and inherited by the Metal port) was measured on gibberish. The parity CLAIM survived
that — both paths got identical input, and agreement is agreement — but the ANALYSIS did not:
the control ran on real Qwen ids while Gemma ran on nonsense, and nonsense flattens the
logits, which inflates near-ties and depresses exact-argmax. I read that signature as
"Gemma is noisier, probably the 262k vocab" instead of as a confound I had created.

Encoding real text makes a wrong id impossible BY CONSTRUCTION rather than by eyeballing, and
makes the two models comparable: same sentence, each in its own vocabulary.
```

## residentCosineParity.floor

Moved from `cuda/gemma_parity_test.go` (the comment above `minCos < 0.95`) on 2026-10-09.

```text
The GATE is the repo's own rule (gpu/kv_i8_parity_test.go): argmax must match, or differ
only inside a 3% near-tie — asserted per position above. Cosine is logged as a DIAGNOSTIC,
with only a gross-breakage floor: an early draft of this test asserted cosine ≥ 0.999 and
that bar failed the SHIPPED dense Qwen path (min 0.9936), which is why the control below
exists. W4A8 int4 does not reproduce CPU int4 to 0.999; a tighter floor here would encode a
number no backend meets.
```

## TestGemmaSublayerCUDA

Moved from `cuda/gemma_sublayer_cuda_test.go` (the comment above `TestGemmaSublayerCUDA`) on 2026-10-09.

```text
TestGemmaSublayerCUDA answers the Metal box's Fork-2 question directly: does CUDA's dp4a W4A8
path show the same 2-6x amplitude inflation Metal shows on the o-proj contribution at
channels 1723/227 (L31-33), or is CUDA's amplitude clean? If clean, Metal's blow-up is a
kernel scale bug separate from int4 quant-hostility.

int4 = the byte-identical Q4_K_M gguf (sha 882e8d2d) run through the CUDA RESIDENT dp4a path
(not CPU int4); f32 truth = the real bf16 safetensors via decoder.ForwardSubCapture. Metal's
numbers (from the relay): 1723 L32 attn +175 vs truth +27 (~6.5x); 227 L33 attn +63 (flipped)
vs truth -12.
```

## TestGemvW8A8Bandwidth

Moved from `cuda/gemv_bw_test.go` (the comment above `TestGemvW8A8Bandwidth`) on 2026-10-09.

```text
TestGemvW8A8Bandwidth is the spike's decisive-proxy experiment: decode is
weight-streaming-bound, and WebGPU sits ~37% below the 2070's bandwidth ceiling
because of its dispatch/glue wall. This measures what a *hand CUDA quant GEMV*
achieves in isolation — the ceiling a megakernel could approach once the glue is
gone. Correctness: exact int accumulation vs a CPU reference (the packing must
match); Bandwidth: weight bytes / CUDA-event kernel time, as % of the ~448 GB/s
peak. A high % here (≫ WebGPU's 37%) is the "the kernel is competent, the lane is
real" signal; a low % is an early NO-GO. Run: CGO_ENABLED=0 go test -tags cuda -run Bandwidth -v
```

## TestGemvRNBandwidth

Moved from `cuda/gemv_rn_test.go` (the comment above `TestGemvRNBandwidth`) on 2026-10-09.

```text
TestGemvRNBandwidth times gemv_w4a8_rn at the gate/up shape vs the 4.41 ms coalesced batched baseline.
```

## TestGLMResidentParity

Moved from `cuda/glm_parity_test.go` (the comment above `minCos < 0.998`) on 2026-10-09.

```text
The 3% rule and this floor divide the work, and BOTH are needed — proven by breaking
each composed piece and measuring (glm-tiny / glm-tiny-bias, this box):

  broken piece                         exact       min cosine     caught by
  ----------------------------------  ----------  -------------  -----------
  correct                              12/12,12/12  0.9998,0.9999  —
  partial-rotary tail not cached       4/12, 3/12   0.498, 0.617   3% rule
  shared-expert combine garbage        0/12, 0/12   -0.08, -0.11   3% rule
  shared expert SKIPPED entirely      12/12,11/12   0.9966,0.9949  FLOOR only
  shared gate/up swapped in glu        9/12,10/12    0.968, 0.980   FLOOR (mostly)

The last two are the ones the argmax rule MISSES: the shared expert at sharedInter=32
over four layers is a small perturbation of 256-dim logits, so dropping or mangling it
barely moves the top token — exactly the mixtral-tiny problem, and the reason a cosine
floor is not optional here. 0.998 sits below every correct run (min 0.9995 across three
prompts) and above the tightest real bug (shared skipped, 0.9966). It is a NARROW gate
(~0.0015 margin), which is the honest ceiling this tiny fixture affords for a component
this small; TestRopePartial gates the tail-caching independently and strongly.
```

## TestGlmOcrResidentParityCUDA

Moved from `cuda/glm_ocr_resident_test.go` (the comment above `TestGlmOcrResidentParityCUDA`) on 2026-10-09.

```text
TestGlmOcrResidentParityCUDA: GLM-OCR's text decoder is CUDA-resident, with GPT-J PAIRWISE rotation
(cuda/rope_pairwise.cu). It was DECLINED until 2026-10-01 because every rope kernel was NeoX:
admitted anyway, resident-vs-CPU read worst cosine -0.34 on this fixture (int8int8, 48-token
prompt). The paths, each against the CPU at the same quantization:
```

## TestGptOssResidentParityCUDA

Moved from `cuda/gptoss_real20b_test.go` (the comment above `TestGptOssResidentParityCUDA`) on 2026-10-09.

```text
TestGptOssResidentParityCUDA is G7's gate: ONE real gpt-oss forward on a resident path.

It had never been run on EITHER backend. docs/queue-correctness.md records the reason on each:
this card has 8 GB against a ~12 GB checkpoint, and the MacBook has 16 GB RAM against weights
that expand to 19.5 GB in memory (measured — it drove swap to exhaustion and never completed).
2224441 declared FeatAttnSink on kernel-level evidence and was correctly reverted, so the
declaration waits on this, not the other way round.

IT FITS AN 8 GB CARD VIA MACHINERY THAT ALREADY EXISTED. --moe-cache-experts holds the experts
in pinned host memory and DMAs the routed ones into device slots per token; the same path
already carries Qwen3.6-35B-A3B on this card (TestQwen36_35B_cache). gpt-oss is the smaller
problem. What was missing was not the streaming but gpt-oss's ability to use it: under caching
it indexed its per-expert bias table by SLOT id (fixed d9829ce, and TestGptOssExpertCacheAB is
the discriminating A/B — it fails by ~2.6% on the pre-fix code, which no cosine bar would have
caught).

The CPU arm is the reference. Both models load at once, which is free here (62 GB host) and is
exactly what is NOT possible on the 16 GB Mac.
```

## TestGptOssResidentParityCUDA.declaration

Moved from `cuda/gptoss_real20b_test.go` (the comment above the FeatAttnSink skip) on 2026-10-09.

```text
Skips until CUDA declares the two features, exactly as metal/gptoss_real_test.go does.
The declaration is NOT made: this gate was run on 2026-08-31 with them declared locally
and FAILED at min cosine 0.681 (see docs/queue-correctness.md G7), so declaring would be
the 2224441 mistake a second time. The test is committed so the next attempt starts from a
reproduction rather than a rebuild.
```

## TestGptOssResidentParityCUDA.assetpath

Moved from `cuda/gptoss_real20b_test.go` (the comment above `decoder.AssetPathForTest`) on 2026-10-09.

```text
decoder.AssetPathForTest, NOT modelPath: this used to call modelPath("gpt-oss-20b-MXFP4.gguf"),
which reads GOINFER_MODELS_DIR — a DIFFERENT variable from the one the comment claimed to
honour. It satisfied TestAssetRegistry_noDirectReads (a source-text regex over
os.Getenv(...) of that name, which this call never spelled) while actually bypassing
the registry's real GOINFER_GPTOSS_GGUF override entirely (audit-2026-09-02.md N-41, found
2026-09-11). AssetPathForTest resolves the SAME registry entry decoder's own
TestGptOssSafetensors_vsGGUF uses, and skips with the reason when absent.
```

## TestGptOssExpertCacheAB.slots

Moved from `cuda/gptoss_cache_ab_test.go` (the comment above `opts.MoECacheSlots = wantSlots`) on 2026-10-09.

```text
G-07: topK+1 = 3, not 2. The fixture is nE=4/topK=2, and a request of 2 was NOT
honoured — `req > topK` was false, so cacheSlots stayed at min(8·topK, nE) = 4,
i.e. one permanent slot per expert. This gate's whole premise is that slot ≠
expert id, and it was getting the identity mapping; it discriminated on the
2026-08-31 run by routing luck. 3 is honoured, is below nE=4, and forces at
least one eviction.
```

## TestGptOssExpertCacheAB.effective

Moved from `cuda/gptoss_cache_ab_test.go` (the comment above `if cache {`) on 2026-10-09.

```text
G-07: ASSERT THE EFFECTIVE SLOT COUNT. Nothing did, and the request was being
silently floored — with topK=2 the old `req > topK` was false for a request of 2, so
cacheSlots stayed at min(8·topK, nE) = 4 = nE: one permanent slot per expert, and
slot ≠ expert only by first-admit order. This gate's premise is that the two index
spaces diverge, so the premise has to be checked rather than requested.
```

## TestCUDA_graphReplayBound

Moved from `cuda/graph_bound_test.go` (the comment above `TestCUDA_graphReplayBound`) on 2026-10-09.

```text
TestCUDA_graphReplayBound is the fail-fast in front of the CUDA-graphs forward restructure
(Step 2). launch_cost bounded a SINGLE live launch at ~10 µs (FFI/purego-bound, grid-independent).
It did NOT measure graph replay — and the whole lever rests on one unproven claim: that replaying a
captured K-kernel segment collapses K host crossings into ~one, rather than still paying per-node
GPU-side dispatch K times. If replay ≈ live at real segment size, the restructure buys nothing and
we bank a negative BEFORE the invasive launchToken surgery.
```

## TestCUDA_graphLiveNoSyncOrdering

Moved from `cuda/graph_nosync_test.go` (the comment above `TestCUDA_graphLiveNoSyncOrdering`) on 2026-10-09.

```text
TestCUDA_graphLiveNoSyncOrdering is the VALID inter-operation ordering test — no sync between the
interleaved ops, which is the regime the full forward runs in and the one my earlier volume tests
wrongly serialized away (a trailing per-iteration Sync hides an inter-op race exactly like
CUDA_LAUNCH_BLOCKING does; the per-layer-drain probe made the forward divergence vanish, proving the
race is inter-operation and sync-maskable).
```

## TestGraphsDecodeSpeedup

Moved from `cuda/graphs_speed_test.go` (the comment above `TestGraphsDecodeSpeedup`) on 2026-10-09.

```text
TestGraphsDecodeSpeedup measures the real decode tok/s of graph replay vs live launch on a real
model, validating the ~1.4–1.7× dispatch-elimination prediction end-to-end through the safe-gate.
Graphs are enabled via the UNSAFE override (this idle box is DEFAULT compute mode; with no churn,
replay is bit-exact — the self-test in admitGraphs confirms it). Heavy; gated.
```

## TestGraphsDecode26B

Moved from `cuda/graphs_speed_test.go` (the comment above `TestGraphsDecode26B`) on 2026-10-09.

```text
TestGraphsDecode26B is the one measurement CUDA graphs were owed before deciding their fate
(docs/cuda-graphs-investigation.md: ~1.01x on the dense 1.5B; the 26B MoE was never measured). On
the C′ path graphs are not free to turn on: they force the DMA overlap off (a captured segment
cannot wait per miss) and block compute-time LoRA. So the comparison is the trade itself —
today's default (graphs off, overlap on) against graphs on (overlap off) — not graphs vs a
strawman. Greedy decode feeds each argmax back, so MoE routing and the expert cache see a real
continuation. Arms are separate loads (two 26B residents do not fit 8 GB), interleaved ABBA.

Pre-registered decision rule (2026-09-24, before running): graphs >= 1.05x default → worth
keeping; <= 1.00x → remove graphs; between → ambiguous, back to the owner.
```

## needsFreshProcess

Moved from `cuda/isolated_marker_test.go` (the comment above `needsFreshProcess`) on 2026-10-09.

```text
WHY IT EXISTS (measured 2026-09-28). The GPU gate's heavy tier runs the whole package in one process.
Seven real-model tests failed there with a resident decline or CUDA_ERROR_OUT_OF_MEMORY, and passed on
the same tree (and on clean main) when run as their own process: 7/7, 861 s. The margins are small —
TestPrefillLongPrompt's 7B needed 0.94 GB of KV + 384 MB reserve against 1.29 GB free, 30 MB short.
This is NOT a leak: A12 (docs/QUEUE.md) measured and refuted both a leak and parallelism, and the
in-process shortfall is still unexplained. The marker does not explain it either; it keeps an
unexplained tens-of-MB drift from turning a correct test red.
```

## TestHiddenLastResidentParityCUDA

Moved from `cuda/hiddenlast_resident_parity_test.go` (the comment above `TestHiddenLastResidentParityCUDA`) on 2026-10-09.

```text
TestHiddenLastResidentParityCUDA is M-10's own gate (docs/audit-2026-09-10.md) — the CUDA twin of
metal/hiddenlast_resident_parity_test.go, which the finding named as missing ("Only Metal has a
hiddenlast_resident_parity_test.go"). Same shape: resident HiddenLast (cudaResident.HiddenLast /
prefillChunked's tailHiddenLast) vs CPU HiddenLast, same int8int8 weights, called directly via
ResidentForwardForTest (bypassing decoder.Model.HiddenLast's own resBusy/fallback dispatch,
already covered by the fake-backend seam tests) so this isolates the KERNEL correctness question.

This is also forceExactKernels' own integration proof (M-09/M-10/M-11): before that fix,
prefillCore's tailHiddenLast pass could silently engage useAttnFused/useGemmMMA whenever M/K/
position crossed their shape thresholds — this fixture's own M may or may not cross them, so a
failing run here without the fix would only be a coincidence; the real proof that forceExactKernels
is wired correctly is decoder/spec_verify_guard_test.go's pure unit coverage plus this test
passing at whatever cosine the real kernels produce, unaffected by prompt length.
```

## TestResidentKVBytes_matchesCUDAAllocation

Moved from `cuda/kvbytes_agreement_test.go` (the comment above `TestResidentKVBytes_matchesCUDAAllocation`) on 2026-10-09.

```text
Before the "cuda" branch, Plan's per-position formula priced MLA at twice the allocation and a requested
f16 / i8 at a half / ~0.28 of it (docs/measurements/memory-accounting-cuda-2026-09-25.md); deepseek-tiny
and the f16 / i8 rows fail without the branch. Three links are checked, so a drift anywhere shows:
the buffers' bytes == kvBytesForCap (the resident's own fit figure) == ResidentKVBytes("cuda"), and each
buffer's driver allocation stays within one allocQuantumBytes (2 MiB) of its bytes.
```

## TestCUDA_launchCost

Moved from `cuda/launch_cost_test.go` (the comment above `TestCUDA_launchCost`) on 2026-10-09.

```text
TestCUDA_launchCost bounds the per-launch host cost — Step 0 for the dispatch-overhead lever.
The refined 26B decomposition put ~19 ms/token across ~600 launches ≈ 32 µs/launch, several times
a normal CUDA launch (~5 µs). That points at the purego FFI crossing (cgo-free: every
cuLaunchKernel is a dlopen'd-symbol call + Go-side arg packing), NOT GPU-side dispatch. If
confirmed, CUDA GRAPHS (capture once, replay in one call — collapses N crossings into one) are the
tool, not batching (which only cuts dispatch COUNT). The split: launch a cheap kernel at the real
grid vs a minimal 1×1 grid — if per-launch is ~equal and both ~30 µs, the cost is launch-bound
(FFI + packing), not compute-bound, and graphs win.

Model-independent (the FFI cost is the same for any model), so it runs on the tiny fixture.
```

## TestResidentCloseFreesVRAM

Moved from `cuda/lifecycle_test.go` (the comment above `TestResidentCloseFreesVRAM`) on 2026-10-09.

```text
TestResidentCloseFreesVRAM is the lifecycle gate.

WHY THIS EXISTS. cudaResident.Close() used to free the page-locked HOST buffer and close the
executor channel — and nothing else. It never freed a single DEVICE allocation and never
released the context, so every decoder.Load(Backend:"cuda") + Close() leaked the ENTIRE model
(weights + per-layer KV cache — gigabytes on a real checkpoint) until the process exited
(d8e81cb).

It hid because it is invisible in a one-model run. It only bites a model zoo, an
/admin/models/unload, or a test binary that loads several models in sequence — and it bit all
three. It reddened the whole CUDA suite: VRAM ratcheted 421 -> 1801 -> 3077 -> 4783 -> 7733 MiB
and pinned at the 8192 ceiling, after which every Alloc/NewStream returned nil, the tests
DROPPED those errors, and the resulting zero-filled buffers surfaced as "cosine 0.000000 —
layout/unpack mismatch". An OOM wore a parity bug's clothes for long enough that two people
independently concluded "the tests just interfere; they pass individually" and moved on.

The gate is the SHAPE of memory across load/close cycles, which is the signal that actually
found it: a sawtooth means Close frees; a staircase means it leaks. Peak alone proves nothing,
and memory measured AFTER the process is worthless — it always looks clean, because the
process exited.
```

## TestResidentCloseFreesVRAM_7B

Moved from `cuda/lifecycle7b_test.go` (the comment above `TestResidentCloseFreesVRAM_7B`) on 2026-10-09.

```text
TestResidentCloseFreesVRAM_7B is TestResidentCloseFreesVRAM's shape at the scale that actually
reproduces A12: qwen2.5-7B and a real decode loop, rather than the 0.5B coder and one token.

WHY A SECOND GATE RATHER THAN WIDENING THE FIRST. A12 measured TestB2DenseFlagship losing
1344 MiB and TestRealForwardParity 1166 MiB, each alone in its own process, each already
deferring Close(). The existing gate is green throughout — accurately, for what it covers. It is
not tautological and it is not exercised-but-never-triggered: it is CORRECTLY SCOPED AND SILENTLY
NARROW, which is the variant that looks most like a working gate. Keeping both makes the scopes
visible side by side instead of hiding one inside the other.

WHAT THIS DISTINGUISHES, pre-registered before the run (A12):

	loss on cycle 1, ~zero on 2 and 3   -> CONTEXT-level retention, almost certainly the
	                                       local-memory backing store A9/A10 measured. NOT a leak:
	                                       a one-time cost per context per kernel set. Close tears
	                                       down a MODEL; it does not destroy the CONTEXT, and every
	                                       kernel a 7B decode touches that a 0.5B one-token forward
	                                       does not will have reserved backing store no model-level
	                                       Close can return.
	loss repeating every cycle          -> a genuine leak; hunt for what Close does not release.
	loss shrinking but not vanishing    -> both, and the components separate before either is fixed.

The differing magnitudes (-1344 vs -1166 from the same 7310 MiB start) already favour the first:
a fixed per-model leak would repeat a fixed size, whereas different kernel sets reserving
different backing stores would not.
```

## TestMinistral3ResidentParityCUDA

Moved from `cuda/ministral3_resident_parity_test.go` (the comment above `TestMinistral3ResidentParityCUDA`) on 2026-10-09.

```text
HONEST LIMIT, checked directly (not assumed): unlike G5 row 1's smollm3-tiny — where a whole
wrong-vs-right control experiment landed within noise on Metal — this cosine floor does NOT
discriminate the fix from a disabled one: with the real fix and with it force-disabled
(qTempScale always 1), the worst cosine over 32 tokens against the CPU reference was 0.999923
and 0.999916 respectively. That is NOT noise-dominance (CUDA's int4 path is far tighter than
Metal's int8 — see below), it is EFFECT SIZE: probing the two configurations' own resident
logits directly (bypassing the CPU reference entirely) shows positions 0-7 (floor=0, scale
exactly 1 either way) bit-identical between configs as expected, and positions 8+ (floor>0,
scale ~1.06-1.28) genuinely differing between them — small, real, reproducible, just too small
relative to this tiny/seeded model's own output variance for a whole-model cosine floor to
isolate. This test still proves the feature doesn't CORRUPT anything (32/32 exact argmax, no
NaN, no drift) — the feature-specific correctness proof is
decoder.TestAttnTempScale_matchesSequentialFormula (the exact formula, no GPU, no quantization
noise) plus the reviewed kernel math (gemv_fwd.cu/prefill_batched.cu's own comments).
```

## TestMellumResidentParityCUDA

Moved from `cuda/mellum_real_test.go` (the comment above `TestMellumResidentParityCUDA`) on 2026-10-09.

```text
TestMellumResidentParityCUDA is the CUDA counterpart of metal/mellum_real_test.go, and it
exists for a reason that is worth stating plainly: declaring FeatRopeMscale so gpt-oss's YaRN
can work ALSO admits Mellum, because mellumArchitecture requires exactly {FeatMoE,
FeatPerLayerRoPE, FeatQKNorm, FeatRopeMscale, FeatSlidingWindow} and CUDA already declared the
other four. One flag is the entire admission.

Metal hit the same coupling (G10) and resolved it by an explicit owner call, because no Mellum
checkpoint was reachable on that machine — waiting was not an available option. On this box a
real 4-layer weight slice IS present, so the choice there is a measurement here.
```

## TestMLAResidentParityCUDA.fullsequence

Moved from `cuda/mla_resident_test.go` (the comment above `inTop2(lgC, gTok)`) on 2026-10-09.

```text
The FULL sequence, not just the first token: found live reviewing this test — a
deliberately-mutated nGroup/topkGroup transposition in the router launch
(cuda/resident.go, the trap decoder/features.go's FeatMLA entry names) left the
per-position forward-logit cosine check above untouched (worst cosine unchanged)
AND left the first generated token matching, so a first-token-only assertion
passed clean while later tokens had already diverged in this same run's own
logged output. Discrete expert selection can stay identical for several tokens
after a wrong-but-plausible routing decision and only visibly diverge once a
different expert combination is actually selected — checking one prefix position
is not enough for a selection bug the same way it would be for a smooth numerical
one. So every divergence is still a HARD stop, UNLESS it passes the near-tie check
immediately below.

NEAR-TIE, NOT A DEFECT (measured 2026-09-28, aikit v1.50.0's binary16 int4 group
scales): on this synthetic/random-weight fixture, gen step 5 diverged (cpu=51,
cuda=87) with cpu's own top1/top2 gap at 0.000746 and cuda's at 0.002231 — an
order of magnitude tighter than every non-diverging step (0.008-0.046) — and each
side's runner-up IS the other side's winner: token 51 is cuda's #2, token 87 is
cpu's #2. That is the MoE router-flip noise floor this repo already has a memory
for (a bit-identical router flips top-k under ~0.5% input noise): the f16-scale
switch changed ONLY CPU's rounding (CUDA's own numerics were gated
byte-identical old-vs-new by 2669bf11's own pre-registered gate 3), so an
already-near-tied greedy pick on a random-weight fixture is now decided by
backend-implementation noise, not by a wrong computation. A real routing defect
(the transposition class above) produces a CONFIDENT wrong pick, not a swap
between each side's own top-2 — so the check below still catches that class: it
requires MUTUAL containment (each side's pick is in the OTHER side's own top-2),
which a random unrelated token from a real bug would not satisfy.
```

## TestMLALatentReuse_prototype

Moved from `cuda/mla_latent_reuse_prototype_test.go` (the comment above `TestMLALatentReuse_prototype`) on 2026-10-09.

```text
TestMLALatentReuse_prototype is micro-leg B of docs/completed/task-mla-cuda-residency.md: the experiment
that decides that task's 3-days-vs-3-weeks fork, and the skeleton of its eventual parity gate.
```

## TestMoEResidentParity.floor

Moved from `cuda/moe_parity_test.go` (the comment above `minCos < 0.995`) on 2026-10-09.

```text
The 3% near-tie rule above is NECESSARY BUT NOT SUFFICIENT on this fixture, and the floor
below is what closes the gap. Both are calibrated by breaking the dispatch on purpose and
measuring — a green parity test proves nothing until it has been seen to go red.

RE-MEASURED 2026-08-06 (RTX 2070 SUPER) after audit C-15 made cuda's f32tof16 match the
canonical cross-backend f16 scale representation (decoder.f32ToF16bits). The OLD table below
had correct=0.999906 and bug B=0.997687 — but that separation was ENTIRELY an artifact of the
old TRUNCATING f32tof16, which happened to resolve one int4 near-tie token the CPU's way. With
the correct scales each control applied alone measures:

  dispatch state                            exact   worst gap   min cosine   3% rule   floor(0.995)
  ---------------------------------------  ------  ----------  -----------  --------  ------------
  correct                                   11/12      0.032%     0.997833   pass      pass
  A down-proj slot pinned to 0              11/12      0.959%     0.988509   PASSES ✗  FAILS ✓
  B glu_quant gOff/uOff swapped             11/12      0.032%     0.997846   PASSES ✗  PASSES ✗  ← see below
  C gate/up GEMV slot pinned to 0           10/12      6.124%     0.989101   fails     (fails)
  D router fed the raw residual              — (>3% argmax gap, structural — caught by the rule)
  E down-proj stack mis-strided              — (79% gap / NaN — caught by the rule + NaN guard)

The correct run is no longer 12/12: one token is a genuine int4-KERNEL near-tie (device W8A8
GEMV vs CPU int4, 0.032% argmax gap, benign), so the correct cosine is 0.997833, not ~1.0. The
floor is therefore 0.995 — below correct (0.997833) and above bug A (0.988509), which is the
only STRUCTURAL bug that survives the 3% argmax rule (C/D/E have >3% gaps). ~2.5e-3 margin on
the correct side, ~6.5e-3 on the bug side.

Bug B (gate/up swap) can NO LONGER be caught here: correct (0.997833) and bug B (0.997846) are
indistinguishable, because on RANDOM-weight experts silu(up)*gate ≈ silu(gate)*up in magnitude
and the difference washes out through down-proj + combine + argmax. It is instead caught by
TestMoeSwigluWiring_C15, which exercises launchGluSplit (the sole gate/up-split dispatch) with
crafted gate≠up and asserts the pre-quant SwiGLU output directly — scale- and fixture-independent.
```

## mustAlloc

Moved from `cuda/mustalloc_test.go` (the comment above `mustAlloc`) on 2026-10-09.

```text
mustAlloc allocates device memory and FAILS THE TEST on error, instead of the
`buf, _ := gc.Alloc[T](...)` that used to be the norm here.

WHY THIS MATTERS MORE THAN IT LOOKS. A dropped alloc error is how an out-of-memory
condition disguises itself as a numerics bug. gc.Alloc returns (nil, err) when the card
is full; drop the err and the nil buffer reads back as ZEROS, so the assertion that
fires is "cosine 0.000000 — layout/unpack mismatch". That sentence sent two people
hunting a kernel bug for a day while the real cause was a VRAM leak saturating an 8 GB
card mid-suite (d8e81cb). The kernels were never wrong; the memory was gone, and every
test lied about why.

The rule this encodes: a RESOURCE failure must say it is a resource failure. Tests may
legitimately skip when a GPU is absent or too small — but they must never silently
compute on nothing and report the result as a correctness verdict.
```

## TestOlmo3ResidentSmokeCUDA

Moved from `cuda/olmo3_resident_smoke_test.go` (the comment above `TestOlmo3ResidentSmokeCUDA`) on 2026-10-09.

```text
Olmo Hybrid additionally exercises a REAL bug found while bringing this up: both backends
assumed every qwen35Params-carrying family's full-attention layer used qwen3.5's own
double-width q-gate scheme (Qwen35ResidentParams hardcoded attnGate=true) — Olmo Hybrid's is
plain (olmo3's own scheme), fixed via a new Architecture.qwen35.AttnGate field. This smoke test
going from "BuildResident declined: qwen35 softmax layer has empty q_norm/k_norm" to passing is
the regression gate for that fix.
```

## TestMoEStreamingDecodeProfile

Moved from `cuda/moe_streaming_decode_profile_test.go` (the comment above `TestMoEStreamingDecodeProfile`) on 2026-10-09.

```text
TestMoEStreamingDecodeProfile is the "measure before building" step for docs/completed/task-moe-streaming.md's
two open CUDA items (gocudrv async-H2D overlap of C′ miss DMAs; P20 redirected toward the expert-DMA
cost). Both turned out to be already-closed elsewhere this session: item 1 by
docs/completed/aikit-subrange-async-upload.md (2026-08-28 — declined as scoped, superseded by the
already-shipped gpu.UploadBatch, which IS what cuda/resident.go's C′ path uses today, +9.3% tok/s
measured on the 35B), item 2 by this session's own P20 expert-major build
(docs/measurements/p20-expert-major-m26-2026-09-21.md, 2.26-2.66x on the real M26 PREFILL). Neither
touches DECODE (M=1, no rows to bucket by expert), so this gets a FRESH, current-code reading of the
decode-side C′ DMA share specifically, using GOINFER_MOE_CACHE_PROF's existing stall/host/dma split,
to check whether the "genuine H2D/compute overlap" condition that decision doc's own §6 "Revisit if"
names has now fired. Synthetic embeddings (EmbedResidentForTest), not a real prompt — a DMA-timing
profile does not need real tokens, only real routing (the model's own trained router still decides
which experts, from an in-vocab embedding).
```

## BenchmarkOptFwd

Moved from `cuda/optfwd_bench_test.go` (the comment above `BenchmarkOptFwd`) on 2026-10-09.

```text
BenchmarkOptFwd measures real end-to-end decode with and without the feature at three
temperatures, on the same prompt and seed — the whole-generation check, not optFwdStep's isolated
cost. Ported from metal/optfwd_bench_test.go so the two backends' numbers are comparable.

The Metal run (qwen2.5-coder-0.5b, two independent runs) reported ~8-15% faster at T=0.2/T=0.7 and
~6-7% SLOWER at T=1.0 — the last being bounded gate-warmup cost, expected rather than a bug, since
the gate must observe some misses before it can turn itself off.
```

## TestP20ExpertLocality

Moved from `cuda/p20_expert_locality_test.go` (the comment above `TestP20ExpertLocality`) on 2026-10-09.

```text
TestP20ExpertLocality is the "measure the split before building" step docs/queue-performance.md's
P20 entry asks for, before any expert-major kernel is written: expert-major batching only pays off
if a whole M-row prefill chunk touches FEW ENOUGH distinct experts per layer to stage them all on
the device at once — if it touches MORE than the box can hold, "fetch each distinct expert once per
chunk" is not achievable at that chunk width regardless of kernel design, and the item needs a
smaller batch width (or is dead on this box), not a kernel.
```

## TestPagerDeterminism

Moved from `cuda/pager_determinism_test.go` (the comment above `package cuda`) on 2026-10-09.

```text
Does repeated greedy generation on a PAGED MoE return the same tokens every time?

FOUND WHILE MEASURING SOMETHING ELSE. In the spec-x-pager run the off arm produced 64 tokens on
the first repeat of a prompt, 1 on the second and 0 on the third — same prompt, temperature 0,
same process, `Generate` returning a nil error each time, while the two neighbouring prompts gave
64/64/64 around it. Greedy decode is deterministic by construction, so identical inputs returning
different outputs means state is carrying between generations.

THE HYPOTHESIS THIS TEST EXISTS TO CHECK, and the reason it belongs next to the pager rather than
in a general decode test: the ONE piece of state deliberately kept across generations here is the
C′ expert slot cache. Everything else is reset or positional — Forward and ForwardNoLogits both
call Reset() at pos 0 (resident.go), so a Gated-DeltaNet's conv ring and matrix state are
re-zeroed per sequence. The LRU is not, by design: it is a cache, and C′ documents itself as
BIT-IDENTICAL to the fully-resident path. If that identity holds, generation N and generation 1
must emit the same token ids no matter what the cache happens to hold. If they do not, a slot is
being read while holding an expert other than the one the router asked for, and "bit-identical"
is false in the configuration that only appears after the cache has been warmed by a previous
generation — invisible to every single-generation test, which is what the existing 26B/35B cache
tests are.

It reports rather than diagnoses. Two outcomes are worth separating and both are recorded: ids
that diverge at some position (a wrong-weights read) and ids that are a strict PREFIX of the
first run (an early stop), because they point at different faults.
```

## TestPagerDeterminism.reuse

Moved from `cuda/pager_determinism_test.go` (the comment above the reuse-on/reuse-off loop) on 2026-10-09.

```text
A/B ON THE ONE SUSPECT, because a repro that only shows the symptom cannot name the cause.
decoder/resident_reuse.go (commit 3358e6ba, today) added prefix reuse on the resident KV and
gates it on GOINFER_NO_RESIDENT_REUSE alone — there is no recurrent-state exclusion in
residentReuseLen. For a Gated-DeltaNet family that is the exact hazard the rest of the tree
already refuses: the conv ring and matrix state are NOT position-truncatable
(decoder/deltanet.go: "why qwen3_5_moe falls back from prefix reuse / speculative"), and
cudaResident.Forward re-zeroes them only at pos == 0 — which a reused prefix never reaches.
So a second generation would decode from the PREVIOUS generation's tail state.

If disabling reuse makes the symptom vanish, that is the cause. If it survives, the reuse
path is exonerated and the hunt moves to the pager, which is why both arms run here rather
than just the one that confirms the guess.
```

## TestPagerDeterminism.stagecount

Moved from `cuda/pager_determinism_test.go` (the comment above `r.CacheStatsForTest()`) on 2026-10-09.

```text
STAGE COUNT IS THE DIAGNOSTIC, not decoration. One staging event happens per routed
MoE layer per forward POSITION, so stages/layers is the number of positions the run
actually pushed through the model — prompt prefill included. In the run that raised
this, the first repeat of a prompt showed 88 positions (24 prompt + 64 generated) and
the second showed 65, i.e. the prompt was not prefilled the second time. Whether that
is KV being reused across a supposedly stateless Generate, or the prefill being
skipped for another reason, the position count distinguishes it from a sampling or
stop-token explanation, which would leave prefill untouched.
```

## TestPairwiseRoPEResidentParityCUDA

Moved from `cuda/pairwise_rope_resident_parity_test.go` (the comment above the file header) on 2026-10-09.

```text
WHY THIS GATE EXISTS. TestCohereResidentParityCUDA (cohere_resident_parity_test.go) loads the
committed cohere-tiny / cohere2-tiny, whose weights are ~0.02 std: attention is nearly UNIFORM
there, a softmax over near-equal scores is blind to which key is which, and so a WRONG ROTATION
(the NeoX half-split kernels run on a GPT-J pairwise family) leaves the logits at cosine 0.9997.
The real checkpoints were not blind: Command-R7B and Aya-expanse-8B at int4 on this very resident,
per-position resident-vs-CPU worst cosine -0.075 / -0.041 (docs/measurements/cuda-pairwise-rope-2026-10-01.md).

So this gate PEAKS the attention: it derives, at test time and in a temp dir, a checkpoint from the
committed fixture with every 2-D weight scaled by peakedScale (0.02 -> ~0.25 std, the factor the
owner's session measured: cosine 0.06 with NeoX kernels against 0.9997 flat). Deterministic, no new
binary in the tree, and the same bytes the CPU path loads.
```

## TestParityFloorControl.readings

Moved from `cuda/parity_floor_control_test.go` (the comment above the model loop) on 2026-10-09.

```text
Measured 2026-08-31 on the RTX 2070 SUPER, all via this harness:

  qwen2.5-coder-0.5b   24 layers  dense          0.973926
  qwen2.5-coder-1.5b   28 layers  dense          0.993496
  qwen3.6-35b-a3b      40 layers  MoE+streaming  0.982171
  gpt-oss-20b          24 layers  MoE+streaming  0.895287   <- the outlier

The 35B row is the one that matters: same path, same card, sparse, streamed, and DEEPER,
yet 0.982. That is what makes gpt-oss's 0.895 a defect rather than this path's floor.
(The 35B is not run here — it needs ~44 GB of host RAM across both arms and ~6 min. Add it
back with MoECacheExperts:true and m.NewCache, not decoder.NewKVCache: it is a DeltaNet
hybrid whose recurrent state the plain constructor does not allocate.)
```

## TestPipelineLint_boundKernelsAreLaunched

Moved from `cuda/pipeline_lint_test.go` (the comment above `TestPipelineLint_boundKernelsAreLaunched`) on 2026-10-09.

```text
TestPipelineLint_boundKernelsAreLaunched closes the FIFTH state a kernel can be in.

The launch-site census sorts kernels into: launched by production and covered by an asserting
gate; launched by production and covered by nothing; launched only from tests; and embedded but
never launched. `gemv_w4a8_batched` was in none of them. It was BOUND into a production pipeline
field at every model load — paying NVRTC JIT time — and launched by nothing, anywhere, while
carrying a parity test AND a bandwidth benchmark that made it look like the shipping batched
int4 kernel. It is not: bGemvB dispatches int4 to `gemv_w4a8_rn` unconditionally
(prefill.go, bGemvB), so the kernel named for the feature was not the one the feature used.

A runtime launch trace CANNOT find this class — it sees no launch for a bound-and-dead field
and no launch for a field that was never bound, and those are the same observation with
opposite causes. Only source can tell them apart, so this is a static lint, in the same family
as TestKernelFMALint_coversEmbeddedPTX: fix the class, not the instance.
```

## TestPrefillCancel

Moved from `cuda/prefill_cancel_test.go` (the comment above `TestPrefillCancel`) on 2026-10-09.

```text
The sequential fallback checks ctx.Err() PER TOKEN — G18 put it there because "an abandoned
client leaves the whole prompt streaming through the device". A batched pass has no such loop, so
before Prefiller carried a context the granularity was the whole pass: measured ~22 s for one
512-row MoE chunk on M26, against the ~46 ms the per-token path it replaced would have taken to
notice. That is the regression under test.
```

## TestPrefillCancel_dense.warm

Moved from `cuda/prefill_cancel_test.go` (the comment above `prefill := func`) on 2026-10-09.

```text
The baseline is a WARM run. The first prefill on a fresh model pays one-off costs (kernel JIT, buffer
first-touch), and a baseline taken from it overstates the steady-state cost several times over: measured, one
run read 1.185 s cold and the cancelled run then finished the whole prefill in 162 ms, before the cancel
scheduled at a fifth of the cold figure (237 ms) could fire. That read as "cancel ignored" and failed the test
without any cancellation being wrong. So: one untimed warm-up, then the fastest of three timed runs.
```

## TestPrefillChunked_fastKernelsOnEveryChunk

Moved from `cuda/prefill_chunk_fast_test.go` (the comment above `TestPrefillChunked_fastKernelsOnEveryChunk`) on 2026-10-09.

```text
TestPrefillChunked_fastKernelsOnEveryChunk pins the fix for the chunk demotion (docs/measurements/prefill-chunk-demotion-2026-09-21.md): a prompt longer than one chunk (512 rows) is
prefilled in several passes, and BEFORE the fix every pass but the last ran tailKVOnly, which forceExactKernels (tail != tailLastLogits) sent to the slow exact GEMM and attention — 7 of 8
chunks at K=3900. It counts attn_fused / gemm_w4a8_mma launches directly instead of inferring them from timing:
```

## TestCUDADeltaNetPrefill_matchesPerToken.postonly

Moved from `cuda/prefill_deltanet_test.go` (the comment above the olmo_hybrid-tiny case) on 2026-10-09.

```text
Post-only norm placement (Olmo 3's): the DeltaNet layers' FFN must still take the pre-MLP norm, as decode's
segBFFN does. The first cut applied the model-level placement there and read cosine 0.888 at 24 rows.
```

## TestCUDADeltaNetPrefill_matchesPerToken.moe

Moved from `cuda/prefill_deltanet_test.go` (the comment above the qwen3_5_moe-tiny case) on 2026-10-09.

```text
The MoE sibling (item 9, task-cuda-deltanet-prefill-2026-09.md): DeltaNet mixer + sparse FFN +
sigmoid-gated shared expert in the SAME layer, unlike the three above. Resident-eligible as-is
(moeInter 64, hidden 64 — both already multiples of 32); qwen3next-tiny was not, until its own
fixture was regenerated at moeInter/sharedInter 32 (scripts/pin_qwen3next_tiny.py) to close the
same gap for the qwen3_next config-shape path specifically.
```

## TestPrefillLast_gemma3

Moved from `cuda/prefill_gemma3_test.go` (the comment above `TestPrefillLast_gemma3`) on 2026-10-09.

```text
TestPrefillLast_gemma3 extends the batched-prefill bit-identity gate to the SANDWICH-NORM family.
After the qk-norm guard was lifted, gemma3's only remaining decline was its 4-norm sandwich: the
attention and MLP sublayer outputs are RMSNorm'd BEFORE the residual add (postAttnNorm/postMLPNorm).
batched prefill now does that per row (o-proj/down → temp, rmsnorm_f32_batched, residual add),
mirroring segB's decode path. Real Gemma-3-4B at int4 (kEqV=0, no attn-softcap): asserts KV
bit-identical (all layers × rows), last-token logits bit-identical, 64-token decode byte-identical.
Heavy; gated. Green ⇒ gemma3 is a validated batched-prefill family.
```

## TestCUDA_graphReplayBound.budget

Moved from `cuda/graph_bound_test.go` (the projection comment above `segsPerToken`) on 2026-10-09.

```text
share of the ~59 ms/token budget this lever can actually reclaim.
```

## TestPrefillLongPrompt

Moved from `cuda/prefill_longprompt_test.go` (the comment above `TestPrefillLongPrompt`) on 2026-10-09.

```text
TestPrefillLongPrompt probes the batched prefill path at the prompt lengths a deep-context
benchmark actually uses (up to 8k), which no existing harness covers: TestPrefillTTFT stops at
M=2048 on a 1.5B model. The M-sized device scratch prefillCore allocates is O(M*inter), so the
path can pass its LOAD-time report (PrefillPath says "batched") and still decline every real
long prompt at call time, silently falling back to the ~6 ms/token sequential loop.

Reports, per M: the static decline (if any), whether the call succeeded, its duration, and the
per-token cost. Diagnostic — it asserts only that the model loaded and that the static gate is
open; the numbers are the output.
```

## TestPrefillMoE_bitIdentical

Moved from `cuda/prefill_moe_test.go` (the comment above `TestPrefillMoE_bitIdentical`) on 2026-10-09.

```text
TestPrefillMoE_bitIdentical gates the third P20 blocker: a MoE layer's FFN now runs row by row off
the BATCHED residual (xB.At(m*hidden*4)) instead of taking the whole model off the batched path.
The attention half is batched; the routed experts keep decode's exact per-token sequence, so the
only thing that may change is speed.

"May change" is the claim, so the assertion is equality against the sequential per-token path on
the same resident at the same positions — every logit, not a tolerance. A MoE model is the WORST
case for a tolerance-based check: routing is a discrete argmax over router logits, so a tiny
numerical difference does not perturb the output slightly, it runs a DIFFERENT EXPERT and the row
is unrelated. A near-match here would mean the routing agreed by luck on this input.

Both fixtures are exercised because they cover different halves: gemma4-moe-scaled carries the
real 26B FFN shapes (hidden 2816, moe_inter 704) and gemma4-moe-kv-tiny puts K=V and MoE in the
same model, which is the combination M26 actually is.
```

## TestPrefillMoE_real26B

Moved from `cuda/prefill_moe_test.go` (the comment above `TestPrefillMoE_real26B`) on 2026-10-09.

```text
TestPrefillMoE_real26B is the same assertion against the model this was built for — M26, the
Gemma-4-26B-A4B kind-4 .giw bundle, loaded with -moe-cache-experts exactly as scripts/bench_peer.py
launches it.

It is NOT redundant with the fixture gate above, and the difference is the point. No fixture here
carries K=V and MoE in the SAME model: gemma4-moe-kv-tiny is the one that would, and it declines
residency ("moeInter(16) and hidden(64) both multiples of 32"), so gemma4-moe-scaled covers MoE
with uniform non-K=V geometry and gemma4-dense-scaled covers K=V without MoE. M26 is the only
checkpoint that exercises both at once — and it is also the only one that exercises the C′ routed
expert DMA inside the per-row loop, which the fixtures run with cacheExperts=false.
```

## TestPrefillMoE_real26B.load

Moved from `cuda/prefill_moe_test.go` (the comment above `TestPrefillMoE_real26B.load`) on 2026-10-09.

```text
Heavy: the load alone is ~2m11s (pinned host allocation for the expert stack).
```

## prefillMoEParity.nonvacuity

Moved from `cuda/prefill_moe_test.go` (the comment above `prefillMoEParity.nonvacuity`) on 2026-10-09.

```text
Non-vacuity: this gate is about MoE layers taking the batched pass, so the fixture must HAVE
them. Without this the test would pass on a dense model by exercising the branch it is not
about — the failure mode the expert-major gate needed a run counter to rule out.
```

## TestPrefillNonUniform_bitIdentical

Moved from `cuda/prefill_nonuniform_test.go` (the comment above `TestPrefillNonUniform_bitIdentical`) on 2026-10-09.

```text
TestPrefillNonUniform_bitIdentical gates the two guards batched prefill dropped so the Gemma-4
families could reach it at all: PER-LAYER geometry and K=V.

The fixture is the scaled dense Gemma 4 (hidden 1024, 12 layers, 5:1 sliding/full, head dim 256
on the local layers and 512 on the global ones, K=V on the globals) — the only checkpoint here
that exercises BOTH at once, and small enough to run in seconds. Before this change
prefillStaticDecline refused it twice over: "non-uniform layer geometry at N" and "K=V layer at
N". The refusals were correct for the code as it stood, because prefillCore hoisted layer 0's
dims into every launch; each launch now binds its own layer's, and the M-sized scratch is
allocated at the max across layers.

The assertion is the batched pass against the SEQUENTIAL per-token path on the same resident, at
the same positions: bit-identical last-token logits. Not "close" — every batched kernel is the
M=1 kernel with an M dimension, so any difference is a striding or ordering bug, and a
near-match is exactly what a wrong-but-plausible stride produces on a K=V layer whose V happens
to correlate with K.
```

## TestPrefillStartOffset

Moved from `cuda/prefill_startoffset_test.go` (the comment above `TestPrefillStartOffset`) on 2026-10-09.

```text
Measured 2026-10-02 on the 1.5B: B is bit-identical at every r from 16 to 512 (the fused kernels ARE start-offset-invariant), A differs at every r below the floor (max|d| 0.7 to 0.9 on random
embeddings) and is identical at r=512. So what a prefix reuse changes is the PROVENANCE of the reused rows (exact-kernel versus fast-kernel numerics), not the start offset. This test pins the invariance;
the A rows show the comparison can go red (it does, 151936/151936). Random embeddings: this is a numerics check, not a text one.
```

## TestPrefillTTFT

Moved from `cuda/prefill_ttft_test.go` (the comment above `TestPrefillTTFT`) on 2026-10-09.

```text
TestPrefillTTFT measures the batched PrefillLast vs the sequential ForwardNoLogits loop on a real
dense model, at the prompt lengths that bracket the Ollama crossover (128/512/2048). It is the
milestone-2 speedup number: goinfer's sequential prefill reads every weight once per prompt token
(weight-bandwidth-bound), so its TTFT grows ~linearly; the batched path reads each weight once for
all M tokens. Heavy (loads a 1.5B model); gated on GOINFER_HEAVY_TESTS + a GPU.

WHAT THE "batched" COLUMN MEANS CHANGED ON 2026-09-05, without this file changing. CUDA fast
prefill (attn_fused + gemm_w4a8_mma) became the DEFAULT above a 512-token floor, so at K >= 512
the batched column now times the FAST path, not the exact one. That is the right thing for a
standing test — it measures what ships — but it means a number from this test taken before that
date and one taken after are not the same quantity. Set GOINFER_CUDA_FAST_PREFILL=0 to time the
exact batched path, which is what every pre-2026-09-05 row in benchmarks.md holds.

THE DEFAULTS ARE THE REGRESSION TEST AND DO NOT MOVE. The four env knobs below only widen what a
deliberate measurement run can ask for; with none of them set this test runs exactly the model,
quants and depths it always has, so its role as a standing check is unchanged. They exist because
docs/completed/task-prefill-gap.md §4 L2 sets its band on an END-TO-END cell this test could not reach —
S at K=3900 — and prices L2/L3 on D7 as a second model, while the fixed list stops at K=2048 on a
1.5B (which is exactly the blind spot prefill-chunking-d7-2026-09-04.md records: "TestPrefillTTFT,
the harness built for exactly this question, stops at M=2048 on a 1.5B model — a shape that fits,
on a model that fits").
```

## prefillpath_test.why

Moved from `cuda/prefillpath_test.go` (the comment at the top of the file) on 2026-10-09.

```text
WHY THIS EXISTS. `--backend cuda --quant int8int8` on a dense model builds a full resident decode
path — ResidentActive is true and decode runs at ~0.7× int4 — but the batched prefill GEMV is
int4-only (gemv_w4a8_batched / _rn read group-scaled int4 words), so prefillCore declines and every
prompt falls back to one forward per token. Measured on a 300-token prompt, real 0.5B, RTX 2070
SUPER: 1.73 s vs 0.19 s (9×), 4.56 vs 0.22 CPU-seconds (20×), with no compute hotspot — the CPU is
the executor spin-waiting through 300 sequential launches. The fallback is silent by design, so the
only defence is reporting it at load, and the only way that report stays true is sharing the guard
with prefillCore (prefillStaticDecline).

These tests need NO DEVICE: the guard reads struct state only.
```

## TestPrefillPath_int8Batched

Moved from `cuda/prefillpath_test.go` (the comment above `TestPrefillPath_int8Batched`) on 2026-10-09.

```text
TestPrefillPath_int8Declines is the gate for the shipped defect: int8 weights must report the
sequential path, name int4 as the requirement, and state the cost.
TestPrefillPath_int8Batched: int8 bundles now get batched prefill (§C6 — the batched W8A8 GEMV is
exact-int32, bit-identical to gemv_w8a8_fwd by construction). This was TestPrefillPath_int8Declines
before int8 batched prefill landed; it now asserts the OPPOSITE, so the 9× TTFT trap is gone.
```

## TestPrefillPath_matchesPrefillCore.deltanet

Moved from `cuda/prefillpath_test.go` (the comment above `TestPrefillPath_matchesPrefillCore.deltanet`) on 2026-10-09.

```text
A recurrent (Gated-DeltaNet) model whose layers ALSO carry valid int4 q/k/o. This is the
case the weight-kind check cannot catch: qwen3_5_moe declines today only because its
DeltaNet layers load no q/k/o, so the real guard was never exercised by any fixture. Here
the projections are present and valid, so the ONLY thing that can refuse it is the
recurrent-state check itself — remove that check and this case goes batched and wrong.
```

## TestPrefillPath_mixedInt4Int8Batches

Moved from `cuda/prefillpath_test.go` (the comment above `TestPrefillPath_mixedInt4Int8Batches`) on 2026-10-09.

```text
TestPrefillPath_mixedQuantNamesTheKind: a bundle that is int4 at layer 0 but int8 deeper (int4mix)
still declines. The message falls back to the generic form naming the layer — worse than the
int8int8 message, but it must not claim the batched path.
TestPrefillPath_mixedInt4Int8Batches: an int4mix-style bundle (int4 in most projections, int8 in
one) now gets batched prefill — dispatch is per projection, so a mix of the two batchable kinds
"falls out for free" (§C6). A genuinely non-batchable kind (native/f32) at a specific layer still
declines with the layer located.
```

## TestPrefillPath_seamGuardsAreMoEOnly

Moved from `cuda/prefillpath_test.go` (the comment above `TestPrefillPath_seamGuardsAreMoEOnly`) on 2026-10-09.

```text
TestPrefillPath_seamGuardsAreMoEOnly pins the SCOPE of the per-token debug-seam declines, which
is a different property from whether they exist.

They exist because prefill calls layerTail from exactly one site — inside the per-row MoE FFN
loop — where a per-token seam would fire M times per layer and hand its consumer M rows where it
expects one. On a DENSE model that site is never reached, so refusing there gains nothing and
costs a real feature: DFlash's block drafter arms hidCapTaps and verifies through the batched
path on a dense model.

The first version of the guard was not scoped, and it broke exactly that —
TestDFlashRoundComposition and TestDFlashCompositionResidual both failed with "per-token
hidden-state taps are armed". They are heavy, GPU-only and 40+ minutes into the suite; this
costs microseconds and fails for the same reason, which is the point of writing it down here.
```

## TestPromptHiddenAllResidentCUDA.moe

Moved from `cuda/prompthiddenall_resident_parity_test.go` (the comment above `TestPromptHiddenAllResidentCUDA.moe`) on 2026-10-09.

```text
DENSE: every row within the bar. MoE: a random-init router has near-tied top-k scores, so a few positions choose a different expert on the device
than on the CPU and ONLY those rows move (measured here: the LAST row stays at 0.9999 while a few interior rows dip to 0.993). The right statistic for
that is the mean, with the last row held to the bar and the worst reported, not asserted (the lesson of the MoE router-flip noise floor: floor the mean,
not the min over rows). A defect that moved EVERY row, or the last, still fails; one confined to a single MoE position would not, and the dense
case (same DeltaNet and attention kernels) is what pins per-position correctness.
```

## TestQwen25VLResidentReal_gate

Moved from `cuda/qwen25vl_resident_real_test.go` (the comment above `TestQwen25VLResidentReal_gate`) on 2026-10-09.

```text
TestQwen25VLResidentReal_gate is gap 0's real-checkpoint continuation of
decoder/qwen25vl_real_test.go's TestQwen25VLReal_gate, which is prefill-only by its own doc
comment ("decoding PAST an image block... is a genuinely different code path that no existing
Go test exercises yet"). This is that continuation: it drives the SAME real image through the
real vision encoder and the real CPU prefill (prefillLogitsQwenVL, via
PrefillLogitsQwenVLForTest), then compares ONE decode step computed two ways — plain CPU
(ForwardForTest) and the gap-0 hybrid (UploadKV then resident ForwardMRoPE) — by COSINE on the
raw logits, both arms fed the identical next token so neither can wander off the other's
trajectory.

WHY COSINE ON A FORCED TRAJECTORY, NOT TOKEN-STREAM IDENTITY THROUGH GenerateQwenVL. An
earlier version of this gate compared GenerateQwenVL's greedy (Temperature=0) SAMPLED token
streams, f32 CPU vs int4 CUDA resident, over 12 tokens — and found a real divergence at step 7.
Investigated before concluding anything: a throwaway probe ran plain decoder.Model.Generate
(NO vision, NO gap-0 code at all, just this repo's existing, already-shipped int4 CUDA resident
decode) on the SAME checkpoint and found divergence starting EVEN EARLIER (step 5) — proving the
effect is pre-existing f32-vs-int4 quantization noise on Qwen2.5-VL-3B's resident decode,
unrelated to anything built for gap 0. Once one token's argmax flips under quantization, every
later token in a GREEDY rollout is computed from a different context than the reference, so the
two streams necessarily diverge completely — an expected property of comparing different
precisions through autoregressive sampling, not a defect. This gate instead does what
gpu/nemotron_resident_parity_test.go's own doc comment calls "matched precision... isolates
WIRING from quant quality": one controlled step, same precision, same forced trajectory, cosine
not exact-match — the comparison that actually answers "is the hybrid decode's OWN math
correct," independent of the orthogonal question of how quantization noise compounds through
free-running greedy sampling.
```

## TestQwen35ResidentDecodeRateCUDA

Moved from `cuda/qwen35_resident_speed_test.go` (the comment above `TestQwen35ResidentDecodeRateCUDA`) on 2026-10-09.

```text
ms/LAYER is the transferable quantity and the ratio is the robust one; the absolute
per-token figure is not, because this fixture omits the released vocab (248320, whose LM
head is ~5% of the real per-token MACs) and runs a short context. Extrapolating
ms/layer × real layer count overstates the CPU side by roughly 2× against the 0.656 tok/s
actually measured on the real 27B — so quote the RATIO, and treat any absolute
extrapolation as indicative only.
```

## TestQwen36_35B_cache

Moved from `cuda/qwen35moe_35b_cache_test.go` (the comment above `TestQwen36_35B_cache`) on 2026-10-09.

```text
TestQwen36_35B_cache is the payoff run for the whole CUDA DeltaNet track: Qwen3.6-35B-A3B,
whose ~20 GB of int4 experts do NOT fit the 8 GB 2070, decoded RESIDENT via C′ expert staging
(experts in pinned host memory, the routed ones DMA'd into device slots per token).

WHY THIS MODEL AND NOT THE DENSE ONE. Qwen3.8-27B is dense: C′ streams EXPERTS, so it does
nothing for a model with none, and 15.3 GB of int4 dense weights simply do not fit. The MoE
siblings are the only members of this family an 8 GB card can host at all, so this is the one
combination where residency for this family is not merely faster but POSSIBLE.

CORRECTION (2026-08-20): an earlier version of this comment said CUDA is "the only backend with
the streaming path". That is wrong about Metal, which has its own shipped per-layer LRU expert
pager (metal/expertpool.go, built for the same gemma4-26B problem) — today wired to the g4moe
path rather than generic MoE, so it would need generalizing, but the mechanism is there. WebGPU
is the one with no equivalent.

It needed three things that did not exist a day ago: the DeltaNet mixer kernels, their wiring,
and FeatMoEGatedShared (the sigmoid-gated shared expert this family carries). Any one missing
and the model declines to the CPU path.

CORRECTNESS + INFORMATIVE LATENCY, NOT A BENCHMARK. Per token the router picks 8 of 256 experts
in each of 40 layers; at ~3.15 M int4 params per expert that is roughly 630 MB of PCIe traffic
per token before any reuse, plus a D2H routing readback per layer. The tok/s here is a floor set
by staging, and improving it is C′ step 2's LRU cache, not this test's business.
```

## TestQwen36_35B_cache.tokenizer

Moved from `cuda/qwen35moe_35b_cache_test.go` (the comment above `TestQwen36_35B_cache.tokenizer`) on 2026-10-09.

```text
THREE CONTAINERS, THREE LOADERS — and the .giw arm was missing while .giw is this test's
DEFAULT path, so the default invocation could not reach the decode it exists to measure. It
failed as `parse …int4.giw: invalid character 'G'`, i.e. tokenizer.Load reading the bundle
magic as JSON, which reads like a corrupt checkpoint rather than a missing case. The runs that
passed all set GOINFER_QWEN36_35B to the .gguf, which took the arm that existed.
```

## TestQwen35VLReal_residentImagePrefillMatchesCPU

Moved from `cuda/qwen35vl_real_resident_test.go` (the comment above `TestQwen35VLReal_residentImagePrefillMatchesCPU`) on 2026-10-09.

```text
TestQwen35VLReal_residentImagePrefillMatchesCPU is the real-checkpoint gate for the CUDA-resident image prefill of a Gated-DeltaNet
hybrid (P26b, docs/queue-performance.md): Qwen3.5-0.8B, the three G2 images (docs/measurements/p8a-qwen35-vl-2026-09), HF's own
image features fed to both sides so only the decoder differs, F32 on both arms so the only difference is where it runs.

WHAT THIS GATE IS, AND WHAT CHANGED AFTER THE FIRST TWO RUNS. It is a GROSS-ERROR gate, not a precision gate. It was first written
at int4 with a last-logits cosine bar of 0.999 and read 0.951 / 0.975 / 0.946; at f32 it read 0.973 / 0.975 / 0.956 against bars of
0.9999. Both bars were wrong, not the path: the controls (docs/measurements/p26b-cuda-hybrid-image-prefill-2026-10-06/) show that on
this 0.8B the CUDA resident sits at about 0.97 to 0.98 cosine from the CPU's f32 logits for ANY prompt, because the existing, shipped
resident TEXT prefill reads 0.978 on the same model; int4 is only 0.85 from f32; and injecting the WRONG m-RoPE layout into the
resident moved the image number from 0.9728 to 0.9699, i.e. inside that noise. So a real-checkpoint cosine cannot resolve the layout.
The layout is gated where it can be resolved: the kernel against decoder.ApplyMRoPEForTest in both modes
(TestRopeKVMRoPEBatched_interleavedModeMatchesCPUReference) and the tiny hybrid fixture end to end, where the wrong mode reads
0.99955 against 0.99995 (TestGenerateQwenVL_hybridResidentPrefillMatchesCPU).

What this real gate does catch is what the tiny one cannot: a path that is wrong on a real, deep checkpoint in a gross way (recurrent
state not built, KV garbage, a stale record reused), none of which a 0.97 cosine could hide. f32 on both arms, HF's own image features
to both sides, so only where it runs differs. Bars, amended from the controls and stated here so they are not read as pre-registered:
```

## BenchmarkRealE2EDecode

Moved from `cuda/reale2e_test.go` (the comment above `BenchmarkRealE2EDecode`) on 2026-10-09.

```text
BenchmarkRealE2EDecode is B step 4: the REAL end-to-end decode tok/s of the parity-green
forward path (TestRealForwardParity), on the real q4_k_m checkpoint. Full per-token
work — mixed int4/int8 GEMVs + RoPE + GQA attention + requant glue + on-device argmax +
per-token sync — driven autoregressively at real advancing positions, through a
LockOSThread-pinned CUDA executor fed by a channel (one round-trip per token), so the
thread-safety executor cost is IN the number (guardrail #3). Wall-clock steady-state
(the number a user feels), vs same-box pinned Ollama 149 / WebGPU 111.6. cgo-free.
NOT A CORRECTNESS GATE, despite the name it used to carry. This drives a HAND-ROLLED sequence of
kernel launches written inside the test — not the production resident path — so its tokens are not
evidence about what ships. MEASURED, not assumed: driving the CPU reference identically (same
prompt, same re-feed of the last prompt token, same argmax rule) yields a DIFFERENT sequence, so
this bespoke pipeline does not reproduce decoder.forward. Making it faithful would just duplicate
the production resident path, which is already gated — so it is labelled instead.

CORRECTNESS FOR THIS PATH LIVES IN TestBackendResidentWired: it loads --backend cuda, takes the
real *cudaResident, and compares argmax against mcpu.ForwardForTest position by position
(measured 7/8 exact, worst near-tie 0.087%, hard fails 0). That is the token-identity evidence;
this file is the throughput number.

WHY IT IS A BENCHMARK AND NO LONGER A TEST (2026-08-19). It was `TestRealE2EDecodeThroughput`,
and as a test it sat in the pre-tag gate — where it failed the teacher-forced argmax guard ONCE
in ~200 tests, at a 28.835% margin, and then passed alone, passed after its immediate
predecessors, and passed on a full re-run of the identical heavy tier. Intermittent.

The decisive evidence for what that flake WAS: in the very run where this hand-rolled sequence
diverged, TestBackendResidentWired — the same comparison against the same CPU reference, on the
PRODUCTION resident path — passed at 7/8 exact, worst near-tie 0.087%, zero hard fails. So the
anomaly was in this file's duplicate forward, not in shipped code.

A test whose failure cannot indict the product does not belong in a correctness gate: it spends
the gate's credibility, and a red that means "the harness wobbled" trains people to re-run reds.
As a benchmark it runs only under `-bench`, so it is out of `go test ./...` and out of
the GPU gate, while the instrument it exists for — tok/s, the launch decomposition, GEMV vs glue
bandwidth — stays available on demand:
```

## BenchmarkRealE2EDecode.teacher

Moved from `cuda/reale2e_test.go` (the comment above `BenchmarkRealE2EDecode.teacher`) on 2026-10-09.

```text
The teacher-forced check is KEPT and still fails the run, deliberately: a throughput number for a
forward that does not reproduce production is worse than no number, and that check is what caught
this file's last real drift (rope_kv missing its `rhalf` argument). Its failures now cost a
benchmark run, not a release gate.
```

## BenchmarkRealE2EDecode.primary

Moved from `cuda/reale2e_test.go` (the comment above `BenchmarkRealE2EDecode.primary`) on 2026-10-09.

```text
Release the primary context when the test ends. dev.Primary() RETAINS a refcounted,
per-device singleton: leaving it retained pins the shared context alive for the whole
test binary, so no OTHER test's Close can ever drop the count to zero — and nothing
anyone allocated is reclaimed until the process exits. That leak saturated the 8 GB card
mid-suite, after which every Alloc/NewStream returned nil and the resulting zero-filled
buffers surfaced as bogus "cosine 0.000000 — layout/unpack mismatch" parity failures.
```

## BenchmarkRealE2EDecode.teacherforced

Moved from `cuda/reale2e_test.go` (the comment above `BenchmarkRealE2EDecode.teacherforced`) on 2026-10-09.

```text
---- CORRECTNESS FIRST: TEACHER-FORCED argmax parity against the CPU reference.

Teacher-forced, not free-running, and that is not a convenience. This path is argmax-equal to
the CPU reference but NOT bit-identical — TestBackendResidentWired measures 7/8 exact with a
worst near-tie of 0.087% on this very model. Free-running amplifies a single near-tie flip
into total divergence: an earlier version of this check drove both sides from the prompt-phase
argmax and got GPU [271 785 3840 ...] vs CPU [448 279 27130 ...], which looks like a
catastrophic bug and is actually one flipped tie plus chaos. Feeding both sides the SAME
tokens isolates the per-position computation, which is the thing under test.

This assertion is also what keeps this file from being a second UNVALIDATED forward. It is a
hand-rolled launch sequence, so it can drift from production silently — and it did: the
rope_kv call was missing its `rhalf` argument (added when partial rotary landed), which the
CUDA launch API does not arity-check, so the kernel read garbage for the rotary half-width and
this test happily reported a throughput number for a broken forward. The check below is what
makes the next drift fail instead of pass.
```

## cpuInt4VsF32GoldenFloor

Moved from `cuda/realforward_test.go` (the comment above `cpuInt4VsF32GoldenFloor`) on 2026-10-09.

```text
cpuInt4VsF32GoldenFloor is the precondition floor for a gap-0 hybrid-decode gate's FIRST check:
its own CPU int4 forward against an f32/bf16 HF golden — "does the reference itself hold, before
testing the CUDA bridge at all" (each such test's own comment). It is NOT the gate's real
assertion (the CPU-vs-hybrid-CUDA comparison right after it, same precision both sides, which
keeps its own tighter 0.99 bar); it only decides whether that real assertion is worth running.

0.98, matching decoder's own int4-vs-bf16/f32-HF precedent (oracleCosFloor's int4 case,
decoder/real_oracle_test.go) rather than the 0.99 these four tests inherited from an int8/f32
bar at introduction (gemma3_resident_real_test.go, gemma3_img_prefill_resident_real_test.go,
qwen25vl_resident_real_test.go, qwen25vl_mrope_prefill_resident_real_test.go).

MEASURED, NOT GUESSED (2026-09-28, after aikit v1.50.0's binary16 int4 group scales landed):
  - gemma3: 0.998167 at introduction (84850d62) -> 0.997912 confirmed on pre-f16-scale aikit
    v1.49.0 in a throwaway worktree -> 0.989747 today. The f16-scale CPU rounding is the whole
    cause: nothing else in the diff between those two aikit versions touches this path.
  - qwen25vl: 0.989088 on pre-f16-scale aikit v1.49.0 (already below the OLD 0.99 bar, so this
    one predates the f16-scale work) -> 0.988259 today, a further ~0.0008 from the scale change.
    TestQwen25VLReal_gate (decoder package) loads the SAME checkpoint at f32 and passes at
    cosine 0.999459 with an exact argmax, so this is int4 quantization noise on this specific
    checkpoint/path, not a defect in goinfer's Qwen2.5-VL forward.

Both numbers clear 0.98 today. Raise this back only with a new measurement, the same discipline
oracleCosFloor's own comment names for its int4 case.
```

## TestResolveCtxCapFit_shortcuts

Moved from `cuda/resident_cap_test.go` (the comment above `TestResolveCtxCapFit_shortcuts`) on 2026-10-09.

```text
TestResolveCtxCapFit_shortcuts pins the branches that need no real device: an explicit request
is untouched either way (fit-by-default only ever applies to an UNPINNED load), --fit=off
(Options.DisableFit) and its GOINFER_NO_FIT_DEFAULT env-var precursor both restore resolveCtxCap
exactly, and a model whose own window is already at or below cudaCtxCapDefault has nothing to
gain from asking Plan at all. The live-probe-driven branch (a real free-VRAM reading) is
exercised on real hardware separately (docs/tasks/task-gpu-paths-2026-09.md's G11 entry has the
nobara numbers). A real (if minimal, tracked-in-git) model is loaded per case rather than
passing nil — m.FitDisabled() reads a real field now, unlike the plain env-var check this
replaced, so a nil *decoder.Model would panic in the request==0 cases where it's evaluated.

`pinned` decides whether the case's Options carries ResidentContext: request, i.e. whether m
itself is a GENUINE pin (decoder.Model.ResidentContextPinned() true) or just a bare request
value passed straight to resolveCtxCapFit with an unpinned m — item 28
(docs/prompts/nobara-mc1-webgpu-2026-09.md §4): before this field existed here, every
request>0 case below loaded m WITHOUT ResidentContext set, so `pinned` was accidentally always
false and these cases were passing by coincidence (fitDefaultCtx is itself 8192, and llama-tiny
is small enough that Plan hands 8192 straight back unshrunk) rather than by actually taking the
"genuine pin, skip fit entirely" branch they claimed to test.
```

## TestResolveCtxCapFit_shortcuts.guardpin

Moved from `cuda/resident_cap_test.go` (the comment above `TestResolveCtxCapFit_shortcuts.guardpin`) on 2026-10-09.

```text
Item 28's own fix: request>0 but NOT pinned is a load-time fit-guard auto-pin (R13), not
a choice — it must still go through fit-by-default (never the request>0 shortcut above),
clamped so it can only ever land AT OR UNDER the guard's own pin, never over it. llama-tiny
is tiny enough that Plan always has room, so this lands exactly at the pin (the clamp,
not Plan, is what's under test) — including the below-cudaCtxCapDefault case, where the
OLD code would have wrongly floored back UP to cudaCtxCapDefault, raising a context past
what the guard proved host RAM could hold.
```

## TestResolveCtxCapFit_agreesWithCheckKVFits

Moved from `cuda/resident_cap_test.go` (the comment above `TestResolveCtxCapFit_agreesWithCheckKVFits`) on 2026-10-09.

```text
TestResolveCtxCapFit_agreesWithCheckKVFits is M-12 and M-22's own gate (docs/audit-2026-09-10.md),
on a real device: resolveCtxCapFit's own PLANNING-time choice must actually pass checkKVFits'
BUILD-time check — before M-12's fix, Plan reserved zero margin while checkKVFits required an
extra 384 MiB on top, so any INTERIOR (non-ceiling, non-floor) choice failed almost every time.
A real card usually has far more free VRAM than testdata/llama-tiny plus 8192 positions could
ever need, so the default candidate hits the CEILING branch (fits outright) — never buggy even
before the fix, and not what this finding is about. ExtraResidentBytes forces the INTERIOR
branch deterministically: computed from a real free-bytes probe so the remaining budget lands
resolveCtxCapFit's candidate choice between the floor and the ceiling, not by chance.

Run with and without an ADDITIONAL simulated companion K/V rate (M-22) on top of that forced
interior scenario: resolveCtxCapFit prices that rate against its own candidate (never smaller
than what it eventually picks), and the real build prices it against the FINAL chosen ctx — this
proves those two, independently-computed prices stay consistent, not just individually
plausible. The companion rate (40960 B/position) is the audit finding's own cited real number: a
5-layer DFlash trunk's K/V measured at "5 x 8192 x 1024 x 8 B ~ 335 MB" for an 8192-position
target, i.e. ~40.96 KB/position — not a round guess.
```

## TestResolveCtxCapFit_agreesWithCheckKVFits.target

Moved from `cuda/resident_cap_test.go` (the comment above `TestResolveCtxCapFit_agreesWithCheckKVFits.target`) on 2026-10-09.

```text
Target the interior of (cudaCtxCapDefault, fitDefaultCtx]: an ExtraResidentBytes big enough
that, with the SAME margin the fixed resolveCtxCapFit itself subtracts before calling Plan,
dense+kv(fitDefaultCtx)+extra does NOT fit, but dense+kv(targetCtx)+extra does — forcing
chooseCtx's budget branch instead of its "fits outright" shortcut. Computed against the
MARGINED budget on purpose: llama-tiny's own perPos is tiny (a plain dense fixture) next to
ctxCapMarginBytes (384 MiB), so a forcing term sized against raw free would leave the fixed
code's own margin subtraction with nothing to bite on and land the floor instead of the
interior — sized this way, the interior landing is exact in the FIXED case; the reverted
case (no internal margin subtraction) gets ~384 MiB of slack this same term doesn't leave
room for, landing it at the ceiling (fitDefaultCtx) instead — still a real, checkable
disagreement between planning and build time (see the checkKVFits assertion below), just not
"interior" in that one arm. Both arms are logged, not asserted on, for exactly that reason.
```

## TestGemma3ResidentReal_imageBlockAtomicity

Moved from `cuda/resident_reuse_vl_parity_test.go` (the comment above `TestGemma3ResidentReal_imageBlockAtomicity`) on 2026-10-09.

```text
TestGemma3ResidentReal_imageBlockAtomicity is P9(a)'s pre-registered kill condition
(docs/multimodal.md's P9(a) plan, "Top risk: the atomicity boundary"), run against the REAL
GenerateVL entrypoint on gemma-3-4b-it's resident CUDA backend — not the decoder package's own
fakeResident unit gates (decoder/generate_vl_resident_test.go's TestGenerateVL_imageReuseFastPath_*
cover the control-flow logic in isolation; this proves the same claim against real kernels).

(A WebGPU version of this gate was tried first, matching gap 0's own CUDA+WebGPU split — but
gemma-3-4b-it's text decoder declines residency on this box's WebGPU backend entirely
("arch needs unimplemented feature(s) [embed-scale gated-gelu sandwich-norm]"), so it could
only ever SKIP, providing no real coverage. P9(a)'s reuse logic lives entirely in the backend-
agnostic decoder package (gap 0 already validated the CUDA and WebGPU kernel primitives
separately), so CUDA — proven to hold Gemma 3 resident on this exact box,
TestGemma3ResidentReal_gate above — is an equally valid real-hardware witness for this
specific claim.)

Same token-IDENTITY methodology as gpu/resident_reuse_parity_test.go's
TestResidentPrefixReuse_tokenIdentical, for the identical reason stated there: a wrong
image-block match produces fluent, confidently wrong output with no error anywhere, so only
bit-for-bit identity against a reuse-disabled cold run is acceptable evidence. Two claims,
both checked, across a 3-turn transcript (image A, the SAME image A resent, then a DIFFERENT
image B at the same placeholder slot):

  - Resending the SAME image must reuse the resident KV — proven by the vision tower's own
    call counter staying flat across that turn, and by Generation.PrefillReused reporting a
    reuse length reaching at least past the image block — AND the warm run's output must be
    bitwise identical to a cold (reuse-forced-off) run's.
  - A DIFFERENT image at the SAME slot must NOT reuse — proven by the tower being called
    again, PrefillReused reporting 0, AND (same as above) bitwise identity against a cold run —
    a wrong-in-the-other-direction bug (claiming reuse it shouldn't) would still show up here
    even if it happened to produce plausible-looking text.

The vision tower's own forward pass is real but MEMOIZED by image identity: a real forward
pass on this checkpoint is documented as minutes-scale on CPU (demo/agent/agent.go's TurnImage
doc comment), and a deterministic f32 forward pass over identical input produces identical
output by construction — reuse-vs-not is a property of GenerateVL/the resident KV, not of the
tower, so recomputing the SAME image's features on every call would only spend real time
proving something this test doesn't need re-proven. The call counter still increments on EVERY
invocation (that is what proves whether GenerateVL actually called the closure); only the
expensive compute behind it is paid once per distinct image across the whole test — 2 real
forward passes total, not up to 6.
```

## TestResidentLaunchVRAMProbe

Moved from `cuda/resident_vram_probe_test.go` (the comment above `TestResidentLaunchVRAMProbe`) on 2026-10-09.

```text
TestA1PreLaunchProbe settles where the 64 MiB in the 34-slot failure came from.

The shipped message reported "265945088 B free" at the failing fRoute launch, and
265,945,088 − 198,836,224 (free at the FIRST launch) is exactly 67,108,864 = 2^26. But
describeLaunchErr is reached only after r.stream.Launch returns non-nil, so that reading is
taken AFTER the failure. It cannot distinguish:

	(a) 64 MiB was released by intervening work, and fRoute then wanted more than 253.6 MiB; from
	(b) nothing was released, and the failed attempt itself freed a driver-side block while
	    unwinding — in which case the 64 MiB is an artifact of where the probe sits.

An exact 2^26 reads more like a driver or module block than like application scratch, which is
what makes (b) the live hypothesis. The probe records free VRAM immediately BEFORE
every launch, so the same event is observed from the other side. The trace also yields A9's
first decrement (free at first launch → free at the failing launch) from this one run.
```

## TestRmsnormBatched_bitIdentical

Moved from `cuda/rmsnorm_batched_test.go` (the comment above `TestRmsnormBatched_bitIdentical`) on 2026-10-09.

```text
TestRmsnormBatched_bitIdentical compares the batched rmsnorm_quant_batched (M=1) against the decode
rmsnorm_quant HEAD-TO-HEAD at the REAL hidden width (1536) — the comparison no existing gate makes.
The batched-vs-decode forward gap (TestBatchedVsDecodeGap; 84% stream divergence) was localized past
the GEMV (bit-identical at real dims) to the RMS by elimination; this pins whether the two RMS
kernels actually diverge, and at what magnitude. Same input, weight, eps, addOne, blockDim (256) as
the two production launches (r.rms / bRmsB).
```

## ropeMRopeArgs.qTempRows

Moved from `cuda/rope_kv_mrope_batched_test.go` (the comment above `ropeMRopeArgs.qTempRows`) on 2026-10-09.

```text
qTempRows, added to rope_kv_batched by b26fa122 (Ministral 3's per-row query scale): null means
no attention temperature, which is what prefillCore's attnTempRows passes for every other family.
Without it the launch has 13 of the kernel's 14 parameters and the driver refuses it with
CUDA_ERROR_INVALID_VALUE, which is how this test sat red from 2026-09-11 to 2026-09-28.
```

## TestRopeMscale

Moved from `cuda/rope_mscale_test.go` (the comment above `TestRopeMscale`) on 2026-10-09.

```text
WHY IN ISOLATION, AND WHY BEFORE THE DECLARATION. `FeatRopeMscale` is a claim that this
backend can express YaRN's cos/sin scaling. Until 2026-08-31 CUDA's three rope kernels took
no scale parameter at all — so declaring the feature would have admitted gpt-oss AND (as a
documented side effect, since CUDA already declares Mellum's other four required features)
Mellum onto a path that silently ignores the factor. Silently: the kernel is correct
arithmetic, just the wrong arithmetic, and nothing errors. A gate that runs a whole model
and checks a cosine can miss a scalar this small; this one cannot, because it compares
against a scalar reference computed in Go.
```

## TestRouteGptOssGrowsPoolPastMoERoute

Moved from `cuda/route_gptoss_reservation_test.go` (the comment above `TestRouteGptOssGrowsPoolPastMoERoute`) on 2026-10-09.

```text
TestRouteGptOssGrowsPoolPastMoERoute measures the gap audit-2026-09-10 G-13(b) closed.

BuildResident pays the deferred local-memory reservation before sizing the expert cache by
launching the kernel with the most per-thread scratch (see TestMoERouteFirstLaunchReservation for
the mechanism). It launched moe_route, 4416 B/thread. route_gptoss declares 4608, and it is the
router on gpt-oss, the model the expert cache exists for. This replays the old warm-up (moe_route
alone), then reads what route_gptoss's first launch takes on top. That figure is what allocSlots
could not see on gpt-oss before the fix.
```

## TestS10DeepstackPrefillCUDA_real

Moved from `cuda/s10_deepstack_prefill_test.go` (the comment above `TestS10DeepstackPrefillCUDA_real`) on 2026-10-09.

```text
TestS10DeepstackPrefillCUDA_real is G-S10g's real half, on Qwen3-VL-2B and the four F2a images at serve's 1,024-row cap, graded by S16's RE-REGISTERED bar (the Mac's, 2026-10-08, owner option a): the strict
per-step 0.9999 is tighter than a shipped batched prefill's own distance from the CPU's int4 prefill, so the resident DeepStack prefill is held to non-inferiority against a text control. For each image prompt, ctlPrompts (12; four until the owner's decision (c) of 2026-10-08)
text-only prompts of the same length run the same isolated comparison (the CPU's prefill and upload against CUDA's batched PrefillLast, both decoding the same 8 teacher-forced steps on CUDA). PASS when the image
turn's worst per-step cosine (the last row and the 8 steps) is at least the control's minimum minus 0.005 and every argmax difference is an R10 near-tie; 0.005-0.015 below the control's minimum is parked; worse fails.
Both image arms see the same features and sets (the CPU encoder's). Heavy: about 6-8 minutes, a line per stage.
```

## TestS10DeepstackPrefillCUDA_real.ctlPrompts

Moved from `cuda/s10_deepstack_prefill_test.go` (the comment above `TestS10DeepstackPrefillCUDA_real.ctlPrompts`) on 2026-10-09.

```text
ctlPrompts is the text control's size per image length: 4 until the owner's decision (c) of 2026-10-08 made it 12. Seeds 1-4 are the original four, so the minimum can only be lower.
```

## TestS10DeepstackPrefillCUDA_diag

Moved from `cuda/s10_deepstack_prefill_test.go` (the comment above `TestS10DeepstackPrefillCUDA_diag`) on 2026-10-09.

```text
TestS10DeepstackPrefillCUDA_diag is an EXPLORATORY diagnostic, not a gate: G-S10g's real reading failed on table.png (0.9125 against a control minimum of 0.9539) and was parked on the 4x6 image. It re-runs the
comparison on those two images in cuts that separate the candidate mechanisms, printing the per-step cosines: (a) as the gate; (b) with NO DeepStack sets in either arm (is the gap the batched prefill on image rows,
not the injection?); (c) as the gate with the fast prefill levers forced off (is it the L2/L3 levers?). Same features, same prompt, same teacher-forced steps in every cut.
```

## TestS10DeepstackPrefillCUDA_speed

Moved from `cuda/s10_deepstack_prefill_test.go` (the comment above `TestS10DeepstackPrefillCUDA_speed`) on 2026-10-09.

```text
TestS10DeepstackPrefillCUDA_speed is an EXPLORATORY split of one image turn's prefill on Qwen3-VL-2B and table.png (986 rows), single samples, nothing graded: today's path (the CPU prefill with the sets, then the
KV upload) against the resident DeepStack prefill. It exists so the decision on the parked image is made with the stake in view; the night record is the S7 cell.
```

## TestS17Profile.overheadlimit

Moved from `cuda/s17_profile_test.go` (the comment above `TestS17Profile.overheadlimit`) on 2026-10-09.

```text
The drains add a fixed bubble per class change, so a forward of tens of milliseconds (the dispatch-bound 14x20 grid) reads over the limit by construction; the registered limit is for the
towers a lever targets. Under half a second the overhead is logged, not failed (the 2026-10-07 heavy tier failed this on that grid: +14.5% of 45 ms).
```

## TestS17ProfileQwen25CUDA

Moved from `cuda/s17_profile_test.go` (the comment above `TestS17ProfileQwen25CUDA`) on 2026-10-09.

```text
TestS17ProfileQwen25CUDA is the step 0 profile of the Qwen2.5-VL tower on the tower base (S7 on CUDA's fix): the first real read missed its registered band (4.04 s on the 896x896 image against 1.7-2.6 s), so the class split
says where the time is. Heavy; run alone.
```

## s4_towers_test.header

Moved from `cuda/s4_towers_test.go` (the comment at the top of the file) on 2026-10-09.

```text
G-S4q of docs/tasks/task-multimodal-support-2026-10.md (S4), the CUDA twin of metal/s3_towers_test.go (G-S3a). It was written for aikit's CUDA Qwen2.5-VL tower (gpu/qwencuda); that tower is no longer imported (S7 on
CUDA's fix, 2026-10-08: goinfer's own, qwen25_vision.go, registers through the same seam), so this now grades whatever EnableResident attaches, which is goinfer's tower, against aikit's CPU tower, f32 on both sides, every output token at cosine >= 0.9999; 0.999-0.9999 is ambiguous
(parked). aikit's CUDA SigLIP tower (gpu/visioncuda) is read in cuda/s4siglip, a package of its own, because inside package cuda goinfer's own SigLIP tower
(cuda/vision_register.go) registers last and wins the global hook; it is information for the aikit defect report, not a gate. The tiny towers always; the real ones (~/models, never the archive) under GOINFER_HEAVY_TESTS=1, on the four images F2a uses.
Times are exploratory.
```

## TestSampledGumbelStreamIdentity

Moved from `cuda/sampled_gumbel_identity_test.go` (the comment above `TestSampledGumbelStreamIdentity`) on 2026-10-09.

```text
TestSampledGumbelStreamIdentity is R7b's end-to-end gate (docs/tasks/red-october.md): with a fixed seed, the token
stream drawn on-device by Gumbel-max is IDENTICAL to the stream the host draws (GOINFER_NO_SAMPLE_FASTPATH=1) —
the host and the device implement one algorithm over one Philox counter stream, so they may differ only where two
candidates' scores are within an f32 rounding (measured ~1e-6 per token, and 0 in 15,840 kernel-level draws).
Real checkpoints, three temperatures, 1,000 tokens each.

PRE-REGISTERED: any divergence fails, and is investigated, not tolerated: at ~1e-6 per token a divergence in ~36k
tokens is a ~3% event, so one is more likely a bug than rounding. The test also fails if the device path never
engaged (DeviceSampled == 0), so it cannot pass vacuously.
```

## TestGemvShapeSweep

Moved from `cuda/shape_sweep_test.go` (the comment above `TestGemvShapeSweep`) on 2026-10-09.

```text
and that shape SATISFIES the precondition the shipped kernel assumed. packWeight guarded
K%32, but the kernel's lanes step in 32-word strides, so the real requirement is on
Kwords = K/8. Qwen2.5-0.5B (hidden 896) gives Kwords = 112, and 112%32 = 16 — the tail lanes
read past the row. K=1536 gives Kwords=192, 192%32 == 0, so the bug was invisible BY
CONSTRUCTION: the one tested shape was the one that could not fail. That cost a real
out-of-bounds read on the 0.5B, which is the model the README's headline number is measured on.
```

## TestGemvShapeSweep.scales

Moved from `cuda/shape_sweep_test.go` (the comment above `TestGemvShapeSweep.scales`) on 2026-10-09.

```text
The kernel reads __half group scales, so the reference must read the SAME rounded
values — not the f32 originals. f16 carries ~5e-4 relative precision, so comparing
against unrounded scales manufactures a ~0.05% "error" that is the TEST's rounding,
not the kernel's. (This first showed up as failures on well-aligned shapes, which is
what proved it was the reference and not the Kwords%32 tail.)
```

## TestAttnShmemLimit_matchesDevice

Moved from `cuda/shmem_limit_test.go` (the comment above `TestAttnShmemLimit_matchesDevice`) on 2026-10-09.

```text
M-16: the single-block attention kernels size their scratch (nWin+128)*4 with NO ceiling, so
past 12,160 attended keys the launch exceeds the 48 KB default and is refused by the driver.
Decode fails at that position; batched prefill errors at layer 0 and falls back to the ~9x
slower sequential path with nothing logged. The trigger is -ctx 16384+ on any geometry whose
perf table says splitkvNever (nH >= 24: Qwen2.5-7B, Llama-3-8B, phi3-mini) — the -ctx 32768
rows in benchmarks.md were on 0.5B/1.5B, whose split-KV engages at 3072/1024.

The audit rated this medium confidence because it hinged on whether anything raises the
kernel into the opt-in range. Nothing does, and this pins BOTH halves against the device.
```

## shmem_limit_test.V05

Moved from `cuda/shmem_limit_test.go` (the comment above `TestCheckPrefillShmem_declinesPastTheLimit`) on 2026-10-09.

```text
V-05 (docs/review-2026-09-04.md): M-16 fixed decode (above) but left batched prefill (also the
spec-decode verify path, since prefillCore serves both) and the drafter block attention with the
SAME unguarded (nWin+128)*4 launch and no split-KV fallback. -ctx 16384+ on a splitkvNever
geometry (nH >= 24: Qwen2.5-7B, Llama-3-8B, phi3-mini) hit the driver refusal at prefill/verify
time instead of decode time, and the caller (decoder/model.go's PrefillLast handling) silently
fell through to the ~9x-slower sequential path with nothing distinguishing "declined" from
"crashed". These need no device: they drive checkPrefillShmem/the drafter's inline check
directly on a struct-only fixture, the same way TestPrefillPath_matchesPrefillCore does.
```

## TestSiglipCUDA_int8VsFloat32

Moved from `cuda/siglip_vision_test.go` (the comment above `TestSiglipCUDA_int8VsFloat32`) on 2026-10-09.

```text
TestSiglipCUDA_int8VsFloat32 is G-S3d's feature half (the S4 addendum): the shipped default's tower, the W8A8 *cuda.VisionEncoder on an int8-loaded encoder, against the
float32 CPU tower on the four F2a images: relative L2 and worst / mean per-token cosine, to set beside docs/measurements/siglip-int8-fidelity-2026-10-07.md (relative L2 0.16-0.52,
worst token 0.01-0.17). A record, not a gate: it asserts only that the int8 tower is what attached. Heavy.
```

## TestSlotAllocation_matchesGranularityForm

Moved from `cuda/slot_granularity_test.go` (the comment above `TestSlotAllocation_matchesGranularityForm`) on 2026-10-09.

```text
TestSlotAllocation_matchesGranularityForm asserts that the expert cache's VRAM consumption is
predicted by rounding EACH slot buffer up to the driver's allocation quantum independently.

This is the gate that would have caught A1. The cap arithmetic in allocSlots sizes the cache from
a raw byte sum — slots × bytes-per-slot × layers — and the driver charges for whole quanta, four
times per layer. On the real 26B the shortfall put the granted cap one step past what fits: at 34
slots all four buffers tip a quantum at once, a 4-quanta step per layer, and the forward died in
a one-block routing kernel with 189.6 MiB still free. Every test in the suite passed throughout,
because allocSlots runs in every MoE test and its arithmetic is never compared against what the
driver actually took.

Structure is asserted BEFORE totals, deliberately. A total that matches under a wrong structure
is worse than a mismatch: it looks like confirmation. So the shape is pinned first — one buffer
group per MoE layer, four buffers each, four distinct sizes in the ratio the int4/group-32 layout
implies — and only then is the arithmetic believed.
```

## TestSlotAllocation_matchesGranularityForm.mutation

Moved from `cuda/slot_granularity_test.go` (the comment above `TestSlotAllocation_matchesGranularityForm.mutation`) on 2026-10-09.

```text
MUTATION CHECK (run before trusting this green): change roundUp's body to `return n`, i.e. sum the
requested sizes without rounding. That is exactly the defect A1 was. At 16 slots on the scaled
fixture the prediction drops from 226,492,416 to 214,106,112 and the test fails by 12,386,304 B.
```

## TestSlotCapArithmetic

Moved from `cuda/slotcap_test.go` (the comment above `TestSlotCapArithmetic`) on 2026-10-09.

```text
TestSlotCapArithmetic gates the expert-cache CAPPING branch — code that ships, decides how much
VRAM to claim, and had never once been executed by a test.

It could not be: the branch only binds when the requested slots exceed free VRAM, and no fixture
is remotely large enough for that. It bound for the first time on the real 26B. That is the
exercised-but-never-triggered shape — a branch inside well-tested code that the tests' inputs can
never reach, so coverage tools report it green and it has never run.

TWO CLAIMS THIS FILE USED TO MAKE, BOTH WITHDRAWN.

It said it "corroborates the sizing", on the grounds that it predicted 34 and the hardware
produced 34. It corroborated a PARALLEL COPY: allocSlots had its own inline arithmetic and this
gate drove capSlots, so agreement between them showed only that two transcriptions of the same
formula agreed. allocSlots now calls capSlots, so the gate points at the shipping path.

And it said the agreement placed the 26B discrepancy "downstream of the sizing decision". It did
not. The formula both copies implemented was WRONG: it summed requested bytes, while the driver
charges each of the four buffers per layer its own whole 2 MiB quanta. 34 was the answer to the
wrong question, and the forward at 34 slots generated zero tokens. The two copies agreeing was
never evidence about the answer — only about the copying.
```

## softcapSerial

Moved from `cuda/softcap_test.go` (the comment above `softcapSerial`) on 2026-10-09.

```text
softcapSerial is the reference: the exact loop that shipped at cuda/resident.go and
cuda/prefill.go before applySoftcap, and that still ships at decoder/forwardn.go,
decoder/model.go and metal/model.go. Kept verbatim so the gate compares against the thing the
other three siblings still do, not against a re-derivation of it.
```

## TestApplySoftcap_mutation

Moved from `cuda/softcap_test.go` (the comment above `TestApplySoftcap_mutation`) on 2026-10-09.

```text
A first attempt used float32(v/sc) as the mutation. That is a NO-OP — v and sc are both float32,
so v/sc is already float32 arithmetic — and the gate said so rather than passing on a mutation
that changed nothing. Recorded because a mutation check that mutates nothing is the same defect
as a gate that cannot fail.
```

## TestSmolLM3ResidentSmokeCUDA.unrun

Moved from `cuda/smollm3_resident_smoke_test.go` (the comment above `TestSmolLM3ResidentSmokeCUDA.unrun`) on 2026-10-09.

```text
WRITTEN, NOT RUN: no CUDA device was available while writing this (see
docs/tasks/task-gpu-paths-2026-09.md's G5 status log) — needs a real run on a CUDA box before it can
be trusted, same posture as G4's cudaResident.HiddenLast.
```

## TestSlotCapArithmetic_mutation

Moved from `cuda/slotcap_test.go` (the comment above `TestSlotCapArithmetic_mutation` (the margin line)) on 2026-10-09.

```text
	margin removed             -> 37 slots. DERIVED under the ROUNDING form, and worth stating how
	                              it was got wrong first: 38 was carried over from the old raw-sum
	                              derivation without re-deriving, and this gate caught it. With
	                              x = n*123904/2MiB, n=37 gives quanta 3+5+18+35 = 61, so
	                              30 x 61 x 2 MiB = 3,837,788,160 <= 3,847,880,704 free; n=38 gives
	                              3+5+18+36 = 62, so 3,900,702,720 > free. The margin costs 4 slots
	                              (33 -> 37), not the 4 the old sum happened to give (34 -> 38) —
	                              same delta, different endpoints, and only one of them is real.
```

## TestSpecAdaptiveConcurrentCUDA

Moved from `cuda/spec_adaptive_concurrent_test.go` (the comment above `TestSpecAdaptiveConcurrentCUDA`) on 2026-10-09.

```text
TestSpecAdaptiveConcurrentCUDA is the gate the MC4 candidate (-spec-adaptive) was missing: N CONCURRENT greedy
GenerateNgramSpeculative generations on a REAL CUDA resident with SetSpecAdaptive(true) must emit exactly what each
emits alone. decoder/spec_adaptive_switch_test.go checks the same property on mc3Fake, which has no numerics and no real
slots, so it passed 200/200 while the 2026-10-01 graded run (docs/measurements/mc4-candidate-cuda-2026-10-01.md) found
nearly every concurrent reply wrong: a round claimed only resBusy and so ran its bind-then-verify while another
generation's exclusive section (slot pick, prefill, commit) bound a different slot, and each wrote the other's. Verified
red without the fix (12 of 12 generations differ at 2 clients, the first token already wrong) and green with it.
```

## TestSpecDecodeCurve.speculative

Moved from `cuda/spec_curve_test.go` (the comment above `TestSpecDecodeCurve.speculative`) on 2026-10-09.

```text
The loop has decoder/spec_ngram.go's shape: the prompt is prefilled ONCE, the first token is
the prime's own argmax (`cur`), and each round verifies [cur, draft…] starting at cur's
position — the last prompt token is never fed again. An earlier version re-fed it (verify
from position depth-1), which rewrites that KV row through the decode kernels. At depth ≥
fastPrefillFloor the prime's row came from the fast (tensor-core) prefill, so the two arms
then attended over different KV and a near-tie flipped at depth 512 token 10 — a harness
artefact, never production (GOINFER_CUDA_FAST_PREFILL=0 made every depth pass).
```

## TestSpecNonCopyLane.doc

Moved from `cuda/spec_noncopy_lane_test.go` (the comment above `TestSpecNonCopyLane.doc`) on 2026-10-09.

```text
A FROZEN copy of the first 14,800 bytes of docs/benchmarks.md as of 33a18e82 (the commit that added this test; 4,911 prompt tokens on the Qwen tokenizer). The test
used to read the live file, which kept growing: by 12c85f4a the same slice was 6,233 tokens, over the 6,144-position context pinned below, so every arm declined
the batched prefill, ran the per-token path (940 s) and died at "KV position 6144(+1) exceeds resident context cap" -- the 2026-10-07 night gate's failure,
root-caused 2026-10-08 by tokenizing the slice at three revisions. A measurement over a mutable document is not reproducible either.
```

## spec_pager_interaction_test.header

Moved from `cuda/spec_pager_interaction_test.go` (the comment at the top of the file) on 2026-10-09.

```text
Does a speculative verify break the expert pager?

The question comes from a field report of a 176B MoE in a hybrid split (experts on CPU,
attention on GPU) where enabling drafting collapsed throughput, with a stated mechanism of a
verify forcing expert re-fetches. That report is the ORIGIN of the hypothesis and nothing else —
n=1, LLM-narrated, several confabulated claims — so no number from it appears here or is
compared against. Only the mechanism is under test, generalized off its CPU/GPU framing (which
this tree cannot run: Lead 5 is proposed, not built) onto the path that does exist — a width-K
verify against a model whose routed experts are paged host→VRAM by C′.

TWO HYPOTHESES, AND WALL-CLOCK CANNOT SEPARATE THEM, which is the whole reason this test exists
in the shape it does:

	H-paging    a width-K verify presents K positions' routing at once, so it asks for several
	            times the distinct experts a decode step does, overflowing a slot budget that was
	            tuned on decode traffic.
	H-noamort   prefill.go declines the batched weight-stationary path for MoE, so a verify
	            walks position by position: the pager sees ordinary decode traffic and is never
	            stressed, but the verify pays K full decode steps to commit at most K tokens and
	            the amortization speculation depends on is simply absent.

Both predict a regression, with the same sign and a similar size. They are told apart by
PagerStageStatsForTest's distinct-experts-per-staging-event, which rises with K under H-paging
and stays pinned at topK under H-noamort. Pre-registration, with the thresholds and the
ambiguous→parked bands fixed before any arm ran: docs/measurements/spec-x-pager-prereg-2026-09-02.md

ONE SLOT RUNG PER PROCESS. The slot depth is read at Load, so a ladder inside one process would
mean reloading a 20 GB pinned allocation per rung; the runner drives the rungs by re-invoking
with GOINFER_MOE_CACHE_SLOTS set, and each process reports the depth actually BUILT (capSlots
caps a request to free VRAM, so the request is not the depth).
```

## specPagerArm

Moved from `cuda/spec_pager_interaction_test.go` (the comment above `specPagerArm`) on 2026-10-09.

```text
specPagerArm is one measured configuration. Times are per (prompt, repeat) and kept RAW rather
than pre-averaged, so the paired ratios R1 asks for can be formed per round — a ratio of medians
disagreed with the paired form by 7.6 pp in one run in this tree and under 1 pp in others, which
is exactly what makes it uncorrectable after the fact.
```

## specPagerArm.secs

Moved from `cuda/spec_pager_interaction_test.go` (the comment above `specPagerArm.secs`) on 2026-10-09.

```text
secs is wall-clock NORMALIZED to a fixed token count: a round commits a whole block, so a
speculative arm overshoots or undershoots MaxTokens by a few tokens while the off arm lands
exactly on it. Comparing raw durations would then charge an arm for tokens it produced as a
bonus. Normalizing is what makes this the denominator-free metric it claims to be — the
divisor is the arm's OWN emitted count, never another arm's decode step, which is the
contamination that flattered a ratio in this tree this week.
```

## specPagerSpread

Moved from `cuda/spec_pager_interaction_test.go` (the comment above `specPagerSpread`) on 2026-10-09.

```text
specPagerSpread is (max-min)/mean, the same form the Metal slot sweep reported its 20.8%
thrashing signature in. Reported because a pager under pressure announces itself in VARIANCE
before it does in the mean.

IT IS APPLIED WITHIN A PROMPT, NEVER POOLED ACROSS THEM. Pooling would fold between-prompt
variance (different lengths, different routing) into a number read as run-to-run noise, and the
prompts here differ by design — the pooled figure would be dominated by the thing the arms hold
constant rather than the thing that varies between repeats. Same rule as differencing matched
observations instead of pooling them: measured elsewhere in this repo, pooled sd of 10-35 tok/s
against an ~8% effect became 5.5-8.7 paired, and the two disagreed about whether an effect existed.
```

## TestSpecPagerInteraction.venue

Moved from `cuda/spec_pager_interaction_test.go` (the comment above `TestSpecPagerInteraction.venue`) on 2026-10-09.

```text
PARAMETERIZED OVER THE VENUE, because one paged MoE cannot answer the whole question.
qwen3.6-35B-A3B is a Gated-DeltaNet MoE: the block path declines on the MoE batched-verify
check AND the n-gram path declines on the recurrent-rollback check, so on that model both
speculative arms refuse for two unrelated reasons and no throughput number exists to be had.
gemma-4-26B-A4B is a paged MoE with NO recurrent state, so its n-gram arm actually runs — it
is the venue where the field report's throughput claim can be tested rather than sidestepped.
```

## TestSpecPagerInteraction.slotcap

Moved from `cuda/spec_pager_interaction_test.go` (the comment above `TestSpecPagerInteraction.slotcap`) on 2026-10-09.

```text
V-24 (docs/review-2026-09-04.md): this used to say a request AT OR BELOW topK was
SILENTLY IGNORED. That was true before G-07 (cuda/backend.go) and is false now: a request
BELOW topK is a HARD ERROR at Load time — the `t.Fatalf("Load(35B, cuda int4): %v", err)`
above already ends the test for one, so this line is never reached with n < r.topK. A
request AT OR ABOVE topK is honoured UNIFORMLY (backend.go no longer distinguishes `==`
from `>`), then possibly capped — first to the model's own expert count, then to measured
free VRAM by capSlots (cuda/resident.go). Either cap is a real rung, just not the
requested one; say so rather than assert a mechanism that no longer exists.
```

## TestSpecPagerInteraction.warmup

Moved from `cuda/spec_pager_interaction_test.go` (the comment above `TestSpecPagerInteraction.warmup`) on 2026-10-09.

```text
WARM UP EVERY PROMPT, not just the first. A single global warm-up left the first repeat of
each prompt paying a cold pager: measured at 64 slots the per-prompt first repeat was slowest
every time (4.48 vs 2.87 s, 5.33 vs 4.21, 5.67 vs 3.64), and the resulting WITHIN-prompt
spread was 49.8% — far too noisy to resolve R1's 10% threshold against. The cold cost is real
but it belongs to neither arm, and whichever arm runs first would otherwise absorb it.
```

## TestSpecPagerInteraction.generr

Moved from `cuda/spec_pager_interaction_test.go` (the comment above `TestSpecPagerInteraction.generr`) on 2026-10-09.

```text
CHECK gen.Err(). Discarding it made a real failure present as "produced no
tokens", which is indistinguishable from an empty generation and sent the
diagnosis chasing recurrent-state resets that were in fact all present. An
error swallowed here is an error attributed to the wrong mechanism.
```

## TestSpecResidentPrefillRegression

Moved from `cuda/spec_prefill_regression_test.go` (the comment above `TestSpecResidentPrefillRegression`) on 2026-10-09.

```text
TestSpecResidentPrefillRegression attributes the resident speculative slowdown to
PROMPT LENGTH, which is the signature of per-token prefill.

THE DEFECT. decoder/model.go's generateInto uses the optional Prefiller seam on a
resident model — `pf.PrefillLast(context.Background(), embs, 0)`, one batched on-device pass — whenever
len(prompt) >= 8. decoder/spec_ngram.go's genNgramInto does NOT: its resident
branch loops `target.resident.Forward(embedResident(id), i)` once per prompt
token. cudaResident implements PrefillLast (cuda/prefill.go), so the batched path
exists and is simply not taken on the speculative path.

WHY IT WAS NEVER SEEN. gpu/spec_ngram_resident_test.go's corpus prompts are 36-74
tokens, where the penalty is a fraction of a second and hides inside generation.
It scales with prompt length, and no harness had a long prompt until the realistic
corpus (656-1039 tokens) in docs/spec/02.

THE CONTROL. If the slowdown is prefill-driven it must be roughly CONSTANT in
absolute terms and vanish as a ratio on a short prompt. If instead speculation
were inherently slow here, the ratio would persist at both lengths. Same model,
same session, interleaved.
```

## TestSpecResidentPrefillRegression.gate

Moved from `cuda/spec_prefill_regression_test.go` (the comment above `TestSpecResidentPrefillRegression.gate`) on 2026-10-09.

```text
THE GATE. This is the assertion the original GPU speculative harness lacked: it
measured the right quantity and only LOGGED it ("Parity is hard-gated; speedup is
logged per workload"), so a 3-4.5x slowdown printed and failed nothing.

It gates the DEFECT SIGNATURE, not "does speculation pay". Per-token prefill makes
the off-vs-spec gap grow LINEARLY in prompt length; whether speculation is a net win
at a given acceptance rate is a separate, noisy question that would make this flap.
Measured on this box: 2.66 ms/prompt-token with the bug, 0.12 ms/prompt-token after
wiring genNgramInto to residentPrefillSeed. The bar sits between them, nearer the
fixed value, so a regression has to be a real return of per-token prefill to trip it.

THIS BAR IS CUDA-CALIBRATED AND IS NOT PORTABLE. CUDA's exposure was DOUBLE and every
other backend's is at most single: it is the only backend implementing ForwardNoLogits,
so even when batched prefill declined its fallback was KV-only while the speculative
path did full-logits — a second asymmetry no one else has. Metal, measured on the
MacBook the same day, showed the same defect roughly 2x HARDER (5.313 ms/prompt-token,
R^2 0.9999) but ONLY under --metal-fast-prefill; at its default it declines batched
prefill, both paths take the per-token loop, and the slope is 0.110 (R^2 0.42) with no
asymmetry to find. A port of this gate must re-derive its own bar from its own
defect/fixed pair rather than inheriting 0.50.
```

## spec_twoturn_test.header

Moved from `cuda/spec_twoturn_test.go` (the comment at the top of the file) on 2026-10-09.

```text
spec_twoturn_test.go is item 30 (docs/prompts/nobara-cuda-spec-trailing-token-2026-09.md §2): the CUDA check of
two fixes measured and shipped on Metal / against stubs only —

  - 0e579400 (the block drafter, --drafter, CUDA only): at a completed exit (EOS or max_tokens) the resident
    commit recorded prompt + every emitted token as held in KV, but the LAST emitted token — the next round's
    anchor — was never forwarded through the target. A second turn then reused that position as it stood: a
    rejected draft's K/V, or nothing at max_tokens 1.
  - 97615930 (the n-gram loop, serve --spec ngram): the same gap — the trailing token was forwarded only as the
    next round's seq[0], so a generation ending at max_tokens left the cache one token short.

Both were proven on Metal (spec_multiturn_test.go) or only against stubs
(TestBlockSpecGenerate_commitsOnlyWrittenPositions) — the real drafter and kernels exist only on CUDA. This
ports spec_multiturn_test.go's shape (fresh load per arm, turn 1 through the speculative path, turn 2 a strict
ChatML extension through plain Generate) with the prompt's own three assertions: turn-1 ids equal (losslessness),
turn-2 PrefillReused equal AND equal to len(prompt1)+len(out1) (the property the fixes are for), turn-2 ids equal.

Run twice, per the prompt: once at HEAD (both fixes in — expect everything equal), once with
decoder/blockspec.go reverted to 0e579400^ and decoder/spec_ngram.go reverted to 97615930^ (expect turn 2 to
diverge). See docs/measurements/spec-vs-batching-metal-2026-09-27.md's "CUDA check" Update section for the
results of both runs.
```

## checkTwoTurn.reused

Moved from `cuda/spec_twoturn_test.go` (the comment above `checkTwoTurn.reused`) on 2026-10-09.

```text
checkReused is false only for the turn2-via-GenerateStream variant: decoder/blockspec.go's GenerateStream never
sets Generation.PrefillReused at all (grep confirms it — BlockSpec has no separate prefill phase to report reuse
for in the same accounting sense Model.Generate's resident path uses), so it always reads 0 there regardless of
whether the trailing-token fix is doing its job. That is a reporting gap in a field this call path never
populates, not evidence about the fix — checked once, live (measured 2026-09-28: every N, reused2=0, while the
SAME run's turn-2 ids matched plain's exactly), rather than assumed. The ids check below is what actually
proves the KV is correct for that variant, and it still runs unconditionally.
```

## TestSpecDecode.baseline

Moved from `cuda/specdecode_test.go` (the comment above `TestSpecDecode.baseline`) on 2026-10-09.

```text
Baseline uses the batched path per-token (PrefillLast M=1) — the SAME forward the verify uses,
so losslessness is w.r.t. the batched forward (the decode-step Forward differs at the last ULP
for pos>0, a batched-vs-decode rope/attention numerics gap under investigation).
```

## TestSplitKVGate_measuredGeometries

Moved from `cuda/splitkv_gate_test.go` (the comment above `TestSplitKVGate_measuredGeometries`) on 2026-10-09.

```text
This test exists because the gate was previously a single constant (splitkvMinKeys = 256)
characterized on ONE geometry (qwen2.5-1.5b) by a tight in-process loop, then applied to all
models. e2e measurement showed that constant regressed three of four geometries by up to 18–25%,
and was wrong on its own geometry too. A future simplification back to one constant MUST fail here.
```

## testhooks_helpers_test.header

Moved from `cuda/testhooks_helpers_test.go` (the comment at the top of the file) on 2026-10-09.

```text
These helpers live behind goinfer_testhooks because mustResident calls Model.ResidentForwardForTest, which exists only under that tag; in heavytest_test.go
(tag cuda alone) they broke the kernel-level suite's build, which gate gpu's group 2a compiles without testhooks (found by the 2026-10-07 night run).
```

## fourBLoadOpts

Moved from `cuda/testhooks_helpers_test.go` (the comment above `fourBLoadOpts`) on 2026-10-09.

```text
fourBLoadOpts is the load a heavy test of the default 4B target (~/models/qwen3-4b, int4) uses. It pins the resident
context at 8192, the default before R19 (2026-10-01) raised the candidate to 16384: unpinned, the 4B's KV at ~16k
positions plus the margin fills the 8 GB card, the resident build declines to the CPU path ("default resident context 16029
positions does not fit"), or builds and leaves AttachDrafter no room for the drafter's weights (CUDA_ERROR_OUT_OF_MEMORY). Found by the
2026-10-06 night gate: the gate-gpu heavy tier lost its drafter, block-spec and attention tests to it, and the first one's panic took the rest of the
package's run with it.
```

## mustResident

Moved from `cuda/testhooks_helpers_test.go` (the comment above `mustResident`) on 2026-10-09.

```text
mustResident returns mc's CUDA resident or fails THIS test. The bare mc.ResidentForwardForTest().(*cudaResident)
panics when the build declined (nil interface), and a panic kills the whole `go test` process: the 2026-10-06
night gate lost every test after the first decline to it, which the verdict then listed as 20 FAILs.
```

## TestThetaAB.off

Moved from `cuda/theta_ab_test.go` (the comment above `TestThetaAB.off`) on 2026-10-09.

```text
`off` is in the arm set because a speculation suite in this repo was once found
where no configuration beat running no drafter at all, and that was only visible
because off was a competitor.
```

## TestThetaAB.warmup

Moved from `cuda/theta_ab_test.go` (the comment above `TestThetaAB.warmup`) on 2026-10-09.

```text
WARM-UP, discarded. The first cell after a model load carries JIT, allocator
and cache effects: between two otherwise-identical runs the `off` control —
which no code change here can affect — moved +32.2% on exactly that cell,
putting the noise floor above the effect being measured. One throwaway
generation before any timed cell removes it.
```

## TestThetaProbe_CUDA

Moved from `cuda/theta_probe_test.go` (the comment above `TestThetaProbe_CUDA`) on 2026-10-09.

```text
TestThetaProbe_CUDA measures Theta — the marginal cost of one extra verify node,
in units of one single-token target step — on the cgo-free CUDA resident path.

WHY THIS EXISTS. decoder/spec_adaptive.go says Theta "is the relative cost of one
extra verify node on *this backend* — measure it", ships 0.5 as the batched-CPU
value, and nothing has ever measured it on a GPU backend. The resident path
therefore runs the adaptive depth controller on a CPU constant. The error is in
the conservative direction (it under-drafts), so it costs throughput rather than
correctness — but on a verify that streams the weights ONCE for the whole block,
the marginal node should be far cheaper than half a step, and the controller is
plausibly drafting several times shallower than it should.

METHOD, identical to the CPU control in decoder/theta_probe_test.go so the two
numbers are comparable: seed a context of `depth` positions, then time ForwardN
over n tokens for a ladder of n, truncating back to `depth` between every call.
Theta = (least-squares slope of T(n)) / T(1). The CPU control reproduced 0.456 at
depth 128 against the documented ~0.5, which is what licenses trusting this one.
```

## TestTopKSelect_matchesReference.timing

Moved from `cuda/topk_test.go` (the comment above `TestTopKSelect_matchesReference.timing`) on 2026-10-09.

```text
Informational: the per-token cost the decode loop pays after a forward — launch + sync + readback,
no upload — at the real vocab, on a NORMAL logits row. The row is uploaded first: the launch-only
hook reads whatever r.logits holds, and after the correctness loop above that is leftover tie-heavy
data that sends the kernel down its slow ordered-gather path (an earlier version of this test timed
exactly that and reported ~320 us).
```

## TestTowerOps_contextVRAM

Moved from `cuda/tower_base_test.go` (the comment above `TestTowerOps_contextVRAM`) on 2026-10-09.

```text
TestTowerOps_contextVRAM measures what a tower base's own CUDA context and kernel modules cost before any weights (S7 on CUDA's fix: the Qwen2.5-VL tower's reserve covered its weights and scratch but the card was
still ~400 MiB short at the first image, a gap that fits a second context). Free VRAM through the driver before and after newTowerOps, and after one tiny launch. A record; asserts only that it can read the figure.
```

## TestRecordUpload_capturesFirstError

Moved from `cuda/upload_error_test.go` (the comment above `TestRecordUpload_capturesFirstError`) on 2026-10-09.

```text
TestRecordUpload_capturesFirstError is the C-08 gate: BuildResident's load-time up* helpers must
record a failed upload into setupErr (the setup job returns r.setupErr, which BuildResident turns
into a decline). Before the fix they discarded gpu.Upload's error with `_ =`, so a failed upload left
a zeroed buffer and the build returned ok=true — a resident that decodes garbage. Device-free: it
exercises the recording contract directly (the seam the executor return-path and backend.go's
`if setupErr != nil { … declined }` depend on), the same shape as the C-24 runJob gate.
```

## TestUploadKV_matchesSequentialForward

Moved from `cuda/uploadkv_parity_test.go` (the comment above `TestUploadKV_matchesSequentialForward`) on 2026-10-09.

```text
TestUploadKV_matchesSequentialForward is UploadKV's FIRST real correctness test (P6b / gap-0,
docs/multimodal.md). Before this, UploadKV had zero non-test call sites and every fake stub
(decoder/*_test.go) ignored its arguments and returned nil — the "no precision/layout
mismatch" claim the hybrid resident-decode design leans on was asserted by doc comments, never
exercised.
```

## TestVisionEncoder_forwardPatchesReleasesDeviceScratch

Moved from `cuda/vision_encoder_leak_test.go` (the comment above `TestVisionEncoder_forwardPatchesReleasesDeviceScratch`) on 2026-10-09.

```text
TestVisionEncoder_forwardPatchesReleasesDeviceScratch is M-18's own gate
(docs/audit-2026-09-10.md): ForwardPatches allocated 19 per-call scratch device buffers plus a
fresh command queue, and released none of them — every call leaked. Mirrors
TestGemma3VisionResidentReal_gate's real-checkpoint setup, but goes straight to
cuda.NewVisionEncoder (bypassing the vision.Encoder wrapper) so this can read r.dev.Context()'s
own MemInfo directly, and calls ForwardPatches repeatedly rather than once — a single call
can't distinguish "leaks every time" from "never releases the very first allocation", both of
which would pass a correctness-only (cosine) gate but not this one.
```

## TestQKNorm_widths

Moved from `cuda/vnorm_test.go` (the comment above `TestQKNorm_widths`) on 2026-10-09.

```text
TestQKNorm_widths sweeps the per-head qk_norm kernel across head widths — 128 (Qwen3), 256
(Gemma 3's max), and 512 (Gemma 4's global head, which NO prior model reached, so this width
has never been exercised for qk_norm). qk_norm and v_norm are the SAME kernel (v_norm reuses
it with nH=0 / unit weight), and both do a per-head RMS reduction over hd on a fixed 128-thread
block: 1 element/thread at 128, 4 at 512. That multi-element reduction path is the suspect for
the two-geometry K=V parity drift (wrong K/V from position 0, compounding through the cache).
Compares BOTH the weighted path (Q/K with a learned norm) and the scale-less path (V) to the
CPU RMSNorm oracle, per element (relative error — a reduction bug or a 2x is visible; cosine
would hide a uniform scale).
```

## TestVNorm_scaleless

Moved from `cuda/vnorm_test.go` (the comment above `TestVNorm_scaleless`) on 2026-10-09.

```text
TestVNorm_scaleless isolates Gemma 4's V-norm (attention_k_eq_v) BEFORE it is wired into
the K=V forward, so a red here is the norm alone — not the skipped projection or the copy
ordering that land with it (the hd=512 lesson: isolate the new primitive first).
```

## TestMain.vramtrace

Moved from `cuda/vramtrace_test.go` (the comment above `TestMain.vramtrace`) on 2026-10-09.

```text
TestMain installs an OPT-IN VRAM sampler for A12: does free VRAM decline monotonically across the
heavy tier (accumulation — something is not freeing), or does it recover after each test with a
high-water mark above the card (a genuine environment limit)?

Go's testing package exposes no per-test hook, and the four tests that fail in-suite do not share
a helper — requireHeavyModel covers 14 call sites and none of them. So the boundaries are taken
from `-v` output and joined to a timestamped sample stream by wall clock.

IT USES cuMemGetInfo, deliberately — the same instrument the whole A-chain used. nvidia-smi would
have been easier to wire from outside the process and would have been a DIFFERENT instrument: the
two disagreed by 852,224 B when A10 mixed them (107,806,720 vs 106,954,752), and the reporting-gap
decomposition only closed with cuMemGetInfo on both sides.

WHAT IT COSTS, stated because it perturbs what it measures: the sampler holds its own context, and
a context reserves ~106,954,752 B (A10's per-context term). Every reading is therefore offset by
roughly that much, and the run has that much less to work with than an untraced one. The SHAPE —
monotonic versus sawtooth — is what this is for, and the shape is unaffected by a constant offset.
Absolute figures from a traced run are not comparable with an untraced one.

Off unless GOINFER_VRAM_TRACE=1, so no ordinary run pays the context or the polling.
```

## TestVsumSplitGateVsReference.amendment

Moved from `cuda/vsum_split_gate_test.go` (the comment above `TestVsumSplitGateVsReference.amendment`) on 2026-10-09.

```text
CRITERION (a) WAS AMENDED BY OWNER DECISION, 2026-09-13, after the S confirmation cell was scored
and before any D7 reference existed. As pre-registered it read `spike HF <= exact HF`, §3's strict
form — which the pre-registration mislabelled "§3.2". S failed it 8 v 7 over 640 positions, a
difference well inside Poisson noise (σ ≈ √7 ≈ 2.6), so the strict count cannot resolve the
question it is asking. It now uses the ceiling task-prefill-gap.md §3.2 specifies and the Metal
pooled gate implements (metal/prefill_gate_ref_test.go, `exact + 2*math.Sqrt(exact)`). The STRICT
result is still computed and printed for every cell, so the amendment is auditable rather than
silent. (b) and (c) are NOT amended: §3.2's noise-aware (b) would be looser than the registered
1.0 pt, and that bar stays.
```

## TestSlidingWindowLongContext

Moved from `cuda/window_longctx_test.go` (the comment above `TestSlidingWindowLongContext`) on 2026-10-09.

```text
WHY A TINY FIXTURE. This test used to drive the real phi3-mini-4k, whose window is 2047: it
needed win+40 = 2087 forwards on BOTH the CPU and the GPU of a 3.8B model, took 15-25
minutes, and so never actually completed — it was skipped in practice and gated nothing. A
gate that cannot finish is not a gate.

The window's SPAN is not a property worth scaling: winStart = max(pos-W+1, 0) is the same
arithmetic at W=16 as at W=4096. testdata/mistral-tiny-window is a seeded 1.9 MB Mistral with
sliding_window=16, so 56 forwards cover the identical logic in ~1s, with ~40 positions PAST
the window where winStart is > 0 and MOVING — which is the whole point. It also closes a
second gap: the README claims Mistral runs GPU-resident, and this is the only place a Mistral
checkpoint is actually run resident.
```

## TestRopePartial

Moved from `cuda/rope_partial_test.go` (the comment above `TestRopePartial`) on 2026-10-09.

```text
TestRopePartial is the kernel gate for partial rotary (rotary_dim < head_dim: GLM, Phi).

It exists because no admitted model reaches that path yet — glm-tiny also needs the shared
expert, which is not built. Landing a kernel with no gate on the reasoning that "the model
test will catch it later" is how dead code rots into a silent bug the day it wakes up, so the
kernel is gated in isolation now, the same way the MoE kernels were gated before their
dispatch existed.

The specific bug this is written to catch: the pre-partial kernel gave each k thread the pair
(d, d+hd/2) and had it store BOTH elements, which covered all hd elements only because
2*(hd/2) == hd. With rhalf < hd/2 the pair threads touch just [0, 2*rhalf), so the tail
[2*rhalf, hd) is never written into the KV cache. Nothing errors — attention simply reads
whatever was in the cache, which for position 0 of a fresh buffer is zeros. A "half the head
dims are silently zero" bug produces plausible-looking logits.
```

## TestQwen35ResidentDecodeRateCUDA.why

Moved from `cuda/qwen35_resident_speed_test.go` (the comment above `TestQwen35ResidentDecodeRateCUDA.why`) on 2026-10-09.

```text
Decode rate, resident vs CPU, for the Gated-DeltaNet hybrid.

This exists because residency was landed as a CAPABILITY and the plan's second kill criterion
is still unanswered: "residency is admitted but not faster" is a real outcome, and the
CUDA-graphs precedent says a 1.01× is a safety improvement mislabelled as a speed one. A
capability with no measurement attached tends to get quoted as a speedup by whoever reads the
hardware matrix next.

READ THE FIXTURE SIZE BEFORE READING THE RATIO. On the tiny fixtures (hidden 64, 4 layers) the
per-dispatch overhead dominates completely — roughly 20 GPU dispatches per DeltaNet layer
against a few microseconds of actual arithmetic — so a ratio below 1 here says nothing about the
27B and everything about dispatch cost. The number that matters comes from a real-WIDTH fixture
(GOINFER_DNET_SPEED_CUDA_CKPT), where the arithmetic is large enough to pay for the dispatches. Both
are reported rather than only the flattering one.
```

## TestPTX_matchesSourcesAndBindings

Moved from `cuda/ptx_consistency_test.go` (the comment above `TestPTX_matchesSourcesAndBindings`) on 2026-10-09.

```text
TestPTX_matchesSourcesAndBindings is PTX freshness without a device or an NVRTC (audit-2026-09-10
G-09). A .cu edit committed without regenerating its .ptx compiles, vets and passes -short CI.
Then at runtime cuModuleGetFunction fails, BuildResident declines, and every family runs on the
CPU; 23c46b1 and 5b44383 each describe that shape. This test is its static guard:

 1. every __global__ a .cu declares is a .visible .entry in its .ptx, with the same parameter
    count and kinds, and the .ptx carries no entry the .cu no longer declares;
 2. every kernel name the production Go code binds is an entry in some embedded PTX.

It reads files only, so it runs in CI's -short cuda job.
```

## TestPTXModules_coverEveryEmbed

Moved from `cuda/ptx_modules_cover_test.go` (the comment above `TestPTXModules_coverEveryEmbed`) on 2026-10-09.

```text
TestPTXModules_coverEveryEmbed holds TestKernelLocalMemoryCensus's module list to what kernels.go
actually embeds (audit-2026-09-10 G-13(b)). The list was written by hand and covered 15 of 22
modules, so the census's "moe_route declares the maximum local memory" precondition was never
checked against the gpt-oss expert-cache path it exists for (gptoss_act.ptx was missing).
TestKernelFMALint_coversEmbeddedPTX closes the same gap for the FMA lint.
```

## tailsBaselinePath

Moved from `cuda/prefill_tails_identity_test.go` (the comment above `tailsBaselinePath`) on 2026-10-09.

```text
R-24 (docs/tasks/task-recompute-audit.md §5): the batched prefill's tail used to download the whole [M, hidden] residual after every pass, though only
ResidualAll reads it (the argmax and all-logits heads read it on the device, and the last-row tails need one row, which they uploaded back). This gate hashes
the output of EVERY tail mode the resident exposes, for each fixture, and compares it with the hashes recorded from the code BEFORE that change
(testdata/prefill_tails_baseline.json). It is device-specific for the same reason keqv_copy_baseline.json is (the PTX is JIT-compiled to the card's own SASS), so
the baseline is compared only on the device and driver that recorded it.

	go test -tags 'cuda goinfer_testhooks' -run TestPrefillTails -v ./cuda/
	GOINFER_TAILS_RECORD=1 go test ...   # (re)writes the baseline: only from code whose tails are the reference
```

## TestPrefillLast_qwen3

Moved from `cuda/prefill_qwen3_test.go` (the comment above `TestPrefillLast_qwen3`) on 2026-10-09.

```text
TestPrefillLast_qwen3 extends milestone 2's bit-identity gate to a QK-NORM family. Qwen3 is the
dense batched lane plus one thing the old guard declined: a per-head Q/K RMSNorm before RoPE.
batched prefill now applies it via qk_norm_batched (M=1 qk_norm + an M dimension), so this asserts
the same three gates as TestPrefillLast_e2e — KV bit-identical (all layers × rows), last-token
logits bit-identical, and 64-token greedy decode byte-identical — on the real Qwen3-1.7B at int4.
Heavy (loads a 1.7B); gated. If it stays green, qwen3 is a validated batched-prefill family.

	GOINFER_HEAVY_TESTS=1 go test -tags cuda -run TestPrefillLast_qwen3 -v
```

## TestPrefillImageChunkRows_defaultAndOverride

Moved from `cuda/prefill_img_test.go` (the comment above `TestPrefillImageChunkRows_defaultAndOverride`) on 2026-10-09.

```text
TestPrefillImageChunkRows_defaultAndOverride pins the default (2048, chosen because
prefillDefaultChunk's own measurement table already measured that width safe on this box — see
prefillImageDefaultChunk's doc comment) and the GOINFER_PREFILL_IMAGE_CHUNK override, mirroring
prefillChunkRows's own env-var behavior (an unparseable/non-positive value is ignored, not fatal
— a typo in a tuning knob must not take a model off the fast path).
```

## TestPrefillPath_matchesPrefillCore.cases

Moved from `cuda/prefillpath_test.go` (the comment above `TestPrefillPath_matchesPrefillCore.cases`) on 2026-10-09.

```text
moe / gemma4moe / non-uniform NO LONGER DECLINE — the MoE FFN runs per row off the batched
residual and geometry is bound per layer. They stay in this list because the property under
test is that the GUARD and the REPORT agree, whichever way they answer; a case that flipped
from declining to batching still exercises that. The names say what the model is, not what
the verdict is, so they do not go stale a second time.
```

## TestGemvShapeSweep.cheap

Moved from `cuda/shape_sweep_test.go` (the comment above `TestGemvShapeSweep.cheap`) on 2026-10-09.

```text
TestGemvShapeSweep sweeps the PRODUCTION forward GEMVs across the geometries the real model
lineup actually produces, instead of the single shape the parity tests pin.

WHY THIS EXISTS. The W4A8 parity test asserts one shape:

	const N, K = 8960, 1536 // FFN shape; K mult of 32

and that shape SATISFIES the precondition the shipped kernel assumed. packWeight guarded
K%32, but the kernel's lanes step in 32-word strides, so the real requirement is on
Kwords = K/8. Qwen2.5-0.5B (hidden 896) gives Kwords = 112, and 112%32 = 16 — the tail lanes
read past the row. K=1536 gives Kwords=192, 192%32 == 0, so the bug was invisible BY
CONSTRUCTION: the one tested shape was the one that could not fail. That cost a real
out-of-bounds read on the 0.5B, which is the model the README's headline number is measured on.

The bar here is EXACT, not cosine. An out-of-bounds read or a bad tail is a wrong ANSWER, not
a numerical drift, so it needs no near-tie threshold — and a threshold would be wrong anyway:
random weights score ~0.94 cosine even when correct (measured independently on CUDA and
Metal), so a cosine bar over synthetic shapes is either too loose to catch anything or too
tight to pass. Same weights, same activations, both sides — the answers must agree to f32
accumulation order, so this compares against the exact CPU reference with a tolerance that
only absorbs float-summation order.

Cheap on purpose: no model loads, ~0.05 s per shape, so the whole sweep is ~1 s. It is the
regression net under the NEXT kernel (CUDA MoE), where shape assumptions bite hardest —
expert count, top-k, intermediate dim, stacked-expert layouts.
```

## TestBlockSpec_twoTurnsMatchPlain

Moved from `cuda/spec_twoturn_test.go` (the comment above `TestBlockSpec_twoTurnsMatchPlain`) on 2026-10-09.

```text
TestBlockSpec_twoTurnsMatchPlain is 0e579400's own CUDA check: qwen3-4b int4 plus the DFlash block drafter, N in
{1, 17, 48} (1 is the seed-only exit — no draft round at all, the shape most likely to expose a stale anchor).
Two spec variants share turn 1 (the drafter's GenerateStream) and differ only in how turn 2 continues: through
plain Generate (the property serve actually depends on — a spec turn followed by an ordinary one), and through
the drafter's own GenerateStream again (spec-into-spec).
```

## TestNgramSpec_twoTurnsMatchPlain

Moved from `cuda/spec_twoturn_test.go` (the comment above `TestNgramSpec_twoTurnsMatchPlain`) on 2026-10-09.

```text
TestNgramSpec_twoTurnsMatchPlain is 97615930's own CUDA check, the same shape on the target alone (no drafter
object to attach — decoder.NgramDrafter{} is stateless) via GenerateNgramSpeculativeAdaptive, serve's own
resident call for `--spec ngram`.
```

## TestSpecNonCopyLane.kinds

Moved from `cuda/spec_noncopy_lane_test.go` (the comment above `TestSpecNonCopyLane.kinds`) on 2026-10-09.

```text
Each kind is (name, question placed BEFORE the document, closing line). The FRESH kinds ask a question the document cannot answer and tell the model to
answer it; a first attempt that put "ignore the text above" AFTER the document was ignored by the 1.5B, which just kept copying the document (identical
acceptance to COPY), so the printed output head is checked for novelty every run.
```

## TestResolveCtxCapFit_shortcuts.moecache

Moved from `cuda/resident_cap_test.go` (the comment above `TestResolveCtxCapFit_shortcuts.moecache`) on 2026-10-09.

```text
Found live 2026-09-15/16: growing ctx starves the expert-slot cache's own claim on the
same free VRAM (resolveCtxCapFit's own doc comment has the measured numbers). A genuine
pin still overrides it below — this only changes the UNPINNED default.
```

## TestResidentLaunchVRAMProbe.skip

Moved from `cuda/resident_vram_probe_test.go` (the comment above `TestResidentLaunchVRAMProbe.skip`) on 2026-10-09.

```text
ITS OWN WORDS ARE A SKIP. "This run says nothing about the probe" is the definition of
could-not-evaluate, and reporting it as FAIL made a correct decline — the resident path
declining and falling back is designed behaviour, logged with its reason — indistinguishable
from a probe that measured something wrong. B8's rule, applied inside a test.
```

## TestResidentLaunchVRAMProbe.setprobe

Moved from `cuda/resident_vram_probe_test.go` (the comment above `TestResidentLaunchVRAMProbe.setprobe`) on 2026-10-09.

```text
Set the probe on the resident directly. The first attempt read an env var into a package-level
var, which is initialised before t.Setenv runs — so it recorded nothing and the guard at the
bottom fired. Nothing about the load ordering matters here: allocSlots is already done, and
every launch is still ahead.
```

## TestSplitKV_bitIdentical.decline

Moved from `cuda/splitkv_bitident_test.go` (the comment above `TestSplitKV_bitIdentical.decline`) on 2026-10-09.

```text
A DECLINE IS DESIGNED BEHAVIOUR, NOT A FAILURE. BuildResident declines when the resident
context does not fit — it logs the reason and falls back to the staged path — and the
unchecked type assertion below turned that into `panic: decoder.ResidentForward is nil`.
Observed in the tier at 1.14 GB of KV against 0.74 GB free. A test that cannot evaluate says
so; it does not panic, and it does not pass either.
```

## TestVsumMechanism

Moved from `cuda/vsum_mechanism_test.go` (the comment above `TestVsumMechanism`) on 2026-10-09.

```text
TestVsumMechanism is R6 step 1 (docs/tasks/red-october.md): why did the V-sum spike's KL against the
f32/f64 reference read 1.0585x exact's on D7@8000 while its argmax metrics were better?
Pre-registered in docs/measurements/vsum-mechanism-PREREGISTERED.md; the hypotheses, the numeric tests
and the decision rule are there and are not restated here.

It re-uses Phase A's cached reference rows and re-scores arms only: exact (skVsumSplit=0) and the spike
at each S in GOINFER_VSUM_MECH_S (default 1,2,4,8,16). S=1 must be BIT-IDENTICAL to exact — it is the
built-in defect detector, because a one-chunk "split" is the same fold in the same order.

	GOINFER_HEAVY_TESTS=1 GOINFER_SPLITKV_VSUM_SPLIT=16 \
	  go test -tags 'cuda goinfer_testhooks' ./cuda/ -run TestVsumMechanism -v -timeout 2h

GOINFER_SPLITKV_VSUM_SPLIT must be >= max(S): it sizes the partials buffer at backend setup.
```

## TestPrefillCoreAndDraftBlockCallTheShmemGuards

Moved from `cuda/shmem_limit_test.go` (the comment above `TestPrefillCoreAndDraftBlockCallTheShmemGuards`) on 2026-10-09.

```text
TestPrefillCoreAndDraftBlockCallTheShmemGuards is the wiring half the tests above cannot cover:
they drive checkPrefillShmem/checkDrafterShmem directly, which proves the guards are correct but
says nothing about whether the real call sites still invoke them. Caught in practice while
mutation-testing this fix: removing prefillCore's call to checkPrefillShmem still builds clean
(an unused METHOD is not a Go compile error the way an unused import or local var is) and every
unit test above still passes, because none of them go through prefillCore/DraftBlock at all.
Asserted structurally, the same way TestStreamTokens_decodesAsAContinuation and
TestWebUI_rootRouteIsUnauthenticated pin their own wiring.
```

## specPagerPrompts

Moved from `cuda/spec_pager_interaction_test.go` (the comment above `specPagerPrompts`) on 2026-10-09.

```text
specPagerPrompts is realistic traffic, spanning prose / code / math.

scripts/prompts.json is deliberately NOT used. It is four-unique-word filler ("the the the …"),
and on a MoE every such position routes to the same experts — the pager never has to stage
anything, so the effect under test cannot appear and the run would manufacture a null. That
confound has already produced one wrong profile in this tree (the mellum2 prefill split, which
contained no MoE frames at all).
```
