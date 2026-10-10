# metal: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `metal`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

## batchMaxSeqs

Moved from `metal/batch.go` (the comment opening the file, above `batchMaxSeqs`) on 2026-10-09.

```text
MC3 (docs/tasks/task-concurrency-2026-09.md): batched decode — one decode token for up to batchMaxSeqs sequences in
ONE command buffer, each sequence on its own MC1 resident KV slot at its own position, and every logit bit-identical
to production's single-token forward on that sequence (TestMC3Step_bitIdentical). Everything that is per-row work
runs production's own kernels, per sequence: norm + int8 quantisation, RoPE, the KV store into that sequence's slot,
attention over that slot (attention_fa at or above attnFADepthFloor, exactly as a single-token step would plan it),
ctx quantisation, SwiGLU + quantisation, the final norm. The five matmuls run once for all sequences, on the
matrix units, reproducing production's arithmetic:
  - W4A8 projections (mc3_bt: qkv +bias, o +residual, gate|up; mc3_btd: down +residual): an 8-token fragment column,
    weights built in registers through thread_elements(), each 32-k group's integer sum exact in the f32 fragment
    (nibble-8 and the int8 activations are exact in half), accumulated per lane residue in the decode GEMV's lane
    order and reduced by simd_sum's own xor tree 1, 2, 4, 8, 16 (batchSimdSumTreeOK checks that tree at build);
  - the int8 LM head (mc3_lm): slab integers converted and summed in int32 — the same integer in any order.
Measured: docs/measurements/concurrency-mc3-s0-2026-09-26.md (kernels), concurrency-mc3-s1-2026-09-26.md (in sequence:
B = 4 at 1.73–1.84x one stream's aggregate; B = 1 loses, so the decoder calls this at B >= batchMinSeqs only).
```

## batchMinSeqs

Moved from `metal/batch.go` (the comment above `batchMinSeqs`) on 2026-10-09.

```text
batchMinSeqs is the smallest batch the decoder should send here. At B = 1 the step costs 1.6–1.95x a production
token (the fragment's cost does not shrink with the batch); B = 2 measured 0.98–1.10x (S1), about even, and it
spares the per-token KV-slot rebinding two interleaved single-token decodes would pay.
```

## mc3FARowsOn

Moved from `metal/batch.go` (the comment above `mc3FARowsOn`) on 2026-10-09.

```text
mc3FARowsOn runs the batched step's rows at attention_fa depth through one multi-row dispatch pair (E-P05,
mc3_attention_fa*_rows), bit-identical to their per-row pairs. ON since 2026-10-04 (owner: "turn E-P05 on"): faster in
every cell measured, 1.016x at 2 rows (serve's default slot count) to 1.06x at 8 rows on the 1.5B at depth 1100-2048,
1.029x on the 7B (docs/tasks/task-metal-audit-2026-10.md, "E-P05").
```

## mc3AdjRowsOn

Moved from `metal/batch.go` (the comment above `mc3AdjRowsOn`) on 2026-10-09.

```text
mc3AdjRowsOn runs the batched step's per-row qkv and gate|up GEMVs (the sizes calibrateRows gives to per-row) as one
dispatch with each weight tile's B threadgroups adjacent (E-P02, mc3_gemv_w4a8_sa_{bias_}rows_adj), bit-identical to
the B per-row dispatches. ON since 2026-10-04 (owner: "turn ... E-P02 on"); its kernel A/B read 1.11-1.15x on the 7B's
gate|up at B = 2-4 and 1.01-1.04x on the 1.5B's.
```

## calibrateRows

Moved from `metal/batch.go` (the comment above `calibrateRows`) on 2026-10-09.

```text
calibrateRows decides, once, at which batch sizes qkv and gate|up are cheaper as B per-row production GEMVs than as
one fragment dispatch (MC3 S4, docs/tasks/task-concurrency-2026-09.md). The fragment's cost is fixed for any B up to
8, and a GEMV's is per row. On the 7B's large shapes the GEMV is bandwidth-bound and the fragment ALU-bound, so two
GEMVs beat one fragment (gate|up 0.87 against 1.17 ms). Both run on the real weights of the first layers, one layer's
weights per dispatch so the cache does not serve them, median of 7 command buffers after 2 warm-ups. The choice
changes speed only: the per-row path is production's own kernel on each row.
```

## stepVerifyCost

Moved from `metal/batch.go` (the comment above `stepVerifyCost`) on 2026-10-09.

```text
stepVerifyCost is the step-kernel verify's measured cost in single-token steps, indexed by rows verified: the
conservative end of TestMC3Verify_rowCost on the M1 Pro (the 7B at depth 2048, the dearest cell of both models; the
1.5B at depth 128 reads 1.59 / 1.71 / 1.76 at 2 / 4 / 8 rows). 1 row is production's own Forward; 3 and 5-7 are
interpolated; 9 rows are an 8-row step plus one Forward. docs/tasks/task-concurrency-2026-09.md, MC4.
```

## promptStepMaxAboveFloor

Moved from `metal/batch.go` (the comment above `promptStepMaxAboveFloor`) on 2026-10-09.

```text
promptStepMaxAboveFloor is the longest suffix the above-floor half of the step route takes (promptStepAboveFloor).
T1.10 measured the step at 0.40× / 0.80× / 1.56× the pass at K = 16 / 32 / 64 on the 1.5B and the 7B
(docs/tasks/task-metal-audit-2026-10.md, Batch A), so it stops at the last K measured faster.
```

## promptStepOK

Moved from `metal/batch.go` (the comment above `promptStepOK`) on 2026-10-09.

```text
promptStepOK reports whether PrefillLast takes the step route for n tokens at startPos (E-P01, audit-metal-2026-09-30):
the resident runs the step-kernel verify (VerifyCost: the batched step exists, and no logit transform separates its
rows from Forward's), and the prompt ends below the fast-prefill floor, where the batched pass declines and the
decoder would run the sequential loop. The step's rows are that loop's bits, so this route changes no output.

Above the floor the step is faster than the pass up to K = 32 too, but there it would replace the pass's numerics with
decode's: promptStepAboveFloor, off. A short suffix there is also a chunked prefill's tail, and on the step it differed
from the whole pass the chunks must equal (TestMC5_prefillChunkInvariance's C = 81 case: 20480 of 20480 logits).
```

## prefillByStep

Moved from `metal/batch.go` (the comment above `prefillByStep`) on 2026-10-09.

```text
prefillByStep is PrefillLast on the step kernels (E-P01): the prompt's positions run as consecutive rows of the bound
slot's sequence, batchMaxSeqs at a time, exactly as PrefillLastNArgmax runs a verify, and the last row's logits come
back. forwardMulti is bit-identical to production decode for such rows (TestMC3Verify_sameSlotRowsBitIdentical,
TestMC3Step_promptInRowsBitIdentical), so this is the sequential path's numerics at about a quarter of its time
(T1.10: 0.22× on the 1.5B, 0.25× on the 7B). Pieces before the last run no head at all (R-19, multiNoHead), and the
last piece copies out only its last row (multiLastRow); a one-row last piece runs production's own Forward.
Cancellation is checked between pieces.
```

## expertPool

Moved from `metal/expertpool.go` (the comment above `expertSlot` (it described the pool and the slot view together)) on 2026-10-09.

```text
expertPool is a bounded per-layer LRU pool of N expert slots for SYNCHRONOUS Metal MoE paging.
The gemma4-26b full expert set is 11.96 GB (per-expert W4A8 ≈ 3.19 MB × 128 × 30 layers) — it does
not fit resident, so each MoE layer keeps only N experts on the GPU and stages the routed top-k in
from host (mmap-backed) bytes on demand, evicting the least-recently-used slot. One pool per MoE
layer (experts differ per layer). This is the CPU expertPager (decoder/moepaging.go) analogue for
the Metal backend — the paging half of the synchronous path; the per-layer submit+wait boundary
that reads the router idx before staging lives in the encode path.

Staging is a host→shared-buffer copy (UMA). Eviction is BOOKKEEPING ONLY: it frees a slot for
reuse, it does NOT return pages to the OS (Darwin mmap.Advise(_,false) is a documented no-op) — so
resident footprint must be MEASURED, not inferred from N × per-expert. See
[[metal-moe-paging-needs-speculation]] and the Step-6 budget probe (paging_budget_test.go).

M-11 (audit-metal-2026-09-12.md): slots live in ONE contiguous Buffer per field (guW/guS/dW/dS),
not N separately-allocated Buffer objects. A separate object per slot forced the paged phase-2
encoder to pick a specific Buffer identity at ENCODE time — after staging, which is exactly why
the paged forward could not be pre-encoded into one command buffer with the rest of the token
(the blocker M-11 names). A contiguous pool lets phase 2 bind the SAME buffer identity every
call and read a slot NUMBER at kernel-execution time instead — the same trick the non-paged
stacked-all-E path already uses via rIdx (see moe.go's gemv_w4a8_moe: "idx[slot]*rowsPerExpert").
expertSlot is now a lightweight VIEW into that contiguous storage (Buffer.At()-offset), valid for
GPU-side binding and for gpu.Upload (both respect Buffer's bind offset) but NOT for U32s()/U16s()
(those always read from the buffer's base, ignoring the offset — see preadIntoPoolSlot's doc
comment for why the pread path does its own offset arithmetic instead of relying on a view).
```

## copyBytesToU32Buf

Moved from `metal/expertpool.go` (the comment above `copyBytesToU32Buf`) on 2026-10-09.

```text
copyBytesToU32Buf memcpys little-endian nibble bytes into a uint32 slot buffer's shared contents,
via gpu.Upload rather than a hand-rolled unsafe.Slice reinterpret (N-33, audit-metal-2026-09-12.md):
the source is an unaligned mmap span, which gpu.Upload's byte-slice copy handles just as the old
reinterpret did (a *uint32 alias of it would be misaligned UB — measured 73% of expert spans are
not 4-aligned), but gpu.Upload also bounds-checks src against the buffer's actual allocated bytes
instead of Go's `copy` silently truncating an oversized src with nothing to say so. On LE this
yields the exact words bytesToU32 would build, so paged ≡ non-paged byte-identity is preserved.
Panics on error: every call site's dst is sized for exactly this src by construction (newExpertPool
/ the stage() contract), so a failure here is an invariant violation, not a runtime condition to
recover from.
```

## distinctExperts

Moved from `metal/expertpool.go` (the comment above `distinctExperts`) on 2026-10-09.

```text
distinctExperts is every expert id this pool has EVER staged, across its whole lifetime —
unlike coldStarts (caps at N once the pool fills) or where (only currently-resident), this
answers "how many distinct experts did this layer's routing actually touch" — the floor an
expert-major (route-then-group-then-stage-once) prefill could hit, vs. stages, which also
counts every re-fetch of a previously-evicted-then-needed-again expert (M-05 investigation,
audit-metal-2026-09-12.md). Negligible memory (≤ nE entries); left in production code as
always-on telemetry, same as the counters above.
```

## stagePread

Moved from `metal/expertpool.go` (the comment above `stagePread`) on 2026-10-09.

```text
stagePread, when set (GOINFER_MOE_PREAD=1 on a .giw-mmap'd model), REPLACES the mmap byte-copy:
it preads expert e's nibbles straight into slot s's unified-memory buffers — one syscall, one
large sequential read, zero page faults (cold pread measured 3687 MB/s vs the mmap demand-fault's
375 MB/s, 9.8×). Fetch and copy collapse into the single read. nil ⇒ the mmap byte-copy path.
```

## ensureResidentBatch

Moved from `metal/expertpool.go` (the comment above `ensureResidentBatch`) on 2026-10-09.

```text
ensureResidentBatch is ensureResident generalized over a whole layer's routed top-k at once
(M-12, audit-metal-2026-09-12.md): slot selection and eviction bookkeeping run SEQUENTIALLY
first — the LRU/where/slotExpert state is not safe for concurrent mutation, and two misses in
the same batch must never be handed the same slot — but the actual staging I/O for every miss
then runs CONCURRENTLY. Serial staging at queue depth 1 was the measured bottleneck (per-miss
cost RISING with N, the signature of latency-bound cold reads with one outstanding request);
pread on a shared fd into disjoint destination buffers is safe (each syscall is positional —
it never moves a shared file offset), and the mmap byte-copy path's per-expert source/dest
spans are equally disjoint, so there is nothing here that needs the calls serialized.

A duplicate id within one call (the router selecting the same expert into two of its top-k
slots — not expected, but not assumed impossible either) resolves to the SAME already-picked
slot rather than staging it twice or racing two goroutines over one slot's buffers.

Callers must still serialize ACROSS calls exactly as before (ensureResident's own doc comment:
"the paged forward must submit+wait at each layer before calling this") — this only parallelizes
the I/O WITHIN one call, on the single host goroutine that already owns this pool exclusively.
```

## mc3RowsKernels

Moved from `metal/batch_rows.go` (the comment opening the file, above `mc3RowsKernels`) on 2026-10-09.

```text
MC3 S3 (docs/tasks/task-concurrency-2026-09.md): multi-row forms of the per-row kernels a batched step runs once per
sequence — rmsnorm_quant, quant_vec, swiglu_quant, rope2. Measured (TestMC3StepBreakdown, 2026-09-27): those small
dispatches cost ~1 ms per sequence per step (4.2 of a B = 4 step's 22.6 ms at depth 128, 8.1 of B = 8's 30.1), each
running a single threadgroup, one sequence after another. One dispatch over all rows runs them side by side.

Their BODIES are production's, byte for byte: each variant is derived from allKernels' own source at init, by
renaming the pointer / per-row parameters in the signature and adding a prologue that points them at the row this
threadgroup (or thread) serves. Nothing in a body is retyped, so a variant cannot drift from its kernel — these
kernels round differently when their code shape changes (see rmsnorm_quant's own notes), which is why they are not
hand-copied — and a production signature edit panics here at init instead of silently diverging. They compile into
the resident's main library (same fast-math setting as the originals).
```

## buildDeltaNetLayer

Moved from `metal/deltanet.go` (the comment above `buildDeltaNetLayer`) on 2026-10-09.

```text
Name the missing tensor — see cuda/backend.go's identical check and comment: an empty slice
here becomes a 0-byte device upload, which fails as "invalid length" with no indication of
which of the four small tensors was the culprit. Two different checkpoints have already
failed exactly that way during the CUDA bring-up.
```

## deltaNetKernels

Moved from `metal/deltanet_kernels.go` (the comment above `deltaNetKernels`) on 2026-10-09.

```text
deltaNetKernels — the Gated-DeltaNet decode mixer (Qwen3.5/3.6-MoE, Qwen3-Next, Qwen3.8),
ported verbatim from cuda/deltanet.cu (read its header comment first; this file mirrors it
kernel-for-kernel and keeps the same five-stage split so the CPU capture hook
(decoder.deltaCapHook) gates each stage the same way on both backends).

WHAT IS NEW HERE VS THE OTHER TWO PORTS. WebGPU reused its Mamba-2 engine for the causal conv
and the state plumbing, so only the delta rule was new there. Metal has no recurrent-state
kernel of any kind before this file (same starting point CUDA was in), so every kernel below is
new code, gated from its own input via the capture hook's mixed slot — not just delta_rule.

THE STATE IS STORED TRANSPOSED RELATIVE TO THE CPU, same as CUDA. decoder/deltanet.go holds S
as [hk, hv] and walks it column-wise with stride hv, in two passes. Here S is [hv, hk], so
thread (headV, vd) owns a contiguous row S[headV][vd][0:hk] and reads it stride-1.

delta_gnorm IS NOT the Mamba gated norm. Mamba normalizes the gated product; DeltaNet
normalizes the recurrence output and gates AFTERWARDS. Substituting one for the other measured
cosine 0.986 with a 12x RMS error on the WebGPU side — a plausible tensor of the right shape
and the wrong values.

THE L2-NORM EPSILON (delta_norm) IS A ZERO-GUARD, NOT A PRECISION KNOB: sqrt(1/(ss+1e-6)),
never rsqrt(ss). silu(0) is exactly 0, so an all-zero head is reachable, and rsqrt(0) is +inf,
poisoning every downstream state entry with NaN where the reference yields a finite scale on a
zero vector. This does not show in the chained-drift gate (1e-6 is below f32 resolution at
normal magnitudes) — TestDeltaNorm_zeroHead exists specifically to catch it.

FMA DISCIPLINE. Metal has no per-operation explicit-rounding intrinsics the way CUDA's
__fmaf_rn/__fmul_rn/__fadd_rn does; the whole kernel library compiles under one library-wide
fast-math setting (metal/model.go's preciseMathCompile). fma() below is used ONLY where the
CUDA reference explicitly fuses (__fmaf_rn); every site the CUDA reference deliberately leaves
unfused (__fmul_rn followed by a separate __fadd_rn) is written here as two separate
expressions, mirroring the CUDA source's literal shape rather than a mathematically-equivalent
rewrite — see docs/completed/task-metal-batched-verify-kernel.md on why literal source form, not just
arithmetic equivalence, is what fast-math contraction keys off.
```

## Device

Moved from `metal/device.go` (the comment above `package metal`) on 2026-10-09.

```text
Metal device layer — now aikit's native-GPU substrate (github.com/townsendmerino/aikit/gpu),
lifted verbatim from what used to be this package's metal.go. goinfer keeps its tuned kernels
here and builds them on these device types — the GPU analogue of the linalg relationship. Only
the device TYPES moved; nothing about the decode path changed, so it must stay bit-identical
(the Metal device-parity suite is the tripwire).
```

## NewBufferFloats

Moved from `metal/device.go` (the comment above `NewBufferFloats`) on 2026-10-09.

```text
Thin re-wraps of aikit gpu v0.29.0's type-suffixed-Buffer-API collapse (NewBufferFloats/
NewBufferInt8/NewBufferU32/NewBufferUint32s/NewBufferU16s deleted in favor of the generic
NewBufferOf[T]). Go has no generic methods, so the aikit replacement is a free function
(gpu.NewBufferOf(d, data)); these keep every one of this package's ~500 existing call sites at
their original method-call shape (now a free function taking d first) instead of touching each
one's argument list.
```

## attnGeom

Moved from `metal/geom.go` (the comment above `attnGeom`) on 2026-10-09.

```text
attnGeom is one distinct per-layer attention geometry, shared by every layer that has it.
It exists because Gemma 4's own-forward residency (9c) interleaves two attention shapes in
one model — local (head_dim 256, kv_heads 8, full rotary) and global (head_dim 512, kv_heads
2, partial rotary, K=V) — so geometry can no longer live model-level on *resident. A launch
site that read r.hd/r.nKV/... would silently run every layer at the model-level (uniform)
shape; those fields were REMOVED so binding the wrong source is a compile error, not a review
catch. On a uniform family every layer resolves to the SAME geometry, so geomFor dedups by
value and the whole model shares one attnGeom — byte-identical to the old model-level path.

nH (query heads) is LOAD-BEARING as a model-level field: it is NOT part of the geom key
because it is constant across a family's layers (Gemma 4 varies head_dim and kv_heads, not the
query-head count — 16 in both the 12B and 26B variants), so it stays on *resident. A future
family with PER-LAYER query-head counts would break this: it would have to move nH onto attnGeom
and add it to the key. This comment is the marker that stops the next person bisecting for it.
nHhd = nH*hd DOES vary with hd, so uNHhd is derived per-geometry here. kEqV joins the key
because a K=V layer's V-store behaviour differs from a v_proj layer's even at identical dims
(9c Step 3: V = v_norm(raw k), its own cache) — two such layers must not share a geom.

This mirrors the WebGPU bridge's value-keyed geometry dedup (gpu/, commit 9ec363f): one
geometry object per distinct {hd, nKV, half, kEqV}, shared across the layers that request it.
```

## chainDrainEvery

Moved from `metal/greedy_chain.go` (the comment above `chainDrainEvery`) on 2026-10-09.

```text
The greedy chain: C-B01 of docs/audit-metal-2026-09-30.md (Phase 3 item 5 of docs/tasks/task-metal-audit-2026-10.md).

The pipelined executor (execLoop) pre-encodes token t+1 while t runs, but cannot commit it until t finishes: t+1's
input is the embedding of the token t's logits pick, and the host picks it. So between t's last kernel and t+1's
first the GPU waits for the host: completion, 608 KB of logits, the decoder's argmax, the embedding, the commit
(T1.3 measured that idle gap at 0.58-0.66 ms per token on the 0.5B and 1.5B). For greedy decoding the pick needs no
host: each chained command buffer ends with the fused argmax head ForwardArgmax uses (gemv_w8a8_amax then
argmax_finish, writing a 4-byte id) and starts with embed_gather_i8, which writes that id's embedding into r.x from
the int8 embedding table on the device (chainEmbedTable: the LM-head buffers for a tied head, else a copy made on
first use). So t+1 can be committed before t finishes, and the queue runs it right after.

In flight: chainNext commits the buffer after the newest before it waits for the oldest, so the next token is always
queued when the host waits, and one buffer is outstanding between calls. Token t binds uniform set t%2 and writes its
id to chainTok[t%2]; t+1 reads chainTok[t%2] and writes chainTok[(t+1)%2]. Buffers are hazard-tracked (aikit's
default) and the encoder is serial, so Metal orders a buffer's reads behind the previous buffer's writes; the host
rewrites set k only for token t+2, after it has waited for token t, the set's last user. Every chainDrainEvery
buffers the chain waits with nothing queued and drains its autorelease pool (one gap), as execLoop does.

Bit-identical to the full-logits path by construction: the same trunk dispatches with the same values per token, the
argmax ForwardArgmax uses (first maximum wins, as the decoder's), and the host embedding's arithmetic (Embed.Row's
float32(q)*scale). TestGreedyChain_bitIdentical checks it. A chain stopped after an EOS has already run one forward
past it; the K/V it wrote sits past the generation's end and is overwritten before anything reads it.
```

## pagedFenceOn

Moved from `metal/paged_fence.go` (the comment above `pagedFenceOn`) on 2026-10-09.

```text
pagedFenceOn: a paged MoE layer's phase 1 ends with paged_fence, and the host spins on its word instead of sleeping in
waitUntilCompleted (M-11 / C-B03, docs/audit-metal-2026-09-30.md). aikit's C-B03 probe measured the spin waking about
88.5 us sooner per boundary (p50 101.8 against 190.3 us) on a one-word payload. OFF: on M26 (phase 2 async on both
arms) it read 0.878x a token with a busy spin and 0.848x with a 10 us nap, 0 of 9 reps above 1 each, every boundary
fenced and none stale; each fenced phase 1 took about 1.3-1.6 ms to be seen. The likely cost is the system-scope fence
publishing a whole decode layer's writes, which the probe's one word never priced (docs/tasks/task-m26-mac-2026-10.md,
"M-11"). Kept, gated bit-exact and not stale (TestPagedFence_bitExactAndNotStale).
```

## pagedFenceSrc

Moved from `metal/paged_fence.go` (the comment above `pagedFenceSrc`) on 2026-10-09.

```text
pagedFenceSrc follows aikit's C-B03 probe (gpu/metal_fence_probe_test.go, after MLX's kernels/fence.metal): the system
thread scope is not public MSL, so it is built from the compiler's __METAL_MEMORY_SCOPE_SYSTEM__. The router wrote its
ids with plain stores in an earlier dispatch of the same serial encoder; a system-scope fence does not publish
non-coherent stores (the probe saw such a payload stale on 47 of 50 boundaries), so this kernel copies the ids through
coherent(system) stores into a mirror the host reads, fences, then stores the sequence number the host spins on.
```

## dnetPrefillOn

Moved from `metal/prefill_deltanet.go` (the comment above `dnetPrefillOn`) on 2026-10-09.

```text
dnetPrefillOn admits Gated-DeltaNet models to the batched prefill pass. ON since D-B01's grade (2026-10-04,
docs/tasks/task-metal-audit-2026-10.md: fidelity SHIPS on Qwen3.5-0.8B, the pass 8.06x the sequential loop at
K = 512); tests turn it off for the sequential arm.
```

## emodelLayerMajorOn

Moved from `metal/prefill_emodel.go` (the comment above `emodelLayerMajorOn`) on 2026-10-09.

```text
emodelLayerMajorOn routes a Gemma 4 E-model's prompt (E2B, E4B: per-layer embeddings, KV-shared layers) through
prefillEModel instead of declining PrefillLast (S9 of docs/tasks/task-multimodal-support-2026-10.md). The f16-MMA pass
declines every dense Gemma 4 (head size varies by layer), so without this a text prompt prefilled one token at a
time on the GPU and an image turn prefilled on the CPU. The pass is bit-identical to the sequential loop by
construction (G-S9a); on by default, as the paged 26B's layer-major pass is, with a night speed grade to turn it off
below 1.00x.
```

## emodelBatchedOn

Moved from `metal/prefill_emodel.go` (the comment above `emodelBatchedOn`) on 2026-10-09.

```text
emodelBatchedOn routes a Gemma 4 E-model's prompt through the f16 batched pass (S9 step 2,
docs/tasks/task-multimodal-support-2026-10.md) instead of the layer-major one, at or above the fast-prefill floor. ON
since 2026-10-09: G-S9c and G-S9d passed by day, and the night speed rule read 7.04x on a ~512-token text prompt
(10.99 -> 1.57 s TTFT, five passes 6.84-7.42x). Below the floor, on any decline, or with --exact-prefill, the
layer-major pass (bit-identical to sequential) keeps the route.
```

## g4ExpertBatchOn

Moved from `metal/prefill_g4batch.go` (the comment above `g4ExpertBatchOn`) on 2026-10-09.

```text
D-P01's batched expert GEMM (docs/audit-metal-2026-09-30.md; docs/tasks/task-m26-mac-2026-10.md, "D-P01"): a
layer-major paged Gemma 4 prefill runs one slot group's routed experts expert by expert, every (row, expert) pair the
group routes to that expert in one pass over its weights, instead of row by row. Bit-identical to the per-row phase 2
(encodeG4Phase2Paged) by construction:
  - gate|up and down run moe_batch_gemv, which computes sa_rows_acc's per-row sums for up to B activations at once:
    each (activation, weight row) accumulator sees exactly sa_rows_acc's operations in its order, and the weights are
    read once for all of them;
  - SwiGLU and its int8 quantisation run mc3_swiglu_quant_rows, swiglu_quant's body per pair;
  - the down projection stores each pair's raw sum, and moe_batch_combine adds a row's k experts in route order with
    the per-row path's own epilogue, fma(wgt*acc, asc, out) from zero. The floating-point order is the per-row one.

g4ExpertBatchOn picks it in g4LayerMajorRows. ON once its gate passes and its speed read clears the owner's bar.
```

## g4LayerMajorOn

Moved from `metal/prefill_g4paged.go` (the comment above `g4LayerMajorOn`) on 2026-10-09.

```text
g4LayerMajorOn routes a paged Gemma 4 MoE's prompt through prefillG4Paged (docs/tasks/task-m26-mac-2026-10.md, 4b)
instead of the sequential loop. ON since 2026-10-04 (owner's bar, ship at >= 1.02x for a bit-identical change; a smoke
on M26 read 1.42x); the night grade at M = 128 and 512 turns it off if it reads below 1.00x.
```

## loraQGateGap

Moved from `metal/lora.go` (the file comment opening the file) on 2026-10-09.

```text
Compute-time LoRA on the resident path (G3, docs/tasks/task-gpu-paths-2026-09.md).

A LoRA adapter's delta is additive: y[o] += scale·Σ_r B[o,r]·(A·x)[r], applied AFTER the base
projection's matmul, on the SAME input x the base matmul consumed (decoder/lora.go's
applyLoRA, the CPU reference this mirrors). On Metal that input is always the already-quantized
int8 activation buffer feeding the base GEMV (r.aq/r.aSc for q/k/v, r.cq/r.cSc for o, r.mq/r.mSc
for gate/up, r.dq/r.dSc for down) — reusing it means no extra dequantize-and-requantize pass,
and it is what the base weight itself sees, so the two projections agree on precision.

Scope: Model.LoadAdapter (decoder/lora.go) only builds a runtime for the generic dense forward
(rejects MoE, own-forward archs, and the non-gated MLP layout), so encodeAttention/encodeLayer's
plain q/k/v/o/gate/up/down sites are the complete surface this needs to cover — MoE/DeltaNet/
GPT-2 branches never see a non-nil loraLayers because no adapter can be loaded for them.

NOT covered yet: the qGate branch (Qwen3.5-style fused double-width q_proj+gate) in
encodeAttention — LoadAdapter does not exclude that family by name, only by the three checks
above, so an adapter loaded against a qGate family would silently apply no q/k/v delta on Metal.
No family with qGate=true is known to pass LoadAdapter's checks today; if one arrives, this is
the gap to close first.
```

# Test files

Moved from the comments of the package's `_test.go` files (CC5); same conventions as above.

## TestWeightAlias_olderBundleTakesCopyPath

Moved from `metal/alias_fixtures_test.go` (the comment above `TestWeightAlias_olderBundleTakesCopyPath`) on 2026-10-09.

```text
A bundle for a target other than Metal must still load with aliasing on: its 16-aligned singles alias, its fused groups
and the scales Metal cannot bind in place take the copy path, the logits stay bit-identical, and the banner says why and what
fixes it. A current metal-target file of the same model carries no such note, and the anonymous figure it reports is the small
remainder the format does not cover.

WHICH FORMAT THE "OLD" BUNDLE IS (F-D02, audit-metal-2026-09-30.md): this test used to call it a v12 file. It is built here by
today's prequant with GIWTargetNone, and the writer emits weights format v15 for every target, so it is a v15 NON-METAL bundle:
its int4 scales are stored as binary16 (v15) but not in the kind-7 / fused-group layout Metal binds, so they are converted into
a new buffer. The test now reads the version from each file's header and asserts it, so the label cannot go stale again.
```

## TestAliasForkProbe

Moved from `metal/alias_forkprobe_test.go` (the comment above `TestAliasForkProbe`) on 2026-10-09.

```text
TestAliasForkProbe is the decisive experiment for the M26 alias collapse
(docs/measurements/s6-alias-2026-09-24.md, "The M26 alias arm collapses"): does a fork() of a process
that holds a GPU-wired no-copy buffer over a PROT_READ|MAP_PRIVATE file mapping copy the WHOLE mapping
eagerly? XNU says it must — IOPL wiring faults each page with write intent (COW copy into a wired shadow,
vm_pageout.c vm_object_iopl_request), marks the object true_share, and vm_map_fork then routes the entry
to slow_vm_map_fork_copy → vm_object_copy_slowly of the entire entry (vm_map.c, vm_object.c). The serving
swap guard forks every 2 s (exec of `sysctl`), so on a 15 GB mapping that copy is the collapse.

No model, no server. A fresh 2 GiB file of random bytes written with F_NOCACHE (uncached, so page-ins are
countable), mapped exactly as decoder.Load maps a .giw (aikit mmap.MapReadOnly), ONE no-copy buffer over
a 16 MiB page-aligned window mid-file, one dispatch touching one float per page of THAT WINDOW ONLY, and
fork+exec(/usr/bin/true) timed: before the mapping (t0), after the buffer exists but before any GPU touch
(t1), three times after the touch (t2a-c), and after the buffer is released and the file unmapped (t3).
An external shell samples vm_stat every 50 ms throughout, so a transient anonymous copy made during a
slow fork is seen even though the child frees it at exec.

Readings: t1 ≈ t0 and wired +≈1,024 pages at the touch, then t2 = seconds with system page-ins ≈ the
file's 131,072 pages and a transient Anonymous rise of the same size ⇒ H1 (window-precise wiring; fork
copies the whole entry) — the fix is a fork-free swap guard (and/or VM_INHERIT_NONE on the mapping).
wired +≈131,072 pages at the touch ⇒ H2 (the driver wires the whole region). t2 ≈ t0 ⇒ neither; the
M26 collapse needs another explanation. t3 ≈ t0 closes the loop: the wired buffer was the cause.
```

## TestAliasForkProbe.p2

Moved from `metal/alias_forkprobe_test.go` (the comment at the P2 step) on 2026-10-09.

```text
P2: the GPU touches one float per page of the window only. NOTHING forks between the touch and the
first timed fork (vm_stat is itself a fork+exec — the first version of this probe called it right
after the touch and that untimed fork paged in the whole file; see the record).
```

## BenchmarkAttention

Moved from `metal/attention_bench_test.go` (the comment above `BenchmarkAttention`) on 2026-10-09.

```text
BenchmarkAttention isolates decode's attention kernel (metal/kernels.go, model.go: r.pAttn,
launch shape (nH*128, 128) one threadgroup per Q head) at a realistic long-context depth --
the V-read loop (`for(uint s=winStart;s<nKeys;s++) a += sc[s]*float(vb[s*kvDim+d])`) is the
per-token cost this isolates; qk-scoring/softmax are the same kernel but the V-read dominates
at depth (docs/legacy-benchmarks.md §B2: attention ~56% of a token at 2048 ctx).
```

## TestAttentionPrefillSteelMatchesFloat64

Moved from `metal/attention_prefill_steel_test.go` (the comment above `TestAttentionPrefillSteelMatchesFloat64`) on 2026-10-09.

```text
TestAttentionPrefillSteelMatchesFloat64 (F-G01, docs/audit-metal-2026-09-30.md) runs attention_prefill_steel, the
prefill attention for every head-dim-128 model, against float64 attention over the same f16 Q, K and V. No test that
runs by default asserted anything about this kernel before; its evidence was one pooled fidelity run at startPos 0.
The cases cover what that run did not: startPos > 0 (every chunk after the first under chunked prefill, and every
prefix-reuse turn), M on both sides of the 32-row tile, a sliding window, three GQA groups, and peaked scores that
move the running max between key blocks. The launch, the fused-QKV row stride and the 8-row padding of Q and the
output are PrefillLast's; the K/V cache holds only the keys the call may read, rounded to 8 rows, not a context's
worth.
```

## TestAttentionPrefillSteelMatchesFloat64.maxAbs

Moved from `metal/attention_prefill_steel_test.go` (the comment above `maxAbs`) on 2026-10-09.

```text
maxAbs is the audit's 0.05 (the fused kernel's bar) tightened to what this kernel measures: 4.1e-4 at worst, about
two f16 rounding steps of a unit-scale output (2026-10-01).
```

## TestLayerB_attentionParity

Moved from `metal/attention_test.go` (the comment above `TestLayerB_attentionParity`) on 2026-10-09.

```text
TestLayerB_attentionParity — Layer B: the non-trivial kernel. GQA causal attention with
ONLINE (numerically-stable) softmax over the resident KV cache: for query head qh
(kv head kvh = qh/(nH/nKV)), score_s = scale·(q·k_s), streamed softmax over s∈[0,nKeys),
output = Σ softmax_s · v_s. One thread per query head (correct-first; the per-head
threadgroup parallelism is the tuning step). Validated vs a plain CPU softmax.
```

## r17Kernels

Moved from `metal/attn_fa_blk_test.go` (the comment above `r17Kernels`) on 2026-10-09.

```text
r17Kernels is R17's step-2 prototype (docs/tasks/red-october.md; docs/measurements/metal-decode-attn-r17-2026-09-25.md)
— attention_fa's first pass restructured in the shape of llama.cpp's flash_attn_ext_vec, as graded on 2026-09-25
(fidelity decision on set B, confirmation run 3.50x). It SHIPPED as attention_fa_blk in kernels.go (allKernels);
this is that shipped text rebuilt into the standalone library form the R17 harnesses compile (the prototype's own
macro name, its own #include), so the tests grade exactly what production dispatches.
TestAttnFABlkIsTheGradedKernel pins the rebuilt text to the SHA-256 of the source that was graded.
```

## r17GradedKernelSourceSHA256

Moved from `metal/attn_fa_blk_test.go` (the comment above `r17GradedKernelSourceSHA256`) on 2026-10-09.

```text
r17GradedKernelSourceSHA256 is the SHA-256 of the prototype source graded 2026-09-25 (r17Kernels as committed in
39545bd6, the commit the pre-registered decision run was built from).
```

## TestAttnFABlkMatchesFloat64

Moved from `metal/attn_fa_blk_test.go` (the comment above `TestAttnFABlkMatchesFloat64`) on 2026-10-09.

```text
TestAttnFABlkMatchesFloat64 checks the SHIPPED attention_fa_blk (both instantiations, at the production split
count) against float64 attention on synthetic inputs — no checkpoint. Key counts cover a chunk tail of one key
(1537), several rounds per simdgroup (3900, 4100), and the depth floor; inputs cover near-uniform and peaked
(sink-like) weights. The bound is a correctness bound, orders of magnitude above f32 rounding: a wrong key range,
scale, rescale or merge gives relative errors of 1e-3 and up. The kernel's measured error on real inputs is
~2e-7 median (metal-decode-attn-r17-2026-09-25.md).
```

## TestAttnFABlkSelection

Moved from `metal/attn_fa_blk_test.go` (the comment above `TestAttnFABlkSelection`) on 2026-10-09.

```text
TestAttnFABlkSelection pins which kernel production dispatches as attention_fa's first pass. On the committed
llama-attnfa-tiny fixture (G=2) that is the block kernel exactly when attnFABlkAnyG is on, which it is since B-P02's
grade (2026-10-03); the G=2 instantiation sits outside the graded, hash-pinned region, and
TestAttnFABlk_anyGMatchesG7 is what pins it head for head against g7. With GOINFER_HEAVY_TESTS=1 and the checkpoints
in ~/models, attention_fa_blk for Qwen2.5-1.5B (G=6) and -7B (G=7) is checked by running the resident's own r.pAttnFA
on synthetic inputs and requiring output BIT-IDENTICAL to the graded kernel source (r17Kernels) compiled on its own.
```

## TestAttentionFA_positionSweep

Moved from `metal/attn_fa_confound_test.go` (the comment above `TestAttentionFA_positionSweep`) on 2026-10-09.

```text
RESOLVED 2026-09-21 (docs/measurements/r2-attn-fa-rootcause-2026-09-21.md): both diagnostics
below measured something real and neither measured a kernel defect. The position linkage the
sweep found is the INPUT's — position 1602's forward pass has an element within ~1e-6 of an int8
activation-quantization rounding boundary, which attention_fa's sparse f32 reduction-order noise
crosses at layer 16 (r2_ctx_diff_test.go's accumulated-divergence table). The ULP control's
"immediate, uniform" divergence was evidence FOR that hypersensitivity, not against it: a scale
nudge is a dense perturbation of every score at every layer and crosses a boundary at once; the
proper control (shipped kernels + a 1e-6 residual nudge after layer 0, same magnitude as the
kernel's real discrepancy) reproduces attention_fa's 0.6-level logit jump at every step.

TestAttentionFA_positionSweep answers a sharp objection to the "call-count, not position"
conclusion in docs/measurements/r2-attn-fa-followup-2026-09-20.md: the two prefill depths
already tried, 1600 and 2200, are BOTH multiples of 8 — so "third decode call" and "key count ≡
3 (mod 4)" were the same event in both runs, and the earlier experiment cannot tell them apart.
attention_fa's own kernel groups work by 4 consecutive vocabulary-adjacent... no, by 4-key tiles
(its own doc comment: "cooperative load, 32 lanes x half4"), so a boundary condition tied to
nKeys mod 4 is a live, structurally-motivated alternative to "call count" that the prior
experiment could not rule out.

This sweeps prefillLen over 1601, 1602, 1603 — three depths NOT sharing 1600/2200's residue —
so the failing step's mod-4 alignment and its call-count alignment come apart. For each depth,
steps 0-4 land at positions (prefillLen+step); if the true trigger is "3rd decode call"
regardless of position, the first-divergent STEP stays at 2 for all three. If it is really
"nKeys ≡ 3 (mod 4)" (or any other position-linked residue), the first-divergent step moves
depth-to-depth (since prefillLen+step's residue mod 4 shifts as prefillLen shifts by 1).
```

## TestAttentionFA_ulpPerturbationControl

Moved from `metal/attn_fa_confound_test.go` (the comment above `TestAttentionFA_ulpPerturbationControl`) on 2026-10-09.

```text
TestAttentionFA_ulpPerturbationControl tests the OTHER sharp alternative to "attention_fa has a
state-carrying bug": that "two clean decode steps, then a stable wrong plateau" is simply what
ANY non-bit-identical kernel eventually looks like once accumulated f32 error crosses a
downstream int8 activation-quantization rounding boundary — a discrete, not gradual, jump, by
construction of rounding. If that is right, deliberately perturbing the REFERENCE (shipped)
kernel's own attention scale by a single float32 ULP — nothing to do with attention_fa at all —
should reproduce the same qualitative signature: some early steps identical, then a stable
divergence once the rounding boundary is crossed.

Both arms here use the SAME (shipped) kernel; only r.uScale's bit pattern differs between them
by exactly one ULP. If this control shows the same "clean, then stable jump" shape, the shape
itself carries no information about a bug in attention_fa specifically — it would be expected
under R2's OWN registered fidelity gate (attention_fa is deliberately not bit-identical), and R2
can proceed to that gate directly rather than keep hunting for a mechanism. If the control stays
clean throughout (or diverges gradually, not in a sudden plateau), the signature IS diagnostic
and the search for a state-carrying mechanism in attention_fa specifically should continue.
```

## TestAttentionFA_endToEndReproductionDepth2200

Moved from `metal/attn_fa_e2e_depth_test.go` (the comment above `TestAttentionFA_endToEndReproductionDepth2200`) on 2026-10-09.

```text
TestAttentionFA_endToEndReproductionDepth2200 is the depth-2200 twin of attn_fa_e2e_test.go's
keeper reproducer of the signature that PARKED R2 (docs/tasks/red-october.md) on 2026-09-19.
EXPLAINED 2026-09-21 (docs/measurements/r2-attn-fa-rootcause-2026-09-21.md; instrument:
r2_ctx_diff_test.go): not a kernel defect — on identical inputs attention_fa matches the shipped
kernel to <=1e-5 at every layer and step; the logit divergence is one int8 activation-quantization
rounding crossing on accumulated f32 reduction-order noise (the kernel is non-bit-identical by
design), amplified downstream and carried forward by the KV cache. The numbers this test
measures are unchanged; what they mean is.

The reproduction is precise and fully deterministic (identical across repeated runs): batch-
prefill a real qwen2.5-1.5b checkpoint to depth 1600 (> attnFADepthFloor), then decode 5 tokens
one at a time, comparing logits against the SAME prefill+decode sequence with attention_fa
disabled. Steps 0-1 (pos 1600-1601) are BIT-PERFECT (cosine 1.0000000, maxAbs ~2.9e-6 — pure
f32 rounding noise, the same order of magnitude the isolated kernel gate shows). Step 2 onward
(pos >= 1602) jumps to a STABLE wrong result (cosine ~0.9995-0.9996, maxAbs ~0.57-0.62) and
stays there — not a growing numerical drift, not run-to-run noise (reproduced byte-for-byte
across independent test executions with fresh GPU state each time).

Ruled out (via GOINFER_ATTNFA_DEBUG=1's per-dispatch print at model.go's canUseAttnFA call
site): the dispatch parameters themselves (curNKeys, nSplit, G, hd) are correct and CONSTANT
across every layer and every decode step, exactly as this uniform-architecture model should
produce — so the divergence is not a parameter-computation bug. Also ruled out: cross-instance
GPU state (the shipped and attention_fa runs execute fully sequentially, one resident closed
before the next loads — see runOne's own teardown). Not yet investigated: whether Metal's
automatic hazard tracking correctly serializes the attention_fa -> attention_fa_combine buffer
dependency across the SPECIFIC pattern of repeated per-layer, per-token buffer reuse this
kernel pair uses (a genuine unknown, not a ruled-out hypothesis); whether something about the
SECOND-AND-LATER command buffer specifically (as opposed to the first) interacts badly with the
shared r.attnFAPartial/r.uAttnFAG/r.uAttnFANSplit scratch buffers this kernel reuses across
calls, unlike the shipped kernel's own per-call-safe buffer usage.
```

## TestAttentionFA_endToEndReproduction

Moved from `metal/attn_fa_e2e_test.go` (the comment above `TestAttentionFA_endToEndReproduction`) on 2026-10-09.

```text
TestAttentionFA_endToEndReproduction is the KEEPER reproducer of the end-to-end signature that
PARKED R2 (docs/tasks/red-october.md) on 2026-09-19: a kernel proven correct in isolation (16/16
adversarial cases in TestAttentionFA_vsReference, cosine 1.0000000 every time) whose logits
diverge from the shipped kernel's from decode step 2 on. EXPLAINED 2026-09-21
(docs/measurements/r2-attn-fa-rootcause-2026-09-21.md; instrument: r2_ctx_diff_test.go): not a
kernel defect. On identical inputs attention_fa matches the shipped kernel to <=1e-5 at every
layer and step; the divergence is ONE int8 activation-quantization rounding crossing at layer 16
of position 1602, on accumulated f32 reduction-order noise (the kernel is non-bit-identical by
design), amplified by the layers after it and carried forward by the KV cache — a shipped-kernel
run with a 1e-6 residual nudge reproduces the same 0.6-level jump. The numbers below are still
exactly what this harness measures; what changed is what they mean. R1's same-day layer-26
investigation, filed beside this one, was likewise an instrument error (a coarser arm used as
ground truth; r1-layer26-rootcause-2026-09-20.md) — a different mechanism, same lesson.

The reproduction is precise and fully deterministic (identical across repeated runs): batch-
prefill a real qwen2.5-1.5b checkpoint to depth 1600 (> attnFADepthFloor), then decode 5 tokens
one at a time, comparing logits against the SAME prefill+decode sequence with attention_fa
disabled. Steps 0-1 (pos 1600-1601) are BIT-PERFECT (cosine 1.0000000, maxAbs ~2.9e-6 — pure
f32 rounding noise, the same order of magnitude the isolated kernel gate shows). Step 2 onward
(pos >= 1602) jumps to a STABLE wrong result (cosine ~0.9995-0.9996, maxAbs ~0.57-0.62) and
stays there — not a growing numerical drift, not run-to-run noise (reproduced byte-for-byte
across independent test executions with fresh GPU state each time).

Ruled out (via GOINFER_ATTNFA_DEBUG=1's per-dispatch print at model.go's canUseAttnFA call
site): the dispatch parameters themselves (curNKeys, nSplit, G, hd) are correct and CONSTANT
across every layer and every decode step, exactly as this uniform-architecture model should
produce — so the divergence is not a parameter-computation bug. Also ruled out: cross-instance
GPU state (the shipped and attention_fa runs execute fully sequentially, one resident closed
before the next loads — see runOne's own teardown). Not yet investigated: whether Metal's
automatic hazard tracking correctly serializes the attention_fa -> attention_fa_combine buffer
dependency across the SPECIFIC pattern of repeated per-layer, per-token buffer reuse this
kernel pair uses (a genuine unknown, not a ruled-out hypothesis); whether something about the
SECOND-AND-LATER command buffer specifically (as opposed to the first) interacts badly with the
shared r.attnFAPartial/r.uAttnFAG/r.uAttnFANSplit scratch buffers this kernel reuses across
calls, unlike the shipped kernel's own per-call-safe buffer usage.
```

## TestAttentionFA_pipelinedEncodeRace

Moved from `metal/attn_fa_pipeline_race_test.go` (the comment above `TestAttentionFA_pipelinedEncodeRace`) on 2026-10-09.

```text
TestAttentionFA_pipelinedEncodeRace is R2's own suggested next step
(r2-attn-fa-2026-09-19.md): a minimal, isolated repro chaining several
command buffers under the REAL production pipelining pattern (encode
buffer N+1 while buffer N is still executing — metal/model.go's execLoop,
Commit()/FinishEncoding()/WaitDone(), NOT the synchronous Begin()/End()
attn_fa_test.go's gate (1) uses) instead of a full 28-layer real model.

Production's shared uniform buffers (r.uAttnFAG/r.uAttnFANSplit) are
SetU32'd — a raw CPU write to shared memory, not a tracked Metal command —
while encoding buffer N+1, which happens WHILE buffer N is still
executing on the GPU. In production this is harmless because G/nSplit
never change between layers or decode steps (the debug print already
proved this). This test uses the SAME shared-buffer-SetU32-during-encode
pattern but varies the value every iteration specifically so a real race
becomes OBSERVABLE — if buffer N's dispatch reads iteration N+1's value
instead of its own, that is the mechanism, confirmed in isolation rather
than inferred from a 28-layer model's stably-wrong logits.
```

## TestAttentionFA_vsReference

Moved from `metal/attn_fa_test.go` (the comment above `TestAttentionFA_vsReference`) on 2026-10-09.

```text
TestAttentionFA_vsReference is R2's (docs/tasks/red-october.md) gate (1), basic form: the new
attention_fa/attention_fa_combine pair against the CPU f64-shaped reference (cpuAttention,
shared with TestAttention_ShippedKernelShapes), at the "control qwen2.5-1.5b" shape (hd=128,
the only head width this kernel supports — see its own doc comment) and a handful of nKeys/nSplit
combinations. Not yet the amended runAttnCase-pattern cases (hot key at split boundaries, a
rising-score ramp) — this is the first correctness pass, proving the mechanism before the harder
adversarial inputs.
```

## TestZZ_attnKVWidthProbe

Moved from `metal/attn_kvwidth_probe_test.go` (the comment above `TestZZ_attnKVWidthProbe`) on 2026-10-09.

```text
TestZZ_attnKVWidthProbe — the P4 deciding measurement (docs/plan-still-slow.md §P4). Re-runs the
2026-08-04 collapse probe with a THIRD arm: half the DRAM bytes per key at the SAME element count
(int8 KV vs the f16 baseline — q8's exact byte profile), to separate BANDWIDTH from LATENCY.

The original probe pinned every K/V read to key 0 (zero distinct DRAM) and saw all-28-layer
attention 21.5→5.3 ms — 75% of attention is distinct per-key reads. But pinning collapses BOTH
bytes AND latency (key 0 stays cached), so it cannot say whether q8 (fewer bytes, same number of
serial reads) helps. This probe adds `attn_q8`: hd elements per key at 1 byte each instead of 2,
same loop / same ALU / same threadgroups, half the DRAM bytes.

  - if q8 time drops ~proportionally toward the full baseline → BANDWIDTH-bound → P4 BUILDS (q8 is a
    Metal speed lever, CUDA a reachability one).
  - if q8 barely moves off full → the reads are latency-exposed, bytes aren't the bind → P4 is
    CUDA-reachability-only (q8 for VRAM, not Metal speed).

`attn_pin0` is the harness self-check: it must reproduce the known collapse (~4× off full), or the
microbench geometry is wrong and the q8 number is not to be trusted. Opt-in; a timing diagnostic,
not a gate.
```

## TestZZ_attnKVWidthProbe.selfcheck

Moved from `metal/attn_kvwidth_probe_test.go` (the comment at the pin0 self-check) on 2026-10-09.

```text
Harness self-check: pin0 must collapse (~4× off full), matching the 2026-08-04 21.5→5.3 ms.
```

## TestZZ_attnM3ThreadWidth

Moved from `metal/attn_m3_probe_test.go` (the comment above `TestZZ_attnM3ThreadWidth`) on 2026-10-09.

```text
TestZZ_attnM3ThreadWidth — plan §M3 first data point: the Metal FA go/no-go. On M=1 decode the
FA "don't materialize the score vector" benefit does NOT apply (the score vector is only nKeys
floats), so the only lever an FA-style rewrite has over the shipped one-threadgroup-per-head serial
pass is MORE IN-FLIGHT PARALLELISM to hide the DRAM-latency wall the half-width probe found (q8
moved attention only 12% → latency-bound, not byte-bound). This sweeps the per-head threadgroup
WIDTH (128 = shipped → 256 → 512): more threads = fewer keys/thread = more concurrent K/V loads in
flight per head. If a wider tile materially beats 128, an occupancy/latency lever exists and M3 is
a GO; if it is flat, the latency wall holds regardless of parallelism and the honest M3 outcome is
a refutation joining the A2-Metal record ("the lever is elsewhere, or accept the floor").

This is not the FA kernel — it is the go/no-go probe that says whether building one could help.
Opt-in timing diagnostic, not a gate.
```

## TestAttention_ShippedKernelShapes

Moved from `metal/attn_shape_test.go` (the comment above `TestAttention_ShippedKernelShapes`) on 2026-10-09.

```text
TestAttention_ShippedKernelShapes drives the SHIPPED attention kernel (allKernels, the one
model.go dispatches) at Gemma 3's exact attention shape and at the control's, on REALISTIC
attention patterns — the case the existing coverage structurally cannot reach.

Why this gap exists. Metal's gemma3 parity carries a Gemma-only residual of -0.104 against its
own CPU-int4 twin (metal/quantbar_test.go) that the weights do not explain (the double
quantization measured free — decoder/requant_test.go), so a Gemma-specific kernel is wrong. The
shape of the residual points here: 9 gaps >3% with a worst near-tie of 40.8% is not what a
per-layer precision delta compounding over 34 layers looks like (that is a smooth droop) — it
is an op failing on PARTICULAR positions. And multi-key attention at hd=256 is the one op that
is both Gemma-specific and untested:

  - the dense control is hd=128, so it is blind to an hd=256 fault by construction;
  - attention_test.go compiles its own INLINE copy of the kernel, not the shipped source;
  - layer_test.go runs the shipped one, but at hd=64 — and does not bind `window` at all;
  - the position-0 analysis could never see it: at pos 0 attention output IS v0 exactly, so
    multi-key softmax never runs. Every Gemma number that mattered lives at pos>0.

The patterns matter as much as the shape. Random q/k give a DIFFUSE softmax where every key
contributes ~1/nKeys and errors average out — which is why a random-weight synthetic passed at
0.997 while the real model does not. Real attention is SHARP (one key dominant) and real V
carries outliers, so this drives both: sharp scores, an outlier V row, and Gemma's own
near-zero-norm sink at key 0.
```

## TestAttention_ShippedKernelShapes.gemma4

Moved from `metal/attn_shape_test.go` (the comment at the gemma4 global rows) on 2026-10-09.

```text
Gemma 4's global layers run at hd=512 (g4.GlobalHeadDim), and NOTHING on Metal has ever
driven the attention kernel there — this is 9c's single biggest kernel unknown, retired
here before the port. The kernel decomposes each head over a fixed 128-thread block, so
hd=512 = 4 elems/thread stresses the wide end the same way CUDA's hd 128/256/512 sweep did
(933201c, cosine ≈ 1.0). The first row is the pure single-variable mutation off the green
gemma3-4b global row above (only hd 256→512 changes), so a red is attributable to head
width, not the contract. The next two are Gemma 4's REAL global geometry (nKV=2, K=V, no
window) at short and long ctx — the shape 9c will actually dispatch.
```

## TestAttention_ShippedKernelShapes.norm

Moved from `metal/attn_shape_test.go` (the comment at the norm log line) on 2026-10-09.

```text
Norm is reported next to cosine deliberately: the sink hunt burned a week on a
cosine that was meaningless because the vector under it was near-zero. Never again
read one without the other.
```

## TestAuditAP02_shortPromptTiming

Moved from `metal/audit_ap02_test.go` (the comment above `TestAuditAP02_shortPromptTiming`) on 2026-10-09.

```text
pre-registration"): a fresh K-token prompt's wall time three ways, as T1.10 timed it, but on the build A-P01 changed
(its smaller tiles made the C <= 32 pass 1.8x faster, so T1.10's step / pass figures no longer describe the pass):
```

## TestAuditM26_pagedProbe.scalecache

Moved from `metal/audit_m26_test.go` (the comment above the build heartbeat) on 2026-10-09.

```text
C-P01 (2026-10-02): the pager stages scales from the mapping, so there is no scale cache to size. The old
binary's figure (experts × per-expert scale words × 2 bytes) was 1361.2 MB; the heap line at token 32 is the
measurement now.
```

## TestGemmaBisect_PerLayer

Moved from `metal/bisect_test.go` (the comment above `TestGemmaBisect_PerLayer`) on 2026-10-09.

```text
TestGemmaBisect_PerLayer walks the residual stream layer by layer, Metal vs CPU, to locate
WHERE Metal's Gemma-only parity residual enters.

What is already known, so this test does not re-litigate it:
  - int4 costs gemma3 0.99→0.92 of logit cosine and the control ~nothing (quantbar_test.go).
    That part is the quantization class at gemma's shape and is not a bug.
  - Metal still adds a further -0.104 on gemma and NOTHING on the control. That part is a bug.
  - It is not the weights: the double quantization measured free (decoder/requant_test.go).
  - It is not multi-key attention at hd=256: the shipped kernel is exact there, including
    sharp softmax, outlier V, sink, and window engaged (attn_shape_test.go).

So a Gemma-specific op in the compute path is wrong, and the residual's SHAPE says which kind:
9 gaps >3% with a worst near-tie of 40.8% is not a per-layer precision delta compounding over
34 layers (that would be a smooth droop). It is an op failing on particular positions.

Three deliberate choices, each one a lesson already paid for:

 1. Reference is CPU-INT4, not CPU-int8. Metal's weights are now measured equivalent to the
    decoder's int4, so int4 is the like-for-like reference; int8 would fold the (large, real,
    not-a-bug) quantization cost into every number and hide the -0.104 underneath it.
 2. NORM is reported beside cosine at every layer. The sink cost a week because a cosine was
    read off a near-zero vector. A collapsing norm and a collapsing cosine are different bugs.
 3. The probe is an ORDINARY token at pos>0. At pos 0 attention output is v0 exactly and RoPE
    is the identity — the two ops most under suspicion do not even run, which is precisely why
    the earlier pos-0 analysis saw nothing.

Each layer is tagged local/global (gemma's 5:1). If the jump tracks the global layers, the
carrier is attention/window/per-layer-RoPE; if it is uniform, it is a per-layer op.
```

## TestGemmaBisect_PerLayer.reconcile

Moved from `metal/bisect_test.go` (the comment at the cross-backend reconciliation) on 2026-10-09.

```text
Cross-backend reconciliation with the CUDA box. Three references at each target channel +
neighbors, at the pre-final-norm tap (pos 5):
  metal      — Metal resident (int4 weight × int8 activation, W4A8)
  cpu-int4   — decoder Quant:int4 == MatmulBTW4A8, ALSO int8 activation → shares the crush
  cpu-int8w  — decoder Quant:int8 (weight-only) == MatmulBTQ8, int8 weight × f32 ACTIVATION
               → NO activation crush, so its SIGN is the ground truth on crushed channels.
The crux the CUDA box surfaced: our two "int4" references disagreed because BOTH quantize
activations to int8 and round the near-zero crushed channels differently. cpu-int8w removes
the activation quant, so whichever of metal/cpu-int4 disagrees with cpu-int8w's SIGN is the
one that flipped. (Off-by-one is ruled out: the 443 spike sits at index 443 on both boxes.)
```

## TestGemmaBisect_Head

Moved from `metal/bisect_test.go` (the comment above `TestGemmaBisect_Head`) on 2026-10-09.

```text
TestGemmaBisect_Head splits the ONE step the per-layer bisect leaves whole: final-norm → LM
head. The per-layer walk showed gemma's trunk lands at cosine ~0.981 (control ~0.993) yet the
logits collapse to 0.818 (control 0.990) — so almost the entire Gemma-only residual enters
HERE, not in the 34 layers. This test says which of the two sub-steps:
```

## TestGemmaTraceDims

Moved from `metal/bisect_test.go` (the comment above `TestGemmaTraceDims`) on 2026-10-09.

```text
TestGemmaTraceDims follows Gemma's massive-activation dims down the layer stack to find WHERE
Metal clobbers them. The head bisect showed the final-norm amplifies a handful of outlier dims
(1698/1730/2482/1723/227) that Metal has zeroed or sign-flipped; an all-dims cosine can't see 6
bad channels in 2560, so this prints those channels explicitly at every layer, Metal vs CPU.
```

## TestBytesToU32_matchesManualLE

Moved from `metal/bytestou32_test.go` (the comment above `TestBytesToU32_matchesManualLE`) on 2026-10-09.

```text
TestBytesToU32_matchesManualLE gates N-29 (audit-metal-2026-09-12.md): bytesToU32 used to
reconstruct each word with a per-byte shift-and-mask loop; it now does one bulk copy into a
freshly-allocated (always 4-aligned) []uint32's own byte view. Confirmed red without the fix by
temporarily reverting to a deliberately wrong byte order (big-endian) — this test caught it
immediately, since the oracle below is independently computed via encoding/binary rather than by
re-deriving the same shift expression bytesToU32 itself uses.
```

## TestCB01ChainAB.warm

Moved from `metal/cb01_grade_test.go` (the comment at the discarded cold generation) on 2026-10-09.

```text
The first Generate prefills the prompt cold; every later one reuses 63 of its 64 tokens and re-forwards the last on
the decode path, whose logits differ by ulps (int8 activations against the pass's f16; the E-P01 entry of the task
doc's log). Greedy's argmax survives that; a near-tie Gumbel draw need not (measured on the 0.5B at T=1: two
chain-off arms, cold then warm, part at token 18). So one cold generation is discarded, and every compared arm is warm.
```

## TestMetal_CloseFreesMemory

Moved from `metal/close_leak_test.go` (the comment above `TestMetal_CloseFreesMemory`) on 2026-10-09.

```text
Why this matters: purego has no ARC and Metal has no context-destroy to reclaim in bulk, so
every MTLBuffer must be released explicitly. Close() used to free NOTHING — it closed the
executor channel and returned, on the documented assumption of a "single-model lifetime".
cmd/serve is multi-model with /admin/models/unload, so that assumption leaked a whole model
(weights + per-layer KV + MoE experts) per load. Invisible in a one-model run — which is
exactly why it survived.
```

## TestMetal_CloseFreesMemory.device

Moved from `metal/close_leak_test.go` (the comment at the device-reported gate) on 2026-10-09.

```text
The device-reported gate (C-G01, docs/audit-metal-2026-09-30.md): on UMA a leaked MTLBuffer need not show in RSS
(macOS compresses idle pages out), so RSS alone can pass a real leak. MTLDevice's own total counts every buffer
still allocated on the GPU, whichever Device handle made it. Measured 2026-10-01: back to the same byte count after
every cycle. The slack is far below one resident's buffers.
```

## TestMetal_CloseFreesMemory.cycle

Moved from `metal/close_leak_test.go` (the comment at the growth bound) on 2026-10-09.

```text
Each cycle allocates ~0.4 GB of Metal buffers (int8 weights re-quantized to int4 + KV; 401 MB by
CurrentAllocatedSize, 2026-10-01). If Close frees, growth across 4 cycles stays near zero; if it leaks, it is GBs.
```

## TestMetal_PrefillScratchDoesNotLeak

Moved from `metal/close_leak_test.go` (the comment above `TestMetal_PrefillScratchDoesNotLeak`) on 2026-10-09.

```text
TestMetal_PrefillScratchDoesNotLeak is the C5 gate: PrefillLast used to allocate ~24 per-call
scratch/uniform buffers onto the device ledger and free NONE until Close, so every request
leaked ~100–150 MB (7B) of unified memory — a ratchet, since cmd/serve calls PrefillLast once
per request. The mustBuf OOM panic that eventually followed is recovered only on BuildResident's
path, not prefill's, so it killed serve. The existing close_leak tests pin load/Forward/Close
cycles, not per-REQUEST prefill growth — which is why this class was invisible.

Signal (same as the sibling gates): run many PrefillLast calls against ONE resident model and
watch the trajectory. Per-call release → flat; the old leak → a staircase of ~24 buffers/call.
```

## TestMetal_CloseWithSecondModelAlive

Moved from `metal/close_leak_test.go` (the comment above `TestMetal_CloseWithSecondModelAlive`) on 2026-10-09.

```text
TestMetal_CloseWithSecondModelAlive is the condition the Linux box warned about: their first
CUDA fix looked correct under a single load/close cycle and was NOT — the bug only showed with
a second context/model alive. Two hazards it covers that a sequential test cannot:
```

## TestMetal_CloseWithSecondModelAlive.rss

Moved from `metal/close_leak_test.go` (the comment at the ledger assertion) on 2026-10-09.

```text
RSS cannot answer this on a loaded machine: an earlier version loaded a C after closing A and
checked RSS stayed flat, but macOS returns freed MTLBuffer pages to the allocator (not the OS)
and COMPRESSES inactive pages under memory pressure — so the reuse probe read a clean free as a
leak when the box was busy (swinging +2 MB idle to +560 MB loaded on identical, correct code),
and, worse, a real leak (ReleaseAll neutered) DID NOT ratchet RSS because the leaked pages were
compressed straight back out. RSS is unreliable in both directions here.
```

## TestMetalResident_C11_argmaxEqualsFullLogits

Moved from `metal/cmdbuf_status_test.go` (the comment above `TestMetalResident_C11_argmaxEqualsFullLogits`) on 2026-10-09.

```text

N-33: this used to say "C-10's buildResident guard declines any non-%8 vocab to the CPU". It
does not, and model.go says so explicitly — vocab is NOT checked there, because the LM head is
pinned int8 and dispatches gemv_w8a8_coal (no hazard) or gemv_w8a8_amax, and ForwardArgmax
ROUTES a non-%8 vocab around the hazardous kernel (full logits + host argmax) rather than
declining the family. The code is right; this comment described a guard that was considered
and not built.
```

## TestCohereResidentSmokeMetal.pairwise

Moved from `metal/cohere_resident_smoke_test.go` (the comment at the pairwise-RoPE decline) on 2026-10-09.

```text
CORRECTION 2026-10-01 (written on the CUDA box, NOT run on a Mac): Cohere/Cohere2 rotate
GPT-J PAIRWISE (dims 2d, 2d+1) and Metal's `rope` kernel is NeoX half-split, so
FeatPairwiseRoPE is declared by CUDA only and Metal now DECLINES these families to the CPU path.
This smoke test could not see why that matters: it checks admission + no NaN, and a wrong
rotation is exact at position 0 and fluent-looking after (the CUDA resident ran real
Command-R7B / Aya wrong that way, worst per-position cosine -0.075 / -0.041). Until Metal has
pairwise rope kernels, a decline naming the feature is the PASS; when it declares the
feature this test falls through to the original admission + no-NaN smoke below, and the
session that ports the kernels must add a peaked-attention resident-vs-CPU gate beside it
(cuda/pairwise_rope_resident_parity_test.go is the template).
```

## TestCopyBytesToU32Buf_oversizedSrcPanics

Moved from `metal/copybytes_bounds_test.go` (the comment above `TestCopyBytesToU32Buf_oversizedSrcPanics`) on 2026-10-09.

```text
TestCopyBytesToU32Buf_oversizedSrcPanics gates N-33 (audit-metal-2026-09-12.md):
copyBytesToU32Buf used to unsafe.Slice-reinterpret dst and Go-copy src into it, which silently
truncates an oversized src with no error — the exact gap gpu.Upload's own bounds check exists to
close. Confirmed red without the fix: the old implementation returned normally on an oversized
src (truncating it into dst) instead of panicking.
```

## TestDB01_prefillFromZeroResetsState

Moved from `metal/db01_prefill_test.go` (the comment above `TestDB01_prefillFromZeroResetsState`) on 2026-10-09.

```text
TestDB01_prefillFromZeroResetsState (D-B01): PrefillLast from position 0 is a fresh sequence, so a hybrid's DeltaNet
window and state start from zero, as Forward(pos 0) makes them. The pass continues whatever state the resident holds
(right for a continuation), so without the reset a new prompt prefilled after another one started from the previous
sequence's state: the fidelity gate's pass arm did, after its sequential arm, and read 115 hard flips to the
sequential arm's 15 on a K = 8 smoke. Through metalResident.PrefillLast, the entry point the decoder and the gate
call: prompt B after prompt A must equal prompt B on a reset resident, logits and state bit for bit.
```

## TestEnsurePrefill_latchesFailure

Moved from `metal/ensureprefill_latch_test.go` (the comment above `TestEnsurePrefill_latchesFailure`) on 2026-10-09.

```text
TestEnsurePrefill_latchesFailure gates N-47 (audit-2026-09-10.md): a failed ensurePrefill used
to leave r.pf == nil with no record of WHY, so every later PrefillLast call re-ran the full MSL
compile from scratch just to panic identically again. Confirmed red without the fix: with the
pfErr short-circuit removed, ensurePrefill fell through to the real compile attempt against this
test's zero-value (no real Metal setup) Device and panicked with an unrelated compile-landmine
message instead of the cached sentinel — proving the short-circuit, not the compile itself, is
what this test pins.

The fast path this test exercises (r.pfErr already set) runs entirely before ensurePrefill
touches r.d, so a bare &resident{} with no real Device proves the short-circuit never reaches
the expensive compile path — no Metal device needed for this half of the fix.
```

## TestDeltaNetKernels_mutations.collapse

Moved from `metal/deltanet_test.go` (the comment at the collapse-the-state-row case) on 2026-10-09.

```text
A first attempt swapped hv/hk in the offset formula, but at this test's REAL geometry
hk==hv==128 makes that a no-op (numerically identical). A second attempt permuted
(headV,vd)->row bijectively — also invisible: a thread's row is never read by any
OTHER thread, so consistently relocating one thread's own private storage changes
nothing about what it computes, only where. The actual bug class "un-transposing"
guards against is state ADDRESSES COLLIDING across threads, which a bijection can't
produce by construction. This drops the vd term instead, so every value head sharing
one key head... no, every vd for a fixed headV now aliases ONE row: max offset
(nv-1)*hk = 47*128 = 6016, safely in-bounds, and the per-thread exclusivity the file
header calls out ("no cross-thread sharing") is exactly what breaks.
```

## TestExecutorAttnPlan

Moved from `metal/exec_plan_test.go` (the comment above `TestExecutorAttnPlan`) on 2026-10-09.

```text
TestExecutorAttnPlan pins the pipelined executor's attention plan to each job's OWN key count.

execLoop encodes token t+1's command buffer while token t runs on the GPU. What that encode bakes in
includes the attention plan: whether a layer dispatches attention_fa (canUseAttnFA's depth gate at
attnFADepthFloor) and attention_fa's split grid (attnFASplitFor). Those decisions used to read the
resident's key count at encode time — the PREVIOUS job's — so (found 2026-09-25, R17:
docs/measurements/metal-decode-attn-r17-2026-09-25.md):
  - in steady decode the plan lagged one token (the first step at 1536 keys ran the shipped kernel);
  - the first decode step of a new request ran the plan of wherever the previous request stopped —
    attention_fa below its floor after a long request, with a grid sized for the old depth and a split
    uniform sized for the new one; the shipped kernel after a short request;
  - after ForwardBatch (which zeroes the depth reading) the next step declined attention_fa at any depth.

The oracle is the synchronous path (ForwardEmb / PrefillLast), which always sets the position before it
encodes. Every sequence below runs through the production adapter (metalResident.Forward -> the executor)
and must be BIT-IDENTICAL to the same sequence run synchronously, with nothing flushed in between — the
executor's state carries over across "requests" exactly as it does in a server.

Fixture: testdata/llama-attnfa-tiny (dense GQA, head dim 128 — the one committed fixture attention_fa
engages on; see snapshot_golden_test.go). A few seconds; no heavy assets.
```

## mc3MMAKernels

Moved from `metal/gemm_mma8_mc3_test.go` (the comment above `mc3MMAKernels`) on 2026-10-09.

```text
mc3MMAKernels is MC3's matrix-unit prototype (docs/tasks/task-concurrency-2026-09.md), TEST-ONLY: a W4A8 GEMM for
up to 8 decode sequences on simdgroup_matrix, sized for decode rather than prefill. The prefill GEMM's 64-token tile
costs the same at M = 1 as at M = 64 (TestMC3S0PrefillGEMMSmallM: 12-14x one decode GEMV), so this one's token tile
is a single 8x8 fragment column: C^T[feature][token] = W[feature][k] . A^T[k][token].
```

## TestGemma4_26B_autoPagedRuns

Moved from `metal/gemma4_26b_autopaged_test.go` (the comment above `TestGemma4_26B_autoPagedRuns`) on 2026-10-09.

```text
TestGemma4_26B_autoPagedRuns is R11(c) (docs/tasks/red-october.md): the M26 row re-run against
the AUTO-SIZED pager (decoder.Options.MoECacheExperts with MoECacheSlots left at 0), not a
hand-picked N the way TestGemma4_26B_pagedRuns (GOINFER_METAL_MOE_SLOTS, N=32 default) exercises.
metalMoESlotsRequest (metal/backend.go) only calls the real autoMoESlots formula when
MoECacheExperts is set and no explicit slot count is given — GOINFER_METAL_MOE_SLOTS bypasses
that path entirely, so the existing test does not answer this brief's question.

SAFETY (the reason this is its own test, run in isolation, not folded into a sweep): this exact
model class (M26, alongside M35/H27) produced a real kernel panic on this machine via the
CPU-staged fallback (benchmarks.md "M35/M26 on the Mac"). MoECacheExperts's paged path is a
DIFFERENT, GPU-resident mechanism (contiguous per-layer expert-slot pool, on-demand pread) that
measured a real, if slow, decode rate in that same record (~2 tok/s) — not the disaster path —
but only when it actually engages, which is why r.g4moe.paged and the auto-sized N are asserted
BEFORE any decode step runs, not inferred from the outcome. Bounded to 4 timed decode steps
(mirrors TestGemma4_26B_pagedRuns's own 5-token bound), RSS logged before/after load and after
every step so a runaway is visible immediately rather than discovered after the fact.

MEASURED, 2026-09-20 (docs/measurements/metal-moe-autopager-m26-2026-09-20.md): this test's own
in-process safeguards (DecodePath/g4moe.paged checks, the RSS kill switch) correctly confirmed
the right mechanism engages, but did NOT prevent a real near-incident — an externally-monitored
run showed system swap spiral to 12+ GB within ~50s of process start, entirely during
decoder.Load/buildResident, well before this test's own RSS check (which reads low because the
spike is transient host-side mmap/parse traffic that settles before buildResident returns) had
anything to catch. Killed manually from outside the test process. A SEPARATE, EXTERNAL memory
monitor is not optional context when running this test — see the record for what one looks
like and why the in-process guards alone were not enough here.
```

## TestGemma4_26B_pagedRuns.informational

Moved from `metal/gemma4_26b_paged_test.go` (the comment at the INFORMATIONAL argmax log) on 2026-10-09.

```text
absolute character — that was a 2-layer tiny-fixture number, the WRONG bar (box finding f93bda1:
"conditioning ≠ geometry"). The geometry-composition GATE is the calibrated int4 envelope on the
scaled-dense fixture (TestGemma4DenseScaled_metalParity: pos-0 0.982, mean within envelope,
```

## TestGemma4DenseScaled_metalParity

Moved from `metal/gemma4_dense_scaled_test.go` (the comment above `TestGemma4DenseScaled_metalParity`) on 2026-10-09.

```text
TestGemma4DenseScaled_metalParity is the Metal mirror of the box's CUDA scaled-dense gate
(cuda/gemma4_dense_scaled_test.go, f93bda1) — the DEPTH/REAL-GEOMETRY control the 26B needed and
the tiny fixtures lacked. Scaled dense Gemma 4: hidden 1024, 12 layers, 5:1 sliding/full, REAL head
dims 256 local / 512 global, K=V globals, sandwich, softcap 30 — fits non-paged (~55 MB int4).

It settles whether the 26B secondary-gate divergence is a Metal composition BUG or int4 CONDITIONING
at depth. The box found "conditioning ≠ geometry": goinfer's CPU forward is bit-right at this
geometry (cosine 1.0 vs HF golden), and CUDA composes within the int4 envelope. The gate is the
CALIBRATED envelope, not an absolute floor: Metal must agree with CPU-int4 at least as well ON
AVERAGE as int4 agrees with f32 (holds by construction — activation perturbation < weight
perturbation — unless a real kernel bug diverges faster than the fixture's own quantization).
If Metal passes here at hd=256/512, the 26B "divergence" is int4 conditioning at 64-layer depth
(worse with more layers + a specific prompt), NOT a Metal attention bug.
```

## TestGemma4DenseScaled_metalParity.pos0

Moved from `metal/gemma4_dense_scaled_test.go` (the comment at the pos-0 floor) on 2026-10-09.

```text
pos-0 kernel correctness (no KV accumulation): the 256-local/512-global geometry must compose.
(S1.0 amendment, docs/tasks/task-multimodal-support-2026-10.md: raised from 0.97 after the dense layer scalar and v_norm fixes, between the before-fix 0.982297 and after-fix 0.999148 readings)
```

## TestGemma4DenseScaled_metalParity.meanfloor

Moved from `metal/gemma4_dense_scaled_test.go` (the comment at the mean floor) on 2026-10-09.

```text
S1.0 amendment (docs/tasks/task-multimodal-support-2026-10.md): a floor on the mean itself, between the
before-fix 0.858647 and after-fix 0.926917 readings of the dense layer scalar and v_norm fixes.
```

## TestGemma4_depthSweep

Moved from `metal/gemma4_depth_sweep_test.go` (the comment above `TestGemma4_depthSweep`) on 2026-10-09.

```text
TestGemma4_depthSweep turns "int4 conditioning at depth" from an argument into a MEASUREMENT
(standing caution: this repo attributed a quality deficit to int4 twice — 625303e, bcadd44 — and
overturned it both times). Same scaled-dense geometry (hd 256/512, K=V globals) at 12/24/48/64
layers, only depth varies. Part A: the floor-vs-depth curve (CPUint4-vs-f32 mean) + the Metal
envelope (Metal-vs-CPUint4 mean). Part B: the FULL 64-layer per-layer trace, to see whether the
smooth-then-steep collapse the 26B showed (0.97→0.73 over 11 layers, then 0.47 over 3) is the
fixture's own near-floor behaviour (explained) or a discontinuity the fixture doesn't reproduce.
Env-gated (loads GB-scale f32 checkpoints). GOINFER_DEPTH_SWEEP=1.
```

## perLayer64

Moved from `metal/gemma4_depth_sweep_test.go` (the comment above `perLayer64`) on 2026-10-09.

```text
perLayer64 captures the 64-layer fixture's per-layer hidden (Metal-int4 vs CPU-int4) at pos 3 and
logs the trace, so its shape can be compared to the 26B's (0.97→0.73→0.26).
```

## TestGemma4EModel_realE2BText.amendment

Moved from `metal/gemma4_emodel_real_test.go` (the comment at the G3 amendment) on 2026-10-09.

```text
G3 amendment (after run 1, docs/tasks/task-multimodal-support-2026-10.md): the CPU loads Metal's sidecar too, so
both sides run the same int8-pinned embedding/LM-head/PLE tables. Run 1 loaded the CPU's own e4h sidecar (those
tables at int4), which compared two quantizations rather than two engines on one set of weights.
```

## TestGemma4EModel_realE2BLocalize

Moved from `metal/gemma4_emodel_real_test.go` (the comment above `TestGemma4EModel_realE2BLocalize`) on 2026-10-09.

```text
position 18, CPU gap 7.63% in run 2) localized per layer. Both sides replay the prompt to that position, then the
```

## TestGemma4MoE_noiseFloor

Moved from `metal/gemma4_moe_noisefloor_test.go` (the comment above `TestGemma4MoE_noiseFloor`) on 2026-10-09.

```text
CPU-only (no Metal), so it runs before the gemma4MoeMLP kernels exist. The CUDA MoE fixture had to
be rebuilt once (9275f94) because the original couldn't hold a tolerance — degenerate routing at
68.8% agreement, 0.77 logit floor. A fixture whose int4-vs-f32 logit cosine or routing agreement
sits below the near-tie bar cannot gate Metal regardless of how good the port is; better to learn
that here than after the MoE forward is wired.
```

## TestGemma4MoE_localize.floor

Moved from `metal/gemma4_moe_parity_test.go` (the comment at the worst-layer floor) on 2026-10-09.

```text
(S1.0 amendment, docs/tasks/task-multimodal-support-2026-10.md: raised from 0.90 after the dense layer scalar and v_norm fixes, between the before-fix 0.999644 and after-fix 0.999982 readings)
```

## TestGemma4MoE_residentParity.mincos

Moved from `metal/gemma4_moe_parity_test.go` (the comment at the minCos floor) on 2026-10-09.

```text
(S1.0 amendment, docs/tasks/task-multimodal-support-2026-10.md: raised from 0.60 after the dense layer scalar and v_norm fixes, between the before-fix 0.872764 and after-fix 0.998690 minCosine readings, s10-both.log and s10-after.log)
```

## TestGemma4TwoGeom_localize.bars

Moved from `metal/gemma4_twogeom_test.go` (the comment at the attribution asserts) on 2026-10-09.

```text
Attribution asserts. Both bars are 0.999 since S1.0's amendment (below). Before it they were 0.95, on the
belief that int4-Metal (f16 group scales) against int4-CPU (f32 scales) floors at ~0.98 per layer; layer 0
read 0.988 with no K=V. That ~0.98 was the two Gemma 4 bugs S1.0 fixed (the dense layer scalar and v_norm),
not quantization: with them fixed, layer 0 reads 1.000000 and layer 1 0.999940. A seam or K=V break (a
misthreaded geometry, the 2x v_norm trap) still craters a layer far below either bar.
(S1.0 amendment, docs/tasks/task-multimodal-support-2026-10.md: raised from 0.95 after the dense layer scalar and v_norm fixes, between the before-fix 0.987958 and after-fix 1.000000 readings)
```

## TestGemma4TwoGeom_localize.c1

Moved from `metal/gemma4_twogeom_test.go` (the comment at the layer-1 bar) on 2026-10-09.

```text
(S1.0 amendment, docs/tasks/task-multimodal-support-2026-10.md: raised from 0.95 after the dense layer scalar and v_norm fixes, between the before-fix 0.980768 and after-fix 0.999940 readings)
```

## TestGemma4TwoGeom_f16ScaleConfound

Moved from `metal/gemma4_twogeom_test.go` (the comment above `TestGemma4TwoGeom_f16ScaleConfound`) on 2026-10-09.

```text
TestGemma4TwoGeom_f16ScaleConfound isolates ONE candidate for the ~0.98 resident-vs-CPU cosine:
the int4 group-scale representation. The resident backends store scales as f16 (CUDA ws16 / Metal
f16 / WebGPU f16-unpack); the default CPU int4 path keeps them f32. Loading the CPU reference with
GOINFER_INT4_F16_SCALES=1 gives both sides the identical f16 scales, so THIS variable is removed.

FINDING (recorded, not inferred): it moves the floor by ~nothing (0.9806 → ~0.981). So the group
scales are NOT the confound — which is unsurprising in hindsight (f16-rounding a scale is a ~5e-4
perturbation, not the ~2e-2 seen). This comment used to blame the residual ~0.98 on the broader
resident quant path (f16 KV, int8 activations). It was not that: S1.0 (docs/tasks/
task-multimodal-support-2026-10.md) found two Gemma 4 bugs on Metal, the dense layer scalar and v_norm,
and with them fixed this test reads 0.999761 (s10-after.log).

The assertion is a crater backstop only: removing a benign confound must not make things worse and
must not reveal a crater. It deliberately does NOT assert 0.999 — that would encode the falsified
"scales are the confound" hypothesis.
```

## TestGemma4TwoGeom_f16ScaleConfound.floor

Moved from `metal/gemma4_twogeom_test.go` (the comment at the f16-scale floor) on 2026-10-09.

```text
(S1.0 amendment, docs/tasks/task-multimodal-support-2026-10.md: raised from 0.95 after the dense layer scalar and v_norm fixes, between the before-fix 0.981251 and after-fix 0.999761 readings) The ~0.98 this
test used to attribute to "the broader resident quant path" was the two Gemma 4 fixes, not quantization.
```

## TestGemma4TwoGeom_residentParity

Moved from `metal/gemma4_twogeom_test.go` (the comment above `TestGemma4TwoGeom_residentParity`) on 2026-10-09.

```text
cosine floor (0.979), NOT Metal's looser inherited gemma3 bar (0.88): a near-tie argmax mismatch
```

## TestGemma4TwoGeom_residentParity.floor

Moved from `metal/gemma4_twogeom_test.go` (the comment at the secondary cosine floor) on 2026-10-09.

```text
SECONDARY, deliberately LOOSE: this cosine is vs the f32-scale CPU, so it floors at the
int4-Metal(f16 scales)-vs-int4-CPU(f32 scales) representation gap (~0.98 on this fixture) — it
cannot detect a small quality regression, only a crater. The SENSITIVE cosine gate is
TestGemma4TwoGeom_f16ScaleConfound, which removes the scale confound. Keep 0.90 here purely as
a "not obviously broken" backstop; do NOT tighten it toward the noise floor (that was the trap).
(S1.0 amendment, docs/tasks/task-multimodal-support-2026-10.md: raised from 0.90 after the dense layer scalar and v_norm fixes, between the before-fix 0.981251 and after-fix 0.999761 readings)
```

## TestGemmaConfirmer_MatchedInput

Moved from `metal/gemma_confirmer_test.go` (the comment above `TestGemmaConfirmer_MatchedInput`) on 2026-10-09.

```text
TestGemmaConfirmer_MatchedInput is the Metal half of the matched-input confirmer (the CUDA box's
TestGemmaConfirmerReference assembles the reference; this injects it). It settles the last cut on
Gemma's dormant crater: int4-direct proved the WEIGHTS are not the cause (L0=1.0 but L1 still
craters to 0.640 with byte-identical weights), leaving two candidates —
```

## seedPrompt

Moved from `metal/gemma_parity_test.go` (the comment above `seedPrompt`) on 2026-10-09.

```text
seedPrompt derives the probe from the MODEL'S OWN tokenizer and VERIFIES it by decoding —
never a hardcoded id literal.

This gate used to carry invented ids that nobody had ever decoded. Gemma's read
"<bos>ath হই of carry Bত্ব忽视ardRep" — not Gemma tokens at all — so every Gemma parity number
on either backend was measured on gibberish, and the resulting "Gemma is noisier than the
control" was a rationalization of a confound in the test data. The control's were no better
("The history of_init with a text of a **"): real tokens, near-nonsense text. Flat logits from
nonsense produce exactly the extra near-ties that got blamed on architecture.

Both models now probe the SAME sentence, so they are actually comparable. Encode is used where
the vocab has merge ranks; Gemma's GGUF ships scores instead, so its pieces are looked up
directly — either way the result is decoded back and logged, so a bad prompt cannot hide again.
```

## residentParity

Moved from `metal/gemma_parity_test.go` (the comment above `residentParity`) on 2026-10-09.

```text
observeCos folds one position's logit cosine into the running min. NaN/Inf is COUNTED, never
fed to the `< minCos` reduction: `NaN < x` is false in Go, so a degenerate (NaN) cosine — the
signature of the worst bugs — would otherwise never update minCos and would sail through the
floor as if parity held. This is the exact vacuity parity-coverage-policy.md § Falsifiable
names, and TestParity_NaNCosineFailsTheGate breaks it on purpose to prove this guard fires.
residentParity drives Metal resident decode against the CPU forward in greedy LOCKSTEP — the
shipped metal convention (model_test.go): the CPU's argmax drives both sides, so they walk a
coherent trajectory instead of an arbitrary id sequence full of near-ties.

On the bar: the bar was set when Metal had NO like-for-like CPU reference. BuildResident took an
int8 load and re-quantized it to its own W4A8 (group=32, scale=max/7), which no CPU load
reproduces, so this was int4-GPU vs int8-CPU. CUDA's 3%-near-tie bar does NOT transfer; it
compares int4-vs-int4. Measured here on the KNOWN-GOOD dense path, CUDA's bar fails. That is why
the control is committed: the bar is read off it, not assumed — the same lesson CUDA learned when
a cosine >= 0.999 draft failed its own shipped path. The threshold is the usual bug. Since slice 1
of docs/tasks/task-metal-int8-2026-10.md a dense int8 model runs its int8 weights natively, so for
those this is int8 against int8, and TestW8Native_F2 holds the pair to the int4 pair's agreement.
```

## residentParityAt.sink

Moved from `metal/gemma_parity_test.go` (the comment at the <bos> sink skip) on 2026-10-09.

```text
Skip the <bos> sink positions in the metric. Gemma's <bos> is an ATTENTION SINK whose
value vector is trained near-zero (|V| 9.4 vs 129 for an ordinary token), so a cosine
there is dominated by rounding — and the position after it attends to that sink. Both
read as catastrophic (-0.047) while the model is fine: measured dNLL decays 15.9 -> 0.06
nats as real keys accumulate, and it generates " Paris." correctly. Gating on min-cosine
over these positions reported the two places the metric is meaningless.
```

## TestR18InSequence.pair

Moved from `metal/gemv_r18_seq_test.go` (the comment at the pair measurement) on 2026-10-09.

```text
pair: `steps` matched pairs, each a full token then the same token with category c no-op'd (4 = all four),
adjacent in time, so a GPU clock change between measurements cannot land on one side of the difference —
measured 2026-09-26 on the 1.5B, 20-step blocks per side let one block shift wholesale (a rep's summed work
read 4.2 ms against 8-9 in the others). Returns the median full token and the median per-pair difference.
```

## TestGemma3_GeneratesCoherently

Moved from `metal/gen_test.go` (the comment above `TestGemma3_GeneratesCoherently`) on 2026-10-09.

```text
THE decisive test, and the one the debug report wrongly called blocked: does Gemma 3 actually
GENERATE on Metal? Merges are encode-only, so a decode-only vocab is enough to read the
model's own output back. Also reports dNLL of the forced next token — the tokenizer-free
"is it actually broken" metric that argmax (trajectory-sensitive: the same known-good path
scores 15/24 vs 20/24 on different id sets) cannot give.
```

## TestGemma3_GeneratesCoherently.prompt

Moved from `metal/gen_test.go` (the comment at the prompt construction) on 2026-10-09.

```text
Build a REAL prompt from vocab lookups. The gate's inherited ids decode to
"<bos>ath হই of carry Bত্ব忽视ardRep" — they are not valid Gemma tokens, so every parity
number measured with them was measured on nonsense. Encode() is unavailable (decode-only
vocab), but TokenID is enough: SPM marks a leading space with ▁.
```

## TestGPT2ResidentParity

Moved from `metal/gpt2_real_test.go` (the comment above `TestGPT2ResidentParity`) on 2026-10-09.

```text
Dormant until all four features are declared (same pattern as TestGemma3ResidentParity): metal
ships the LayerNorm/non-gated-MLP/learned-pos/out-bias kernels and the encodeLayer/
encodeAttention wiring, but does not yet DECLARE them, so gpt2 still declines to CPU — skip
rather than fail, and residentParity t.Fatals on a decline once the declaration lands (catching
a silent CPU fallback rather than an honest skip).
```

## TestGPT2ResidentParityMetal

Moved from `metal/gpt2_resident_parity_test.go` (the comment above `TestGPT2ResidentParityMetal`) on 2026-10-09.

```text
TestGPT2ResidentParityMetal is a whole-model resident-vs-CPU gate for GPT-2 -- the family that
dispatches layernorm_quant with hasBias=1 (metal/kernels.go; Cohere is the bias-free caller but
does not go resident on Metal at all, declined for unimplemented features [logit-scale,
parallel-block] -- confirmed directly before writing this test). layernorm_quant had NO
whole-model coverage before this (only the isolated TestLayerNormQuant unit-kernel test). Built
specifically because rmsnorm_quant (rounds 7-9) demonstrated that an isolated kernel test can
pass exactly while a real bug still shows up only at the whole-model level -- this closes that
gap for layernorm_quant before any autoresearch candidate touches it. Mirrors
qwen35_resident_parity_test.go's structure, generic (no recurrent-state specifics): resident vs
CPU over many tokens (drift check), then a replay after Reset (KV cache must actually clear,
not merely start empty).
```

## gptoss_kernels_test.go.header

Moved from `metal/gptoss_kernels_test.go` (the comment at the top of the file) on 2026-10-09.

```text
actually ships. FeatAttnSink is now declared for Metal and the resident runs all three end to end
(TestGptOssResidentParity); this file was written before that wiring (D-D01, audit-metal-2026-09-30.md).
```

## TestGptOssMoEDownBias_pagedIndexesBiasByExpert

Moved from `metal/gptoss_kernels_test.go` (the comment above `TestGptOssMoEDownBias_pagedIndexesBiasByExpert`) on 2026-10-09.

```text
C-09: THE PAGED PATH INDEXED THE BIAS TABLE BY THE WEIGHT INDEX.

encodeMoEExpertsPaged substitutes a zero buffer for rIdx so the reused GEMVs read row 0 of a
one-expert slot — correct for the WEIGHTS, which is what the slot holds. But the same index
addressed ml.expDBias / ml.expGuBias, which stay the STACKED all-expert tables, so every routed
expert got expert 0's bias. Finite, plausible, wrong: the class CUDA fixed in d9829ce, here
indexed by a constant zero.

This is the paged shape exactly: one expert's weights staged at slot row 0, the bias table
stacked, and the two indices therefore DIFFERENT. Expert 0's bias is a decoy, so the pre-fix
addressing is not merely inaccurate — it is unmistakable.
```

## TestGptOssResidentParityReal20B

Moved from `metal/gptoss_real20b_test.go` (the comment above `TestGptOssResidentParityReal20B`) on 2026-10-09.

```text
gpt-oss forward on the resident path" on EITHER backend, and 2224441 is the precedent for why
that matters: a declaration made on kernel-level parity was correctly reverted. This is that
```

## TestGptOssResidentParityReal20B.path

Moved from `metal/gptoss_real20b_test.go` (the comment at the asset path) on 2026-10-09.

```text
decoder.AssetPathForTest, NOT modelPath: this used to call modelPath("gpt-oss-20b-MXFP4.gguf"),
which reads GOINFER_MODELS_DIR — a DIFFERENT variable from the one the comment claimed to
honour. It satisfied TestAssetRegistry_noDirectReads (a source-text regex over
os.Getenv(...) of that name, which this call never spelled) while actually bypassing
the registry's real GOINFER_GPTOSS_GGUF override entirely (audit-2026-09-02.md N-41, found
2026-09-11). AssetPathForTest resolves the SAME registry entry decoder's own
TestGptOssSafetensors_vsGGUF uses, and skips with the reason when absent.
```

## TestGptOssResidentParityReal20B.memguard

Moved from `metal/gptoss_real20b_test.go` (the comment at the fits-in-RAM guard) on 2026-10-09.

```text
FITS-IN-RAM GUARD — this test WILL hang a machine without it, and nothing in the engine
stops it. Measured 2026-08-31 on a 16 GB MacBook: loading this 11.28 GB checkpoint on the
Metal resident path drove swap to 35.98 GB of 36 GB (885 MB free), left the process in
uninterruptible I/O wait at 29% CPU with RSS creeping 1.8 -> 2.0 GB over 12 minutes, and
never completed or declined. The resident path has a KV CONTEXT cap (metal/backend.go, the resident context cap)
but NO weight-size feasibility check, so it accepts a model larger than RAM and thrashes.
Keyed on bytes computed here, not on the OS's account of what is free: Darwin's UBC reclaims
under pressure, so "available" reports what survived rather than what can be asked for.
```

## TestGptOssResidentParity

Moved from `metal/gptoss_real_test.go` (the comment above `TestGptOssResidentParity`) on 2026-10-09.

```text
Dormant until FeatAttnSink is declared: metal ships the sink term (kernels.go's `attention`),
the clamped-SwiGLU expert + custom router (moe.go's swiglu_quant_gptoss/route_gptoss) and the
bias-in-combine down projection (gemv_w4a8_moe_wacc_bias), and the moe.go isGptOss wiring, but
does not yet DECLARE the feature — skip rather than fail, and residentParity t.Fatals on a
decline once the declaration lands (catching a silent CPU fallback rather than an honest skip).
```

## gvDeepstackCompare.planted

Moved from `metal/grid_vision_deepstack_test.go` (the comment at the planted defect) on 2026-10-09.

```text
The planted defect: every tap a block late. The tiny tower has three blocks with DeepStack at 0 and 1 (re-pinned
2026-10-07; it had two, so the late tap fell past the end and any refusal counted as red), so the late taps, 1 and 2,
are inside the tower and the defect is measured as a number: it must come back as sets, under the bar.
```

## requireHeavyModel

Moved from `metal/heavytest_test.go` (the comment above `requireHeavyModel`) on 2026-10-09.

```text
The bug this closes: metal's real-model tests decided whether to run by PATH EXISTENCE alone
(os.ExpandEnv("$HOME/models/...") → skip if absent, else Load()). On a box with the model zoo
present, `go test ./metal/` fired them all opportunistically — dozens of GB-scale loads that
blew the 10-minute timeout (and one, TestGemmaBisect_PerLayer, panics on the qwen control's
hidden dim). The asset happening to be on disk is not a request to run a multi-GB test. The
per-test os.Stat skip stays as a second guard, so opting in on a bare box is still harmless.
```

## TestImagePrefillResident_gemma3TwoImagesReal

Moved from `metal/image_prefill_resident_real_test.go` (the comment above `TestImagePrefillResident_gemma3TwoImagesReal`) on 2026-10-09.

```text
differences against the reference and whether each is an R10 near-tie. The first read (2026-10-09) had one that is not
(step 1, " are" 0.819 against "'" 0.143 in the reference), the G-IP4 pattern; whether it is accepted for several images is
the owner's open decision (docs/tasks/task-multimodal-support-2026-10.md, S11 step 4). GOINFER_HEAVY_TESTS=1.
```

## TestGemma_Int4DirectContext

Moved from `metal/int4direct_test.go` (the comment above `TestGemma_Int4DirectContext`) on 2026-10-09.

```text
TestGemma_Int4DirectContext validates the int4-direct fix. The default resident path
double-quantizes weights (f32→int8→int4); Gemma's low-magnitude attention contexts amplify the
int8-intermediate drift so the pre-o-proj context craters (cos(Metal,int4-ref) = 0.39/0.52/0.57
at L31-33, L1 already 0.649 — metal/gemma_sublayer_test.go). int4-direct consumes the decoder's
int4 nibbles verbatim (what CUDA does), removing the int8 step. If the mechanism is right, a
resident built from a Quant:"int4" model should produce a context that TRACKS the int4 forward
(goinfer's own int4 == CUDA-int4) instead of cratering.

Success criterion (the CUDA box's): Metal-int4-direct context vs int4-ref ≈ 1.0 (same weights,
faithful kernels), and vs f32-truth ≈ 0.92/0.85/0.91 (the int4-quant bar CUDA also sits at) —
NOT the double-quant path's 0.39.
```

## TestGemma_Int4DirectContext.finding

Moved from `metal/int4direct_test.go` (the comment at the FINDING log) on 2026-10-09.

```text
The FINDING (logged, not a failure): L1 still craters (~0.64), matching the double-quant path's
0.649. int4-direct removed the int8 intermediate and made ZERO difference to the crater — so
the weight double-quant was NOT the cause. With byte-identical weights Metal still diverges
from the CPU int4 forward at L1, which localizes the bug to Metal's reduced-precision COMPUTE
(f16 KV cache / f16 activations), amplified by Gemma's sensitive attention — not the weights.
```

## TestDumpLogitsForBisect

Moved from `metal/logit_dump_test.go` (the comment above `TestDumpLogitsForBisect`) on 2026-10-09.

```text
TestDumpLogitsForBisect writes the resident GPU logits for a fixed 4-token sequence to
$GOINFER_DUMP_LOGITS as raw little-endian float32. It exists to back the 9c Step-1
"byte-identical" claim with an actual BITWISE diff (parent commit vs HEAD), not just a
cosine-vs-CPU threshold: run it on the parent, run it on HEAD, `cmp` the two files. It uses
only BuildResident + ForwardEmb + EmbedResidentForTest — API identical across the refactor —
so the same source compiles and runs on both commits. Skips unless the env var is set.
```

## TestSetAdapter_partialBindErrorReleasesBuffers

Moved from `metal/lora_bind_leak_test.go` (the comment above `TestSetAdapter_partialBindErrorReleasesBuffers`) on 2026-10-09.

```text
TestSetAdapter_partialBindErrorReleasesBuffers gates C-04 (audit-metal-2026-09-12.md):
SetAdapter's conversion loop used to return immediately on the first bad projection, leaving
every earlier layer's already-converted device buffers (A/B plus their uniform buffers) on the
device ledger — referenced by nothing (r.loraLayers/r.loraCached are never set on an error
return), so they leaked until Close. Every failed bind attempt — a bad rank is the easy way to
trigger it, but any mid-loop error does — added to the leak.

llama-tiny (4 layers) lets layer 0 convert cleanly before layer 1's invalid rank fails the bind,
exercising the actual partial-progress path rather than failing on the very first projection.
```

## TestLoRAResidentParityMetal.floor

Moved from `metal/lora_resident_parity_test.go` (the comment above `TestLoRAResidentParityMetal.floor`) on 2026-10-09.

```text
N-14/N-52 (audit-metal-2026-09-12.md, audit-2026-09-10.md): this floor is far looser than
what a correct bind actually measures on this machine — 0.999969 here, 0.998835 on the
armed-executor variant below (both this session's real runs, not a single cherry-picked
number) — so a bug that drops one of the seven per-layer projections could plausibly still
clear 0.95 if that projection's contribution is small relative to total variance; the
vacuousness check below only proves "some effect survives", not "every targeted projection
fired". NOT tightened here: two single-machine runs a few thousandths apart is not enough to
pick a real "measured floor minus noise" number with confidence, and a mis-set tight floor
risks flaking CI on legitimate cross-machine/quantization variance — parked per N-52's own
recommendation, not rejected.
```

## TestMC3Chain_newcomerJoinsLongPrompts

Moved from `metal/mc3_chain_long_test.go` (the comment above `TestMC3Chain_newcomerJoinsLongPrompts`) on 2026-10-09.

```text
TestMC3Chain_newcomerJoinsLongPrompts and TestMC3Chain_stalledConsumerLongPrompts are the two TestMC3Chain_ scenarios
with prompts long enough for the batched pass (100/80 and 90/70 tokens) instead of the step. Their alone runs are
cold and their joint runs reuse the slot the alone run left, so they also pin warm-against-cold identity: before
decoder.PrefillTailExact a warm prompt re-ran its last position through decode where the cold one had used the pass,
and both generations diverged tens of tokens in (A at token 42 of 160, B at 47 of 64). That was first read as an MC3
batching defect; batching is bit-identical (the joint run on a fresh model matches), the cold/warm route was not.
```

## mc3ChainFixture

Moved from `metal/mc3_chain_test.go` (the comment above `mc3ChainFixture`) on 2026-10-09.

```text
The greedy and sampled chains (C-B01 / C-P02) under MC3: a generation alone on the batcher runs the chain with the
resident held across its tokens (decoder's holdSolo), and hands the resident back the first token anyone else wants
it. Before this, every generation on a model with more than one KV slot took an MC3 place, the chains required
mc3 == nil, and serve (2 slots by default on Metal) never ran them (the post-merge peer read, 2026-10-03). These
tests drive production's Generate with the batcher on, which is where that gap hid: the chains' own A/Bs ran with
no batcher at all.
```

## TestMC3Step_qwen3BitIdentical

Moved from `metal/mc3_step_test.go` (the comment above `TestMC3Step_qwen3BitIdentical`) on 2026-10-09.

```text
TestMC3Step_qwen3BitIdentical (E-P07): a Qwen3-shaped model (per-head QK-norm) runs the batched step, and its rows
equal production's single-token forward bit for bit, as TestMC3Step_bitIdentical checks for the qwen2 shape. On the
generated qwen3 fixture by default; under GOINFER_METAL_MC3=1 the identity suite runs on whatever checkpoint
GOINFER_METAL_MC3_MODEL names (the real Qwen3-0.6B read 0 differing values, 2026-10-04).
```

## TestMC5_prefillChunkInvariance.sizes

Moved from `metal/mc5_chunk_invariance_test.go` (the comment above `TestMC5_prefillChunkInvariance.sizes`) on 2026-10-09.

```text
100 and 77 start chunks off the steel kernel's 32-row tiles; a prefix-reuse turn's startPos is arbitrary too. The
aligned sizes cannot see a bug at a tile edge, which shows the same way whole and chunked: with the kernel's causal
limit moved one key, every aligned size still matched bit for bit, and 100 and 77 differed from the first chunk
boundary on (2026-10-01, F-G02). 512 is serve's default chunk. 81 leaves a 28-token tail at position 972: the
step route (E-P01, promptStepOK) must not take a short chunk above the floor, or the tail runs decode's numerics
and differs from the whole pass (measured with the route open there: all 20480 logits).

16, 32 and 48 cross A-P01's GEMM tile selector (gemmTile: 32-token tiles at <= 32 rows, 32-feature tiles for the
narrow GEMMs at <= 64 rows), so their chunks run the smaller tiles where the whole pass runs 64 × 64. Their first
chunk is 64 tokens: a chunk that ends below the fast-prefill floor never reaches the batched pass (it ran
sequentially before E-P01 and on the step kernels since, decode's numerics either way), so it cannot equal it.
```

## TestMellumResidentParity

Moved from `metal/mellum_real_test.go` (the comment above `TestMellumResidentParity`) on 2026-10-09.

```text
TestMellumResidentParity is G11's Metal half — the real-weight successor to the abandoned
synthetic-random-weight attempt (G10, docs/queue-correctness.md). G10 declared FeatRopeMscale
for Metal to unblock gpt-oss's YaRN, which as a documented side effect also admits Mellum onto
the Metal resident path with zero end-to-end validation there. decoder/mellum_slice_test.go
(tag realckpt) already proved goinfer's own CPU forward matches the real HF reference on a
REAL 4-layer weight slice (argmax 417, cosine 1.00000000) — same discipline as
TestGPT2ResidentParity/TestGptOssResidentParity, just pointed at the sliced checkpoint
directory instead of a small full model. Regenerate the slice per docs/queue-correctness.md
G11 if it's absent (4 GB, gitignored — only the golden ships).
```

## TestMinistral3ResidentSmokeMetal

Moved from `metal/ministral3_resident_parity_test.go` (the comment above `TestMinistral3ResidentSmokeMetal`) on 2026-10-09.

```text
TestMinistral3ResidentSmokeMetal is G5's FeatAttnTemp row (docs/tasks/task-gpu-paths-2026-09.md)
smoke gate: the model actually goes resident and produces finite, non-degenerate output when
the post-RoPE query scale is genuinely exercised (testdata/ministral3-tiny's
AttnTempOrigMaxPos=8, so a 32-token run steps through four distinct scale values, past the
identity-at-short-prompts trap AttnTempBeta's own comment warns about).

This is DELIBERATELY NOT a resident-vs-CPU cosine floor — same finding as G5 row 1's
smollm3-tiny (metal/smollm3_resident_parity_test.go's TestSmolLM3ResidentSmokeMetal has the
full writeup). MEASURED here too, not assumed: with the real fix and with it force-disabled
(qTempScale always 1, i.e. the attn-temp scale silently dropped on every position), the worst
cosine against the CPU reference over 32 tokens was 0.96446 and 0.96525 respectively — a
~0.0008 spread, indistinguishable from noise. testdata/ministral3-tiny is ALSO seeded/synthetic
(scripts/pin_ministral3_tiny.py), so despite its own test file's comment about deliberately
exercising AttnTempOrigMaxPos and a real YaRN mscale ratio nontrivially, int8-on-random-weights
noise still dominates a whole-model resident-vs-CPU comparison at this scale.

The actual correctness gate is decoder.TestAttnTempScale_matchesSequentialFormula — a pure,
backend-agnostic unit test of the exact formula both Model.AttnTempScale (decode) and
Model.AttnTempParams (CUDA's batched prefill, which must recompute it per row device-side)
expose, no GPU, no quantization noise. This test's only job: does declaring FeatAttnTemp let
the model go resident and run without error/NaN.
```

## TestMoEExpertReuseProbe

Moved from `metal/moe_expert_reuse_probe_test.go` (the comment above `TestMoEExpertReuseProbe`) on 2026-10-09.

```text
TestMoEExpertReuseProbe investigates M-05 (audit-metal-2026-09-12.md) cheaply, BEFORE building
the expert-major prefill restructuring it proposes: on a REAL paged MoE resident, run genuine
text through the EXACT path paged MoE prefill already uses today — prefillOK is hard-false for
a paged generic MoE (metal/model.go), so every prompt token already goes through the sequential
per-token decode loop, one router readback + stage per token per layer — and read off the
expert pool's own telemetry (stages / distinctExperts / nE) to see whether an expert-major
regroup (stage each routed expert ONCE per prefill instead of once per token that routes to it)
would converge close to nE (little win left — routing already touches most experts anyway) or
stay far below today's stage count (real cache churn from LRU thrashing, so M-05 is worth
building).

Manual/one-off by design (a real multi-GB checkpoint, several minutes of GPU time) — gated
behind GOINFER_MOE_REUSE_PROBE_CKPT rather than GOINFER_HEAVY_TESTS' usual asset registry, same
convention as TestMoEPrefillMeasure_batchedVsSequential (moe_prefill_measure_test.go). Unlike
that test, this one REQUESTS Metal's own GPU expert-cache paging (Options.MoECacheExperts) with
slots left at 0 (auto-sized, M-13) rather than requiring the whole expert set resident — the
qwen15-moe-a27b shape this repo already has on disk needs ~16.7 GB resident non-paged, which
does not fit a 16 GB Mac at all (confirmed 2026-09-10, ~/models/moe_prefill_measure.log), but
paged only holds N << nE experts per layer.

	GOINFER_MOE_REUSE_PROBE_CKPT=~/models/qwen15-moe-a27b GOINFER_MOE_REUSE_PROBE_M=512 \
	  go test -tags metal ./metal/ -run TestMoEExpertReuseProbe -v -timeout 30m
```

## TestMoE_declinesPrefill

Moved from `metal/moe_model_test.go` (the comment above `TestMoE_declinesPrefill`) on 2026-10-09.

```text
TestMoE_declinesPrefill: MoE moved from the decline side to the admit side on 2026-09-08 (G8
MoE half, docs/tasks/task-gpu-paths-2026-09.md) — the f16 MMA prefill path now runs a MoE layer's FFN
row by row off the batched residual, reusing the unchanged per-token decode MoE dispatch chain
(encodeMoERoute/encodeMoEExperts/encodeMoESharedExpert, metal/moe.go); the name is historical
(kept so `git log -p` on it still tells the right story — decoder/gptoss_decline_test.go
precedent). Both MoE and its dense twin must now accept.

Reuses the same identical-experts trick as TestMoE_assemblyVsDense above (8 identical experts +
zeroed shared expert ⇒ MoE FFN is mathematically equal to the dense FFN regardless of routing),
so this also gets a free numeric check that the dst-redirection (moeDst, docs/task-gpu-paths-
2026-09.md) didn't drop or double-count anything: MoE prefill logits must match dense prefill
logits closely, not just both "succeed".
```

## TestMoE_declinesPrefill.gate

Moved from `metal/moe_model_test.go` (the comment above `TestMoE_declinesPrefill.gate`) on 2026-10-09.

```text
This test exercises the ARCH admission (MoE now has a row-by-row FFN path) — not the
bit-identity decline that gates the Metal backend by default (54% divergence, §A2-Metal).
Opt past that outer gate so the arch logic is what's under test.
```

## TestMoEPrefillMeasure_batchedVsSequential

Moved from `metal/moe_prefill_measure_test.go` (the comment above `TestMoEPrefillMeasure_batchedVsSequential`) on 2026-10-09.

```text
TestMoEPrefillMeasure_batchedVsSequential is P-15's real-checkpoint measurement
(audit-2026-09-10): Metal's batched f16-MMA prefill (metal/prefill.go's PrefillLast) has been
default-ON above the 512-token floor for MoE since 306b16b, but §3.2's decision set that
justified default-ON was dense-only — nobody has actually timed the batched path against the
sequential per-token loop on a real, generic (non-Gemma-4) MoE. This times both, directly on
the raw resident (bypassing metalResident's floor/env-var wrapper, which only decides which
path a caller reaches — it does not change either kernel path's own cost), at K ∈ {512, 1024,
2048}.

Manual/one-off by design (real 28.6GB checkpoint, several minutes of GPU time) — gated behind
GOINFER_MOE_PREFILL_CKPT rather than GOINFER_HEAVY_TESTS' usual asset registry, since this is a
measurement script (docs/measurements/), not a correctness gate:

	GOINFER_MOE_PREFILL_CKPT=~/models/qwen15-moe-a27b \
	  go test -tags metal ./metal/ -run TestMoEPrefillMeasure_batchedVsSequential -v -timeout 30m
```

## TestMoECacheExperts_bitExactMetal

Moved from `metal/moecache_bitexact_test.go` (the comment above `TestMoECacheExperts_bitExactMetal`) on 2026-10-09.

```text
TestMoECacheExperts_bitExactMetal is G-1 (docs/tasks/task-option-path-admission-2026-10.md §4.3, finding 2, registered
before the code and amended with step 0's mechanism): Metal's expert cache (MoECacheExperts with fewer slots than
experts) is bit-identical to the fully resident load, through the real caller (decoder.Load, then Generate): the
generated tokens, their top-5 log-probabilities, and every layer's K/V.

The reference is the fully resident load with ExactPrefill. A paged model never takes the batched f16 prefill
(paging is never used by prefill), so against the DEFAULT resident load a prompt over the batched-prefill floor
differs by the batched prefill's own tolerance (0.0044 here), which is the gap finding 2 measured; step 0 showed
--exact-prefill alone reproduces it to the last bit and paging adds nothing. Two prompts, under and over the floor;
at top-k, top-k+1 and top-k+2 slots and at all but one, each asserted to have evicted. The generic MoE path
(mixtral-tiny: 8 experts, top-2) and Gemma 4's (gemma4-moe-tiny: 4 experts, top-2) page separately. Each planted
defect (expertPoolDefect) must go red: an evicted slot not restaged, the GPU told the next slot, a neighbour's scales.
```

## TestOptionPathMetal_kvPrecision

Moved from `metal/optiongrid_metal_test.go` (the comment above `TestOptionPathMetal_kvPrecision`) on 2026-10-09.

```text
TestOptionPathMetal_kvPrecision: KVPrecision's resident decode, speculative verify and session
cells. Metal keeps its resident KV at f16 whatever is asked (ResidentKVPrecision says so), so "f16"
must match the baseline exactly; "i8" is lossy, so its decode is held to a cosine floor against the
f16 baseline, its speculative path to exact agreement with its own plain decoding, and its session
reuse to agreement with a cold run of the same turn. Not exactness there: resident prefix reuse on
Metal is not bit-identical to a cold prefill even at f16 (about 0.004 in log-probability on this
model, kernels differing between decode and prefill), and resident_reuse.go does not claim it is.
```

## TestOptionPathMetal_prefillChunk

Moved from `metal/optiongrid_metal_test.go` (the comment above `TestOptionPathMetal_prefillChunk`) on 2026-10-09.

```text
TestOptionPathMetal_prefillChunk: ResidentPrefillChunk under MC3 prefills a long prompt in chunks while
another generation is decoding, and the chunked generation must match the same generation prefilled whole,
with the batcher's prefill-pass count showing the chunks ran.

Chunk 8 and 16 are the regression case (option-path admission finding 3, 2026-10-08): with two KV slots
Metal's batched prefill runs decode rows below a 32-token whole-prompt floor and f16 MMA from it, so a first
chunk under the floor ran on the other kernel class, and the reply differed from the whole prefill by
0.003-0.004 in log-probability: it depended on whether another generation was decoding. mc3Prefill now
raises a chunk below the resident's PrefillKernelFloor to the floor.
```

## TestPageCost_submissionStructure

Moved from `metal/pagecost_measure_test.go` (the comment above `TestPageCost_submissionStructure`) on 2026-10-09.

```text
TestPageCost_submissionStructure is Step-6 Step-0: PRICE what losing the value-independent
pre-encode costs, to decide whether a synchronous Metal expert-paging path is viable or whether
the speculative (prefetch-last-token's-experts) design is mandatory.

HONEST SCOPE (read this before trusting the number): the ask was to measure the GENERIC Mixtral-
class MoE path on a fitting checkpoint. There is NO such checkpoint on this Mac (only gemma4-26b,
which doesn't fit and is the new path), so this measures the SUBMISSION-STRUCTURE cost on a DENSE
model (qwen2.5-1.5b, 28 layers) — the dominant, architecture-independent term (1 command buffer/
token vs ~nL/token). It is a faithful LOWER BOUND on the per-layer MoE regime: MoE adds the router
readback + expert-stage host work at each boundary, which overlaps the GPU-idle gap this already
pays for. It is NOT the MoE-path number and is not reported as one. Option 3 (MTLSharedEvent
handshake — one submit/token + per-layer CPU↔GPU events) needs aikit bindings that don't exist yet;
it is measured only if regime (2) here is expensive enough to matter.

Reports baseline (pipelined pre-encode) and per-layer-submit as tok/s + ms/token, best of 3 warm
runs (first discarded), greedy, same model + prompt. Heavy-gated (loads a ~1 GB checkpoint).
```

## TestPageCost_submissionStructure.baseline

Moved from `metal/pagecost_measure_test.go` (the comment above `TestPageCost_submissionStructure.baseline`) on 2026-10-09.

```text
Baseline is ForwardEmb (INLINE single command buffer/token), NOT ForwardEmbPipe: the pipelined
path runs a persistent executor GOROUTINE, and measuring the inline per-layer regime while that
goroutine is alive contends the single Metal GPU and inflates per-layer (an earlier revision
read +106% that way; clean it is ~+43%). Both regimes measured inline, same conditions. The
encode-ahead OVERLAP that Pipe adds is small here anyway (~1-2%) — decode is GPU-bound, so the
dominant cost is the SUBMISSION STRUCTURE (1 command buffer/token vs ~nL), which this isolates.
```

## TestPageCost_sharedEventReal

Moved from `metal/pagecost_sharedevent_test.go` (the comment above `TestPageCost_sharedEventReal`) on 2026-10-09.

```text
TestPageCost_sharedEventReal is Step-6 Step-0 regime (3) on the REAL forward — the authoritative
measurement that the synthetic aikit probe (gpu/metal_sharedevent_test.go) understates. It reprices
all three regimes inline on qwen2.5-1.5b so shared-event is directly comparable to baseline and
per-layer-submit. FINDING (2026-08): shared-event handshake (~0.26 ms/boundary) is ≈ per-layer
submit (~0.23 ms/boundary) — recovers ~0%; both synchronous shapes cost ~+45%. Conclusion:
synchronous Metal MoE paging is not viable by either shape → speculative prefetch is the path.

This is the committed, repo-reproducible form of the regime-3 result (was run under a local go.work

	override at Step-0). Run: GOINFER_HEAVY_TESTS=1 GOINFER_HANDSHAKE_PROBE=1 go test ./metal/ \
	  -run TestPageCost_sharedEventReal -v  (loads a ~1 GB checkpoint; needs aikit/gpu >= v0.23.0).
```

## TestPairwiseRoPERealMetal.meanbar

Moved from `metal/pairwise_rope_real_test.go` (the comment above `TestPairwiseRoPERealMetal.meanbar`) on 2026-10-09.

```text
The mean bar is 0.99 for every model but Command-R7B, whose 0.989242 the owner accepted on 2026-10-09 ("close
enough, so it's a go"; docs/tasks/task-metal-pairwise-rope-2026-10.md G-PR5). Part C of
docs/tasks/task-metal-pairwise-followups-2026-10.md localizes that gap; this floor records the decision, it is
not a measured tolerance.
```

## TestMetalPrefillDecomp.reps

Moved from `metal/prefill_decomp_test.go` (the comment above `TestMetalPrefillDecomp.reps`) on 2026-10-09.

```text
Every repeat of every category that matters, so a wide spread can be traced to its reps
(S0's first run printed medians only, and its two >50% spreads could not be read back).
```

## TestMetalPrefillDecomp.loo

Moved from `metal/prefill_decomp_test.go` (the comment above `TestMetalPrefillDecomp.loo`) on 2026-10-09.

```text
Leave-one-out (GOINFER_METAL_DECOMP_LOO=1): a category timed in its own command buffer ran in two
modes ~2x apart on the MLP GEMMs at K=3900 (S0's first run, 52-55% spreads), and in a stable
all-fast run the categories summed to only 64.6% of the full replay — so an isolated time is not
that kernel's cost inside production. Here each large category's IN-SEQUENCE cost is the full
replay's time minus the time of the same replay with that category's dispatches removed, rep by
rep interleaved. Kernel timing does not depend on the values it reads (no data-dependent
branches), so the stale inputs a removed category leaves behind change no downstream kernel's
work — only the cache and memory state it sees, which is the effect being measured.
```

## TestMetalPrefillDecomp.state

Moved from `metal/prefill_decomp_test.go` (the comment above `TestMetalPrefillDecomp.state`) on 2026-10-09.

```text
Prior-state test (GOINFER_METAL_DECOMP_STATE=1): the MLP GEMMs run ~1.9x slower inside the sequence
than alone, at both K, and alone they flip between two levels ~2x apart. If that tracks the GPU's
RECENT WORKLOAD (a clock/power state) rather than the kernel's inputs, gate/up timed alone should
be slow straight after heavy work and fast after an idle gap. Four conditions, interleaved per rep.
```

## TestPrefillGateVsReference.ctxpin

Moved from `metal/prefill_gate_ref_test.go` (the comment above `TestPrefillGateVsReference.ctxpin`) on 2026-10-09.

```text
Pin ResidentContext to metalCtxCapDefault (4096): the 0 (backend-default) auto-cap sizes ctx off
AVAILABLE MEMORY, not this backend's fixed kernel-score-buffer ceiling, so on a box
with generous free RAM it picks something above metalCtxCapMax and BuildResident
declines outright ("resident context ... exceeds this backend's hard ceiling") —
falling back to CPU/staged, which fails this test's *metalResident type assertion.
4096 comfortably covers every decision/confirm cell here (2026-09-25: this pin used to be metalCtxCapMax,
written when that was 4096; it has been 32768 since 26f64807, so the fit guard priced a 32k KV — ~1.8 GB on
the 1.5B — and refused loads that fit): the
widest, K=3900 + continuationN(64) teacher-forced steps, tops out at pos 3962.
GOINFER_METAL_GATE_QUANT=int8int8 grades the native int8 path (docs/tasks/task-metal-int8-2026-10.md, slice
2): an int8int8 load with nativeInt8 on, both arms on its W8 kernels.
```

## runPrefillGateSet.g02

Moved from `metal/prefill_gate_ref_test.go` (the comment above `runPrefillGateSet.g02`) on 2026-10-09.

```text
G-02 (audit-metal-2026-09-12.md): a missing reference file used to silently drop that K
from the pool (runPrefillRefGateCellK returns nil, the loop above just skips appending it) —
so a pooled verdict that says "SHIPS" could rest on fewer decision cells than decisionKs
names, with nothing in the output calling that out unless a reader compares the header's K
list against decisionCells' own count by hand. Observed for real, repeatedly, this same
audit: S set "a" K=512 has no reference file and was pooled over silently in every prior
M-03/M-04 oracle run. A partial decision set is fine to REPORT (re-scoring, deciding=false,
is explicitly "never fails the test on its own" per this function's own doc comment above)
but must not silently DECIDE — Fatalf here, not there, so the message names exactly which K
is missing rather than requiring a reader to diff the header against the cell count.
```

## TestPrefillGate.floor

Moved from `metal/prefill_gate_test.go` (the comment above `TestPrefillGate.floor`) on 2026-10-09.

```text
G-07 (audit-metal-2026-09-12.md): this test's K=256 decision cell used to Fatalf outright —
metalFastPrefillFloor was 512 and nothing here overrode it, so PrefillLast declined before
any comparison ran. The floor has since dropped to 256 (M-02) and then 64 (R3), which clears
K=256 on its own, but disable it explicitly anyway so this test does not silently break
again the next time the floor default moves.
```

## TestPrefillGate.divergence

Moved from `metal/prefill_gate_test.go` (the comment above `TestPrefillGate.divergence`) on 2026-10-09.

```text
G-07 (audit-metal-2026-09-12.md): TestMetalPrefillDivergenceRate, the test that originally
measured this 54%, no longer exists in the tree (superseded by TestPrefillGateVsReference's
pooled §3.2 criteria, which withdrew the exact-as-oracle scoring that number came from) —
the figure itself is historical record, not a live test (docs/ollama-chase.md).
```

## runPrefillGateCell.exact

Moved from `metal/prefill_gate_test.go` (the comment above `runPrefillGateCell.exact`) on 2026-10-09.

```text
EXACT — sequential Forward per token, today's shipped default.

Forward's return is a REUSED buffer (metal/model.go ForwardEmbPipe: "Returns logits[V]
(reused buffer; consume before the next call)") — every capture below is cloned
immediately. Storing the raw slice instead silently aliases whatever the LAST Forward call
in the whole cell wrote, which is exactly the bug this comment is here to stop someone
from reintroducing: it first shipped that way, and the seed and every continuation position
all came back reading the same final buffer, producing a self-contradictory result (a
"42% seed gap" while position 0 of the continuation — which SHOULD be the same comparison —
quietly agreed).
```

## gemmS2Kernels

Moved from `metal/prefill_gemm_s2_test.go` (the comment above `gemmS2Kernels`) on 2026-10-09.

```text
gemmS2Kernels holds R16's prototypes (1-4; prototype 4 became production 2026-09-25) and the retired production
kernel gemm_w4f16_store_r15. It began as R16's first prototype (docs/tasks/red-october.md; design in
docs/measurements/metal-prefill-gemm-s2-2026-09-25.md): the int4 prefill GEMM restructured in the shape of
llama.cpp's classic kernel_mul_mm, which S1b measured at >= 2.96 TFLOPS on this machine against the current
gemm_w4f16_store's ~0.75 under sustained load. TEST-ONLY until R16's band is met — nothing in production
compiles it.
```

## TestPrefillParityMoEGatedShared

Moved from `metal/prefill_moe_parity_test.go` (the comment above `TestPrefillParityMoEGatedShared`) on 2026-10-09.

```text
TestPrefillParityMoEGatedShared exercises the OTHER branch encodeMoESharedExpert's
parameterization (this row) must get right: a sigmoid-GATED always-on shared expert
(FeatMoEGatedShared, Qwen2-MoE), not just the ungated GLM/DeepSeek shape the plain mixtral
fixture above never touches at all (mixtral has no shared expert).

The fixtures are the two committed gated-shared tiny checkpoints, both DeltaNet hybrids whose MoE layers take the same
batched shared-expert dispatch (since D-B01). It used to be testdata/tiny-qwen2-moe, a per-machine download whose
moe_intermediate_size of 44 the int4 group of 32 rejects, so BuildResident declined it and the test could only fail.
The cosine bar is 0.999, not the plain MoE test's 0.95: on these fixtures the gate is a small term, and dropping the
sigmoid (gate fixed at 1) still clears 0.95 on qwen35-tiny (0.99813) while reading 0.9998 / 0.9994 unmutated
(measured 2026-10-05). At 0.999 that mutation fails both.
```

## TestPrefillNoNaN

Moved from `metal/prefill_nan_test.go` (the comment above `TestPrefillNoNaN`) on 2026-10-09.

```text
TestPrefillNoNaN asserts the f16-MMA prefill produces finite logits across a range of prompt
lengths — the direct regression gate for the bug where the int8-pinned LM head was run through
the int4 gemm_w4f16 kernel (weights misread as packed nibbles), yielding NaN logits at EVERY M
including the minimal single-tile M=8. Complements TestPrefillParity (which pins the value); a
NaN here is the specific shipped-path failure that a hand-run caught only because no CI ran
Metal against a checkpoint.
```

## TestPrefillRefIdentity

Moved from `metal/prefill_ref_identity_test.go` (the comment above `TestPrefillRefIdentity`) on 2026-10-09.

```text
resident's. A reference generated from the same text lands at the W4A8-vs-CPU level (measured 0.0015–0.072 on
set B's S-K3900); one generated from different text lands orders of magnitude higher (3–18 on set A's). Found
2026-09-25: set A's 2026-09-05 files predate the 2026-09-09 snapshot (metal-decode-attn-r17-2026-09-25.md).
```

## TestPromptHiddenResidentMetal.int4

Moved from `metal/prompthidden_resident_parity_test.go` (the comment above `TestPromptHiddenResidentMetal.int4`) on 2026-10-09.

```text
Both sides run int4. Metal has no int8 GEMV (its decode path reports int8int8 -> int4), so an int8int8 CPU side compares
two quantizations, not two executions: measured 2026-09-30 on this fixture, CPU int8int8 vs Metal read cosine 0.977 with
the adapter merged, while CPU int4 vs Metal int4 reads 0.99985 (0.99998 without it). The adapter-merged tiny weights are
unusually int4-sensitive (CPU f32 vs CPU int4 is 0.980), which is a property of the fixture, not of the resident path.
```

## TestQuantBar_CPUInt4VsInt8

Moved from `metal/quantbar_test.go` (the comment above `TestQuantBar_CPUInt4VsInt8`) on 2026-10-09.

```text
TestQuantBar_CPUInt4VsInt8 answers the question every Metal Gemma parity number has been
unable to: how much of the gap is QUANTIZATION and how much is METAL?

The problem it solves: Metal has no like-for-like CPU reference by design. BuildResident
requires an int8 load and re-quantizes to its own W4A8, so residentParity is always
int4-GPU vs int8-CPU — a comparison that mixes two independent effects. Gemma measures
0.818 there and the control 0.990, and for months that delta was read as "Metal has a Gemma
bug". It might instead be "int4 costs more at Gemma's shape", and the two are indistinguishable
from a single number.

This fork separates them on the CPU alone, no GPU involved. decoder's int4 mode is the near-
exact quantization twin of Metal's resident W4A8: group-32 symmetric weights at scale=maxabs/7
(int4GroupSize, weightmat.go) and an int8 LM head + embedding (quantMode.embedding() pins them —
the same pin Metal ships). Same weights, same activations, same arithmetic class — the ONLY
thing that changes between the two runs is int4-vs-int8. So:

	cpu-int4 vs cpu-int8  =  the cost of the quantization class, at THIS model's shape
	metal    vs cpu-int8  =  that cost + whatever Metal adds

If Gemma's CPU fork lands near its Metal number (~0.82), then 0.818 IS the int4 bar at Gemma's
shape and there is nothing left to fix — the model ships. If the fork comes back ~0.99 while
Metal stays at 0.82, the quantization is exonerated and Metal really does have a Gemma-specific
bug, which reopens the hunt with the suspect list cut in half either way.

Both models run so the control calibrates the subject, per the lesson that a bar must be
measured on this box rather than assumed.
```

## TestQwen35_35B_pagedRuns

Moved from `metal/qwen35_35b_paged_test.go` (the comment above `TestQwen35_35B_pagedRuns`) on 2026-10-09.

```text
WHAT THIS MEASURES AND WHAT IT DOES NOT. Steady-state decode rate through the pager, plus the
staging decomposition (how much of a token is expert I/O). It is NOT a peer comparison: no
Ollama/llama.cpp arm runs here, and the CPU-paged 1.3-1.4 tok/s figure this lane has quoted was
measured in another session on another path, so no ratio against it is computed. See
~/goinfer-bench-logs/PREREGISTERED-qwen35-metal-paging.md for the rule this run was written
against, BEFORE it produced a number.
```

## TestQwen35_35B_cpuPagedBaseline

Moved from `metal/qwen35_35b_paged_test.go` (the comment above `TestQwen35_35B_cpuPagedBaseline`) on 2026-10-09.

```text
TestQwen35_35B_cpuPagedBaseline is the DO-NOTHING ARM for the run above: the same checkpoint,
same box, same session, decoded through the CPU expert pager (StreamWeights, decoder/moepaging.go)
instead of the Metal one. Without it, "Metal paging is faster" would be a cross-session ratio
against a number measured on another day — exactly the comparison this repo's own
docs/completed/task-zeno-compare.md shows can drift 2.6x from machine load alone (kind-3 gemma4 at one identical
config read 1.128 and then 2.917 tok/s).

IT LIVES IN metal/ AND USES NO METAL. That is deliberate: its only reason to exist is to be the
baseline for the Metal number, and a baseline that drifts away from its comparand into another
package (and another run) is how the 1.3-1.4 figure went stale in the first place. Same helpers,
same seed token, same greedy self-feeding trajectory, same token count.
```

## TestR1_laneVsCPU

Moved from `metal/r1_lane_vs_cpu_test.go` (the comment above `TestR1_laneVsCPU`) on 2026-10-09.

```text
TestR1_laneVsCPU is experiment X2 of the R1 (W4F16 decode lane) re-investigation. X1
(r1_gu_reference_test.go) showed that, within the resident int4 model, the f16 lane's gate/up
GEMV sits on the f64 reference and the shipped W4A8 lane is the coarse arm (its single
per-tensor int8 activation scale zeroes 97% of the FFN-26 input at the attention-sink
position). That leaves the one question that decides R1's fate: against an EXTERNAL
reference, is the f16 lane's full-model output better, equal, or worse than W4A8's?

Reference: the CPU backend, Options{Backend:"cpu", Quant:"int8"} — weight-only per-row int8,
f32 activations, exact f64-accumulating attention (GOINFER_CPU_FAST_ATTENTION=0, as
decoder/prefill_ref_gen_test.go forces it). The repo's own S reference is Quant:"" (f32 weights,
~6 GB) and does not fit beside anything on this 16 GB machine today (~5 GB free); "int8" is the
same choice prefill_ref_gen_test.go's d7RefQuant documents for the same reason. It is an
external reference: its weight requantisation differs from Metal's int4-g32, so BOTH Metal arms
carry the same weight-quant noise floor against it and the PAIRED comparison between arms is
what carries information. The CPU int4 path is deliberately NOT used (it quantizes activations).

Teacher-forced, no generation: the same N prompt tokens are fed one per step to the CPU
reference and to each Metal arm; per position we score cosine(logits), KL(softmax(cpu) ||
softmax(arm)) in float64, top-1 agreement and hard flips (decoder.NearTieArgmaxForTest).
Both Metal arms come from ONE model load: the lane is toggled at runtime via
r.decodeLaneW4F16 (canUseF16Lane reads the field on every call); Forward(id,pos) calls
setPos(pos), which rewrites curNKeys=pos+1 and the KV slot for pos, so restarting the arm at
pos 0 is clean (no other per-sequence state accumulates in forwardLogits).

Memory: ONE checkpoint resident at a time — the CPU model is Closed before the Metal load.

	GOINFER_HEAVY_TESTS=1 go test -tags "darwin goinfer_testhooks" ./metal/ -run 'TestR1_laneVsCPU$' -v
```

## TestR2_perLayerCtxDiff

Moved from `metal/r2_ctx_diff_test.go` (the comment above `TestR2_perLayerCtxDiff`) on 2026-10-09.

```text
TestR2_perLayerCtxDiff is the measurement R2's two prior investigations never took
(docs/measurements/r2-attn-fa-2026-09-19.md, r2-attn-fa-followup-2026-09-20.md): they compared
FINAL LOGITS of an attention_fa generation against a shipped-kernel generation, 28 layers and an
int8-activation-quantized pipeline downstream of the kernel under test, on Gaussian-noise
embeddings. The follow-up's own ULP control showed that pipeline is hypersensitive on that input
(a 1-ULP scale nudge flips logits by ~0.9 immediately), so "two clean steps, then a stable
plateau" is consistent with BOTH a real position-linked kernel defect AND a single int8 rounding
flip whose timing happened to line up. This test separates them by comparing the kernels'
OWN outputs (r.ctx) on IDENTICAL inputs, per layer, per decode step:

  - trajectory 1: the shipped kernel end to end via ForwardEmb (the record's reference arm);
  - trajectory 2: a manual per-layer harness with attention_fa OFF on both arms — must match
    trajectory 1 bit-for-bit, or the harness itself is wrong (checked, fatal if not);
  - trajectory 3: attention_fa's trajectory, where at every (step, layer) the shipped kernel is
    run from the same pre-layer residual and KV state first, its ctx and layer output captured,
    the residual restored, then attention_fa run — so |ctxFA - ctxShipped| is a pure kernel
    comparison on identical (q, K, V, nKeys), and |xFA - xShipped| shows where any downstream
    rounding flip lands. The trajectory then continues from attention_fa's own output, so its
    per-step logits reproduce the record's arm exactly.

The runtime toggle is r.decodeAttnFA (canUseAttnFA reads it on every dispatch; pipelines and the
partial buffer are always built) — the same technique R1's root-cause tests used for its lane.
One model load. Prefill length from GOINFER_R2_PREFILL (default 1600, the record's), steps from
GOINFER_R2_STEPS (default 5). If a layer's ctx diverges locally (maxRel > 1e-3, three orders
above f32 reduction noise), that layer's exact inputs and both outputs are dumped to
GOINFER_R2_DUMP_DIR for TestR2_replayDump below.

	GOINFER_HEAVY_TESTS=1 go test -tags "darwin goinfer_testhooks" ./metal/ -run 'TestR2_perLayerCtxDiff$' -v -timeout 20m
```

## runDecodeFidelityGate.identity

Moved from `metal/r2_gate_test.go` (the comment above `runDecodeFidelityGate.identity`) on 2026-10-09.

```text
Prompt identity: the reference files carry no prompt ids, so check the one row both arms share — the
prompt-final logits (reference row 0 vs the batched prefill's seed). A reference generated from different
text lands far above the W4A8 level here. Found 2026-09-25: set A's S-K3900 files (generated 2026-09-05 from
the live docs) predate the 2026-09-09 snapshot the prompts now come from, and 4 of 10 prompts changed inside
the 3900-token window (metal-decode-attn-r17-2026-09-25.md).
```

## TestZZ_residencyProbe

Moved from `metal/residency_probe_test.go` (the comment above `TestZZ_residencyProbe`) on 2026-10-09.

```text
TestZZ_residencyProbe isolates the per-submit RESIDENCY cost that dominates the paged-MoE decode
(~15 ms/boundary of GPU-idle-in-wait, 72× Step-0's 0.213 ms). Three arms separate per-buffer from
per-byte from re-validation, with trivial GPU work (1 thread) and many buffers REFERENCED (bound +
read) per command buffer. Load-bearing result (Arm A): a REPEATED identical set caches — submit[0]
~70 ms, submit[1..] ~0.4 ms — so the cost is the referenced set CHANGING per submit, and pinning
the working set resident once (MTLResidencySet / heap useHeap) collapses it. Arms B/C show the
uncached cost has both a per-buffer term (tracked-allocation count) and a per-byte term. Diagnostic,
not a gate; documents why the residency-set fix is the lever and would flag an OS residency change.
```

## TestS16MRoPEPrefill_real.sidecar

Moved from `metal/s16_mrope_prefill_test.go` (the comment above `TestS16MRoPEPrefill_real.sidecar`) on 2026-10-09.

```text
Through the directory's sidecar when it exists (S18: `<dir>.int4.metal.giw`, what serve's Metal load reads):
its int4 weights are mapped, not held twice, which is what the 2026-10-08 night's live budget (2.80 GB
against 3.77 GB for Qwen2.5-VL) lacked. It generates exactly what the direct load does
(prequant.TestDirSidecar_matchesDirectLoad).
```

## TestS18ShrinkCtxToFit

Moved from `metal/s18_ctx_shrink_test.go` (the comment above `TestS18ShrinkCtxToFit`) on 2026-10-09.

```text
The probe is a Metal load too: a CPU load prices more (it keeps arm64-repacked weights a Metal load does not), which put the budget above the
Metal build's 4096 need on the first try.
```

## TestS9EModelBatched_tiny

Moved from `metal/s9_emodel_batched_test.go` (the comment above `TestS9EModelBatched_tiny`) on 2026-10-09.

```text
TestS9EModelBatched_tiny is G-S9c of S9 step 2 (docs/tasks/task-multimodal-support-2026-10.md, registered before any
code): on gemma4-emodel-tiny (6 layers; head dims 32 and 64; 2 KV-shared layers with their own sources and double FFN
width; PLE P=32; layer scalars; v_norm), the f16 batched pass against the S9 layer-major pass (the validated path,
bit-identical to sequential) on one Metal resident. Compared: the last prompt row and 8 decode steps teacher-forced
along the layer-major pass's own greedy tokens; the bar is cosine >= 0.98 with every argmax equal or an R10 near-tie
(amended 2026-10-08 by the owner from the registered 0.9999, which the pass missed at 0.9907/0.9916 with no non-tie
argmax difference: the task doc's G-S9c record has the mechanism; 0.98 sits between that and the nearest defect, 0.94).
Two prompts: the golden 12-token prompt and a synthetic 96-token one, which runs every local layer past its window of 4
and both shared layers over long sources. The batched pass is called directly, so a silent fallback cannot stand in for
it. Each planted defect (emodelBatchDefect) must go red on at least one prompt.
```

## s3Sharpen

Moved from `metal/s3_towers_test.go` (the comment above `s3Sharpen`) on 2026-10-09.

```text
The tiny towers are degenerate along several of the registered defects' axes, and G-S3a's first tiny run showed it:
their biases are all zero (a dropped patch bias cannot show) and Qwen2.5-VL's q/k weights are at init scale (0.02), so
attention is nearly uniform and the scale, RoPE and windowing barely matter (each defect read 0.99999 or better). So,
before comparing, the check sharpens the fixture through the exports' aliasing slices, which the CPU tower reads too:
random biases (s3Sharpen) and q/k projections scaled up (s3Scale), so the attention logits are not near zero; for
Qwen2.5-VL also v and the output projection, whose init scale left attention a negligible share of the residual stream
(defect (3) still read 0.99992 with q/k alone: the tiny tower has one windowed block, then a full one).
```

## TestSAGemv_OutlierRegime

Moved from `metal/sa_outlier_test.go` (the comment above `TestSAGemv_OutlierRegime`) on 2026-10-09.

```text
TestSAGemv_OutlierRegime is the decisive experiment for the Gemma o-proj amplitude bug
(docs/prompts/gemma-metal-signflip-bisect.md, Fork 2). The CUDA box showed Metal's o-proj
contribution inflates 2–6× and flips on the secondary channels while CUDA's dp4a stays clean,
and named it a Metal W4A8 GEMV scale bug. TestSAGemvLargeK already proves the kernel correct at
K=4096 — but with a BENIGN random activation. This drives the same production kernel in the
regime that actually triggers the bug: at Gemma's o-proj K=2048, with an activation carrying
ONE massive outlier channel that sets the int8 scale (~558) and crushes every other channel to
near-zero int8 — exactly the post-quant state of the attention context feeding the o-proj.

The CPU reference computes the kernel's OWN formula bit-for-bit (dequant nibble × int8 act ×
f16 group scale × asc). So this isolates the KERNEL from the quant scheme:
  - Metal == CPU here ⇒ the o-proj GEMV arithmetic is FAITHFUL even in the outlier regime, and
    Metal's divergence-vs-CUDA lives upstream (a different attention context) or in the quant
    policy, NOT this kernel — which would redirect the fix.
  - Metal ≠ CPU here ⇒ the kernel mis-handles the outlier regime (group-scale accumulation or
    asc application under extreme dynamic range) — the bug, localized.

Reported per-channel on the crushed rows, because a whole-vector cosine is dominated by the one
massive output and would hide a 6× error on the secondary rows (the sink lesson).
```

## TestSAQVFusion_correctnessAndThroughput

Moved from `metal/sa_qv_fusion_test.go` (the comment above `TestSAQVFusion_correctnessAndThroughput`) on 2026-10-09.

```text
TestSAQVFusion_correctnessAndThroughput is the measurement item #5 of the 9-finding audit
needs before it goes near a production dispatch site: does fusing quant_vec into
gemv_w4a8_sa (gemv_w4a8_sa_qv) actually win, given every threadgroup the fused kernel
launches redoes quant_vec's O(K) amax reduction independently (no cheap way for one
threadgroup to hand a computed scale to another within one Metal dispatch)? Unlike item #4
(the rope2 merge, a genuine reduction in total work), this trades one dispatch launch +
one K-element device-memory round-trip against (N/8 - 1) redundant K-element reductions.

VERDICT (measured, not inspected): roughly NEUTRAL, leaning slightly negative — NOT the win
the audit's dispatch-count estimate implied. Four interleaved runs (see the interleaving
comment below for why non-interleaved gave a false 1.28x win) at real dims: 0.974x, 0.949x,
0.992x, 0.972x speedup — a tight cluster around ~0.97x, i.e. the fused kernel is a few
percent SLOWER, not faster. The redundant per-threadgroup reduction cost roughly cancels the
removed-dispatch savings at this K/N. Kept in the tree as a correctness-proven (bit-identical
to the two-dispatch path) but NOT-production-worthwhile experiment — do not wire this into
model.go on the strength of the dispatch-count argument alone; the wall-clock number doesn't
back it up here.
```

## TestSAQVFusion_correctnessAndThroughput.reps

Moved from `metal/sa_qv_fusion_test.go` (the comment above `TestSAQVFusion_correctnessAndThroughput.reps`) on 2026-10-09.

```text
reps=20: NOT the real per-token count (a real decode token issues far fewer than this per
layer) — chosen to stay clear of a real, separate, pre-existing issue: repeatedly calling
Encoder.Dispatch on the SAME reused buffers hundreds of times in one encoder (via the
general Dispatch path, which rebinds every buffer each call, unlike Run1DBatch's
bind-once-dispatch-many) hits a probabilistic crash unrelated to gemv_w4a8_sa_qv's
correctness (reproduces with the plain two-dispatch pattern alone, no fused kernel
involved, and isn't a hard threshold — it can still fire occasionally even at reps=20).
Worth its own investigation separately; out of scope here.
```

## TestSAQVFusion_correctnessAndThroughput.interleave

Moved from `metal/sa_qv_fusion_test.go` (the comment above `TestSAQVFusion_correctnessAndThroughput.interleave`) on 2026-10-09.

```text
INTERLEAVED, not two separate blocks: measuring "all of A then all of B" confounds the
comparison with whatever changes between the two blocks (thermal ramp, GPU contention
drift) — measured directly here: a first pass with A-then-B block order showed fused
WINNING 1.28x; two immediate re-runs of the same block order showed fused LOSING ~0.47x,
consistently with each other but not with the first run. That is a confound, not a real
effect, and it wouldn't have been visible without deliberately re-running. Alternating A/B
every sample makes drift affect both roughly equally instead of favoring whichever block
happens to run when conditions are better.
```

## TestSink_NormsNotCosines

Moved from `metal/sink_test.go` (the comment above `TestSink_NormsNotCosines`) on 2026-10-09.

```text
Fable's hypothesis: the probe token is <bos> (id 2), Gemma's ATTENTION SINK. Sink V vectors
are trained near-zero (sink K is a strong direction; sink V is a no-op). A cosine between two
near-zero vectors is rounding noise — which would make "layer-1 V cos = -0.047 ⇒ x after
layer 0 is orthogonal" a measurement artifact, not a bug. Every number in the debug report was
a cosine; nobody measured a NORM. So measure norms — and probe a NON-sink token too.
```

## TestZZ_metalSoftcapTokenShare

Moved from `metal/softcap_ab_test.go` (the comment above `TestZZ_metalSoftcapTokenShare`) on 2026-10-09.

```text
TestZZ_metalSoftcapTokenShare loads a real dense Gemma (final-logit softcap, 262k vocab) and times
the full-logits sampling path (ForwardEmb → forwardLogits → finalizeLogits) per token, so the
isolated softcap A/B (BenchmarkSoftcap_gemmaVocab_*: serial ~3.4ms, parallel ~0.86ms) can be
expressed as a share of the token. Opt-in timing diagnostic, not a gate.
```

## TestSpecNgram_copyOnStepVerify

Moved from `metal/spec_multiturn_test.go` (the comment above `TestSpecNgram_copyOnStepVerify`) on 2026-10-09.

```text
Both compared arms reuse the cache's prompt prefix, so both re-decode the prompt's last position. A cold arm would
have prefilled it, and the f16 prefill is not bit-identical to decode: comparing spec against the cold arm is a
cold-against-warm comparison, which on 2026-10-03 diverged at token 105 with attnFADepthFloor at 1024 and was taken
for a spec defect (B-P03's revert, since undone). The cold run is a warm-up only.
```

## TestMetalSnapshotGolden

Moved from `metal/snapshot_golden_test.go` (the comment above `TestMetalSnapshotGolden`) on 2026-10-09.

```text
TestMetalSnapshotGolden is the ABSOLUTE STORED REFERENCE the Metal gate suite otherwise lacks.

Every other Metal gate is self-consistent or tolerance-based: `paged ≡ non-paged` compares one
kernel against itself under different residency (any change to the kernel moves BOTH arms
identically → passes), and `Metal-vs-CPU` is cosine/tolerance (small movements pass by
construction, unavoidable given the f16 scale gap). That whole class — a reduction-WIDTH change
(float sum is non-associative, wired to threadgroup width; see tgReduce* in model.go), a
fused-kernel rewrite, a different accumulation order, a moved scale-application point — is
invisible to those gates. Only a reference that does NOT move when the code moves catches it.

This decodes a FIXED token sequence through the Metal resident path on tiny committed models, to
depths PAST the reduction widths (128 and 256), and byte-compares the logits (sha256) to a committed
golden. It is self-referential: it detects that something moved, not which side is correct — exactly
what's missing. It is MACHINE-PINNED (Metal float results are deterministic run-to-run and across
code versions on a given GPU, but not guaranteed identical across chip families). It WILL go red on
a legitimate improvement — that's the point; regenerate with the refresh flag after verifying the
change is intentional (the same goldens-refresh discipline the CUDA track uses):

	GOINFER_UPDATE_GOLDENS=1 go test -run TestMetalSnapshotGolden ./metal/

N-29: the re-bake this note demanded HAS happened — TestMetalSnapshotGolden reports
6 checkpoints byte-identical on this machine. A standing "EXPECTED TO FAIL" note turns
the suite's only absolute gate into noise: a red here is a REAL drift and must be read
as one. The G-02 history is kept below for provenance, in the past tense.

RE-BAKE DONE (audit G-02, fixed on Linux where this suite cannot run). The checkpoint call now
drives ForwardEmb with the production-scaled embedding row instead of Forward with a raw one, and
Forward/ForwardArgmax now apply the arch embed scale. For `gemma4-dense-scaled` (EmbedScale =
√hidden) that CHANGES the hashed stream — the stored entries pin the pre-fix, non-production
computation, so that CHANGED the hashed stream, and the golden was re-baked accordingly:

	GOINFER_UPDATE_GOLDENS=1 go test -run TestMetalSnapshotGolden ./metal/

`mixtral-tiny` has no embed scale and its entries must NOT move; if they do, something other than
G-02 changed and the re-bake should be refused pending investigation.

Regenerate too on a hardware change (different Mac). Runs on every `go test` for the two
COMMITTED models (mixtral-tiny, llama-attnfa-tiny — no heavy-model dependency, always
available). gemma4-dense-scaled (449 MB) is NOT committed — over GitHub's practical push
limit, unlike the tiny ones — so it's a local-only fixture regenerated deterministically via
`scripts/pin_gemma4_dense_scaled.py`; when it's absent this test skips it and still fully
checks the other two (keyed comparison, not positional — see below), rather than failing
opaquely or silently losing coverage for the always-available fixtures too (fixed 2026-09-21;
every OTHER consumer of that fixture already skipped gracefully, this test was the outlier).
Coverage: mixtral-tiny is full-causal (attention softmax denom over
>256 keys → the width coupling at multi-iteration depth) + rmsnorm_quant; gemma4-dense-scaled covers
rmsnorm_f32 + qk_norm. llama-attnfa-tiny (added 2026-09-21, R2 golden-coverage follow-up) covers
`attention_fa`, DEFAULT ON past depth 1024 (metal/model.go's attnFADepthFloor, 1536 until B-P03) — the other two
fixtures both fail canUseAttnFA's head_dim==128 guard (8 and 256 respectively) and can never
dispatch it regardless of depth, so this kernel had zero coverage from this suite until this
fixture existed. Its checkpoints straddle the floor exactly (1022 declines, 1023 engages) so an
off-by-one at the boundary is caught, not just steady-state behavior on either side. Union = every
pinned-width reduction kernel plus `attention_fa` this build DISPATCHES.

N-28: this used to claim attention_f32 as well. It is not covered and cannot be — model.go
hard-wires `r.kvF32 = false` and builds kv_store_f32/attention_f32 only inside `if r.kvF32`,
so both are dead code no path reaches. A coverage claim naming a kernel nothing dispatches is
the "a doc comment claiming coverage is not coverage" class: the reader matches the claim to a
plausible kernel name and stops. See §A2-Metal.
prodEmbedRow fills dst with token id's LAYER-0 INPUT exactly as production builds it:
decoder.embedResident dequantizes the embedding row and multiplies by the arch's embed scale
(Gemma's √hidden). Mirroring it here is what makes the golden a reference for the SHIPPED
computation rather than for an entry point production never calls (audit G-02).
```

## TestMetalSnapshotGolden.attnfa

Moved from `metal/snapshot_golden_test.go` (the comment above `TestMetalSnapshotGolden.attnfa`) on 2026-10-09.

```text
R2: attention_fa, DEFAULT ON past attnFADepthFloor=1024 (metal/model.go; 1536 until B-P03). Every other
fixture here has head_dim != 128 (mixtral-tiny: 8, gemma4-dense-scaled: 256), so
canUseAttnFA's hd==128 guard declines on both, regardless of depth — neither can ever
cover this kernel. llama-attnfa-tiny (scripts/pin_llama_attnfa_tiny.py) is a plain dense
GQA Llama shaped to clear every other guard too (no sandwich/postOnly/parallelBlock/
attnSink/kvI8/lora/MoE/DeltaNet/qGate/window — see canUseAttnFA). Checkpoints straddle
the floor exactly: curNKeys = pos+1, so pos=1022 (curNKeys=1023) is the last declining
position and pos=1023 (curNKeys=1024) is the first engaging one. 900 is a shipped-kernel-only
control below the floor; 1100 confirms the engaged kernel stays stable past the boundary, not
just at it. Re-baked for B-P03's floor move (G-04; it was 1400 / 1534 / 1535 / 1600 at 1536).
```

## TestMetalSnapshotGolden.skip

Moved from `metal/snapshot_golden_test.go` (the comment above `TestMetalSnapshotGolden.skip`) on 2026-10-09.

```text
gemma4-dense-scaled (449 MB) is NOT committed — GitHub's practical push limit is well
under that, so it can't be, unlike mixtral-tiny/llama-attnfa-tiny — and every OTHER
consumer of this fixture (metal/gemma4_dense_scaled_test.go, the cuda/gpu prefill tests,
decoder/gemma4_moe_forward_test.go) already skips gracefully when it's absent, pointing
at scripts/pin_gemma4_dense_scaled.py (deterministic, seed 0) to regenerate it. This test
was the one outlier that didn't, so a fresh machine got an opaque t.Fatalf instead of a
clean skip. Skipping one model here must not cost the OTHER (committed, always-available)
fixtures' coverage — see the schema-tolerant comparison below, keyed by (Model,Quant,Depth)
rather than positional/count equality, so mixtral-tiny and llama-attnfa-tiny still get
fully checked on any machine even when this one is locally absent.
```

## TestMetalSpecPrefillRegression

Moved from `metal/spec_prefill_regression_test.go` (the comment above `TestMetalSpecPrefillRegression`) on 2026-10-09.

```text
TestMetalSpecPrefillRegression is the Metal half of the 2026-08-31 speculative-prefill
regression that was measured on CUDA and asserted on Metal by interface only.

THE DEFECT: decoder/model.go's generateInto ingests a prompt through the batched Prefiller
seam (m.resident.(Prefiller).PrefillLast); decoder/spec_ngram.go's genNgramInto instead
loops target.resident.Forward(...) one token at a time. Speculative decode therefore pays a
per-prompt-token cost that plain generation does not, and the gap grows LINEARLY in prompt
length — on CUDA, 2.66 ms/prompt-token, R² 0.9977, which at 839 tokens made the speculative
path 4.1x SLOWER than not speculating at all.

THE GATE IS THE SLOPE, NOT THE RATIO. The spec-vs-off ratio moves with draft acceptance,
which moves with the corpus, so it flaps; the slope of (spec - off) against prompt length is
the defect's signature and is nearly acceptance-independent. Bar: 0.50 ms/prompt-token
(CUDA measured 2.66 with the bug, 0.12 without).

WHY THIS ASSERTS RATHER THAN LOGS: gpu/spec_ngram_resident_test.go measures the right
quantity and only t.Logf's it — its own header says "speedup is logged per workload" — so a
0.3x printed and failed nothing for six weeks. The slope check below is t.Fatalf.

WHY THE CORPUS IS READ FROM DISK: specWorkloads/ngramWorkloads are hand-written and
deliberately copy-heavy (4-7x the copy density of real code) and only 36-74 tokens long,
which is precisely why a regression that needs LENGTH to show went unseen. This reads real
repository source at run time instead.

METAL-SPECIFIC PRECONDITION — GOINFER_METAL_BATCHED_PREFILL=1 IS MANDATORY HERE, and a green
without it is meaningless. Historically (when this test was written) Metal's PrefillLast
(metal/backend.go) declined by default — Metal's batched prefill was not bit-identical to its
decode path (54% stream divergence, a figure once measured by TestMetalPrefillDivergenceRate,
docs/ollama-chase.md; that test no longer exists — superseded by TestPrefillGateVsReference's
pooled §3.2 criteria, G-07 audit-metal-2026-09-12.md). Batched prefill is now default-ON above
metalFastPrefillFloor (M-06/M-02, same audit), so this override is no longer strictly load-
bearing for a prompt past the floor — kept anyway so this test's precondition never depends on
the floor's current value or default state. Below the floor (or with the override removed),
PrefillLast still declines and generateInto falls through to the same per-token loop the
speculative path already uses, so there would be NO asymmetry to measure and the slope would
come back ~0 for a reason that has nothing to do with the fix. Metal now ALSO implements
ResidentPrefillKV (M-01, same audit: ForwardNoLogits skips the LM head on every prefill token
but the last) — but that lives inside residentPrefillSeed, which genNgramInto already shares
with generateInto (see that call site's own comment), so both arms of THIS test's comparison
benefit from it identically. No new asymmetry: Metal still has exactly one exposure to the
original bug (the batched-vs-per-token prefill seam this test targets), gated behind this
variable.
```

## TestSpecVerifyCurveMetal

Moved from `metal/spec_verify_curve_test.go` (the comment above `TestSpecVerifyCurveMetal`) on 2026-10-09.

```text
CRITICAL, and the reason this needs stating before any number below: `PrefillLast` — the batched
primitive this test times — is NOT bit-identical to decode's int8 path (its f16-MMA activation
path once measured 54% stream divergence, §A2-Metal, docs/ollama-chase.md — historical
record; the test that produced that figure, TestMetalPrefillDivergenceRate, no longer exists,
superseded by TestPrefillGateVsReference's pooled §3.2 criteria, G-07 audit-metal-2026-09-12.md).
PrefillLast is now default-ON above metalFastPrefillFloor for admitted architectures (it declined
by default only historically, when this comment was written); GOINFER_METAL_BATCHED_PREFILL=1
here forces it on regardless of the floor, for exactly the "measurement/TTFT-at-the-cost-of-
exactness" use this test is. So every number this test produces characterizes a kernel that is
NOT currently usable as P10's verify oracle on Metal — P10's own design requires the verify step
to reproduce sequential greedy exactly (00-core's lossless contract). A real Metal P10 leg needs
that bit-identity gap closed FIRST; this test answers "is the timing shape even worth it", not
"is this safe to ship".
```

## TestSwigluQuant_MassiveChannel

Moved from `metal/swiglu_massive_test.go` (the comment above `TestSwigluQuant_MassiveChannel`) on 2026-10-09.

```text
TestSwigluQuant_MassiveChannel isolates the BOS geglu bug (b04d799) to KERNEL vs DATA. The L0
BOS trace found swiglu_quant emitting ~0 for the dominant channel (gate=12.14, up=-18.49, so
geglu should be -224.4). This drives the SHIPPED swiglu_quant (allKernels) with exactly that
value at one channel and small values elsewhere. If the dequantized output there is ~-224 the
kernel is fine and the r.gu data was corrupted upstream; if it's ~0 the kernel itself drops the
massive geglu channel.
```

## skipIfMemoryDeclined

Moved from `metal/testshared_test.go` (the comment above `skipIfMemoryDeclined`) on 2026-10-09.

```text
skipIfMemoryDeclined skips a resident parity test whose resident build the memory guard declined. That is "this machine
does not have the free memory right now", and no forward ran, so it is not a parity result. It used to fail as
"admission says it should be admitted", which `go run ./cmd/gate gpu` reports as "a Metal forward moved":
TestMellumResidentParity did exactly that on 2026-09-30 (needs 4.30 GB; the guard's budget is 70% of live free memory,
about 3.4 GB with the owner's apps open). A skip is still named under the gate's "does NOT cover" list, so the gap
stays visible. Any other decline (a feature or admission mismatch) still fails at the caller.
```

## TestMaxThreadgroupStageBytes

Moved from `metal/tgbudget_test.go` (the comment above `TestMaxThreadgroupStageBytes`) on 2026-10-09.

```text
N-32: DeltaNet's out-proj staging (deltanet.go dispatches dp.valueDim*2) was missing
from this budget, so a model whose value dim exceeds every other staged width passed
the M-11 check and then exceeded the device limit at dispatch. Only these two rows
exercise the new term — without them the fix is ungated, which is where this test was
after the first attempt at adding them silently no-opped.
```

## TestMetalThetaAB

Moved from `metal/theta_ab_test.go` (the comment above `TestMetalThetaAB`) on 2026-10-09.

```text
TestMetalThetaAB — does wiring the MEASURED Theta actually make Metal faster?

Theta was reachable only as 0.5 on every backend, because AdaptiveDepth's
domain was [0,1) and Metal measures 1.006-1.048. Under 0.5 the controller
drafts; under the measured value it declines to draft, because a Metal verify
node costs a full target step (ForwardN is a loop of single-token Forwards, so
T(n) = n*T(1) — measured linear to n=16).

That predicts the speculative path under Theta=0.5 is SLOWER than not
speculating at all on Metal, and that the wired default recovers it by
declining. This measures that rather than asserting it. Three arms, one
prompt, interleaved:

	off        plain Generate, no speculation — the do-nothing arm, which is
	           the whole point: "beats every configuration" means nothing if
	           off wins, and here off is EXPECTED to win against Theta=0.5
	theta=0.5  the shipped-until-now behaviour, forced explicitly
	wired      Theta unset, so verifyTheta() supplies the measured 1.02

The assertion is deliberately weak in one direction and strong in the other:
`wired` must not be materially slower than `off` (it should be within noise of
it, since it declines to draft), and it must beat `theta=0.5`. Nothing here
claims speculation is bad in general — it claims this backend's verify is not
batched, which is exactly what item (2) would change.
```

## TestThetaProbe_Metal

Moved from `metal/theta_probe_test.go` (the comment above `TestThetaProbe_Metal`) on 2026-10-09.

```text
TestThetaProbe_Metal measures Theta — the marginal cost of one extra verify node, in units
of one single-token target step — on the Metal resident path. Method is identical to the
CUDA probe (cuda/theta_probe_test.go) and the CPU control (decoder/theta_probe_test.go) so
the three numbers are directly comparable: seed `depth` positions, then time ForwardN over a
ladder of widths, and take Theta = (least-squares slope of T(n)) / T(1).

decoder/spec_adaptive.go ships Theta = 0.5, calls it the batched-CPU value, and says
"measure it". CPU measured 0.456, CUDA 0.155-0.251. Metal was unmeasured.

WHAT TO EXPECT HERE, AND WHY IT IS NOT THE CUDA STORY. CUDA's low Theta comes from a verify
that streams the weights ONCE for the whole block, so the marginal node is far cheaper than a
step. Metal's ForwardN (metal/backend.go:metalResident.ForwardN) is NOT a batched kernel — it is a plain loop of
single-token Forward calls. If that is the whole story then T(n) = n*T(1) and Theta ~ 1.0,
which would mean the controller is running Metal on a constant that is too LOW and therefore
OVER-drafting — the opposite direction from CUDA, where 0.5 is too high and under-drafts.
That is a prediction from reading the dispatch, and the point of this test is to measure it
rather than assert it.

TruncateTo is a NO-OP on Metal (metal/backend.go:metalResident.TruncateTo), unlike CUDA where the probe leans on
it to hold context depth constant between timed calls. It is safe here for the reason its own
comment gives — KV positions are overwritten on write and attention reads only keys[0..pos],
so re-running ForwardN at the same startPos re-attends over the same span. It is called
anyway, so the two probes stay line-for-line comparable.
```

## TestTowerGemmKernel

Moved from `metal/tower_attn_test.go` (the comment above `TestTowerGemmKernel`) on 2026-10-09.

```text
TestTowerGemmKernel is S17 lever B's kernel gate on Metal (docs/tasks/task-multimodal-support-2026-10.md, "S17's
Metal lever B", registered before the code): tower_gemm_w32 and tower_gemm_w16 against a float64 A·Wᵀ + bias from the
f32 inputs, at the towers' projection shapes (rows trimmed so the host reference stays fast) and at edge shapes (M, N
and K off every tile multiple). Bars: every output within 1e-5 (w32: f32 accumulation order only) or 1e-3 (w16: the
weights rounded to f16) of its own Σ|a·w| + |bias|. (The first f16-activation kernel's bar was 2e-3; it passed here
and failed the tower bars, see the doc.) Weights at a tower's scale (σ 0.02), so a dropped bias is far over either bar. Each
planted defect must miss it, on both kernels.

tower_gemm_w8 is G-S18e (S18 on the Mac, registered before the code): int8 weights in groups of 32 with an f32 scale
each, against a float64 reference over the same dequantized weights (float(q)·scale), so the bar is 1e-5 (only the
accumulation differs); its fourth planted defect takes each group's scale from the next group.
```

## TestVLImageTurn_metalResident

Moved from `metal/vl_image_turn_test.go` (the comment above `TestVLImageTurn_metalResident`) on 2026-10-09.

```text
TestVLImageTurn_metalResident is S3's tiny-fixture check (2026-10-07, first run: Gemma 3 8/8 tokens identical,
Qwen2.5-VL 6/8) (docs/tasks/task-multimodal-support-2026-10.md): a Gemma 3 and a
Qwen2.5-VL image turn decode on a Metal resident, from the committed tiny checkpoints and their goldens' image
features, against the CPU decoder (both int4). It is the first image turn on a Mac's GPU for either family; the
registered S3 gate is the served turn on the real checkpoints.
```

## TestW8Native_F3_closerToF32

Moved from `metal/w8_native_gate_test.go` (the comment above `TestW8Native_F3_closerToF32`) on 2026-10-09.

```text
TestW8Native_F3_closerToF32 is gate F3: on one token sequence (the prompt, then the CPU f32 model's greedy
continuation, 24 positions), the mean per-position KL(f32 ‖ Metal int8int8) is at most 1.10 × KL(f32 ‖ CPU
int8int8). KL(f32 ‖ Metal int4) is reported beside it. Positions 0 and 1 are left out, as residentParity's cosine
leaves them out. The models load one at a time at a 1024-position context, so a 16 GB Mac holds the f32 reference
only while it runs; by day the fit guard still refused it there (needs ~2.8 GB against a 2.2 GB budget), so F3
runs on the night queue.
```

## TestGenerate_warmRepeatMatchesCold

Moved from `metal/warm_cold_identity_test.go` (the comment above `TestGenerate_warmRepeatMatchesCold`) on 2026-10-09.

```text
TestGenerate_warmRepeatMatchesCold: the same prompt generated twice on one model emits the same tokens, the second time
reusing every cached position but the last. Before decoder.PrefillTailExact, a prompt the cold run had prefilled with
the batched pass ran its last position through decode on the repeat and diverged some tens of tokens later (on the
2-slot fixture, token 42 of 160); TestMC3Chain_newcomerJoinsAndBothMatchAlone met it as an "MC3 identity defect"
because its alone runs were cold and its joint runs warm. Single- and 2-slot models, at 12 tokens (below the 16-token
floor: the exact lane both ways), 40 (the pass on one slot, the step on two) and 100 (the pass), greedy.
```
