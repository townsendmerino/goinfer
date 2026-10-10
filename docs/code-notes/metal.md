# metal: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `metal`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

## batchMaxSeqs

Moved from `metal/batch.go` (the comment opening the file, above `batchMaxSeqs`) on 2026-10-10.

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

Moved from `metal/batch.go` (the comment above `batchMinSeqs`) on 2026-10-10.

```text
batchMinSeqs is the smallest batch the decoder should send here. At B = 1 the step costs 1.6–1.95x a production
token (the fragment's cost does not shrink with the batch); B = 2 measured 0.98–1.10x (S1), about even, and it
spares the per-token KV-slot rebinding two interleaved single-token decodes would pay.
```

## mc3FARowsOn

Moved from `metal/batch.go` (the comment above `mc3FARowsOn`) on 2026-10-10.

```text
mc3FARowsOn runs the batched step's rows at attention_fa depth through one multi-row dispatch pair (E-P05,
mc3_attention_fa*_rows), bit-identical to their per-row pairs. ON since 2026-10-04 (owner: "turn E-P05 on"): faster in
every cell measured, 1.016x at 2 rows (serve's default slot count) to 1.06x at 8 rows on the 1.5B at depth 1100-2048,
1.029x on the 7B (docs/tasks/task-metal-audit-2026-10.md, "E-P05").
```

## mc3AdjRowsOn

Moved from `metal/batch.go` (the comment above `mc3AdjRowsOn`) on 2026-10-10.

```text
mc3AdjRowsOn runs the batched step's per-row qkv and gate|up GEMVs (the sizes calibrateRows gives to per-row) as one
dispatch with each weight tile's B threadgroups adjacent (E-P02, mc3_gemv_w4a8_sa_{bias_}rows_adj), bit-identical to
the B per-row dispatches. ON since 2026-10-04 (owner: "turn ... E-P02 on"); its kernel A/B read 1.11-1.15x on the 7B's
gate|up at B = 2-4 and 1.01-1.04x on the 1.5B's.
```

## calibrateRows

Moved from `metal/batch.go` (the comment above `calibrateRows`) on 2026-10-10.

```text
calibrateRows decides, once, at which batch sizes qkv and gate|up are cheaper as B per-row production GEMVs than as
one fragment dispatch (MC3 S4, docs/tasks/task-concurrency-2026-09.md). The fragment's cost is fixed for any B up to
8, and a GEMV's is per row. On the 7B's large shapes the GEMV is bandwidth-bound and the fragment ALU-bound, so two
GEMVs beat one fragment (gate|up 0.87 against 1.17 ms). Both run on the real weights of the first layers, one layer's
weights per dispatch so the cache does not serve them, median of 7 command buffers after 2 warm-ups. The choice
changes speed only: the per-row path is production's own kernel on each row.
```

## stepVerifyCost

Moved from `metal/batch.go` (the comment above `stepVerifyCost`) on 2026-10-10.

```text
stepVerifyCost is the step-kernel verify's measured cost in single-token steps, indexed by rows verified: the
conservative end of TestMC3Verify_rowCost on the M1 Pro (the 7B at depth 2048, the dearest cell of both models; the
1.5B at depth 128 reads 1.59 / 1.71 / 1.76 at 2 / 4 / 8 rows). 1 row is production's own Forward; 3 and 5-7 are
interpolated; 9 rows are an 8-row step plus one Forward. docs/tasks/task-concurrency-2026-09.md, MC4.
```

## promptStepMaxAboveFloor

Moved from `metal/batch.go` (the comment above `promptStepMaxAboveFloor`) on 2026-10-10.

```text
promptStepMaxAboveFloor is the longest suffix the above-floor half of the step route takes (promptStepAboveFloor).
T1.10 measured the step at 0.40× / 0.80× / 1.56× the pass at K = 16 / 32 / 64 on the 1.5B and the 7B
(docs/tasks/task-metal-audit-2026-10.md, Batch A), so it stops at the last K measured faster.
```

## promptStepOK

Moved from `metal/batch.go` (the comment above `promptStepOK`) on 2026-10-10.

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

Moved from `metal/batch.go` (the comment above `prefillByStep`) on 2026-10-10.

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

Moved from `metal/expertpool.go` (the comment above `expertSlot` (it described the pool and the slot view together)) on 2026-10-10.

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

Moved from `metal/expertpool.go` (the comment above `copyBytesToU32Buf`) on 2026-10-10.

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

Moved from `metal/expertpool.go` (the comment above `distinctExperts`) on 2026-10-10.

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

Moved from `metal/expertpool.go` (the comment above `stagePread`) on 2026-10-10.

```text
stagePread, when set (GOINFER_MOE_PREAD=1 on a .giw-mmap'd model), REPLACES the mmap byte-copy:
it preads expert e's nibbles straight into slot s's unified-memory buffers — one syscall, one
large sequential read, zero page faults (cold pread measured 3687 MB/s vs the mmap demand-fault's
375 MB/s, 9.8×). Fetch and copy collapse into the single read. nil ⇒ the mmap byte-copy path.
```

## ensureResidentBatch

Moved from `metal/expertpool.go` (the comment above `ensureResidentBatch`) on 2026-10-10.

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

Moved from `metal/batch_rows.go` (the comment opening the file, above `mc3RowsKernels`) on 2026-10-10.

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

Moved from `metal/deltanet.go` (the comment above `buildDeltaNetLayer`) on 2026-10-10.

```text
Name the missing tensor — see cuda/backend.go's identical check and comment: an empty slice
here becomes a 0-byte device upload, which fails as "invalid length" with no indication of
which of the four small tensors was the culprit. Two different checkpoints have already
failed exactly that way during the CUDA bring-up.
```

## deltaNetKernels

Moved from `metal/deltanet_kernels.go` (the comment above `deltaNetKernels`) on 2026-10-10.

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

Moved from `metal/device.go` (the comment above `package metal`) on 2026-10-10.

```text
Metal device layer — now aikit's native-GPU substrate (github.com/townsendmerino/aikit/gpu),
lifted verbatim from what used to be this package's metal.go. goinfer keeps its tuned kernels
here and builds them on these device types — the GPU analogue of the linalg relationship. Only
the device TYPES moved; nothing about the decode path changed, so it must stay bit-identical
(the Metal device-parity suite is the tripwire).
```

## NewBufferFloats

Moved from `metal/device.go` (the comment above `NewBufferFloats`) on 2026-10-10.

```text
Thin re-wraps of aikit gpu v0.29.0's type-suffixed-Buffer-API collapse (NewBufferFloats/
NewBufferInt8/NewBufferU32/NewBufferUint32s/NewBufferU16s deleted in favor of the generic
NewBufferOf[T]). Go has no generic methods, so the aikit replacement is a free function
(gpu.NewBufferOf(d, data)); these keep every one of this package's ~500 existing call sites at
their original method-call shape (now a free function taking d first) instead of touching each
one's argument list.
```

## attnGeom

Moved from `metal/geom.go` (the comment above `attnGeom`) on 2026-10-10.

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

Moved from `metal/greedy_chain.go` (the comment above `chainDrainEvery`) on 2026-10-10.

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

Moved from `metal/paged_fence.go` (the comment above `pagedFenceOn`) on 2026-10-10.

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

Moved from `metal/paged_fence.go` (the comment above `pagedFenceSrc`) on 2026-10-10.

```text
pagedFenceSrc follows aikit's C-B03 probe (gpu/metal_fence_probe_test.go, after MLX's kernels/fence.metal): the system
thread scope is not public MSL, so it is built from the compiler's __METAL_MEMORY_SCOPE_SYSTEM__. The router wrote its
ids with plain stores in an earlier dispatch of the same serial encoder; a system-scope fence does not publish
non-coherent stores (the probe saw such a payload stale on 47 of 50 boundaries), so this kernel copies the ids through
coherent(system) stores into a mirror the host reads, fences, then stores the sequence number the host spins on.
```

## dnetPrefillOn

Moved from `metal/prefill_deltanet.go` (the comment above `dnetPrefillOn`) on 2026-10-10.

```text
dnetPrefillOn admits Gated-DeltaNet models to the batched prefill pass. ON since D-B01's grade (2026-10-04,
docs/tasks/task-metal-audit-2026-10.md: fidelity SHIPS on Qwen3.5-0.8B, the pass 8.06x the sequential loop at
K = 512); tests turn it off for the sequential arm.
```

## emodelLayerMajorOn

Moved from `metal/prefill_emodel.go` (the comment above `emodelLayerMajorOn`) on 2026-10-10.

```text
emodelLayerMajorOn routes a Gemma 4 E-model's prompt (E2B, E4B: per-layer embeddings, KV-shared layers) through
prefillEModel instead of declining PrefillLast (S9 of docs/tasks/task-multimodal-support-2026-10.md). The f16-MMA pass
declines every dense Gemma 4 (head size varies by layer), so without this a text prompt prefilled one token at a
time on the GPU and an image turn prefilled on the CPU. The pass is bit-identical to the sequential loop by
construction (G-S9a); on by default, as the paged 26B's layer-major pass is, with a night speed grade to turn it off
below 1.00x.
```

## emodelBatchedOn

Moved from `metal/prefill_emodel.go` (the comment above `emodelBatchedOn`) on 2026-10-10.

```text
emodelBatchedOn routes a Gemma 4 E-model's prompt through the f16 batched pass (S9 step 2,
docs/tasks/task-multimodal-support-2026-10.md) instead of the layer-major one, at or above the fast-prefill floor. ON
since 2026-10-09: G-S9c and G-S9d passed by day, and the night speed rule read 7.04x on a ~512-token text prompt
(10.99 -> 1.57 s TTFT, five passes 6.84-7.42x). Below the floor, on any decline, or with --exact-prefill, the
layer-major pass (bit-identical to sequential) keeps the route.
```

## g4ExpertBatchOn

Moved from `metal/prefill_g4batch.go` (the comment above `g4ExpertBatchOn`) on 2026-10-10.

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

Moved from `metal/prefill_g4paged.go` (the comment above `g4LayerMajorOn`) on 2026-10-10.

```text
g4LayerMajorOn routes a paged Gemma 4 MoE's prompt through prefillG4Paged (docs/tasks/task-m26-mac-2026-10.md, 4b)
instead of the sequential loop. ON since 2026-10-04 (owner's bar, ship at >= 1.02x for a bit-identical change; a smoke
on M26 read 1.42x); the night grade at M = 128 and 512 turns it off if it reads below 1.00x.
```

## loraQGateGap

Moved from `metal/lora.go` (the file comment opening the file) on 2026-10-10.

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
