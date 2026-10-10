# decoder: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `decoder`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

## ReadGoldenJSONForTest

Moved from `decoder/fidelity_testhook.go` (the comment above `ReadGoldenJSONForTest`) on 2026-10-10.

```text
ReadGoldenJSONForTest reads a real-checkpoint golden fixture, transparently gunzipping if path
ends in ".gz". Convention (2026-09-08): a real-checkpoint pin script for a FIXED-resolution
vision family (SigLIP: 896×896 for every image, no choice of a small test photo the way
Qwen2.5-VL's dynamic resolution allows) writes its golden gzip-compressed — gzip on JSON text
this repetitive (a `pixel_values` float array dominates the file) routinely gets 5-10×, the
difference between "fits comfortably in git" and "GitHub warns about it": measured,
testdata/gemma3_real_golden.json was 52.72 MB uncompressed, over GitHub's 50 MB recommendation.
Every golden over 1 MB was migrated to `.json.gz` 2026-09-25; small `.json` goldens stay
uncompressed, which this reads unchanged. Callers keep their own existing
skip-if-missing handling (os.Open's error, not this function, carries that signal).
```

## Model.PrefillLogitsForTest

Moved from `decoder/fidelity_testhook.go` (the comment above `Model.PrefillLogitsForTest`) on 2026-10-10.

```text
PrefillLogitsForTest exposes the CPU backend's batched prompt prefill (prefillLogits) — weights
streamed once and reused across all K positions rather than K separate M=1 passes, ~1.7-2x
faster than the sequential loop, bit-identical to it (the seed token's math is unchanged; only
the earlier positions' unused logits are skipped). Used by
decoder/prefill_ref_gen_test.go so building the §3.1 CPU reference doesn't pay the sequential
loop's cost twice over (once here, once again on the Metal side, which has no batched CPU
equivalent to borrow). Caller controls exact vs fast attention via GOINFER_CPU_FAST_ATTENTION
(t.Setenv("GOINFER_CPU_FAST_ATTENTION", "0") for the exact f64-accumulating kernel the reference
needs) exactly as it would calling the production path.
```

## Model.PrefillLogitsWithAdapterForTest

Moved from `decoder/fidelity_testhook.go` (the comment above `Model.PrefillLogitsWithAdapterForTest`) on 2026-10-10.

```text
PrefillLogitsWithAdapterForTest is PrefillLogitsForTest with a compute-time LoRA adapter
(already loaded via Model.LoadAdapter) bound to a fresh cache before prefilling — the CPU-side
half of a resident-vs-CPU LoRA numeric parity gate (G3, docs/tasks/task-gpu-paths-2026-09.md) driven
from a package (e.g. metal) that cannot reach KVCache.lora or Model.adapter directly, both
unexported.
```

## Model.PrefillLogitsQwenVLForTest

Moved from `decoder/fidelity_testhook.go` (the comment above `Model.PrefillLogitsQwenVLForTest`) on 2026-10-10.

```text
PrefillLogitsQwenVLForTest exposes prefillLogitsQwenVL — the bidirectional-image-block CPU
prefill GenerateQwenVL drives — so a cross-package real-checkpoint gate (gap 0, docs/
multimodal.md) can build a real image's CPU-computed KVCache directly, without going through
GenerateQwenVL's channel-only public API (which exposes sampled tokens, not per-step logits —
unusable for a per-step cosine comparison once a resident hybrid decode's own quantization
noise, an orthogonal and pre-existing property, would otherwise compound through greedy
sampling and swamp the signal this gate actually needs).
```

## NearTieArgmaxForTest

Moved from `decoder/fidelity_testhook.go` (the comment above `NearTieArgmaxForTest`) on 2026-10-10.

```text
NearTieArgmaxForTest reproduces the 3%-near-tie rule cuda/realforward_test.go's argmaxF
comparison established: comparing two logit vectors' argmax, a flip is a defect only if the
REFERENCE's own margin between its pick and the candidate's pick exceeds NearTieHardFailPct of
the reference's logit range -- smaller gaps are quant/reassociation noise, not a real
preference change. gapPct is always computed (0 when they agree), so a caller can report the
worst gap seen across a run even on ticks that don't hard-fail.

READ THE 3% IN LOGIT-RANGE UNITS, NOT PROBABILITY UNITS (N-20, docs/audit-2026-09-10.md): on
Qwen's typical logit range this threshold is ~2 nats, which is NOT a subtle tie in softmax
space -- a 2-nat gap is roughly the separation between probabilities 0.85 and 0.11. "Near-tie"
here means "near" the reference's own full logit spread, not "near" in the sense a reader of
softmax probabilities would expect. This is the tree's established rule (also used inline by
cuda/realforward_test.go and gpu/kv_i8_parity_test.go) restated here for one gate, not a
proposal to change the threshold -- doing that would need to move in all three places at once.
```

## TeacherForcedTop1AgreementForTest

Moved from `decoder/fidelity_testhook.go` (the comment above `TeacherForcedTop1AgreementForTest`) on 2026-10-10.

```text
TeacherForcedTop1AgreementForTest measures how faithfully an engine reproduces a reference
continuation WITHOUT the cascade a free-running greedy comparison carries, where one early
near-tie flip makes every later token diverge and the score collapses to "how long before the
first flip" instead of "how good is the engine at each position on its own". candLogits[i] is
the engine's output at continuation position i when fed the reference's own tokens as context
through position i-1 (teacher-forced, not autoregressive on the engine's own output);
refTokens[i] is the token the reference continuation actually placed at position i. Reports
the fraction of positions where the engine's argmax equals the reference token, and the first
position that disagrees (-1 if none). Returns 0, 0 if the slices are empty or mismatched in
length -- a caller error, not a measurement. firstDivergence is 0 there, never -1, so a caller
gating only on firstDivergence == -1 ("no position disagreed") cannot pass on a comparison that
never ran (audit-2026-09-10 G-13(k)).
```

## PrefillGateProseFiles

Moved from `decoder/fidelity_testhook.go` (the comment above `PrefillGateProseFiles`) on 2026-10-10.

```text
PrefillGateProseFiles are real prose read at run time — not scripts/prompts.json's word-
repetition filler, which docs/completed/task-prefill-gap.md §0 rules out for anything content-dependent
("the fidelity gate (§3) uses prose"). Ten distinct real technical documents from this repo,
chosen only for being real, sizeable (each encodes to well over 3900 tokens on its own, so no
prompt needs repeating to reach the deepest K), and stable — not for their content, the same
reasoning metal/spec_prefill_regression_test.go's readRepoCorpus gives for reading real
repository source instead of a short hand-written corpus.

Paths are relative to a package directory one level under the repo root (as metal/'s and
decoder/'s own test packages both are), so the same list resolves identically from either —
this is shared between metal/prefill_gate_test.go (Metal arms) and
decoder/prefill_ref_gen_test.go (the CPU f32-activation reference, §3.1) precisely so the two
runs score the SAME ten prompts.
```

## PrefillGateProseFilesB

Moved from `decoder/fidelity_testhook.go` (the comment above `PrefillGateProseFilesB`) on 2026-10-10.

```text
PrefillGateProseFilesB is prompt set B (docs/completed/task-prefill-gap.md §4 L1's fresh-prompt decision
run, 2026-09-09): ten more real repo documents, disjoint from set A above, each verified >3900
tokens against S's own tokenizer (queue-correctness.md, the brief's own tenth candidate, was
dropped at 1571 tokens — task-gpu-paths-2026-09.md substitutes, at 48027). LIVE paths, same
"resolves from decoder/ or metal/" convention as set A — but the gate itself reads the SNAPSHOT
under testdata/prefill-gate-prose-b/ (see PrefillGatePromptSet), not these live paths, so a run
stays reproducible as these documents keep changing. Kept here only as the record of what the
snapshot was populated FROM and when.
```

## PrefillGatePromptSet

Moved from `decoder/fidelity_testhook.go` (the comment above `PrefillGatePromptSet`) on 2026-10-10.

```text
PrefillGatePromptSet selects the SNAPSHOT prompt files for the L1 gate, per
GOINFER_PREFILL_GATE_PROMPTS ("" or "a" = set A, "b" = set B). Returns the set's label (for
output-path/log labelling, so set A and set B runs never collide or get confused) and the
snapshot paths themselves — testdata/prefill-gate-prose-<label>/<basename>, resolved with the
same "../testdata/..." convention PrefillGateProseFiles already documents, so decoder/'s and
metal/'s own test packages both resolve it identically.

SNAPSHOTS, NOT THE LIVE PATHS ABOVE. docs/QUEUE.md and docs/benchmarks.md (both in set A) change
most days; task-gpu-paths-2026-09.md (set B) changed within this very session. A gate whose
prompt content silently drifts between Phase A (the reference) and a later Phase B re-run, or
between two Phase B re-runs, is not reproducible — the snapshot is taken once, per set, and the
gate always reads it, so re-running the gate later scores the SAME prompts even if the source
docs have since moved on.
```

## Model.HiddenLast

Moved from `decoder/embed.go` (the comment above `Model.HiddenLast`) on 2026-10-10.

```text
HiddenLast runs ids through the layer stack causally in one fresh KV cache and returns the final
hidden state of the LAST token, AFTER the model's final norm — exactly HF's
`last_hidden_state[:, -1, :]`, which is what sentence-transformers' last-token pooling reads.

It deliberately stops before the LM head: an embedder never needs the logits, and the head is the
single most expensive matmul in a forward (vocab×hidden — for a tied 151k-row Qwen3 head, far
more work than the rest of the token combined).

Padding note (the classic last-token-pooling silent-wrong): this takes ONE sequence and pools the
last element of it, so "last token" is always the last REAL token. There is no padded batch here
in which the last slot could be a pad — callers that want a batch must call this per sequence.

The returned slice is a fresh copy the caller owns: the hidden state aliases the per-token decode
scratch, which the next token would overwrite.

Generic decode path only. The families with their own runLayers (gemma4/qwen35/granite/nemotron/
mla/llama4) return an error rather than a silently wrong vector — the same contract, and the same
guard, as ForwardCapture.
```

## Model.HiddenLast.ownForward

Moved from `decoder/embed.go` (the comment above `Model.HiddenLast.ownForward`) on 2026-10-10.

```text
Derived from the dispatch table. The hand-written list had fallen TWO families behind, not
one: lfm2 and gpt-oss both reached this seam (audit-2026-09-02 C-02).
```

## Model.HiddenLast.resident

Moved from `decoder/embed.go` (the comment above `Model.HiddenLast.resident`) on 2026-10-10.

```text
G4 (docs/tasks/task-gpu-paths-2026-09.md): on a GPU box this arch may decode resident while
embedding requests still ran the whole text decoder on the CPU — the same class of gap G2
documents for image turns. Same resBusy claim Generate uses (M9): a loser (a generation
already in flight on this Model) falls through to the CPU path below exactly like it
always has, and resIDs is left unknown either way (residentForgetIDs), since a HiddenLast
prefill has nothing durable worth remembering for the next Generate call.
```

## Model.HiddenLast.canBatchN

Moved from `decoder/embed.go` (the comment above `Model.HiddenLast.canBatchN`) on 2026-10-10.

```text
P-17: canBatchN excludes K==1 (nothing to batch), the own-runLayers families (already
rejected above), and the NonGatedMLP/LearnedPosEmbed families runLayersFromEmbedN doesn't
implement — those keep the per-token loop (hiddenLastSequential). Everything else runs the
whole sequence through the SAME batched prefill path plain generation's prompt phase
already uses (the "sequential prefill" this seam took the ~9x-slower name from), instead of
one runLayers call per token.
```

## Model.checkHiddenIDs

Moved from `decoder/embed.go` (the comment above `Model.checkHiddenIDs`) on 2026-10-10.

```text
A LENGTH BOUND, NOT JUST A VOCAB ONE. This preallocates KV for len(ids) positions and then
runs one sequential forward per token with no context to cancel it, so an over-long input is
not slow — it is a ~114 GB allocation (28 layers, kvDim 1024, 500k positions) and attention
over up to len(ids) keys per token, holding the caller's mutex until the process is OOM-killed.
The serving embedder now truncates to MaxPositions before it gets here (C-07), and this is
the same bound stated where the cost is actually incurred, so a DIFFERENT caller cannot
reintroduce it. Positions past the window would also pool from out-of-range RoPE — plausible,
and wrong — which is the quieter half of the same defect.
m.Config().MaxPositions (max_position_embeddings), NOT a.MaxPositions — the Architecture field
of that name is the GPT-2 learned-position TABLE SIZE and is 0 for every RoPE family, so
keying on it would have made this guard silently inert for almost every model. Same shape as
the LFM2 bugs: the wrong key reads as a legal zero.
```

## Model.PromptHidden

Moved from `decoder/embed.go` (the comment above `Model.PromptHidden`) on 2026-10-10.

```text
PromptHidden returns the final-norm hidden state at the last position of prompt: HF's output_hidden_states[-1][:, -1], which is
last_hidden_state[:, -1]. It is the input a Route B decision head reads (D2, docs/tasks/task-constrained-confidence.md): JEV's
readout takes the last prompt token's final-norm hidden state through a linear head.

Unlike HiddenLast it serves every family, those with their own layer loop included (Qwen3.5, the JEV models' family): it runs
the same per-token forward Generate's CPU path runs, runLayers, which dispatches to them, in a fresh cache, then the final norm
logitsFromHidden applies before the LM head. So the hidden state is the one the logits are computed from, by construction. A
family on the generic batched path (canBatchN) takes hiddenLastBatched instead, which ends at the same final norm, and
Qwen3.5 takes its own batched forward (runLayersQwen35N: every projection one matmul over the prompt, the DeltaNet
recurrence still sequential), bounded against the per-token forward by TestPromptHidden_batchedMatchesSequential.

On a resident backend it runs on the device first, through ResidentHiddenLast (CUDA's batched prefill with a headless tail,
Metal's per-token headless forward), and falls back to the CPU on a decline. The resident's numerics are its kernels', not
the CPU reference's: TestPromptHidden_residentMatchesCPU in the backend modules bounds the difference. ctx is checked between
tokens (CPU) or at the backend's own granularity, so a long prompt can be abandoned.
```

## fusedattn

Moved from `decoder/fusedattn.go` (the comment above `fusedattn`) on 2026-10-10.

```text
Fused (FlashAttention-style) prefill attention — P19.

The materialized schedule writes a kt x nKeys score block, reads and rewrites
it in the softmax, and reads it again for scores*V: three trips through memory
for a block that is 8 MiB at the production tile budget. This blocks over KEYS
instead, keeping the score block small enough to stay in cache and folding it
into the output accumulator with a running max and a running sum, so the
kt x nKeys matrix never exists.

MEASURED before it was written into the forward
(docs/measurements/p19-fused-attention-2026-09-01.md). Causal, row-parallel,
summed over all 32 tiles of an 8192-token prefill: 1.69-1.73x, cosine
1.000000000. Four configurations were measured and three of them imply the
WRONG verdict -- column-parallel says 0.70x (close it), a single last tile says
1.2x (park it) -- so the number above is specifically the one production runs.

IT IS NOT BIT-IDENTICAL, and that is structural rather than incidental: the
running-max rescale re-associates the softmax denominator and the AV fold. This
is the same category as --cpu-fast-attention, and it rides that flag rather
than adding a second user-facing one -- but NOT for the reason first written
here. That claimed the added divergence was "~5 orders of magnitude smaller",
comparing a KERNEL number (max|diff| 9.3e-9) to a MODEL-LEVEL one. Measured on
one checkpoint at one depth, both at model level: acc64 vs f32-materialized is
cosine 0.998283, acc64 vs f32-fused is 0.998262. Same order; what is small is
fusion's INCREMENT (~2e-5 of cosine), not its magnitude. The conclusion holds,
the arithmetic behind it did not. GOINFER_FUSED_ATTENTION is a
developer A/B handle, not a user setting -- it exists so fusion's win stays
attributable separately from A3's, and so it can be rolled back without losing
A3's.
DEFAULT ON since 2026-09-01, by operator decision. The measurement does not
make the case on its own and that is recorded rather than smoothed: the fused
schedule is 1.69-1.73x at the KERNEL over a whole prefill's tiles, but only
1.080x END-TO-END (dense 1.5B, K=4096, paired) -- because A3's head fan-out
already took most of what attention had to give, leaving it ~18% of this
prefill by Amdahl. Eight percent, bought with a user-visible output change.

GOINFER_FUSED_ATTENTION=0 restores the materialized schedule.
```

## fusedKeyBlock

Moved from `decoder/fusedattn.go` (the comment above `fusedKeyBlock`) on 2026-10-10.

```text
fusedKeyBlock is the key-block width. 256 and 512 measured within noise of each
other (1.731x / 1.687x) and both beat 1024; 512 keeps the per-tile score block
at kt*512 floats (512 KiB at kt=256), which is the point -- small enough to stay
resident, which is the entire mechanism.
```

## ggufQwen35DenseConfig

Moved from `decoder/gguf_qwen35.go` (the comment above `ggufQwen35DenseConfig`) on 2026-10-10.

```text
ggufQwen35DenseConfig reads the DENSE hybrid (llama.cpp arch "qwen35" — Qwen3.8), the sibling of
"qwen35moe" above. Same Gated-DeltaNet/softmax 3:1 interleave, same SSM geometry keys; the only
structural difference is a plain SwiGLU where the MoE sibling has a router, so the expert keys are
replaced by feed_forward_length. It resolves to the qwen3_5 adapter, which is the dense descriptor
the safetensors bring-up added.

WHY IT IS WORTH HAVING at all, given the safetensors loader already works: the 27.8B ships as
55.6 GB of bf16, which goinfer re-quantizes on EVERY load (68 s measured), against 16.5 GB of
pre-quantized GGUF that mmaps in about a second. Same model, 3.4x less disk and a load that stops
dominating short runs.
```

## ggufQwen35DenseConfig.rope

Moved from `decoder/gguf_qwen35.go` (the comment above `ggufQwen35DenseConfig.rope`) on 2026-10-10.

```text
rope.dimension_sections ([11,11,10,0]) is the m-RoPE split. For TEXT the three position
components are identical, so interleaved m-RoPE reduces exactly to standard partial RoPE
(verified against modeling_qwen3_5.py during the safetensors bring-up) and ropeAt never reads
the split without image positions; it is carried (ggufMRopeJSON) so a GGUF text model can take
an image turn with an mmproj tower (P8b, docs/multimodal.md F5), as the safetensors config does.
```

## geluErf

Moved from `decoder/rmsnorm.go` (the comment above `geluErf`) on 2026-10-10.

```text
geluErf is the EXACT GELU — x·Φ(x) with the true Gaussian CDF, HF's "gelu"
(ACT2FN["gelu"] = GELUActivation). It is a DIFFERENT FUNCTION from geluTanh, not a
spelling of it: they differ by up to 4.73e-4 (worst at x ≈ -2.7).

That gap is small — well under int8 quantization error — which is exactly why the
conflation survived. goinfer previously accepted `activation_function: "gelu"` for GPT-2
and ran geluTanh regardless, so a checkpoint asking for the exact function silently got
the approximation. aikit hit the mirror image of this on its encoder side (three tanh
names routed through erf) and fixed it in v1.19.0; this is the decoder-side counterpart.

In float64 for the same reason geluTanh is: parity with the reference implementation.
```

## loadprofile

Moved from `decoder/loadprofile.go` (the comment above `loadprofile`) on 2026-10-10.

```text
Load-time instrumentation (task: knowing what to download, and how long it takes to load,
Part B). Before this, the only record of load cost anywhere in the tree was a prose claim in a
comment — that fanning the GGUF parse across cores turned a 12B's roughly two-minute load into
seconds. A real and significant result with no measurement behind it, and no way for a user to
see the number on their own machine.

THE SPLIT IS THE POINT, not the total. A slow load caused by storage and a slow load caused by
repacking call for completely different responses — a faster disk versus a different quant — and
a single number cannot tell them apart. The phases are therefore chosen to separate those:

	map    the file becoming addressable: open + mmap + header parse. NOT the file read — see below.
	build  tensors becoming resident weights: dequantize, quantize, repack. CPU-bound.
	resident  weights becoming a device-side runner, where a backend builds one. PCIe/GPU-bound.

`map` DOES NOT ISOLATE STORAGE, and the first measurement is what showed it. The loader mmaps, so
pages fault in lazily during `build`, not during `map` — `map` is header parse alone. Measured
2026-09-06 on NVMe: Phi-3-mini (2.23 GB) spent 7ms in `map` and 5.18s in `build`, and the 0.5B
model spent MORE map time (48ms) on a fifth of the bytes, because that phase tracks metadata
count rather than size. The storage cost is real but lands inside `build`, where it shows up as
the cold-minus-warm delta rather than as its own phase.

That delta turned out to be small — 3.3% and 4.4% across the two models — because the repack is
CPU-bound enough that the kernel's readahead hides most of the I/O behind it. On a spinning disk
or a network mount that would not hold, which is exactly why the regime has to be recorded
rather than assumed. Isolating storage properly would need a non-mmap read path or per-phase
fault accounting; neither is worth it while the answer is "storage is not the problem here".

WHAT THIS CANNOT TELL YOU, and what therefore has to be recorded by whoever measures: whether the
page cache was cold. A warm-cache `map` phase measures memcpy; a cold one measures the disk, and
re-running a benchmark gives you a warm one by accident. The profile reports bytes and a rate so
the regime is at least visible in the number, but the label belongs in docs/benchmarks.md next
to the storage class, not here.
```

## layerPager

Moved from `decoder/layerpaging.go` (the comment above `layerPager`) on 2026-10-10.

```text
layerPager streams a DENSE model's per-layer weights out of the read-only .giw
mapping (idea #4, docs/ideas-weight-memory.md). Unlike MoE expert paging, the
transformer layer loop is sequential and fully known in advance, so the pager
PREFETCHES the upcoming layer (Advise WILLNEED) while the current one computes —
overlapping the fault with compute — and RELEASES (Advise DONTNEED) the layer
that slides out the back of a window. This is a windowed prefetch, NOT an LRU, so
it does not use aikit/mmap.SpanCache (whose policy is least-recently-touched); it
borrows only the generic span-alignment (WeightMat.MappedSpan), the residency hint
(mmap.Advise), and the RAM budget (mmap.AutoBudget) — the layer-order demand signal
stays here. Resident weight RAM is bounded to ~window layers, so a dense model too
big for RAM still runs (the floor is NVMe bandwidth: a model that doesn't fit is
re-read ~once per token). Bit-exact — the mapping is read-only and file-backed, so
a released layer re-faults from disk with identical bytes (aikit's
TestMadvise_dontneedRefaultsIntact proves the property).

The resident floor is NOT zero: only the 7 per-layer projections stream. The token
embedding, final norm, and LM head aren't per-layer, so they stay resident, plus
the live window. For a big-vocab model embed+head alone can be a multi-GB floor —
the complementary lever there is idea #3 (sub-int8 embed/head). So "bigger than
RAM" means bounded to floor + window, not ≈ 0.

Built only for mmap-backed dense .giw models; nil when the model is MoE (that's
idea #2's expertPager), heap-backed, or small enough to fit the budget whole.
Guarded by an internal mutex (audit C-30): the pager lives on *Model and StreamWeights
supports concurrent decode streams, so its shared paging state is locked.
```

## newLayerPager.ownForward

Moved from `decoder/layerpaging.go` (the comment above `newLayerPager.ownForward`) on 2026-10-10.

```text
Own-forward families run their own layer loop that never calls enterLayer, so a pager would
print a RAM-bound banner it can't deliver (N-13; the dense ones are gemma4, nemotron and
lfm2 — the rest are MoE, already excluded above). Only the generic dense forward pages.
Derived from the dispatch table: the hand-written list here missed lfm2 (C-02/C-03).
```

## newLayerPager.row4

Moved from `decoder/layerpaging.go` (the comment above `newLayerPager.row4`) on 2026-10-10.

```text
Register only the span the M=1 decode kernel will actually read: row4
when present, canonical otherwise — never both. Registering both under
one cache key was a real, measured bug (moepaging.go's addExpert,
docs/completed/task-zeno-compare.md's "At-scale acceptance run"): SpanCache.Touch
WILLNEEDs every span under a key unconditionally, so a cold kind-4 touch
prefetched the unread canonical copy too — a fixed ~2x I/O tax per miss.
```

## cpu_batch

Moved from `decoder/cpu_batch.go` (the comment above `cpu_batch`) on 2026-10-10.

```text
MC3c step 2 (docs/tasks/task-concurrency-2026-09.md): several CPU generations of one model decoding at once join
their decode tokens into one batched forward (decodeMultiStep) instead of running N independent forwards. The
trigger measured it on nobara's 7B: batched B = 4 is 2.25-2.41x J8's 4 independent workers, which barely scale
there (docs/measurements/concurrency-mc2-2026-09-26.md, "Linux 7B cell").

The coalescing is MC3's (tokenCoalescer, mc3_batch.go): a run starts once every decoding generation has submitted
its token, or when the straggler window expires. A run of at least two tokens is one decodeMultiStep; a lone token is
production's own m.forward, so a request served alone takes exactly today's path. Unlike the resident batcher there
is nothing to hold exclusively — prefill and every non-batched path run outside it, as under step 1.
```

## cpuBatchAutoMinBytes

Moved from `decoder/cpu_batch.go` (the comment above `cpuBatchAutoMinBytes`) on 2026-10-10.

```text
cpuBatchAutoMinBytes is CPUBatchAuto's threshold, pre-registered 2026-09-27. Batching's lead over the workers at
B = N = 4 measured 0.64-0.75x on the 0.5B (0.47 GB of dense weights), 1.02-1.11x on the 1.5B (1.23 GB) and 2.25-2.41x
on the 7B (4.93 GB) on nobara, and the workers led on the Mac's 0.5B and 1.5B. On the Mac's 7B batching leads by
1.54x (depth 128) and 1.38x (512) (2026-09-27, docs/measurements/concurrency-mc2-2026-09-26.md, "Mac 7B cell"), so
the threshold holds on darwin too. Below it the workers are no worse off; above it batching wins by a wide margin.
The range between 2 GiB and the 7B is unmeasured.
```

## cpuBatchFusedW4A8

Moved from `decoder/cpu_batch.go` (the comment above `cpuBatchFusedW4A8`) on 2026-10-10.

```text
cpuBatchFusedW4A8 lets decodeMultiStep run an int4 layer's q‖k‖v, and its gate‖up, each as ONE batched W4A8 call
over the B rows (MC3c step 2 S1, docs/tasks/task-concurrency-2026-09.md) instead of one matmul per projection. The
kernel (linalg.MatmulBTW4A8Batch) is numerically identical to calling MatmulBTW4A8Into once per op, so this changes
only how many fork/joins a step pays: the probe put the small projections (q/o, k/v) at 1.6-3.0x one row's cost at
M = 4, which is fork/join and compute, not bandwidth. A package variable, not an environment read: the S1
measurement flips it in-process to interleave its arms.
```

## fp8Reconstruct.shape

Moved from `decoder/fp8.go` (the comment above `fp8Reconstruct.shape`) on 2026-10-10.

```text
Shape is checked against the ARCHITECTURE (in/out from the config), not just against
the scale grid: a weight/scale pair can be self-consistent and still be the wrong
tensor, which is the check gptoss_safetensors.go's own comment insists on.

N-06: this comment was true of the INTENT and false of the code, which compared only the
element COUNT. out*in == in*out, so a [in,out]-shaped tensor passed and then loaded
transposed — every weight present, every one in the wrong place. The declared shape is
what distinguishes them, so compare that.
```

## routerCapture

Moved from `decoder/routercapture.go` (the comment above `routerCapture`) on 2026-10-10.

```text
routerCapture is a test seam (default off; tests set it directly — its GOINFER_ROUTER_CAPTURE env
read was retired 2026-09-24, docs/tasks/task-env-config-2026-09.md phase 6): when on, gemma4MoEFFN appends each MoE-layer call's selected top-k expert
indices to routerCaptureBuf, in call order (token-outer, layer-inner — one entry per
layer per token). It is OBSERVE-ONLY: it copies out `idx` and changes no compute, so with
it off the forward is byte-identical.

Probe #1 in docs/task-gemma4-moe.md uses it to tell routing collapse from uniform weight
noise: capture selections for the int8 run and a 4-bit run over the SAME teacher-forced
token sequence, then compare per-layer top-k overlap and per-layer selection entropy.
Repetitive-English output is the signature of routing collapse, not of weight noise; if
the 4-bit run's selections have degenerated vs int8, more bits on the expert weights
won't close the gap and the plan redirects to router-input cleanliness.
```

## routerCaptureMu

Moved from `decoder/routercapture.go` (the comment above `routerCaptureMu`) on 2026-10-10.

```text
N-27: THE BUFFERS ABOVE ARE PACKAGE-LEVEL AND WERE APPENDED FROM INSIDE THE FORWARD WITH NO
LOCK AND NO BOUND.

Two separate problems, and the diagnostic framing hid both. Under the documented
concurrent-sequence contract two goroutines can be in a forward at once, so the appends are a
data race on a slice header — a crash, not a wrong number. And there is no cap: set on a long
running `serve` process this grows without limit, one entry per MoE decision per layer per
token, each carrying a copy of a hidden-sized vector.

Fixed here rather than by refusing under `serve`: the decoder cannot see who its caller is,
and a diagnostic that is safe everywhere is better than one that is refused in the one place
it is dangerous. The mutex removes the race; the cap turns an unbounded leak into a bounded
buffer that says when it stopped.
```

## sampler_topk

Moved from `decoder/sampler_topk.go` (the comment above `sampler_topk`) on 2026-10-10.

```text
TEMPERATURE-ONLY SAMPLING IS NOT SERVED HERE, and cannot be: it draws by inverse CDF in vocabulary
index order over a full-V normalisation (sampleChunked), and P2b already refuted a truncated-tail
shortcut. It stays on the full-row path.

top-p's Z comes from the device (an f32 exp sum reduced in f64), not from chunkedZ's f64 sum, so it
differs from the host's by rounding. It can only matter when the cumulative mass lands within that
rounding of topP·Z — the same class of given-seed shift P2b accepted for regrouping Z on the host.
The mismatch rate against the full path is measured and recorded (cuda TestSampledTopKStreamIdentity)
rather than assumed.
```

## SampleFromTopK.nucleus

Moved from `decoder/sampler_topk.go` (the comment above `SampleFromTopK.nucleus`) on 2026-10-10.

```text
Nucleus-only. Walk the candidates in row order — (logit desc, id asc), which is (prob desc, id
asc) for every prefix that matters, since distinct float32 logits never collide after exp — and
stop as soon as the cumulative mass reaches topP·Z. That both PROVES the K carry the nucleus (if
the loop runs out first, they might not) and bounds the exp work to the nucleus, which is what
this costs on a core that has just woken from the GPU sync: math.Exp measured ~360 ns/call
there, so the old "sum all K, then exp all K again" pass was ~190 us per token. cum here is
accumulated in the same order finishFilter will use, so its cut lands at the same index. No
tie handling past the cut is needed: two distinct float32 logits do not give the same float64
exp at any realistic temperature (it takes T around 1e7), so equal probabilities mean equal logits,
which the row already orders by ascending id —
the order finishFilter sorts to. (A mutation that deleted such a loop survived every test, which
is how this was found to be dead.)
```

## mamba2

Moved from `decoder/mamba2.go` (the comment above `mamba2`) on 2026-10-10.

```text
Mamba-2 selective state-space sequence mixer (Granite-4.0 hybrid). Like Gated
DeltaNet (deltanet.go) it is an inherently sequential recurrence driven one token
at a time by both prefill and decode, with a small per-token causal conv before
the scan. This is the parity-first SEQUENTIAL form, and the only one: a chunked/segsum
scan was built against it as an oracle and removed unused on 2026-09-24 (4f8da77a). It
mirrors the HF GraniteMoeHybridMambaLayer single-token (cache) path exactly.
```

## DSparkDrafter

Moved from `decoder/dspark.go` (the comment above `DSparkDrafter`) on 2026-10-10.

```text
DSparkDrafter is an imported DeepSeek DSpark block drafter (P10 / docs/spec/08): the same
non-causal block trunk DFlash uses, plus the three things DFlash does not have — its own
embedding and LM head, a rank-256 Markov chain, and a confidence head.

It reuses blockTrunk rather than reimplementing the forward, and that is a measured claim,
not a convenience: DeepSpec's `_forward_backbone` and z-lab's `DFlashDraftModel.forward`
compute the same thing, down to the split RoPE application. See blockTrunk's doc.

The differences that DO matter, all of which a port gets wrong silently:

  - **logits_start = 0.** All blockSize positions are draft predictions; slot 0 both embeds
    the anchor AND predicts the first token. DFlash reserves slot 0 and predicts from 1.
    Slicing the wrong one makes every draft land one position late, which halves acceptance
    while the text stays correct — so nothing crashes and no gate but this one notices.
  - **Its own embed/head.** DSpark ships frozen COPIES of the target's (778 M of its 1.39 B).
    They are loaded here; reusing the resident target's instead is a later optimization that
    must be proven equal first, not assumed.
  - **The Markov chain is SEQUENTIAL.** logits[i] += w2(w1[prev]) where prev is the token
    just sampled at i-1, so the block is parallel in the trunk and serial in a blockSize-step
    scalar chain. Each step is a [vocab, 256] matvec — small against a layer, but latency-serial
    and therefore inside the draft term that gate 3 is most sensitive to.
  - **The confidence head is adaptive block LENGTH, not a fire/don't-fire router.** Measured:
    gating trims the proposal (chat 6.96 -> 4.87 positions) and barely moves acceptance
    (3.04 -> 2.96), so what it buys is a cheaper verify. See docs/spec/08.
```

## swapwatch

Moved from `decoder/swapwatch.go` (the comment above `swapwatch`) on 2026-10-10.

```text
S3 (docs/tasks/task-never-swap-2026-09.md): the swap tripwire. goinfer notices swap growing
during its own run and acts — instead of leaving "kill at swap-used baseline + 500 MB" to a
human watching `free -m`/`sysctl vm.swapusage` in another terminal, which is exactly what the
cold-user harness and the R11(c) runs have had to do by hand. The external shell script that
polled `vm.swapusage` every 1s and SIGKILLed on two consecutive >80 MB ticks (or one >300 MB
jump) cut the peak excursion 5-7x against manual monitoring
(docs/measurements/metal-moe-autopager-m26-2026-09-20.md's third attempt) — this moves that
same idea inside the process and gives it a way to act short of SIGKILL.

Keys on SWAP-USED, never RSS: darwin RSS reports what survived reclaim under memory pressure
(CLAUDE.md's "a guard that INVERTS under the condition it exists for" — measured on this
exact R11(c) run: RSS read "7 MB -> 892 MB after build" while swap grew from 2.3 to 12 GB
during that same build, because the MTLBuffer pages were being compressed and evicted as fast
as they were written). Swap-used only grows when the machine is actually losing memory, which
is the one figure this guard can trust.
```

## quantConfig.CheckpointFormat

Moved from `decoder/gptq.go` (the comment above `quantConfig.CheckpointFormat`) on 2026-10-10.

```text
N-05: GPTQModel's v2 export stores zero-points WITHOUT the v1 "+1" bias. This loader
applies the +1 unconditionally, so a gptq_v2 checkpoint dequantizes every weight one
scale step low — finite, plausible, and wrong everywhere at once. The field was not
even parsed, so there was nothing to notice.
```

## parseQuantConfig.gptqV2

Moved from `decoder/gptq.go` (the comment above `parseQuantConfig.gptqV2`) on 2026-10-10.

```text
N-05: gptqReconstruct applies the v1 "+1" zero-point bias UNCONDITIONALLY. GPTQModel's
v2 export drops that bias, so a v2 checkpoint would dequantize every weight one scale
step low — no error, no NaN, just a uniformly wrong model. Refuse rather than guess:
implementing the v2 path without a v2 checkpoint to validate against would be the same
unverified change this audit has declined elsewhere, and the failure mode here is
specifically the silent kind.

"" and "gptq" both mean v1 (the field predates v2 and older exports omit it).
```

## doc.families

Moved from `decoder/doc.go` (the comment above `doc.families`) on 2026-10-10.

```text
  - Families: Gemma 3/4, Qwen3, Qwen2.5 (+ Qwen2-MoE), Llama-2/3, Mistral,
    Mellum2 (MoE + sliding/full interleave + YaRN), GPT-2, and Mixtral (sparse-MoE).
```

## sampler_gumbel

Moved from `decoder/sampler_gumbel.go` (the comment above `sampler_gumbel`) on 2026-10-10.

```text
Temperature-only sampling by Gumbel-max (docs/tasks/red-october.md R7b, owner decision 2026-09-20).

WHAT IT IS. For a logit row l and temperature T, the token is

	argmax_i ( l_i/T + G_i ),   G_i = -ln(E_i),   E_i = -ln(1 - w_i),   w_i = (h_i + 0.5) / 2^32

where h_i is a 32-bit word from Philox4x32-10 keyed by (seed) with counter (i>>2, draw, draw>>32, 0), lane
i&3. By the Gumbel-max theorem that is an exact draw from softmax(l/T), with no normalisation, no
cumulative sum and no per-token dependence on the other tokens — so it is an argmax, which every GPU
backend can do in one parallel pass with no f64, and which needs no full-vocabulary readback.

WHY THIS REPLACED THE INVERSE-CDF DRAW. The old temperature-only path (sampleChunked, now the reference
in sampler_chunked_ref_test.go) drew by cumulative search in vocabulary INDEX order over a full
normalisation. That cannot be reproduced from a top-K and needs f64 exp to reproduce on a device, so the
server's default sampling shape (temperature 1, no filters) sat at ~0.74 of greedy speed on CUDA. This
path is ~0.97 there and works on Metal and WebGPU.

THE COST, DISCLOSED: for a given seed this draws DIFFERENT tokens than every earlier release (the
distribution is unchanged; the stream is not). Speculative sampled decoding keeps drawing from explicit
probabilities (it needs them for accept/reject) and was never stream-equal to plain decoding — it is
in-distribution lossless (spec_sample.go).
```

## optiongrid

Moved from `decoder/optiongrid.go` (the comment above `optiongrid`) on 2026-10-10.

```text
The option grid (docs/tasks/task-option-path-admission-2026-10.md §4.0, stage 2): every
decoder.Options field against every execution path, with what that path does with it and the
evidence. Step 1 (docs/measurements/audit-classes-2026-10-08.md) found 22 audit findings of one
shape here — a load option no guard on a path knew about (-kv i8 into Metal's batched prefill,
09-30 A-C01; paged MoE slots into three Metal entry points, 09-10 C-08 and 09-12 C-02; int4 into
WebGPU's speculative verify, 09-10 M-09).

Owner decision 2026-10-08: no behaviour changes in this stage. A combination that runs today
without a test that drives it is declared ogUntested and keeps running; TestOptionGrid_ratchet
holds the number of such cells at a ceiling that can only come down. What fails closed is the
registry itself: a new Options field fails TestOptionGrid_everyOptionClassified until it is
classified on every path, and a cell that claims a test or a decline must name one that exists.
```

## optiongrid.MoECacheExperts

Moved from `decoder/optiongrid.go` (the comment above `optiongrid.MoECacheExperts`) on 2026-10-10.

```text
Metal's expert cache (docs/tasks/task-option-path-admission-2026-10.md §4.3, finding 2, closed 2026-10-08): bit-identical
to fully resident when both prefill the same way. A paged model's prompt takes the layer-major prefill, bit-identical
to sequential, so the reference is the resident load with ExactPrefill; the 0.004 first recorded here was the
default resident's batched f16 prefill, which --exact-prefill alone reproduces. int8 experts decline by name
(errPagedExpertsNotInt4) instead of a recovered panic.
```

## sampler_chunked

Moved from `decoder/sampler_chunked.go` (the comment above `sampler_chunked`) on 2026-10-10.

```text
P2b — deterministic parallel host normalization.

The temperature-only path's cost is the full-vocabulary exp+sum, measured at ~31-34 ns per
vocabulary entry and ~78% of the whole temperature-only penalty (bc59c56). Lazy Z tried to SKIP
that work and was refuted; this instead does the same work in parallel, and deletes the separate
normalize-divide pass by drawing against unnormalized weights.
```

## chunkedZ

Moved from `decoder/sampler_chunked.go` (the comment above `chunkedZ`) on 2026-10-10.

```text
chunkedZ is the top-p denominator (step 3), given the same fixed-chunk treatment so the nucleus
cut shifts in the SAME release as the temperature-only change rather than dribbling out later.
scratch is the caller's reused vocab-sized buffer (Sampler.vocabBuf via vocabBufN) — expChunked's
per-id `e` output is discarded here (only the folded sum matters), so scratch just needs len(logits).
```

## HostRAMBytes

Moved from `decoder/hostram_darwin.go` (the comment above `HostRAMBytes`) on 2026-10-10.

```text
HostRAMBytes is this machine's physical RAM, or 0 when it cannot be determined — and 0 is a
real answer that every caller must treat as "proceed", never as "no memory".

The hw.memsize sysctl read with a bare sysctl(2) through the stdlib, once per process. It is a
little-endian u64; syscall.Sysctl drops a single trailing NUL byte (16 GB = 0x4_0000_0000 loses its
zero high byte), so the value is zero-extended back to 8 bytes. An earlier version shelled out to
`sysctl -n hw.memsize` on the belief that syscall.Sysctl truncates at the FIRST NUL; it drops only
the last one. Not forking matters here: every exec is a fork(), and a fork of a process whose .giw
mapping a GPU backend has wired used to copy the whole mapping
(docs/measurements/m26-alias-fork-collapse-2026-09-24.md).
```

## HostRAMAvailableBytes

Moved from `decoder/hostram_darwin.go` (the comment above `HostRAMAvailableBytes`) on 2026-10-10.

```text
HostRAMAvailableBytes is this machine's CURRENTLY AVAILABLE memory — free plus reclaimable
pages, an approximation of what `vm_stat` and Activity Monitor's "memory pressure" both draw
from — or 0 when it cannot be determined. Unlike HostRAMBytes, this is NOT cached: it changes
continuously as other processes run, which is the entire reason it exists (R13-follow-on,
docs/measurements/cold-user-2026-09-07-macbook-arm64.md's live re-run of the R13 fix): a
budget computed as a fraction of TOTAL RAM assumes nothing else on the machine ever needs more
than the remaining fraction, which a real laptop with a browser and an IDE open routinely
breaks. `serve check`'s own requests pushed a load that the load-time guard had already
correctly auto-pinned into 9.7 GB of swap, because "70% of 16 GB" was never actually free —
this reads what IS actually free instead.

Still an exec of vm_stat (the inactive-page count has no sysctl): it runs at load and from the web
UI's status, not on a timer, and the .giw mapping is VM_INHERIT_NONE (forkinherit_darwin.go), so the
fork it costs no longer copies the weights.

APPROXIMATION, STATED RATHER THAN HIDDEN: free + inactive + speculative + purgeable pages,
matching the pages vm_stat itself reports and the ones macOS reclaims before it would ever
swap (inactive is the disk-cache equivalent; purgeable is explicitly discardable on demand).
wired and active pages are excluded — those are genuinely in use. This is the same kind of
measured-approximation-with-a-stated-residual as quantBytesPerElem's probe matrix (fitguard.go)
— real numbers from the real tool, with the one simplification named.

Apple Silicon uses 16 KB pages, not the traditional 4 KB (confirmed on the M1 Pro that found
this bug) — the page size is read from vm_stat's own header, never assumed.
```

## cachestate

Moved from `decoder/cachestate.go` (the comment above `cachestate`) on 2026-10-10.

```text
The cache-state grid (docs/tasks/task-option-path-admission-2026-10.md §4.0): every kind of
per-sequence state a KVCache can hold, against every lifecycle path that has to know about it,
with what that path does with it. Step 1 of that task (docs/measurements/audit-classes-2026-10-08.md)
counted ten audit findings of one shape here — a kind of state a lifecycle path never accounted
for: Mamba-2 and DeltaNet state across a reset (08-05 C-01, C-17), LFM2's conv window at every
site (09-02 C-02), Bailing Hybrid's KDA state at every site (09-10 C-03), the adapter that built
a reused prefix (09-10 C-02). Each was found by an audit, after it shipped, one path at a time.
```

## kda_rehearsal

Moved from `decoder/kda_rehearsal.go` (the comment above `kda_rehearsal`) on 2026-10-10.

```text
F4 (docs/completed/task-families-2026-09.md): KDA (Kimi Delta Attention) recurrence rehearsal for
Ling-3.0-tiny / Kimi K3's linear-attention mixer, written as the scoped bring-up the F4 brief
asked for: prove the one genuinely new piece of KDA's math against a real reference before any
registry work. N-08 (audit-2026-09-10): "NOT wired to any registered family" went stale as of
e6b31cc — kdaMixerStep (decoder/kda.go, called from forward_bailing.go's served path for
bailing_hybrid/Ling 3.0) reuses kdaLowerBoundGate and kdaRecurrentStep below directly, so this
is production code now, not just a bring-up rehearsal.

Verified against fla-org/flash-linear-attention's actual source (fla/ops/kda/{naive,gate}.py,
not the HF modeling file's paraphrase, which only calls the opaque Triton kernel): KDA's
delta-rule recurrence is structurally IDENTICAL to the Gated DeltaNet this repo already ships
for qwen3_5_moe (gatedDeltaNetStep, decoder/deltanet.go) -- same beta write-gate, same
outer-product delta update, same q/k L2-norm-in-kernel, same final q·S read -- except the decay
that Gated DeltaNet applies as ONE SCALAR to the whole [head_k_dim, head_v_dim] state block is,
in KDA, PER-CHANNEL: one decay value per row of S (one per key-dimension), not one for the
entire block. That is the one new primitive; everything else composes from what qwen3_5_moe
already validated.
```

## selftest_gpu.fixtures

Moved from `decoder/selftest_gpu.go` (the comment above `selftest_gpu.fixtures`) on 2026-10-10.

```text
The fixtures are four tiny random-weight checkpoints, chosen for the kernels they reach: qwen35vl-tiny (a 512-position window, so attention runs past 256 keys, beyond both softmax reduction widths,
beside DeltaNet layers), llama-tiny (the plain dense path), gemma3-vl-tiny (GELU-tanh and QK-norm: the sink-token GELU NaN class) and qwen35-tiny (the DeltaNet hybrid with a MoE FFN). Phi-3 was tried
first and declines to go resident on CUDA by design, so it would have checked nothing. They are embedded copies of testdata/ ones, and a
test asserts they stay byte-identical.
```

## probeMeasured

Moved from `decoder/selftest_gpu.go` (the comment above `probeMeasured`) on 2026-10-10.

```text
probeMeasured lists the backends whose healthy-hardware margins against the bars were measured, each on ONE device (the task doc has the figures). A backend not on it is NOT probed (its result is
simply absent from the report) rather than probed against bars nobody has seen it clear. WebGPU is on it from one NVIDIA adapter over Vulkan; AMD, Intel and DX12 adapters are probed against those bars
unmeasured, which is a risk of a false decline, disclosed in the task doc. Metal is on it from one M1 Pro (int4 worst cosine 0.99937 / relative L2 0.0358, int8int8 0.99971 / 0.0243, three runs
each); a fixture Metal re-quantizes to int4 is not compared (residentPrecision). Add a backend here with its margins.
```

## prefix_share

Moved from `decoder/prefix_share.go` (the comment above `prefix_share`) on 2026-10-10.

```text
D8 (docs/tasks/task-constrained-confidence.md): shared state, many questions. A decision request asks several questions about one state, and on the
hybrid Qwen3.5 family each question used to prefill the whole prompt again (decisions-d7-2026-09-28.md: five questions cost exactly five times one).
The state is the same in every question's prompt, so its prefill can be done once and its CACHE (the KV rows of the attention layers and the Gated
DeltaNet's recurrent state, which has no per-position history and so cannot be rewound, only copied) resumed once per question.

What this covers, and what it deliberately does not:
  - The CPU path only. A model with a resident backend keeps answering from the device (PromptHidden and Generate prefer it), whose own prefill is
    faster than a CPU one; copying the DEVICE's recurrent state is a per-backend change that does not exist, so on a resident model every method here
    falls back to the one-prompt-at-a-time path and says nothing different.
  - The Qwen3.5 batched family (qwen35BatchN) with the plain f32 or int8 KV: no sliding-window ring, no m-RoPE, no capture hooks, no layer pager. Anything
    else takes the fallback.
  - A shared prefix is found, not assumed: the longest common token prefix of the prompts actually built (the ids, after tokenization), so a template that
    puts the question before the state (bare-v1 puts "[kind]" first) shares only what it really shares.
```

## fakeQuantScheme

Moved from `decoder/fakequant.go` (the comment above `fakeQuantScheme`) on 2026-10-10.

```text
fakeQuantScheme (GOINFER_FAKEQUANT=affine|sym|symmse) is a DIAGNOSTIC: when set, the
int4 quantization path (quantizeWM's quantInt4 case) instead quantize→dequantizes each
weight with the named 4-bit scheme and stores the result at int8 (W8A8). This isolates
the 4-bit WEIGHT-scheme quality without any packed-nibble format, dequant routine, W4A8
kernel, or .giw change — the int8 path is essentially transparent (0.99995 reconstruction),
so the fake-int4 error survives to the forward. Lands ~26 GB (fits a 64 GB box). "sym" is
the CONTROL: it must reproduce the real-int4 garbage, or the harness is lying — asserted
bit-for-bit by TestFakeQuantSymMatchesRuntimeInt4. Not for prod.

It built the int4-quality matrix in docs/task-gemma4-moe.md (sym/affine × int8/f32 act ×
full/experts-only). The env is read ONCE here at package load; default (unset) is a strict
no-op → every int4 consumer stays bit-identical (TestFakeQuantOffBitIdentical).
```

## fakeQuantF32Act

Moved from `decoder/fakequant.go` (the comment above `fakeQuantF32Act`) on 2026-10-10.

```text
fakeQuantF32Act (GOINFER_FAKEQUANT_ACT=f32) stores the fake-quant reconstruction at
weight-only int8 (Q8, f32 activations) instead of W8A8 (int8 activations) — so the probe
can test a scheme against BOTH activation precisions. MLX's coherent config is affine
weights + f16 activations, so affine+f32-act is the direct comparison.

fakeQuantExpertsOnly (GOINFER_FAKEQUANT_EXPERTS=1) restricts the fake-quant to the MoE
experts (streamExperts) — load at int8int8 so attention/dense/embed stay int8 and only
the experts take the 4-bit scheme. Tests whether the experts alone tolerate 4-bit when
the router's input hidden state is kept accurate (the ~14.5 GB config).
```

## buildGptOssWeights.layout

Moved from `decoder/gptoss_safetensors.go` (the comment above `buildGptOssWeights.layout`) on 2026-10-10.

```text
TWO LAYOUT FACTS DRIVE THIS FILE, and both were established by diffing a dequantized
expert against the same weight read through the already-T3-validated GGUF path — not
from the format's documentation, which does not say either one:

 1. MXFP4 nibbles are SEQUENTIAL here (byte j holds elements 2j and 2j+1), where GGML
    packs j and j+16. Measured: cosine 1.000000 sequential vs 0.081 GGML.
 2. gate_up_proj is INTERLEAVED, not concatenated: row 2k is gate row k and row 2k+1 is
    UP row k. Measured the same way (st row 1 == gguf up row 0, cosine 1.000000).

Either mistake yields correct shapes, finite values, plausible magnitudes and entirely
wrong weights, so both are asserted by the parity gate rather than trusted.
```

## PromptHiddenAll

Moved from `decoder/prompt_hidden_all.go` (the comment above `PromptHiddenAll`) on 2026-10-10.

```text
PromptHiddenAll returns the final-norm hidden state at EVERY position of prompt: HF's last_hidden_state[0, :] (which is
output_hidden_states[-1]), K rows of HiddenDim. It is what a head that reads all positions consumes (D11 of
docs/tasks/task-constrained-confidence.md, Route C: Clef's joint schema head, docs/measurements/decisions-d10-clef-2026-10-02.md); D2's
PromptHidden returns only the last row.

A resident backend that implements ResidentResidualAll answers first (CUDA does; D11's follow-up): the whole prompt runs on the device, the residual stream of
every row comes back, and the final norm is applied HERE on the host in f32, so the rows are not the int8-requantized vectors ResidentHiddenLast returns. Its numerics are
the device kernels', not the CPU reference's (the backend modules bound the difference); a decline falls through to the CPU exactly as if there were no resident, and a
cancellation returns. Otherwise it takes the same CPU paths PromptHidden takes, in the same order, so the last row is the row PromptHidden returns on a model with no resident:
```

## cpuFusedGateUp

Moved from `decoder/cpu_gateup_fused.go` (the comment above `cpuFusedGateUp`) on 2026-10-10.

```text
It removes two costs the per-component roofline attributes to the MLP block: a fork/join (~one
goroutine-wake stagger, aikit S-02) and the SwiGLU's 3.65 ms/token of scalar float64 exp on the 1.5B,
which ran serial because fanning it out separately loses (cpu-decode-attribution-2026-09-22-linux.md).
Folded into the matmul's own barrier it costs nothing extra to run in parallel.
```

## giwverify

Moved from `decoder/giwverify.go` (the comment above `giwverify`) on 2026-10-10.

```text
A .giw's trailing CRC-32 covers every byte of the weight payload, and checking it forces a
read of the whole mapped file. On a streamed load that is the entire load time and the
entire page-cache footprint: measured 2026-09-23, a 22 GB M35 .giw read over a ~10 MB/s link
spent 27-28 minutes in that one check (docs/measurements/moe-pager-m35-smb-2026-09-23.md), and
on a 5.17 GB local .giw the CRC was ~100% of LoadSerializedWeights' cost.
```

## DefaultDecodeParallelThreshold

Moved from `decoder/tune.go` (the comment above `DefaultDecodeParallelThreshold`) on 2026-10-10.

```text
DefaultDecodeParallelThreshold is the matmul parallelism crossover (in MACs) for the
int8 (W8A8) decode path: parallelize the per-token weight matmuls, leaving trivially
small ops serial. Measured optimum on Apple M1 Pro / Qwen2.5-Coder-0.5B (~68 tok/s vs
~51 serial), independently re-confirmed on Ryzen 7 3700X (0.5B 22→36, 1.5B 10.5→15.1
tok/s vs aikit's conservative 16.78M default).

It is applied PER-WORKSPACE, automatically: newDecodeScratch sets it on the decode
Workspace (matmulInto path) and matmul()'s free W8A8 branch sets it per-call. So every
decode stream — library Load, serve, tests, future entry points (sidecar, c-archive
FFI) — gets it without any startup call, and it is race-free across concurrent streams
(unlike a process global). The int4 (W4A8) path has its OWN crossover, int4ParThreshold
= 1<<20 (weightmat.go): the two values genuinely differ — measured separately, on
different kernels/models, and a same-context sweep confirmed 300K is slightly better for
small int8 while 1<<20 is the int4/gemma4 optimum. Unify only after a proper joint sweep.
Hardware-specific (shifts with core count / memory latency); the M1 Pro is Phase 5's rig.

RE-CONFIRMED on M1 Pro against the gemma4-26b decode shapes (BenchmarkInt8ParThresholdSweep):
300K is below all four (down 1.98M … attn 11.5M), so it parallelizes every one — 1.36×–2.18×
vs serial — and thr=0 shows no over-parallelize penalty, so it stays comfortably optimal.
```

## deltaNetCoreN

Moved from `decoder/deltanet_n.go` (the comment above `deltaNetCoreN`) on 2026-10-10.

```text
P26c (docs/queue-performance.md). The batched Qwen3.5 forward (runLayersQwen35N) runs every projection as one matmul over the prompt, but then called
deltaNetCore once per token on ONE thread: on Qwen3.5-0.8B at int4 that was ~6.8 s of an 11 s, 684-token prefill (a CPU profile: deltaNetRecurrence
~4.9 s, the conv's SiLU ~2 s), while the matmuls it sits between ran on all 16 threads in about 2 s.

deltaNetCoreN is deltaNetCore over K consecutive rows with the independent parts fanned out, and NOTHING reordered inside any one element:
  - the depthwise conv (+SiLU) of row i reads only the mixed inputs of rows i-K+1..i, all known up front, so rows run in parallel;
  - the recurrence is sequential over TOKENS but independent across value HEADS (each owns its [head_k_dim, head_v_dim] block of the state), so heads run in
    parallel, each walking its K tokens in order with the very loop body deltaNetRecurrence runs;
  - the gated RMSNorm is per token and per head, so rows run in parallel.

Every float operation of every output element happens in the same order as in the per-token loop, so the result is BIT-IDENTICAL to it, state included
(TestDeltaNetCoreN_bitIdentical holds it to exact bit equality). It is used only when no capture hook or timing is on, which keep the per-token loop.
```

## attnGroupedKernels

Moved from `decoder/cpu_tuning_other.go` (the comment above `attnGroupedKernels`) on 2026-10-10.

```text
attnGroupedKernels: aikit ships the grouped acc64 attention kernels as NEON only
(linalg/attn_acc64_group_other.go returns 0 blocks — the whole group falls to the Go path), so
on every other architecture the "grouped" path is a pure-Go loop that measured 1.18× (depth
128) to 2.86× (depth 4096) SLOWER than the per-head path on the 1.5B's 6-heads-per-KV
geometry. Off until a port exists.
```

## activationFanoutEnabled

Moved from `decoder/cpu_tuning_other.go` (the comment above `activationFanoutEnabled`) on 2026-10-10.

```text
activationFanoutEnabled: the 6-goroutine activation fan-out costs ~3× what the serial loop
does per element on this box (goroutine wake stagger dwarfs 9-19k scalar silu calls) — 1.5B
13.5 → 3.9 ms/token, 7B 27.9 → 8.0. Serial.
```

## fusedGateUpDefault

Moved from `decoder/cpu_tuning_other.go` (the comment above `fusedGateUpDefault`) on 2026-10-10.

```text
fusedGateUpDefault: the fused gate+up+SwiGLU fork/join (cpu_gateup_fused.go) — one barrier per
layer instead of two, with the activation folded into it. Paired ABBA on the Ryzen 7 3700X,
bit-identical: 1.5B 1.066×, 0.5B 1.113×, 7B 1.029× against the unfused path
(docs/measurements/cpu-decode-roofline-2026-09-23.md). On.
```

## w4a8BatchDefault

Moved from `decoder/cpu_tuning_other.go` (the comment above `w4a8BatchDefault`) on 2026-10-10.

```text
w4a8BatchDefault: R-06's one fork/join for q/k/v (weightmat.go). Bit-identical; paired ABBA on top of the
fused gate+up, Ryzen 7 3700X: 0.5B 1.016×, 1.5B 1.030×, 7B 1.018× (docs/tasks/task-cpu-decode-peer-gap-2026-09.md,
L2; owner decision 2026-09-27). On.
```

## HostRAMBytes.linux

Moved from `decoder/hostram_linux.go` (the comment above `HostRAMBytes.linux`) on 2026-10-10.

```text
HostRAMBytes is this machine's physical RAM, or 0 when it cannot be determined — and 0 is a
real answer that every caller must treat as "proceed", never as "no memory".

Read from /proc/meminfo rather than through a syscall wrapper so the pure-Go root module stays
free of golang.org/x/sys (audit M-19 keeps the root dependency graph minimal; the Metal module,
which already depends on it, uses unix.SysctlUint64 for the same figure).

KNOWN LIMIT: MemTotal is the HOST's memory. Inside a container with a cgroup memory limit this
over-reports, so the guard can let a doomed load through there — the safe direction, and the
direction the whole guard errs in. Reading memory.max is a follow-on, not this pass.
```

## HostRAMAvailableBytes.linux

Moved from `decoder/hostram_linux.go` (the comment above `HostRAMAvailableBytes.linux`) on 2026-10-10.

```text
HostRAMAvailableBytes is this machine's CURRENTLY AVAILABLE memory — what the kernel estimates
can be allocated by a new process without swapping (free pages plus reclaimable cache) — or 0
when it cannot be determined. Unlike HostRAMBytes, this is NOT cached: it changes continuously
as other processes run, which is the entire reason it exists (R13-follow-on,
docs/measurements/... the Mac re-run that found this). HostRAMBytes reports total physical RAM,
which never changes and is safe to read once; MemAvailable is the opposite by nature.

KNOWN LIMIT: same container caveat as HostRAMBytes — /proc/meminfo reports the HOST's memory
inside a container with a cgroup limit, over-reporting availability there. The safe direction.
```

## capture

Moved from `decoder/capture.go` (the comment above `capture`) on 2026-10-10.

```text
The hidden-state capture seam, extracted so the families with their own runLayers
can offer it without each reimplementing the copy-on-match loop.

WHY A HELPER RATHER THAN SEVEN COPIES. The generic path (runLayersFromEmbed) grew this
inline for EAGLE-3 (05; that head was removed on 2026-09-24, be9aeea8). P10's block drafters need the same residuals from families that
never routed through it — qwen3_5_moe, gemma4, gpt-oss are the three whose targets we hold
locally with a licensed drafter. Copying seven lines four times is how the two halves drift
apart: the generic one copies AFTER the MLP add, and a copy placed a few lines earlier would
capture a residual that is off by one sublayer while still looking plausible in every test
that only checks shape. One definition, called at the tail of each loop body.
```

## ActivationQuantHazard

Moved from `decoder/actquant_hazard.go` (the comment above `ActivationQuantHazard`) on 2026-10-10.

```text
Measured for Phi-3-mini (queue-engineering.md H2, 2026-09-25): at a filler position the down_proj
input's max/rms is ~80-90 in most layers, so a per-row scale rounds 99.9% of the row to zero;
int4 and int8int8 fall to logit cosine ~0 vs f32 by position 16 while weight-only int8 holds
>= 0.95 and f32 matches HF exactly. Keyed on model type, so Phi-4 (also "phi3") is covered
without having been measured. This is a guard until per-group activation scales land in the
kernels; it is not the fix.
```

## excludeFromFork

Moved from `decoder/forkinherit_darwin.go` (the comment above `excludeFromFork`) on 2026-10-10.

```text
Why a read-only weight mapping needs this (docs/measurements/m26-alias-fork-collapse-2026-09-24.md):
a .giw is mapped PROT_READ|MAP_PRIVATE. Once anything wires a page of it through IOKit — Metal does,
for every page a command buffer reads through a no-copy buffer — the kernel copy-on-write-copies the
wired page (vm_object_iopl_request faults with write intent) and marks the mapping's object
true_share. From then on vm_map_fork cannot share the entry lazily: vm_object_copy_delayed refuses
wired pages and the kernel copies the WHOLE entry, page by page, on the forking thread. Every
os/exec in the process is a fork on darwin; on a 15 GB mapping on a 16 GB Mac the first one after
the first GPU request paged the whole server out. Measured with metal/alias_forkprobe_test.go: a
fork+exec went 2.8 ms → 1,092 ms after one GPU touch of a 16 MiB window of a 2 GiB mapping, and
stayed ~3.8 ms with the mapping VM_INHERIT_NONE. A child never needs the weights (the only children
are exec'd helpers, which discard the address space immediately), so leaving them out is free.
```

## mapGIW

Moved from `decoder/giwmap_darwin.go` (the comment above `mapGIW`) on 2026-10-10.

```text
Why shared, and why only here (docs/measurements/s6-alias-2026-09-24.md; the probe is
metal/alias_sharedprobe_test.go): a Metal no-copy buffer over a window of the mapping is wired by IOKit
with WRITE intent. Over a PRIVATE mapping that turns every page the GPU reads into a wired anonymous
copy-on-write copy (+15,131 COW faults for a 16,384-page window, persisting for the mapping's life) — so
"aliasing" the weights saved no memory — and marks the object true_share, after which any fork() copies
the whole mapping (the M26 collapse, docs/measurements/m26-alias-fork-collapse-2026-09-24.md). Over a
SHARED read-only mapping the GPU reads the file's own page-cache pages: 0 COW faults, correct values,
fork+exec 4-5 ms while wired. The CPU reads either mapping identically.
```

## support_table

Moved from `decoder/support_table.go` (the comment above `support_table`) on 2026-10-10.

```text
S8 (docs/tasks/task-multimodal-support-2026-10.md, "S8, plan and gates"): what the README and docs/multimodal.md
support table's decoder cells are computed from. Each cell answers "after an image, does this family's turn decode on
this backend's resident?", from the declarations the runtime itself admits on, so the table cannot say more than the
code does (it did, once, by hand: Qwen3.5+ on Metal).
```

## SwapUsedBytes.darwin

Moved from `decoder/memwatch_darwin.go` (the comment above `SwapUsedBytes.darwin`) on 2026-10-10.

```text
A bare sysctl(2) through the stdlib, NOT `exec sysctl -n vm.swapusage`: the serving swap guard calls
this every 2 s, and every exec is a fork() — which, while a GPU backend has wired a page of the .giw
mapping, used to copy the whole mapping (docs/measurements/m26-alias-fork-collapse-2026-09-24.md;
Load now also marks the mapping VM_INHERIT_NONE, so this is the second of two independent fixes).
The value is `struct xsw_usage` (sys/sysctl.h): u64 xsu_total, u64 xsu_avail, u64 xsu_used,
u32 xsu_pagesize, boolean_t xsu_encrypted — 32 bytes, little-endian on every darwin target.
syscall.Sysctl drops only a single trailing NUL (the high byte of xsu_encrypted, always 0), so the
struct arrives intact through byte 24. An earlier comment here claimed it "truncates at the first
NUL" and that there was therefore no bare-syscall alternative; neither is true.
```

## DecisionHeadQuant

Moved from `decoder/decisions.go` (the comment above `DecisionHeadQuant`) on 2026-10-10.

```text
DecisionHeadQuant is the weight precision a model with a trained decision head loads at when no quant was chosen
(`goinfer-chat decide --head`, and `goinfer-serve`'s `head=`); an explicit `--quant` or `quant=` still wins.

The owner decided it on 2026-10-01 from D6b (docs/measurements/decisions-d6b-2026-09/results.md), where Route B on
JEV-9B was graded against the transformers f32 reference on 150 items. f32 was exact; int8int8 read mean KL 0.009 and
top-1 agreement 92.7%; int4 read 0.030 and 94.0%; and no arm was resolvably worse calibrated than the reference. The
pre-registered rule names int8 as the decision models' fallback, and int8int8 fits where f32 (about 36 GB for
JEV-9B) does not. The capability matrix renders this constant, so the two cannot disagree.
```

## dropped.actquant_hazard.16

Moved from `decoder/actquant_hazard.go` (a comment block rewritten to the rule; it carried tracker ids, dates or a provenance note) on 2026-10-10.

```text
Measured for Phi-3-mini (queue-engineering.md H2, 2026-09-25): at a filler position the down_proj
input's max/rms is ~80-90 in most layers, so a per-row scale rounds 99.9% of the row to zero;
int4 and int8int8 fall to logit cosine ~0 vs f32 by position 16 while weight-only int8 holds
>= 0.95 and f32 matches HF exactly. Keyed on model type, so Phi-4 (also "phi3") is covered
without having been measured. This is a guard until per-group activation scales land in the
kernels; it is not the fix.
```

## dropped.cpu_batch.167

Moved from `decoder/cpu_batch.go` (a comment block rewritten to the rule; it carried tracker ids, dates or a provenance note) on 2026-10-10.

```text
decodeMultiStep is MC3c step 2's batched step (promoted from MC2's test-only prototype): ONE forward carrying B
independent sequences, each with its own KV cache at its own position, one token each. The projections, o-proj, MLP
and LM head run as M = B matmuls through the kernels forwardN uses (bit-identical to M = 1 row for row, by forwardN's
own contract). Attention runs per sequence over its own cache, exactly as causalAttention's default (f32 KV,
append-forever) case does at decode. Every row's logits are bit-identical to that sequence's own m.forward
(TestMC2_decodeMultiStepBitIdentical, TestCPUBatch_everyEligibleFixtureBitIdentical).
```

## dropped.cpu_gateup_fused.28

Moved from `decoder/cpu_gateup_fused.go` (a comment block rewritten to the rule; it carried tracker ids, dates or a provenance note) on 2026-10-10.

```text
Default per architecture (fusedGateUpDefault, cpu_tuning_{arm64,other}.go): measured on the amd64
Ryzen 7 3700X only, so arm64 stays off until the Mac measures it. GOINFER_CPU_FUSED_GATEUP=0 forces the
unfused path (an A/B handle and an escape hatch), =1 forces it on.
```

## dropped.cpu_gateup_fused.97

Moved from `decoder/cpu_gateup_fused.go` (a comment block rewritten to the rule; it carried tracker ids, dates or a provenance note) on 2026-10-10.

```text
h quantized once for every worker's gate and up chunk (R-13): each MatmulBTW4A8F16Into re-quantized all of h,
2w times a layer. The workers' workspaces carry the weight's activation group, or none (then aikit's process
default applies), so the block is quantized with the group they resolve to; the kernels only read it.
w4a8PreOff (tests only) restores a quantization per call, as it was.
```

## dropped.doc.50

Moved from `decoder/doc.go` (a comment block rewritten to the rule; it carried tracker ids, dates or a provenance note) on 2026-10-10.

```text
errNotImplemented is the sentinel for a path that cannot run — today only
the "weights not loaded" guard in runLayers.
```

## dropped.embed.9

Moved from `decoder/embed.go` (a comment block rewritten to the rule; it carried tracker ids, dates or a provenance note) on 2026-10-10.

```text
Decoder-as-embedder seam (docs/completed/task-decoder-as-embedder.md).

qwen3-embedding and embeddinggemma are causal decoders used as EMBEDDERS: the sentence-
transformers stack around them is just `Transformer → Pooling → Normalize`. The only thing the
decoder itself must expose for that is the transformer's output — the final hidden state — which
until now was unreachable: forward() runs the stack and immediately consumes the hidden state
through the LM head (logitsFromHidden normalizes it IN PLACE, then projects to vocab), and
ForwardCapture hands back per-LAYER residuals, which are pre-final-norm and so are NOT what
```

## dropped.embed.176

Moved from `decoder/embed.go` (a comment block rewritten to the rule; it carried tracker ids, dates or a provenance note) on 2026-10-10.

```text
A resident backend answers first, through the same seam HiddenLast uses (G4): the whole prompt on the device,
stopping at the final norm. A decline (an arch the backend's headless forward does not cover, a paged MoE, a
cap) falls through to the CPU exactly as if there were no resident; a cancellation returns. The resident's
prefix-reuse record is forgotten either way, since the prompt overwrote its KV.
```

## dropped.fakequant.45

Moved from `decoder/fakequant.go` (a comment block rewritten to the rule; it carried tracker ids, dates or a provenance note) on 2026-10-10.

```text
fakeQuantPerRow (set by the §7 Phase 0b test; its GOINFER_FAKEQUANT_PERROW env read was retired
2026-09-24 — §7's per-row fork is decided, docs/tasks/task-env-config-2026-09.md phase 6) forces the group to the FULL row — one scale per output row, the §7 per-row/IMMA granularity — so
the probe can measure per-row vs per-group forward quality (Phase 0b) without a new kernel. Default
off ⇒ the shipped 32-elem grouping, so the fakequant-off invariant is untouched.
```

## dropped.fidelity_testhook.3

Moved from `decoder/fidelity_testhook.go` (a comment block rewritten to the rule; it carried tracker ids, dates or a provenance note) on 2026-10-10.

```text
Code added for docs/completed/task-prefill-gap.md §3's fidelity gate (a backend's fast/batched prefill
vs its own exact/sequential path) and docs/tasks/task-peer-benchmarks.md §4's fidelity column
(goinfer vs a peer engine) -- both want the same teacher-forced top-1 agreement and KL
divergence scorer, so it is written once here rather than twice. Test-only hook (B-08): these
gate correctness/quality, not production inference, so they stay off the public API surface.
```

## dropped.fidelity_testhook.80

Moved from `decoder/fidelity_testhook.go` (a comment block rewritten to the rule; it carried tracker ids, dates or a provenance note) on 2026-10-10.

```text
ResidentAdapterLayersForTest exposes residentAdapterLayers' conversion of a loaded compute-time
adapter to the exported per-layer shape a resident backend's SetAdapter consumes — the same
conversion generateInto calls in production, so a backend's own parity test can drive
ResidentAdapter.SetAdapter directly (bypassing Session/generateInto's session-cache plumbing to
isolate the backend's numerics) with exactly the deltas a real request would bind.
```

## dropped.fidelity_testhook.116

Moved from `decoder/fidelity_testhook.go` (a comment block rewritten to the rule; it carried tracker ids, dates or a provenance note) on 2026-10-10.

```text
PrefillLogitsGemma4VLForTest exposes prefillLogitsGemma4VL — Gemma 4's sequential,
causal-only CPU prefill GenerateGemma4VL drives — same reason as the two hooks above:
a cross-package real-checkpoint gate needs the per-step logits, not GenerateGemma4VL's
channel-only sampled-token API.
```

## dropped.fidelity_testhook.124

Moved from `decoder/fidelity_testhook.go` (a comment block rewritten to the rule; it carried tracker ids, dates or a provenance note) on 2026-10-10.

```text
PrefillLogitsGemma4VLBidirectionalForTest exposes prefillLogitsGemma4VLBidirectional —
Gemma 4's batched, blockwise-masked CPU prefill for 26B-A4B/31B-class
(use_bidirectional_attention: "vision") checkpoints — the twin of
PrefillLogitsGemma4VLForTest above, so a cross-package real-hardware gate (the GPU-resident
decode bridge, decoder/generate_gemma4_vl.go) can build the exact CPU-computed KVCache
GenerateGemma4VL's resident branch uploads, without going through its channel-only API.
```

## dropped.fidelity_testhook.134

Moved from `decoder/fidelity_testhook.go` (a comment block rewritten to the rule; it carried tracker ids, dates or a provenance note) on 2026-10-10.

```text
MRopePositionsForTest exposes mropePositions — Qwen2.5-VL's per-token (t,h,w) rotary position
triples from the image grid — so a cross-package test can build the same cache.mropeDelta the
production GenerateQwenVL path computes, for the same reason as PrefillLogitsQwenVLForTest.
```

## dropped.fidelity_testhook.141

Moved from `decoder/fidelity_testhook.go` (a comment block rewritten to the rule; it carried tracker ids, dates or a provenance note) on 2026-10-10.

```text
MRopeDeltaForTest exposes a KVCache's m-RoPE decode-position delta (rope.go's mropeDelta —
scalar decode past the prefill rotates at seqPos+delta), set by prefillLogitsQwenVL. A
cross-package resident-decode test needs this to compute the same ropePos GenerateQwenVL's
production decode loop does (gap 0, docs/multimodal.md).
```

## dropped.fidelity_testhook.147

Moved from `decoder/fidelity_testhook.go` (a comment block rewritten to the rule; it carried tracker ids, dates or a provenance note) on 2026-10-10.

```text
NearTieHardFailPct is the bar NearTieArgmaxForTest hard-fails at -- the same 3% every existing
near-tie gate in this tree already uses inline (cuda/realforward_test.go's argmaxF comparison,
gpu/kv_i8_parity_test.go), named here so a new gate cites the rule instead of retyping the
literal.
```

## dropped.fidelity_testhook.214

Moved from `decoder/fidelity_testhook.go` (a comment block rewritten to the rule; it carried tracker ids, dates or a provenance note) on 2026-10-10.

```text
KLDivergenceForTest computes KL(p || q) in nats between two logit vectors, after converting
each to a probability distribution the same way sampling does (softmaxStable, temperature 1).
p is the reference/exact distribution and q the candidate/approximate one, so the result reads
as "how much information is lost approximating p with q" -- the §3 gate's "reported, not
gating" KL-vs-exact figure. Terms where p is ~0 are skipped rather than evaluated: the limit of
p*log(p/q) as p->0 is 0 regardless of q, and evaluating it risks NaN from log(0).
```

## dropped.fidelity_testhook.301

Moved from `decoder/fidelity_testhook.go` (a comment block rewritten to the rule; it carried tracker ids, dates or a provenance note) on 2026-10-10.

```text
PrefillGatePromptSetFor returns one NAMED set's snapshot files regardless of
GOINFER_PREFILL_GATE_PROMPTS — for a caller that needs a specific set explicitly (e.g.
re-scoring set A's stored results alongside a set-B decision run, docs/completed/task-prefill-gap.md §4 L1)
rather than "whichever set the environment currently selects". label other than "b" means "a".
```

## dropped.fidelity_testhook.319

Moved from `decoder/fidelity_testhook.go` (a comment block rewritten to the rule; it carried tracker ids, dates or a provenance note) on 2026-10-10.

```text
PrefillGateProseIDsForTest reads f, encodes it with tk, and returns at least minTokens ids
(repeating the same real content if one file is somehow too short for a future larger K, rather
than padding with filler — a repeated real paragraph is still content-dependent, unlike
scripts/prompts.json's "the the the").
```

## dropped.fidelity_testhook.344

Moved from `decoder/fidelity_testhook.go` (a comment block rewritten to the rule; it carried tracker ids, dates or a provenance note) on 2026-10-10.

```text
WritePrefillReferenceForTest serializes docs/completed/task-prefill-gap.md §3.1's CPU f32-activation
reference for a later cross-process read (ReadPrefillReferenceForTest) — Phase A
(decoder/prefill_ref_gen_test.go, its own process, CPU only) writes these; Phase B
(metal/prefill_gate_ref_test.go) reads them back to score both Metal arms against a reference
neither of them is. Layout, all little-endian: int32 vocab, int32 continuationN, seedLogits
[vocab]float32, refTokens [continuationN]int32, then continuationN rows of [vocab]float32
(refLogits). This is a private, single-machine, single-session scratch format — not versioned
or exported for reuse beyond this pair, which is why it carries no header/magic beyond its two
size fields.
```

## dropped.fidelity_testhook.428

Moved from `decoder/fidelity_testhook.go` (a comment block rewritten to the rule; it carried tracker ids, dates or a provenance note) on 2026-10-10.

```text
W8F3Ref is F3′'s CPU f32 reference (docs/tasks/task-metal-int8-2026-10.md, "F3′"), written by
TestW8F3Reference_write on a machine whose fit guard admits the f32 model (nobara for the 1.5B) and read by
metal's TestW8Native_F3amended_closerToF32 through GOINFER_W8_F3_REF_IN: the 8 prompts, the f32 model's greedy
continuation (the tokens every arm is teacher-forced on), and the f32 logits at every position.
```

## dropped.fusedattn.100

Moved from `decoder/fusedattn.go` (a comment block rewritten to the rule; it carried tracker ids, dates or a provenance note) on 2026-10-10.

```text
fits reports whether this scratch is already large enough for the requested shape, so a pool slot
can be REUSED across calls instead of reallocated per token (M-03). Every buffer is used as a
prefix slice, so larger is fine.
```

## dropped.gguf_qwen35.34

Moved from `decoder/gguf_qwen35.go` (a comment block rewritten to the rule; it carried tracker ids, dates or a provenance note) on 2026-10-10.

```text
Bound before the per-layer append loop below: a hostile block_count would grow
LayerTypes unboundedly (a fatal OOM recover can't catch) before validateGGUFDims
runs (M16).
```

## dropped.gptoss_safetensors.38

Moved from `decoder/gptoss_safetensors.go` (a comment block rewritten to the rule; it carried tracker ids, dates or a provenance note) on 2026-10-10.

```text
qw is quantizeWM/quantizeWMSkipRow4 (M-07, audit-metal-2026-09-12.md) for every layer
projection below — Embed/LMHead deliberately stay on plain quantizeWM. streamQ is
streamQuantized's own skip-aware twin, for the MoE experts (streamed row by row — see this
file's own MEMORY note on why they never materialize as f32).
```

## dropped.hostram_other.5

Moved from `decoder/hostram_other.go` (a comment block rewritten to the rule; it carried tracker ids, dates or a provenance note) on 2026-10-10.

```text
HostRAMBytes returns 0 — unknown — on every platform without a probe here (the BSDs and the rest).
The fit guard treats 0 as "proceed", so those platforms behave exactly as they did before the
guard existed rather than getting a wrong number: say "unknown", never guess. Windows has its own
probe (hostram_windows.go) since 2026-10-03.
```

## dropped.image_spans.5

Moved from `decoder/image_spans.go` (a comment block rewritten to the rule; it carried tracker ids, dates or a provenance note) on 2026-10-10.

```text
ImageSpan is one image's placeholder run in a prompt (S11, docs/tasks/task-multimodal-support-2026-10.md): ids[Pos,
Pos+Len) are replaced by that image's projected features. Hash identifies the image for resident prefix reuse (P9a);
0 makes no claim and never matches (M-06).
```

## dropped.internlm2.46

Moved from `decoder/internlm2.go` (a comment block rewritten to the rule; it carried tracker ids, dates or a provenance note) on 2026-10-10.

```text
qw is quantizeWM/quantizeWMSkipRow4 (M-07, audit-metal-2026-09-12.md) for every layer
projection below — Embed/LMHead deliberately stay on plain quantizeWM. Named qw, not q,
because q is already the query-tensor local below.
```

## dropped.layerpaging.42

Moved from `decoder/layerpaging.go` (a comment block rewritten to the rule; it carried tracker ids, dates or a provenance note) on 2026-10-10.

```text
mu guards the mutable paging state below (audit C-30). The pager lives on *Model, shared across
every Generate; StreamWeights explicitly supports concurrent streams, and each drives the layer
loop, so two streams race on state[]/counters without this. The mapping (and thus the WILLNEED/
DONTNEED hints) is genuinely model-level, so one guarded view is correct — the madvise hints are
advisory, so a stream re-faults a page another stream released; only the state writes need the lock.
```

## dropped.layerpaging.108

Moved from `decoder/layerpaging.go` (a comment block rewritten to the rule; it carried tracker ids, dates or a provenance note) on 2026-10-10.

```text
enterLayer prefetches l+ahead before releasing l-window, so the resident
set at steady state is window+ahead layers (P-12), not window — subtract
ahead here so window+ahead layers' bytes actually fit budget.
```

## dropped.optiongrid.137

Moved from `decoder/optiongrid.go` (a comment block rewritten to the rule; it carried tracker ids, dates or a provenance note) on 2026-10-10.

```text
Tested on CUDA, the one resident with per-group activation scales. Metal and WebGPU decline the load to
the CPU (actGroupResidentDecline, since 2026-10-08; before, Metal ran it resident at per-vector scales).
```

## dropped.rmsnorm.233

Moved from `decoder/rmsnorm.go` (a comment block rewritten to the rule; it carried tracker ids, dates or a provenance note) on 2026-10-10.

```text
geluTanh is the tanh-approximate GELU Gemma's GeGLU MLP uses
("gelu_pytorch_tanh"). Provided here so mlp.go (stub) and tests have the
activation ready.
```

## dropped.routercapture.29

Moved from `decoder/routercapture.go` (a comment block rewritten to the rule; it carried tracker ids, dates or a provenance note) on 2026-10-10.

```text
routerRnBuf records, per MoE decision (same order/index as routerCaptureBuf), a COPY of the
finalized router input rn = (weightless-norm(h) · routerScale · hidden^-0.5) — the exact f32
vector that feeds routerProj. It exists so a CUDA resident-router unit test can replay identical
inputs through the device selection kernels and gate resident idx[] against the CPU idx[]
(routerCaptureBuf), isolating a ROUTING FLIP from any expert-GEMV numeric difference — the
"router first" discipline. Captured only when routerCapture is on; observe-only, byte-identical
with the env unset.
```

## dropped.routercapture.38

Moved from `decoder/routercapture.go` (a comment block rewritten to the rule; it carried tracker ids, dates or a provenance note) on 2026-10-10.

```text
routerMarginBuf records, per MoE decision (same order/index as routerCaptureBuf), the top-k
BOUNDARY MARGIN: the smallest selected expert's softmax prob minus the largest REJECTED
expert's prob. This is the quantity that decides whether a small quant perturbation flips the
top-k — the MoE-specific failure mode. A resident-gate-ready fixture wants this margin to stay
well above the per-decision quant perturbation on every decision, not merely to AGREE on one
int4-vs-f32 pair (agreement can be luck; a wide margin is robustness). Captured only when
routerCapture is on; observe-only, so the forward stays byte-identical with the env unset.
```

## dropped.routercapture.76

Moved from `decoder/routercapture.go` (a comment block rewritten to the rule; it carried tracker ids, dates or a provenance note) on 2026-10-10.

```text
routerCaptureMax bounds each buffer. Generous for the diagnostic's actual use — a
teacher-forced pass of a few hundred tokens over 30 layers — and small enough that a
forgotten env var on a serving process costs bounded memory instead of the process.
```

## dropped.stream_dir.39

Moved from `decoder/stream_dir.go` (a comment block rewritten to the rule; it carried tracker ids, dates or a provenance note) on 2026-10-10.

```text
Load writes the EOS ids generation_config.json adds back into the config a bundle serializes (decoder/model.go);
the stream writes the config in its head, before any layer, so it resolves them first (StreamTranscodeGGUF's M-04).
```

## dropped.swapwatch.35

Moved from `decoder/swapwatch.go` (a comment block rewritten to the rule; it carried tracker ids, dates or a provenance note) on 2026-10-10.

```text
PollInterval between samples. Default 2s (S3's registered interval).
```

## dropped.swapwatch.38

Moved from `decoder/swapwatch.go` (a comment block rewritten to the rule; it carried tracker ids, dates or a provenance note) on 2026-10-10.

```text
Threshold is how far swap-used may grow above the FIRST successful sample's baseline
before OnTrip fires. Default 512 MiB (S3's registered default). A machine already deep in
swap from something else unrelated has a baseline that includes it — this measures only
what grows AFTER the watch starts, i.e. what the watched process itself adds, not the
machine's pre-existing state.
```
