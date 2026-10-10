# metal: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `metal`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

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
