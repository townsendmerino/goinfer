
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
