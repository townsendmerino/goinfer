# decoder: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `decoder`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

## TestFitGuard_refusesBeforeAllocating

Moved from `decoder/fitguard_test.go` (the comment above `TestFitGuard_refusesBeforeAllocating`) on 2026-10-09.

```text
R3 gate (docs/measurements/cold-user-2026-09-06.md, scenario D). On v0.16.0 a 21 GB model on a
16 GB machine loaded without a word and drove the box +7,819 MB into swap in five seconds. The
bar for this test is therefore not "an error is returned" — it is that NOTHING WAS ALLOCATED
when the error was returned, and that the message names the flag that fixes it.

The RAM figure is injected, so the test exercises the 16 GB machine's arithmetic on any box.
```

## TestFitGuard_refusesBeforeAllocating.retry

Moved from `decoder/fitguard_test.go` (the comment above `TestFitGuard_refusesBeforeAllocating.retry`) on 2026-10-09.

```text
gptoss_tiny.gguf is gpt-oss — MoE, own-forward — so an automatic -stream-weights retry must
NOT be offered: that CPU path is the one docs/benchmarks.md "M35/M26 on the Mac" measured as
2h10min/zero completions on a real checkpoint. Both the sentinel (errors.Is) and the typed
field (errors.As) have to agree, since main.go's retry decision reads the field directly.
```

## TestFitGuard_pricesTheOnDiskGGUFDuringLoadNotJustFinalWeights

Moved from `decoder/fitguard_test.go` (the comment above `TestFitGuard_pricesTheOnDiskGGUFDuringLoadNotJustFinalWeights`) on 2026-10-09.

```text
TestFitGuard_pricesTheOnDiskGGUFDuringLoadNotJustFinalWeights gates the cold-user 2026-09-18
nobara-pc Scenario D defect (docs/measurements/cold-user-2026-09-18-nobara-pc.md): loading
gpt-oss-20b (12.11 GB MXFP4 GGUF) via `chat --backend cuda` (no special flags) or
`serve --backend cuda --moe-cache-experts` drove REAL, incremental swap growth on a machine
with 37+ GB RAM free, with zero warning printed first — while `chat fit`'s pre-flight estimate
(dense+experts+KV, ~13.3 GB) reported comfortably fitting. Reproduced directly on nobara-pc (the
same box) with a heap profile + /proc RSS sampling around a bare decoder.Load of the real
checkpoint (CUDA and --moe-cache-experts are red herrings here: gpt-oss declines CUDA residency
via the missing FeatAttnSink feature — decoder/registry.go's own gptOssArchitecture comment — so
both commands actually ran this exact CPU-resident .gguf load path): peak RSS reached ~24.5 GB,
matching weightBytes+fileSize (12.58+12.11=24.69 GB) to within 2%, not the ~12.58 GB the OLD
(pre-fix) estimator priced.

This test pins the fix on the tiny fixture without needing the 12 GB real checkpoint: read the
REAL weightBytes+kvBytes+srcFileBytes off fitCheckFor itself (not a hand-derived guess — the
fixture's real config resolves a non-zero kvBytes too, so reimplementing the arithmetic by hand
would silently under-count and prove nothing; caught by mutation-checking this test against the
pre-fix `need()`, which passed spuriously until this was fixed to read the real fields), pick a
RAM figure whose budget sits strictly between (weightBytes+kvBytes) and
(weightBytes+kvBytes+srcFileBytes), and confirm the guard now refuses on the strength of
srcFileBytes alone.
```

## TestFitGuard_pricesTheCUDAExpertCacheBuildPeak

Moved from `decoder/fitguard_test.go` (the comment above `TestFitGuard_pricesTheCUDAExpertCacheBuildPeak`) on 2026-10-09.

```text
TestFitGuard_pricesTheCUDAExpertCacheBuildPeak: --backend cuda --moe-cache-experts holds the
canonical weights, a packed copy of them and a pinned copy of the experts on the host at once
(measured on the real gpt-oss-20b: 39 GB peak for a 12 GB checkpoint). In the regime that
matters that exceeds the weights+KV+file total the CPU path is priced at, and it does not stack
with the file term (the source is unmapped before the build), so need() takes the larger.

Two halves, because the tiny fixture cannot be in that regime (its KV at the full window dwarfs
its weights): the wiring is checked on the real GGUF, the arithmetic on realistic numbers.
```

## TestFitGuard_pricesAgainstAvailableNotTotalRAM

Moved from `decoder/fitguard_test.go` (the comment above `TestFitGuard_pricesAgainstAvailableNotTotalRAM`) on 2026-10-09.

```text
R13-follow-on (docs/measurements/cold-user-2026-09-07-macbook-arm64.md's SECOND live re-run):
the load-time guard must price against CURRENTLY AVAILABLE memory, not total RAM. Ample total
RAM with tight availability is exactly the live failure's shape — the load-time guard on the
real Mac reported a healthy-looking margin against total RAM while the machine was, in fact,
already out of room, and swap began within 15 seconds of load completing, before any request.
```

## TestFitCheck_unpinnedPricesKVAtTheModelsMaximum

Moved from `decoder/fitguard_test.go` (the comment above `TestFitCheck_unpinnedPricesKVAtTheModelsMaximum`) on 2026-10-09.

```text
R13 (docs/measurements/cold-user-2026-09-07-macbook-arm64.md): the guard priced weights and,
only if -ctx was pinned, KV — so an UNPINNED load that fits at idle can still swap the machine
on its first real request, because KV was priced at 0 regardless of how large the model's own
context window is. Driven with numbers shaped like the actual failure: a 7B-class model whose
weights alone fit comfortably, but whose KV at its full context window does not.
```

## TestFitGuard_unpinnedLoadAutoPinsASmallerContextRatherThanRefusing.fixture

Moved from `decoder/fitguard_test.go` (the comment above `TestFitGuard_unpinnedLoadAutoPinsASmallerContextRatherThanRefusing.fixture`) on 2026-10-09.

```text
MEASURED, not assumed: this fixture's own metadata gives ggufConfig a MaxPositions of 0
(context_length is not set the way this synthetic build's config parses it), so
fitCheckFor's "unknown ⇒ proceed" branch fires and this specific fixture cannot exercise
the auto-pin path at all — confirmed by direct inspection (kvBytesPerPosition=512,
MaxPositions=0), not by running this test and rationalizing a SKIP after the fact. The pure
arithmetic is fully covered by TestFitCheck_unpinnedPricesKVAtTheModelsMaximum above, which
does not depend on any fixture's real dimensions; this test exists to prove the OTHER half —
that a fixture with a real MaxPositions and comfortable weights does not regress into
spuriously capping or refusing — and to auto-upgrade to a real auto-pin assertion the day a
fixture with MaxPositions>0 is available at this path.
```

## TestFitEstimate_agreesWithResidentWeightBytes.band

Moved from `decoder/fitguard_test.go` (the comment above `TestFitEstimate_agreesWithResidentWeightBytes.band`) on 2026-10-09.

```text
TIGHT on purpose, and it was not always. The band started at 0.6-1.6 and passed on
linux/amd64 at 0.96 while darwin/arm64 sat at 0.53 — the arm64 W4A8 row4 repack
keeps a second buffer, so int4 costs about twice its encoding there and a
hand-derived constant was ~1.8x low on the one platform the guard exists for.
quantBytesPerElem now MEASURES through quantizeWM, so a wide band would only hide
the next such divergence. Some slack remains because the estimator prices every
tensor uniformly while the accountant reads the real backing slices.
```

## TestFitEstimate_safetensorsAgreesWithResidentWeightBytes

Moved from `decoder/fitguard_test.go` (the comment above `TestFitEstimate_safetensorsAgreesWithResidentWeightBytes`) on 2026-10-09.

```text
TestFitEstimate_safetensorsAgreesWithResidentWeightBytes is
TestFitEstimate_agreesWithResidentWeightBytes's safetensors twin (P9b, docs/multimodal.md):
proves estimateSafetensorsWeightBytes's shape-based, quant-priced estimate actually tracks a
real load's resident weight bytes, on a real (tracked, non-gitignored) safetensors checkpoint —
not just that it compiles or returns something positive. This is what closes the gap
fitCheckFor's OLD behavior left: a safetensors path returned a zero weight estimate
unconditionally, so fits() always reported true regardless of the machine's actual RAM.
```

## TestFitCheckFor_pricesSafetensorsNotJustGGUF

Moved from `decoder/fitguard_test.go` (the comment above `TestFitCheckFor_pricesSafetensorsNotJustGGUF`) on 2026-10-09.

```text
TestFitCheckFor_pricesSafetensorsNotJustGGUF pins the actual regression this closes: before
P9b, fitCheckFor returned a zero-weight (therefore always-fits) check for ANY non-.gguf path —
a safetensors checkpoint's fit guard was silent by construction, never refusing regardless of
how little RAM was injected. Confirms both weightBytes and kvBytes are now non-zero for a real
safetensors directory with a resolvable context.
```

## TestFitGuard_remedyNamesPrequantNotStreamWeightsForDirectory

Moved from `decoder/fitguard_test.go` (the comment above `TestFitGuard_remedyNamesPrequantNotStreamWeightsForDirectory`) on 2026-10-09.

```text
TestFitGuard_remedyNamesPrequantNotStreamWeightsForDirectory gates M-30: the remedy text for a
refused safetensors DIRECTORY used to unconditionally recommend -stream-weights — a flag that
is a genuine no-op for a directory (decoder.Load ignores it for anything but a .giw, and
serve's own -stream-weights gates are .gguf-suffix-only) — so a user who did exactly what the
message told them got an identical refusal back. The remedy must now name the escape hatch
that actually works for this source (GOINFER_NO_FIT_GUARD=1) and the real path to a permanent
fix (cmd/prequant), and must NOT recommend -stream-weights at all.
```

## TestQuantBytesPerElem_everyModeIsPlausible.int4

Moved from `decoder/fitguard_test.go` (the comment above `TestQuantBytesPerElem_everyModeIsPlausible.int4`) on 2026-10-09.

```text
int4 has exactly TWO legitimate costs, and which one applies is a property of the host, not
of the encoding:

  ~0.5625 canonical nibbles + one binary16 scale per group of 32 (aikit v1.50.0; 0.625 with the f32
          scales before it), and no repack
  ~1.125  the same, PLUS a second repacked buffer that the loader keeps beside it —
          RepackInt4Row4 on arm64-with-dotprod (nibbles and scales both doubled; 1.250 before)

Pinning the pair rather than a range is the point: a wrong measurement usually lands
BETWEEN them, and a range wide enough to hold both would accept it.

This replaces an assertion that int4 must be cheaper than int8, which CI proved false.
Measured on darwin/arm64 2026-09-06: int4 1.2500 against int8 1.0156 — on Apple Silicon
int4 weights occupy about 23% MORE resident RAM than int8int8, because int8 gets no repack.
See docs/tasks/task-first-hour.md for what that means for the help text's "int4 ... smallest".
```

## TestKvDimAt_zeroForNemotronNonAttentionLayers

Moved from `decoder/fitkv_test.go` (the comment above `TestKvDimAt_zeroForNemotronNonAttentionLayers`) on 2026-10-09.

```text
TestKvDimAt_zeroForNemotronNonAttentionLayers gates a residual gap in M-28 found while
implementing P-02 (docs/audit-2026-09-10.md): Nemotron's mixer identity is per-layer RUNTIME
DATA (nemotronParams.blockKind, read from layers_block_type), not a registry-time closure like
Granite's layerIsMamba — so isMambaLayer never fires for a Nemotron mamba layer at all, and its
mlp/moe block kinds (decoder/forward_nemotron.go's own switch: only nemoAttn ever touches
cache) touch no K/V either, uncaught by any generic predicate. Real fixture, not synthetic:
testdata/nemotron-tiny's layers_block_type is exactly
["mamba","attention","mlp","mamba","attention"].
```

## TestKvDimAt_mlaUsesCompressedLatentNotReconstructedWidth

Moved from `decoder/fitkv_test.go` (the comment above `TestKvDimAt_mlaUsesCompressedLatentNotReconstructedWidth`) on 2026-10-09.

```text
TestKvDimAt_mlaUsesCompressedLatentNotReconstructedWidth gates M-28's MLA sub-fix: the cache
holds the compressed KVLoRARank+QKRopeHeadDim latent (forward_deepseek.go reconstructs
per-head K/V from it each step), not NumKVHeads*HeadDim's full reconstructed width — the audit's
own DeepSeek-V2-Lite figure ("MLA stores 576/layer, priced 4096") is exactly this gap.
```

## TestKvPositionsAt_slidingWindowCapsLocalLayersNotGlobal

Moved from `decoder/fitkv_test.go` (the comment above `TestKvPositionsAt_slidingWindowCapsLocalLayersNotGlobal`) on 2026-10-09.

```text
TestKvPositionsAt_slidingWindowCapsLocalLayersNotGlobal gates M-28's sliding-window sub-fix: a
local (non-global) layer's ring never grows past SlidingWindow regardless of ctx, while a
global layer's cost keeps growing with ctx — the gemma3-12b "~6x over at 131k" figure traces
to exactly this asymmetry being ignored.
```

## TestFitGuard_mlaKVPricedAtCompressedLatentNotFullWidth

Moved from `decoder/fitkv_test.go` (the comment above `TestFitGuard_mlaKVPricedAtCompressedLatentNotFullWidth`) on 2026-10-09.

```text
TestFitGuard_mlaKVPricedAtCompressedLatentNotFullWidth is M-28's DeepSeek-V2-Lite case from the
audit ("MLA stores 576/layer, priced 4096"), on the real fixture: the fixed estimator must price
meaningfully LESS than the flat formula it replaces, because the flat formula charges the full
reconstructed per-head width where MLA actually caches only the compressed latent.
```

## TestFitGuard_hybridKVSkipsRecurrentLayers

Moved from `decoder/fitkv_test.go` (the comment above `TestFitGuard_hybridKVSkipsRecurrentLayers`) on 2026-10-09.

```text
TestFitGuard_hybridKVSkipsRecurrentLayers is M-28's DeltaNet-hybrid case, on the real qwen3.5
fixture the audit itself names ("Qwen3.5-4B: priced 128 KiB/position vs 32 KiB stored").
```

## TestFitGuard_slidingWindowFlattensPastTheWindow

Moved from `decoder/fitkv_test.go` (the comment above `TestFitGuard_slidingWindowFlattensPastTheWindow`) on 2026-10-09.

```text
TestFitGuard_slidingWindowFlattensPastTheWindow is M-28's olmo3-7B case from the audit
("~3.4x over at 65k"), on the real fixture: at a ctx well past the model's own sliding window,
the fixed estimator's local layers must have stopped growing while the flat formula keeps
charging every layer as if it held the full context.
```

## TestPlan_tableDriven

Moved from `decoder/fitplan_test.go` (the comment above `TestPlan_tableDriven`) on 2026-10-09.

```text
TestPlan_tableDriven is G4 (docs/tasks/task-fit-to-hardware.md §6): "a table-driven unit test on
synthetic headers — dense, MoE, hybrid, every backend, every budget... pins the placement and
the ctx cap. A change to the priority order is a change to this table, reviewed." Not literally
synthetic headers (Phase 1 was scoped Load()-based, docs/tasks/task-gpu-paths-2026-09.md's G11 entry)
— real tiny checkpoints instead, budgets scaled to each fixture's OWN measured byte counts
rather than the doc's literal "6 to 64 GB" (these are toy-sized parity fixtures, not real
deployment checkpoints, so a literal GB range would never exercise the decline path at all).
```

## TestPlan_extraBytesReservedAheadOfExperts

Moved from `decoder/fitplan_test.go` (the comment above `TestPlan_extraBytesReservedAheadOfExperts`) on 2026-10-09.

```text
TestPlan_extraBytesReservedAheadOfExperts is the regression this session's own G11 CUDA guard
work (docs/tasks/task-gpu-paths-2026-09.md) traces back to: tasks/task-fit-to-hardware.md's motivating
example (a --drafter attach after BuildResident grabbed VRAM an MoE expert cache had already
claimed). PlanRequest.ExtraBytes exists so the CALLER can price a companion allocation (a
drafter, a vision tower) as a FIXED term ahead of the elastic expert-slot count, per §2's "every
allocation is a term of the plan, including the ones that attach after load."
```

## TestPlan_unrecognisedBackendDeclinesEvenForAFeatureFreeArch

Moved from `decoder/fitplan_test.go` (the comment above `TestPlan_unrecognisedBackendDeclinesEvenForAFeatureFreeArch`) on 2026-10-09.

```text
TestPlan_unrecognisedBackendDeclinesEvenForAFeatureFreeArch is the M-08 gate
(docs/audit-2026-09-10.md): the existing "nonsense backend declines" test
(moe/nonsense_backend_declines_on_features_not_bytes, above) only proves the decline for an
arch with NON-EMPTY RequiredResidentFeatures — a plain Llama has none, so
MissingResidentFeatures(nil) returned empty (nothing required, nothing implemented, so
"nothing missing") and Plan fell through to RESIDENT for ANY backend name, including one that
does not exist. This is the same shape as "Llama-4/cuda", "dense Gemma-4/webgpu" and
"Kimi-K2/metal" in the finding: a family whose feature list alone doesn't catch the decline.
```

## TestFitEstimate_f32PinnedMixersAgreeWithResident

Moved from `decoder/fitweights_test.go` (the comment above `TestFitEstimate_f32PinnedMixersAgreeWithResident`) on 2026-10-09.

```text
TestFitEstimate_f32PinnedMixersAgreeWithResident is M-29's headline case (docs/audit-2026-09-10.md):
before the fix, a Mamba-2 mixer's or MLA's f32-pinned tensors were priced at the requested
quant's rate instead of f32, drastically UNDER-counting them at int4/int8 (the audit's own
figure: "a Nemotron-H-8B '≈4.7 GB at int4' lands at ≈13 GB, admitted"). Same
estimate-vs-accountant band as TestFitEstimate_safetensorsAgreesWithResidentWeightBytes,
extended to real f32-pinned-mixer fixtures.
```

## TestFitEstimate_f32PinnedMixersAgreeWithResident.band

Moved from `decoder/fitweights_test.go` (the comment above `TestFitEstimate_f32PinnedMixersAgreeWithResident.band`) on 2026-10-09.

```text
Same band as the dense estimate-vs-accountant tests — before M-29 this was
nowhere near 1.0 for these fixtures at int4 (the f32-pinned tensors alone are
4x their int4-priced estimate, and dominate a Mamba-2/MLA-heavy model).

MLA's floor is a hair lower (0.80, not 0.85): f32PinnedTensorName's own doc
comment documents a known, deliberate residual gap — MLA's output projection
(o_proj/dense.weight) is ALSO f32-pinned but shares its name with every
ordinary family's quantizable o_proj, so it is left unmatched rather than
risking a much larger false-positive elsewhere. This is that gap's real,
measured size, not a loosened bar to paper over a regression.
```

## requireFixtureIdentity

Moved from `decoder/fixture_identity_test.go` (the comment above `requireFixtureIdentity`) on 2026-10-09.

```text
The tiny parity checkpoints are gitignored and regenerated per machine by scripts/pin_*.py, but the
goldens recorded from them are committed — and the pin scripts do not reproduce the same random weights
across torch/transformers versions. So a box whose checkpoint was re-pinned elsewhere compares the
committed golden against DIFFERENT weights, and the parity test reports a meaningless cosine (−0.04 for
gemma3-vl-tiny) that reads exactly like a forward-pass regression. That kept five decoder tests red on
the Mac from 2026-09-18 to 2026-09-24, pre-registered as "known unrelated" and never investigated
(docs/measurements/tiny-fixture-golden-mismatch-2026-09-24.md).

requireFixtureIdentity fails fast, with the actual diagnosis, when a fixture listed in
testdata/fixture_identity.json does not hold the weights its goldens were recorded from. Since
2026-09-24 the small fixtures are COMMITTED instead (git pins their bytes); the manifest covers only
the ones too large to commit, and TestInt4_forwardParity's per-fixture subtest is the caller.

The comparison is NUMERIC, not a file hash: the same pin script with the same seed produces
byte-different checkpoints on arm64 and amd64 (measured: nemotron3nano-tiny and qwen3next-tiny differ in
33-36% of elements between the Mac and nobara, by at most 2.4e-7 — float32 ULP noise from torch's normal_
on the two architectures), and both correctly pass every golden. A different random draw moves a
tensor's sum by O(sum|x|/sqrt(n)); ULP noise moves it by ~1e-7 of sum|x|. The 1e-4 relative tolerance
sits three orders of magnitude from each.
```

## forcedFallbackFloors

Moved from `decoder/forced_fallbacks_test.go` (the comment above `forcedFallbackFloors`) on 2026-10-09.

```text
Forced-fallback floors. Under a forced narrower kernel the numerics are a different realization of the same arithmetic, so the goldens that pin ONE path's token sequence or sample values
(recorded on the default path; the int4 goldens are arm64-baked) cannot be compared exactly. These replace them under the tags with a closeness bound, measured 2026-10-04 on a Ryzen 7 3700X
(docs/tasks/task-hardware-coverage-2026-10.md, H1.3), chosen between the measured healthy value and what a real defect does, and each shown red by a mutation.
```

## forcedInt4CosFloor

Moved from `decoder/forced_fallbacks_test.go` (the comment above `forcedInt4CosFloor`) on 2026-10-09.

```text
forcedInt4CosFloor: centered cosine of an int4 fixture's logit samples against its recorded golden. Forced noavx2 measured 1.0 on 21 of 22 fixtures and 0.99381096 on gemma4-dense-scaled (the
fixture built to amplify numeric differences; argmax equal). The healthy cross-arch deficit on gpt2 is 2.2e-3 (int4_golden_test.go); a group-size mutation moves it by 2.4e-2.
```

## forcedLogitCloseness

Moved from `decoder/forced_fallbacks_test.go` (the comment above `forcedLogitCloseness`) on 2026-10-09.

```text
forcedLogitCloseness: 1 - cosine between two paths' logits for the same ids, where the pair differs by design: int4 against int8int8 (measured 1.64e-2 pure Go, 1.75e-2 AVX2) and the f32
fast-attention prefill against the exact path (measured 8.3e-3 pure Go, 1.6e-2 AVX2, 768 tokens).
```

## TestExcludeFromFork_syscallReachesMinherit

Moved from `decoder/forkinherit_darwin_test.go` (the comment above `TestExcludeFromFork_syscallReachesMinherit`) on 2026-10-09.

```text
The syscall number and argument order excludeFromFork relies on are real: the same call with an
invalid inheritance value must come back EINVAL from the kernel's minherit (a wrong syscall number
would give ENOSYS or act on something else). XNU accepts minherit over an unmapped hole silently, so
"does it fail on a bad range" is not a usable check; the effect itself is measured by
metal/alias_forkprobe_test.go (fork 1,092 ms -> 3.8 ms while a page is wired).
```

## TestForwardN_matchesSequential

Moved from `decoder/forwardn_test.go` (the comment above `TestForwardN_matchesSequential`) on 2026-10-09.

```text
TestForwardN_matchesSequential checks the batched multi-position forward
(forwardLayersN) against running forward() one token at a time. The bar is BIT-IDENTITY: every
logit equal, not argmax plus a cosine floor.

THE COMMENT THAT USED TO BE HERE DESCRIBED A DIFFERENT CODEBASE. It said "NOT bit-identical,
since batched attention moved QKᵀ/scores·V onto the f32 SIMD A·Bᵀ kernel" against "the scalar
attendQuery" — but decode has not used attendQuery since single-token decode was routed through
attendBatchedHeads at K=1 with acc64. attention.go says so where it does it, names THIS test as
the gate, and gives the reason: f32's reduction is M-dependent, so K=1 decode ≠ M=K verify, and
that flipped ~11% of argmaxes and left ~7% of speculations rejected. f64 is order-independent, so
the two are exact.

So the gate was one assertion weaker than the contract it guards: a regression breaking
decode == prefill by 1e-6 passed here and only the heavy token-level spec-parity gates would have
noticed (audit-2026-09-02 G-06). Measured before tightening — 0 differing logits of 19,447,808 at
K=128 on qwen2.5-coder-0.5b int8int8 — so this asserts a property the code has, not one it ought
to have. forwardN passes fastAttn=false, so the A3 f32 path is not in play here; that path is
deliberately NOT bit-identical and has its own gate.

K=128 is included so the O(L²) attention term is actually exercised (K=6 barely touches it).
Skips without the model asset.
```

## TestG15PrefillProfile.file

Moved from `decoder/g15_prefill_profile_test.go` (the comment above `TestG15PrefillProfile.file`) on 2026-10-09.

```text
Prefill profiler — time and profile ONE batched CPU prefill at a chosen prompt
length and quantization.

Built for queue G15 (a suspected int4-specific prefill cliff at ~3k tokens).
**G15 was WITHDRAWN: the cliff was a measurement artifact** — one 3020-token
int4 timing of 1587.1 s that three later measurements put at ~350 s, int4 and
int8int8 scaling alike at ~n^1.85. This instrument is what caught it, by
disagreeing with the number, so it is kept: the disagreement was the finding.

Two things it does that the original measurement did not, both of which are now
required by `docs/benchmarks.md`'s methodology list:

 1. It records MACHINE STATE beside the number (load average before and after).
    A timing with no recorded state cannot be argued with later, which is what
    made the artifact expensive rather than merely wrong.
 2. It REFUSES to run on a busy box unless explicitly overridden, because the
    artifact's leading suspect is an abandoned prefill still burning a core.

The profile covers the PREFILL ONLY. Model load is excluded deliberately: it is
seconds of unrelated I/O and quantization that would otherwise dominate a short
profile and differ between arms by construction.

Run (one quant per process — loadBenchModel is a sync.Once):

	GOINFER_PREQUANT_GGUF=<model.gguf> GOINFER_BENCH_QUANT=int4 \
	GOINFER_G15_K=3020 GOINFER_G15_PROF=/tmp/int4-3020.prof \
	go test ./decoder/ -run TestG15PrefillProfile -timeout 3600s -v

Then: go tool pprof -top -nodecount=30 <prof>

Set GOINFER_G15_ALLOW_BUSY=1 to run anyway on a loaded box — and then say so
beside any number it produces.
```

## TestG15PrefillProfile.preflight

Moved from `decoder/g15_prefill_profile_test.go` (the comment above `TestG15PrefillProfile.preflight`) on 2026-10-09.

```text
Pre-flight. The hazard that produced the G15 artifact was OUR OWN competing
work — an abandoned prefill still saturating a core — not a busy desktop.
The first version of this check refused on absolute load > 1.5, which is
wrong for the machine it runs on: a developer Mac idles above that with
Spotlight and an editor open, so the check skipped every real measurement
and its only effect would have been to train people to set ALLOW_BUSY=1.

So: refuse on a COMPETING PROCESS OF OURS (the real hazard, and precisely
reproducible), and always RECORD the load rather than gating on it. A number
with its machine state attached can be argued with later, which is the
property that was actually missing.
```

## TestG15PrefillProfile.alloc

Moved from `decoder/g15_prefill_profile_test.go` (the comment above `TestG15PrefillProfile.alloc`) on 2026-10-09.

```text
Report the allocation delta alongside the profile: the GC hypothesis for the
cliff predicts a large one, and it costs nothing to answer here rather than
in a second run.
```

## TestG24AttnKernelRatio

Moved from `decoder/g24_attnkernel_test.go` (the comment above `TestG24AttnKernelRatio`) on 2026-10-09.

```text
G24 — how much would an f32 attention path actually recover?

At K=8192 prefill, acc64 attention is ~70% of the time (MatmulAVAcc64 51.1% +
MatmulQKAcc64 18.7%). The acc64 comment calls f64 "~3.7× slower than f32", and
A3 proposes a gated f32 path. But the honest comparison is NOT end-to-end:

  - acc64 reads K/V DIRECTLY by stride, "skipping a kh gather entirely" and
    "skipping a vt gather+transpose". f32 must pay both, so the f32 arm here
    includes those gathers — otherwise the ratio flatters f32 by omitting work
    it cannot avoid.
  - the f32 branch in attendBatchedHeads is single-threaded by construction
    (its per-kv-group gather is shared mutable state), so an end-to-end A/B
    would race parallel-acc64 against serial-f32 and measure the confound.
    Both arms here are single-threaded, which is the like-for-like comparison.

Shapes are the ones G20's tiling actually calls at an 8k prompt.
```

## TestG24AttnKernelRatio.parallelism

Moved from `decoder/g24_attnkernel_test.go` (the comment above `TestG24AttnKernelRatio.parallelism`) on 2026-10-09.

```text
EQUALIZE PARALLELISM. MatmulBT fans out via parallelCols; the acc64 kernels
are plain serial loops. A first pass of this benchmark compared serial acc64
against parallel f32 and reported 17.6x against a documented ~3.7x — the
ratio was mostly core count. Force MatmulBT serial so the number is the
arithmetic-plus-gather truth, then report the parallel figure separately as
what it is: a SECOND, separable effect.
```

## TestG24AttnKernelRatio.correctness

Moved from `decoder/g24_attnkernel_test.go` (the comment above `TestG24AttnKernelRatio.correctness`) on 2026-10-09.

```text
CORRECTNESS FIRST. A ratio between two kernels that compute different things
is worthless, and the first version of this benchmark already produced one
misleading number (17.6x, from unequal parallelism). So: run both arms once
and compare outputs before timing anything.
```

## g26BenchLogits.prohibition

Moved from `decoder/g26_sampler_bench_test.go` (the comment above `g26BenchLogits.prohibition`) on 2026-10-09.

```text
SAMPLER MICROBENCHMARKS IN THIS PACKAGE MAY NOT PRODUCE QUOTABLE FIGURES.

They are a tool for deciding WHERE TO LOOK. No number they emit belongs in a doc, a queue item, a
commit message, or a comparison between two commits. This is a standing prohibition, not advice,
and it was earned twice in a single investigation (G26, 2026-08-27):

  1. BenchmarkExpChunked's ~96 us was used to argue that "sampling is a low-single-digit
     percentage of per-token time, so no sampler change can move end-to-end by 5.9%." The real
     in-situ sampled tail is 703-950 us — 7-10x larger. The bound retired the correct hypothesis
     for two rounds.
  2. A whole-path Sampler.Sample benchmark reported HEAD 23% SLOWER at 152k vocab. Measured
     end-to-end on a 151936-vocab model, HEAD is 31% FASTER. Not a mis-scaled magnitude — an
     INVERTED SIGN, and it had already been filed as a finding before the end-to-end run.

WHY the loop lies here specifically: it keeps the vocab-sized scratch hot in cache and the
allocator warm in its free-list, whereas decode runs one draw per token behind an ~8 ms forward
that evicts both. The benchmark measures a cache state that never occurs.

MEASURE IT IN SITU INSTEAD, which costs nothing extra: the same-build end-to-end greedy vs
temp1.0 difference, with GOINFER_NO_OPTFWD=1 so the optimistic-forward overlap cannot confound
it. On CUDA that difference is the whole sampled tail (greedy takes ForwardArgmax's on-device
path and never reads back the logit vector), not the sampler alone — but it is measured where the
code runs, and both times the two disagreed, the microbenchmark was the one that was wrong.
```

## g26BenchLogits.localisation

Moved from `decoder/g26_sampler_bench_test.go` (the comment above `g26BenchLogits.localisation`) on 2026-10-09.

```text
G26 localisation: the FULL temp1.0_notrunc sampling path, not expChunked alone.

WHY THE WHOLE PATH. G26's earlier round eliminated the sampler using BenchmarkExpChunked's
~96 us, and that bound was wrong by 7-10x: the end-to-end greedy-vs-temp1.0 gap puts the real
sampling step at 703 us (ca29d6c) / 950 us (HEAD) per token on phi3-mini. A microbenchmark of
one function inside the path cannot bound the path. This benchmarks Sampler.Sample itself.

The greedy arm is the control, and it is the point: subtracting it here reproduces the same
decomposition the peer sweep produced end-to-end, so the two are comparable rather than merely
suggestive.

RESULT, 2026-08-27: THIS BENCHMARK'S 152k COMPARISON IS WRONG — do not use it as evidence.
Measured end-to-end on qwen2.5-coder-1.5B (vocab 151936) with optFwd disabled in both arms, the
production sampling step goes 1467 us -> 1009 us, i.e. HEAD is 31% FASTER, where this benchmark
reports HEAD 23% SLOWER. The sign is inverted, not merely the magnitude. A tight loop keeps the
scratch buffer hot and the allocator warm; decode does neither, running it behind an ~8 ms
forward. Prefer the in-situ measure: the same-build greedy-vs-temp1.0 end-to-end difference with
GOINFER_NO_OPTFWD=1. Kept only because the retraction is worth more than the file.

CAVEAT, stated because it decides how a null result is read: these logits are SYNTHETIC. If the
regression is data-dependent (a denormal-heavy tail, say), synthetic logits may not reproduce it
-- and a flat result here is then evidence about the shape of the cause, not an absence of one.
```

## TestGPT2_activationSelectsTheNamedFunction

Moved from `decoder/gelu_dispatch_test.go` (the comment above `TestGPT2_activationSelectsTheNamedFunction`) on 2026-10-09.

```text
GPT-2's activation_function must select the function it NAMES.

The bug this pins: validateGPT2 accepted both "gelu_new" and "gelu", and
gpt2Architecture hardcoded Act: ActGeluTanh — so a checkpoint declaring HF's exact
"gelu" silently ran the tanh approximation. In HF these are different entries,
ACT2FN["gelu"] = GELUActivation (exact erf) vs ACT2FN["gelu_new"] = NewGELUActivation
(tanh), not two spellings of one function.

aikit hit the mirror image on its encoder side in v1.19.0 — three tanh names routed
through the exact erf — and its note said no shipping checkpoint was affected, latent
for a future addition. The same was true here, which is the point: this test exists so
the next GPT-2-family checkpoint that declares "gelu" gets the function it asked for
instead of a silently approximated one.
```

## TestGemma3Real.file

Moved from `decoder/gemma3_real_test.go` (the comment above `TestGemma3Real.file`) on 2026-10-09.

```text
Real-model gate for gemma-3-4b-it TEXT decoder (gemma3) — the safetensors loader + gemma3
forward on actual released weights, text-only (the VL wrapper's vision path is not exercised).
4B fits an f32 forward in RAM, so Options{} ⇒ quantNone and the gate is a TIGHT cosine vs the
HF f32 golden (argmax + greedy continuation + cosine ≥ 0.9999). Verifies the gemma3 axis on
real weights: sandwich (4-norm) placement + per-head QK-norm + sliding-window interleave +
embed scale. Fixture: scripts/pin_gemma3_4b_text.py. This is the real-oracle emit gate that
moves gemma3 pending → validated (batched-prefill coverage: gemma3 is canBatchN-batchable).
```

## TestGemma3TowerSensitivity.decoder

Moved from `decoder/gemma3_tower_sensitivity_real_test.go` (the comment above `TestGemma3TowerSensitivity.decoder`) on 2026-10-09.

```text
EmbedInt4 off: what serve's --backend metal loads (loadflags.embedInt4), so the G-S3b arms' own decoder. Through the
directory's sidecar when it exists (S18: `<dir>.int4.metal.giw`, the file serve's Metal load reads, int4 with the
plain head), mapped rather than quantized into the heap, which the load guard refused on the 16 GB Mac (the
2026-10-08 night: 6.2 GB against 4.8). The sidecar generates exactly what the direct load does
(prequant.TestDirSidecar_matchesDirectLoad).
```

## TestGemma3TowerSensitivity.prompt

Moved from `decoder/gemma3_tower_sensitivity_real_test.go` (the comment above `TestGemma3TowerSensitivity.prompt`) on 2026-10-09.

```text
serve's prompt, built the way serve builds it (internal/serveapp encodeVisionSegments): the model's chat template
rendered as segments, the image block (the processor's "\n\n" on both sides, M-38) spliced in as its own Special
segment, then EncodeSegments. Until 2026-10-08 this hand-wrote the template and encoded it as one string, which merges
the template's "\n" with the block's "\n\n" into one token: 277 tokens against serve's 278, so the greedy reference
was not the served path (the night of 2026-10-07: "...a table of..." where serve says "...quarterly unit sales...").
```

## distinctTrigramRatio

Moved from `decoder/gemma4_26b_real_test.go` (the comment above `distinctTrigramRatio`) on 2026-10-09.

```text
distinctTrigramRatio returns |distinct 3-rune windows| / |total| over s — a
language-agnostic degeneracy metric. Coherent prose cycles through many trigrams (high
ratio); a looping forward reuses a few ("water-water-water", "얓숌면-얓숌면-얓숌면",
"true or true or") so the ratio collapses. This is what the old printable-ASCII-majority
check was blind to: it passed int8's English-ish repetition and flagged int4's CJK
repetition, though both are the SAME degeneracy — see the RESOLUTION in
docs/task-gemma4-moe.md (da5a6ec).
```

## wantGemma4Prompt

Moved from `decoder/gemma4_26b_real_test.go` (the comment above `wantGemma4Prompt`) on 2026-10-09.

```text
wantGemma4Prompt is the exact rendered generation prompt for the gate's user turn.
Asserted verbatim so a template/marker change (e.g. the <|turn>/<turn|> rename that
681db0c made optional in tokenizer/sentencepiece.go) surfaces HERE as a clear diff,
not downstream as mystery garbage.
```

## TestGemma4_26B.precisions

Moved from `decoder/gemma4_26b_real_test.go` (the comment above `TestGemma4_26B.precisions`) on 2026-10-09.

```text
Run the gate at BOTH precisions. int4 (~13 GB) is the config Phase 5 benchmarks and
is fully coherent under the chat template — the earlier "int4 is incoherent" claim
was a raw-prompt artifact (retracted, da5a6ec). int8 (~26 GB) is the control.
```

## gemma4CoherenceGate

Moved from `decoder/gemma4_26b_real_test.go` (the comment above `gemma4CoherenceGate`) on 2026-10-09.

```text
gemma4CoherenceGate is the real gate: a greedy continuation of a PROPERLY TEMPLATED
chat turn must contain the known answer and clear the degeneracy floor.

The prompt MUST go through Gemma 4's chat template. A raw completion prompt ("The
capital of France is") is off-distribution for this instruction-tuned checkpoint: BOTH
int8 and int4 degenerate there (int8 → "water-water-water is 100°C", int4 → CJK
repetition), and the old printable-ASCII gate mistook int8's English-ish garbage for
coherence while flagging int4's CJK — a false precision signal that cost a week
(resolved da5a6ec). DO NOT reintroduce a raw prompt here as a quality check;
gemma4RawPromptControl keeps the raw case as a NEGATIVE control only. Greedy + fixed
prompt is deterministic (sampling does not rescue a raw prompt anyway — verified), so
the known-answer substring is a legitimate, non-flaky signal.
```

## TestGemma4_26B_s10Grade.margins

Moved from `decoder/gemma4_26b_s10grade_test.go` (the comment above `TestGemma4_26B_s10Grade.margins`) on 2026-10-09.

```text
Per-position margins for the disagreement (the owner's follow-up of 2026-10-07: is the 26B's low argmax agreement near-ties?).
G1c's rule: benign iff the CPU's top-1 and the index Metal chose are within 3% in the CPU's logits (relative to the CPU's top-1).
The G3 rule is on probability: p(Metal's choice) >= half p(CPU's top-1), the softmax taken over the CPU's logits.
```

## TestGemma4Admission_unconditional

Moved from `decoder/gemma4_admission_test.go` (the comment above `TestGemma4Admission_unconditional`) on 2026-10-09.

```text
TestGemma4Admission_unconditional pins Gemma-4 admission after the Check-A backfill removed
GOINFER_GEMMA4_RESIDENT from the arch predicate. Dense and enable_moe_block are admitted
unconditionally; the per-backend answer stays with the feature gate, so WebGPU still declines
(no Gemma kernels) and E-models still decline everywhere (PLE).

It asserts the variable is INERT rather than dropping the coverage: a reintroduced read would
silently reopen the hole the flag used to hide — a family gated on an env var with no gate
behind it — so this fails if setting or clearing it changes any answer. The matrix's
GPUResident column is arch.decodeRunnerEligible() (capability_matrix_test.go), so this is also
what keeps docs/hardware-matrix.md honest.
```

## TestGemma4Admission_unconditional.inert

Moved from `decoder/gemma4_admission_test.go` (the comment above `TestGemma4Admission_unconditional.inert`) on 2026-10-09.

```text
The variable is INERT in both states. Previously env-off meant "declined everywhere", which
is what kept the generated matrix reporting CPU for a path that worked — see the flip commit.
```

## TestGemma4Admission_unconditional.moeadmit

Moved from `decoder/gemma4_admission_test.go` (the comment above `TestGemma4Admission_unconditional.moeadmit`) on 2026-10-09.

```text
enable_moe_block now admits — Split B (splitB.2c) landed the parallel dense‖MoE
FFN on its own cuda path (gemma4MoeMLP, routed around the generic MoE checks via
HasGemma4MoEResident), so it falls through the arch predicate like the dense variant. CUDA
ships every required feature; WebGPU still lacks the Gemma kernels, so the feature gate
refuses it — same no-overclaim shape as dense. It is the PLE-free 26B-A4B shape, so no
FeatGemma4EModel.
```

## TestGemma4Admission_unconditional.moe

Moved from `decoder/gemma4_admission_test.go` (the comment above `TestGemma4Admission_unconditional.moe`) on 2026-10-09.

```text
The MoE variant is inert to the variable too. This used to assert the OPPOSITE — that env-off
declined everywhere, "so a regenerated matrix cannot claim gemma4_text residency users can't
reach". That reasoning was sound and the conclusion still holds; what changed is which way it
resolves. The matrix is honest either way, but only one of the two is honest AND useful.
```

## TestGemma4EndToEndThroughput

Moved from `decoder/gemma4_endtoend_throughput_test.go` (the comment above `TestGemma4EndToEndThroughput`) on 2026-10-09.

```text
TestGemma4EndToEndThroughput re-measures the END-TO-END paged-decode gap
(docs/completed/task-zeno-compare.md's "Quiet-machine re-measure": gemma4-26b kind-4
vs kind-3, -47.0%/-49.0% at 4GB/8GB, budget-invariant) — the number the
whole cold-touch investigation was chasing an explanation for, now that
the kernel-level 69%-slower finding has failed to reproduce 3/3 on a
corrected methodology. This is NOT the isolated kernel microbenchmark;
it's the real production path (Load with StreamWeights+WeightCacheBytes,
then Model.Generate), same steady-state total_tokens/wall_time metric the
original measurement used, adapted from cmd/serve+HTTP to a direct
in-process call (no HTTP round-trip noise; the model is loaded ONCE and
generation run 3 times against the same resident/paged state, matching
the original "3 runs against one running server" shape without paying a
multi-minute reload cost 3 times over).

Env vars: GOINFER_EE_GIW (path), GOINFER_EE_BUDGET_GB (float, WeightCacheBytes
in GB), GOINFER_EE_LABEL (free-text label for the log line).
```

## TestGemma4Config_realGGUF

Moved from `decoder/gemma4_load_test.go` (the comment above `TestGemma4Config_realGGUF`) on 2026-10-09.

```text
TestGemma4Config_realGGUF runs the real E2B GGUF through ggufConfig +
resolveArchitecture (the Increment-1 config/descriptor path) and asserts the
parsed descriptor matches the verified metadata. Weight loading is guarded off
(the next increment), so this stops at the descriptor. Skips without the asset
(set the file at ~/models/gemma-4-E2B_q4_0-it.gguf).
```

## TestGemma4VLBidir_textParity

Moved from `decoder/gemma4_vl_bidir_test.go` (the comment above `TestGemma4VLBidir_textParity`) on 2026-10-09.

```text
TestGemma4VLBidir_textParity is the P0 invariant for the batched path,
mirroring TestGemma4VL_textParity's shape: on a use_bidirectional_attention=
"vision" checkpoint, a TEXT-ONLY forward (no image) must still match HF —
confirmed this session that HF's own forward degrades block_sequence_ids to
an all -1 tensor when there is no multimodal content, which makes
blockwise_overlay unconditionally false, i.e. plain causal (see
scripts/pin_gemma4_vl_bidir_tiny.py's docstring for the citation). This test
checks BOTH the unchanged sequential path (m.forward, via runLayersGemma4)
AND the new batched path (runLayersGemma4FromEmbedN with imgLen=0) against
the HF golden, and against each other — the regression check the plan
called for: an M=K batched matmul is not guaranteed bit-identical reduction
order to K separate M=1 calls, so the bound here is a tight tolerance, not
literal bit-exactness.
```

## TestGenerateGemma4VL_residentContextCapPublishesBudgetClamped

Moved from `decoder/gemma4_vl_resident_test.go` (the comment above `TestGenerateGemma4VL_residentContextCapPublishesBudgetClamped`) on 2026-10-09.

```text
TestGenerateGemma4VL_residentContextCapPublishesBudgetClamped mirrors
TestGenerateVL_residentContextCapPublishesBudgetClamped (generate_vl_resident_test.go) for
GenerateGemma4VL's own single clamp site (M-02, docs/audit-2026-09-10.md) — before this fix,
none of the 7 VL resident clamp sites published Budget/BudgetClamped, so a cap-truncated turn
silently reported the same finish_reason as an ordinary EOS-terminated one.
```

## TestGenerateGemma4VL_sequentialEModelUploadsOwningLayers

Moved from `decoder/gemma4_vl_resident_test.go` (the comment above `TestGenerateGemma4VL_sequentialEModelUploadsOwningLayers`) on 2026-10-09.

```text
TestGenerateGemma4VL_sequentialEModelUploadsOwningLayers is the gitignored gemma4-vl-tiny fixture's check of the
S1.8 contract (docs/tasks/task-multimodal-support-2026-10.md): a sequential (causal) checkpoint of the E-model class
(this one has KV-shared layers) with a resident attached uploads its CPU prefill's K/V for the layers that own it,
never a KV-shared one, then decodes resident. Until S1.8 the bridge was gated on UseBidirectionalAttention and this
test asserted it was never touched; Metal now runs E-models, so that premise changed (nobara caught the stale test,
2026-10-07: CI skips it, the fixture being gitignored). The same contract runs on a committed-path fixture in
TestGenerateGemma4VL_eModelDecodesResident.
```

## TestGemma4VL_textParity

Moved from `decoder/gemma4_vl_test.go` (the comment above `TestGemma4VL_textParity`) on 2026-10-09.

```text
TestGemma4VL_textParity loads the tiny Gemma 4 VL checkpoint (scripts/
pin_gemma4_vl_tiny.py) through goinfer's loader + forward and asserts the
TEXT-ONLY path matches the HF golden. Same P0 invariant as Gemma 3's own
TestGemma3VL_textParity: a VL checkpoint's text decoder (vision_tower/
embed_vision ignored) loads and runs exactly like a plain gemma4.

This fixture's num_kv_shared_layers=2 (of 4 layers) is not incidental — it
is the exact shape that caught two real, pre-existing bugs in
buildWeightsFromSafetensors's gemma4 branch this session (a missing
cross-layer-KV-sharing skip, and a missing per-layer FFN-width discovery),
found only by loading a real checkpoint of this shape for the first time.
This test is what gives that fix CI-repeatable coverage.
```

## TestGemmaConfirmerReference

Moved from `decoder/gemma_confirmer_test.go` (the comment above `TestGemmaConfirmerReference`) on 2026-10-09.

```text
TestGemmaConfirmerReference assembles the matched-input confirmer's reference bundle from
existing seams — no new capture code — so the Metal box can inject goinfer's exact L1 state
into Metal's attention and isolate the crater (0.9994 @ L0 -> 0.640 @ L1) to one of:

	Metal's L1 context MATCHES this reference given matched input  -> the crater is accumulated
	  f16/precision drift in the residual+KV feeding attention (fix: f32 KV / f32 attention
	  accumulate for Gemma's low-magnitude contexts).
	Metal's L1 context STILL inflates on matched input            -> Metal's attention op
	  (softmax/scale/accumulate) has a real bug, independent of drift.

The bundle at layer L1, position 5 ("The capital of France is"):

	residual entering L1 = ForwardCapture(prompt, [0])[0]   (layer-0 output = L1 input)
	K/V at L1            = cache.Keys(1) / cache.Vals(1)     (post-RoPE K, raw V, f32)
	target context       = ForwardSubCapture -> subCtx[1]

Runs on CPU int4 over the byte-identical Q4_K_M gguf (sha 882e8d2d), which reproduces the
CUDA resident context to decimals — so the Metal box gets the same reference either box would
produce, and the injection stays entirely on its side (naturally-f32 KV, no faked state).
```

## TestGenerateVL_residentDecodeEngagesAndUploadsKV

Moved from `decoder/generate_vl_resident_test.go` (the comment above `TestGenerateVL_residentDecodeEngagesAndUploadsKV`) on 2026-10-09.

```text
TestGenerateVL_residentDecodeEngagesAndUploadsKV is gap 0's wiring gate for GenerateVL, extended
for P9(a): given a resident backend, the turn must (a) call UploadKV once per layer with the CPU
prefill's K/V, (b) dispatch decode through the resident Forward (not the CPU m.forward), and (c)
COMMIT resIDs + the image block afterward — a stale resIDs from an unrelated prior generation
must not survive as a false reuse candidate (m.resIDs starts at a value that cannot satisfy the
image-block check below), and this turn's own prefix becomes the reuse candidate for the NEXT
turn if the same image comes back (the entire point of P9(a); GenerateVL used to be forbidden
from touching resIDs at all, then gap 0 made it unconditionally forget — now it commits).
```

## TestGenerateVL_residentContextCapPublishesBudgetClamped

Moved from `decoder/generate_vl_resident_test.go` (the comment above `TestGenerateVL_residentContextCapPublishesBudgetClamped`) on 2026-10-09.

```text
TestGenerateVL_residentContextCapPublishesBudgetClamped is M-02's (docs/audit-2026-09-10.md)
gate: when the resident context cap clamps maxTokens down, the Generation must publish
Budget/BudgetClamped (openai.go's effectiveBudget trusts Budget only when BudgetClamped is
true) — before this fix, a cap-truncated VL turn silently reported the same finish_reason as an
ordinary EOS-terminated one. capPos leaves room for exactly 2 decode tokens; maxNew (5) asks for
more, forcing the clamp on this — the "ordinary path" (residentUploadPrefill) — site.
```

## TestGenerateVL_residentConcurrencyRace

Moved from `decoder/generate_vl_resident_test.go` (the comment above `TestGenerateVL_residentConcurrencyRace`) on 2026-10-09.

```text
TestGenerateVL_residentConcurrencyRace is the direct regression guard for V-11's actual bug
shape (docs/review-2026-09-04.md), now meaningful for the first time since that fix: a
resident-touching plain Generate and a resident-touching GenerateVL running concurrently on
the SAME *Model must not corrupt resIDs or double-claim resBusy. Run with -race.
```

## TestGenerateVL_imageReuseFastPath_declinesPastResidentCap

Moved from `decoder/generate_vl_resident_test.go` (the comment above `TestGenerateVL_imageReuseFastPath_declinesPastResidentCap`) on 2026-10-09.

```text
TestGenerateVL_imageReuseFastPath_declinesPastResidentCap is M-01's own gate for the P9a fast
path (docs/audit-2026-09-10.md): a prompt in (ResidentContextCap, MaxPositions) — a strict
extension of an already-committed prefix, so P9a's own full-reuse condition (reuseFrom >=
imgPos+imgLen) is satisfied — must NOT take the fast path once it would overrun the resident
cap. Before the fix, residentPrefillSeed's own error there just set g.err and returned with no
fallback; this must instead decline cleanly and fall through to the ordinary path (tower runs
again), exactly as "not fully reused" already does.
```

## gguf_dims_test.file

Moved from `decoder/gguf_dims_test.go` (the comment above `gguf_dims_test.file`) on 2026-10-09.

```text
Track 2.2 (testing campaign): goinfer's INTERPRETATION of a GGUF — metadata →
Config synthesis — must turn a hostile header into a typed error, never a
panic. FuzzGGUFConfig fuzzes that path; TestGGUF_hostileDims_typedError is the
regression for the makeslice panic that surfaced (block_count overflowing int
→ negative NumLayers). The container parse below it (string lengths, map
pre-sizing) is aikit's and was hardened in v1.2.1, so the fuzzer now reaches
goinfer's layer instead of dying in the parser.
```

## ggufSeeds.corpus

Moved from `decoder/gguf_dims_test.go` (the comment above `ggufSeeds.corpus`) on 2026-10-09.

```text
Every dispatchable architecture gets a seed so the fuzzer exercises its family
builder — the div-by-zero / makeslice sites M16 hardened live inside these, and
the previous corpus reached none of the last seven (granite/nemotron/deepseek/
glm4moe/qwen35moe/phi3/llama4). A generic dense seed is enough to drive each
builder's entry; the fuzzer mutates from there.
```

## TestGemma4GGUF_moeLayerScalarMatchesLayer

Moved from `decoder/gguf_gemma4_layerscalar_test.go` (the comment above `TestGemma4GGUF_moeLayerScalarMatchesLayer`) on 2026-10-09.

```text
A direct GGUF load of the gemma4 26B-A4B produced nothing but <pad> (token 0) on CPU, on HEAD, while the sidecar
.giw built from the same file generated text (docs/measurements/transcode-streaming-gemma4-2026-09-24.md). The
cause: loadG4 built the layer's gemma4MoEWeights — which copies l.LayerScalar into its own layerScalar — BEFORE
it assigned l.LayerScalar (1, or blk.{i}.layer_output_scale.weight). Every MoE layer's output is multiplied by
that copy (forward_gemma4_moe.go: out = (h + comb) * layerScalar), so it was zeroed, and the argmax fell to
token 0. The .giw reader and the safetensors loader both set LayerScalar first, which is why only the direct GGUF
path broke — and why Metal's resident build (residency.go copies gm.layerScalar) would break the same way on a
direct GGUF load.

Pinned on the synthetic 26B-shaped GGUF (every layer MoE, layer_output_scale present and non-zero): the MoE copy
must equal the layer's own scalar, on every layer, and must not be zero.
```

## gguf_gemma4_stream_test.file

Moved from `decoder/gguf_gemma4_stream_test.go` (the comment above `gguf_gemma4_stream_test.file`) on 2026-10-09.

```text
S2 (task-never-swap-2026-09.md): gemma4 was the last family transcoded by building the whole model
resident and serializing it once — 34.7 GB RSS for the 26B-A4B, which no 16 GB Mac can build. It now
streams (build → write → release per layer). These gates pin that the streamed bundle is
byte-identical to the resident one on synthetic gemma4 GGUFs shaped like the two real layouts:

  - "26b": the 26B-A4B's parallel dense+MoE FFN (router, stacked gate‖up and down experts, the three
    extra norms) and a global layer with no attn_v (K=V, VFromK);
  - "e2b": the E-models' Per-Layer Embeddings (model-level per_layer_* inputs in the head, per-layer
    inp_gate/proj/post_norm) and a KV-shared tail layer, with per-layer FFN widths.

Both carry a sliding/global pattern with different head dims and KV-head counts per type, so every
per-layer geometry branch in loadG4 is exercised. The resident arm is exactly what
StreamTranscodeGGUF did for gemma4 before (resolve EOS, build with needCanonical, serialize), so the
comparison is against the behaviour being replaced. The quant label is excluded by construction: a
streamed body records "" (the layers do not exist yet when the head is written — see
writeHeadGlobals' B11 note), the resident one the resolved label.
```

## gguf_granite_permute_test.file

Moved from `decoder/gguf_granite_permute_test.go` (the comment above `gguf_granite_permute_test.file`) on 2026-10-09.

```text
Audit 2026-09-10 C-05: llama.cpp converts dense Granite (GraniteForCausalLM) through
GraniteModel(LlamaModel), which inherits undo_permute=True — so every GGUF whose
general.architecture is "granite" stores attn_q/attn_k rows in llama.cpp's interleaved RoPE
order. goinfer un-permuted only llama and mellum, so dense Granite loaded exact at position 0
and rotated the wrong pairs at every position after — fluent, and wrong, with no error.

These gates write a real (data-bearing) GGUF with q/k permuted by a line-for-line port of
llama.cpp's own permute() (conversion/llama.py), then load it through the real ggufConfig ->
resolveArchitecture -> buildWeightsFromGGUF path and compare rows. Rows, not logits: the defect
is a row order, so the row order is the thing to pin.
```

## tinyNormRopeGGUF.multipliers

Moved from `decoder/gguf_granite_permute_test.go` (the comment above `tinyNormRopeGGUF.multipliers`) on 2026-10-09.

```text
The REAL Granite 4.2 multipliers (granite-4.2-3b config.json). residual_scale must be 1.0:
every released 4.2 size ships it, and validateGraniteDense rejects anything else because
the generic forward has no residual hook. (A first draft used Granite 3.x's 0.22 and the
gate "failed" at resolveArchitecture, never reaching a row — a red that proved nothing.)
```

## TestGGUFConfig_everyArchitectureReadsMaxPositions

Moved from `decoder/gguf_maxpositions_test.go` (the comment above `TestGGUFConfig_everyArchitectureReadsMaxPositions`) on 2026-10-09.

```text
R13 (docs/measurements/cold-user-2026-09-07-macbook-arm64.md): found while wiring up the new
request-time memory guard (AdmitPrefillMemory, prefill_budget.go) — Config.MaxPositions came
back 0 for a real, freshly-downloaded qwen2.5-coder-0.5b GGUF, silently disabling both the new
guard and the pre-existing contextLengthError/clampMaxTokens checks (C-18/C-20). The cause: 16
of the 18 GGUF architecture config builders never read "context_length" into
Config.MaxPositions at all — only ggufPhi3Config did, by what looks like accident rather than
design (nothing about that family is special). This is a regression gate for every one of
them, driven through the real dispatch table (ggufConfig), not the individual functions, so a
future architecture that forgets the field fails here too.

Mutation: comment out any one architecture's `MaxPositions: u("context_length"),` line — that
architecture's subtest goes red with MaxPositions=0 (want 4096); every other subtest stays
green, isolating exactly which family broke.
```

## TestGGUFRouter_staysF32AtQuant

Moved from `decoder/gguf_router_quant_test.go` (the comment above `TestGGUFRouter_staysF32AtQuant`) on 2026-10-09.

```text
TestGGUFRouter_staysF32AtQuant gates M-27: several GGUF MoE loaders quantized the router
(ffn_gate_inp) via the generic mat() helper instead of routing it through
streamMat(..., quantNone, ...) like gpt-oss/qwen35 already did — CUDA/Metal residency
requires an f32 router (quantizing it flips which experts win near a tie, not just rounding
noise), so every affected family at a non-f32 quant silently declined resident build. This
fixture's shared generic loadLayer branch covers GLM/Mellum/Qwen3-MoE/DeepSeek2.
```

## TestStreamableFamilyClosures_onlyReadPerLayerTensors

Moved from `decoder/gguf_streaming_shape_test.go` (the comment above `TestStreamableFamilyClosures_onlyReadPerLayerTensors`) on 2026-10-09.

```text
task-never-swap-2026-09.md S2: a family's per-layer loader closure is safe to stream
(build -> sink.layer -> release, the loadQ35/loadGptOss shape) only if EVERY tensor it reads is
genuinely per-layer — named "blk.{i}." + something, never a bare/model-level name. gpt-oss's own
closure was verified this way by hand before its fix landed (byte-identical to the resident
path, on a real fixture — decoder/testdata/gptoss_tiny.gguf, internal/prequant's
TestGptOss_streamedMatchesResident). laguna, granite (the Mamba-2+MoE hybrid, arch.granite —
not the plain dense Granite family gguf_granite_permute_test.go covers), nemotron and llama4
have NO comparable fixture (no small, tokenizer-bearing, architecture-correct GGUF for any of
them exists in this repo or under ~/models at the time this landed) — building one per family
from scratch is real, separate work this pass did not do (a genuinely different Mamba-2/MoE
tensor set per family). This test is the structural half of the proof that DID ship with the
code: same technique stream_test.go's own TestTranscode_writesViaTempThenRenames already uses
for a property real execution can't force either — parse the source and check the invariant
directly, since the property this checks (nothing is read that would silently corrupt a
streamed bundle) IS provable from source, independent of a real checkpoint's numbers.

Verified, not assumed, when this landed: read every one of these four closures by hand,
confirmed zero non-per-layer tensor reads, THEN wrote this test to keep that true — not the
other way around (a test written first and never checked against the real code proves nothing).
```

## ggufQuantCosFloor

Moved from `decoder/gguf_tensorquant_test.go` (the comment above `ggufQuantCosFloor`) on 2026-10-09.

```text
ggufQuantCosFloor is the per-source-quant agreement floor a CORRECT loader must clear when
its f32 reconstruction is diffed against the bf16 safetensors reference.

These are dequant noise budgets, not quality targets. Each sits at roughly 2x the MEASURED
(1-cos) for that format, from TestQwen38GGUF_weightDiff over layers 0-3 of
unsloth/Qwen3.8-27B-GGUF UD-Q4_K_M against the bf16 safetensors (2026-09-12, 44 s,
goinfer-logs/qwen38-weightdiff-20260912-103901.log):

	quant  n   measured cosine      1-cos      floor   budget used
	F32    18  1.000000 (maxAbs 0)  0          0.999999  0%
	Q8_0    6  0.999982-0.999986    1.7e-5     0.9999    17%
	Q6_K    2  0.999742-0.999758    2.5e-4     0.9995    52%
	Q5_K    6  0.999186-0.999257    8.0e-4     0.998     40%
	Q4_K    5  0.996974-0.997047    3.0e-3     0.995     61%

What makes that table evidence rather than a curve fit: the cosine is a function of the
SOURCE QUANT ALONE. Q4_K spans 7e-5 across two different layer kinds (DeltaNet in_proj_qkv
and in_proj_z, softmax k_proj) and five tensor roles and shapes; Q5_K likewise. A transform
defect cannot produce agreement that tracks bit-width and ignores what the tensor is for.

The quants not present in that file (Q4_0/Q4_1/Q5_0/Q5_1, the IQ and TQ families, MXFP4)
are set from the same first-principles model the measured ones confirm to within 3e-4 — for
a k-quant with 32-element sub-blocks over roughly-Gaussian weights, a b-bit code has step
~4.2*sigma/(2^b-1), so relL2 ~ step/sqrt(12)/sigma and 1-cos ~ relL2^2/2, predicting Q4_K
0.9967 / Q5_K 0.99924 / Q6_K 0.99982 / Q8_0 0.99995 against the measurements above.

The margin matters less than it looks, because the defect class these gates exist for does
not produce a near-miss. A wrong un-tile order PERMUTES elements, a missing (1+w) norm
un-bake shifts every element by one, a sign error on -exp(A_log) inverts: all land near
zero or negative cosine, orders of magnitude below even the 2-bit floor. The floors only
have to sit above dequant noise and below "cratered", and that gap is enormous.

THAT CLAIM WAS MEASURED, NOT ASSUMED — a loosened gate that can no longer go red is worth
less than the tight one it replaced, and "it passes now" is exactly what that looks like.
Deleting ONE real transform (the untileVHeads on in_proj_z in decoder/gguf.go's loadQ35)
and rerunning took that tensor from 0.996974 to 0.045919 — 47704% of its budget, ~160x past
the floor — while every other tensor stayed green, so the failure named the one broken
transform. Dequant noise tops out at 61% of budget; a transform bug is three orders of
magnitude past it. Run 2026-09-12, goinfer-logs/qwen38-weightdiff-MUTATION-20260912-*.log.

An unrecognised quant deliberately gets the old whole-file bar: a format nobody has
calibrated must not silently widen a gate.
```

## giwalign_test.file

Moved from `decoder/giwalign_test.go` (the comment above `giwalign_test.file`) on 2026-10-09.

```text
A .giw's int4/int8 group SCALES used to be copied to the Go heap on every load (giwReader.f32
copies because the file did not align them): 3.75 GB of a streamed M35's 5.8 GB heap, and 3.0 GB of
the M26 Metal load's 4.4 GB. v12 pads every weight-matrix payload array to a 16-byte boundary
(giwWriter.alignArray) inside a v3 bundle whose blob starts at file offset 64, so the reader can
alias them out of the mapping. These are the gates: the scales really live in the mapping, an
older layout and a misaligned blob still load (by copying), and a corrupted layout is caught.
```

## TestGIWTargetForBackend_cpuArm64NeedsDotProd

Moved from `decoder/giwalign_test.go` (the comment above `TestGIWTargetForBackend_cpuArm64NeedsDotProd`) on 2026-10-09.

```text
TestGIWTargetForBackend_cpuArm64NeedsDotProd: the default CPU target on arm64 is cpu-arm64 (row4-only sidecars) only on a core that can read that layout. Found 2026-10-04 by the first
windows-arm64 CI run: aikit assumes no DotProd there, the default load wrote a cpu-arm64 sidecar its own core refused ("this core cannot use that layout"), and it was rebuilt on every start.
The test follows the core it runs on, so it asserts the right half on a DotProd Mac or Linux runner and on a no-DotProd one (QEMU Cortex-A72, windows-11-arm).
```

## giwf16_test.file

Moved from `decoder/giwf16_test.go` (the comment above `giwf16_test.file`) on 2026-10-09.

```text
Weights format v14 (metal target): every canonical group-32 int4 tensor also carries its group scales
pre-converted by F16Bits, so a Metal no-copy buffer can alias them instead of converting the f32 scales
into a new buffer (~389 MB on the 7B, S6). Singles are kind 7; kind-6 groups gain an f16 block. These pin
that the f16 arrays exist for exactly those tensors, equal the kernels' own conversion, live in the
mapping where the Metal build can find them, and that other targets and older files are unaffected.
```

## giwfitguard_test.file

Moved from `decoder/giwfitguard_test.go` (the comment above `giwfitguard_test.file`) on 2026-10-09.

```text
task-never-swap-2026-09.md S4 item 1: a .giw load's weights are file-backed (fitCheckFor is
never called for one, by design — its own srcFileBytes doc comment), but KV and prefill scratch
ARE real anonymous cost this path priced nowhere before. guardGIWFit is the load-time guard;
these are its own gates, mirroring fitguard_test.go's existing conventions for the .gguf side
(injectHostRAM, the same Config field shape) rather than inventing new ones.
```

## TestLoad_giwAutoPinsUnderTightMemory.cfg

Moved from `decoder/giwfitguard_test.go` (the comment above `TestLoad_giwAutoPinsUnderTightMemory.cfg`) on 2026-10-09.

```text
&w.Cfg directly — the actual struct Load will use internally — rather than hand-copying
fields into a fresh Config, which silently dropped HeadDim to 0 the first time this was
written (caught here: HeadDim=0 makes estimateKVBytes return 0 regardless of context,
so the "floor vs max" need never actually differed and this test could not have told
an auto-pin from a no-op).
```

## TestGlm4MoeAir.coherence

Moved from `decoder/glm4moe_air_test.go` (the comment above `TestGlm4MoeAir.coherence`) on 2026-10-09.

```text
AUDIT NOTE (da5a6ec): raw completion prompt on an instruction-tuned checkpoint
(GLM-4.5-Air), gated only by the distinct<3 floor below — which measures "did the
forward avoid TOTAL collapse", not coherence. On gemma-4-26b-a4b-it a raw prompt
manufactured a false "int4 is broken" signal that survived a week, and distinct<3
would not have caught it (repetition has >3 distinct tokens). This gate currently
passes, so the completion is in-distribution ENOUGH for this checkpoint — but when
it is next revalidated, adopt TestGemma4_26B_gate's pattern (render the family chat
template + distinctTrigramRatio floor) instead of trusting distinct<3.
```

## TestGlmOcr_residentDeclined

Moved from `decoder/glm_ocr_test.go` (the comment above `TestGlmOcr_residentDeclined`) on 2026-10-09.

```text
TestGlmOcr_residentDeclined: GLM-OCR rotates PAIRWISE (GPT-J) over m-RoPE sections, so a backend may
run it resident only if it declares FeatPairwiseRoPE and FeatPairwiseMRoPE, i.e. has pairwise rope
kernels. CUDA does (cuda/rope_pairwise.cu, gated by cuda.TestGlmOcrResidentParityCUDA), and Metal since
2026-10-09 (metal.TestGlmOcrResidentParityMetal); WebGPU still has only the NeoX half-split kernels, and
admitting glm_ocr there gave logit cosine -0.34 resident-vs-CPU on the CUDA twin of that kernel set
(2026-10-01), with no error. The decline must name the missing features so `serve check` shows the real cause.
```

## TestGlmOcr_generateQwenVL_cpuOnly

Moved from `decoder/glm_ocr_test.go` (the comment above `TestGlmOcr_generateQwenVL_cpuOnly`) on 2026-10-09.

```text
TestGlmOcr_generateQwenVL_cpuOnly: GenerateQwenVL, the entry point serve calls for an image turn, on a CPU-only load of
glm_ocr (m.resident == nil), through the SAME tiny image fixture and HF continuation as TestGlmOcr_mropeParity. The O1
review flagged that GenerateQwenVL's resident branches had never been checked for a nil resident. Every one of them sits
behind a type assertion on m.resident (ok is false on a nil interface) or behind tryClaimResident, so a CPU load takes the
CPU prefill + CPU decode and never touches a resident — this proves it by running the whole turn, twice with the same
image hash (the second call is where a reuse branch would be reached if it were reachable), and checks the tokens are
HF's and that no resident flag is set.
```

## TestGoldenNames_matchTheFileOnDisk

Moved from `decoder/golden_names_test.go` (the comment above `TestGoldenNames_matchTheFileOnDisk`) on 2026-10-09.

```text
Every real-checkpoint gate SKIPS when its golden is missing ("no golden — run the pin script"), so a
test that names x_golden.json while the tree holds x_golden.json.gz (or the reverse) does not fail — it
quietly stops gating. That is the one mistake the 2026-09-25 compression of the large goldens could
make, and a fixture-less run cannot see it, so this reads the source instead: for every golden name
quoted in a *_test.go anywhere in the repo, wherever the OTHER spelling exists on disk, this one must
too. A golden that is absent in both spellings (gitignored, not pinned on this box) is not its concern.
```

## readGolden

Moved from `decoder/goldenread_test.go` (the comment above `readGolden`) on 2026-10-09.

```text
readGolden reads a test golden, gunzipping it when the name ends in .gz. Goldens over ~1 MB are
committed compressed (CLAUDE.md — the convention for new goldens, and the backlog of 24 large ones
was converted 2026-09-25), so a reader goes through this instead of os.ReadFile: the same call
reads a small .json and a large .json.gz. A missing file is still an os.IsNotExist error, which is
what the callers' "no golden — run the pin script" skips test for.
```

## gpt2_test.regen

Moved from `decoder/gpt2_test.go` (the comment above `gpt2_test.regen`) on 2026-10-09.

```text
Regenerate:  ~/.venv-vl/bin/python scripts/pin_gpt2_real.py
(that script writes both the committed golden and the gitignored full-logit dump the
cosine reads; the previously-named pin_llama_forward.py no longer exists.)
```

## gptoss120_test.file

Moved from `decoder/gptoss120_test.go` (the comment above `gptoss120_test.file`) on 2026-10-09.

```text
gpt-oss-120b loader gate — does the safetensors path generalize past the 20b it was
written against?

WHAT THIS DOES AND DOES NOT CHECK, stated because the difference decides its value. The
forward math is already proven end to end on 20b (argmax-identical, cosine 0.999121 vs
the T3-validated GGUF reader). What 120b changes is SHAPE, not math: 36 layers instead
of 24, 128 experts instead of 32, and 14 shards instead of 2. So the risk this gate is
pointed at is EXPERT INDEXING at four times the count and tensors spanning many shards —
an off-by-N in the expert stride reads the wrong expert's weights while every shape check
still passes.

It deliberately does NOT build an HF reference. Four layers of 120b dequantize to ~51GB
in f32, which does not fit; claiming a numeric oracle here would mean pretending to a
check that cannot run. Instead it asserts geometry, spot-checks the HIGHEST expert index
(where a stride bug shows first and a low-index check would miss), and requires finite,
non-degenerate logits.
```

## TestGptOss_webgpuDecline

Moved from `decoder/gptoss_decline_test.go` (the comment above `TestGptOss_webgpuDecline`) on 2026-10-09.

```text
TestGptOss_webgpuDecline asserts the load-bearing guarantee (docs/task-mxfp4-gptoss.md §3/§6.4):
a backend that does NOT implement gpt-oss's novel ops must DECLINE it and fall back to CPU,
never mis-run it — and a backend that DOES ship and dispatch them must actually ADMIT, not stay
declined by a stale check. All three resident backends are now on the admit side; the name is
historical (kept so `git log -p` on it still tells the right story).

CUDA MOVED FROM THE DECLINE SIDE TO THE ADMIT SIDE on 2026-08-31 (G7): wiring the already-loaded
kernels found three silent defects no kernel test could see, because each was a term the WIRING
dropped rather than a kernel computing it wrongly:

	d9829ce  the gate‖up bias table indexed by SLOT id under expert caching
	610ce7f  the per-expert down bias never applied (needed gemv_w4a8_moe_wacc_bias)
	6cfb15c  route_gptoss never LOADED, so the router fell back to moe_route, which takes the
	         mixing weight from the UNBIASED score — same experts, different weights

The declaration rests on a real 20B forward, resident on an 8 GB card via --moe-cache-experts:
7/8 argmax-exact, min cosine 0.996392 (cuda.TestGptOssResidentParityCUDA). 2224441 declared
FeatAttnSink once on kernel-level evidence and was correctly reverted; this time the whole model
ran.

WEBGPU MOVED FROM THE DECLINE SIDE TO THE ADMIT SIDE on 2026-09-08 (G6,
docs/tasks/task-gpu-paths-2026-09.md): the sink threaded through every attention kernel (attn,
attn-keys, attn-f16, attn-i8, and all three wide variants — 7 pipelines), plus three brand-new
MoE kernels (gpt-oss disagrees with the generic MoE path on what the router bias means and what
the activation clamps — routeGptOssWGSL/gptossGluQuantWGSL/moeExpertGptOssDownGEMVWGSL,
gpu/moe.go). Verified against the real (non-seeded) decoder/testdata/gptoss_tiny.gguf, resident
vs CPU: min cosine 0.9943 across 8 positions (gpu.TestGptOssResidentParityWebGPU) — the same
0.95 floor Metal's own gate uses on this fixture.
```

## TestGptOss_webgpuDecline.admit

Moved from `decoder/gptoss_decline_test.go` (the comment above `TestGptOss_webgpuDecline.admit`) on 2026-10-09.

```text
A backend that ships AND DISPATCHES the kernels must actually admit, not stay declined by a
stale check — which is the failure this test was written to catch in the other direction
(and did, twice: CUDA in 2026-08-31, WebGPU in 2026-09-08).
```

## TestGptOss_webgpuDecline.archgate

Moved from `decoder/gptoss_decline_test.go` (the comment above `TestGptOss_webgpuDecline.archgate`) on 2026-10-09.

```text
The arch-shape gate falls through for gpt-oss (2026-08-18, mirroring gemma4 and the
GPT-2 NonGatedMLP/LearnedPosEmbed/OutBias precedent): the decline lives at the feature gate
(FeatAttnSink, asserted above), not here — a backend that implements the sink/clamped-SwiGLU
kernels admits through ResidentEligible without this arch predicate needing a second edit.
```

## TestGptOssReal.assetPath

Moved from `decoder/gptoss_real_test.go` (the comment above `TestGptOssReal.assetPath`) on 2026-10-09.

```text
assetPath, not a hand-rolled env+fallback: the asset registry is what makes the
gate and the sweep preflight apply the SAME predicate to the same candidate paths
(testdata/assets.json). This file predated GOINFER_GPTOSS_GGUF being registered.
```

## TestGptOssReal_logitParity.assetPath

Moved from `decoder/gptoss_real_test.go` (the comment above `TestGptOssReal_logitParity.assetPath`) on 2026-10-09.

```text
assetPath, not a hand-rolled env+fallback: the asset registry is what makes the
gate and the sweep preflight apply the SAME predicate to the same candidate paths
(testdata/assets.json). This file predated GOINFER_GPTOSS_GGUF being registered.
```

## TestGraniteReal.assetPath

Moved from `decoder/granite_real_test.go` (the comment above `TestGraniteReal.assetPath`) on 2026-10-09.

```text
assetPath, not a hand-rolled env+fallback: the registry is what makes this gate and the
sweep preflight apply the SAME predicate to the same candidate paths. Required by the sweep
since 2026-09-02, and a required gate whose presence check disagrees with the preflight's is
a SKIP nobody can attribute.
```

## TestGraniteReal.coherence

Moved from `decoder/granite_real_test.go` (the comment above `TestGraniteReal.coherence`) on 2026-10-09.

```text
AUDIT NOTE (da5a6ec): raw completion prompt on an instruction-tuned checkpoint
(granite-4.0-h), gated only by the distinct<3 floor below — which measures "did the
forward avoid TOTAL collapse", not coherence. On gemma-4-26b-a4b-it a raw prompt
manufactured a false "int4 is broken" signal that survived a week, and distinct<3
would not have caught it (repetition has >3 distinct tokens). This gate currently
passes, so the completion is in-distribution ENOUGH for this checkpoint — but when
it is next revalidated, adopt TestGemma4_26B_gate's pattern (render the family chat
template + distinctTrigramRatio floor) instead of trusting distinct<3.
```

## TestGraniteReal.nope

Moved from `decoder/granite_real_test.go` (the comment above `TestGraniteReal.nope`) on 2026-10-09.

```text
NoPE on the GGUF path. The Q8_0 convert carries the same base weights as the
safetensors release, so the bf16 golden is a valid reference for it too — and it is
the ONLY check that the rope.scaling.finetuned → NoPE mapping in ggufGraniteConfig is
right, since llama.cpp writes rope.dimension_count/freq_base on this model regardless.
Roped, this reads ~0.9936 with a diverging continuation; NoPE, ~0.9958 and exact. Not
a parity row: the T3 row is the safetensors oracle below, on weights HF actually ran.
```

## TestGraniteReal_oracle

Moved from `decoder/granite_real_test.go` (the comment above `TestGraniteReal_oracle`) on 2026-10-09.

```text
TestGraniteReal_oracle is the T3 row: the released bf16 safetensors loaded at int8 and
matched against an HF bf16 forward of the SAME weights. The gate above runs on a
DIFFERENT artifact (a llama.cpp Q8_0 convert), and until this golden existed it had no
reference to compare against at all — which is why granitemoehybrid sat at `pending`
while having a passing "real gate": coherent-generation is not a T3 method. The row is
recorded here, on the weights HF actually ran.

Fixture: scripts/pin_granite_real.py.
```

## TestGraniteResidentBridge

Moved from `decoder/granite_resident_test.go` (the comment above `TestGraniteResidentBridge`) on 2026-10-09.

```text
TestGraniteResidentBridge validates the P5b.1 accessor bridge: the resident SSM builder
reads the Mamba geometry, per-layer mixer-kind, and correctly-shaped f32 mixer tensors
through these accessors. Shape-checks the weights against the projDim/convDim/dInner the
kernels expect, so a wiring error surfaces here before any GPU dispatch. Additive — no
production routing change (eligibility flip is P6).
```

## TestHardwareMatrix_fresh.crlf

Moved from `decoder/hardware_matrix_test.go` (the comment above `TestHardwareMatrix_fresh.crlf`) on 2026-10-09.

```text
stripCR before comparing: the generated md is always LF (Go "\n" literals), but a Windows
checkout with no eol pin stores the committed doc as CRLF, so a bare bytes.Equal would report
it "stale" on line endings alone. .gitattributes pins these docs to eol=lf; this is the
other half, so the gate passes under any git autocrlf setting. (Windows CI would trip it —
goinfer's is ubuntu+macos only, so this is a latent bug, not an active one.)
```

## stripCR

Moved from `decoder/hardware_matrix_test.go` (the comment above `stripCR`) on 2026-10-09.

```text
stripCR removes carriage returns so a generated-doc freshness comparison is line-ending agnostic
(mirrors aikit cfda968). Shared by the hardware- and capability-matrix gates.
```

## requireHeavyModel

Moved from `decoder/heavytest_test.go` (the comment above `requireHeavyModel`) on 2026-10-09.

```text
requireHeavyModel gates a test that loads a multi-GB checkpoint from ~/models
behind an explicit env opt-in.

The bug this closes: several tests decided whether to run by PATH EXISTENCE
alone (os.Stat(~/models/...) → skip if absent, else Load()). On a box with the
model zoo present, `go test ./decoder/` therefore fired all of them
opportunistically — the EAGLE group alone reloaded qwen3-1.7b (+ the 785 MB
head) once PER TEST, climbing to tens of GB RSS and thrashing swap, never
finishing. Path auto-detection is not a gate; the asset happening to exist is
not a request to run a multi-GB test.

Policy (same shape as GOINFER_MOE_GIW / GOINFER_SSM_RESIDENT): a test that
loads a multi-GB model must OPT IN via GOINFER_HEAVY_TESTS. Default `go test
./decoder/` skips them regardless of what is on disk, so the package is
runnable everywhere; set GOINFER_HEAVY_TESTS=1 when you mean to run them. The
per-test os.Stat skip stays as a second guard, so opting in on a bare box is
still harmless.
```

## sampleVMStat

Moved from `decoder/hostram_darwin_test.go` (the comment above `sampleVMStat`) on 2026-10-09.

```text
R13-follow-on (docs/measurements/cold-user-2026-09-07-macbook-arm64.md's live re-run): real
vm_stat output shape, 16 KB pages (Apple Silicon — the M1 Pro that found this bug, not the
traditional 4 KB), so a parser tested only against a 4 KB assumption would pass here and still
be wrong on the machine that matters.
```

## TestInt4_forwardParity.file

Moved from `decoder/int4_golden_test.go` (the comment above `TestInt4_forwardParity.file`) on 2026-10-09.

```text
Q1(c) — the int4 forward goldens.

int4 is the runtime's documented default quantization and, until this file, NOTHING gated it.
Every golden that ran was f32: the three int8int8 goldens skipped for want of an env var (fixed,
23b2ee7) and the one int8 golden sits behind `//go:build realckpt` plus a missing checkpoint. So
`scripts/refresh_parity_hashes.sh` — the sanctioned freeze-exception path, and the only numeric
proof a `decoder/` core edit gets — proved f32 numerics and silently nothing else.

WHAT AN int4 GOLDEN IS, and what it deliberately is NOT.

It is **int4 output compared against recorded int4 output**. It is NOT int4 compared to f32 within
a tolerance. That distinction is the whole point: a tolerance band against f32 measures how lossy
the quantizer is, which is a real question with its own gate (the policy's quant axis:
quant-vs-f32 argmax + cosine), and it would read on a dashboard as "int4 is covered" while proving
nothing about whether the int4 CODE PATH still computes what it computed yesterday. Only a
self-comparison catches a regression in the W4A8 path itself, which is what the freeze is
protecting and what P7 will change.

The comparison is argmax-exact plus a tight value tolerance on recorded samples and moments —
matching the existing goldens' treatment, and for the same reason: the CPU reference is
bit-identical WITHIN an architecture but not across one (see parity-coverage-policy.md), so a
bit-exact checksum would be a machine assertion rather than a code assertion.

SCOPE, measured before authoring rather than described after: 23 fixtures across 16 model_types
load at int4 today. int4 has no divisibility constraint (`nGroups` is a ceiling divide), so the
limit is which fixtures exist, not which families are eligible. Recorded absences, which are NOT
counted as gaps: `gpt_oss` is MXFP4-prequant and rejects a conflicting `--quant` by design;
`siglip_vision_model` is an encoder, not a decoder; `gpt2`, `mellum`, `qwen2` and `qwen3` have no
tiny safetensors fixture; `qwen2_moe` and the `gemma4-dense-scaled-{24,48,64}` variants have
incomplete fixture dirs (no config.json).

Regenerate with: GOINFER_INT4_GOLDEN_UPDATE=1 go test ./decoder/ -run TestInt4_forwardParity
```

## gpt2CosFloor

Moved from `decoder/int4_golden_test.go` (the comment above `gpt2CosFloor`) on 2026-10-09.

```text
gpt2 is the ONLY real, fully-trained checkpoint this gate runs (every other fixture is a
tiny/near-random test config — e.g. qwen35-tiny and phi3-tiny are both hidden_size=64,
vocab_size in the hundreds — which quantizes far more cleanly than real learned weights
with outlier features). Bisected 2026-08-22 (ForwardCapture, per-layer, same box, no cross-
arch involved): int4-vs-f32 relative hidden-state error is already ~5-10% at LAYER 0 and
stays in a 4-20% band through all 12 layers; int8-vs-f32 shows the SAME shape at ~7-8x
SMALLER error at every layer — exactly the expected 4-bit-vs-8-bit precision ratio. That is
the signature of ordinary (if large) round-to-nearest int4 quantization noise on a real
trained model, not a logic bug — naive int4 without GPTQ/AWQ-style calibration is
documented to do this to real checkpoints. Holding gpt2 to the same tight per-sample
absolute gate as the tiny fixtures was never appropriate; argmax-exactness + a floor on
centered cosine similarity (the same bar TestGPT2_forwardParity's f32 golden uses) is.
Mutation-checked (int4GroupSize 32->64, decoder/weightmat.go): that mutation moves individual
samples by up to 7.39 while argmax stays unchanged, so only a cosine floor catches it.

Re-calibrated 2026-08-26. The previous calibration ("baseline 0.9995 on both, mutation 0.9938",
2026-08-22) was measured against goldens that a11c56b REPLACED the next day, when the int4 LM
head moved weight-only-Q8 -> full W8A8. The current goldens are arm64-baked, so amd64 carries a
fixed FMA-contraction offset and the stale 0.999 floor failed a HEALTHY Linux box. All four
numbers below re-measured at c5ae3c1 on both boxes, golden restored after every run:

	                arm64 (Mac)     amd64 (Linux)
	  healthy       1.0000000000    0.9977745721   <- golden is arm64-baked, so exact there
	  32->64 mut.   0.9761066556    0.9761066320   <- mutation dwarfs the cross-arch offset

argmax is unaffected by the cross-arch offset (16 on both) and is gated separately above.
0.995 sits between with balanced margin: 2.2x the healthy amd64 deficit (2.23e-3), and it fails
the mutation by 4.8x (2.39e-2). Do NOT raise this toward 0.999 without baking per-arch goldens
first -- 0.999 is BELOW the natural cross-arch baseline and fails a correct build.
```

## TestInt4_forwardParity.forced

Moved from `decoder/int4_golden_test.go` (the comment above `TestInt4_forwardParity.forced`) on 2026-10-09.

```text
A forced narrower kernel is another numeric realization of the same arithmetic, so the recorded samples cannot be matched to 5e-3 (H1.3: gemma4-dense-scaled reads 0.9938). The argmax above
is still exact; the samples are held to a centered-cosine floor, the way gpt2 is (see forced_fallbacks_test.go for the measurement).
```

## TestGemma4EModel_safetensorsParity

Moved from `decoder/gemma4_emodel_test.go` (the comment above `TestGemma4EModel_safetensorsParity`) on 2026-10-09.

```text
TestGemma4EModel_safetensorsParity is S1.1's gate (docs/tasks/task-multimodal-support-2026-10.md): a
safetensors Gemma 4 E-model — Per-Layer Embeddings, cross-layer KV sharing, double-wide FFNs on the shared
layers — loads (until 2026-10-06 this loader refused any PLE checkpoint) and its CPU forward matches the HF
oracle at EVERY prompt position, f32: argmax equal and cosine >= 0.99999 at each, then the greedy
continuation byte-identical. All positions, not just the last, because the KV-shared layers only differ
from their own-KV counterparts once there is history to attend over, and the sliding window (4) is
shorter than the prompt (12).
```

## TestGemma4Load

Moved from `decoder/gemma4_load_test.go` (the comment above `TestGemma4Load`) on 2026-10-09.

```text
TestGemma4Load is Increment 2: the full weight load. It exercises the gemma4
loader against the real E2B GGUF — per-layer head_dim/FFN, the KV-shared tail
(layers ≥ 15 carry no k/v), and the PLE tensors. int4 keeps it light on 16 GB.
(Forward pass is Increment 3, so this checks the bundle, not generation.)
```

## TestGemma4_logitParity

Moved from `decoder/gemma4_parity_test.go` (the comment above `TestGemma4_logitParity`) on 2026-10-09.

```text
TestGemma4_logitParity is the Increment-3 gate: goinfer's gemma4 forward (int8
quantized GGUF) vs the HF bf16 oracle. Argmax must match (the correctness
gate); cosine over the sampled-256 logits must clear the quant-vs-bf16 bar.
Skips without the golden or the GGUF asset.
```

## gguf_qwen35_test.file

Moved from `decoder/gguf_qwen35_test.go` (the comment above `gguf_qwen35_test.file`) on 2026-10-09.

```text
STEP 1 (de-risk the crux, zero download): prove the inverse V-head reorder is
bit-clean. llama.cpp's converter tiles HF's grouped-by-K V heads when
num_v_heads > num_k_heads; the GGUF loader must undo it exactly or it silently
corrupts every linear-attn layer. We replicate the converter's FORWARD with an
INDEPENDENT permutation-index reference (gather, not the production copy loop),
then require untileVHeads to recover the original bit-for-bit across every
tensor block geometry the loader will touch — with the real ratio num_k=16,
num_v=32 (num_v_per_k=2), head_dim=128.
```

## TestInt4MixMode

Moved from `decoder/int4mix_test.go` (the comment above `TestInt4MixMode`) on 2026-10-09.

```text
TestInt4MixMode gates the per-tensor mixed-precision mode (idea #5): -quant int4mix
keeps attention (q/k/v/o) at int8 — where the spike found the int4→int8 quality loss
concentrated — and the FFN bulk (gate/up/down) at int4. Asserts the per-tensor kinds,
that Quant() reports it distinctly (so the KV fingerprint won't collide with int4/int8),
that it generates, and that a mixed model round-trips through the .giw serialize format.
```

## sampleMeminfo

Moved from `decoder/hostram_linux_test.go` (the comment above `sampleMeminfo`) on 2026-10-09.

```text
R13-follow-on (docs/measurements/cold-user-2026-09-07-macbook-arm64.md's live re-run of the R13
fix): HostRAMAvailableBytes exists because HostRAMBytes (total physical RAM) is what the fit
guard budgeted against, and a fixed fraction of TOTAL RAM assumes nothing else on the machine
ever needs more than the rest — false on a real machine with a browser or IDE open. This tests
the shared meminfo-field parser both readers now use.
```

## TestGemma4_12B_trace

Moved from `decoder/gemma4_12btrace_test.go` (the comment above `TestGemma4_12B_trace`) on 2026-10-09.

```text
TestGemma4_12B_trace drives the 12B GGUF over the HF trace's prompt and dumps a
last-position per-layer residual trace + the layer 4/5 q/k/v, for an offline
cosine diff against the HF reference (scripts/diff_gemma4_12b.py). Debug aid for
the parked K=V forward — gated on G4_TRACE so it never runs in normal CI.
```

## TestFitEstimate_agreesWithResidentWeightBytes

Moved from `decoder/fitguard_test.go` (the comment above `TestFitEstimate_agreesWithResidentWeightBytes`) on 2026-10-09.

```text
The estimator must not be free to drift from the accountant M-01 completed. It prices the model
from GGUF metadata BEFORE the load; ResidentWeightBytes sums the matrices AFTER it. They answer
the same question from opposite sides, so a large disagreement means one of them is wrong.
```

## TestGemma4RunLayersFromEmbed_matchesTokenPath

Moved from `decoder/gemma4_embed_test.go` (the comment above `TestGemma4RunLayersFromEmbed_matchesTokenPath`) on 2026-10-09.

```text
TestGemma4RunLayersFromEmbed_matchesTokenPath proves runLayersGemma4FromEmbed
(the new P7 embed-by-vector seam) is behavior-preserving for the ordinary
text path: feeding it the SAME already-scaled embedding runLayersGemma4
itself builds, with pleTokenID equal to the real token id, must reproduce
runLayersGemma4's own output bit-for-bit (they now share every line after
the embedding prologue — see forward_gemma4.go).

Uses the real E2B GGUF (the only local fixture with PLE enabled —
hidden_size_per_layer_input=256; every tiny synthetic fixture in this
package has PLE off), so it is heavy-gated like the other real-checkpoint
gemma4 gates in this file's neighbors.
```

## TestGenerateVL_streams.resident

Moved from `decoder/generate_vl_test.go` (the comment above `TestGenerateVL_streams.resident`) on 2026-10-09.

```text
V-11 (docs/review-2026-09-04.md): this model has no resident backend (Options{}, plain
CPU), so GenerateVL's resident-touching branches (gap 0, P9a) never engage at all — a
pre-existing resIDs describing an UNRELATED resident generation must survive this call
untouched, exactly as it did before either of those existed.
```

## TestFitGuard_refusesBeforeAllocating.message

Moved from `decoder/fitguard_test.go` (the comment above `TestFitGuard_refusesBeforeAllocating.message`) on 2026-10-09.

```text
The message has to carry the remedy and the arithmetic, because the user who reads it is
the one who has no idea -stream-weights exists — that was the whole finding.
```
