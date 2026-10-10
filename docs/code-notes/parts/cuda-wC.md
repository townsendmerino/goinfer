# cuda: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `cuda`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

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
