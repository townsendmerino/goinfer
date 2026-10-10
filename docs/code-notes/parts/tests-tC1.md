# cuda: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `cuda`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

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
