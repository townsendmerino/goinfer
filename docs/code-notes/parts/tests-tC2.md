# cuda: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `cuda`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

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
