//go:build cuda && goinfer_testhooks

package cuda

import (
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"testing"

	gpu "github.com/townsendmerino/aikit/gpu"
)

// A9 reopened here. The reservation is a confirmed COST and was not yet a confirmed CAUSE, because
// the arithmetic does not close:
//
//	free immediately before the failing launch   198,836,224 B
//	measured moe_route reservation               138,412,032 B
//	spare                                         60,424,192 B
//
// The reservation fits, and the launch failed anyway. Worse, free after the failure was 265,945,088
// — 67,108,864 B ABOVE the pre-attempt level. An unwind returns to the pre-attempt level; it cannot
// exceed it. So something that existed BEFORE the attempt was released, which reads as the driver
// trimming a cache to satisfy a request it still could not satisfy. If that is right the true demand
// is above 265,945,088 and the 132 MiB reservation is one component of it.
//
// This measures the demand directly instead of inferring it: balloon the device to leave a chosen
// number of bytes free, launch moe_route, and binary-search the pass/fail boundary.
//
// PRE-REGISTERED readings:
//
//	threshold ~= 138,412,032           the reservation is the whole demand, and the 34-slot failure
//	                                   needs a different explanation entirely
//	198,836,224 < threshold <= 265,945,088   consistent with the observed failure; the reservation is
//	                                   one component and the remainder needs naming
//	threshold > 265,945,088            demand exceeds even the post-trim free, and the trim behaviour
//	                                   is part of the mechanism
//
// If the result VARIES run to run at the same balloon size, that is contiguity rather than capacity
// and it is a different finding. Contiguity was refuted earlier in this campaign against a different
// observation (a fresh heap had worse contiguity than the slot-loaded one at equal free); that
// refutation was about slot buffers and does not carry here.

const a9ChildEnv = "GOINFER_A9_LEAVE_FREE"

// TestMoERouteDemandThresholdChild is the per-trial worker. Each trial needs a FRESH context,
// because once moe_route's backing store is reserved it stays reserved for the life of the context
// — a second trial in the same process would measure nothing. It reports on stdout and exits 0 on
// both outcomes: a launch failure is the measurement, not an error.
func TestMoERouteDemandThresholdChild(t *testing.T) {
	want, err := strconv.ParseInt(os.Getenv(a9ChildEnv), 10, 64)
	if err != nil {
		t.Skipf("%s unset — this is the worker half of TestMoERouteDemandThreshold", a9ChildEnv)
	}
	dev, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	read := func() int64 {
		f, _, e := dev.Context().MemInfo()
		if e != nil {
			t.Fatalf("MemInfo: %v", e)
		}
		return int64(f)
	}
	// Balloon in chunks. A single multi-GB request can fail for reasons that have nothing to do with
	// the quantity under test, and chunking also lets the last chunk land the remainder precisely.
	// gpu allocation PANICS on failure, so each chunk is guarded — a failed chunk means we have
	// ballooned as far as this heap allows, which is a legitimate stopping point, not a test error.
	var hold []gpu.Buffer
	alloc := func(n int) (ok bool) {
		defer func() {
			if recover() != nil {
				ok = false
			}
		}()
		hold = append(hold, gpu.NewBufferLenOf[byte](dev, n))
		return true
	}
	// Everything the launch needs is allocated BEFORE ballooning: the module, the pipeline, and the
	// four small buffers. That is both the right shape (in production they exist long before the
	// launch) and the fix for a real bug in the first version, which ballooned first and then could
	// not allocate 32 bytes for an argument buffer.
	mod, err := dev.CompileLibrary(moePTXOrOverride())
	if err != nil {
		t.Fatalf("CompileLibrary(moePTX): %v", err)
	}
	p, err := dev.NewComputePipeline(mod, "moe_route")
	if err != nil {
		t.Fatalf("NewComputePipeline(moe_route): %v", err)
	}
	rLogits := gpu.NewBufferLenOf[float32](dev, 8)
	rBias := gpu.NewBufferLenOf[float32](dev, 8)
	rIdx := gpu.NewBufferLenOf[uint32](dev, 2)
	rWgt := gpu.NewBufferLenOf[float32](dev, 2)

	// Census mode launches EVERY entry point the local-memory census flags, not just moe_route.
	// The margin has to cover the whole set, and whether the driver sums the backing stores or
	// shares one sized by the largest is a property of the driver that must be measured rather than
	// assumed — Sigma and max differ by a factor here.
	census := os.Getenv("GOINFER_A9_KERNELS") == "census"
	var pRope, pRopeB Pipeline
	var rq, rk, rv, rInv, rKc, rVc gpu.Buffer
	if census {
		gm, e := dev.CompileLibrary(gemvFwdPTX)
		if e != nil {
			t.Fatalf("CompileLibrary(gemv_fwd): %v", e)
		}
		if pRope, e = dev.NewComputePipeline(gm, "rope_kv"); e != nil {
			t.Fatalf("rope_kv: %v", e)
		}
		pb, e := dev.CompileLibrary(prefillBatchedPTX)
		if e != nil {
			t.Fatalf("CompileLibrary(prefill_batched): %v", e)
		}
		if pRopeB, e = dev.NewComputePipeline(pb, "rope_kv_batched"); e != nil {
			t.Fatalf("rope_kv_batched: %v", e)
		}
		rq, rk, rv = gpu.NewBufferLenOf[float32](dev, 64), gpu.NewBufferLenOf[float32](dev, 64), gpu.NewBufferLenOf[float32](dev, 64)
		rInv, rKc, rVc = gpu.NewBufferLenOf[float32](dev, 64), gpu.NewBufferLenOf[float32](dev, 64), gpu.NewBufferLenOf[float32](dev, 64)
	}

	// Back off on failure rather than stopping. A failed request does not mean the heap is full —
	// it means THAT SIZE does not fit, which is a statement about contiguity, not capacity. Halving
	// until the quantum is reached is what actually drains the pool; the first version gave up on
	// the first refusal and left 307 MiB unballooned against a 64 MiB target. The bracket check in
	// the parent caught that, which is the only reason it is not silently in the numbers below.
	// Balloon SHAPE is a variable, not a detail. A deterministic balloon produces a deterministic
	// heap layout, so identical repeats do NOT by themselves exclude contiguity — they only exclude
	// run-to-run noise. Filling with many small blocks instead of a few large ones leaves the same
	// free BYTES in a very different arrangement; if the threshold is capacity it should barely
	// move, and if it is contiguity it should.
	if os.Getenv("GOINFER_A9_BALLOON") == "fine" {
		for {
			rem := read() - want
			if rem < (2 << 20) {
				break
			}
			if !alloc(2 << 20) {
				break
			}
		}
	} else {
		for size := int64(1) << 30; size >= (2 << 20); {
			rem := read() - want
			if rem <= 0 {
				break
			}
			if size > rem {
				size = rem
			}
			if !alloc(int(size)) {
				size /= 2
			}
		}
	}

	q := dev.NewCommandQueue()
	one := LaunchConfig{GridX: 1, GridY: 1, GridZ: 1, BlockX: 1, BlockY: 1, BlockZ: 1}
	before := read()

	// A10 discriminator: launch ONLY a kernel with zero declared local memory, from a freshly loaded
	// module. If the ~151 MiB floor is still there, it is per-module or per-context and has nothing
	// to do with local-memory backing; if it vanishes, it is part of backing-store setup.
	if os.Getenv("GOINFER_A9_KERNELS") == "zerolocal" {
		zp, ze := dev.NewComputePipeline(mod, "shared_gate_combine")
		if ze != nil {
			t.Fatalf("shared_gate_combine: %v", ze)
		}
		zerr := q.Launch(zp, one, Arg(rLogits), Arg(rBias), Arg(rIdx),
			gpu.ArgValue(int32(1)), gpu.ArgValue(int32(1)))
		zs := q.Sync()
		zafter := read()
		fmt.Printf("A9CHILD ok=%t freeBefore=%d freeAfter=%d err=%q\n",
			zerr == nil && zs == nil, before, zafter, strings.TrimSpace(fmt.Sprintf("%v|%v", zerr, zs)))
		return
	}

	lerr := q.Launch(p, one,
		Arg(rLogits), Arg(rBias), Arg(rIdx), Arg(rWgt),
		gpu.ArgValue(int32(8)), gpu.ArgValue(int32(2)), gpu.ArgValue(int32(1)),
		gpu.ArgValue(int32(0)), gpu.ArgValue(float32(1)),
		gpu.ArgValue(int32(1)), gpu.ArgValue(int32(1)))
	if census && lerr == nil {
		// nH=1, nKV=1, hd=2, rhalf=1, pos=0 keeps idx=0 on the first branch, touching q[0..1] and
		// invFreq[0] only. 64-float buffers are far larger than anything reachable.
		lerr = q.Launch(pRope, one, Arg(rq), Arg(rk), Arg(rv), Arg(rInv), Arg(rKc), Arg(rVc),
			gpu.ArgValue(int32(1)), gpu.ArgValue(int32(1)), gpu.ArgValue(int32(2)),
			gpu.ArgValue(int32(0)), gpu.ArgValue(int32(1)))
	}
	if census && lerr == nil {
		lerr = q.Launch(pRopeB, one, Arg(rq), Arg(rk), Arg(rv), Arg(rInv), Arg(rKc), Arg(rVc),
			gpu.ArgValue(int32(1)), gpu.ArgValue(int32(1)), gpu.ArgValue(int32(2)),
			gpu.ArgValue(int32(0)), gpu.ArgValue(int32(1)), gpu.ArgValue(int32(1)))
	}
	serr := q.Sync()
	after := read()
	ok := lerr == nil && serr == nil
	// Single machine-readable line, so the parent parses one thing and a change to the human log
	// cannot silently break the search.
	fmt.Printf("A9CHILD ok=%t freeBefore=%d freeAfter=%d err=%q\n", ok, before, after,
		strings.TrimSpace(fmt.Sprintf("%v|%v", lerr, serr)))
	_ = hold
}

type a9Trial struct {
	ok                  bool
	freeBefore, freeAft int64
}

func a9Run(t *testing.T, leave int64) a9Trial {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=TestMoERouteDemandThresholdChild", "-test.timeout=5m")
	cmd.Env = append(os.Environ(), a9ChildEnv+"="+strconv.FormatInt(leave, 10))
	out, err := cmd.CombinedOutput()
	for ln := range strings.SplitSeq(string(out), "\n") {
		if !strings.HasPrefix(ln, "A9CHILD ") {
			continue
		}
		var tr a9Trial
		for _, kv := range strings.Fields(ln)[1:] {
			k, v, _ := strings.Cut(kv, "=")
			switch k {
			case "ok":
				tr.ok = v == "true"
			case "freeBefore":
				tr.freeBefore, _ = strconv.ParseInt(v, 10, 64)
			case "freeAft", "freeAfter":
				tr.freeAft, _ = strconv.ParseInt(v, 10, 64)
			}
		}
		return tr
	}
	t.Fatalf("child produced no A9CHILD line (err=%v)\n%s", err, out)
	return a9Trial{}
}

// TestMoERouteDemandThreshold binary-searches the free-VRAM level at which moe_route's first launch
// starts to fail. The x-axis is the MEASURED free immediately before the launch, not the requested
// balloon target — allocation granularity means those differ, and the measured one is the quantity
// the claim is about.
func TestMoERouteDemandThreshold(t *testing.T) {
	if os.Getenv(a9ChildEnv) != "" {
		t.Skip("running as the child worker")
	}
	// MARKED AS A DRAINER, and found by the gate rather than by the derivation — see the note in
	// cuda/drain_marker_test.go. The bisection deliberately balloons the device to leave as little as
	// 64 MiB (below the 144 MiB floor) and records the resulting refusal as data: `bracket low: leave
	// 67108864 -> ok=false` IS a refusal, driven on purpose. It balloons through child processes, so
	// each child's memory is returned when it exits, but a child that fails or hangs leaves the
	// device at the floor for whatever runs next in this process — which is exactly what the log
	// showed: the following test opened with `free at start 151191552 B`.
	drainsDevice(t, "bisects by ballooning the device to as little as 64 MiB free, refusals included")
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	// Bracket. Low must fail and high must pass, and BOTH are checked rather than assumed — a
	// binary search over a bracket whose ends were never verified reports a boundary that may be
	// outside it.
	const lo, hi = 64 << 20, 1024 << 20
	loT, hiT := a9Run(t, lo), a9Run(t, hi)
	t.Logf("bracket low  : leave %d -> free %d, ok=%t", lo, loT.freeBefore, loT.ok)
	t.Logf("bracket high : leave %d -> free %d, ok=%t", hi, hiT.freeBefore, hiT.ok)
	if loT.ok {
		t.Fatalf("bracket invalid: the launch SUCCEEDED with only %d B free, so the threshold is "+
			"below the low end and the search would report a boundary it never contained", loT.freeBefore)
	}
	if !hiT.ok {
		t.Fatalf("bracket invalid: the launch FAILED with %d B free, so the threshold is above the "+
			"high end", hiT.freeBefore)
	}

	// Bisect to the allocation quantum. Finer than 2 MiB is not meaningful: the balloon cannot
	// place the boundary more precisely than the driver's granularity.
	l, h := int64(lo), int64(hi)
	var lastFail, firstPass a9Trial
	lastFail, firstPass = loT, hiT
	for h-l > (2 << 20) {
		mid := (l + h) / 2
		tr := a9Run(t, mid)
		t.Logf("  probe leave %10d -> free %10d  ok=%t  freeAfter=%d", mid, tr.freeBefore, tr.ok, tr.freeAft)
		if tr.ok {
			h, firstPass = mid, tr
		} else {
			l, lastFail = mid, tr
		}
	}
	t.Logf("THRESHOLD: highest observed FAIL at free=%d B (%.1f MiB); lowest observed PASS at "+
		"free=%d B (%.1f MiB)", lastFail.freeBefore, float64(lastFail.freeBefore)/(1<<20),
		firstPass.freeBefore, float64(firstPass.freeBefore)/(1<<20))
	t.Logf("  measured moe_route reservation      138412032 B")
	t.Logf("  free before the 26B failing launch  198836224 B")
	t.Logf("  free after  the 26B failure         265945088 B")
	if firstPass.freeAft > 0 {
		t.Logf("  at the lowest PASS: free %d -> %d, steady-state cost %d B (%.1f MiB), while the "+
			"launch needed %d B (%.1f MiB) to get there — the launch's PEAK demand is %.2fx its "+
			"residual cost, and the difference is transient and unnamed",
			firstPass.freeBefore, firstPass.freeAft, firstPass.freeBefore-firstPass.freeAft,
			float64(firstPass.freeBefore-firstPass.freeAft)/(1<<20), firstPass.freeBefore,
			float64(firstPass.freeBefore)/(1<<20),
			float64(firstPass.freeBefore)/float64(firstPass.freeBefore-firstPass.freeAft))
	}

	// Balloon-shape control: same free bytes, different arrangement.
	if os.Getenv("GOINFER_A9_BALLOON") == "" {
		t.Logf("balloon-shape control: repeating the boundary with many 2 MiB blocks instead of a " +
			"few large ones (same free bytes, different layout)")
		for _, leave := range []int64{l, h} {
			cmd := exec.Command(os.Args[0], "-test.run=TestMoERouteDemandThresholdChild", "-test.timeout=5m")
			cmd.Env = append(os.Environ(), a9ChildEnv+"="+strconv.FormatInt(leave, 10), "GOINFER_A9_BALLOON=fine")
			out, _ := cmd.CombinedOutput()
			for ln := range strings.SplitSeq(string(out), "\n") {
				if after, ok := strings.CutPrefix(ln, "A9CHILD "); ok {
					t.Logf("  fine balloon, leave %d: %s", leave, after)
				}
			}
		}
	}

	// PINNED (item 6). The threshold is the number the cap analysis depends on; leaving it unasserted
	// makes this a report rather than a gate. Both bounds are pinned because the pair brackets the
	// demand and a one-sided pin would drift.
	//
	// RE-DERIVED 2026-08-12 (A11), not edited to match a red gate. The pins moved +589,824 B, and
	// that is the number A9-RESID recorded as "baseline drift" — the amount by which
	// demand = floor + residual failed to close at MOE_MAX_E=512 while closing EXACTLY at 256:
	//
	//     256:  151,191,552 + 54,525,952 = 205,717,504   measured 205,717,504   EXACT
	//     512:  151,191,552 + 138,412,032 = 289,603,584   measured 289,013,760   short by 589,824
	//
	// The measurement now reads 289,603,584 — the closed form, to the byte. Both components were
	// re-measured here and BOTH HELD: the floor is 151,191,552 (allocate-until-failure in a fresh
	// context: 7,665,287,168 reported, 7,514,095,616 obtained) and the residual is 138,412,032.
	// So nothing about the machine or the kernel moved; the OLD PIN was the outlier, recorded from
	// the one measurement that did not close, and the 589,824 was misattributed to drift rather
	// than read as a failure to close.
	//
	// The new values are therefore the DERIVED ones, and the identity is what justifies them. If
	// these ever move again, check the identity first: if floor + residual still equals the demand,
	// the components are what moved and this pin is downstream of them.
	// RE-DERIVED 2026-08-19, because the old pin (287,506,432 / 289,603,584) failed — and it failed
	// for a reason worth more than the number it was guarding.
	//
	// THE OLD PIN WAS A SUM WHOSE VALUE DEPENDS ON A PRECONDITION THIS TEST DOES NOT CONTROL:
	// whether another CUDA context is alive on the device while the child launches. Measured both
	// ways on one box, one commit, minutes apart:
	//
	//	child driven by this test (parent process holds a context):
	//	  leave 141,819,904 -> freeBefore 141,557,760  ok=TRUE
	//	child driven straight from a shell (nothing else on the card):
	//	  leave 141,819,904 -> freeBefore 140,050,432  ok=FALSE
	//	  leave 289,603,584 -> freeBefore 288,948,224  ok=TRUE
	//
	// So there are two regimes, and the launch's requirement differs by exactly the device-wide
	// reserve that the FIRST context on the card pays:
	//
	//	WARM (another context alive)  demand = residual                     ~= 138.4 MiB
	//	COLD (this is the first)      demand = deviceFloor + residual        = 289,603,584
	//
	// The RESIDUAL held bit-for-bit across every measurement (138,412,032), and the floor is the
	// same 151,191,552 TestAllocFloor reports. Nothing about the kernel or the machine moved; the
	// pin recorded the COLD sum and the test now runs WARM (it was moved into the drain group's own
	// process, where the preceding A10 tests leave a context on the device). A pin that flips with
	// its neighbours is measuring the neighbourhood.
	//
	// So pin the COMPONENTS, which are stable, and assert the IDENTITY against the regime actually
	// observed. That is the re-derivation the old assertion demanded ("re-deriving, not editing") —
	// and it now fails for a moved KERNEL rather than for a moved test-ordering.
	// RE-DERIVED 2026-08-21, and this time the component that moved was the FLOOR — which the
	// previous revision could not have discovered, because it asserted the opposite.
	//
	// The gate failed with "measured 192,675,840, expected 289,603,584..295,895,040" and concluded
	// "a break here means the KERNEL's launch requirement moved". IT DID NOT. Three measurements
	// settle it:
	//
	//	1. The value is bit-stable — 192,675,840 on four consecutive runs, not a wobble.
	//	2. It is IDENTICAL at c6760d7, the commit that recorded 141,557,760 and pinned against it.
	//	   Same commit, same box, two days apart, different number: nothing in the tree moved.
	//	3. TestAllocFloor now measures the floor at 54,263,808, not 151,191,552.
	//
	// And then the identity closes to the byte:
	//
	//	54,263,808 (floor) + 138,412,032 (residual) = 192,675,840 = the measured demand
	//
	// So the model is exactly right and one of its INPUTS changed underneath it. The residual is
	// unchanged (it is also what the launch actually consumes at the threshold: 192,675,840 ->
	// 54,263,808 leaves precisely 138,412,032). The floor dropped by 96,927,744 B for a reason
	// outside this repo — same driver, same uptime, no reboot between the two measurements.
	//
	// WHY THE GATE BLAMED THE KERNEL: its own comment claimed "the residual and the floor are each
	// pinned by their own gate". That is TRUE of the residual and FALSE of the floor —
	// TestAllocFloor says in as many words "Not a threshold assertion: the number is the finding",
	// so a floor move cannot fail there and surfaces here instead, wearing a kernel move's clothes.
	// The fix is not this constant; it is the missing pin, now added in TestAllocFloor, so the next
	// component move fails where the component is.
	//
	// SAFETY DIRECTION, checked rather than assumed: a SMALLER floor means less memory is reported
	// free but unallocatable, so there is MORE headroom than the cap analysis assumed, not less. The
	// margin clears it by 332.2 MiB (slotMarginBytes 402,653,184 vs floor 54,263,808). A1/A5/A7/A9's
	// conclusions are unaffected in the safe direction; the 33-slot cap stays safe and the 34-slot
	// cap stays unsafe for the residual reason, which did not move.
	const (
		pinnedResidual    = 138412032 // moe_route's steady-state reservation (TestMoERouteFirstLaunchReservation)
		pinnedDeviceFloor = 54263808  // the reserve a fresh context pays (TestAllocFloor, which now PINS it)
		// The bisection stops at a 2 MiB quantum and the launch's peak is a further ~1-3 MiB above
		// its residual (the transient the log below names), so the identity is asserted to a window
		// rather than to the byte. A byte-exact pin here is what made the old one brittle.
		demandWindow = 6 << 20
	)
	warm := firstPass.freeBefore < int64(pinnedDeviceFloor)
	wantDemand, regime := int64(pinnedResidual), "WARM (another CUDA context alive — the device reserve is already paid)"
	if !warm {
		wantDemand = int64(pinnedDeviceFloor) + int64(pinnedResidual)
		regime = "COLD (this launch is the first context on the device and pays the reserve itself)"
	}
	t.Logf("REGIME: %s — expected demand %d B, measured %d B", regime, wantDemand, firstPass.freeBefore)
	if firstPass.freeBefore < wantDemand || firstPass.freeBefore > wantDemand+demandWindow {
		// CHECK THE COMPONENTS BEFORE BLAMING THE KERNEL. An earlier revision of this message
		// asserted flatly that a break here means the kernel moved, and it was wrong the first time
		// it fired: the floor had halved and the identity still closed. So the message now says what
		// is actually known — the SUM disagrees — and names the two ways that happens, in the order
		// they should be checked.
		t.Errorf("demand identity BROKEN in the %s regime: measured %d B, expected %d..%d B "+
			"(residual %d + floor %d when cold). "+
			"CHECK THE COMPONENTS FIRST, in this order: "+
			"(1) does floor + residual still equal the MEASURED demand? Run TestAllocFloor and "+
			"TestMoERouteFirstLaunchReservation. If it closes, a COMPONENT moved and this pin is "+
			"downstream of it — update the component, not this number. That is what happened on "+
			"2026-08-21: the floor went 151,191,552 -> 54,263,808 and the sum closed to the byte. "+
			"(2) ONLY if the identity does not close has the KERNEL's launch requirement moved — "+
			"which is what makes the 34-slot cap unsafe and the 33-slot cap safe, and then "+
			"docs/QUEUE.md A1/A5/A7/A9 need re-deriving, not editing.",
			regime, firstPass.freeBefore, wantDemand, wantDemand+int64(demandWindow),
			int64(pinnedResidual), int64(pinnedDeviceFloor))
	}

	// ---- the RELATIONSHIP, not just the figures (item 2) ----
	//
	// The three per-kernel byte pins say "a number changed". This says "the safety property broke",
	// which is the one that explains why anyone should care. slotMarginBytes exists to leave room
	// for exactly the costs measured here, and nothing checked that it does.
	//
	// MAX, not SIGMA. Launching the whole census (moe_route + rope_kv + rope_kv_batched) gives a
	// threshold and a residual IDENTICAL to moe_route alone, to the byte — the driver shares one
	// local-memory backing store sized by the largest kernel rather than summing them. Summing would
	// overstate the requirement, so the assertion is against the maximum, and the census gate is what
	// guarantees the maximum is taken over every kernel rather than a remembered one.
	//
	// THE REGIME IS PART OF THE CLAIM. That measurement launched the census SEQUENTIALLY IN ONE
	// CONTEXT, which is what goinfer does today: batch-1, single stream, one resident model. Under
	// concurrent residency on separate streams there is no reason the bound stays `max` — two
	// kernels in flight may each need their own backing store — and this assertion would then be
	// wrong WITHOUT FAILING, which is the worse of the two ways to be wrong. If goinfer gains
	// concurrent streams or multi-model residency on one context, re-measure before trusting this.
	// AGAINST THE WORST REGIME, not the measured one (2026-08-19). The measurement above may be
	// WARM, where the launch does not pay the device reserve — but the margin's job is to be
	// sufficient whatever the card's state, and asserting it against the smaller warm figure would
	// let the cold requirement exceed the margin without failing. So the safety check uses
	// max(measured, cold), which is regime-independent by construction.
	coldDemand := int64(pinnedDeviceFloor) + int64(pinnedResidual)
	worstDemand := max(firstPass.freeBefore, coldDemand)
	if int64(slotMarginBytes) < worstDemand {
		t.Errorf("slotMarginBytes (%d) is BELOW the WORST-REGIME launch demand (%d = max of measured "+
			"%d and cold floor+residual %d). The margin must cover the cold case too: a box where "+
			"goinfer's is the first context on the card pays the device reserve inside the launch.",
			int64(slotMarginBytes), worstDemand, firstPass.freeBefore, coldDemand)
	}
	if int64(slotMarginBytes) < firstPass.freeBefore {
		t.Errorf("slotMarginBytes (%d) is BELOW the measured peak launch demand (%d). The margin's "+
			"whole job is to leave room for deferred first-launch costs, so the expert-cache cap "+
			"can now be granted at a size whose forward cannot run — the 34-slot failure, "+
			"structurally, at whatever the new cap is. NOTE the demand here is a MAX over the "+
			"kernel census, valid for SEQUENTIAL SINGLE-STREAM launch, which is the regime goinfer "+
			"runs in; if concurrent streams were added, the bound may be a sum and this figure is "+
			"then a lower bound rather than the requirement",
			int64(slotMarginBytes), firstPass.freeBefore)
	}
	t.Logf("margin check: slotMarginBytes %d >= worst-regime demand %d (measured %d, cold %d), "+
		"clear by %d B (%.1f MiB)", int64(slotMarginBytes), worstDemand, firstPass.freeBefore,
		coldDemand, int64(slotMarginBytes)-worstDemand,
		float64(int64(slotMarginBytes)-worstDemand)/(1<<20))

	// Capacity or contiguity? Repeat the boundary. A threshold that moves between identical runs is
	// not a capacity threshold.
	const reps = 3
	t.Logf("repeating the lowest PASS %d times to separate capacity from contiguity", reps)
	varied := false
	for i := range reps {
		tr := a9Run(t, h)
		t.Logf("  repeat %d: leave %d -> free %d ok=%t", i+1, h, tr.freeBefore, tr.ok)
		if !tr.ok {
			varied = true
		}
	}
	if varied {
		t.Errorf("the pass/fail outcome VARIES at a fixed balloon size — this is contiguity, not "+
			"capacity, and the threshold above is not a demand figure. Free at the boundary is "+
			"%d B", firstPass.freeBefore)
	}
}
