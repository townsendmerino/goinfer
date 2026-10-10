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

// These tests measure moe_route's first-launch demand directly instead of inferring it from the
// size of its reservation: balloon the device to leave a chosen number of bytes free, launch
// moe_route in a fresh-context child, and binary-search the pass/fail boundary. A result that
// varies run to run at the same balloon size is contiguity rather than capacity, a different
// finding; the balloon-shape control exists to tell the two apart.

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
	// four small buffers. That is the production shape (they exist long before the launch), and
	// ballooning first leaves nothing to allocate an argument buffer from.
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

	// Back off on failure rather than stopping. A failed request means THAT SIZE does not fit, which
	// is a statement about contiguity, not capacity; halving until the quantum is reached is what
	// drains the pool, and stopping at the first refusal leaves the balloon short of its target (the
	// parent's bracket check catches that).
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
			gpu.ArgValue(int32(0)), gpu.ArgValue(int32(0)), // pos, ropePos (equal — not under test here)
			gpu.ArgValue(int32(1)), gpu.ArgValue(float32(1)), // mscale 1.0 = unscaled
			gpu.ArgValue(float32(1))) // qTempScale 1.0 = no-op (FeatAttnTemp, G5) — not under test here
	}
	if census && lerr == nil {
		lerr = q.Launch(pRopeB, one, Arg(rq), Arg(rk), Arg(rv), Arg(rInv), Arg(rKc), Arg(rVc),
			gpu.ArgValue(int32(1)), gpu.ArgValue(int32(1)), gpu.ArgValue(int32(2)),
			gpu.ArgValue(int32(0)), gpu.ArgValue(int32(1)), gpu.ArgValue(int32(1)), gpu.ArgValue(float32(1)),
			gpu.ArgValue(float32(0)), gpu.ArgValue(float32(0))) // attnTempBeta=0 ⇒ off (FeatAttnTemp, G5) — not under test here
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
	// MARKED AS A DRAINER (see cuda/drain_marker_test.go). The bisection deliberately balloons the
	// device to as little as 64 MiB (below moe_route's launch demand) and records the resulting
	// refusal as data: `bracket low: leave 67108864 -> ok=false` IS a refusal, driven on purpose. It
	// balloons through child processes, so each child's memory is returned when it exits, but a child
	// that fails or hangs leaves the device at the floor for whatever runs next in this process.
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

	// PINNED. The threshold is the number the slot-cap analysis depends on; leaving it unasserted
	// makes this a report rather than a gate. The pins are the COMPONENTS, which are stable, and the
	// assertion is the IDENTITY against the regime observed:
	//
	// 	demand = pinnedDeviceFloor + pinnedResidual
	//
	// where the floor is the device-wide reserve the FIRST context on the card pays and the residual
	// is the launch's own reservation. A pin on the sum alone depends on preconditions this test does
	// not control (whether another CUDA context is alive, the driver and distro build) and flips with
	// its neighbours.
	//
	// A break here: check the components first (TestAllocFloor pins the floor,
	// TestMoERouteFirstLaunchReservation the residual). The floor has moved for reasons outside the
	// repo while the identity kept closing to the byte. Only if the identity does not close has the
	// kernel's launch requirement moved. A smaller floor means MORE headroom against slotMarginBytes
	// (asserted below), so a floor move is safe in that direction. Re-derivation records:
	// docs/code-notes/cuda.md#TestMoERouteDemandThreshold.
	const (
		pinnedResidual    = 138412032 // moe_route's steady-state reservation (TestMoERouteFirstLaunchReservation)
		pinnedDeviceFloor = 1769472   // the reserve a fresh context pays (TestAllocFloor, which PINS it)
		// The bisection stops at a 2 MiB quantum and the launch's peak is a further ~1-3 MiB above
		// its residual (the transient the log below names), so the identity is asserted to a window
		// rather than to the byte. A byte-exact pin here is what made the old one brittle.
		demandWindow = 6 << 20
	)
	// REGIME, asked of the device rather than inferred from a number. With another CUDA context alive
	// (a desktop compositor counts) the device floor is not pinned, so this SKIPS rather than assert
	// the cold pin: a skip is not a pass, and the gate reports it as uncovered. The demand identity
	// itself still closes there (measured floor + residual = demand), so a failure from a foreign
	// context is the environment, not the kernel.
	foreign, known := foreignCUDAContexts()
	if !known {
		t.Skip("cannot determine whether another CUDA context is alive (no nvidia-smi): " +
			"the expected demand depends on that, and guessing it is what this check replaced")
	}
	if len(foreign) > 0 {
		t.Skipf("another CUDA context is alive on the device — %s. The device reserve, and so the "+
			"expected demand, is not pinned for that case (measured 16 MiB higher on 2026-09-01). "+
			"Run this with an exclusive device (e.g. from a TTY with the compositor stopped). "+
			"NOTE: the demand IDENTITY held there — measured floor + residual equalled the measured "+
			"demand to the byte — so a failure here is about the environment, NOT the kernel, and "+
			"docs/QUEUE.md A1/A5/A7/A9 do not need re-deriving.", describeForeign(foreign))
	}
	wantDemand := int64(pinnedDeviceFloor) + int64(pinnedResidual)
	regime := "COLD (this launch is the first context on the device and pays the reserve itself)"
	t.Logf("REGIME: %s — expected demand %d B, measured %d B", regime, wantDemand, firstPass.freeBefore)
	if firstPass.freeBefore < wantDemand || firstPass.freeBefore > wantDemand+demandWindow {
		// CHECK THE COMPONENTS BEFORE BLAMING THE KERNEL: the message says only what is known (the SUM
		// disagrees) and names the two ways that happens, in the order they should be checked.
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

	// ---- the RELATIONSHIP, not just the figures ----
	//
	// The per-kernel byte pins say "a number changed". This says "the safety property broke":
	// slotMarginBytes exists to leave room for exactly the costs measured here.
	//
	// MAX, not SIGMA: launching the whole census (moe_route + rope_kv + rope_kv_batched) gives a
	// threshold and residual identical to moe_route alone, because the driver shares one local-memory
	// backing store sized by the largest kernel. The census gate is what guarantees the maximum is
	// taken over every kernel rather than a remembered one.
	//
	// THE REGIME IS PART OF THE CLAIM. That holds for SEQUENTIAL launch in one context (batch-1, single
	// stream, one resident model). Under concurrent residency on separate streams two kernels in
	// flight may each need their own backing store, and this assertion would be wrong WITHOUT
	// FAILING. If goinfer gains concurrent streams or multi-model residency on one context,
	// re-measure before trusting it.
	//
	// The check is against the WORST regime, not the measured one: the measurement may be warm and
	// not pay the device reserve, but the margin must be sufficient whatever the card's state, so it
	// uses max(measured, cold).
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
