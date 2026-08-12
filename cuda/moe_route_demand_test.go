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
	mod, err := dev.CompileLibrary(moePTX)
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
	before := read()
	lerr := q.Launch(p, LaunchConfig{GridX: 1, GridY: 1, GridZ: 1, BlockX: 1, BlockY: 1, BlockZ: 1},
		Arg(rLogits), Arg(rBias), Arg(rIdx), Arg(rWgt),
		gpu.ArgValue(int32(8)), gpu.ArgValue(int32(2)), gpu.ArgValue(int32(1)),
		gpu.ArgValue(int32(0)), gpu.ArgValue(float32(1)),
		gpu.ArgValue(int32(1)), gpu.ArgValue(int32(1)))
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
	for _, ln := range strings.Split(string(out), "\n") {
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
			for _, ln := range strings.Split(string(out), "\n") {
				if strings.HasPrefix(ln, "A9CHILD ") {
					t.Logf("  fine balloon, leave %d: %s", leave, strings.TrimPrefix(ln, "A9CHILD "))
				}
			}
		}
	}

	// PINNED (item 6). The threshold is the number the cap analysis depends on; leaving it unasserted
	// makes this a report rather than a gate. Both bounds are pinned because the pair brackets the
	// demand and a one-sided pin would drift.
	const pinnedFail, pinnedPass = 286916608, 289013760
	if lastFail.freeBefore != pinnedFail || firstPass.freeBefore != pinnedPass {
		t.Errorf("demand threshold moved: FAIL at %d (pinned %d), PASS at %d (pinned %d). "+
			"moe_route's peak launch demand is what makes the 34-slot cap unsafe and the 33-slot "+
			"cap safe; if it has moved, docs/QUEUE.md A1/A5/A7/A9 need re-deriving, not editing",
			lastFail.freeBefore, int64(pinnedFail), firstPass.freeBefore, int64(pinnedPass))
	}

	// Capacity or contiguity? Repeat the boundary. A threshold that moves between identical runs is
	// not a capacity threshold.
	const reps = 3
	t.Logf("repeating the lowest PASS %d times to separate capacity from contiguity", reps)
	varied := false
	for i := 0; i < reps; i++ {
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
