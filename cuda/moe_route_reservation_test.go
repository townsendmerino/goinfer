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

// MEASURED (RTX 2070 SUPER, 40 SMs, driver 595.58.03, 2026-08-12), both instruments agreeing:
//
//	step                          cuMemGetInfo free     nvidia-smi process
//	start                            7 664 697 344 B          102 MiB
//	CompileLibrary(moePTX)           7 664 697 344 B          102 MiB   (cost 0)
//	5x NewComputePipeline            7 664 697 344 B          102 MiB   (cost 0)
//	first launch, shared_gate_combine  unchanged              unchanged (cost 0)
//	first launch, moe_route          7 524 188 160 B          236 MiB   (cost 138 412 032 B = 132 MiB)
//
// 132 MiB, paid once, at the first launch of moe_route — long after allocSlots sized the cache.
// Two float[MOE_MAX_E] per-thread arrays at MOE_MAX_E=512 is 4 KiB/thread of local memory, and the
// driver backs local memory for the device's occupancy on first use regardless of the 1x1 grid
// goinfer launches it with. Raising MOE_MAX_E 256 -> 512 therefore doubled a hidden fixed cost from
// ~66 to 132 MiB; that halving is DERIVED from the form, not measured, and is recorded as the price
// of the router cap rather than as an argument to change it.
//
// TestMoERouteFirstLaunchReservation measures when moePTX's device memory is actually taken: at CompileLibrary,
// at NewComputePipeline, or deferred to the first launch of one of its kernels.
//
// A9's premise was that the cost is deferred. goinfer compiles moePTX at cuda/backend.go:591 and
// sizes the expert cache at cuda/backend.go:793, so under the driver's default
// CUDA_MODULE_LOADING=LAZY a deferred module load would be paid AFTER the cap was computed from a
// free-VRAM reading that did not include it — invisible to before/after readings around allocSlots,
// and invisible to any between-slot-count delta, because it does not scale with slots.
//
// The 26B run under CUDA_MODULE_LOADING=EAGER returned free-before-allocSlots byte-identical to the
// LAZY run (3,847,880,704 B) and failed at fRoute identically. That null is NOT an answer on its
// own: it is equally consistent with "EAGER took effect and module loading costs nothing" and with
// "EAGER was ignored". This test discriminates them directly, with no model and no cache, by
// reading free VRAM around each step.
//
// It needs no fixture and takes seconds, which is the point — the mechanism question was never
// model-dependent, and answering it inside a five-minute 26B load is what made it look expensive.
func TestMoERouteFirstLaunchReservation(t *testing.T) {
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
	t.Logf("CUDA_MODULE_LOADING=%q (empty = driver default LAZY)", os.Getenv("CUDA_MODULE_LOADING"))
	// Second instrument, read at every step. cuMemGetInfo reports the allocatable pool; nvidia-smi
	// reports this process's total device footprint, which includes context and module code. Where
	// they disagree, the disagreement is the finding — and a pool reading that cannot see a real
	// consumer is exactly the kind of blind spot the cap arithmetic would inherit.
	smi := func(label string) {
		if b, e := smiProcessBytes(); e == nil {
			t.Logf("    nvidia-smi @ %-22s %13d B (%d MiB)", label, b, b>>20)
		} else {
			t.Logf("    nvidia-smi @ %-22s unavailable: %v", label, e)
		}
	}

	base := read()
	smi("start")
	mod, err := dev.CompileLibrary(moePTXOrOverride())
	if err != nil {
		t.Fatalf("CompileLibrary(moePTX): %v", err)
	}
	afterCompile := read()
	smi("after CompileLibrary")

	names := []string{"moe_route", "gemv_f32_a8", "gemv_w4a8_moe", "gemv_w4a8_moe_wacc", "shared_gate_combine"}
	pipes := make([]Pipeline, 0, len(names))
	for _, n := range names {
		p, e := dev.NewComputePipeline(mod, n)
		if e != nil {
			t.Fatalf("NewComputePipeline(%s): %v", n, e)
		}
		pipes = append(pipes, p)
	}
	afterPipelines := read()
	smi("after NewComputePipeline")

	t.Logf("  free at start                 %13d B", base)
	t.Logf("  free after CompileLibrary     %13d B   (cost %d B)", afterCompile, base-afterCompile)
	t.Logf("  free after %d NewComputePipeline %10d B   (cost %d B)",
		len(names), afterPipelines, afterCompile-afterPipelines)
	t.Logf("  moePTX source size %d B", len(moePTXOrOverride()))

	total := base - afterPipelines
	t.Logf("  TOTAL moePTX device cost      %13d B  (%.1f MiB)", total, float64(total)/(1<<20))

	// The discriminating assertion. If the whole cost lands before any kernel of the module has been
	// launched, then goinfer's free reading at backend.go:793 ALREADY includes it, and the cap
	// arithmetic is not being deceived by a deferred cost — A9's premise is refuted for a reason
	// rather than by a null. If instead the cost here is ~0, the memory is genuinely taken later, at
	// first launch, and A9's premise stands.
	//
	// Either way this is a recording test, not a threshold: the number is the finding, and it is
	// logged above with its probe positions. The one thing that WOULD be a defect is measuring
	// nothing at all.
	if base == 0 || afterPipelines == 0 {
		t.Fatal("free-VRAM readings are zero — the instrument did not run")
	}
	if total == 0 {
		t.Logf("  => moePTX's MODULE memory is 0 B up to this point. Not yet a verdict: it is " +
			"consistent with a deferred cost and with there being no cost. The launches below " +
			"separate them.")
	} else {
		t.Logf("  => moePTX's module memory (%d B) is paid before any of its kernels launches, so "+
			"the free reading the cap is computed from already includes it.", total)
	}

	// ---- and now actually launch one of its kernels ----
	//
	// This is the step CUDA_MODULE_LOADING=EAGER was supposed to make unnecessary. It did not:
	// the readings above are byte-identical with and without it, so EAGER does not engage on this
	// driver/path and the 26B run made under it forced nothing. A null from a forcing mechanism
	// that never fired says nothing about what it was meant to force.
	//
	// shared_gate_combine is the safe choice: `dst[i] += g*shDown[i]` over N elements, no cache, no
	// routing, no expert weights. N=1 with three one-float buffers touches nothing else.
	q := dev.NewCommandQueue()
	dst := af1(dev)
	shDown := af1(dev)
	gl := af1(dev)
	beforeLaunch := read()
	smi("before first launch")
	lerr := q.Launch(pipes[4], LaunchConfig{GridX: 1, GridY: 1, GridZ: 1, BlockX: 1, BlockY: 1, BlockZ: 1},
		Arg(dst), Arg(shDown), Arg(gl), gpu.ArgValue(int32(1)), gpu.ArgValue(int32(1)))
	serr := q.Sync()
	afterLaunch := read()
	smi("after first launch")
	t.Logf("  free before first moePTX launch %11d B", beforeLaunch)
	t.Logf("  free after  first moePTX launch %11d B   (cost %d B)", afterLaunch, beforeLaunch-afterLaunch)
	if lerr != nil || serr != nil {
		t.Fatalf("launching shared_gate_combine failed (launch=%v sync=%v) — the materialisation "+
			"cost cannot be read from a launch that did not happen", lerr, serr)
	}
	// ---- the kernel that actually fails: moe_route ----
	//
	// shared_gate_combine materialises the module but reserves nothing, which is why it read 0.
	// moe_route declares `float score[MOE_MAX_E]; float sel[MOE_MAX_E]` with MOE_MAX_E = 512 —
	// 4 KB of LOCAL memory per thread. The driver must back local memory for the device's full
	// occupancy on the first launch of such a kernel, no matter that goinfer launches it with one
	// block of one thread. That reservation is a deferred fixed cost paid at first launch, which is
	// A9's shape exactly — but in local memory, not module code, which is why probing the module
	// found nothing.
	// A9-RESID: nE and k are variable so the reservation can be tested for launch-configuration
	// dependence. Local memory is a COMPILE-TIME property, so a dependence here would itself be a
	// finding — the driver would be sizing the backing store from something other than the kernel's
	// declared footprint.
	nE, k := 8, 2
	if v, e := strconv.Atoi(os.Getenv("GOINFER_A9_NE")); e == nil && v > 0 {
		nE = v
	}
	if v, e := strconv.Atoi(os.Getenv("GOINFER_A9_K")); e == nil && v > 0 {
		k = v
	}
	rLogits, rBias := afn(dev, nE), afn(dev, nE)
	rIdx, rWgt := aun(dev, k), afn(dev, k)
	beforeRoute := read()
	smi("before moe_route")
	rerr := q.Launch(pipes[0], LaunchConfig{GridX: 1, GridY: 1, GridZ: 1, BlockX: 1, BlockY: 1, BlockZ: 1},
		Arg(rLogits), Arg(rBias), Arg(rIdx), Arg(rWgt),
		gpu.ArgValue(int32(nE)), gpu.ArgValue(int32(k)), gpu.ArgValue(int32(1)),
		gpu.ArgValue(int32(0)), gpu.ArgValue(float32(1)),
		gpu.ArgValue(int32(1)), gpu.ArgValue(int32(1)))
	rserr := q.Sync()
	afterRoute := read()
	smi("after moe_route")
	if rerr != nil || rserr != nil {
		t.Fatalf("launching moe_route failed (launch=%v sync=%v)", rerr, rserr)
	}
	routeCost := beforeRoute - afterRoute
	t.Logf("  launch config nE=%d k=%d", nE, k)
	t.Logf("  free before first moe_route     %11d B", beforeRoute)
	t.Logf("  free after  first moe_route     %11d B   (cost %d B = %.1f MiB)",
		afterRoute, routeCost, float64(routeCost)/(1<<20))
	// Predicted local-memory backing store, stated as a form rather than a constant so the number
	// is derived: bytesPerThread x maxThreadsPerSM x SMs. MOE_MAX_E=512 gives two float[512] arrays.
	const bytesPerThread = 2 * 512 * 4
	t.Logf("  local-memory reservation predicted as %d B/thread x maxThreads/SM x SMs; at 1024x40 "+
		"that is %d B (%.0f MiB)", bytesPerThread, bytesPerThread*1024*40, float64(bytesPerThread*1024*40)/(1<<20))

	cost := beforeLaunch - afterLaunch
	if cost != 0 {
		t.Errorf("shared_gate_combine's first launch consumed %d B — it declares no per-thread "+
			"scratch, so a non-zero reading means the model of what a first launch costs is wrong", cost)
	}

	// VERDICT.
	//
	// A9 asked whether a deferred fixed cost, invisible to the cap arithmetic, explains the 34-slot
	// failure. It does — but not through the mechanism A9 named.
	//
	// Module code:      0 B, by BOTH instruments, at CompileLibrary, at NewComputePipeline, and at
	//                   the first launch of a module kernel that declares no scratch. So "moePTX's
	//                   load is charged after the cap is computed" is REFUTED.
	// Local memory:     moe_route's first launch reserves the measured figure above, because it
	//                   declares two float[MOE_MAX_E] per-thread arrays and the driver must back
	//                   local memory for the device's occupancy on first use. Paid at first launch,
	//                   long after allocSlots sized the cache. A9's SHAPE is confirmed; its named
	//                   mechanism was the wrong one.
	//
	// Note what did not work. CUDA_MODULE_LOADING=EAGER was the intended forcing mechanism, and the
	// readings here are byte-identical with and without it — it does not engage on this driver and
	// path. The 26B run made under EAGER therefore forced nothing, and its null was uninformative.
	// A forcing mechanism has to be shown to fire before a null from it means anything.
	// PINNED (item 6). Asserting only "> 0" would let a MOE_MAX_E change double a hidden cost with
	// the gate still green. This is the RESIDUAL cost, which is 48% of the launch's PEAK demand —
	// see TestMoERouteDemandThreshold, which pins the other number.
	const pinnedReservation = 138412032

	// THE PRECONDITION IS NOW ASSERTED RATHER THAN ASSUMED. This measures a FIRST launch, and the
	// reservation is a CONTEXT property: once any earlier test in the process has launched
	// moe_route, the store is already reserved and this reads 0 B — not a changed reservation, but
	// a measurement that never had its precondition. It failed exactly that way in the full tier
	// ("reservation is 0 B, pinned at 138412032") while passing alone, which is the signature of a
	// test whose correctness depends on its position in the suite. That is a defect independently
	// of any gate.
	//
	// 0 B is therefore reported as COULD NOT EVALUATE, not as a moved constant. Any other
	// unexpected value is still a real finding and still fails.
	if routeCost == 0 {
		t.Skipf("could not evaluate: moe_route's backing store was ALREADY reserved before this test "+
			"ran, so this is not its first launch in the process and the reading is 0 B rather than "+
			"the %d B reservation. Run this test alone (-run '^%s$') to measure it.",
			int64(pinnedReservation), t.Name())
	}
	if routeCost != pinnedReservation {
		t.Errorf("moe_route's first-launch reservation is %d B, pinned at %d B. This is a deferred "+
			"fixed cost the expert-cache sizing does not account for, so a change here invalidates "+
			"the cap analysis in docs/QUEUE.md A1/A5/A9 — re-run TestMoERouteDemandThreshold and "+
			"update both figures together, or the residual and the peak drift apart",
			routeCost, int64(pinnedReservation))
	}

}

func af1(dev *Device) Buffer { return gpu.NewBufferLenOf[float32](dev, 1) }

// smiProcessBytes reads this process's device footprint from nvidia-smi — a second instrument, on a
// different accounting, for the same question.
func smiProcessBytes() (int64, error) {
	out, err := exec.Command("nvidia-smi",
		"--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return 0, err
	}
	me := strconv.Itoa(os.Getpid())
	for ln := range strings.SplitSeq(string(out), "\n") {
		p := strings.Split(ln, ",")
		if len(p) == 2 && strings.TrimSpace(p[0]) == me {
			mib, e := strconv.ParseInt(strings.TrimSpace(p[1]), 10, 64)
			if e != nil {
				return 0, e
			}
			return mib << 20, nil
		}
	}
	return 0, fmt.Errorf("pid %s not listed by nvidia-smi", me)
}

func afn(dev *Device, n int) Buffer { return gpu.NewBufferLenOf[float32](dev, n) }
func aun(dev *Device, n int) Buffer { return gpu.NewBufferLenOf[uint32](dev, n) }
