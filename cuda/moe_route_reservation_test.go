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

// TestMoERouteFirstLaunchReservation measures when moePTX's device memory is actually taken: at
// CompileLibrary, at NewComputePipeline, or deferred to the first launch of one of its kernels.
// It needs no fixture and no model, and takes seconds.
//
// It matters because goinfer sizes the expert cache from a free-VRAM reading; a cost paid after
// that reading is invisible to the cap arithmetic and, not scaling with slots, to any
// between-slot-count delta. The first launch of moe_route reserves local memory (two
// float[MOE_MAX_E] per-thread arrays, 4 KiB/thread at MOE_MAX_E=512) for the device's occupancy,
// whatever grid goinfer launches it with, so raising MOE_MAX_E raises a hidden fixed cost. Two
// instruments read at every step (cuMemGetInfo and nvidia-smi); where they disagree, that is the
// finding. Measured table: docs/code-notes/cuda.md#TestMoERouteFirstLaunchReservation.
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
	// launched, goinfer's free reading when it sizes the cache already includes it and the cap
	// arithmetic is not deceived by a deferred cost; if the cost here is ~0 the memory is taken later,
	// at first launch.
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
	// CUDA_MODULE_LOADING=EAGER does not engage on this driver/path (the readings above are
	// byte-identical with and without it), so a real launch is the only way to force the load.
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
	// ---- the kernel with a local-memory footprint: moe_route ----
	//
	// shared_gate_combine materialises the module but reserves nothing. moe_route declares
	// `float score[MOE_MAX_E]; float sel[MOE_MAX_E]` with MOE_MAX_E = 512 — 4 KB of LOCAL memory per
	// thread — and the driver backs local memory for the device's full occupancy on the first launch
	// of such a kernel, however small the grid goinfer launches it with. nE and k are variable so the
	// reservation can be tested for launch-configuration dependence: local memory is a compile-time
	// property, so a dependence would itself be a finding.
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

	// WHAT THIS ESTABLISHES. Module code costs 0 B, by both instruments, at CompileLibrary, at
	// NewComputePipeline, and at the first launch of a kernel that declares no scratch, so "moePTX's
	// load is charged after the cap is computed" is false. Local memory is the deferred cost:
	// moe_route's first launch reserves the measured figure above, because it declares two
	// float[MOE_MAX_E] per-thread arrays and the driver backs local memory for the device's occupancy
	// on first use — paid at first launch, long after allocSlots sized the cache.
	//
	// PINNED. Asserting only "> 0" would let a MOE_MAX_E change double a hidden cost with the gate
	// still green. This is the RESIDUAL cost; TestMoERouteDemandThreshold pins the launch's peak demand.
	const pinnedReservation = 138412032

	// THE PRECONDITION IS ASSERTED, not assumed. This measures a FIRST launch, and the reservation is
	// a CONTEXT property: once an earlier test in the process has launched moe_route the store is
	// already reserved and this reads 0 B — a measurement that never had its precondition, not a
	// changed reservation. 0 B is therefore reported as COULD NOT EVALUATE, not as a moved constant;
	// any other unexpected value is a real finding and fails.
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
