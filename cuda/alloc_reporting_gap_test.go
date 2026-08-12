//go:build cuda && goinfer_testhooks

package cuda

import (
	"fmt"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"

	gpu "github.com/townsendmerino/aikit/gpu"
)

// A10 model under test: cuMemGetInfo reports ~151,191,552 B more free than is allocatable, by
// anyone, so usable = reported_free - gap.
//
// TestA10ReportingGap is the cheap cross-check: no launch, no balloon-and-bisect. Allocate directly
// in a fresh context until even a 1 MiB request fails, and compare the total actually obtained
// against the free figure reported at the start. If the shortfall is ~151 MiB, the reporting gap is
// confirmed with no kernel involved at all — which separates "the allocator reserves" from anything
// about launches.
func TestA10ReportingGap(t *testing.T) {
	dev, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	read := func() int64 { f, _, _ := dev.Context().MemInfo(); return int64(f) }
	var hold []gpu.Buffer
	var got int64
	alloc := func(n int) (ok bool) {
		defer func() {
			if recover() != nil {
				ok = false
			}
		}()
		hold = append(hold, gpu.NewBufferLenOf[byte](dev, n))
		got += int64(n)
		return true
	}
	start := read()
	for size := int64(1) << 30; size >= (1 << 20); size /= 2 {
		for alloc(int(size)) {
		}
	}
	end := read()
	t.Logf("  reported free at start   %13d B", start)
	t.Logf("  total obtained           %13d B", got)
	t.Logf("  reported free at end     %13d B", end)
	t.Logf("  SHORTFALL (start-got)    %13d B  (%.1f MiB)", start-got, float64(start-got)/(1<<20))
	t.Logf("  measured floor           %13d B  (TestAllocFloor)", 151191552)
	// The shortfall is the requested total against reported free; per-allocation 2 MiB rounding
	// inflates what the driver actually took, so the shortfall is an UPPER bound on the gap.
	if start-got <= 0 {
		t.Fatalf("obtained %d B against %d B reported — the instrument is wrong", got, start)
	}
	_ = hold
}

// TestA10FloorIsPerProcessOrPerDevice varies the CONTEXT rather than the kernel — the axis the floor
// has never been tested against.
//
// A child process drains to the floor and HOLDS. This process, with its own context, then reads free
// and tries to allocate.
//
//	parent can still allocate -> the floor is per-process/per-context; N contexts cost N x the
//	                             reserve, and the margin is NOT a constant
//	parent cannot             -> one device-wide reserve, and the margin CAN be derived from it
//
// RESULT 2026-08-12: NEITHER — the parent cannot create a context at all while the child holds
// (cuDevicePrimaryCtxRetain: CUDA_ERROR_OUT_OF_MEMORY at 151,191,552 B reported free). That is a
// finding on its own — the floor is not available for context setup either — but it means this arm
// cannot measure what it was built to measure. The in-process arm is blocked too: gocudrv exposes
// only primary-context retain, not cuCtxCreate, so a second simultaneous context cannot be made.
//
// What IS established: the floor is 151,191,552 B in every separate process measured, so it is a
// stable per-device property rather than something accumulating per process. Whether two SIMULTANEOUS
// contexts each pay it is untested and untestable with the current API surface.
// probeFreeWithoutContext reads free VRAM from nvidia-smi, which needs no CUDA context — so it can
// be read BEFORE this process retains one, which cuMemGetInfo cannot.
func probeFreeWithoutContext(t *testing.T) int64 {
	t.Helper()
	out, err := exec.Command("nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits").Output()
	if err != nil {
		t.Skipf("nvidia-smi unavailable: %v", err)
	}
	var mib int64
	fmt.Sscanf(strings.TrimSpace(strings.Split(string(out), "\n")[0]), "%d", &mib)
	return mib << 20
}

func TestA10FloorIsPerProcessOrPerDevice(t *testing.T) {
	if os.Getenv("GOINFER_A10_DRAIN_CHILD") != "" {
		dev, err := CreateSystemDefaultDevice()
		if err != nil {
			return
		}
		read := func() int64 { f, _, _ := dev.Context().MemInfo(); return int64(f) }
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
		// Leave ~300 MiB reported free rather than draining to the floor. The first attempt drained
		// completely, and the parent then could not create a context at all — context setup needs
		// memory the floor does not provide, so the arm could not measure what it was built for.
		leave := int64(300 << 20)
		if os.Getenv("GOINFER_A10_DRAIN_CHILD") == "roomy" {
			leave = 620 << 20 // room for three contexts, not two
		}
		for size := int64(1) << 30; size >= (2 << 20); {
			rem := read() - leave
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
		fmt.Printf("A10DRAINED free=%d\n", read())
		os.Stdout.Sync()
		time.Sleep(25 * time.Second) // hold the allocations while the parent probes
		_ = hold
		return
	}

	mode := os.Getenv("GOINFER_A10_MODE") // "third" adds a middle context-holder
	cmd := exec.Command(os.Args[0], "-test.run=TestA10FloorIsPerProcessOrPerDevice", "-test.timeout=2m")
	drainMode := "1"
	if mode == "third" {
		drainMode = "roomy"
	}
	cmd.Env = append(os.Environ(), "GOINFER_A10_DRAIN_CHILD="+drainMode)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatalf("pipe: %v", err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatalf("start: %v", err)
	}
	defer func() { _ = cmd.Process.Kill(); _, _ = cmd.Process.Wait() }()

	buf := make([]byte, 4096)
	deadline := time.Now().Add(90 * time.Second)
	var childFree int64
	for time.Now().Before(deadline) {
		n, _ := stdout.Read(buf)
		if n > 0 {
			for _, ln := range strings.Split(string(buf[:n]), "\n") {
				if strings.HasPrefix(ln, "A10DRAINED free=") {
					fmt.Sscanf(ln, "A10DRAINED free=%d", &childFree)
				}
			}
		}
		if childFree > 0 {
			break
		}
	}
	if childFree == 0 {
		t.Skip("child never reported draining — cannot run the discriminator")
	}

	// Read free BEFORE and AFTER retaining the primary context in this second process. The delta is
	// what a context costs on a device that already has one — which is the whole question.
	if mode == "third" {
		// A middle process that only retains a context and holds, so the parent below becomes the
		// THIRD context on the device rather than the second.
		mid := exec.Command(os.Args[0], "-test.run=TestA10CtxHolder", "-test.timeout=2m")
		mid.Env = append(os.Environ(), "GOINFER_A10_CTX_HOLDER=1")
		mp, _ := mid.StdoutPipe()
		if e := mid.Start(); e != nil {
			t.Fatalf("mid: %v", e)
		}
		defer func() { _ = mid.Process.Kill(); _, _ = mid.Process.Wait() }()
		mb := make([]byte, 4096)
		for dl := time.Now().Add(60 * time.Second); time.Now().Before(dl); {
			n, _ := mp.Read(mb)
			if n > 0 && strings.Contains(string(mb[:n]), "A10CTXHELD") {
				for _, ln := range strings.Split(string(mb[:n]), "\n") {
					if strings.HasPrefix(ln, "A10CTXHELD") {
						t.Logf("  middle process: %s", strings.TrimPrefix(ln, "A10CTXHELD "))
					}
				}
				break
			}
		}
	}

	// BOTH readings from nvidia-smi. An earlier version took `pre` from nvidia-smi and `post` from
	// cuMemGetInfo, so the delta silently carried the disagreement between two instruments (~832 KiB
	// here) as if it were context cost. Same shape as the measurement-shape class: the number was
	// real and the comparison was not like-for-like.
	pre := probeFreeWithoutContext(t)
	dev, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Fatalf("parent still cannot retain a context with ~300 MiB free: %v", err)
	}
	_, _, _ = dev.Context().MemInfo() // force the context to be live before re-reading
	post := probeFreeWithoutContext(t)
	t.Logf("  child holds, reporting     %13d B free", childFree)
	t.Logf("  parent BEFORE its context  %13d B free", pre)
	t.Logf("  parent AFTER  its context  %13d B free", post)
	t.Logf("  DELTA (this context cost) %13d B (%.2f MiB)", pre-post, float64(pre-post)/(1<<20))
	if pre-post > 100<<20 {
		t.Logf("  => PER-PROCESS: the reserve is paid again per context, so N contexts cost N x it " +
			"and slotMarginBytes is not a device constant.")
	} else {
		t.Logf("  => PER-DEVICE: a second context costs far less than the 151,191,552 B gap, so the " +
			"gap is one device-wide reserve and the margin derivation is a constant.")
	}
}

// TestA10CtxHolder retains the primary context, reports what it cost by the same instrument on both
// sides, and holds. Used as the middle process when measuring a THIRD context's cost.
func TestA10CtxHolder(t *testing.T) {
	if os.Getenv("GOINFER_A10_CTX_HOLDER") == "" {
		t.Skip("helper for TestA10FloorIsPerProcessOrPerDevice")
	}
	pre := probeFreeWithoutContext(t)
	dev, err := CreateSystemDefaultDevice()
	if err != nil {
		fmt.Printf("A10CTXHELD FAILED %v\n", err)
		os.Stdout.Sync()
		time.Sleep(25 * time.Second)
		return
	}
	_, _, _ = dev.Context().MemInfo()
	post := probeFreeWithoutContext(t)
	fmt.Printf("A10CTXHELD before=%d after=%d cost=%d\n", pre, post, pre-post)
	os.Stdout.Sync()
	time.Sleep(25 * time.Second)
}
