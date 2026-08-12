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
		for size := int64(1) << 30; size >= (1 << 20); size /= 2 {
			for alloc(int(size)) {
			}
		}
		fmt.Printf("A10DRAINED free=%d\n", read())
		os.Stdout.Sync()
		time.Sleep(25 * time.Second) // hold the allocations while the parent probes
		_ = hold
		return
	}

	cmd := exec.Command(os.Args[0], "-test.run=TestA10FloorIsPerProcessOrPerDevice", "-test.timeout=2m")
	cmd.Env = append(os.Environ(), "GOINFER_A10_DRAIN_CHILD=1")
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

	dev, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("parent cannot open a device while the child holds: %v", err)
	}
	f, _, _ := dev.Context().MemInfo()
	ok := func() (o bool) {
		defer func() {
			if recover() != nil {
				o = false
			}
		}()
		_ = gpu.NewBufferLenOf[byte](dev, 2<<20)
		return true
	}()
	t.Logf("  child drained to           %13d B free (its own reading)", childFree)
	t.Logf("  parent, separate process:  %13d B free", f)
	t.Logf("  parent 2 MiB allocation:   %v", ok)
	if ok {
		t.Logf("  => the floor is PER-PROCESS/PER-CONTEXT. N contexts cost N x the reserve, and " +
			"slotMarginBytes is not a constant that can be derived once.")
	} else {
		t.Logf("  => ONE DEVICE-WIDE reserve. The margin CAN be derived from it: " +
			"margin >= reporting gap + peak transient.")
	}
}
