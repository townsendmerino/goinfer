//go:build cuda && goinfer_testhooks

package cuda

import (
	"os"
	"testing"
	"time"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/aikit/gpu"
)

// TestLead3_pinOrderMicrobench is docs/tasks/task-freetoken-techniques.md Lead 3's own "cheap,
// narrow experiment against an already-known, already-measured cost": does populating ordinary
// memory FIRST and pinning it afterward (gocudrv's cuda.RegisterHost, exposed but unused by
// cuda/resident.go's mapBytes today) beat allocating pinned memory first and copying into it
// second (cuMemAllocHost then copy — mapBytes's current order), at the ~11.4 GB scale C′'s
// expert-stack staging actually runs at?
//
// Both arms do the SAME work — allocate N bytes of host memory reachable by the GPU, fill it
// with real content, in the order that differs — timed separately so the allocation-side cost
// and the fill-side cost don't hide in one number. Same process, same box, interleaved (A,B,A,B)
// so thermal/allocator-warmup drift cannot pose as an effect.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks' ./cuda/ -run TestLead3_pinOrderMicrobench -v -timeout 15m
func TestLead3_pinOrderMicrobench(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("GOINFER_HEAVY_TESTS unset — allocates ~11 GB of pinned host memory")
	}
	dev, err := gpu.CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	defer dev.ReleaseObjects()
	ctx := dev.Context()

	const sizeBytes = 11_400_000_000 // C′'s real expert-stack scale (task-moe-streaming.md's own figure)
	src := make([]byte, sizeBytes)
	for i := range src {
		src[i] = byte(i) // real content, not zero pages — a zero-fill-only page would be an unfair best case for either arm
	}

	// Arm A: today's mapBytes order — allocate pinned, then copy.
	pinThenFill := func() (allocDur, fillDur time.Duration) {
		t0 := time.Now()
		mb, err := dev.NewMappedHostBuffer(sizeBytes)
		if err != nil {
			t.Fatalf("NewMappedHostBuffer: %v", err)
		}
		allocDur = time.Since(t0)
		t1 := time.Now()
		copy(mb.Bytes(), src)
		fillDur = time.Since(t1)
		if err := mb.Close(); err != nil {
			t.Fatalf("Close: %v", err)
		}
		return
	}

	// Arm B: Lead 3's proposed order — fill ordinary memory, then register (pin) it.
	fillThenPin := func() (fillDur, pinDur time.Duration) {
		dst := make([]byte, sizeBytes)
		t0 := time.Now()
		copy(dst, src)
		fillDur = time.Since(t0)
		t1 := time.Now()
		rh, err := gc.RegisterHost[byte](ctx, dst)
		if err != nil {
			t.Fatalf("RegisterHost: %v", err)
		}
		pinDur = time.Since(t1)
		if err := rh.Close(); err != nil {
			t.Fatalf("Close: %v", err)
		}
		return
	}

	// Warm-up both, discarded (first CUDA host-memory call per process pays a one-time driver cost).
	pinThenFill()
	fillThenPin()

	const pairs = 3
	var sumA, sumB time.Duration
	for p := 0; p < pairs; p++ {
		var allocA, fillA, fillB, pinB time.Duration
		if p%2 == 0 {
			allocA, fillA = pinThenFill()
			fillB, pinB = fillThenPin()
		} else {
			fillB, pinB = fillThenPin()
			allocA, fillA = pinThenFill()
		}
		totalA, totalB := allocA+fillA, fillB+pinB
		sumA += totalA
		sumB += totalB
		t.Logf("pair %d: A(alloc-then-copy) alloc %s + copy %s = %s  |  B(fill-then-pin) copy %s + register %s = %s  |  B/A speedup %.3fx",
			p, allocA.Round(time.Millisecond), fillA.Round(time.Millisecond), totalA.Round(time.Millisecond),
			fillB.Round(time.Millisecond), pinB.Round(time.Millisecond), totalB.Round(time.Millisecond),
			totalA.Seconds()/totalB.Seconds())
	}
	speedup := sumA.Seconds() / sumB.Seconds()
	bytesPerSec := func(d time.Duration) float64 { return float64(sizeBytes) * pairs / d.Seconds() / 1e9 }
	verdict := "KILL (<1.15x)"
	switch {
	case speedup >= 1.5:
		verdict = "SHIP (>=1.5x)"
	case speedup >= 1.15:
		verdict = "PARK (1.15-1.5x)"
	}
	t.Logf("TOTAL over %d pairs: A %s (%.2f GB/s) | B %s (%.2f GB/s) | speedup %.3fx — %s",
		pairs, sumA.Round(time.Millisecond), bytesPerSec(sumA), sumB.Round(time.Millisecond), bytesPerSec(sumB), speedup, verdict)
}
