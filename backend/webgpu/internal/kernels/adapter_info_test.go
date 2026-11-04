//go:build gpu

package kernels

import (
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// adapterInfoDeep grows the calling goroutine's stack by n frames before calling f, so a GC has a stack to shrink while f runs.
func adapterInfoDeep(n int, f func()) {
	var pad [256]byte
	if n == 0 {
		f()
		return
	}
	adapterInfoDeep(n-1, f)
	_ = pad
}

// TestAdapterGetInfo_survivesStackShrink calls the binding's Adapter.GetInfo from 8 goroutines with grown stacks for 3 seconds
// while another goroutine runs the GC without pause, so stacks are shrunk (copied) while GetInfo's frame is live. wgpu-native
// returns an empty string as a dangling pointer (0x1), so a binding must hold the info struct in C memory: on the Go stack the
// runtime aborts with "invalid pointer found on stack" when it copies the stack. Every adapter here reports at least one empty
// string, so the hazard is live. The CI failure and the version that fixed it: docs/measurements/ci-flakes-2026-10-25.md.
func TestAdapterGetInfo_survivesStackShrink(t *testing.T) {
	c := newOrSkipHW(t)
	defer c.Close()
	var stop atomic.Bool
	defer stop.Store(true)
	go func() {
		for !stop.Load() {
			runtime.GC()
		}
	}()
	var wg sync.WaitGroup
	var calls atomic.Int64
	deadline := time.Now().Add(3 * time.Second)
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for time.Now().Before(deadline) {
				adapterInfoDeep(40, func() { _ = c.adapter.GetInfo() })
				calls.Add(1)
			}
		}()
	}
	wg.Wait()
	t.Logf("%d GetInfo calls under continuous GC", calls.Load())
}
