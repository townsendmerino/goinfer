package decoder

import (
	"context"
	"fmt"
	"os"
	"runtime/pprof"
	"runtime/trace"
	"strconv"
	"sync/atomic"
	"testing"
	"time"
)

func TestZZDiagGroupedFires(t *testing.T) {
	m, err := loadBenchModel()
	if err != nil {
		t.Skipf("no model (%v)", err)
	}
	arch := m.w.arch
	fmt.Printf("NumHeads=%d NumKVHeads=%d HeadDim=%d group=%d NumLayers=%d\n",
		arch.NumHeads, arch.NumKVHeads, arch.HeadDim, arch.NumHeads/arch.NumKVHeads, arch.NumLayers)

	depth := 2048
	if d, err := strconv.Atoi(os.Getenv("GOINFER_BENCH_DEPTH")); err == nil {
		depth = d
	}
	cache := m.NewCache(depth + 8)
	tok := 785
	ids := make([]int, depth)
	for i := range ids {
		ids[i] = tok
	}
	if !m.canBatchN(depth) {
		t.Skip("no batched prefill")
	}
	if _, err := m.forwardLayersN(context.Background(), ids, cache, false); err != nil {
		t.Fatalf("prefill: %v", err)
	}

	attnTiming = true // R13 attention timer (was GOINFER_ATTN_TIMING_DEBUG=1)
	defer func() { attnTiming = false }()
	steps := 100
	if s, err := strconv.Atoi(os.Getenv("GOINFER_DIAG_STEPS")); err == nil {
		steps = s
	}
	for _, arm := range []struct {
		name string
		env  string
	}{
		{"grouped_off", "0"},
		{"grouped_on", "1"},
	} {
		setKnob(t, m, knobAttnGrouped, arm.env)
		runsBefore := atomic.LoadInt64(&attnGroupedRuns)
		attnBefore := atomic.LoadInt64(&attnElapsedNanos)

		if profPath := os.Getenv("GOINFER_DIAG_CPUPROFILE"); profPath != "" {
			f, err := os.Create(profPath + "." + arm.name + ".pprof")
			if err != nil {
				t.Fatalf("create profile: %v", err)
			}
			if err := pprof.StartCPUProfile(f); err != nil {
				t.Fatalf("start profile: %v", err)
			}
			t0 := time.Now()
			for i := 0; i < steps; i++ {
				if _, err := m.forward(tok, cache); err != nil {
					t.Fatalf("forward: %v", err)
				}
			}
			elapsed := time.Since(t0)
			pprof.StopCPUProfile()
			f.Close()
			runsAfter := atomic.LoadInt64(&attnGroupedRuns)
			attnAfter := atomic.LoadInt64(&attnElapsedNanos)
			attnElapsed := time.Duration(attnAfter - attnBefore)
			fmt.Printf("[%s] %d steps: total=%v (%.2f tok/s)  attnOnly=%v (%.1f%% of total)  attnGroupedRuns delta=%d  profile=%s\n",
				arm.name, steps, elapsed, float64(steps)/elapsed.Seconds(),
				attnElapsed, 100*attnElapsed.Seconds()/elapsed.Seconds(), runsAfter-runsBefore, profPath+"."+arm.name+".pprof")
			continue
		}

		// go tool trace, not pprof: a prior investigation in this repo
		// (CPU-decode 8ms is an idle-M artifact) found plain CPU pprof
		// MISCOUNTS idle parked workers as real cost — this is the correct
		// instrument for a park/wake question, per that finding's own
		// "use go tool trace -pprof=sync/sched" conclusion.
		if tracePath := os.Getenv("GOINFER_DIAG_TRACE"); tracePath != "" {
			f, err := os.Create(tracePath + "." + arm.name + ".trace")
			if err != nil {
				t.Fatalf("create trace: %v", err)
			}
			if err := trace.Start(f); err != nil {
				t.Fatalf("start trace: %v", err)
			}
			t0 := time.Now()
			for i := 0; i < steps; i++ {
				if _, err := m.forward(tok, cache); err != nil {
					t.Fatalf("forward: %v", err)
				}
			}
			elapsed := time.Since(t0)
			trace.Stop()
			f.Close()
			runsAfter := atomic.LoadInt64(&attnGroupedRuns)
			attnAfter := atomic.LoadInt64(&attnElapsedNanos)
			attnElapsed := time.Duration(attnAfter - attnBefore)
			fmt.Printf("[%s] %d steps: total=%v (%.2f tok/s)  attnOnly=%v (%.1f%% of total)  attnGroupedRuns delta=%d  trace=%s\n",
				arm.name, steps, elapsed, float64(steps)/elapsed.Seconds(),
				attnElapsed, 100*attnElapsed.Seconds()/elapsed.Seconds(), runsAfter-runsBefore, tracePath+"."+arm.name+".trace")
			continue
		}

		t0 := time.Now()
		for i := 0; i < steps; i++ {
			if _, err := m.forward(tok, cache); err != nil {
				t.Fatalf("forward: %v", err)
			}
		}
		elapsed := time.Since(t0)
		runsAfter := atomic.LoadInt64(&attnGroupedRuns)
		attnAfter := atomic.LoadInt64(&attnElapsedNanos)
		attnElapsed := time.Duration(attnAfter - attnBefore)
		fmt.Printf("[%s] %d steps: total=%v (%.2f tok/s)  attnOnly=%v (%.1f%% of total)  attnGroupedRuns delta=%d\n",
			arm.name, steps, elapsed, float64(steps)/elapsed.Seconds(),
			attnElapsed, 100*attnElapsed.Seconds()/elapsed.Seconds(), runsAfter-runsBefore)
	}
}
