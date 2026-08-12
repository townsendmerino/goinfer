package decoder

import (
	"context"
	"os"
	"runtime"
	"runtime/pprof"
	"strings"
	"testing"
	"time"
)

// TestGoroutineLeakCheck exercises the generation-goroutine lifecycle on a tiny CPU model
// and asks the runtime's goroutineleak profiler (GOEXPERIMENT=goroutineleakprofile) whether
// any goroutine is left permanently blocked and unreachable. Skips cleanly when the
// experiment is off (the profile is nil). Diagnostic — run with:
//
//	GOEXPERIMENT=goroutineleakprofile go test ./decoder -run TestGoroutineLeakCheck -v
func TestGoroutineLeakCheck(t *testing.T) {
	p := pprof.Lookup("goroutineleak")
	if p == nil {
		t.Skip("goroutineleak profile unavailable — build with GOEXPERIMENT=goroutineleakprofile")
	}
	const ckpt = "../testdata/cohere2-tiny"
	if _, err := os.Stat(ckpt); err != nil {
		t.Skipf("no tiny checkpoint (%s)", ckpt)
	}
	m, err := Load(ckpt, Options{Quant: "f32"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	prompt := []int{1, 2, 3, 4}
	sp := SamplingParams{Temperature: 0}

	// (a) full drain to completion — the goroutine must exit on its own.
	for range 3 {
		out, _ := m.Generate(context.Background(), prompt, 6, sp)
		for range out { // drain until closed
		}
	}

	// (b) read one token, cancel, and ABANDON the channel — the goroutine must observe
	// ctx cancellation and exit, not block forever on `out <- tok` with no reader.
	for range 3 {
		ctx, cancel := context.WithCancel(context.Background())
		out, _ := m.Generate(ctx, prompt, 64, sp)
		<-out    // consume one token
		cancel() // ask it to stop
		_ = out  // drop the reference; stop reading
	}

	// (c) cancel BEFORE reading anything.
	for range 3 {
		ctx, cancel := context.WithCancel(context.Background())
		_, _ = m.Generate(ctx, prompt, 64, sp)
		cancel()
	}

	// Let the abandoned goroutines run to their ctx check / send, then make their
	// channels unreachable and GC so the leak analysis can prove them leaked (or not).
	time.Sleep(200 * time.Millisecond)
	runtime.GC()
	runtime.GC()

	var sb strings.Builder
	if err := p.WriteTo(&sb, 1); err != nil {
		t.Fatalf("write goroutineleak profile: %v", err)
	}
	report := sb.String()
	// Header line is "goroutineleak profile: total N".
	total := "unknown"
	for ln := range strings.SplitSeq(report, "\n") {
		if after, ok := strings.CutPrefix(ln, "goroutineleak profile: total "); ok {
			total = after
			break
		}
	}
	t.Logf("goroutineleak total: %s", total)
	// Only fail on leaks whose stack touches goinfer's own code (ignore any runtime/test
	// harness background goroutines the profiler might list).
	if goinferLeaks := countFrames(report, "townsendmerino/goinfer"); goinferLeaks > 0 {
		t.Errorf("goroutineleak: %d leaked goroutine(s) in goinfer code:\n%s", goinferLeaks, report)
	}
}

func countFrames(report, needle string) int {
	n := 0
	for block := range strings.SplitSeq(report, "\n\n") {
		if strings.Contains(block, needle) && strings.Contains(block, "@") {
			n++
		}
	}
	return n
}
