package decoder

import (
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
	"time"
)

// The heap-vs-mapping discriminator (opt-in via GOINFER_MEM_PROBE). The point,
// from the box's thrash triage: a Go heap profile (pprof -inuse_space) can come
// back clean while RSS is huge, because mmap'd file pages (.giw / GGUF, MAP_PRIVATE
// read-only) count toward RSS but are invisible to the heap profiler. So the
// question "is the memory the Go heap or file mappings?" is answered by comparing
// runtime.MemStats against process RSS, NOT by a heap profile:
//
//   - HeapInuse high            → LIVE heap retention (e.g. per-position capture
//                                 buffers held across a loop). A real leak/holding.
//   - RSS ≫ Sys                 → non-Go memory = mmap'd model files. If it persists
//                                 after Close, it's a missing-Unmap/Close, not heap.
//   - (HeapIdle-HeapReleased)   → heap the runtime freed but hasn't returned to the
//     large, and FreeOSMemory     OS. Benign high-water (GC policy), not a leak: it
//     reclaims it              → drops after debug.FreeOSMemory().
//
// Runs from TestMain after the suite. Scope it with -run to a single test to get
// that test's post-state (e.g. -run TestEagleAcceptedLength for the accept loop).

// memRSS returns the process resident set size in bytes, cgo-free. Linux reads
// /proc/self/status; macOS/BSD shell out to `ps`. 0 if it can't be determined.
func memRSS() uint64 {
	if b, err := os.ReadFile("/proc/self/status"); err == nil { // Linux (the box)
		for line := range strings.SplitSeq(string(b), "\n") {
			if strings.HasPrefix(line, "VmRSS:") {
				if f := strings.Fields(line); len(f) >= 2 {
					if kb, err := strconv.ParseUint(f[1], 10, 64); err == nil {
						return kb * 1024
					}
				}
			}
		}
	}
	// macOS/BSD: ps reports RSS in KiB.
	if out, err := exec.Command("ps", "-o", "rss=", "-p", strconv.Itoa(os.Getpid())).Output(); err == nil {
		if kb, err := strconv.ParseUint(strings.TrimSpace(string(out)), 10, 64); err == nil {
			return kb * 1024
		}
	}
	return 0
}

// logMemProbe prints the discriminator line for the current process state.
func logMemProbe(label string) {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	rss := memRSS()
	mb := func(b uint64) float64 { return float64(b) / (1 << 20) }
	nonGo := (int64(rss) - int64(m.Sys)) >> 20 // RSS beyond what the Go runtime got from the OS ≈ mmap'd files
	fmt.Printf("[memprobe] %-30s RSS=%6.0fMB  Sys=%6.0fMB  HeapInuse=%6.0fMB  HeapIdle-Released=%6.0fMB  nonGo(RSS-Sys)=%dMB\n",
		label, mb(rss), mb(m.Sys), mb(m.HeapInuse), mb(m.HeapIdle-m.HeapReleased), nonGo)
}

// memProbeSuite samples the post-suite state, then again after forcing the runtime
// to return freed heap to the OS. The delta between the two RSS lines is the
// verdict: a big drop = benign freed-but-unreturned heap (high-water artifact);
// little drop with low HeapInuse = mmap'd files still resident (Close discipline);
// little drop with high HeapInuse = live heap retention.
func memProbeSuite() {
	if os.Getenv("GOINFER_MEM_PROBE") == "" {
		return
	}
	logMemProbe("post-suite (pre-GC)")
	runtime.GC()
	debug.FreeOSMemory()
	logMemProbe("post-suite (post-FreeOSMemory)")
}

// startMemTicker launches a background sampler (GOINFER_MEM_PROBE) that logs the
// heap-vs-RSS line every 2s. There is no per-test hook in Go's testing, so this
// names the accumulating test by INTERLEAVING: with `go test -v`, each tick prints
// to the same stdout stream as the `=== RUN`/`--- PASS` lines, so the test whose
// RUN/PASS the rising ticks fall between is the one accumulating. Bounded and
// cgo-free — safe to leave on. stop() ends it (called before the post-suite probe
// so the ticks don't muddy the FreeOSMemory delta).
func startMemTicker() (stop func()) {
	if os.Getenv("GOINFER_MEM_PROBE") == "" {
		return func() {}
	}
	done := make(chan struct{})
	go func() {
		start := time.Now()
		tk := time.NewTicker(2 * time.Second)
		defer tk.Stop()
		for {
			select {
			case <-done:
				return
			case <-tk.C:
				logMemProbe(fmt.Sprintf("tick t=%ds", int(time.Since(start).Seconds())))
			}
		}
	}()
	return func() { close(done) }
}
