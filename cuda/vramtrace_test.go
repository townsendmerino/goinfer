//go:build cuda && goinfer_testhooks

package cuda

import (
	"fmt"
	gpu "github.com/townsendmerino/goinfer/cuda/internal/gpudev"
	"github.com/townsendmerino/goinfer/internal/mem"
	"os"
	"runtime"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"
)

// TestMain installs an OPT-IN VRAM sampler: does free VRAM decline monotonically across the heavy tier (accumulation:
// something is not freeing), or does it recover after each test with a high-water mark above the card (a genuine
// environment limit)?
//
// Go's testing package exposes no per-test hook, so the test boundaries are taken from `-v` output and joined to a
// timestamped sample stream by wall clock.
//
// IT USES cuMemGetInfo, deliberately: nvidia-smi is a DIFFERENT instrument (the two disagree by a context-sized margin),
// and a reporting-gap decomposition only closes with cuMemGetInfo on both sides.
//
// WHAT IT COSTS, because it perturbs what it measures: the sampler holds its own context, which reserves ~106,954,752 B, so
// every reading is offset by roughly that and the run has that much less to work with. The SHAPE (monotonic versus
// sawtooth) is what this is for, and a constant offset leaves it unaffected; absolute figures from a traced run are not
// comparable with an untraced one.
//
// Off unless GOINFER_VRAM_TRACE=1, so no ordinary run pays the context or the polling.
func TestMain(m *testing.M) {
	// The flash-decode lane is DEFAULT ON (R6). It is checked before split-KV in the decode dispatch and is not bit-identical to
	// the exact path, so a test that loads a resident with the variable unset would silently exercise the lane at >= 2048 keys,
	// and a split-KV or exact-path comparison would compare the lane against itself. Pin it OFF for the suite; the flash-decode
	// tests set the variable themselves (t.Setenv overrides this), and a test of the DEFAULT must unset it explicitly (see
	// flash_decode_default_test.go's unsetenv).
	if _, set := os.LookupEnv("GOINFER_CUDA_FLASH_DECODE"); !set {
		os.Setenv("GOINFER_CUDA_FLASH_DECODE", "0")
	}
	stop := func() {}
	if os.Getenv("GOINFER_VRAM_TRACE") != "" {
		stop = startVRAMTrace()
	}
	// MM3 (G-MM3a-2, G-MM3a-3): a test's own buffers lease on the test line, and every device the run releases must have had
	// nothing unplanned and must close to zero.
	gpu.UnlinedLine = mem.LineTest
	gpu.ReleasedForTest = recordReleased
	gpu.NewLeaser = trackOpen(gpu.NewLeaser) // after the VRAM sampler's device, which stays open for the whole run
	code := m.Run()
	stop()
	if bad := ledgerBreaks(); len(bad) > 0 {
		fmt.Fprintf(os.Stderr, "FAIL: %d released device(s) broke the ledger's invariants (I1 nothing unplanned, I2 close to zero):\n", len(bad))
		for _, b := range bad {
			fmt.Fprintln(os.Stderr, "  "+b)
		}
		code = 1
	}
	if open := openDevices(); len(open) > 0 {
		fmt.Fprintf(os.Stderr, "FAIL: %d device(s) never released in this run, by the test that created each: %s\n", len(open),
			strings.Join(open, ", "))
		code = 1
	}
	os.Exit(code)
}

var (
	openMu   sync.Mutex
	openDevs = map[gpu.Leaser]string{}
)

// trackOpen wraps the backend's leaser constructor so every device CreateSystemDefaultDevice creates is recorded as open,
// with the test on the creating goroutine's stack, until markReleased sees its release. Devices share the primary context and
// its small-allocation pool, so one left open moves what a later device's delta measures
// (docs/measurements/sb12-memtests-2026-10-28.md §2).
func trackOpen(next func() gpu.Leaser) func() gpu.Leaser {
	return func() gpu.Leaser {
		l := next()
		openMu.Lock()
		openDevs[l] = creatingTest()
		openMu.Unlock()
		return l
	}
}

// markReleased drops a released device from the open set. recordReleased calls it; a test that swaps gpu.ReleasedForTest
// must call it from its own hook, or its devices read as open.
func markReleased(d *Device) {
	openMu.Lock()
	delete(openDevs, d.Leaser())
	openMu.Unlock()
}

// openDevices names the test that created each device still open, sorted.
func openDevices() []string {
	openMu.Lock()
	defer openMu.Unlock()
	out := make([]string, 0, len(openDevs))
	for _, name := range openDevs {
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}

// creatingTest is the outermost Test or Benchmark function on the caller's stack. A device created off a test goroutine (a
// resident build creates its device on the executor thread) is named by the tests whose goroutines are in flight at that moment.
func creatingTest() string {
	pc := make([]uintptr, 64)
	frames := runtime.CallersFrames(pc[:runtime.Callers(2, pc)])
	name := ""
	for {
		f, more := frames.Next()
		if top := testOf(f.Function); top != "" {
			name = top
		}
		if !more {
			break
		}
	}
	if name != "" {
		return name
	}
	return "(off a test goroutine, while in flight: " + strings.Join(testsInFlight(), " ") + ")"
}

// testOf is the top-level Test or Benchmark name of a qualified function name, or "".
func testOf(fn string) string {
	fn = fn[strings.LastIndex(fn, "/")+1:]
	if _, after, ok := strings.Cut(fn, "."); ok {
		if top, _, _ := strings.Cut(after, "."); strings.HasPrefix(top, "Test") || strings.HasPrefix(top, "Benchmark") {
			return top
		}
	}
	return ""
}

// testsInFlight is every Test or Benchmark function on any goroutine's stack, sorted.
func testsInFlight() []string {
	buf := make([]byte, 1<<20)
	for {
		n := runtime.Stack(buf, true)
		if n < len(buf) {
			buf = buf[:n]
			break
		}
		buf = make([]byte, 2*len(buf))
	}
	seen := map[string]bool{}
	for _, line := range strings.Split(string(buf), "\n") {
		if line == "" || line[0] == '\t' || strings.HasPrefix(line, "goroutine ") || strings.HasPrefix(line, "created by ") {
			continue
		}
		fn, _, _ := strings.Cut(line, "(")
		if top := testOf(fn); top != "" {
			seen[top] = true
		}
	}
	out := make([]string, 0, len(seen))
	for t := range seen {
		out = append(out, t)
	}
	sort.Strings(out)
	return out
}

var (
	releasedMu   sync.Mutex
	releasedSeen int
	releasedBad  []string
)

// recordReleased is gpu.ReleasedForTest in this package's tests: it keeps every released device whose ledger breaks I1 or I2.
func recordReleased(d *Device) {
	markReleased(d)
	b := ledgerBreak(ledgerOf(d))
	releasedMu.Lock()
	defer releasedMu.Unlock()
	releasedSeen++
	if b != "" {
		releasedBad = append(releasedBad, b)
	}
}

// ledgerBreak says how a released device's ledger breaks I1 (an unplanned byte ever) or I2 (a lease open after release), or "".
// A device with no ledger breaks I1: nothing it allocated was leased.
func ledgerBreak(l *mem.Ledger) string {
	if l == nil {
		return "a device with no ledger: its allocations were not leased"
	}
	u := l.Usage()
	if u.PeakUnplannedCharged > 0 || u.Leases > 0 {
		return fmt.Sprintf("unplanned peak %d bytes; %d leases (%d bytes) open after release, by line %v",
			u.PeakUnplannedCharged, u.Leases, u.Charged, u.ByLine)
	}
	return ""
}

// ledgerOf is the ledger createDevice attached to d, or nil.
func ledgerOf(d *Device) *mem.Ledger {
	if l, ok := d.Leaser().(*deviceLedger); ok {
		return &l.Ledger
	}
	return nil
}

func ledgerBreaks() []string {
	releasedMu.Lock()
	defer releasedMu.Unlock()
	fmt.Fprintf(os.Stderr, "ledger: %d device(s) released in this run\n", releasedSeen)
	return releasedBad
}

// startVRAMTrace polls free VRAM and prints one line per sample. Returns a stop function.
//
// Lines are machine-readable and go to STDERR. Not a style choice: `go test` captures the binary's
// stdout and replays it attributed to tests, and anything written outside a test's context — which
// is exactly where a TestMain-owned sampler goroutine lives — is silently dropped. Verified: a
// stderr probe from TestMain appeared, an identical stdout probe did not. The join to test
// boundaries is by wall clock, and both streams are captured together with 2>&1.
func startVRAMTrace() func() {
	dev, err := CreateSystemDefaultDevice()
	if err != nil {
		fmt.Fprintf(os.Stderr, "VRAMTRACE unavailable: %v\n", err)
		return func() {}
	}
	done := make(chan struct{})
	stopped := make(chan struct{})
	sample := func(now time.Time) {
		free, total, e := dev.Context().MemInfo()
		if e != nil {
			fmt.Fprintf(os.Stderr, "VRAMTRACE %d err=%v\n", now.UnixNano(), e)
			return
		}
		fmt.Fprintf(os.Stderr, "VRAMTRACE %d free=%d total=%d\n", now.UnixNano(), free, total)
	}
	go func() {
		defer close(stopped)
		// 50 ms, and a sample IMMEDIATELY. A 200 ms ticker produced zero samples on a test that
		// completes in 0.04 s — and it completes that fast precisely BECAUSE the tracer's context
		// holds VRAM, so a drain-to-exhaustion test has less to exhaust. The instrument shortened
		// the very interval it was trying to sample.
		sample(time.Now())
		t := time.NewTicker(50 * time.Millisecond)
		defer t.Stop()
		for {
			select {
			case <-done:
				return
			case now := <-t.C:
				sample(now)
			}
		}
	}()
	return func() { close(done); <-stopped }
}
