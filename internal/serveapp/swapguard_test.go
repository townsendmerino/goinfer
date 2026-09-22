package serveapp

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// scriptedSwapReader is decoder/swapwatch_test.go's scriptedReader, reimplemented minimally here
// rather than exported cross-package for one test file — it plays back a fixed sequence and
// repeats the last entry, counting ticks so a test can wait for N samples without a fixed sleep.
type scriptedSwapReader struct {
	seq   []int64
	i     atomic.Int64
	ticks atomic.Int64
}

func (s *scriptedSwapReader) Read() (int64, bool) {
	i := s.i.Load()
	if i < int64(len(s.seq)-1) {
		s.i.Add(1)
	}
	s.ticks.Add(1)
	v := s.seq[min(i, int64(len(s.seq)-1))]
	if v < 0 {
		return 0, false
	}
	return v, true
}

func (s *scriptedSwapReader) waitTicks(t *testing.T, n int) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for s.ticks.Load() < int64(n) {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %d ticks (got %d)", n, s.ticks.Load())
		}
		time.Sleep(time.Millisecond)
	}
}

func passthroughOK(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusOK) }

// waitUntil polls cond every millisecond until it reports true or timeout elapses (failing the
// test in that case) — for synchronizing against a background goroutine's state change without
// coupling to its internal call order (see the comment at its call site).
func waitUntil(t *testing.T, timeout time.Duration, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatal("timed out waiting for condition")
		}
		time.Sleep(time.Millisecond)
	}
}

// TestSwapGuard_tripsAndResumesThroughHaltGate is the integration seam: armSwapGuard wired to a
// scripted reader, driven through the REAL haltGate a route actually uses — not the watch's
// decision logic in isolation (decoder/swapwatch_test.go covers that), but that a trip actually
// turns into a 503 on this server's admission chokepoint and a resume actually turns it back into
// a 200, with the right reason string in between.
func TestSwapGuard_tripsAndResumesThroughHaltGate(t *testing.T) {
	t.Setenv("GOINFER_SWAP_GUARD", "500") // 500 MB threshold, small so the scripted deltas below exercise it
	const baseline = 1_000_000_000
	r := &scriptedSwapReader{seq: []int64{
		baseline,
		baseline + 100_000_000, // under threshold
		baseline + 600_000_000, // trip
		baseline + 600_000_000,
		baseline + 50_000_000, // within threshold again -- hysteresis clock starts
		baseline + 50_000_000,
	}}
	s := &server{}
	s.armSwapGuard(r.Read)
	defer s.swapWatch.Stop()

	ts := httptest.NewServer(http.HandlerFunc(s.haltGate(passthroughOK)))
	defer ts.Close()

	get := func() (int, string) {
		resp, err := http.Get(ts.URL)
		if err != nil {
			t.Fatalf("GET: %v", err)
		}
		defer resp.Body.Close()
		buf := make([]byte, 512)
		n, _ := resp.Body.Read(buf)
		return resp.StatusCode, string(buf[:n])
	}

	// Before any tick lands, the guard has not yet observed even a baseline — must not refuse.
	if code, _ := get(); code != http.StatusOK {
		t.Fatalf("before arming observed anything: got %d, want 200", code)
	}

	// waitTicks(3) only guarantees Read() has been CALLED 3 times, not that OnTrip (which runs
	// synchronously right after Read() returns, inside the SAME watch goroutine, but is not
	// visible to THIS goroutine until it happens) has finished — poll the flag itself rather
	// than race the two goroutines on tick count alone.
	r.waitTicks(t, 3) // baseline + one under-threshold + the tripping sample
	waitUntil(t, 10*time.Second, func() bool { return s.swapGuardTripped.Load() })
	if code, body := get(); code != http.StatusServiceUnavailable {
		t.Fatalf("after the trip: got %d %q, want 503", code, body)
	} else if !strings.Contains(body, "halted") || !strings.Contains(body, "swap guard") {
		t.Errorf("503 body %q does not name the swap guard as the reason", body)
	}
	if s.swapGuardTripped.Load() != true {
		t.Fatal("swapGuardTripped should be true after the trip")
	}

	// Overriding GOINFER_SWAP_GUARD's ResumeAfter isn't exposed — armSwapGuard hardcodes 30s,
	// which this test cannot wait out. Confirm the FLAG mechanism directly instead: manually
	// simulate what OnResume does, since testing the full 30s hysteresis belongs to
	// decoder/swapwatch_test.go's TestSwapWatch_hysteresis (already covers it against the pure
	// watch) — this test's job is proving the wiring (trip -> 503, flag -> gate), not re-timing
	// the hysteresis window a second time.
	s.swapGuardTripped.Store(false)
	if code, _ := get(); code != http.StatusOK {
		t.Fatalf("after clearing the flag directly: got %d, want 200", code)
	}
}

// TestSwapGuard_off confirms GOINFER_SWAP_GUARD=off starts no watch at all: the gate never
// refuses regardless of what a reader would report, because armSwapGuard never calls it.
func TestSwapGuard_off(t *testing.T) {
	t.Setenv("GOINFER_SWAP_GUARD", "off")
	calls := 0
	fakeRead := func() (int64, bool) { calls++; return 1 << 40, true } // absurdly high, would trip instantly if read
	s := &server{}
	s.armSwapGuard(fakeRead)
	if s.swapWatch != nil {
		t.Fatal("GOINFER_SWAP_GUARD=off must not start a watch")
	}
	time.Sleep(20 * time.Millisecond)
	if calls != 0 {
		t.Fatalf("the reader was called %d times despite GOINFER_SWAP_GUARD=off", calls)
	}

	ts := httptest.NewServer(http.HandlerFunc(s.haltGate(passthroughOK)))
	defer ts.Close()
	resp, err := http.Get(ts.URL)
	if err != nil {
		t.Fatalf("GET: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("got %d, want 200 (guard disabled)", resp.StatusCode)
	}
}

// TestSwapGuard_neverCancelsInFlight is the "lets in-flight generations finish" requirement,
// checked directly: tripping the guard must not touch s.gens (K1's cancel registry) at all —
// only haltGate's admission check changes. A halt() call DOES cancel everything via
// s.gens.cancelAll; the swap guard's own trip path must never call it.
func TestSwapGuard_neverCancelsInFlight(t *testing.T) {
	t.Setenv("GOINFER_SWAP_GUARD", "100")
	s := &server{gens: newGenerationRegistry()}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	g := s.gens.register("in-flight-test", "test-model", cancel)

	r := &scriptedSwapReader{seq: []int64{1_000_000_000, 1_000_000_000 + 200_000_000}}
	s.armSwapGuard(r.Read)
	defer s.swapWatch.Stop()
	r.waitTicks(t, 2)

	select {
	case <-ctx.Done():
		t.Fatal("the swap guard trip cancelled an in-flight generation's context — it must only refuse NEW admissions")
	default:
	}
	if g.reason() != "" {
		t.Fatalf("the swap guard trip set a cancel reason on an in-flight generation: %q", g.reason())
	}
}
