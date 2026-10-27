package serveapp

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/internal/swapguard"
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
	// internal/mem/probe's TestWatch_hysteresis (already covers it against the pure
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

// The two-source guard (swapguard.Select): the server's own swap at the default threshold, system swap at the backstop, and the system-wide
// guard alone where the per-process figure is unavailable. Readers are scripted: a fixed baseline, then a level that holds.
func constSwap(base, level int64) *scriptedSwapReader {
	return &scriptedSwapReader{seq: []int64{base, level}}
}

// ownSwap scripts the per-process reader: Select reads it once to learn that it works, so the first entry is that probe, the second the watch's baseline.
func ownSwap(base, level int64) *scriptedSwapReader {
	return &scriptedSwapReader{seq: []int64{base, base, level}}
}

// armBoth arms a server's guard with a 5 ms poll and returns it with its two readers' tick waits done: enough samples that a trip, if one is coming, has happened.
func armBoth(t *testing.T, own, sys *scriptedSwapReader) *server {
	t.Helper()
	t.Setenv("GOINFER_SWAP_GUARD", "") // the default: 512 MB own, 4096 MB backstop
	old := swapguard.PollInterval
	swapguard.PollInterval = 5 * time.Millisecond
	t.Cleanup(func() { swapguard.PollInterval = old })
	s := &server{}
	if own == nil {
		s.armSwapGuardSources(nil, "test: no per-process reader", sys.Read)
	} else {
		s.armSwapGuardSources(own.Read, "", sys.Read)
	}
	t.Cleanup(s.swapWatch.Stop)
	sys.waitTicks(t, 30)
	if own != nil {
		own.waitTicks(t, 30)
	}
	return s
}

const (
	gb = int64(1_000_000_000)
	mB = int64(1_000_000)
)

func TestSwapGuard_otherProcessesSwapDoesNotTripTheServerGuard(t *testing.T) {
	// Other processes push system swap up 2 GB (below the 4096 MB backstop) while the server's own stays flat: no trip.
	s := armBoth(t, ownSwap(0, 0), constSwap(9*gb, 11*gb))
	if s.swapGuardTripped.Load() {
		t.Fatal("system swap +2 GB with the server flat tripped the guard")
	}
	// The planted old reader: the same system figure watched alone at 512 MB (the guard as it was) trips on exactly this sequence. If this stops
	// tripping, the test above no longer distinguishes the two guards.
	old := armBoth(t, nil, constSwap(9*gb, 11*gb))
	waitUntil(t, 5*time.Second, func() bool { return old.swapGuardTripped.Load() })
}

func TestSwapGuard_theServersOwnSwapTrips(t *testing.T) {
	s := armBoth(t, ownSwap(0, 600*mB), constSwap(9*gb, 9*gb)) // the server (pid and descendants) +600 MB, the system flat
	waitUntil(t, 5*time.Second, func() bool { return s.swapGuardTripped.Load() })
	under := armBoth(t, ownSwap(0, 400*mB), constSwap(9*gb, 9*gb)) // +400 MB is under the default
	if under.swapGuardTripped.Load() {
		t.Fatal("the server's own swap +400 MB tripped the 512 MB default")
	}
}

func TestSwapGuard_systemBackstopTrips(t *testing.T) {
	s := armBoth(t, ownSwap(0, 0), constSwap(9*gb, 14*gb)) // system +5 GB with the server flat: the backstop at 4096 MB
	waitUntil(t, 5*time.Second, func() bool { return s.swapGuardTripped.Load() })
	under := armBoth(t, ownSwap(0, 0), constSwap(9*gb, 12*gb)) // +3 GB is under it
	if under.swapGuardTripped.Load() {
		t.Fatal("system swap +3 GB with the server flat tripped the 4096 MB backstop")
	}
}

func TestSwapGuard_fallsBackToSystemWideWithoutAPerProcessReader(t *testing.T) {
	s := armBoth(t, nil, constSwap(9*gb, 9*gb+600*mB)) // no per-process reader: system-wide +600 MB trips the 512 MB default
	waitUntil(t, 5*time.Second, func() bool { return s.swapGuardTripped.Load() })
	// A per-process reader that cannot read (no VmSwap) falls back the same way.
	t.Setenv("GOINFER_SWAP_GUARD", "")
	old := swapguard.PollInterval
	swapguard.PollInterval = 5 * time.Millisecond
	defer func() { swapguard.PollInterval = old }()
	sys := constSwap(9*gb, 9*gb+600*mB)
	u := &server{}
	u.armSwapGuardSources(func() (int64, bool) { return 0, false }, "", sys.Read)
	defer u.swapWatch.Stop()
	waitUntil(t, 5*time.Second, func() bool { return u.swapGuardTripped.Load() })
}

// The halt a runtime load causes, and its cure (B34 step 1): the serving watch is armed before a runtime (admin) load, the load's swap trips it, and since swapped-out
// pages stay in swap the gate never re-opens by itself. publishLoaded re-baselines when a load has finished: the gate re-opens at once, and growth past
// the threshold over the NEW baseline trips it again.
func TestSwapGuard_rebaselineAfterALoadReopensTheGateAndTripsOnNewGrowth(t *testing.T) {
	t.Setenv("GOINFER_SWAP_GUARD", "500")
	old := swapguard.PollInterval
	swapguard.PollInterval = time.Millisecond
	t.Cleanup(func() { swapguard.PollInterval = old })
	const base = int64(1_000_000_000)
	var cur, reads atomic.Int64
	cur.Store(base)
	s := &server{}
	s.armSwapGuard(func() (int64, bool) { reads.Add(1); return cur.Load(), true })
	defer s.swapWatch.Stop()
	waitUntil(t, 10*time.Second, func() bool { return reads.Load() >= 3 }) // the baseline has landed

	cur.Store(base + 900_000_000) // the model load puts 900 MB in swap
	waitUntil(t, 10*time.Second, func() bool { return s.swapGuardTripped.Load() })

	// The old behaviour, as a control: nothing but time passes and the figure does not fall, so the gate stays shut.
	n := reads.Load()
	waitUntil(t, 10*time.Second, func() bool { return reads.Load() >= n+20 })
	if !s.swapGuardTripped.Load() {
		t.Fatal("VOID: the gate re-opened by itself with the swap still in place, so the re-baseline below proves nothing")
	}

	s.rebaselineSwapGuard("m35") // publishLoaded does this once the load has finished
	waitUntil(t, 10*time.Second, func() bool { return !s.swapGuardTripped.Load() })

	cur.Store(base + 900_000_000 + 400_000_000) // under the threshold over the new baseline
	n = reads.Load()
	waitUntil(t, 10*time.Second, func() bool { return reads.Load() >= n+20 })
	if s.swapGuardTripped.Load() {
		t.Error("400 MB over the new baseline tripped the gate; want quiet under a 500 MB threshold")
	}
	cur.Store(base + 900_000_000 + 600_000_000)
	waitUntil(t, 10*time.Second, func() bool { return s.swapGuardTripped.Load() })

	(&server{}).rebaselineSwapGuard("no watch armed") // GOINFER_SWAP_GUARD=off leaves swapWatch nil: a no-op, not a panic
}

// publishLoaded is the caller that matters (the admin load and the web UI's both end there): a finished load re-opens a gate the load itself tripped.
// (The lost-race arm closes the loaded model, which needs a real one; it re-baselines at the same line, before the race is decided.)
func TestSwapGuard_publishLoadedRebaselines(t *testing.T) {
	t.Setenv("GOINFER_SWAP_GUARD", "500")
	old := swapguard.PollInterval
	swapguard.PollInterval = time.Millisecond
	t.Cleanup(func() { swapguard.PollInterval = old })
	const base = int64(1_000_000_000)
	var cur, reads atomic.Int64
	cur.Store(base)
	s := &server{models: map[string]*loadedModel{}}
	s.armSwapGuard(func() (int64, bool) { reads.Add(1); return cur.Load(), true })
	defer s.swapWatch.Stop()
	waitUntil(t, 10*time.Second, func() bool { return reads.Load() >= 3 })
	cur.Store(base + 900_000_000)
	waitUntil(t, 10*time.Second, func() bool { return s.swapGuardTripped.Load() })

	if !s.publishLoaded(&loadedModel{name: "m"}) {
		t.Fatal("publishLoaded refused a fresh name")
	}
	waitUntil(t, 10*time.Second, func() bool { return !s.swapGuardTripped.Load() })
}
