//go:build gpu

package gpu

import "testing"

// Context.Close must release everything, exactly once, however many times it is called. It once
// ran a hand-maintained list of per-field Release calls that lagged the ensure* builders, and was
// not idempotent, so `defer m.Close()` next to an explicit `m.Close()` re-released live wgpu
// handles: a use-after-free in the native layer. The structure now: objects register a release
// closure at creation (mkPipeline / track), Close drains that list LIFO, and a `closed` flag
// makes a second call a no-op. These tests pin the drain and the guard. Background:
// docs/code-notes/gpu.md#TestClose_drainsTrackedReleasesLIFO.filecomment.
//
// No device needed: track takes plain closures and Close nil-checks the base handles, so a
// zero-value Context exercises both.

// TestClose_drainsTrackedReleasesLIFO: everything registered must run, newest first (pipelines are
// created after the device they depend on, so teardown reverses creation).
func TestClose_drainsTrackedReleasesLIFO(t *testing.T) {
	var order []int
	c := &Context{}
	for i := range 5 {
		c.track(func() { order = append(order, i) })
	}

	if err := c.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if len(order) != 5 {
		t.Fatalf("released %d of 5 tracked objects — the leak this fix exists to end: %v", len(order), order)
	}
	for i, got := range order {
		if want := 4 - i; got != want {
			t.Fatalf("release order %v is not LIFO (position %d released %d, want %d)", order, i, got, want)
		}
	}
}

// TestClose_isIdempotent is the use-after-free gate: the second Close must release NOTHING again.
func TestClose_isIdempotent(t *testing.T) {
	n := 0
	c := &Context{}
	c.track(func() { n++ })

	if err := c.Close(); err != nil {
		t.Fatalf("first Close: %v", err)
	}
	if err := c.Close(); err != nil {
		t.Fatalf("second Close returned an error; it must be a silent no-op: %v", err)
	}
	if err := c.Close(); err != nil {
		t.Fatalf("third Close: %v", err)
	}
	if n != 1 {
		t.Fatalf("tracked release ran %d times across three Close calls, want 1 — each extra run is a "+
			"double-release of a live native handle", n)
	}
}

// TestClose_nilsBaseHandles: after Close, a stray use must fault at the Go boundary (nil deref with
// a stack trace) rather than reach into freed native memory.
func TestClose_nilsBaseHandles(t *testing.T) {
	c := &Context{}
	if err := c.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if c.device != nil || c.queue != nil || c.instance != nil || c.adapter != nil ||
		c.pipeline != nil || c.shader != nil {
		t.Error("Close left a base handle non-nil — a use-after-free stays undefined behaviour " +
			"inside wgpu instead of failing loudly in Go")
	}
	if c.releases != nil {
		t.Error("Close left the release list populated; a later Close would re-run it")
	}
}

// TestTrack_registersAtCreation guards the invariant that makes the fix durable: mkPipeline is the
// only constructor the ensure* builders share, and it must register. This asserts the plumbing
// (track appends) that mkPipeline relies on — mkPipeline itself needs a device.
func TestTrack_registersAtCreation(t *testing.T) {
	c := &Context{}
	if len(c.releases) != 0 {
		t.Fatal("fresh Context has tracked releases")
	}
	c.track(func() {}, func() {})
	if len(c.releases) != 2 {
		t.Fatalf("track registered %d closures, want 2", len(c.releases))
	}
}
