//go:build darwin

package metal

import (
	"errors"
	"testing"
)

// TestEnsurePrefill_latchesFailure gates N-47 (audit-2026-09-10.md): a failed ensurePrefill latches its error in
// r.pfErr, so every later PrefillLast returns the cached sentinel instead of re-running the full MSL compile just to panic
// identically again.
//
// The fast path this test exercises (r.pfErr already set) runs entirely before ensurePrefill touches r.d, so a bare
// &resident{} with no real Device proves the short-circuit never reaches the expensive compile path: with the
// short-circuit removed the call falls through to the compile against this zero-value Device and panics with an
// unrelated message instead of the cached sentinel. No Metal device needed.
func TestEnsurePrefill_latchesFailure(t *testing.T) {
	sentinel := errors.New("metal prefill compile: sentinel failure from a prior attempt")
	r := &resident{pfErr: sentinel}

	defer func() {
		p := recover()
		if p == nil {
			t.Fatal("ensurePrefill did not panic despite a cached pfErr")
		}
		if err, ok := p.(error); !ok || !errors.Is(err, sentinel) {
			t.Fatalf("ensurePrefill panicked with %v (%T), want the cached sentinel error", p, p)
		}
		if r.pf != nil {
			t.Error("ensurePrefill set r.pf on the cached-failure fast path")
		}
	}()
	r.ensurePrefill()
}
