//go:build darwin

package metal

import (
	"errors"
	"testing"
)

// TestEnsurePrefill_latchesFailure gates N-47 (audit-2026-09-10.md): a failed ensurePrefill used
// to leave r.pf == nil with no record of WHY, so every later PrefillLast call re-ran the full MSL
// compile from scratch just to panic identically again. Confirmed red without the fix: with the
// pfErr short-circuit removed, ensurePrefill fell through to the real compile attempt against this
// test's zero-value (no real Metal setup) Device and panicked with an unrelated compile-landmine
// message instead of the cached sentinel — proving the short-circuit, not the compile itself, is
// what this test pins.
//
// The fast path this test exercises (r.pfErr already set) runs entirely before ensurePrefill
// touches r.d, so a bare &resident{} with no real Device proves the short-circuit never reaches
// the expensive compile path — no Metal device needed for this half of the fix.
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
