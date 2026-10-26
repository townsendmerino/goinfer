package serveapp

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/vision"
)

// TestLookupLockedOnlyViaResolve enforces item-5 of the design: withModel/withModelAnthropic are the
// ONLY route from a request to a *loadedModel, because they take the liveness RLock that keeps the
// model alive for the request. pick was deleted; lookupLocked is unexported and lock-requiring. This
// lint fails the build if any non-test file other than liveness.go calls s.lookupLocked directly —
// the way a future eighth handler would accidentally skip the lock and reopen the use-after-free.
func TestLookupLockedOnlyViaResolve(t *testing.T) {
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range files {
		if strings.HasSuffix(f, "_test.go") || f == "liveness.go" {
			continue // resolveAndLock (liveness.go) is the sole caller; tests use pickTest
		}
		b, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		if bytes.Contains(b, []byte("s.lookupLocked(")) {
			t.Errorf("%s calls s.lookupLocked directly — resolve via withModel/withModelAnthropic instead "+
				"(they take the liveness RLock); only resolveAndLock in liveness.go may call it", f)
		}
	}
}

// TestRetainRelease gates the last-owner refcount: a base and its adapters share one *decoder.Model,
// so Close is safe only when the LAST entry backed by it unloads. releaseLocked must report last only
// then, and (delete-before-decide, tested via the sequence) two siblings can never both decline.
func TestRetainRelease(t *testing.T) {
	s := &server{liveness: map[*decoder.Model]*modelLiveness{}}
	m := &decoder.Model{} // sentinel; never Closed here

	s.retainLocked(m) // base
	s.retainLocked(m) // adapter sharing the base's model
	if got := s.liveness[m].refs; got != 2 {
		t.Fatalf("refs after two retains = %d, want 2", got)
	}

	ml, last := s.releaseLocked(m) // unload one sibling
	if last || ml == nil || ml.refs != 1 {
		t.Errorf("first release: last=%v refs=%d, want last=false refs=1", last, ml.refs)
	}
	_, last = s.releaseLocked(m) // unload the last owner
	if !last {
		t.Errorf("second release: want last=true (model now closable)")
	}
	if _, ok := s.liveness[m]; ok {
		t.Errorf("liveness entry not removed on last release")
	}

	// Releasing an untracked model is safe (defensive: leak-not-crash, never a spurious last-owner).
	if ml, last := s.releaseLocked(&decoder.Model{}); ml != nil || last {
		t.Errorf("release of untracked model = (%v,%v), want (nil,false)", ml, last)
	}
}

// TestCloseEntryNatives_dropsGemma4Enc is N-32 (docs/audit-2026-09-10.md): closeEntryNatives must drop gemma4Enc, or
// unloading a served Gemma 4 vision model leaves its encoder referenced from the closed entry, retaining that memory past
// unload. Closing a zero encoder must also be safe.
func TestCloseEntryNatives_dropsGemma4Enc(t *testing.T) {
	lm := &loadedModel{gemma4Enc: &vision.Gemma4Encoder{}}
	lm.closeEntryNatives()
	if lm.gemma4Enc != nil {
		t.Error("gemma4Enc still referenced after closeEntryNatives — its weight memory is not " +
			"eligible for GC after unload")
	}
}

// countingG4Tower is a Gemma 4 device tower that counts its Close calls.
type countingG4Tower struct{ closes *int }

func (countingG4Tower) Name() string                                      { return "counting" }
func (countingG4Tower) Hidden(p []float32, _ [][2]int) ([]float32, error) { return p, nil }
func (c countingG4Tower) Close() error                                    { *c.closes++; return nil }

// TestCloseEntryNatives_sharedVisionTowerClosesOnLast: the startup vision tower is the same object on every startup
// entry (an adapter and its base, say), so unloading one entry must leave it open for the others and the last unload
// must close it, exactly once. An entry that shares it with nobody closes it on its own unload.
func TestCloseEntryNatives_sharedVisionTowerClosesOnLast(t *testing.T) {
	var closes int
	tower := countingG4Tower{&closes}
	share := &visionShare{n: 2}
	base := &loadedModel{gemma4Enc: &vision.Gemma4Encoder{}, gemma4Tower: tower, visionShare: share}
	adapter := &loadedModel{gemma4Enc: base.gemma4Enc, gemma4Tower: tower, visionShare: share}

	adapter.closeEntryNatives()
	if closes != 0 {
		t.Fatalf("unloading one of two holders closed the shared tower (%d closes)", closes)
	}
	if adapter.gemma4Tower != nil || adapter.gemma4Enc != nil {
		t.Error("the unloaded entry still references the shared tower")
	}
	if base.gemma4Tower == nil {
		t.Fatal("the remaining holder lost its tower")
	}
	base.closeEntryNatives()
	base.closeEntryNatives() // idempotent
	if closes != 1 {
		t.Errorf("the last holder's unload closed the shared tower %d times, want 1", closes)
	}

	closes = 0
	alone := &loadedModel{gemma4Tower: tower}
	alone.closeEntryNatives()
	if closes != 1 {
		t.Errorf("an unshared entry's unload closed its tower %d times, want 1", closes)
	}
}
