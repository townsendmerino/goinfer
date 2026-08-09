package serveapp

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
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
