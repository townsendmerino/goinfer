package serveapp

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/vision"
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

// closeCounter is a device tower accelerator that counts its Close calls.
type closeCounter struct{ closed int }

func (c *closeCounter) Name() string                                  { return "fake" }
func (c *closeCounter) Close() error                                  { c.closed++; return nil }
func (c *closeCounter) Hidden([]float32, [][3]int) ([]float32, error) { return nil, nil }
func (c *closeCounter) Blocks([]float32, int) ([]float32, error)      { return nil, nil }

type closeCounterG4 struct{ closeCounter }

func (c *closeCounterG4) Hidden([]float32, [][2]int) ([]float32, error) { return nil, nil }

// TestCloseEntryNatives_closesDeviceTowers: unload closes each device tower an entry built (GLM-OCR's and Qwen3.5's
// grid towers, Gemma 4's vision tower and its audio blocks) once, and drops it; their device memory is released only
// by Close.
func TestCloseEntryNatives_closesDeviceTowers(t *testing.T) {
	glm, qwen, g4v, g4a := &closeCounter{}, &closeCounter{}, &closeCounterG4{}, &closeCounter{}
	lm := &loadedModel{glm: &glmOcrTower{acc: glm}, qwen3: &qwen3Tower{acc: qwen}, gemma4Tower: g4v, gemma4AudioAcc: g4a}
	lm.closeEntryNatives()
	lm.closeEntryNatives()
	for name, n := range map[string]int{"GLM-OCR": glm.closed, "Qwen3.5": qwen.closed, "Gemma 4 vision": g4v.closed, "Gemma 4 audio": g4a.closed} {
		if n != 1 {
			t.Errorf("the %s device tower was closed %d times, want 1", name, n)
		}
	}
	if lm.glm != nil || lm.qwen3 != nil || lm.gemma4Tower != nil || lm.gemma4AudioAcc != nil {
		t.Error("a device tower is still referenced after closeEntryNatives")
	}
}

// TestCloseEntryNatives_dropsGemma4Enc is N-32 (docs/audit-2026-09-10.md): closeEntryNatives nils
// venc/qwenEnc/vproj and must also drop gemma4Enc, or unloading a served Gemma 4 vision model leaves its
// encoder (weights, no native Close of its own, same shape as vproj) referenced from the closed entry,
// retaining that memory past unload. gemma4Enc has no Close method, so this only observes that the reference
// is dropped.
func TestCloseEntryNatives_dropsGemma4Enc(t *testing.T) {
	lm := &loadedModel{gemma4Enc: &vision.Gemma4Encoder{}}
	lm.closeEntryNatives()
	if lm.gemma4Enc != nil {
		t.Error("gemma4Enc still referenced after closeEntryNatives — its weight memory is not " +
			"eligible for GC after unload")
	}
}
