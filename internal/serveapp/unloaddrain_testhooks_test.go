//go:build goinfer_testhooks

package serveapp

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// newDrainTestServer wires a server around a SENTINEL *decoder.Model — Model.Close() on a zero Model
// is a no-op, so the drain machinery runs end-to-end with no backend. allowAdmin is on.
func newDrainTestServer(wait time.Duration) (*server, *decoder.Model) {
	m := &decoder.Model{}
	lm := &loadedModel{name: "t", model: m}
	s := &server{
		models:   map[string]*loadedModel{"t": lm},
		liveness: map[*decoder.Model]*modelLiveness{m: {refs: 1}},
		draining: map[string]struct{}{},
		cfg:      config{allowAdmin: true, unloadDrainWait: wait},
	}
	return s, m
}

func unload(s *server, query string) *httptest.ResponseRecorder {
	w := httptest.NewRecorder()
	r := httptest.NewRequest("POST", "/admin/models/unload"+query, strings.NewReader(`{"name":"t"}`))
	s.handleAdminUnload(w, r)
	return w
}

func isDraining(s *server, name string) bool {
	s.regMu.RLock()
	defer s.regMu.RUnlock()
	_, ok := s.draining[name]
	return ok
}

// TestUnloadDrain_blocksOnParkedRequest is the regression gate for the preamble use-after-free: a
// request parked INSIDE the pick→enter window holds the liveness RLock, so unload must NOT free the
// model — it returns 202 (still draining) and only completes once the request releases. The park
// happens at the preamblePark seam, which lives at the RLock-acquisition site: if someone later moves
// the lock down to enter(), the parked request stops holding it, the drain completes while it is
// parked, and the "still draining" assertion fails — which is exactly the reopened bug.
func TestUnloadDrain_blocksOnParkedRequest(t *testing.T) {
	s, _ := newDrainTestServer(50 * time.Millisecond)

	parked := make(chan struct{})
	release := make(chan struct{})
	old := preamblePark
	preamblePark = func() { close(parked); <-release }
	defer func() { preamblePark = old }()

	go s.withModel(httptest.NewRecorder(), "t", func(*loadedModel) {})
	<-parked // the request now holds the liveness RLock, parked in the preamble

	w := unload(s, "?wait=false")
	if w.Code != http.StatusAccepted {
		t.Fatalf("unload while a request is parked in the window: status %d, want 202", w.Code)
	}
	var resp struct {
		Freed bool `json:"freed"`
	}
	_ = json.Unmarshal(w.Body.Bytes(), &resp)
	if resp.Freed {
		t.Errorf("unload reported freed:true while a request still held the model")
	}
	if !isDraining(s, "t") {
		t.Errorf("model not listed as draining while the drain is blocked")
	}

	close(release) // request finishes → RLock released → drain can complete → Close runs
	deadline := time.Now().Add(2 * time.Second)
	for isDraining(s, "t") {
		if time.Now().After(deadline) {
			t.Fatalf("drain did not complete after the parked request was released")
		}
		time.Sleep(time.Millisecond)
	}
}

// TestUnloadDrain_freesWhenIdle: with no in-flight request, unload drains at once and returns
// 200 freed:true (last owner), and the model leaves the draining set.
func TestUnloadDrain_freesWhenIdle(t *testing.T) {
	s, _ := newDrainTestServer(2 * time.Second)

	w := unload(s, "")
	if w.Code != http.StatusOK {
		t.Fatalf("idle unload: status %d, want 200", w.Code)
	}
	var resp struct {
		Freed bool `json:"freed"`
	}
	_ = json.Unmarshal(w.Body.Bytes(), &resp)
	if !resp.Freed {
		t.Errorf("idle unload: freed=false, want true (last owner, drained instantly)")
	}
	if isDraining(s, "t") {
		t.Errorf("model still listed as draining after a completed idle unload")
	}
	// The registry entry is gone.
	if s.pickTest("t") != nil {
		t.Errorf("model still resolvable after unload")
	}
}

// silence unused import in builds where decoder is only used via the sentinel above.
var _ = decoder.Model{}
