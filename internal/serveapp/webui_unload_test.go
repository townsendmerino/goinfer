package serveapp

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/townsendmerino/goinfer/internal/loadflags"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// postUnload posts {"name": name} to handleWebUnload directly, mirroring postLoad.
func postUnload(s *server, name string) *httptest.ResponseRecorder {
	b, _ := json.Marshal(webUnloadReq{Name: name})
	r := httptest.NewRequest(http.MethodPost, "/web/models/unload", strings.NewReader(string(b)))
	w := httptest.NewRecorder()
	s.handleWebUnload(w, r)
	return w
}

// TestWebUnload_publishesRemoval: W32's own surface, over the machinery admin unload and
// unloaddrain_testhooks_test.go already prove safe — this only has to show the web route reaches
// that machinery, that a freed model is really gone from /v1/models, and that unloading it again is
// a plain 404 (not a panic, not a second free).
func TestWebUnload_publishesRemoval(t *testing.T) {
	_, root := fakeCache(t)
	p := pulledModel(t, root, "gone.gguf")
	fakeLoads(t, nil)
	s, err := newServer(config{web: true, load: loadflags.Flags{Backend: "cpu"}})
	if err != nil {
		t.Fatal(err)
	}
	if w := postLoad(s, context.Background(), p); w.Code != http.StatusOK {
		t.Fatalf("load: %d %s", w.Code, w.Body.String())
	}

	w := postUnload(s, "gone")
	if w.Code != http.StatusOK {
		t.Fatalf("unload: %d %s, want 200", w.Code, w.Body.String())
	}
	var body map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatalf("unload body not JSON: %s", w.Body.String())
	}
	// freed is false here, honestly: fakeLoads's entries carry no real *decoder.Model (retainLocked
	// is a no-op for a nil model, liveness.go:30-32), so there is nothing for the liveness tracker
	// to own — the same reason every other load test in this file never asserts freed at all.
	if body["id"] != "gone" || body["status"] != "unloaded" || body["freed"] != false {
		t.Errorf("unload body = %+v, want id=gone status=unloaded freed=false", body)
	}

	mw := httptest.NewRecorder()
	s.handleModels(mw, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	if strings.Contains(mw.Body.String(), `"gone"`) {
		t.Errorf("/v1/models still lists the unloaded model: %s", mw.Body.String())
	}

	w = postUnload(s, "gone")
	if w.Code != http.StatusNotFound {
		t.Errorf("unloading an already-unloaded name: %d %s, want 404", w.Code, w.Body.String())
	}
}

// TestWebUnload_unknownNameIs404: unlike load, unload takes no filesystem path — the only "policy"
// is that the name must already be in the registry GET /v1/models already publishes. A name never
// loaded is exactly the same shape of refusal as one already gone.
func TestWebUnload_unknownNameIs404(t *testing.T) {
	s, err := newServer(config{web: true, load: loadflags.Flags{Backend: "cpu"}})
	if err != nil {
		t.Fatal(err)
	}
	w := postUnload(s, "never-loaded")
	if w.Code != http.StatusNotFound {
		t.Errorf("unload of an unknown name: %d %s, want 404", w.Code, w.Body.String())
	}
}

// TestWebUnload_freedTrueForARealModel proves freed:true end to end through the web route, using the
// committed tiny fixture loaded directly (not via tinyServed, which registers its own t.Cleanup
// Close — this test's unload IS the close, via the real drain, and Close is documented "safe to call
// ONCE", so the two must not compete over the same *decoder.Model). retainLocked is called here to
// match the one-entry-one-ref invariant every real load keeps.
func TestWebUnload_freedTrueForARealModel(t *testing.T) {
	p := filepath.Join("..", "..", "testdata", "glm-tiny.gguf")
	if _, err := os.Stat(p); err != nil {
		t.Skipf("no committed tiny fixture at %s", p)
	}
	m, err := decoder.Load(p, decoder.Options{Backend: "cpu", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load tiny fixture: %v", err)
	}
	lm := &loadedModel{name: "tiny", model: m}
	s := &server{models: map[string]*loadedModel{"tiny": lm}}
	s.cfg.web = true
	s.cfg.unloadDrainWait = 2 * time.Second // nothing is in flight; the drain should finish well inside this
	s.regMu.Lock()
	s.retainLocked(lm.model)
	s.regMu.Unlock()

	w := postUnload(s, "tiny")
	if w.Code != http.StatusOK {
		t.Fatalf("unload: %d %s, want 200", w.Code, w.Body.String())
	}
	var body map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatalf("unload body not JSON: %s", w.Body.String())
	}
	if body["freed"] != true {
		t.Errorf("unload body = %+v, want freed=true — this was the only owner of a real model", body)
	}
}

// TestUnloadSuggestion: bullet 6 of W32 — a fit refusal should name a resident model to unload, when
// one exists, and must never invent one for an unrelated error.
func TestUnloadSuggestion(t *testing.T) {
	empty := &server{}
	other := fmt.Errorf("decoder: encode: %w", fmt.Errorf("boom"))
	if got := empty.unloadSuggestion(other); got != "" {
		t.Errorf("non-fit error: suggestion = %q, want \"\"", got)
	}
	if got := empty.unloadSuggestion(decoder.ErrWontFitResident); got != "" {
		t.Errorf("fit error, nothing resident: suggestion = %q, want \"\" (no fix to offer)", got)
	}

	s, lm := tinyServed(t)
	wrapped := fmt.Errorf("decoder: won't fit: %w", decoder.ErrWontFitResident)
	got := s.unloadSuggestion(wrapped)
	want := lm.model.ResidentWeightBytes() + lm.model.ExtraResidentBytes()
	if !strings.Contains(got, `Unload "tiny"`) || !strings.Contains(got, fmt.Sprintf("%.1f GB", float64(want)/(1<<30))) {
		t.Errorf("suggestion = %q, want it to name %q and its size (%.1f GB)", got, "tiny", float64(want)/(1<<30))
	}
	if got := s.unloadSuggestion(other); got != "" {
		t.Errorf("non-fit error with a model resident: suggestion = %q, want \"\" — must not fire on the wrong error", got)
	}
}
