package serveapp

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// TestPrepare_temperatureValidation gates G4: temperature has the same lower bound as top_p.
// A negative temperature is rejected (previously it was accepted and decoded greedily, since
// SampleWithInfo treats Temperature <= 0 as argmax — a validation inconsistency, not inverted
// output). 0 (greedy) and positive values stay valid.
func TestPrepare_temperatureValidation(t *testing.T) {
	lm := &loadedModel{}
	f := func(v float64) *float64 { return &v }

	if _, err := lm.prepare(sampling{Temperature: f(-1)}, []int{1}, true); err == nil {
		t.Errorf("temperature=-1 was accepted; want a 400-shaped error (top_p=-1 already errors)")
	}
	// 0 is greedy and must remain valid.
	if _, err := lm.prepare(sampling{Temperature: f(0)}, []int{1}, true); err != nil {
		t.Errorf("temperature=0 (greedy) was rejected: %v", err)
	}
	if _, err := lm.prepare(sampling{Temperature: f(0.8)}, []int{1}, true); err != nil {
		t.Errorf("temperature=0.8 was rejected: %v", err)
	}
	// Consistency with top_p, which the same call rejects below zero.
	if _, err := lm.prepare(sampling{TopP: f(-1)}, []int{1}, true); err == nil {
		t.Errorf("top_p=-1 was accepted; the two params must be consistent")
	}
}

// TestPick_modelValidation gates G6: an unknown NON-EMPTY model name is rejected on both a
// single-model and a multi-model server, while an omitted name still routes on a single-model
// server. Previously a single-model server served any name (confident wrong-model output).
func TestPick_modelValidation(t *testing.T) {
	single := &server{models: map[string]*loadedModel{"m": {name: "m"}}}
	if single.pick("m") == nil {
		t.Errorf("single-model: correct name did not resolve")
	}
	if single.pick("") == nil {
		t.Errorf("single-model: omitted name should route to the only model")
	}
	if single.pick("does-not-exist") != nil {
		t.Errorf("single-model: unknown name resolved (should be rejected → modelNotFound)")
	}

	multi := &server{models: map[string]*loadedModel{"a": {name: "a"}, "b": {name: "b"}}}
	if multi.pick("a") == nil {
		t.Errorf("multi-model: correct name did not resolve")
	}
	if multi.pick("does-not-exist") != nil {
		t.Errorf("multi-model: unknown name resolved")
	}
	if multi.pick("") != nil {
		t.Errorf("multi-model: omitted name is ambiguous and must not resolve")
	}
}

// TestHandleChat_emptyMessages gates G5: a chat request with no messages is a 400 naming the
// field, not a 200 generated from a BOS-only prompt. Reaches the guard at the top of handleChat
// (before model routing), so an empty server suffices.
func TestHandleChat_emptyMessages(t *testing.T) {
	s := &server{models: map[string]*loadedModel{}}
	for _, body := range []string{`{"model":"m"}`, `{"model":"m","messages":[]}`} {
		r := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
		w := httptest.NewRecorder()
		s.handleChat(w, r)
		if w.Code != http.StatusBadRequest {
			t.Errorf("body %s: status %d, want 400", body, w.Code)
		}
		if !strings.Contains(w.Body.String(), "messages") {
			t.Errorf("body %s: 400 does not name the messages field: %s", body, w.Body.String())
		}
	}
}
