package serveapp

import (
	"io"
	"net/http"
	"net/http/httptest"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"
)

// panicReader fails the test if the handler reads from it — proves the Content-Length
// pre-check rejected the request WITHOUT touching the body (G1a: costs nothing).
type panicReader struct{ t *testing.T }

func (p panicReader) Read([]byte) (int, error) {
	p.t.Fatalf("body was read despite Content-Length exceeding the cap — pre-check did not fire")
	return 0, io.EOF
}
func (panicReader) Close() error { return nil }

// TestOversizeBody_ContentLengthRejectedCheaply is the G1 gate. It asserts, on a body whose
// declared Content-Length exceeds the cap, all three properties the field report demanded —
// each capable of failing:
//   - status 413 with the limit named in the JSON body
//   - the rejection reads none of the body (bounded time + allocation, by construction)
func TestOversizeBody_ContentLengthRejectedCheaply(t *testing.T) {
	const cap = 512 << 10 // 512 KiB, so 1/20/40 MB all exceed it
	h := maxBytes(cap, func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Model string `json:"model"`
		}
		if !decodeJSON(w, r, &req) {
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"ok": true})
	})

	// Reject is O(1) in body size: measure at 1 MB, 20 MB, 40 MB (the field-report sizes).
	for _, mb := range []int64{1, 20, 40} {
		declared := mb << 20
		r := httptest.NewRequest("POST", "/v1/chat/completions", panicReader{t})
		r.ContentLength = declared // panicReader.Read fails the test if the body is touched
		w := httptest.NewRecorder()

		var m0, m1 runtime.MemStats
		runtime.GC()
		runtime.ReadMemStats(&m0)
		t0 := time.Now()
		h(w, r)
		dur := time.Since(t0)
		runtime.ReadMemStats(&m1)
		delta := int64(m1.TotalAlloc - m0.TotalAlloc)
		t.Logf("declared %2d MB → status %d in %v, heap Δ %d B", mb, w.Code, dur, delta)

		if w.Code != http.StatusRequestEntityTooLarge {
			t.Fatalf("%d MB: status = %d, want 413", mb, w.Code)
		}
		if !strings.Contains(w.Body.String(), "524288") { // the cap, named
			t.Errorf("%d MB: 413 body does not name the limit: %s", mb, w.Body.String())
		}
		if !strings.Contains(w.Body.String(), strconv.FormatInt(declared, 10)) { // received size, named
			t.Errorf("%d MB: 413 body does not name the received size: %s", mb, w.Body.String())
		}
		if dur > time.Second {
			t.Errorf("%d MB: rejection took %v, want < 1s (a full read/tokenize would be far slower)", mb, dur)
		}
		if delta > 1<<20 {
			t.Errorf("%d MB: rejection allocated %d bytes, want < 1 MiB (must not buffer the body)", mb, delta)
		}
	}
}

// TestOversizeBody_BackstopBounded is the G1 backstop gate: a chunked body (Content-Length
// unknown, so the pre-check cannot fire) is bounded by MaxBytesReader — the handler reads at
// most cap+1 bytes, returns 413 naming the limit, and allocation stays bounded regardless of
// how much the client tries to send.
func TestOversizeBody_BackstopBounded(t *testing.T) {
	const cap = 1 << 20
	h := maxBytes(cap, func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Model string `json:"model"`
		}
		if !decodeJSON(w, r, &req) {
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"ok": true})
	})

	// A 64 MiB body with Content-Length = -1 (chunked): the pre-check is skipped, MaxBytesReader
	// must stop the read at the cap.
	huge := strings.NewReader(`{"model":"` + strings.Repeat("x", 64<<20) + `"}`)
	r := httptest.NewRequest("POST", "/v1/chat/completions", huge)
	r.ContentLength = -1
	w := httptest.NewRecorder()

	var m0, m1 runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&m0)
	t0 := time.Now()
	h(w, r)
	dur := time.Since(t0)
	runtime.ReadMemStats(&m1)

	if w.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("status = %d, want 413", w.Code)
	}
	if !strings.Contains(w.Body.String(), "1048576") {
		t.Errorf("413 body does not name the limit: %s", w.Body.String())
	}
	if dur > 2*time.Second {
		t.Errorf("backstop rejection took %v, want < 2s", dur)
	}
	// Allocation must be bounded by the cap, not the 64 MiB the client sent.
	if delta := int64(m1.TotalAlloc - m0.TotalAlloc); delta > 8<<20 {
		t.Errorf("backstop allocated %d bytes, want < 8 MiB for a 64 MiB body", delta)
	}
}

// TestPromptByteBudget gates G1c: an input whose text cannot fit the context window is
// rejected before tokenization, and a servable input is never rejected.
func TestPromptByteBudget(t *testing.T) {
	const ctx, maxTok = 4096, 32 // 4k-token window, longest token 32 bytes → 131072-byte ceiling
	// Servable: right at the ceiling passes (never over-reject a fittable prompt).
	if err := promptByteBudgetError(ctx*maxTok, ctx, maxTok); err != nil {
		t.Errorf("input at the ceiling was rejected: %v", err)
	}
	// Unservable: one byte over needs > ctx tokens.
	if err := promptByteBudgetError(ctx*maxTok+1, ctx, maxTok); err == nil {
		t.Errorf("over-ceiling input was NOT rejected")
	}
	// A 20 MiB body against a 4k model is rejected (the reported ~27s case, now a comparison).
	if err := promptByteBudgetError(20<<20, ctx, maxTok); err == nil {
		t.Errorf("20 MiB input against a 4k-token model was NOT rejected")
	}
	// Unknown context (ctx <= 0) never rejects.
	if err := promptByteBudgetError(1<<30, 0, maxTok); err != nil {
		t.Errorf("unknown-context input was rejected: %v", err)
	}
}
