package serveapp

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// M-13 (docs/audit-2026-09-10.md, M-17 (09-02)'s sibling): writeJSON must not block forever
// against a client that stops reading. sseWriter.frame already gates the streaming half
// (sse_writer_test.go's TestSSEWriter_setsAWriteDeadlineOnARealConnection); this is the same gate
// for the buffered, non-streaming responses every OpenAI/Anthropic/responses route uses.
//
// httptest.NewRecorder has no ResponseController deadline support, so — same as the SSE test —
// this needs a REAL connection: a server whose handler writes a large body through writeJSON,
// and a client that connects but never reads it, filling the kernel socket buffer.
func TestWriteJSON_stalledClientFailsTheWriteInsteadOfBlocking(t *testing.T) {
	old := jsonWriteTimeout
	jsonWriteTimeout = 40 * time.Millisecond
	defer func() { jsonWriteTimeout = old }()

	errc := make(chan struct{}, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Large enough to exceed any OS socket send buffer regardless of local tuning — the
		// audit's own trigger is "logprobs:true, top_logprobs:20, max_tokens:8000 -> a ~10 MB
		// body"; this goes well past that so the write is guaranteed to block against an unread
		// connection, not merely buffer entirely in the kernel and return early.
		big := strings.Repeat("x", 32<<20) // 32 MB
		writeJSON(w, http.StatusOK, map[string]any{"text": big})
		close(errc)
	}))
	defer srv.Close()

	resp, err := http.Get(srv.URL)
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	defer resp.Body.Close()
	// Deliberately never read the body — the whole point is a client that stops reading.

	select {
	case <-errc:
		// Returned within the deadline (or the body was small enough to buffer entirely, in
		// which case this still passes — the assertion that matters is the timeout below).
	case <-time.After(15 * time.Second):
		t.Fatal("writeJSON never returned: a stalled client still pins the handler, which is M-13")
	}
}
