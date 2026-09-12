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

	// http.Get itself can race the deadline: jsonWriteTimeout is set before WriteHeader, so on a
	// slow/loaded connection the deadline can fire before the client has even finished reading
	// the response status line and headers, and http.Get then returns an error (typically EOF)
	// instead of a *Response — measured flaky (~1/5) with a plain `if err != nil { t.Fatal }`
	// here. That failure mode is not a test-infra problem, it IS the property under test: the
	// connection got cut by the deadline either way, just before or after the client saw headers.
	// Race both outcomes against the same 15s "never returned at all" ceiling instead of treating
	// only one shape of "the deadline fired" as success.
	//
	// The response (when there is one) must stay OPEN — and its body deliberately unread — until
	// this whole test is done, not closed the moment http.Get returns: closing it early tells the
	// server the reader is gone, which unblocks writeJSON's write on ITS OWN, for a reason that
	// has nothing to do with jsonWriteTimeout (measured: this silently defeated the gate, passing
	// even with the deadline code deleted).
	type getResult struct {
		resp *http.Response
		err  error
	}
	getc := make(chan getResult, 1)
	go func() {
		resp, err := http.Get(srv.URL)
		getc <- getResult{resp, err}
	}()

	var resp *http.Response
	select {
	case r := <-getc:
		resp = r.resp
		if resp != nil {
			defer resp.Body.Close()
			// Deliberately never read the body — the whole point is a client that stops reading.
		}
		// r.err != nil (typically EOF) is ALSO a pass: the deadline cut the connection before
		// headers finished, which is the same underlying event as a body write that gets cut
		// after headers — this test does not need to distinguish the two.
	case <-time.After(15 * time.Second):
		t.Fatal("http.Get never returned at all: the connection attempt itself hung, which is not what M-13 is about")
	}

	select {
	case <-errc:
		// The handler returned within the deadline (or the body was small enough to buffer
		// entirely, in which case this still passes — the assertion that matters is the timeout).
	case <-time.After(15 * time.Second):
		t.Fatal("writeJSON never returned: a stalled client still pins the handler, which is M-13")
	}
}
