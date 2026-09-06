package serveapp

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"
)

// C-06: THE HEARTBEAT GOROUTINE AND THE HANDLER WRITE THE SAME ResponseWriter.
//
// G19 started a ticker that owns w while the handler is silent; G21 then made the incremental tool
// paths emit prose deltas to that same w during the same window. net/http's response/bufio.Writer
// is not safe for concurrent Write/Flush. The existing gate could not see it:
// heartbeat_test.go's model "never emits prose", so the overlap never happens there and -race has
// nothing to observe.
//
// This drives the overlap directly. Run with -race, which is what turns it from "probably fine" to
// a verdict — the failure mode is a torn frame or a bufio panic, and the panic would be in the
// TICKER goroutine, outside net/http's per-request recover, i.e. the process.
func TestSSEWriter_heartbeatAndHandlerDoNotRace(t *testing.T) {
	rec := httptest.NewRecorder()
	ss := newSSEWriter(rec, rec)

	old := sseHeartbeatInterval
	sseHeartbeatInterval = 50 * time.Microsecond
	defer func() { sseHeartbeatInterval = old }()

	stop := sseHeartbeat(ss)
	var wg sync.WaitGroup
	for i := range 4 { // several "handlers" is not the real shape, but it widens the window
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			for j := range 200 {
				sseSend(ss, map[string]any{"delta": fmt.Sprintf("w%d-%d", i, j)})
			}
		}(i)
	}
	wg.Wait()
	stop()

	// Every frame must be intact: a ": ping" spliced into a "data:" line is the non-panicking
	// failure, and it is the one a client silently drops.
	body := rec.Body.String()
	for line := range strings.SplitSeq(body, "\n") {
		if line == "" {
			continue
		}
		if !strings.HasPrefix(line, "data: ") && !strings.HasPrefix(line, ": ping") {
			t.Fatalf("torn frame in the stream: %q", line)
		}
		if strings.HasPrefix(line, "data: ") && strings.Contains(line, ": ping") {
			t.Fatalf("heartbeat spliced into a data frame: %q", line)
		}
	}
	if !strings.Contains(body, ": ping") {
		t.Error("no heartbeat frame at all — the test did not exercise the overlap it exists for")
	}
}

// M-17: a client that stops READING must not pin the handler forever.
//
// sseSend's Flush blocked in net.Conn.Write with no deadline, so a stalled reader held the model's
// queue slot — r.Context() cancels when the connection CLOSES, which a stalled client never does —
// and every other request queued then 429'd. The fix is a per-write deadline, which is not the
// server-wide WriteTimeout the M3 comment conflated it with.
func TestSSEWriter_stalledClientFailsTheWriteInsteadOfBlocking(t *testing.T) {
	old := sseWriteTimeout
	sseWriteTimeout = 30 * time.Millisecond
	defer func() { sseWriteTimeout = old }()

	blocked := make(chan struct{})
	ss := &sseWriter{w: blockingWriter{blocked}, f: nopFlusher{}}
	// No ResponseController: this pins the OTHER half of the contract — the error must be sticky
	// and reported — while the deadline itself is exercised end-to-end below.
	done := make(chan error, 1)
	go func() { done <- ss.frame("data: %s\n\n", "x") }()
	select {
	case err := <-done:
		if err == nil {
			t.Fatal("a failing write reported success")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("frame() never returned — the write is unbounded")
	}
	close(blocked)

	if ss.Err() == nil {
		t.Error("the write error is not sticky; the handler cannot tell the client is gone")
	}
	// Once the error is sticky, later frames must be cheap no-ops rather than more blocked writes.
	if err := ss.frame("data: %s\n\n", "y"); err == nil {
		t.Error("a frame after a failed write reported success")
	}
}

// The deadline is actually installed on a real connection. httptest.NewRecorder has no
// ResponseController support, so the unit test above cannot cover this half.
func TestSSEWriter_setsAWriteDeadlineOnARealConnection(t *testing.T) {
	old := sseWriteTimeout
	sseWriteTimeout = 40 * time.Millisecond
	defer func() { sseWriteTimeout = old }()

	errc := make(chan error, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ss, ok := sseStart(w)
		if !ok {
			errc <- fmt.Errorf("sseStart failed")
			return
		}
		// Write until the socket buffer fills against a client that never reads. Without a
		// deadline this loop never returns and the handler is pinned.
		big := strings.Repeat("x", 64<<10)
		var err error
		for i := 0; i < 2000 && err == nil; i++ {
			err = ss.frame("data: %s\n\n", big)
		}
		errc <- err
	}))
	defer srv.Close()

	resp, err := http.Get(srv.URL)
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	defer resp.Body.Close()
	// Deliberately never read the body.
	select {
	case err := <-errc:
		if err == nil {
			t.Skip("the socket buffer swallowed every frame on this machine — the deadline was " +
				"never reached, so this run proves nothing about it")
		}
	case <-time.After(15 * time.Second):
		t.Fatal("the handler never returned: a stalled reader still pins it, which is M-17")
	}
}

type blockingWriter struct{ release <-chan struct{} }

func (b blockingWriter) Header() http.Header { return http.Header{} }
func (b blockingWriter) WriteHeader(int)     {}
func (b blockingWriter) Write([]byte) (int, error) {
	return 0, fmt.Errorf("write: connection timed out")
}

type nopFlusher struct{}

func (nopFlusher) Flush() {}

// M-19: EVERY BUFFER-THEN-STREAM SITE MUST START A HEARTBEAT.
//
// G19 fixed two of three. The third — streamMessagesTools, the Anthropic tool path — emitted
// nothing after message_start's single `ping` until the whole generation finished, on the surface
// docs/server.md markets for Claude Code, where tool-bearing requests are the norm. The existing
// gates (heartbeat_test.go) cannot catch a missing site: they need GOINFER_SERVE_MODEL and skip
// without it, and they test the sites that already had one.
//
// A buffer-then-stream site is definable, so it is checked rather than remembered: a handler that
// drives a generation whose callback only appends to a builder — writing nothing to the client —
// must start a heartbeat, because it is silent for the whole generation by construction.
func TestSSE_everyBufferedStreamSiteHeartbeats(t *testing.T) {
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	sort.Strings(files)
	// func ... { ... } at column 0 — one top-level function per match.
	fnRe := regexp.MustCompile(`(?ms)^func (?:\([^)]*\) )?(\w+)\([^\n]*\{\n(.*?)\n\}`)
	checked, buffered := 0, []string{}
	for _, f := range files {
		if strings.HasSuffix(f, "_test.go") {
			continue
		}
		b, err := os.ReadFile(f)
		if err != nil {
			continue
		}
		for _, m := range fnRe.FindAllStringSubmatch(string(b), -1) {
			name, body := m[1], m[2]
			if !strings.Contains(body, "lm.drive(") {
				continue
			}
			checked++
			// It must be ON a streaming path: it holds an *sseWriter, either as a parameter or by
			// opening the stream itself. Without this, serveMessagesWith matches — it dispatches to
			// streamMessages AND has its own non-streaming drive whose callback buffers, and a
			// heartbeat there would be writing keep-alives to a client that is not streaming.
			decl := m[0][:strings.Index(m[0], "{")]
			onStream := strings.Contains(decl, "ss *sseWriter") ||
				strings.Contains(body, "sseStart(w)") || strings.Contains(body, "anthropicSSEStart(w)")
			if !onStream {
				continue
			}
			// THE CALLBACK, NOT THE BODY. streamMessagesTools writes plenty — tool_use blocks,
			// message_end — but all of it AFTER drive returns. What makes a site silent for the
			// whole generation is that its drive CALLBACK sends nothing, and an earlier cut of
			// this check looked at the whole function and so excluded the one site it exists for.
			cb := driveCallback(body)
			if cb == "" {
				continue // not the literal-callback shape this classifier understands
			}
			if unconditionalStreamCall(cb) {
				continue // streams as it generates: the client hears from it either way
			}
			buffered = append(buffered, f+":"+name)
			if !strings.Contains(body, "sseHeartbeat(") {
				t.Errorf("%s in %s buffers the whole generation (its drive callback only appends to "+
					"a builder) and starts NO heartbeat — it sends zero bytes for the entire "+
					"generation. That is audit-2026-09-02 M-19, which was G19 reaching two of "+
					"three sites; this is the check that counts them.", name, f)
			}
		}
	}
	if checked == 0 {
		t.Fatal("no lm.drive call site found — the scan is broken, and a broken scan makes this " +
			"gate pass over nothing")
	}
	if len(buffered) == 0 {
		t.Fatalf("scanned %d drive site(s) and classified NONE as buffer-then-stream; the "+
			"classifier no longer matches the code it is about", checked)
	}
	t.Logf("%d drive site(s), %d buffer-then-stream: %s", checked, len(buffered), strings.Join(buffered, " "))
}

// driveCallback returns the text of the `func(t string) { … }` argument to an lm.drive call, or ""
// when the call does not use that literal shape.
func driveCallback(body string) string {
	i := strings.Index(body, "lm.drive(")
	if i < 0 {
		return ""
	}
	j := strings.Index(body[i:], "func(t string) {")
	if j < 0 {
		return ""
	}
	start := i + j
	depth := 0
	for k := start; k < len(body); k++ {
		switch body[k] {
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return body[start : k+1]
			}
		}
	}
	return ""
}

// unconditionalStreamCall reports whether cb — a driveCallback result — calls one of the
// per-token streaming sends AT THE CALLBACK'S OWN TOP LEVEL, i.e. on every invocation, not
// merely somewhere inside it behind a conditional.
//
// V-21 (docs/review-2026-09-04.md): a plain strings.Contains(cb, "sseSend(") used to exempt a
// call site the moment the literal substring appeared anywhere in the callback — including
// inside an `if` that is frequently false. serveChatToolsWith's callback is exactly that shape:
// sb.WriteString(t); if prose == nil { return }; if out := prose.Push(t); out != "" {
// sseSend(...) } — sseSend is reachable only for incremental families, and even then only once
// prose.Push has enough content to flush, so most tokens (and the whole non-incremental family
// case) hit NEITHER branch. The old check still saw the substring and skipped the site entirely,
// so it never even looked for sseHeartbeat(...) in the enclosing function — dropping that
// heartbeat would have failed nothing. This walks brace depth relative to the callback's own `{`
// (depth 1 = the callback's own top level) and only counts a marker call found there, exactly as
// driveCallback already walks depth to find the callback's closing brace.
func unconditionalStreamCall(cb string) bool {
	markers := []string{"sseSend(", "sseEvent(", "anthropicEvent(", "ss.frame("}
	depth := 0
	for i := 0; i < len(cb); i++ {
		switch cb[i] {
		case '{':
			depth++
		case '}':
			depth--
		}
		if depth != 1 {
			continue
		}
		for _, m := range markers {
			if strings.HasPrefix(cb[i:], m) {
				return true
			}
		}
	}
	return false
}
