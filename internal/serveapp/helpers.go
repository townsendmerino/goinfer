package serveapp

import (
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"
)

// Request-body size ceilings (M3). Bodies are buffered before validation, so an
// unbounded body is a memory-exhaustion DoS. Text prompts/tools/embeddings/admin
// fit comfortably in a few MB; the vision endpoints carry base64 image_url data
// (≈1.33× the raw image), so they get a larger cap.
const (
	maxBodyBytes       = 4 << 20  // 4 MiB
	maxVisionBodyBytes = 32 << 20 // 32 MiB
	// maxEmbedBodyBytes is /v1/embeddings' own floor, deliberately NOT derived from any decoder's
	// context window. A batch embeddings body scales with (batch count × input length) — quantities
	// the route already bounds itself (maxEmbedInputs=2048, maxEmbedInputBytes=1 MiB) — and has
	// nothing to do with a chat model's MaxPositions. Deriving it from the text cap made the limit
	// arbitrary in both directions: on an embed-only server (no decoder loaded) it collapsed to the
	// 4 MiB text floor, rejecting a perfectly legal 2048×4 KiB batch at 8 MiB before
	// checkEmbedInputBounds could accept it; alongside a 128k-context chat model it ballooned for no
	// reason. 64 MiB covers a realistic maximal RAG batch (2048 inputs × ~32 KiB) while still
	// bounding the read. -max-body-bytes overrides it like the others.
	maxEmbedBodyBytes = 64 << 20 // 64 MiB
)

// maxBytes wraps a handler so its request body is bounded to n bytes (n <= 0 disables).
// Two layers (G1/G2/G3):
//   - Content-Length pre-check: a declared body over the cap is rejected 413 BEFORE a byte
//     is read — the case that matters, and it costs nothing (no allocation, no upload wait).
//   - http.MaxBytesReader backstop for chunked encoding or a lying Content-Length; it fails
//     the read with *http.MaxBytesError, which the decode helpers render as 413.
//
// The 413 names the limit (and the received size when the client declared one) so a client
// sees why it was rejected rather than a bare close. A client still uploading when the
// pre-check fires may see EPIPE regardless — but today it gets no HTTP response at all (G3).
// note, when non-empty, is appended to the 413. A route whose own validator declares limits in
// DIFFERENT units than the body cap needs it: /v1/embeddings advertises "up to 2048 inputs of up to
// 1 MiB each", which multiplies out to 2 GiB and can therefore never all be satisfied at once. Those
// are per-DIMENSION bounds; the body cap bounds the TOTAL. A request can respect both per-dimension
// limits and still exceed the total, and a 413 naming only the total leaves the client unable to
// tell which of the three numbers it actually violated.
func maxBytes(n int64, h http.HandlerFunc, note ...string) http.HandlerFunc {
	extra := ""
	if len(note) > 0 && note[0] != "" {
		extra = " " + note[0]
	}
	return func(w http.ResponseWriter, r *http.Request) {
		if n > 0 && r.ContentLength > n {
			writeErr(w, http.StatusRequestEntityTooLarge,
				fmt.Sprintf("request body is %d bytes, which exceeds the %d-byte limit (raise it with -max-body-bytes)%s", r.ContentLength, n, extra))
			return
		}
		if n > 0 {
			r.Body = http.MaxBytesReader(w, r.Body, n)
		}
		h(w, r)
	}
}

// limitInflight caps how many wrapped handlers run concurrently across the whole server, via a
// shared semaphore. It bounds the PRE-QUEUE stage — JSON + base64-image decode, tokenization,
// template render, the vision-tower Forward, constrain.TokenBytes over the full vocab — which
// runs before a request reaches the per-model decode queue (lm.enter). Without it that stage has
// unbounded concurrency: 200 parallel 32 MiB vision requests each allocate before any backpressure
// applies (audit M-01). A full cap returns 503 + Retry-After (an orchestrator/back-off signal),
// distinct from the per-model 429. sem == nil disables it (-max-inflight 0). The slot is held for
// the whole handler including generation — a hard ceiling on total concurrent requests — which
// composes with the finer per-model queue.
func limitInflight(sem chan struct{}, h http.HandlerFunc) http.HandlerFunc {
	if sem == nil {
		return h
	}
	return func(w http.ResponseWriter, r *http.Request) {
		select {
		case sem <- struct{}{}:
			defer func() { <-sem }()
			h(w, r)
		default:
			w.Header().Set("Retry-After", "1")
			writeErr(w, http.StatusServiceUnavailable, "server at capacity (max in-flight requests reached); retry")
		}
	}
}

// requireAuth wraps a handler with an optional shared-secret check (audit B-14).
// When key == "" it is a pass-through (auth disabled — the historical behaviour,
// safe only because -addr now defaults to loopback). When key is set, the request
// must present it as `Authorization: Bearer <key>` or `x-api-key: <key>`; a
// constant-time compare avoids leaking the key length/prefix via timing. This is a
// coarse gate (one shared secret, not per-user) — enough to keep an exposed port
// from being open inference, and required whenever -allow-admin is on.
func requireAuth(key string, h http.HandlerFunc) http.HandlerFunc {
	if key == "" {
		return h
	}
	want := []byte(key)
	return func(w http.ResponseWriter, r *http.Request) {
		got := r.Header.Get("x-api-key")
		if got == "" {
			got = strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
		}
		if subtle.ConstantTimeCompare([]byte(got), want) != 1 {
			writeErr(w, http.StatusUnauthorized, "missing or invalid API key")
			return
		}
		h(w, r)
	}
}

// decodeJSON reads the (size-bounded, see maxBytes) request body into v, writing
// an OpenAI-shaped error on failure: 413 when the body exceeded the limit, else
// 400. Returns false iff it wrote an error. M3.
func decodeJSON(w http.ResponseWriter, r *http.Request, v any) bool {
	if err := json.NewDecoder(r.Body).Decode(v); err != nil {
		if mbe, ok := errors.AsType[*http.MaxBytesError](err); ok {
			writeErr(w, http.StatusRequestEntityTooLarge,
				fmt.Sprintf("request body exceeds the %d-byte limit (raise it with -max-body-bytes)", mbe.Limit))
			return false
		}
		// Don't echo the raw json error: UnmarshalTypeError's default string leaks the Go struct name
		// (e.g. "…Go struct field completionReq.logprobs of type bool"), which M-06 exists to prevent.
		// Report the JSON-side field + expected type instead (audit R-11).
		if ute, ok := errors.AsType[*json.UnmarshalTypeError](err); ok {
			// ute.Type is a reflect.Type; for a composite field it renders the internal named type
			// (e.g. "[]serveapp.chatMessage"), re-leaking the Go type names M-06/R-11 exist to hide.
			// Name the expected type only for scalar kinds (bool/number/string); else stay generic (F-03).
			switch ute.Type.Kind() {
			case reflect.Bool, reflect.String,
				reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
				reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64,
				reflect.Float32, reflect.Float64:
				writeErr(w, http.StatusBadRequest, fmt.Sprintf("invalid request body: field %q has the wrong type (expected %s)", ute.Field, ute.Type.Kind()))
			default:
				writeErr(w, http.StatusBadRequest, fmt.Sprintf("invalid request body: field %q has the wrong type", ute.Field))
			}
			return false
		}
		if se, ok := errors.AsType[*json.SyntaxError](err); ok {
			writeErr(w, http.StatusBadRequest, fmt.Sprintf("invalid request body: malformed JSON at byte %d", se.Offset))
			return false
		}
		writeErr(w, http.StatusBadRequest, "invalid request body")
		return false
	}
	return true
}

// --- SSE ---

// sseWriter owns one SSE response: every frame goes through it, one at a time, under a bounded
// write deadline. Both the OpenAI and the Anthropic stream paths write through it.
//
// IT EXISTS BECAUSE TWO GOROUTINES WERE WRITING THE SAME ResponseWriter. G19 added a heartbeat
// goroutine that owns w while the handler is silent; G21 then made the incremental tool paths write
// prose deltas to that same w during the same window, deliberately. net/http's response/bufio.Writer
// is not safe for concurrent Write/Flush, so outcomes ran from ": ping" spliced into a data: line
// (the client drops or mis-parses the chunk) to a bufio slice-bounds panic — and a panic in the
// HEARTBEAT goroutine is outside net/http's per-request recover, so it takes the PROCESS
// (audit-2026-09-02 C-06). sseHeartbeat's own doc promised "no risk of interleaving two writers";
// that was true only on the paths whose callback appends to a builder.
//
// The deadline is a second defect on the same write path. Flush blocks in net.Conn.Write with no
// deadline, so a client that stops READING without closing pins the handler inside onText holding
// the model's queue slot — r.Context() cancels on close, not on a stalled read — and every other
// request queues then 429s for as long as that socket stays open (M-17). This is NOT the
// server-wide WriteTimeout the M3 comment conflated it with: that would truncate a legitimately
// long stream; this bounds one frame.
//
// The first write error is sticky and readable: a stream whose client vanished should stop
// generating rather than keep pushing frames into a dead socket.
type sseWriter struct {
	mu  sync.Mutex
	w   http.ResponseWriter
	f   http.Flusher
	rc  *http.ResponseController
	err error
}

// sseWriteTimeout bounds one frame's write+flush. A var so tests can drive it. Generous against a
// slow but live client, short enough that a stalled one frees the decode worker.
var sseWriteTimeout = 30 * time.Second

// frame writes one preformatted SSE frame under the lock, and is the ONLY place this package
// writes to a streaming ResponseWriter.
func (s *sseWriter) frame(format string, args ...any) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.err != nil {
		return s.err // sticky: the client is gone, stop trying
	}
	if s.rc != nil {
		// An unsupported controller is not a stream failure — httptest's recorder has no deadline
		// support — so the error is deliberately dropped rather than made sticky.
		_ = s.rc.SetWriteDeadline(time.Now().Add(sseWriteTimeout))
	}
	if _, err := fmt.Fprintf(s.w, format, args...); err != nil {
		s.err = err
		return err
	}
	s.f.Flush()
	return nil
}

// Err reports the first write failure. Non-nil means the client is gone.
func (s *sseWriter) Err() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.err
}

// newSSEWriter wraps a streaming response. Shared by both protocols' start helpers.
func newSSEWriter(w http.ResponseWriter, f http.Flusher) *sseWriter {
	return &sseWriter{w: w, f: f, rc: http.NewResponseController(w)}
}

func sseStart(w http.ResponseWriter) (*sseWriter, bool) {
	f, ok := w.(http.Flusher)
	if !ok {
		writeErr(w, http.StatusInternalServerError, "streaming unsupported")
		return nil, false
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	return newSSEWriter(w, f), true
}

func sseSend(ss *sseWriter, v any) {
	b, _ := json.Marshal(v)
	ss.frame("data: %s\n\n", b)
}

func sseDone(ss *sseWriter) {
	ss.frame("data: [DONE]\n\n")
}

// sseErr emits an OpenAI-style error object mid-stream (the response is already
// 200 with headers flushed, so a status code is no longer available). Callers
// send this in place of the normal finish chunk when a generation fails. M1.
func sseErr(ss *sseWriter, msg string) {
	sseSend(ss, map[string]any{"error": map[string]any{"message": msg, "type": "api_error"}})
}

// sseHeartbeatInterval is how often a buffered generation emits a keep-alive
// comment frame. A package var, not a const, so tests can drive the mechanism
// without waiting on wall-clock. 10s is far inside every harness idle timeout we
// know of (dsh's default is 300s) and costs 8 bytes per tick.
var sseHeartbeatInterval = 10 * time.Second

// sseHeartbeat keeps a stream alive while the handler is producing nothing to
// send (G19). The tool paths must buffer the whole generation before they can
// parse a tool call, so they would otherwise send zero bytes for the entire
// generation — measured at 1682.6s against a client whose idle timeout was 300s.
//
// A COMMENT frame (":" + text) is the right instrument: it is protocol-legal,
// carries no data, and every SSE parser drops it, so nothing downstream can
// mistake a keep-alive for content. The buffering guarantee is untouched.
//
// The returned stop JOINS the goroutine before returning. That join was once the ONLY thing keeping
// two writers apart, and it was not enough: it orders the heartbeat against the handler's writes
// AFTER drive returns, not against the prose deltas the incremental paths emit WHILE it runs. The
// serialization now comes from sseWriter's lock; the join remains so nothing ticks after the
// caller's final frame (audit-2026-09-02 C-06).
func sseHeartbeat(ss *sseWriter) (stop func()) {
	done, finished := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(finished)
		t := time.NewTicker(sseHeartbeatInterval)
		defer t.Stop()
		for {
			select {
			case <-done:
				return
			case <-t.C:
				ss.frame(": ping\n\n")
			}
		}
	}()
	return func() {
		close(done)
		<-finished
	}
}

type delta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

func chatChunk(id string, created int64, model string, d delta, finish *string) map[string]any {
	return map[string]any{
		"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
		"choices": []any{map[string]any{"index": 0, "delta": d, "finish_reason": finish}},
	}
}

// usageChunk is the final streaming chunk when stream_options.include_usage is set: empty choices,
// a usage object. Matches OpenAI, so a client that already understands their stream needs no
// goinfer-specific handling.
func usageChunk(id string, created int64, model string, u usage) map[string]any {
	return map[string]any{
		"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
		"choices": []any{}, "usage": u,
	}
}

func completionChunk(id string, created int64, model, text string, finish *string) map[string]any {
	return map[string]any{
		"id": id, "object": "text_completion", "created": created, "model": model,
		"choices": []any{map[string]any{"index": 0, "text": text, "finish_reason": finish}},
	}
}

// --- JSON responses ---

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func writeErr(w http.ResponseWriter, code int, msg string) {
	// Type mirrors the status so typed OpenAI-SDK client retry logic keys off it
	// (429 → back off, 5xx → retry, 4xx → don't).
	typ := "invalid_request_error"
	switch {
	case code == http.StatusTooManyRequests:
		typ = "rate_limit_error"
	case code >= 500:
		typ = "api_error"
	}
	writeJSON(w, code, map[string]any{"error": map[string]any{"message": msg, "type": typ}})
}

// writeServerErr reports a generation/encode failure as a 500 with type
// "api_error" (distinct from the 4xx "invalid_request_error" of a bad request);
// used before any body is written, on the non-streaming paths. M1.
func writeServerErr(w http.ResponseWriter, msg string) {
	writeJSON(w, http.StatusInternalServerError, map[string]any{"error": map[string]any{"message": msg, "type": "api_error"}})
}

// --- request helpers ---

func deref[T any](p *T, def T) T {
	if p != nil {
		return *p
	}
	return def
}

// parseStop reads OpenAI's "stop" (a string or array of strings).
func parseStop(raw json.RawMessage) []string {
	if len(raw) == 0 {
		return nil
	}
	var one string
	if json.Unmarshal(raw, &one) == nil {
		if one == "" {
			return nil
		}
		return []string{one}
	}
	var many []string
	_ = json.Unmarshal(raw, &many)
	return many
}

// singlePromptString accepts the /v1/completions prompt shapes goinfer supports: a
// string, or a single-element []string. A multi-element batch, a []int / [][]int
// token-id prompt, or anything else is a 400 rather than silently decoding to "" (a
// BOS-only generation returned as an unrelated 200).
func singlePromptString(raw json.RawMessage) (string, error) {
	if len(raw) == 0 {
		return "", nil
	}
	var one string
	if json.Unmarshal(raw, &one) == nil {
		return one, nil
	}
	var many []string
	if json.Unmarshal(raw, &many) == nil {
		if len(many) == 1 {
			return many[0], nil
		}
		return "", fmt.Errorf("prompt: batch prompts are not supported (%d entries); send a single string", len(many))
	}
	return "", fmt.Errorf("prompt: unsupported shape — send a string; token-id arrays and batches are not supported")
}

// firstStop returns the byte index of the earliest stop string in text (and the
// string itself, and true), so the caller can truncate there — OpenAI omits the
// stop sequence, and the Anthropic endpoint reports which one was hit.
func firstStop(text string, stops []string) (int, string, bool) {
	cut, which := -1, ""
	for _, st := range stops {
		if st == "" {
			continue
		}
		if i := strings.Index(text, st); i >= 0 && (cut < 0 || i < cut) {
			cut, which = i, st
		}
	}
	if cut < 0 {
		return 0, "", false
	}
	return cut, which, true
}

// stopTailHold returns the length of the longest suffix of text that is a proper
// (non-empty, shorter-than-whole) prefix of some stop string — bytes the streamer
// must hold back, because the next token could extend them into a full stop
// sequence, which must be removed *entirely* (prefix included). Without this, a
// stop of "END" arriving as "E"+"ND" leaks the "E" before the stop is recognized.
// A full match is a real stop, handled by firstStop, not here. M2.
func stopTailHold(text string, stops []string) int {
	hold := 0
	for _, st := range stops {
		k := min(
			// proper prefixes st[:k], 1 <= k < len(st)
			len(st)-1, len(text))
		for ; k > hold; k-- {
			if strings.HasSuffix(text, st[:k]) {
				hold = k
				break
			}
		}
	}
	return hold
}

// completeUTF8 returns the length of the longest prefix of s ending on a rune
// boundary, holding back a trailing incomplete multi-byte sequence (a
// byte-fallback token can split a rune).
func completeUTF8(s string) int {
	end := 0
	for i := 0; i < len(s); {
		if !utf8.FullRuneInString(s[i:]) {
			break // incomplete trailing sequence — hold back
		}
		_, size := utf8.DecodeRuneInString(s[i:])
		i += size
		end = i
	}
	return end
}

// reqID is a short, monotonically increasing id suffix (unique per process run).
var reqCounter atomic.Uint64

func init() { reqCounter.Store(uint64(time.Now().UnixNano())) }

func reqID() string { return fmt.Sprintf("%x", reqCounter.Add(1)) }

// humanBytes formats a byte count as MiB/GiB for the startup line (whole MiB is exact here —
// every cap is a multiple of a MiB or a small model-derived product).
func humanBytes(n int64) string {
	switch {
	case n >= 1<<30:
		return fmt.Sprintf("%.1f GiB", float64(n)/(1<<30))
	case n >= 1<<20:
		return fmt.Sprintf("%.1f MiB", float64(n)/(1<<20))
	case n >= 1<<10:
		return fmt.Sprintf("%.1f KiB", float64(n)/(1<<10))
	default:
		return fmt.Sprintf("%d B", n)
	}
}
