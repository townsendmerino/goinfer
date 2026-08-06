package main

import (
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
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
)

// maxBytes wraps a handler so its request body is bounded to n bytes: a larger
// body fails the read with *http.MaxBytesError, which the decode helpers render
// as 413. It also caps non-JSON reads on the same body. M3.
func maxBytes(n int64, h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		r.Body = http.MaxBytesReader(w, r.Body, n)
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
		var mbe *http.MaxBytesError
		if errors.As(err, &mbe) {
			writeErr(w, http.StatusRequestEntityTooLarge, "request body too large")
			return false
		}
		writeErr(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return false
	}
	return true
}

// --- SSE ---

func sseStart(w http.ResponseWriter) (http.Flusher, bool) {
	f, ok := w.(http.Flusher)
	if !ok {
		writeErr(w, http.StatusInternalServerError, "streaming unsupported")
		return nil, false
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	return f, true
}

func sseSend(w http.ResponseWriter, f http.Flusher, v any) {
	b, _ := json.Marshal(v)
	fmt.Fprintf(w, "data: %s\n\n", b)
	f.Flush()
}

func sseDone(w http.ResponseWriter, f http.Flusher) {
	fmt.Fprint(w, "data: [DONE]\n\n")
	f.Flush()
}

// sseErr emits an OpenAI-style error object mid-stream (the response is already
// 200 with headers flushed, so a status code is no longer available). Callers
// send this in place of the normal finish chunk when a generation fails. M1.
func sseErr(w http.ResponseWriter, f http.Flusher, msg string) {
	sseSend(w, f, map[string]any{"error": map[string]any{"message": msg, "type": "api_error"}})
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
		k := len(st) - 1 // proper prefixes st[:k], 1 <= k < len(st)
		if k > len(text) {
			k = len(text)
		}
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
