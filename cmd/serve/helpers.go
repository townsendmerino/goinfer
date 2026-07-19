package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync/atomic"
	"time"
	"unicode/utf8"
)

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
	writeJSON(w, code, map[string]any{"error": map[string]any{"message": msg, "type": "invalid_request_error"}})
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

// firstString reads a "prompt" that is a string (or the first of an array).
func firstString(raw json.RawMessage) string {
	var one string
	if json.Unmarshal(raw, &one) == nil {
		return one
	}
	var many []string
	if json.Unmarshal(raw, &many) == nil && len(many) > 0 {
		return many[0]
	}
	return ""
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
