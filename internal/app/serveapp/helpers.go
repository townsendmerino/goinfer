package serveapp

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"unicode/utf8"

	"github.com/townsendmerino/goinfer/internal/app/serveapp/jobs"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/netguard"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/stores"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/streamframe"
)

const (
	maxBodyBytes        = netguard.MaxBodyBytes
	maxVisionBodyBytes  = netguard.MaxVisionBodyBytes
	maxEmbedBodyBytes   = netguard.MaxEmbedBodyBytes
	maxBatchFileBytes   = netguard.MaxBatchFileBytes
	maxFileUploadMemory = netguard.MaxFileUploadMemory
)

var (
	maxBytes          = netguard.MaxBytes
	limitInflight     = netguard.LimitInflight
	requireAuth       = netguard.RequireAuth
	writeBodyErr      = netguard.WriteBodyErr
	parseUpload       = netguard.ParseUpload
	jsonDecodeMessage = netguard.JSONDecodeMessage
	decodeJSON        = netguard.DecodeJSON
	serveHandler      = netguard.ServeHandler
	sameOriginGuard   = netguard.SameOriginGuard
	splitHosts        = netguard.SplitHosts
)

// --- SSE ---

type sseWriter = streamframe.SSEWriter

var (
	newSSEWriter  = streamframe.NewSSEWriter
	sseJSONSender = streamframe.JSONSender
	sseSend       = streamframe.SendData
	sseDone       = streamframe.SendDone
	sseErr        = streamframe.SendOpenAIError
	sseHeartbeat  = streamframe.Heartbeat
)

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

type delta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
	// ReasoningContent is the model's think-block text (think.go), streamed before the answer's content. llama.cpp and
	// the DeepSeek API spell it reasoning_content; omitted when the model did not reason.
	ReasoningContent string `json:"reasoning_content,omitempty"`
}

func chatChunk(id string, created int64, model string, d delta, finish *string) map[string]any {
	return map[string]any{
		"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
		"choices": []any{map[string]any{"index": 0, "delta": d, "finish_reason": finish}},
	}
}

// streamOptions is OpenAI's stream_options. It is named, not anonymous, so every streaming surface can accept it and
// hand it to sendUsage.
type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

// sendUsage emits the include_usage chunk, if the client asked for one, immediately before sseDone. This is the one
// place that decision is made: routing every streaming surface (chat, tools, vision, completions) through one helper
// is what lets a test count the sites, so a new streaming surface cannot be added quietly without one. Agent
// harnesses, which declare tools on every turn, depend on the usage chunk for context accounting. The chunk carries the
// turn's timings and, when asked, its cost beside the usage (clock, nil for none).
func sendUsage(ss *sseWriter, clock *jobs.TurnClock, so *streamOptions, id string, created int64, model string, u usage) {
	if so == nil || !so.IncludeUsage {
		return
	}
	sseSend(ss, clock.AddTurn(usageChunk(id, created, model, u)))
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

const statusCancelled = netguard.StatusCancelled

var (
	writeJSON      = netguard.WriteJSON
	writeErr       = netguard.WriteErr
	writeServerErr = netguard.WriteServerErr
)

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

// stopTailHold returns the length of the longest suffix of text that is a proper (non-empty, shorter-than-whole)
// prefix of some stop string: bytes the streamer must hold back, because the next token could extend them into a full
// stop sequence, which must be removed entirely (prefix included). Without this, a stop of "END" arriving as "E"+"ND"
// leaks the "E" before the stop is recognized. A full match is a real stop, handled by firstStop, not here.
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

// reqID returns an opaque, unguessable id for a response, message or tool call. It is not a counter: sequential ids
// (even seeded from UnixNano) let a client holding its own `resp_<hex>` walk +-1 onto other clients' ids, and
// `previous_response_id` continues a stored conversation, so with `-addr 0.0.0.0` and one shared API key a guessed id
// reads back someone else's turns. crypto/rand, not math/rand, because guessability is the whole property; 16 bytes
// because these ids go in URLs and logs, and 128 bits is beyond enumeration.
func reqID() string {
	return stores.MintID()
}

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
