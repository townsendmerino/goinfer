// Command gemma-web serves a local, single-page web chat GUI for a decoder LLM
// checkpoint running on aikit's pure-Go decoder. Pure standard library only
// (net/http + Server-Sent Events); no external assets, no cgo.
//
// It runs any family the decoder + tokenizer support — Gemma 3/4, Qwen, Llama-3,
// Mistral — picking the chat template (chat.Detect) and stop tokens from the
// loaded checkpoint, so the same binary chats with any of them.
//
// The model + tokenizer are loaded ONCE at startup and reused across requests;
// the naive CPU backend is single-stream, so generations are serialized (a
// second in-flight request gets 409 Busy).
//
// Usage:
//
//	go run ./demo/gemma-web --model ~/models/gemma-3-270m
//	go run ./demo/gemma-web --model ~/models/qwen3-1.7b
//	# then open the printed http://127.0.0.1:8080
package main

import (
	"context"
	_ "embed"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

//go:embed index.html
var indexHTML []byte

const (
	defaultMaxTokens = 256
	maxMaxTokens     = 2048
)

// chatMessage is one turn the client sends each request (the model is stateless
// per call, so the full history comes every time).
type chatMessage struct {
	Role    string `json:"role"` // "user" | "model"
	Content string `json:"content"`
}

// chatRequest is the POST /chat body.
type chatRequest struct {
	Messages    []chatMessage `json:"messages"`
	System      string        `json:"system"`
	Temperature float64       `json:"temperature"`
	TopK        int           `json:"topK"`
	TopP        float64       `json:"topP"`
	Seed        int64         `json:"seed"`
	MaxTokens   int           `json:"maxTokens"`
}

// server holds the shared, immutable model + tokenizer and a mutex that admits
// one generation at a time (the CPU backend is single-stream).
type server struct {
	tk      *tokenizer.Tokenizer
	model   *decoder.Model
	special tokenizer.SpecialTokens
	tmpl    *chat.Template // nil → raw-completion fallback
	stopIDs []int          // turn-stop ids resolved from tmpl.Stops()
	mu      sync.Mutex     // held for the duration of a generation
}

func main() {
	var (
		addr     = flag.String("addr", "127.0.0.1:8080", "listen address")
		modelDir = flag.String("model", "", "path to a checkpoint dir (config.json + model.safetensors + tokenizer); Gemma 3 or Qwen/Llama-3")
		backend  = flag.String("backend", "cpu", "compute backend: cpu | webgpu")
		quant    = flag.String("quant", "", "weight quantization: \"\" (f32) | int8 | int8int8 | int4")
	)
	flag.Parse()
	if *modelDir == "" {
		fmt.Fprintln(log.Writer(), "error: --model is required (path to a checkpoint dir)")
		flag.Usage()
		return
	}

	s, err := newServer(*modelDir, *backend, *quant)
	if err != nil {
		log.Fatal(err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/", s.handleIndex)
	mux.HandleFunc("/chat", s.handleChat)

	fmt.Printf("gemma-web listening on http://%s\n", *addr)
	if err := http.ListenAndServe(*addr, mux); err != nil {
		log.Fatal(err)
	}
}

// newServer loads the tokenizer + model once.
func newServer(modelDir, backend, quant string) (*server, error) {
	tk, err := tokenizer.Load(modelDir)
	if err != nil {
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}
	t0 := time.Now()
	model, err := decoder.Load(modelDir, decoder.Options{Backend: backend, Quant: quant})
	if err != nil {
		return nil, fmt.Errorf("load model: %w", err)
	}
	cfg := model.Config()
	q := quant
	if q == "" {
		q = "f32"
	}
	fmt.Printf("loaded %d-layer model (hidden %d, vocab %d) in %s [backend=%s quant=%s]\n",
		cfg.NumLayers, cfg.HiddenDim, cfg.VocabSize, time.Since(t0).Round(time.Millisecond), model.BackendReport(), q)
	srv := &server{tk: tk, model: model, special: tk.Special()}
	if tmpl, derr := chat.Detect(chat.Meta{ChatTemplate: tk.ChatTemplate(), HasToken: tk.Has}); derr == nil {
		srv.tmpl = tmpl
		for _, str := range tmpl.Stops().Strings {
			if id, ok := tk.TokenID(str); ok {
				srv.stopIDs = append(srv.stopIDs, id)
			}
		}
		log.Printf("chat template: %s", tmpl.Name())
	} else {
		log.Printf("note: no recognized chat template; replies are raw completions")
	}
	return srv, nil
}

func (s *server) handleIndex(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write(indexHTML)
}

func (s *server) handleChat(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	var req chatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad json: "+err.Error(), http.StatusBadRequest)
		return
	}
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}
	// One generation at a time (single-stream CPU backend). Don't block: a
	// second request gets 409 so the UI can surface "busy" rather than hang.
	if !s.mu.TryLock() {
		http.Error(w, "busy: a generation is already in flight", http.StatusConflict)
		return
	}
	defer s.mu.Unlock()

	// Rendered templates include the family's BOS marker; only the raw fallback
	// needs Encode to prepend one.
	promptIDs, err := s.tk.Encode(s.buildPrompt(req), s.tmpl == nil /* addBOS */)
	if err != nil {
		http.Error(w, "encode: "+err.Error(), http.StatusInternalServerError)
		return
	}
	maxTok := req.MaxTokens
	if maxTok <= 0 {
		maxTok = defaultMaxTokens
	}
	if maxTok > maxMaxTokens {
		maxTok = maxMaxTokens
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	s.stream(r.Context(), w, flusher, promptIDs, maxTok, decoder.SamplingParams{
		Temperature: req.Temperature,
		TopK:        req.TopK,
		TopP:        req.TopP,
		Seed:        req.Seed,
		StopIDs:     s.stopIDs,
	})
}

// stream runs one generation and emits SSE token/done(/error) events. It
// reuses demo/gemma's holdback: decode the whole running id slice each step and
// send only the newly-COMPLETED bytes, so a partial byte-fallback token never
// emits broken UTF-8.
func (s *server) stream(ctx context.Context, w http.ResponseWriter, f http.Flusher, promptIDs []int, maxTok int, sp decoder.SamplingParams) {
	stream, gen := s.model.Generate(ctx, promptIDs, maxTok, sp)

	var out []int
	printed := 0
	start := time.Now()
	var ttft time.Duration
	emit := func(final bool) {
		text, derr := s.tk.Decode(out)
		if derr != nil {
			return
		}
		b := []byte(text)
		end := len(b)
		if !final {
			end = completeUTF8Len(b)
		}
		if end > printed {
			sendEvent(w, f, "token", map[string]any{"text": string(b[printed:end]), "n": len(out)})
			printed = end
		}
	}

	for id := range stream {
		if len(out) == 0 {
			ttft = time.Since(start)
		}
		out = append(out, id)
		emit(false)
	}
	emit(true) // flush any held-back trailing bytes

	if err := gen.Err(); err != nil && ctx.Err() == nil {
		sendEvent(w, f, "error", map[string]any{"message": err.Error()})
		return
	}
	elapsed := time.Since(start)
	var tps float64
	if elapsed > 0 {
		tps = float64(len(out)) / elapsed.Seconds()
	}
	sendEvent(w, f, "done", map[string]any{
		"tokens":       len(out),
		"promptTokens": len(promptIDs),
		"elapsedMs":    elapsed.Milliseconds(),
		"ttftMs":       ttft.Milliseconds(),
		"tokPerSec":    tps,
	})
}

// buildPrompt renders the conversation into the model's chat-template string via
// the chat package (family resolved at startup). Roles are normalized to
// user/assistant; the system prompt is passed separately. Unrecognized template
// → a plain raw completion.
func (s *server) buildPrompt(req chatRequest) string {
	system := strings.TrimSpace(req.System)
	turns := make([]chat.Turn, 0, len(req.Messages))
	for _, m := range req.Messages {
		role := m.Role
		if role == "model" {
			role = "assistant"
		}
		if role != "assistant" {
			role = "user"
		}
		turns = append(turns, chat.Turn{Role: role, Content: m.Content})
	}
	if s.tmpl != nil {
		return s.tmpl.Render(system, turns)
	}
	var b strings.Builder
	if system != "" {
		b.WriteString(system + "\n\n")
	}
	for _, t := range turns {
		b.WriteString(t.Content + "\n")
	}
	return b.String()
}

// sendEvent writes one SSE event with a JSON data payload (JSON keeps the data
// on a single line, sidestepping SSE's newline framing) and flushes it.
func sendEvent(w http.ResponseWriter, f http.Flusher, event string, data any) {
	b, _ := json.Marshal(data)
	fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, b)
	f.Flush()
}

// completeUTF8Len returns the length of the longest prefix of b that ends on a
// complete UTF-8 rune boundary, so a partial trailing byte-fallback sequence is
// held back until the bytes that finish the character arrive. (Copied from
// demo/gemma — the two demos are separate main packages.)
func completeUTF8Len(b []byte) int {
	i := 0
	for i < len(b) {
		if b[i] < utf8.RuneSelf {
			i++
			continue
		}
		if !utf8.FullRune(b[i:]) {
			break // incomplete trailing sequence — hold back from here
		}
		_, size := utf8.DecodeRune(b[i:])
		i += size
	}
	return i
}
