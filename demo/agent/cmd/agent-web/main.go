// Command agent-web is the browser UI over the shared agent core — a local
// "Claude-style" chat for the Go standard library, served from one static
// binary. Pure Go standard library on the server: net/http + a streamed
// NDJSON response, one embedded HTML page (//go:embed), no external assets,
// no CDNs, no cgo, no new dependencies beyond the agent core itself.
//
// The page renders the agent's machinery, not just its words: the constrained
// JSON tool call appears as an expandable chip, and ken's file:line-cited
// chunks land in a collapsible results card under it.
//
// Usage:
//
//	go run ./cmd/agent-web --model ~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf \
//	    --ken /path/to/ken-demo-go-stdlib
//	# → agent-web listening on http://127.0.0.1:8484
//
// Wire protocol (POST /api/chat {"message": "..."} → application/x-ndjson):
//
//	{"type":"decision","action":"search","query":"..."}   ← the constrained JSON, verbatim
//	{"type":"search","query":"...","results":"...","error":""}
//	{"type":"token","text":"..."}                          ← streamed answer spans
//	{"type":"done","reply":"..."} | {"type":"error","error":"..."}
package main

import (
	"context"
	_ "embed"
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"strings"
	"sync"

	"github.com/townsendmerino/goinfer/demo/agent/agent"
	"github.com/townsendmerino/goinfer/demo/agent/internal/embedmodel"
)

//go:embed index.html
var indexHTML []byte

// event is one NDJSON line in the /api/chat stream.
type event struct {
	Type    string `json:"type"`
	Action  string `json:"action,omitempty"`
	Query   string `json:"query,omitempty"`
	Results string `json:"results,omitempty"`
	Text    string `json:"text,omitempty"`
	Reply   string `json:"reply,omitempty"`
	Error   string `json:"error,omitempty"`
}

type server struct {
	sess *agent.Session
	mu   sync.Mutex // one turn at a time; the UI disables send while streaming
}

func main() {
	var (
		addr    = flag.String("addr", "127.0.0.1:8484", "listen address")
		model   = flag.String("model", "", "path to a .gguf file or HF checkpoint dir (omit in the -tags embed build)")
		visDir  = flag.String("vision", "", "optional Gemma 3 VL vision-tower dir to enable image input (defaults to --model when it carries a tower)")
		ken     = flag.String("ken", "ken-demo-go-stdlib", "path to the ken go-stdlib MCP demo binary (or any ken-mcp server)")
		quant   = flag.String("quant", "int8int8", "weight quant: \"\" | int8 | int8int8 | int4")
		maxTok  = flag.Int("max", 512, "max tokens per answer")
		temp    = flag.Float64("temp", 0.3, "answer-phase sampling temperature")
		topK    = flag.Int("top-k", 20, "top-k filter (0 = off)")
		topP    = flag.Float64("top-p", 0.8, "top-p / nucleus (0 = off)")
		kTop    = flag.Int("ken-top-k", 4, "chunks requested per ken search")
		freqPen = flag.Float64("freq-penalty", 0.3, "answer-phase frequency penalty (repetition/loop guard; 0 = off)")
		presPen = flag.Float64("presence-penalty", 0.0, "answer-phase presence penalty (0 = off)")
	)
	flag.Parse()

	opts := agent.Options{
		ModelPath: *model, Quant: *quant, Vision: *visDir,
		KenBin: *ken, KenTopK: *kTop,
		MaxTokens: *maxTok, Temperature: *temp, TopK: *topK, TopP: *topP,
		FrequencyPenalty: *freqPen, PresencePenalty: *presPen,
	}
	if *model == "" {
		raw, ok := embedmodel.Bytes()
		if !ok {
			fmt.Fprintln(os.Stderr, "error: --model is required (or build with -tags embed)")
			flag.Usage()
			os.Exit(2)
		}
		opts.ModelBytes = raw
	}

	fmt.Fprintln(os.Stderr, "loading model…")
	sess, err := agent.New(context.Background(), opts)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
	defer sess.Close()
	fmt.Fprintln(os.Stderr, sess.LoadSummary)
	fmt.Fprintf(os.Stderr, "ken up — tools: %s\n", strings.Join(sess.Tools(), ", "))

	s := &server{sess: sess}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /{$}", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Write(indexHTML)
	})
	mux.HandleFunc("GET /api/info", s.handleInfo)
	mux.HandleFunc("POST /api/chat", s.handleChat)
	mux.HandleFunc("POST /api/reset", s.handleReset)

	fmt.Fprintf(os.Stderr, "agent-web listening on http://%s\n", *addr)
	if err := http.ListenAndServe(*addr, mux); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
}

func (s *server) handleInfo(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"summary": s.sess.LoadSummary,
		"tools":   s.sess.Tools(),
	})
}

func (s *server) handleReset(w http.ResponseWriter, _ *http.Request) {
	s.sess.Reset()
	w.WriteHeader(http.StatusNoContent)
}

// handleChat runs one agent turn, streaming events as NDJSON. Closing the
// connection (the UI's Stop button aborts the fetch) cancels generation via
// the request context.
func (s *server) handleChat(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Message string `json:"message"`
		Image   string `json:"image"` // optional base64 data-URI (data:image/...;base64,...) or raw base64
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "expected JSON body {\"message\": \"...\", \"image\"?: \"data:...\"}", http.StatusBadRequest)
		return
	}
	hasImage := strings.TrimSpace(req.Image) != ""
	if strings.TrimSpace(req.Message) == "" && !hasImage {
		http.Error(w, "expected a message or an image", http.StatusBadRequest)
		return
	}
	var imgBytes []byte
	if hasImage {
		if !s.sess.HasVision() {
			http.Error(w, "this model has no vision tower; start with --vision <dir> (e.g. a gemma-3-4b-it dir)", http.StatusBadRequest)
			return
		}
		b, derr := decodeImage(req.Image)
		if derr != nil {
			http.Error(w, "decode image: "+derr.Error(), http.StatusBadRequest)
			return
		}
		imgBytes = b
		if strings.TrimSpace(req.Message) == "" {
			req.Message = "Describe this image."
		}
	}
	if !s.mu.TryLock() {
		http.Error(w, "a turn is already in progress", http.StatusConflict)
		return
	}
	defer s.mu.Unlock()

	w.Header().Set("Content-Type", "application/x-ndjson")
	w.Header().Set("Cache-Control", "no-store")
	flusher, _ := w.(http.Flusher)
	enc := json.NewEncoder(w)
	emit := func(e event) {
		_ = enc.Encode(e) // Encode appends the newline NDJSON needs
		if flusher != nil {
			flusher.Flush()
		}
	}

	ev := agent.Events{
		Decision: func(action, query string) {
			emit(event{Type: "decision", Action: action, Query: query})
		},
		Search: func(query, results string, err error) {
			e := event{Type: "search", Query: query, Results: results}
			if err != nil {
				e.Error = err.Error()
			}
			emit(e)
		},
		Token: func(text string) {
			emit(event{Type: "token", Text: text})
		},
	}

	var reply string
	var err error
	if hasImage {
		// The SigLIP prefill is heavy on CPU (minutes/image) — tell the UI to wait.
		emit(event{Type: "vision", Text: "analyzing image…"})
		reply, err = s.sess.TurnImage(r.Context(), req.Message, imgBytes, ev)
	} else {
		reply, err = s.sess.Turn(r.Context(), req.Message, ev)
	}
	if err != nil && r.Context().Err() == nil {
		emit(event{Type: "error", Error: err.Error()})
		return
	}
	emit(event{Type: "done", Reply: reply})
}

// decodeImage decodes a base64 image: a data: URI (data:<media>;base64,<payload>)
// or a bare base64 string. URL fetching is intentionally unsupported.
func decodeImage(s string) ([]byte, error) {
	payload := s
	if strings.HasPrefix(s, "data:") {
		i := strings.IndexByte(s, ',')
		if i < 0 {
			return nil, fmt.Errorf("malformed data: URI")
		}
		payload = s[i+1:]
	} else if strings.Contains(s, "://") {
		return nil, fmt.Errorf("image must be base64 (a remote URL is not fetched)")
	}
	return base64.StdEncoding.DecodeString(strings.TrimSpace(payload))
}
