// Command serve is an OpenAI-compatible HTTP server for goinfer models: pure
// stdlib net/http, no dependencies. It speaks /v1/chat/completions,
// /v1/completions, /v1/embeddings, and /v1/models — enough for Open WebUI,
// LangChain, the OpenAI SDKs, and anything else that points at an OpenAI base
// URL — including streaming (SSE) and `response_format: json_schema` constrained
// decoding (the model physically cannot emit non-conforming JSON; see the
// constrain package).
//
// A generative (decoder) model is served via -model; an embedding (encoder)
// model via -embed-model. Either or both may be loaded in one process — like
// running llama.cpp/vLLM with a model per task, but without a separate router.
// Endpoints are registered for whatever is loaded.
//
//	go run ./cmd/serve --model ~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf
//	go run ./cmd/serve --embed-model ~/models/coderankembed       # /v1/embeddings only
//	# then point a client at http://localhost:8080/v1
//	curl localhost:8080/v1/chat/completions -d '{"model":"local",
//	  "messages":[{"role":"user","content":"hi"}]}'
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/encoder"
	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// config is the resolved command line for newServer (the flag set outgrew a
// positional signature once embeddings landed).
type config struct {
	modelPath  string // decoder (-model); "" = no generative endpoints
	backend    string
	quant      string
	lora       string
	name       string // -served-model-name
	kvSessions int

	embedPath  string // encoder (-embed-model); "" = no /v1/embeddings
	embedQuant string // "" | f32 | q8
	embedName  string // -embed-served-model-name
}

func main() {
	var (
		cfg        config
		addr       = flag.String("addr", ":8080", "listen address")
		sessionDir = flag.String("session-dir", "", "optional dir to persist/restore KV sessions across restarts (.giw-kv snapshots)")
	)
	flag.StringVar(&cfg.modelPath, "model", "", "generative model: a .gguf file or HF checkpoint dir (chat/completions)")
	flag.StringVar(&cfg.backend, "backend", "cpu", "compute backend: cpu | webgpu")
	flag.StringVar(&cfg.quant, "quant", "int8int8", "decoder weight quant: \"\" | int8 | int8int8 | int4")
	flag.StringVar(&cfg.lora, "lora", "", "optional PEFT LoRA adapter dir, merged into the (safetensors) base at load")
	flag.StringVar(&cfg.name, "served-model-name", "", "decoder id reported by /v1/models (default: file/dir basename)")
	flag.IntVar(&cfg.kvSessions, "kv-sessions", 4, "number of conversations to keep prefilled for prompt-prefix KV reuse (0 disables)")
	flag.StringVar(&cfg.embedPath, "embed-model", "", "embedding model: a CodeRankEmbed HF dir (config.json + model.safetensors + tokenizer.json) for /v1/embeddings")
	flag.StringVar(&cfg.embedQuant, "embed-quant", "f32", "embedding weight precision: f32 | q8")
	flag.StringVar(&cfg.embedName, "embed-served-model-name", "", "embedding model id reported by /v1/models (default: dir basename)")
	flag.Parse()
	if cfg.modelPath == "" && cfg.embedPath == "" {
		fmt.Fprintln(os.Stderr, "error: at least one of --model or --embed-model is required")
		flag.Usage()
		os.Exit(2)
	}
	if err := sessionDirOK(*sessionDir); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(2)
	}

	srv, err := newServer(cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
	if srv.sessions != nil && *sessionDir != "" && cfg.kvSessions > 0 {
		srv.sessions.load(*sessionDir)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/models", srv.handleModels)
	if srv.model != nil {
		mux.HandleFunc("POST /v1/chat/completions", srv.handleChat)
		mux.HandleFunc("POST /v1/completions", srv.handleCompletions)
	}
	if srv.embed != nil {
		mux.HandleFunc("POST /v1/embeddings", srv.handleEmbeddings)
	}

	httpSrv := &http.Server{Addr: *addr, Handler: mux}
	// Graceful shutdown: on SIGINT/SIGTERM, stop accepting, drain in-flight
	// generations, then checkpoint the KV sessions to -session-dir (if set).
	// done closes once that's complete so main waits for the save before exit.
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	done := make(chan struct{})
	go func() {
		defer close(done)
		<-sig
		fmt.Fprintln(os.Stderr, "\nshutting down…")
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		_ = httpSrv.Shutdown(ctx)
		if srv.sessions != nil && *sessionDir != "" && cfg.kvSessions > 0 {
			srv.mu.Lock()
			_ = srv.sessions.save(*sessionDir)
			srv.mu.Unlock()
		}
	}()

	fmt.Fprintf(os.Stderr, "goinfer serving on %s [%s]\n", *addr, srv.endpointSummary())
	if err := httpSrv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		fmt.Fprintf(os.Stderr, "server: %v\n", err)
		os.Exit(1)
	}
	<-done // let the shutdown handler finish checkpointing before we exit
}

// newServer loads the configured model(s): a decoder (with tokenizer, chat
// template, and KV sessions) and/or an encoder (with its tokenizer for token
// counting). At least one must be configured.
func newServer(cfg config) (*server, error) {
	s := &server{}
	if cfg.modelPath != "" {
		if err := s.loadDecoder(cfg); err != nil {
			return nil, err
		}
	}
	if cfg.embedPath != "" {
		if err := s.loadEncoder(cfg); err != nil {
			return nil, err
		}
	}
	return s, nil
}

// loadDecoder loads the generative model + tokenizer and resolves the chat template.
func (s *server) loadDecoder(cfg config) error {
	loadTok := tokenizer.Load
	if strings.HasSuffix(cfg.modelPath, ".gguf") {
		loadTok = tokenizer.LoadGGUF
	}
	tk, err := loadTok(cfg.modelPath)
	if err != nil {
		return fmt.Errorf("load tokenizer: %w", err)
	}
	t0 := time.Now()
	model, err := decoder.Load(cfg.modelPath, decoder.Options{Backend: cfg.backend, Quant: cfg.quant, LoRA: cfg.lora})
	if err != nil {
		return fmt.Errorf("load model: %w", err)
	}
	mcfg := model.Config()
	name := cfg.name
	if name == "" {
		name = strings.TrimSuffix(filepath.Base(cfg.modelPath), ".gguf")
	}
	s.tk, s.model = tk, model
	s.vocab, s.eosIDs, s.modelID = mcfg.VocabSize, mcfg.EOSIDs(), name
	// capHint 0: KV grows on demand. The fingerprint binds disk snapshots to this
	// exact model+quant so a -session-dir reused across models is rejected.
	s.sessions = newSessionLRU(model, cfg.kvSessions, 0, modelFingerprint(cfg.modelPath, model.Quant()))
	if tmpl, derr := chat.Detect(chat.Meta{ChatTemplate: tk.ChatTemplate(), HasToken: tk.Has}); derr == nil {
		s.tmpl = tmpl
		for _, str := range tmpl.Stops().Strings {
			if id, ok := tk.TokenID(str); ok {
				s.stopIDs = append(s.stopIDs, id)
			}
		}
	}
	fmt.Fprintf(os.Stderr, "loaded %d-layer model (vocab %d) in %s [chat: %s]\n",
		mcfg.NumLayers, mcfg.VocabSize, time.Since(t0).Round(time.Millisecond), templateName(s.tmpl))
	return nil
}

// loadEncoder loads the embedding model (f32 or int8) plus its tokenizer (used
// only to count tokens for usage.prompt_tokens — counting needs no forward pass,
// so it is cheaper than running EncodeTokensWithIDs, and works for both
// precisions, whereas that method is f32-only).
func (s *server) loadEncoder(cfg config) error {
	t0 := time.Now()
	var (
		enc  encoder.Encoder
		err  error
		prec string
	)
	switch strings.ToLower(cfg.embedQuant) {
	case "q8", "int8":
		enc, err = encoder.LoadQ8(cfg.embedPath)
		prec = "int8"
	case "", "f32":
		enc, err = encoder.Load(cfg.embedPath)
		prec = "f32"
	default:
		return fmt.Errorf("invalid -embed-quant %q (want f32 | q8)", cfg.embedQuant)
	}
	if err != nil {
		return fmt.Errorf("load embedding model: %w", err)
	}
	tok, err := embed.LoadTokenizer(filepath.Join(cfg.embedPath, "tokenizer.json"))
	if err != nil {
		return fmt.Errorf("load embedding tokenizer: %w", err)
	}
	name := cfg.embedName
	if name == "" {
		name = filepath.Base(strings.TrimRight(cfg.embedPath, "/"))
	}
	s.embed, s.embedTok, s.embedID, s.embedDim = enc, tok, name, enc.HiddenDim()
	fmt.Fprintf(os.Stderr, "loaded embedding model %q (dim %d, %s) in %s\n",
		name, s.embedDim, prec, time.Since(t0).Round(time.Millisecond))
	return nil
}

// endpointSummary describes the registered endpoints for the startup banner.
func (s *server) endpointSummary() string {
	var parts []string
	if s.model != nil {
		parts = append(parts, fmt.Sprintf("chat:%q kv-sessions:%d", s.modelID, s.sessions.size))
	}
	if s.embed != nil {
		parts = append(parts, fmt.Sprintf("embeddings:%q", s.embedID))
	}
	return strings.Join(parts, " | ")
}

// modelFingerprint identifies the loaded model for binding KV snapshots to it:
// the checkpoint's basename + size + mtime + resident quant. Two different
// models (or the same weights at a different quant — whose KV is incompatible)
// produce different fingerprints, so a -session-dir reused across them is
// rejected on load rather than fed stale KV. A missing/unstattable path degrades
// to name+quant (still distinguishes different files by name).
func modelFingerprint(path, quant string) string {
	base := filepath.Base(path)
	if fi, err := os.Stat(path); err == nil {
		return fmt.Sprintf("%s|%d|%d|%s", base, fi.Size(), fi.ModTime().UnixNano(), quant)
	}
	return fmt.Sprintf("%s|%s", base, quant)
}

func templateName(t *chat.Template) string {
	if t == nil {
		return "raw (no template)"
	}
	return t.Name()
}
