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
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"
	"syscall"
	"time"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/encoder"
	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// modelSpec is one --model entry: a served name (optional, from name=path) and
// the checkpoint path. Repeatable to serve a model zoo from one process.
type modelSpec struct{ name, path string }

// modelFlag collects repeated --model flags. Each value is "path" or "name=path".
type modelFlag []modelSpec

func (m *modelFlag) String() string {
	ps := make([]string, len(*m))
	for i, s := range *m {
		ps[i] = s.path
	}
	return strings.Join(ps, ",")
}

func (m *modelFlag) Set(v string) error {
	spec := modelSpec{path: v}
	// "name=path": split on the first '=' (a served name has no '=').
	if i := strings.IndexByte(v, '='); i > 0 {
		spec.name, spec.path = v[:i], v[i+1:]
	}
	if spec.path == "" {
		return fmt.Errorf("empty model path in %q", v)
	}
	*m = append(*m, spec)
	return nil
}

// config is the resolved command line for newServer (the flag set outgrew a
// positional signature once embeddings landed).
type config struct {
	models     modelFlag // decoder(s) (-model, repeatable); empty = no generative endpoints
	backend    string
	quant      string // global (per-model overrides are a follow-on)
	lora       string
	name       string // -served-model-name (applies only to a single unnamed --model)
	kvSessions int
	sessionDir string // -session-dir (also where /admin unload snapshots warm KV)
	allowAdmin bool   // -allow-admin: enable POST /admin/models/{load,unload}

	embedPath  string // encoder (-embed-model); "" = no /v1/embeddings
	embedQuant string // "" | f32 | q8
	embedName  string // -embed-served-model-name
}

func main() {
	var (
		cfg  config
		addr = flag.String("addr", ":8080", "listen address")
	)
	flag.StringVar(&cfg.sessionDir, "session-dir", "", "optional dir to persist/restore KV sessions across restarts (.giw-kv snapshots)")
	flag.BoolVar(&cfg.allowAdmin, "allow-admin", false, "enable POST /admin/models/{load,unload} (loads attacker-named paths — deliberate opt-in)")
	flag.Var(&cfg.models, "model", "generative model: a .gguf/.giw file or HF dir (chat/completions). Repeatable\n"+
		"as `name=path` to serve a model zoo from one process; requests route on the\n"+
		"OpenAI `model` field. N resident int8 models are expensive — prequant `.giw`\n"+
		"(--model name=path.giw) maps weights zero-copy and is the cheap way to keep a zoo.")
	flag.StringVar(&cfg.backend, "backend", "cpu", "compute backend: cpu | webgpu")
	flag.StringVar(&cfg.quant, "quant", "int8int8", "decoder weight quant (global): \"\" | int8 | int8int8 | int4")
	flag.StringVar(&cfg.lora, "lora", "", "optional PEFT LoRA adapter dir, merged into the (safetensors) base at load")
	flag.StringVar(&cfg.name, "served-model-name", "", "served id for a single unnamed --model (default: file/dir basename)")
	flag.IntVar(&cfg.kvSessions, "kv-sessions", 4, "number of conversations to keep prefilled for prompt-prefix KV reuse (0 disables)")
	flag.StringVar(&cfg.embedPath, "embed-model", "", "embedding model: a CodeRankEmbed HF dir (config.json + model.safetensors + tokenizer.json) for /v1/embeddings")
	flag.StringVar(&cfg.embedQuant, "embed-quant", "f32", "embedding weight precision: f32 | q8")
	flag.StringVar(&cfg.embedName, "embed-served-model-name", "", "embedding model id reported by /v1/models (default: dir basename)")
	flag.Parse()
	if len(cfg.models) == 0 && cfg.embedPath == "" && !cfg.allowAdmin {
		fmt.Fprintln(os.Stderr, "error: need at least one of --model, --embed-model, or --allow-admin")
		flag.Usage()
		os.Exit(2)
	}
	if err := sessionDirOK(cfg.sessionDir); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(2)
	}

	srv, err := newServer(cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
	if cfg.sessionDir != "" && cfg.kvSessions > 0 {
		for _, lm := range srv.models {
			lm.sessions.load(sessionSubdir(cfg.sessionDir, lm.fp))
		}
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/models", srv.handleModels)
	if len(srv.models) > 0 {
		mux.HandleFunc("POST /v1/chat/completions", srv.handleChat)
		mux.HandleFunc("POST /v1/completions", srv.handleCompletions)
		mux.HandleFunc("POST /v1/responses", srv.handleResponses)
	}
	if srv.embed != nil {
		mux.HandleFunc("POST /v1/embeddings", srv.handleEmbeddings)
	}
	mux.HandleFunc("POST /admin/models/load", srv.handleAdminLoad)
	mux.HandleFunc("POST /admin/models/unload", srv.handleAdminUnload)

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
		if cfg.sessionDir != "" && cfg.kvSessions > 0 {
			for _, lm := range srv.models {
				lm.mu.Lock()
				_ = lm.sessions.save(sessionSubdir(cfg.sessionDir, lm.fp))
				lm.mu.Unlock()
			}
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
	s := &server{models: map[string]*loadedModel{}, cfg: cfg, responses: newResponseStore(256)}
	for _, spec := range cfg.models {
		lm, err := loadDecoder(spec, cfg)
		if err != nil {
			return nil, err
		}
		if _, dup := s.models[lm.name]; dup {
			return nil, fmt.Errorf("duplicate served model name %q (use --model name=path to disambiguate)", lm.name)
		}
		s.models[lm.name] = lm
	}
	if cfg.embedPath != "" {
		if err := s.loadEncoder(cfg); err != nil {
			return nil, err
		}
	}
	return s, nil
}

// loadDecoder loads one generative model + tokenizer, resolves its chat template,
// and returns it as a *loadedModel. The served name is the spec's name=, else (a
// single unnamed --model) --served-model-name, else the file/dir basename.
func loadDecoder(spec modelSpec, cfg config) (*loadedModel, error) {
	loadTok := tokenizer.Load
	if strings.HasSuffix(spec.path, ".gguf") {
		loadTok = tokenizer.LoadGGUF
	}
	tk, err := loadTok(spec.path)
	if err != nil {
		return nil, fmt.Errorf("load tokenizer (%s): %w", spec.path, err)
	}
	t0 := time.Now()
	model, err := decoder.Load(spec.path, decoder.Options{Backend: cfg.backend, Quant: cfg.quant, LoRA: cfg.lora})
	if err != nil {
		return nil, fmt.Errorf("load model (%s): %w", spec.path, err)
	}
	mcfg := model.Config()
	name := spec.name
	if name == "" {
		if len(cfg.models) == 1 && cfg.name != "" {
			name = cfg.name
		} else {
			name = strings.TrimSuffix(filepath.Base(spec.path), ".gguf")
		}
	}
	fp := modelFingerprint(spec.path, model.Quant())
	lm := &loadedModel{
		tk: tk, model: model, vocab: mcfg.VocabSize, eosIDs: mcfg.EOSIDs(), name: name, fp: fp,
		// capHint 0: KV grows on demand. The fingerprint binds disk snapshots to
		// this exact model+quant so a -session-dir reused across models is rejected.
		sessions: newSessionLRU(model, cfg.kvSessions, 0, fp),
	}
	if tmpl, derr := chat.Detect(chat.Meta{ChatTemplate: tk.ChatTemplate(), HasToken: tk.Has}); derr == nil {
		lm.tmpl = tmpl
		for _, str := range tmpl.Stops().Strings {
			if id, ok := tk.TokenID(str); ok {
				lm.stopIDs = append(lm.stopIDs, id)
			}
		}
	}
	fmt.Fprintf(os.Stderr, "loaded %q: %d-layer model (vocab %d) in %s [chat: %s]\n",
		name, mcfg.NumLayers, mcfg.VocabSize, time.Since(t0).Round(time.Millisecond), templateName(lm.tmpl))
	return lm, nil
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
	if len(s.models) > 0 {
		names := make([]string, 0, len(s.models))
		for n := range s.models {
			names = append(names, n)
		}
		sort.Strings(names)
		parts = append(parts, fmt.Sprintf("chat:[%s]", strings.Join(names, " ")))
	}
	if s.embed != nil {
		parts = append(parts, fmt.Sprintf("embeddings:%q", s.embedID))
	}
	return strings.Join(parts, " | ")
}

// sessionSubdir gives a model its own --session-dir folder so warm-KV snapshots
// from different models don't collide (the dir name is a short hash of the
// fingerprint; the snapshot's own identity guard still rejects a mismatch).
func sessionSubdir(base, fp string) string {
	h := sha256.Sum256([]byte(fp))
	return filepath.Join(base, hex.EncodeToString(h[:8]))
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
