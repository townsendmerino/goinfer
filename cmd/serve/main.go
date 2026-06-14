// Command serve is an OpenAI- and Anthropic-compatible HTTP server for goinfer
// models: pure stdlib net/http, no dependencies. It speaks /v1/chat/completions,
// /v1/completions, /v1/responses, /v1/embeddings, /v1/models, and the Anthropic
// Messages API (/v1/messages, /v1/messages/count_tokens) — enough for Open WebUI,
// LangChain, the OpenAI SDKs, Claude Code, and anything else that points at an
// OpenAI or Anthropic base URL — including streaming (SSE) and
// `response_format: json_schema` constrained decoding (the model physically
// cannot emit non-conforming JSON; see the constrain package).
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
	"github.com/townsendmerino/aikit/vision"
	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/giw"
	"github.com/townsendmerino/goinfer/multimodal"
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
	models        modelFlag // decoder(s) (-model, repeatable); empty = no generative endpoints
	backend       string
	quant         string // global (per-model overrides are a follow-on)
	kvPrec        string // GPU residency KV cache precision: "" | f32 | f16 (-kv)
	kvQuant       string // CPU KV cache storage precision: "" | f32 | i8 (-kv-quant)
	lora          string
	name          string // -served-model-name (applies only to a single unnamed --model)
	kvSessions    int
	sessionDir    string        // -session-dir (also where /admin unload snapshots warm KV)
	kvIdleDemote  time.Duration // -kv-idle-demote: tiered KV — demote a session idle this long to disk (0 = off)
	kvDemotedMax  int           // -kv-demoted-max: cap on the on-disk cold tier
	streamWeights bool          // -stream-weights: page MoE expert weights out of an mmap'd .giw under a RAM budget
	weightCacheGB float64       // -weight-cache: resident expert-weight budget in GB (0 = auto)
	maxQueue      int           // -max-queue: bounded per-model queue depth (0 = unbounded)
	allowAdmin    bool          // -allow-admin: enable POST /admin/models/{load,unload}
	visionPath    string        // -vision: dir holding the vision tower (SigLIP + projector) for a multimodal --model
	visionQuant   string        // -vision-quant: "f32" (default) | "int8" (W8A8; only faster on AVX512-VNNI — a WASH on AVX2)

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
	flag.StringVar(&cfg.visionPath, "vision", "", "vision tower dir (SigLIP encoder + projector) for a multimodal --model; enables image content parts. Defaults to the --model dir when it contains a vision tower")
	flag.StringVar(&cfg.visionQuant, "vision-quant", "f32", "vision encoder weight quant: f32 (default, bit-exact) | int8 (W8A8, cosine ~0.999) — int8 only speeds the compute-bound ViT prefill on AVX512-VNNI; on AVX2 it's a wash, so f32 is the default")
	flag.Var(&cfg.models, "model", "generative model: a .gguf/.giw file or HF dir (chat/completions). Repeatable\n"+
		"as `name=path` to serve a model zoo from one process; requests route on the\n"+
		"OpenAI `model` field. N resident int8 models are expensive — prequant `.giw`\n"+
		"(--model name=path.giw) maps weights zero-copy and is the cheap way to keep a zoo.")
	flag.StringVar(&cfg.backend, "backend", "cpu", "compute backend: cpu | webgpu")
	flag.StringVar(&cfg.quant, "quant", "int8int8", "decoder weight quant (global): \"\" | int8 | int8int8 | int4")
	flag.StringVar(&cfg.kvPrec, "kv", "f32", "GPU residency KV cache precision: f32 (bit-exact, 16k ctx) | f16 (lossy, 32k ctx) | i8 (lossy, ~64k ctx) — webgpu backend only")
	flag.StringVar(&cfg.kvQuant, "kv-quant", "f32", "CPU KV cache storage: f32 (default, bit-exact) | i8 (per-head int8, ~4× smaller, lossy — argmax ~90%+; excludes MoE/gemma4/qwen3.5)")
	flag.StringVar(&cfg.lora, "lora", "", "optional PEFT LoRA adapter dir, merged into the (safetensors) base at load")
	flag.StringVar(&cfg.name, "served-model-name", "", "served id for a single unnamed --model (default: file/dir basename)")
	flag.IntVar(&cfg.kvSessions, "kv-sessions", 4, "number of conversations to keep prefilled in RAM for prompt-prefix KV reuse (0 disables)")
	flag.DurationVar(&cfg.kvIdleDemote, "kv-idle-demote", 0, "tiered KV: demote a warm session's KV to -session-dir once it's been idle this long, faulting it back on the next matching request (e.g. 10m; 0 = off). Lets a small-RAM box serve many intermittent chats. Needs -session-dir and -kv-sessions > 0")
	flag.IntVar(&cfg.kvDemotedMax, "kv-demoted-max", 64, "tiered KV: max demoted (on-disk) sessions to keep; older ones are dropped (only with -kv-idle-demote)")
	flag.BoolVar(&cfg.streamWeights, "stream-weights", false, "page model weights on demand out of an mmap'd .giw, capping resident RAM to -weight-cache instead of holding all weights: MoE expert demand-paging (run a 35B-A3B on ~16-20 GB) or dense per-layer streaming (run a model bigger than RAM). Bit-exact; trades RAM for fault latency. Needs a .giw model")
	flag.Float64Var(&cfg.weightCacheGB, "weight-cache", 0, "resident expert-weight budget in GB for -stream-weights (0 = auto, ~half of available RAM)")
	flag.IntVar(&cfg.maxQueue, "max-queue", 8, "per-model backpressure: max queued requests before 429 (0 = unbounded)")
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
	if cfg.kvIdleDemote > 0 && (cfg.sessionDir == "" || cfg.kvSessions <= 0) {
		fmt.Fprintln(os.Stderr, "error: -kv-idle-demote needs -session-dir and -kv-sessions > 0")
		os.Exit(2)
	}

	srv, err := newServer(cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
	if cfg.sessionDir != "" && cfg.kvSessions > 0 {
		for _, lm := range srv.models {
			sub := sessionSubdir(cfg.sessionDir, lm.fp)
			lm.sessions.load(sub)
			if cfg.kvIdleDemote > 0 {
				lm.sessions.enableTiering(sub, cfg.kvIdleDemote, cfg.kvDemotedMax)
			}
		}
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/models", srv.handleModels)
	if len(srv.models) > 0 {
		mux.HandleFunc("POST /v1/chat/completions", srv.handleChat)
		mux.HandleFunc("POST /v1/completions", srv.handleCompletions)
		mux.HandleFunc("POST /v1/responses", srv.handleResponses)
		mux.HandleFunc("POST /v1/messages", srv.handleMessages)
		mux.HandleFunc("POST /v1/messages/count_tokens", srv.handleCountTokens)
	}
	if srv.embed != nil {
		mux.HandleFunc("POST /v1/embeddings", srv.handleEmbeddings)
	}
	mux.HandleFunc("POST /admin/models/load", srv.handleAdminLoad)
	mux.HandleFunc("POST /admin/models/unload", srv.handleAdminUnload)

	httpSrv := &http.Server{Addr: *addr, Handler: mux}

	// Tiered KV: a background ticker demotes idle sessions to disk. It takes the
	// same per-model lock the request path holds, so it never races a generation;
	// stopDemote halts it before the shutdown checkpoint runs.
	stopDemote := make(chan struct{})
	if cfg.kvIdleDemote > 0 {
		go demoteLoop(srv, cfg.kvIdleDemote, stopDemote)
	}

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
		close(stopDemote) // stop demoting before we checkpoint
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		_ = httpSrv.Shutdown(ctx)
		if cfg.sessionDir != "" && cfg.kvSessions > 0 {
			for _, lm := range srv.models {
				lm.mu.Lock()
				_ = lm.sessions.save(sessionSubdir(cfg.sessionDir, lm.fp))
				lm.sessions.removeColdFiles() // the cold tier is in-process; clear its scratch
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
	if err := s.loadVisionTower(cfg); err != nil {
		return nil, err
	}
	return s, nil
}

// loadVisionTower attaches a SigLIP encoder + projector to the (single) loaded
// model, making it vision-capable (serve then accepts image content parts). The
// dir is -vision if set, else the sole --model's own dir when it carries a vision
// tower (auto-discovery). A multimodal tower only makes sense for a single model,
// so it errors if -vision is set with a model zoo. Absent a tower it is a no-op:
// text-only serving is unchanged.
func (s *server) loadVisionTower(cfg config) error {
	dir := cfg.visionPath
	if dir == "" {
		// Auto-discover: a single --model whose dir holds the projector weights.
		if len(cfg.models) == 1 {
			cand := cfg.models[0].path
			if fi, err := os.Stat(cand); err == nil && fi.IsDir() {
				if _, err := multimodal.LoadProjector(cand); err == nil {
					dir = cand
				}
			}
		}
		if dir == "" {
			return nil // no vision tower — text-only model
		}
	}
	if len(s.models) != 1 {
		return fmt.Errorf("-vision needs exactly one --model (got %d)", len(s.models))
	}
	// The resident GPU encoder needs int8 (W8A8) matmul weights, so --backend
	// webgpu implies an int8 tower even if --vision-quant wasn't set.
	int8Tower := cfg.visionQuant == "int8" || cfg.backend == "webgpu"
	enc, err := vision.LoadEncoder(dir, int8Tower)
	if err != nil {
		return fmt.Errorf("load vision encoder (%s): %w", dir, err)
	}
	if cfg.backend == "webgpu" {
		if err := enc.EnableResident(); err != nil {
			return fmt.Errorf("enable resident GPU vision encoder: %w", err)
		}
	}
	proj, err := multimodal.LoadProjector(dir)
	if err != nil {
		return fmt.Errorf("load vision projector (%s): %w", dir, err)
	}
	for _, lm := range s.models {
		lm.venc, lm.vproj, lm.vcfg = enc, proj, vision.Gemma3()
		lm.vimgTok = -1
		if id, ok := lm.tk.TokenID(imageSoftToken); ok {
			lm.vimgTok = id
		}
		if lm.vimgTok < 0 {
			return fmt.Errorf("vision: tokenizer has no %q token (needed to place image embeddings)", imageSoftToken)
		}
		vq := "f32"
		if int8Tower {
			vq = "int8"
		}
		if cfg.backend == "webgpu" {
			vq = "int8/webgpu-resident"
		}
		fmt.Fprintf(os.Stderr, "loaded vision tower for %q (%d image tokens/image, soft-token id %d, encoder %s) from %s\n", lm.name, proj.MMTokens(), lm.vimgTok, vq, dir)
	}
	return nil
}

// loadDecoder loads one generative model + tokenizer, resolves its chat template,
// and returns it as a *loadedModel. The served name is the spec's name=, else (a
// single unnamed --model) --served-model-name, else the file/dir basename.
func loadDecoder(spec modelSpec, cfg config) (*loadedModel, error) {
	tk, err := loadDecoderTokenizer(spec.path)
	if err != nil {
		return nil, fmt.Errorf("load tokenizer (%s): %w", spec.path, err)
	}
	t0 := time.Now()
	model, err := decoder.Load(spec.path, decoder.Options{Backend: cfg.backend, Quant: cfg.quant, LoRA: cfg.lora, KVPrecision: cfg.kvPrec, KVQuant: cfg.kvQuant, StreamWeights: cfg.streamWeights, WeightCacheBytes: int64(cfg.weightCacheGB * 1e9)})
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
	if cfg.maxQueue > 0 {
		lm.queue = make(chan struct{}, 1+cfg.maxQueue)
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

// loadDecoderTokenizer loads the tokenizer for a decoder model, picking the loader
// by extension: a prequant .giw carries its tokenizer as an embedded metadata-GGUF
// (read just that half — the weights are tens of GB and mmap'd by decoder.Load), a
// .gguf reads its own metadata, and anything else is a SentencePiece/HF dir.
func loadDecoderTokenizer(path string) (*tokenizer.Tokenizer, error) {
	switch {
	case strings.HasSuffix(path, ".giw"):
		tokGGUF, err := giw.ReadTokFile(path)
		if err != nil {
			return nil, err
		}
		return tokenizer.LoadGGUFBytes(tokGGUF)
	case strings.HasSuffix(path, ".gguf"):
		return tokenizer.LoadGGUF(path)
	default:
		return tokenizer.Load(path)
	}
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

// demoteLoop periodically demotes idle KV sessions across all models to disk
// (tiered KV). It polls at a fraction of the idle threshold (clamped to [5s, 1m])
// and takes each model's lock per sweep, so it stalls no in-flight generation and
// skips a busy model until its lock is free. Returns when stop is closed.
func demoteLoop(srv *server, idle time.Duration, stop <-chan struct{}) {
	period := idle / 4
	if period < 5*time.Second {
		period = 5 * time.Second
	}
	if period > time.Minute {
		period = time.Minute
	}
	t := time.NewTicker(period)
	defer t.Stop()
	for {
		select {
		case <-stop:
			return
		case <-t.C:
			for _, lm := range srv.models {
				lm.mu.Lock()
				if n := lm.sessions.demoteIdle(); n > 0 {
					fmt.Fprintf(os.Stderr, "tiered-kv: demoted %d idle session(s) for %q\n", n, lm.name)
				}
				lm.mu.Unlock()
			}
		}
	}
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
