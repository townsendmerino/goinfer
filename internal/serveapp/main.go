// Command serve is an OpenAI- and Anthropic-compatible HTTP server for goinfer
// models: pure stdlib net/http, no dependencies. It speaks /v1/chat/completions,
// /v1/completions, /v1/responses, /v1/embeddings, /v1/models, and the Anthropic
// Messages API (/v1/messages, /v1/messages/count_tokens), and TypeSafe's decisions
// shape (/v1/systemone, answered by label scoring) — enough for Open WebUI,
// LangChain, the OpenAI SDKs, Claude Code, jevx, and anything else that points at an
// OpenAI, Anthropic or TypeSafe base URL — including streaming (SSE) and
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
package serveapp

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/encoder"
	"github.com/townsendmerino/aikit/vision"
	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/clef"
	"github.com/townsendmerino/goinfer/internal/decide"
	"github.com/townsendmerino/goinfer/internal/loadflags"
	"github.com/townsendmerino/goinfer/internal/modelload"
	"github.com/townsendmerino/goinfer/internal/prequant"
	"github.com/townsendmerino/goinfer/internal/pullcmd"
	"github.com/townsendmerino/goinfer/internal/servecheck"
	"github.com/townsendmerino/goinfer/multimodal"
	"github.com/townsendmerino/goinfer/pull"
)

// modelSpec is one --model entry: a served name (optional, from name=path), the
// checkpoint path, and optional per-model overrides of knobs that are otherwise
// server-global defaults — so a zoo can be heterogeneous (e.g. stream-page a big
// MoE while a small dense model stays resident). Override fields are pointers
// because "inherit the default" must be distinct from the zero value (quant="" is
// f32, a real setting, not "unset").
type modelSpec struct {
	name, path string

	quant       *string  // quant=
	lora        *string  // lora=
	kvPrec      *string  // kv=
	kvQuant     *string  // kv-quant=
	stream      *bool    // stream (bare) or stream=true|false
	weightCache *float64 // weight-cache= (GB)
	embedInt4   *bool    // embed-int4 (bare) or embed-int4=true|false
	ctxSize     *int     // ctx= (resident KV positions)
	head        *string  // head= (a trained decision head dir: POST /v1/systemone answers by Route B)
}

// modelFlag collects repeated --model flags. Each value is
// "[name=]path[,key=val|,flag]..." — the first comma field is the (optionally
// named) path, the rest are per-model overrides. Paths may not contain commas.
type modelFlag []modelSpec

func (m *modelFlag) String() string {
	ps := make([]string, len(*m))
	for i, s := range *m {
		ps[i] = s.path
	}
	return strings.Join(ps, ",")
}

func (m *modelFlag) Set(v string) error {
	fields := strings.Split(v, ",")
	spec := modelSpec{path: fields[0]}
	// "name=path": split the first field on the first '=' (a served name has no '=').
	if i := strings.IndexByte(fields[0], '='); i > 0 {
		spec.name, spec.path = fields[0][:i], fields[0][i+1:]
	}
	if spec.path == "" {
		return fmt.Errorf("empty model path in %q", v)
	}
	for _, f := range fields[1:] {
		key, val, hasVal := strings.Cut(f, "=")
		if err := spec.setOverride(key, val, hasVal); err != nil {
			return fmt.Errorf("--model %q: %w", spec.path, err)
		}
	}
	*m = append(*m, spec)
	return nil
}

// setOverride records one per-model "key=val" (or bare "flag") override.
func (s *modelSpec) setOverride(key, val string, hasVal bool) error {
	pbool := func() (*bool, error) {
		b := true
		if hasVal {
			var err error
			if b, err = strconv.ParseBool(val); err != nil {
				return nil, fmt.Errorf("%s=%q: %w", key, val, err)
			}
		}
		return &b, nil
	}
	var err error
	switch key {
	case "quant":
		s.quant = &val
	case "lora":
		s.lora = &val
	case "kv":
		s.kvPrec = &val
	case "kv-quant":
		s.kvQuant = &val
	case "head":
		if val == "" {
			return errors.New("head= needs a directory (judge_config.json, head.safetensors, calibration.json)")
		}
		s.head = &val
	case "ctx":
		n, perr := strconv.Atoi(val)
		if perr != nil {
			return fmt.Errorf("ctx=%q: %w", val, perr)
		}
		if n < 0 {
			return fmt.Errorf("ctx=%d: must be >= 0 (0 = backend default)", n)
		}
		s.ctxSize = &n
	case "weight-cache":
		gb, perr := strconv.ParseFloat(val, 64)
		if perr != nil {
			return fmt.Errorf("weight-cache=%q: %w", val, perr)
		}
		s.weightCache = &gb
	case "stream":
		s.stream, err = pbool()
	case "embed-int4":
		s.embedInt4, err = pbool()
	default:
		return fmt.Errorf("unknown per-model option %q", key)
	}
	return err
}

// explicitQuant returns the quant the user EXPLICITLY chose for this model — the per-model
// `quant=` override if present, else the global --quant if it was actually passed — or "" if
// neither was set (the process default). Used only for the .giw mismatch check (T1-7): a bare
// default must never conflict with an already-baked bundle.
func (s modelSpec) explicitQuant(cfg config) string {
	if s.quant != nil {
		return *s.quant
	}
	return cfg.load.ExplicitQuant()
}

// options resolves this spec's overrides over the server-global defaults (cfg.load, the flags chat
// shares) into a decoder.Options. Backend is process-wide (GPU device init), never per-model.
func (s modelSpec) options(cfg config) decoder.Options {
	o := cfg.load.Options()
	o.Quant = orStr(s.quant, o.Quant)
	if s.head != nil || isClefDir(s.path) { // a decision head (JEV or Clef) loads at the decision models' default unless a quant was chosen (D6b)
		o.Quant = decide.HeadQuant(s.explicitQuant(cfg), s.quant != nil || cfg.load.QuantSet)
	}
	o.LoRA = orStr(s.lora, o.LoRA)
	o.KVPrecision = orStr(s.kvPrec, cfg.load.KV)
	o.KVQuant = loadflags.CPUKV(orStr(s.kvQuant, cfg.kvQuant), o.KVPrecision)
	o.StreamWeights = orBool(s.stream, o.StreamWeights)
	o.WeightCacheBytes = int64(orFloat(s.weightCache, cfg.load.WeightCacheGB) * 1e9)
	o.EmbedInt4 = orBool(s.embedInt4, o.EmbedInt4)
	o.ResidentContext = orInt(s.ctxSize, o.ResidentContext)
	// MC1 (docs/tasks/task-concurrency-2026-09.md): -kv-sessions also asks a GPU-resident backend for that many KV
	// slots, so interleaved conversations keep their own prefix resident. The backend clamps it to its memory guard;
	// the banner reports what it allocated.
	o.ResidentKVSlots = cfg.kvSessions
	o.ResidentKVSlotsDefault = !cfg.kvSessionsSet // E-P09: Metal keeps 2 slots unless -kv-sessions was given
	o.ResidentPrefillChunk = cfg.prefillChunk     // MC3 chunked prefill (docs/tasks/task-concurrency-2026-09.md); 0 = off
	o.CPUBatchDecode = cfg.cpuBatch               // MC3c step 2: batched CPU decode (-cpu-batch)
	return o
}

// specHead loads a --model entry's head= (a trained decision head, D4) and, for a head whose adapter is unmerged, makes
// that adapter the model's LoRA so it is merged at load, as goinfer-chat decide --head does. A different lora= is
// refused rather than one of the two silently winning. nil, nil when the entry has no head.
func specHead(s modelSpec, o *decoder.Options) (*decide.Head, error) {
	if s.head == nil {
		return nil, nil
	}
	h, err := decide.LoadHead(*s.head)
	if err != nil {
		return nil, err
	}
	// A .giw built with the adapter merged in (prequant -lora) needs no LoRA; AdapterLoRA checks its sidecar.
	a, err := prequant.AdapterLoRA(h.AdapterDir(), s.path)
	if err != nil {
		return nil, err
	}
	if a != "" {
		if o.LoRA != "" && o.LoRA != a {
			return nil, fmt.Errorf("lora=%s and the head's own adapter %s: an unmerged head brings its adapter", o.LoRA, a)
		}
		o.LoRA = a
	}
	return h, nil
}

// adapterSpec is one --adapter entry (#7): a served name, the --model it attaches
// to (base), and the PEFT adapter dir. The base's resident weights are shared, so
// each adapter costs only its low-rank A/B bytes — N fine-tunes off one base.
type adapterSpec struct {
	name, base, dir string
}

// adapterFlag collects repeated --adapter flags, each "serveName=baseName=dir".
type adapterFlag []adapterSpec

func (a *adapterFlag) String() string {
	ns := make([]string, len(*a))
	for i, s := range *a {
		ns[i] = s.name
	}
	return strings.Join(ns, ",")
}

func (a *adapterFlag) Set(v string) error {
	name, rest, ok := strings.Cut(v, "=")
	if !ok {
		return fmt.Errorf("--adapter %q: want serveName=baseName=dir", v)
	}
	base, dir, ok := strings.Cut(rest, "=")
	if !ok {
		return fmt.Errorf("--adapter %q: want serveName=baseName=dir", v)
	}
	if name == "" || base == "" || dir == "" {
		return fmt.Errorf("--adapter %q: serveName, baseName, and dir are all required", v)
	}
	*a = append(*a, adapterSpec{name: name, base: base, dir: dir})
	return nil
}

func orStr(p *string, def string) string {
	if p != nil {
		return *p
	}
	return def
}
func orBool(p *bool, def bool) bool {
	if p != nil {
		return *p
	}
	return def
}
func orFloat(p *float64, def float64) float64 {
	if p != nil {
		return *p
	}
	return def
}
func orInt(p *int, def int) int {
	if p != nil {
		return *p
	}
	return def
}

// addrIsLoopback reports whether a -addr value binds ONLY the loopback
// interface — never reachable from another machine, so unauthenticated and
// unencrypted are the local-desktop default rather than a network exposure.
// A bare port (":8080") and 0.0.0.0/[::] bind every interface and don't count,
// nor does any other hostname (it might resolve off-box on some networks).
func addrIsLoopback(addr string) bool {
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		host = addr // no port separator; treat the whole string as the host
	}
	if host == "" {
		return false
	}
	if host == "localhost" {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

// config is the resolved command line for newServer (the flag set outgrew a
// positional signature once embeddings landed).
type config struct {
	models   modelFlag   // decoder(s) (-model, repeatable); empty = no generative endpoints
	adapters adapterFlag // compute-time LoRA adapters (-adapter, repeatable); each shares a base model's resident weights (#7)
	// load is the model-loading flags goinfer-chat shares (internal/loadflags): server-global
	// DEFAULTS, each of which a --model spec can override (see modelSpec / modelFlag.Set).
	load            loadflags.Flags
	kvQuant         string // DEPRECATED -kv-quant: CPU KV cache override, "" = follow -kv
	name            string // -served-model-name (applies only to a single unnamed --model)
	kvSessions      int
	kvSessionsSet   bool          // -kv-sessions was given on the command line (flag.Visit after Parse)
	sessionDir      string        // -session-dir (also where /admin unload snapshots warm KV)
	kvIdleDemote    time.Duration // -kv-idle-demote: tiered KV — demote a session idle this long to disk (0 = off)
	kvDemotedMax    int           // -kv-demoted-max: cap on the on-disk cold tier
	maxQueue        int           // -max-queue: bounded per-model queue depth (0 = unbounded)
	maxConcurrent   int           // -max-concurrent: generations one CPU model may run at once (MC3c; default 4, owner 2026-09-26; 1 = serialized)
	cpuBatch        int           // -cpu-batch: decoder.CPUBatchAuto (default) / CPUBatchOn / CPUBatchOff (MC3c step 2)
	prefillChunk    int           // -prefill-chunk: under MC3, a long prompt arriving while others decode prefills in chunks of this many tokens (default 512); 0 = off
	jobDir          string        // -job-dir (J2, task-work-queue-2026-09.md): optional dir for the job journal (one JSONL line per state transition); "" = in-memory job tracking only, no durability
	maxInflight     int           // -max-inflight: global cap on concurrent inference handlers (bounds pre-queue work; 0 = unbounded)
	maxBodyBytes    int64         // -max-body-bytes: request-body cap (0 = derive from the model's context window)
	unloadDrainWait time.Duration // -unload-drain-wait: how long an unload waits for in-flight requests to drain before 202 (native free continues detached)
	spec            string        // -spec: "" (off) | "ngram" — lossless n-gram speculative decode
	specAdaptive    bool          // -spec-adaptive: MC4 candidate, "speculate when alone, batch under load" (needs -spec ngram and a resident that batches; off has no effect otherwise)
	drafter         string        // -drafter: dir of a pretrained BLOCK drafter (DFlash); resident GPU backends only
	allowAdmin      bool          // -allow-admin: enable POST /admin/models/{load,unload}
	logRequests     bool          // -log-requests: one stderr line per generation request (R25)
	haltFile        string        // -halt-file: polled every 250ms; present ⇒ halted, absent ⇒ resumed (K2)
	haltExitCode    int           // -halt-exit-code: nonzero ⇒ halt exits the process with this code after quiescence (K2); 0 = stay up
	adminSocket     string        // -admin-socket: serve /admin/* on this Unix socket instead of the TCP listener (K5); "" = off
	web             bool          // -web: serve the local browser UI + its model-pull routes
	requireBE       bool          // -require-backend: refuse to start when a model silently fell back off the requested backend's fast paths (resident decode / batched prefill)
	visionPath      string        // -vision: dir holding the vision tower (SigLIP + projector) for a multimodal --model
	noSelfTest      bool          // -no-selftest: skip the startup self-tests (H2)
	visionMaxPixels int           // -vision-max-pixels: lowers GLM-OCR's image pixel budget (0 = the model's own 4.82 MP ceiling)
	visionQuant     string        // -vision-quant: "f32" (default) | "int8" (W8A8; only faster on AVX512-VNNI — a WASH on AVX2)
	visionDevice    string        // -vision-device: "auto" (default: the backend's device tower when it has one) | "cpu"
	// decisions (D5, docs/tasks/task-constrained-confidence.md): POST /v1/systemone's label-scoring template, and an
	// optional calibration.json of per-kind temperatures ("" = none: every answer is uncalibrated).
	decisionsTemplate string
	decisionsCal      string

	// thinking / reasoningFmt: how a reasoning model's think block is prompted and surfaced (think.go). thinking is the
	// server default render mode (-thinking: asis | template | on | off), a request overrides it per call;
	// reasoningFmt is -reasoning-format (deepseek | deepseek-legacy | none).
	thinking     string
	reasoningFmt string
	// reasoningBudget is -reasoning-budget (auto | unlimited | N tokens): a ceiling on how long a thinking reply may think
	// before serve forces the block closed, so a reply always has room to answer (budget.go).
	reasoningBudget string
	// toolFormat is -tool-format (auto | hermes | template): how a family whose own template declares a native tool form (Qwen3.5, Gemma 4's canonical
	// template) has its tools put in the prompt (chat/qwen_xml_tools.go, chat/gemma4_tools.go). Applied to the model's template at load, like -thinking.
	toolFormat string
	// lenientToolCalls is -lenient-tool-calls: also read ONE fenced JSON call at the end of a reply as a tool call, on the <tool_call> families
	// (chat.Template.WithLenientToolCalls, docs/queue-correctness.md G39). Off by default. Applied to the model's template at load.
	lenientToolCalls bool

	embedPath  string // encoder (-embed-model); "" = no /v1/embeddings
	embedQuant string // "" | f32 | q8
	embedName  string // -embed-served-model-name
	// embedResize is -embed-image-resize: how an image-embedding model resizes an image before its tower ("" = the
	// model's default). EmbeddingGemma 2 takes bilinear (aikit's) or bicubic (its reference processor's).
	embedResize string
}

// serveFlags is what the serve flags parse into: the config every model shares, the four listener flags that
// are not part of it, --version, and the model-loading flags goinfer-chat shares (loadflags).
// markGivenFlags records, after fs is parsed, which of cfg's flags the command line gave rather than defaulted: today
// -kv-sessions, whose default 4 Metal lowers to 2 GPU KV slots (E-P09) while a given count is kept.
func markGivenFlags(fs *flag.FlagSet, cfg *config) {
	fs.Visit(func(f *flag.Flag) {
		if f.Name == "kv-sessions" {
			cfg.kvSessionsSet = true
		}
	})
}

type serveFlags struct {
	cfg                           config
	addr, apiKey, tlsCert, tlsKey *string
	showVersion                   *bool
	lf                            *loadflags.Flags
}

// registerFlags adds every serve flag to fs. Main registers on flag.CommandLine; TestFlagsDoc registers on a
// fresh set, so docs/flags.md is generated from the flags themselves and cannot drift from them.
func registerFlags(fs *flag.FlagSet) *serveFlags {
	sf := &serveFlags{}
	cfg := &sf.cfg
	sf.addr = fs.String("addr", "127.0.0.1:8080", "listen address (defaults to loopback; use 0.0.0.0:8080 to expose, and set -api-key and -tls-cert/-tls-key — or put a TLS-terminating reverse proxy in front — when you do)")
	sf.apiKey = fs.String("api-key", "", "optional shared secret; when set, every request must send it as `Authorization: Bearer <key>` or `x-api-key: <key>`. Falls back to $GOINFER_API_KEY. REQUIRED with -allow-admin.")
	sf.tlsCert = fs.String("tls-cert", "", "PEM certificate file; with -tls-key, serves HTTPS instead of plaintext HTTP. Without it, -api-key and every prompt/completion travel in cleartext — fine on loopback, not on a shared network. For ACME/auto-renewal, put a reverse proxy (Caddy, nginx, Traefik) in front instead and leave this unset.")
	sf.tlsKey = fs.String("tls-key", "", "PEM private key file, paired with -tls-cert")
	fs.StringVar(&cfg.sessionDir, "session-dir", "", "optional dir to persist/restore KV sessions across restarts (.giw-kv snapshots)")
	fs.StringVar(&cfg.jobDir, "job-dir", "", "J2 (task-work-queue-2026-09.md): optional dir for a durable job journal — one JSONL line per generation state transition (pending/running/done/failed/cancelled), 0700/0600 permissions matching -session-dir. On restart, any job still 'running' at the last recorded transition is marked 'interrupted', never silently 'failed'. Off by default: every generation still gets an in-memory job record, just no durability across a restart")
	fs.BoolVar(&cfg.web, "web", false, "serve a local browser UI at / — chat with the loaded model, and pull and load GGUF files or whole safetensors checkpoints from HuggingFace, on the same server and the same /v1 routes any other client uses (one embedded HTML file; no external assets, so it works offline). Off by default: the page is static, but its pull route starts a caller-named multi-gigabyte download and writes it to disk. On a non-loopback bind the existing -api-key requirement applies as usual")
	fs.BoolVar(&cfg.allowAdmin, "allow-admin", false, "enable /admin/* on THIS (TCP) listener — model load/unload (loads attacker-named paths), GET /admin/generations, POST /admin/generations/{id}/cancel, POST /admin/halt, POST /admin/resume (deliberate opt-in; requires -api-key). A /v1 client holding the same key can reach every one of these routes too, including halt/resume. Ignored when -admin-socket is set: /admin/* is then not registered on TCP at all (a request 404s, not 403s — this listener does not admit the surface exists), and is served on the socket instead with no key check")
	fs.BoolVar(&cfg.logRequests, "log-requests", false, "write one line to stderr per generation request (/v1/chat/completions, /v1/completions, /v1/responses, /v1/messages) when it finishes: route, model, status, prompt and completion tokens, time to first token and total time. Off by default; a request that failed before generating shows `-` for the model and token counts")
	fs.StringVar(&cfg.haltFile, "halt-file", "", "K2: poll this path every 250ms — present halts the server (every inference route 503s, in-flight generations are cancelled), absent resumes it. No HTTP call, socket, or signal needed; a supervisor halts with `touch` and resumes with `rm`. The model stays loaded either way; resume is instant. Off by default")
	fs.IntVar(&cfg.haltExitCode, "halt-exit-code", 0, "K2: when nonzero, any halt (admin, -halt-file, or SIGUSR1) exits the process with this code once every cancelled generation has actually stopped, instead of staying up halted. For a supervisor whose restart policy must not undo a deliberate halt (RestartPreventExitStatus=N or the equivalent). 0 (default) means halt never exits the process")
	fs.StringVar(&cfg.adminSocket, "admin-socket", "", fmt.Sprintf("K5: serve /admin/* (load/unload plus K1/K2's cancel/list/halt/resume) on a Unix socket instead of the TCP listener — mode 0600, unlinked and recreated fresh at start, no -api-key check (the socket's file permissions are the auth). Removes /admin/* from the TCP listener entirely (404, not 403) — see -allow-admin. Suggested path: %s. Off by default (empty = /admin/* stays on TCP, gated by -allow-admin as before). Control it with the same binary: `%[2]s status|ls|cancel <id>|halt [reason]|resume` talks to this socket (defaults to the suggested path above when -admin-socket is not repeated on that command line)", defaultAdminSocketPath(), filepath.Base(os.Args[0])))
	fs.StringVar(&cfg.visionPath, "vision", "", "vision tower dir for a multimodal --model (auto-discovered per family: SigLIP+projector for Gemma 3, Qwen2.5-VL's own ViT, or Gemma 4's own encoder — N-35, docs/audit-2026-09-10.md); enables image content parts. Defaults to the --model dir when it contains a vision tower")
	fs.StringVar(&cfg.decisionsTemplate, "decisions-template", "chat-v1", "POST /v1/systemone's prompt template: chat-v1 (the model's chat template; for instruct models) or bare-v1 (JEV's own, no chat template)")
	fs.StringVar(&cfg.decisionsCal, "decisions-calibration", "", "calibration.json with per-kind temperatures for /v1/systemone (from goinfer-chat decisions-calibrate, fitted under the same template); none: every answer is uncalibrated")
	fs.StringVar(&cfg.thinking, "thinking", "template", "default thinking mode for a model whose chat template has a recognised thinking control (Qwen3, Qwen3.5, Gemma 4): template (what the checkpoint's own template renders — Qwen3.5-0.8B: off, Qwen3.5-9B: on), asis (the prompt bytes serve rendered before thinking was modelled: nothing written, the model decides), on, or off. A request overrides it with chat_template_kwargs.enable_thinking, reasoning_effort, or Anthropic's thinking. A model whose template control is not recognised ignores this.")
	fs.StringVar(&cfg.toolFormat, "tool-format", "auto", "how tools are put in the prompt for a model whose own chat template declares a tool form goinfer can render byte for byte — Qwen3.5 (its <function=…><parameter=…> XML) and Gemma 4's canonical template: auto (the default: each family's measured default — the model's own template form for Gemma 4, goinfer's own prompt for Qwen3.5), hermes (goinfer's own prompt: for Qwen a JSON call that tool_choice can constrain, for Gemma 4 its earlier order, text before a call) or template (the model's own chat template; tool_choice naming a function is then a 400, as for any form with no JSON wrapper). The Qwen reply parser reads both call forms whichever is chosen. No effect on a model with no such form.")
	fs.BoolVar(&cfg.lenientToolCalls, "lenient-tool-calls", false, "also read a tool call a model writes as ONE fenced JSON block at the end of its reply (```json {\"name\": ..., \"arguments\": {...}} ```) as a call, for the Qwen-style <tool_call> families (Qwen2.5-Coder under an agent writes its calls this way and otherwise makes none). Narrow on purpose: exactly one fence, last in the reply, tag json or none, only name/arguments/id keys, a supplied tool name, arguments that validate against that tool's schema. Off by default because a demonstration that ends on one valid call looks the same as a meant call, so a model asked to show an example can have it executed; turn it on for an agent whose client confirms before it acts")
	fs.StringVar(&cfg.reasoningBudget, "reasoning-budget", "auto", "ceiling on how long a thinking reply may think before serve forces the block closed so the reply can answer: auto (the default: thinking takes at most three quarters of the request's max_tokens), unlimited (no ceiling of serve's own), or N (cap every thinking reply at N tokens, still leaving room to answer). A request's own budget (thinking_token_budget, or Anthropic's thinking.budget_tokens) applies too, clamped so a quarter of max_tokens is left to answer. Not applied under -spec or -drafter, or to a model with no recognised thinking control.")
	fs.StringVar(&cfg.reasoningFmt, "reasoning-format", "deepseek", "how a reply's reasoning reaches the client: deepseek (content is the clean answer, reasoning goes in reasoning_content / Anthropic thinking blocks), deepseek-legacy (reasoning_content is filled and content keeps the raw <think> tags), or none (nothing is separated: the raw text is the content). A request may override it with reasoning_format.")
	fs.BoolVar(&cfg.noSelfTest, "no-selftest", false, "skip the startup self-tests: a kernel check against a reference that steps a CPU kernel tier down, or declines a GPU backend, when its output disagrees. On by default; skip it only if it misjudges a healthy machine (and tell us: `check --hardware` prints what it found).")
	fs.IntVar(&cfg.visionMaxPixels, "vision-max-pixels", 0, "GLM-OCR only: lower the image pixel budget to this many pixels (0 = the model's own ceiling, 4.82 MP; it is never raised). The CPU tower costs about 29 s at 1 MP, 92 s at 2 MP and 7 min at 4.8 MP on an M1 Pro.")
	fs.StringVar(&cfg.visionDevice, "vision-device", "auto", "where the vision tower runs: auto (default: on the --backend's GPU when this binary has a tower there, else the CPU) | cpu (the CPU whatever the backend; the language model keeps its own backend)")
	fs.StringVar(&cfg.visionQuant, "vision-quant", "f32", "vision encoder weight quant: f32 (default, bit-exact) | int8 (W8A8, cosine ~0.999) — int8 only speeds the compute-bound ViT prefill on AVX512-VNNI; on AVX2 it's a wash, so f32 is the default")
	fs.Var(&cfg.models, "model", "generative model: a .gguf/.giw file, an HF dir, or a reference that is fetched on first use — hf:<owner>/<repo>:<quant> (e.g. hf:Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF:q4_k_m), hf:<owner>/<repo>:safetensors (a safetensors checkpoint, fetched as a verified set) or demo:<tier>. A reference is sha256-verified and cached; a path is used as-is. Repeatable\n"+
		"as `name=path` to serve a model zoo from one process; requests route on the\n"+
		"OpenAI `model` field. Append comma-separated per-model overrides of the global\n"+
		"defaults below: `--model big=moe.giw,stream,weight-cache=16 --model fast=small.giw`\n"+
		"streams only the big MoE. Keys: quant,lora,kv,kv-quant,ctx,stream,weight-cache,embed-int4,head.\n"+
		"head=DIR attaches a trained decision head (autotrust's JEV layout): POST /v1/systemone on that model\n"+
		"answers with it (Route B), and an unmerged head's adapter is merged into the model at load. The model loads\n"+
		"at "+decoder.DecisionHeadQuant+" unless quant= or -quant chooses otherwise (the default for decision models, D6b).\n"+
		"A model directory that carries joint_head.safetensors is a Clef decision model (Route C): POST /v1/systemone on it\n"+
		"answers with its joint head, one backbone pass for every question, and it loads at the same default.\n"+
		"(Paths may not contain commas.)")
	fs.StringVar(&cfg.drafter, "drafter", "", "directory of a pretrained BLOCK drafter (z-lab DFlash) paired with --model: the drafter proposes a whole block of tokens per round and the target verifies them in ONE batched pass, measured 1.6-1.8x on code/math and ~0.96x on open chat (docs/spec/08). LOSSLESS — every emitted token is one the target's own argmax produced, so output is identical to plain greedy. Greedy only: a request with temperature, penalties or logit bias falls back to normal decoding automatically. Requires a resident GPU backend (--backend cuda); declines with a reason otherwise serve-only; goinfer-chat offers --spec ngram and --draft instead.")
	sf.showVersion = fs.Bool("version", false, "print version, the backends COMPILED INTO this binary, and the Go toolchain, then exit. `backends:` is the compiled-in truth — --backend accepts names this build cannot run and falls back to cpu")
	// The model-loading flags goinfer-chat shares — one registration, so the two binaries cannot drift.
	sf.lf = loadflags.Register(fs, loadflags.Serve)
	fs.BoolVar(&cfg.requireBE, "require-backend", false, "strict mode: exit non-zero at startup if a model did not resolve to the requested --backend's fast paths — no resident decode path, or a prefill that declined to the sequential per-token loop (e.g. native f32 on cuda, ~9x slower TTFT — N-35, docs/audit-2026-09-10.md: every quantized mode gets batched CUDA prefill, see -quant's own help above; only f32 falls back). Both fall back silently by design; a batch client should fail at second zero instead of discovering it under load. Under -backend auto it also refuses to start when auto passed over a GPU backend this binary has (no device answered) rather than run on the CPU; -backend cpu runs strict on the CPU")
	fs.StringVar(&cfg.kvQuant, "kv-quant", "", "DEPRECATED — use --kv, which now covers the CPU cache too. When given, overrides the CPU KV cache alone: f32 | i8")
	fs.Var(&cfg.adapters, "adapter", "compute-time LoRA adapter sharing a base model's resident weights: `serveName=baseName=dir`.\n"+
		"Repeatable. Unlike --lora (merged, one base per fine-tune), N adapters of one base cost ~base + N\n"+
		"low-rank deltas — request the fine-tune via the OpenAI `model` field. Base must be a safetensors\n"+
		"--model (dense, gated MLP; not MoE/gemma4/qwen3.5). Incompatible with --stream-weights.")
	fs.StringVar(&cfg.name, "served-model-name", "", "served id for a single unnamed --model (default: file/dir basename)")
	fs.IntVar(&cfg.kvSessions, "kv-sessions", 4, "number of conversations to keep prefilled in RAM for prompt-prefix KV reuse (0 disables); on Metal, CUDA and WebGPU, also how many GPU KV slots a resident model keeps (clamped by its memory guard). Metal keeps 2 GPU slots unless this flag is given: on unified memory each slot's KV is resident from the first token, about 112 MB per slot on the 1.5B and 235 MB on the 7B at the default context")
	fs.DurationVar(&cfg.kvIdleDemote, "kv-idle-demote", 0, "tiered KV: demote a warm session's KV to -session-dir once it's been idle this long, faulting it back on the next matching request (e.g. 10m; 0 = off). Lets a small-RAM box serve many intermittent chats. Needs -session-dir and -kv-sessions > 0")
	fs.IntVar(&cfg.kvDemotedMax, "kv-demoted-max", 64, "tiered KV: max demoted (on-disk) sessions to keep; older ones are dropped (only with -kv-idle-demote)")
	fs.IntVar(&cfg.maxQueue, "max-queue", 8, "per-model backpressure: max queued requests before 429 (0 = unbounded)")
	fs.IntVar(&cfg.maxConcurrent, "max-concurrent", 4, "generations one CPU model may run at once, each on its own session KV (capped by -kv-sessions; GPU-resident, weight-streaming and vision models always run one; 1 = serialized)")
	fs.Func("cpu-batch", "auto|on|off: whether concurrent CPU generations of one model join their decode tokens into one batched forward (MC3c step 2; replies are bit-identical either way). auto (the default) batches models with at least 2 GiB of dense weights, where it measured 2.25-2.41x the independent workers on a 7B (1.38-1.54x on an M1 Pro), and keeps smaller models on the workers", func(v string) error {
		switch v {
		case "auto":
			cfg.cpuBatch = decoder.CPUBatchAuto
		case "on":
			cfg.cpuBatch = decoder.CPUBatchOn
		case "off":
			cfg.cpuBatch = decoder.CPUBatchOff
		default:
			return fmt.Errorf("want auto, on or off")
		}
		return nil
	})
	fs.IntVar(&cfg.prefillChunk, "prefill-chunk", 512, "on a GPU-resident model running several generations at once (MC3), prefill a long prompt that arrives while others are decoding in chunks of this many tokens, one decode step between chunks, instead of stalling them for the whole prompt (replies are unchanged: Metal's prefill is chunk-invariant; 512 graded 2026-09-27: the decoders' longest stall 0.23x); 0 = off")
	fs.IntVar(&cfg.maxInflight, "max-inflight", 128, "global cap on concurrent inference requests, bounding the pre-queue stage (JSON+image decode, tokenization, template render, vision Forward) that runs before the per-model queue; a full cap returns 503 Retry-After (0 = unbounded)")
	fs.Int64Var(&cfg.maxBodyBytes, "max-body-bytes", 0, "cap on request body size in bytes; a larger body is rejected 413 before it is read. 0 = derive from the model's context window (a body that could never fit is rejected up front). The vision endpoints get at least 32 MiB on top for base64 image data")
	fs.DurationVar(&cfg.unloadDrainWait, "unload-drain-wait", 5*time.Second, "how long POST /admin/models/unload waits for in-flight requests to drain before returning 202 (native memory is freed as they finish either way; the model is unroutable immediately). ?wait=false returns 202 at once")
	fs.StringVar(&cfg.spec, "spec", "", "speculative decoding: \"\" (off) | ngram — lossless n-gram (prompt-lookup) drafting with adaptive depth. Wins on copy-heavy traffic (code edits / RAG / agent loops) on the CPU and Metal backends; output is identical (greedy bit-exact, sampled in-distribution incl. temperature/top-k/p/min-p + repetition penalties + logit bias). On greedy constrained/tool requests (response_format / tool grammar) it switches to grammar-fused drafting — the grammar's forced bytes are drafted for free, fused with the n-gram source. Auto-falls back to plain decode per-request when the sampler isn't yet supported on the spec path (e.g. constrained + temperature>0) goinfer-chat has the same --spec ngram, and additionally --draft (a separate small draft model).")
	fs.BoolVar(&cfg.specAdaptive, "spec-adaptive", false, "MC4 candidate (docs/tasks/task-concurrency-2026-09.md): with -spec ngram on a resident whose decode can batch (MC3), keep MC3's concurrency instead of forcing one generation at a time — a generation speculates only while it is alone, joining MC3's batched decode at a round boundary when others arrive, and resumes speculating after 8 consecutive rounds alone. Output is unaffected (the same lossless verify either way). No effect without -spec ngram, or on a resident MC3 cannot batch.")
	fs.StringVar(&cfg.embedPath, "embed-model", "", "embedding model for /v1/embeddings: a CodeRankEmbed/NomicBert HF dir (config.json + model.safetensors + tokenizer.json), a decoder-as-embedder .gguf, or a HuggingFace reference: hf:<owner>/<repo>:safetensors fetches a NomicBert encoder checkpoint (anything else is refused before the weights), hf:<owner>/<repo>:<quant> a GGUF")
	fs.StringVar(&cfg.embedQuant, "embed-quant", "f32", "embedding weight precision: f32 | q8")
	fs.StringVar(&cfg.embedName, "embed-served-model-name", "", "embedding model id reported by /v1/models (default: dir basename)")
	fs.StringVar(&cfg.embedResize, "embed-image-resize", "", "image resize before an image-embedding model's tower (EmbeddingGemma 2): bilinear | bicubic (the reference processor's torchvision antialiased bicubic). Default: the model's")
	return sf
}

func Main() {
	// Subcommand dispatch, before flag.Parse so `pull` gets its own flag set. Shares one
	// implementation with `goinfer-chat pull` (internal/pullcmd) rather than repeating the
	// flags and error messages — serve users need a model on disk for exactly the same
	// reason chat users do.
	if len(os.Args) > 1 && os.Args[1] == "pull" {
		os.Exit(pullcmd.Run(os.Args[2:]))
	}
	// `check` drives a RUNNING server through a harness's conversation. A client rather than
	// an embedded server, so it exercises whatever is actually serving with the flags its
	// operator chose — which is what a harness meets.
	if len(os.Args) > 1 && os.Args[1] == "check" {
		os.Exit(servecheck.Run(os.Args[2:], filepath.Base(os.Args[0])))
	}
	// K5 (docs/tasks/task-halt-2026-09.md): `status|ls|cancel|halt|resume` are a CLIENT talking to a
	// RUNNING server's admin socket (-admin-socket), not the server itself — same dispatch shape
	// as pull/check above, so the operator's command is one word instead of a raw curl-to-a-Unix-
	// socket incantation.
	if len(os.Args) > 1 && isAdminCLICmd(os.Args[1]) {
		os.Exit(runAdminCLI(os.Args[1], os.Args[2:], filepath.Base(os.Args[0])))
	}
	// A SKIMMABLE HELP HEADER, printed before the 39-flag dump.
	//
	// Cold-user run 2026-09-06, scenario B: "--help is 13,583 bytes / 39 flags / 100 lines, with
	// paragraph-length prose per flag containing commit SHAs and self-critique. Unusable as a quick
	// reference; I could not skim it for the flag I needed." That is not a style complaint — the
	// SAME tester then drove a 16 GB machine +7.8 GB into swap because they did not find
	// -stream-weights, whose help text names their exact model and their exact RAM. The flag was
	// there and the document was too long to find it in.
	//
	// The long text stays: every paragraph in it is a disclosure some measurement earned, and
	// deleting disclosures to shorten a page is how a trade-off stops being disclosed. This adds a
	// map ABOVE it rather than trimming it, so skimming and reading are both possible.
	flag.Usage = func() {
		out := flag.CommandLine.Output()
		// The examples use the name the binary was actually INVOKED as, not a hardcoded
		// "goinfer-serve". `go install .../cmd/serve@latest` drops a binary called `serve`, which
		// the same cold run flagged ("Not goinfer-serve. On $PATH that is a collision waiting to
		// happen"), so a help page that shows a name the reader does not have is one more thing to
		// translate.
		self := filepath.Base(os.Args[0])
		fmt.Fprintf(out, `%[1]s — OpenAI/Anthropic-compatible local inference server.

  %[1]s --model <file.gguf|dir|hf:owner/repo:quant>     serve one model
  %[1]s --model m.gguf --web                            + browser UI at /
  %[1]s check                                           drive a RUNNING server, per-feature verdicts
  %[1]s pull <ref>                                      fetch a model, sha256-verified
  %[1]s --version                                       version + the backends COMPILED IN

%[3]s

The flags people actually reach for:

  --model      what to serve; repeatable as name=path to serve several at once
  --backend    auto (default: the first of cuda, metal that finds its device, else cpu) | cpu | cuda | metal | webgpu — --version says which this binary has
  --quant      int4 (default) | int8int8 | int4mix | f32 — see docs/quantization.md
  --ctx        KV capacity in positions
  --addr       listen address (loopback by default)
  --api-key    required to bind anywhere but loopback

  --stream-weights   RUN A MODEL BIGGER THAN YOUR RAM. Pages weights from disk instead of
                     holding them resident; the one flag to know before loading something large.

All %[2]d flags, with the trade-offs each one makes, follow.

`, self, countFlags(), pull.CacheHelp())
		flag.PrintDefaults()
	}

	// --version answers "what is in this binary" WITHOUT a model, which is the question the
	// cold run could not ask (R2). Handled here rather than only as a parsed flag so it works
	// on a binary whose other required flags are absent — and registered below as well, so it
	// appears in --help.
	if len(os.Args) > 1 && isVersionArg(os.Args[1]) {
		fmt.Print(versionReport(filepath.Base(os.Args[0])))
		return
	}
	sf := registerFlags(flag.CommandLine)
	flag.Parse()
	markGivenFlags(flag.CommandLine, &sf.cfg)
	cfg, addr, apiKey, tlsCert, tlsKey, showVersion, lf := sf.cfg, sf.addr, sf.apiKey, sf.tlsCert, sf.tlsKey, sf.showVersion, sf.lf
	if *showVersion {
		fmt.Print(versionReport(filepath.Base(os.Args[0])))
		return
	}
	if cfg.noSelfTest {
		decoder.SkipSelfTests() // before the first Load, which is what runs the cpu self-test
	}
	// R6's other half (docs/measurements/cold-user-2026-09-06-nobara-pc.md): an unrecognized
	// subcommand/positional falls through silently otherwise. Every argument here is a --flag;
	// anything flag.Parse left in flag.Args() is a typo, not a feature.
	if args := flag.Args(); len(args) > 0 {
		fmt.Fprintf(os.Stderr, "%s: unrecognized argument %q\n\nknown subcommands: pull <ref>, check, --version. Or pass --model <file.gguf|dir|hf:owner/repo:quant>.\n",
			filepath.Base(os.Args[0]), args[0])
		os.Exit(2)
	}
	cfg.load = *lf
	if err := cfg.load.Validate(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(2)
	}
	if l := cfg.load.BackendLine(); l != "" {
		fmt.Fprintln(os.Stderr, l)
	}
	if err := requireAutoBackend(cfg); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(2)
	}
	// -web counts as a reason to start with no model: fetching one is the whole point of
	// the UI's Models tab, and requiring a model in order to go and get a model is a
	// bootstrap the user cannot satisfy.
	if len(cfg.models) == 0 && cfg.embedPath == "" && !cfg.allowAdmin && !cfg.web && cfg.adminSocket == "" {
		fmt.Fprintln(os.Stderr, "error: need at least one of --model, --embed-model, --web, --allow-admin, or --admin-socket")
		flag.Usage()
		os.Exit(2)
	}
	// Resolve the optional shared secret (flag wins over env). -allow-admin exposes an
	// arbitrary-path model load + sidecar write, so it must not run unauthenticated (B-14).
	authKey := *apiKey
	if authKey == "" {
		authKey = os.Getenv("GOINFER_API_KEY")
	}
	if cfg.allowAdmin && authKey == "" {
		fmt.Fprintln(os.Stderr, "error: -allow-admin requires -api-key (or $GOINFER_API_KEY) — admin load/unload must be authenticated")
		os.Exit(2)
	}
	// Mirrors the -allow-admin check above: the -addr help text itself invites
	// 0.0.0.0 exposure with "set -api-key when you do", but nothing previously
	// enforced the second half — a non-loopback bind with no key started up fully
	// open to the network. Loopback stays key-free by default (no auth friction
	// for the common single-user desktop case).
	if !addrIsLoopback(*addr) && authKey == "" {
		fmt.Fprintf(os.Stderr, "error: -addr %s is not loopback-only — requires -api-key (or $GOINFER_API_KEY), or bind to 127.0.0.1 instead\n", *addr)
		os.Exit(2)
	}
	if (*tlsCert == "") != (*tlsKey == "") {
		fmt.Fprintln(os.Stderr, "error: -tls-cert and -tls-key must be set together")
		os.Exit(2)
	}
	if err := sessionDirOK(cfg.sessionDir); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(2)
	}
	if cfg.spec != "" && cfg.spec != "ngram" {
		fmt.Fprintf(os.Stderr, "error: -spec must be \"\" or \"ngram\" (got %q)\n", cfg.spec)
		os.Exit(2)
	}
	if cfg.kvIdleDemote > 0 && (cfg.sessionDir == "" || cfg.kvSessions <= 0) {
		fmt.Fprintln(os.Stderr, "error: -kv-idle-demote needs -session-dir and -kv-sessions > 0")
		os.Exit(2)
	}
	if _, ok := chat.ParseThinkMode(cfg.thinking); !ok {
		fmt.Fprintf(os.Stderr, "error: -thinking %q: want asis, template, on or off\n", cfg.thinking)
		os.Exit(2)
	}
	if _, ok := chat.ParseToolFormat(cfg.toolFormat); !ok {
		fmt.Fprintf(os.Stderr, "error: -tool-format %q: want hermes or template\n", cfg.toolFormat)
		os.Exit(2)
	}
	if _, err := parseBudgetFlag(cfg.reasoningBudget); err != nil {
		fmt.Fprintf(os.Stderr, "error: -reasoning-budget %q: %v\n", cfg.reasoningBudget, err)
		os.Exit(2)
	}
	if _, ok := parseReasoningFormat(cfg.reasoningFmt); !ok {
		fmt.Fprintf(os.Stderr, "error: -reasoning-format %q: want deepseek, deepseek-legacy or none\n", cfg.reasoningFmt)
		os.Exit(2)
	}
	if cfg.decisionsTemplate != decide.TemplateChat && cfg.decisionsTemplate != decide.TemplateBare {
		fmt.Fprintf(os.Stderr, "error: -decisions-template %q: want %s or %s\n", cfg.decisionsTemplate, decide.TemplateChat, decide.TemplateBare)
		os.Exit(2)
	}
	if cfg.decisionsCal != "" {
		// Fail at startup, not on the first /v1/systemone request: a bad file, or one fitted under the other template.
		cal, err := decide.LoadCalibration(cfg.decisionsCal)
		if err == nil && cal.Template != "" && cal.Template != cfg.decisionsTemplate {
			err = fmt.Errorf("fitted under template %q, not -decisions-template %q", cal.Template, cfg.decisionsTemplate)
		}
		if err != nil {
			fmt.Fprintf(os.Stderr, "error: -decisions-calibration %s: %v\n", cfg.decisionsCal, err)
			os.Exit(2)
		}
	}

	srv, err := newServer(cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
	if cfg.sessionDir != "" && cfg.kvSessions > 0 {
		for _, lm := range srv.modelList() {
			sub := sessionSubdir(cfg.sessionDir, lm.fp)
			lm.sessions.load(sub)
			if cfg.kvIdleDemote > 0 {
				lm.sessions.enableTiering(sub, cfg.kvIdleDemote, cfg.kvDemotedMax)
			}
		}
	}

	// Every POST body is size-bounded (M3): the chat/messages endpoints carry
	// base64 image_url data, so they get the larger vision cap; the rest a few MB.
	// auth wraps a handler with the optional shared-secret check (no-op when authKey
	// is ""); every route below goes through it so a set key protects the whole surface.
	auth := func(h http.HandlerFunc) http.HandlerFunc { return requireAuth(authKey, h) }
	// inflight is the global concurrency cap over the inference POST handlers (the pre-queue
	// stage), shared across all of them (audit M-01). GET/health and admin stay uncapped so an
	// operator's health probe is always answered and never consumes a slot. Order: auth outermost
	// (reject bad auth without taking a slot), then the inflight gate, then the body cap.
	var inflight chan struct{}
	if cfg.maxInflight > 0 {
		inflight = make(chan struct{}, cfg.maxInflight)
	}
	inf := func(h http.HandlerFunc) http.HandlerFunc { return limitInflight(inflight, h) }
	// Resolve the request-body caps (G1d). The largest servable text prompt is ctx tokens ×
	// the longest token's bytes; ×4 covers JSON structure/escaping. Derived per the largest
	// served model's context window, floored at the historical constants so a small-context
	// model keeps a usable body budget (the per-request tokenization guard, not this cap,
	// protects it), and overridable with -max-body-bytes. Reported on the startup line.
	textCap, visionCap, embedCap, fileCap := srv.resolveBodyCaps(cfg.maxBodyBytes)
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/models", auth(srv.handleModels))
	// Operator surface for the resolved compute paths — same fields as the /v1/models vendor
	// extension, on a payload with no OpenAI-schema contract to break. See handleHealth.
	mux.HandleFunc("GET /health", auth(srv.handleHealth))
	// K2 (docs/tasks/task-halt-2026-09.md): halt is checked AFTER auth (a bad key is still rejected
	// during a halt) and BEFORE inf (a halt must not wait for an inflight slot — "a halt that
	// has to wait for a slot is not a halt", the doc's own words). /admin/* and /health are
	// deliberately NOT wrapped in this — an operator must always be able to resume/check status.
	// Registered whether or not a model is loaded at startup. A server started with only --web,
	// --allow-admin or --admin-socket loads its model later (the web UI's Models tab, /admin/models/load);
	// the mux is built once, so these routes registered only when a model existed at startup left that
	// server with no /v1/chat/completions and no /v1/jobs for its whole life, and the web UI's own chat
	// got a 404. Every handler resolves its model through resolveAndLock, which answers an unknown or
	// absent model with the OpenAI-shaped 404 "model not found (served: …)".
	// -log-requests wraps the four generation routes OUTERMOST, so a request the auth, halt or queue gates turned away is logged with its status too.
	var reqLogOut io.Writer
	if cfg.logRequests {
		reqLogOut = os.Stderr
	}
	rl := func(h http.HandlerFunc) http.HandlerFunc { return logRequests(reqLogOut, h) }
	mux.HandleFunc("POST /v1/chat/completions", rl(auth(srv.haltGate(inf(maxBytes(visionCap, srv.handleChat))))))
	mux.HandleFunc("POST /v1/completions", rl(auth(srv.haltGate(inf(maxBytes(textCap, srv.handleCompletions))))))
	mux.HandleFunc("POST /v1/responses", rl(auth(srv.haltGate(inf(maxBytes(textCap, srv.handleResponses))))))
	mux.HandleFunc("POST /v1/messages", rl(auth(srv.haltGate(inf(maxBytes(visionCap, srv.handleMessages))))))
	mux.HandleFunc("POST /v1/messages/count_tokens", auth(srv.haltGate(inf(maxBytes(textCap, srv.handleCountTokens)))))
	// Decisions (D5, docs/tasks/task-constrained-confidence.md): TypeSafe's POST /v1/systemone wire shape, served by
	// label scoring (internal/decide), so jevx and the TypeSafe SDKs work against goinfer through their base-URL override.
	mux.HandleFunc("POST /v1/systemone", auth(srv.haltGate(inf(maxBytes(textCap, srv.handleSystemOne)))))
	// J3 (task-work-queue-2026-09.md): submitting a job starts new admission, so it gets the
	// same haltGate/inf/maxBytes stack as every other POST above. Polling state (GET), reading
	// the event stream (GET .../events), and cancelling (DELETE) are NOT new inference work —
	// haltGate'ing them would 503 a client just trying to learn that its job was halted, and
	// inf's inflight cap exists to bound pre-queue JSON/image decode + tokenization, none of
	// which these three do — so they get auth only.
	mux.HandleFunc("POST /v1/jobs", auth(srv.haltGate(inf(maxBytes(textCap, srv.handleCreateJob)))))
	mux.HandleFunc("GET /v1/jobs/{id}", auth(srv.handleGetJob))
	mux.HandleFunc("GET /v1/jobs/{id}/events", auth(srv.handleJobEvents))
	mux.HandleFunc("DELETE /v1/jobs/{id}", auth(srv.handleCancelJob))
	// J4 (task-work-queue-2026-09.md): the two batch APIs, over the same job store. POST
	// /v1/files is an upload, not inference — it carries no generation work by itself, so it
	// follows /web/models/pull's own precedent (main.go, below) rather than /v1/jobs': auth +
	// maxBytes only, no haltGate/inf (those bound decode concurrency/backpressure, not upload
	// I/O). POST /v1/batches and POST /v1/messages/batches DO queue real generation work (one
	// job per line, same pipeline as /v1/jobs), so they get the full stack. Every GET and every
	// .../cancel gets auth only, same reasoning as /v1/jobs' own GET/DELETE routes above.
	mux.HandleFunc("POST /v1/files", auth(maxBytes(fileCap, srv.handleCreateFile)))
	mux.HandleFunc("GET /v1/files/{id}", auth(srv.handleGetFile))
	mux.HandleFunc("GET /v1/files/{id}/content", auth(srv.handleGetFileContent))
	mux.HandleFunc("POST /v1/batches", auth(srv.haltGate(inf(maxBytes(textCap, srv.handleCreateBatch)))))
	mux.HandleFunc("GET /v1/batches/{id}", auth(srv.handleGetBatch))
	mux.HandleFunc("POST /v1/batches/{id}/cancel", auth(srv.handleCancelBatch))
	mux.HandleFunc("POST /v1/messages/batches", auth(srv.haltGate(inf(maxBytes(textCap, srv.handleCreateMessageBatch)))))
	mux.HandleFunc("GET /v1/messages/batches/{id}", auth(srv.handleGetMessageBatch))
	mux.HandleFunc("GET /v1/messages/batches/{id}/results", auth(srv.handleGetMessageBatchResults))
	mux.HandleFunc("POST /v1/messages/batches/{id}/cancel", auth(srv.handleCancelMessageBatch))
	// Registered unconditionally (G7): with no embedding model, handleEmbeddings returns a JSON
	// error naming -embed-model rather than a bare 404, so an SDK sees "unconfigured" not "wrong URL".
	mux.HandleFunc("POST /v1/embeddings", auth(srv.haltGate(inf(maxBytes(embedCap, srv.handleEmbeddings,
		// The per-input / per-batch bounds are per-DIMENSION and multiply out past any body cap, so
		// name all three: a client within both per-dimension limits can still exceed the total.
		fmt.Sprintf("this route also limits each request to %d inputs of at most %d bytes each; "+
			"the body cap bounds their total", maxEmbedInputs, maxEmbedInputBytes))))))
	// /admin/* (load/unload, K1's cancel-by-id, K2's halt/resume, K5's status) lives on EITHER
	// the TCP listener (gated by auth+ -allow-admin, as always) OR the admin socket (K5,
	// docs/tasks/task-halt-2026-09.md) when -admin-socket is set — never both, so a request against
	// the surface that was deliberately not chosen 404s instead of merely being refused (a 403
	// would confirm the surface exists; a 404 does not). See admin_socket.go for the socket side.
	if cfg.adminSocket == "" {
		registerAdminRoutes(mux, srv, textCap, func(h http.HandlerFunc) http.HandlerFunc { return auth(srv.requireAdmin(h)) })
	}
	if cfg.web {
		// "GET /{$}" matches the root path EXACTLY. A bare "GET /" would be a catch-all and
		// would turn every unknown GET into the UI page instead of a 404, which is worse than
		// unhelpful for an API server — a typo'd route would render HTML to an SDK.
		//
		// UNAUTHENTICATED on purpose (V-02, docs/review-2026-09-04.md): the page embeds no
		// secrets (handleWebUI's own comment), but a browser's plain navigation sends no
		// Authorization header, and the page is the ONLY place a user could type the key in —
		// its own JS holds it for the fetch() calls to /web/models/*. Wrapping this route in
		// auth() made that impossible whenever -api-key was set (required off loopback): the
		// page needed the key to load, and there was nowhere to enter the key without the page.
		// auth stays on the two routes below, which actually act (list a repo, pull a model).
		mux.HandleFunc("GET /{$}", srv.handleWebUI)
		// The page's own CSS and JS (task-web-ui-2026-09.md §6.1). Unauthenticated for the same
		// V-02 reason as the page: it cannot load, and so the key cannot be entered, without them.
		// {file} is one path segment, so "GET /ui/{file}" cannot become a catch-all.
		mux.HandleFunc("GET /ui/{file}", srv.handleWebAsset)
		// sameOrigin (V-20, docs/review-2026-09-04.md): on the key-free loopback default,
		// auth() alone is a no-op, and these two routes act — pull triggers a caller-named
		// multi-gigabyte download. See sameOrigin's own doc comment in webui.go.
		mux.HandleFunc("POST /web/models/list", sameOrigin(auth(maxBytes(textCap, srv.handleWebList))))
		// Live search-as-you-type suggestions over the repo box (pull.Search, GGUF only today):
		// read-only, same origin/auth stack as list/list-adjacent routes above.
		mux.HandleFunc("POST /web/models/search", sameOrigin(auth(maxBytes(textCap, srv.handleWebSearch))))
		// Not wrapped in inf(): the inflight gate bounds INFERENCE, and a download that runs
		// for minutes must not occupy one of those slots. handleWebPull is single-flighted on
		// its own (pullState), which is the bound that actually fits it.
		mux.HandleFunc("POST /web/models/pull", sameOrigin(auth(maxBytes(textCap, srv.handleWebPull))))
		// W5 (task-web-ui-2026-09.md): load what the pull just downloaded. NOT the admin load —
		// that takes any caller-named path and stays behind -allow-admin. This one is confined to
		// regular .gguf files under the pull cache (webLoadPath), and gets the same stack as pull:
		// it is at least as heavy an action. Not in inf() either, for the pull's reason.
		mux.HandleFunc("POST /web/models/load", sameOrigin(auth(maxBytes(textCap, srv.handleWebLoad))))
		// W32 (task-web-ui-2026-09.md): unload a currently-loaded model. Needs no path policy the way
		// load does — the only names it can act on are ones GET /v1/models already publishes — so it
		// reuses unloadByName directly rather than gating a new admin surface. Same stack as load.
		mux.HandleFunc("POST /web/models/unload", sameOrigin(auth(maxBytes(textCap, srv.handleWebUnload))))
		// P8 (task-checkpoint-fetch-2026-09.md): what the pull cache holds on disk, for the page's "On disk" card. Read-only,
		// and it names only paths under the pull cache, the ones the load route above would accept anyway.
		mux.HandleFunc("GET /web/models/cache", sameOrigin(auth(srv.handleWebCache)))
	}

	// K5 (docs/tasks/task-halt-2026-09.md): the admin socket. closeAdminSock is a no-op when
	// -admin-socket is unset, so the shutdown handler below can call it unconditionally.
	closeAdminSock := func() {}
	if cfg.adminSocket != "" {
		var err error
		closeAdminSock, err = startAdminSocket(srv, cfg.adminSocket, textCap)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error: -admin-socket %s: %v\n", cfg.adminSocket, err)
			os.Exit(1)
		}
		fmt.Fprintf(os.Stderr, "admin socket: %s (mode 0600, no api-key — file permissions are the auth; /admin/* is NOT registered on the TCP listener while this is set)\n", cfg.adminSocket)
	}

	// ReadHeaderTimeout + ReadTimeout + IdleTimeout bound slow-header (slowloris), slow-body
	// dribble, and idle keep-alive connections. ReadTimeout is the whole-request read deadline
	// (60s: generous for a 32 MiB vision body on a slow link) — before it, ReadHeaderTimeout
	// bounded only the headers, so a client sending the body one byte per minute pinned a
	// goroutine indefinitely (audit M-01). It only bounds the request READ; the SSE response is a
	// write, so a long stream is unaffected. WriteTimeout stays 0: SSE responses are long-lived
	// and a write deadline would truncate a legitimate stream (M3).
	// srvCtx is the server-lifetime context. BaseContext makes every request's r.Context() a child of
	// it, so cancelling srvCtx at shutdown cancels every in-flight generation (drive derives its
	// context from r.Context()). Without this, httpSrv.Shutdown waits for a long streaming generation
	// but never cancels it, so it runs past the 30s timeout — the checkpoint loop below no longer
	// deadlocks on that specific generation's lm.sessMu (J1, task-work-queue-2026-09.md, split
	// sessMu out from the admission turn it used to share: sessMu is now held only briefly, around
	// sessions.acquire, not for the whole generation), but Shutdown itself still waits for the
	// handler to return, so cancelling the generation is still what bounds the overall shutdown
	// (audit C-22); tryLockUntil's own deadline is the remaining belt-and-braces bound on sessMu
	// specifically, for whatever brief window a generation is actually inside sessions.acquire.
	srvCtx, srvCancel := context.WithCancel(context.Background())
	defer srvCancel()
	httpSrv := &http.Server{
		Addr:              *addr,
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       60 * time.Second,
		IdleTimeout:       120 * time.Second,
		BaseContext:       func(net.Listener) context.Context { return srvCtx },
	}

	// Tiered KV: a background ticker demotes idle sessions to disk. It takes sessMu, the
	// lock the request path takes around acquire and checkin, and skips any session a
	// generation has checked out, so it never touches a running generation's session;
	// stopDemote halts it before the shutdown checkpoint runs.
	stopDemote := make(chan struct{})
	if cfg.kvIdleDemote > 0 {
		go demoteLoop(srv, cfg.kvIdleDemote, stopDemote)
	}
	// K2's -halt-file poller stop channel; declared here (not beside its goroutine start below)
	// so the shutdown handler just below can close it alongside stopDemote.
	stopHaltPoll := make(chan struct{})

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
		// A SECOND signal during the drain force-exits instead of being swallowed by the buffered
		// channel — so Ctrl-C twice always kills the server, not only SIGKILL (audit C-22).
		go func() {
			<-sig
			fmt.Fprintln(os.Stderr, "second signal — forcing exit")
			os.Exit(1)
		}()
		close(stopDemote)   // stop demoting before we checkpoint
		close(stopHaltPoll) // stop polling -halt-file; nothing left to react to it after shutdown
		closeAdminSock()    // close + unlink the admin socket (K5), if one was started
		srvCancel()         // cancel in-flight generations (via BaseContext) so they release lm.mu
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		_ = httpSrv.Shutdown(ctx)
		// MC3c step 2: what the CPU batcher did over this server's life — the record that decode tokens actually ran
		// in batched steps, and how wide (docs/tasks/task-concurrency-2026-09.md).
		for _, lm := range srv.modelList() {
			if lm.model != nil && lm.model.CPUBatchActive() {
				st := lm.model.CPUBatchStats()
				fmt.Fprintf(os.Stderr, "cpu batch %q: %d runs, %d batched steps (%d tokens), %d solo tokens, %d straggler runs, steps by size %v\n",
					lm.name, st.Runs, st.Steps, st.StepTokens, st.SoloTokens, st.StragglerRuns, st.StepSizes[:max(2, lm.concurrent+1)])
			}
			if lm.model != nil && lm.model.ResidentConcurrency() > 1 {
				// MC3 (Metal, CUDA): the resident batcher's counters, the same shape.
				st := lm.model.ResidentBatchStats()
				fmt.Fprintf(os.Stderr, "resident batch %q: %d runs, %d batched steps (%d tokens), %d solo tokens, %d straggler runs, steps by size %v\n",
					lm.name, st.Runs, st.Steps, st.StepTokens, st.SoloTokens, st.StragglerRuns, st.StepSizes[:max(2, lm.concurrent+1)])
				// The tokens a lone generation ran with the resident held (the chained decode, holdSolo): the record that
				// the greedy and sampled chains reach serve.
				fmt.Fprintf(os.Stderr, "resident batch %q held: %d tokens in %d holds\n", lm.name, st.HeldTokens, st.Holds)
				fmt.Fprintf(os.Stderr, "resident batch %q time: runs %.3f s, exclusive %.3f s (prefill %d passes %.3f s, bookkeeping %.3f s)\n",
					lm.name, float64(st.RunNs)/1e9, float64(st.ExclusiveNs)/1e9, st.PrefillPasses, float64(st.PrefillNs)/1e9,
					float64(st.ExclusiveNs-st.PrefillNs)/1e9)
			}
		}
		if cfg.sessionDir != "" && cfg.kvSessions > 0 {
			deadline := time.Now().Add(5 * time.Second)
			for _, lm := range srv.modelList() {
				// TryLock with a deadline: srvCancel above should have freed a generation still
				// inside sessions.acquire (a generation mid-forward, not yet at a ctx check, could
				// briefly still hold lm.sessMu) — skip its checkpoint rather than deadlock the
				// whole shutdown (audit C-22).
				if !tryLockUntil(&lm.sessMu, deadline) {
					fmt.Fprintf(os.Stderr, "shutdown: model %q still busy — skipping its session checkpoint\n", lm.name)
					continue
				}
				_ = lm.sessions.save(sessionSubdir(cfg.sessionDir, lm.fp))
				lm.sessions.removeColdFiles() // the cold tier is in-process; clear its scratch
				lm.sessMu.Unlock()
			}
		}
	}()

	// K2 (docs/tasks/task-halt-2026-09.md): SIGUSR1 halts, SIGUSR2 resumes — a supervisor can pull
	// this switch without opening a socket or an HTTP client. Separate from the SIGINT/SIGTERM
	// channel above: those are one-shot (shutdown then exit), these repeat for the life of the
	// process, so they get their own Notify and a loop rather than a single <-sig receive.
	// Platform-specific (haltsignal_unix.go / haltsignal_windows.go): SIGUSR1/SIGUSR2 are
	// undefined identifiers on Windows, not just signals it never raises.
	startHaltSignalLoop(srv)

	// K2: -halt-file. Polled in its own goroutine; stopHaltPoll (declared above, closed
	// alongside stopDemote at shutdown) is best-effort background work with no result main()
	// waits on.
	if cfg.haltFile != "" {
		go haltFilePoller(srv, cfg.haltFile, stopHaltPoll)
	}

	useTLS := *tlsCert != ""
	scheme := "http"
	if useTLS {
		scheme = "https"
	}
	fmt.Fprintf(os.Stderr, "goinfer serving on %s://%s [%s]\n", scheme, *addr, srv.endpointSummary())
	for _, line := range serverBanner(srv, cfg) {
		fmt.Fprintf(os.Stderr, "  %s\n", line)
	}
	// No -api-key: every route above is unauthenticated. Binding to loopback keeps
	// other MACHINES out, but not other TABS — any page open in a browser on this
	// machine can still fetch()/POST to it while it's running (the request is sent
	// regardless of CORS; CORS only gates whether the page can READ the response),
	// the classic "localhost drive-by" pattern. This is a deliberate product default
	// (no auth friction for the common single-user desktop case), not an oversight —
	// but it needs to be visible, not just documented, since most users never read
	// -h before running the one-liner from the README.
	if authKey == "" {
		fmt.Fprintln(os.Stderr, "warning: no -api-key set — any web page open in your browser can silently send requests to this API while it's running. Set -api-key (or $GOINFER_API_KEY) to require authentication.")
	}
	// A non-loopback bind with no TLS sends -api-key (a bearer token, every request)
	// and every prompt/completion in cleartext to anyone on the network path between
	// client and server — the -addr help text itself invites 0.0.0.0 exposure, so
	// this needs to be loud, not just in -h. Loopback is exempt: traffic never
	// leaves the machine, so there is no network path to sniff.
	if !useTLS && !addrIsLoopback(*addr) {
		fmt.Fprintln(os.Stderr, "warning: serving non-loopback with no TLS — -api-key and every prompt/completion travel in cleartext on the network. Set -tls-cert/-tls-key, or put a TLS-terminating reverse proxy in front.")
	}
	capSrc := "derived from context window"
	if cfg.maxBodyBytes > 0 {
		capSrc = "-max-body-bytes"
	}
	fmt.Fprintf(os.Stderr, "request body cap: %s (text) / %s (vision) / %s (embeddings) / %s (batch files) [%s]\n", humanBytes(textCap), humanBytes(visionCap), humanBytes(embedCap), humanBytes(fileCap), capSrc)
	var listenErr error
	if useTLS {
		listenErr = httpSrv.ListenAndServeTLS(*tlsCert, *tlsKey)
	} else {
		listenErr = httpSrv.ListenAndServe()
	}
	if listenErr != nil && !errors.Is(listenErr, http.ErrServerClosed) {
		fmt.Fprintf(os.Stderr, "server: %v\n", listenErr)
		os.Exit(1)
	}
	<-done // let the shutdown handler finish checkpointing before we exit
}

// newServer loads the configured model(s): a decoder (with tokenizer, chat
// template, and KV sessions) and/or an encoder (with its tokenizer for token
// counting). At least one must be configured.
func newServer(cfg config) (*server, error) {
	if cfg.unloadDrainWait <= 0 {
		cfg.unloadDrainWait = 5 * time.Second // floor; the flag defaults here too. ?wait=false is the per-request path to an immediate 202.
	}
	// 256 matches responseStore's own bound (newResponseStore(256), just below) — no dedicated
	// -job-cap flag yet; a job past this many jobs old is only evicted once terminal (evictLocked).
	jobs, err := newJobStore(cfg.jobDir, 256)
	if err != nil {
		return nil, fmt.Errorf("-job-dir %q: %w", cfg.jobDir, err)
	}
	s := &server{
		models:    map[string]*loadedModel{},
		liveness:  map[*decoder.Model]*modelLiveness{},
		draining:  map[string]struct{}{},
		cfg:       cfg,
		responses: newResponseStore(256),
		gens:      newGenerationRegistry(),
		jobs:      jobs,
		// 256 matches jobs'/responses' own bound above — no dedicated -file-cap/-batch-cap flag
		// yet, same reasoning as jobs' own comment (J3): a size past this many old is only evicted
		// once terminal.
		files:   newFileStore(256),
		batches: newBatchStore(256),
	}
	for _, spec := range cfg.models {
		// An `hf:`/`demo:` spec is fetched (or found in the cache) inside loadDecoder
		// (modelload.Resolve), before anything else, so the served name derives from the real
		// filename. A plain path is returned untouched, so no existing --model changes meaning —
		// the property that lets a reference form be added to a Hard-tier flag.
		// Startup load: a transparent .gguf→.giw transcode here isn't request-scoped, so
		// context.Background() (a Ctrl-C during startup already ends the process). The admin
		// load path below passes the request context so a disconnect cancels it (M-21).
		lm, err := loadDecoder(context.Background(), spec, cfg)
		if err != nil {
			return nil, err
		}
		if _, dup := s.models[lm.name]; dup {
			return nil, fmt.Errorf("duplicate served model name %q (use --model name=path to disambiguate)", lm.name)
		}
		s.models[lm.name] = lm
		s.retainLocked(lm.model) // liveness refs (startup is single-threaded; no lock contention)
	}
	if err := s.loadAdapters(cfg); err != nil {
		return nil, err
	}
	if cfg.embedPath != "" {
		// An hf: reference is fetched (or found in the cache) first: hf:<repo>:safetensors as an encoder checkpoint, checked
		// against what loadEncoder can load before any weight byte; hf:<repo>:<quant> as a GGUF, the decoder-as-embedder.
		p, err := resolveEmbedModel(context.Background(), cfg.embedPath)
		if err != nil {
			return nil, fmt.Errorf("-embed-model %s: %w", cfg.embedPath, err)
		}
		cfg.embedPath = p
		if err := s.loadEncoder(cfg); err != nil {
			return nil, err
		}
	}
	if err := s.loadVisionTower(cfg); err != nil {
		return nil, err
	}
	// MC3c: decided last, once adapters and vision towers are attached — a vision model runs one generation at a time.
	for _, lm := range s.models {
		if line := lm.setConcurrency(cfg); line != "" {
			fmt.Fprintf(os.Stderr, "%q %s\n", lm.name, line) // decided only now, so not part of its load banner
		}
	}
	// S3 (docs/tasks/task-never-swap-2026-09.md): armed last, after every startup load that could
	// itself grow swap has already finished — the guard's baseline should be "steady state after
	// startup," not a reading taken mid-load that then reads every startup byte as its own delta.
	s.startSwapGuard()
	return s, nil
}

// loadAdapters registers each --adapter (#7) against its base --model: it shares
// the base's resident decoder.Model (the RAM win — only the low-rank A/B bytes are
// new) but gets its own served name, KV-session LRU, decode mutex, and queue. A
// request routes to the fine-tune via the OpenAI `model` field; the per-LRU
// adapter binding makes each session project through it. The shared all-resident
// weights are read-only during a forward (per-stream scratch + KV), so the base
// and its adapters run as independent decode workers safely — hence --stream-weights
// (mutable per-layer paging on the shared model) is rejected.
func (s *server) loadAdapters(cfg config) error {
	for _, spec := range cfg.adapters {
		if cfg.load.StreamWeights {
			return fmt.Errorf("--adapter %q: compute-time LoRA is incompatible with --stream-weights", spec.name)
		}
		base, ok := s.models[spec.base]
		if !ok {
			return fmt.Errorf("--adapter %q: base model %q not loaded (declare it with --model %s=…)", spec.name, spec.base, spec.base)
		}
		if base.adapter != "" {
			return fmt.Errorf("--adapter %q: base %q is itself an adapter — attach to the underlying --model", spec.name, spec.base)
		}
		if _, dup := s.models[spec.name]; dup {
			return fmt.Errorf("--adapter %q: name collides with a loaded model", spec.name)
		}
		t0 := time.Now()
		if err := base.model.LoadAdapter(spec.name, spec.dir); err != nil {
			return fmt.Errorf("--adapter %q: %w", spec.name, err)
		}
		fp := base.fp + "+adapter:" + spec.name // distinct snapshot namespace from the base + sibling adapters
		lm := &loadedModel{
			tk: base.tk, model: base.model, tmpl: base.tmpl, stopIDs: base.stopIDs,
			eosIDs: base.eosIDs, vocab: base.vocab, name: spec.name, fp: fp, adapter: spec.name,
			spec:         cfg.spec == "ngram",
			specAdaptive: cfg.specAdaptive,
			sessions:     newSessionLRU(base.model, cfg.kvSessions, 0, fp),
		}
		lm.sessions.adapter = spec.name
		if cfg.maxQueue > 0 {
			lm.queue = make(chan struct{}, 1+cfg.maxQueue)
		}
		s.models[spec.name] = lm
		s.retainLocked(lm.model) // adapter shares base.model → same liveness entry, refs++
		fmt.Fprintf(os.Stderr, "loaded adapter %q on base %q in %s\n", spec.name, spec.base, time.Since(t0).Round(time.Millisecond))
	}
	return nil
}

// loadVisionTower attaches a vision tower to the (single) loaded model, making it
// vision-capable (serve then accepts image content parts). Per-family, not SigLIP-only
// (N-35, docs/audit-2026-09-10.md): Gemma 3 gets a SigLIP encoder + projector, Qwen2.5-VL
// its own ViT, Gemma 4 its own encoder — visionModelType below picks the family. The
// dir is -vision if set, else the sole --model's own dir when it carries a vision
// tower (auto-discovery). A multimodal tower only makes sense for a single model,
// so it errors if -vision is set with a model zoo. Absent a tower it is a no-op:
// text-only serving is unchanged.
// towerBackend is the backend a device vision tower is chosen for: the model's own, unless -vision-device cpu keeps the
// tower on the CPU (S2, docs/tasks/task-multimodal-support-2026-10.md: it also isolates the tower in a served comparison,
// with the language model on the same backend in both arms).
func (cfg config) towerBackend() string {
	if cfg.visionDevice == "cpu" {
		return "cpu"
	}
	return cfg.load.Backend
}

func (s *server) loadVisionTower(cfg config) error {
	if cfg.visionDevice != "" && cfg.visionDevice != "auto" && cfg.visionDevice != "cpu" { // "" (a config built without flags) is auto
		return fmt.Errorf("-vision-device %q: want auto or cpu", cfg.visionDevice)
	}
	dir := cfg.visionPath
	if dir == "" {
		// Auto-discover: a single --model dir that holds a vision tower — either the
		// Gemma 3 projector or a Qwen2.5-VL checkpoint (its ViT lives in the same dir).
		if len(cfg.models) == 1 {
			cand := s.soleModelSource(cfg)
			if fi, err := os.Stat(cand); err == nil && fi.IsDir() {
				if visionModelType(cand) == "qwen2_5_vl" {
					dir = cand
				} else if isQwen35VisionDir(cand) {
					dir = cand
				} else if isGlmOcrVisionDir(cand) {
					dir = cand
				} else if visionModelType(cand) == "gemma4" {
					dir = cand
				} else if _, err := multimodal.LoadProjector(cand); err == nil {
					dir = cand
				}
			}
		}
		if dir == "" {
			return nil // no vision tower — text-only model
		}
	}
	if err := visionPathError(dir); err != nil {
		return err
	}
	// N-28 (docs/audit-2026-09-10.md): cfg.models, not s.models — loadAdapters (called just
	// above) already populated s.models with each --adapter's OWN served name too, so
	// `--model base --adapter ft=…` counted 2 and refused a perfectly valid single-base-model
	// vision setup. cfg.models is the raw --model list, matching the auto-discovery branch's
	// own len(cfg.models) == 1 check above.
	if len(cfg.models) != 1 {
		return fmt.Errorf("-vision needs exactly one --model (got %d)", len(cfg.models))
	}
	if fi, err := os.Stat(dir); err == nil && !fi.IsDir() && strings.HasSuffix(strings.ToLower(dir), ".gguf") {
		return s.loadQwen35MMProj(dir, towerInt8("qwen3_5", cfg.visionQuant, cfg.load.Backend), cfg.towerBackend(), cfg.requireBE)
	}
	mt := visionModelType(dir)
	// -vision-device cpu keeps EVERY tower on the CPU, Gemma 3's resident SigLIP encoder included: towerBackend() is "cpu" under it, and the int8 default and
	// the resident attach below both key on it, not on the model's own backend. Until 2026-10-07 this branch used cfg.load.Backend for both, so a
	// `--backend cuda -vision-device cpu` Gemma 3 still ran an int8 resident tower (found by G-S3c on CUDA, where the CPU arms ran f32 on the CPU).
	int8Tower := towerInt8(mt, cfg.visionQuant, cfg.load.Backend)
	if cfg.towerBackend() == "cpu" {
		int8Tower = towerInt8(mt, cfg.visionQuant, "cpu")
	}
	if mt == "qwen2_5_vl" {
		return s.loadQwenVisionTower(dir, int8Tower, cfg.towerBackend(), cfg.requireBE)
	}
	if mt == "qwen3_5" || mt == "qwen3_5_moe" {
		return s.loadQwen35VisionTower(dir, int8Tower, cfg.towerBackend(), cfg.requireBE)
	}
	if mt == "glm_ocr" {
		return s.loadGlmOcrVisionTower(dir, int8Tower, cfg.visionMaxPixels, cfg.towerBackend(), cfg.requireBE)
	}
	if mt == "gemma4" {
		return s.loadGemma4VisionTower(dir, int8Tower, cfg.towerBackend(), cfg.requireBE)
	}
	enc, err := vision.LoadEncoder(dir, int8Tower)
	if err != nil {
		return fmt.Errorf("load vision encoder (%s): %w", dir, err)
	}
	// M-18 (docs/audit-2026-09-10.md): cuda joins webgpu here now that the resident CUDA vision
	// tower's own leak/threading bugs are fixed (cuda/vision_encoder.go) — cuda/vision_register.go
	// already registered its factory with vision.RegisterResident via cuda/cmd/serve's blank
	// import; this gate was the only thing that never called EnableResident() for it.
	residentOK := cfg.towerBackend() != "cpu" && enableResidentTower(enc, cfg.load.Backend, os.Stderr)
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
		if residentOK {
			vq = "int8/" + cfg.load.Backend + "-resident"
		}
		fmt.Fprintf(os.Stderr, "loaded vision tower for %q (%d image tokens/image, soft-token id %d, encoder %s) from %s\n", lm.name, proj.MMTokens(), lm.vimgTok, vq, dir)
	}
	return nil
}

// towerInt8 says whether a vision tower loads with int8 matmul weights. Only Gemma 3's SigLIP tower has a resident GPU encoder, and
// that encoder needs int8 (W8A8), so --backend webgpu/cuda implies int8 for it even without --vision-quant. Every other tower
// (Qwen2.5-VL, Qwen3.5+, Gemma 4, GLM-OCR) is CPU-only whatever the backend: it gets int8 only when asked for. The old rule forced int8
// on three of them under cuda/webgpu, which bought no speed (the CPU int8 tower is not faster) and cost fidelity: measured 2026-10-02
// against each tower's own f32 on the same image, relative L2 0.21 (Qwen2.5-VL), 0.14 (Qwen3.5-0.8B), 0.31 (Gemma 4), per-token
// cosine mean 0.975 / 0.992 / 0.950 (docs/measurements/vision-tower-int8-fidelity-2026-10-02.md). The gates for all of them ran f32.
func towerInt8(modelType, visionQuant, backend string) bool {
	if visionQuant == "int8" {
		return true
	}
	switch modelType {
	case "qwen2_5_vl", "qwen3_5", "qwen3_5_moe", "gemma4", "glm_ocr":
		return false
	}
	return backend == "webgpu" || backend == "cuda"
}

// enableResidentTower attaches the device-resident vision tower when the backend is webgpu or cuda and reports whether it is
// attached. A failed attach (no VRAM left for the tower, a build without the backend) is a warning, not an error: it used to abort
// serve startup and throw away the model already loaded on the GPU, but EnableResident leaves the CPU path intact, so the tower
// runs there (slower) and the banner does not claim "-resident".
func enableResidentTower(enc interface{ EnableResident() error }, backend string, warn io.Writer) bool {
	if backend != "webgpu" && backend != "cuda" {
		return false
	}
	if err := enc.EnableResident(); err != nil {
		fmt.Fprintf(warn, "warning: the resident GPU vision tower could not be enabled (%v); images will run through the CPU tower (slower)\n", err)
		return false
	}
	return true
}

// visionModelType returns dir/config.json's model_type ("" if absent/unreadable) —
// the family discriminator for the vision path (qwen2_5_vl vs Gemma 3 SigLIP).
func visionModelType(dir string) string {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return ""
	}
	var c struct {
		ModelType string `json:"model_type"`
	}
	_ = json.Unmarshal(raw, &c)
	return c.ModelType
}

// soleModelSource is the single --model as loadDecoder resolved it: for an `hf:<repo>:safetensors` reference, the
// checkpoint directory it fetched, which is where a VL repo's tower lives (task-checkpoint-fetch P7). The typed string
// names no directory, so stat-ing it found no tower and served the model text-only without a word. Falls back to the
// typed path for an entry built without a source.
func (s *server) soleModelSource(cfg config) string {
	for _, lm := range s.models {
		if lm.source != "" {
			return lm.source
		}
	}
	return cfg.models[0].path
}

// loadQwenVisionTower attaches the Qwen2.5-VL ViT (aikit) to the single loaded
// model. Unlike Gemma 3 there's no separate projector — the merger is in the
// encoder; preprocessing + m-RoPE are Qwen-specific (the image path branches on
// qwenEnc). Image placeholders use <|image_pad|>, expanded per image to the merged
// patch count.
func (s *server) loadQwenVisionTower(dir string, int8Tower bool, backend string, require bool) error {
	enc, err := vision.LoadQwenVisionEncoder(dir, int8Tower)
	if err != nil {
		return fmt.Errorf("load qwen2.5-vl vision encoder (%s): %w", dir, err)
	}
	where, err := qwenTowerPlacement(backend, int8Tower, require, enc.EnableResident, os.Stderr)
	if err != nil {
		return err
	}
	pp, err := multimodal.LoadQwenPreprocessConfig(dir)
	if err != nil {
		return fmt.Errorf("qwen2.5-vl preprocessor config (%s): %w", dir, err)
	}
	for _, lm := range s.models {
		lm.qwenEnc = enc
		lm.qwenPP = pp
		lm.qwenMerge = enc.Cfg.SpatialMergeSize
		lm.qwenImgTok = -1
		if id, ok := lm.tk.TokenID(multimodal.QwenImagePad); ok {
			lm.qwenImgTok = id
		}
		if lm.qwenImgTok < 0 {
			return fmt.Errorf("vision: tokenizer has no %q token (needed to place image embeddings)", multimodal.QwenImagePad)
		}
		fmt.Fprintf(os.Stderr, "loaded Qwen2.5-VL vision tower for %q (%s; merge %d, image-pad id %d) from %s\n", lm.name, where, lm.qwenMerge, lm.qwenImgTok, dir)
	}
	return nil
}

// loadGemma4VisionTower attaches the Gemma 4 vision tower (aikit) to the single
// loaded model. No separate projector — Gemma4Encoder.Forward bakes the
// embed_vision projection in. decoder.GenerateGemma4VL dispatches between two
// forwards depending on the checkpoint: the E2B/E4B-class sequential/causal
// path (use_bidirectional_attention unset) and the 26B-A4B/31B-class batched
// path (use_bidirectional_attention: "vision", decoder/forward_gemma4_batched.go).
// Any OTHER value is refused at load time — rather than silently serving it
// with the wrong mask — since only those two are implemented. No GPU-resident
// vision path either way: aikit's Gemma4Encoder has no EnableResident method
// (unlike vision.Encoder), so --backend webgpu has no effect on this tower
// beyond the optional int8 CPU weight format.
func (s *server) loadGemma4VisionTower(dir string, int8Tower bool, backend string, require bool) error {
	for _, lm := range s.models {
		if bd := lm.model.Config().UseBidirectionalAttention; bd != "" && bd != "vision" {
			return fmt.Errorf("gemma4 vision: %q sets use_bidirectional_attention=%q, not the supported %q value; GenerateGemma4VL only implements the E2B/E4B-class causal case (unset) and the 26B-A4B/31B-class %q blockwise case", lm.name, bd, "vision", "vision")
		}
	}
	enc, err := vision.LoadGemma4Encoder(dir, int8Tower)
	if err != nil {
		return fmt.Errorf("load gemma4 vision encoder (%s): %w", dir, err)
	}
	pp, err := multimodal.LoadGemma4PreprocessConfig(dir)
	if err != nil {
		return fmt.Errorf("gemma4 vision preprocessor config (%s): %w", dir, err)
	}
	tower, where, err := chooseGemma4Tower(enc, int8Tower, backend, require)
	if err != nil {
		return err
	}
	for _, lm := range s.models {
		lm.gemma4Enc, lm.gemma4Tower = enc, tower
		lm.gemma4MaxSoft = pp.MaxSoftTokens
		lm.gemma4ImgTok = -1
		if id, ok := lm.tk.TokenID(multimodal.Gemma4ImageSoftToken); ok {
			lm.gemma4ImgTok = id
		}
		if lm.gemma4ImgTok < 0 {
			return fmt.Errorf("vision: tokenizer has no %q token (needed to place image embeddings)", multimodal.Gemma4ImageSoftToken)
		}
		fmt.Fprintf(os.Stderr, "loaded Gemma 4 vision tower for %q (max %d soft tokens/image, soft-token id %d, tower on %s) from %s\n", lm.name, lm.gemma4MaxSoft, lm.gemma4ImgTok, where, dir)
	}
	return nil
}

// chooseGemma4Tower puts Gemma 4's vision tower on the GPU when the backend is Metal and this binary registers a Metal
// tower (docs/multimodal.md, "Finishing this doc", F2), and says where it runs. The Metal tower is float32, so an int8
// tower (-vision-quant int8) stays on the CPU; a tower that fails to start falls back to the CPU with the reason,
// unless -require-backend asks for a refusal. Other backends run it on the CPU.
func chooseGemma4Tower(enc *vision.Gemma4Encoder, int8Tower bool, backend string, require bool) (multimodal.Gemma4TowerAccelerator, string, error) {
	name, ok := deviceTowerName(backend)
	if !ok {
		return nil, "CPU", nil
	}
	if int8Tower {
		if require {
			return nil, "", fmt.Errorf("-require-backend: the Gemma 4 %s tower is float32; -vision-quant int8 keeps it on the CPU", name)
		}
		return nil, "CPU (-vision-quant int8; the " + name + " tower is float32)", nil
	}
	if !slices.Contains(multimodal.Gemma4Towers(), backend) {
		if require {
			return nil, "", fmt.Errorf("-require-backend: this binary has no %s Gemma 4 tower", name)
		}
		return nil, "CPU (no " + name + " tower in this binary)", nil
	}
	t, err := multimodal.NewGemma4Tower(backend, enc)
	if err != nil {
		if require {
			return nil, "", fmt.Errorf("-require-backend: the Gemma 4 tower could not start on %s: %w", name, err)
		}
		return nil, "CPU (" + name + " declined: " + err.Error() + ")", nil
	}
	return t, name, nil
}

// splitShardSuffix is a split GGUF's first-shard suffix, "-00001-of-00004.gguf".
var splitShardSuffix = regexp.MustCompile(`(?i)-00001-of-\d{5}\.gguf$`)

// servedNameFor is the default served name for a model path: its base name without ".gguf", and for a split GGUF
// (named by its first shard) without the "-00001-of-NNNNN" too, so a split model is served as the model it is.
// Only ".gguf" is cut: a checkpoint directory named "Qwen2.5-0.5B-Instruct" has no extension, and filepath.Ext would
// take ".5B-Instruct" for one.
func servedNameFor(path string) string {
	name := filepath.Base(path)
	if loc := splitShardSuffix.FindStringIndex(name); loc != nil {
		return name[:loc[0]]
	}
	if strings.EqualFold(filepath.Ext(name), ".gguf") {
		name = name[:len(name)-len(".gguf")]
	}
	return name
}

// loadDecoder loads one generative model + tokenizer, resolves its chat template,
// and returns it as a *loadedModel. The served name is the spec's name=, else (a
// single unnamed --model) --served-model-name, else the file/dir basename.
func loadDecoder(ctx context.Context, spec modelSpec, cfg config) (*loadedModel, error) {
	// Resolve this model's knobs (per-model overrides over server-global defaults)
	// and reject invalid enums up front.
	opts := spec.options(cfg)
	head, err := specHead(spec, &opts)
	if err != nil {
		return nil, fmt.Errorf("--model %q: %w", spec.path, err)
	}
	// A model directory that carries joint_head.safetensors is a Clef decision model (Route C): its backbone is a plain qwen3_5 with the adapters merged, and
	// the head is loaded and checked here so a bad one stops startup, not the first request.
	var clefHead *clef.Head
	if isClefDir(spec.path) {
		if spec.head != nil {
			return nil, fmt.Errorf("--model %q carries joint_head.safetensors (a Clef model) and also head=%s (a JEV head): an entry is one or the other", spec.path, *spec.head)
		}
		if err := clefQuantRefusal(opts.Quant); err != nil {
			return nil, fmt.Errorf("--model %q: %w", spec.path, err)
		}
		if clefHead, err = clef.LoadHead(spec.path); err != nil {
			return nil, fmt.Errorf("--model %q: %w", spec.path, err)
		}
	}
	if err := opts.Validate(); err != nil {
		return nil, fmt.Errorf("--model %q: %w", spec.path, err)
	}

	// tasks/task-fit-to-hardware.md §2's drafter-aware sizing: --drafter attaches AFTER this model's own
	// residency is built (below, attachBlockDrafter), but the elastic terms BuildResident sizes
	// against live free VRAM (CUDA's expert-cache slots, its unpinned ctx-by-default) have no way
	// to know it is coming unless something prices it first. Load the drafter HERE, before
	// decoder.Load, so Options.ExtraResidentBytes carries a real number into BuildResident — this
	// is the §2 example itself: a 26B auto-sized its expert cache to every free byte, then
	// --drafter attached and NewBlockSpec failed with nowhere left to go. Loaded once and reused
	// at the attach call site below, so pricing and the actual attach see the same weights rather
	// than two independent reads of the same file.
	var drafter *decoder.DFlashDrafter
	if cfg.drafter != "" {
		var derr error
		if drafter, derr = decoder.LoadDFlashDrafter(cfg.drafter); derr != nil {
			return nil, fmt.Errorf("--drafter %q: load drafter: %w", cfg.drafter, derr)
		}
		opts.ExtraResidentBytes = decoder.DrafterResidentBytesEstimate(drafter)
		// M-22 (docs/audit-2026-09-10.md): the drafter's own device K/V scales with whatever
		// resident context the target ends up choosing, which is not known yet here — see
		// Options.ExtraResidentKVPerPosition's own doc comment for why this is a rate, not a
		// total, and who multiplies it by what.
		opts.ExtraResidentKVPerPosition = decoder.DrafterKVBytesPerPosition(drafter)
	}

	// Resolve, sidecar / streaming transcode, tokenizer, swap-guarded load, the one automatic
	// streaming retry on a dense fit decline, and the .giw quant check: the path chat and fit share
	// (internal/modelload). The served name and fingerprint derive from the resolved SOURCE, never
	// the cache it may have loaded through.
	res, err := modelload.Load(ctx, modelload.Request{
		Spec: spec.path, Opts: opts, DirectLoad: cfg.load.DirectLoad,
		ExplicitQuant: spec.explicitQuant(cfg), GuardAdvice: "use -stream-weights or a smaller quant",
	})
	if err != nil {
		return nil, err
	}
	spec.path, opts = res.Source, res.Opts
	tk, model := res.Tokenizer, res.Model
	t0 := time.Now().Add(-res.LoadTime) // the banner's "in X": the model load plus the setup below, as before
	mcfg := model.Config()
	name := spec.name
	if name == "" {
		if len(cfg.models) == 1 && cfg.name != "" {
			name = cfg.name
		} else {
			name = strings.TrimSuffix(filepath.Base(spec.path), ".gguf")
			if loc := splitShardSuffix.FindStringIndex(filepath.Base(spec.path)); loc != nil {
				name = filepath.Base(spec.path)[:loc[0]] // a split model is served as the model, not as its first shard
			}
		}
	}
	fp := modelFingerprint(spec.path, model.Quant())
	lm := &loadedModel{
		tk: tk, model: model, vocab: mcfg.VocabSize, eosIDs: mcfg.EOSIDs(), name: name, fp: fp, head: head, source: spec.path,
		spec:         cfg.spec == "ngram",
		specAdaptive: cfg.specAdaptive,
		// capHint 0: KV grows on demand. The fingerprint binds disk snapshots to
		// this exact model+quant so a -session-dir reused across models is rejected.
		sessions: newSessionLRU(model, cfg.kvSessions, 0, fp),
	}
	// --drafter: attach a pretrained block drafter ONCE, here, on the BASE model only.
	// Adapter-bearing requests route down the session path (audit R-01) where the block-spec
	// branch does not run, so attaching one there would upload ~500 MB and never be used.
	//
	// It fails startup rather than degrading silently: an operator who passed --drafter wants
	// block drafting, and a wrong pairing or an incapable backend should be one startup error
	// they see, not a fleet quietly serving at 1x.
	if drafter != nil {
		if err := attachBlockDrafter(lm, drafter); err != nil {
			return nil, fmt.Errorf("--drafter %q: %w", cfg.drafter, err)
		}
	}
	// --spec ngram: the same startup-refusal rule, for the same reason. The resident branch in
	// openai.go treats a spec error as "fall back to plain Generate" PER REQUEST, which is right for
	// a sampler the spec path does not support and wrong for a load-time property: it would leave an
	// operator who asked for speculation serving every request at 1x with no signal. --drafter needs
	// no separate check here — NewBlockSpec refuses inside attachBlockDrafter above.
	if lm.spec {
		if err := model.SpecDecodeConflict(); err != nil {
			return nil, fmt.Errorf("--spec ngram: %w", err)
		}
	}
	if clefHead != nil {
		if tk == nil {
			return nil, fmt.Errorf("--model %q is a Clef model but has no tokenizer.json: its record encoder tokenizes the schema fragment by fragment", spec.path)
		}
		if lm.clef, err = clef.NewModel(model, clefHead, clef.TokenizerFunc(tk), 0); err != nil {
			return nil, fmt.Errorf("--model %q: %w", spec.path, err)
		}
	}
	if cfg.maxQueue > 0 {
		lm.queue = make(chan struct{}, 1+cfg.maxQueue)
	}
	if tmpl, derr := chat.Detect(chat.Meta{ChatTemplate: tk.ChatTemplate(), HasToken: tk.Has}); derr == nil {
		lm.tmpl = cfg.tuneTemplate(tmpl)
		for _, str := range tmpl.Stops().Strings {
			if id, ok := tk.TokenID(str); ok {
				lm.stopIDs = append(lm.stopIDs, id)
			}
		}
	}
	fmt.Fprintf(os.Stderr, "loaded %q: %d-layer model (vocab %d) in %s [chat: %s]\n",
		name, mcfg.NumLayers, mcfg.VocabSize, time.Since(t0).Round(time.Millisecond), templateName(lm.tmpl))
	// State the RESOLVED paths, not the requested ones. Both the resident decode path and the batched
	// prefill are optional capabilities that fall back silently — a model can load clean, report a GPU
	// backend, and still take one forward per prompt token (cuda int8int8: ~9× TTFT). Printing them at
	// load is what makes that visible; -require-backend turns a decline into a startup failure.
	batched, why := lm.model.PrefillPath()
	// The rest of the resolved state — context cap, KV precision, session reuse (and WHY when
	// it is off), and the features a harness asks about — comes from modelBanner so a test can
	// hold it to the runtime's own state rather than trusting a run of Fprintf calls.
	bannerCfg := cfg
	bannerCfg.load.Ctx = opts.ResidentContext // the -ctx this model actually asked for (a per-model ctx= wins)
	for _, line := range modelBanner(lm, bannerCfg) {
		fmt.Fprintf(os.Stderr, "  %s\n", line)
	}
	if lm.clef != nil {
		fmt.Fprintf(os.Stderr, "  decisions: route C (clef), joint head %d-wide (CPU, f32), backbone through PromptHiddenAll: on the device when the model is resident on a backend that implements it (CUDA), the CPU otherwise, POST /v1/systemone\n", clefHead.Cfg.Width)
	}
	if h := lm.head; h != nil {
		how := "weights merged"
		if h.AdapterDir() != "" {
			how = "adapter merged at load; this entry's text generation is the decision model's, not the base's"
		}
		fmt.Fprintf(os.Stderr, "  decisions: route B, head %s %s (%s), POST /v1/systemone\n", h.Name, h.Version, how)
	}
	// C-10: the same reasoning one line up, applied to the TOKENIZER. A pre-tokenizer this build
	// does not walk produces a different id stream from HF and from llama.cpp with no error
	// anywhere — count_tokens and usage drift by the same amount — and the only way anyone finds
	// out is by diffing ids against AutoTokenizer. Say it at load, where the other silent
	// fallbacks are already said.
	if lm.tk != nil {
		if d := lm.tk.PreTokenizerDecline(); d != "" {
			fmt.Fprintf(os.Stderr, "  !! tokenizer: %s\n", d)
		}
	}
	if cfg.requireBE {
		if err := requireFastPaths(name, cfg, lm, batched, why); err != nil {
			return nil, err
		}
	}
	return lm, nil
}

// requireFastPaths implements -require-backend for one loaded model: it fails the LOAD (so serve
// exits non-zero before binding a port) when the model didn't get the requested backend's fast
// paths. Two independent declines are checked, because either one is a large, silent regression:
// a GPU backend that produced no resident decode path (the whole forward is staged/CPU), and a
// prefill that declined to the sequential per-token loop. The CPU backend has no resident path by
// definition, so only the prefill half applies there.
func requireFastPaths(name string, cfg config, lm *loadedModel, batched bool, why string) error {
	if cfg.load.Backend != "cpu" && !lm.model.ResidentActive() {
		// Covers the whole ladder, not just an ineligible arch: an untagged build (`--backend cuda`
		// without `-tags cuda` resolves to the CPU backend), a box with no usable device (the cuda
		// factory always succeeds — the driver is only touched at BuildResident), and a model shape
		// the runner refuses. All three otherwise present as a healthy server that is silently on CPU.
		reason := lm.model.ResidentDecline()
		if reason == "" {
			reason = "no reason recorded"
		}
		return fmt.Errorf("--require-backend: model %q did not build a resident decode path on backend %q: %s (resolved: %s)",
			name, cfg.load.Backend, reason, lm.model.DecodePath())
	}
	if !batched {
		return fmt.Errorf("--require-backend: model %q declined the batched prefill: %s", name, why)
	}
	return nil
}

// requireAutoBackend is -require-backend's check on -backend auto, made at startup before any model loads. When auto
// passed over a GPU backend this binary links (no device answered, it is untested on this machine, or it is webgpu), it
// would run on the CPU, which is the silent fallback strict mode exists to stop, so it refuses and says how to choose.
// A binary with no GPU backend has only the CPU, so auto there is the CPU, and the per-model checks run as for
// -backend cpu.
func requireAutoBackend(cfg config) error {
	if !cfg.requireBE || cfg.load.Auto == nil || !cfg.load.Auto.Skipped {
		return nil
	}
	return fmt.Errorf("--require-backend: -backend auto would run on the CPU (%s); pass -backend cpu to run strict on the CPU, or name a GPU backend to require it",
		cfg.load.Auto.Reason)
}

// embedEncoderLoads is the plan check for an -embed-model safetensors checkpoint (task-checkpoint-fetch P7). loadEncoder's
// directory path is aikit's encoder.Load, which loads a NomicBert (CodeRankEmbed, nomic-embed-text) and checks no
// model_type itself, so anything else would fail late, or load a foreign checkpoint's tensors under the wrong
// architecture. It is refused here instead, after config.json and before any weight. A decoder used as an embedder
// comes as a GGUF (hf:<repo>:<quant>), which is not a checkpoint plan at all.
func embedEncoderLoads(modelType string) (string, error) {
	if fam, err := pull.EncoderLoads(modelType); err == nil {
		return fam, nil
	}
	return "", fmt.Errorf("model_type %q, which -embed-model's encoders do not load (they load nomic_bert: CodeRankEmbed, nomic-embed-text; and embedding_gemma2: google/embeddinggemma-2; a decoder used as an embedder comes as a GGUF, hf:<owner>/<repo>:<quant>)", modelType)
}

// resolveEmbedModel turns -embed-model into a local path: a plain path unchanged, hf:<repo>:safetensors fetched as an
// encoder checkpoint under embedEncoderLoads, any other hf: reference resolved as a GGUF. A seam, so a test can stand in
// the fetch.
var resolveEmbedModel = func(ctx context.Context, spec string) (string, error) {
	return pull.ResolveCheckpointFor(ctx, spec, embedEncoderLoads)
}

// loadEncoder loads the embedding model (f32 or int8) plus its tokenizer (used
// only to count tokens for usage.prompt_tokens — counting needs no forward pass,
// so it is cheaper than running EncodeTokensWithIDs, and works for both
// precisions, whereas that method is f32-only).
func (s *server) loadEncoder(cfg config) error {
	// A causal decoder used as an embedder (qwen3-embedding / embeddinggemma) arrives as a .gguf
	// FILE; the aikit encoder path takes an HF directory. Dispatch on that rather than adding
	// another flag — see docs/completed/task-decoder-as-embedder.md.
	if fi, statErr := os.Stat(cfg.embedPath); statErr == nil && !fi.IsDir() {
		return s.loadDecoderEmbedder(cfg)
	}
	// EmbeddingGemma 2 is a directory too, but aikit's encoder.Load reads it as a NomicBert and would fail late; it
	// has its own loader (embeddinggemma2_embedder.go).
	if isEmbeddingGemma2(cfg.embedPath) {
		return s.loadEmbeddingGemma2(cfg)
	}
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
	// Matryoshka floor from aikit's exported registry — the same source of truth behind its
	// published Truncatable column. Unknown/non-MRL models get 0, i.e. `dimensions` is refused
	// rather than honored into a silently worse-retrieving vector. Keyed off the model PATH (the
	// directory name is the HF model name); embedName is an operator-chosen alias, so it must not
	// decide this.
	s.embedMRLMin, _ = encoder.MatryoshkaFloor(cfg.embedPath)
	trunc := "not truncatable"
	if s.embedMRLMin > 0 {
		trunc = fmt.Sprintf("truncatable to %d", s.embedMRLMin)
	}
	fmt.Fprintf(os.Stderr, "loaded embedding model %q (dim %d, %s, %s) in %s\n",
		name, s.embedDim, prec, trunc, time.Since(t0).Round(time.Millisecond))
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
// modelList snapshots the registry under regMu. Background sweeps (demote, shutdown
// checkpoint) must iterate this, not range s.models directly: admin load/unload mutate the
// map under regMu (admin.go), and a concurrent map iteration+write is a runtime-fatal panic,
// not just a race (M4). The returned slice is a copy of the pointers; each loadedModel is
// still locked via its own lm.mu by the caller.
// tryLockUntil acquires mu, giving up at deadline instead of blocking forever, so the shutdown
// checkpoint can never deadlock on a generation that outlived the drain (audit C-22). Returns false
// if the lock was not taken by the deadline.
func tryLockUntil(mu *sync.Mutex, deadline time.Time) bool {
	for {
		if mu.TryLock() {
			return true
		}
		if !time.Now().Before(deadline) {
			return false
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func (s *server) modelList() []*loadedModel {
	s.regMu.RLock()
	defer s.regMu.RUnlock()
	out := make([]*loadedModel, 0, len(s.models))
	for _, lm := range s.models {
		out = append(out, lm)
	}
	return out
}

func demoteLoop(srv *server, idle time.Duration, stop <-chan struct{}) {
	period := min(max(idle/4, 5*time.Second), time.Minute)
	t := time.NewTicker(period)
	defer t.Stop()
	for {
		select {
		case <-stop:
			return
		case <-t.C:
			for _, lm := range srv.modelList() {
				lm.sessMu.Lock()
				if n := lm.sessions.demoteIdle(); n > 0 {
					fmt.Fprintf(os.Stderr, "tiered-kv: demoted %d idle session(s) for %q\n", n, lm.name)
				}
				lm.sessMu.Unlock()
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
	return t.Name() + thinkingNote(t) + toolFormatNote(t)
}

// toolFormatNote says, for a model whose template declares a native tool form, which prompt serve renders tools with and how to get the
// other — the same reason thinkingNote exists: an operator should not have to find out from a model's behaviour.
func toolFormatNote(t *chat.Template) string {
	if !t.DeclaresNativeTools() {
		return ""
	}
	if t.UsesNativeTools() {
		return ", tools: the model's own template form (-tool-format hermes for goinfer's)"
	}
	return ", tools: goinfer's own form (-tool-format template for the model's own)"
}

// thinkingNote is the load log's statement of what serve will do about thinking for this model: the checkpoint's own
// default (read from its template), the mode -thinking selected, or "unmanaged" — a ChatML/Gemma template whose thinking
// control was not recognised is served exactly as before, and saying so beats leaving an operator to find out.
func thinkingNote(t *chat.Template) string {
	r := t.Reasoning()
	if r == nil {
		if t.Name() == "harmony" { // gpt-oss: always reasons (no off form in its prompt), and the reply is channel messages
			return ", reasoning: analysis channel split from the answer (gpt-oss always reasons; -thinking cannot turn it off)"
		}
		if t.Name() == "chatml" || t.Name() == "gemma4" {
			return ", thinking: unmanaged"
		}
		return ""
	}
	def := map[bool]string{true: "on", false: "off"}[r.DefaultOn()]
	return fmt.Sprintf(", thinking: template default %s, serving %s", def, t.ThinkMode())
}

// visionPathError is -vision's refusal for a path that is a file, not a vision-tower directory (R22, docs/tasks/task-first-hour.md).
// A GGUF mmproj handed to it used to fail inside the encoder loader as ".../mmproj-....gguf/config.json: not a directory". A path that
// does not exist, or a directory, is the loaders' to judge, with their own messages.
func visionPathError(dir string) error {
	fi, err := os.Stat(dir)
	if err != nil || fi.IsDir() {
		return nil
	}
	if strings.HasSuffix(strings.ToLower(dir), ".gguf") {
		return nil // a GGUF mmproj: loadVision routes it (Qwen3.5+ only, P8b), with its own refusals
	}
	return fmt.Errorf("-vision %s is a file; -vision takes a directory with a vision tower (config.json and safetensors)", dir)
}

// qwenTowerPlacement decides where Qwen2.5-VL's vision tower runs and attaches it (S4, docs/tasks/task-multimodal-support-2026-10.md): aikit's gpu/qwencuda
// tower under --backend cuda when the binary registers one (cuda/vision_towers.go imports it) and the tower is float32 (G-S4q: correct at real size, 1.6-2.7x the
// CPU tower); the CPU everywhere else, with the reason named. -require-backend turns each CPU fallback under cuda into a refusal. -vision-device cpu arrives
// here as backend "cpu". Metal has no Qwen2.5-VL device tower yet (the owner's rebuild on the Metal base is the Mac's), so only cuda asks for one.
func qwenTowerPlacement(backend string, int8Tower, require bool, attach func() error, warn io.Writer) (string, error) {
	if backend != "cuda" {
		return "CPU", nil
	}
	if int8Tower {
		if require {
			return "", fmt.Errorf("-require-backend: the Qwen2.5-VL CUDA tower is float32; -vision-quant int8 keeps it on the CPU")
		}
		return "CPU (-vision-quant int8; the CUDA tower is float32)", nil
	}
	if err := attach(); err != nil {
		if require {
			return "", fmt.Errorf("-require-backend: the Qwen2.5-VL tower could not start on cuda: %w", err)
		}
		fmt.Fprintf(warn, "vision: the Qwen2.5-VL tower runs on the CPU: cuda declined it: %v\n", err)
		return "CPU (cuda declined)", nil
	}
	return "CUDA", nil
}
