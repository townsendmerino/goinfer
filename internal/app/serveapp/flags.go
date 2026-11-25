package serveapp

import (
	"errors"
	"flag"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/app/loadflags"
	"github.com/townsendmerino/goinfer/internal/decide"
	"github.com/townsendmerino/goinfer/internal/runtime/prequant"
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
	if name, path, ok := strings.Cut(fields[0], "="); ok && name != "" {
		spec.name, spec.path = name, path
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

// explicitQuant returns the quant the user EXPLICITLY chose for this model: the per-model `quant=` override, else the
// global --quant if it was actually passed, else "" (the process default). Used only for the .giw mismatch check: a bare
// default must never conflict with an already-baked bundle.
func (s *modelSpec) explicitQuant(cfg *config) string {
	if s.quant != nil {
		return *s.quant
	}
	return cfg.load.ExplicitQuant()
}

// options resolves this spec's overrides over the server-global defaults (cfg.load, the flags chat
// shares) into a decoder.Options. Backend is process-wide (GPU device init), never per-model.
func (s *modelSpec) options(cfg *config) decoder.Options {
	ov := loadflags.Overrides{Quant: s.quant, LoRA: s.lora, KV: s.kvPrec, KVQuant: s.kvQuant, Stream: s.stream,
		WeightCacheGB: s.weightCache, EmbedInt4: s.embedInt4, Ctx: s.ctxSize, Serve: &loadflags.ServeLoad{
			KVQuant: cfg.kvQuant, KVSessions: cfg.kvSessions, KVSessionsSet: cfg.kvSessionsSet, PrefillChunk: cfg.prefillChunk,
			KVShareOff: cfg.kvShareOff, KVHostBytes: cfg.kvHostBytes, KVDemote: cfg.kvDemote, CPUBatch: cfg.cpuBatch}}
	if s.head != nil || isClefDir(s.path) { // a decision head (JEV or Clef) loads at the decision models' default unless a quant was chosen
		q := decide.HeadQuant(s.explicitQuant(cfg), s.quant != nil || cfg.load.QuantSet)
		ov.Quant = &q
	}
	return cfg.load.OptionsWith(ov)
}

// specHead loads a --model entry's head= (a trained decision head) and, for a head whose adapter is unmerged, makes
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

// adapterSpec is one --adapter entry: a served name, the --model it attaches to (base), and the PEFT adapter dir. The
// base's resident weights are shared, so each adapter costs only its low-rank A/B bytes.
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

// config is the resolved command line for newServer.
type config struct {
	models   modelFlag   // decoder(s) (-model, repeatable); empty = no generative endpoints
	adapters adapterFlag // compute-time LoRA adapters (-adapter, repeatable); each shares a base model's resident weights (#7)
	// load is the model-loading flags goinfer-chat shares (internal/loadflags): server-global
	// DEFAULTS, each of which a --model spec can override (see modelSpec / modelFlag.Set).
	load              loadflags.Flags
	kvQuant           string // DEPRECATED -kv-quant: CPU KV cache override, "" = follow -kv
	name              string // -served-model-name (applies only to a single unnamed --model)
	kvSessions        int
	kvSessionsSet     bool          // -kv-sessions was given on the command line (flag.Visit after Parse)
	sessionDir        string        // -session-dir (also where /admin unload snapshots warm KV)
	kvIdleDemote      time.Duration // -kv-idle-demote: tiered KV — demote a session idle this long to disk (0 = off)
	kvDemotedMax      int           // -kv-demoted-max: cap on the on-disk cold tier
	maxQueue          int           // -max-queue: bounded per-model queue depth (0 = unbounded)
	maxConcurrent     int           // -max-concurrent: generations one CPU model may run at once (1 = serialized)
	cpuBatch          int           // -cpu-batch: decoder.CPUBatchAuto (default) / CPUBatchOn / CPUBatchOff
	prefillChunk      int           // -prefill-chunk: under concurrent generation, a long prompt arriving while others decode prefills in chunks of this many tokens (default 512); 0 = off
	kvShare           string        // -kv-share: "on" (default) or "off"
	kvShareOff        bool          // kvShareOffFor's answer, set once the API key and bind are known
	kvHostBytesFlag   string        // -kv-host-bytes: auto, 0, or a byte count
	kvHostBytes       int64         // the resolved budget for prefix snapshots (0: off); a config built without the flag keeps none
	kvHostBytesHow    string        // how it was reached, for the startup line
	kvDemote          bool          // the flag names a budget, not auto: released conversations' rows wait in it too
	jobDir            string        // -job-dir: optional dir for the job journal (one JSONL line per state transition); "" = in-memory job tracking only, no durability
	maxInflight       int           // -max-inflight: global cap on concurrent inference handlers (bounds pre-queue work; 0 = unbounded)
	maxBodyBytes      int64         // -max-body-bytes: request-body cap (0 = derive from the model's context window)
	maxFileStoreBytes int64         // -max-file-store-bytes: what uploaded files and batch results may hold in memory (0 = no limit on bytes)
	unloadDrainWait   time.Duration // -unload-drain-wait: how long an unload waits for in-flight requests to drain before 202 (native free continues detached)
	spec              string        // -spec: "" (off) | "ngram" — lossless n-gram speculative decode
	specAdaptive      bool          // -spec-adaptive: "speculate when alone, batch under load" (needs -spec ngram and a resident that batches; off has no effect otherwise)
	drafter           string        // -drafter: dir of a pretrained BLOCK drafter (DFlash); resident GPU backends only
	drafterWidth      int           // -drafter-width: the verify width a greedy request's block-spec rounds run at; 0 = decoder's default
	allowAdmin        bool          // -allow-admin: enable POST /admin/models/{load,unload}
	allowedHosts      string        // -allowed-hosts: extra Host names a loopback-bound listener answers (hostGuard)
	logRequests       bool          // -log-requests: one stderr line per generation request
	logTokenIDs       string        // -log-token-ids: append one JSON line per generation (prompt hash, generated ids) to this file; "" = off
	idsKeyFile        string        // -ids-key-file: the key that signs carried token ids; "" = a random key per start
	noCarryIDs        bool          // -carry-ids=false: do not carry a reply's token ids back in its reasoning item (the zero value carries them)
	fastAlias         bool          // -fast-alias: serve "<model>-fast", the same model with thinking off, for an agent's side requests
	haltFile          string        // -halt-file: polled every 250ms; present ⇒ halted, absent ⇒ resumed
	haltExitCode      int           // -halt-exit-code: nonzero ⇒ halt exits the process with this code after quiescence; 0 = stay up
	adminSocket       string        // -admin-socket: serve /admin/* on this Unix socket instead of the TCP listener; "" = off
	web               bool          // -web: serve the local browser UI + its model-pull routes
	requireBE         bool          // -require-backend: refuse to start when a model silently fell back off the requested backend's fast paths (resident decode / batched prefill)
	visionPath        string        // -vision: dir holding the vision tower (SigLIP + projector) for a multimodal --model
	noSelfTest        bool          // -no-selftest: skip the startup self-tests
	visionMaxPixels   int           // -vision-max-pixels: lowers GLM-OCR's image pixel budget (0 = the model's own 4.82 MP ceiling)
	visionQuant       string        // -vision-quant: "" (unset: f32; Gemma 3 on cuda may fall back to int8) | "f32" (explicit) | "int8" (W8A8; only faster on AVX512-VNNI — a WASH on AVX2)
	visionDevice      string        // -vision-device: "auto" (default: the backend's device tower when it has one) | "cpu" | "gpu" (as auto, and it also puts on the device the towers auto keeps on the CPU under webgpu: on a discrete GPU, and a Qwen3.5+ .gguf mmproj tower)
	// decisions: POST /v1/systemone's label-scoring template, and an
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
	// lenientToolCalls is -lenient-tool-calls: also read ONE fenced JSON call at the end of a reply as a tool call, on the <tool_call> families.
	// Off by default. Applied to the model's template at load.
	lenientToolCalls bool

	embedPath  string // encoder (-embed-model); "" = no /v1/embeddings
	embedQuant string // "" | f32 | q8
	embedName  string // -embed-served-model-name
	// embedResize is -embed-image-resize: how an image-embedding model resizes an image before its tower ("" = the
	// model's default). EmbeddingGemma 2 takes bilinear (aikit's) or bicubic (its reference processor's).
	embedResize string
}

// markGivenFlags records, after fs is parsed, which of cfg's flags the command line gave rather than defaulted:
// -kv-sessions, whose default lowers Metal to 2 GPU KV slots while a given count is kept.
func markGivenFlags(fs *flag.FlagSet, cfg *config) {
	fs.Visit(func(f *flag.Flag) {
		if f.Name == "kv-sessions" {
			cfg.kvSessionsSet = true
		}
	})
}

// serveFlags is what the serve flags parse into: the config every model shares, the four listener flags that
// are not part of it, --version, and the model-loading flags goinfer-chat shares (loadflags).
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
	fs.StringVar(&cfg.jobDir, "job-dir", "", "optional dir for a durable job journal — one JSONL line per generation state transition (pending/running/done/failed/cancelled), 0700/0600 permissions matching -session-dir. On restart, any job still 'running' at the last recorded transition is marked 'interrupted', never silently 'failed'. Off by default: every generation still gets an in-memory job record, just no durability across a restart")
	fs.BoolVar(&cfg.web, "web", false, "serve a local browser UI at / — chat with the loaded model, and pull and load GGUF files or whole safetensors checkpoints from HuggingFace, on the same server and the same /v1 routes any other client uses (one embedded HTML file; no external assets, so it works offline). Off by default: the page is static, but its pull route starts a caller-named multi-gigabyte download and writes it to disk. On a non-loopback bind the existing -api-key requirement applies as usual")
	fs.BoolVar(&cfg.allowAdmin, "allow-admin", false, "enable /admin/* on THIS (TCP) listener — model load/unload (loads attacker-named paths), GET /admin/generations, POST /admin/generations/{id}/cancel, POST /admin/halt, POST /admin/resume (deliberate opt-in; requires -api-key). A /v1 client holding the same key can reach every one of these routes too, including halt/resume. Ignored when -admin-socket is set: /admin/* is then not registered on TCP at all (a request 404s, not 403s — this listener does not admit the surface exists), and is served on the socket instead with no key check")
	fs.StringVar(&cfg.allowedHosts, "allowed-hosts", "", "comma-separated Host names, beyond localhost, 127.0.0.1, [::1] and the -addr host, that a listener bound to the loopback answers (default none). A request whose Host is any other name gets 421: that is what stops a web page from reaching a loopback server by DNS rebinding. Set it when a reverse proxy on this machine forwards a public name to the loopback port. Has no effect on a listener bound beyond the loopback, which answers any Host and is protected by -api-key")
	fs.BoolVar(&cfg.logRequests, "log-requests", false, "write one line to stderr per generation request (/v1/chat/completions, /v1/completions, /v1/responses, /v1/messages) when it finishes: route, model, status, prompt and completion tokens, time to first token and total time, then the first generation's KV reuse (reused, the slot's length before the request, and div, where the prompt left it), canon (1 when the generated ids are the tokenizer's own split of their text), and on /v1/responses and /v1/messages the carried token ids (ids=<used>/<carried>, ids_fallback, the reason for each turn that re-tokenized, and ids_fmt when used turns replaced template-owned formatting), spec=<drafted>/<accepted> on a request that speculated, and lazy=armed when a speculating server's auto tool turn took the constrained path at its call opener. Off by default; a request that failed before generating shows `-` for the model and token counts")
	fs.StringVar(&cfg.logTokenIDs, "log-token-ids", "", "for measurement runs: append one JSON line per generation to this file — time, route, response id, the prompt's length and SHA-256 (of its ids as little-endian uint32s), reused and slot positions, the generated token ids, and whether they are the tokenizer's own encoding of their text. The file holds the model's output as token ids; it is created 0600. Off by default")
	fs.Var(invertedBool{&cfg.noCarryIDs}, "carry-ids", "carry a /v1/responses reply's generated token ids back to the server in its reasoning item's encrypted_content (signed), and use them when the reply is echoed, so it re-tokenizes to the ids the model generated. false writes the text-only blob and re-tokenizes every echoed turn, as before this existed")
	fs.BoolVar(&cfg.fastAlias, "fast-alias", true, "a served model N is also reachable as N-fast: the same weights, template and KV slot, with thinking off and nothing else different, for the side requests an agent sends for its own bookkeeping (a session title). connect points Claude Code's small-model setting and opencode's small_model at it. false turns the alias off")
	fs.StringVar(&cfg.idsKeyFile, "ids-key-file", "", "a file holding the key (32 bytes or more, mode 0600) that signs the token ids a /v1/responses reply carries back in its reasoning item's encrypted_content, so an echoed reply re-tokenizes to the ids the model generated. Default: a random key made at start and never written, so after a restart, or on another server, older replies fall back to re-tokenizing their text (counted as `key`). Set it to keep carried ids valid across restarts or between servers that share it")
	fs.StringVar(&cfg.haltFile, "halt-file", "", "poll this path every 250ms — present halts the server (every inference route 503s, in-flight generations are cancelled), absent resumes it. No HTTP call, socket, or signal needed; a supervisor halts with `touch` and resumes with `rm`. The model stays loaded either way; resume is instant. Off by default")
	fs.IntVar(&cfg.haltExitCode, "halt-exit-code", 0, "when nonzero, any halt (admin, -halt-file, or SIGUSR1) exits the process with this code once every cancelled generation has actually stopped, instead of staying up halted. For a supervisor whose restart policy must not undo a deliberate halt (RestartPreventExitStatus=N or the equivalent). 0 (default) means halt never exits the process")
	fs.StringVar(&cfg.adminSocket, "admin-socket", "", fmt.Sprintf("serve /admin/* (load/unload, cancel/list, halt/resume) on a Unix socket instead of the TCP listener — mode 0600, unlinked and recreated fresh at start, no -api-key check (the socket's file permissions are the auth, so its directory must be one nobody else can rewrite: not writable by its group or by everyone, unless it is sticky like /tmp). Removes /admin/* from the TCP listener entirely (404, not 403) — see -allow-admin. Suggested path: %s. Off by default (empty = /admin/* stays on TCP, gated by -allow-admin as before). Control it with the same binary: `%[2]s status|ls|cancel <id>|halt [reason]|resume` talks to this socket (defaults to the suggested path above when -admin-socket is not repeated on that command line)", defaultAdminSocketPath(), filepath.Base(os.Args[0])))
	fs.StringVar(&cfg.visionPath, "vision", "", "vision tower dir for a multimodal --model (auto-discovered per family: SigLIP+projector for Gemma 3, Qwen2.5-VL's own ViT, or Gemma 4's own encoder); enables image content parts. Defaults to the --model dir when it contains a vision tower")
	fs.StringVar(&cfg.decisionsTemplate, "decisions-template", "chat-v1", "POST /v1/systemone's prompt template: chat-v1 (the model's chat template; for instruct models) or bare-v1 (JEV's own, no chat template)")
	fs.StringVar(&cfg.decisionsCal, "decisions-calibration", "", "calibration.json with per-kind temperatures for /v1/systemone (from goinfer-chat decisions-calibrate, fitted under the same template); none: every answer is uncalibrated")
	fs.StringVar(&cfg.thinking, "thinking", "template", "default thinking mode for a model whose chat template has a recognised thinking control (Qwen3, Qwen3.5, Gemma 4): template (what the checkpoint's own template renders — Qwen3.5-0.8B: off, Qwen3.5-9B: on), asis (the prompt bytes serve rendered before thinking was modelled: nothing written, the model decides), on, or off. A request overrides it with chat_template_kwargs.enable_thinking, reasoning_effort, or Anthropic's thinking. A model whose template control is not recognised ignores this.")
	fs.StringVar(&cfg.toolFormat, "tool-format", "auto", "how tools are put in the prompt for a model whose own chat template declares a tool form goinfer can render byte for byte — Qwen3.5 (its <function=…><parameter=…> XML) and Gemma 4's canonical template: auto (the default: each family's measured default — the model's own template form for Gemma 4, goinfer's own prompt for Qwen3.5), hermes (goinfer's own prompt: for Qwen a JSON call that tool_choice can constrain, for Gemma 4 its earlier order, text before a call) or template (the model's own chat template; tool_choice naming a function is then a 400, as for any form with no JSON wrapper). The Qwen reply parser reads both call forms whichever is chosen. No effect on a model with no such form.")
	fs.BoolVar(&cfg.lenientToolCalls, "lenient-tool-calls", false, "also read a tool call a model writes as ONE fenced JSON block at the end of its reply (```json {\"name\": ..., \"arguments\": {...}} ```) as a call, for the Qwen-style <tool_call> families (Qwen2.5-Coder under an agent writes its calls this way and otherwise makes none). Narrow on purpose: exactly one fence, last in the reply, tag json or none, only name/arguments/id keys, a supplied tool name, arguments that validate against that tool's schema. Off by default because a demonstration that ends on one valid call looks the same as a meant call, so a model asked to show an example can have it executed; turn it on for an agent whose client confirms before it acts")
	fs.StringVar(&cfg.reasoningBudget, "reasoning-budget", "auto", "ceiling on how long a thinking reply may think before serve forces the block closed so the reply can answer: auto (the default: thinking takes at most three quarters of the request's max_tokens), unlimited (no ceiling of serve's own), or N (cap every thinking reply at N tokens, still leaving room to answer). A request's own budget (thinking_token_budget, or Anthropic's thinking.budget_tokens) applies too, clamped so a quarter of max_tokens is left to answer. Not applied under -spec or -drafter, or to a model with no recognised thinking control.")
	fs.StringVar(&cfg.reasoningFmt, "reasoning-format", "deepseek", "how a reply's reasoning reaches the client: deepseek (content is the clean answer, reasoning goes in reasoning_content / Anthropic thinking blocks), deepseek-legacy (reasoning_content is filled and content keeps the raw <think> tags), or none (nothing is separated: the raw text is the content). A request may override it with reasoning_format.")
	fs.BoolVar(&cfg.noSelfTest, "no-selftest", false, "skip the startup self-tests: a kernel check against a reference that steps a CPU kernel tier down, or declines a GPU backend, when its output disagrees. On by default; skip it only if it misjudges a healthy machine (and tell us: `check --hardware` prints what it found).")
	fs.IntVar(&cfg.visionMaxPixels, "vision-max-pixels", 0, "GLM-OCR only: lower the image pixel budget to this many pixels (0 = the model's own ceiling, 4.82 MP; it is never raised). The CPU tower costs about 29 s at 1 MP, 92 s at 2 MP and 7 min at 4.8 MP on an M1 Pro.")
	fs.StringVar(&cfg.visionDevice, "vision-device", "auto", "where the vision towers and the Whisper-style audio encoder (Whisper, Voxtral) run: auto (default: on the --backend's GPU when this binary has one there, else the CPU; under --backend webgpu that is Gemma 3's, Gemma 4's, Qwen2.5-VL's, Qwen3.5+'s, Qwen3-VL's and GLM-OCR's towers, in float32, on a unified-memory Mac; a discrete GPU and a Qwen3.5+ .gguf mmproj tower keep the CPU) | cpu (the CPU whatever the backend; the language model keeps its own backend) | gpu (as auto, and it also puts on the device the towers auto keeps on the CPU under webgpu: on a discrete GPU, and a Qwen3.5+ .gguf mmproj tower)")
	fs.StringVar(&cfg.visionQuant, "vision-quant", "", "vision encoder weight quant: f32 (bit-exact) | int8 (W8A8; lossy: relative L2 0.14-0.52 against f32, docs/measurements/vision-tower-int8-fidelity-*) — int8 only speeds the compute-bound ViT prefill on AVX512-VNNI; on AVX2 it's a wash. Unset (the default) is f32 everywhere; on CUDA Gemma 3 is int8 instead (with a note) if the card cannot hold the decoder, one KV slot and the float32 tower, or if that tower fails to attach (an explicit f32 never falls back). Under --backend webgpu Gemma 3's unset default is the float32 tower on the GPU on a unified-memory Mac (a discrete GPU keeps the CPU tower, and an explicit f32 or int8 asks for the device); int8 given explicitly puts the int8 one there")
	fs.Var(&cfg.models, "model", "generative model: a .gguf/.giw file, an HF dir, or a reference that is fetched on first use — hf:<owner>/<repo>:<quant> (e.g. hf:Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF:q4_k_m), hf:<owner>/<repo>:safetensors (a safetensors checkpoint, fetched as a verified set) or demo:<tier>. A reference is sha256-verified and cached; a path is used as-is. Repeatable\n"+
		"as `name=path` to serve a model zoo from one process; requests route on the\n"+
		"OpenAI `model` field. Append comma-separated per-model overrides of the global\n"+
		"defaults below: `--model big=moe.giw,stream,weight-cache=16 --model fast=small.giw`\n"+
		"streams only the big MoE. Keys: quant,lora,kv,kv-quant,ctx,stream,weight-cache,embed-int4,head.\n"+
		"head=DIR attaches a trained decision head (autotrust's JEV layout): POST /v1/systemone on that model\n"+
		"answers with it (Route B), and an unmerged head's adapter is merged into the model at load. The model loads\n"+
		"at "+decoder.DecisionHeadQuant+" unless quant= or -quant chooses otherwise (the default for decision models).\n"+
		"A model directory that carries joint_head.safetensors is a Clef decision model (Route C): POST /v1/systemone on it\n"+
		"answers with its joint head, one backbone pass for every question, and it loads at the same default.\n"+
		"(Paths may not contain commas.)")
	fs.IntVar(&cfg.drafterWidth, "drafter-width", 0, "verify width for --drafter: how many block positions the target verifies per round (the anchor and width-1 drafts). 0 keeps the default of 8; the drafter's block caps it. The best width depends on the pairing and the box: a narrower width can beat the default (docs/measurements/sp2-2026-10/README.md). Widths under 4 are refused (a round then commits too few tokens for the acceptance guard, so most turns fall back to plain decode), and so is a width the target cannot verify (past a recurrent model's fold rows).")
	fs.StringVar(&cfg.drafter, "drafter", "", "directory of a pretrained BLOCK drafter (z-lab DFlash) paired with --model: the drafter proposes a whole block of tokens per round and the target verifies them in ONE batched pass, measured 1.6-1.8x on code/math and ~0.96x on open chat (docs/spec/08). LOSSLESS — every emitted token is one the target's own argmax produced, so output is identical to plain greedy. Greedy only: a request with temperature, penalties or logit bias falls back to normal decoding automatically. Requires a resident GPU backend (--backend cuda); declines with a reason otherwise serve-only; goinfer-chat offers --spec ngram and --draft instead.")
	sf.showVersion = fs.Bool("version", false, "print version, the backends COMPILED INTO this binary, and the Go toolchain, then exit. `backends:` is the compiled-in truth — --backend accepts names this build cannot run and falls back to cpu")
	// The model-loading flags goinfer-chat shares — one registration, so the two binaries cannot drift.
	sf.lf = loadflags.Register(fs, loadflags.Serve)
	fs.BoolVar(&cfg.requireBE, "require-backend", false, "strict mode: exit non-zero at startup if a model did not resolve to the requested --backend's fast paths — no resident decode path, or a prefill that declined to the sequential per-token loop (e.g. native f32 on cuda, a much slower time to first token: every quantized mode gets batched CUDA prefill, see -quant's own help above; only f32 falls back). Both fall back silently by design; a batch client should fail at second zero instead of discovering it under load. Under -backend auto it also refuses to start when auto passed over a GPU backend this binary has (no device answered) rather than run on the CPU; -backend cpu runs strict on the CPU")
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
	fs.Func("cpu-batch", "auto|on|off: whether concurrent CPU generations of one model join their decode tokens into one batched forward (replies are bit-identical either way). auto (the default) batches models with at least 2 GiB of dense weights, where batching beats independent workers, and keeps smaller models on the workers", func(v string) error {
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
	fs.StringVar(&cfg.kvHostBytesFlag, "kv-host-bytes", "auto", "host memory the server may hold as prefix snapshots on a GPU-resident model: after a cold first prefill of a long prompt, the prompt's KV is copied to host memory, and a later request that starts with the same ids restores it instead of prefilling it. auto (the default) is a tenth of the machine's RAM, at most 8 GiB; 0 turns it off; or a byte count with an optional K, M or G. Memory is held only while a snapshot is, so an idle server holds none. A budget named here (not auto) also keeps a released idle conversation's rows in host memory on CUDA, and they come back when it returns; that is experimental, and a slot holding a wrapped upload is not kept. -kv-share off turns snapshots off too")
	fs.StringVar(&cfg.kvShare, "kv-share", "on", "on|off: whether a conversation may reuse KV rows another conversation computed, such as a shared system prompt, on a GPU-resident model (on: a new conversation copies the shared rows instead of prefilling them; off: a request reuses only its own conversation's rows). Sharing makes time to first token reveal something about what another client sent, so it is off whatever this says when the server answers the network without -api-key (an -allowed-hosts name on a loopback bind)")
	fs.IntVar(&cfg.prefillChunk, "prefill-chunk", 512, "on a GPU-resident model running several generations at once, prefill a long prompt that arrives while others are decoding in chunks of this many tokens, one decode step between chunks, instead of stalling them for the whole prompt (replies are unchanged: Metal's prefill is chunk-invariant; 512 keeps the other decoders' longest stall well below a whole prompt's); 0 = off")
	fs.IntVar(&cfg.maxInflight, "max-inflight", 128, "global cap on concurrent inference requests, bounding the pre-queue stage (JSON+image decode, tokenization, template render, vision Forward) that runs before the per-model queue; a full cap returns 503 Retry-After (0 = unbounded)")
	fs.Int64Var(&cfg.maxFileStoreBytes, "max-file-store-bytes", 1<<30, "cap on the bytes of uploaded files and batch results held in memory. The oldest go first, and the newest upload or batch's result files are always kept, so a dropped file's id answers 404: download a batch's results before uploading much more. 0 = no limit on bytes (the store still keeps at most 256 files)")
	fs.Int64Var(&cfg.maxBodyBytes, "max-body-bytes", 0, "cap on request body size in bytes; a larger body is rejected 413 before it is read. 0 = derive from the model's context window (a body that could never fit is rejected up front). The vision endpoints get at least 32 MiB on top for base64 image data")
	fs.DurationVar(&cfg.unloadDrainWait, "unload-drain-wait", 5*time.Second, "how long POST /admin/models/unload waits for in-flight requests to drain before returning 202 (native memory is freed as they finish either way; the model is unroutable immediately). ?wait=false returns 202 at once")
	fs.StringVar(&cfg.spec, "spec", "", "speculative decoding: \"\" (off) | ngram — lossless n-gram (prompt-lookup) drafting with adaptive depth. Wins on copy-heavy traffic (code edits / RAG / agent loops) on the CPU and Metal backends; output is identical (greedy bit-exact, sampled in-distribution incl. temperature/top-k/p/min-p + repetition penalties + logit bias). On greedy constrained/tool requests (response_format / tool grammar) it switches to grammar-fused drafting — the grammar's forced bytes are drafted for free, fused with the n-gram source. Auto-falls back to plain decode per-request when the sampler isn't yet supported on the spec path (e.g. constrained + temperature>0) goinfer-chat has the same --spec ngram, and additionally --draft (a separate small draft model).")
	fs.BoolVar(&cfg.specAdaptive, "spec-adaptive", false, "Experimental, opt-in: with -spec ngram on a resident whose decode can batch, keep concurrent generations instead of forcing one at a time — a generation speculates only while it is alone, joins the batched decode at a round boundary when others arrive, and resumes speculating after 8 consecutive rounds alone. Measured cost under load (CUDA, 4 clients, docs/measurements/mc4-candidate-cuda-2026-10-01.md): 0.858x batching alone on chat traffic, and 0.80x plain -spec ngram on copy traffic; alone it is faster than batching (copy 2.13x, chat 1.26x). No effect without -spec ngram, or on a resident that cannot batch.")
	fs.StringVar(&cfg.embedPath, "embed-model", "", "embedding model for /v1/embeddings: a CodeRankEmbed/NomicBert HF dir (config.json + model.safetensors + tokenizer.json), a decoder-as-embedder .gguf, or a HuggingFace reference: hf:<owner>/<repo>:safetensors fetches a NomicBert encoder checkpoint (anything else is refused before the weights), hf:<owner>/<repo>:<quant> a GGUF")
	fs.StringVar(&cfg.embedQuant, "embed-quant", "f32", "embedding weight precision: f32 | q8")
	fs.StringVar(&cfg.embedName, "embed-served-model-name", "", "embedding model id reported by /v1/models (default: dir basename)")
	fs.StringVar(&cfg.embedResize, "embed-image-resize", "", "image resize before an image-embedding model's tower (EmbeddingGemma 2): bilinear | bicubic (the reference processor's torchvision antialiased bicubic). Default: the model's")
	return sf
}

// validate checks the settings that do not depend on how the server is reached: the session directory, the
// speculation and tiering choices, the thinking and tool formats, and the decision-route calibration. The listener's
// own checks (the address, the API key, TLS) stay with the caller.
func (cfg *config) validate() error {
	if err := sessionDirOK(cfg.sessionDir); err != nil {
		return err
	}
	if cfg.kvShare != "" && cfg.kvShare != "on" && cfg.kvShare != "off" {
		return fmt.Errorf("-kv-share must be on or off (got %q)", cfg.kvShare)
	}
	if _, _, err := parseKVHostBytes(cfg.kvHostBytesFlag); err != nil {
		return err
	}
	if cfg.spec != "" && cfg.spec != "ngram" {
		return fmt.Errorf("-spec must be \"\" or \"ngram\" (got %q)", cfg.spec)
	}
	if cfg.kvIdleDemote > 0 && (cfg.sessionDir == "" || cfg.kvSessions <= 0) {
		return errors.New("-kv-idle-demote needs -session-dir and -kv-sessions > 0")
	}
	if cfg.drafterWidth != 0 && cfg.drafter == "" {
		return fmt.Errorf("-drafter-width %d without -drafter: the width applies only to a block drafter's verify", cfg.drafterWidth)
	}
	if cfg.drafterWidth < 0 || (cfg.drafterWidth > 0 && cfg.drafterWidth < minDrafterWidth) {
		return fmt.Errorf("-drafter-width %d: want 0 (the default) or at least %d: under %d a round commits too few tokens for the "+
			"acceptance guard's bar, so most turns fall back to plain decode", cfg.drafterWidth, minDrafterWidth, minDrafterWidth)
	}
	if cfg.maxFileStoreBytes < 0 {
		return fmt.Errorf("-max-file-store-bytes %d: want a size in bytes, or 0 for no limit on bytes", cfg.maxFileStoreBytes)
	}
	if _, ok := chat.ParseThinkMode(cfg.thinking); !ok {
		return fmt.Errorf("-thinking %q: want asis, template, on or off", cfg.thinking)
	}
	if _, ok := chat.ParseToolFormat(cfg.toolFormat); !ok {
		return fmt.Errorf("-tool-format %q: want hermes or template", cfg.toolFormat)
	}
	if _, err := parseBudgetFlag(cfg.reasoningBudget); err != nil {
		return fmt.Errorf("-reasoning-budget %q: %w", cfg.reasoningBudget, err)
	}
	if _, ok := parseReasoningFormat(cfg.reasoningFmt); !ok {
		return fmt.Errorf("-reasoning-format %q: want deepseek, deepseek-legacy or none", cfg.reasoningFmt)
	}
	if cfg.decisionsTemplate != decide.TemplateChat && cfg.decisionsTemplate != decide.TemplateBare {
		return fmt.Errorf("-decisions-template %q: want %s or %s", cfg.decisionsTemplate, decide.TemplateChat, decide.TemplateBare)
	}
	if cfg.decisionsCal != "" {
		// Fail at startup, not on the first /v1/systemone request: a bad file, or one fitted under the other template.
		cal, err := decide.LoadCalibration(cfg.decisionsCal)
		if err == nil && cal.Template != "" && cal.Template != cfg.decisionsTemplate {
			err = fmt.Errorf("fitted under template %q, not -decisions-template %q", cal.Template, cfg.decisionsTemplate)
		}
		if err != nil {
			return fmt.Errorf("-decisions-calibration %s: %w", cfg.decisionsCal, err)
		}
	}
	return nil
}

// kvShareOffFor is whether requests may not take KV rows another conversation computed (decoder.Options.KVShareOff):
// when -kv-share is off, and when the server answers the network without an API key. serve has one key, so every
// request that passes it is one sharing scope. Without a key
// only loopback may be bound, but a loopback bind answering an -allowed-hosts name is a reverse proxy's public
// face, and is treated as the network.
func kvShareOffFor(kvShare, apiKey, addr, allowedHosts string) bool {
	if kvShare == "off" {
		return true
	}
	return apiKey == "" && (!addrIsLoopback(addr) || strings.TrimSpace(allowedHosts) != "")
}
