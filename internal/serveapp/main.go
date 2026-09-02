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
package serveapp

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
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
	"github.com/townsendmerino/goinfer/internal/giw"
	"github.com/townsendmerino/goinfer/internal/prequant"
	"github.com/townsendmerino/goinfer/multimodal"
	"github.com/townsendmerino/goinfer/tokenizer"
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
	if cfg.quantSet {
		return cfg.quant
	}
	return ""
}

// options resolves this spec's overrides over the server-global defaults in cfg
// into a decoder.Options. Backend is process-wide (GPU device init), never per-model.
func (s modelSpec) options(cfg config) decoder.Options {
	return decoder.Options{
		Backend:          cfg.backend,
		Quant:            orStr(s.quant, cfg.quant),
		LoRA:             orStr(s.lora, cfg.lora),
		KVPrecision:      orStr(s.kvPrec, cfg.kvPrec),
		MoECacheExperts:  cfg.moeCacheExperts,
		MoECacheSlots:    cfg.moeCacheSlots,
		KVQuant:          orStr(s.kvQuant, cfg.kvQuant),
		StreamWeights:    orBool(s.stream, cfg.streamWeights),
		WeightCacheBytes: int64(orFloat(s.weightCache, cfg.weightCacheGB) * 1e9),
		EmbedInt4:        orBool(s.embedInt4, cfg.embedInt4),
		ResidentContext:  orInt(s.ctxSize, cfg.ctxSize),
	}
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
	backend  string
	// quant + the per-model knobs below are server-global DEFAULTS; a --model spec
	// can override each one (see modelSpec / modelFlag.Set).
	quant            string
	quantSet         bool   // was --quant given on the CLI? (vs the "int4" default) — for the .giw explicit-quant check (T1-7)
	kvPrec           string // GPU residency KV cache precision: "" | f32 | f16 (-kv)
	moeCacheExperts  bool   // stream routed MoE experts host→VRAM (--moe-cache-experts)
	moeCacheSlots    int    // per-layer expert slot REQUEST (--moe-cache-slots); an upper bound, 0 = built-in default
	metalFastPrefill bool   // opt into Metal's batched (non-bit-identical) prompt prefill (--metal-fast-prefill)
	cpuFastAttention bool   // CPU f32 prefill attention, DEFAULT ON (non-bit-identical) (--cpu-fast-attention)
	cpuExactPrefill  bool   // opt OUT of the above: bit-exact prefill (--cpu-exact-prefill)
	ctxSize          int    // -ctx: requested GPU-resident KV capacity in positions (0 = backend default). Effective cap = min(model context window, this)
	kvQuant          string // CPU KV cache storage precision: "" | f32 | i8 (-kv-quant)
	lora             string
	name             string // -served-model-name (applies only to a single unnamed --model)
	kvSessions       int
	sessionDir       string        // -session-dir (also where /admin unload snapshots warm KV)
	kvIdleDemote     time.Duration // -kv-idle-demote: tiered KV — demote a session idle this long to disk (0 = off)
	kvDemotedMax     int           // -kv-demoted-max: cap on the on-disk cold tier
	streamWeights    bool          // -stream-weights: page MoE expert weights out of an mmap'd .giw under a RAM budget
	weightCacheGB    float64       // -weight-cache: resident expert-weight budget in GB (0 = auto)
	embedInt4        bool          // -embed-int4: relax the int8 embed/head pin to int4 (lossy, big-vocab small models)
	maxQueue         int           // -max-queue: bounded per-model queue depth (0 = unbounded)
	maxInflight      int           // -max-inflight: global cap on concurrent inference handlers (bounds pre-queue work; 0 = unbounded)
	maxBodyBytes     int64         // -max-body-bytes: request-body cap (0 = derive from the model's context window)
	unloadDrainWait  time.Duration // -unload-drain-wait: how long an unload waits for in-flight requests to drain before 202 (native free continues detached)
	spec             string        // -spec: "" (off) | "ngram" — lossless n-gram speculative decode
	drafter          string        // -drafter: dir of a pretrained BLOCK drafter (DFlash); resident GPU backends only
	allowAdmin       bool          // -allow-admin: enable POST /admin/models/{load,unload}
	requireBE        bool          // -require-backend: refuse to start when a model silently fell back off the requested backend's fast paths (resident decode / batched prefill)
	visionPath       string        // -vision: dir holding the vision tower (SigLIP + projector) for a multimodal --model
	visionQuant      string        // -vision-quant: "f32" (default) | "int8" (W8A8; only faster on AVX512-VNNI — a WASH on AVX2)

	embedPath  string // encoder (-embed-model); "" = no /v1/embeddings
	embedQuant string // "" | f32 | q8
	embedName  string // -embed-served-model-name
}

// cpuFastAttentionHelp is --cpu-fast-attention's usage text, held as a const so a
// test can assert that the trade is actually disclosed. The precedent
// (--metal-fast-prefill) exists precisely so a divergence is "disclosed in
// --help, not something a user has to already know to type" — and a help string
// nothing checks is one edit away from quietly losing the disclosure.
// cpuExactPrefillHelp is the opt-OUT half. Held as a const for the same reason as
// cpuFastAttentionHelp: the test asserts on the text, so the disclosure cannot silently drift
// away from the behaviour.
const cpuExactPrefillHelp = "force BIT-EXACT prompt ingestion: use the f64-accumulating attention kernel for prefill instead of the f32 one that is now the default. Costs the speed --cpu-fast-attention buys (measured 2.28x slower prefill on an 8k prompt, dense 1.5B) and buys back decode==prefill bit-identity, so a long-prompt response is reproducible against a build from before f32 prefill became the default. Use it when you are diffing outputs across versions, reproducing a bug report, or anything where 'same prompt, same tokens' matters more than time-to-first-token. Wins over --cpu-fast-attention if both are given. CPU backend only; MoE models take the same f32 path as dense ones — the exclusion was measured and dropped in 66d0a05, so this is the only way to get bit-exact prefill for them too"

const cpuFastAttentionHelp = "DEFAULT ON since 2026-08-31 (pass --cpu-exact-prefill to turn it off). Compute PROMPT attention in f32 instead of the f64-accumulating kernel — measured 2.28x faster prefill on an 8k prompt (dense 1.5B, M1 Pro: 602.9s to 264.6s) because attention is ~70% of a long prefill and the f64 path is ~8x slower than f32 at those shapes. NOT bit-identical: measured cosine 0.9976 against the default (stable across 256/1024/2048-token prompts), so a long-prompt response CAN differ from what you would get with this off, even at temperature 0. Decode is unaffected — this changes only how the prompt is ingested. Speculative decoding is never affected (its verify pass always uses the exact kernel, or verify would stop matching greedy). Applies to MoE models too: the old REFUSAL was dropped in 66d0a05 after being measured (1-cosine 2.126e-3 for MoE against 2.400e-3 for the dense case, depth-matched, with a 48/48 identical greedy continuation), and this help text went on claiming it for two days afterwards. FLOORED AT 512 PROMPT TOKENS: below that the exact kernel runs regardless, because the win scales with prompt length and the divergence does not — an 8-token prompt diverged at the third generated token while buying nothing (1.15x at 512, 1.43x at 2048, 2.28x at 8192). CPU backend only"

func Main() {
	var (
		cfg     config
		addr    = flag.String("addr", "127.0.0.1:8080", "listen address (defaults to loopback; use 0.0.0.0:8080 to expose, and set -api-key and -tls-cert/-tls-key — or put a TLS-terminating reverse proxy in front — when you do)")
		apiKey  = flag.String("api-key", "", "optional shared secret; when set, every request must send it as `Authorization: Bearer <key>` or `x-api-key: <key>`. Falls back to $GOINFER_API_KEY. REQUIRED with -allow-admin.")
		tlsCert = flag.String("tls-cert", "", "PEM certificate file; with -tls-key, serves HTTPS instead of plaintext HTTP. Without it, -api-key and every prompt/completion travel in cleartext — fine on loopback, not on a shared network. For ACME/auto-renewal, put a reverse proxy (Caddy, nginx, Traefik) in front instead and leave this unset.")
		tlsKey  = flag.String("tls-key", "", "PEM private key file, paired with -tls-cert")
	)
	flag.StringVar(&cfg.sessionDir, "session-dir", "", "optional dir to persist/restore KV sessions across restarts (.giw-kv snapshots)")
	flag.BoolVar(&cfg.allowAdmin, "allow-admin", false, "enable POST /admin/models/{load,unload} (loads attacker-named paths — deliberate opt-in; requires -api-key)")
	flag.StringVar(&cfg.visionPath, "vision", "", "vision tower dir (SigLIP encoder + projector) for a multimodal --model; enables image content parts. Defaults to the --model dir when it contains a vision tower")
	flag.StringVar(&cfg.visionQuant, "vision-quant", "f32", "vision encoder weight quant: f32 (default, bit-exact) | int8 (W8A8, cosine ~0.999) — int8 only speeds the compute-bound ViT prefill on AVX512-VNNI; on AVX2 it's a wash, so f32 is the default")
	flag.Var(&cfg.models, "model", "generative model: a .gguf/.giw file or HF dir (chat/completions). Repeatable\n"+
		"as `name=path` to serve a model zoo from one process; requests route on the\n"+
		"OpenAI `model` field. Append comma-separated per-model overrides of the global\n"+
		"defaults below: `--model big=moe.giw,stream,weight-cache=16 --model fast=small.giw`\n"+
		"streams only the big MoE. Keys: quant,lora,kv,kv-quant,stream,weight-cache,embed-int4.\n"+
		"(Paths may not contain commas.)")
	flag.StringVar(&cfg.drafter, "drafter", "", "directory of a pretrained BLOCK drafter (z-lab DFlash) paired with --model: the drafter proposes a whole block of tokens per round and the target verifies them in ONE batched pass, measured 1.6-1.8x on code/math and ~0.96x on open chat (docs/spec/08). LOSSLESS — every emitted token is one the target's own argmax produced, so output is identical to plain greedy. Greedy only: a request with temperature, penalties or logit bias falls back to normal decoding automatically. Requires a resident GPU backend (--backend cuda); declines with a reason otherwise")
	flag.StringVar(&cfg.backend, "backend", "cpu", "compute backend: cpu | webgpu | cuda | metal (process-wide; cuda/metal: dense-only, cgo-free native, -tags cuda|metal)")
	flag.StringVar(&cfg.quant, "quant", "int4", "default decoder weight quant — the accuracy/speed/RAM tradeoff (per-model override: --model name=path,quant=…):\n"+
		"  int4      W4A8 (int4 weights, int8 activations): smallest, and fastest on every backend including\n"+
		"            Apple Silicon CPU (measured M1 Pro, goinfer a11c56b 2026-08-24, docs/benchmarks.md: at or\n"+
		"            above int8int8's decode rate at half the RAM -- an earlier reading had int8int8 ~60%\n"+
		"            faster on Apple Silicon CPU, which was correct at the time but diagnosed a since-fixed LM\n"+
		"            head, not the W4A8 kernel; see docs/task-w4a8-neon-bandwidth.md). Lossier than int8\n"+
		"            (4-bit weights). THE DEFAULT.\n"+
		"  int4mix   attn int8 + FFN int4 (GGUF only): near-int8 quality at below-int8 RAM.\n"+
		"  int8int8  W8A8 (int8 weights + int8 activations, native SDOT): higher accuracy, ~2x the RAM of int4.\n"+
		"            REQUIRED for --backend metal (int4 declines to CPU on the dense Metal resident path).\n"+
		"  int8      int8 weights with wider activations: between int8int8 and native.\n"+
		"  \"\"        native (no quantization, f32): most accurate, largest, slowest.\n"+
		"All quantized modes (int4/int4mix/int8/int8int8) get batched CUDA prefill (fast TTFT); only native f32\n"+
		"falls back to the ~9x slower sequential prefill. A prequantized .giw model carries its own baked-in quant.")
	flag.BoolVar(&cfg.requireBE, "require-backend", false, "strict mode: exit non-zero at startup if a model did not resolve to the requested --backend's fast paths — no resident decode path, or a prefill that declined to the sequential per-token loop (e.g. int8int8 on cuda, ~9× slower TTFT). Both fall back silently by design; a batch client should fail at second zero instead of discovering it under load")
	flag.BoolVar(&cfg.moeCacheExperts, "moe-cache-experts", false, "run a MoE model whose experts EXCEED VRAM: routed experts stream host→VRAM per token instead of being held resident, so every expert still executes on the GPU (no CPU offload). Costs a per-token PCIe transfer; bit-identical to fully-resident. Off by default — with it off, a model that doesn't fit declines to the CPU path and says why. CUDA only")
	flag.BoolVar(&cfg.metalFastPrefill, "metal-fast-prefill", false, "batch the WHOLE prompt through Metal's f16-MMA prefill kernel instead of ingesting it one token at a time — measured 3.9-4.6x faster time-to-first-token on long prompts (a 2048-token prompt: ~51s to ~13s). NOT bit-identical to sequential/CPU decode: the f16-MMA activation path diverges from decode's int8 path on ~54% of runs, so the FIRST FEW TOKENS of a response can differ from what you would get with this off, even at temperature 0. Off by default — decode itself is unaffected either way; this only changes how the prompt is ingested. Metal backend only")
	flag.BoolVar(&cfg.cpuFastAttention, "cpu-fast-attention", true, cpuFastAttentionHelp)
	flag.BoolVar(&cfg.cpuExactPrefill, "cpu-exact-prefill", false, cpuExactPrefillHelp)
	flag.IntVar(&cfg.moeCacheSlots, "moe-cache-slots", 0, "per-layer expert slots for --moe-cache-experts: request AT MOST this many. The runtime measures free VRAM and lowers it if the request does not fit, logging what it chose (\"C′ cache: … capping to N\"), so this is an upper bound and not a value you have to get right. 0 keeps the built-in default. More slots ⇒ higher LRU hit rate ⇒ fewer per-token transfers, at VRAM cost")
	flag.StringVar(&cfg.kvPrec, "kv", "f32", "GPU residency KV cache precision: f32 (bit-exact, 16k ctx) | f16 (lossy, 32k ctx) | i8 (lossy, ~64k ctx) — webgpu backend only")
	flag.IntVar(&cfg.ctxSize, "ctx", 0, "GPU-resident KV capacity in positions (per-model override: --model name=path,ctx=…). 0 (default) keeps the backend default of 4096, so nothing you did not ask for allocates deep-KV VRAM. When set, the effective cap is min(model context window, this) and the KV it implies is VRAM-checked AT LOAD — the server refuses to start, naming the GB, rather than OOM mid-decode. A request past the cap still fails cleanly and falls back to the staged path")
	flag.StringVar(&cfg.kvQuant, "kv-quant", "f32", "CPU KV cache storage: f32 (default, bit-exact) | i8 (per-head int8, ~4× smaller, lossy — argmax ~90%+; excludes MoE/gemma4/qwen3.5)")
	flag.StringVar(&cfg.lora, "lora", "", "optional PEFT LoRA adapter dir, merged into the (safetensors) base at load")
	flag.Var(&cfg.adapters, "adapter", "compute-time LoRA adapter sharing a base model's resident weights: `serveName=baseName=dir`.\n"+
		"Repeatable. Unlike --lora (merged, one base per fine-tune), N adapters of one base cost ~base + N\n"+
		"low-rank deltas — request the fine-tune via the OpenAI `model` field. Base must be a safetensors\n"+
		"--model (dense, gated MLP; not MoE/gemma4/qwen3.5). Incompatible with --stream-weights.")
	flag.StringVar(&cfg.name, "served-model-name", "", "served id for a single unnamed --model (default: file/dir basename)")
	flag.IntVar(&cfg.kvSessions, "kv-sessions", 4, "number of conversations to keep prefilled in RAM for prompt-prefix KV reuse (0 disables)")
	flag.DurationVar(&cfg.kvIdleDemote, "kv-idle-demote", 0, "tiered KV: demote a warm session's KV to -session-dir once it's been idle this long, faulting it back on the next matching request (e.g. 10m; 0 = off). Lets a small-RAM box serve many intermittent chats. Needs -session-dir and -kv-sessions > 0")
	flag.IntVar(&cfg.kvDemotedMax, "kv-demoted-max", 64, "tiered KV: max demoted (on-disk) sessions to keep; older ones are dropped (only with -kv-idle-demote)")
	flag.BoolVar(&cfg.streamWeights, "stream-weights", false, "page model weights on demand out of an mmap'd .giw, capping resident RAM to -weight-cache instead of holding all weights: MoE expert demand-paging (run a 35B-A3B on ~16-20 GB) or dense per-layer streaming (run a model bigger than RAM). Bit-exact; trades RAM for fault latency. A plain .gguf is transparently transcoded to a sidecar .giw cache on first use (one-time)")
	flag.Float64Var(&cfg.weightCacheGB, "weight-cache", 0, "resident expert-weight budget in GB for -stream-weights (0 = auto, ~half of available RAM)")
	flag.BoolVar(&cfg.embedInt4, "embed-int4", false, "with -quant int4, store the token-embedding/LM-head table at int4 too instead of the int8 pin — halves the largest resident tensor on a big-vocab small model. Lossy (~2.3 pts top-1, mostly rare tokens); GGUF direct load only (not the -stream-weights .giw cache)")
	flag.IntVar(&cfg.maxQueue, "max-queue", 8, "per-model backpressure: max queued requests before 429 (0 = unbounded)")
	flag.IntVar(&cfg.maxInflight, "max-inflight", 128, "global cap on concurrent inference requests, bounding the pre-queue stage (JSON+image decode, tokenization, template render, vision Forward) that runs before the per-model queue; a full cap returns 503 Retry-After (0 = unbounded)")
	flag.Int64Var(&cfg.maxBodyBytes, "max-body-bytes", 0, "cap on request body size in bytes; a larger body is rejected 413 before it is read. 0 = derive from the model's context window (a body that could never fit is rejected up front). The vision endpoints get at least 32 MiB on top for base64 image data")
	flag.DurationVar(&cfg.unloadDrainWait, "unload-drain-wait", 5*time.Second, "how long POST /admin/models/unload waits for in-flight requests to drain before returning 202 (native memory is freed as they finish either way; the model is unroutable immediately). ?wait=false returns 202 at once")
	flag.StringVar(&cfg.spec, "spec", "", "speculative decoding: \"\" (off) | ngram — lossless n-gram (prompt-lookup) drafting with adaptive depth. Wins on copy-heavy traffic (code edits / RAG / agent loops) on the CPU backend; output is identical (greedy bit-exact, sampled in-distribution incl. temperature/top-k/p/min-p + repetition penalties + logit bias). On greedy constrained/tool requests (response_format / tool grammar) it switches to grammar-fused drafting — the grammar's forced bytes are drafted for free, fused with the n-gram source. Auto-falls back to plain decode per-request when the sampler isn't yet supported on the spec path (e.g. constrained + temperature>0)")
	flag.StringVar(&cfg.embedPath, "embed-model", "", "embedding model: a CodeRankEmbed HF dir (config.json + model.safetensors + tokenizer.json) for /v1/embeddings")
	flag.StringVar(&cfg.embedQuant, "embed-quant", "f32", "embedding weight precision: f32 | q8")
	flag.StringVar(&cfg.embedName, "embed-served-model-name", "", "embedding model id reported by /v1/models (default: dir basename)")
	flag.Parse()
	// --metal-fast-prefill sets the internal gate metal/backend.go's PrefillLast already checks
	// (GOINFER_METAL_BATCHED_PREFILL) — reusing the existing, already-tested decline/opt-in
	// machinery rather than threading a new decoder.Options field through frozen core for what is
	// fundamentally a startup-time choice. Setting it when --backend isn't metal is harmless (the
	// var is read nowhere else); scoped to a flag rather than left as an internal-only env var so
	// the tradeoff is disclosed in --help, not something a user has to already know to type.
	if cfg.metalFastPrefill {
		os.Setenv("GOINFER_METAL_BATCHED_PREFILL", "1")
	}
	// Same disclosure argument as above: the decoder reads GOINFER_CPU_FAST_ATTENTION,
	// and a divergence a user opts into should be spelled out in --help rather than
	// discoverable only by reading the source. The MoE refusal and the
	// speculative-verify exclusion are enforced in the decoder, not here, so they
	// hold however the env var arrives.
	// DEFAULT ON, so the env is set EXPLICITLY either way rather than left unset. The decoder
	// treats unset as on, but an inherited GOINFER_CPU_FAST_ATTENTION from the caller's
	// environment would otherwise outrank the flags — the server's own flags must win over
	// whatever the shell happened to export.
	//
	// --cpu-exact-prefill wins over --cpu-fast-attention when both are given: between a speed
	// request and a correctness request, the correctness one is the safe way to resolve a
	// contradiction the user did not realise they had expressed.
	if cfg.cpuExactPrefill || !cfg.cpuFastAttention {
		os.Setenv("GOINFER_CPU_FAST_ATTENTION", "0")
	} else {
		os.Setenv("GOINFER_CPU_FAST_ATTENTION", "1")
	}
	// Was --quant given, or is cfg.quant the "int4" default? The .giw explicit-quant check (T1-7)
	// must fire only on an explicit request — the default must not "mismatch" a non-int4 bundle.
	flag.Visit(func(f *flag.Flag) {
		if f.Name == "quant" {
			cfg.quantSet = true
		}
	})
	if len(cfg.models) == 0 && cfg.embedPath == "" && !cfg.allowAdmin {
		fmt.Fprintln(os.Stderr, "error: need at least one of --model, --embed-model, or --allow-admin")
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
	textCap, visionCap, embedCap := srv.resolveBodyCaps(cfg.maxBodyBytes)
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/models", auth(srv.handleModels))
	// Operator surface for the resolved compute paths — same fields as the /v1/models vendor
	// extension, on a payload with no OpenAI-schema contract to break. See handleHealth.
	mux.HandleFunc("GET /health", auth(srv.handleHealth))
	if len(srv.models) > 0 {
		mux.HandleFunc("POST /v1/chat/completions", auth(inf(maxBytes(visionCap, srv.handleChat))))
		mux.HandleFunc("POST /v1/completions", auth(inf(maxBytes(textCap, srv.handleCompletions))))
		mux.HandleFunc("POST /v1/responses", auth(inf(maxBytes(textCap, srv.handleResponses))))
		mux.HandleFunc("POST /v1/messages", auth(inf(maxBytes(visionCap, srv.handleMessages))))
		mux.HandleFunc("POST /v1/messages/count_tokens", auth(inf(maxBytes(textCap, srv.handleCountTokens))))
	}
	// Registered unconditionally (G7): with no embedding model, handleEmbeddings returns a JSON
	// error naming -embed-model rather than a bare 404, so an SDK sees "unconfigured" not "wrong URL".
	mux.HandleFunc("POST /v1/embeddings", auth(inf(maxBytes(embedCap, srv.handleEmbeddings,
		// The per-input / per-batch bounds are per-DIMENSION and multiply out past any body cap, so
		// name all three: a client within both per-dimension limits can still exceed the total.
		fmt.Sprintf("this route also limits each request to %d inputs of at most %d bytes each; "+
			"the body cap bounds their total", maxEmbedInputs, maxEmbedInputBytes)))))
	mux.HandleFunc("POST /admin/models/load", auth(maxBytes(textCap, srv.handleAdminLoad)))
	mux.HandleFunc("POST /admin/models/unload", auth(maxBytes(textCap, srv.handleAdminUnload)))

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
	// but never cancels it, so it runs past the 30s timeout still holding lm.mu and the checkpoint loop
	// below deadlocks on lm.mu.Lock() forever (audit C-22).
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
		// A SECOND signal during the drain force-exits instead of being swallowed by the buffered
		// channel — so Ctrl-C twice always kills the server, not only SIGKILL (audit C-22).
		go func() {
			<-sig
			fmt.Fprintln(os.Stderr, "second signal — forcing exit")
			os.Exit(1)
		}()
		close(stopDemote) // stop demoting before we checkpoint
		srvCancel()       // cancel in-flight generations (via BaseContext) so they release lm.mu
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		_ = httpSrv.Shutdown(ctx)
		if cfg.sessionDir != "" && cfg.kvSessions > 0 {
			deadline := time.Now().Add(5 * time.Second)
			for _, lm := range srv.modelList() {
				// TryLock with a deadline: srvCancel above should have freed lm.mu, but a generation
				// mid-forward (not yet at a ctx check) could still hold it — skip its checkpoint rather
				// than deadlock the whole shutdown (audit C-22).
				if !tryLockUntil(&lm.mu, deadline) {
					fmt.Fprintf(os.Stderr, "shutdown: model %q still busy — skipping its session checkpoint\n", lm.name)
					continue
				}
				_ = lm.sessions.save(sessionSubdir(cfg.sessionDir, lm.fp))
				lm.sessions.removeColdFiles() // the cold tier is in-process; clear its scratch
				lm.mu.Unlock()
			}
		}
	}()

	useTLS := *tlsCert != ""
	scheme := "http"
	if useTLS {
		scheme = "https"
	}
	fmt.Fprintf(os.Stderr, "goinfer serving on %s://%s [%s]\n", scheme, *addr, srv.endpointSummary())
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
	fmt.Fprintf(os.Stderr, "request body cap: %s (text) / %s (vision) / %s (embeddings) [%s]\n", humanBytes(textCap), humanBytes(visionCap), humanBytes(embedCap), capSrc)
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
	s := &server{
		models:    map[string]*loadedModel{},
		liveness:  map[*decoder.Model]*modelLiveness{},
		draining:  map[string]struct{}{},
		cfg:       cfg,
		responses: newResponseStore(256),
	}
	for _, spec := range cfg.models {
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
		if err := s.loadEncoder(cfg); err != nil {
			return nil, err
		}
	}
	if err := s.loadVisionTower(cfg); err != nil {
		return nil, err
	}
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
		if cfg.streamWeights {
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
			spec:     cfg.spec == "ngram",
			sessions: newSessionLRU(base.model, cfg.kvSessions, 0, fp),
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

// loadVisionTower attaches a SigLIP encoder + projector to the (single) loaded
// model, making it vision-capable (serve then accepts image content parts). The
// dir is -vision if set, else the sole --model's own dir when it carries a vision
// tower (auto-discovery). A multimodal tower only makes sense for a single model,
// so it errors if -vision is set with a model zoo. Absent a tower it is a no-op:
// text-only serving is unchanged.
func (s *server) loadVisionTower(cfg config) error {
	dir := cfg.visionPath
	if dir == "" {
		// Auto-discover: a single --model dir that holds a vision tower — either the
		// Gemma 3 projector or a Qwen2.5-VL checkpoint (its ViT lives in the same dir).
		if len(cfg.models) == 1 {
			cand := cfg.models[0].path
			if fi, err := os.Stat(cand); err == nil && fi.IsDir() {
				if visionModelType(cand) == "qwen2_5_vl" {
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
	if len(s.models) != 1 {
		return fmt.Errorf("-vision needs exactly one --model (got %d)", len(s.models))
	}
	// The resident GPU encoder needs int8 (W8A8) matmul weights, so --backend
	// webgpu implies an int8 tower even if --vision-quant wasn't set.
	int8Tower := cfg.visionQuant == "int8" || cfg.backend == "webgpu"
	if visionModelType(dir) == "qwen2_5_vl" {
		return s.loadQwenVisionTower(dir, int8Tower)
	}
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

// loadQwenVisionTower attaches the Qwen2.5-VL ViT (aikit) to the single loaded
// model. Unlike Gemma 3 there's no separate projector — the merger is in the
// encoder; preprocessing + m-RoPE are Qwen-specific (the image path branches on
// qwenEnc). Image placeholders use <|image_pad|>, expanded per image to the merged
// patch count.
func (s *server) loadQwenVisionTower(dir string, int8Tower bool) error {
	enc, err := vision.LoadQwenVisionEncoder(dir, int8Tower)
	if err != nil {
		return fmt.Errorf("load qwen2.5-vl vision encoder (%s): %w", dir, err)
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
		fmt.Fprintf(os.Stderr, "loaded Qwen2.5-VL vision tower for %q (merge %d, image-pad id %d) from %s\n", lm.name, lm.qwenMerge, lm.qwenImgTok, dir)
	}
	return nil
}

// loadDecoder loads one generative model + tokenizer, resolves its chat template,
// and returns it as a *loadedModel. The served name is the spec's name=, else (a
// single unnamed --model) --served-model-name, else the file/dir basename.
func loadDecoder(ctx context.Context, spec modelSpec, cfg config) (*loadedModel, error) {
	// Resolve this model's knobs (per-model overrides over server-global defaults)
	// and reject invalid enums up front.
	opts := spec.options(cfg)
	if err := opts.Validate(); err != nil {
		return nil, fmt.Errorf("--model %q: %w", spec.path, err)
	}

	// Weight streaming needs the read-only mmap that only .giw provides. For a plain
	// .gguf, transparently transcode to a sidecar .giw cache once (idea #1 "D") and
	// load that — so --stream-weights "just works" without a manual prequant step.
	// The served name still derives from the original --model spec, not the cache.
	loadPath := spec.path
	if opts.StreamWeights && strings.HasSuffix(spec.path, ".gguf") {
		if opts.EmbedInt4 {
			fmt.Fprintln(os.Stderr, "note: embed-int4 is ignored with stream-weights (the cached .giw keeps the int8 pin); prequant the model with embed-int4 to bake it")
		}
		giwPath, err := prequant.EnsureCachedGIW(ctx, spec.path, opts.Quant)
		if err != nil {
			return nil, fmt.Errorf("stream-weights cache (%s): %w", spec.path, err)
		}
		loadPath = giwPath
	}

	tk, err := loadDecoderTokenizer(loadPath)
	if err != nil {
		return nil, fmt.Errorf("load tokenizer (%s): %w", loadPath, err)
	}
	t0 := time.Now()
	model, err := decoder.Load(loadPath, opts)
	if err != nil {
		return nil, fmt.Errorf("load model (%s): %w", loadPath, err)
	}
	// A prequant .giw carries its own quant, so --quant cannot re-quantize it. If the user
	// explicitly asked for a different one, fail here (before binding) rather than silently
	// serving the baked precision (T1-7). A bare default never conflicts (explicitQuant == "").
	if err := model.CheckGiwQuantMatch(spec.explicitQuant(cfg)); err != nil {
		model.Close()
		return nil, fmt.Errorf("--model %q: %w", spec.path, err)
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
		spec: cfg.spec == "ngram",
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
	if cfg.drafter != "" {
		if err := attachBlockDrafter(lm, cfg.drafter); err != nil {
			return nil, fmt.Errorf("--drafter %q: %w", cfg.drafter, err)
		}
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
	// State the RESOLVED paths, not the requested ones. Both the resident decode path and the batched
	// prefill are optional capabilities that fall back silently — a model can load clean, report a GPU
	// backend, and still take one forward per prompt token (cuda int8int8: ~9× TTFT). Printing them at
	// load is what makes that visible; -require-backend turns a decline into a startup failure.
	batched, why := lm.model.PrefillPath()
	fmt.Fprintf(os.Stderr, "  decode path: %s\n  prefill path: %s\n", lm.model.DecodePath(), why)
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
	if cfg.backend != "cpu" && !lm.model.ResidentActive() {
		// Covers the whole ladder, not just an ineligible arch: an untagged build (`--backend cuda`
		// without `-tags cuda` resolves to the CPU backend), a box with no usable device (the cuda
		// factory always succeeds — the driver is only touched at BuildResident), and a model shape
		// the runner refuses. All three otherwise present as a healthy server that is silently on CPU.
		reason := lm.model.ResidentDecline()
		if reason == "" {
			reason = "no reason recorded"
		}
		return fmt.Errorf("--require-backend: model %q did not build a resident decode path on backend %q: %s (resolved: %s)",
			name, cfg.backend, reason, lm.model.DecodePath())
	}
	if !batched {
		return fmt.Errorf("--require-backend: model %q declined the batched prefill: %s", name, why)
	}
	return nil
}

// loadDecoderTokenizer loads the tokenizer for a decoder model, picking the loader
// by extension: a prequant .giw carries its tokenizer as an embedded metadata-GGUF
// (read just that half — the weights are tens of GB and mmap'd by decoder.Load), a
// .gguf reads its own metadata, and anything else is a SentencePiece/HF dir.
func loadDecoderTokenizer(path string) (*tokenizer.Tokenizer, error) {
	switch {
	case strings.HasSuffix(path, ".giw"):
		tokBytes, err := giw.ReadTokFile(path)
		if err != nil {
			return nil, err
		}
		// The tok half is GGUF metadata for a GGUF-sourced bundle, or the raw
		// tokenizer.json for a safetensors-sourced one (prequant transcodeDir).
		if tk, err := tokenizer.LoadGGUFBytes(tokBytes); err == nil {
			return tk, nil
		}
		return tokenizer.LoadJSONBytes(tokBytes)
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
	// A causal decoder used as an embedder (qwen3-embedding / embeddinggemma) arrives as a .gguf
	// FILE; the aikit encoder path takes an HF directory. Dispatch on that rather than adding
	// another flag — see docs/completed/task-decoder-as-embedder.md.
	if fi, statErr := os.Stat(cfg.embedPath); statErr == nil && !fi.IsDir() {
		return s.loadDecoderEmbedder(cfg)
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
