// Package loadflags registers the model-loading flags goinfer-chat and goinfer-serve share, and turns
// their values into a decoder.Options. The two binaries have loaded through one path
// (internal/modelload) since 2026-09-24, but each still registered its own flags, and the sets
// drifted: chat had no --ctx, --stream-weights, --moe-cache-experts or --moe-cache-slots, and a
// cold-user run reached for --moe-cache-experts in chat and found nothing. A loading flag registered
// here exists in both binaries by construction. What stays app-specific is what does not describe how
// to load a model: serve's routes, queues, sessions and per-model zoo; chat's sampling and REPL.
package loadflags

import (
	"flag"
	"fmt"
	"os"
	"runtime"
	"strconv"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/cliutil"
)

// App is the binary registering the flags. A few help texts differ: serve's --model takes per-model
// overrides of some of these flags, and serve can refuse to start where chat just falls back.
type App int

const (
	Chat App = iota
	Serve
)

// Flags holds the shared loading flags' values once the flag set has been parsed.
type Flags struct {
	Backend         string              // --backend; "auto" (the default) is replaced by the backend it resolves to in Validate
	Auto            *decoder.AutoChoice // what -backend auto chose and why; nil when the user named a backend
	Quant           string              // --quant
	QuantSet        bool                // --quant was given on the command line, not left at its default
	LoRA            string              // --lora
	KV              string              // --kv
	Ctx             int                 // --ctx
	StreamWeights   bool                // --stream-weights
	WeightCacheGB   float64             // --weight-cache
	MoECacheExperts bool                // --moe-cache-experts
	MoECacheSlots   int                 // --moe-cache-slots
	MoEPager        string              // --moe-pager
	AcceptSlow      bool                // --accept-slow
	EmbedInt4       bool                // --embed-int4
	EmbedInt4Set    bool                // --embed-int4 was given on the command line, not left at its default
	DirectLoad      bool                // --direct-load
	Fit             bool                // --fit
	ExactPrefill    bool                // --exact-prefill
	CPUExactPrefill bool                // --cpu-exact-prefill
}

// Register adds the shared loading flags to fs and returns the Flags their values are parsed into.
func Register(fs *flag.FlagSet, app App) *Flags {
	f := &Flags{Quant: "int4", Fit: true}
	per := func(key string) string {
		if app != Serve {
			return ""
		}
		return " (per-model override: --model name=path," + key + "=…)"
	}
	fs.StringVar(&f.Backend, "backend", "auto", backendHelp)
	fs.Var(quantFlag{f}, "quant", "decoder weight quant — the accuracy/speed/RAM tradeoff"+per("quant")+":\n"+quantHelp)
	fs.StringVar(&f.LoRA, "lora", "", "optional PEFT LoRA adapter dir, merged into the (safetensors) base at load"+per("lora"))
	fs.StringVar(&f.KV, "kv", "f32", kvHelp+per("kv"))
	fs.IntVar(&f.Ctx, "ctx", 0, ctxHelp(app, per("ctx")))
	fs.BoolVar(&f.StreamWeights, "stream-weights", false, streamWeightsHelp+per("stream"))
	fs.Float64Var(&f.WeightCacheGB, "weight-cache", 0, "resident expert-weight budget in GB for -stream-weights (0 = auto, ~half of available RAM)"+per("weight-cache"))
	fs.BoolVar(&f.MoECacheExperts, "moe-cache-experts", false, moeCacheExpertsHelp)
	fs.IntVar(&f.MoECacheSlots, "moe-cache-slots", 0, moeCacheSlotsHelp)
	fs.StringVar(&f.MoEPager, "moe-pager", decoder.MoEPagerDefault(runtime.GOOS), moePagerHelp)
	fs.BoolVar(&f.AcceptSlow, "accept-slow", false, acceptSlowHelp)
	f.EmbedInt4 = true
	fs.Var(embedInt4Flag{f}, "embed-int4", embedInt4Help+per("embed-int4"))
	fs.BoolVar(&f.DirectLoad, "direct-load", os.Getenv("GOINFER_GGUF_DIRECT") != "", directLoadHelp)
	fs.Var((*cliutil.OnOff)(&f.Fit), "fit", fitHelp) // lenient: --fit=off works (M-14)
	fs.BoolVar(&f.ExactPrefill, "exact-prefill", false, ExactPrefillHelp)
	fs.BoolVar(&f.CPUExactPrefill, "cpu-exact-prefill", false, CPUExactPrefillHelp)
	return f
}

// Options is the decoder.Options the flags describe, before any per-model override (serve's
// --model name=path,key=… layers its own on top).
func (f *Flags) Options() decoder.Options {
	f.resolveAuto() // Validate has normally done it; embedInt4 below reads the resolved name
	return decoder.Options{
		Backend:          f.Backend,
		BackendAuto:      f.Auto != nil,
		Quant:            f.Quant,
		LoRA:             f.LoRA,
		KVPrecision:      f.KV,
		KVQuant:          CPUKV("", f.KV),
		ResidentContext:  f.Ctx,
		StreamWeights:    f.StreamWeights,
		WeightCacheBytes: int64(f.WeightCacheGB * 1e9),
		MoECacheExperts:  f.MoECacheExperts,
		MoECacheSlots:    f.MoECacheSlots,
		MoEPager:         f.MoEPager,
		AcceptSlowMoE:    f.AcceptSlow,
		EmbedInt4:        f.embedInt4(),
		DisableFit:       !f.Fit,
		ExactPrefill:     f.ExactPrefill,
		Knobs:            f.Knobs(),
	}
}

// Validate checks what decoder.Options.Validate does not: the flag values that only make sense as a
// choice from a list, or as a size. Call it after parsing, before Options().Validate().
func (f *Flags) Validate() error {
	// Resolve "auto" first: every check below and every caller after this reads Backend by name (the Metal
	// --embed-int4 default, serve's tower and -require-backend logic), and "auto" matches none of them. Each app
	// calls Validate right after parsing its flags, before anything reads Backend; Options resolves too, for a
	// caller that skipped Validate.
	f.resolveAuto()
	if f.MoEPager != "mmap" && f.MoEPager != "pool" {
		return fmt.Errorf("-moe-pager must be \"mmap\" or \"pool\" (got %q)", f.MoEPager)
	}
	if f.Ctx < 0 {
		return fmt.Errorf("-ctx %d: must be >= 0 (0 = the backend default)", f.Ctx)
	}
	if f.MoECacheSlots < 0 {
		return fmt.Errorf("-moe-cache-slots %d: must be >= 0 (0 = the built-in default)", f.MoECacheSlots)
	}
	if f.WeightCacheGB < 0 {
		return fmt.Errorf("-weight-cache %g: must be >= 0 (0 = auto)", f.WeightCacheGB)
	}
	return nil
}

// embedInt4 is --embed-int4 as the load should see it. The default is on, except on Metal: the Metal resident runner
// does not accept an int4 embedding/LM-head table yet ("weight kind \"int4\" is not int8"), so with the default on, a
// plain `--backend metal` load declined the GPU and decoded on the CPU (found 2026-09-28, docs/quantization.md "Known
// issue"; reproduced 2026-09-30 on the 0.5B, while CUDA and WebGPU stayed resident with the int4 table). An explicit
// --embed-int4 is still honoured on Metal, and then runs on the CPU as asked.
func (f *Flags) embedInt4() bool {
	if f.Backend == "metal" && !f.EmbedInt4Set {
		return false
	}
	return f.EmbedInt4
}

// autoBackend is decoder.AutoBackend, a variable so a test can stand in a GPU this test binary does not link.
var autoBackend = decoder.AutoBackend

// resolveAuto replaces a Backend of "auto" with the backend auto picks and keeps the choice in Auto.
// Once Backend names a backend it does nothing, so calling it twice probes the devices once.
func (f *Flags) resolveAuto() {
	if f.Backend != "auto" {
		return
	}
	c := autoBackend()
	f.Backend, f.Auto = c.Backend, &c
}

// BackendLine is the one line an app prints when -backend auto chose its backend: what it chose, why, and how to
// choose otherwise. It is "" when the user named a backend.
func (f *Flags) BackendLine() string {
	switch {
	case f.Auto == nil:
		return ""
	case f.Backend == "cpu":
		return fmt.Sprintf("backend: cpu (%s)", f.Auto.Reason)
	}
	return fmt.Sprintf("backend: %s (%s; -backend cpu to use the CPU)", f.Backend, f.Auto.Reason)
}

// ExplicitQuant is --quant when it was given, else "". A .giw carries its own quant, and only an
// explicit --quant that disagrees is refused (decoder.Model.CheckGiwQuantMatch): the default must
// never conflict with an already-baked bundle.
func (f *Flags) ExplicitQuant() string {
	if f.QuantSet {
		return f.Quant
	}
	return ""
}

// Knobs carries the prompt-ingestion flags to the model's decoder.Options (phase 5,
// docs/tasks/task-env-config-2026-09.md). --exact-prefill itself travels as Options.ExactPrefill,
// which every backend consults per model, so only the CPU-specific flag needs a knob.
//
// The CPU knob is set EXPLICITLY either way rather than left unset. The decoder treats unset as on,
// but an inherited GOINFER_CPU_FAST_ATTENTION in the caller's environment would otherwise outrank the
// flags — the binary's own flags must win over whatever the shell happened to export, and
// Options.Knobs does win. --exact-prefill and --cpu-exact-prefill both disable it.
func (f *Flags) Knobs() *decoder.Knobs {
	fast := "1"
	if f.ExactPrefill || f.CPUExactPrefill {
		fast = "0"
	}
	return &decoder.Knobs{"GOINFER_CPU_FAST_ATTENTION": fast}
}

// CPUKV is the CPU KV cache precision: serve's deprecated --kv-quant when given, else --kv mapped
// onto the CPU cache, which has no f16 form — so f16 stays f32 there and i8 selects the per-head int8
// cache. (Both Options fields remain; only the command line unified.)
func CPUKV(kvQuant, kv string) string {
	if kvQuant != "" {
		return kvQuant
	}
	if kv == "i8" {
		return "i8"
	}
	return "f32"
}

// quantFlag is --quant: a string flag that also records that it was given, so ExplicitQuant can tell
// an explicit --quant from its default.
type quantFlag struct{ f *Flags }

func (q quantFlag) String() string {
	if q.f == nil {
		return ""
	}
	return q.f.Quant
}

func (q quantFlag) Set(v string) error {
	q.f.Quant, q.f.QuantSet = v, true
	return nil
}

// embedInt4Flag is --embed-int4: a bool flag that also records that it was given, so the Metal default (embedInt4) can
// tell an explicit --embed-int4 from its default.
type embedInt4Flag struct{ f *Flags }

func (e embedInt4Flag) String() string {
	if e.f == nil {
		return "true"
	}
	return strconv.FormatBool(e.f.EmbedInt4)
}

func (e embedInt4Flag) Set(v string) error {
	b, err := strconv.ParseBool(v)
	if err != nil {
		return err
	}
	e.f.EmbedInt4, e.f.EmbedInt4Set = b, true
	return nil
}

func (e embedInt4Flag) IsBoolFlag() bool { return true }

func ctxHelp(app App, per string) string {
	s := "GPU-resident KV capacity in positions" + per + ". 0 (default) keeps the backend default — on CUDA, " +
		"4096 with -fit=off, or 8192 (cuda/resident.go's fitDefaultCtx, whatever the card's free VRAM actually admits) " +
		"with -fit at its default of ON; on Metal, 4096 (up to 32768); on webgpu, the ceiling -kv's precision sets. " +
		"The real per-card ceiling is typically far higher and worth measuring for your model/quant " +
		"(docs/tasks/parked/task-kv-cache-streaming.md: an RTX 2070 SUPER 8GB ran a dense 7B at int4 fine at -ctx 20000, " +
		"refused at 24576). When set, the effective cap is min(model context window, this) and the KV it implies is " +
		"VRAM-checked AT LOAD; if the whole model then can't build resident, it loads on the CPU path instead — measured " +
		"~15x slower decode, same model/quant"
	if app == Serve {
		s += " — unless -require-backend is set, which refuses to start the server and names the GB shortfall instead. " +
			"That is a WHOLE-MODEL decision made once at load; a single request whose PROMPT exceeds the active cap is " +
			"rejected with a clean 400 context_length_exceeded (audit R-10), never silently moved to a slower path"
	}
	return s + ". On webgpu this LOWERS the backend cap when smaller; a value LARGER than the backend cap is ignored " +
		"rather than honoured, since those caps are proven-fit ceilings"
}

const backendHelp = "compute backend: auto | cpu | webgpu | cuda | metal (cgo-free native). " +
	"auto (the default) uses the first GPU backend in this binary that finds its device, cuda then metal, else the CPU, " +
	"and prints one line saying which; it never picks webgpu. metal is picked on Apple silicon only, and for int4 models only: " +
	"it re-quantizes other precisions to int4, so auto keeps those on the CPU (-backend metal runs them re-quantized). " +
	"cuda/metal support both dense and MoE architectures resident. cuda needs `-tags cuda`; metal's own submodule " +
	"entrypoints (goinfer/metal/cmd/chat, goinfer/metal/cmd/serve) are darwin-gated and need no build tag of their own. " +
	"On cuda/metal, GPU means fully resident, full stop — a model/arch that does not build the resident runner declines straight to CPU; neither backend has a partial \"staged\" GPU path (R9, docs/measurements/cold-user-2026-09-06-nobara-pc.md). " +
	"The only bigger-than-the-card story on cuda is MoE expert streaming (-moe-cache-experts); webgpu is the one backend with a real staged (non-resident) path, for f32/int8/int8int8, plus int4 decode only (M=1; prefill still declines to CPU)."

const quantHelp = "  int4      W4A8 (int4 weights, int8 activations): fastest on every backend including\n" +
	"            Apple Silicon CPU (measured M1 Pro, goinfer a11c56b 2026-08-24, docs/benchmarks.md: at or\n" +
	"            above int8int8's decode rate -- an earlier reading had int8int8 ~60%\n" +
	"            faster on Apple Silicon CPU, which was correct at the time but diagnosed a since-fixed LM\n" +
	"            head, not the W4A8 kernel; see docs/completed/task-w4a8-neon-bandwidth.md). Lossier than int8\n" +
	"            (4-bit weights). NOT the smallest on Apple Silicon or non-VNNI amd64: the loader keeps a\n" +
	"            repacked second copy of the nibbles beside the canonical ones there, so it measures ~1.25\n" +
	"            bytes/element against int8int8's ~1.02 -- more resident RAM, not less. THE DEFAULT anyway,\n" +
	"            for speed, not for RAM.\n" +
	"  int4mix   attn int8 + FFN int4 (GGUF only): near-int8 quality at below-int8 RAM.\n" +
	"  q4k       EXPERIMENTAL (GGUF only, CPU only for now): the file's Q4_K tensors kept exactly as stored,\n" +
	"            every other tensor int8, per-32 activations. int4's size with int8-class quality on Q4_K_M files.\n" +
	"  int8int8  W8A8 (int8 weights + int8 activations, native SDOT): higher accuracy; on Apple Silicon and\n" +
	"            non-VNNI amd64 it is actually the SMALLER option (see int4's note above), not larger.\n" +
	"            Metal itself consumes int4 directly (no re-pack to int8int8 needed to run resident there).\n" +
	"  int8      int8 weights with wider activations: between int8int8 and native.\n" +
	"  \"\"        native (no quantization, f32): most accurate, largest, slowest.\n" +
	"All quantized modes (int4/int4mix/int8/int8int8) get batched CUDA prefill (fast TTFT); only native f32\n" +
	"falls back to the ~9x slower sequential prefill. A prequantized .giw model carries its own baked-in quant."

const kvHelp = "KV cache precision, for whichever backend serves the model: f32 (bit-exact) | f16 (lossy; GPU residency only — the CPU cache has no f16 form and stays f32) | i8 (lossy: the GPU residency cache, ~64k ctx on 8 GB; the CPU cache as per-head int8, ~4× smaller, argmax ~90%+, excluding MoE/gemma4/qwen3.5 which keep f32)"

const streamWeightsHelp = "page model weights on demand out of an mmap'd .giw, capping resident RAM to -weight-cache instead of holding all weights: MoE expert demand-paging (run a 35B-A3B on ~16-20 GB) or dense per-layer streaming (run a model bigger than RAM). Bit-exact; trades RAM for fault latency. A plain .gguf is transparently transcoded to a sidecar .giw cache on first use (one-time). Also triggered automatically, without this flag, for a dense .gguf that does not fit resident RAM (--fit, default on) -- MoE is deliberately excluded from the automatic path (see --fit's help)"

const moeCacheExpertsHelp = "run a MoE model whose experts EXCEED VRAM/RAM: routed experts stream host→device per token instead of being held resident, so every expert still executes on the GPU (no CPU offload). Costs a per-token transfer; bit-identical to fully-resident. Off by default — with it off, a model that doesn't fit declines to the CPU path and says why. CUDA and Metal (with no --moe-cache-slots, Metal auto-sizes the slot count from free RAM — audit-metal-2026-09-12.md M-13)"

const moeCacheSlotsHelp = "per-layer expert slots to keep resident for a paged MoE model (CUDA: --moe-cache-experts; Metal: the GOINFER_METAL_MOE_SLOTS env var's replacement, docs/tasks/task-gpu-paths-2026-09.md Phase 2). On CUDA this is an UPPER BOUND: the runtime measures free VRAM and lowers it if the request does not fit, logging what it chose (\"C′ cache: … capping to N\"). On Metal an EXPLICIT value here is NOT auto-lowered — the request is used as given, and a model that does not fit at that count declines to the CPU path instead (the load-time memory guard, metal/backend.go). 0 keeps the built-in default: CUDA asks for all and auto-caps; Metal auto-sizes from free RAM ONLY when --moe-cache-experts is also set (audit-metal-2026-09-12.md M-13), else every expert stays resident, unpaged. More slots ⇒ higher LRU hit rate ⇒ fewer per-token transfers, at more memory cost"

const moePagerHelp = "CPU backing mode for a .giw-paged MoE model's expert pager: mmap (advice-based, zero-copy, but on darwin MADV_DONTNEED is a no-op so the budget is NOT enforced — measured 2026-09-23 on M35: ~4 GB of expert pages resident against a 1.5 GB budget) | pool (owned-buffer pread — a firm cap on every platform, costs ~1.4 GB of owned anonymous buffers at that budget, measured 1.02x the mmap decode rate). Default: pool on darwin, mmap elsewhere (docs/measurements/moe-pager-mode-darwin-2026-09-23.md; task-never-swap-2026-09.md S5). CPU decode of a paged MoE model only — CUDA/Metal experts use --moe-cache-experts/--moe-cache-slots instead"

const acceptSlowHelp = "acknowledge a -stream-weights paged-MoE load whose predicted working-set rate falls below decoder's own floor (2 tok/s) and load it anyway. Without this, such a load is refused with the predicted rate named, rather than run for hours with zero completions the way an unacknowledged M35/M26-class load did before this flag existed (task-never-swap-2026-09.md S4). The prediction is a PRIOR borrowed from an unrelated CUDA cache curve, not a measurement of this pager — raising -weight-cache to shrink the predicted miss rate is usually the better fix"

const embedInt4Help = "with -quant int4, store the token-embedding/LM-head table at int4 too instead of the int8 pin — halves the largest resident tensor on a big-vocab small model, and on CPU measured a further 1.05-1.12x on total token time once the head became a bigger share of it (docs/measurements/cpu-decode-peer-gap-2026-09-27.md). Default ON since 2026-09-28 (owner decision, quality re-eval parked), EXCEPT on metal (named, or chosen by -backend auto), where the default is off: the Metal resident runner does not take an int4 table yet, so the default would move the model to the CPU (an explicit -embed-int4 there is honoured, and runs on the CPU). Pass -embed-int4=false for the int8 pin instead. Lossy (~2.3 pts top-1, mostly rare tokens, last measured pre-2026-09-28 — not independently re-verified since). Works with the -stream-weights .giw cache (baked into its own \"e4h\" sidecar, distinct from the int8-pin one) as well as a direct load"

const directLoadHelp = "load a plain .gguf straight into the heap instead of through its sidecar .giw cache. On darwin (since S1, task-never-swap-2026-09.md) and linux (since 2026-09-24) a .gguf resolves to its sidecar by default — this opts back out to the direct-heap-dequant load, which is still the default on other platforms. Also via GOINFER_GGUF_DIRECT=1. Ignored with -stream-weights, which always needs the sidecar's mmap regardless of platform"

const fitHelp = "size an unpinned load to what this machine actually has, instead of a flat historical default (docs/tasks/task-gpu-paths-2026-09.md, tasks/task-fit-to-hardware.md Phase 2). CUDA: an unpinned resident context gets more than the historical 4096 positions when the card has the free VRAM for it (cudaCtxCapDefault's own measurement found the real per-card ceiling is often 5-6x that). CPU: a plain .gguf that will not fit resident RAM gets one automatic retry with weight streaming (a dense model only — see --stream-weights) instead of just refusing. Never touches an EXPLICITLY set -ctx/-quant/--moe-cache-slots/--stream-weights — those are always honoured or refused as asked, with or without this flag. --fit=off restores every pre-Phase-2 default exactly; does not affect bug fixes shipped alongside this work (e.g. Metal now honouring an explicit -ctx at all)"

// ExactPrefillHelp is --exact-prefill's usage text, held as a const so the disclosure can be asserted by
// test. It is the universal opt-out: one flag that disables fast prefill on EVERY backend that has one.
const ExactPrefillHelp = "force BIT-EXACT prompt ingestion on ALL backends — disables the CPU's default f32 prompt attention (above 512 prompt tokens), Metal's f16-MMA batched prefill (default-on since 2026-09-09, above 64 prompt tokens), AND CUDA's tensor-core batched prefill (GOINFER_CUDA_FAST_PREFILL, default-on above 512 prompt tokens). Use when diffing outputs across versions, reproducing a bug report, or whenever decode==prefill bit-identity matters more than time-to-first-token. CPU and Metal's fast paths are fidelity-gated before becoming the default (CPU: §3.1; Metal: §3.2, pooled form) — the exact path is a regression reference, not a correctness emergency"

// CPUExactPrefillHelp is --cpu-exact-prefill's usage text — the only CPU-specific prefill flag since
// --cpu-fast-attention was removed (it defaulted to true, so its only reachable use, =false, was this
// flag under another name). The DEFAULT it opts out of is a documented divergence, so this help carries
// that disclosure too: a divergence must be "disclosed in --help, not something a user has to already
// know to type". Held as a const so TestCPUExactPrefillDisclosesTheDefaultsTrade asserts on the text and
// the disclosure cannot silently drift away from the behaviour.
const CPUExactPrefillHelp = "force BIT-EXACT prompt ingestion on the CPU backend: use the f64-accumulating attention kernel for prefill instead of the f32 one that is the DEFAULT (since 2026-08-31). The default is measured 2.28x faster prefill on an 8k prompt (dense 1.5B, M1 Pro: 602.9s to 264.6s — attention is ~70% of a long prefill and the f64 path is ~8x slower at those shapes) and NOT bit-identical: cosine 0.9976 against the exact kernel, stable across 256/1024/2048-token prompts, so a long-prompt response CAN differ from a build before the default changed, even at temperature 0. The default is FLOORED AT 512 PROMPT TOKENS (below that the exact kernel runs anyway: the win scales with prompt length, the divergence does not). Decode is unaffected, and speculative decoding's verify pass always uses the exact kernel. MoE models take the same f32 path as dense ones — the exclusion was measured and dropped in 66d0a05 — so this is the only way to get bit-exact prefill for them too. Use it when diffing outputs across versions, reproducing a bug report, or anything where 'same prompt, same tokens' matters more than time-to-first-token. CPU backend only — use --exact-prefill to disable fast prefill on all backends at once"
