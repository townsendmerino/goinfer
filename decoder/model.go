package decoder

import (
	"context"
	"fmt"
	"math"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/townsendmerino/aikit/mmap"
	"github.com/townsendmerino/goinfer/internal/giw"
)

// decodeTiming env-gates per-component decode timing (GOINFER_DECODE_TIMING=1) so
// the residency path's per-token cost can be decomposed without touching the hot
// path otherwise.
var decodeTiming = os.Getenv("GOINFER_DECODE_TIMING") != ""

// Model is a loaded Gemma 3 checkpoint plus the compute backend. Goroutine
// safety follows encoder.Model: Weights are immutable after Load; per-
// sequence state (the KV cache) is owned by each Generate call, so distinct
// sequences can run concurrently, but a single KVCache is not shared.
type Model struct {
	w        *Weights
	be       Backend
	eosIDs   []int           // end-of-sequence ids from config (generation stops on these)
	resident ResidentForward // GPU full-residency decode path (webgpu + eligible arch); nil ⇒ staged/CPU
	resBusy  int32           // atomic: claims the single shared resident KV for one in-flight generation (M9). Raw int32 (not atomic.Bool) so Model stays copyable for the value-copy test seam.
	// resIDs is the token sequence currently committed to the resident positional KV, or nil
	// when its contents are unknown. Guarded by the same resBusy claim that serialises writes
	// to that cache; see resident_reuse.go for why nil is the safe default.
	resIDs     []int
	kvF16      bool         // residency KV cache precision request (Options.KVPrecision == "f16")
	kvPrecI8   bool         // residency KV cache int8 request (Options.KVPrecision == "i8") — GPU
	kvI8       bool         // CPU KV cache int8 storage request (Options.KVQuant == "i8") — CPU staged path
	resCtxReq  int          // requested GPU-resident KV capacity in positions (Options.ResidentContext); 0 ⇒ backend default
	moeCache   bool         // stream routed MoE experts host→VRAM (Options.MoECacheExperts)
	moeSlots   int          // per-layer expert slot request (Options.MoECacheSlots); 0 ⇒ ask for all, auto-cap to VRAM
	mmap       []byte       // .giw mmap region the int8/int4 weights alias; munmap'd by Close (nil off the .giw mmap path)
	srcPath    string       // the .giw path this model mmap-loaded from ("" off the .giw path) — for pread-staging over the same file
	pager      *expertPager // MoE expert demand-paging over the mapping (Options.StreamWeights); nil = all-resident
	layerPager *layerPager  // dense per-layer streaming over the mapping (Options.StreamWeights); nil = all-resident
	quant      string       // the requested Options.Quant for a direct load ("" for a prequant .giw → Quant() derives from kinds)
	// resDecline records WHY resident is nil on a non-CPU backend — the reason withResidency
	// would otherwise discard. Empty when residency was built, or when it was never attempted
	// (CPU backend). DecodePath / -require-backend read it; see withResidency.
	resDecline string

	// adapters holds compute-time LoRA adapters loaded against this base (#7), behind a POINTER so
	// *Model stays value-copyable — the kvi8 test seam does `mm := *m` to flip kvI8, and a sync.Mutex
	// field would make `go vet` reject the copy (and resBusy is a raw int32 for the same reason). nil
	// until the first LoadAdapter. The mutex inside guards concurrent LoadAdapter vs UseAdapter/
	// HasAdapter (audit C-29).
	adapters *adapterRegistry
}

// adapterRegistry is the compute-time LoRA state, heap-held so *Model has no lock value.
type adapterRegistry struct {
	mu     sync.Mutex
	byName map[string]*loraRuntime
	// retired holds runtimes displaced by a re-registration of the same name. A live Session may
	// still hold the old *loraRuntime in its cache.lora and read its mmap'd deltas mid-generation, so
	// it must NOT be munmap'd on re-register (that SIGSEGVs the reader, audit C-29). Released only at
	// Model.Close — a small, bounded leak (one entry per re-registration of a live name) traded for
	// read-after-free safety.
	retired []*loraRuntime
}

func newAdapterRegistry() *adapterRegistry {
	return &adapterRegistry{byName: map[string]*loraRuntime{}}
}

// GiwPath returns the .giw file path this model was mmap-loaded from, or "" if it was not loaded
// from a .giw (safetensors/GGUF have no single mmap-backed weight file to pread from). The metal
// expert-paging path re-opens this for pread-staging (the mmap closes its own fd after mapping).
func (m *Model) GiwPath() string { return m.srcPath }

// MmapByteOffset returns the byte offset of slice b within this model's .giw mmap region, or
// ok=false if b does not alias that region (a heap-backed weight, or no .giw mapping). MapReadOnly
// maps the whole file from offset 0, so this offset is exactly where to pread b's bytes from the
// .giw. Pure pointer arithmetic — does NOT touch b's pages (no fault).
func (m *Model) MmapByteOffset(b []byte) (int64, bool) {
	if len(m.mmap) == 0 || len(b) == 0 {
		return 0, false
	}
	base := uintptr(unsafe.Pointer(&m.mmap[0]))
	p := uintptr(unsafe.Pointer(&b[0]))
	if p < base || p+uintptr(len(b)) > base+uintptr(len(m.mmap)) {
		return 0, false
	}
	return int64(p - base), true
}

// KVCacheF16 reports whether the GPU residency path should use an f16 KV cache
// (Options.KVPrecision == "f16"): 2× context (32k) on the same VRAM, lossy. The
// residency builder reads it; off the residency path it has no effect.
func (m *Model) KVCacheF16() bool { return m.kvF16 }

// KVCacheI8 reports whether the GPU residency path should use an int8 KV cache
// (Options.KVPrecision == "i8"): 4× vs f32 / 2× vs f16, ~64k context on the 8 GB
// card. Lossy; f32 + f16 paths unchanged. Distinct from KVQuant (the CPU cache).
func (m *Model) KVCacheI8() bool { return m.kvPrecI8 }

// ResidentContextRequest returns the requested GPU-resident KV capacity in positions
// (Options.ResidentContext), or 0 for "use the backend default". The residency builder resolves the
// effective cap as min(model context window, this) and VRAM-checks it at load; off the residency
// path it has no effect. See cuda.resolveCtxCap.
func (m *Model) ResidentContextRequest() int { return m.resCtxReq }

// MoECacheExperts reports whether routed MoE experts should stream host→VRAM per token instead of
// being held resident (Options.MoECacheExperts, `--moe-cache-experts`). This is what lets a model
// whose experts exceed VRAM run with every expert still executing ON the GPU. Off by default:
// running a model larger than your card is a deliberate act, it costs a per-token PCIe transfer,
// and with it off the runtime declines honestly instead of silently going slow.
//
// Falls back to GOINFER_MOE_CACHE_EXPERTS so existing scripts keep working.
func (m *Model) MoECacheExperts() bool {
	return m.moeCache || os.Getenv("GOINFER_MOE_CACHE_EXPERTS") != ""
}

// MoECacheSlotsRequest returns the requested per-layer expert-slot count, or 0 for "as many as
// fit". Only meaningful with MoECacheExperts.
//
// 0 means ask for ALL experts and let the builder cap to measured free VRAM — deliberately, and
// this is a change from the env-var-only behaviour, where an unset value meant topK. topK is the
// WORST setting for the only situation in which this applies: it degenerates to fresh-loading
// every routed expert every token (~714 MB/token on the 26B, ~5 tok/s instead of ~17). A user who
// asked for expert streaming and said nothing about slots wants it to work, not to be safe; the
// safety is already provided by allocSlots, which measures free VRAM and caps-and-logs rather
// than OOMing.
//
// Falls back to GOINFER_MOE_CACHE_SLOTS.
func (m *Model) MoECacheSlotsRequest() int {
	if m.moeSlots > 0 {
		return m.moeSlots
	}
	if v, err := strconv.Atoi(os.Getenv("GOINFER_MOE_CACHE_SLOTS")); err == nil && v > 0 {
		return v
	}
	return 0
}

// Options configures Load.
type Options struct {
	Backend string // "cpu" (default) or "webgpu"
	Quant   string // "" (f32), "int8" (weight-only per-row), "int8int8" (full int8×int8 W8A8), or "int4" (group-wise) (M8)
	LoRA    string // optional PEFT adapter dir (adapter_config.json + adapter_model.safetensors), merged into the base at load. Safetensors base only.
	// KVPrecision selects the GPU residency KV cache precision: "" / "f32"
	// (default, bit-exact, 16k context cap), "f16" (lossy, 2× context to 32k), or
	// "i8" (lossy, 4× vs f32 → ~64k context). Ignored off the residency path. See
	// task-gpu-f16-kv.md / task-gpu-kv-i8.md.
	KVPrecision string
	// MoECacheExperts streams routed MoE experts host→VRAM per token instead of holding the whole
	// expert stack resident — the path to running a model whose experts exceed VRAM with every
	// expert still executing on the GPU. Off by default; bit-identical to fully-resident when on
	// (cuda.TestGemma4MoE_cacheExpertsBitExact_*). CUDA residency only.
	MoECacheExperts bool
	// MoECacheSlots is the per-layer expert-slot count for MoECacheExperts (0 = ask for all and
	// auto-cap to measured free VRAM). More slots ⇒ higher LRU hit rate ⇒ fewer per-token DMAs,
	// at VRAM cost. Only meaningful with MoECacheExperts.
	MoECacheSlots int
	// KVQuant selects the CPU KV cache storage precision: "" / "f32" (default,
	// bit-exact) or "i8" (per-(position,KV-head) symmetric int8, 4× smaller +
	// SDOT decode). Lossy, opt-in; excluded on MoE / gemma4 / qwen3_5_moe in v1.
	// See task-cpu-kv-quant.md.
	KVQuant string
	// StreamWeights enables on-demand weight residency (idea #2): for an mmap-backed
	// .giw MoE model, expert weights are paged out of the mapping under a RAM budget
	// (WeightCacheBytes) instead of all held resident. Bit-exact (read-only re-fault);
	// trades RAM for cold-miss fault latency. No-op for non-MoE / non-.giw models.
	StreamWeights bool
	// WeightCacheBytes is the resident-bytes budget for streamed weights (0 = auto,
	// ~half of available RAM). Only meaningful with StreamWeights.
	WeightCacheBytes int64
	// EmbedInt4 relaxes the int8 pin on the token-embedding/LM-head table in int4
	// mode, storing it at int4 too — halving the single largest resident tensor on a
	// big-vocab small model. Lossy + opt-in (~2.3 pts top-1, mostly on rare tokens);
	// default off keeps the bit-exact int8 pin. GGUF load path only.
	EmbedInt4 bool
	// ResidentContext requests a GPU-resident KV capacity in positions. 0 (default) keeps the
	// backend's built-in default, so nobody who did not ask allocates deep-KV VRAM. When set, the
	// backend caps it at the model's own context window — the effective cap is
	// min(model context window, this) — and fails at LOAD if the KV that implies does not fit
	// beside the weights, rather than OOM-ing mid-decode. Ignored off the residency path.
	ResidentContext int
}

// Load reads a Gemma 3 snapshot (config.json + model.safetensors) from dir
// and selects a backend. The forward pass (M3) is implemented; the CPU
// backend is the default and the only one wired (webgpu falls back to CPU).
func Load(dir string, opts Options) (*Model, error) {
	be, beErr := NewBackend(opts.Backend)
	// A nil backend means the name was genuinely unknown (not a registered/fallback backend) —
	// abort rather than proceed and panic at the first matmul (M14). A non-nil be with a
	// non-nil beErr is the CPU-fallback note (webgpu/cuda/metal not built in); keep the (cpu)
	// backend and surface the note rather than abort.
	if be == nil {
		return nil, beErr
	}

	// Prequant bundle (.giw): the weights are already quantized and serialized, so
	// they alias straight out of the file — no GGUF/safetensors load, no requant
	// (opts.Quant/LoRA do not apply). The file is mmap'd read-only so the aliased
	// int8/int4 weights are pageable (faulted from the page cache, evictable) rather
	// than copied to the heap — the substrate the streaming / expert-paging policies
	// build on. giw.Read splits the weight blob from the metadata-GGUF tokenizer; the
	// mapping is held on the Model and released by Close.
	if strings.HasSuffix(dir, ".giw") {
		data, rerr := mmap.MapReadOnly(dir)
		if rerr != nil {
			closeBackend(be)
			return nil, fmt.Errorf("decoder: mmap .giw: %w", rerr)
		}
		weightsBlob, _, gerr := giw.Read(data)
		if gerr != nil {
			_ = mmap.Unmap(data)
			closeBackend(be)
			return nil, fmt.Errorf("decoder: parse .giw bundle: %w", gerr)
		}
		w, lerr := LoadSerializedWeights(weightsBlob)
		if lerr != nil {
			_ = mmap.Unmap(data)
			closeBackend(be)
			return nil, lerr
		}
		if beErr != nil {
			fmt.Fprintln(os.Stderr, beErr)
		}
		m := &Model{w: w, be: be, mmap: data, srcPath: dir, eosIDs: w.Cfg.EOSIDs(), kvF16: opts.KVPrecision == "f16", kvPrecI8: opts.KVPrecision == "i8", kvI8: opts.KVQuant == "i8", resCtxReq: opts.ResidentContext, moeCache: opts.MoECacheExperts, moeSlots: opts.MoECacheSlots}
		if opts.StreamWeights {
			// MoE → expert demand-paging (#2); dense → per-layer streaming (#4).
			if w.arch.MoE != nil {
				if m.pager = newExpertPager(w, data, opts.WeightCacheBytes); m.pager != nil {
					fmt.Fprintln(os.Stderr, "decoder: "+pagerSummary(m.pager))
				} else {
					fmt.Fprintln(os.Stderr, "decoder: --stream-weights ignored (no mmap-backed MoE experts to page)")
				}
			} else if m.layerPager = newLayerPager(w, data, opts.WeightCacheBytes); m.layerPager != nil {
				fmt.Fprintln(os.Stderr, "decoder: "+layerPagerSummary(m.layerPager))
			} else {
				fmt.Fprintln(os.Stderr, "decoder: --stream-weights ignored (model fits the budget, or no mmap-backed layer weights)")
			}
		}
		return m.withResidency(), nil
	}

	// Resolve the quant mode first so the weights stream straight into the
	// chosen precision at load — no whole-model f32 spike (see loadWeights).
	quant, err := parseQuant(opts.Quant)
	if err != nil {
		closeBackend(be)
		return nil, err
	}

	// Optional LoRA adapter, merged into the base weights at load (safetensors base
	// only — PEFT targets HF module names).
	var lora *loraAdapter
	if opts.LoRA != "" {
		if strings.HasSuffix(dir, ".gguf") {
			closeBackend(be)
			return nil, fmt.Errorf("decoder: LoRA merge needs a safetensors base, not a .gguf")
		}
		if lora, err = loadLoRA(opts.LoRA); err != nil {
			closeBackend(be)
			return nil, err
		}
		defer lora.close()
	}

	w, err := loadWeights(dir, quant, opts.EmbedInt4, lora)
	if err != nil {
		closeBackend(be)
		return nil, err
	}
	if beErr != nil {
		// webgpu requested but fell back — not fatal.
		fmt.Fprintln(os.Stderr, beErr)
	}
	if opts.StreamWeights {
		// Weight streaming pages weights out of the read-only mmap, which only the
		// .giw path provides; a GGUF/safetensors load dequantizes into the heap, so
		// there's nothing to page. Make the no-op visible rather than silently
		// running fully resident (prequant to .giw with cmd/prequant to use it).
		fmt.Fprintln(os.Stderr, "decoder: --stream-weights ignored — weights are heap-resident; prequant to .giw (cmd/prequant) to enable streaming")
	}
	return (&Model{w: w, be: be, quant: opts.Quant, eosIDs: resolveEOSIDs(dir, &w.Cfg), kvF16: opts.KVPrecision == "f16", kvPrecI8: opts.KVPrecision == "i8", kvI8: opts.KVQuant == "i8", resCtxReq: opts.ResidentContext, moeCache: opts.MoECacheExperts, moeSlots: opts.MoECacheSlots}).withResidency(), nil
}

// Validate checks the stringly-typed knobs against their allowed values, so an
// invalid enum (e.g. -kv-quant=int8 instead of i8) is a clear load-time error
// rather than a silent fall-through to the default. Callers building Options from
// untrusted input (cmd/serve's per-model flags) should call it before Load; Load
// itself does not, to keep existing direct callers unchanged. Path/arch-dependent
// rules (StreamWeights needs a .giw; KVQuant i8 excludes MoE) stay runtime checks.
func (o Options) Validate() error {
	if _, err := parseQuant(o.Quant); err != nil {
		return err
	}
	switch o.Backend {
	case "", "cpu", "webgpu", "cuda", "metal":
		// Accepting the NAME is not a claim that the backend is built in: an unregistered
		// one falls back to CPU with a note (NewBackend). Rejecting it here instead meant
		// `serve --backend cuda|metal` failed at flag-validation even when the module WAS
		// compiled in (-tags cuda / -tags metal), because Validate runs before registration
		// is ever consulted.
	default:
		return fmt.Errorf("decoder: invalid backend %q (cpu | webgpu | cuda | metal)", o.Backend)
	}
	switch o.KVPrecision {
	case "", "f32", "f16", "i8":
	default:
		return fmt.Errorf("decoder: invalid -kv %q (f32 | f16 | i8)", o.KVPrecision)
	}
	switch o.KVQuant {
	case "", "f32", "i8":
	default:
		return fmt.Errorf("decoder: invalid -kv-quant %q (f32 | i8)", o.KVQuant)
	}
	return nil
}

// parseQuant maps Options.Quant to the internal quantMode.
func parseQuant(q string) (quantMode, error) {
	switch q {
	case "", "f32":
		return quantNone, nil
	case "int8":
		return quantInt8, nil
	case "int8int8":
		return quantInt8I8, nil
	case "int4":
		return quantInt4, nil
	case "int4mix":
		return quantInt4Mix, nil
	default:
		return quantNone, fmt.Errorf("decoder: unknown quant %q (have: int8, int8int8, int4, int4mix)", q)
	}
}

// closeBackend closes a backend on a load-error path (nil-safe).
func closeBackend(be Backend) {
	if be != nil {
		_ = be.Close()
	}
}

// Config returns a SNAPSHOT of the loaded architecture config — a copy, not the live struct
// (audit M-23). The forward pass reads a derived, unexported *Architecture plus precomputed RoPE
// tables built from this config AT LOAD; the config itself is not re-read per token. Returning
// the live &m.w.Cfg let a caller do m.Config().NumLayers = N and silently desync those caches
// from the config — wrong logits on the path the project calls its stable contract. The copy
// makes such a write land on a throwaway value instead. (Scalar fields — the realistic footgun —
// are fully isolated; the copy is shallow, so its slice/pointer fields still alias the model's
// and must be treated as read-only. There is no supported way to reconfigure a loaded model.)
func (m *Model) Config() *Config {
	c := m.w.Cfg
	return &c
}

// Close releases backend resources (GPU resident buffers + the backend) and
// unmaps the .giw mapping if the model was loaded from a prequant bundle. A no-op
// for the CPU backend with no mapping. Safe to call once after the model is done;
// the weights must not be touched afterward (the mapping is gone).
func (m *Model) Close() error {
	if m.resident != nil {
		_ = m.resident.Close()
		m.resident = nil
	}
	var err error
	if m.be != nil {
		err = m.be.Close()
	}
	if m.mmap != nil {
		_ = mmap.Unmap(m.mmap)
		m.mmap = nil
	}
	if m.w != nil && m.w.st != nil { // release the safetensors mmap + per-shard fds; serve load/unload cycles otherwise retain them until GC
		_ = m.w.st.Close()
		m.w.st = nil
	}
	if reg := m.adapters; reg != nil {
		reg.mu.Lock()
		for _, rt := range reg.byName { // release each compute-time adapter's mmap (#7)
			rt.close()
		}
		for _, rt := range reg.retired { // and any displaced by a re-registration (C-29)
			rt.close()
		}
		reg.byName, reg.retired = nil, nil
		reg.mu.Unlock()
	}
	return err
}

// NewCache allocates a KV cache sized for this model. capHint pre-sizes for
// a known max length (0 = grow on demand).
func (m *Model) NewCache(capHint int) *KVCache {
	a := m.w.arch
	c := NewKVCache(a.NumLayers, a.NumKVHeads, a.HeadDim, a.SlidingWindow, capHint)
	c.scr = newDecodeScratch(a)
	// int8 KV storage (opt-in, Options.KVQuant=="i8"): the uniform dense families
	// only — MoE routes attention through the acc64 kernel for bit-stable expert
	// routing (quantized KV would reopen that), and gemma4/qwen3_5_moe have their
	// own forward. Must precede enableRings so local layers inherit the mode.
	if m.kvI8 && a.gemma4 == nil && a.qwen35 == nil && a.granite == nil && a.nemotron == nil && a.MoE == nil {
		c.setQuant(kvI8, capHint)
	}
	// Ring-buffer storage on sliding-window (local) layers: keep only the W most
	// recent positions, the only ones a future query can read. Restricted to the
	// uniform-stride families whose forward uses attendQuery/attendBatchedHeads;
	// gemma4 (per-layer widths + KV-sharing) and qwen3_5_moe (linear attention)
	// have their own forward and keep append-forever for now (a later increment).
	// See docs/completed/task-kv-ring-eviction.md.
	if a.gemma4 == nil && a.qwen35 == nil && a.granite == nil && a.nemotron == nil && a.gptoss == nil {
		c.enableRings(a.SlidingWindow, a.isGlobalLayer)
	}
	if a.gemma4 != nil {
		c.manualPos = true // gemma4's last layer is KV-shared; pos advances via Advance()
	}
	if a.qwen35 != nil {
		// Hybrid cache: KV for the softmax layers + a recurrent DeltaState for each
		// linear layer. manualPos because the linear layers never Append.
		c.manualPos = true
		c.delta = make([]*deltaState, a.NumLayers)
		for l := 0; l < a.NumLayers; l++ {
			if a.isLinearLayer(l) {
				c.delta[l] = newDeltaState(*a.qwen35)
			}
		}
	}
	if a.lfm2 != nil {
		// Hybrid cache: KV for the 8 attention layers + a rolling conv window for each of
		// the 22 conv layers. manualPos because the conv layers never Append, so position
		// cannot be inferred from the KV length.
		c.manualPos = true
		c.conv = make([]*shortConvState, a.NumLayers)
		for l := 0; l < a.NumLayers; l++ {
			if a.isConvLayer(l) {
				c.conv[l] = newShortConvState()
			}
		}
	}
	if a.granite != nil {
		// Hybrid cache: KV for the attention layers + a Mamba-2 recurrent state for
		// each mamba layer. manualPos because the mamba layers never Append.
		c.manualPos = true
		c.mamba = make([]*mamba2State, a.NumLayers)
		mp := mamba2Params{NHeads: a.granite.NHeads, HeadDim: a.granite.HeadDim, DState: a.granite.DState, NGroups: a.granite.NGroups, DConv: a.granite.DConv, Hidden: a.HiddenDim}
		for l := 0; l < a.NumLayers; l++ {
			if a.isMambaLayer(l) {
				c.mamba[l] = newMamba2State(mp)
			}
		}
	}
	if a.nemotron != nil {
		// Single-op-block hybrid: only attention layers touch KV (manualPos), mamba
		// layers carry a Mamba-2 recurrent state, mlp layers neither.
		c.manualPos = true
		c.mamba = make([]*mamba2State, a.NumLayers)
		mp := mamba2Params{NHeads: a.nemotron.NHeads, HeadDim: a.nemotron.HeadDim, DState: a.nemotron.DState, NGroups: a.nemotron.NGroups, DConv: a.nemotron.DConv, Hidden: a.HiddenDim}
		for l := 0; l < a.NumLayers; l++ {
			if a.nemotron.blockKind[l] == nemoMamba {
				c.mamba[l] = newMamba2State(mp)
			}
		}
	}
	if a.mla != nil {
		// DeepSeek MLA: the cache is the per-layer compressed latent, not full K/V.
		// manualPos because the forward appends via AppendLatent (not the standard
		// Append) and advances pos once per token after the full layer sweep.
		c.manualPos = true
		c.mlaLatent = make([][]float32, a.NumLayers)
	}
	return c
}

// runLayers advances one decode step for token id at position cache.Pos():
// it embeds the token, runs the block stack (appending this position's K/V to
// the cache), and returns the residual-stream hidden state after the final
// layer — BEFORE the final norm and LM head. Splitting it out lets prefill skip
// the (vocab-sized) LM head on every token but the last.
//
// The loop is generic over the Architecture descriptor (G0): embedding scale,
// norm placement (Gemma's 4-norm sandwich vs Llama's pre-2), the (1+w) RMS
// offset, and the activation are all knobs. Gemma 3 is one descriptor:
//
//	h = Embed[id] * EmbedScale
//	for each layer l:
//	  n  = rmsNorm(h, PreAttnNorm)
//	  a  = causalAttention(l, n, …)
//	  if Sandwich4 { a = rmsNorm(a, PostAttnNorm) }
//	  h += a
//	  n2 = rmsNorm(h, PreMLPNorm)
//	  g  = gatedMLP(n2, …)
//	  if Sandwich4 { g = rmsNorm(g, PostMLPNorm) }
//	  h += g
func (m *Model) runLayers(id int, cache *KVCache) ([]float32, error) {
	arch := m.w.arch
	if m.w.Embed.Rows() == 0 {
		return nil, fmt.Errorf("decoder.forward: weights not loaded %w [M1]", errNotImplemented)
	}
	if f, ok := arch.ownForward(); ok {
		return f.run(m, id, cache)
	}
	if cache.scr == nil { // caches from NewKVCache directly (tests); Generate uses NewCache
		cache.scr = newDecodeScratch(arch)
	}
	scr := cache.scr
	hidden := arch.HiddenDim
	h := scr.h           // residual stream (reused per stream; fully overwritten below)
	m.w.Embed.Row(id, h) // f32 copy, or int8 dequant when quantized
	// Embedding scale (Gemma = √hidden; 0/1 = none). NOTE: HF computes this
	// normalizer as sqrt(hidden) cast to the model's dtype — bf16 for a bf16
	// checkpoint (≈25.25 here) — then multiplies. We use the f32 value
	// (≈25.2982). It matches our parity gate because the next op (PreAttnNorm
	// RMSNorm) divides out a global scalar, so the difference only survives in
	// the residual and stays well under the ≥1−1e-4 cosine bar. If that bar is
	// ever tightened past ~1e-5, round the scale to bf16.
	if arch.EmbedScale != 0 && arch.EmbedScale != 1 {
		scale := float32(arch.EmbedScale)
		for i := range h {
			h[i] *= scale
		}
	}
	// Learned absolute position embedding (GPT-2): add wpe[pos], where pos is
	// this token's absolute position (the cache advances on Append inside
	// attention, so cache.Pos() here is still this step's position).
	if arch.LearnedPosEmbed {
		pe := make([]float32, hidden)
		m.w.PosEmbed.Row(cache.Pos(), pe)
		for i := range h {
			h[i] += pe[i]
		}
	}
	return m.runLayersFromEmbed(h, cache)
}

// runLayersFromEmbed runs the transformer layers over a precomputed
// residual-stream embedding h ([hidden]) and returns it (mutated in place). For
// a text token h is the embedding lookup (+ scale/pos) that runLayers just
// computed; for an IMAGE position (multimodal) it is the projected vision
// embedding the interleaver substitutes in place of a token-id lookup. Splitting
// it out is the embed-by-vector seam: text behavior is unchanged (runLayers is
// exactly embed-then-this), and image embeddings reach the decoder without
// passing through embed_tokens. Generic path only (gemma4/qwen35 have their own
// runLayers and grow the same seam when those families go multimodal).
func (m *Model) runLayersFromEmbed(h []float32, cache *KVCache) ([]float32, error) {
	arch := m.w.arch
	if cache.scr == nil { // direct callers (tests / the interleaver) may skip runLayers
		cache.scr = newDecodeScratch(arch)
	}
	scr := cache.scr
	hidden := arch.HiddenDim
	sandwich := arch.NormPlacement == NormSandwich4
	parallel := arch.NormPlacement == NormParallel
	if m.layerPager != nil {
		defer m.layerPager.finishLayers()
	}
	for l := 0; l < arch.NumLayers; l++ {
		if m.layerPager != nil {
			m.layerPager.enterLayer(l) // prefetch l+1, release the layer behind (#4)
		}
		lw := &m.w.Layers[l]
		var ld *loraLayerDelta // compute-time LoRA deltas for this layer (#7); nil when no adapter is active
		if cache.lora != nil {
			ld = &cache.lora.layers[l]
		}
		if parallel {
			// Cohere/GPT-J parallel block: ONE shared input norm feeds both
			// sublayers, whose outputs sum into a SINGLE residual add —
			// h += attn(n) + mlp(n), where n = norm(h). Both read the same n
			// (attention leaves scr.norm intact), so MLP takes scr.norm, NOT the
			// post-attention residual. No pre-MLP norm, no post-sublayer norms.
			copy(scr.norm, h)
			normalize(arch, scr.norm, lw.PreAttnNorm, lw.PreAttnNormBias, hidden)
			if err := causalAttention(l, scr.norm, scr.sub, lw, arch, cache, m.be, ld); err != nil {
				return nil, err
			}
			if cache.subCapture { // scr.ctx is the pre-o-proj context; scr.sub is not yet overwritten
				cache.subCtx[l] = append(cache.subCtx[l][:0], scr.ctx...)
				cache.subAttn[l] = append(cache.subAttn[l][:0], scr.sub...)
			}
			if err := mlp(scr.norm, scr.sub2, lw, arch, m.be, scr, m.pager, ld); err != nil {
				return nil, err
			}
			if cache.subCapture { // no post-norm ⇒ the pre and final MLP contributions are identical
				cache.subMLPpre[l] = append(cache.subMLPpre[l][:0], scr.sub2...)
				cache.subMLP[l] = append(cache.subMLP[l][:0], scr.sub2...)
			}
			for i := range h {
				h[i] += scr.sub[i] + scr.sub2[i]
			}
		} else {
			copy(scr.norm, h)
			normalize(arch, scr.norm, lw.PreAttnNorm, lw.PreAttnNormBias, hidden)
			if err := causalAttention(l, scr.norm, scr.sub, lw, arch, cache, m.be, ld); err != nil {
				return nil, err
			}
			if cache.subCapture { // scr.ctx is the pre-o-proj context; scr.sub is not yet overwritten
				cache.subCtx[l] = append(cache.subCtx[l][:0], scr.ctx...)
			}
			if sandwich {
				normalize(arch, scr.sub, lw.PostAttnNorm, nil, hidden)
			}
			if cache.subCapture { // scr.sub is now the attention contribution about to hit the residual
				cache.subAttn[l] = append(cache.subAttn[l][:0], scr.sub...)
			}
			for i := range h {
				h[i] += scr.sub[i]
			}
			copy(scr.norm, h)
			normalize(arch, scr.norm, lw.PreMLPNorm, lw.PreMLPNormBias, hidden)
			if err := mlp(scr.norm, scr.sub, lw, arch, m.be, scr, m.pager, ld); err != nil {
				return nil, err
			}
			if cache.subCapture { // scr.sub is the down output BEFORE the post-MLP sandwich norm
				cache.subMLPpre[l] = append(cache.subMLPpre[l][:0], scr.sub...)
			}
			if sandwich {
				normalize(arch, scr.sub, lw.PostMLPNorm, nil, hidden)
			}
			if cache.subCapture { // scr.sub is now the MLP contribution about to hit the residual
				cache.subMLP[l] = append(cache.subMLP[l][:0], scr.sub...)
			}
			for i := range h {
				h[i] += scr.sub[i]
			}
		}
		// Read-only hidden-state seam (05): copy this layer's output residual stream
		// when requested. A copy (not a reference) — h is mutated by later layers.
		if cache.captureLayers != nil {
			for i, cl := range cache.captureLayers {
				if cl == l {
					cache.captured[i] = append(cache.captured[i][:0], h...)
				}
			}
		}
	}
	return h, nil
}

// embedToken writes the residual-stream embedding for token id into dst ([hidden])
// — the embed_tokens row with the architecture's embedding scale + GPT-2 learned
// positional embedding applied, i.e. exactly the vector runLayers feeds into
// runLayersFromEmbed. Exposed so the multimodal interleaver can build a mixed
// text/image embedding sequence (text positions via this, image positions via
// the projected vision features) and drive runLayersFromEmbed per position.
func (m *Model) embedToken(id int, dst []float32) {
	arch := m.w.arch
	m.w.Embed.Row(id, dst)
	if arch.EmbedScale != 0 && arch.EmbedScale != 1 {
		scale := float32(arch.EmbedScale)
		for i := range dst {
			dst[i] *= scale
		}
	}
	// NOTE: GPT-2 learned positional embedding is position-dependent, so it is
	// applied in runLayers (which knows cache.Pos()), not here. embedToken is the
	// position-independent token embedding; the interleaver adds pos if a future
	// learned-pos family goes multimodal (none today do).
}

// normalize applies the architecture's normalization in place over one row:
// LayerNorm (mean-centered, with bias) for GPT-2/NeoX, else RMSNorm. bias is
// ignored by RMSNorm (and nil for the Sandwich4 post-norms).
func normalize(arch *Architecture, x, weight, bias []float32, dim int) {
	if arch.Norm == NormLayer {
		layerNorm(x, weight, bias, 1, dim, arch.NormEps)
		return
	}
	rmsNorm(x, weight, 1, dim, arch.NormEps, arch.RMSAddOne)
}

// forward runs runLayers then the final norm + LM head, returning the logit
// vector ([VocabSize]) for the next token. The head is the tied embedding
// (Gemma) or a separate lm_head (untied). Optional final
// logit soft-capping (Gemma 2; Gemma 3 = none).
func (m *Model) forward(id int, cache *KVCache) ([]float32, error) {
	h, err := m.runLayers(id, cache)
	if err != nil {
		return nil, err
	}
	return m.logitsFromHidden(h, cache), nil
}

// ForwardCapture runs one forward for token id and returns the next-token logits
// PLUS the residual stream after each layer in `layers` (cloned) — the read-only
// hidden-state seam an EAGLE-3 draft head fuses (05). The forward is byte-identical
// to forward(id): the captures are copies that never feed back. Layer indices are
// 0-based into [0, NumLayers); out[i] corresponds to layers[i].
//
// Wired for the generic decode path plus the own-runLayers families whose loops call
// cache.captureResidual (see decoder/capture.go): qwen3_5_moe, gemma4, gpt-oss — the three
// P10 block-drafting targets we hold locally with a licensed drafter. The rest still return
// an error rather than silently producing nothing: granite and nemotron_h interleave recurrent
// mixers whose "residual after layer l" needs deciding rather than assuming, mla and llama4_text
// are simply not done. A family is wired only when BOTH its loop captures and it leaves this
// list, so a half-wired one fails here loudly instead of handing back nil rows.
func (m *Model) ForwardCapture(id int, cache *KVCache, layers []int) (logits []float32, hidden [][]float32, err error) {
	a := m.w.arch
	// Derived from the dispatch table's Captures bit rather than re-listed: the families whose own
	// loop calls captureResidual are wired, every other own-forward family is not. LFM2 was in
	// neither list, so runLayersLFM2 — which never captures — returned nil rows through a seam
	// documented to fail loudly instead (audit-2026-09-02 C-02; EAGLE's fuseAt would panic on them).
	if f, own := a.ownForward(); own && !f.Captures {
		return nil, nil, fmt.Errorf("decoder.ForwardCapture: hidden-state seam not wired for arch %q (own runLayers)", a.Name)
	}
	for _, l := range layers {
		if l < 0 || l >= a.NumLayers {
			return nil, nil, fmt.Errorf("decoder.ForwardCapture: layer %d out of range [0,%d)", l, a.NumLayers)
		}
	}
	cache.captureLayers = layers
	cache.captured = make([][]float32, len(layers))
	defer func() { cache.captureLayers, cache.captured = nil, nil }()
	lg, ferr := m.forward(id, cache)
	if ferr != nil {
		return nil, nil, ferr
	}
	return lg, cache.captured, nil
}

// ForwardSubCapture runs one token and returns, per layer, the attention contribution and the
// MLP contribution to the residual (scr.sub after each sublayer's sandwich-norm, before the add).
// This is the finer seam ForwardCapture doesn't give: it separates the two sublayers, which is
// what localizing a channel's sign flip to attention-vs-MLP at a specific layer requires.
// Diagnostic — same byte-identical-output contract as ForwardCapture. Not wired for own-forward
// families (they don't route through runLayersFromEmbed's uniform block).
func (m *Model) ForwardSubCapture(id int, cache *KVCache) (attn, mlp, ctx, mlpPre [][]float32, err error) {
	a := m.w.arch
	// EVERY own-forward family, derived: this seam needs runLayersFromEmbed's uniform block, which
	// no own-forward loop routes through. The hand-written list was that set minus lfm2.
	if _, own := a.ownForward(); own {
		return nil, nil, nil, nil, fmt.Errorf("decoder.ForwardSubCapture: not wired for arch %q (own runLayers)", a.Name)
	}
	nL := a.NumLayers
	cache.subCapture = true
	cache.subAttn = make([][]float32, nL)
	cache.subMLP = make([][]float32, nL)
	cache.subMLPpre = make([][]float32, nL)
	cache.subCtx = make([][]float32, nL)
	defer func() {
		cache.subCapture = false
		cache.subAttn, cache.subMLP, cache.subMLPpre, cache.subCtx = nil, nil, nil, nil
	}()
	if _, ferr := m.forward(id, cache); ferr != nil {
		return nil, nil, nil, nil, ferr
	}
	return cache.subAttn, cache.subMLP, cache.subCtx, cache.subMLPpre, nil
}

// forwardFromEmbed is forward for a position whose residual-stream embedding is
// supplied directly (the multimodal embed-by-vector seam): it runs the layers
// from h and projects to logits, identical to forward(id) when h is that token's
// embedding. The generic path only (gemma4/qwen35 keep their own forward).
func (m *Model) forwardFromEmbed(h []float32, cache *KVCache) ([]float32, error) {
	h, err := m.runLayersFromEmbed(h, cache)
	if err != nil {
		return nil, err
	}
	return m.logitsFromHidden(h, cache), nil
}

// logitsFromHidden applies the final norm + LM head (tied embedding or separate
// lm_head) + optional Gemma logit soft-cap to a layer-stack output h, returning
// the next-token logits ([VocabSize]). One home for the head math, shared by
// forward and forwardFromEmbed.
func (m *Model) logitsFromHidden(h []float32, cache *KVCache) []float32 {
	arch := m.w.arch
	normalize(arch, h, m.w.FinalNorm, m.w.FinalNormBias, arch.HiddenDim)
	logits := cache.scr.logits // reused per stream; matmul fully overwrites it
	if arch.TiedLMHead {
		matmulInto(cache.scr.ws, m.be, &m.w.Embed, h, logits, 1) // tied: embedding doubles as the head
	} else {
		matmulInto(cache.scr.ws, m.be, &m.w.LMHead, h, logits, 1) // separate output projection
	}
	if arch.FinalLogitSoftcap > 0 {
		softcap := float32(arch.FinalLogitSoftcap)
		for i, v := range logits {
			logits[i] = softcap * float32(math.Tanh(float64(v/softcap)))
		}
	}
	if arch.LogitScale != 0 && arch.LogitScale != 1 { // Granite logits_scaling: logits /= scale
		inv := float32(1 / arch.LogitScale)
		for i := range logits {
			logits[i] *= inv
		}
	}
	return logits
}

// Generate streams generated token ids over the returned channel until EOS,
// a stop id, maxTokens, or ctx cancellation. prompt is already-tokenized
// ids (the demo runs the tokenizer). The channel closes when generation
// ends; check Err after the range loop for a terminal error.
//
// Sampling is greedy at Temperature 0, else temperature/top-k/top-p (see
// Sampler). A SamplingParams.LogitProcessor, if set, masks each step's logits
// before sampling — the seam for constrained/structured decoding.
func (m *Model) Generate(ctx context.Context, prompt []int, maxTokens int, sp SamplingParams) (<-chan int, *Generation) {
	out := make(chan int)
	g := &Generation{}
	cache := m.NewCache(len(prompt) + maxTokens)
	go func() {
		defer close(out)
		m.generateInto(ctx, out, g, cache, prompt, 0, maxTokens, sp, nil)
	}()
	return out, g
}

// residentPrefillSeed ingests the whole prompt into the resident KV and returns the
// LAST token's logits — the seed for decode. Both resident generation paths call it:
// generateInto (plain) and genNgramInto (speculative).
//
// IT IS SHARED ON PURPOSE. These were two copies, and they drifted: batched prefill
// was wired into generateInto by c36698a (2026-07-16) and NOT into the speculative
// twin, which had carried its own per-token loop since 0fd54e8 (2026-06-20). The
// speculative path therefore prefilled one token at a time on every backend that
// implements Prefiller, measured at +2.66 ms per prompt token (R^2 0.9977) against
// the batched path's 0.42 — a 6.3x per-token penalty that made `serve --spec ngram`
// on resident CUDA 3-4.5x SLOWER than no drafter at all on realistic prompts. It went
// unseen for six weeks because the only GPU speculative harness used 36-74 token
// prompts and logged its speedup without asserting on it (docs/spec/02). One
// function, so the next optimisation cannot land on one path and miss the other.
//
// Batched prefill (optional Prefiller) ingests the prompt in one pass — much faster
// TTFT for long prompts. It declines (falls back) past the backend's cap, is absent
// for backends without a batched forward (WebGPU implements no Prefiller; CUDA and
// Metal do), and is skipped for tiny prompts.
//
// DEFAULT ON — bit-identical to sequential decode again (restored 2026-08-04). It was
// briefly default-off after an 84% token-stream divergence traced to a compiler
// fma-vs-mul+add contraction difference between the separately-compiled batched and
// decode GEMV/RMS kernels. FIXED: every float MAC in both paths is now an explicit
// __fmaf_rn (no compiler discretion), enforced at build time by cuda.TestKernelFMALint.
// The decode-side half shipped in aikit/gpu@v0.25.0 (gemv_w4a8_fwd); with the dep
// bumped, TestPrefillDivergenceRate is 0/50 on the real 1.5B (was 42/50), gap
// byte-identical. GOINFER_BATCHED_PREFILL=0 force-disables.
// See docs/task-batched-prefill-bitidentity.md.
// from is the first position to compute: prompt[:from] is already committed to the resident
// KV (prefix reuse, resident_reuse.go) and positions carry through unchanged because the cache
// prefillDeclineOnce keeps warnPrefillDeclined to one line per process.
var prefillDeclineOnce sync.Once

// warnPrefillDeclined reports, ONCE, that a backend's batched prefill refused a prompt at call time
// and this prompt (and every one after it) is being ingested one token at a time instead.
//
// It exists because the load-time report and the runtime behaviour could disagree with nothing
// saying so. PrefillPath() answers from the model's static properties, so a serve banner and
// /v1/models both said "batched (one weight-stationary CUDA pass)" while every long prompt hit an
// M-dependent decline — the CUDA scratch is O(M·inter), and an 8k prompt on an 8 GB card asked for
// 2.28 GB it did not have. Measured on qwen2.5-7b at int4: the fallback runs at 12.5 ms/token
// against the batched path's 2.8, and the only visible symptom was a slow benchmark cell.
// prefillChunked now keeps that case on the fast path, so this should be rare — which is exactly why
// it is worth a line when it happens rather than another silent 4.5×.
//
// Stderr and not an error: the fallback is CORRECT, just slow, and failing the request over a
// performance decline would be worse than serving it.
func warnPrefillDeclined(n int, err error) {
	prefillDeclineOnce.Do(func() {
		fmt.Fprintf(os.Stderr, "goinfer: batched prefill declined for a %d-token prompt, falling back to "+
			"the per-token path (slower TTFT; this is reported once): %v\n", n, err)
	})
}

// is positional. from == 0 is the cold path.
func (m *Model) residentPrefillSeed(ctx context.Context, prompt []int, from int) ([]float32, error) {
	if from < 0 || from >= len(prompt) {
		from = 0 // never skip the seed token, whose logits start decode
	}
	suffix := prompt[from:]
	if os.Getenv("GOINFER_BATCHED_PREFILL") != "0" && len(suffix) >= 8 {
		if pf, ok := m.resident.(Prefiller); ok {
			embs := make([][]float32, len(suffix))
			for i, id := range suffix {
				embs[i] = m.embedResident(id)
			}
			// startPos is why the batched path needs no change for reuse: it already
			// places the run at an offset.
			lg, perr := pf.PrefillLast(embs, from)
			if perr == nil {
				return lg, nil
			}
			warnPrefillDeclined(len(suffix), perr)
		}
	}
	// KV-only prefill: prompt[:-1] tokens need only their K/V in the cache — skip the LM head
	// (a big-vocab matmul + ~1 MB readback + softcap) on every prefill token but the last, whose
	// logits seed decode. Byte-identical (same layer chain → same KV → same last-token logits);
	// GOINFER_NO_KVONLY_PREFILL forces the full-logits prefill (A/B / escape hatch).
	kvOnly, hasKV := m.resident.(ResidentPrefillKV)
	useKV := hasKV && os.Getenv("GOINFER_NO_KVONLY_PREFILL") == ""
	var logits []float32
	var err error
	for i, id := range prompt {
		if i < from {
			continue // already in the cache at position i
		}
		// G18: the resident prefill loop is the GPU-side twin of the batched CPU
		// path's per-layer check. Same failure without it — an abandoned client
		// leaves the whole prompt streaming through the device.
		if err = ctx.Err(); err != nil {
			return nil, err
		}
		emb := m.embedResident(id)
		if useKV && i < len(prompt)-1 {
			if err = kvOnly.ForwardNoLogits(emb, i); err != nil {
				return nil, err
			}
			continue
		}
		if logits, err = m.resident.Forward(emb, i); err != nil {
			return nil, err
		}
	}
	return logits, nil
}

// generateInto is the shared prefill+decode loop behind Model.Generate and
// Session.Generate. It assumes cache already holds prompt[:prefillFrom] (0 for a
// fresh generation), prefills prompt[prefillFrom:] (always ≥1 token — the seed,
// whose last position's logits start the decode), then decodes up to maxTokens,
// streaming each id to out. It does NOT close out: the caller owns the channel so
// it can run post-generation bookkeeping (e.g. a Session reconciling its token
// list) before the consumer observes the close. commit, if non-nil, is invoked
// with each id once its forward has committed that position to the cache — the
// seam Session uses to track exactly what the cache holds. Terminal status lands
// on g.err.
func (m *Model) generateInto(ctx context.Context, out chan<- int, g *Generation, cache *KVCache, prompt []int, prefillFrom, maxTokens int, sp SamplingParams, commit func(int)) {
	if len(prompt) == 0 {
		g.err = fmt.Errorf("decoder.Generate: empty prompt")
		return
	}
	sampler := NewSampler(sp)
	sampler.Observe(prompt...) // repetition penalties see the whole prompt, reused prefix included
	// Prefill the (divergent suffix of the) prompt and seed the first token's
	// logits. On the batched archs this runs the layers at M=len in one pass (each
	// weight streamed once — ~1.7–2× faster TTFT than sequential), LM head on the
	// last position only. Reuse means len here is the suffix, not the whole prompt.
	// GPU full-residency decode (webgpu + eligible arch + plain stateless
	// Generate). Sessions (commit != nil) and prefix-reuse (prefillFrom > 0) keep
	// the CPU/staged path. Prefill = option (a): run the prompt through the
	// resident DecodeRunner sequentially to build its GPU KV (also warms the
	// pipelines); the last token's logits seed decode. O(prompt-len) GPU Runs —
	// fast for typical prompts since a GPU Run ≫ a CPU int4 forward (the K/V-upload
	// bridge stays for future prefix-reuse; batched on-device prefill, the
	// long-prompt fix, is deferred).
	useGPU := m.resident != nil && prefillFrom == 0 && commit == nil
	if useGPU {
		// The resident path drives the model's ONE shared positional KV; two concurrent
		// generations would interleave writes at overlapping positions and corrupt it.
		// Claim it non-blockingly — a loser falls back to the staged CPU path, which uses
		// this call's own cache, so both still complete correctly (M9). The doc's
		// "distinct sequences can run concurrently" holds; only resident speed is lost.
		if atomic.CompareAndSwapInt32(&m.resBusy, 0, 1) {
			defer atomic.StoreInt32(&m.resBusy, 0)
		} else {
			useGPU = false
		}
	}
	gpuPos := 0
	var logits []float32
	var err error
	if useGPU {
		// Prefix reuse: skip the leading tokens already committed to the resident positional
		// KV and prefill only the divergent suffix. Forget FIRST — from here until the
		// generation completes the cache is mid-write, and any early return must leave the
		// next turn cold rather than trusting a half-written cache (resident_reuse.go).
		reuseFrom := m.residentReuseLen(prompt)
		m.residentForgetIDs()
		if logits, err = m.residentPrefillSeed(ctx, prompt, reuseFrom); err != nil {
			g.err = err
			return
		}
		g.PrefillReused = reuseFrom
		gpuPos = len(prompt)
	} else {
		if logits, err = m.prefillLogits(ctx, prompt[prefillFrom:], cache); err != nil {
			g.err = err
			return
		}
	}
	// Greedy fast path: when the resident can pick the argmax on-device AND the sampler's
	// choice is exactly argmax(raw logits) with nothing else reading them, skip the
	// full-logits readback (594 KB/token at a 151936 vocab). Emitted tokens are identical;
	// GOINFER_NO_GREEDY_FASTPATH forces the logits path (escape hatch / A-B check).
	// fastNext >= 0 means "the resident already picked the next token"; the first token
	// still comes from the prefill logits through the sampler.
	// P1: `top_k=1` takes this path too (GreedyEquivalent), at ANY temperature — monotone scaling
	// preserves ordering and a one-token distribution is deterministic, so the emitted tokens are
	// the same ones greedy emits. Both predicates are consulted; neither is widened, so this
	// routing decision is the ONLY behaviour that changes.
	//
	// SCOPE — the speculative paths are deliberately NOT affected. They gate on `sp.Temperature <= 0`
	// directly (speculative.go, spec_grammar.go, spec_eagle.go, spec_ngram.go), never on these
	// predicates, so `top_k=1` with a temperature stays speculative-INELIGIBLE exactly as before.
	// That is the conservative half of P1: making it eligible would be correct (argmax verification
	// reproduces greedy, which top_k=1 equals) but is a second behaviour change, and it does not
	// ride along silently here.
	//
	// RNG: this path skips the per-token rng.Float64() draw that SampleWithInfo would make. That is
	// unobservable rather than merely harmless — under top_k=1 every step is deterministic, so no
	// later draw's VALUE can depend on the skipped ones, and no emitted token can differ. (The RNG
	// stream position does advance differently, which is why nothing may depend on it downstream.)
	fastNext := -1
	greedyRF, hasGreedy := m.resident.(ResidentGreedy)
	fastGreedy := useGPU && hasGreedy && sp.LogitProcessor == nil &&
		(sampler.ArgmaxEquivalent() || sampler.GreedyEquivalent()) &&
		os.Getenv("GOINFER_NO_GREEDY_FASTPATH") == ""

	// Optimistic forward: sampled decode's (Temperature>0) sibling of the greedy fast path
	// above, but overlapping rather than skipping the CPU sampler -- see spec_optfwd.go.
	// Excludes fastGreedy's own (rare: Temperature>0 AND top_k==1) overlap with this predicate
	// so the two mechanisms never both try to drive the same step. GOINFER_NO_OPTFWD forces
	// the plain sequential path (escape hatch / A-B check), same convention as
	// GOINFER_NO_GREEDY_FASTPATH.
	optFwd := useGPU && !fastGreedy && m.optFwdEligible(sp) && os.Getenv("GOINFER_NO_OPTFWD") == ""
	var optGate *optFwdGate
	if optFwd {
		optGate = &optFwdGate{}
		g.OptFwd = &OptFwdStats{}
	}

	// Clamp the decode length to the resident KV cap up front (C3/M20). A Forward past
	// the cap is refused mid-generation (the silent-corruption guard), but a resident
	// backend that exposes its cap lets us stop cleanly AT it instead of erroring after
	// N tokens. gpuPos is the next decode position; the last valid one is ctxCap-1, and
	// the prefill loop already guarantees gpuPos <= ctxCap.
	if useGPU {
		if capper, ok := m.resident.(ResidentCapped); ok {
			if ctxCap := capper.ContextCap(); ctxCap > 0 && gpuPos+maxTokens > ctxCap {
				maxTokens = ctxCap - gpuPos // may be 0 (prompt used the whole context)
				g.BudgetClamped = true      // the resident cap (not the request) bounds this turn (M-04/R-09)
			}
		}
	}
	// Publish the effective budget so the caller reports finish_reason "length" when this
	// clamp (not an EOS) ends the turn (audit M-04). Set before any send; read after close.
	// BudgetClamped disambiguates a genuine clamp-to-0 (prompt fills the cap → "length") from an
	// unclamped turn whose Budget is coincidentally 0 (R-09).
	g.Budget = maxTokens

	// Decode loop.
	var generated []int
	var tProc, tSample, tEmbed, tFwd time.Duration
	var nFwd int
	for range maxTokens {
		select {
		case <-ctx.Done():
			g.err = ctx.Err()
			// V-04 (docs/review-2026-09-04.md): a cancel that arrives during the PREVIOUS
			// iteration's Forward -- the dominant per-iteration cost, milliseconds against the
			// send-select's microseconds -- is not observed until here, at the top of the next
			// iteration, because nothing between generated=append(...,next) and Forward
			// returning is select-guarded. By the time we reach this select, that iteration's
			// `next` has both been appended to `generated` AND had its own Forward already run
			// (the resident cache write happens synchronously inside it), so the cache is
			// exactly as consistent as the natural-completion commit below -- this was the
			// ORIGINAL R-02 fix's blind spot: it only committed at the send-select exit, which
			// is the rarer of the two cancel-observation points, not the common one.
			if useGPU {
				m.residentCommitIDs(prompt, generated)
			}
			return
		default:
		}
		var t0 time.Time
		if decodeTiming {
			t0 = time.Now()
		}
		var next int
		var info SampleInfo
		var optResolved bool
		var optNextLogits []float32
		if fastNext >= 0 {
			// The resident already picked this token's argmax on-device (greedy fast
			// path): nothing reads logits this step, so there is nothing to process or
			// sample. Identical to the logits path — guarded by ArgmaxEquivalent/GreedyEquivalent.
			next = fastNext
		} else if optFwd && optGate.Should() {
			// Optimistic forward (spec_optfwd.go): samples AND resolves gpuPos+1's logits
			// together, overlapping the real sampler with a speculative Forward on a free
			// argmax guess. decodeTiming intentionally does not split this into
			// tProc/tSample/tEmbed — the whole overlapped operation folds into tFwd below
			// (t0 was set at the top of this iteration, untouched here), since sample and
			// forward no longer have a clean sequential boundary to attribute separately.
			res, operr := m.optFwdStep(sampler, logits, gpuPos, optGate, g.OptFwd)
			if operr != nil {
				g.err = operr
				return
			}
			info = res.info
			next = info.ID
			optNextLogits = res.nextLogits
			optResolved = true
		} else {
			// Constrained decoding: let the processor mask this step's logits
			// (based on what's been generated) before sampling and the stop check.
			if sp.LogitProcessor != nil {
				sp.LogitProcessor(generated, logits)
			}
			if decodeTiming {
				tProc += time.Since(t0)
				t0 = time.Now()
			}
			var serr error
			info, serr = sampler.SampleWithInfo(logits)
			if serr != nil {
				g.err = serr
				return
			}
			if decodeTiming {
				tSample += time.Since(t0)
			}
			next = info.ID
		}
		if m.isStop(next, sp) {
			break
		}
		// Both fast-path predicates exclude Logprobs, so the fast path never reaches this.
		if sp.Logprobs {
			g.Logprobs = append(g.Logprobs, info)
		}
		// A bare send wedges this goroutine forever if the consumer stops ranging
		// (even to cancel ctx, the documented stop) — it holds the KV cache and, for
		// Session.Generate, blocks the post-loop reconciliation, poisoning the session.
		// Select on ctx.Done like every speculative path (M8).
		select {
		case <-ctx.Done():
			g.err = ctx.Err()
			// audit R-02: at this exit `next` has been sampled but never forwarded (its own
			// K/V write is later in this same iteration, which cancellation skips), so the
			// resident cache is consistent with exactly prompt+generated — record it instead
			// of leaving resIDs nil and forcing the next turn to cold-prefill an interrupt
			// that left nothing inconsistent behind. Agent harnesses cancel constantly
			// (interrupts, timeouts, disconnects), so today every one of them pays this cost.
			if useGPU {
				m.residentCommitIDs(prompt, generated)
			}
			return
		case out <- next:
		}
		generated = append(generated, next)
		if optResolved {
			// optFwdStep already produced (or redid) this position's Forward and its
			// result logits for gpuPos+1 — nothing left to do but advance the position.
			logits = optNextLogits
			gpuPos++
		} else if useGPU {
			var emb []float32
			if decodeTiming {
				t0 = time.Now()
				emb = m.embedResident(next)
				tEmbed += time.Since(t0)
				t0 = time.Now()
			} else {
				emb = m.embedResident(next)
			}
			if fastGreedy {
				// Greedy fast path: the resident picks the argmax on-device and returns
				// just the id, skipping the full-logits readback.
				fastNext, err = greedyRF.ForwardArgmax(emb, gpuPos)
			} else {
				logits, err = m.resident.Forward(emb, gpuPos)
			}
			gpuPos++
		} else {
			logits, err = m.forward(next, cache)
		}
		if decodeTiming {
			tFwd += time.Since(t0)
			nFwd++
		}
		if err != nil {
			g.err = err
			return
		}
		if commit != nil {
			commit(next)
		}
	}
	// The ONLY place the resident cache's contents are recorded: a generation that ran to
	// completion. Every other exit above left resIDs nil, so the next turn cold-prefills.
	if useGPU {
		m.residentCommitIDs(prompt, generated)
	}
	if decodeTiming && nFwd > 0 {
		ms := func(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 / float64(nFwd) }
		fmt.Printf("DECODE TIMING (%d tok, gpu=%v): forward %.1f ms | sample %.2f ms | logitProc %.2f ms | embed %.2f ms /token\n",
			nFwd, useGPU, ms(tFwd), ms(tSample), ms(tProc), ms(tEmbed))
	}
}

// isStop reports whether id ends generation: a checkpoint EOS id (from
// config) or a caller-supplied stop id (SamplingParams.StopIDs, e.g.
// <end_of_turn> for chat).
func (m *Model) isStop(id int, sp SamplingParams) bool {
	if slices.Contains(m.eosIDs, id) {
		return true
	}
	return slices.Contains(sp.StopIDs, id)
}

// Generation carries the terminal status of a Generate stream. Spec is non-nil
// for GenerateSpeculative and carries acceptance telemetry.
type Generation struct {
	err error
	// PrefillReused is how many leading prompt tokens this generation skipped because they
	// were already committed to the resident positional KV (resident_reuse.go). 0 on a cold
	// prefill and on every non-resident path. Diagnostic: it is what makes an agent loop's
	// per-turn prefill cost visible without timing it.
	PrefillReused int
	Spec          *SpecStats
	OptFwd        *OptFwdStats // non-nil when optFwdEligible held for this run; see spec_optfwd.go
	// Logprobs holds one entry per emitted token (in order) when
	// SamplingParams.Logprobs was set — the chosen token's log-probability and
	// any requested top alternatives. Complete once the stream has closed.
	Logprobs []SampleInfo
	// Budget is the effective max-token budget after the resident context-cap clamp
	// (audit M-04). It equals the requested maxTokens unless prompt+maxTokens would
	// exceed the resident KV cap, in which case it is the remaining room (may be 0).
	// A caller that reports finish_reason must compare the emitted count against this,
	// not the requested value, or a context-clamped generation is mis-reported as a
	// clean "stop" and the client never continues. Set before the first token is sent;
	// read it after the channel closes (like Err). 0 on generation paths that don't
	// clamp (speculative/VL).
	Budget int
	// BudgetClamped is true iff the resident context cap (not the request) bounded this turn, so
	// Budget is the authoritative limit — including a clamp to 0 when the prompt fills the whole
	// context. A caller must judge finish_reason against Budget only when this is set; otherwise it
	// falls back to the requested max_tokens. Without it, a genuine clamp-to-0 is indistinguishable
	// from an unclamped Budget-0 turn and an empty-context-full response mis-reports "stop" (R-09).
	BudgetClamped bool
}

// Err returns the error that ended the stream, or nil if it ended cleanly
// (EOS / stop / maxTokens). Read it after the channel closes.
func (g *Generation) Err() error { return g.err }
