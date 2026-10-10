package decoder

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/aikit/mmap"
	"github.com/townsendmerino/goinfer/internal/giw"
)

// decodeTiming env-gates per-component decode timing (GOINFER_DECODE_TIMING=1) so
// the residency path's per-token cost can be decomposed without touching the hot
// path otherwise.
var decodeTiming = os.Getenv("GOINFER_DECODE_TIMING") != ""

// dt* are decodeTiming's fine split of forward (attention: q/k/v matmuls, the rope+KV+scores/softmax/AV core, o-proj; MLP:
// gate+up matmuls, the activation, down; the LM head), accumulated in ns behind the same env gate (one bool check per site
// when off) and printed with the DECODE TIMING line, so any machine can read its own split without a patch.
var dtAttn, dtMLP, dtHead, dtGU, dtAct, dtActDown, dtQKV, dtO, dtAttnAll int64

// lastDecodeSplit is the most recent DECODE TIMING split in ms/token, for tests that A/B a
// component in-process (decodeTiming must be on for it to fill).
var lastDecodeSplit struct{ fwd, attn, mlp, head, gu, act, down, qkv, core, o float64 }

// Model is a loaded checkpoint plus the compute backend. Goroutine safety follows encoder.Model: Weights are immutable after
// Load; per-sequence state (the KV cache) is owned by each Generate call, so distinct sequences can run concurrently, but a
// single KVCache is not shared.
type Model struct {
	w        *Weights
	be       Backend
	eosIDs   []int           // end-of-sequence ids from config (generation stops on these)
	resident ResidentForward // GPU full-residency decode path (webgpu + eligible arch); nil ⇒ staged/CPU
	resBusy  int32           // atomic: claims the single shared resident KV for one in-flight generation. Raw int32 (not atomic.Bool) so Model stays copyable for the value-copy test seam.
	// batcher: MC3's coordinator (mc3_batch.go) — several generations on the resident at once, each on its own KV
	// slot, their decode tokens joined into shared steps. nil unless EnableResidentConcurrency enabled it.
	batcher *residentBatcher
	// specAdaptive: MC4's "speculate when alone, batch under load" candidate (SetSpecAdaptive,
	// docs/tasks/task-concurrency-2026-09.md). off by default — genNgramInto's exclusive claim is unchanged unless
	// this is on AND batcher is non-nil.
	specAdaptive bool
	// cpuBatch: MC3c step 2's coordinator (cpu_batch.go) — several CPU generations' decode tokens joined into one
	// batched forward. nil unless EnableCPUBatch enabled it. cpuBatchMode is Options.CPUBatchDecode.
	cpuBatch     *cpuBatcher
	cpuBatchMode int
	// resIDs is the token sequence currently committed to the resident positional KV, or nil
	// when its contents are unknown. Guarded by the same resBusy claim that serialises writes
	// to that cache; see resident_reuse.go for why nil is the safe default.
	resIDs     []int
	resIDsLora *loraRuntime // the adapter that built resIDs' KV, nil = base (resident_reuse.go)
	// resImgBlocks records every image block committed within resIDs (P9a, docs/multimodal.md)
	// — nil in the overwhelming common case (no image ever touched this resident KV). Cleared
	// together with resIDs by residentForgetIDs, always; see resident_reuse.go.
	resImgBlocks []residentImageBlock
	// resDrafterSynced identifies which *BlockSpec's own drafter context is in sync with resIDs; nil means no drafter context is
	// trustworthy for reuse. resIDs alone is not enough: a plain Generate or n-gram-speculative turn can commit resIDs without
	// touching a block drafter's context, so the token prefix can match while the drafter's state does not reflect it. Cleared
	// by residentForgetIDs (every resident write invalidates it) and set only by BlockSpec.generate's fully-completed exit, the
	// one path that keeps resIDs and the drafter's context advancing together.
	resDrafterSynced *BlockSpec
	// resSlots / resCur / resTick: MC1's resident KV slots (resident_reuse.go residentAcquire). nil / 0 / 0 with one
	// slot; the bound slot's bookkeeping lives in the resIDs fields above, the others' is parked in resSlots.
	resSlots      []residentSlot
	resCur        int
	resTick       uint64
	kvF16         bool         // residency KV cache precision request (Options.KVPrecision == "f16")
	kvPrecI8      bool         // residency KV cache int8 request (Options.KVPrecision == "i8") — GPU
	kvI8          bool         // CPU KV cache int8 storage request (Options.KVQuant == "i8") — CPU staged path
	exactPrefill  bool         // Options.ExactPrefill: THIS model's prompt ingestion stays bit-exact on every backend (ExactPrefill())
	backendAuto   bool         // Options.BackendAuto: the backend was chosen by "auto", not named (withResidency's Metal precision guard)
	noSelfTest    bool         // Options.noSelfTest: this model IS a self-test fixture, so withResidency must not probe for it (selftest_gpu.go)
	knobs         *knobSet     // per-model operator knobs, snapshotted once at Load (knobs.go)
	resCtxReq     int          // requested GPU-resident KV capacity in positions (Options.ResidentContext); 0 ⇒ backend default
	resSlotsReq   int          // requested resident KV slot count (Options.ResidentKVSlots); 0/1 ⇒ one slot
	resSlotsDef   bool         // Options.ResidentKVSlotsDefault: resSlotsReq is the caller's default (ResidentKVSlotsIsDefault)
	windowedKV    bool         // Options.ResidentWindowedKV
	resCtxPinned  bool         // the caller chose resCtxReq (not the fit guard's auto-pin) — ResidentContextPinned
	prefillChunk  int          // Options.ResidentPrefillChunk: MC3 chunked prefill's chunk size, 0 = off
	disableFit    bool         // tasks/task-fit-to-hardware.md --fit=off (Options.DisableFit) — see FitDisabled's own doc comment
	moeCache      bool         // stream routed MoE experts host→VRAM (Options.MoECacheExperts)
	moeSlots      int          // per-layer expert slot request (Options.MoECacheSlots); 0 ⇒ ask for all, auto-cap to VRAM
	actGroup      int          // activation quantization group (Options.ActQuantGroup); 0 = per-vector
	extraBytes    int64        // Options.ExtraResidentBytes — see that field's own doc comment
	extraKVPerPos int64        // Options.ExtraResidentKVPerPosition — see that field's own doc comment
	mmap          []byte       // .giw mmap region the int8/int4 weights alias; munmap'd by Close (nil off the .giw mmap path)
	srcPath       string       // the .giw path this model mmap-loaded from ("" off the .giw path) — for pread-staging over the same file
	pager         *expertPager // MoE expert demand-paging over the mapping (Options.StreamWeights); nil = all-resident
	layerPager    *layerPager  // dense per-layer streaming over the mapping (Options.StreamWeights); nil = all-resident
	quant         string       // the requested Options.Quant for a direct load ("" for a prequant .giw → Quant() derives from kinds)
	// resDecline records WHY resident is nil on a non-CPU backend — the reason withResidency
	// would otherwise discard. Empty when residency was built, or when it was never attempted
	// (CPU backend). DecodePath / -require-backend read it; see withResidency.
	resDecline string

	// prof is this load's phase timing, when the loader instrumented it. nil otherwise.
	prof *LoadProfile

	// reqBackend / effBackend record what the caller asked for and what is actually executing. They differ when NewBackend
	// falls back (`--backend metal` on a build without the metal submodule runs on CPU), so the load banner can name the
	// backend that is running (BackendSummary).
	reqBackend string
	effBackend string
	// beDecline is NewBackend's fallback note, if any — the reason effBackend != reqBackend.
	beDecline string

	// adapters holds compute-time LoRA adapters loaded against this base, behind a pointer so *Model stays value-copyable: the
	// kvi8 test seam does `mm := *m`, and a sync.Mutex field would make `go vet` reject the copy (resBusy is a raw int32 for the
	// same reason). nil until the first LoadAdapter. The mutex inside guards concurrent LoadAdapter vs UseAdapter/HasAdapter.
	adapters *adapterRegistry
}

// adapterRegistry is the compute-time LoRA state, heap-held so *Model has no lock value.
type adapterRegistry struct {
	mu     sync.Mutex
	byName map[string]*loraRuntime
	// retired holds runtimes displaced by a re-registration of the same name. A live Session may still hold the old *loraRuntime in
	// its cache.lora and read its mmap'd deltas mid-generation, so it must not be munmap'd on re-register (that SIGSEGVs the
	// reader). Released only at Model.Close: a small, bounded leak (one entry per re-registration of a live name) traded for
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

// Int4ScalesF16 returns w's group scales as f16 bit patterns (F16Bits) when this model was loaded from a v14
// metal-target .giw that stores them — aliased from the mapping, so a Metal no-copy buffer can bind them in
// place instead of converting the f32 scales into a new buffer — or ok=false (any other load, or a tensor the
// file did not give f16 scales to). Keyed by the nibbles' address: pass the WeightMat the model itself holds.
func (m *Model) Int4ScalesF16(w *linalg.WeightMat) ([]uint16, bool) {
	if m == nil || m.w == nil || m.w.int4F16 == nil || w == nil {
		return nil, false
	}
	q4, _, _, ok := w.Int4F16()
	if !ok || len(q4) == 0 {
		return nil, false
	}
	f, ok := m.w.int4F16[uintptr(unsafe.Pointer(&q4[0]))]
	return f, ok
}

// protectGIWMapping is applied to every .giw mapping Load creates (excludeFromFork). A variable only so a
// test can observe that Load really calls it on the whole mapping.
var protectGIWMapping = excludeFromFork

// MmapAliasWindow returns the page-aligned window of this model's .giw mapping that encloses b, for
// a consumer that wants to wrap those pages without copying them (Metal's newBufferWithBytesNoCopy
// needs a page-aligned base and a page-multiple length): base is the pointer to the window's first
// byte, n its length (a multiple of pageSize), and off b's offset inside it. ok=false when b does
// not alias the mapping (a heap-backed weight, or no .giw mapping), or pageSize is not a power of two.
//
// The window can extend past b at both ends, and past len(mmap) by up to the rest of the last page —
// the OS maps whole pages, so those bytes are readable (zero-filled past the file's end). The window
// only NAMES pages; nothing here touches them (no fault). The caller must keep the model — and so
// the mapping — alive for as long as anything it built over the window is (Model.Close unmaps after
// the resident closes).
func (m *Model) MmapAliasWindow(b []byte, pageSize int) (base unsafe.Pointer, n, off int, ok bool) {
	if pageSize <= 0 || pageSize&(pageSize-1) != 0 {
		return nil, 0, 0, false
	}
	o, in := m.MmapByteOffset(b)
	if !in {
		return nil, 0, 0, false
	}
	mapBase := uintptr(unsafe.Pointer(&m.mmap[0]))
	if mapBase&uintptr(pageSize-1) != 0 { // the mapping is not page-aligned at this page size
		return nil, 0, 0, false
	}
	ps := int64(pageSize)
	lo := o &^ (ps - 1)
	hi := (o + int64(len(b)) + ps - 1) &^ (ps - 1)
	if limit := (int64(len(m.mmap)) + ps - 1) &^ (ps - 1); hi > limit {
		return nil, 0, 0, false
	}
	return unsafe.Add(unsafe.Pointer(&m.mmap[0]), lo), int(hi - lo), int(o - lo), true
}

// KVCacheF16 reports whether the GPU residency path should use an f16 KV cache (Options.KVPrecision == "f16"): twice the
// context on the same VRAM, lossy. The residency builder reads it; off the residency path it has no effect.
func (m *Model) KVCacheF16() bool { return m.kvF16 }

// KVCacheI8 reports whether the GPU residency path should use an int8 KV cache (Options.KVPrecision == "i8"): lossy, 4x
// smaller than f32 and 2x smaller than f16. Distinct from KVQuant (the CPU cache).
func (m *Model) KVCacheI8() bool { return m.kvPrecI8 }

// ResidentContextRequest returns the requested GPU-resident KV capacity in positions
// (Options.ResidentContext), or 0 for "use the backend default". The residency builder resolves the
// effective cap as min(model context window, this) and VRAM-checks it at load; off the residency
// path it has no effect. See cuda.resolveCtxCap.
func (m *Model) ResidentContextRequest() int { return m.resCtxReq }

// ResidentContextPinned reports whether the caller chose the resident context (Options.ResidentContext > 0), as opposed to
// leaving it to the backend. A context the load-time fit guard auto-pinned for an unrequested load is not pinned: it is a
// ceiling that fits one KV slot, and ResidentContextRequest still reports it as the upper bound. A backend that trades
// context for KV slots shrinks only an unpinned context, never an explicit -ctx.
func (m *Model) ResidentContextPinned() bool { return m.resCtxPinned }

// ResidentKVSlotsIsDefault reports Options.ResidentKVSlotsDefault: the slot request is the caller's default, which a backend
// may lower to its own (Metal: 2).
func (m *Model) ResidentKVSlotsIsDefault() bool { return m.resSlotsDef }

// WindowedKVSlack is the headroom, in positions, a windowed-KV layer holds beyond its window: a layer keeps window+WindowedKVSlack
// positions and compacts (copies its live tail to the start) once per slack positions written. It must cover the widest batch of rows
// written in one pass: a prefill chunk (512 by default), a verify batch, an MC3 row. The CUDA allocation and decoder's pricing both use
// it, so priced bytes equal allocated bytes.
const WindowedKVSlack = 512

// ResidentWindowedKVCap is how many positions a windowed-KV layer of the given window holds at a resident context of ctx: the whole
// context when that is no larger than the window plus its slack, else window+WindowedKVSlack.
func ResidentWindowedKVCap(window, ctx int) int { return min(ctx, window+WindowedKVSlack) }

// ResidentWindowedKV reports Options.ResidentWindowedKV: the caller asked for windowed K/V on a backend that supports it.
func (m *Model) ResidentWindowedKV() bool { return m.windowedKV }

// WindowedKVLayer reports whether layer l's resident K/V is windowed under Options.ResidentWindowedKV: the option is on and the layer is
// a sliding-window attention layer with a real K/V cache (not an MLA latent, not a recurrent mixer, whose kvDimAt is not a K/V width
// bounded by a window). Every windowed layer shares SlidingWindowResident's one window.
func (m *Model) WindowedKVLayer(l int) bool {
	if !m.windowedKV || m.SlidingWindowResident() <= 0 || !m.LayerIsLocalResident(l) {
		return false
	}
	a := m.w.arch
	return a != nil && a.mla == nil && a.kvDimAt(l) > 0
}

// ResidentKVSlotsRequest returns the requested number of resident KV slots (Options.ResidentKVSlots), at least 1, and 1 for a
// family with recurrent state. A residency builder that supports slots (ResidentKVSlotter) allocates up to this many,
// clamped by its fit guard.
func (m *Model) ResidentKVSlotsRequest() int {
	if m.hasRecurrentState() {
		return 1 // its state is not part of a KV slot (resident_reuse.go residentSlotCount)
	}
	return max(1, m.resSlotsReq)
}

// ResidentKVSlots reports how many resident KV slots this model's generations choose among — what the resident
// allocated (ResidentKVSlotter), 1 for a resident without slots, 0 off the resident path.
func (m *Model) ResidentKVSlots() int {
	if m.resident == nil {
		return 0
	}
	return m.residentSlotCount()
}

// ExtraResidentBytes returns Options.ExtraResidentBytes: VRAM a companion allocation (a --drafter's weights; a vision tower
// is the same class) will claim on the same device after this model's own residency is built, priced ahead of time so the
// elastic terms a backend sizes against live free VRAM (CUDA's capSlots expert cache, resolveCtxCapFit's
// context-by-default) leave room for it instead of taking everything free and having the later attach fail. 0 means nothing
// else is attaching.
func (m *Model) ExtraResidentBytes() int64 { return m.extraBytes }

// ExtraResidentKVPerPosition returns Options.ExtraResidentKVPerPosition: the rate at which a companion allocation's device
// K/V scales with the resident context THIS model ends up choosing, which ExtraResidentBytes (priced in loadDecoder, before
// this model's residency exists) cannot know. A residency builder that knows its own candidate/final ctx multiplies by it
// locally (see decoder.DrafterKVBytesPerPosition). 0 means the companion, if any, has no ctx-scaling K/V term.
func (m *Model) ExtraResidentKVPerPosition() int64 { return m.extraKVPerPos }

// FitDisabled is --fit=off (Options.DisableFit), also set by GOINFER_NO_FIT_DEFAULT (kept working as the original, narrower
// escape hatch): every "fit by default" behavior reverts to its previous default. Checked by backend packages that implement
// a fit-by-default policy (cuda.resolveCtxCapFit); decoder itself has none to gate. See docs/tasks/task-fit-to-hardware.md.
func (m *Model) FitDisabled() bool {
	return m.disableFit || m.knobs.get(knobNoFitDefault) != ""
}

// MoECacheExperts reports whether routed MoE experts should stream host→VRAM per token instead of
// being held resident (Options.MoECacheExperts, `--moe-cache-experts`). This is what lets a model
// whose experts exceed VRAM run with every expert still executing ON the GPU. Off by default:
// running a model larger than your card is a deliberate act, it costs a per-token PCIe transfer,
// and with it off the runtime declines honestly instead of silently going slow.
//
// Falls back to GOINFER_MOE_CACHE_EXPERTS so existing scripts keep working.
func (m *Model) MoECacheExperts() bool {
	return m.moeCache || m.knobs.get(knobMoECacheExperts) != ""
}

// MoECacheSlotsRequest returns the requested per-layer expert-slot count, or 0 for "as many as fit"; only meaningful with
// MoECacheExperts. 0 asks for ALL experts and lets the builder cap to measured free VRAM, deliberately: topK slots is the
// worst setting for the situation this applies to, since it degenerates to fresh-loading every routed expert every token.
// allocSlots measures free VRAM and caps-and-logs rather than OOMing, so asking for all is safe. Falls back to
// GOINFER_MOE_CACHE_SLOTS.
func (m *Model) MoECacheSlotsRequest() int {
	if m.moeSlots > 0 {
		return m.moeSlots
	}
	if v, err := strconv.Atoi(m.knobs.get(knobMoECacheSlots)); err == nil && v > 0 {
		return v
	}
	return 0
}

// Options configures Load.
type Options struct {
	noSelfTest bool   // set only by the resident self-test's own fixture loads (selftest_gpu.go); unexported so the type stays comparable and the API unchanged
	Backend    string // "cpu" (default), "webgpu", "cuda", "metal", or "auto" (AutoBackend); a name not compiled in falls back to cpu
	Quant      string // "" (f32), "int8" (weight-only per-row), "int8int8" (full int8×int8 W8A8), or "int4" (group-wise)
	LoRA       string // optional PEFT adapter dir (adapter_config.json + adapter_model.safetensors), merged into the base at load. Safetensors base only.
	// KVPrecision selects the GPU residency KV cache precision: "" / "f32" (default, bit-exact), "f16" (lossy, twice the
	// context) or "i8" (lossy, four times). Ignored off the residency path. See task-gpu-f16-kv.md / task-gpu-kv-i8.md.
	KVPrecision string
	// MoECacheExperts streams routed MoE experts host→VRAM per token instead of holding the whole
	// expert stack resident — the path to running a model whose experts exceed VRAM with every
	// expert still executing on the GPU. Off by default; bit-identical to fully-resident when on
	// (cuda.TestGemma4MoE_cacheExpertsBitExact_*; on Metal metal.TestMoECacheExperts_bitExactMetal, against
	// a resident load that prefills the same way: a paged Metal model's prompt takes a prefill
	// bit-identical to sequential, so beside the default batched f16 prefill a prompt over its floor
	// differs exactly as ExactPrefill does). The experts must be int4: Metal declines int8 experts by
	// name. CUDA and Metal residency; the CPU's expert paging is StreamWeights.
	MoECacheExperts bool
	// MoECacheSlots is the per-layer expert-slot count for MoECacheExperts (0 = ask for all and
	// auto-cap to measured free VRAM). More slots ⇒ higher LRU hit rate ⇒ fewer per-token DMAs,
	// at VRAM cost. Only meaningful with MoECacheExperts.
	MoECacheSlots int
	// KVQuant selects the CPU KV cache storage precision: "" / "f32" (default, bit-exact) or "i8" (per-(position,KV-head)
	// symmetric int8, 4x smaller, SDOT decode). Lossy, opt-in; declined where Architecture.kvInt8OK says no (MoE, and own-forward
	// families not marked KVInt8). See task-cpu-kv-quant.md.
	KVQuant string
	// StreamWeights enables on-demand weight residency (idea #2): for an mmap-backed
	// .giw MoE model, expert weights are paged out of the mapping under a RAM budget
	// (WeightCacheBytes) instead of all held resident. Bit-exact (read-only re-fault);
	// trades RAM for cold-miss fault latency. No-op for non-MoE / non-.giw models.
	StreamWeights bool
	// WeightCacheBytes is the resident-bytes budget for streamed weights (0 = auto,
	// ~half of available RAM). Only meaningful with StreamWeights.
	WeightCacheBytes int64
	// AcceptSlowMoE is the explicit acknowledgement a paged-MoE StreamWeights load needs when its predicted working-set rate
	// falls below moeSlowTokPerSecThreshold (moeworkingset.go): without it the load is refused. Only meaningful for a .giw MoE
	// load under StreamWeights; a no-op everywhere else.
	AcceptSlowMoE bool
	// MoEPager is the CPU expert pager's backing mode for a .giw-paged MoE model: "mmap" (advice-based,
	// zero-copy) or "pool" (owned buffers + pread; a firm cap on every platform). "" = the platform
	// default (MoEPagerDefault: pool on darwin, mmap elsewhere), unless GOINFER_MOE_PREAD_CPU overrides.
	MoEPager string
	// Knobs sets per-model operator knobs by environment-variable name (knobs.go's list, e.g.
	// "GOINFER_FUSED_ATTENTION": "0"), overriding the process environment for THIS model only. Unset names
	// take the environment's value, read once at Load. Unknown names are ignored. A pointer, so Options
	// stays comparable (a hard-tier API property, docs/api-tiers.md): Knobs: &decoder.Knobs{...}.
	Knobs *Knobs
	// EmbedInt4 relaxes the int8 pin on the token-embedding/LM-head table in int4 mode, storing it at int4 too: this halves the
	// single largest resident tensor on a big-vocab small model, at a lossy cost (mostly on rare tokens). Off in a zero Options,
	// which keeps the bit-exact int8 pin; the CLIs turn it on by default (internal/loadflags). GGUF load path only.
	EmbedInt4 bool
	// ResidentContext requests a GPU-resident KV capacity in positions. 0 (default) keeps the
	// backend's built-in default, so nobody who did not ask allocates deep-KV VRAM. When set, the
	// backend caps it at the model's own context window — the effective cap is
	// min(model context window, this) — and fails at LOAD if the KV that implies does not fit
	// beside the weights, rather than OOM-ing mid-decode. Ignored off the residency path.
	ResidentContext int
	// DisableFit is --fit=off: it restores every "fit by default" behavior to its previous default. Currently: CUDA's unpinned
	// resident context stays the flat historical constant instead of asking Plan for more when there is room
	// (cuda/resident.go's resolveCtxCapFit), and a dense .gguf that will not fit resident RAM stays a plain refusal instead of
	// getting an automatic -stream-weights retry (guardFit in fitguard.go, read by internal/modelload's Load, not by Load itself).
	// An explicitly pinned request (ResidentContext, MoECacheSlots, StreamWeights, ...) is never affected in either direction:
	// fit-by-default only acts on the unpinned case. See docs/tasks/task-fit-to-hardware.md.
	DisableFit bool
	// ExtraResidentBytes prices a companion allocation that will claim VRAM on the same device after this model's own residency
	// is built (a --drafter's weights; internal/serveapp's loadDecoder computes it via decoder.DrafterResidentBytesEstimate before
	// calling Load). See Model.ExtraResidentBytes. 0 (the default) means nothing else is attaching.
	ExtraResidentBytes int64
	// ExtraResidentKVPerPosition is ExtraResidentBytes' ctx-scaling twin: a companion allocation's device K/V scales with the
	// resident context THIS model ends up choosing, which is unknowable when ExtraResidentBytes is priced (in loadDecoder). See
	// Model.ExtraResidentKVPerPosition and decoder.DrafterKVBytesPerPosition. 0 (the default) means no such term.
	ExtraResidentKVPerPosition int64
	// ExactPrefill forces bit-exact prompt ingestion on every backend that has a faster, non-exact default: CUDA's tensor-core
	// batched prefill (GOINFER_CUDA_FAST_PREFILL), Metal's f16-MMA batched prefill (GOINFER_METAL_FAST_PREFILL) and CPU's
	// f32-attention fast path (GOINFER_CPU_FAST_ATTENTION), all default ON above their own thresholds. false (the default)
	// leaves whatever env state the process already has untouched, so a caller managing these knobs itself (serve's
	// --cpu-exact-prefill and --cpu-fast-attention, more granular than this bool, travel through Options.Knobs) is not
	// overridden. This is the library-level chokepoint docs/completed/task-prefill-gap.md describes; --exact-prefill sets it.
	ExactPrefill bool
	// ResidentKVSlots asks a GPU-resident backend for this many independent KV caches ("slots"), so several interleaved
	// conversations each keep their own prefix resident instead of evicting one another's (docs/tasks/task-concurrency-2026-09.md).
	// Still one generation at a time unless MC3 is enabled: a slot is bound per generation. 0 or 1 = one slot. A backend clamps it
	// to what its fit guard allows and says so; a backend that does not implement ResidentKVSlotter, and every family with
	// recurrent state, keep one slot. serve sets it from -kv-sessions.
	ResidentKVSlots int
	// ResidentKVSlotsDefault says ResidentKVSlots is the caller's default, not a count the operator chose (serve sets it when
	// -kv-sessions was not given). A backend may then lower it to its own default: Metal keeps 2 slots, since every slot's KV is
	// resident from the first token on unified memory.
	ResidentKVSlotsDefault bool
	// ResidentWindowedKV asks the CUDA resident to keep only a window's worth of K/V (plus WindowedKVSlack positions of headroom) for each
	// sliding-window layer instead of the whole context, which lets a windowed model such as Mellum2.1 fit a long context fully resident
	// beside its weights (docs/tasks/task-cuda-windowed-kv-2026-10.md). Off by default: it changes how much memory a load takes. A
	// backend or family that cannot honour it keeps the full cache and says why (ResidentWindowedKVDecline); the decode is bit-identical
	// either way. Only CUDA reads it today.
	ResidentWindowedKV bool
	// ResidentPrefillChunk, under MC3 (EnableResidentConcurrency), prefills a long prompt suffix in chunks of this many tokens
	// while other generations are decoding, one decode step between chunks, instead of in one pass that stalls them all for the
	// whole prompt (docs/tasks/task-concurrency-2026-09.md). 0 = off: whole prefill. Sound only where the resident's batched
	// prefill is chunk-invariant (Metal's is: TestMC5_prefillChunkInvariance). serve defaults it to 512, the graded value
	// (docs/measurements/chunked-prefill-2026-09-27.md).
	ResidentPrefillChunk int
	// CPUBatchDecode chooses whether concurrent CPU generations of this model join their decode tokens into one batched
	// forward (MC3c step 2, docs/tasks/task-concurrency-2026-09.md; Model.EnableCPUBatch): CPUBatchAuto (0, the
	// default) batches an eligible model with at least 2 GiB of dense weights; CPUBatchOn batches every
	// eligible model; CPUBatchOff keeps step 1's independent workers. Every reply is bit-identical either way.
	CPUBatchDecode int
	// ActQuantGroup selects per-group activation quantization for the int8-activation projections (int4 = W4A8, int8int8 =
	// W8A8, int4mix): 0 (the default) scales each activation vector by one max/127, 32 gives every 32 inputs their own scale. A
	// family with massive activation outliers (Phi-3) loses nearly the whole vector under one scale; per-32 keeps an outlier's
	// damage inside its group (docs/tasks/task-actquant-pergroup-2026-09.md). Per model: two models in one process may differ.
	// Honoured on the CPU and by CUDA residency; other resident backends decline to the CPU path when it is set.
	ActQuantGroup int

	// LoadAbort, if non-nil, is checked between layers during a direct (non-.giw) GGUF weight build: the load-time consumer of
	// the swap tripwire. Closing it aborts the load with an error satisfying errors.Is(err, ErrLoadAborted). Load does not know
	// why it closed (it does not own decoder.SwapWatch), so a caller arming a watch is expected to wrap the returned error with
	// its own reason/pricing detail. A nil channel (the zero value) blocks forever in a select, so it is a genuine no-op.
	//
	// Limitation: only the GGUF direct-build path (loadGGUFWeights/buildWeightsFromGGUF's resident, non-streaming branch) checks
	// it. The safetensors direct-build path and StreamTranscodeGGUF's transcode do not
	// (docs/code-notes/decoder.md#Options.LoadAbort).
	LoadAbort <-chan struct{}

	// BackendAuto says Backend was chosen by "auto" rather than named. A backend auto chose declines a model it would
	// run only at another precision (Metal re-quantizes int8 weights to int4), where a named one runs it as asked.
	// Load sets it when Backend is "auto"; a caller that resolved auto itself (the CLIs, for their own checks by name)
	// sets it beside the name.
	BackendAuto bool
}

// ErrLoadAborted is returned (wrapped) from Load when opts.LoadAbort closed mid-build. Check
// with errors.Is, not ==, since it may be wrapped with additional context by the time a caller
// sees it.
var ErrLoadAborted = errLoadAborted

// modelFromOptions is a Model over w carrying every per-model field Options sets: the one place the constructors (Load's
// .giw and direct paths, LoadGGUFBytes, NewModelWithOptions) read them, so they cannot drift apart (a constructor that skipped
// a field silently ignored that option). Callers set what is specific to their path (the requested quant, the resolved EOS
// ids, the file mapping), then apply backend names, knobs, streaming and residency in the order their path needs.
func modelFromOptions(w *Weights, be Backend, opts Options) *Model {
	// Stamp the activation group on every quantizable weight: the CPU matmul helpers read it from
	// the weight they hold, whichever backend object they run under.
	if opts.Quant == "q4k" && opts.ActQuantGroup == 0 {
		// q4k is per-32 by construction (its Q4_K kind cannot run otherwise); its int8 tensors follow.
		opts.ActQuantGroup = 32
	}
	if opts.ActQuantGroup > 0 {
		for _, wm := range w.matmulWeights() {
			wm.SetActQuantGroup(opts.ActQuantGroup)
		}
	}
	return &Model{w: w, be: be, eosIDs: w.Cfg.EOSIDs(),
		kvF16: opts.KVPrecision == "f16", kvPrecI8: opts.KVPrecision == "i8", kvI8: opts.KVQuant == "i8",
		resCtxReq: opts.ResidentContext, resCtxPinned: opts.ResidentContext > 0, // Load overrides after its fit guard
		resSlotsReq: opts.ResidentKVSlots, resSlotsDef: opts.ResidentKVSlotsDefault, windowedKV: opts.ResidentWindowedKV, prefillChunk: opts.ResidentPrefillChunk, disableFit: opts.DisableFit,
		cpuBatchMode: opts.CPUBatchDecode,
		moeCache:     opts.MoECacheExperts, moeSlots: opts.MoECacheSlots,
		extraBytes: opts.ExtraResidentBytes, extraKVPerPos: opts.ExtraResidentKVPerPosition,
		exactPrefill: opts.ExactPrefill, actGroup: opts.ActQuantGroup, backendAuto: opts.BackendAuto, noSelfTest: opts.noSelfTest}
}

// Load loads a model from dir, which is a checkpoint directory (config.json and its safetensors shards), a
// .gguf file, or a prequantized .giw bundle, for any supported family. opts chooses the backend, the
// weight quantization and the rest. "cpu" is always available. "cuda", "metal" and "webgpu" become
// available when their module is imported: github.com/townsendmerino/goinfer/cuda (built with -tags cuda),
// .../metal (on darwin) and .../gpu (built with -tags gpu). As Options.Backend says, a name not compiled
// into the binary falls back to cpu. A model that would not fit this machine's memory is refused with an
// error wrapping ErrWontFitResident.
func Load(dir string, opts Options) (*Model, error) {
	ensureCPUSelfTest() // once per process, before any kernel runs concurrently (H2): a CPU kernel tier that disagrees with its reference is stepped down, not trusted
	opts = opts.withAutoBackend()
	// Options.ExactPrefill is recorded on the Model (exactPrefill, set in each constructor below before withResidency, because
	// CUDA reads it while building its resident) and consulted by each backend's fast-prefill switch alongside its env var. It is
	// deliberately not applied with os.Setenv, which is process-global and never undone: every model loaded later in the process
	// would inherit it. Serve's flags travel as this field and Options.Knobs.
	//
	// ctxFromCaller records whether the caller chose the resident context, read before either fit guard below can auto-pin one
	// into opts.ResidentContext: that pin is a one-slot ceiling, not a choice (Model.ResidentContextPinned). It is stamped on the
	// Model after modelFromOptions and before withResidency, where a backend reads it.
	ctxFromCaller := opts.ResidentContext > 0
	be, beErr := NewBackend(opts.Backend)
	// A nil backend means the name was genuinely unknown: abort rather than panic at the first matmul. A non-nil be with a non-nil
	// beErr is the CPU-fallback note (webgpu/cuda/metal not built in): keep the cpu backend and surface the note.
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
		if opts.LoRA != "" {
			closeBackend(be)
			return nil, fmt.Errorf("decoder: LoRA merge needs a safetensors base; %s is a prequantized .giw with no base to merge into", dir)
		}
		data, rerr := mapGIW(dir) // MAP_SHARED on darwin, aikit's MAP_PRIVATE elsewhere (giwmap_*.go)
		if rerr != nil {
			closeBackend(be)
			return nil, fmt.Errorf("decoder: mmap .giw: %w", rerr)
		}
		// Keep the weights out of any fork()ed child (forkinherit_darwin.go): once a GPU backend has wired a page of this private
		// mapping, a fork would otherwise copy all of it eagerly. Non-fatal: failing it only restores the old behaviour.
		if perr := protectGIWMapping(data); perr != nil {
			fmt.Fprintf(os.Stderr, "decoder: minherit(VM_INHERIT_NONE) on the .giw mapping failed (%v) — a fork of this process may copy the whole mapping\n", perr)
		}
		weightsBlob, _, gerr := giw.Read(data)
		if gerr != nil {
			_ = mmap.Unmap(data)
			closeBackend(be)
			return nil, fmt.Errorf("decoder: parse .giw bundle: %w", gerr)
		}
		// The trailing CRC reads every byte of the mapping, which dominates the load of a large streamed .giw. It is a property of the
		// file, so check it once per (size, mtime) and skip it on later loads (giwverify.go). The stat is taken before the load so a
		// file replaced mid-load cannot inherit the marker.
		gfi, statErr := os.Stat(dir)
		crcDone := statErr == nil && giwVerified(dir, gfi)
		w, lerr := loadSerializedWeights(weightsBlob, crcDone)
		if lerr != nil {
			_ = mmap.Unmap(data)
			closeBackend(be)
			return nil, lerr
		}
		if statErr == nil && !crcDone {
			markGIWVerified(dir, gfi)
		}
		// A .giw's weights are file-backed (no fitCheckFor call here, by design; see fitguard.go's srcFileBytes), but its KV cache and
		// prefill scratch are real anonymous allocations. guardGIWFit refuses or auto-pins like the .gguf path's guardFit, against a
		// flat margin over live available memory rather than fitMemFraction's 70%-of-available (sized for a load that commits its
		// weights too). Not everything a .giw load allocates is file-backed: the paged MoE scale cache and the rest of the heap are
		// anonymous, and nothing here prices them.
		if pinnedCtx, gerr := guardGIWFit(&w.Cfg, opts); gerr != nil {
			_ = mmap.Unmap(data)
			closeBackend(be)
			return nil, gerr
		} else if pinnedCtx > 0 {
			opts.ResidentContext = pinnedCtx
		}
		// A .giw bakes its int4 representation in at write time (giwWriter.target), so unlike a GGUF/safetensors load, where
		// wantsCanonicalInt4 decides from this opts.Backend before a byte is quantized, the reader must check the file's promise
		// against what this Load needs (docs/tasks/task-int4-layout-2026-09.md). giwReader.weightMat already refuses a kind-5 tensor
		// this core cannot run (wrong arch/shape); this catches the other mismatch, a kind-5 file under a backend that needs canonical
		// bytes (e.g. Metal), which withResidency's decline would not fail loudly for (it logs and falls back to CPU/staged,
		// non-fatal by design).
		if wantsCanonicalInt4(opts.Backend, be) {
			if n := repackedOnlyInt4Count(w); n > 0 {
				_ = mmap.Unmap(data)
				closeBackend(be)
				return nil, fmt.Errorf("decoder: %s: %d int4 tensor(s) are stored row4-only (kind 5, a cpu-arm64 prequant target) but Backend %q needs canonical bytes — rebuild with `go run ./cmd/prequant -target <matching this backend>` (or delete the stream-weights cache so it rebuilds automatically)", dir, n, opts.Backend)
			}
		}
		if beErr != nil {
			fmt.Fprintln(os.Stderr, beErr)
		}
		m := modelFromOptions(w, be, opts)
		m.resCtxPinned = ctxFromCaller
		m.mmap, m.srcPath = data, dir
		m.bindKnobs(opts.Knobs.values())
		m.withBackendNames(opts.Backend, beErr)
		if opts.StreamWeights {
			// Resolve an "auto" (0) weight-cache request from this platform's own live probe before either pager sees it, so darwin gets
			// a real figure instead of aikit's Linux-only /proc probe plus a fixed darwin fallback.
			opts.WeightCacheBytes = resolveWeightCacheBudget(opts.WeightCacheBytes)
			// MoE → expert demand-paging (#2); dense → per-layer streaming (#4).
			if w.arch.MoE != nil {
				if m.pager = newExpertPager(w, data, opts.WeightCacheBytes, dir, resolveMoEPagerPool(opts.MoEPager, loadKnob(opts, knobMoEPreadCPU))); m.pager != nil {
					fmt.Fprintln(os.Stderr, "decoder: "+pagerSummary(m.pager))
					// Predict the working-set rate and require an explicit acknowledgement below the registered floor, rather than let a long
					// run discover it.
					if predicted, ok := moeWorkingSetPrediction(m); ok {
						fmt.Fprintf(os.Stderr, "decoder: predicted paged-MoE decode rate ~%.2f tok/s "+
							"(a prior, not a measurement — see moeHitRatePrior's own doc comment)\n", predicted)
						if rerr := moeWorkingSetRefusal(filepath.Base(dir), predicted, opts.AcceptSlowMoE); rerr != nil {
							m.Close()
							return nil, rerr
						}
					}
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

	// The fit guard runs here: after the quant is resolved (it moves the weight term more than anything else) and before
	// loadWeights allocates a byte. Refusing after the allocation would be refusing after the swap storm it exists to prevent.
	// The guard may return a smaller context to pin than requested (0 = unrequested, the common case); it is applied to opts here,
	// before opts.ResidentContext is read again below (resCtxReq) and by the banner, so the user gets a working server with a
	// visible, honest limit instead of swap on the first big request.
	pinnedCtx, err := guardFit(fitCheckFor(dir, opts.Quant, quant, opts))
	if err != nil {
		closeBackend(be)
		return nil, err
	}
	if pinnedCtx > 0 {
		opts.ResidentContext = pinnedCtx
	}

	w, err := loadWeights(dir, quant, opts.EmbedInt4, wantsCanonicalInt4(opts.Backend, be), !wantsRow4Fallback(opts.Backend), lora, opts.LoadAbort)
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
	// Write the resolved EOS set (config.json plus any extra ids generation_config.json adds) back into w.Cfg.EOSTokenID, not
	// just onto this Model's eosIDs. w.Cfg is what a .giw bundle serializes (internal/prequant, via
	// SerializeWeightsToForTarget), and a .giw's own Load branch reads eosIDs straight from w.Cfg.EOSIDs() with no directory to
	// re-resolve generation_config.json from. Without this, a checkpoint whose stop ids live only in generation_config.json
	// (Qwen3: <|endoftext|> beside config.json's <|im_end|>) loses the extra id when it round-trips through cmd/prequant, and a
	// completion that emits it runs to max_tokens. Cfg.EOSTokenID has no other reader that needs the unresolved value, so
	// overwriting it is safe.
	//
	// resolveEOSIDs reads generation_config.json through os.DirFS(eosDir), a real directory. For a .gguf load dir is the file
	// path, so os.DirFS(dir) can never open anything inside it (ggufEOS's comment describes a fallback that does not cover this
	// case); a GGUF conversion keeps generation_config.json beside the file, in its parent directory.
	eosDir := dir
	if strings.HasSuffix(dir, ".gguf") {
		eosDir = filepath.Dir(dir)
	}
	resolvedEOS := resolveEOSIDs(eosDir, &w.Cfg)
	if raw, err := json.Marshal(resolvedEOS); err == nil {
		w.Cfg.EOSTokenID = raw
	}
	m := modelFromOptions(w, be, opts)
	m.resCtxPinned = ctxFromCaller
	m.quant, m.eosIDs = opts.Quant, resolvedEOS
	m = m.withBackendNames(opts.Backend, beErr)
	m.bindKnobs(opts.Knobs.values())
	// `resident` is the third phase: weights becoming a device-side runner. Timed here rather than
	// inside withResidency because a backend that DECLINES still costs its probe, and a user
	// wondering where nine seconds went is owed that time too.
	tRes := time.Now()
	m = m.withResidency()
	if w.prof != nil {
		if d := time.Since(tRes); d > time.Millisecond {
			w.prof.record("resident", d)
		}
		m.prof = w.prof
	}
	return m, nil
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
	case "", "cpu", "webgpu", "cuda", "metal", "auto":
		// Accepting the NAME is not a claim that the backend is built in: an unregistered
		// one falls back to CPU with a note (NewBackend). Rejecting it here instead meant
		// `serve --backend cuda|metal` failed at flag-validation even when the module WAS
		// compiled in (-tags cuda / -tags metal), because Validate runs before registration
		// is ever consulted.
	default:
		return fmt.Errorf("decoder: invalid backend %q (cpu | webgpu | cuda | metal | auto)", o.Backend)
	}
	switch o.ActQuantGroup {
	case 0, 32:
	default:
		return fmt.Errorf("decoder: invalid ActQuantGroup %d (0 | 32)", o.ActQuantGroup)
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
	case "q4k":
		return quantQ4K, nil
	default:
		return quantNone, fmt.Errorf("decoder: unknown quant %q (have: int8, int8int8, int4, int4mix, q4k)", q)
	}
}

// closeBackend closes a backend on a load-error path (nil-safe).
func closeBackend(be Backend) {
	if be != nil {
		_ = be.Close()
	}
}

// Config returns a snapshot of the loaded architecture config: a copy, not the live struct. The forward pass reads a derived,
// unexported *Architecture plus RoPE tables built from this config at load, not the config itself, so returning the live
// &m.w.Cfg would let m.Config().NumLayers = N silently desync those caches from the config. Scalar fields are fully
// isolated; the copy is shallow, so its slice/pointer fields still alias the model's and must be treated as read-only. There
// is no supported way to reconfigure a loaded model.
func (m *Model) Config() *Config {
	c := m.w.Cfg
	return &c
}

// Close releases backend resources (GPU resident buffers + the backend) and
// unmaps the .giw mapping if the model was loaded from a prequant bundle. A no-op
// for the CPU backend with no mapping. Safe to call once after the model is done;
// the weights must not be touched afterward (the mapping is gone).
func (m *Model) Close() error {
	if m.pager != nil {
		_ = m.pager.close()
		m.pager = nil
	}
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

// NewCache allocates a KV cache sized for this model. capHint pre-sizes for a known max length (0 = grow on demand).
//
// prefillEnters counts every call, so a test can observe that a request AdmitPrefillMemory refused never reached here: a
// check placed one line too late produces the same error text and the same swap storm (the discipline fitguard.go's
// weightAllocs applies to loadWeights).
func (m *Model) NewCache(capHint int) *KVCache {
	prefillEnters.Add(1)
	a := m.w.arch
	// An MLA family never writes c.keys[l]/c.vals[l] on any layer (the per-layer compressed latent, c.mlaLatent, set up below, is
	// the whole store), so reserving capHint*kvDim for them is dead weight. kvCapHint=0 makes the reservation itself match what
	// happens: an empty slice that ordinary append would still grow correctly if anything ever did write to it.
	kvCapHint := capHint
	if a.mla != nil {
		kvCapHint = 0
	}
	// A linear/mamba/conv mixer layer (or one of Nemotron's mlp/moe block kinds) never writes c.keys[l]/c.vals[l] either: its
	// recurrent state (c.delta/c.mamba/c.conv/c.kda, set up below) is the whole store, so it reserves no K/V capacity.
	// hasNoAttentionKVAt (arch.go) is the one place that knows all these cases, shared with kvDimAt's pricing.
	c := NewKVCache(a.NumLayers, a.NumKVHeads, a.HeadDim, a.SlidingWindow, kvCapHint, a.hasNoAttentionKVAt)
	c.scr = newDecodeScratch(a)
	// int8 KV storage (opt-in, Options.KVQuant == "i8") where kvInt8OK allows it; see ownForwardFamily.KVInt8 for why a family's
	// own loop must opt in. Must precede enableRings so local layers inherit the mode.
	if m.kvI8 && a.kvInt8OK() {
		c.setQuant(kvI8, capHint)
	}
	// Ring-buffer storage on sliding-window (local) layers: keep only the W most recent positions, the only ones a future query
	// can read. Only for families whose forward uses attendQuery/attendBatchedHeads (kvRingsOK); the others keep append-forever.
	// See docs/completed/task-kv-ring-eviction.md.
	if a.kvRingsOK() {
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
		// Hybrid cache: KV for the attention layers + a rolling conv window for each conv layer. manualPos because the conv layers
		// never Append, so position cannot be inferred from the KV length.
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
	if a.kda != nil {
		// Bailing Hybrid (Ling 3.0): a third hybrid-cache shape. The MLA layers' latent cache is handled by the a.mla block above
		// (bailingHybridArchitecture sets both), so only the KDA linear layers' recurrent state needs its own array here.
		c.manualPos = true
		c.kda = make([]*kdaState, a.NumLayers)
		for l := 0; l < a.NumLayers; l++ {
			if a.isLinearLayer(l) {
				c.kda[l] = newKDAState(*a.kda)
			}
		}
	}
	return c
}

// runLayers advances one decode step for token id at position cache.Pos(): it embeds the token, runs the block stack
// (appending this position's K/V to the cache), and returns the residual-stream hidden state after the final layer, before the
// final norm and LM head. Splitting it out lets prefill skip the (vocab-sized) LM head on every token but the last.
//
// The loop is generic over the Architecture descriptor: embedding scale, norm placement (Gemma's 4-norm sandwich vs Llama's
// pre-2), the (1+w) RMS offset, and the activation are all knobs. Gemma 3 is one descriptor:
//
//	h = Embed[id] * EmbedScale
//	for each layer l:
//	  n  = rmsNorm(h, PreAttnNorm)
//	  a  = causalAttention(l, n, ...)
//	  if Sandwich4 { a = rmsNorm(a, PostAttnNorm) }
//	  h += a
//	  n2 = rmsNorm(h, PreMLPNorm)
//	  g  = gatedMLP(n2, ...)
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
		// Resolved per layer, not hoisted: NormPlacementLinear (Olmo Hybrid) can give
		// this layer a different placement than the rest of the model.
		placement := arch.normPlacementAt(l)
		sandwich := placement == NormSandwich4
		// postOnly (Olmo 3, and Olmo Hybrid's full-attention layers): no pre-norm at all —
		// the sublayer reads the RAW residual stream — but the sublayer's OUTPUT still gets
		// normalized before the residual add, the same as Sandwich4's post-norm half.
		// postNorm covers both placements that need it.
		postOnly := placement == NormPostOnly
		postNorm := sandwich || postOnly
		parallel := placement == NormParallel
		if parallel {
			// Cohere/GPT-J parallel block: ONE shared input norm feeds both
			// sublayers, whose outputs sum into a SINGLE residual add —
			// h += attn(n) + mlp(n), where n = norm(h). Both read the same n
			// (attention leaves scr.norm intact), so MLP takes scr.norm, NOT the
			// post-attention residual. No pre-MLP norm, no post-sublayer norms.
			normalizeInto(arch, scr.norm, h, lw.PreAttnNorm, lw.PreAttnNormBias, hidden)
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
			addResidual2(h, scr.sub, scr.sub2)
		} else {
			if !postOnly {
				normalizeInto(arch, scr.norm, h, lw.PreAttnNorm, lw.PreAttnNormBias, hidden)
			} else {
				copy(scr.norm, h)
			}
			var dt0 time.Time
			if decodeTiming {
				dt0 = time.Now()
			}
			if err := causalAttention(l, scr.norm, scr.sub, lw, arch, cache, m.be, ld); err != nil {
				return nil, err
			}
			if decodeTiming {
				atomic.AddInt64(&dtAttn, int64(time.Since(dt0)))
			}
			if cache.subCapture { // scr.ctx is the pre-o-proj context; scr.sub is not yet overwritten
				cache.subCtx[l] = append(cache.subCtx[l][:0], scr.ctx...)
			}
			if postNorm {
				normalize(arch, scr.sub, lw.PostAttnNorm, nil, hidden)
			}
			if cache.subCapture { // scr.sub is now the attention contribution about to hit the residual
				cache.subAttn[l] = append(cache.subAttn[l][:0], scr.sub...)
			}
			addResidual(h, scr.sub)
			if !postOnly {
				normalizeInto(arch, scr.norm, h, lw.PreMLPNorm, lw.PreMLPNormBias, hidden)
			} else {
				copy(scr.norm, h)
			}
			if decodeTiming {
				dt0 = time.Now()
			}
			if err := mlp(scr.norm, scr.sub, lw, arch, m.be, scr, m.pager, ld); err != nil {
				return nil, err
			}
			if decodeTiming {
				atomic.AddInt64(&dtMLP, int64(time.Since(dt0)))
			}
			if cache.subCapture { // scr.sub is the down output BEFORE the post-MLP sandwich norm
				cache.subMLPpre[l] = append(cache.subMLPpre[l][:0], scr.sub...)
			}
			if postNorm {
				normalize(arch, scr.sub, lw.PostMLPNorm, nil, hidden)
			}
			if cache.subCapture { // scr.sub is now the MLP contribution about to hit the residual
				cache.subMLP[l] = append(cache.subMLP[l][:0], scr.sub...)
			}
			addResidual(h, scr.sub)
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

func addResidual(h, sub []float32) {
	n := min(len(h), len(sub))
	if n == 0 {
		return
	}
	h = h[:n]
	sub = sub[:n]
	_ = h[n-1]
	_ = sub[n-1]
	i := 0
	for ; i+3 < n; i += 4 {
		_ = h[i+3]
		_ = sub[i+3]
		h[i] += sub[i]
		h[i+1] += sub[i+1]
		h[i+2] += sub[i+2]
		h[i+3] += sub[i+3]
	}
	for ; i < n; i++ {
		h[i] += sub[i]
	}
}

func addResidual2(h, sub, sub2 []float32) {
	n := min(len(h), min(len(sub), len(sub2)))
	if n == 0 {
		return
	}
	h = h[:n]
	sub = sub[:n]
	sub2 = sub2[:n]
	_ = h[n-1]
	_ = sub[n-1]
	_ = sub2[n-1]
	i := 0
	for ; i+3 < n; i += 4 {
		_ = h[i+3]
		_ = sub[i+3]
		_ = sub2[i+3]
		h[i] += sub[i] + sub2[i]
		h[i+1] += sub[i+1] + sub2[i+1]
		h[i+2] += sub[i+2] + sub2[i+2]
		h[i+3] += sub[i+3] + sub2[i+3]
	}
	for ; i < n; i++ {
		h[i] += sub[i] + sub2[i]
	}
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

// normalizeInto applies the architecture's normalization from src into dst over one row:
// LayerNorm (mean-centered, with bias) for GPT-2/NeoX, else RMSNorm. bias is
// ignored by RMSNorm (and nil for the Sandwich4 post-norms).
func normalizeInto(arch *Architecture, dst, src, weight, bias []float32, dim int) {
	if arch.Norm == NormLayer {
		layerNormInto(dst, src, weight, bias, 1, dim, arch.NormEps)
		return
	}
	rmsNormInto(dst, src, weight, 1, dim, arch.NormEps, arch.RMSAddOne)
}

// normalize applies the architecture's normalization in place over one row.
func normalize(arch *Architecture, x, weight, bias []float32, dim int) {
	normalizeInto(arch, x, x, weight, bias, dim)
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
	if decodeTiming {
		t0 := time.Now()
		lg := m.logitsFromHidden(h, cache)
		atomic.AddInt64(&dtHead, int64(time.Since(t0)))
		return lg, nil
	}
	return m.logitsFromHidden(h, cache), nil
}

// ForwardCapture runs one forward for token id and returns the next-token logits plus the residual stream after each layer in
// `layers` (cloned): the read-only hidden-state seam a draft head reads (block drafters). The forward is byte-identical to
// forward(id); the captures are copies that never feed back. Layer indices are 0-based into [0, NumLayers); out[i]
// corresponds to layers[i].
//
// Wired for the generic decode path plus the own-forward families whose ownForwards entry sets Captures (their loops call
// cache.captureResidual, see capture.go). Every other own-forward family returns an error rather than silently producing nil
// rows: granite and nemotron_h interleave recurrent mixers whose "residual after layer l" needs deciding, and mla and
// llama4_text are not done. A family is wired only when both its loop captures and its entry says so, so a half-wired one
// fails here loudly.
func (m *Model) ForwardCapture(id int, cache *KVCache, layers []int) (logits []float32, hidden [][]float32, err error) {
	a := m.w.arch
	// Derived from the dispatch table's Captures bit rather than re-listed: the families whose own loop calls captureResidual are
	// wired, every other own-forward family is not. A hand-written list once missed LFM2, whose loop never captures and returned
	// nil rows through a seam documented to fail loudly.
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
	// Every own-forward family, derived: this seam needs runLayersFromEmbed's uniform block, which no own-forward loop routes
	// through.
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

// softcapParallel applies Gemma's final-logit softcap sc*tanh(x/sc) in place, fanned out via parallelElementwise. Every
// element is independent and math.Tanh is deterministic, so splitting the loop is byte-identical to the serial form
// (TestSoftcapParallel_bitIdentical). It mirrors metal/model.go's softcapParallel.
func softcapParallel(logits []float32, softcap float32) {
	sc := softcap
	parallelElementwise(len(logits), func(lo, hi int) {
		for j := lo; j < hi; j++ {
			logits[j] = sc * float32(math.Tanh(float64(logits[j]/sc)))
		}
	})
}

// logitsFromHidden applies the final norm + LM head (tied embedding or separate
// lm_head) + optional Gemma logit soft-cap to a layer-stack output h, returning
// the next-token logits ([VocabSize]). One home for the head math, shared by
// forward and forwardFromEmbed.
func (m *Model) logitsFromHidden(h []float32, cache *KVCache) []float32 {
	arch := m.w.arch
	normalize(arch, h, m.w.FinalNorm, m.w.FinalNormBias, arch.HiddenDim)
	return m.logitsFromNormed(h, cache)
}

// logitsFromNormed is logitsFromHidden for a hidden state that has ALREADY been through the final norm (the batched
// Qwen3.5 forward returns its rows normed): the LM head, softcap and logit scale, exactly as logitsFromHidden runs them.
func (m *Model) logitsFromNormed(h []float32, cache *KVCache) []float32 {
	arch := m.w.arch
	logits := cache.scr.logits // reused per stream; matmul fully overwrites it
	if arch.TiedLMHead {
		matmulInto(cache.scr.ws, m.be, &m.w.Embed, h, logits, 1) // tied: embedding doubles as the head
	} else {
		matmulInto(cache.scr.ws, m.be, &m.w.LMHead, h, logits, 1) // separate output projection
	}
	if arch.FinalLogitSoftcap > 0 {
		softcapParallel(logits, float32(arch.FinalLogitSoftcap))
	}
	if arch.LogitScale != 0 && arch.LogitScale != 1 { // Granite logits_scaling: logits /= scale
		inv := float32(1 / arch.LogitScale)
		n := len(logits)
		if n > 0 {
			_ = logits[n-1]
			i := 0
			for ; i+3 < n; i += 4 {
				_ = logits[i+3]
				logits[i] *= inv
				logits[i+1] *= inv
				logits[i+2] *= inv
				logits[i+3] *= inv
			}
			for ; i < n; i++ {
				logits[i] *= inv
			}
		}
	}
	return logits
}

// Generate streams generated token ids over the returned channel until EOS, a stop id, maxTokens, or ctx cancellation. prompt
// is already-tokenized ids. The channel closes when generation ends; check Err after the range loop for a terminal error.
//
// Generate is a raw completion primitive: it continues prompt verbatim and knows nothing about chat turns, roles, or a
// checkpoint's template, so encoding a user message with no formatting makes an instruct-tuned model degenerate into
// repetition. Rendering a template is the caller's job: see github.com/townsendmerino/goinfer/chat (chat.Detect resolves a
// checkpoint's own template from its tokenizer metadata) and examples/embed/main.go for the whole sequence.
//
// Sampling is greedy at Temperature 0, else temperature/top-k/top-p (see Sampler). A SamplingParams.LogitProcessor, if set,
// masks each step's logits before sampling, the seam for constrained/structured decoding.
func (m *Model) Generate(ctx context.Context, prompt []int, maxTokens int, sp SamplingParams) (<-chan int, *Generation) {
	out := make(chan int)
	g := &Generation{}
	// The host KV cache is not allocated here: generateInto calls newCache only when it reaches the CPU path (never resident, or
	// lost the resBusy race), so a resident-and-won call never pays for host KV capacity it never touches.
	newCache := func() *KVCache { return m.NewCache(len(prompt) + maxTokens) }
	go func() {
		defer close(out)
		m.generateInto(ctx, out, g, nil, newCache, prompt, 0, maxTokens, sp, nil)
	}()
	return out, g
}

// prefillDeclineDigitsRE normalizes a decline error's varying numbers (prompt length, floor, byte counts) out of the dedup
// key below, so every below-floor prompt collapses to the same reason instead of re-triggering the warning.
var prefillDeclineDigitsRE = regexp.MustCompile(`\d+`)

// prefillDeclineMu guards prefillDeclineSeen.
var prefillDeclineMu sync.Mutex

// prefillDeclineSeen is the per-reason dedup set warnPrefillDeclined reports through. It is keyed per normalized reason, not
// a process-lifetime sync.Once: a Once let the first decline of any kind (on Metal the routine below-floor case, which fires
// on nearly every short prompt) silence every later, different one an operator would want to see.
var prefillDeclineSeen = map[string]bool{}

// resetPrefillDeclineDedup clears the per-reason dedup state; test-only.
func resetPrefillDeclineDedup() {
	prefillDeclineMu.Lock()
	defer prefillDeclineMu.Unlock()
	prefillDeclineSeen = map[string]bool{}
}

// warnPrefillDeclined reports, once per distinct reason, that a backend's batched prefill refused a prompt at call time and
// the prompt (and every later one with the same reason) is being ingested one token at a time. It exists because
// PrefillPath() answers from static model properties, so the banner and /v1/models can say "batched" while a long prompt hits
// an M-dependent decline. Stderr and not an error: the fallback is correct, just slow, and failing the request over a
// performance decline would be worse than serving it.
func warnPrefillDeclined(n int, err error) {
	key := prefillDeclineDigitsRE.ReplaceAllString(err.Error(), "#")
	prefillDeclineMu.Lock()
	seen := prefillDeclineSeen[key]
	prefillDeclineSeen[key] = true
	prefillDeclineMu.Unlock()
	if seen {
		return
	}
	fmt.Fprintf(os.Stderr, "goinfer: batched prefill declined for a %d-token prompt, falling back to "+
		"the per-token path (slower TTFT; each distinct reason is reported once): %v\n", n, err)
}

// residentPrefillSeed ingests the prompt into the resident KV and returns the last token's logits, the seed for decode. Both
// resident generation paths call it (generateInto and genNgramInto); keep it one function, so an optimisation cannot land on
// one path and miss the other (docs/code-notes/decoder.md#Model.residentPrefillSeed.shared).
//
// from is the first position to compute: prompt[:from] is already committed to the resident KV (prefix reuse,
// resident_reuse.go) and positions carry through unchanged because the cache is positional. from == 0 is the cold path.
//
// Batched prefill (optional Prefiller) ingests the prompt in one pass, much faster TTFT for long prompts. It declines (falls
// back) past the backend's cap, is absent for a backend without a batched forward, and is skipped for tiny prompts. It is on
// by default and must stay bit-identical to sequential decode: every float MAC in both paths is an explicit __fmaf_rn,
// enforced by cuda.TestKernelFMALint and guarded by TestPrefillDivergenceRate. GOINFER_BATCHED_PREFILL=0 force-disables
// (docs/completed/task-batched-prefill-bitidentity.md).
//
// hasAdapter must be true whenever this call runs under a bound resident adapter (cache.lora != nil at generateInto, the only
// caller where that is possible; GenerateVL, the n-gram spec target and GenerateSpeculative's target and draft never bind
// one). The batched Prefiller path is a separate encoded launch per backend that never reads the bound delta, so an adapter
// session that reached it would prefill the prompt's K/V from the base weights and only start applying the adapter at
// decode: plausible, wrong, HTTP 200. This mirrors the CPU rule: the canBatchN caller in forwardn.go declines batched prefill
// whenever cache.lora != nil, because compute-time LoRA is wired only into the sequential forward.
func (m *Model) residentPrefillSeed(ctx context.Context, prompt []int, from int, hasAdapter bool) ([]float32, error) {
	if from < 0 || from >= len(prompt) {
		from = 0 // never skip the seed token, whose logits start decode
	}
	suffix := prompt[from:]
	// Offer the batched path suffixes of 8 tokens or more (a shorter one is cheaper token by token) -- and, when
	// the backend's short continuation is exact (PrefillTailExact), any suffix of a prompt the cold run would have
	// offered whole, so a reused prompt takes the cold run's route and returns its bits.
	offer := len(suffix) >= 8
	if !offer && from > 0 && len(prompt) >= 8 {
		if te, ok := m.resident.(PrefillTailExact); ok && te.PrefillTailExact() {
			offer = true
		}
	}
	if m.knobs.get(knobBatchedPrefill) != "0" && offer && !hasAdapter {
		if pf, ok := m.resident.(Prefiller); ok {
			embs := make([][]float32, len(suffix))
			for i, id := range suffix {
				embs[i] = m.embedResident(id)
			}
			// startPos is why the batched path needs no change for reuse: it already
			// places the run at an offset.
			lg, perr := pf.PrefillLast(ctx, embs, from)
			if perr == nil {
				return lg, nil
			}
			// A CANCELLED PREFILL IS NOT A DECLINE. Falling through to the sequential loop would
			// re-ingest the whole prompt the caller just abandoned — and that loop's own ctx check
			// would abort it one token later anyway, after another forward. Return it.
			if errors.Is(perr, context.Canceled) || errors.Is(perr, context.DeadlineExceeded) {
				return nil, perr
			}
			warnPrefillDeclined(len(suffix), perr)
		}
	}
	// KV-only prefill: prompt[:-1] tokens need only their K/V in the cache — skip the LM head
	// (a big-vocab matmul + readback + softcap) on every prefill token but the last, whose
	// logits seed decode. Byte-identical (same layer chain → same KV → same last-token logits);
	// GOINFER_NO_KVONLY_PREFILL forces the full-logits prefill (A/B / escape hatch).
	kvOnly, hasKV := m.resident.(ResidentPrefillKV)
	useKV := hasKV && m.knobs.get(knobNoKVOnlyPrefill) == ""
	var logits []float32
	var err error
	for i, id := range prompt {
		if i < from {
			continue // already in the cache at position i
		}
		// The resident prefill loop is the GPU-side twin of the batched CPU path's per-layer ctx check:
		// without it an abandoned client leaves the whole prompt streaming through the device.
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

// residentPrefillSeedMRoPE is residentPrefillSeed's m-RoPE-aware sibling, used by GenerateQwenVL's image-reuse fast path
// (docs/multimodal.md), where decode past the (already-resident) image block needs pos+mropeDelta for the rotation, not plain
// pos. Deliberately simple (no batched or KV-only prefill): the reused-suffix case is an agent turn's short trailing
// extension, not a long cold prompt worth the batched machinery.
func (m *Model) residentPrefillSeedMRoPE(ctx context.Context, mrope ResidentMRoPE, prompt []int, from, mropeDelta int) ([]float32, error) {
	if from < 0 || from >= len(prompt) {
		from = 0
	}
	var logits []float32
	var err error
	for i := from; i < len(prompt); i++ {
		if err = ctx.Err(); err != nil {
			return nil, err
		}
		emb := m.embedResident(prompt[i])
		if logits, err = mrope.ForwardMRoPE(emb, i, i+mropeDelta); err != nil {
			return nil, err
		}
	}
	return logits, nil
}

// mc3Prefill prefills prompt[from:] into an MC3 generation's slot and returns the seed logits. A long suffix is taken
// in chunks while other generations are decoding (prefillChunkTokens; mc3_batch.go), each chunk in its own exclusive
// section with one decode step yielded between them; otherwise — a short suffix, nobody decoding, or a resident whose
// batched prefill declines — the rest goes through residentPrefillSeed in one section, exactly as before.
func (m *Model) mc3Prefill(ctx context.Context, mc3 *residentBatcher, slot int, prompt []int, from int) ([]float32, error) {
	if from < 0 || from >= len(prompt) {
		from = 0 // never skip the seed token (residentPrefillSeed's own rule)
	}
	pf, ok := m.resident.(Prefiller)
	C := m.prefillChunk
	// A chunk below the resident's kernel-class floor would put the first chunk on the other class from the
	// whole prompt's; raising the chunk to the floor keeps every pass on the class a whole prefill uses.
	if f := m.prefillKernelFloor(); f > 0 && len(prompt) >= f && C > 0 && C < f {
		C = f
	}
	chunk := ok && C > 0 && m.knobs.get(knobBatchedPrefill) != "0"
	var logits []float32
	var err error
	// Cut a chunk only while a chunk plus prefillTailMin tokens remain, so the pass that ends the prefill has at least
	// prefillTailMin tokens (residentPrefillSeed sends a suffix under 8 down the sequential path, whose numerics are
	// not the batched prefill's) and at most C + prefillTailMin - 1 — never the up-to-2C tail the first candidate
	// left, which was its longest pass (docs/measurements/chunked-prefill-2026-09-27.md).
	for chunk && len(prompt)-from >= C+prefillTailMin && mc3.decoding() > 0 {
		if err = ctx.Err(); err != nil {
			return nil, err
		}
		end := from + C
		embs := make([][]float32, end-from)
		for i, id := range prompt[from:end] {
			embs[i] = m.embedResident(id)
		}
		mc3.prefillExclusive(func() {
			if err = m.residentBind(slot); err == nil {
				_, err = pf.PrefillLast(ctx, embs, from)
			}
		})
		if err != nil {
			if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				return nil, err
			}
			// The batched prefill declined (or failed) on this chunk: everything before `from` is already written,
			// so the remainder, this chunk included, goes through the ordinary path below.
			break
		}
		from = end
		mc3.yieldToDecode()
	}
	err = nil
	mc3.prefillExclusive(func() {
		if err = m.residentBind(slot); err == nil {
			logits, err = m.residentPrefillSeed(ctx, prompt, from, false)
			// The seed may be the resident's shared host buffer (a Forward, or a prefill path that returns it), and the generation reads
			// it after this exclusive section, when another generation's token may be rewriting it: copy it while the resident is still
			// ours (see generateInto's mc3Logits).
			if err == nil {
				logits = append([]float32(nil), logits...)
			}
		}
	})
	return logits, err
}

// tryClaimResident attempts the exclusive claim on the model's single shared resident KV (the resBusy CAS), the one CAS site
// every resident-touching caller shares. It returns false when there is no resident backend or another generation holds the
// claim; the caller must then fall back to the CPU/staged path (distinct sequences still complete correctly, only resident
// speed is lost). On true the caller owns the claim and must release it with atomic.StoreInt32(&m.resBusy, 0), typically
// deferred, once its resident-touching work is done.
func (m *Model) tryClaimResident() bool {
	if m.resident == nil {
		return false
	}
	if m.batcher != nil { // MC3: the exclusive claim waits out no one — it simply fails while a holder has a slot
		return m.batcher.claimExclusive(&m.resBusy)
	}
	return atomic.CompareAndSwapInt32(&m.resBusy, 0, 1)
}

// generateInto is the shared prefill+decode loop behind Model.Generate and Session.Generate. It assumes cache already holds
// prompt[:prefillFrom] (0 for a fresh generation), prefills prompt[prefillFrom:] (always at least one token: the seed, whose
// last position's logits start the decode), then decodes up to maxTokens, streaming each id to out. It does not close out:
// the caller owns the channel so it can run post-generation bookkeeping (a Session reconciling its token list) before the
// consumer observes the close. commit, if non-nil, is called with each id once its forward has committed that position to
// the cache, the seam Session uses to track exactly what the cache holds. Terminal status lands on g.err.
func (m *Model) generateInto(ctx context.Context, out chan<- int, g *Generation, cache *KVCache, newCache func() *KVCache, prompt []int, prefillFrom, maxTokens int, sp SamplingParams, commit func(int)) {
	if len(prompt) == 0 {
		g.err = fmt.Errorf("decoder.Generate: empty prompt")
		return
	}
	sampler := NewSampler(sp)
	sampler.Observe(prompt...) // repetition penalties see the whole prompt, reused prefix included
	// Prefill the (divergent suffix of the) prompt and seed the first token's logits. On the batched archs this runs the layers at
	// M=len in one pass (each weight streamed once), LM head on the last position only; len here is the suffix, not the whole
	// prompt.
	//
	// GPU full-residency decode applies to a plain stateless Generate (prefillFrom == 0). Prefix reuse (prefillFrom > 0) keeps the
	// CPU/staged path: a session's prefix cache is CPU-side and the resident's positional KV is GPU-side, so they cannot both be
	// the source of truth for a reused prefix. A plain session (commit != nil, no adapter) therefore stays on the CPU/staged path.
	//
	// An adapter session (cache.lora != nil) is admitted: going through Session.UseAdapter -> cache.lora -> applyLoRA is the only
	// way a compute-time LoRA is applied, and with prefillFrom==0 it never meets the reused-prefix conflict. It is admitted only if
	// the resident implements ResidentAdapter; otherwise it declines to CPU like any other missing resident capability, never
	// running an adapter session's tokens through the base model's resident weights, which returns correct-looking but wrong
	// output.
	//
	// cache may be nil here (Model.Generate defers allocation to the branch that needs it). Only Session.Generate's cache ever
	// carries lora, so the nil-safe read gives the same value either caller would have produced.
	var lora *loraRuntime
	if cache != nil {
		lora = cache.lora
	}
	var resAdapter ResidentAdapter
	if lora != nil {
		resAdapter, _ = m.resident.(ResidentAdapter)
	}
	useGPU := m.resident != nil && prefillFrom == 0 && (commit == nil || (lora != nil && resAdapter != nil))
	var mc3 *residentBatcher // non-nil: this generation holds an MC3 place and, after prefill, the KV slot mc3Slot
	mc3Slot := -1
	if useGPU {
		// The resident path drives the model's one shared positional KV; two concurrent generations would interleave writes at
		// overlapping positions and corrupt it. Claim it non-blockingly: a loser falls back to the staged CPU path, which uses this
		// call's own cache, so both complete correctly and only resident speed is lost.
		//
		// The claim also makes binding an adapter race-free even though N adapters of one base share this runner: only the resBusy
		// winner's SetAdapter/Forward/SetAdapter(nil) sequence runs at a time, so two adapter sessions (or an adapter session and a
		// base-model session) never observe each other's bound delta.
		if ctxCap := m.ResidentContextCap(); ctxCap > 0 && len(prompt) > ctxCap {
			// Decline before claiming the resident when the prompt exceeds its fixed KV cap: a prefill past it fails mid-write (there is
			// no CPU fallback once it commits), so fall through to the staged CPU path exactly as "not resident" does. Chiefly the adapter
			// case: an adapter's first turn (prefillFrom==0) reaches this branch, and prepare() enforces MaxPositions for it unless the
			// caller derives residentPath from ResidentActive(). This is the decoder-seam half, defense in depth for any caller that
			// does not.
			useGPU = false
		} else if bt := m.batcher; bt != nil && lora == nil && bt.claim(&m.resBusy) {
			// MC3: one of several concurrent generations on the resident, on its own KV slot (mc3_batch.go).
			mc3 = bt
			defer bt.release()
		} else if m.tryClaimResident() {
			if resAdapter != nil {
				if err := resAdapter.SetAdapter(residentAdapterLayers(lora)); err != nil {
					// Bind failed after claiming resBusy — release immediately (not via defer)
					// and fall back to CPU, which applies the adapter correctly on its own
					// (applyLoRA); do NOT proceed resident with no delta bound.
					atomic.StoreInt32(&m.resBusy, 0)
					useGPU = false
				} else {
					defer func() {
						resAdapter.SetAdapter(nil)
						atomic.StoreInt32(&m.resBusy, 0)
					}()
				}
			} else {
				defer atomic.StoreInt32(&m.resBusy, 0)
			}
		} else {
			useGPU = false
		}
	}
	// Allocate the host KV cache here, not before the CAS above: this covers both "never going to be resident" and "lost the race
	// for the shared resident KV", the only two ways to reach the CPU path below. A resident-and-won call never allocates one, and
	// newCache is never called when cache is non-nil (every Session.Generate call).
	if !useGPU && cache == nil {
		cache = newCache()
	}
	gpuPos := 0
	var logits []float32
	var err error
	if useGPU {
		// A request cancelled while waiting for the model lock (tryEnter blocks with no context) reaches here with ctx already done;
		// residentForgetIDs below would discard a warm cache for a prefill residentPrefillSeed is about to refuse anyway. Check
		// before the forget: residentPrefillSeed's own ctx check runs too late to matter.
		if err := ctx.Err(); err != nil {
			g.err = err
			return
		}
		// Prefix reuse: skip the leading tokens already committed to the resident positional
		// KV and prefill only the divergent suffix. Forget FIRST — from here until the
		// generation completes the cache is mid-write, and any early return must leave the
		// next turn cold rather than trusting a half-written cache (resident_reuse.go).
		var reuseFrom int
		if mc3 != nil {
			// MC3: choose a slot no other running generation holds, and prefill into it, with the resident to
			// ourselves (between the other generations' steps).
			mc3.exclusive(func() {
				reuseFrom, mc3Slot = m.residentAcquireSlot(prompt, nil, lora, mc3.busySlots())
				if mc3Slot < 0 {
					err = fmt.Errorf("decoder: no free resident KV slot for a concurrent generation")
					return
				}
				mc3.markSlot(mc3Slot, true)
				m.residentForgetIDs()
			})
			if mc3Slot >= 0 {
				defer mc3.markSlot(mc3Slot, false) // after the commit below: the slot is free again once it holds our ids
			}
			if err == nil {
				logits, err = m.mc3Prefill(ctx, mc3, mc3Slot, prompt, reuseFrom)
			}
			if err != nil {
				g.err = err
				return
			}
		} else {
			reuseFrom = m.residentAcquire(prompt, nil, lora) // MC1: binds the KV slot this prompt will use
			m.residentForgetIDs()
			if logits, err = m.residentPrefillSeed(ctx, prompt, reuseFrom, lora != nil); err != nil {
				g.err = err
				return
			}
		}
		g.PrefillReused = reuseFrom
		gpuPos = len(prompt)
	} else {
		if logits, err = m.prefillLogits(ctx, prompt[prefillFrom:], cache); err != nil {
			g.err = err
			return
		}
	}
	// Greedy fast path: when the resident can pick the argmax on-device AND the sampler's choice is exactly argmax(raw logits)
	// with nothing else reading them, skip the full-logits readback. Emitted tokens are identical; GOINFER_NO_GREEDY_FASTPATH
	// forces the logits path (escape hatch / A-B check). fastNext >= 0 means "the resident already picked the next token"; the
	// first token still comes from the prefill logits through the sampler.
	//
	// `top_k=1` takes this path too (GreedyEquivalent), at any temperature: monotone scaling preserves ordering and a one-token
	// distribution is deterministic. Both predicates are consulted and neither is widened.
	//
	// The speculative paths are deliberately not affected: they gate on `sp.Temperature <= 0` directly (speculative.go,
	// spec_grammar.go, spec_ngram.go), so `top_k=1` with a temperature stays speculative-ineligible. Making it eligible would be
	// correct (argmax verification reproduces greedy) but is a second behaviour change and must not ride along here.
	//
	// RNG: this path skips the per-token rng.Float64() draw SampleWithInfo would make. That is unobservable: under top_k=1 every
	// step is deterministic, so no later draw's value can depend on the skipped ones and no emitted token can differ. The RNG
	// stream position does advance differently, so nothing may depend on it downstream.
	fastNext := -1
	// A GATED processor (SamplingParams.LogitProcessorGate) leaves the fast paths armed: each step
	// then asks the gate, and only a step it opens takes the full-logits path below (needFull).
	gated := sp.LogitProcessor != nil && sp.LogitProcessorGate != nil
	procFree := sp.LogitProcessor == nil || gated
	needFull := gated && sp.LogitProcessorGate(nil)
	greedyRF, hasGreedy := m.resident.(ResidentGreedy)
	fastGreedy := useGPU && hasGreedy && procFree &&
		(sampler.ArgmaxEquivalent() || sampler.GreedyEquivalent()) &&
		m.knobs.get(knobNoGreedyFastpath) == ""
	// The greedy chain (ResidentGreedyChain): the same tokens with the next token's forward queued on the device before the host
	// has seen this one. Only with no processor at all (a gated one would need the full row mid-chain), no adapter, and where the
	// embedding lookup is a plain table row, which the resident's gather reproduces. Under MC3 it runs only on the tokens this
	// generation decodes alone, with the resident held across them (holdSolo); every other token takes the batcher as before, so
	// fastGreedy stays the path for those.
	chainRF, hasChain := m.resident.(ResidentGreedyChain)
	useChain := useGPU && hasChain && sp.LogitProcessor == nil && lora == nil &&
		(sampler.ArgmaxEquivalent() || sampler.GreedyEquivalent()) && m.knobs.get(knobNoGreedyFastpath) == "" &&
		m.embedIsTableRow() && chainRF.GreedyChainAvailable()
	if useChain && mc3 == nil {
		fastGreedy = false
	}
	chainTok := mc3 == nil // this token may run the chain: always unbatched, under MC3 only on a held token
	chainOpen, chainPos := false, 0
	stopChain := func() {
		if chainOpen {
			chainRF.GreedyChainStop()
			chainOpen = false
		}
	}
	defer stopChain()

	// Optimistic forward (spec_optfwd.go): sampled decode's (Temperature>0) sibling of the greedy fast path, overlapping the CPU
	// sampler with a speculative Forward instead of skipping it. Excludes the rare overlap with fastGreedy (Temperature>0 AND
	// top_k==1) so the two never drive the same step. GOINFER_NO_OPTFWD forces the plain sequential path (escape hatch / A-B
	// check). Off under MC3: optFwdStep drives the resident outside the batcher's exclusive section.
	optFwd := useGPU && mc3 == nil && !fastGreedy && !useChain && m.optFwdEligible(sp) && m.knobs.get(knobNoOptFwd) == ""

	// Device top-K fast path (sampler_topk.go): a filtered sampler (top_k / top_p / min_p at temperature > 0) needs only the K
	// best logits, so the resident reduces the row on-device and the host filters K candidates instead of V. Excluded: anything
	// that needs or rewrites the full row (bias, penalties, logprobs, a LogitProcessor), the greedy and optimistic-forward paths,
	// and backends whose logits are transformed on the host after readback. GOINFER_NO_TOPK_FASTPATH forces the full-row path
	// (escape hatch / A-B check), same convention as GOINFER_NO_GREEDY_FASTPATH.
	var topKRF ResidentTopK
	topKWidth, topKVocab := 0, len(logits)
	// Under MC3 a TopKRow's Full() reads the resident's logits, which a later step of another generation may have overwritten, so
	// the draw is resolved inside the resident call (topKPre / topKPreFull below) while this generation still holds the resident,
	// not at the top of the next iteration. Do not switch it off under MC3: that sent every CUDA top-p request down the
	// full-row path (docs/measurements/topp-regression-2026-09-30.md).
	if useGPU && !fastGreedy && !useChain && !optFwd && procFree && m.knobs.get(knobNoTopKFastpath) == "" && sampler.TopKEligible() {
		if rf, ok := m.resident.(ResidentTopK); ok && rf.TopKAvailable() {
			if w, wok := sampler.TopKWidth(len(logits)); wok {
				topKRF, topKWidth = rf, w
			}
		}
	}
	// Device temperature-only sampling (sampler_gumbel.go): the resident draws the next token by Gumbel-max on-device and returns
	// just the id, reusing the greedy fast path's fastNext mechanism. Same exclusions as the top-K path;
	// GOINFER_NO_SAMPLE_FASTPATH forces the host draw (A/B check, escape hatch).
	var sampleRF ResidentSample
	if useGPU && !fastGreedy && !useChain && !optFwd && procFree && m.knobs.get(knobNoSampleFastpath) == "" && sampler.SampleEligible() {
		if rf, ok := m.resident.(ResidentSample); ok && rf.SampleAvailable() {
			sampleRF = rf
		}
	}
	// A greedy generation's batched token asks the step for its argmax id (a Greedy draw) where the resident offers that, instead
	// of its whole logits row; the id is the argmax the sampler would take from that row. Rows run alone keep residentCall's path.
	batchGreedy := false
	if mc3 != nil && !fastGreedy && sampleRF == nil && procFree && (sampler.ArgmaxEquivalent() || sampler.GreedyEquivalent()) {
		if bg, ok := m.resident.(ResidentBatchGreedy); ok && bg.BatchGreedyDraw() {
			batchGreedy = true
		}
	}
	// The sampled chain (ResidentSampleChain): where the device draw serves the token, the same draw with the next token's forward
	// queued on the device first, as the greedy chain does for the argmax. Same exclusions as the greedy chain, and not where
	// ForwardSample itself takes the argmax (a temperature so small 1/T is infinite).
	var sChainRF ResidentSampleChain
	if sampleRF != nil && sp.LogitProcessor == nil && lora == nil && m.embedIsTableRow() &&
		!math.IsInf(1/sp.Temperature, 0) {
		if rf, ok := m.resident.(ResidentSampleChain); ok && rf.SampleChainAvailable() {
			sChainRF = rf
		}
	}
	var topKRow *TopKRow // this step's device top-K row, set instead of logits when topKRF is active
	// Under MC3 the top-K draw is made inside the resident call (see topKRF above): topKPre is the drawn token when the row
	// held the whole retained set, else topKPreFull says topKFullBuf holds this step's full row, copied before the resident
	// was released. Both are consumed, and cleared, where topKRow is.
	var topKPre *SampleInfo
	var topKPreFull bool
	var topKFullBuf []float32
	// mc3Logits holds an MC3 solo token's logits: a resident's Forward returns its one host buffer, reused by every call, and this
	// generation reads the logits after leaving the resident (in its LogitProcessor and its sampler) while another generation's
	// solo token may already be rewriting that buffer.
	var mc3Logits []float32
	var optGate *optFwdGate
	if optFwd {
		optGate = &optFwdGate{}
		g.OptFwd = &OptFwdStats{}
	}

	// Clamp the decode length to the resident KV cap up front. A Forward past the cap is refused mid-generation (the
	// silent-corruption guard), but a resident that exposes its cap lets us stop cleanly at it. gpuPos is the next decode
	// position; the last valid one is ctxCap-1, and the prefill loop guarantees gpuPos <= ctxCap.
	if useGPU {
		if capper, ok := m.resident.(ResidentCapped); ok {
			if ctxCap := capper.ContextCap(); ctxCap > 0 && gpuPos+maxTokens > ctxCap {
				maxTokens = ctxCap - gpuPos // may be 0 (prompt used the whole context)
				g.BudgetClamped = true      // the resident cap (not the request) bounds this turn
			}
		}
	}
	// Publish the effective budget so the caller reports finish_reason "length" when this clamp, not an EOS, ends the turn. Set
	// before any send; read after close. BudgetClamped disambiguates a genuine clamp-to-0 (the prompt fills the cap) from an
	// unclamped turn whose Budget is coincidentally 0.
	g.Budget = maxTokens

	// Decode loop.
	var generated []int
	var tProc, tSample, tEmbed, tFwd time.Duration
	var nFwd int
	// embScratch is reused across iterations: each token's embedding is consumed synchronously by Forward/ForwardArgmax before the
	// next is requested, so one buffer replaces a fresh [hidden]float32 per token.
	var embScratch []float32
	// commitResident records what the resident KV now holds for this generation (residentCommitIDs). Under MC3 it binds
	// this generation's slot first, with the resident to itself, and stops counting it as decoding (so no run waits for
	// its token while it commits).
	decoding := false
	exitDecode := func() {
		if decoding {
			decoding = false
			mc3.exitDecode()
		}
	}
	commitResident := func() {
		stopChain() // every exit commits; nothing queued on the device may outlive the decode loop
		if mc3 == nil {
			m.residentCommitIDs(prompt, generated, nil, lora)
			return
		}
		mc3.releaseHold(stopChain) // before the exclusive section below, which waits for the resident a hold keeps
		exitDecode()
		mc3.exclusive(func() {
			if m.residentBind(mc3Slot) == nil {
				m.residentCommitIDs(prompt, generated, nil, lora)
			}
		})
	}
	if mc3 != nil {
		mc3.enterDecode()
		decoding = true
		defer exitDecode()
		// Runs before exitDecode (defers unwind last-first): any exit, an error's included, ends a kept hold.
		defer mc3.releaseHold(stopChain)
	}
	var holdGrace *time.Timer // the send grace for a kept hold (one timer per generation, reset per token)
	if mc3 != nil {
		holdGrace = time.NewTimer(mc3HoldSendGrace)
		holdGrace.Stop()
		defer holdGrace.Stop()
	}
	// MC3c step 2 (cpu_batch.go): a CPU generation on an eligible cache submits its decode tokens to the model's CPU
	// batcher, which joins them with other generations' into one batched step; alone, a token runs m.forward as always.
	var cb *cpuBatcher
	if !useGPU && m.cpuBatch != nil && cpuBatchCacheEligible(cache) == nil {
		cb = m.cpuBatch
		cb.enterDecode()
		defer cb.exitDecode()
	}
	for range maxTokens {
		select {
		case <-ctx.Done():
			g.err = ctx.Err()
			// A cancel that arrives during the previous iteration's Forward, the dominant per-iteration cost, is not observed until here,
			// because nothing between appending next and Forward returning is select-guarded. By now that iteration's next is in
			// generated and its Forward has run (the resident cache write is synchronous inside it), so the cache is exactly as
			// consistent as at the natural-completion commit below. The send-select exit below is the rarer of the two points where a
			// cancel is observed; both must commit.
			if useGPU {
				commitResident()
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
			// Device top-K row from the previous forward: draw from it if the K candidates prove they
			// hold the whole retained set; otherwise read the full row for this token (no randomness
			// has been consumed, so the fallback is invisible) and take the ordinary path below.
			drew := false
			if topKRow != nil {
				if topKPre != nil {
					info, next, drew = *topKPre, topKPre.ID, true
					g.TopKServed++
				} else if topKPreFull {
					logits = topKFullBuf
					g.TopKFallbacks++
				} else if inf, ok := sampler.SampleFromTopK(*topKRow, topKVocab); ok {
					info, next, drew = inf, inf.ID, true
					g.TopKServed++
				} else {
					full, ferr := topKRow.Full()
					if ferr != nil {
						g.err = ferr
						return
					}
					logits = full
					g.TopKFallbacks++
				}
				topKRow, topKPre, topKPreFull = nil, nil, false
				if decodeTiming {
					tSample += time.Since(t0)
				}
			}
			if !drew {
				// Constrained decoding: let the processor mask this step's logits
				// (based on what's been generated) before sampling and the stop check.
				if sp.LogitProcessor != nil && (!gated || needFull) {
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
		// A kept hold (a chained decode alone under MC3) must not keep the resident while the consumer is not reading: a
		// newcomer would wait on this send. Give the consumer a short grace, then let go before blocking.
		if mc3 != nil && mc3.isHolding() {
			select {
			case out <- next:
				goto sent
			default:
			}
			holdGrace.Reset(mc3HoldSendGrace)
			select {
			case out <- next:
				holdGrace.Stop()
				goto sent
			case <-holdGrace.C:
				mc3.releaseHold(stopChain)
			case <-ctx.Done():
				holdGrace.Stop()
			}
		}
		// Select on ctx.Done like every speculative path.
		select {
		case <-ctx.Done():
			g.err = ctx.Err()
			// At this exit next has been sampled but never forwarded, so the resident cache is consistent with exactly prompt+generated:
			// record it instead of leaving resIDs nil, which would force the next turn to cold-prefill an interrupt that left nothing
			// inconsistent behind. Agent harnesses cancel constantly (interrupts, timeouts, disconnects).
			if useGPU {
				commitResident()
			}
			return
		case out <- next:
		}
	sent:
		generated = append(generated, next)
		if gated {
			needFull = sp.LogitProcessorGate(generated)
		}
		if optResolved {
			// optFwdStep already produced (or redid) this position's Forward and its
			// result logits for gpuPos+1 — nothing left to do but advance the position.
			logits = optNextLogits
			gpuPos++
		} else if useGPU {
			var emb []float32
			if decodeTiming {
				t0 = time.Now()
				emb = m.embedResidentInto(next, embScratch)
				tEmbed += time.Since(t0)
				t0 = time.Now()
			} else {
				emb = m.embedResidentInto(next, embScratch)
			}
			embScratch = emb
			fastNext = -1 // set again below only by a path that picks the next token on-device
			pos := gpuPos
			// A device-drawn token takes its draw HERE, once, whichever path then serves it — the sampler's RNG
			// stream advances exactly as it always has.
			var draw *ResidentBatchDraw
			if !needFull && !fastGreedy && sampleRF != nil {
				seed, d := sampler.NextDraw()
				draw = &ResidentBatchDraw{Temperature: sp.Temperature, Seed: seed, Draw: d}
			} else if batchGreedy && !needFull {
				draw = &ResidentBatchDraw{Greedy: true}
			}
			// residentCall is this token's production resident call.
			viaForward := false // residentCall returned Forward's logits, the resident's shared host buffer
			residentCall := func() error {
				var ferr error
				if needFull {
					// The gated processor needs this position's full logits (it is about to mask them).
					logits, ferr = m.resident.Forward(emb, pos)
					viaForward = true
				} else if useChain && chainTok {
					// The chain fed itself this token: it is the argmax of the chain's previous forward. The first token
					// (from the prompt's seed logits) opens it.
					if !chainOpen {
						if ferr = chainRF.GreedyChainStart(next, pos); ferr == nil {
							chainOpen, chainPos = true, pos
						}
					}
					if ferr == nil && pos != chainPos {
						ferr = fmt.Errorf("decoder: greedy chain is at position %d, the decode loop at %d", chainPos, pos)
					}
					if ferr == nil {
						fastNext, ferr = chainRF.GreedyChainNext()
						chainPos++
					}
				} else if sChainRF != nil && chainTok {
					// The sampled chain: as the greedy chain, with this token's draw (taken above, as ForwardSample's is).
					if !chainOpen {
						if ferr = sChainRF.SampleChainStart(next, pos, draw.Temperature, draw.Seed, draw.Draw); ferr == nil {
							chainOpen, chainPos = true, pos
						}
					}
					if ferr == nil && pos != chainPos {
						ferr = fmt.Errorf("decoder: sampled chain is at position %d, the decode loop at %d", chainPos, pos)
					}
					if ferr == nil {
						fastNext, ferr = sChainRF.SampleChainNext(draw.Seed, draw.Draw)
						chainPos++
					}
					if ferr == nil {
						g.DeviceSampled++
					}
				} else if fastGreedy {
					// Greedy fast path: the resident picks the argmax on-device and returns
					// just the id, skipping the full-logits readback.
					fastNext, ferr = greedyRF.ForwardArgmax(emb, pos)
				} else if sampleRF != nil {
					fastNext, ferr = sampleRF.ForwardSample(emb, pos, draw.Temperature, draw.Seed, draw.Draw)
					if ferr == nil {
						g.DeviceSampled++
					}
				} else if topKRF != nil {
					var row TopKRow
					row, ferr = topKRF.ForwardTopK(emb, pos, topKWidth, sp.Temperature, sampler.topPActive())
					if ferr == nil && mc3 != nil {
						// Resolve the draw now, while this generation holds the resident (see topKRF). The sampler is this
						// generation's own and nothing else draws from it before the next iteration would have, so the RNG
						// is consumed in the same order as the unbatched path.
						if inf, ok := sampler.SampleFromTopK(row, topKVocab); ok {
							topKPre = &inf
						} else if full, e := row.Full(); e != nil {
							ferr = e
						} else {
							topKFullBuf = append(topKFullBuf[:0], full...) // the resident's host buffer is shared
							topKPreFull = true
						}
					}
					if ferr == nil {
						topKRow = &row
					}
				} else {
					logits, ferr = m.resident.Forward(emb, pos)
					viaForward = true
				}
				return ferr
			}
			if mc3 != nil {
				// MC3: submit the token; it runs in a shared step with the other generations' tokens, or — alone — as
				// residentCall on our own slot. A step's row is bit-identical to this token's own call: Forward's
				// logits, or ForwardSample's id for a device-drawn token (the same draw). A greedy token served by a
				// step continues on the full-logits path, which production's fast path equals by its own contract
				// (ArgmaxEquivalent/GreedyEquivalent above).
				solo := func() error {
					if berr := m.residentBind(mc3Slot); berr != nil {
						return berr
					}
					if ferr := residentCall(); ferr != nil {
						return ferr
					}
					if viaForward { // copy while this generation still holds the resident (mc3Logits)
						mc3Logits = append(mc3Logits[:0], logits...)
						logits = mc3Logits
					}
					return nil
				}
				// Alone on the batcher, a chainable token runs the chain with the resident held across tokens; the
				// first token anyone else wants the resident, the chain stops and this token takes the batcher.
				held := false
				if (useChain || sChainRF != nil) && !needFull {
					chainTok = true
					held, err = mc3.holdSolo(solo, stopChain)
					chainTok = false
				}
				q := &batchReq{seq: ResidentBatchSeq{Slot: mc3Slot, Pos: pos, Emb: emb, Draw: draw}, solo: solo}
				if !held {
					err = mc3.forward(q)
				}
				if !held && err == nil && q.out != nil {
					if draw != nil {
						fastNext = q.out.ID
						if !draw.Greedy {
							g.DeviceSampled++
						}
					} else {
						logits, fastNext = q.out.Logits, -1
					}
				}
			} else {
				err = residentCall()
			}
			gpuPos++
		} else if cb != nil {
			// A batched step's row is bit-identical to this token's own m.forward (decodeMultiStep's contract).
			q := &batchReq{id: next, cache: cache, solo: func() error {
				var ferr error
				logits, ferr = m.forward(next, cache)
				return ferr
			}}
			err = cb.forward(q)
			if err == nil && q.out != nil {
				logits = q.out.Logits
			}
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
		commitResident()
	}
	if decodeTiming && nFwd > 0 {
		ms := func(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 / float64(nFwd) }
		fmt.Printf("DECODE TIMING (%d tok, gpu=%v): forward %.1f ms | sample %.2f ms | logitProc %.2f ms | embed %.2f ms /token\n",
			nFwd, useGPU, ms(tFwd), ms(tSample), ms(tProc), ms(tEmbed))
		d := func(p *int64) float64 { return ms(time.Duration(atomic.SwapInt64(p, 0))) }
		a, ml, hd := d(&dtAttn), d(&dtMLP), d(&dtHead)
		gu, act, ad := d(&dtGU), d(&dtAct), d(&dtActDown)
		qkv, o, aa := d(&dtQKV), d(&dtO), d(&dtAttnAll)
		fmt.Printf("DECODE SPLIT: attention %.2f ms | MLP %.2f ms | LM head %.2f ms | residual %.2f ms /token\n",
			a, ml, hd, ms(tFwd)-a-ml-hd)
		fmt.Printf("DECODE SPLIT MLP: gate+up matmuls %.2f ms | activation %.2f ms | down matmul %.2f ms /token\n", gu, act, ad-act)
		fmt.Printf("DECODE SPLIT ATTN: q/k/v matmuls %.2f ms | rope+KV+scores/softmax/AV core %.2f ms | o matmul %.2f ms /token\n", qkv, aa-qkv-o, o)
		lastDecodeSplit.fwd, lastDecodeSplit.attn, lastDecodeSplit.mlp, lastDecodeSplit.head = ms(tFwd), a, ml, hd
		lastDecodeSplit.gu, lastDecodeSplit.act, lastDecodeSplit.down = gu, act, ad-act
		lastDecodeSplit.qkv, lastDecodeSplit.core, lastDecodeSplit.o = qkv, aa-qkv-o, o
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
	// PrefillReused is how many leading prompt tokens this generation skipped because they were already committed to the
	// resident positional KV (resident_reuse.go); 0 on a cold prefill and on every non-resident path. Diagnostic: it makes an
	// agent loop's per-turn prefill cost visible without timing it. GenerateVL/GenerateQwenVL set it only on the full-image-reuse
	// fast path (docs/multimodal.md): 0 there means the turn's image (or everything before it) was not fully reused.
	PrefillReused int
	// ImgPrefillResident reports whether GenerateVL's resident image-prefill fast path (ResidentImagePrefill, the bidirectional
	// image-block kernel) ran this turn's prefill on the GPU; false means it fell through to the CPU-prefill+UploadKV bridge,
	// whether because no resident implements it, the prompt was too long for one chunk, or any other decline. Diagnostic: a
	// real-checkpoint gate needs it to confirm the fast path fired rather than passing vacuously through the (correct) fallback.
	ImgPrefillResident bool
	// ImgPrefillDecline is the reason the resident image prefill declined this turn, when a backend that implements it
	// was asked and said no (a multi-image turn on a backend whose resident prefill takes one block). Empty when it
	// ran, or when no resident implements it.
	ImgPrefillDecline string
	// DecodeResident reports whether a multimodal turn's decode ran on the resident after its CPU prefill was uploaded
	// (GenerateGemma4VL's bridge). Diagnostic, as ImgPrefillResident: a gate must show the image turn decoded resident, not on
	// the CPU.
	DecodeResident bool
	Spec           *SpecStats
	OptFwd         *OptFwdStats // non-nil when optFwdEligible held for this run; see spec_optfwd.go
	// TopKServed / TopKFallbacks count decode steps sampled from the device top-K row and the steps where
	// K could not prove it held the retained set (the full row was read instead). Both 0 unless the
	// device top-K fast path was active; served/(served+fallbacks) is the fast path's hit rate.
	TopKServed, TopKFallbacks int
	// DeviceSampled counts decode steps whose token the resident drew on-device by Gumbel-max; 0 unless that fast path was active.
	DeviceSampled int
	// Logprobs holds one entry per emitted token (in order) when
	// SamplingParams.Logprobs was set — the chosen token's log-probability and
	// any requested top alternatives. Complete once the stream has closed.
	Logprobs []SampleInfo
	// Budget is the effective max-token budget after the resident context-cap clamp. It equals the requested maxTokens unless
	// prompt+maxTokens would exceed the resident KV cap, in which case it is the remaining room (may be 0). A caller that reports
	// finish_reason must compare the emitted count against this, not the requested value, or a context-clamped generation is
	// mis-reported as a clean "stop" and the client never continues. Set before the first token is sent; read it after the
	// channel closes (like Err). 0 on generation paths that do not clamp (speculative/VL).
	Budget int
	// BudgetClamped is true iff the resident context cap (not the request) bounded this turn, so Budget is the authoritative
	// limit, including a clamp to 0 when the prompt fills the whole context. A caller must judge finish_reason against Budget only
	// when this is set; otherwise it falls back to the requested max_tokens. Without it, a genuine clamp-to-0 is
	// indistinguishable from an unclamped Budget-0 turn, and a context-full empty response mis-reports "stop".
	BudgetClamped bool
}

// Err returns the error that ended the stream, or nil if it ended cleanly
// (EOS / stop / maxTokens). Read it after the channel closes.
func (g *Generation) Err() error { return g.err }
