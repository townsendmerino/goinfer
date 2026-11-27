package decoder

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/townsendmerino/goinfer/internal/runtime/cpuengine"
	"github.com/townsendmerino/goinfer/internal/runtime/wbuild"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/townsendmerino/goinfer/internal/cost"
	"github.com/townsendmerino/goinfer/internal/runtime/kvmgr"
	"github.com/townsendmerino/goinfer/internal/runtime/logs"

	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/decoder/config"
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

// Model is a loaded checkpoint plus the compute backend. It is safe for concurrent use: weights are immutable after Load, and
// any number of goroutines may call Generate (and each its own Session) at once; every call completes correctly. Where a call
// runs is the only thing concurrency changes. On the CPU each generation owns its KV cache. A GPU-resident model's resident KV
// is claimed per generation: by default one generation holds it and the others run on the CPU path; after
// EnableResidentConcurrency(n), up to n run on the device at once, each on its own KV slot, and any further one runs on the
// CPU. A prompt longer than ResidentContextCap also runs on the CPU. Falling back costs speed, never correctness; whether
// CPU generations are worth running side by side is CPUConcurrentSafe's answer. A single KVCache or Session is not safe to
// share between goroutines.
type Model struct {
	planSlackOf *planSlackMemo // PlanSlack per backend, asked once; nil on a Model built by hand (asked every time)
	fitOf       *fitMemo       // what Plan derives from the config and architecture alone; nil on a Model built by hand
	w           *darch.Weights
	be          Backend
	eosIDs      []int           // end-of-sequence ids from config (generation stops on these)
	resident    ResidentForward // GPU full-residency decode path (webgpu + eligible arch); nil ⇒ staged/CPU
	lifeP       unsafe.Pointer  // *modelLife (lifecycle.go): the in-flight generations Close ends; made on first use
	resBusy     int32           // atomic: claims the single shared resident KV for one in-flight generation. Raw int32 (not atomic.Bool) so Model stays copyable for the value-copy test seam.
	capWarned   int32           // atomic: the first prompt past the resident's context has been reported (warnResidentCap)
	// batcher: the batched resident decode's coordinator (mc3_batch.go) — several generations on the resident at once, each on its own KV
	// slot, their decode tokens joined into shared steps. nil unless EnableResidentConcurrency enabled it.
	batcher *residentBatcher
	// specAdaptive: the "speculate when alone, batch under load" candidate (SetSpecAdaptive). off by default — genNgramInto's exclusive claim is unchanged unless
	// this is on AND batcher is non-nil.
	specAdaptive bool
	// cpuBatch: the batched CPU decode's coordinator (cpu_batch.go) — several CPU generations' decode tokens joined into one
	// batched forward. nil unless EnableCPUBatch enabled it; Options.CPUBatchDecode chooses when it may.
	cpuBatch *cpuBatcher
	// resCkptPrompt is the prompt whose end the resident's recurrent checkpoint holds, saved by the running generation and
	// recorded on its sequence by the commit (residentCheckpointSave, residentCommitIDs); nil otherwise.
	resCkptPrompt []int
	// opts is the Options the model was loaded with: the accessors for each load option read it, and ResolveKV reads it for the
	// precision each cache delivers (KVPrecisionReport).
	opts Options
	// resDrafterSynced identifies which *BlockSpec's own drafter context is in sync with the bound slot's sequence; nil means no
	// drafter context is trustworthy for reuse. The sequence alone is not enough: a plain Generate or n-gram-speculative turn can
	// commit one without touching a block drafter's context, so the token prefix can match while the drafter's state does not
	// reflect it. Cleared by residentForgetIDs (every resident write invalidates it) and set only by BlockSpec.generate's
	// fully-completed exit, the one path that keeps the sequence and the drafter's context advancing together.
	resDrafterSynced *BlockSpec
	// kv is the KV manager over the resident's slots (internal/kvmgr, resident_reuse.go): one sequence per slot, the
	// slot a prompt takes, what it may reuse, and the index of published blocks; the one record of what each slot holds,
	// the bound one included. nil until first used. Guarded by the same resBusy claim that serialises writes to the
	// resident KV. kvNS is the sharing namespace of the request that took each slot (KVScope), which the manager does
	// not keep: the commit that ends the request keys the slot's sequence under it.
	kv           *kvmgr.Slots
	kvNS         []string
	kvF16        bool          // the resident KV cache is f16: Options.KVPrecision == "f16", or on CUDA "auto" (the default) for a cleared family (kv16_cuda.go)
	kvF16Decline string        // CUDA only: why an explicit -kv f16 was held at f32 ("" when it was not asked for or not held back)
	kvPrecI8     bool          // residency KV cache int8 request (Options.KVPrecision == "i8") — GPU
	kvI8         bool          // CPU KV cache int8 storage request (Options.KVQuant == "i8") — CPU staged path
	kvH          bool          // CPU KV cache f16 storage request (Options.KVQuant == "f16"), where kvF16OK allows it
	prog         *layerProgram // the forward as operations (buildProgram), built once by modelFromOptions
	knobs        *knobSet      // per-model operator knobs, snapshotted once at Load (knobs.go)
	resCtxReq    int           // requested GPU-resident KV capacity in positions (Options.ResidentContext); 0 ⇒ backend default
	resCtxPinned bool          // the caller chose resCtxReq (not the fit guard's auto-pin) — ResidentContextPinned
	kvSnaps      *kvSnapStore  // Options.KVHostBytes: the held prefix snapshots, nil when off

	loadAbortErr error // set by withResidency when the build returned ErrLoadAborted or ErrPinnedContextWontFit; Load turns it into the load's error

	extraBytes int64        // Options.ExtraResidentBytes — see that field's own doc comment
	mmap       []byte       // .giw mmap region the int8/int4 weights alias; munmap'd by Close (nil off the .giw mmap path)
	srcPath    string       // the .giw path this model mmap-loaded from ("" off the .giw path) — for pread-staging over the same file
	loadPath   string       // the file or directory Load opened, after the sidecar route ("" for a model not loaded from a path)
	pager      *expertPager // MoE expert demand-paging over the mapping (Options.StreamWeights); nil = all-resident
	layerPager *layerPager  // dense per-layer streaming over the mapping (Options.StreamWeights); nil = all-resident
	quant      string       // the requested Options.Quant for a direct load ("" for a prequant .giw → Quant() derives from kinds)
	// resDecline records WHY resident is nil on a non-CPU backend — the reason withResidency
	// would otherwise discard. Empty when residency was built, or when it was never attempted
	// (CPU backend). DecodePath / -require-backend read it; see withResidency.
	resDecline string

	// prof is this load's phase timing, when the loader instrumented it. nil otherwise.
	prof *darch.LoadProfile

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

// SourcePath is the file or checkpoint directory Load opened (the sidecar .giw when the route took it), or "" for a model not loaded from a
// path. The CUDA expert cache writes its packed-experts file beside it under MoEExpertsUnpinned.
func (m *Model) SourcePath() string { return m.loadPath }

// MmapByteOffset returns the byte offset of slice b within this model's .giw mmap region, or
// ok=false if b does not alias that region (a heap-backed weight, or no .giw mapping). MapReadOnly
// maps the whole file from offset 0, so this offset is exactly where to pread b's bytes from the.
// giw. Pure pointer arithmetic — does NOT touch b's pages (no fault).
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

// EmbedInt4 reports whether this model was loaded with the int4 output head (Options.EmbedInt4 after EmbedInt4For). A
// .giw's head is the one its transcode baked, whatever this says.
func (m *Model) EmbedInt4() bool { return m.opts.EmbedInt4 }

// ResidentKVSlotsIsDefault reports Options.ResidentKVSlotsDefault: the slot request is the caller's default, which a backend
// may lower to its own (Metal: 2).
func (m *Model) ResidentKVSlotsIsDefault() bool { return m.opts.ResidentKVSlotsDefault }

// WindowedKVSlack is the headroom, in positions, a windowed-KV layer holds beyond its window: a layer keeps window+WindowedKVSlack
// positions and compacts (copies its live tail to the start) once per slack positions written. It must cover the widest batch of rows
// written in one pass: a prefill chunk (512 by default), a verify batch, a batched-decode row. The CUDA allocation and decoder's pricing both use
// it, so priced bytes equal allocated bytes.
const WindowedKVSlack = 512

// ResidentWindowedKVCap is how many positions a windowed-KV layer of the given window holds at a resident context of ctx: the whole
// context when that is no larger than the window plus its slack, else window+WindowedKVSlack.
func ResidentWindowedKVCap(window, ctx int) int { return min(ctx, window+WindowedKVSlack) }

// ResidentWindowedKV reports Options.ResidentWindowedKV: the caller asked for windowed K/V on a backend that supports it.
func (m *Model) ResidentWindowedKV() bool { return m.opts.ResidentWindowedKV }

// ResidentMoEStep reports Options.ResidentMoEStep: the caller let a generic MoE model run on a backend's multi-sequence step.
func (m *Model) ResidentMoEStep() bool { return m.opts.ResidentMoEStep }

// ResidentRecurrentStep reports Options.ResidentRecurrentStep: the caller let a Qwen3.5-family model (Gated-DeltaNet and gated
// attention) or a Nemotron-H model (Mamba-2) run on a backend's multi-sequence step.
func (m *Model) ResidentRecurrentStep() bool { return m.opts.ResidentRecurrentStep }

// ResidentRecurrentStepOn reports whether a backend granting slots KV slots runs the recurrent step: chosen, always; a default, only with
// two or more.
func (m *Model) ResidentRecurrentStepOn(slots int) bool {
	return m.opts.ResidentRecurrentStep && (!m.opts.ResidentRecurrentStepDefault || slots >= 2)
}

// ResidentGemma4StepOn is ResidentRecurrentStepOn for the Gemma 4 step.
func (m *Model) ResidentGemma4StepOn(slots int) bool {
	return m.opts.ResidentGemma4Step && (!m.opts.ResidentGemma4StepDefault || slots >= 2)
}

// ResidentGemma4Step reports Options.ResidentGemma4Step: the caller let a Gemma 4 dense model run on a backend's
// multi-sequence step.
func (m *Model) ResidentGemma4Step() bool { return m.opts.ResidentGemma4Step }

// WindowedKVLayer reports whether layer l's resident K/V is windowed under Options.ResidentWindowedKV: the option is on and the layer is
// a sliding-window attention layer with a real K/V cache (not an MLA latent, not a recurrent mixer, whose kvDimAt is not a K/V width
// bounded by a window). Every windowed layer shares SlidingWindow's one window.
func (m *Model) WindowedKVLayer(l int) bool {
	if !m.opts.ResidentWindowedKV || m.BackendView().SlidingWindow() <= 0 || !m.BackendView().Layer(l).IsLocal() {
		return false
	}
	a := m.w.Arch
	return a != nil && a.MLA == nil && a.KVDimAt(l) > 0
}

// ResidentKVSlotsRequest returns the requested number of resident KV slots (Options.ResidentKVSlots), at least 1, and 1 for a
// family with recurrent state. A residency builder that supports slots (KVSlotterAvailable) allocates up to this many,
// clamped by its fit guard.
func (m *Model) ResidentKVSlotsRequest() int { return m.ResidentKVSlotsRequestFor("") }

// ResidentKVSlotsRequestFor is ResidentKVSlotsRequest for a build on backend. A family with recurrent state gets more than one
// slot only when backend's table keeps every one of its recurrent operations once per slot (recurrentSlotsOn) and the caller
// asked for the slots, not just its default (ResidentKVSlotsIsDefault); otherwise its state would be one copy shared by every
// slot, or a default load would take memory it did not ask for.
func (m *Model) ResidentKVSlotsRequestFor(backend string) int {
	if m.hasRecurrentState() && (m.opts.ResidentKVSlotsDefault || !m.recurrentSlotsOn(backend)) {
		return 1
	}
	return max(1, m.opts.ResidentKVSlots)
}

// recurrentSlotsOn reports whether backend's operation table keeps every recurrent operation of m's program once per KV slot
// (program.Support.RecurrentSlots). A backend without a table keeps none.
func (m *Model) recurrentSlotsOn(backend string) bool {
	t := residentBackendTables[backend]
	return t != nil && t.RecurrentSlotsCover(m.Program())
}

// ResidentKVSlots reports how many resident KV slots this model's generations choose among — what the resident
// allocated (KVSlotterAvailable), 1 for a resident without slots, 0 off the resident path.
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
func (m *Model) ExtraResidentKVPerPosition() int64 { return m.opts.ExtraResidentKVPerPosition }

// ExpertCacheStats reports the resident's MoE expert cache (ExpertCacheAvailable); ok is false off the resident path or when
// the resident keeps no cache.
func (m *Model) ExpertCacheStats() (ExpertCacheStats, bool) {
	return m.resRuntime().ExpertCacheStats()
}

// ResizeExpertCache resizes the resident's MoE expert cache (ExpertCacheAvailable) to leave reserve bytes of device memory
// free, and returns its slots per layer. It errors when there is no resident or the resident keeps no cache.
func (m *Model) ResizeExpertCache(reserve int64) (int, error) {
	rt := m.resRuntime()
	if !rt.ExpertCacheAvailable() {
		return 0, fmt.Errorf("decoder: this model's resident keeps no expert cache")
	}
	return rt.ResizeExpertCache(reserve)
}

// SidecarFor is Options.SidecarFor: the backend whose sidecar .giw this load stands in for, "" for none.
func (m *Model) SidecarFor() string { return m.opts.SidecarFor }

// FitDisabled is --fit=off (Options.DisableFit), also set by GOINFER_NO_FIT_DEFAULT (kept working as the original, narrower
// escape hatch): every "fit by default" behavior reverts to its previous default. Checked by backend packages that implement
// a fit-by-default policy (cuda.resolveCtxCapFit); decoder itself has none to gate.
func (m *Model) FitDisabled() bool {
	return m.opts.DisableFit || m.knobs.get(knobNoFitDefault) != ""
}

// MoECacheExperts reports whether routed MoE experts should stream host→VRAM per token instead of
// being held resident (Options.MoECacheExperts, `--moe-cache-experts`). This is what lets a model
// whose experts exceed VRAM run with every expert still executing ON the GPU. Off by default:
// running a model larger than your card is a deliberate act, it costs a per-token PCIe transfer,
// and with it off the runtime declines honestly instead of silently going slow.
//
// Falls back to GOINFER_MOE_CACHE_EXPERTS so existing scripts keep working.
func (m *Model) MoECacheExperts() bool {
	return m.opts.MoECacheExperts || m.knobs.get(knobMoECacheExperts) != ""
}

// MoEExpertsUnpinned reports whether a CUDA expert cache (MoECacheExperts) uploads its slots from pageable pages of a file holding the packed
// experts, instead of a pinned host copy (Options.MoEExpertsUnpinned, `--moe-experts-unpinned`): about 12 GiB less host memory on Gemma 4 26B, at
// 0.82x decode at short context and 0.75x at long (docs/measurements/mem10-2026-11/ab-registration.md). Off by default.
func (m *Model) MoEExpertsUnpinned() bool { return m.opts.MoEExpertsUnpinned }

// FlashDecodeSoftcap reports Options.FlashDecodeSoftcap: a model with an attention softcap may take CUDA's flash-decode lane, in its capped
// variant. Off by default.
func (m *Model) FlashDecodeSoftcap() bool { return m.opts.FlashDecodeSoftcap }

// FlashDecodeHD96 reports Options.FlashDecodeHD96: a head-dim-96 model may take the flash-decode lane on its hd-96 kernels (CUDA, Metal).
// FlashDecodeHD96Default says whether that is the caller's default.
func (m *Model) FlashDecodeHD96() bool { return m.opts.FlashDecodeHD96 }

// FlashDecodeHD96Default reports Options.FlashDecodeHD96Default: FlashDecodeHD96 is the caller's default, which CUDA applies only at a
// resident context of at most 4096.
func (m *Model) FlashDecodeHD96Default() bool { return m.opts.FlashDecodeHD96Default }

// MoECacheSlotsRequest returns the requested per-layer expert-slot count, or 0 for "as many as fit"; only meaningful with
// MoECacheExperts. 0 asks for ALL experts and lets the builder cap to measured free VRAM, deliberately: cpuengine.TopK slots is the
// worst setting for the situation this applies to, since it degenerates to fresh-loading every routed expert every token.
// allocSlots measures free VRAM and caps-and-logs rather than OOMing, so asking for all is safe. Falls back to
// GOINFER_MOE_CACHE_SLOTS.
func (m *Model) MoECacheSlotsRequest() int {
	if m.opts.MoECacheSlots > 0 {
		return m.opts.MoECacheSlots
	}
	if v, err := strconv.Atoi(m.knobs.get(knobMoECacheSlots)); err == nil && v > 0 {
		return v
	}
	return 0
}

// ErrLoadAborted is returned (wrapped) from Load when opts.LoadAbort closed mid-build. Check
// with errors.Is, not ==, since it may be wrapped with additional context by the time a caller
// sees it.
var ErrLoadAborted = errLoadAborted

// ErrPinnedContextWontFit marks a resident build's refusal of a context the caller pinned (Options.ResidentContext) whose KV does
// not fit the device. Load returns it as the load's error instead of continuing on the CPU: a pinned context asked for the
// device, so running it slowly elsewhere would hide the configuration error. Check with errors.Is.
var ErrPinnedContextWontFit = errors.New("decoder: the pinned resident context does not fit the device")

// PinnedContextRefusal is err, a backend's refusal of a pinned context, marked so that errors.Is matches ErrPinnedContextWontFit
// as well as err's own chain. Its text is err's.
func PinnedContextRefusal(err error) error { return &pinnedRefusal{err} }

type pinnedRefusal struct{ err error }

func (e *pinnedRefusal) Error() string        { return e.err.Error() }
func (e *pinnedRefusal) Unwrap() error        { return e.err }
func (e *pinnedRefusal) Is(target error) bool { return target == ErrPinnedContextWontFit }

// modelFromOptions is a Model over w carrying every per-model field Options sets: the one place the constructors (Load's.
// giw and direct paths, LoadGGUFBytes, NewModelWithOptions) read them, so they cannot drift apart (a constructor that skipped
// a field silently ignored that option). Callers set what is specific to their path (the requested quant, the resolved EOS
// ids, the file mapping), then apply backend names, knobs, streaming and residency in the order their path needs.
func modelFromOptions(w *darch.Weights, be Backend, opts Options) *Model {
	// Stamp the activation group on every quantizable weight: the CPU cpuengine.Matmul helpers read it from
	// the weight they hold, whichever backend object they run under.
	if opts.Quant == "q4k" && opts.ActQuantGroup == 0 {
		// q4k is per-32 by construction (its Q4_K kind cannot run otherwise); its int8 tensors follow.
		opts.ActQuantGroup = 32
	}
	if opts.ActQuantGroup > 0 {
		for _, wm := range w.MatmulWeights() {
			wm.SetActQuantGroup(opts.ActQuantGroup)
		}
	}
	kvF16, kvDecline := opts.KVPrecision == "f16", ""
	if be != nil {
		// A backend with a registered KV policy builds what ResolveKV says (kvresolve.go); an explicit "f16" it cannot hold says why
		// (Model.KVF16DeclineReason), and Load refuses it (kvRefusal).
		if _, ok := kvPolicies[be.Name()]; ok {
			got, why := ResolveKV(be.Name(), w.Arch, opts)
			kvF16 = got == kvPrecF16
			if !kvF16 && opts.KVPrecision == "f16" {
				kvDecline = why
			}
		}
	}
	return &Model{planSlackOf: &planSlackMemo{}, fitOf: &fitMemo{}, w: w, be: be, eosIDs: w.Cfg.EOSIDs(), kvF16Decline: kvDecline, prog: buildProgram(w.Arch, w.HasExperts), opts: opts,
		kvF16: kvF16, kvPrecI8: opts.KVPrecision == "i8", kvI8: opts.KVQuant == "i8", kvH: opts.KVQuant == "f16",
		resCtxReq: opts.ResidentContext, resCtxPinned: opts.ResidentContext > 0, // Load overrides after its fit guard
		kvSnaps: newKVSnapStore(opts.KVHostBytes), extraBytes: opts.ExtraResidentBytes + recurrentSpecReserve(w.Arch, opts.Drafter)}
}

// Load loads a model from dir, which is a checkpoint directory (config.json and its safetensors shards), a.
// gguf file, or a prequantized .giw bundle, for any supported family. opts chooses the backend, the
// weight quantization and the rest. "cpu" is always available. "cuda", "metal" and "webgpu" become
// available when their module is imported: github.com/townsendmerino/goinfer/backend/cuda (built with -tags cuda),
// ../backend/metal (on darwin) and .../backend/webgpu (built with -tags gpu). As Options.Backend says, a name not compiled
// into the binary falls back to cpu. A model that would not fit this machine's memory is refused with an
// error wrapping ErrWontFitResident.
func Load(dir string, opts Options) (*Model, error) {
	loadCalls.Add(1)
	ensureCPUSelfTest() // once per process, before any kernel runs concurrently: a CPU kernel tier that disagrees with its reference is stepped down, not trusted
	// The head resolves before the sidecar is chosen: its cache key carries the head.
	opts = resolveEmbedInt4(dir, opts.withAutoBackend())
	if p, err := sidecarRoute(dir, opts); err != nil {
		return nil, err
	} else {
		dir = p
	}
	// Options.ExactPrefill is recorded on the Model (in its opts, set by modelFromOptions before withResidency, because
	// CUDA reads it while building its resident) and consulted by each backend's fast-prefill switch alongside its env var. It is
	// deliberately not applied with os.Setenv, which is process-global and never undone: every model loaded later in the process
	// would inherit it. Serve's flags travel as this field and Options.Knobs.
	//
	// ctxFromCaller records whether the caller chose the resident context, read before either fit guard below can auto-pin one
	// into opts.ResidentContext: that pin is a one-slot ceiling, not a choice (Model.ResidentContextPinned). It is stamped on the
	// Model after modelFromOptions and before withResidency, where a backend reads it.
	ctxFromCaller := opts.ResidentContext > 0
	be, beErr := NewBackend(opts.Backend)
	// A nil backend means the name was genuinely unknown: abort rather than panic at the first cpuengine.Matmul. A non-nil be with a non-nil
	// beErr is the CPU-fallback note (webgpu/cuda/metal not built in): keep the cpu backend and surface the note.
	if be == nil {
		return nil, beErr
	}

	// A prequant bundle (.giw) aliases its weights straight out of the file (loadGIW).
	if strings.HasSuffix(dir, ".giw") {
		return loadGIW(dir, opts, be, beErr, ctxFromCaller)
	}

	// Resolve the quant mode first so the weights stream straight into the
	// chosen precision at load — no whole-model f32 spike (see loadWeights).
	quant, err := parseQuant(opts.Quant)
	if err != nil {
		closeBackend(be)
		return nil, err
	}
	if err := checkHalf16Load(quant, dir, opts.Backend); err != nil {
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

	w, err := loadWeights(dir, weightLoadOpts{quant: quant, embedInt4: opts.EmbedInt4, needCanonical: wantsCanonicalInt4(opts.Backend, be), skipRow4: !wbuild.WantsRow4Fallback(opts.layoutBackend()), lora: lora, abort: opts.LoadAbort})
	if err != nil {
		closeBackend(be)
		return nil, err
	}
	if beErr != nil {
		// webgpu requested but fell back — not fatal.
		logs.Warnf("%v", beErr)
	}
	if opts.StreamWeights {
		// Weight streaming pages weights out of the read-only mmap, which only the.
		// giw path provides; a GGUF/safetensors load dequantizes into the heap, so
		// there's nothing to page. Make the no-op visible rather than silently
		// running fully resident (prequant to .giw with cmd/prequant to use it).
		logs.Warnf("decoder: --stream-weights ignored — weights are heap-resident; prequant to .giw (cmd/prequant) to enable streaming")
	}
	resolvedEOS := resolveLoadEOS(dir, w)
	m := modelFromOptions(w, be, opts)
	m.resCtxPinned = ctxFromCaller
	m.loadPath = dir
	m.quant, m.eosIDs = opts.Quant, resolvedEOS
	m = m.withBackendNames(opts.Backend, beErr)
	m.bindKnobs(opts.Knobs.values())
	// `resident` is the third phase: weights becoming a device-side runner. Timed here rather than
	// inside withResidency because a backend that DECLINES still costs its probe, and a user
	// wondering where nine seconds went is owed that time too.
	tRes := time.Now()
	if m, err = m.withResidencyOrAbort(); err != nil {
		return nil, err
	}
	if w.Prof != nil {
		if d := time.Since(tRes); d > time.Millisecond {
			w.Prof.Record("resident", d)
		}
		m.prof = w.Prof
	}
	return m, nil
}

// resolveLoadEOS resolves a safetensors or GGUF load's EOS set (config.json plus any extra ids generation_config.json adds)
// and writes it back into w.Cfg.EOSTokenID, not just onto the Model's eosIDs. w.Cfg is what a .giw bundle serializes
// (TranscodeGIW, via SerializeWeightsToForTarget), and a .giw's own Load branch reads eosIDs straight from w.Cfg.EOSIDs()
// with no directory to re-resolve generation_config.json from. Without this, a checkpoint whose stop ids live only in
// generation_config.json (Qwen3: <|endoftext|> beside config.json's <|im_end|>) loses the extra id when it round-trips
// through cmd/prequant, and a completion that emits it runs to max_tokens. Cfg.EOSTokenID has no other reader that needs the
// unresolved value, so overwriting it is safe.
//
// generation_config.json is read through os.DirFS of a real directory. For a .gguf load dir is the file path, so
// os.DirFS(dir) can never open anything inside it (internal/runtime/ggufconfig's ggufEOS comment describes a fallback that does not cover this case); a
// GGUF conversion keeps generation_config.json beside the file, in its parent directory.
func resolveLoadEOS(dir string, w *darch.Weights) []int {
	eosDir := dir
	if strings.HasSuffix(dir, ".gguf") {
		eosDir = filepath.Dir(dir)
	}
	resolved := config.ResolveEOSIDs(eosDir, &w.Cfg)
	if raw, err := json.Marshal(resolved); err == nil {
		w.Cfg.EOSTokenID = raw
	}
	return resolved
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
	case "bf16":
		return quantBF16, nil
	case "f16":
		return quantF16, nil
	default:
		return quantNone, errorOfKind(ErrUnknownQuant, "decoder: unknown quant %q (have: int8, int8int8, int4, int4mix, q4k, bf16, f16)", q)
	}
}

// checkHalf16Load refuses a 16-bit weight mode where it cannot run: it is a safetensors, CPU-path mode (no backend uploads a 16-bit
// matrix, and a GGUF load or a bundle has no form for one).
func checkHalf16Load(q quantMode, dir, backend string) error {
	if !q.Half16() {
		return nil
	}
	if strings.HasSuffix(dir, ".gguf") {
		return fmt.Errorf("decoder: quant bf16/f16 applies to safetensors checkpoints, not %s", dir)
	}
	if backend != "" && backend != "cpu" {
		return fmt.Errorf("decoder: quant bf16/f16 runs on the CPU path only (backend %q); load with Backend \"cpu\"", backend)
	}
	return nil
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

// Close ends every generation in flight, waits for each to finish, then releases backend resources (GPU resident
// buffers + the backend) and unmaps the .giw mapping if the model was loaded from a prequant bundle. A no-op for the CPU
// backend with no mapping.
//
// Contract: Close may be called while generations are running, from any goroutine. Each running generation is cancelled
// and its channel closes with Generation.Err reporting ErrClosed; Close returns once they have all ended, so it can wait
// as long as the longest uninterruptible step (one decode step, or a prefill chunk). A Generate, Session.Generate or
// speculative call made after Close returns a closed channel at once with ErrClosed. The calls that read the weights for a
// caller are guarded the same way (HiddenLast, PromptHidden, PromptHiddenAll, PromptHiddenMany, PromptLogitsMany,
// ForwardCapture, OutputEmbeddingRow, NewBlockSpec, LoadAdapter): one running at Close is cancelled when it takes a context
// and waited for when it does not, and one made afterward returns ErrClosed. Do not call Close from inside a LogitProcessor
// or another callback these run, which would wait on itself. Calling Close again is safe and returns nil. The accessors that
// return slices of the weights (Weights, Layer(l).MLAWeights and the like) hand out views of the mapping, which is gone after
// Close; Config, Dims, Quant, NewCache and NewSession only read the config or allocate, and keep working.
func (m *Model) Close() error {
	m.endGenerations()
	if !m.lifeState().beginRelease() {
		return nil
	}
	if m.pager != nil {
		_ = m.pager.close()
		m.pager = nil
	}
	m.kvSnaps.releaseAll() // pinned host memory, freed before the device it was pinned for
	if m.resident != nil {
		_ = m.resident.Close()
		m.resident = nil
	}
	m.releaseKQuantFile()
	var err error
	if m.be != nil {
		err = m.be.Close()
	}
	if m.mmap != nil {
		_ = unmapGIW(m.mmap)
		m.mmap = nil
	}
	if m.w != nil && m.w.ST != nil { // release the safetensors mmap + per-shard fds; serve load/unload cycles otherwise retain them until GC
		_ = m.w.ST.Close()
		m.w.ST = nil
	}
	if reg := m.adapters; reg != nil {
		reg.mu.Lock()
		for _, rt := range reg.byName { // release each compute-time adapter's mmap (#7)
			rt.close()
		}
		for _, rt := range reg.retired { // and any displaced by a re-registration
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
	syncCPUHooks()
	prefillEnters.Add(1)
	var ks cpuengine.KnobSource
	if m.knobs != nil {
		ks = m.knobs
	}
	return &KVCache{c: cpuengine.NewCacheForArch(m.w.Arch, capHint, m.kvI8, m.kvH, ks)}
}

// embedToken writes the residual-stream embedding for token id into dst ([hidden])
// — the embed_tokens row with the architecture's embedding scale + GPT-2 learned
// positional embedding applied, i.e. exactly the vector runLayers feeds into
// runLayersFromEmbed. Exposed so the multimodal interleaver can build a mixed
// text/image embedding sequence (text positions via this, image positions via
// the projected vision features) and drive runLayersFromEmbed per position.
func (m *Model) embedToken(id int, dst []float32) {
	m.w.Embed.Row(id, dst)
	if emb := m.program().Embed; emb.Scale != 0 && emb.Scale != 1 {
		scale := float32(emb.Scale)
		for i := range dst {
			dst[i] *= scale
		}
	}
	// NOTE: GPT-2 learned positional embedding is position-dependent, so it is
	// applied in runLayers (which knows cache.Pos()), not here. embedToken is the
	// position-independent token embedding; the interleaver adds pos if a future
	// learned-pos family goes multimodal (none today do).
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
//
// Cancelling ctx ends the generation at its next check: before each layer of a batched CPU prefill, before each token of a
// token-at-a-time prefill (CPU or resident), between tokens while decoding, and at a resident backend's own boundaries inside
// its batched prefill (PrefillLast takes ctx). Generation.Err then reports the context's error.
func (m *Model) Generate(ctx context.Context, prompt []int, maxTokens int, sp SamplingParams) (<-chan int, *Generation) {
	out := make(chan int)
	g := &Generation{}
	// The host KV cache is not allocated here: generateInto calls newCache only when it reaches the CPU path (never resident, or
	// lost the resBusy race), so a resident-and-won call never pays for host KV capacity it never touches.
	newCache := func() *KVCache { return m.NewCache(len(prompt) + maxTokens) }
	m.runGen(ctx, out, g, nil, func(ctx context.Context) {
		m.generateInto(ctx, out, g, nil, newCache, prompt, 0, maxTokens, sp, nil)
	})
	return out, g
}

// genRun is the state of one generateInto call: its parameters and the locals its phases and closures share, so a phase can read
// and write them as fields instead of capturing them.
type genRun struct {
	m              *Model
	ctx            context.Context
	out            chan<- int
	g              *Generation
	cache          *KVCache
	newCache       func() *KVCache
	prompt         []int
	prefillFrom    int
	maxTokens      int
	sp             SamplingParams
	commit         func(int)
	sampler        *Sampler
	lora           *loraRuntime
	resAdapter     ResidentRuntimeSubsystem
	useGPU         bool
	mc3            *residentBatcher // non-nil: this generation holds a batched-decode place and, after prefill, the KV slot mc3Slot
	mc3Slot        int
	mc3Committed   bool
	gpuPos         int
	logits         []float32
	err            error
	resCost        ResidentRuntimeSubsystem
	fastNext       int
	gated          bool
	procFree       bool
	hasGreedy      bool
	fastGreedy     bool
	chainRF        ResidentDecodeSubsystem
	useChain       bool
	chainTok       bool
	chainOpen      bool
	chainPos       int
	stopChain      func()
	optFwd         bool
	topKOn         bool
	topKWidth      int
	topKVocab      int
	sampleOn       bool
	batchGreedy    bool
	sChainRF       ResidentDecodeSubsystem
	stepRow        [1][]float32 // the decode token's one-row batch, reused across tokens so a step allocates nothing
	topKRow        *TopKRow     // this step's device top-K row, set instead of logits when topKOn
	topKPre        *SampleInfo
	topKPreFull    bool
	topKFullBuf    []float32
	mc3Logits      []float32
	optGate        *optFwdGate
	clk            decodeClock
	embScratch     []float32
	decoding       bool
	exitDecode     func()
	commitResident func(generated []int)
	holdGrace      *time.Timer // the send grace for a kept hold (one timer per generation, reset per token)
	cb             *cpuBatcher
	optResolved    bool
	optNextLogits  []float32
	tg             decodeTarget
	decoded        int
	completed      bool
}

func (m *Model) generateInto(ctx context.Context, out chan<- int, g *Generation, cache *KVCache, newCache func() *KVCache, prompt []int, prefillFrom, maxTokens int, sp SamplingParams, commit func(int)) {
	r := &genRun{m: m, ctx: ctx, out: out, g: g, cache: cache, newCache: newCache, prompt: prompt, prefillFrom: prefillFrom, maxTokens: maxTokens, sp: sp, commit: commit}
	if len(r.prompt) == 0 {
		r.g.err = fmt.Errorf("decoder.Generate: empty prompt")
		return
	}
	r.sampler = NewSampler(r.sp)
	defer r.sampler.release()
	r.sampler.Observe(r.prompt...) // repetition penalties see the whole prompt, reused prefix included
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
	// the resident implements AdapterAvailable; otherwise it declines to CPU like any other missing resident capability, never
	// running an adapter session's tokens through the base model's resident weights, which returns correct-looking but wrong
	// output.
	//
	// cache may be nil here (Model.Generate defers allocation to the branch that needs it). Only Session.Generate's cache ever
	// carries lora, so the nil-safe read gives the same value either caller would have produced.

	if r.cache != nil {
		r.lora = r.cache.lora
	}

	if r.lora != nil {
		if rt := m.resRuntime(); rt.AdapterAvailable() {
			r.resAdapter = rt
		}
	}
	r.useGPU = m.resident != nil && r.prefillFrom == 0 && (r.commit == nil || (r.lora != nil && r.resAdapter != nil))
	r.mc3Slot = -1
	r.mc3Committed = false // the generation committed its slot (commitResident); else its published prompt is forgotten
	if r.useGPU {
		// The resident path drives the model's one shared positional KV; two concurrent generations would interleave writes at
		// overlapping positions and corrupt it. Claim it non-blockingly: a loser falls back to the staged CPU path, which uses this
		// call's own cache, so both complete correctly and only resident speed is lost.
		//
		// The claim also makes binding an adapter race-free even though N adapters of one base share this runner: only the resBusy
		// winner's SetAdapter/Forward/SetAdapter(nil) sequence runs at a time, so two adapter sessions (or an adapter session and a
		// base-model session) never observe each other's bound delta.
		if ctxCap := m.ResidentContextCap(); ctxCap > 0 && len(r.prompt) > ctxCap {
			// Decline before claiming the resident when the prompt exceeds its fixed KV cap: a prefill past it fails mid-write (there is
			// no CPU fallback once it commits), so fall through to the staged CPU path exactly as "not resident" does. Chiefly the adapter
			// case: an adapter's first turn (prefillFrom==0) reaches this branch, and prepare() enforces MaxPositions for it unless the
			// caller derives residentPath from ResidentActive(). This is the decoder-seam half, defense in depth for any caller that
			// does not.
			r.useGPU = false
			r.g.ResidentDeclined = fmt.Sprintf("the prompt's %d tokens pass the resident's context of %d", len(r.prompt), ctxCap)
			m.warnResidentCap(r.g.ResidentDeclined)
		} else if bt := m.batcher; bt != nil && r.lora == nil && bt.claim(&m.resBusy) {
			// Batched decode: one of several concurrent generations on the resident, on its own KV slot (mc3_batch.go).
			r.mc3 = bt
			defer bt.release()
		} else if m.tryClaimResident() {
			if r.resAdapter != nil {
				if err := r.resAdapter.SetAdapter(residentAdapterLayers(r.lora)); err != nil {
					// Bind failed after claiming resBusy — release immediately (not via defer)
					// and fall back to CPU, which applies the adapter correctly on its own
					// (applyLoRA); do NOT proceed resident with no delta bound.
					atomic.StoreInt32(&m.resBusy, 0)
					r.useGPU = false
					r.g.ResidentDeclined = "the adapter could not be bound on the resident: " + err.Error()
				} else {
					defer func() {
						r.resAdapter.SetAdapter(nil)
						atomic.StoreInt32(&m.resBusy, 0)
					}()
				}
			} else {
				defer atomic.StoreInt32(&m.resBusy, 0)
			}
		} else {
			r.useGPU = false
			r.g.ResidentDeclined = "another generation held the resident"
		}
	}
	// Allocate the host KV cache here, not before the CAS above: this covers both "never going to be resident" and "lost the race
	// for the shared resident KV", the only two ways to reach the CPU path below. A resident-and-won call never allocates one, and
	// newCache is never called when cache is non-nil (every Session.Generate call).
	if !r.useGPU && r.cache == nil {
		r.cache = r.newCache()
	}
	r.gpuPos = 0

	// resCost is set when this generation counts its cost on a resident that can (never under the batched decode). A counted generation
	// takes no decode path that queues a forward ahead of the host (the greedy and sampled chains, the optimistic forward), so
	// its launches are exactly its emitted tokens'.

	if r.useGPU {
		// A request cancelled while waiting for the model lock (tryEnter blocks with no context) reaches here with ctx already done;
		// residentForgetIDs below would discard a warm cache for a prefill residentPrefillSeed is about to refuse anyway. Check
		// before the forget: residentPrefillSeed's own ctx check runs too late to matter.
		if err := r.ctx.Err(); err != nil {
			r.g.err = err
			return
		}
		// Prefix reuse: skip the leading tokens already committed to the resident positional
		// KV and prefill only the divergent suffix. Forget FIRST — from here until the
		// generation completes the cache is mid-write, and any early return must leave the
		// next turn cold rather than trusting a half-written cache (resident_reuse.go).
		var acq kvAcquired
		scope := kvScopeOf(r.ctx)
		if r.mc3 != nil {
			// Batched decode: choose a slot no other running generation holds, and prefill into it, with the resident to
			// ourselves (between the other generations' steps).
			r.mc3.exclusive(func() {
				acq = m.residentAcquireIn(r.prompt, nil, r.lora, r.mc3.busySlots(), scope)
				r.mc3Slot = acq.slot
				if r.mc3Slot < 0 {
					r.err = fmt.Errorf("decoder: no free resident KV slot for a concurrent generation")
					return
				}
				r.mc3.markSlot(r.mc3Slot, true)
				m.residentForgetIDs()
				if acq.reuse <= 0 {
					m.residentWritesFromZero()
				}
			})
			if r.mc3Slot >= 0 {
				defer r.mc3.markSlot(r.mc3Slot, false) // after the commit below: the slot is free again once it holds our ids
			}
			if r.err == nil {
				r.logits, r.err = m.mc3Prefill(r.ctx, r.mc3, r.mc3Slot, r.prompt, acq.reuse)
			}
			if kvPoolFullEnds(r.g, r.err, 0) {
				return
			}
			if r.err != nil {
				r.g.err = r.err
				return
			}
			// The prompt's rows are final: publish them for a request that arrives while this one decodes. A generation
			// that ends without its commit forgets them, as it would anything it wrote (before markSlot frees the slot).
			r.mc3.exclusive(func() {
				m.residentPublishPrompt(r.mc3Slot, r.prompt, r.lora)
				m.residentSnapshotPrompt(r.mc3Slot, r.prompt, r.lora, scope, acq)
			})
			defer func() {
				if !r.mc3Committed {
					r.mc3.exclusive(func() { m.kvSlots().Forget(r.mc3Slot) })
				}
			}()
		} else {
			scope.checkpoint = m.recurrentCheckpointer() != nil          // recurrent checkpoint: this path restores, and saves after its prefill
			acq = m.residentAcquireIn(r.prompt, nil, r.lora, nil, scope) // binds the resident KV slot this prompt will use
			m.residentForgetIDs()
			mk := m.kvPrefillMark(acq.slot, r.prompt, acq.reuse, r.lora, scope, &acq) // the boundary snapshot, taken between two passes
			r.resCost = m.residentCostFor(cost.FromContext(r.ctx), false)
			r.err = costSpan(r.resCost, cost.FromContext(r.ctx), cost.Prefill, func() (int, error) {
				var perr error
				r.logits, perr = m.residentPrefillSeedMarked(r.ctx, r.prompt, acq.reuse, r.lora != nil, mk)
				return prefillComputed(len(r.prompt), acq.reuse), perr
			})
			if r.err != nil {
				if !kvPoolFullEnds(r.g, r.err, 0) {
					r.g.err = r.err
				}
				return
			}
			m.residentSnapshotPrompt(acq.slot, r.prompt, r.lora, scope, acq)             // before the first decode step writes past the prompt
			r.g.CheckpointSave, r.g.CheckpointSaved = m.residentCheckpointSave(r.prompt) // likewise: the state at the end of the prompt
		}
		r.g.PrefillReused, r.g.PrefillSlot, r.g.PrefillMatch = acq.reuse, acq.slotLen, acq.match
		r.g.PrefillShared, r.g.KVCopy, r.g.KVCopyDeclined = acq.shared, acq.copyDur, acq.declined
		r.g.KVSnapRestored, r.g.KVSnapRestore = acq.snapRows, acq.snapDur
		r.g.KVPromotedBlocks, r.g.KVPromotedRows = acq.promotedBlocks, acq.promotedRows
		r.g.CheckpointRestored, r.g.CheckpointRestore, r.g.CheckpointDeclined = acq.ckptRestored, acq.ckptDur, acq.ckptDeclined
		r.gpuPos = len(r.prompt)
	} else {
		counted := false
		if c := cost.FromContext(r.ctx); c != nil { // the request asked for its cost; the walker counts into c
			r.cache.c.Cost, r.cache.c.CostPhase, counted = c, cost.Prefill, true
			defer func() { r.cache.c.Cost = nil }() // a session's cache outlives this generation
		}
		r.cache.c.PlanRequest(len(r.prompt), r.maxTokens, m.kvRowCeiling()) // the KV is sized once for the prompt and its budget
		if r.logits, r.err = m.prefillLogits(r.ctx, r.prompt[r.prefillFrom:], r.cache); r.err != nil {
			r.g.err = r.err
			return
		}
		if counted {
			r.cache.c.CostPhase = cost.Decode
		}
	}
	r.chooseDecodePath()
	defer r.stopChain()

	r.chooseFastPaths()
	// Clamp the decode length to the resident KV cap up front. A Forward past the cap is refused mid-generation (the
	// silent-corruption guard), but a resident that exposes its cap lets us stop cleanly at it. gpuPos is the next decode
	// position; the last valid one is ctxCap-1, and the prefill loop guarantees gpuPos <= ctxCap.
	if r.useGPU {
		if c, clamped := m.clampToResidentCap(r.gpuPos, r.maxTokens); clamped {
			r.maxTokens, r.g.BudgetClamped = c, true
		}
	}
	// Publish the effective budget so the caller reports finish_reason "length" when this clamp, not an EOS, ends the turn. Set
	// before any send; read after close. BudgetClamped disambiguates a genuine clamp-to-0 (the prompt fills the cap) from an
	// unclamped turn whose Budget is coincidentally 0.
	r.g.Budget = r.maxTokens

	// The decode loop is plainDecodeLoop (policy_decode.go); the hooks below are this generation's paths.

	// embScratch is reused across iterations: each token's embedding is consumed synchronously by Forward/ForwardArgmax before the
	// next is requested, so one buffer replaces a fresh [hidden]float32 per token.

	r.bindDecodeExits()
	if r.mc3 != nil {
		r.mc3.enterDecode()
		r.decoding = true
		defer r.exitDecode()
		// Runs before exitDecode (defers unwind last-first): any exit, an error's included, ends a kept hold.
		defer r.mc3.releaseHold(r.stopChain)
	}
	if r.mc3 != nil {
		r.holdGrace = time.NewTimer(mc3HoldSendGrace)
		r.holdGrace.Stop()
		defer r.holdGrace.Stop()
	}
	// Batched CPU decode (cpu_batch.go): a CPU generation on an eligible cache submits its decode tokens to the model's CPU
	// batcher, which joins them with other generations' into one batched step; alone, a token runs m.forward as always.
	// A generation that counts its cost decodes alone: a batched step's work is not attributed per generation.

	if !r.useGPU && m.cpuBatch != nil && cpuengine.CPUBatchCacheEligible(unwrapCache(r.cache)) == nil && r.cache.c.Cost == nil {
		r.cb = m.cpuBatch
		r.cb.enterDecode()
		defer r.cb.exitDecode()
	}

	r.buildTarget()
	r.completed = false
	_ = costSpan(r.resCost, cost.FromContext(r.ctx), cost.Decode, func() (int, error) {
		r.completed = plainDecodeLoop(r.ctx, r.tg, r.out, r.g, r.sampler, r.sp, r.maxTokens, func(id int) bool { return m.isStop(id, r.sp) }, r.commit, &r.clk)
		return r.decoded, nil
	})
	if !r.completed {
		return
	}
	r.logDecodeTiming()
}

// kvPoolFullEnds reports whether err is the resident pool's refusal of a conversation past its guaranteed share (ErrKVPoolFull), and if so
// ends g as the context cap would: the tokens it emitted are its budget, so the turn reports finish_reason "length".
func kvPoolFullEnds(g *Generation, err error, emitted int) bool {
	if !errors.Is(err, ErrKVPoolFull) {
		return false
	}
	g.Budget, g.BudgetClamped = emitted, true
	return true
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
	// PrefillSlot is how many positions the KV record this generation reused from held before it started (the bound resident
	// slot's, or a Session's), and PrefillMatch how many leading prompt tokens equal that record's, before any rounding to what
	// may be reused: when PrefillMatch < PrefillSlot the prompt left the record at PrefillMatch. Diagnostics for -log-requests;
	// 0 on paths that do not report them.
	PrefillSlot, PrefillMatch int
	// PrefillShared is how many of the PrefillReused rows were copied from another conversation's KV slot
	// (KVCopierAvailable); KVCopy is that copy's wall time, and KVCopyDeclined why a copy the index offered was not
	// made ("" when none was declined). Diagnostics for the sharing gate (docs/measurements/kv2-2026-10-14.md).
	PrefillShared  int
	KVCopy         time.Duration
	KVCopyDeclined string
	// KVSnapRestored is how many of the PrefillReused rows a host snapshot restored (Options.KVHostBytes), and KVSnapRestore that
	// restore's wall time. 0 when none did.
	KVSnapRestored int
	KVSnapRestore  time.Duration
	// KVPromotedBlocks is how many blocks came back from host memory after the pool released them (KVDemoterAvailable): the bound slot's own, or
	// those of another slot's copy written into it. KVPromotedRows is how many of the PrefillReused rows the bound slot's serve; rows that reached it
	// through a copy count in PrefillShared. 0 when none did.
	KVPromotedBlocks, KVPromotedRows int
	// CheckpointRestored is the position a recurrent checkpoint was restored at for this prefill (0: none), CheckpointRestore its
	// copy's time, CheckpointDeclined why a checkpoint hit prefilled cold instead; CheckpointSaved reports whether the state at the
	// end of this prompt was saved for the next request, CheckpointSave that copy's time (residentCheckpointSave).
	CheckpointRestored int
	CheckpointRestore  time.Duration
	CheckpointDeclined string
	CheckpointSaved    bool
	CheckpointSave     time.Duration
	// ImgPrefillResident reports whether GenerateVL's resident image-prefill fast path (ImagePrefillAvailable, the bidirectional
	// image-block kernel) ran this turn's prefill on the GPU; false means it fell through to the CPU-prefill+UploadKV bridge,
	// whether because no resident implements it, the prompt was too long for one chunk, or any other decline. Diagnostic: a
	// real-checkpoint gate needs it to confirm the fast path fired rather than passing vacuously through the (correct) fallback.
	ImgPrefillResident bool
	// ImgPrefillDecline is the reason the resident image prefill declined this turn, when a backend that implements it
	// was asked and said no (a multi-image turn on a backend whose resident prefill takes one block). Empty when it
	// ran, or when no resident implements it.
	ImgPrefillDecline string
	// ResidentDeclined is why this generation ran on the CPU path on a model with a resident runner it could have used: its
	// prompt passed the resident's context (ResidentContextCap), another generation held the resident, or an adapter failed
	// to bind. "" when it ran on the resident, or the model has no resident.
	ResidentDeclined string
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

// Err returns the error that ended the stream, or nil if it ended cleanly (EOS / stop / maxTokens). Read it after the
// channel closes. A cancelled or expired context ends the stream with an error that errors.Is matches against ctx.Err()
// (context.Canceled or context.DeadlineExceeded), whether it fired before the first token or mid-decode.
func (g *Generation) Err() error { return g.err }
