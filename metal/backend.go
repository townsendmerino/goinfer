//go:build darwin

package metal

// Metal decoder backend — plugs the cgo-free native-Metal resident decoder into the goinfer
// decode loop via decoder.RegisterBackend, mirroring the cuda backend. Blank-import this
// package from a main (darwin only; no build tag) to enable `--backend metal`. Dense and MoE
// residency, per docs/hardware-matrix.md; a model it does not admit runs entirely on the CPU
// (Metal has no partial staged path).

import (
	"context"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"sync"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/decoder"
	"golang.org/x/sys/unix"
)

func init() {
	decoder.RegisterBackend("metal", func() (decoder.Backend, error) {
		return &metalBackend{}, nil
	})
	// residentFitsMemory's own budget, exposed to decoder.Model.Plan (docs/tasks/task-fit-to-hardware.md
	// Phase 1) via the SAME arithmetic — not a live "available" query, deliberately: darwin's UBC
	// reclaim makes "available" report what survived rather than what can be asked for (the
	// residentMemFraction comment above this file's own guard). Plan sees exactly the number the
	// real guard would judge it against, so the two can never disagree.
	decoder.RegisterMemoryProbe("metal", func() (int64, bool) {
		ram, err := unix.SysctlUint64("hw.memsize")
		if err != nil || ram == 0 {
			return 0, false
		}
		return metalMemoryCeiling(ram), true
	})
}

// Compile-time seams: catch signature drift against decoder/residency.go + decoder/backend.go.
var (
	_ decoder.Backend               = (*metalBackend)(nil)
	_ decoder.ResidencyBackend      = (*metalBackend)(nil)
	_ decoder.ResidentForward       = (*metalResident)(nil)
	_ decoder.ResidentMRoPE         = (*metalResident)(nil)
	_ decoder.ResidentAdapter       = (*metalResident)(nil)
	_ decoder.Prefiller             = (*metalResident)(nil)
	_ decoder.PrefillPathReporter   = (*metalResident)(nil)
	_ decoder.VerifyPathReporter    = (*metalResident)(nil)
	_ decoder.ResidentQuantReporter = (*metalResident)(nil)
)

// metalBackend implements decoder.Backend + decoder.ResidencyBackend.
type metalBackend struct {
	resident *metalResident // set by BuildResident; released in Close
}

func (b *metalBackend) Name() string { return "metal" }

// MatmulBT is the CPU fallback on the shared SIMD linalg kernels (same as the CPU backend).
// Metal has no partial (staged) GPU path: a model the resident runner does not admit runs
// entirely on the CPU, and this keeps `--backend metal` correct in that case.
func (b *metalBackend) MatmulBT(a, bmat, dst []float32, M, K, N int) {
	linalg.MatmulBT(a, bmat, dst, M, K, N)
}

// BuildResident builds the resident Metal decoder from a loaded dense Model and wraps it in
// an adapter satisfying decoder.ResidentForward. Never crashes the process: BuildResident
// compiles MSL / creates the device and panics on failure — recover → decline (ok=false) →
// the decoder falls back to the staged/CPU path. Callers gate on DecodeRunnerEligible first;
// weights load as either int4 or int8 (Options{Quant:"int4"/"int8int8"}); an f32 projection declines with the
// quant to use (f32Projection). A dense int8 model runs natively on W8A8 kernels (r.w8, docs/tasks/task-metal-int8-2026-10.md);
// one native int8 does not cover (MoE, DeltaNet, int4mix) is re-quantized to int4 through the W4A8 packer an int4 load
// uses, and decoder.Model.DecodePath() names that re-quantization rather than repeating the requested quant.
func (b *metalBackend) BuildResident(m *decoder.Model) (rf decoder.ResidentForward, ok bool, err error) {
	defer func() {
		if p := recover(); p != nil {
			rf, ok, err = nil, false, decoder.DeclineResident("metal build panicked: %v", p)
		}
	}()
	// Admission check: DecodeRunnerEligible was scoped to the richer WebGPU/CUDA runner, which
	// admits QK-norm / sliding-window / partial-rotary and more. The Metal kernel set implements
	// only a subset (decoder.ResidentBackendFeatures("metal")); anything it doesn't implement must
	// DECLINE (→ correct CPU fallback) rather than run with the feature silently dropped. The
	// subset check uses the shared taxonomy (one source of truth; a new arch classifies itself).
	if missing := m.MissingResidentFeatures(decoder.ResidentBackendFeatures("metal")); len(missing) > 0 {
		return nil, false, decoder.DeclineResident("metal does not implement %v, which this model needs", missing)
	}
	// FITS-IN-MEMORY GUARD. Metal's unified memory IS host RAM, and it WIRES the mmap pages a
	// command buffer touches, so a model whose weights exceed RAM does not merely run slowly —
	// it pages until swap is exhausted. Measured 2026-08-31 on a 16 GB MacBook with gpt-oss-20b
	// (11.28 GB of weights): swap went to 35.98 GB of 36 GB, the process sat in uninterruptible
	// I/O wait with RSS creeping 1.8 -> 2.0 GB over 12 minutes, and the load NEVER completed and
	// never declined. Declining is strictly better than that, and there was no check at all: the
	// only size guard here caps the KV CONTEXT (checkCap, above), not the weights.
	//
	// Keyed on two quantities WE compute — the model's own weight bytes and the machine's
	// physical RAM — never on the OS's account of what is free. Darwin's UBC reclaims under
	// pressure, so "available" reports what survived rather than what can be asked for; an
	// RSS-keyed ceiling once reported LESS memory at a known failure point than at baseline,
	// which is a guard that inverts exactly when it is needed.
	// G6 (docs/tasks/task-gpu-paths-2026-09.md — "honoured or refused with numbers"): an explicit -ctx
	// above metalCtxCapMax cannot be honoured (a fixed-size kernel score buffer, not a tunable
	// budget) and must be a NAMED refusal, not folded into the generic "BuildResident declined"
	// swallow below (buildResident itself also calls resolveMetalCtxCap and would hit the exact
	// same error, but by then it's just another opaque build failure) — mirrors CUDA's own
	// errKVWontFit precedent (cuda/backend.go): an operator's explicit request that cannot be
	// honoured gets a specific, propagated error naming the numbers, not a silent CPU fallback
	// indistinguishable from "this arch doesn't fit here at all".
	if _, cerr := resolveMetalCtxCap(m); cerr != nil {
		fmt.Fprintf(os.Stderr, "[metal] %v\n", cerr)
		return nil, false, cerr
	}
	from, _ := resolveMetalCtxCap(m)
	if why := residentMemoryDecline(m); why != "" {
		if !shrinkCtxToFit(m) {
			return nil, false, decoder.DeclineResident("%s", why)
		}
	}
	res, e := buildResident(m)
	if e != nil {
		metalCtxCeiling.Delete(m)
		return nil, false, decoder.DeclineResident("metal: %v", e)
	}
	if _, ok := metalCtxCeiling.Load(m); ok {
		res.ctxCeilKey = m // the entry lives as long as the resident (resident.Close deletes it)
	}
	b.resident = &metalResident{r: res, hidden: res.H, exact: m.ExactPrefill()}
	if got, want := b.resident.embLen(), m.ResidentEmbedLen(); got != want {
		_ = res.Close()
		b.resident = nil
		return nil, false, decoder.DeclineResident("metal: resident takes %d-float embedding rows, the decoder builds %d", got, want)
	}
	if res.w8 || res.w8Attn {
		b.resident.quant = m.Quant()
	}
	if ctx, _ := resolveMetalCtxCap(m); ctx < from || m.ExtraResidentBytes() > 0 {
		fmt.Fprint(os.Stderr, metalKVPlanLine(res.kvSlotCount(), ctx, from, m.ExtraResidentBytes()))
	}
	return b.resident, true, nil
}

// metalCtxCeiling is S18's context ceiling per model (docs/tasks/task-multimodal-support-2026-10.md, "S18 on the Mac"):
// what shrinkCtxToFit lowered an unpinned resident context to, read by resolveMetalCtxCap. A map keyed by the model, so
// the decoder keeps no backend state.
var metalCtxCeiling sync.Map // *decoder.Model -> int

// metalCtxFloor is the shortest context shrinkCtxToFit gives a model before the guard declines: an image turn (Gemma 3's
// 256 soft tokens, a few hundred of text) and some conversation fit.
const metalCtxFloor = 2048

// shrinkCtxToFit halves an unpinned resident context, down to metalCtxFloor, until the memory guard passes (S18: the
// context gives way before the resident declines; an explicit -ctx never does). It reports whether one passed; when none
// does the context is left as it was.
func shrinkCtxToFit(m *decoder.Model) bool {
	if m.ResidentContextPinned() {
		return false
	}
	cur, err := resolveMetalCtxCap(m)
	if err != nil {
		return false
	}
	for cur > metalCtxFloor {
		cur = max(cur/2, metalCtxFloor)
		metalCtxCeiling.Store(m, cur)
		if residentMemoryDecline(m) == "" {
			return true
		}
	}
	metalCtxCeiling.Delete(m)
	return false
}

// metalKVPlanLine is serve's KV plan line (S18's banner): the slots and context the build chose, and what for when it
// is less than asked or a companion (a vision tower, a drafter) is priced beside it.
func metalKVPlanLine(slots, ctx, from int, extra int64) string {
	s := fmt.Sprintf("metal: KV plan: %d conversation(s) x %d positions", slots, ctx)
	switch {
	case ctx < from && extra > 0:
		s += fmt.Sprintf(" (context reduced from %d to leave %.1f GB for the vision tower or drafter)", from, float64(extra)/(1<<30))
	case ctx < from:
		s += fmt.Sprintf(" (context reduced from %d to fit this machine's memory)", from)
	case extra > 0:
		s += fmt.Sprintf(" (%.1f GB left for the vision tower or drafter)", float64(extra)/(1<<30))
	}
	return s + "\n"
}

// residentMemFraction is the share of physical RAM the WEIGHTS alone may occupy. The remainder
// is not slack: the KV cache, per-layer scratch, the command buffers, and the rest of the system
// all live in the same unified memory.
//
// 0.70 is set from ONE measured failure (11.28 GB of 16 GB = 70.5% thrashed to swap exhaustion)
// and is therefore a threshold, not a curve — it is honestly a single point, and a machine that
// would in fact have fit can override with GOINFER_NO_RESIDENT_MEM_GUARD=1 rather than be told
// no by a number nobody has swept. What it must not do is silently pass the case it was written
// for, which is why the bar sits just below that measurement rather than at a rounder 0.75.
const residentMemFraction = decoder.WeightsMemFraction

// metalLiveAvailable is decoder.HostRAMAvailableBytes, indirected so a test can inject a
// machine's live-available figure instead of reading the real one — mirrors decoder/fitguard.go's
// own hostRAMAvailable indirection for the identical reason (a test needs a machine's worth of RAM
// without needing one).
var metalLiveAvailable = decoder.HostRAMAvailableBytes

// metalStaticCeiling is residentMemFraction's own arithmetic, split out so fitsResidentBudget and
// metalMemoryCeiling below share one formula rather than two copies that happen to agree.
func metalStaticCeiling(ram uint64) int64 { return int64(float64(ram) * residentMemFraction) }

// fitsResidentBudget is the STATIC arithmetic alone, split out so it can be driven with the
// numbers from the measurement instead of requiring a 12 GB checkpoint to exercise the guard.
// Deliberately never calls the live probe (metalMemoryCeiling does that) — its own tests hand it
// exact ram/need figures and expect the pure ratio, not whatever this machine's real available
// memory happens to be at test time.
func fitsResidentBudget(need int64, ram uint64) bool {
	if need <= 0 || ram == 0 {
		return true // unknown ⇒ do not refuse
	}
	return uint64(need) <= uint64(metalStaticCeiling(ram))
}

// metalMemoryCeiling is S4 item 3 (task-never-swap-2026-09.md): the SINGLE arithmetic both
// residentFitsMemory (the real load-time guard) and the registered "metal" memory probe
// (decoder.Model.Plan's own view, above) now share, so the two can never disagree — the property
// RegisterMemoryProbe's own comment already promised and, before this, did not fully keep: it
// recomputed ram*residentMemFraction independently of residentFitsMemory's own copy of the same
// formula, two copies that happened to agree only because neither had a second term yet.
//
// Adds a SECOND bound from HostRAMAvailableBytes, min'd against the static 70% ceiling — so the
// result can only get STRICTER than the static ceiling alone, never looser. This is deliberate:
// darwin's UBC reclaim means "available" can look artificially GENEROUS under memory pressure (a
// guard keyed on the live figure ALONE would invert exactly when it's needed — the same class of
// mistake an RSS-keyed budget guard made elsewhere in this repo, reporting LESS memory at a known
// failure point than at baseline), but min() can never make the COMBINED ceiling look artificially
// SCARCE, because it only ever lowers what the static 70% already allowed — the one direction a
// stricter-not-looser bound is safe to move in. live<=0 (no probe on this platform, or genuinely
// unknown) leaves the static ceiling untouched, same as every other "unknown ⇒ proceed" case here.
//
// ram==0 returns 0: every caller here already treats a zero budget from an unreadable hw.memsize
// as "unknown", unchanged.
func metalMemoryCeiling(ram uint64) int64 {
	if ram == 0 {
		return 0
	}
	ceiling := metalStaticCeiling(ram)
	if live := metalLiveAvailable(); live > 0 && live < ceiling {
		return live
	}
	return ceiling
}

// metalMoESlotsRequest is the resolved expert-slot request as a string, ready for the SAME
// strconv.Atoi + validation metal/moe.go and metal/gemma4_moe.go already do at their real
// dispatch-building call sites: `--moe-cache-slots` / decoder.Options.MoECacheSlots
// (m.MoECacheSlotsRequest(), the SAME flag CUDA's own auto-cap-to-VRAM already reads) wins when
// set (Phase 2, docs/tasks/task-gpu-paths-2026-09.md — "Metal slots become an Option and a flag");
// GOINFER_METAL_MOE_SLOTS is kept as a deprecated fallback for anyone still setting it directly.
// "" means unset either way — n==0/unset ⇒ every expert resident, today's behavior, unchanged.
func metalMoESlotsRequest(m *decoder.Model) string {
	if n := m.MoECacheSlotsRequest(); n > 0 {
		return strconv.Itoa(n)
	}
	// M-13 (audit-metal-2026-09-12.md): --moe-cache-experts alone (no explicit --moe-cache-slots)
	// used to read as "0 ⇒ unpaged" here, so a user following the CLI's own advice for a
	// bigger-than-RAM MoE (26B/35B/gpt-oss-20b) got every-expert-resident anyway, which the memory
	// guard then declined to the CPU-staged path the peer-matrix row actually measured. CUDA
	// already auto-caps to free VRAM in this exact shape (MoECacheExperts set, slots unset); this
	// is Metal's twin.
	if m.MoECacheExperts() {
		return strconv.Itoa(autoMoESlots(m))
	}
	return modelKnob(m, "GOINFER_METAL_MOE_SLOTS")
}

// moeTopK is this model's own top-k routed-experts-per-token count — the floor autoMoESlots must
// clamp above, since fewer slots than top-k cannot hold one token's own routed set simultaneously
// (the same floor buildResident enforces on an explicit --moe-cache-slots, gated by
// TestMoESlotsViaOptions_belowTopKRefusesWithNumbers). Gemma-4's MoE and the generic MoE families
// store their top-k under different Config fields (registry.go's own per-architecture TopK:
// assignments), so this branches on which shape the model actually is rather than guessing one
// field name works for both.
func moeTopK(m *decoder.Model) int {
	if m.HasGemma4MoEResident() {
		if k := m.Config().TopKExperts; k > 0 {
			return k
		}
		return 1
	}
	if _, k, _, _, _, _, _, _, _, _, ok := m.MoEResidentParams(); ok && k > 0 {
		return k
	}
	return 1
}

// autoMoESlotsMax is the per-layer expert-slot ceiling this auto-sizer will request even when
// memory allows more — task-metal-expert-streaming-at-scale.md's own measurement run settled on
// N=64 as the recommended default; raise it only if a future measurement moves that number.
const autoMoESlotsMax = 64

// autoMoESlotsFor is autoMoESlots' pure formula, split out (fitsResidentBudget's own pattern, and
// residentNeedBytes' own reason for existing) so it can be unit-tested directly against a
// deliberately small ram value — a real machine holding less RAM than a real MoE model's dense
// term is not available to test against otherwise.
//
// Solves the same inequality residentFitsMemory's guard checks (0.7·RAM ≥ need) for N instead of a
// pass/fail: need(N) = needFixed + N·perSlot, where needFixed is everything that does NOT scale
// with the slot count (dense weights doubled by Metal's host-copy-plus-device-buffer footprint,
// plus KV) and perSlot is one additional expert slot's marginal bytes. Clamped to
// [topK, autoMoESlotsMax]: below topK a token's own routed set cannot fit simultaneously (the same
// floor buildResident enforces on an explicit --moe-cache-slots); autoMoESlotsMax is
// task-metal-expert-streaming-at-scale.md's own measured recommendation. perSlot<=0 or a
// non-positive remaining budget both fall back to a safe default (autoMoESlotsMax or topK
// respectively) rather than a fabricated small number — the memory guard downstream still has the
// final say either way.
func autoMoESlotsFor(topK int, perSlot, needFixed int64, ram uint64) int {
	return autoMoESlotsForBudget(topK, perSlot, needFixed, metalStaticCeiling(ram))
}

// autoMoESlotsForBudget is the formula against an explicit byte budget. autoMoESlots passes the
// guard's OWN budget (metalMemoryCeiling — the static fraction, tightened by live-available memory
// since S4): sizing against the static fraction while residentFitsMemory checks the live one made
// the two disagree under memory pressure — the sizer picked N slots, the guard refused N, and the
// load fell to CPU instead of taking fewer slots.
func autoMoESlotsForBudget(topK int, perSlot, needFixed, budget int64) int {
	if perSlot <= 0 {
		return max(topK, autoMoESlotsMax)
	}
	remaining := budget - needFixed
	if remaining <= 0 {
		return topK // the fixed part alone doesn't fit; buildResident/the guard will refuse either way
	}
	n := int(remaining / perSlot)
	return max(topK, min(n, autoMoESlotsMax))
}

// autoMoESlots derives a per-layer expert-slot count from live memory for a model that asked to
// stream MoE experts (MoECacheExperts) but gave no explicit slot count (M-13). perSlotBytes is
// read from the SAME byte accessor residentNeedBytes already calls, at the marginal cost of one
// additional slot (ResidentWeightBytesPaged(2)-ResidentWeightBytesPaged(1)) rather than re-derived
// by hand — a layer's experts are uniform in shape (that accessor's own doc comment), so the
// marginal slot cost is exact, not approximated, and stays correct if the byte accounting itself
// ever changes. An unreadable hw.memsize falls back to autoMoESlotsMax (see autoMoESlotsFor).
func autoMoESlots(m *decoder.Model) int {
	topK := moeTopK(m)
	perSlot := m.ResidentWeightBytesPaged(2) - m.ResidentWeightBytesPaged(1)
	ram, err := unix.SysctlUint64("hw.memsize")
	if err != nil || ram == 0 {
		return max(topK, autoMoESlotsMax)
	}
	needFixed := m.ResidentDenseWeightBytes() + m.ResidentHostCopyBytes(1) + residentKVBytes(m) // ResidentHostCopyBytes(>0) is dense-only (paged)
	return autoMoESlotsForBudget(topK, perSlot, needFixed, metalMemoryCeiling(ram))
}

// metalMoESlotsFromEnv is the guard's own reader (residentNeedBytes, below) — it needs an int,
// not a validated dispatch-ready string, and unlike the two real call sites an invalid or unset
// value is not an error here: it just means "assume unpaged" (the guard's existing, safe
// behavior); buildResident itself still validates and declines on a bad value. Despite the name
// (kept for now — see metalMoESlotsRequest's own doc comment on why the underlying knob is no
// longer env-only), this reads the SAME resolved request metalMoESlotsRequest does.
func metalMoESlotsFromEnv(m *decoder.Model) int {
	n, err := strconv.Atoi(metalMoESlotsRequest(m))
	if err != nil || n <= 0 {
		return 0
	}
	return n
}

// residentKVBytes is the resident KV cache's footprint: the resolved ctx cap × kvDim × 2 bytes
// (f16 KV — the only path this backend ships; the f32 KV kernels exist but are compiled out,
// metal/model.go's kvF32 hardcoded false since the Gemma "crater" traced to BOS K/V, not
// precision) × 2 buffers (K and V), summed per layer since a per-layer geometry family (Gemma 4)
// varies kvDim between local and global layers — a single model-level figure would misprice
// whichever shape isn't the majority. This is the dominant "scratch" term M-02 named as missing;
// the many small per-model buffers (r.mq/r.gu/r.logits/etc, metal/model.go) are each at most a
// few hundred KB — elementwise or vocab/hidden-sized — and round to nothing beside it, the same
// exemption ResidentWeightBytes' own doc comment already gives norms/biases.
//
// Runs BEFORE buildResident (this is the pre-build guard), so there is no real *resident yet to
// read ctxCap off of — resolveMetalCtxCap(m) is called independently here. An error (an explicit
// ctx above metalCtxCapMax) is NOT this function's problem to report: residentFitsMemory's caller
// (metalBackend.BuildResident) checks that specific case on its own, earlier and more clearly, so
// this just falls back to metalCtxCapMax on error — a safe over-estimate for a guard whose whole
// job is "don't under-count", never the thing that actually explains the refusal to the user.
func residentKVBytes(m *decoder.Model) int64 {
	ctxCap, err := resolveMetalCtxCap(m)
	if err != nil {
		ctxCap = metalCtxCapMax
	}
	// The one definition, shared with decoder.Model.Plan("metal") so `fit` and this guard price KV
	// identically (docs/tasks/task-memory-accounting-2026-09.md): f16 (or int8 + per-head scales),
	// every attention layer at the full ctx padded to 8, no Gated-DeltaNet layer.
	return m.ResidentKVBytes("metal", ctxCap, false, false)
}

// residentNeedBytes is the byte count residentFitsMemory judges against — split out from
// residentFitsMemory so it can be unit-tested directly, without real RAM or a checkpoint large
// enough to swing the guard's verdict.
//
// M-02: this used to always be m.ResidentWeightBytes() — the UNPAGED number — even when the
// caller had set GOINFER_METAL_MOE_SLOTS, so a model that fits fine under paging (a few GB) was
// declined on the number it would need with every expert resident (tens of GB for a large MoE).
// Reading the same knob buildResident is about to honor and asking for the PAGED estimate instead
// fixes the audit's Qwen3.5-35B-A3B example without moving the guard itself — it still runs
// before buildResident, on the byte count that will actually apply once paging is resolved.
//
// M-02 (continued, 2026-09-09): the weight term alone under-counted by ~2x on a real GGUF/
// safetensors load (docs/audit-2026-09-02.md's "~9 GB Q4_K_M GGUF... lands at ~18 GB anonymous"
// example) because Metal's unified memory holds the quantized HOST WeightMat AND a freshly
// re-packed device buffer for the same weights (decoder.Model.ResidentHostCopyBytes' own doc
// comment). Genuinely paged experts are exempt (they stream, no host copy), which is why this
// asks for the host-copy addend at the SAME slot count rather than assuming it doubles the whole
// weight term. KV was entirely absent; residentKVBytes above closes that.
func residentNeedBytes(m *decoder.Model) int64 {
	ctxCap, err := resolveMetalCtxCap(m)
	if err != nil {
		ctxCap = metalCtxCapMax
	}
	// decoder.Model.ResidentNeedBytes is also what Plan("metal").NeedBytes() is built from, so `fit`'s
	// verdict and this guard's decision are the same number by construction.
	// S18: plus what a companion will claim beside the resident (a vision tower priced by serve's towerReserve, a
	// drafter): Metal's "device" memory is the same RAM, and the guard used to see none of it.
	return m.ResidentNeedBytes("metal", metalMoESlotsFromEnv(m), ctxCap, false, false) +
		m.ExtraResidentBytes() + m.ExtraResidentKVPerPosition()*int64(ctxCap)
}

// residentFitsMemory reports whether this model's weights fit the machine, declining loudly when
// they do not. True (proceed) whenever the answer is unknown — an unreadable hw.memsize or a
// model reporting zero bytes must not silently disable residency for everyone.
func residentFitsMemory(m *decoder.Model) bool { return residentMemoryDecline(m) == "" }

// metalKVSlots is how many resident KV slots buildResident allocates for m (MC1, docs/tasks/task-concurrency-2026-09.md):
// the model's request (decoder.Model.ResidentKVSlotsRequest, which is already 1 for a family with recurrent state),
// reduced while the extra slots would take the build over the memory guard's budget. The first slot is the build's own
// KV, which residentMemoryDecline prices; each further one costs another residentKVBytes. The guard's own override
// (GOINFER_NO_RESIDENT_MEM_GUARD) allocates the request as asked; an unreadable RAM size allocates one.
func metalKVSlots(m *decoder.Model) int {
	n := m.ResidentKVSlotsRequest()
	if m.ResidentKVSlotsIsDefault() && n > metalDefaultKVSlots {
		fmt.Fprintf(os.Stderr, "metal: %d resident KV slots (Metal's default; %d asked by default, -kv-sessions N keeps N if memory allows)\n",
			metalDefaultKVSlots, n)
		n = metalDefaultKVSlots // E-P09: the caller's default, not the operator's count
	}
	if n <= 1 {
		return 1
	}
	if modelKnob(m, "GOINFER_NO_RESIDENT_MEM_GUARD") != "" {
		return n
	}
	ram, err := unix.SysctlUint64("hw.memsize")
	if err != nil || ram == 0 {
		return 1
	}
	got := kvSlotsWithin(n, metalMemoryCeiling(ram), residentNeedBytes(m), residentKVBytes(m))
	if got < n {
		fmt.Fprintf(os.Stderr, "metal: %d resident KV slots of %d requested — each costs %.0f MB of KV at the resident context, and the memory guard allows %d\n",
			got, n, float64(residentKVBytes(m))/(1<<20), got)
	}
	return got
}

// metalDefaultKVSlots is how many resident KV slots Metal keeps when the count asked is the caller's default
// (decoder.Options.ResidentKVSlotsDefault: serve without -kv-sessions), E-P09 of docs/audit-metal-2026-09-30.md. Each
// slot's KV is resident from the first token on (T1.6: 4 slots cost 335 MB more than 1 on the 1.5B after one token,
// though only slot 0 was written), so the default 4 cost about 224 MB on the 1.5B and 470 MB on the 7B over 2. Two
// keep MC3's batched step (it needs 2 slots) and a second conversation's prefix; a third and fourth concurrent client
// lose their own slot and their batching. -kv-sessions N still asks for N.
const metalDefaultKVSlots = 2

// kvSlotsWithin is metalKVSlots' arithmetic: the largest slot count up to want whose resident build — base bytes for
// the first slot (weights, host copy, one KV) plus perSlot for each further one — fits budget, and never below 1 (the
// first slot is priced, and refused if it does not fit, by residentMemoryDecline, not here).
func kvSlotsWithin(want int, budget, base, perSlot int64) int {
	n := max(1, want)
	for n > 1 && base+int64(n-1)*perSlot > budget {
		n--
	}
	return n
}

// residentMemoryDeclinePrefix opens every memory-guard decline, so a caller can tell "this machine does not have the
// memory right now" from a feature or admission decline (the resident parity tests skip on the first and fail on the
// second; skipIfMemoryDeclined).
const residentMemoryDeclinePrefix = "the resident build needs "

// residentMemoryDecline is residentFitsMemory with its reason — "" when the build fits — so BuildResident
// can hand it to the load path as a typed decline (decoder.DeclineResident) instead of printing it.
func residentMemoryDecline(m *decoder.Model) string {
	if modelKnob(m, "GOINFER_NO_RESIDENT_MEM_GUARD") != "" {
		return ""
	}
	need := residentNeedBytes(m)
	if need <= 0 {
		return "" // nothing to compare against; not a reason to refuse
	}
	ram, err := unix.SysctlUint64("hw.memsize")
	if err != nil || ram == 0 {
		return ""
	}
	// S4 item 3: the SAME combined ceiling the registered "metal" memory probe reports to
	// decoder.Model.Plan, so the two can never disagree.
	budget := metalMemoryCeiling(ram)
	if need <= budget {
		return ""
	}
	const gb = 1 << 30
	// M-13: name the actual escape hatch for an MoE model, not just the guard override — a model
	// with routed experts that doesn't fit resident may still fit PAGED, and the decline line
	// used to say nothing about how to reach that path.
	moeHint := ""
	if m.HasGemma4MoEResident() {
		moeHint = " This model routes MoE experts — try --moe-cache-experts (Metal pages them; auto-sizes the slot count from free RAM) or --moe-cache-slots N to pick one yourself."
	} else if _, _, _, _, _, _, _, _, _, _, ok := m.MoEResidentParams(); ok {
		moeHint = " This model routes MoE experts — try --moe-cache-experts (Metal pages them; auto-sizes the slot count from free RAM) or --moe-cache-slots N to pick one yourself."
	}
	tightened := ""
	if staticCeiling := metalStaticCeiling(ram); budget < staticCeiling {
		tightened = fmt.Sprintf(" (this machine's live-available memory tightened the static %.0f%%/%.2f GB ceiling further)",
			residentMemFraction*100, float64(staticCeiling)/gb)
	}
	return fmt.Sprintf(residentMemoryDeclinePrefix+"%.2f GB (weights, host copy and KV), over %.0f%% of %.1f GB RAM "+
		"(budget %.2f GB%s). Metal wires the pages it touches, so loading this would page to swap "+
		"exhaustion rather than run. Override with GOINFER_NO_RESIDENT_MEM_GUARD=1 if this machine really fits it.%s",
		float64(need)/gb, residentMemFraction*100, float64(ram)/gb, float64(budget)/gb, tightened, moeHint)
}

func (b *metalBackend) Close() error {
	if b.resident != nil {
		return b.resident.Close()
	}
	return nil
}

// metalResident adapts *resident (whose Forward takes a token id and returns logits, no error)
// to decoder.ResidentForward (Forward takes a precomputed embedding and returns logits+error).
type metalResident struct {
	r      *resident
	hidden int
	exact  bool // the model was loaded with Options.ExactPrefill: fast prefill off for THIS resident
	// poisonPrefillLogitsForTest makes PrefillLast write a NaN into its logits before the finite check, so a test can
	// show the check declines. False in production; only tests set it.
	poisonPrefillLogitsForTest bool
	// quant is the precision ResidentQuant reports: the model's own quant when it runs int8 weights natively (r.w8),
	// "" otherwise, which leaves the decoder's label (an int8 model on the int4 path is re-quantized, and says so).
	quant string

	// fast is the batched-prefill decision, made ONCE per resident on first use: the
	// GOINFER_METAL_FAST_PREFILL env var used to be re-read on every prefill call, so a change to
	// the process environment mid-serve silently switched a loaded model's kernels. Lazy (not set at
	// build) so a resident built as a struct literal in a test gets the same default a real one does.
	fastOnce sync.Once
	fast     bool
}

// fastPrefill reports whether this resident's prefill takes the batched fast path.
func (a *metalResident) fastPrefill() bool {
	a.fastOnce.Do(func() {
		a.fast = !a.exact && metalFastPrefillEnabled(a.r.knobValue("GOINFER_METAL_FAST_PREFILL"), a.r.knobValue("GOINFER_METAL_BATCHED_PREFILL"))
	})
	return a.fast
}

// ctxCap is this resident's resolved KV capacity — a.r.ctxCap when a real *resident exists, else
// metalCtxCapDefault. The fallback matters for TestMetalResidentCheckCap (metal/resident_cap_test.go),
// which deliberately constructs a zero-value &metalResident{} (r == nil) to test checkCap/ContextCap as
// pure logic with no Metal device — that test predates G6's per-build ctxCap and is meant to keep
// working unmodified against "the historical constant" semantics, so a nil/zero r reads as "no
// explicit request was ever resolved here", not as 0.
func (a *metalResident) ctxCap() int {
	if a.r == nil || a.r.ctxCap == 0 {
		return metalCtxCapDefault
	}
	return a.r.ctxCap
}

// checkCap guards the resident KV allocation (C3). Every layer's cache is r.kc[l]/r.vc[l], sized
// ctxCap()*kvDim, so kv_store writes absolute position p at kc[p*kvDim ...]; valid positions
// are [0, ctxCap()). Writing past it is an out-of-bounds device write — on Metal's UNIFIED
// memory that silently corrupts adjacent MTLBuffers (other models' resident weights), and once
// nKeys > 4096 the attention kernel's `threadgroup float sc[4096]` overflows too. The decode loop
// increments pos unbounded (a ≤cap prompt + a large max_tokens is enough), so refuse here; the
// decode loop surfaces the error (model.go) and the caller can fall back to the staged path.
func (a *metalResident) checkCap(pos, n int) error {
	c := a.ctxCap()
	if pos < 0 || pos+n > c {
		return fmt.Errorf("metal: KV position %d(+%d) exceeds resident context cap %d — use the staged path for longer contexts", pos, n, c)
	}
	return nil
}

// ContextCap is the resident KV capacity in positions. Implementing it makes metalResident satisfy
// decoder.ResidentCapped, so generateInto clamps maxTokens to the cap UP FRONT (stops cleanly at
// the cap instead of erroring mid-decode). Queryable so callers clamp rather than discover mid-run.
func (a *metalResident) ContextCap() int { return a.ctxCap() }

// ForwardMRoPE is decoder.ResidentMRoPE: like Forward, but the rotation angle (ropePos) and the
// KV-cache/attention position (pos) are supplied separately — Qwen2.5-VL decode past an image
// block needs them to differ. Forward itself is ForwardMRoPE(pos, pos).
func (a *metalResident) ForwardMRoPE(embedding []float32, pos, ropePos int) ([]float32, error) {
	if len(embedding) != a.embLen() {
		return nil, fmt.Errorf("metal: embedding len %d != %d (hidden %d + PLE inputs)", len(embedding), a.embLen(), a.hidden)
	}
	if e := a.checkCap(pos, 1); e != nil {
		return nil, e
	}
	if pos == 0 {
		// Fresh sequence: re-zero any Gated-DeltaNet state before it carries over from a prior
		// Generate on this resident (audit C-01's CUDA analogue). No-op for every other family.
		a.Reset()
	}
	logits := a.r.ForwardEmbMRoPEPipe(embedding, pos, ropePos)
	if err := a.r.takeExecErr(); err != nil {
		return nil, err // C-09: a command buffer aborted — surface it, do NOT return stale logits
	}
	return logits, nil
}

// Forward runs one token given its embedding[H] at absolute position pos, returning logits[V].
// The returned slice is reused across calls (the decode loop consumes it before the next call).
func (a *metalResident) Forward(embedding []float32, pos int) ([]float32, error) {
	return a.ForwardMRoPE(embedding, pos, pos)
}

var _ decoder.ResidentSample = (*metalResident)(nil)

// SampleAvailable/ForwardSample (decoder.ResidentSample, R7b Mac half — gumbel_sample.go) delegate
// straight to *resident: this wrapper adds no logic of its own here, but the methods must exist on
// *metalResident (not just *resident) because *decoder.Model.resident holds a *metalResident, and
// Go does not promote a NAMED field's (a.r) methods the way an embedded one would — see this
// struct's own field comment and every other delegating method below it.
func (a *metalResident) SampleAvailable() bool { return a.r != nil && a.r.SampleAvailable() }

func (a *metalResident) ForwardSample(embedding []float32, pos int, temperature float64, seed, draw uint64) (int, error) {
	if len(embedding) != a.embLen() {
		return 0, fmt.Errorf("metal: embedding len %d != %d (hidden %d + PLE inputs)", len(embedding), a.embLen(), a.hidden)
	}
	return a.r.ForwardSample(embedding, pos, temperature, seed, draw)
}

// embLen is every embedding row's length: hidden, or for a Gemma 4 E-model hidden + L·P PLE inputs (S1.5), the
// decoder's ResidentEmbedLen. Derived from the resident, so a test-built metalResident gets it right too.
func (a *metalResident) embLen() int {
	if a.r == nil {
		return a.hidden
	}
	return a.r.H + a.r.nL*a.r.pleP
}

// ForwardNoLogits (decoder.ResidentPrefillKV) runs the token's forward to build ONLY its
// resident K/V — skipping the final norm's LM-head dispatch, the ~1 MB logits readback, and any
// softcap. residentPrefillSeed calls this for every prompt token but the last (audit-
// metal-2026-09-12.md M-01): before this existed every sequential-prefill token on Metal paid the
// full int8 LM head + a 608 KB readback for logits nobody read (233 MB/token on a 1.5B, 671 MB on
// a Gemma-class vocabulary). The layer chain — hence the K/V written at pos — is identical to
// Forward, so decode from the last prompt token is byte-identical.
//
// Pipelined through the encode-ahead executor (ForwardEmbNoLogitsPipe) via a noHead bit on execJob:
// overlaps token t+1's trunk encode with token t's GPU execution while skipping the LM head
// dispatch and logits readback (~0.9 ms/token prefill latency recovery).
//
// On a paged MoE (g4moe or generic moe) the executor's trunk encoder has no paged branch, so
// ForwardEmbNoLogitsPipe runs the synchronous paged forward with its final norm and LM head skipped
// (pagedNoHead): the same layers and K/V, without the head (task-m26-mac-2026-10.md).
func (a *metalResident) ForwardNoLogits(embedding []float32, pos int) error {
	if len(embedding) != a.embLen() {
		return fmt.Errorf("metal: embedding len %d != %d (hidden %d + PLE inputs)", len(embedding), a.embLen(), a.hidden)
	}
	if e := a.checkCap(pos, 1); e != nil {
		return e
	}
	if pos == 0 {
		a.Reset() // fresh sequence — same reason as Forward
	}
	a.r.ForwardEmbNoLogitsPipe(embedding, pos)
	return a.r.takeExecErr() // C-09: a command buffer aborted — surface it
}

// metalFastPrefillFloor is the PROMPT-LENGTH floor (whole prompt = startPos+M) below which
// the f16-MMA batched pass declines and the sequential per-token loop runs instead. Lowered to
// 64 (2026-09-20, R3, docs/tasks/red-october.md and docs/measurements/metal-prefill-floor-
// 2026-09-20.md): the §3.2 pooled gate SHIPS at K=64 alone (critA/B/C all true, S model) and at
// K=64+128 pooled together, and the batched arm beats sequential on served TTFT by 3.83x at
// K=64 and 4.66x at K=128 (bench_peer_prefill.py, three arms interleaved, n=6) — both well past
// the >=2x ships band, not just the fidelity precondition. K=64 is the smaller of the two
// registered candidates {64, 128}, so it is the new floor per the brief's own decision rule.
// Previously lowered to 256 (2026-09-12, audit-metal-2026-09-12.md M-02).
//
// 16 since A-P02 (docs/tasks/task-metal-audit-2026-10.md, graded 2026-10-04 against its pre-registration): the §3.2 pooled
// gate SHIPS on the 1.5B with K = 16, 32, 48 and 64 pooled, and on each new cell alone, against CPU f32 references built
// together; the pass beats the sequential loop 3.48x at K = 16 in-process (7 of 7 reps above the 1.5 bar; the 7B 2.52x).
// GOINFER_METAL_FAST_PREFILL_FLOOR overrides it, 0 = no floor. A resident with the batched step keeps the exact step
// below metalStepPrefillCeiling instead (promptStepOK).
const metalFastPrefillFloor = 16

// metalStepPrefillCeiling (A-P02's step bound): a resident with the batched step runs a prompt that ends below this many
// tokens as exact decode rows on the step (E-P01) rather than the pass. The step is bit-identical to the sequential
// loop and the pass is not, so it keeps every length where it is not slower: the 1.5B measured step / pass 0.755 at
// K = 16 (the step faster in 7 of 7 reps) and 1.485 at K = 32 (the pass faster in 7 of 7), the 7B 0.627 and 1.243.
// Held at 64 for a morning (2026-10-04) on what was read as an MC3 identity defect and was a warm prompt taking a
// different route from the cold one (fixed: decoder.PrefillTailExact); released to the graded 32 the same day.
const metalStepPrefillCeiling = 32

// metalFastPrefillEnabled reports whether the batched f16-MMA prefill path is selected.
//
// Default ON above metalFastPrefillFloor (16 tokens since A-P02; 64 from R3) since §3.2 gate (TestPrefillGateVsReference)
// passed 2026-09-20 (S model, K=64/128 pooled and K=64 alone; see docs/measurements/metal-
// prefill-floor-2026-09-20.md). GOINFER_METAL_FAST_PREFILL=0/false/off or
// --exact-prefill to opt out.
//
//	GOINFER_METAL_FAST_PREFILL  1 | true | on   on (even below the floor — for tests)
//	                            0 | false | off  off (explicit opt-out; use --exact-prefill on the server)
//
// The old GOINFER_METAL_BATCHED_PREFILL continues to work: =1 forces on, =0 forces off, unset defers to the default.
func metalFastPrefillEnabled(fast, batched string) bool {
	switch strings.ToLower(strings.TrimSpace(fast)) {
	case "0", "false", "off":
		return false
	case "1", "true", "on":
		return true
	}
	// Unset: honour the old var for backward compat (=1 on, =0 off, unset → new default).
	if v := strings.ToLower(strings.TrimSpace(batched)); v != "" {
		return v == "1"
	}
	return true // §3.2 gate passed 2026-09-09 (S cells K=256/512/1024); floor lowered to 64 2026-09-20 (R3)
}

// metalAttnFAEnabled reports whether decode attention defaults to attention_fa (R2,
// docs/tasks/red-october.md) — the kvHead×split-gridded kernel, gated per layer by
// canUseAttnFA (dense-GQA, hd=128, key count >= attnFADepthFloor). DEFAULT ON since
// 2026-09-21. The fidelity pass that admitted it (gate (3), docs/measurements/r2-attn-fa-rootcause-2026-09-21.md)
// is VOID: its arms were contaminated by the executor's stale pre-encoded buffer and 4 of its 10 set-A references
// did not match their prompts. Re-gated 2026-09-25 on set B under the owner's amendment
// (docs/measurements/metal-decode-attn-fidelity-setb-PREREGISTERED.md): PASSES, with kernel error vs float64 ~3x below
// the shipped kernel's (docs/measurements/metal-decode-attn-r17-2026-09-25.md). Since 2026-09-25 the first pass is the
// R17 block kernel attention_fa_blk for GQA group sizes 6 and 7 (buildResident). The divergence that parked the Build
// attempt was an end-to-end-logits instrument crossing one int8 rounding boundary, not a kernel defect,
// and it is a real, deterministic 1.11-1.19x at depth (docs/measurements/r2-attn-fa-
// speed-2026-09-21.md) — KILLED on the brief's own peer-parity band (needs >=60 tok/s at
// depth 4000, measured 44.9), shipped anyway by owner decision as an incremental win. NOT
// bit-identical to the shipped `attention` kernel (reduction/combine order differs by
// design), so this moves argmax at the margin on some inputs (gate (3)'s own hard-flip
// counts) and is a SECOND source of decode/ForwardN divergence on top of the pre-existing
// one docs/spec/08-dspark-dflash.md already names (PrefillLast's f16-MMA activations) —
// Metal spec-decode verify (P10) was already not a legal oracle for that reason and stays
// so; nothing here newly breaks a path that was shippable before.
//
//	GOINFER_METAL_ATTN_FA  1 | true | on    on (explicit; harmless, matches the default)
//	                       0 | false | off  off (opt out; the shipped `attention` kernel only)
func metalAttnFAEnabled(v string) bool {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "0", "false", "off":
		return false
	case "1", "true", "on":
		return true
	}
	return true // default ON 2026-09-21 (R2) — see the doc comment above
}

// metalFastPrefillFloorFor returns the prompt-length floor, allowing experiment or escape.
// Set GOINFER_METAL_FAST_PREFILL_FLOOR to override; 0 disables the floor entirely.
func metalFastPrefillFloorFor(v string) int {
	if v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			return n
		}
	}
	return metalFastPrefillFloor
}

// metalFusedAttentionEnabled reports whether attention_prefill_fused (the simdgroup_matrix
// flash-attention twin of attention_prefill, L2-Metal — docs/completed/task-prefill-gap.md §4) runs in
// place of the exact scalar kernel. Default ON since §3 gate passed 2026-09-10 (S model, set B
// decision cells K=256/512/1024 + K=3900 confirm; fused beat exact on all three pooled criteria —
// see docs/measurements/prefill-l2-metal-fused-attn-2026-09-09.md §5). GOINFER_METAL_FUSED_ATTENTION=0
// or --exact-prefill (which also covers metalFastPrefillEnabled) opts back to the exact kernel.
// attention_prefill_fused also requires hd%8==0 && hd<=128 (ATTN_MAXHD in metal/prefill.go) —
// PrefillLast falls back to the exact kernel outside that range regardless of this flag.
//
//	GOINFER_METAL_FUSED_ATTENTION  1 | true | on   on
//	                               0 | false | off  off (explicit opt-out; use --exact-prefill on the server)
func metalFusedAttentionEnabled(v string) bool {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "0", "false", "off":
		return false
	case "1", "true", "on":
		return true
	}
	return true // §3 gate passed 2026-09-10 (S set B decision cells K=256/512/1024 + K=3900 confirm)
}

// PrefillPath (decoder.PrefillPathReporter) reports at load time whether this resident will use
// the batched f16-MMA path. Default ON above metalFastPrefillFloor (16 tokens since A-P02, 64 from R3) since §3.2
// gate passed 2026-09-20. The floor applies per-call; PrefillPath reports true iff the enabled
// state AND arch both allow batching.
// ResidentQuant (decoder.ResidentQuantReporter) is the precision this resident runs its weights at, when the decoder
// cannot infer it: an int8 model on the native path (r.w8) runs at its own quant, where one on the int4 path is
// re-quantized. "" leaves the decoder's label.
func (a *metalResident) ResidentQuant() string { return a.quant }

// PrefillKernelFloor (decoder.ResidentPrefillKernelFloor) is the whole-prompt length from which the batched
// prefill runs the f16 MMA kernels, below which it runs decode rows or the sequential loop: two kernel classes
// that are not bit-identical. 0 when one class runs at every length (a layer-major path, int8 KV, an arch the
// MMA kernels do not cover, fast prefill off, or GOINFER_METAL_FAST_PREFILL_FLOOR=0). PrefillPath reports the same
// number; both read prefillKernelFloor.
func (a *metalResident) PrefillKernelFloor() int { return a.prefillKernelFloor() }

func (a *metalResident) prefillKernelFloor() int {
	if a.g4LayerMajor() || a.moeLayerMajor() || a.emodelLayerMajor() || a.r.kvI8 || a.r.attnSink || !a.r.prefillOK || !a.fastPrefill() {
		return 0
	}
	floor := metalFastPrefillFloorFor(a.r.knobValue("GOINFER_METAL_FAST_PREFILL_FLOOR"))
	if floor > 0 && a.VerifyCost() != nil && !a.r.promptStepOff {
		return max(floor, metalStepPrefillCeiling)
	}
	return floor
}

func (a *metalResident) PrefillPath() (bool, string) {
	if a.g4LayerMajor() || a.moeLayerMajor() {
		return true, "layer-major on decode's kernels (bit-identical to sequential; a paged MoE)"
	}
	if a.emodelLayerMajor() {
		if a.emodelBatched() && a.fastPrefill() {
			floor := metalFastPrefillFloorFor(a.r.knobValue("GOINFER_METAL_FAST_PREFILL_FLOOR"))
			return true, fmt.Sprintf("batched f16-MMA from %d prompt tokens (a Gemma 4 E-model, S9 step 2); layer-major on decode's kernels below it (bit-identical to sequential)", floor)
		}
		return true, "layer-major on decode's kernels (bit-identical to sequential; a Gemma 4 E-model, S9)"
	}
	if a.r.kvI8 {
		return false, "sequential — the f16 MMA prefill kernels write half-precision K/V, and this model's KV cache is int8 (-kv i8)"
	}
	if a.r.attnSink {
		return false, "sequential — the batched prefill kernels implement neither gpt-oss's attention sink nor its clamped SwiGLU with biases"
	}
	if !a.r.prefillOK {
		return false, "sequential — arch/geometry not supported by f16 MMA prefill kernel"
	}
	if !a.fastPrefill() {
		return false, "sequential — fast prefill disabled (GOINFER_METAL_FAST_PREFILL=0 or --exact-prefill)"
	}
	floor := metalFastPrefillFloorFor(a.r.knobValue("GOINFER_METAL_FAST_PREFILL_FLOOR"))
	if floor > 0 && a.VerifyCost() != nil && !a.r.promptStepOff {
		return true, fmt.Sprintf("batched f16-MMA from %d prompt tokens; below it, decode rows on the batched step kernels (bit-identical to sequential, E-P01)", a.prefillKernelFloor())
	}
	if floor > 0 {
		return true, fmt.Sprintf("batched f16-MMA above %d prompt tokens; sequential below (§3 floor)", a.prefillKernelFloor())
	}
	return true, "batched f16-MMA (GOINFER_METAL_FAST_PREFILL_FLOOR=0; §3.2 gate passed 2026-09-09)"
}

// g4LayerMajor reports whether this resident's prompts take prefillG4Paged: a paged Gemma 4 MoE with no learned
// positions (the layer-major rows add none), with g4LayerMajorOn.
func (a *metalResident) g4LayerMajor() bool {
	return g4LayerMajorOn && a.r.g4moe != nil && a.r.g4moe.paged && !a.r.learnedPos
}

// moeLayerMajor is g4LayerMajor for the generic pager (prefillMoEPaged): a paged generic MoE, not gpt-oss (its sink and
// biased experts), with moeLayerMajorOn.
func (a *metalResident) moeLayerMajor() bool {
	return moeLayerMajorOn && a.r.moe != nil && a.r.moe.paged && !a.r.moe.isGptOss && !a.r.learnedPos
}

// PrefillLast (decoder.Prefiller) ingests the whole prompt in one batched f16-MMA pass and
// returns the last token's logits, populating the resident KV. Falls back (declines) for prompts
// shorter than the fast-prefill floor or longer than the resident KV/attention cap.
func (a *metalResident) PrefillLast(ctx context.Context, embeddings [][]float32, startPos int) ([]float32, error) {
	// One pass, so one check: this backend ingests the prompt in a single command buffer and has no
	// inner loop to interrupt, so checking at entry is the only granularity there is. The decoder's
	// chunked prefill (serve -prefill-chunk) cuts a prompt only while another conversation is
	// decoding, so a lone long prompt is still one pass that cannot be cancelled midway; that needs
	// chunking inside PrefillLast, as cuda's prefillChunked does (A-D02, audit-metal-2026-09-30.md).
	if e := ctx.Err(); e != nil {
		return nil, e
	}
	// 4b (docs/tasks/task-m26-mac-2026-10.md): a paged Gemma 4 MoE (M26) takes no batched pass; its prompt runs layer by
	// layer on decode's own kernels, bit-identical to the sequential loop, so neither the floor nor --exact-prefill
	// applies. Behind g4LayerMajorOn until graded.
	// S9 step 2: an E-model prompt at or above the floor takes the batched pass when it is on; below the floor, or on any
	// decline, the layer-major pass below runs it as before.
	if a.emodelBatched() && a.fastPrefill() && startPos+len(embeddings) >= metalFastPrefillFloorFor(a.r.knobValue("GOINFER_METAL_FAST_PREFILL_FLOOR")) {
		ok := true
		for _, e := range embeddings {
			ok = ok && len(e) == a.embLen()
		}
		if ok {
			if lg, err := a.batchedPrefill(embeddings, startPos, 0, nil, nil); err == nil {
				return lg, nil
			}
		}
	}
	if a.g4LayerMajor() || a.moeLayerMajor() || a.emodelLayerMajor() {
		if e := a.checkCap(startPos, len(embeddings)); e != nil {
			return nil, e
		}
		if startPos == 0 {
			a.Reset() // a fresh sequence, as Forward at position 0 does (it zeroes a DeltaNet's state)
		}
		var lg []float32
		if a.emodelLayerMajor() { // S9: a Gemma 4 E-model, layer by layer on decode's kernels (prefill_emodel.go)
			for _, e := range embeddings {
				if len(e) != a.embLen() {
					return nil, fmt.Errorf("metal: an E-model prefill row of %d values, want %d ([h | L*P])", len(e), a.embLen())
				}
			}
			lg = a.r.prefillEModel(embeddings, startPos, true)
		} else if a.g4LayerMajor() {
			lg = a.r.prefillG4Paged(embeddings, startPos, true)
		} else {
			lg = a.r.prefillMoEPaged(embeddings, startPos, true)
		}
		if err := a.r.takeExecErr(); err != nil {
			return nil, err
		}
		return lg, nil
	}
	// Default on since the §3.2 gate passed on 2026-09-09 (S cells K=256/512/1024), above metalFastPrefillFloor: 16
	// tokens since A-P02 (cells K=16..64), 64 from R3, 256 before. GOINFER_METAL_FAST_PREFILL=0 or --exact-prefill opts
	// out.
	if !a.fastPrefill() {
		return nil, fmt.Errorf("metal: fast prefill disabled (GOINFER_METAL_FAST_PREFILL=0 / --exact-prefill / GOINFER_METAL_BATCHED_PREFILL=0); using sequential path")
	}
	// FLOOR: below metalFastPrefillFloor no decision cell has passed, so the batched pass declines there; the lowest that
	// passed is K=16 (the floor's own doc comment). E-P01: a resident with the step kernels runs a prompt that ends below
	// max(floor, metalStepPrefillCeiling) as decode rows on those kernels instead: the sequential loop's bits, faster
	// than the pass there (promptStepOK).
	promptLen := startPos + len(embeddings)
	floor := metalFastPrefillFloorFor(a.r.knobValue("GOINFER_METAL_FAST_PREFILL_FLOOR"))
	if a.promptStepOK(len(embeddings), startPos, floor) {
		if e := a.checkCap(startPos, len(embeddings)); e != nil {
			return nil, e
		}
		return a.prefillByStep(ctx, embeddings, startPos)
	}
	// One token whose whole prompt ends in the step's range (a repeated prompt's 1-token suffix, PrefillTailExact): the
	// step needs two rows, and the pass would not compute the step's bits, which the cold prompt of this length ran on.
	// The sequential path's decode does.
	if len(embeddings) < 2 && a.inStepRange(promptLen, floor) {
		return nil, fmt.Errorf("metal: a 1-token suffix in the batched step's range runs on the sequential path (the step's bits)")
	}
	return a.batchedPrefill(embeddings, startPos, floor, nil, nil)
}

// batchedPrefill is the f16 batched pass behind PrefillLast and PrefillMRoPELast (S16): the same declines (the floor, an
// int8 KV cache, a family the pass does not implement, the resident's cap, the exact attention kernel's key limit), the
// same recovery of a request-time panic, and the same non-finite-logit check. mrope is nil for a text prompt.
func (a *metalResident) batchedPrefill(embeddings [][]float32, startPos, floor int, mrope [][3]int, deep *prefillDeep) ([]float32, error) {
	promptLen := startPos + len(embeddings)
	if floor > 0 && promptLen < floor {
		return nil, fmt.Errorf("metal: prompt too short (%d tokens) for fast prefill (floor=%d; §3 floor); using sequential path", promptLen, floor)
	}
	// The f16 MMA prefill kernels implement a dense gated FFN (SiLU or GeGLU, G8) out of
	// L.guW/L.dW with per-layer rope/window, per-head QK-norm, and Gemma's sandwich norms, and
	// (G8 MoE half) a generically-shaped gated-SwiGLU MoE FFN — expert-major by default, the
	// per-token decode MoE chain row by row with GOINFER_MOE_EXPERT_MAJOR=0 (prefill.go) — but
	// NOT per-layer-varying attention geometry (dense Gemma 4's local/global head_dim split;
	// prefillOK's own per-layer-geometry guard, metal/model.go) or Gemma 4's enable_moe_block
	// variant (residLayer.g4moe, a third FFN shape this path never reads; explicitly excluded
	// via HasGemma4MoEResident regardless of per-layer geometry). Any of these declines here,
	// and the caller re-runs the prompt through the (correct) sequential Forward loop.
	if a.r.kvI8 {
		return nil, fmt.Errorf("metal: prefill writes half-precision K/V and this model's KV cache is int8 (-kv i8); using sequential path")
	}
	if !a.r.prefillOK && !a.emodelBatched() {
		return nil, fmt.Errorf("metal: prefill not implemented for this arch's FFN shape (use the sequential path)")
	}
	// startPos < 0 would wrap to a huge uint32 and make kv_store_f16 write far out of bounds — on
	// UMA that silently corrupts adjacent buffers (audit R-27). The decoder never passes one (it passes 0,
	// a reused prefix's length, or a chunk's offset; F-G01), but the guard is cheap.
	if startPos < 0 || len(embeddings) == 0 || startPos+len(embeddings) > a.ctxCap() {
		return nil, fmt.Errorf("metal: prompt len %d at startPos %d out of resident cap %d", len(embeddings), startPos, a.ctxCap())
	}
	// F-C02: the exact attention kernel (no fused kernel for this head dim, or fused attention off) holds at most
	// prefillExactAttnMaxKeys scores; the sequential path's decode kernels tile theirs.
	if a.r.prefillExactAttn() && startPos+len(embeddings) > prefillExactAttnMaxKeys {
		return nil, fmt.Errorf("metal: prompt reaches %d keys and the exact prefill attention kernel holds %d (head dim %d has no fused kernel, or it is off); using sequential path",
			startPos+len(embeddings), prefillExactAttnMaxKeys, a.r.prefillGeom().hd)
	}
	// A fresh sequence (startPos 0) starts a Gated-DeltaNet hybrid's recurrent state from zero, as Forward(pos 0) does: the
	// pass continues whatever conv window and state the resident holds (D-B01), which for a continuation is the point and
	// for a new prompt would be the previous sequence's. A no-op for every other family.
	if startPos == 0 {
		a.Reset()
	}
	// ensurePrefill's compile panic and the ~24 per-call MustBuf OOM panics fire HERE, at request
	// time, with no recover of their own (buildResident's is build-scoped). A transient OOM would kill
	// the server; recover into an error so the request fails and the caller falls back to sequential
	// decode (audit R-23; B-10 class).
	var logits []float32
	if err := func() (err error) {
		defer func() {
			if p := recover(); p != nil {
				err = fmt.Errorf("metal: batched prefill aborted: %v", p)
			}
		}()
		logits = a.r.prefillLast(embeddings, startPos, mrope, deep)
		return nil
	}(); err != nil {
		return nil, err
	}
	if err := a.r.takeExecErr(); err != nil {
		return nil, err // C-09
	}
	if a.poisonPrefillLogitsForTest {
		logits[0] = float32(math.NaN())
	}
	// A-C02 (docs/audit-metal-2026-09-30.md): the batched pass carries f16 activations, which a checkpoint the
	// graded models never exercised could overflow; the sequential decode kernels carry f32. A non-finite logit
	// declines, so the decoder re-runs the prompt sequentially, which rewrites every K/V row this pass wrote. The
	// scan is about 0.05 ms at a 152k vocabulary.
	if i := firstNonFinite(logits); i >= 0 {
		return nil, fmt.Errorf("metal: batched prefill produced a non-finite logit (%v at %d); using sequential path", logits[i], i)
	}
	return logits, nil
}

// PrefillMRoPELast (decoder.ResidentMRoPEPrefill, S16 of docs/tasks/task-multimodal-support-2026-10.md) is PrefillLast for a
// Qwen image turn: the batched f16 pass with every row rotated by its own (temporal, height, width) m-RoPE triple, so the
// image turn prefills here instead of on the CPU with its K/V uploaded after. mropePos covers the whole prompt from 0. It
// declines (the decoder then takes the CPU-prefill+UploadKV bridge) when the model has no m-RoPE axis table, when a
// layer-major route owns this family, or when the batched pass itself would; the batched step never takes it (its kernels
// rotate by one scalar position).
func (a *metalResident) PrefillMRoPELast(ctx context.Context, embeddings [][]float32, startPos int, mropePos [][3]int) ([]float32, error) {
	return a.prefillMRoPE(ctx, embeddings, startPos, mropePos, nil)
}

// PrefillMRoPEDeepstackLast (decoder.ResidentMRoPEDeepstackPrefill, S16) is PrefillMRoPELast for Qwen3-VL: the same pass,
// which also adds DeepStack set l to the image rows after decoder layer l. Each set must be imgLen rows of the hidden size.
func (a *metalResident) PrefillMRoPEDeepstackLast(ctx context.Context, embeddings [][]float32, startPos int, mropePos [][3]int, deep [][]float32, imgStart, imgLen int) ([]float32, error) {
	if imgStart < 0 || imgLen <= 0 || len(deep) > a.r.nL {
		return nil, fmt.Errorf("metal: DeepStack prefill of %d sets over [%d,%d) for %d layers", len(deep), imgStart, imgStart+imgLen, a.r.nL)
	}
	for l, set := range deep {
		if len(set) != imgLen*a.r.H {
			return nil, fmt.Errorf("metal: DeepStack set %d has %d values, want %d x %d", l, len(set), imgLen, a.r.H)
		}
	}
	return a.prefillMRoPE(ctx, embeddings, startPos, mropePos, &prefillDeep{start: imgStart, n: imgLen, sets: deep})
}

// metalMRoPEPrefillOn and metalDeepstackPrefillOn turn S16's resident m-RoPE prefill on, for a turn without DeepStack sets
// (Qwen2.5-VL; Qwen3.5+ is not claimed) and with them (Qwen3-VL). Off, an image turn takes the CPU prefill and the upload.
//   - DeepStack (Qwen3-VL): ON since 2026-10-09 (owner). G-S16c real PASS through its sidecar on all four images over healthy
//     text controls (0.87-0.95), the served comparison IDENTICAL, image-turn TTFT 8.25 -> 2.86 s (2.9x).
//   - Without DeepStack (Qwen2.5-VL): ON since 2026-10-09 (owner). G-S16c and the served comparison passed (19.46 -> 5.73
//     s). It was held for a text control that fell to -0.36, but that control's reference was W4A8, the less precise arm
//     (docs/completed/task-metal-prefill-precision-2026-10.md). Against non-W4A8 references the image turns read even
//     with the CPU prefill (mean 0.948 against weight-only int8) and closer to the same weights (0.990 against 0.977).
//
// Tests set them.
var metalMRoPEPrefillOn, metalDeepstackPrefillOn = true, true

func (a *metalResident) prefillMRoPE(ctx context.Context, embeddings [][]float32, startPos int, mropePos [][3]int, deep *prefillDeep) ([]float32, error) {
	if e := ctx.Err(); e != nil {
		return nil, e
	}
	if on := metalMRoPEPrefillOn; deep != nil && !metalDeepstackPrefillOn || deep == nil && !on {
		return nil, fmt.Errorf("metal: the resident m-RoPE prefill is off for this model until its S16 gates are graded; the image prefill runs on the CPU")
	}
	if a.r.mropeAxis == (Buffer{}) {
		return nil, fmt.Errorf("metal: no m-RoPE axis table for this model; the image prefill runs on the CPU")
	}
	if len(embeddings) == 0 || startPos < 0 || len(mropePos) != startPos+len(embeddings) {
		return nil, fmt.Errorf("metal: m-RoPE prefill of %d rows at %d with %d positions (want startPos+rows)", len(embeddings), startPos, len(mropePos))
	}
	if a.g4LayerMajor() || a.moeLayerMajor() || a.emodelLayerMajor() {
		return nil, fmt.Errorf("metal: a layer-major prefill family has no m-RoPE pass")
	}
	if !a.fastPrefill() {
		return nil, fmt.Errorf("metal: fast prefill disabled (GOINFER_METAL_FAST_PREFILL=0 / --exact-prefill); the image prefill runs on the CPU")
	}
	return a.batchedPrefill(embeddings, startPos, metalFastPrefillFloorFor(a.r.knobValue("GOINFER_METAL_FAST_PREFILL_FLOOR")), mropePos, deep)
}

// firstNonFinite is the index of the first NaN or ±Inf in v, or -1.
func firstNonFinite(v []float32) int {
	for i, x := range v {
		if math.Float32bits(x)&0x7f800000 == 0x7f800000 {
			return i
		}
	}
	return -1
}

// HiddenLast (decoder.ResidentHiddenLast) ingests a whole sequence starting at startPos and
// returns the LAST position's hidden state after the model's final norm — the resident twin of
// PrefillLast, but for embedding requests (G4, docs/tasks/task-gpu-paths-2026-09.md) instead of
// generation: it never runs the LM head. This runs the SAME per-token sequential kernels decode
// uses — one forwardHiddenNoHead call per position. It does not match the CPU reference bit for
// bit: the decode kernels round differently and forwardHiddenNoHead returns the int8-dequantized
// activation. Measured against the CPU: cosine about 0.9991-0.9993 on gpt2
// (hiddenlast_resident_parity_test.go, bar 0.998), and 0.99985 int4 against int4 on qwen3_5-tiny
// with its adapter merged, 0.99998 without (prompthidden_resident_parity_test.go; F-D03,
// audit-metal-2026-09-30.md). The cost is one
// command-buffer submit per token (≈K × 13-18ms) instead of a single batched pass (≈1.8s for K=512).
//
// N-25 (audit-metal-2026-09-12.md): unlike when this doc comment was first written, Metal's
// batched (f16-MMA) PrefillLast is NOT declined by default for generation anymore —
// metalFastPrefillEnabled() defaults true (§3.2 gate passed 2026-09-09) — so the fidelity bar
// that justifies this sequential loop's cost for embeddings is already accepted for decode's own
// output. The audit's suggested fix (PrefillLast's dispatch graph minus its last two dispatches —
// the LM head + softcap — reused here) is a real, scoped lever, not a design question, but it is
// its own parity-gated engineering task (a HiddenLast-batched path needs the same kind of S-cell
// tolerance gate PrefillLast itself passed, verified against THIS function as the oracle) rather
// than a same-sitting fix; left as follow-up work.
func (a *metalResident) HiddenLast(ctx context.Context, embeddings [][]float32, startPos int) ([]float32, error) {
	if len(embeddings) == 0 {
		return nil, fmt.Errorf("metal: HiddenLast called with no embeddings")
	}
	// forwardHiddenNoHead's trunk encoder (encodeTrunkInto → encodeLayer) has no paged branch — only
	// Forward's dispatch to forwardLogitsPaged/forwardLogitsMoEPaged does — so on a paged MoE it would
	// bind the zero-value stacked expert buffers and return a finite garbage hidden state with no
	// error (audit-metal-2026-09-12.md C-02). Decline instead: decoder.Model.HiddenLast treats a
	// resident decline as "no resident backend for this request" and falls through to the CPU path,
	// exactly like an OOM or cap decline does (decoder/embed.go:81-83).
	if (a.r.g4moe != nil && a.r.g4moe.paged) || (a.r.moe != nil && a.r.moe.paged) {
		return nil, fmt.Errorf("metal: HiddenLast not implemented for paged MoE (no headless paged forward); use the CPU path")
	}
	if e := a.checkCap(startPos, len(embeddings)); e != nil {
		return nil, e
	}
	if startPos == 0 {
		a.Reset() // fresh sequence — same DeltaNet-state reset Forward(pos==0) does
	}
	var out []float32
	for i, emb := range embeddings {
		if len(emb) != a.embLen() {
			return nil, fmt.Errorf("metal: embedding[%d] len %d != %d (hidden %d + PLE inputs)", i, len(emb), a.embLen(), a.hidden)
		}
		// G18: an abandoned client otherwise leaves the whole sequence streaming through the
		// device with nothing watching — same discipline as residentPrefillSeed's sequential loop.
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		want := i == len(embeddings)-1
		h, err := a.r.forwardHiddenNoHead(emb, startPos+i, want)
		if err != nil {
			return nil, err
		}
		if err := a.r.takeExecErr(); err != nil {
			return nil, err // C-09: a command buffer aborted — surface it, do NOT return a stale/zero vector
		}
		if want {
			out = h
		}
	}
	return out, nil
}

var _ decoder.ResidentResidualAll = (*metalResident)(nil)

// ResidualAll (decoder.ResidentResidualAll) ingests a whole sequence starting at startPos and returns EVERY position's
// residual stream after the last layer and BEFORE the final norm, each row a fresh f32 slice. The decoder applies the
// final norm on the host in f32 (decoder/prompt_hidden_all.go residualAllResident), so the rows never pass through the
// int8 the head path reads: Metal fuses the final norm into the trunk (encodeTrunkWith: layers, then encodeNorm into
// r.aq/r.aSc), and r.aq/r.aSc is that int8 vector, which is exactly what this exists to avoid.
//
// Where the pre-norm residual lives: every layer writes its residual into r.x (encodeLayerWith passes r.x as the
// residual buffer), and the final norm reads r.x through rmsnorm_quant / layernorm_quant, whose x is
// `device const float*`. So once the trunk has run, r.x holds the last layer's output untouched by the norm, and
// this copies it out per token.
//
// The same per-token sequential kernels HiddenLast uses: one forwardHiddenNoHead (one command-buffer submit) per
// position, so it costs what HiddenLast costs (≈ K × 13-18 ms) plus an H-float copy per row. A batched version
// (PrefillLast's dispatch graph collecting the residual block) is the speed lever and its own parity-gated job. Guards
// mirror HiddenLast's exactly; the rows are returned only once every token has succeeded.
func (a *metalResident) ResidualAll(ctx context.Context, embeddings [][]float32, startPos int) ([][]float32, error) {
	if len(embeddings) == 0 {
		return nil, fmt.Errorf("metal: ResidualAll called with no embeddings")
	}
	// The trunk encoder has no paged branch (see HiddenLast): on a paged MoE it would read zero-value expert buffers
	// and return finite garbage. Decline, and the decoder falls back to the CPU.
	if (a.r.g4moe != nil && a.r.g4moe.paged) || (a.r.moe != nil && a.r.moe.paged) {
		return nil, fmt.Errorf("metal: ResidualAll not implemented for paged MoE (no headless paged forward); use the CPU path")
	}
	if e := a.checkCap(startPos, len(embeddings)); e != nil {
		return nil, e
	}
	if startPos == 0 {
		a.Reset() // fresh sequence: the same DeltaNet-state reset Forward(pos==0) does
	}
	out := make([][]float32, 0, len(embeddings))
	for i, emb := range embeddings {
		if len(emb) != a.embLen() {
			return nil, fmt.Errorf("metal: embedding[%d] len %d != %d (hidden %d + PLE inputs)", i, len(emb), a.embLen(), a.hidden)
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if _, err := a.r.forwardHiddenNoHead(emb, startPos+i, false); err != nil {
			return nil, err
		}
		if err := a.r.takeExecErr(); err != nil {
			return nil, err // C-09: a command buffer aborted; never return a stale residual
		}
		out = append(out, append([]float32(nil), a.r.x.Floats()[:a.r.H]...))
	}
	return out, nil
}

// ForwardN runs a batch of embeddings at consecutive positions (prefill/verify).
// It sequences all N token forward steps inside a SINGLE Metal command buffer
// and single compute encoder (one commit, one wait), returning all N logits vectors.
// For paged MoE (which requires mid-layer host interaction), it falls back to the
// sequential loop.
func (a *metalResident) ForwardN(embeddings [][]float32, startPos int) ([][]float32, error) {
	// Fail-fast before any write: checking the whole batch up front refuses an over-cap run
	// without partial KV writes.
	if e := a.checkCap(startPos, len(embeddings)); e != nil {
		return nil, e
	}
	if len(embeddings) == 0 {
		return nil, nil
	}
	// Fall back to sequential loop for paged MoE where per-layer host interaction is required, and for a Gemma 4
	// E-model, whose PLE inputs ForwardBatch's layer-major encode has no per-row buffer for (S1.5).
	if (a.r.g4moe != nil && a.r.g4moe.paged) || (a.r.moe != nil && a.r.moe.paged) || a.r.pleP > 0 {
		out := make([][]float32, len(embeddings))
		for i, emb := range embeddings {
			l, err := a.Forward(emb, startPos+i)
			if err != nil {
				return nil, err
			}
			out[i] = append([]float32(nil), l...)
		}
		return out, nil
	}
	if startPos == 0 {
		a.Reset() // fresh sequence — same DeltaNet-state reset Forward(pos==0) does
	}
	out, err := a.r.ForwardBatch(embeddings, startPos)
	if err != nil {
		return nil, err
	}
	if err := a.r.takeExecErr(); err != nil {
		return nil, err // C-09: a command buffer aborted — surface it
	}
	return out, nil
}

// VerifyPath (decoder.VerifyPathReporter) reports whether this resident's ForwardN executes
// a single-command-buffer batched pass or falls back to a sequential loop.
func (a *metalResident) VerifyPath() (bool, string) {
	if (a.r.g4moe != nil && a.r.g4moe.paged) || (a.r.moe != nil && a.r.moe.paged) {
		return false, "sequential — paged MoE requires per-layer host staging"
	}
	return true, "batched layer-major single-command-buffer"
}

// SetAdapter implements decoder.ResidentAdapter (G3, docs/tasks/task-gpu-paths-2026-09.md) —
// generateInto calls this to bind/clear a compute-time LoRA adapter for an admitted session.
func (a *metalResident) SetAdapter(layers []decoder.ResidentAdapterLayer) error {
	return a.r.SetAdapter(layers)
}

// UploadKV writes a layer's post-RoPE K and raw V into the resident caches at absolute
// position base..base+n-1, where n = len(keys)/kvDim. Used by decoder.residentUploadPrefill
// to push a CPU-computed prefill's KV into the resident GPU cache, letting GenerateVL and
// GenerateQwenVL run text decode on resident GPU rather than dropping to CPU.
func (a *metalResident) UploadKV(layer, base int, keys, vals []float32) error {
	if a.r == nil {
		return fmt.Errorf("metal: resident is nil")
	}
	if layer < 0 || layer >= len(a.r.layers) {
		return fmt.Errorf("metal: UploadKV layer %d out of range", layer)
	}
	if a.r.kc[layer] == (Buffer{}) {
		return nil // recurrent DeltaNet layer with no attention KV cache
	}
	if a.r.layers[layer].kvShared {
		// S1.4: its kc/vc alias the source layer's cache; writing here would overwrite the source's rows.
		return fmt.Errorf("metal: UploadKV layer %d owns no KV (it reads layer %d's)", layer, a.r.layers[layer].kvSrc)
	}
	g := a.r.layers[layer].geom
	if g == nil {
		return fmt.Errorf("metal: layer %d has no attention geometry", layer)
	}
	kvDim := g.kvDim
	if kvDim == 0 {
		return fmt.Errorf("metal: layer %d has zero kvDim", layer)
	}
	if len(keys) != len(vals) || len(keys)%kvDim != 0 {
		return fmt.Errorf("metal: UploadKV len(keys)=%d len(vals)=%d not aligned to kvDim=%d", len(keys), len(vals), kvDim)
	}
	n := len(keys) / kvDim
	if e := a.checkCap(base, n); e != nil {
		return e
	}
	off := base * kvDim
	if a.r.kvI8 {
		nKV := g.nKV
		hd := g.hd
		kc := a.r.kc[layer].Int8s()
		vc := a.r.vc[layer].Int8s()
		ks := a.r.ks[layer].Floats()
		vs := a.r.vs[layer].Floats()
		for p := range n {
			pos := base + p
			for h := range nKV {
				kHead := keys[p*kvDim+h*hd : p*kvDim+(h+1)*hd]
				vHead := vals[p*kvDim+h*hd : p*kvDim+(h+1)*hd]
				var amaxK, amaxV float32
				for _, v := range kHead {
					if a := float32(math.Abs(float64(v))); a > amaxK {
						amaxK = a
					}
				}
				for _, v := range vHead {
					if a := float32(math.Abs(float64(v))); a > amaxV {
						amaxV = a
					}
				}
				scK := amaxK / 127.0
				if scK == 0 {
					scK = 1.0
				}
				scV := amaxV / 127.0
				if scV == 0 {
					scV = 1.0
				}
				ks[pos*nKV+h] = scK
				vs[pos*nKV+h] = scV
				invK := 1.0 / scK
				invV := 1.0 / scV
				for d := range hd {
					kc[pos*kvDim+h*hd+d] = int8(math.Round(float64(kHead[d] * invK)))
					vc[pos*kvDim+h*hd+d] = int8(math.Round(float64(vHead[d] * invV)))
				}
			}
		}
	} else if a.r.kvF32 {
		off += a.r.kvHostOff(layer, 4) // the bound slot's start, when the slots are one allocation (MC3 S3)
		copy(a.r.kc[layer].Floats()[off:off+len(keys)], keys)
		copy(a.r.vc[layer].Floats()[off:off+len(vals)], vals)
	} else {
		off += a.r.kvHostOff(layer, 2)
		kc := a.r.kc[layer].U16s()[off : off+len(keys)]
		vc := a.r.vc[layer].U16s()[off : off+len(vals)]
		parallelF32ToF16(kc, keys)
		parallelF32ToF16(vc, vals)
	}
	return nil
}

// TruncateTo is a no-op: KV positions are overwritten on write, and attention only reads
// keys[0..pos], so stale positions past the current one are never observed.
// KVSlots / UseKVSlot implement decoder.ResidentKVSlotter (MC1): the slots metalKVSlots allocated, and binding one.
func (a *metalResident) KVSlots() int          { return a.r.kvSlotCount() }
func (a *metalResident) UseKVSlot(i int) error { return a.r.useKVSlot(i) }

func (a *metalResident) TruncateTo(pos int) {}

// Reset zeroes every Gated-DeltaNet layer's causal-conv ring and recurrent matrix state (no-op
// for every other family — a.r.dnet is nil). Unlike KV positions, this state COMPOUNDS: a fresh
// Generate on the same resident without this would continue decaying stale state from the PRIOR
// sequence (audit C-01's CUDA analogue). See resetDeltaNet (deltanet.go).
func (a *metalResident) Reset() { a.r.resetDeltaNet() }

var _ decoder.ResidentGreedyChain = (*metalResident)(nil)

// GreedyChainAvailable (decoder.ResidentGreedyChain, C-B01): the greedy chain is exact on this resident
// (greedyChainWhyNot, greedy_chain.go) and its gather table is on the device, or fits there now (chainEmbedTable).
func (a *metalResident) GreedyChainAvailable() bool {
	if a.r.greedyChainWhyNot() != "" {
		return false
	}
	_, _, why := a.r.chainEmbedTable()
	return why == ""
}

// GreedyChainStart opens the chain at token id, position pos.
func (a *metalResident) GreedyChainStart(id, pos int) error {
	if e := a.checkCap(pos, 1); e != nil {
		return e
	}
	return a.r.chainStart(id, pos)
}

// GreedyChainNext returns the next greedy token.
func (a *metalResident) GreedyChainNext() (int, error) { return a.r.chainNext() }

// GreedyChainStop closes the chain.
func (a *metalResident) GreedyChainStop() { a.r.stopChain() }

var _ decoder.ResidentSampleChain = (*metalResident)(nil)

// SampleChainAvailable (decoder.ResidentSampleChain, C-P02): the chain is available and so is the device draw.
func (a *metalResident) SampleChainAvailable() bool {
	return a.r.SampleAvailable() && a.GreedyChainAvailable()
}

// SampleChainStart opens a sampled chain at token id, position pos, drawing that forward with (seed, draw).
func (a *metalResident) SampleChainStart(id, pos int, temperature float64, seed, draw uint64) error {
	if e := a.checkCap(pos, 1); e != nil {
		return e
	}
	return a.r.chainStartSampled(id, pos, temperature, seed, draw)
}

// SampleChainNext returns the next sampled token; (seed, draw) is the caller's draw for it.
func (a *metalResident) SampleChainNext(seed, draw uint64) (int, error) {
	return a.r.chainNextSampled(seed, draw)
}

// Close stops the pipelined executor (waiting for it) and frees every MTLBuffer this resident
// allocated. Metal buffers are unified/system memory and purego has no ARC, so without this a
// multi-model serve (or /admin/models/unload) leaks the whole model per load.
func (a *metalResident) Close() error { return a.r.Close() }

// PrefillTailExact (decoder.PrefillTailExact): a short PrefillLast continuing a prefix the pass prefilled reproduces the
// cold pass bit for bit (TestPrefillLast_tailContinuationMatchesCold), and the routing (floor, step range) is on the
// whole prompt's length. Proven for the dense and Gated-DeltaNet passes (TestMC5_prefillChunkInvariance,
// TestDB01_chunkedPrefillMatchesWhole) and, since 2026-10-04, for the generic resident MoE's expert-major pass
// (TestPrefillLast_tailContinuationMatchesCold_MoE: Mixtral, two Qwen3-MoE shapes and the Qwen3.5 MoE hybrid, tails of
// 1-33 rows). A paged Gemma 4 MoE's layer-major path is decode's kernels, so exact by construction; every other MoE shape
// (paged generic, resident Gemma 4, gpt-oss) declines the pass, and its suffix runs the sequential loop as before.
func (a *metalResident) PrefillTailExact() bool { return true }

// inStepRange: a prompt of promptLen tokens is in the range the batched step takes on this resident (promptStepOK's
// length test, without its two-row minimum).
func (a *metalResident) inStepRange(promptLen, floor int) bool {
	return !a.r.promptStepOff && a.VerifyCost() != nil && floor > 0 && promptLen < max(floor, metalStepPrefillCeiling)
}
