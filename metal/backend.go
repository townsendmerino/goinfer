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
	// The "metal" memory probe decoder.Model.Plan reads: the same metalMemoryCeiling the real guard (residentMemoryDecline)
	// judges against, so `fit` and the guard cannot disagree.
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

// BuildResident builds the resident Metal decoder from a loaded Model. A panic in the build (MSL compile, device creation) is
// recovered into a decline (ok=false), and the decoder falls back to the CPU. Callers gate on DecodeRunnerEligible first.
// Weights load as int4 or int8 (Options.Quant "int4" / "int8int8"); an f32 projection declines naming the quant to use
// (f32Projection). A dense int8 model runs natively on W8A8 kernels (r.w8, docs/tasks/task-metal-int8-2026-10.md); one native
// int8 does not cover (MoE, DeltaNet, int4mix) is re-quantized to int4 through the W4A8 packer an int4 load uses, and
// decoder.Model.DecodePath() names that re-quantization rather than repeating the requested quant.
func (b *metalBackend) BuildResident(m *decoder.Model) (rf decoder.ResidentForward, ok bool, err error) {
	defer func() {
		if p := recover(); p != nil {
			rf, ok, err = nil, false, decoder.DeclineResident("metal build panicked: %v", p)
		}
	}()
	// The Metal kernels implement a subset of what DecodeRunnerEligible admits (decoder.ResidentBackendFeatures("metal")). A model
	// that needs anything outside it must DECLINE (CPU fallback), never run with the feature silently dropped.
	if missing := m.MissingResidentFeatures(decoder.ResidentBackendFeatures("metal")); len(missing) > 0 {
		return nil, false, decoder.DeclineResident("metal does not implement %v, which this model needs", missing)
	}
	// An explicit -ctx above metalCtxCapMax cannot be honoured (a fixed-size kernel score buffer, not a tunable budget): refuse it
	// with a named error carrying the numbers, as cuda's errKVWontFit does, rather than folding it into the generic decline below.
	//
	// Memory guard: Metal's unified memory is host RAM and it wires the mmap pages a command buffer touches, so a model larger than
	// RAM pages until swap is exhausted and the load neither completes nor declines. The guard is keyed on two quantities we compute
	// (the model's own bytes and physical RAM, via metalMemoryCeiling), never on the OS's account of free memory: darwin's UBC reclaim
	// makes "available" report what survived, so a guard keyed on it inverts when it is needed. Evidence:
	// docs/code-notes/metal.md#metalBackend.BuildResident.guard.
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

// metalCtxCeiling is the context shrinkCtxToFit lowered an unpinned resident to, per model, read by resolveMetalCtxCap. A map
// keyed by the model so the decoder keeps no backend state; resident.Close deletes the entry (docs/tasks/task-multimodal-support-2026-10.md).
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

// residentMemFraction is the share of physical RAM the resident build (weights, host copy and KV; residentNeedBytes) may
// occupy; the rest is scratch, command buffers and the system, all in the same unified memory. It is a threshold set from one
// measured failure, not a curve: the bar sits just below that failure, so it must not be rounded up to pass it. A machine that
// would fit overrides with GOINFER_NO_RESIDENT_MEM_GUARD=1. Evidence: docs/code-notes/metal.md#residentMemFraction.
const residentMemFraction = decoder.WeightsMemFraction

// metalLiveAvailable is decoder.HostRAMAvailableBytes behind a variable, so a test can inject a machine's live-available
// figure (decoder/fitguard.go's hostRAMAvailable does the same).
var metalLiveAvailable = decoder.HostRAMAvailableBytes

// metalStaticCeiling is residentMemFraction's own arithmetic, split out so fitsResidentBudget and
// metalMemoryCeiling below share one formula rather than two copies that happen to agree.
func metalStaticCeiling(ram uint64) int64 { return int64(float64(ram) * residentMemFraction) }

// fitsResidentBudget is the static arithmetic alone (metalStaticCeiling). It never calls the live probe, so a test hands it exact
// ram and need figures and gets the pure ratio.
func fitsResidentBudget(need int64, ram uint64) bool {
	if need <= 0 || ram == 0 {
		return true // unknown ⇒ do not refuse
	}
	return uint64(need) <= uint64(metalStaticCeiling(ram))
}

// metalMemoryCeiling is the one budget both residentMemoryDecline (the load-time guard) and the registered "metal" memory probe
// (decoder.Model.Plan) use, so `fit` and the guard cannot disagree (docs/tasks/task-never-swap-2026-09.md). It is the static
// ceiling (residentMemFraction of ram) min'd with decoder.HostRAMAvailableBytes, so the live figure can only tighten the ceiling.
// That direction is deliberate: under pressure darwin's UBC reclaim makes "available" look generous, so a guard keyed on the live
// figure alone would invert exactly when needed, while min() cannot make the ceiling scarcer than the static bound already
// allowed. live<=0 (no probe, or unknown) leaves the static ceiling; ram==0 (an unreadable hw.memsize) returns 0, which every
// caller reads as "unknown".
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

// metalMoESlotsRequest is the resolved expert-slot request as a string, ready for the strconv.Atoi and validation that moe.go and
// gemma4_moe.go do at their dispatch-building call sites. `--moe-cache-slots` (Model.MoECacheSlotsRequest, the flag CUDA's
// auto-cap also reads) wins; GOINFER_METAL_MOE_SLOTS is a deprecated fallback. "" means unset: every expert resident.
func metalMoESlotsRequest(m *decoder.Model) string {
	if n := m.MoECacheSlotsRequest(); n > 0 {
		return strconv.Itoa(n)
	}
	// --moe-cache-experts without --moe-cache-slots auto-sizes the slots from memory (autoMoESlots), as CUDA does against free VRAM.
	if m.MoECacheExperts() {
		return strconv.Itoa(autoMoESlots(m))
	}
	return modelKnob(m, "GOINFER_METAL_MOE_SLOTS")
}

// moeTopK is the model's routed-experts-per-token count, the floor autoMoESlots clamps above: fewer slots than top-k cannot hold
// one token's routed set (buildResident enforces the same on an explicit --moe-cache-slots,
// TestMoESlotsViaOptions_belowTopKRefusesWithNumbers). Gemma 4's MoE and the generic MoE families keep top-k in different Config
// fields, so it branches on the shape.
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

// autoMoESlotsMax is the per-layer slot ceiling the auto-sizer requests even when memory allows more; N=64 is the default
// docs/completed/task-metal-expert-streaming-at-scale.md recommends. Raise it only if a later measurement moves that.
const autoMoESlotsMax = 64

// autoMoESlotsFor is autoMoESlots' formula against the static ceiling, split out so a test can drive it with a small ram value.
// It solves the guard's inequality (ceiling >= needFixed + N*perSlot) for N: needFixed is everything that does not scale with the
// slot count (dense weights doubled by Metal's host copy plus device buffer, and KV), perSlot one expert slot's marginal bytes.
// Clamped to [topK, autoMoESlotsMax]: below topK a token's routed set cannot fit.
func autoMoESlotsFor(topK int, perSlot, needFixed int64, ram uint64) int {
	return autoMoESlotsForBudget(topK, perSlot, needFixed, metalStaticCeiling(ram))
}

// autoMoESlotsForBudget is the formula against an explicit byte budget. autoMoESlots passes the guard's own budget
// (metalMemoryCeiling): sizing against the static fraction while the guard checks the live one would let them disagree under
// pressure (the sizer picks N, the guard refuses N, and the load falls to the CPU instead of taking fewer slots). perSlot<=0
// gives autoMoESlotsMax and a non-positive remaining budget gives topK, not a fabricated number; the guard has the final say.
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

// autoMoESlots derives a per-layer expert-slot count from live memory for a model that asked to stream MoE experts
// (MoECacheExperts) with no slot count. perSlot is the marginal bytes of one more slot (ResidentWeightBytesPaged(2) - (1)): a
// layer's experts are uniform, so it is exact and follows any change to the byte accounting. An unreadable hw.memsize falls back
// to autoMoESlotsMax.
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

// metalMoESlotsFromEnv is the guard's reader of metalMoESlotsRequest: an int, where an invalid or unset value means "assume
// unpaged" rather than an error (buildResident still validates and declines a bad one). The name predates the option; it reads
// the same resolved request.
func metalMoESlotsFromEnv(m *decoder.Model) int {
	n, err := strconv.Atoi(metalMoESlotsRequest(m))
	if err != nil || n <= 0 {
		return 0
	}
	return n
}

// residentKVBytes is the resident KV cache's footprint at the resolved ctx cap. It runs before buildResident, so it resolves the
// cap itself; on an error (an explicit ctx above metalCtxCapMax) it takes metalCtxCapMax, a safe over-estimate for a guard whose
// job is not to under-count, since BuildResident reports that error on its own, earlier. The small per-model buffers (mq, gu,
// logits, ...) are a few hundred KB each and round to nothing beside KV, the exemption ResidentWeightBytes gives norms and biases.
func residentKVBytes(m *decoder.Model) int64 {
	ctxCap, err := resolveMetalCtxCap(m)
	if err != nil {
		ctxCap = metalCtxCapMax
	}
	// decoder.Model.ResidentKVBytes is the one definition shared with Plan("metal"), so `fit` and this guard price KV identically
	// (docs/tasks/task-memory-accounting-2026-09.md): f16, or int8 plus per-head scales, every attention layer at the full ctx padded
	// to 8, no Gated-DeltaNet layer.
	return m.ResidentKVBytes("metal", ctxCap, false, false)
}

// residentNeedBytes is the byte count residentMemoryDecline judges against, split out so a test can drive it without real RAM. It
// prices the paged weight estimate when a slot count is requested (metalMoESlotsFromEnv), not the every-expert-resident figure,
// and adds Metal's host copy (the quantized host WeightMat stays beside the re-packed device buffer; genuinely paged experts have
// none, so the host-copy addend is asked at the same slot count) and KV (residentKVBytes).
func residentNeedBytes(m *decoder.Model) int64 {
	ctxCap, err := resolveMetalCtxCap(m)
	if err != nil {
		ctxCap = metalCtxCapMax
	}
	// decoder.Model.ResidentNeedBytes is also what Plan("metal").NeedBytes() is built from, so `fit` and this guard are the same
	// number by construction. The rest is what a companion claims beside the resident (a vision tower priced by serve's
	// towerReserve, a drafter): Metal's device memory is the same RAM.
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
// (decoder.Options.ResidentKVSlotsDefault: serve without -kv-sessions). Every slot's KV is resident from the first token, so
// the default 4 costs memory on every load. Two keep the batched step (it needs 2 slots) and a second conversation's prefix; a
// third and fourth concurrent client lose their own slot and their batching. -kv-sessions N still asks for N
// (docs/audit-metal-2026-09-30.md, E-P09).
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
	// The same ceiling the registered "metal" memory probe reports to decoder.Model.Plan.
	budget := metalMemoryCeiling(ram)
	if need <= budget {
		return ""
	}
	const gb = 1 << 30
	// A model with routed experts that does not fit resident may still fit paged: the decline names how to reach it.
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

	// fast is the batched-prefill decision, made once per resident on first use, so a change to the process environment mid-serve
	// cannot switch a loaded model's kernels. Lazy (not set at build) so a test's struct-literal resident gets the same default.
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

// ctxCap is this resident's resolved KV capacity: a.r.ctxCap, or metalCtxCapDefault for a zero-value &metalResident{} (r == nil),
// which TestMetalResidentCheckCap builds to test checkCap and ContextCap as pure logic with no Metal device.
func (a *metalResident) ctxCap() int {
	if a.r == nil || a.r.ctxCap == 0 {
		return metalCtxCapDefault
	}
	return a.r.ctxCap
}

// checkCap refuses positions [pos, pos+n) outside the resident KV [0, ctxCap()). Every layer's cache is sized ctxCap()*kvDim, so a
// write past it is an out-of-bounds device write that on unified memory silently corrupts adjacent MTLBuffers (other models'
// resident weights). The decode loop increments pos unbounded, so this is the guard: the loop surfaces the error and the caller
// can fall back to the staged path.
func (a *metalResident) checkCap(pos, n int) error {
	c := a.ctxCap()
	if pos < 0 || pos+n > c {
		return fmt.Errorf("metal: KV position %d(+%d) exceeds resident context cap %d — use the staged path for longer contexts", pos, n, c)
	}
	return nil
}

// ContextCap is the resident KV capacity in positions. It makes metalResident a decoder.ResidentCapped, so generateInto clamps
// maxTokens to the cap up front instead of erroring mid-decode.
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
		// A fresh sequence re-zeroes any Gated-DeltaNet state left by a prior Generate on this resident (no-op for other families).
		a.Reset()
	}
	logits := a.r.ForwardEmbMRoPEPipe(embedding, pos, ropePos)
	if err := a.r.takeExecErr(); err != nil {
		return nil, err // a command buffer aborted: surface it, never return stale logits
	}
	return logits, nil
}

// Forward runs one token given its embedding[H] at absolute position pos, returning logits[V].
// The returned slice is reused across calls (the decode loop consumes it before the next call).
func (a *metalResident) Forward(embedding []float32, pos int) ([]float32, error) {
	return a.ForwardMRoPE(embedding, pos, pos)
}

var _ decoder.ResidentSample = (*metalResident)(nil)

// SampleAvailable and ForwardSample (decoder.ResidentSample, gumbel_sample.go) delegate to *resident. They must exist on
// *metalResident because decoder holds one, and Go does not promote a named field's methods.
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

// ForwardNoLogits (decoder.ResidentPrefillKV) runs the token's forward to build only its resident K/V: no final-norm LM-head
// dispatch, no logits readback, no softcap. residentPrefillSeed calls it for every prompt token but the last; the layers, hence
// the K/V written at pos, are identical to Forward, so decode from the last prompt token is byte-identical. It runs through the
// encode-ahead executor (ForwardEmbNoLogitsPipe, a noHead bit on execJob). A paged MoE has no paged branch in the executor's
// trunk encoder, so it runs the synchronous paged forward with its final norm and head skipped (pagedNoHead).
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
	return a.r.takeExecErr() // a command buffer aborted: surface it
}

// metalFastPrefillFloor is the prompt-length floor (whole prompt = startPos+M) below which the f16-MMA batched pass declines and
// the sequential per-token loop runs instead (or, on a resident with the batched step, the exact step: metalStepPrefillCeiling).
// It is the lowest K at which the §3.2 pooled fidelity gate passed and the pass beat the sequential loop; below it no decision
// cell has passed. Lowering it needs that gate at the new K (docs/measurements/metal-prefill-floor-2026-09-20.md,
// docs/tasks/task-metal-audit-2026-10.md). GOINFER_METAL_FAST_PREFILL_FLOOR overrides it, 0 = no floor.
const metalFastPrefillFloor = 16

// metalStepPrefillCeiling bounds the batched step's range: a resident with the step runs a prompt that ends below this many
// tokens as exact decode rows on it (E-P01) rather than the pass. The step is bit-identical to the sequential loop and the pass
// is not, so it keeps every length where it is not slower (the step is faster at K = 16, the pass at K = 32, on both measured
// models; docs/tasks/task-metal-audit-2026-10.md, A-P02).
const metalStepPrefillCeiling = 32

// metalFastPrefillEnabled reports whether the batched f16-MMA prefill is selected. Default on above metalFastPrefillFloor
// (TestPrefillGateVsReference, docs/measurements/metal-prefill-floor-2026-09-20.md); GOINFER_METAL_FAST_PREFILL=0/false/off or
// --exact-prefill opts out.
//
//	GOINFER_METAL_FAST_PREFILL  1 | true | on   on (even below the floor, for tests)
//	                            0 | false | off  off (explicit opt-out; use --exact-prefill on the server)
//
// The old GOINFER_METAL_BATCHED_PREFILL still works: =1 forces on, =0 forces off, unset defers to the default.
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
	return true
}

// metalAttnFAEnabled reports whether decode attention defaults to attention_fa, the kvHead x split-gridded kernel gated per layer
// by canUseAttnFA (dense GQA, hd=128, key count >= attnFADepthFloor; GQA group sizes 6 and 7 take the block kernel
// attention_fa_blk). Default on, by owner decision after the fidelity gate of
// docs/measurements/metal-decode-attn-fidelity-setb-PREREGISTERED.md (kernel error in
// docs/measurements/metal-decode-attn-r17-2026-09-25.md); the first gate was void:
// docs/code-notes/metal.md#metalAttnFAEnabled.
//
// It is NOT bit-identical to the `attention` kernel (reduction and combine order differ by design), so it moves argmax at the
// margin on some inputs and is a second source of decode/ForwardN divergence beside PrefillLast's f16-MMA activations
// (docs/spec/08-dspark-dflash.md). Metal spec-decode verify is not a legal oracle for that reason and stays so.
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
	return true
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

// metalFusedAttentionEnabled reports whether attention_prefill_fused (the simdgroup_matrix flash-attention twin of
// attention_prefill, docs/completed/task-prefill-gap.md §4) runs in place of the exact scalar kernel. Default on
// (docs/measurements/prefill-l2-metal-fused-attn-2026-09-09.md §5); GOINFER_METAL_FUSED_ATTENTION=0 or --exact-prefill (which
// also covers metalFastPrefillEnabled) opts back to the exact kernel. The fused kernel needs hd%8==0 && hd<=128 (ATTN_MAXHD in
// prefill.go); PrefillLast uses the exact kernel outside that range regardless of this flag.
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
	return true
}

// ResidentQuant (decoder.ResidentQuantReporter) is the precision this resident runs its weights at, when the decoder cannot
// infer it: an int8 model on the native path (r.w8) runs at its own quant, where one on the int4 path is re-quantized. ""
// leaves the decoder's label.
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

// PrefillPath (decoder.PrefillPathReporter) reports at load time whether this resident's prefill takes the batched f16-MMA
// path, and why or why not. The floor applies per call (PrefillKernelFloor); it reports true iff the enabled state and the arch
// both allow batching.
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
	// One check at entry: the pass is a single command buffer with no inner loop to interrupt, and the decoder's chunked prefill
	// cuts a prompt only while another conversation decodes, so a lone long prompt cannot be cancelled midway. That would need
	// chunking inside PrefillLast, as cuda's prefillChunked does (docs/audit-metal-2026-09-30.md, A-D02).
	if e := ctx.Err(); e != nil {
		return nil, e
	}
	// A paged Gemma 4 MoE, a paged generic MoE and a Gemma 4 E-model run their prompts layer by layer on decode's own kernels,
	// bit-identical to the sequential loop, so neither the floor nor --exact-prefill applies to them (each behind its own switch;
	// docs/tasks/task-m26-mac-2026-10.md). An E-model prompt at or above the floor takes the batched pass when it is on; below the
	// floor, or on any decline, the layer-major pass below runs it.
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
	// The batched pass is on by default (metalFastPrefillEnabled); GOINFER_METAL_FAST_PREFILL=0 or --exact-prefill opts out.
	if !a.fastPrefill() {
		return nil, fmt.Errorf("metal: fast prefill disabled (GOINFER_METAL_FAST_PREFILL=0 / --exact-prefill / GOINFER_METAL_BATCHED_PREFILL=0); using sequential path")
	}
	// Below metalFastPrefillFloor the batched pass declines. A resident with the step kernels runs a prompt that ends below
	// max(floor, metalStepPrefillCeiling) as decode rows on them: the sequential loop's bits, faster than the pass there
	// (promptStepOK).
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

// prefillUnaligned names the first reduction length the batched pass's GEMMs cannot take, or "" when every one is a multiple of
// 32: gemm_w4f16_tile and its int8 twin stage K in slabs of 32 with no partial-slab handling (prefill.go). An int4 resident
// cannot violate it (int4Buf refuses K%32 != 0 at build), but a native int8 one can (docs/tasks/task-metal-pairwise-followups-2026-10.md,
// Part A). Real checkpoints are 32-aligned; this keeps an unaligned model on the sequential path instead of computing garbage.
func (r *resident) prefillUnaligned() string {
	if r.H%32 != 0 {
		return fmt.Sprintf("the hidden size is %d", r.H)
	}
	for l := range r.layers {
		L := &r.layers[l]
		if L.geom != nil && (r.nH*L.geom.hd)%32 != 0 {
			return fmt.Sprintf("layer %d's attention output width is %d", l, r.nH*L.geom.hd)
		}
		if fi := L.ffnI; L.moe == nil && fi == 0 && r.I%32 != 0 {
			return fmt.Sprintf("the FFN width is %d", r.I)
		} else if fi%32 != 0 {
			return fmt.Sprintf("layer %d's FFN width is %d", l, fi)
		}
	}
	return ""
}

// batchedPrefill is the f16 batched pass behind PrefillLast and PrefillMRoPELast (S16): the same declines (the floor, an
// int8 KV cache, a family the pass does not implement, the resident's cap, the exact attention kernel's key limit), the
// same recovery of a request-time panic, and the same non-finite-logit check. mrope is nil for a text prompt.
func (a *metalResident) batchedPrefill(embeddings [][]float32, startPos, floor int, mrope [][3]int, deep *prefillDeep) ([]float32, error) {
	return a.batchedPrefillImg(embeddings, startPos, floor, mrope, deep, nil)
}

// PrefillImageLast (decoder.ResidentImagePrefill) is the batched f16 pass over an image turn's spliced embeddings, with the image
// block [imgStart, imgEnd) attending bidirectionally (attention_prefill_img), as CUDA's PrefillImageLast and the CPU's
// prefillLogitsVL do. The whole prompt is one pass, never chunked across the block. It takes the plain uniform pass only: a
// Gemma 4 E-model (its rows carry per-layer inputs this route does not build), a layer-major family or a recurrent hybrid
// declines, and the decoder takes its CPU-prefill and upload bridge.
func (a *metalResident) PrefillImageLast(ctx context.Context, embeddings [][]float32, startPos, imgStart, imgEnd int) ([]float32, error) {
	return a.PrefillImageBlocksLast(ctx, embeddings, startPos, [][2]int{{imgStart, imgEnd}})
}

// PrefillImageBlocksLast (decoder.ResidentImageBlocksPrefill; S11) is PrefillImageLast for several images: each block
// [start, end), in order and disjoint, attends bidirectionally within itself.
func (a *metalResident) PrefillImageBlocksLast(ctx context.Context, embeddings [][]float32, startPos int, blocks [][2]int) ([]float32, error) {
	if e := ctx.Err(); e != nil {
		return nil, e
	}
	prev := startPos
	for _, b := range blocks {
		if b[0] < prev || b[1] <= b[0] || b[1] > startPos+len(embeddings) {
			return nil, fmt.Errorf("metal: image blocks %v invalid for %d rows at %d", blocks, len(embeddings), startPos)
		}
		prev = b[1]
	}
	if len(blocks) == 0 {
		return nil, fmt.Errorf("metal: no image block")
	}
	if !a.r.prefillOK || a.r.pleP > 0 || a.r.dnet != nil || a.g4LayerMajor() || a.moeLayerMajor() || a.emodelLayerMajor() {
		return nil, fmt.Errorf("metal: the resident image prefill takes the plain uniform pass; this model's image turn prefills on the CPU")
	}
	if !a.fastPrefill() {
		return nil, fmt.Errorf("metal: fast prefill disabled (GOINFER_METAL_FAST_PREFILL=0 / --exact-prefill); the image prefill runs on the CPU")
	}
	// attention_prefill_img is the exact kernel: its score buffer holds prefillExactAttnMaxKeys keys.
	if startPos+len(embeddings) > prefillExactAttnMaxKeys {
		return nil, fmt.Errorf("metal: an image turn of %d positions is past the image prefill's %d-key limit; the image prefill runs on the CPU",
			startPos+len(embeddings), prefillExactAttnMaxKeys)
	}
	return a.batchedPrefillImg(embeddings, startPos, 0, nil, nil, blocks)
}

var (
	_ decoder.ResidentImagePrefill       = (*metalResident)(nil)
	_ decoder.ResidentImageBlocksPrefill = (*metalResident)(nil)
)

// batchedPrefillImg is batchedPrefill with optional bidirectional image blocks (nil: none).
func (a *metalResident) batchedPrefillImg(embeddings [][]float32, startPos, floor int, mrope [][3]int, deep *prefillDeep, blocks [][2]int) ([]float32, error) {
	promptLen := startPos + len(embeddings)
	if floor > 0 && promptLen < floor {
		return nil, fmt.Errorf("metal: prompt too short (%d tokens) for fast prefill (floor=%d; §3 floor); using sequential path", promptLen, floor)
	}
	// The f16 MMA prefill kernels implement a dense gated FFN (SiLU or GeGLU) and a generically shaped gated-SwiGLU MoE FFN
	// (expert-major by default, the per-token decode MoE chain row by row with GOINFER_MOE_EXPERT_MAJOR=0), with per-layer
	// rope/window, per-head QK-norm and Gemma's sandwich norms. They do NOT implement per-layer-varying attention geometry (dense
	// Gemma 4's local/global head_dim split; prefillOK) or Gemma 4's enable_moe_block variant (HasGemma4MoEResident, a third FFN
	// shape). Any of these declines here, and the caller re-runs the prompt through the sequential Forward loop.
	if a.r.kvI8 {
		return nil, fmt.Errorf("metal: prefill writes half-precision K/V and this model's KV cache is int8 (-kv i8); using sequential path")
	}
	if !a.r.prefillOK && !a.emodelBatched() {
		return nil, fmt.Errorf("metal: prefill not implemented for this arch's FFN shape (use the sequential path)")
	}
	if a.r.layerNorm && (a.r.layerNormBias || a.r.qkNorm) {
		return nil, fmt.Errorf("metal: the prefill's LayerNorm is bias-free with no QK-norm, and this model has a LayerNorm bias or QK-norm; using sequential path")
	}
	if why := a.r.prefillUnaligned(); why != "" {
		return nil, fmt.Errorf("metal: the prefill GEMMs need every reduction length a multiple of 32 and %s; using sequential path", why)
	}
	// startPos < 0 would wrap to a huge uint32 and make kv_store_f16 write far out of bounds, silently corrupting adjacent buffers on
	// unified memory. The decoder never passes one, but the guard is cheap.
	if startPos < 0 || len(embeddings) == 0 || startPos+len(embeddings) > a.ctxCap() {
		return nil, fmt.Errorf("metal: prompt len %d at startPos %d out of resident cap %d", len(embeddings), startPos, a.ctxCap())
	}
	// The exact attention kernel (no fused kernel for this head dim, or fused attention off) holds at most prefillExactAttnMaxKeys
	// scores; the sequential path's decode kernels tile theirs.
	if a.r.prefillExactAttn() && startPos+len(embeddings) > prefillExactAttnMaxKeys {
		return nil, fmt.Errorf("metal: prompt reaches %d keys and the exact prefill attention kernel holds %d (head dim %d has no fused kernel, or it is off); using sequential path",
			startPos+len(embeddings), prefillExactAttnMaxKeys, a.r.prefillGeom().hd)
	}
	// A fresh sequence starts a Gated-DeltaNet hybrid's recurrent state from zero, as Forward(pos 0) does; a continuation keeps the
	// conv window and state the resident holds. No-op for every other family.
	if startPos == 0 {
		a.Reset()
	}
	// ensurePrefill's compile panic and the per-call MustBuf OOM panics fire here, at request time, outside buildResident's recover.
	// Recover into an error so a transient OOM fails the request (the caller falls back to sequential decode) rather than killing
	// the server.
	var logits []float32
	if err := func() (err error) {
		defer func() {
			if p := recover(); p != nil {
				err = fmt.Errorf("metal: batched prefill aborted: %v", p)
			}
		}()
		logits = a.r.prefillLastImg(embeddings, startPos, mrope, deep, blocks)
		return nil
	}(); err != nil {
		return nil, err
	}
	if err := a.r.takeExecErr(); err != nil {
		return nil, err
	}
	// The f16 residual overflowed: the head reads the NaN row as finite zeros, so the logit check below cannot see it.
	if a.r.prefillResidNonFinite {
		return nil, fmt.Errorf("metal: batched prefill's f16 residual overflowed (its last row is non-finite after the trunk); using sequential path")
	}
	if a.poisonPrefillLogitsForTest {
		logits[0] = float32(math.NaN())
	}
	// The batched pass carries f16 activations, which a checkpoint the graded models never exercised could overflow; the sequential
	// kernels carry f32. A non-finite logit declines, and the decoder re-runs the prompt sequentially, which rewrites every K/V row
	// this pass wrote. The scan costs about 0.05 ms at a 152k vocabulary.
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

// metalMRoPEPrefillOn and metalDeepstackPrefillOn turn the resident m-RoPE prefill on, for a turn without DeepStack sets
// (Qwen2.5-VL; Qwen3.5+ is not claimed) and with them (Qwen3-VL). Off, an image turn takes the CPU prefill and the upload. Both
// are on by owner decision after the real-checkpoint gates (docs/tasks/task-multimodal-support-2026-10.md,
// docs/completed/task-metal-prefill-precision-2026-10.md). Tests set them.
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

// HiddenLast (decoder.ResidentHiddenLast) ingests a whole sequence at startPos and returns the last position's hidden state after
// the model's final norm, for embedding requests: the resident twin of PrefillLast that never runs the LM head. It runs the same
// per-token sequential kernels as decode (one forwardHiddenNoHead and one command-buffer submit per position), so it does not
// match the CPU reference bit for bit: the decode kernels round differently and forwardHiddenNoHead returns the
// int8-dequantized activation (the bar is in hiddenlast_resident_parity_test.go). A batched version, PrefillLast's dispatch
// graph minus the LM head and softcap, gated against this function as its oracle, is not built
// (docs/code-notes/metal.md#metalResident.HiddenLast).
func (a *metalResident) HiddenLast(ctx context.Context, embeddings [][]float32, startPos int) ([]float32, error) {
	if len(embeddings) == 0 {
		return nil, fmt.Errorf("metal: HiddenLast called with no embeddings")
	}
	// forwardHiddenNoHead's trunk encoder has no paged branch, so on a paged MoE it would bind the zero-value stacked expert buffers
	// and return a finite garbage hidden state with no error. Decline instead: decoder.Model.HiddenLast treats a resident decline as
	// "no resident backend for this request" and falls through to the CPU, as it does for an OOM or cap decline.
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
		// An abandoned client must not leave the sequence streaming through the device (residentPrefillSeed's loop checks the same way).
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		want := i == len(embeddings)-1
		h, err := a.r.forwardHiddenNoHead(emb, startPos+i, want)
		if err != nil {
			return nil, err
		}
		if err := a.r.takeExecErr(); err != nil {
			return nil, err // a command buffer aborted: never return a stale vector
		}
		if want {
			out = h
		}
	}
	return out, nil
}

var _ decoder.ResidentResidualAll = (*metalResident)(nil)

// ResidualAll (decoder.ResidentResidualAll) ingests a whole sequence at startPos and returns every position's residual stream
// after the last layer and before the final norm, each row a fresh f32 slice. The decoder applies the final norm on the host in
// f32 (decoder.Model.residualAllResident), so the rows never pass through the int8 the head path reads: Metal fuses the final
// norm into the trunk (encodeTrunkWith: layers, then encodeNorm into r.aq/r.aSc), and that int8 vector is what this avoids.
// Every layer writes its residual into r.x and the norm reads r.x through `device const float*`, so once the trunk has run r.x
// holds the last layer's output untouched; this copies it out per token.
//
// It uses the same per-token sequential kernels as HiddenLast and costs what HiddenLast costs plus an H-float copy per row; a
// batched version is not built. Guards mirror HiddenLast's; rows are returned only once every token has succeeded.
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
			return nil, err // a command buffer aborted: never return a stale residual
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
		return nil, err // a command buffer aborted: surface it
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

// KVSlots and UseKVSlot implement decoder.ResidentKVSlotter (docs/tasks/task-concurrency-2026-09.md): the slots metalKVSlots
// allocated, and binding one.
func (a *metalResident) KVSlots() int          { return a.r.kvSlotCount() }
func (a *metalResident) UseKVSlot(i int) error { return a.r.useKVSlot(i) }

// TruncateTo is a no-op: KV positions are overwritten on write, and attention reads only keys[0..pos], so stale positions past
// the current one are never observed.
func (a *metalResident) TruncateTo(pos int) {}

// Reset zeroes every Gated-DeltaNet layer's causal-conv ring and recurrent matrix state (no-op for every other family; a.r.dnet is
// nil). Unlike KV positions this state compounds, so a fresh sequence on the same resident must reset it or it keeps decaying the
// previous sequence's state (resetDeltaNet).
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

// PrefillTailExact (decoder.PrefillTailExact): a short PrefillLast continuing a prefix the pass prefilled reproduces the cold pass
// bit for bit (TestPrefillLast_tailContinuationMatchesCold), and the routing (floor, step range) is on the whole prompt's length.
// Proven for the dense and Gated-DeltaNet passes (TestMC5_prefillChunkInvariance, TestDB01_chunkedPrefillMatchesWhole) and for the
// generic resident MoE's expert-major pass (TestPrefillLast_tailContinuationMatchesCold_MoE). A paged Gemma 4 MoE's layer-major
// path is decode's kernels, so exact by construction; every other MoE shape (paged generic, resident Gemma 4, gpt-oss) declines
// the pass, and its suffix runs the sequential loop.
func (a *metalResident) PrefillTailExact() bool { return true }

// inStepRange: a prompt of promptLen tokens is in the range the batched step takes on this resident (promptStepOK's
// length test, without its two-row minimum).
func (a *metalResident) inStepRange(promptLen, floor int) bool {
	return !a.r.promptStepOff && a.VerifyCost() != nil && floor > 0 && promptLen < max(floor, metalStepPrefillCeiling)
}
