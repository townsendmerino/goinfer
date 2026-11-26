//go:build darwin

package metal

// Metal decoder backend — plugs the cgo-free native-Metal resident decoder into the goinfer
// decode loop via decoder.RegisterBackend, mirroring the cuda backend. Blank-import this
// package from a main (darwin only; no build tag) to enable `--backend metal`. Dense and MoE
// residency, per docs/hardware-matrix.md; a model it does not admit runs entirely on the CPU
// (Metal has no partial staged path).

import (
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/runtime/logs"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
	"github.com/townsendmerino/goinfer/internal/runtime/mem/probe"
	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

// Compile-time seams: catch signature drift against decoder/residency.go + decoder/backend.go.
var (
	_ decoder.Backend             = (*metalBackend)(nil)
	_ decoder.ResidencyBackend    = (*metalBackend)(nil)
	_ decoder.ResidencyPricer     = (*metalBackend)(nil)
	_ decoder.ResidentForward     = (*metalResident)(nil)
	_ decoder.Prefiller           = (*metalResident)(nil)
	_ decoder.PrefillPathReporter = (*metalResident)(nil)
	_ decoder.VerifyPathReporter  = (*metalResident)(nil)
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
// (f32Projection). A dense int8 model runs natively on W8A8 kernels; one native
// int8 does not cover (MoE, DeltaNet, int4mix) is re-quantized to int4 through the W4A8 packer an int4 load uses, and
// decoder.Model.DecodePath() names that re-quantization rather than repeating the requested quant.
func (b *metalBackend) BuildResident(m *decoder.Model) (rf decoder.ResidentForward, ok bool, err error) {
	defer func() {
		if p := recover(); p != nil {
			mem.RepanicBug(p)
			rf, ok, err = nil, false, decoder.DeclineResident("metal build panicked: %v", p)
		}
	}()
	plan, from, perr := residentPreBuild(m)
	if perr != nil {
		return nil, false, perr
	}
	res, e := buildResident(m, plan)
	if e != nil {
		return nil, false, decoder.DeclineResident("metal: %v", e)
	}
	res.ctxCeil = plan.ceil
	b.resident = &metalResident{r: res, hidden: res.H, exact: m.ExactPrefill(), need: func() int64 { return residentNeedBytes(m, res.ctxPlan()) }}
	share := res.ctxCap // the note prices the context the plan sized each slot for, not the cap one conversation may borrow up to
	if res.kvShareCtx > 0 {
		share = res.kvShareCtx
	}
	b.resident.ctxNote = metalContextNote(m, plan.fit, share, res.kvSlotCount())
	if got, want := b.resident.embLen(), m.BackendView().EmbedLen(); got != want {
		_ = res.Close()
		b.resident = nil
		return nil, false, decoder.DeclineResident("metal: resident takes %d-float embedding rows, the decoder builds %d", got, want)
	}
	if res.w8 || res.w8Attn {
		b.resident.quant = m.Quant()
	}
	if ctx, _ := resolveMetalCtxCap(m, plan); ctx < from || m.ExtraResidentBytes() > 0 {
		logs.InfoText(metalKVPlanLine(res.kvSlotCount(), ctx, from, m.ExtraResidentBytes()))
	}
	return b.resident, true, nil
}

// metalCtxPlan is the context one resident build decides, handed to every reader that sizes from it: the unpinned
// context's fit choice, made once per build (fitSet; a plan without one asks metalFitNow at each read), and the context
// shrinkCtxToFit lowered an unpinned resident to (ceil, 0 for none). The zero value is no build: each reader decides now.
type metalCtxPlan struct {
	fit    decoder.MetalFit
	fitSet bool
	ceil   int
}

// ctxPlan is the plan a built resident's readers size from: its build's ceiling, with the fit choice asked again at each read.
func (r *resident) ctxPlan() metalCtxPlan { return metalCtxPlan{ceil: r.ctxCeil} }

// residentPreBuild is BuildResident's decisions before buildResident allocates anything, which PriceResident asks too: the
// plan the build sizes from, and the context before the memory guard lowered it (from). err is BuildResident's own return
// for a refusal.
func residentPreBuild(m *decoder.Model) (plan metalCtxPlan, from int, err error) {
	// The Metal kernels implement what Metal's operation table lists (program.Metal). A model that needs anything outside it
	// must DECLINE (CPU fallback), never run with an operation or parameter silently dropped.
	if err := program.Metal.Check(m.Program()); err != nil {
		return plan, 0, decoder.DeclineResident("%v", err)
	}
	if missing := m.FeaturesOutsideProgram(); len(missing) > 0 {
		return plan, 0, decoder.DeclineResident("metal does not implement %v, which this model needs", missing)
	}
	// An explicit -ctx above metalCtxCapMax cannot be honoured (a fixed-size kernel score buffer, not a tunable budget): refuse it
	// with a named error carrying the numbers, as cuda's errKVWontFit does, rather than folding it into the generic decline below.
	//
	// Memory guard: Metal's unified memory is host RAM and it wires the mmap pages a command buffer touches, so a model larger than
	// RAM pages until swap is exhausted and the load neither completes nor declines. The guard is keyed on two quantities we compute
	// (the model's own bytes and physical RAM, via metalMemoryCeiling), never on the OS's account of free memory: darwin's UBC reclaim
	// makes "available" report what survived, so a guard keyed on it inverts when it is needed. Evidence:
	// docs/code-notes/metal.md#metalBackend.BuildResident.guard.
	if _, cerr := resolveMetalCtxCap(m, plan); cerr != nil {
		return plan, 0, cerr
	}
	// The unpinned context is chosen once, here: its budget is live, and every later reader in this build (the guard, the
	// slot count, the KV sizing) must see one number.
	plan.fit, plan.fitSet = metalFitNow(m), true
	from, _ = resolveMetalCtxCap(m, plan)
	if why := residentMemoryDecline(m, plan); why != "" {
		if !shrinkCtxToFit(m, &plan) {
			return plan, from, decoder.DeclineResident("%s", why)
		}
	}
	return plan, from, nil
}

// PriceResident (decoder.ResidencyPricer) is BuildResident's answer before it allocates: residentPreBuild, with the
// context the resident would hold. Whether an int8 load runs natively or re-quantized to int4 is decided inside the build
// (w8Eligible reads the resident it is building), so for those quants the label is left to the load.
func (b *metalBackend) PriceResident(m *decoder.Model) (decoder.ResidentPrice, error) {
	plan, _, err := residentPreBuild(m)
	if err != nil {
		return decoder.ResidentPrice{}, err
	}
	ctx, _ := resolveMetalCtxCap(m, plan)
	p := decoder.ResidentPrice{Ctx: ctx}
	switch q := m.Quant(); q {
	case "int4", "q4k":
	default:
		p.Unknown = fmt.Sprintf("metal goes resident at this load's memory and context; whether its %s weights run natively or "+
			"re-quantized to int4 is decided by the build", q)
	}
	return p, nil
}

// metalFitFor is the context an unpinned load takes (decoder.Model.MetalFitCtx): the plan's choice when its build made one,
// otherwise the choice a build would make now. docs/code-notes/metal.md#metalFitFor.
func metalFitFor(m *decoder.Model, plan metalCtxPlan) decoder.MetalFit {
	if plan.fitSet {
		return plan.fit
	}
	return metalFitNow(m)
}

// metalFitNow asks decoder.Model.MetalFitCtx with this load's facts: the memory guard's own budget (the registered "metal"
// probe), the KV slots the build will ask for, whether it pages MoE experts, and a guard pin as the limit it is.
func metalFitNow(m *decoder.Model) decoder.MetalFit {
	free, known := decoder.FreeBytesFor("metal")
	in := decoder.MetalFitInput{FreeBytes: free, FreeKnown: known && !metalFitFreeUnknownForTest, KVSlots: metalKVSlotsWanted(m), PagedMoE: metalPagesMoE(m)}
	if req := m.ResidentContextRequest(); req > 0 && !m.ResidentContextPinned() {
		in.Lower = req
	}
	return m.MetalFitCtx(in)
}

// metalFitFreeUnknownForTest makes metalFitNow treat the Metal budget as unknown, so the admission census's plant can show that
// exactly the lines whose context the fit choice raised go red. False in production.
var metalFitFreeUnknownForTest bool

// metalPagesMoE reports whether the load asked to page its routed experts, by any of the three ways metalMoESlotsRequest
// reads. It reads the requests and not the resolved slot count, which is sized from the KV this decides.
func metalPagesMoE(m *decoder.Model) bool {
	return m.MoECacheSlotsRequest() > 0 || m.MoECacheExperts() || modelKnob(m, "GOINFER_METAL_MOE_SLOTS") != ""
}

// metalContextNote is the resident's ContextNote: fit's reason, restated for what the build really holds when the guard
// lowered the context after the choice or granted fewer slots. "" for an explicit -ctx.
func metalContextNote(m *decoder.Model, fit decoder.MetalFit, ctx, slots int) string {
	switch {
	case m.ResidentContextRequest() > 0 && m.ResidentContextPinned():
		return ""
	case ctx < fit.Ctx:
		return "reduced to fit this machine's memory"
	case ctx > metalCtxCapDefault:
		return decoder.MetalFitWhy(slots, m.MetalKVBytes(slots, ctx))
	}
	return fit.Why
}

// metalCtxFloor is the shortest context shrinkCtxToFit gives a model before the guard declines: an image turn (Gemma 3's
// 256 soft tokens, a few hundred of text) and some conversation fit.
const metalCtxFloor = 2048

// shrinkCtxToFit halves an unpinned resident context, down to metalCtxFloor, until the memory guard passes (the
// context gives way before the resident declines; an explicit -ctx never does), recording it as plan.ceil. It reports
// whether one passed; when none does the plan is left as it was.
func shrinkCtxToFit(m *decoder.Model, plan *metalCtxPlan) bool {
	if m.ResidentContextPinned() {
		return false
	}
	cur, err := resolveMetalCtxCap(m, *plan)
	if err != nil {
		return false
	}
	was := plan.ceil
	for cur > metalCtxFloor {
		cur = max(cur/2, metalCtxFloor)
		plan.ceil = cur
		if residentMemoryDecline(m, *plan) == "" {
			return true
		}
	}
	plan.ceil = was
	return false
}

// metalKVPlanLine is serve's KV plan line: the slots and context the build chose, and what for when it
// is less than asked or a companion (a vision tower, a drafter) is priced beside it.
func metalKVPlanLine(slots, ctx, from int, extra int64) string {
	s := fmt.Sprintf("metal: KV plan: %d conversation(s) x %d positions", slots, ctx)
	switch {
	case ctx < from && extra > 0:
		s += fmt.Sprintf(" (context reduced from %d to leave %s for the vision tower or drafter)", from, mem.GiB(extra))
	case ctx < from:
		s += fmt.Sprintf(" (context reduced from %d to fit this machine's memory)", from)
	case extra > 0:
		s += fmt.Sprintf(" (%s left for the vision tower or drafter)", mem.GiB(extra))
	}
	return s + "\n"
}

// metalLiveAvailable is decoder.HostRAMAvailableBytes behind a variable, so a test can inject a machine's live-available
// figure (decoder/fitguard.go's hostRAMAvailable does the same).
var metalLiveAvailable = decoder.HostRAMAvailableBytes

// errRAMUnknown is physicalRAM's answer when the machine's RAM cannot be read.
var errRAMUnknown = errors.New("physical RAM unknown")

// physicalRAM is this machine's physical RAM (hw.memsize, read by internal/mem/probe), or errRAMUnknown. A variable so a test
// can price a recorded machine (metal's TestBackendAdmissionCensus_metal).
var physicalRAM = func() (uint64, error) {
	if b := probe.Physical(); b > 0 {
		return uint64(b), nil
	}
	return 0, errRAMUnknown
}

// metalStaticCeiling is the share of physical RAM the resident build (weights, host copy and KV; residentNeedBytes) may
// occupy (mem.RAMShare); the rest is scratch, command buffers and the system, all in the same unified memory. The share is a
// threshold set from one measured failure, not a curve: the bar sits just below that failure, so it must not be rounded up
// to pass it. A machine that would fit overrides with GOINFER_NO_RESIDENT_MEM_GUARD=1. Evidence:
// docs/code-notes/metal.md#residentMemFraction. fitsResidentBudget and metalMemoryCeiling share this one formula.
func metalStaticCeiling(ram uint64) int64 { return mem.RAMShare(int64(ram)) }

// fitsResidentBudget is the static arithmetic alone (metalStaticCeiling). It never calls the live probe, so a test hands it exact
// ram and need figures and gets the pure ratio.
func fitsResidentBudget(need int64, ram uint64) bool {
	if need <= 0 || ram == 0 {
		return true // unknown ⇒ do not refuse
	}
	return uint64(need) <= uint64(metalStaticCeiling(ram))
}

// metalMemoryCeiling is the one budget both residentMemoryDecline (the load-time guard) and the registered "metal" memory probe
// (decoder.Model.Plan) use, so `fit` and the guard cannot disagree. It is the static
// ceiling (metalStaticCeiling of ram) min'd with decoder.HostRAMAvailableBytes, so the live figure can only tighten the ceiling.
// That direction is deliberate: under pressure darwin's UBC reclaim makes "available" look generous, so a guard keyed on the live
// figure alone would invert exactly when needed, while min() cannot make the ceiling scarcer than the static bound already
// allowed. live<=0 (no probe, or unknown) leaves the static ceiling; ram==0 (an unreadable hw.memsize) returns 0, which every
// caller reads as "unknown".
func metalMemoryCeiling(ram uint64) int64 {
	if metalMemoryCeilingForTest > 0 {
		return metalMemoryCeilingForTest
	}
	if ram == 0 {
		return 0
	}
	ceiling := metalStaticCeiling(ram)
	if live := metalLiveAvailable(); live > 0 && live < ceiling {
		return live
	}
	return ceiling
}

// metalMemoryCeilingForTest, when positive, is the ceiling metalMemoryCeiling returns, so a test can size a load against it to
// the byte. Zero in production.
var metalMemoryCeilingForTest int64

// metalStepHeadroom is the bytes the memory guard leaves beside a build of `slots` KV slots, read before any buffer exists for
// metalKVSlots' reason. The batched step's scratch must fit in it, or buildBatch declines the step for this load and the load
// goes ahead. The guard's override leaves it unlimited; an unreadable RAM size leaves -1, which declines the step.
func metalStepHeadroom(m *decoder.Model, plan metalCtxPlan, slots int) int64 {
	if modelKnob(m, "GOINFER_NO_RESIDENT_MEM_GUARD") != "" {
		return math.MaxInt64
	}
	ram, err := physicalRAM()
	if (err != nil || ram == 0) && metalMemoryCeilingForTest <= 0 {
		return -1
	}
	return metalMemoryCeiling(ram) - residentNeedBytes(m, plan) - int64(max(slots, 1)-1)*metalSlotBytes(m, plan)
}

// metalMoESlotsRequest is the resolved expert-slot request as a string, ready for the strconv.Atoi and validation that moe.go and
// gemma4_moe.go do at their dispatch-building call sites. `--moe-cache-slots` (Model.MoECacheSlotsRequest, the flag CUDA's
// auto-cap also reads) wins; GOINFER_METAL_MOE_SLOTS is a deprecated fallback. "" means unset: every expert resident.
func metalMoESlotsRequest(m *decoder.Model, plan metalCtxPlan) string {
	if n := m.MoECacheSlotsRequest(); n > 0 {
		return strconv.Itoa(n)
	}
	// --moe-cache-experts without --moe-cache-slots auto-sizes the slots from memory (autoMoESlots), as CUDA does against free VRAM.
	if m.MoECacheExperts() {
		return strconv.Itoa(autoMoESlots(m, plan))
	}
	return modelKnob(m, "GOINFER_METAL_MOE_SLOTS")
}

// moeTopK is the model's routed-experts-per-token count, the floor autoMoESlots clamps above: fewer slots than top-k cannot hold
// one token's routed set (buildResident enforces the same on an explicit --moe-cache-slots,
// TestMoESlotsViaOptions_belowTopKRefusesWithNumbers). Gemma 4's MoE and the generic MoE families keep top-k in different Config
// fields, so it branches on the shape.
func moeTopK(m *decoder.Model) int {
	if m.BackendView().HasGemma4MoE() {
		if k := m.Config().TopKExperts; k > 0 {
			return k
		}
		return 1
	}
	if mo, ok := m.BackendView().MoE(); ok && mo.TopK > 0 {
		return mo.TopK
	}
	return 1
}

// autoMoESlotsMax is the per-layer slot ceiling the auto-sizer requests even when memory allows more; N=64 is the default
// the expert-streaming measurements recommend. Raise it only if a later measurement moves that.
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
func autoMoESlots(m *decoder.Model, plan metalCtxPlan) int {
	topK := moeTopK(m)
	perSlot := m.ResidentWeightBytesPaged(2) - m.ResidentWeightBytesPaged(1)
	ram, err := physicalRAM()
	if err != nil || ram == 0 {
		return max(topK, autoMoESlotsMax)
	}
	needFixed := m.ResidentWeightBytesPaged(1) - perSlot + m.ResidentHostCopyBytes(1) + residentKVBytes(m, plan) // ResidentHostCopyBytes(>0) leaves paged experts out
	return autoMoESlotsForBudget(topK, perSlot, needFixed, metalMemoryCeiling(ram))
}

// metalMoESlotsFromEnv is the guard's reader of metalMoESlotsRequest: an int, where an invalid or unset value means "assume
// unpaged" rather than an error (buildResident still validates and declines a bad one). The name predates the option; it reads
// the same resolved request.
func metalMoESlotsFromEnv(m *decoder.Model, plan metalCtxPlan) int {
	n, err := strconv.Atoi(metalMoESlotsRequest(m, plan))
	if err != nil || n <= 0 {
		return 0
	}
	return n
}

// residentKVBytes is the resident KV cache's footprint at the resolved ctx cap. It runs before buildResident, so it resolves the
// cap itself; on an error (an explicit ctx above metalCtxCapMax) it takes metalCtxCapMax, a safe over-estimate for a guard whose
// job is not to under-count, since BuildResident reports that error on its own, earlier. The small per-model buffers (mq, gu,
// logits, ...) are a few hundred KB each and round to nothing beside KV.
func residentKVBytes(m *decoder.Model, plan metalCtxPlan) int64 {
	ctxCap, err := resolveMetalCtxCap(m, plan)
	if err != nil {
		ctxCap = metalCtxCapMax
	}
	// decoder.Model.ResidentKVBytes is the one definition shared with Plan("metal"), so `fit` and this guard price KV identically:
	// f16, or int8 plus per-head scales, every attention layer at the full ctx padded
	// to 8, no Gated-DeltaNet layer.
	return m.ResidentKVBytes("metal", ctxCap, false, false)
}

// residentNeedBytes is the byte count residentMemoryDecline judges against, split out so a test can drive it without real RAM. It
// prices the paged weight estimate when a slot count is requested (metalMoESlotsFromEnv), not the every-expert-resident figure,
// and adds Metal's host copy (the quantized host WeightMat stays beside the re-packed device buffer; genuinely paged experts have
// none, so the host-copy addend is asked at the same slot count) and KV (residentKVBytes).
func residentNeedBytes(m *decoder.Model, plan metalCtxPlan) int64 {
	ctxCap, err := resolveMetalCtxCap(m, plan)
	if err != nil {
		ctxCap = metalCtxCapMax
	}
	// decoder.Model.ResidentNeedBytes is also what Plan("metal").NeedBytes() is built from, so `fit` and this guard are the same
	// number by construction. The rest is what a companion claims beside the resident (a vision tower priced by serve's
	// towerReserve, a drafter): Metal's device memory is the same RAM.
	return m.ResidentNeedBytes("metal", metalMoESlotsFromEnv(m, plan), ctxCap, false, false) +
		m.ExtraResidentBytes() + m.ExtraResidentKVPerPosition()*int64(ctxCap)
}

// residentFitsMemory reports whether this model's weights fit the machine, declining loudly when
// they do not. True (proceed) whenever the answer is unknown — an unreadable hw.memsize or a
// model reporting zero bytes must not silently disable residency for everyone.
func residentFitsMemory(m *decoder.Model, plan metalCtxPlan) bool {
	return residentMemoryDecline(m, plan) == ""
}

// metalKVSlots is how many resident KV slots buildResident allocates for m:
// the model's request (decoder.Model.ResidentKVSlotsRequestFor("metal"), 1 for a family with recurrent state that
// program.Metal does not keep per slot), reduced while the extra slots would take the build over the memory guard's budget.
// The first slot is the build's own KV, which residentMemoryDecline prices; each further one costs metalSlotBytes. The guard's own override
// (GOINFER_NO_RESIDENT_MEM_GUARD) allocates the request as asked; an unreadable RAM size allocates one.
func metalKVSlots(m *decoder.Model, plan metalCtxPlan) int {
	n := metalKVSlotsWanted(m)
	if asked := m.ResidentKVSlotsRequestFor("metal"); n < asked {
		logs.Infof("metal: %d resident KV slots (Metal's default; %d asked by default, -kv-sessions N keeps N if memory allows)",
			n, asked)
	}
	if n <= 1 {
		return 1
	}
	if modelKnob(m, "GOINFER_NO_RESIDENT_MEM_GUARD") != "" {
		return n
	}
	ram, err := physicalRAM()
	if err != nil || ram == 0 {
		return 1
	}
	budget, need := metalMemoryCeiling(ram), residentNeedBytes(m, plan)+metalRaisedTableBytes(m, plan, n)
	got := mem.KVSlotsWithin(n, budget, need, metalSlotBytes(m, plan))
	// An unpinned slot count gives up a slot for the batched step's scratch, since a default slot is worth less than batched prefill,
	// but only where the step then fits. A count the caller asked for keeps its slots, and the step declines if it no longer fits.
	if reserve := metalStepScratchReserve(m, plan); m.ResidentKVSlotsIsDefault() && reserve > 0 && need+reserve <= budget {
		if within := mem.KVSlotsWithin(n, budget-reserve, need, metalSlotBytes(m, plan)); within < got {
			logs.Infof("metal: %d resident KV slots of %d by default, which leaves room for the batched step's %.1f MB of scratch; -kv-sessions N keeps N if memory allows",
				within, n, float64(reserve)/(1<<20))
			return within
		}
	}
	if got < n {
		logs.Infof("metal: %d resident KV slots of %d requested — each costs %.0f MB (KV at the resident context, and any recurrent state), and the memory guard allows %d",
			got, n, float64(metalSlotBytes(m, plan))/(1<<20), got)
	}
	return got
}

// metalRaisedTableBytes is what n slots' block tables add at one conversation's raised cap (decoder.Model.MetalKVTableRaiseBytes), priced
// for the most slots asked: fewer raise the cap less.
func metalRaisedTableBytes(m *decoder.Model, plan metalCtxPlan, n int) int64 {
	ctxCap, err := resolveMetalCtxCap(m, plan)
	if err != nil {
		ctxCap = metalCtxCapMax
	}
	return m.MetalKVTableRaiseBytes(n, ctxCap)
}

// metalSlotBytes is what each KV slot after the first costs: its KV at the resident context, and its own recurrent state
// (metalRecurrentSlotBytes, 0 for a model without a recurrent mixer).
func metalSlotBytes(m *decoder.Model, plan metalCtxPlan) int64 {
	ctxCap, err := resolveMetalCtxCap(m, plan)
	if err != nil {
		ctxCap = metalCtxCapMax
	}
	return m.MetalKVSlotBytes(ctxCap) + metalRecurrentSlotBytes(m)
}

// metalKVSlotsWanted is the slot count the build asks memory for (decoder.Model.MetalKVSlotsWanted). metalKVSlots grants as
// many of them as fit.
func metalKVSlotsWanted(m *decoder.Model) int { return m.MetalKVSlotsWanted() }

// metalDefaultKVSlots is how many resident KV slots Metal keeps when the count asked is the caller's default
// (decoder.Options.ResidentKVSlotsDefault: serve without -kv-sessions). Every slot's KV is resident from the first token, so
// the default 4 costs memory on every load. Two keep the batched step (it needs 2 slots) and a second conversation's prefix; a
// third and fourth concurrent client lose their own slot and their batching. -kv-sessions N still asks for N
// (docs/audit-metal-2026-09-30.md).
const metalDefaultKVSlots = decoder.MetalDefaultKVSlots

// residentMemoryDeclinePrefix opens every memory-guard decline, so a caller can tell "this machine does not have the
// memory right now" from a feature or admission decline (the resident parity tests skip on the first and fail on the
// second; skipIfMemoryDeclined).
const residentMemoryDeclinePrefix = "the resident build needs "

// residentMemoryDecline is residentFitsMemory with its reason — "" when the build fits — so BuildResident
// can hand it to the load path as a typed decline (decoder.DeclineResident) instead of printing it.
// residentBudget is the memory guard's budget for this machine (metalMemoryCeiling of physical RAM), and false when RAM is unreadable.
func residentBudget() (int64, bool) {
	ram, err := physicalRAM()
	if err != nil || ram == 0 {
		return 0, false
	}
	return metalMemoryCeiling(ram), true
}

func residentMemoryDecline(m *decoder.Model, plan metalCtxPlan) string {
	if modelKnob(m, "GOINFER_NO_RESIDENT_MEM_GUARD") != "" {
		return ""
	}
	need := residentNeedBytes(m, plan)
	if need <= 0 {
		return "" // nothing to compare against; not a reason to refuse
	}
	// The GPU's own limit: the Metal allocation (the need less the host copy, which is ordinary RAM) against the device's
	// recommendedMaxWorkingSetSize (docs/measurements/hostmem-available-2026-10/README.md).
	if wsCap := metalWorkingSetCap(); wsCap > 0 {
		if alloc := need - m.ResidentHostCopyBytes(metalMoESlotsFromEnv(m, plan)); alloc > wsCap {
			return fmt.Sprintf(residentMemoryDeclinePrefix+"%s of Metal allocations (weights, KV and what a companion claims), over this GPU's "+
				"recommendedMaxWorkingSetSize of %s. Override with GOINFER_NO_RESIDENT_MEM_GUARD=1 if this machine really fits it.",
				mem.GiB(alloc), mem.GiB(wsCap))
		}
	}
	ram, err := physicalRAM()
	if err != nil || ram == 0 {
		return ""
	}
	// The same ceiling the registered "metal" memory probe reports to decoder.Model.Plan.
	budget := metalMemoryCeiling(ram)
	if need <= budget {
		return ""
	}
	// A model with routed experts that does not fit resident may still fit paged: the decline names how to reach it.
	moeHint := ""
	if m.BackendView().HasGemma4MoE() {
		moeHint = " This model routes MoE experts — try --moe-cache-experts (Metal pages them; auto-sizes the slot count from free RAM) or --moe-cache-slots N to pick one yourself."
	} else if _, ok := m.BackendView().MoE(); ok {
		moeHint = " This model routes MoE experts — try --moe-cache-experts (Metal pages them; auto-sizes the slot count from free RAM) or --moe-cache-slots N to pick one yourself."
	}
	tightened := ""
	if staticCeiling := metalStaticCeiling(ram); budget < staticCeiling {
		tightened = fmt.Sprintf(" (this machine's live-available memory tightened the static %.0f%%/%s ceiling further)",
			mem.WeightsPercent(), mem.GiB(staticCeiling))
	}
	return fmt.Sprintf(residentMemoryDeclinePrefix+"%s (weights, host copy and KV), over %.0f%% of %s RAM "+
		"(budget %s%s). Metal wires the pages it touches, so loading this would page to swap "+
		"exhaustion rather than run. Override with GOINFER_NO_RESIDENT_MEM_GUARD=1 if this machine really fits it.%s",
		mem.GiB(need), mem.WeightsPercent(), mem.GiB(ram), mem.GiB(budget), tightened, moeHint)
}

func (b *metalBackend) Close() error {
	if b.resident != nil {
		return b.resident.Close()
	}
	return nil
}

// metalFastPrefillFloor is the prompt-length floor (whole prompt = startPos+M) below which the f16-MMA batched pass declines and
// the sequential per-token loop runs instead (or, on a resident with the batched step, the exact step: metalStepPrefillCeiling).
// It is the lowest K at which the §3.2 pooled fidelity gate passed and the pass beat the sequential loop; below it no decision
// cell has passed. Lowering it needs that gate at the new K (docs/measurements/metal-prefill-floor-2026-09-20.md,
// ). GOINFER_METAL_FAST_PREFILL_FLOOR overrides it, 0 = no floor.
const metalFastPrefillFloor = 16

// metalStepPrefillCeiling bounds the batched step's range: a resident with the step runs a prompt that ends below this many
// tokens as exact decode rows on it rather than the pass. The step is bit-identical to the sequential loop and the pass
// is not, so it keeps every length where it is not slower (the step is faster at K = 16, the pass at K = 32, on both measured
// models).
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
// attention_prefill) runs in place of the exact scalar kernel. Default on
// (docs/measurements/prefill-l2-metal-fused-attn-2026-09-09.md §5); GOINFER_METAL_FUSED_ATTENTION=0 or --exact-prefill (which
// also covers metalFastPrefillEnabled) opts back to the exact kernel. The fused kernel needs hd%8==0 && hd<=128 (ATTN_MAXHD in
// prefill.go); head dims 256 and 512 take attention_prefill_steel_split under the same flag (exactAttnLayer), and every other
// head dim the exact kernel regardless of it.
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
