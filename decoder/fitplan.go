package decoder

import (
	"fmt"

	"github.com/townsendmerino/goinfer/internal/mem"
)

// Placement is which of docs/tasks/task-fit-to-hardware.md's five strategies Plan chose for one backend. PlacementHostComputedExperts is
// reserved and never chosen, so adding it later does not rewrite the type.
type Placement int

const (
	PlacementResident Placement = iota
	PlacementExpertCached
	PlacementHostComputedExperts // reserved: misses computed on CPU from a pinned host copy; never chosen
	PlacementWeightPaged
	PlacementDecline
)

func (p Placement) String() string {
	switch p {
	case PlacementResident:
		return "resident"
	case PlacementExpertCached:
		return "expert-cached"
	case PlacementHostComputedExperts:
		return "host-computed-experts"
	case PlacementWeightPaged:
		return "weight-paged"
	case PlacementDecline:
		return "decline"
	default:
		return fmt.Sprintf("Placement(%d)", int(p))
	}
}

// ctxPlanFloor is the smallest context Plan will shrink to before giving up (docs/tasks/task-fit-to-hardware.md §2: shrink context
// toward a floor before moving anything else). It is not fitguard.go's ctxFloor, the CPU staged-load guard's own floor: the two guards are
// independent by design, so a change to one floor must not silently move the other's.
const ctxPlanFloor = mem.PlanCtxFloor

// PlanRequest is what the user asked for, with every default already resolved by the CALLER (docs/tasks/task-fit-to-hardware.md §2).
// Plan itself invents no defaults, so a caller (goinfer-chat fit, the startup banner, a table test) can see and vary exactly what it asked
// for rather than a default buried inside the function under test.
type PlanRequest struct {
	Ctx       int  // requested context; Plan may shrink it toward ctxPlanFloor unless CtxPinned
	CtxPinned bool // true for an explicit -ctx: refuse (PlacementDecline) rather than silently shrink
	Slots     int  // an explicit --moe-cache-slots / GOINFER_METAL_MOE_SLOTS; 0 means "let Plan choose"

	// KVF16/KVI8 select a lossy KV precision; both false means f32 (bit-exact, the default). They must come from an explicit flag the caller
	// read, never be set by Plan on its own initiative: the plan never selects a lossy precision to make something fit without saying so
	// (docs/tasks/task-fit-to-hardware.md §0).
	KVF16, KVI8 bool

	// ExtraBytes prices whatever the model itself does not know about but will share its device: a block drafter's weights and verify/capture
	// buffers, a vision tower (every allocation is a term of the plan, including the ones that attach after load). Without it an expert cache
	// is sized before a later --drafter attach claims the room NewBlockSpec then needs. The backend learns the same value out of band through
	// Options.ExtraResidentBytes / Model.ExtraResidentBytes(), and resolveCtxCapFit passes it here when it asks Plan for an unpinned ctx, so
	// Plan's dry run and the real load price a drafter identically.
	ExtraBytes int64
}

// Plan is one backend's placement decision, with every number that justified it — so a decline
// message, a dry-run printout, and a unit test assertion all read the same fields rather than
// three independent re-derivations of the same arithmetic.
type Plan struct {
	Backend   string
	Placement Placement
	Ctx       int // the context actually used — may be smaller than PlanRequest.Ctx
	Slots     int // expert-cache slot count actually used; 0 when N/A (dense) or "every expert" (unpaged)

	DenseBytes      int64 // decoder.Model.ResidentDenseWeightBytesFor(backend) — the fixed, never-shrinks term
	ExpertBytesFull int64 // every routed expert, unpaged — reported even when Slots caps what's actually resident
	ExpertBytesUsed int64 // what Slots actually keeps resident (equals ExpertBytesFull when Slots==0/unpaged/dense)
	KVBytes         int64 // at Ctx, as this backend allocates it (ResidentKVBytes)
	HostCopyBytes   int64 // Metal only: the host copy of the resident weights unified memory also holds (ResidentHostCopyBytes; 0 when aliased from a .giw)
	AllocSlackBytes int64 // the backend's allocator rounding over the weight buffers (PlanSlack): CUDA's 2 MiB granularity
	ExtraBytes      int64 // PlanRequest.ExtraBytes, carried through for NeedBytes()/reporting

	FreeBytes int64  // what Plan was told is available on this backend/device
	Reason    string // human-readable justification (admit) or decline message with the numbers
}

// NeedBytes is the total this Plan actually asks the device to hold — DenseBytes + whatever the
// chosen Slots keeps of the experts + KV at the chosen Ctx + (Metal) the host copy + the caller's own
// extra terms. Never ExpertBytesFull: that field exists for reporting what was capped, not what is
// resident. Without ExtraBytes it equals Model.ResidentNeedBytes at the same Slots and Ctx — the
// number Metal's resident guard judges — which TestPlan_metalAgreesWithResidentNeedBytes pins.
func (p Plan) NeedBytes() int64 {
	return p.DenseBytes + p.AllocSlackBytes + p.ExpertBytesUsed + p.KVBytes + p.HostCopyBytes + p.ExtraBytes
}

// kvBytesPerPositionAllLayers is Plan's KV-cost rate: per layer (a per-layer-varying family like Gemma 4 is not approximated by one
// model-level figure, as metal/backend.go's residentKVBytes also sums per layer), K+V, at the requested precision. Bytes per element match
// kvBytesForCtx in decoder/arch.go (f32 4.0, f16 2, int8 1.125 including scale). The per-layer width is Architecture.kvDimAt: zero for a
// linear/mamba/conv mixer layer (no position-indexed K/V), MLA's compressed latent width.
//
// It deliberately does NOT apply kvBytesForCtx's sliding-window cap (kvPositionsAt): this returns a flat per-position rate that the caller
// multiplies by ctx, while a sliding-window layer's cost flattens at SlidingWindow, so capping it needs tryCtx/chooseCtx to search rather
// than multiply, and a sliding-window fixture to verify against. The rate stays exact for families without a window (MLA, DeltaNet/Mamba/
// conv hybrids); a family with both would be overpriced, the safe direction.
func (m *Model) kvBytesPerPositionAllLayers(f16, i8 bool) int64 {
	perElem := 4.0
	switch {
	case i8:
		perElem = 1.125
	case f16:
		perElem = 2
	}
	arch := m.w.arch
	_, nLayers, _, _, _, _, _ := m.Dims()
	var perPos float64
	for l := range nLayers {
		var kvDim int
		if arch != nil {
			kvDim = arch.kvDimAt(l)
		} else {
			kvDim = m.KVHeadsAtResident(l) * m.HeadDimAtResident(l)
		}
		perPos += 2 * float64(kvDim) * perElem // ×2 for K and V
	}
	return int64(perPos)
}

// moeGeometry reads what Plan needs about this model's routed experts without touching anything
// PlanRequest-shaped: total expert count per layer (uniform across layers by construction, same
// assumption ResidentWeightBytesPaged already makes) and the top-k floor below which even one
// token's routed set cannot be resident.
func (m *Model) moeGeometry() (nExperts, topK int, isMoE bool) {
	nE, k, _, _, _, _, _, _, _, _, ok := m.MoEResidentParams()
	if !ok || nE == 0 {
		return 0, 0, false
	}
	return nE, k, true
}

// WebGPUCtxCeiling is the WebGPU backend's fixed per-precision KV-capacity ceiling, gpu/residency.go's ctxCap before any -ctx request
// lowers it further via min(): 16384 positions at f32, 32768 at f16 (half the per-token bytes), 65536 at i8 (a quarter). gpu/residency.go
// calls this function instead of repeating the literals, so the planner's ctx choice and the backend's allocation cannot drift apart: a
// freeBytes-driven plan could otherwise pick a context above the ceiling, which BuildResident would not honour (min() keeps the ceiling),
// and the plan's promise would be wrong.
func WebGPUCtxCeiling(kvF16, kvI8 bool) int {
	switch {
	case kvI8:
		return 65536
	case kvF16:
		return 32768
	default:
		return 16384
	}
}

// MetalCtxDefault, MetalFitDefaultCtx and MetalCtxCeiling are the Metal backend's resident KV capacity: the floor an unpinned load never
// goes under (and what it takes when the fit is off or cannot judge), the candidate an unpinned load asks Plan for (MetalFitCtx: a coding
// agent's first turn with headroom, CUDA's figure), and the most an explicit -ctx may ask for. metal/model.go takes its constants from
// these, so Plan("metal") plans exactly what Metal allocates, as WebGPUCtxCeiling does for webgpu.
const (
	MetalCtxDefault    = 4096
	MetalFitDefaultCtx = 16384
	MetalCtxCeiling    = 32768
)

// Plan is the planner's pure function (docs/tasks/task-fit-to-hardware.md §2): given this model, a candidate backend, how many bytes are
// free on it and what the caller asked for, decide a placement (resident, expert-cached, weight-paged, or decline) in §2's priority order:
// shrink context toward ctxPlanFloor first, since it costs no numerics; then cap routed experts into a cache; dense weights always stay
// resident; CPU alone falls to weight-paging rather than ever declining.
//
// No I/O, no side effects, and no defaults invented (see PlanRequest). backend is "cuda", "metal", "cpu" or "webgpu"; an unrecognised name
// gets the same feature-eligibility decline an unsupported arch would, enforced by the ResidentEligible check below rather than by
// MissingResidentFeatures alone, which passes a feature-free arch on any name. webgpu's declines mirror gpu/residency.go's BuildResident
// (the Nemotron/Qwen3.5/MLA combination with KVF16/KVI8, and WebGPUCtxCeiling).
func (m *Model) Plan(backend string, freeBytes int64, req PlanRequest) Plan {
	p := Plan{Backend: backend, FreeBytes: freeBytes, ExtraBytes: req.ExtraBytes}
	if reason := m.planIneligible(backend, req); reason != "" {
		p.Placement = PlacementDecline
		p.Reason = reason
		return p
	}
	l := m.deviceLoad(backend, req.KVF16, req.KVI8, req.CtxPinned)
	p.DenseBytes, p.ExpertBytesFull, p.AllocSlackBytes = l.DenseBytes, l.ExpertBytesFull, l.FixedSlack
	d := mem.SolvePlacement(l, mem.PlanRequest{Ctx: req.Ctx, CtxPinned: req.CtxPinned, Slots: req.Slots, ExtraBytes: req.ExtraBytes},
		mem.Machine{Free: map[mem.Tier]int64{l.Tier: freeBytes}}, mem.DefaultPolicy())
	p.Placement, p.Ctx, p.Slots = Placement(d.Placement), d.Ctx, d.Slots
	p.ExpertBytesUsed, p.KVBytes, p.HostCopyBytes, p.Reason = d.ExpertBytesUsed, d.KVBytes, d.HostCopyBytes, d.Reason
	return p
}

// planIneligible is why backend has no resident path for this model at all, before any byte is priced ("" when it has one):
// a feature the backend does not implement, the runner's shape, MoE router capacity or per-layer geometry (ResidentEligible),
// and on webgpu a lossy KV flag on a family whose kernels allocate f32 KV regardless (gpu/residency.go's own decline).
func (m *Model) planIneligible(backend string, req PlanRequest) string {
	if backend != "cpu" {
		if missing := m.MissingResidentFeatures(ResidentBackendFeatures(backend)); len(missing) > 0 {
			return fmt.Sprintf("%s does not implement %v for this architecture — use the staged/CPU path", backend, missing)
		}
		// MissingResidentFeatures alone is not admission: an unrecognised name plus a feature-free arch passes it vacuously.
		if !ResidentEligible(m.w.arch, backend) {
			return fmt.Sprintf("%s: not eligible for this architecture's resident path (unrecognised backend, "+
				"an own-forward shape the runner does not bridge, MoE router capacity, or per-layer attention "+
				"geometry) — use the staged/CPU path", backend)
		}
	}
	if backend == "webgpu" && (req.KVF16 || req.KVI8) {
		_, _, _, _, _, _, _, _, mlaOK := m.MLAResidentParams()
		_, _, _, _, _, _, dnetOK := m.Qwen35ResidentParams()
		_, _, _, _, _, _, _, nemoOK := m.NemotronResidentParams()
		if family := map[bool]string{true: "nemotron"}[nemoOK] + map[bool]string{true: "qwen3_5"}[dnetOK] +
			map[bool]string{true: "mla"}[mlaOK]; family != "" {
			flag := map[bool]string{true: "--kv-i8"}[req.KVI8] + map[bool]string{true: "--kv-f16"}[req.KVF16]
			return fmt.Sprintf("webgpu: %s does not implement %s KV (only the generic GQA path does) — drop the flag to plan the resident path", family, flag)
		}
	}
	return ""
}

// deviceLoad is this model on backend as internal/mem's planner prices it: the dense weights less what the backend keeps on the
// host, the routed experts, Metal's host copies, the backend's KV at a context (CUDA at the load's own KV precision, not a flag
// the planner is handed), and the backend's context grid and ceiling: Metal's is MetalCtxCeiling for a pinned request and
// MetalFitDefaultCtx for an unpinned one (MetalFitCtx), WebGPU's its fixed per-precision cap.
func (m *Model) deviceLoad(backend string, kvF16, kvI8, pinned bool) mem.DeviceLoad {
	l := mem.DeviceLoad{Backend: backend, Tier: planTier(backend), DenseBytes: m.ResidentDenseWeightBytesFor(backend), CtxStep: 1,
		FixedSlack: m.planSlack(backend)}
	if nExperts, topK, isMoE := m.moeGeometry(); isMoE {
		l.ExpertBytesFull = m.ResidentWeightBytesPaged(0) - m.ResidentDenseWeightBytes()
		l.NExperts, l.TopK = nExperts, topK
	}
	l.HostFixed = m.residentHostCopyFor(backend, 1)
	l.HostAll = func() int64 { return m.residentHostCopyFor(backend, 0) }
	switch backend {
	case "metal":
		l.CtxStep, l.CtxCeiling = metalKVPad, MetalCtxCeiling
		if !pinned {
			l.CtxCeiling = MetalFitDefaultCtx
		}
	case "webgpu":
		l.CtxCeiling = WebGPUCtxCeiling(kvF16, kvI8)
	case "cuda":
		kvF16 = m.kvF16
		l.CtxSearch = true
	}
	l.KV = func(ctx int) int64 { return m.ResidentKVBytes(backend, ctx, kvF16, kvI8) }
	return l
}

// planTier is the internal/mem tier whose reading is backend's free memory.
func planTier(backend string) mem.Tier {
	switch backend {
	case "cuda", "webgpu":
		return mem.VRAM
	case "metal":
		return mem.Unified
	default:
		return mem.HostAnon
	}
}
