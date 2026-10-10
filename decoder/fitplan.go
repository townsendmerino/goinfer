package decoder

import "fmt"

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
const ctxPlanFloor = 4096

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
	return p.DenseBytes + p.ExpertBytesUsed + p.KVBytes + p.HostCopyBytes + p.ExtraBytes
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

// MetalCtxDefault and MetalCtxCeiling are the Metal backend's resident KV capacity: what an unpinned load allocates, and the most an
// explicit -ctx may ask for. metal/model.go takes its constants from these, so Plan("metal") plans exactly what Metal allocates, as
// WebGPUCtxCeiling does for webgpu.
const (
	MetalCtxDefault = 4096
	MetalCtxCeiling = 32768
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

	if backend != "cpu" {
		if missing := m.MissingResidentFeatures(ResidentBackendFeatures(backend)); len(missing) > 0 {
			p.Placement = PlacementDecline
			p.Reason = fmt.Sprintf("%s does not implement %v for this architecture — use the staged/CPU path", backend, missing)
			return p
		}
		// MissingResidentFeatures alone is not admission: ResidentEligible also checks that the backend is a registered one (an unrecognised name
		// plus a feature-free arch passes the check above vacuously), that the arch's own forward is bridged to the resident runner
		// (decodeRunnerEligible), that its MoE router fits the backend's fixed-size scoreboard, and per-layer attention geometry.
		if !ResidentEligible(m.w.arch, backend) {
			p.Placement = PlacementDecline
			p.Reason = fmt.Sprintf("%s: not eligible for this architecture's resident path (unrecognised backend, "+
				"an own-forward shape the runner does not bridge, MoE router capacity, or per-layer attention "+
				"geometry) — use the staged/CPU path", backend)
			return p
		}
	}

	// webgpu only: the Nemotron, Qwen3.5 and MLA branches always allocate f32 KV regardless of the flag. This mirrors gpu/residency.go's own
	// decline exactly (same three eligibility checks, same combination), so Plan never promises a KV precision BuildResident will not honour.
	var ctxCeiling int
	if backend == "metal" {
		// An unpinned Metal load allocates MetalCtxDefault positions however much memory there is; only an explicit -ctx reaches past it, up to
		// MetalCtxCeiling, above which Metal refuses.
		ctxCeiling = MetalCtxCeiling
		if !req.CtxPinned {
			ctxCeiling = MetalCtxDefault
		}
	}
	if backend == "webgpu" {
		ctxCeiling = WebGPUCtxCeiling(req.KVF16, req.KVI8)
		if req.KVF16 || req.KVI8 {
			_, _, _, _, _, _, _, _, mlaOK := m.MLAResidentParams()
			_, _, _, _, _, _, dnetOK := m.Qwen35ResidentParams()
			_, _, _, _, _, _, _, nemoOK := m.NemotronResidentParams()
			if family := map[bool]string{true: "nemotron"}[nemoOK] + map[bool]string{true: "qwen3_5"}[dnetOK] +
				map[bool]string{true: "mla"}[mlaOK]; family != "" {
				flag := map[bool]string{true: "--kv-i8"}[req.KVI8] + map[bool]string{true: "--kv-f16"}[req.KVF16]
				p.Placement = PlacementDecline
				p.Reason = fmt.Sprintf("webgpu: %s does not implement %s KV (only the generic GQA path does) — drop the flag to plan the resident path", family, flag)
				return p
			}
		}
	}

	p.DenseBytes = m.ResidentDenseWeightBytesFor(backend) // less what backend keeps host-side (CUDA: an untied embedding table)
	nExperts, topK, isMoE := m.moeGeometry()
	if isMoE {
		p.ExpertBytesFull = m.ResidentWeightBytesPaged(0) - m.ResidentDenseWeightBytes()
	}

	// tryCtx computes KV bytes at ctx and reports whether dense+KV+extra alone (a model with no experts, or one whose experts this attempt
	// ignores) fits, and on webgpu whether ctx is within WebGPUCtxCeiling: bytes fitting is not enough there, since the fixed per-precision cap
	// refuses positions past it regardless of free VRAM. hostFixed is the host copy that stays whatever the expert cache does: the dense
	// weights' (paged experts stream and hold no host copy, so slots=1 prices exactly that). Zero off Metal.
	hostFixed := m.residentHostCopyFor(backend, 1)
	kvAt := func(ctx int) int64 { return m.ResidentKVBytes(backend, ctx, req.KVF16, req.KVI8) }
	tryCtx := func(ctx int) (kv int64, fits bool) {
		kv = kvAt(ctx)
		fits = p.DenseBytes+hostFixed+kv+req.ExtraBytes <= freeBytes
		if ctxCeiling > 0 && ctx > ctxCeiling {
			fits = false
		}
		return kv, fits
	}

	// chooseCtx applies the priority order's first step: the requested ctx if it fits (with dense
	// weights and extras, ignoring experts for now — the CHEAPEST possible fixed-term check, since
	// if even this fails no expert cap will save it), else the largest multiple-free ctx down to
	// ctxPlanFloor. Pinned requests never shrink. KV bytes are exactly linear in ctx (same
	// reasoning fitguard.go's smallerFittingContext relies on), so this is arithmetic, not a search.
	chooseCtx := func() (ctx int, kv int64, ok bool) {
		if kv, fits := tryCtx(req.Ctx); fits {
			return req.Ctx, kv, true
		}
		if req.CtxPinned {
			kv, _ := tryCtx(req.Ctx)
			return req.Ctx, kv, false
		}
		if backend == "cuda" && m.windowedKV {
			// KV bytes are affine in ctx here, not linear: a windowed layer flattens at window+slack. The linear rate below prices every
			// layer at its full-context rate and would shrink the context needlessly, so search for the largest context that fits
			// (tryCtx is monotone in ctx). Same floor and same "never grow past what was asked" as the arithmetic path.
			if _, fits := tryCtx(ctxPlanFloor); !fits {
				return req.Ctx, kvAt(req.Ctx), false
			}
			lo, hi := ctxPlanFloor, req.Ctx // fits(lo), !fits(hi)
			for hi-lo > 1 {
				if mid := lo + (hi-lo)/2; func() bool { _, ok := tryCtx(mid); return ok }() {
					lo = mid
				} else {
					hi = mid
				}
			}
			return lo, kvAt(lo), true
		}
		// Metal allocates KV in 8-position steps, so evaluate the rate over one step and keep the
		// shrunk ctx on that grid: then kvAt(fitCtx) is exactly perPos*fitCtx, what Metal allocates.
		step := 1
		if backend == "metal" {
			step = metalKVPad
		}
		perPos := kvAt(step) / int64(step)
		if perPos <= 0 {
			return req.Ctx, 0, false
		}
		budget := freeBytes - p.DenseBytes - hostFixed - req.ExtraBytes
		fitCtx := int(budget/perPos) / step * step
		if ctxCeiling > 0 && fitCtx > ctxCeiling {
			fitCtx = ctxCeiling // the fixed per-precision cap, not a byte budget — never grow past it either
		}
		if fitCtx < ctxPlanFloor {
			return req.Ctx, kvAt(req.Ctx), false
		}
		if fitCtx > req.Ctx {
			fitCtx = req.Ctx // never GROW past what was asked
		}
		return fitCtx, kvAt(fitCtx), true
	}

	ctx, kv, ctxOK := chooseCtx()
	p.Ctx, p.KVBytes = ctx, kv

	if !ctxOK {
		if ctxCeiling > 0 && req.Ctx > ctxCeiling {
			if backend == "metal" {
				return p.decline(backend, "context %d exceeds metal's %d-position ceiling — pass a smaller -ctx", req.Ctx, ctxCeiling)
			}
			return p.decline(backend, "context %d exceeds webgpu's fixed %d-position ceiling at this KV precision (M-32) — pass a smaller -ctx or drop --kv-f16/--kv-i8",
				req.Ctx, ctxCeiling)
		}
		return p.decline(backend, "context %d does not fit even at the %d-position floor: dense %.2f GB%s + KV %.2f GB exceeds %.2f GB free",
			req.Ctx, ctxPlanFloor, gb(p.DenseBytes), hostCopyNote(hostFixed), gb(kv), gb(freeBytes))
	}

	if !isMoE {
		p.Placement = PlacementResident
		p.HostCopyBytes = hostFixed
		p.Reason = fmt.Sprintf("dense %.2f GB%s + KV@%d %.2f GB fits %.2f GB free", gb(p.DenseBytes), hostCopyNote(hostFixed), ctx, gb(kv), gb(freeBytes))
		return p
	}

	// MoE: try fully resident first (every expert, unpaged) at the ctx just chosen. Unpaged experts
	// carry a host copy too on Metal (slots=0 prices it).
	if hostAll := m.residentHostCopyFor(backend, 0); p.DenseBytes+p.ExpertBytesFull+hostAll+kv+req.ExtraBytes <= freeBytes {
		p.Placement = PlacementResident
		p.ExpertBytesUsed = p.ExpertBytesFull
		p.HostCopyBytes = hostAll
		p.Reason = fmt.Sprintf("dense %.2f GB + experts %.2f GB%s + KV@%d %.2f GB fits %.2f GB free",
			gb(p.DenseBytes), gb(p.ExpertBytesFull), hostCopyNote(hostAll), ctx, gb(kv), gb(freeBytes))
		return p
	}

	// Cache the routed experts: the largest slot count whose bytes fit beside the fixed terms.
	// A layer's experts are uniform in shape (same assumption ResidentWeightBytesPaged makes), so
	// per-expert bytes = full/nExperts exactly and the slot count is arithmetic, not a search.
	remaining := freeBytes - p.DenseBytes - hostFixed - kv - req.ExtraBytes
	if remaining <= 0 {
		return p.decline(backend, "dense %.2f GB%s + KV@%d %.2f GB alone exceeds %.2f GB free — no room left for any expert cache",
			gb(p.DenseBytes), hostCopyNote(hostFixed), ctx, gb(kv), gb(freeBytes))
	}
	perExpert := p.ExpertBytesFull / int64(nExperts)
	slots := int(remaining / perExpert)
	if req.Slots > 0 && req.Slots < slots {
		slots = req.Slots // an explicit, smaller request is honoured even if more would fit
	}
	if slots > nExperts {
		slots = nExperts
	}
	if slots < topK {
		return p.decline(backend, "expert-cached needs at least top-k=%d slots/layer (%.2f GB) but only %.2f GB is free beside dense+KV",
			topK, gb(int64(topK)*perExpert), gb(remaining))
	}

	p.Placement = PlacementExpertCached
	p.Slots = slots
	p.ExpertBytesUsed = int64(slots) * perExpert
	p.HostCopyBytes = hostFixed
	p.Reason = fmt.Sprintf("dense %.2f GB%s + %d/%d experts %.2f GB + KV@%d %.2f GB fits %.2f GB free",
		gb(p.DenseBytes), hostCopyNote(hostFixed), slots, nExperts, gb(p.ExpertBytesUsed), ctx, gb(kv), gb(freeBytes))
	return p
}

// hostCopyNote names Metal's host-copy term in a Plan reason when there is one.
func hostCopyNote(b int64) string {
	if b <= 0 {
		return ""
	}
	return fmt.Sprintf(" + host copy %.2f GB", gb(b))
}

// decline finishes Plan on the non-fitting path. CPU alone never declines outright: it can always fall back to weight-paging from disk
// (docs/tasks/task-fit-to-hardware.md §1: -weight-cache 0 is auto, about half of available RAM), just slower, so a CPU decline reads as
// weight-paged with the same numbers instead of a hard refusal.
func (p Plan) decline(backend, format string, args ...any) Plan {
	p.Reason = fmt.Sprintf(format, args...)
	if backend == "cpu" {
		p.Placement = PlacementWeightPaged
	} else {
		p.Placement = PlacementDecline
	}
	return p
}

func gb(b int64) float64 { return float64(b) / (1 << 30) }
