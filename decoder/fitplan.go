package decoder

import "fmt"

// Placement is which of task-fit-to-hardware.md's five strategies Plan chose for one backend.
// The enum exists in full even though this phase only ever returns three of them — Phase 1 is
// scoped to the pure decision function ("no behaviour change yet"), and PlacementHostComputedExperts
// is reserved so a later phase (L-01) does not need to rewrite the type.
type Placement int

const (
	PlacementResident Placement = iota
	PlacementExpertCached
	PlacementHostComputedExperts // reserved for L-01 (misses computed on CPU from a pinned host copy); never chosen today
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

// ctxPlanFloor is the smallest context Plan will shrink to before giving up — task-fit-to-
// hardware.md §2's priority order ("shrink context toward a floor of 4096 before moving anything
// [else]"). Same figure decoder/fitguard.go's ctxFloor uses for the CPU staged-load guard, kept
// as its own constant here rather than imported: the two guards are independent by design
// (fitguard.go predates this file and is not being folded into it this phase), and a future
// change to one floor should not silently move the other's.
const ctxPlanFloor = 4096

// PlanRequest is what the user asked for, with every default already resolved by the CALLER
// (task-fit-to-hardware.md §2: "request — what the user asked for, with the defaults filled").
// Plan itself invents no defaults, so a caller (goinfer-chat fit, the startup banner, a table
// test) can see and vary exactly what it asked for rather than a default buried inside the
// function under test.
type PlanRequest struct {
	Ctx       int  // requested context; Plan may shrink it toward ctxPlanFloor unless CtxPinned
	CtxPinned bool // true for an explicit -ctx: refuse (PlacementDecline) rather than silently shrink
	Slots     int  // an explicit --moe-cache-slots / GOINFER_METAL_MOE_SLOTS; 0 means "let Plan choose"

	// KVF16/KVI8 select a lossy KV precision. Both false means f32 (bit-exact, the default per
	// task-fit-to-hardware.md §0: "the plan never selects a lossy... precision to make something
	// fit without saying so... and requiring the flag" — so these must come from an explicit flag
	// the caller read, never be set by Plan on its own initiative).
	KVF16, KVI8 bool

	// ExtraBytes prices whatever the model itself does not know about but will share its device:
	// a block drafter's weights + verify/capture buffers, a vision tower. task-fit-to-hardware.md
	// §2's "every allocation is a term of the plan, including the ones that attach after load" —
	// the concrete example that motivated it (a 26B's expert cache sized before a later --drafter
	// attach grabbed room NewBlockSpec then needed) is exactly what this term exists to prevent
	// PLAN from repeating. capSlots itself is now fixed too (docs/task-gpu-paths-2026-09.md,
	// 2026-09-09): no ResidencyBackend interface change was needed after all — an out-of-band
	// hint on decoder.Model (Options.ExtraResidentBytes / Model.ExtraResidentBytes()) was enough,
	// since cuda/backend.go's BuildResident already receives *Model and can read it directly.
	// resolveCtxCapFit passes the SAME value as this field's own ExtraBytes when it asks Plan for
	// an unpinned ctx, so the two paths (Plan's dry run and the real load) price a drafter
	// identically.
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

	DenseBytes      int64 // decoder.Model.ResidentDenseWeightBytes() — the fixed, never-shrinks term
	ExpertBytesFull int64 // every routed expert, unpaged — reported even when Slots caps what's actually resident
	ExpertBytesUsed int64 // what Slots actually keeps resident (equals ExpertBytesFull when Slots==0/unpaged/dense)
	KVBytes         int64 // at Ctx, this backend's KV precision
	ExtraBytes      int64 // PlanRequest.ExtraBytes, carried through for NeedBytes()/reporting

	FreeBytes int64  // what Plan was told is available on this backend/device
	Reason    string // human-readable justification (admit) or decline message with the numbers
}

// NeedBytes is the total this Plan actually asks the device to hold — DenseBytes + whatever the
// chosen Slots keeps of the experts + KV at the chosen Ctx + the caller's own extra terms. Never
// ExpertBytesFull: that field exists for reporting what was capped, not what is resident.
func (p Plan) NeedBytes() int64 {
	return p.DenseBytes + p.ExpertBytesUsed + p.KVBytes + p.ExtraBytes
}

// kvBytesPerPosition is Plan's own KV-cost formula: per-layer (a per-layer-varying family like
// Gemma 4 is not approximated by one model-level figure — the same reason metal/backend.go's
// residentKVBytes sums per layer rather than using a single kvDim), summed for K+V, at the
// requested precision. Bytes-per-element figures match decoder/fitguard.go's kvBytesPerPosition
// exactly (f32 4.0, f16 2, int8 1.125-including-scale) — kept as an independent constant set
// rather than a shared call, same reasoning as ctxPlanFloor above.
func (m *Model) kvBytesPerPositionAllLayers(f16, i8 bool) int64 {
	perElem := 4.0
	switch {
	case i8:
		perElem = 1.125
	case f16:
		perElem = 2
	}
	_, nLayers, _, _, _, _, _ := m.Dims()
	var perPos float64
	for l := 0; l < nLayers; l++ {
		kvDim := m.KVHeadsAtResident(l) * m.HeadDimAtResident(l)
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

// WebGPUCtxCeiling is the WebGPU backend's fixed per-precision KV-capacity ceiling —
// gpu/residency.go's own ctxCap before any -ctx request lowers it further via min(): 16384
// positions at f32 (the proven 8 GB fit), 32768 at f16 (half the per-token bytes), 65536 at i8
// (a quarter). Shared here, and gpu/residency.go calls THIS function instead of repeating the
// three literals, so the planner's ctx choice and the backend's actual allocation can never
// drift apart — exactly what Phase 3 (task-fit-to-hardware.md §7) needs before admitting webgpu:
// a freeBytes-driven plan alone could pick a context above this fixed ceiling (VRAM allowing),
// which BuildResident would then silently NOT honour (min() keeps the ceiling, and the plan's
// promise would be wrong).
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

// Plan is task-fit-to-hardware.md §2's pure function ("no behaviour change yet [Phase 1] — the
// plan is printed beside today's decision"): given this model, a candidate backend, how many
// bytes are free on it, and what the caller asked for, decide a placement — resident,
// expert-cached, weight-paged, or decline — following §2's priority order (shrink context toward
// ctxPlanFloor first, since it costs no numerics; then cap routed experts into a cache; dense
// weights always stay resident; CPU alone falls to weight-paging rather than ever declining).
//
// No I/O, no side effects, and no defaults invented — see PlanRequest's own doc comment. backend
// is "cuda", "metal", "cpu", or (Phase 3, task-fit-to-hardware.md §7) "webgpu" — admitted now that
// M-32 is fixed (gpu/residency.go: BuildResident declines the same Nemotron/Qwen3.5/MLA +
// KVF16/KVI8 combo Plan declines below, and honours -ctx via the same WebGPUCtxCeiling); an
// unrecognised backend name gets the same GPU-shaped feature-eligibility decline a real one would
// for an unsupported arch, rather than a panic or a silent wrong answer — enforced by the
// ResidentEligible(m.w.arch, backend) check below, not by MissingResidentFeatures alone (M-08:
// that check alone passed a feature-free arch on ANY name, including an unregistered one).
func (m *Model) Plan(backend string, freeBytes int64, req PlanRequest) Plan {
	p := Plan{Backend: backend, FreeBytes: freeBytes, ExtraBytes: req.ExtraBytes}

	if backend != "cpu" {
		if missing := m.MissingResidentFeatures(ResidentBackendFeatures(backend)); len(missing) > 0 {
			p.Placement = PlacementDecline
			p.Reason = fmt.Sprintf("%s does not implement %v for this architecture — use the staged/CPU path", backend, missing)
			return p
		}
		// M-08 (docs/audit-2026-09-10.md): MissingResidentFeatures alone is not admission —
		// ResidentEligible additionally checks that the backend is a REGISTERED one at all (an
		// unrecognised name plus a feature-free arch made the check above vacuously pass, since
		// missingFeatures(nil-required, nil-implemented) is empty), that the arch's own forward
		// is bridged to the resident runner (decodeRunnerEligible), that its MoE router fits the
		// backend's fixed-size scoreboard, and that per-layer attention geometry (Gemma 4's split
		// head_dim) is implemented. Missing any of those reported "resident" for Llama-4/cuda,
		// dense Gemma-4/webgpu, Kimi-K2/metal, and any unrecognised backend name.
		if !ResidentEligible(m.w.arch, backend) {
			p.Placement = PlacementDecline
			p.Reason = fmt.Sprintf("%s: not eligible for this architecture's resident path (unrecognised backend, "+
				"an own-forward shape the runner does not bridge, MoE router capacity, or per-layer attention "+
				"geometry) — use the staged/CPU path", backend)
			return p
		}
	}

	// M-32 (webgpu only): the Nemotron, Qwen3.5 and MLA branches always allocate f32 KV
	// regardless of the flag — mirrors gpu/residency.go's own decline EXACTLY (same three
	// eligibility checks, same combination), so Plan never promises a KV precision BuildResident
	// will not actually honour.
	var ctxCeiling int
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

	p.DenseBytes = m.ResidentDenseWeightBytes()
	nExperts, topK, isMoE := m.moeGeometry()
	if isMoE {
		p.ExpertBytesFull = m.ResidentWeightBytesPaged(0) - p.DenseBytes
	}

	// tryCtx computes KV bytes at ctx and reports whether dense+KV+extra alone (i.e. a model with
	// no experts, or one whose experts are being ignored by THIS attempt) fits — AND, on webgpu,
	// whether ctx is within WebGPUCtxCeiling: bytes fitting is not enough there, since the fixed
	// per-precision cap refuses positions past it regardless of free VRAM (M-32).
	tryCtx := func(ctx int) (kv int64, fits bool) {
		kv = m.kvBytesPerPositionAllLayers(req.KVF16, req.KVI8) * int64(ctx)
		fits = p.DenseBytes+kv+req.ExtraBytes <= freeBytes
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
		perPos := m.kvBytesPerPositionAllLayers(req.KVF16, req.KVI8)
		if perPos <= 0 {
			return req.Ctx, 0, false
		}
		budget := freeBytes - p.DenseBytes - req.ExtraBytes
		fitCtx := int(budget / perPos)
		if ctxCeiling > 0 && fitCtx > ctxCeiling {
			fitCtx = ctxCeiling // the fixed per-precision cap, not a byte budget — never grow past it either
		}
		if fitCtx < ctxPlanFloor {
			return req.Ctx, perPos * int64(req.Ctx), false
		}
		if fitCtx > req.Ctx {
			fitCtx = req.Ctx // never GROW past what was asked
		}
		return fitCtx, perPos * int64(fitCtx), true
	}

	ctx, kv, ctxOK := chooseCtx()
	p.Ctx, p.KVBytes = ctx, kv

	if !ctxOK {
		if ctxCeiling > 0 && req.Ctx > ctxCeiling {
			return p.decline(backend, "context %d exceeds webgpu's fixed %d-position ceiling at this KV precision (M-32) — pass a smaller -ctx or drop --kv-f16/--kv-i8",
				req.Ctx, ctxCeiling)
		}
		return p.decline(backend, "context %d does not fit even at the %d-position floor: dense %.2f GB + KV %.2f GB exceeds %.2f GB free",
			req.Ctx, ctxPlanFloor, gb(p.DenseBytes), gb(kv), gb(freeBytes))
	}

	if !isMoE {
		p.Placement = PlacementResident
		p.Reason = fmt.Sprintf("dense %.2f GB + KV@%d %.2f GB fits %.2f GB free", gb(p.DenseBytes), ctx, gb(kv), gb(freeBytes))
		return p
	}

	// MoE: try fully resident first (every expert, unpaged) at the ctx just chosen.
	if p.DenseBytes+p.ExpertBytesFull+kv+req.ExtraBytes <= freeBytes {
		p.Placement = PlacementResident
		p.ExpertBytesUsed = p.ExpertBytesFull
		p.Reason = fmt.Sprintf("dense %.2f GB + experts %.2f GB + KV@%d %.2f GB fits %.2f GB free",
			gb(p.DenseBytes), gb(p.ExpertBytesFull), ctx, gb(kv), gb(freeBytes))
		return p
	}

	// Cache the routed experts: the largest slot count whose bytes fit beside the fixed terms.
	// A layer's experts are uniform in shape (same assumption ResidentWeightBytesPaged makes), so
	// per-expert bytes = full/nExperts exactly and the slot count is arithmetic, not a search.
	remaining := freeBytes - p.DenseBytes - kv - req.ExtraBytes
	if remaining <= 0 {
		return p.decline(backend, "dense %.2f GB + KV@%d %.2f GB alone exceeds %.2f GB free — no room left for any expert cache",
			gb(p.DenseBytes), ctx, gb(kv), gb(freeBytes))
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
	p.Reason = fmt.Sprintf("dense %.2f GB + %d/%d experts %.2f GB + KV@%d %.2f GB fits %.2f GB free",
		gb(p.DenseBytes), slots, nExperts, gb(p.ExpertBytesUsed), ctx, gb(kv), gb(freeBytes))
	return p
}

// decline finishes Plan on the non-fitting path: CPU alone never declines outright — its whole
// point is that it can always fall back to weight-paging from disk (task-fit-to-hardware.md §1:
// "-weight-cache 0 is auto, ~half of available RAM"), just slower, so a CPU "decline" reads as
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
