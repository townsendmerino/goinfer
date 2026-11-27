package decoder

import (
	"fmt"

	"github.com/townsendmerino/goinfer/internal/runtime/mem"
	"github.com/townsendmerino/goinfer/internal/runtime/memprice"
)

// ctxPlanFloor is the smallest context Plan will shrink to before giving up (shrink context
// toward a floor before moving anything else). It is not fitguard.go's ctxFloor, the CPU staged-load guard's own floor: the two guards are
// independent by design, so a change to one floor must not silently move the other's.
const ctxPlanFloor = mem.PlanCtxFloor

// kvBytesPerPositionAllLayers is the KV-cost rate Plan prices WebGPU at (ResidentKVBytes; the CPU, CUDA and Metal have their own).
func (m *Model) kvBytesPerPositionAllLayers(f16, i8 bool) int64 {
	if m.w.Arch != nil {
		return memprice.WebGPUKVBytesPerPos(m.w.Arch.NumLayers, m.LayerBuffers, f16, i8)
	}
	perElem := 4.0
	if f16 && !i8 {
		perElem = 2
	}
	bv := m.BackendView()
	nLayers := bv.NumLayers()
	var perPos float64
	for l := range nLayers {
		lv := bv.Layer(l)
		kvDim := lv.KVHeads() * lv.HeadDim()
		if i8 && kvDim > 0 {
			perPos += 2 * (float64(kvDim) + 4*float64(lv.KVHeads()))
			continue
		}
		perPos += 2 * float64(kvDim) * perElem
	}
	return int64(perPos)
}

// moeGeometry reads what Plan needs about this model's routed experts.
func (m *Model) moeGeometry() (nExperts, topK int, isMoE bool) {
	mo := m.BackendView().Spec().MoE
	if mo == nil || mo.NumExperts == 0 {
		return 0, 0, false
	}
	return mo.NumExperts, mo.TopK, true
}

// WebGPUCtxCeiling is the WebGPU backend's fixed per-precision KV-capacity ceiling.
func WebGPUCtxCeiling(kvF16, kvI8 bool) int { return memprice.WebGPUCtxCeilingFor(kvF16, kvI8) }

// MetalCtxDefault, MetalFitDefaultCtx and MetalCtxCeiling are the Metal backend's resident KV capacity bounds.
const (
	MetalCtxDefault    = memprice.MetalCtxDefault
	MetalFitDefaultCtx = memprice.MetalFitDefaultCtx
	MetalCtxCeiling    = memprice.MetalCtxCeiling
)

// Plan is the planner's pure function: given this model, a candidate backend, how many bytes are
// free on it and what the caller asked for, decide a placement (resident, expert-cached, weight-paged, or decline) in §2's priority order:
// shrink context toward ctxPlanFloor first, since it costs no numerics; then cap routed experts into a cache; dense weights always stay
// resident; CPU alone falls to weight-paging rather than ever declining.
//
// No I/O, no side effects, and no defaults invented (see PlanRequest). backend is "cuda", "metal", "cpu" or "webgpu"; an unrecognised name
// declines. Whether backend may run this model resident is the load's own admission (Admit, with this model's facts), so Plan and the
// load agree. webgpu's declines mirror gpu/residency.go's BuildResident (the Nemotron/Qwen3.5/MLA combination with KVF16/KVI8, and
// WebGPUCtxCeiling).
func (m *Model) Plan(backend string, freeBytes int64, req PlanRequest) Plan {
	return m.plan(backend, freeBytes, req, true)
}

// plan is Plan, pricing the backend's allocation rounding (PlanSlack) only when slack is set: a test's before side passes false
// per call, so no other Plan sees it.
func (m *Model) plan(backend string, freeBytes int64, req PlanRequest, slack bool) Plan {
	p := Plan{Backend: backend, FreeBytes: freeBytes, ExtraBytes: req.ExtraBytes}
	if reason := m.planIneligible(backend, req); reason != "" {
		p.Placement = PlacementDecline
		p.Reason = reason
		return p
	}
	l := m.deviceLoad(backend, req)
	if !slack {
		l.FixedSlack = 0
	}
	p.DenseBytes, p.ExpertBytesFull, p.AllocSlackBytes, p.StateBytes = l.DenseBytes, l.ExpertBytesFull, l.FixedSlack, l.StateBytes
	d := mem.SolvePlacement(l, mem.PlanRequest{Ctx: req.Ctx, CtxPinned: req.CtxPinned, Slots: req.Slots, ExtraBytes: req.ExtraBytes},
		mem.Machine{Free: map[mem.Tier]int64{l.Tier: freeBytes}}, mem.DefaultPolicy())
	p.Placement, p.Ctx, p.Slots, p.ScratchBytes = Placement(d.Placement), d.Ctx, d.Slots, d.ScratchBytes
	p.ExpertBytesUsed, p.KVBytes, p.HostCopyBytes, p.Reason = d.ExpertBytesUsed, d.KVBytes, d.HostCopyBytes, d.Reason
	if backend == "webgpu" && memprice.WebGPUVulkan && p.Placement != PlacementDecline {
		// The search priced Vulkan's block rounding on the KV term, where it moves with the context; the plan reports it as slack, so
		// KVBytes is the KV the build allocates (one slot's, as a caller multiplies it) and the total is unchanged.
		r := m.webgpuBlockRounding(p.Ctx, req.KVF16, req.KVI8)
		p.KVBytes -= r
		p.AllocSlackBytes += r
	}
	if ckpt, perSlot := m.checkpointBytes(backend); p.Placement != PlacementDecline {
		if copies := checkpointCopies(checkpointHeadroom(p, freeBytes, req), ckpt, max(req.KVSlots, 1), perSlot); copies > 0 {
			p.CheckpointBytes, p.CheckpointCopies = ckpt, copies
		}
	}
	return p
}

// checkpointHeadroom is the room p's load has for checkpoints, in the build's order: beside the weights, every KV slot and its state. A build allocates them before it sizes the
// expert cache and attaches a companion, and holds room for those in its reserve, so a named reserve stands in for them here rather than adding to them.
func checkpointHeadroom(p Plan, free int64, req PlanRequest) int64 {
	need := p.NeedBytes()
	if req.CheckpointReserve > 0 {
		need -= p.ExtraBytes
		if p.Placement == PlacementExpertCached {
			need -= p.ExpertBytesUsed
		}
	}
	return free - need - int64(max(req.KVSlots, 1)-1)*(p.KVBytes+p.StateBytes) - req.CheckpointReserve
}

// checkpointCopies is how many checkpoints of ckpt bytes fit in headroom: at most one for each of n KV slots on a backend that keeps one per slot, else at most one.
func checkpointCopies(headroom, ckpt int64, n int, perSlot bool) int {
	if ckpt <= 0 || headroom < ckpt {
		return 0
	}
	if !perSlot {
		return 1
	}
	return int(min(headroom/ckpt, int64(n)))
}

// planAdmitDecline is the decoder's admission of this model on backend, with its own facts (Admit): "" when the load would reach
// the backend's build. The CPU always admits; a name with no declared resident features never does.
func (m *Model) planAdmitDecline(backend string) string {
	if backend == "cpu" {
		return ""
	}
	if !residentDeclared(backend) {
		return fmt.Sprintf("%s: no resident path is declared for this backend — use the staged/CPU path", backend)
	}
	f := m.admitFacts()
	f.Backend, f.Residency, f.program = backend, true, m.admitProgram(backend)
	if why := Admit(f).Why; why != "" {
		return fmt.Sprintf("%s: %s — use the staged/CPU path", backend, why)
	}
	return ""
}

// planIneligible is why backend has no resident path for this model at all, before any byte is priced ("" when it has one):
// the decoder's admission (planAdmitDecline), then webgpu's own model-only declines, which it mirrors from
// gpu/residency.go:staticResidentDecline: routed experts of a kind its MoE kernels do not take, and a lossy KV flag on a family
// whose kernels allocate f32 KV regardless.
func (m *Model) planIneligible(backend string, req PlanRequest) string {
	if why := m.planAdmitDecline(backend); why != "" {
		return why
	}
	bv := m.BackendView()
	if backend == "webgpu" {
		// webgpu's stacked-expert kernels take int8 or int4 (gpu/residency.go:staticResidentDecline).
		if k := bv.MoEExpertKind(); k != "" && k != "int8" && k != "int4" {
			return fmt.Sprintf("webgpu: the routed experts are %s, and webgpu's MoE kernels take int8 or int4 — load with --quant int4 "+
				"or int8int8 to plan the resident path", k)
		}
	}
	if backend == "webgpu" && (req.KVF16 || req.KVI8) {
		spec := bv.Spec()
		mlaOK := spec.MLA != nil
		dnetGated := spec.Qwen35 != nil && spec.Qwen35.AttnGate
		nemoOK := spec.Nemotron != nil
		if family := map[bool]string{true: "nemotron"}[nemoOK] + map[bool]string{true: "qwen3_5"}[dnetGated] +
			map[bool]string{true: "mla"}[mlaOK]; family != "" {
			flag := map[bool]string{true: "--kv-i8"}[req.KVI8] + map[bool]string{true: "--kv-f16"}[req.KVF16]
			return fmt.Sprintf("webgpu: %s does not implement %s KV (only the generic GQA path does) — drop the flag to plan the resident path", family, flag)
		}
	}
	return ""
}

// deviceLoad is this model on backend as internal/mem's planner prices it: the dense weights less what the backend keeps on the
// host, one slot's recurrent state (BuildDemand.StateBytes), the routed experts, Metal's host copies, the backend's KV at a context (CUDA at the load's own KV precision, not a flag
// the planner is handed), and the backend's context grid and ceiling: Metal's is MetalCtxCeiling for a pinned request and
// MetalFitDefaultCtx for an unpinned one (MetalFitCtx), WebGPU's its fixed per-precision cap. It reads req's KV precision
// (KVF16, KVI8) and CtxPinned.
func (m *Model) deviceLoad(backend string, req PlanRequest) mem.DeviceLoad {
	kvF16, kvI8, pinned := req.KVF16, req.KVI8, req.CtxPinned
	l := mem.DeviceLoad{Backend: backend, Tier: planTier(backend), DenseBytes: m.ResidentDenseWeightBytesFor(backend), CtxStep: 1,
		FixedSlack: m.planSlack(backend), StateBytes: m.stateBytes(backend)}
	if nExperts, topK, isMoE := m.moeGeometry(); isMoE {
		eb := memprice.ExpertBiasBytes(m.w)
		l.ExpertBytesFull = m.ResidentWeightBytesPaged(0) - m.ResidentDenseWeightBytes() - eb
		switch backend {
		case "cpu":
			_, structs := memprice.CPUStructBytes(m.w)
			l.ExpertBytesFixed = structs + eb
		case "metal":
			l.ExpertBytesFixed = eb
		}
		l.NExperts, l.TopK = nExperts, topK
	}
	l.HostFixed = m.residentHostCopyFor(backend, 1)
	l.HostAll = func() int64 { return m.residentHostCopyFor(backend, 0) }
	switch backend {
	case "metal":
		l.CtxStep, l.CtxCeiling, l.CtxSearch = metalKVPad, MetalCtxCeiling, true // whole blocks beside the pools' null blocks: no one rate
		l.ScratchBytes = m.MetalDecodeScratchBytes(0)
		if l.TopK < l.NExperts { // an expert cache holds at least the top k, under the expert count when it pages
			l.ScratchPagedExtra = m.MetalDecodeScratchBytes(l.TopK) - l.ScratchBytes
		}
		if !pinned {
			l.CtxCeiling = MetalFitDefaultCtx
		}
	case "webgpu":
		l.CtxCeiling = WebGPUCtxCeiling(kvF16, kvI8)
		l.DenseBytes = memprice.WebGPUDenseLine(m.w, m.LayerBuffers)
		if !memprice.WebGPUVulkan {
			break
		}
		// Vulkan's blocks, not the buffers' sum: the device and the kept host block are fixed, and the block
		// rounding moves with the context, so it rides on the KV term and the context is searched.
		l.FixedSlack += memprice.WebGPUVkFixed
		l.CtxSearch = true
		l.KV = func(ctx int) int64 {
			return m.ResidentKVBytes(backend, ctx, kvF16, kvI8) + m.webgpuBlockRounding(ctx, kvF16, kvI8)
		}
		return l
	case "cuda":
		kvF16, kvI8 = m.residentKVFlags(backend)
		l.CtxSearch = true
	}
	l.KV = func(ctx int) int64 { return m.ResidentKVBytes(backend, ctx, kvF16, kvI8) }
	return l
}

func (m *Model) webgpuBlockRounding(ctx int, kvF16, kvI8 bool) int64 {
	slot0, _, dense := memprice.WebGPUDeviceBuffers(m.w, m.LayerBuffers, ctx, kvF16, kvI8)
	var experts int64
	if _, _, isMoE := m.moeGeometry(); isMoE {
		experts = m.ResidentWeightBytesPaged(0) - m.ResidentDenseWeightBytes() - memprice.ExpertBiasBytes(m.w)
	}
	return memprice.WebGPUVkBlocks(slot0) - dense - experts - m.ResidentKVBytes("webgpu", ctx, kvF16, kvI8) - m.stateBytes("webgpu")
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
