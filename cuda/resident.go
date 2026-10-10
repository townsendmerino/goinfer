//go:build cuda

package cuda

import (
	"context"
	"errors"
	"fmt"
	"os"
	"reflect"
	"runtime/debug"
	"sort"
	"strings"
	"sync/atomic"
	"time"
	"unsafe"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/decoder"
)

// cudaResident implements the decoder's resident-decode interfaces; the compile-time checks below keep that true.
var (
	_ decoder.ResidentForward    = (*cudaResident)(nil)
	_ decoder.ResidentGreedy     = (*cudaResident)(nil)
	_ decoder.ResidentMRoPE      = (*cudaResident)(nil)
	_ decoder.VerifyPathReporter = (*cudaResident)(nil)
	_ decoder.ResidentKVSlotter  = (*cudaResident)(nil)
)

// cudaCtxCapDefault is the resident KV capacity in positions when nothing asks for more; the staged path handles
// longer. It is a DEFAULT, not a ceiling: decoder.Options.ResidentContext raises it (see resolveCtxCap). It stays
// 4096 so that a caller who did not ask never allocates deep-KV VRAM: raising it would silently multiply every
// resident model's KV footprint. 4096 is a round conservative figure, not tuned to VRAM headroom; a real card
// holds several times more (docs/code-notes/cuda.md#cudaCtxCapDefault), so raise -ctx and read checkKVFits' own
// error to find a deployment's real ceiling.
const cudaCtxCapDefault = 4096

// ctxCapMarginBytes is the VRAM left free beside weights+KV at the load-time fit check: driver
// overhead plus the transient allocations decode makes (logits readback, split-KV scratch). Same
// margin the C′ expert cache uses, for the same reason.
const ctxCapMarginBytes = 384 << 20

// resolveCtxCap turns a request into the effective resident KV capacity:
//
//	cap = min(model context window, request)   — request 0 ⇒ cudaCtxCapDefault
//
// Clamping to the model's own context window matters because the KV beyond it can never be attended:
// allocating it would burn VRAM to hold positions the RoPE tables and the model's training never
// cover. modelCtx 0 means "unknown" (some architectures do not report one), in which case the request
// stands on its own — the VRAM fit check is then the only guard, which is why that check is not
// optional.
func resolveCtxCap(request, modelCtx int) int {
	if request <= 0 {
		return cudaCtxCapDefault
	}
	if modelCtx > 0 && request > modelCtx {
		return modelCtx
	}
	return request
}

// fitDefaultCtx is the candidate context resolveCtxCapFit asks Plan for on an UNPINNED request: the size of a
// coding agent's turn, not the model's full window (docs/tasks/task-fit-to-hardware.md §8). 8192 refused a real
// first agent request; 16384 holds it with headroom (docs/code-notes/cuda.md#fitDefaultCtx). It is a CANDIDATE,
// not a grant: Plan shrinks it to what the card holds and ctxForSlots shrinks it further until the requested KV
// slots all fit, so a card or model that cannot hold it lands where it always did. goinfer-chat fit's own -ctx
// default (internal/fitcmd/fit.go) uses the same figure so the dry run and the real load agree.
const fitDefaultCtx = 16384

// resolveCtxCapFit is the "fit by default" resident context for CUDA (docs/tasks/task-fit-to-hardware.md Phase 2):
// an UNPINNED load asks Plan for a larger candidate (fitDefaultCtx, clamped to the model's window) instead of a
// flat cudaCtxCapDefault, and uses whatever Plan lands on. Because of the floor checks below that is never less
// than cudaCtxCapDefault, except under a guard pin (below). A caller with m.FitDisabled() (--fit=off), or whose
// free-VRAM probe is unknown, gets exactly resolveCtxCap, so this can only improve on the historical default.
//
// m.MoECacheExperts() opts out the same way. A MoE-cache-experts load has a second, elastic claimant on free
// VRAM, the host-to-VRAM expert-slot cache, whose design is "whatever is left after everything pinned"; every
// byte handed to KV is a byte the expert cache never sees, and growing the context there cut its slots and
// decode speed sharply (docs/code-notes/cuda.md#resolveCtxCapFit). Unlike a drafter's fixed cost it cannot be
// priced into ExtraBytes, so the context must not grow into that leftover at all.
//
// slots is MC1's requested resident KV slot count (cudaKVSlotsRequest). Above 1, the unpinned choice gives up
// context until every slot fits (ctxForSlots), never below cudaCtxCapDefault, or below request when request is
// itself a guard pin under that default.
//
// request > 0 is NOT always a choice. decoder.Model.ResidentContextPinned() tells a genuine -ctx from the
// load-time fit guard auto-pinning a smaller context for an UNREQUESTED load; request still reports that pin
// (ResidentContextRequest's doc), but it is a ceiling already proved safe against host RAM, not the caller's
// decision. A guard pin gets fit-by-default and the slots rule too, with its ceiling and floor both following
// it down, never up: decoder.ctxFloor (2048) can sit under cudaCtxCapDefault, and raising a guard-shrunk
// context back to cudaCtxCapDefault is exactly the regression the guard pinned it to prevent.
func resolveCtxCapFit(m *decoder.Model, request, modelCtx, slots int) int {
	return resolveCtxCapFitSlack(m, request, modelCtx, slots, 0)
}

// resolveCtxCapFitSlack is resolveCtxCapFit for a build that knows its packed weight buffers: allocSlack is the
// device memory the driver's allocation rounding adds to them (packedAllocSlack), which Plan's dense figure -- the
// requested bytes -- cannot see. It is priced as ExtraBytes here and NOT in checkKVFits, whose probe is read after the
// weights are on the device and is already net of it. 0 for a caller without the packed weights.
func resolveCtxCapFitSlack(m *decoder.Model, request, modelCtx, slots int, allocSlack int64) int {
	if (request > 0 && m.ResidentContextPinned()) || m.FitDisabled() || m.MoECacheExperts() {
		return resolveCtxCap(request, modelCtx) // a genuine explicit -ctx is untouched either way
	}
	candidate := fitDefaultCtx
	if modelCtx > 0 && candidate > modelCtx {
		candidate = modelCtx
	}
	floor := cudaCtxCapDefault
	if request > 0 {
		// A guard pin (request>0, not ResidentContextPinned — the branch above already returned
		// otherwise): the ceiling AND the floor both follow it down, never up. Growing past it
		// asks Plan for more than the guard already determined host RAM can hold; flooring above
		// it (at the historical cudaCtxCapDefault) would do the same by a different route.
		if candidate > request {
			candidate = request
		}
		if floor > request {
			floor = request
		}
	}
	if candidate <= floor {
		return floor // nothing to gain from asking Plan: already at or below the real floor
	}
	free, ok := decoder.FreeBytesFor("cuda")
	if !ok {
		return floor // unknown ⇒ the safe floor, never guess
	}
	// Plan's chooseCtx reserves NO margin, but checkKVFits requires an ADDITIONAL ctxCapMarginBytes beyond
	// dense+KV+extra. Subtracting the same margin here, before Plan sees the free bytes, makes the two checks agree:
	// whatever ctx Plan picks already leaves room for it (otherwise an interior choice failed checkKVFits and the
	// whole build declined to CPU-only). Floored at 0 on a tiny probe; Plan declines an unfittable budget cleanly.
	marginedFree := max(free-ctxCapMarginBytes, 0)
	// ExtraBytes prices what attaches after BuildResident (m.ExtraResidentBytes: a --drafter) so the context Plan
	// chooses leaves it room; zero when nothing attaches. The drafter's device K/V is priced at candidate, the widest
	// ctx this call asks of Plan: Plan only shrinks from the value it is given (fitplan.go, "never GROW past what was
	// asked"), so that can over-estimate the real cost but never under-price it.
	extraBytes := m.ExtraResidentBytes() + m.ExtraResidentKVPerPosition()*int64(candidate) + allocSlack
	p := m.Plan("cuda", marginedFree, decoder.PlanRequest{Ctx: candidate, ExtraBytes: extraBytes})
	if p.Placement == decoder.PlacementDecline || p.Ctx < floor {
		return floor // Plan could not confidently improve on the floor
	}
	if slots <= 1 {
		return p.Ctx
	}
	return ctxForSlots(m, marginedFree, extraBytes, p.Ctx, slots, floor)
}

// ctxForSlots is resolveCtxCapFit's answer when MC1's resident KV slots are requested
// (docs/tasks/task-concurrency-2026-09.md MC1 on CUDA): the largest context in [floor, oneSlot] at which Plan fits
// every requested slot (the build's own KV plus slots-1 more copies, priced as ExtraBytes), so an unpinned load
// gives up context before conversations (owner decision). When not even floor holds them all it returns floor and
// checkKVFits clamps the count. An explicit -ctx never reaches here.
//
// floor is cudaCtxCapDefault, unless resolveCtxCapFit's caller passed a guard-pinned request below it: the slots
// rule may shrink a guard-pinned load like an unpinned one, but never past the guard's own safety floor.
//
// KV is exactly linear in the context and Plan's other terms do not grow with it, so "fits" is monotone and a
// binary search finds the edge. Plan's weight figure is conservative, so the result can land lower than the build
// would allow (docs/code-notes/cuda.md#ctxForSlots).
func ctxForSlots(m *decoder.Model, marginedFree, extraBytes int64, oneSlot, slots, floor int) int {
	fits := func(ctx int) bool {
		more := int64(slots-1) * m.ResidentKVBytes("cuda", ctx, false, false) // CUDA's KV is f32 whatever was requested
		p := m.Plan("cuda", marginedFree, decoder.PlanRequest{Ctx: ctx, CtxPinned: true, ExtraBytes: extraBytes + more})
		return p.Placement != decoder.PlacementDecline
	}
	if fits(oneSlot) {
		return oneSlot
	}
	if !fits(floor) {
		fmt.Fprintf(os.Stderr, "cuda: resident context %d (the floor), not %d, for the %d requested KV slots (--kv-sessions); "+
			"the build grants as many as fit there, and an explicit --ctx keeps a longer context with fewer slots\n",
			floor, oneSlot, slots)
		return floor
	}
	lo, hi := floor, oneSlot // fits(lo), !fits(hi)
	for hi-lo > 1 {
		if mid := lo + (hi-lo)/2; fits(mid) {
			lo = mid
		} else {
			hi = mid
		}
	}
	fmt.Fprintf(os.Stderr, "cuda: resident context %d, not %d, so the %d requested KV slots fit (--kv-sessions); an "+
		"explicit --ctx keeps a longer context with fewer slots\n", lo, oneSlot, slots)
	return lo
}

// kvBytesForCap is the device bytes the resident K+V caches occupy at a given capacity: every layer holds K and V
// as f32[cap*kvDim]. docs/benchmarks.md derives its deep-context sizing from this formula.
func kvBytesForCap(cap int, layers []cudaLayer) int64 {
	var perPos, windowed int64
	for i := range layers {
		if layers[i].kvShared {
			continue // aliases its source's cache: no bytes of its own
		}
		if layers[i].isMLA {
			perPos += int64(layers[i].kvDim) * 4 // MLA caches 1 row of latDim floats per pos (not 2x for K+V)
		} else if layers[i].kvWin {
			// a windowed layer holds window+slack positions however long the context (kvwindow.go), and decoder.cudaKVBytes prices the same
			windowed += int64(layers[i].kvDim) * 2 * 4 * int64(min(cap, int(layers[i].window)+kvWindowSlack))
		} else {
			perPos += int64(layers[i].kvDim) * 2 * 4 // K+V
		}
	}
	return perPos*int64(cap) + windowed
}

// kvCapForBytes is the largest capacity in [lo, hi] whose K+V caches occupy at most budget bytes (ok=false when even lo does not fit). kvBytesForCap is
// linear in the capacity unless a layer is windowed, where it is piecewise linear and non-decreasing, so a search finds the edge either way.
func kvCapForBytes(budget int64, lo, hi int, layers []cudaLayer) (c int, ok bool) {
	if hi < lo || kvBytesForCap(lo, layers) > budget {
		return lo, false
	}
	for hi > lo {
		mid := lo + (hi-lo+1)/2
		if kvBytesForCap(mid, layers) <= budget {
			lo = mid
		} else {
			hi = mid - 1
		}
	}
	return lo, true
}

// kvSlotsFit is MC1's slot arithmetic on CUDA (docs/tasks/task-concurrency-2026-09.md): the largest n in [1, want]
// whose n resident KV slots, perSlot bytes each, fit free VRAM beside reserve (a companion attach plus
// ctxCapMarginBytes). Never below 1: the first slot is checkKVFits' own and is refused there, not clamped here.
//
// free is read AFTER the weights are on the device and perSlot counts KV only, so both sides price the same
// thing. Do not compare a weights-inclusive base with that live figure: it counts the weights twice.
func kvSlotsFit(want int, free, perSlot, reserve int64) int {
	n := max(1, want)
	for n > 1 && int64(n)*perSlot+reserve > free {
		n--
	}
	return n
}

// cudaKVSlot is one resident KV slot's per-layer buffers (MC1). A DeltaNet layer has none; an MLA layer has only kc.
type cudaKVSlot struct{ kc, vc []Buffer }

// kvSlotAllocFailAtForTest, when > 0, makes allocKVSlot fail as a device OOM at that slot index (test seam; zero in production).
var kvSlotAllocFailAtForTest int

// allocKVSlot allocates slot s's per-layer K/V buffers as copies of slot 0's shapes. A device allocation failure (aikit's MustBuf panics on OOM, "device allocation failed") is
// recovered into ok=false with the partial slot released, so the caller can keep the slots that fit; any other panic is re-raised.
func (r *cudaResident) allocKVSlot(s, nLayers int) (b cudaKVSlot, ok bool) {
	var own []Buffer
	defer func() {
		if v := recover(); v != nil {
			if !strings.Contains(fmt.Sprint(v), "device allocation failed") {
				panic(v)
			}
			for _, o := range own {
				r.dev.ReleaseBuf(o)
			}
			b, ok = cudaKVSlot{}, false
		}
	}()
	if kvSlotAllocFailAtForTest == s {
		panic("cuda: device allocation failed (test seam): cuMemAlloc_v2: CUDA_ERROR_OUT_OF_MEMORY")
	}
	b = cudaKVSlot{kc: make([]Buffer, nLayers), vc: make([]Buffer, nLayers)}
	for l := range nLayers {
		if r.layers[l].kvShared {
			continue // aliased to its source's slot buffers below
		}
		if r.kc[l].Len() > 0 {
			b.kc[l] = r.af(r.kc[l].Len())
			own = append(own, b.kc[l])
		}
		if r.vc[l].Len() > 0 {
			b.vc[l] = r.af(r.vc[l].Len())
			own = append(own, b.vc[l])
		}
	}
	for l := range nLayers {
		if r.layers[l].kvShared {
			b.kc[l], b.vc[l] = b.kc[r.layers[l].kvSrc], b.vc[r.layers[l].kvSrc]
		}
	}
	return b, true
}

// cudaFreeVRAM is the free-VRAM probe checkKVFits prices the resident KV (every slot) against. A variable so
// TestCUDAKVSlots_pricedAgainstWhatIsLeft can stub it.
var cudaFreeVRAM = func(r *cudaResident) (uint64, error) {
	free, _, err := r.dev.Context().MemInfo()
	return free, err
}

// splitkvNever disables the split-KV decode attention for a geometry (no depth within the resident's
// capacity pays for it). Larger than any reachable nWin, so the gate comparison stays a plain >=.
const splitkvNever = 1 << 30

// singleBlockAttnShmemLimit is the dynamic shared memory a single-block attention launch may request. The
// glue/batched attention kernels size their scratch (nWin+128)*4 with no ceiling. 48 KB is the device's default
// per-block limit (the 64 KB opt-in needs a per-kernel cuFuncSetAttribute, which nothing here calls), so past
// 12,160 attended keys the launch is refused: decode fails outright at that position, and batched prefill errors
// at layer 0 and silently falls back to the much slower sequential path. If per-kernel SetAttribute support
// lands, this becomes a device query (docs/code-notes/cuda.md#singleBlockAttnShmemLimit).
const singleBlockAttnShmemLimit = 48 * 1024

// attnShmemBytes is the dynamic shared memory a single-block attention launch needs for an
// attended span of nWin keys — the sizing every such launch site uses.
func attnShmemBytes(nWin int) int { return (nWin + 128) * 4 }

// splitKVRequired reports whether the single-block attention kernel CANNOT run at this span, so split-KV is the
// only option rather than the faster one. splitkvThreshold decides when split-KV is preferred; this decides when
// it is mandatory.
func splitKVRequired(nWin int) bool { return attnShmemBytes(nWin) > singleBlockAttnShmemLimit }

// splitkvThreshold returns the EFFECTIVE attended-key count (nWin: window-clamped, not the raw position) at or above
// which split-KV decode attention beats the single-block attn_batched(M=1) for this geometry, or splitkvNever to
// disable it.
//
// It is a lookup, not a formula. Split-KV buys SM occupancy (attn_batched launches only nH blocks) and pays for it
// in DRAM: it materializes an nH×nWin f32 score array in global memory and touches it three times. Whether that
// wins depends on the occupancy deficit against the score traffic, and no one-parameter law reproduced the four
// measured geometries (qwen2.5-0.5b, qwen2.5-1.5b, windowed gemma3-1b, phi3-mini's "never").
//
// Do not loosen it from the obvious end. Every threshold is rounded UP, and an unmeasured geometry gets
// splitkvConservative rather than an extrapolation: firing early costs up to 18-25%, firing late a few percent.
// The table is NOT device-portable: the occupancy term scales with SM count and every cell was measured on one
// 40-SM Turing part, so re-measure per device class and do not scale by SM count on paper. Re-measure with an
// end-to-end serving decode: TestSplitKVCrossover's tight in-process loop with best-of-min flatters split-KV and
// must not set a threshold. Measured table and method: docs/benchmarks.md §B6 and
// docs/code-notes/cuda.md#splitkvThreshold.
func splitkvThreshold(nH, nKV, hd int) int {
	switch {
	case nKV*hd >= splitkvNeverKVFloats:
		// The never class is keyed on KV traffic per key, not on query-head count; see splitkvNeverKVFloats.
		return splitkvNever
	case nH == 12 && hd == 128:
		return 1024 // qwen2.5-1.5b class: measured crossover in (512, 1024]
	default:
		return splitkvConservative
	}
}

// splitkvNeverKVFloats: at or above this many KV floats per key (nKV*hd, one of K or V) the single-block kernel is
// close enough to the DRAM roof that extra blocks cannot help, and split-KV is pure cost.
//
// Do not key this on query-head count: models with the same nH measured OPPOSITE signs (phi3-mini, MHA, loses;
// mistral-7b, GQA, wins), and "already fills the device" was measured false. The sign follows how close the
// kernel already runs to the DRAM roof, which nKV*hd sets. 3072 is the lowest MEASURED loss (phi3-mini), not a
// midpoint: everything at or above its traffic is excluded, everything below keeps whatever depth threshold it
// already had, and the unmeasured gap below lands in the conservative default. Evidence:
// docs/measurements/splitkv-aa-floor-2026-09-12.md, splitkv-d7-fthreshold-2026-09-13.md; the refuted rule
// (splitkvMaxHeads = 24) and the traffic table: docs/code-notes/cuda.md#splitkvNeverKVFloats.
const splitkvNeverKVFloats = 3072

// splitkvConservative is qwen2.5-0.5b's measured first-clear-win depth, and the default for any geometry not in
// the table. Most measured geometries do not cross over until past 2048, so an unmeasured one is assumed not to
// either: that forfeits a few percent at depth rather than risk a 18-25% shallow regression.
const splitkvConservative = 3072

// splitkvMin is splitkvThreshold with the runtime override applied: GOINFER_SPLITKV_MIN_KEYS=<n> re-gates a stock
// binary (0 means always take the split path, the force-on A/B arm) and GOINFER_SPLITKV_ATTN=0 disables it
// entirely. nKV is taken PER LAYER, not from layers[0]: gemma4's layers differ in KV width, and a global nKV
// would misclassify its narrow layers against its wide ones.
func (r *cudaResident) splitkvMin(nKV, hd int) int {
	if r.skMinKeys >= 0 {
		return r.skMinKeys
	}
	return splitkvThreshold(r.nH, nKV, hd)
}

// cudaWQ is a device projection weight in whatever precision the checkpoint stored it.
type cudaWQ struct {
	kind string
	W    Buffer // packed weights (int4 fast-layout nibbles, or int8x4)
	ws   Buffer // int8 row scales (N)
	ws16 Buffer // int4 group scales as f16 (N*K/32) — f32 would be 20% of the
	//              int4 byte stream; f16 halves that (decode is byte-bound).
	N, K int
	// C′ VRAM expert cache: srcW/srcS are the FULL expert stack in pinned host memory (the DMA
	// source); W/ws16 are then small nSlots-deep DEVICE slot buffers the GEMV reads. perExpertW /
	// perExpertS are the per-expert strides (uint32 words / uint16 scales) for the H2D fill.
	srcW, srcS             *gpu.MappedHostBuffer
	perExpertW, perExpertS int
}

type cudaLayer struct {
	idx                 int // this layer's index, for per-layer side tables (gpt-oss sinks / expert biases)
	q, k, v, o, g, u, d cudaWQ
	qb, kb, vb          Buffer // QKV bias (absent ⇒ none)
	ob                  Buffer // attention output-projection bias (GPT-2 c_proj, gpt-oss o_proj); absent ⇒ none
	hasOBias            bool
	qNorm, kNorm        Buffer // per-head QK-norm weights (absent ⇒ arch has none)
	window              int32  // sliding-window span for THIS layer; 0 = full causal
	kvWin               bool   // this layer's K/V hold window+slack positions, not the context (kvwindow.go)
	// Per-layer attention geometry. Uniform families set the model values on every layer; Gemma 4 varies them (local
	// head_dim 16 / global 512, K=V). launchToken reads ONLY these: the model-level hd/nKV/qDim/kvDim/rhalf are
	// deliberately absent from cudaResident so a launch site cannot bind the wrong (uniform) source.
	// qDim = nH*hd, kvDim = nKV*hd, rhalf = rotaryDim/2 (rotated pairs per head).
	hd, nKV, qDim, kvDim, rhalf int
	// kEqV (attention_k_eq_v, Gemma 4 global layers): this layer has NO v_proj — V is
	// v_norm(the raw pre-RoPE k_proj output), stored un-rotated in its OWN vCache (NOT aliased
	// to kCache; kvDim is NOT halved). launchToken derives V from k before rope_kv mutates it.
	kEqV bool
	// ffnI: this layer's dense FFN width (g.N), 0 on a layer with no dense FFN. Every dense-FFN launch reads it, not
	// r.inter, because an E-model's layers differ.
	ffnI int
	// PLE branch (Gemma 4 E-model): the per-layer input gate [P x hidden], projection [hidden x P] and the post-norm weight [hidden]. Zero otherwise.
	pleGate, pleProj cudaWQ
	postPLENorm      Buffer
	// kvShared / kvSrc (Gemma 4 E-models, num_kv_shared_layers): this layer owns no K/V. It projects Q only, rotates only Q, and attends over
	// layer kvSrc's cache, which r.kc[l]/r.vc[l] alias (docs/tasks/task-multimodal-support-2026-10.md). kEqV and vNorm are false on it.
	kvShared bool
	kvSrc    int
	// vNorm: scale-less v_norm on this layer's V before it is stored. HF and the CPU apply it on EVERY Gemma 4 layer
	// that owns its K/V, K=V or not (docs/tasks/task-multimodal-support-2026-10.md). Implies kEqV or a real v_proj.
	vNorm             bool
	preNorm, postNorm Buffer
	// Gemma sandwich norms (absent unless NormSandwich4): applied to the SUBLAYER OUTPUT before
	// the residual add, not to a GEMV input.
	postAttnNorm, postMLPNorm Buffer
	invF                      Buffer // per-layer RoPE inv-freq (local vs global base)
	// mscale: YaRN's attention_factor for THIS layer (decoder.Model.RopeMscaleLayer). 1.0 for every family without
	// YaRN and on a YaRN family's non-scaled layers (Mellum's sliding layers), so it is per-layer, not per-model.
	// Passed to rope_kv / rope_kv_batched, which fold it into cos/sin.
	mscale  float32
	hasBias bool

	// Sparse MoE FFN. Per LAYER, not per model: GLM/DeepSeek's first_k_dense_replace makes the
	// first FirstKDense layers plain dense MLPs while the rest route, so the two blocks coexist
	// in one model and the dispatch picks per layer (the decoder keys off the same thing —
	// mlp.go: `arch.MoE != nil && lw.Experts != nil`).
	isMoE    bool
	routerW  Buffer       // [nE, hidden] f32 — see cudaResident.moe on why it is not quantized
	routerB  Buffer       // [nE] selection bias; ALWAYS allocated (zeros when the arch has none)
	expGU    cudaWQ       // stacked [nE * 2*moeInter, hidden]: expert e's gate at e*2*moeInter, up at +moeInter
	expDown  cudaWQ       // stacked [nE * hidden, moeInter]
	expCache *expertCache // C′ step 2: per-layer LRU slot residency (nil unless cacheExperts)

	// Always-on shared expert (GLM/DeepSeek): an ungated SwiGLU MLP at sharedInter, added to the
	// routed output. hasShared is false for a plain MoE (Mixtral). gate‖up is concatenated the
	// same way the routed experts are, so one dense GEMV + the glu_quant offset split covers it.
	hasShared bool
	shGU      cudaWQ // [2*sharedInter, hidden]
	shDown    cudaWQ // [hidden, sharedInter]
	shGateW   cudaWQ // [1, hidden] sigmoid gate (Qwen-MoE); zero-value ⇒ ungated combine

	// Gemma-4 enable_moe_block layer (parallel dense‖MoE FFN — its own gemma4MoeMLP, NOT the generic
	// moeMLP). The dense branch reuses g/u/d; the router reuses routerW (the f32 proj with
	// routerScale·hidden^-0.5 folded into its columns at build) + routerB; the experts reuse
	// expGU/expDown. g4moe selects the path; these are its extra params.
	g4moe                                                  bool
	g4preFFN, g4postFFN1, g4preFFN2, g4postFFN2, g4postFFN Buffer  // the 5 gemma4 RMSNorm weights
	perExpertScaleB                                        Buffer  // [nE] learned scale on the renormalized top-k weights
	layerScalar                                            float32 // per-layer output scalar (out = (h+combined)*layerScalar)

	// Gated-DeltaNet mixer layer. When isDeltaNet the sequence mixer is the recurrent delta rule (deltanet.cu)
	// instead of attention: no KV cache, no q/k/v/o, and TWO persistent state buffers updated in place per token.
	// dnState is [nv*hv*hk], stored TRANSPOSED relative to the CPU's [hk,hv] so each thread owns a contiguous row;
	// dnWin is the causal-conv ring, [(K-1)*convDim]. Both COMPOUND, so both are re-zeroed per generation (Reset),
	// unlike a KV cache, which the next sequence simply overwrites.
	isDeltaNet                   bool
	dnQKV, dnZ, dnOut, dnB, dnA  cudaWQ // the five DeltaNet projections
	dnConvW, dnDtBias, dnNegExpA Buffer // conv taps, dt bias, precomputed -exp(A_log)
	dnNormW                      Buffer // [hv] gated-RMSNorm weight, shared across heads
	dnWin, dnState               Buffer // persistent: conv ring, recurrent matrix state
	// qGate marks a SOFTMAX layer of the same family: q_proj is double width ([query ‖ gate] per
	// head) and the context is scaled by sigmoid(gate) before o_proj. The weight stays fused
	// because it is quantized; the split happens on the activation.
	qGate bool

	// MLA (DeepSeek / Kimi, FeatMLA)
	isMLA                      bool
	mlaQA, mlaQB, mlaQ, mlaKVA cudaWQ
	mlaQANorm, mlaKVANorm      Buffer
	mlaWUK, mlaWUV             Buffer

	// CUDA-graph capture of this layer's three STATIC launch segments (r.graphs). segA = QKV proj +
	// qk/v-norm (pre-RoPE); segB = ctx-quant + o-proj + the MLP up to the router readback; segC =
	// the expert loop + join (nil for a dense layer — no readback gap). rope_kv, attention, the g4x2
	// zero-upload and the loadRoutedExperts D2H stay LIVE in the gaps (per-token pos/nKeys, or a host
	// round-trip — neither is graph-capturable). Captured once at build, replayed per token; each
	// replay reads the CURRENT buffer contents (TestCUDA_graphReplay), so the routing that changes
	// per token flows through unchanged. nil unless r.graphs.
	gSegA, gSegB, gSegC *gpu.Graph
}

// lookupKnob reads one operator knob from the model's snapshot. A resident built by hand in a test, with no
// model behind it, reads the live environment instead — decoder's nil-snapshot rule.
func (r *cudaResident) lookupKnob(name string) (string, bool) {
	if r.knob == nil {
		return os.LookupEnv(name)
	}
	return r.knob(name)
}

// knobValue is lookupKnob's value, "" when unset.
func (r *cudaResident) knobValue(name string) string { v, _ := r.lookupKnob(name); return v }

// cudaResident is the production resident decode runner: the parity-gated cgo-free forward, promoted from the test
// harness. All CUDA state is owned by a single LockOSThread-pinned executor goroutine; Forward routes one channel
// round-trip per token. BuildResident decides which families it takes (dense, MoE, MLA, Gated-DeltaNet, ...), with
// mixed int4/int8/f32 weights as the real q4_k_m checkpoint stores them.
type cudaResident struct {
	reqCh chan func() error
	ackCh chan error

	// knob reads this model's operator knobs (decoder.Model.Knob): the snapshot taken at Load, with
	// Options.Knobs applied, never the live environment (docs/tasks/task-env-config-2026-09.md).
	knob func(name string) (string, bool)

	hidden, nLayers, inter, vocab int
	// Gemma 4 E-model: pleP is the per-layer-embedding width (0 otherwise), embLen the length of every embedding row a forward
	// takes (hidden, or hidden+nLayers*pleP), pleG the PLE gate's output scratch. r.x is embLen floats; the PLE inputs are its tail.
	pleP, embLen int
	pleG         Buffer
	eModel       bool // pleP > 0 or any KV-shared layer: the batched prefill, drafters and LoRA decline it by name
	// nH (query-head count) is the ONE model-level attention dimension: constant across a family's layers (Gemma 4
	// is 16 in both variants), so GQA tracks per-layer nKV via nH/Ly.nKV. hd/nKV/qDim/kvDim/rhalf are DELIBERATELY
	// per-layer only (cudaLayer): a uniform-source threading bug is then a compile error, not a silent byte-identical
	// pass on uniform models. Allocation-time maxima live as backend.go locals; the per-layer KV cache and UploadKV
	// read r.layers[l].kvDim.
	nH             int
	eps, attnScale float32
	finalSoftcap   float32 // Gemma final-logit softcap (30); 0 ⇒ none. Applied host-side in step().
	// attnTempBeta/attnTempOrigMaxPos (Ministral 3, FeatAttnTemp, docs/tasks/task-gpu-paths-2026-09.md): the raw
	// params behind the post-RoPE query scale (decoder.Model.AttnTempParams), 0 for every family without it.
	// Recomputed into qTempScale per decode call (rope_kv's parameter) or passed raw into rope_kv_batched, which
	// must recompute it per row (position varies within one batched launch, unlike decode's single scalar per call).
	attnTempBeta, attnTempOrigMaxPos float64
	// qTempRowsB is rope_kv_batched's per-row query-scale table (attnTempRows), sized to the largest batched pass seen; qTempRowsKey is the (startPos, M) it currently holds.
	qTempRowsB   Buffer
	qTempRowsCap int
	qTempRowsKey [2]int
	vNormUnit    Buffer // [maxHd] of 1.0 — unit weight so qk_norm (x*inv*w, addOne=0) computes scale-less v_norm (Gemma 4: every layer with vNorm). nil unless any layer has vNorm.
	qkNorm       bool   // arch needs per-head Q/K RMSNorm before RoPE
	qkNormWhole  bool   // QK-norm reduces over the WHOLE q/k vector, not per head (Olmo 3/Olmo Hybrid) — qkNorm must also be true
	rmsAddOne    bool   // (1+w) offset — false for Qwen3/Llama
	act          int32  // gated MLP activation, decoder.ActKind (0=gelu-tanh, 1=silu)
	sandwich     bool   // Gemma 4-norm sandwich: extra post-attn / post-MLP norms
	postOnly     bool   // NO pre-norm at all (Olmo 3/Olmo Hybrid) — segA quantizes the raw residual; PostAttnNorm/PostMLPNorm still dispatch, same as sandwich's post half
	// Per-32 activation groups, LayerNorm, parallel-block and logit-scale fields (Cohere/Command-R, Cohere2).
	actG32        bool    // decoder.Options.ActQuantGroup == 32: actgroup.cu's per-32 kernels are bound into fRms/fQ/fSw/gemvW4/gemvW8, and every activation scale buffer holds 2·K/32 floats (actScaleLen: scales, then per-group sums for gemv_q4k_g32)
	layerNorm     bool    // arch.Norm==NormLayer — layernorm_quant (mean-centered, bias-free) instead of rmsnorm_quant at every norm site that feeds a GEMV; see the r.norm dispatcher
	parallelBlock bool    // FeatParallelBlock: ONE shared input norm feeds attn AND MLP independently (x_final = x_orig + attn_out + mlp_out) — segBFFN reuses segA's r.aq/r.aSc instead of re-normalizing r.x; no post-attn/post-MLP norm exists for this family
	logitScale    float32 // host-side final-logit multiplier (1/arch.LogitScale), applied in step(); 0 ⇒ none (FeatLogitScale)
	cacheExperts  bool    // C′: routed experts DMA'd host→VRAM slots per token (device read; correct)
	// overlap (GOINFER_MOE_DMA_OVERLAP, default on with C′; off under graphs): the C′ round trip no longer drains the
	// stream. The host waits on evRoute (recorded right after the router), the misses are copied on dmaQ, a second
	// stream, while the kernels issued after the router keep running, and segC's rank loop waits device-side only
	// before a rank that missed. Everything keeps its launch order and arithmetic, so the output is bit-identical to
	// the draining path; only the idle time moves. Ceiling and decision rule:
	// docs/measurements/moe-streaming-decode-overlap-ceiling-2026-09-22.md.
	overlap     bool
	dmaQ        Queue
	evRoute     gpu.Event
	evMiss      []gpu.Event        // per miss of one call (≤ topK): recorded on dmaQ after that miss's two copies
	missEv      []int              // per rank: index into evMiss, or -1 for a hit
	slotIdxHost *HostBuffer[uint8] // pinned source of the per-token slot-id upload (UploadAsync needs pinned)
	cacheProf   bool               // GOINFER_MOE_CACHE_PROF: time the per-layer routing round trip
	profStall   time.Duration
	profHost    time.Duration
	profDMA     time.Duration
	// The expert DMA is split into the BIG weight copy and the TINY scale copy, with bytes and call counts for each:
	// if a small scale upload costs about as much as a large weight upload, the path is per-call-overhead bound and
	// the fix is batching. Zero cost when cacheProf is off.
	profWBytes, profSBytes uint64
	profWCalls, profSCalls uint64
	// expBatch queues one layer's host-to-device copies, issued together by one gpu.UploadBatch (appendExpertSlot).
	// profWCalls/profSCalls count logical COPIES, so the per-token copy count stays comparable across the batching;
	// profSyncCalls counts the SYNCHRONIZES, which is what batching moves. There is no per-kind timing: the copies
	// are appended here and issued together later, so a "weight upload took Xµs" figure would name something that no
	// longer happens.
	expBatch []gpu.HostCopy
	// pendingAdmits are the slot claims loadRoutedExperts made before its DMA, rolled back if the upload fails so the
	// cache cannot assert residency for bytes never copied.
	pendingAdmits []pendingAdmit
	// resetErr holds the last Reset() failure. Reset cannot return one (the cross-backend interface returns nothing),
	// so Forward/ForwardN surface it.
	resetErr      error
	profBatchTime time.Duration
	profSyncCalls uint64
	profCalls     uint64
	// Per-class decode timing for a g4moe token, sync-bounded (profMark), same cacheProf gate. It
	// exists to size the compute/DMA overlap ceiling: only dense (independent of the routing) and
	// the hit-expert prefix of segC can run under a miss DMA, so those two are the classes that
	// bound any overlap design. Sync-to-sync, so each class includes its own launch latency and
	// the token is slower with the profiler on; the numbers are upper bounds on class cost.
	headProf *headArgProf
	// The batched head tails' argmax runs on the device (fArgRows, M ints back) unless hostArgmaxPath forces the
	// M×vocab download + host loop (the A/B's do-nothing arm). argCheck additionally runs the host loop on the same
	// logits and counts agreement.
	fArgRows                                                               Pipeline
	hostArgmaxPath                                                         bool
	argCheck                                                               bool
	argChecks                                                              int
	profT0                                                                 time.Time
	profAttn, profDense, profRouter, profClear, profRT, profSegC, profHead time.Duration
	// loraLayers is nil until SetAdapter binds a compute-time LoRA adapter, then non-nil for the life of that bind.
	// segA/segB check it directly (cuda/lora.go's applyLora), so it must never be bound while r.graphs replays a
	// captured segment: the graph was captured with this nil, so replay would silently skip every LoRA dispatch.
	// SetAdapter refuses (returns an error) rather than bind under graphs; graphs are opt-in (GOINFER_CUDA_GRAPHS).
	loraLayers  []cudaLoraLayer
	loraT       Buffer         // [loraRMax]f32 scratch, shared by every projection/layer's down→up pair
	lora        loraCacheState // the bound adapter's device cache + bind counters (cuda/lora.go)
	graphs      bool           // CUDA graphs: replay each layer's static segments instead of re-issuing launches (off ⇒ byte-identical)
	graphsSync  bool           // DEBUG probe: r.stream.Sync() after each segment replay (bisects inter- vs intra-segment ordering hazards)
	graphMask   string         // DEBUG probe: if non-empty, replay ONLY the named segments (e.g. "A","B","C","AB") and issue the rest live — localizes a replay hazard to a segment
	layerCap    bool           // DEBUG probe: snapshot the residual r.x after every layer (localizes where a full-forward divergence first appears)
	layerCapBuf [][]float32

	// hidCap is the hidden-state seam a resident CUDA target would need to feed a hidden-state drafter (DFlash,
	// DSpark): the resident analogue of decoder.Model.ForwardCapture, which exists only on the CPU forward. It is NOT
	// wired into production: SetHiddenCapture/HiddenCapture have no non-test caller (the block drafter arms the batched
	// counterpart below, via SetBatchedCapture), so nothing reaches this per-token seam yet
	// (docs/code-notes/cuda.md#cudaResident.hidCap).
	//
	// It is distinct from layerCap, deliberately: layerCap is a divergence-localization probe (EVERY layer, a
	// stream.Sync() and a download each, into an unbounded buffer: 36 syncs per token at 36 layers), far too
	// expensive to decode against. hidCap copies only the TAPPED layers into fixed slots, so a 5-tap drafter costs 5.
	hidCapTaps []int       // layer indices to capture, ascending; nil ⇒ seam off
	hidCapOut  [][]float32 // [len(hidCapTaps)][hidden], overwritten per token
	// The BATCHED counterpart, for the block-drafting verify: one download per tap covering
	// all M rows, instead of the per-token seam's one per tap PER TOKEN.
	capBTaps   []int       // layer indices, ascending; nil ⇒ off
	capBOut    [][]float32 // [len(capBTaps)][M*hidden], overwritten per batched call
	launchN    int         // diagnostic: per-forward dispatch count (graph-capturable-fraction bound)
	cacheSlots int         // C′ step 2: device slots per layer (≥ topK; = topK ⇒ step-1 fresh-load, no reuse)
	slotIdx    Buffer      // C′: per-token slot ids for the routed experts, bound as the GEMV's idx
	hostIdx    []uint32    // C′: scratch for the per-layer rIdx device→host readback
	hostSlot   []uint32    // C′: scratch for the per-token slot ids uploaded to slotIdx

	// Sparse MoE. The router projection stays f32 (gemv_f32_a8) while the experts are int4: its output steers a
	// DISCRETE choice, so a quantization error near a tie does not perturb the result slightly, it runs a DIFFERENT
	// expert and the output is unrelated (the Granite SSM expert-flip class; docs/code-notes/cuda.md#cudaResident.moe).
	// This is the one place in this backend where the cheap precision is not worth it.
	moe                     bool
	nE, topK, moeInter      int
	moeSigmoid, moeNormTopK int32
	moeScale                float32
	nGroup, topkGroup       int

	// Gemma-4 parallel dense||MoE (any layer g4moe: JIT router_f32, alloc g4 scratch, take gemma4MoeMLP). g4cap
	// (GOINFER_G4_CAPTURE) is a DEBUG readback of the four MoE-layer buffers (rn / wgt / x1 / x2) so a whole-forward
	// miss localizes to router vs dense vs expert vs join in ONE run.
	gemma4Moe bool
	// gemma4Dense is true for ANY gemma4 checkpoint (dense or MoE), broader than gemma4Moe (enable_moe_block layers).
	// It gates compiling fScaleVec, the dense per-layer-output-scalar kernel (segB's dense tail), which has nothing
	// router-specific about it.
	gemma4Dense                         bool
	g4cap                               bool
	g4capRn, g4capWgt, g4capX1, g4capX2 [][]float32
	g4capIdx                            [][]uint32 // APPEND order (token-outer, layer-inner), matching the CPU routerCaptureBuf — for the per-POSITION routing-agreement check (a top-k flip at pos N reads like accumulation in a cosine)
	// g4capLayer is g4capIdx's layer index, recorded in parallel so a consumer replaying the trace through a
	// per-layer cache need not infer the (token, layer) shape from the append order and a divisor.
	g4capLayer []int

	// Per-sublayer contribution capture (diagnostic; off in production, zero cost). When subCap
	// is set, launchToken copies the sandwich-normed o-proj output (attention contribution) and
	// down output (MLP contribution) per layer — the exact dp4a-path analogue of the decoder's
	// ForwardSubCapture, so a cross-backend per-sublayer diff is possible.
	subCap                                 bool
	subAttnC, subMLPC, subCtxC, subMLPpreC [][]float32

	sharedInter int // width of the always-on shared expert (0 ⇒ none)

	// device state — touched ONLY on the executor thread.
	dev                                                                                     *Device
	stream                                                                                  Queue
	gemvW4, gemvW8, ropeKV, fRms, fRmsF32, fQ, fAttn, fSw, fRes, fArg, fQKV, fGU, fQKN, fLN Pipeline
	fTopK                                                                                   Pipeline // topk_select
	kvCopy                                                                                  Pipeline // kv_store at pos 0 = an exact device copy (K=V layers copy raw k into vB instead of projecting twice)
	fQKVRows                                                                                Pipeline // fused_rms_qkv_rows: fused_rms_qkv with rows-per-warp (zero when its module did not load)
	fGURows                                                                                 Pipeline // fused_rms_gu_rows: fused_rms_gu with rows-per-warp (zero when its module did not load)
	smCount, smThreads, smSmem                                                              int      // device shape for wave-sized fused-projection grids (zero when unread): SM count, max threads per SM, max shared memory per SM
	fGumbel1, fGumbel2                                                                      Pipeline // gumbel_stage1/2
	// gemvQ4K is actgroup.cu's gemv_q4k_g32: native GGUF Q4_K weights (--quant q4k), bound only
	// under per-32 activations, the only mode that carries the per-group activation sums it needs.
	gemvQ4K Pipeline
	// fQKVg32 / fGUg32 are actgroup.cu's fused_rms_qkv_g32 / fused_rms_gu_g32: the per-32 twins of
	// fQKV / fGU (rmsnorm + per-32 quant redundantly per block + the projection rows, any of int8 /
	// int4 / q4k per projection). fuseG32 selects them (BuildResident); bound only under actG32.
	fQKVg32, fGUg32 Pipeline
	fuseG32         bool
	// Compute-time LoRA (docs/tasks/task-gpu-paths-2026-09.md, cuda/lora.go). Own module (lora.ptx), loaded
	// unconditionally like every other glue pipeline: cheap, and whether a model will ever receive an adapter is not
	// known at BuildResident time.
	fLoraDown, fLoraUp Pipeline
	// Batched (M=len) prefill pipelines (prefill_batched.ptx), the weight-stationary path. bGemv is the batched W4A8
	// GEMV; the rest are the M=1 glue kernels with an M dimension, each bit-identical per row. Loaded once at build
	// (small module).
	bRN, bRms, bRopeKV, bAttn, bQuant, bSw, bRes Pipeline
	bW8                                          Pipeline // batched W8A8 GEMV (int8 bundles); nil ⇒ int8 prefill declines
	bQKN                                         Pipeline // batched per-head Q/K RMSNorm (qwen3 etc.); loaded with the batched set
	bNormF32                                     Pipeline // batched plain f32 RMSNorm for Gemma sandwich post-norms; loaded with the batched set
	bLN                                          Pipeline // batched mean-centered LayerNorm+quant (layernorm_quant.ptx) for Cohere/Command-R
	zeroBias                                     Buffer   // [hidden] zeros for bias-free layernorm_quant_batched
	skScores, skSoftmax, skVsum                  Pipeline // split-KV decode attention (high-occupancy, bit-identical)
	skScoreBuf, skInvBuf                         Buffer   // split-KV scratch: [nH·ctxCap] raw/exp scores, [nH] inverse denominators
	skVsumPartial, skVsumCombine                 Pipeline // flash-decode V-sum SPIKE (opt-in, NOT bit-identical) — scoping-decode-tree-recanon.md §6
	skPartialBuf                                 Buffer   // [nH·maxHd·nSplit] partial folds for the spike
	skVsumSplit                                  int      // GOINFER_SPLITKV_VSUM_SPLIT; 0 = off (the shipped path)
	// Flash-decode lane (decode_fa.cu): default on (GOINFER_CUDA_FLASH_DECODE=0 turns it off), NOT bit-identical.
	faPartial     [3]Pipeline // fa_partial_{64,128,256}
	faCombine     Pipeline
	faSplit       int            // S key splits per kv head; 0 = lane off (the exact path)
	faMinKeys     int            // attended-span floor below which the exact path runs (GOINFER_CUDA_FLASH_DECODE_MIN_KEYS)
	faBuf         Buffer         // partials [nH][S*8][hd+4]
	faLaunches    int            // launches of fa_partial, so a test can prove the lane ran
	faExactScope  atomic.Int32   // >0: a speculative generation holds the exact-attention scope, so the lane is bypassed
	faRows        [3][9]Pipeline // fa_partial_rows_{64,128,256}_g{1..8} (multi-row verify variant); zero when not loaded
	faCombineRows Pipeline
	faVerify      bool // the multi-row lane serves speculative verify rows (loaded, and not disabled by GOINFER_CUDA_FLASH_DECODE_VERIFY=0)
	faRowLaunches int  // launches of fa_partial_rows, so a test can prove verify rows ran on the multi-row lane

	// L2 (docs/completed/task-prefill-gap.md §4): the fused prefill attention, one instantiation per supported head
	// dim. Zero-valued unless GOINFER_CUDA_FAST_PREFILL selected it AND the module loaded; every selection site
	// treats the zero Pipeline as "use attn_batched".
	bAttnFused64, bAttnFused128 Pipeline
	// Tile-shape arms of attn_fused (attn_fused_bm.cu): 32x64 and 32x32 (query rows x keys), per head dim.
	// attnTile: 0 = default (128x64 for hd128 layers without a window, else 64x64), -1 = 64x64 everywhere (GOINFER_CUDA_ATTN_FUSED_TILE=64x64), 1/2/3 = force 32x64 / 32x32 / 128x64 (experiment arms).
	bAttnBM32x64hd64, bAttnBM32x64hd128, bAttnBM32x32hd64, bAttnBM32x32hd128 Pipeline
	bAttnBM128hd64, bAttnBM128hd128                                          Pipeline
	attnTile                                                                 int
	// L3 (§4 L3): the tensor-core int4 GEMM. Zero-valued unless selected AND loaded; bGemvB
	// treats the zero Pipeline as "use gemv_w4a8_rn", which is the exact path.
	bGemmMMA Pipeline
	// Per-lever selection (§5 attribution): fastAttn drives L2, fastGemm drives L3.
	fastAttn, fastGemm bool
	// Gated-DeltaNet mixer (deltanet.ptx, its own module). Loaded only for that family; every other model leaves
	// these zero and never dispatches them.
	dnConv, dnGates, dnNorm, dnRule, dnGNorm Pipeline
	// Row-batched twins of the five, for prompt prefill (prefillDeltaNetRows): one launch per layer covers M rows.
	dnConvRows, dnGatesRows, dnNormRows, dnRuleRows, dnGNormRows Pipeline
	dnRuleRows128                                                Pipeline // the scan with its state row in registers (hk == hv == 128)
	dnQSplit, dnAttnGate                                         Pipeline // the family's fused double-width q_proj + output gate
	dnet                                                         *dnetParams

	// gpt-oss only (nil elsewhere, and every other family's launches pass ArgNull so the kernels stay bit-identical): the
	// clamped interleaved-SwiGLU expert epilogue, plus the per-layer attention sinks and the per-expert gate‖up bias
	// table it needs.
	gptOssSw                 Pipeline // glu_quant_gptoss (own module — audited glue.ptx/moe.ptx untouched)
	gptOssRoute              Pipeline // route_gptoss — top-k + softmax over the BIASED logits (moe_route's contract is wrong for this family)
	gptOssAlpha, gptOssLimit float32
	gptOssSinks              []Buffer // [layer] → [nH] learned per-head attention sink logits
	gptOssDownBias           []Buffer // [layer] → [nExpert*hidden] per-expert down-projection bias
	gptOssExpBias            []Buffer // [layer] → [nExpert·2·moeInter] gate‖up biases, indexed on-device by the router
	splitkvAttn              bool     // GOINFER_SPLITKV_ATTN: use the split-KV decode attention (else the single-block attn_batched(M=1))
	skMinKeys                int      // GOINFER_SPLITKV_MIN_KEYS: -1 ⇒ per-geometry table; ≥0 overrides it (0 ⇒ always split)
	prefillReady             bool     // batched kernels loaded; PrefillLast usable
	bAttnImg                 Pipeline // attn_img_batched (attn_img_prefill.ptx) — Gemma 3's bidirectional image-block prefill attention; own module, see cuda/attn_img_prefill.cu
	imgPrefillReady          bool     // bAttnImg loaded; PrefillImageLast usable. A load failure is not fatal: it stays false and the caller falls back to CPU prefill + UploadKV
	bRopeKVMRoPE             Pipeline // rope_kv_mrope_batched (rope_mrope_prefill.ptx) — Qwen2.5-VL's m-RoPE batched-prefill rotation; own module, see cuda/rope_mrope_prefill.cu
	pairwiseRoPE             bool     // GPT-J pairwise rotation (Cohere/Cohere2/Aya/GLM-OCR): ropeKV/bRopeKV/bRopeKVMRoPE are bound to the rope_pairwise.ptx twins, not the NeoX kernels. Set from Model.PairwiseRoPEResident() at build
	mropePrefillReady        bool     // bRopeKVMRoPE loaded AND this model has MRopeSection; PrefillMRoPELast usable. A load failure (or a non-m-RoPE model) is not fatal: it stays false and the caller falls back to CPU prefill + UploadKV
	mropeTakesMode           bool     // the bound bRopeKVMRoPE is rope_kv_mrope_batched (takes the layout mode as a final argument), not the pairwise twin (no such argument)
	mropeMode                int32    // rope_kv_mrope_batched's layout: 0 contiguous sections (Qwen2.5-VL), 1 interleaved per index (Qwen3-VL, Qwen3.5+); mropeSec0/1 mean what rope_mrope_prefill.cu's mrope_pos says for that mode
	mropeSec0, mropeSec1     int32    // cumulative MRopeSection boundaries (sec0=section[0], sec1=section[0]+section[1]), computed once at build time — see rope_kv_mrope_batched's own doc comment for the (d<sec0)?t:(d<sec1?h:w) rule this feeds
	// prefillChunkCap is the LEARNED row budget shared by every batched-prefill caller that has a row-count knob to
	// shrink: prefillChunked (retries the SAME pass smaller and stores whatever width worked) and PrefillImageLast
	// (cannot retry a bidirectional image block smaller, since an image OOM is a function of that block's own M, but
	// halves and stores the budget on OOM so the NEXT image call is caught by its cheap pre-check). 0 until a pass
	// OOMs, then the width that worked (for the image path, the width to try next). A card that cannot hold the
	// default chunk is thereby discovered ONCE, not on every prompt, which matters beyond waste: a context taken to
	// refusal and kept in use can afterwards launch kernels that "return SUCCESS and execute NOTHING" (backend.go's
	// A13 note; docs/code-notes/cuda.md#cudaResident.prefillChunkCap). Atomic because both callers run on their own
	// CALLER's goroutine (only the per-pass job is serialized through the executor).
	prefillChunkCap atomic.Int64
	prof            *prefillProf // non-nil ⇒ PrefillLast times each kernel category (test-only; adds stream syncs)
	// passPromptLen is the TOTAL prompt length this batched pass belongs to (startPos+M), set once at the top of
	// prefillCore. The fast-prefill floor is a property of the PROMPT, not of the chunk: prefillChunked splits a long
	// prompt into passes of <=512 rows, so gating on M alone would judge a 3900-token prompt by its 512-row chunk.
	// Per-pass mutable state on the resident, the same shape as prof above.
	//
	// It is NOT cleared after prefillCore returns: residentDrafter.DraftBlock (cuda/drafter.go) reads it too, via
	// aboveFastPrefillFloor/bGemvB, and inherits whatever the TARGET's last prefill set. DraftBlock's doc says why
	// that costs acceptance rate at worst, never correctness.
	passPromptLen int
	// forceExactKernels disables useAttnFused/useGemmMMA (the L2/L3 fast levers) for the DURATION of one prefillCore
	// pass: per-pass mutable state like passPromptLen, set and cleared at prefillCore's own r.do(...) boundary. Those
	// levers are chosen purely on shape, never on WHO is asking, and they are cosine-close to
	// gemv_w4a8_rn/attn_batched, not bit-identical: HiddenLast's bit-identity contract and speculative verify's
	// tailAllLogits/tailAllArgmax tails need decode-identical numerics. So it is true for every prefillCore tail
	// except tailLastLogits (ordinary single-row-output prefill, where the cosine-gated tolerance applies). The
	// drafter's own forward (cuda/drafter.go) never touches it: a proposal need not be bit-identical to anything,
	// only the target's verify of it.
	forceExactKernels bool
	// chunkOrdinary is set by prefillChunked for the duration of a multi-chunk prefill whose FINAL tail is the ordinary tailLastLogits: its non-final passes (tailKVOnly) are part of that
	// same prompt prefill and must get the same kernels as the final pass. Without it forceExactKernels (tail != tailLastLogits) demoted EVERY non-final chunk to the slow exact path
	// (docs/measurements/prefill-chunk-demotion-2026-09-21.md). A HiddenLast prefill (tailHiddenLast) leaves it false, so every one of its chunks stays exact.
	chunkOrdinary bool
	// chunkPromptLen is the WHOLE prompt's length (startPos + all rows), set by prefillChunked for the duration of a multi-chunk prefill so the fast-prefill floor is judged on the prompt and not on
	// the prompt-so-far: passPromptLen is startPos+M of THIS pass, so a first chunk narrower than the floor (the OOM-halving path, or GOINFER_PREFILL_CHUNK) would run the exact kernels and
	// its later chunks the fast ones, mixing numerics inside one prompt (TestPrefillChunked_bitIdentical, chunk=300). Zero outside a chunked prefill.
	chunkPromptLen int
	// fastAttnLaunches / fastGemmLaunches count attn_fused and gemm_w4a8_mma launches, so a test can prove WHICH kernels a prefill pass used rather than infer it from timing.
	fastAttnLaunches, fastGemmLaunches                      int
	tile128Launches                                         int // attn_fused launches that used the 128-row-tile kernel (default for hd128 layers without a window)
	fRoute, fRouterGemv, fMoEGemv, fMoEWacc, fSharedCombine Pipeline
	fMoEWaccBias                                            Pipeline // gpt-oss: wacc + per-expert down bias
	fRouterF32, fScaleWgt, fRmsNW, fScaleVec                Pipeline // gemma4 MoE (router_f32 module)
	fMlaStore, fMlaHeadMV, fMlaQRope, fMlaAttn              Pipeline // MLA (mla module, FeatMLA)

	// MLA (DeepSeek / Kimi, FeatMLA) geometry & params
	isMLA         bool
	mlaRank       int // kv_lora_rank
	mlaLatDim     int // kv_lora_rank + qk_rope_head_dim
	mlaQKHead     int // qk_nope + qk_rope
	mlaQKNope     int // qk_nope_head_dim
	mlaQKRope     int // qk_rope_head_dim
	mlaVHead      int // v_head_dim
	mlaQLoRA      int // q_lora_rank (0 if direct q_proj)
	mlaInterleave bool
	mlaRopeScale  float32

	fuseQKV     bool  // all of Q/K/V/gate/up int4 ⇒ the fused K1 (fQKV) + fGU super-kernels are usable
	launchErr   error // sticky first launch error within a launchToken call (reset per token) — M23
	layers      []cudaLayer
	ctxExplicit bool  // the cap came from configuration (Options.ResidentContext), not the default — decides whether a VRAM miss is a hard error or a decline
	ctxCap      int   // effective resident KV capacity in positions = resolveCtxCap(request, model ctx). Every kc/vc is sized cap*kvDim; checkCap guards against it.
	extraBytes  int64 // decoder.Model.ExtraResidentBytes() at construction — a companion allocation (--drafter) reserved out of every free-VRAM check this resident makes after it (checkWeightsFit, checkKVFits, allocSlots)
	lmW         cudaWQ
	finalNorm   Buffer

	// per-token scratch + KV caches (device).
	x, aSc, qB, kB, vB, cctx, cSc, oO, mSc, gO, uO, dSc, dScr, dO, logits Buffer
	aq, cq, mq, dq, argIdx                                                Buffer
	argVal                                                                Buffer
	topkOut                                                               Buffer // topk_select output: ids, logit bits, Z hi/lo bits
	gbKey, gbIdx, gbOut                                                   Buffer // gumbel_stage1 per-block winners, and the drawn id
	kc, vc                                                                []Buffer

	// MC1's resident KV slots (docs/tasks/task-concurrency-2026-09.md). kvSlotsReq is the model's request
	// (decoder.Model.ResidentKVSlotsRequest; 1 for a recurrent family and under expert streaming), kvSlotBufs each
	// slot's per-layer K/V (nil with one slot), kvSlot the bound index. kc/vc above are the BOUND slot's buffers —
	// every kernel launch and UploadKV reads them at call time — and UseKVSlot rebinds them.
	kvSlotsReq      int
	ctxPlanned      int           // ctxCap as the plan chose it, before checkKVFits can trim it against the real free VRAM
	deepPlan        *cudaDeepPlan // the DeepStack sets of the PrefillMRoPEDeepstackLast call in progress (prefill.go); nil otherwise
	allocSlackBytes int64         // the driver's allocation rounding over the packed weights, priced into the plan (packedAllocSlack)
	kvSlotsN        int           // what checkKVFits granted (>= 1)
	kvSlot          int
	kvSlotBufs      []cudaKVSlot

	// Windowed KV (kvwindow.go): kvWin is set when some layers hold window+slack positions. kvBases[s] is the absolute position at physical slot 0
	// of slot s's windowed layers, kvBase the bound slot's, kvScratch the compaction's staging buffer.
	kvWin     bool
	kvWindow  int
	kvBase    int
	kvBases   []int
	kvScratch Buffer

	// MoE per-token scratch (allocated only when moe). Sized to the MoE expert width, which is
	// NOT the dense one — Mellum's moe_intermediate_size differs from intermediate_size, so
	// reusing gO/uO/dq here would overrun on some archs and silently under-read on others.
	rLogits, rWgt, moeGU, moeSc, moeScr Buffer
	rIdx                                Buffer

	// Gated-DeltaNet per-token scratch (allocated only when any layer isDeltaNet). Sized from
	// dnetParams, which is NOT the attention geometry: convDim is 2*nk*hk + nv*hv and has no
	// relation to qDim/kvDim, so none of the buffers above can be reused for it.
	dnMixed, dnConvOut, dnQn, dnKn, dnHeadP, dnBt, dnAt, dnZOut, dnCore, dnGated Buffer
	dnGq, dnGSc                                                                  Buffer // int8 activation + scale for the gated output's out_proj GEMV
	dnQg                                                                         Buffer // [2*qDim] the fused [query ‖ gate] q_proj output, before the split
	dnAGate                                                                      Buffer // [qDim] attention output gate (qGate layers)
	moeQ                                                                         Buffer

	// MLA per-token scratch (allocated only when isMLA).
	mlaQAOut, mlaQAQ, mlaQASc Buffer
	mlaKVDown, qAbs, wsum     Buffer

	// Gemma-4 MoE branch scratch [hidden] (allocated only when gemma4Moe). x1 = dense branch, x2 =
	// expert-sum branch, rn = the router's weightless-normed raw-h input. Kept SEPARATE from r.x
	// because the join norms (x1+x2) BEFORE adding the residual h — the experts can't wacc into the
	// residual stream the way the generic moeMLP does.
	g4x1, g4x2, g4rn Buffer

	// Shared-expert scratch (allocated only when any layer hasShared). Sized to sharedInter,
	// which is its own width — distinct from both the dense inter and the routed moeInter.
	shGUout, shSc, shScr, shDownOut Buffer
	shGl                            Buffer // [1] the Qwen-MoE shared-gate logit; unused when ungated
	shQ                             Buffer

	// logitsPinned is PAGE-LOCKED host memory for the per-token logits readback: a pageable D2H stages through a driver
	// bounce buffer, pinned memory DMAs straight out. Slice() is a zero-copy view, so Forward still returns without an
	// extra copy. Reused across calls (decode consumes each before the next).
	logitsPinned *HostBuffer[float32]
	logitsHost   []float32 // zero-copy view of logitsPinned
	// Batched-head verify scratch (PrefillLastNArgmax). Allocated lazily INSIDE the executor
	// job — af/ai require r.dev's context current — and reused across rounds. logitsBCap is the
	// row count the current allocation covers; a wider block reallocates once and then holds.
	logitsB    Buffer
	logitsBIdx Buffer // [logitsBCap] int32: argmax_rows' output for the verify tail
	logitsBCap int
	setupErr   error // first alloc/upload error during BuildResident's setup job
	// Instrument fields, RECORDING ONLY, never read by production logic: free device VRAM immediately before and after
	// allocSlots in the SAME process, so a test measures consumption without a cross-run assumption.
	dbgFreeBefore, dbgFreeAfter uint64
	dbgPredInline               int64  // consumption predicted by the arithmetic production executes
	dbgPredCapSlots             int64  // consumption predicted from capSlots' chosen count
	dbgSlotsInline              int    // slot count the INLINE copy at allocSlots chose
	dbgSlotsCapSlots            int    // slot count capSlots would choose from the same free
	dbgFreeBeforeLaunch         uint64 // free VRAM immediately before the first forward's launches
	dbgFreePreLaunch            uint64 // free VRAM immediately before the MOST RECENT launch
	dbgLaunchTrace              []string
	// dbgProbe enables the per-launch pre-launch probe. A FIELD, not a package var read from the environment: a
	// package var is initialised before t.Setenv can run, so it would silently record nothing; the test sets the field
	// on the resident it already holds. Off by default: the probe costs a MemInfo round-trip and a reflection scan on
	// EVERY launch.
	dbgProbe bool
	// routeRecord, when non-nil, is called once per loadRoutedExperts call with the layer index and that call's routed
	// expert ids (r.hostIdx[:topK]): a measurement hook for how many DISTINCT experts a whole M-row chunk touches per
	// layer (docs/queue-performance.md P20). A field for the same reason dbgProbe is.
	routeRecord   func(layer int, ids []uint32)
	cacheSlotsReq int   // slots REQUESTED (pre-cap), so errors can name both
	dbgAllocSizes []int // every requested slot-buffer size, recording only
}

// alloc/upload helpers, called ONLY inside the setup job (r.dev's context current on the executor thread).
// gpu.NewBufferLenOf PANICS on OOM (recorded into the Device ledger); the executor's runJob recover, NOT
// BuildResident's own defer (a different goroutine, which cannot reach an executor-thread panic), turns it into
// setupErr, so BuildResident declines gracefully (staged fallback) instead of proceeding with unusable buffers.
func (r *cudaResident) af(n int) Buffer {
	return gpu.NewBufferLenOf[float32](r.dev, n)
}
func (r *cudaResident) ai(n int) Buffer {
	return gpu.NewBufferLenOf[int32](r.dev, n)
}
func (r *cudaResident) au32(n int) Buffer {
	return gpu.NewBufferLenOf[uint32](r.dev, n)
}

// recordUpload captures the FIRST alloc/upload error hit during BuildResident's setup job into r.setupErr. The
// up* helpers must not discard gpu.Upload's error: a failed weight upload would leave a ZEROED device buffer and
// the build would still return ok=true, a resident that decodes garbage. The setup job's last statement returns
// r.setupErr, which BuildResident turns into a graceful decline (staged/CPU fallback); later errors in the same
// doomed job are noise. Called only at load time.
func (r *cudaResident) recordUpload(e error) {
	if e != nil && r.setupErr == nil {
		r.setupErr = e
	}
}
func (r *cudaResident) up32(v []float32) Buffer {
	b := r.af(len(v))
	r.recordUpload(gpu.Upload(b, v))
	return b
}
func (r *cudaResident) upu32(v []uint32) Buffer {
	b := r.au32(len(v))
	r.recordUpload(gpu.Upload(b, v))
	return b
}
func (r *cudaResident) upu16(v []uint16) Buffer {
	b := gpu.NewBufferLenOf[uint16](r.dev, len(v))
	r.recordUpload(gpu.Upload(b, v))
	return b
}
func (r *cudaResident) upW(h hostW) cudaWQ {
	w := cudaWQ{kind: h.kind, W: r.upu32(h.wpk), N: h.N, K: h.K}
	switch h.kind {
	case "int4":
		w.ws16 = r.upu16(h.ws16)
	case "q4k":
		// scales and minimums live inside the super-blocks
	default:
		w.ws = r.up32(h.ws)
	}
	return w
}

// upExperts uploads an expert stack, VRAM-slot-cached (C′) when cacheExperts is set and the weight is int4 (the
// only kind the resident MoE GEMVs accept), else fully VRAM-resident. The GEMVs read device memory only: a
// zero-copy direct read of the host stack was tried and mis-read by gemv_w4a8_moe at width
// (docs/completed/task-moe-streaming.md, NewMappedHostBuffer's doc).
func (r *cudaResident) upExperts(h hostW) cudaWQ {
	if r.cacheExperts && h.kind == "int4" {
		return r.cacheWQ(h)
	}
	return r.upW(h)
}

// cacheWQ (C′) keeps the FULL expert stack in pinned host memory (srcW/srcS, the DMA source) and
// allocates a small nSlots-deep DEVICE slot buffer (W/ws16) that the GEMV actually reads. Per token
// the routed experts are DMA'd from the host source into the slots (loadWQ); the kernel indexes by
// SLOT id, so it runs unmodified against the smaller stack (no moe.ptx change). The device stack is
// r.cacheSlots deep (≥ topK); loadRoutedExperts fills it via the per-layer LRU expertCache — at
// nSlots=topK it degenerates to step-1 fresh-load, above that it reuses across tokens.
// rowsPerExpert = N/nE (the stack has nE experts, N total rows).
func (r *cudaResident) cacheWQ(h hostW) cudaWQ {
	w := cudaWQ{kind: h.kind, N: h.N, K: h.K}
	rowsPerExpert := h.N / r.nE
	w.perExpertW = rowsPerExpert * (h.K / 8)  // uint32 words per expert weight (Kwords = K/8)
	w.perExpertS = rowsPerExpert * (h.K / 32) // uint16 group scales per expert (Kgroups = K/32)
	w.srcW = r.mapBytes(u32bytes(h.wpk))
	w.srcS = r.mapBytes(u16bytes(h.ws16))
	// Device slot buffers (W/ws16) are NOT allocated here — allocSlots does it after the core +
	// KV are up, so it can size r.cacheSlots to the MEASURED free VRAM and never OOM.
	return w
}

// slotBytesPerLayer is the device VRAM one slot's worth of BOTH expert projections costs (int4 weight + f16
// scales), used to size the cache to free VRAM. It must be measured from a ROUTED layer, not from layer 0: layer 0
// is dense on every family with `first_k_dense_replace` (GLM-4.5/4.6, DeepSeek-V2/V3, Kimi), so its strides are
// zero and the caller's `budget / len(moeLayers) / perLayer` would be an integer divide by zero, a panic on the
// executor goroutine.
func (r *cudaResident) slotBytesPerLayer(layer int) int {
	if layer < 0 || layer >= len(r.layers) {
		return 0
	}
	gu, dn := &r.layers[layer].expGU, &r.layers[layer].expDown
	return (gu.perExpertW+dn.perExpertW)*4 + (gu.perExpertS+dn.perExpertS)*2
}

// slotStrides is the per-slot byte size of each buffer allocated per MoE layer, IN THE ORDER AND
// GROUPING allocSlots allocates them. The sum is slotBytesPerLayer; the split matters because the
// driver charges each allocation its own whole quanta, so the total is not a function of the sum.
func (r *cudaResident) slotStrides(layer int) []int64 {
	if layer < 0 || layer >= len(r.layers) {
		return nil
	}
	gu, dn := &r.layers[layer].expGU, &r.layers[layer].expDown
	return []int64{
		int64(gu.perExpertW) * 4, int64(gu.perExpertS) * 2,
		int64(dn.perExpertW) * 4, int64(dn.perExpertS) * 2,
	}
}

// allocQuantumBytes is the driver's allocation granularity, MEASURED (cuda/allocgran_test.go:
// 5 MiB -> 6, 6 -> 6, 9 -> 10, so 2 MiB granular and NOT next-power-of-two, which would over-charge
// by up to 2x on any buffer not sitting just above a power of two).
const allocQuantumBytes = 2 << 20

// allocRoundSlack is what the driver adds to ONE buffer of n bytes: a buffer of a quantum or more is rounded up to
// the next quantum, a smaller one is not (rounding the small ones too over-predicts the gap several-fold;
// docs/code-notes/cuda.md#allocRoundSlack).
func allocRoundSlack(n int64) int64 {
	if n < allocQuantumBytes {
		return 0
	}
	return (allocQuantumBytes - n%allocQuantumBytes) % allocQuantumBytes
}

// allocSlackOffForTest prices no allocation slack into the plan (the planted defect of TestBuildScratchAccounting_plantedDefect).
var allocSlackOffForTest bool

var hostWType = reflect.TypeFor[hostW]()

// packedAllocSlack is the allocation rounding of every packed weight the build is about to upload: each hostW found
// under v (a struct, slice, array or pointer, however nested), as the two buffers upW allocates for it. Walking by
// reflection, not by field name, is what keeps a family's extra projections (DeltaNet, MLA, Mamba, experts) counted
// without this function learning about them -- the per-family struct literal that omits a field is the defect class
// this repo has been bitten by. Experts under C' are slot-cached rather than uploaded whole, so they are over-counted
// there, which only shrinks the planned context.
func packedAllocSlack(v any) int64 {
	var walk func(rv reflect.Value) int64
	walk = func(rv reflect.Value) int64 {
		switch rv.Kind() {
		case reflect.Pointer, reflect.Interface:
			if rv.IsNil() {
				return 0
			}
			return walk(rv.Elem())
		case reflect.Slice, reflect.Array:
			switch rv.Type().Elem().Kind() {
			case reflect.Struct, reflect.Pointer, reflect.Interface, reflect.Slice, reflect.Array:
			default:
				return 0 // a vector of numbers (biases, norms) holds no hostW: do not walk its elements
			}
			var n int64
			for i := 0; i < rv.Len(); i++ {
				n += walk(rv.Index(i))
			}
			return n
		case reflect.Struct:
			if rv.Type() == hostWType {
				n := allocRoundSlack(int64(rv.FieldByName("wpk").Len()) * 4)
				n += allocRoundSlack(int64(rv.FieldByName("ws").Len()) * 4)
				n += allocRoundSlack(int64(rv.FieldByName("ws16").Len()) * 2)
				return n
			}
			var n int64
			for _, field := range rv.Fields() {
				n += walk(field)
			}
			return n
		}
		return 0
	}
	return walk(reflect.ValueOf(v))
}

// slotRequirement is the device VRAM n slots/layer actually costs: each buffer rounded up to its own
// quantum, times the layer count. Monotone non-decreasing in n, which is what makes capSlots'
// binary search valid.
func slotRequirement(n int, nLayers int64, strides []int64) int64 {
	var perLayer int64
	for _, p := range strides {
		b := int64(n) * p
		perLayer += (b + allocQuantumBytes - 1) / allocQuantumBytes * allocQuantumBytes
	}
	return nLayers * perLayer
}

// slotMarginBytes is the launch-time headroom the cap must leave free. It is a flat constant that TestAllocFloor
// CHECKS against the measured driver allocation floor, not one derived from what it must cover: the floor is
// unattributed and sits inside it (docs/queue-release.md, the "margin derived rather than asserted" item;
// docs/code-notes/cuda.md#slotMarginBytes).
const slotMarginBytes = 384 << 20

// capSlots is the sizing arithmetic, and it is a SEARCH rather than a division. Per-buffer 2 MiB rounding makes
// the requirement a step function of the slot count, and `(free - slotMarginBytes) / nLayers / perLayer` cannot
// invert a step function: it is wrong precisely at the boundaries the failure lives on (on the real 26B it
// returned 34 slots where the true requirement exceeded free by ~200 MB, because all four buffers tip a quantum
// AT ONCE). A division plus a correction term would reproduce the same class one boundary over, so there is no
// fudge factor here. The requirement is monotone non-decreasing in n, so bisect it. A pure function of its
// inputs, so synthetic free-VRAM figures exercise a branch that in production binds only on models far larger
// than any fixture.
//
// Returns the slot count to use, or decline=true when not even topK fits.
func capSlots(free, nLayers int64, strides []int64, topK, request int) (slots int, decline bool) {
	if nLayers <= 0 || len(strides) == 0 {
		return request, false
	}
	var any bool
	for _, p := range strides {
		if p > 0 {
			any = true
		}
	}
	if !any {
		return request, false
	}
	fits := func(n int) bool { return slotRequirement(n, nLayers, strides)+slotMarginBytes <= free }
	if fits(request) {
		return request, false
	}
	// Largest n in [0, request) that fits. lo always fits (n=0 costs nothing), hi never does.
	lo, hi := 0, request
	for hi-lo > 1 {
		mid := (lo + hi) / 2
		if fits(mid) {
			lo = mid
		} else {
			hi = mid
		}
	}
	if lo < topK {
		return lo, true
	}
	return lo, false
}

// allocSlots caps r.cacheSlots to the MEASURED free device VRAM (with a safety margin) and then
// allocates each MoE layer's slot buffers + its LRU cache. Called after the resident core + KV +
// scratch are up, so the cap reflects what is actually left — an over-large GOINFER_MOE_CACHE_SLOTS
// is capped-and-logged (the repo's "decline honestly at load" discipline), never OOM'd mid-build.
func (r *cudaResident) allocSlots() error {
	var moeLayers []int
	for i := range r.layers {
		if r.layers[i].expGU.srcW != nil {
			moeLayers = append(moeLayers, i)
		}
	}
	if len(moeLayers) == 0 {
		return nil
	}
	// Size from the FIRST ROUTED layer (moeLayers[0]), not layer 0: see slotBytesPerLayer.
	perLayer := r.slotBytesPerLayer(moeLayers[0])
	if perLayer <= 0 {
		// No routed layer reports a per-expert stride (a degenerate blob): nothing to cache and nothing to divide by.
		// Clear cacheExperts so the decode path cannot later read slots/expCache that were never allocated. This does NOT
		// yield a working "hold every expert" path (upExperts left the stacks host-mapped-only, so the expert GEMVs would
		// bind zero-value device buffers); it is unreachable with any real MoE checkpoint and exists only to fail safe. A
		// blob that trips it should be declined to the staged/CPU path upstream.
		r.cacheExperts = false
		fmt.Fprintf(os.Stderr, "[cuda] C′ cache: routed layer %d reports zero per-expert bytes — expert cache disabled\n", moeLayers[0])
		return nil
	}
	strides := r.slotStrides(moeLayers[0])
	if f0, _, e0 := r.dev.Context().MemInfo(); e0 == nil {
		r.dbgFreeBefore = f0 // recording only
	}
	if free, _, err := r.dev.Context().MemInfo(); err == nil {
		// capSlots is the ONE implementation of this arithmetic: a former inline copy let the gate corroborate a
		// parallel copy, so a change to either was uncontradicted by the other (the sibling-drift class,
		// docs/parity-coverage-policy.md).
		//
		// budget, not free, goes into capSlots: r.extraBytes reserves room for a companion attach (--drafter) coming after
		// this build (Model.ExtraResidentBytes's doc; docs/tasks/task-fit-to-hardware.md §2). Without it a 26B auto-sized
		// to 31 slots/layer left the drafter no room. 0 when nothing attaches, so budget == free then.
		budget := reservedBudget(int64(free), r.extraBytes)
		fit, decline := capSlots(budget, int64(len(moeLayers)), strides, r.topK, r.cacheSlots)
		if decline {
			// FLOOR. topK slots is the minimum that can work (one token's routed set must be simultaneously resident), so if
			// even that does not fit, DECLINE naming the shortfall rather than clamp UP to a figure just computed not to fit:
			// that would allocate it and surface the failure later as CUDA_ERROR_OUT_OF_MEMORY from cuLaunchKernel or an
			// empty generation, neither of which points back here.
			need := slotRequirement(r.topK, int64(len(moeLayers)), strides)
			if r.extraBytes > 0 {
				return fmt.Errorf("expert cache (C′) cannot fit its MINIMUM: top-%d routed experts across %d MoE "+
					"layers need %.2f GB of slots, but only %.2f GB is free (%.2f GB reserved for a companion "+
					"attach, --drafter) — %d slots/layer fit. Free VRAM, lower --ctx, or drop "+
					"GOINFER_MOE_CACHE_EXPERTS and use a card that holds the experts outright",
					r.topK, len(moeLayers), float64(need)/1e9, float64(free)/1e9, float64(r.extraBytes)/1e9, fit)
			}
			return fmt.Errorf("expert cache (C′) cannot fit its MINIMUM: top-%d routed experts across %d MoE "+
				"layers need %.2f GB of slots, but only %.2f GB is free — %d slots/layer fit. Free VRAM, "+
				"lower --ctx, or drop GOINFER_MOE_CACHE_EXPERTS and use a card that holds the experts outright",
				r.topK, len(moeLayers), float64(need)/1e9, float64(free)/1e9, fit)
		}
		if fit < r.cacheSlots {
			if r.extraBytes > 0 {
				fmt.Fprintf(os.Stderr, "[cuda] C′ cache: %d slots/layer would need %.1f GB VRAM but only %.1f GB free "+
					"(%.1f GB of %.1f GB total reserved for a companion attach, --drafter) — capping to %d (%.1f GB)\n",
					r.cacheSlots, float64(slotRequirement(r.cacheSlots, int64(len(moeLayers)), strides))/1e9,
					float64(budget)/1e9, float64(r.extraBytes)/1e9, float64(free)/1e9,
					fit, float64(slotRequirement(fit, int64(len(moeLayers)), strides))/1e9)
			} else {
				fmt.Fprintf(os.Stderr, "[cuda] C′ cache: %d slots/layer would need %.1f GB VRAM but only %.1f GB free — "+
					"capping to %d (%.1f GB)\n", r.cacheSlots,
					float64(slotRequirement(r.cacheSlots, int64(len(moeLayers)), strides))/1e9, float64(free)/1e9,
					fit, float64(slotRequirement(fit, int64(len(moeLayers)), strides))/1e9)
			}
			r.cacheSlots = fit
		}
	}
	// probe issues each allocation; it is a pass-through.
	probe := func(_ int, alloc func()) { alloc() }
	// Issue LARGEST FIRST across all layers, not group by group. The total is identical (capSlots and the granularity
	// form are untouched; only the order moves). This is a packing improvement, not a fix: the real constraint is a
	// driver ALLOCATION FLOOR (~144 MiB that cuMemGetInfo reports free and cuMemAlloc will not hand out at any request
	// size, TestAllocFloor), and the leftover after allocSlots must exceed it, which slotMarginBytes provides. The
	// order is kept on its measured merit (it drains ~27 MiB further before hitting the floor), not on the refuted
	// contiguity theory it was introduced for (docs/code-notes/cuda.md#allocSlots.order).
	//
	// Sort by MEASURED request size, not an assumed stride order: per-layer geometry is not guaranteed uniform (Gemma
	// 4's KV widths differ per layer, and slotBytesPerLayer exists because layer 0 can be dense). A stable sort keeps
	// layer order within each size class, so the sequence stays deterministic.
	type slotReq struct {
		w    *cudaWQ
		n    int // element count for the allocator
		size int // bytes, for ordering and for the probe
		isW  bool
	}
	reqs := make([]slotReq, 0, 4*len(moeLayers))
	for _, i := range moeLayers {
		L := &r.layers[i]
		for _, w := range []*cudaWQ{&L.expGU, &L.expDown} {
			reqs = append(reqs,
				slotReq{w, r.cacheSlots * w.perExpertW, r.cacheSlots * w.perExpertW * 4, true},
				slotReq{w, r.cacheSlots * w.perExpertS, r.cacheSlots * w.perExpertS * 2, false})
		}
	}
	sort.SliceStable(reqs, func(a, b int) bool { return reqs[a].size > reqs[b].size })
	for _, q := range reqs {
		if q.isW {
			probe(q.size, func() { q.w.W = r.au32(q.n) })
		} else {
			probe(q.size, func() { q.w.ws16 = gpu.NewBufferLenOf[uint16](r.dev, q.n) })
		}
		// Recording only: the REQUESTED byte size of each allocation, in issue (sorted) order, so predicted consumption
		// can be recomputed with per-buffer 2 MiB rounding instead of a raw sum. Every gate on it reads it
		// order-independently (counts, distinct sizes, occurrences, sum).
		r.dbgAllocSizes = append(r.dbgAllocSizes, q.size)
	}
	for _, i := range moeLayers {
		r.layers[i].expCache = newExpertCache(r.nE, r.cacheSlots)
	}
	if f1, _, e1 := r.dev.Context().MemInfo(); e1 == nil {
		r.dbgFreeAfter = f1 // recording only
	}
	// The inline and capSlots figures are one value: there is one implementation.
	r.dbgSlotsInline = r.cacheSlots
	r.dbgPredInline = slotRequirement(r.cacheSlots, int64(len(moeLayers)), strides)
	r.dbgSlotsCapSlots = r.cacheSlots
	r.dbgPredCapSlots = r.dbgPredInline
	return nil
}

// expertCache is C′ step 2's per-layer LRU slot residency: which cached expert occupies which of the nSlots device
// slots, so a routed expert already resident skips its H2D DMA. nSlots = topK is the step-1 staging degenerate
// case (every token evicts, no reuse); nSlots > topK gives cross-token reuse. LRU because the router signal is a
// stationary skew where recency is a sufficient statistic for frequency (docs/completed/task-moe-streaming.md).
// expGU and expDown share the slot index (loaded together), so the cache is per-LAYER, not per-projection.
type expertCache struct {
	nSlots       int
	slotOf       []int32  // [nE]     expert → slot, -1 = not resident
	inSlot       []int32  // [nSlots] slot → expert, -1 = empty
	used         []uint64 // [nSlots] last-touch clock (LRU)
	clock        uint64
	hits, misses uint64 // reuse accounting (a miss is one expert's H2D DMA)

	// DEMAND accounting answers a different question from hits/misses: those say how much DMA reuse SAVED, these say
	// how much the caller ASKED FOR per staging event. stages counts loadRoutedExperts calls on this layer; distinct
	// sums the UNIQUE experts each stage requested. distinct/stages separates "speculation blows the slot budget"
	// from "speculation is just slow here": a verify that presented its whole width to the pager at once would push it
	// ABOVE topK and grow with the width, while one that stages position by position pins it AT topK.
	stages   uint64
	distinct uint64
}

func newExpertCache(nE, nSlots int) *expertCache {
	c := &expertCache{nSlots: nSlots, slotOf: make([]int32, nE), inSlot: make([]int32, nSlots), used: make([]uint64, nSlots)}
	for i := range c.slotOf {
		c.slotOf[i] = -1
	}
	for i := range c.inSlot {
		c.inSlot[i] = -1
	}
	return c
}

// admit returns the slot holding expert e, evicting the LRU slot on a miss. hit=false means the
// caller must DMA e's weights into `slot` before the GEMV reads it. An expert admitted earlier in
// the SAME token is the newest, so it is never the victim (nSlots ≥ topK guarantees room).
func (c *expertCache) admit(e uint32) (slot int, hit bool) {
	c.clock++
	if s := c.slotOf[e]; s >= 0 {
		c.used[s] = c.clock
		c.hits++
		return int(s), true
	}
	c.misses++
	victim, oldest := 0, ^uint64(0)
	for s := 0; s < c.nSlots; s++ {
		if c.inSlot[s] < 0 { // an empty slot always wins
			victim = s
			break
		}
		if c.used[s] < oldest {
			oldest = c.used[s]
			victim = s
		}
	}
	if old := c.inSlot[victim]; old >= 0 {
		c.slotOf[old] = -1
	}
	c.inSlot[victim] = int32(e)
	c.slotOf[e] = int32(victim)
	c.used[victim] = c.clock
	return victim, false
}

// unadmit marks slot empty and forgets expert e, so the next admit(e) misses and re-uploads. admit commits the
// slot mapping BEFORE the DMA, which is what lets the caller batch a layer's copies into one UploadBatch; if that
// upload fails the cache would claim e is resident in a slot holding the EVICTED expert's bytes, and the next
// token routing e would hit, skip the DMA and run the wrong expert with no error anywhere. Rolled back to EMPTY
// rather than to the evicted expert: a failed batch may have completed some copies, so the slot's contents are
// unknown.
func (c *expertCache) unadmit(slot int, e uint32) {
	if c.inSlot[slot] == int32(e) {
		c.inSlot[slot] = -1
	}
	if c.slotOf[e] == int32(slot) {
		c.slotOf[e] = -1
	}
}

// appendExpertSlot QUEUES one expert's weight+scales copy from the pinned host source into device slot `slot`. It
// does not upload: loadRoutedExperts submits the layer's whole batch with a single gpu.UploadBatch, so one
// synchronize covers every miss in the layer instead of two per miss (each gpu.Upload ends in a full device
// Synchronize, and a MoE decode token loads ~120 slots). UploadBatch still synchronizes before it returns, so the
// guarantee equals Upload's, including the race that sync exists for (non-blocking streams unordered against the
// null stream); nothing moves into a caller's hands.
func (r *cudaResident) appendExpertSlot(w *cudaWQ, e, slot int) {
	srcW, srcS := w.srcW.Bytes(), w.srcS.Bytes()
	wOff, wLen := e*w.perExpertW*4, w.perExpertW*4
	sOff, sLen := e*w.perExpertS*2, w.perExpertS*2
	r.expBatch = append(r.expBatch,
		gpu.HostCopy{Dst: w.W.At(slot * w.perExpertW * 4), Src: srcW[wOff : wOff+wLen]},
		gpu.HostCopy{Dst: w.ws16.At(slot * w.perExpertS * 2), Src: srcS[sOff : sOff+sLen]},
	)
	if r.cacheProf {
		r.profWBytes += uint64(wLen)
		r.profWCalls++
		r.profSBytes += uint64(sLen)
		r.profSCalls++
	}
}

// loadRoutedExperts reads back the router's idx (device→host — C′'s acknowledged per-layer sync),
// admits each routed expert into the layer's slot cache (DMAing only cache misses), and uploads the
// per-token slot ids the GEMV binds as `idx` (slot j ← the slot now holding routed expert j).
func (r *cudaResident) loadRoutedExperts(L *cudaLayer) error {
	// Timing seam (GOINFER_MOE_CACHE_PROF). This is the only host round trip on the decode path, once per MoE layer
	// per token, and it splits into three costs that need different fixes: stall (the stream drain waiting for the
	// router; fixed by removing the round trip, not by making it faster), host (the LRU bookkeeping; ordinary
	// optimization) and dma (the H2D of missed experts; more slots or fewer bytes). Off by default, one branch when
	// off; it adds no syncs, because the stall it measures is a sync that already exists.
	var t0 time.Time
	if r.cacheProf {
		t0 = time.Now()
	}
	if r.overlap {
		// Wait for the ROUTER, not the stream: everything issued after it keeps executing while the
		// idx is read back and the misses are copied on dmaQ.
		if e := r.evRoute.Sync(); e != nil {
			return e
		}
	} else if e := r.stream.Sync(); e != nil {
		return e
	}
	if r.cacheProf {
		r.profStall += time.Since(t0)
		t0 = time.Now()
	}
	if e := gpu.Download(r.rIdx, r.hostIdx[:r.topK]); e != nil {
		return e
	}
	if r.routeRecord != nil {
		r.routeRecord(L.idx, r.hostIdx[:r.topK])
	}
	c := L.expCache
	// Demand accounting (see expertCache), counted BEFORE admission on purpose: it must describe what this stage
	// REQUESTED, not what the cache already held, or it would fall as the cache warmed. topK is ~8, so the quadratic
	// scan is negligible next to the DMA it describes.
	c.stages++
	for j := 0; j < r.topK; j++ {
		dup := false
		for k := 0; k < j; k++ {
			if r.hostIdx[k] == r.hostIdx[j] {
				dup = true
				break
			}
		}
		if !dup {
			c.distinct++
		}
	}
	r.expBatch = r.expBatch[:0] // reused across layers; the slice keeps its capacity
	// Every admit made in THIS call is provisional until the batch lands. On any error below they are rolled back, so
	// a failed upload cannot leave the cache asserting residency for bytes that were never copied.
	r.pendingAdmits = r.pendingAdmits[:0]
	for j := 0; j < r.topK; j++ {
		e := r.hostIdx[j]
		slot, hit := c.admit(e)
		r.hostSlot[j] = uint32(slot)
		if r.overlap {
			r.missEv[j] = -1
		}
		if !hit {
			r.pendingAdmits = append(r.pendingAdmits, pendingAdmit{slot: slot, expert: e})
			var td time.Time
			if r.cacheProf {
				r.profHost += time.Since(t0)
				td = time.Now()
			}
			if r.overlap {
				// Both copies on dmaQ, then the event this rank's launches will wait on. A later rank
				// routed to the same expert is ordered behind this rank's wait by the stream itself.
				k := len(r.pendingAdmits) - 1
				r.missEv[j] = k
				err := r.dmaExpertSlot(&L.expGU, int(e), slot)
				if err == nil {
					err = r.dmaExpertSlot(&L.expDown, int(e), slot)
				}
				if err == nil {
					err = r.dmaQ.Record(r.evMiss[k])
				}
				if err != nil {
					r.rollbackAdmits(c)
					return err
				}
			} else {
				r.appendExpertSlot(&L.expGU, int(e), slot)
				r.appendExpertSlot(&L.expDown, int(e), slot)
			}
			if r.cacheProf {
				r.profDMA += time.Since(td) // overlap: enqueue time, not transfer time
				t0 = time.Now()
			}
		}
	}
	if r.overlap {
		// The slot-id table rides r.stream, ordered before segC by the stream; its pinned source is
		// next rewritten only after the next layer's evRoute wait, which follows this layer's segC.
		hs := r.slotIdxHost.Slice()
		for j := 0; j < r.topK; j++ {
			v := r.hostSlot[j]
			hs[4*j], hs[4*j+1], hs[4*j+2], hs[4*j+3] = byte(v), byte(v>>8), byte(v>>16), byte(v>>24)
		}
		if e := r.stream.UploadAsync(r.slotIdx, r.slotIdxHost); e != nil {
			r.rollbackAdmits(c)
			return e
		}
		if r.cacheProf {
			r.profHost += time.Since(t0)
			r.profCalls++
		}
		return nil
	}
	// One synchronize for the WHOLE layer: every expert-slot miss plus the per-token slot-index upload the GEMV reads
	// this round's routing from, folded into the SAME batch (slotIdx is always present and tiny, so keeping it apart
	// would only add a synchronize). All destinations are slot buffers on the one resident context, so UploadBatch's
	// mixed-context refusal should never fire; if it does, the layer's buffers are not what this code believes.
	r.expBatch = append(r.expBatch, gpu.HostCopy{Dst: r.slotIdx, Src: u32bytes(r.hostSlot[:r.topK])})
	var tb time.Time
	if r.cacheProf {
		r.profHost += time.Since(t0)
		tb = time.Now()
	}
	e := gpu.UploadBatch(r.expBatch)
	if e != nil {
		// The expert bytes and/or the slot-index table may be partially applied: the cheap conservative move is to make
		// the next call re-establish both.
		r.rollbackAdmits(c)
	}
	if r.cacheProf {
		r.profBatchTime += time.Since(tb)
		r.profSyncCalls++
		r.profDMA += time.Since(tb)
		r.profCalls++
	}
	return e
}

// pendingAdmit is one provisional slot claim made by loadRoutedExperts before its DMA.
type pendingAdmit struct {
	slot   int
	expert uint32
}

// rollbackAdmits undoes every admit made in the current loadRoutedExperts call.
func (r *cudaResident) rollbackAdmits(c *expertCache) {
	for _, p := range r.pendingAdmits {
		c.unadmit(p.slot, p.expert)
	}
	r.pendingAdmits = r.pendingAdmits[:0]
}

// CacheProfForTest reports the C′ round-trip decomposition (stall / host / dma) and the call
// count. Zero unless GOINFER_MOE_CACHE_PROF is set.
func (r *cudaResident) CacheProfForTest() (stall, host, dma time.Duration, calls uint64) {
	return r.profStall, r.profHost, r.profDMA, r.profCalls
}

// profMark closes one decode class: drains the stream, books the time since the previous mark
// into acc, and restarts the clock. No-op unless cacheProf. A drain error is parked in launchErr
// (the sticky accumulator launchToken already returns) rather than dropped.
func (r *cudaResident) profMark(acc *time.Duration) {
	if !r.cacheProf {
		return
	}
	if e := r.stream.Sync(); e != nil && r.launchErr == nil {
		r.launchErr = e
	}
	now := time.Now()
	*acc += now.Sub(r.profT0)
	r.profT0 = now
}

// initOverlap builds the overlap path's resources: the DMA stream, the router event, one event per
// possible miss rank.
func (r *cudaResident) initOverlap() error {
	r.dmaQ = r.dev.NewCommandQueue()
	ev, e := r.dev.NewEvent()
	if e != nil {
		return e
	}
	r.evRoute = ev
	r.evMiss = make([]gpu.Event, r.topK)
	for k := range r.evMiss {
		if r.evMiss[k], e = r.dev.NewEvent(); e != nil {
			return e
		}
	}
	r.missEv = make([]int, r.topK)
	for j := range r.missEv {
		r.missEv[j] = -1
	}
	return nil
}

// recordRouted marks the router's position on r.stream — the ONLY thing loadRoutedExperts waits for
// in overlap mode. Called by every MoE pre-half right after its route kernel.
func (r *cudaResident) recordRouted() error {
	if !r.overlap {
		return nil
	}
	return r.stream.Record(r.evRoute)
}

// waitMiss makes r.stream wait, device-side, for rank j's expert DMA if that rank missed the cache
// this call. Hit ranks — and every rank when overlap is off — pass straight through, so the ranks
// ahead of a miss execute under its DMA in exactly the order they always did.
func (r *cudaResident) waitMiss(j int) error {
	if !r.overlap || r.missEv[j] < 0 {
		return nil
	}
	return r.stream.Wait(r.evMiss[r.missEv[j]])
}

// dmaExpertSlot enqueues one expert's weight+scales copy on dmaQ, stream-ordered and unsynchronized —
// the overlap-mode twin of appendExpertSlot.
func (r *cudaResident) dmaExpertSlot(w *cudaWQ, e, slot int) error {
	wOff, wLen := e*w.perExpertW*4, w.perExpertW*4
	sOff, sLen := e*w.perExpertS*2, w.perExpertS*2
	// UploadAsyncAtFrom, not UploadAsyncAt+.Host(): srcW/srcS may be either MappedHostBuffer origin (NewMappedHostBuffer
	// or, under GOINFER_MOE_PIN_REGISTER, RegisterMappedHostBuffer) and .Host() returns nil for the latter, which has no
	// *gpu.HostBuffer to hand UploadAsyncAt, only Bytes(); UploadAsyncAtFrom reads through either origin.
	if err := r.dmaQ.UploadAsyncAtFrom(w.W.At(slot*w.perExpertW*4), w.srcW, wOff, wLen); err != nil {
		return err
	}
	if err := r.dmaQ.UploadAsyncAtFrom(w.ws16.At(slot*w.perExpertS*2), w.srcS, sOff, sLen); err != nil {
		return err
	}
	if r.cacheProf {
		r.profWBytes += uint64(wLen)
		r.profWCalls++
		r.profSBytes += uint64(sLen)
		r.profSCalls++
	}
	return nil
}

// SetOverlapForTest flips the overlap path between tokens for an A/B on one loaded model. Returns
// false (and changes nothing) if the model was loaded without the overlap resources.
func (r *cudaResident) SetOverlapForTest(on bool) bool {
	if r.evMiss == nil {
		return false
	}
	r.overlap = on
	return true
}

// headArgProf times the three phases of a batched head-to-argmax tail (stream drain, M×vocab D2H, host argmax loop)
// at its two sites, the drafter's DraftTokens and the verify's batchedHeadArgmax. Nil by default.
type headArgProf struct {
	sync, dl, host time.Duration
	calls, rows    int
}

// SetHeadArgProfForTest arms (or disarms) the head-argmax phase profiler.
func (r *cudaResident) SetHeadArgProfForTest(on bool) {
	if on {
		r.headProf = &headArgProf{}
	} else {
		r.headProf = nil
	}
}

// HeadArgProfForTest reports the accumulated phases; zero when disarmed.
func (r *cudaResident) HeadArgProfForTest() (sync, dl, host time.Duration, calls, rows int) {
	if r.headProf == nil {
		return 0, 0, 0, 0, 0
	}
	p := r.headProf
	return p.sync, p.dl, p.host, p.calls, p.rows
}

// hostArgmaxRows is the serial host argmax both tails share: strict > from element 0, so ties go
// to the lowest index — the same tie-break argmax_reduce implements on the device.
func (r *cudaResident) hostArgmaxRows(host []float32, M int, ids []int) {
	for m := range M {
		row := host[m*r.vocab : (m+1)*r.vocab]
		bi, bv := 0, row[0]
		for i, v := range row {
			if v > bv {
				bi, bv = i, v
			}
		}
		ids[m] = bi
	}
}

// argmaxRows is the batched head tail's reduction: argmax_rows over the M rows of logits on the device, one drain,
// an M-int readback, in place of downloading M×vocab floats and looping on the host. Same strided scan and
// tie-break as argmax_reduce, so row m gets the token the host loop (strict >, lowest index) gets. hostArgmaxPath
// keeps the host tail for the A/B; argCheck runs both and compares.
func (r *cudaResident) argmaxRows(logits, idxBuf Buffer, M int, ids []int) error {
	var t0 time.Time
	if r.hostArgmaxPath {
		if r.headProf != nil {
			t0 = time.Now()
		}
		if e := r.stream.Sync(); e != nil {
			return e
		}
		if r.headProf != nil {
			r.headProf.sync += time.Since(t0)
			t0 = time.Now()
		}
		host := make([]float32, M*r.vocab)
		if e := gpu.Download(logits, host); e != nil {
			return e
		}
		if r.headProf != nil {
			r.headProf.dl += time.Since(t0)
			t0 = time.Now()
		}
		r.hostArgmaxRows(host, M, ids)
		if r.headProf != nil {
			r.headProf.host += time.Since(t0)
			r.headProf.calls++
			r.headProf.rows += M
		}
		return nil
	}
	if e := r.launch(r.fArgRows, LaunchConfig{GridX: uint32(M), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256*4 + 256*4},
		Arg(logits), gpu.ArgValue(int32(r.vocab)), gpu.ArgValue(int32(M)), Arg(idxBuf)); e != nil {
		return e
	}
	if r.headProf != nil {
		t0 = time.Now()
	}
	if e := r.stream.Sync(); e != nil {
		return e
	}
	if r.headProf != nil {
		r.headProf.sync += time.Since(t0)
		t0 = time.Now()
	}
	out := make([]int32, M)
	if e := gpu.Download(idxBuf, out); e != nil {
		return e
	}
	for m := range out {
		ids[m] = int(out[m])
	}
	if r.headProf != nil {
		r.headProf.dl += time.Since(t0)
		r.headProf.calls++
		r.headProf.rows += M
	}
	if r.argCheck {
		host := make([]float32, M*r.vocab)
		if e := gpu.Download(logits, host); e != nil {
			return e
		}
		ref := make([]int, M)
		r.hostArgmaxRows(host, M, ref)
		for m := range ref {
			if ref[m] != ids[m] {
				return fmt.Errorf("cuda: argmax_rows row %d = %d, host loop = %d (logit %g vs %g)",
					m, ids[m], ref[m], host[m*r.vocab+ids[m]], host[m*r.vocab+ref[m]])
			}
		}
		r.argChecks += M
	}
	return nil
}

// SetHostArgmaxForTest forces (true) or releases (false) the host argmax tail, the A/B's do-nothing arm.
func (r *cudaResident) SetHostArgmaxForTest(host bool) { r.hostArgmaxPath = host }

// SetArgmaxCheckForTest arms the row-for-row device-vs-host comparison inside argmaxRows.
func (r *cudaResident) SetArgmaxCheckForTest(on bool) { r.argCheck = on; r.argChecks = 0 }

// ArgmaxChecksForTest reports how many rows the check mode compared (all agreed, or the call errored).
func (r *cudaResident) ArgmaxChecksForTest() int { return r.argChecks }

// DecodeClassProfForTest reports the per-class g4moe decode split (see profMark). Zero unless
// GOINFER_MOE_CACHE_PROF is set.
func (r *cudaResident) DecodeClassProfForTest() (attn, dense, router, clear, rt, segC, head time.Duration) {
	return r.profAttn, r.profDense, r.profRouter, r.profClear, r.profRT, r.profSegC, r.profHead
}

// UploadProfForTest reports the expert-DMA split: the big weight copies vs the tiny scale copies, by bytes moved
// and COPY count. Zero unless GOINFER_MOE_CACHE_PROF is set. It reports no per-kind elapsed times: the copies are
// queued by appendExpertSlot and issued together by one UploadBatch, so there is no per-kind upload to time; the
// transfer+sync time is BatchProfForTest's.
func (r *cudaResident) UploadProfForTest() (wB, sB, wC, sC uint64) {
	return r.profWBytes, r.profSBytes, r.profWCalls, r.profSCalls
}

// BatchProfForTest reports the batched-upload cost: total time inside gpu.UploadBatch and the
// number of SYNCHRONIZES it performed (one per layer with at least one miss). This is the quantity
// the batching change moves — copy count is unchanged, sync count is not. Zero unless
// GOINFER_MOE_CACHE_PROF is set.
func (r *cudaResident) BatchProfForTest() (batchTime time.Duration, syncCalls uint64) {
	return r.profBatchTime, r.profSyncCalls
}

// THERE ARE TWO INDEX SPACES HERE AND THEY DIVERGE ONLY WHEN EXPERT CACHING IS ON. Keeping them
// as two named accessors rather than one is the whole guard: with caching off they return the
// same buffer, so a site that binds the wrong one is correct in every configuration anyone has
// run, and wrong — silently, with plausible logits — in the one configuration that needs it.
//
//	expIdx        WHERE the weights live  → slot ids when caching, expert ids otherwise
//	expertBiasIdx WHICH expert is running → ALWAYS expert ids
//
// expIdx is the idx argument the expert GEMVs bind: the slot ids when caching (slot j holds
// routed expert j), else the router's real rIdx (fully-resident path).
func (r *cudaResident) expIdx() Buffer {
	if r.cacheExperts {
		return r.slotIdx
	}
	return r.rIdx
}

// expertBiasIdx is the index for PER-EXPERT TABLES that are uploaded ONCE for all experts and indexed on the
// device (today only gpt-oss's [nExpert][2*I] gate||up bias table). It is always the router's real expert ids,
// NEVER the slot ids: the table is expert-indexed and does not move when an expert is streamed into a slot.
// Binding expIdx here is a silent defect: glu_quant_gptoss does `biasGU + idx[slot]*2*I`, so with caching on it
// selects the bias row by SLOT id, the wrong expert's gate/up biases, with no error and plausible output
// (docs/code-notes/cuda.md#expertBiasIdx).
func (r *cudaResident) expertBiasIdx() Buffer { return r.rIdx }

func u32bytes(v []uint32) []byte {
	if len(v) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&v[0])), len(v)*4)
}

func u16bytes(v []uint16) []byte {
	if len(v) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&v[0])), len(v)*2)
}

// mapBytes stages src as the C′ DMA source: pinned (device-mapped) host memory holding exactly src's bytes. src is
// caller-owned and already fully populated (the merged expert-stack slice packWeight built), and never touched
// again by anyone else, which is what makes the register-in-place arm safe: the MappedHostBuffer keeps src's
// backing array reachable for as long as the pin lives. Register in place is the default
// (GOINFER_MOE_PIN_REGISTER=0 restores allocate-then-copy): it pins the caller's own bytes, so the DMA source is
// byte-for-byte the same either way and no numerics are at stake (docs/code-notes/cuda.md#mapBytes.register).
// Panics on the UVA guard failing: eligibility already asserted a UVA device, so a failure here is a broken
// invariant, not a runtime condition.
func (r *cudaResident) mapBytes(src []byte) *gpu.MappedHostBuffer {
	if r.knobValue("GOINFER_MOE_PIN_REGISTER") != "0" { // register in place, above
		mb, err := r.dev.RegisterMappedHostBuffer(src)
		if err != nil {
			panic(fmt.Sprintf("cacheWQ: RegisterMappedHostBuffer(%d): %v", len(src), err))
		}
		return mb
	}
	mb, err := r.dev.NewMappedHostBuffer(len(src))
	if err != nil {
		panic(fmt.Sprintf("cacheWQ: NewMappedHostBuffer(%d): %v", len(src), err))
	}
	copy(mb.Bytes(), src)
	return mb
}

func (r *cudaResident) do(j func() error) error { r.reqCh <- j; return <-r.ackCh }

// runJob is the executor goroutine's PANIC BOUNDARY: it runs one job and converts a panic into an ordinary error,
// so the pinned thread survives and `do` returns to its caller. It has to live here and not at a call site: every
// job runs on the executor goroutine but `do` blocks on a different one, so a `defer recover()` in BuildResident
// or any caller cannot catch a panic raised inside j(). The case is reachable by design: gpu.NewBufferLenOf
// PANICS on allocation failure per its contract, and prefillCore allocates M*(2*inter+2*hidden+...) floats,
// hundreds of MB on a long prompt; without this a long prompt against a nearly-full card killed the serve process
// at the one seam whose job is to decline to the sequential path.
//
// The recovered error is deliberately NOT wrapped in errPrefillDeclined here: `do` is shared by every job (setup,
// decode, prefill), and only the prefill caller knows a decline is the right response. prefillCore wraps it at its
// own boundary.
func runJob(j func() error) (err error) {
	defer func() {
		if p := recover(); p != nil {
			// The STACK, not just the value: a recovered executor panic becomes a resident DECLINE printed once to stderr, and
			// "device allocation failed (0 bytes)" with no frame is unactionable. The stack is a few KB on a path that runs at
			// most once per model load.
			err = fmt.Errorf("cuda: executor job panicked: %v\n%s", p, debug.Stack())
		}
	}()
	return j()
}

// errKVWontFit marks the one setup failure that must NOT degrade quietly to the staged path: an
// explicitly configured resident context whose KV does not fit. BuildResident turns it into a hard
// startup error instead of a decline. An UNCONFIGURED (default-cap) miss stays a decline, because
// that is the historical behaviour for "this device cannot host this model".
var errKVWontFit = errors.New("resident KV does not fit in device memory")

// kvWontFitError carries the full operator-facing message while still matching errKVWontFit under
// errors.Is. A plain %w wrap would append the sentinel's text to a message that already says all of
// this, so the reader sees it twice — the classification must not leak into the prose.
type kvWontFitError struct{ msg string }

func (e *kvWontFitError) Error() string        { return e.msg }
func (e *kvWontFitError) Is(target error) bool { return target == errKVWontFit }

// reservedBudget is what allocSlots' elastic expert-cache sizing gets to work with once a companion allocation
// (extraBytes: a --drafter attaching after this build, Model.ExtraResidentBytes's doc) has reserved its share of
// free VRAM. Split out, like fitsWeightsBudget below, so the clamp-at-zero (extraBytes can exceed free on a card
// too small for the pairing, and capSlots must then see 0, not a negative budget) is unit-testable without a
// device. extraBytes <= 0 returns free unchanged.
func reservedBudget(free, extraBytes int64) int64 {
	b := free - extraBytes
	if b < 0 {
		return 0
	}
	return b
}

// fitsWeightsBudget is checkWeightsFit's arithmetic alone, split out so it is unit-testable with
// synthetic numbers rather than a checkpoint large enough to swing a real device's verdict — same
// reasoning as metal/backend.go's fitsResidentBudget, which this mirrors. Unknown inputs (need or
// free <= 0) always fit: the guard's failure mode must be letting a doomed load through, never
// refusing one that would have run.
func fitsWeightsBudget(need, free int64) bool {
	if need <= 0 || free <= 0 {
		return true
	}
	return need+ctxCapMarginBytes <= free
}

// checkWeightsFit fails the load EARLY, right after the device exists and before any kernel compile or weight
// upload, when the model's FIXED weight bytes alone exceed free VRAM. Without it a dense model whose weights do
// not fit ran the whole compile+upload sequence and failed on whichever allocation first did not fit: a raw
// driver error, not a decline naming the numbers (checkKVFits' point, but it only runs after the weights are up).
//
// `need` is decoder.Model.ResidentDenseWeightBytesFor, deliberately NOT ResidentWeightBytesPaged(0): routed MoE
// experts have their own elastic sizing (capSlots), a bisection against LIVE free VRAM that runs after the dense
// weights are uploaded. Pricing experts at their full unpaged size here would decline a model whose experts are
// about to be capped down to something that fits.
func (r *cudaResident) checkWeightsFit(m *decoder.Model) error {
	need := m.ResidentDenseWeightBytesFor("cuda") // device bytes: an untied embedding table stays on the host
	// A companion allocation (a --drafter's weights) attaches on this same device AFTER BuildResident returns: price
	// it here too (m.ExtraResidentBytes's doc), or a model whose weights alone fit could leave the drafter nowhere to
	// go. 0 when nothing attaches.
	extra := m.ExtraResidentBytes()
	free, _, err := r.dev.Context().MemInfo()
	if err != nil {
		// No MemInfo ⇒ no fit check possible. Same non-failure as checkKVFits: the allocation
		// itself still errors if it truly cannot fit.
		return nil
	}
	if fitsWeightsBudget(need+extra, int64(free)) {
		return nil
	}
	if extra > 0 {
		return fmt.Errorf("cuda: dense weights need %.2f GB plus %.2f GB reserved for a companion "+
			"attach (--drafter) but only %.2f GB is free on the device (plus %.0f MB reserved for "+
			"driver and decode scratch) — use a smaller/more-quantized model, or the staged/CPU path",
			float64(need)/1e9, float64(extra)/1e9, float64(free)/1e9, float64(ctxCapMarginBytes)/(1<<20))
	}
	return fmt.Errorf("cuda: dense weights need %.2f GB but only %.2f GB is free on the device "+
		"(plus %.0f MB reserved for driver and decode scratch) — use a smaller/more-quantized "+
		"model, or the staged/CPU path",
		float64(need)/1e9, float64(free)/1e9, float64(ctxCapMarginBytes)/(1<<20))
}

// checkKVFits fails the LOAD if the KV the configured cap implies does not fit beside what is already on the device.
// It runs after the weights are uploaded and before the K/V caches are allocated, so `free` is genuinely "what is
// left for KV". The point is the failure MODE, not the arithmetic: without it an over-large cap surfaces as an
// allocation failure part-way through sizing the per-layer caches, or as an OOM mid-decode after the server reported
// ready. Naming the number at startup turns a production incident into a config error: the message states what was
// asked for, what it costs, and what is actually free.
//
// It also sets r.kvSlotsN, MC1's slot count: the request, clamped (kvSlotsFit) to the slots whose KV fits the same
// budget, and logged when clamped. The first slot is the one this check refuses; further ones only shrink.
func (r *cudaResident) checkKVFits() error {
	need := kvBytesForCap(r.ctxCap, r.layers)
	r.kvSlotsN = 1
	free, err := cudaFreeVRAM(r)
	if err != nil {
		// No MemInfo ⇒ no fit check possible. Do not fail the load on that: the default cap has
		// always been allocated without one, and a hard failure here would regress every driver
		// that does not report memory. The allocation itself still errors if it truly cannot fit.
		// MC1's further slots are not allocated blind: one slot, as before.
		return nil
	}
	// r.extraBytes reserves room for a companion attach (--drafter) coming after this build —
	// see the field's own doc comment. 0 for every load without one, so this is unchanged then.
	if !r.ctxExplicit && r.ctxCap > cudaCtxCapDefault {
		// An unpinned context that even ONE slot of KV overshoots (by the build's own scratch, which ctxForSlots cannot
		// see) gives up positions, never below cudaCtxCapDefault, instead of declining the whole resident path to the CPU
		// with a log line. Only the multi-slot arm below used to shrink to what the card holds, so a plain decoder.Load of
		// a 7B or Gemma-3 on an 8 GB card fell to the CPU (docs/code-notes/cuda.md#checkKVFits.trim). serve plans its own
		// context and never hit it.
		c, ok := trimUnpinnedCtx(r.ctxCap, need, int64(free), r.extraBytes, ctxCapMarginBytes)
		if r.kvWin {
			// trimUnpinnedCtx divides by a per-position cost; a windowed layer's cost is not per position, so search the capacity instead
			ok = false
			if need+r.extraBytes+ctxCapMarginBytes > int64(free) {
				if cc, fits := kvCapForBytes(int64(free)-r.extraBytes-ctxCapMarginBytes, cudaCtxCapDefault, r.ctxCap-1, r.layers); fits {
					c, ok = cc, true
				}
			}
		}
		if ok {
			fmt.Fprintf(os.Stderr, "cuda: resident context %d, trimmed from %d at the build, so one KV slot fits beside the weights (%.0f MB free, less %.0f MB reserved)\n",
				c, r.ctxCap, float64(free)/(1<<20), float64(r.extraBytes+ctxCapMarginBytes)/(1<<20))
			r.ctxCap, need = c, kvBytesForCap(c, r.layers)
		}
	}
	if need+r.extraBytes+ctxCapMarginBytes <= int64(free) {
		// MC1: as many of the requested KV slots as fit the same budget, each another `need`.
		reserve := r.extraBytes + ctxCapMarginBytes
		r.kvSlotsN = kvSlotsFit(r.kvSlotsReq, int64(free), need, reserve)
		if r.kvSlotsN < r.kvSlotsReq && !r.ctxExplicit && r.ctxCap > cudaCtxCapDefault {
			// Slots before context (owner decision): an unpinned context gives up positions, never below cudaCtxCapDefault,
			// until every requested slot fits. ctxForSlots planned this with Plan, which cannot see the build's own scratch and
			// kernel modules (TestResidentDenseBytes_matchesCUDADevice), so its choice can be a few dozen positions long; this
			// trims it against the real free VRAM, before any KV exists. Nothing allocated so far depends on ctxCap exactly:
			// the split-KV score scratch is indexed by the attended span, so a smaller cap uses less of it.
			perPos := need / int64(r.ctxCap) // kvBytesForCap is exactly linear
			c := max(cudaCtxCapDefault, int((int64(free)-reserve)/(int64(r.kvSlotsReq)*perPos)))
			if c < r.ctxCap {
				fmt.Fprintf(os.Stderr, "cuda: resident context %d, trimmed from %d at the build, so the %d requested KV slots "+
					"fit beside the weights (%.0f MB free, less %.0f MB reserved)\n",
					c, r.ctxCap, r.kvSlotsReq, float64(free)/(1<<20), float64(reserve)/(1<<20))
				r.ctxCap, need = c, kvBytesForCap(c, r.layers)
				r.kvSlotsN = kvSlotsFit(r.kvSlotsReq, int64(free), need, reserve)
			}
		}
		if r.kvSlotsN < r.kvSlotsReq {
			fmt.Fprintf(os.Stderr, "cuda: %d resident KV slots of %d requested — each costs %.0f MB of KV at the resident "+
				"context %d, and free VRAM beside the weights (%.0f MB, less %.0f MB reserved) allows %d\n",
				r.kvSlotsN, r.kvSlotsReq, float64(need)/(1<<20), r.ctxCap, float64(free)/(1<<20),
				float64(r.extraBytes+ctxCapMarginBytes)/(1<<20), r.kvSlotsN)
		}
		return nil
	}
	perPos := float64(need) / float64(max(r.ctxCap, 1)) / 1024
	msg := fmt.Sprintf("cuda: resident context %d positions needs %.2f GB of KV "+
		"(%.1f KB/position across %d layers) but only %.2f GB is free on the device beside the weights "+
		"(plus %.0f MB reserved for driver and decode scratch) — lower the serve context setting (-ctx), "+
		"or use a smaller/more-quantized model",
		r.ctxCap, float64(need)/1e9, perPos, len(r.layers), float64(free)/1e9, float64(ctxCapMarginBytes)/(1<<20))
	if r.extraBytes > 0 {
		msg = fmt.Sprintf("cuda: resident context %d positions needs %.2f GB of KV "+
			"(%.1f KB/position across %d layers) plus %.2f GB reserved for a companion attach "+
			"(--drafter) but only %.2f GB is free on the device beside the weights (plus %.0f MB "+
			"reserved for driver and decode scratch) — lower the serve context setting (-ctx), or "+
			"use a smaller/more-quantized model",
			r.ctxCap, float64(need)/1e9, perPos, len(r.layers), float64(r.extraBytes)/1e9,
			float64(free)/1e9, float64(ctxCapMarginBytes)/(1<<20))
	}
	e := &kvWontFitError{msg: msg}
	if !r.ctxExplicit {
		// Default cap: keep the historical decline. Strip the sentinel so BuildResident treats it as
		// an ordinary "cannot host this here" and the staged path takes over, as it always has.
		if r.extraBytes > 0 {
			return fmt.Errorf("cuda: default resident context %d positions does not fit (%.2f GB of KV "+
				"plus %.2f GB reserved for a companion attach, --drafter, %.2f GB free) — staged path",
				r.ctxCap, float64(need)/1e9, float64(r.extraBytes)/1e9, float64(free)/1e9)
		}
		return fmt.Errorf("cuda: default resident context %d positions does not fit (%.2f GB of KV, %.2f GB free) — staged path",
			r.ctxCap, float64(need)/1e9, float64(free)/1e9)
	}
	return e
}

// checkCap guards the resident KV allocation. Every layer's cache is sized r.ctxCap*kvDim, so a write for
// absolute position p lands at kc[p*kvDim ...]; valid positions are [0, r.ctxCap). Writing past it
// (rope_kv/kv_store) is an out-of-bounds DEVICE write: silent memory corruption, and the attention launch's
// shared-mem request eventually exceeds the block limit. Nothing upstream clamps prompt+max_tokens to the cap, so
// return an error here; the decode loop stops on it (model.go) and the caller can fall back to the staged path,
// which handles longer contexts. r.ctxCap is always the capacity the caches were actually SIZED with, read from
// the field the allocation used.
func (r *cudaResident) checkCap(pos, n int) error {
	if pos < 0 || pos+n > r.ctxCap {
		return fmt.Errorf("cuda: KV position %d(+%d) exceeds resident context cap %d — raise it with the serve context setting (bounded by the model's own context window), or use the staged path for longer contexts", pos, n, r.ctxCap)
	}
	return nil
}

// ContextCap is the resident KV capacity in positions (queryable so callers can clamp max_tokens
// up front rather than discover the limit mid-generation).
func (r *cudaResident) ContextCap() int { return r.ctxCap }

// Forward runs one token at absolute position pos and returns logits[vocab].
func (r *cudaResident) Forward(embedding []float32, pos int) ([]float32, error) {
	if e := r.checkCap(pos, 1); e != nil {
		return nil, e
	}
	if pos == 0 {
		// Fresh sequence: re-zero the compounding Gated-DeltaNet state so it does not carry over from a prior Generate on this *Model. No-op for every other family.
		r.Reset()
		if r.resetErr != nil {
			return nil, r.resetErr // a failed re-zero would decode from the previous sequence
		}
	}
	var out []float32
	err := r.do(func() error {
		o, e := r.step(embedding, pos, pos)
		out = o
		return e
	})
	return out, err
}

// ForwardMRoPE is decoder.ResidentMRoPE: like Forward, but the rotation angle (ropePos) and the KV-cache/attention
// position (pos) are supplied separately: Qwen2.5-VL decode past an image block needs them to differ (see
// gemv_fwd.cu's rope_kv doc comment; decoder/rope.go's CPU reference is the ground truth this mirrors). Forward is
// the pos == ropePos case.
func (r *cudaResident) ForwardMRoPE(embedding []float32, pos, ropePos int) ([]float32, error) {
	if e := r.checkCap(pos, 1); e != nil {
		return nil, e
	}
	if pos == 0 {
		r.Reset() // fresh sequence — same reason as Forward
		if r.resetErr != nil {
			return nil, r.resetErr
		}
	}
	var out []float32
	err := r.do(func() error {
		o, e := r.step(embedding, pos, ropePos)
		out = o
		return e
	})
	return out, err
}

// ForwardNoLogits (ResidentPrefillKV) runs the token's forward to build ONLY its resident K/V —
// skipping the final norm + LM head matmul + logits readback + softcap. Used for prompt[:-1] during
// prefill; the layer chain (hence the K/V written at pos) is identical to Forward, so decode from the
// last prompt token is byte-identical. No readback, so nothing is returned but the error.
func (r *cudaResident) ForwardNoLogits(embedding []float32, pos int) error {
	if e := r.checkCap(pos, 1); e != nil {
		return e
	}
	if pos == 0 {
		r.Reset() // prefill from 0 is also a fresh sequence — same reason as Forward
		if r.resetErr != nil {
			return r.resetErr
		}
	}
	return r.do(func() error {
		if e := r.launchToken(embedding, pos, pos, false); e != nil {
			return e
		}
		return r.stream.Sync() // step()'s trailing sync is skipped here; drain so the KV write completes
	})
}

// ForwardN runs K tokens at consecutive positions. Correctness-first: sequential steps in a
// single executor round-trip (bit-identical to K Forward calls; amortizes the channel hop).
func (r *cudaResident) ForwardN(embeddings [][]float32, startPos int) ([][]float32, error) {
	if len(embeddings) == 0 {
		return nil, nil // no-op on empty, matching the cpu/webgpu ForwardN contract: prefillReady would
		// otherwise route into prefillCore and return a spurious "empty prompt" error.
	}
	if e := r.checkCap(startPos, len(embeddings)); e != nil {
		return nil, e
	}
	// Batched verify: the whole [cur, draft...] run in ONE weight-stationary pass (prefillCore, allLogits=true)
	// instead of len(embeddings) sequential decode steps: each weight is read once for all M positions, which is what
	// makes speculative decode a win on the resident CUDA path. Bit-identical to the sequential step loop below (every
	// batched kernel is the M=1 kernel with an M dimension + explicit FMA, gated by TestSpecDecodeCurve's
	// lossless-vs-sequential check). Falls back to the per-token loop for archs the batched path does not cover:
	// errPrefillDeclined only, a real compute error propagates. Spec verify runs M<=9, so the batched allocation is
	// tiny. A recurrent (Gated-DeltaNet) family always takes the loop: the weight-stationary pass has no notion of
	// recurrent state, which must advance one token at a time and in order, and the loop is the same r.step this
	// family's decode uses, so prefill and decode see the same state evolution.
	if r.prefillReady && r.dnet == nil {
		// context.Background(): ForwardN is the spec-decode verify, M<=9 rows, and its own
		// interface carries no context. Nothing here is long enough to want cancelling.
		if outs, _, err := r.prefillCore(context.Background(), embeddings, startPos, tailAllLogits, nil, nil, nil); err == nil {
			return outs, nil
		} else if !errors.Is(err, errPrefillDeclined) {
			return nil, err
		}
	}
	if startPos == 0 {
		r.Reset() // fresh sequence — the recurrent state compounds and is not positional
		if r.resetErr != nil {
			return nil, r.resetErr
		}
	}
	out := make([][]float32, len(embeddings))
	err := r.do(func() error {
		for i, emb := range embeddings {
			l, e := r.step(emb, startPos+i, startPos+i)
			if e != nil {
				return e
			}
			out[i] = append([]float32(nil), l...) // each row kept, so copy off the reused host buf
		}
		return nil
	})
	return out, err
}

// VerifyPath (decoder.VerifyPathReporter) reports whether this resident's ForwardN executes
// a batched weight-stationary pass or falls back to a sequential per-row loop.
func (r *cudaResident) VerifyPath() (bool, string) {
	if !r.prefillReady || r.dnet != nil {
		return false, "sequential — arch/geometry not supported by batched verify"
	}
	return true, "batched weight-stationary CUDA pass"
}

// UploadKV writes a layer's post-RoPE K and raw V into the resident caches at absolute positions
// base..base+n-1 (prefill bridge, same packed layout the kernels read: [pos*kvDim + head*hd + d]). base is
// normally 0; a wrapped CPU-side sliding-window ring's live K/V can start later (KVCache.LayerKV's base return),
// and Buffer.At gives a zero-copy view at the matching byte offset.
func (r *cudaResident) UploadKV(layer, base int, keys, vals []float32) error {
	if r.layers[layer].kvShared {
		// Its r.kc/r.vc ARE its source's buffers: an upload here would overwrite the source layer's cache with another layer's rows.
		return fmt.Errorf("cuda: UploadKV to layer %d, which shares layer %d's KV cache (Gemma 4 E-model): upload the owning layer", layer, r.layers[layer].kvSrc)
	}
	kvDim := r.layers[layer].kvDim
	if kvDim > 0 {
		if e := r.checkCap(base, len(keys)/kvDim); e != nil {
			return e
		}
	}
	byteOff := base * kvDim * 4 // f32 elements, matches gpu.Upload[float32]'s element width
	if r.layers[layer].kvWin {
		// a windowed layer keeps only the last window-1 positions, at physical slot 0 (kvwindow.go)
		keys, vals = r.uploadKVWindowed(layer, base, keys, vals)
		byteOff = 0
	}
	return r.do(func() error {
		if e := gpu.Upload(r.kc[layer].At(byteOff), keys); e != nil {
			return e
		}
		return gpu.Upload(r.vc[layer].At(byteOff), vals)
	})
}

// TruncateTo is a no-op: KV is positional and Forward sets nKeys=pos+1, so entries past pos
// are never read and get overwritten (matches the WebGPU path).
func (r *cudaResident) TruncateTo(pos int) {}

// KVSlots / UseKVSlot implement decoder.ResidentKVSlotter (MC1, docs/tasks/task-concurrency-2026-09.md): how many
// resident KV slots the build allocated, and binding one.
//
// Binding is a pointer swap of kc/vc, done on the executor so it is ordered after every job already posted. That
// is the whole switch because nothing holds a slot's buffers across calls: every launch that reads or writes KV
// (rope_kv, the attention kernels, the batched prefill, the MLA latent store, UploadKV) binds r.kc[l] / r.vc[l]
// when it is issued, and the CUDA graphs (GOINFER_CUDA_GRAPHS) capture only segA/B/C, which touch no KV: rope_kv
// and attention run live in the gap between them. A recurrent family's state is not part of a slot, which is why
// it is allocated one (kvSlotsReq).
func (r *cudaResident) KVSlots() int { return max(1, len(r.kvSlotBufs)) }

func (r *cudaResident) UseKVSlot(i int) error {
	if i < 0 || i >= r.KVSlots() {
		return fmt.Errorf("cuda: KV slot %d out of range (%d allocated)", i, r.KVSlots())
	}
	if len(r.kvSlotBufs) == 0 || i == r.kvSlot {
		return nil
	}
	return r.do(func() error {
		b := r.kvSlotBufs[i]
		r.kc, r.vc, r.kvSlot = b.kc, b.vc, i
		if r.kvWin {
			r.kvBase = r.kvBases[i]
		}
		return nil
	})
}

// Reset re-zeroes the compounding recurrent state for a fresh generation; it is a no-op for every other family,
// because a KV cache needs no reset (the next sequence overwrites it positionally). A Gated-DeltaNet layer's
// {conv ring, matrix state} is NOT positional: it accumulates, and without this the next Generate on the same
// *Model decodes from the previous one's state. The state buffers are allocated zeroed, so the FIRST generation is
// correct either way, which is exactly what makes the omission invisible to a single-sequence test.
func (r *cudaResident) Reset() {
	if r.dnet == nil {
		return
	}
	// Reset satisfies the cross-backend ResidentForward interface and returns nothing, so a failed re-zero cannot be
	// returned; it would be the same silent failure (the next sequence decodes from the PREVIOUS one's recurrent
	// state, invisible to a single-sequence test). Recorded in resetErr and surfaced by the next Forward/ForwardN, the
	// same shape gpu/residency.go uses.
	r.resetErr = r.do(func() error { return r.resetState() })
}

// resetState is Reset's body WITHOUT the executor hop, for callers already running on the executor
// thread. graphsSelfTest is one: it runs inside r.do, so calling Reset there posts to reqCh from
// the goroutine that services reqCh and deadlocks — a hang, not an error, and one that only appears
// once graphs are actually admitted for a recurrent model.
func (r *cudaResident) resetState() error {
	if r.dnet == nil {
		return nil
	}
	dp := r.dnet
	winZ := make([]float32, (dp.convK-1)*dp.convDim)
	stZ := make([]float32, dp.stateElems)
	for i := range r.layers {
		L := &r.layers[i]
		if !L.isDeltaNet {
			continue
		}
		if e := gpu.Upload(L.dnWin, winZ); e != nil {
			return e
		}
		if e := gpu.Upload(L.dnState, stZ); e != nil {
			return e
		}
	}
	return nil
}

// Close releases the model's GPU memory and tears down the pinned executor (unpinning its OS thread).
//
// Freeing the DEVICE memory is the whole job: a resident model owns the weight buffers and the per-layer KV
// cache, gigabytes for a real checkpoint. A Close that only closed the channel leaks the entire model until the
// process exits: invisible in a one-model run, fatal for a model zoo, an /admin/models/unload, or a test binary
// loading models in sequence (it saturates an 8 GB card, after which every Alloc silently returns nil and the
// zero-filled buffers look like a parity bug rather than an OOM).
//
// Every buffer is freed EXPLICITLY rather than by leaning on context destruction. Releasing our primary-context
// reference only reclaims memory if the refcount reaches ZERO, and dev.Primary() hands out a refcounted
// per-device singleton, so any other holder (a second model in a zoo, another subsystem, a test's own probe
// context) keeps the context alive and the "freed" model's VRAM never comes back. TestResidentCloseFreesVRAM
// pins it.
//
// All of it runs ON the executor thread (that thread made the context current) and therefore before reqCh
// closes. Page-locked host memory goes first: it must be freed before the context.
//
// Returns error to satisfy io.Closer (see the assertion in backend.go); the native releases are best-effort and
// cannot meaningfully fail, so it always returns nil.
func (r *cudaResident) Close() error {
	if r.reqCh == nil {
		return nil
	}
	r.reqCh <- func() error {
		// Page-locked host memory must be freed before the context (its free reaches the
		// context's executor), so it goes first and out of the device ledger.
		if r.logitsPinned != nil {
			_ = r.logitsPinned.Close()
			r.logitsPinned, r.logitsHost = nil, nil
		}
		// Host-mapped expert stacks are page-locked host memory too (not in the device ledger), so free them here alongside logitsPinned, before ReleaseObjects.
		for i := range r.layers {
			for _, m := range []*gpu.MappedHostBuffer{
				r.layers[i].expGU.srcW, r.layers[i].expGU.srcS, r.layers[i].expDown.srcW, r.layers[i].expDown.srcS} {
				if m != nil {
					_ = m.Close()
				}
			}
			// CUDA graphs own a driver graphExec each; destroy them before the context teardown.
			for _, g := range []*gpu.Graph{r.layers[i].gSegA, r.layers[i].gSegB, r.layers[i].gSegC} {
				if g != nil {
					_ = g.Close()
				}
			}
		}
		// The Device OWNS every device allocation (weights, KV caches, scratch) in its ledger; ReleaseObjects frees all of
		// them in the correct order, then the modules + stream, then releases our primary-context ref.
		if r.dev != nil {
			r.dev.ReleaseObjects()
			r.dev = nil
		}
		r.layers = nil
		r.kc, r.vc = nil, nil
		return nil
	}
	<-r.ackCh
	close(r.reqCh)
	r.reqCh = nil
	return nil
}

// --- launch helpers (executor-thread only) ---

// copyF32 copies n float32 from src to dst on the decode stream: a launch of kv_store at pos 0 (cache[i] = src[i]), so it is an ordinary kernel launch in the same stream order as the projections
// around it and, unlike gpu.CopyDevice (which synchronizes the context), it records into a CUDA graph. Exact: no arithmetic. The two buffers must not overlap.
func (r *cudaResident) copyF32(src, dst Buffer, n int) error {
	return r.launch(r.kvCopy, g1cfg(n, 256), Arg(src), Arg(dst), gpu.ArgValue(int32(0)), gpu.ArgValue(int32(n)))
}

func g1cfg(n, b int) LaunchConfig {
	return LaunchConfig{GridX: uint32((n + b - 1) / b), GridY: 1, GridZ: 1, BlockX: uint32(b), BlockY: 1, BlockZ: 1}
}
func onecfg(b, sh int) LaunchConfig {
	return LaunchConfig{GridX: 1, GridY: 1, GridZ: 1, BlockX: uint32(b), BlockY: 1, BlockZ: 1, SharedMemBytes: uint32(sh)}
}

// glueQuantThreads is glu_quant's block size. The kernel is ONE block (its int8 scale needs the max over the whole
// intermediate vector), so its time is the serial per-thread loop over I elements, and 1024 threads divide it. Its
// only reduction is a MAX (exact, order-independent) and every element's value and packing are per-element, so the
// block size changes nothing but how the work is divided: the output is bit-identical (checked on real logits,
// docs/measurements/d7-decode-breakdown-2026-09-21.md).
const glueQuantThreads = 1024

func (r *cudaResident) launch(f Pipeline, cfg LaunchConfig, args ...KernelArg) error {
	if r.dbgFreeBeforeLaunch == 0 { // free VRAM at the FIRST launch, recording only
		if f0, _, e0 := r.dev.Context().MemInfo(); e0 == nil {
			r.dbgFreeBeforeLaunch = f0
		}
	}
	// Recording only: free VRAM immediately BEFORE the launch, the other side of the event from describeLaunchErr's
	// reading. That one is reached only after Launch returns non-nil, so it cannot distinguish "memory was released
	// before this launch was attempted" from "the failed attempt released it while unwinding". Recorded at every
	// launch it also yields the decrement from free-at-first-launch to free-at-the-failing-launch from one run.
	if r.dbgProbe {
		if f0, _, e0 := r.dev.Context().MemInfo(); e0 == nil {
			r.dbgFreePreLaunch = f0
			if len(r.dbgLaunchTrace) < 512 {
				r.dbgLaunchTrace = append(r.dbgLaunchTrace,
					fmt.Sprintf("%3d %-14s free=%d", len(r.dbgLaunchTrace), r.pipeName(f), f0))
			}
		}
	}
	r.launchN++ // per-token dispatch count (diagnostic: graph-capturable-fraction bound)
	e := r.stream.Launch(f, cfg, args...)
	if e != nil {
		e = r.describeLaunchErr(f, e)
	}
	if e != nil && r.launchErr == nil {
		// Sticky: launchToken's dense hot chain discards many launch errors (`_ = r.launch(...)`), so a config error (bad
		// shared-mem size, bad args) would let the token "succeed" with stale buffers. Record the first here;
		// launchToken returns it. doG/rms funnel through launch too, so this covers the whole chain without touching every
		// call site.
		r.launchErr = e
	}
	return e
}

// pipeName recovers the struct-field name a Pipeline was bound to, by scanning cudaResident's
// fields for one equal to it. Pipeline is a single-pointer struct, so it compares by identity.
//
// aikit's Pipeline carries no name (it is `struct{ f *gc.Function }`) and lives in another module,
// so the name cannot come from the value itself — but goinfer binds every pipeline to a named
// field, which makes the mapping recoverable here. Error path only; the reflection cost never
// touches a working launch.
func (r *cudaResident) pipeName(f Pipeline) (name string) {
	// TOTAL BY CONSTRUCTION. This runs only when something has already failed, so it must not be
	// able to turn a diagnosable error into a panic — not on a nil receiver, not on an unexported
	// field, not on a pipeline held somewhere other than a named field. The recover is the
	// guarantee; "all 48 sites resolve today" is an observation about today.
	name = "?"
	defer func() {
		if recover() != nil {
			name = "?"
		}
	}()
	if r == nil {
		return "?"
	}
	v := reflect.ValueOf(r).Elem()
	t := v.Type()
	pt := reflect.TypeFor[Pipeline]()
	for i := 0; i < t.NumField(); i++ {
		if t.Field(i).Type != pt {
			continue
		}
		fv := reflect.NewAt(pt, unsafe.Pointer(v.Field(i).UnsafeAddr())).Elem().Interface().(Pipeline)
		if fv == f {
			return t.Field(i).Name
		}
	}
	return "?"
}

// describeLaunchErr turns a bare driver status into something that names what ran out and what the operator can do
// about it. A launch that runs out of memory with the expert cache on otherwise reads as exactly
// `cuLaunchKernel: CUDA_ERROR_OUT_OF_MEMORY`: the API call, not the kernel, with no connection to the setting that
// caused it, and capSlots' decline floor (below topK) does not cover it
// (docs/code-notes/cuda.md#describeLaunchErr). Message content only: the error is wrapped with %w, so type and
// classification are unchanged.
func (r *cudaResident) describeLaunchErr(f Pipeline, e error) error {
	name := r.pipeName(f)
	// Reframe an out-of-memory launch against the expert cache when it is on: the slot count is by
	// far the largest operator-controlled VRAM consumer on this path. Deliberately does NOT suggest
	// a specific safe value — the computation that would produce one (capSlots) is the thing under
	// suspicion, and printing a number from it would launder a suspect figure into advice.
	if r.cacheExperts && strings.Contains(e.Error(), "OUT_OF_MEMORY") {
		// Name the REQUESTED and EFFECTIVE counts both. They differ when the cap fires, and naming
		// only the effective one sends a user who set 48 to lower it to 40 — which caps to the same
		// value, fails identically, and makes the advice look wrong.
		slots := fmt.Sprintf("%d slots/layer", r.cacheSlots)
		if r.cacheSlotsReq != r.cacheSlots {
			slots = fmt.Sprintf("%d slots/layer requested, capped to %d", r.cacheSlotsReq, r.cacheSlots)
		}
		// Deliberately NO free-VRAM reading here. This site is reached only after Launch has returned non-nil, so any
		// figure taken here is a POST-failure state that a reader cannot tell from a pre-launch one: its meaning depends
		// on where the probe sits, which an error string cannot carry. Free VRAM belongs to instrumentation, which can
		// state its own probe position.
		return fmt.Errorf("launch %s: out of device memory with the expert cache at %s — the slot "+
			"count is the likely cause; lower GOINFER_MOE_CACHE_SLOTS below the effective value "+
			"above and retry: %w", name, slots, e)
	}
	return fmt.Errorf("launch %s: %w", name, e)
}

// capVec copies the first n elements of a device vector to host into dst[l] (diagnostic sublayer capture: n is
// hidden for the o-proj/down contributions, qDim for the pre-o-proj context). Runs on the executor thread inside
// launchToken, so it syncs before the readback. It cannot return an error (its callers are capture hooks inside
// launchToken, with no error channel), so it records into setupErr, the field the build path uses, and leaves
// dst[l] nil rather than zero-filled: this feeds HiddenCapture(), and a zero row is a plausible-looking vector a
// speculative drafter would fuse from, while a nil row is something the caller can notice.
func (r *cudaResident) capVec(src Buffer, dst [][]float32, l, n int) {
	if e := r.stream.Sync(); e != nil {
		r.recordUpload(fmt.Errorf("cuda capVec: sync: %w", e))
		return
	}
	h := make([]float32, n)
	if e := gpu.Download(src, h); e != nil {
		r.recordUpload(fmt.Errorf("cuda capVec: download: %w", e))
		return
	}
	dst[l] = h
}

// addOneArg is the (1+w) RMS selector as the kernels take it (Architecture.RMSAddOne).
// Gemma stores norm weights as deviations from 1.0; Llama/Qwen scale by w directly.
func (r *cudaResident) addOneArg() int32 {
	if r.rmsAddOne {
		return 1
	}
	return 0
}

func (r *cudaResident) rmsDim(src, nrm Buffer, dim int, qOut Buffer, sOut Buffer) error {
	return r.launch(r.fRms, onecfg(256, (dim+256)*4),
		Arg(src), Arg(nrm), gpu.ArgValue(int32(dim)), gpu.ArgValue(r.eps),
		gpu.ArgValue(r.addOneArg()), Arg(qOut), Arg(sOut))
}

func (r *cudaResident) rms(src, nrm Buffer, qOut Buffer, sOut Buffer) error {
	return r.rmsDim(src, nrm, r.hidden, qOut, sOut)
}

// layerNormQuant is rms's mean-centered twin (Cohere/Command-R, FeatLayerNorm): no addOne
// selector (bias-free, no Gemma-style (1+w) family uses this norm kind on this backend).
func (r *cudaResident) layerNormQuant(src, nrm Buffer, qOut Buffer, sOut Buffer) error {
	return r.launch(r.fLN, onecfg(256, (r.hidden+256)*4),
		Arg(src), Arg(nrm), gpu.ArgValue(int32(r.hidden)), gpu.ArgValue(r.eps), Arg(qOut), Arg(sOut))
}

// norm dispatches the family's pre-GEMV norm+quant into qOut/sOut — layerNormQuant (Cohere's
// bias-free mean-centered LayerNorm) when r.layerNorm, else the default rms — mirroring Metal's
// encodeNorm/decoder/model.go's normalize() dispatch on arch.Norm.
func (r *cudaResident) norm(src, nrm Buffer, qOut Buffer, sOut Buffer) error {
	if r.layerNorm {
		return r.layerNormQuant(src, nrm, qOut, sOut)
	}
	return r.rms(src, nrm, qOut, sOut)
}

// splitKVAttnDecode runs the high-occupancy, bit-identical decode attention for layer l at position pos (M=1):
// three launches replacing the single attn_batched(M=1). scores tile over keys (nH·⌈nWin/128⌉ blocks), softmax
// keeps the exact 128-wide partition+tree (byte-identical max + denominator), vsum tiles over output dims
// (nH·⌈hd/32⌉ blocks, each thread the whole per-d fold). Writes r.cctx exactly as attn_batched would. See
// docs/tasks/task-decode-splitkv-attention.md.
func (r *cudaResident) splitKVAttnDecode(l, pos int) error {
	Ly := &r.layers[l]
	nKeys := pos + 1
	winStart := 0
	if Ly.window > 0 && nKeys > int(Ly.window) {
		winStart = nKeys - int(Ly.window)
	}
	nWin := nKeys - winStart
	const dTile = 32 // 1 warp/block; keeps the coalesced V-read, maximizes blocks without sub-warp waste
	// 1. scores → r.skScoreBuf[h*nWin + i] (raw, ·scale). One thread per key; no reduction. SharedMemBytes: hd floats
	// for the staged q: splitkv_scores stages the block-invariant q vector once instead of every thread re-loading it
	// from global (docs/measurements/splitkv-stall-profile-2026-09-13.md).
	if e := r.launch(r.skScores, LaunchConfig{GridX: uint32(r.nH), GridY: uint32((nWin + 127) / 128), GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32(Ly.hd * 4)},
		Arg(r.qB), Arg(r.kvK(l)), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(Ly.nKV)), gpu.ArgValue(int32(Ly.hd)),
		gpu.ArgValue(int32(winStart)), gpu.ArgValue(int32(nKeys)), gpu.ArgValue(r.attnScale), Arg(r.skScoreBuf), gpu.ArgValue(int32(nWin))); e != nil {
		return e
	}
	// 2. softmax in place (block 128 — MUST match attn_batched for byte-identical max/denominator).
	if e := r.launch(r.skSoftmax, LaunchConfig{GridX: uint32(r.nH), GridY: 1, GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: 128 * 4},
		// gpt-oss is the only family with an attention sink, and it is resident-eligible, so r.sinkArg(l) below returns
		// the real per-layer sink for it and ArgNull() for every other family: not an always-null argument.
		Arg(r.skScoreBuf), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(nWin)), Arg(r.skInvBuf), r.sinkArg(l)); e != nil {
		return e
	}
	// 3. V-sum → r.cctx (each thread the whole ascending-s fold for one output dim). The branch below is the opt-in
	// SPIKE path: it splits the V fold over S key chunks and combines in fixed order, so it is NOT bit-identical to
	// attn_batched and is unreachable unless GOINFER_SPLITKV_VSUM_SPLIT is set.
	if r.skVsumSplit >= 1 && r.skVsumPartial != (Pipeline{}) && r.skVsumCombine != (Pipeline{}) {
		nSplit := r.skVsumSplit
		dy := uint32((Ly.hd + dTile - 1) / dTile)
		if e := r.launch(r.skVsumPartial, LaunchConfig{GridX: uint32(r.nH), GridY: dy, GridZ: uint32(nSplit), BlockX: dTile, BlockY: 1, BlockZ: 1},
			Arg(r.skScoreBuf), Arg(r.kvV(l)), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(Ly.nKV)), gpu.ArgValue(int32(Ly.hd)),
			gpu.ArgValue(int32(winStart)), gpu.ArgValue(int32(nKeys)), gpu.ArgValue(int32(nWin)),
			gpu.ArgValue(int32(nSplit)), Arg(r.skPartialBuf)); e != nil {
			return e
		}
		return r.launch(r.skVsumCombine, LaunchConfig{GridX: uint32(r.nH), GridY: dy, GridZ: 1, BlockX: dTile, BlockY: 1, BlockZ: 1},
			Arg(r.skPartialBuf), Arg(r.skInvBuf), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(Ly.hd)),
			gpu.ArgValue(int32(nSplit)), Arg(r.cctx))
	}
	return r.launch(r.skVsum, LaunchConfig{GridX: uint32(r.nH), GridY: uint32((Ly.hd + dTile - 1) / dTile), GridZ: 1, BlockX: dTile, BlockY: 1, BlockZ: 1},
		Arg(r.skScoreBuf), Arg(r.kvV(l)), Arg(r.skInvBuf), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(Ly.nKV)), gpu.ArgValue(int32(Ly.hd)),
		gpu.ArgValue(int32(winStart)), gpu.ArgValue(int32(nKeys)), gpu.ArgValue(int32(nWin)), Arg(r.cctx))
}

// normF32 is Gemma's sandwich post-norm: a plain in-place RMSNorm of a SUBLAYER OUTPUT
// (no quant — it lands straight in the f32 residual stream). No-op when the arch has no
// sandwich norms, so non-Gemma families pay nothing.
func (r *cudaResident) normF32(x, w Buffer) error {
	if w.Len() == 0 {
		return nil
	}
	return r.launch(r.fRmsF32, onecfg(256, 256*4),
		Arg(x), Arg(w), gpu.ArgValue(int32(r.hidden)), gpu.ArgValue(r.eps), gpu.ArgValue(r.addOneArg()))
}

// doG launches the projection GEMV. accum=1 makes the epilogue do dst[n] += result, which
// absorbs the separate `residual` launch (bit-identical: same operands, same rounding, just
// no round-trip through a temp buffer). Only lane 0 of the row's warp touches dst[n], and the
// GEMV's input activation is never x, so accumulating straight into the residual stream is
// race-free.
func (r *cudaResident) doG(wt cudaWQ, a Buffer, as Buffer, bias KernelArg, dst Buffer, accum int32) error {
	cfg := LaunchConfig{GridX: uint32((wt.N + 7) / 8), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1}
	if wt.kind == "q4k" {
		// BuildResident admits q4k only with actG32 (the per-group sums in as[K/32:] come from quantG32).
		return r.launch(r.gemvQ4K, cfg, Arg(wt.W), Arg(a), Arg(as), bias,
			gpu.ArgValue(int32(wt.N)), gpu.ArgValue(int32(wt.K/256)), Arg(dst), gpu.ArgValue(accum))
	}
	if wt.kind == "int4" {
		return r.launch(r.gemvW4, cfg, Arg(wt.W), Arg(a), Arg(wt.ws16), Arg(as), bias,
			gpu.ArgValue(int32(wt.N)), gpu.ArgValue(int32(wt.K/8)), gpu.ArgValue(int32(wt.K/32)), Arg(dst), gpu.ArgValue(accum))
	}
	return r.launch(r.gemvW8, cfg, Arg(wt.W), Arg(a), Arg(wt.ws), Arg(as), bias,
		gpu.ArgValue(int32(wt.N)), gpu.ArgValue(int32(wt.K/4)), Arg(dst), gpu.ArgValue(accum))
}

// moeMLPPre issues the pre-readback half of one MoE FFN block (decoder/mlp.go's moeMLP is the CPU reference): the
// shared normed activation (mq/mSc) and the router (logits → top-k idx/wgt, left on the device). It ends exactly
// where the cacheExperts path must read rIdx back to the host, so this half is graph-static (segB) and the
// loadRoutedExperts D2H stays live in the gap between segB and segC. moeMLPPost issues the rest. The SAME normed
// activation feeds router AND experts; the router logits are f32 (see cudaResident.moe).
func (r *cudaResident) moeMLPPre(Ly *cudaLayer, x Buffer) error {
	// Explicit rmsnorm, NOT the fused fGU path: the fused kernel folds the norm into the dense
	// gate/up GEMV and never writes r.mq, but the router needs that quantized activation too.
	if e := r.rms(x, Ly.postNorm, r.mq, r.mSc); e != nil {
		return e
	}
	// Router logits (one block per expert row) → top-k idx/wgt, both left on the device. The
	// selection never round-trips to the host: a D2H here would serialize the whole token.
	if e := r.launch(r.fRouterGemv, LaunchConfig{GridX: uint32(r.nE), GridY: 1, GridZ: 1,
		BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4},
		Arg(Ly.routerW), Arg(r.mq), Arg(r.mSc), gpu.ArgValue(int32(r.nE)),
		gpu.ArgValue(int32(r.hidden)), Arg(r.rLogits)); e != nil {
		return e
	}
	// gpt-oss routes through its OWN kernel: moe_route takes the mixing weight from the UNBIASED score (bias steers
	// selection only), while gpt-oss softmaxes over the SELECTED BIASED logits. Same selection, different weights,
	// which is why the wrong one produces plausible output rather than an error.
	if r.gptOssRoute != (Pipeline{}) {
		if e := r.launch(r.gptOssRoute, onecfg(1, 0),
			Arg(r.rLogits), Arg(Ly.routerB), Arg(r.rIdx), Arg(r.rWgt),
			gpu.ArgValue(int32(r.nE)), gpu.ArgValue(int32(r.topK))); e != nil {
			return e
		}
		return r.recordRouted()
	}
	// Argument ORDER is easy to transpose silently here: group-routed families (DeepSeek/Kimi; testdata/deepseek-tiny sets
	// n_group=2, topk_group=1) pass nGroup then topkGroup. TestMoERoute drives the kernel with its own argument list, so it
	// validates the kernel's math, not this call's order, and a transposition once passed TestMLAResidentParityCUDA while
	// it compared only the first generated token (a discrete-selection bug need not show in the first tokens); it now
	// compares the full sequence (cuda/mla_resident_test.go). Typed launch wrappers make a transposition a COMPILE error but
	// cannot check that a value is the right way round; see the reciprocal trap at decoder/features.go's cuda entry and
	// docs/parity-coverage-policy.md § "Relevant".
	if e := r.launch(r.fRoute, onecfg(1, 0),
		Arg(r.rLogits), Arg(Ly.routerB), Arg(r.rIdx), Arg(r.rWgt),
		gpu.ArgValue(int32(r.nE)), gpu.ArgValue(int32(r.topK)), gpu.ArgValue(r.moeSigmoid),
		gpu.ArgValue(r.moeNormTopK), gpu.ArgValue(r.moeScale),
		gpu.ArgValue(int32(r.nGroup)), gpu.ArgValue(int32(r.topkGroup))); e != nil {
		return e
	}
	return r.recordRouted()
}

// launchGluSplit runs the fused gate||up SwiGLU (glu_quant): dscratch[k]=act(gu[k])*gu[inter+k], then
// symmetric-int8-quantizes it into (outQ,outSc). The stacked-expert GEMV lays gate and up CONTIGUOUS in one
// buffer, so this passes the same pointer twice with gOff=0, uOff=inter: the ONE place that gate/up split
// convention lives, shared by every MoE call site (routed, shared, gemma-4). Keep it the sole gate/up-split
// dispatch: a gOff/uOff swap silently computes silu(up)*gate, which the e2e MoE parity gate CANNOT catch on
// random-weight experts (the two are near-equal in magnitude), so TestMoeSwigluWiring exercises this exact helper
// with crafted gate != up and asserts gate-first.
func (r *cudaResident) launchGluSplit(gu Buffer, inter int, outQ, outSc, outScr Buffer) error {
	return r.launchGluSplitExpert(gu, inter, outQ, outSc, outScr, Buffer{}, -1)
}

// expBiasArg returns this layer's per-expert gate‖up bias table, or the zero Buffer when the
// family has none. Keyed off cudaLayer.idx because moeMLPPost works from *cudaLayer; threading
// a separate index alongside it would be one more thing that can fall out of step with the
// weights the bias must match.
func (r *cudaResident) expBiasArg(Ly *cudaLayer) Buffer {
	if l := Ly.idx; l >= 0 && l < len(r.gptOssExpBias) {
		return r.gptOssExpBias[l]
	}
	return Buffer{}
}

// expDownBiasArg returns layer Ly's per-expert down-projection bias table, or the zero Buffer
// when the family has none. A per-layer SIDE TABLE like the sinks and the gate‖up biases, not a
// cudaLayer field: these are uploaded in the gpt-oss block that runs BEFORE r.layers is
// allocated, so a cudaLayer field there writes into a nil slice.
func (r *cudaResident) expDownBiasArg(Ly *cudaLayer) Buffer {
	if l := Ly.idx; l >= 0 && l < len(r.gptOssDownBias) {
		return r.gptOssDownBias[l]
	}
	return Buffer{}
}

// sinkArg returns layer l's attention-sink argument, or null when the family has none.
// Centralized so the DECODE and PREFILL attention launches cannot disagree — a sink applied
// on one path and not the other presents as drift partway through a sequence, which is much
// harder to attribute than a missing term everywhere.
func (r *cudaResident) sinkArg(l int) gpu.KernelArg {
	if l >= 0 && l < len(r.gptOssSinks) && r.gptOssSinks[l] != (Buffer{}) {
		return Arg(r.gptOssSinks[l])
	}
	return ArgNull()
}

// oBiasArg returns layer Ly's attention output-projection bias, or null when the family has none. Centralized for
// the reason sinkArg is: the DECODE and the PREFILL o_proj launches must not disagree, and a bias applied on one
// path only reads as drift partway through a sequence, once decode takes over from prefill. It needs no dedicated
// kernel: both GEMV pairs fold the bias into the value BEFORE the accumulate select (aikit gemv_quant.cu:
// `val = fma(facc, aScale, bias?bias[n]:0); dst[n] = accum ? dst[n]+val : val`, and goinfer's batched
// gemv_w4a8_rn.cu identically), so bias-plus-residual is one instruction on this backend.
func (r *cudaResident) oBiasArg(Ly *cudaLayer) gpu.KernelArg {
	if Ly != nil && Ly.hasOBias {
		return Arg(Ly.ob)
	}
	return ArgNull()
}

// launchGluSplitExpert is launchGluSplit with the routed-expert context the gpt-oss epilogue
// needs. bias is that layer's [nExpert·2·inter] gate‖up table and slot is the top-k position
// whose expert is running; the KERNEL does biasRow = idx[slot]*2*inter, because which expert
// runs is a device-side routing decision and the launch geometry must not depend on it.
//
// Families other than gpt-oss route through the nil/-1 path and land on glu_quant exactly as
// before — same kernel, same arguments, bit-identical. gpt-oss is the only family whose
// activation is not act(g)*u, so branching here rather than adding a mode to glu_quant keeps
// the clamped variant out of the audited glue.ptx.
func (r *cudaResident) launchGluSplitExpert(gu Buffer, inter int, outQ, outSc, outScr Buffer, bias Buffer, slot int) error {
	// Buffer and Pipeline are STRUCTS here, not interfaces, so "absent" is the zero value
	// rather than nil — the same test resident.go already uses for the optional split-KV
	// pipelines (r.skScores != (Pipeline{})).
	if r.gptOssSw != (Pipeline{}) {
		// expertBiasIdx, NOT expIdx: this kernel uses idx ONLY to pick a row of the per-expert
		// bias table (it consumes already-computed gate/up activations and runs no GEMV), so it
		// needs the expert id even when the weights it followed were read from a cache slot.
		idx := Arg(r.expertBiasIdx())
		bArg := ArgNull()
		if bias != (Buffer{}) {
			bArg = Arg(bias)
		}
		if slot < 0 {
			slot = 0 // the shared/dense caller has no routing slot; gpt-oss has no shared expert
		}
		return r.launch(r.gptOssSw, onecfg(256, 256*4),
			Arg(gu), Arg(gu), gpu.ArgValue(int32(0)), gpu.ArgValue(int32(inter)),
			gpu.ArgValue(int32(inter)), bArg, idx, gpu.ArgValue(int32(slot)),
			gpu.ArgValue(r.gptOssAlpha), gpu.ArgValue(r.gptOssLimit),
			Arg(outQ), Arg(outSc), Arg(outScr))
	}
	return r.launch(r.fSw, onecfg(glueQuantThreads, glueQuantThreads*4),
		Arg(gu), Arg(gu), gpu.ArgValue(int32(0)), gpu.ArgValue(int32(inter)),
		gpu.ArgValue(int32(inter)), gpu.ArgValue(r.act),
		Arg(outQ), Arg(outSc), Arg(outScr))
}

// moeMLPPost issues the post-readback half of one MoE FFN block (mirrors decoder/mlp.go's moeMLP): the sequential
// expert loop, then the always-on shared expert, weight-accumulating into the residual x. Every launch has the
// same geometry whatever the router picked: each expert is selected by ARITHMETIC on r.expIdx() (the weight-row
// index inside the kernel), not by binding a different buffer, which is what keeps the dispatch chain static
// (graph-capturable). The final GEMV weight-accumulates into x, so the per-expert combine and the residual add are
// one instruction (no scratch, no combine pass). The cacheExperts readback (if any) has already filled the slots
// when this runs.
func (r *cudaResident) moeMLPPost(Ly *cudaLayer, x Buffer) error {
	gu := 2 * r.moeInter
	for j := 0; j < r.topK; j++ {
		if e := r.waitMiss(j); e != nil {
			return e
		}
		// gate‖up for the routed expert, in ONE indexed GEMV: the stack interleaves each
		// expert's gate and up rows (packWeightStack(g0,u0,g1,u1,...)), so one row range of
		// width 2*moeInter is exactly this expert's pair.
		if e := r.launch(r.fMoEGemv, LaunchConfig{GridX: uint32((gu + 7) / 8), GridY: 1, GridZ: 1,
			BlockX: 256, BlockY: 1, BlockZ: 1},
			Arg(Ly.expGU.W), Arg(r.mq), Arg(Ly.expGU.ws16), Arg(r.mSc),
			Arg(r.expIdx()), gpu.ArgValue(int32(j)), gpu.ArgValue(int32(gu)),
			gpu.ArgValue(int32(gu)), gpu.ArgValue(int32(r.hidden/8)), gpu.ArgValue(int32(r.hidden/32)),
			Arg(r.moeGU)); e != nil {
			return e
		}
		// SwiGLU over the halves of that one buffer. gocudrv exposes no buffer view/offset, so
		// the split is the kernel's gOff/uOff rather than Go-side pointer arithmetic.
		if e := r.launchGluSplitExpert(r.moeGU, r.moeInter, r.moeQ, r.moeSc, r.moeScr, r.expBiasArg(Ly), j); e != nil {
			return e
		}
		// down-proj, weight-accumulating into the residual: x += wgt[j] * (Down_e · act). gpt-oss adds its per-expert down
		// bias INSIDE that product via its own kernel; every other family keeps the untouched wacc, so their numerics are
		// unchanged.
		if db := r.expDownBiasArg(Ly); db != (Buffer{}) && r.fMoEWaccBias != (Pipeline{}) {
			if e := r.launch(r.fMoEWaccBias, LaunchConfig{GridX: uint32((r.hidden + 7) / 8), GridY: 1, GridZ: 1,
				BlockX: 256, BlockY: 1, BlockZ: 1},
				Arg(Ly.expDown.W), Arg(r.moeQ), Arg(Ly.expDown.ws16), Arg(r.moeSc),
				Arg(r.expIdx()), Arg(r.expertBiasIdx()), Arg(db), Arg(r.rWgt),
				gpu.ArgValue(int32(j)), gpu.ArgValue(int32(r.hidden)),
				gpu.ArgValue(int32(r.hidden)), gpu.ArgValue(int32(r.moeInter/8)), gpu.ArgValue(int32(r.moeInter/32)),
				Arg(x)); e != nil {
				return e
			}
		} else if e := r.launch(r.fMoEWacc, LaunchConfig{GridX: uint32((r.hidden + 7) / 8), GridY: 1, GridZ: 1,
			BlockX: 256, BlockY: 1, BlockZ: 1},
			Arg(Ly.expDown.W), Arg(r.moeQ), Arg(Ly.expDown.ws16), Arg(r.moeSc),
			Arg(r.expIdx()), Arg(r.rWgt), gpu.ArgValue(int32(j)), gpu.ArgValue(int32(r.hidden)),
			gpu.ArgValue(int32(r.hidden)), gpu.ArgValue(int32(r.moeInter/8)), gpu.ArgValue(int32(r.moeInter/32)),
			Arg(x)); e != nil {
			return e
		}
	}
	// Always-on shared expert (GLM/DeepSeek): an ungated SwiGLU MLP over the SAME normed
	// activation, added to the residual. Structurally a routed expert with no routing — a dense
	// gate‖up GEMV, the same glu_quant offset split, a dense down-proj, then the combine adds
	// it in ungated (dst += shDown). decoder/mlp.go does exactly this after the routed sum.
	if Ly.hasShared {
		nullBias := ArgNull()
		if e := r.doG(Ly.shGU, r.mq, r.mSc, nullBias, r.shGUout, 0); e != nil {
			return e
		}
		if e := r.launchGluSplit(r.shGUout, r.sharedInter, r.shQ, r.shSc, r.shScr); e != nil {
			return e
		}
		if e := r.doG(Ly.shDown, r.shQ, r.shSc, nullBias, r.shDownOut, 0); e != nil {
			return e
		}
		// GLM/DeepSeek add the shared output ungated; Qwen-MoE scales it by sigmoid(SharedGate·h)
		// first. Same kernel either way. When ungated the gl pointer is unread, but the kernel
		// still takes it, so pass a valid buffer (shSc, spare) rather than a null.
		gl, ungated := r.shSc, int32(1)
		if Ly.shGateW.W != (Buffer{}) {
			if e := r.doG(Ly.shGateW, r.mq, r.mSc, nullBias, r.shGl, 0); e != nil {
				return e
			}
			gl, ungated = r.shGl, 0
		}
		if e := r.launch(r.fSharedCombine, g1cfg(r.hidden, 256),
			Arg(x), Arg(r.shDownOut), Arg(gl), gpu.ArgValue(int32(r.hidden)),
			gpu.ArgValue(ungated)); e != nil {
			return e
		}
	}
	return nil
}

// gemma4MoeMLPPre/Post issue Gemma-4's parallel dense‖MoE FFN for layer Ly (enable_moe_block). Unlike the generic
// moeMLPPre/Post (one branch, wacc straight into the residual), Gemma-4 runs TWO branches off the SAME residual h
// and joins them under a shared post-norm, so h is read three times and written only at the end. Mirrors
// decoder/forward_gemma4_moe.go exactly:
//
//	x1 = postFFNNorm1( mlpDown( geluTanh(mlpGate·xd)·(mlpUp·xd) ) )      xd = preFFNNorm(h)   [dense]
//	rn = rmsnorm_nw(h);  logits = RouterProjScaled·rn;  idx,wgt = route  wgt *= perExpertScale[idx]
//	x2 = postFFNNorm2( Σ_j wgt[j]·expertDown_j(geluTanh(gu_j)·up_j) )    xe = preFFNNorm2(h)  [MoE]
//	h  = (h + postFFNNorm(x1 + x2)) · layerScalar                                             [join]
//
// The router GEMV is PURE f32 (gemv_f32_f32) with routerScale·hidden^-0.5 folded into RouterProjScaled
// at build; rn is the weightless OUT-OF-PLACE norm (rmsnorm_nw) so h stays intact for the other two
// branches and the residual add.
// gemma4MoeMLPPre issues the pre-readback half of Gemma-4's parallel dense‖MoE FFN: the dense branch
// (→ g4x1), the router (on RAW h → idx/wgt with the per-expert scale folded in), and the expert-branch
// input norm (xe → mq/mSc). It ends before the g4x2 accumulator clear + the cacheExperts readback,
// both of which stay live in the segB→segC gap (an H2D and, optionally, a D2H — neither capturable).
func (r *cudaResident) gemma4MoeMLPPre(Ly *cudaLayer, l int, x Buffer) error {
	nullBias := ArgNull()
	r.profMark(&r.profAttn)

	// --- router FIRST (on RAW h): rmsnorm_nw → gemv_f32_f32(folded proj) → moe_route → per-expert-scale
	// fold. It needs only h, and it is the one thing the host waits for (recordRouted); the dense
	// branch below then executes while the routing is read back and the misses are DMA'd. The two
	// share no buffers, so this order is bit-identical to dense-first. ---
	if e := r.launch(r.fRmsNW, onecfg(256, 256*4), Arg(x), Arg(r.g4rn), gpu.ArgValue(int32(r.hidden)), gpu.ArgValue(r.eps)); e != nil {
		return e
	}
	if e := r.launch(r.fRouterF32, LaunchConfig{GridX: uint32(r.nE), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4},
		Arg(Ly.routerW), Arg(r.g4rn), gpu.ArgValue(int32(r.nE)), gpu.ArgValue(int32(r.hidden)), Arg(r.rLogits)); e != nil {
		return e
	}
	// softmax (sigmoid=0), UNCONDITIONAL renorm (norm=1), scale=1, no group routing.
	if e := r.launch(r.fRoute, onecfg(1, 0), Arg(r.rLogits), Arg(Ly.routerB), Arg(r.rIdx), Arg(r.rWgt),
		gpu.ArgValue(int32(r.nE)), gpu.ArgValue(int32(r.topK)), gpu.ArgValue(int32(0)), gpu.ArgValue(int32(1)),
		gpu.ArgValue(float32(1)), gpu.ArgValue(int32(1)), gpu.ArgValue(int32(1))); e != nil {
		return e
	}
	if e := r.launch(r.fScaleWgt, LaunchConfig{GridX: 1, GridY: 1, GridZ: 1, BlockX: uint32(r.topK), BlockY: 1, BlockZ: 1},
		Arg(r.rWgt), Arg(r.rIdx), Arg(Ly.perExpertScaleB), gpu.ArgValue(int32(r.topK))); e != nil {
		return e
	}
	if e := r.recordRouted(); e != nil {
		return e
	}
	r.profMark(&r.profRouter)

	// --- dense branch → g4x1 ---
	if e := r.rms(x, Ly.g4preFFN, r.mq, r.mSc); e != nil { // xd = preFFNNorm(h), int8
		return e
	}
	if e := r.doG(Ly.g, r.mq, r.mSc, nullBias, r.gO, 0); e != nil {
		return e
	}
	if e := r.doG(Ly.u, r.mq, r.mSc, nullBias, r.uO, 0); e != nil {
		return e
	}
	if e := r.launch(r.fSw, onecfg(glueQuantThreads, glueQuantThreads*4), Arg(r.gO), Arg(r.uO), gpu.ArgValue(int32(0)), gpu.ArgValue(int32(0)),
		gpu.ArgValue(int32(r.inter)), gpu.ArgValue(r.act), Arg(r.dq), Arg(r.dSc), Arg(r.dScr)); e != nil {
		return e
	}
	if e := r.doG(Ly.d, r.dq, r.dSc, nullBias, r.g4x1, 0); e != nil {
		return e
	}
	if e := r.normF32(r.g4x1, Ly.g4postFFN1); e != nil {
		return e
	}
	r.profMark(&r.profDense)

	// xe = preFFNNorm2(h); reuse mq/mSc (dense branch done with them). This is the last static op
	// before the gap: launchToken clears g4x2 (H2D) and, if caching, reads back the routing (D2H).
	if e := r.rms(x, Ly.g4preFFN2, r.mq, r.mSc); e != nil {
		return e
	}
	r.profMark(&r.profRouter)
	return nil
}

// gemma4MoeMLPPost issues the post-readback half: the expert loop accumulating into the (already
// cleared) g4x2, its post-norm, and the join — sum x1+x2, joint post-norm, add the residual h, then
// the per-layer scalar. g4x2 was zeroed and the expert slots filled in the gap before this runs.
func (r *cudaResident) gemma4MoeMLPPost(Ly *cudaLayer, l int, x Buffer) error {
	gu := 2 * r.moeInter
	for j := 0; j < r.topK; j++ {
		if e := r.waitMiss(j); e != nil {
			return e
		}
		if e := r.launch(r.fMoEGemv, LaunchConfig{GridX: uint32((gu + 7) / 8), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1},
			Arg(Ly.expGU.W), Arg(r.mq), Arg(Ly.expGU.ws16), Arg(r.mSc), Arg(r.expIdx()),
			gpu.ArgValue(int32(j)), gpu.ArgValue(int32(gu)), gpu.ArgValue(int32(gu)),
			gpu.ArgValue(int32(r.hidden/8)), gpu.ArgValue(int32(r.hidden/32)), Arg(r.moeGU)); e != nil {
			return e
		}
		if e := r.launchGluSplitExpert(r.moeGU, r.moeInter, r.moeQ, r.moeSc, r.moeScr, r.expBiasArg(Ly), j); e != nil {
			return e
		}
		if e := r.launch(r.fMoEWacc, LaunchConfig{GridX: uint32((r.hidden + 7) / 8), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1},
			Arg(Ly.expDown.W), Arg(r.moeQ), Arg(Ly.expDown.ws16), Arg(r.moeSc), Arg(r.expIdx()), Arg(r.rWgt),
			gpu.ArgValue(int32(j)), gpu.ArgValue(int32(r.hidden)), gpu.ArgValue(int32(r.hidden)),
			gpu.ArgValue(int32(r.moeInter/8)), gpu.ArgValue(int32(r.moeInter/32)), Arg(r.g4x2)); e != nil {
			return e
		}
	}
	if e := r.normF32(r.g4x2, Ly.g4postFFN2); e != nil {
		return e
	}

	// DEBUG capture BEFORE the join — the four buffers that localize a whole-forward miss to router
	// (rn/wgt) vs dense (x1) vs expert (x2) vs join (logits). Off unless GOINFER_G4_CAPTURE.
	if r.g4cap {
		r.capVec(r.g4rn, r.g4capRn, l, r.hidden)
		r.capVec(r.rWgt, r.g4capWgt, l, r.topK)
		r.capVec(r.g4x1, r.g4capX1, l, r.hidden)
		r.capVec(r.g4x2, r.g4capX2, l, r.hidden)
		if err := r.stream.Sync(); err != nil {
			return err
		}
		idx := make([]uint32, r.topK)
		if err := gpu.Download(r.rIdx, idx); err != nil {
			return err
		}
		r.g4capIdx = append(r.g4capIdx, idx) // append order = CPU routerCaptureBuf order (per-position routing check)
		r.g4capLayer = append(r.g4capLayer, l)
	}

	// JOIN: sum x1+x2 BEFORE the joint norm; add the residual h AFTER it; then the per-layer scalar. A mis-order is
	// plausible-but-wrong: cosine shows it but will not localize it.
	if e := r.launch(r.fRes, g1cfg(r.hidden, 256), Arg(r.g4x1), Arg(r.g4x2), gpu.ArgValue(int32(r.hidden))); e != nil { // x1 += x2
		return e
	}
	if e := r.normF32(r.g4x1, Ly.g4postFFN); e != nil { // x1 = postFFNNorm(x1 + x2)
		return e
	}
	if e := r.launch(r.fRes, g1cfg(r.hidden, 256), Arg(x), Arg(r.g4x1), gpu.ArgValue(int32(r.hidden))); e != nil { // x = h + comb
		return e
	}
	return r.launch(r.fScaleVec, g1cfg(r.hidden, 256), Arg(x), gpu.ArgValue(Ly.layerScalar), gpu.ArgValue(int32(r.hidden))) // x *= layerScalar
}

// segA / segB / segC are the three graph-STATIC launch runs of one layer, factored out of launchToken so a SINGLE
// source of truth serves both the live path (call them in sequence) and the graph path (replay Ly.gSegA/B/C).
// Every launch here binds only fixed buffers and per-layer constants; the per-token dynamics (rope_kv/attention
// bind pos/nKeys; the g4x2 clear and the routing readback are host copies) stay live in the gaps between them. See
// cudaLayer.gSegA and captureGraphs.

// segA: QKV projection (fused K1 super-kernel when every projection is int4, else rmsnorm+quant +
// three GEMVs) + per-head QK-norm + scale-less K=V v-norm. Ends before rope_kv.
func (r *cudaResident) segA(Ly *cudaLayer, l int) error {
	nullBias := ArgNull()
	qb, kb, vb := nullBias, nullBias, nullBias
	if Ly.hasBias {
		qb, kb, vb = Arg(Ly.qb), Arg(Ly.kb), Arg(Ly.vb)
	}
	if r.fuseQKV {
		// K1: rmsnorm+quant redundantly per block + this block's Q/K/V rows — one launch instead of four.
		nrows := Ly.qDim + 2*Ly.kvDim
		cfg := LaunchConfig{GridX: uint32((nrows + 7) / 8), GridY: 1, GridZ: 1,
			BlockX: 256, BlockY: 1, BlockZ: 1,
			SharedMemBytes: uint32((r.hidden + 256 + r.hidden/4) * 4)}
		// Rows-per-warp variant (fused_qkv_rows.cu): each warp walks several rows off one shared activation, so fewer blocks pay the redundant
		// rmsnorm+quant prologue. Bit-identical to the one-row-per-warp kernel (TestFusedQKVRowsBitIdentical), so this only ever changes speed.
		rpw := r.waveRowsPerWarp(nrows, int(cfg.SharedMemBytes))
		if rpw == 0 {
			rpw = fusedQKVRowsPerWarp(nrows) // device shape unread: the static rule
		}
		if rpw > 1 && r.fQKVRows != (Pipeline{}) {
			rcfg := cfg
			rcfg.GridX = uint32((nrows + 8*rpw - 1) / (8 * rpw))
			if e := r.launch(r.fQKVRows, rcfg,
				Arg(r.x), Arg(Ly.preNorm), gpu.ArgValue(int32(r.hidden)), gpu.ArgValue(r.eps),
				gpu.ArgValue(r.addOneArg()),
				Arg(Ly.q.W), Arg(Ly.q.ws16), qb,
				Arg(Ly.k.W), Arg(Ly.k.ws16), kb,
				Arg(Ly.v.W), Arg(Ly.v.ws16), vb,
				gpu.ArgValue(int32(Ly.qDim)), gpu.ArgValue(int32(Ly.kvDim)),
				gpu.ArgValue(int32(r.hidden/8)), gpu.ArgValue(int32(r.hidden/32)), gpu.ArgValue(int32(rpw)),
				Arg(r.qB), Arg(r.kB), Arg(r.vB)); e != nil {
				return e
			}
		} else if e := r.launch(r.fQKV, cfg,
			Arg(r.x), Arg(Ly.preNorm), gpu.ArgValue(int32(r.hidden)), gpu.ArgValue(r.eps),
			gpu.ArgValue(r.addOneArg()),
			Arg(Ly.q.W), Arg(Ly.q.ws16), qb,
			Arg(Ly.k.W), Arg(Ly.k.ws16), kb,
			Arg(Ly.v.W), Arg(Ly.v.ws16), vb,
			gpu.ArgValue(int32(Ly.qDim)), gpu.ArgValue(int32(Ly.kvDim)),
			gpu.ArgValue(int32(r.hidden/8)), gpu.ArgValue(int32(r.hidden/32)),
			Arg(r.qB), Arg(r.kB), Arg(r.vB)); e != nil {
			return e
		}
	} else if r.fuseG32 {
		// Per-32 twin of K1: rmsnorm + per-32 quant redundantly per block + Q/K/V rows of any per-32
		// kind — one launch instead of four. Bit-identical to the unfused chain below (shared row code).
		nrows := Ly.qDim + 2*Ly.kvDim
		rpw := fusedQKVRowsPerWarp(nrows)
		cfg := LaunchConfig{GridX: uint32((nrows + 8*rpw - 1) / (8 * rpw)), GridY: 1, GridZ: 1,
			BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: fusedG32Shmem(r.hidden)}
		if e := r.launch(r.fQKVg32, cfg,
			Arg(r.x), Arg(Ly.preNorm), gpu.ArgValue(int32(r.hidden)), gpu.ArgValue(r.eps),
			gpu.ArgValue(r.addOneArg()),
			Arg(Ly.q.W), wkScale(Ly.q), qb, gpu.ArgValue(wkCode(Ly.q.kind)),
			Arg(Ly.k.W), wkScale(Ly.k), kb, gpu.ArgValue(wkCode(Ly.k.kind)),
			Arg(Ly.v.W), wkScale(Ly.v), vb, gpu.ArgValue(wkCode(Ly.v.kind)),
			gpu.ArgValue(int32(Ly.qDim)), gpu.ArgValue(int32(Ly.kvDim)), gpu.ArgValue(int32(rpw)),
			Arg(r.qB), Arg(r.kB), Arg(r.vB)); e != nil {
			return e
		}
	} else {
		if r.postOnly {
			// Olmo 3/Olmo Hybrid: no pre-norm at all — quantize the RAW residual (quant_vec,
			// the same symmetric int8 quantizer ctx-before-o-proj already uses) instead of
			// rmsnorm_quant. fuseQKV is forced off for postOnly (see BuildResident), so this is
			// the only path a postOnly layer's QKV projection takes.
			if e := r.launch(r.fQ, onecfg(256, 256*4), Arg(r.x), gpu.ArgValue(int32(r.hidden)), Arg(r.aq), Arg(r.aSc)); e != nil {
				return e
			}
		} else if e := r.norm(r.x, Ly.preNorm, r.aq, r.aSc); e != nil {
			return e
		}
		if Ly.qGate {
			// attn_output_gate: q_proj emits [query ‖ gate] PER HEAD at double width. Project into
			// a 2*qDim scratch and split on the ACTIVATION — the weight stays fused because it is
			// quantized, and slicing rows out of an int4 bundle with its per-group scales is real
			// surgery. Interleaved per head, NOT two concatenated blocks.
			if e := r.doG(Ly.q, r.aq, r.aSc, qb, r.dnQg, 0); e != nil {
				return e
			}
			if e := r.launch(r.dnQSplit, g1cfg(Ly.qDim, 256),
				Arg(r.dnQg), Arg(r.qB), Arg(r.dnAGate),
				gpu.ArgValue(int32(Ly.qDim)), gpu.ArgValue(int32(Ly.hd))); e != nil {
				return e
			}
		} else if e := r.doG(Ly.q, r.aq, r.aSc, qb, r.qB, 0); e != nil {
			return e
		}
		// A K/V-shared layer (Gemma 4 E-model) has no k_proj and no v_proj: K/V come from its source layer's cache.
		if !Ly.kvShared {
			if err := r.doG(Ly.k, r.aq, r.aSc, kb, r.kB, 0); err != nil {
				return err
			}
			if Ly.kEqV {
				// K=V: vB is the RAW (pre-norm) k projection, which v_norm consumes below. A copy of kB is bit-identical to
				// projecting k a second time, and is a launch like every other op here, so it records into the CUDA graph and is
				// ordered on this stream between the projection and qk_norm (which rewrites kB in place).
				if err := r.copyF32(r.kB, r.vB, Ly.kvDim); err != nil {
					return err
				}
			} else {
				if err := r.doG(Ly.v, r.aq, r.aSc, vb, r.vB, 0); err != nil {
					return err
				}
			}
		}
	}
	// Compute-time LoRA (docs/tasks/task-gpu-paths-2026-09.md): q/k/v deltas, right after the base projection and
	// before qk_norm/RoPE below, matching applyLoRA's CPU order exactly. Not reached for Ly.qGate (a documented gap,
	// see cuda/lora.go's file comment): qGate's q_proj takes a different path entirely, into r.dnQg, split afterward,
	// so no single hook point covers both.
	if r.loraLayers != nil && !Ly.qGate {
		if r.fuseQKV {
			// The K1 fused kernel computes rmsnorm+quant INTERNALLY and never writes r.aq/r.aSc
			// out — materialize them here (redundant work, correct result: rmsnorm_quant is a
			// deterministic function of r.x/Ly.preNorm) so the LoRA kernels have a valid input.
			if e := r.norm(r.x, Ly.preNorm, r.aq, r.aSc); e != nil {
				return e
			}
		}
		LA := &r.loraLayers[l]
		if e := r.applyLora(LA.q, r.aq, r.aSc, r.hidden, r.qB); e != nil {
			return e
		}
		if e := r.applyLora(LA.k, r.aq, r.aSc, r.hidden, r.kB); e != nil {
			return e
		}
		if e := r.applyLora(LA.v, r.aq, r.aSc, r.hidden, r.vB); e != nil {
			return e
		}
	}
	if r.qkNorm { // per-head (or, for QKNormWhole, whole-vector) Q/K RMSNorm before RoPE
		addOne := int32(0)
		if r.rmsAddOne {
			addOne = 1
		}
		// QKNormWhole (Olmo 3/Olmo Hybrid): the SAME per-head kernel, grid collapsed to one Q
		// block + one K block by passing nH=1,nKV=1,hd=nH_orig*hd — qk_norm's block h<nH reduces
		// [q+h*hd, +hd), so h=0 with hd=nH_orig*hd_orig spans the WHOLE contiguous Q vector, and
		// h=nH=1 spans the whole K vector, matching decoder/attention.go's rmsNorm(q,QNorm,1,
		// nH*hd,...) exactly. Only correct when nH==Ly.nKV (MHA) — BuildResident already declines
		// otherwise (backend.go), so that invariant holds here unconditionally.
		qkNH, qkNKV, qkHD := r.nH, Ly.nKV, Ly.hd
		if r.qkNormWhole {
			qkNH, qkNKV, qkHD = 1, 1, r.nH*Ly.hd
		}
		kNormW := Ly.kNorm
		if Ly.kvShared {
			// Q only: nKV=0 launches no K blocks, and a shared layer has no k_norm weight to bind (an empty Buffer is a nil deref).
			qkNKV, kNormW = 0, Ly.qNorm
		}
		if e := r.launch(r.fQKN, LaunchConfig{GridX: uint32(qkNH + qkNKV), GridY: 1, GridZ: 1,
			BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: 128 * 8},
			Arg(r.qB), Arg(r.kB), Arg(Ly.qNorm), Arg(kNormW),
			gpu.ArgValue(int32(qkNH)), gpu.ArgValue(int32(qkNKV)), gpu.ArgValue(int32(qkHD)),
			gpu.ArgValue(r.eps), gpu.ArgValue(addOne)); e != nil {
			return e
		}
	}
	if Ly.vNorm { // scale-less v_norm on vB (the raw k copy on a K=V layer, the v_proj output otherwise), BEFORE rope_kv rotates k
		if err := r.launch(r.fQKN, LaunchConfig{GridX: uint32(Ly.nKV), GridY: 1, GridZ: 1,
			BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: 128 * 8},
			Arg(r.vB), Arg(r.vB), Arg(r.vNormUnit), Arg(r.vNormUnit),
			gpu.ArgValue(int32(0)), gpu.ArgValue(int32(Ly.nKV)), gpu.ArgValue(int32(Ly.hd)),
			gpu.ArgValue(r.eps), gpu.ArgValue(int32(0))); err != nil {
			return err
		}
	}
	return nil
}

// segAMLA runs the MLA attention block: pre-norm -> Q (direct or LoRA bottleneck) ->
// KV-A projection -> latent store -> W_UK absorb -> Q RoPE -> MLA latent attention ->
// W_UV lift. Leaves the attention context in r.cctx [Ly.qDim], ready for segB (o-proj + FFN).
func (r *cudaResident) segAMLA(Ly *cudaLayer, l, pos, ropePos int) error {
	nullBias := ArgNull()
	// 1. Pre-norm on r.x -> (r.aq, r.aSc)
	if e := r.norm(r.x, Ly.preNorm, r.aq, r.aSc); e != nil {
		return e
	}
	// 2. Query projection: direct or LoRA bottleneck (q_a -> norm -> q_b)
	if r.mlaQLoRA > 0 {
		if e := r.doG(Ly.mlaQA, r.aq, r.aSc, nullBias, r.mlaQAOut, 0); e != nil {
			return e
		}
		if e := r.rmsDim(r.mlaQAOut, Ly.mlaQANorm, r.mlaQLoRA, r.mlaQAQ, r.mlaQASc); e != nil {
			return e
		}
		if e := r.doG(Ly.mlaQB, r.mlaQAQ, r.mlaQASc, nullBias, r.qB, 0); e != nil {
			return e
		}
	} else {
		if e := r.doG(Ly.mlaQ, r.aq, r.aSc, nullBias, r.qB, 0); e != nil {
			return e
		}
	}
	// 3. KV-A projection -> r.mlaKVDown [latDim]
	if e := r.doG(Ly.mlaKVA, r.aq, r.aSc, nullBias, r.mlaKVDown, 0); e != nil {
		return e
	}
	// 4. Latent Store: RMSNorm(kvDown[:rank]) + decoupled RoPE(kvDown[rank:]) -> r.kc[l][pos*latDim]
	var intl int32
	if r.mlaInterleave {
		intl = 1
	}
	base := pos * r.mlaLatDim
	if err := r.launch(r.fMlaStore, onecfg(64, 0),
		Arg(r.mlaKVDown), Arg(Ly.mlaKVANorm), Arg(Ly.invF), Arg(r.kc[l]),
		gpu.ArgValue(int32(r.mlaRank)), gpu.ArgValue(int32(r.mlaQKRope)), gpu.ArgValue(int32(ropePos)),
		gpu.ArgValue(r.eps), gpu.ArgValue(int32(base)), gpu.ArgValue(r.mlaRopeScale), gpu.ArgValue(intl)); err != nil {
		return err
	}
	// 5. W_UK Absorb: per-head block-diagonal matvec of W_UKᵀ with Q's qkNope slice -> r.qAbs
	elemCountAbs := r.nH * r.mlaRank
	cfgAbs := LaunchConfig{GridX: uint32(elemCountAbs), GridY: 1, GridZ: 1, BlockX: 64, BlockY: 1, BlockZ: 1}
	if err := r.launch(r.fMlaHeadMV, cfgAbs,
		Arg(r.qB), Arg(Ly.mlaWUK), Arg(r.qAbs),
		gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(r.mlaRank)), gpu.ArgValue(int32(r.mlaQKNope)),
		gpu.ArgValue(int32(r.mlaQKHead)), gpu.ArgValue(int32(r.mlaLatDim))); err != nil {
		return err
	}
	// 6. Q RoPE: decoupled RoPE on Q's qkRope slice -> r.qAbs[h*latDim + rank]
	half := r.mlaQKRope / 2
	cfgRope := LaunchConfig{GridX: uint32((r.nH*half + 63) / 64), GridY: 1, GridZ: 1, BlockX: 64, BlockY: 1, BlockZ: 1}
	if err := r.launch(r.fMlaQRope, cfgRope,
		Arg(r.qB), Arg(Ly.invF), Arg(r.qAbs),
		gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(r.mlaQKHead)), gpu.ArgValue(int32(r.mlaQKNope)),
		gpu.ArgValue(int32(r.mlaQKRope)), gpu.ArgValue(int32(r.mlaRank)), gpu.ArgValue(int32(r.mlaLatDim)),
		gpu.ArgValue(int32(ropePos)), gpu.ArgValue(intl), gpu.ArgValue(r.mlaRopeScale)); err != nil {
		return err
	}
	// 7. MLA Latent Attention: causal dot over latDim, two-pass softmax, value accumulation over rank prefix -> r.wsum
	nKeys := pos + 1
	nWin := nKeys
	if Ly.window > 0 && nKeys > int(Ly.window) {
		nWin = int(Ly.window)
	}
	cfgAttn := LaunchConfig{
		GridX: uint32(r.nH), GridY: 1, GridZ: 1,
		BlockX: 128, BlockY: 1, BlockZ: 1,
		SharedMemBytes: uint32((nWin + 128) * 4),
	}
	if err := r.launch(r.fMlaAttn, cfgAttn,
		Arg(r.qAbs), Arg(r.kc[l]), Arg(r.wsum),
		gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(r.mlaLatDim)), gpu.ArgValue(int32(r.mlaRank)),
		gpu.ArgValue(int32(pos)), gpu.ArgValue(r.attnScale), gpu.ArgValue(Ly.window), gpu.ArgValue(int32(1))); err != nil {
		return err
	}
	// 8. W_UV Lift: per-head block-diagonal matvec of W_UV with r.wsum -> r.cctx [Ly.qDim]
	elemCountLift := r.nH * r.mlaVHead
	cfgLift := LaunchConfig{GridX: uint32(elemCountLift), GridY: 1, GridZ: 1, BlockX: 64, BlockY: 1, BlockZ: 1}
	if err := r.launch(r.fMlaHeadMV, cfgLift,
		Arg(r.wsum), Arg(Ly.mlaWUV), Arg(r.cctx),
		gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(r.mlaVHead)), gpu.ArgValue(int32(r.mlaRank)),
		gpu.ArgValue(int32(r.mlaRank)), gpu.ArgValue(int32(r.mlaVHead))); err != nil {
		return err
	}
	return nil
}

// segB: context-quant + o-proj (accum into the residual, or sandwich-norm then add) + the MLP up to
// the router readback — the whole dense MLP for a dense layer (no readback gap, segC is nil), or the
// MoE pre-readback half (moeMLPPre / gemma4MoeMLPPre) for a routed layer.
func (r *cudaResident) segB(Ly *cudaLayer, l int, x Buffer) error {
	if Ly.qGate { // ctx *= sigmoid(gate), before o_proj (matches the CPU qwen35Attention)
		if err := r.launch(r.dnAttnGate, g1cfg(Ly.qDim, 256),
			Arg(r.cctx), Arg(r.dnAGate), gpu.ArgValue(int32(Ly.qDim))); err != nil {
			return err
		}
	}
	if err := r.launch(r.fQ, onecfg(256, 256*4), Arg(r.cctx), gpu.ArgValue(int32(Ly.qDim)), Arg(r.cq), Arg(r.cSc)); err != nil {
		return err
	}
	if r.sandwich || r.postOnly || r.parallelBlock {
		if err := r.doG(Ly.o, r.cq, r.cSc, r.oBiasArg(Ly), r.oO, 0); err != nil {
			return err
		}
		// LoRA delta added into the RAW o-proj output, BEFORE the post-attn norm below, matching applyLoRA's CPU order (delta added to `out`, caller norms afterward).
		if r.loraLayers != nil {
			if e := r.applyLora(r.loraLayers[l].o, r.cq, r.cSc, Ly.qDim, r.oO); e != nil {
				return e
			}
		}
		// normF32 no-ops on an empty weight (w.Len()==0) — parallelBlock (Cohere) has no post-attn
		// norm at all, so Ly.postAttnNorm stays unbuilt and this is a pure quantize-then-add for it.
		if err := r.normF32(r.oO, Ly.postAttnNorm); err != nil {
			return err
		}
		if r.subCap {
			r.capVec(r.oO, r.subAttnC, l, r.hidden)
		}
		if err := r.launch(r.fRes, g1cfg(r.hidden, 256), Arg(r.x), Arg(r.oO), gpu.ArgValue(int32(r.hidden))); err != nil {
			return err
		}
	} else {
		if err := r.doG(Ly.o, r.cq, r.cSc, r.oBiasArg(Ly), r.x, 1); err != nil {
			return err
		}
		if r.loraLayers != nil {
			if e := r.applyLora(r.loraLayers[l].o, r.cq, r.cSc, Ly.qDim, r.x); e != nil {
				return e
			}
		}
	}
	return r.segBFFN(Ly, l, x)
}

// segBFFN is segB's FFN half, split out so the Gated-DeltaNet mixer can reuse it. The mixer
// replaces everything segB does BEFORE this point (ctx-quant, o-proj, residual) and nothing after
// it: a DeltaNet layer's FFN sub-block is the ordinary one, dense or MoE, and the router readback
// gap + segC that follow in launchToken are likewise unchanged.
func (r *cudaResident) segBFFN(Ly *cudaLayer, l int, x Buffer) error {
	nullBias := ArgNull()
	if Ly.g4moe {
		return r.gemma4MoeMLPPre(Ly, l, x)
	}
	if Ly.isMoE {
		return r.moeMLPPre(Ly, x)
	}
	// postOnly is a MODEL-level flag, but segBFFN is shared with the DeltaNet mixer's own call site (Olmo Hybrid: the
	// mixer replaces segA+attention, not the FFN half). A DeltaNet layer reaches NormPre2 via NormPlacementLinear
	// regardless of the model-level placement and has a real Ly.postNorm (and no postMLPNorm; see BuildResident's
	// DeltaNet-continue comment), so it must take the normal branches.
	postOnlyHere := r.postOnly && !Ly.isDeltaNet
	// This layer's own FFN width: Gemma 4 E-models mix 6144 and 12288 (docs/tasks/task-multimodal-support-2026-10.md), and the
	// scratch is sized to the widest layer, so every launch below takes the layer's width and never the model-level r.inter.
	ffnI := Ly.ffnI
	if ffnI == 0 {
		ffnI = r.inter
	}
	// Dense MLP (whole): no readback gap, so segC is nil for this layer.
	if r.fuseQKV {
		cfg := LaunchConfig{GridX: uint32((2*ffnI + 63) / 64), GridY: 1, GridZ: 1,
			BlockX: 256, BlockY: 1, BlockZ: 1,
			SharedMemBytes: uint32((r.hidden + 256 + r.hidden/4) * 4)}
		// Rows-per-warp variant (fused_gu_rows.cu) for the geometries measured to win; bit-identical to the original (TestFusedGURowsBitIdentical).
		rpw := r.waveRowsPerWarp(2*ffnI, int(cfg.SharedMemBytes))
		if rpw == 0 {
			rpw = fusedGURowsPerWarp(r.hidden, ffnI) // device shape unread: the measured-geometry table
		}
		if rpw != 8 && r.fGURows != (Pipeline{}) {
			rcfg := cfg
			rcfg.GridX = uint32((2*ffnI + 8*rpw - 1) / (8 * rpw))
			if e := r.launch(r.fGURows, rcfg,
				Arg(x), Arg(Ly.postNorm), gpu.ArgValue(int32(r.hidden)), gpu.ArgValue(r.eps),
				gpu.ArgValue(r.addOneArg()),
				Arg(Ly.g.W), Arg(Ly.g.ws16),
				Arg(Ly.u.W), Arg(Ly.u.ws16),
				gpu.ArgValue(int32(ffnI)), gpu.ArgValue(int32(r.hidden/8)), gpu.ArgValue(int32(r.hidden/32)), gpu.ArgValue(int32(rpw)),
				Arg(r.gO), Arg(r.uO)); e != nil {
				return e
			}
		} else if e := r.launch(r.fGU, cfg,
			Arg(x), Arg(Ly.postNorm), gpu.ArgValue(int32(r.hidden)), gpu.ArgValue(r.eps),
			gpu.ArgValue(r.addOneArg()),
			Arg(Ly.g.W), Arg(Ly.g.ws16),
			Arg(Ly.u.W), Arg(Ly.u.ws16),
			gpu.ArgValue(int32(ffnI)), gpu.ArgValue(int32(r.hidden/8)), gpu.ArgValue(int32(r.hidden/32)),
			Arg(r.gO), Arg(r.uO)); e != nil {
			return e
		}
	} else if r.fuseG32 && !postOnlyHere {
		// Per-32 twin of fGU (fused_rms_gu_g32): bit-identical to the unfused chain below.
		rpw := fusedGURowsPerWarp(r.hidden, ffnI)
		cfg := LaunchConfig{GridX: uint32((2*ffnI + 8*rpw - 1) / (8 * rpw)), GridY: 1, GridZ: 1,
			BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: fusedG32Shmem(r.hidden)}
		if e := r.launch(r.fGUg32, cfg,
			Arg(x), Arg(Ly.postNorm), gpu.ArgValue(int32(r.hidden)), gpu.ArgValue(r.eps),
			gpu.ArgValue(r.addOneArg()),
			Arg(Ly.g.W), wkScale(Ly.g), gpu.ArgValue(wkCode(Ly.g.kind)),
			Arg(Ly.u.W), wkScale(Ly.u), gpu.ArgValue(wkCode(Ly.u.kind)),
			gpu.ArgValue(int32(ffnI)), gpu.ArgValue(int32(rpw)),
			Arg(r.gO), Arg(r.uO)); e != nil {
			return e
		}
	} else {
		gq, gSc := r.mq, r.mSc
		switch {
		case r.parallelBlock:
			// Cohere/Command-R: reuse segA's shared input norm (r.aq/r.aSc) — the MLP consumes
			// the SAME normed+quantized activation the attention branch already computed, not a
			// fresh norm of the post-attention residual (Model.ParallelBlockResident's comment).
			gq, gSc = r.aq, r.aSc
		case postOnlyHere:
			// Olmo 3/Olmo Hybrid: no pre-MLP norm either — quantize the raw residual x directly,
			// same reasoning as segA's attention-side branch.
			if err := r.launch(r.fQ, onecfg(256, 256*4), Arg(x), gpu.ArgValue(int32(r.hidden)), Arg(r.mq), Arg(r.mSc)); err != nil {
				return err
			}
		default:
			if err := r.norm(x, Ly.postNorm, r.mq, r.mSc); err != nil {
				return err
			}
		}
		if err := r.doG(Ly.g, gq, gSc, nullBias, r.gO, 0); err != nil {
			return err
		}
		if err := r.doG(Ly.u, gq, gSc, nullBias, r.uO, 0); err != nil {
			return err
		}
	}
	// Gate/up deltas BEFORE the SwiGLU activation below, matching decoder/mlp.go ("delta into gate/up before the activation").
	if r.loraLayers != nil {
		if r.fuseQKV {
			// fGU computes rmsnorm+quant INTERNALLY (same reasoning as segA's fQKV case) — this
			// branch only runs when !postOnlyHere/!parallelBlock (both force fuseQKV off at
			// BuildResident), so Ly.postNorm is unambiguously the right weight to materialize with.
			if e := r.norm(x, Ly.postNorm, r.mq, r.mSc); e != nil {
				return e
			}
		}
		LA := &r.loraLayers[l]
		if e := r.applyLora(LA.gate, r.mq, r.mSc, r.hidden, r.gO); e != nil {
			return e
		}
		if e := r.applyLora(LA.up, r.mq, r.mSc, r.hidden, r.uO); e != nil {
			return e
		}
	}
	if err := r.launch(r.fSw, onecfg(glueQuantThreads, glueQuantThreads*4), Arg(r.gO), Arg(r.uO), gpu.ArgValue(int32(0)), gpu.ArgValue(int32(0)), gpu.ArgValue(int32(ffnI)),
		gpu.ArgValue(r.act), Arg(r.dq), Arg(r.dSc), Arg(r.dScr)); err != nil {
		return err
	}
	if r.sandwich || postOnlyHere || r.parallelBlock {
		if e := r.doG(Ly.d, r.dq, r.dSc, nullBias, r.dO, 0); e != nil {
			return e
		}
		if r.loraLayers != nil {
			// Added BEFORE the post-MLP norm (sandwich/postOnly) or the deferred residual add
			// (parallelBlock, which has none) — same "delta before any subsequent norm" order.
			if e := r.applyLora(r.loraLayers[l].down, r.dq, r.dSc, ffnI, r.dO); e != nil {
				return e
			}
		}
		if r.subCap {
			r.capVec(r.dO, r.subMLPpreC, l, r.hidden)
		}
		// normF32 no-ops on an empty weight — parallelBlock (Cohere) has no post-MLP norm at all,
		// so Ly.postMLPNorm stays unbuilt and this is a pure down-proj-then-add for it.
		if err := r.normF32(r.dO, Ly.postMLPNorm); err != nil {
			return err
		}
		if r.subCap {
			r.capVec(r.dO, r.subMLPC, l, r.hidden)
		}
		if err := r.launch(r.fRes, g1cfg(r.hidden, 256), Arg(x), Arg(r.dO), gpu.ArgValue(int32(r.hidden))); err != nil {
			return err
		}
		if r.pleP > 0 && !g4SkipPLEForTest {
			// Gemma 4 E-model per-layer embedding branch (decoder/forward_gemma4.go: gate on the RAW residual, geluTanh(gate)*this layer's input,
			// projection, post-norm, residual add), between the FFN residual and the layer scalar. Existing launches only: quant_vec, the GEMV,
			// glu_quant with a GELU-tanh act (0) and the layer's slice of the row's tail as its `up`, the GEMV, the post-norm, the add.
			if e := r.launch(r.fQ, onecfg(256, 256*4), Arg(x), gpu.ArgValue(int32(r.hidden)), Arg(r.mq), Arg(r.mSc)); e != nil {
				return e
			}
			if e := r.doG(Ly.pleGate, r.mq, r.mSc, nullBias, r.pleG, 0); e != nil {
				return e
			}
			if e := r.launch(r.fSw, onecfg(glueQuantThreads, glueQuantThreads*4), Arg(r.pleG), Arg(r.x), gpu.ArgValue(int32(0)), gpu.ArgValue(int32(r.hidden+(l+pleLayerShiftForTest)*r.pleP)), gpu.ArgValue(int32(r.pleP)),
				gpu.ArgValue(int32(0)), Arg(r.dq), Arg(r.dSc), Arg(r.dScr)); e != nil {
				return e
			}
			if e := r.doG(Ly.pleProj, r.dq, r.dSc, nullBias, r.dO, 0); e != nil {
				return e
			}
			if e := r.normF32(r.dO, Ly.postPLENorm); e != nil {
				return e
			}
			if e := r.launch(r.fRes, g1cfg(r.hidden, 256), Arg(x), Arg(r.dO), gpu.ArgValue(int32(r.hidden))); e != nil {
				return e
			}
		}
		// Gemma 4 dense per-layer output scalar (forward_gemma4.go's `if lw.LayerScalar != 0 { h *= lw.LayerScalar }`),
		// applied AFTER the MLP residual add, matching the CPU order; on an E-model it follows the PLE branch above. 0
		// (every non-gemma4 family, and Gemma4DenseLayerScalarAtResident's own "skip" sentinel) never launches the kernel.
		if Ly.layerScalar != 0 && !g4DropLayerScalarForTest {
			if err := r.launch(r.fScaleVec, g1cfg(r.hidden, 256), Arg(x), gpu.ArgValue(Ly.layerScalar), gpu.ArgValue(int32(r.hidden))); err != nil {
				return err
			}
		}
	} else {
		if e := r.doG(Ly.d, r.dq, r.dSc, nullBias, x, 1); e != nil {
			return e
		}
		if r.loraLayers != nil {
			if e := r.applyLora(r.loraLayers[l].down, r.dq, r.dSc, ffnI, x); e != nil {
				return e
			}
		}
	}
	return nil
}

// segC: the post-readback MoE half (expert loop + combine/join). nil for a dense layer.
func (r *cudaResident) segC(Ly *cudaLayer, l int, x Buffer) error {
	if Ly.g4moe {
		return r.gemma4MoeMLPPost(Ly, l, x)
	}
	if Ly.isMoE {
		return r.moeMLPPost(Ly, x)
	}
	return nil
}

// captureGraphs records each layer's three static segments once (on the executor thread, so the
// thread-local capture matches the replay thread). Off unless r.graphs; incompatible with the subCap/
// g4cap diagnostics (they sync inside a segment, which stream capture forbids) — the caller gates on
// !g4cap and launchToken's useGraphs gates on !subCap. A capture failure is a build error → the
// resident declines to the staged path, never runs a half-captured chain.
func (r *cudaResident) captureGraphs() error {
	for l := range r.layers {
		Ly, ll := &r.layers[l], l
		if Ly.isDeltaNet {
			// A Gated-DeltaNet layer is MORE graph-static than an attention one: no rope and no attention, so no per-token
			// dynamic uniform. The recurrence's only per-token input is the residual, which is buffer CONTENTS, and replay
			// reads current contents (TestCUDA_graphReplay). So the whole pre-routing half captures as ONE segment, mixer +
			// FFN-pre, and segB stays nil (no attention gap to split around).
			gM, e := r.stream.Capture(func() error {
				if err := r.deltaNetMixer(Ly, ll); err != nil {
					return err
				}
				return r.segBFFN(Ly, ll, r.x)
			})
			if e != nil {
				return fmt.Errorf("layer %d deltanet mixer+ffn: %w", l, e)
			}
			r.layers[l].gSegA = gM
			if Ly.g4moe || Ly.isMoE {
				gC, e := r.stream.Capture(func() error { return r.segC(Ly, ll, r.x) })
				if e != nil {
					return fmt.Errorf("layer %d segC: %w", l, e)
				}
				r.layers[l].gSegC = gC
			}
			continue
		}
		if Ly.isMLA {
			// For MLA, segA is dynamic (rope and attention depend on pos) and graphs are disabled.
			continue
		}
		gA, e := r.stream.Capture(func() error { return r.segA(Ly, ll) })
		if e != nil {
			return fmt.Errorf("layer %d segA: %w", l, e)
		}
		gB, e := r.stream.Capture(func() error { return r.segB(Ly, ll, r.x) })
		if e != nil {
			return fmt.Errorf("layer %d segB: %w", l, e)
		}
		r.layers[l].gSegA, r.layers[l].gSegB = gA, gB
		if Ly.g4moe || Ly.isMoE {
			gC, e := r.stream.Capture(func() error { return r.segC(Ly, ll, r.x) })
			if e != nil {
				return fmt.Errorf("layer %d segC: %w", l, e)
			}
			r.layers[l].gSegC = gC
		}
	}
	r.launchErr = nil // capture invoked launch() to RECORD (not execute); clear the sticky accumulator
	return nil
}

// launchToken issues one token's whole kernel chain, leaving logits[vocab] on the device.
// ropePos is the rope_kv kernel's rotation-angle position — equal to pos for every ordinary
// (non-m-RoPE) call; only Qwen2.5-VL decode past an image block needs them to differ (see
// gemv_fwd.cu's rope_kv doc comment). pos alone still drives KV storage/attention range.
func (r *cudaResident) launchToken(emb []float32, pos, ropePos int, head bool) error {
	r.launchErr = nil // reset the sticky launch-error accumulator for this token (M23)
	nullBias := ArgNull()
	// qTempScale (Ministral 3, FeatAttnTemp; docs/tasks/task-gpu-paths-2026-09.md): computed ONCE per token (the same
	// value at every layer, unlike mscale, which is per-layer); mirrors decoder.Model.AttnTempScale. attnTempBeta==0
	// (every family without the feature) skips the division: attnTempOrigMaxPos is 0 for those, and pos/0 would poison
	// Q with NaN.
	qTempScale := r.qTempScaleAt(pos)
	if e := r.kvEnsure(pos, 1); e != nil {
		return e
	}
	if r.pleP > 0 && len(emb) != r.embLen {
		// An E-model row is [hidden ‖ nLayers*P]. A short row would leave the PLE tail stale from the previous token, and a plain hidden-sized
		// row (what every other model passes) would silently read the previous token's inputs: refuse by length, here, for every entry point.
		return fmt.Errorf("cuda: Gemma 4 E-model embedding row has %d floats, want %d (hidden %d + %d layers x PLE width %d)", len(emb), r.embLen, r.hidden, r.nLayers, r.pleP)
	}
	if e := gpu.Upload(r.x, emb); e != nil {
		return e
	}
	if r.cacheProf {
		r.profT0 = time.Now()
	}
	// useGraphs replays the captured static segments instead of re-issuing their launches. Gated on
	// !subCap: the sublayer-capture diagnostic syncs mid-segment, which a captured graph cannot do
	// (and which a test may enable on a graphs-built runner) — so it falls back to the live seg calls.
	useGraphs := r.graphs && !r.subCap
	gA := useGraphs && (r.graphMask == "" || strings.Contains(r.graphMask, "A"))
	gB := useGraphs && (r.graphMask == "" || strings.Contains(r.graphMask, "B"))
	gC := useGraphs && (r.graphMask == "" || strings.Contains(r.graphMask, "C"))
	for l := 0; l < r.nLayers; l++ {
		Ly := &r.layers[l]
		if Ly.isDeltaNet {
			// Gated-DeltaNet mixer: replaces segA + rope + attention + o-proj (no KV cache, nothing positional: the state IS
			// the history). The FFN sub-block and the router-readback gap below are the ordinary ones, so this rejoins the
			// shared path at segBFFN. Captured as ONE segment (gSegA) when graphs are on: with no rope/attention gap there is
			// nothing dynamic to split around.
			if gA {
				if e := Ly.gSegA.Replay(); e != nil {
					return e
				}
				if r.graphsSync {
					if e := r.stream.Sync(); e != nil {
						return e
					}
				}
			} else {
				if e := r.deltaNetMixer(Ly, l); e != nil {
					return e
				}
				if e := r.segBFFN(Ly, l, r.x); e != nil {
					return e
				}
			}
			if e := r.layerTail(Ly, l, gC, r.x); e != nil {
				return e
			}
			continue
		}
		if Ly.isMLA {
			if e := r.segAMLA(Ly, l, pos, ropePos); e != nil {
				return e
			}
		} else {
			// segA: QKV proj + qk/v-norm (pre-RoPE, static).
			if gA {
				if e := Ly.gSegA.Replay(); e != nil {
					return e
				}
				if r.graphsSync {
					if err := r.stream.Sync(); err != nil {
						return err
					}
				}
			} else if e := r.segA(Ly, l); e != nil {
				return e
			}
			// --- dynamic gap: rope_kv + attention (decodeAttnGap; MC3 on CUDA runs it per sequence inside a step).
			if e := r.decodeAttnGap(Ly, l, pos, ropePos, qTempScale); e != nil {
				return e
			}
		}
		if r.subCap { // pre-o-proj attention context (qDim), before quant — the cross-box discriminator (live path only)
			r.capVec(r.cctx, r.subCtxC, l, Ly.qDim)
		}
		// segB: ctx-quant + o-proj + MLP up to the router readback (whole dense MLP for a dense layer).
		if gB {
			if e := Ly.gSegB.Replay(); e != nil {
				return e
			}
			if r.graphsSync {
				if err := r.stream.Sync(); err != nil {
					return err
				}
			}
		} else if e := r.segB(Ly, l, r.x); e != nil {
			return e
		}
		// --- dynamic gap: clear the g4moe accumulator (H2D), then, if caching, read the routing back
		// to the host and DMA the routed experts into their VRAM slots (D2H+H2D). Host copies, not
		// graph-capturable; synchronous, so the slots are filled before segC replays.
		if e := r.layerTail(Ly, l, gC, r.x); e != nil {
			return e
		}
	}
	// Final norm + LM head — skipped for KV-only prefill (head=false): prompt[:-1] tokens need only
	// their K/V in the cache, and the head is a big-vocab matmul + ~1 MB readback + softcap. The layer
	// loop above already wrote this position's K/V identically, so decode stays byte-identical.
	if head {
		if e := r.norm(r.x, r.finalNorm, r.aq, r.aSc); e != nil {
			return e
		}
		if e := r.doG(r.lmW, r.aq, r.aSc, nullBias, r.logits, 0); e != nil {
			return e
		}
		r.profMark(&r.profHead)
	}
	return r.launchErr // surface any launch error discarded in the dense chain above (M23)
}

// decodeAttnGap is one layer's dynamic gap of a single-token decode: rope_kv at (pos, ropePos) into the BOUND KV
// slot, then the attention launch decode chooses for this position (the flash-decode lane, split-KV, the coalesced
// attn_batched(M=1), or the glue kernel), reading r.qB/r.kB/r.vB and writing r.cctx. MC3 on CUDA (batchstep.go)
// calls it per sequence inside a step, with those four buffers pointed at the sequence's row and its slot bound,
// so every row gets exactly its own decode's kernels.
func (r *cudaResident) decodeAttnGap(Ly *cudaLayer, l, pos, ropePos int, qTempScale float32) error {
	// fused rope(q)+rope(k)+kv_store(k)+kv_store(v): rhalf == hd/2 for full rotary, rotaryDim/2 for partial.
	// A KV-shared layer (Gemma 4 E-model) rotates Q only: nKV=0 leaves rope_kv's K and V work empty, so nothing is stored
	// through the alias into its source's cache. The attention launches below keep the real Ly.nKV, which is how they read it.
	ropeNKV := Ly.nKV
	if Ly.kvShared && !g4KeepSharedKVStoreForTest {
		ropeNKV = 0
	}
	if err := r.launch(r.ropeKV, g1cfg(r.nH*Ly.rhalf+ropeNKV*Ly.rhalf+ropeNKV*(Ly.hd-2*Ly.rhalf), 256),
		Arg(r.qB), Arg(r.kB), Arg(r.vB), Arg(Ly.invF), Arg(r.kvK(l)), Arg(r.kvV(l)),
		gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(ropeNKV)), gpu.ArgValue(int32(Ly.hd)),
		gpu.ArgValue(int32(pos)), gpu.ArgValue(int32(ropePos)), gpu.ArgValue(int32(Ly.rhalf)),
		gpu.ArgValue(Ly.mscale), gpu.ArgValue(qTempScale)); err != nil {
		return err
	}
	nKeys := pos + 1
	// Sliding window (per layer: Mistral all-local, Mellum interleaves); shared sized to the attended span.
	nWin := nKeys
	if Ly.window > 0 && nKeys > int(Ly.window) {
		nWin = int(Ly.window)
	}
	// Split-KV is REQUIRED, not merely preferred, once the single-block launch would exceed the device's shared-memory
	// limit. The `r.splitkvAttn` gate and the per-geometry perf threshold both describe when split-KV is FASTER;
	// neither knows when the alternative cannot run at all. Without this, -ctx 16384 on a geometry where split-KV is
	// otherwise off (splitkvNever, or GOINFER_SPLITKV_ATTN=0) fails at position 12,160, or silently drops to the
	// sequential prefill.
	mustSplit := splitKVRequired(nWin)
	if mustSplit && (!r.splitkvAttn || r.skScores == (Pipeline{})) {
		return fmt.Errorf("cuda: attention at %d attended keys needs %d B of shared memory, "+
			"past this device's %d B limit, and split-KV is unavailable (%s) — lower -ctx or "+
			"re-enable GOINFER_SPLITKV_ATTN (M-16)", nWin, attnShmemBytes(nWin),
			singleBlockAttnShmemLimit,
			map[bool]string{true: "kernel not loaded", false: "disabled by GOINFER_SPLITKV_ATTN"}[r.skScores == (Pipeline{})])
	}
	if r.faSplit > 0 && r.faExactScope.Load() == 0 && nWin >= r.faMinKeys && r.faEligible(l) {
		// Flash-decode lane (default ON, GOINFER_CUDA_FLASH_DECODE=0 to disable): NOT bit-identical, fidelity-gated.
		// Checked first because it replaces the exact path's three launches outright when eligible.
		if err := r.flashDecodeAttn(l, pos); err != nil {
			return err
		}
	} else if r.splitkvAttn && r.skScores != (Pipeline{}) && (mustSplit || nWin >= r.splitkvMin(Ly.nKV, Ly.hd)) {
		// Split-KV: high-occupancy and BIT-IDENTICAL to attn_batched(M=1) (TestSplitKV_bitIdentical); fills the SMs the
		// single-block kernel leaves idle at long ctx. Gated PER LAYER on nWin (the EFFECTIVE attended span) against a
		// per-geometry threshold, so shallow decode keeps the cheaper single-block path. nWin, not nKeys: a sliding-window
		// layer never attends more than `window` keys, so its cost is set by the window, not by position (gating on
		// position made gemma3's windowed layers take the split path, their loss regime, at every depth past the window).
		// Both arms are byte-identical, so a layer flipping arms mid-request as nWin grows is safe by construction.
		if err := r.splitKVAttnDecode(l, pos); err != nil {
			return err
		}
	} else if r.prefillReady {
		// Coalesced M=1 decode attention: attn_batched with M=1 is BIT-IDENTICAL to the glue `attention`
		// (TestAttnBatched_bitIdentical) but reads K via float4, which recovers the L1TEX-latency-bound glue decode
		// attention at long context. startPos=pos, M=1 gives nKeys = pos+1 and the same GridX/block/shared/ctx-layout as
		// the glue launch, so decode stays byte-identical. The glue `attention` (audited) is UNTOUCHED and is the fallback
		// below.
		if err := r.launch(r.bAttn, LaunchConfig{GridX: uint32(r.nH), GridY: 1, GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32((nWin + 128) * 4)},
			Arg(r.qB), Arg(r.kvK(l)), Arg(r.kvV(l)), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(Ly.nKV)), gpu.ArgValue(int32(Ly.hd)), gpu.ArgValue(int32(pos)), gpu.ArgValue(r.attnScale), gpu.ArgValue(Ly.window), gpu.ArgValue(int32(1)), Arg(r.cctx), r.sinkArg(l)); err != nil {
			return err
		}
	} else {
		if err := r.launch(r.fAttn, LaunchConfig{GridX: uint32(r.nH), GridY: 1, GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32((nWin + 128) * 4)},
			Arg(r.qB), Arg(r.kvK(l)), Arg(r.kvV(l)), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(Ly.nKV)), gpu.ArgValue(int32(Ly.hd)), gpu.ArgValue(int32(nKeys)), gpu.ArgValue(r.attnScale), gpu.ArgValue(Ly.window), Arg(r.cctx)); err != nil {
			return err
		}
	}
	return nil
}

// step returns full logits — the general contract (sampler / constrained decode / logprobs).
// Costs a vocab*4 B D2H every token (594 KB at a 151936 vocab). ropePos: see launchToken.
func (r *cudaResident) step(emb []float32, pos, ropePos int) ([]float32, error) {
	if e := r.launchToken(emb, pos, ropePos, true); e != nil {
		return nil, e
	}
	if e := r.stream.Sync(); e != nil {
		return nil, e
	}
	if e := gpu.ReadToHost(r.logits, r.logitsPinned); e != nil {
		return nil, e
	}
	// Final-logit softcap (Gemma 2/4): softcap·tanh(logits/softcap), applied ONCE to the logit vector after the LM
	// head, host-side exactly as the CPU path (forwardn.go / logitsFromHidden) and as FeatEmbedScale's √hidden is.
	// This is what FeatFinalLogitSoftcap declares on this backend; 0 for every non-softcapped family (no-op). Covers
	// Forward and ForwardN (both route through step).
	applySoftcap(r.logitsHost, r.finalSoftcap)
	// Cohere/Command-R's logits_scaling (FeatLogitScale): a plain host-side multiply, the same
	// shape and site as the softcap just above. 0 for every non-FeatLogitScale family (no-op).
	applyLogitScale(r.logitsHost, r.logitScale)
	return r.logitsHost, nil
}

// ForwardArgmax is the greedy fast path (decoder.ResidentGreedy): reduce the argmax on-device
// and read back 4 B instead of the whole logits vector. Same kernel chain, same numerics —
// only the readback differs, so the id equals argmax(Forward(...)) exactly.
func (r *cudaResident) ForwardArgmax(embedding []float32, pos int) (int, error) {
	if e := r.checkCap(pos, 1); e != nil {
		return 0, e
	}
	var id int
	err := r.do(func() error {
		if e := r.launchToken(embedding, pos, pos, true); e != nil {
			return e
		}
		if e := r.launch(r.fArg, onecfg(256, 256*4+256*4), Arg(r.logits),
			gpu.ArgValue(int32(r.vocab)), Arg(r.argIdx), Arg(r.argVal)); e != nil {
			return e
		}
		if e := r.stream.Sync(); e != nil {
			return e
		}
		out := make([]int32, 1)
		if e := gpu.Download(r.argIdx, out); e != nil {
			return e
		}
		id = int(out[0])
		return nil
	})
	return id, err
}

// --- host-side weight packing (CPU; runs before any CUDA) ---

type hostW struct {
	kind string
	wpk  []uint32
	ws   []float32 // int8 row scales
	ws16 []uint16  // int4 group scales (f16)
	N, K int
}

func packI8(q8 []int8, N, K int) []uint32 {
	p := make([]uint32, N*(K/4))
	for i := range p {
		p[i] = uint32(uint8(q8[i*4])) | uint32(uint8(q8[i*4+1]))<<8 | uint32(uint8(q8[i*4+2]))<<16 | uint32(uint8(q8[i*4+3]))<<24
	}
	return p
}

// packWeight quantizes/repacks a projection into the resident device layout, or errors
// (so BuildResident can decline gracefully → staged fallback) for shapes it can't handle.
func packWeight(w *linalg.WeightMat) (hostW, error) {
	N, K := w.Rows(), w.Cols()
	if K%4 != 0 {
		return hostW{}, fmt.Errorf("cuda: K=%d not a multiple of 4", K)
	}
	switch w.Kind() {
	case "int4":
		if K%32 != 0 {
			return hostW{}, fmt.Errorf("cuda: int4 K=%d not a multiple of 32", K)
		}
		// Kind() reports precision, not layout, and stays "int4" for a repacked-only tensor, so this switch alone cannot
		// tell canonical from repacked-only; without the ok, a repacked-only tensor's nil q4 would index out of range
		// below instead of declining cleanly (decoder/weightmat.go's wantsCanonicalInt4 doc has the policy). Repacked-only
		// activates only for Options.Backend=="cpu", never "cuda", so this is defense in depth, not a path expected to
		// fire (docs/code-notes/cuda.md#packWeight.int4).
		q4, sc, _, ok := decoder.Int4F32(w)
		if !ok {
			return hostW{}, fmt.Errorf("cuda: int4 tensor has no canonical bytes (layout %s-only): "+
				"model was loaded for a CPU-only backend; load with Options.Backend set to "+
				"\"cuda\" to keep canonical bytes for residency", w.Int4Layout())
		}
		wpk := make([]uint32, N*(K/8))
		for i := range wpk {
			b := q4[i*4 : i*4+4]
			wpk[i] = permuteFast(uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24)
		}
		gs := make([]uint16, len(sc))
		for i, v := range sc {
			gs[i] = f32tof16(v)
		}
		return hostW{kind: "int4", wpk: wpk, ws16: gs, N: N, K: K}, nil
	case "int8":
		q8, sc, _, _ := w.Int8()
		return hostW{kind: "int8", wpk: packI8(q8, N, K), ws: sc, N: N, K: K}, nil
	case "q4k":
		// GGUF Q4_K super-blocks verbatim (--quant q4k), as little-endian words: 36 per 256 weights.
		raw, _ := w.Q4K()
		if K%256 != 0 || len(raw) != N*K/256*144 {
			return hostW{}, fmt.Errorf("cuda: q4k tensor %dx%d with %d bytes", N, K, len(raw))
		}
		wpk := make([]uint32, len(raw)/4)
		for i := range wpk {
			wpk[i] = uint32(raw[4*i]) | uint32(raw[4*i+1])<<8 | uint32(raw[4*i+2])<<16 | uint32(raw[4*i+3])<<24
		}
		return hostW{kind: "q4k", wpk: wpk, N: N, K: K}, nil
	case "f32":
		f32, _ := w.F32()
		q8, sc := linalg.QuantizeRowsInt8(f32, N, K)
		return hostW{kind: "int8", wpk: packI8(q8, N, K), ws: sc, N: N, K: K}, nil
	default:
		return hostW{}, fmt.Errorf("cuda: unsupported projection kind %q", w.Kind())
	}
}

// packWeightStack row-stacks several same-K weights into ONE packed buffer, so a routed expert
// is selected by INDEXING a row range rather than by binding a different buffer per token
// (gemv_w4a8_moe: wrow = idx[slot]*rowsPerExpert + row). Fixed launch geometry is what lets the
// resident runner keep a static dispatch chain regardless of which experts a token picks.
//
// It composes packWeight rather than re-implementing the layout, so a stacked expert's bytes
// are IDENTICAL to the same weight packed alone — the nibble permutation, group-scale f16
// rounding and row order all come from the one packer. A second copy of that layout is how the
// indexed reads would silently land on garbage: the GEMV cannot tell a mis-packed row from a
// real one, it just returns a plausible wrong number.
//
// Every input must share kind and K: the kernel derives its stride from one Kwords/Kgroups, so
// a ragged stack would read across row boundaries.
func packWeightStack(ws ...*linalg.WeightMat) (hostW, error) {
	if len(ws) == 0 {
		return hostW{}, fmt.Errorf("cuda: packWeightStack needs at least one weight")
	}
	var out hostW
	for i, w := range ws {
		h, err := packWeight(w)
		if err != nil {
			return hostW{}, fmt.Errorf("cuda: packWeightStack[%d]: %w", i, err)
		}
		if i == 0 {
			out.kind, out.K = h.kind, h.K
			// Reserve the whole stack up front: without it the appends below regrow the slice geometrically and leave every
			// outgrown copy as garbage, and across all layers' expert stacks the collector lets that garbage pile up
			// (docs/code-notes/cuda.md#packWeightStack.reserve).
			if h.N > 0 {
				totalN := 0
				for _, x := range ws {
					totalN += x.Rows()
				}
				if n := len(h.wpk); n > 0 {
					out.wpk = make([]uint32, 0, n/h.N*totalN)
				}
				if n := len(h.ws); n > 0 {
					out.ws = make([]float32, 0, n/h.N*totalN)
				}
				if n := len(h.ws16); n > 0 {
					out.ws16 = make([]uint16, 0, n/h.N*totalN)
				}
			}
		}
		if h.kind != out.kind {
			return hostW{}, fmt.Errorf("cuda: packWeightStack[%d]: kind %q != %q — a mixed-precision "+
				"stack cannot share one kernel's unpack path", i, h.kind, out.kind)
		}
		if h.K != out.K {
			return hostW{}, fmt.Errorf("cuda: packWeightStack[%d]: K=%d != %d — the kernel strides by a "+
				"single Kwords, so a ragged stack reads across row boundaries", i, h.K, out.K)
		}
		out.wpk = append(out.wpk, h.wpk...)
		out.ws = append(out.ws, h.ws...)
		out.ws16 = append(out.ws16, h.ws16...)
		out.N += h.N
	}
	return out, nil
}

// SetHiddenCapture arms the resident hidden-state seam for the given target layer indices (layer OUTPUTS,
// ascending: the convention decoder.Model.ForwardCapture uses and DeepSpec/z-lab's `target_layer_ids` name). Pass
// nil to disarm. It is the resident counterpart of the CPU ForwardCapture seam that a hidden-state drafter needs
// (not yet wired into production; see cudaResident.hidCap). Cost is one sync + one hidden-sized download per
// TAPPED layer per token, so arm only the taps the drafter actually reads.
func (r *cudaResident) SetHiddenCapture(taps []int) error {
	if len(taps) == 0 {
		r.hidCapTaps, r.hidCapOut = nil, nil
		return nil
	}
	prev := -1
	for _, t := range taps {
		if t < 0 || t >= r.nLayers {
			return fmt.Errorf("cuda: hidden-capture tap %d out of range [0,%d)", t, r.nLayers)
		}
		if t <= prev {
			return fmt.Errorf("cuda: hidden-capture taps must be ascending and distinct, got %v", taps)
		}
		prev = t
	}
	r.hidCapTaps = append([]int(nil), taps...)
	r.hidCapOut = make([][]float32, len(taps))
	return nil
}

// HiddenCapture returns the most recent token's captured layer outputs, one row per tap in
// SetHiddenCapture order. The rows are owned by the caller (capVec allocates per token).
func (r *cudaResident) HiddenCapture() [][]float32 { return r.hidCapOut }

// dnetParams carries the model-level Gated-DeltaNet geometry (uniform across the linear layers).
// keyDim/valueDim/convDim are derived once rather than at every dispatch because the three are
// easy to conflate: convDim is 2*keyDim+valueDim (the conv runs over [q|k|v] together), and
// rep = nv/nk is the GVA factor mapping value heads to key heads.
type dnetParams struct {
	convK      int
	hk, hv     int
	nk, nv     int
	rep        int
	keyDim     int
	valueDim   int
	convDim    int
	stateElems int
	qScale     float32
}

// deltaNetMixer runs one Gated-DeltaNet layer's sequence mixer and folds its output into the
// residual — the recurrent replacement for segA + rope + attention + o-proj.
//
//	norm → in_proj_qkv → conv(ring) → l2norm(q,k) → gates → delta rule(state) → gated norm × silu(z)
//	     → out_proj + residual
//
// With graphs on, captureGraphs records this together with segBFFN as the layer's single gSegA segment: the state
// buffers' contents ARE the per-token state, and replay reads current contents like any other buffer.
//
// The two small gate projections (dnB/dnA) run off the SAME quantized activation as the big ones.
// On the CPU they are f32 by deliberate choice: deltaNetWeights keeps inProjB/inProjA unquantized
// because they feed the write/decay gates, where the recurrence is most precision-sensitive. This
// is the one place the resident path is knowingly coarser than the reference, and the parity gate
// scores the gates as their own stage so the cost of that choice is visible rather than assumed.
func (r *cudaResident) deltaNetMixer(Ly *cudaLayer, l int) error {
	dp := r.dnet
	nullBias := ArgNull()
	if e := r.rms(r.x, Ly.preNorm, r.aq, r.aSc); e != nil {
		return e
	}
	if e := r.doG(Ly.dnQKV, r.aq, r.aSc, nullBias, r.dnMixed, 0); e != nil {
		return e
	}
	if e := r.doG(Ly.dnB, r.aq, r.aSc, nullBias, r.dnBt, 0); e != nil {
		return e
	}
	if e := r.doG(Ly.dnA, r.aq, r.aSc, nullBias, r.dnAt, 0); e != nil {
		return e
	}
	if e := r.doG(Ly.dnZ, r.aq, r.aSc, nullBias, r.dnZOut, 0); e != nil {
		return e
	}
	if e := r.launch(r.dnConv, g1cfg(dp.convDim, 256),
		Arg(r.dnMixed), Arg(Ly.dnConvW), Arg(Ly.dnWin), Arg(r.dnConvOut),
		gpu.ArgValue(int32(dp.convDim)), gpu.ArgValue(int32(dp.convK))); e != nil {
		return e
	}
	if e := r.launch(r.dnGates, g1cfg(dp.nv, 64),
		Arg(r.dnBt), Arg(r.dnAt), Arg(Ly.dnDtBias), Arg(Ly.dnNegExpA), Arg(r.dnHeadP),
		gpu.ArgValue(int32(dp.nv))); e != nil {
		return e
	}
	if e := r.launch(r.dnNorm, LaunchConfig{GridX: uint32(dp.nk), GridY: 1, GridZ: 1,
		BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: 2 * 128 * 4},
		Arg(r.dnConvOut), Arg(r.dnQn), Arg(r.dnKn),
		gpu.ArgValue(int32(dp.nk)), gpu.ArgValue(int32(dp.hk)), gpu.ArgValue(int32(dp.keyDim)),
		gpu.ArgValue(dp.qScale)); e != nil {
		return e
	}
	// v is the conv output's third slice; vBase points at it rather than copying it out.
	if e := r.launch(r.dnRule, g1cfg(dp.valueDim, 128),
		Arg(r.dnQn), Arg(r.dnKn), Arg(r.dnConvOut), Arg(r.dnHeadP), Arg(Ly.dnState), Arg(r.dnCore),
		gpu.ArgValue(int32(dp.nv)), gpu.ArgValue(int32(dp.hk)), gpu.ArgValue(int32(dp.hv)),
		gpu.ArgValue(int32(dp.rep)), gpu.ArgValue(int32(2*dp.keyDim))); e != nil {
		return e
	}
	if e := r.launch(r.dnGNorm, LaunchConfig{GridX: uint32(dp.nv), GridY: 1, GridZ: 1,
		BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: 128 * 4},
		Arg(r.dnCore), Arg(r.dnZOut), Arg(Ly.dnNormW), Arg(r.dnGated),
		gpu.ArgValue(int32(dp.nv)), gpu.ArgValue(int32(dp.hv)), gpu.ArgValue(r.eps)); e != nil {
		return e
	}
	// Quantize the gated output, then out_proj straight into the residual (accum=1), exactly as
	// the attention path's o-proj does.
	if e := r.launch(r.fQ, onecfg(256, 256*4),
		Arg(r.dnGated), gpu.ArgValue(int32(dp.valueDim)), Arg(r.dnGq), Arg(r.dnGSc)); e != nil {
		return e
	}
	return r.doG(Ly.dnOut, r.dnGq, r.dnGSc, nullBias, r.x, 1)
}

// layerTail is everything launchToken does AFTER the FFN pre-half: the g4moe accumulator clear,
// the C′ expert DMA, segC, and the two capture seams. Split out so the Gated-DeltaNet mixer path
// rejoins it instead of duplicating it — a DeltaNet layer's FFN can be MoE (qwen3_5_moe,
// qwen3_next) and would otherwise skip the router readback entirely.
func (r *cudaResident) layerTail(Ly *cudaLayer, l int, gC bool, x Buffer) error {
	if Ly.g4moe {
		// segC(l-1) writes AND reads g4x2 on r.stream (CU_STREAM_NON_BLOCKING). The clear must land after that read and
		// before segC(l) accumulates into it. A memset ON r.stream is ordered by the stream itself, in every mode, and
		// drains nothing. Do not clear with gpu.Upload of a host zero slice: it runs its copy on the context's legacy null
		// stream, which has NO ordering against r.stream (aikit gpu/cuda.go), so it needs a Sync first, plus two more
		// context syncs and a pageable copy. The router readback that follows (loadRoutedExperts) still syncs r.stream
		// itself, so the wait is not lost, only moved to where it was needed anyway.
		if e := r.stream.ZeroAsync(r.g4x2, r.hidden*4); e != nil {
			return e
		}
		r.profMark(&r.profClear)
	}
	if (Ly.g4moe || Ly.isMoE) && r.cacheExperts {
		if e := r.loadRoutedExperts(Ly); e != nil {
			return e
		}
		if Ly.g4moe {
			r.profMark(&r.profRT)
		}
	}
	// segC: the post-readback MoE half (expert loop + join). Dense layers have none.
	if Ly.g4moe || Ly.isMoE {
		if gC {
			if e := Ly.gSegC.Replay(); e != nil {
				return e
			}
			if r.graphsSync {
				if err := r.stream.Sync(); err != nil {
					return err
				}
			}
		} else if e := r.segC(Ly, l, x); e != nil {
			return e
		}
		if Ly.g4moe {
			r.profMark(&r.profSegC)
		}
	}
	// hidden-state seam: this layer's OUTPUT residual, for a drafter that taps it.
	if len(r.hidCapTaps) > 0 {
		for slot, tap := range r.hidCapTaps {
			if tap == l {
				r.capVec(x, r.hidCapOut, slot, r.hidden)
				break
			}
		}
	}
	if r.layerCap { // DEBUG: snapshot the residual after this layer (divergence-localization probe)
		if err := r.stream.Sync(); err != nil {
			return err
		}
		h := make([]float32, r.hidden)
		if err := gpu.Download(x, h); err != nil {
			return err
		}
		r.layerCapBuf = append(r.layerCapBuf, h)
	}
	return nil
}

// actScaleLen is how many f32 activation scales a K-wide int8 activation carries: 1 per vector, or
// K/32 under per-32 activation quantization (actG32).
func (r *cudaResident) actScaleLen(k int) int {
	if r.actG32 {
		return 2 * k / 32 // scales, then aS·Σaq per group (quantG32; read by gemv_q4k_g32)
	}
	return 1
}

// wkCode is actgroup.cu's WK_* code for a per-32 weight kind (fused_rms_qkv_g32 / fused_rms_gu_g32).
func wkCode(kind string) int32 {
	switch kind {
	case "q4k":
		return 2
	case "int4":
		return 1
	default:
		return 0 // int8
	}
}

// g32Fusable reports whether every listed projection has a kind the fused per-32 kernels read.
func g32Fusable(kinds ...string) bool {
	for _, k := range kinds {
		if k != "int8" && k != "int4" && k != "q4k" {
			return false
		}
	}
	return true
}

// wkScale is a projection's scale argument for the fused per-32 kernels: the int8 row scales, the
// int4 f16 group scales, or null for q4k (its scales live in the super-blocks).
func wkScale(w cudaWQ) KernelArg {
	switch w.kind {
	case "q4k":
		return ArgNull()
	case "int4":
		return Arg(w.ws16)
	default:
		return Arg(w.ws)
	}
}

// fusedG32MaxHidden is the widest hidden size the fused per-32 kernels are used at: the largest size measured to
// win (H=1536). Wider measured sizes (Phi-3 at 3072, qwen2.5-7b at 3584) were slower fused than unfused
// (docs/tasks/task-int4-weight-quality-2026-09.md, lever 3), and the sizes between are unmeasured and stay unfused.
const fusedG32MaxHidden = 1536

// fusedG32Shmem is fused_rms_qkv_g32 / fused_rms_gu_g32's shared memory for hidden size H:
// normed[H] | red[256] | aq[H/4] int32 | aS[2·H/32].
func fusedG32Shmem(H int) uint32 { return uint32((H + 256 + H/4 + 2*(H/32)) * 4) }

// g4DropVNormForTest re-drops Gemma 4's v_norm on the layers that are not K=V, so a gate can show the fix is what
// moved the parity numbers (docs/tasks/task-multimodal-support-2026-10.md). Set only by a test seam; never in production.
var g4DropVNormForTest bool

// pleLayerShiftForTest shifts every PLE input slice by this many layers (a planted off-by-one per-layer offset). 0 in production; set only by a test seam.
var pleLayerShiftForTest int

// Test seams that each re-plant one defect (docs/tasks/task-multimodal-support-2026-10.md), so a gate can show it
// goes red. All false in production; only tests set them.
var (
	g4SkipPLEForTest           bool // (1) the PLE branch skipped
	g4KVSrcOffForTest          bool // (3) a shared layer's source one owning layer of its type off
	g4OneFFNWidthForTest       bool // (4) every dense layer takes layer 0's FFN width
	g4DropLayerScalarForTest   bool // (6) the dense layer scalar dropped
	g4KeepSharedKVStoreForTest bool // (5) a shared layer's K/V store NOT skipped (rope_kv keeps the real nKV)
)

// trimUnpinnedCtx is the positions an UNPINNED resident context gives up so that one KV slot (need bytes at ctxCap positions, linear in positions) fits
// beside the weights: free VRAM less extra (a companion attach) and margin. ok is false when it already fits, or when the fit would fall below
// cudaCtxCapDefault (then the caller declines as before).
func trimUnpinnedCtx(ctxCap int, need, free, extra, margin int64) (c int, ok bool) {
	if ctxCap <= 0 || need+extra+margin <= free {
		return ctxCap, false
	}
	perPos := need / int64(ctxCap)
	if perPos <= 0 {
		return ctxCap, false
	}
	c = int((free - extra - margin) / perPos)
	if c >= cudaCtxCapDefault && c < ctxCap {
		return c, true
	}
	return ctxCap, false
}
