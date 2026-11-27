package cpuengine

import (
	"sync"

	"github.com/townsendmerino/aikit/linalg"
)

// Int4ParThreshold sets the fan-out threshold for int4 (W4A8 and Q4_K) matmuls below aikit's default.
const Int4ParThreshold = 1 << 19

// MatmulW8A8Batch W4A8BatchEnabled is set by decoder at init/sync time (mirrored from GOINFER_W4A8_BATCH).
// Keeping hooks on Engine/Hooks avoids any package-init env read in cpuengine.
func MatmulW8A8Batch(be Backend, ws *linalg.Workspace, a []float32, M, K int, ops []linalg.BatchOp, actGroup int) {
	if qb, ok := be.(QuantBatchBackend); ok && actGroup == 0 && qb.MatmulW8A8Batch(a, M, K, ops) {
		return
	}
	ws.SetActQuantGroup(actGroup)
	linalg.MatmulBTBatchInto(ws, a, M, ops)
}

func MatmulW4A8Batch(be Backend, ws *linalg.Workspace, a []float32, M, K, group int, ops []linalg.BatchOp, actGroup int) {
	if activeHooks().Int4FloatAct {
		Int4FloatActMissed.Add(1)
	}
	if qb, ok := be.(QuantBatchBackend4); ok && actGroup == 0 && qb.MatmulW4A8Batch(a, M, K, group, ops) {
		return
	}
	ws.SetActQuantGroup(actGroup)
	linalg.MatmulBTBatchInto(ws, a, M, ops)
}

// IsW8A8 reports whether w uses the int8×int8 (W8A8) path — the only one with a
// zero-alloc Workspace + batched-dispatch kernel.
func IsW8A8(w *linalg.WeightMat) bool {
	_, _, w8a8, ok := w.Int8()
	return ok && w8a8
}

// IsW4A8 reports whether w is int4-resident, the only precision with a batched dispatch on the W4A8 path. It uses
// IsInt4(), not Int4()'s narrower "canonical bytes present": a repacked-only WeightMat is still int4 and routes here, and
// WMW4A8Op builds the matching op shape for it.
func IsW4A8(w *linalg.WeightMat) bool {
	return w.IsInt4()
}

// WMW4A8Op builds one linalg.BatchOp for the batched W4A8 dispatch and reports w's weight group, which a GPU batch
// backend needs and which every op of one call shares (the same layer's quant config). linalg.MatmulBTBatchInto reads
// the layouts from the WeightMat: canonical, with row4 when RepackInt4Row4 has run (arm64, heap-backed weights; see
// repackW4A8Row4IfEligible), or the repacked layout alone for a repacked-only tensor.
func WMW4A8Op(w *linalg.WeightMat, dst []float32) (op linalg.BatchOp, group int) {
	_, _, group, _ = w.Int4F16()
	return linalg.BatchOp{W: w, Dst: dst}, group
}

// matmulWSPool recycles the Workspace Matmul falls back to when the caller has no decodeScratch. A fresh Workspace per
// call would reallocate its quant buffers on every Matmul. sync.Pool's GC-driven eviction keeps a workspace that saw one
// huge call (the vocab-sized LM head) from pinning that size forever.
var matmulWSPool = sync.Pool{New: func() any { return new(linalg.Workspace) }}

// MatmulWorkspace is own when the caller holds a Workspace, else one from matmulWSPool, which the caller returns when pooled.
func MatmulWorkspace(own *linalg.Workspace) (ws *linalg.Workspace, pooled bool) {
	if own != nil {
		return own, false
	}
	return matmulWSPool.Get().(*linalg.Workspace), true
}

// W4A8Act is one activation quantized once for several CPU W4A8 matmuls over it: q, k and v of one normed row, gate and
// up, every expert of a MoE layer. Quantization is deterministic, so repeating it per call was dead work. prepare
// quantizes exactly as Matmul's CPU W4A8 path would, and MatmulPre runs aikit's Pre entry on it, so the result is the bits
// Matmul gives. A weight the block does not fit (another K, another activation group, a backend that may take the call)
// runs Matmul instead.
type W4A8Act struct {
	q        linalg.ActQ
	set      bool
	group, M int
}

// CPUW4A8 reports whether Matmul runs w on the CPU W4A8 path for certain: an int4 weight no QuantBackend4 can claim
// (Matmul offers a per-row-scale int4 Matmul to one first).
func CPUW4A8(be Backend, w *linalg.WeightMat) bool {
	if !w.IsInt4() {
		return false
	}
	if _, ok := be.(QuantBackend4); ok && w.ActQuantGroup() == 0 {
		return false
	}
	return true
}

// Prepare quantizes a (M rows of w.Cols()) for w and every weight sharing its K and activation group, or clears the
// block when w does not run the CPU W4A8 path.
func (p *W4A8Act) Prepare(be Backend, w *linalg.WeightMat, a []float32, M int) {
	p.prepareWS(be, nil, w, a, M)
}

// prepareWS is prepare through own, a caller-held Workspace, or a pooled one when own is nil.
func (p *W4A8Act) prepareWS(be Backend, own *linalg.Workspace, w *linalg.WeightMat, a []float32, M int) {
	p.set = false
	if activeHooks().W4A8PreOff || !CPUW4A8(be, w) {
		return
	}
	ws, pooled := MatmulWorkspace(own)
	if pooled {
		defer matmulWSPool.Put(ws)
	}
	ws.SetActQuantGroup(w.ActQuantGroup())
	w.QuantizeActW4A8(ws, a, M, &p.q)
	p.set, p.group, p.M = true, w.ActQuantGroup(), M
}

// MatmulPre is Matmul(be, w, a, dst, M), using p's block when it was quantized for an input w can take.
func MatmulPre(be Backend, w *linalg.WeightMat, p *W4A8Act, a, dst []float32, M int) {
	MatmulPreWS(be, nil, w, p, a, dst, M)
}

// MatmulPreWS is MatmulPre through own, a caller-held Workspace (a batched pass's), or a pooled one when own is nil.
func MatmulPreWS(be Backend, own *linalg.Workspace, w *linalg.WeightMat, p *W4A8Act, a, dst []float32, M int) {
	if activeHooks().Int4FloatAct && w.IsInt4() {
		matmulInt4Float(w, a, dst, M)
		return
	}
	if p == nil || !p.set || p.M != M || p.q.K != w.Cols() || p.group != w.ActQuantGroup() || !CPUW4A8(be, w) {
		MatmulWS(be, own, w, a, dst, M)
		return
	}
	ws, pooled := MatmulWorkspace(own)
	if pooled {
		defer matmulWSPool.Put(ws)
	}
	ws.SetThreshold(Int4ParThreshold)
	ws.SetActQuantGroup(w.ActQuantGroup())
	w.MatmulBTW4A8PreInto(ws, &p.q, dst, M)
	if c := activeHooks().W4A8PreCalls; c != nil {
		c.Add(1)
	}
}

// Matmul computes dst[M, rows] = a[M, cols] . w^T, dispatching on w's precision with goinfer's backend routing: the f32,
// W8A8 and W4A8 paths can run on a GPU backend (be.MatmulBT, QuantBackend.MatmulW8A8, QuantBackend4.MatmulW4A8);
// weight-only int8 (Q8) stays on the CPU.
func Matmul(be Backend, w *linalg.WeightMat, a, dst []float32, M int) {
	MatmulWS(be, nil, w, a, dst, M)
}

// MatmulWS is Matmul through own, a caller-held Workspace (a batched pass's, which a chunked prefill keeps across its
// chunks), or a pooled one when own is nil.
func MatmulWS(be Backend, own *linalg.Workspace, w *linalg.WeightMat, a, dst []float32, M int) {
	if activeHooks().Int4FloatAct && w.IsInt4() {
		matmulInt4Float(w, a, dst, M)
		return
	}
	if w.IsInt4() {
		// Staged int4 backend consult, mirrored in MatmulInto and the W8A8 branch below. Nested under Int4()'s ok (canonical
		// bytes present), not IsInt4(): the consult hands q4/q4s to the backend on every call, so a repacked-only tensor has
		// nothing to hand it, and rebuilding canonical per token would allocate on a per-call path. It falls through to
		// w.MatmulBTW4A8Into, which dispatches on whichever layout is present. A per-group activation weight (ActQuantGroup)
		// skips the consult: no staged GPU kernel reads per-group activation scales.
		if q4, q4s, group, ok := w.Int4F16(); ok && w.ActQuantGroup() == 0 {
			if qb, ok := be.(QuantBackend4); ok && qb.MatmulW4A8(a, q4, q4s, group, dst, M, w.Cols(), w.Rows()) {
				return
			}
		}
		// int4 weights run the int8-activation W4A8 integer kernel at every M this CPU path reaches. Its per-output result is
		// M-independent, so batched prefill matches sequential decode on this kernel. That does not make the whole Matmul call
		// M-independent: the QuantBackend4 consult above takes M=1 on a staged webgpu backend (its MatmulW4A8 declines any
		// M != 1) through a different kernel, so decode and verify run two kernels there. Model.SpecDecodeConflict refuses
		// speculative decoding for that combination; read it before assuming this path covers a device-backed one.
		//
		// The pooled Workspace lowers the fan-out threshold (see Int4ParThreshold). Each Get is exclusive to this call, so
		// concurrent decode streams never share one.
		//
		// Use w.MatmulBTW4A8Into, not the raw linalg free function: aikit picks the repacked layout (arm64 row4, amd64
		// split-half) inside it, at M=1 only, and the load-time repacks do nothing without this call.
		ws, pooled := MatmulWorkspace(own)
		if pooled {
			defer matmulWSPool.Put(ws)
		}
		ws.SetThreshold(Int4ParThreshold)
		ws.SetActQuantGroup(w.ActQuantGroup()) // pooled: always set, so a previous model's group never leaks
		w.MatmulBTW4A8Into(ws, a, dst, M)
		return
	}
	if q8, scales, w8a8, ok := w.Int8(); ok {
		if w8a8 {
			if qb, ok := be.(QuantBackend); ok && w.ActQuantGroup() == 0 && qb.MatmulW8A8(a, q8, scales, dst, M, w.Cols(), w.Rows()) {
				return
			}
			// Pooled Workspace with the int8 decode threshold: the free-Matmul path (e.g. gemma4's own forward) has no scratch
			// Workspace, and without this its W8A8 decode matmuls would run at aikit's conservative default. MatmulInto gets the same
			// through decodeScratch. The threshold differs from int4's (see DefaultDecodeParallelThreshold and Int4ParThreshold).
			ws, pooled := MatmulWorkspace(own)
			if pooled {
				defer matmulWSPool.Put(ws)
			}
			ws.SetThreshold(DefaultDecodeParallelThreshold)
			ws.SetActQuantGroup(w.ActQuantGroup())
			w.MatmulBTInto(ws, a, dst, M)
			return
		}
		// Pooled Workspace, as in the W8A8 case: the bare linalg.MatmulBTQ8 wrapper builds a fresh, non-pooled Workspace every
		// call, and the weight-only Q8 LM head is the largest per-token cost on this path.
		ws, pooled := MatmulWorkspace(own)
		if pooled {
			defer matmulWSPool.Put(ws)
		}
		ws.SetThreshold(DefaultDecodeParallelThreshold)
		w.MatmulBTInto(ws, a, dst, M)
		return
	}
	if k := w.Kind(); k == "q4k" || k == "q5k" || k == "q6k" {
		// Native K-quants (quantQ4K) run on the CPU here: there is no staged-backend consult for them. Per-32 activations are
		// built into the kinds.
		ws, pooled := MatmulWorkspace(own)
		if pooled {
			defer matmulWSPool.Put(ws)
		}
		ws.SetThreshold(Int4ParThreshold)
		w.MatmulBTInto(ws, a, dst, M)
		return
	}
	if _, _, ok := w.Half16(); ok {
		// 16-bit weights (quantBF16, quantF16): the kind widens a tile of rows and runs the f32 kernel, so the result is the f32
		// Matmul's over the widened weights. No backend consult: no backend takes this kind.
		ws, pooled := MatmulWorkspace(own)
		if pooled {
			defer matmulWSPool.Put(ws)
		}
		ws.SetThreshold(DefaultDecodeParallelThreshold)
		w.MatmulBTInto(ws, a, dst, M)
		return
	}
	f32, _ := w.F32()
	be.MatmulBT(a, f32, dst, M, w.Cols(), w.Rows())
}

// MatmulIntoPre is MatmulInto(ws, be, w, a, dst, M), using p's block when it was quantized for an input w can take (see
// W4A8Act).
func MatmulIntoPre(ws *linalg.Workspace, be Backend, w *linalg.WeightMat, p *W4A8Act, a, dst []float32, M int) {
	if activeHooks().Int4FloatAct && w.IsInt4() {
		matmulInt4Float(w, a, dst, M)
		return
	}
	if p == nil || !p.set || p.M != M || p.q.K != w.Cols() || p.group != w.ActQuantGroup() || !CPUW4A8(be, w) {
		MatmulInto(ws, be, w, a, dst, M)
		return
	}
	ws.SetActQuantGroup(w.ActQuantGroup())
	ws.SetThreshold(Int4ParThreshold)
	w.MatmulBTW4A8PreInto(ws, &p.q, dst, M)
	if c := activeHooks().W4A8PreCalls; c != nil {
		c.Add(1)
	}
}

// MatmulInto is Matmul using the caller's Workspace, so steady-state decode quantizes the activation once into reusable
// scratch instead of allocating per call. It dispatches on the question "does this weight have an Into form that takes a
// Workspace", not on one quantization's name (a dispatch that names one member creates sibling drift, see
// docs/parity-coverage-policy.md); adding a third such quantization needs a case here and nothing else.
//
// Race-freedom: ws is the per-stream Workspace on decodeScratch, and a cache is one generation stream, so the buffers are
// never shared concurrently (decoder/scratch.go). Matmul's per-call Workspace stays for callers with no scratch at all.
func MatmulInto(ws *linalg.Workspace, be Backend, w *linalg.WeightMat, a, dst []float32, M int) {
	if activeHooks().Int4FloatAct && w.IsInt4() {
		matmulInt4Float(w, a, dst, M)
		return
	}
	if IsW8A8(w) {
		q8, scales, _, _ := w.Int8()
		if qb, ok := be.(QuantBackend); ok && w.ActQuantGroup() == 0 && qb.MatmulW8A8(a, q8, scales, dst, M, w.Cols(), w.Rows()) {
			return
		}
		ws.SetActQuantGroup(w.ActQuantGroup())
		w.MatmulBTInto(ws, a, dst, M)
		return
	}
	if _, _, w8a8, ok := w.Int8(); ok && !w8a8 {
		// Weight-only Q8 (the int8-pinned LM head in int4 mode): thread the caller's own scratch Workspace rather than falling
		// through to Matmul's pool round-trip.
		ws.SetThreshold(DefaultDecodeParallelThreshold)
		w.MatmulBTInto(ws, a, dst, M)
		return
	}
	if w.IsInt4() {
		// Staged int4 backend consult, as in Matmul: nested under Int4()'s ok (canonical bytes present), not IsInt4(), because a
		// repacked-only tensor has no canonical bytes to hand a staged backend.
		if q4, q4s, group, ok := w.Int4F16(); ok && w.ActQuantGroup() == 0 {
			if qb, ok := be.(QuantBackend4); ok && qb.MatmulW4A8(a, q4, q4s, group, dst, M, w.Cols(), w.Rows()) {
				return
			}
		}
		ws.SetActQuantGroup(w.ActQuantGroup())
		// Same threshold as Matmul's fresh Workspace: reusing one only stops a per-projection, per-token allocation and does not
		// change how the work fans out. w.MatmulBTW4A8Into, not the raw free function (see Matmul).
		ws.SetThreshold(Int4ParThreshold)
		w.MatmulBTW4A8Into(ws, a, dst, M)
		return
	}
	Matmul(be, w, a, dst, M)
}
