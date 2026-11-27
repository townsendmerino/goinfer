package decoder

import (
	"os"
	"sync"
	"sync/atomic"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/internal/runtime/giwcodec"
	"github.com/townsendmerino/goinfer/internal/runtime/wbuild"
)

type quantMode = wbuild.QuantMode

const (
	quantNone    = wbuild.QuantNone
	quantInt8    = wbuild.QuantInt8
	quantInt4    = wbuild.QuantInt4
	quantInt8I8  = wbuild.QuantInt8I8
	quantInt4Mix = wbuild.QuantInt4Mix
	quantQ4K     = wbuild.QuantQ4K
	quantBF16    = wbuild.QuantBF16
	quantF16     = wbuild.QuantF16

	int4GroupSize = wbuild.Int4GroupSize
)

// int4ParThreshold sets the fan-out threshold for the int4 (W4A8 and Q4_K) matmul below aikit's default (parThreshold =
// 1<<24 MACs) so the small decode matmuls parallelize. aikit runs a call serial when M*N*K (M*totalN*K for a batch) is
// under it. 1<<19 lets the smallest projections of the sub-1B models fan out (qwen2.5 0.5B's q/k/v batch, 1,032,192 MACs,
// and its o projection, 802,816), while ops under 512K MACs stay serial; threshold 0, which fans out everything,
// over-parallelizes on the 8-core Ryzen. TestInt4ParThreshold_qwen05bShapesFanOut pins the 0.5B shapes.
//
// Output is byte-identical (aikit partitions output columns in 8-wide groups, the width-invariant contract), and it only
// widens fan-out, so prefill's already-parallel large-M matmuls are unaffected. Re-run BenchmarkInt4ParThresholdSweep if
// the core topology or aikit's kernel changes. Measurements on both rigs: docs/code-notes/decoder.md#int4ParThreshold.
const int4ParThreshold = 1 << 19

// w4a8SplitHalfRepacked, w4a8SplitHalfSkipped and w4a8SplitHalfBytes count what the repack did, at load only (one atomic
// add per weight tensor). They exist because the repack is otherwise silent: it returns a bool nobody reads, and a wiring
// that repacked nothing (wrong quant, wrong load path, a shape rule rejecting every tensor) would yield a benchmark that
// measures no difference and gets recorded as flat. An A/B against this repack must read these first and confirm the
// repacked count is non-zero.
var w4a8SplitHalfRepacked, w4a8SplitHalfSkipped, w4a8SplitHalfBytes atomic.Int64

// w4a8SplitHalfRepackEnabled is default-off by decision, not caution; GOINFER_W4A8_SPLITHALF=1 opts in. The repack is a
// second copy of every eligible tensor's nibbles (+0.5 byte per weight on top of an int4 tensor's 0.625, about +80% int4
// weight bytes), and canonical is never dropped (M>1 prefill and every non-AVX2 path read it). Its decode gain landed in
// the band the pre-registration named in advance as ambiguous, so it is parked with the code and wiring intact.
//
// Turning it on is defensible where decode latency outranks resident memory, on amd64 with AVX2 and no AVX-512 VNNI
// (aikit declines on VNNI hosts). It is not defensible as a default. Re-open the decision if the kernel gets faster or
// canonical can be dropped for a decode-only build. The A/B and the figures:
// docs/measurements/w4a8-splithalf-decode-ab-PREREGISTERED.md and docs/code-notes/decoder.md#w4a8SplitHalfRepackEnabled.
var w4a8SplitHalfRepackEnabled = os.Getenv("GOINFER_W4A8_SPLITHALF") != ""

// w4a8BatchEnabled runs a layer's q/k/v (and gate/up, where the fused gate+up does not take them) as one W4A8 fork/join
// instead of one per projection (aikit MatmulBTBatchInto). Bit-identical: every output column is the same dot product
// either way. The default is per architecture (w4a8BatchDefault, cpu_tuning_{arm64,other}.go); GOINFER_W4A8_BATCH=0
// opts out and =1 forces it on. Why the defaults differ: docs/code-notes/decoder.md#w4a8BatchEnabled.
var w4a8BatchEnabled = envBoolDefault("GOINFER_W4A8_BATCH", w4a8BatchDefault)

// row4RepackCallsForTest, when set, counts the tensors offered to the in-RAM row4 repack.
var row4RepackCallsForTest *atomic.Int64

// w4a8Row4RepackEnabled is a load-time-measurement toggle only: production never sets it, so it stays true in a real
// build. A test measuring the repack's load-time and resident-memory delta flips it off to get a baseline from the same
// load path.
var w4a8Row4RepackEnabled = true

// wantsCanonicalInt4 reports whether this load might need canonical int4 bytes (packed nibbles + scales) for a tensor,
// that is, whether the repacked-only layout is unsafe. Two consumers read them directly rather than through WeightMat's
// layout-agnostic methods:
//
//   - The staged per-token GPU consult (QuantBackend4.MatmulW4A8, QuantBatchBackend4.MatmulW4A8Batch in matmul,
//     matmulInto, matmulW4A8Batch) hands q4/q4s to the backend on every call. A repacked-only tensor has nothing to hand
//     it, and rebuilding canonical per token is the wrong shape for a per-call path.
//   - A resident GPU build (ResidencyBackend.BuildResident) reads w.Int4() directly to upload once at build time.
//
// backendName is Options.Backend as the caller wrote it, not be's resolved capabilities. Repacked-only is a promise that
// no GPU backend will ever touch this *Model, resident or staged, and a promise must be stated, never inferred from an
// omission: only the literal "cpu" states it. An empty name promises nothing, because a generic load may later be handed
// to any backend's resident-build machinery outside decoder.Load's dispatch (metal's tests do exactly this), and keying
// on be's interfaces alone sees the plain CPU backend, which implements none of them, and calls it safe.
//
// The interface assertions are a second guard for the "cpu" case itself: a "cpu" backend that implemented one of these
// would still decline repacked-only. Every other backendName returns true, so canonical(+row4) stays the default.
func wantsCanonicalInt4(backendName string, be Backend) bool {
	if backendName != "cpu" {
		return true
	}
	if _, ok := be.(QuantBackend4); ok {
		return true
	}
	if _, ok := be.(QuantBatchBackend4); ok {
		return true
	}
	_, ok := be.(ResidencyBackend)
	return ok
}

// GIWTarget names the single consumer a .giw bundle is built for.
type GIWTarget = giwcodec.Target

const (
	GIWTargetNone     = giwcodec.TargetNone
	GIWTargetCPUArm64 = giwcodec.TargetCPUArm64
	GIWTargetCPUAmd64 = giwcodec.TargetCPUAmd64
	GIWTargetMetal    = giwcodec.TargetMetal
	GIWTargetCUDA     = giwcodec.TargetCUDA
	GIWTargetWebGPU   = giwcodec.TargetWebGPU
)

// GIWTargetForBackend derives the GIWTarget for a caller naming backendName.
func GIWTargetForBackend(backendName string) GIWTarget {
	return giwcodec.TargetForBackend(backendName, linalg.PreferredInt4Layout(4, 32, 32) == linalg.LayoutRow4)
}

// ParseGIWTarget parses a -target flag value.
func ParseGIWTarget(s string) (GIWTarget, error) {
	return giwcodec.ParseTarget(s, linalg.PreferredInt4Layout(4, 32, 32) == linalg.LayoutRow4)
}

// isW8A8 reports whether w uses the int8×int8 (W8A8) path — the only one with a
// zero-alloc Workspace + batched-dispatch kernel.
func isW8A8(w *linalg.WeightMat) bool {
	_, _, w8a8, ok := w.Int8()
	return ok && w8a8
}

// isW4A8 reports whether w is int4-resident, the only precision with a batched dispatch on the W4A8 path. It uses
// IsInt4(), not Int4()'s narrower "canonical bytes present": a repacked-only WeightMat is still int4 and routes here, and
// wmW4A8Op builds the matching op shape for it.
func isW4A8(w *linalg.WeightMat) bool {
	return w.IsInt4()
}

// Int4F32 returns w's canonical int4 nibbles and its binary16 group scales widened, exactly, to f32 in a NEW
// slice — for load-time consumers written against f32 scales (GPU uploads, the .giw writer). ok and group are
// linalg.WeightMat.Int4F16's. It allocates, so it is not for a per-token path; decode reads Int4F16.
func Int4F32(w *linalg.WeightMat) (q4 []byte, scales []float32, group int, ok bool) {
	q4, s16, group, ok := w.Int4F16()
	if s16 != nil {
		scales = make([]float32, len(s16))
		linalg.F16ToF32Slice(scales, s16)
	}
	return q4, scales, group, ok
}

// wmW4A8Op builds one linalg.BatchOp for the batched W4A8 dispatch and reports w's weight group, which a GPU batch
// backend needs and which every op of one call shares (the same layer's quant config). linalg.MatmulBTBatchInto reads
// the layouts from the WeightMat: canonical, with row4 when RepackInt4Row4 has run (arm64, heap-backed weights; see
// repackW4A8Row4IfEligible), or the repacked layout alone for a repacked-only tensor.
func wmW4A8Op(w *linalg.WeightMat, dst []float32) (op linalg.BatchOp, group int) {
	_, _, group, _ = w.Int4F16()
	return linalg.BatchOp{W: w, Dst: dst}, group
}

// matmulWSPool recycles the Workspace matmul falls back to when the caller has no decodeScratch. A fresh Workspace per
// call would reallocate its quant buffers on every matmul. sync.Pool's GC-driven eviction keeps a workspace that saw one
// huge call (the vocab-sized LM head) from pinning that size forever.
var matmulWSPool = sync.Pool{New: func() any { return new(linalg.Workspace) }}

// matmulWorkspace is own when the caller holds a Workspace, else one from matmulWSPool, which the caller returns when pooled.
func matmulWorkspace(own *linalg.Workspace) (ws *linalg.Workspace, pooled bool) {
	if own != nil {
		return own, false
	}
	return matmulWSPool.Get().(*linalg.Workspace), true
}

// w4a8PreOff (tests only) makes prepare leave every block unset, so each matmul quantizes its own input, for the gate
// that compares the two paths and the A/B that times them. w4a8PreCalls counts the matmuls that ran on a shared block, so
// a test can see the path was taken.
var (
	w4a8PreOff   = false
	w4a8PreCalls atomic.Int64
)

// w4a8Act is one activation quantized once for several CPU W4A8 matmuls over it: q, k and v of one normed row, gate and
// up, every expert of a MoE layer. Quantization is deterministic, so repeating it per call was dead work. prepare
// quantizes exactly as matmul's CPU W4A8 path would, and matmulPre runs aikit's Pre entry on it, so the result is the bits
// matmul gives. A weight the block does not fit (another K, another activation group, a backend that may take the call)
// runs matmul instead.
type w4a8Act struct {
	q        linalg.ActQ
	set      bool
	group, M int
}

// cpuW4A8 reports whether matmul runs w on the CPU W4A8 path for certain: an int4 weight no QuantBackend4 can claim
// (matmul offers a per-row-scale int4 matmul to one first).
func cpuW4A8(be Backend, w *linalg.WeightMat) bool {
	if !w.IsInt4() {
		return false
	}
	if _, ok := be.(QuantBackend4); ok && w.ActQuantGroup() == 0 {
		return false
	}
	return true
}

// prepare quantizes a (M rows of w.Cols()) for w and every weight sharing its K and activation group, or clears the
// block when w does not run the CPU W4A8 path.
func (p *w4a8Act) prepare(be Backend, w *linalg.WeightMat, a []float32, M int) {
	p.prepareWS(be, nil, w, a, M)
}

// prepareWS is prepare through own, a caller-held Workspace, or a pooled one when own is nil.
func (p *w4a8Act) prepareWS(be Backend, own *linalg.Workspace, w *linalg.WeightMat, a []float32, M int) {
	p.set = false
	if w4a8PreOff || !cpuW4A8(be, w) {
		return
	}
	ws, pooled := matmulWorkspace(own)
	if pooled {
		defer matmulWSPool.Put(ws)
	}
	ws.SetActQuantGroup(w.ActQuantGroup())
	w.QuantizeActW4A8(ws, a, M, &p.q)
	p.set, p.group, p.M = true, w.ActQuantGroup(), M
}

// matmulPre is matmul(be, w, a, dst, M), using p's block when it was quantized for an input w can take.
func matmulPre(be Backend, w *linalg.WeightMat, p *w4a8Act, a, dst []float32, M int) {
	matmulPreWS(be, nil, w, p, a, dst, M)
}

// matmulPreWS is matmulPre through own, a caller-held Workspace (a batched pass's), or a pooled one when own is nil.
func matmulPreWS(be Backend, own *linalg.Workspace, w *linalg.WeightMat, p *w4a8Act, a, dst []float32, M int) {
	if int4FloatActForTest && w.IsInt4() {
		matmulInt4Float(w, a, dst, M)
		return
	}
	if p == nil || !p.set || p.M != M || p.q.K != w.Cols() || p.group != w.ActQuantGroup() || !cpuW4A8(be, w) {
		matmulWS(be, own, w, a, dst, M)
		return
	}
	ws, pooled := matmulWorkspace(own)
	if pooled {
		defer matmulWSPool.Put(ws)
	}
	ws.SetThreshold(int4ParThreshold)
	ws.SetActQuantGroup(w.ActQuantGroup())
	w.MatmulBTW4A8PreInto(ws, &p.q, dst, M)
	w4a8PreCalls.Add(1)
}

// matmul computes dst[M, rows] = a[M, cols] . w^T, dispatching on w's precision with goinfer's backend routing: the f32,
// W8A8 and W4A8 paths can run on a GPU backend (be.MatmulBT, QuantBackend.MatmulW8A8, QuantBackend4.MatmulW4A8);
// weight-only int8 (Q8) stays on the CPU.
func matmul(be Backend, w *linalg.WeightMat, a, dst []float32, M int) {
	matmulWS(be, nil, w, a, dst, M)
}

// matmulWS is matmul through own, a caller-held Workspace (a batched pass's, which a chunked prefill keeps across its
// chunks), or a pooled one when own is nil.
func matmulWS(be Backend, own *linalg.Workspace, w *linalg.WeightMat, a, dst []float32, M int) {
	if int4FloatActForTest && w.IsInt4() {
		matmulInt4Float(w, a, dst, M)
		return
	}
	if w.IsInt4() {
		// Staged int4 backend consult, mirrored in matmulInto and the W8A8 branch below. Nested under Int4()'s ok (canonical
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
		// M-independent, so batched prefill matches sequential decode on this kernel. That does not make the whole matmul call
		// M-independent: the QuantBackend4 consult above takes M=1 on a staged webgpu backend (its MatmulW4A8 declines any
		// M != 1) through a different kernel, so decode and verify run two kernels there. Model.SpecDecodeConflict refuses
		// speculative decoding for that combination; read it before assuming this path covers a device-backed one.
		//
		// The pooled Workspace lowers the fan-out threshold (see int4ParThreshold). Each Get is exclusive to this call, so
		// concurrent decode streams never share one.
		//
		// Use w.MatmulBTW4A8Into, not the raw linalg free function: aikit picks the repacked layout (arm64 row4, amd64
		// split-half) inside it, at M=1 only, and the load-time repacks do nothing without this call.
		ws, pooled := matmulWorkspace(own)
		if pooled {
			defer matmulWSPool.Put(ws)
		}
		ws.SetThreshold(int4ParThreshold)
		ws.SetActQuantGroup(w.ActQuantGroup()) // pooled: always set, so a previous model's group never leaks
		w.MatmulBTW4A8Into(ws, a, dst, M)
		return
	}
	if q8, scales, w8a8, ok := w.Int8(); ok {
		if w8a8 {
			if qb, ok := be.(QuantBackend); ok && w.ActQuantGroup() == 0 && qb.MatmulW8A8(a, q8, scales, dst, M, w.Cols(), w.Rows()) {
				return
			}
			// Pooled Workspace with the int8 decode threshold: the free-matmul path (e.g. gemma4's own forward) has no scratch
			// Workspace, and without this its W8A8 decode matmuls would run at aikit's conservative default. matmulInto gets the same
			// through decodeScratch. The threshold differs from int4's (see DefaultDecodeParallelThreshold and int4ParThreshold).
			ws, pooled := matmulWorkspace(own)
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
		ws, pooled := matmulWorkspace(own)
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
		ws, pooled := matmulWorkspace(own)
		if pooled {
			defer matmulWSPool.Put(ws)
		}
		ws.SetThreshold(int4ParThreshold)
		w.MatmulBTInto(ws, a, dst, M)
		return
	}
	if _, _, ok := w.Half16(); ok {
		// 16-bit weights (quantBF16, quantF16): the kind widens a tile of rows and runs the f32 kernel, so the result is the f32
		// matmul's over the widened weights. No backend consult: no backend takes this kind.
		ws, pooled := matmulWorkspace(own)
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

// matmulIntoPre is matmulInto(ws, be, w, a, dst, M), using p's block when it was quantized for an input w can take (see
// w4a8Act).
func matmulIntoPre(ws *linalg.Workspace, be Backend, w *linalg.WeightMat, p *w4a8Act, a, dst []float32, M int) {
	if int4FloatActForTest && w.IsInt4() {
		matmulInt4Float(w, a, dst, M)
		return
	}
	if p == nil || !p.set || p.M != M || p.q.K != w.Cols() || p.group != w.ActQuantGroup() || !cpuW4A8(be, w) {
		matmulInto(ws, be, w, a, dst, M)
		return
	}
	ws.SetActQuantGroup(w.ActQuantGroup())
	ws.SetThreshold(int4ParThreshold)
	w.MatmulBTW4A8PreInto(ws, &p.q, dst, M)
	w4a8PreCalls.Add(1)
}

// matmulInto is matmul using the caller's Workspace, so steady-state decode quantizes the activation once into reusable
// scratch instead of allocating per call. It dispatches on the question "does this weight have an Into form that takes a
// Workspace", not on one quantization's name (a dispatch that names one member creates sibling drift, see
// docs/parity-coverage-policy.md); adding a third such quantization needs a case here and nothing else.
//
// Race-freedom: ws is the per-stream Workspace on decodeScratch, and a cache is one generation stream, so the buffers are
// never shared concurrently (decoder/scratch.go). matmul's per-call Workspace stays for callers with no scratch at all.
func matmulInto(ws *linalg.Workspace, be Backend, w *linalg.WeightMat, a, dst []float32, M int) {
	if int4FloatActForTest && w.IsInt4() {
		matmulInt4Float(w, a, dst, M)
		return
	}
	if isW8A8(w) {
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
		// through to matmul's pool round-trip.
		ws.SetThreshold(DefaultDecodeParallelThreshold)
		w.MatmulBTInto(ws, a, dst, M)
		return
	}
	if w.IsInt4() {
		// Staged int4 backend consult, as in matmul: nested under Int4()'s ok (canonical bytes present), not IsInt4(), because a
		// repacked-only tensor has no canonical bytes to hand a staged backend.
		if q4, q4s, group, ok := w.Int4F16(); ok && w.ActQuantGroup() == 0 {
			if qb, ok := be.(QuantBackend4); ok && qb.MatmulW4A8(a, q4, q4s, group, dst, M, w.Cols(), w.Rows()) {
				return
			}
		}
		ws.SetActQuantGroup(w.ActQuantGroup())
		// Same threshold as matmul's fresh Workspace: reusing one only stops a per-projection, per-token allocation and does not
		// change how the work fans out. w.MatmulBTW4A8Into, not the raw free function (see matmul).
		ws.SetThreshold(int4ParThreshold)
		w.MatmulBTW4A8Into(ws, a, dst, M)
		return
	}
	matmul(be, w, a, dst, M)
}
