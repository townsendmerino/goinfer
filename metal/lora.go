//go:build darwin

package metal

import (
	"fmt"
	"unsafe"

	"github.com/townsendmerino/goinfer/decoder"
)

// Compute-time LoRA on the resident path (G3, docs/tasks/task-gpu-paths-2026-09.md).
//
// A LoRA delta is additive: y[o] += scale·Σ_r B[o,r]·(A·x)[r], applied AFTER the base projection's matmul on the SAME input
// x the base matmul consumed (decoder/lora.go's applyLoRA, the CPU reference this mirrors). On Metal that input is the
// already-quantized int8 activation buffer feeding the base GEMV (r.aq/r.aSc for q/k/v, r.cq/r.cSc for o, r.mq/r.mSc for
// gate/up, r.dq/r.dSc for down): no extra requantize pass, and the delta sees what the base weight sees.
//
// Scope: Model.LoadAdapter builds a runtime only for the generic dense forward (it rejects MoE, own-forward archs and the
// non-gated MLP layout), so encodeAttention/encodeLayer's plain q/k/v/o/gate/up/down sites are the whole surface; the MoE,
// DeltaNet and GPT-2 branches never see a non-nil loraLayers.
//
// Known gap: the qGate branch (Qwen3.5-style fused double-width q_proj+gate) in encodeAttention applies no q/k/v delta.
// LoadAdapter excludes that family only through the checks above, so a qGate family that passed them would silently get no
// delta on Metal; none does today (docs/code-notes/metal.md#loraQGateGap).

// loraRMax bounds the LoRA rank this backend accepts — generous against real adapters (typically
// rank 4-64, rarely above 128) and small enough that lora_delta's fixed-size threadgroup t[256]
// scratch (kernels.go) always fits it. SetAdapter declines (does not silently truncate) a
// projection whose rank exceeds it.
const loraRMax = 256

// residLoRAProj is one projection's bound delta: the device-resident A[R,In]/B[Out,R] matrices (row-major, PEFT's layout,
// converted f32→f16 at bind time: the delta feeds an int8-quantised activation, so f32 A/B precision is not load-bearing, and
// A is re-read whole by every threadgroup) plus the uniforms the fused lora_delta kernel needs. outN is Out as a plain Go
// int, beside the uOut uniform the kernel reads for its bounds check: the multi-threadgroup grid (each threadgroup owns a
// 256-row block) needs Out on the Go side to size it.
type residLoRAProj struct {
	a, b         Buffer // A[R,In], B[Out,R], both f16
	uK, uR, uOut Buffer // uint32 uniforms: K=In, R, Out
	uScale       Buffer // float32 uniform: this projection's LoRA scale (alpha/rank)
	outN         int    // Out, for grid sizing (applyResidentLoRA)
}

// residLoRALayer is one transformer layer's per-projection bound deltas; a nil field is a
// projection this adapter does not target, matching decoder.ResidentAdapterLayer's own
// nil-means-untargeted convention exactly.
type residLoRALayer struct {
	q, k, v, o, gate, up, down *residLoRAProj
}

// releaseLoRALayers frees every device buffer a previously-bound adapter allocated. Buffers created outside BuildResident
// are not tracked for release until Close, so SetAdapter must release the PREVIOUS bind's buffers itself before installing a
// new one or clearing to none.
func releaseLoRALayers(d *Device, layers []residLoRALayer) {
	for _, l := range layers {
		for _, p := range []*residLoRAProj{l.q, l.k, l.v, l.o, l.gate, l.up, l.down} {
			if p == nil {
				continue
			}
			d.ReleaseBuf(p.a)
			d.ReleaseBuf(p.b)
			d.ReleaseBuf(p.uK)
			d.ReleaseBuf(p.uR)
			d.ReleaseBuf(p.uOut)
			d.ReleaseBuf(p.uScale)
		}
	}
}

// loraProjIdentical reports whether two ResidentAdapterProj values describe the SAME uploaded delta, compared by the A/B
// slices' data pointers, not contents: residentAdapterLayers (decoder/residency.go) never copies a loraDelta's A/B, it wraps
// the loraRuntime's load-time, never-mutated arrays, so one loaded adapter yields the same pointers on every SetAdapter call.
// Two nils match (both untargeted); one nil and one non-nil do not.
func loraProjIdentical(a, b *decoder.ResidentAdapterProj) bool {
	if a == nil || b == nil {
		return a == b
	}
	return unsafe.SliceData(a.A) == unsafe.SliceData(b.A) && unsafe.SliceData(a.B) == unsafe.SliceData(b.B)
}

// loraLayersIdentical reports whether layers is the SAME adapter bind as cached (the cache key): same length, every
// projection pointer-identical per loraProjIdentical. A nil or length-mismatched cached side never matches.
func loraLayersIdentical(layers, cached []decoder.ResidentAdapterLayer) bool {
	if len(layers) != len(cached) || layers == nil {
		return false
	}
	for i := range layers {
		l, c := layers[i], cached[i]
		if !loraProjIdentical(l.Q, c.Q) || !loraProjIdentical(l.K, c.K) || !loraProjIdentical(l.V, c.V) ||
			!loraProjIdentical(l.O, c.O) || !loraProjIdentical(l.Gate, c.Gate) ||
			!loraProjIdentical(l.Up, c.Up) || !loraProjIdentical(l.Down, c.Down) {
			return false
		}
	}
	return true
}

// SetAdapter implements decoder.ResidentAdapter: uploads (layers != nil) or clears (layers == nil) the compute-time LoRA
// delta for every subsequent Forward/ForwardN call until the next SetAdapter. A bind that returns an error leaves NO adapter
// bound (r.loraLayers is nil), which is what generateInto's fallback-to-CPU path assumes.
//
// stopExec runs FIRST, before r.loraLayers is touched: the pipelined executor may hold a pre-encoded "next" command buffer
// that baked in the OLD r.loraLayers, and committing it under the new adapter state would compute one token's K/V with the
// wrong (or a just-released) delta and corrupt every later token that attends to it. ForwardEmbPipe re-arms the executor
// lazily. No extra lock: SetAdapter and ForwardEmbPipe are never concurrent.
//
// layers == nil (the end-of-generation clear) does NOT release r.loraCached: it only nils r.loraLayers so
// applyResidentLoRA's dispatch sites no-op. The device buffers stay, keyed on r.loraCacheSrc, so a rebind of the SAME
// adapter (the dominant shape: bind then clear every turn of one chat session) re-uploads and allocates nothing. Binding a
// DIFFERENT adapter releases the cached buffers, so at most one adapter's device memory is held.
func (r *resident) SetAdapter(layers []decoder.ResidentAdapterLayer) error {
	r.stopExec()
	if layers == nil {
		r.loraLayers = nil
		return nil
	}
	if len(layers) != r.nL {
		return fmt.Errorf("metal: SetAdapter got %d layers, model has %d", len(layers), r.nL)
	}
	if r.q4kLane { // the resident LoRA kernels read the int8 activations this lane never makes
		return fmt.Errorf("metal: a LoRA adapter is not supported on a --quant q4k model")
	}
	if loraLayersIdentical(layers, r.loraCacheSrc) {
		r.loraLayers = r.loraCached
		return nil
	}
	conv := func(p *decoder.ResidentAdapterProj) (*residLoRAProj, error) {
		if p == nil {
			return nil, nil
		}
		if p.R <= 0 || p.R > loraRMax {
			return nil, fmt.Errorf("metal: LoRA rank %d out of range (want 1..%d)", p.R, loraRMax)
		}
		aHalf, bHalf := make([]uint16, len(p.A)), make([]uint16, len(p.B))
		parallelF32ToF16(aHalf, p.A)
		parallelF32ToF16(bHalf, p.B)
		return &residLoRAProj{
			a: NewBufferU16s(r.d, aHalf), b: NewBufferU16s(r.d, bHalf),
			uK: NewBufferU32(r.d, uint32(p.In)), uR: NewBufferU32(r.d, uint32(p.R)),
			uOut:   NewBufferU32(r.d, uint32(p.Out)),
			uScale: NewBufferFloats(r.d, []float32{p.Scale}),
			outN:   p.Out,
		}, nil
	}
	out := make([]residLoRALayer, len(layers))
	// An error partway through (a bad rank on layer i, say) must not leak layers 0..i-1's already-converted device buffers,
	// which nothing would release: bound latches true only once every projection in every layer has converted cleanly, and the
	// deferred release on any earlier return undoes exactly the partial work this call allocated.
	bound := false
	defer func() {
		if !bound {
			releaseLoRALayers(r.d, out)
		}
	}()
	for i, l := range layers {
		var err error
		if out[i].q, err = conv(l.Q); err != nil {
			return err
		}
		if out[i].k, err = conv(l.K); err != nil {
			return err
		}
		if out[i].v, err = conv(l.V); err != nil {
			return err
		}
		if out[i].o, err = conv(l.O); err != nil {
			return err
		}
		if out[i].gate, err = conv(l.Gate); err != nil {
			return err
		}
		if out[i].up, err = conv(l.Up); err != nil {
			return err
		}
		if out[i].down, err = conv(l.Down); err != nil {
			return err
		}
	}
	bound = true
	releaseLoRALayers(r.d, r.loraCached) // evict the previous (different) adapter's device buffers
	r.loraLayers = out
	r.loraCached = out
	r.loraCacheSrc = layers
	return nil
}

// applyResidentLoRA dispatches one projection's compute-time LoRA delta into out, ADDITIVELY; no-op if p is nil (an
// untargeted projection, or no adapter bound). aq/aSc must be the SAME quantized activation the base projection already
// consumed (see the file comment). out must already hold the base projection's result: the delta is added on top, matching
// applyLoRA's "matmul, then add" order.
//
// ONE dispatch, not two: lora_delta (kernels.go) fuses the down and up GEMVs, saving a launch per targeted projection. The
// grid is ceil(Out/256) threadgroups; each owns a fixed 256-row block of the up stage and recomputes the down stage's t[R]
// itself, trading redundant compute for up-stage parallelism on wide projections.
func (r *resident) applyResidentLoRA(e *Encoder, p *residLoRAProj, aq, aSc, out Buffer) {
	if p == nil {
		return
	}
	total := (p.outN + tgReduceNorm - 1) / tgReduceNorm * tgReduceNorm
	e.Dispatch(r.pLoraDelta, total, tgReduceNorm, aq, aSc, p.a, p.b, out, p.uK, p.uR, p.uOut, p.uScale)
}
