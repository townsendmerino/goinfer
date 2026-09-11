//go:build darwin

package metal

import (
	"fmt"
	"unsafe"

	"github.com/townsendmerino/goinfer/decoder"
)

// Compute-time LoRA on the resident path (G3, docs/task-gpu-paths-2026-09.md).
//
// A LoRA adapter's delta is additive: y[o] += scale·Σ_r B[o,r]·(A·x)[r], applied AFTER the base
// projection's matmul, on the SAME input x the base matmul consumed (decoder/lora.go's
// applyLoRA, the CPU reference this mirrors). On Metal that input is always the already-quantized
// int8 activation buffer feeding the base GEMV (r.aq/r.aSc for q/k/v, r.cq/r.cSc for o, r.mq/r.mSc
// for gate/up, r.dq/r.dSc for down) — reusing it means no extra dequantize-and-requantize pass,
// and it is what the base weight itself sees, so the two projections agree on precision.
//
// Scope: Model.LoadAdapter (decoder/lora.go) only builds a runtime for the generic dense forward
// (rejects MoE, own-forward archs, and the non-gated MLP layout), so encodeAttention/encodeLayer's
// plain q/k/v/o/gate/up/down sites are the complete surface this needs to cover — MoE/DeltaNet/
// GPT-2 branches never see a non-nil loraLayers because no adapter can be loaded for them.
//
// NOT covered yet: the qGate branch (Qwen3.5-style fused double-width q_proj+gate) in
// encodeAttention — LoadAdapter does not exclude that family by name, only by the three checks
// above, so an adapter loaded against a qGate family would silently apply no q/k/v delta on Metal.
// No family with qGate=true is known to pass LoadAdapter's checks today; if one arrives, this is
// the gap to close first.

// loraRMax bounds the LoRA rank this backend accepts — generous against real adapters (typically
// rank 4-64, rarely above 128) and small enough that lora_delta's fixed-size threadgroup t[256]
// scratch (kernels.go) always fits it. SetAdapter declines (does not silently truncate) a
// projection whose rank exceeds it.
const loraRMax = 256

// residLoRAProj is one projection's bound delta: the device-resident A[R,In]/B[Out,R] matrices
// (row-major f32, PEFT's own layout) plus the uniforms the fused lora_delta kernel needs (P-11,
// audit-2026-09-10). uOut is a uniform buffer, not a plain Go dispatch-size int, because the
// fused kernel's up stage strides ITS OWN threads over Out rows internally — it needs the value
// INSIDE the kernel now, not just to size an external grid.
type residLoRAProj struct {
	a, b         Buffer // A[R,In], B[Out,R]
	uK, uR, uOut Buffer // uint32 uniforms: K=In, R, Out
	uScale       Buffer // float32 uniform: this projection's LoRA scale (alpha/rank)
}

// residLoRALayer is one transformer layer's per-projection bound deltas; a nil field is a
// projection this adapter does not target, matching decoder.ResidentAdapterLayer's own
// nil-means-untargeted convention exactly.
type residLoRALayer struct {
	q, k, v, o, gate, up, down *residLoRAProj
}

// releaseLoRALayers frees every device buffer a previously-bound adapter allocated. Buffers
// created outside BuildResident are NOT tracked for automatic release until Close (see
// prefill.go's C5 fix for the same class of leak) — SetAdapter must release the PREVIOUS bind's
// buffers itself before installing a new one, or clearing to none.
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

// loraProjIdentical reports whether two ResidentAdapterProj values describe the SAME uploaded
// delta — compared by the A/B slices' DATA POINTERS, not their contents. residentAdapterLayers
// (decoder/residency.go) never copies a loraDelta's A/B — it wraps the loraRuntime's own,
// load-time-allocated-and-never-mutated backing arrays in a fresh []ResidentAdapterLayer on
// every call, so the same adapter produces the same data pointers on every SetAdapter call for
// as long as it stays loaded. Two nils match (both untargeted); one nil and one non-nil do not.
func loraProjIdentical(a, b *decoder.ResidentAdapterProj) bool {
	if a == nil || b == nil {
		return a == b
	}
	return unsafe.SliceData(a.A) == unsafe.SliceData(b.A) && unsafe.SliceData(a.B) == unsafe.SliceData(b.B)
}

// loraLayersIdentical reports whether layers is the SAME adapter bind as cached (P-10's cache
// key) — same length, and every projection in every layer pointer-identical per
// loraProjIdentical. A nil or length-mismatched cached side never matches.
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

// SetAdapter implements decoder.ResidentAdapter: uploads (layers != nil) or clears (layers ==
// nil) the compute-time LoRA delta for every subsequent Forward/ForwardN call, until the next
// SetAdapter. A bind that returns an error leaves NO adapter bound (r.loraLayers is nil), which
// is what generateInto's fallback-to-CPU path assumes (it never calls Forward on a half-bound
// resident).
//
// C-07: stopExec FIRST, before touching r.loraLayers at all — the pipelined executor
// (ForwardEmbPipe/execLoop) may already hold a pre-encoded "next" command buffer that baked in
// the OLD r.loraLayers at encode time. Left running, that buffer would be committed under the
// NEW adapter state on the very next Forward call: one token's K/V computed with the wrong (or a
// just-released) delta, corrupting every later token that attends to it. stopExec drains and
// tears the executor down; ForwardEmbPipe re-arms it lazily on its next call, encoding fresh
// under whatever r.loraLayers this call leaves behind. Safe with no extra lock: SetAdapter and
// ForwardEmbPipe are never concurrent (the resBusy winner's own sequential SetAdapter → Forward*
// → SetAdapter(nil)).
//
// P-10: layers == nil (the end-of-generation clear) does NOT release r.loraCached — it only nils
// r.loraLayers, so applyResidentLoRA's dispatch sites no-op. The device buffers stay resident,
// keyed against r.loraCacheSrc, for a future SAME-adapter rebind (the dominant real shape: one
// chat session, many turns, bind→clear→bind→clear on the identical adapter every turn) to reuse
// with NO re-upload and NO device allocation. A bind of a DIFFERENT adapter evicts the cache
// (releases the old buffers) exactly as before P-10 — this never holds more than one adapter's
// worth of device memory at a time.
func (r *resident) SetAdapter(layers []decoder.ResidentAdapterLayer) error {
	r.stopExec()
	if layers == nil {
		r.loraLayers = nil
		return nil
	}
	if len(layers) != r.nL {
		return fmt.Errorf("metal: SetAdapter got %d layers, model has %d", len(layers), r.nL)
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
		return &residLoRAProj{
			a: NewBufferFloats(r.d, p.A), b: NewBufferFloats(r.d, p.B),
			uK: NewBufferU32(r.d, uint32(p.In)), uR: NewBufferU32(r.d, uint32(p.R)),
			uOut:   NewBufferU32(r.d, uint32(p.Out)),
			uScale: NewBufferFloats(r.d, []float32{p.Scale}),
		}, nil
	}
	out := make([]residLoRALayer, len(layers))
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
	releaseLoRALayers(r.d, r.loraCached) // evict the previous (different) adapter's device buffers
	r.loraLayers = out
	r.loraCached = out
	r.loraCacheSrc = layers
	return nil
}

// applyResidentLoRA dispatches one projection's compute-time LoRA delta into out, ADDITIVELY —
// no-op if p is nil (an untargeted projection, or no adapter bound at all). aq/aSc must be the
// SAME quantized activation the base projection this delta rides alongside already consumed
// (see the file comment). out must already hold the base projection's result — the delta is
// added on top, matching applyLoRA's "matmul, then add" order exactly.
//
// ONE dispatch, not two (P-11, audit-2026-09-10) — lora_delta (kernels.go) fuses the down and up
// GEMVs into a single kernel over one threadgroup, removing a whole dispatch's launch overhead
// per targeted projection.
func (r *resident) applyResidentLoRA(e *Encoder, p *residLoRAProj, aq, aSc, out Buffer) {
	if p == nil {
		return
	}
	e.Dispatch(r.pLoraDelta, tgReduceNorm, tgReduceNorm, aq, aSc, p.a, p.b, out, p.uK, p.uR, p.uOut, p.uScale)
}
