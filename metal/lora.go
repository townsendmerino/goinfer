//go:build darwin

package metal

import (
	"fmt"

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
// rank 4-64, rarely above 128) and small enough that loraT (below) costs nothing. SetAdapter
// declines (does not silently truncate) a projection whose rank exceeds it.
const loraRMax = 256

// residLoRAProj is one projection's bound delta: the device-resident A[R,In]/B[Out,R] matrices
// (row-major f32, PEFT's own layout) plus the small uniforms lora_delta_down/_up need. out is
// kept as a plain Go int (not a uniform buffer) — it only sizes lora_delta_up's dispatch grid,
// never read inside a kernel (an exact non-uniform grid needs no bounds check).
type residLoRAProj struct {
	a, b   Buffer // A[R,In], B[Out,R]
	uK, uR Buffer // uint32 uniforms: K=In (lora_delta_down's reduction width), R (both kernels)
	uScale Buffer // float32 uniform: this projection's LoRA scale (alpha/rank)
	out    int
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
			d.ReleaseBuf(p.uScale)
		}
	}
}

// SetAdapter implements decoder.ResidentAdapter: uploads (layers != nil) or clears (layers ==
// nil) the compute-time LoRA delta for every subsequent Forward/ForwardN call, until the next
// SetAdapter. Always releases whatever was previously bound first — a bind that returns an error
// leaves NO adapter bound (r.loraLayers is nil), which is what generateInto's fallback-to-CPU
// path assumes (it never calls Forward on a half-bound resident).
func (r *resident) SetAdapter(layers []decoder.ResidentAdapterLayer) error {
	releaseLoRALayers(r.d, r.loraLayers)
	r.loraLayers = nil
	if layers == nil {
		return nil
	}
	if len(layers) != r.nL {
		return fmt.Errorf("metal: SetAdapter got %d layers, model has %d", len(layers), r.nL)
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
			uScale: NewBufferFloats(r.d, []float32{p.Scale}),
			out:    p.Out,
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
	r.loraLayers = out
	return nil
}

// applyResidentLoRA dispatches one projection's compute-time LoRA delta into out, ADDITIVELY —
// no-op if p is nil (an untargeted projection, or no adapter bound at all). aq/aSc must be the
// SAME quantized activation the base projection this delta rides alongside already consumed
// (see the file comment). out must already hold the base projection's result — the delta is
// added on top, matching applyLoRA's "matmul, then add" order exactly.
func (r *resident) applyResidentLoRA(e *Encoder, p *residLoRAProj, aq, aSc, out Buffer) {
	if p == nil {
		return
	}
	e.Dispatch(r.pLoraDown, tgReduceNorm, tgReduceNorm, aq, aSc, p.a, r.loraT, p.uK, p.uR)
	tg := p.out
	if tg > 256 {
		tg = 256
	}
	e.Dispatch(r.pLoraUp, p.out, tg, p.b, r.loraT, out, p.uR, p.uScale)
}
