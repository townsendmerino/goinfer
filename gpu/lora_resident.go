//go:build gpu

package gpu

import (
	"fmt"

	"github.com/cogentcore/webgpu/wgpu"
	"github.com/townsendmerino/goinfer/decoder"
)

// loraRMax bounds the LoRA rank this backend accepts — generous against real adapters
// (typically rank 4-64, rarely above 128); r.loraT (decoderunner.go) is sized to it.
// SetAdapter declines (does not silently truncate) a projection whose rank exceeds it.
const loraRMax = 256

// loraRunProj is one projection's bound delta: the device-resident A[R,In]/B[Out,R] buffers
// (row-major f32, PEFT's own layout), their uniforms, and the two bind groups
// lora_delta_down/lora_delta_up dispatch against — built fresh per SetAdapter call since they
// bind the specific buffers a hook recorded at construction (h.aq/h.ascale/h.dst) together with
// this call's adapter data (rank/scale unknown until now).
type loraRunProj struct {
	a, b         *wgpu.Buffer
	uDown, uUp   *wgpu.Buffer
	bgDown, bgUp *wgpu.BindGroup
	outn         int // Go-side dispatch grid size for lora_delta_up (ceil(outn/64) workgroups)
}

// loraRunLayer is one transformer layer's per-projection bound deltas; a nil field is a
// projection this adapter does not target, matching decoder.ResidentAdapterLayer's own
// nil-means-untargeted convention exactly.
type loraRunLayer struct {
	q, k, v, o, gate, up, down *loraRunProj
}

func (l *loraRunLayer) fieldFor(kind loraProjKind) **loraRunProj {
	switch kind {
	case loraQ:
		return &l.q
	case loraK:
		return &l.k
	case loraV:
		return &l.v
	case loraO:
		return &l.o
	case loraGate:
		return &l.gate
	case loraUp:
		return &l.up
	default:
		return &l.down
	}
}

func projFor(kind loraProjKind, l *decoder.ResidentAdapterLayer) *decoder.ResidentAdapterProj {
	switch kind {
	case loraQ:
		return l.Q
	case loraK:
		return l.K
	case loraV:
		return l.V
	case loraO:
		return l.O
	case loraGate:
		return l.Gate
	case loraUp:
		return l.Up
	default:
		return l.Down
	}
}

// releaseLoraLayers frees every device buffer/bind group a previously-bound adapter allocated —
// buffers created outside newDecodeRunner are not tracked for automatic release (unlike the
// fixed plan's own buffers, released via r.keep at Close), so SetAdapter must release the
// PREVIOUS bind's resources itself before installing a new one or clearing to none (the same
// per-call-buffer-leak class metal/lora.go's own releaseLoRALayers exists to avoid).
func releaseLoraLayers(layers []loraRunLayer) {
	for _, l := range layers {
		for _, p := range []*loraRunProj{l.q, l.k, l.v, l.o, l.gate, l.up, l.down} {
			if p == nil {
				continue
			}
			p.a.Release()
			p.b.Release()
			p.uDown.Release()
			p.uUp.Release()
			p.bgDown.Release()
			p.bgUp.Release()
		}
	}
}

// SetAdapter implements decoder.ResidentAdapter: uploads (layers != nil) or clears (layers ==
// nil) the compute-time LoRA delta for every subsequent Run call, until the next SetAdapter.
// Always releases whatever was previously bound first — a bind that returns an error leaves NO
// adapter bound (r.loraLayers is nil, r.steps restored to r.baseSteps), matching
// generateInto's fallback-to-CPU assumption that it never calls Run on a half-bound resident.
//
// Architecture note (see lora.go's file comment): this backend's dispatch plan is a flat,
// Go-side step list re-walked fresh every Run — nothing is baked GPU-side — so binding an
// adapter is REBUILDING that list from the pristine r.baseSteps plus this call's LoRA steps
// spliced in at the hook points r.loraHooks recorded during construction, not a per-token
// decision the way Metal's re-encoded-every-token trunk can make with a plain `if`.
func (r *DecodeRunner) SetAdapter(layers []decoder.ResidentAdapterLayer) error {
	releaseLoraLayers(r.loraLayers)
	r.loraLayers = nil
	if layers == nil {
		r.steps = r.baseSteps
		return nil
	}
	if len(layers) != r.nLayers {
		return fmt.Errorf("gpu: SetAdapter got %d layers, model has %d", len(layers), r.nLayers)
	}
	hookByLayerKind := make(map[[2]int]*loraHook, len(r.loraHooks))
	for i := range r.loraHooks {
		h := &r.loraHooks[i]
		hookByLayerKind[[2]int{h.layer, int(h.kind)}] = h
	}
	mk := func(layer int, kind loraProjKind, proj *decoder.ResidentAdapterProj) (*loraRunProj, error) {
		if proj == nil {
			return nil, nil
		}
		if proj.R <= 0 || proj.R > loraRMax {
			return nil, fmt.Errorf("gpu: SetAdapter: LoRA rank %d out of range (want 1..%d)", proj.R, loraRMax)
		}
		h, ok := hookByLayerKind[[2]int{layer, int(kind)}]
		if !ok {
			return nil, fmt.Errorf("gpu: SetAdapter: no dispatch hook for layer %d projection %d "+
				"(arch shape mismatch — LoadAdapter should have rejected this model)", layer, kind)
		}
		aBuf, err := r.c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
			Label: "lora-A", Contents: wgpu.ToBytes(proj.A), Usage: wgpu.BufferUsageStorage})
		if err != nil {
			return nil, err
		}
		bBuf, err := r.c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
			Label: "lora-B", Contents: wgpu.ToBytes(proj.B), Usage: wgpu.BufferUsageStorage})
		if err != nil {
			aBuf.Release()
			return nil, err
		}
		uDown, err := r.c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
			Label: "lora-uDown", Contents: wgpu.ToBytes([]uint32{uint32(h.k), uint32(proj.R), 0, 0}),
			Usage: wgpu.BufferUsageUniform})
		if err != nil {
			aBuf.Release()
			bBuf.Release()
			return nil, err
		}
		uUp, err := r.c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
			Label: "lora-uUp", Contents: wgpu.ToBytes([]uint32{uint32(proj.R), uint32(proj.Out), f32bits(proj.Scale), 0}),
			Usage: wgpu.BufferUsageUniform})
		if err != nil {
			aBuf.Release()
			bBuf.Release()
			uDown.Release()
			return nil, err
		}
		bgDown, err := r.c.device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: r.c.loraDownLayout, Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: h.aq, Size: h.aq.GetSize()},
			{Binding: 1, Buffer: h.ascale, Size: h.ascale.GetSize()},
			{Binding: 2, Buffer: aBuf, Size: aBuf.GetSize()},
			{Binding: 3, Buffer: r.loraT, Size: r.loraT.GetSize()},
			{Binding: 4, Buffer: uDown, Size: uDown.GetSize()},
		}})
		if err != nil {
			aBuf.Release()
			bBuf.Release()
			uDown.Release()
			uUp.Release()
			return nil, err
		}
		bgUp, err := r.c.device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: r.c.loraUpLayout, Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: bBuf, Size: bBuf.GetSize()},
			{Binding: 1, Buffer: r.loraT, Size: r.loraT.GetSize()},
			{Binding: 2, Buffer: h.dst, Size: h.dst.GetSize()},
			{Binding: 3, Buffer: uUp, Size: uUp.GetSize()},
		}})
		if err != nil {
			aBuf.Release()
			bBuf.Release()
			uDown.Release()
			uUp.Release()
			bgDown.Release()
			return nil, err
		}
		return &loraRunProj{a: aBuf, b: bBuf, uDown: uDown, uUp: uUp, bgDown: bgDown, bgUp: bgUp, outn: proj.Out}, nil
	}
	built := make([]loraRunLayer, len(layers))
	for l := range layers {
		src := &layers[l]
		for _, kind := range []loraProjKind{loraQ, loraK, loraV, loraO, loraGate, loraUp, loraDown} {
			p, err := mk(l, kind, projFor(kind, src))
			if err != nil {
				return err
			}
			*built[l].fieldFor(kind) = p
		}
	}
	r.loraLayers = built
	r.rebuildSteps()
	return nil
}

// rebuildSteps derives r.steps from r.baseSteps (never mutated) plus r.loraHooks, splicing in
// each hook's down+up dispatch pair right after its recorded base step — or skipping it if the
// bound adapter doesn't target that (layer, projection), or no adapter is bound at all.
func (r *DecodeRunner) rebuildSteps() {
	if r.loraLayers == nil {
		r.steps = r.baseSteps
		return
	}
	out := make([]runStep, 0, len(r.baseSteps)+2*len(r.loraHooks))
	hi := 0
	for i, s := range r.baseSteps {
		out = append(out, s)
		for hi < len(r.loraHooks) && r.loraHooks[hi].afterIdx == i {
			h := &r.loraHooks[hi]
			hi++
			p := *r.loraLayers[h.layer].fieldFor(h.kind)
			if p == nil {
				continue
			}
			out = append(out,
				runStep{pl: r.c.loraDownPipeline, bg: p.bgDown, gx: 1, gy: 1}, // ONE workgroup only (no workgroup_id in the kernel)
				runStep{pl: r.c.loraUpPipeline, bg: p.bgUp, gx: uint32((p.outn + 63) / 64), gy: 1},
			)
		}
	}
	r.steps = out
}
