package decoder

import (
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

// webgpuLoRAScratchRows is the f32 rows a WebGPU build always leases for compute-time LoRA (kernels' loraRMax). The memory gate holds
// WebGPUDecodeScratch to the build's own scratch list, so a drift here is red there.
const webgpuLoRAScratchRows = 256

// WebGPUScratchTerm is one named part of WebGPU's decode scratch: the bytes the runner's op builders of that role lease at build.
type WebGPUScratchTerm struct {
	Role  string
	Bytes int64
}

// WebGPUDecodeScratch is what a WebGPU resident build of m leases on the scratch line for decode (the sum of the build's scratch list,
// kernels.DecodeRunner.ScratchList), as terms named by the op builder that leases them: each from the program's declared per-row
// activations, plus the per-dispatch parameter buffers the builders create. ok is false for a model whose layers are not all a
// plain attention sublayer and a gated dense FFN with one geometry (the runner's other mixers are not yet derived): no figure is
// given for it, rather than a guess. Reported, not charged: no admission decision reads it.
func (m *Model) WebGPUDecodeScratch() (terms []WebGPUScratchTerm, ok bool) {
	H, _, _, _, _, _, V := m.Dims()
	if m.w == nil || H <= 0 || V <= 0 {
		return nil, false
	}
	d := m.metalDecodeRows(H, V)
	view := m.BackendView()
	kv := m.KVDeliveredFor("webgpu")
	win := view.SlidingWindow() > 0
	type geom struct{ hd, nKV, half int }
	var g0 geom
	scales := map[float32]bool{}
	var gemv, gemvAdd, actQuant, attnLayer, quantB, rmsQuantB int64
	pk := func(k int) int64 { return int64((k + 31) &^ 31) }
	rmsQuantB = pk(H) + 20 // one call per sublayer norm and the final norm: the int8 row, its scale, its uniform
	nRms := int64(1)
	for l, ops := range m.Program().Layers {
		if !denseAttnLayer(ops) || d.routed[l] {
			return nil, false
		}
		rows := d.layers[l]
		if len(rows) != 6 || rows["q"] == 0 || rows["k"] == 0 || rows["v"] == 0 || rows["ctx"] == 0 || rows["gate"] == 0 || rows["up"] == 0 {
			return nil, false
		}
		lv := view.Layer(l)
		w := &m.w.Layers[l]
		rs := float32(lv.RopeMscale())
		if rs == 0 || lv.IsLocal() || lv.VFromK() || len(w.QBias) > 0 || len(w.OBias) > 0 || len(w.QNorm) > 0 {
			return nil, false
		}
		g := geom{lv.HeadDim(), lv.KVHeads(), len(lv.RopeInvFreq())}
		if l == 0 {
			g0 = g
		} else if g != g0 {
			return nil, false
		}
		scales[rs] = true
		nRms += 2
		gemv += 4*int64(rows["q"]+rows["k"]+rows["v"]+rows["gate"]+rows["up"]) + 5*16
		gemvAdd += 2 * 16
		actQuant += pk(rows["gate"]) + 20
		attnLayer += 4*int64(rows["ctx"]) + 16
		quantB += pk(rows["ctx"]) + 20
	}
	if len(scales) == 0 {
		return nil, false
	}
	gemv += 4*int64(V) + 16 // the head
	geomB := int64(16 + 32) // the v-store and attention uniforms
	if win {
		geomB += 32
	}
	if kv == mem.KVI8 {
		geomB += 32
	}
	nScale := int64(len(scales)) // each rope scale's uniforms: one fused (f32 KV), or q's and k's (f16 and int8 KV)
	terms = []WebGPUScratchTerm{
		{"build", 4 + 4*webgpuLoRAScratchRows},
		{"residual", 4 * int64(H)},
		{"logitsStaging", 4 * int64(V)},
		{"rmsQuant", nRms * rmsQuantB},
		{"quant", quantB},
		{"gemv", gemv},
		{"gemvAdd", gemvAdd},
		{"actQuantPad", actQuant},
		{"attnLayer", attnLayer},
		{"geomFor", geomB},
	}
	if kv == mem.KVF16 || kv == mem.KVI8 {
		terms = append(terms, WebGPUScratchTerm{"ropeQUniFor", 32 * nScale}, WebGPUScratchTerm{"ropeKUniFor", 32 * nScale})
	} else {
		terms = append(terms, WebGPUScratchTerm{"qkvFinUniFor", 32 * nScale})
	}
	return terms, true
}

// WebGPUDecodeScratchBytes is the sum of WebGPUDecodeScratch's terms; ok is false where it gives none.
func (m *Model) WebGPUDecodeScratchBytes() (int64, bool) {
	terms, ok := m.WebGPUDecodeScratch()
	var n int64
	for _, t := range terms {
		n += t.Bytes
	}
	return n, ok
}

// denseAttnLayer reports whether a layer's operations are exactly a plain attention sublayer and a dense FFN, each behind its own
// pre-norm, with nothing else (no QK-norm, output gate, post-norm or layer scale).
func denseAttnLayer(ops []op) bool {
	want := []program.OpKind{program.OpNorm, program.OpProjQKV, program.OpRope, program.OpAttention, program.OpProjO, program.OpResidual,
		program.OpNorm, program.OpFFN, program.OpResidual}
	if len(ops) != len(want) {
		return false
	}
	for i, o := range ops {
		if o.Kind != want[i] {
			return false
		}
	}
	return true
}
