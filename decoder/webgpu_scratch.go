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
	var gemv, gemvAdd, actQuant, attnLayer, quantB, rmsQuantB, qkNormB, rmsnormB, biasAddB int64
	pk := func(k int) int64 { return int64((k + 31) &^ 31) }
	rmsQuantB = pk(H) + 20 // one call per sublayer norm and the final norm: the int8 row, its scale, its uniform
	nRms := int64(1)
	for l, ops := range m.Program().Layers {
		shape, isDense := denseAttnLayer(ops)
		if !isDense || d.routed[l] {
			return nil, false
		}
		rows := d.layers[l]
		if len(rows) != 6 || rows["q"] == 0 || rows["k"] == 0 || rows["v"] == 0 || rows["ctx"] == 0 || rows["gate"] == 0 || rows["up"] == 0 {
			return nil, false
		}
		lv := view.Layer(l)
		w := &m.w.Layers[l]
		rs := float32(lv.RopeMscale())
		if rs == 0 || lv.IsLocal() || lv.VFromK() || len(w.QBias) > 0 || len(w.OBias) > 0 {
			return nil, false
		}
		// Each optional operation is held by the weights it runs on: the build creates its dispatches from the weights, not the program.
		if shape.qkNorm != (len(w.QNorm) > 0) || shape.postAttn != (len(w.PostAttnNorm) > 0) || shape.postMLP != (len(w.PostMLPNorm) > 0) {
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
		if shape.qkNorm {
			qkNormB += 2 * 16 // q's and k's
		}
		// A post-sublayer norm splits the fused output projection into a bare gemv (its own output row), a norm (its own row and
		// uniform) and a separate residual add (one uniform).
		for _, post := range []bool{shape.postAttn, shape.postMLP} {
			if post {
				gemvAdd -= 16
				gemv += 4*int64(H) + 16
				rmsnormB += 4*int64(H) + 16
				biasAddB += 16
			}
		}
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
		{"qkNorm", qkNormB},
		{"rmsnormF32", rmsnormB},
		{"biasAdd", biasAddB},
		{"geomFor", geomB},
	}
	if kv == mem.KVF16 || kv == mem.KVI8 {
		terms = append(terms, WebGPUScratchTerm{"ropeQUniFor", 32 * nScale}, WebGPUScratchTerm{"ropeKUniFor", 32 * nScale})
	} else {
		terms = append(terms, WebGPUScratchTerm{"qkvFinUniFor", 32 * nScale})
	}
	kept := terms[:0]
	for _, t := range terms {
		if t.Bytes != 0 { // a role the model's layers never run
			kept = append(kept, t)
		}
	}
	return kept, true
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

// denseShape is the optional operations a plain attention layer carries: a QK-norm after the projections, and a norm on each
// sublayer's output before its residual add.
type denseShape struct{ qkNorm, postAttn, postMLP bool }

// denseAttnLayer reports whether a layer's operations are a plain attention sublayer and a dense FFN, each behind its own pre-norm,
// with only the optional operations of denseShape and nothing else (no output gate, temperature or layer scale).
func denseAttnLayer(ops []op) (s denseShape, ok bool) {
	i := 0
	next := func(k program.OpKind) bool {
		if i < len(ops) && ops[i].Kind == k {
			i++
			return true
		}
		return false
	}
	opt := func(k program.OpKind) bool { return next(k) }
	if !next(program.OpNorm) || !next(program.OpProjQKV) {
		return s, false
	}
	s.qkNorm = opt(program.OpQKNorm)
	if !next(program.OpRope) || !next(program.OpAttention) || !next(program.OpProjO) {
		return s, false
	}
	s.postAttn = opt(program.OpNorm)
	if !next(program.OpResidual) || !next(program.OpNorm) || !next(program.OpFFN) {
		return s, false
	}
	s.postMLP = opt(program.OpNorm)
	return s, next(program.OpResidual) && i == len(ops)
}
