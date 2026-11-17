package decoder

import (
	"fmt"

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
	terms, why := m.webgpuDecodeScratch()
	return terms, why == ""
}

// WebGPUDecodeScratchWhy is the first reason WebGPUDecodeScratch gives no figure for m, or "" where it gives one.
func (m *Model) WebGPUDecodeScratchWhy() string {
	_, why := m.webgpuDecodeScratch()
	return why
}

func (m *Model) webgpuDecodeScratch() (terms []WebGPUScratchTerm, why string) {
	H, _, _, _, _, _, V := m.Dims()
	if m.w == nil || H <= 0 || V <= 0 {
		return nil, "the model has no hidden or vocabulary width"
	}
	d := m.metalDecodeRows(H, V)
	view := m.BackendView()
	kv := m.KVDeliveredFor("webgpu")
	win := view.SlidingWindow() > 0
	type geom struct{ hd, nKV, half int }
	var g0 geom
	haveGeom := false // layer 0 may be a delta layer, which has no attention geometry
	scales := map[float32]bool{}
	var gemv, gemvAdd, actQuant, attnLayer, quantB, rmsQuantB, qkNormB, rmsnormB, biasAddB, gemvBiasB, moeFFN, moeRoute, moeExpert, swiglu, sharedCombine int64
	var moeNE int
	var deltaLayer, qSplit, attnGate int64
	anyDelta := false
	pk := func(k int) int64 { return int64((k + 31) &^ 31) }
	rmsQuantB = pk(H) + 20 // one call per sublayer norm and the final norm: the int8 row, its scale, its uniform
	nRms, nQuant := int64(1), int64(0)
	ffnTerms := func(l int, shape denseShape, rows map[string]int, nE, topK, shared int, hasShared bool) string {
		if shape.postMLP { // the output projection splits as the attention's does
			if shape.moe {
				return fmt.Sprintf("layer %d: a routed FFN with an output norm", l)
			}
			gemvAdd -= 16
			gemv += 4*int64(H) + 16
			rmsnormB += 4*int64(H) + 16
			biasAddB += 16
		}
		inter := int64(rows["gate"])
		if !shape.moe {
			gemv += 4*(inter+int64(rows["up"])) + 2*16
			gemvAdd += 16
			actQuant += pk(rows["gate"]) + 20
		} else {
			// The router's logits row; the selection rows and the experts' gate and up rows (reused over the top-k slots); the router's
			// uniform; per slot the expert projections' uniforms and the fused activation's int8 row, scale and uniform.
			gemv += 4*int64(nE) + 16
			moeFFN += 4*int64(2*topK) + 4*2*inter
			moeRoute += 32
			moeExpert += 3 * int64(topK) * 16
			swiglu += int64(topK) * (pk(rows["gate"]) + 20)
			moeNE = nE
			if hasShared { // the shared expert: a gated SwiGLU MLP, sigmoid-gated by a one-row projection or added ungated
				gemv += 4*2*int64(shared) + 2*16
				swiglu += pk(shared) + 20
				if m.w.Layers[l].SharedGate.Rows() > 0 {
					gemv += 4*int64(H) + 16 + 4 + 16
					sharedCombine += 16
				} else {
					gemvAdd += 16
				}
			}
		}
		return ""
	}
	for l, ops := range m.Program().Layers {
		shape, isDense := denseAttnLayer(ops)
		if !isDense {
			return nil, fmt.Sprintf("layer %d: operations %v are not a plain attention and dense FFN sublayer", l, opKinds(ops))
		}
		if d.routed[l] != shape.moe {
			return nil, fmt.Sprintf("layer %d: a routed-experts FFN of a form not derived", l)
		}
		rows := d.layers[l]
		mixer := []string{"q", "k", "v", "ctx"}
		switch {
		case shape.gated:
			mixer = append(mixer, "q-gate", "attn-gate")
		case shape.delta:
			mixer = []string{"dn-ba", "dn-beta", "dn-conv", "dn-core", "dn-decay", "dn-gated", "dn-k", "dn-mixed", "dn-q", "dn-z"}
		}
		want := len(mixer) + 2 // and the FFN's gate and up
		var nE, topK, shared int
		hasShared := shape.moe && m.w.Layers[l].SharedExpert.Gate.Rows() > 0
		if shape.moe {
			nE, topK, shared = rows["moe-router"], rows["moe-experts"], rows["shared-gate"]
			want += 3
			if hasShared {
				want += 2
			}
			if nE == 0 || topK == 0 || rows["moe-weights"] != topK || hasShared != (shared > 0) || rows["shared-up"] != shared {
				return nil, fmt.Sprintf("layer %d: the routed FFN's declared rows are not the generic router's", l)
			}
		}
		for _, k := range append(mixer, "gate", "up") {
			if rows[k] == 0 {
				return nil, fmt.Sprintf("layer %d: the declared row %q is missing", l, k)
			}
		}
		if len(rows) != want {
			return nil, fmt.Sprintf("layer %d: declared rows beyond the derived ones", l)
		}
		if shape.delta {
			anyDelta = true
			for _, pre := range [][]float32{m.w.Layers[l].PreAttnNorm, m.w.Layers[l].PreMLPNorm} {
				if len(pre) == 0 {
					nQuant++
				} else {
					nRms++
				}
			}
			dn := int64(rows["dn-k"])
			gemv += 4*int64(rows["dn-mixed"]+rows["dn-beta"]+rows["dn-decay"]+rows["dn-z"]) + 4*16
			deltaLayer += 4 * (int64(rows["dn-conv"]) + 2*dn + int64(rows["dn-ba"]) + int64(rows["dn-core"]) + int64(rows["dn-gated"]))
			quantB += pk(rows["dn-gated"]) + 20
			gemvAdd += 16
			if err := ffnTerms(l, shape, rows, nE, topK, shared, hasShared); err != "" {
				return nil, err
			}
			continue
		}
		lv := view.Layer(l)
		w := &m.w.Layers[l]
		rs := float32(lv.RopeMscale())
		switch {
		case rs == 0:
			return nil, fmt.Sprintf("layer %d: rope scale 0", l)
		case lv.VFromK():
			return nil, fmt.Sprintf("layer %d: V from K", l)
		case len(w.OBias) > 0:
			return nil, fmt.Sprintf("layer %d: an output bias", l)
		}
		// Each optional operation is held by the weights it runs on: the build creates its dispatches from the weights, not the program.
		if shape.gated && (len(w.QBias) > 0 || len(w.OBias) > 0 || len(w.PostAttnNorm) > 0 || len(w.PostMLPNorm) > 0) {
			return nil, fmt.Sprintf("layer %d: a bias or output norm on a gated attention layer", l)
		}
		hasQNorm := len(w.QNorm) > 0
		if shape.gated { // the gated attention's QK-norm is part of its one operation, and its weights live apart from the layer's own
			hasQNorm = gatedAttnQKNorm(lv)
			shape.qkNorm = hasQNorm
		}
		if shape.qkNorm != hasQNorm || shape.postAttn != (len(w.PostAttnNorm) > 0) || shape.postMLP != (len(w.PostMLPNorm) > 0) {
			return nil, fmt.Sprintf("layer %d: a norm or QK-norm the weights and the operations disagree on", l)
		}
		g := geom{lv.HeadDim(), lv.KVHeads(), len(lv.RopeInvFreq())}
		if !haveGeom {
			g0, haveGeom = g, true
		} else if g != g0 {
			return nil, fmt.Sprintf("layer %d: its geometry differs from the first attention layer's", l)
		}
		scales[rs] = true
		for _, pre := range [][]float32{w.PreAttnNorm, w.PreMLPNorm} {
			if len(pre) == 0 { // a family with no pre-norm (norms after the sublayer only) quantizes the raw residual
				nQuant++
			} else {
				nRms++
			}
		}
		gemv += 4*int64(rows["q"]+rows["k"]+rows["v"]) + 3*16
		gemvAdd += 16    // the output projection, fused with its residual add
		if shape.gated { // q_proj emits [query | gate] per head: its output row is the wide one, then a split and a gate applied to the context
			gemv += 4 * int64(rows["q-gate"]-rows["q"])
			qSplit += 4*2*int64(rows["q"]) + 16
			attnGate += 16
		}
		if shape.qkNorm {
			qkNormB += 2 * 16 // q's and k's
		}
		if len(w.QBias) > 0 { // q, k and v's bias: folded into an int8 weight's projection, else a separate add each
			if w.QProj.Kind() == "int4" {
				biasAddB += 3 * 16
			} else {
				q3 := 4*int64(rows["q"]+rows["k"]+rows["v"]) + 3*16
				gemv -= q3
				gemvBiasB += q3
			}
		}
		// A post-sublayer norm splits the fused output projection into a bare gemv (its own output row), a norm (its own row and
		// uniform) and a separate residual add (one uniform).
		if shape.postAttn {
			gemvAdd -= 16
			gemv += 4*int64(H) + 16
			rmsnormB += 4*int64(H) + 16
			biasAddB += 16
		}
		if err := ffnTerms(l, shape, rows, nE, topK, shared, hasShared); err != "" {
			return nil, err
		}
		attnLayer += 4*int64(rows["ctx"]) + 16
		quantB += pk(rows["ctx"]) + 20
	}
	if len(scales) == 0 {
		return nil, "the model has no layers"
	}
	gemv += 4*int64(V) + 16 // the head
	geomB := int64(16 + 32) // the v-store and attention uniforms
	if win {
		geomB += 32
	}
	if kv == mem.KVI8 {
		geomB += 32
	}
	deltaSetupB := int64(0)
	if anyDelta {
		deltaSetupB = 16 + 16 + 32 + 32 + 32 // the five dispatch parameter buffers the delta mixer shares across layers
	}
	nScale := int64(len(scales)) // each rope scale's uniforms: one fused (f32 KV), or q's and k's (f16 and int8 KV)
	terms = []WebGPUScratchTerm{
		{"build", 4 + 4*webgpuLoRAScratchRows},
		{"residual", 4 * int64(H)},
		{"logitsStaging", 4 * int64(V)},
		{"rmsQuant", nRms * rmsQuantB},
		{"quant", quantB + nQuant*(pk(H)+20)},
		{"gemv", gemv},
		{"gemvBias", gemvBiasB},
		{"gemvAdd", gemvAdd},
		{"moeFFN", moeFFN},
		{"moeRoute", moeRoute},
		{"moeExpert", moeExpert},
		{"swigluQuant", swiglu},
		{"sharedGatedCombine", sharedCombine},
		{"moeZeroBias", 4 * int64(moeNE)},
		{"deltaLayer", deltaLayer},
		{"qSplit", qSplit},
		{"attnGate", attnGate},
		{"deltaSetup", deltaSetupB},
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
	return kept, ""
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
type denseShape struct{ qkNorm, postAttn, postMLP, moe, gated, delta bool }

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
	if !next(program.OpNorm) {
		return s, false
	}
	switch {
	case next(program.OpDeltaNet):
		s.delta = true
	case next(program.OpGatedAttn):
		s.gated = true
	default:
		if !next(program.OpProjQKV) {
			return s, false
		}
		s.qkNorm = opt(program.OpQKNorm)
		// A layer with no rotary embedding still runs the rope dispatch (over a zeroed frequency table, an identity), and a temperature
		// operation scales the query in a dispatch of its own that the runner folds into the attention's uniform.
		opt(program.OpRope)
		opt(program.OpAttnTemp)
		opt(program.OpRope)
		if !next(program.OpAttention) || !next(program.OpProjO) {
			return s, false
		}
		s.postAttn = opt(program.OpNorm)
	}
	if !next(program.OpResidual) || !next(program.OpNorm) || !next(program.OpFFN) {
		return s, false
	}
	switch ops[i-1].FFN.Form {
	case program.FfnMoE:
		s.moe = true
	case program.FfnGated, program.FfnNonGated:
	default: // gpt-oss, Llama 4, Nemotron and Gemma 4's own routed forms are other builders
		return s, false
	}
	s.postMLP = opt(program.OpNorm)
	return s, next(program.OpResidual) && i == len(ops)
}

// opKinds is a layer's operation kinds in order, for the reason a layer has no figure.
func opKinds(ops []op) []program.OpKind {
	ks := make([]program.OpKind, len(ops))
	for i, o := range ops {
		ks[i] = o.Kind
	}
	return ks
}

// gatedAttnQKNorm reports whether a gated attention layer carries a QK-norm weight, which it keeps with its projections, not in the
// layer's own norm fields. Like denseAttnLayer it reads the layer shape's own accessor, so the pricing body names no family.
func gatedAttnQKNorm(lv ResidentLayerView) bool {
	_, _, _, _, qn, _ := lv.Qwen35AttnWeights()
	return len(qn) > 0
}
