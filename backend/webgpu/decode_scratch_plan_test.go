//go:build gpu && goinfer_testhooks

package webgpu

import (
	"fmt"
	"strings"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

// webgpuLoRAScratchRows is the f32 rows a WebGPU build always leases for compute-time LoRA (kernels' loraRMax). The memory gate holds
// webgpuDecodeScratchTerms to the build's own scratch list, so a drift here is red there.
const webgpuLoRAScratchRows = 256

// webgpuScratchTerm is one named part of WebGPU's decode scratch: the bytes the runner's op builders of that role lease at build.
type webgpuScratchTerm struct {
	Role  string
	Bytes int64
}

// webgpuDecodeScratchTerms is what a WebGPU resident build of m leases on the scratch line for decode (the sum of the build's scratch list,
// kernels.DecodeRunner.ScratchList), as terms named by the op builder that leases them: each from the program's declared per-row
// activations, plus the per-dispatch parameter buffers the builders create. ok is false for a model whose layers are not all a
// plain attention sublayer and a gated dense FFN with one geometry (the runner's other mixers are not yet derived): no figure is
// given for it, rather than a guess. Reported, not charged: no admission decision reads it.
func webgpuDecodeScratchTerms(m *decoder.Model) (terms []webgpuScratchTerm, ok bool) {
	terms, why := webgpuDecodeScratch(m)
	return terms, why == ""
}

// webgpuDecodeScratchWhy is the first reason webgpuDecodeScratchTerms gives no figure for m, or "" where it gives one.
func webgpuDecodeScratchWhy(m *decoder.Model) string {
	_, why := webgpuDecodeScratch(m)
	return why
}

func webgpuDecodeScratch(m *decoder.Model) (terms []webgpuScratchTerm, why string) {
	H, _, _, _, _, _, V := m.Dims()
	view := m.BackendView()
	wts := view.Weights()
	if wts == nil || H <= 0 || V <= 0 {
		return nil, "the model has no hidden or vocabulary width"
	}
	declared := m.DecodeRowsForTest()
	kv := m.KVDeliveredFor("webgpu")
	win := view.SlidingWindow() > 0
	type geom struct {
		hd, nKV, half int
		kEqV          bool
	}
	geoms := map[geom]map[float32]bool{} // each distinct attention geometry, and the rope scales its uniforms are built for
	var gemv, gemvAdd, actQuant, attnLayer, quantB, rmsQuantB, qkNormB, rmsnormB, biasAddB, gemvBiasB, moeFFN, moeRoute, moeExpert, swiglu, sharedCombine int64
	var moeNE int
	var deltaLayer, qSplit, attnGate, mlaLayer, mlaAbsorb, mlaLift, nRmsExtra int64
	anyDelta, anyLatent, anySSM := false, false, false
	var moeBlock, vNormB, scaleVecB, routedRows, mambaLayer, relu2, clampedRoute, clampedAct, clampedDown int64
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
			if shape.clamped { // the router's four words, the gate and up uniforms, the clamped activation's eight, the down's four
				clampedRoute += 16
				moeExpert += 2 * int64(topK) * 16
				clampedAct += int64(topK) * (pk(rows["gate"]) + 4 + 32)
				clampedDown += int64(topK) * 16
			} else {
				moeRoute += 32
				moeExpert += 3 * int64(topK) * 16
				swiglu += int64(topK) * (pk(rows["gate"]) + 20)
			}
			moeNE = nE
			if hasShared { // the shared expert: a gated SwiGLU MLP, sigmoid-gated by a one-row projection or added ungated
				gemv += 4*2*int64(shared) + 2*16
				swiglu += pk(shared) + 20
				if wts.Layers[l].SharedGate.Rows() > 0 {
					gemv += 4*int64(H) + 16 + 4 + 16
					sharedCombine += 16
				} else {
					gemvAdd += 16
				}
			}
		}
		return ""
	}
	// g4Terms is a parallel dense and routed block on the post-attention residual: the router's norm and projection, a dense branch and an
	// expert branch each behind its own pre-norm, the post-norms, the join and the layer's output scalar.
	g4Terms := func(l int) string {
		g, ok := parallelBlockOf(view.Layer(l))
		if !ok {
			return fmt.Sprintf("layer %d: no parallel routed block", l)
		}
		rows := declared[l]
		if rows["g4moe-router"] != g.nE || rows["g4moe-experts"] != g.topK || rows["gate"] != g.denseInter {
			return fmt.Sprintf("layer %d: the block's declared rows disagree with its weights", l)
		}
		nRms += 2 // the dense branch's and the expert branch's pre-norms
		moeNE = g.nE
		rmsnormB += 4 * (4*int64(H) + 16)
		moeBlock += 4*int64(g.nE) + 4*int64(2*g.topK) + 4*int64(H) + 4*3*int64(g.moeInter) + 16 + 32 + 16 + int64(g.topK)*16
		gemv += 4*int64(2*g.denseInter) + 2*16 + 4*int64(H) + 16
		actQuant += pk(g.denseInter) + 20 + int64(g.topK)*(pk(g.moeInter)+20)
		moeExpert += 2 * int64(g.topK) * 16
		scaleVecB += 16
		biasAddB += 2 * 16
		if sc := g.layerScalar; sc != 0 && sc != 1 {
			scaleVecB += 16
		}
		return ""
	}
	for l, ops := range m.Program().Layers {
		shape, isDense := denseAttnLayer(ops)
		if !isDense {
			return nil, fmt.Sprintf("layer %d: operations %v (FFN form %d) are not a plain attention and dense FFN sublayer", l, opKinds(ops), shape.otherForm)
		}
		if routesExperts(declared[l]) != (shape.moe || shape.block == blockRouted || shape.g4) {
			return nil, fmt.Sprintf("layer %d: a routed-experts FFN of a form not derived", l)
		}
		rows := declared[l]
		mixer := []string{"q", "k", "v", "ctx"}
		switch {
		case shape.gated:
			mixer = append(mixer, "q-gate", "attn-gate")
		case shape.g4:
			mixer = []string{"q", "k", "v", "ctx", "g4moe-dense-out", "g4moe-experts", "g4moe-moe-out", "g4moe-router", "g4moe-router-in", "g4moe-weights"}
		case shape.block == blockSSM:
			mixer = []string{"mb-conv", "mb-gated", "mb-proj", "mb-y"}
		case shape.block == blockMLP:
			mixer = []string{"up"}
		case shape.block == blockRouted:
			p := shape.routedPrefix
			mixer = []string{"up", "shared-up", p + "-router", p + "-experts", p + "-weights"}
		case shape.latent:
			mixer = []string{"mla-ctx", "mla-kv", "mla-q", "mla-q-abs", "mla-wsum"}
			if rows["mla-qa"] > 0 { // a low-rank query projection
				mixer = append(mixer, "mla-qa")
			}
		case shape.delta:
			mixer = []string{"dn-ba", "dn-beta", "dn-conv", "dn-core", "dn-decay", "dn-gated", "dn-k", "dn-mixed", "dn-q", "dn-z"}
		}
		want := len(mixer) + 2 // and the FFN's gate and up
		if shape.block != blockPair {
			want = len(mixer) // a layer of one operation has no FFN rows beside it
		}
		var nE, topK, shared int
		hasShared := shape.moe && wts.Layers[l].SharedExpert.Gate.Rows() > 0
		if shape.moe {
			pre := "moe" // the router's declared rows are named by the form
			if shape.clamped {
				pre = "gptoss"
			}
			nE, topK, shared = rows[pre+"-router"], rows[pre+"-experts"], rows["shared-gate"]
			want += 3
			if hasShared {
				want += 2
			}
			if nE == 0 || topK == 0 || rows[pre+"-weights"] != topK || hasShared != (shared > 0) || rows["shared-up"] != shared {
				return nil, fmt.Sprintf("layer %d: the routed FFN's declared rows are not the generic router's", l)
			}
		}
		need := append(mixer, "gate", "up")
		if shape.block != blockPair {
			need = mixer
		}
		for _, k := range need {
			if rows[k] == 0 {
				return nil, fmt.Sprintf("layer %d: the declared row %q is missing", l, k)
			}
		}
		if len(rows) != want {
			return nil, fmt.Sprintf("layer %d: declared rows beyond the derived ones", l)
		}
		if shape.block == blockRouted {
			p := shape.routedPrefix
			nE, topK, inter, sh := rows[p+"-router"], rows[p+"-experts"], rows["up"], rows["shared-up"]
			if nE == 0 || topK == 0 || rows[p+"-weights"] != topK || len(wts.Layers[l].PreAttnNorm) == 0 {
				return nil, fmt.Sprintf("layer %d: the routed block's declared rows or its norm are not the derived ones", l)
			}
			nRms++
			gemv += 4*int64(nE) + 16
			routedRows += 4*int64(2*topK) + 4*int64(topK)*int64(inter)
			moeRoute += 32
			moeExpert += 2 * int64(topK) * 16
			relu2 += int64(topK)*(pk(inter)+20) + pk(sh) + 20
			gemv += 4*int64(sh) + 16
			gemvAdd += 16
			moeNE = nE
			continue
		}
		if shape.block == blockSSM || shape.block == blockMLP {
			if len(wts.Layers[l].PreAttnNorm) == 0 {
				nQuant++
			} else {
				nRms++
			}
			if shape.block == blockSSM {
				anySSM = true
				gemv += 4*int64(rows["mb-proj"]) + 16
				mambaLayer += 4 * int64(rows["mb-conv"]+rows["mb-y"]+rows["mb-gated"])
				quantB += pk(rows["mb-gated"]) + 20
			} else {
				gemv += 4*int64(rows["up"]) + 16
				relu2 += pk(rows["up"]) + 20
			}
			gemvAdd += 16
			continue
		}
		if shape.latent {
			anyLatent = true
			for _, pre := range [][]float32{wts.Layers[l].PreAttnNorm, wts.Layers[l].PreMLPNorm} {
				if len(pre) == 0 {
					nQuant++
				} else {
					nRms++
				}
			}
			// The query projection (through a low-rank bottleneck and its norm where there is one), the latent projection; the absorbed
			// query, the rank-space context sum and the lifted context rows; the absorb and lift dispatches' uniforms.
			gemv += 4*int64(rows["mla-q"]+rows["mla-kv"]) + 2*16
			if qa := rows["mla-qa"]; qa > 0 {
				gemv += 4*int64(qa) + 16
				nRmsExtra += pk(qa) + 20
			}
			mlaLayer += 4 * int64(rows["mla-q-abs"]+rows["mla-wsum"]+rows["mla-ctx"])
			mlaAbsorb += 32
			mlaLift += 32
			quantB += pk(rows["mla-ctx"]) + 20
			gemvAdd += 16
			if err := ffnTerms(l, shape, rows, nE, topK, shared, hasShared); err != "" {
				return nil, err
			}
			continue
		}
		if shape.delta {
			anyDelta = true
			for _, pre := range [][]float32{wts.Layers[l].PreAttnNorm, wts.Layers[l].PreMLPNorm} {
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
		w := &wts.Layers[l]
		rs := float32(lv.RopeMscale())
		switch {
		case rs == 0:
			return nil, fmt.Sprintf("layer %d: rope scale 0", l)
		case lv.VFromK() && !shape.g4:
			return nil, fmt.Sprintf("layer %d: V from K", l)
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
		if shape.g4 { // the block's own post-norms are its weights', not a sublayer output norm the layer's weights carry
			shape.qkNorm, shape.postMLP = hasQNorm, len(w.PostMLPNorm) > 0
		}
		if shape.qkNorm != hasQNorm || shape.postAttn != (len(w.PostAttnNorm) > 0) || shape.postMLP != (len(w.PostMLPNorm) > 0) {
			return nil, fmt.Sprintf("layer %d: a norm or QK-norm the weights and the operations disagree on", l)
		}
		g := geom{lv.HeadDim(), lv.KVHeads(), len(lv.RopeInvFreq()), lv.VFromK()}
		if geoms[g] == nil {
			geoms[g] = map[float32]bool{}
		}
		geoms[g][rs] = true
		if shape.g4 && lv.KVSrc() != l {
			return nil, fmt.Sprintf("layer %d: it shares another layer's K and V", l)
		}
		pres := [][]float32{w.PreAttnNorm, w.PreMLPNorm}
		if shape.block == blockAttn || shape.g4 { // one pre-norm: a parallel block's own are counted with it
			pres = pres[:1]
		}
		for _, pre := range pres {
			if len(pre) == 0 { // a family with no pre-norm (norms after the sublayer only) quantizes the raw residual
				nQuant++
			} else {
				nRms++
			}
		}
		gemv += 4*int64(rows["q"]+rows["k"]+rows["v"]) + 3*16
		if shape.g4 { // the scale-less V norm writes its own row; a layer whose V is its K has no V projection
			attnLayer += 4 * int64(rows["k"])
			vNormB += 16
			if lv.VFromK() {
				gemv -= 4*int64(rows["v"]) + 16
			}
		}
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
		if shape.postAttn || len(w.OBias) > 0 { // an output bias splits it the same way
			gemvAdd -= 16
			gemv += 4*int64(H) + 16
			biasAddB += 16
		}
		if shape.postAttn {
			rmsnormB += 4*int64(H) + 16
		}
		if len(w.OBias) > 0 {
			biasAddB += 16 // the bias add on the bare output
		}
		if shape.block == blockPair && !shape.g4 {
			if err := ffnTerms(l, shape, rows, nE, topK, shared, hasShared); err != "" {
				return nil, err
			}
		}
		if shape.g4 {
			if err := g4Terms(l); err != "" {
				return nil, err
			}
		}
		attnLayer += 4*int64(rows["ctx"]) + 16
		quantB += pk(rows["ctx"]) + 20
	}
	if len(geoms) == 0 && !anyDelta && !anyLatent && !anySSM {
		return nil, "the model has no layers"
	}
	gemv += 4*int64(V) + 16 // the head
	geomB := int64(0)
	nScale := int64(0) // each rope scale's uniforms: one fused (f32 KV), or q's and k's (f16 and int8 KV)
	for _, scales := range geoms {
		one := int64(16 + 32) // the v-store and attention uniforms
		if win {
			one += 32
		}
		if kv == mem.KVI8 {
			one += 32
		}
		geomB += one
		nScale += int64(len(scales))
	}
	mambaSetupB := int64(0)
	if anySSM {
		mambaSetupB = 16 + 32 + 32 // the conv's, the scan's and the gated norm's uniforms, shared across layers
	}
	mlaSetupB := int64(0)
	if anyLatent {
		mlaSetupB = 32 + 48 + 32 // the latent store's, the query rope's (twelve words) and the attention's uniforms, shared across layers
	}
	deltaSetupB := int64(0)
	if anyDelta {
		deltaSetupB = 16 + 16 + 32 + 32 + 32 // the five dispatch parameter buffers the delta mixer shares across layers
	}
	terms = []webgpuScratchTerm{
		{"build", 4 + 4*webgpuLoRAScratchRows},
		{"residual", 4 * int64(H)},
		{"logitsStaging", 4 * int64(V)},
		{"rmsQuant", nRms*rmsQuantB + nRmsExtra},
		{"mlaLayer", mlaLayer},
		{"mlaAbsorb", mlaAbsorb},
		{"mlaLift", mlaLift},
		{"mlaSetup", mlaSetupB},
		{"mambaLayer", mambaLayer},
		{"mambaSetup", mambaSetupB},
		{"relu2Quant", relu2},
		{"vNorm", vNormB},
		{"gemma4MoE", moeBlock},
		{"scaleVec", scaleVecB},
		{"nemoMoE", routedRows},
		{clampedRoles[0], clampedRoute},
		{clampedRoles[1], clampedAct},
		{clampedRoles[2], clampedDown},
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
		terms = append(terms, webgpuScratchTerm{"ropeQUniFor", 32 * nScale}, webgpuScratchTerm{"ropeKUniFor", 32 * nScale})
	} else {
		terms = append(terms, webgpuScratchTerm{"qkvFinUniFor", 32 * nScale})
	}
	kept := terms[:0]
	for _, t := range terms {
		if t.Bytes != 0 { // a role the model's layers never run
			kept = append(kept, t)
		}
	}
	return kept, ""
}

// routesExperts reports whether a layer's declared rows include a router's.
func routesExperts(rows map[string]int) bool {
	for name := range rows {
		if strings.HasSuffix(name, "-router") {
			return true
		}
	}
	return false
}

// denseShape is the optional operations a plain attention layer carries: a QK-norm after the projections, and a norm on each
// sublayer's output before its residual add.
type denseShape struct {
	qkNorm, postAttn, postMLP, moe, gated, delta, latent, clamped, g4 bool
	routedPrefix                                                      string          // the declared rows' name prefix of a one-operation routed layer
	otherForm                                                         program.FFNForm // the routed form a layer has that is not derived
	block                                                             int             // blockPair, or a layer of one operation behind its one norm
}

const (
	blockPair = iota
	blockSSM
	blockAttn
	blockMLP
	blockRouted
)

// routedPrefixOf names the declared router rows of the routed forms that stand alone in a layer.
var routedPrefixOf = map[program.FFNForm]string{program.FfnNemotronMoE: "nemotronmoe"}

// denseAttnLayer reports whether a layer's operations are a plain attention sublayer and a dense FFN, each behind its own pre-norm,
// with only the optional operations of denseShape and nothing else (no output gate, temperature or layer scale).
func denseAttnLayer(ops []program.Op) (s denseShape, ok bool) {
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
	// A layer of one operation behind its one norm: a state-space mixer, an attention sublayer or a non-gated FFN.
	switch {
	case next(program.OpMamba2):
		s.block = blockSSM
		return s, next(program.OpResidual) && i == len(ops)
	case next(program.OpFFN):
		switch ops[i-1].FFN.Form {
		case program.FfnNonGated:
			s.block = blockMLP
		case program.FfnNemotronMoE:
			s.block, s.routedPrefix = blockRouted, routedPrefixOf[ops[i-1].FFN.Form]
		default:
			s.otherForm = ops[i-1].FFN.Form
			return s, false
		}
		return s, next(program.OpResidual) && i == len(ops)
	}
	switch {
	case next(program.OpDeltaNet):
		s.delta = true
	case next(program.OpGatedAttn):
		s.gated = true
	case next(program.OpMLA):
		s.latent = true
	case next(program.OpGemma4Attn):
		s.g4 = true
		s.postAttn = opt(program.OpNorm)
		return s, next(program.OpResidual) && next(program.OpGemma4MoE) && i == len(ops)
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
		if i == len(ops)-1 && next(program.OpResidual) { // an attention block: no FFN follows in the layer
			s.block = blockAttn
			return s, true
		}
		s.postAttn = opt(program.OpNorm)
	}
	if !next(program.OpResidual) || !next(program.OpNorm) || !next(program.OpFFN) {
		return s, false
	}
	switch ops[i-1].FFN.Form {
	case program.FfnMoE:
		s.moe = true
	case program.FfnGptOss: // a router with a logit bias and clamped experts: its own builders, but the same dispatch shape
		s.moe, s.clamped = true, true
	case program.FfnGated, program.FfnNonGated:
	default: // gpt-oss, Llama 4, Nemotron and Gemma 4's own routed forms are other builders
		s.otherForm = ops[i-1].FFN.Form
		return s, false
	}
	s.postMLP = opt(program.OpNorm)
	return s, next(program.OpResidual) && i == len(ops)
}

// opKinds is a layer's operation kinds in order, for the reason a layer has no figure.
func opKinds(ops []program.Op) []program.OpKind {
	ks := make([]program.OpKind, len(ops))
	for i, o := range ops {
		ks[i] = o.Kind
	}
	return ks
}

// gatedAttnQKNorm reports whether a gated attention layer carries a QK-norm weight, which it keeps with its projections, not in the
// layer's own norm fields. Like denseAttnLayer it reads the layer shape's own accessor, so the pricing body names no family.
func gatedAttnQKNorm(lv decoder.ResidentLayerView) bool {
	_, _, _, _, qn, _ := lv.Qwen35AttnWeights()
	return len(qn) > 0
}

// clampedRoles are the runner's role names for the clamped router's three builders: the route, the clamped activation and the biased
// down projection. They are the runner's names for builders it names by family, so they live beside the matcher, not in the pricing.
var clampedRoles = [3]string{"routeGptOss", "gptossActQuant", "moeExpertGptOssDown"}

// parallelBlock is the dimensions of a layer's parallel dense and routed block.
type parallelBlock struct {
	nE, topK, denseInter, moeInter int
	layerScalar                    float32
}

// parallelBlockOf reads a layer's parallel block from the bundle the runner builds it from, as gatedAttnQKNorm reads the gated
// attention's: the pricing body names no family.
func parallelBlockOf(lv decoder.ResidentLayerView) (parallelBlock, bool) {
	g, ok := lv.Gemma4MoEBundle()
	if !ok {
		return parallelBlock{}, false
	}
	return parallelBlock{g.NE, g.TopK, g.DenseInter, g.MoeInter, g.LayerScalar}, true
}
