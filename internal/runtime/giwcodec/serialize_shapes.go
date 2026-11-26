package giwcodec

import (
	"fmt"

	"github.com/townsendmerino/aikit/linalg"
	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/archparts"
	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

// The per-group checks validateShapes runs. Each returns the first mismatch it finds as a *SerializeError naming the layer
// and tensor, or nil.

// shapeEq checks a projection's row count.
func shapeEq(name string, got, want int) *SerializeError {
	if got != want {
		return &SerializeError{fmt.Sprintf("%s: %d rows, arch expects %d", name, got, want)}
	}
	return nil
}

// shapeVec checks a per-layer f32 vector (bias / norm weight) whose length the blob controls but the forward indexes at an
// arch-derived width: addBias iterates over the projection output and rmsNorm indexes weight[0:dim], so a short vector
// slice-panics in the decode goroutine and a long one is silently mis-consumed. 0 = absent (a family that omits it),
// allowed.
func shapeVec(name string, got, want int) *SerializeError {
	if got != 0 && got != want {
		return &SerializeError{fmt.Sprintf("%s: len %d, arch expects %d", name, got, want)}
	}
	return nil
}

// shapeReq is shapeVec for a vector the family's forward dereferences unconditionally, where "absent" is not a family that
// omits it but a bundle that is missing it. shapeVec's `got == 0 => allowed` is what would let a pre-v6 gpt-oss sidecar
// through: it is within giwMinReadV, "fresh" by mtime, reads AttnSinks as nil, passes validateShapes, and panics at
// forward_gptoss.go's `lw.AttnSinks[qh]` on the first request. Same shape as the LFM2 conv presence check.
func shapeReq(name string, got, want int) *SerializeError {
	if got == 0 {
		return &SerializeError{fmt.Sprintf("%s: absent, arch requires len %d — the bundle "+
			"predates this field (rewrite the .giw sidecar; mtime freshness cannot see a "+
			"missing tensor)", name, want)}
	}
	return shapeVec(name, got, want)
}

// shapeCols checks a table's row width. The embedding lookup widens one row into a HiddenDim-sized scratch (WeightMat.Row),
// so a table the blob declares wider than that is refused here, by name, instead of reaching the kernel; the LM head is
// multiplied against a HiddenDim activation, and a gpt-2 position row is added to one.
func shapeCols(name string, got, want int) *SerializeError {
	if got != want {
		return &SerializeError{fmt.Sprintf("%s: %d cols, arch expects %d", name, got, want)}
	}
	return nil
}

// shapeMat checks a projection's [rows, cols]. A matmul writes Rows() outputs into an arch-sized scratch and reads Cols()
// inputs from an arch-sized activation, so either dimension wrong is an out-of-bounds access or a silent mis-read. An absent
// matrix (Rows() == 0) is an error only when required: the layer's program runs a matmul over it.
func shapeMat(name string, m *linalg.WeightMat, rows, cols int, required bool) *SerializeError {
	if m.Rows() == 0 {
		if required {
			return &SerializeError{fmt.Sprintf("%s: absent, the layer's forward requires [%d, %d]", name, rows, cols)}
		}
		return nil
	}
	if e := shapeEq(name, m.Rows(), rows); e != nil {
		return e
	}
	return shapeCols(name, m.Cols(), cols)
}

// shapeFlat checks a projection held either as a flat f32 [rows*cols] slice or as a quantized matrix (the MLA and Mamba-2
// projections: mambaProj's two encodings).
func shapeFlat(name string, f []float32, q *linalg.WeightMat, rows, cols int, required bool) *SerializeError {
	if f != nil {
		if len(f) != rows*cols {
			return &SerializeError{fmt.Sprintf("%s: len %d, arch expects %d×%d", name, len(f), rows, cols)}
		}
		return nil
	}
	return shapeMat(name, q, rows, cols, required)
}

// layerNeeds is what layer i's program dereferences: which mixer struct, and which generic projections it multiplies by.
type layerNeeds struct {
	qkv, kv, v, o                bool // generic attention: QProj, KProj, VProj, OProj
	gate, up, down, router       bool // FFN projections
	delta, qattn, mla, kda, conv bool // hybrid mixers
	mamba                        bool
	mambaNormGroups              int
	mlaHeadWise                  bool
	gemma4MoE                    bool
	ple                          bool // gemma-4's per-layer-embedding branch (program.OpPLE)
}

// needsAt reads layer i's needs from the ops the forward walks (Architecture.layerOps), so a check cannot drift from the
// program: each op that dereferences a struct or multiplies by a projection makes it mandatory here.
func needsAt(w *darch.Weights, arch *darch.Architecture, i int) layerNeeds {
	var n layerNeeds
	lw := &w.Layers[i]
	for _, o := range arch.LayerOps(i, w.HasExperts(i)) {
		switch o.Kind {
		case program.OpProjQKV:
			n.qkv, n.kv, n.v = true, true, !lw.VFromK
		case program.OpProjO:
			n.o = true
		case program.OpGemma4Attn:
			n.qkv, n.o = true, true
			n.kv = o.G4.OwnsKV
			n.v = o.G4.OwnsKV && !lw.VFromK
		case program.OpFFN:
			switch o.FFN.Form {
			case program.FfnGated, program.FfnGemma4Dense:
				n.gate, n.up, n.down = true, true, true
			case program.FfnNonGated:
				n.up, n.down = true, true
			default:
				n.router = true
			}
		case program.OpGemma4MoE:
			n.gemma4MoE = true
		case program.OpPLE:
			n.ple = true
		case program.OpDeltaNet:
			n.delta = true
		case program.OpGatedAttn:
			n.qattn = true
		case program.OpMLA:
			n.mla, n.mlaHeadWise = true, o.MLA.HeadWise
		case program.OpKDA:
			n.kda = true
		case program.OpShortConv:
			n.conv = true
		case program.OpMamba2:
			n.mamba, n.mambaNormGroups = true, o.Mamba.NormGroups
		}
	}
	return n
}

// validateLayerAttnShapes checks layer i's generic attention projections, gate, biases, norms and sinks.
func validateLayerAttnShapes(lw *darch.LayerWeights, arch *darch.Architecture, i int, n layerNeeds) *SerializeError {
	hd := arch.HeadDimAt(i)
	// headsAt(i), not NumHeads: some families (Laguna) vary the query head count per layer, so a uniform NumHeads would
	// reject a correctly written bundle. headDimAt/kvHeadsAt/ffnAt are per-layer here too.
	qDim, kvDim := arch.HeadsAt(i)*hd, arch.KVHeadsAt(i)*hd
	for _, c := range []struct {
		name       string
		m          *linalg.WeightMat
		rows, cols int
		required   bool
	}{
		{"QProj", &lw.QProj, qDim, arch.HiddenDim, n.qkv},
		{"KProj", &lw.KProj, kvDim, arch.HiddenDim, n.kv},
		{"VProj", &lw.VProj, kvDim, arch.HiddenDim, n.v},
		{"OProj", &lw.OProj, arch.HiddenDim, qDim, n.o},
	} {
		if e := shapeMat(fmt.Sprintf("layer %d %s", i, c.name), c.m, c.rows, c.cols, c.required); e != nil {
			return e
		}
	}
	// GProj (Laguna's attention output gate) is legal at EITHER width — per-head (one scalar per
	// query head) or per-element (the full qDim) — and which one ships is decided by the tensor,
	// not the config (XS.2 declares per-element and ships per-head). So both are accepted and
	// anything else is rejected, rather than leaving the field unchecked.
	if g := lw.GProj.Rows(); g > 0 {
		if g != arch.HeadsAt(i) && g != qDim {
			return &SerializeError{fmt.Sprintf("layer %d GProj: %d rows, arch expects %d (per-head) or %d (per-element)",
				i, g, arch.HeadsAt(i), qDim)}
		}
		if e := shapeCols(fmt.Sprintf("layer %d GProj", i), lw.GProj.Cols(), arch.HiddenDim); e != nil {
			return e
		}
	}
	// Per-layer f32 vectors: biases feed addBias over the projection output, norm weights feed rmsNorm at their consumed width
	// (QK-norm is per-head-dim hd normally, but the whole projected width when the family sets QKNormWhole: olmo3's
	// whole-vector QK-norm, which reuses rmsNorm at rows=1/dim=qDim; the block norms are hidden). The blob controls their
	// length; the forward indexes an arch width with no check.
	qNormWant, kNormWant := hd, hd
	if arch.QKNormWhole {
		qNormWant, kNormWant = qDim, kvDim
	}
	for _, c := range []struct {
		name string
		got  int
		want int
	}{
		{"QBias", len(lw.QBias), qDim}, {"KBias", len(lw.KBias), kvDim},
		{"VBias", len(lw.VBias), kvDim}, {"OBias", len(lw.OBias), arch.HiddenDim},
		{"QNorm", len(lw.QNorm), qNormWant}, {"KNorm", len(lw.KNorm), kNormWant},
		{"PreAttnNorm", len(lw.PreAttnNorm), arch.HiddenDim}, {"PostAttnNorm", len(lw.PostAttnNorm), arch.HiddenDim},
		{"PreMLPNorm", len(lw.PreMLPNorm), arch.HiddenDim}, {"PostMLPNorm", len(lw.PostMLPNorm), arch.HiddenDim},
		{"UpBias", len(lw.UpBias), arch.FFNAt(i)}, {"DownBias", len(lw.DownBias), arch.HiddenDim},
	} {
		if e := shapeVec(fmt.Sprintf("layer %d %s", i, c.name), c.got, c.want); e != nil {
			return e
		}
	}
	// The v6 completeness tail. forward_gptoss.go and the generic sink attention path index AttnSinks[qh] for every head, so for
	// these families they are required, not optional; a sidecar written before v6 is sink-free, within giwMinReadV, and judged
	// fresh by mtime.
	if arch.HasAttnSinks() {
		if e := shapeReq(fmt.Sprintf("layer %d AttnSinks", i), len(lw.AttnSinks), arch.HeadsAt(i)); e != nil {
			return e
		}
	}
	return nil
}

// validateLayerMLPShapes checks layer i's dense FFN, the generic MoE block (router, experts, shared expert) and gemma-4's
// PLE branch and dense||MoE sub-block.
func validateLayerMLPShapes(lw *darch.LayerWeights, arch *darch.Architecture, i int, n layerNeeds) *SerializeError {
	ffn := arch.FFNAt(i)
	for _, c := range []struct {
		name       string
		m          *linalg.WeightMat
		rows, cols int
		required   bool
	}{
		{"GateProj", &lw.GateProj, ffn, arch.HiddenDim, n.gate},
		{"UpProj", &lw.UpProj, ffn, arch.HiddenDim, n.up},
		{"DownProj", &lw.DownProj, arch.HiddenDim, ffn, n.down},
	} {
		if e := shapeMat(fmt.Sprintf("layer %d %s", i, c.name), c.m, c.rows, c.cols, c.required); e != nil {
			return e
		}
	}
	if e := validateLayerPLEShapes(lw, arch, i, n.ple); e != nil {
		return e
	}
	if e := validateGemma4MoEShapes(lw, arch, i, n.gemma4MoE); e != nil {
		return e
	}
	if arch.MoE == nil {
		return nil
	}
	moe := arch.MoE
	// RouterBias is addBias'd over the [NumExperts] router logits; the shared expert's Gate/Up write SharedIntermediateDim
	// scratch, Down writes hidden, and SharedGate is the scalar sigmoid gate (1 row). A short blob panics; validate them.
	if e := shapeVec(fmt.Sprintf("layer %d RouterBias", i), len(lw.RouterBias), moe.NumExperts); e != nil {
		return e
	}
	// Router feeds moeMLP's make([]float32, NumExperts) — the exploit the audit names.
	for _, c := range []struct {
		name       string
		m          *linalg.WeightMat
		rows, cols int
		required   bool
	}{
		{"SharedExpert Gate", &lw.SharedExpert.Gate, moe.SharedIntermediateDim, arch.HiddenDim, false},
		{"SharedExpert Up", &lw.SharedExpert.Up, moe.SharedIntermediateDim, arch.HiddenDim, false},
		{"SharedExpert Down", &lw.SharedExpert.Down, arch.HiddenDim, moe.SharedIntermediateDim, false},
		{"SharedGate", &lw.SharedGate, 1, arch.HiddenDim, false},
		{"Router", &lw.Router, moe.NumExperts, arch.HiddenDim, n.router},
	} {
		if e := shapeMat(fmt.Sprintf("layer %d %s", i, c.name), c.m, c.rows, c.cols, c.required); e != nil {
			return e
		}
	}
	// Per-expert Gate/Up (moe intermediate width) and Down (hidden) — same config-sized-scratch
	// write class as the dense projections. Empty for families that stack experts elsewhere.
	for xe := range lw.Experts {
		ex := &lw.Experts[xe]
		for _, c := range []struct {
			name       string
			m          *linalg.WeightMat
			rows, cols int
		}{
			{"Gate", &ex.Gate, moe.IntermediateDim, arch.HiddenDim},
			{"Up", &ex.Up, moe.IntermediateDim, arch.HiddenDim},
			{"Down", &ex.Down, arch.HiddenDim, moe.IntermediateDim},
		} {
			if e := shapeMat(fmt.Sprintf("layer %d expert %d %s", i, xe, c.name), c.m, c.rows, c.cols, false); e != nil {
				return e
			}
		}
	}
	// gpt-oss: addBias iterates every expert bias with no nil check, so they are required (the v6 completeness tail).
	if arch.GPTOSS != nil {
		for xe := range lw.Experts {
			ex := &lw.Experts[xe]
			if ex.Gate.Rows() == 0 {
				continue // an expert the bundle stacks elsewhere; its biases live there too
			}
			for _, c := range []struct {
				name string
				got  int
				want int
			}{
				{"GateBias", len(ex.GateBias), moe.IntermediateDim},
				{"UpBias", len(ex.UpBias), moe.IntermediateDim},
				{"DownBias", len(ex.DownBias), arch.HiddenDim},
			} {
				if e := shapeReq(fmt.Sprintf("layer %d expert %d %s", i, xe, c.name), c.got, c.want); e != nil {
					return e
				}
			}
		}
	}
	return nil
}

// validateLayerPLEShapes checks gemma-4's per-layer-embedding (PLE) branch: PLEGate/PLEProj are matmul'd and PostPLENorm
// rmsNorm'd at fixed widths; a wrong-shape blob OOB-panics or silently truncates the PLE activation. required is whether the
// layer's program runs the branch, which then dereferences all three unconditionally.
func validateLayerPLEShapes(lw *darch.LayerWeights, arch *darch.Architecture, i int, required bool) *SerializeError {
	if arch.Gemma4 == nil {
		return nil
	}
	if pleDim := arch.Gemma4.HiddenSizePerLayerInput; pleDim > 0 {
		if e := shapeMat(fmt.Sprintf("layer %d PLEGate", i), &lw.PLEGate, pleDim, arch.HiddenDim, required); e != nil {
			return e
		}
		if e := shapeMat(fmt.Sprintf("layer %d PLEProj", i), &lw.PLEProj, arch.HiddenDim, pleDim, required); e != nil {
			return e
		}
	}
	if required {
		return shapeReq(fmt.Sprintf("layer %d PostPLENorm", i), len(lw.PostPLENorm), arch.HiddenDim)
	}
	return shapeVec(fmt.Sprintf("layer %d PostPLENorm", i), len(lw.PostPLENorm), arch.HiddenDim)
}

// validateGemma4MoEShapes checks gemma-4's dense||MoE sub-block (l.gemma4moe): the standard Router/Experts are empty for
// gemma-4, so the arch.MoE block never validates it. The router feeds make([]float32, nE) and each expert index runs to nE:
// a short router mis-routes silently, a long one or ne != NumExperts panics in the decode goroutine. Cross-check against the
// arch. required is whether the layer's program runs the sub-block.
func validateGemma4MoEShapes(lw *darch.LayerWeights, arch *darch.Architecture, i int, required bool) *SerializeError {
	mo := lw.Gemma4MoE
	if mo == nil || arch.MoE == nil {
		if required {
			return &SerializeError{fmt.Sprintf("layer %d: gemma4 MoE layer has no dense||MoE sub-block", i)}
		}
		return nil
	}
	nE, moeInter := arch.MoE.NumExperts, arch.MoE.IntermediateDim
	if e := shapeMat(fmt.Sprintf("layer %d gemma4moe router", i), &mo.RouterProj, nE, arch.HiddenDim, true); e != nil {
		return e
	}
	if len(mo.ExpertsGateUp) != nE || len(mo.ExpertsDown) != nE {
		return &SerializeError{fmt.Sprintf("layer %d gemma4moe expert count: gate|up %d / down %d, arch expects %d", i, len(mo.ExpertsGateUp), len(mo.ExpertsDown), nE)}
	}
	if e := shapeVec(fmt.Sprintf("layer %d gemma4moe perExpertScale", i), len(mo.PerExpertScale), nE); e != nil {
		return e
	}
	// routerScale scales the [hidden] router input and the five norms rmsNorm at hidden, all unconditionally: an absent or
	// short slice OOB-panics in the decode goroutine.
	for _, c := range []struct {
		name string
		got  int
	}{
		{"gemma4moe routerScale", len(mo.RouterScale)},
		{"gemma4moe preFFNNorm", len(mo.PreFFNNorm)},
		{"gemma4moe postFFNNorm1", len(mo.PostFFNNorm1)},
		{"gemma4moe preFFNNorm2", len(mo.PreFFNNorm2)},
		{"gemma4moe postFFNNorm2", len(mo.PostFFNNorm2)},
		{"gemma4moe postFFNNorm", len(mo.PostFFNNorm)},
	} {
		if e := shapeReq(fmt.Sprintf("layer %d %s", i, c.name), c.got, arch.HiddenDim); e != nil {
			return e
		}
	}
	// The dense branch writes denseInter-sized scratch from mlpGate/mlpUp and hidden from mlpDown.
	for _, c := range []struct {
		name       string
		m          *linalg.WeightMat
		rows, cols int
	}{
		{"gemma4moe mlpGate", &mo.MlpGate, mo.DenseInter, arch.HiddenDim},
		{"gemma4moe mlpUp", &mo.MlpUp, mo.DenseInter, arch.HiddenDim},
		{"gemma4moe mlpDown", &mo.MlpDown, arch.HiddenDim, mo.DenseInter},
	} {
		if e := shapeMat(fmt.Sprintf("layer %d %s", i, c.name), c.m, c.rows, c.cols, true); e != nil {
			return e
		}
	}
	for xe := range mo.ExpertsGateUp {
		if e := shapeMat(fmt.Sprintf("layer %d gemma4moe expert %d gate|up", i, xe), &mo.ExpertsGateUp[xe], 2*moeInter, arch.HiddenDim, true); e != nil {
			return e
		}
		if e := shapeMat(fmt.Sprintf("layer %d gemma4moe expert %d down", i, xe), &mo.ExpertsDown[xe], arch.HiddenDim, moeInter, true); e != nil {
			return e
		}
	}
	return nil
}

// validateLayerHybridShapes checks layer i's hybrid mixer: the Gated DeltaNet and gated softmax attention of the qwen3.5
// family, DeepSeek MLA, Bailing's KDA, Mamba-2 and LFM2's short conv. Each struct is flat f32 slices and matrices the forward
// indexes at arch-derived widths; a mixer the program runs but the bundle lacks is a nil dereference at the first forward.
func validateLayerHybridShapes(lw *darch.LayerWeights, arch *darch.Architecture, i int, n layerNeeds) *SerializeError {
	for _, c := range []struct {
		name    string
		need    bool
		present bool
	}{
		{"Gated DeltaNet", n.delta, lw.Delta != nil},
		{"gated attention", n.qattn, lw.QAttn != nil},
		{"MLA attention", n.mla, lw.MLA != nil},
		{"KDA attention", n.kda, lw.KDA != nil},
		{"Mamba-2 mixer", n.mamba, lw.Mamba != nil},
	} {
		if c.need && !c.present {
			return &SerializeError{fmt.Sprintf("layer %d: the program runs a %s but the bundle has none", i, c.name)}
		}
	}
	if e := validateShortConvShapes(lw, arch, i, n.conv); e != nil {
		return e
	}
	if d := lw.Delta; d != nil && arch.Qwen35 != nil {
		if e := validateDeltaNetShapes(d, arch, i); e != nil {
			return e
		}
	}
	if q := lw.QAttn; q != nil {
		if e := validateGatedAttnShapes(q, arch, i); e != nil {
			return e
		}
	}
	if m := lw.MLA; m != nil && arch.MLA != nil {
		if e := validateMLAShapes(m, arch, i, n.mlaHeadWise); e != nil {
			return e
		}
	}
	if k := lw.KDA; k != nil && arch.KDA != nil {
		if e := validateKDAShapes(k, arch, i); e != nil {
			return e
		}
	}
	if m := lw.Mamba; m != nil && (arch.Granite != nil || arch.Nemotron != nil) {
		if e := validateMamba2Shapes(m, arch.Mamba2Shapes(n.mambaNormGroups), i); e != nil {
			return e
		}
	}
	return nil
}

// validateShortConvShapes checks LFM2's short-conv mixer. Its three tensors are flat f32 slices the forward indexes at
// arch-derived widths, so a short one slice-panics in the decode goroutine, as for the bias vectors. The presence check is
// the load-bearing half: a conv layer whose mixer is absent is what a pre-v8 .giw hands back, and the panic it produces at
// the first forward is the defect this validation converts into a refusal at load.
func validateShortConvShapes(lw *darch.LayerWeights, arch *darch.Architecture, i int, required bool) *SerializeError {
	if arch.LFM2 == nil {
		return nil
	}
	cd, k := arch.LFM2.ConvDim, arch.LFM2.ConvLCache
	isConv := required || lw.QProj.Rows() == 0 // a conv layer loads no attention projections
	switch {
	case isConv && lw.ShortConv == nil:
		return &SerializeError{fmt.Sprintf("layer %d: lfm2 conv layer has no short-conv "+
			"mixer — the bundle was written before the v8 tail and would nil-deref at the "+
			"first forward", i)}
	case lw.ShortConv == nil:
		return nil
	}
	c := lw.ShortConv
	if e := shapeMat(fmt.Sprintf("layer %d shortConv.inProj", i), &c.InProj, 3*cd, arch.HiddenDim, true); e != nil {
		return e
	}
	if e := shapeEq(fmt.Sprintf("layer %d shortConv.convW", i), len(c.ConvW), cd*k); e != nil {
		return e
	}
	return shapeMat(fmt.Sprintf("layer %d shortConv.outProj", i), &c.OutProj, arch.HiddenDim, cd, true)
}

// validateDeltaNetShapes checks a Gated DeltaNet layer against the qwen35 geometry: in_proj_qkv emits [q;k;v] (convDim),
// in_proj_z the valueDim output gate, the b/a gates and the per-value-head vectors one value per value head.
func validateDeltaNetShapes(d *archparts.DeltaNetWeights, arch *darch.Architecture, i int) *SerializeError {
	g, hidden := arch.Qwen35, arch.HiddenDim
	keyDim, valueDim, numV := g.NumKeyHeads*g.KeyHeadDim, g.NumValueHeads*g.ValueHeadDim, g.NumValueHeads
	convDim := 2*keyDim + valueDim
	for _, c := range []struct {
		name       string
		m          *linalg.WeightMat
		rows, cols int
	}{
		{"inProjQKV", &d.InProjQKV, convDim, hidden},
		{"inProjZ", &d.InProjZ, valueDim, hidden},
		{"outProj", &d.OutProj, hidden, valueDim},
	} {
		if e := shapeMat(fmt.Sprintf("layer %d delta.%s", i, c.name), c.m, c.rows, c.cols, true); e != nil {
			return e
		}
	}
	for _, c := range []struct {
		name string
		got  int
		want int
	}{
		{"inProjB", len(d.InProjB), numV * hidden}, {"inProjA", len(d.InProjA), numV * hidden},
		{"convW", len(d.ConvW), convDim * g.ConvKernel}, {"dtBias", len(d.DtBias), numV},
		{"negExpA", len(d.NegExpA), numV}, {"normW", len(d.NormW), g.ValueHeadDim},
	} {
		if e := shapeReq(fmt.Sprintf("layer %d delta.%s", i, c.name), c.got, c.want); e != nil {
			return e
		}
	}
	return nil
}

// validateGatedAttnShapes checks a qwen3.5 gated softmax layer: q_proj is double width ([query ‖ gate] per head).
func validateGatedAttnShapes(q *archparts.QwenAttnWeights, arch *darch.Architecture, i int) *SerializeError {
	hd, hidden := arch.HeadDimAt(i), arch.HiddenDim
	qDim, kvDim := arch.HeadsAt(i)*hd, arch.KVHeadsAt(i)*hd
	for _, c := range []struct {
		name       string
		m          *linalg.WeightMat
		rows, cols int
	}{
		{"qProj", &q.QProj, 2 * qDim, hidden},
		{"kProj", &q.KProj, kvDim, hidden},
		{"vProj", &q.VProj, kvDim, hidden},
		{"oProj", &q.OProj, hidden, qDim},
	} {
		if e := shapeMat(fmt.Sprintf("layer %d qattn.%s", i, c.name), c.m, c.rows, c.cols, true); e != nil {
			return e
		}
	}
	if e := shapeVec(fmt.Sprintf("layer %d qattn.qNorm", i), len(q.QNorm), hd); e != nil {
		return e
	}
	return shapeVec(fmt.Sprintf("layer %d qattn.kNorm", i), len(q.KNorm), hd)
}

// validateMLAShapes checks a DeepSeek MLA layer: the q LoRA bottleneck (or the direct q_proj when QLoRARank is 0), the
// latent down-projection [KVLoRARank+QKRopeHeadDim], the per-head up-projection and the optional output gate.
func validateMLAShapes(m *archparts.MLAWeights, arch *darch.Architecture, i int, headWise bool) *SerializeError {
	p, hidden, H := arch.MLA, arch.HiddenDim, arch.NumHeads
	qk, rank := p.QKHeadDim(), p.KVLoRARank
	type flat struct {
		name       string
		f          []float32
		q          *linalg.WeightMat
		rows, cols int
	}
	checks := []flat{
		{"kvAProj", m.KVAProj, &m.KVAQ, rank + p.QKRopeHeadDim, hidden},
		{"kvBProj", m.KVBProj, &linalg.WeightMat{}, H * (p.QKNopeHeadDim + p.VHeadDim), rank},
		{"oProj", m.OProj, &m.OQ, hidden, H * p.VHeadDim},
	}
	if p.QLoRARank > 0 {
		checks = append(checks, flat{"qAProj", m.QAProj, &m.QAQ, p.QLoRARank, hidden}, flat{"qBProj", m.QBProj, &m.QBQ, H * qk, p.QLoRARank})
		if e := shapeReq(fmt.Sprintf("layer %d mla.qALayernorm", i), len(m.QALayernorm), p.QLoRARank); e != nil {
			return e
		}
	} else {
		checks = append(checks, flat{"qProj", m.QProj, &m.QQ, H * qk, hidden})
	}
	for _, c := range checks {
		if e := shapeFlat(fmt.Sprintf("layer %d mla.%s", i, c.name), c.f, c.q, c.rows, c.cols, true); e != nil {
			return e
		}
	}
	if e := shapeReq(fmt.Sprintf("layer %d mla.kvALayernorm", i), len(m.KVALayernorm), rank); e != nil {
		return e
	}
	if m.GProj != nil {
		gRows := H * p.VHeadDim
		if headWise {
			gRows = H
		}
		if e := shapeFlat(fmt.Sprintf("layer %d mla.gProj", i), m.GProj, nil, gRows, hidden, true); e != nil {
			return e
		}
	}
	return nil
}

// validateKDAShapes checks a Bailing Hybrid KDA layer: q/k/v and the two gate projections are [numHeads*headDim, hidden],
// the depthwise convs ConvKernel taps per channel, the decay bias per channel and A_log and beta per head.
func validateKDAShapes(k *archparts.KDAWeights, arch *darch.Architecture, i int) *SerializeError {
	p, hidden := arch.KDA, arch.HiddenDim
	dim := p.NumHeads * p.HeadDim
	for _, c := range []struct {
		name       string
		m          *linalg.WeightMat
		rows, cols int
	}{
		{"qProj", &k.QProj, dim, hidden}, {"kProj", &k.KProj, dim, hidden}, {"vProj", &k.VProj, dim, hidden},
		{"fProj", &k.FProj, dim, hidden}, {"gProj", &k.GProj, dim, hidden}, {"oProj", &k.OProj, hidden, dim},
	} {
		if e := shapeMat(fmt.Sprintf("layer %d kda.%s", i, c.name), c.m, c.rows, c.cols, true); e != nil {
			return e
		}
	}
	for _, c := range []struct {
		name string
		got  int
		want int
	}{
		{"qConvW", len(k.QConvW), dim * p.ConvKernel}, {"kConvW", len(k.KConvW), dim * p.ConvKernel},
		{"vConvW", len(k.VConvW), dim * p.ConvKernel}, {"dtBias", len(k.DtBias), dim},
		{"aLog", len(k.ALog), p.NumHeads}, {"bProj", len(k.BProj), p.NumHeads * hidden},
		{"oNormW", len(k.ONormW), p.HeadDim},
	} {
		if e := shapeReq(fmt.Sprintf("layer %d kda.%s", i, c.name), c.got, c.want); e != nil {
			return e
		}
	}
	return nil
}

// validateMamba2Shapes checks a Mamba-2 mixer against p: in_proj emits [z (dInner) ‖ xBC (convDim) ‖ dt (nHeads)], the
// conv runs over convDim channels, and the per-head vectors are nHeads long.
func validateMamba2Shapes(m *archparts.Mamba2Weights, p archparts.Mamba2Params, i int) *SerializeError {
	dInner := p.DInner()
	convDim := dInner + 2*p.NGroups*p.DState
	if e := shapeFlat(fmt.Sprintf("layer %d mamba.inProj", i), m.InProj, &m.InQ, p.ProjDim(), p.Hidden, true); e != nil {
		return e
	}
	if e := shapeFlat(fmt.Sprintf("layer %d mamba.outProj", i), m.OutProj, &m.OutQ, p.Hidden, dInner, true); e != nil {
		return e
	}
	for _, c := range []struct {
		name string
		got  int
		want int
	}{
		{"convW", len(m.ConvW), convDim * p.DConv}, {"convB", len(m.ConvB), convDim},
		{"aLog", len(m.ALog), p.NHeads}, {"d", len(m.D), p.NHeads}, {"dtBias", len(m.DtBias), p.NHeads},
		{"normW", len(m.NormW), dInner},
	} {
		if e := shapeReq(fmt.Sprintf("layer %d mamba.%s", i, c.name), c.got, c.want); e != nil {
			return e
		}
	}
	return nil
}
