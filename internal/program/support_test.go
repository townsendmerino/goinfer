package program

import (
	"strings"
	"testing"
)

// layer is a program of one layer holding ops.
func layer(ops ...Op) *Program { return &Program{Layers: [][]Op{ops}} }

// TestSupport_eachParameterCheckDeclines holds every check of Support.gaps to a case: an operation whose parameter Metal does not
// implement is a gap naming it, and the same operation at a value Metal implements is none.
func TestSupport_eachParameterCheckDeclines(t *testing.T) {
	for _, c := range []struct {
		name     string
		bad, ok  Op
		wantText string
	}{
		{"an FFN activation", Op{Kind: OpFFN, FFN: FFNParams{Form: FfnGated, Act: ActGelu}}, Op{Kind: OpFFN, FFN: FFNParams{Form: FfnGated, Act: ActSiLU}},
			"has no GELU activation in its gated block"},
		{"an FFN form", Op{Kind: OpFFN, FFN: FFNParams{Form: FfnLlama4MoE, Act: ActSiLU}}, Op{Kind: OpFFN, FFN: FFNParams{Form: FfnMoE, Act: ActSiLU}},
			"has no block for the FFN form Llama 4 MoE"},
		{"an operation kind", Op{Kind: OpKDA}, Op{Kind: OpDeltaNet}, "has no step for it"},
		{"a QK-norm form", Op{Kind: OpQKNorm, QKNorm: QKNormParams{Form: QKL2}}, Op{Kind: OpQKNorm, QKNorm: QKNormParams{Form: QKRMS}},
			"parameter-free L2 QK-norm"},
		{"a rotation kernel", Op{Kind: OpRope, Rope: RopeParams{Kernel: RopeMLAKernel}}, Op{Kind: OpRope, Rope: RopeParams{Kernel: RopeAtKernel}},
			"de-interleaving (mlaRope) rotation"},
		{"a chunked attention", Op{Kind: OpAttention, Attn: AttentionParams{Chunk: 8192}}, Op{Kind: OpAttention, Attn: AttentionParams{Window: 4096}},
			"chunked attention"},
		{"a residual scale", Op{Kind: OpResidual, Res: ResidualParams{Scale: 0.22}}, Op{Kind: OpResidual, Res: ResidualParams{Scale: 1}},
			"a residual scale (0.22)"},
		{"DeltaNet's widened write gate", Op{Kind: OpDeltaNet, Delta: DeltaNetOpParams{Eps: 1e-6, NegEigval: true}}, Op{Kind: OpDeltaNet, Delta: DeltaNetOpParams{Eps: 1e-6}},
			"widened write gate (NegEigval)"},
		{"DeltaNet's own output-norm epsilon", Op{Kind: OpDeltaNet, Delta: DeltaNetOpParams{Eps: 1e-6, ONormEps: 1e-5}},
			Op{Kind: OpDeltaNet, Delta: DeltaNetOpParams{Eps: 1e-6, ONormEps: 1e-6}}, "own output-norm epsilon"},
		{"gated attention's QK-norm form", Op{Kind: OpGatedAttn, Gated: GatedAttnParams{QKNorm: QKNormParams{Form: QKL2}, Rope: RopeParams{Kernel: RopeAtKernel}}},
			Op{Kind: OpGatedAttn, Gated: GatedAttnParams{QKNorm: QKNormParams{Form: QKRMS}, Rope: RopeParams{Kernel: RopeAtKernel}}}, "parameter-free L2 QK-norm"},
	} {
		gaps := Metal.Gaps(layer(c.bad))
		if len(gaps) != 1 || !strings.Contains(gaps[0], c.wantText) {
			t.Errorf("%s: gaps %q, want one naming %q", c.name, gaps, c.wantText)
		}
		if err := Metal.Check(layer(c.bad)); err == nil || !strings.Contains(err.Error(), c.wantText) {
			t.Errorf("%s: Check gives %v, want it to name %q", c.name, err, c.wantText)
		}
		if gaps := Metal.Gaps(layer(c.ok)); len(gaps) != 0 {
			t.Errorf("%s: the implemented value gives gaps %q", c.name, gaps)
		}
	}
}

// TestSupport_gapsOnOneOperationAllNamed (Gemini on #548): an operation outside the table on several parameters is a gap for
// each, so lifting one leaves the others in view.
func TestSupport_gapsOnOneOperationAllNamed(t *testing.T) {
	for _, c := range []struct {
		name string
		op   Op
		want []string
	}{
		{"Olmo Hybrid's DeltaNet", Op{Kind: OpDeltaNet, Delta: DeltaNetOpParams{Eps: 1e-6, ONormEps: 1e-5, NegEigval: true}},
			[]string{"widened write gate (NegEigval)", "own output-norm epsilon"}},
		{"a gated attention outside on three parameters", Op{Kind: OpGatedAttn, Gated: GatedAttnParams{QKNorm: QKNormParams{Form: QKL2},
			Rope: RopeParams{Kernel: RopeMLAKernel}, Attn: AttentionParams{Chunk: 8192}}},
			[]string{"parameter-free L2 QK-norm", "de-interleaving (mlaRope) rotation", "chunked attention"}},
	} {
		gaps := Metal.Gaps(layer(c.op))
		if len(gaps) != len(c.want) {
			t.Errorf("%s: %d gaps %q, want %d", c.name, len(gaps), gaps, len(c.want))
			continue
		}
		for i, w := range c.want {
			if !strings.Contains(gaps[i], w) {
				t.Errorf("%s: gap %d is %q, want it to name %q", c.name, i, gaps[i], w)
			}
		}
	}
}

// TestSupport_pairwiseRotationIsATableField (Gemini on #548): GPT-J pairwise rotation is a rope parameter a table states, so a
// table without it declines a pairwise rotation (plain and inside gated attention) and keeps the NeoX one.
func TestSupport_pairwiseRotationIsATableField(t *testing.T) {
	neox := *Metal
	neox.PairwiseRope = false
	pair := Op{Kind: OpRope, Rope: RopeParams{Kernel: RopeAtKernel, Interleave: true}}
	gatedPair := Op{Kind: OpGatedAttn, Gated: GatedAttnParams{QKNorm: QKNormParams{Form: QKRMS}, Rope: RopeParams{Kernel: RopeAtKernel, Interleave: true}}}
	for _, o := range []Op{pair, gatedPair} {
		if gaps := neox.Gaps(layer(o)); len(gaps) != 1 || !strings.Contains(gaps[0], "pairwise rotation") {
			t.Errorf("%v with pairwise rotation on a NeoX-only table: gaps %q, want the pairwise gap", o.Kind, gaps)
		}
		if gaps := Metal.Gaps(layer(o)); len(gaps) != 0 {
			t.Errorf("%v with pairwise rotation on Metal, which has the pairwise kernels: gaps %q", o.Kind, gaps)
		}
	}
	if gaps := neox.Gaps(layer(Op{Kind: OpRope, Rope: RopeParams{Kernel: RopeAtKernel}})); len(gaps) != 0 {
		t.Errorf("a NeoX rotation on a NeoX-only table: gaps %q", gaps)
	}
}

// TestSupport_metalStepDeclinesEachPlacement holds each placement and form check to a case on the multi-sequence step's table
// (MetalStep), which implements none of them, while decode's table (Metal) admits every one; and the operations the step has
// no step for decline by kind.
func TestSupport_metalStepDeclinesEachPlacement(t *testing.T) {
	attn := []Op{{Kind: OpNorm, Norm: NormParams{Slot: SlotPreAttn, Form: NormRMS}}, {Kind: OpProjQKV}, {Kind: OpAttention}, {Kind: OpProjO},
		{Kind: OpResidual}}
	with := func(i int, o Op) *Program {
		ops := append([]Op(nil), attn...)
		ops[i] = o
		return layer(ops...)
	}
	for _, c := range []struct {
		name   string
		p      *Program
		want   string
		byKind bool // a kind or form the step lacks, which decode's table may lack too
	}{
		{"a sandwich norm", layer(append(append([]Op(nil), attn[:4]...), Op{Kind: OpNorm, Norm: NormParams{Slot: SlotPostAttn, Form: NormRMS}}, attn[4])...),
			"a norm on a sublayer's output", false},
		{"post-only placement", with(0, Op{Kind: OpNorm, Norm: NormParams{Slot: SlotPreAttn, Form: NormCopy}}), "post-only placement", false},
		{"a parallel block's residual", with(4, Op{Kind: OpResidual, Res: ResidualParams{Two: true}}), "residual of two outputs", false},
		{"a layer's LayerNorm", with(0, Op{Kind: OpNorm, Norm: NormParams{Slot: SlotPreAttn, Form: NormLayer}}), "does not implement LayerNorm", false},
		{"an output bias", with(3, Op{Kind: OpProjO, Proj: ProjParams{Bias: true}}), "output projection bias", false},
		{"a whole-vector QK-norm", layer(attn[0], attn[1], Op{Kind: OpQKNorm, QKNorm: QKNormParams{Form: QKRMS, Whole: true}}, attn[2], attn[3], attn[4]),
			"whole-vector QK-norm", false},
		{"a windowed attention", with(2, Op{Kind: OpAttention, Attn: AttentionParams{Window: 4096}}), "windowed attention (4096)", false},
		{"an attention softcap", with(2, Op{Kind: OpAttention, Attn: AttentionParams{Softcap: 50}}), "attention softcap (50)", false},
		{"learned positions", &Program{Embed: EmbedParams{LearnedPos: true}, Layers: [][]Op{attn}}, "learned position embeddings", false},
		{"the head's LayerNorm", &Program{Head: HeadParams{Norm: NormParams{Form: NormLayer}}, Layers: [][]Op{attn}}, "the head", false},
		{"gated attention", layer(attn[0], Op{Kind: OpGatedAttn, Gated: GatedAttnParams{QKNorm: QKNormParams{Form: QKRMS}, Rope: RopeParams{Kernel: RopeAtKernel}}}, attn[4]),
			"has no step for it", true},
		{"a DeltaNet mixer", layer(attn[0], Op{Kind: OpDeltaNet}, attn[4]), "has no step for it", true},
		{"a non-gated FFN", layer(attn[0], Op{Kind: OpFFN, FFN: FFNParams{Form: FfnNonGated, Act: ActSiLU}}, attn[4]), "FFN form non-gated", true},
	} {
		gaps := MetalStep.Gaps(c.p)
		if len(gaps) != 1 || !strings.Contains(gaps[0], c.want) {
			t.Errorf("%s: the step's gaps %q, want one naming %q", c.name, gaps, c.want)
		}
		if err := MetalStep.Check(c.p); err == nil {
			t.Errorf("%s: the step's Check admits it", c.name)
		}
		if g := Metal.Gaps(c.p); !c.byKind && len(g) != 0 {
			t.Errorf("%s: decode's table, which implements it, gives gaps %q", c.name, g)
		}
	}
	if g := MetalStep.Gaps(layer(attn...)); len(g) != 0 {
		t.Errorf("the plain pre-norm attention sublayer: the step's gaps %q, want none", g)
	}
	// A LayerNorm model names the form once, at its first layer, not again at the head (Gemini on #554).
	ln := &Program{Head: HeadParams{Norm: NormParams{Form: NormLayer}}, Layers: [][]Op{{
		{Kind: OpNorm, Norm: NormParams{Slot: SlotPreAttn, Form: NormLayer}}, attn[1], attn[2], attn[3], attn[4]}}}
	if g := MetalStep.Gaps(ln); len(g) != 1 {
		t.Errorf("a LayerNorm model: the step's gaps %q, want LayerNorm named once", g)
	}
}
