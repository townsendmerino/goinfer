package metal

import (
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/program"
)

// shapeLayer builds one layer of a hand-written program: an input norm (a copy for post-only), q/k/v with an optional QK-norm,
// attention or a recurrent mixer, the post-attention norm for sandwich, the output projection, and an FFN, with a residual add of
// two outputs for a parallel block.
func shapeLayer(placement string, recurrent, qk, layerNorm, nonGated, outBias bool) []program.Op {
	form := program.NormRMS
	if layerNorm {
		form = program.NormLayer
	}
	in := program.Op{Kind: program.OpNorm, Norm: program.NormParams{Slot: program.SlotPreAttn, Form: form}}
	if placement == "post-only" {
		in.Norm.Form = program.NormCopy
	}
	ops := []program.Op{in}
	if recurrent {
		ops = append(ops, program.Op{Kind: program.OpDeltaNet})
	} else {
		ops = append(ops, program.Op{Kind: program.OpProjQKV})
		if qk {
			ops = append(ops, program.Op{Kind: program.OpQKNorm, QKNorm: program.QKNormParams{Form: program.QKRMS}})
		}
		ops = append(ops, program.Op{Kind: program.OpAttention}, program.Op{Kind: program.OpProjO, Proj: program.ProjParams{Bias: outBias}})
	}
	ffn := program.Op{Kind: program.OpFFN, FFN: program.FFNParams{Form: program.FfnGated}}
	if nonGated {
		ffn.FFN.Form = program.FfnNonGated
	}
	switch placement {
	case "parallel":
		return append(ops, ffn, program.Op{Kind: program.OpResidual, Res: program.ResidualParams{Two: true}})
	case "sandwich":
		ops = append(ops, program.Op{Kind: program.OpNorm, Norm: program.NormParams{Slot: program.SlotPostAttn, Form: form}})
	}
	return append(ops, program.Op{Kind: program.OpResidual}, ffn, program.Op{Kind: program.OpResidual})
}

// TestDenseShapeOf_declinesLayersThatDisagree (OP4m-b1, Gemini's review of #539): the encoder takes each part of the shape at
// model level, so a program whose attention layers disagree on the placement, or whose layers disagree on the norm's form, the
// FFN's form, the QK-norm or the output bias, declines and names both layers. A recurrent-mixer layer may stay pre-norm beside
// post-only attention layers (Olmo Hybrid), which the encoder handles per layer.
func TestDenseShapeOf_declinesLayersThatDisagree(t *testing.T) {
	pre := shapeLayer("pre", false, false, false, false, false)
	for _, c := range []struct {
		name   string
		layers [][]program.Op
	}{
		{"pre then sandwich", [][]program.Op{pre, shapeLayer("sandwich", false, false, false, false, false)}},
		{"pre then post-only", [][]program.Op{pre, shapeLayer("post-only", false, false, false, false, false)}},
		{"pre then parallel", [][]program.Op{pre, shapeLayer("parallel", false, false, false, false, false)}},
		{"RMS then LayerNorm", [][]program.Op{pre, shapeLayer("pre", false, false, true, false, false)}},
		{"gated then non-gated FFN", [][]program.Op{pre, shapeLayer("pre", false, false, false, true, false)}},
		{"without then with QK-norm", [][]program.Op{pre, shapeLayer("pre", false, true, false, false, false)}},
		{"without then with output bias", [][]program.Op{pre, shapeLayer("pre", false, false, false, false, true)}},
		{"a sandwich recurrent layer beside pre attention", [][]program.Op{pre, shapeLayer("sandwich", true, false, false, false, false)}},
	} {
		if _, err := denseShapeOf(&program.Program{Layers: c.layers}); err == nil {
			t.Errorf("%s: no error, want a decline naming both layers", c.name)
		} else if !strings.Contains(err.Error(), "layer 1") || !strings.Contains(err.Error(), "layer 0") {
			t.Errorf("%s: %v does not name both layers", c.name, err)
		}
	}
	hybrid := &program.Program{Layers: [][]program.Op{shapeLayer("post-only", false, false, false, false, false), shapeLayer("pre", true, false, false, false, false)}}
	s, err := denseShapeOf(hybrid)
	if err != nil {
		t.Fatalf("post-only attention beside a pre-norm recurrent layer (Olmo Hybrid): %v", err)
	}
	if !s.postOnly || s.sandwich || s.parallel {
		t.Fatalf("Olmo Hybrid's shape: post-only %v, sandwich %v, parallel %v; want post-only alone", s.postOnly, s.sandwich, s.parallel)
	}
}
