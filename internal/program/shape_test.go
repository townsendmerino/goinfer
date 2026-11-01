package program

import (
	"strings"
	"testing"
)

// The pure tests of ShapeOf, moved from metal's resident_memguard_test.go with the derivation (OP4cu-b1,
// docs/tasks/task-forward-architecture-2026-10.md §4.9a); their assertions are unchanged.

// shapeLayer builds one layer of a hand-written program: an input norm (a copy for post-only), q/k/v with an optional QK-norm,
// attention or a recurrent mixer, the post-attention norm for sandwich, the output projection, and an FFN, with a residual add of
// two outputs for a parallel block.
func shapeLayer(placement string, recurrent, qk, layerNorm, nonGated, outBias bool) []Op {
	form := NormRMS
	if layerNorm {
		form = NormLayer
	}
	in := Op{Kind: OpNorm, Norm: NormParams{Slot: SlotPreAttn, Form: form}}
	if placement == "post-only" {
		in.Norm.Form = NormCopy
	}
	ops := []Op{in}
	if recurrent {
		ops = append(ops, Op{Kind: OpDeltaNet})
	} else {
		ops = append(ops, Op{Kind: OpProjQKV})
		if qk {
			ops = append(ops, Op{Kind: OpQKNorm, QKNorm: QKNormParams{Form: QKRMS}})
		}
		ops = append(ops, Op{Kind: OpAttention}, Op{Kind: OpProjO, Proj: ProjParams{Bias: outBias}})
	}
	ffn := Op{Kind: OpFFN, FFN: FFNParams{Form: FfnGated}}
	if nonGated {
		ffn.FFN.Form = FfnNonGated
	}
	switch placement {
	case "parallel":
		return append(ops, ffn, Op{Kind: OpResidual, Res: ResidualParams{Two: true}})
	case "sandwich":
		ops = append(ops, Op{Kind: OpNorm, Norm: NormParams{Slot: SlotPostAttn, Form: form}})
	}
	return append(ops, Op{Kind: OpResidual}, ffn, Op{Kind: OpResidual})
}

// TestShapeOf_declinesLayersThatDisagree (OP4m-b1, Gemini's review of #539): the encoder takes each part of the shape at
// model level, so a program whose attention layers disagree on the placement, or whose layers disagree on the norm's form, the
// FFN's form, the QK-norm or the output bias, declines and names both layers. A recurrent-mixer layer may stay pre-norm beside
// post-only attention layers (Olmo Hybrid), which the encoder handles per layer.
func TestShapeOf_declinesLayersThatDisagree(t *testing.T) {
	pre := shapeLayer("pre", false, false, false, false, false)
	for _, c := range []struct {
		name   string
		layers [][]Op
	}{
		{"pre then sandwich", [][]Op{pre, shapeLayer("sandwich", false, false, false, false, false)}},
		{"pre then post-only", [][]Op{pre, shapeLayer("post-only", false, false, false, false, false)}},
		{"pre then parallel", [][]Op{pre, shapeLayer("parallel", false, false, false, false, false)}},
		{"RMS then LayerNorm", [][]Op{pre, shapeLayer("pre", false, false, true, false, false)}},
		{"gated then non-gated FFN", [][]Op{pre, shapeLayer("pre", false, false, false, true, false)}},
		{"without then with QK-norm", [][]Op{pre, shapeLayer("pre", false, true, false, false, false)}},
		{"without then with output bias", [][]Op{pre, shapeLayer("pre", false, false, false, false, true)}},
	} {
		if _, err := ShapeOf(&Program{Layers: c.layers}); err == nil {
			t.Errorf("%s: no error, want a decline naming both layers", c.name)
		} else if !strings.Contains(err.Error(), "layer 1") || !strings.Contains(err.Error(), "layer 0") {
			t.Errorf("%s: %v does not name both layers", c.name, err)
		}
	}
	hybrid := &Program{Layers: [][]Op{shapeLayer("post-only", false, false, false, false, false), shapeLayer("pre", true, false, false, false, false)}}
	s, err := ShapeOf(hybrid)
	if err != nil {
		t.Fatalf("post-only attention beside a pre-norm recurrent layer (Olmo Hybrid): %v", err)
	}
	if !s.PostOnly || s.Sandwich || s.Parallel {
		t.Fatalf("Olmo Hybrid's shape: post-only %v, sandwich %v, parallel %v; want post-only alone", s.PostOnly, s.Sandwich, s.Parallel)
	}
}

// TestShapeOf_acceptsOnlyAgreementOrOlmoHybrid (OP4m-b1, Gemini's re-read of #539): every layer's placement and QK-norm is
// recorded, and a program is accepted only when every layer's placement agrees, or for the one mix the encoder handles per
// layer, Olmo Hybrid's post-only attention layers beside pre-norm recurrent layers. A layer that asks for two placements at once
// declines, and so does a layer with both an attention and a recurrent mixer whose QK-norm disagrees with another layer's.
func TestShapeOf_acceptsOnlyAgreementOrOlmoHybrid(t *testing.T) {
	ffnOnly := func(placement string) []Op {
		l := shapeLayer(placement, false, false, false, false, false)
		var out []Op
		for _, o := range l {
			if o.Kind != OpProjQKV && o.Kind != OpQKNorm && o.Kind != OpAttention && o.Kind != OpProjO {
				out = append(out, o)
			}
		}
		return out
	}
	both := func(qk bool) []Op { // a layer with an attention and a recurrent mixer
		l := shapeLayer("pre", false, qk, false, false, false)
		return append(l[:1:1], append([]Op{{Kind: OpDeltaNet}}, l[1:]...)...)
	}
	twoPlacements := func() []Op { // a parallel block that also carries a post-attention norm
		l := shapeLayer("parallel", false, false, false, false, false)
		return append(l[:len(l)-2:len(l)-2], append([]Op{{Kind: OpNorm, Norm: NormParams{Slot: SlotPostAttn, Form: NormRMS}}}, l[len(l)-2:]...)...)
	}
	for _, c := range []struct {
		name   string
		layers [][]Op
	}{
		{"sandwich attention beside a pre-norm recurrent layer", [][]Op{shapeLayer("sandwich", false, false, false, false, false), shapeLayer("pre", true, false, false, false, false)}},
		{"parallel attention beside a pre-norm FFN-only layer", [][]Op{shapeLayer("parallel", false, false, false, false, false), ffnOnly("pre")}},
		{"post-only attention beside a pre-norm FFN-only layer", [][]Op{shapeLayer("post-only", false, false, false, false, false), ffnOnly("pre")}},
	} {
		if _, err := ShapeOf(&Program{Layers: c.layers}); err == nil {
			t.Errorf("%s: no error, want a decline naming both layers", c.name)
		} else if !strings.Contains(err.Error(), "layer 1") || !strings.Contains(err.Error(), "layer 0") {
			t.Errorf("%s: %v does not name both layers", c.name, err)
		}
	}
	if _, err := ShapeOf(&Program{Layers: [][]Op{twoPlacements()}}); err == nil || !strings.Contains(err.Error(), "two placements at once") {
		t.Errorf("a layer asking for parallel and sandwich at once: %v, want the two-placements decline", err)
	}
	// Gemini on #540: a layer with both mixers declines whatever the placement and QK-norm, every layer agreeing included.
	for _, c := range []struct {
		name   string
		layers [][]Op
	}{
		{"attention+recurrent layers, every layer pre-norm", [][]Op{both(false), both(false)}},
		{"attention+recurrent layers with QK-norm", [][]Op{both(true), both(true)}},
		{"an attention+recurrent layer beside one with QK-norm", [][]Op{shapeLayer("pre", false, true, false, false, false), both(false)}},
	} {
		if _, err := ShapeOf(&Program{Layers: c.layers}); err == nil || !strings.Contains(err.Error(), "both an attention and a recurrent mixer") {
			t.Errorf("%s: %v, want the both-mixers decline", c.name, err)
		}
	}
	// A failed mix names the layer that fits neither half: here layer 0, a pre-norm attention layer, not layer 1.
	mixed := [][]Op{shapeLayer("pre", false, false, false, false, false), shapeLayer("post-only", false, false, false, false, false), shapeLayer("pre", true, false, false, false, false)}
	if _, err := ShapeOf(&Program{Layers: mixed}); err == nil || !strings.Contains(err.Error(), "layer 0 (an attention layer, pre-norm) is not") {
		t.Errorf("pre-norm attention, post-only attention and pre-norm recurrent layers: %v, want layer 0 named as fitting neither half", err)
	}
	nemotronLike := &Program{Layers: [][]Op{shapeLayer("pre", false, false, false, false, false), shapeLayer("pre", true, false, false, false, false), ffnOnly("pre")}}
	if _, err := ShapeOf(nemotronLike); err != nil {
		t.Errorf("pre-norm attention, recurrent and FFN-only layers (Nemotron-H's shape): %v", err)
	}
}

// TestShapeOf_placementOnlyOnAttentionLayers (OP4m-b2, Gemini on #540): the encoder applies a non-pre-norm placement to
// attention layers alone, so a recurrent or FFN-only layer carrying one declines, naming the layer, even when every layer's
// placement agrees.
func TestShapeOf_placementOnlyOnAttentionLayers(t *testing.T) {
	ffnOnly := func(placement string) []Op {
		var out []Op
		for _, o := range shapeLayer(placement, false, false, false, false, false) {
			if o.Kind != OpProjQKV && o.Kind != OpAttention && o.Kind != OpProjO {
				out = append(out, o)
			}
		}
		return out
	}
	declines := func(name string, layers [][]Op) {
		if _, err := ShapeOf(&Program{Layers: layers}); err == nil {
			t.Errorf("%s: no error, want a decline", name)
		} else if !strings.Contains(err.Error(), "layer 1") {
			t.Errorf("%s: %v does not name layer 1", name, err)
		}
	}
	for _, placement := range []string{"post-only", "sandwich", "parallel"} {
		attn := shapeLayer(placement, false, false, false, false, false)
		declines("every layer "+placement+", layer 1 a recurrent layer", [][]Op{attn, shapeLayer(placement, true, false, false, false, false)})
		declines("every layer "+placement+", layer 1 an FFN-only layer", [][]Op{attn, ffnOnly(placement)})
		if _, err := ShapeOf(&Program{Layers: [][]Op{attn, attn}}); err != nil {
			t.Errorf("every layer %s attention: %v", placement, err)
		}
	}
	declines("a sandwich recurrent layer beside pre attention", [][]Op{shapeLayer("pre", false, false, false, false, false), shapeLayer("sandwich", true, false, false, false, false)})
	declines("recurrent only, pre then sandwich", [][]Op{shapeLayer("pre", true, false, false, false, false), shapeLayer("sandwich", true, false, false, false, false)})
}
