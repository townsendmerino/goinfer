//go:build realckpt

// G1: GGUF-vs-safetensors weightDiff for the DENSE
// qwen3_5 hybrid (Qwen3.8-27B, llama.cpp arch "qwen35") — the T3 parity item the MoE sibling
// already has (qwen35_gguf_weightdiff_test.go) but the dense one didn't. Same method, same
// reason it needs no HF oracle: the safetensors loader is already Gate-1 bit-exact vs HF, so it
// is the reference, and every TRANSFORM-bearing tensor (the V-head un-tile, the fused q‖gate
// q_proj, the −exp(A_log) bake, the (1+w) norm un-bake) is diffed directly against it. A correct
// GGUF loader lands at the dequant floor for each tensor's OWN source quant; a transform bug
// craters one tensor's cosine and names it.
//
// THE FLOOR IS PER TENSOR, AND THAT IS THE WHOLE POINT. A single whole-file bar (inherited from the MoE sibling, whose asset
// is a uniform Q8_0 file) does not fit this asset: unsloth's UD-Q4_K_M is a DYNAMIC quant carrying nine ggml types chosen
// per tensor by sensitivity, so under one bar the gate stops measuring the loader and measures whichever tensor the quantizer
// spent the fewest bits on (Q4_K tensors sit near cosine 0.9967 by first-principles estimate, below a 0.999 bar). Each tensor
// is held to the dequant floor of its OWN source quant; see ggufQuantCosFloor (gguf_tensorquant_test.go) for the budgets and
// the arithmetic.
//
// Unlike the MoE sibling, this checkpoint has no router — TestQwen35GGUF_weightDiff's router
// checks are no-ops here (lr.Router.Rows() == 0 on both sides) rather than removed, so the two
// tests stay structurally comparable.
//
//	GOINFER_QWEN38=~/models/qwen3.8-27b \
//	GOINFER_QWEN38_GGUF=~/models/qwen38-gguf/Qwen3.8-27B-UD-Q4_K_M.gguf \
//	  go test -tags realckpt ./decoder/ -run TestQwen38GGUF_weightDiff -v -timeout 15m
package decoder

import (
	"fmt"
	"math"
	"runtime"
	"testing"

	"github.com/townsendmerino/goinfer/internal/testgate"
)

func TestQwen38GGUF_weightDiff(t *testing.T) {
	testgate.RequireHeavy(t)
	dir := assetPath(t, "GOINFER_QWEN38") // safetensors (skips if absent)
	gguf := assetPath(t, "GOINFER_QWEN38_GGUF")

	const nLayer = 4
	prev := runtime.GOMAXPROCS(2)
	mRef, freeRef := loadQwen35Slice(t, dir, nLayer)
	wRef := mRef.w
	if wRef.Arch.Name != "qwen3_5" || wRef.Arch.MoE != nil {
		freeRef()
		runtime.GOMAXPROCS(prev)
		t.Fatalf("safetensors resolved arch %q (MoE=%v), want qwen3_5 dense", wRef.Arch.Name, wRef.Arch.MoE != nil)
	}
	gW, freeG := loadQwen35GGUFSlice(t, gguf, nLayer)
	runtime.GOMAXPROCS(prev)
	defer freeRef()
	defer freeG()
	if gW.Arch.Name != "qwen3_5" || gW.Arch.MoE != nil {
		t.Fatalf("GGUF resolved arch %q (MoE=%v), want qwen3_5 dense", gW.Arch.Name, gW.Arch.MoE != nil)
	}

	quants, err := ggufTensorQuants(gguf)
	if err != nil {
		t.Fatalf("ggufTensorQuants: %v", err)
	}

	// The summary ranks tensors by the FRACTION of their own dequant budget consumed,
	// (1-cos)/(1-floor), not by raw cosine and not by absolute margin. Raw cosine just names
	// whichever tensor the quantizer spent the fewest bits on, which is the reading that made
	// this gate misfire in the first place. Absolute margin is no better in the other
	// direction: an F32 tensor is bit-exact and its floor is 1-1e-6, so it has a millionth of
	// headroom by construction and would always win a contest it is never at risk of losing.
	// The fraction is comparable across all nine quants in the file: >1 is the failure.
	worstUsed, worstName, worstDetail := -1.0, "", ""
	check := func(name, ggufName string, got, ref []float32) {
		q, ok := quants[ggufName]
		if !ok {
			t.Errorf("%s: no tensor %q in %s — the GGUF name this gate diffs against must exist, "+
				"or the floor is being chosen for a tensor that was never read", name, ggufName, gguf)
			return
		}
		floor := ggufQuantCosFloor(q)
		cos, maxAbs, relL2 := tensorAgreement(got, ref)
		used := math.Inf(1)
		if floor < 1 {
			used = (1 - cos) / (1 - floor)
		}
		t.Logf("  %-28s %-7s floor=%-8.6g budget=%4.0f%% len=%-9d cos=%.6f maxAbs=%.4g relL2=%.4g",
			name, q, floor, 100*used, len(ref), cos, maxAbs, relL2)
		if used > worstUsed {
			worstUsed, worstName = used, name
			worstDetail = fmt.Sprintf("cos=%.6f vs %s floor %.6g", cos, q, floor)
		}
		if cos < floor {
			t.Errorf("%s cosine %.6f < %.6g — GGUF loader transform bug. The floor is this "+
				"tensor's own %s dequant budget, so quantization noise does not reach it; a "+
				"transform error (un-tile order, a missing norm un-bake, a sign) craters far "+
				"below it.", name, cos, floor, q)
		}
	}

	for i := range nLayer {
		t.Logf("--- layer %d (%s) ---", i, layerKind(wRef.Arch, i))
		p := fmt.Sprintf("blk.%d.", i) // the GGUF prefix loadQ35 uses, for the per-tensor quant lookup
		lr, lg := &wRef.Layers[i], &gW.Layers[i]
		check("attn_norm", p+"attn_norm.weight", lg.PreAttnNorm, lr.PreAttnNorm)
		check("post_attention_norm", p+"post_attention_norm.weight", lg.PreMLPNorm, lr.PreMLPNorm)
		if lr.Delta != nil && lg.Delta != nil {
			dr, dg := lr.Delta, lg.Delta
			check("in_proj_qkv", p+"attn_qkv.weight", wmDense(t, "in_proj_qkv", &dg.InProjQKV), wmDense(t, "in_proj_qkv", &dr.InProjQKV))
			check("in_proj_z", p+"attn_gate.weight", wmDense(t, "in_proj_z", &dg.InProjZ), wmDense(t, "in_proj_z", &dr.InProjZ))
			check("in_proj_a", p+"ssm_alpha.weight", dg.InProjA, dr.InProjA)
			check("in_proj_b", p+"ssm_beta.weight", dg.InProjB, dr.InProjB)
			check("conv1d", p+"ssm_conv1d.weight", dg.ConvW, dr.ConvW)
			check("dt_bias", p+"ssm_dt.bias", dg.DtBias, dr.DtBias)
			check("negExpA", p+"ssm_a", dg.NegExpA, dr.NegExpA)
			check("ssm_norm", p+"ssm_norm.weight", dg.NormW, dr.NormW)
			check("out_proj", p+"ssm_out.weight", wmDense(t, "out_proj", &dg.OutProj), wmDense(t, "out_proj", &dr.OutProj))
		} else if lr.QAttn != nil && lg.QAttn != nil {
			ar, ag := lr.QAttn, lg.QAttn
			check("q_proj(query‖gate)", p+"attn_q.weight", wmDense(t, "q_proj", &ag.QProj), wmDense(t, "q_proj", &ar.QProj))
			check("k_proj", p+"attn_k.weight", wmDense(t, "k_proj", &ag.KProj), wmDense(t, "k_proj", &ar.KProj))
			check("v_proj", p+"attn_v.weight", wmDense(t, "v_proj", &ag.VProj), wmDense(t, "v_proj", &ar.VProj))
			check("o_proj", p+"attn_output.weight", wmDense(t, "o_proj", &ag.OProj), wmDense(t, "o_proj", &ar.OProj))
			check("q_norm", p+"attn_q_norm.weight", ag.QNorm, ar.QNorm)
			check("k_norm", p+"attn_k_norm.weight", ag.KNorm, ar.KNorm)
		} else {
			t.Errorf("layer %d: attn kind mismatch (ref delta=%v qattn=%v / gguf delta=%v qattn=%v)",
				i, lr.Delta != nil, lr.QAttn != nil, lg.Delta != nil, lg.QAttn != nil)
		}
	}
	t.Logf("=== weightDiff: worst tensor used %.0f%% of its own dequant budget — %s (%s) — over layers 0-%d ===",
		100*worstUsed, worstName, worstDetail, nLayer-1)
}
