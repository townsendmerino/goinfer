//go:build realckpt

// G1 (docs/task-families-2026-09.md, batch 2): GGUF-vs-safetensors weightDiff for the DENSE
// qwen3_5 hybrid (Qwen3.8-27B, llama.cpp arch "qwen35") — the T3 parity item the MoE sibling
// already has (qwen35_gguf_weightdiff_test.go) but the dense one didn't. Same method, same
// reason it needs no HF oracle: the safetensors loader is already Gate-1 bit-exact vs HF, so it
// is the reference, and every TRANSFORM-bearing tensor (the V-head un-tile, the fused q‖gate
// q_proj, the −exp(A_log) bake, the (1+w) norm un-bake) is diffed directly against it. A correct
// GGUF loader lands at the dequant floor for each tensor's OWN source quant; a transform bug
// craters one tensor's cosine and names it.
//
// THE FLOOR IS PER TENSOR, AND THAT IS THE WHOLE POINT — it was not, and the gate was wrong
// for it. A single 0.999 bar was inherited from the MoE sibling, whose asset is a uniform
// Q8_0 file. This one's asset is unsloth's UD-Q4_K_M, a DYNAMIC quant carrying nine ggml
// types chosen per tensor by sensitivity, and under one whole-file bar the gate stopped being
// a statement about the loader: it became a statement about whichever tensor the quantizer
// spent the fewest bits on. It duly failed on its first-ever execution (2026-09-06) at
// k_proj 0.997047 / in_proj_z 0.996974, printing "loader transform bug" — and those are
// exactly, and only, the Q4_K tensors (blk.3.attn_k, blk.{1,2}.attn_gate), sitting where the
// first-principles Q4_K dequant estimate of ~0.9967 says they should, while their Q5_K and
// Q6_K siblings in the same layer cleared the bar. Nothing was wrong with the loader. See
// ggufQuantCosFloor (gguf_tensorquant_test.go) for the budgets and the arithmetic.
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
)

func TestQwen38GGUF_weightDiff(t *testing.T) {
	requireHeavyModel(t)
	dir := assetPath(t, "GOINFER_QWEN38") // safetensors (skips if absent)
	gguf := assetPath(t, "GOINFER_QWEN38_GGUF")

	const nLayer = 4
	prev := runtime.GOMAXPROCS(2)
	mRef, freeRef := loadQwen35Slice(t, dir, nLayer)
	wRef := mRef.w
	if wRef.arch.Name != "qwen3_5" || wRef.arch.MoE != nil {
		freeRef()
		runtime.GOMAXPROCS(prev)
		t.Fatalf("safetensors resolved arch %q (MoE=%v), want qwen3_5 dense", wRef.arch.Name, wRef.arch.MoE != nil)
	}
	gW, freeG := loadQwen35GGUFSlice(t, gguf, nLayer)
	runtime.GOMAXPROCS(prev)
	defer freeRef()
	defer freeG()
	if gW.arch.Name != "qwen3_5" || gW.arch.MoE != nil {
		t.Fatalf("GGUF resolved arch %q (MoE=%v), want qwen3_5 dense", gW.arch.Name, gW.arch.MoE != nil)
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

	for i := 0; i < nLayer; i++ {
		t.Logf("--- layer %d (%s) ---", i, layerKind(wRef.arch, i))
		p := fmt.Sprintf("blk.%d.", i) // the GGUF prefix loadQ35 uses, for the per-tensor quant lookup
		lr, lg := &wRef.Layers[i], &gW.Layers[i]
		check("attn_norm", p+"attn_norm.weight", lg.PreAttnNorm, lr.PreAttnNorm)
		check("post_attention_norm", p+"post_attention_norm.weight", lg.PreMLPNorm, lr.PreMLPNorm)
		if lr.delta != nil && lg.delta != nil {
			dr, dg := lr.delta, lg.delta
			check("in_proj_qkv", p+"attn_qkv.weight", wmDense(t, "in_proj_qkv", &dg.inProjQKV), wmDense(t, "in_proj_qkv", &dr.inProjQKV))
			check("in_proj_z", p+"attn_gate.weight", wmDense(t, "in_proj_z", &dg.inProjZ), wmDense(t, "in_proj_z", &dr.inProjZ))
			check("in_proj_a", p+"ssm_alpha.weight", dg.inProjA, dr.inProjA)
			check("in_proj_b", p+"ssm_beta.weight", dg.inProjB, dr.inProjB)
			check("conv1d", p+"ssm_conv1d.weight", dg.convW, dr.convW)
			check("dt_bias", p+"ssm_dt.bias", dg.dtBias, dr.dtBias)
			check("negExpA", p+"ssm_a", dg.negExpA, dr.negExpA)
			check("ssm_norm", p+"ssm_norm.weight", dg.normW, dr.normW)
			check("out_proj", p+"ssm_out.weight", wmDense(t, "out_proj", &dg.outProj), wmDense(t, "out_proj", &dr.outProj))
		} else if lr.qattn != nil && lg.qattn != nil {
			ar, ag := lr.qattn, lg.qattn
			check("q_proj(query‖gate)", p+"attn_q.weight", wmDense(t, "q_proj", &ag.qProj), wmDense(t, "q_proj", &ar.qProj))
			check("k_proj", p+"attn_k.weight", wmDense(t, "k_proj", &ag.kProj), wmDense(t, "k_proj", &ar.kProj))
			check("v_proj", p+"attn_v.weight", wmDense(t, "v_proj", &ag.vProj), wmDense(t, "v_proj", &ar.vProj))
			check("o_proj", p+"attn_output.weight", wmDense(t, "o_proj", &ag.oProj), wmDense(t, "o_proj", &ar.oProj))
			check("q_norm", p+"attn_q_norm.weight", ag.qNorm, ar.qNorm)
			check("k_norm", p+"attn_k_norm.weight", ag.kNorm, ar.kNorm)
		} else {
			t.Errorf("layer %d: attn kind mismatch (ref delta=%v qattn=%v / gguf delta=%v qattn=%v)",
				i, lr.delta != nil, lr.qattn != nil, lg.delta != nil, lg.qattn != nil)
		}
	}
	t.Logf("=== weightDiff: worst tensor used %.0f%% of its own dequant budget — %s (%s) — over layers 0-%d ===",
		100*worstUsed, worstName, worstDetail, nLayer-1)
}
