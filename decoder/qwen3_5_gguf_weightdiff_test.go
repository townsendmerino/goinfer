//go:build realckpt

// G1 (docs/task-families-2026-09.md, batch 2): GGUF-vs-safetensors weightDiff for the DENSE
// qwen3_5 hybrid (Qwen3.8-27B, llama.cpp arch "qwen35") — the T3 parity item the MoE sibling
// already has (qwen35_gguf_weightdiff_test.go) but the dense one didn't. Same method, same
// reason it needs no HF oracle: the safetensors loader is already Gate-1 bit-exact vs HF, so it
// is the reference, and every TRANSFORM-bearing tensor (the V-head un-tile, the fused q‖gate
// q_proj, the −exp(A_log) bake, the (1+w) norm un-bake) is diffed directly against it. A correct
// GGUF loader lands at the Q8_0-vs-bf16 dequant floor (cosine ≳ 0.999); a transform bug craters
// one tensor's cosine and names it.
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

	const cosFloor = 0.999
	worstCos, worstName := 1.0, ""
	check := func(name string, got, ref []float32) {
		cos, maxAbs, relL2 := tensorAgreement(got, ref)
		t.Logf("  %-28s len=%-9d cos=%.6f maxAbs=%.4g relL2=%.4g", name, len(ref), cos, maxAbs, relL2)
		if cos < worstCos {
			worstCos, worstName = cos, name
		}
		if cos < cosFloor {
			t.Errorf("%s cosine %.6f < %.3f — GGUF loader transform bug (not Q8_0/Q4_K quant)", name, cos, cosFloor)
		}
	}

	for i := 0; i < nLayer; i++ {
		t.Logf("--- layer %d (%s) ---", i, layerKind(wRef.arch, i))
		lr, lg := &wRef.Layers[i], &gW.Layers[i]
		check("attn_norm", lg.PreAttnNorm, lr.PreAttnNorm)
		check("post_attention_norm", lg.PreMLPNorm, lr.PreMLPNorm)
		if lr.delta != nil && lg.delta != nil {
			dr, dg := lr.delta, lg.delta
			check("in_proj_qkv", wmDense(t, "in_proj_qkv", &dg.inProjQKV), wmDense(t, "in_proj_qkv", &dr.inProjQKV))
			check("in_proj_z", wmDense(t, "in_proj_z", &dg.inProjZ), wmDense(t, "in_proj_z", &dr.inProjZ))
			check("in_proj_a", dg.inProjA, dr.inProjA)
			check("in_proj_b", dg.inProjB, dr.inProjB)
			check("conv1d", dg.convW, dr.convW)
			check("dt_bias", dg.dtBias, dr.dtBias)
			check("negExpA", dg.negExpA, dr.negExpA)
			check("ssm_norm", dg.normW, dr.normW)
			check("out_proj", wmDense(t, "out_proj", &dg.outProj), wmDense(t, "out_proj", &dr.outProj))
		} else if lr.qattn != nil && lg.qattn != nil {
			ar, ag := lr.qattn, lg.qattn
			check("q_proj(query‖gate)", wmDense(t, "q_proj", &ag.qProj), wmDense(t, "q_proj", &ar.qProj))
			check("k_proj", wmDense(t, "k_proj", &ag.kProj), wmDense(t, "k_proj", &ar.kProj))
			check("v_proj", wmDense(t, "v_proj", &ag.vProj), wmDense(t, "v_proj", &ar.vProj))
			check("o_proj", wmDense(t, "o_proj", &ag.oProj), wmDense(t, "o_proj", &ar.oProj))
			check("q_norm", ag.qNorm, ar.qNorm)
			check("k_norm", ag.kNorm, ar.kNorm)
		} else {
			t.Errorf("layer %d: attn kind mismatch (ref delta=%v qattn=%v / gguf delta=%v qattn=%v)",
				i, lr.delta != nil, lr.qattn != nil, lg.delta != nil, lg.qattn != nil)
		}
	}
	t.Logf("=== weightDiff: worst tensor cosine %.6f (%s) over layers 0-%d ===", worstCos, worstName, nLayer-1)
}
