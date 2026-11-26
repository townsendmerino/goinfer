//go:build realckpt

// Cheap loader-vs-quant disambiguator (no full-model forward, no 60-min oracle). The bf16-golden gate
// (qwen35_gguf_gate_test.go) cannot prove that its small misses are quant noise rather than a cratered loader; this does: it
// loads only the first 4 layers of BOTH containers at f32 (the existing loadQwen35Slice for safetensors, a sibling for the
// GGUF) and diffs every TRANSFORM-bearing tensor directly: the V-head un-tile, the q‖gate fused q_proj, the −exp(A_log) bake,
// the norm(+1) un-bake. The safetensors loader is Gate-1 bit-exact vs HF, so it is the reference. A correct GGUF loader ⇒
// each tensor matches to the Q8_0-vs-bf16 dequant delta (cosine ≳ 0.9995); a transform bug ⇒ that one tensor's cosine
// craters, pinpointing it. Slice 0–3 spans both layer kinds (DeltaNet 0,1,2 + softmax 3).
//
//	GOINFER_QWEN35_REAL=~/models/qwen3.6-35b-a3b \
//	GOINFER_QWEN35_GGUF=~/models/qwen3.6-35b-a3b-Q8_0.gguf \
//	  go test -tags realckpt ./decoder/ -run TestQwen35GGUF_weightDiff -v -timeout 15m
package decoder

import (
	"math"
	"runtime"
	"testing"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"

	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/testgate"
)

// loadQwen35GGUFSlice loads embed + the first nLayer layers of the qwen35moe GGUF
// at f32 (the DeltaNet/softmax tensors are always f32 on this path; experts go
// int8 since this probe never compares them). Truncating NumLayers makes the
// loader touch only blk.0..nLayer-1 — fast, a few GB vs the 37 GB file.
func loadQwen35GGUFSlice(t *testing.T, path string, nLayer int) (*darch.Weights, func()) {
	t.Helper()
	g, err := embed.OpenGGUFMmap(path)
	if err != nil {
		t.Fatalf("OpenGGUFMmap: %v", err)
	}
	cfg, err := ggufConfig(g)
	if err != nil {
		g.Close()
		t.Fatalf("ggufConfig: %v", err)
	}
	if cfg.NumLayers < nLayer {
		g.Close()
		t.Fatalf("GGUF has %d layers, slice wants %d", cfg.NumLayers, nLayer)
	}
	cfg.NumLayers = nLayer
	cfg.LayerTypes = cfg.LayerTypes[:nLayer]
	arch, _, err := resolveArchitecture(cfg)
	if err != nil {
		g.Close()
		t.Fatalf("resolveArchitecture: %v", err)
	}
	quant, _ := parseQuant("int8int8") // experts int8; since 6d4fc79 the attn/delta
	// projections honour this too, so the compared tensors are int8 and wmDense dequantizes them.
	w, err := buildWeightsFromGGUF(cfg, arch, g, quant, false, true, false, nil, "", nil)
	if err != nil {
		g.Close()
		t.Fatalf("buildWeightsFromGGUF (GGUF slice): %v", err)
	}
	return w, func() { g.Close(); runtime.GC() }
}

// tensorAgreement reports the per-tensor agreement between the GGUF and
// safetensors f32 reconstructions: full cosine + max-abs and relative-L2 deltas.
func tensorAgreement(got, ref []float32) (cos float64, maxAbs, relL2 float64) {
	if len(got) != len(ref) || len(ref) == 0 {
		return -1, math.Inf(1), math.Inf(1) // length mismatch is itself a loader bug
	}
	var dot, ng, nr, sd, sr float64
	for i := range ref {
		g, r := float64(got[i]), float64(ref[i])
		dot += g * r
		ng += g * g
		nr += r * r
		d := g - r
		sd += d * d
		sr += r * r
		if a := math.Abs(d); a > maxAbs {
			maxAbs = a
		}
	}
	cos = 1
	if ng > 0 && nr > 0 {
		cos = dot / math.Sqrt(ng*nr)
	}
	relL2 = math.Inf(1)
	if sr > 0 {
		relL2 = math.Sqrt(sd / sr)
	}
	return cos, maxAbs, relL2
}

func TestQwen35GGUF_weightDiff(t *testing.T) {
	testgate.RequireHeavy(t)
	dir := realQwen35Dir(t) // safetensors (skips if absent)
	gguf := assetPath(t, "GOINFER_QWEN35_GGUF")

	const nLayer = 4
	// Safetensors reference first (Gate-1 bit-exact vs HF), then the GGUF slice.
	prev := runtime.GOMAXPROCS(2)
	mRef, freeRef := loadQwen35Slice(t, dir, nLayer)
	wRef := mRef.w
	gW, freeG := loadQwen35GGUFSlice(t, gguf, nLayer)
	runtime.GOMAXPROCS(prev)
	defer freeRef()
	defer freeG()

	// The Q8_0-vs-bf16 dequant delta is the noise floor; a correct transform lands
	// well above it. A real transform bug (wrong un-tile/q‖gate/un-bake) craters
	// cosine far below this. Pinpoints the offending tensor by name.
	const cosFloor = 0.999
	worstCos, worstName := 1.0, ""
	check := func(name string, got, ref []float32) {
		cos, maxAbs, relL2 := tensorAgreement(got, ref)
		t.Logf("  %-28s len=%-9d cos=%.6f maxAbs=%.4g relL2=%.4g", name, len(ref), cos, maxAbs, relL2)
		if cos < worstCos {
			worstCos, worstName = cos, name
		}
		if cos < cosFloor {
			t.Errorf("%s cosine %.6f < %.3f — GGUF loader transform bug (not Q8_0 quant)", name, cos, cosFloor)
		}
	}

	for i := range nLayer {
		t.Logf("--- layer %d (%s) ---", i, layerKind(wRef.Arch, i))
		lr, lg := &wRef.Layers[i], &gW.Layers[i]
		check("attn_norm", lg.PreAttnNorm, lr.PreAttnNorm)
		check("post_attention_norm", lg.PreMLPNorm, lr.PreMLPNorm)
		// THE ROUTER is compared because it decides which experts run, and pervasive routing divergence with an undiffed router is
		// where a real loader defect could hide (the routing-flip experiment, qwen35_gguf_routeflip_test.go, found the two
		// containers choosing different top-8 sets in many (step,layer) pairs). Both loaders keep the router at f32 on purpose ("the
		// router is logit-critical", no quant), so this is a straight comparison: agreement at the same Q8_0 floor as every other
		// tensor means the flips are quant noise on near-tied experts; anything worse means the router itself is mis-read and the
		// flips are a defect.
		if lr.Router.Rows() > 0 || lg.Router.Rows() > 0 {
			check("router(mlp.gate)", wmDense(t, "router", &lg.Router), wmDense(t, "router", &lr.Router))
		}
		if lr.RouterBias != nil || lg.RouterBias != nil {
			check("router_bias", lg.RouterBias, lr.RouterBias)
		}
		if lr.Delta != nil && lg.Delta != nil {
			dr, dg := lr.Delta, lg.Delta
			check("in_proj_qkv", wmDense(t, "in_proj_qkv", &dg.InProjQKV), wmDense(t, "in_proj_qkv", &dr.InProjQKV)) // V-block un-tile
			check("in_proj_z", wmDense(t, "in_proj_z", &dg.InProjZ), wmDense(t, "in_proj_z", &dr.InProjZ))           // full un-tile
			check("in_proj_a", dg.InProjA, dr.InProjA)
			check("in_proj_b", dg.InProjB, dr.InProjB)
			check("conv1d", dg.ConvW, dr.ConvW) // V-channel un-tile
			check("dt_bias", dg.DtBias, dr.DtBias)
			check("negExpA", dg.NegExpA, dr.NegExpA)                                                    // −exp(A_log) bake vs computed
			check("ssm_norm", dg.NormW, dr.NormW)                                                       // NOT (1+w)'d — raw load
			check("out_proj", wmDense(t, "out_proj", &dg.OutProj), wmDense(t, "out_proj", &dr.OutProj)) // column un-tile
		} else if lr.QAttn != nil && lg.QAttn != nil {
			ar, ag := lr.QAttn, lg.QAttn
			check("q_proj(query‖gate)", wmDense(t, "q_proj", &ag.QProj), wmDense(t, "q_proj", &ar.QProj)) // fused double-width — prime suspect
			check("k_proj", wmDense(t, "k_proj", &ag.KProj), wmDense(t, "k_proj", &ar.KProj))
			check("v_proj", wmDense(t, "v_proj", &ag.VProj), wmDense(t, "v_proj", &ar.VProj))
			check("o_proj", wmDense(t, "o_proj", &ag.OProj), wmDense(t, "o_proj", &ar.OProj))
			check("q_norm", ag.QNorm, ar.QNorm)
			check("k_norm", ag.KNorm, ar.KNorm)
		} else {
			t.Errorf("layer %d: attn kind mismatch (ref delta=%v qattn=%v / gguf delta=%v qattn=%v)",
				i, lr.Delta != nil, lr.QAttn != nil, lg.Delta != nil, lg.QAttn != nil)
		}
	}
	t.Logf("=== weightDiff: worst tensor cosine %.6f (%s) over layers 0-%d ===", worstCos, worstName, nLayer-1)
}

func layerKind(arch *darch.Architecture, i int) string {
	if arch.IsLinearLayer(i) {
		return "DeltaNet/linear"
	}
	return "softmax"
}

// wmDense reconstructs w as dense f32 whatever precision it is stored in. name labels the tensor in the failure message
// (t.Helper() reports the caller's line, so the helper cannot know which tensor it was given).
//
// Dequantizing is sound for this probe: both sides run through the SAME quantizer from the same source values, so a real
// transform bug (wrong un-tile, q‖gate, un-bake) still craters cosine. Only the noise floor moves, and it moves for both
// columns equally.
func wmDense(t *testing.T, name string, w *linalg.WeightMat) []float32 {
	t.Helper()
	if f, ok := w.F32(); ok {
		return f
	}
	if w.Kind() == "" {
		t.Fatalf("%s: empty WeightMat (rows=%d cols=%d) — nothing to compare", name, w.Rows(), w.Cols())
	}
	rows, cols := w.Rows(), w.Cols()
	out := make([]float32, rows*cols)
	for i := range rows {
		w.Row(i, out[i*cols:(i+1)*cols])
	}
	return out
}
