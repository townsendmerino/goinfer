//go:build realckpt

// Cheap loader-vs-quant disambiguator (no full-model forward, no 60-min oracle).
// The bf16-golden gate (qwen35_gguf_gate_test.go) failed by a hair (argmax 68/80,
// cosine 0.99445 vs the 74 / 0.99466 bars), with the divergences being rank-2
// near-ties. That's the signature of quant noise at the threshold, NOT a cratered
// loader — but the bf16 gate can't prove it. This does: it loads only the first 4
// layers of BOTH containers at f32 (the existing loadQwen35Slice for safetensors,
// a sibling for the GGUF) and diffs every TRANSFORM-bearing tensor directly —
// the V-head un-tile, the q‖gate fused q_proj, the −exp(A_log) bake, the norm(+1)
// un-bake. The safetensors loader is Gate-1 bit-exact vs HF, so it is the
// reference. A correct GGUF loader ⇒ each tensor matches to the Q8_0-vs-bf16
// dequant delta (cosine ≳ 0.9995); a transform bug ⇒ that one tensor's cosine
// craters, pinpointing it. Slice 0–3 spans both layer kinds (DeltaNet 0,1,2 +
// softmax 3).
//
//	GOINFER_QWEN35_REAL=~/models/qwen3.6-35b-a3b \
//	GOINFER_QWEN35_GGUF=~/models/qwen3.6-35b-a3b-Q8_0.gguf \
//	  go test -tags realckpt ./decoder/ -run TestQwen35GGUF_weightDiff -v -timeout 15m
package decoder

import (
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/townsendmerino/aikit/embed"
)

// loadQwen35GGUFSlice loads embed + the first nLayer layers of the qwen35moe GGUF
// at f32 (the DeltaNet/softmax tensors are always f32 on this path; experts go
// int8 since this probe never compares them). Truncating NumLayers makes the
// loader touch only blk.0..nLayer-1 — fast, a few GB vs the 37 GB file.
func loadQwen35GGUFSlice(t *testing.T, path string, nLayer int) (*Weights, func()) {
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
	quant, _ := parseQuant("int8int8") // experts int8; the compared attn tensors stay f32
	w, err := buildWeightsFromGGUF(cfg, arch, g, quant, false, nil, "")
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
	requireHeavyModel(t)
	home, _ := os.UserHomeDir()
	dir := realQwen35Dir(t) // safetensors (skips if absent)
	gguf := os.Getenv("GOINFER_QWEN35_GGUF")
	if gguf == "" {
		gguf = filepath.Join(home, "models", "qwen3.6-35b-a3b-Q8_0.gguf")
	}
	if _, err := os.Stat(gguf); err != nil {
		t.Skipf("no qwen35moe GGUF at %s: %v", gguf, err)
	}

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

	for i := 0; i < nLayer; i++ {
		t.Logf("--- layer %d (%s) ---", i, layerKind(wRef.arch, i))
		lr, lg := &wRef.Layers[i], &gW.Layers[i]
		check("attn_norm", lg.PreAttnNorm, lr.PreAttnNorm)
		check("post_attention_norm", lg.PreMLPNorm, lr.PreMLPNorm)
		if lr.delta != nil && lg.delta != nil {
			dr, dg := lr.delta, lg.delta
			check("in_proj_qkv", dg.inProjQKV, dr.inProjQKV) // V-block un-tile
			check("in_proj_z", dg.inProjZ, dr.inProjZ)       // full un-tile
			check("in_proj_a", dg.inProjA, dr.inProjA)
			check("in_proj_b", dg.inProjB, dr.inProjB)
			check("conv1d", dg.convW, dr.convW) // V-channel un-tile
			check("dt_bias", dg.dtBias, dr.dtBias)
			check("negExpA", dg.negExpA, dr.negExpA)  // −exp(A_log) bake vs computed
			check("ssm_norm", dg.normW, dr.normW)     // NOT (1+w)'d — raw load
			check("out_proj", dg.outProj, dr.outProj) // column un-tile
		} else if lr.qattn != nil && lg.qattn != nil {
			ar, ag := lr.qattn, lg.qattn
			check("q_proj(query‖gate)", ag.qProj, ar.qProj) // fused double-width — prime suspect
			check("k_proj", ag.kProj, ar.kProj)
			check("v_proj", ag.vProj, ar.vProj)
			check("o_proj", ag.oProj, ar.oProj)
			check("q_norm", ag.qNorm, ar.qNorm)
			check("k_norm", ag.kNorm, ar.kNorm)
		} else {
			t.Errorf("layer %d: attn kind mismatch (ref delta=%v qattn=%v / gguf delta=%v qattn=%v)",
				i, lr.delta != nil, lr.qattn != nil, lg.delta != nil, lg.qattn != nil)
		}
	}
	t.Logf("=== weightDiff: worst tensor cosine %.6f (%s) over layers 0-%d ===", worstCos, worstName, nLayer-1)
}

func layerKind(arch *Architecture, i int) string {
	if arch.isLinearLayer(i) {
		return "DeltaNet/linear"
	}
	return "softmax"
}
