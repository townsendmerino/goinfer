//go:build realckpt

// Real-GGUF gate for Laguna — the GGUF loader + forward on poolside's own
// Laguna-XS-2.1-Q4_K_M.gguf.
//
// WHY THIS EXISTS SEPARATELY FROM THE SAFETENSORS GATE. It is a different loader
// reading a different layout, and llama.cpp's metadata expresses this family
// differently than the HF config does — so the two paths can disagree in ways
// neither gate alone would show:
//
//   - Experts are FUSED+STACKED here (ffn_*_exps, one 3-D tensor per projection)
//     where safetensors ships one tensor per expert. A fused stack read with the
//     wrong stride yields correct shapes, finite values, and confident nonsense.
//
//   - Per-layer QUERY heads arrive as an ARRAY (laguna.attention.head_count)
//     rather than as num_attention_heads_per_layer.
//
//   - There is NO layer_types and no sliding-window pattern key, so which layers
//     are full is DERIVED from that array. This gate is what makes the inference
//     safe: it asserts the 10/30 split and the per-layer q_proj widths.
//
//   - llama.cpp writes rope.scaling.yarn_attn_factor = 1.0 as its "unset"
//     sentinel. Passing that through would REPLACE YaRN's mscale with a no-op,
//     which is a silent ~quality regression rather than a failure — so the gate
//     pins the computed mscale against the value the HF config states outright.
//
//     GOINFER_HEAVY_TESTS=1 GOINFER_LAGUNA_GGUF=~/models/laguna-xs21-gguf/Laguna-XS-2.1-Q4_K_M.gguf \
//     go test -tags realckpt ./decoder/ -run TestLagunaGGUF -v -timeout 120m
package decoder

import (
	"context"
	"math"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
)

func TestLagunaGGUF_gate(t *testing.T) {
	requireHeavyModel(t)
	path := assetPath(t, "GOINFER_LAGUNA_GGUF")

	m, err := Load(path, Options{})
	if err != nil {
		t.Fatalf("Load(%s): %v", path, err)
	}
	defer m.Close()

	a := m.w.arch
	if a.Name != "laguna" || a.laguna == nil {
		t.Fatalf("arch = %q (laguna=%v), want laguna", a.Name, a.laguna != nil)
	}
	if a.NumLayers != 40 || a.HiddenDim != 2048 || a.HeadDim != 128 || a.NumKVHeads != 8 {
		t.Errorf("geometry = %dL/%dh/hd%d/kv%d, want 40/2048/128/8",
			a.NumLayers, a.HiddenDim, a.HeadDim, a.NumKVHeads)
	}

	// The derived full/sliding split, and the per-layer widths it implies. This is
	// the assertion that makes deriving layer types from the head-count array safe.
	full, sliding := 0, 0
	for i := range a.NumLayers {
		wantHeads := 64
		if a.isGlobalLayer(i) {
			wantHeads, full = 48, full+1
		} else {
			sliding++
		}
		if got := a.headsAt(i); got != wantHeads {
			t.Fatalf("headsAt(%d) = %d, want %d (global=%v)", i, got, wantHeads, a.isGlobalLayer(i))
		}
		if got := m.w.Layers[i].QProj.Rows(); got != wantHeads*a.HeadDim {
			t.Fatalf("layer %d attn_q rows = %d, want %d", i, got, wantHeads*a.HeadDim)
		}
		// attn_gate is per-head at THIS layer's own head count.
		if got := m.w.Layers[i].GProj.Rows(); got != wantHeads {
			t.Errorf("layer %d attn_gate rows = %d, want %d (per-head)", i, got, wantHeads)
		}
	}
	if full != 10 || sliding != 30 {
		t.Errorf("derived layer types = %d full / %d sliding, want 10/30 — the head-count array "+
			"no longer implies the released 3:1 interleave", full, sliding)
	}

	// RoPE: per-layer-type bases AND widths, from rope.dimension_count(_swa).
	if a.RoPEGlobalBase != 500000 || a.RoPELocalBase != 10000 {
		t.Errorf("rope bases = %v/%v, want 500000/10000", a.RoPEGlobalBase, a.RoPELocalBase)
	}
	if a.RotaryDim != 64 || a.RotaryDimLocal != 128 {
		t.Errorf("RotaryDim/Local = %d/%d, want 64/128", a.RotaryDim, a.RotaryDimLocal)
	}
	if got := len(a.ropeInvFreqGlobal); got != 32 {
		t.Errorf("len(ropeInvFreqGlobal) = %d, want 32 (partial rotary 0.5 on full layers)", got)
	}
	if got := len(a.ropeInvFreqLocal); got != 64 {
		t.Errorf("len(ropeInvFreqLocal) = %d, want 64 (full-width rotary on sliding layers)", got)
	}
	// THE yarn_attn_factor SENTINEL. llama.cpp stores 1.0 meaning "unset"; goinfer
	// must fall back to get_mscale(factor) = 0.1·ln(32)+1 = 1.3465735902799727, which
	// is exactly what poolside's own config.json states. Taking the 1.0 at face value
	// would silently disable YaRN's attention scaling on every full layer.
	const wantMscale = 1.3465735902799727
	if got := a.ropeMscale(0); math.Abs(got-wantMscale) > 1e-9 {
		t.Errorf("full-layer YaRN mscale = %.16f, want %.16f — llama.cpp's yarn_attn_factor=1.0 "+
			"is an UNSET sentinel, not a real attention_factor", got, wantMscale)
	}
	if got := a.ropeMscale(1); got != 1 {
		t.Errorf("sliding-layer mscale = %v, want 1 (plain RoPE, no YaRN)", got)
	}

	// MoE, including the dense prefix that leading_dense_block_count implies.
	if a.MoE == nil {
		t.Fatal("arch.MoE is nil")
	}
	if a.MoE.NumExperts != 256 || a.MoE.TopK != 8 {
		t.Errorf("MoE = %d experts / top-%d, want 256/8", a.MoE.NumExperts, a.MoE.TopK)
	}
	if a.MoE.IntermediateDim != 512 || a.MoE.SharedIntermediateDim != 512 {
		t.Errorf("expert/shared width = %d/%d, want 512/512", a.MoE.IntermediateDim, a.MoE.SharedIntermediateDim)
	}
	if !a.MoE.RouterSigmoid || !a.MoE.SharedUngated {
		t.Errorf("RouterSigmoid=%v SharedUngated=%v, want true/true", a.MoE.RouterSigmoid, a.MoE.SharedUngated)
	}
	if a.MoE.RoutedScale != 2.5 {
		t.Errorf("RoutedScale = %v, want 2.5 (expert_weights_scale)", a.MoE.RoutedScale)
	}
	if a.FirstKDense != 1 {
		t.Errorf("FirstKDense = %d, want 1 (leading_dense_block_count)", a.FirstKDense)
	}
	if m.w.Layers[0].Router.Rows() != 0 || m.w.Layers[0].GateProj.Rows() != 8192 {
		t.Errorf("layer 0 should be the dense prefix (router=%d rows, ffn_gate=%d rows, want 0/8192)",
			m.w.Layers[0].Router.Rows(), m.w.Layers[0].GateProj.Rows())
	}
	if len(m.w.Layers[1].RouterBias) != 256 {
		t.Errorf("layer 1 exp_probs_b len = %d, want 256", len(m.w.Layers[1].RouterBias))
	}

	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatalf("LoadGGUF tokenizer: %v", err)
	}
	// XS-2.1's chat template is laguna_glm_thinking_v8, which differs from XS.2's v5
	// (no newlines inside the tags), so this literal is rendered from THIS model's own
	// template rather than reused. The trailing "</think>" suppresses thinking mode;
	// without it a short budget is spent reasoning and never answers.
	const prompt = "〈|EOS|〉<system>You are a helpful, conversationally-fluent assistant made by " +
		"Poolside. You are here to be helpful to users through natural language conversations.</system>\n" +
		"<user>Name three landmarks in Paris.</user>\n<assistant></think>"
	ids, err := tk.Encode(prompt, false)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if len(ids) != 47 {
		t.Errorf("prompt encoded to %d ids, want 47 (transformers) — special tokens may not have "+
			"been recognized as single ids", len(ids))
	}

	out, _ := m.Generate(context.Background(), ids, 48, SamplingParams{})
	gen := make([]int, 0, 48)
	for id := range out {
		gen = append(gen, id)
	}
	if len(gen) == 0 {
		t.Fatal("no tokens generated")
	}
	text, _ := tk.Decode(gen)
	ratio := distinctTrigramRatio(text)
	t.Logf("laguna XS-2.1 GGUF Q4_K_M: %d tokens, distinct-trigram %.3f: %q", len(gen), ratio, strings.TrimSpace(text))
	if ratio < 0.5 {
		t.Errorf("distinct-trigram %.3f < 0.5 — degenerate output, the shape a mis-strided fused "+
			"expert stack produces", ratio)
	}
	low := strings.ToLower(text)
	hits := 0
	for _, want := range []string{"eiffel", "louvre", "notre", "arc de triomphe", "sacr", "seine", "champs"} {
		if strings.Contains(low, want) {
			hits++
		}
	}
	if hits < 2 {
		t.Errorf("only %d Paris landmarks named in %q — fluent but wrong is what a stride bug looks like",
			hits, strings.TrimSpace(text))
	}
}
