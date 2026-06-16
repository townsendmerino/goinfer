//go:build realckpt

// Real-model gates for DeepSeek MLA on actual released weights — the cleanly-validated
// end-to-end proof that the MLA loader + latent-cache forward + DeepSeekMoE scale to real
// dims, both routing flavors, and live YaRN. Two complementary small models cover the
// matrix the tiny f32 golden (q-LoRA + sigmoid + group routing, cosine 1.0) leaves to
// real weights:
//
//   - DeepSeek-V2-Lite (deepseek_v2, 15.7B-A2.4B): direct-q (q_lora_rank=None), SOFTMAX
//     greedy routing (n_group=1, no e_score_correction_bias), 512-wide latent, live YaRN
//     (factor 40, mscale==mscale_all_dim ⇒ attention_factor 1.0).
//   - Moonlight-16B-A3B (deepseek_v3, Moonshot): direct-q, SIGMOID noaux_tc routing WITH
//     a real e_score_correction_bias + routed_scaling_factor 2.446, rope_theta 50000, no
//     YaRN. The real-weights proof of the V3 routing flavor (V2-Lite is softmax).
//
// goinfer loads at int8 (~16 GB resident; f32 ~63 GB is over this box's RAM and there is
// no bf16-resident mode), so each gate matches the last-token argmax + greedy continuation
// against the HF bf16 golden (cosine logged; int8 W8A8 is high-fidelity, ~0.999, not
// bit-exact). Fixtures: scripts/pin_deepseek_v2lite.py, scripts/pin_deepseek_moonlight.py.
//
//	go test -tags realckpt ./decoder/ -run TestDeepseek.*Real -v -timeout 40m
package decoder

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// deepseekRealGate loads a real DeepSeek MLA checkpoint at int8 and matches it against an
// HF bf16 golden (argmax + greedy continuation + cosine). wantSigmoid asserts the resolved
// routing flavor (V3 sigmoid/noaux_tc vs V2 softmax).
func deepseekRealGate(t *testing.T, ckpt, golden, wantArch string, wantSigmoid bool) {
	if _, err := os.Stat(ckpt); err != nil {
		t.Skipf("no checkpoint at %s: %v", ckpt, err)
	}
	raw, err := os.ReadFile(golden)
	if err != nil {
		t.Skipf("no golden (%v) — run the pin script", err)
	}
	var g struct {
		PromptIDs       []int     `json:"prompt_ids"`
		Argmax          int       `json:"argmax"`
		LastLogits      []float32 `json:"last_logits"`
		NNew            int       `json:"n_new"`
		ContinuationIDs []int     `json:"continuation_ids"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}

	m, err := Load(ckpt, Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("Load(%s): %v", ckpt, err)
	}
	defer m.Close()

	a := m.w.arch
	if a.Name != wantArch || a.mla == nil {
		t.Fatalf("arch = %q (mla=%v), want %s", a.Name, a.mla != nil, wantArch)
	}
	t.Logf("%s: %d layers, H=%d, q_lora=%d kv_lora=%d qk=%d v=%d, experts=%d top%d firstKDense=%d sigmoid=%v nGroup=%d routedScale=%.3f attnScale=%.6f ropeMscale=%.4f",
		wantArch, a.NumLayers, a.NumHeads, a.mla.QLoRARank, a.mla.KVLoRARank, a.mla.qkHeadDim(), a.mla.VHeadDim,
		a.MoE.NumExperts, a.MoE.TopK, a.FirstKDense, a.MoE.RouterSigmoid, a.MoE.NGroup, a.MoE.RoutedScale, a.AttnScale, a.ropeMscale(0))
	if a.MoE.RouterSigmoid != wantSigmoid {
		t.Errorf("RouterSigmoid = %v, want %v", a.MoE.RouterSigmoid, wantSigmoid)
	}

	cache := m.NewCache(len(g.PromptIDs) + g.NNew)
	var logits []float32
	for _, id := range g.PromptIDs {
		if logits, err = m.forward(id, cache); err != nil {
			t.Fatalf("forward: %v", err)
		}
	}
	gotArg := argmax(logits)
	cos := logitCosine(logits, g.LastLogits)
	t.Logf("%s parity: argmax got=%d want=%d | logit cosine=%.6f", wantArch, gotArg, g.Argmax, cos)
	if gotArg != g.Argmax {
		t.Errorf("last argmax = %d, want %d", gotArg, g.Argmax)
	}
	if cos < 0.99 { // int8 W8A8 vs bf16; the tiny f32 golden is the cosine-1.0 gate
		t.Errorf("last-logit cosine %.6f < 0.99", cos)
	}

	got := make([]int, 0, g.NNew)
	for range g.NNew {
		id := argmax(logits)
		got = append(got, id)
		if logits, err = m.forward(id, cache); err != nil {
			t.Fatalf("continuation forward: %v", err)
		}
	}
	t.Logf("%s continuation got=%v want=%v", wantArch, got, g.ContinuationIDs)
	for i := range g.ContinuationIDs {
		if got[i] != g.ContinuationIDs[i] {
			t.Errorf("continuation[%d] = %d, want %d", i, got[i], g.ContinuationIDs[i])
			break
		}
	}
}

// TestDeepseekV2LiteReal_gate — V2-Lite (deepseek_v2): direct-q + SOFTMAX routing + YaRN.
func TestDeepseekV2LiteReal_gate(t *testing.T) {
	home, _ := os.UserHomeDir()
	ckpt := os.Getenv("GOINFER_DEEPSEEK_V2LITE")
	if ckpt == "" {
		ckpt = filepath.Join(home, "models", "deepseek-v2-lite")
	}
	deepseekRealGate(t, ckpt, "../testdata/deepseek_v2lite_golden.json", "deepseek_v2", false)
}

// TestDeepseekMoonlightReal_gate — Moonlight-16B (deepseek_v3): direct-q + SIGMOID
// noaux_tc routing with a real e_score_correction_bias + routed_scaling_factor.
func TestDeepseekMoonlightReal_gate(t *testing.T) {
	home, _ := os.UserHomeDir()
	ckpt := os.Getenv("GOINFER_DEEPSEEK_MOONLIGHT")
	if ckpt == "" {
		ckpt = filepath.Join(home, "models", "moonlight-16b")
	}
	deepseekRealGate(t, ckpt, "../testdata/deepseek_moonlight_golden.json", "deepseek_v3", true)
}
