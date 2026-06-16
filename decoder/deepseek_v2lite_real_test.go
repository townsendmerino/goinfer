//go:build realckpt

// Real-model gate for DeepSeek-V2-Lite (deepseek_v2, 15.7B-A2.4B MLA) — the MLA loader +
// latent-cache forward + DeepSeekMoE on actual released weights, the cleanly-validated
// end-to-end MLA proof. V2-Lite exercises the direct-q path (q_lora_rank=None), softmax/
// greedy routing (n_group=1, no e_score_correction_bias), the 512-wide compressed KV
// latent, and live YaRN (factor 40, mscale==mscale_all_dim ⇒ attention_factor 1.0).
//
// goinfer loads at int8 (~16 GB resident — the f32 weights are ~63 GB, over this box's
// RAM, and there is no bf16-resident mode), so the gate matches the last-token argmax and
// greedy continuation against the HF bf16 golden (cosine logged; int8 W8A8 is high-fidelity
// but not bit-exact). Fixture: scripts/pin_deepseek_v2lite.py.
//
//	go test -tags realckpt ./decoder/ -run TestDeepseekV2LiteReal -v -timeout 30m
package decoder

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestDeepseekV2LiteReal_gate(t *testing.T) {
	home, _ := os.UserHomeDir()
	ckpt := os.Getenv("GOINFER_DEEPSEEK_V2LITE")
	if ckpt == "" {
		ckpt = filepath.Join(home, "models", "deepseek-v2-lite")
	}
	if _, err := os.Stat(ckpt); err != nil {
		t.Skipf("no DeepSeek-V2-Lite at %s: %v", ckpt, err)
	}
	const golden = "../testdata/deepseek_v2lite_golden.json"
	raw, err := os.ReadFile(golden)
	if err != nil {
		t.Skipf("no golden (%v) — run scripts/pin_deepseek_v2lite.py", err)
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
	if a.Name != "deepseek_v2" || a.mla == nil {
		t.Fatalf("arch = %q (mla=%v), want deepseek_v2", a.Name, a.mla != nil)
	}
	t.Logf("V2-Lite: %d layers, H=%d, q_lora=%d kv_lora=%d qk=%d v=%d, experts=%d top%d shared=%d, firstKDense=%d, attnScale=%.6f ropeMscale=%.4f",
		a.NumLayers, a.NumHeads, a.mla.QLoRARank, a.mla.KVLoRARank, a.mla.qkHeadDim(), a.mla.VHeadDim,
		a.MoE.NumExperts, a.MoE.TopK, a.MoE.SharedIntermediateDim/a.MoE.IntermediateDim, a.FirstKDense, a.AttnScale, a.ropeMscale(0))
	if a.mla.QLoRARank != 0 {
		t.Errorf("V2-Lite should be direct-q (q_lora_rank 0), got %d", a.mla.QLoRARank)
	}
	if a.MoE.RouterSigmoid {
		t.Errorf("V2-Lite should use softmax routing, not sigmoid")
	}

	// Prefill the prompt, compare the last-token logits to the HF bf16 reference.
	cache := m.NewCache(len(g.PromptIDs) + g.NNew)
	var logits []float32
	for _, id := range g.PromptIDs {
		if logits, err = m.forward(id, cache); err != nil {
			t.Fatalf("forward: %v", err)
		}
	}
	gotArg := argmax(logits)
	cos := logitCosine(logits, g.LastLogits)
	t.Logf("V2-Lite parity: argmax got=%d want=%d | logit cosine=%.6f", gotArg, g.Argmax, cos)
	if gotArg != g.Argmax {
		t.Errorf("last argmax = %d, want %d", gotArg, g.Argmax)
	}
	if cos < 0.99 { // int8 W8A8 vs bf16 reference; the tiny f32 golden is the cosine-1.0 gate
		t.Errorf("last-logit cosine %.6f < 0.99", cos)
	}

	// Greedy continuation must track HF (int8 is high-fidelity; a clear prompt should match).
	got := make([]int, 0, g.NNew)
	for range g.NNew {
		id := argmax(logits)
		got = append(got, id)
		if logits, err = m.forward(id, cache); err != nil {
			t.Fatalf("continuation forward: %v", err)
		}
	}
	t.Logf("continuation got=%v want=%v", got, g.ContinuationIDs)
	for i := range g.ContinuationIDs {
		if got[i] != g.ContinuationIDs[i] {
			t.Errorf("continuation[%d] = %d, want %d", i, got[i], g.ContinuationIDs[i])
			break
		}
	}
}
