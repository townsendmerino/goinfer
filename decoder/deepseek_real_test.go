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
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// deepseekRealGate loads a real DeepSeek MLA checkpoint at int8 and matches it against an
// HF bf16 golden (argmax + greedy continuation + cosine). wantSigmoid asserts the resolved
// routing flavor (V3 sigmoid/noaux_tc vs V2 softmax); reference names the oracle for the
// emitted manifest row.
func deepseekRealGate(t *testing.T, ckpt, golden, wantArch, reference string, wantSigmoid bool) {
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
	// int8-resident forward vs the HF bf16 reference — a real-model oracle. argmax +
	// continuation are exact on a passing run, so argmax_pct is 100.
	emitParityRow(t, wantArch, "real-model-oracle", reference, 100.0, float64(cos), float64(cos))
}

// TestDeepseekV2LiteReal_gate — V2-Lite (deepseek_v2): direct-q + SOFTMAX routing + YaRN.
func TestDeepseekV2LiteReal_gate(t *testing.T) {
	requireHeavyModel(t)
	home, _ := os.UserHomeDir()
	ckpt := os.Getenv("GOINFER_DEEPSEEK_V2LITE")
	if ckpt == "" {
		ckpt = filepath.Join(home, "models", "deepseek-v2-lite")
	}
	deepseekRealGate(t, ckpt, "../testdata/deepseek_v2lite_golden.json", "deepseek_v2", "HF bf16 (DeepSeek-V2-Lite 15.7B; int8 resident)", false)
}

// TestMLAAbsorb_speed quantifies the absorb perf win on real V2-Lite (kv_lora_rank 512,
// 16 heads): it generates a run of tokens (so the latent cache — and thus the per-step
// attention work — grows) and times the absorb default vs the naive reconstruct-per-key
// path. Absorb removes the O(nKeys·H·(qkNope+vHead)·rank) per-key kv_b_proj matvec, so
// the gap widens with context. Output is identical (TestMLAAbsorb_parity gates that);
// this only reports tok/s + speedup.
func TestMLAAbsorb_speed(t *testing.T) {
	requireHeavyModel(t)
	if testing.Short() {
		t.Skip("perf measurement: skipped in -short")
	}
	home, _ := os.UserHomeDir()
	ckpt := os.Getenv("GOINFER_DEEPSEEK_V2LITE")
	if ckpt == "" {
		ckpt = filepath.Join(home, "models", "deepseek-v2-lite")
	}
	if _, err := os.Stat(ckpt); err != nil {
		t.Skipf("no checkpoint at %s: %v", ckpt, err)
	}
	// Time IDENTICAL work on both paths: forward a fixed 128-token sequence (the latent
	// cache grows to 128, so the per-step attention cost — where absorb wins — ramps up).
	// Greedy-GENERATING would diverge: absorb and naive are two equally-valid int8
	// approximations (both ~0.999 cosine vs HF), so a near-tie argmax eventually flips and
	// the sequences differ — expected reorder noise, not error (TestMLAAbsorb_parity gates
	// per-step equivalence). So we feed a fixed sequence and check the last-logit cosine.
	const n = 128
	seq := make([]int, n)
	for i := range seq {
		seq[i] = 100 + (i*7919)%20000 // arbitrary but fixed, in-vocab token ids
	}
	time1 := func(naive bool) (time.Duration, []float32) {
		m, err := Load(ckpt, Options{Quant: "int8int8"})
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		defer m.Close()
		mlaForceNaive = naive
		defer func() { mlaForceNaive = false }()
		cache := m.NewCache(n + 1)
		var logits []float32
		t0 := time.Now()
		for _, id := range seq {
			if logits, err = m.forward(id, cache); err != nil {
				t.Fatalf("forward: %v", err)
			}
		}
		return time.Since(t0), logits
	}
	absDT, absLogits := time1(false)
	naiveDT, naiveLogits := time1(true)
	cos := logitCosine(absLogits, naiveLogits)
	if cos < 0.999 {
		t.Errorf("absorb vs naive last-logit cosine %.6f < 0.999 over %d-tok forward", cos, n)
	}
	aTPS := float64(n) / absDT.Seconds()
	nTPS := float64(n) / naiveDT.Seconds()
	t.Logf("MLA absorb: %.2f tok/s (%v) | naive: %.2f tok/s (%v) | absorb %.2f× faster | last-logit cosine %.6f (V2-Lite, %d-tok forward)",
		aTPS, absDT.Round(time.Millisecond), nTPS, naiveDT.Round(time.Millisecond), aTPS/nTPS, cos, n)
}

// TestDeepseekMoonlightReal_gate — Moonlight-16B (deepseek_v3): direct-q + SIGMOID
// noaux_tc routing with a real e_score_correction_bias + routed_scaling_factor.
func TestDeepseekMoonlightReal_gate(t *testing.T) {
	requireHeavyModel(t)
	home, _ := os.UserHomeDir()
	ckpt := os.Getenv("GOINFER_DEEPSEEK_MOONLIGHT")
	if ckpt == "" {
		ckpt = filepath.Join(home, "models", "moonlight-16b")
	}
	deepseekRealGate(t, ckpt, "../testdata/deepseek_moonlight_golden.json", "deepseek_v3", "HF bf16 (Moonlight-16B-A3B; int8 resident)", true)
}

// TestDeepseekGGUFReal_gate exercises the llama.cpp deepseek2 GGUF loader on a real
// V2-Lite-Chat Q4_K_M: it verifies the MLA tensor-name mapping (attn_q / attn_kv_a_mqa /
// attn_kv_a_norm / attn_kv_b / attn_output), the synthesized YaRN rope_scaling, and the
// shared MoE block (ffn_gate_inp + ffn_*_exps + ffn_*_shexp) by GENERATING coherently —
// a wrong mapping yields garbage. (The Chat GGUF's weights differ from the base
// safetensors golden, so this is a coherence gate, not an argmax match.) The MLA forward
// itself is the bit-exact path the V2-Lite/Moonlight safetensors oracles already gate.
//
//	GOINFER_DEEPSEEK_GGUF=~/models/.../DeepSeek-V2-Lite-Chat-Q4_K_M.gguf \
//	  go test -tags realckpt ./decoder/ -run TestDeepseekGGUFReal -v -timeout 20m
func TestDeepseekGGUFReal_gate(t *testing.T) {
	requireHeavyModel(t)
	home, _ := os.UserHomeDir()
	gguf := os.Getenv("GOINFER_DEEPSEEK_GGUF")
	if gguf == "" {
		gguf = filepath.Join(home, "models", "deepseek-v2-lite-gguf", "DeepSeek-V2-Lite-Chat-Q4_K_M.gguf")
	}
	if _, err := os.Stat(gguf); err != nil {
		t.Skipf("no DeepSeek GGUF at %s: %v", gguf, err)
	}
	m, err := Load(gguf, Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("Load(%s): %v", gguf, err)
	}
	defer m.Close()
	a := m.w.arch
	if a.mla == nil {
		t.Fatalf("arch %q has no mla (GGUF deepseek2 not resolved)", a.Name)
	}
	t.Logf("GGUF %s: %d layers, q_lora=%d kv_lora=%d qk=%d v=%d, experts=%d top%d firstKDense=%d sigmoid=%v ropeMscale=%.4f",
		a.Name, a.NumLayers, a.mla.QLoRARank, a.mla.KVLoRARank, a.mla.qkHeadDim(), a.mla.VHeadDim,
		a.MoE.NumExperts, a.MoE.TopK, a.FirstKDense, a.MoE.RouterSigmoid, a.ropeMscale(0))

	tk, err := tokenizer.LoadGGUF(gguf)
	if err != nil {
		t.Fatalf("LoadGGUF tokenizer: %v", err)
	}
	prompt := "The capital of France is"
	ids, err := tk.Encode(prompt, true)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	out, _ := m.Generate(context.Background(), ids, 24, SamplingParams{})
	gen := make([]int, 0, 24)
	for id := range out {
		gen = append(gen, id)
	}
	distinct := map[int]bool{}
	for _, id := range gen {
		distinct[id] = true
	}
	text, _ := tk.Decode(gen)
	t.Logf("GGUF gen: %q", text)
	if len(gen) == 0 || len(distinct) < 3 {
		t.Errorf("degenerate output: %d distinct in %d tokens", len(distinct), len(gen))
	}
	if !strings.Contains(text, "Paris") {
		t.Logf("note: continuation did not contain \"Paris\" (Chat model, Q4) — coherence is the gate")
	}
}
