package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"os"
	"testing"
)

// G7 GGUF parity. Loads quantized TinyLlama GGUFs (Q8_0, Q4_0) — the same model
// as testdata/tinyllama-1.1b — through the generic forward and compares to the
// f32 oracle (llama_forward_golden.json + llama_forward_full.json). Quantization
// is lossy, so the bar is the right one for each type: argmax must still land on
// ' Paris', and the cosine vs the f32 logits must clear a per-type floor (Q8_0
// near-lossless; Q4_0 coarser). This validates the GGUF container parse, block
// dequant, llama.cpp→config metadata mapping, and the q/k RoPE un-permutation.
//
// Models: TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF (ungated).
func testGGUFParity(t *testing.T, ggufPath string, minCosine float64) {
	if testing.Short() {
		t.Skip("slow: loads + runs a TinyLlama GGUF")
	}
	raw, err := os.ReadFile(llamaForwardGolden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no llama golden at %s — regenerate with scripts/pin_llama_forward.py", llamaForwardGolden)
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g forwardGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	if _, err := os.Stat(ggufPath); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no GGUF at %s", ggufPath)
	}

	m, err := Load(ggufPath, Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if m.w.arch.Name != "llama" {
		t.Fatalf("resolved arch %q, want llama", m.w.arch.Name)
	}

	cache := m.NewCache(len(g.IDs))
	for _, id := range g.IDs[:len(g.IDs)-1] {
		if _, err := m.runLayers(id, cache); err != nil {
			t.Fatalf("runLayers: %v", err)
		}
	}
	logits, err := m.forward(g.IDs[len(g.IDs)-1], cache)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}

	// Quantized argmax must still match the f32 oracle (a confident token).
	if got := argmax(logits); got != g.Argmax {
		t.Errorf("argmax = %d, want %d (logit[got]=%.4f logit[want]=%.4f)",
			got, g.Argmax, logits[got], logits[g.Argmax])
	}
	cos := cosineToFull(t, logits, llamaForwardFullPath)
	if !math.IsNaN(cos) && cos < minCosine {
		t.Errorf("cosine vs f32 oracle = %v, want ≥ %v", cos, minCosine)
	}
	t.Logf("gguf %s: argmax=%d (want %d) | cosine vs f32 = %v", ggufPath, argmax(logits), g.Argmax, cos)
}

func TestGGUF_Q8_0_parity(t *testing.T) {
	testGGUFParity(t, "../testdata/tinyllama-gguf/tinyllama-1.1b-chat-v1.0.Q8_0.gguf", 0.999)
}

// TestGGUF_gemma3_parity loads a real gemma-3-270m Q8_0 GGUF against the same
// model's f32 oracle. Gemma 3 is the most involved GGUF arch: sandwich norms
// (post-attention + post-FFN, loaded here via post_attention_norm/post_ffw_norm),
// QK-norm, GeGLU, the embed scale √hidden, the 5:1 sliding/global pattern with
// dual RoPE bases, and a tied head — all on the NEOX (no-permute) path. Q8_0, so
// argmax must match and cosine clear 0.995. ~0.3 GB (gitignored); skip if absent.
//
//	hf download ggml-org/gemma-3-270m-GGUF gemma-3-270m-Q8_0.gguf --local-dir testdata/gemma3-gguf
func TestGGUF_gemma3_parity(t *testing.T) {
	if testing.Short() {
		t.Skip("slow: loads + runs a gemma-3-270m GGUF")
	}
	raw, err := os.ReadFile(gemmaForwardGoldenPath)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no gemma golden at %s", gemmaForwardGoldenPath)
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g forwardGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	const ggufPath = "../testdata/gemma3-gguf/gemma-3-270m-Q8_0.gguf"
	if _, err := os.Stat(ggufPath); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no gemma3 GGUF at %s", ggufPath)
	}

	m, err := Load(ggufPath, Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if m.w.arch.Name != "gemma3" {
		t.Fatalf("resolved arch %q, want gemma3", m.w.arch.Name)
	}
	if m.w.arch.NormPlacement != NormSandwich4 {
		t.Errorf("expected NormSandwich4 for gemma3")
	}
	if m.w.Layers[0].PostAttnNorm == nil || m.w.Layers[0].PostMLPNorm == nil {
		t.Fatalf("sandwich post-norms not loaded for layer 0")
	}

	cache := m.NewCache(len(g.IDs))
	for _, id := range g.IDs[:len(g.IDs)-1] {
		if _, err := m.runLayers(id, cache); err != nil {
			t.Fatalf("runLayers: %v", err)
		}
	}
	logits, err := m.forward(g.IDs[len(g.IDs)-1], cache)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	if got := argmax(logits); got != g.Argmax {
		t.Errorf("argmax = %d, want %d (logit[got]=%.4f logit[want]=%.4f)",
			got, g.Argmax, logits[got], logits[g.Argmax])
	}
	cos := cosineToFull(t, logits, gemmaForwardFullPath)
	if !math.IsNaN(cos) && cos < 0.995 {
		t.Errorf("gemma3 GGUF cosine vs f32 oracle = %v, want ≥ 0.995", cos)
	}
	t.Logf("gguf gemma3 Q8_0: argmax=%d (want %d) | cosine vs f32 = %v", argmax(logits), g.Argmax, cos)
}

// TestGGUF_qwen3_parity loads a real Qwen3-1.7B Q8_0 GGUF against the same model's
// f32 oracle. Qwen3 over qwen2: no q/k/v bias, but QK-norm (per-head RMSNorm over
// head_dim) and an explicit head_dim — exercising the QKNorm tensor load on the
// NEOX (no-permute) path, plus the tied LM head (no output.weight). Q8_0, so
// argmax must match and cosine clear 0.995. ~1.8 GB (gitignored); skip if absent.
//
//	hf download Qwen/Qwen3-1.7B-GGUF Qwen3-1.7B-Q8_0.gguf --local-dir testdata/qwen3-gguf
func TestGGUF_qwen3_parity(t *testing.T) {
	if testing.Short() {
		t.Skip("slow: loads + runs a Qwen3-1.7B GGUF")
	}
	raw, err := os.ReadFile(qwen3ForwardGolden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no Qwen3 golden at %s", qwen3ForwardGolden)
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g forwardGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	const ggufPath = "../testdata/qwen3-gguf/Qwen3-1.7B-Q8_0.gguf"
	if _, err := os.Stat(ggufPath); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no Qwen3 GGUF at %s", ggufPath)
	}

	m, err := Load(ggufPath, Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if m.w.arch.Name != "qwen3" {
		t.Fatalf("resolved arch %q, want qwen3", m.w.arch.Name)
	}
	if !m.w.arch.QKNorm {
		t.Errorf("expected QKNorm=true for Qwen3")
	}
	if m.w.Layers[0].QNorm == nil || m.w.Layers[0].KNorm == nil {
		t.Fatalf("q/k norm not loaded for layer 0")
	}

	cache := m.NewCache(len(g.IDs))
	for _, id := range g.IDs[:len(g.IDs)-1] {
		if _, err := m.runLayers(id, cache); err != nil {
			t.Fatalf("runLayers: %v", err)
		}
	}
	logits, err := m.forward(g.IDs[len(g.IDs)-1], cache)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	if got := argmax(logits); got != g.Argmax {
		t.Errorf("argmax = %d, want %d (logit[got]=%.4f logit[want]=%.4f)",
			got, g.Argmax, logits[got], logits[g.Argmax])
	}
	cos := cosineToFull(t, logits, qwen3ForwardFullPath)
	if !math.IsNaN(cos) && cos < 0.995 {
		t.Errorf("qwen3 GGUF cosine vs f32 oracle = %v, want ≥ 0.995", cos)
	}
	t.Logf("gguf qwen3 Q8_0: argmax=%d (want %d) | cosine vs f32 = %v", argmax(logits), g.Argmax, cos)
}

// TestGGUF_qwen2_parity loads a real Qwen2.5-0.5B-Instruct Q8_0 GGUF and checks
// its forward against the same model's f32 oracle (qwen2_forward_*.json). This
// validates the new qwen2 GGUF architecture path end-to-end: the qwen2.* metadata
// mapping, and — the one thing qwen2 adds over llama — loading the q/k/v
// projection biases (with the q/k bias RoPE-permuted like the q/k weight rows).
// Q8_0 is near-lossless, so the argmax must still land on ' Paris' and the cosine
// clear 0.995 (looser than TinyLlama's 0.999 floor: at 0.5 B with a 152k-vocab
// Q8_0 head the per-logit quant error spreads wider — measured ~0.997). Loads
// ~0.6 GB (gitignored); skips when absent or under -short.
//
//	hf download Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q8_0.gguf \
//	  --local-dir testdata/qwen2-gguf
func TestGGUF_qwen2_parity(t *testing.T) {
	if testing.Short() {
		t.Skip("slow: loads + runs a Qwen2.5-0.5B GGUF")
	}
	raw, err := os.ReadFile(qwen2ForwardGolden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no Qwen2 golden at %s", qwen2ForwardGolden)
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g forwardGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	const ggufPath = "../testdata/qwen2-gguf/qwen2.5-0.5b-instruct-q8_0.gguf"
	if _, err := os.Stat(ggufPath); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no Qwen2 GGUF at %s", ggufPath)
	}

	m, err := Load(ggufPath, Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if m.w.arch.Name != "qwen2" {
		t.Fatalf("resolved arch %q, want qwen2", m.w.arch.Name)
	}
	if !m.w.arch.QKVBias {
		t.Errorf("expected QKVBias=true for Qwen2")
	}
	if m.w.Layers[0].QBias == nil || m.w.Layers[0].KBias == nil || m.w.Layers[0].VBias == nil {
		t.Fatalf("q/k/v bias not loaded for layer 0")
	}

	cache := m.NewCache(len(g.IDs))
	for _, id := range g.IDs[:len(g.IDs)-1] {
		if _, err := m.runLayers(id, cache); err != nil {
			t.Fatalf("runLayers: %v", err)
		}
	}
	logits, err := m.forward(g.IDs[len(g.IDs)-1], cache)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	if got := argmax(logits); got != g.Argmax {
		t.Errorf("argmax = %d, want %d (logit[got]=%.4f logit[want]=%.4f)",
			got, g.Argmax, logits[got], logits[g.Argmax])
	}
	cos := cosineToFull(t, logits, qwen2ForwardFullPath)
	if !math.IsNaN(cos) && cos < 0.995 {
		t.Errorf("qwen2 GGUF cosine vs f32 oracle = %v, want ≥ 0.995", cos)
	}
	t.Logf("gguf qwen2 Q8_0: argmax=%d (want %d) | cosine vs f32 = %v", argmax(logits), g.Argmax, cos)
}

func TestGGUF_Q4_0_parity(t *testing.T) {
	testGGUFParity(t, "../testdata/tinyllama-gguf/tinyllama-1.1b-chat-v1.0.Q4_0.gguf", 0.99)
}

// Q4_K_M — the most-downloaded laptop quant — mixes Q4_K (most weights) and
// Q6_K (output + some attention/ffn), exercising both K-quant dequants.
func TestGGUF_Q4_K_M_parity(t *testing.T) {
	testGGUFParity(t, "../testdata/tinyllama-gguf/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", 0.99)
}

// Q5_K_M mixes Q5_K (most weights) and Q6_K — exercises the Q5_K dequant (the
// 5th bit from qh on top of the Q4_K scale/min packing). Tighter than Q4_K_M.
func TestGGUF_Q5_K_M_parity(t *testing.T) {
	testGGUFParity(t, "../testdata/tinyllama-gguf/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf", 0.995)
}

// Q3_K_M mixes Q3_K (most weights), Q4_K, and Q6_K — exercises the Q3_K dequant
// (the 6-bit-scale aux unpack + hmask 3rd bit). 3-bit is coarse, so a lower floor.
func TestGGUF_Q3_K_M_parity(t *testing.T) {
	testGGUFParity(t, "../testdata/tinyllama-gguf/tinyllama-1.1b-chat-v1.0.Q3_K_M.gguf", 0.98)
}

// Q2_K mixes Q2_K (most weights) with Q3_K/Q4_K/Q6_K for a few — exercises the
// Q2_K dequant (4-bit scale+min per sub-block, 2-bit quants). The coarsest quant,
// so the loosest floor; argmax must still hold.
func TestGGUF_Q2_K_parity(t *testing.T) {
	testGGUFParity(t, "../testdata/tinyllama-gguf/tinyllama-1.1b-chat-v1.0.Q2_K.gguf", 0.97)
}

// GGUF + resident int8 (streaming-quant). Loads the Q8_0 GGUF with
// Options{Quant:"int8"}: each tensor is dequantized then re-quantized to per-row
// int8 as it loads — the f32 of one tensor is freed before the next, so there is
// no whole-model f32 spike — and the forward runs off the int8 store. Asserts
// the matmul weights are resident int8 (f32 freed), and that the doubly-lossy
// path (Q8_0 base, near-lossless, + int8 re-quant) still keeps the argmax and a
// high cosine vs the f32 oracle.
func TestGGUF_int8_resident(t *testing.T) {
	if testing.Short() {
		t.Skip("slow: loads + runs a TinyLlama GGUF")
	}
	raw, err := os.ReadFile(llamaForwardGolden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no llama golden at %s — regenerate with scripts/pin_llama_forward.py", llamaForwardGolden)
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g forwardGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	ggufPath := "../testdata/tinyllama-gguf/tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
	if _, err := os.Stat(ggufPath); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no GGUF at %s", ggufPath)
	}

	m, err := Load(ggufPath, Options{Quant: "int8"})
	if err != nil {
		t.Fatalf("Load int8: %v", err)
	}
	// Resident int8: every matmul weight kept its int8 codes and freed its f32.
	for _, wm := range m.w.matmulWeights() {
		if wm.q8 == nil || wm.f32 != nil {
			t.Fatalf("matmul weight %dx%d not resident int8 (q8=%v f32=%v)",
				wm.rows, wm.cols, wm.q8 != nil, wm.f32 != nil)
		}
	}

	cache := m.NewCache(len(g.IDs))
	for _, id := range g.IDs[:len(g.IDs)-1] {
		if _, err := m.runLayers(id, cache); err != nil {
			t.Fatalf("runLayers: %v", err)
		}
	}
	logits, err := m.forward(g.IDs[len(g.IDs)-1], cache)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	if got := argmax(logits); got != g.Argmax {
		t.Errorf("argmax = %d, want %d (logit[got]=%.4f logit[want]=%.4f)",
			got, g.Argmax, logits[got], logits[g.Argmax])
	}
	cos := cosineToFull(t, logits, llamaForwardFullPath)
	if !math.IsNaN(cos) && cos < 0.99 {
		t.Errorf("cosine vs f32 oracle = %v, want ≥ 0.99", cos)
	}
	t.Logf("gguf Q8_0 + int8 resident: argmax=%d (want %d) | cosine vs f32 = %v", argmax(logits), g.Argmax, cos)
}

// GGUF + resident int4 (group-wise 4-bit projections, int8 embedding/head). The
// strict int4 accuracy gate: on TinyLlama (1.1B) 4-bit is well-tolerated — argmax
// must still match the f32 oracle and the cosine clears 0.98 (≈ GGUF Q4_K_M's own
// 0.9975). Proves the int4 resident path (~⅛ f32 on the projections) is wired
// and faithful where int4 is meant to be used.
func TestGGUF_int4_resident(t *testing.T) {
	if testing.Short() {
		t.Skip("slow: loads + runs a TinyLlama GGUF")
	}
	raw, err := os.ReadFile(llamaForwardGolden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no llama golden at %s — regenerate with scripts/pin_llama_forward.py", llamaForwardGolden)
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g forwardGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	ggufPath := "../testdata/tinyllama-gguf/tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
	if _, err := os.Stat(ggufPath); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no GGUF at %s", ggufPath)
	}

	m, err := Load(ggufPath, Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("Load int4: %v", err)
	}
	// Projections int4, embedding/head int8 (the embedding policy), f32 freed.
	if gate := &m.w.Layers[0].GateProj; gate.q4 == nil || gate.f32 != nil {
		t.Fatalf("GateProj not int4 (q4=%v f32=%v)", gate.q4 != nil, gate.f32 != nil)
	}
	if m.w.Embed.q8 == nil || m.w.Embed.f32 != nil {
		t.Fatalf("Embed not int8 (q8=%v f32=%v)", m.w.Embed.q8 != nil, m.w.Embed.f32 != nil)
	}

	cache := m.NewCache(len(g.IDs))
	for _, id := range g.IDs[:len(g.IDs)-1] {
		if _, err := m.runLayers(id, cache); err != nil {
			t.Fatalf("runLayers: %v", err)
		}
	}
	logits, err := m.forward(g.IDs[len(g.IDs)-1], cache)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	if got := argmax(logits); got != g.Argmax {
		t.Errorf("int4 argmax = %d, want %d (logit[got]=%.4f logit[want]=%.4f)",
			got, g.Argmax, logits[got], logits[g.Argmax])
	}
	cos := cosineToFull(t, logits, llamaForwardFullPath)
	if !math.IsNaN(cos) && cos < 0.98 {
		t.Errorf("int4 cosine vs f32 oracle = %v, want ≥ 0.98", cos)
	}
	t.Logf("gguf + int4 resident: argmax=%d (want %d) | cosine vs f32 = %v", argmax(logits), g.Argmax, cos)
}
