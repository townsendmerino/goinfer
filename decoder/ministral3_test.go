package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"os"
	"testing"
)

// G3 (docs/task-families-2026-09.md, batch 2) Ministral 3 parity (mistralai/Ministral-3-{3b,8b,
// 14b}, model_type "mistral3" — the outer VL wrapper's type, "ministral3" nested): Mistral's GQA
// skeleton plus YaRN RoPE with a DeepSeek-style mscale/mscale_all_dim override and Llama4-style
// attention-temperature tuning on every layer (AttnTempBeta/AttnTempOrigMaxPos). The fixture's
// prompt is deliberately longer than original_max_position_embeddings=8 (12 tokens) so the
// attn-temp scale is genuinely exercised, and mscale≠mscale_all_dim (0.5/0.8) so the YaRN
// override is a real ratio, not the trivially-1.0 case the real release's own 1.0/1.0 would let
// a broken formula pass too.
//
// Regenerate (seeded tiny Ministral3ForCausalLM checkpoint + golden, both reproducible):
//
//	~/.venv-nemotron3/bin/python scripts/pin_ministral3_tiny.py
const (
	ministral3ModelDir        = "../testdata/ministral3-tiny"
	ministral3ForwardGolden   = "../testdata/ministral3_forward_golden.json"
	ministral3ForwardFullPath = "../testdata/ministral3_forward_full.json"
)

func TestMinistral3_forwardParity(t *testing.T) {
	if testing.Short() {
		t.Skip("slow: loads + runs Ministral3-tiny")
	}
	raw, err := os.ReadFile(ministral3ForwardGolden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no Ministral3 golden at %s — regenerate with scripts/pin_ministral3_tiny.py", ministral3ForwardGolden)
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g forwardGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	if len(g.IDs) <= 8 {
		t.Fatalf("fixture prompt has %d ids, want >8 (original_max_position_embeddings) so the "+
			"attn-temp scale is actually exercised, not identity", len(g.IDs))
	}
	if _, err := os.Stat(ministral3ModelDir + "/model.safetensors"); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no Ministral3 checkpoint at %s — regenerate with scripts/pin_ministral3_tiny.py", ministral3ModelDir)
	}

	m, err := Load(ministral3ModelDir, Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if m.w.arch.Name != "mistral3" {
		t.Fatalf("resolved arch %q, want mistral3", m.w.arch.Name)
	}
	if m.w.arch.SlidingWindow != 0 {
		t.Errorf("SlidingWindow = %d, want 0 (the real releases ship sliding_window: null)", m.w.arch.SlidingWindow)
	}
	if got, want := m.w.arch.AttnTempBeta, 0.2; got != want {
		t.Errorf("AttnTempBeta = %v, want %v (llama_4_scaling_beta)", got, want)
	}
	if got, want := m.w.arch.AttnTempOrigMaxPos, 8.0; got != want {
		t.Errorf("AttnTempOrigMaxPos = %v, want %v (original_max_position_embeddings)", got, want)
	}
	if m.w.arch.ropeScaling == nil || m.w.arch.ropeScaling.kind != ropeScaleYarn {
		t.Fatalf("ropeScaling = %+v, want YaRN", m.w.arch.ropeScaling)
	}
	// Proves the DeepSeek-style mscale/mscale_all_dim override actually ran: the DEFAULT
	// (unhandled) attention_factor for factor=4 would be 0.1*ln(4)+1 ≈ 1.1386 — measurably
	// different from the ratio this fixture's distinct mscale/mscale_all_dim (0.5/0.8) produces.
	wantMscale := yarnGetMscale(4.0, 0.5) / yarnGetMscale(4.0, 0.8)
	if got := m.w.arch.ropeScaling.mscale; math.Abs(got-wantMscale) > 1e-9 {
		t.Errorf("ropeScaling.mscale = %v, want %v (the mscale/mscale_all_dim ratio, not the generic YaRN default ~1.1386)", got, wantMscale)
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
	if len(logits) != g.Vocab {
		t.Fatalf("got %d logits, want vocab %d", len(logits), g.Vocab)
	}

	if got := argmax(logits); got != g.Argmax {
		t.Errorf("argmax = %d, want %d (logit[got]=%.4f logit[want]=%.4f)",
			got, g.Argmax, logits[got], logits[g.Argmax])
	}

	const valTol = 5e-3
	var maxSampleΔ float64
	for _, kv := range g.Sample {
		id := int(kv[0])
		d := math.Abs(float64(logits[id]) - kv[1])
		if d > maxSampleΔ {
			maxSampleΔ = d
		}
		if d > valTol {
			t.Errorf("sample id=%d logit=%.5f want %.5f (Δ%.5f)", id, logits[id], kv[1], d)
		}
	}
	for r, kv := range g.TopK {
		id := int(kv[0])
		if d := math.Abs(float64(logits[id]) - kv[1]); d > valTol {
			t.Errorf("top_k[%d] id=%d logit=%.5f want %.5f (Δ%.5f)", r, id, logits[id], kv[1], d)
		}
	}

	cos := fullCosine(t, logits, ministral3ForwardFullPath)
	t.Logf("mistral3: argmax=%d (want %d) | maxSampleΔ=%.5f | cosine=%v",
		argmax(logits), g.Argmax, maxSampleΔ, cos)
	emitParityRow(t, "mistral3", "tiny-golden", "HF f32 (ministral3-tiny seeded fixture, prompt > original_max_position_embeddings to exercise attn-temp)", 100.0, cos, cos)
}

// TestMinistral3_batchedMatchesSequential proves the batched-prefill twin of the attn-temp step
// (decoder/forwardn.go) agrees with the sequential one (decoder/attention.go) — the two were
// hand-written separately (the batched path can't call the sequential helper, since pos varies
// per row), and a family whose only test drives the sequential path exclusively would never
// catch the batched copy drifting. Mirrors this repo's own established pattern
// (TestForwardN_matchesSequential) for the family this is new to.
func TestMinistral3_batchedMatchesSequential(t *testing.T) {
	if testing.Short() {
		t.Skip("slow: loads + runs Ministral3-tiny")
	}
	if _, err := os.Stat(ministral3ModelDir + "/model.safetensors"); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no Ministral3 checkpoint at %s — regenerate with scripts/pin_ministral3_tiny.py", ministral3ModelDir)
	}
	raw, err := os.ReadFile(ministral3ForwardGolden)
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g forwardGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}

	mSeq, err := Load(ministral3ModelDir, Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	cacheSeq := mSeq.NewCache(len(g.IDs))
	var seqLogits []float32
	for _, id := range g.IDs {
		seqLogits, err = mSeq.forward(id, cacheSeq)
		if err != nil {
			t.Fatalf("sequential forward: %v", err)
		}
	}

	mBatch, err := Load(ministral3ModelDir, Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if !mBatch.canBatchN(len(g.IDs)) {
		t.Fatalf("canBatchN(%d) = false for mistral3 — expected true (plain GQA, no own-forward)", len(g.IDs))
	}
	cacheBatch := mBatch.NewCache(len(g.IDs))
	batchLogits, err := mBatch.forwardN(t.Context(), g.IDs, cacheBatch)
	if err != nil {
		t.Fatalf("batched forwardN: %v", err)
	}
	last := batchLogits[len(batchLogits)-1]

	if len(last) != len(seqLogits) {
		t.Fatalf("batched produced %d logits, sequential %d", len(last), len(seqLogits))
	}
	for i := range seqLogits {
		if seqLogits[i] != last[i] {
			t.Fatalf("batched/sequential diverge at logit %d: seq=%v batch=%v — the attn-temp step's "+
				"batched twin (forwardn.go) disagrees with the sequential one (attention.go)",
				i, seqLogits[i], last[i])
		}
	}
	t.Logf("mistral3 batched==sequential: %d logits bit-identical", len(seqLogits))
}
