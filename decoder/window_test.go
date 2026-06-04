package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"os"
	"testing"
)

// M5 sliding-window parity. Runs the forward over a >512-token prompt so the
// local (sliding_attention) layers evict early keys while the global
// (full_attention) layers attend everything, and asserts the last-position
// logits still match the HF float32 oracle. This is the only attention path
// M3/M4 left untested — their sequences were far below the 512 window, so
// local==causal there.
//
// Regenerate the oracle:
//
//	.venv/bin/python scripts/pin_gemma_window.py
const (
	gemmaWindowGoldenPath = "../testdata/gemma_window_golden.json"
	gemmaWindowFullPath   = "../testdata/gemma_window_full.json"
)

func TestForward_slidingWindowParity(t *testing.T) {
	if testing.Short() {
		t.Skip("slow: ~750-token prefill on the naive backend (M7 perf pending)")
	}
	raw, err := os.ReadFile(gemmaWindowGoldenPath)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no window golden at %s — regenerate with scripts/pin_gemma_window.py", gemmaWindowGoldenPath)
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g forwardGolden // same shape as the M3 golden (extra window fields ignored)
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	if _, err := os.Stat(gemmaModelDir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no checkpoint at %s — huggingface-cli download google/gemma-3-270m --local-dir %s",
			gemmaModelDir, gemmaModelDir)
	}
	m, err := Load(gemmaModelDir, Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if g.IDs[len(g.IDs)-1] == 0 || len(g.IDs) <= m.w.Cfg.SlidingWindow {
		t.Fatalf("golden prompt (%d ids) must exceed the sliding window (%d)", len(g.IDs), m.w.Cfg.SlidingWindow)
	}

	// Prefill the long prompt; the last token's forward yields the logits.
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

	var sum float64
	for _, v := range logits {
		sum += float64(v)
	}
	if d := relDiff(sum, g.Stats.Sum); d > 1e-3 {
		t.Errorf("sum = %.4f, want %.4f (rel %.2e)", sum, g.Stats.Sum, d)
	}

	cos := fullCosine(t, logits, gemmaWindowFullPath)
	t.Logf("%d tokens (window %d) | argmax=%d (want %d) | maxSampleΔ=%.5f | cosine=%v",
		len(g.IDs), m.w.Cfg.SlidingWindow, argmax(logits), g.Argmax, maxSampleΔ, cos)
}
