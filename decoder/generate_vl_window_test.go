package decoder

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
)

// TestGenerateVL_refusesImageBlockLongerThanWindow is F3 of docs/multimodal.md ("Finishing this doc"): the
// bidirectional image-block mask (KVCache.attendHi) does not extend a sliding layer's keys back to the block's start,
// so GenerateVL refuses an image block longer than the sliding window, by name, before the tower runs or a token is
// emitted. Driven through GenerateVL itself on the tiny Gemma 3 VL model, with its window narrowed to one token under
// the image block; at the block's own length it is not refused.
func TestGenerateVL_refusesImageBlockLongerThanWindow(t *testing.T) {
	raw, err := os.ReadFile("../testdata/gemma3_vl_tiny_image_golden.json")
	if err != nil {
		t.Skipf("no image golden: %v", err)
	}
	var g struct {
		InputIDs        []int     `json:"input_ids"`
		ImageTokenStart int       `json:"image_token_start"`
		MMTokens        int       `json:"mm_tokens_per_image"`
		ImageFeatures   []float32 `json:"image_features"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	m, err := Load("../testdata/gemma3-vl-tiny", Options{})
	if err != nil {
		t.Skipf("no checkpoint: %v", err)
	}
	defer m.Close()
	saved := m.w.arch.SlidingWindow
	defer func() { m.w.arch.SlidingWindow = saved }()
	run := func(window int) (int, bool, error) {
		m.w.arch.SlidingWindow = window
		towerRan := false
		features := func() ([]float32, error) { towerRan = true; return g.ImageFeatures, nil }
		stream, gen := m.GenerateVL(context.Background(), g.InputIDs, g.ImageTokenStart, g.MMTokens, 0, features, 2, SamplingParams{Temperature: 0})
		n := 0
		for range stream {
			n++
		}
		return n, towerRan, gen.Err()
	}
	n, towerRan, err := run(g.MMTokens - 1)
	if err == nil || !strings.Contains(err.Error(), "sliding window") {
		t.Fatalf("window %d under a %d-token image block: err %v, want a refusal naming the sliding window", g.MMTokens-1, g.MMTokens, err)
	}
	if n != 0 || towerRan {
		t.Fatalf("refused, yet %d tokens streamed and the tower ran=%v", n, towerRan)
	}
	if n, _, err := run(g.MMTokens); err != nil || n == 0 {
		t.Fatalf("window equal to the block: %d tokens, err %v (want it served)", n, err)
	}
}
