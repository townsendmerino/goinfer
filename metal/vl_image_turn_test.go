//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestVLImageTurn_metalResident is S3's tiny-fixture check (docs/tasks/task-multimodal-support-2026-10.md): a Gemma 3 and a
// Qwen2.5-VL image turn decode on a Metal resident, from the committed tiny checkpoints and their goldens' image
// features, against the CPU decoder (both int4). It is the first image turn on a Mac's GPU for either family; the
// registered S3 gate is the served turn on the real checkpoints.
func TestVLImageTurn_metalResident(t *testing.T) {
	const n = 8
	type golden struct {
		InputIDs        []int     `json:"input_ids"`
		ImageTokenStart int       `json:"image_token_start"`
		MMTokens        int       `json:"mm_tokens_per_image"`
		ImageToken      int       `json:"image_token_id"`
		ImageStart      int       `json:"image_token_start_qwen"`
		NImageTokens    int       `json:"n_image_tokens"`
		GridTHW         [][3]int  `json:"grid_thw"`
		ImageFeatures   []float32 `json:"image_features"`
	}
	run := func(m *decoder.Model, g golden, qwen bool) ([]int, *decoder.Generation) {
		feats := func() ([]float32, error) { return g.ImageFeatures, nil }
		var stream <-chan int
		var gen *decoder.Generation
		if qwen {
			stream, gen = m.GenerateQwenVL(context.Background(), g.InputIDs, g.ImageTokenStart, g.NImageTokens, 0, feats, g.GridTHW, 2, g.ImageToken, n, decoder.SamplingParams{Temperature: 0})
		} else {
			stream, gen = m.GenerateVL(context.Background(), g.InputIDs, g.ImageTokenStart, g.MMTokens, 0, feats, n, decoder.SamplingParams{Temperature: 0})
		}
		var out []int
		for id := range stream {
			out = append(out, id)
		}
		if err := gen.Err(); err != nil {
			t.Fatal(err)
		}
		return out, gen
	}
	for _, c := range []struct {
		name, ckpt, golden string
		qwen               bool
	}{
		{"gemma3", "../testdata/gemma3-vl-tiny", "../testdata/gemma3_vl_tiny_image_golden.json", false},
		{"qwen2.5-vl", "../testdata/qwen25vl-tiny", "../testdata/qwen25vl_tiny_image_golden.json", true},
	} {
		t.Run(c.name, func(t *testing.T) {
			raw, err := os.ReadFile(c.golden)
			if err != nil {
				t.Skipf("no golden: %v", err)
			}
			var g golden
			if err := json.Unmarshal(raw, &g); err != nil {
				t.Fatal(err)
			}
			mc, err := decoder.Load(c.ckpt, decoder.Options{Quant: "int4"})
			if err != nil {
				t.Fatal(err)
			}
			defer mc.Close()
			mg, err := decoder.Load(c.ckpt, decoder.Options{Backend: "metal", Quant: "int4"})
			if err != nil {
				t.Fatal(err)
			}
			defer mg.Close()
			path := mg.DecodePath()
			if !strings.HasPrefix(path, "metal-resident") {
				t.Fatalf("Metal did not build a resident for %s: %s", c.name, path)
			}
			cpu, _ := run(mc, g, c.qwen)
			gpu, gen := run(mg, g, c.qwen)
			agree := 0
			for agree < min(len(cpu), len(gpu)) && cpu[agree] == gpu[agree] {
				agree++
			}
			t.Logf("%s on %s: cpu %v | metal %v | %d/%d tokens agree before the first difference; image prefill resident %v",
				c.name, path, cpu, gpu, agree, n, gen.ImgPrefillResident)
			// The image prefill runs on the CPU in both arms here, so the first token comes from the same logits; decode then
			// runs on Metal in one arm and the CPU in the other, and on a tiny random int4 model a flip within a few tokens is
			// expected (G3 measured ~5% per position on a real one). Agreement is the served gate's question, on the real
			// checkpoints; this checks the turn runs on the resident at all.
			if len(gpu) != n || len(cpu) != n {
				t.Fatalf("streamed %d (metal) and %d (cpu) tokens, want %d", len(gpu), len(cpu), n)
			}
			if gpu[0] != cpu[0] {
				t.Errorf("the first token differs (%d vs %d) although both arms prefill on the CPU", gpu[0], cpu[0])
			}
		})
	}
}
