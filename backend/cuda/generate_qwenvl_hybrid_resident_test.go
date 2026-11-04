//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// P26b (docs/queue-performance.md): an image turn on a Gated-DeltaNet hybrid runs its prefill AND decode on the CUDA resident, through
// GenerateQwenVL's resident m-RoPE prefill branch, which a recurrent family could not enter before (it had no hybrid m-RoPE prefill and
// no way to upload recurrent state). The tiny Qwen3.5 vision-language fixture, the same image prompt and features on both sides, both at
// int4 so the only difference is where it runs.
//
// What this asserts, and why each part is not circular:
//   - the GPU run reports ImgPrefillResident (the new branch ran) and the CPU run does not (it is the reference, not the same path);
//   - the m-RoPE layout the resident used is the interleaved one (mode 1), so a resident rotating by the wrong rule would diverge;
//   - the greedy continuation after the image is the same on both: that exercises the recurrent state the resident prefill built,
//     the m-RoPE decode position (pos + delta) and the KV, since every decode token depends on all three;
//   - a SECOND request on the same resident (the same image, a new question) still matches the CPU: the resident's recorded ids for a
//     recurrent family are only ever a strict extension (residentReuseLen), and nothing here may let a stale record resume from state
//     the first turn left behind.
func TestGenerateQwenVL_hybridResidentPrefillMatchesCPU(t *testing.T) {
	requireCUDADevice(t)
	path := filepath.Join("..", "testdata", "qwen35vl-tiny")
	rawG, err := os.ReadFile(filepath.Join("..", "testdata", "qwen35vl_tiny_image_golden.json"))
	if err != nil {
		t.Skipf("no golden: %v", err)
	}
	if _, err := os.Stat(filepath.Join(path, "model.safetensors")); err != nil {
		t.Skipf("no fixture at %s", path)
	}
	var g struct {
		InputIDs      []int     `json:"input_ids"`
		ImageToken    int       `json:"image_token_id"`
		ImageStart    int       `json:"image_token_start"`
		NImageTokens  int       `json:"n_image_tokens"`
		GridTHW       [][3]int  `json:"grid_thw"`
		ImageFeatures []float32 `json:"image_features"`
	}
	if err := json.Unmarshal(rawG, &g); err != nil {
		t.Fatal(err)
	}
	const merge = 2 // vision_config.spatial_merge_size

	gpuM, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4", ResidentContext: 4096})
	if err != nil {
		t.Fatalf("load (cuda): %v", err)
	}
	defer gpuM.Close()
	r, ok := gpuM.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Skipf("not CUDA-resident (%T)", gpuM.ResidentForwardForTest())
	}
	if !r.HybridMRoPEPrefill() {
		t.Skip("the resident does not claim hybrid m-RoPE prefill on this build")
	}
	if r.mropeMode != 1 {
		t.Fatalf("m-RoPE mode %d for the interleaved Qwen3.5 layout, want 1", r.mropeMode)
	}
	cpuM, err := decoder.Load(path, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cpu): %v", err)
	}
	defer cpuM.Close()

	run := func(m *decoder.Model, ids []int, hash uint64) ([]int, *decoder.Generation) {
		stream, gen := m.GenerateQwenVL(context.Background(), ids, g.ImageStart, g.NImageTokens, hash,
			func() ([]float32, error) { return g.ImageFeatures, nil }, g.GridTHW, merge, g.ImageToken, 8, decoder.SamplingParams{})
		var out []int
		for id := range stream {
			out = append(out, id)
		}
		if e := gen.Err(); e != nil {
			t.Fatalf("generation: %v", e)
		}
		return out, gen
	}

	// The continuation above is a coarse check on a tiny random model (it did not react to a wrong m-RoPE layout in the mutation run),
	// so the last-token LOGITS of the resident prefill are held to the CPU prefill's too. The 0.9999 bar is the one the HF-golden tests
	// use for the same fixture (decoder/qwen35vl_test.go); dropping the image rotation reads ~0.99 there.
	mropePos, err := decoder.MRopePositionsForTest(g.InputIDs, g.ImageToken, g.GridTHW, merge)
	if err != nil {
		t.Fatal(err)
	}
	wantLogits, err := cpuM.PrefillLogitsQwenVLForTest(context.Background(), g.InputIDs, g.ImageFeatures, g.ImageStart, g.NImageTokens, mropePos, cpuM.NewCache(len(g.InputIDs)+8))
	if err != nil {
		t.Fatalf("cpu prefill: %v", err)
	}
	r.Reset()
	gotLogits, _, err := gpuM.ResidentMRoPEPrefillForTest(context.Background(), r, g.InputIDs, g.ImageFeatures, g.ImageStart, g.NImageTokens, mropePos)
	if err != nil {
		t.Fatalf("resident m-RoPE prefill: %v", err)
	}
	if c := logitCos(gotLogits, wantLogits); c < 0.9999 {
		t.Errorf("resident vs CPU last-token logits cosine %.6f < 0.9999 (same int4 weights, same image features)", c)
	} else {
		t.Logf("resident vs CPU last-token logits cosine %.6f", c)
	}

	// A second turn: the same image and prompt followed by extra text, so it is a strict extension of the first prompt's ids.
	second := slices.Concat(g.InputIDs, []int{5, 6, 7})
	for turn, ids := range [][]int{g.InputIDs, second} {
		want, cg := run(cpuM, ids, 0xfeed)
		got, gg := run(gpuM, ids, 0xfeed)
		if cg.ImgPrefillResident {
			t.Fatalf("turn %d: the CPU reference reports a resident prefill; the comparison would be a path against itself", turn+1)
		}
		if !gg.ImgPrefillResident {
			t.Fatalf("turn %d: the resident prefill did not run (ImgPrefillResident=false), so this compares the CPU path with itself", turn+1)
		}
		if !slices.Equal(got, want) {
			t.Errorf("turn %d: resident continuation %v, CPU %v (same int4 weights, same image features and prompt)", turn+1, got, want)
		}
	}
}

func logitCos(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-30)
}
