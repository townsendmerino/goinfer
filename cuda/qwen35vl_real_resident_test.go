//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestQwen35VLReal_residentImagePrefillMatchesCPU is the real-checkpoint gate for the CUDA-resident image prefill of a Gated-DeltaNet
// hybrid (P26b, docs/queue-performance.md): Qwen3.5-0.8B, the three G2 images (docs/measurements/p8a-qwen35-vl-2026-09), HF's own
// image features fed to both sides so only the decoder differs, F32 on both arms so the only difference is where it runs.
//
// WHAT THIS GATE IS, AND WHAT CHANGED AFTER THE FIRST TWO RUNS. It is a GROSS-ERROR gate, not a precision gate. It was first written
// at int4 with a last-logits cosine bar of 0.999 and read 0.951 / 0.975 / 0.946; at f32 it read 0.973 / 0.975 / 0.956 against bars of
// 0.9999. Both bars were wrong, not the path: the controls (docs/measurements/p26b-cuda-hybrid-image-prefill-2026-10-06/) show that on
// this 0.8B the CUDA resident sits at about 0.97 to 0.98 cosine from the CPU's f32 logits for ANY prompt, because the existing, shipped
// resident TEXT prefill reads 0.978 on the same model; int4 is only 0.85 from f32; and injecting the WRONG m-RoPE layout into the
// resident moved the image number from 0.9728 to 0.9699, i.e. inside that noise. So a real-checkpoint cosine cannot resolve the layout.
// The layout is gated where it can be resolved: the kernel against decoder.ApplyMRoPEForTest in both modes
// (TestRopeKVMRoPEBatched_interleavedModeMatchesCPUReference) and the tiny hybrid fixture end to end, where the wrong mode reads
// 0.99955 against 0.99995 (TestGenerateQwenVL_hybridResidentPrefillMatchesCPU).
//
// What this real gate does catch is what the tiny one cannot: a path that is wrong on a real, deep checkpoint in a gross way (recurrent
// state not built, KV garbage, a stale record reused), none of which a 0.97 cosine could hide. f32 on both arms, HF's own image features
// to both sides, so only where it runs differs. Bars, amended from the controls and stated here so they are not read as pre-registered:
//
//   - last-token logits cosine >= 0.9 (observed 0.956 to 0.975; a zeroed recurrent state or wrong KV falls far below) and the same argmax;
//
//   - GenerateQwenVL through the resident branch (ImgPrefillResident true; the CPU run's false) agrees on the first 8 generated tokens
//     (observed 16 of 16 on all three images).
//
//     GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks' ./cuda/ -run TestQwen35VLReal_residentImagePrefillMatchesCPU -v -timeout 30m
func TestQwen35VLReal_residentImagePrefillMatchesCPU(t *testing.T) {
	requireHeavyModel(t)
	requireCUDADevice(t)
	ckpt := decoder.AssetPathForTest(t, "GOINFER_QWEN35VL_08B")
	dir := decoder.AssetPathForTest(t, "GOINFER_QWEN35VL_G2")
	const merge = 2

	gpuM, err := decoder.Load(ckpt, decoder.Options{Backend: "cuda", ResidentContext: 4096})
	if err != nil {
		t.Fatalf("load (cuda): %v", err)
	}
	defer gpuM.Close()
	r, ok := gpuM.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatalf("Qwen3.5-0.8B did not go CUDA-resident (%T)", gpuM.ResidentForwardForTest())
	}
	if !r.HybridMRoPEPrefill() {
		t.Fatal("the resident does not claim hybrid m-RoPE prefill for Qwen3.5-0.8B")
	}
	cpuM, err := decoder.Load(ckpt, decoder.Options{})
	if err != nil {
		t.Fatalf("load (cpu): %v", err)
	}
	defer cpuM.Close()

	for _, k := range []string{"A", "B", "C"} {
		t.Run(k, func(t *testing.T) {
			var g struct {
				InputIDs      []int     `json:"input_ids"`
				ImageToken    int       `json:"image_token_id"`
				ImageStart    int       `json:"image_token_start"`
				NImageTokens  int       `json:"n_image_tokens"`
				GridTHW       [][3]int  `json:"grid_thw"`
				ImageFeatures []float32 `json:"image_features"`
			}
			raw, err := os.ReadFile(filepath.Join(dir, "golden_"+k+".json"))
			if err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(raw, &g); err != nil {
				t.Fatal(err)
			}
			mropePos, err := decoder.MRopePositionsForTest(g.InputIDs, g.ImageToken, g.GridTHW, merge)
			if err != nil {
				t.Fatal(err)
			}
			want, err := cpuM.PrefillLogitsQwenVLForTest(context.Background(), g.InputIDs, g.ImageFeatures, g.ImageStart, g.NImageTokens, mropePos, cpuM.NewCache(len(g.InputIDs)+16))
			if err != nil {
				t.Fatalf("cpu prefill: %v", err)
			}
			r.Reset()
			got, _, err := gpuM.ResidentMRoPEPrefillForTest(context.Background(), r, g.InputIDs, g.ImageFeatures, g.ImageStart, g.NImageTokens, mropePos)
			if err != nil {
				t.Fatalf("resident prefill: %v", err)
			}
			cos := logitCos(got, want)
			t.Logf("%s: %d prompt tokens, resident vs CPU last-token logits cosine %.6f, argmax %d vs %d", k, len(g.InputIDs), cos, argmaxF(got), argmaxF(want))
			if cos < 0.9 {
				t.Errorf("cosine %.6f < 0.9", cos)
			}
			if argmaxF(got) != argmaxF(want) {
				t.Errorf("argmax resident %d, CPU %d", argmaxF(got), argmaxF(want))
			}

			gen := func(m *decoder.Model) ([]int, *decoder.Generation) {
				stream, gn := m.GenerateQwenVL(context.Background(), g.InputIDs, g.ImageStart, g.NImageTokens, 0xabc,
					func() ([]float32, error) { return g.ImageFeatures, nil }, g.GridTHW, merge, g.ImageToken, 16, decoder.SamplingParams{})
				var out []int
				for id := range stream {
					out = append(out, id)
				}
				if e := gn.Err(); e != nil {
					t.Fatalf("generation: %v", e)
				}
				return out, gn
			}
			cpuIDs, cg := gen(cpuM)
			gpuIDs, gg := gen(gpuM)
			if cg.ImgPrefillResident || !gg.ImgPrefillResident {
				t.Fatalf("paths: CPU ImgPrefillResident=%v, resident %v; want false and true", cg.ImgPrefillResident, gg.ImgPrefillResident)
			}
			same := 0
			for same < len(cpuIDs) && same < len(gpuIDs) && cpuIDs[same] == gpuIDs[same] {
				same++
			}
			t.Logf("%s: the continuations agree on the first %d of %d tokens", k, same, len(cpuIDs))
			if same < 8 {
				t.Errorf("the continuations agree on only the first %d tokens, want 8: CPU %v, resident %v", same, cpuIDs, gpuIDs)
			}
		})
	}
}
