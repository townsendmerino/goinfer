//go:build realckpt

package decoder

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/aikit/vision"
	"github.com/townsendmerino/goinfer/multimodal"
)

// TestQwen35VLReal_mmproj is P8b's F5a and F5b (docs/multimodal.md, "Finishing this doc"; the Phase 0 amendments are
// above it there), on nobara's downloads (GOINFER_MMPROJ_DIR, default ~/models/mmproj):
//
//	F5a: the Qwen3.5-0.8B tower loaded from each mmproj (unsloth F32, F16 and BF16, Ollama's projector blob) against the
//	     same tower from the safetensors checkpoint, on the three P8a G2 images through goinfer's own preprocessing:
//	     every merged token at cosine >= 0.9999; the max |diff| is reported, and the containers whose weights are exact
//	     (F32, BF16, Ollama) must give 0. The 9B pair (Ollama's 9B projector, the 9B checkpoint) is read the same way
//	     when its checkpoint is present.
//	F5b: the same text GGUF (unsloth's Qwen3.5-0.8B-Q8_0) with the tower from the BF16 mmproj, against the same text
//	     GGUF with the safetensors tower: identical 32 greedy tokens on all three images through GenerateQwenVL.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags realckpt ./decoder/ -run TestQwen35VLReal_mmproj -v -timeout 30m
func TestQwen35VLReal_mmproj(t *testing.T) {
	requireHeavyModel(t)
	home, _ := os.UserHomeDir()
	mdir := os.Getenv("GOINFER_MMPROJ_DIR")
	if mdir == "" {
		mdir = filepath.Join(home, "models", "mmproj")
	}
	ckpt, g2 := assetPath(t, "GOINFER_QWEN35VL_08B"), assetPath(t, "GOINFER_QWEN35VL_G2")
	pp, err := multimodal.LoadQwen3PreprocessConfig(ckpt)
	if err != nil {
		t.Fatal(err)
	}
	ref, err := vision.LoadQwen3VisionEncoder(ckpt, false)
	if err != nil {
		t.Fatal(err)
	}
	type img struct {
		pv   []float32
		grid [3]int
		g    struct {
			InputIDs     []int    `json:"input_ids"`
			ImageToken   int      `json:"image_token_id"`
			ImageStart   int      `json:"image_token_start"`
			NImageTokens int      `json:"n_image_tokens"`
			GridTHW      [][3]int `json:"grid_thw"`
			HFTokens     []int    `json:"hf_tokens"`
		}
		ref []float32
	}
	var imgs []*img
	for _, k := range []string{"A", "B", "C"} {
		im := &img{}
		raw, err := os.ReadFile(filepath.Join(g2, "golden_"+k+".json"))
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(raw, &im.g); err != nil {
			t.Fatal(err)
		}
		png, err := os.ReadFile(filepath.Join(g2, "img_"+k+".png"))
		if err != nil {
			t.Fatal(err)
		}
		if im.pv, im.grid, err = multimodal.QwenPreprocess(png, pp); err != nil {
			t.Fatal(err)
		}
		if im.ref, err = ref.Forward(im.pv, [][3]int{im.grid}); err != nil {
			t.Fatal(err)
		}
		imgs = append(imgs, im)
	}
	out := ref.Cfg.OutHiddenSize
	// F5a
	for _, c := range []struct {
		file  string
		exact bool
	}{{"qwen3.5-0.8b-unsloth-mmproj-F32.gguf", true}, {"qwen3.5-0.8b-unsloth-mmproj-BF16.gguf", true},
		{"qwen3.5-0.8b-ollama-projector.gguf", true}, {"qwen3.5-0.8b-unsloth-mmproj-F16.gguf", false}} {
		enc, err := vision.LoadQwen3VisionEncoderMMProj(filepath.Join(mdir, c.file), false)
		if err != nil {
			t.Fatalf("%s: %v", c.file, err)
		}
		for i, im := range imgs {
			f, err := enc.Forward(im.pv, [][3]int{im.grid})
			if err != nil {
				t.Fatal(err)
			}
			w := rowCosMin(f, im.ref, out)
			md := 0.0
			for j := range f {
				md = math.Max(md, math.Abs(float64(f[j]-im.ref[j])))
			}
			t.Logf("F5a %-40s image %c: worst merged-token cosine %.9f, max|diff| %.3g", c.file, 'A'+i, w, md)
			if w < 0.9999 {
				t.Errorf("F5a %s image %c: worst cosine %.9f under 0.9999", c.file, 'A'+i, w)
			}
			if c.exact && md != 0 {
				t.Errorf("F5a %s image %c: max|diff| %.3g on a container whose weights are exact (amendment: investigate)", c.file, 'A'+i, md)
			}
		}
	}
	// F5b
	gguf := filepath.Join(mdir, "qwen3.5-0.8b-unsloth-Qwen3.5-0.8B-Q8_0.gguf")
	m, err := Load(gguf, Options{})
	if err != nil {
		t.Fatalf("text GGUF: %v", err)
	}
	defer m.Close()
	mm, err := vision.LoadQwen3VisionEncoderMMProj(filepath.Join(mdir, "qwen3.5-0.8b-unsloth-mmproj-BF16.gguf"), false)
	if err != nil {
		t.Fatal(err)
	}
	gen := func(im *img, feats []float32) []int {
		stream, g := m.GenerateQwenVL(context.Background(), im.g.InputIDs, im.g.ImageStart, im.g.NImageTokens, 0,
			func() ([]float32, error) { return feats, nil }, im.g.GridTHW, pp.MergeSize, im.g.ImageToken, 32, SamplingParams{Temperature: 0})
		var got []int
		for id := range stream {
			got = append(got, id)
		}
		if err := g.Err(); err != nil {
			t.Fatal(err)
		}
		return got
	}
	for i, im := range imgs {
		f, err := mm.Forward(im.pv, [][3]int{im.grid})
		if err != nil {
			t.Fatal(err)
		}
		a, b := gen(im, f), gen(im, im.ref)
		same := len(a) == len(b)
		for j := 0; same && j < len(a); j++ {
			same = a[j] == b[j]
		}
		t.Logf("F5b image %c: Q8_0 text + mmproj tower vs + safetensors tower: %d tokens, identical %v", 'A'+i, len(a), same)
		if !same {
			t.Errorf("F5b image %c: tokens differ\n mmproj      %v\n safetensors %v", 'A'+i, a, b)
		}
	}
	// The 9B pair, when its checkpoint is present (the asset registry's predicate; absent is not a failure here).
	if ck9, err := lookupAsset("GOINFER_QWEN35VL_9B"); err == nil {
		enc9, err := vision.LoadQwen3VisionEncoderMMProj(filepath.Join(mdir, "qwen3.5-9b-ollama-projector.gguf"), false)
		if err != nil {
			t.Fatal(err)
		}
		ref9, err := vision.LoadQwen3VisionEncoder(ck9, false)
		if err != nil {
			t.Fatal(err)
		}
		pp9, err := multimodal.LoadQwen3PreprocessConfig(ck9)
		if err != nil {
			t.Fatal(err)
		}
		png, _ := os.ReadFile(filepath.Join(g2, "img_A.png"))
		pv, grid, err := multimodal.QwenPreprocess(png, pp9)
		if err != nil {
			t.Fatal(err)
		}
		a, err := enc9.Forward(pv, [][3]int{grid})
		if err != nil {
			t.Fatal(err)
		}
		b, err := ref9.Forward(pv, [][3]int{grid})
		if err != nil {
			t.Fatal(err)
		}
		w := rowCosMin(a, b, ref9.Cfg.OutHiddenSize)
		md := 0.0
		for j := range a {
			md = math.Max(md, math.Abs(float64(a[j]-b[j])))
		}
		t.Logf("F5a 9B Ollama projector vs the 9B checkpoint, image A: worst cosine %.9f, max|diff| %.3g", w, md)
		if w < 0.9999 || md != 0 {
			t.Errorf("F5a 9B: worst cosine %.9f, max|diff| %.3g", w, md)
		}
	}
}
