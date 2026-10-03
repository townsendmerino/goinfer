//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/vision"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/multimodal"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// GLM-OCR on the CUDA resident, on a REAL page through the REAL tower (gate O3 item 3, docs/tasks/task-glm-ocr-2026-10.md).
// TestGlmOcrResidentParityCUDA proves the pairwise kernels on a tiny fixture; these prove the composition on the real
// 1.1 B checkpoint: the image prefill through rope_kv_mrope_batched_pw, then decode past the image through ForwardMRoPE at
// ropePos != pos. int4 on both arms (the resident and the CPU), the images are the procedurally rendered documents of
// scripts/gen_glm_ocr_doc_images.py (not real scans).
//
//	GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks' ./cuda/ -run TestGlmOcrCUDAContextFit -v -timeout 20m
//	GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks' ./cuda/ -run 'TestGlmOcrCUDAResidentRealImage/invoice' -v -timeout 20m

type glmOcrRealGolden struct {
	InputIDs   []int    `json:"input_ids"`
	ImageStart int      `json:"image_token_start"`
	NImgTokens int      `json:"n_image_tokens"`
	Grid       [][3]int `json:"grid_thw"`
	HFTokens   []int    `json:"hf_tokens"`
	LastLogits string   `json:"last_logits_f32_b64"`
	NNew       int      `json:"n_new"`
}

func b64F32(t *testing.T, s string) []float32 {
	raw, err := base64.StdEncoding.DecodeString(s)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float32, len(raw)/4)
	for i := range out {
		out[i] = math.Float32frombits(uint32(raw[4*i]) | uint32(raw[4*i+1])<<8 | uint32(raw[4*i+2])<<16 | uint32(raw[4*i+3])<<24)
	}
	return out
}

// TestGlmOcrCUDAContextFit records what resident context the 8 GB card holds beside the real int4 decoder: for each
// requested context, whether the resident was built, the cap it reports, and the decline reason if not. It asserts nothing
// about the numbers (they are a property of this card); it exists so the figure in the O3 record is reproducible.
func TestGlmOcrCUDAContextFit(t *testing.T) {
	requireHeavyModel(t)
	ckpt := decoder.AssetPathForTest(t, "GOINFER_GLM_OCR")
	for _, ctx := range []int{2048, 4096, 8192, 12288, 16384, 24576, 32768, 65536} {
		m, err := decoder.Load(ckpt, decoder.Options{Backend: "cuda", Quant: "int4", ResidentContext: ctx})
		if err != nil {
			t.Logf("ResidentContext %6d: load error: %v", ctx, err)
			continue
		}
		t.Logf("ResidentContext %6d: resident=%v cap=%d decline=%q path=%s", ctx, m.ResidentActive(), m.ResidentContextCap(), m.ResidentDecline(), m.DecodePath())
		m.Close()
	}
}

// glmOcrResidentCtx is the resident context the image gate runs at: the invoice is a 1,668-token prompt, so 4,096 holds it
// and the 64 new tokens with room to spare. TestGlmOcrCUDAContextFit records how much further the card goes.
const glmOcrResidentCtx = 4096

func TestGlmOcrCUDAResidentRealImage(t *testing.T) {
	requireHeavyModel(t)
	ckpt := decoder.AssetPathForTest(t, "GOINFER_GLM_OCR")
	pp, err := multimodal.LoadGlmOcrPreprocessConfig(ckpt)
	if err != nil {
		t.Fatal(err)
	}
	enc, err := vision.LoadGlmOcrVisionEncoder(ckpt, false)
	if err != nil {
		t.Fatal(err)
	}
	mc, err := decoder.Load(ckpt, decoder.Options{Backend: "cuda", Quant: "int4", ResidentContext: glmOcrResidentCtx})
	if err != nil {
		t.Fatalf("load cuda: %v", err)
	}
	defer mc.Close()
	if !mc.ResidentActive() {
		t.Fatalf("glm_ocr is not resident at ResidentContext %d: %s", glmOcrResidentCtx, mc.ResidentDecline())
	}
	t.Logf("CUDA resident: %s (cap %d)", mc.DecodePath(), mc.ResidentContextCap())
	rmp, ok := mc.ResidentForwardForTest().(decoder.ResidentMRoPEPrefill)
	if !ok {
		t.Fatal("the resident does not satisfy ResidentMRoPEPrefill")
	}
	mcpu, err := decoder.Load(ckpt, decoder.Options{Backend: "cpu", Quant: "int4"})
	if err != nil {
		t.Fatalf("load cpu: %v", err)
	}
	defer mcpu.Close()
	if mcpu.ResidentActive() {
		t.Fatal("the CPU arm has a resident")
	}

	for _, name := range []string{"invoice", "table", "formula"} {
		t.Run(name, func(t *testing.T) {
			var g glmOcrRealGolden
			raw, err := readGolden(filepath.Join("..", "testdata", "glm_ocr", "golden_"+name+".json.gz"))
			if err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(raw, &g); err != nil {
				t.Fatal(err)
			}
			if len(g.InputIDs)+g.NNew > glmOcrResidentCtx {
				t.Skipf("%d prompt tokens + %d new do not fit the %d-token context this test runs at", len(g.InputIDs), g.NNew, glmOcrResidentCtx)
			}
			png, err := os.ReadFile(filepath.Join("..", "testdata", "glm_ocr", name+".png"))
			if err != nil {
				t.Fatal(err)
			}
			pv, grid, err := multimodal.QwenPreprocess(png, pp)
			if err != nil {
				t.Fatal(err)
			}
			t0 := time.Now()
			feats, err := enc.Forward(pv, [][3]int{grid})
			if err != nil {
				t.Fatal(err)
			}
			fmt.Fprintf(os.Stderr, "[glm-ocr cuda] %s: tower %.0fs\n", name, time.Since(t0).Seconds())
			const imageToken, merge = 59280, 2
			mropePos, err := decoder.MRopePositionsForTest(g.InputIDs, imageToken, [][3]int{grid}, merge)
			if err != nil {
				t.Fatal(err)
			}

			// Half 1: the prefill logits, resident m-RoPE prefill against the CPU prefill (same int4 weights).
			t0 = time.Now()
			cpuLogits, err := mcpu.PrefillLogitsQwenVLForTest(context.Background(), g.InputIDs, feats, g.ImageStart, g.NImgTokens, mropePos, mcpu.NewCache(len(g.InputIDs)+8))
			if err != nil {
				t.Fatal(err)
			}
			fmt.Fprintf(os.Stderr, "[glm-ocr cuda] %s: CPU prefill %.0fs\n", name, time.Since(t0).Seconds())
			resLogits, gpuPos, err := mc.ResidentMRoPEPrefillForTest(context.Background(), rmp, g.InputIDs, feats, g.ImageStart, g.NImgTokens, mropePos)
			if err != nil {
				t.Fatalf("resident m-RoPE prefill: %v", err)
			}
			if gpuPos != len(g.InputIDs) {
				t.Errorf("gpuPos %d, want %d", gpuPos, len(g.InputIDs))
			}
			hf := b64F32(t, g.LastLogits)
			cs := cosine(cpuLogits, resLogits)
			t.Logf("prefill last-token logits, CUDA resident vs CPU (int4 both): cosine %.6f, argmax resident %d CPU %d HF %d; "+
				"cosine to HF f32: CPU %.6f resident %.6f", cs, argmaxF(resLogits), argmaxF(cpuLogits), argmaxF(hf), cosine(cpuLogits, hf), cosine(resLogits, hf))
			if cs < 0.99 {
				t.Errorf("resident-vs-CPU prefill cosine %.6f < 0.99", cs)
			}
			if argmaxF(resLogits) != argmaxF(cpuLogits) {
				t.Errorf("argmax resident %d != CPU %d", argmaxF(resLogits), argmaxF(cpuLogits))
			}

			// Half 2: the whole turn through GenerateQwenVL, 64 greedy tokens, resident then CPU.
			features := func() ([]float32, error) { return feats, nil }
			run := func(m *decoder.Model, label string) ([]int, *decoder.Generation) {
				t0 := time.Now()
				stream, gen := m.GenerateQwenVL(context.Background(), g.InputIDs, g.ImageStart, g.NImgTokens, 0, features, [][3]int{grid}, merge, imageToken, g.NNew, decoder.SamplingParams{Temperature: 0})
				var out []int
				for id := range stream {
					out = append(out, id)
				}
				if err := gen.Err(); err != nil {
					t.Fatalf("%s: %v", label, err)
				}
				fmt.Fprintf(os.Stderr, "[glm-ocr cuda] %s: %s generate %.0fs\n", name, label, time.Since(t0).Seconds())
				return out, gen
			}
			resToks, rg := run(mc, "resident")
			cpuToks, cg := run(mcpu, "cpu")
			if !rg.ImgPrefillResident {
				t.Error("the resident turn did not take the resident m-RoPE image prefill (ImgPrefillResident false)")
			}
			if cg.ImgPrefillResident {
				t.Error("the CPU arm reports a resident prefill")
			}
			first := -1
			for i := 0; i < max(len(resToks), len(cpuToks)); i++ {
				if i >= len(resToks) || i >= len(cpuToks) || resToks[i] != cpuToks[i] {
					first = i
					break
				}
			}
			ref := g.HFTokens
			t.Logf("greedy %d tokens: resident vs CPU identical=%v (first divergence %d); resident vs HF f32 first divergence %d, CPU int4 vs HF f32 first divergence %d",
				g.NNew, first < 0, first, firstDiff(resToks, ref), firstDiff(cpuToks, ref))
			if tk, terr := tokenizer.Load(ckpt); terr == nil { // the library-level text the serve HTTP smoke is compared against
				rt, _ := tk.Decode(resToks)
				ct, _ := tk.Decode(cpuToks)
				ht, _ := tk.Decode(ref)
				t.Logf("TEXT resident int4 : %q\nTEXT cpu int4      : %q\nTEXT HF f32        : %q", rt, ct, ht)
			}
			if first >= 0 {
				t.Errorf("resident and CPU int4 greedy tokens diverge at step %d:\n resident %v\n cpu      %v", first, resToks, cpuToks)
			}
			if !slices.Equal(resToks, cpuToks) && first < 0 {
				t.Error("unreachable")
			}
		})
	}
}

func firstDiff(a, b []int) int {
	for i := 0; i < max(len(a), len(b)); i++ {
		if i >= len(a) || i >= len(b) || a[i] != b[i] {
			return i
		}
	}
	return -1
}

// TestGlmOcrCUDAResidentFullBudget runs the processor's CEILING, a 6,144-image-token prompt (the 4.8 MP, 24,576-patch
// budget; grid 128x192), through the CUDA resident at ResidentContext 8192, against the CPU prefill (int4 both), and then 16
// greedy tokens through GenerateQwenVL on both. The tower for a real 4.8 MP page costs minutes of CPU, so the image rows are
// the REAL invoice tower rows (real feature statistics) tiled to 6,144: the point here is the decoder at that length (the
// resident m-RoPE prefill's chunking, the KV at 6.2k positions, decode past a 128x192 grid whose position delta is large),
// not what the model reads. The prompt shape is the real template's.
func TestGlmOcrCUDAResidentFullBudget(t *testing.T) {
	requireHeavyModel(t)
	ckpt := decoder.AssetPathForTest(t, "GOINFER_GLM_OCR")
	pp, err := multimodal.LoadGlmOcrPreprocessConfig(ckpt)
	if err != nil {
		t.Fatal(err)
	}
	enc, err := vision.LoadGlmOcrVisionEncoder(ckpt, false)
	if err != nil {
		t.Fatal(err)
	}
	png, err := os.ReadFile(filepath.Join("..", "testdata", "glm_ocr", "invoice.png"))
	if err != nil {
		t.Fatal(err)
	}
	pv, grid, err := multimodal.QwenPreprocess(png, pp)
	if err != nil {
		t.Fatal(err)
	}
	base, err := enc.Forward(pv, [][3]int{grid})
	if err != nil {
		t.Fatal(err)
	}
	const hidden, nImg = 1536, 6144
	feats := make([]float32, 0, nImg*hidden)
	for len(feats) < nImg*hidden {
		feats = append(feats, base[:min(len(base), nImg*hidden-len(feats))]...)
	}
	fullGrid := [3]int{1, 128, 192}
	if multimodal.QwenMergedTokens(fullGrid, 2) != nImg {
		t.Fatal("grid arithmetic")
	}
	const imageToken = 59280
	ids := slices.Concat([]int{59248, 59250, 59253, 10, 59256}, slices.Repeat([]int{imageToken}, nImg), []int{59257, 3649, 7404, 49600, 58, 59254, 10})
	const ctx, maxNew = 8192, 16
	if len(ids)+maxNew > ctx {
		t.Fatalf("%d + %d > %d", len(ids), maxNew, ctx)
	}
	mc, err := decoder.Load(ckpt, decoder.Options{Backend: "cuda", Quant: "int4", ResidentContext: ctx})
	if err != nil {
		t.Fatal(err)
	}
	defer mc.Close()
	if !mc.ResidentActive() {
		t.Fatalf("not resident at %d: %s", ctx, mc.ResidentDecline())
	}
	rmp, ok := mc.ResidentForwardForTest().(decoder.ResidentMRoPEPrefill)
	if !ok {
		t.Fatal("no ResidentMRoPEPrefill")
	}
	mcpu, err := decoder.Load(ckpt, decoder.Options{Backend: "cpu", Quant: "int4"})
	if err != nil {
		t.Fatal(err)
	}
	defer mcpu.Close()
	mropePos, err := decoder.MRopePositionsForTest(ids, imageToken, [][3]int{fullGrid}, 2)
	if err != nil {
		t.Fatal(err)
	}
	t0 := time.Now()
	cpuLogits, err := mcpu.PrefillLogitsQwenVLForTest(context.Background(), ids, feats, 5, nImg, mropePos, mcpu.NewCache(len(ids)+maxNew))
	if err != nil {
		t.Fatal(err)
	}
	fmt.Fprintf(os.Stderr, "[glm-ocr cuda] full budget: CPU prefill of %d tokens %.0fs\n", len(ids), time.Since(t0).Seconds())
	resLogits, gpuPos, err := mc.ResidentMRoPEPrefillForTest(context.Background(), rmp, ids, feats, 5, nImg, mropePos)
	if err != nil {
		t.Fatalf("resident m-RoPE prefill at %d tokens: %v", len(ids), err)
	}
	if gpuPos != len(ids) {
		t.Errorf("gpuPos %d want %d", gpuPos, len(ids))
	}
	cs := cosine(cpuLogits, resLogits)
	t.Logf("%d-token prompt (%d image tokens, grid %v): resident vs CPU prefill logits cosine %.6f, argmax resident %d CPU %d", len(ids), nImg, fullGrid, cs, argmaxF(resLogits), argmaxF(cpuLogits))
	if cs < 0.99 || argmaxF(resLogits) != argmaxF(cpuLogits) {
		t.Errorf("resident prefill at the full image budget diverges from the CPU: cosine %.6f", cs)
	}
	features := func() ([]float32, error) { return feats, nil }
	run := func(m *decoder.Model) ([]int, *decoder.Generation) {
		stream, gen := m.GenerateQwenVL(context.Background(), ids, 5, nImg, 0, features, [][3]int{fullGrid}, 2, imageToken, maxNew, decoder.SamplingParams{Temperature: 0})
		var out []int
		for id := range stream {
			out = append(out, id)
		}
		if err := gen.Err(); err != nil {
			t.Fatal(err)
		}
		return out, gen
	}
	rt, rg := run(mc)
	ct, _ := run(mcpu)
	t.Logf("GenerateQwenVL %d tokens: resident ImgPrefillResident=%v, identical to CPU int4: %v (first divergence %d)", maxNew, rg.ImgPrefillResident, slices.Equal(rt, ct), firstDiff(rt, ct))
	if !rg.ImgPrefillResident {
		t.Error("the full-budget turn did not take the resident m-RoPE prefill (it fell back to the CPU prefill + upload)")
	}
	if !slices.Equal(rt, ct) {
		t.Errorf("tokens differ: resident %v cpu %v", rt, ct)
	}
	_ = pp
}
