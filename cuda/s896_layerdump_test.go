//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/vision"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/loadflags"
	"github.com/townsendmerino/goinfer/multimodal"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestS896_layerDump is the 896-pixel investigation's Go half (docs/tasks/task-multimodal-support-2026-10.md, "The 896-pixel image prefill"): per-layer differencing, which CLAUDE.md prefers
// over reasoning from final logits. For each image it runs the two prefills G-S10j compared, on IDENTICAL inputs, each with its batched hidden-state capture armed on EVERY layer:
//   - "cpu": the off arm's CPU prefill with the DeepStack sets (KVCache.SetCaptureLayersForTest), the residual after each layer and its DeepStack add;
//   - "cuda": the on arm's resident DeepStack prefill (cudaResident.SetBatchedCapture), the same convention.
//
// and writes the ids, the m-RoPE positions, the merged image rows, the DeepStack sets and both arms' [layers][rows*hidden] residuals for scripts/s896_hf_layers.py to difference against
// Hugging Face's float32 text model fed the SAME embeddings, positions and sets. Options and tower are G-S10j round 3's (serve's loadflags defaults, the CUDA tower). Nothing is graded here.
// GOINFER_S896_DIR (default ~/goinfer-logs/s896), GOINFER_S896_IMAGES (comma list under testdata/, default the 896 square and the 4x6 control), GOINFER_S896_KNOBS (NAME=VALUE,...).
func TestS896_layerDump(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy: set GOINFER_HEAVY_TESTS=1")
	}
	prevDeep := cudaDeepstackPrefillOn
	cudaDeepstackPrefillOn = true
	defer func() { cudaDeepstackPrefillOn = prevDeep }()
	requireCUDADevice(t)
	home, _ := os.UserHomeDir()
	dir := filepath.Join(home, "models", "qwen3-vl-2b-instruct")
	if _, err := os.Stat(dir); err != nil {
		t.Skipf("no %s", dir)
	}
	out := os.Getenv("GOINFER_S896_DIR")
	if out == "" {
		out = filepath.Join(home, "goinfer-logs", "s896")
	}
	if err := os.MkdirAll(out, 0o755); err != nil {
		t.Fatal(err)
	}
	images := []string{"gemma3_preprocess_image.png", "qwen25vl_preprocess_image.png"}
	if v := os.Getenv("GOINFER_S896_IMAGES"); v != "" {
		images = strings.Split(v, ",")
	}
	const merge = 2
	prompt := "What does this image show? Answer briefly." // the served request's text (G-S10j)
	tk, err := tokenizer.Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	imgTok, ok := tk.TokenID("<|image_pad|>")
	if !ok {
		t.Fatal("no <|image_pad|> token")
	}
	enc, err := vision.LoadQwen3VisionEncoder(dir, false)
	if err != nil {
		t.Fatal(err)
	}
	pp, err := multimodal.LoadQwen3PreprocessConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	if limit := 1024 * pp.MergeSize * pp.MergeSize * pp.PatchSize * pp.PatchSize; pp.MaxPixels > limit { // serve's cap
		pp.MaxPixels = limit
	}
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	lf := loadflags.Register(fs, loadflags.Serve)
	if err := fs.Parse([]string{"--backend", "cuda"}); err != nil {
		t.Fatal(err)
	}
	if err := lf.Validate(); err != nil {
		t.Fatal(err)
	}
	opts := lf.Options()
	opts.ResidentContext = 2048
	if kv := os.Getenv("GOINFER_S896_KNOBS"); kv != "" {
		knobs := decoder.Knobs{}
		if opts.Knobs != nil {
			for k, v := range *opts.Knobs {
				knobs[k] = v
			}
		}
		for _, p := range strings.Split(kv, ",") {
			k, v, _ := strings.Cut(p, "=")
			knobs[strings.TrimSpace(k)] = strings.TrimSpace(v)
		}
		opts.Knobs = &knobs
	}
	t.Logf("served options: quant %q embedInt4 %v kv %q/%q knobs %v", opts.Quant, opts.EmbedInt4, opts.KVPrecision, opts.KVQuant, opts.Knobs)
	m, err := decoder.Load(dir, opts)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	r, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok || !r.mropePrefillReady {
		t.Fatalf("not CUDA-resident with the m-RoPE batched kernel: %s", m.ResidentDecline())
	}
	acc, err := newQwen3Tower(enc)
	if err != nil {
		t.Fatalf("CUDA tower: %v", err)
	}
	t.Cleanup(func() { _ = acc.Close() })
	layers := make([]int, r.nLayers)
	for i := range layers {
		layers[i] = i
	}
	ctx := context.Background()
	for _, name := range images {
		tag := filepath.Base(name)
		data, err := os.ReadFile(filepath.Join("../testdata", name))
		if err != nil {
			t.Fatal(err)
		}
		px, grid, err := multimodal.QwenPreprocess(data, pp)
		if err != nil {
			t.Fatal(err)
		}
		feats, deep, err := multimodal.Qwen3TowerFeaturesDeepstack(enc, acc, px, [][3]int{grid})
		if err != nil {
			t.Fatal(err)
		}
		nImg := grid[0] * grid[1] * grid[2] / (merge * merge)
		pre, _ := tk.Encode("<|im_start|>user\n<|vision_start|>", false)
		post, _ := tk.Encode("<|vision_end|>"+prompt+"<|im_end|>\n<|im_start|>assistant\n", false)
		ids := append(append([]int{}, pre...), make([]int, nImg)...)
		for k := range nImg {
			ids[len(pre)+k] = imgTok
		}
		ids = append(ids, post...)
		n, start := len(ids), len(pre)
		mrope, err := decoder.MRopePositionsForTest(ids, imgTok, [][3]int{grid}, merge)
		if err != nil {
			t.Fatal(err)
		}
		fmt.Fprintf(os.Stderr, "[s896] %s: %d rows (image rows %d..%d), %d DeepStack sets; CPU prefill with capture\n", tag, n, start, start+nImg, len(deep))
		cache := m.NewCache(n + 2)
		cache.SetDeepstackForTest(start, nImg, deep)
		cache.SetCaptureLayersForTest(layers)
		if _, err := m.PrefillLogitsQwenVLForTest(ctx, ids, feats, start, nImg, mrope, cache); err != nil {
			t.Fatal(err)
		}
		cpu := cache.CapturedForTest()
		cache.SetDeepstackForTest(0, 0, nil)
		cache.SetCaptureLayersForTest(nil)
		fmt.Fprintf(os.Stderr, "[s896] %s: resident DeepStack prefill with capture\n", tag)
		r.Reset()
		if err := r.SetBatchedCapture(layers); err != nil {
			t.Fatal(err)
		}
		if _, _, err := m.ResidentMRoPEDeepstackPrefillForTest(ctx, r, ids, feats, start, nImg, mrope, deep); err != nil {
			t.Fatalf("%s: resident DeepStack prefill: %v", tag, err)
		}
		cuda := r.BatchedCapture()
		_ = r.SetBatchedCapture(nil)
		if len(cpu) != len(layers) || len(cuda) != len(layers) {
			t.Fatalf("captured %d CPU and %d CUDA layers, want %d", len(cpu), len(cuda), len(layers))
		}
		for l := range layers {
			if len(cpu[l]) != n*r.hidden || len(cuda[l]) != n*r.hidden {
				t.Fatalf("layer %d: %d CPU and %d CUDA floats, want %d", l, len(cpu[l]), len(cuda[l]), n*r.hidden)
			}
		}
		flat := func(x [][]float32) []float32 {
			var o []float32
			for _, v := range x {
				o = append(o, v...)
			}
			return o
		}
		meta, _ := json.Marshal(map[string]any{"image": name, "n": n, "start": start, "nImg": nImg, "grid": grid, "ids": ids, "mrope": mrope, "hidden": r.hidden, "layers": len(layers), "nDeep": len(deep), "prompt": prompt, "embedInt4": opts.EmbedInt4})
		for suffix, v := range map[string][]float32{"feats": feats, "deep": flat(deep), "cpu": flat(cpu), "cuda": flat(cuda)} {
			if err := writeF32(filepath.Join(out, tag+"."+suffix+".f32"), v); err != nil {
				t.Fatal(err)
			}
		}
		if err := os.WriteFile(filepath.Join(out, tag+".meta.json"), meta, 0o644); err != nil {
			t.Fatal(err)
		}
	}
}
