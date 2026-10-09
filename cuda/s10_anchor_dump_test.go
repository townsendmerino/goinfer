//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"math"
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

// writeF32 writes v as raw little-endian float32.
func writeF32(path string, v []float32) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return binary.Write(f, binary.LittleEndian, v)
}

// TestS10DeepstackPrefillCUDA_anchorDump is G-S10h's Go half (docs/tasks/task-multimodal-support-2026-10.md, "G-S10h", registered before this code): for each of the four F2a images on Qwen3-VL-2B it
// writes the token ids, the CPU encoder's pixels and grid, the teacher tokens (the off path's greedy, as the real gate teacher-forces) and the 9 logit vectors (the prefill's last row, then 8 teacher-forced
// decode steps) of BOTH arms: "off" (the CPU prefill with the sets, then the upload) and "on" (the resident DeepStack prefill), to $GOINFER_S10H_DIR (default ~/goinfer-logs/s10h). scripts/anchor_s10h_hf.py
// reads them and computes the Hugging Face float32 side. Nothing is graded here; the line it prints per image is instrument check (i): the off-against-on worst cosine, which must equal the real gate's.
func TestS10DeepstackPrefillCUDA_anchorDump(t *testing.T) {
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
	out := os.Getenv("GOINFER_S10H_DIR")
	if out == "" {
		out = filepath.Join(home, "goinfer-logs", "s10h")
	}
	if err := os.MkdirAll(out, 0o755); err != nil {
		t.Fatal(err)
	}
	const merge, steps = 2, 8
	prompt := os.Getenv("GOINFER_S10H_PROMPT") // G-S10i: the served request's text; the default is the gate's (G-S10h)
	if prompt == "" {
		prompt = "Describe this image."
	}
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
	embed4 := os.Getenv("GOINFER_S10H_EMBED4") == "1" // serve's --embed-int4 defaults to true with --quant int4; Options' zero value is false
	opts := decoder.Options{Quant: "int4", Backend: "cuda", ResidentContext: 2048, EmbedInt4: embed4}
	if os.Getenv("GOINFER_S10H_SERVED") == "1" { // G-S10j round 1: the options serve builds from its flag defaults (internal/loadflags), the context kept at 2048 (no numerics there)
		fs := flag.NewFlagSet("serve", flag.ContinueOnError)
		lf := loadflags.Register(fs, loadflags.Serve)
		if err := fs.Parse([]string{"--backend", "cuda"}); err != nil {
			t.Fatal(err)
		}
		if err := lf.Validate(); err != nil {
			t.Fatal(err)
		}
		opts = lf.Options()
		opts.ResidentContext = 2048
		embed4 = opts.EmbedInt4
		t.Logf("served options: quant %q embedInt4 %v kv %q/%q fit-disabled %v exactPrefill %v knobs %v", opts.Quant, opts.EmbedInt4, opts.KVPrecision, opts.KVQuant, opts.DisableFit, opts.ExactPrefill, opts.Knobs)
	}
	if kv := os.Getenv("GOINFER_S10H_KNOBS"); kv != "" { // the 896 investigation: operator knobs added to the arm's Options, "NAME=VALUE,NAME=VALUE" (e.g. GOINFER_PREFILL_CHUNK=2048)
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
		t.Logf("knobs: %v", knobs)
	}
	m, err := decoder.Load(dir, opts)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	r, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok || !r.mropePrefillReady {
		t.Fatalf("not CUDA-resident with the m-RoPE batched kernel: %s", m.ResidentDecline())
	}
	var acc *gridTower
	if os.Getenv("GOINFER_S10H_CUDATOWER") == "1" {
		if acc, err = newQwen3Tower(enc); err != nil {
			t.Fatalf("CUDA tower: %v", err)
		}
		t.Cleanup(func() { _ = acc.Close() })
	}
	ctx := context.Background()
	hb := func(format string, a ...any) { fmt.Fprintf(os.Stderr, "[G-S10h] "+format+"\n", a...) }
	for _, name := range []string{"gemma3_preprocess_image.png", "qwen25vl_preprocess_image.png", "glm_ocr/formula.png", "glm_ocr/table.png"} {
		tag := filepath.Base(name)
		data, err := os.ReadFile(filepath.Join("../testdata", name))
		if err != nil {
			t.Fatal(err)
		}
		px, grid, err := multimodal.QwenPreprocess(data, pp)
		if err != nil {
			t.Fatal(err)
		}
		hb("%s: tower features and sets (CPU encoder)", tag)
		var feats []float32
		var deep [][]float32
		if acc != nil { // G-S10j round 3: the features serve's -vision-device auto produces (the CUDA tower and aikit's host tails)
			feats, deep, err = multimodal.Qwen3TowerFeaturesDeepstack(enc, acc, px, [][3]int{grid})
		} else {
			feats, deep, err = enc.ForwardDeepstack(px, [][3]int{grid})
		}
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
		delta := mrope[n-1][0] + 1 - n
		hb("%s: %d rows; off path (CPU prefill with the sets, upload, CUDA decode)", tag, n)
		cache := m.NewCache(n + steps + 1)
		cache.SetDeepstackForTest(start, nImg, deep)
		refLast, err := m.PrefillLogitsQwenVLForTest(ctx, ids, feats, start, nImg, mrope, cache)
		cache.SetDeepstackForTest(0, 0, nil)
		if err != nil {
			t.Fatal(err)
		}
		refLast = append([]float32(nil), refLast...)
		r.Reset()
		if err := m.ResidentUploadPrefillForTest(cache); err != nil {
			t.Fatal(err)
		}
		decode := func(first []float32, forced []int) ([][]float32, []int) {
			o, toks := [][]float32{first}, []int{argmaxF(first)}
			for k := range steps {
				tok := toks[k]
				if forced != nil {
					tok = forced[k]
				}
				lg, err := r.ForwardMRoPE(m.EmbedResidentForTest(tok), n+k, n+k+delta)
				if err != nil {
					t.Fatal(err)
				}
				o = append(o, append([]float32(nil), lg...))
				toks = append(toks, argmaxF(lg))
			}
			return o, toks
		}
		off, teacher := decode(refLast, nil)
		hb("%s: on path (the resident DeepStack prefill)", tag)
		r.Reset()
		newLast, _, err := m.ResidentMRoPEDeepstackPrefillForTest(ctx, r, ids, feats, start, nImg, mrope, deep)
		if err != nil {
			t.Fatalf("%s: resident DeepStack prefill: %v", tag, err)
		}
		on, _ := decode(append([]float32(nil), newLast...), teacher)
		worst := 1.0
		flat := func(x [][]float32) []float32 {
			var o []float32
			for _, v := range x {
				o = append(o, v...)
			}
			return o
		}
		for k := range off {
			worst = math.Min(worst, dsCosine(off[k], on[k]))
		}
		hb("%s: INSTRUMENT CHECK (i) off-against-on worst cosine %.4f over %d rows", tag, worst, n)
		meta, _ := json.Marshal(map[string]any{"image": name, "n": n, "start": start, "nImg": nImg, "grid": grid, "ids": ids, "teacher": teacher[:steps], "vocab": len(off[0]), "steps": steps + 1, "prompt": prompt, "embedInt4": embed4, "cudaTower": acc != nil})
		for suffix, v := range map[string][]float32{"px": px, "off": flat(off), "on": flat(on)} {
			if err := writeF32(filepath.Join(out, tag+"."+suffix+".f32"), v); err != nil {
				t.Fatal(err)
			}
		}
		if err := os.WriteFile(filepath.Join(out, tag+".meta.json"), meta, 0o644); err != nil {
			t.Fatal(err)
		}
	}
}
