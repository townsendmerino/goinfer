//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/vision"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/loadflags"
	"github.com/townsendmerino/goinfer/multimodal"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestS896_gateDump is G-S10k's Go half (docs/tasks/task-multimodal-support-2026-10.md, "G-S10k"): 4 images x the 16 pinned prompts (scripts/s896_prompts.json), each
// unit run through serve's configuration (loadflags Options, the CUDA tower's features and sets, context 2048; G-S10j round 3's instrument) and dumped as float32 logits at 4 positions (the prefill's
// last row and 3 teacher-forced steps):
//   - "off": the CPU prefill with the sets, then the upload (the production path), at whatever CPU attention the run's knobs say (run 1: serve's fast kernel = OFF; run 2: GOINFER_CPU_FAST_ATTENTION=0 = OFFX);
//   - "on": the resident DeepStack prefill, teacher-forced along the teacher path;
//   - "onx_<name>" (GOINFER_S896G_PLANTED=late,notadded,textrows): ON with one of the planted DeepStack defects, the red controls the gate must see.
//
// The teacher path is run 1's OFF greedy continuation; run 2 reads it from GOINFER_S896G_TEACHER_DIR so every arm of a unit is forced along the same tokens. Nothing is graded here (scripts/s896_gate_grade.py).
// Env: GOINFER_S896G_DIR (output), GOINFER_S896G_PROMPTS (prompts per image, default all 16), GOINFER_S896G_KNOBS (NAME=VALUE,...), GOINFER_S896G_TEACHER_DIR, GOINFER_S896G_PLANTED (a comma list, below).
func TestS896_gateDump(t *testing.T) {
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
	out := os.Getenv("GOINFER_S896G_DIR")
	if out == "" {
		t.Skip("set GOINFER_S896G_DIR")
	}
	if err := os.MkdirAll(out, 0o755); err != nil {
		t.Fatal(err)
	}
	teacherDir := os.Getenv("GOINFER_S896G_TEACHER_DIR")
	// GOINFER_S896G_PLANTED: a comma list of the planted DeepStack defects to dump as extra ON arms (late = every set one layer late, notadded = no sets, textrows = the sets added to the text rows too).
	plantedList := map[string]int{"late": deepDefectOneLayerLate, "notadded": deepDefectNotAdded, "textrows": deepDefectTextRows}
	planted := map[string]int{}
	for _, nm := range strings.Split(os.Getenv("GOINFER_S896G_PLANTED"), ",") {
		if nm = strings.TrimSpace(nm); nm != "" {
			code, ok := plantedList[nm]
			if !ok {
				t.Fatalf("unknown planted defect %q", nm)
			}
			planted[nm] = code
		}
	}
	raw, err := os.ReadFile("../scripts/s896_prompts.json")
	if err != nil {
		t.Fatal(err)
	}
	var prompts []string
	if err := json.Unmarshal(raw, &prompts); err != nil {
		t.Fatal(err)
	}
	if v := os.Getenv("GOINFER_S896G_PROMPTS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 && n < len(prompts) {
			prompts = prompts[:n]
		}
	}
	const merge, steps = 2, 3
	tk, err := tokenizer.Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	imgTok, _ := tk.TokenID("<|image_pad|>")
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
	if kv := os.Getenv("GOINFER_S896G_KNOBS"); kv != "" {
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
	t.Logf("options: quant %q embedInt4 %v kv %q/%q knobs %v; %d prompts per image; planted %v; teacher dir %q", opts.Quant, opts.EmbedInt4, opts.KVPrecision, opts.KVQuant, opts.Knobs, len(prompts), planted, teacherDir)
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
	ctx := context.Background()
	T0 := time.Now()
	hb := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "[s896g %5.0fs] %s\n", time.Since(T0).Seconds(), fmt.Sprintf(format, a...))
	}
	images := []string{"gemma3_preprocess_image.png", "qwen25vl_preprocess_image.png", "glm_ocr/formula.png", "glm_ocr/table.png"}
	unit := 0
	for ii, name := range images {
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
		if err := writeF32(filepath.Join(out, tag+".px.f32"), px); err != nil {
			t.Fatal(err)
		}
		imeta, _ := json.Marshal(map[string]any{"image": name, "grid": grid, "nImg": nImg, "nDeep": len(deep)})
		if err := os.WriteFile(filepath.Join(out, tag+".img.json"), imeta, 0o644); err != nil {
			t.Fatal(err)
		}
		for pi, prompt := range prompts {
			utag := fmt.Sprintf("%s.p%02d", tag, pi)
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
			var teacher []int
			if teacherDir != "" {
				b, err := os.ReadFile(filepath.Join(teacherDir, utag+".meta.json"))
				if err != nil {
					t.Fatalf("teacher %s: %v", utag, err)
				}
				var tm struct{ Teacher []int }
				if err := json.Unmarshal(b, &tm); err != nil || len(tm.Teacher) < steps {
					t.Fatalf("teacher %s: %v (%d tokens)", utag, err, len(tm.Teacher))
				}
				teacher = tm.Teacher[:steps]
			}
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
			off, own := decode(refLast, teacher)
			if teacher == nil {
				teacher = own[:steps]
			}
			arm := func(defect int) [][]float32 {
				r.Reset()
				deepDefectForTest = defect
				defer func() { deepDefectForTest = 0 }()
				last, _, err := m.ResidentMRoPEDeepstackPrefillForTest(ctx, r, ids, feats, start, nImg, mrope, deep)
				if err != nil {
					t.Fatalf("%s: resident DeepStack prefill: %v", utag, err)
				}
				o, _ := decode(append([]float32(nil), last...), teacher)
				return o
			}
			on := arm(0)
			files := map[string][][]float32{"off": off, "on": on}
			for nm, code := range planted {
				files["onx_"+nm] = arm(code)
			}
			flat := func(x [][]float32) []float32 {
				var o []float32
				for _, v := range x {
					o = append(o, v...)
				}
				return o
			}
			for suffix, v := range files {
				if err := writeF32(filepath.Join(out, utag+"."+suffix+".f32"), flat(v)); err != nil {
					t.Fatal(err)
				}
			}
			worst := 1.0
			for k := range off {
				worst = math.Min(worst, dsCosine(off[k], on[k]))
			}
			meta, _ := json.Marshal(map[string]any{"image": name, "prompt": prompt, "promptIndex": pi, "n": n, "start": start, "nImg": nImg, "grid": grid, "ids": ids, "teacher": teacher, "vocab": len(off[0]), "steps": steps + 1, "embedInt4": opts.EmbedInt4})
			if err := os.WriteFile(filepath.Join(out, utag+".meta.json"), meta, 0o644); err != nil {
				t.Fatal(err)
			}
			unit++
			hb("image %d/%d %s, prompt %d/%d: %d rows, unit %d done, off-vs-on worst cosine %.4f", ii+1, len(images), tag, pi+1, len(prompts), n, unit, worst)
		}
	}
	hb("done: %d units written to %s", unit, out)
}
