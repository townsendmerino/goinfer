//go:build realckpt

package decoder

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/vision"
	"github.com/townsendmerino/goinfer/multimodal"
)

// GLM-OCR end to end on the REAL checkpoint (gate O3, docs/tasks/task-glm-ocr-2026-10.md). Goinfer at f32 (Options{}, no
// Quant) reads three PROCEDURALLY RENDERED documents (scripts/gen_glm_ocr_doc_images.py: an invoice, a ruled table, a page
// with an equation; not real scans) through its OWN preprocessing, tower and decoder (GenerateQwenVL, the entry point serve
// uses), and must produce the SAME 64 greedy tokens as transformers 5.12.0 f32 CPU eager (scripts/pin_glm_ocr_e2e.py,
// golden_<name>.json.gz). The input ids fed in are HF's; that serve's own prompt construction reproduces them is the
// serveapp gate (internal/serveapp/glm_ocr_vision_real_test.go).
//
//	GOINFER_HEAVY_TESTS=1 go test -tags realckpt ./decoder/ -run 'TestGlmOcrE2E_f32/invoice' -v -timeout 20m 2>&1 | tee /tmp/glm-ocr-e2e-invoice.log
//
// One image takes about two to three minutes (the f32 tower is ~60 s, the 64-token decode ~15 s, and the prefill runs
// twice: once for the logits, once inside GenerateQwenVL). A heartbeat prints every 30 s. Run an image per invocation.
//
// Stages are separated so a failure names its stage: grid, then pixel_values against HF's own (reported, not gated: the
// resize is PIL-bicubic-tolerance-matched, not bit-exact), then the last-prompt-token logits (cosine, max|diff|, argmax),
// then the tokens. A divergence is reported with its step and both logit gaps and classified by the repo's 3% near-tie rule
// (the same 3% as decoder.NearTieHardFailPct, which sits behind the goinfer_testhooks tag, in the units of the logit range); it FAILS the test either way, because "token-identical"
// is the gate and a near-tie is a finding to be written up, not a pass.

const (
	glmOcrE2EDocs  = "../testdata/glm_ocr"
	glmOcrNearTie  = 0.03   // the tree's 3% near-tie rule, in units of the reference's logit range (fidelity_testhook.go)
	glmOcrE2EBarCS = 0.9999 // last-prompt-token logits cosine, f32 vs f32: the bar the other real-checkpoint f32 gates hold
)

type glmOcrGolden struct {
	Image       string    `json:"image"`
	ImageSHA256 string    `json:"image_sha256"`
	Prompt      string    `json:"prompt"`
	Grid        [][3]int  `json:"grid_thw"`
	NPatches    int       `json:"n_patches"`
	NImgTokens  int       `json:"n_image_tokens"`
	PVSHA256    string    `json:"pixel_values_sha256"`
	InputIDs    []int     `json:"input_ids"`
	ImageStart  int       `json:"image_token_start"`
	EOS         []int     `json:"eos_ids"`
	NNew        int       `json:"n_new"`
	HFTokens    []int     `json:"hf_tokens"`
	HFText      string    `json:"hf_text"`
	HFGaps      []float64 `json:"hf_gaps"`
	HFStopped   *int      `json:"hf_stopped_at"`
	LastLogits  string    `json:"last_logits_f32_b64"`
	Vocab       int       `json:"vocab"`
}

func (g *glmOcrGolden) lastLogits(t testing.TB) []float32 {
	raw, err := base64.StdEncoding.DecodeString(g.LastLogits)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float32, len(raw)/4)
	for i := range out {
		out[i] = math.Float32frombits(uint32(raw[4*i]) | uint32(raw[4*i+1])<<8 | uint32(raw[4*i+2])<<16 | uint32(raw[4*i+3])<<24)
	}
	if len(out) != g.Vocab {
		t.Fatalf("golden logits %d, vocab %d", len(out), g.Vocab)
	}
	return out
}

// refTokens is HF's continuation up to (not including) its stop token: generate() pads past EOS with the pad id, and
// goinfer's stream never emits a stop token.
func (g *glmOcrGolden) refTokens() []int {
	if g.HFStopped != nil {
		return g.HFTokens[:*g.HFStopped]
	}
	return g.HFTokens
}

var glmOcrShared struct {
	once sync.Once
	m    *Model
	enc  *vision.GlmOcrVisionEncoder
	pp   multimodal.QwenPreprocessConfig
	err  error
}

// glmOcrF32 loads, once per test process, the checkpoint at f32 (Options{}: no Quant) and the f32 tower.
func glmOcrF32(t *testing.T) (*Model, *vision.GlmOcrVisionEncoder, multimodal.QwenPreprocessConfig) {
	t.Helper()
	ckpt := assetPath(t, "GOINFER_GLM_OCR")
	glmOcrShared.once.Do(func() {
		t0 := time.Now()
		glmOcrShared.pp, glmOcrShared.err = multimodal.LoadGlmOcrPreprocessConfig(ckpt)
		if glmOcrShared.err != nil {
			return
		}
		glmOcrShared.enc, glmOcrShared.err = vision.LoadGlmOcrVisionEncoder(ckpt, false)
		if glmOcrShared.err != nil {
			return
		}
		glmOcrShared.m, glmOcrShared.err = Load(ckpt, Options{})
		fmt.Fprintf(os.Stderr, "[glm-ocr e2e] loaded f32 decoder + tower in %.0fs\n", time.Since(t0).Seconds())
	})
	if glmOcrShared.err != nil {
		t.Fatal(glmOcrShared.err)
	}
	m := glmOcrShared.m
	if a := m.w.arch; a.Name != "glm_ocr" || len(a.MRopeSection) != 3 {
		t.Fatalf("arch %q mrope %v: the wrong adapter ran", a.Name, a.MRopeSection)
	}
	if m.quant != "" {
		t.Fatalf("the gate is f32; model quant is %q", m.quant)
	}
	return m, glmOcrShared.enc, glmOcrShared.pp
}

// beat prints a progress line to stderr every 30 s until the returned stop is called (stderr, because under -v it streams as
// it happens where t.Logf is held until the test returns).
func beat(label string) (stop func()) {
	t0 := time.Now()
	done := make(chan struct{})
	go func() {
		tk := time.NewTicker(30 * time.Second)
		defer tk.Stop()
		for {
			select {
			case <-done:
				return
			case <-tk.C:
				fmt.Fprintf(os.Stderr, "[glm-ocr e2e] %s: still running, %.0fs\n", label, time.Since(t0).Seconds())
			}
		}
	}()
	return func() { close(done) }
}

func glmOcrReadGolden(t *testing.T, name string) (*glmOcrGolden, []byte) {
	t.Helper()
	var g glmOcrGolden
	raw, err := readGolden(filepath.Join(glmOcrE2EDocs, "golden_"+name+".json.gz")) // gunzips on a .gz suffix (CLAUDE.md)
	if err != nil {
		t.Fatalf("golden for %s: %v (scripts/pin_glm_ocr_e2e.py writes it)", name, err)
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	png, err := os.ReadFile(filepath.Join(glmOcrE2EDocs, name+".png"))
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(png)
	if hex.EncodeToString(sum[:]) != g.ImageSHA256 {
		t.Fatalf("%s.png sha256 %x != the golden's %s: the image was regenerated after the HF reference was taken", name, sum, g.ImageSHA256)
	}
	return &g, png
}

// glmOcrCapture records, at every decode step, the logit vector the sampler saw (a LogitProcessor that only observes).
type glmOcrCapture struct{ rows [][]float32 }

func (c *glmOcrCapture) process(_ []int, logits []float32) {
	c.rows = append(c.rows, append([]float32(nil), logits...))
}

func top2(l []float32) (a, b int) {
	a, b = 0, 1
	if l[b] > l[a] {
		a, b = b, a
	}
	for i := 2; i < len(l); i++ {
		switch {
		case l[i] > l[a]:
			a, b = i, a
		case l[i] > l[b]:
			b = i
		}
	}
	return
}

func logitRange(l []float32) float64 {
	lo, hi := l[0], l[0]
	for _, v := range l {
		lo, hi = min(lo, v), max(hi, v)
	}
	return float64(hi - lo)
}

func maxAbsDiff32(a, b []float32) (worst float64, at int) {
	for i := range a {
		if d := math.Abs(float64(a[i] - b[i])); d > worst {
			worst, at = d, i
		}
	}
	return
}

// glmOcrRun feeds pv through the f32 tower and decoder with the golden's input ids and applies every stage of the gate.
// tag labels the variant in the log ("goinfer-pixels", "hf-pixels"). It returns the number of failures it reported.
func glmOcrRun(t *testing.T, g *glmOcrGolden, pv []float32, grid [3]int, tag string) {
	t.Helper()
	m, enc, pp := glmOcrF32(t)
	ctx, cancel := context.WithTimeout(context.Background(), 12*time.Minute)
	defer cancel()
	hidden := m.w.arch.HiddenDim

	stop := beat(tag + " tower")
	t0 := time.Now()
	feats, err := enc.Forward(pv, [][3]int{grid})
	stop()
	if err != nil {
		t.Fatalf("tower: %v", err)
	}
	if len(feats) != g.NImgTokens*hidden {
		t.Fatalf("tower emitted %d floats, want %d image tokens x %d", len(feats), g.NImgTokens, hidden)
	}
	t.Logf("[%s] tower %.0fs for %d patches -> %d image tokens", tag, time.Since(t0).Seconds(), len(pv)/1176, g.NImgTokens)

	// Stage: positions are the caller's (mropePositions), with the model's own image id and merge.
	const imageToken = 59280
	mropePos, err := mropePositions(g.InputIDs, imageToken, [][3]int{grid}, pp.MergeSize)
	if err != nil {
		t.Fatal(err)
	}

	// Stage: the last-prompt-token logits, through the same prefill GenerateQwenVL runs.
	stop = beat(tag + " prefill")
	cache := m.NewCache(len(g.InputIDs) + g.NNew)
	logits, err := m.prefillLogitsQwenVL(ctx, g.InputIDs, feats, g.ImageStart, g.NImgTokens, mropePos, cache)
	stop()
	if err != nil {
		t.Fatalf("prefill: %v", err)
	}
	want := g.lastLogits(t)
	cs := logitCosine(logits, want)
	worst, at := maxAbsDiff32(logits, want)
	gotArg, wantArg := argmax(logits), argmax(want)
	t.Logf("[%s] last-prompt-token logits vs HF: cosine %.9f, max|diff| %.3g at id %d (HF logit %.4f, range %.2f), argmax goinfer %d HF %d",
		tag, cs, worst, at, want[at], logitRange(want), gotArg, wantArg)
	if cs < glmOcrE2EBarCS {
		t.Errorf("[%s] last-token logits cosine %.9f < %.4f", tag, cs, glmOcrE2EBarCS)
	}
	if gotArg != wantArg {
		t.Errorf("[%s] last-token argmax %d, HF %d", tag, gotArg, wantArg)
	}

	// Stage: tokens, through GenerateQwenVL (the serve entry point), capturing goinfer's logits at every step.
	cap := &glmOcrCapture{}
	stop = beat(tag + " generate")
	t0 = time.Now()
	stream, gen := m.GenerateQwenVL(ctx, g.InputIDs, g.ImageStart, g.NImgTokens, 0,
		func() ([]float32, error) { return feats, nil }, [][3]int{grid}, pp.MergeSize, imageToken, g.NNew,
		SamplingParams{Temperature: 0, LogitProcessor: cap.process})
	var got []int
	for id := range stream {
		got = append(got, id)
	}
	stop()
	if err := gen.Err(); err != nil {
		t.Fatalf("[%s] generate: %v", tag, err)
	}
	t.Logf("[%s] generate %.0fs (prefill + %d decode steps)", tag, time.Since(t0).Seconds(), len(got))
	ref := g.refTokens()
	div := -1
	for i := 0; i < max(len(got), len(ref)); i++ {
		if i >= len(got) || i >= len(ref) || got[i] != ref[i] {
			div = i
			break
		}
	}
	if div < 0 {
		t.Logf("[%s] TOKEN-IDENTICAL: %d/%d tokens equal HF's (HF stopped at %v)", tag, len(got), len(ref), g.HFStopped)
		return
	}
	// A divergence: report the step, both logit gaps, and the near-tie classification.
	line := fmt.Sprintf("[%s] FIRST DIVERGENCE at step %d of %d (%d/%d identical before it)", tag, div, len(ref), div, len(ref))
	if div < len(cap.rows) {
		row := cap.rows[div]
		a, b := top2(row)
		gapGo := float64(row[a] - row[b])
		line += fmt.Sprintf("\n   goinfer picked %d, HF picked %v; goinfer top1-top2 gap %.4f", got2(got, div), got2(ref, div), gapGo)
		if div < len(ref) {
			hfTok := ref[div]
			gapToHF := float64(row[a] - row[hfTok])
			rng := logitRange(row)
			line += fmt.Sprintf("; goinfer's own logit for HF's token trails its pick by %.4f = %.3f%% of its logit range %.2f (3%% rule: %s)",
				gapToHF, 100*gapToHF/rng, rng, map[bool]string{true: "INSIDE the near-tie band", false: "OUTSIDE it, a forward error is suspected"}[gapToHF/rng <= glmOcrNearTie])
		}
	}
	if div < len(g.HFGaps) {
		line += fmt.Sprintf("\n   HF's own top1-top2 gap at that step: %.4f", g.HFGaps[div])
	}
	line += fmt.Sprintf("\n   goinfer: %v\n   HF:      %v", got, ref)
	t.Errorf("%s", line)
}

func got2(s []int, i int) any {
	if i < len(s) {
		return s[i]
	}
	return "<end>"
}

func TestGlmOcrE2E_f32(t *testing.T) {
	requireHeavyModel(t)
	for _, name := range []string{"invoice", "table", "formula"} {
		t.Run(name, func(t *testing.T) {
			g, png := glmOcrReadGolden(t, name)
			_, _, pp := glmOcrF32(t)

			// Stage: goinfer's own preprocessing of the PNG; the grid must be the processor's.
			t0 := time.Now()
			pv, grid, err := multimodal.QwenPreprocess(png, pp)
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("preprocess %.1fs: grid %v (%d patches, %d image tokens)", time.Since(t0).Seconds(), grid, len(pv)/1176, multimodal.QwenMergedTokens(grid, pp.MergeSize))
			if grid != g.Grid[0] || len(pv) != g.NPatches*1176 {
				t.Fatalf("grid %v len %d, HF's processor gave %v with %d patches", grid, len(pv), g.Grid[0], g.NPatches)
			}
			// Stage: pixel_values against HF's own, when the dump is on disk (reported, not gated).
			if dir, err := lookupAsset("GOINFER_GLM_OCR_E2E"); err == nil {
				hf := readF32LE(t, filepath.Join(dir, name+".pv.f32"))
				logPixelDiff(t, pv, hf)
			} else {
				t.Logf("pixel_values vs HF not compared: %v", err)
			}
			glmOcrRun(t, g, pv, grid, "goinfer-pixels")
		})
	}
}

// TestGlmOcrE2E_f32_hfPixels is the same gate with HF's OWN pixel_values (dumped by pin_glm_ocr_e2e.py) fed to goinfer's
// tower: a divergence in TestGlmOcrE2E_f32 that vanishes here came from preprocessing (PIL-style bicubic vs torchvision's);
// one that stays here is the model path.
func TestGlmOcrE2E_f32_hfPixels(t *testing.T) {
	requireHeavyModel(t)
	dir := assetPath(t, "GOINFER_GLM_OCR_E2E")
	for _, name := range []string{"invoice", "table", "formula"} {
		t.Run(name, func(t *testing.T) {
			g, _ := glmOcrReadGolden(t, name)
			pv := readF32LE(t, filepath.Join(dir, name+".pv.f32"))
			if len(pv) != g.NPatches*1176 {
				t.Fatalf("hf pixel dump has %d floats, want %d", len(pv), g.NPatches*1176)
			}
			glmOcrRun(t, g, pv, g.Grid[0], "hf-pixels")
		})
	}
}

// logPixelDiff reports how far goinfer's pixel_values are from HF's. One 8-bit level is 1/(255*std) ~ 0.0146 in normalized
// units, so the numbers are also given in levels.
func logPixelDiff(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Errorf("pixel_values length %d, HF %d", len(got), len(want))
		return
	}
	const level = 1.0 / (255 * 0.27) // ~ one 8-bit level in normalized units (std ~ 0.27)
	var maxd, sum float64
	var differ, over1 int
	diffs := make([]float64, 0, len(got)/64)
	for i := range got {
		d := math.Abs(float64(got[i] - want[i]))
		if d > 0 {
			differ++
		}
		if d > level*1.01 {
			over1++
		}
		maxd = math.Max(maxd, d)
		sum += d
		if i%64 == 0 {
			diffs = append(diffs, d)
		}
	}
	sort.Float64s(diffs)
	t.Logf("pixel_values vs HF (%d floats): bit-identical %.4f%%, differ %d, > 1 level %d (%.4f%%), max|diff| %.4f (%.1f levels), mean|diff| %.2e, p99 (sampled) %.4f",
		len(got), 100*float64(len(got)-differ)/float64(len(got)), differ, over1, 100*float64(over1)/float64(len(got)), maxd, maxd/level, sum/float64(len(got)), diffs[len(diffs)*99/100])
}
