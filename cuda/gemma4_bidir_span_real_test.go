//go:build cuda && goinfer_testhooks

package cuda

import (
	"cmp"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/vision"
	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/modelload"
	"github.com/townsendmerino/goinfer/multimodal"
)

// TestGemma4BidirSpanPrefill_real26B is G-S19c's logits half (docs/tasks/task-multimodal-support-2026-10.md, "S19"): the
// real 26B-A4B loaded with -backend cuda -moe-cache-experts at the fit-default context, one image prompt per form-like image, the
// tower run once on the CPU, then the last-position logits from the resident span prefill against the CPU bidirectional
// prefill on the same model and the same features. It reports cosine and argmax; the served half grades the replies.
//
//	GOINFER_HEAVY_TESTS=1 ./cuda.test -test.run TestGemma4BidirSpanPrefill_real26B -test.v -test.timeout 60m
//
// GOINFER_S19_MODEL / GOINFER_S19_VISION override ~/models/gemma4-26b-int4.giw and ~/models/gemma-4-26b-a4b-it-vis560.
func TestGemma4BidirSpanPrefill_real26B(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("GOINFER_HEAVY_TESTS unset: real 26B")
	}
	home, _ := os.UserHomeDir()
	model := envOr("GOINFER_S19_MODEL", filepath.Join(home, "models", "gemma4-26b-int4.giw"))
	visDir := envOr("GOINFER_S19_VISION", filepath.Join(home, "models", "gemma-4-26b-a4b-it-vis560"))
	for _, p := range []string{model, visDir} {
		if _, err := os.Stat(p); err != nil {
			t.Skipf("no %s: %v", p, err)
		}
	}
	tk, err := modelload.Tokenizer(visDir)
	if err != nil {
		t.Fatal(err)
	}
	m, err := decoder.Load(model, decoder.Options{Backend: "cuda", MoECacheExperts: true})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	r := mustResident(t, m)
	enc, err := vision.LoadGemma4Encoder(visDir, false)
	if err != nil {
		t.Fatal(err)
	}
	pp, err := multimodal.LoadGemma4PreprocessConfig(visDir)
	if err != nil {
		t.Fatal(err)
	}
	softTok, ok := tk.TokenID(multimodal.Gemma4ImageSoftToken)
	if !ok {
		t.Fatal("no soft token")
	}
	tmpl := chat.Gemma4()
	ctx := context.Background()
	for _, img := range []string{"../testdata/glm_ocr/invoices/inv01.png", "../testdata/glm_ocr/invoices/inv02.png", "../testdata/glm_ocr/table.png"} {
		data, err := os.ReadFile(img)
		if err != nil {
			t.Fatal(err)
		}
		patches, pos, err := vision.Gemma4Preprocess(data, pp.MaxSoftTokens)
		if err != nil {
			t.Fatal(err)
		}
		n := multimodal.Gemma4PooledTokens(pos, enc.Cfg.PoolingKernelSize)
		t0 := time.Now()
		feats, err := multimodal.Gemma4TowerFeatures(enc, nil, patches, pos)
		if err != nil {
			t.Fatal(err)
		}
		tower := time.Since(t0)
		block := multimodal.Gemma4ImageBlock(n)
		turns := []chat.Turn{{Role: "user", Content: block + "Transcribe every field of this form as key: value lines."}}
		segs, err := multimodal.SpliceImageBlocks(tmpl.RenderSegments("", turns), []string{block})
		if err != nil {
			t.Fatal(err)
		}
		ids, err := tk.EncodeSegments(segs, false)
		if err != nil {
			t.Fatal(err)
		}
		runs := multimodal.FindImageRuns(ids, softTok)
		if len(runs) != 1 || runs[0][1] != n { // a run is [start, length]
			t.Fatalf("%s: image runs %v, want one of %d", img, runs, n)
		}
		spans := []decoder.ImageSpan{{Pos: runs[0][0], Len: n}}
		rows, err := m.Gemma4BidirImageRowsForTest(ids, feats, spans)
		if err != nil {
			t.Fatal(err)
		}
		t1 := time.Now()
		gpu, err := r.PrefillSpansLast(ctx, rows, 0, spanBlocks(spans))
		if err != nil {
			t.Fatalf("%s: PrefillSpansLast: %v", img, err)
		}
		gpu = append([]float32(nil), gpu...)
		tg := time.Since(t1)
		t2 := time.Now()
		cpu, err := m.PrefillLogitsGemma4VLBidirectionalSpansForTest(ctx, ids, feats, spans, m.NewCache(len(ids)+1))
		if err != nil {
			t.Fatal(err)
		}
		tc := time.Since(t2)
		c, _ := cosMaxAbs(gpu, cpu)
		ag, ac := argmaxF(gpu), argmaxF(cpu)
		t.Logf("%s: %d rows (span [%d,%d)), tower %.1fs | resident span prefill %.2fs, CPU prefill %.1fs | last-position cosine %.6f, argmax gpu=%d %q cpu=%d %q",
			filepath.Base(img), len(ids), spans[0].Pos, spans[0].Pos+n, tower.Seconds(), tg.Seconds(), tc.Seconds(), c, ag, dec(tk, ag), ac, dec(tk, ac))
		if ag != ac {
			t.Errorf("%s: argmax differs (gpu %d, cpu %d)", img, ag, ac)
		}
	}
}

func envOr(k, d string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return d
}

// dec is one token's text for a log line.
func dec(tk interface {
	Decode([]int) (string, error)
}, id int) string {
	s, err := tk.Decode([]int{id})
	if err != nil {
		return fmt.Sprint(err)
	}
	return strings.ReplaceAll(s, "\n", `\n`)
}

// TestGemma4BidirSpanPrefill_real26BDecode is a diagnostic for G-S19c's served reading: on the real 26B, each image's prompt
// prefilled on the resident's span path and on the CPU (then uploaded), each followed by greedy resident decode, in a fixed
// order of turns, with the first image repeated last, so a difference that depends on the turn's place in the order and one
// that depends on the image can be told apart.
func TestGemma4BidirSpanPrefill_real26BDecode(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("GOINFER_HEAVY_TESTS unset: real 26B")
	}
	home, _ := os.UserHomeDir()
	model := envOr("GOINFER_S19_MODEL", filepath.Join(home, "models", "gemma4-26b-int4.giw"))
	visDir := envOr("GOINFER_S19_VISION", filepath.Join(home, "models", "gemma-4-26b-a4b-it-vis560"))
	tk, err := modelload.Tokenizer(visDir)
	if err != nil {
		t.Skip(err)
	}
	enc, err := vision.LoadGemma4Encoder(visDir, false)
	if err != nil {
		t.Fatal(err)
	}
	pp, err := multimodal.LoadGemma4PreprocessConfig(visDir)
	if err != nil {
		t.Fatal(err)
	}
	softTok, _ := tk.TokenID(multimodal.Gemma4ImageSoftToken)
	tmpl := chat.Gemma4()
	ctx := context.Background()
	// GOINFER_S19_TOWER=cuda: serve's default tower, built before the model as serve's held-back VRAM lets it be; else aikit's
	// CPU tower.
	var tower multimodal.Gemma4TowerAccelerator
	if os.Getenv("GOINFER_S19_TOWER") == "cuda" {
		if tower, err = multimodal.NewGemma4Tower("cuda", enc); err != nil {
			t.Fatal(err)
		}
	}
	m, err := decoder.Load(model, decoder.Options{Backend: "cuda", MoECacheExperts: true, ResidentContext: 16384, ResidentWindowedKV: true})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	r := mustResident(t, m)
	steps := 100
	if v := os.Getenv("GOINFER_S19_STEPS"); v != "" {
		fmt.Sscan(v, &steps)
	}
	type prompt struct {
		ids   []int
		feats []float32
		spans []decoder.ImageSpan
	}
	prompts := map[string]prompt{}
	for _, img := range []string{"inv01", "inv02", "table"} {
		path := "../testdata/glm_ocr/invoices/" + img + ".png"
		if img == "table" {
			path = "../testdata/glm_ocr/table.png"
		}
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		patches, pos, err := vision.Gemma4Preprocess(data, pp.MaxSoftTokens)
		if err != nil {
			t.Fatal(err)
		}
		n := multimodal.Gemma4PooledTokens(pos, enc.Cfg.PoolingKernelSize)
		feats, err := multimodal.Gemma4TowerFeatures(enc, tower, patches, pos)
		if err != nil {
			t.Fatal(err)
		}
		block := multimodal.Gemma4ImageBlock(n)
		turns := []chat.Turn{{Role: "user", Content: block + "Transcribe every field of this form as key: value lines."}}
		segs, _ := multimodal.SpliceImageBlocks(tmpl.RenderSegments("", turns), []string{block})
		ids, _ := tk.EncodeSegments(segs, false)
		runs := multimodal.FindImageRuns(ids, softTok)
		prompts[img] = prompt{ids, feats, []decoder.ImageSpan{{Pos: runs[0][0], Len: runs[0][1]}}}
	}
	decode := func(logits []float32, at int) []int {
		var out []int
		for i := 0; i < steps; i++ {
			next := argmaxF(logits)
			out = append(out, next)
			var err error
			if logits, err = r.Forward(m.EmbedResidentForTest(next), at+i); err != nil {
				t.Fatal(err)
			}
		}
		return out
	}
	runSpan := func(img string) []int {
		p := prompts[img]
		rows, err := m.Gemma4BidirImageRowsForTest(p.ids, p.feats, p.spans)
		if err != nil {
			t.Fatal(err)
		}
		lg, err := r.PrefillSpansLast(ctx, rows, 0, spanBlocks(p.spans))
		if err != nil {
			t.Fatal(err)
		}
		return decode(append([]float32(nil), lg...), len(p.ids))
	}
	cpuRef := map[string][]int{}
	runCPU := func(img string) []int {
		p := prompts[img]
		c := m.NewCache(len(p.ids) + steps + 1)
		lg, err := m.PrefillLogitsGemma4VLBidirectionalSpansForTest(ctx, p.ids, p.feats, p.spans, c)
		if err != nil {
			t.Fatal(err)
		}
		if err := m.ResidentUploadPrefillForTest(c); err != nil {
			t.Fatal(err)
		}
		return decode(lg, len(p.ids))
	}
	firstDiff := func(a, b []int) int {
		for i := range min(len(a), len(b)) {
			if a[i] != b[i] {
				return i
			}
		}
		return -1
	}
	show := func(ids []int) string {
		s, _ := tk.Decode(ids)
		return strings.ReplaceAll(s, "\n", `\n`)
	}
	order := strings.Fields(envOr("GOINFER_S19_ORDER", "span:inv01 span:inv02 span:table cpu:inv01 cpu:inv02 cpu:table span:table span:inv01"))
	got := map[string][][]int{}
	for i, step := range order {
		kind, img, _ := strings.Cut(step, ":")
		var toks []int
		switch kind {
		case "cpu":
			toks = runCPU(img)
			cpuRef[img] = toks
		case "cpuonly": // CPU prefill and CPU decode: no upload, no resident
			p := prompts[img]
			c := m.NewCache(len(p.ids) + steps + 1)
			lg, err := m.PrefillLogitsGemma4VLBidirectionalSpansForTest(ctx, p.ids, p.feats, p.spans, c)
			if err != nil {
				t.Fatal(err)
			}
			for range steps {
				next := argmaxF(lg)
				toks = append(toks, next)
				if lg, err = m.ForwardForTest(next, c); err != nil {
					t.Fatal(err)
				}
			}
		default:
			toks = runSpan(img)
		}
		got[step] = append(got[step], toks)
		t.Logf("turn %d %s: %q", i+1, step, show(toks))
	}
	for _, img := range []string{"inv01", "inv02", "table"} {
		ref := cpuRef[img]
		for k, toks := range got["span:"+img] {
			t.Logf("%s span run %d vs CPU+upload: first difference at decode step %d", img, k+1, firstDiff(toks, ref))
		}
	}
}

// TestGemma4_26BDecodeLocalize localizes the CUDA resident decode defect S19 found on the real 26B: the table form's prompt
// prefilled on the CPU and uploaded, then the CPU's own greedy tokens teacher-forced through the resident decode and the CPU
// decode side by side. Every step logs the logit cosine and both argmaxes. Over GOINFER_S19_LOC_FROM..TO the residual after
// every layer is captured on both sides; at the first step where the argmaxes differ, the per-layer cosine names the first
// layer that departs.
func TestGemma4_26BDecodeLocalize(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("GOINFER_HEAVY_TESTS unset: real 26B")
	}
	home, _ := os.UserHomeDir()
	model := envOr("GOINFER_S19_MODEL", filepath.Join(home, "models", "gemma4-26b-int4.giw"))
	visDir := envOr("GOINFER_S19_VISION", filepath.Join(home, "models", "gemma-4-26b-a4b-it-vis560"))
	img := envOr("GOINFER_S19_IMAGE", "../testdata/glm_ocr/table.png")
	steps, from, to := 120, 95, 110
	fmt.Sscan(envOr("GOINFER_S19_STEPS", "120"), &steps)
	fmt.Sscan(envOr("GOINFER_S19_LOC_FROM", "95"), &from)
	fmt.Sscan(envOr("GOINFER_S19_LOC_TO", "110"), &to)
	tk, err := modelload.Tokenizer(visDir)
	if err != nil {
		t.Skip(err)
	}
	enc, err := vision.LoadGemma4Encoder(visDir, false)
	if err != nil {
		t.Fatal(err)
	}
	pp, err := multimodal.LoadGemma4PreprocessConfig(visDir)
	if err != nil {
		t.Fatal(err)
	}
	m, err := decoder.Load(model, decoder.Options{Backend: "cuda", MoECacheExperts: true, ResidentContext: 16384, ResidentWindowedKV: true})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	r := mustResident(t, m)
	data, err := os.ReadFile(img)
	if err != nil {
		t.Fatal(err)
	}
	patches, pos, err := vision.Gemma4Preprocess(data, pp.MaxSoftTokens)
	if err != nil {
		t.Fatal(err)
	}
	n := multimodal.Gemma4PooledTokens(pos, enc.Cfg.PoolingKernelSize)
	feats, err := multimodal.Gemma4TowerFeatures(enc, nil, patches, pos)
	if err != nil {
		t.Fatal(err)
	}
	softTok, _ := tk.TokenID(multimodal.Gemma4ImageSoftToken)
	block := multimodal.Gemma4ImageBlock(n)
	segs, _ := multimodal.SpliceImageBlocks(chat.Gemma4().RenderSegments("", []chat.Turn{{Role: "user", Content: block + "Transcribe every field of this form as key: value lines."}}), []string{block})
	ids, _ := tk.EncodeSegments(segs, false)
	runs := multimodal.FindImageRuns(ids, softTok)
	spans := []decoder.ImageSpan{{Pos: runs[0][0], Len: runs[0][1]}}
	c := m.NewCache(len(ids) + steps + 1)
	lc, err := m.PrefillLogitsGemma4VLBidirectionalSpansForTest(context.Background(), ids, feats, spans, c)
	if err != nil {
		t.Fatal(err)
	}
	if err := m.ResidentUploadPrefillForTest(c); err != nil {
		t.Fatal(err)
	}
	lg := append([]float32(nil), lc...)
	firstSplit := -1
	for i := 0; i < steps; i++ {
		cos, _ := cosMaxAbs(lg, lc)
		ag, ac := argmaxF(lg), argmaxF(lc)
		t.Logf("step %3d pos %d: logits cos %.6f argmax gpu %d %q cpu %d %q", i, len(ids)+i, cos, ag, dec(tk, ag), ac, dec(tk, ac))
		if ag != ac && firstSplit < 0 {
			firstSplit = i
		}
		tok := ac // the CPU's clean trajectory, fed to both
		capture := i >= from && i <= to
		if capture {
			r.layerCap, r.layerCapBuf = true, nil
			decoder.SetGemma4HiddenCaptureForTest(true)
		}
		route := r.g4cap && capture // GOINFER_G4_CAPTURE=1: per-layer routing on both sides
		if route {
			r.g4capIdx, r.g4capLayer = nil, nil
			decoder.SetRouterCaptureForTest(true)
		}
		g, err := r.Forward(m.EmbedResidentForTest(tok), len(ids)+i)
		if err != nil {
			t.Fatal(err)
		}
		lg = append([]float32(nil), g...)
		if lc, err = m.ForwardForTest(tok, c); err != nil {
			t.Fatal(err)
		}
		if route {
			cIdx, cRn := decoder.RouterCaptureForTest()
			cMargin := decoder.RouterMarginForTest()
			_, cX1, cX2 := decoder.Gemma4MoECaptureForTest()
			decoder.SetRouterCaptureForTest(false)
			line := fmt.Sprintf("step %3d routing per MoE layer (same experts / CPU top-k margin / rn x1 x2 cos):", i)
			for k := range min(len(r.g4capIdx), len(cIdx)) {
				l := r.g4capLayer[k]
				gs := map[int]bool{}
				for _, e := range r.g4capIdx[k] {
					gs[int(e)] = true
				}
				same := 0
				for _, e := range cIdx[k] {
					if gs[e] {
						same++
					}
				}
				crn, _ := cosMaxAbs(r.g4capRn[l], cRn[k])
				cx1, _ := cosMaxAbs(r.g4capX1[l], cX1[k])
				cx2, _ := cosMaxAbs(r.g4capX2[l], cX2[k])
				mg := float32(0)
				if k < len(cMargin) {
					mg = cMargin[k]
				}
				line += fmt.Sprintf("\n    L%d %d/%d %.4f %.5f %.5f %.5f", l, same, len(cIdx[k]), mg, crn, cx1, cx2)
			}
			t.Log(line)
		}
		if capture {
			gl, cl := r.layerCapBuf, decoder.Gemma4HiddenCaptureForTest()
			r.layerCap, r.layerCapBuf = false, nil
			decoder.SetGemma4HiddenCaptureForTest(false)
			line := fmt.Sprintf("step %3d (input %q at pos %d) per-layer residual cos gpu/cpu:", i, dec(tk, tok), len(ids)+i)
			for l := range min(len(gl), len(cl)-1) {
				cl2, _ := cosMaxAbs(gl[l], cl[l+1])
				line += fmt.Sprintf(" L%d %.5f", l, cl2)
			}
			t.Log(line)
			if len(gl) == r.nLayers {
				// The GPU's last residual through the CPU's final norm and head, beside both sides' own logits.
				hc := m.LogitsFromHiddenForTest(gl[len(gl)-1])
				t.Logf("  next-token top 5: gpu %s | cpu %s | gpu residual + cpu head %s", top5(tk, lg), top5(tk, lc), top5(tk, hc))
				for _, id := range []int{argmaxF(lg), argmaxF(lc)} {
					t.Logf("  token %d %q: gpu %.4f cpu %.4f gpu-residual+cpu-head %.4f", id, dec(tk, id), lg[id], lc[id], hc[id])
				}
			}
		}
	}
	t.Logf("first step where the resident's argmax departs from the CPU's: %d", firstSplit)
}

// top5 renders the five largest logits as id "text" value.
func top5(tk interface {
	Decode([]int) (string, error)
}, v []float32) string {
	idx := make([]int, len(v))
	for i := range idx {
		idx[i] = i
	}
	slices.SortFunc(idx, func(a, b int) int { return cmp.Compare(v[b], v[a]) })
	out := ""
	for _, i := range idx[:5] {
		out += fmt.Sprintf("%d %q %.3f; ", i, dec(tk, i), v[i])
	}
	return out
}

// TestGemma4_26BActQuantSensitivity asks, on the CPU alone, whether per-row int8 activation scales (or, with
// GOINFER_S19_QUANT, int4 weights) are what turns a tiny difference into the 26B's garbage digits. For GOINFER_S19_ACTGROUP (0, per row, the default; or 32), the table form is
// prefilled and greedily decoded twice: once on its features as computed and once on features nudged by 1e-5 relative noise
// (the size of the CUDA-vs-CPU tower difference). The second run is teacher-forced on the first run's tokens, and each step
// logs the logits cosine, both argmaxes and, over GOINFER_S19_LOC_FROM..TO, the per-layer residual cosine between the two.
func TestGemma4_26BActQuantSensitivity(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("GOINFER_HEAVY_TESTS unset: real 26B")
	}
	home, _ := os.UserHomeDir()
	model := envOr("GOINFER_S19_MODEL", filepath.Join(home, "models", "gemma4-26b-int4.giw"))
	visDir := envOr("GOINFER_S19_VISION", filepath.Join(home, "models", "gemma-4-26b-a4b-it-vis560"))
	img := envOr("GOINFER_S19_IMAGE", "../testdata/glm_ocr/table.png")
	var group, steps, from, to int
	fmt.Sscan(envOr("GOINFER_S19_ACTGROUP", "0"), &group)
	fmt.Sscan(envOr("GOINFER_S19_STEPS", "120"), &steps)
	fmt.Sscan(envOr("GOINFER_S19_LOC_FROM", "95"), &from)
	fmt.Sscan(envOr("GOINFER_S19_LOC_TO", "110"), &to)
	tk, err := modelload.Tokenizer(visDir)
	if err != nil {
		t.Skip(err)
	}
	enc, err := vision.LoadGemma4Encoder(visDir, false)
	if err != nil {
		t.Fatal(err)
	}
	pp, err := multimodal.LoadGemma4PreprocessConfig(visDir)
	if err != nil {
		t.Fatal(err)
	}
	// GOINFER_S19_QUANT loads another weight precision (int8: weight-only, f32 activations), from a safetensors directory.
	m, err := decoder.Load(model, decoder.Options{ActQuantGroup: group, Quant: os.Getenv("GOINFER_S19_QUANT")})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	data, err := os.ReadFile(img)
	if err != nil {
		t.Fatal(err)
	}
	patches, pos, err := vision.Gemma4Preprocess(data, pp.MaxSoftTokens)
	if err != nil {
		t.Fatal(err)
	}
	n := multimodal.Gemma4PooledTokens(pos, enc.Cfg.PoolingKernelSize)
	feats, err := multimodal.Gemma4TowerFeatures(enc, nil, patches, pos)
	if err != nil {
		t.Fatal(err)
	}
	nudged := append([]float32(nil), feats...)
	var s uint32 = 12345
	for i := range nudged {
		s = s*1664525 + 1013904223
		nudged[i] *= 1 + 1e-5*(float32(int32(s>>16)%2001-1000)/1000)
	}
	softTok, _ := tk.TokenID(multimodal.Gemma4ImageSoftToken)
	block := multimodal.Gemma4ImageBlock(n)
	segs, _ := multimodal.SpliceImageBlocks(chat.Gemma4().RenderSegments("", []chat.Turn{{Role: "user", Content: block + "Transcribe every field of this form as key: value lines."}}), []string{block})
	ids, _ := tk.EncodeSegments(segs, false)
	runs := multimodal.FindImageRuns(ids, softTok)
	spans := []decoder.ImageSpan{{Pos: runs[0][0], Len: runs[0][1]}}
	ctx := context.Background()
	ca, cb := m.NewCache(len(ids)+steps+1), m.NewCache(len(ids)+steps+1)
	la, err := m.PrefillLogitsGemma4VLBidirectionalSpansForTest(ctx, ids, feats, spans, ca)
	if err != nil {
		t.Fatal(err)
	}
	la = append([]float32(nil), la...)
	lb, err := m.PrefillLogitsGemma4VLBidirectionalSpansForTest(ctx, ids, nudged, spans, cb)
	if err != nil {
		t.Fatal(err)
	}
	lb = append([]float32(nil), lb...)
	var toks []int
	split := -1
	for i := 0; i < steps; i++ {
		c, _ := cosMaxAbs(la, lb)
		aa, ab := argmaxF(la), argmaxF(lb)
		if aa != ab && split < 0 {
			split = i
		}
		t.Logf("group %d step %3d: logits cos %.6f argmax base %d %q nudged %d %q", group, i, c, aa, dec(tk, aa), ab, dec(tk, ab))
		tok := aa
		toks = append(toks, tok)
		capture := i >= from && i <= to
		var ha [][]float32
		if capture {
			decoder.SetGemma4HiddenCaptureForTest(true)
		}
		if la, err = m.ForwardForTest(tok, ca); err != nil {
			t.Fatal(err)
		}
		la = append([]float32(nil), la...)
		if capture {
			ha = append([][]float32(nil), decoder.Gemma4HiddenCaptureForTest()...)
			decoder.SetGemma4HiddenCaptureForTest(true)
		}
		if lb, err = m.ForwardForTest(tok, cb); err != nil {
			t.Fatal(err)
		}
		lb = append([]float32(nil), lb...)
		if capture {
			hb := decoder.Gemma4HiddenCaptureForTest()
			decoder.SetGemma4HiddenCaptureForTest(false)
			worst, at := 1.0, -1
			for l := 1; l < min(len(ha), len(hb)); l++ {
				if c, _ := cosMaxAbs(ha[l], hb[l]); c < worst {
					worst, at = c, l-1
				}
			}
			t.Logf("group %d step %3d: worst per-layer residual cos base/nudged %.5f at L%d", group, i, worst, at)
		}
	}
	text, _ := tk.Decode(toks)
	t.Logf("group %d: base reply %q", group, strings.ReplaceAll(text, "\n", `\n`))
	t.Logf("group %d: first step where the nudged run's argmax departs: %d", group, split)
}
