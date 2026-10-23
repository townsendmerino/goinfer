//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
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
	m, err := decoder.Load(model, decoder.Options{Backend: "cuda", MoECacheExperts: true, ResidentContext: 16384, ResidentWindowedKV: true})
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
	softTok, _ := tk.TokenID(multimodal.Gemma4ImageSoftToken)
	tmpl := chat.Gemma4()
	ctx := context.Background()
	var tower multimodal.Gemma4TowerAccelerator // GOINFER_S19_TOWER=cuda: serve's default tower, else aikit's CPU tower
	if os.Getenv("GOINFER_S19_TOWER") == "cuda" {
		if tower, err = multimodal.NewGemma4Tower("cuda", enc); err != nil {
			t.Fatal(err)
		}
	}
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
		if kind == "cpu" {
			toks = runCPU(img)
			cpuRef[img] = toks
		} else {
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
