//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/vision"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/multimodal"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// G-S10g of docs/tasks/task-multimodal-support-2026-10.md ("S10 on CUDA", registered before this code), tiny half; CUDA twin of metal/s16_mrope_prefill_test.go's TestS16DeepstackPrefill_tiny. The Qwen3-VL tiny decoder
// takes a synthetic 16x24 image with random features and DeepStack sets: the resident DeepStack prefill (PrefillMRoPEDeepstackLast) against today's path (the CPU prefill with the sets, the KV uploaded, CUDA decode), on
// the last-row logits and 8 teacher-forced decode steps, at cosine >= 0.9999 per row with an equal argmax (an argmax difference counts only when it is not an R10 near-tie: this fixture's next-token distribution is
// nearly uniform). Three planted defects, each alone red: the sets not added, added one layer late, added to the text rows too. Then the decoder's own entry point, which must take the resident prefill.

func dsCosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return dot / math.Sqrt(na*nb)
}

func dsLogSoftmax(x []float32) []float64 {
	mx := float64(x[0])
	for _, v := range x {
		mx = math.Max(mx, float64(v))
	}
	var s float64
	for _, v := range x {
		s += math.Exp(float64(v) - mx)
	}
	lse := mx + math.Log(s)
	out := make([]float64, len(x))
	for i, v := range x {
		out[i] = float64(v) - lse
	}
	return out
}

// dsSpliceRows is the resident prefill's row building: the embedding lookup, the raw image features over the image block.
func dsSpliceRows(m *decoder.Model, ids []int, feats []float32, start, n, hid int) [][]float32 {
	rows := make([][]float32, len(ids))
	for i, id := range ids {
		if i >= start && i < start+n {
			rows[i] = feats[(i-start)*hid : (i-start+1)*hid]
		} else {
			rows[i] = m.EmbedResidentForTest(id)
		}
	}
	return rows
}

func TestS10DeepstackPrefillCUDA_tiny(t *testing.T) {
	prevDeep := cudaDeepstackPrefillOn // production default is ON since G-S10g passed; the gate sets it explicitly and puts it back
	cudaDeepstackPrefillOn = true
	defer func() { cudaDeepstackPrefillOn = prevDeep }()
	const merge, steps, bar, gh, gw = 2, 8, 0.9999, 8, 12
	requireCUDADevice(t) // before the fixture check: the fixture is gitignored (regenerated locally), so a deviceless CI runner has no fixture and must skip on the device, not fail on it
	path := filepath.Join("..", "testdata", "qwen3vl-tiny")
	if _, err := os.Stat(filepath.Join(path, "model.safetensors")); err != nil {
		t.Fatalf("no fixture at %s", path)
	}
	m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4", ResidentContext: 256})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	r, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok || !r.mropePrefillReady {
		t.Fatalf("not CUDA-resident with the m-RoPE batched kernel: %s", m.ResidentDecline())
	}
	const imgTok, visStart, hid = 299, 298, 64
	nb := gh * gw
	ids := []int{10, 11, 12, visStart}
	start := len(ids)
	for range nb {
		ids = append(ids, imgTok)
	}
	ids = append(ids, 13, 14, 15, 16, 17)
	n := len(ids)
	rng := rand.New(rand.NewSource(17))
	rnd := func(k int) []float32 {
		v := make([]float32, k)
		for i := range v {
			v[i] = float32(rng.NormFloat64())
		}
		return v
	}
	feats, sets := rnd(nb*hid), [][]float32{rnd(nb * hid), rnd(nb * hid)}
	mrope, err := decoder.MRopePositionsForTest(ids, imgTok, [][3]int{{1, 2 * gh, 2 * gw}}, merge)
	if err != nil {
		t.Fatal(err)
	}
	delta := mrope[n-1][0] + 1 - n
	ctx := context.Background()
	cache := m.NewCache(n + steps + 1)
	cache.SetDeepstackForTest(start, nb, sets)
	refLast, err := m.PrefillLogitsQwenVLForTest(ctx, ids, feats, start, nb, mrope, cache)
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
		out, toks := [][]float32{first}, []int{argmaxF(first)}
		for k := range steps {
			tok := toks[k]
			if forced != nil {
				tok = forced[k]
			}
			lg, err := r.ForwardMRoPE(m.EmbedResidentForTest(tok), n+k, n+k+delta)
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, append([]float32(nil), lg...))
			toks = append(toks, argmaxF(lg))
		}
		return out, toks
	}
	ref, teacher := decode(refLast, nil)
	for _, c := range []struct {
		name   string
		defect int
	}{
		{"none", 0},
		{"(4) the sets not added", deepDefectNotAdded},
		{"(5) added one layer late", deepDefectOneLayerLate},
		{"(6) added to the text rows too", deepDefectTextRows},
	} {
		deepDefectForTest = c.defect
		r.Reset()
		newLast, err := r.PrefillMRoPEDeepstackLast(ctx, dsSpliceRows(m, ids, feats, start, nb, hid), 0, mrope, sets, start, nb)
		deepDefectForTest = 0
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		got, _ := decode(append([]float32(nil), newLast...), teacher)
		worst, agree, real := 1.0, 0, 0
		for k := range ref {
			worst = math.Min(worst, dsCosine(ref[k], got[k]))
			ra, ga := argmaxF(ref[k]), argmaxF(got[k])
			if ra == ga {
				agree++
				continue
			}
			lp := dsLogSoftmax(ref[k])
			tie := math.Exp(lp[ga]) >= math.Exp(lp[ra])/2
			if !tie {
				real++
			}
			fmt.Printf("    step %d: today's top %d p %.4f, new %d (today's p %.4f): near-tie %v\n", k, ra, math.Exp(lp[ra]), ga, math.Exp(lp[ga]), tie)
		}
		pass := worst >= bar && real == 0
		fmt.Printf("[G-S10g] qwen3vl-tiny (16x24 image) %s: worst cosine %.7f, argmax %d/%d, non-tie differences %d\n", c.name, worst, agree, len(ref), real)
		if c.defect == 0 && !pass {
			t.Errorf("the DeepStack prefill misses the bar (worst %.7f, %d non-tie argmax differences)", worst, real)
		}
		if c.defect != 0 && pass {
			t.Errorf("planted defect %s left the bar green (worst %.7f)", c.name, worst)
		}
	}
	// A pass given too few sets is declined by name, not mangled; one with a set of the wrong size is an error.
	if _, err := r.PrefillMRoPEDeepstackLast(ctx, dsSpliceRows(m, ids, feats, start, nb, hid), 0, mrope, nil, start, nb); err == nil {
		t.Error("no sets was not declined")
	}
	if _, err := r.PrefillMRoPEDeepstackLast(ctx, dsSpliceRows(m, ids, feats, start, nb, hid), 0, mrope, [][]float32{sets[0][:hid]}, start, nb); err == nil {
		t.Error("a short set was not refused")
	}
	// Through the decoder's entry point: a turn with DeepStack sets now takes the resident prefill.
	stream, gen := m.GenerateQwenVLDeepstack(ctx, ids, start, nb, 0xfeed, func() ([]float32, [][]float32, error) { return feats, sets, nil },
		[][3]int{{1, 2 * gh, 2 * gw}}, merge, imgTok, steps, decoder.SamplingParams{})
	var gotToks []int
	for id := range stream {
		gotToks = append(gotToks, id)
	}
	if e := gen.Err(); e != nil {
		t.Fatal(e)
	}
	fmt.Printf("[G-S10g] through GenerateQwenVLDeepstack: resident prefill %v, tokens %v (today's path %v)\n", gen.ImgPrefillResident, gotToks, teacher[:steps])
	if !gen.ImgPrefillResident {
		t.Error("GenerateQwenVLDeepstack did not take the resident DeepStack prefill")
	}
	for k := range min(len(gotToks), steps) { // the first differing token must be a near-tie in today's logits at that step
		if gotToks[k] != teacher[k] {
			lp := dsLogSoftmax(ref[k])
			if math.Exp(lp[gotToks[k]]) < math.Exp(lp[teacher[k]])/2 {
				t.Errorf("the free runs first differ at step %d (%d against %d) and it is not a near-tie (p %.4f against %.4f)", k, gotToks[k], teacher[k], math.Exp(lp[gotToks[k]]), math.Exp(lp[teacher[k]]))
			}
			break
		}
	}
}

// TestS10DeepstackPrefillCUDA_real is G-S10g's real half, on Qwen3-VL-2B and the four F2a images at serve's 1,024-row cap, graded by S16's RE-REGISTERED bar (the Mac's, 2026-10-08, owner option a): the strict
// per-step 0.9999 is tighter than a shipped batched prefill's own distance from the CPU's int4 prefill, so the resident DeepStack prefill is held to non-inferiority against a text control. For each image prompt, ctlPrompts (12; four until the owner's decision (c) of 2026-10-08)
// text-only prompts of the same length run the same isolated comparison (the CPU's prefill and upload against CUDA's batched PrefillLast, both decoding the same 8 teacher-forced steps on CUDA). PASS when the image
// turn's worst per-step cosine (the last row and the 8 steps) is at least the control's minimum minus 0.005 and every argmax difference is an R10 near-tie; 0.005-0.015 below the control's minimum is parked; worse fails.
// Both image arms see the same features and sets (the CPU encoder's). Heavy: about 6-8 minutes, a line per stage.
func TestS10DeepstackPrefillCUDA_real(t *testing.T) {
	prevDeep := cudaDeepstackPrefillOn // production default is ON since G-S10g passed; the gate sets it explicitly and puts it back
	cudaDeepstackPrefillOn = true
	defer func() { cudaDeepstackPrefillOn = prevDeep }()
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy: set GOINFER_HEAVY_TESTS=1")
	}
	requireCUDADevice(t)
	home, _ := os.UserHomeDir()
	dir := filepath.Join(home, "models", "qwen3-vl-2b-instruct")
	if _, err := os.Stat(dir); err != nil {
		t.Skipf("no %s", dir)
	}
	// ctlPrompts is the text control's size per image length: 4 until the owner's decision (c) of 2026-10-08 made it 12. Seeds 1-4 are the original four, so the minimum can only be lower.
	const merge, steps, ctlPrompts = 2, 8, 12
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
	m, err := decoder.Load(dir, decoder.Options{Quant: "int4", Backend: "cuda", ResidentContext: 2048})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	r, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok || !r.mropePrefillReady {
		t.Fatalf("not CUDA-resident with the m-RoPE batched kernel: %s", m.ResidentDecline())
	}
	ctx := context.Background()
	hb := func(format string, a ...any) { fmt.Fprintf(os.Stderr, "[G-S10g] "+format+"\n", a...) } // a heartbeat per stage: this test runs for minutes
	for _, name := range []string{"gemma3_preprocess_image.png", "qwen25vl_preprocess_image.png", "glm_ocr/formula.png", "glm_ocr/table.png"} {
		t.Run(name, func(t *testing.T) { // a subtest per image so one can be re-read alone (-run 'TestS10DeepstackPrefillCUDA_real/qwen25vl')
			data, err := os.ReadFile(filepath.Join("../testdata", name))
			if err != nil {
				t.Fatal(err)
			}
			px, grid, err := multimodal.QwenPreprocess(data, pp)
			if err != nil {
				t.Fatal(err)
			}
			hb("%s: tower features and sets (CPU encoder)", name)
			feats, deep, err := enc.ForwardDeepstack(px, [][3]int{grid})
			if err != nil {
				t.Fatal(err)
			}
			nImg := grid[0] * grid[1] * grid[2] / (merge * merge)
			pre, err := tk.Encode("<|im_start|>user\n<|vision_start|>", false)
			if err != nil {
				t.Fatal(err)
			}
			post, err := tk.Encode("<|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n", false)
			if err != nil {
				t.Fatal(err)
			}
			ids := append(append([]int{}, pre...), make([]int, nImg)...)
			for k := range nImg {
				ids[len(pre)+k] = imgTok
			}
			ids = append(ids, post...)
			n, start := len(ids), len(pre)
			hid := len(feats) / nImg
			mrope, err := decoder.MRopePositionsForTest(ids, imgTok, [][3]int{grid}, merge)
			if err != nil {
				t.Fatal(err)
			}
			delta := mrope[n-1][0] + 1 - n
			hb("%s: %d rows, %d image; today's path (CPU prefill with the sets, upload, CUDA decode)", name, n, nImg)
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
				out, toks := [][]float32{first}, []int{argmaxF(first)}
				for k := range steps {
					tok := toks[k]
					if forced != nil {
						tok = forced[k]
					}
					lg, err := r.ForwardMRoPE(m.EmbedResidentForTest(tok), n+k, n+k+delta)
					if err != nil {
						t.Fatal(err)
					}
					out = append(out, append([]float32(nil), lg...))
					toks = append(toks, argmaxF(lg))
				}
				return out, toks
			}
			ref, teacher := decode(refLast, nil)
			hb("%s: the resident DeepStack prefill", name)
			r.Reset()
			newLast, _, err := m.ResidentMRoPEDeepstackPrefillForTest(ctx, r, ids, feats, start, nImg, mrope, deep)
			if err != nil {
				t.Fatalf("%s: resident DeepStack prefill: %v", name, err)
			}
			got, _ := decode(append([]float32(nil), newLast...), teacher)
			worst, ties, real := 1.0, 0, 0
			for k := range ref {
				worst = math.Min(worst, dsCosine(ref[k], got[k]))
				if ra, ga := argmaxF(ref[k]), argmaxF(got[k]); ra != ga {
					if lp := dsLogSoftmax(ref[k]); math.Exp(lp[ga]) >= math.Exp(lp[ra])/2 {
						ties++
					} else {
						real++
					}
				}
			}
			ctlMin := 1.0
			var perPrompt []float64 // each control prompt's own minimum, printed for context; the verdict uses their minimum as registered
			for seed := 1; seed <= ctlPrompts; seed++ {
				hb("%s: text control %d of %d (%d rows)", name, seed, ctlPrompts, n)
				tids := make([]int, n)
				for i := range tids {
					tids[i] = 1000 + (i*7919+seed*104729)%50000
				}
				tc := m.NewCache(n + steps + 1)
				tl, err := m.PrefillLogitsForTest(ctx, tids, tc)
				if err != nil {
					t.Fatal(err)
				}
				r.Reset()
				if err := m.ResidentUploadPrefillForTest(tc); err != nil {
					t.Fatal(err)
				}
				textDecode := func(first []float32, forced []int) ([][]float32, []int) {
					out, toks := [][]float32{append([]float32(nil), first...)}, []int{argmaxF(first)}
					for k := range steps {
						tok := toks[k]
						if forced != nil {
							tok = forced[k]
						}
						lg, err := r.Forward(m.EmbedResidentForTest(tok), n+k)
						if err != nil {
							t.Fatal(err)
						}
						out = append(out, append([]float32(nil), lg...))
						toks = append(toks, argmaxF(lg))
					}
					return out, toks
				}
				cRef, cTeach := textDecode(tl, nil)
				embs := make([][]float32, n)
				for i, id := range tids {
					embs[i] = m.EmbedResidentForTest(id)
				}
				r.Reset()
				first, err := r.PrefillLast(ctx, embs, 0)
				if err != nil {
					t.Fatalf("control's batched pass: %v", err)
				}
				cGot, _ := textDecode(first, cTeach)
				pm := 1.0
				for k := range cRef {
					pm = math.Min(pm, dsCosine(cRef[k], cGot[k]))
				}
				perPrompt = append(perPrompt, pm)
				ctlMin = math.Min(ctlMin, pm)
			}
			verdict := "PASS"
			switch {
			case real > 0 || worst < ctlMin-0.015:
				verdict = "FAIL"
			case worst < ctlMin-0.005:
				verdict = "PARKED"
			}
			fmt.Fprintf(os.Stderr, "[G-S10g] qwen3-vl-2b %s (%d rows, %d image, hidden %d): worst cosine %.7f against the text control's minimum %.7f (%d prompts; each prompt's own minimum %.4f); argmax: %d near-ties, %d real: %s\n",
				name, n, nImg, hid, worst, ctlMin, ctlPrompts, perPrompt, ties, real, verdict)
			if verdict != "PASS" {
				t.Errorf("%s: %s (worst %.7f, control minimum %.7f, %d non-tie argmax differences)", name, verdict, worst, ctlMin, real)
			}
		})
	}
}

// TestS10DeepstackPrefillCUDA_diag is an EXPLORATORY diagnostic, not a gate: G-S10g's real reading failed on table.png (0.9125 against a control minimum of 0.9539) and was parked on the 4x6 image. It re-runs the
// comparison on those two images in cuts that separate the candidate mechanisms, printing the per-step cosines: (a) as the gate; (b) with NO DeepStack sets in either arm (is the gap the batched prefill on image rows,
// not the injection?); (c) as the gate with the fast prefill levers forced off (is it the L2/L3 levers?). Same features, same prompt, same teacher-forced steps in every cut.
func TestS10DeepstackPrefillCUDA_diag(t *testing.T) {
	prevDeep := cudaDeepstackPrefillOn // production default is ON since G-S10g passed; the gate sets it explicitly and puts it back
	cudaDeepstackPrefillOn = true
	defer func() { cudaDeepstackPrefillOn = prevDeep }()
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy: set GOINFER_HEAVY_TESTS=1")
	}
	requireCUDADevice(t)
	home, _ := os.UserHomeDir()
	dir := filepath.Join(home, "models", "qwen3-vl-2b-instruct")
	const merge, steps = 2, 8
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
	if limit := 1024 * pp.MergeSize * pp.MergeSize * pp.PatchSize * pp.PatchSize; pp.MaxPixels > limit {
		pp.MaxPixels = limit
	}
	m, err := decoder.Load(dir, decoder.Options{Quant: "int4", Backend: "cuda", ResidentContext: 2048})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	r := m.ResidentForwardForTest().(*cudaResident)
	ctx := context.Background()
	for _, name := range []string{"glm_ocr/table.png", "qwen25vl_preprocess_image.png"} {
		data, _ := os.ReadFile(filepath.Join("../testdata", name))
		px, grid, err := multimodal.QwenPreprocess(data, pp)
		if err != nil {
			t.Fatal(err)
		}
		feats, deep, err := enc.ForwardDeepstack(px, [][3]int{grid})
		if err != nil {
			t.Fatal(err)
		}
		nImg := grid[0] * grid[1] * grid[2] / (merge * merge)
		pre, _ := tk.Encode("<|im_start|>user\n<|vision_start|>", false)
		post, _ := tk.Encode("<|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n", false)
		ids := append(append([]int{}, pre...), make([]int, nImg)...)
		for k := range nImg {
			ids[len(pre)+k] = imgTok
		}
		ids = append(ids, post...)
		n, start := len(ids), len(pre)
		hid := len(feats) / nImg
		mrope, _ := decoder.MRopePositionsForTest(ids, imgTok, [][3]int{grid}, merge)
		delta := mrope[n-1][0] + 1 - n
		for _, cut := range []struct {
			name     string
			noSets   bool
			exactKrn bool
		}{{"(a) as the gate", false, false}, {"(b) no DeepStack sets in either arm", true, false}, {"(c) as the gate, fast prefill levers off", false, true}} {
			sets := deep
			if cut.noSets {
				sets = nil
			}
			if cut.exactKrn {
				decoder.SetKnobEnvForTest(t, m, "GOINFER_CUDA_FAST_PREFILL_FLOOR", "1000000")
			}
			cache := m.NewCache(n + steps + 1)
			cache.SetDeepstackForTest(start, nImg, sets)
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
				out, toks := [][]float32{first}, []int{argmaxF(first)}
				for k := range steps {
					tok := toks[k]
					if forced != nil {
						tok = forced[k]
					}
					lg, err := r.ForwardMRoPE(m.EmbedResidentForTest(tok), n+k, n+k+delta)
					if err != nil {
						t.Fatal(err)
					}
					out = append(out, append([]float32(nil), lg...))
					toks = append(toks, argmaxF(lg))
				}
				return out, toks
			}
			ref, teacher := decode(refLast, nil)
			r.Reset()
			var newLast []float32
			if sets == nil {
				newLast, _, err = m.ResidentMRoPEPrefillForTest(ctx, r, ids, feats, start, nImg, mrope)
			} else {
				newLast, _, err = m.ResidentMRoPEDeepstackPrefillForTest(ctx, r, ids, feats, start, nImg, mrope, sets)
			}
			if err != nil {
				t.Fatal(err)
			}
			got, _ := decode(append([]float32(nil), newLast...), teacher)
			line := ""
			for k := range ref {
				line += fmt.Sprintf(" %.4f", dsCosine(ref[k], got[k]))
			}
			fmt.Fprintf(os.Stderr, "[G-S10g diag] %s (%d rows, %d image, hidden %d) %s: per-step cosine:%s\n", name, n, nImg, hid, cut.name, line)
		}
	}
}

// TestS10DeepstackPrefillCUDA_speed is an EXPLORATORY split of one image turn's prefill on Qwen3-VL-2B and table.png (986 rows), single samples, nothing graded: today's path (the CPU prefill with the sets, then the
// KV upload) against the resident DeepStack prefill. It exists so the decision on the parked image is made with the stake in view; the night record is the S7 cell.
func TestS10DeepstackPrefillCUDA_speed(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy: set GOINFER_HEAVY_TESTS=1")
	}
	prevDeep := cudaDeepstackPrefillOn // production default is ON since G-S10g passed; the gate sets it explicitly and puts it back
	cudaDeepstackPrefillOn = true
	defer func() { cudaDeepstackPrefillOn = prevDeep }()
	requireCUDADevice(t)
	home, _ := os.UserHomeDir()
	dir := filepath.Join(home, "models", "qwen3-vl-2b-instruct")
	const merge = 2
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
	if limit := 1024 * pp.MergeSize * pp.MergeSize * pp.PatchSize * pp.PatchSize; pp.MaxPixels > limit {
		pp.MaxPixels = limit
	}
	m, err := decoder.Load(dir, decoder.Options{Quant: "int4", Backend: "cuda", ResidentContext: 2048})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	r := m.ResidentForwardForTest().(*cudaResident)
	ctx := context.Background()
	data, _ := os.ReadFile("../testdata/glm_ocr/table.png")
	px, grid, err := multimodal.QwenPreprocess(data, pp)
	if err != nil {
		t.Fatal(err)
	}
	feats, deep, err := enc.ForwardDeepstack(px, [][3]int{grid})
	if err != nil {
		t.Fatal(err)
	}
	nImg := grid[0] * grid[1] * grid[2] / (merge * merge)
	pre, _ := tk.Encode("<|im_start|>user\n<|vision_start|>", false)
	post, _ := tk.Encode("<|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n", false)
	ids := append(append([]int{}, pre...), make([]int, nImg)...)
	for k := range nImg {
		ids[len(pre)+k] = imgTok
	}
	ids = append(ids, post...)
	n, start := len(ids), len(pre)
	mrope, _ := decoder.MRopePositionsForTest(ids, imgTok, [][3]int{grid}, merge)
	for round := range 2 { // the second round is warm
		t0 := time.Now()
		cache := m.NewCache(n + 9)
		cache.SetDeepstackForTest(start, nImg, deep)
		if _, err := m.PrefillLogitsQwenVLForTest(ctx, ids, feats, start, nImg, mrope, cache); err != nil {
			t.Fatal(err)
		}
		cache.SetDeepstackForTest(0, 0, nil)
		r.Reset()
		if err := m.ResidentUploadPrefillForTest(cache); err != nil {
			t.Fatal(err)
		}
		cpu := time.Since(t0)
		r.Reset()
		t1 := time.Now()
		if _, _, err := m.ResidentMRoPEDeepstackPrefillForTest(ctx, r, ids, feats, start, nImg, mrope, deep); err != nil {
			t.Fatal(err)
		}
		fmt.Fprintf(os.Stderr, "[G-S10g speed] round %d, %d rows: CPU prefill + upload %s, resident DeepStack prefill %s (exploratory, single samples)\n", round, n, cpu.Round(time.Millisecond), time.Since(t1).Round(time.Millisecond))
	}
}
