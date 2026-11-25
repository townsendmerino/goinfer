//go:build gpu && goinfer_testhooks

package webgpu

import (
	"context"
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/backend/webgpu/internal/kernels"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/testgate"
	"github.com/townsendmerino/goinfer/internal/vision"
	"github.com/townsendmerino/goinfer/multimodal"
)

// TestForwardMRoPE_constantShiftInvariance is the WebGPU twin of
// cuda/forwardmrope_parity_test.go, whose doc comment has the full rationale: RoPE's
// relative-angle invariance means a decode run where every step's rope angle is shifted by the
// same constant delta must match delta=0 exactly, a property independent of any family's m-RoPE
// usage.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags 'gpu goinfer_testhooks' ./gpu/ -run TestForwardMRoPE_constantShiftInvariance -v -timeout 20m
func TestForwardMRoPE_constantShiftInvariance(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1")
	}
	if _, err := New(); err != nil {
		t.Skipf("no webgpu: %v", err)
	}
	home, _ := os.UserHomeDir()
	path := os.Getenv("GOINFER_QWEN15_CKPT")
	if path == "" {
		path = home + "/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s: %v", path, err)
	}

	for _, delta := range []int{17, -13} {
		t.Run(map[bool]string{true: "positive", false: "negative"}[delta > 0], func(t *testing.T) {
			mc, err := decoder.Load(path, decoder.Options{Backend: "webgpu", Quant: "int8int8"})
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			defer mc.Close()
			rf := mc.ResidentForwardForTest()
			if rf == nil {
				t.Skip("model not resident-eligible on this build")
			}
			mrope := decoder.SubsystemsOf(rf).Decode
			ok := mrope.MRoPEDecodeAvailable()
			if !ok {
				t.Fatal("residentDecoder does not satisfy decoder.ResidentDecodeSubsystem — interface wiring broken")
			}
			_, _, _, _, _, _, vocab := mc.Dims()

			const N = 24
			rng := rand.New(rand.NewSource(11))
			embs := make([][]float32, N)
			for i := range embs {
				embs[i] = mc.EmbedResidentForTest(rng.Intn(vocab - 1))
			}

			var want []float32
			for i, e := range embs {
				l, err := rf.Forward(e, i)
				if err != nil {
					t.Fatalf("ordinary Forward step %d: %v", i, err)
				}
				want = l
			}

			rf.Reset()

			var got []float32
			for i, e := range embs {
				l, err := mrope.ForwardMRoPE(e, i, i+delta)
				if err != nil {
					t.Fatalf("ForwardMRoPE step %d (ropePos=%d): %v", i, i+delta, err)
				}
				got = l
			}

			cos, maxAbs := cosSim(want, got)
			t.Logf("delta=%d: cosine(ordinary, constant-shifted) = %.8f maxAbs=%.4g", delta, cos, maxAbs)
			if cos < 0.9999 {
				t.Errorf("delta=%d: cosine = %.8f, want ~1.0 — RoPE's relative-angle invariance is violated, ForwardMRoPE's ropePos wiring is likely wrong", delta, cos)
			}
		})
	}
}

// TestForwardMRoPE_equalArgsMatchesForward checks the interface's own stated contract
// (decoder/residency.go: "ForwardMRoPE(emb, pos, pos) must equal Forward(emb, pos) exactly").
func TestForwardMRoPE_equalArgsMatchesForward(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1")
	}
	if _, err := New(); err != nil {
		t.Skipf("no webgpu: %v", err)
	}
	home, _ := os.UserHomeDir()
	path := os.Getenv("GOINFER_QWEN15_CKPT")
	if path == "" {
		path = home + "/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s: %v", path, err)
	}
	mc, err := decoder.Load(path, decoder.Options{Backend: "webgpu", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer mc.Close()
	rf := mc.ResidentForwardForTest()
	if rf == nil {
		t.Skip("model not resident-eligible on this build")
	}
	mrope := decoder.SubsystemsOf(rf).Decode
	ok := mrope.MRoPEDecodeAvailable()
	if !ok {
		t.Fatal("residentDecoder does not satisfy decoder.ResidentDecodeSubsystem")
	}
	_, _, _, _, _, _, vocab := mc.Dims()
	rng := rand.New(rand.NewSource(3))
	const N = 8
	embs := make([][]float32, N)
	for i := range embs {
		embs[i] = mc.EmbedResidentForTest(rng.Intn(vocab - 1))
	}

	var want []float32
	for i, e := range embs {
		l, err := rf.Forward(e, i)
		if err != nil {
			t.Fatalf("Forward step %d: %v", i, err)
		}
		want = l
	}
	rf.Reset()
	var got []float32
	for i, e := range embs {
		l, err := mrope.ForwardMRoPE(e, i, i) // ropePos == pos
		if err != nil {
			t.Fatalf("ForwardMRoPE step %d: %v", i, err)
		}
		got = l
	}
	if len(want) != len(got) {
		t.Fatalf("length %d vs %d", len(want), len(got))
	}
	for i := range want {
		if want[i] != got[i] {
			t.Fatalf("logit %d: Forward=%v ForwardMRoPE(pos,pos)=%v — must be bit-identical, same dispatch", i, want[i], got[i])
		}
	}
}

// pwWorst is a path's worst agreement with its reference: the lowest cosine and the highest relative L2 error.
type pwWorst struct{ cos, rel float64 }

// A NaN logit, an all-zero row or a length mismatch reads as the worst value (cosine -2, relL2 +Inf), so no bar passes it.
func (w *pwWorst) add(got, want []float32) {
	if len(got) != len(want) || len(want) == 0 {
		w.cos, w.rel = -2, math.Inf(1)
		return
	}
	var ab, aa, bb, dd float64
	for i := range want {
		g, r := float64(got[i]), float64(want[i])
		ab, aa, bb, dd = ab+g*r, aa+g*g, bb+r*r, dd+(g-r)*(g-r)
	}
	c, rel := ab/math.Sqrt(aa*bb), math.Sqrt(dd/bb)
	if math.IsNaN(c) || math.IsInf(c, 0) {
		c = -2
	}
	if math.IsNaN(rel) {
		rel = math.Inf(1)
	}
	w.cos, w.rel = math.Min(w.cos, c), math.Max(w.rel, rel)
}

// TestGlmOcrResidentParityWebGPU is Metal's G-PR3 gate (TestGlmOcrResidentParityMetal) on WebGPU: glm-ocr-tiny's text
// decoder, pairwise RoPE over contiguous m-RoPE sections, resident at int8int8 against the CPU int8int8, unpeaked as
// Metal's and CUDA's GLM-OCR gates run it. Paths: text decode over 48 positions; the image prompt of
// glm_ocr_tiny_mrope_golden.json by the route production takes (the resident m-RoPE prefill declines on a pairwise model,
// so the CPU prefill is uploaded); and decode past the image through ForwardMRoPE (rotation position != storage
// position), teacher-forced on the golden continuation. Bars, Metal's: worst cosine >= 0.995 and relL2 <= 0.15 on every
// path. Text decode and decode past the uploaded image also run on the f16 and int8 caches (their own rope-store twins),
// each against the CPU holding its cache at the same precision (Options.KVQuant: per-row f16, per-(position, KV-head)
// symmetric int8, the GPU's schemes). Every path with the NeoX kernels bound (kernels.PairwiseNeoXForTest) must read below
// the cosine bar, and so must every f16 and int8 path with only the K-cache stores bound NeoX
// (kernels.PairwiseNeoXStoresForTest).
func TestGlmOcrResidentParityWebGPU(t *testing.T) {
	if c, err := New(); err != nil {
		t.Skipf("no webgpu: %v", err)
	} else {
		c.Close()
	}
	raw, err := os.ReadFile("../../testdata/glm_ocr_tiny_mrope_golden.json")
	if err != nil {
		t.Fatalf("the committed m-RoPE golden: %v", err)
	}
	var g struct {
		PromptIDs       []int       `json:"prompt_ids"`
		ImageToken      int         `json:"image_token_id"`
		ImageStart      int         `json:"image_start"`
		NImage          int         `json:"n_image_tokens"`
		Grid            [3]int      `json:"grid_thw"`
		Merge           int         `json:"merge"`
		Features        [][]float32 `json:"image_features"`
		NNew            int         `json:"n_new"`
		ContinuationIDs []int       `json:"continuation_ids"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	const ckpt, cosBar, relBar = "../../testdata/glm-ocr-tiny", 0.995, 0.15
	// load opens a model the caller closes; cpuLoad's live for the test.
	load := func(backend, kv string) *decoder.Model {
		o := decoder.Options{Backend: backend, Quant: "int8int8", KVPrecision: kv}
		if backend == "cpu" {
			o.KVPrecision, o.KVQuant = "", kv
		}
		m, err := decoder.Load(ckpt, o)
		if err != nil {
			t.Fatalf("load %s --kv %q: %v", backend, kv, err)
		}
		return m
	}
	cpuLoad := func(kv string) *decoder.Model {
		m := load("cpu", kv)
		t.Cleanup(func() { m.Close() })
		return m
	}
	cpu := cpuLoad("")
	_, _, _, _, _, _, vocab := cpu.Dims()
	text := make([]int, 48)
	for i := range text {
		text[i] = (i*37 + 3) % vocab
	}
	var feats []float32
	for _, row := range g.Features {
		feats = append(feats, row...)
	}
	mropePos, err := decoder.MRopePositionsForTest(g.PromptIDs, g.ImageToken, [][3]int{g.Grid}, g.Merge)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	rowC := cpu.NewCache(len(text))
	row := make([][]float32, len(text))
	for i, tok := range text {
		l, err := cpu.ForwardForTest(tok, rowC)
		if err != nil {
			t.Fatal(err)
		}
		row[i] = slices.Clone(l)
	}
	imgC := cpu.NewCache(len(g.PromptIDs) + g.NNew)
	if _, err := cpu.PrefillLogitsQwenVLForTest(ctx, g.PromptIDs, feats, g.ImageStart, g.NImage, mropePos, imgC); err != nil {
		t.Fatal(err)
	}
	delta := imgC.MRopeDeltaForTest()
	if delta == 0 {
		t.Fatal("mropeDelta == 0: decode past the image would not exercise ropePos != pos")
	}
	var past [][]float32
	for k := 0; k < g.NNew-1; k++ {
		l, err := cpu.ForwardForTest(g.ContinuationIDs[k], imgC)
		if err != nil {
			t.Fatal(err)
		}
		past = append(past, slices.Clone(l))
	}
	rowKV, pastKV, cpuKV := map[string][][]float32{}, map[string][][]float32{}, map[string]*decoder.Model{}
	for _, kv := range []string{"f16", "i8"} {
		c := cpuLoad(kv)
		cpuKV[kv] = c
		cc := c.NewCache(len(text))
		for _, tok := range text {
			l, err := c.ForwardForTest(tok, cc)
			if err != nil {
				t.Fatal(err)
			}
			rowKV[kv] = append(rowKV[kv], slices.Clone(l))
		}
		ic := c.NewCache(len(g.PromptIDs) + g.NNew)
		if _, err := c.PrefillLogitsQwenVLForTest(ctx, g.PromptIDs, feats, g.ImageStart, g.NImage, mropePos, ic); err != nil {
			t.Fatal(err)
		}
		for k := 0; k < g.NNew-1; k++ {
			l, err := c.ForwardForTest(g.ContinuationIDs[k], ic)
			if err != nil {
				t.Fatal(err)
			}
			pastKV[kv] = append(pastKV[kv], slices.Clone(l))
		}
	}
	measure := func() map[string]pwWorst {
		out := map[string]pwWorst{}
		for _, kv := range []string{"f16", "i8"} { // the f16 and int8 caches' rope-store twins, on text decode
			func() {
				m := load("webgpu", kv)
				defer m.Close()
				rd, ok := m.ResidentForwardForTest().(*residentDecoder)
				if !ok || m.KVDeliveredFor("webgpu") != kv {
					t.Fatalf("--kv %s: resident %v, KV delivered %q; decline: %s", kv, ok, m.KVDeliveredFor("webgpu"), m.ResidentDecline())
				}
				w := pwWorst{cos: 1}
				rd.Reset()
				for i, tok := range text {
					l, err := rd.Forward(m.EmbedResidentForTest(tok), i)
					if err != nil {
						t.Fatal(err)
					}
					w.add(l, rowKV[kv][i])
				}
				out["text decode, --kv "+kv] = w
				upC := cpuKV[kv].NewCache(len(g.PromptIDs) + g.NNew)
				if _, err := cpuKV[kv].PrefillLogitsQwenVLForTest(ctx, g.PromptIDs, feats, g.ImageStart, g.NImage, mropePos, upC); err != nil {
					t.Fatal(err)
				}
				rd.Reset()
				if err := m.ResidentUploadPrefillForTest(upC); err != nil {
					t.Fatalf("--kv %s: upload the CPU image prefill: %v", kv, err)
				}
				pst, at := pwWorst{cos: 1}, len(g.PromptIDs)
				for k := 0; k < g.NNew-1; k++ {
					l, err := rd.ForwardMRoPE(m.EmbedResidentForTest(g.ContinuationIDs[k]), at+k, at+k+delta)
					if err != nil {
						t.Fatal(err)
					}
					pst.add(l, pastKV[kv][k])
				}
				out["past the image, --kv "+kv] = pst
			}()
		}
		m := load("webgpu", "")
		defer m.Close()
		rd, ok := m.ResidentForwardForTest().(*residentDecoder)
		if !ok {
			t.Fatalf("not webgpu-resident: %s", m.ResidentDecline())
		}
		if !rd.rm.Pairwise {
			t.Fatal("the resident was built without RunModel.Pairwise")
		}
		dec, pst := pwWorst{cos: 1}, pwWorst{cos: 1}
		rd.Reset()
		for i, tok := range text {
			l, err := rd.Forward(m.EmbedResidentForTest(tok), i)
			if err != nil {
				t.Fatal(err)
			}
			dec.add(l, row[i])
		}
		rd.Reset()
		if _, _, err := m.ResidentMRoPEPrefillForTest(ctx, rd, g.PromptIDs, feats, g.ImageStart, g.NImage, mropePos); err == nil {
			t.Fatal("the resident m-RoPE prefill ran on a pairwise model; its kernels are NeoX")
		}
		upC := cpu.NewCache(len(g.PromptIDs) + g.NNew)
		if _, err := cpu.PrefillLogitsQwenVLForTest(ctx, g.PromptIDs, feats, g.ImageStart, g.NImage, mropePos, upC); err != nil {
			t.Fatal(err)
		}
		rd.Reset()
		if err := m.ResidentUploadPrefillForTest(upC); err != nil {
			t.Fatalf("upload the CPU image prefill: %v", err)
		}
		at := len(g.PromptIDs)
		for k := 0; k < g.NNew-1; k++ {
			l, err := rd.ForwardMRoPE(m.EmbedResidentForTest(g.ContinuationIDs[k]), at+k, at+k+delta)
			if err != nil {
				t.Fatal(err)
			}
			pst.add(l, past[k])
		}
		out["text decode, 48 positions"], out["decode past the image"] = dec, pst
		return out
	}
	for name, w := range measure() {
		t.Logf("PAIRWISE %-28s worst cos %.6f relL2 %.2e", name, w.cos, w.rel)
		if !(w.cos >= cosBar && w.rel <= relBar) {
			t.Errorf("%s diverges from the CPU: worst cosine %.6f (want >= %.3f), relL2 %.4f (want <= %.2f)", name, w.cos, cosBar, w.rel, relBar)
		}
	}
	kernels.PairwiseNeoXForTest = true
	t.Cleanup(func() { kernels.PairwiseNeoXForTest = false })
	neox := measure()
	kernels.PairwiseNeoXForTest = false
	for name, w := range neox {
		t.Logf("NeoX     %-28s worst cos %.6f relL2 %.4f", name, w.cos, w.rel)
		if !(w.cos < cosBar) {
			t.Errorf("the gate is BLIND on %q: with the NeoX kernels it still reads cosine %.6f >= %.3f", name, w.cos, cosBar)
		}
	}
	kernels.PairwiseNeoXStoresForTest = true // only the K-cache stores miswired: every f16 and int8 cache path must read red
	t.Cleanup(func() { kernels.PairwiseNeoXStoresForTest = false })
	stores := measure()
	kernels.PairwiseNeoXStoresForTest = false
	for _, kv := range []string{"f16", "i8"} {
		for _, name := range []string{"text decode, --kv " + kv, "past the image, --kv " + kv} {
			w, ok := stores[name]
			t.Logf("NeoX K-store %-24s worst cos %.6f relL2 %.4f", name, w.cos, w.rel)
			if !ok || !(w.cos < cosBar) {
				t.Errorf("the gate is BLIND to a miswired K-cache store on %q: cosine %.6f >= %.3f (measured %v)", name, w.cos, cosBar, ok)
			}
		}
	}
}

// TestGlmOcrWebGPUResidentRealImage is CUDA's TestGlmOcrCUDAResidentRealImage, its whole-turn half, on WebGPU: the real
// GLM-OCR checkpoint at int4 on both arms, a rendered page through the real tower, 64 greedy tokens through GenerateQwenVL
// on the resident and on the CPU. The image prefill is the CPU's, uploaded (the resident m-RoPE prefill declines on a
// pairwise model), and every token after it decodes on the resident through the pairwise twins. The bar is CUDA's: the
// two arms' tokens are identical.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_GLM_OCR=~/models/glm-ocr go test -tags 'gpu goinfer_testhooks' ./backend/webgpu/ \
//	  -run TestGlmOcrWebGPUResidentRealImage -v -timeout 30m
func TestGlmOcrWebGPUResidentRealImage(t *testing.T) {
	testgate.RequireHeavy(t)
	ckpt := decoder.AssetPathForTest(t, "GOINFER_GLM_OCR")
	if c, err := New(); err != nil {
		t.Skipf("no webgpu: %v", err)
	} else {
		c.Close()
	}
	pp, err := multimodal.LoadGlmOcrPreprocessConfig(ckpt)
	if err != nil {
		t.Fatal(err)
	}
	enc, err := vision.LoadGlmOcrVisionEncoder(ckpt, false)
	if err != nil {
		t.Fatal(err)
	}
	const ctxLen, imageToken, merge = 4096, 59280, 2
	mg, err := decoder.Load(ckpt, decoder.Options{Backend: "webgpu", Quant: "int4", ResidentContext: ctxLen})
	if err != nil {
		t.Fatalf("load webgpu: %v", err)
	}
	defer mg.Close()
	if rd, ok := mg.ResidentForwardForTest().(*residentDecoder); !ok || !rd.rm.Pairwise {
		t.Fatalf("glm_ocr is not resident with the pairwise twins: %s", mg.ResidentDecline())
	}
	t.Logf("WebGPU resident: %s (cap %d)", mg.DecodePath(), mg.ResidentContextCap())
	mcpu, err := decoder.Load(ckpt, decoder.Options{Backend: "cpu", Quant: "int4"})
	if err != nil {
		t.Fatalf("load cpu: %v", err)
	}
	defer mcpu.Close()
	for _, name := range []string{"invoice", "table", "formula"} {
		t.Run(name, func(t *testing.T) {
			var g struct {
				InputIDs   []int `json:"input_ids"`
				ImageStart int   `json:"image_token_start"`
				NImgTokens int   `json:"n_image_tokens"`
				HFTokens   []int `json:"hf_tokens"`
				NNew       int   `json:"n_new"`
			}
			if err := decoder.ReadGoldenJSONForTest(filepath.Join("../../testdata/glm_ocr/golden_"+name+".json.gz"), &g); err != nil {
				t.Fatal(err)
			}
			if len(g.InputIDs)+g.NNew > ctxLen {
				t.Skipf("%d prompt tokens + %d new do not fit the %d-token context this test runs at", len(g.InputIDs), g.NNew, ctxLen)
			}
			png, err := os.ReadFile(filepath.Join("../../testdata/glm_ocr", name+".png"))
			if err != nil {
				t.Fatal(err)
			}
			pv, grid, err := multimodal.QwenPreprocess(png, pp)
			if err != nil {
				t.Fatal(err)
			}
			feats, err := enc.Forward(pv, [][3]int{grid})
			if err != nil {
				t.Fatal(err)
			}
			features := func() ([]float32, error) { return feats, nil }
			run := func(m *decoder.Model, label string) []int {
				stream, gen := m.GenerateQwenVL(context.Background(), g.InputIDs, g.ImageStart, g.NImgTokens, 0, features, [][3]int{grid}, merge, imageToken, g.NNew, decoder.SamplingParams{Temperature: 0})
				var out []int
				for id := range stream {
					out = append(out, id)
				}
				if err := gen.Err(); err != nil {
					t.Fatalf("%s: %v", label, err)
				}
				if gen.ImgPrefillResident {
					t.Errorf("%s: the image prefill ran resident; on a pairwise model it must be the CPU's, uploaded", label)
				}
				if m == mg && gen.ResidentDeclined != "" {
					t.Errorf("%s: the turn did not decode on the resident: %s", label, gen.ResidentDeclined)
				}
				return out
			}
			res, cpu := run(mg, "resident"), run(mcpu, "cpu")
			first := -1
			for i := 0; i < max(len(res), len(cpu)); i++ {
				if i >= len(res) || i >= len(cpu) || res[i] != cpu[i] {
					first = i
					break
				}
			}
			t.Logf("greedy %d tokens: resident vs CPU int4 first divergence %d (-1: identical); HF f32 tokens equal to the resident's: %v",
				g.NNew, first, slices.Equal(res, g.HFTokens))
			if first >= 0 {
				t.Errorf("resident and CPU int4 greedy tokens diverge at step %d:\n resident %v\n cpu      %v", first, res, cpu)
			}
		})
	}
}
