//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// glmOcrMRopeGolden is the subset of testdata/glm_ocr_tiny_mrope_golden.json this gate reads: the
// prompt with a 12-token image-placeholder run, random "image features" standing in for the tower's
// merged rows, and the continuation HF produced. Positions are recomputed by the REAL
// mropePositions (decoder.MRopePositionsForTest), exactly as GenerateQwenVL does.
type glmOcrMRopeGolden struct {
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

// TestGlmOcrResidentParityCUDA: GLM-OCR's text decoder is CUDA-resident, with GPT-J PAIRWISE rotation
// (cuda/rope_pairwise.cu). Every other rope kernel is NeoX, and a resident built on them reads a
// negative cosine against the CPU on this fixture. The paths, each against the CPU at the same
// quantization:
//
//   - text only, 48 tokens, sequential decode (rope_kv_pw) and batched prefill (rope_kv_batched_pw);
//   - an IMAGE prompt through the m-RoPE batched prefill (PrefillMRoPELast -> rope_kv_mrope_batched_pw)
//     with the real mropePositions triples, 12 image rows carrying distinct (t,h,w);
//   - DECODE PAST THE IMAGE through ForwardMRoPE(emb, pos, pos+mropeDelta): the rotation position is
//     not the storage position there (the image grid compresses the position count), teacher-forced
//     on the HF continuation so both sides see the same tokens.
//
// And it proves it can fail: the same resident with the NeoX kernels rebound (decode, batched and
// m-RoPE) must read below the bar on every path.
func TestGlmOcrResidentParityCUDA(t *testing.T) {
	const ckpt = "../testdata/glm-ocr-tiny"
	requireDeviceAndFixture(t, ckpt)
	raw, err := os.ReadFile("../testdata/glm_ocr_tiny_mrope_golden.json")
	if err != nil {
		t.Skipf("no m-RoPE golden: %v (run scripts/pin_glm_ocr_tiny.py)", err)
	}
	var g glmOcrMRopeGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	mRes, err := decoder.Load(ckpt, decoder.Options{Backend: "cuda", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load cuda: %v", err)
	}
	defer mRes.Close()
	rf := mRes.ResidentForwardForTest()
	cr, ok := rf.(*cudaResident)
	if !ok {
		t.Fatalf("glm_ocr did not go CUDA-resident (%T) — decline: %s", rf, mRes.ResidentDecline())
	}
	if !cr.pairwiseRoPE || !cr.mropePrefillReady {
		t.Fatalf("pairwiseRoPE=%v mropePrefillReady=%v: the pairwise kernels (decode, batched, m-RoPE) are not all bound", cr.pairwiseRoPE, cr.mropePrefillReady)
	}
	mrope, ok := rf.(decoder.ResidentMRoPE)
	if !ok {
		t.Fatal("cudaResident does not satisfy decoder.ResidentMRoPE")
	}
	rmp, ok := rf.(decoder.ResidentMRoPEPrefill)
	if !ok {
		t.Fatal("cudaResident does not satisfy decoder.ResidentMRoPEPrefill")
	}
	mCPU, err := decoder.Load(ckpt, decoder.Options{Backend: "cpu", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load cpu: %v", err)
	}
	defer mCPU.Close()
	_, _, _, _, _, _, vocab := mCPU.Dims()

	type metrics struct{ cos, rel float64 }
	worst := func(m *metrics, got, want []float32) {
		c, _ := cosF32(got, want)
		m.cos = math.Min(m.cos, c)
		m.rel = math.Max(m.rel, relL2(got, want))
	}
	type result struct{ decode, prefill, imagePrefill, pastImage metrics }

	feats := make([]float32, 0, g.NImage*len(g.Features[0]))
	for _, row := range g.Features {
		feats = append(feats, row...)
	}
	mropePos, err := decoder.MRopePositionsForTest(g.PromptIDs, g.ImageToken, [][3]int{g.Grid}, g.Merge)
	if err != nil {
		t.Fatalf("mropePositions: %v", err)
	}
	text := make([]int, 48) // >= 32 positions, the length the O1 bypass measurement used
	for i := range text {
		text[i] = (i*37 + 3) % vocab
	}

	measure := func() result {
		r := result{decode: metrics{1, 0}, prefill: metrics{1, 0}, imagePrefill: metrics{1, 0}, pastImage: metrics{1, 0}}
		// text only: sequential decode, every position.
		embs := make([][]float32, len(text))
		for i, tok := range text {
			embs[i] = mRes.EmbedResidentForTest(tok)
		}
		cache := mCPU.NewCache(len(text))
		cpu := make([][]float32, len(text))
		for i, tok := range text {
			lc, err := mCPU.ForwardForTest(tok, cache)
			if err != nil {
				t.Fatalf("cpu forward[%d]: %v", i, err)
			}
			cpu[i] = append([]float32(nil), lc...)
		}
		cr.Reset()
		for i := range text {
			lr, err := cr.Forward(embs[i], i)
			if err != nil {
				t.Fatalf("resident forward[%d]: %v", i, err)
			}
			worst(&r.decode, lr, cpu[i])
		}
		// text only: batched prefill, last token.
		cr.Reset()
		got, err := cr.PrefillLast(context.Background(), embs, 0)
		if err != nil {
			t.Fatalf("PrefillLast: %v", err)
		}
		worst(&r.prefill, got, cpu[len(text)-1])

		// image prompt: m-RoPE batched prefill on the resident vs the CPU prefill, then decode past the image.
		ccache := mCPU.NewCache(len(g.PromptIDs) + g.NNew)
		wantImg, err := mCPU.PrefillLogitsQwenVLForTest(context.Background(), g.PromptIDs, feats, g.ImageStart, g.NImage, mropePos, ccache)
		if err != nil {
			t.Fatalf("cpu image prefill: %v", err)
		}
		cr.Reset()
		gotImg, gpuPos, err := mRes.ResidentMRoPEPrefillForTest(context.Background(), rmp, g.PromptIDs, feats, g.ImageStart, g.NImage, mropePos)
		if err != nil {
			t.Fatalf("resident m-RoPE prefill: %v", err)
		}
		worst(&r.imagePrefill, gotImg, wantImg)
		delta := ccache.MRopeDeltaForTest()
		if delta == 0 {
			t.Fatalf("mropeDelta == 0: the image block did not compress the positions, so decode-past-image would not exercise ropePos != pos")
		}
		pos := gpuPos
		for k := 0; k < g.NNew-1; k++ {
			tok := g.ContinuationIDs[k] // teacher-forced: identical tokens on both sides
			lc, err := mCPU.ForwardForTest(tok, ccache)
			if err != nil {
				t.Fatalf("cpu decode past image[%d]: %v", k, err)
			}
			lr, err := mrope.ForwardMRoPE(mRes.EmbedResidentForTest(tok), pos+k, pos+k+delta)
			if err != nil {
				t.Fatalf("resident ForwardMRoPE[%d]: %v", k, err)
			}
			worst(&r.pastImage, lr, lc)
		}
		return r
	}
	show := func(label string, r result) {
		t.Logf("%s: text decode 48 pos worst cos %.6f relL2 %.4f | text batched prefill cos %.6f relL2 %.4f | image m-RoPE prefill cos %.6f relL2 %.4f | decode past image (%d steps, ropePos=pos+mropeDelta) worst cos %.6f relL2 %.4f",
			label, r.decode.cos, r.decode.rel, r.prefill.cos, r.prefill.rel, r.imagePrefill.cos, r.imagePrefill.rel, g.NNew-1, r.pastImage.cos, r.pastImage.rel)
	}
	const cosBar, relBar = 0.995, 0.10

	got := measure()
	show("PAIRWISE kernels", got)
	for name, m := range map[string]metrics{"text decode": got.decode, "text batched prefill": got.prefill, "image m-RoPE prefill": got.imagePrefill, "decode past image": got.pastImage} {
		if m.cos < cosBar || m.rel > relBar {
			t.Errorf("%s diverges from the CPU: worst cosine %.6f (want >= %.3f), relL2 %.4f (want <= %.2f)", name, m.cos, cosBar, m.rel, relBar)
		}
	}

	restore := forceNeoXRope(t, cr)
	neox := measure()
	restore()
	show("NeoX control   ", neox)
	for name, m := range map[string]metrics{"text decode": neox.decode, "text batched prefill": neox.prefill, "image m-RoPE prefill": neox.imagePrefill, "decode past image": neox.pastImage} {
		if m.cos >= cosBar {
			t.Errorf("the gate is BLIND on %q: with the NeoX kernels forced it still reads cosine %.6f >= %.3f", name, m.cos, cosBar)
		}
	}
}

// TestGlmOcrResidentAdmissionCUDA pins the admission truth on CUDA: glm_ocr is resident (the pairwise
// kernels are declared) and carries no pairwise decline reason. The Metal/WebGPU half — still
// declined, naming the missing features — lives in decoder.TestGlmOcr_residentDeclined.
func TestGlmOcrResidentAdmissionCUDA(t *testing.T) {
	for _, ckpt := range []string{"../testdata/glm-ocr-tiny"} {
		requireDeviceAndFixture(t, ckpt)
		m, err := decoder.Load(ckpt, decoder.Options{Backend: "cuda", Quant: "int8int8"})
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		defer m.Close()
		if _, ok := m.ResidentForwardForTest().(*cudaResident); !ok {
			t.Fatalf("glm_ocr must be CUDA-resident now (pairwise kernels declared): %s", m.ResidentDecline())
		}
		if why := m.ResidentDecline(); strings.Contains(why, "pairwise") {
			t.Errorf("a resident model carries a pairwise decline reason: %q", why)
		}
	}
}
