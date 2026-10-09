//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// G-PR2 and G-PR3 (docs/tasks/task-metal-pairwise-rope-2026-10.md): the GPT-J pairwise families on the Metal resident
// against the CPU, with attention PEAKED so a wrong rotation shows. The committed tiny fixtures are ~0.02-std, where
// attention is nearly uniform and the NeoX kernels on a pairwise model read cosine 0.9997; CUDA's gate
// (cuda/pairwise_rope_resident_parity_test.go) scales every 2-D weight by 12.5 at test time, and so does this one.
//
// References, as registered: the CPU int4 at per-row activation scales for every DECODE path (Metal decode's own W4A8
// arithmetic), and the CPU int4 at per-32 scales for the BATCHED-PREFILL logits (the batched pass runs f16 activations,
// which Step 0 of docs/completed/task-metal-prefill-precision-2026-10.md measured as closer to that arm). Bars:
// cosine >= 0.995 and relL2 <= 0.15 on every path. Every run then rebinds the NeoX kernels into the same resident, and
// every path must read below the cosine bar, or the gate has gone blind.
//
// int4, not int8int8: an int8int8 tiny load is native int8 on Metal, whose prefill declines first.

const (
	pwPeaked = 12.5
	pwCosBar = 0.995
	pwRelBar = 0.15
)

// pwPeakedCheckpoint copies a tiny safetensors fixture with every 2-D F32 tensor scaled by factor (CUDA's
// peakedCheckpoint): the header and every offset stay as they are.
func pwPeakedCheckpoint(t *testing.T, src string, factor float32) string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(src, "model.safetensors"))
	if err != nil {
		t.Skipf("no fixture %s: %v", src, err)
	}
	n := binary.LittleEndian.Uint64(raw[:8])
	var hdr map[string]json.RawMessage
	if err := json.Unmarshal(raw[8:8+n], &hdr); err != nil {
		t.Fatalf("safetensors header: %v", err)
	}
	data := raw[8+n:]
	scaled := 0
	for name, h := range hdr {
		if name == "__metadata__" {
			continue
		}
		var e struct {
			Dtype   string   `json:"dtype"`
			Shape   []int    `json:"shape"`
			Offsets [2]int64 `json:"data_offsets"`
		}
		if err := json.Unmarshal(h, &e); err != nil {
			t.Fatalf("tensor %s: %v", name, err)
		}
		if len(e.Shape) != 2 {
			continue
		}
		if e.Dtype != "F32" {
			t.Fatalf("tensor %s is %s: the transform handles F32 only", name, e.Dtype)
		}
		for off := e.Offsets[0]; off < e.Offsets[1]; off += 4 {
			v := math.Float32frombits(binary.LittleEndian.Uint32(data[off:])) * factor
			binary.LittleEndian.PutUint32(data[off:], math.Float32bits(v))
		}
		scaled++
	}
	if scaled < 8 {
		t.Fatalf("scaled only %d tensors", scaled)
	}
	dst := t.TempDir()
	if err := os.WriteFile(filepath.Join(dst, "model.safetensors"), raw, 0o644); err != nil {
		t.Fatal(err)
	}
	for _, f := range []string{"config.json", "generation_config.json"} {
		if b, err := os.ReadFile(filepath.Join(src, f)); err == nil {
			if err := os.WriteFile(filepath.Join(dst, f), b, 0o644); err != nil {
				t.Fatal(err)
			}
		}
	}
	return dst
}

func pwRelL2(got, want []float32) float64 {
	var num, den float64
	for i := range got {
		d := float64(got[i]) - float64(want[i])
		num, den = num+d*d, den+float64(want[i])*float64(want[i])
	}
	return math.Sqrt(num / den)
}

type pwMetric struct{ cos, rel float64 }

func (m *pwMetric) add(got, want []float32) {
	m.cos = math.Min(m.cos, cosF(got, want))
	m.rel = math.Max(m.rel, pwRelL2(got, want))
}

func newPWMetric() pwMetric { return pwMetric{cos: 1} }

// pwLoad loads ckpt on Metal (int4) and twice on the CPU (int4 per-row, int4 per-32), and checks the pairwise twins are bound.
func pwLoad(t *testing.T, ckpt, quant string) (mRes, cpuRow, cpu32 *decoder.Model, a *metalResident) {
	t.Helper()
	var err error
	if mRes, err = decoder.Load(ckpt, decoder.Options{Backend: "metal", Quant: quant}); err != nil {
		t.Fatalf("load metal: %v", err)
	}
	t.Cleanup(func() { mRes.Close() })
	var ok bool
	if a, ok = mRes.ResidentForwardForTest().(*metalResident); !ok {
		t.Fatalf("not Metal-resident: %s", mRes.ResidentDecline())
	}
	if !a.r.pairwiseRoPE {
		t.Fatal("resident.pairwiseRoPE is false: the pairwise twins are not bound")
	}
	if cpuRow, err = decoder.Load(ckpt, decoder.Options{Backend: "cpu", Quant: quant}); err != nil {
		t.Fatalf("load cpu: %v", err)
	}
	t.Cleanup(func() { cpuRow.Close() })
	if cpu32, err = decoder.Load(ckpt, decoder.Options{Backend: "cpu", Quant: quant, ActQuantGroup: 32}); err != nil {
		t.Fatalf("load cpu per-32: %v", err)
	}
	t.Cleanup(func() { cpu32.Close() })
	return
}

// pwForceNeoX rebinds the NeoX rope pipelines (decode and batched prefill, the m-RoPE pass included) into a's resident and
// returns the restore. Test-only: the production binding stays in model.go and prefill.go.
func pwForceNeoX(t *testing.T, a *metalResident) func() {
	t.Helper()
	r := a.r
	r.ensurePrefill()
	mk := func(src, n string) Pipeline {
		lib, err := r.d.CompileLibrary(src, MSL3_1)
		if err != nil {
			t.Fatal(err)
		}
		p, err := r.d.NewComputePipeline(lib, n)
		if err != nil {
			t.Fatalf("pipeline %s: %v", n, err)
		}
		return p
	}
	old := [4]Pipeline{r.pRope, r.pRope2, r.pf.pRope, r.pf.pRopeM}
	r.stopExec()
	r.pRope, r.pRope2 = mk(allKernels, "rope"), mk(allKernels, "rope2")
	r.pf.pRope, r.pf.pRopeM = mk(prefillKernels, "rope_f16"), mk(prefillKernels, "rope_mrope_f16")
	return func() {
		r.stopExec()
		r.pRope, r.pRope2, r.pf.pRope, r.pf.pRopeM = old[0], old[1], old[2], old[3]
	}
}

func pwGrade(t *testing.T, label string, paths map[string]pwMetric, neox bool) {
	t.Helper()
	for name, m := range paths {
		t.Logf("%s %-30s worst cos %.6f relL2 %.4f", label, name, m.cos, m.rel)
		switch {
		case !neox && (m.cos < pwCosBar || m.rel > pwRelBar):
			t.Errorf("%s diverges from the CPU: worst cosine %.6f (want >= %.3f), relL2 %.4f (want <= %.2f)", name, m.cos, pwCosBar, m.rel, pwRelBar)
		case neox && m.cos >= pwCosBar:
			t.Errorf("the gate is BLIND on %q: with the NeoX kernels it still reads cosine %.6f >= %.3f", name, m.cos, pwCosBar)
		}
	}
}

// TestPairwiseRoPEResidentParityMetal (G-PR2): Cohere (cohere-tiny) and Cohere2 (cohere2-tiny: sliding window and a
// NoPE global layer), 40 tokens, three paths: sequential decode, batched prefill (last token), and decode at [32, 40)
// after a 32-row batched prefill (reads the K the batched kernel stored).
func TestPairwiseRoPEResidentParityMetal(t *testing.T) {
	const n, split = 40, 32
	for _, fx := range []string{"cohere-tiny", "cohere2-tiny"} {
		t.Run(fx, func(t *testing.T) {
			ckpt := pwPeakedCheckpoint(t, filepath.Join("..", "testdata", fx), pwPeaked)
			mRes, cpuRow, cpu32, a := pwLoad(t, ckpt, "int4")
			_, _, _, _, _, _, vocab := cpuRow.Dims()
			prompt := make([]int, n)
			for i := range prompt {
				prompt[i] = (i*37 + 3) % vocab
			}
			embs := make([][]float32, n)
			for i, tok := range prompt {
				embs[i] = mRes.EmbedResidentForTest(tok)
			}
			ref := func(m *decoder.Model) [][]float32 {
				c := m.NewCache(n)
				out := make([][]float32, n)
				for i, tok := range prompt {
					l, err := m.ForwardForTest(tok, c)
					if err != nil {
						t.Fatal(err)
					}
					out[i] = append([]float32(nil), l...)
				}
				return out
			}
			row := ref(cpuRow)
			last32, err := cpu32.PrefillLogitsForTest(context.Background(), prompt, cpu32.NewCache(n))
			if err != nil {
				t.Fatal(err)
			}
			last32 = append([]float32(nil), last32...)
			measure := func() map[string]pwMetric {
				dec, pre, mixed := newPWMetric(), newPWMetric(), newPWMetric()
				a.Reset()
				for i := range prompt {
					l, err := a.Forward(embs[i], i)
					if err != nil {
						t.Fatal(err)
					}
					dec.add(l, row[i])
				}
				a.Reset()
				if _, err := a.PrefillLast(context.Background(), embs, 0); err != nil {
					// Production prefills this family sequentially on Metal (the batched pass has no parallel-block FFN), so
					// the batched paths do not exist to grade; glm-ocr-tiny carries the batched and m-RoPE coverage.
					t.Logf("batched prefill declined, NOT graded: %v", err)
					return map[string]pwMetric{"decode, 40 positions": dec}
				}
				pre.add(func() []float32 { l, _ := a.PrefillLast(context.Background(), embs, 0); return l }(), last32)
				a.Reset()
				if _, err := a.PrefillLast(context.Background(), embs[:split], 0); err != nil {
					t.Fatalf("PrefillLast(%d): %v", split, err)
				}
				for i := split; i < n; i++ {
					l, err := a.Forward(embs[i], i)
					if err != nil {
						t.Fatal(err)
					}
					mixed.add(l, row[i])
				}
				return map[string]pwMetric{"decode, 40 positions": dec, "batched prefill, last token": pre, "decode after batched prefill": mixed}
			}
			pwGrade(t, "PAIRWISE", measure(), false)
			restore := pwForceNeoX(t, a)
			neox := measure()
			restore()
			pwGrade(t, "NeoX    ", neox, true)
		})
	}
}

// TestGlmOcrResidentParityMetal (G-PR3): glm-ocr-tiny's text decoder (pairwise rope over contiguous m-RoPE sections),
// unpeaked (see the load below).
// Paths: text decode (48 positions), text batched prefill, the image prompt of glm_ocr_tiny_mrope_golden.json through the
// resident m-RoPE prefill (S16's route), and decode past the image through ForwardMRoPE (rotation position != storage
// position), teacher-forced on the golden continuation.
func TestGlmOcrResidentParityMetal(t *testing.T) {
	raw, err := os.ReadFile("../testdata/glm_ocr_tiny_mrope_golden.json")
	if err != nil {
		t.Skipf("no m-RoPE golden: %v", err)
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
	// Unpeaked, as CUDA's glm_ocr gate runs it (amendment 2026-10-09 in the task doc): GLM-OCR-tiny's NeoX control already
	// reads -0.33 flat, and x12.5 makes its 48-wide attention near-argmax, which turns int8 activation rounding into
	// isolated flips (one position at 0.989 against >= 0.9999 elsewhere, where the CPU's own two quantizations disagree by
	// more) that say nothing about the rotation.
	ckpt := "../testdata/glm-ocr-tiny"
	// int8int8: glm-ocr-tiny's hidden size (48) is not a multiple of 32, which Metal's int4 needs; CUDA's gate runs it at
	// int8int8 too. On Metal that is the native int8 resident.
	mRes, cpuRow, cpu32, a := pwLoad(t, ckpt, "int8int8")
	rmp, ok := mRes.ResidentForwardForTest().(decoder.ResidentMRoPEPrefill)
	if !ok {
		t.Fatal("the Metal resident is not a decoder.ResidentMRoPEPrefill")
	}
	_, _, _, _, _, _, vocab := cpuRow.Dims()
	text := make([]int, 48)
	for i := range text {
		text[i] = (i*37 + 3) % vocab
	}
	embs := make([][]float32, len(text))
	for i, tok := range text {
		embs[i] = mRes.EmbedResidentForTest(tok)
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
	// References: per-row CPU logits at every text position and past the image; per-32 CPU for both prefills' last rows.
	rowC := cpuRow.NewCache(len(text))
	row := make([][]float32, len(text))
	for i, tok := range text {
		l, err := cpuRow.ForwardForTest(tok, rowC)
		if err != nil {
			t.Fatal(err)
		}
		row[i] = append([]float32(nil), l...)
	}
	textLast32, err := cpu32.PrefillLogitsForTest(ctx, text, cpu32.NewCache(len(text)))
	if err != nil {
		t.Fatal(err)
	}
	textLast32 = append([]float32(nil), textLast32...)
	img32, err := cpu32.PrefillLogitsQwenVLForTest(ctx, g.PromptIDs, feats, g.ImageStart, g.NImage, mropePos, cpu32.NewCache(len(g.PromptIDs)))
	if err != nil {
		t.Fatal(err)
	}
	img32 = append([]float32(nil), img32...)
	imgC := cpuRow.NewCache(len(g.PromptIDs) + g.NNew)
	if _, err := cpuRow.PrefillLogitsQwenVLForTest(ctx, g.PromptIDs, feats, g.ImageStart, g.NImage, mropePos, imgC); err != nil {
		t.Fatal(err)
	}
	delta := imgC.MRopeDeltaForTest()
	if delta == 0 {
		t.Fatal("mropeDelta == 0: decode past the image would not exercise ropePos != pos")
	}
	var past [][]float32
	for k := 0; k < g.NNew-1; k++ {
		l, err := cpuRow.ForwardForTest(g.ContinuationIDs[k], imgC)
		if err != nil {
			t.Fatal(err)
		}
		past = append(past, append([]float32(nil), l...))
	}
	measure := func() map[string]pwMetric {
		out := map[string]pwMetric{}
		dec := newPWMetric()
		a.Reset()
		for i := range text {
			l, err := a.Forward(embs[i], i)
			if err != nil {
				t.Fatal(err)
			}
			dec.add(l, row[i])
		}
		out["text decode, 48 positions"] = dec
		a.Reset()
		if l, err := a.PrefillLast(ctx, embs, 0); err != nil {
			t.Logf("text batched prefill declined, NOT graded: %v", err)
		} else {
			pre := newPWMetric()
			pre.add(l, textLast32)
			out["text batched prefill"] = pre
		}
		// The image turn by the route production takes: the resident m-RoPE prefill (S16) when the batched pass runs for
		// this family, else the CPU prefill uploaded; then decode past the image through ForwardMRoPE.
		a.Reset()
		gpuPos := len(g.PromptIDs)
		if li, gp, err := mRes.ResidentMRoPEPrefillForTest(ctx, rmp, g.PromptIDs, feats, g.ImageStart, g.NImage, mropePos); err != nil {
			t.Logf("image m-RoPE prefill declined, NOT graded (the image turn takes the CPU prefill and the upload): %v", err)
			upC := cpuRow.NewCache(len(g.PromptIDs) + g.NNew)
			if _, err := cpuRow.PrefillLogitsQwenVLForTest(ctx, g.PromptIDs, feats, g.ImageStart, g.NImage, mropePos, upC); err != nil {
				t.Fatal(err)
			}
			a.Reset()
			if err := mRes.ResidentUploadPrefillForTest(upC); err != nil {
				t.Fatalf("upload the CPU image prefill: %v", err)
			}
		} else {
			img := newPWMetric()
			img.add(li, img32)
			out["image m-RoPE prefill"] = img
			gpuPos = gp
		}
		pst := newPWMetric()
		for k := 0; k < g.NNew-1; k++ {
			l, err := a.ForwardMRoPE(mRes.EmbedResidentForTest(g.ContinuationIDs[k]), gpuPos+k, gpuPos+k+delta)
			if err != nil {
				t.Fatal(err)
			}
			pst.add(l, past[k])
		}
		out["decode past the image"] = pst
		return out
	}
	pwGrade(t, "PAIRWISE", measure(), false)
	restore := pwForceNeoX(t, a)
	neox := measure()
	restore()
	pwGrade(t, "NeoX    ", neox, true)
}

// G-PR6 (structural): a NeoX family binds exactly the kernels it bound before the pairwise twins existed, and the prefill's
// kernel names map one to one. The twins' existence must not touch any other family's dispatch.
func TestPairwiseRoPE_neoxFamiliesUnchanged(t *testing.T) {
	for _, tc := range []struct {
		pairwise, mrope bool
		want            string
	}{{false, false, "rope_f16"}, {false, true, "rope_mrope_f16"}, {true, false, "rope_f16_pw"}, {true, true, "rope_mrope_f16_pw"}} {
		if got := ropeF16Name(tc.pairwise, tc.mrope); got != tc.want {
			t.Errorf("ropeF16Name(%v, %v) = %q, want %q", tc.pairwise, tc.mrope, got, tc.want)
		}
	}
	m, err := decoder.Load("../testdata/llama-tiny", decoder.Options{Backend: "metal", Quant: "int4"})
	if err != nil {
		t.Skipf("no NeoX tiny fixture: %v", err)
	}
	defer m.Close()
	a, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok {
		t.Skipf("llama-tiny not Metal-resident: %s", m.ResidentDecline())
	}
	if m.PairwiseRoPEResident() || a.r.pairwiseRoPE {
		t.Fatalf("a NeoX family (llama) reads pairwise: model %v, resident %v", m.PairwiseRoPEResident(), a.r.pairwiseRoPE)
	}
}
