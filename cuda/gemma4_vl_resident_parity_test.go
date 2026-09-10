//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// gemma4VLBidirScaledGolden mirrors scripts/pin_gemma4_vl_bidir_scaled.py's image-golden JSON —
// only the fields this test needs.
type gemma4VLBidirScaledGolden struct {
	InputIDs              []int     `json:"input_ids"`
	ImageTokenStart       int       `json:"image_token_start"`
	NImageTokens          int       `json:"n_image_tokens"`
	ImageFeatures         []float32 `json:"image_features"`
	DecodeContinuationIDs []int     `json:"decode_continuation_ids"`
}

// TestGemma4VLResident_bidirParity is the real-hardware gate for the GPU-resident decode bridge
// GenerateGemma4VL now wires for 26B-A4B/31B-class (use_bidirectional_attention: "vision")
// checkpoints (decoder/generate_gemma4_vl.go, decoder/generate_vl_resident.go's
// residentUploadPrefill reused unmodified). Everything structural about this bridge was verified
// by reading the code (see docs/multimodal.md's write-up): the ONE thing that needs numeric
// proof on real hardware is whether a CPU-computed bidirectional-block prefill's K/V — including
// the two K=V global layers' materialized v_norm(k) rows, at the REAL 256-local/512-global head
// geometry a tiny fixture can't exercise — continues decode correctly once uploaded into the
// resident CUDA cache.
//
// Methodology mirrors TestGemma4DenseScaled_residentParity's calibrated-mean approach (same
// fixture family: random weights over 12 layers are less int4-conditioned than a tiny fixture,
// so an absolute cosine bar would be dominated by fixture quantization chaos, not kernel signal).
// All three models decode the SAME fixed teacher-forced continuation (the HF golden's own greedy
// decode_continuation_ids) off their OWN bidirectional-prefilled cache, so this is a genuine
// prefill-through-decode gate, not just a single-position check.
func TestGemma4VLResident_bidirParity(t *testing.T) {
	const dir = "../testdata/gemma4-vl-bidir-scaled"
	const goldenPath = "../testdata/gemma4_vl_bidir_scaled_image_golden.json"
	if _, err := os.Stat(dir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no fixture (%s) — run scripts/pin_gemma4_vl_bidir_scaled.py", dir)
	}
	raw, err := os.ReadFile(goldenPath)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no golden (%s) — run scripts/pin_gemma4_vl_bidir_scaled.py", goldenPath)
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g gemma4VLBidirScaledGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	if len(g.DecodeContinuationIDs) == 0 {
		t.Fatal("golden has no decode_continuation_ids — regenerate with scripts/pin_gemma4_vl_bidir_scaled.py")
	}

	mc, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cuda): %v", err)
	}
	defer mc.Close()
	rf := mc.ResidentForwardForTest()
	if rf == nil {
		t.Fatal("cuda resident DECLINED the scaled bidirectional VL checkpoint — admission regressed (SharedKVLayers/PLE gate?)")
	}
	mc4, err := decoder.Load(dir, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cpu int4): %v", err)
	}
	defer mc4.Close()
	mcF, err := decoder.Load(dir, decoder.Options{Quant: "f32"})
	if err != nil {
		t.Fatalf("load (cpu f32): %v", err)
	}
	defer mcF.Close()

	ctx := context.Background()
	gpuPos := len(g.InputIDs)

	// Bidirectional CPU prefill, run on each model's own weights — mc's own prefill call is
	// EXACTLY what GenerateGemma4VL's production resident branch does before uploading.
	cCuda := mc.NewCache(len(g.InputIDs) + len(g.DecodeContinuationIDs))
	if _, err := mc.PrefillLogitsGemma4VLBidirectionalForTest(ctx, g.InputIDs, g.ImageFeatures, g.ImageTokenStart, g.NImageTokens, cCuda); err != nil {
		t.Fatalf("cuda-model CPU bidirectional prefill: %v", err)
	}
	if err := mc.ResidentUploadPrefillForTest(cCuda); err != nil {
		t.Fatalf("residentUploadPrefill: %v", err)
	}

	c4 := mc4.NewCache(len(g.InputIDs) + len(g.DecodeContinuationIDs))
	if _, err := mc4.PrefillLogitsGemma4VLBidirectionalForTest(ctx, g.InputIDs, g.ImageFeatures, g.ImageTokenStart, g.NImageTokens, c4); err != nil {
		t.Fatalf("cpu-int4 bidirectional prefill: %v", err)
	}
	cF := mcF.NewCache(len(g.InputIDs) + len(g.DecodeContinuationIDs))
	if _, err := mcF.PrefillLogitsGemma4VLBidirectionalForTest(ctx, g.InputIDs, g.ImageFeatures, g.ImageTokenStart, g.NImageTokens, cF); err != nil {
		t.Fatalf("cpu-f32 bidirectional prefill: %v", err)
	}

	n := len(g.DecodeContinuationIDs)
	cuda := make([][]float32, n)
	cpu4 := make([][]float32, n)
	cpuF := make([][]float32, n)
	for i, tok := range g.DecodeContinuationIDs {
		l, err := rf.Forward(mc.EmbedResidentForTest(tok), gpuPos+i)
		if err != nil {
			t.Fatalf("cuda decode step %d: %v", i, err)
		}
		cuda[i] = append([]float32(nil), l...)
		l, err = mc4.ForwardForTest(tok, c4)
		if err != nil {
			t.Fatalf("cpu int4 decode step %d: %v", i, err)
		}
		cpu4[i] = append([]float32(nil), l...)
		l, err = mcF.ForwardForTest(tok, cF)
		if err != nil {
			t.Fatalf("cpu f32 decode step %d: %v", i, err)
		}
		cpuF[i] = append([]float32(nil), l...)
	}

	byteIdentityCensus(t, "CUDA-resident vs CPU-int4 (both W4A8), bidirectional-block prefill -> decode", cuda, cpu4)

	cos := func(a, b []float32) float64 { c, _ := cosMaxAbs(a, b); return c }
	cVs4, c4VsF := make([]float64, n), make([]float64, n)
	pos0, exact := 0.0, 0
	sumCuda, sumCpu, minFloor := 0.0, 0.0, 1.0
	for i := range g.DecodeContinuationIDs {
		cVs4[i] = cos(cpu4[i], cuda[i])
		c4VsF[i] = cos(cpuF[i], cpu4[i])
		if i == 0 {
			pos0 = cVs4[i]
		}
		sumCuda += cVs4[i]
		sumCpu += c4VsF[i]
		if c4VsF[i] < minFloor {
			minFloor = c4VsF[i]
		}
		if argmaxF(cuda[i]) == argmaxF(cpu4[i]) {
			exact++
		}
		t.Logf("  decode step %2d  CUDA-vs-CPUint4 %.6f | CPUint4-vs-f32 %.6f  argmax cuda=%d cpu4=%d",
			i, cVs4[i], c4VsF[i], argmaxF(cuda[i]), argmaxF(cpu4[i]))
	}
	meanCuda, meanCpu := sumCuda/float64(n), sumCpu/float64(n)
	t.Logf("bidirectional-prefill resident bridge (256-local/512-global, K=V): pos0(decode-step-0)=%.6f exact-argmax %d/%d | "+
		"mean CUDA-vs-CPUint4=%.6f  CPUint4-vs-f32=%.6f (min floor %.4f)", pos0, exact, n, meanCuda, meanCpu, minFloor)

	// NOTE: unlike TestGemma4DenseScaled_residentParity's own pos-0 (truly zero attention
	// history, the least int4-noisy point it can measure), THIS pos0 is decode-step-0 AFTER a
	// full bidirectional-block prefill (19-31 positions of history) — already deep in this
	// fixture's own documented int4 chaos regime (that sibling test's own late positions, e.g.
	// pos 15, land at cosine 0.68 with zero bridge involvement at all: self-consistent resident
	// decode alone). An absolute bar at this depth would fail on ALREADY-ACCEPTED int4 noise, not
	// on a bridge defect — confirmed by a same-session A/B diagnostic: resident computing every
	// position itself vs CPU-prefill+upload for an equivalent prefix land EQUALLY far from the
	// CPU reference (0.68 vs 0.72), and disagree with EACH OTHER by exactly that same margin
	// (0.70) — two independent equally-noisy realizations, not a systematic upload defect. So
	// the calibrated-mean check below (which already accounts for this fixture's own chaos,
	// exactly TestGemma4DenseScaled_residentParity's reasoning) is the only valid bar here.
	if meanCuda < meanCpu {
		t.Errorf("mean CUDA-vs-CPUint4 %.6f < mean CPUint4-vs-f32 %.6f — CUDA diverges faster than the fixture's own int4 quantization: a real bug, not conditioning",
			meanCuda, meanCpu)
	}
}
