//go:build realckpt

package decoder

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/aikit/vision"
	"github.com/townsendmerino/goinfer/multimodal"
)

// TestQwen35VLReal_G2 is P8a's G2 (docs/measurements/p8a-qwen35-vl-2026-09/preregistration.md) on the
// real Qwen3.5-0.8B: three grid-aligned images through goinfer's OWN preprocessing, tower and
// decoder (GenerateQwenVL, the entry point serve uses), CPU f32 greedy, 32 tokens, against HF f32.
//
// Stage-isolated first, so a failure names its stage: pixel_values BIT-EXACT to the numpy
// transcription, image_features per-row cosine >= 0.9999 (G0's real bars govern the tower alone;
// this is the same tower on real image pixels), then the tokens. Bar: all tokens identical on all 3
// images. Pre-registered ambiguous band: a divergence at a step where HF's own top-1 - top-2 logit gap
// is < 0.02 is a NEAR-TIE and PARKS the gate (neither pass nor defect); a divergence with a gap >= 0.02
// is a FAIL. The ids fed in are HF's, so prompt construction is G4's business, not this test's.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags realckpt ./decoder/ -run TestQwen35VLReal_G2 -v -timeout 30m
func TestQwen35VLReal_G2(t *testing.T) {
	requireHeavyModel(t)
	qwen35VLRealG2(t, assetPath(t, "GOINFER_QWEN35VL_08B"), assetPath(t, "GOINFER_QWEN35VL_G2"))
}

// TestQwen35VLReal_G2_9B is the same gate on Qwen3.5-9B (tower 27x1152 -> 4096). A night-queue job:
// docs/measurements/p8a-qwen35-vl-2026-09/run-g2-9b.sh downloads the checkpoint at a pinned revision,
// writes the HF f32 references, then runs this. CPU f32 is ~36 GB, so the HF process must have exited.
func TestQwen35VLReal_G2_9B(t *testing.T) {
	requireHeavyModel(t)
	qwen35VLRealG2(t, assetPath(t, "GOINFER_QWEN35VL_9B"), assetPath(t, "GOINFER_QWEN35VL_G2_9B"))
}

func qwen35VLRealG2(t *testing.T, ckpt, dir string) {
	t.Helper()

	pp, err := multimodal.LoadQwen3PreprocessConfig(ckpt)
	if err != nil {
		t.Fatal(err)
	}
	enc, err := vision.LoadQwen3VisionEncoder(ckpt, false)
	if err != nil {
		t.Fatalf("tower: %v", err)
	}
	m, err := Load(ckpt, Options{}) // Quant "" = f32
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()
	if a := m.w.arch; a.Name != "qwen3_5" || len(a.MRopeSection) != 3 {
		t.Fatalf("arch %q mrope %v", a.Name, a.MRopeSection)
	}

	var parked, failed bool
	for _, k := range []string{"A", "B", "C"} {
		t.Run(k, func(t *testing.T) {
			var g struct {
				InputIDs      []int     `json:"input_ids"`
				ImageToken    int       `json:"image_token_id"`
				ImageStart    int       `json:"image_token_start"`
				NImageTokens  int       `json:"n_image_tokens"`
				GridTHW       [][3]int  `json:"grid_thw"`
				ImageFeatures []float32 `json:"image_features"`
				RopeDelta     int       `json:"rope_delta"`
				HFTokens      []int     `json:"hf_tokens"`
				HFGaps        []float64 `json:"hf_gaps"`
				HFText        string    `json:"hf_text"`
			}
			raw, err := os.ReadFile(filepath.Join(dir, "golden_"+k+".json"))
			if err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(raw, &g); err != nil {
				t.Fatal(err)
			}
			png, err := os.ReadFile(filepath.Join(dir, "img_"+k+".png"))
			if err != nil {
				t.Fatal(err)
			}
			wantPV := readF32LE(t, filepath.Join(dir, "img_"+k+".pv.f32"))

			// Stage: preprocessing, bit-exact.
			pv, grid, err := multimodal.QwenPreprocess(png, pp)
			if err != nil {
				t.Fatal(err)
			}
			if grid != g.GridTHW[0] || len(pv) != len(wantPV) {
				t.Fatalf("grid %v len %d, HF %v len %d", grid, len(pv), g.GridTHW[0], len(wantPV))
			}
			for i := range pv {
				if math.Float32bits(pv[i]) != math.Float32bits(wantPV[i]) {
					t.Fatalf("pixel_values[%d] differs in bits: %v vs %v", i, pv[i], wantPV[i])
				}
			}
			// Stage: tower on the real image.
			feats, err := enc.Forward(pv, [][3]int{grid})
			if err != nil {
				t.Fatal(err)
			}
			hidden := m.w.arch.HiddenDim
			if len(feats) != g.NImageTokens*hidden {
				t.Fatalf("tower emitted %d floats, want %d", len(feats), g.NImageTokens*hidden)
			}
			fc := rowCosMin(feats, g.ImageFeatures, hidden)
			t.Logf("image_features worst-row cosine %.9f", fc)
			if fc < 0.9999 {
				t.Fatalf("image_features worst-row cosine %.9f < 0.9999", fc)
			}
			// Stage: positions.
			pos, _ := mropePositions(g.InputIDs, g.ImageToken, g.GridTHW, pp.MergeSize)
			if d := mropeDelta(pos, len(g.InputIDs)); d != g.RopeDelta {
				t.Fatalf("mropeDelta %d, HF %d", d, g.RopeDelta)
			}

			// Stage: tokens, through GenerateQwenVL.
			stream, gen := m.GenerateQwenVL(context.Background(), g.InputIDs, g.ImageStart, g.NImageTokens, 0,
				func() ([]float32, error) { return feats, nil }, g.GridTHW, pp.MergeSize, g.ImageToken, len(g.HFTokens), SamplingParams{Temperature: 0})
			var got []int
			for id := range stream {
				got = append(got, id)
			}
			if err := gen.Err(); err != nil {
				t.Fatal(err)
			}
			div := -1
			for i := 0; i < max(len(got), len(g.HFTokens)); i++ {
				if i >= len(got) || i >= len(g.HFTokens) || got[i] != g.HFTokens[i] {
					div = i
					break
				}
			}
			if div < 0 {
				t.Logf("32-token identity: %d/%d tokens identical", len(got), len(g.HFTokens))
				return
			}
			gap := -1.0
			if div < len(g.HFGaps) {
				gap = g.HFGaps[div]
			}
			if gap >= 0 && gap < 0.02 {
				parked = true
				t.Errorf("PARKED near-tie: first divergence at step %d, HF top1-top2 gap %.4f < 0.02 (got %v want %v) — resolve by teacher forcing before calling it", div, gap, got[max(0, div-1):min(len(got), div+2)], g.HFTokens[max(0, div-1):min(len(g.HFTokens), div+2)])
			} else {
				failed = true
				t.Errorf("FAIL: first divergence at step %d, HF gap %.4f\n got  %v\n want %v", div, gap, got, g.HFTokens)
			}
		})
	}
	t.Logf("G2 verdict inputs: parked=%v failed=%v", parked, failed)
}

func readF32LE(t *testing.T, path string) []float32 {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(uint32(b[4*i]) | uint32(b[4*i+1])<<8 | uint32(b[4*i+2])<<16 | uint32(b[4*i+3])<<24)
	}
	return out
}
