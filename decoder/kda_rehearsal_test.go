package decoder

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

// TestKDARehearsal_matchesReference is F4's synthetic-tiny bring-up (docs/task-families-2026-09.md):
// per-layer HF differencing for the one new primitive in Ling-3.0-tiny's KDA mixer, against
// fla-org/flash-linear-attention's actual reference implementation (not a from-scratch
// reimplementation compared to itself).
//
// Regenerate: python3 scripts/pin_kda_rehearsal.py -> testdata/kda_rehearsal_golden.json
func TestKDARehearsal_matchesReference(t *testing.T) {
	raw, err := os.ReadFile("../testdata/kda_rehearsal_golden.json")
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g struct {
		Shape      struct{ B, T, H, K, V int } `json:"shape"`
		LowerBound float32                     `json:"lower_bound"`
		Q          [][][][]float32             `json:"q"` // [B][T][H][K]
		K          [][][][]float32             `json:"k"` // [B][T][H][K]
		V          [][][][]float32             `json:"v"` // [B][T][H][V]
		RawGateIn  [][][][]float32             `json:"raw_gate_in"`
		BetaLogits [][][]float32               `json:"beta_logits"` // [B][T][H]
		ALog       []float32                   `json:"A_log"`
		DtBias     [][]float32                 `json:"dt_bias"` // [H][K]
		Gate       [][][][]float32             `json:"gate"`    // [B][T][H][K] — cross-checked, not just trusted
		Output     [][][][]float32             `json:"output"`  // [B][T][H][V]
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	if g.Shape.B != 1 {
		t.Fatalf("rehearsal fixture must be batch=1, got %d", g.Shape.B)
	}
	T, H, K, V := g.Shape.T, g.Shape.H, g.Shape.K, g.Shape.V
	qScale := float32(1 / math.Sqrt(float64(K)))

	// Cross-check kdaLowerBoundGate against the golden's own "gate" field BEFORE using it to
	// drive the recurrence, so a gate bug and a recurrence bug can't cancel out and hide behind
	// one passing final-output comparison.
	for h := 0; h < H; h++ {
		for ti := 0; ti < T; ti++ {
			got := kdaLowerBoundGate(g.RawGateIn[0][ti][h], g.DtBias[h], g.ALog[h], g.LowerBound)
			want := g.Gate[0][ti][h]
			for i := range got {
				if d := math.Abs(float64(got[i] - want[i])); d > 1e-5 {
					t.Fatalf("gate[t=%d,h=%d,%d] = %v, want %v (Δ%.2e)", ti, h, i, got[i], want[i], d)
				}
			}
		}
	}

	S := make([][]float32, H)
	for h := range S {
		S[h] = make([]float32, K*V)
	}
	var maxAbsDiff float64
	var sumDot, sumGot, sumWant float64
	for ti := 0; ti < T; ti++ {
		for h := 0; h < H; h++ {
			q := l2normScaled(g.Q[0][ti][h], qScale)
			k := l2normScaled(g.K[0][ti][h], 1)
			v := g.V[0][ti][h]
			gateLog := kdaLowerBoundGate(g.RawGateIn[0][ti][h], g.DtBias[h], g.ALog[h], g.LowerBound)
			decay := make([]float32, K)
			for i, x := range gateLog {
				decay[i] = float32(math.Exp(float64(x)))
			}
			beta := sigmoidf(g.BetaLogits[0][ti][h])

			out := kdaRecurrentStep(q, k, v, decay, beta, S[h])
			want := g.Output[0][ti][h]
			if ti == T-1 {
				t.Logf("t=%d h=%d out=%v want=%v", ti, h, out, want)
			}
			for i := range out {
				d := math.Abs(float64(out[i]) - float64(want[i]))
				if d > maxAbsDiff {
					maxAbsDiff = d
				}
				sumDot += float64(out[i]) * float64(want[i])
				sumGot += float64(out[i]) * float64(out[i])
				sumWant += float64(want[i]) * float64(want[i])
			}
		}
	}
	cos := sumDot / (math.Sqrt(sumGot)*math.Sqrt(sumWant) + 1e-12)
	t.Logf("KDA rehearsal: %d timesteps × %d heads | maxAbsDiff=%.3e | cosine=%.8f", T, H, maxAbsDiff, cos)
	if maxAbsDiff > 1e-4 {
		t.Errorf("maxAbsDiff = %.3e, want < 1e-4", maxAbsDiff)
	}
	if cos < 0.999999 {
		t.Errorf("cosine = %.8f, want ≥ 0.999999", cos)
	}
}
