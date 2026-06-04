package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"os"
	"testing"
)

// TestYarn_matchesHF: computeInvFreq with a YaRN scaling must reproduce HF's
// _compute_yarn_parameters inv_freq id-for-id (and the plain table for the
// sliding layers), validated against the committed golden pinned from the HF
// source for Mellum2's rope params. Independent of the 12B checkpoint.
//
// Regenerate: .venv/bin/python scripts/pin_mellum_rope.py
func TestYarn_matchesHF(t *testing.T) {
	const path = "../testdata/mellum_rope_golden.json"
	raw, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no golden at %s — regenerate with scripts/pin_mellum_rope.py", path)
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var gg struct {
		Params map[string]float64 `json:"params"`
		Full   struct {
			AttentionFactor float64   `json:"attention_factor"`
			InvFreq         []float64 `json:"inv_freq"`
		} `json:"full"`
		Sliding struct {
			InvFreq []float64 `json:"inv_freq"`
		} `json:"sliding"`
	}
	if err := json.Unmarshal(raw, &gg); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	base := gg.Params["base"]
	dim := int(gg.Params["dim"])

	yarn, err := newYarnScaling(gg.Params["factor"], gg.Params["original_max_position_embeddings"], nil, nil, &gg.Full.AttentionFactor)
	if err != nil {
		t.Fatalf("newYarnScaling: %v", err)
	}
	if yarn.mscale != gg.Full.AttentionFactor {
		t.Errorf("mscale = %v, want %v", yarn.mscale, gg.Full.AttentionFactor)
	}

	gotYarn := computeInvFreq(base, dim, yarn)
	closeVec(t, "yarn inv_freq", gotYarn, gg.Full.InvFreq)

	gotPlain := computeInvFreq(base, dim, nil)
	closeVec(t, "plain inv_freq", gotPlain, gg.Sliding.InvFreq)
}

func closeVec(t *testing.T, name string, got, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len %d, want %d", name, len(got), len(want))
	}
	for i := range want {
		// Relative tolerance: the values span ~7 orders of magnitude.
		den := math.Abs(want[i])
		if den == 0 {
			den = 1
		}
		if rel := math.Abs(got[i]-want[i]) / den; rel > 1e-12 {
			t.Errorf("%s[%d] = %.17g, want %.17g (rel %.2e)", name, i, got[i], want[i], rel)
		}
	}
}
