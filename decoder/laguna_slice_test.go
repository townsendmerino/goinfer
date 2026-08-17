//go:build realckpt

// REAL-WEIGHT layer-slice oracle for Laguna-XS.2 — the strongest numeric gate this
// box can hold for a 33B model.
//
// WHY A SLICE, AND WHAT IT IS AND IS NOT. T3 means cosine/argmax against a full
// reference forward of a released checkpoint. Laguna-XS.2 is ~63GB in bf16, so a
// reference forward alongside goinfer's own copy does not fit in 62GB. A slice
// keeps the part that matters — REAL trained weights, with their real routing
// distributions, real softplus-gate magnitudes and real QK-norm scales — and drops
// only depth. It does NOT establish full-model parity, so the manifest row stays
// `experimental`; it is strictly stronger than the tiny goldens and strictly weaker
// than T3.
//
// It matters because random-init fixtures flatter this family specifically: a
// random router is near-uniform, so top-8-of-256 selection is barely exercised and
// e_score_correction_bias barely moves the choice, while a random g_proj puts
// softplus in a narrow band around log 2. Real weights produce peaked routing and a
// wide gate range, which is where a subtly wrong gate or a mis-ordered
// bias-vs-weight in the router would actually show.
//
// The slice is layers [0,4): layer 0 is the DENSE prefix and full_attention (48
// query heads); layers 1-3 are MoE and sliding_attention (64 heads). That is
// precisely the geometry that made per-layer query heads necessary, on real values.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags realckpt ./decoder/ -run TestLagunaSlice -v
package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"testing"
)

func TestLagunaSlice_realWeightOracle(t *testing.T) {
	requireHeavyModel(t)
	const golden = "testdata/laguna_xs2_slice_golden.json"
	const ckpt = "testdata/laguna-xs2-slice"

	raw, err := os.ReadFile(golden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no golden — run scripts/pin_laguna_slice.py")
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if _, err := os.Stat(filepath.Join(ckpt, "model.safetensors")); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no slice checkpoint at %s — run scripts/pin_laguna_slice.py", ckpt)
	}
	var g struct {
		PromptIDs       []int     `json:"prompt_ids"`
		Argmax          int       `json:"argmax"`
		LastLogits      []float32 `json:"last_logits"`
		NNew            int       `json:"n_new"`
		ContinuationIDs []int     `json:"continuation_ids"`
		NLayers         int       `json:"n_layers"`
		GProjRows       int       `json:"g_proj_rows_layer0"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}

	m, err := Load(ckpt, Options{}) // f32 weights: this is the NUMERIC gate, no quantization
	if err != nil {
		t.Fatalf("Load(%s): %v", ckpt, err)
	}
	defer m.Close()

	a := m.w.arch
	if a.Name != "laguna" || a.NumLayers != g.NLayers {
		t.Fatalf("arch = %q with %d layers, want laguna with %d", a.Name, a.NumLayers, g.NLayers)
	}
	// The slice preserves the real geometry: full layer 0 at 48 heads, sliding 1-3 at 64.
	if a.headsAt(0) != 48 || a.headsAt(1) != 64 {
		t.Fatalf("headsAt(0)/headsAt(1) = %d/%d, want 48/64", a.headsAt(0), a.headsAt(1))
	}
	if !a.isGlobalLayer(0) || a.isGlobalLayer(1) {
		t.Fatalf("layer types wrong: layer0 global=%v layer1 global=%v, want true/false",
			a.isGlobalLayer(0), a.isGlobalLayer(1))
	}
	if got := m.w.Layers[0].GProj.Rows(); got != g.GProjRows {
		t.Errorf("layer 0 g_proj rows = %d, want %d", got, g.GProjRows)
	}
	if a.FirstKDense != 1 || m.w.Layers[0].Router.Rows() != 0 {
		t.Errorf("FirstKDense=%d layer0 router rows=%d, want 1/0", a.FirstKDense, m.w.Layers[0].Router.Rows())
	}

	cache := m.NewCache(len(g.PromptIDs) + g.NNew)
	var logits []float32
	for _, id := range g.PromptIDs {
		if logits, err = m.forward(id, cache); err != nil {
			t.Fatalf("forward: %v", err)
		}
	}
	gotArg := argmax(logits)
	cos := logitCosine(logits, g.LastLogits)
	t.Logf("laguna slice (REAL weights, %d layers): argmax got=%d want=%d | logit cosine=%.8f",
		g.NLayers, gotArg, g.Argmax, cos)
	if gotArg != g.Argmax {
		t.Errorf("last argmax = %d, want %d", gotArg, g.Argmax)
	}
	if cos < 0.9999 {
		t.Errorf("last-logit cosine %.8f < 0.9999 on REAL weights", cos)
	}

	// Batched prefill against the same real-weight reference. The sequential/batched
	// split is where this family's worst bug lived (the gate was applied on one path
	// only, reading as a plausible 0.957 rather than as a failure), so both paths are
	// held to the real reference, not just to each other.
	if !m.canBatchN(len(g.PromptIDs)) {
		t.Error("canBatchN = false — laguna should use the batched prefill path")
	} else {
		bc := m.NewCache(len(g.PromptIDs) + g.NNew)
		bl, err := m.prefillLogits(g.PromptIDs, bc)
		if err != nil {
			t.Fatalf("prefillLogits: %v", err)
		}
		bcos := logitCosine(bl, g.LastLogits)
		t.Logf("laguna slice batched prefill: argmax=%d cosine=%.8f", argmax(bl), bcos)
		if argmax(bl) != g.Argmax {
			t.Errorf("batched-prefill argmax = %d, want %d", argmax(bl), g.Argmax)
		}
		if bcos < 0.9999 {
			t.Errorf("batched-prefill cosine %.8f < 0.9999 on REAL weights", bcos)
		}
	}

	// Greedy continuation on real weights — per-position errors (sliding-window start,
	// per-layer rotary width) that one position cannot reveal.
	cur := append([]int(nil), g.PromptIDs...)
	for i := range g.NNew {
		c2 := m.NewCache(len(cur) + 1)
		var lg []float32
		for _, id := range cur {
			if lg, err = m.forward(id, c2); err != nil {
				t.Fatalf("forward: %v", err)
			}
		}
		nxt := argmax(lg)
		if i < len(g.ContinuationIDs) && nxt != g.ContinuationIDs[i] {
			t.Errorf("continuation[%d] = %d, want %d", i, nxt, g.ContinuationIDs[i])
			break
		}
		cur = append(cur, nxt)
	}
}
