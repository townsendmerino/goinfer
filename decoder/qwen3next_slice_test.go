//go:build realckpt

// REAL-WEIGHT layer-slice oracle for Qwen3-Next — the T3 the macbook's Phase 0/1 left
// open and tagged `linux`.
//
// WHY A SLICE. Qwen3-Next-80B-A3B is 163GB in bf16; a full reference forward does not fit
// in 62GB, so full-model T3 is not available on this box. A slice keeps the part that
// matters — REAL trained weights with their real routing distributions and real DeltaNet
// state dynamics — and drops only depth.
//
// WHY FOUR LAYERS, EXACTLY. `full_attention_interval: 4` makes layer i full-attention when
// (i+1)%4 == 0, so layers 0-2 are Gated DeltaNet and layer 3 is full attention. Four is the
// MINIMUM depth that spans both halves of the hybrid; three would gate only the linear path
// and would look like a passing T3 while testing half the model.
//
// It also only needs 6 of the checkpoint's 41 shards (~24GB rather than 163GB), because a
// leading slice touches just the first layers plus embeddings/norm/head.
//
// WHAT THIS COVERS THAT THE TINY GOLDEN CANNOT. The tiny fixture is random-init, where the
// 512-expert top-10 router is near-uniform and the DeltaNet recurrent state stays small.
// Real weights give peaked routing and real state magnitudes. It also exercises the
// checkpoint-layout delta the macbook flagged — the FUSED DeltaNet input projections
// (in_proj_qkvz / in_proj_ba) split at load — against tensors that actually ship fused.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags realckpt ./decoder/ -run TestQwen3NextSlice -v
package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"testing"
)

func TestQwen3NextSlice_realWeightOracle(t *testing.T) {
	requireHeavyModel(t)
	const golden = "testdata/qwen3next_q3next_slice_golden.json"
	const ckpt = "testdata/qwen3next-q3next-slice"

	raw, err := os.ReadFile(golden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no golden — run scripts/pin_slice_oracle.py (SLICE_TAG=q3next SLICE_PREFIX=qwen3next)")
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if _, err := os.Stat(filepath.Join(ckpt, "model.safetensors")); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no slice checkpoint at %s — run scripts/pin_slice_oracle.py", ckpt)
	}
	var g struct {
		PromptIDs       []int     `json:"prompt_ids"`
		Argmax          int       `json:"argmax"`
		LastLogits      []float32 `json:"last_logits"`
		NNew            int       `json:"n_new"`
		ContinuationIDs []int     `json:"continuation_ids"`
		NLayers         int       `json:"n_layers"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}

	m, err := Load(ckpt, Options{}) // f32: this is the NUMERIC gate, no quantization
	if err != nil {
		t.Fatalf("Load(%s): %v", ckpt, err)
	}
	defer m.Close()

	a := m.w.arch
	if a.Name != "qwen3_next" || a.NumLayers != g.NLayers {
		t.Fatalf("arch = %q with %d layers, want qwen3_next with %d", a.Name, a.NumLayers, g.NLayers)
	}
	// The hybrid split, synthesized from full_attention_interval rather than read from a
	// layer_types list. Layers 0-2 linear (Gated DeltaNet), layer 3 full attention.
	for i, wantLinear := range []bool{true, true, true, false} {
		if got := a.isLinearLayer(i); got != wantLinear {
			t.Fatalf("isLinearLayer(%d) = %v, want %v — full_attention_interval=4 puts the only "+
				"full-attention layer of this slice at index 3", i, got, wantLinear)
		}
	}
	if a.qwen35 == nil {
		t.Fatal("arch.qwen35 is nil — qwen3_next rides the qwen3_5_moe DeltaNet geometry")
	}
	if a.MoE == nil || a.MoE.NumExperts != 512 || a.MoE.TopK != 10 {
		t.Errorf("MoE = %v, want 512 experts / top-10", a.MoE)
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
	t.Logf("qwen3_next slice (REAL weights, %d layers: 3 DeltaNet + 1 full): argmax got=%d want=%d | cosine=%.8f",
		g.NLayers, gotArg, g.Argmax, cos)
	if gotArg != g.Argmax {
		t.Errorf("last argmax = %d, want %d", gotArg, g.Argmax)
	}
	if cos < 0.9999 {
		t.Errorf("last-logit cosine %.8f < 0.9999 on REAL weights", cos)
	}

	// Greedy continuation. For a recurrent family this is the part that matters most: a
	// DeltaNet state updated slightly wrongly still gives a near-perfect FIRST logit and
	// drifts only as the state accumulates, so a single-position compare is exactly the
	// check that would miss it.
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
			t.Errorf("continuation[%d] = %d, want %d — DeltaNet state drift shows here, not in "+
				"the first logit", i, nxt, g.ContinuationIDs[i])
			break
		}
		cur = append(cur, nxt)
	}
}
