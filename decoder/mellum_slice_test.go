//go:build realckpt

// REAL-WEIGHT layer-slice oracle for Mellum2 (JetBrains Mellum2-12B-A2.5B-Instruct) — the
// CPU half of G11.
//
// WHY IT EXISTS, AND WHY IT IS NOT A SYNTHETIC FIXTURE. G10 declared FeatRopeMscale for
// Metal to unblock gpt-oss's YaRN, which as a documented side effect also admits Mellum
// onto the Metal resident path — a second GPU path for this family with ZERO end-to-end
// validation. G10's attempt to close that with a hand-rolled random-weight fixture failed
// for a reason worth not repeating: at realistic dims a plain dense qwen2 control with NO
// QK-norm at all also misses the 0.95 cosine bar against fully-random weights (0.898).
// Random weights lack the outlier structure real checkpoints have, so int4/int8 noise
// swamps whatever feature is under test. A synthetic fixture CANNOT discriminate a real
// bug from that noise floor.
//
// A real slice can. It keeps what matters — trained weights, with their real routing
// distributions, real QK-norm scales and real YaRN interaction — and drops only depth.
// Layers [0,4) is the coverage floor, not a size choice: Mellum2's real layer_types is a
// 3:1 sliding/full interleave, so layer 3 (0-indexed) is the FIRST full_attention layer
// and therefore the first one carrying YaRN. A 3-layer slice would gate the sliding path
// and silently skip the mscale that G10 actually changed. Verified against the released
// config before slicing rather than assumed.
//
// This is the DECODER-side gate: it proves goinfer's own CPU forward matches the HF f32
// reference on real weights. The Metal half (metal/mellum_real_test.go) runs residentParity
// against the SAME slice; see docs/queue-correctness.md G11 for the handoff, including how
// to regenerate this slice bit-identically (it is 4 GB, so the checkpoint is gitignored and
// only the golden is tracked).
//
//	GOINFER_HEAVY_TESTS=1 go test -tags realckpt ./decoder/ -run TestMellumSlice -v
package decoder

import (
	"context"
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestMellumSlice_realWeightOracle(t *testing.T) {
	requireHeavyModel(t)
	const golden = "testdata/mellum_mellum2_slice_golden.json"
	const ckpt = "testdata/mellum-mellum2-slice"

	raw, err := os.ReadFile(golden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no golden — see the regeneration recipe in docs/queue-correctness.md (G11)")
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if _, err := os.Stat(filepath.Join(ckpt, "model.safetensors")); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no slice checkpoint at %s (4 GB, gitignored) — regenerate per G11", ckpt)
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

	m, err := Load(ckpt, Options{}) // f32 weights: this is the NUMERIC gate, no quantization
	if err != nil {
		t.Fatalf("Load(%s): %v", ckpt, err)
	}
	defer m.Close()

	a := m.w.arch
	if a.Name != "mellum" || a.NumLayers != g.NLayers {
		t.Fatalf("arch = %q with %d layers, want mellum with %d", a.Name, a.NumLayers, g.NLayers)
	}
	// The real geometry survived the slice. head_dim 128 at hidden 2304 with 32 heads means
	// nH·hd = 4096 != hidden — assert it, because a slice rebuilt by a script that derived
	// head_dim from hidden would load and produce plausible garbage.
	if a.HiddenDim != 2304 || a.HeadDim != 128 || a.NumHeads != 32 || a.NumKVHeads != 4 {
		t.Errorf("geometry = h%d/hd%d/%dq/%dkv, want 2304/128/32/4",
			a.HiddenDim, a.HeadDim, a.NumHeads, a.NumKVHeads)
	}
	if a.NumHeads*a.HeadDim == a.HiddenDim {
		t.Error("nH·hd == hidden — the slice no longer exercises the independent-head_dim shape")
	}
	// THE COVERAGE CLAIM, asserted rather than trusted: 3 sliding then 1 full, with the
	// full layer last. If the truncation had dropped layer_types the loader would default
	// every layer to one kind and this slice would gate half of what it claims to.
	for i := range a.NumLayers {
		wantGlobal := i == 3
		if got := a.isGlobalLayer(i); got != wantGlobal {
			t.Fatalf("isGlobalLayer(%d) = %v, want %v — the 3:1 interleave did not survive slicing",
				i, got, wantGlobal)
		}
	}
	if a.SlidingWindow != 1024 {
		t.Errorf("SlidingWindow = %d, want 1024", a.SlidingWindow)
	}
	// YaRN mscale on the FULL layer and 1.0 on the sliding ones. This is the exact scalar
	// G10's FeatRopeMscale declaration is about, so the slice pins it on real weights before
	// Metal is allowed to claim it.
	const wantMscale = 1.2772588722239782
	if got := a.ropeMscale(3); math.Abs(got-wantMscale) > 1e-12 {
		t.Errorf("full-layer YaRN mscale = %.16f, want %.16f", got, wantMscale)
	}
	if got := a.ropeMscale(0); got != 1 {
		t.Errorf("sliding-layer mscale = %v, want 1 (plain RoPE)", got)
	}
	if a.MoE == nil || a.MoE.NumExperts != 64 || a.MoE.TopK != 8 {
		t.Fatalf("MoE = %v, want 64 experts top-8", a.MoE)
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
	t.Logf("mellum2 slice (REAL weights, %d layers): argmax got=%d want=%d | logit cosine=%.8f",
		g.NLayers, gotArg, g.Argmax, cos)
	if gotArg != g.Argmax {
		t.Errorf("last argmax = %d, want %d", gotArg, g.Argmax)
	}
	if cos < 0.9999 {
		t.Errorf("last-logit cosine %.8f < 0.9999 on REAL weights", cos)
	}

	// Batched prefill against the same real reference, not merely against the sequential
	// path. Laguna's worst bug was a gate applied on one path only, which read as a
	// plausible 0.957 rather than as a failure.
	if !m.canBatchN(len(g.PromptIDs)) {
		t.Error("canBatchN = false — mellum should use the batched prefill path")
	} else {
		bc := m.NewCache(len(g.PromptIDs) + g.NNew)
		bl, err := m.prefillLogits(context.Background(), g.PromptIDs, bc)
		if err != nil {
			t.Fatalf("prefillLogits: %v", err)
		}
		bcos := logitCosine(bl, g.LastLogits)
		t.Logf("mellum2 slice batched prefill: argmax=%d cosine=%.8f", argmax(bl), bcos)
		if argmax(bl) != g.Argmax {
			t.Errorf("batched-prefill argmax = %d, want %d", argmax(bl), g.Argmax)
		}
		if bcos < 0.9999 {
			t.Errorf("batched-prefill cosine %.8f < 0.9999 on REAL weights", bcos)
		}
	}

	// Greedy continuation: per-position errors (sliding-window start, per-layer-type RoPE
	// table selection) that a single position cannot reveal.
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
