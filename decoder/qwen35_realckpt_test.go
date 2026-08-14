//go:build realckpt

// Gate 1 (real Qwen3.6-35B-A3B slice f32 parity) is gated behind the `realckpt`
// build tag: it needs the 67 GB checkpoint (this box only) and its f32 slice
// loads are too heavy for the default 10-min `go test ./decoder/` budget, which
// already carries the Gemma4-12B parity test. Run it explicitly:
//
//	go test -tags realckpt ./decoder/ -run TestQwen35Real -v
//
// CI never has the checkpoint, so excluding it by tag (rather than self-skip)
// also keeps the default suite from compiling a test it can never run there.
package decoder

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// realQwen35Dir returns the on-disk path to the real Qwen3.6-35B-A3B checkpoint,
// overridable via GOINFER_QWEN35_REAL. Empty (skip the test) when neither the env
// var nor the default ~/models/qwen3.6-35b-a3b resolves to a directory.
func realQwen35Dir(t *testing.T) string {
	t.Helper()
	return assetPath(t, "GOINFER_QWEN35_REAL")
}

// loadQwen35Slice loads embed + the first nLayer real layers at f32 (quantNone)
// through goinfer's actual safetensors loader — the VL-prefix + fused-stacked-MoE
// path. Truncating NumLayers makes the loader request only layers 0..nLayer-1, so
// it never touches (or holds) the other 36 layers or the 67 GB of expert weights.
//
// Returns a free() that munmaps the checkpoint, closes the backend, and forces a
// GC. The f32 4-layer slice is ~17 GB resident; in the full suite (run after the
// Gemma4-12B test, several loads in a row) the mmaps must be released eagerly
// rather than waiting on finalizers, or the BF16→F32 widening of a 2.1 GB expert
// tensor faults under memory pressure. CALL free() before the next load.
func loadQwen35Slice(t *testing.T, dir string, nLayer int) (*Model, func()) {
	t.Helper()
	cfg, err := loadConfig(os.DirFS(dir), "config.json")
	if err != nil {
		t.Fatalf("loadConfig: %v", err)
	}
	if cfg.NumLayers < nLayer {
		t.Fatalf("checkpoint has %d layers, slice wants %d", cfg.NumLayers, nLayer)
	}
	cfg.NumLayers = nLayer // load only the slice
	if len(cfg.LayerTypes) >= nLayer {
		cfg.LayerTypes = cfg.LayerTypes[:nLayer] // validateQwen35 wants len==NumLayers
	}
	arch, schema, err := resolveArchitecture(cfg)
	if err != nil {
		t.Fatalf("resolveArchitecture: %v", err)
	}
	st, err := openCheckpointMmap(dir)
	if err != nil {
		t.Fatalf("openCheckpointMmap: %v", err)
	}
	w, err := buildWeightsFromSafetensors(cfg, arch, schema, st, quantNone, false, nil)
	if err != nil {
		_ = st.Close()
		t.Fatalf("buildWeightsFromSafetensors (loader on real tensors): %v", err)
	}
	be, _ := NewBackend("")
	m := &Model{w: w, be: be, eosIDs: w.Cfg.EOSIDs()}
	free := func() {
		_ = m.Close()
		_ = st.Close()
		runtime.GC()
	}
	return m, free
}

// TestQwen35Real_loaderSlice is the Gate-1 LOADER probe: it loads embed + layers
// 0–3 of the REAL Qwen3.6-35B-A3B checkpoint at f32 and runs goinfer's forward on
// that slice. Every loadF32/loadMat shape-validates against the real tensor, so a
// wrong name/prefix/expert-stacking fails loudly here — cheaply, without the full
// 35B or any HF reference. The slice deliberately spans both novel surfaces:
// DeltaNet linear layers (0,1,2) and the 256-expert fused-stacked MoE on every
// layer, plus the softmax layer (3). The numeric parity compare vs HF is the
// sibling test below (gated on a slice reference file).
func TestQwen35Real_loaderSlice(t *testing.T) {
	requireHeavyModel(t)
	dir := realQwen35Dir(t)
	m, free := loadQwen35Slice(t, dir, 4)
	defer free()

	// Loader sanity: the hybrid arch resolved, the slice has 4 layers, and the
	// per-layer-kind tensor sets landed where they should.
	if m.w.arch.qwen35 == nil {
		t.Fatalf("arch = %q, want qwen3_5_moe", m.w.arch.Name)
	}
	if got := len(m.w.Layers); got != 4 {
		t.Fatalf("loaded %d layers, want 4", got)
	}
	for l := 0; l < 3; l++ { // 0,1,2 are Gated DeltaNet (linear)
		if m.w.Layers[l].delta == nil {
			t.Errorf("layer %d: delta (linear-attn) weights not loaded", l)
		}
		if m.w.Layers[l].qattn != nil {
			t.Errorf("layer %d: unexpected softmax-attn weights on a linear layer", l)
		}
	}
	if m.w.Layers[3].qattn == nil { // 3 is full (softmax) attention
		t.Errorf("layer 3: softmax-attn weights not loaded")
	}
	if m.w.Layers[3].delta != nil {
		t.Errorf("layer 3: unexpected delta weights on a softmax layer")
	}
	// Fused-stacked MoE: all 256 experts de-stacked, each gate/up [moe_inter,hidden].
	moe := m.w.arch.MoE
	for _, l := range []int{0, 3} {
		exps := m.w.Layers[l].Experts
		if len(exps) != moe.NumExperts {
			t.Fatalf("layer %d: %d experts, want %d", l, len(exps), moe.NumExperts)
		}
		e0 := exps[0]
		if e0.Gate.Rows() != moe.IntermediateDim || e0.Gate.Cols() != m.w.arch.HiddenDim {
			t.Errorf("layer %d expert0 gate shape = [%d,%d], want [%d,%d]",
				l, e0.Gate.Rows(), e0.Gate.Cols(), moe.IntermediateDim, m.w.arch.HiddenDim)
		}
		if e0.Down.Rows() != m.w.arch.HiddenDim || e0.Down.Cols() != moe.IntermediateDim {
			t.Errorf("layer %d expert0 down shape = [%d,%d], want [%d,%d]",
				l, e0.Down.Rows(), e0.Down.Cols(), m.w.arch.HiddenDim, moe.IntermediateDim)
		}
		if m.w.Layers[l].SharedExpert.Gate.Rows() != moe.SharedIntermediateDim {
			t.Errorf("layer %d shared-expert gate rows = %d, want %d",
				l, m.w.Layers[l].SharedExpert.Gate.Rows(), moe.SharedIntermediateDim)
		}
	}

	// Run the slice forward over the fixed Gate-2 prompt00 ids ("The capital of
	// France is"). Prefill drives one token per call so the DeltaNet recurrence
	// sees every position; keep the last-position hidden after layer 3.
	promptIDs := []int{760, 6511, 314, 9338, 369}
	cache := m.NewCache(len(promptIDs))
	var h []float32
	var err error
	for i, id := range promptIDs {
		if h, err = m.runLayers(id, cache); err != nil {
			t.Fatalf("runLayers token %d (id %d): %v", i, id, err)
		}
	}
	if len(h) != m.w.arch.HiddenDim {
		t.Fatalf("hidden len %d, want %d", len(h), m.w.arch.HiddenDim)
	}
	var ss float64
	for _, x := range h {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			t.Fatalf("non-finite value in slice hidden state: %v", x)
		}
		ss += float64(x) * float64(x)
	}
	t.Logf("Gate-1 loader probe GREEN: 4-layer slice loaded + forwarded on real "+
		"checkpoint; post-layer-3 hidden ‖h‖=%.4f, h[:4]=%v", math.Sqrt(ss), h[:4])
}

// TestQwen35Real_gate1SliceParity is Gate 1 proper: bit-exact (cosine ≥ 1−1e-5)
// agreement between goinfer's f32 slice forward and an HF f32 reference on the
// SAME embed + layers 0–3 slice, asserted at EVERY layer boundary so a future
// regression localizes to the layer that breaks. The reference is produced by
// scripts/pin_qwen35_slice.py (HF, num_hidden_layers=4, f32, no offload) and
// banked at $GOINFER_QWEN35_SLICE_REF (default
// ~/models/qwen35_real_golden/slice4_f32.json): {prompt_ids, hidden_all} where
// hidden_all[k] is the last-position PRE-final-norm residual after layer k-1
// (k=0 → embeddings) — exactly what an N-layer slice's runLayers returns. (HF's
// output_hidden_states applies model.norm to its last entry; the script captures
// the pre-norm input instead, else the ‖1+w‖≈119 final RMSNorm gain would make
// the compare fail for the wrong reason.) Skipped until the file exists so the
// loader probe above can run on its own.
func TestQwen35Real_gate1SliceParity(t *testing.T) {
	requireHeavyModel(t)
	dir := realQwen35Dir(t)
	ref := os.Getenv("GOINFER_QWEN35_SLICE_REF")
	if ref == "" {
		home, _ := os.UserHomeDir()
		ref = filepath.Join(home, "models", "qwen35_real_golden", "slice4_f32.json")
	}
	raw, err := os.ReadFile(ref)
	if err != nil {
		t.Skipf("no slice reference at %s — run scripts/pin_qwen35_slice.py: %v", ref, err)
	}
	var g struct {
		PromptIDs []int       `json:"prompt_ids"`
		HiddenAll [][]float32 `json:"hidden_all"` // [k] = pre-norm residual after layer k-1
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse slice reference: %v", err)
	}

	// Each N-layer slice (N=1..4) reuses the loader; its final hidden must match
	// HF's hidden_all[N] bit-exact. Walking N up isolates a regression to the
	// first layer that diverges instead of only flagging the slice end.
	for n := 1; n <= 4; n++ {
		m, free := loadQwen35Slice(t, dir, n)
		cache := m.NewCache(len(g.PromptIDs))
		var h []float32
		for i, id := range g.PromptIDs {
			if h, err = m.runLayers(id, cache); err != nil {
				free()
				t.Fatalf("n=%d runLayers token %d: %v", n, i, err)
			}
		}
		free() // h is a fresh slice; release the ~17 GB slice before the next load
		want := g.HiddenAll[n]
		if len(h) != len(want) {
			t.Fatalf("n=%d hidden len mismatch: goinfer %d vs HF %d", n, len(h), len(want))
		}
		var dot, na, nb float64
		for i := range h {
			dot += float64(h[i]) * float64(want[i])
			na += float64(h[i]) * float64(h[i])
			nb += float64(want[i]) * float64(want[i])
		}
		cos := dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-12)
		t.Logf("after layer %d: cosine=%.10f (‖goinfer‖=%.4f ‖HF‖=%.4f)", n-1, cos, math.Sqrt(na), math.Sqrt(nb))
		if cos < 1-1e-5 {
			t.Errorf("after layer %d: cosine %.10f < 1−1e-5 — loader/forward diverges from HF on real tensors", n-1, cos)
		}
	}
}
