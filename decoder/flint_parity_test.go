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

	"github.com/townsendmerino/aikit/embed"
)

type flintGolden struct {
	PromptIDs       []int     `json:"prompt_ids"`
	Argmax          int       `json:"argmax"`
	LastLogits      []float32 `json:"last_logits"`
	NNew            int       `json:"n_new"`
	ContinuationIDs []int     `json:"continuation_ids"`
	Window513       struct {
		SlidingWindow int `json:"sliding_window"`
		PromptLen     int `json:"prompt_len"`
		Checkpoints   []struct {
			Len    int       `json:"len"`
			Pos    int       `json:"pos"`
			Argmax int       `json:"argmax"`
			Logits []float32 `json:"logits"`
		} `json:"checkpoints"`
	} `json:"window_513"`
}

func loadFlintGolden(t *testing.T) flintGolden {
	t.Helper()
	raw, err := os.ReadFile("../testdata/flint_tiny_golden.json")
	if errors.Is(err, fs.ErrNotExist) {
		t.Skip("no flint golden — run python3 scripts/pin_flint_tiny.py")
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g flintGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	return g
}

func maxAbsLogitDiff(a, b []float32) float64 {
	var m float64
	for i := range a {
		if d := math.Abs(float64(a[i]) - float64(b[i])); d > m {
			m = d
		}
	}
	return m
}

// TestFlint_tinyForwardParity is F2.1 (docs/tasks/task-flint-2026-10.md): verifies that
// goinfer's flint descriptor and forward match the upstream transformers PR #49824
// FlintForCausalLM oracle on testdata/flint-tiny (2 dense full/NoPE lead layers + 3 MoE
// sliding/RoPE layers + 1 MoE full/NoPE final layer, top-2 sigmoid normalized router
// without bias, ungated shared expert, per-head QK RMSNorm, and learned attention sinks).
func TestFlint_tinyForwardParity(t *testing.T) {
	g := loadFlintGolden(t)
	const ckpt = "../testdata/flint-tiny"
	if _, err := os.Stat(filepath.Join(ckpt, "model.safetensors")); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no checkpoint at %s — run python3 scripts/pin_flint_tiny.py", ckpt)
	}

	m, err := Load(ckpt, Options{Quant: "f32"})
	if err != nil {
		t.Fatalf("Load(%s): %v", ckpt, err)
	}
	defer m.Close()

	a := m.w.arch
	if a.Name != "flint" {
		t.Fatalf("arch.Name = %q, want flint", a.Name)
	}
	if a.FirstKDense != 2 {
		t.Fatalf("FirstKDense = %d, want 2", a.FirstKDense)
	}
	if !a.AttnSinks {
		t.Fatal("arch.AttnSinks = false, want true")
	}
	if !a.QKNorm {
		t.Fatal("arch.QKNorm = false, want true")
	}
	if a.MoE == nil || !a.MoE.RouterSigmoid || !a.MoE.SharedUngated {
		t.Fatalf("MoE = %+v, want RouterSigmoid=true SharedUngated=true", a.MoE)
	}
	wantGlobal := []bool{true, true, false, false, false, true}
	for l := range a.NumLayers {
		if got := a.isGlobalLayer(l); got != wantGlobal[l] {
			t.Errorf("layer %d isGlobalLayer = %v, want %v", l, got, wantGlobal[l])
		}
		if got := a.isNoPELayer(l); got != wantGlobal[l] {
			t.Errorf("layer %d isNoPELayer = %v, want %v", l, got, wantGlobal[l])
		}
		if len(m.w.Layers[l].AttnSinks) != a.NumHeads {
			t.Errorf("layer %d len(AttnSinks) = %d, want %d", l, len(m.w.Layers[l].AttnSinks), a.NumHeads)
		}
		if len(m.w.Layers[l].QNorm) != a.HeadDim || len(m.w.Layers[l].KNorm) != a.HeadDim {
			t.Errorf("layer %d len(QNorm,KNorm) = (%d,%d), want (%d,%d)", l, len(m.w.Layers[l].QNorm), len(m.w.Layers[l].KNorm), a.HeadDim, a.HeadDim)
		}
		if m.w.Layers[l].RouterBias != nil {
			t.Errorf("layer %d RouterBias = %v, want nil", l, m.w.Layers[l].RouterBias)
		}
		hasExp := m.w.Layers[l].Experts != nil
		wantExp := l >= a.FirstKDense
		if hasExp != wantExp {
			t.Errorf("layer %d hasExperts = %v, want %v", l, hasExp, wantExp)
		}
	}

	// 1. Sequential per-token decode.
	cache := m.NewCache(len(g.PromptIDs) + g.NNew)
	var logits []float32
	for _, id := range g.PromptIDs {
		if logits, err = m.forward(id, cache); err != nil {
			t.Fatalf("forward: %v", err)
		}
	}
	gotArg := argmax(logits)
	cos := logitCosine(logits, g.LastLogits)
	maxAbs := maxAbsLogitDiff(logits, g.LastLogits)
	t.Logf("flint-tiny sequential parity: argmax=%d (want %d) cosine=%.8f maxAbs=%.3e",
		gotArg, g.Argmax, cos, maxAbs)
	if gotArg != g.Argmax {
		t.Errorf("argmax = %d, want %d", gotArg, g.Argmax)
	}
	if cos < 0.999999 {
		t.Errorf("cosine = %.8f < 0.999999", cos)
	}
	if maxAbs > 1e-5 {
		t.Errorf("maxAbs = %.3e > 1e-5", maxAbs)
	}

	// 2. Batched prefill (prefillLogits / forwardN).
	if !m.canBatchN(len(g.PromptIDs)) {
		t.Fatalf("canBatchN(%d) = false, want true", len(g.PromptIDs))
	}
	bc := m.NewCache(len(g.PromptIDs) + g.NNew)
	bl, err := m.prefillLogits(context.Background(), g.PromptIDs, bc)
	if err != nil {
		t.Fatalf("prefillLogits: %v", err)
	}
	bMaxAbs := maxAbsLogitDiff(bl, g.LastLogits)
	bCos := logitCosine(bl, g.LastLogits)
	t.Logf("flint-tiny batched prefill parity: argmax=%d cosine=%.8f maxAbs=%.3e",
		argmax(bl), bCos, bMaxAbs)
	if argmax(bl) != g.Argmax {
		t.Errorf("batched argmax = %d, want %d", argmax(bl), g.Argmax)
	}
	if bMaxAbs > 1e-5 {
		t.Errorf("batched maxAbs = %.3e > 1e-5", bMaxAbs)
	}

	// 3. Greedy continuation past the sliding window.
	cont := make([]int, 0, g.NNew)
	nextLogits := append([]float32(nil), logits...)
	for range g.NNew {
		id := argmax(nextLogits)
		cont = append(cont, id)
		if nextLogits, err = m.forward(id, cache); err != nil {
			t.Fatalf("continuation forward: %v", err)
		}
	}
	for i := range g.ContinuationIDs {
		if cont[i] != g.ContinuationIDs[i] {
			t.Fatalf("greedy continuation diverges at step %d: got %v, want %v", i, cont, g.ContinuationIDs)
		}
	}

	// 4. .giw serialization round-trip preserves AttnSinks, layerNoPE, FirstKDense, and logits.
	giwBlob, err := SerializeWeights(m.w, "flint-tiny")
	if err != nil {
		t.Fatalf("SerializeWeights: %v", err)
	}
	wGiw, err := LoadSerializedWeights(giwBlob)
	if err != nil {
		t.Fatalf("LoadSerializedWeights: %v", err)
	}
	mGiw, err := NewModel(wGiw, "cpu")
	if err != nil {
		t.Fatalf("NewModel(.giw): %v", err)
	}
	defer mGiw.Close()
	gc := mGiw.NewCache(len(g.PromptIDs))
	gl, err := mGiw.prefillLogits(context.Background(), g.PromptIDs, gc)
	if err != nil {
		t.Fatalf("giw prefillLogits: %v", err)
	}
	if diff := maxAbsLogitDiff(gl, bl); diff > 1e-6 {
		t.Errorf(".giw round-trip maxAbs diff vs safetensors = %.3e > 1e-6", diff)
	}
}

// TestFlint_windowBoundaryParity is F2.2 (docs/tasks/task-flint-2026-10.md): verifies
// position-by-position parity against transformers FlintForCausalLM across the real
// sliding_window=513 boundary (prompts of 512, 513, 514, and 1,100 tokens, plus
// intermediate positions across both the first and second ring wraps).
func TestFlint_windowBoundaryParity(t *testing.T) {
	g := loadFlintGolden(t)
	if g.Window513.SlidingWindow != 513 || len(g.Window513.Checkpoints) == 0 {
		t.Fatal("golden missing window_513 checkpoints — re-run python3 scripts/pin_flint_tiny.py")
	}

	const srcDir = "../testdata/flint-tiny"
	if _, err := os.Stat(filepath.Join(srcDir, "model.safetensors")); errors.Is(err, fs.ErrNotExist) {
		t.Skip("no flint-tiny checkpoint")
	}

	// Create a temporary checkpoint directory with sliding_window=513 sharing flint-tiny's weights.
	tmpDir := t.TempDir()
	rawCfg, err := os.ReadFile(filepath.Join(srcDir, "config.json"))
	if err != nil {
		t.Fatalf("read config.json: %v", err)
	}
	var cfgMap map[string]any
	if err := json.Unmarshal(rawCfg, &cfgMap); err != nil {
		t.Fatalf("parse config.json: %v", err)
	}
	cfgMap["sliding_window"] = g.Window513.SlidingWindow
	patchedCfg, err := json.Marshal(cfgMap)
	if err != nil {
		t.Fatalf("marshal config.json: %v", err)
	}
	if err := os.WriteFile(filepath.Join(tmpDir, "config.json"), patchedCfg, 0o644); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	absWeights, err := filepath.Abs(filepath.Join(srcDir, "model.safetensors"))
	if err != nil {
		t.Fatalf("abs weights: %v", err)
	}
	if err := os.Symlink(absWeights, filepath.Join(tmpDir, "model.safetensors")); err != nil {
		t.Fatalf("symlink weights: %v", err)
	}

	m, err := Load(tmpDir, Options{Quant: "f32"})
	if err != nil {
		t.Fatalf("Load(sliding_window=513): %v", err)
	}
	defer m.Close()
	if m.w.arch.SlidingWindow != 513 {
		t.Fatalf("SlidingWindow = %d, want 513", m.w.arch.SlidingWindow)
	}

	ids := make([]int, g.Window513.PromptLen)
	for i := range ids {
		ids[i] = (i*31 + 7) % m.w.arch.VocabSize
	}

	byPos := map[int]struct {
		Len    int
		Argmax int
		Logits []float32
	}{}
	for _, cp := range g.Window513.Checkpoints {
		byPos[cp.Pos] = struct {
			Len    int
			Argmax int
			Logits []float32
		}{Len: cp.Len, Argmax: cp.Argmax, Logits: cp.Logits}
	}

	// 1. Sequential decode across all 1,100 tokens, checking every recorded position.
	cache := m.NewCache(len(ids))
	for pos, id := range ids {
		lg, err := m.forward(id, cache)
		if err != nil {
			t.Fatalf("forward at pos %d: %v", pos, err)
		}
		want, ok := byPos[pos]
		if !ok {
			continue
		}
		gotArg := argmax(lg)
		cos := logitCosine(lg, want.Logits)
		maxAbs := maxAbsLogitDiff(lg, want.Logits)
		if want.Len == 512 || want.Len == 513 || want.Len == 514 || want.Len == 1100 {
			t.Logf("window_513 sequential len=%d (pos=%d): argmax=%d (want %d) cosine=%.8f maxAbs=%.3e",
				want.Len, pos, gotArg, want.Argmax, cos, maxAbs)
		}
		if gotArg != want.Argmax {
			t.Errorf("sequential len=%d (pos=%d): argmax = %d, want %d", want.Len, pos, gotArg, want.Argmax)
		}
		if cos < 0.999999 {
			t.Errorf("sequential len=%d (pos=%d): cosine = %.8f < 0.999999", want.Len, pos, cos)
		}
		if maxAbs > 1e-5 {
			t.Errorf("sequential len=%d (pos=%d): maxAbs = %.3e > 1e-5", want.Len, pos, maxAbs)
		}
	}

	// 2. Batched prefill at the required window boundary lengths: 512, 513, 514, and 1,100.
	for _, length := range []int{512, 513, 514, 1100} {
		pos := length - 1
		want := byPos[pos]
		bc := m.NewCache(length)
		bl, err := m.prefillLogits(context.Background(), ids[:length], bc)
		if err != nil {
			t.Fatalf("prefillLogits(len=%d): %v", length, err)
		}
		gotArg := argmax(bl)
		cos := logitCosine(bl, want.Logits)
		maxAbs := maxAbsLogitDiff(bl, want.Logits)
		t.Logf("window_513 batched len=%d (pos=%d): argmax=%d (want %d) cosine=%.8f maxAbs=%.3e",
			length, pos, gotArg, want.Argmax, cos, maxAbs)
		if gotArg != want.Argmax {
			t.Errorf("batched len=%d: argmax = %d, want %d", length, gotArg, want.Argmax)
		}
		if cos < 0.999999 {
			t.Errorf("batched len=%d: cosine = %.8f < 0.999999", length, cos)
		}
		if maxAbs > 1e-5 {
			t.Errorf("batched len=%d: maxAbs = %.3e > 1e-5", length, maxAbs)
		}
	}
}

// TestFlint_breakItFirst is F2.4 (docs/tasks/task-flint-2026-10.md): perturbs the
// learned attention sinks, the per-layer NoPE mask, and the sliding-window boundary,
// confirming that the parity gate catches each perturbation loudly.
func TestFlint_breakItFirst(t *testing.T) {
	g := loadFlintGolden(t)
	const ckpt = "../testdata/flint-tiny"
	if _, err := os.Stat(filepath.Join(ckpt, "model.safetensors")); errors.Is(err, fs.ErrNotExist) {
		t.Skip("no flint-tiny checkpoint")
	}

	t.Run("drop_attention_sinks", func(t *testing.T) {
		m, err := Load(ckpt, Options{Quant: "f32"})
		if err != nil {
			t.Fatalf("Load: %v", err)
		}
		defer m.Close()

		// Perturb: strip learned attention sinks from all layers.
		m.w.arch.AttnSinks = false
		for l := range m.w.Layers {
			m.w.Layers[l].AttnSinks = nil
		}
		m.rebuildProgramForTest()
		c := m.NewCache(len(g.PromptIDs))
		lg, err := m.prefillLogits(context.Background(), g.PromptIDs, c)
		if err != nil {
			t.Fatalf("prefillLogits: %v", err)
		}
		maxAbs := maxAbsLogitDiff(lg, g.LastLogits)
		cos := logitCosine(lg, g.LastLogits)
		t.Logf("break-it-first (drop sinks): maxAbs=%.4f cosine=%.6f", maxAbs, cos)
		if maxAbs < 1e-2 {
			t.Fatalf("dropping attention sinks barely moved logits (maxAbs=%.3e < 1e-2) — gate would not catch missing sinks", maxAbs)
		}
	})

	t.Run("disable_nope_on_full_layers", func(t *testing.T) {
		m, err := Load(ckpt, Options{Quant: "f32"})
		if err != nil {
			t.Fatalf("Load: %v", err)
		}
		defer m.Close()

		// Perturb: apply RoPE on all layers (clearing layerNoPE on full-attention layers).
		m.w.arch.layerNoPE = nil
		m.rebuildProgramForTest()

		c := m.NewCache(len(g.PromptIDs))
		lg, err := m.prefillLogits(context.Background(), g.PromptIDs, c)
		if err != nil {
			t.Fatalf("prefillLogits: %v", err)
		}
		maxAbs := maxAbsLogitDiff(lg, g.LastLogits)
		cos := logitCosine(lg, g.LastLogits)
		t.Logf("break-it-first (disable NoPE): maxAbs=%.4f cosine=%.6f", maxAbs, cos)
		if maxAbs < 1e-2 {
			t.Fatalf("disabling NoPE on full_attention layers barely moved logits (maxAbs=%.3e < 1e-2) — gate would not catch wrong RoPE mask", maxAbs)
		}
	})

	t.Run("window_off_by_one_512_vs_513", func(t *testing.T) {
		m, err := Load(ckpt, Options{Quant: "f32"})
		if err != nil {
			t.Fatalf("Load: %v", err)
		}
		defer m.Close()

		// Perturb: set SlidingWindow = 512 instead of 513 (the classic off-by-one between
		// inclusive 513 and exclusive 512). At len=512 (pos=511), 512 keys fit inside both
		// windows so logits must be identical; at len=513 (pos=512), window=512 evicts pos 0
		// while window=513 keeps it, so logits must diverge.
		m.w.arch.SlidingWindow = 512
		m.rebuildProgramForTest()

		ids := make([]int, 514)
		for i := range ids {
			ids[i] = (i*31 + 7) % m.w.arch.VocabSize
		}
		var want512, want513 []float32
		for _, cp := range g.Window513.Checkpoints {
			if cp.Len == 512 {
				want512 = cp.Logits
			}
			if cp.Len == 513 {
				want513 = cp.Logits
			}
		}

		c512 := m.NewCache(512)
		lg512, err := m.prefillLogits(context.Background(), ids[:512], c512)
		if err != nil {
			t.Fatalf("prefillLogits(512): %v", err)
		}
		if diff512 := maxAbsLogitDiff(lg512, want512); diff512 > 1e-5 {
			t.Fatalf("at len=512 (inside both 512 and 513 windows), maxAbs=%.3e > 1e-5", diff512)
		}

		c513 := m.NewCache(513)
		lg513, err := m.prefillLogits(context.Background(), ids[:513], c513)
		if err != nil {
			t.Fatalf("prefillLogits(513): %v", err)
		}
		diff513 := maxAbsLogitDiff(lg513, want513)
		t.Logf("break-it-first (window 512 vs 513 at len=513): maxAbs=%.4f", diff513)
		if diff513 < 1e-3 {
			t.Fatalf("window=512 vs 513 at len=513 did not diverge (maxAbs=%.3e < 1e-3)", diff513)
		}
	})

	t.Run("drop_qk_norm", func(t *testing.T) {
		m, err := Load(ckpt, Options{Quant: "f32"})
		if err != nil {
			t.Fatalf("Load: %v", err)
		}
		defer m.Close()

		// Perturb: disable QKNorm so q_norm and k_norm are skipped.
		m.w.arch.QKNorm = false
		m.rebuildProgramForTest()

		c := m.NewCache(len(g.PromptIDs))
		lg, err := m.prefillLogits(context.Background(), g.PromptIDs, c)
		if err != nil {
			t.Fatalf("prefillLogits: %v", err)
		}
		maxAbs := maxAbsLogitDiff(lg, g.LastLogits)
		cos := logitCosine(lg, g.LastLogits)
		t.Logf("break-it-first (drop QKNorm): maxAbs=%.4f cosine=%.6f", maxAbs, cos)
		if maxAbs < 1e-2 {
			t.Fatalf("dropping QKNorm barely moved logits (maxAbs=%.3e < 1e-2) — gate would not catch missing QKNorm", maxAbs)
		}
	})
}

// TestFlint_realCheckpointParity is F2.3 (docs/tasks/task-flint-2026-10.md):
// env-gated Tier 2 verification on a real RecursiveSI/flint checkpoint (GGUF or
// safetensors directory pointed to by GOINFER_FLINT_DIR).
func TestFlint_realCheckpointParity(t *testing.T) {
	dir := os.Getenv("GOINFER_FLINT_DIR")
	if dir == "" {
		t.Skip("GOINFER_FLINT_DIR not set — skipping real-checkpoint Flint parity check")
	}
	m, err := Load(dir, Options{})
	if err != nil {
		t.Fatalf("Load(%s): %v", dir, err)
	}
	defer m.Close()
	if m.w.arch.Name != "flint" {
		t.Fatalf("arch.Name = %q, want flint", m.w.arch.Name)
	}
	if !m.w.arch.AttnSinks || !m.w.arch.QKNorm || m.w.arch.SlidingWindow != 513 {
		t.Fatalf("arch = %+v, want AttnSinks=true QKNorm=true SlidingWindow=513", m.w.arch)
	}
	cache := m.NewCache(64)
	logits, err := m.prefillLogits(context.Background(), []int{199999, 9125, 198}, cache)
	if err != nil {
		t.Fatalf("prefillLogits: %v", err)
	}
	for i, v := range logits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("logit[%d] = %v (NaN/Inf)", i, v)
		}
	}
	t.Logf("GOINFER_FLINT_DIR(%s): argmax=%d vocab=%d", dir, argmax(logits), len(logits))
}

// TestFlint_ggufConfig verifies ggufFlintConfig metadata extraction for GGUF files
// with general.architecture = "flint" (llama.cpp PR #30928).
func TestFlint_ggufConfig(t *testing.T) {
	g := &embed.GGUFFile{
		Metadata: map[string]any{
			"general.architecture":                    "flint",
			"flint.block_count":                       uint32(6),
			"flint.embedding_length":                  uint32(64),
			"flint.feed_forward_length":               uint32(128),
			"flint.attention.head_count":              uint32(4),
			"flint.attention.head_count_kv":           uint32(2),
			"flint.attention.key_length":              uint32(16),
			"flint.attention.layer_norm_rms_epsilon":  float32(1e-6),
			"flint.rope.freq_base":                    float32(10000.0),
			"flint.context_length":                    uint32(2048),
			"flint.attention.sliding_window":          uint32(513),
			"flint.leading_dense_block_count":         uint32(2),
			"flint.expert_count":                      uint32(8),
			"flint.expert_used_count":                 uint32(2),
			"flint.expert_feed_forward_length":        uint32(32),
			"flint.expert_shared_feed_forward_length": uint32(32),
			"flint.expert_gating_func":                uint32(2),
			"flint.expert_weights_norm":               true,
			"flint.attention.sliding_window_pattern":  []any{false, false, true, true, true, false},
			"flint.attention.rope_pattern":            []any{false, false, true, true, true, false},
			"tokenizer.ggml.tokens":                   make([]any, 256),
		},
	}
	cfg, err := ggufConfig(g)
	if err != nil {
		t.Fatalf("ggufConfig: %v", err)
	}
	arch, _, err := resolveArchitecture(cfg)
	if err != nil {
		t.Fatalf("resolveArchitecture: %v", err)
	}
	if arch.Name != "flint" || arch.FirstKDense != 2 || arch.SlidingWindow != 513 || !arch.AttnSinks {
		t.Fatalf("resolved arch = %+v", arch)
	}
	wantGlobal := []bool{true, true, false, false, false, true}
	for l := range arch.NumLayers {
		if arch.isGlobalLayer(l) != wantGlobal[l] || arch.isNoPELayer(l) != wantGlobal[l] {
			t.Fatalf("layer %d global=%v nope=%v, want %v", l, arch.isGlobalLayer(l), arch.isNoPELayer(l), wantGlobal[l])
		}
	}
}
