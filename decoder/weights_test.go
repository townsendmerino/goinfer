package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"os"
	"testing"
)

// M1 loader parity. Assets live under testdata/ and are per-machine (the
// 270M checkpoint is ~340 MB); the test SKIPS cleanly when they're absent,
// so a fresh checkout stays green — the same convention encoder/ uses.
//
// Get the checkpoint + regenerate the golden:
//
//	huggingface-cli download google/gemma-3-270m --local-dir testdata/gemma-3-270m
//	.venv/bin/python scripts/pin_gemma.py
const (
	gemmaModelDir   = "../testdata/gemma-3-270m"
	gemmaGoldenPath = "../testdata/gemma_golden.json"
)

type gemmaGolden struct {
	ModelID        string                       `json:"model_id"`
	SampledTensors []string                     `json:"sampled_tensors"`
	Config         map[string]any               `json:"config"`
	Tensors        map[string]gemmaTensorGolden `json:"tensors"`
}

type gemmaTensorGolden struct {
	Shape []int   `json:"shape"`
	DType string  `json:"dtype"`
	N     int     `json:"n"`
	Sum   float64 `json:"sum"`
	SumSq float64 `json:"sum_sq"`
}

// checksumF64 reproduces pin_gemma.py's reduction exactly: float64 sum and
// sum-of-squares over the (already widened) f32 values.
func checksumF64(xs []float32) (sum, sumSq float64) {
	for _, v := range xs {
		f := float64(v)
		sum += f
		sumSq += f * f
	}
	return sum, sumSq
}

// TestLoadWeights_goldenChecksums loads the real checkpoint and verifies the
// sampled tensors match the pinned golden: shape (catches a transpose, since
// Gemma's matrices are non-square), element count, stored dtype (proves the
// BF16 widen path ran), and the float64 checksums (catch a dtype-misread,
// byte-order slip, or wrong tensor — value fidelity, which shape can't see).
func TestLoadWeights_goldenChecksums(t *testing.T) {
	g := loadGemmaGolden(t) // skips if golden absent
	if _, err := os.Stat(gemmaModelDir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no checkpoint at %s — huggingface-cli download google/gemma-3-270m --local-dir %s",
			gemmaModelDir, gemmaModelDir)
	}

	w, err := LoadWeights(gemmaModelDir)
	if err != nil {
		t.Fatalf("LoadWeights: %v", err)
	}
	defer w.st.Close() // release the mmap (in-package access)

	// Cross-check the parsed config against the golden's config dict, so a
	// config-field drift fails here rather than as a mystery shape error.
	assertConfigInt(t, g, "num_hidden_layers", w.Cfg.NumLayers)
	assertConfigInt(t, g, "hidden_size", w.Cfg.HiddenDim)
	assertConfigInt(t, g, "vocab_size", w.Cfg.VocabSize)

	checkSampledChecksums(t, w, g)
}

// checkSampledChecksums verifies the loaded weights reproduce the golden's
// sampled-tensor shape/dtype/checksums. Shared by the single-file (M1) and
// sharded (G1) loader tests so both go through the identical bar.
func checkSampledChecksums(t *testing.T, w *Weights, g *gemmaGolden) {
	t.Helper()
	loaded := map[string][]float32{
		"model.embed_tokens.weight":              w.Embed.f32,
		"model.norm.weight":                      w.FinalNorm,
		"model.layers.0.self_attn.q_proj.weight": w.Layers[0].QProj.f32,
	}
	const relTol = 1e-6
	for name, want := range g.Tensors {
		got, ok := loaded[name]
		if !ok {
			t.Errorf("golden samples %q but the test has no field mapping for it", name)
			continue
		}
		// Element count (== product of shape; shape itself was validated
		// against Cfg inside loadF32, so a successful load already pins it).
		if len(got) != want.N {
			t.Errorf("%s: loaded %d elems, golden N=%d (shape %v)", name, len(got), want.N, want.Shape)
			continue
		}
		// Stored dtype — confirms we actually exercised the BF16/F16 path,
		// not an accidental F32 checkpoint.
		if tn, err := w.st.Tensor(name); err == nil && tn.DType != want.DType {
			t.Errorf("%s: dtype %q, golden %q", name, tn.DType, want.DType)
		}

		gotSum, gotSumSq := checksumF64(got)

		// sum_sq is all-positive and stable: plain relative tolerance.
		if relErr(gotSumSq, want.SumSq) > relTol {
			t.Errorf("%s: sum_sq %.10g, golden %.10g (rel %.2e > %.0e)",
				name, gotSumSq, want.SumSq, relErr(gotSumSq, want.SumSq), relTol)
		}
		// sum can cancel (mixed signs over millions of f64 adds), so a
		// sum-relative tolerance is unstable. Compare to the data SCALE
		// (sqrt(sum_sq), i.e. the L2 norm) instead: a real loading bug shifts
		// the sum by O(scale), far above this bar, while benign reduction-
		// order differences (torch pairwise vs Go sequential) stay well under.
		scale := math.Sqrt(want.SumSq)
		if math.Abs(gotSum-want.Sum) > relTol*scale+1e-9 {
			t.Errorf("%s: sum %.10g, golden %.10g (|Δ| %.3e > %.3e = %.0e·scale)",
				name, gotSum, want.Sum, math.Abs(gotSum-want.Sum), relTol*scale, relTol)
		}
	}
}

func loadGemmaGolden(t *testing.T) *gemmaGolden {
	t.Helper()
	b, err := os.ReadFile(gemmaGoldenPath)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no golden at %s — regenerate with scripts/pin_gemma.py", gemmaGoldenPath)
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g gemmaGolden
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	if len(g.Tensors) == 0 {
		t.Fatalf("golden has no tensors")
	}
	return &g
}

func assertConfigInt(t *testing.T, g *gemmaGolden, key string, got int) {
	t.Helper()
	raw, ok := g.Config[key]
	if !ok {
		return // golden config didn't record this key; nothing to cross-check
	}
	f, ok := raw.(float64) // JSON numbers decode to float64
	if !ok {
		t.Errorf("golden config %q is %T, not a number", key, raw)
		return
	}
	if int(f) != got {
		t.Errorf("config %q: loaded %d, golden %d", key, got, int(f))
	}
}

func relErr(got, want float64) float64 {
	d := math.Abs(got - want)
	if a := math.Abs(want); a > 0 {
		return d / a
	}
	return d
}

// TestValidateAssumptions covers the acceptance criterion that a Gemma 2
// soft-capping checkpoint is rejected, plus the other guards. No model
// assets needed — pure synthetic configs, always runs.
func TestValidateAssumptions(t *testing.T) {
	base := func() Config {
		return Config{
			VocabSize: 262144, HiddenDim: 640, NumLayers: 18,
			NumHeads: 4, NumKVHeads: 1, HeadDim: 256, IntermediateDim: 2048,
			MaxPositions: 32768, RMSNormEps: 1e-6,
			RoPELocalBase: 10000, RoPEGlobalBase: 1000000,
			SlidingWindow: 512, SlidingWindowPattern: 6,
			QueryPreAttnScalar: 256, UseQKNorm: true,
			HiddenActivation: "gelu_pytorch_tanh",
		}
	}
	valid := base()
	if err := valid.ValidateAssumptions(); err != nil {
		t.Fatalf("valid 270M config rejected: %v", err)
	}

	cases := []struct {
		name   string
		mutate func(*Config)
	}{
		{"final_logit_softcapping", func(c *Config) { c.FinalLogitSoftcap = 30.0 }},
		{"attn_logit_softcapping", func(c *Config) { c.AttnLogitSoftcap = 50.0 }},
		{"gqa_not_divisible", func(c *Config) { c.NumKVHeads = 3 }}, // 4 % 3 != 0
		{"missing_hidden", func(c *Config) { c.HiddenDim = 0 }},
		{"zero_vocab", func(c *Config) { c.VocabSize = 0 }},
		{"bad_activation", func(c *Config) { c.HiddenActivation = "relu" }},
		{"nonpositive_eps", func(c *Config) { c.RMSNormEps = 0 }},
		{"zero_rope_base", func(c *Config) { c.RoPELocalBase = 0 }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := base()
			tc.mutate(&c)
			if err := c.ValidateAssumptions(); err == nil {
				t.Errorf("%s: expected rejection, got nil", tc.name)
			}
		})
	}
}
