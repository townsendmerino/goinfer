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

	// Run BOTH sides of P13, because they answer different questions and only
	// one of them existed before P13 did.
	//
	//   p13_off     — the source mapping is retained, so every assertion below
	//                 runs, INCLUDING the stored-dtype check that proves the
	//                 BF16 widen path ran. That check reads w.st, which P13
	//                 closes by default, so without this arm it would quietly
	//                 stop running rather than fail.
	//   p13_default — the shipped path, where the mapping is released at end of
	//                 load. The checksums here are what says that releasing it
	//                 did not corrupt the weights already loaded out of it,
	//                 which is the one thing P13 could plausibly break and the
	//                 exact risk mmapAliasRisk is guarding.
	//
	// P13 closing w.st is also why this test panicked rather than failed when it
	// was first run against a checkpoint: the Mac skips it for want of the asset,
	// so the nil deref only ever appeared on the box.
	for _, tc := range []struct {
		name   string
		p13Off bool
	}{{"p13_off", true}, {"p13_default", false}} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.p13Off {
				t.Setenv("GOINFER_P13_OFF", "1")
			}
			w, err := LoadWeights(gemmaModelDir)
			if err != nil {
				t.Fatalf("LoadWeights: %v", err)
			}
			defer func() {
				if w.st != nil { // nil under P13's default: already closed at end of load
					w.st.Close()
				}
			}()

			// Cross-check the parsed config against the golden's config dict, so a
			// config-field drift fails here rather than as a mystery shape error.
			assertConfigInt(t, g, "num_hidden_layers", w.Cfg.NumLayers)
			assertConfigInt(t, g, "hidden_size", w.Cfg.HiddenDim)
			assertConfigInt(t, g, "vocab_size", w.Cfg.VocabSize)

			if tc.p13Off && w.st == nil {
				t.Fatal("GOINFER_P13_OFF=1 but the source mapping was released anyway — this arm exists to keep the dtype assertion running, and it just stopped running")
			}
			checkSampledChecksums(t, w, g)
		})
	}
}

// checkSampledChecksums verifies the loaded weights reproduce the golden's
// sampled-tensor shape/dtype/checksums. Shared by the single-file (M1) and
// sharded (G1) loader tests so both go through the identical bar.
func checkSampledChecksums(t *testing.T, w *Weights, g *gemmaGolden) {
	t.Helper()
	loaded := map[string][]float32{
		"model.embed_tokens.weight":              tF32(&w.Embed),
		"model.norm.weight":                      w.FinalNorm,
		"model.layers.0.self_attn.q_proj.weight": tF32(&w.Layers[0].QProj),
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
		// not an accidental F32 checkpoint. Needs the source mapping, which P13
		// releases at end of load, so it can only run on the retained arm. Logged
		// rather than silently passed over: an assertion that stops running looks
		// exactly like an assertion that passes.
		if w.st == nil {
			t.Logf("%s: dtype check skipped — source mapping released (P13); the p13_off arm covers it", name)
		} else if tn, err := w.st.Tensor(name); err == nil && tn.DType != want.DType {
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
