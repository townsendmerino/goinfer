package config

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// A config.json whose eos_token_id is JSON null (transformers writes the key with None when a model leaves it to
// generation_config.json: Qwen3-ASR's text_config does) must have NO config-side stop id. json.Unmarshal of null into
// an int succeeds and leaves 0, so EOSIDs() would return [0] and token id 0 would end generation (in Qwen's vocabulary
// id 0 is "!", so a Qwen3-ASR transcription would stop at its first "!").
func TestConfigEOSIDs_nullIsAbsent(t *testing.T) {
	for _, tc := range []struct {
		name string
		raw  string // "" = the key is absent
		want []int
	}{
		{"absent", "", nil},
		{"null", "null", nil},
		{"one", "151645", []int{151645}},
		{"list", "[151645,151643]", []int{151645, 151643}},
		{"zero is a real id when written", "0", []int{0}},
	} {
		var c Config
		if tc.raw != "" {
			c.EOSTokenID = json.RawMessage(tc.raw)
		}
		if got := c.EOSIDs(); !slices.Equal(got, tc.want) {
			t.Errorf("%s: EOSIDs() = %v, want %v", tc.name, got, tc.want)
		}
	}
}

// TestEOSIDs covers the scalar / list / absent shapes HF emits for
// eos_token_id directly on a Config.
func TestEOSIDs(t *testing.T) {
	cases := []struct {
		name string
		raw  string
		want []int
	}{
		{"absent", "", nil},
		{"scalar", "1", []int{1}},
		{"list", "[1, 106]", []int{1, 106}},
		{"qwen3", "[151645, 151643]", []int{151645, 151643}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			cfg := Config{}
			if c.raw != "" {
				cfg.EOSTokenID = []byte(c.raw)
			}
			got := cfg.EOSIDs()
			if !eqInts(got, c.want) {
				t.Errorf("EOSIDs() = %v, want %v", got, c.want)
			}
		})
	}
}

// TestResolveEOSIDs verifies the load-time merge of config.json and
// generation_config.json — the path Qwen3 chat relies on: config.json carries
// only <|im_end|> (151645) while generation_config adds <|endoftext|>
// (151643), and both must stop a turn. Deduped, config.json first.
func TestResolveEOSIDs(t *testing.T) {
	dir := t.TempDir()
	write := func(name, body string) {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	// config.json eos is a scalar; generation_config adds a second id plus a
	// duplicate of the first (must be deduped).
	write("generation_config.json", `{"eos_token_id": [151645, 151643]}`)
	cfg := &Config{EOSTokenID: []byte("151645")}
	got := ResolveEOSIDs(dir, cfg)
	want := []int{151645, 151643}
	if !eqInts(got, want) {
		t.Errorf("resolveEOSIDs = %v, want %v (config first, deduped)", got, want)
	}

	// Missing generation_config.json → just config.json's ids (best-effort).
	empty := t.TempDir()
	if got := ResolveEOSIDs(empty, &Config{EOSTokenID: []byte("[2]")}); !eqInts(got, []int{2}) {
		t.Errorf("resolveEOSIDs (no gen config) = %v, want [2]", got)
	}
}

// TestGemma4GlobalGeometry pins how the full-attention geometry is read: the flat fields as before, the
// per_layer_config map transformers >= 5.16 writes in their place, and the refusals for any override goinfer
// cannot represent (it has exactly two attention geometries).
func TestGemma4GlobalGeometry(t *testing.T) {
	types := []string{"sliding_attention", "full_attention", "sliding_attention", "full_attention"}
	cases := []struct {
		name         string
		flatHD, flKV int
		plc          string
		hd, kv       int
		err          string
	}{
		{name: "flat only", flatHD: 512, flKV: 1, hd: 512, kv: 1},
		{name: "per_layer_config only", plc: `{"1":{"head_dim":64},"3":{"head_dim":64,"num_key_value_heads":2}}`, hd: 64, kv: 2},
		{name: "null map", flatHD: 512, plc: `null`, hd: 512},
		{name: "both agree", flatHD: 64, plc: `{"1":{"head_dim":64},"3":{"head_dim":64}}`, hd: 64},
		{name: "both disagree", flatHD: 512, plc: `{"1":{"head_dim":64}}`, err: "disagrees"},
		{name: "full layers differ", plc: `{"1":{"head_dim":64},"3":{"head_dim":128}}`, err: "different head_dim"},
		{name: "sliding override", plc: `{"0":{"head_dim":64}}`, err: "sliding layer 0"},
		{name: "unknown field", plc: `{"1":{"sliding_window":8}}`, err: `"sliding_window"`},
		{name: "bad index", plc: `{"9":{"head_dim":64}}`, err: "not a layer index"},
	}
	for _, c := range cases {
		cfg := &Config{NumLayers: len(types), LayerTypes: types, GlobalHeadDim: c.flatHD, NumGlobalKVHeads: c.flKV}
		if c.plc != "" {
			cfg.PerLayerConfig = json.RawMessage(c.plc)
		}
		hd, kv, err := cfg.Gemma4GlobalGeometry()
		switch {
		case c.err != "":
			if err == nil || !strings.Contains(err.Error(), c.err) {
				t.Errorf("%s: err %v, want one containing %q", c.name, err, c.err)
			}
		case err != nil:
			t.Errorf("%s: %v", c.name, err)
		case hd != c.hd || kv != c.kv:
			t.Errorf("%s: got (%d, %d), want (%d, %d)", c.name, hd, kv, c.hd, c.kv)
		}
	}
}

// TestLagunaGating_allThreeSpellings pins the resolution of `gating`, which ships
// as a different JSON type in each of the three releases.
//
// The declared value is only an EXPECTATION: Laguna-XS.2 says `gating: true`,
// which the XS-2.1/M.1 module resolves to per-element, yet XS.2's own module
// hardcodes nn.Linear(hidden, num_heads) and its shipped g_proj is [64, 2048] —
// per-HEAD. So the loader picks granularity from the tensor's row count and this
// only has to parse each spelling without erroring. The test records that the
// vendor rule is generation-specific rather than pretending config is decisive.
func TestLagunaGating_allThreeSpellings(t *testing.T) {
	for _, tc := range []struct {
		name          string
		raw           string
		wantEnabled   bool
		wantPerHead   bool
		wantParseFail bool
	}{
		{name: "XS-2.1 string per-head", raw: `"per-head"`, wantEnabled: true, wantPerHead: true},
		{name: "M.1 string per-element", raw: `"per-element"`, wantEnabled: true, wantPerHead: false},
		{name: "XS.2 bool true", raw: `true`, wantEnabled: true, wantPerHead: false},
		{name: "explicit false", raw: `false`, wantEnabled: false, wantPerHead: false},
		{name: "absent defaults on", raw: ``, wantEnabled: true, wantPerHead: false},
		{name: "unknown string rejected", raw: `"per-block"`, wantParseFail: true},
		{name: "wrong type rejected", raw: `7`, wantParseFail: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := &Config{}
			if tc.raw != "" {
				cfg.Gating = json.RawMessage(tc.raw)
			}
			enabled, perHead, err := cfg.LagunaGatePerHead()
			if tc.wantParseFail {
				if err == nil {
					t.Fatalf("gating=%s parsed as (%v,%v), want an error", tc.raw, enabled, perHead)
				}
				return
			}
			if err != nil {
				t.Fatalf("gating=%s: %v", tc.raw, err)
			}
			if enabled != tc.wantEnabled || perHead != tc.wantPerHead {
				t.Errorf("gating=%s → (enabled=%v, perHead=%v), want (%v, %v)",
					tc.raw, enabled, perHead, tc.wantEnabled, tc.wantPerHead)
			}
		})
	}
}

// TestLagunaFirstKDense_contiguousOnly pins that mlp_only_layers is accepted only
// as a contiguous prefix — the shape FirstKDense can express. Every released
// config is contiguous ([0] on XS-2.1, [0,1,2] on M.1); a non-contiguous list is a
// layout FirstKDense would silently truncate, so it is rejected instead.
func TestLagunaFirstKDense_contiguousOnly(t *testing.T) {
	for _, tc := range []struct {
		name    string
		layers  []int
		want    int
		wantErr bool
	}{
		{name: "absent", layers: nil, want: 0},
		{name: "mlp_only_layers used when mlp_layer_types absent", layers: []int{0, 1}, want: 2},
		{name: "XS [0]", layers: []int{0}, want: 1},
		{name: "M.1 [0,1,2]", layers: []int{0, 1, 2}, want: 3},
		{name: "unordered but contiguous", layers: []int{2, 0, 1}, want: 3},
		{name: "gap rejected", layers: []int{0, 2}, wantErr: true},
		{name: "non-prefix rejected", layers: []int{5}, wantErr: true},
		{name: "out of range rejected", layers: []int{99}, wantErr: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := &Config{NumLayers: 40, MlpOnlyLayers: tc.layers}
			got, err := cfg.LagunaFirstKDense()
			if tc.wantErr {
				if err == nil {
					t.Fatalf("mlp_only_layers=%v → %d, want an error", tc.layers, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("mlp_only_layers=%v: %v", tc.layers, err)
			}
			if got != tc.want {
				t.Errorf("mlp_only_layers=%v → FirstKDense %d, want %d", tc.layers, got, tc.want)
			}
		})
	}
}

// TestLagunaFirstKDense_mlpLayerTypes pins the PREFERRED spelling. XS.2 ships
// mlp_layer_types but NOT mlp_only_layers, so a reader that only knows the latter
// returns FirstKDense=0 for it and the loader then treats XS.2's dense layer 0 as
// MoE — demanding expert tensors the checkpoint does not contain. All three
// released configs carry mlp_layer_types, so it wins whenever present.
func TestLagunaFirstKDense_mlpLayerTypes(t *testing.T) {
	dense, sparse := "dense", "sparse"
	for _, tc := range []struct {
		name    string
		types   []string
		only    []int
		want    int
		wantErr bool
	}{
		{name: "XS one dense prefix", types: []string{dense, sparse, sparse, sparse}, want: 1},
		{name: "M.1 three dense prefix", types: []string{dense, dense, dense, sparse}, want: 3},
		{name: "all sparse", types: []string{sparse, sparse, sparse, sparse}, want: 0},
		// mlp_layer_types must WIN over a disagreeing mlp_only_layers, since it is the
		// spelling every release carries.
		{name: "beats disagreeing mlp_only_layers", types: []string{dense, sparse, sparse, sparse}, only: []int{0, 1, 2}, want: 1},
		{name: "non-prefix dense rejected", types: []string{sparse, dense, sparse, sparse}, wantErr: true},
		{name: "unknown kind rejected", types: []string{dense, "moe", sparse, sparse}, wantErr: true},
		{name: "wrong length rejected", types: []string{dense, sparse}, wantErr: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := &Config{NumLayers: 4, MlpLayerTypes: tc.types, MlpOnlyLayers: tc.only}
			got, err := cfg.LagunaFirstKDense()
			if tc.wantErr {
				if err == nil {
					t.Fatalf("mlp_layer_types=%v → %d, want an error", tc.types, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("mlp_layer_types=%v: %v", tc.types, err)
			}
			if got != tc.want {
				t.Errorf("mlp_layer_types=%v → FirstKDense %d, want %d", tc.types, got, tc.want)
			}
		})
	}
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

func eqInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
