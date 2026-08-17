package decoder

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// TestLagunaArchitecture_realConfigs resolves the THREE REAL released Laguna
// configs — Laguna-XS-2.1, Laguna-XS.2, Laguna-M.1, vendored verbatim under
// testdata/laguna_configs — and pins what the adapter derives from each.
//
// WHY REAL CONFIGS RATHER THAN A HAND-WRITTEN ONE. Every surprise in this family
// so far came from a released file disagreeing with the vendor's own module or
// with the previous generation: `gating` ships as three different JSON TYPES
// across three releases, `num_attention_heads_per_layer` and `layer_types` are
// present on the XS line and absent on M.1, and the YaRN factor changes between
// generations. A synthetic config would encode my assumptions and gate nothing.
// These are the actual bytes the loader will see.
//
// This is a config-resolution gate, not a numerical one — the tiny-golden and
// real-checkpoint parity gates cover the forward pass.
func TestLagunaArchitecture_realConfigs(t *testing.T) {
	const headDim = 128
	for _, tc := range []struct {
		file       string
		repo       string
		layers     int
		hidden     int
		heads      int // uniform num_attention_heads
		headsAt0   int // layer 0 query heads (full_attention on the XS line)
		headsAt1   int // layer 1 query heads (sliding_attention on the XS line)
		experts    int
		topK       int
		moeInter   int
		sharedDim  int
		routedScl  float64
		firstDense int
		sliding    int
		globalAt0  bool
		globalAt1  bool
		rotaryDim  int // global/full layers; 0 ⇒ full head_dim
		rotaryLoc  int // local/sliding layers; 0 ⇒ same as rotaryDim
		yarnFactor float64
	}{{
		file: "xs21.json", repo: "poolside/Laguna-XS-2.1",
		layers: 40, hidden: 2048, heads: 48, headsAt0: 48, headsAt1: 64,
		experts: 256, topK: 8, moeInter: 512, sharedDim: 512, routedScl: 2.5,
		firstDense: 1, sliding: 512, globalAt0: true, globalAt1: false,
		rotaryDim: 64, rotaryLoc: 128, yarnFactor: 32,
	}, {
		file: "xs2.json", repo: "poolside/Laguna-XS.2",
		layers: 40, hidden: 2048, heads: 48, headsAt0: 48, headsAt1: 64,
		experts: 256, topK: 8, moeInter: 512, sharedDim: 512, routedScl: 2.5,
		// XS.2 drops mlp_only_layers; its layer 0 is still the dense one on disk
		// (mlp.gate_proj.weight exists only for layer 0), so FirstKDense comes out 0
		// from config alone and the loader's per-layer Router presence decides.
		firstDense: 0, sliding: 512, globalAt0: true, globalAt1: false,
		rotaryDim: 64, rotaryLoc: 128, yarnFactor: 64,
	}, {
		file: "m1.json", repo: "poolside/Laguna-M.1",
		layers: 70, hidden: 4096, heads: 64, headsAt0: 64, headsAt1: 64,
		experts: 256, topK: 16, moeInter: 1024, sharedDim: 1024, routedScl: 1.0,
		firstDense: 3, sliding: 0, globalAt0: true, globalAt1: true,
		// M.1 is all-full-attention with partial_rotary_factor 1.0 ⇒ full-width rotary
		// and no separate local table.
		rotaryDim: 0, rotaryLoc: 0, yarnFactor: 64,
	}} {
		t.Run(tc.file, func(t *testing.T) {
			raw, err := os.ReadFile(filepath.Join("testdata", "laguna_configs", tc.file))
			if err != nil {
				t.Fatalf("read: %v", err)
			}
			var cfg Config
			if err := json.Unmarshal(raw, &cfg); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if cfg.ModelType != "laguna" {
				t.Fatalf("model_type = %q, want laguna", cfg.ModelType)
			}
			arch, schema, err := lagunaArchitecture(&cfg)
			if err != nil {
				t.Fatalf("lagunaArchitecture(%s): %v", tc.repo, err)
			}
			arch.finalizeRoPE()

			if arch.NumLayers != tc.layers || arch.HiddenDim != tc.hidden || arch.NumHeads != tc.heads {
				t.Errorf("layers/hidden/heads = %d/%d/%d, want %d/%d/%d",
					arch.NumLayers, arch.HiddenDim, arch.NumHeads, tc.layers, tc.hidden, tc.heads)
			}
			if arch.HeadDim != headDim || arch.NumKVHeads != 8 {
				t.Errorf("head_dim/kv_heads = %d/%d, want %d/8", arch.HeadDim, arch.NumKVHeads, headDim)
			}
			// Per-layer QUERY heads — the accessor the XS line needs and M.1 does not.
			if got := arch.headsAt(0); got != tc.headsAt0 {
				t.Errorf("headsAt(0) = %d, want %d", got, tc.headsAt0)
			}
			if got := arch.headsAt(1); got != tc.headsAt1 {
				t.Errorf("headsAt(1) = %d, want %d", got, tc.headsAt1)
			}
			if got := arch.maxHeads(); got != max(tc.headsAt0, tc.headsAt1) {
				t.Errorf("maxHeads() = %d, want %d", got, max(tc.headsAt0, tc.headsAt1))
			}
			// Every per-layer head count must stay a whole number of GQA groups.
			for i := range arch.NumLayers {
				if h := arch.headsAt(i); h%arch.NumKVHeads != 0 {
					t.Fatalf("headsAt(%d) = %d is not a multiple of kv_heads %d", i, h, arch.NumKVHeads)
				}
			}
			// MoE.
			if arch.MoE == nil {
				t.Fatalf("MoE is nil")
			}
			if arch.MoE.NumExperts != tc.experts || arch.MoE.TopK != tc.topK {
				t.Errorf("experts/topK = %d/%d, want %d/%d", arch.MoE.NumExperts, arch.MoE.TopK, tc.experts, tc.topK)
			}
			if arch.MoE.IntermediateDim != tc.moeInter || arch.MoE.SharedIntermediateDim != tc.sharedDim {
				t.Errorf("moeInter/sharedDim = %d/%d, want %d/%d",
					arch.MoE.IntermediateDim, arch.MoE.SharedIntermediateDim, tc.moeInter, tc.sharedDim)
			}
			if arch.MoE.RoutedScale != tc.routedScl {
				t.Errorf("RoutedScale = %v, want %v", arch.MoE.RoutedScale, tc.routedScl)
			}
			if !arch.MoE.RouterSigmoid {
				t.Error("RouterSigmoid = false, want true (Laguna scores experts by sigmoid)")
			}
			// SharedUngated is true because Laguna adds the shared expert with NO outer
			// sigmoid gate. Reading "LagunaMLP is a gated SwiGLU" as SharedUngated:false
			// is the natural-looking mistake and would multiply the shared branch by a
			// sigmoid the model never trained with.
			if !arch.MoE.SharedUngated {
				t.Error("SharedUngated = false, want true (no outer sigmoid on the shared branch)")
			}
			if !arch.MoE.NormTopKProb && tc.file != "xs2.json" {
				t.Error("NormTopKProb = false, want true")
			}
			if arch.FirstKDense != tc.firstDense {
				t.Errorf("FirstKDense = %d, want %d", arch.FirstKDense, tc.firstDense)
			}
			// Attention shape flags.
			if arch.QKVBias {
				t.Error("QKVBias = true, want false (the vendor lists 'no QKV bias' explicitly)")
			}
			if !arch.QKNorm {
				t.Error("QKNorm = false, want true — q_norm/k_norm are UNCONDITIONAL in all three released modules and ship in the checkpoint")
			}
			// Sliding-window interleave.
			if arch.SlidingWindow != tc.sliding {
				t.Errorf("SlidingWindow = %d, want %d", arch.SlidingWindow, tc.sliding)
			}
			if got := arch.isGlobalLayer(0); got != tc.globalAt0 {
				t.Errorf("isGlobalLayer(0) = %v, want %v", got, tc.globalAt0)
			}
			if got := arch.isGlobalLayer(1); got != tc.globalAt1 {
				t.Errorf("isGlobalLayer(1) = %v, want %v", got, tc.globalAt1)
			}
			// RoPE: base, per-layer-type rotary width, and YaRN presence.
			if arch.RoPEGlobalBase != 500000 {
				t.Errorf("RoPEGlobalBase = %v, want 500000", arch.RoPEGlobalBase)
			}
			wantLocalBase := float64(10000)
			if tc.sliding == 0 {
				wantLocalBase = 500000 // M.1: no sliding layers, tables coincide
			}
			if arch.RoPELocalBase != wantLocalBase {
				t.Errorf("RoPELocalBase = %v, want %v", arch.RoPELocalBase, wantLocalBase)
			}
			if arch.RotaryDim != tc.rotaryDim || arch.RotaryDimLocal != tc.rotaryLoc {
				t.Errorf("RotaryDim/Local = %d/%d, want %d/%d",
					arch.RotaryDim, arch.RotaryDimLocal, tc.rotaryDim, tc.rotaryLoc)
			}
			if arch.ropeScaling == nil {
				t.Fatal("ropeScaling = nil, want YaRN on the full-attention layers")
			}
			if arch.ropeScaling.factor != tc.yarnFactor {
				t.Errorf("YaRN factor = %v, want %v", arch.ropeScaling.factor, tc.yarnFactor)
			}
			if tc.sliding > 0 && arch.ropeScalingLocal != nil {
				t.Error("ropeScalingLocal != nil, want nil (sliding layers use plain RoPE, rope_type default)")
			}
			// The two inv-freq tables must have the per-layer-type WIDTHS, since
			// applyRoPE reads the rotated half-width as len(invFreq). Getting this wrong
			// silently rotates the wrong number of dims rather than failing.
			wantGlobalHalf := headDim / 2
			if tc.rotaryDim > 0 {
				wantGlobalHalf = tc.rotaryDim / 2
			}
			if got := len(arch.ropeInvFreqGlobal); got != wantGlobalHalf {
				t.Errorf("len(ropeInvFreqGlobal) = %d, want %d", got, wantGlobalHalf)
			}
			wantLocalHalf := wantGlobalHalf
			if tc.rotaryLoc > 0 {
				wantLocalHalf = tc.rotaryLoc / 2
			}
			if got := len(arch.ropeInvFreqLocal); got != wantLocalHalf {
				t.Errorf("len(ropeInvFreqLocal) = %d, want %d", got, wantLocalHalf)
			}
			if schema.GProj == "" {
				t.Error("schema.GProj is empty; Laguna's attention output gate would never load")
			}
		})
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
			enabled, perHead, err := cfg.lagunaGatePerHead()
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
		{name: "XS [0]", layers: []int{0}, want: 1},
		{name: "M.1 [0,1,2]", layers: []int{0, 1, 2}, want: 3},
		{name: "unordered but contiguous", layers: []int{2, 0, 1}, want: 3},
		{name: "gap rejected", layers: []int{0, 2}, wantErr: true},
		{name: "non-prefix rejected", layers: []int{5}, wantErr: true},
		{name: "out of range rejected", layers: []int{99}, wantErr: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := &Config{NumLayers: 40, MlpOnlyLayers: tc.layers}
			got, err := cfg.lagunaFirstKDense()
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
