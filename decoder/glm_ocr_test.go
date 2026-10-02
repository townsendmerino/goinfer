package decoder

import (
	"context"
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"
)

// GLM-OCR text decoder (docs/tasks/task-glm-ocr-2026-10.md, O1). The gates, in the order a defect
// would surface:
//
//   - TestGlmOcr_realConfig: the real zai-org/GLM-OCR config.json resolves to the descriptor O0
//     measured (head_dim 128 not 96, [16 24 24], 16 layers, the two stop ids);
//   - TestGlmOcr_tensorNameMapping: the four norm names (GLM's are not Gemma's), the gate-first
//     split of the fused gate_up_proj, and the MTP layer being skipped — hermetic, never skips, and
//     built so that swapping post_self_attn_layernorm with post_attention_layernorm fails it;
//   - TestApplyMRoPEPairwise_*: the rotation pinned against HF's own cos/sin and apply_rotary_pos_emb;
//   - TestGlmOcr_forwardParity / _mropeParity: the whole decoder against HF, text-only and with an
//     image-style block of m-RoPE positions, so the pairwise m-RoPE is exercised THROUGH its caller.
//
// Regenerate the goldens and the checkpoint: ~/.venv-vl/bin/python scripts/pin_glm_ocr_tiny.py

const glmOcrTinyCkpt = "../testdata/glm-ocr-tiny"

func TestGlmOcr_realConfig(t *testing.T) {
	// testdata/glm_ocr_real_config.json is zai-org/GLM-OCR's config.json byte for byte, revision
	// 2e85a62840ccac27daa451df36c736c4636b8628.
	cfg, err := loadConfig(os.DirFS("../testdata"), "glm_ocr_real_config.json")
	if err != nil {
		t.Fatalf("loadConfig: %v", err)
	}
	if cfg.ModelType != "glm_ocr" {
		t.Fatalf("ModelType = %q, want the wrapper's top-level \"glm_ocr\" (text_config says glm_ocr_text; top-level must win)", cfg.ModelType)
	}
	arch, schema, err := resolveArchitecture(cfg)
	if err != nil {
		t.Fatalf("resolveArchitecture: %v", err)
	}
	if schema != &glmOcrTensorSchema {
		t.Fatalf("schema = %p, want glmOcrTensorSchema", schema)
	}
	checks := []struct {
		name      string
		got, want any
	}{
		{"Name", arch.Name, "glm_ocr"},
		{"HiddenDim", arch.HiddenDim, 1536},
		{"NumLayers (MTP layer excluded)", arch.NumLayers, 16},
		{"NumHeads", arch.NumHeads, 16},
		{"NumKVHeads", arch.NumKVHeads, 8},
		{"HeadDim (explicit, hidden/heads would be 96)", arch.HeadDim, 128},
		{"IntermediateDim", arch.IntermediateDim, 4608},
		{"VocabSize", arch.VocabSize, 59392},
		{"Norm", arch.Norm, NormRMS},
		{"RMSAddOne", arch.RMSAddOne, false},
		{"NormEps", arch.NormEps, 1e-5},
		{"NormPlacement", arch.NormPlacement, NormSandwich4},
		{"Act", arch.Act, ActSiLU},
		{"NonGatedMLP", arch.NonGatedMLP, false},
		{"MoE", arch.MoE == nil, true},
		{"QKVBias", arch.QKVBias, false},
		{"OutBias", arch.OutBias, false},
		{"QKNorm", arch.QKNorm, false},
		{"AttnScale (128^-0.5)", arch.AttnScale, math.Pow(128, -0.5)},
		{"SlidingWindow", arch.SlidingWindow, 0},
		{"RoPEGlobalBase", arch.RoPEGlobalBase, 10000.0},
		{"RoPELocalBase", arch.RoPELocalBase, 10000.0},
		{"RotaryDim (partial_rotary_factor 1.0)", arch.RotaryDim, 128},
		{"MRopeSection", arch.MRopeSection, []int{16, 24, 24}},
		{"MRopeInterleaved (contiguous layout)", arch.MRopeInterleaved, false},
		{"ropeInterleave (pairwise)", arch.ropeInterleave, true},
		{"EmbedScale", arch.EmbedScale, 0.0},
		{"len(inv-freq table) = head_dim/2", len(arch.ropeInvFreq(0)), 64},
		{"EOSIDs", cfg.EOSIDs(), []int{59246, 59253}},
	}
	for _, c := range checks {
		if !reflect.DeepEqual(c.got, c.want) {
			t.Errorf("%s = %v, want %v", c.name, c.got, c.want)
		}
	}
	// Same config under the text decoder's own model_type resolves identically.
	cfg2, _ := loadConfig(os.DirFS("../testdata"), "glm_ocr_real_config.json")
	cfg2.ModelType = "glm_ocr_text"
	arch2, _, err := resolveArchitecture(cfg2)
	if err != nil || arch2.HeadDim != 128 || !reflect.DeepEqual(arch2.MRopeSection, []int{16, 24, 24}) {
		t.Errorf("model_type glm_ocr_text: arch=%+v err=%v", arch2, err)
	}
	if got, want := arch.residentFeatures(), []ResidentFeature{FeatPairwiseMRoPE, FeatPairwiseRoPE, FeatSandwichNorm}; !slices.Equal(got, want) {
		t.Errorf("residentFeatures = %v, want %v", got, want)
	}
}

// TestGlmOcr_residentDeclined: GLM-OCR rotates PAIRWISE (GPT-J) over m-RoPE sections, so a backend may
// run it resident only if it declares FeatPairwiseRoPE and FeatPairwiseMRoPE, i.e. has pairwise rope
// kernels. CUDA does (cuda/rope_pairwise.cu, gated by cuda.TestGlmOcrResidentParityCUDA); Metal and
// WebGPU still have only the NeoX half-split kernels, and admitting glm_ocr there gave logit cosine
// -0.34 resident-vs-CPU on the CUDA twin of that kernel set (2026-10-01), with no error. The decline
// must name the missing features so `serve check` shows the real cause.
func TestGlmOcr_residentDeclined(t *testing.T) {
	cfg, err := loadConfig(os.DirFS("../testdata"), "glm_ocr_real_config.json")
	if err != nil {
		t.Fatal(err)
	}
	arch, _, err := resolveArchitecture(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for backend, feats := range residentBackendFeatures {
		wantAdmit := feats[FeatPairwiseRoPE] && feats[FeatPairwiseMRoPE]
		if backend == "cuda" && !wantAdmit {
			t.Errorf("cuda must declare both pairwise features (cuda/rope_pairwise.cu)")
		}
		if backend != "cuda" && wantAdmit {
			t.Errorf("backend %q declares the pairwise features: it needs pairwise rope kernels and a resident-vs-CPU gate on peaked attention first", backend)
		}
		got := ResidentEligible(arch, backend)
		why := residentGateReason(arch, backend)
		if backend == "cuda" {
			if !got {
				t.Errorf("cuda declines glm_ocr: %s", why)
			}
			continue
		}
		if got {
			t.Errorf("backend %q admits glm_ocr: its rope kernels are NeoX, the model's rotation is pairwise", backend)
			continue
		}
		for _, f := range []ResidentFeature{FeatPairwiseRoPE, FeatPairwiseMRoPE} {
			if !strings.Contains(why, string(f)) {
				t.Errorf("backend %q decline reason %q does not name %q", backend, why, f)
			}
		}
	}
}

// TestGlmOcr_configRejects: the adapter refuses configs it does not implement rather than computing
// something else with them.
func TestGlmOcr_configRejects(t *testing.T) {
	base := func() *Config {
		cfg, err := loadConfig(os.DirFS("../testdata"), "glm_ocr_real_config.json")
		if err != nil {
			t.Fatal(err)
		}
		return cfg
	}
	for _, tc := range []struct {
		name string
		mut  func(c *Config)
		want string
	}{
		{"bias", func(c *Config) { c.AttentionBias = true }, "attention_bias"},
		{"act", func(c *Config) { c.HiddenAct = "gelu" }, "hidden_act"},
		{"section sum", func(c *Config) {
			c.RopeParameters = json.RawMessage(`{"rope_type":"default","rope_theta":10000,"mrope_section":[8,12,12]}`)
		}, "sums to 32, want head_dim/2 = 64"},
		{"no section", func(c *Config) {
			c.RopeParameters = json.RawMessage(`{"rope_type":"default","rope_theta":10000}`)
		}, "mrope_section"},
		{"partial rotary", func(c *Config) {
			c.RopeParameters = json.RawMessage(`{"rope_type":"default","rope_theta":10000,"partial_rotary_factor":0.5,"mrope_section":[16,24,24]}`)
		}, "partial_rotary_factor"},
		{"interleaved layout", func(c *Config) {
			c.RopeParameters = json.RawMessage(`{"rope_type":"default","rope_theta":10000,"mrope_interleaved":true,"mrope_section":[16,24,24]}`)
		}, "mrope_interleaved"},
		{"no rope", func(c *Config) { c.RopeParameters = nil }, "rope_parameters"},
		{"scaled rope", func(c *Config) {
			c.RopeParameters = json.RawMessage(`{"rope_type":"linear","factor":2.0,"rope_theta":10000,"mrope_section":[16,24,24]}`)
		}, "scaling"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := base()
			tc.mut(c)
			_, _, err := resolveArchitecture(c)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err = %v, want one containing %q", err, tc.want)
			}
		})
	}
}

// glmSynth builds a synthetic GLM-OCR checkpoint in dir whose every tensor is DISTINCT (a value
// pattern keyed by the tensor's own name), so a loader that crossed two same-shaped tensors is
// visible. hd (32) is deliberately not hidden/heads (24/4 = 6); the MTP layer at index nLayers has
// the real layer-16 tensor set, with NaN values and a WRONG q_proj shape: a loader that touched
// any of it would fail or poison the model.
func glmSynth(t *testing.T) (dir string, tensors map[string]stTensor) {
	t.Helper()
	const (
		hidden, nLayers, nH, nKV, hd, inter, vocab = 24, 2, 4, 2, 16, 40, 50
	)
	dir = t.TempDir()
	cfg := `{"architectures":["GlmOcrForConditionalGeneration"],"model_type":"glm_ocr","tie_word_embeddings":false,
	"text_config":{"model_type":"glm_ocr_text","hidden_size":24,"num_hidden_layers":2,"num_attention_heads":4,
	"num_key_value_heads":2,"head_dim":16,"intermediate_size":40,"vocab_size":50,"rms_norm_eps":1e-5,"hidden_act":"silu",
	"attention_bias":false,"eos_token_id":[3,5],"num_nextn_predict_layers":1,
	"rope_parameters":{"rope_type":"default","rope_theta":10000.0,"partial_rotary_factor":1.0,"mrope_section":[2,3,3]}}}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfg), 0o644); err != nil {
		t.Fatal(err)
	}
	tensors = map[string]stTensor{}
	salt := 0
	add := func(name string, shape ...int) {
		n := 1
		for _, s := range shape {
			n *= s
		}
		salt++
		d := make([]float32, n)
		for i := range d {
			// distinct per tensor AND per element; magnitudes small and non-degenerate
			d[i] = 0.05 * float32(math.Sin(float64(salt)*1.7+float64(i)*0.37+0.3))
		}
		tensors[name] = stTensor{shape: shape, data: d}
	}
	pre := "model.language_model."
	add(pre+"embed_tokens.weight", vocab, hidden)
	add(pre+"norm.weight", hidden)
	add("lm_head.weight", vocab, hidden)
	for l := range nLayers {
		p := pre + "layers." + string(rune('0'+l)) + "."
		add(p+"input_layernorm.weight", hidden)
		add(p+"post_self_attn_layernorm.weight", hidden)
		add(p+"post_attention_layernorm.weight", hidden)
		add(p+"post_mlp_layernorm.weight", hidden)
		add(p+"self_attn.q_proj.weight", nH*hd, hidden)
		add(p+"self_attn.k_proj.weight", nKV*hd, hidden)
		add(p+"self_attn.v_proj.weight", nKV*hd, hidden)
		add(p+"self_attn.o_proj.weight", hidden, nH*hd)
		add(p+"mlp.gate_up_proj.weight", 2*inter, hidden)
		add(p+"mlp.down_proj.weight", hidden, inter)
	}
	// The MTP layer. Wrong shapes and NaNs on purpose.
	nan := float32(math.NaN())
	poison := func(name string, shape ...int) {
		n := 1
		for _, s := range shape {
			n *= s
		}
		d := make([]float32, n)
		for i := range d {
			d[i] = nan
		}
		tensors[name] = stTensor{shape: shape, data: d}
	}
	mtp := pre + "layers.2."
	for _, n := range []string{"eh_proj.weight", "enorm.weight", "hnorm.weight", "embed_tokens.weight",
		"shared_head.head.weight", "shared_head.norm.weight", "input_layernorm.weight",
		"post_self_attn_layernorm.weight", "post_attention_layernorm.weight", "post_mlp_layernorm.weight",
		"self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
		"self_attn.o_proj.weight", "mlp.gate_up_proj.weight", "mlp.down_proj.weight"} {
		poison(mtp+n, 7, 3) // the wrong shape for every one of them
	}
	// An index far past the MTP layer, and a vision tower tensor: neither may be requested.
	poison(pre+"layers.9.self_attn.q_proj.weight", 1)
	poison("model.visual.blocks.0.attn.qkv.weight", 2, 2)
	writeSafetensors(t, filepath.Join(dir, "model.safetensors"), tensors)
	return dir, tensors
}

func f32Of(t *testing.T, w interface{ F32() ([]float32, bool) }, what string) []float32 {
	t.Helper()
	d, ok := w.F32()
	if !ok {
		t.Fatalf("%s is not an f32 weight at Quant f32", what)
	}
	return d
}

// TestGlmOcr_tensorNameMapping pins the loader's tensor-name map. The four norms are all [hidden]
// plain-weight RMSNorms, so a swapped pair loads cleanly and computes a different model; each one
// here has its own distinct values, and the test compares every LayerWeights field with the exact
// checkpoint tensor it must come from.
//
//	input_layernorm           -> PreAttnNorm
//	post_self_attn_layernorm  -> PostAttnNorm  (the norm on the attention OUTPUT)
//	post_attention_layernorm  -> PreMLPNorm    (the PRE-MLP norm: the REVERSE of Gemma's naming)
//	post_mlp_layernorm        -> PostMLPNorm
//
// It also pins the fused gate_up_proj split (GATE rows first), the head_dim-explicit projection
// widths (a derived head_dim of 6 would fail to load q_proj [64, 24]), the untied head, and that the
// MTP layer, a far-out layer index and the vision tower are never touched.
func TestGlmOcr_tensorNameMapping(t *testing.T) {
	dir, ts := glmSynth(t)
	m, err := Load(dir, Options{Quant: "f32"})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()
	w := m.w
	if len(w.Layers) != 2 {
		t.Fatalf("loaded %d layers, want 2 (the MTP layer 2 must not be one of them)", len(w.Layers))
	}
	if w.arch.TiedLMHead {
		t.Errorf("TiedLMHead = true with a lm_head.weight in the checkpoint")
	}
	if w.arch.HeadDim != 16 {
		t.Errorf("HeadDim = %d, want the explicit 16 (hidden/heads would be 6)", w.arch.HeadDim)
	}
	eq := func(what string, got, want []float32) {
		t.Helper()
		if len(got) != len(want) {
			t.Errorf("%s: len %d, want %d", what, len(got), len(want))
			return
		}
		for i := range got {
			if got[i] != want[i] {
				t.Errorf("%s: element %d = %v, want %v (wrong source tensor)", what, i, got[i], want[i])
				return
			}
		}
	}
	pre := "model.language_model."
	eq("Embed", f32Of(t, &w.Embed, "Embed"), ts[pre+"embed_tokens.weight"].data)
	eq("FinalNorm", w.FinalNorm, ts[pre+"norm.weight"].data)
	eq("LMHead", f32Of(t, &w.LMHead, "LMHead"), ts["lm_head.weight"].data)
	const inter, hidden = 40, 24
	for l := range 2 {
		p := pre + "layers." + string(rune('0'+l)) + "."
		lw := w.Layers[l]
		eq("PreAttnNorm", lw.PreAttnNorm, ts[p+"input_layernorm.weight"].data)
		eq("PostAttnNorm (post_self_attn_layernorm)", lw.PostAttnNorm, ts[p+"post_self_attn_layernorm.weight"].data)
		eq("PreMLPNorm (post_attention_layernorm)", lw.PreMLPNorm, ts[p+"post_attention_layernorm.weight"].data)
		eq("PostMLPNorm", lw.PostMLPNorm, ts[p+"post_mlp_layernorm.weight"].data)
		eq("QProj", f32Of(t, &lw.QProj, "QProj"), ts[p+"self_attn.q_proj.weight"].data)
		eq("KProj", f32Of(t, &lw.KProj, "KProj"), ts[p+"self_attn.k_proj.weight"].data)
		eq("VProj", f32Of(t, &lw.VProj, "VProj"), ts[p+"self_attn.v_proj.weight"].data)
		eq("OProj", f32Of(t, &lw.OProj, "OProj"), ts[p+"self_attn.o_proj.weight"].data)
		eq("DownProj", f32Of(t, &lw.DownProj, "DownProj"), ts[p+"mlp.down_proj.weight"].data)
		gu := ts[p+"mlp.gate_up_proj.weight"].data
		eq("GateProj (rows [0,inter) of gate_up_proj)", f32Of(t, &lw.GateProj, "GateProj"), gu[:inter*hidden])
		eq("UpProj (rows [inter,2*inter) of gate_up_proj)", f32Of(t, &lw.UpProj, "UpProj"), gu[inter*hidden:])
		if lw.QNorm != nil || lw.KNorm != nil || lw.QBias != nil || lw.KBias != nil || lw.VBias != nil || lw.OBias != nil {
			t.Errorf("layer %d carries a QK-norm or a bias the family does not have", l)
		}
	}
	// The four norms are pairwise different, so the equalities above cannot hold under a swap.
	l0 := w.Layers[0]
	if slices.Equal(l0.PostAttnNorm, l0.PreMLPNorm) || slices.Equal(l0.PreAttnNorm, l0.PostMLPNorm) {
		t.Fatalf("degenerate fixture: two of the four norms hold identical values")
	}
	// The forward over the loaded model is finite: nothing from the NaN-poisoned MTP block got in.
	cache := m.NewCache(8)
	var logits []float32
	for i, id := range []int{1, 2, 3, 4, 5, 6} {
		if logits, err = m.forward(id, cache); err != nil {
			t.Fatalf("forward[%d]: %v", i, err)
		}
	}
	for i, v := range logits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("logit %d = %v: poisoned tensors reached the forward", i, v)
		}
	}
}

// TestGlmOcr_mtpLayerAbsentIsFine: a checkpoint without the MTP layer (a re-save that dropped it)
// loads identically — the skip is by never asking, not by tolerating a presence.
func TestGlmOcr_mtpLayerAbsentIsFine(t *testing.T) {
	dir, ts := glmSynth(t)
	for k := range ts {
		if strings.Contains(k, "layers.2.") || strings.Contains(k, "layers.9.") || strings.Contains(k, "model.visual.") {
			delete(ts, k)
		}
	}
	writeSafetensors(t, filepath.Join(dir, "model.safetensors"), ts)
	m, err := Load(dir, Options{Quant: "f32"})
	if err != nil {
		t.Fatalf("Load without the MTP layer: %v", err)
	}
	m.Close()
}

// TestGlmOcr_missingLayerIsLoud: dropping a REAL layer's tensor fails the load by name, not silently.
func TestGlmOcr_missingLayerIsLoud(t *testing.T) {
	dir, ts := glmSynth(t)
	delete(ts, "model.language_model.layers.1.post_self_attn_layernorm.weight")
	writeSafetensors(t, filepath.Join(dir, "model.safetensors"), ts)
	if _, err := Load(dir, Options{Quant: "f32"}); err == nil || !strings.Contains(err.Error(), "post_self_attn_layernorm") {
		t.Fatalf("err = %v, want a load failure naming post_self_attn_layernorm", err)
	}
}

// ---- rotary pin ---------------------------------------------------------------------------------

type glmRopeGolden struct {
	HeadDim      int           `json:"head_dim"`
	Heads        int           `json:"heads"`
	MRopeSection []int         `json:"mrope_section"`
	RopeTheta    float64       `json:"rope_theta"`
	Positions    [][3]int      `json:"positions"`
	QIn          [][][]float32 `json:"q_in"`
	QOut         [][][]float32 `json:"q_out"`
}

func readGlmRopeGolden(t *testing.T) glmRopeGolden {
	t.Helper()
	raw, err := os.ReadFile("../testdata/glm_ocr_rope_golden.json")
	if errors.Is(err, fs.ErrNotExist) {
		t.Skip("no golden — run scripts/pin_glm_ocr_tiny.py")
	}
	if err != nil {
		t.Fatal(err)
	}
	var g glmRopeGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	return g
}

func flatHeads(row [][]float32) []float32 {
	var v []float32
	for _, h := range row {
		v = append(v, h...)
	}
	return v
}

func maxAbsDiff(a, b []float32) float64 {
	var m float64
	for i := range a {
		if d := math.Abs(float64(a[i]) - float64(b[i])); d > m {
			m = d
		}
	}
	return m
}

// TestApplyMRoPEPairwise_matchesHF pins the rotation against HF's actual GlmOcrTextRotaryEmbedding
// cos/sin (contiguous t/h/w blocks over freqs.split([16,24,24]), cat(freqs, freqs)) and
// apply_rotary_pos_emb (first half of cos/sin, repeat_interleave(2), rotate_half_llm), at the real
// geometry (head_dim 128), on 32 positions mixing plain text with two image-style blocks (t = base,
// h = base + row, w = base + col, one at base 5 and one at base 200). Same way
// mropeComponentInterleaved was pinned: ground truth from the reference implementation, not a
// transcription of the Go.
//
// HF builds cos/sin in float32 (a float32 inv_freq and a float32 position matmul); goinfer in float64.
// The tolerance (1e-4 absolute on unit-normal inputs) sits above that rounding at positions <= 251
// and far below what any misrouted component produces (checked below).
func TestApplyMRoPEPairwise_matchesHF(t *testing.T) {
	g := readGlmRopeGolden(t)
	inv := computeInvFreq(g.RopeTheta, g.HeadDim, nil)
	if len(inv) != 64 || !reflect.DeepEqual(g.MRopeSection, []int{16, 24, 24}) {
		t.Fatalf("golden geometry: %d freqs, section %v", len(inv), g.MRopeSection)
	}
	var worst float64
	var imageRows int
	for s, pos := range g.Positions {
		q := flatHeads(g.QIn[s])
		applyMRoPEPairwise(q, g.Heads, g.HeadDim, pos, g.MRopeSection, inv, 1)
		d := maxAbsDiff(q, flatHeads(g.QOut[s]))
		if d > worst {
			worst = d
		}
		if d > 1e-4 {
			t.Errorf("position %d %v: max|diff| vs HF = %.3e", s, pos, d)
		}
		if pos[0] != pos[1] || pos[1] != pos[2] {
			imageRows++
		}
	}
	t.Logf("applyMRoPEPairwise vs HF: worst max|diff| %.3e over %d positions (%d with unequal t/h/w components)",
		worst, len(g.Positions), imageRows)
	if imageRows < 20 {
		t.Fatalf("golden has only %d image-block rows; the pin needs t, h and w to differ", imageRows)
	}

	// The pin can fail: the three things this function must NOT be differ from HF by far more than the
	// tolerance on the image rows.
	wrong := map[string]func(q []float32, pos [3]int){
		"NeoX m-RoPE (applyMRoPE, contiguous)": func(q []float32, pos [3]int) {
			applyMRoPE(q, g.Heads, g.HeadDim, pos, g.MRopeSection, inv, 1, false)
		},
		"pairwise at the temporal position only": func(q []float32, pos [3]int) {
			applyRoPEInterleaved(q, g.Heads, g.HeadDim, pos[0], inv, 1)
		},
		"pairwise with Qwen3-VL's strided layout": func(q []float32, pos [3]int) {
			// reference formulation of the strided layout through applyMRoPEPairwise's own loop is not
			// exposed; emulate it by rotating with per-index components directly
			half := len(inv)
			for h := range g.Heads {
				v := q[h*g.HeadDim : (h+1)*g.HeadDim]
				for d := range half {
					th := float64(pos[mropeComponentInterleaved(d, g.MRopeSection)]) * inv[d]
					c, s := math.Cos(th), math.Sin(th)
					x1, x2 := float64(v[2*d]), float64(v[2*d+1])
					v[2*d], v[2*d+1] = float32(x1*c-x2*s), float32(x2*c+x1*s)
				}
			}
		},
	}
	for name, fn := range wrong {
		var wmax float64
		for s, pos := range g.Positions {
			if pos[0] == pos[1] && pos[1] == pos[2] {
				continue
			}
			q := flatHeads(g.QIn[s])
			fn(q, pos)
			if d := maxAbsDiff(q, flatHeads(g.QOut[s])); d > wmax {
				wmax = d
			}
		}
		t.Logf("  (control) %s: max|diff| vs HF on image rows = %.3e", name, wmax)
		if wmax < 1e-2 {
			t.Errorf("control %q is within %.3e of HF on the image rows: the golden does not discriminate it", name, wmax)
		}
	}
}

// TestApplyMRoPEPairwise_textIsInterleaved: with equal positions the function IS applyRoPEInterleaved,
// bit for bit (text tokens must not change at all), for every head_dim / section the family uses.
func TestApplyMRoPEPairwise_textIsInterleaved(t *testing.T) {
	for _, tc := range []struct {
		hd      int
		section []int
	}{{128, []int{16, 24, 24}}, {32, []int{4, 6, 6}}, {16, []int{2, 3, 3}}} {
		inv := computeInvFreq(10000, tc.hd, nil)
		for _, p := range []int{0, 1, 7, 300, 4096} {
			for _, heads := range []int{1, 3} {
				vec := make([]float32, heads*tc.hd)
				for i := range vec {
					vec[i] = float32(math.Sin(float64(i)*0.91 + float64(p)))
				}
				a := slices.Clone(vec)
				b := slices.Clone(vec)
				applyMRoPEPairwise(a, heads, tc.hd, [3]int{p, p, p}, tc.section, inv, 1)
				applyRoPEInterleaved(b, heads, tc.hd, p, inv, 1)
				if !slices.Equal(a, b) {
					t.Fatalf("hd %d pos %d heads %d: text rotation differs from applyRoPEInterleaved (max|diff| %.3e)", tc.hd, p, heads, maxAbsDiff(a, b))
				}
			}
		}
	}
}

// TestRopeAt_dispatch pins which rotation ropeAt picks, so the Qwen paths provably keep theirs.
func TestRopeAt_dispatch(t *testing.T) {
	const hd = 32
	inv := computeInvFreq(10000, hd, nil)
	section := []int{4, 6, 6}
	mpos := [][3]int{{0, 0, 0}, {1, 1, 1}, {2, 2, 5}, {2, 3, 2}} // an image-style block at rows 2..3
	vec := func() []float32 {
		v := make([]float32, 2*hd)
		for i := range v {
			v[i] = float32(math.Cos(float64(i)*0.53) + 0.2)
		}
		return v
	}
	run := func(seq int, interleave, mInter bool) []float32 {
		v := vec()
		ropeAt(v, 2, hd, seq, inv, 1, section, mpos, -3, interleave, mInter)
		return v
	}
	want := func(f func(v []float32)) []float32 { v := vec(); f(v); return v }

	// pairwise + contiguous (GLM-OCR) -> applyMRoPEPairwise, inside the block
	if got, w := run(2, true, false), want(func(v []float32) { applyMRoPEPairwise(v, 2, hd, mpos[2], section, inv, 1) }); !slices.Equal(got, w) {
		t.Errorf("interleave && !mropeInterleaved inside the block did not reach applyMRoPEPairwise")
	}
	// NeoX m-RoPE (Qwen2.5-VL) -> applyMRoPE, bit-identical to calling it directly
	if got, w := run(2, false, false), want(func(v []float32) { applyMRoPE(v, 2, hd, mpos[2], section, inv, 1, false) }); !slices.Equal(got, w) {
		t.Errorf("Qwen2.5-VL path (interleave=false) changed: not applyMRoPE")
	}
	// Qwen3-VL strided layout, no pairwise -> applyMRoPE(interleaved)
	if got, w := run(3, false, true), want(func(v []float32) { applyMRoPE(v, 2, hd, mpos[3], section, inv, 1, true) }); !slices.Equal(got, w) {
		t.Errorf("Qwen3-VL path changed: not applyMRoPE(interleaved)")
	}
	// the unregistered combination keeps the old behaviour
	if got, w := run(3, true, true), want(func(v []float32) { applyMRoPE(v, 2, hd, mpos[3], section, inv, 1, true) }); !slices.Equal(got, w) {
		t.Errorf("pairwise + strided layout changed behaviour")
	}
	// past the block: scalar rotation at seqPos+delta, pairwise when the flag is set
	if got, w := run(6, true, false), want(func(v []float32) { applyRoPEInterleaved(v, 2, hd, 6-3, inv, 1) }); !slices.Equal(got, w) {
		t.Errorf("decode past the block is not pairwise scalar RoPE at seqPos+delta")
	}
	if got, w := run(6, false, false), want(func(v []float32) { applyRoPE(v, 2, hd, 6-3, inv, 1) }); !slices.Equal(got, w) {
		t.Errorf("Qwen decode past the block changed")
	}
	// no m-RoPE positions at all: scalar, pairwise per the flag
	v := vec()
	ropeAt(v, 2, hd, 9, inv, 1, section, nil, 0, true, false)
	if w := want(func(v []float32) { applyRoPEInterleaved(v, 2, hd, 9, inv, 1) }); !slices.Equal(v, w) {
		t.Errorf("text-only path is not pairwise scalar RoPE")
	}
}

// ---- forward parity -----------------------------------------------------------------------------

type glmLogitGolden struct {
	PromptIDs       []int     `json:"prompt_ids"`
	Argmax          int       `json:"argmax"`
	LastLogits      []float64 `json:"last_logits"`
	NNew            int       `json:"n_new"`
	ContinuationIDs []int     `json:"continuation_ids"`
}

func cosMaxAbs(got []float32, want []float64) (cos, maxAbs float64) {
	var dot, na, nb float64
	for i, w := range want {
		a := float64(got[i])
		if d := math.Abs(a - w); d > maxAbs {
			maxAbs = d
		}
		dot += a * w
		na += a * a
		nb += w * w
	}
	return dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-12), maxAbs
}

func requireGlmTiny(t *testing.T) {
	t.Helper()
	if _, err := os.Stat(glmOcrTinyCkpt); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no tiny checkpoint (%s) — run scripts/pin_glm_ocr_tiny.py", glmOcrTinyCkpt)
	}
}

// TestGlmOcr_forwardParity is the end-to-end text-only gate against the HF GlmOcrForConditionalGeneration
// oracle (scripts/pin_glm_ocr_tiny.py): f32 weights, a 48-token prompt, so a TIGHT gate — last-token
// argmax exact, full-vocab logits by cosine and max|diff|, greedy continuation identical. The fixture
// also carries a fake MTP layer at layers.3 (loaded NaN-free but unrelated), head_dim 32 against
// hidden/heads 12, GQA, and four distinct non-trivial norm weights per layer.
func TestGlmOcr_forwardParity(t *testing.T) {
	raw, err := os.ReadFile("../testdata/glm_ocr_tiny_golden.json")
	if errors.Is(err, fs.ErrNotExist) {
		t.Skip("no golden — run scripts/pin_glm_ocr_tiny.py")
	}
	if err != nil {
		t.Fatal(err)
	}
	var g glmLogitGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	requireGlmTiny(t)
	m, err := Load(glmOcrTinyCkpt, Options{Quant: "f32"})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()
	if n := len(m.w.Layers); n != 3 {
		t.Fatalf("loaded %d layers, want 3", n)
	}
	if len(g.PromptIDs) < 40 {
		t.Fatalf("golden prompt is %d tokens; the rotation angles need >= 40", len(g.PromptIDs))
	}

	// Sequential decode path.
	cache := m.NewCache(len(g.PromptIDs) + g.NNew)
	for _, id := range g.PromptIDs[:len(g.PromptIDs)-1] {
		if _, err := m.runLayers(id, cache); err != nil {
			t.Fatalf("prefill runLayers: %v", err)
		}
	}
	logits, err := m.forward(g.PromptIDs[len(g.PromptIDs)-1], cache)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	if len(logits) != len(g.LastLogits) {
		t.Fatalf("got %d logits, want %d", len(logits), len(g.LastLogits))
	}
	if got := argmax(logits); got != g.Argmax {
		t.Errorf("argmax = %d (logit %.4f), want %d (golden logit %.4f)", got, logits[got], g.Argmax, logits[g.Argmax])
	}
	cos, maxAbs := cosMaxAbs(logits, g.LastLogits)
	t.Logf("sequential last-logit parity vs HF: cosine=%.9f max|diff|=%.3e over %d vocab, argmax %d", cos, maxAbs, len(g.LastLogits), argmax(logits))
	if cos < 0.99999 {
		t.Errorf("cosine %.9f < 0.99999", cos)
	}
	if maxAbs > 1e-3 {
		t.Errorf("max|diff| %.3e > 1e-3", maxAbs)
	}

	// Batched prefill (forwardN): the same logits.
	rows, err := m.forwardN(context.Background(), g.PromptIDs, m.NewCache(len(g.PromptIDs)))
	if err != nil {
		t.Fatalf("forwardN: %v", err)
	}
	bcos, bmax := cosMaxAbs(rows[len(rows)-1], g.LastLogits)
	t.Logf("batched last-logit parity vs HF: cosine=%.9f max|diff|=%.3e", bcos, bmax)
	if bcos < 0.99999 || bmax > 1e-3 {
		t.Errorf("batched prefill: cosine %.9f max|diff| %.3e", bcos, bmax)
	}

	// Greedy continuation must be identical (argmax at every step).
	cur := slices.Clone(g.PromptIDs)
	var cont []int
	for range g.NNew {
		c2 := m.NewCache(len(cur))
		for _, id := range cur[:len(cur)-1] {
			if _, err := m.runLayers(id, c2); err != nil {
				t.Fatalf("continuation runLayers: %v", err)
			}
		}
		lg, err := m.forward(cur[len(cur)-1], c2)
		if err != nil {
			t.Fatalf("continuation forward: %v", err)
		}
		nxt := argmax(lg)
		cont = append(cont, nxt)
		cur = append(cur, nxt)
	}
	t.Logf("continuation: got %v | want %v", cont, g.ContinuationIDs)
	if !slices.Equal(cont, g.ContinuationIDs) {
		t.Errorf("continuation = %v, want %v", cont, g.ContinuationIDs)
	}
	emitParityRow(t, "glm_ocr", "tiny-golden", "HF f32 (glm-ocr-tiny seeded fixture: GlmOcrForConditionalGeneration text-only, head_dim 32 != hidden/heads, GQA, fused gate_up, pairwise m-RoPE config, MTP layer skipped)", 100.0, cos, cos)
}

type glmMropeGolden struct {
	glmLogitGolden
	ImageToken int         `json:"image_token_id"`
	ImageStart int         `json:"image_start"`
	NImage     int         `json:"n_image_tokens"`
	Grid       [3]int      `json:"grid_thw"`
	Merge      int         `json:"merge"`
	Positions  [][3]int    `json:"positions"`
	Features   [][]float32 `json:"image_features"`
	ScalarCos  float64     `json:"scalar_position_logits_cosine"`
}

// TestGlmOcr_mropeParity drives the pairwise m-RoPE THROUGH its caller: a prompt with an
// image-placeholder run whose embeddings are random features, positions from the REAL
// mropePositions (checked against the positions HF's get_rope_index rule produced), the generic
// batched prefill (prefillLogitsQwenVL: ropeAt reads cache.mropePos), then greedy decode past the
// image on the sequential path (cache.mropeDelta resumes the scalar position). Compared with HF's
// text model run on inputs_embeds + the same 3-D position ids. A unit test of applyMRoPEPairwise
// proves the function; only this proves the forward calls it with the right arguments.
func TestGlmOcr_mropeParity(t *testing.T) {
	raw, err := os.ReadFile("../testdata/glm_ocr_tiny_mrope_golden.json")
	if errors.Is(err, fs.ErrNotExist) {
		t.Skip("no golden — run scripts/pin_glm_ocr_tiny.py")
	}
	if err != nil {
		t.Fatal(err)
	}
	var g glmMropeGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	requireGlmTiny(t)
	m, err := Load(glmOcrTinyCkpt, Options{Quant: "f32"})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()

	pos, err := mropePositions(g.PromptIDs, g.ImageToken, [][3]int{g.Grid}, g.Merge)
	if err != nil {
		t.Fatalf("mropePositions: %v", err)
	}
	if !reflect.DeepEqual(pos, g.Positions) {
		t.Fatalf("mropePositions != the positions HF's rule produced:\n got  %v\n want %v", pos, g.Positions)
	}
	hidden := m.w.arch.HiddenDim
	feats := make([]float32, 0, g.NImage*hidden)
	for _, row := range g.Features {
		feats = append(feats, row...)
	}
	cache := m.NewCache(len(g.PromptIDs) + g.NNew)
	logits, err := m.prefillLogitsQwenVL(context.Background(), g.PromptIDs, feats, g.ImageStart, g.NImage, pos, cache)
	if err != nil {
		t.Fatalf("prefillLogitsQwenVL: %v", err)
	}
	cos, maxAbs := cosMaxAbs(logits, g.LastLogits)
	t.Logf("m-RoPE prefill last-logit parity vs HF: cosine=%.9f max|diff|=%.3e, argmax %d (want %d); scalar-position cosine would be %.3f",
		cos, maxAbs, argmax(logits), g.Argmax, g.ScalarCos)
	if argmax(logits) != g.Argmax {
		t.Errorf("argmax = %d, want %d", argmax(logits), g.Argmax)
	}
	if cos < 0.99999 || maxAbs > 1e-3 {
		t.Errorf("cosine %.9f max|diff| %.3e", cos, maxAbs)
	}
	if g.ScalarCos > 0.9 {
		t.Fatalf("golden is insensitive to the m-RoPE positions (scalar-position cosine %.3f)", g.ScalarCos)
	}

	// Decode past the image: the scalar position resumes at seqPos + mropeDelta.
	var cont []int
	next := argmax(logits)
	for i := range g.NNew {
		cont = append(cont, next)
		if i == g.NNew-1 {
			break
		}
		lg, err := m.forward(next, cache)
		if err != nil {
			t.Fatalf("decode forward: %v", err)
		}
		next = argmax(lg)
	}
	t.Logf("continuation past the image: got %v | want %v", cont, g.ContinuationIDs)
	if !slices.Equal(cont, g.ContinuationIDs) {
		t.Errorf("continuation = %v, want %v", cont, g.ContinuationIDs)
	}
}

// TestGlmOcr_generateQwenVL_cpuOnly: GenerateQwenVL, the entry point serve calls for an image turn, on a CPU-only load of
// glm_ocr (m.resident == nil), through the SAME tiny image fixture and HF continuation as TestGlmOcr_mropeParity. The O1
// review flagged that GenerateQwenVL's resident branches had never been checked for a nil resident. Every one of them sits
// behind a type assertion on m.resident (ok is false on a nil interface) or behind tryClaimResident, so a CPU load takes the
// CPU prefill + CPU decode and never touches a resident — this proves it by running the whole turn, twice with the same
// image hash (the second call is where a reuse branch would be reached if it were reachable), and checks the tokens are
// HF's and that no resident flag is set.
func TestGlmOcr_generateQwenVL_cpuOnly(t *testing.T) {
	raw, err := os.ReadFile("../testdata/glm_ocr_tiny_mrope_golden.json")
	if errors.Is(err, fs.ErrNotExist) {
		t.Skip("no golden — run scripts/pin_glm_ocr_tiny.py")
	}
	if err != nil {
		t.Fatal(err)
	}
	var g glmMropeGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	requireGlmTiny(t)
	m, err := Load(glmOcrTinyCkpt, Options{Quant: "f32"})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()
	if m.resident != nil || m.ResidentActive() {
		t.Fatalf("a CPU-only load has a resident (%v): this test would not exercise the nil-resident path", m.DecodePath())
	}
	hidden := m.w.arch.HiddenDim
	feats := make([]float32, 0, g.NImage*hidden)
	for _, row := range g.Features {
		feats = append(feats, row...)
	}
	for round := range 2 {
		var calls int
		stream, gen := m.GenerateQwenVL(context.Background(), g.PromptIDs, g.ImageStart, g.NImage, 0xfeedface,
			func() ([]float32, error) { calls++; return feats, nil }, [][3]int{g.Grid}, g.Merge, g.ImageToken, g.NNew, SamplingParams{Temperature: 0})
		var got []int
		for id := range stream {
			got = append(got, id)
		}
		if err := gen.Err(); err != nil {
			t.Fatalf("round %d: %v", round, err)
		}
		if gen.ImgPrefillResident || gen.PrefillReused != 0 {
			t.Errorf("round %d: a resident flag is set on a CPU-only model (ImgPrefillResident %v, PrefillReused %d)", round, gen.ImgPrefillResident, gen.PrefillReused)
		}
		if calls != 1 {
			t.Errorf("round %d: the lazy features closure ran %d times, want 1 (no reuse is possible without a resident)", round, calls)
		}
		if !slices.Equal(got, g.ContinuationIDs) {
			t.Errorf("round %d: tokens %v, HF continuation %v", round, got, g.ContinuationIDs)
		}
	}
}
