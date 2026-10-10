package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"os"
	"strings"
	"testing"
)

const (
	gemma4EModelDir    = "../testdata/gemma4-emodel-tiny"
	gemma4EModelGolden = "../testdata/gemma4_emodel_tiny_golden.json"
)

// TestGemma4EModel_safetensorsParity is S1.1's gate (docs/tasks/task-multimodal-support-2026-10.md): a safetensors
// Gemma 4 E-model (Per-Layer Embeddings, cross-layer KV sharing, double-wide FFNs on the shared layers) loads and its
// CPU forward matches the HF oracle at EVERY prompt position, f32: argmax equal and cosine >= 0.99999 at each, then the
// greedy continuation byte-identical. All positions, not just the last, because the KV-shared layers only differ from
// their own-KV counterparts once there is history to attend over, and the sliding window (4) is shorter than the prompt
// (12).
func TestGemma4EModel_safetensorsParity(t *testing.T) {
	raw, err := os.ReadFile(gemma4EModelGolden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skip("no golden — run scripts/pin_gemma4_emodel_tiny.py")
	}
	if err != nil {
		t.Fatal(err)
	}
	var g struct {
		PromptIDs       []int       `json:"prompt_ids"`
		Argmax          []int       `json:"argmax"`
		Logits          [][]float64 `json:"logits"`
		NNew            int         `json:"n_new"`
		ContinuationIDs []int       `json:"continuation_ids"`
		FFNWidths       []int       `json:"ffn_widths"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(gemma4EModelDir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no tiny checkpoint (%s) — run scripts/pin_gemma4_emodel_tiny.py", gemma4EModelDir)
	}
	requireFixtureIdentity(t, gemma4EModelDir)
	m, err := Load(gemma4EModelDir, Options{Quant: "f32"})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()

	// The load really carries the E-model parts (a silently-skipped tensor would leave a zero WeightMat).
	g4 := m.w.arch.gemma4
	if g4 == nil || g4.HiddenSizePerLayerInput != 32 || g4.SharedKVLayers != 2 {
		t.Fatalf("arch: %+v, want PLE 32 and 2 shared-KV layers", g4)
	}
	if m.w.PerLayerTokenEmbed.Rows() != 256 || m.w.PerLayerModelProj.Rows() != 6*32 || len(m.w.PerLayerProjNorm) != 32 {
		t.Fatalf("model-level PLE not loaded: table %d rows, proj %d rows, norm %d", m.w.PerLayerTokenEmbed.Rows(), m.w.PerLayerModelProj.Rows(), len(m.w.PerLayerProjNorm))
	}
	for i := range m.w.Layers {
		l := &m.w.Layers[i]
		if l.PLEGate.Rows() != 32 || l.PLEProj.Rows() != m.w.arch.HiddenDim || len(l.PostPLENorm) != m.w.arch.HiddenDim {
			t.Fatalf("layer %d: PLE branch not loaded", i)
		}
		if w := m.w.arch.ffnAt(i); w != g.FFNWidths[i] {
			t.Fatalf("layer %d: ffn width %d, want %d", i, w, g.FFNWidths[i])
		}
	}

	cache := m.NewCache(len(g.PromptIDs) + g.NNew)
	worst := 1.0
	for p, id := range g.PromptIDs {
		logits, err := m.forward(id, cache)
		if err != nil {
			t.Fatalf("pos %d: forward: %v", p, err)
		}
		want := g.Logits[p]
		if len(logits) != len(want) {
			t.Fatalf("pos %d: %d logits, want %d", p, len(logits), len(want))
		}
		var dot, na, nb float64
		for i, w := range want {
			a := float64(logits[i])
			dot += a * w
			na += a * a
			nb += w * w
		}
		cos := dot / math.Sqrt(na*nb)
		worst = math.Min(worst, cos)
		if got := argmax(logits); got != g.Argmax[p] {
			t.Errorf("pos %d: argmax %d, want %d (cosine %.8f)", p, got, g.Argmax[p], cos)
		}
		if cos < 0.99999 {
			t.Errorf("pos %d: cosine %.8f < 0.99999", p, cos)
		}
	}
	t.Logf("all %d positions: worst cosine %.8f", len(g.PromptIDs), worst)

	// Greedy continuation from the cache the prompt left.
	next := g.Argmax[len(g.Argmax)-1]
	var cont []int
	for range g.NNew {
		cont = append(cont, next)
		logits, err := m.forward(next, cache)
		if err != nil {
			t.Fatal(err)
		}
		next = argmax(logits)
	}
	for i := range cont {
		if cont[i] != g.ContinuationIDs[i] {
			t.Fatalf("continuation %v, want %v", cont, g.ContinuationIDs)
		}
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
		hd, kv, err := cfg.gemma4GlobalGeometry()
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

// TestGemma4EModel_residentEmbedRow is S1.2's check: the row a GPU backend receives for an E-model is
// [h ‖ PLE inputs], ResidentEmbedLen() long, where h is the scaled token embedding and the tail matches HF's own
// per-layer inputs (get_per_layer_inputs + project_per_layer_inputs, recorded by the pin script) at every prompt
// position. The oracle is HF, not gemma4PLEInputs, so a helper that drifted from the CPU forward's old inline code
// would show here even though both goinfer paths now share it. The reuse path (a caller-owned buffer)
// must produce the same row.
func TestGemma4EModel_residentEmbedRow(t *testing.T) {
	raw, err := os.ReadFile(gemma4EModelGolden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skip("no golden — run scripts/pin_gemma4_emodel_tiny.py")
	}
	if err != nil {
		t.Fatal(err)
	}
	var g struct {
		PromptIDs []int       `json:"prompt_ids"`
		PLEInputs [][]float64 `json:"ple_inputs"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	if len(g.PLEInputs) != len(g.PromptIDs) {
		t.Fatalf("golden has %d ple_inputs rows for %d prompt ids — re-run the pin script", len(g.PLEInputs), len(g.PromptIDs))
	}
	if _, err := os.Stat(gemma4EModelDir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no tiny checkpoint (%s) — run scripts/pin_gemma4_emodel_tiny.py", gemma4EModelDir)
	}
	requireFixtureIdentity(t, gemma4EModelDir)
	m, err := Load(gemma4EModelDir, Options{Quant: "f32"})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()
	H := m.w.arch.HiddenDim
	LP := m.w.arch.NumLayers * m.Gemma4PLEDimResident()
	if LP != 6*32 || m.ResidentEmbedLen() != H+LP {
		t.Fatalf("ResidentEmbedLen %d, PLE %d; want %d + %d", m.ResidentEmbedLen(), LP, H, 6*32)
	}
	var scratch []float32
	worst := 0.0
	for p, id := range g.PromptIDs {
		row := m.embedResident(id)
		if len(row) != H+LP {
			t.Fatalf("pos %d: row length %d, want %d", p, len(row), H+LP)
		}
		h := make([]float32, H)
		m.w.Embed.Row(id, h)
		for i := range h {
			if want := h[i] * float32(m.w.arch.EmbedScale); row[i] != want {
				t.Fatalf("pos %d: row[%d] = %v, want the scaled embedding %v", p, i, row[i], want)
			}
		}
		for i, want := range g.PLEInputs[p] {
			worst = math.Max(worst, math.Abs(float64(row[H+i])-want))
		}
		scratch = m.embedResidentInto(id, scratch)
		for i := range row {
			if scratch[i] != row[i] {
				t.Fatalf("pos %d: embedResidentInto with a reused buffer differs at %d", p, i)
			}
		}
	}
	t.Logf("PLE tail vs HF over %d positions: max |diff| %.3g", len(g.PromptIDs), worst)
	if worst > 1e-4 {
		t.Errorf("PLE tail max |diff| %.3g vs HF, want <= 1e-4 (f32)", worst)
	}
}
