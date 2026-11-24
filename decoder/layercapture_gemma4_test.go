//go:build goinfer_testhooks

package decoder

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// loadTinyGemma4GGUF writes a synthetic Gemma 4 GGUF of variant v and loads it on the CPU at f32.
func loadTinyGemma4GGUF(t *testing.T, v g4Variant) *Model {
	t.Helper()
	path := filepath.Join(t.TempDir(), "gemma4-"+v.name+".gguf")
	if err := os.WriteFile(path, tinyGemma4GGUF(v), 0o644); err != nil {
		t.Fatal(err)
	}
	m, err := Load(path, Options{Backend: "cpu", Quant: "f32"})
	if err != nil {
		t.Fatalf("Load %s: %v", v.name, err)
	}
	t.Cleanup(func() { m.Close() })
	return m
}

// g4EModelNoShare is the synthetic E2B-shaped GGUF (PLE, layer scalars) without cross-layer KV sharing: tinyGemma4GGUF makes
// only its last layer global, so a shared last layer would have no global layer to read its KV from.
var g4EModelNoShare = g4Variant{name: "e2b-noshare", ple: true, layers: 4, ffnGlobal: 256}

// gemma4EModelsForLayerCapture is the E-model fixtures the layer capture is checked on: g4EModelNoShare (always present), and
// the safetensors tiny E-model, which shares KV, when it is on disk (gitignored).
func gemma4EModelsForLayerCapture(t *testing.T) map[string]*Model {
	ms := map[string]*Model{"e2b-gguf": loadTinyGemma4GGUF(t, g4EModelNoShare)}
	if _, err := os.Stat(gemma4EModelDir); errors.Is(err, fs.ErrNotExist) {
		t.Logf("%s NOT RUN: the fixture is not present (a skip is not a pass)", gemma4EModelDir)
		return ms
	}
	m, err := Load(gemma4EModelDir, Options{Backend: "cpu", Quant: "f32"})
	if err != nil {
		t.Fatalf("Load %s: %v", gemma4EModelDir, err)
	}
	t.Cleanup(func() { m.Close() })
	ms["gemma4-emodel-tiny"] = m
	return ms
}

// checkLayerCapture holds ForwardLayerCaptureLogitsForTest on the last token of ids to: logits bit-identical to a plain forward;
// every layer's every capture present and the layer's output rebuilt bit for bit from the residual before it (the embedding at
// layer 0) and its captured contributions, applied as the layer applies them; each output differing from the layer's input; the
// last output through the head giving the logits; and the next token's logits unchanged by the capture.
func checkLayerCapture(t *testing.T, m *Model, ids []int) error {
	t.Helper()
	arch := m.w.arch
	if arch.gemma4 == nil || arch.gemma4.HiddenSizePerLayerInput == 0 {
		return fmt.Errorf("not a Gemma 4 E-model")
	}
	n := len(ids)
	plain := m.NewCache(n + 1)
	var want []float32
	for _, id := range ids {
		lg, err := m.forward(id, plain)
		if err != nil {
			return err
		}
		want = append(want[:0], lg...)
	}
	cache := m.NewCache(n + 1)
	for _, id := range ids[:n-1] {
		if _, err := m.forward(id, cache); err != nil {
			return err
		}
	}
	last := ids[n-1]
	lg, c, err := m.ForwardLayerCaptureLogitsForTest(last, cache)
	if err != nil {
		return err
	}
	if !sameBits(lg, want) {
		return fmt.Errorf("the capture changed the logits")
	}

	hidden := arch.HiddenDim
	hIn := make([]float32, hidden)
	m.w.Embed.Row(last, hIn)
	if s := m.program().Embed.Scale; s != 0 && s != 1 {
		for i := range hIn {
			hIn[i] *= float32(s)
		}
	}
	for l := range arch.NumLayers {
		for _, r := range []struct {
			name string
			rows [][]float32
			n    int // the row width; 0 for the context, whose width a global layer changes
		}{{"Ctx", c.Ctx, 0}, {"Attn", c.Attn, hidden}, {"MLPPre", c.MLPPre, hidden},
			{"MLP", c.MLP, hidden}, {"PLE", c.PLE, hidden}, {"Out", c.Out, hidden}} {
			if len(r.rows) != arch.NumLayers || len(r.rows[l]) == 0 || (r.n > 0 && len(r.rows[l]) != r.n) {
				return fmt.Errorf("layer %d: %s not captured (%d layers)", l, r.name, len(r.rows))
			}
		}
		h := append([]float32(nil), hIn...)
		addResidual(h, c.Attn[l])
		addResidual(h, c.MLP[l])
		addResidual(h, c.PLE[l])
		if s := m.w.Layers[l].LayerScalar; s != 0 {
			for i := range h {
				h[i] *= s
			}
		}
		if !sameBits(h, c.Out[l]) {
			return fmt.Errorf("layer %d: the output is not rebuilt from the input and the captured contributions", l)
		}
		if sameBits(c.Out[l], hIn) {
			return fmt.Errorf("layer %d: the captured output is the layer's input", l)
		}
		hIn = c.Out[l]
	}
	if !sameBits(m.logitsFromHidden(append([]float32(nil), c.Out[arch.NumLayers-1]...), cache), lg) {
		return fmt.Errorf("the last layer's captured output through the head is not the logits")
	}

	lp, err := m.forward(ids[0], plain)
	if err != nil {
		return err
	}
	lp = append([]float32(nil), lp...)
	lc, err := m.forward(ids[0], cache)
	if err != nil {
		return err
	}
	if !sameBits(lp, lc) {
		return fmt.Errorf("the capture changed the stream: the next token's logits differ")
	}
	return nil
}

var layerCapturePrompt = []int{3, 10, 17, 24, 31, 38, 45, 52}

// TestLayerCapture_gemma4EModel: the per-layer capture serves Gemma 4's E-models (the PLE branch and the layer scalar
// included), and checkLayerCapture holds on each.
func TestLayerCapture_gemma4EModel(t *testing.T) {
	for name, m := range gemma4EModelsForLayerCapture(t) {
		if err := checkLayerCapture(t, m, layerCapturePrompt); err != nil {
			t.Errorf("%s: %v", name, err)
		} else {
			t.Logf("%s: %d layers rebuilt from their captures", name, m.w.arch.NumLayers)
		}
	}
}

// TestLayerCapture_plantSkippedPLEGoesRed: with layer 1's PLE contribution left unrecorded, checkLayerCapture goes red.
func TestLayerCapture_plantSkippedPLEGoesRed(t *testing.T) {
	m := loadTinyGemma4GGUF(t, g4EModelNoShare)
	t.Cleanup(func() { seamPlantForTest = plantNone })
	seamPlantForTest = plantSkipPLECapture
	err := checkLayerCapture(t, m, layerCapturePrompt)
	seamPlantForTest = plantNone
	if err == nil || !strings.Contains(err.Error(), "layer 1: PLE not captured") {
		t.Fatalf("the skipped PLE capture was not seen: %v", err)
	}
	if err := checkLayerCapture(t, m, layerCapturePrompt); err != nil {
		t.Fatalf("unplanted: %v", err)
	}
}

// TestLayerCapture_refusesGemma4MoE: the 26B's MoE block replaces the residual with no captured part, so the capture refuses it.
func TestLayerCapture_refusesGemma4MoE(t *testing.T) {
	m := loadTinyGemma4GGUF(t, gemma4Variants[0])
	_, _, err := m.ForwardLayerCaptureLogitsForTest(layerCapturePrompt[0], m.NewCache(4))
	if err == nil || !strings.Contains(err.Error(), "MoE block") {
		t.Fatalf("not refused: %v", err)
	}
}
