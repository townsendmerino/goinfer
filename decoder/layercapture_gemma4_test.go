//go:build goinfer_testhooks

package decoder

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/runtime/program"
	"github.com/townsendmerino/goinfer/internal/testgate"
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
	arch := m.w.Arch
	if arch.Gemma4 == nil || arch.Gemma4.HiddenSizePerLayerInput == 0 {
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
			n    int // the row width
		}{{"Ctx", c.Ctx, arch.NumHeads * arch.HeadDimAt(l)}, {"Attn", c.Attn, hidden}, {"MLPPre", c.MLPPre, hidden},
			{"MLP", c.MLP, hidden}, {"PLE", c.PLE, hidden}, {"Out", c.Out, hidden}} {
			if len(r.rows) != arch.NumLayers || len(r.rows[l]) == 0 {
				return fmt.Errorf("layer %d: %s not captured (%d layers)", l, r.name, len(r.rows))
			}
			if len(r.rows[l]) != r.n {
				return fmt.Errorf("layer %d: %s is %d wide, want %d", l, r.name, len(r.rows[l]), r.n)
			}
		}
		if err := checkCaptureValues(m, l, c); err != nil {
			return err
		}
		h := slices.Clone(hIn)
		addResidual(h, c.Attn[l])
		addResidual(h, c.MLP[l])
		addResidual(h, c.PLE[l])
		if s := c.Scalar[l]; s != 0 {
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
	if !sameBits(m.logitsFromHidden(slices.Clone(c.Out[arch.NumLayers-1]), cache), lg) {
		return fmt.Errorf("the last layer's captured output through the head is not the logits")
	}

	lp, err := m.forward(ids[0], plain)
	if err != nil {
		return err
	}
	lp = slices.Clone(lp)
	lc, err := m.forward(ids[0], cache)
	if err != nil {
		return err
	}
	if !sameBits(lp, lc) {
		return fmt.Errorf("the capture changed the stream: the next token's logits differ")
	}
	return nil
}

// checkCaptureValues holds layer l's context and pre-norm MLP output to the contributions they produce: the context through
// the output projection and the post-attention norm is Attn, and MLPPre through the post-MLP norm is MLP, bit for bit.
func checkCaptureValues(m *Model, l int, c LayerCaptureForTest) error {
	lw := &m.w.Layers[l]
	hidden := m.w.Arch.HiddenDim
	post := func(slot program.NormSlot, src []float32) []float32 {
		for _, o := range m.program().Layers[l] {
			if o.Kind == opNorm && o.Norm.Slot == slot {
				dst := make([]float32, hidden)
				w, b := lw.NormWeights(o.Norm)
				applyNorm(o.Norm, dst, src, w, b, hidden)
				return dst
			}
		}
		return nil
	}
	proj := make([]float32, hidden)
	matmul(m.be, &lw.OProj, c.Ctx[l], proj, 1)
	if !sameBits(post(slotPostAttn, proj), c.Attn[l]) {
		return fmt.Errorf("layer %d: Ctx through the output projection and post-attention norm is not Attn", l)
	}
	if !sameBits(post(slotPostMLP, c.MLPPre[l]), c.MLP[l]) {
		return fmt.Errorf("layer %d: MLPPre through the post-MLP norm is not MLP", l)
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
			t.Logf("%s: %d layers rebuilt from their captures", name, m.w.Arch.NumLayers)
		}
	}
}

// TestLayerCapture_plantsGoRed: each planted capture defect turns checkLayerCapture red with its own message.
func TestLayerCapture_plantsGoRed(t *testing.T) {
	m := loadTinyGemma4GGUF(t, g4EModelNoShare)
	global := m.w.Arch.NumLayers - 1 // tinyGemma4GGUF's one global layer
	for _, pl := range []struct {
		name  string
		plant seamPlant
		want  string
	}{
		{"layer 1's PLE capture skipped", plantSkipPLECapture, "layer 1: PLE not captured"},
		{"layer 1's context before attention", plantCtxBeforeAttend, "layer 1: Ctx through the output projection"},
		{"the global layer's context at the model-level width", plantCtxNarrow, fmt.Sprintf("layer %d: Ctx is", global)},
		{"layer 1's MLP output after its post-norm", plantMLPPreAfterNorm, "layer 1: MLPPre through the post-MLP norm"},
	} {
		testgate.SetForTest(t, &seamPlantForTest, pl.plant)
		err := checkLayerCapture(t, m, layerCapturePrompt)
		testgate.SetForTest(t, &seamPlantForTest, plantNone)
		if err == nil || !strings.Contains(err.Error(), pl.want) {
			t.Errorf("%s: not seen (%v)", pl.name, err)
		}
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

// TestLayerInputs_isolateEachLayer holds SetLayerInputsForTest to its contract over three tokens: feeding every layer the input
// it would have had anyway (the previous layer's captured output) is bit-identical to a plain forward, logits and every layer's
// output, so the seam does not perturb; and replacing one layer's input with a different vector changes that layer's output and
// the logits, so the seam is not ignored. The second check fails if walkToken stops reading layerInputs.
func TestLayerInputs_isolateEachLayer(t *testing.T) {
	for _, dir := range []string{"../testdata/cohere2-tiny", "../testdata/llama-tiny"} {
		t.Run(filepath.Base(dir), func(t *testing.T) {
			if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
				t.Fatalf("the committed fixture %s: %v", dir, err)
			}
			m, err := Load(dir, Options{})
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			nL := m.w.Arch.NumLayers
			plain, fed := m.NewCache(8), m.NewCache(8)
			for _, tok := range []int{1, 7, 42} {
				want, lc, err := m.ForwardLayerCaptureLogitsForTest(tok, plain)
				if err != nil {
					t.Fatal(err)
				}
				in := make([][]float32, nL)
				for l := 1; l < nL; l++ {
					in[l] = lc.Out[l-1]
				}
				fed.SetLayerInputsForTest(in)
				got, gc, err := m.ForwardLayerCaptureLogitsForTest(tok, fed)
				fed.SetLayerInputsForTest(nil)
				if err != nil {
					t.Fatal(err)
				}
				if !slices.Equal(want, got) {
					t.Fatalf("token %d: the logits moved when every layer was fed its own input", tok)
				}
				for l := range nL {
					if !slices.Equal(lc.Out[l], gc.Out[l]) {
						t.Fatalf("token %d layer %d: the output moved when every layer was fed its own input", tok, l)
					}
				}
			}
			probe := m.NewCache(8)
			ref, lc, err := m.ForwardLayerCaptureLogitsForTest(1, m.NewCache(8))
			if err != nil {
				t.Fatal(err)
			}
			last := nL - 1
			other := slices.Clone(lc.Out[0])
			for j := range other {
				other[j] = -other[j]
			}
			in := make([][]float32, nL)
			in[last] = other
			probe.SetLayerInputsForTest(in)
			got, pc, err := m.ForwardLayerCaptureLogitsForTest(1, probe)
			if err != nil {
				t.Fatal(err)
			}
			if slices.Equal(ref, got) || slices.Equal(lc.Out[last], pc.Out[last]) {
				t.Errorf("replacing layer %d's input changed nothing: the forward ignores SetLayerInputsForTest", last)
			}
		})
	}
}
