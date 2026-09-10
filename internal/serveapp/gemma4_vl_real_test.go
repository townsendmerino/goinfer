package serveapp

import (
	"bytes"
	"context"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/decoder"
)

// distinctTrigramRatioServeapp is decoder/gemma4_26b_real_test.go's distinctTrigramRatio,
// copied locally — that one lives in a _test.go file in a different package and can't be
// imported. Same language-agnostic degeneracy metric: |distinct 3-rune windows| / |total|.
func distinctTrigramRatioServeapp(s string) float64 {
	r := []rune(s)
	if len(r) < 3 {
		return 1
	}
	seen := make(map[string]struct{})
	total := 0
	for i := 0; i+3 <= len(r); i++ {
		seen[string(r[i:i+3])] = struct{}{}
		total++
	}
	return float64(len(seen)) / float64(total)
}

// solidColorPNG builds a small solid-color PNG in memory — no binary asset to commit, and its
// "what color is this" answer is unambiguous and mechanically checkable.
func solidColorPNG(t *testing.T, size int, c color.RGBA) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, size, size))
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			img.SetRGBA(x, y, c)
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("encode png: %v", err)
	}
	return buf.Bytes()
}

// loadGemma4VLReal loads a real Gemma 4 checkpoint's TEXT decoder from modelPath and its vision
// tower from visionDir — separate paths on purpose: the safetensors loader has no Per-Layer-
// Embedding (PLE) support yet (decoder/weights.go's deliberate refusal), so a real E2B/E4B
// (PLE>0) checkpoint's TEXT side must load via GGUF (whose loader implements PLE), while its
// vision tower — a wholly separate feature, decoder/weights.go's PLE refusal never touches it —
// loads fine from the sibling safetensors directory that carries the vision_tower.*/
// embed_vision.* tensors. Mirrors production `--model x.gguf --vision <dir>` exactly. For
// 26B-A4B (no PLE at all), modelPath and visionDir are the same safetensors directory.
func loadGemma4VLReal(t *testing.T, modelPath, visionDir string, opts decoder.Options) *loadedModel {
	t.Helper()
	requireHeavyModel(t)
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("no checkpoint at %s: %v", modelPath, err)
	}
	if _, err := os.Stat(filepath.Join(visionDir, "config.json")); err != nil {
		t.Skipf("no vision checkpoint dir at %s: %v", visionDir, err)
	}
	tk, err := loadDecoderTokenizer(modelPath)
	if err != nil {
		t.Fatalf("load tokenizer (%s): %v", modelPath, err)
	}
	m, err := decoder.Load(modelPath, opts)
	if err != nil {
		t.Fatalf("Load(%s): %v", modelPath, err)
	}
	t.Cleanup(func() { _ = m.Close() })
	mcfg := m.Config()
	lm := &loadedModel{
		tk: tk, model: m, tmpl: chat.Gemma4(),
		vocab: mcfg.VocabSize, eosIDs: mcfg.EOSIDs(), name: "gemma4-real-vl",
	}
	for _, str := range lm.tmpl.Stops().Strings {
		if id, ok := tk.TokenID(str); ok {
			lm.stopIDs = append(lm.stopIDs, id)
		}
	}
	s := &server{models: map[string]*loadedModel{lm.name: lm}}
	if err := s.loadGemma4VisionTower(visionDir, false); err != nil {
		t.Fatalf("loadGemma4VisionTower(%s): %v", visionDir, err)
	}
	return lm
}

// askAboutImage drives the REAL serving-side vision wiring end to end on real weights:
// lm.gemma4VisionPrompt (real tower forward, real preprocessing) -> lm.prepare -> lm.driveVL
// (real GenerateGemma4VL). This is the first-ever real run of this exact seam — every prior
// gemma4 VL gate this session called decoder-level functions directly with precomputed
// image_features, never gemma4VisionPrompt's own real-tower invocation (docs/multimodal.md's
// Phase B "HTTP-level integration smoke test" gap).
func askAboutImage(t *testing.T, lm *loadedModel, question string, imgData []byte, maxTokens int) string {
	t.Helper()
	turns := []chat.Turn{{Role: "user", Content: question}}
	vi, err := lm.gemma4VisionPrompt("", turns, 0, imageRef{mediaType: "image/png", data: imgData})
	if err != nil {
		t.Fatalf("gemma4VisionPrompt: %v", err)
	}
	gr, err := lm.prepare(sampling{}, vi.ids, false)
	if err != nil {
		t.Fatalf("prepare: %v", err)
	}
	gr.maxTokens = maxTokens
	var sb strings.Builder
	_, n, _, _, err := lm.driveVL(context.Background(), gr, vi, func(s string) { sb.WriteString(s) })
	if err != nil {
		t.Fatalf("driveVL: %v", err)
	}
	text := sb.String()
	if n == 0 || strings.TrimSpace(text) == "" {
		t.Fatal("no tokens generated / empty continuation")
	}
	return text
}

// TestGemma4VLReal_E2B is the real-checkpoint end-to-end gate for the E2B/E4B-class
// (use_bidirectional_attention unset, sequential causal prefill, Phase B) path: real vision
// tower, real text decoder (via GGUF, since E2B carries PLE and the safetensors loader
// doesn't implement it), real serving wiring, asking about a real (synthetic, solid-color)
// image and checking the answer names the actual color.
func TestGemma4VLReal_E2B(t *testing.T) {
	home, _ := os.UserHomeDir()
	modelPath := os.Getenv("GOINFER_GEMMA4_E2B_GGUF")
	if modelPath == "" {
		modelPath = filepath.Join(home, "models", "gemma-4-E2B_q4_0-it.gguf")
	}
	visionDir := os.Getenv("GOINFER_GEMMA4_E2B_VISION")
	if visionDir == "" {
		visionDir = filepath.Join(home, "models", "gemma-4-E2B-unq")
	}
	lm := loadGemma4VLReal(t, modelPath, visionDir, decoder.Options{})
	if got := string(lm.model.Config().UseBidirectionalAttention); got != "" {
		t.Fatalf("UseBidirectionalAttention = %q, want \"\" — this test exists specifically to "+
			"real-checkpoint-validate Phase B's sequential causal path", got)
	}

	img := solidColorPNG(t, 128, color.RGBA{R: 220, G: 30, B: 30, A: 255}) // red
	text := askAboutImage(t, lm, "What color is this image? Answer in one word.", img, 32)
	t.Logf("E2B real VL answer: %q", text)
	if low := strings.ToLower(text); !strings.Contains(low, "red") {
		t.Errorf("answer does not name the image's actual color (red): %q", text)
	}
	if r := distinctTrigramRatioServeapp(text); r < 0.5 {
		t.Errorf("answer looks degenerate (distinct-trigram %.3f < 0.5): %q", r, text)
	}
}

// TestGemma4VLReal_26BA4B is the real-checkpoint end-to-end gate for the 26B-A4B/31B-class
// (use_bidirectional_attention: "vision", batched blockwise prefill, Phase C) path — the
// first real-checkpoint proof of the batched forward, not just the tiny/scaled synthetic
// fixtures. 26B-A4B carries NO PLE (hidden_size_per_layer_input=0, confirmed directly against
// the real config — corrects docs/multimodal.md's earlier, over-generalized claim that every
// real vision-capable checkpoint has PLE), so this loads straight off safetensors, no GGUF
// needed.
func TestGemma4VLReal_26BA4B(t *testing.T) {
	home, _ := os.UserHomeDir()
	dir := os.Getenv("GOINFER_GEMMA4_26B")
	if dir == "" {
		dir = filepath.Join(home, "models", "gemma-4-26b-a4b-it")
	}
	lm := loadGemma4VLReal(t, dir, dir, decoder.Options{Quant: "int4"})
	if got := string(lm.model.Config().UseBidirectionalAttention); got != "vision" {
		t.Fatalf("UseBidirectionalAttention = %q, want \"vision\" — this test exists specifically "+
			"to real-checkpoint-validate Phase C's batched, blockwise-masked forward", got)
	}

	img := solidColorPNG(t, 128, color.RGBA{R: 30, G: 140, B: 230, A: 255}) // blue
	text := askAboutImage(t, lm, "What color is this image? Answer in one word.", img, 32)
	t.Logf("26B-A4B real VL answer: %q", text)
	if low := strings.ToLower(text); !strings.Contains(low, "blue") {
		t.Errorf("answer does not name the image's actual color (blue): %q", text)
	}
	if r := distinctTrigramRatioServeapp(text); r < 0.5 {
		t.Errorf("answer looks degenerate (distinct-trigram %.3f < 0.5): %q", r, text)
	}
}
