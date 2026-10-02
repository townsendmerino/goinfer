package multimodal

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// glmOcrPreprocessorConfig is zai-org/GLM-OCR's preprocessor_config.json at revision 2e85a628, verbatim.
const glmOcrPreprocessorConfig = `{
    "size": {"shortest_edge": 12544, "longest_edge": 9633792},
    "do_rescale": true,
    "patch_size": 14,
    "temporal_patch_size": 2,
    "merge_size": 2,
    "image_mean": [0.48145466, 0.4578275, 0.40821073],
    "image_std": [0.26862954, 0.26130258, 0.27577711],
    "image_processor_type": "Glm46VImageProcessor",
    "processor_class": "Glm46VProcessor"
}`

// glmGridCases are Glm46VImageProcessor's own grids (transformers 5.12.0, torchvision backend, the checkpoint's
// preprocessor_config.json), measured by running the real processor on blank images of these sizes: source w x h ->
// grid (t,h,w) in patch units. 6000x4000 lands exactly on the 6,144-token ceiling; 100x100 on the floor.
var glmGridCases = []struct {
	w, h int
	grid [3]int
}{
	{6000, 4000, [3]int{1, 128, 192}},
	{4000, 3000, [3]int{1, 134, 180}},
	{3000, 3000, [3]int{1, 156, 156}},
	{5000, 4000, [3]int{1, 140, 174}},
	{1700, 2200, [3]int{1, 158, 122}},
	{100, 100, [3]int{1, 8, 8}},
}

func gridFor(cfg QwenPreprocessConfig, w, h int) [3]int {
	factor := cfg.PatchSize * cfg.MergeSize
	hb, wb := qwenSmartResize(h, w, factor, cfg.MinPixels, cfg.MaxPixels)
	return [3]int{1, hb / cfg.PatchSize, wb / cfg.PatchSize}
}

// TestLoadGlmOcrPreprocessConfig_halvesPixelBounds pins the one trap in this loader (task doc §1, O0): the file's
// size.shortest_edge / size.longest_edge bound TWO temporal frames, so one frame's budget is half. It fails if the file's
// values reach qwenSmartResize unhalved — checked on the numbers AND on the grids HF's processor produces, and a negative
// control shows the grid check is sensitive to exactly that mistake (the unhalved config picks a different grid for the
// same image), so it cannot pass vacuously.
func TestLoadGlmOcrPreprocessConfig_halvesPixelBounds(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "preprocessor_config.json"), []byte(glmOcrPreprocessorConfig), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg, err := LoadGlmOcrPreprocessConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.MinPixels != 6272 || cfg.MaxPixels != 4816896 {
		t.Fatalf("pixel bounds %d..%d, want the file's 12544..9633792 halved: 6272..4816896", cfg.MinPixels, cfg.MaxPixels)
	}
	if cfg.MaxPixels == 9633792 || cfg.MinPixels == 12544 {
		t.Fatal("the file's pixel bounds were passed through unhalved")
	}
	if cfg.PatchSize != 14 || cfg.MergeSize != 2 || cfg.TemporalPatchSize != 2 {
		t.Fatalf("geometry patch %d merge %d temporal %d, want 14/2/2", cfg.PatchSize, cfg.MergeSize, cfg.TemporalPatchSize)
	}
	if cfg.Mean != [3]float32{0.48145466, 0.4578275, 0.40821073} || cfg.Std != [3]float32{0.26862954, 0.26130258, 0.27577711} {
		t.Fatalf("CLIP mean/std %v %v", cfg.Mean, cfg.Std)
	}
	if !cfg.FusedNormalize {
		t.Fatal("GLM-OCR's processor is the torchvision backend: FusedNormalize must be on")
	}
	// The ceiling in tokens: 6,144 image tokens (24,576 patches), not the 12,288 the unhalved bound would allow.
	if tok := QwenMergedTokens(gridFor(cfg, 6000, 4000), cfg.MergeSize); tok != 6144 {
		t.Fatalf("6000x4000 -> %d image tokens, want the 6,144 ceiling", tok)
	}
	for _, c := range glmGridCases {
		if got := gridFor(cfg, c.w, c.h); got != c.grid {
			t.Errorf("%dx%d -> grid %v, HF's processor gives %v", c.w, c.h, got, c.grid)
		}
	}

	// Negative control: the SAME config with the file's values unhalved must disagree with HF on the large cases.
	// (If it did not, the grid assertions above would not be able to see the bug this test exists for.)
	raw, err := LoadQwen3PreprocessConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	if raw.MaxPixels != 9633792 {
		t.Fatalf("control: LoadQwen3PreprocessConfig MaxPixels %d, want the file's 9633792", raw.MaxPixels)
	}
	wrong := 0
	for _, c := range glmGridCases {
		if gridFor(raw, c.w, c.h) != c.grid {
			wrong++
		}
	}
	if wrong < 3 {
		t.Fatalf("control: the unhalved bounds disagreed with HF on only %d of %d cases; the pin cannot see the halving mistake", wrong, len(glmGridCases))
	}
	t.Logf("unhalved bounds would give the wrong grid on %d of %d HF-measured sizes", wrong, len(glmGridCases))
}

// TestLoadGlmOcrPreprocessConfig_realFile checks the checkpoint's own file when it is on disk (a missing checkpoint is
// not this test's business: the literal above is the same file).
func TestLoadGlmOcrPreprocessConfig_realFile(t *testing.T) {
	dir := filepath.Join(os.Getenv("HOME"), "models", "glm-ocr")
	if _, err := os.Stat(filepath.Join(dir, "preprocessor_config.json")); err != nil {
		t.Skip("no ~/models/glm-ocr/preprocessor_config.json")
	}
	cfg, err := LoadGlmOcrPreprocessConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.MinPixels != 6272 || cfg.MaxPixels != 4816896 {
		t.Fatalf("real file: bounds %d..%d, want 6272..4816896", cfg.MinPixels, cfg.MaxPixels)
	}
}

func TestLoadGlmOcrPreprocessConfig_refusesMissingFile(t *testing.T) {
	if _, err := LoadGlmOcrPreprocessConfig(t.TempDir()); err == nil {
		t.Fatal("a directory with no preprocessor_config.json must be an error, not a default")
	}
}

func TestGlmOcrImageBlock(t *testing.T) {
	got := GlmOcrImageBlock(3)
	if got != "<|begin_of_image|><|image|><|image|><|image|><|end_of_image|>" {
		t.Fatalf("block %q", got)
	}
	if strings.Count(GlmOcrImageBlock(1656), GlmOcrImagePad) != 1656 {
		t.Fatal("pad count")
	}
}

// The instruction is the model card's, byte for byte: it is the string the model was trained to read, so a "tidied"
// punctuation mark (the colon is ASCII ':' in the card) changes the prompt. The expected side is written as escapes so a
// normalising editor cannot change both sides together.
func TestGlmOcrExtractionInstruction_isTheCards(t *testing.T) {
	const card = "请按下列JSON格式输出图中信息:"
	if GlmOcrExtractionInstruction != card {
		t.Fatalf("instruction %q, the card's is %q", GlmOcrExtractionInstruction, card)
	}
	if got := GlmOcrExtractionPrompt("{}"); got != card+"\n{}" {
		t.Fatalf("prompt %q: want the instruction, a newline, the template", got)
	}
	// And against the checkpoint's own README when it is on this machine (skipped otherwise: a clean clone has no model).
	if raw, err := os.ReadFile(filepath.Join(os.Getenv("HOME"), "models", "glm-ocr", "README.md")); err == nil {
		if !strings.Contains(string(raw), card+"\n{") {
			t.Errorf("the model card no longer contains the instruction followed by a newline and an object")
		}
	}
}

// The O5 rule: the template prompt replaces EMPTY text and a BARE task prompt, and nothing else. The replaced=false rows
// are the ones that must go red if the rule is widened to overwrite whatever the user wrote.
func TestGlmOcrExtractionText_rule(t *testing.T) {
	const tmpl = "{\n    \"total\": \"\"\n}"
	want := GlmOcrExtractionPrompt(tmpl)
	for _, in := range []string{"", "   ", "\n", GlmOcrPromptText, GlmOcrPromptTable, GlmOcrPromptFormula, "  Text Recognition:\n"} {
		got, replaced := GlmOcrExtractionText(in, tmpl)
		if !replaced || got != want {
			t.Errorf("%q: replaced=%v text=%q; want the extraction prompt", in, replaced, got)
		}
	}
	for _, in := range []string{
		GlmOcrExtractionPrompt("{\n    \"vendor\": \"\"\n}"), // the user's OWN extraction prompt, different fields
		"What is the invoice total?",
		"Text Recognition: and the total",
		"text recognition:", // case matters: only the card's exact strings are bare task prompts
	} {
		got, replaced := GlmOcrExtractionText(in, tmpl)
		if replaced || got != in {
			t.Errorf("%q: replaced=%v text=%q; the user's own text must be kept verbatim", in, replaced, got)
		}
	}
}
