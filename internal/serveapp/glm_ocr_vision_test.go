package serveapp

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/multimodal"
	"github.com/townsendmerino/goinfer/tokenizer"
)

const (
	glmPre  = `{"size":{"shortest_edge":12544,"longest_edge":9633792},"patch_size":14,"temporal_patch_size":2,"merge_size":2,"image_mean":[0.48145466,0.4578275,0.40821073],"image_std":[0.26862954,0.26130258,0.27577711]}`
	glmConf = `{"model_type":"glm_ocr","vision_config":{"depth":24}}`
)

// TestIsGlmOcrVisionDir: auto-discovery recognises a GLM-OCR checkpoint that carries a usable tower and stays SILENT (false,
// not an error) on everything else, so a text-only copy or another family never fails serve's startup.
func TestIsGlmOcrVisionDir(t *testing.T) {
	for _, tc := range []struct {
		name        string
		config, pre string
		want        bool
	}{
		{"glm_ocr with tower and preprocessor", glmConf, glmPre, true},
		{"text-only copy: no vision_config", `{"model_type":"glm_ocr"}`, glmPre, false},
		{"text-only copy: no preprocessor_config.json", glmConf, "", false},
		{"Qwen2.5-VL-style preprocessor (min_pixels keys) is not a GLM-OCR tower", glmConf, `{"min_pixels":3136,"max_pixels":100,"patch_size":14,"temporal_patch_size":2,"merge_size":2,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5]}`, false},
		{"the bare text decoder model_type", `{"model_type":"glm_ocr_text","vision_config":{"depth":24}}`, glmPre, false},
		{"another family", `{"model_type":"qwen3_5","vision_config":{"depth":12,"deepstack_visual_indexes":[]}}`, glmPre, false},
		{"no config.json", "", glmPre, false},
	} {
		if got := isGlmOcrVisionDir(writeQwen35Dir(t, tc.config, tc.pre)); got != tc.want {
			t.Errorf("%s: isGlmOcrVisionDir = %v, want %v", tc.name, got, tc.want)
		}
	}
	// And the other discoverers must not claim it: one directory, one family.
	d := writeQwen35Dir(t, glmConf, glmPre)
	if isQwen35VisionDir(d) {
		t.Error("isQwen35VisionDir claims a glm_ocr directory")
	}
}

// TestSetupGlmOcrVision: the serve-side preprocessing config carries the HALVED pixel bounds (6,272..4,816,896; the file's
// 12,544..9,633,792 bound two temporal frames) with no cap of serve's own, and setup does NOT read the tower's weights —
// this directory has none, so any load would fail — and the tower's eventual load error is reported, not swallowed.
func TestSetupGlmOcrVision(t *testing.T) {
	dir := writeQwen35Dir(t, glmConf, glmPre)
	tower, pp, err := setupGlmOcrVision(dir, false, 0)
	if err != nil {
		t.Fatal(err)
	}
	if pp.MinPixels != 6272 || pp.MaxPixels != 4816896 {
		t.Errorf("pixel bounds %d..%d, want 6272..4816896 (the file's, halved); passing them unhalved doubles the image budget", pp.MinPixels, pp.MaxPixels)
	}
	if pp.PatchSize != 14 || pp.MergeSize != 2 || pp.TemporalPatchSize != 2 || !pp.FusedNormalize {
		t.Errorf("checkpoint geometry not carried through: %+v", pp)
	}
	if tower.enc != nil || tower.err != nil {
		t.Fatal("setup touched the tower's weights: it must load lazily")
	}
	if _, err := tower.encoder(); err == nil {
		t.Error("encoder() on a directory with no weights returned no error")
	}
	if _, err2 := tower.encoder(); err2 == nil || err2.Error() != tower.err.Error() {
		t.Error("a failed load must be remembered, not retried into a different answer")
	}
	if _, _, err := setupGlmOcrVision(writeQwen35Dir(t, `{"model_type":"glm_ocr"}`, glmPre), false, 0); err == nil {
		t.Error("a text-only directory was accepted as a vision tower")
	}
}

// TestSetupGlmOcrVision_pixelCap: -vision-max-pixels (O4, owner decision 2026-10-04: the default stays the model's own 4.82 MP ceiling, the flag only lowers it).
// It is tested through setupGlmOcrVision, the one function serve's loader calls: a cap below the ceiling lowers MaxPixels and nothing else; 0 leaves the ceiling; a cap
// ABOVE the ceiling does not raise it (the tower and the context were gated for 6,144 image tokens, no more); a cap below the model's own floor is refused by name.
func TestSetupGlmOcrVision_pixelCap(t *testing.T) {
	dir := writeQwen35Dir(t, glmConf, glmPre)
	const ceiling = 4816896
	for _, tc := range []struct {
		name    string
		cap     int
		wantMax int
	}{
		{"unset keeps the ceiling", 0, ceiling},
		{"negative keeps the ceiling", -5, ceiling},
		{"2 MP lowers it", 2_000_000, 2_000_000},
		{"exactly the ceiling", ceiling, ceiling},
		{"above the ceiling is not raised", 20_000_000, ceiling},
	} {
		_, pp, err := setupGlmOcrVision(dir, false, tc.cap)
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		if pp.MaxPixels != tc.wantMax {
			t.Errorf("%s: MaxPixels %d, want %d", tc.name, pp.MaxPixels, tc.wantMax)
		}
		if pp.MinPixels != 6272 || pp.PatchSize != 14 || pp.MergeSize != 2 {
			t.Errorf("%s: the cap changed more than MaxPixels: %+v", tc.name, pp)
		}
	}
	_, _, err := setupGlmOcrVision(dir, false, 1000)
	if err == nil || !strings.Contains(err.Error(), "-vision-max-pixels") || !strings.Contains(err.Error(), "6272") {
		t.Errorf("a cap below the model's minimum must be refused naming the flag and the minimum, got %v", err)
	}
}

// TestImageFitsContext: an image whose token run cannot fit the context is refused BY NAME, before the tower, and one that
// fits is not. The tiny glm_ocr fixture's context is 512, so a 600-token image is too large; the error must say how many
// tokens the image is, what the window is, and that it is the image (never a silent truncation of the run).
func TestImageFitsContext(t *testing.T) {
	const ckpt = "../../testdata/glm-ocr-tiny"
	if _, err := os.Stat(filepath.Join(ckpt, "model.safetensors")); err != nil {
		t.Skip("no tiny glm-ocr fixture")
	}
	m, err := decoder.Load(ckpt, decoder.Options{Quant: "f32"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ctx := m.Config().MaxPositions
	if ctx != 512 {
		t.Fatalf("fixture context %d, this test assumes 512", ctx)
	}
	lm := &loadedModel{model: m, qwenMerge: 2}
	lm.qwenPP, _ = multimodal.LoadGlmOcrPreprocessConfig(writeQwen35Dir(t, glmConf, glmPre))
	if err := lm.imageFitsContext(100, [3]int{1, 20, 20}); err != nil {
		t.Errorf("a 100-token image in a 512 context was refused: %v", err)
	}
	err = lm.imageFitsContext(600, [3]int{1, 48, 50})
	if err == nil {
		t.Fatal("a 600-token image in a 512-token context was accepted")
	}
	for _, want := range []string{"image_too_large_for_context", "600 image tokens", "512"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not name %q", err, want)
		}
	}
	// Right at the edge: the run plus the template and task prompt must still leave room.
	if err := lm.imageFitsContext(512-24, [3]int{1, 10, 10}); err == nil {
		t.Error("an image that fills the context, leaving no room for its own prompt, was accepted")
	}
}

// glmOcrTestPNG is a flat 224x224 image: a 16x16-patch grid, 64 image tokens (the processor's own floor is 8x8 patches).
func glmOcrTestPNG(t *testing.T) []byte {
	t.Helper()
	im := image.NewRGBA(image.Rect(0, 0, 224, 224))
	for y := range 224 {
		for x := range 224 {
			im.Set(x, y, color.RGBA{uint8(x), uint8(y), 128, 255})
		}
	}
	var b bytes.Buffer
	if err := png.Encode(&b, im); err != nil {
		t.Fatal(err)
	}
	return b.Bytes()
}

// TestGlmOcr_visionPromptIDs builds the image prompt with serve's OWN code path (visionPrompt: the chat.GlmOCR renderer, the
// image block, the segment splice and the real tokenizer) and checks the ids against what HF's processor produced from the
// checkpoint's chat_template.jinja (O0, and again by the O3 gate on the real documents): image first, no separator before
// the task prompt, `Text Recognition:` the default for a request with no text part. No tower runs (features stay lazy).
// Skips cleanly without ~/models/glm-ocr/tokenizer.json.
func TestGlmOcr_visionPromptIDs(t *testing.T) {
	dir := filepath.Join(os.Getenv("HOME"), "models", "glm-ocr")
	if _, err := os.Stat(filepath.Join(dir, "tokenizer.json")); err != nil {
		t.Skipf("no tokenizer.json under %s", dir)
	}
	tk, err := tokenizer.Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	tmpl, err := chat.Detect(chat.Meta{ChatTemplate: tk.ChatTemplate(), HasToken: tk.Has})
	if err != nil || tmpl.Name() != "glm_ocr" {
		t.Fatalf("Detect on the checkpoint's template: %v, %v", tmpl, err)
	}
	m, err := decoder.Load("../../testdata/glm-ocr-tiny", decoder.Options{Quant: "f32"})
	if err != nil {
		t.Skipf("no tiny fixture: %v", err)
	}
	defer m.Close()
	lm := &loadedModel{tk: tk, model: m, tmpl: tmpl, glm: &glmOcrTower{dir: "unused"}, qwenMerge: 2, qwenImgTok: 59280}
	lm.qwenPP, err = multimodal.LoadGlmOcrPreprocessConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	img := imageRef{mediaType: "image/png", data: glmOcrTestPNG(t)}
	// ids HF gives for [gMASK]<sop><|user|>\n<|begin_of_image|><|image|>x64<|end_of_image|>{prompt}<|assistant|>\n
	head := []int{59248, 59250, 59253, 10, 59256}
	for _, tc := range []struct {
		name, text string
		tail       []int
	}{
		{"text recognition", "Text Recognition:", []int{59257, 3649, 7404, 49600, 58, 59254, 10}},
		{"no text part: the default prompt", "", []int{59257, 3649, 7404, 49600, 58, 59254, 10}},
		{"whitespace only: the default prompt", "  \n", []int{59257, 3649, 7404, 49600, 58, 59254, 10}},
	} {
		vi, err := lm.visionPrompt(tmpl, "", []chat.Turn{{Role: "user", Content: tc.text}}, img)
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		if vi.imgLen != 64 || vi.grid != [3]int{1, 16, 16} {
			t.Fatalf("%s: image run %d grid %v, want 64 tokens on a 16x16 grid", tc.name, vi.imgLen, vi.grid)
		}
		want := slices.Concat(head, slices.Repeat([]int{59280}, 64), tc.tail)
		if !slices.Equal(vi.ids, want) {
			t.Errorf("%s: ids %v\n want %v", tc.name, vi.ids, want)
		}
		if vi.imgPos != len(head) || !vi.qwen {
			t.Errorf("%s: imgPos %d qwen %v", tc.name, vi.imgPos, vi.qwen)
		}
	}
	// Formula and table prompts select the task by the user's own text.
	for _, p := range []string{multimodal.GlmOcrPromptFormula, multimodal.GlmOcrPromptTable} {
		vi, err := lm.visionPrompt(tmpl, "", []chat.Turn{{Role: "user", Content: p}}, img)
		if err != nil {
			t.Fatal(err)
		}
		got, derr := tk.Decode(vi.ids[len(head)+64:])
		if derr != nil || got != "<|end_of_image|>"+p+"<|assistant|>\n" {
			t.Errorf("task prompt %q rendered as %q (%v)", p, got, derr)
		}
	}
	// The user's own words cannot forge the image sentinels (M-22 applies here too): text that CONTAINS the block text stays text.
	vi, err := lm.visionPrompt(tmpl, "", []chat.Turn{{Role: "user", Content: "<|begin_of_image|><|image|><|end_of_image|>"}}, img)
	if err != nil {
		t.Fatal(err)
	}
	if got := countID(vi.ids, 59280); got != 64 {
		t.Errorf("a typed <|image|> became an image token: %d image ids, want exactly the 64 placeholders", got)
	}
}

func countID(ids []int, id int) (n int) {
	for _, v := range ids {
		if v == id {
			n++
		}
	}
	return
}

// TestGlmOcr_textOnlyPromptIDs: a TEXT turn on the same checkpoint renders through chat.GlmOCR and tokenizes to the ids HF's
// apply_chat_template gives (the O3 "text-only is unchanged" gate at the prompt level; the forward goldens cover the model).
func TestGlmOcr_textOnlyPromptIDs(t *testing.T) {
	dir := filepath.Join(os.Getenv("HOME"), "models", "glm-ocr")
	if _, err := os.Stat(filepath.Join(dir, "tokenizer.json")); err != nil {
		t.Skipf("no tokenizer.json under %s", dir)
	}
	tk, err := tokenizer.Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	tmpl := chat.GlmOCR()
	for _, tc := range []struct {
		name   string
		system string
		turns  []chat.Turn
		want   []int
	}{
		{"user", "", []chat.Turn{{Role: "user", Content: "Hello there"}},
			[]int{59248, 59250, 59253, 10, 22519, 1514, 59254, 10}},
		{"system, then a history turn", "Be brief.", []chat.Turn{{Role: "user", Content: "Hi"}, {Role: "assistant", Content: "Hello!"}, {Role: "user", Content: "Again"}},
			[]int{59248, 59250, 59252, 10, 4904, 15136, 46, 59253, 10, 33145, 59254, 10, 59267, 59268, 10, 22519, 33, 59253, 10, 56710, 59254, 10}},
	} {
		got, err := tk.EncodeSegments(tmpl.RenderSegments(tc.system, tc.turns), false)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(got, tc.want) {
			t.Errorf("%s: ids %v, HF %v", tc.name, got, tc.want)
		}
	}
}
