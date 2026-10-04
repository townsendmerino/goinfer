package chatapp

import (
	"bytes"
	"encoding/json"
	"flag"
	"image"
	"image/color"
	"image/png"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/constrain"
	"github.com/townsendmerino/goinfer/multimodal"
	"github.com/townsendmerino/goinfer/tokenizer"
)

const (
	glmTestConf = `{"model_type":"glm_ocr","vision_config":{"depth":24}}`
	glmTestPre  = `{"size":{"shortest_edge":12544,"longest_edge":9633792},"patch_size":14,"temporal_patch_size":2,"merge_size":2,"image_mean":[0.48145466,0.4578275,0.40821073],"image_std":[0.26862954,0.26130258,0.27577711]}`
	imgTestText = `{"type":"object","additionalProperties":false,"properties":{"invoice_number":{"type":"string"},"total":{"type":"number"}}}`
)

// writeGlmDir writes a directory shaped like a GLM-OCR checkpoint's config files (no weights: nothing here loads a model).
func writeGlmDir(t *testing.T, config, pre string) string {
	t.Helper()
	d := t.TempDir()
	if config != "" {
		if err := os.WriteFile(filepath.Join(d, "config.json"), []byte(config), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if pre != "" {
		if err := os.WriteFile(filepath.Join(d, "preprocessor_config.json"), []byte(pre), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return d
}

func writeTestPNG(t *testing.T, dir string) string {
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
	p := filepath.Join(dir, "page.png")
	if err := os.WriteFile(p, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

// --image is a real flag on chat's own FlagSet (the list serveOnlyInvocation and the docs test read), takes a path, and
// defaults to off; a command line the docs show parses to the values the demo needs.
func TestFlags_imageParses(t *testing.T) {
	fs := flag.NewFlagSet("chat", flag.ContinueOnError)
	cf := registerFlags(fs)
	if f := fs.Lookup("image"); f == nil {
		t.Fatal("--image is not registered")
	} else if f.DefValue != "" || !strings.Contains(f.Usage, "GLM-OCR") {
		t.Errorf("--image default %q, usage %q", f.DefValue, f.Usage)
	}
	if err := fs.Parse([]string{"--model", "/m/glm-ocr", "--image", "invoice.png", "--schema", "invoice.schema.json"}); err != nil {
		t.Fatal(err)
	}
	if *cf.image != "invoice.png" || *cf.schema != "invoice.schema.json" || *cf.model != "/m/glm-ocr" {
		t.Errorf("parsed image=%q schema=%q model=%q", *cf.image, *cf.schema, *cf.model)
	}
	fs2 := flag.NewFlagSet("chat", flag.ContinueOnError)
	cf2 := registerFlags(fs2)
	if err := fs2.Parse(nil); err != nil || *cf2.image != "" {
		t.Errorf("--image defaults to %q (%v), want off", *cf2.image, err)
	}
}

func TestCheckImageFlags(t *testing.T) {
	for _, tc := range []struct {
		name, image, model, batch, wantErr string
	}{
		{"no image: nothing to check", "", "", "", ""},
		{"no image with a batch", "", "", "b.jsonl", ""},
		{"image with a model dir", "a.png", "/m/glm-ocr", "", ""},
		{"image without --model", "a.png", "", "", "--model"},
		{"image with --batch", "a.png", "/m/glm-ocr", "b.jsonl", "--batch"},
	} {
		err := checkImageFlags(tc.image, tc.model, tc.batch)
		switch {
		case tc.wantErr == "" && err != nil:
			t.Errorf("%s: %v", tc.name, err)
		case tc.wantErr != "" && (err == nil || !strings.Contains(err.Error(), tc.wantErr)):
			t.Errorf("%s: error %v, want one naming %q", tc.name, err, tc.wantErr)
		}
	}
}

// loadImageInput fails EARLY and by name: before a model load, for a directory that is not a GLM-OCR checkpoint, a missing
// file, and bytes that are not an image; and on a good pair it returns the grid and token count the tower will see.
func TestLoadImageInput(t *testing.T) {
	good := writeGlmDir(t, glmTestConf, glmTestPre)
	img := writeTestPNG(t, t.TempDir())
	in, err := loadImageInput(good, img)
	if err != nil {
		t.Fatal(err)
	}
	if in.grid != [3]int{1, 16, 16} || in.nImg != 64 {
		t.Errorf("grid %v, %d image tokens; want [1 16 16] and 64 for a 224x224 image", in.grid, in.nImg)
	}
	if in.pp.MinPixels != 6272 || in.pp.MaxPixels != 4816896 {
		t.Errorf("pixel bounds %d..%d: the file's must be HALVED (6272..4816896), as in serve", in.pp.MinPixels, in.pp.MaxPixels)
	}
	notImage := filepath.Join(t.TempDir(), "x.png")
	if err := os.WriteFile(notImage, []byte("not an image"), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct{ name, dir, file, want string }{
		{"text-only copy (no vision_config)", writeGlmDir(t, `{"model_type":"glm_ocr"}`, glmTestPre), img, "GLM-OCR checkpoint directory"},
		{"another family", writeGlmDir(t, `{"model_type":"qwen3_5","vision_config":{"depth":12}}`, glmTestPre), img, "GLM-OCR checkpoint directory"},
		{"not a directory", filepath.Join(t.TempDir(), "model.gguf"), img, "GLM-OCR checkpoint directory"},
		{"missing image", good, filepath.Join(t.TempDir(), "nope.png"), syscall.ENOENT.Error()}, // the OS's own not-found text: "no such file or directory", or Windows' "The system cannot find the file specified."
		{"not an image", good, notImage, "x.png"},
	} {
		if _, err := loadImageInput(tc.dir, tc.file); err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: error %v, want one containing %q", tc.name, err, tc.want)
		}
	}
}

// The O5 rule on the turns an image run sends: with --schema the template prompt replaces an empty or bare-task-prompt user
// turn and nothing else; without a schema an empty turn is plain text recognition; earlier turns are never rewritten.
func TestImageTurns(t *testing.T) {
	tmpl, err := constrainTemplate(imgTestText)
	if err != nil {
		t.Fatal(err)
	}
	want := multimodal.GlmOcrExtractionPrompt(tmpl)
	withSchema := &session{schema: []byte(imgTestText)}
	noSchema := &session{}
	last := func(s *session, turns []chat.Turn) (string, []chat.Turn) {
		out, err := s.imageTurns(turns)
		if err != nil {
			t.Fatal(err)
		}
		return out[len(out)-1].Content, out
	}
	for _, in := range []string{"", "  ", multimodal.GlmOcrPromptText, multimodal.GlmOcrPromptTable} {
		if got, _ := last(withSchema, []chat.Turn{{Role: "user", Content: in}}); got != want {
			t.Errorf("--schema, text %q: sent %q, want the extraction prompt", in, got)
		}
	}
	own := "What is the total?"
	if got, _ := last(withSchema, []chat.Turn{{Role: "user", Content: own}}); got != own {
		t.Errorf("--schema, the user's own -p text became %q: it must be sent as written", got)
	}
	if got, _ := last(noSchema, []chat.Turn{{Role: "user", Content: ""}}); got != multimodal.GlmOcrDefaultPrompt {
		t.Errorf("no schema, empty text: sent %q, want %q", got, multimodal.GlmOcrDefaultPrompt)
	}
	if got, _ := last(noSchema, []chat.Turn{{Role: "user", Content: multimodal.GlmOcrPromptFormula}}); got != multimodal.GlmOcrPromptFormula {
		t.Errorf("no schema: the user's task prompt became %q", got)
	}
	hist := []chat.Turn{{Role: "user", Content: "earlier"}, {Role: "assistant", Content: "reply"}, {Role: "user", Content: ""}}
	if _, out := last(withSchema, hist); out[0].Content != "earlier" || hist[2].Content != "" {
		t.Errorf("an earlier turn was rewritten, or the caller's slice was mutated: %+v / %+v", out, hist)
	}
	if got, _ := last(withSchema, nil); got != want {
		t.Errorf("no turns at all: sent %q, want the extraction prompt", got)
	}
	if _, err := (&session{schema: []byte(`{"type":`)}).imageTurns(nil); err == nil || !strings.Contains(err.Error(), "extraction template") {
		t.Errorf("a bad schema: error %v", err)
	}
	// The REPL's coding system prompt is not sent to an OCR model; an explicit --system is.
	if got := (&session{system: defaultSystem}).imageSystem(); got != "" {
		t.Errorf("default system prompt reached the image request: %q", got)
	}
	if got := (&session{system: "Answer in French."}).imageSystem(); got != "Answer in French." {
		t.Errorf("an explicit system prompt was dropped: %q", got)
	}
}

// imagePrompt with the REAL tokenizer: what the model reads is the image first, the card's instruction, a newline and the
// schema's template, then the assistant marker, with the sentinels real tokens. Skips cleanly without the checkpoint's
// tokenizer.json.
func TestImagePrompt_modelReadsTheTemplate(t *testing.T) {
	dir := filepath.Join(os.Getenv("HOME"), "models", "glm-ocr")
	if _, err := os.Stat(filepath.Join(dir, "tokenizer.json")); err != nil {
		t.Skipf("no tokenizer.json under %s", dir)
	}
	tk, err := tokenizer.Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	tmpl, _ := constrainTemplate(imgTestText)
	s := &session{tk: tk, tmpl: chat.GlmOCR(), system: defaultSystem, schema: []byte(imgTestText),
		img: &imageInput{nImg: 64}}
	ids, pos, n, tok, err := s.imagePrompt([]chat.Turn{{Role: "user", Content: ""}})
	if err != nil {
		t.Fatal(err)
	}
	if n != 64 || tok != 59280 || pos != 5 {
		t.Errorf("image run at %d, %d tokens, id %d; want 5, 64, 59280", pos, n, tok)
	}
	got, err := tk.Decode(ids)
	if err != nil {
		t.Fatal(err)
	}
	want := "[gMASK]<sop><|user|>\n" + multimodal.GlmOcrImageBlock(64) + multimodal.GlmOcrExtractionInstruction + "\n" + tmpl + "<|assistant|>\n"
	if got != want {
		t.Errorf("the model reads\n%q\nwant\n%q", got, want)
	}
}

// TestChatImage_demoEndToEnd is the one-line demo, on the real checkpoint: the real goinfer-chat binary (pure Go, CPU,
// int4: about 75 s, most of it the vision tower) on the committed rendered invoice (NOT a real scan) with the committed
// schema. Heavy: needs GOINFER_HEAVY_TESTS=1 and ~/models/glm-ocr, and skips cleanly without either. The assertions are loose
// on purpose (the exact text is int4-sensitive): stdout is JSON with every schema key, the invoice number is the one printed
// at the top of the page, and the total is 1140.55; "every key present" is the grammar's contract, the two values are the
// model reading the page.
func TestChatImage_demoEndToEnd(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1 to opt in (loads a 2.7 GB model, runs the CPU tower)")
	}
	ckpt := filepath.Join(os.Getenv("HOME"), "models", "glm-ocr")
	if _, err := os.Stat(filepath.Join(ckpt, "model.safetensors")); err != nil {
		t.Skipf("no GLM-OCR checkpoint at %s", ckpt)
	}
	bin := filepath.Join(t.TempDir(), "chat")
	if runtime.GOOS == "windows" {
		bin += ".exe"
	}
	if out, err := exec.Command("go", "build", "-o", bin, "../../demo/chat").CombinedOutput(); err != nil {
		t.Fatalf("build: %v\n%s", err, out)
	}
	cmd := exec.Command(bin, "--backend", "cpu", "--model", ckpt, "--image", "../../testdata/glm_ocr/invoice.png", "--schema", "../../testdata/glm_ocr/invoice.schema.json")
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("run: %v\n%s", err, stderr.String())
	}
	var got map[string]any
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatalf("stdout is not JSON: %v\n%s", err, out)
	}
	for _, k := range []string{"invoice_number", "date", "due_date", "vendor", "bill_to", "currency", "paid", "line_items", "subtotal", "tax", "total"} {
		if _, ok := got[k]; !ok {
			t.Errorf("schema key %q missing from the output", k)
		}
	}
	if got["invoice_number"] != "INV-2026-0417" {
		t.Errorf("invoice_number %v, the page prints INV-2026-0417", got["invoice_number"])
	}
	if got["total"] != 1140.55 {
		t.Errorf("total %v, the page prints 1140.55", got["total"])
	}
	t.Logf("%s\n%s", out, stderr.String())
}

// constrainTemplate is constrain.TemplateFromSchema for a schema given as a string.
func constrainTemplate(schema string) (string, error) {
	return constrain.TemplateFromSchema([]byte(schema))
}
