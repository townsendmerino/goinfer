package serveapp

import (
	"encoding/json"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/multimodal"
	"github.com/townsendmerino/goinfer/tokenizer"
)

const extractSchema = `{"type":"object","additionalProperties":false,"required":["invoice_number","total"],
 "properties":{"invoice_number":{"type":"string"},"total":{"type":"string"}}}`

func schemaFormat(schema string) *respFormat {
	rf := &respFormat{Type: "json_schema"}
	rf.JSONSchema = &struct {
		Name   string          `json:"name"`
		Schema json.RawMessage `json:"schema"`
	}{Name: "invoice", Schema: json.RawMessage(schema)}
	return rf
}

const extractTemplate = "{\n    \"invoice_number\": \"\",\n    \"total\": \"\"\n}"

// TestGlmOcr_extractionTurn is O5's serve rule on the turns the handler builds: with response_format json_schema, an image
// request whose text is empty or a BARE task prompt gets the model's extraction prompt (the card's instruction and the
// schema's JSON template); text the user wrote is kept verbatim. And nothing else changes: json_object, no response_format,
// a model that is not GLM-OCR, and a text-less schema-less request all leave the turn as it was.
func TestGlmOcr_extractionTurn(t *testing.T) {
	glm := &loadedModel{glm: &glmOcrTower{dir: "unused"}}
	want := multimodal.GlmOcrExtractionInstruction + "\n" + extractTemplate
	run := func(lm *loadedModel, rf *respFormat, text string) (string, error) {
		turns := []chat.Turn{{Role: "assistant", Content: "earlier"}, {Role: "user", Content: text}}
		err := lm.glmOcrExtractionTurn(rf, turns)
		if turns[0].Content != "earlier" {
			t.Errorf("an earlier turn was rewritten: %q", turns[0].Content)
		}
		return turns[1].Content, err
	}
	for _, in := range []string{"", "  ", multimodal.GlmOcrPromptText, multimodal.GlmOcrPromptTable} {
		if got, err := run(glm, schemaFormat(extractSchema), in); err != nil || got != want {
			t.Errorf("text %q: got %q, %v; want the extraction prompt", in, got, err)
		}
	}
	for _, in := range []string{"What is the total?", multimodal.GlmOcrExtractionPrompt("{\n    \"vendor\": \"\"\n}")} {
		if got, err := run(glm, schemaFormat(extractSchema), in); err != nil || got != in {
			t.Errorf("the user's own text %q was changed to %q (%v)", in, got, err)
		}
	}
	for name, c := range map[string]struct {
		lm *loadedModel
		rf *respFormat
	}{
		"no response_format":       {glm, nil},
		"json_object has no shape": {glm, &respFormat{Type: "json_object"}},
		"text":                     {glm, &respFormat{Type: "text"}},
		"not a GLM-OCR model":      {&loadedModel{}, schemaFormat(extractSchema)},
	} {
		if got, err := run(c.lm, c.rf, multimodal.GlmOcrPromptText); err != nil || got != multimodal.GlmOcrPromptText {
			t.Errorf("%s: turn became %q (%v), want it untouched", name, got, err)
		}
	}
	if _, err := run(glm, schemaFormat(`{"type":`), ""); err == nil || !strings.Contains(err.Error(), "extraction template") {
		t.Errorf("an unparseable schema: error %v, want one naming the extraction template", err)
	}
}

// TestGlmOcr_extractionPromptIDs runs the rule and serve's own prompt builder (visionPrompt: the chat.GlmOCR renderer, the
// image block, the segment splice, the real tokenizer) together, the two calls serveVisionChatWith makes in that order, and
// checks what the MODEL would read: the image first, then the card's instruction, a newline and the schema's template, the
// whole of it text, then the assistant marker. Skips cleanly without ~/models/glm-ocr/tokenizer.json.
func TestGlmOcr_extractionPromptIDs(t *testing.T) {
	dir := filepath.Join(os.Getenv("HOME"), "models", "glm-ocr")
	if _, err := os.Stat(filepath.Join(dir, "tokenizer.json")); err != nil {
		t.Skipf("no tokenizer.json under %s", dir)
	}
	tk, err := tokenizer.Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	m, err := decoder.Load("../../testdata/glm-ocr-tiny", decoder.Options{Quant: "f32"})
	if err != nil {
		t.Skipf("no tiny fixture: %v", err)
	}
	defer m.Close()
	lm := &loadedModel{tk: tk, model: m, tmpl: chat.GlmOCR(), glm: &glmOcrTower{dir: "unused"}, qwenMerge: 2, qwenImgTok: 59280}
	if lm.qwenPP, err = multimodal.LoadGlmOcrPreprocessConfig(dir); err != nil {
		t.Fatal(err)
	}
	img := imageRef{mediaType: "image/png", data: glmOcrTestPNG(t)}
	build := func(text string) string {
		turns := []chat.Turn{{Role: "user", Content: text}}
		if err := lm.glmOcrExtractionTurn(schemaFormat(extractSchema), turns); err != nil {
			t.Fatal(err)
		}
		vi, err := lm.visionPrompt(lm.tmpl, "", turns, img)
		if err != nil {
			t.Fatal(err)
		}
		if vi.imgLen != 64 {
			t.Fatalf("image run %d, want 64", vi.imgLen)
		}
		got, err := tk.Decode(vi.ids)
		if err != nil {
			t.Fatal(err)
		}
		return got
	}
	block := multimodal.GlmOcrImageBlock(64)
	tail := "<|assistant|>\n"
	extraction := "[gMASK]<sop><|user|>\n" + block + multimodal.GlmOcrExtractionInstruction + "\n" + extractTemplate + tail
	for _, in := range []string{"", multimodal.GlmOcrPromptText} {
		if got := build(in); got != extraction {
			t.Errorf("text %q: the model reads\n%q\nwant\n%q", in, got, extraction)
		}
	}
	own := "Read the total off this invoice."
	if got, want := build(own), "[gMASK]<sop><|user|>\n"+block+own+tail; got != want {
		t.Errorf("the user's own prompt: the model reads\n%q\nwant\n%q", got, want)
	}
}

// TestServeVisionChat_appliesExtractionRuleBeforeThePromptIsBuilt pins the CALL: the helpers above are correct when called,
// and nothing but this proves serveVisionChatWith calls the rule, and before it builds the prompt from the turns (afterwards
// it would rewrite turns nobody reads). The through-the-handler check on the real checkpoint is
// TestServe_glmOcrExtraction_O5 (realckpt, heavy).
func TestServeVisionChat_appliesExtractionRuleBeforeThePromptIsBuilt(t *testing.T) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "vision_serve.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	pos := map[string]token.Pos{}
	for _, d := range f.Decls {
		fd, ok := d.(*ast.FuncDecl)
		if !ok || fd.Name.Name != "serveVisionChatWith" {
			continue
		}
		ast.Inspect(fd.Body, func(n ast.Node) bool {
			if c, ok := n.(*ast.CallExpr); ok {
				if sel, ok := c.Fun.(*ast.SelectorExpr); ok {
					if _, seen := pos[sel.Sel.Name]; !seen {
						pos[sel.Sel.Name] = c.Pos()
					}
				}
			}
			return true
		})
	}
	ext, vp, pr := pos["glmOcrExtractionTurn"], pos["visionPrompt"], pos["prepare"]
	if !ext.IsValid() || !vp.IsValid() || !pr.IsValid() {
		t.Fatalf("serveVisionChatWith must call glmOcrExtractionTurn, visionPrompt and prepare (found %v)", pos)
	}
	if !(ext < vp && vp < pr) {
		t.Errorf("serveVisionChatWith must call glmOcrExtractionTurn, then visionPrompt, then prepare; the positions are %v, %v, %v", ext, vp, pr)
	}
}
