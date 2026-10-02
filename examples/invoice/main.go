// Command invoice reads a scanned invoice into a Go struct: an image goes in, an Invoice comes out, on the CPU of the machine
// you are sitting at, from one binary. The model is zai-org/GLM-OCR (a 0.9B document-OCR model); the struct is the single
// source of truth for BOTH halves of the request:
//
//   - the PROMPT: GLM-OCR's information-extraction prompt carries a JSON template, an object of blank values, not a JSON
//     Schema, so constrain.TemplateFromStruct(Invoice{}) is what the model is shown;
//   - the GUARANTEE: constrain.GrammarFromStruct(Invoice{}) is the grammar the masker applies at every decode step, so the
//     reply is JSON that has this struct's shape whatever the model would have preferred.
//
// What is guaranteed is the SHAPE (keys, which are required, each value's JSON type), not that a value is right: see
// constrain.GrammarFromStruct, and docs/measurements/glm-ocr-o5-2026-10/ for what the model gets right on rendered invoices.
//
//	go run ./examples/invoice ~/models/glm-ocr testdata/glm_ocr/invoice.png
//
// The checkpoint is zai-org/GLM-OCR (MIT; about 2.7 GB), a directory with config.json, tokenizer.json and the safetensors.
// On a CPU expect a few minutes (the vision tower alone is about 45 s a page); with the CUDA build of goinfer-serve the
// same request is seconds, see docs/server.md.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/townsendmerino/aikit/vision"
	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/constrain"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/multimodal"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// LineItem and Invoice are what the caller wants back. Amounts are numbers, not strings: the template then shows the model
// `0` for them, which it answers with a bare number the grammar accepts (a string-typed amount invites the model to write a
// bare number anyway, which a string grammar forbids; see the measurements note).
type LineItem struct {
	Description string  `json:"description"`
	Quantity    int     `json:"quantity"`
	UnitPrice   float64 `json:"unit_price"`
	Amount      float64 `json:"amount"`
}

type Invoice struct {
	InvoiceNumber string     `json:"invoice_number"`
	Date          string     `json:"date"`
	DueDate       string     `json:"due_date"`
	Vendor        string     `json:"vendor"`
	BillTo        string     `json:"bill_to"`
	Currency      string     `json:"currency"`
	Paid          bool       `json:"paid"`
	LineItems     []LineItem `json:"line_items"`
	Subtotal      float64    `json:"subtotal"`
	Tax           float64    `json:"tax"`
	Total         float64    `json:"total"`
}

func main() {
	if len(os.Args) < 3 {
		fmt.Fprintln(os.Stderr, "usage: invoice <glm-ocr checkpoint dir> <invoice.png|jpg>")
		os.Exit(2)
	}
	dir, imagePath := os.Args[1], os.Args[2]

	// One struct, two derivations.
	template, err := constrain.TemplateFromStruct(Invoice{})
	check(err, "template")
	grammar, err := constrain.GrammarFromStruct(Invoice{})
	check(err, "grammar")

	// The model, its tokenizer and the vision tower (the tower runs on the CPU, f32).
	m, err := decoder.Load(dir, decoder.Options{Quant: "int4"}) // the zero Options loads f32: slower and bigger, a little more accurate
	check(err, "load model")
	defer m.Close()
	tk, err := tokenizer.Load(dir)
	check(err, "tokenizer")
	tower, err := vision.LoadGlmOcrVisionEncoder(dir, false)
	check(err, "vision tower")
	pp, err := multimodal.LoadGlmOcrPreprocessConfig(dir) // the checkpoint's pixel budget, halved the way the model's processor does
	check(err, "preprocess config")

	// The image: pixels in the tower's patch order, and how many image tokens the tower will produce for them.
	raw, err := os.ReadFile(imagePath)
	check(err, "read image")
	pixels, grid, err := multimodal.QwenPreprocess(raw, pp)
	check(err, "preprocess")
	nImg := multimodal.QwenMergedTokens(grid, pp.MergeSize)

	// The prompt: [gMASK]<sop><|user|>, the image block, the extraction prompt, <|assistant|>. The block must reach the
	// tokenizer as SPECIAL text (its sentinels become real tokens); SpliceImageBlock does that and refuses anything else.
	prompt := multimodal.GlmOcrExtractionPrompt(template)
	block := multimodal.GlmOcrImageBlock(nImg)
	segs, err := multimodal.SpliceImageBlock(chat.GlmOCR().RenderSegments("", []chat.Turn{{Role: "user", Content: block + prompt}}), block)
	check(err, "splice image block")
	ids, err := tk.EncodeSegments(segs, false)
	check(err, "encode")
	imageToken, ok := tk.TokenID(multimodal.GlmOcrImagePad)
	if !ok {
		check(fmt.Errorf("no %s token", multimodal.GlmOcrImagePad), "tokenizer")
	}
	imgPos, imgLen := multimodal.FindImageRun(ids, imageToken)

	// The stop ids end generation; the masker allows them only once the document is complete.
	var stop []int
	for _, s := range chat.GlmOCR().Stops().Strings {
		if id, ok := tk.TokenID(s); ok {
			stop = append(stop, id)
		}
	}
	mask := constrain.NewMasker(grammar, constrain.TokenBytes(m.Config().VocabSize, tk.TokenText), stop).StopWhenComplete()

	features := func() ([]float32, error) { return tower.Forward(pixels, [][3]int{grid}) }
	ch, gen := m.GenerateQwenVL(context.Background(), ids, imgPos, imgLen, multimodal.HashImageBytes(raw), features,
		[][3]int{grid}, pp.MergeSize, imageToken, 1500,
		decoder.SamplingParams{Temperature: 0, LogitProcessor: mask.Process, StopIDs: stop})
	var out []int
	for id := range ch {
		out = append(out, id)
	}
	check(gen.Err(), "generate")
	text, err := tk.Decode(out)
	check(err, "decode")

	var inv Invoice
	check(json.Unmarshal([]byte(text), &inv), "unmarshal") // the shape is guaranteed; a value can still be wrong
	pretty, _ := json.MarshalIndent(inv, "", "  ")
	fmt.Println(string(pretty))
}

func check(err error, what string) {
	if err != nil {
		fmt.Fprintln(os.Stderr, what+":", err)
		os.Exit(1)
	}
}
