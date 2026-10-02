//go:build realckpt

package serveapp

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/constrain"
	"github.com/townsendmerino/goinfer/internal/loadflags"
	"github.com/townsendmerino/goinfer/multimodal"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestServe_glmOcrExtraction_O5 is O5's serve gate, on the real GLM-OCR checkpoint (CPU, int4: about two minutes, most of it
// the f32 vision tower) through the real handler: an image_url part (the committed rendered invoice, NOT a real scan), NO text
// part, and response_format json_schema (the committed invoice schema, which is constrain.SchemaFromStruct of the example's
// Invoice struct).
//
//	(a) the PROMPT: usage.prompt_tokens equals the count of an independently built
//	    [gMASK]<sop><|user|>\n<image block><instruction>\n<template><|assistant|>\n, so the schema's JSON template reached the
//	    model through the handler. A handler that does not apply the rule sends the bare task prompt (1,668 ids) and fails
//	    here. Checked with max_tokens 1 first, so the failure costs the tower and a prefill, not a whole generation;
//	(b) the GUARANTEE: the reply, the full generation, decodes into the typed struct with unknown fields DISALLOWED and every
//	    key present, finish_reason stop, so the grammar governed the vision route;
//	(c) the model READING the page: the invoice number printed in the header and the bold total. These two are loose on
//	    purpose, a two-field smoke on an int4 model whose exact text is precision-sensitive (O3: int4 drops the header line of
//	    plain OCR here); the per-field numbers over 15 documents are in docs/measurements/glm-ocr-o5-2026-10/, reported, not gated;
//	(d) goinfer_confidence now rides the vision route: a request that asks gets a record per integer and boolean field;
//	(e) the grammar DISCRIMINATES on the vision route: the same page under a schema that caps line_items at 3 and restricts
//	    currency to an enum comes back with exactly 3 items and a legal currency, where the unconstrained model would write
//	    the page's six lines. (Well-behaved output decodes either way, so (b) alone cannot tell a constrained route from an
//	    unconstrained one.)
//
//	GOINFER_HEAVY_TESTS=1 go test -tags realckpt ./internal/serveapp/ -run TestServe_glmOcrExtraction_O5 -v -timeout 20m 2>&1 | tee /tmp/glm-ocr-extract.log
func TestServe_glmOcrExtraction_O5(t *testing.T) {
	requireHeavyModel(t)
	ckpt := filepath.Join(os.Getenv("HOME"), "models", "glm-ocr")
	if _, err := os.Stat(filepath.Join(ckpt, "config.json")); err != nil {
		t.Skipf("asset missing: %s", ckpt)
	}
	docs := filepath.Join("..", "..", "testdata", "glm_ocr")
	png, err := os.ReadFile(filepath.Join(docs, "invoice.png"))
	if err != nil {
		t.Fatal(err)
	}
	schema, err := os.ReadFile(filepath.Join(docs, "invoice.schema.json"))
	if err != nil {
		t.Fatal(err)
	}
	srv, err := newServer(config{models: modelFlag{{name: "glm", path: ckpt}}, load: loadflags.Flags{Backend: "cpu", Quant: "int4", QuantSet: true}, kvSessions: 2})
	if err != nil {
		t.Fatalf("newServer: %v", err)
	}
	lm := srv.models["glm"]
	if lm == nil || lm.glm == nil {
		t.Fatal("auto-discovery did not attach the GLM-OCR tower")
	}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", srv.handleChat)
	ts := httptest.NewServer(mux)
	defer ts.Close()

	post := func(maxTokens int, extra map[string]any) (struct {
		Choices []struct {
			Message      struct{ Content string } `json:"message"`
			FinishReason string                   `json:"finish_reason"`
		} `json:"choices"`
		Usage usage             `json:"usage"`
		Conf  []json.RawMessage `json:"goinfer_confidence"`
	}, int) {
		t.Helper()
		body := map[string]any{
			"model": "glm", "temperature": 0, "max_tokens": maxTokens,
			"messages": []map[string]any{{"role": "user", "content": []map[string]any{
				{"type": "image_url", "image_url": map[string]string{"url": "data:image/png;base64," + base64.StdEncoding.EncodeToString(png)}},
			}}},
			"response_format": map[string]any{"type": "json_schema", "json_schema": map[string]any{"name": "invoice", "schema": json.RawMessage(schema)}},
		}
		for k, v := range extra {
			body[k] = v
		}
		raw, _ := json.Marshal(body)
		resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", bytes.NewReader(raw))
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		var out struct {
			Choices []struct {
				Message      struct{ Content string } `json:"message"`
				FinishReason string                   `json:"finish_reason"`
			} `json:"choices"`
			Usage usage             `json:"usage"`
			Conf  []json.RawMessage `json:"goinfer_confidence"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
			t.Fatalf("decode (status %d): %v", resp.StatusCode, err)
		}
		return out, resp.StatusCode
	}

	// (a) the prompt, from an independent construction.
	tk, err := tokenizer.Load(ckpt)
	if err != nil {
		t.Fatal(err)
	}
	template, err := constrain.TemplateFromSchema(schema)
	if err != nil {
		t.Fatal(err)
	}
	pre, err := multimodal.LoadGlmOcrPreprocessConfig(ckpt)
	if err != nil {
		t.Fatal(err)
	}
	_, grid, err := multimodal.QwenPreprocess(png, pre)
	if err != nil {
		t.Fatal(err)
	}
	block := multimodal.GlmOcrImageBlock(multimodal.QwenMergedTokens(grid, pre.MergeSize))
	want, err := tk.EncodeSegments([]tokenizer.Segment{
		{Text: "[gMASK]", Special: true}, {Text: "<sop>", Special: true}, {Text: "<|user|>", Special: true}, {Text: "\n"},
		{Text: block, Special: true}, {Text: multimodal.GlmOcrExtractionPrompt(template)},
		{Text: "<|assistant|>", Special: true}, {Text: "\n"},
	}, false)
	if err != nil {
		t.Fatal(err)
	}
	probe, status := post(1, nil)
	if status != 200 {
		t.Fatalf("max_tokens 1 request: status %d", status)
	}
	if probe.Usage.PromptTokens != len(want) {
		t.Fatalf("usage.prompt_tokens = %d, the extraction prompt is %d ids: the schema's template did not reach the model "+
			"(a bare task prompt would be about 90 ids shorter)", probe.Usage.PromptTokens, len(want))
	}
	t.Logf("(a) prompt: %d ids = the independently built extraction prompt (image run %d)", probe.Usage.PromptTokens, multimodal.QwenMergedTokens(grid, pre.MergeSize))

	// (b) + (c) + (d): the full generation, with confidence.
	full, status := post(1500, map[string]any{"goinfer_confidence": true})
	if status != 200 || len(full.Choices) != 1 {
		t.Fatalf("status %d, %d choices", status, len(full.Choices))
	}
	content := full.Choices[0].Message.Content
	t.Logf("reply (%d completion tokens, finish %s):\n%s", full.Usage.CompletionTokens, full.Choices[0].FinishReason, content)
	if full.Choices[0].FinishReason != "stop" {
		t.Errorf("finish_reason %q, want stop (a document the grammar completed)", full.Choices[0].FinishReason)
	}
	var inv struct {
		InvoiceNumber string  `json:"invoice_number"`
		Date          string  `json:"date"`
		DueDate       string  `json:"due_date"`
		Vendor        string  `json:"vendor"`
		BillTo        string  `json:"bill_to"`
		Currency      string  `json:"currency"`
		Paid          *bool   `json:"paid"`
		Subtotal      float64 `json:"subtotal"`
		Tax           float64 `json:"tax"`
		Total         *float64
		LineItems     []struct {
			Description string  `json:"description"`
			Quantity    int     `json:"quantity"`
			UnitPrice   float64 `json:"unit_price"`
			Amount      float64 `json:"amount"`
		} `json:"line_items"`
	}
	dec := json.NewDecoder(bytes.NewReader([]byte(content)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&inv); err != nil {
		t.Fatalf("(b) the reply does not decode into the schema's struct: %v", err)
	}
	var keys map[string]json.RawMessage
	_ = json.Unmarshal([]byte(content), &keys)
	for _, k := range []string{"invoice_number", "date", "due_date", "vendor", "bill_to", "currency", "paid", "line_items", "subtotal", "tax", "total"} {
		if _, ok := keys[k]; !ok {
			t.Errorf("(b) required key %q missing", k)
		}
	}
	if len(inv.LineItems) == 0 {
		t.Error("(b) no line items")
	}
	if inv.InvoiceNumber != "INV-2026-0417" {
		t.Errorf("(c) invoice_number %q, the page prints INV-2026-0417", inv.InvoiceNumber)
	}
	if inv.Total == nil || *inv.Total != 1140.55 {
		t.Errorf("(c) total %v, the page prints 1140.55", inv.Total)
	}
	// (d) confidence on the vision route: integer quantities and the boolean `paid` are reported.
	if len(full.Conf) < len(inv.LineItems)+1 {
		t.Errorf("(d) goinfer_confidence has %d records for %d line items and a boolean: the vision route did not write it back", len(full.Conf), len(inv.LineItems))
	}

	// (e) a schema the model would not satisfy on its own.
	var capped map[string]any
	if err := json.Unmarshal(schema, &capped); err != nil {
		t.Fatal(err)
	}
	props := capped["properties"].(map[string]any)
	props["line_items"].(map[string]any)["maxItems"] = 3
	props["currency"] = map[string]any{"type": "string", "enum": []string{"EUR", "GBP"}} // the page is USD: the grammar must override the model
	cappedRaw, _ := json.Marshal(capped)
	schema = cappedRaw
	limited, status := post(1500, nil)
	if status != 200 || len(limited.Choices) != 1 {
		t.Fatalf("(e) status %d, %d choices", status, len(limited.Choices))
	}
	t.Logf("(e) capped reply:\n%s", limited.Choices[0].Message.Content)
	var lim struct {
		Currency  string            `json:"currency"`
		LineItems []json.RawMessage `json:"line_items"`
	}
	if err := json.Unmarshal([]byte(limited.Choices[0].Message.Content), &lim); err != nil {
		t.Fatalf("(e) reply is not JSON: %v", err)
	}
	if len(lim.LineItems) != 3 {
		t.Errorf("(e) %d line items under maxItems 3 (the page has six): the grammar did not govern the vision route", len(lim.LineItems))
	}
	if lim.Currency != "EUR" && lim.Currency != "GBP" {
		t.Errorf("(e) currency %q is outside the schema's enum", lim.Currency)
	}
}
