// Command confidence is per-field confidence on constrained output: the answer is forced into a JSON Schema, and
// each enum, boolean and integer field comes back with the model's probability over what the schema allowed at the
// position that decided it (docs/tasks/task-constrained-confidence.md, C1).
//
// The number ranks answers — a low one is wrong more often (docs/measurements/confidence-c0-2026-09-27.md) — but it
// is NOT the probability that the value is right, and it is not calibrated. Use it to route ("ask a person below
// 0.7"), not as a probability. docs/server.md has the full caveat.
//
//	go run ./examples/confidence ~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf "I was charged twice for order #123. - Ann"
package main

import (
	"context"
	"fmt"
	"os"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/constrain"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

const schema = `{"type":"object","additionalProperties":false,"required":["category","urgent","orders"],"properties":{
  "category":{"enum":["billing","technical","shipping","other"]},
  "urgent":{"type":"boolean"},
  "orders":{"type":"integer"}}}`

func main() {
	if len(os.Args) < 3 {
		fmt.Fprintln(os.Stderr, "usage: confidence <model.gguf> <support ticket>")
		os.Exit(2)
	}
	path, ticket := os.Args[1], os.Args[2]
	m, err := decoder.Load(path, decoder.Options{})
	check(err, "load")
	defer m.Close()
	tok, err := tokenizer.LoadGGUF(path)
	check(err, "tokenizer")
	tmpl, err := chat.Detect(chat.Meta{ChatTemplate: tok.ChatTemplate(), HasToken: tok.Has})
	check(err, "chat template")

	ask := "Triage this support ticket. Return only a JSON object: category, urgent (true only for a deadline " +
		"within 48 hours), and orders (how many order numbers it mentions).\n\nTicket: " + ticket
	ids, err := tok.EncodeSegments(tmpl.RenderSegments("", []chat.Turn{{Role: "user", Content: ask}}), false)
	check(err, "encode")

	// The stop ids end generation; the masker allows them only once the document is complete.
	var stop []int
	for _, id := range []int{tok.Special().EOS, tok.Special().EndOfTurn} {
		if id >= 0 {
			stop = append(stop, id)
		}
	}
	for _, s := range tmpl.Stops().Strings {
		if id, ok := tok.TokenID(s); ok {
			stop = append(stop, id)
		}
	}
	g, err := constrain.JSONSchema([]byte(schema))
	check(err, "schema")
	mask := constrain.NewMasker(g, constrain.TokenBytes(m.Config().VocabSize, tok.TokenText), stop).
		StopWhenComplete().CaptureConfidence(constrain.ConfidenceOptions{})

	ch, gen := m.Generate(context.Background(), ids, 128, decoder.SamplingParams{LogitProcessor: mask.Process, StopIDs: stop})
	var out []int
	for id := range ch {
		out = append(out, id)
	}
	check(gen.Err(), "generate")
	text, err := tok.Decode(out)
	check(err, "decode")
	fmt.Println(text)

	fields, err := mask.FieldConfidence(out)
	check(err, "confidence")
	for _, f := range fields {
		fmt.Printf("%-10s %-12s confidence %.2f", f.Path, f.Value, f.Confidence)
		if f.Distribution != nil {
			fmt.Printf("  %v", f.Distribution)
		}
		fmt.Println()
	}
}

func check(err error, what string) {
	if err != nil {
		fmt.Fprintln(os.Stderr, what+":", err)
		os.Exit(1)
	}
}
