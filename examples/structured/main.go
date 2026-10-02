// Command structured fills in a Go struct: the output is constrained to the JSON Schema derived from the struct's json
// tags, so json.Unmarshal into it cannot fail on shape. docs/use-from-go.md walks through it.
//
// What is guaranteed is the SHAPE (keys, which are required, each value's JSON type), not the magnitude of a number
// and not that the values are right. constrain.GrammarFromStruct's doc comment has the details.
//
//	go run ./examples/structured ~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf "Ann Lee, 34, writes Go and Rust"
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/constrain"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

type Person struct {
	Name  string   `json:"name"`
	Age   int      `json:"age"`
	Langs []string `json:"langs"`
	Email string   `json:"email,omitempty"` // omitempty (or a pointer) makes a field optional
}

func main() {
	if len(os.Args) < 3 {
		fmt.Fprintln(os.Stderr, "usage: structured <model.gguf> <text about a person>")
		os.Exit(2)
	}
	path, text := os.Args[1], os.Args[2]
	m, err := decoder.Load(path, decoder.Options{Quant: "int4"}) // the zero Options loads f32
	check(err, "load")
	defer m.Close()
	tok, err := tokenizer.LoadGGUF(path)
	check(err, "tokenizer")
	tmpl, err := chat.Detect(chat.Meta{ChatTemplate: tok.ChatTemplate(), HasToken: tok.Has})
	check(err, "chat template")

	ask := "Extract the person as a JSON object with name, age and langs.\n\n" + text
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

	g, err := constrain.GrammarFromStruct(Person{}) // struct -> JSON Schema -> grammar
	check(err, "grammar")
	mask := constrain.NewMasker(g, constrain.TokenBytes(m.Config().VocabSize, tok.TokenText), stop).StopWhenComplete()

	ch, gen := m.Generate(context.Background(), ids, 128, decoder.SamplingParams{LogitProcessor: mask.Process, StopIDs: stop})
	var out []int
	for id := range ch {
		out = append(out, id)
	}
	check(gen.Err(), "generate")
	raw, err := tok.Decode(out)
	check(err, "decode")

	var p Person
	check(json.Unmarshal([]byte(raw), &p), "unmarshal") // the shape is guaranteed; a value can still be wrong
	fmt.Printf("%+v\n", p)
}

func check(err error, what string) {
	if err != nil {
		fmt.Fprintln(os.Stderr, what+":", err)
		os.Exit(1)
	}
}
