// Command embed is the smallest complete goinfer program: load a GGUF, render a chat prompt,
// generate, print. It exists because a cold-user run reached this API only by probing pkg.go.dev
// URLs for an HTTP 200 — there was no "use it as a library" example to copy
// (docs/measurements/cold-user-2026-09-06.md, scenario C).
//
// Kept short on purpose: 40 lines was the scenario's own bar, and a longer example would stop
// being the thing you can read in one sitting. CI builds and vets it, so it cannot rot into the
// state the README's `constrain` snippet was in — referencing identifiers that do not exist.
//
//	go run ./examples/embed ~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf "Reverse a string in Go"
package main

import (
	"context"
	"fmt"
	"os"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

func main() {
	if len(os.Args) < 3 {
		fmt.Fprintln(os.Stderr, "usage: embed <model.gguf> <prompt>")
		os.Exit(2)
	}
	path, prompt := os.Args[1], os.Args[2]

	// Load takes the PATH TO THE .gguf FILE (or a directory of safetensors) — not a directory
	// containing one, which is the guess a cold reader makes from the parameter name.
	m, err := decoder.Load(path, decoder.Options{})
	check(err, "load")
	defer m.Close()

	tok, err := tokenizer.LoadGGUF(path)
	check(err, "tokenizer")

	text := chat.ChatML().Render("", []chat.Turn{{Role: "user", Content: prompt}})
	ids, err := tok.Encode(text, true)
	check(err, "encode")

	out, _ := m.Generate(context.Background(), ids, 256, decoder.SamplingParams{})
	for id := range out {
		s, err := tok.Decode([]int{id})
		check(err, "decode")
		fmt.Print(s)
	}
	fmt.Println()
}

func check(err error, what string) {
	if err != nil {
		fmt.Fprintln(os.Stderr, what+":", err)
		os.Exit(1)
	}
}
