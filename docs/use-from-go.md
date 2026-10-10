# Use it from Go

goinfer is a Go module. You import it, and the model runs inside your own process: no cgo, no
daemon, no Python, and it cross-compiles like any other Go program. This page is the path for
that use. The HTTP server is covered separately in [server.md](server.md).

Package documentation is on [pkg.go.dev](https://pkg.go.dev/github.com/townsendmerino/goinfer).
The packages this page uses are `decoder` (load and generate), `tokenizer`, `chat` (prompt
templates) and `constrain` (structured output).

Every code block below is copied from a program under
[`examples/`](https://github.com/townsendmerino/goinfer/tree/main/examples). CI builds and vets
those programs, and a test fails if a block here stops matching its source, so what you read
here compiles.

## Install

Requires Go 1.27 or newer. From inside your own module (`go mod init …` first):

```bash
go get github.com/townsendmerino/goinfer/decoder@latest github.com/townsendmerino/goinfer/tokenizer@latest
```

Name the packages you import, as above. A `go get` of the bare module path, with no package,
records the requirement but does not fetch enough to build against, and the next build fails
with `missing go.sum entry for module providing package …`. One command like the one above is
enough for every other package in the module (`chat`, `constrain`, …).

You also need a model file. Any GGUF of a [supported family](https://goinfer.dev/models/)
works; `goinfer-chat pull qwen2.5-coder-0.5b` fetches a small one.

## The smallest program

[`examples/embed/main.go`](../examples/embed/main.go) loads a GGUF, renders a chat prompt,
generates and prints. `decoder.Load` takes the path to the model file itself:

```go
	// Load takes the PATH TO THE .gguf FILE (or a directory of safetensors) — not a directory
	// containing one, which is the guess a cold reader makes from the parameter name.
	m, err := decoder.Load(path, decoder.Options{})
	check(err, "load")
	defer m.Close()
```

The zero `decoder.Options` loads the weights at f32. `Options{Quant: "int4"}` is what the
command-line tools default to, and uses about a quarter of the memory.

The tokenizer comes from the same file. `PreTokenizerDecline` names a pre-tokenizer the
checkpoint declares and goinfer does not implement. Text can then split into different tokens
than the reference tokenizer would produce, and the example says so rather than staying quiet:

```go
	tok, err := tokenizer.LoadGGUF(path)
	check(err, "tokenizer")
	if d := tok.PreTokenizerDecline(); d != "" {
		fmt.Fprintln(os.Stderr, "!! tokenizer:", d)
	}
```

**Why `chat.Detect` comes before encoding.** `Model.Generate` is a raw completion primitive: it
continues whatever token sequence it is given, and it has no idea of chat turns. An instruct
model was trained on its own template (special tokens around each turn), and fed a bare
question it tends to repeat itself instead of answering. `chat.Detect` reads the template the
checkpoint ships and returns a renderer for it. When it does not recognize the template
(`chat.ErrUnknownTemplate`), the example falls back to the raw prompt. A rendered template
already carries the model's own beginning-of-sequence marker, so only the fallback asks the
tokenizer to add one:

```go
// buildPrompt renders the checkpoint's own chat template when Detect recognizes it (ErrUnknownTemplate
// means fall back to the raw prompt) and encodes it. A rendered template already carries the family's
// own BOS marker (Template.Render's contract, chat/chat.go:120-121); addBOS=true would double it on
// Gemma/Llama-3/Mistral (M-39, audit-2026-09-10). Only the raw-text fallback needs the tokenizer to
// prepend one.
func buildPrompt(tok *tokenizer.Tokenizer, prompt string) ([]int, error) {
	text := prompt
	rendered := false
	if tmpl, terr := chat.Detect(chat.Meta{ChatTemplate: tok.ChatTemplate(), HasToken: tok.Has}); terr == nil {
		text = tmpl.Render("", []chat.Turn{{Role: "user", Content: prompt}})
		rendered = true
	}
	return tok.Encode(text, !rendered)
}
```

`Generate` returns a channel of token ids, which closes when generation ends, and a
`*decoder.Generation`: once the channel has closed, its `Err` reports an error that stopped
generation. The smallest example ignores it;
the next one checks it.

```go
	out, _ := m.Generate(context.Background(), ids, 256, decoder.SamplingParams{})
	for id := range out {
		s, err := tok.Decode([]int{id})
		check(err, "decode")
		fmt.Print(s)
	}
	fmt.Println()
```

## Output shaped like a Go struct

[`examples/structured/main.go`](../examples/structured/main.go) derives a JSON Schema from a
struct's `json` tags and constrains generation to it. A field is required unless it is a pointer
or has `omitempty`:

```go
type Person struct {
	Name  string   `json:"name"`
	Age   int      `json:"age"`
	Langs []string `json:"langs"`
	Email string   `json:"email,omitempty"` // omitempty (or a pointer) makes a field optional
}
```

The constraint is a logit mask. At every step, `Masker.Process` sets every token that would
break the schema to −∞, so an invalid token cannot be sampled at all; nothing is retried.
`StopWhenComplete` ends generation at the first complete document. The stop ids are the model's
end-of-sequence and end-of-turn tokens, which the masker allows only once the document is
complete:

```go
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
```

```go
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
```

**What is guaranteed is the shape**: the keys, which are required, and each value's JSON type.
**What is not** is a number's magnitude (JSON Schema's integer has no width, so a `uint8` field
can be handed 99999 and `json.Unmarshal` returns an error), and that the values are correct. The
model can still fill in the wrong name. `GrammarFromStruct` refuses a type it cannot describe:
a recursive type, a struct with no exported fields, and a type with its own `UnmarshalJSON`
(unless it decodes from a string through `UnmarshalText`, which maps to a string).

## Output shaped like a JSON Schema

`constrain.JSONSchema` compiles a schema you already have.
[`examples/confidence/main.go`](../examples/confidence/main.go) uses this one:

```go
const schema = `{"type":"object","additionalProperties":false,"required":["category","urgent","orders"],"properties":{
  "category":{"enum":["billing","technical","shipping","other"]},
  "urgent":{"type":"boolean"},
  "orders":{"type":"integer"}}}`
```

```go
	g, err := constrain.JSONSchema([]byte(schema))
	check(err, "schema")
```

The supported subset:

- objects, with required and optional properties and `additionalProperties: false`;
- arrays, with `items`, `minItems` and `maxItems`;
- `string`, `number`, `integer`, `boolean` and `null`;
- `enum` and `const`;
- nesting to any depth.

An object may have up to 64 properties and an enum up to 64 values. A keyword or shape outside
the subset is an error from `JSONSchema`, so a constraint that could not be enforced is never
silently dropped. A property-based test in `constrain` checks that every constrained generation
validates against its schema.

## How sure was it: `CaptureConfidence`

The same example turns on confidence capture on the masker, then asks for it per field once
generation is done:

```go
	mask := constrain.NewMasker(g, constrain.TokenBytes(m.Config().VocabSize, tok.TokenText), stop).
		StopWhenComplete().CaptureConfidence(constrain.ConfidenceOptions{})
```

```go
	fields, err := mask.FieldConfidence(out)
	check(err, "confidence")
	for _, f := range fields {
		fmt.Printf("%-10s %-12s confidence %.2f", f.Path, f.Value, f.Confidence)
		if f.Distribution != nil {
			fmt.Printf("  %v", f.Distribution)
		}
		fmt.Println()
	}
```

For each enum, boolean and integer field, `FieldConfidence` reports the model's probability over
the values the schema allowed, at the position that decided the value. A low number is wrong
more often ([measurement](measurements/confidence-c0-2026-09-27.md)), which makes it useful for
routing, such as "ask a person below 0.7".

It is **not** the probability that the value is right, and it is **not calibrated**: 0.9 does
not mean right nine times in ten. [server.md](server.md) has the full description; the server
exposes the same numbers with `"goinfer_confidence": true`.

## An image in, a struct out (GLM-OCR)

[`examples/invoice/main.go`](../examples/invoice/main.go) reads a scanned invoice into a Go struct. The struct is the one source
for both halves of the request: `constrain.TemplateFromStruct(Invoice{})` is the prompt, because GLM-OCR is prompted for extraction
with a JSON *template* (an object of blank values) and not a JSON Schema, and `constrain.GrammarFromStruct(Invoice{})` is the
guarantee. Type the amounts as numbers (`float64`, `int`): the model answers a numeric column with a bare number, and a
string-typed field would leave the grammar only whitespace to emit (see `docs/server.md`). The image reaches the decoder through
`multimodal.QwenPreprocess`, aikit's `vision.LoadGlmOcrVisionEncoder` and `Model.GenerateQwenVL`, with the image block spliced into the
prompt by `multimodal.SpliceImageBlock`. Needs the GLM-OCR checkpoint (`zai-org/GLM-OCR`, about 2.7 GB); about a minute and a half
on a CPU. Over HTTP the same request is an `image_url` part plus `response_format` (`docs/server.md`); in the terminal it is
`goinfer-chat --model ~/models/glm-ocr --image invoice.png --schema invoice.schema.json`.

## What v1.0 will bind

goinfer is pre-1.0, so any of this can still change. Which surfaces the v1.0 tag will hold to
semver was decided on 2026-08-18 in [api-tiers.md](api-tiers.md). Its Hard tier covers what this
page uses from `decoder`, `tokenizer` and `chat`:

- `decoder.Load`, `decoder.Options` (with `Quant`), `Model.Generate`, `Close` and `Config`,
  `decoder.SamplingParams` and `decoder.Generation`;
- the `tokenizer.Tokenizer` methods used here (`Encode`, `EncodeSegments`, `Decode`,
  `TokenText`, `TokenID`, `Has`, `Special`, `ChatTemplate`) and `tokenizer.LoadGGUF`;
- `chat.Detect`, `chat.Meta`, `chat.ErrUnknownTemplate`, `chat.Template`'s `Render` and
  `RenderSegments`, and the `chat.Turn` and `chat.Stops` types;
- from `constrain`: `NewMasker`, the `Masker`'s `Process` and `StopWhenComplete`, `TokenBytes`
  and `SchemaFromStruct`.

Three things this page uses are in neither of that page's lists today: `GrammarFromStruct`,
`JSONSchema`, and the confidence API (`CaptureConfidence`, `FieldConfidence`). Until
api-tiers.md places them, treat them as not yet promised. Two smaller ones are unlisted too:
`Tokenizer.PreTokenizerDecline` and the `Template.Stops` method.

## Run the examples

```bash
go run ./examples/embed      ~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf "Reverse a string in Go"
go run ./examples/structured ~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf "Ann Lee, 34, writes Go and Rust"
go run ./examples/confidence ~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf "I was charged twice for order #123. - Ann"
go run ./examples/invoice    ~/models/glm-ocr testdata/glm_ocr/invoice.png    # GLM-OCR checkpoint, not a GGUF
```
