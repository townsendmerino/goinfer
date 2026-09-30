---
title: "A Go struct the model can't break"
area: "Structured output"
order: 4
summary: "Give goinfer a Go struct and the model can only write JSON that fits it, so json.Unmarshal into that struct parses. It fixes the shape, not whether the values are right."
stand: "goinfer turns a Go struct's json tags into a grammar and holds the model to it while it writes. The output is always the struct's shape. What each field says is still the model's."
measured: 2026-09-02
reviewed:
facts:
  - {label: "output", value: "always the struct's shape, so json.Unmarshal parses it"}
  - {label: "not guaranteed", value: "that a value is true, or that a number fits its Go type"}
  - {label: "cost of the mask", value: "1.21x a 6.2 ms decode step, one box, one document"}
  - {label: "a parse rate on a real model", value: "not measured"}
doesnt:
  - title: "It doesn't check what the values say."
    text: "The grammar knows a field is a string or an integer, not whether it is correct. An age of 412 fits an int. Free text inside a string is the model's own, and a special token id that is not a stop id can still appear as text there (audit finding N-79, deferred, not fixed)."
  - title: "It doesn't bound a number's size."
    text: "JSON Schema integers have no width, so a uint8 field can be given 99999 and json.Unmarshal will return an error for it. Unsigned fields are stopped from going negative, and that is all."
  - title: "It doesn't take every schema."
    text: "Supported: objects, arrays with minItems and maxItems, string, number, integer, boolean, null, enum, const, and minimum only when it is 0. Any other keyword (pattern, maxLength, oneOf, $ref, uniqueItems) is a compile error, not a silent skip. So are a freeform object with no declared properties, more than 64 properties or enum entries, and nesting deeper than 64."
  - title: "It doesn't take every Go type."
    text: "Maps, interfaces and recursive types are refused, as is a type with its own UnmarshalJSON. Only json tags are read, so a struct has no way to say 'one of these three strings'; for that, write the schema by hand. A pointer field is optional, but the model cannot write null into it."
  - title: "It can't finish a document the token budget won't fit."
    text: "The grammar holds the model to valid prefixes; it does not make it hurry. If generation stops at max_tokens before the closing brace, you get a valid prefix that will not parse. The server reports that stop as finish_reason length."
figures:
  - {text: "1.299 ms/step", source: "docs/QUEUE.md"}
  - {text: "1.21×", source: "docs/QUEUE.md"}
  - {text: "1.18×", source: "docs/QUEUE.md"}
  - {text: "1.72×", source: "docs/QUEUE.md"}
  - {text: "6.2 ms", source: "docs/QUEUE.md"}
  - {text: "7.4 ms", source: "docs/QUEUE.md"}
  - {text: "151,936", source: "docs/QUEUE.md"}
  - {text: "96.88%", source: "docs/QUEUE.md"}
  - {text: "1.65×", source: "docs/QUEUE.md"}
  - {text: "3000", source: "constrain/constrain_test.go"}
  - {text: "200", source: "constrain/schema_test.go"}
sources:
  - "constrain"
  - "docs/QUEUE.md"
  - "docs/server.md"
  - "README.md"
  - "examples/confidence"
---
## The problem

Ask a language model for JSON and you usually get JSON. Usually. Sometimes there is a trailing comma, a key spelled differently, a number where you wanted a string, or a sentence of chat before the opening brace. Go's `json.Unmarshal` rejects each of these. The common answer is a retry loop: parse, fail, ask again, and hope. That costs time, and it never becomes a guarantee.

## What goinfer does

You give the `constrain` package a Go value of your struct type. It reads the type and its `json` tags, derives a JSON Schema, and compiles the schema into a grammar. While the model writes, goinfer removes every next token that would take the output outside that grammar. This is the shape from the README:

```go
type Person struct {
    Name string   `json:"name"`
    Age  int      `json:"age"`
    Tags []string `json:"tags"`
}

g, _ := constrain.GrammarFromStruct(Person{})       // struct → JSON Schema → grammar
sp.LogitProcessor = constrain.NewMasker(g, toks, eos).StopWhenComplete().Process

out := generate(sp)                                  // constrained decode
var p Person
_ = json.Unmarshal(out, &p)                          // shape guaranteed, not magnitude
```

`generate`, `toks` and `eos` are stand-ins there. The real calls are in `examples/confidence`, which uses a hand-written schema instead of a struct and also switches on per-field confidence (the "How sure was it?" page). This is its generation half, unchanged:

```go
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
```

To see what a struct becomes, `constrain.SchemaFromStruct` returns the schema itself. This output is asserted by `ExampleSchemaFromStruct` in the package's tests, for a `Person` with `Name`, `Age` and an `Email string` tagged `json:"email,omitempty"`:

```json
{"additionalProperties":false,"properties":{"age":{"type":"integer"},"email":{"type":"string"},"name":{"type":"string"}},"required":["name","age"],"type":"object"}
```

Fields without `omitempty`, and not pointers, are required. `omitempty` and pointer fields are optional.

## How it works

At each step the model scores every token in its vocabulary (151,936 of them for Qwen2.5). `Masker.Process` runs before the sampler. For each token it asks the grammar whether that token's bytes keep the output a valid start of a document that fits the schema. If not, the token's score becomes minus infinity, so the sampler cannot pick it. Nothing is retried; the bad token is unreachable.

The grammar works on bytes, so a token that spans structure, like a closing quote and brace together, is judged correctly. Keys are matched against the properties not yet written, each at most once, in any order. The end-of-sequence token stays blocked until the document is complete, and `StopWhenComplete` then ends generation at the first complete document.

A schema keyword the compiler cannot enforce is an error when you build the grammar, so you never believe a constraint is in force when it is not.

## What was measured

No record here measures a parse rate on a real model. Nothing was counted as "N answers, N parsed". What does exist is tests of the grammar itself. They drive the masker with random choices over a small made-up vocabulary, not a model, and assert that it never dead-ends and that the output is valid:

<table>
<thead><tr><th>what</th><th>what it asserts</th><th>where</th></tr></thead>
<tbody>
<tr><td>5 schemas, 200 random generations each</td><td>output parses and conforms, checked by an independent validator</td><td><code>TestSchema_propertyValidates</code></td></tr>
<tr><td>a struct with nested and optional fields, 200 generations</td><td>output unmarshals into the struct with unknown fields disallowed</td><td><code>TestGrammarFromStruct_roundTrip</code></td></tr>
<tr><td>3000 random walks over the plain JSON grammar</td><td>every finished document is valid per <code>encoding/json</code></td><td><code>TestConstrainedDecode_alwaysValidJSON</code></td></tr>
</tbody>
</table>

The cost of the mask was measured once, on 2026-09-02, and is recorded in `docs/QUEUE.md` (section G37). The mask work is the same size whatever the model is. Timed at every step of a 17-token JSON document, with a 151,936-token vocabulary, it averaged 1.299 ms/step. Against a 1.5B model's resident-GPU decode step, that made constrained decoding 1.21× the unconstrained time at 6.2 ms per step and 1.18× at 7.4 ms. The slowest single grammar state was a 1.72× upper bound, not a typical step.

A large part of that came from one fix: inside a string, 96.88% of the vocabulary is legal without walking the grammar, so goinfer answers those with one bit test.

The record also says where this stops holding. One box, one vocabulary and one small struct were timed. The mask does not get cheaper for a faster model, so against a step of about 2 ms it would be about 1.65×. And it left two costs unsized: a request with a logit processor skips some faster decode paths.

## Use it

```sh
go get github.com/townsendmerino/goinfer/decoder@latest github.com/townsendmerino/goinfer/tokenizer@latest
```

Then import `github.com/townsendmerino/goinfer/constrain` beside `decoder`, `tokenizer` and `chat`. The complete program to copy is `examples/confidence`:

```sh
go run ./examples/confidence ~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf "I was charged twice for order #123. - Ann"
```

The chat demo takes a schema file with `--schema` (from the README: `go run ./demo/chat --model … --schema person.schema.json`). The server takes the same subset as `response_format` with `{"type":"json_schema", …}`; that path has no Go struct in it, so it takes a schema. Sketch, not run: a Go client can call `constrain.SchemaFromStruct(Person{})` and send those bytes as the `schema`.
