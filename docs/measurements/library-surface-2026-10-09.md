# goinfer library surface — cold-user review, 2026-10-09

## 0. Prior contact and sources

**Prior contact: none.** I had not seen this project before this session. I did not read, list, grep or cd into `~/tmcode` or any checkout of the project, did not clone the repo, and did not read source files in the module cache. (Two stack traces from crashes print module-cache file paths; that is the runtime's output, not reading.)

**Sources used, and only these:**
- `README.md` of github.com/townsendmerino/goinfer at tag v0.22.0 (raw README only, no other repo files).
- pkg.go.dev for `github.com/townsendmerino/goinfer@v0.22.0` (the directory list and package synopses; the page shows no identifiers for the root).
- goinfer.dev/docs (index), goinfer.dev/docs/use-from-go/ and goinfer.dev/docs/api-tiers/.
- `go doc -all` on the downloaded module for `decoder`, `tokenizer`, `chat`, `constrain`, `embeddinggemma2`, `pull`, `multimodal`, `metal` (what an editor shows).
- `~/models` listed and read-only.

**Deviation from the task spec:** the spec named v0.20.0; I used **v0.22.0**, the latest published tag (as instructed). The metal submodule was also at v0.22.0.

Working directory: `~/libreview-2026-10-09/`, one module per program (`p1-chat`, `p2-structured`, `p3-tools`, `p4-embed`, `p5-agent`), plus `probes/` (contract probes) and `measure/` (timing script and outputs). Every go command ran with `GOWORK=off` and private `GOMODCACHE`/`GOCACHE` dirs under the working directory.

## 1. The five programs

All five build with `CGO_ENABLED=0` and ran to completion. Program 5 imports the Metal submodule and ran on Metal (`BackendReport()` = `metal`), cgo-free.

| # | program | checkpoint that worked | result |
|---|---|---|---|
| 1 | multi-turn chat via the checkpoint's own template | qwen2.5-coder-0.5b q4_k_m (also Phi-3-mini q4) | 3 turns, prefix KV reused across turns (38, then 66 tokens) |
| 2 | Go struct, then hand-written JSON Schema | qwen2.5-coder-1.5b q4_k_m | both parsed with `json.Unmarshal`, values correct |
| 3 | 2 tools, call → run → result → answer | qwen3.5-0.8b (safetensors) | called `get_weather` and `add`, final answer correct (17°C overcast; 5555) |
| 4 | embeddings + ranking | embeddinggemma-2 | top hit correct for 3 of 3 queries |
| 5 | agent loop, deadline, cancel from another goroutine, **Metal** | qwen3.5-0.8b on Metal | list_files → read_file → answer in 26 s; `CANCEL_AFTER=8s` stopped at 8.014 s (`context.Canceled`), `DEADLINE=8s` at 8.016 s (`DeadlineExceeded`) |

Checkpoints that did not work for program 3/5, and why: qwen2.5-coder-1.5b (writes a fenced bare call under "auto", so nothing parses; under "required" it repeats the same call every turn, a model limitation), qwen2.5-7b-instruct and gemma-3-4b (refused by the fit guard, with 6.0–7.6 GB available on a 16 GB Mac).

### 1.1 Program 1 — chat (`p1-chat/main.go`)

```go
// Program 1: multi-turn chat through the checkpoint's own chat template.
// Written from goinfer.dev/docs/use-from-go and `go doc` only.
package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

func check(err error, what string) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s: %v\n", what, err)
		os.Exit(1)
	}
}

func main() {
	t0 := time.Now()
	path := os.Args[1]
	perToken := os.Getenv("PERTOKEN") == "1" // decode each id alone, as the docs' smallest program does
	quant := os.Getenv("QUANT")
	if quant == "" {
		quant = "int4"
	}

	m, err := decoder.Load(path, decoder.Options{Quant: quant})
	check(err, "load")
	defer m.Close()

	tok, err := tokenizer.LoadGGUF(path)
	check(err, "tokenizer")
	if d := tok.PreTokenizerDecline(); d != "" {
		fmt.Fprintln(os.Stderr, "!! tokenizer:", d)
	}

	tmpl, err := chat.Detect(chat.Meta{ChatTemplate: tok.ChatTemplate(), HasToken: tok.Has})
	if errors.Is(err, chat.ErrUnknownTemplate) {
		fmt.Fprintln(os.Stderr, "unknown chat template; this program needs one")
		os.Exit(2)
	}
	check(err, "detect")
	fmt.Fprintf(os.Stderr, "template=%s backend=%s load=%v\n", tmpl.Name(), m.BackendReport(), time.Since(t0))

	// Stop ids: EOS, end-of-turn, and the template's own stop strings (as examples/structured does).
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

	sess := m.NewSession(0)
	system := "You are a terse assistant. Answer in one sentence."
	var turns []chat.Turn
	questions := []string{
		"What is the capital of France?",
		"And roughly how many people live there?",
		"What river runs through it?",
	}
	first := true
	for _, q := range questions {
		turns = append(turns, chat.Turn{Role: "user", Content: q})
		ids, err := tok.EncodeSegments(tmpl.RenderSegments(system, turns), false)
		check(err, "encode")

		ch, gen := sess.Generate(context.Background(), ids, 96, decoder.SamplingParams{StopIDs: stop})
		var out []int
		fmt.Printf("\nUSER: %s\nASSISTANT: ", q)
		for id := range ch {
			if first {
				fmt.Fprintf(os.Stderr, "[first token after %v since process start]\n", time.Since(t0))
				first = false
			}
			out = append(out, id)
			if perToken {
				s, err := tok.Decode([]int{id})
				check(err, "decode")
				fmt.Print(s)
			}
		}
		check(gen.Err(), "generate")
		reply, err := tok.Decode(out)
		check(err, "decode")
		// Generated stop ids are sent on the channel? strip any literal stop strings to be safe.
		for _, s := range tmpl.Stops().Strings {
			reply = strings.ReplaceAll(reply, s, "")
		}
		if !perToken {
			fmt.Print(reply)
		}
		fmt.Printf("\n  (%d tokens, prefill reused %d)\n", len(out), gen.PrefillReused)
		turns = append(turns, chat.Turn{Role: "assistant", Content: strings.TrimSpace(reply)})
	}
	fmt.Fprintf(os.Stderr, "total %v\n", time.Since(t0))
}
```

### 1.2 Program 2 — structured output (`p2-structured/main.go`)

```go
// Program 2: structured output — a Go struct, then a hand-written JSON Schema.
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
	Email string   `json:"email,omitempty"`
}

// A schema this program did not derive from a struct.
const ticketSchema = `{
  "type": "object",
  "additionalProperties": false,
  "required": ["category", "urgent", "order_ids", "summary"],
  "properties": {
    "category": {"enum": ["billing", "technical", "shipping", "other"]},
    "urgent":   {"type": "boolean"},
    "order_ids": {"type": "array", "items": {"type": "integer"}, "maxItems": 4},
    "summary":  {"type": "string"}
  }
}`

func check(err error, what string) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s: %v\n", what, err)
		os.Exit(1)
	}
}

type rig struct {
	m     *decoder.Model
	tok   *tokenizer.Tokenizer
	tmpl  *chat.Template
	stop  []int
	vocab [][]byte
}

func (r *rig) run(g constrain.Grammar, system, user string) string {
	ids, err := r.tok.EncodeSegments(r.tmpl.RenderSegments(system, []chat.Turn{{Role: "user", Content: user}}), false)
	check(err, "encode")
	mask := constrain.NewMasker(g, r.vocab, r.stop).StopWhenComplete()
	ch, gen := r.m.Generate(context.Background(), ids, 200, decoder.SamplingParams{LogitProcessor: mask.Process, StopIDs: r.stop})
	var out []int
	for id := range ch {
		out = append(out, id)
	}
	check(gen.Err(), "generate")
	raw, err := r.tok.Decode(out)
	check(err, "decode")
	return raw
}

func main() {
	path := os.Args[1]
	m, err := decoder.Load(path, decoder.Options{Quant: "int4"})
	check(err, "load")
	defer m.Close()
	tok, err := tokenizer.LoadGGUF(path)
	check(err, "tokenizer")
	tmpl, err := chat.Detect(chat.Meta{ChatTemplate: tok.ChatTemplate(), HasToken: tok.Has})
	check(err, "detect")

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
	r := &rig{m: m, tok: tok, tmpl: tmpl, stop: stop, vocab: constrain.TokenBytes(m.Config().VocabSize, tok.TokenText)}

	// 1. Go struct.
	g, err := constrain.GrammarFromStruct(Person{})
	check(err, "grammar")
	raw := r.run(g, "Extract the person as JSON.", "Ann Lee, 34, writes Go and Rust. Reach her at ann@example.com.")
	fmt.Println("struct raw:", raw)
	var p Person
	check(json.Unmarshal([]byte(raw), &p), "unmarshal person")
	fmt.Printf("struct parsed: %+v\n", p)

	// 2. A JSON Schema we wrote by hand.
	g2, err := constrain.JSONSchema([]byte(ticketSchema))
	check(err, "schema")
	raw = r.run(g2, "Classify the support ticket as JSON.", "URGENT: I was charged twice for orders 1123 and 1124, please refund one. - Bob")
	fmt.Println("schema raw:", raw)
	var v map[string]any
	check(json.Unmarshal([]byte(raw), &v), "unmarshal ticket")
	fmt.Printf("schema parsed: %v\n", v)

	// 3. A schema outside the subset: must be refused loudly, not dropped.
	_, err = constrain.JSONSchema([]byte(`{"type":"string","pattern":"^[a-z]+$"}`))
	fmt.Println("unsupported keyword 'pattern' ->", err)
	_, err = constrain.JSONSchema([]byte(`{"type":"object","properties":{"a":{"$ref":"#/defs/x"}}}`))
	fmt.Println("unsupported keyword '$ref' ->", err)
	_, err = constrain.JSONSchema([]byte(`{"type":"string","maxLength":5}`))
	fmt.Println("unsupported keyword 'maxLength' ->", err)
	_, err = constrain.JSONSchema([]byte(`{"type":"string","format":"email"}`))
	fmt.Println("unsupported keyword 'format' ->", err)
}
```

Helper probes used for findings (`p2-structured/kwprobe/main.go`, `p2-structured/addlprobe/main.go`) are in the directory.

### 1.3 Program 3 — tool calls (`p3-tools/main.go`)

```go
// Program 3: tool calls — declare two tools, take a call, run it, return its result, take the answer.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/constrain"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

func check(err error, what string) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s: %v\n", what, err)
		os.Exit(1)
	}
}

var tools = []chat.Tool{
	{
		Name:        "get_weather",
		Description: "Current weather for a city.",
		Parameters:  json.RawMessage(`{"type":"object","additionalProperties":false,"required":["city"],"properties":{"city":{"type":"string"},"unit":{"enum":["c","f"]}}}`),
	},
	{
		Name:        "add",
		Description: "Add two integers.",
		Parameters:  json.RawMessage(`{"type":"object","additionalProperties":false,"required":["a","b"],"properties":{"a":{"type":"integer"},"b":{"type":"integer"}}}`),
	},
}

func runTool(c chat.ToolCall) string {
	switch c.Name {
	case "get_weather":
		var a struct{ City, Unit string }
		_ = json.Unmarshal(c.Arguments, &a)
		return fmt.Sprintf(`{"city":%q,"temp_c":17,"sky":"overcast"}`, a.City)
	case "add":
		var a struct{ A, B int }
		_ = json.Unmarshal(c.Arguments, &a)
		return fmt.Sprintf(`{"sum":%d}`, a.A+a.B)
	}
	return `{"error":"unknown tool"}`
}

func main() {
	path := os.Args[1]
	forced := 0 // FORCE=n: steps 0..n-1 must call a tool (tool_choice "required"); later steps are "auto"
	fmt.Sscan(os.Getenv("FORCE"), &forced)
	m, err := decoder.Load(path, decoder.Options{Quant: "int4"})
	check(err, "load")
	defer m.Close()
	// Two tokenizer entry points: LoadGGUF for a .gguf file, Load for a safetensors directory.
	var tok *tokenizer.Tokenizer
	if strings.HasSuffix(path, ".gguf") {
		tok, err = tokenizer.LoadGGUF(path)
	} else {
		tok, err = tokenizer.Load(path)
	}
	check(err, "tokenizer")
	tmpl, err := chat.Detect(chat.Meta{ChatTemplate: tok.ChatTemplate(), HasToken: tok.Has})
	check(err, "detect")
	if !tmpl.SupportsTools() {
		check(fmt.Errorf("template %s has no tool support", tmpl.Name()), "tools")
	}

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

	// The tool grammar: one call to any of our tools, in this family's wrapper.
	prefix, suffix, argsKey, array, ok := tmpl.ToolCallWrapper()
	if !ok {
		check(fmt.Errorf("no JSON-constrainable call form for %s", tmpl.Name()), "wrapper")
	}
	fmt.Fprintf(os.Stderr, "think=%v template=%s wrapper prefix=%q suffix=%q argsKey=%q array=%v\n", tmpl.ThinkMode(), tmpl.Name(), prefix, suffix, argsKey, array)
	var specs []constrain.ToolSpec
	for _, t := range tools {
		specs = append(specs, constrain.ToolSpec{Name: t.Name, Parameters: t.Parameters})
	}
	g, err := constrain.ToolCallsGrammar(prefix, suffix, argsKey, array, specs)
	check(err, "tool grammar")
	vocab := constrain.TokenBytes(m.Config().VocabSize, tok.TokenText)

	system := "You are a helpful assistant. Use the tools when they help."
	turns := []chat.Turn{{Role: "user", Content: "What's the weather in Lisbon right now, and what is 1234 + 4321?"}}

	for step := 0; step < 4; step++ {
		ids, err := tok.EncodeSegments(tmpl.RenderToolsSegments(system, turns, tools), false)
		check(err, "encode")
		opener, _ := tmpl.ToolCallOpener()
		lazy := constrain.NewLazyMasker(constrain.NewMasker(g, vocab, stop), strings.TrimSpace(opener))
		sp := decoder.SamplingParams{StopIDs: stop, LogitProcessor: lazy.Process, LogitProcessorGate: lazy.Gate}
		if step < forced { // tool_choice "required": the call grammar is live from token 1
			sp.LogitProcessor, sp.LogitProcessorGate = constrain.NewMasker(g, vocab, stop).StopWhenComplete().Process, nil
		}
		ch, gen := m.Generate(context.Background(), ids, 256, sp)
		var out []int
		for id := range ch {
			out = append(out, id)
		}
		check(gen.Err(), "generate")
		raw, err := tok.Decode(out)
		check(err, "decode")
		fmt.Printf("\n--- step %d raw output (lazy armed %d times):\n%s\n", step, lazy.Arms(), raw)

		calls, lead := tmpl.ParseToolCallsFor(raw, tools)
		_, answer := tmpl.SplitThink(lead) // the lead still carries a <think> block on reasoning families
		if len(calls) == 0 {
			fmt.Printf("--- final answer: %s\n", answer)
			return
		}
		for i := range calls {
			if calls[i].ID == "" { // ParseToolCalls leaves ID empty on ChatML; the caller mints one
				calls[i].ID = fmt.Sprintf("call_%d_%d", step, i)
			}
		}
		turns = append(turns, chat.Turn{Role: "assistant", Content: lead, ToolCalls: calls})
		for _, c := range calls {
			res := runTool(c)
			fmt.Printf("--- call %s(%s) -> %s\n", c.Name, c.Arguments, res)
			turns = append(turns, chat.Turn{Role: "tool", ToolName: c.Name, ToolCallID: c.ID, Content: res})
		}
	}
	fmt.Println("--- gave up after 4 steps")
}
```

### 1.4 Program 4 — embeddings (`p4-embed/main.go`)

```go
// Program 4: embed a handful of strings with an embedding checkpoint and rank them by similarity.
// The entry point is github.com/townsendmerino/goinfer/embeddinggemma2 (found from the pkg.go.dev
// directory list; neither the README nor use-from-go names it).
package main

import (
	"fmt"
	"os"
	"sort"
	"time"

	"github.com/townsendmerino/goinfer/embeddinggemma2"
)

func check(err error, what string) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s: %v\n", what, err)
		os.Exit(1)
	}
}

func dot(a, b []float32) float64 {
	var s float64
	for i := range a {
		s += float64(a[i]) * float64(b[i])
	}
	return s
}

func main() {
	dir := os.Args[1]
	t0 := time.Now()
	enc, err := embeddinggemma2.LoadEncoder(dir)
	check(err, "load")
	fmt.Fprintf(os.Stderr, "loaded in %v, dim=%d, prompts=%v\n", time.Since(t0), enc.HiddenDim(), enc.PromptNames())

	docs := []string{
		"The Seine flows through Paris before reaching the English Channel.",
		"Go's goroutines are multiplexed onto a small number of OS threads.",
		"A sourdough starter needs regular feeding with flour and water.",
		"Channels let goroutines communicate without sharing memory.",
		"The Eiffel Tower was completed in 1889.",
		"Knead the dough until it passes the windowpane test.",
	}
	queries := []string{"how does concurrency work in Go?", "which river runs through Paris?", "baking bread at home"}

	t1 := time.Now()
	isQ := make([]bool, len(docs)) // false = the checkpoint's "document" prompt
	dv, err := enc.EncodeBatch(docs, isQ, 0)
	check(err, "encode docs")
	fmt.Fprintf(os.Stderr, "embedded %d docs in %v\n", len(docs), time.Since(t1))

	for _, q := range queries {
		qv, err := enc.Encode(q, true) // true = the checkpoint's "query" prompt
		check(err, "encode query")
		type hit struct {
			i int
			s float64
		}
		var hits []hit
		for i := range dv {
			hits = append(hits, hit{i, dot(qv, dv[i])}) // unit-length vectors: dot == cosine
		}
		sort.Slice(hits, func(a, b int) bool { return hits[a].s > hits[b].s })
		fmt.Printf("\nQ: %s\n", q)
		for _, h := range hits[:3] {
			fmt.Printf("  %.3f  %s\n", h.s, docs[h.i])
		}
	}

	// A misspelled prompt name: error, or silent "no prompt"?
	_, _, err = enc.EmbedText("hello", "Retreival-query")
	fmt.Println("\nmisspelled prompt name ->", err)
	// The width a caller might truncate to (Matryoshka): documented widths only.
	fmt.Println("MatryoshkaWidths:", embeddinggemma2.MatryoshkaWidths)
}
```

### 1.5 Program 5 — agent loop on Metal (`p5-agent/main.go`)

```go
// Program 5: an agent loop — tools plus a loop that runs until the model stops calling them,
// with an overall deadline and a cancellation from another goroutine. Runs on Metal.
//
// Modes (env MODE):
//
//	agent   (default) the loop
//	cancel  measure how fast a cancelled ctx stops Generate, and what Err() says
package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"sync/atomic"
	"time"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/constrain"
	"github.com/townsendmerino/goinfer/decoder"
	_ "github.com/townsendmerino/goinfer/metal" // registers the "metal" backend (decoder.Load's doc)
	"github.com/townsendmerino/goinfer/tokenizer"
)

func check(err error, what string) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s: %v\n", what, err)
		os.Exit(1)
	}
}

var files = map[string]string{
	"README.md":  "# shop\nA tiny shop service. See TODO.md.",
	"TODO.md":    "- fix tax rounding in tax.go\n- add tests",
	"tax.go":     "package shop\n\nfunc Tax(cents int) int { return cents * 8 / 100 }",
	"go.mod":     "module shop\n\ngo 1.27",
	"handler.go": "package shop\n\n// HTTP handlers",
}

var tools = []chat.Tool{
	{Name: "list_files", Description: "List the files in the project.", Parameters: json.RawMessage(`{"type":"object","additionalProperties":false,"properties":{}}`)},
	{Name: "read_file", Description: "Read one file by name.", Parameters: json.RawMessage(`{"type":"object","additionalProperties":false,"required":["name"],"properties":{"name":{"type":"string"}}}`)},
}

func runTool(c chat.ToolCall) string {
	switch c.Name {
	case "list_files":
		var names []string
		for n := range files {
			names = append(names, n)
		}
		b, _ := json.Marshal(names)
		return string(b)
	case "read_file":
		var a struct{ Name string }
		_ = json.Unmarshal(c.Arguments, &a)
		if s, ok := files[a.Name]; ok {
			return s
		}
		return `{"error":"no such file"}`
	}
	return `{"error":"unknown tool"}`
}

type agent struct {
	m     *decoder.Model
	tok   *tokenizer.Tokenizer
	tmpl  *chat.Template
	stop  []int
	g     constrain.Grammar
	vocab [][]byte
}

func load(path, backend string) *agent {
	opts := decoder.Options{Quant: "int4", Backend: backend}
	fmt.Sscan(os.Getenv("RESCTX"), &opts.ResidentContext) // GPU-resident KV capacity; 0 = backend default (Metal: 4096)
	m, err := decoder.Load(path, opts)
	check(err, "load")
	fmt.Fprintf(os.Stderr, "backend: %s\n", m.BackendReport())
	var tok *tokenizer.Tokenizer
	if strings.HasSuffix(path, ".gguf") {
		tok, err = tokenizer.LoadGGUF(path)
	} else {
		tok, err = tokenizer.Load(path)
	}
	check(err, "tokenizer")
	tmpl, err := chat.Detect(chat.Meta{ChatTemplate: tok.ChatTemplate(), HasToken: tok.Has})
	check(err, "detect")
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
	prefix, suffix, argsKey, array, ok := tmpl.ToolCallWrapper()
	if !ok {
		check(errors.New("no constrainable tool form"), "wrapper")
	}
	var specs []constrain.ToolSpec
	for _, t := range tools {
		specs = append(specs, constrain.ToolSpec{Name: t.Name, Parameters: t.Parameters})
	}
	g, err := constrain.ToolCallsGrammar(prefix, suffix, argsKey, array, specs)
	check(err, "grammar")
	return &agent{m: m, tok: tok, tmpl: tmpl, stop: stop, g: g, vocab: constrain.TokenBytes(m.Config().VocabSize, tok.TokenText)}
}

// turn runs one model turn; forceCall makes the tool grammar live from token 1 (tool_choice "required").
func (a *agent) turn(ctx context.Context, sess *decoder.Session, system string, turns []chat.Turn, forceCall bool) (string, error) {
	ids, err := a.tok.EncodeSegments(a.tmpl.RenderToolsSegments(system, turns, tools), false)
	if err != nil {
		return "", err
	}
	opener, _ := a.tmpl.ToolCallOpener()
	lazy := constrain.NewLazyMasker(constrain.NewMasker(a.g, a.vocab, a.stop), strings.TrimSpace(opener))
	sp := decoder.SamplingParams{StopIDs: a.stop, LogitProcessor: lazy.Process, LogitProcessorGate: lazy.Gate}
	if forceCall {
		sp.LogitProcessor, sp.LogitProcessorGate = constrain.NewMasker(a.g, a.vocab, a.stop).StopWhenComplete().Process, nil
	}
	ch, gen := sess.Generate(ctx, ids, 384, sp)
	var out []int
	for id := range ch {
		out = append(out, id)
	}
	text, derr := a.tok.Decode(out)
	if err := gen.Err(); err != nil {
		return text, err
	}
	return text, derr
}

func runAgent(a *agent) {
	// Overall deadline for the whole loop.
	deadline := 90 * time.Second
	if d, err := time.ParseDuration(os.Getenv("DEADLINE")); err == nil {
		deadline = d
	}
	ctx, cancel := context.WithTimeout(context.Background(), deadline)
	defer cancel()
	// Cancellation from another goroutine: Ctrl-C, or CANCEL_AFTER=<duration> to simulate a user abort.
	go func() {
		sig := make(chan os.Signal, 1)
		signal.Notify(sig, os.Interrupt)
		var after <-chan time.Time
		if d, err := time.ParseDuration(os.Getenv("CANCEL_AFTER")); err == nil {
			after = time.After(d)
		}
		select {
		case <-sig:
			fmt.Fprintln(os.Stderr, "\n[interrupt: cancelling]")
		case <-after:
			fmt.Fprintln(os.Stderr, "\n[CANCEL_AFTER elapsed: cancelling from watchdog goroutine]")
		case <-ctx.Done():
			return
		}
		cancel()
	}()

	sess := a.m.NewSession(0)
	system := "You are a coding agent. Use the tools to inspect the project, then answer. Call one tool at a time."
	turns := []chat.Turn{{Role: "user", Content: "What does TODO.md ask for, and what file does it point at? Read both before answering."}}
	t0 := time.Now()
	for step := 0; ; step++ {
		raw, err := a.turn(ctx, sess, system, turns, step == 0)
		if err != nil {
			fmt.Printf("\n[step %d stopped after %v: %v | canceled=%v deadline=%v]\n", step, time.Since(t0).Round(time.Millisecond), err,
				errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded))
			return
		}
		calls, lead := a.tmpl.ParseToolCallsFor(raw, tools)
		_, answer := a.tmpl.SplitThink(lead)
		if len(calls) == 0 {
			fmt.Printf("\n[final after %d steps, %v]\n%s\n", step, time.Since(t0).Round(time.Millisecond), answer)
			return
		}
		for i := range calls {
			if calls[i].ID == "" {
				calls[i].ID = fmt.Sprintf("call_%d_%d", step, i)
			}
		}
		turns = append(turns, chat.Turn{Role: "assistant", Content: answer, ToolCalls: calls})
		for _, c := range calls {
			res := runTool(c)
			fmt.Printf("step %d: %s(%s) -> %q\n", step, c.Name, compact(c.Arguments), res)
			turns = append(turns, chat.Turn{Role: "tool", ToolName: c.Name, ToolCallID: c.ID, Content: res})
		}
		if step > 8 {
			fmt.Println("[step budget exhausted]")
			return
		}
	}
}

func compact(b json.RawMessage) string {
	var v any
	if json.Unmarshal(b, &v) != nil {
		return string(b)
	}
	o, _ := json.Marshal(v)
	return string(o)
}

// measureCancel starts a long generation, cancels it from another goroutine after `after`,
// and reports how long the stream took to close and what Err says.
func measureCancel(a *agent, after time.Duration, longPrompt bool) {
	user := "Write a very long story about a lighthouse keeper. At least 2000 words."
	if longPrompt { // ~4000 tokens of prompt, so the cancel lands inside prefill
		user = strings.Repeat("The keeper climbed the stairs and lit the lamp again that night. ", 330) + user
	}
	ids, err := a.tok.EncodeSegments(a.tmpl.RenderSegments("", []chat.Turn{{Role: "user", Content: user}}), false)
	check(err, "encode")
	ctx, cancel := context.WithCancel(context.Background())
	var cancelledAt atomic.Int64 // unix nanos; 0 until cancel() is called
	go func() {
		time.Sleep(after)
		cancelledAt.Store(time.Now().UnixNano())
		cancel()
	}()
	start := time.Now()
	ch, gen := a.m.Generate(ctx, ids, 4000, decoder.SamplingParams{StopIDs: a.stop})
	n, afterCancel := 0, 0
	for range ch {
		n++
		if cancelledAt.Load() != 0 {
			afterCancel++
		}
	}
	closed := time.Now()
	err = gen.Err()
	fmt.Printf("cancel: prompt=%d tokens=%d (after cancel: %d) cancel->close=%v run=%v err=%v errors.Is(Canceled)=%v\n",
		len(ids), n, afterCancel, closed.Sub(time.Unix(0, cancelledAt.Load())).Round(time.Microsecond), closed.Sub(start).Round(time.Millisecond), err, errors.Is(err, context.Canceled))
}

func main() {
	path := os.Args[1]
	backend := os.Getenv("BACKEND")
	if backend == "" {
		backend = "metal"
	}
	a := load(path, backend)
	defer a.m.Close()
	switch os.Getenv("MODE") {
	case "cancel":
		for i := 0; i < 3; i++ {
			measureCancel(a, 3*time.Second, false)
		}
	case "prefilltime": // uncancelled: how long the same long prompt takes to its first token
		for i := 0; i < 3; i++ {
			reps := 330
			fmt.Sscan(os.Getenv("REPS"), &reps)
			user := strings.Repeat("The keeper climbed the stairs and lit the lamp again that night. ", reps) + "Write a very long story."
			ids, _ := a.tok.EncodeSegments(a.tmpl.RenderSegments("", []chat.Turn{{Role: "user", Content: user}}), false)
			t := time.Now()
			ch, gen := a.m.Generate(context.Background(), ids, 1, decoder.SamplingParams{StopIDs: a.stop})
			for range ch {
			}
			fmt.Printf("prefill %d tokens -> first token in %v (err=%v)\n", len(ids), time.Since(t).Round(time.Millisecond), gen.Err())
		}
	case "cancelprefill":
		for i := 0; i < 3; i++ {
			measureCancel(a, 300*time.Millisecond, true)
		}
	default:
		runAgent(a)
	}
}
```

The contract probes are in `probes/main.go` (built twice: `probes-cpu` without the Metal import, `probes-metal` with `-tags withmetal`).

## 2. Friction log

Times are PDT on 2026-10-09, taken from file timestamps.

| time | tag | where | what happened |
|---|---|---|---|
| 16:10 | Wanted and absent | goinfer.dev/docs, pkg.go.dev | pkg.go.dev's root page lists no identifiers, and WebFetch only returns summaries. I read the README and docs pages raw. Not a library defect, but the root page gives a newcomer nothing to start from. |
| 16:11 | Guessed | `metal` package doc | The metal package's `go doc` has no usage line. How to turn Metal on is only in `decoder.Load`'s doc: import the module, then set `Options.Backend: "metal"`. I guessed a blank import. |
| 16:12 | Guessed | stop ids | Neither `Generate` nor `chat.Template` builds the stop-id list. I copied the 12-line loop (EOS + EndOfTurn + `Stops().Strings` via `TokenID`) from use-from-go. Every program here repeats it. |
| 16:13 | Guessed | `Session` vs `Model.Generate` | I used `NewSession` for multi-turn reuse. Later found `Model.Generate` also reuses resident KV on Metal (`PrefillReused`), so when a Session is needed is unclear. |
| 16:14 | Error | `decoder.Load` (Phi-3, 4B Gemma, 7B Qwen) | Fit guard refused the load. The message says "Re-run goinfer-serve with -stream-weights", a CLI remedy. The library's `Options.StreamWeights` is documented as a no-op for non-.giw, non-MoE models. The only library-side fix in the text is the env var `GOINFER_NO_FIT_GUARD=1`. |
| 16:14 | Error (silent) | `tokenizer.Decode` per token | Following use-from-go's smallest program (`tok.Decode([]int{id})` per id), Phi-3 printed `ThecapitalofFranceisParis.Itisalso…`: every space was gone, with no error. |
| 16:14 | Error (silent) | `chat.Phi3Orig` | My system prompt ("one sentence") was dropped without warning. Phi-3 answered in 76 tokens. Documented in `Phi3Orig`'s doc comment only. |
| 16:15 | Wanted and absent | library logging | The library writes to stderr on its own ("fit is tight…", "context capped at N… pass -ctx to choose", "batched prefill declined…") with no logger hook. `-ctx` is a CLI flag; the library has no CPU-side context option (`ResidentContext` is "ignored off the residency path"). |
| 16:16 | Error (silent) | `constrain.JSONSchema` | `"format":"email"` and `"format":"date-time"` compile with no error and are not enforced. The use-from-go doc says an unsupported keyword "is an error … never silently dropped". |
| 16:16 | Error | `constrain.JSONSchema` | `{"type":["string","null"]}` fails with "schema needs a \"type\"…", which is misleading because it has one. The union form is simply unsupported. |
| 16:17 | Guessed | tool grammar wiring | No tool-call example exists in the README, use-from-go or `go doc`. I assembled `ToolCallWrapper` → `ToolSpec`s → `ToolCallsGrammar` → `NewMasker` → `NewLazyMasker(opener)` → `Process` + `Gate` from the doc comments. I guessed that the trigger is `strings.TrimSpace(ToolCallOpener())` and that a "required" turn uses a plain `Masker`. |
| 16:17 | Error | tools on qwen2.5-coder-1.5b | The model wrote a fenced bare call, so the lazy masker never armed and `ParseToolCallsFor` returned prose. This is predicted by `ParseToolCallsFor`'s doc (a model limitation). |
| 16:19 | Guessed | `ParseToolCalls` | The returned `ToolCall.ID` is empty on ChatML. I minted ids myself; nothing says whether that is expected. |
| 16:20 | Guessed | tokenizer for a safetensors dir | `decoder.Load` takes a .gguf or a dir, but the tokenizer needs `LoadGGUF` (file) or `Load` (dir), chosen by the caller. There is no `Model.Tokenizer()` or single loader. |
| 16:23 | Guessed | reasoning models | `ParseToolCalls`'s lead still contained the `<think>…</think>` block (Qwen3.5). I added `tmpl.SplitThink(lead)` after reading the chat doc. |
| 16:22 | Guessed | embeddings entry point | The README and use-from-go never mention a library embedding API. I found `embeddinggemma2` from pkg.go.dev's directory list; it worked first time once found. |
| 16:26 | Error (env) | device | The remote device disconnected mid-run, which killed one run. Not a library issue. |
| 16:32 | Error (silent) | Metal prefill | A 4,304-token prompt on default Options took 40–42 s to first token on "metal", at about 600% CPU (CPU backend: 29 s). At 3,784 tokens it took 4.4 s. Setting `ResidentContext: 8192` brings 4,304 tokens to 4.9 s. No error or warning, and `BackendReport()` still says `metal`. |
| 16:33 | Wanted and absent | backend fallback | Requesting `Backend: "metal"` without the import returns `err == nil` and runs on CPU, printing "build the submodule entrypoint (go build …/metal/cmd/serve)", a CLI remedy. There is no Options field to require a backend (`-require-backend` is serve-only). |
| 16:33 | Wanted and absent | error kinds | "Unsupported model_type", unknown quant and unknown backend are bare `errors.errorString`, with no sentinel or type. |
| 16:33 | Error | `decoder.Load` on a dir containing a .gguf | The error is "read config.json: open config.json: no such file or directory". It does not say to pass the file (use-from-go warns about this guess in a code comment). |
| 16:34 | Error (crash) | `Model.Close` on Metal | `Close()` while a generation is in flight killed the process with `panic: send on closed channel` in `metal.(*resident).chainOpen`. A variant (second model) panicked with `close of closed channel` inside `Close`. |
| 16:35 | Error | `decoder.Load(*.giw, Options{})` | A cpu-arm64 `.giw` with the zero `Backend` (documented as "cpu (default)") fails with "Backend \"\" needs canonical bytes — rebuild with go run ./cmd/prequant…". `Backend: "cpu"` loads it fine. |
| 16:35 | Error (crash) | use after `Close` (.giw) | `Generate` after `Close` on an mmap'd `.giw` model is a SIGSEGV that kills the process. On a GGUF model the same call silently keeps working. |

## 3. Contract answers

### 3.1 Cancellation
- **Does a cancelled context stop Generate promptly?** In decode, yes. The stream closed 13–19 ms after `cancel()` on Metal and 9 µs–17 ms on CPU, with 0 tokens delivered after cancel (3 runs each; 1.5B q4_k_m).
- **During prefill:** cancel at 300 ms into a 4,318-token prefill closed the stream 1.43–1.82 s later on Metal (where that prompt takes about 40 s uncancelled) and 0.84–0.89 s later on CPU (about 29 s uncancelled). Cancellation is checked inside prefill at a coarse grain. This is written nowhere.
- **Does the error say cancelled?** Yes. `gen.Err()` = `context canceled` and `errors.Is(err, context.Canceled)` is true in all 12 runs. In the agent loop on Metal (`p5-agent/agent-cancel-deadline.log`), a watchdog-goroutine cancel at 8 s stopped the loop at 8.014 s with `context.Canceled`, and an 8 s deadline stopped it at 8.016 s with `errors.Is(err, context.DeadlineExceeded)` true.
- **Where written:** `Model.Generate`: "streams … until EOS, a stop id, maxTokens, or ctx cancellation". `Generation.Err`: "the error that ended the stream, or nil if it ended cleanly". Neither says that Err wraps `ctx.Err()` or how long prefill may take to notice. **Not written.**

### 3.2 Errors — can a caller tell kinds apart without string matching?
| kind | how it surfaced | errors.Is/As-able? |
|---|---|---|
| refused by fit guard | `*decoder.FitDeclineError` wrapping `decoder.ErrWontFitResident` | **Yes**, both `Is` and `As` (documented on `ErrWontFitResident`). |
| unknown chat template | `chat.ErrUnknownTemplate` | **Yes**, `Is` (documented). |
| unsupported family | `*errors.errorString` "decoder: unsupported model_type \"embedding_gemma2\" (have: …)" | **No.** Strings only. |
| resident/GPU decline | **never returned as an error.** `Load` falls back to CPU with `err == nil`; the reason is only in `Model.ResidentDecline()` / `BackendReport()` / `EffectiveBackend()` strings. `ResidentDeclineError` exists but a library caller never receives it. | **No.** A caller can compare `EffectiveBackend()` to what it asked for. |
| (also) bad quant / bad backend name | `*errors.errorString` | No. |
| (also) missing file | wraps `*fs.PathError` | `errors.Is(err, fs.ErrNotExist)` works. |

### 3.3 Concurrency — is a `decoder.Model` safe from several goroutines?
- **Measured:** 4 goroutines calling `Model.Generate` at once on one Model produced output token-identical to solo runs, with no error or panic, on both CPU (`CPUConcurrentSafe()=true`) and Metal (`CPUConcurrentSafe()=false`).
- **What is written:**
  - `type Model`: "Goroutine safety follows encoder.Model: Weights are immutable after Load; per-sequence state (the KV cache) is owned by each Generate call, so distinct sequences can run concurrently, but a single KVCache is not shared." The same comment opens with "Model is a loaded Gemma 3 checkpoint", which is stale.
  - `Model.CPUConcurrentSafe`: "whether several generations of this model may run at once on the CPU path, each on its own Session … False for a GPU-resident model — it has ONE claimed resident KV (resBusy) — and for a model that streams its weights."
  - `GenerateVL` (not `Generate`): resident decode "claims the shared resident KV when available and not busy with a concurrent generation; falls back to the CPU decode loop otherwise."
  - `NewCache`: allocates a cache; `KVCache`: "Not goroutine-safe: one cache belongs to one in-flight sequence." `NewSession`: allocates; `Session`: "Not goroutine-safe … backs one in-flight sequence at a time."
  - `Generate` itself promises nothing about concurrency.
- **Answer:** on CPU it is safe and written (Model doc plus `CPUConcurrentSafe`). On a GPU-resident model the docs contradict each other. The type doc says distinct sequences can run concurrently, while `CPUConcurrentSafe` says false. What a concurrent `Generate` actually does on Metal (silently decode on the CPU at CPU speed?) is written only for the VL path. Sessions and caches: one per goroutine, written.

### 3.4 Ownership — what must be Closed, in what order, second Close, use after Close
- **What has Close:** `decoder.Model.Close() error` only. `Session`, `KVCache`, `Tokenizer`, `chat.Template`, `Masker` and `embeddinggemma2.Encoder` have no Close (the Encoder holds about 1.5 GB of f32 weights; release is by GC).
- **Written:** "Close releases backend resources (GPU resident buffers + the backend) and unmaps the .giw mapping … A no-op for the CPU backend with no mapping. Safe to call once after the model is done; the weights must not be touched afterward (the mapping is gone)." "Done" is not defined, there is no `Wait`, and the only way to know a generation finished is to drain its channel.
- **Order:** drain every generation's channel, then `Close`. That ordering is inferred, not written.
- **Second Close (tried):** returns `nil` on CPU and Metal. Not written either way.
- **Use after Close (tried):**
  - GGUF, CPU: `Generate`, `Session.Generate` and `NewSession` keep working with no error.
  - GGUF, Metal: same; output identical (presumably now on the CPU).
  - `.giw`, CPU: **SIGSEGV, process dies** (`linalg.(*WeightMat).Row`).
- **Close while generating (tried):** CPU keeps streaming (199 more tokens, `Err()==nil`). Metal: **`panic: send on closed channel`** in a library goroutine, which the caller cannot recover; the process dies.
- **Two models on Metal:** closing A then using and closing B works when no generation overlaps.

### 3.5 Module pins
- **`go get` lines:** **one** per program. CPU programs: `go get github.com/townsendmerino/goinfer/decoder@v0.22.0 github.com/townsendmerino/goinfer/tokenizer@v0.22.0` covers `chat` and `constrain` too. Embeddings: one line for `embeddinggemma2`. Metal: `go get github.com/townsendmerino/goinfer/metal@v0.22.0` alone was enough to build program 5 (it pulls the root module in). A bare `go get github.com/townsendmerino/goinfer` records the requirement and the build then fails with `missing go.sum entry` (documented, and Go's error names the fix).
- **aikit pin, root vs GPU submodule:** they **agree** at every tag I checked:

| tag | root go.mod aikit | metal go.mod aikit | metal requires root |
|---|---|---|---|
| v0.20.0 | v1.51.1 | v1.51.1 | v0.20.0 |
| v0.21.0 | v1.57.0 | v1.57.0 | v0.21.0 |
| v0.22.0 | v1.57.0 | v1.57.0 | v0.22.0 |

  The `go list -m all` result for program 5 shows `aikit v1.57.0` (only one version), `aikit/gpu v0.33.5`, `goinfer v0.22.0`, `goinfer/metal v0.22.0`, `purego v0.10.1` and `gocudrv v0.3.2`. Note that a CUDA driver binding (`gocudrv`) comes along into a macOS Metal build via `aikit/gpu`.

## 4. Measurements (with provenance)

- **Machine:** Apple M1 Pro (`sysctl -n machdep.cpu.brand_string`), 16 GB, macOS 26.6.2 (25G83). The machine was in normal use, with 6.0–7.6 GB reported available.
- **Go:** go1.27.0 darwin/arm64. **GOPROXY:** `https://proxy.golang.org,direct`. **Network:** wired LAN (en0, no Wi-Fi association); proxy.golang.org reached directly. **Date:** 2026-10-09, about 16:36 PDT.
- **Model for first-token timing:** qwen2.5-coder-0.5b-instruct q4_k_m GGUF, `Quant: "int4"`, CPU backend (program 1 has no Metal import). The model file was already on local disk, so model download is not included.

**`go get` → first token, program 1** (`measure/measure.sh`; `go mod init` + `go get` + `go build` + run until first token):

| run | module cache | go get | go build | process start → first token | get + build + first token |
|---|---|---|---|---|---|
| cold-1 | fresh GOMODCACHE + GOCACHE | 7.25 s | 4.39 s | 0.90 s | **12.54 s** |
| cold-2 | fresh | 6.21 s | 4.25 s | 0.90 s | **11.36 s** |
| cold-3 | fresh | 7.28 s | 4.17 s | 0.88 s | **12.33 s** |
| warm-1 | reuse cold-1 caches | 0.23 s | 0.47 s | 0.92 s | **1.62 s** |
| warm-2 | warm | 0.22 s | 0.27 s | 0.89 s | **1.38 s** |
| warm-3 | warm | 0.23 s | 0.27 s | 0.89 s | **1.40 s** |

Cold spread: 11.36–12.54 s (go get 6.2–7.3 s depends on the network; build 4.2–4.4 s). Warm spread: 1.38–1.62 s. "Cold" means an empty local module cache; the public proxy's own cache was presumably warm.

**Clean `go build` time** (private GOCACHE, `go clean -cache` before each build, warm module cache, `CGO_ENABLED=0`):

| program | run 1 | run 2 | run 3 | spread |
|---|---|---|---|---|
| p1-chat | 4.32 s | 4.16 s | 4.16 s | 4.16–4.32 |
| p2-structured | 4.11 s | 4.27 s | 4.24 s | 4.11–4.27 |
| p3-tools | 4.18 s | 4.11 s | 4.16 s | 4.11–4.18 |
| p4-embed | 2.99 s | 3.01 s | 3.01 s | 2.99–3.01 |
| p5-agent (Metal) | 4.58 s | 4.63 s | 4.52 s | 4.52–4.63 |

**Binary size, `CGO_ENABLED=0`** (deterministic, so measured once). The Metal program needed **no cgo** (purego), so it builds cgo-free as well:

| program | default | `-ldflags='-s -w'` |
|---|---|---|
| p1-chat | 11,706,002 B (11.2 MiB) | 9,177,122 B |
| p2-structured | 11,793,938 B | 9,228,114 B |
| p3-tools | 12,069,490 B | 9,428,578 B |
| p4-embed | 4,734,178 B | 3,155,234 B |
| p5-agent (Metal) | 13,454,530 B (12.8 MiB) | 10,473,282 B |

**Other measured numbers referenced by findings** (3 runs each, same machine, qwen2.5-coder-1.5b q4_k_m):
- Cancel → stream closed during decode: Metal 13.4 / 14.3 / 19.1 ms; CPU 0.009 / 12.7 / 16.7 ms.
- Cancel → stream closed during a 4,318-token prefill (cancelled at 300 ms): Metal 1.43 / 1.48 / 1.82 s; CPU 0.84 / 0.85 / 0.89 s.
- First token for a 4,304-token prompt: Metal default Options 40.4 / 41.4 / 42.1 s; CPU backend 29.3 / 29.3 / 29.5 s; Metal with `ResidentContext: 8192` 4.89 s (first run; later runs 92–95 ms from resident prefix reuse). A 3,784-token prompt on Metal default took 4.35 s.

## 5. Graded findings

Grading follows the pre-registered criteria as written: **must-fix** = a (silent wrong result), b (trap), c (required boilerplate), d (unstated contract); otherwise **should-fix** or **note**. A finding that is a doc gap only, with the code behaving as documented somewhere, is a should-fix.

### Must-fix before v1.0

| id | grade | program | package / symbol | evidence |
|---|---|---|---|---|
| **M1** | must-fix **a** | p2 | `constrain.JSONSchema` (`format`) | `{"type":"string","format":"email"}` and `"format":"date-time"` compile with `err == nil`, and the grammar accepts `"not-an-email"`, `"yesterday"` and `"ab"` (`p2-structured/kwprobe`). use-from-go states: "A keyword or shape outside the subset is an error from JSONSchema, so a constraint that could not be enforced is never silently dropped." Other unsupported keywords (`pattern`, `maxLength`, `minimum/maximum`, `uniqueItems`, `oneOf`, `anyOf`, `$ref`) are refused correctly. `format` is the exception, and it is the keyword most likely to appear in a schema "you already have". |
| **M2** | must-fix **c** | p1, p2, p3, p5, probes | `chat` + `tokenizer` + `decoder` (render → encode → stop ids) | Every program, and use-from-go itself, repeats the same sequence: `chat.Detect(chat.Meta{tok.ChatTemplate(), tok.Has})`, then `RenderSegments`, then `EncodeSegments(…, false)` (addBOS must be false for templates, true for the raw fallback), then a 12-line loop building stop ids from `Special().EOS`, `Special().EndOfTurn` and `tmpl.Stops().Strings` via `TokenID`. Then `Generate(…, SamplingParams{StopIDs: stop})`. Getting addBOS or the stops wrong silently degrades output (the README warns "an instruct model will repeat itself"). There is no `tmpl.StopIDs(tok)`, no `Encode(tmpl, system, turns)` and no chat-level `Generate`. |
| **M3** | must-fix **c** | p1 | `tokenizer.Decode` / `DecodePiece` (streaming text out) | The docs' own smallest program streams with `tok.Decode([]int{id})` per token. On Phi-3 (SentencePiece) that printed `ThecapitalofFranceisParis.Itisalso…`, every space deleted, with no error (`p1-chat/phi3-int4.log`). The correct streaming call is `DecodePiece`, plus buffering of partial UTF-8 across calls, plus stripping the sequence's leading space once ("a demo concern, not the tokenizer's"). Every streaming caller has to hand-write that, and the documented attempt gets it wrong. There is no streaming detokenizer in the surface. |
| **M4** | must-fix **c** | p3, p5 | `chat.Template.ToolCallWrapper` / `ToolCallOpener`, `constrain.ToolCallsGrammar` / `NewLazyMasker` / `Masker`, `SamplingParams.LogitProcessorGate` | Tool calls are a README headline, with no code example anywhere. Each caller assembles about 15 lines: wrapper → `ToolSpec`s → `ToolCallsGrammar` → `NewMasker(g, TokenBytes(...), stop)` → `NewLazyMasker(m, opener)` → `Process` + `Gate`, plus a plain `Masker` for "required". Then `ParseToolCallsFor`, then `SplitThink(lead)` (the lead carries `<think>`), then minting `ToolCall.ID` (left empty). The trigger form (`TrimSpace(opener)`?) was a guess. |
| **M5** | must-fix **b** | probes | `decoder.Load` with a `.giw`, `Options.Backend` zero value | `Load("…int4.cpu-arm64.giw", Options{})` fails: "Backend \"\" needs canonical bytes — rebuild with `go run ./cmd/prequant -target …`". `Options.Backend`'s doc says `"cpu" (default)`, and `Load(…, Options{Backend: "cpu"})` loads the same file fine. The error points at rebuilding the bundle (a source-checkout command), not at the one-word fix. |
| **M6** | must-fix **d** | probes | `decoder.Model.Close` (ownership order) | `Close()` while a generation is in flight on Metal kills the process (`panic: send on closed channel` at `metal.(*resident).chainOpen`, a library goroutine, so it cannot be recovered). On CPU the same call is ignored and the stream continues. The doc says only "Safe to call once after the model is done". "Done" (channel drained? Err read?) is not defined, there is no way to wait for in-flight generations, and nothing warns that violating it crashes. `defer m.Close()` in `main` with a generating goroutine is the obvious code. |
| **M7** | must-fix **d** | probes, p5 | `decoder.Model` / `Model.Generate` concurrency on a GPU-resident model | The `Model` type doc says distinct sequences "can run concurrently". `CPUConcurrentSafe()` says false for a GPU-resident model ("ONE claimed resident KV"). The only statement of what happens when the resident KV is busy (silent fall-back to CPU decode) is on `GenerateVL`. Measured: 4 concurrent `Generate`s on Metal all matched solo output, so it works. A caller cannot find in writing whether it is safe or what speed the extra ones run at. |
| **M8** | must-fix **d** | p5 | `decoder.Model.Generate`, `decoder.Generation.Err` (cancellation) | Measured: decode stops 9 µs–19 ms after cancel, and `Err()` satisfies `errors.Is(err, context.Canceled)`. During a long prefill, the stop takes 0.84–1.82 s. Neither the error identity (that `Err` wraps `ctx.Err()`, so `errors.Is(err, context.Canceled/DeadlineExceeded)` works) nor the prefill granularity is written anywhere. An agent loop needs both to tell "user aborted" from "model failed". |
| **M9** | must-fix **d** | probes | `decoder.Load` error kinds (unsupported family, resident/GPU decline) | Unsupported family, bad quant and bad backend are bare `errors.errorString`. A resident/GPU decline is never an error at all: `Load` returns `err == nil` on CPU, and the reason lives in `ResidentDecline()` / `BackendReport()` strings. `ResidentDeclineError` is exported but unreachable for a library caller. There is no Options field to make a missing backend an error (the serve-only `-require-backend`). Fit guard and unknown template are fine (`ErrWontFitResident` + `*FitDeclineError`, `ErrUnknownTemplate`). |

### Should-fix (19)

| id | program | package / symbol | evidence |
|---|---|---|---|
| S1 | p5 | Metal resident context (`MetalCtxDefault`, `Options.ResidentContext`) | A 4,304-token prompt on default Options: 40–42 s to first token on "metal", against 4.9 s with `ResidentContext: 8192`, and 29 s on the CPU backend. No error or warning; `BackendReport()` still says `metal`. The data is right, only the speed is wrong (so not kind a). The fix is findable via `MetalCtxDefault` and README's "default GPU context on Metal (4096)", so not kind b. A per-call warning or `Generation` field saying "fell off the resident path" would remove it. |
| S2 | p1, p3 | `decoder.Load` fit-guard and capping messages | Library errors and stderr lines prescribe CLI remedies: "Re-run goinfer-serve with -stream-weights", "pass -ctx to choose", "build the submodule entrypoint (go build …/metal/cmd/serve)". `Options.StreamWeights` is documented as a no-op for GGUF. The library has no CPU context cap (`ResidentContext` is "ignored off the residency path"), so the 4B Gemma is refused for pricing 5.5 GB of KV at its full context, with no library knob to ask for less. |
| S3 | p1–p5 | library logging | The library prints to stderr unprompted ("fit is tight…", "context capped…", "batched prefill declined…", "metal backend not built in…") with no `Logger`/`slog` hook in `Options`. |
| S4 | probes | `decoder.Load`, `Options.Backend: "metal"` without the import | `err == nil`, runs on CPU. Documented in `Load` ("a name not compiled into the binary falls back to cpu"), but the printed remedy names the CLI build, not the one-line blank import. |
| S5 | probes | `Model.Close` / use after Close | On a `.giw` model, `Generate` after `Close` is a SIGSEGV. On GGUF it silently works. Documented ("the weights must not be touched afterward"), so not must-fix, but a closed flag returning an `ErrClosed` from `Generate` is cheap. Second `Close` returns nil; that is unstated. |
| S6 | p1 | `chat.Phi3Orig` | A system prompt is dropped silently (documented on `Phi3Orig`). The checkpoint's q4 GGUF selects this template, so `Render(system, …)` loses the caller's instructions with no signal. Could return or flag it. |
| S7 | p4 | `embeddinggemma2` discoverability | The README and use-from-go mention no library embedding API (the README points to the HTTP server). The entry point is only findable from pkg.go.dev's directory list. Once found, `LoadEncoder` / `Encode` / `EncodeBatch` worked first time with good errors. |
| S8 | p3, p5 | tokenizer selection | `decoder.Load` accepts a .gguf or a directory, but the caller must pick `tokenizer.LoadGGUF` or `tokenizer.Load` by path shape. A `Model.Tokenizer()` or one `tokenizer.Open(path)` would remove the branch. |
| S9 | p3 | `chat.Template.ParseToolCalls` | The returned `ToolCall.ID` is empty on ChatML, and the caller must mint ids before rendering history. The lead includes the `<think>` block on reasoning families. Neither is stated on `ParseToolCalls`. |
| S10 | p2 | `constrain.JSONSchema` error text | `{"type":["string","null"]}` is rejected with "schema needs a \"type\", \"enum\", or \"const\"", but it has a type. The error should say "type arrays are not supported; use … ". |
| S11 | probes | `decoder.Load` on a directory that contains a .gguf | The error reads "read config.json: open config.json: no such file or directory". use-from-go's code comment anticipates this exact guess; the error could say "dir has no config.json; pass the .gguf file itself". |
| S12 | — | api-tiers vs README | The README's headline API (`GrammarFromStruct`, `JSONSchema`, `CaptureConfidence`) is in neither api-tiers list. use-from-go says so itself ("treat them as not yet promised"). A v1.0 that does not bind its headline feature is a doc decision still owed. |
| S13 | p1 | `decoder.Model` type doc | It opens with "Model is a loaded Gemma 3 checkpoint", which is stale for a 40-family loader and sits directly on the concurrency statement (M7). |
| S14 | p5 | `metal` package doc | No usage line (blank import + `Backend: "metal"`). The package doc talks about device types and a "spike". |
| S15 | p1 | `Session` vs `Model.Generate` | On Metal, `Model.Generate` already reuses resident KV across calls (`PrefillReused` 92–95 ms reruns). When a `Session` is needed (CPU only?) is not stated on `NewSession` or `Generate`. |
| S16 | p1 | `decoder.Options` zero value | The zero Options loads f32 (about 4× the memory of the CLIs' int4 default) and leaves `EmbedInt4` off while the CLIs turn it on. This is documented, but a library user gets a different model footprint than the CLI with no hint at load time. |
| S17 | p3 | `constrain.NewLazyMasker` trigger | "The grammar must START with the trigger's bytes". `ToolCallWrapper`'s prefix is `"<tool_call>\n"`, and whether the trigger is `ToolCallOpener()` as-is or trimmed is not stated. A `Template.ToolGrammar(tools)` returning a ready LazyMasker would settle it (overlaps M4). |
| S18 | — | pkg.go.dev root page | No package doc at the module root, so the first page a Go user lands on lists directories only. |
| S19 | p3 | `constrain` (absent `additionalProperties`) | A schema with no `additionalProperties` rejects extra keys (it behaves as `false`), while an explicit `true` is an error. Stricter than JSON Schema's default; unstated. Output still validates against the schema, so it is not wrong. |

### Notes
- N1: qwen2.5-coder 1.5B does not make usable tool calls under "auto" and loops under "required". This is consistent with the library's own measurements cited in `ParseToolCallsFor`. Qwen3.5-0.8B worked; the README's recommended 7B could not load on this 16 GB machine.
- N2: Phi-3 at int4 gave coherent answers here, despite `ActivationQuantHazard`'s description of Phi-3 under int8-activation quant. I did not measure quality.
- N3: The metal module brings `github.com/eitamring/gocudrv` (a CUDA driver binding) into macOS builds via `aikit/gpu`.
- N4: Things that worked well: the fit guard's typed error, `ErrUnknownTemplate`, refusal of most unsupported schema keywords with a clear message, the embedding prompt-name error that lists valid names, prefix KV reuse, cancel latency in decode, and token-identical concurrent CPU generation.

## 6. Summary

- **Must-fix (9):** M1 `format` silently unenforced (a); M2 chat render/encode/stop-id boilerplate (c); M3 no streaming detokenizer, and the docs' per-token `Decode` deletes spaces on SentencePiece (c); M4 tool-call grammar wiring boilerplate (c); M5 `.giw` + zero `Backend` fails with a rebuild instruction (b); M6 `Close` during generation crashes on Metal, with "done" undefined (d); M7 GPU-resident concurrency contract contradictory (d); M8 cancellation error identity and prefill latency unwritten (d); M9 unsupported-family and GPU-decline not typed (d).
- **Should-fix:** 19. **Notes:** 4.
- **Blocked or partly blocked:** the README's recommended tool model (Qwen2.5-7B) and Gemma-3-4B were refused by the fit guard at the memory available. Tool calls and the agent were shown on Qwen3.5-0.8B instead. The device disconnected once (one run lost and repeated).
