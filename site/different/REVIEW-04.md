# REVIEW-04: "A Go struct the model can't break"

Draft: `04-a-go-struct-the-model-cant-break.md` (body about 846 words including code blocks). `reviewed:` left empty. Build passes (`cmd/build -drafts`). Nothing run: no tests, no models. Every claim below was read from code or a record.

## Claims and sources

| Claim | Source |
|---|---|
| `GrammarFromStruct(v any) (Grammar, error)` and `SchemaFromStruct(v any) ([]byte, error)` exist; json tags read; `omitempty` and pointer fields optional, others required | `constrain/reflect.go` (`GrammarFromStruct`, `SchemaFromStruct`, `jsonField`, `structSchema`) |
| README snippet (Person, GrammarFromStruct, NewMasker...StopWhenComplete().Process, "shape guaranteed, not magnitude") copied verbatim; `generate`, `toks`, `eos` are stand-ins | `README.md` section "A Go struct the model cannot violate" |
| Real generation code copied from the example (JSONSchema, NewMasker, TokenBytes, Generate, LogitProcessor, StopIDs) | `examples/confidence/main.go` |
| Derived schema for Person{Name, Age, Email omitempty} | `constrain/example_test.go` `ExampleSchemaFromStruct` (`// Output:` line, copied) |
| Mask: disallowed logits set to -inf; EOS blocked until CanEnd; StopWhenComplete | `constrain/constrain.go` (`Process`, `maskID`, `StopWhenComplete`) |
| Keys matched against unseen properties, each at most once, any order | `constrain/schema_grammar.go` (`enterKey`, `keyStep`, `seen` mask) |
| Unsupported keyword = compile error; keyword lists; minimum only 0; >64 props, >64 enum entries, depth 64; freeform object refused; additionalProperties true refused | `constrain/schema.go` (`schemaKeywords`, `checkSchemaKeys`, `nonNegativeKeyword`, `compileObject`, `maxSchemaDepth`, enum check in `compile`) |
| `type` arrays / `oneOf` / `$ref` not supported | same: not in `schemaKeywords`; `type` read only as a string (`compile`) |
| Go types refused: recursive, own UnmarshalJSON, no exported fields; maps and interfaces fall to "unsupported type"; only json tags read; `*T` is optional, schema type is T (no null) | `constrain/reflect.go` (`typeSchema` default, `jsonField`, `structSchema`); the doc comment there calls `*T` "the optional/nullable form" but the code emits no null, so I said "cannot write null" from the code |
| Magnitude not bounded; uint8 can get 99999; unsigned gets minimum 0 | `constrain/reflect.go` doc comment and `typeSchema` |
| 151,936 vocab for Qwen2.5 | `docs/QUEUE.md` G37, "Real vocab (V=151,936 Qwen2.5)" |
| Mask timing: 1.299 ms/step over a 17-token document; 1.21x (pos 64) / 1.18x (pos 512) against 6.2 / 7.4 ms; worst state "<= 1.72x upper bound"; 96.88% plain-string ids; ~1.65x against a ~2 ms step; measured 2026-09-02 | `docs/QUEUE.md`, "## G37" (the "second lever" and "Limits, stated" subsections); date from "Measured 2026-09-02" and commit `2a44d4e2` |
| Test facts: 5 schemas x 200 seeds, independent validator; struct round trip 200, DisallowUnknownFields; 3000 random walks over `JSON()` valid per encoding/json; small made-up vocabulary | `constrain/schema_test.go` (`propertySchemas`, `TestSchema_propertyValidates`, `TestGrammarFromStruct_roundTrip`); `constrain/constrain_test.go` (`TestConstrainedDecode_alwaysValidJSON`, `const trials = 3000`) |
| N-79: a control/special id that is not EOS can appear as text inside a string; deferred | `constrain/constrain.go` comment in `maskID`; `docs/audit-2026-09-10.md` N-79 and its header ("DEFERRED") |
| Length stop reported as finish_reason "length" | `internal/serveapp/openai.go` comment at `effectiveBudget` (~line 1687). The claim that the grammar cannot force a close before max_tokens is from reading `Process` (no forced close); the sibling page 03 says the same of tool calls. Not run |
| `go get` line | `README.md` install section |
| `go run ./examples/confidence ...` command | doc comment atop `examples/confidence/main.go` |
| `--schema` flag on the chat demo | `internal/chatapp/main.go` (flag "schema"); README's `go run ./demo/chat --model ... --schema person.schema.json` (`demo/chat/main.go` imports `internal/chatapp`) |
| Server takes `response_format` json_schema through `constrain.JSONSchema`; `json_object` uses `constrain.JSON()` | `internal/serveapp/openai.go` `grammarFor`; `docs/server.md` |

## Conflicts

1. **Brief says "examples/ (grep for a struct-based example)". There is none.** Only `examples/confidence` and `examples/embed` exist; neither calls `GrammarFromStruct`. The only end-to-end struct code is the README snippet, which is a sketch (`generate`, `toks`, `eos`, `sp` undefined). I showed the README code as written, then the real generation code from `examples/confidence` (a hand-written schema), and said so on the page.
2. **`docs/QUEUE.md` G37's "constrained decoding excludes speculative decoding" is dated 2026-09-02.** Later records (`docs/server.md`, `docs/tasks/task-constrained-confidence.md`) refer to "grammar-fused speculative decoding", which drives the masker. So that limit may be stale in part. I kept only "skips some faster decode paths" and attributed it to the record, without naming spec.
3. **`constrain/reflect.go` doc says `*T` is "the optional/nullable form"; the code makes it optional only.** I followed the code.
4. **`docs/api-tiers.md` lists only `SchemaFromStruct` (and `NewMasker`, `Process`, etc.) under the stable tier, not `GrammarFromStruct` or `JSONSchema`.** The page makes no stability claim, so nothing to reconcile, but see question 2.

## Left out as unverified

- Any parse rate or "always valid" figure from a real model run: none exists. Page says so.
- Which machine G37's mask timing ran on. The record says "this box's post-G36 resident-GPU 1.5B token" and G36 names an RTX 2070 SUPER, but G37 does not name the box for the mask timing itself. The page says "one box" and gives no hardware.
- What the server returns (status code) for an unsupported `response_format` schema: `grammarFor` returns the error from `prepare`, but I found no test naming the status. The page does not say "400".
- Whether the mask cost still reads 1.21x today (code touched since: confidence hook adds a nil check; N-tier fixes). Not re-measured; the page is dated 2026-09-02.
- Any stability tier for these functions (see above).

## Questions for the owner

1. Is the README snippet fine to show as-is with its stand-ins, or would you rather this page wait for a small runnable struct example (for instance `examples/struct`)? There is none today.
2. `GrammarFromStruct` and `JSONSchema` are not in `docs/api-tiers.md`'s stable list (only `SchemaFromStruct` is). Should the page avoid implying they are stable? It says nothing about tiers now.
3. The headline number is 1.21x from 2026-09-02, on a 6.2 ms GPU step. Do you want a fresh mask-cost measurement (`TestMaskCost_P20`, heavy, night queue) before this goes out, given the code has moved since?
