# constrain: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `constrain`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

## Masker.isEOS

Moved from `constrain/constrain.go` (the comment above `Masker.isEOS`) on 2026-10-09.

```text
isEOS is indexed, not mapped: it is probed once per vocab id per decode step —
151,936 probes/step on Qwen2.5 — and a map lookup there was pure overhead on the
hottest loop in constrained decoding (audit P-20's cheapest lever, measured).
Sized to cover the largest EOS id; ids past the end are simply not EOS, which is the
same answer the map gave for an absent key.
```

## Masker.plainOK

Moved from `constrain/constrain.go` (the comment above `Masker.plainOK`) on 2026-10-09.

```text
plainOK marks ids that are unconditionally legal inside a JSON string and leave the
string state unchanged — 96.88% of a real vocab. One bit test replaces a grammar walk
for those whenever the grammar reports a plain-string state (plainstring.go).
```

## Masker.maskID

Moved from `constrain/constrain.go` (the comment above `Masker.maskID`) on 2026-10-09.

```text
maskID reports whether token id must be masked in grammar state g. It is the ONE
masking rule; Process and MaskAt both call it, because they had two verbatim copies
of it and M-27 was a defect in the copy — the shape this audit keeps turning up.
```

## Masker.maskID: control tokens

Moved from `constrain/constrain.go` (the comment above `Masker.maskID`) on 2026-10-09.

```text
N-79 (docs/audit-2026-09-10.md, investigated 2026-09-16, NOT fixed — deferred, see below):
only ids in eosIDs are masked-until-canEnd; every other id, including a special/control
token that is neither EOS nor a template stop id, is judged purely by TryBytes(b) below. A
control token's literal surface (tokenizer.TokenText's documented behavior, see
sentencepiece.go's TokenText) is ordinary printable text with no '"' or backslash, which is
plain-string-legal JSON content — so inside a string value TryBytes accepts it like any
other run of bytes, and a real control-token id can leak into constrained output as if it
were content. Fixing this needs a DIFFERENT semantics than eosIDs: eosIDs are masked only
UNTIL canEnd, but a control token must be forbidden ALWAYS, everywhere, including mid-
string — that is new plumbing (an always-forbidden id set threaded through NewMasker and
every call site: internal/serveapp/openai.go, internal/chatapp/main.go,
demo/agent/agent/agent.go, plus a tokenizer-side way to enumerate "special/control ids"
uniformly across byte-level/SentencePiece/WordPiece modes — isAdded exists but is
unexported and not obviously complete for this purpose), not a local fix to this function.
Deferred to individual review rather than bolted on here.
```

## Masker.maskID: StopWhenComplete whitespace rule

Moved from `constrain/constrain.go` (the comment above `Masker.maskID`) on 2026-10-09.

```text
StopWhenComplete: stop at the first complete document rather than trailing to
maxTokens. M-27: this used to mask EVERY non-EOS token at a completion point,
which conflates MAY-end with MUST-end. CanEnd is true after `1` for a top-level
number — but `12` is a longer legal document, not trailing filler, so blanket
masking made `{"type":"integer"}` return exactly one digit and
`{"enum":[1,10,100]}` able to produce only `1`.

What StopWhenComplete is actually for is suppressing the WHITESPACE the grammars
permit at every structural boundary, so that is what it suppresses. A token that
genuinely extends the VALUE has already passed TryBytes above and is kept. This
needs no per-grammar may-end/must-end split: whitespace-only is the exact
property, and it is the same one for json, schema and tool grammars.
```

## Per-field confidence: what the number is

Moved from `constrain/confidence.go` (the comment in the file-level comment of `constrain/confidence.go`) on 2026-10-09.

```text
confidence unless a fitted temperature was supplied (FieldConfidence.Calibrated). C0 measured that it discriminates
(a low-confidence value is wrong more often: AUROC 0.85 / 0.73 / 0.68 for enum / boolean / integer on the 1.5B,
docs/measurements/confidence-c0-2026-09-27.md); it did not measure calibration.
```

## LazyMasker

Moved from `constrain/lazy.go` (the comment above `LazyMasker`) on 2026-10-09.

```text
complete. Task T2, docs/tasks/task-tool-grammar-union-2026-09.md: under tool_choice "auto" a
prose answer must stay legal (ground rule 1), so the constraint cannot start at token 1 — it
arms on the model's own decision to call, and from there the call cannot be malformed.
Disarming after each complete call means a second call in the same turn is constrained
independently (T5, decided: repeated wrapper, each constrained).
```

## The plain-string fast path

Moved from `constrain/plainstring.go` (the comment at the top of `constrain/plainstring.go`) on 2026-10-09.

```text
The plain-string fast path.

Masking is O(V) grammar walks per decode step — 151,936 of them on Qwen2.5 — and the
measured cost is dominated by TryBytes's snapshot/restore of the frame stack rather than by
the byte walk itself (audit P-20, measured in G37: 6.03 ms/step at `fsStr`, and the cost
tracks stack DEPTH, not token length).

But inside a JSON string, both grammars answer with the same three-line rule:

	'"'      → closes the string   (legal, changes state)
	'\\'     → escape              (legal, changes state)
	b < 0x20 → illegal
	otherwise → legal, AND THE STATE DOES NOT MOVE

So for any token containing none of those three byte classes, legality inside a string is a
property of the TOKEN ALONE, not of the grammar — precomputable once per vocabulary and
answerable with one bit test. Measured on Qwen2.5's vocab: 96.88% of ids qualify (1,226 ids,
0.81%, contain '"' or '\\'; 3,518, 2.32%, contain a control byte).

EXACT, not probabilistic. A Bloom filter was considered and is dominated here: its false
positives would force the very walk being avoided, and at 19 KB for the whole vocabulary
there is no space pressure to trade accuracy for. TestPlainString_exact proves the fast path
agrees with the full walk for EVERY id, rather than sampling.

Conservative by construction: only "definitely legal and state-invariant" is fast-pathed.
Control-byte tokens are NOT classified illegal here even though most are, because a token
like `"` + 0x0A closes the string before the control byte is read and is then judged in a
different state. Everything not provably safe takes the ordinary walk.
```

## maxStructuralWS

Moved from `constrain/json.go` (the comment above `maxStructuralWS`) on 2026-10-09.

```text
maxStructuralWS bounds how many whitespace bytes a grammar accepts in a row BETWEEN JSON tokens (after a
'{', ',' or '[', before a ':' or '}', or after the document is complete). Without a bound, whitespace is legal
at every structural boundary, so when the model's preferred token is illegal there (it wants a bare number where
the schema says "string"; a string needs '"' first) the mask leaves whitespace as the top legal token and
generation pads whitespace until max_tokens, silently, with no error (found 2026-10-02 on a GLM-OCR extraction
with string-typed amounts; docs/measurements/glm-ocr-o5-2026-10/string_typed_quantity_whitespace_runaway.txt).

64 is generous for formatting (llama.cpp's JSON grammar allows one newline plus 20 spaces) and still ends a
runaway in a few tokens. It limits only FORMATTING: no value the schema allows becomes unreachable. String
content is not structural, so a string's own spaces never count toward it.
```

## maxValueWS

Moved from `constrain/json.go` (the comment above `maxValueWS`) on 2026-10-09.

```text
maxValueWS is the tighter bound on whitespace between a ':' and the value it introduces (the schema grammar's
fsValue state, which is only ever the document root or an object member's value). That is the position where the
runaway happens, and a long pad there is also what makes the forced token's CONTEXT unnatural: measured on the
same GLM-OCR invoice with every numeric field typed string (CUDA int4, 2026-10-02), a 64-byte pad after
"quantity": finished but filled the numbers with junk ("", "The"; 1,546 tokens), 4 gave junk and an extra line
item (964 tokens), and 1, the canonical `": "`, gave six line items and correct amounts (445 tokens). Real
output has zero or one space there, so one is the whole bound.
```

## GrammarFromStruct

Moved from `constrain/reflect.go` (the comment above `GrammarFromStruct`) on 2026-10-09.

```text
WHAT IS NOT is MAGNITUDE. JSON Schema integer has no width, so a uint8 field can be
given 99999 and json.Unmarshal will return an error for it. This used to read "always
succeeds", which was false for three separate reasons (M-28) — the other two are now
compile errors rather than silently wrong schemas: a type with its own UnmarshalJSON
is refused (its fields do not describe the JSON it accepts) unless it decodes from a
string via UnmarshalText, in which case it maps to "string"; and a struct with no
exported fields is refused instead of compiling to "{} only".
```

## structSchema: embedded fields of unexported type

Moved from `constrain/reflect.go` (the comment in `structSchema`) on 2026-10-09.

```text
V-14 (docs/review-2026-09-04.md): an anonymous field's reflect name IS its type
name, so an embedded struct of UNEXPORTED type (type base struct{...}; embedded as
`base`) reads IsExported()==false and used to be skipped here entirely — before ever
reaching the promotion branch below. encoding/json does NOT skip it: its own
typeFields has the identical special case ("do not ignore embedded fields of
unexported struct types since they may have exported fields"), so an exported
promoted field from an unexported-typed embed silently vanished from the schema
while json.Unmarshal still populated it — M-28's exact silent-zero-field outcome,
for the variant M-28's own fix didn't cover.
```

## hasExportedFields

Moved from `constrain/reflect.go` (the comment above `hasExportedFields`) on 2026-10-09.

```text
hasExportedFields reports whether t has at least one exported, non-json:"-" field.

N-75 (docs/audit-2026-09-10.md): this used to skip EVERY unexported field uniformly,
including an anonymous embed of an unexported-TYPE struct — but encoding/json (and this
package's own structSchema, fixed for the same reason at V-14, docs/review-2026-09-04.md)
still promotes THAT struct's own exported fields; only the embed's field NAME being
unexported (it equals the type name) doesn't mean it carries no exported content. The gap
was inconsistent rather than silent: SchemaFromStruct calls structSchema directly for the
TOP-level struct (never through this gate), so a struct shaped this way was accepted there
but refused the moment the identical shape appeared as a NESTED field type (typeSchema's
own hasExportedFields gate, which every non-top-level struct goes through) — "has no
exported fields" for a type that, one level up, plainly did.
```

## JSONSchema: trailing data

Moved from `constrain/schema.go` (the comment in `JSONSchema`) on 2026-10-09.

```text
Decoder.Decode stops after the first top-level value and leaves the rest of the
reader unread — unlike json.Unmarshal, it does NOT reject trailing garbage, so
`{"type":"number"}0` compiled as if the schema were just the object (found by
FuzzJSONSchema, which uses json.Unmarshal as its own independent oracle and
disagreed). The same principle this package already applies to an unsupported
keyword — a silently-ignored piece of input is worse than a loud compile error —
applies here too: a caller who accidentally concatenates or truncates a schema
string should not get back a Grammar that quietly compiled a PREFIX of it.
```

## maxSchemaDepth

Moved from `constrain/schema.go` (the comment above `maxSchemaDepth`) on 2026-10-09.

```text
maxSchemaDepth caps object/array nesting (M-40, docs/audit-2026-09-10.md): unbounded depth let
a several-MB `response_format` body force the model down ~160k levels of
{"type":"array","minItems":1,"items":...}, making one Process step's mask-frame-stack copy
O(vocab × depth) — 151936 × 5000 × 64 B ≈ 49 GB of memcpy per step at depth 5000. 64 is
generous against any schema a human writes (the package's own >64 properties/enum-entries caps
use the same order of magnitude) and small against the attack shape.
```

## nonNegativeKeyword

Moved from `constrain/schema.go` (the comment above `nonNegativeKeyword`) on 2026-10-09.

```text
nonNegativeKeyword reads `minimum`, which is supported for the value 0 ONLY — the
grammar can enforce "no leading minus" as a prefix rule, which is all minimum:0 is,
but a general numeric bound needs magnitude comparison the byte FSM does not do.
Anything else is refused rather than silently ignored, per this compiler's rule that
an unenforceable constraint is loud. `minimum` was previously not a known keyword at
all, so every schema carrying it already errored; this only ever widens what compiles.

SchemaFromStruct emits minimum:0 for the unsigned Go kinds (M-28).
```

## schemaGrammar.CanEnd

Moved from `constrain/schema_grammar.go` (the comment above `schemaGrammar.CanEnd`) on 2026-10-09.

```text
MAY-end, not MUST-end. `1` satisfies this and `12` is still reachable — the old
comment here said "done as soon as it can't extend", which describes neither this
function nor any caller, and StopWhenComplete acted on that reading and truncated
every top-level number to one digit (M-27). Extension is the caller's business:
Masker.maskID keeps any token that genuinely extends the value.
```

## toolGrammar.InPlainString

Moved from `constrain/tool_grammar.go` (the comment above `toolGrammar.InPlainString`) on 2026-10-09.

```text
InPlainString (P-17, audit-2026-09-10) lets forced tool-call decoding take the same
plain-string fast path (P-20, audit-2026-09-02) every other JSON-shaped grammar already gets.
Without it, plainStringGrammar's type assertion (constrain/plainstring.go's inPlainString)
simply never matches a *toolGrammar, so every forced tool call paid the full walk on the
prefix/suffix's own JSON body — mostly string content, the exact case the fast path exists
for. Only meaningful during phase 1 (the JSON value): phases 0/2/3 are the literal
prefix/suffix/done, never a JSON string, and inner's own InPlainString is only valid to
consult once inner is actually the grammar in play.
```

## closeEmptyToolObject

Moved from `constrain/tool_grammar.go` (the comment above `closeEmptyToolObject`) on 2026-10-09.

```text
closeEmptyToolObject rewrites a no-argument tool schema into the closed-empty form
the compiler can build (M-30).

`{"type":"object","properties":{}}` is THE canonical no-argument tool schema —
OpenAI's own examples, most MCP servers, and every Pydantic/zod tool with no
parameters emit it. compile() rejects an object with no properties and no
`additionalProperties:false`, correctly, because in JSON Schema that shape means
"any object" and the grammar can only build a closed one. But a TOOL that declared
its parameters and declared NONE means it takes no arguments, and reading it as
"any object" is the wrong of the two readings. So the narrowing happens here, in
the tool path where the extra context justifies it, and compile() is left strict —
an omitted `parameters` already mapped to this same closed-empty form, so the
explicit spelling now behaves like the implicit one rather than being a 400.

Only the ambiguous case is touched: an explicit additionalProperties (true or
false) is left exactly as written, so `additionalProperties:true` still reaches
compile() and is still refused rather than being silently narrowed.
```

## ToolCallsGrammar

Moved from `constrain/tools_union.go` (the comment above `ToolCallsGrammar`) on 2026-10-09.

```text
ToolCallsGrammar constrains ONE tool call to any of several tools (task T1,
docs/tasks/task-tool-grammar-union-2026-09.md). It is the union of the single-tool
grammars ToolCallGrammar already builds, run as parallel branches: a byte is legal iff at
least one live branch accepts it, and a branch that rejects a committed byte is dropped.
Every branch shares the wrapper and the object shape; they differ only in the "name"
const and in what follows it, so the set collapses to one branch as soon as the name
discriminates — and, because each branch is the complete single-tool grammar, that holds
whichever order the model writes the object's keys in.

Provable by construction: the language is exactly the union of the single-tool languages,
so every string it accepts is a well-formed call to exactly one supplied tool, with
arguments matching that tool's schema, and no string naming an absent tool is accepted.
schemaGrammar is not modified (the task's ground rule 3).

Duplicate names collapse to their first spec. An empty tool list, or a tool whose schema
cannot be compiled, is an error: the caller then decodes unconstrained, as today, rather
than silently dropping a tool the model was offered (ground rule 1 — the model keeps its
choice).
```

## TestConfidenceCost_C1

Moved from `constrain/confidence_cost_test.go` (the comment above `TestConfidenceCost_C1`) on 2026-10-09.

```text
TestConfidenceCost_C1 prices CaptureConfidence against plain masking, per decode step, at three grammar states on
a real vocabulary (V = 151,936, Qwen): an object key, an enum value, and inside a free string. C0 measured an
every-position readout at 1.44–4.20% of a decode token (docs/measurements/confidence-c0-2026-09-27.md); C1 reads
only outside free strings, where it keeps the legal tokens' entries too. Min of N, interleaved on/off.
```

## TestMaskCost_P20

Moved from `constrain/maskcost_test.go` (the comment above `TestMaskCost_P20`) on 2026-10-09.

```text
TestMaskCost_P20 measures what audit item P-20 only ESTIMATED.

P-20 says `Masker.Process` is O(V) grammar walks per decode step and reasons: "Estimate
40–120 ns/token → 6–30 ms per step against ~2–5 ms per resident-GPU decode step —
constrained decoding plausibly 3–10× slower per token on GPU", closing with the instruction
this test carries out: "Measure one Process call at fsStr and at fsObjKeyOrClose for
V=151,936 against the unconstrained step."

It matters more than an optimisation note, which is why it is measured before anything is
designed on top of it: constrained generation is the README's headline promise ("a Go struct
the model cannot violate"), and whether it costs 1.2× or 10× decides whether that promise is
usable on the fast backends or has to be documented as slow.

Method: MaskAt is Process's hot loop without the commit, so driving a grammar to a chosen
state by committing BYTES and then timing MaskAt isolates exactly the per-step masking cost
at that state — no tokenizer round-trip, no decode, nothing else in the sample. Real vocab
(V=151,936 Qwen tokens), min-of-N to trim scheduler noise.
```

## TestMaskCost_P20.comparison

Moved from `constrain/maskcost_test.go` (the comment above `TestMaskCost_P20.comparison`) on 2026-10-09.

```text
The comparison P-20 asks for. Decode-step times are this box's measured resident-GPU
numbers for the 1.5B AFTER the G35/G36 kernel work (docs/QUEUE.md): the whole token is
~6.2 ms at pos 64 and ~7.4 ms at pos 512, so the mask is compared against the cheaper
(harder) end. Quoting the pre-G36 figure would flatter the mask by ~3x.
```

## fuzz_test.header

Moved from `constrain/fuzz_test.go` (the comment at the top of the file) on 2026-10-09.

```text
Track 2.1 (testing campaign): constrain is the only ATTACKER-SUPPLIED grammar
surface — cmd/serve compiles a caller's response_format JSON Schema on every
request via JSONSchema. The contract is the repo promise: a typed error or a
clean compile, never a panic/hang, and — the structural property — if a schema
compiles, the masker it produces must always be able to drive SOME complete,
conforming document (it can never paint itself into a corner where no token,
not even EOS, is legal). These fuzz targets enforce both.
```

## FuzzJSONSchema.usenumber

Moved from `constrain/fuzz_test.go` (the comment inside `FuzzJSONSchema`) on 2026-10-09.

```text
Re-parse for the independent conformance oracle (schema compiled, so it
is within the supported subset `conforms` understands). UseNumber, matching
JSONSchema's own parse and the generated-output decode in driveAndValidate
below: a plain json.Unmarshal here decoded an enum/const like 0.0 as
float64(0), which re-marshals as "0" — while the SAME literal surviving
through the grammar (encodeLiteral keeps the source json.Number text
verbatim, M-29) and back through driveAndValidate's UseNumber decode stays
"0.0". eqJSON then compared "0" against "0.0" and flagged a correct,
conforming document as non-conformant — found by fuzzing past what CI's
time-boxed run reached (schema {"enum":[0.0]}, corpus
testdata/fuzz/FuzzJSONSchema/enum_float_literal_precision).
```

## TestForcedBytesRun

Moved from `constrain/forced_run_test.go` (the comment above `TestForcedBytesRun`) on 2026-10-09.

```text
TestForcedBytesRun gates the BYTE-level forced-run primitive (the BPE-appropriate
one). After `{"` the only property "k" forces the key char + its closing quote at
the byte level — the same bytes ForcedRun finds, but byte-level forcing keeps
firing where token-level forcing wouldn't on a real vocab (inc-2 finding).
```

## TestMasker_stopWhenComplete_scalarsMayStillExtend

Moved from `constrain/constrain_test.go` (the comment above `TestMasker_stopWhenComplete_scalarsMayStillExtend`) on 2026-10-09.

```text
M-27: StopWhenComplete truncated a top-level scalar at its first completion point.

CanEnd is a MAY-end predicate — `1` is a complete integer document and `12` is a longer
one — but StopWhenComplete read it as MUST-end and masked every non-EOS token there. So
`response_format: {"type":"integer"}` could only ever return a SINGLE DIGIT, and
`{"enum":[1,10,100]}` could only ever produce `1`. No test caught it because none drove a
TOP-LEVEL scalar: every existing case is an object or array, whose completion point really
does admit nothing but whitespace, so the bug is invisible there.

This drives the real Masker and asks what it permits, rather than asserting a generated
string — the defect is in the mask, and a sampler that happened to pick EOS would hide it.
```

## TestToolGrammar_plainStringExact

Moved from `constrain/plainstring_test.go` (the comment above `TestToolGrammar_plainStringExact`) on 2026-10-09.

```text
TestToolGrammar_plainStringExact is P-17 (audit-2026-09-10): toolGrammar had no InPlainString
at all, so inPlainString's type assertion never matched it and every forced tool call paid
the full walk on its own JSON body — the exact regression TestPlainString_exact above already
gates for schemaGrammar/jsonGrammar directly. Same harness, applied to *toolGrammar's own
wrapped shape (a literal prefix, then {"name":const,"arguments":<paramSchema>}, then a literal
suffix) so a wrong phase boundary (e.g. treating the literal prefix/suffix as plain-string-able)
would be caught here, not just proven absent by construction.
```

## TestGrammarFromStruct_byteSliceAndFixedArray

Moved from `constrain/schema_test.go` (the comment above `TestGrammarFromStruct_byteSliceAndFixedArray`) on 2026-10-09.

```text
TestGrammarFromStruct_byteSliceAndFixedArray is N-74 (docs/audit-2026-09-10.md):
  - []byte used to map to {"type":"array","items":{"type":"integer"}}, but encoding/json's
    Marshal/Unmarshal treat []byte as a SPECIAL CASE — a base64-encoded STRING, never an
    element-wise array of integers — so every grammar-legal output was guaranteed to fail
    json.Unmarshal. It must map to {"type":"string"} instead.
  - A fixed-size Go array ([N]T) got no minItems/maxItems, so the grammar could legally
    produce the wrong element count; json.Unmarshal into [N]T does not error on that (it
    silently truncates or zero-pads), so the schema's "shape is guaranteed" promise (M-28,
    09-02) held even less than the slice case — no error ANYWHERE, just silently wrong data.

Not covered here: base64 CONTENT validity. A random grammar-legal string is not guaranteed to
be valid base64 (this package has no "pattern" JSON Schema keyword to constrain that), so this
checks the SCHEMA shape directly rather than fuzzing a full struct round-trip — the fix's own
scope is the "array vs string" and "no length limit" shape guarantees M-28 is about, matching
what json.Unmarshal actually rejects on SHAPE (before ever getting to whether the bytes decode).
```

## TestSchema_rejectsUnsatisfiable.maxItems

Moved from `constrain/schema_test.go` (the comment inside `TestSchema_rejectsUnsatisfiable`) on 2026-10-09.

```text
N-76 (docs/audit-2026-09-10.md): a maxItems too large to fit an int used to convert
via implementation-defined float64->int behavior, which can come back NEGATIVE — and
maxItems<0 means "unbounded" (the opposite of what a huge bound should mean).
```

## TestSchemaFromStruct_unexportedEmbedIsStillPromoted

Moved from `constrain/schema_test.go` (the comment above `TestSchemaFromStruct_unexportedEmbedIsStillPromoted`) on 2026-10-09.

```text
TestSchemaFromStruct_unexportedEmbedIsStillPromoted pins V-14 (docs/review-2026-09-04.md):
an anonymous field's reflect name IS its type name, so an embedded struct of UNEXPORTED type
reads f.IsExported()==false — structSchema used to skip it on that check alone, before ever
reaching the promotion logic M-28 added. encoding/json does not skip it: its own typeFields
has the identical special case ("do not ignore embedded fields of unexported struct types
since they may have exported fields"), so json.Unmarshal still promotes `id`. Same silent
zero-field outcome M-28 fixed for the exported-embed case, left open for this one.
```

## TestSchemaFromStruct_unexportedEmbedPromotedAsNestedFieldToo

Moved from `constrain/schema_test.go` (the comment above `TestSchemaFromStruct_unexportedEmbedPromotedAsNestedFieldToo`) on 2026-10-09.

```text
TestSchemaFromStruct_unexportedEmbedPromotedAsNestedFieldToo is N-75 (docs/audit-2026-09-10.md):
the SAME shape as TestSchemaFromStruct_unexportedEmbedIsStillPromoted above — accepted there
because SchemaFromStruct calls structSchema directly for the TOP-level struct — used to be
LOUDLY refused the moment it appeared as a NESTED field type instead, because typeSchema's own
hasExportedFields gate (checked before structSchema ever runs for a nested struct) did not
share structSchema's own V-14 fix for an unexported-type anonymous embed.
```

## TestSchemaFromStruct_unsignedRejectsNegative

Moved from `constrain/schema_test.go` (the comment above `TestSchemaFromStruct_unsignedRejectsNegative`) on 2026-10-09.

```text
The unsigned half. This is also where the audit found the "test supplies its own calling
convention" trap: the property test at schema_test.go worked around unbounded integers with
its OWN 15-digit cap, which made the schema look adequate.
```
