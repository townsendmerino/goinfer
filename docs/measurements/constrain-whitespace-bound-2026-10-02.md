# constrain: bound structural whitespace, tighter after a colon (2026-10-02)

The defect: a **string-typed numeric column made the grammar loop on whitespace until `max_tokens`, silently**
(found by the GLM-OCR O5 agent: [`glm-ocr-o5-2026-10/string_typed_quantity_whitespace_runaway.txt`](glm-ocr-o5-2026-10/string_typed_quantity_whitespace_runaway.txt)).
Whitespace was legal at every JSON structural boundary with no limit. After `"quantity":` a string schema requires `"`, the
model wanted a bare number (illegal), and the only legal token it rated highest was whitespace, forever.

## The fix

A consecutive-whitespace counter in both grammars (`constrain/json.go`, `constrain/schema_grammar.go`), saved and restored with
the trial snapshot and copied by `Clone`, reset by any other byte; string content is not structural and never counts.
- `maxStructuralWS = 64`: the bound between tokens everywhere (after `{`, `,`, `[`; before `:` or `}`; after the document is
  complete). It limits formatting only; no value the schema allows becomes unreachable.
- `maxValueWS = 1`: the bound between a `:` and its value in the schema grammar (its `fsValue` state, only ever the root or an
  object member's value). The generic JSON grammar keeps only the 64 (its value state also covers array elements after a comma,
  where a newline plus indentation is normal).

## Why 1 and not just 64: measured, but on ONE image and ONE schema

Same input every time: `testdata/glm_ocr/invoice.png` (a procedurally rendered invoice, not a scan), the invoice schema with
every numeric field retyped `string`, `goinfer-chat --image … --schema …`, CUDA int4, greedy, RTX 2070 SUPER. One run per row
(greedy, so deterministic; not a distribution).

| value-position bound | result |
|---|---|
| none (before the fix) | runs to `max_tokens` on whitespace; no usable output (the agent's capture) |
| 64 | finishes, 1,546 tokens, valid JSON, but the values are junk: `quantity ""`, `subtotal "The"`, `total ""` |
| 4 | finishes, 964 tokens, valid JSON, a hallucinated seventh line item, `quantity ", "` |
| **1** | **finishes, 445 tokens, six line items, `unit_price "18.40"`, `amount "220.80"`, `subtotal "1051.20"`, `tax "89.35"`, `total "1140.55"`** (the invoice's true total); `quantity` still `"{0.0}"` on two rows, the model's own difficulty with a string-typed column |

Reading: a long pad after the colon is an out-of-distribution context for the forced `"`, so the model fills the string with
junk; the canonical single space gives it the context it was trained on. This explains the table but is one image; it is not
proof the value is optimal for other models or schemas, and the shorter the bound the more formatting it removes (`": "` and `":"`
are legal; two spaces or `":\n  "` are not). Real model output has zero or one space there.

**No regression for a well-typed schema:** the numeric-typed demo (`goinfer-chat --image testdata/glm_ocr/invoice.png --schema
testdata/glm_ocr/invoice.schema.json`) is byte-identical to the O5 agent's capture before the fix (416 tokens, the same JSON).

## Tests

`constrain/whitespace_bound_test.go` (8 tests, hermetic): the value-position bound and its steer (the opening quote stays legal after
the maximal pad, a bare number stays illegal); the structural bound at three boundaries; reset after any other byte; string content
exempt; trailing whitespace after a complete document (the document can still end); the generic JSON grammar; `TryBytes` leaves the
count alone and `Clone` carries it; and a Masker-level reproduction of the runaway with a scripted model that prefers a bare number,
then whitespace (it must finish as valid JSON, the pad after the colon at most 1).

**Red/green:** with every bound disabled 7 of the 8 go red, including the runaway reproduction (`generation did not finish in 400
steps: the whitespace runaway`); the eighth, "string content is exempt", correctly stays green. With ONLY the value-position bound
removed, exactly the value-position and the runaway tests go red. Whole `constrain` package: 76 PASS, 2 SKIP (the two cost tests, by
design), 0 FAIL. `internal/serveapp` 292 PASS / 51 SKIP / 0 FAIL and chatapp + examples + multimodal 62 PASS / 2 SKIP / 0 FAIL on
the final code.

## What is not done

The grammar is still permissive about whitespace elsewhere (at most 64 bytes, as before). A schema that types a field `string` when
the model expects a number is still a poor schema: the bound makes the failure finish and keeps the content readable, it does not
make `"6"` the model's first choice. The advice to type amounts as numbers stands.
