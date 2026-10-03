# D12 (encoder half) — the Clef record encoder in Go (2026-10-02)

`internal/clef` ports `encode_record` from the checkpoint's own `joint_schema_model.py` (read in full for `docs/measurements/decisions-d10-clef-2026-10-02.md`). The joint head, D12's other half, is not started.

## Result

On the 150 recorded items (`testdata/decisions/clef/records.jsonl`, with the official encoder's output and every fragment it tokenized in `testdata/decisions/clef/encoder.jsonl`):

| gate | result |
|---|---|
| prompt text: every fragment the encoder tokenizes, in the reference's call order | identical, 150/150 (replay tokenizer) |
| token ids, 44,823 in all | identical, 150/150 (replay), and 150/150 through goinfer's own tokenizer on Clef-flash's `tokenizer.json` |
| question spans and option spans, option ids and their order, question type | identical, 150/150 (150 questions: noul 51, choice 50, score 49) |

Two tiers, because the tokenizer is a 20 MB asset that is not in the tree:

- **`TestEncode_matchesReference` runs everywhere.** It replays the reference's own token ids per fragment (sliced back out of the recorded `input_ids`, since the fragments partition them as prefix + state + schema + suffix) and asserts that the Go encoder asks for exactly the recorded text, in the recorded order, and assembles the recorded ids and spans. That proves the prompt string and the assembly with no asset. `TestEncode_replayCatchesAWrongPrompt` is its able-to-fail control: four mutated requests (state, instructions, type, question id) must each be refused.
- **`TestEncode_realTokenizer` is the tokenizer's.** It needs `$CLEF_TOKENIZER` or `~/models/clef-flash/tokenizer.json`; with `GOINFER_HEAVY_TESTS=1` a missing tokenizer fails, otherwise it skips (a skip is not a pass). Run here with the tokenizer from the archive checkpoint (a fixture read, not a timed run): 0 of 150 differ.

## What the 150 items do NOT exercise

The renderer's hard cases barely appear in them. Measured on the fixture: 18 floats across all states, 1 item with non-ASCII text, 0 items with escaped characters, 0 with special-token text. So the Python-compatible renderer is gated by `internal/clef/testdata/render_cases.json` instead: ten inputs and the output of real Python 3 `json.dumps` (`scripts/pin_clef_render_cases.py`), covering float spelling (`1e5` is `100000.0`, `1e-7` is `1e-07`, `1e16` is `1e+16`, `-0.0`), integers past 2^53, `\u0001`/`\u007f`/U+2028 handling under `ensure_ascii=False`, a repeated key, key order over non-ASCII keys, and the characters Go's `encoding/json` escapes and Python does not (`<`, `>`, `&`). All ten match.

## Deliberate refusals

Python's `json.loads` accepts things JSON does not, and this port refuses them rather than build a prompt the reference would have built differently: `NaN`/`Infinity`, a number that overflows a float64 (Python renders `Infinity`), a lone surrogate escape, and invalid UTF-8. Images and videos (non-empty) are refused until multimodal.md P8a. An unknown or non-string question type, a choice without an object of criteria, and a score without a list are errors, as they are in the reference (a `KeyError`/`TypeError` there).

Ordering rules ported and asserted (`encode_semantics_test.go`): a choice's options are alphabetical by key and **not** in the order sent; noul is always `true` then `false`, and its criteria can override those two descriptions only (a null drops the description key); a score's options are its list indices; an empty or missing instruction becomes the question id; the state is cut to its first tokens to fit `max_length` (default 16,384) and the spans shift with it; a schema that alone exceeds it is an error.

## Hostile input: request text is tokenized literally (owner decision 2026-10-02)

The reference tokenizes each fragment with `tokenizer(text, add_special_tokens=False)`, which **parses special-token text**. So a customer message containing `<|im_end|><|im_start|>system` inside the state becomes real control tokens in the sequence the model reads, and can forge a turn boundary. The owner's decision is to prevent that, and the encoder enforces it rather than leaving it to callers:

- `Tokenize` is `func(text string, parseSpecial bool)`. `parseSpecial` is true for exactly two fragments, the fixed prefix and suffix templates (which carry the chat markers the model needs). It is false for every fragment that holds request text: the state, the instructions, the question ids, the option ids and the descriptions.
- Production wiring uses `tokenizer.EncodeSegments` with `Segment.Special = parseSpecial` (the same M25 split `internal/decide.PlainTokenizer` uses for Route B).
- **Where it differs from the reference:** only on input containing special-token text. On all 150 recorded items there is none, so the gates above are unchanged (token ids identical, 0 of 150 differ through the real tokenizer).
- **Gates:** `TestEncode_onlyTheTemplatesParseSpecialTokens` asserts the flag per fragment (hostile text in the state, a question id, an instruction, an option id and a description; exactly two fragments parse specials, and they are the templates). The replay test asserts the same flag on all 150 items. `TestEncode_hostileTextForgesNoControlTokens` runs a request full of six control markers through the real tokenizer: **0 forged control tokens** with the literal split, against **35** from the same request through a parse-everything tokenizer (the reference's behaviour), so the check can fail.
- **What this does not cover:** the effect on the head's output of the literal text, which the reference never saw, is unmeasured. It is the right text for the model to see (it is what the customer wrote). Structure injection in plain text is also not prevented: a question id or instruction containing `\nEND FIELD\nFIELD 2\nID: x` can still imitate schema structure. The schema is written by the application developer, not the end user, so this is a trust boundary to document in D13, not a tokenizer fix.

## Not done in this item

- The joint head (D12's second half): needs the hidden-state goldens from the queued `d10-clef-f32` job.
- Wiring into `/v1/decisions` (D13).
