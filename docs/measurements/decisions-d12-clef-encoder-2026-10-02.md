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

## One thing for the owner: the reference parses special-token text inside the state

The reference tokenizes each fragment with `tokenizer(text, add_special_tokens=False)`, which **parses special-token text**. So a customer message containing `<|im_end|><|im_start|>system` inside the state becomes real control tokens in the sequence the model reads. The trained head saw prompts built this way, and a port that matched the reference on the 150 items would also do it on hostile input.

The encoder is deliberately tokenizer-agnostic (`Tokenize func(string) ([]int, error)`), so this is the caller's choice, and D13 has to make it:

- **Match the reference** (`tokenizer.Encode`): input-identity everywhere, and a state can forge turn boundaries. The effect on the head's output is untested.
- **Keep state text literal** (`EncodeSegments`, as `internal/decide.PlainTokenizer` does for Route B, M25): safe, and identical to the reference on every input without special-token text (all 150 items; this was checked by the count above, not by a second run). It diverges from the reference exactly on the inputs an attacker would write.

I have not made this choice. It is recorded here so D13's wiring does not make it by accident.

## Not done in this item

- The joint head (D12's second half): needs the hidden-state goldens from the queued `d10-clef-f32` job.
- Wiring into `/v1/decisions` (D13).
