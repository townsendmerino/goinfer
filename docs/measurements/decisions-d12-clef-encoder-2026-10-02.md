# D12 — the Clef record encoder and joint head in Go (2026-10-02)

`internal/clef` ports `encode_record` and `JointSchemaHead` from the checkpoint's own `joint_schema_model.py` (read in full for `docs/measurements/decisions-d10-clef-2026-10-02.md`). The encoder is gated of record (below). The head is gated on a SYNTHETIC golden only; its gate of record waits on the D10 hidden-state fixture (see "The head").

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

## The head (`internal/clef/head.go`)

121.8M parameters, widened from the checkpoint's bf16 to f32 on load (about 487 MB). `LoadHead` refuses a config with a missing or unknown key, a tensor the layout does not name, a missing tensor and a tensor of the wrong shape. `Forward(hidden, ids, questions, lm_head rows)` returns one logit per option per question, in the encoder's option order; `Softmax` gives the probabilities.

**Gate (head isolated, f32): per-option probabilities within 1e-5 absolute of the reference.** Result on the synthetic golden (4 items; a first run, nothing tuned): **worst probability difference 1.23e-7, worst logit difference 9.5e-7**, about 80 times inside the bar.

**What the synthetic golden is, and is not.** It is the reference `JointSchemaHead` (imported unmodified, f32) on seeded random hidden states and seeded random lm_head rows (`scripts/pin_clef_head_synthetic.py`): the shortest noul, choice and score fixture items, plus one 4-question record (noul, choice with 3 options, score with 6, a second choice; 904 tokens) encoded by the OFFICIAL `encode_record`. It isolates the head from the backbone, so the port could be checked before the backbone's hidden states existed. It is not the gate of record, for two reasons: random hidden states do not have the statistics of a trained backbone's (scale, outlier dimensions), so they exercise every code path but not the numerical regime the head runs in, and it is not committed (about 8 MB of incompressible floats). **The gate of record is the same test on the D10 f32 fixture's real `last_hidden_state` for the 3 shortest items** (the queued `d10-clef-f32` job), with the real lm_head rows extracted from the backbone shards. Until that has run, the head is "matches the reference on random inputs", not "gated".

**It also covers the multi-question path**, which the 150 fixture records (one question each) cannot: the 4-question record exercises the decoder's self-attention over the questions, the per-question option split, and the encoder's FIELD numbering, several questions' spans shifted together, and an instruction with a quote and an accented letter. The Go encoder reproduces the official encoder's ids and spans on it (`TestEncode_multiQuestionMatchesReference`).

**Able to fail.** Four porting mistakes, each applied alone to a copy of `head.go` and each turning `TestHead_matchesReference` red: leaving the routing layers' memory un-normalised, normalising the decoder cross-attention's memory (the trap: the reference normalises it in the routing layers and not in the decoder layers), dropping the sigmoid of the residual gate, and ignoring the prior logit scale. The file was restored byte for byte after each.

**Placement of the assets.** `TestHead_matchesReference` and `TestEncode_multiQuestionMatchesReference` read `$CLEF_HEAD_GOLDEN` (else `testdata/decisions/clef/head_golden`) and the weights from `$CLEF_HEAD` (else `~/models/clef-flash`); either missing fails under `GOINFER_HEAVY_TESTS=1` and otherwise skips, and **a skip is not a pass**: on a checkout without them these two tests have run nothing.

## The whole pipeline, end to end (tiny fixture; `internal/clef/model.go`, `pipeline_test.go`)

`clef.Model.Decide(request)` runs the encoder, one backbone pass (`decoder.PromptHiddenAll`) and the head, taking the lm_head rows for the option tokens from the loaded model through the new `decoder.OutputEmbeddingRow` (the separate lm_head when the family has one, the input embedding when it is tied; a quantized load returns the dequantized row of the quantized weight, so the arm's quantization is part of what is measured). `NewModel` refuses a backbone and a head whose widths disagree.

**Gate: probabilities within 1e-4 absolute of the reference (bar written before the first run). Measured: worst difference 1.8e-7** over four requests (noul, choice, score, and three questions at once), against the official `joint_schema_model.py` chained as `ClefModel.forward` chains it, at f32, on the committed tiny qwen3_5 (hidden 64, untied lm_head, random final-norm weight) with a tiny head (width 32, 2 routing layers, 2 decoder layers, 4 heads, seeded random weights in every parameter, because the release's init leaves the scales at 0 and the norms at identity, which would let whole paths drop out unseen). `scripts/pin_clef_e2e_tiny.py` writes the fixture (247 KB of head weights, committed with a `.gitignore` exception) and asserts the probabilities are not near-uniform. The encoder's ids and spans are also compared exactly on the same requests. The stand-in tokenizer (4-byte chunks, one id each) gates the prompt TEXT; the real tokenizer is gated by `TestEncode_realTokenizer`. Unlike the head test on the real weights, this one runs on every checkout, CI included, and a missing fixture fails instead of skipping.

**Able to fail.** Three end-to-end mistakes, each applied alone and each turning it red: taking the lm_head rows from the input embedding (the tied-head assumption), an off-by-one in the span offset, and pooling the global vector from the first position instead of the last. Sources restored byte for byte after each.

**It found a real bug the isolated gates could not.** `LoadHead` closed the checkpoint's memory map after loading, and `TensorF32` returns a zero-copy view into that map for an F32 tensor. The real head is bf16 (always widened into a new slice), so the head test never saw it; the tiny head is F32 and the process faulted on first use, after first producing uniform output from the unmapped memory. `LoadHead` now copies F32 tensors so the head owns its weights whatever the checkpoint's dtype.

## Owner decisions recorded here

- **Request text is tokenized literally** (2026-10-02; above).
- **The `clef` route reports `confidence` as the reference does: the top probability** (2026-10-03). This route only: `/v1/systemone` on the other routes keeps its margin over uniform, `(n*p - 1)/(n - 1)`, because TypeSafe's demo uses that form. The response says which one it is. For a `score` question the reference's `confidence` is also the top probability and its `score` is the expected level; for `choice` the argmax is taken over the REQUEST's criteria order, first maximum winning (`systemone_answer`, read in full in the D10 record). D13 wires this.

## Not done in this item

- **The head's gate of record on real hidden states: tooling built 2026-10-03, not yet run.** It needs the `d10-clef-f32` night job (it saves the real backbone `last_hidden_state`, f32, for the three shortest records). Then, from the repo root:

      python3 scripts/build_clef_head_golden.py          # D10's outputs and ~/models/clef-flash in, ~/goinfer-bench/decisions-d12-head-gate out
      CLEF_HEAD_GOLDEN=~/goinfer-bench/decisions-d12-head-gate CLEF_HEAD=~/models/clef-flash GOINFER_HEAVY_TESTS=1 go test -count=1 -v -run TestHead_matchesReference ./internal/clef/

  `scripts/build_clef_head_golden.py` reads the lm_head rows the options use straight from the shards and writes the golden in the test's format. **Its self-check** re-runs the official `JointSchemaHead` on the stored hidden states and the extracted rows and requires D10's recorded logits back (hard failure above 1e-2, warning above 1e-4), which proves that each hidden file belongs to its record and that the extracted rows are the ones the reference used. The bar is the registered one, 1e-5 absolute in probability, given the reference's own hidden states.

  **Tested today on a stand-in** (the real head weights and the real lm_head rows, loaded by a different route than the extractor's, with SYNTHETIC hidden states): the self-check read max |diff| 0, the Go gate passed with a worst probability difference of 1.41e-7, and the extractor refused (exit 1, |diff| 3.76) a copy in which two same-length hidden files had been swapped. That shows the chain works, not that the head is right on a trained backbone's hidden states: only the real run says that. Whether to commit the gzipped goldens (about 9 MB of hidden states) is decided after seeing them; the manifest records the sha256 of every input and output either way. The tiny pipeline fixture does not replace it: it proves the wiring, not the numerical regime of a trained 9B backbone.
- Speed: the head runs the reference's structure with a row-parallel f32 GEMM and a scalar attention; D14 owns making it fast, and nothing here is a performance claim.
- Wiring into `/v1/systemone` and `/v1/decisions` (D13), including the response shaping above.
