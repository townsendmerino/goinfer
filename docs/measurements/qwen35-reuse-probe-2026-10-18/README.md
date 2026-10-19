# Qwen3.5-9B: does prefix reuse work when each request only appends? (2026-10-18)

Asked by the owner after tack's C3 run (docs/measurements/c3-logs-reading-2026-10-18.md §2a): tack's "reused" cell changes the last 512 tokens, which a recurrent model can never resume;
its loop appends the previous reply and a new user turn instead. This plays that loop. Build `794a1359` (main, CUDA), the RTX 2070 SUPER, Qwen3.5-9B Q4_K_M at `-ctx 16384`, under the timing lock,
22:36 to 22:42 PDT, one run of each probe (greedy, 24 new tokens). Exploratory in the sense of the day rules (a mechanism question, not a published number); the sizes are the ones tack measured.

## Result

| probe | size | turn 1 (cold) | turn 2 (appends) | turn 3 (appends) | cold control |
|---|---|---|---|---|---|
| `/v1/chat/completions` (`probe.py`) | 4,098 prompt tokens | 15.36 s | 15.51 s | 15.58 s | 15.63 s |
| `/v1/chat/completions` (`probe.py`) | 8,146 prompt tokens | 38.96 s | 39.15 s | 39.35 s | 39.45 s |
| `/v1/completions`, plain text (`probe_raw.py`) | 4,058 prompt tokens | 15.18 s | **0.04 s** | **0.04 s** | 15.65 s |

(time to the first streamed token; raw data in `raw/`.)

**The engine reuses an appended request; the chat endpoint's request never extends the held sequence.** On plain text, turn 2 and turn 3 start in 0.04 s, so `internal/kvmgr/slots.go:scoreRecurrent` does what it says (the held sequence is the prompt plus the generated
tokens, and a request that starts with all of them continues from the state as it stopped). Through the chat template the same conversation gets no reuse at either size.

## Why: the history render drops the empty think block the generation prompt ends with

Rendered offline with the 9B's own template and tokenizer (`chat.Detect`, thinking `template`, as serve runs it), turn 1 and turn 2 of the same conversation:

- turn 1 prompt: `…<|im_start|>assistant\n<think>\n\n</think>\n\n` (26 ids; the last 4 are the empty think block the generation prompt adds)
- turn 2 prompt: `…<|im_start|>assistant\nRepetitive.<|im_end|>\n<|im_start|>user\n…` (47 ids): the earlier assistant turn is rendered **without** the think block.

The two share 22 ids and diverge at the first think-block token. The held sequence is turn 1's 26 prompt ids plus the generated ones, so the new request is not an extension of it, and a recurrent state can only be continued from exactly where it stopped:
no partial credit, a cold prefill. A model with positional KV (Qwen3-8B, Qwen3-4B) truncates to the 22 shared ids and reuses them, which is why those rows reuse and this one does not. The same divergence is built into every
chat turn of a model whose template adds a generation prompt that the history rendering omits.

## Options (none built)

1. **Checkpoint the recurrent state where the history render and the generation prompt part ways**, i.e. at the end of the prompt minus the generation-prompt tail (the 4 think-block ids here; the template knows its tail). The state of the 24 DeltaNet layers is tens of MB
   (a host copy and a copy back take milliseconds). `scoreRecurrent` then continues from the checkpoint when the request shares that prefix, and prefills the rest (the reply and the new turn). The prefill has to stop at that position, and a restore equals a cold prefill only at a pass boundary of the cold prefill (the KV7 restore rule),
   so the split has to be placed on the grid. Size: M to L (the resident's save and restore of DeltaNet state, the kvmgr rule, the template's tail, the gate).
2. **Render the history turn with the empty think block** when thinking is off (a template change behind a flag): the held sequence is then a prefix of the next prompt and the existing rule applies. Size: S. It changes the prompt the model sees for earlier assistant turns
   from the reference template's, which the owner has to decide; it is a fidelity change, not a speed fix.

Not run: the 35B-A3B (thinking on by default; its generation prompt ends `<think>\n`) and Gemma/Granite hybrids; the divergence is the template's, so each family's history render needs its own look.
