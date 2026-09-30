---
title: "Decisions without generating"
area: "Structured output"
order: 5
summary: "goinfer answers a choice among a few options from one read of the prompt, with no generation. The mechanism ships; its accuracy on a real checkpoint is not yet measured."
stand: "Ask a yes/no, a pick among options, or a 0-to-5 score, and goinfer answers from one read of the prompt, with a probability for each option and nothing generated. Whether the answers are good enough is still being measured."
measured: 2026-09-28
reviewed:
facts:
  - {label: "generation", value: "none: one prefill per question, no decode"}
  - {label: "endpoint", value: "POST /v1/systemone"}
  - {label: "kinds", value: "yes/no, choice, score"}
  - {label: "graded accuracy", value: "not yet measured (D6a pending)"}
doesnt:
  - title: "It hasn't been graded on a real checkpoint."
    text: "The pre-registered run (D6a, Qwen3.5-9B on the CUDA box) is still pending, so there is no goinfer accuracy or calibration figure for label scoring on any checkpoint. The only published number is the JEV authors' own run of this method, in bf16, and it is far behind their trained head."
  - title: "It isn't a trained decision model."
    text: "The probabilities are the served model's own over the options it was shown, not the output of a trained head such as JEV-9B's. Building one (D2 to D4) waits on the D6a result. It isn't TypeSafe's hosted model either."
  - title: "Its probabilities are not calibrated by default."
    text: "Every answer says \"calibrated\": false unless you supply a calibration.json, fitted on your own labelled examples under the same template. Even then it is calibrated only in the sense that fit measured."
  - title: "It can't answer from a compute-time adapter entry."
    text: "Label scoring would read the base model under the adapter's name, so the server refuses with a 400 and says to ask the base model. /v1/models leaves the decisions field off such an entry, and off entries that are not generative models."
  - title: "Many questions about one state cost one prefill each."
    text: "On the Gated-DeltaNet families, Qwen3.5 among them, no prefix is reused between questions, so a request costs more with each question. Whether one schema-constrained answer is faster is a projection so far (D7), not a measurement. Choice answers can also lean on option order: --permute averages over several orders at the cost of one prefill each."
figures:
  - {text: "0.5323", source: "docs/measurements/decisions-d0-prior-art-2026-09-27.md"}
  - {text: "0.898", source: "docs/measurements/decisions-d0-prior-art-2026-09-27.md"}
  - {text: "0.5180", source: "docs/measurements/decisions-d0-prior-art-2026-09-27.md"}
  - {text: "0.9181", source: "docs/measurements/decisions-d0-prior-art-2026-09-27.md"}
  - {text: "0.0720", source: "docs/measurements/decisions-d0-prior-art-2026-09-27.md"}
  - {text: "0.0396", source: "docs/measurements/decisions-d0-prior-art-2026-09-27.md"}
  - {text: "0.8881", source: "docs/tasks/task-constrained-confidence.md"}
  - {text: "0.8181", source: "docs/tasks/task-constrained-confidence.md"}
  - {text: "0.05", source: "docs/tasks/task-constrained-confidence.md"}
  - {text: "3/6", source: "docs/tasks/task-constrained-confidence.md"}
  - {text: "1/6", source: "docs/tasks/task-constrained-confidence.md"}
  - {text: "872", source: "docs/tasks/task-constrained-confidence.md"}
  - {text: "0.847", source: "docs/measurements/confidence-c0-2026-09-27.md"}
sources:
  - "docs/server.md"
  - "docs/tasks/task-constrained-confidence.md"
  - "docs/measurements/decisions-d0-prior-art-2026-09-27.md"
  - "docs/measurements/decisions-d7-2026-09-28.md"
  - "docs/measurements/confidence-c0-2026-09-27.md"
  - "docs/integrations/typesafe-jevx.md"
  - "internal/serveapp/systemone.go"
  - "internal/decide/decide.go"
  - "internal/decidecmd/decidecmd.go"
---

## The problem

Programs often need to pick one option from a few: which team gets this ticket, is this message urgent, how angry is this customer from 0 to 5. The usual way is to ask a model to write the answer, then parse what it wrote. That spends one decode step per output token, and the reply carries only the answer, not how sure the model was.

goinfer can already force a reply into a JSON schema. This is a lighter tool for when the answer is one label.

## What goinfer does

`POST /v1/systemone` takes a state and named questions, each with a closed set of answers, and returns a probability for every option. The request and response follow the shape TypeSafe's decisions API uses, so clients written for it, such as jevx and TypeSafe's SDKs, can be pointed at goinfer. The values below are illustrative, in the shape `docs/server.md` shows, not a recorded run.

```json
{"model": "local", "state": "Help! My payouts have been failing for 3 days.",
 "questions": {"department": {"type": "choice", "instructions": "Which team should handle this?",
                              "criteria": {"billing": "Payments, invoicing, refunds", "technical": "Bugs, outages", "sales": null}},
               "is_urgent":  {"type": "noul", "instructions": "Is this urgent?"}}}
```

```json
{"model": "local",
 "answers": {"department": {"type": "choice", "choice": "billing", "confidence": 0.775,
                            "probabilities": {"billing": 0.85, "technical": 0.12, "sales": 0.03}},
             "is_urgent":  {"type": "noul", "noul": 0.91}},
 "usage": {"input_tokens": 212, "output_tokens": 0},
 "goinfer": {"route": "label", "template": "chat-v1", "calibrated": {"choice": false, "noul": false}}}
```

A `noul` question is yes/no and returns P(true). A `choice` takes 2 to 16 options. A `score` takes 2 to 10 levels and returns the expected level. `confidence` is the top probability's margin over an even split, `(n·p − 1)/(n − 1)`. A question that breaks a rule, such as an empty `instructions`, gets a 422 before any prefill. `/v1/models` lists each entry's `decisions` support, including its template and whether a calibration is loaded.

## How it works

The server builds a short prompt from the state, the question and the options, with each choice labelled A, B, C and so on. With the default `chat-v1` template that prompt goes in the model's chat template, ending with an instruction to answer with one letter. The model reads it once. goinfer takes the scores at the last position, keeps only those for the label tokens (`true` and `false`, the digits, or the letters), and turns them into probabilities that sum to 1. Nothing is decoded, so `output_tokens` is 0.

Each label must be a single token in the model's vocabulary, or the server refuses that model. A calibration file adds one temperature per kind, applied before the softmax.

## What was measured

Nothing here grades goinfer's label scoring on a real checkpoint yet. That is the D6a run, and it is pending. This is what the records hold.

| Evidence | Checkpoint and machine | Date | Result |
|---|---|---|---|
| JEV authors' own run of this method (their "B0", bare template, bf16), not goinfer | Qwen3.5-9B | 2026-09-27 (read) | Choice top-1 0.5323 against 0.898 for their trained head; overall OOD top-1 0.5180 against 0.9181; OOD ECE 0.0720 against 0.0396 |
| D6a pre-registered bar, on arm A (chat-v1) after calibration | Qwen3.5-9B Q4_K_M, nobara CUDA | 2026-09-28 | Ship without a trained head if top-1 ≥ 0.8881 and ECE ≤ the bias-adjusted bar; build one if top-1 < 0.8181; between is the owner's call. Sample 872 rows. Not yet run, so neither met nor missed |
| D6a smoke run, disclosed before the bar was written | Qwen3.5-9B | 2026-09-28 | chat-v1 got 3/6 right and bare-v1 1/6; the record says it decides nothing |
| C0, a different feature: per-field confidence on constrained JSON, ranking only | qwen2.5-coder 1.5B, MacBook, Metal | 2026-09-27 | Enum AUROC 0.847 against a bar of 0.65. It uses the same probability readout, but it does not grade label scoring |

The bias-adjusted ECE bar, adopted on 2026-09-28 before any graded result, is the square root of 0.05 squared plus a sampling floor, because ECE reads high on small samples. The floor is simulated from arm A's own confidences. The bare 0.05 is reported next to it.

The record reads the authors' run as landing this method, with the bare template, in the "build a trained head" branch. D6a grades a different template (chat-v1), so it can still come out otherwise. The speed comparison against constrained generation (D7) is only a projection, so no timing is quoted.

## Use it

```sh
goinfer-serve -model local=<path>/model.gguf -ctx 4096
goinfer-serve ... -decisions-template chat-v1 -decisions-calibration calibration.json
```

- `-decisions-template` is `chat-v1` (the server default, for instruct models) or `bare-v1` (JEV's own, no chat template). `-decisions-calibration` must be fitted under the same template, or the server refuses to start.
- Send goinfer's served name as `model`; an unknown name is refused.
- To fit a calibration on your own labelled lines, or score a file without a server:

```sh
goinfer-chat decisions-calibrate --model <file.gguf> --template chat-v1 -o calibration.json labelled.jsonl
goinfer-chat decide --model <file.gguf> --template chat-v1 --calibration calibration.json in.jsonl
```

- `decide` reads one JSON line per question (`id`, `kind`, `state`, `question`, `options`) and writes one line with the distribution and the decision. Its default template is `bare-v1`, unlike the server's, so pass `--template`. `--permute n` averages a choice over n option orders.
- On Metal, pass `--embed-int4=false`. The 2026-09-28 default sends a Metal load to the CPU (`docs/quantization.md`, "Known issue"). The pre-registered D6a run is on CUDA because the 9B fell back to the CPU on the Mac.
