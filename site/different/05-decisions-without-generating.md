---
title: "Decisions without generating"
area: "Structured output"
order: 5
summary: "goinfer answers a choice among a few options from one read of the prompt, with no generation. It ships; its accuracy on a real model is not yet measured."
stand: "Ask a yes/no, a pick among options, or a 0-to-5 score, and goinfer answers from one read of the prompt, with a probability for each option and nothing generated. Whether the answers are good enough is still being measured."
measured: 2026-09-28
reviewed: 2026-09-29
facts:
  - {label: "generation", value: "none: one read of the prompt per question, nothing written"}
  - {label: "endpoint", value: "POST /v1/systemone"}
  - {label: "kinds", value: "yes/no, choice, score"}
  - {label: "graded accuracy", value: "not yet measured (the graded run is pending)"}
doesnt:
  - title: "It hasn't been graded on a real model."
    text: "goinfer's own graded run, with its bar pre-registered, is still pending: Qwen3.5-9B on a Linux PC's NVIDIA GPU. So there is no goinfer accuracy or calibration figure for label scoring on any model. The only published number is the JEV authors' own run of this method, with 16-bit (bf16) weights, and it is far behind their trained head."
  - title: "It isn't a trained decision model."
    text: "The probabilities are the served model's own over the options it was shown, not the output of a trained head such as JEV-9B's. Building one for goinfer is planned, and waits on the graded run's result. It isn't TypeSafe's hosted model either."
  - title: "Its probabilities are not calibrated by default."
    text: "Every answer says \"calibrated\": false unless you supply a calibration.json, fitted on your own labelled examples under the same template. Even then, it is only as calibrated as that fit on your examples measured."
  - title: "It can't answer through an adapter."
    text: "A model served with -adapter (a LoRA adapter: a small set of fine-tuned weights applied over a base model at run time) cannot answer. Label scoring would read the base model under the adapter's name, so the server refuses with a 400 and says to ask the base model. /v1/models leaves the decisions field off such an entry, and off entries that are not generative models."
  - title: "Many questions about one state cost one prefill each."
    text: "On the model families built on Gated DeltaNet layers, Qwen3.5 among them, goinfer reuses no shared start of the prompt between questions, so a request costs more with each question it asks. Whether one schema-constrained answer to every question is faster is only a projection so far, not a measurement. Choice answers can also lean on option order: the decide command's --permute averages over several orders, at the cost of one prefill each."
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

Programs often need to pick one option from a few: which team gets this ticket, is this message urgent, how angry is this customer from 0 to 5. The usual way is to ask a model to write the answer, then parse what it wrote. A model writes one token at a time (a token is a word or a piece of one), and each token costs one decode step, a full pass through the model. The reply also carries only the answer, not how sure the model was.

goinfer can already [force a reply into a JSON schema](/different/04-a-go-struct-the-model-cant-break/). This is a lighter tool for when the answer is one label.

## What goinfer does

`POST /v1/systemone` takes a state (the text to judge) and named questions, each with a closed set of answers. It returns a probability for every option. The request and response follow the shape of TypeSafe's decisions API; TypeSafe is a hosted decisions service. So clients written for it, such as jevx (a Go command-line tool) and TypeSafe's SDKs, can be pointed at goinfer. None of them has been run against goinfer end to end yet, as [the integration guide](https://github.com/townsendmerino/goinfer/blob/main/docs/integrations/typesafe-jevx.md) says. The values below are illustrative, in the shape [the server docs](https://github.com/townsendmerino/goinfer/blob/main/docs/server.md) show, not a recorded run.

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

A `noul` question (TypeSafe's name for yes/no) returns P(true), the probability of yes. A `choice` takes 2 to 16 options. A `score` takes 2 to 10 levels and returns the expected level. `confidence` is the top probability's margin over an even split, `(n·p − 1)/(n − 1)` for n options and a top probability p: 0 when the model cannot tell the options apart, 1 when it is certain. A question that breaks a rule, such as an empty `instructions`, gets a 422 before the model reads anything. `/v1/models` lists each entry's `decisions` support, including its template and whether a calibration (below) is loaded.

## How it works

The server builds a short prompt from the state, the question and the options, with each choice labelled A, B, C and so on. With the default `chat-v1` template, that prompt goes inside the model's chat template and ends with an instruction to answer with one letter. The model reads the prompt once. That single pass is called a prefill; it is the step before a model would start writing. At the last position the model has a score for every token in its vocabulary. goinfer keeps only the scores for the label tokens (`true` and `false`, the digits, or the letters), and turns them into probabilities that sum to 1. Nothing is decoded, so `output_tokens` is 0.

Each label must be a single token in the model's vocabulary, or the server refuses that model. The probabilities are not calibrated by default; calibrated would mean that a stated 0.8 is right about 80% of the time. A calibration file adds one temperature per kind: a single number, fitted on labelled examples, that sharpens or flattens the scores before they become probabilities.

## What was measured

No record in the repo grades goinfer's label scoring on a real model yet. goinfer's own graded run has its bar written down, but it has not run. This is what the records hold.

Three terms first. JEV-9B is an open decision model: Qwen3.5-9B with fine-tuned weights and a small trained output layer, the "trained head", that gives the answer's probabilities. Top-1 is how often the most probable option is the right one. ECE (expected calibration error) is the average gap between how sure the answers say they are and how often they are right; lower is better.

| Evidence | Model and machine | Date | Result |
|---|---|---|---|
| The JEV authors' own run of this method, not goinfer: their "B0" baseline, with their bare template and 16-bit (bf16) weights | Qwen3.5-9B | 2026-09-27 (date read) | Choice top-1 0.5323 against 0.898 for their trained head. On the out-of-distribution (OOD) test split, overall top-1 0.5180 against 0.9181, and ECE 0.0720 against 0.0396 |
| goinfer's graded run: its bar for the chat-v1 template after calibration, set in writing before the run (pre-registered) | Qwen3.5-9B Q4_K_M (4-bit weights), on a Linux PC (Ryzen 7 3700X, RTX 2070 SUPER 8 GB) through CUDA | 2026-09-28 | Ship without a trained head if top-1 ≥ 0.8881 and ECE ≤ the bias-adjusted bar; build one if top-1 < 0.8181; between is the owner's call. The sample is 872 questions. Not yet run, so neither met nor missed |
| A 6-question trial of the same setup, on questions outside the graded sample, seen before the bar was written and disclosed with it | Qwen3.5-9B | 2026-09-28 | chat-v1 got 3/6 right and bare-v1 1/6; the record says it decides nothing |
| A different feature, per-field confidence on constrained JSON ([How sure was it?](/different/02-how-sure-was-it/)), graded for ranking only | qwen2.5-coder-1.5b-instruct, on a MacBook Pro (M1 Pro, 16 GB) through Metal | 2026-09-27 | Enum AUROC 0.847 against a bar of 0.65. AUROC is how often a right answer carries a higher confidence than a wrong one; 0.5 is chance. It uses the same probability readout, but it does not grade label scoring |

The bias-adjusted ECE bar was adopted on 2026-09-28, before any graded result. ECE reads high on small samples, even for a well-calibrated model, so the bar allows for that: it is the square root of 0.05 squared plus a sampling floor squared. The floor is simulated from the chat-v1 run's own confidences. The bare 0.05 is reported next to it.

The record reads the authors' run as landing this method, with the bare template, in the "build a trained head" branch. goinfer's graded run uses a different template, chat-v1, so it can still come out otherwise. A speed comparison against constrained generation exists only as a projection, so no timing is quoted.

## Use it

```sh
goinfer-serve -model local=<path>/model.gguf -ctx 4096
goinfer-serve ... -decisions-template chat-v1 -decisions-calibration calibration.json
```

- `-model local=<path>` serves the model under the name `local`. With the request above saved as `request.json`, ask it with `curl -s localhost:8080/v1/systemone -d @request.json`.
- `-decisions-template` is `chat-v1` (the server default, for instruct models) or `bare-v1` (JEV's own, no chat template). `-decisions-calibration` must be fitted under the same template, or the server refuses to start.
- Send goinfer's served name as `model`; an unknown name is refused.
- To fit a calibration on your own labelled lines, or score a file without a server:

```sh
goinfer-chat decisions-calibrate --model <file.gguf> --template chat-v1 -o calibration.json labelled.jsonl
goinfer-chat decide --model <file.gguf> --template chat-v1 --calibration calibration.json in.jsonl
```

- `decide` reads one JSON line per question (`id`, `kind`, `state`, `question`, `options`) and writes one line with the distribution and the decision. Its default template is `bare-v1`, unlike the server's, so pass `--template`. `--permute n` averages a choice over n option orders, at one prefill each, to cancel a lean towards one position.
- On Metal, an explicit `--embed-int4` sends a load to the CPU, which is why it defaults off there (docs/quantization.md). The graded run is planned for the Linux PC because on the 16 GB MacBook the 9B did not fit Metal's memory budget and fell back to the CPU.
