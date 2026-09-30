---
title: "Checked against the reference"
area: "Correctness"
order: 6
summary: "Each model family is compared with HuggingFace's own implementation, the result is printed on the Models page, and the label says how strong the check was."
stand: "goinfer compares every model family's output with HuggingFace transformers on the same weights. The Models page prints the result next to the family, and says whether the check used the released model or only a small test model."
measured: 2026-09-29
reviewed: 2026-09-29
facts:
  - {label: "compared with", value: "HuggingFace transformers, same weights"}
  - {label: "two numbers per family", value: "how often the next token is the same, and the worst-position cosine of the scores"}
  - {label: "four labels", value: "against the released model, against a small test model, shares another family's check, not recorded"}
  - {label: "families", value: "37, counted from the repo's capability matrix"}
doesnt:
  - title: "It doesn't cover every quantization on every machine."
    text: "A run covers only what it ran. On 2026-08-31 the same set of tests at the same commit passed 29 on the 16 GB MacBook and 49 on the Linux PC, because the MacBook lacked the model files for the rest and skipped them. Qwen3-Next's row was measured with 4-bit weights (int4) and Gemma 4's with 8-bit weights and activations (int8int8), so neither says anything about the other quantizations. goinfer's CPU path is bit-identical within one CPU architecture, not across arm64 and amd64."
  - title: "It checks one checkpoint per family, not every size."
    text: "The Llama row was run on TinyLlama-1.1B-Chat. A family that passes there can still have a defect that appears only at another size or configuration. For the families that have nothing better, the \"Against a small test model\" label admits exactly this."
  - title: "Some families are checked against a small test model only."
    text: "That label means no released model was small enough to run on the hardware available. A small test model built from the same wiring cannot catch a defect in that wiring. Two families that later moved up to a check against released weights turned out to have real defects that their passing small test models had missed."
  - title: "It isn't always an exact match."
    text: "Qwen3.5-MoE's row reads 77.5% of next tokens the same, cosine 0.99069, and the Models page prints that instead of hiding it. The record of the 2026-09-06 release run calls it a deliberate trade for memory bandwidth, traced to one commit that stored more of the model's weights in fewer bits. It shows the check is honest about a gap, not that the gap is nothing."
  - title: "It isn't a quality benchmark."
    text: "It shows goinfer computes what the reference computes on the same weights. It says nothing about whether the model is good, and it can miss a wrong chat template, a bad stop token or a tokenizer edge case, which keep the numbers right and the behaviour wrong."
figures:
  - {text: "0.897", source: "docs/completed/queue-correctness.md"}
  - {text: "0.873148", source: "docs/parity-coverage-policy.md"}
  - {text: "0.9887", source: "docs/parity-coverage-policy.md"}
  - {text: "0.9977", source: "docs/parity-coverage-policy.md"}
  - {text: "0.98931", source: "docs/capability-matrix.md"}
  - {text: "0.99069", source: "docs/capability-matrix.md"}
  - {text: "0.98972", source: "docs/capability-matrix.md"}
  - {text: "77.5%", source: "docs/capability-matrix.md"}
  - {text: "1.00000", source: "docs/capability-matrix.md"}
  - {text: "0.99924", source: "testdata/parity_manifest.json"}
  - {text: "1.000000", source: "docs/parity-coverage-policy.md"}
  - {text: "2h51m", source: "docs/measurements/parity-sweep-aikit-rearm-2026-09-06.md"}
sources:
  - "docs/what-parity-gated-means.md"
  - "docs/parity-coverage-policy.md"
  - "docs/capability-matrix.md"
  - "docs/capability-matrix.json"
  - "testdata/parity_manifest.json"
  - "docs/completed/queue-correctness.md"
  - "docs/measurements/parity-sweep-aikit-rearm-2026-09-06.md"
  - "docs/queue-engineering.md"
  - "site/internal/site/templates/models.html"
---

## The problem

A model family is many small choices: how positions are encoded, where the normalisation steps sit, how experts are picked. Get one wrong and the engine still loads, still generates and still writes fluent text. Nothing errors.

On 2026-08-31, while adding the LFM2 family on the 16 GB MacBook, goinfer ran the real 5 GB checkpoint (the released model's weights file). It produced fluent-looking output whose next-token pick matched HuggingFace's. The logit cosine was 0.897. Logits are the raw scores a model gives every possible next token, and cosine compares two lists of them: 1.00000 means identical. A quick run, a look at the reply and a next-token check would all have passed it.

## What goinfer does

For each family, goinfer's forward pass (one run of the model over its input) is compared with the same checkpoint running in HuggingFace `transformers`, the standard Python library for these models. The reference is Python, and goinfer is the thing under test. The [Models page](/models/) prints two numbers beside each family: how often the next token is the same, and the worst-position cosine. It also prints one of four labels, strongest first:

| Label on the Models page | What it means |
|---|---|
| Against the released model | compared with `transformers` on a real checkpoint |
| Against a small test model | no released model was small enough to run, so a small test model built from the same wiring stands in |
| Shares another family's check | it runs another family's code, so that family's numbers stand in |
| Not recorded | no check is on file |

The repo's [capability matrix](https://github.com/townsendmerino/goinfer/blob/main/docs/capability-matrix.md) is its table of every family. Counted from it, 37 families are registered: 30 against the released model, 6 against a small test model only, 1 sharing another family's check (Kimi K2, which runs DeepSeek-V3's code), and none unrecorded.

Three examples, as the Models page shows them:

- **Qwen3-Next:** against the released model. The next token is the same at 100.0% of positions, and the worst cosine is 0.98931.
- **Qwen3.5-MoE:** against the released model, 77.5% and 0.99069. Its page says it picks the reference's token at 77.5% of positions, not all of them.
- **Mixtral:** against a small test model, 100.0% and 1.00000. It is registered as experimental, and the page labels it that way.

## How it works

Each family has a test that runs the check and records a row in the repo's parity record, [`testdata/parity_manifest.json`](https://github.com/townsendmerino/goinfer/blob/main/testdata/parity_manifest.json). A row holds the commit, date, machine, reference, method and both numbers. A second test, which needs no model files, hashes each family's source files. It fails CI with "parity stale" if the code changed after the row was recorded. A family that shares another family's check is tied to that family by the same hash, so the two go stale together.

The pass bars are set from measured failures, not by taste. Some models are a mixture of experts: each token is sent to a few of many sub-networks, called experts. On small test models, two real bugs in that routing passed a next-token check at cosine 0.9887 and 0.9977. The test models' experts are nearly interchangeable, so a wrong one leaves the top token alone. A check is also not trusted until it has been seen to fail on deliberately broken code.

When the numbers disagree, goinfer compares its output with the reference's after every layer (HuggingFace's `output_hidden_states`), instead of reasoning back from the final scores. That named the first bad layer in one run for each LFM2 bug. A normalisation constant read as zero showed at layer 0. An attention scale left at zero showed at layer 2, the first attention layer. The second bug was invisible at one token. Softmax, which turns scores into weights that sum to 1, gives 1.0 for a single element at any scale. Any prompt of two or more tokens shows it; it was found with five.

The reference can be the wrong one. When InternLM2 was moved up to a check against its released weights, the check failed on 2026-09-07 at cosine 0.873148. The cause was in `transformers`, not goinfer. The model's own Python code computes a table of rotary position frequencies (part of how it encodes token positions) when it loads, and does not save that table with the weights. The installed `transformers` version's fast loading path never filled it in, so it held uninitialised memory. Stepping through the loaded Python model line by line found it. With the table patched, the reference matched goinfer's own output to seven significant figures, and the check passed at cosine 1.000000.

## What was measured

These five rows are copied from the parity record as it stands today. All five were recorded on a Linux PC (Ryzen 7 3700X, RTX 2070 SUPER 8 GB, 62 GB of RAM), which the record labels `linux-amd64` or `linux-62gb`.

| Family | Reference model and number formats (from the record) | Date | Next token same / worst cosine |
|---|---|---|---|
| Llama | HuggingFace in f32, TinyLlama-1.1B-Chat | 2026-09-18 | 100.0% / 1.00000 |
| InternLM2 | HuggingFace in f32, internlm2_5-1_8b-chat | 2026-09-18 | 100.0% / 1.00000 |
| Gemma 4 | HuggingFace in bf16, E2B and 12B; goinfer loaded a q4_0 GGUF file at int8int8 | 2026-09-28 | 100.0% / 0.98972 (12B); E2B 0.99924 |
| Qwen3-Next | HuggingFace in bf16, 80B-A3B; goinfer at int4 weights, f32 activations | 2026-09-29 | 100.0% / 0.98931 |
| Qwen3.5-MoE | HuggingFace in bf16, Qwen3.6-35B-A3B text decoder | 2026-09-06 | 77.5% / 0.99069 |

f32 and bf16 are 32-bit and 16-bit floating point. Quantization stores weights in fewer bits to save memory: int4 means 4-bit weights, int8int8 means 8-bit weights and 8-bit activations (the values passed between layers), and q4_0 is a 4-bit GGUF file format. Gemma 4 and Qwen3-Next were run with quantized weights, as the reference column says. Qwen3.5-MoE's gap is the trade its release-run record names. A check that fails is reported as failing, not forced to pass. The record does not say how many positions each percentage is taken over. The family counts above are counted from the capability matrix, not measured.

The full release run of these checks, `go run ./cmd/gate parity`, took 2h51m on the Linux PC on 2026-09-06.

## Use it

- Open the [Models page](/models/) and use the "Checked" filter: Any way, Against released weights, Against a test model, Shares a check. Each family page shows the two numbers and the record's raw string, such as `real-oracle 100.0%/0.98931`. In that string, `real-oracle` and `full-oracle` mean against the released model, and `tiny-oracle` means against a small test model.
- The [capability matrix](https://github.com/townsendmerino/goinfer/blob/main/docs/capability-matrix.md) lists every family's row, and the [parity record](https://github.com/townsendmerino/goinfer/blob/main/testdata/parity_manifest.json) holds the data behind it.
- `go test ./decoder -run TestParityManifest_fresh -v` runs the "parity stale" check. It needs no model files.
- `go run ./cmd/gate parity` is the full release run over real checkpoints. It takes hours, and a check whose checkpoint is missing is skipped, not failed. `REALCKPT=0 go run ./cmd/gate parity` skips the real-checkpoint checks.
- `go run ./cmd/gate census` counts passed, skipped and failed tests. Read the skips: a Go package whose tests all skipped still prints `ok`.
