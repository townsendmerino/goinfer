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
  - {label: "two numbers per family", value: "next token the same, and worst-position logit cosine"}
  - {label: "four labels", value: "released model, small test model, shares another family's check, not recorded"}
  - {label: "family counts", value: "counted from the capability matrix"}
doesnt:
  - title: "It doesn't cover every quantization on every machine."
    text: "A run names what it tested. On 2026-08-31 the same test selector at the same commit passed 29 tests on the MacBook and 49 on nobara-pc, because the MacBook lacked the checkpoints for the rest and skipped them. Qwen3-Next's row was measured with int4 weights, Gemma 4's at int8int8, so neither says anything about the other quantizations. The CPU reference is bit-identical within one CPU architecture, not across arm64 and amd64."
  - title: "It checks one checkpoint per family, not every size."
    text: "The Llama row was run on TinyLlama-1.1B-Chat. A family that passes there can still have a defect that appears only at another size or configuration. That is what the small-test-model label admits for the families that have nothing better."
  - title: "Some families are checked against a small test model only."
    text: "That label means no released model was small enough to run on the hardware available. A fixture built from the same wiring cannot catch a defect in that wiring, and two families promoted past it turned out to have real defects behind a passing fixture."
  - title: "It isn't always an exact match."
    text: "Qwen3.5-MoE's row reads 77.5% of next tokens the same, cosine 0.99069, and the Models page prints that instead of hiding it. The sweep record calls it a deliberate bandwidth trade, traced to one commit. It shows the check is honest about a gap, not that the gap is nothing."
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
  - "site/internal/site/templates/models.html"
---

## The problem

A model family is many small choices: how positions are encoded, where the norms sit, how experts are picked. Get one wrong and the engine still loads, still decodes and still writes fluent text. Nothing errors.

On 2026-08-31, bringing up LFM2 on the MacBook, goinfer ran the real 5 GB checkpoint and produced fluent-looking output whose next-token pick matched HuggingFace's. The logit cosine was 0.897. Logits are the raw scores a model gives every possible next token, and cosine compares two lists of them: 1.00000 means identical.

## What goinfer does

For each family, goinfer's forward pass is compared with the same checkpoint running in HuggingFace `transformers`. The reference is Python, and goinfer is the thing under test. The Models page prints two numbers beside each family: how often the next token is the same, and the worst-position cosine. It also prints one of four labels, strongest first:

| Label on the Models page | What it means |
|---|---|
| Against the released model | compared with `transformers` on a real checkpoint |
| Against a small test model | no released model was small enough to run, so a small test model built from the same wiring stands in |
| Shares another family's check | it runs another family's code, so that family's numbers stand in |
| Not recorded | no check is on file |

Counted from the capability matrix, 37 families are registered: 30 against the released model (19 `full-oracle`, 11 `real-oracle`), 6 against a small test model only, 1 sharing another family's check (Kimi K2, riding DeepSeek-V3), and none unrecorded.

Qwen3-Next reads `real-oracle 100.0%/0.98931`. Qwen3.5-MoE reads `full-oracle 77.5%/0.99069`, and its page says it picks the reference's token at 77.5% of positions, not all of them. Mixtral reads `experimental: tiny-oracle 100.0%/1.00000`, which is the small-test-model label, and the page says so.

## How it works

Each family has a gate that runs the check and records a row in `testdata/parity_manifest.json`: the commit, date, machine, reference, method and both numbers. A model-free test hashes each family's source files and fails CI with "parity stale" if the code changed after the row was recorded. A shared-path family is tied to its source family by the same hash, so the two go stale together.

The bars are set from measured failures, not by taste. On tiny fixtures, two real expert-dispatch bugs passed an argmax check at cosine 0.9887 and 0.9977, because near-interchangeable experts leave the top token alone. A gate is also not trusted until it has been seen to fail on a broken run.

When numbers disagree, goinfer differences layer by layer against the reference's per-layer outputs (`output_hidden_states`) instead of reasoning from the final logits. That named the first bad layer in one run for each LFM2 bug: a zeroed norm epsilon showed at layer 0, and a zeroed attention scale showed at layer 2, the first attention layer. The second one was invisible at one token, because softmax over a single element is 1.0 at any scale. It took five tokens to see.

The reference can be the wrong one. Promoting InternLM2 against released weights, the gate failed on 2026-09-07 at cosine 0.873148. The cause was in `transformers`: the model's remote code registers its rotary frequency table as a non-persistent buffer, and the installed version's fast-init path never filled it, leaving uninitialised memory. Bisecting the loaded module found it. With the table patched, the reference matched goinfer's own output to seven significant figures, and the gate passed at cosine 1.000000.

## What was measured

These rows are the manifest's, as it prints them today. The machine column is the manifest's own label.

| Family | Reference (per the manifest) | Machine | Date | Next token same / worst cosine |
|---|---|---|---|---|
| Llama | HF f32, TinyLlama-1.1B-Chat | linux-amd64 | 2026-09-18 | 100.0% / 1.00000 |
| InternLM2 | HF f32, internlm2_5-1_8b-chat | linux-amd64 | 2026-09-18 | 100.0% / 1.00000 |
| Gemma 4 | HF bf16, E2B and 12B; q4_0 GGUF loaded at int8int8 | linux-amd64 | 2026-09-28 | 100.0% / 0.98972 (12B); E2B 0.99924 |
| Qwen3-Next | HF bf16, 80B-A3B; int4 weights, f32 activations | linux-amd64 | 2026-09-29 | 100.0% / 0.98931 |
| Qwen3.5-MoE | HF bf16, Qwen3.6-35B-A3B text decoder | linux-62gb | 2026-09-06 | 77.5% / 0.99069 |

Gemma 4 and Qwen3-Next were run with quantized weights, as the reference column says. Qwen3.5-MoE's gap is the trade its sweep record names. A gate that fails is reported red, not forced green. The family counts above are counted from the matrix, not measured.

The full release sweep, `go run ./cmd/gate parity`, took 2h51m on nobara-pc on 2026-09-06.

## Use it

- Open the Models page and use the "Checked" filter: any way, against released weights, against a test model, shares a check. Each family page shows the two numbers and the raw string, such as `real-oracle 100.0%/0.98931`.
- In a checkout, `docs/capability-matrix.md` lists every family's row, and `testdata/parity_manifest.json` holds the record behind it.
- `go test ./decoder -run TestParityManifest_fresh -v` runs the staleness check. It needs no model files.
- `go run ./cmd/gate parity` is the release sweep over real checkpoints, and `REALCKPT=0` skips the real-checkpoint gates.
- `go run ./cmd/gate census` counts PASS, SKIP and FAIL. Read the skips: a package of all skips still prints `ok`.
