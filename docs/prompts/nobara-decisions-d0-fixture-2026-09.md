# Prompt: decisions D0 — the JEV-9B reference fixture (nobara)

For a Claude Code session on `nobara-pc`, repo `~/mycode/goinfer`. Run `git pull` first.

## The job

Produce D0's reference fixture for [`docs/tasks/task-constrained-confidence.md`](../tasks/task-constrained-confidence.md).
It is autotrust/JEV-9B's own distributions on a fixed set of held-out items, committed as goldens under
`testdata/decisions/`, so goinfer's Route A (D1/D6a) and later Route B (D2–D4/D6b) can be graded against the
reference.

**autotrust's server is not published** (D0 record §2), so the reference is the model card's `decide()`. Read
[`docs/measurements/decisions-d0-prior-art-2026-09-27.md`](../measurements/decisions-d0-prior-art-2026-09-27.md)
first. It has the verbatim `judge_config.json`, `calibration.json`, template code, verbalizer ids and pins.

## Inputs (pin every one)

- **Model:** `hf download autotrust/JEV-9B --revision 4ab5dfb9331c4eb3a212742e1a1aa5446c1fda35 --local-dir ~/models/JEV-9B`
  (18.27 GB).
  - It goes on **`~/models` (NVMe), never `/srv/models`**, which is the archive. Put a copy there afterwards with
    `models-push` if you want one.
- **Data:** `SargeDev/jev-distill-corpus-v3` at `fc99c6357a9f89f7512c4a987314352addead049`, files
  `calibration.jsonl` and `ood.jsonl`.
- **Python:** a venv with `transformers==5.16.1`, `peft==0.21.0`, torch (CPU is fine), safetensors and accelerate.
  - Record every exact version, and whether flash-linear-attention is installed.
  - Unverified: whether transformers' Gated-DeltaNet path runs on CPU without it. Check first, on one item.

## The items (deterministic; write the selection into the script)

150 items, as the task doc registered. Within each stratum, take the rows with the lowest `sha256(id)`:

| split | stratum | n |
|---|---|---:|
| calibration | noul, `openjev_v2` (gold) | 17 |
| calibration | noul, `yuri_v3` | 17 |
| calibration | choice, `openjev_v2` (gold) | 17 |
| calibration | choice, `yuri_v3` | 16 |
| calibration | score, `yuri_v3` (the split has no gold score rows) | 33 |
| ood | noul (`openjev_v2`, gold) | 17 |
| ood | choice (`openjev_v2`, gold) | 17 |
| ood | score (`openjev_v2`, gold) | 16 |

- Skip `yuri_v1` rows, which are all `[0.5, 0.5]` placeholders.
- Include choice rows with up to 16 options, so some 16-option rows land in the set.
- **Record each row whole:** `id`, `kind`, `options`, `target`, `state`, `question`, `source`, `domain` and
  `family`. `target` is gold only for `openjev_v2` rows; label that.

## What to compute, per item

1. **The rendered bare-v1 prompt bytes** and **its token ids** under JEV-9B's tokenizer (`add_special_tokens=False`).
   goinfer's tokenizer will be required to reproduce these ids exactly. A mismatch there is a bug to find before any
   distribution is compared.
2. **Route B, the reference:** the card's `decide()`.
   - That is the backbone with the LoRA, the post-final-norm hidden state of the last token, the fp32 head, `z / T`
     with T from `calibration.json`, and a softmax over the kind's active slots.
   - Record the distribution **before and after** the temperature, and the last-token hidden state's first 8
     values plus its L2 norm (a cheap parity probe for goinfer's D2 seam).
   - **Run it twice if memory allows:** once in bf16 (the card's autocast) and once in f32, which D6b's f32 bar
     needs; ~36 GB plus activations. Check `free -g` first. If f32 does not fit, record bf16 only and say so.
   - **Record which LoRA mode you ran,** because the card merges in bf16 (`merge_and_unload`) and the Space runs it
     unmerged. Unmerged is preferred.
3. **Route A with the same template (B0):** the adapter off, and the base lm_head logits of the verbalizer ids at
   the last position, softmax over the active ones, no temperature.
   - This is exactly what goinfer's D1 computes. It lets D1 be checked against transformers on the same bytes.
   - One run is enough; f32 if it fits, else bf16.
4. **The ceiling:** 1,024 input tokens, as the card's serving ceiling. Record any item over it, and how
   `jev_core.py` truncates it (head 60% / tail 40% of the state).

## Outputs (commit these; never the weights)

- `scripts/pin_decisions_d0.py`: the whole thing, runnable end to end, with the pins at the top.
- **The goldens:**
  - `testdata/decisions/items.jsonl`: the 150 rows plus the rendered prompt and token ids;
  - `testdata/decisions/jev9b_ref_bf16.jsonl` (and `…_f32.jsonl` if run): per item, the distribution before and
    after T, the hidden-state probe, the LoRA mode, and the timing;
  - `testdata/decisions/route_a_b0.jsonl`.
  - If any file exceeds 1 MB, write it `.jsonl.gz`.
- `testdata/decisions/README.md`: the pins, library versions, precision, LoRA mode, machine, and date.
- **A line in the task doc's D0 result:** what was produced, and anything that did not match the D0 record (for
  example, if `decide()` does not reproduce the card's published numbers on these rows).

## Sanity checks before committing

- **The verbalizer ids** from the tokenizer must equal `judge_config.json`'s `verbalizer_ids`.
- **Top-1 against gold** on the gold rows should be in the neighbourhood of the card's OOD figures (top-1 0.918,
  choice top-1 0.837). This is a smoke check, not a gate: 50 items is too few to confirm a number.
- **bf16 against f32** (if both ran): report the max |Δp| and argmax agreement beside the authors' own
  vLLM-against-HF floor (argmax agreement 0.9912, max |Δp| 0.258).

## Rules that apply

- **Pushing:** commit increments, and push with every outstanding file, staged by explicit path (never `git add -A`).
  Run `git fetch` and rebase first, and use `gh auth switch --user townsendmerino`.
- **Lints:** run the citation lint and read its exit code directly.
- **Long runs:** run them detached, give the start and expected finish time in local time and UTC, and archive the
  logs out of `/tmp`.
