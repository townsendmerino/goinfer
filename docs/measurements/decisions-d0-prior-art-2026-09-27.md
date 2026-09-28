# D0 — prior art for decisions: JEV, TypeSafe's API, jevx, and the peer engines (2026-09-27)

D0 of [`task-constrained-confidence.md`](../tasks/task-constrained-confidence.md): research only, no goinfer code.
Every fact below was read from a primary artifact: HF repo files and API metadata, dataset files, package tarballs
and wheels, GitHub sources, and TypeSafe's own `.md` docs. **NOT FOUND** and **UNVERIFIED** mark what could not be
confirmed.

**Pins:**
- autotrust/JEV-9B @ `4ab5dfb9331c4eb3a212742e1a1aa5446c1fda35`
- autotrust/JEV-27B @ `9b1f6fda6c26b2f7191d437eb34a263911dd1575`
- SargeDev/jev-distill-corpus-v3 @ `fc99c6357a9f89f7512c4a987314352addead049`
- muthuishere/jevx @ `67ed9a6277fae8f4211ba358064b606174f3cd58`
- typesafe-sdk (PyPI) 0.7.2; @typesafe-ai/sdk 0.6.0; @ai-sdk/typesafe-ai 3.0.8; langchain-typesafe 0.0.1a3;
  @langchain/typesafe 0.0.1
- llama.cpp master `4da63377` (2026-09-27); ollama main `16b4376a` (2026-09-26); vllm main `2407f405` (2026-09-28)

## What changes for the task (read this first)

1. **autotrust's `jev_judge` server is not published anywhere.** It is not in either HF repo, not in the squashed
   history, not on PyPI, and not on GitHub. The response fields `{distribution, decision, confidence, latency_ms}`
   that the task doc names were found in no source.
   - The reference fixture must come from the model card's `decide()` (transformers + peft), or from the demo
     Space's `jev_core.py` (`JEV_DEVICE=cpu`, `JEV_MODEL_PATH`).
   - Published serving is vLLM plus a client-side readout.
2. **D6a's question has a published answer for the reference template.** The authors' B0 report runs Route A
   (untrained Qwen3.5-9B, the restricted next-token distribution, the bare-v1 template) against JEV-9B:
   - choice top-1 **0.532 against 0.898** on test_set_30k;
   - overall top-1 **0.518 against 0.918** on the OOD split.
   - Temperature scaling cannot move top-1. With the bare-v1 template, D6a lands in its "more than 10 points
     behind → build D2–D4" branch.
   - Only a different template on goinfer's side could change that, and nothing published shows one does.
3. **TypeSafe's `/v1/systemone` shape is recorded verbatim below, and every SDK accepts a base-URL override**
   (JS, Python, Vercel, LangChain JS and Python), as does jevx. A compatible route would work with all of them.
4. **The recipe numbers in the task doc's D4 are JEV-27B's.** JEV-9B, the Mac target, has temperatures
   noul 1.0022 / choice 0.9840 / score 1.0122 and a 160,486,456 B adapter.
5. **The reference uses no chat template.** The labels are bare `false`/`true`, `0`–`5` and `A`–`P`, with no
   leading space; the leading-space tokens are different ids.

## 1. autotrust's JEV models

**Existence.**
- `autotrust/JEV-9B` exists (created 2026-09-23).
- `autotrust/JEV-27B` exists (created 2026-09-25).
- `autotrust/JEV` is a redirect to JEV-9B (the repo was renamed).
- autotrust has no datasets and two Spaces: `jev-9b-decision-demo` and `JEV-27B-Demo`.

**License and base models.**
- Both repos are Apache-2.0.
- JEV-9B's base is `Qwen/Qwen3.5-9B`. That is the post-trained model, not `-Base`.
- JEV-27B's base is `Qwen/Qwen3.8-27B`.
- Both are `model_type: qwen3_5_text` / `Qwen3_5ForCausalLM`, a type goinfer already registers.

**Files (JEV-9B; 27B has the same layout).**
- The text tower plus `lm_head`: 5 shards, 17.9 GB bf16. The card calls it bit-identical to the base; not
  verified.
- The LoRA adapter (`adapter_model.safetensors` in the repo's `adapter` folder): 160,486,456 B, 400 F32 tensors. 27B: 435,268,528 B.
- An `adapter_vllm` folder: an r = 32 variant that also targets `lm_head`, plus `decision_head.json` with a 24-slot bias.
- `head.safetensors`: `proj.weight` F32 [24, 4096] + `proj.bias` [24]. 27B: [24, 5120].
- `calibration.json`, `judge_config.json`, `chat_template.jinja`, the tokenizer, and `reports/*`.

**`judge_config.json`, JEV-9B**, verbatim (https://huggingface.co/autotrust/JEV-9B/resolve/main/judge_config.json):

```json
{
  "base_model_path": "/root/models/Qwen3.5-9B",
  "hidden_size": 4096,
  "slots": {
    "num_slots": 24,
    "ranges": { "noul": [0, 2], "score": [2, 8], "choice": [8, 24] },
    "verbalizers": ["false","true","0","1","2","3","4","5","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P"],
    "template_version": "bare-v1"
  },
  "verbalizer_ids": [3721,1802,15,16,17,18,19,20,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47],
  "kinds": ["noul", "choice", "score"],
  "model_name": "jev-judge-qwen35-9b",
  "model_version": "0.8.0",
  "lm_head_included": true,
  "weights_mode": "unmerged",
  "adapter_subfolder": "adapter",
  "lora": { "r": 16, "alpha": 32, "dropout": 0.05,
            "target_modules": ["in_proj_qkv","in_proj_z","q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","out_proj"] },
  "trained_stage": "s2",
  "source_checkpoint": "/root/jev-judge/checkpoints/s2_9b_epoch1/best",
  "softcap": null
}
```

- (The array whitespace above is compacted; the values are verbatim.)
- JEV-27B's file differs in four fields only: `base_model_path` (`/root/models/Qwen3.8-27B`), `hidden_size` (5120),
  `model_name` (`jev-judge-qwen38-27b`) and `source_checkpoint`.
- The template version is at `slots.template_version`; there is no top-level version field.
- The verbalizer ids were checked against JEV-9B's `tokenizer.json`. They are the bare tokens, with no leading
  space (` false` = 867, ` true` = 804, ` A` = 357). The authors' M0 report records an equivalence gate: step-0
  head against restricted decoding, max |Δp| 1.49e-06.

**`calibration.json`, JEV-9B**, verbatim (https://huggingface.co/autotrust/JEV-9B/resolve/main/calibration.json):

```json
{
  "version": 1,
  "per_kind": { "noul": 1.0022178236691426, "choice": 0.9840233476014968, "score": 1.012184473350984 },
  "per_kind_family": {},
  "fit": {
    "noul":   { "n": 4363, "T": 1.0022178236691426, "kl_before": 0.004595278762280941, "kl_after": 0.0045948573388159275 },
    "choice": { "n": 3418, "T": 0.9840233476014968, "kl_before": 0.04468083754181862, "kl_after": 0.04462483152747154 },
    "score":  { "n": 3173, "T": 1.012184473350984,  "kl_before": 0.022512147203087807, "kl_after": 0.022472143173217773 }
  },
  "diagnostic_calibration_split": {
    "raw":        { "kl": 0.02229316346347332, "ece": 0.0014972193181917403, "mce": 0.0034518837928771973 },
    "calibrated": { "kl": 0.02226393297314644, "ece": 0.0013502779446837457, "mce": 0.005257606506347656 }
  },
  "source": "checkpoints/s2_9b_epoch1/best",
  "fit_rows": 10954,
  "d1_excluded": true
}
```

- JEV-27B `per_kind`: noul 1.0143134751376188, choice 1.0161280671450397, score 1.0036213515883572.
- The temperature is applied as `z / T` on the head logits.
- `per_kind_family` semantics: UNVERIFIED (empty in both files).

**The template (bare-v1)** is published only as code. From the JEV-9B README ("Other ways to run it"):

```python
def decide(kind, state, question, options):
    letters = "ABCDEFGHIJKLMNOP"
    lines = options if kind != "choice" else [f"{letters[i]}) {o}" for i, o in enumerate(options)]
    text = f"[kind] {kind}\n[state] {state}\n[question] {question}\n[options]\n" + "\n".join(lines) + "\n[decision]:"
    ids = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        h = model.model(**ids).last_hidden_state[0, -1].float()          # backbone only, last token
    z = (W @ h + b) / temp[kind]
    s, _ = cfg["slots"]["ranges"][kind]
    p = torch.softmax(z[s : s + len(options)], 0)
    return dict(zip(options, p.tolist()))
```

- **No chat template and no BOS**, and the whole string is tokenized once.
- **Options:**
  - `noul` must be `["false","true"]`;
  - `score` must be `["0".."5"]`;
  - `choice` takes 2–16 free-text options, as lines `"{L}) {option}"`.
- **Slot use:** a choice with n options uses slots 8 … 8+n−1. The slots after them are excluded before the
  softmax (the code slices `z[s:s+len(options)]`).
- **Head:** its input is the post-final-norm hidden state of the last token. It was initialized from the lm_head
  rows of the verbalizers; at step 0 it equals the zero-shot restricted next-token distribution.
- **Truncation:** inputs over 1,024 tokens are truncated at serving, in the state only, keeping the head 60% and
  the tail 40%.
- A rendered noul example (JEV-9B tokenizer, 44 tokens, last token `]:` = 5491):
  `'[kind] noul\n[state] Customer says the parcel arrived damaged and wants their money back.\n[question] Is the customer asking for a refund?\n[options]\nfalse\ntrue\n[decision]:'`

**Published numbers** (model-index, README, `reports/eval_s2_*.md`):

| split | model | mean KL | top-1 | ECE | noul AUROC | choice top-1 |
|---|---|---:|---:|---:|---:|---:|
| test_set_30k | JEV-9B | 0.0210 | — | 0.0007 | 0.996 | 0.898 |
| test_set_30k | JEV-27B | 0.0186 | — | 0.0009 | 0.996 | 0.903 |
| OOD (13,058) | JEV-9B | 0.2335 | 0.9181 | 0.0396 | 0.9886 | 0.8372 |
| OOD | JEV-27B | 0.1043 | 0.9419 | 0.0067 | 0.9964 | 0.8593 |
| test_set_30k | **B0 = Route A, Qwen3.5-9B, bare-v1** | 0.5100 | 0.4411 | 0.0942 | 0.8236 | 0.5323 |
| OOD | **B0** | 0.8567 | 0.5180 | 0.0720 | 0.6880 | 0.3641 |

- Option-shuffle top-1 flips: 3.9% (9B) and 2.9% (27B). B0's shuffle flip rate is 0.3815.
- The noise floor between the authors' own bf16 vLLM and HF runs of the same model: argmax agreement 0.9912, max
  |Δp| 0.258.

## 2. Server code: NOT FOUND

- **No published server.** Neither repo has server code, there is no `jev_judge` package on PyPI, and there is no
  autotrust repo on GitHub.
- **One unverified trace.** WebSearch snippets of an earlier card mention
  `python -m jev_judge.server --export JEV --port 18080` and a `POST /v1/decisions:batch` (≤ 256 items, 422 on a bad
  kind or options, 413 unless `truncate`). The live cards do not contain this, so it is second-hand search-engine
  text.
- **Published serving: vLLM plus a client-side readout.**
  - `vllm serve JEV-9B --enable-lora --max-lora-rank 32 --lora-modules jev-decision=JEV-9B/adapter_vllm
    --logprobs-mode processed_logprobs`.
  - Then `/v1/completions` with `max_tokens: 1`, `allowed_token_ids` = the verbalizers, and `logprobs`.
  - The client adds the head bias and divides by T.
- **A private contract, for reference only.** The JEV-27B demo Space posts `{kind, state, question, options}` to a
  secret `…/decide` and validates `{model, options, probabilities, choice_index, choice, elapsed_seconds, revision}`.
  That backend is unpublished.

**Datasets:** `SargeDev/jev-distill-corpus-v3` (Apache-2.0; the `openjev_v2` stream is also CC0; not gated).

| split | rows | file |
|---|---:|---|
| calibration | 13,766 | `calibration.jsonl`, 9.0 MB |
| ood | 13,058 | `ood.jsonl`, 14.1 MB |
| test_set_30k | 29,955 | 17.1 MB |
| train / validation / test | 655,806 / 14,111 / 14,261 | |

- **Row schema:** `{id, kind, options, target, state, question, domain, family, source}`.
- **Gold labels** exist only on the `openjev_v2` rows (programmatic, mostly one-hot):
  - calibration: 1,109 noul and 463 choice;
  - ood: 9,767 noul, 3,219 choice and 72 score.
- **The other sources are not gold.**
  - `yuri_v3` targets are Jev 1.13's own soft distributions: a teacher, not gold.
  - `yuri_v1` rows are all placeholder `[0.5, 0.5]`.
  - The temperature fit's 10,954 rows are the calibration split minus `yuri_v1`.
- **Upstream:** `ZefanCai/Open-Jev` (CC0).

## 3. TypeSafe's `POST /v1/systemone` (https://docs.typesafe.ai/api.md)

```http
POST https://api.typesafe.ai/v1/systemone
Authorization: Bearer <API_KEY>
Content-Type: application/json
```

```json
{
  "state": "Help! My payouts have been failing for 3 days.",
  "model": "jev-latest",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "billing": "Payments, invoicing, refunds",
        "technical": "Bugs, outages, integrations",
        "sales": "Pricing, upgrades, new accounts"
      }
    }
  }
}
```

```json
{
  "model": "jev-1.13.0",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "billing",
      "probabilities": { "billing": 0.88, "technical": 0.12, "sales": 0.0 },
      "confidence": 0.81
    }
  },
  "usage": { "input_tokens": 318, "output_tokens": 34 }
}
```

A score answer and a noul answer:

```json
{ "model": "jev-1.13.0",
  "answers": { "frustration": { "type": "score", "score": 1.05,
      "legend": { "0": "Calm", "1": "Frustrated", "2": "Very angry" },
      "probabilities": { "0": 0.0, "1": 0.95, "2": 0.05 }, "confidence": 0.92 } },
  "usage": { "input_tokens": 304, "output_tokens": 18 } }
```

```json
{ "model": "jev-1.13.0", "answers": { "is_urgent": { "type": "noul", "noul": 0.95 } },
  "usage": { "input_tokens": 296, "output_tokens": 20 } }
```

- **Kinds:** `noul`, `choice` and `score`. `state` and `instructions` may each be a string, object or array.
  - `noul`: `criteria` is optional `{true, false}` descriptions.
  - `choice`: `criteria` is a required `option → description|null` map, with at most 255 options.
  - `score`: `criteria` is an ordered array of 2–10 levels.
- **Many questions per state:** the `questions` map's keys are the caller's and are "not sent to the underlying
  model".
- **What each answer carries:** noul answers have no `confidence`. The `confidence` formula is not published; the
  docs' demo approximates it as `(3 × max − 1) / 2` for three options.
- **Errors:** 401, 422, 429, 529.
- **Models:** `jev-latest` = `jev-preview` = `jev-1.13.0`. Its `GET /v1/models` returns `{"models":[{name,
  description, release_date}]}`, which is not OpenAI-shaped.

**Base-URL override in every SDK:**

| package | option | path |
|---|---|---|
| `@typesafe-ai/sdk` 0.6.0 | `baseURL`, env `TYPESAFE_BASE_URL` | posts `/v1/systemone` |
| `typesafe-sdk` (PyPI; the GitHub repo is `typesafe-sdk-python`) 0.7.2 | `base_url=`, env `TYPESAFE_BASE_URL` | `/v1/systemone` |
| `@ai-sdk/typesafe-ai` 3.0.8 (Vercel) | `baseURL` in `createTypeSafeAi` | default `https://api.typesafe.ai/v1`, posts `${baseURL}/systemone`, so the override must include `/v1` |
| `langchain-typesafe` 0.0.1a3 | `base_url`, env `TYPESAFE_BASE_URL` | `/v1/systemone` |
| `@langchain/typesafe` 0.0.1 | `baseUrl`, env `TYPESAFE_BASE_URL` | `/v1/systemone` |

The Python SDK's response models ignore extra fields, and `usage.{input_tokens, output_tokens}` is required.

## 4. muthuishere/jevx

- **What it is:** a Go CLI (MIT, 2026-09-27) that runs no model. Verbs: `is`, `pick`, `filter`, `rank`,
  `ask --noul/--choice/--score` and `judge`.
- **Profile:** `{url (the full endpoint URL), model, headers ($VAR expanded), questions, defaults, note}`, e.g.
  `"local": {"url": "http://127.0.0.1:21131/v1/systemone", "model": "my-model"}`.
- **Request:** `{"model", "state", "questions"}`.
  - Each question is `{type, instructions, criteria}`.
  - The state is always sent as a string.
  - At most 32 questions go in one request.
- **Fail-closed validation of the reply:**
  - exactly the questions asked;
  - noul P in [0, 1];
  - probabilities sum to 0.98–1.02;
  - the choice is one of the offered keys.
- **Exit codes:** 0 = yes / decided; **1 = no**; 3 = unsure (between noul `no` 0.2 and `yes` 0.8, or a choice/score
  below `min_confidence` 0.6); 4 = any error. A missing `confidence` counts as 1.0.
- **The scenario guide** (https://muthuishere.github.io/jevx/guides/scenarios/) recorded its outputs against hosted
  `jev-1.13.0`.
  - About 9 scenarios need input files that are not published (`app.log`, `tickets.jsonl`, `invoice.txt`, …).
  - The inline ones are reproducible: push vs ask, force-push, effort score, question vs instruction, a password in
    a line, the build status, which function, the multi-question echo, `judge`, and the "fix the question" pairs.

## 5. Other reproductions

- **kyegomez/open-jev** (Apache-2.0) is confirmed as a random-weights, from-first-principles reconstruction with a
  different architecture.
- **Other open students**, from the awesome-list README (`valentynkit/awesome-jev-typesafe`; the wiki is empty) and
  `OmniJev/awesome-jev-gallery`. Base models are as the lists describe them; not verified.
  - Open-Jev (ZefanCai; `ZefanCai/Open-Jev-9B` on `Qwen/Qwen3.5-9B`, ships `head.pt` + `temperature.json`) and
    Bespoke Nimble.
  - Jebadiah (serves `/v1/systemone`), mini-jev (reads letter logits on a frozen Qwen3-4B), and kev (Qwen2.5, "speaks
    the TypeSafe wire format").
  - Many small Qwen3.5 / LFM2.5 / Gemma variants and some encoders (ModernBERT).
  - Serving shims over existing engines: ollaya (Ollama-style), local-jev, and SGLang/vLLM gateways.

## 6. Peer engines: none ships a decisions endpoint

- **llama.cpp server:** `n_probs`, `logit_bias`, grammar and `/reranking`; no decisions route.
- **Ollama:** `logprobs` + `top_logprobs` on the native `/api/generate` and `/api/chat` only (not on the
  OpenAI-compat routes); no decisions route.
- **vLLM:** `/classify` covers only a model's fixed trained labels, and `/score` and `/rerank` are for
  cross-encoders.
  - Label scoring works through `allowed_token_ids` + `logprobs` + `--logprobs-mode processed_logprobs`, which is
    how autotrust serves JEV.

## 7. What to download for D6a / D6b

- **JEV-9B's base is `Qwen/Qwen3.5-9B`,** the post-trained model. Its card says it "operate[s] in thinking mode by
  default".
  - `-Base` exists, but it is not JEV's backbone.
  - There is no official Qwen GGUF.
- **JEV-9B's own repo already holds the backbone and `lm_head`,** so one 18.27 GB download serves Route A (adapter
  off) and Route B.
- **Third-party Q4_K_M GGUFs** of `Qwen/Qwen3.5-9B`:
  - `unsloth/Qwen3.5-9B-GGUF`: 5,680,522,464 B, sha256 `03b74727…b7e8`;
  - `bartowski/Qwen_Qwen3.5-9B-GGUF`: 6,169,341,984 B;
  - `lmstudio-community/Qwen3.5-9B-GGUF`: 5,627,044,256 B.

## What the reference fixture needs (Linux box)

- **Code:** the card's `decide()`, or the Space's `jev_core.py` (`JEV_DEVICE=cpu`, `JEV_MODEL_PATH`).
- **Pins:** `transformers==5.16.1`, `peft==0.21.0`.
- **Weights:** `hf download autotrust/JEV-9B --revision 4ab5dfb9… --local-dir …` (18.27 GB).
- **Data:** `calibration.jsonl` and `ood.jsonl` from the dataset at `fc99c635…`.
- **Memory:** about 18 GB RAM in bf16. An f32 reference, which D6b's f32 bar needs, takes about 36 GB plus
  activations.
- **Unverified:** whether transformers' Gated-DeltaNet path runs on CPU without flash-linear-attention.
- **Two things to record with the goldens:**
  - whether the LoRA was merged (the card does `merge_and_unload` in bf16) or left unmerged (the Space);
  - the vLLM/HF noise floor above, when reading any agreement figure.
