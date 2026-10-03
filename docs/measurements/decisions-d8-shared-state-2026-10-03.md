# D8 — shared state, many questions: the CPU checkpoint path (2026-10-03)

D8 of `docs/tasks/task-constrained-confidence.md`: a decision request asks several questions about one state, and on the hybrid `qwen3_5` family each question used to prefill its whole prompt again (`decisions-d7-2026-09-28.md` section 5: five questions cost five times one, to 0.1%). The trigger fired at a state-prefill share of 0.909. This is the first half of the fix: **on the CPU path, the prefix a request's prompts share is prefilled once and each question resumes from a copy of that cache.**

## What was built

- **`decoder` (`prefix_share.go`).** `Model.PromptHiddenMany` and `PromptLogitsMany` take several prompts, find the longest common token prefix of each group of prompts that share at least 16 tokens, prefill it once into a cache, and resume every member from a deep copy (the KV rows, f32 or int8, and the Gated DeltaNet's state, which has no per-position history and so cannot be rewound, only copied). `CanSharePrefix` says whether a model takes this path: no resident backend, the batched Qwen3.5 forward, a cache that is only KV plus DeltaNet state. Everywhere else the methods are `PromptHidden` per prompt, exactly as before.
- **`internal/decide`.** `Decider.DecideMany` builds every question's prompt, hands them to the model in one call, and answers each question as `Decide` does (`scoreIDs` takes the precomputed output). A request that asks for several answer orders (`Permute > 1`) goes alone. The head route always gets the hook; the label route only on a model that can share.
- **`/v1/systemone`.** A request's questions go to the decider in one `DecideMany` call; the Clef route is unaffected (one backbone pass already).

## Gates

- **Tiny Qwen3.5, dense and MoE, f32 and int4, and an int8 KV cache** (`TestPromptHiddenMany_matchesSeparate`, `_int8KV`): five prompts sharing 40 tokens, shared against alone, worst relative L2 **6.65e-8** against a bar of 1e-6 (D11's batched-versus-sequential bar), and the hook proves exactly one checkpoint was taken for 5 members.
- **The checkpoint is a copy** (`TestPrefixCheckpoint_isNotMutatedByResuming`): resuming one checkpoint repeatedly, in different orders, gives identical results. **Able to fail:** sharing the recurrent state with the checkpoint, not restoring it, and not copying the KV rows each turn the tests red (checked by mutating `copyStateFrom`; the source was restored byte for byte).
- **Fallbacks:** a dense llama answers bit for bit as `PromptHidden` does and takes no checkpoint; a cancelled context stops the call; grouping (`TestShareGroups`) shares only what is really shared, keeps one token of every prompt for its own suffix, and ignores a prefix under 16 tokens.
- **`DecideMany`** (`internal/decide`): one many-call for three questions, identical distributions to `Decide` per question, fallbacks with no hook, with one request and with `Permute > 1`; a hook returning the wrong number of outputs is an error. **The endpoint** (`TestSystemOne_questionsShareOneModelCall`) sends the three questions as one call and answers them as they answer alone.

## On the real weights (EXPLORATORY: one run by day, never quoted as a result)

Clef-flash's backbone (a plain `qwen3_5`, 32 layers, 9B) at `int8int8` on nobara's CPU, D7's five questions about its first K=256 state, built as each decision template builds them (`TestPromptHiddenMany_realWeights`, `GOINFER_HEAVY_TESTS=1`):

| template | prompts (tokens) | groups found | shared | separate | ratio | shared vs separate hidden state |
|---|---|---|---|---|---|---|
| bare-v1 (JEV's head) | 5 (1,478) | `{noul, noul}` prefix 271, `{score}` alone, `{choice, choice}` prefix 270 | 52.7 s | 80.0 s | 1.52x | **bit-identical** (relative L2 exactly 0 on all 5) |
| chat-v1 (label scoring) | 5 (1,549) | all five, prefix 268 | 29.1 s | 83.3 s | 2.87x | **bit-identical** (all 5) |

The hidden states are bit-identical on the real int8int8 model, where the tiny f32 models differ by about 5e-8: per-row activation quantization makes each row independent of how many rows share the matmul. Do not read that as a guarantee for other precisions; the gate's bar is 1e-6.

## What it does not do, stated

- **The GPU-resident path is unchanged.** D7's numbers (and its trigger) were measured on a CUDA-resident model; this path never runs there (`CanSharePrefix` is false for any resident model, whose own prefill is far faster than a CPU one, about 3 ms per token against 65). Sharing on a resident backend means copying the DEVICE's recurrent state, a per-backend change in CUDA, WebGPU and Metal that does not exist.
- **bare-v1 shares less than chat-v1.** JEV's head template puts `[kind] <kind>` BEFORE the state and the head was trained on exactly that layout, so it cannot be reordered: only questions of the same kind share a prefix. Five mixed questions become three prefills, not one (measured above). Label scoring's chat template puts the state first and shares fully.
- **Where it applies today:** a Qwen3.5-family model on the CPU, which includes the Mac, where the 9B does not run resident (so the Mac's D14 JEV arm now runs with this on; see `decisions-d14-clef-speed-2026-10-03.md` section 2a).
- Only Qwen3.5 dense and MoE are covered. Mamba, conv, KDA and MLA caches, windowed and multimodal caches are refused by `copyStateFrom`, and those models fall back.
