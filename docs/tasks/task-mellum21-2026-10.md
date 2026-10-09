# Mellum2.1 (JetBrains/Mellum2.1-12B-A2.5B-Thinking): is it a checkpoint swap?

Developed on branch `mellum21` uncommitted (owner's house rule for this task, 2026-10-09), then committed and landed on `main` at the owner's word ("commit and push to origin main", same day). Owner's brief: three gates; Gate 0 today on either box, Gate 1 on the night queue, Gate 2 (the reason this is worth doing) on nobara-pc with its bar
registered before the run. Everything that loads real weights runs on nobara-pc from `~/models`; the 2.0 numbers are cited, not re-derived (`mellum` at real-oracle 100.0% / 0.99969).

## Gate 0, read 2026-10-09 (minutes, no weights): the vocabulary is unchanged, the template changed, and the config moved in two ways

Fetched from the pinned revision `92ddae9fc7665e9f801d141d2e5a6b2caf2460c4` (last modified 2026-10-07; ungated, Apache-2.0): `config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, and three more small non-weight files
(`generation_config.json`, `special_tokens_map.json`, `model.safetensors.index.json`) and the card. Compared byte for byte with 2.0's copies (the archive's `mellum2-unq`; read for a diff only, no measurement reads from the archive).

| file | 2.0 against 2.1 |
|---|---|
| `tokenizer.json` (7,091,587 B) | **byte-identical** (sha256 `58548a346eb0...`) |
| `tokenizer_config.json`, `special_tokens_map.json` | **byte-identical** |
| `chat_template.jinja` | **changed** (3,456 -> 4,849 B) |
| `config.json` | **two keys**: `eos_token_id` 0 -> 28; new `"use_sliding_window": true` |
| `generation_config.json` | `eos_token_id` 0 -> 28; `transformers_version` 4.57.1 -> 5.8.1 |
| `model.safetensors.index.json` | same size and tensor names (5,631 tensors, 5 shards, 24,299,846,144 bytes); the weights differ, as they should |

- **Vocab: identical, and no thinking tokens were added.** `<think>` (23), `</think>` (24), `<|im_start|>` (27), `<|im_end|>` (28), `<tool_call>` (29), `<tool_response>` (31) are all in 2.0's vocabulary already. **`TestByteLevel_mellum2GoldenParity` carries over, and it passes for real against the 2.1 tokenizer files (21 passes);** on this box it had been *skipping*, because `testdata/mellum2-tokenizer` is a symlink to `~/models/mellum2-unq`, which is no longer on this disk (the run used a staging directory and the symlink was restored untouched; Gate 1 compares the pulled `tokenizer.json` with the staged copy by `cmp`).
- **The brief's claims, checked against the real `config.json`:** 28 layers; hidden 2304; 32 query / 4 KV heads, head dim 128; 64 experts, 8 per token, `moe_intermediate_size` 896; `sliding_window` 1024 on **21 of 28** layers (`layer_types` is the pattern `SSSF` seven times: three of every four); 131,072 positions; BF16; YaRN (factor 16, attention_factor 1.2773) on the full layers and plain RoPE on the sliding ones, theta 500,000 both; untied head; Apache-2.0. The card says to serve it with vLLM's `--reasoning-parser qwen3` and that **GGUF builds and the MTP head for speculative decoding are "coming soon"** (not landed).
- **`eos_token_id` is now 28 (`<|im_end|>`), where 2.0's was 0 (`<|endoftext|>`).** goinfer reads it correctly: the 2.1 config resolves through `resolveArchitecture` as family `mellum` (28 layers, 7 global / 21 sliding, window 1024, 32/4/128, 64 experts top-8) and its stop ids resolve to `[28]`. A harness that assumed 0 would run on; goinfer's chat path already stops on `<|im_end|>`.
- **`use_sliding_window: true` is inert on both sides:** goinfer's mellum path ignores it (it parses the field for Qwen2, which rejects `true`, but `validateMellum` does not read it) and Hugging Face's `MellumConfig` never reads it either (windowing comes only from `layer_types` and `sliding_window`).
- **Hugging Face loads the 2.1 config: `transformers` 5.15.0 has `models/mellum` (the checkpoint was written by 5.8.1).**

### The chat template changed, and goinfer rendered it wrongly until this branch

2.1's template adds Qwen3's thinking control and history rule to the same ChatML body: (1) `enable_thinking is false` appends `<think>\n\n</think>\n\n` after the generation prompt; (2) an assistant turn AFTER the last user query is rendered `<think>\n{reasoning}\n</think>\n\n{content}` when it has reasoning, turns before it content only; (3) `reasoning_content` (or tags inside the content) is read the way Qwen3's template reads it. Unlike Qwen3, an after-query turn with **no** reasoning gets no empty block.

- **Measured, before any change:** goinfer detects the template as its built-in `mellum2` renderer, which declares no reasoning behaviour, so it rendered 2.0's rule. Against Hugging Face's rendering of the real 2.1 template over `scripts/pin_chat_think_history.py`'s 20 conversations x 3 thinking modes (60 prompts, tool loops included): **10 matched, 50 differed.**
- **Fixed on this branch (two small edits, uncommitted):** `chat/chat.go`: the `mellum2` branch of `Detect` now attaches `detectChatMLReasoning`, `declaresQwen35XMLTools` and `detectGroupedToolResults` as the generic ChatML branch does; `chat/history.go`: a new history kind `histMellum21` (Qwen3's rule, the block only for a turn with reasoning). **After: 60 of 60 prompts byte-identical to Hugging Face.** The whole `chat` package passes, including the existing 2.0 Mellum2 goldens (2.0's template has no `enable_thinking`, so its detectors still return nil: unchanged).
- **Committed as tests:** `chat/mellum21_history_test.go` (all 60 prompts byte-identical to Hugging Face's, pinned in `testdata/chat_think_goldens/mellum21_think_history.json` by `PIN_CKPTS=mellum2.1 scripts/pin_chat_think_history.py`; red with `chat/chat.go` reverted: "Detect attached no reasoning behaviour") and `decoder/mellum21_config_test.go` (the real 2.1 `config.json` in `testdata/mellum21-config/`: shape 28 layers / 7 global / window 1024 / 32-4-128 / 64 experts top-8, stop ids `[28]`). The earlier throwaway `zz_` tests are gone.

## Gate 1: the weights (queued, two night jobs, `~/models` on nobara-pc)

**What the existing gates are, read from their code before writing this** (the brief names them as if they ran against the checkpoint; two facts differ):

1. `TestMellum2_logitParity` and `TestMellum2_windowParity` compare goinfer (int8int8) with a **pinned golden** made from 2.0's weights by `scripts/pin_mellum2.py` (HF bf16, CPU), at argmax exact and a sample-256 cosine floor of 0.98 (2.0 measured 0.99955 and 0.99636). They hardcoded `~/models/mellum2-unq` and `testdata/mellum2_*_golden.json`. **For 2.1 a new golden pair must be pinned from 2.1's real weights first.** Both now read `GOINFER_MELLUM_CKPT` and `GOINFER_MELLUM_GOLDEN_PREFIX` (defaults unchanged; a non-default pair never writes the 2.0 family's manifest row), and `pin_mellum2.py` takes `MELLUM_PIN_PATH / _PREFIX / _MODEL_ID / _REVISION / _NOTHINK` and writes a provenance block (revision, per-shard sha256 of the weights read, transformers and torch versions).
2. `TestMoEExpertMajor_endToEnd` is **not an identity gate**: it is P18's timed decision measurement (FUND at >= 15%, PARK under 8%; 1,207 s per-row against 277 s expert-major at K=4096 on this box, about 113 minutes). The identity is `TestMoEExpertMajor_bitIdentical` (`!=` on every logit through the real forward, flag on against off, K=600 by default, the 2.0 record also at K=4096). The job runs the identity test as is and the end-to-end test at K=600, one pair, **as an engagement check only** (it asserts expert-major ran in the "on" arm and not in the "off" arm), and does not re-make the P18 decision.

**Checkpoint pinning:** repository `JetBrains/Mellum2.1-12B-A2.5B-Thinking`, revision `92ddae9fc7665e9f801d141d2e5a6b2caf2460c4`, 24.30 GB; each shard is verified against Hugging Face's own sha256 (`f5ccf801...`, `266beea6...`, `da9a04fb...`, `32f27ab7...`, `d8bbb9b8...` are the first 16 hex); the goldens carry the per-shard hashes of the weights actually read and `transformers` 5.15.0 / `torch` 2.12.0+cpu.

**Bars (2.0's, unchanged; nothing is re-based):** argmax exact; sample-256 cosine >= 0.98 on both goldens; bit-identity: every logit equal with expert-major on and off; the engagement check reads "ran" in one arm and "did not" in the other. **All four at the 2.0 bar -> "the family covers 2.1, and the capability matrix gets the checkpoint added, not a new family."** Any miss is the finding, reported with the first divergent layer from a per-layer diff, and the reference is checked for a `persistent=False` buffer problem first (Mellum's YaRN `inv_freq` is exactly that kind of buffer; check it finite and of the expected magnitude before blaming goinfer).

**One honest difference from 2.0's golden:** 2.1 is a *thinking* model. With the default template the first generated token after `<|im_start|>assistant\n` is the model opening its own block, not an answer; the chat golden is therefore pinned with `enable_thinking=False` (the closed block, then the answer token), which keeps the "argmax is the first answer token" coherence property 2.0's golden had. The default-thinking prompt is covered by the template comparison above, not by a logit golden.

**Cost (estimates, not measurements; the basis is stated):**

| job | what | est | basis |
|---|---|---|---|
| `mellum21-pull-pin` | pull 24.3 GB, sha256 of it twice, HF bf16 load, the two forwards | 90 min queued (about 50 expected) | network speed unknown (5-25 min); sha256 of 24 GB about 2-3 min each; the 2.0 pin was feasible on this box in one session, its time not recorded; the 1,441-token bf16 forward on a CPU without native bf16 is the long pole |
| `mellum21-gates` | three loads of the 12B, the window prefill, K=600 forwards | 150 min queued (about 100 expected) | P18's own record: 0.29 s per row per-row, 0.066 s expert-major (K=4096), so K=600 is about 176 s and 40 s per forward; the sequential window prefill of 1,441 tokens and the int8int8 load have no prior timing (they skipped in the last suite run) |

**Not runnable by day** (every item is over the 10-minute rule, and the `endToEnd` check is a timed run that holds the box's timing lock).

## Gate 1 RESULT, read 2026-10-09 (night queue run by the owner at 13:34 PDT; both jobs ok, 27 min against 4 h estimated)

Logs: `~/goinfer-logs/night/runs/2026-10-09/mellum21-pull-pin.log` and `mellum21-gates.log`; goldens in `~/goinfer-logs/mellum21-gold/` (not committed; regenerable by the pull-pin script). Pinned: repo `JetBrains/Mellum2.1-12B-A2.5B-Thinking` @ `92ddae9fc7665e9f801d141d2e5a6b2caf2460c4`, every shard verified against Hugging Face's sha256, transformers 5.15.0, torch 2.12.0+cpu. Binary: `~/goinfer-bench/mellum21/decoder.test` from tree `e98b0905fa9d9d7be998c606a571a77261efc773` (branch `mellum21`, working tree dirty).

| gate | bar (2.0's, unchanged) | 2.1 | 2.0 |
|---|---|---|---|
| tokenizer | byte-identical to the copy the golden test passed on | PASS (sha256 `58548a346eb073e5`) | |
| `TestMellum2_logitParity` | argmax exact; sample-256 cosine >= 0.98 | PASS: argmax 50195 equal; cosine **0.99971** (31 s) | 0.99955 |
| `TestMellum2_windowParity` | same, 1,441 tokens past the 1,024 window | PASS: argmax 233 equal; cosine **0.98600** (883 s) | 0.99636 |
| `TestMoEExpertMajor_bitIdentical` | every logit equal, expert-major on vs off, K=600 | PASS (73 s) | |
| engagement (`endToEnd`, K=600, one pair) | ran in "on" arm, not in "off" | PASS: per-row 39.3 s, expert-major 11.6 s (3.37x; one pair, an engagement check, NOT the P18 decision) | |

**Verdict at the registered bar: all gates pass; the `mellum` family covers 2.1; the capability matrix gets the checkpoint added, not a new family.** **Unexplained, stated plainly:** the window cosine is 0.986 against 2.0's 0.996, still above the 0.98 floor but 0.01 lower. Not investigated (the registered bar is met); a per-layer diff on the window golden would say whether it is RL weight drift or the window path. The `persistent=False` check on the reference (YaRN `inv_freq` finite and of the expected magnitude) was not run because nothing missed the bar.

## Gate 2: the reason this is worth doing. PRE-REGISTERED 2026-10-09, before any Gate 2 run (nobara-pc, RTX 2070 SUPER 8 GB, `~/models`)

**The question:** can a 12B-A2.5B MoE hold a harness-scale tool loop, and what does it cost in resident memory? The standing finding (`docs/integrations/opencode.md`): nothing under about 7B has held one, and the only measured success is Qwen2.5-7B-Instruct q4_k_m on this box (all 8 `serve check` rows including `tools, harness-scale`; 6,824 / 8,192 MiB; opencode's build agent made real `Read` and `Glob` calls and answered correctly, two turns of 7,165 and 7,399 input tokens). Memory-fit and tool-calling capability are independent axes; the active set of 2.5B may behave like 2.5B, not like 12B.

**The run (fixed now):** `serve --model ~/models/mellum2.1-thinking --quant int4 --backend cuda --ctx 16384 --addr 127.0.0.1:<port>`, adding `--moe-cache-experts` only if the plan declines a fully resident load, and nothing else; then `serve check <url>` once on a cold server, followed by `nvidia-smi` for the footprint. No flag is tuned after seeing a reading.

**What is recorded first, whatever happens:** the banner's `decode path`, KV plan and `weights ... GB resident` lines; VRAM in MiB at idle and after the check; load time. The footprint is the second axis and is reported beside the outcome whether the outcome is good or bad.

**The bar, on the unmodified `serve check` (the instrument whose rows predicted every entry in the table above):**

| outcome | meaning, fixed in advance |
|---|---|
| `tools, OpenAI` ok **and** `tools, harness-scale` **ok** | **harness-scale success** (this model's second measured one and the first at this size). Then a REAL opencode loop is driven end to end and recorded the way the Qwen2.5-7B success is: the configuration line, load time and decode path, resident GB and VRAM before/during/after, the same style of task (list the directory with the tool, then read a file and say what is in it), the tool calls actually made, the answer against the file's content, input and output tokens per turn. |
| `tools, harness-scale` **skip** (the model answered in prose or called nothing) | **a size/capability result against the footprint**: recorded in `opencode.md`'s table as "skip" with the measured footprint, with the note that an A2.5B active set behaves like its active size and not its total. Not a failure of goinfer. |
| any other row **fails**, or the server cannot load it | the finding is the failing row or the load decline, root-caused before anything about tool capability is said; a `tools, OpenAI` failure is a bug (as on 2026-09-06), not a capability result. |

**One confound named in advance, and how it is handled:** `serve check`'s harness-scale row sends `max_tokens: 96`. 2.1 is a thinking model and opens its own `<think>` block by default, so it can spend all 96 tokens reasoning and "not call the tool" for a budget reason, not a capability one. **The registered reading is the unmodified row as it comes.** If it reads skip AND the response shows the model was still reasoning at the limit (a reasoning-only body, `finish_reason` length), that is reported as **"skip, budget-exhausted"**, never as a capability result; the same request is then replayed once by hand with `chat_template_kwargs: {"enable_thinking": false}` and the same 12 tools and `max_tokens` 96, and that second reading is reported separately and labelled as not the registered row. It does not replace the first.

**Prediction, written now:** the model loads resident or with `--moe-cache-experts` (an int4 12B is about 6.5 GB of weights; the 2070 has about 7.5 GB free), at most one of `tools, OpenAI` and `tools, harness-scale` is not ok; I put `harness-scale ok` at **0.40** (RL on tool use and SWE tasks is the best argument for it; the evidence that nothing under 7B has done it, and a thinking default under a 96-token cap, the best against), `skip, budget-exhausted` at **0.25**, `skip` as a capability result at **0.25**, and a load or row failure at **0.10**. What would surprise me: any row other than the two named failing.

**Limits, stated now:** one box, one quantisation (int4 from safetensors; the GGUF is not out), one sampled `serve check` at temperature 0 (the check is deterministic but a single draw of one prompt), one opencode task if the bar is cleared, so a success would be "this configuration did it once, measured", the same kind of claim as the 7B's; no claim about longer loops or other tasks.

**Cost:** about 40 minutes (the int4 sidecar/quantisation of a 24 GB bf16 checkpoint at first load; the check itself is under 2 minutes; an opencode loop about 10); not queued until Gate 1 reads, because a failed Gate 1 changes what Gate 2 would mean.

### Amendment A1, 2026-10-09 14:30 PDT, written BEFORE any `serve check` reading (none has run)

Two load attempts at the registered flags failed before serving, so no row was read. Facts, from `~/goinfer-logs/mellum21-gate2-2026-10-09/attempt1/` and the files beside it:
- On a 24 GB bf16 checkpoint the server first transcodes to `mellum2.1-thinking.int4.e4h.cuda.giw` (6,539 MB, 30 s; "stream-weights", automatic). The resident plan then **declined**: ctx 16384 needs 1.88 GB of KV (112.0 KB/position across 28 layers; the sizing does not shrink the 21 sliding-window layers to the window) against 0.66 GB free beside the weights.
- Both that attempt and the registered `--moe-cache-experts` fallback (C′ capped to 45 of 64 slots/layer, 4.5 GB) then stopped at the loader's own check: `--quant "int4" cannot apply to the prequantized .giw bundle ... it is baked at "int4mix" ... pass --quant int4mix or omit --quant`.

**Amendment:** the registered command line is unrunnable as written for this checkpoint, so `--quant int4` is dropped (omitted, as the loader instructs; the bundle's own quant, `int4mix`, is what is served). Everything else is unchanged: `--backend cuda --ctx 16384`, `--moe-cache-experts` only after a declined resident plan, the bar, the confound handling, the prediction. The `.giw` quant is a *mixed* int4 recipe, so the result is stated as "int4mix as the loader bakes it for this checkpoint", never as plain int4.
**Two findings regardless of the outcome:** (1) goinfer cannot hold Mellum2.1 resident on 8 GB at ctx 16384 because the KV plan ignores the sliding window on 21 of 28 layers (a lever, untested); (2) the first-run path (`--quant int4` on safetensors that stream-transcode) rejects its own cache, which is a user-facing defect in serve.

## Gate 2 RESULT, read 2026-10-09 14:27-14:35 PDT (nobara-pc, RTX 2070 SUPER 8 GB, driver 595.91.07), records in `~/goinfer-logs/mellum21-gate2-2026-10-09/` (`attempt1/`, `attempt2/`, the main-built reading at the top level, `branch-build/`)

**Configuration actually run (amendment A1 applies):** `serve --model ~/models/mellum2.1-thinking --backend cuda --ctx 16384 --moe-cache-experts` (no `--quant`; the loader serves its own `int4mix` bundle, 6,539 MB on disk, built once in 30 s). `--moe-cache-experts` was added by the registered rule, because the plain load declined a resident build. `decode path: cuda-resident (int4mix)`, C′ cache capped at **45 of 64 expert slots per layer (4.5 GB)**, KV f32, one conversation at 16,384 positions, one generation at a time. **Footprint: 7,377 MiB of 8,192 idle, 7,379-7,381 MiB after the check** (no growth through it). Load from the cached bundle: 20 s. First-time transcode of the 24 GB checkpoint: 30 s.

**Two readings, and the difference between them is a bug in the first binary, not noise.**

| binary | `tools, OpenAI` | `tools, harness-scale` | other rows |
|---|---|---|---|
| `main` build (`70de7994`; knows only 2.0's template: no reasoning detection) | ok | **skip**: `finish_reason="length"`; the 96 tokens were all inside the model's own `<think>` block, which this build leaves in `content` | 7 of 9 pass, `long prompt` ok (19.58 s TTFT, 63.2 tok/s), chat 69.1 tok/s |
| this branch's tree (`e98b0905` + the uncommitted template edits), same flags | ok | **ok**: `call get_weather({"city": "Paris"}) among 12 tools` | `long prompt` **FAIL** "stream produced no tokens" (see below); chat 71.4 tok/s; 7 ok, 1 skip (vision), 1 FAIL |

**Which is the reading of record.** The registration asks whether goinfer serving this checkpoint can hold a harness-scale loop. The first binary renders 2.1's template wrongly (10 of 60 prompts matched in Gate 0) and leaves reasoning in the content stream, so it is not the code that serves 2.1; the branch tree is, and it is the reading that **meets the registered "ok and ok" bar**. Both are reported because the registration did not name the binary and I chose the branch build AFTER seeing the first skip. The first reading is the registered "skip, budget-exhausted" case (body shown: reasoning only, `finish_reason=length`), exactly as the confound paragraph predicted. **The success is therefore conditional on the template edits being committed**; it does not hold for a `main` build.

**Why the branch build calls the tool inside 96 tokens.** With reasoning detection the server clamps the thinking block to leave room to answer (`chat/budget.go`, `internal/serveapp/think.go`): replay A, same request as the row: reasoning cut after about 75 tokens, then the grammar-constrained call, 96 completion tokens, `finish_reason="tool_calls"`, `reasoning_content` carries the think text, `content` empty. Replay B (the labelled not-registered second reading, `enable_thinking:false`): the call in **21 tokens**, no reasoning at all. Replay C (exploratory, 2,048 budget): 131 tokens, same call. All three emit `get_weather({"city": "Paris"})`; on the `main` build replay B had no effect (the old renderer ignores `enable_thinking`), which is how the missing template support first showed.

**The `long prompt` FAIL is the row's own 8-token budget, not a serving fault.** The row sends `max_tokens: 8` and counts only `delta.content` events (`internal/servecheck/check.go:183`). Measured on this server by streaming: at 8 and 16 tokens, 0 content deltas and 5 / 11 reasoning deltas (all spent thinking); at 48 tokens, 8 content and 35 reasoning. A thinking model needs a larger budget for that row, or the row must count `reasoning_content` as proof of a live stream. That is a `serve check` defect against reasoning models, to be fixed in the instrument, not in the model.

**Against the prediction (written first):** harness-scale ok at 0.40, skip-budget 0.25, capability skip 0.25, load/row failure 0.10. The outcome on the model-serving code is the 0.40 branch; the first binary landed in the 0.25 budget branch; the load failures before either (a `--quant` rejection, a KV decline) were not in the prediction at all.

**What a harness-scale `ok` here does and does not say.** One draw at temperature 0 of one prompt (the same single-sample limit the 7B's `ok` has), plus replay C and the loop below. The footprint is the headline cost: **a 12B-A2.5B MoE holds a harness-scale tool call on an 8 GB card only through expert streaming (45 of 64 slots), at 71 tok/s decode, 7.4 GB VRAM, not fully resident.**

### The opencode loop (registered follow-up for "ok and ok"), 2026-10-09 ~14:50 PDT

opencode 1.18.29 (`npm install --prefix`, an isolated directory, nothing global; the version the owner's existing plugin config matches), project-local `opencode.json` pointing at the branch-built server above (`serve-cuda-tmpl`, rev file beside it, tree dirty), same flags; `opencode run -m goinfer/mellum2.1-thinking` in a directory holding `notes.txt` (4 lines) and `opencode.json`, with the 7B's prompt: *"list the files in this directory using your tool, then tell me what's in notes.txt"*. Records: `~/goinfer-logs/mellum21-gate2-2026-10-09/opencode/`.
- **Three turns, two real tool calls, then the answer:** `glob "*"` (listed `notes.txt`, `opencode.json`), `read notes.txt` (the file), and a final text answer that reproduces the four lines of the file exactly ("The deploy key rotates every 90 days / Staging runs on port 4410 / Owner: Dana"), matching the file's content.
- **Tokens per turn (opencode's own counts):** input 11,646 / 11,842 / 12,021, output 136 / 70 / 38 (the 7B's turns were about 7.2-7.4k in, 44 and 16 out; opencode's prompt is larger in this version). Context 16,384, so it fits with about 4.3k to spare.
- **VRAM:** 7,377 MiB before; samples every 3 s during: 7,377, 7,473, 7,503 MiB peak (about +126 MiB); 7,377 after. Margin about 690 MiB at the peak, thinner than the 7B's (6,824 flat) but no growth beyond the sampled peak.
- **Reading:** this is a measured success of the same kind as the Qwen2.5-7B's: one task, one run, this configuration, one box; no claim about longer loops. It is a **different kind of fit** (streaming experts, 7.4 GB) and it requires the branch's template/reasoning support.

## Night follow-ups, PRE-REGISTERED 2026-10-09 ~16:30 PDT, before any of the three has run (owner: "queue all 3")

Three jobs on nobara-pc, `~/models`, one timed run per box (the queue serialises them and holds the timing lock). Each names tier, instrument, stopping rule, cost basis and bar (TE11). Binaries are pinned: `~/goinfer-bench/mellum21/followups/{decoder.test,serve-cuda,rev}`, built from the commit that carries this section. No result below is known; the only prior reads are cited.

### A. Where does 2.1's window-golden cosine come from? (`run-mellum21-followup-a.sh`, est 70 min queued, about 35 expected)

- **Question.** `TestMellum2_windowParity` reads 0.98600 on 2.1 against 2.0's 0.99636 (floor 0.98, argmax exact on both); the short chat golden reads 0.99971 against 0.99955. Is the length-dependent drop a window-path defect, a long-context full-attention (YaRN) defect, or neither?
- **Instrument.** `TestMellum2_layerDump` (goinfer int8int8, the same sequential path as the window gate, `ForwardCapture` of all 28 layers' residuals at positions 10, 300, 600, 900, 1000, 1030, 1100, 1300, 1440 of the 1,441-token golden; it checks this run's own argmax and logit cosine against the golden) and `scripts/diff_mellum_layers.py run` (HF bf16, a forward hook on every layer: the residual after the layer, before the final norm; HF's `hidden_states[-1]` is post-norm and is not used). Positions <= 1023 are "short" (the 1024 window cannot have altered them); >= 1024 are "long". Proven before queueing: the planted-defect self-test names a planted layer and its type in both cases (`selftest`), and the whole pipeline ran on a tiny random Mellum (cosines 0.9998-0.99997, class N, rotary buffers printed).
- **Rule.** `d_short(l)` / `d_long(l)` = 1 - the median per-layer cosine over short / long positions. An EXCESS layer has `d_long > d_short + 0.005`. **W:** the first excess layer is a sliding layer (the window path is suspect). **F:** it is a full-attention layer (long-context full attention / YaRN suspect). **N:** no excess layer (the logit-level drop is not localised by layer; the final-layer `d_long` and `d_short` are reported). The reference is also checked for the `persistent=False` trap: every `inv_freq` buffer must be finite and in range, and a non-finite one is "the reference is broken", not goinfer.
- **Stopping rule.** Fixed: one dump, one HF forward, all positions. No second run on a miss.
- **Cost basis.** The window gate's 883 s (measured tonight) plus the pin's HF forwards (195 s for two, measured); about 35 min, queued at 70.
- **What it cannot say.** There is no 2.0 weights copy on this disk, so "2.1's weights are more sensitive than 2.0's" is NOT testable here: class N would leave exactly that possibility open, and the 2.0 arm needs `models-pull mellum2-unq` (24 GB) and a second job.
- **Prediction (written now).** N, at 0.50 (a 0.01 cosine at the logit is an amplified small per-layer difference that no layer crosses 0.005 on); F at 0.25; W at 0.15; a broken instrument or a non-finite reference buffer at 0.10.

### B. Does expert-major still earn its default on 2.1? (`run-mellum21-followup-b.sh`, est 130 min queued, about 80 expected)

- **Instrument.** `TestMoEExpertMajor_endToEnd`, exactly P18's: K=4096, 2 pairs interleaved with alternating lead after one discarded warm forward, one process, the flag on against off, int4 on the CPU path; it fails if expert-major did not run in the "on" arm or did run in the "off" arm.
- **Bar (P18's own, written before 2.0 ran, `docs/queue-performance.md` P18).** Net >= 15% end to end: FUND (it keeps its default). < 8%: PARK (it no longer earns the default on this checkpoint: the finding). 8-15%: AMBIGUOUS, parked pending a second mechanism. Nothing is re-based.
- **Stopping rule.** Fixed N (2 pairs); no early stop.
- **Cost basis.** The 2.0 record: 1,207 s per-row, 277 s expert-major per forward at K=4096. Warm (1,207) + 2 x (1,207 + 277) + int4 load = about 80 min; queued at 130.
- **Prior read, not a result.** Tonight's K=600 engagement check measured 3.37x on one pair; the 2.0 ratio at K=4096 was 4.36x. **Prediction:** FUND with a ratio above 3x, at 0.90. A PARK would say the RL checkpoint's expert routing no longer concentrates the way 2.0's did.

### C. What is the prize for a windowed-KV plan on CUDA? (`run-mellum21-followup-c.sh`, est 30 min queued, about 12 expected)

- **Why it is a measurement and not the change.** The CUDA resident plan prices KV for all 28 layers although 21 are windowed to 1024 (`112.0 KB/position`). Measured today on this card with the 6.5 GB int4mix bundle: **fully resident loads at ctx 1024 and 2048 and declines at 3072 and 4096** (the plan wants more than the KV alone: 0.47 GB of KV at 4096 against 0.66 GB "free" still declines), and at 16384 the model runs only through expert streaming (C'). The lever (windowed KV on CUDA) is a kernel and cache change and is NOT built; this job sizes it.
- **Instrument.** `resident_vs_cprime.py`: the same model at ctx 2048, fully resident (R) against `--moe-cache-experts` (S, 45 of 64 slots per layer), sessions ABBA (R S S R), one cold server each, one discarded warm request, then 3 fixed prompts x 3 reps, greedy, thinking off, 128 tokens; decode rate = (n-1)/(t_last - t_first) over the streamed content deltas. A resident arm that did not load `cuda-resident` without a C' line, or an S arm without the C' line, **voids** the job (it exits `VOID`). The greedy texts are compared across arms (C' is documented bit-identical to resident); a difference is reported, not hidden.
- **Bar.** Overall = the median over prompts of (median R rate / median S rate). **>= 1.15: worth building** the windowed-KV plan for residency. **< 1.05: park** (streaming already costs under 5% at this size). 1.05-1.15: AMBIGUOUS, parked. The two sessions of each arm are compared to read the session drift, and the per-prompt spread is printed; a ratio inside that spread is read as ambiguous even if it clears a bar.
- **Stopping rule.** Fixed (2 sessions per arm, 9 requests per session).
- **Cost basis.** Four loads of the cached bundle at about 20-25 s (measured today) plus 36 streamed requests of about 2 s each.
- **Limit, stated now.** ctx 2048 is the only context at which the two arms can both run; the prize at the context a harness needs (16384) is an extrapolation of this one, because attention cost differs. A smoke of the driver today (two requests per arm, 32 tokens) ran end to end and is **not a result**; it is not quoted here for that reason.
- **Prediction.** Ratio in 1.05-1.15 (ambiguous) at 0.45; >= 1.15 at 0.35; < 1.05 at 0.20.

## What this work touched

- `chat/chat.go`, `chat/history.go` (the template work); `decoder/mellum2_parity_test.go` (env overrides); `scripts/pin_mellum2.py` (parameters and a provenance block, defaults unchanged).
- New: `docs/tasks/task-mellum21-2026-10.md` (this), `docs/measurements/mellum21-2026-10/run-mellum21-pull-pin.sh` and `run-mellum21-gates.sh`, and the two committed tests above with their fixtures (`testdata/chat_think_goldens/mellum21_think_history.json`, `testdata/mellum21-config/`); `scripts/pin_chat_think_history.py` gained a `PIN_CKPTS` override (default unchanged).
- Gate 2 (new): `docs/measurements/mellum21-2026-10/run-mellum21-gate2.sh`; its binaries `~/goinfer-bench/mellum21/serve-cuda-tmpl` (built from this tree, rev file beside it) and, from earlier, `~/goinfer-bench/s6fix/serve-cuda`; and `~/models/mellum2.1-thinking.int4.e4h.cuda.giw` (6.4 GB) + `.verified` next to the checkpoint.
- Outside git: `~/goinfer-bench/mellum21/` (the fetched small files `gate0/`, the staged tokenizer files `tok21/`, the pinned `decoder.test`, `rev` and `tree.diff`, the HF template golden `mellum21_think_history.json`); the night queue gains two jobs. Nothing else on disk moved; the `testdata/mellum2-tokenizer` symlink was repointed for one test run and restored.
