# OP2b on CUDA: the Nemotron-H 4B CPU check and the fit indicator (2026-10-16)

Before any CUDA code, per §4.7's registered rule ("Look for a smaller real Nemotron-H first",
`docs/tasks/task-forward-architecture-2026-10.md`, OP2b on CUDA). nobara-pc, after the device gate had finished, nothing else loading the box.

## The checkpoint

`nvidia/Nemotron-H-4B-Instruct-128K` (public, not gated, license `other`), pulled to `~/models/nemotron-h-4b-instruct-128k` (8.4 GB on disk, two
safetensors shards). Its config: 52 layers (24 Mamba-2, 4 attention, 24 MLP by `hybrid_override_pattern`), hidden 3072, 112 Mamba heads of 64,
state 128, 8 groups, conv kernel 4, 32 attention heads over 8 KV heads, attention head dimension 128, vocabulary 131,072.

## Step 1 of the rule: does it load and generate on the CPU through today's `nemotron_h` adapter?

**Not as released. Two findings, both in the checkpoint's `config.json` and both product-code matters** (the loader, not a gate):

1. **`config.json` does not parse.** Line 47 is a bare `Infinity` (`"time_step_limit": [0.0, Infinity]`, written by Python's `json.dump`). Go's
   `encoding/json` rejects it: `decoder: parse config.json: invalid character 'I' looking for beginning of value`. Any HF checkpoint written
   this way fails the same way (`NaN` too).
2. **The attention head dimension is under a key the adapter does not read.** The config has `"attention_head_dim": 128` and no `head_dim`, so the
   adapter derives hidden / heads = 96, and the first attention layer's `q_proj` `[4096, 3072]` fails its shape check against `[3072, 3072]`.
   (The 9B is loaded from a GGUF, which carries the dimension in its own metadata, which is why it never met this.)

**A recorded workaround, not a fix:** a sibling directory `~/models/nemotron-h-4b-cfgfix` with the two shards symlinked and a `config.json` that differs
from the original in exactly two lines (`Infinity` to `1e30`, and an added `"head_dim": 128`). With it:
- `serve -backend cpu -quant int4` loads the 52-layer model; decode path `cpu (int4)`.
- Greedy decoding is coherent: "The ocean is a vast and mysterious body of water that covers over 70% of the Earth's surface, teeming with diverse
  marine life and playing a crucial role in regulating the planet's"; a raw "The capital of France is" answers "Paris" (the model has no chat template
  in this checkpoint, so that prompt then loops the prompt text, as a raw completion does).

So the checkpoint runs through the adapter once those two are handled. **Proposed, for the owner (product code, not done):** the config reader
tolerates `Infinity`, `-Infinity` and `NaN` tokens outside strings, and the Nemotron-H adapter reads `attention_head_dim` when `head_dim` is absent.
Each needs a test that goes red without it.

## The fit indicator (arithmetic, not gate F)

The CUDA int4 bundle of the 4B (`cmd/prequant -quant int4 -target cuda`, 5.5 s) is **3,665,816,296 bytes (3.41 GiB)**; the CPU bundle is 3.31 GB.
The card has 8,192 MiB with about 445 MiB held by the desktop. The recurrent state is small beside that: 24 Mamba layers of 112 x 64 x 128 f32
(3.7 MB each, about 88 MB) plus a conv window, and KV for 4 attention layers. So the 4B should fit with several GiB to spare; **gate F proper (the
server's own plan on a resident) needs the resident and waits for step 2.** The 9B's v17 bundle is 7.27 GB, the case that may not fit.

## Stale sidecars and what rebuilt them (the same day)

The device gate at `b04b0273` failed 10 heavy-tier tests on v16 sidecars (`gemma4-26b-int4.giw`, `qwen3.6-35b-a3b-int4.giw`). They were rebuilt at
v17 with `cmd/prequant -quant int4 -target canonical` from the sources the v16 rebuild record names
(`docs/measurements/giw-v16-transcode-2026-10/README.md`, Decision 4): the 26B in 80 s (13,914 MB) from `gemma-4-26b-a4b-it`, the 35B in 153 s
(19,117 MB) from `qwen3.6-35b-a3b-Q8_0.gguf`. The two fresh-process tests then passed by day: `TestP20ExpertLocality` (64 s) and
`TestSpecPagerInteraction` (169 s). The re-run of the rest is the night job `op1-gate-rerun-cuda`.
