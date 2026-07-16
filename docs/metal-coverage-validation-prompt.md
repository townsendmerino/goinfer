# Prompt for the Linux box — validate Metal model-coverage (Qwen3 / Mistral / Phi-3)

> Paste everything below to the Linux-box Claude. It's self-contained.

---

You're on the Linux box for `goinfer` (pull latest `main` first). The Mac added GPU **Metal**
support for three new architecture families and I need help validating them. **Key constraint:
the Metal backend is `//go:build darwin` — it only runs on macOS, so the Metal-vs-CPU numeric
comparison must run on the Mac. Your job is the parts that DON'T need Metal and that de-risk the
Mac run.** Do NOT try to build/run the `metal/` package (it's empty on Linux).

## What landed on main (Mac side, all unit-validated, NOT yet end-to-end)

Recent commits: `fix(metal): admission check`, `feat(metal coverage): Qwen3 …`,
`feat(metal coverage): Phi-3 partial RoPE + Mistral sliding-window …`. Three features were added
to the Metal resident decoder + prefill, each gated on a decoder-side arch flag:

| Family | Feature | Metal reads (decoder accessor) | CPU truth it must match |
|---|---|---|---|
| **Qwen3** | per-head Q/K RMSNorm before RoPE | `m.HasQKNorm()` (+ `m.RMSAddOne()`) | `decoder/attention.go:94-96` `rmsNorm(q/k, QNorm/KNorm, …, RMSAddOne)` |
| **Mistral** | sliding-window attention | `m.SlidingWindowResident()`, `m.LayerIsLocalResident(l)` | `decoder/kvcache.go` `WindowStart(pos, global)` |
| **Phi-3** | partial RoPE (`rotaryDim < headDim`) | `m.PartialRotary()`, `len(m.RopeInvFreq())` | `decoder/rope.go` `applyRoPE` (half = rotaryDim/2) |

The Mac unit-tested each kernel bit-exact vs the CPU math (`TestQKNorm`, `TestRopePartial`,
`TestSlidingWindow`) and confirmed qwen2.5 is unaffected. **What's unvalidated: a real
Qwen3/Mistral/Phi-3 checkpoint through the full Metal forward.** The single biggest risk is
NOT the kernels — it's whether the **decoder correctly detects these arch flags** from a real
checkpoint. If `HasQKNorm()` returns false for a Qwen3 GGUF, Metal silently skips QK-norm →
wrong output (exactly the silent-wrong class the admission fix just closed).

## Your tasks (all Linux-runnable, no Metal)

### 1. Verify decoder arch detection for each family (highest value)
For a **Qwen3**, a **Mistral**, and a **Phi-3** checkpoint (whatever you have, or fetch small
ones — see §2), load with `decoder.Load(path, decoder.Options{Quant:"int8int8"})` and assert the
flags Metal depends on. Add a focused test (or a scratch `main`) in the **decoder** package:

- **Qwen3**: `m.HasQKNorm() == true`; `lw.QNorm`/`lw.KNorm` non-empty (len == headDim) for each
  layer; `m.RMSAddOne()` == false (Qwen3 is plain `w`, not Gemma's `1+w`); no sliding window;
  full rotary (`m.PartialRotary() == false`); **and `m.DecodeRunnerEligible() == true`** (else
  Metal never even builds).
- **Mistral**: `m.SlidingWindowResident() == <the model's window, e.g. 4096>`; `HasQKNorm()==false`;
  full rotary; `DecodeRunnerEligible() == true`. Note whether all layers are local
  (`m.LayerIsLocalResident(l)` true for all) — Mistral is; report if not.
- **Phi-3-mini**: `m.PartialRotary()` — TRUE iff `partial_rotary_factor < 1`; if so,
  `len(m.RopeInvFreq()) == rotaryDim/2 < headDim/2`. If the checkpoint is full-rotary
  (`factor==1`), `PartialRotary()==false` and it's just a plain dense model (still a useful
  positive control). `HasQKNorm()==false`, no window, `DecodeRunnerEligible()==true`.

Report the actual values. **Any mismatch here is a real bug** (either the decoder's GGUF/config
parsing or my accessor) that must be fixed before the Mac run means anything.

### 2. Recommend concrete checkpoints for the Mac to validate with
The Mac has only qwen2.5-coder + gemma-4. Tell me the smallest solid **GGUF q4_k_m** checkpoints
for each family that goinfer's GGUF loader handles (loadable at `Quant:"int8int8"`), e.g. a
Qwen3-0.6B/1.7B, a small Mistral, Phi-3-mini. For each: the exact HF repo/file, and confirm from
its `config.json` the expected flag values from §1 (so the Mac knows what "correct" looks like).
If you can drop them somewhere the Mac pulls from, even better; otherwise just the download list.

### 3. (Optional) sanity-check my accessors
Skim `decoder/residency.go` `HasQKNorm` / `PartialRotary` / `EmbedScaleResident` (added this
session) and `metal/backend.go` `metalUnsupported` — flag anything that looks wrong vs how the
arch is actually populated for these families (I couldn't test the accessors without the models).

## Then the Mac runs (for reference — this part is mine)
Once you confirm detection + point me at checkpoints, on the Mac:
```
GOINFER_METAL_MODEL=<qwen3.gguf> CGO_ENABLED=0 go test ./metal -run 'TestRealModel_parityAndThroughput|TestPrefillParity' -v
go run -tags metal ./demo/gemma --backend metal --quant int8int8 --model <qwen3.gguf> --prompt "…" --max 40
```
Pass = argmax parity vs the CPU reference in the same band as qwen2.5 (~21/24), cosine ≈ 0.98+,
and coherent generation. I'll report results back.

**Deliverable:** the §1 flag values per family (pass/fail + any decoder bug found), the §2
checkpoint list, and any §3 findings.
