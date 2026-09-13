# MacBook task: the demo/chat numbers only Apple Silicon can settle — and Gemma 4 E2B as the Tier 2 candidate

> **ARCHIVED — a record, not instructions.** This file is closed work kept for its reasoning and
> its numbers. Checkboxes record the state at the moment it was archived: an unticked box means
> "not ticked when this closed", **not** "still to do", and nothing in `docs/completed/` is
> actionable. If you need a task, use the live docs; if something here reads as an instruction to
> a future reader, it was missed at archival — see the doc-closeout rule in
> `docs/parity-coverage-policy.md`, and move it to live policy or strike it.

> **STATUS 2026-09-13 — all three items closed; run and reported the same week it was written.**
> - **Item 1 (demo/chat speed provenance) — DONE.** Measured on an Apple M1 Pro
>   (`docs/measurements/demo-chat-macbook-2026-08-22.md`, commit `3a27941`): 0.5B median 52.3 tok/s
>   (band 46–58), 1.5B median 27.8 tok/s (band 27.6–28.2) — on/near the README's prior unattributed
>   ~57/~26 claims. `demo/chat/README.md` now states the measured medians and bands with full
>   provenance (box, quant, threshold, commit, date) rather than the old bare numbers, landed in
>   `b8fde587` and reconciled to one figure per tier in `e26e1e9a`; the size claims (~617 MB / ~1.7 GB)
>   were also verified against real `build-embed.sh` output in the same pass.
> - **Item 2 (Gemma 4 E2B as Tier 2 candidate) — attempted, blocked, then formally killed.** The
>   downloaded safetensors checkpoint hit a real loader gap: `num_kv_shared_layers` (Gemma 4 E-model
>   KV-layer sharing) was wired for the GGUF loader but not the safetensors path, so it failed before
>   any number could be measured (`docs/measurements/demo-chat-gemma4e2b-blocked-2026-08-22.md`). The
>   follow-up brief `docs/prompts/mac-demo-finish.md` closed the question outright rather than
>   leaving it open: *"Tier 2. Dead twice over: Qwen3.5-0.8B on the 248 K-vocab decode penalty, Gemma 4
>   E2B on the `num_kv_shared_layers` safetensors loader gap you found. The incumbent Qwen2.5 tier
>   stays."* (The safetensors loader has since gained `SharedKVLayers` handling for other reasons —
>   `decoder/weights.go` around the `arch.gemma4.SharedKVLayers` check — but Tier 2 was not revisited;
>   that decision stands.)
> - **Item 3 (does `vhs` render on the Mac) — confirmed yes.** The follow-up brief's own header states
>   "You are now the only box that can render tapes (vhs works there, not here)", and the refreshed
>   demo GIF (`a8ba9fd5`) was in fact rendered on this Mac. The Linux `ttyd`/`vhs` root cause was never
>   found and was explicitly marked not worth further time — "the practical answer is that tapes get
>   rendered on the Mac now."
>
> Full follow-on work (GIF attribution fix, size-table verification, gpt2 gate floor) is its own
> brief, `docs/prompts/mac-demo-finish.md` — its archival status was not evaluated here; this
> file's own three items are what closed.

> **Why this must be the Mac.** Two open items from the demo refresh both reduce to "measure it on
> Apple Silicon", and the Linux box cannot answer either. See `docs/task-demo-refresh.md` and the two
> measurement records it cites.

## Item 1 — `demo/chat`'s advertised speeds are unattributable, and I could not fix that from here

`demo/chat/README.md` claims **~57 tok/s (0.5B)** and **~26 tok/s (1.5B)** "on a laptop CPU", with no
commit, box, quant or config. I re-measured the incumbent on the Linux box
(`docs/measurements/demo-chat-incumbent-2026-08-22.md`):

| tier | Ryzen 7 3700X, int8int8 | README claims |
|---|---|---|
| 0.5B | **28.1 tok/s** | ~57 |
| 1.5B | **12.1 tok/s** | ~12.1 vs ~26 |

Five runs each, spread under 2%. **This is not evidence the README is wrong** — the likeliest benign
explanation is that those were Apple Silicon numbers, which plausibly do reach that range and are a
reasonable thing to put on a laptop-facing page. I deliberately did NOT substitute my desktop figures
into the README: that trades one misleading number for another. Only a measured Apple Silicon run can
resolve it.

**Run exactly this**, so the number is comparable rather than merely new:

```sh
GOINFER_PREQUANT_GGUF=~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf \
  go test ./decoder -run '^$' -bench '^BenchmarkDecode$' -benchtime 200x -count 5
GOINFER_PREQUANT_GGUF=~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
  go test ./decoder -run '^$' -bench '^BenchmarkDecode$' -benchtime 200x -count 5
```

`BenchmarkDecode` reports `tok/s` directly, defaults to **int8int8** (what `build-embed.sh`'s prequant
path bakes), and uses `DefaultDecodeParallelThreshold` — the shipped decode config. Do not substitute
a different harness; the point is that both boxes ran the same one.

**Then update the README with provenance**, per `benchmarks.md`'s rule: box, quant, commit. If Apple
Silicon lands near 57/26, the claims were right all along and only their attribution was missing —
say so. If it lands near the Ryzen figures, the claims are simply stale and should move.

## Item 2 — Gemma 4 E2B is the better Tier 2 candidate, and the reason is measurable

Tier 2's original candidate, **Qwen3.5-0.8B, was killed by gate 4 on the Linux box**: 13.5 tok/s
against the incumbent 0.5B's 28.1 — 2.08× slower, barely ahead of the *1.5B* tier
(`docs/measurements/demo-chat-tier2-gates-2026-08-22.md`). Gates 1 and 2 passed cleanly (apache-2.0;
loads through the existing dense `qwen3_5` adapter with **no loader work** — the multimodal wrapper,
`text_config` nesting and 3:1 DeltaNet hybrid are all already handled).

**The cause is the vocabulary, not DeltaNet**, which is what makes Gemma 4 interesting:

| | hidden | vocab | LM-head params |
|---|---|---|---|
| qwen2.5-coder-0.5B | 896 | 151,936 | 136.1 M |
| Qwen3.5-0.8B | 1024 | **248,320** | **254.3 M** |

The LM head alone is 1.87× larger against a measured 2.08× slowdown. At this size the head is a third
of the model and is read in full every token. Since that 248 K vocabulary is shared across the whole
Qwen3.5 line, **no member of the family escapes it** — "try the 2B" is not a way out. (Attribution
from parameter counts, not a measured decomposition; a profile would settle the residual.)

Gemma 4 E2B has no such penalty, is dense, and already carries a full-oracle parity row in the
capability matrix. Measure it the same way:

```sh
hf download google/gemma-4-E2B-it --local-dir ~/models/gemma-4-E2B-it   # ungated, no account needed
GOINFER_PREQUANT_GGUF=~/models/gemma-4-E2B-it \
  go test ./decoder -run '^$' -bench '^BenchmarkDecode$' -benchtime 200x -count 5
```

If the asset predicate rejects a directory (it wanted a `.gguf` on my box), copy the tiny harness from
the gate-4 record rather than switching benchmarks.

### One thing to check with human eyes, not a script

`google/gemma-4-E2B-it` is tagged **`license: apache-2.0`** and **`gated=False`**, unlike every Gemma 3
checkpoint (`license: gemma`, `gated=manual`). If that holds, it **retires `task-demo-refresh.md`'s
"Gemma models stay flag-loaded, never embedded" constraint for Gemma 4** — that constraint is a
Gemma 3 fact the doc over-generalises — and Gemma 4 E2B becomes embeddable.

**I did not act on this and nothing embeds Gemma.** The apache-2.0 claim rests on card metadata; there
is no LICENSE file in the repo to corroborate it, and this repo's own rule is that the HF endpoint has
lied before (P10). Embedding is redistribution: please read the model card yourself before anyone puts
those weights in a release binary.

## Item 3 (cheap, optional) — does `vhs` render on the Mac?

Tier 3's drafter tape is blocked here twice over: no runnable v1 DFlash pair on an 8 GB card, and the
render toolchain does not work. I installed ttyd 1.7.7 (upstream static binary, SHA256-verified) and
reinstalled vhs v0.11.0; **a render still never completes** — ttyd exits immediately, vhs waits
forever at 0 CPU, headless Chrome spawns and idles. ttyd serves fine standalone with vhs's exact
argument list, so the two are individually healthy and jointly broken. Root cause not established.

If `vhs demo/mellum2-gpu.tape`-style rendering works on the Mac, that is worth knowing: it means tapes
can be produced there for any non-CUDA subject, and it isolates the Linux failure to this box.
**Note the tape's build line was wrong** and is fixed as of `ba79c2f` — it had been an impossible
build since v0.10.0.

## Reporting back

Numbers with provenance (box, quant, commit), and for item 2 a plain verdict: does Gemma 4 E2B clear
the "tiny + fast" bar the 0.5B sets on that machine? Declining any item is fine — say which and why.
