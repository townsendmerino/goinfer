# MacBook: finish the demo refresh — one correctness issue in the new GIF, plus the last unattributed numbers

> Written 2026-08-22 at `a8ba9fd`. You are now the only box that can render tapes (vhs works there,
> not here), so both items below are yours. Item 1 is a real problem with the refresh you just
> landed; item 2 is small.

## Item 1 — the refreshed GIF shows a linux-amd64 filename next to an Apple Silicon speed

Your commit reasoned that "the recorded terminal UI is host-OS-agnostic, so this substitution
doesn't change what the demo shows". That holds for the UI chrome, but two things on screen are not
host-agnostic:

1. **The typed command is visible.** `demo.tape:21,26` type
   `./goinfer-chat-1.5b-linux-amd64` — unmodified, as you noted — so the recording displays that
   filename while a `darwin-arm64` binary is what actually ran.
2. **The demo prints its own throughput.** `internal/chatapp/main.go:369` writes
   `[%d tok, %.1f tok/s]` after each answer, so a **measured speed is on screen in the GIF**.

Together those make the most-seen artifact in the repo show a **Linux x86 binary name beside an
Apple Silicon throughput**. We established this week that the gap is not cosmetic: the 1.5B measures
**27.8 tok/s on the M1 Pro and 12.1 on a Ryzen 7 3700X** — the identical harness, roughly 2×. A
Linux reader sees a linux-amd64 filename and reasonably expects the number next to it.

That is the same unattributed-number problem `e26e1e9` just fixed in that page's prose, reintroduced
in the image above it. Not your error to have caused — the substitution is documented and the
reasoning was explicit — but the speed line was not part of the reasoning.

**Pick whichever you prefer; any of them closes it:**

- **Rename the built binary to `goinfer-chat-1.5b-darwin-arm64` and adjust the two `Type` lines**,
  then re-render. The GIF then shows what actually ran. Costs a tape edit, which you deliberately
  avoided — if you take this, consider a `demo-darwin.tape` beside the existing one rather than
  changing the linux-targeted script.
- **Keep the render and caption it** where the GIF is embedded (root `README.md`): one line saying
  the recording is an M1 Pro and that a desktop x86 measures roughly half, pointing at
  `docs/measurements/demo-chat-macbook-2026-08-22.md`.
- **Say the throughput is not representative** — weakest, since the number is still the first
  quantitative thing a viewer sees.

My preference is the caption: cheapest, keeps your verified render, and puts the attribution exactly
where the claim is made. But you rendered it and you can see the frames; I cannot.

## Item 2 — the size table is the last unattributed figure on that page

`demo/chat/README.md` advertises **~617 MB** (0.5B) and **~1.7 GB** (1.5B). `task-demo-refresh.md`
says these should come "from the built artifacts, not projections", and nobody has checked them
against a real build. **You just built a 1.5B binary**, so half of this is a `ls -l` away.

If the figures are right, say so with a provenance line like the speed figures now carry. If they
have drifted, correct them. If only the 1.5B is checkable without another build, do that one and
mark the 0.5B as unverified rather than leaving both looking equally solid.

## Not asking you to do

- **The Linux render failure.** You isolated it to that box, which was the useful half. Root cause is
  still unknown there (ttyd exits immediately, vhs waits at 0 CPU, ttyd serves fine standalone with
  vhs's own argument list). Not worth your time; the practical answer is that tapes get rendered on
  the Mac now.
- **Tier 2.** Dead twice over: Qwen3.5-0.8B on the 248 K-vocab decode penalty, Gemma 4 E2B on the
  `num_kv_shared_layers` safetensors loader gap you found. The incumbent Qwen2.5 tier stays.

## Also outstanding from your gpt2 fix, if you have budget

The new gpt2 gate is **vacuous for a real int4 change**. I mutation-checked it here with
`gate mutation`: `int4GroupSize 32 -> 64` moves individual logits by up to **7.39** and drops
centered cosine to **0.99376**, but the floor is **0.99**, so it PASSES — and argmax is unchanged
(4), so that check does not catch it either. Baseline cosine is 0.9995, so the floor sits ~20× looser
than the natural variation.

**A floor of 0.999 passes the baseline and catches that mutation.** What I cannot determine from here
is whether 0.999 still passes on arm64 — that was the whole point of the exemption. **Please report
gpt2's centered cosine on your box.** If it is ≥0.999, raise the floor and the gate works again. If
it is well below, no single floor both tolerates arm64 and catches a 7.4-magnitude change, and the
gate needs a different discriminator (per-arch goldens, or int4-vs-that-box's-own-f32 rather than a
recorded cross-arch golden).
