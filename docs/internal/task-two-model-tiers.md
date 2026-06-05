# Task (goinfer): ship the chat demo in two model-size tiers

> **For:** Claude Code, in `~/tmcode/goinfer`.
> **Goal:** offer the single-file chat demo in **two sizes** — the existing
> **0.5B** (the "entire LLM in one file" headline: tiny, fast) and a larger
> **Qwen2.5-Coder-1.5B-Instruct** (noticeably smarter, still one file). Same
> program, built twice with different embedded models. The prequant/in-memory
> work already done is what makes the 1.5B viable on RAM.
>
> **Scope note:** this is *two size tiers of `demo/chat`*, NOT the Mellum-4B
> `FROM scratch` container (that remains a separate future demo in
> `docs/demo-plan.md`). Don't conflate them in the docs.

## The constraints (already reasoned out — verify, don't re-derive)

- **GitHub release-asset cap is 2 GiB/file.** int8 ≈ ~1 byte/param, so
  Qwen2.5-Coder-1.5B int8 ≈ **~1.6 GB** — fits with headroom. (3B ≈ ~3.2 GB would
  NOT fit; that's why 1.5B is the ceiling for a single-asset int8 model.)
- **RAM is not the limit** post-prequant: the `.giw` int8 weights are mapped from
  the binary image (~78 MB resident heap on the 0.5B), so the 1.5B's larger
  weights don't blow up per-launch heap.
- **Speed is the real question.** 1.5B is ~3× the compute of 0.5B, so pure-Go CPU
  tok/s drops ~3×. Whether that's still pleasant is the one thing to *measure*,
  not assume (see the gate below).

## Model

- **Qwen2.5-Coder-1.5B-Instruct** (GGUF, Q4_K_M source → built at `int8int8`,
  same as the 0.5B). Same `qwen2` architecture as the 0.5B, so it should load and
  run through the existing path with no decoder changes — **confirm** with a load
  + generate + `.giw` round-trip, don't assume.
- (Alternative, only if 1.5B-Coder disappoints: Qwen3-1.7B — newer, ~1.8 GB int8,
  less code-specialized. Stick with 1.5B-Coder unless there's a reason.)

## 1. Verify the 1.5B runs end-to-end

```bash
go run ./demo/chat --model ~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
```
- Confirm it loads (banner shows the bigger hidden/layers), generates coherently,
  the canned `/demo` prompts work, and `/demo json` still yields valid JSON.
- Build a `.giw` for it via `cmd/prequant` and confirm the round-trip parity test
  passes for this model too (byte-identical weights + identical greedy token).

## 2. Parameterize the build so the two tiers don't clobber

`build-embed.sh` currently emits `goinfer-chat-<os>-<arch>` from a single
embedded `model.giw`. Make the output basename a parameter so two models produce
distinct binaries:

- Add a `--name <basename>` flag (default `goinfer-chat`). Output becomes
  `<basename>-<os>-<arch>[.exe]`.
- Keep the generate-`.giw` → `go build -tags prequant` flow; it already
  regenerates the embedded bundle per invocation, so running the script twice
  (once per model, different `--name`) is clobber-free.
- Suggested names: `goinfer-chat-0.5b` and `goinfer-chat-1.5b` (put the size in
  *both* so downloads are unambiguous). Update the gitignore'd `dist/` accordingly.
- Optional: a tiny wrapper (or a `--name`+`--model` pair loop) that builds both
  tiers for all five platforms in one command. Keep it simple.

No change to `demo/chat`'s Go code is expected — the model identity already prints
from its own metadata. If you want the banner to name the tier, derive it from the
GGUF metadata, not a hardcoded string.

## 3. Measure both, record the numbers

For each tier (0.5B, 1.5B), on the same machine, capture: **binary size, cold
start, resident heap (`phys_footprint`), and tok/s** (a fixed prompt+seed).
Confirm the **1.5B binary is < 2 GiB**. Record in:
- `docs/demo-plan.md` (a two-tier table),
- `docs/ARCHITECTURE.md` (extend the existing measured table to both models),
- the `demo/chat/README.md` download section (see §4).

## 4. Docs: present two download tiers

- **`demo/chat/README.md`:** keep the **0.5B as the headline** ("an entire LLM in
  one file"). Add the 1.5B as a second tier: "bigger, smarter, still one file —
  ~1.6 GB, slower but more capable." Two rows (or two small tables) in the
  download section, each with all five platform assets. Be honest about the
  tradeoff (download size + tok/s vs quality).
- **`demo/chat/RELEASE_TEMPLATE.md`:** list both tiers' assets (10 files total:
  5 platforms × 2 sizes), and update the `gh release create` example to upload
  both `dist/goinfer-chat-0.5b-*` and `dist/goinfer-chat-1.5b-*`.
- **`CHANGELOG.md`:** one entry — "demo/chat now ships in two tiers (0.5B / 1.5B);
  `build-embed.sh --name` parameterizes the output."

## 5. The gate (decide after measuring)

After step 3, judge the 1.5B's tok/s:
- If it streams readably (roughly ≥ ~6–8 tok/s), ship both tiers.
- If it crawls, **still keep the 0.5B as the headline** and either (a) ship the
  1.5B anyway labelled "slower, for capability" or (b) hold it. Record the call in
  `demo-plan.md`. The 0.5B + portability story carries the launch regardless;
  the 1.5B is upside, not a dependency.

## 6. Definition of done

- [ ] 1.5B loads, generates coherently, `/demo json` valid; `.giw` round-trip
      parity passes for the 1.5B.
- [ ] `build-embed.sh --name` parameterized; both tiers build for all 5 platforms
      without clobbering; 1.5B binary confirmed < 2 GiB.
- [ ] Measured size / cold start / heap / tok/s for both tiers recorded in
      `demo-plan.md` + `ARCHITECTURE.md`; README shows two download tiers.
- [ ] `gofmt` / `go vet` / `go test ./...` green in all build modes.
- [ ] Ship/hold decision on the 1.5B recorded per the §5 gate.
```
