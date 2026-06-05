# goinfer launch demo — plan

> Working doc for the demo(s) that ship before any social-media push.
> Captures decisions made so far + open questions. Not user-facing.

## Strategic framing (the rule everything follows)

**Compete on distribution and embeddability, not on tokens/sec.** goinfer's
pure-Go matmul will lose a raw speed race to llama.cpp's Metal/AVX kernels — so
never frame the demo as a benchmark. The moat is: no cgo, no Python, no
per-platform toolchain, no model-download step, and it runs *inside* a Go
program. The demo must make someone say *"wait, it's one file and it runs
anywhere?"* — not *"is it faster?"*

## Decisions locked

- **Headline model: Qwen2.5-Coder-0.5B-Instruct.** Small enough to embed,
  code-flavored (on-brand with ken/aikit), and Instruct-tuned so it can actually
  respond. It's the smallest in the Qwen2.5-Coder family.
- **Quant: Q4_K_M (~380–400 MB).** Verified goinfer decodes Q4_K_M. This is the
  quality floor for a 0.5B — do NOT go lower. The model is *embedding-bound*
  (151,936-token vocab → the token-embedding tensor is ~25% of params and is the
  tensor K-quants keep at higher precision), so Q3/Q2 save little and degrade
  fast. Q4_K_M is the size/quality sweet spot.
- **Distribution: GitHub release assets, not committed files.** Release assets
  allow **2 GiB/file, 1000 files/release**, free storage on public repos. A
  ~400 MB per-platform binary is well under the cap. (Committed git files are
  blocked at 100 MB, so the fat binary cannot live in the repo.)
- **Don't over-quantize for MB.** "400 MB" is already "one downloadable file";
  the narrative isn't better at 250 MB, and the coherence loss is real. If
  400 MB ever feels too big, drop to Gemma 3 270M — don't crush the 0.5B.

## Demo structure (TBD: one or two — user deciding)

### Demo 1 — headline: "an entire LLM in one file" (broad reach)
`//go:embed` the Q4_K_M Qwen-Coder-0.5B into the binary so the executable *is*
the model + runtime. User downloads one file (`goinfer-chat-darwin-arm64`),
runs it, gets an offline code assistant — no install, no `ollama pull`, no
internet, no libstdc++. The kicker: all five platform binaries
(darwin/linux/windows × amd64/arm64) built from one machine with plain
`GOOS=… GOARCH=… go build`, no cross-toolchain.

- **Frame around short code completions / structured tasks**, not open chat — a
  0.5B fumbles long conversation. Script the prompts so it looks good.
- **Lean on `constrain`**: tiny model + JSON grammar = reliably valid structured
  output despite the size. Turns the size weakness into a feature showcase.

### Demo 2 — chaser: "pure-Go JetBrains Mellum, no cgo" (credibility / dev-tooling)
Mellum-4B is a FIM *code-completion* model (not chat). Running JetBrains'
production completion model in pure Go with zero native deps is a sharper, more
novel flex for the dev-tooling crowd. At 4B int4 (~2.5 GB) it's **too big for a
release asset (>2 GiB cap) and for the "one file" pitch** — so this is a
`FROM scratch` **container** / completion-backend demo, hosted or built locally,
not a download. Use it as the serious follow-up to Demo 1's accessible hook.

> **Open question (user thinking):** ship one demo or both? Recommendation:
> Demo 1 as the opener (widest "whoa", lowest friction), Demo 2 as the chaser
> (novelty + credibility). They're the same artifact shown two ways.

## Runtime speed — run W8A8, not int4 (important)

The demo loads a **Q4_K_M** GGUF (small asset) but **runs `--quant int8int8`**
(W8A8) — now the default in `demo/chat`. Why: the int4 matmul kernel
(`MatmulBTQ4`) scalar-unpacks 4-bit nibbles to f32 for every neuron on every
token before the dot; the int8 path (`MatmulBTW8A8`) feeds int8 weights straight
into a native int8×int8 **SDOT** SIMD kernel with no unpack. Measured on the
embedded binary (Qwen2.5-Coder-0.5B, M-series, CPU): **int4 ≈ 3.4 tok/s (a crawl)
→ int8int8 ≈ 42.6 tok/s** — a ~12× jump from interactive to snappy, same binary.
Same q4-level quality (the
weights are dequantized from the q4 file and re-quantized to int8 at load), ~+1–2 s
startup, ~0.5 GB resident RAM instead of ~0.25 GB. All fine on a laptop.

Notes:
- **The embedded asset stays the ~400 MB Q4_K_M GGUF** — int8int8 is a *runtime*
  setting, so disk/binary size is unchanged. (Alternative: ship a native Q8_0
  GGUF ~530 MB for slightly better quality; not needed.)
- **GPU (`-tags gpu`) does NOT help** — quantized matmul paths are CPU-only
  (`weightmat.go`: the backend only substitutes for the f32 path). Metal only
  accelerates an unquantized model. Don't promise GPU speedup for the demo.
- The kernel is already multi-threaded (`parallelCols`), so cores aren't the
  bottleneck; the nibble-unpack was.
- **Future goinfer optimization (not demo-day):** give int4 an SDOT-style kernel
  (SIMD-unpack nibbles to int8, or carry an int8 shadow) so it keeps int4's size
  win at int8 speed.

## Build specifics

- **Strip:** `go build -ldflags="-s -w" -trimpath`. Free hygiene, but the binary
  is model-dominated (~400 MB model vs ~10 MB Go code+runtime), so stripping is a
  rounding error here. The only lever that moves a model-embedded binary is quant.
- **Optional `//go:embed` + zstd:** store the model asset zstd-compressed,
  inflate at startup. Int4 weights are high-entropy (~10% gain) but the
  higher-precision embedding tensor compresses better → maybe shave 30–50 MB off
  the *download*, at the cost of a second or two of cold start + transient RAM.
  Wire as a build flag if we want the asset under ~360 MB.
- **Optional build-tag trim:** compile only the Qwen architecture (tag out
  Gemma/Llama/Mistral/etc.) and exclude the `gpu` path — single-digit MB, low
  priority.
- **Skip UPX:** high-entropy model payload → negligible gain + antivirus false
  positives.
- **Cross-compile matrix:** darwin/{amd64,arm64}, linux/{amd64,arm64},
  windows/amd64 from one machine, CGO_ENABLED=0. Goes in a `Makefile` /
  goreleaser config; outputs become release assets.

## One-two punch for the social thread

1. Download one file → instant offline code assistant (Demo 1). Screenshot-friendly.
2. `FROM scratch` container running JetBrains Mellum, pure Go, no cgo (Demo 2).
   "My LLM container has no OS."

## Not yet decided / to revisit

- One demo vs two (user deciding).
- Whether to also do a WASM-in-browser stretch demo (flashy but perf/memory dicey
  — prove the binary story first).
- Exact scripted prompts for Demo 1 (must flatter a 0.5B).
- zstd-embed: in or out for v1 of the demo.
