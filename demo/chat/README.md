# goinfer chat — an entire LLM in one file

A local coding assistant that's **one static binary**. Download a single file,
run it, and you're chatting with a code model — **offline, no install, no Python,
no cgo, no `ollama pull`, no internet.** The model is *baked into the executable*.

It's built with plain `go build`, so the same source cross-compiles to macOS,
Linux, and Windows (Intel + ARM) from one machine — no per-platform toolchain,
no native libraries. That's the whole point of goinfer: a pure-Go LLM runtime you
can `scp`.

```
$ ./goinfer-chat-0.5b-darwin-arm64
mapping weights…
loaded 24-layer model (hidden 896, vocab 151936) in 0.48s [backend=cpu quant=int8int8]
goinfer chat — /help for commands, /quit to exit.
you> write a Go function that reverses a slice of ints in place
```

The embedded model is **Qwen2.5-Coder-Instruct** (run with the fast int8×int8
kernel) — interactive on a laptop CPU, no GPU required.

---

## Download & run

Two size tiers — same program, pick one. Grab the binary for your platform from
the [latest release](https://github.com/townsendmerino/goinfer/releases/latest):

Speed figures below are measured, not estimated: Apple M1 Pro, int8int8,
`DefaultDecodeParallelThreshold`, median of 5 runs, 2026-08-22 @ `3a27941`
(`docs/measurements/demo-chat-macbook-2026-08-22.md`). **A desktop CPU is not faster here** — a
Ryzen 7 3700X measures roughly half on the identical harness, 28.1 / 12.1
(`docs/measurements/demo-chat-incumbent-2026-08-22.md`), so treat these as Apple Silicon numbers
rather than a floor.

**0.5B — the headline: tiny + fast** (~617 MB, **~52 tok/s** on an M1 Pro; the five runs
spanned 46–58, so treat it as a band rather than a point):

| Platform | Asset |
|---|---|
| macOS (Apple Silicon) | `goinfer-chat-0.5b-darwin-arm64` |
| macOS (Intel) | `goinfer-chat-0.5b-darwin-amd64` |
| Linux (x86-64) | `goinfer-chat-0.5b-linux-amd64` |
| Linux (ARM64) | `goinfer-chat-0.5b-linux-arm64` |
| Windows (x86-64) | `goinfer-chat-0.5b-windows-amd64.exe` |

**1.5B — bigger, smarter, still one file** (~1.7 GB, **~28 tok/s**, runs within 27.6–28.2 —
slower but noticeably more capable):

| Platform | Asset |
|---|---|
| macOS (Apple Silicon) | `goinfer-chat-1.5b-darwin-arm64` |
| macOS (Intel) | `goinfer-chat-1.5b-darwin-amd64` |
| Linux (x86-64) | `goinfer-chat-1.5b-linux-amd64` |
| Linux (ARM64) | `goinfer-chat-1.5b-linux-arm64` |
| Windows (x86-64) | `goinfer-chat-1.5b-windows-amd64.exe` |

The tradeoff: the 1.5B is ~1.1 GB more to download and ~2× slower per token, for
better code and reasoning. Both load in ~1 s and use < 100 MB of heap (the
weights are mapped from the binary image). Start with the 0.5B; reach for the
1.5B when you want more capability.

```bash
chmod +x goinfer-chat-0.5b-darwin-arm64
./goinfer-chat-0.5b-darwin-arm64
```

**macOS:** the binary is unsigned, so Gatekeeper will block it on first run.
Clear the quarantine flag once:

```bash
xattr -dr com.apple.quarantine ./goinfer-chat-0.5b-darwin-arm64
```

(or right-click → Open the first time). That's it — there's nothing else to
install.

---

## Commands

Type a message to chat. Lines starting with `/` are commands:

```
/system <text>   set the system prompt (steers the model), resets history
/temp <f>        sampling temperature (0 = greedy / deterministic)
/max <n>         max tokens per reply         /json    toggle JSON-only output
/reset           clear history               /params  show current settings
/demos           list canned demo prompts    /demo <n|name>   run one
/help            full list                   /quit    exit
```

### Canned demos (`/demo`)

So you don't have to type prompts (handy for a quick try or a screen recording),
`/demos` lists ready-made ones and `/demo <name>` runs them. **Every demo runs on
either binary** — the `[1.5B]` ones just look better on the bigger model.

Great on the **0.5B** (short, verifiable):

| name | shows |
|---|---|
| `bug` | spot-and-fix a classic Go off-by-one |
| `dedup` | complete a generic `Dedup[T comparable]` function |
| `mutex` | a thread-safe counter with `sync.Mutex` |
| `reverse` | reverse a slice in place |
| `fim` | fill in a function body from its doc comment |
| `range` | refactor a C-style loop to `range` |
| `json` / `extract` | **guaranteed-valid JSON** extraction (constrained decoding) |

Show off the **1.5B** (`[1.5B]` in `/demos` — multi-step, longer):

| name | shows |
|---|---|
| `race` | find + fix a goroutine loop-variable data race |
| `lru` | implement an O(1) LRU cache |
| `pool` | a worker pool with a jobs channel + `sync.WaitGroup` |
| `test` | `IsBalanced` + a table-driven test for it |
| `niltl` | explain nil vs empty slice (a correct conceptual answer) |
| `wrap` | error wrapping with `%w` + `errors.Is` |

The `json` / `extract` ones are the party trick: with constrained decoding on,
the output *cannot* be malformed JSON — the grammar masks the logits — even from
the 0.5B.

---

## Why this is cool

- **One file, zero dependencies.** The runtime *and* the model are a single
  static binary. No Python, no llama.cpp, no shared libraries, no model download.
- **No cgo → trivial cross-compilation.** `GOOS=windows GOARCH=arm64 go build`
  produces a working LLM. Try that with a C++ inference stack.
- **Runs anywhere Go runs**, including locked-down or air-gapped machines with no
  compiler and no admin rights. Drop the binary, run it. It loads the model
  straight from memory, so it even runs on a **read-only filesystem** (e.g. a
  `FROM scratch` container) — no temp file, no writable disk.
- **Constrained decoding** means structured output you can actually trust from a
  tiny model — a feature, not a wrapper around hope.
- **Embeddable.** This whole demo is ~150 lines wrapping `goinfer/decoder`. The
  same packages drop into *your* Go program — an LLM as a library call, not a
  subprocess or a server.

---

## Build it yourself

**From source, with your own GGUF** (any goinfer-supported model — Qwen, Gemma,
Llama, Mistral, …):

```bash
go run ./demo/chat --model ~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf
```

`--quant int8int8` is the default: it runs the weights through the native
int8×int8 (W8A8) SDOT kernel, which is **much** faster than the int4 path's
per-token nibble unpacking. Other flags: `--system`, `--temp`, `--top-k`,
`--top-p`, `--max`, `--seed`, `--backend`.

**The single-file embedded binary** (what the releases are): bake the model into
a static, no-cgo, cross-compiled binary.

```bash
# 0.5B tier (default name goinfer-chat; --name sets the tier so tiers don't clobber):
./demo/chat/build-embed.sh --name goinfer-chat-0.5b \
    ~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf \
    darwin/arm64 darwin/amd64 linux/amd64 linux/arm64 windows/amd64
# 1.5B tier:
./demo/chat/build-embed.sh --name goinfer-chat-1.5b \
    ~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
    darwin/arm64 darwin/amd64 linux/amd64 linux/arm64 windows/amd64
# → demo/chat/dist/<name>-<os>-<arch>
```

The model is a build input (gitignored — it exceeds GitHub's 100 MB file cap),
so you supply its path; the resulting binaries are the release assets.

By default this builds a **prequant bundle** (`-tags prequant`): the int8 weights
are pre-serialized at build time, so the binary **maps them straight from its own
image at launch** — no decompress, no dequant/requant, no weight heap copy.
Versus loading the raw GGUF, that's a ~5× faster cold start (≈0.5 s vs ≈2.3 s on
the 0.5B) and ~10× less resident heap (≈78 MB vs ≈772 MB) — and the RAM gap grows
with model size. Pass `--gguf` to bake the raw GGUF instead (smaller asset,
slower start, full weight heap) if asset size matters more than RAM/speed.

**Memory vs. disk.** The prequant binary needs no writable disk and barely any
heap for weights (they're image-mapped). The GGUF (`--gguf`) build loads into RAM
by default, also writing nothing; on a RAM-constrained machine `--model-tmp` (or
`GOINFER_MODEL_TMP=1`) streams that model to a temp file + mmap instead — lower
peak RAM, but needs a writable temp dir. Caveat: a **tmpfs** temp dir (RAM-backed,
common on Linux) saves no RAM. (The prequant build ignores `--model-tmp`; it's
already image-mapped.)

---

## Honest notes

- **It's a 0.5B model.** It's good at short, concrete code tasks — completions,
  fixes, small functions, structured extraction. It is *not* a chat genius; ask
  it to write a function, not to explain distributed consensus. The canned demos
  are tuned to its strengths.
- **It's not a speed record.** goinfer's pure-Go kernels won't out-throughput
  llama.cpp's hand-tuned Metal/AVX. The point here is *portability and
  embeddability* — one dependency-free file that runs everywhere — not tokens/sec.
- **GPU doesn't help this build.** The quantized path is CPU-only; `-tags gpu`
  (WebGPU) only accelerates unquantized models. The demo is CPU by design.
