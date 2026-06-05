# goinfer chat — an entire LLM in one file

A local coding assistant that's **one static binary**. Download a single file,
run it, and you're chatting with a code model — **offline, no install, no Python,
no cgo, no `ollama pull`, no internet.** The model is *baked into the executable*.

It's built with plain `go build`, so the same source cross-compiles to macOS,
Linux, and Windows (Intel + ARM) from one machine — no per-platform toolchain,
no native libraries. That's the whole point of goinfer: a pure-Go LLM runtime you
can `scp`.

```
$ ./goinfer-chat-darwin-arm64
loaded 24-layer model (hidden 896, vocab 151936) in 2.1s [backend=cpu quant=int8int8]
goinfer chat — /help for commands, /quit to exit.
you> write a Go function that reverses a slice of ints in place
```

The embedded model is **Qwen2.5-Coder-0.5B-Instruct** (Q4_K_M), run with the fast
int8×int8 kernel — interactive on a laptop CPU, no GPU required.

---

## Download & run

Grab the binary for your platform from the [latest release](https://github.com/townsendmerino/goinfer/releases/latest):

| Platform | Asset |
|---|---|
| macOS (Apple Silicon) | `goinfer-chat-darwin-arm64` |
| macOS (Intel) | `goinfer-chat-darwin-amd64` |
| Linux (x86-64) | `goinfer-chat-linux-amd64` |
| Linux (ARM64) | `goinfer-chat-linux-arm64` |
| Windows (x86-64) | `goinfer-chat-windows-amd64.exe` |

```bash
chmod +x goinfer-chat-darwin-arm64
./goinfer-chat-darwin-arm64
```

**macOS:** the binary is unsigned, so Gatekeeper will block it on first run.
Clear the quarantine flag once:

```bash
xattr -dr com.apple.quarantine ./goinfer-chat-darwin-arm64
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
`/demos` lists ready-made ones and `/demo <name>` runs them:

| name | shows |
|---|---|
| `bug` | spot-and-fix a classic Go off-by-one |
| `dedup` | complete a generic `Dedup[T comparable]` function |
| `mutex` | a thread-safe counter with `sync.Mutex` |
| `reverse` | reverse a slice in place |
| `fim` | fill in a function body from its doc comment |
| `range` | refactor a C-style loop to `range` |
| `json` | **guaranteed-valid JSON** extraction (constrained decoding) |

The `json` one is the party trick: with constrained decoding on, the output
*cannot* be malformed JSON — the grammar masks the logits — even from a 0.5B
model.

---

## Why this is cool

- **One file, zero dependencies.** The runtime *and* the model are a single
  static binary. No Python, no llama.cpp, no shared libraries, no model download.
- **No cgo → trivial cross-compilation.** `GOOS=windows GOARCH=arm64 go build`
  produces a working LLM. Try that with a C++ inference stack.
- **Runs anywhere Go runs**, including locked-down or air-gapped machines with no
  compiler and no admin rights. Drop the binary, run it.
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

**The single-file embedded binary** (what the releases are): compress a GGUF,
bake it in with `//go:embed`, and cross-compile static, no-cgo binaries:

```bash
./demo/chat/build-embed.sh ~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf \
    darwin/arm64 darwin/amd64 linux/amd64 linux/arm64 windows/amd64
# → demo/chat/dist/goinfer-chat-<os>-<arch>
```

The model file is a build input (gitignored — at ~430 MB it exceeds GitHub's
100 MB file cap), so you supply its path; the resulting binaries are the release
assets.

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
