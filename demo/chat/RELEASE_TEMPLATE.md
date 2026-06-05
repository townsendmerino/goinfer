# Release checklist + body for the chat demo

## How to tag (read first)

`demo/chat` is a **package in the main module**, not a submodule — so do **NOT**
create a `demo/vX.Y.Z` tag (Go would read that as a submodule version, and there's
no `go.mod` under `demo/`). Instead:

1. Cut a normal module tag, e.g. `v0.1.3` (or whatever's next).
2. Build **both tiers** for all five platforms and attach them as **release
   assets** (10 files: 0.5B ~617 MB, 1.5B ~1.7 GB — both under GitHub's
   2 GiB/asset cap; release-asset storage is free on public repos):

   ```bash
   ./demo/chat/build-embed.sh --name goinfer-chat-0.5b \
       ~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf \
       darwin/arm64 darwin/amd64 linux/amd64 linux/arm64 windows/amd64
   ./demo/chat/build-embed.sh --name goinfer-chat-1.5b \
       ~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
       darwin/arm64 darwin/amd64 linux/amd64 linux/arm64 windows/amd64
   # then create the release and upload both tiers' assets, e.g.:
   ( cd demo/chat/dist && shasum -a 256 goinfer-chat-* > checksums.txt )
   gh release create v0.1.3 demo/chat/dist/* \
       --title "goinfer v0.1.3 — chat demo (0.5B + 1.5B)" \
       --notes-file demo/chat/RELEASE_TEMPLATE.md
   ```

---

## ⬇️ Paste-ready release body (everything below the line)

---

## An entire LLM in one file 🤯

`goinfer-chat` is a local coding assistant that's **a single static binary** —
the runtime *and* the model baked into one file. Download it, run it, and you're
chatting with a code model **offline: no install, no Python, no cgo, no
`ollama pull`, no internet.**

Built with plain `go build`, so it cross-compiles to every platform below from
one machine — no per-platform toolchain, no native libraries.

### Download — two sizes, same program

**0.5B** — tiny + fast (~617 MB, ~57 tok/s): `goinfer-chat-0.5b-<platform>`
**1.5B** — bigger, smarter (~1.7 GB, ~26 tok/s): `goinfer-chat-1.5b-<platform>`

| Platform | 0.5B | 1.5B |
|---|---|---|
| macOS · Apple Silicon | `goinfer-chat-0.5b-darwin-arm64` | `goinfer-chat-1.5b-darwin-arm64` |
| macOS · Intel | `goinfer-chat-0.5b-darwin-amd64` | `goinfer-chat-1.5b-darwin-amd64` |
| Linux · x86-64 | `goinfer-chat-0.5b-linux-amd64` | `goinfer-chat-1.5b-linux-amd64` |
| Linux · ARM64 | `goinfer-chat-0.5b-linux-arm64` | `goinfer-chat-1.5b-linux-arm64` |
| Windows · x86-64 | `goinfer-chat-0.5b-windows-amd64.exe` | `goinfer-chat-1.5b-windows-amd64.exe` |

Start with the 0.5B; grab the 1.5B for more capability.

```bash
chmod +x goinfer-chat-0.5b-darwin-arm64
# macOS only, clears the Gatekeeper quarantine on the unsigned binary:
xattr -dr com.apple.quarantine ./goinfer-chat-0.5b-darwin-arm64
./goinfer-chat-0.5b-darwin-arm64
```

Then type a message, or `/demos` to try canned prompts. `/demo json` shows
**guaranteed-valid JSON** even from the 0.5B via constrained decoding.

### What's inside

- Model: **Qwen2.5-Coder-Instruct** (0.5B or 1.5B), run on the fast int8×int8 CPU
  kernel — interactive on a laptop, no GPU. The int8 weights are mapped straight
  from the binary image, so cold start is ~1 s and heap stays < 100 MB.
- Runtime: [goinfer](https://github.com/townsendmerino/goinfer) — a pure-Go,
  no-cgo decoder-only LLM runtime with HuggingFace logit parity.

### Honest notes

These are 0.5B/1.5B models: great at short code tasks (completions, fixes,
structured extraction), not chat geniuses. And it's not a speed record — goinfer
trades llama.cpp's raw throughput for *portability*: one dependency-free file
that runs anywhere Go runs. That's the trick.

Full demo docs: [`demo/chat/README.md`](https://github.com/townsendmerino/goinfer/blob/main/demo/chat/README.md).
