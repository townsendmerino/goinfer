# Release checklist + body for the chat demo

## How to tag (read first)

`demo/chat` is a **package in the main module**, not a submodule — so do **NOT**
create a `demo/vX.Y.Z` tag (Go would read that as a submodule version, and there's
no `go.mod` under `demo/`). Instead:

1. Cut a normal module tag, e.g. `v0.1.2` (or whatever's next).
2. Build the binaries and attach them as **release assets** (each ~400 MB, well
   under GitHub's 2 GiB/asset cap; release-asset storage is free on public repos):

   ```bash
   ./demo/chat/build-embed.sh ~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf \
       darwin/arm64 darwin/amd64 linux/amd64 linux/arm64 windows/amd64
   # then create the release and upload demo/chat/dist/* as assets, e.g.:
   gh release create v0.1.2 demo/chat/dist/* \
       --title "goinfer v0.1.2 — chat demo binaries" \
       --notes-file demo/chat/RELEASE_TEMPLATE.md
   ```

3. (Optional) checksums: `shasum -a 256 demo/chat/dist/* > dist/checksums.txt`
   and attach it too.

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

### Download

| Platform | File |
|---|---|
| macOS · Apple Silicon | `goinfer-chat-darwin-arm64` |
| macOS · Intel | `goinfer-chat-darwin-amd64` |
| Linux · x86-64 | `goinfer-chat-linux-amd64` |
| Linux · ARM64 | `goinfer-chat-linux-arm64` |
| Windows · x86-64 | `goinfer-chat-windows-amd64.exe` |

```bash
chmod +x goinfer-chat-darwin-arm64
# macOS only, clears the Gatekeeper quarantine on the unsigned binary:
xattr -dr com.apple.quarantine ./goinfer-chat-darwin-arm64
./goinfer-chat-darwin-arm64
```

Then type a message, or `/demos` to try canned prompts. `/demo json` shows
**guaranteed-valid JSON** from a 0.5B model via constrained decoding.

### What's inside

- Model: **Qwen2.5-Coder-0.5B-Instruct** (Q4_K_M), run on the fast int8×int8 CPU
  kernel — interactive on a laptop, no GPU.
- Runtime: [goinfer](https://github.com/townsendmerino/goinfer) — a pure-Go,
  no-cgo decoder-only LLM runtime with HuggingFace logit parity.

### Honest notes

It's a 0.5B model: great at short code tasks (completions, fixes, structured
extraction), not a chat genius. And it's not a speed record — goinfer trades
llama.cpp's raw throughput for *portability*: one dependency-free file that runs
anywhere Go runs. That's the trick.

Full demo docs: [`demo/chat/README.md`](https://github.com/townsendmerino/goinfer/blob/main/demo/chat/README.md).
