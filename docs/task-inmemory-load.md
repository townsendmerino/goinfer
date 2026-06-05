# Task (goinfer): load the embedded model from memory, fall back to /tmp

> **For:** Claude Code, in `~/tmcode/goinfer` (with `~/tmcode/aikit` in the
> workspace via `go.work`).
> **Why:** the single-file demo currently inflates its ~400 MB embedded model to
> a temp `.gguf` and mmaps it back — a wasted 400 MB disk write + reread on every
> launch, and it requires a writable temp dir (breaks read-only / `FROM scratch`).
> Loading from the inflated bytes makes cold start snappier AND lets the binary
> run with no writable disk. Step 2 of 2; depends on aikit's `OpenGGUFBytes`
> (`aikit/docs/task-gguf-from-bytes.md`, aikit ≥ v0.4.2).

## 1. decoder: add an in-memory GGUF load entry point

Today `decoder.Load(dir, opts)` → `loadWeights(dir, quant)` → (for GGUF)
`embed.OpenGGUFMmap(path)`. Refactor so the GGUF build core takes an already-open
`*embed.GGUFFile`, and add a bytes entry point:

```go
// LoadGGUFBytes loads a GGUF model from an in-memory slice (e.g. an inflated
// //go:embed asset). Same result as Load on the equivalent file, but nothing
// touches the filesystem. raw is retained until the model is built.
func LoadGGUFBytes(raw []byte, opts Options) (*Model, error)
```

Implementation notes:
- Extract the shared core: `buildModelFromGGUF(g *embed.GGUFFile, opts Options)
  (*Model, error)` used by both the path-based GGUF branch of `Load` and
  `LoadGGUFBytes`. `LoadGGUFBytes` calls `embed.OpenGGUFBytes(raw)` then the core.
- **EOS / special tokens:** the path-based code resolves EOS via
  `resolveEOSIDs(dir, &Cfg)`. For the bytes path there is no dir — resolve EOS
  from the **GGUF metadata** already parsed into `Cfg` (the GGUF carries
  `tokenizer.ggml.eos_token_id` etc.). Make the shared core resolve from the
  GGUF object, not a filesystem path, so both routes agree. Verify the demo's
  stop token (`<|im_end|>`) still resolves.
- Quant handling (`int8int8` default etc.) is unchanged — it's already applied in
  the weight-build path, independent of how the GGUF was opened.
- Keep the existing `Load(dir, opts)` signature and behavior intact.

## 2. demo/chat: inflate to memory, with a /tmp fallback

`embed.go` (`-tags embed`) currently writes a temp file. Replace with:

- `materializeEmbedded()` inflates `modelZst` into a `[]byte` (cap the
  `io.Copy`/decode into a `bytes.Buffer` or stream-decode to a preallocated
  slice). Return the bytes.
- `main.go`: when embedded, call `decoder.LoadGGUFBytes(raw, opts)` instead of
  writing a file. Keep the non-embed path (`--model <path>`) on `decoder.Load`.

### The low-RAM fallback (default in-memory, opt-out to disk)

Rationale (do not auto-detect RAM across OSes — fragile, and OOM isn't
catchable in Go): default to in-memory; provide an explicit opt-out that streams
to a temp file + mmap, which has a **lower peak RSS** (source paged from disk,
not held on heap).

- Add `--model-tmp` flag and `GOINFER_MODEL_TMP=1` env (either forces the disk
  path). When set: stream-inflate `modelZst` directly to `os.CreateTemp` (small
  buffer, no 400 MB heap spike), then `decoder.Load(tempPath, opts)`; remove the
  temp file on exit (the existing cleanup pattern).
- **Caveat to document, not solve:** `os.TempDir()` may be a tmpfs (RAM-backed,
  common on Linux), in which case the fallback saves no RAM. State this in
  `--help` text and the README rather than trying to detect it.
- Optional (nice-to-have, Linux-only, no dep): if `/proc/meminfo`'s
  `MemAvailable` is comfortably less than ~2× the model size, auto-pick the disk
  path; on darwin/windows default to in-memory. Keep it behind a single helper so
  it's easy to delete if it misbehaves. **Skip this if it adds any complexity** —
  the flag/env is the contract; auto-detect is a bonus.

For the 0.5B demo this fallback is near-moot (in-memory peak ~900 MB; anything
that can run inference can do it). It matters for large models (Mellum-4B:
~7 GB in-memory peak vs ~5 GB via disk). Implement it cleanly, don't over-invest.

## 3. Progress output (perceived snappiness, free)

During the silent ~2 s, print to **stderr** so it reads as deliberate work:
- `decompressing model…` before inflate,
- `loading + quantizing…` before the weight build,
- keep the existing `loaded N-layer model … in Xs [backend=… quant=…]` line.

(stderr only — never stdout, in case anyone pipes the chat.)

## 4. Docs

- Update `demo/chat/README.md`: note the binary needs **no writable disk** by
  default (loads from memory), and document `--model-tmp` / `GOINFER_MODEL_TMP`
  for RAM-constrained machines + the tmpfs caveat. Update the cold-start line if
  the number changes.
- `CHANGELOG.md`: entry under the next version — "decoder: `LoadGGUFBytes` for
  in-memory GGUF loading; demo loads the embedded model from memory (no temp
  file), `--model-tmp` opt-out."
- `docs/demo-plan.md`: tick the cold-start item; note in-memory load shipped.

## 5. Definition of done

- [ ] aikit `OpenGGUFBytes` available (≥ v0.4.2; via `go.work` locally).
- [ ] `decoder.LoadGGUFBytes` loads a GGUF from `[]byte`; shares the build core
      with `Load`; EOS resolves from GGUF metadata; existing `Load` unchanged.
- [ ] Embedded demo loads from memory by default — **no temp file written**
      (verify: run it with `TMPDIR` pointed at a read-only dir and it still works).
- [ ] `--model-tmp` / `GOINFER_MODEL_TMP=1` streams to a temp file + mmap and
      cleans up; documented with the tmpfs caveat.
- [ ] Progress lines on stderr; stdout stays clean.
- [ ] `gofmt` / `go vet` / `go test ./...` green; demo runs and is interactive.
- [ ] After merge: bump goinfer `go.mod` to `aikit v0.4.2`, re-tidy, retest.
