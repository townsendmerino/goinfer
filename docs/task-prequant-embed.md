# Task (goinfer): build-time pre-quantized weights (skip dequant+requant)

> **For:** Claude Code, in `~/tmcode/goinfer`.
> **Why:** even after in-memory loading (#1), every launch still
> dequantizes the q4 GGUF → f32 → re-quantizes to int8 (~0.6 s), *and* inflates
> ~477 MB of zstd. Bake the already-int8 resident weights into the binary and the
> launch skips both. This is the exact analog of ken's pre-built index
> (ADR-024); **mirror `ken/internal/search/index_serialize.go`** for the format
> discipline (magic, version, config guard, CRC, lazy fallback).

## The key realization (why this is worth it)

The current embed asset `model.gguf.zst` is **477 MB** — *larger* than the ~430 MB
q4 GGUF, because q4 is high-entropy and doesn't compress. So zstd is buying us
nothing on size while costing inflate time on every launch.

Therefore: embed the pre-quantized int8 weights **uncompressed**, and **alias**
them at load (the `//go:embed` `[]byte` is already page-mapped from the binary's
data segment). That skips inflate *and* dequant *and* requant — startup becomes
"parse header + set up slice headers," i.e. near-instant — while the binary stays
about the same size (~500 MB int8 vs ~477 MB zstd-q4). This is the big win;
prioritize the uncompressed-alias path.

## Step 0 — inventory before you write anything

Read `decoder/weights.go` (the `Weights` struct), the per-`Layer` struct, and
`weightMat` (`decoder/weightmat.go`), plus the arch descriptor / `Config`. List
**every field** the forward pass reads — all `weightMat`s (q/k/v/o proj, gate/up/
down, etc.), every norm (`[]float32`), token embeddings, the LM head (note the
**tied vs untied** case — `gguf.go` has "else tied to the embedding"), rope
params, and the scalar arch fields. The serializer must capture this complete
set; a missing tensor = wrong output, not a crash. Build the format from the
actual struct, not from this doc's examples.

## 1. Serializer / deserializer (decoder)

Add a versioned binary format for a quantized `*Weights` bundle. Public API:

```go
// SerializeWeights writes the resident weight bundle (already quantized to the
// Options.Quant precision) to a flat little-endian blob. Intended to be produced
// once at build time and embedded.
func SerializeWeights(w *Weights) ([]byte, error)

// LoadSerializedWeights reconstructs a *Weights from a blob produced by
// SerializeWeights, WITHOUT any dequant/requant. Big int8 weight arrays are
// ALIASED into data (zero-copy); small per-row scale floats are copied (see
// alignment note). Returns a typed error on any magic/version/quant/arch/CRC
// mismatch so the caller can fall back to building from the GGUF.
func LoadSerializedWeights(data []byte) (*Weights, error)
```

Format (mirror ken's `index_serialize.go`):
- **Magic** (e.g. `"GINFW"`) + **uint32 format version**. Bump the version on any
  layout change.
- **Header**: the `Config` / arch descriptor, the quant mode, and a **model
  identity** field (hash of the source GGUF, or name+version) so a blob can be
  matched to its model.
- **Tensors**: for each `weightMat`, write kind (f32/q8/q4) + `rows`, `cols`,
  `group`, `w8a8`, the scale array, and the raw weight bytes. Include the
  non-matmul tensors too (norms, token embeddings, tied/untied LM head, rope
  params) — everything `Weights` needs to run.
- **CRC32** over the payload; verify on load.
- **Little-endian explicitly.** All targets (amd64/arm64) are LE, but write/read
  LE explicitly so the format is portable and self-describing.

### Aliasing & alignment (the zero-copy detail)
- Alias the big `[]int8` (and `[]byte` int4) weight arrays directly over `data`
  (`unsafe` []byte→[]int8 reslice is safe — same element size, read-only at
  inference). This is what makes load near-instant.
- **Do NOT alias `[]float32`** out of `data` — the embedded bytes aren't
  guaranteed 4-byte aligned. The scale arrays are tiny (per-row), so **copy**
  them into fresh `[]float32`. Negligible cost, avoids unaligned-read UB.
- `data` (the embedded slice) must stay alive for the model's lifetime — document
  it; the demo holds the embed var, so it does.

### Lazy fallback (non-negotiable, like ken)
Any mismatch — wrong magic, older/newer version, different quant mode, arch
mismatch, CRC failure — returns an error, **never panics**. The caller logs a
stderr note and falls back to building from the GGUF. A stale/corrupt blob yields
a slower-but-correct launch, not a crash.

## 2. Build-time generation

Add a tiny generator (a `cmd/prequant` or a flag on the existing flow):
- Load the GGUF with `Quant: "int8int8"`, call `SerializeWeights`, write
  `demo/chat/model.giw` (goinfer weights). Default **uncompressed**; allow
  `--zstd` if someone wants the smaller-but-slower variant.
- Extend `build-embed.sh`: if a `.giw` is present (or generate it from the GGUF
  arg), embed it; else fall back to embedding the GGUF as today.
- `.gitignore`: add `*.giw` (build input, per-machine, not committed — like the
  GGUF/zst).

## 3. Demo wiring (prefer prequant, fall back to GGUF)

- New build path: `//go:embed model.giw` → `decoder.LoadSerializedWeights(bytes)`.
- If the `.giw` is absent at build time, or `LoadSerializedWeights` returns a
  mismatch error at runtime, **fall back** to the existing embedded-GGUF +
  `LoadGGUFBytes` path. Keep the GGUF path as the safety net; never hard-depend on
  the blob.
- Update the `progress(...)` lines: prequant path prints `mapping weights…`
  (fast) vs the GGUF path's `decompressing… / loading + quantizing…`.

## 4. Tests

- **Round-trip parity (the important one):** load a small fixture GGUF with
  `int8int8` → `SerializeWeights` → `LoadSerializedWeights` → assert the resident
  arrays are **byte-identical** to the live-requant model, and that a fixed
  prompt+seed yields **identical token ids / logits**. (Skips without the model
  asset, like the other GGUF tests.)
- **Guards:** corrupt magic, bump version, wrong quant mode, flipped CRC byte →
  each returns an error, none panic; and the demo-level fallback still produces a
  working model.
- **Aliasing safety:** confirm the int8 arrays alias `data` (mutating `data`
  would change them — just assert identity/length, don't actually mutate) and the
  scale floats are copies (independent of `data`).

## 5. Honest cost / benefit (put in the CHANGELOG entry)

- **Win:** skips inflate + dequant + requant → near-instant cold start (header
  parse + slice setup). Best perceived-speed improvement available.
- **Cost:** a versioned binary format to maintain (bump the version on any
  weightMat/layout change — the guard turns a mismatch into a safe rebuild, not a
  crash). Asset ~500 MB uncompressed int8 vs ~477 MB zstd-q4 — about the same,
  and well under GitHub's 2 GiB cap.
- **Scope:** quant is fixed at build time (int8int8). `--quant` at runtime only
  applies on the GGUF fallback path; document that.

## 6. Definition of done

- [ ] `SerializeWeights` / `LoadSerializedWeights` with magic+version+quant+arch
      guard + CRC; lazy fallback on every mismatch; LE; big arrays aliased, scales
      copied.
- [ ] Generator emits `model.giw`; `build-embed.sh` embeds it (uncompressed
      default), GGUF fallback intact; `*.giw` gitignored.
- [ ] Demo prefers the prequant blob, falls back to `LoadGGUFBytes` on absent or
      mismatched blob; progress lines updated.
- [ ] Round-trip parity test (byte-identical weights + identical logits vs live
      requant); guard tests; aliasing test. All skip cleanly without the asset.
- [ ] `gofmt` / `go vet` / `go test ./...` green; measured cold start recorded in
      `docs/demo-plan.md` and the README banner updated.
- [ ] Reference checked: format discipline matches
      `ken/internal/search/index_serialize.go` (magic/version/guard/CRC/fallback).
```
