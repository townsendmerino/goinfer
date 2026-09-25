# Proposal: `pull` converts to `.giw` right after the download, and keeps the `.gguf` only on request (2026-09)

> **Status 2026-09-25: proposal; P0 done, the rest not started.** Written at the owner's request after the question "when goinfer downloads a
> model, can we convert it to `.giw` as it downloads, still verify it's a good file, and have a flag to keep the original
> format?" The facts below were checked against `main` at `fe970d7c`. Four decisions are the owner's (§ Decisions).

## Summary

- **Recommended:** convert right after the download finishes and verifies, then delete the `.gguf` unless `--keep-gguf` is
  passed. At rest, a pulled model takes about half the disk it takes today.
- **Not recommended:** converting while the download is still running. For the two models goinfer ships, the converter
  cannot write anything useful until the whole file has arrived, and the most it could save is 20–100 s of conversion
  overlap.
- **Integrity is not the hard part.** `pull` already hashes every byte as it arrives and checks the digest before keeping
  the file, so the `.giw` would only ever be built from verified bytes.
- **The hard part is that today's `.giw` is found *through* its `.gguf`.** Delete the `.gguf` and the `.giw` can no
  longer be found, or checked for freshness. Most of the work is making the `.giw` the cached artifact in its own right.

## Why do it

Since the sidecar default (darwin 2026-09-22, Linux 2026-09-24), every `.gguf` that chat or serve loads also gets a
`.giw` next to it, and both stay on disk. The int4 `.giw` measured 1.02–1.16× the size of its q4 `.gguf`:

| model | `.giw` / `.gguf` |
|---|---|
| Qwen2.5-Coder 0.5B, int4 | 1.024× |
| Qwen2.5-Coder 1.5B, int4 | 1.158× |
| Qwen2.5 7B, int4 | 1.105× |
| Llama 3.2 1B, int4 (cuda target) | 1.089× |
| Gemma 4 26B-A4B, int4 | 0.959× |
| Qwen2.5-Coder 1.5B, int8int8 | ~1.60× |

So a pulled model now costs roughly 2.0–2.2× its download size. After a load, only the `.giw` is ever read; the `.gguf`
is dead weight until someone asks for a different quant or backend layout.

## What exists today

**The download (`pull.Download`)** is the one path behind `pull` in chat and serve, the web UI's Models tab, and
`--model hf:…` / `demo:…`.
- It is one sequential GET per file, written to `<name>.part` and resumed with an open-ended `Range` request.
- The sha256 is computed while streaming (`io.MultiWriter`).
- The digest is checked before the `.part` is renamed into place. The expected digest is Hugging Face's LFS `oid` for
  `hf:` references, and the pin in `pull/curated.json` for `demo:` tiers.
- A mismatch deletes the `.part` and writes nothing.
- The streamed digest is not saved: the `.<file>.sha256` sidecar is written later, by the first cache check, which re-reads
  the whole file once.
- Nothing happens after the download. The package doc says `pull` deliberately does not convert, and the conversion
  happens on first load.

**The conversion (`prequant.EnsureCachedGIW` → `StreamTranscodeGGUF`)**
- It writes `<base>.<quant>.<target>.giw` next to the `.gguf`, one layer at a time.
  - The quant comes from `--quant` (default `int4`).
  - The target comes from `--backend` via `GIWTargetForBackend`: `cpu` maps to `cpu-amd64` or `cpu-arm64`, and `metal`,
    `cuda` and `webgpu` map to their own names. `cpu-amd64` and `cuda` currently write byte-identical files.
- It writes to `.tmp.giw`, checks the result by loading it (a full CRC read, then a `.giw.verified` marker), and renames
  it into place.
- It mmaps the whole source and fetches tensors by name, so the complete file must exist when it starts.
- **Measured cost:**
  - Ryzen 7 3700X, int4, cpu target: 20.9 s for the 1.5B and 102 s for the 7B
    ([cpu-giw-vs-direct](../measurements/cpu-giw-vs-direct-2026-09-24.md)).
  - Gemma 4 26B-A4B to a Metal int4 `.giw`: 1:50 wall, with 1.6 GB of anonymous memory at peak
    ([transcode-streaming-gemma4](../measurements/transcode-streaming-gemma4-2026-09-24.md)).

**How a `.giw` is found and trusted today**
- It is found by name from its `.gguf`: the `.gguf` path, plus the quant and the target.
- It counts as fresh (`cacheFresh`) if the source `.gguf` exists, the `.giw` is newer than it, and the `.giw` loads. The
  source's mtime is the only thing compared: no size, no hash.
- It records only the source's basename, as its header id, and the reader discards that. There is no provenance hash
  in the format.

**What happens today if the `.gguf` is deleted**
- `--model x.gguf` fails ("sidecar cache: read gguf head…").
- `hf:` and `demo:` references download the `.gguf` again and convert again.
- `fit` fails.
- `--model x.giw` still works.

## Why not convert during the download

The converter writes the model's head first: the embedding, the LM head and `output_norm`. It then writes layers
0…N-1 in numeric order. GGUF data follows the tensor-info order, and the producer chooses that order. The files checked:

| file | head tensors | layer order in the file |
|---|---|---|
| `demo:0.5b` (Qwen2.5-Coder 0.5B q4_k_m) | `output_norm.weight` is the **last** tensor in the file | 0, 1, 10–19, 2, 20–23, 3–9 |
| `demo:1.5b` (Qwen2.5-Coder 1.5B q4_k_m) | `output_norm.weight` last | 0, 1, 10–19, 2, 20–27, 3–9 |
| Qwen2.5 7B q4_k_m | `output.weight` at 74–83% of the file | blk 6, 14 and 22 each split into two non-adjacent pieces |
| Gemma 3 4B, gpt-oss-20b, Qwen3.6-35B-A3B, Gemma 4 26B | head in the first 4–22% | in order |

So for both curated tiers, nothing after the header can be written until 100% of the file is present. Making it work
would take one of two things:
- a converter that buffers out-of-order tensors, up to the whole file in the worst case;
- a `.giw` layout that writes the head last, which is a format change.

Either would buy only the overlap of 20–100 s of conversion with the download. The newer files that are in order would
benefit; the files goinfer ships would not. **Revisit only if** the curated tiers move to in-order GGUFs, *and* a
measured download-plus-convert wall time shows the overlap is worth a format change.

Integrity would be the same either way. The digest is known only at the end of the download, so a `.giw` built during
the download could only be committed after the final check, which is exactly what the recommended design does too.

## The proposal

### What `pull` does

1. **Download and verify as today.** Keep the streamed sha256, and write it to the `.sha256` sidecar at once, so the
   first cache check doesn't re-read the file.
2. **Convert** to `<base>.<quant>.<target>.giw` through the existing `EnsureCachedGIW` path.
   - `pull` gains `--quant` and `--backend`. Their defaults are the same as chat's and serve's (`int4`, `cpu`), so a
     default pull produces the `.giw` a default load looks for.
   - The web UI's pull does the same with the server's own `--quant`/`--backend`.
3. **Record provenance** for the `.giw`: the verified source sha256, the source size and name, the reference (`hf:…` or
   `demo:…`), and the quant and target. Where it lives is Decision 3.
4. **Delete the `.gguf`**, and its `.sha256` sidecar, unless `--keep-gguf` is given. This happens only after the `.giw`
   has passed its load check.

### How a load finds the `.giw` without its `.gguf`

5. `modelload` resolves an `hf:`/`demo:` reference, or a `.gguf` path whose file is gone, to the `.giw` whose provenance
   matches: same source hash or pin, same quant, same target.
   - Freshness for such a `.giw` is its recorded source hash against the reference's expected hash, not an mtime.
   - An `hf:` reference whose repo moved on, where the API's current `oid` differs from the recorded one, is stale, and is
     treated as it is today (download again).
6. **A load that needs a different quant or target** than any `.giw` on disk has no `.gguf` to convert from. What it
   does then is Decision 2.
7. **`fit`** uses the same resolution. (Its narrower gap, that it looked only for a `canonical` `.giw` while chat and
   serve build `cpu-amd64`/`cpu-arm64` ones, was fixed on 2026-09-25: it now tries the sidecars those loads write.)

### Disk

- **At rest:** about 1.0–1.16× the download size for int4, down from about 2.0–2.2×.
- **While converting:** unchanged at `.gguf` plus `.tmp.giw`.
- **The pre-check had to be fixed first, and is (P0).** It refused a conversion only when free space was below the
  source's size, which assumed the `.giw` is never bigger than its source. That is false: int4 runs up to 1.16×, and
  int8int8 about 1.6×. It now prices the output.

## Decisions (owner)

1. **Default: delete or keep?** This proposal says delete, with `--keep-gguf` to keep, because the disk saving is the
   point.
   - The cost of that default: a later `--quant` or `--backend` change means a download again.
   - The alternative is to keep by default, with a `--no-keep-gguf` flag or a separate `goinfer-chat prune` command.
2. **A load that needs a quant or target that isn't on disk:**
   - (a) refuse, naming the `pull --quant … --backend …` or `--keep-gguf` command that fixes it (recommended; no
     surprise download); or
   - (b) download the `.gguf` again and convert (convenient, but multi-GB and silent).
3. **Where provenance lives:**
   - (a) a small `<name>.giw.json` beside the `.giw`, like today's `.giw.verified` marker (recommended; no format bump);
   - (b) a field in the `.giw` header (a format-version bump, and a stale-sidecar rebuild for every existing file); or
   - (c) the tokenizer half's GGUF metadata.
4. **Existing caches:** leave today's `.gguf` + `.giw` pairs alone (recommended), or offer a one-time prune.

## Work, in order

- **P0 — fix the disk pre-check** to price the `.giw`, not the source. **Done 2026-09-25:** `projectedSidecarBytes`
  prices the output from the GGUF header's tensor shapes, 1.03–1.14× the actual size on six real sidecars.
- **P1 — provenance and `.gguf`-free resolution**:
  - write provenance when any sidecar is built;
  - resolve an `hf:`/`demo:` reference or a missing `.gguf` to its `.giw`;
  - key freshness on the recorded hash;
  - `fit` on the same resolution.

  Gate: after deleting a pulled `.gguf`, chat, serve and `fit` all load its `.giw` offline; an `hf:` reference with a
  changed `oid` is detected as stale; a `.giw` whose recorded hash doesn't match is never used.
- **P2 — `pull` converts, with `--quant`, `--backend` and `--keep-gguf`**, in the CLI and the web UI. Gate: a default
  `pull demo:0.5b` leaves one `.giw` and no `.gguf`, and a default `goinfer-chat --model demo:0.5b` then loads without
  converting or downloading.
- **P3 (optional) — `goinfer-chat prune`** for existing pairs, if Decision 4 wants it.

**Out of scope:** split GGUFs and safetensors repositories (`task-checkpoint-fetch-2026-09.md` owns fetching them;
`pull` refuses a split quant today). Converting during the download, for the reasons above.

## Related

- `docs/completed/task-model-pull.md` planned a `pull --prequant` and dropped it as redundant once conversion on first
  load existed. This proposal is about disk, not about when conversion happens.
- `docs/tasks/task-never-swap-2026-09.md` S1 introduced the sidecar default this builds on.
- `docs/giw-bundles.md` documents the `.giw` format and when sidecars are rebuilt.

<!-- doc-reviewed: 2026-09-25 -->
