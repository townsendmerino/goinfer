# Task: fetch any checkpoint goinfer can load — beyond one GGUF file (P1–P9) — 2026-09

> **Status: SCOPED 2026-09-17, unstarted.** Filed after the owner asked for the complete solution:
> every supported model reachable from the page, multi-file checkpoints downloadable, and loadable
> once down.
>
> **The gap in one number: 12 of the 36 families in `docs/capability-matrix.json` have no GGUF
> loader at all.** `pull` searches, lists and fetches GGUF only, so those twelve are invisible to
> the web UI and to `--model hf:…` — not deprioritised, *unreachable*. Both vision-language
> families are among them.
>
> Siblings: [`task-web-ui-2026-09.md`](task-web-ui-2026-09.md) W33 (the fit tag on a file row) and
> W5/W32 (load and unload from the page) are the surfaces this feeds;
> `docs/completed/task-model-pull.md` is the closed record of the single-file flow this extends
> without replacing.

---

## 1. What is unreachable today, and why

**Twelve families, safetensors-only** (`loaders` column, `docs/capability-matrix.json`):
`olmo_hybrid`, `qwen3_next`, `bailing_hybrid`, `cohere`, `cohere2`, `internlm2`, `lfm2`,
`mistral3`, `olmo3`, `qwen2_5_vl`, `qwen3_vl`, `smollm3`. Twenty-three more load from either
format, and one adds GPTQ/AWQ.

Three separate walls produce that:

1. **Search is GGUF-only.** `searchKinds` has exactly one entry, and `Search` errors on any other
   kind — deliberately, with a comment saying `gguf` is "the only one this build's pull flow
   actually loads today".
2. **`List` filters to `.gguf`.** A safetensors repo lists as empty.
3. **`Download` moves one file.** A safetensors checkpoint is `config.json` +
   `tokenizer.json` + either `model.safetensors` or `model.safetensors.index.json` plus N shards.

And a fourth, separate from the twelve: **split GGUF is refused on purpose.** `Select` detects
llama.cpp's `-00001-of-00003.gguf` naming and returns `shardedError` — "pulling split checkpoints
is not supported yet" — offering the nearest single-file quant instead. That is the correct
behaviour for a fetcher that cannot assemble a set, and it is what P4 replaces.

**The good news, and it is most of the work:** *the loader already handles this.*
`decoder.Load` takes a directory, `openCheckpointMmap` reads `model.safetensors.index.json` when
present and falls back to `model.safetensors`, and the fit guard treats sharded and single
checkpoints identically. The cache layout is already a directory per repo —
`<user cache>/goinfer/models/<owner>/<repo>`. So a multi-file checkpoint has somewhere to land and
something to open it. **The missing half is entirely in `pull`.**

## 2. The decision that gates the value — read before scoping the rest

**Gated repos.** `pull` is anonymous-only, and `CheckAccess`'s own comment explains why that has
been survivable: GGUF *re-uploads* are usually ungated even when the original is not
(`bartowski/…-GGUF` is open where `google/gemma-3-4b-it` is `gated="manual"`).

That property does not transfer. The safetensors-only families are reached through **original**
repos, and the originals are exactly the ones that gate. Shipping P1–P3 without an answer here
produces a page that finds twelve new families and then refuses to download several of them.

Options, to be decided before P1:

- **(a) Anonymous only, decline clearly.** Cheapest, and honest. Some of the twelve stay
  unreachable; the message already names why.
- **(b) Optional HF token.** `HF_TOKEN`/`--hf-token`, sent only to `huggingface.co`, never logged,
  never written to the cache. Unlocks the gated originals. Adds a credential to a product that has
  none today, and the web UI would want a field for it — which is a different security
  conversation from W5's load gate, and should not be smuggled in as part of a download feature.
- **(c) (a) now, (b) as its own item.** Recommended: it makes P1–P3 shippable and keeps the
  credential decision separate and visible.

## 3. Ground rules

1. **One fetcher.** The CLI, `--model hf:…`, and the web UI go through the same `pull` code, as
   they do today. No second implementation for directories.
2. **Nothing lands half-finished.** Today's guarantee is per-file: `.part`, digest-verified, renamed
   only on success, so a corrupted pull can never leave something loadable-looking. A multi-file
   checkpoint must extend that to the *set* — a directory is either complete or visibly not.
3. **Resume stays.** Per-file resume exists; a set that dies at file 9 of 12 resumes at file 9.
4. **The size difference must be stated before the transfer, not after.** A GGUF q4 of a 7B is
   ~4 GB; the safetensors original is bf16, ~15 GB, and goinfer quantizes at load. Users pulling a
   safetensors repo are downloading 3–4× what they would for the same model as GGUF, to run it at
   the same quant. W33's fit tag prices *resident* memory; this is *disk and bandwidth*, and it is a
   different number the row must show.
5. **Refusing is a feature.** A repo whose architecture this build cannot load should be declined
   before the bytes move, not after.

## 4. The items

### P1 — repo kinds beyond GGUF
Extend `searchKinds` with a safetensors/transformers kind, keeping the existing explicit error for
unknown kinds. The web UI's repo search gains a kind selector or searches both and labels each hit.
The seam is already there; this is the smallest item and unlocks discovery.

### P2 — list a repo as a *plan*, not a file list
`List` returns `[]File` filtered to `.gguf`. Replace with a checkpoint **plan**: the set of files
this build would need, classified, plus a total. Concretely, per repo kind:

- **GGUF, single:** the one file (today's behaviour, unchanged).
- **GGUF, split:** the whole shard group (P4).
- **safetensors:** `config.json`, the tokenizer files, and either `model.safetensors` or
  `model.safetensors.index.json` + every shard it names. Optional extras when present:
  `generation_config.json`, and a vision tower's files for the VL families.

The plan is what the page renders, what the fit tag prices, and what the downloader executes.

### P3 — download a set
`Download` takes one `File`; add a set-level fetch with one aggregate progress (bytes and files),
per-file resume, and **atomic completion**: assemble under a staging directory and publish the final
directory only when every file has verified. Keep the existing single-flight — one pull at a time is
still right when each is multi-gigabyte.

### P4 — split GGUF: decide, then build
Two routes, and the choice is not obvious:

- **Fetch and merge** into one `.gguf` at rest. Simple downstream, doubles peak disk during the
  merge, and re-derives what `llama-gguf-split` already does.
- **Teach the loader the shard set.** No merge, no extra disk, but it touches the GGUF reader,
  which is parity-gated territory.

Lean: fetch-and-merge first *only if* the merge can be proven byte-equivalent to the upstream
single-file build of the same quant; otherwise the loader route. Either way, `shardedError` and its
"nearest single-file quant" hint stay until the replacement is real.

### P5 — "will this actually load?"
The matrix has `loaders` and `modality` per family, and a safetensors repo's `config.json` carries
`model_type`. So before any transfer the page can say **supported / not supported by this build /
supported but not on this backend**, from data already in the tree. Pair it with W33's fit tag: one
answers *will it fit*, this answers *will it load at all*, and a user needs both to click with
confidence.

### P6 — load a directory from the page
`webLoadPath` confines loads under the pull cache root and resolves symlinks; `handleWebLoad`
derives the served name from the file's basename. Both need to accept a **directory** and name it
from the repo. Check the fit guard's decline path renders as well for a directory as it does for a
file.

### P7 — what comes free, and what does not
- **Vision towers.** `-vision` already defaults to the `--model` directory when it contains a
  tower, so pulling a VL repo whole enables image turns with no further work. State this as an
  expected consequence, and test it, rather than discovering it.
- **Embedding models.** `-embed-model` takes an HF dir; P2/P3 make those pullable too. A small
  addition to the plan kinds, worth naming now.
- **LoRA adapters.** `-lora` takes a PEFT dir. Same shape, and explicitly *not* in scope here —
  it needs its own decision about pairing an adapter with a base.

### P8 — the cache, and what is already on disk
Layout is already `<cache>/goinfer/models/<owner>/<repo>`, which fits a directory checkpoint
without change. Add: a resumable staging area that survives a restart, detection of an
already-complete checkpoint (so a second pull is a no-op rather than a re-download), and a way to
see what the cache holds — which W32's "what is resident" view is the natural neighbour of, since
*on disk* and *loaded* are different questions a user will conflate.

### P9 — docs and gates
- `docs/server.md`, `README.md`, and the `--model` help all describe a one-GGUF-file world.
- **Gates:** a safetensors repo with a shard index round-trips (plan → fetch → load → one token);
  an interrupted multi-file pull resumes without re-fetching completed files; an incomplete set
  never publishes a loadable directory (kill the process mid-set and assert the final path is
  absent); an unsupported architecture is declined **before** the first byte; and the size warning
  fires on a bf16 repo. The mid-set kill is the one that matters most — it is the guarantee
  today's `.part`-then-rename gives per file, and the set must not lose it.

## 5. Not in scope, stated

- **GPTQ/AWQ.** One matrix row mentions them; loading them is a separate question from fetching.
- **An HF token**, unless §2 picks (b) — and then it is its own item, not a sub-task of downloading.
- **Uploading or converting.** `prequant`/`.giw` already own conversion.
- **A model registry beyond `curated.json`**, whose own comment says it is deliberately tiny and
  not a name registry to grow.

## Sources

`pull/pull.go` (`searchKinds` and `Search`'s single kind; `List`'s `.gguf` filter; `Select`,
`multiPart`, `shardedError` and the split-GGUF refusal; `Download`'s single-file `.part`-then-rename
guarantee; `CacheRoot`/`CacheDir`'s per-repo directory; `CheckAccess`'s note on GGUF re-uploads
being ungated where originals are not) · `decoder/weights.go` (`shardIndexFile`,
`openCheckpointMmap` — sharded safetensors already load) · `decoder/model.go` (`Load` takes a
directory) · `decoder/fitguard.go` (sharded and single checkpoints priced identically) ·
`internal/serveapp/webui.go` (`webLoadPath`, `handleWebLoad` — file-shaped today) ·
`docs/capability-matrix.json` (the `loaders` and `modality` columns this doc counts) ·
`docs/completed/task-model-pull.md` (the single-file flow) ·
[`task-web-ui-2026-09.md`](task-web-ui-2026-09.md) W5, W32, W33 ·
[`task-fit-to-hardware.md`](task-fit-to-hardware.md) (the fit verdict this pairs with)

<!-- doc-reviewed: 2026-09-17 -->
