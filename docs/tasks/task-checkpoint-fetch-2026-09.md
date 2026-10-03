# Task: fetch any checkpoint goinfer can load — beyond one GGUF file (P1–P9) — 2026-09

> **Status: IN PROGRESS 2026-10-03.** §2 decided (option (c), owner 2026-10-03). P1, P2, P3 and P5 are
> built in `pull` and reachable from the CLI, `--model` and the web UI (P6); P9's gates pass and the server
> docs name the checkpoint path. P7's vision half is fixed. P4 (split GGUF) is built: route (b), owner 2026-10-03.
> P7's embedding half and P8's cache view are built. Every P item is done. See "Progress" below.
>
> Filed after the owner asked for the complete solution: every supported model reachable from the
> page, multi-file checkpoints downloadable, and loadable once down.
>
> **The gap in one number: 14 of the 40 families in `docs/capability-matrix.json` have no GGUF
> loader at all** (12 of 36 when this was scoped; `glm_ocr` and `spark2_5` have joined since). `pull` searches, lists and fetches GGUF only, so those twelve are invisible to
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

## Progress

**§2: option (c), owner 2026-10-03.** Anonymous only for now. A gated original is declined by `CheckAccess`
before the tree or any file is read, and the HF token becomes its own item.

**Built 2026-10-03 (`pull/checkpoint.go`; the CLI in `internal/pullcmd`; `Resolve` in `pull/resolve.go`):**
- **The selector `:safetensors`** names a repo's safetensors checkpoint (`pull owner/repo:safetensors`,
  `--model hf:owner/repo:safetensors`). No GGUF quant has that name, so it cannot shadow a quant selector. A bare
  `pull owner/repo` still lists the repo, and when it has no GGUF files it lists the checkpoint plan instead of
  "0 GGUF files".
- **P1:** `searchKinds` gains `safetensors`. The web UI's kind selector is P6's part.
- **P2, `PlanCheckpoint`:** reads the tree, `config.json` and the shard index. It plans `config.json`, the
  tokenizer, chat-template, generation and processor files present, and either `model.safetensors` or the index
  plus every shard it names. It leaves out READMEs, legacy `.bin` files and GGUFs. It refuses:
  - a repo without a config or safetensors weights;
  - an index naming a file the repo lacks;
  - an unsafe path, since a tree listing is remote input.
- **P5:** `config.json`'s `model_type` is looked up in the embedded capability matrix, and a type with no
  safetensors loader is refused after reading `config.json` and before any weight file. `Plan.SizeNote` states the
  download size and, for a bf16/f16/f32 original, that a GGUF q4 would be about a quarter of it (§3 rule 4).
- **P3, `DownloadCheckpoint`:** builds the set in `<dest>.partial`, each file through `Download`, so each gets its
  own `.part`, digest check and resume. Once every file has verified it writes a marker naming the set, then makes
  one rename to `<dest>`. A complete `<dest>` is a no-op, and a re-run resumes in the staging directory, skipping
  verified files. A `<dest>` that exists without a matching marker is refused, not overwritten.
- **Offline:** `Resolve` returns a complete cached checkpoint from its marker with no network. This is P8's
  detection half.

**P9's gates, `pull/checkpoint_test.go`, against a fake HuggingFace:**
- `llama-tiny` split into two shards with an index goes plan, fetch, `decoder.Load`, one token, and a second
  `Resolve` makes no request.
- A transfer cut inside shard 2 publishes nothing at the final path, keeps the staging directory, and the re-run
  does not re-fetch shard 1.
- An unknown `model_type` is refused with only `config.json` read.
- A gated repo is refused with no file read.
- The bf16 size note fires.
- Bad indexes and paths, and a foreign directory at the destination, are refused.
- Mutations fail the right gates: building the set at the destination fails the round-trip and the interrupted
  gate, and wiping the staging directory at start fails the resume.

**By day against real HuggingFace:**
- `pull HuggingFaceTB/SmolLM3-3B` lists 9 files, 5.7 GiB, with the bf16 note.
- `google/gemma-3-4b-it` is refused as gated.
- A GGUF repo's listing is unchanged.
- `pull HuggingFaceTB/SmolLM2-135M-Instruct:safetensors` fetched 8 files (259.8 MiB) in 13 s with no staging left
  behind, and `serve --model <dir> --backend cpu` answered an 8-token chat request from it.

**P6, built 2026-10-03 (`internal/serveapp/webui.go`, the page's `app.js`):**
- **List:** a repo that lists no GGUF, or one asked for as a checkpoint by the kind selector, gets its checkpoint
  plan in the list response. The plan carries its size line and the family that loads it. A plan that declines puts
  its reason where the offer would be. A GGUF repo lists exactly as before and makes no plan request.
  - The plan has **no fit tag**. Its bytes are the full-precision original, which goinfer quantizes at load, so a
    size-against-free-memory read would say "won't fit" for models that fit. The load's fit guard answers that
    instead.
- **Pull:** `{repo, checkpoint: true}` streams the set through `DownloadCheckpoint`. Progress is over the whole set
  and names the file in flight. `done` says how many files carried a sha256: HF publishes one only for the LFS
  weights, so the small config and tokenizer files are checked by size, and the page says so rather than calling
  the set verified.
- **Load:** `webLoadPath` accepts a directory under the cache root only when `CachedCheckpoint` verifies it. It
  refuses a directory with no marker, a `.partial` staging directory (by name, too, for the instant it carries
  the marker before the rename) and one whose weights no longer verify. The served name is the directory's, which is
  the repo's. `filepath.Ext` would have cut `Qwen2.5-0.5B-Instruct` to `Qwen2.5-0`, so only `.gguf` is cut.
- **The fit guard's decline** reaches the page for a directory in the same shape as for a file: an error event
  carrying the guard's message, status 400.
- **The kind selector** (GGUF file / safetensors checkpoint) narrows the search and the list.
- **Gates** (`internal/serveapp/webui_checkpoint_test.go`):
  - list → pull → load with the real loader, then one decoded token;
  - the GGUF-repo and declined-plan listings;
  - the directory confinement cases;
  - the directory fit decline.
  The HF calls are package seams, like `webLoadDecoder`. Mutations caught: dropping the staging refusal, and naming
  by `filepath.Ext`.
- **Live by day:**
  - The page's routes ran on a scratch `serve -web`, with HOME in the scratchpad so the real cache was untouched.
  - The SmolLM2-135M-Instruct plan listed 8 files, 259.8 MiB, with the bf16 note.
  - The pull took 15 s and left no staging directory.
  - The load came up under `SmolLM2-135M-Instruct`, and a chat answered.
  - The gated repo was refused, the GGUF repo was unchanged, and a `safetensors` search returned SmolLM3-3B first.

**P7's vision half, fixed 2026-10-03.** The doc's "pulling a VL repo whole enables image turns" was true only for a
plain directory path. Vision auto-discovery stat-ed the typed `--model` string, so `--model hf:…:safetensors` found no
tower and served the model text-only, with nothing said. `loadDecoder` now keeps the resolved source on the loaded
model, and discovery looks there. `TestLoadVisionTower_discoversInResolvedSource` was red before the fix (nil: no
tower tried) and is green after (the resolved directory is tried and named).
- A page-loaded checkpoint is text-only by design, because the tower is attached once, at startup.
- `goinfer-chat --image` needs a GLM-OCR directory path and refuses an `hf:` reference loudly, before any load.

**P4, split GGUF: route (b), owner 2026-10-03.** The decision rested on three facts:
- **A split quant is almost always a model over ~50 GB.** Uploaders split mainly to get under HuggingFace's 50 GB
  per-file limit. In `bartowski/Llama-3.3-70B-Instruct-GGUF`, every single-file quant is 45 GiB or less, and the
  split ones sit in subfolders that the old top-level listing never showed.
- **Route (a)'s condition can't be tested.** "Prove the merge equals the upstream single-file build" has no
  reference: a quant is split because no single-file build of it exists. llama.cpp's own merge doesn't reproduce
  the file it split either. `llama-gguf-split --merge` of a split 0.5B differs from the original from byte 17, in
  the header, because it keeps the split keys.
- **Route (b) touches less than the scoping feared.** No load ever runs a model from GGUF bytes: the transcoder
  reads them once into a `.giw` sidecar, or `-direct-load` reads them into the heap. So teaching the readers the
  shard set leaves the code that runs the model unchanged.

Built:
- **aikit `embed.OpenGGUFSplitMmap`:** maps every shard, takes the first shard's metadata, and unions the tensor
  directories. Each tensor reads from its own shard's data section. It refuses a missing, misordered or duplicated
  shard, or a tensor count short of `split.tensors.count`.
- **`decoder.OpenGGUFMmap`:** every GGUF open site in decoder and prequant goes through it. Given a first shard it
  opens the set, it names a missing shard, and it refuses a later shard by naming the first.
  - `GGUFFileBytes` sizes the whole set for the fit guard's mapped-source term and the sidecar disk check. Shard 1
    alone would undercount on exactly the machine where that is dangerous.
  - Sidecar freshness checks every shard.
- **`pull`:**
  - The listing is recursive, and `Collapse` shows a split set as one row: its first shard, the set's size and its
    shard count.
  - `SelectSet` turns a quant, or any shard's exact name, into the whole set in order. It refuses a set the listing
    holds only part of. `Select` keeps its one-file contract.
  - `DownloadSet` fetches the set shard by shard, each resumable and digest-checked, and returns the first shard.
  - `Resolve`, the CLI and the web pull all use the set. A split model is served without its shard suffix.

Gates. The fixture is `testdata/gguf-split/`: `glm-tiny.gguf` split into 4 shards by llama.cpp's own
`llama-gguf-split` 0.3.0, committed so CI reads the real tool's format.
- **`.giw` from shards:** the weights transcoded from the shards are byte-identical to those from the single file,
  at int4 and int8int8.
- **Direct load:** logits are bit-identical over 4 positions.
- **Pull:** a subfolder split quant is listed, fetched whole by its quant, and loads and decodes. A transfer cut in
  shard 3 leaves a set the loader refuses by naming shard 3 of 4. The re-run fetches only shard 3 again.
- **Refusals:** a later shard, a missing shard, and an incomplete listing.
- **aikit:** the set reads bit-identically to the single file through `Tensor` and `RowDequantizer`. A bad set is
  refused four ways, and a read after Close errors.
- **Mutations caught:** opening only the first shard, which fails both decoder gates; and dropping a tensor's
  shard section, which fails aikit's.

**P7's embedding half, built 2026-10-03.** `-embed-model` takes an `hf:` reference:
- **`hf:<repo>:safetensors`** is planned with the encoder's own check. That's `pull.PlanCheckpointFor` with serve's
  `embedEncoderLoads`, because an embedding encoder is not a generative family in the capability matrix.
  - aikit's `encoder.Load` loads a NomicBert and checks no `model_type` itself. Anything else is refused after
    `config.json` and before any weight, rather than failing late or loading a foreign checkpoint's tensors under the
    wrong architecture.
  - The plan now also takes sentence-transformers' module files, `modules.json` and `1_Pooling/config.json`. The
    encoder reads its pooling from them; without them it falls back to its default.
- **`hf:<repo>:<quant>`** resolves as a GGUF, the decoder-as-embedder.
- **Gates:** `pull`'s `TestPlanCheckpointFor_embeddingEncoder`, and serve's `TestEmbedModel_hfReferenceIsResolvedFirst`
  and `TestEmbedEncoderLoads`.
- **Live by day:** `serve -embed-model hf:nomic-ai/CodeRankEmbed:safetensors` fetched 522 MiB with the CLS pooling
  config and loaded "CodeRankEmbed". Embeddings came back 768-wide at unit norm, and a restart resolved from the cache
  in 1 s.
- **Not done:** the `pull` CLI still plans with the generative check, so `pull nomic-ai/CodeRankEmbed:safetensors`
  declines it. Serve fetches it itself.

**P8, the cache view, built 2026-10-03.** The detection half was already there: a complete checkpoint resolves
offline from its marker, and a complete GGUF from its size and digest. This adds a way to see what the cache holds.
- **`pull.CacheEntries`** walks `<cache>/<owner>/<repo>` and returns one entry per model:
  - a GGUF;
  - a split set (its first shard, the set's size, its shard count);
  - a checkpoint directory.

  Each entry carries its size, the bytes of its `.giw` sidecars (the transcoded copy a load runs from), and whether its
  pull finished. The unfinished cases are a `.part`, a split set missing a shard, a checkpoint still in staging, or a
  checkpoint whose files no longer match its marker. It reads sizes only, so it stays fast on a large cache; a load
  still verifies in full. Digest sidecars, verified markers and the checkpoint marker are not listed as models.
- **`goinfer-chat cache`** prints the list with each model's `--model` path.
- **The Models tab's "On disk" card** (`GET /web/models/cache`) lists the same entries. An entry that's loaded says
  under which name, matched through each loaded model's resolved source path. An incomplete one says to re-run its
  pull. Any other entry gets a Load button, the same confined route as the pull flow's own.
- **Gates:**
  - `TestCacheEntries`: every shape above, with sidecar attribution and the non-model files ignored.
  - `TestCacheEntries_noCache`.
  - `TestWebCache_listsWhatIsOnDiskAndWhatIsLoaded`: the loaded mark found through a symlinked source. Dropping the
    symlink resolution fails it.
- **Live by day:**
  - `goinfer-chat cache` listed this Mac's real cache: three GGUFs, 6.5 GiB.
  - A scratch `serve -web` listed the SmolLM2 checkpoint as complete and not loaded, loaded it from the card's path,
    and then listed it as loaded.

**Remaining:** nothing in P1 to P9.
- **Out of scope by the doc:** LoRA adapters, GPTQ/AWQ, and uploading or converting.
- **Its own item, by §2 (c):** the HF token.
- **Small known limits, recorded above:** a VL checkpoint loaded from the page serves text only; `chat --image` needs a
  directory; `-embed` refuses a split set; the `pull` CLI does not fetch an encoder checkpoint.

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
