# Task: knowing what to download, and how long it takes to load

> **Status: BOTH PARTS BUILT 2026-09-06, left uncommitted for review.** Written against a brief
> filed 2026-09-02 that predates `internal/modelpull`, `pull/curated.json`,
> [`quantization.md`](quantization.md) and the fit guard. The brief asked for those to be read and
> reconciled first; §0 is that reconciliation, and it changed the design in two places.

## 0. Reconciliation — three things the brief asks for already existed

**"Take a stance on quantization" — done, earlier the same day.**
[`quantization.md`](quantization.md) states which quants this project stands behind, which it
measured and refused, and where there is no evidence. It also establishes that the peer's rule the
brief cites (*"deliberately avoids Q1–Q3"*) **cannot be copied here**: this project's own
real-model gates for GLM-4.5-Air and Llama-4-Scout run from **Q2_K** checkpoints, because that is
the only form in which those models fit the hardware they were validated on. Declining to
*recommend* a format while still reading it is a different position from refusing to load it, and
it is the one the practice already implements.

**A curated registry already partly ships, and its design explicitly rejects growing it.**
`pull/curated.json` pins `demo:<tier>` models and its own comment reads: *"Deliberately tiny: the
explicit owner/repo:quant form is the real interface, and this is not a name registry to grow."*
It is also gated against `.github/workflows/release-assets.yml`, because those tiers are the models
**embedded in release binaries** — a different question from *which checkpoints do we recommend*.

> **So Part A does not extend it.** The recommendation registry derives from
> `capability-matrix.json` as the brief requires, and `TestRegistry_doesNotCollideWithDemoTiers`
> asserts the two name-spaces stay disjoint. Two lists with two purposes, neither claiming to be
> the other, beats one list answering a question it was explicitly designed not to answer.

**`task-fit-to-hardware.md` Phase 0 landed the same day** as the load-time fit guard. Part A's
"feed the plan function" therefore has nothing to feed yet — `plan` does not exist, only the
refusal. The registry's `needs` field is written to be what that function would consume.

---

## Part B — load time, measured (built first, per the brief's ordering)

**What existed before: nothing.** The only record of load cost anywhere was a prose claim in a
comment — that fanning the GGUF parse across cores turned a 12B's roughly two-minute load into
seconds. A real result with no measurement behind it, and no way for a user to see the number on
their own machine.

`decoder/loadprofile.go` times three phases and both banners print one line:

```
load 4.9s (map 8ms build 4.9s) — 100% build; 2.23 GB source, 0.45 GB/s effective
```

**The split is the product, not the total.** "load 4.9s" is a number to be annoyed by; "100%
build" says a faster disk will not help and a different quant might.

### The result contradicts the brief's expectation

The brief says *"a warm-cache load measures memcpy; a cold one measures the disk"*. On NVMe, it
measures neither — it measures the repack:

| model | source | cold (3 runs) | warm (3 runs) | cold − warm | dominant |
|---|---|---|---|---|---|
| Qwen2.5-Coder-0.5B q4_k_m → int4 | 0.46 GB | 2.80 / 2.75 / 2.81 s | 2.72 / 2.66 / 2.72 s | **+3.3%** | 98% build |
| Phi-3-mini-4k q4 → int4 | 2.23 GB | 5.19 / 5.19 / 5.20 s | 4.96 / 4.96 / 4.99 s | **+4.4%** | 100% build |

`build` — dequantize every tensor and re-quantize/repack it — is **98–100%** of both loads. The
repack is slow enough that the kernel's readahead hides most of the NVMe read behind it. Full
provenance in [`benchmarks.md`](benchmarks.md) Table 4.

### Two limitations, recorded rather than smoothed over

**`map` does not isolate storage, and the measurement is what showed it.** The loader mmaps, so
pages fault in during `build`; `map` is header parse alone. The tell is that the 0.46 GB model
spends *more* time in `map` (46–57 ms) than the 2.23 GB one (6–10 ms) — that phase tracks metadata
count, not bytes. Storage cost appears as the cold−warm delta instead. Isolating it properly needs
a non-mmap read path or per-phase fault accounting, neither worth building while the answer is
"storage is not the problem here".

**The 3–4% delta is an NVMe result and nothing more.** On a spinning disk or a network mount the
same load would be storage-dominated. That is exactly why the regime is recorded rather than
assumed — and why the cold runs were verified cold with `mincore` after `posix_fadvise(DONTNEED)`
(`119971/119971 → 0`), since an fadvise that silently did nothing would produce a warm number
wearing a cold label.

**No peer comparison is claimed.** Load time is directly comparable against runtimes reading the
same GGUF, and a static binary with no daemon is a plausible place to do well. Nobody has measured
it, and a plausible advantage is not a measured one.

---

## Part A — a registry, so nobody has to guess

`pull owner/repo:quant` requires the user to already know three things: that a GGUF conversion
exists, who published it, and which quantization to ask for. That is knowledge from having spent
time on Hugging Face, which is what a first-time user does not have.

```
$ goinfer-chat models
  qwen2.5-coder-0.5b       0.49 GB  q4_k_m   code completion and small edits; …
                         ~1 GB resident at --quant int4; runs on any laptop
                         Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF · qwen2 · parity full-oracle 100.0%/1.00000
$ goinfer-chat pull qwen2.5-coder-0.5b
```

**It derives from the capability matrix.** A checkpoint entry lives on its family's row in
`docs/capability-matrix.json`; `pull/` embeds a byte copy only because `go:embed` cannot reach
outside a package directory, and `TestRegistry_embeddedMatrixMatchesTheDoc` compares them byte for
byte so the copy cannot drift. There is no second list to maintain.

**Three entries, one per family whose digest could be verified locally.** Small on purpose: an
entry exists because someone ran that checkpoint, not because the family is supported. The 0.5B's
locally-computed sha256 matched `curated.json`'s independently-recorded pin exactly, which is what
established that local files can supply digests matching what Hugging Face serves.

**No CDN, no hosted weights.** Every entry points at Hugging Face with a sha256 the existing fetch
verifies; the short name rewrites into the same `repo:file` reference `ParseRef` already takes, so
verification and resume are unchanged.

### Gates

| gate | what it prevents |
|---|---|
| `embeddedMatrixMatchesTheDoc` | the embedded copy drifting from the canonical matrix — **mutation-checked** |
| `everyEntryTracesToItsFamily` | a recommendation for a family the matrix does not have |
| `noEntryOutrunsItsParity` | recommending a family that is only `experimental` — **mutation-checked**; an `olmo_hybrid` entry is rejected by name and parity |
| `everyEntryIsVerifiable` | an entry without a 64-hex digest, size, repo, file, or the prose that makes a name worth more than a path |
| `listedSmallestFirst` | listing order that is alphabetical rather than cheapest-to-try |
| `doesNotCollideWithDemoTiers` | the two registries shadowing each other |

An empty registry fails rather than passes, so the suite cannot go green having checked nothing.

## What is not built

- **`serve` does not take a short name yet** — only `goinfer-chat pull`/`models`. Same registry,
  one call site.
- **No fit-aware recommendation.** The genuinely useful version is "on *this* machine, run *this*
  checkpoint", which needs `task-fit-to-hardware.md`'s `plan`. The `needs` field is written for it.
- **Three entries is not coverage.** Growing it means verifying a digest per checkpoint, which is
  a download and a hash, not a decision.
- **Load-time instrumentation covers the GGUF path only.** safetensors and `.giw` return a nil
  profile and print nothing, which is why `Summary()` is empty rather than zero.
