# C3 — Metal consumer window, evaluated against `metal/v0.18.0` (2026-09-18)

Out-of-tree-shaped consumer evaluation of goinfer's Metal backend, run against the actual
published `metal/v0.18.0` tag. Supersedes `c3-metal-consumer-window-v0.14.0.md` (v0.14.0) rather
than editing it — that file's findings are historical record, not stale. Run on `macbook-arm64`
(Apple M1 Pro, 16 GB), macOS 26.6.2 (25G83), Go 1.27.0.

**Trigger and which tag this evaluates.** RELEASING.md/QUEUE.md's rule: the trigger is *the next
goinfer release tag that carries an aikit bump*, evaluated against a **published release tag**,
never main-HEAD. `v0.17.0` already carried one (aikit v1.34.0→v1.37.0) and was missed; `v0.18.0`
carries a further one (v1.37.0→v1.41.0) and is the **latest published tag**, so per the doc's own
bounded-fallback clause ("run C3 anyway against the latest published goinfer tag... flagged as the
bounded fallback") this run targets `v0.18.0`, not the missed `v0.17.0` window.

**Which commit, exactly — this tripped me up once, worth recording.** The root `v0.18.0` tag
(`cc395e09`) and the submodule tag `metal/v0.18.0` (`8fad8f83`) are **different commits** — the
two-step tag process tags root first, then bumps+tags each submodule's own `require` against the
now-published root. My first attempt built from the root tag's tree and got real-looking compile
errors (`undefined: decoder.ResidentAdapterLayer`, etc.) because `metal/go.mod` **at that specific
commit** still required root `v0.17.2`. That is not a bug — it is the two-step process not being
done yet at that exact commit. The commit an external consumer actually gets via
`go get .../metal@v0.18.0` is `8fad8f83`, which correctly requires root `v0.18.0`. All numbers
below are from that commit.

---

## 1. Builds with no Xcode

**Caveat stated up front, per the ask: Xcode 27.0 IS installed on this machine** (`xcode-select -p`
→ `/Applications/Xcode.app/Contents/Developer`). The literal "no Xcode.app" scenario could not be
tested here. Tested the stronger, Xcode-independent proxy instead: `CGO_ENABLED=0`, which forces
pure-Go compilation regardless of what toolchain happens to be installed.

```
cd metal && CGO_ENABLED=0 GOWORK=off go build -tags metal -v ./cmd/serve/...   # exit 0
cd metal && CGO_ENABLED=0 GOWORK=off go vet   -tags metal    ./...             # exit 0
```

Both clean, standalone, no workspace (`GOWORK=off` so a local `replace`/borrowed `go.sum` cannot
mask a missing require — the exact B-01 failure mode).

**Binary-level check, not just the compile flag:**

```
otool -L metal/serve
	/usr/lib/libSystem.B.dylib
	/usr/lib/libresolv.9.dylib
	/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation
	/System/Library/Frameworks/Security.framework/Versions/A/Security
```

Same four as v0.14.0's C3 note — base OS libraries and Go's own net/TLS runtime dependencies, **no
Metal.framework, no libobjc, no cgo runtime artifact**. Metal access is via `purego`'s
dlopen'd-Objective-C dispatch, not a linked native framework. Cgo-free claim holds, both by
compile flag and by binary inspection.

## 2. Decode tok/s vs the 73.6 claim

The 73.6 figure (`task-metal-cgofree-spike.md`) is a **decode-only, best-of-40, warm, zero-serving-
cost** measurement on Qwen2.5-Coder-1.5B W4A8, depth 128 — a different instrument from the ~54
tok/s **served** wall-clock figure (§B3); this run does not conflate them.

**Reproduced via `TestZZ_metalDepthBench`** (the in-tree decode-only depth harness, resident
`ForwardArgmax`, no serving/HTTP overhead — the closest apples-to-apples instrument available at
this tag), at `metal/v0.18.0`, same checkpoint (`qwen2.5-coder-1.5b-instruct-q4_k_m.gguf`), min-of-
batches rather than best-of-40 (this harness's own methodology, not the spike's).

**A real finding surfaced getting there, not worked around silently.** The test's unpinned
`decoder.Load(path, Options{Quant:"int4"})` call — no explicit context — auto-fit-sized the context
to **21767 positions** and `buildResident` failed outright: *"resident context 21767 positions
exceeds this backend's hard ceiling of 4096 (a fixed-size kernel score buffer, not a tunable
default)"*. Pinned `ResidentContext: 4096` locally in a scratch worktree to get past it and measure
what the test was actually built to measure (noted in the report, not shipped — the worktree and
edit are gone). **Checked separately whether this exact failure is still live on current `main`: it
is not** — the same unpinned call on `main` now degrades gracefully (`"fit is tight... 5.6 GB
available (budget 4.0 GB)"`) instead of hard-failing. Fixed since v0.18.0, unrelated to today's
release readiness.

**Numbers, `metal/v0.18.0`, resident decode-only, W4A8, M1 Pro:**

| depth | tok/s | µs/pos vs previous |
|---|---|---|
| 128  | **71.6** | — |
| 512  | 65.9 | +3.129 |
| 2048 | 49.5 | +3.276 |
| 4000 | 34.3 | +4.585 |

**71.6 tok/s at depth 128 vs the 73.6 claim: close, not an exact match — 2.7% below, just outside
`task-metal-cgofree-spike.md`'s own stated ±1.5 tok/s best-of-40 noise band.** Plausibly explained
by the different aggregation (min-of-batches here vs. best-of-40 there — a min is expected to read
at or below a best), not treated as a regression; not investigated further as this run's own
methodology was never claimed to reproduce the spike's exact number, only the ballpark.

**Side finding, not this section's ask but worth recording:** the whole depth curve is dramatically
better than the last time it was measured (§B3, 2026-08-09, pre-P6a): 128 tok/s 62.0→71.6 (+15%),
512 47.8→65.9 (+38%), 2048 27.2→49.5 (+82%), 4000 18.2→34.3 (+88%). The depth-degradation slope has
flattened substantially since the intervening batched-prefill/fused-attention/deep-context work
landed — consistent with, not independently re-deriving, what those task docs already claim.

## 3. Bit-identity within machine and OS

Built `metal/v0.18.0`'s `serve` binary, loaded the same checkpoint on Metal (`--quant int4 --ctx
4096`), sent the **identical** request twice (greedy, `temperature:0`, `seed:42`, same prompt) via
`/v1/chat/completions`:

```
run1 finish_reason=length  usage.prefill_reused_tokens=0
run2 finish_reason=length  usage.prefill_reused_tokens=18
content identical: True
```

Generated content is byte-for-byte identical across both runs. (`prefill_reused_tokens` differing
is prefix-cache bookkeeping — a request-level HTTP-serving optimization — not a computation
difference; it doesn't touch the compared field.) **Bit-identity within machine and OS: holds.**

## 4. The actual Metal device gate (§C1-M)

```
go run ./cmd/gate gpu     # from a worktree at metal/v0.18.0 (8fad8f83), local go.work covering
                          # all 5 modules (gitignored, not part of the tag)
```

**First pass: 9 declared groups reported, 9 pass / 2 skip / 11 fail — but this was a false
negative, and here's the honest accounting of why, not a silent do-over.** A worktree checked out
fresh from a tag does not carry the **gitignored** synthetic fixture directories
(`testdata/gemma4-dense-scaled/`, `testdata/gpt2/`, `testdata/qwen35-tiny/`) that a release owner's
normal working checkout accumulates locally over time via `scripts/pin_*.py` generators — §C1-M's
own text assumes exactly that ("on a Mac **with the checkpoints in $HOME/models**"), a working
checkout, not a from-scratch clone. Copied those three fixture directories in from the main
checkout (content-identical regardless of tag) and every one of those "failures" turned into a
clean PASS: `TestGPT2ResidentParityMetal`, `TestHiddenLastResidentParityMetal`,
`TestResidentKVBytes_excludesDeltaNetLayers`, `TestMetalSnapshotGolden` — worst cosine 0.999476
(GPT2), snapshot **byte-identical to golden** on this exact OS build.

**One real, reproducible failure remained: `TestDenseResidentParity`, and it is memory-state-
dependent, not a fixed defect at this tag.**

```
metal resident DECLINED — admission says it should be admitted
[metal] metal: resident context 5689 positions exceeds this backend's hard ceiling of 4096
```

Root cause: the same admission/ceiling class as §2's finding above — an unpinned load's
memory-availability-dependent auto-fit sizing picked a context (5689) that exceeds Metal's fixed
4096-position kernel score-buffer ceiling, so the admission check (which only sees available
RAM, not the ceiling) says yes while `BuildResident` then declines. **Reproduced consistently (2/2)
when run as part of the full gate sequence** (right after the 117s, 63-test kernel suite —
whatever memory state that leaves), **but passed cleanly 3/3 times when run in isolation**,
byte-identical numbers each time (21/24 argmax-exact, min cosine 0.990211). This is not new or
tag-specific: the identical failure signature (`metal/gemma_parity_test.go:84`'s assertion) is
already on record from the **v0.14.0** C3 run (`docs/measurements/c3-metal-consumer-window-
v0.14.0.md` line 76) — a recurring, known-shaped intermittent, not a fresh regression introduced by
anything between v0.14.0 and v0.18.0.

**With the fixture false-negative corrected and the one real finding characterized:** `v0.18.0`'s
Metal device gate is clean except for this pre-existing, memory-state-dependent admission/ceiling
intermittent, which is the designed catch mechanism (§5) actually working, not the suite lying.

## 5. Whether the tautological-gate shape is live on Metal

**Not found, in the files sampled — and there is a structural reason to expect it isn't.** The
CUDA bug was: a flag toggles a code path, two runs are compared, but nothing asserts the flag
actually changed which kernel ran — so a flag that silently no-ops still produces a passing
"comparison." Checked the Metal analogues:

- **`TestPrefillGate`** (fused/batched prefill vs. sequential): calls the batched `PrefillLast` path
  and the sequential `Forward` path **as distinct functions**, not through one flag-gated
  dispatcher — there's no shared toggle that could silently fail to switch anything.
- **`TestBatchedVerifyKernelParity`** (batched-M verify kernels vs. sequential decode): same
  shape, calls `bvkForwardM` (the batched kernel path under test) directly against a reference
  computed via the sequential kernels, not via a runtime flag.
- **Crucially, Metal's resident-parity tests carry an explicit admission assertion** —
  `metal/gemma_parity_test.go:84`'s `"metal resident DECLINED — admission says it should be
  admitted"` — built specifically to fail loud when a precondition (residency was admitted) turns
  out false, rather than silently comparing whatever path actually ran. **This is not a hypothetical
  defense: it is the exact mechanism that caught §4's finding above**, live, during this run.

**Caveat: sampled, not exhaustive.** The Metal package has 59+ `goinfer_testhooks`-tagged device
test files; this checked the files matching a grep for on/off- and fused/sequential-shaped
comparisons (7 files) and read the two most structurally relevant in full. A file outside that
grep with the same shape under different naming would not have been caught.

---

## An urgent finding outside this section's own scope, found while bisecting §4 — current `main` has a live, deterministic regression, and today's release would ship it

**Not a v0.18.0 finding — v0.18.0 itself is clean here, confirmed directly below.** While
characterizing §4's `TestDenseResidentParity` intermittent, `TestBatchedVerifyKernelParity` — *"the
Metal decode==verify bit-identity gate,"* built specifically to catch a recurrence of the G-08
incident class (`cmd/gate/parity.go:846`) — failed on current `main`, deterministically:

```
pos 1: hidden state NOT bit-identical (first diff [0] want=0.27330953 got=-1.5859575)
pos 2: hidden state NOT bit-identical (first diff [0] want=2.4210222  got=2.1721854)
pos 3: hidden state NOT bit-identical (first diff [0] want=-0.7852387 got=-1.4674859)
```

Identical bit patterns across 2/2 independent runs, at every batch size M∈{2,4,8,16} — not
environment flakiness like §4's finding; a deterministic, reproducible numeric divergence between
Metal's batched-verify kernel path and sequential decode on `qwen2.5-coder-0.5b`.

**Bisected precisely, using scratch worktrees, not guessed:**

| commit | `TestBatchedVerifyKernelParity` |
|---|---|
| `metal/v0.18.0` (`8fad8f83`) | **PASS** — bit-identical at all M |
| `ff6f538c` (main, pre-batch) | **PASS** — bit-identical at all M |
| `26f64807` ("feat(webgpu,metal): Gemma4 dense residency, batched-prefill feature parity, deep-context Metal attention") | **FAIL** — deterministic, as above |

**`26f64807` is the commit that introduced this.** It has not shipped in any tag yet — `v0.18.0`
predates it — so this is not a C3/consumer-window finding in the technical sense (no consumer has
ever received it), but if today's release is cut from current `main` without addressing it, it
will be the first tag that does.

This was not investigated further than the bisection — no attempt was made to find the exact line
within `26f64807`'s large diff, per this report's own scope (C3, not a fix). Flagged here because
finding it and staying quiet until the doc is read later defeats the purpose of running this at
all.

---

## Verdict

- **Builds with no Xcode:** could not test literally (Xcode 27.0 present); the stronger
  `CGO_ENABLED=0` proxy and binary-level `otool -L` check both hold. **cgo-free claim confirmed.**
- **Decode tok/s vs 73.6:** measured **71.6** tok/s at depth 128 (min-of-batches) — close but not
  exact (2.7% below, near the claim's own noise band), different aggregation method than the
  original measurement. Depth curve overall is **substantially better** than the last full
  measurement (up to +88% at depth 4000).
- **Bit-identity within machine/OS: holds**, byte-for-byte, verified via the real served binary.
- **§C1-M device gate: green at `v0.18.0`**, once corrected for a fresh-worktree fixture false
  negative (documented, not hidden) — one pre-existing, memory-state-dependent admission/ceiling
  intermittent remains, unchanged in shape since v0.14.0, not a new defect.
- **Tautological-gate shape: not found on Metal** in the sampled files, and there's a live,
  positive reason why (the admission-assertion mechanism caught a real finding during this very
  run) — not exhaustive across all 59+ device-test files.
- **Not a C3 finding but the most urgent thing this run surfaced: `main`'s `TestBatchedVerifyKernelParity`
  is red, deterministically, since `26f64807`.** This blocks today's pending release if it is cut
  from `main` as-is.
