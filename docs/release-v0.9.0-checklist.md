# Release checklist — goinfer v0.9.0

> **Status (HEAD `30392b9`, main):** v0.9.0 is a **ship-what's-already-built**
> release, not a build. Two large campaigns landed on `main` after v0.8.0 and are
> **working, gated, and correct — but unreleased and absent from the CHANGELOG**: the
> **speculative-decoding** campaign (`--spec ngram` + grammar-fused, serve-wired;
> EAGLE/Stage-B built-and-parked) and the **cgo-free CUDA backend (B)** (Part-1
> wiring + CI equivalence gate + attribution already in). v0.9.0 packages both, adds
> the two greenlit builds (the embedded-**hero** CUDA binary + buffer-coalescing),
> and runs one bounded **Metal** revisit. **Pre-1.0** — the 14-family parity backfill
> is the separate 1.0 gate; those families ship *experimental* here.
>
> **Freeze rule (once §D1 passes):** no edits to `serialize.go` / `weights.go` /
> `gguf.go` / core (`model.go`, `mlp.go`, `arch.go`, `kvcache.go`, `forwardn.go`) /
> any `forward_*.go` between re-validation and the tag. Each such edit resets §D1.
> Land the tag; resume on 0.9.1.

## The scope in one line

Ship spec-decode (CPU n-gram) + the cgo-free CUDA backend → build the two artifacts
(plain cuda binary **A** + the embedded-hero binary **B**) + buffer-coalescing → run
the bounded Metal W4A8 revisit (non-blocking) → mirror the v0.8.0 seven-step release,
now **tri-module** (main + gpu + cuda), leading the announcement with **B**.

---

## A. Ship the two built-but-unreleased features (the headline)

### A1. Speculative decoding — release the CPU n-gram win
Built on `spec-decode-ngram`, lossless-gated (greedy bit-exact vs non-spec). Public
defensive-publication docs already in `docs/spec/`. **Almost no new code — this is a
framing + CHANGELOG + confirm-green task.**

- [ ] Confirm `--spec ngram` (+ grammar-fused for constrained/tool requests) is
      serve-wired at HEAD (`cmd/serve/openai.go`) and its lossless parity gate is green.
- [ ] Confirm recurrent families (Mamba-2 SSM, Gated DeltaNet) are **guarded out** of
      speculation → fall back to plain decode (`TestSpecRollbackSafetyGuard`).
- [ ] Frame it honestly in the notes: *opt-in, lossless CPU speculative — n-gram wins
      on copy-heavy traffic (code edits / RAG / agent loops) where the draft is free.*
      Do **not** imply a general speedup (EAGLE is a CPU loss — see below).
- [ ] Document EAGLE-3 (built end-to-end + lossless) and Stage-B GPU verify as
      **built-and-parked, GPU-blocked** — not vaporware, not shipped-fast. Point to
      `docs/spec/` as the disclosure.

### A2. cgo-free CUDA backend (B) — finish last-mile + hygiene
Part-1 wiring (`--backend cuda`, opt-in + fallback), the CI equivalence gate, and
attribution/licenses already landed (`308a610` / `e16fd93` / `8705eb6` / `2ba621e`).
The `[ ]` boxes in `task-cuda-b-ship-checklist.md` are **stale** — genuinely remaining:

- [ ] **Part 1 confirm:** `demo/chat` streams a real q4_k_m dense checkpoint
      CUDA-resident end-to-end (load → decode → stream) on the 2070 SUPER — not just
      the parity harness. Finish the "GPU (CUDA, cgo-free)" docs section.
- [ ] **Part 3 hygiene:** NVIDIA-only / Linux-Windows-x86-64 labels *alongside* the
      portable binaries; the requirement-line pitch ("your NVIDIA driver — no toolkit,
      no Python, no cgo"); `sha256` + the `CGO_ENABLED=0`-reproducible note; **claim
      discipline** — "dense-model GPU decode at native-CUDA-class speed,
      parity-to-modestly-ahead of same-box Ollama," **never headline the raw 1.47×**.
- [ ] **Part 4 parity:** broaden the CUDA==CPU sample beyond 9/10 (more prompts /
      longer gens) before the headline; cite the 3-fact verification (ldd / executor
      tax / mutation-proven gate) from `task-cuda-cgofree-spike.md`.
- [ ] Confirm the **gocudrv / eitam ring credit-by-name** block is in the release
      notes (NOTICE + THIRD_PARTY_LICENSES already carry the license texts; the notes
      still need the human credit + the ten-weekends link). Post-release: send him the
      2070 SUPER result.

## B. The two greenlit builds

### B1. CUDA artifacts — A (practical) + B (the hero): full promotion push
- [ ] **Artifact A:** `goinfer-<ver>-{linux,windows}-amd64-cuda` — the
      `demo/chat` / `cmd/serve` binary, `-tags cuda`, `CGO_ENABLED=0`, a few MB; model
      is a one-line HF download in the notes.
- [ ] **Artifact B (lead with this):** a `demo/chat-gpu` single binary with a **tiny
      dense Apache-2.0 model embedded** (`go:embed`, Qwen2.5-0.5B, ~300–400 MB — NOT
      the 1.5B). Dense (CUDA residency is dense-only) + Apache-2.0 (clean to
      redistribute). "An LLM on your GPU, one file, no CUDA install" — the artifact no
      other Go runtime can ship. The announcement centerpiece.

### B2. Buffer-coalescing — the named 0.9 load-time win
- [ ] Collapse the thousands of per-projection `CreateBufferInit` calls into a handful
      of big buffers (the dominant cost in the remaining ~13 s int4-resident load).
      Lives in the `gpu/` upload path — **not** decoder core — so it lands clean with
      its own `gpu/` tag and **zero** forward-parity re-validation. The real path from
      ~13 s toward "seconds."

## C. Bounded Metal revisit (spike — NON-BLOCKING)
The Metal spike measured **NO-GO** on the untuned int8 kernel (~18–30 tok/s ≈
0.3–0.4× the ~71 bar; viability + correctness GREEN). One bounded lever remains:

- [ ] Swap the GEMV to the **already-bit-exact int4 / W4A8** kernel + ILP-unroll +
      multi-row-per-threadgroup; re-measure vs the **~71 tok/s** bar (85% of same-box
      Ollama-Metal 83.3). Timebox it.
- [ ] **Its own go/no-go:** **GO** (≥ 71) → append "measured GO" to
      `task-metal-cgofree-spike.md`; it becomes a documented **parked capability** (not
      a shipped backend) and an optional 0.9.0 talking point. **NO-GO** → Metal stays
      parked, the number is banked, move on.
- [ ] **Do not let it gate the tag.** If it drags past its timebox, park it and ship
      v0.9.0 without it.

## D. Release mechanics (mirror the v0.8.0 checklist, now tri-module)

### D1. Parity re-validation ⛔ blocker — the spec campaign touched core
The spec work added **tree attention to `forwardN`** plus the `speculative.go`
verify/rollback substrate. If any hashed-core / `forward_*.go` file moved since a
family's `validated_at`, its row is green-by-staleness, not re-validated — the exact
v0.8.0 §1 trap.

- [ ] Run `TestParityManifest_fresh` at HEAD. For any family it flags — especially the
      **10 generic-forward families sharing one `deps_hash`** (the coupling caveat: one
      core edit re-stales all ten) — **re-run the real parity gate**, bump
      `validated_at` + metrics. Do this **at one commit, as late as possible** before
      freeze.
- [ ] Confirm the deps-split still excludes the CUDA backend + the `gpu/`
      buffer-coalesce from the CPU-numerics hashed set (separate axis — CUDA==CPU is
      gated by `TestRealForwardParity`, not a per-family T3).

### D2. Pre-tag gates (on the box)
- [ ] `go build ./...`, `-tags gpu`, **and `-tags cuda`** all clean.
- [ ] `go test ./...` green — incl. `TestParityManifest_fresh`, capability-matrix
      freshness + coverage, the `.giw`/resident gates, the spec lossless gates, and
      `TestRealForwardParity` (CUDA backend axis, CI-wired).
- [ ] Capability matrix regenerated, no diff. `!unix` mmap fallback compiles.

### D3. Stale-doc cleanup (forgotten-work risks — fix while here)
- [ ] `task-cuda-b-ship-checklist.md` — mark Part-1 wiring / CI / attribution LANDED.
- [ ] `multimodal.md` — Qwen2.5-VL P5 shipped in v0.7.0; only the GGUF `mmproj` loader
      seam remains open. Correct the header.
- [ ] `roadmap.md` (dated 2026-06-19) — its "MTP / spec — parked" lines are superseded
      by the shipped spec campaign; add a pointer to `docs/spec/`.
- [ ] `docs/spec/06-acceptance-analysis.md` — its unchecked boxes lag the finished
      run-log; reconcile.

### D4. CHANGELOG
- [ ] Create `## [v0.9.0] — <date>` (there is currently **no `[Unreleased]`** section).
- [ ] Cover the whole arc since v0.8.0: **speculative decoding** (n-gram +
      grammar-fused, lossless, opt-in; EAGLE / Stage-B built-and-parked) + the
      **cgo-free CUDA backend** (opt-in `--backend cuda`, dense-only, driver-only) +
      **buffer-coalescing** + the **Metal spike finding** (GO/NO-GO per §C).

### D5. Tag — tri-module (main + gpu + cuda)
- [ ] **Decide the `cuda/` module's first tag** — it's a new module (`cuda/go.mod`);
      pick the convention (`cuda/v0.9.0`, mirroring `gpu/`) and whether it tracks the
      main version.
- [ ] Bump `gpu/go.mod` (and `cuda/go.mod`) require → `v0.9.0`; reconcile the
      `replace => ../`.
- [ ] Tag `v0.9.0`, `gpu/v0.9.0`, `cuda/v0.9.0`; push; verify
      `go list -m github.com/townsendmerino/goinfer@v0.9.0` resolves.

### D6. Publish + promote
- [ ] GitHub release from the `[v0.9.0]` CHANGELOG; attach artifacts **A + B** and the
      portable CPU/WebGPU binaries; `sha256` for each.
- [ ] **Lead the announcement with artifact B** (the embedded-hero, "one file, no CUDA
      install") and the **intersection** framing (portable + cgo-free + every attention
      family **and** native-CUDA-class speed) — not a speed pivot. Credit eitam ring /
      gocudrv by name.
- [ ] Heads-up notes: any cache/fingerprint invalidations from the spec / CUDA work.

---

**Bottom line:** the two headline features are already built and correct — v0.9.0 is
mostly packaging, honesty (labels + claim discipline), and one ⛔ real gate (§D1
parity re-validation, because the spec campaign touched `forwardn.go`). The only
genuinely *new* engineering is artifact B's embed plumbing, buffer-coalescing, and the
bounded Metal revisit — and the Metal spike must not gate the tag. Lead with B; sell
the intersection.
