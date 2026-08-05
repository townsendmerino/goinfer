# Release checklist — goinfer v0.9.0

> **Status (main, 2026-08-04):** v0.9.0 is a **ship-what's-already-built** release. The
> engineering landed across ~520 commits since v0.8.0 and is working, gated, and correct;
> what remains is **re-validation, the tri-module tag, and publishing**. This checklist was
> rewritten on 2026-08-04 — the earlier version described a stale, smaller scope (HEAD
> `30392b9`: "CUDA Part-1 wiring", an embedded-hero binary, buffer-coalescing, a *bounded
> Metal revisit to run*). All of that is superseded: **both** GPU backends are fully landed,
> the Metal campaign is concluded and characterized, and the CHANGELOG is written.
>
> **What actually landed** (see `CHANGELOG.md [v0.9.0]` and `docs/releases/v0.9.0.md`):
> two cgo-free GPU backends (CUDA + Metal), speculative decoding (CPU n-gram + grammar-fused
> + resident CUDA D1), batched prefill (CUDA default-on, Metal declined), split-KV
> long-context decode, attention coalescing, family coverage (Qwen3 / Mistral / Phi-3 /
> Gemma-3 / MoE / Cohere-Command-R), the shared admission taxonomy, the 26B host↔VRAM paging,
> the bit-identity contract (contraction fix + FMA lint + Metal snapshot golden + width pins),
> and the Ollama v0.32.5 re-anchor + §B4 retraction.
>
> **Pre-1.0:** families without both a T1 committed golden and a current T3 manifest row ship
> *experimental* (`docs/parity-coverage-policy.md`).
>
> **Freeze rule (once §D1 passes):** no edits to `serialize.go` / `weights.go` / `gguf.go` /
> core (`model.go`, `mlp.go`, `arch.go`, `kvcache.go`, `forwardn.go`) / any `forward_*.go`
> between re-validation and the tag. Each such edit resets §D1. Land the tag; resume on 0.9.1.

## The scope in one line

Re-validate parity at one late commit (§D1) → confirm the pre-tag gates on each box (§D2) →
tri-module tag (main + gpu + cuda) → publish, leading with **the two cgo-free GPU backends +
the bit-identity contract**, claims disciplined (absolute tok/s + cgo-free, never a peer
multiple).

---

## A. Confirm the landed features (framing + confirm-green, not build)

### A1. Speculative decoding
- [x] `--spec ngram` + grammar-fused serve-wired (`cmd/serve/openai.go`), lossless gates green
      (`TestNgramSpeculativeGreedyParity`, `TestResidentSpecServe`, `TestGrammarSpecParity`).
- [x] Resident CUDA D1 serve-wired + lossless-vs-sequential (`32dcc8f`, `TestSpecDecodeCurve`).
- [x] Recurrent families + staged sliding-window guarded out (`TestSpecRollbackSafetyGuard`).
- [ ] Notes framing: *opt-in, lossless; n-gram wins on copy-heavy traffic; do not imply a
      general speedup.* EAGLE-3 / Stage-B documented as **built-and-parked, GPU-blocked**
      (`docs/spec/`) — not vaporware, not shipped-fast.

### A2. cgo-free CUDA backend
- [x] Resident decode + prefill, `--backend cuda`, opt-in + CPU fallback, CI `-tags cuda`.
- [x] Forward guarded by the mutation-proven `TestRealForwardParity` (CI-wired).
- [ ] Claim discipline in the notes: "dense-model GPU decode, cgo-free / driver-only (no
      toolkit, no Python, no cgo)"; lead with **absolute tok/s** (1.5B 218.6, 0.5B 507.5),
      **never** a peer multiple. gocudrv / eitam-ring credited by name (NOTICE +
      THIRD_PARTY_LICENSES already carry the texts).

### A3. cgo-free Metal backend (campaign concluded — no "revisit" pending)
- [x] Resident decode + prefill, `--backend metal`, darwin-only, opt-in + CPU fallback.
- [x] Characterized to the ceiling (`docs/metal-verdict.md`): decode dispatch-/issue-bound;
      long-context gap structural; **M1 = M-A decided** (stay bit-identical, M-B deferred).
- [x] §B3 re-anchored vs Ollama-Metal v0.32.5 (0.96× / 0.74×).
- [ ] Notes framing: cgo-free / no-Xcode + correctness parity — **not** a raw-speed multiple.

## B. Optional / post-release (do NOT gate the tag)
- [ ] **Buffer-coalescing** (`gpu/` upload path, load-time win, non-numeric) — not required for
      v0.9.0; land on 0.9.1 if not already in. It touches only `gpu/`, so it needs no
      forward-parity re-validation.
- [ ] **Embedded single-file demo binary** (`go:embed` a small Apache-2.0 dense model) — a nice
      promotion artifact ("an LLM on your GPU, one file"), optional. If cut, ship the plain
      `-tags cuda` / `-tags metal` binaries + a one-line model download in the notes.

## C. Release mechanics (mirror v0.8.0, now tri-module)

### C1. Parity re-validation ⛔ BLOCKER — the spec + cohere + gemma4 work touched core
The spec campaign added **tree attention to `forwardN`**; Cohere is the first forwardN-eligible
logit-scale family (`lmHeadN` fix); gemma4_text merge gaps were closed. Any hashed-core /
`forward_*.go` movement since a family's `validated_at` makes its row green-by-staleness, not
re-validated — the v0.8.0 §1 trap.

- [ ] Run `TestParityManifest_fresh` at HEAD (**green in the Mac pure-Go census, 2026-08-04** —
      confirm it still is at the freeze commit). For any family it flags — especially the
      generic-forward families sharing one `deps_hash` (one core edit re-stales all of them) —
      **re-run the real parity gate on the box**, bump `validated_at` + metrics. Do it **at one
      commit, as late as possible** before freeze.
- [ ] **On the Linux box:** `go test -tags cuda ./cuda/` incl. `TestRealForwardParity`, the
      spec curve, and the CUDA real-checkpoint parity — the CUDA==CPU axis is not covered by the
      Mac census. Record the numbers.
- [ ] Confirm the deps-split still excludes the CUDA backend + `gpu/` from the CPU-numerics
      hashed set (separate axis, gated by `TestRealForwardParity`, not a per-family T3).

### C2. Pre-tag gates (per box)
- [ ] `go build ./...`, `-tags gpu`, `-tags metal` (darwin), **`-tags cuda`** (Linux) all clean.
- [x] **Mac census captured (`scripts/skip_census.py`, 2026-08-04):** pure-Go **415 pass / 0
      fail / 167 skip**; Metal **59 pass / 0 fail / 30 skip** — every skip asset/GPU/heavy-gated,
      no silent failures; the Metal snapshot golden (bit-identity) passes; the package-level
      `fault 0x10` under concurrent GPU load is the known spurious contention crash (re-run as
      sole GPU user), not a test failure.
- [ ] **Box census:** run `scripts/skip_census.py -- -tags cuda ./cuda/` on the Linux box;
      record pass/skip/fail + buckets in the release notes. Optionally
      `GOINFER_REQUIRE_FIXTURES=1` to make committed-fixture skips hard-fail.
- [ ] Capability matrix regenerated, no diff. `!unix` mmap fallback compiles.

### C3. Stale-doc cleanup
- [x] `parity-coverage-policy.md` — committed-fixture set enumerated as chosen policy.
- [x] `task-rotation-perrow-imma.md` — decision folded in (DEFER; committed, not left untracked).
- [ ] `task-cuda-b-ship-checklist.md` — mark Part-1 wiring / CI / attribution LANDED.
- [ ] `roadmap.md` — its "MTP / spec parked" lines are superseded by the shipped spec campaign;
      point to `docs/spec/`.

### C4. CHANGELOG + release doc
- [x] `## [v0.9.0]` written (`CHANGELOG.md`), covering the whole arc since v0.8.0.
- [x] `docs/releases/v0.9.0.md` added (the directory previously stopped at v0.8.0).

### C5. Tag — tri-module (main + gpu + cuda)
- [ ] **`cuda/` module's first tag:** it's a new module (`cuda/go.mod`); use `cuda/v0.9.0`,
      mirroring `gpu/`, tracking the main version.
- [ ] Bump `gpu/go.mod` + `cuda/go.mod` require → `v0.9.0`; reconcile the `replace => ../`
      (per the release-hygiene rule: the require bump lands in the release commit so `gpu/<ver>`
      and `cuda/<ver>` are go-gettable).
- [ ] Tag `v0.9.0`, `gpu/v0.9.0`, `cuda/v0.9.0`; push (`gh auth switch --user townsendmerino`
      first); verify `go list -m github.com/townsendmerino/goinfer@v0.9.0` resolves.

### C6. Publish + promote
- [ ] GitHub release from the `[v0.9.0]` CHANGELOG; attach the `-tags cuda` / `-tags metal` +
      portable CPU/WebGPU binaries; `sha256` for each.
- [ ] **Lead the announcement with the two cgo-free GPU backends + the bit-identity contract**
      (portable, `CGO_ENABLED=0`, byte-reproducible against CPU) — the intersection no other Go
      runtime ships. Not a speed pivot. Credit eitam-ring / gocudrv by name.
- [ ] Heads-up notes: any cache/fingerprint invalidations from the spec / CUDA work.

---

**Bottom line:** the features are built, correct, and documented. v0.9.0 is now **re-validation
(§C1, the one ⛔ gate, because spec/cohere/gemma4 touched core) + the tri-module tag +
publishing**. The Mac side is green and censused; the CUDA real-parity axis and the box census
are the remaining measured gates. Lead with the two backends and the bit-identity story; keep
claims disciplined.
