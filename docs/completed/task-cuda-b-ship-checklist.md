# Ship checklist: dense CUDA-residency backend (B) → downloadable artifact

> **ARCHIVED — a record, not instructions.** This file is closed work kept for its reasoning and
> its numbers. Checkboxes record the state at the moment it was archived: an unticked box means
> "not ticked when this closed", **not** "still to do", and nothing in `docs/completed/` is
> actionable. If you need a task, use the live docs; if something here reads as an instruction to
> a future reader, it was missed at archival — see the doc-closeout rule in
> `docs/parity-coverage-policy.md`, and move it to live policy or strike it.


> **⚠ Peer numbers below predate the Ollama v0.32.5 re-anchor (2026-08-04).** Competitive figures
> in this doc (e.g. Ollama-CUDA ~149, Ollama-Metal 83.3, llama.cpp-CUDA 72.8, and any "×Ollama"
> multiple) were measured against **Ollama 0.5.7 (2025-01) / Ollama-Metal 0.32.0 / llama.cpp as of
> v0.5.0** — historical working records, not current claims. Current same-box numbers vs Ollama
> **v0.32.5** are in `docs/benchmarks.md` §B2 (CUDA) / §B3 (Metal).


> **Status:** (B) is a **verified capability** — kernel tuned (80% peak, 1.47×
> raw / ~parity-to-modestly-ahead discounted vs same-box Ollama), cgo-free proven
> (`ldd`: no libnvrtc/toolkit; `e804aa6`), executor-path number with the 0.34%
> channel tax measured, and a **committed hard equivalence gate**
> (`TestRealForwardParity`, mutation-proven to bite). What remains is **last-mile
> integration + packaging**, not more proving. Scope: **dense residency only,
> opt-in.** NOT MoE/MLA/Mamba CUDA kernels (the declined treadmill). Timing: 0.9-era.

## Part 1 — Last-mile integration (the gap between "test-green" and "a user can run it")

- [ ] **Expose the CUDA backend as a selectable, opt-in path.** Wire it into the
      real generate/serve path (`decoder.Generate` residency seam → `cmd/serve`,
      `demo/chat`), gated behind a build tag (e.g. `-tags cuda`) **and** a runtime
      switch (`--backend cuda`, or auto-detect NVIDIA + explicit opt-in). Default
      build/behavior unchanged: pure-Go CPU + WebGPU stay the default; CUDA is an
      additive backend, never the default.
- [ ] **Graceful fallback.** No NVIDIA driver / `dlopen libcuda` fails / non-dense
      model → fall back to the existing path with a clear one-line stderr note, never
      a crash. (Non-dense families aren't residency-eligible on CUDA — same as WebGPU;
      route them to the staged/CPU path.)
- [ ] **Confirm it runs the whole demo, not just the parity harness** — `demo/chat`
      streams a real q4_k_m dense checkpoint GPU-resident via the CUDA backend, end to
      end (load → decode → stream), on the 2070 SUPER.
- [ ] **Docs:** a short "GPU (CUDA, cgo-free)" section — the `-tags cuda` build, the
      `--backend cuda` switch, the driver-only requirement, the dense-only scope.

## Part 2 — The artifacts (ship the binary; keep the big model out of it)

**A — practical release: small binary + one-line model download.** The default.

- [ ] Build `goinfer-<ver>-linux-amd64-cuda` (and `windows-amd64-cuda`) — the
      `demo/chat`/`cmd/serve` binary with `-tags cuda`, `CGO_ENABLED=0`. A few MB
      (embeds PTX, dlopens the driver). Model is a **one-liner** in the notes
      (`huggingface-cli download …` a dense q4_k_m). License-clean, llama.cpp/Ollama
      convention.

**B — the viral hero: "an LLM on your GPU, in one file, no CUDA install."**

- [ ] A `demo/chat-gpu`-style single binary with a **tiny dense Apache-2.0 model
      embedded** (`go:embed`) — **0.5B-class, ~300–400 MB**, NOT the 1.5B (~1 GB is
      too big for the "one file" magic). Must be **dense** (CUDA residency is
      dense-only) and **Apache-2.0** (Qwen2.5-0.5B) so embedding/redistributing is
      clean. This is the artifact **no one else can ship**; lead the announcement with
      it.

## Part 3 — Release hygiene (the trust + honesty layer)

- [ ] **Credit + attribution (REQUIRED — gocudrv is a real dependency, MIT).**
      `cuda/go.mod` directly requires `github.com/eitamring/gocudrv v0.2.0` (+ purego,
      Apache-2.0), so his code ships in the CUDA binary. Two obligations:
      - **Release notes — credit by name:** goinfer's cgo-free CUDA backend is built
        on [`gocudrv`](https://github.com/eitamring/gocudrv) by **eitam ring**
        ([@eitamring](https://github.com/eitamring)); his write-up
        ["Ten weekends of CUDA in Go"](https://eitamring.github.io/posts/gocudrv-ten-weekends.html)
        is where this line of work started — the driver-only, embedded-PTX approach is
        what lets goinfer decode on the GPU with `CGO_ENABLED=0` and no toolkit.
      - **License texts must ship:** regenerate `THIRD_PARTY_LICENSES.md` via
        `go-licenses report -tags cuda ./cuda/...` (pulls the exact gocudrv MIT +
        purego Apache-2.0 + NOTICE) and confirm `NOTICE`'s CUDA section is present.
        MIT/Apache both require the full notice travel with the binary — the interim
        scaffold must be replaced by the generated file before tagging.
      - **Nice-to-do (not blocking):** after the release, send eitam ring the result —
        his article explicitly asked for reports from cards other than his 4070 Ti, and
        goinfer validated gocudrv on a 2070 SUPER at production scale (the `ldd`
        driver-only proof + the tok/s). Good citizenship; point him at the release.
- [ ] **Platform labels, honest.** The CUDA artifacts are **NVIDIA-only,
      Linux/Windows x86-64**, and sit *alongside* the pure-Go CPU/WebGPU binaries that
      run everywhere. Frame: "the portable one, **plus** a fast NVIDIA GPU build" —
      never imply the project is NVIDIA-only.
- [ ] **Requirement line = the pitch:** "needs your NVIDIA driver (already present if
      you have the GPU) — **no CUDA toolkit, no Python, no cgo.**"
- [ ] **Checksums + reproducibility note.** Ship `sha256`, and state the binary is
      **`CGO_ENABLED=0 go build`-reproducible from source** — anyone can rebuild and
      verify it. This is a real differentiator vs "download our mystery `.so`"; lean
      on it.
- [ ] **Claim discipline — the number.** README/notes say **"dense-model GPU decode
      at native-CUDA-class speed — parity-to-modestly-ahead of same-box Ollama —
      cgo-free, driver-only,"** with the **dense** qualifier. Do NOT headline the raw
      **1.47×**: the methodology asymmetry (our loop omits Ollama's sampling/detokenize/
      server overhead) discounts it ~10–15% to ~parity-to-modestly-ahead. The intersection
      is the story, not the ratio.
- [ ] **The intersection framing** (from `positioning-cuda-and-promotion.md`): sell
      "the embeddable, cgo-free, single-binary Go runtime that runs every modern
      attention family **and** doesn't give up native-CUDA-class GPU speed" — not a
      speed pivot. The downloadable binary *is* the proof; let it carry the pitch.

## Part 4 — Parity / provenance (so the shipped backend stays honest)

- [ ] The CUDA==CPU equivalence gate runs in CI (it does — `TestRealForwardParity`).
      This is the **backend axis** of `parity-coverage-policy.md`; it's the right bar
      for a backend (not a per-family T3).
- [ ] Broaden the parity sample once before the release headline: more prompts /
      longer generations than the current 9/10, so "output matches CPU" rests on a
      wider base than a single short run. Still the same 3%-logit-range gate.
- [ ] The three-fact verification (`ldd`, executor tax, mutation-proven gate) stays in
      `task-cuda-cgofree-spike.md` — cite it from the release notes as the evidence
      behind "cgo-free, driver-only."

## Non-goals (hold the line)

- No MoE/MLA/Mamba/vision CUDA kernels — dense residency only; each is its own
  demand-gated decision.
- CUDA is **never** the default build or backend — additive, opt-in, with fallback.
- Don't ship the big (1.5B+) model *inside* a binary — separate download (A) or a
  tiny embedded model (B). GitHub allows 2 GB assets, but a 1 GB "binary" kills the
  single-file story.
- Don't headline the raw multiplier. Parity-to-modestly-ahead + the intersection.

## The one-line sequence

Wire it in (opt-in `--backend cuda` + fallback + demo runs) → build A (small binary +
HF model) and B (tiny embedded dense model) → label NVIDIA-only alongside the
everywhere binaries, checksum + reproducibility note, dense-qualified honest claim →
lead the announcement with B, the file nobody else can ship.
