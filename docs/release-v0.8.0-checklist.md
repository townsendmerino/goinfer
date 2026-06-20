# Release checklist — goinfer v0.8.0

> **Status (HEAD `a7ffb42`):** feature-complete; the GPU resident-decode expansion +
> Mellum2 fast-load arc is done and the demo/README showcase is consistent. The
> staleness gate is *mechanically* green, but two rows are green via a hash
> re-baseline, not a re-validation — that's the one real pre-tag item (§1). Then
> it's routine: gates → CHANGELOG → dual-module tag. **Target: v0.8.0** (pre-1.0;
> 1.0 is a separate freeze + full backfill, not this release).
>
> **Freeze rule for the duration:** no edits to `serialize.go` / `weights.go` /
> `gguf.go` / core (`model.go`, `mlp.go`, `arch.go`, `kvcache.go`) / any
> `forward_*.go` between now and the tag. Each such edit resets the re-validation
> below. Land the tag first; resume on `0.8.1`.

## 1. Parity: confirm "green = re-validated", not "green = re-hashed"  ⛔ blocker

`15f844d` ("refresh parity-manifest deps_hashes — re-baseline to HEAD, numerics
proven intact") rewrote recorded `deps_hash` values to match HEAD while leaving
`validated_at` at pre-change commits. That makes `TestParityManifest_fresh` pass
mechanically. It's only *honest* if the families' parity gates were actually re-run
at HEAD. Two families changed their **own forward file** after `validated_at`, so
they need the real confirm:

- **`deepseek_v2` / `deepseek_v3`** — `forward_deepseek.go` changed at `c5ac8c9`
  ("absorb kv_b_proj into q/o"). The MLA absorb is algebraically equivalent but runs
  attention in compressed space → **changes float reassociation** → argmax can flip
  on near-ties. Needs the argmax+cosine oracle, not the algebra.
- **`qwen3_5_moe`** — `forward_qwen35.go` changed 3× since `e3eb033`. Same question.

**Action per family:**
- If its parity gate **was** re-run at HEAD and passed → **bump `validated_at`** to
  `15f844d` (or HEAD) so the row's provenance stops pointing at a pre-change commit.
- If it was only **asserted** equivalent → **run the gate** (`scripts/parity_sweep.sh`
  / the family's real-oracle or weightDiff test) at HEAD; on pass, update
  `validated_at` + metrics.

**Lower-risk (spot-confirm, don't block):** `gemma4` (`d64afe4`), `llama4_text`
(`3f23c96`), `phi3` (`af813a0`) — **no own-forward change**; their only exposure is
shared-core edits that the deps-split argues are non-CPU-numerics. Confirm the
deps-split actually excludes the GPU-residency churn from their hashed set; if so,
they're legitimately still valid.

**14 pending families** — not a blocker. v0.8.0 is pre-1.0, so they ship as
*experimental* and the capability matrix honestly shows `pending`. (Full validation
of all 14 is the **1.0** gate, separate.)

## 2. Policy guardrail — close the re-baseline hole  (do with §1)

The "re-baseline deps_hashes to HEAD" operation can silently convert stale→green
without re-validation — the exact hole the staleness gate exists to close. Add to
`parity-coverage-policy.md`: *a `deps_hash` refresh for a `validated` family is only
permitted when either (a) its parity gate was re-run at that commit and
`validated_at` is bumped with it, or (b) the changed files are provably non-numeric
(serialize / GPU-residency sets already excluded by the deps-split). A bare re-hash
of a validated row that changed a forward/core file is forbidden.* Otherwise the
tool edits the answer key.

## 3. Pre-tag gates (on the box)

- [ ] `go build ./...` and the `gpu/` module build (`-tags gpu`) both clean.
- [ ] `go test ./...` green — **including** `TestParityManifest_fresh`,
      `TestCapabilityMatrix` freshness, the family↔matrix coverage test, and the
      `.giw`/resident gates (`TestGIWInt4_resident`, `TestInt4LayoutMatch`).
- [ ] Capability matrix regenerated and clean (`go test ./decoder -run
      CapabilityMatrix -update` → no diff).
- [ ] Cross-platform build sanity: the `!unix` mmap/madvise fallback compiles
      (post aikit/mmap move).

## 4. CHANGELOG

- [ ] `[Unreleased]` → `## [v0.8.0] — <date>`.
- [ ] Confirm it covers the whole arc since v0.7.0: GPU resident-decode expansion
      (most families) + **MLA latent-attention residency** + **Mamba-2 SSM decode
      engine** + Nemotron-H (default-on int4) + Granite-4.0-H (opt-in); **Mellum2
      fast-load** (safetensors→`.giw` Path B, direct int4 upload 66s→~13s warm) +
      the **GGUF rope_parameters fix** + the **serialize deps-set split** + the
      **`quant=int4mix`** label; the **aikit/mmap substrate move**; findings
      (Granite int8 cliff, no v29 decode penalty).
- [ ] Note as a known limitation if anything's deferred (none outstanding — rope
      fixed, GPU-layout resolved without a format).

## 5. Tag (dual module)

`gpu/` is a separate module (`go.mod` requires the main module, `replace => ../`).

- [ ] **Bump `gpu/go.mod`** require `github.com/townsendmerino/goinfer` →
      `v0.8.0`; reconcile the `replace => ../` per the v0.6.0/v0.7.0 release
      practice (prior releases bumped the gpu require alongside the main tag).
- [ ] Tag main: `git tag v0.8.0 && git push origin v0.8.0`.
- [ ] Tag gpu submodule: `git tag gpu/v0.8.0 && git push origin gpu/v0.8.0`
      (verify the convention against the current `gpu/v0.7.0` tag).
- [ ] `go list -m github.com/townsendmerino/goinfer@v0.8.0` resolves post-push.

## 6. Publish + after

- [ ] GitHub release from the `[v0.8.0]` CHANGELOG section; attach the
      `demo/chat` single-file binaries if that's the release-artifact practice.
- [ ] Confirm the README showcase gif (`docs/assets/mellum2-gpu.gif`,
      `[backend=webgpu quant=int4mix]`) renders on the release page.
- [ ] Heads-up in notes: the `quant`-label fingerprint change invalidates
      pre-existing **mixed-bundle** `.giw-kv` warm snapshots (one-time cold prefill
      on upgrade; per-version caches, expected).

## 7. First 0.8.1 / 0.9 item (not now)

**Buffer-coalescing** — collapse the thousands of per-projection `CreateBufferInit`
calls into a handful of big buffers (the dominant cost in the remaining ~13s, since
the 7 GB PCIe transfer is sub-second). Lives in the `gpu/` upload path, **not** the
decoder `loaders`/core set, so — thanks to the deps-split — it lands clean post-tag
with its own `gpu/` tag and **zero** forward-parity re-validation. This is the real
path toward "seconds."

---

**Bottom line:** the only thing between you and an honest tag is §1 — run (or confirm
you ran) the deepseek + qwen3_5_moe parity gates at HEAD and fix their `validated_at`.
Everything else is mechanical. Ship it.
