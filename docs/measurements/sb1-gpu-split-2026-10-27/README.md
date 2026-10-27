# SB1: gpu's kernel layer moves to gpu/internal/kernels (2026-10-27, nobara)

Francis's decisions: the alias split with a breaking change allowed (05:46), contained in an internal package (06:34),
after decoder's adapter types moved to `internal/adapter` (06:57, #314). The gates are registered in
`docs/tasks/task-subsystems-2026-10.md`, "SB1 — gpu's kernel layer: decided, and its gates".

- **Base:** `cdfb45c53` (the registration commit; package gpu as on `main`). **Pre-steps:** `b34c7fbef`.
  **Head:** `de965a392` (the split). Box: nobara, RTX 2070 SUPER on Vulkan through wgpu-native, driver 595.91.07, Go
  1.27.3 linux/amd64; `kwin_wayland`'s 17 MiB context on the card for every run; the timing lock held shared.

## The cut

A type-checked declaration graph of package gpu with its in-package tests (`gsplit/dgraph.go.txt`, `gsplit/xref.go.txt`,
`cut.py`): a declaration stays in `gpu` if it uses `decoder`, uses a declaration that stays, or is a method of a type that
stays; an exported method that stays keeps its type in `gpu` (Go puts methods with their type), an unexported one
becomes a function. Two in-package pre-steps first: `Context.uploadProj` became `uploadProj(c, w)`, and the deprecated
`Release` aliases moved from `deprecated_release.go` to `release_aliases.go`. `cut.json` is the result:

- **stays in `gpu`:** `backend.go` (with `decoder.RegisterBackend`, so WebGPU still registers through `import _
  ".../gpu"`), `hwinfo.go`, `kv_slots.go`, `kvcopy.go`, `prefill_scratch.go`, `residency.go`, `resident_sample.go`,
  `testhooks_kvscale.go`, `deprecated_release.go`'s backend check, `doc.go`; the tests that use them; `n_batch_test.go`
  and `kvprec_decline_test.go` (source guards on `residency.go`); `voidrule_test.go`; AX's `tower_identity_test.go`; the 19
  external `gpu_test` files.
- **moves to `gpu/internal/kernels`:** the other 48 non-test files, the four vision towers among them
  (`gemma4_vision.go`, `grid_vision.go`, `qwen25_vision.go`, `vision.go`, with `vision_encoder.go` and
  `vision_register.go`), and 75 test files.

## The edits

All but four are the rewriting tool's (`gsplit/gsplit.go.txt`, run by `apply.sh` on `cut.json`; its record is
`renames.tsv`):

| edit | count |
|---|---:|
| aliases (`type T = kernels.T`) for gpu's public types, in `gpu/api.go` | 12 |
| wrappers for gpu's public functions (`New`, `GPUEverAvailable`, `LiveBufferBytes`) | 3 |
| package-level names the glue uses, exported inside the internal package | 61 |
| fields and methods of non-public types the glue uses, exported there | 165 + 3 |
| accessors for fields and methods of public types (`*kernels.ContextDevice(c)`, `kernels.ContextEnsureAttn(c, ...)`; `...Func` for a method value; `DeviceBufferLit` for a literal) | 48 + 27 |
| test helpers copied across the line (3 `Context` methods copied as functions) | 15 |

`post.sh` makes the rest, each listed in `hand_edits.tsv`: `lint:file-ignore SA1019` in the four files that call the
deprecated `Release` across the new package line (same calls as before; staticcheck flags deprecation only across
packages); the moved tests' 29 paths to the repository's `testdata` (`"../testdata` → `"../../../testdata`); and
`n_batch_test.go`'s guards reading `attention.go` and `decoderunner.go` from `internal/kernels/`, with the `Reset` guard's
pattern following the accessor form `residency.go` now writes its queue through.

## Gates

| gate | result | record |
|---|---|---|
| **G1** declaration identity (`declident.go`, standard library, type-aware) | **PASS**: all 1141 base declarations reappear with identical code after the renames, accessors, conversions and hand edits are mapped back; the generated declarations have their generated shapes. A planted one-character WGSL change fails it (`gemvW8A8ShaderWGSL: differs`), and was reverted. | `g1-declident.txt` |
| **G2** the gpu suite by name, `go test -count=1 -p 1 -v -tags 'gpu goinfer_testhooks' ./...` in `gpu/` | **PASS**: 281 top-level tests at the base (185 PASS, 96 SKIP, 5 min), the same 281 at the head across `gpu` and `gpu/internal/kernels`, every outcome and every skip reason identical | `g2-base-cdfb45c53.log`, `g2-head-de965a392.log` |
| **G3** logits, `go run ./cmd/gate identity -backend webgpu -assets tiny cdfb45c53 de965a392` | **PASS**: 43 families, 39 byte-IDENTICAL (full logits, prefill and 8 steps, the head run twice for determinism), 0 within tolerance, 0 different; 4 NOT RUN for want of a tiny checkpoint of that kind (mellum's fixture is a 4.2 GB slice, deepseek_v2 has none, qwen3_asr and voxtral keep their goldens zipped). 2 m 49 s. The gitignored fixtures were linked in from the main checkout for the run and removed after. | `g3-identity.log` |
| **G4** towers, `TestTowerIdentity_webgpu` with `GOINFER_TOWER_ID_OUT`, compared by `ax2_compare.py` | **PASS** on nobara: 15 cases, 0 differ. The comparison with AX2a's Mac record is mac-ax01's run (`/tmp/sb01-mac-tower.md`, relayed). | `g4-tower-*-webgpu-linux-amd64.json` |
| **G5** apidiff on package gpu, base export vs head | **PASS** by the rule: the only incompatible changes are the 12 types' identity and `New`'s signature, which names `*Context`; nothing removed, nothing added | `g5-apidiff.txt` |

Build and vet of the gpu module under `gpu goinfer_testhooks` and `gpu`, `gpu/cmd` and `demo/agent` builds, and CI's
pinned staticcheck are clean. `go list -deps -test ./gpu/internal/kernels/` names no `decoder`.

## Also changed

- `cmd/gate`: `gate gpu`'s WebGPU cells and its adapter probe include `./gpu/internal/kernels/`, its WebGPU gate scan reads
  that directory, and the Slow-pattern check reads a module's subpackages, as `gate quick`'s `./<module>/...` run does.
- `internal/layout`: gpu reach 172 → 99; `gpu/internal/kernels` joins the `kernels-free-of-decoder` rule (a hand edit of
  the rules, which `-update` never makes); the new import arrows recorded.
- CHANGELOG: the moved types, the `%T` and reflection change, no new public package.
