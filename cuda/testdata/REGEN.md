# Regenerating `moe.ptx` (and the other audited 12.6 artifacts)

> **STATUS 2026-09-10 (audit M-35): CLOSED via option (a).** `moe.ptx`, `glue.ptx` and
> `gemv_fwd.ptx` — the three files `cuda/kernels.go` names as the audited set — had each
> independently drifted to this box's ambient NVRTC 12.9.86 across three unrelated regens
> (`moe.ptx` in `610ce7f`, `glue.ptx` in `23c46b13`, `gemv_fwd.ptx` in `5b443834`), none of them
> at the pinned toolchain. Re-pinned at the genuine 12.6.85 via the procedure below (`/tmp/venv-
> nvrtc12685`, same wheel versions), on the Linux box. Per-kernel hash audit (splitting each PTX
> on `.visible .entry` and hashing each kernel's own section — the method this doc's control
> below prescribes) found 14 of 16 kernels across the three files BYTE-IDENTICAL between the two
> toolchains; only `gemv_f32_a8` (moe.ptx) and `glu_quant` (glue.ptx) differ, and both diffs are
> register-allocation/scheduling only (confirmed via `diff`: parameter/register renumbering and
> basic-block relabeling, no operation reordering that would change FMA contraction) — `glu_quant`
> is additionally in `lintedKernels` (FMA-linted), so its source already forces explicit
> `__fmaf_rn` ordering regardless of what the compiler would otherwise choose. The real
> MoE-resident-parity gate (`moe_parity_test.go`) measures the EXACT SAME `min cosine 0.997829`
> before and after — not just passing, byte-for-byte the same number, on the box's real GPU.
> `TestMoEPTX_versionMatchesItsDocumentation` is green now that the claim and the artifact agree
> again; it stays as a standing drift guard rather than being deleted, since nothing prevents a
> fourth ambient regen.

`moe.ptx` is an **audited artifact**: it ships built at CUDA **NVRTC 12.6.85**, while this dev box
runs 12.9.86. The standing rule is often paraphrased as "never regenerate moe.ptx". That is not
quite the rule, and the imprecision costs real work — it pushes the next person toward writing a
duplicate kernel in a new file to dodge a regen that was actually fine.

**The rule is: never regenerate at a DIFFERENT toolchain.**

- **Adding a NEW kernel** → new `.cu` file, new `.ptx`. Built at whatever NVRTC is present; the
  audited artifacts are untouched. This is why `router_f32.cu` and `argmax.cu` exist.
- **Changing an EXISTING kernel already inside `moe.ptx`** → regenerate `moe.ptx` **at the pinned,
  identical 12.6.85**, with the control below. A same-version regen is provably a no-op on every
  kernel you did not edit, so the diff is auditable. Do **not** clone the kernel into a second file
  to avoid this: two implementations that must agree is a worse failure mode than one artifact with
  a reviewed diff.

## Procedure

```bash
# 1. Pinned toolchain — the version is load-bearing, not a floor.
python3 -m venv /tmp/venv-nvrtc12685
/tmp/venv-nvrtc12685/bin/pip install \
    "nvidia-cuda-nvrtc-cu12==12.6.85" "nvidia-cuda-runtime-cu12==12.6.77"
V=/tmp/venv-nvrtc12685/lib/python3.*/site-packages/nvidia

# 2. CONTROL FIRST — rebuild UNCHANGED and prove the artifact is byte-identical.
#    If this does not reproduce, STOP: your toolchain is not the one that built the artifact,
#    and any diff you produce afterwards is a toolchain bump wearing a kernel edit's clothes.
sha256sum cuda/testdata/moe.ptx                     # record BEFORE
cd cuda && NVRTC_LIB="$V/cuda_nvrtc/lib" CUDA_INC="$V/cuda_runtime/include" \
    bash build_ptx.sh moe
sha256sum testdata/moe.ptx                          # MUST equal BEFORE

# 3. Now make the source edit, rebuild the same way, and AUDIT THE DIFF:
#    - `diff -u` the old and new PTX; the hunks must be confined to the kernel you edited.
#    - split the PTX on `.visible .entry` and hash each kernel's section; every kernel you did
#      not edit must be byte-identical. Do not rely on the whole-file diff looking small.
```

Comments and whitespace in the `.cu` are codegen-neutral — verified by rebuilding after a
comment-only edit and getting the same sha — so documenting a kernel never dirties the artifact.

## Record: re-pin at 12.6.85 after three ambient drifts (2026-09-10, audit M-35)

No source edit — this is option (a) closing the M-35 status note above. `moe.ptx`, `glue.ptx`,
`gemv_fwd.ptx` were each rebuilt from their CURRENT (unchanged) `.cu` sources at the pinned
12.6.85 toolchain (`/tmp/venv-nvrtc12685`, `nvidia-cuda-nvrtc-cu12==12.6.85` +
`nvidia-cuda-runtime-cu12==12.6.77`), replacing what three separate ambient-NVRTC regens
(`610ce7f`, `23c46b13`, `5b443834`) had each left at this box's then-current 12.9.86.

| file | kernels | byte-identical across toolchains | differ |
|---|---|---|---|
| `moe.ptx` | 6 | `gemv_w4a8_moe`, `gemv_w4a8_moe_wacc`, `gemv_w4a8_moe_wacc_bias`, `moe_route`, `shared_gate_combine` | `gemv_f32_a8` |
| `glue.ptx` | 8 | `attention`, `layernorm_quant`, `quant_vec`, `residual`, `rmsnorm_f32`, `rmsnorm_quant`, `rope` | `glu_quant` |
| `gemv_fwd.ptx` | 2 | `kv_store`, `rope_kv` | — (both identical) |

**Confined-diff audit** on the two kernels that differ: both are register-allocation/basic-block
relabeling only (`diff` on each kernel's own PTX text shows parameter/register renumbering and
branch-target relabeling, no reordering of the arithmetic operations themselves).
`glu_quant` is in `lintedKernels` (FMA-linted — glue.cu is not exempt), so its source already
forces explicit `__fmaf_rn` ordering; the toolchain has no contraction discretion to exercise
there regardless. `gemv_f32_a8` lives in moe.cu (still FMA-lint exempt, bare MACs) so is the one
kernel where a real, if tiny, numeric drift was structurally possible — measured, not assumed:
`TestMoEResidentParity`'s real-GPU gate (`moe_parity_test.go`) reports `min cosine 0.997829`
identically before and after, the same run-to-run-stable number this gate has always reported on
this fixture.

| step | file | sha256 (first 16) |
|---|---|---|
| checked in before this fix (12.9.86) | `moe.ptx` | see `git log -p` at this commit's parent |
| checked in before this fix (12.9.86) | `glue.ptx` | see `git log -p` at this commit's parent |
| checked in before this fix (12.9.86) | `gemv_fwd.ptx` | see `git log -p` at this commit's parent |
| after re-pin at 12.6.85 | `moe.ptx` / `glue.ptx` / `gemv_fwd.ptx` | recorded in this commit's own diff |

`argmax.ptx`, `router_f32.ptx`, `prefill_batched.ptx` and everything else in `testdata/` are
built at whatever NVRTC was on hand when they were added (per this doc's own rule: adding a NEW
kernel needs no pin) — their 12.9.86 headers are expected and unchanged by this fix.

## Record: `MOE_MAX_E` 256 → 512 (2026-08-09)

Raised so Kimi-K2 (384 routed experts) stops declining to CPU on CUDA. See
`decoder/features.go` (`residentBackendMoECap`) and `docs/completed/task-model-family-deepseek-v4-kimi-k3.md`.

| step | sha256 (first 16) | bytes |
|---|---|---|
| checked-in, before anything | `1e08efd5411ab0b3` | 64287 |
| **control**: rebuilt UNCHANGED at 12.6.85 | `1e08efd5411ab0b3` ✅ identical | 64287 |
| after `MOE_MAX_E 256 → 512` | `42308114b33b1099` | 64287 |
| after comment-only edits, rebuilt | `42308114b33b1099` ✅ unchanged | 64287 |

**Confined-diff audit.** The entire PTX diff was two hunks in `moe_route`:

```
-	.local .align 16 .b8 	__local_depot0[2368];
+	.local .align 16 .b8 	__local_depot0[4416];
-	add.u64 	%rd6, %SPL, 1024;      +	add.u64 	%rd6, %SPL, 2048;
-	add.u64 	%rd7, %SPL, 2048;      +	add.u64 	%rd7, %SPL, 4096;
-	add.u64 	%rd8, %SPL, 2304;      +	add.u64 	%rd8, %SPL, 4352;
```

The depot arithmetic accounts for it exactly:
`2368 = 256*4 (score) + 256*4 (sel) + 64*4 (gscore) + 64 (keep)` →
`4416 = 512*4 + 512*4 + 64*4 + 64`. Per-kernel hashes: `moe_route` CHANGED;
`gemv_f32_a8`, `gemv_w4a8_moe`, `gemv_w4a8_moe_wacc`, `shared_gate_combine` **byte-identical**.

## Perf control for the 256 → 512 raise

The depot grows **unconditionally**, so a small-`nE` model pays the larger frame while using none of
it. Measured by loading the old and new PTX side by side and timing `moe_route` itself (an end-to-end
decode cannot resolve a router-only change — the router is one launch of ~13 per layer, so a null
e2e result would be uninformative rather than reassuring):

```
moe_route @ nE=8 (mixtral-class, order-alternated, best-of):
  MOE_MAX_E=256   4933 ns/launch
  MOE_MAX_E=512   5120 ns/launch    ratio 1.038×
```

Within the 5% budget, and at the resolution limit: at ~5 µs the measurement is dominated by launch
overhead, and the unused depot bytes are never touched, so the physical expectation is ~0. One launch
per layer per token makes it immaterial end-to-end either way.

**A first pass at this measured it WRONG — worth recording.** Running old-then-new once gave
**0.872×**, i.e. the strictly-larger kernel apparently 13% *faster*, which is impossible. The cause
was GPU clock ramp: whichever arm ran second won. Alternating the order fixed it. If you redo this
measurement, alternate the arms — a single-order A/B on this box measures the ordering, not the
kernel.

> `build_ptx.sh` prints "wrote … (64288 bytes)" for a 64287-byte file — an off-by-one in its own
> reporting, not a content difference. Trust `sha256sum`, not that line.
