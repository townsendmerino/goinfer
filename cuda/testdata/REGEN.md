# Regenerating `moe.ptx` (and the other audited 12.6 artifacts)

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

## Record: `MOE_MAX_E` 256 → 512 (2026-08-09)

Raised so Kimi-K2 (384 routed experts) stops declining to CPU on CUDA. See
`decoder/features.go` (`residentBackendMoECap`) and `docs/task-model-family-deepseek-v4-kimi-k3.md`.

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
