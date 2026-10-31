# SB2-vk, G-VK1: Vulkan's device use against the build's buffer list (nobara-sb01)

The design and the rule are registered in `docs/tasks/task-subsystems-2026-10.md`, "SB2-vk". This README is written before the run
with M1's source reading. The run's results are appended below it.

## M1's source reading (2026-10-31, before the run)

Sources: the crates.io tarballs of wgpu-hal 29.0.3, wgpu-core 29.0.3, wgpu-types 29.0.3 and gpu-allocator 0.28.0, and the GitHub tarball
of wgpu-native v29.0.1.1.

- **The build.**
  - `gpu/go.mod` pins `oliverbestmann/webgpu` v1.37.1. Its `libs-linux` module ships a static `libwgpu_native.a`.
  - The binding's `update-wgpu.sh` sets `VERSION="v29.0.1.1"`, and that string is in the archive. The README's v29.0.0.0 is stale.
  - The crate paths embedded in the archive name wgpu-core/hal/types 29.0.3, gpu-allocator 0.28.0 and ash 0.38.0. There is no
    gpu-alloc.
- **Hints.**
  - wgpu-native's `map_device_descriptor` (`src/conv.rs`) sets `memory_hints: Default::default()`, which is `Performance`
    (`wgpu-types src/device.rs`). Neither the C header nor the Go binding has a field to change it.
  - goinfer's `New` (`gpu/internal/kernels/gpu.go`) sets none.
- **Block sizes.**
  - `AllocationSizes::from_memory_hints` (`wgpu-hal src/lib.rs`): Performance gives device-local blocks of 128 MiB growing to
    256 MiB, and host-visible blocks of 64 MiB growing to 128 MiB.
  - gpu-allocator grows a block as `min(min << min(count, 7), max)` over the type's active general blocks
    (`gpu-allocator src/lib.rs`, `get_memblock_size`).
- **Placement** (`gpu-allocator src/vulkan/mod.rs`, `MemoryType::allocate`):
  - A request larger than the current block size gets a block of its own, of exactly its size.
  - Otherwise the request is placed best fit into the general blocks, newest first, or opens a new block.
  - A type's last general block is never freed.
- **wgpu-hal never asks for a dedicated allocation** (`src/vulkan/device.rs`, `create_buffer`: `GpuAllocatorManaged`, `linear: true`).
- **Where a buffer goes, by usage:** `MAP_READ` → `GpuToCpu` (host-cached system memory); `MAP_WRITE` → `CpuToGpu`, which prefers
  `DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT`; anything else → `GpuOnly`.
- **wgpu-core's sizes and buffers** (`src/device/resource.rs`).
  - It pads every size to 4 bytes and adds `COPY_DST`.
  - Each device creates a 512 KiB zero buffer, an external-texture params buffer of about 208 B, and a 24 B indirect-validation
    buffer.
  - Each `WriteBuffer` stages through a fresh `CpuToGpu` buffer of the data's size.

**Not verified from source:**
- the driver's `req.size` and `req.alignment` for a buffer (M1 assumes 256 B);
- the exact size of the params buffer;
- whether the storage buffers share the zero buffer's memory type.

The memory types themselves are read by `vulkaninfo` in the run (`<pass>-vulkaninfo-memory.txt`).

## G-VK1, pass 1 (2026-10-31, 07:20–07:21 and 07:52–07:53 PDT): no model holds

- **Conditions.** Driver 595.112.07. The card held only the desktop shell's clients before every measurement, and the void check passed
  each time. The timing lock was held exclusively. Load average was under 2.3 at each start.
- **Split pass.** The pass ran in two parts, because mac-tack5's tack rows took nobara's lock in between.
- **Mistral at 8192.** Mistral 7B at 16384 positions does not fit on WebGPU: the KV allocation was refused at layer 27
  (`pass1-refused-mistral7b-ctx16384-*`). Its 8 KV heads need about 2 GiB of f16 KV beside 4.1 GB of weights. It was run at 8192.
  That context was chosen by the fit, before any reading of it. Every other load is at 16384, except the registered 1.5B at 4096.
- **The CUDA context.** It reads 105 MiB on every load (nvidia-smi 532 → 637 MiB while the probe held it).
- **Memory types.** `vulkaninfo`: the RTX 2070 SUPER has a 246 MiB `DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT` type (the BAR heap),
  so M1's host term applies. The capture also lists llvmpipe's types, which do not affect the read: NVIDIA's own types decide it.

`pass1-score.txt` (each error is predicted − measured, in MiB; the bar is ±64):

| load | buffers | rise over the load | M0 (ledger) | M1 | M2 | M1 without the host term |
|---|---|---|---|---|---|---|
| qwen2.5-coder-0.5b | 1240 | 1104 | −394.9 | **−144.0** | (fit) | −208.0 |
| qwen2.5-coder-1.5b | 1444 | 2128 | −301.6 | **−144.0** | +371.4 | −208.0 |
| qwen2.5-coder-1.5b @4096 | 1444 | 1615 | −460.6 | **−143.0** | +212.4 | −207.0 |
| llama-3.2-1b | 736 | 2128 | −328.2 | **−144.0** | −620.4 | −208.0 |
| mistral-7b @8192 | 1456 | 6485 | −561.0 | **−405.0** | +128.4 | −469.0 |
| qwen2.5-7b | 1444 | 6494 | −673.1 | **−406.2** | (fit) | −470.2 |

**Read: no model holds.** SB2-vk stays open, and the discrete census lines keep their G-WS3 band.
- **M0** misses by 302–673 MiB.
- **M2's fit is degenerate.** Its two fitting loads differ by only 204 buffers, so it gives a = −1,296 MiB and b = 1.4 MiB a buffer,
  and it misses by 128–620 MiB.
- **M1 misses by a structured amount,** not scatter:
  - **−144 MiB on all four loads of 2 GB or less,** constant to 1 MiB across three architectures and two contexts. That is the
    term M1 left out by design: the Vulkan device's own memory (its context, pipelines and descriptor pools).
  - **−405/−406 MiB on both 7Bs,** in two architectures. *Corrected 08:08 the same morning: this was the scorer's error, not
    a mechanism.* The graded packer (`_pack`) placed each buffer in the fullest block that fits. gpu-allocator walks the open
    blocks newest first and takes the first that fits (`MemoryType::allocate`; the source reading above says so). With that walk
    (`_pack_ff`), the 7Bs open one more 256 MiB block each, and the residual is 149 and 150 MiB, like the others' 143–144
    (`m1p_fit.txt`). The graded read stands as registered: M1 as scored does not hold.

**What follows: M1′** (the coordinator's pick, 08:10): the placement as gpu-allocator walks it, the host term, and a device
constant of 144 MiB fitted on this pass (`m1p_fit.txt`: residuals 143.0–150.2 MiB, median 144). Registered in the task doc
before its held-out loads, and judged only on them.
