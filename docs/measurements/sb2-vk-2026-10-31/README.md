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
