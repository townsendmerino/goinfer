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
  (`pass1-refused-mistral7b-ctx16384-*`). Its KV is f32 on WebGPU: 4 GiB at 16384 beside 4.1 GB of weights. *Corrected the same
  day after Gemini's audit: this line first said "about 2 GiB of f16 KV".* After that refusal 8192 was set by hand in the run
  script (2 GiB of f32 KV), before the 8192 load's own readings were seen. Every other load is at 16384, except the registered 1.5B
  at 4096.
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

## M1′'s held-out pass (2026-10-31 12:30–12:32 PDT): M1′ holds

- **What was graded.** M1′ is registered in the task doc ("SB2-vk", M1′, 08:08 PDT): gpu-allocator's first-fit placement, the
  64 MiB host term, and a 144 MiB device constant fitted on pass 1 (`m1p_fit.txt`). It was judged only on five checkpoints and
  contexts pass 1 did not use (`run-m1p.sh heldout1`).
- **Conditions.** Driver 595.112.07. The void check passed before every measurement, and the timing lock was held exclusively.
  Load average was 1.6–2.1 at every start, and the CUDA context again read 105 MiB.
- **All five loads went resident.**

`heldout1-score.txt` (each error is predicted − nvidia-smi's rise over the load, in MiB):

| load | rise | **M1′** | M1 as graded | M0 |
|---|---|---|---|---|
| qwen2.5-0.5b (the Q6_K file, int4 at load) @8192 | 847 | **+1.0** | −143.0 | −329.9 |
| qwen3-1.7b (the Q8_0 file, int4 at load) @16384 | 5496 | **+0.8** | −655.2 | −854.7 |
| gemma-3-4b-it Q4_K_M @8192 | 5330 | **−2.0** | −658.0 | −783.0 |
| qwen2.5-coder-7b @12288 | 5981 | **−5.2** | −405.2 | −608.1 |
| seed-coder-8b @8192 | 7091 | **−5.0** | −405.0 | −684.6 |

**Read: M1′ holds.** Every held-out load is within ±64 MiB; the worst is 5.2 MiB, across five checkpoints, 847 MiB to 7.1 GB, and
contexts from 8192 to 16384.
- **Not five new layouts** (Gemini's audit, the same day). Every load was re-quantized to int4 at load. qwen2.5-0.5b and
  qwen2.5-coder-7b match their pass-1 twins buffer for buffer except for the KV, and seed-coder-8b matches mistral-7b on 1,452 of
  1,456 buffers.
- Only qwen3-1.7b and gemma-3-4b are new layouts. A second held-out set of four further architectures is registered in the task
  doc before the planner change ships.

**What it settles.** Vulkan's device use on this card is the build's live buffers placed as gpu-allocator 0.28.0 places them, plus
one 64 MiB host block, plus 144 MiB for the device itself. G-WS3's overhead, which grew with the model, was the block rounding.

**What it leaves open, for SB2-vk's next step:**
- The census has no buffer list, so M1′ has to be restated over the plan's own figures before a discrete line of WebGPU's slot
  census can use it, as the design says.
- It is one card, on one driver and one wgpu-native build. Another box's Vulkan device constant is its own measurement.

## M1′ in the planner and the slot census (2026-10-31 17:15–17:40 PDT; gates registered 12:50 and 13:40)

- **Approval.** Francis approved option (a) at 13:38, relayed by the coordinator. It is a pricing correction, and no guard
  threshold moves.
- **The planner** (`decoder/webgpu_vulkan.go`, `decoder/fitplan.go`'s `deviceLoad`, `decoder/weightbytes.go`):
  - webgpu's dense bytes leave out the host-side tables, as cuda's do;
  - its fixed slack carries the device's 144 MiB and the kept 64 MiB host block;
  - its KV term carries gpu-allocator's block rounding over `webgpuDeviceBuffers`, the build's buffers in its own creation
    order.
- **The slot census** predicts `newKVSlot` over the same list (`vkSlots`), against the profile's figure plus nobara's 105 MiB
  CUDA context.
- All runs were CPU only, under the timing lock held shared, once mac-tack5's night 2 had released it.

| gate | read |
|---|---|
| G-VP1, red first (`gvp1-red.log.gz`, `gvp1-green.log.gz`) | **red then green**: on llama-tiny, webgpu's dense bytes carried the 17,408-byte untied token table before the change, and do not after |
| G-VR0 | **pass**: the census's packer and the planner's both give `score_vk1.py`'s M1′ to the byte on G-VK1's six lists |
| G-VR2, red first (`gvr2-red.log.gz`, `gvr2-green.log.gz`) | **red then green**: 382 discrete lines restated (298 grant more than one slot); the 573 unified lines byte-identical; the capacity −1 GiB plant changes discrete lines only; the tight plant turns all 298 to one slot; the headroom plant still does |
| G-VR1 (`vr1.txt`) | **holds**: on the eleven recorded loads the list is within 5.3 MiB of the ledger, and M1′ over it within 6.2 MiB of the card's rise |
| G-VP2 (`vr1.txt`) | **holds**: the webgpu plan's priced bytes are within 6.2 MiB of the card's rise on all eleven |
| G-VR3 (`vr1.txt`) | **matches**: the restated prediction for G-WS3's realClamp loads is 4 of 4 (1.5B) and 1 of 4 (7B), as G-WS3 granted |
| G-VP3 | **pass**: in the MM1 decision census 1,151 of 29,637 groups change, every one a webgpu group. No slot decision changes on the tiny fixtures; only the discrete lines' text does. |

**Before and after, at 16384 positions (one slot, int4):**

| model | the plan before (G-WS3's D) | the plan after | the card (G-VK1) |
|---|---|---|---|
| qwen2.5-coder-1.5b q4_k_m | 2,045 MiB (83 short) | 2,128 MiB | 2,128 MiB |
| qwen2.5-7b q4_k_m | 6,333 MiB (161 short) | 6,488 MiB (6 short) | 6,494 MiB |

**The arm64 golden** was taken from CI's gpu-darwin runner (run 41261159044, job 124714576112): its 382 discrete lines are restated, and its unified lines and every slot decision are unchanged.

## Held-out set 2 (2026-10-31 17:31–17:35 PDT card, 17:37 CPU; registered 17:30): M1′ holds, the plan does not

- **Why a second set.** Gemini's audit found three of the five held-out loads near-duplicates of fit-set layouts, because every
  load was int4 at load. This set is four further architectures, all int4 at load, one slot.
- **Conditions:** exclusive timing lock, every void check clean. Load average was 1.6–2.6 at each load's start (`heldout2-*-card.txt`). *Corrected: this line first said 0.3, which was `uptime` before the lock was taken.*
- **Gemma 2 is not scored,** as registered: WebGPU has no attention-logit softcap, so it does not go resident.

| load | rise MiB | M1′ (recorded list) | list vs ledger | the plan vs the card |
|---|---|---|---|---|
| codegemma-2b (Q8_0 file, int4) @8192 | 2117 | −1.0 | +2.8 | −1.0 |
| gemma-4-E2B (q4_0 file, int4) @8192 | 1999 | +1.0 | **+383.7** | **+513.0** |
| Qwen3.5-9B (Q4_K_M file, int4) @4096 | 5409 | −7.0 | −61.7 | −7.0 |

**Read, by the registered rule: fails.** M1′ itself holds on every scored load (worst 7 MiB, `heldout2-score.txt`), but the plan
misses Gemma 4 E2B by 513 MiB, beyond 96 (`vr1.txt`). So the planner change does not ship.

**The mechanism, from the ledgers:**
- **Gemma 4 E2B** holds KV caches on 15 of its 35 layers (30 buffers, 288 MiB): its later layers share KV.
  `webgpuDeviceBuffers`, and the planner's `ResidentKVBytes("webgpu")` before this change too, price a cache on every layer.
- **Qwen3.5-9B** holds caches on its 8 attention layers only, which the list gets right. It also holds 50 MiB of DeltaNet
  recurrent state in 48 buffers, which the list leaves out.

M1′'s placement was not the error in either case. G-VR1's verdict line now also counts set 2's loads; on its registered eleven it
holds, as read at 17:40.

## The list fix, and held-out set 3 (2026-10-31 17:48–17:56 PDT): set 3 void, the fix fits sets 1 and 2

- **The fix** (`decoder/webgpu_vulkan.go`, `webgpuLayerCaches`):
  - K and V caches only on the attention layers that own one (Gemma 4's KV-shared layers alias their source's);
  - the latent cache on MLA layers;
  - the conv window and recurrent state on DeltaNet and Mamba-2 layers.
- **The planner.** `ResidentKVBytes("webgpu")` prices the same caches. The plan's rounding term is the blocks less the weights and
  the context, so the recurrent state, the scratch and wgpu's own buffers ride on it. The block packing over weights and KV at the
  planned context is unchanged.
- **Held-out set 3 is void: none of its three architectures went resident on WebGPU,** so no set-3 reading exists. Each checkpoint
  was copied from `/srv/models` and deleted after its load.
  - Phi-3-mini declines: its activation outliers rule out the int8 activation path.
  - Granite-4.0-H-tiny declines at the SSM bring-up gate by default (`heldout3-default-*`). With the gate set, as registered in
    the amendment before any reading, it declines on NoPE, which WebGPU does not implement.
  - Nemotron-Nano-9B-v2's weight buffers do not fit the 8 GB card.
  - The one other unseen architecture in the archive that fits the card, command-r7b (cohere2), is 16.08 GB, over the 15 GB
    disk limit for this work.
- **The fit (`vr1.txt`).** Gemma 4 E2B and Qwen3.5-9B are now fit data, not held out: the fix was read off their ledgers. On all
  fourteen scored loads of sets 1 and 2:
  - each list is within 11.5 MiB of its ledger;
  - M1′ over it and the webgpu plan are within 7.0 MiB of the card;
  - Gemma 4 E2B moves from +513 to +1.0 MiB, and Qwen3.5-9B's list from −61.7 to −11.5.
- **Not modelled:** WebGPU uploads some f32 mixer weights (Granite's and Nemotron's Mamba projections) re-quantized to int8. No
  scored load exercises that.

## Held-out set 4 and what ships (2026-10-31, option (c))

- **Set 4 is void.** InternLM2.5-1.8B-chat went resident, with M1′ at +1.0 MiB. Granite-3.3-2B-instruct does not load, because
  goinfer does not support its `residual_multiplier` of 0.22. One scored load is fewer than the two the rule asks.
- **Both downloads were deleted after their loads.**
- **The first pass was skipped by the harness** (`heldout4-skipped-*`): its fixture helper wants a single `model.safetensors`.
  The probe now loads the checkpoint directly.
- **What ships is the slot census restated with M1′ over the fixed buffer list.** The list lives in the decoder behind
  `goinfer_testhooks`.
- **The planner change does not ship.** It stays on branch `sb2-vk-plan-2026-10-31`: the host-side tables, M1′ in webgpu's
  `deviceLoad`, and the KV only on cache-owning layers. It fits all fourteen scored loads within 7 MiB, but no held-out pass of
  the list fix exists yet. The MM1 census golden is therefore unchanged here, and the G-VP rows above describe that branch.

**Which run each figure comes from.**
- **This branch's `vr1.txt` and `vr1.log`:** the probe as committed here, over G-VR1's eleven registered loads and G-VR3. The plan is
  unchanged on this branch.
- **The planner branch's run of the probe:** the plan's figures (G-VP2) and the list and plan figures quoted for held-out sets 2
  and 3 above. They stay with that branch.
- **The buffer list here** carries K and V caches by owning layer. An MLA layer's latent cache and a hybrid's recurrent state are
  listed only on the planner branch.
