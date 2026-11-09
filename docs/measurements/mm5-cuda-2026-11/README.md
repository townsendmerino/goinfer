# MM5, CUDA's half: KV priced from the operations' declared rows (nobara-sb01, 2026-11-08)

**What.** Step 3 of `docs/completed/task-forward-architecture-2026-10.md` §4.9b ("CUDA, nobara-sb01, once #739 is merged"):
- `decoder.Model.cudaKVParts` and `cudaKVRowBytes` price CUDA's KV from each layer's declared per-position rows (`Model.LayerBuffers`),
  as Metal's half (#741) does. They no longer read `KVSrcAtResident`, the MLA parameters or `WindowedKVLayer`.
- CUDA's build holds each layer's declared rows to what its kernels address (`kvCheckDeclared`, before the pools are sized) and refuses
  a mismatch.

Nothing computed changes. CUDA's step scratch stays with OP4 on CUDA, as §4.9b says.

## Where the CUDA half reads §4.9b differently (decided here, recorded before the gates were read green)

1. **MLA's latent row on CUDA.** The shared declaration marks the latent "f32 at any KV precision", which is Metal's rule. CUDA's MLA kernels
   have f16 twins: deepseek_v2, deepseek_v3 and kimi_k2 are in `cudaKVF16Delivered`, so CUDA holds the latent at f16 under an f16 cache.
   CUDA's sizing rule (`cudaKVRowBytes`) is therefore: a row declared at the KV precision takes int8 codes plus one f32 scale per head, or
   2 or 4 B an element. A row declared f32 follows an f16 cache and has no int8 form. The rule keys on the declared precision, not on a
   family, so G-MP4 holds, and the build refuses an int8 layer that declares an f32 row.
2. **The pools are held to the declaration, not sized through it.** Metal sizes each pool from the width `kvCheckRows` returns. CUDA's
   pools are sized from the layer's kernel geometry (`kvRowBytes`), as before, and `kvCheckDeclared` refuses the build unless that
   geometry is the declared row: width, heads, precision class and window. The guarantee is the same: a declaration the kernels do not
   address refuses the build instead of sizing a pool.
3. **The windowed context.** §4.9b's 256 positions cross every fixture's window. That is what Metal needs, since it holds the full context
   on sliding layers. CUDA's windowed cache engages only past the window plus `decoder.WindowedKVSlack` (512). At 256 positions CUDA's
   sliding layers hold the full context too, which G-MP2's main pass reads. The engaged windowed cache is read separately, at production's
   slack, from copies of mistral-tiny-window, gemma2-tiny and cohere2-tiny whose config allows 2,048 positions (their weights linked), at
   1,024 positions, with no rollback-grant headroom. Before this, nothing on the device held the engaged windowed cache to the plan:
   kv4b's G3 comment pointed at a `TestKVBytes_agreement` that does not exist.

## Gates (all tier 0, seconds)

- **G-MP4, the scan** (`TestMM5_noFamilySwitchInPricing`): `cudaKVParts` and `cudaKVRowBytes` join the list.
  - Red on `main`: `cudaKVParts` names `KVSrcAtResident` and `.mla`, `cudaKVRowBytes` names `.mla` (`tier0-red.log`).
  - Green with the change.
- **G-MP3, the hand table** (`TestMM5_handTableCUDA`): the six fixtures' KV per block at -kv f16 and at f32, llama-tiny at int8, and one
  windowed gemma2-tiny slot at 1,024 positions, each written as arithmetic on the config. It is green on `main` and with the change: the
  old pricing already gave these figures, so the table guards the rewrite rather than catching a defect.
- **G-MP2, the allocation and the kernels** (`TestMM5_cudaPoolsAreTheDeclaredRows`, `TestMM5_cudaBuildRefusesADeclarationItsKernelsDoNotAddress`,
  backend/cuda, the shared timing lock):
  - Every census fixture layer D admits at int4, built resident on CUDA at -kv f16 and -kv i8, at two slots and 256 positions, plus the
    three windowed copies at f16 and f32: each pool holds its declared rows over the pool's rows, int8's scales hold one per row and
    declared head, a layer that declares no row holds no pool, and the build's whole KV equals the plan for its slots, to the byte.
    218 pools on 39 builds (4 int8, 29 f16, 6 windowed). A build refused on its own declaration fails the test.
  - The refusal test plants ten declarations on llama-tiny, deepseek-tiny and qwen35-tiny: K one element short, V with a head too many, K
    at f32, K windowed on a full layer, a row no pool has, K twice, V missing, the latent at the KV precision, the latent one element
    short, and a DeltaNet layer declaring K and V. All ten are accepted on `main` (`tier0-red-device.log`) and refused with the change.
    The declarations as built pass.
- **G-MP1, the plan:** MM1's decision census (amd64 natively, arm64 under qemu-user), SB2's admission census and CUDA's admission and
  memory censuses, read without `-update`. Readings below.
- **G-MP5:** this change moves no computation: it changes how the pricing reads its rows, and adds one check before the pools are
  sized. The CUDA KV suites (`TestKV4b_*`, `TestResidentKVBytes_*`, `TestKV8*`, `TestKV16*`, `TestKVPool*`, `TestWindowed*`, the
  ragged-step rows test and the unpinned-experts identity) pass with the check in place: 52 PASS, and the skips are the heavy gates.
- **Plants** (each applied, run and reverted; `plant-*.log` keeps each run's last lines):
  - **A K/V row declared one element short** (`program.Buffers`).
    - Red on G-MP3: for example nemotron-tiny's f32 block is 31,744 B against 32,768, and llama-tiny's int8 block 19,968 against 20,480.
    - Red on G-MP2: the builds refuse their own declaration ("the program declares k rows of 31 elements … the kernels address 32").
  - **The null blocks counted twice** (`cudaKVParts`).
    - Red on G-MP3: for example nemotron-tiny's f16 block is 32,768 B against 16,384.
    - Red on G-MP2: mistral-tiny-window f16, windowed, the build holds 1,032,448 B and the plan 1,081,600.
  - **The windowed term one block short.**
    - Red on G-MP3: gemma2-tiny's slot is 819,328 B against 852,096.
    - Red on G-MP2's windowed pass: mistral-tiny-window f16, the build 1,032,448 B, the plan 934,144.

## Readings

- **G-MP1: nothing moved** (`g-mp1-census.log`, 17:30 to 17:37 PST, read without `-update` on the change, the build check
  included).
  - MM1's decision census is `2c04c49a…` on amd64 and `714f1f00…` on arm64 (under qemu-user), the hashes `main` holds.
  - SB2's admission census passes on both classes.
  - CUDA's admission census (412 lines) and memory census (748 lines) pass.
  - No byte figure moved: CUDA's pricing already priced the rows the declarations give, and this change makes it read them.
- **G-MP2, G-MP3, G-MP4 and the plants:** as above. Logs: `tier0-red.log`, `tier0-red-device.log` and `plant-*.log`.
