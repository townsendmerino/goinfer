# MM5b: what CUDA's, WebGPU's and the CPU's decode scratch hold, read beside the program's declaration (2026-11-15)

Question (row MM5b, `docs/tasks/task-forward-and-memory-2026-11.md`): can the decode scratch of the backends that price none from the program
(WebGPU, CUDA, the CPU) be priced from the operations' per-row declarations (`program.Buffers`' `PerRow` unit and `program.HeadBuffers`), as
Metal's is since #772? This is a read, not a gate: nothing here changes a price. Tier 0, a Mesa software Vulkan adapter (an Intel Iris Plus 645)
and the CPU; **CUDA was not read** (no device in this read).

## WebGPU: the scratch line a build holds, against the declared rows

The ledger's `scratch` line (every buffer a build and a generation lease on it) after `decoder.Load` with `Backend: "webgpu"`, int4, context 256, and
after a 4-token generation, beside the sum of the declared `PerRow` rows (each role at the widest layer that declares it, f32, as Metal's rule
sizes them) and the declared head rows. Six fixtures built resident of the eight tried (gemma2-tiny is not resident on WebGPU; gpt2-tiny does not load from this path).

| fixture | scratch line after the build (B) | after a 4-token generation (B) | declared layer rows, widest layer per role (B) | declared head rows (B) | declared total / ledger |
|---|---|---|---|---|---|
| testdata/llama-tiny | 12,792 | 12,792 | 1,792 | 1,792 | 28% |
| testdata/qwen3moe-tiny | 11,816 | 11,816 | 1,664 | 2,816 | 38% |
| testdata/deepseek-tiny | 14,112 | 14,112 | 2,496 | 1,280 | 27% |
| testdata/nemotron-tiny | 9,936 | 9,936 | 2,848 | 1,792 | 47% |
| selftestdata/qwen35-tiny | 17,656 | 17,656 | 3,648 | 1,792 | 31% |
| testdata/gemma4-moe-kv-tiny | 34,416 | 34,416 | 20,864 | 1,792 | 66% |

The scratch line does not move after a generation: WebGPU's decode scratch is leased at build. It is **1.5 to 3.7 times what the declared rows sum
to** (the declared rows are 27 to 66 percent of it), because most of what WebGPU leases on `scratch` is the per-dispatch parameter buffers its
runner creates for each layer (`backend/webgpu/internal/kernels/decodelayer.go`: the `rmsn-p`, `rope-p`, `attn-p` buffers and the like),
which no operation declares. WebGPU's plan has no scratch line of its own: the memory gate names `webgpuSlotHeadroom` (384 MiB, the margin
a further slot leaves for per-call prefill scratch and the lazily grown runners) for it.

## The CPU: `newDecodeScratch` against the declared rows

`decoder/scratch.go:newDecodeScratch` allocates the host float buffers one stream's decode reuses. Over the 56 census fixtures that load,
its float buffers equal the declared layer rows plus head rows on **2** and range from 0.17 to 1.46 of them on the rest: the CPU
walker reuses one `norm` and one `sub` where the declaration lists each operation's own rows, holds no DeltaNet, Mamba or KDA rows in this struct,
and sizes q and ctx at the widest head count. It is host memory, which no guard prices.

## CUDA

Not read. CUDA leases its decode scratch on the same `scratch` line (`backend/cuda`, eight files name `mem.LineScratch`) and keeps the same
384 MiB margin (`kvSlotsFit`); the registration of its step scratch is OP4 on CUDA's, by the 2026-11-06 split.

## What this says

A declared-rows price for these three would price **activations only**, which is 27 to 66 percent of what WebGPU leases and a different set from
what the CPU holds; to equal the ledger (Metal's G-MS2) each backend's decode would have to allocate its activations from a list the way Metal's
`decodeShape.buffers` does, and name its own terms (the parameter buffers) beside it. That is a rewrite of WebGPU's runner build and of CUDA's
decode build, not a pricing step. Separately, charging any such figure in `NeedBytes` or narrowing a margin (`webgpuSlotHeadroom`, CUDA's 384 MiB)
is a guard threshold, which needs Francis's word.

## The probes (throwaway, not committed)

WebGPU (`backend/webgpu`, tag `gpu`): for each fixture, `decoder.Load(dir, decoder.Options{Backend: "webgpu", Quant: "int4", ResidentContext: 256})`;
`rd := m.ResidentForwardForTest().(*residentDecoder)`; `rd.c.Ledger().Usage().ByLine[mem.LineScratch]` after the load and after
`m.Generate(ctx, []int{1,2,3,4,5}, 4, SamplingParams{})`; declared rows: for each layer `m.LayerBuffers(i)`, the `PerRow` ones, the widest `Elems`
a name has, times 4, summed; head rows: `program.HeadBuffers(m.Program(), program.HeadShape{Hidden: cfg.HiddenDim, Vocab: cfg.VocabSize})`, times 4.
CPU (`decoder`): `newDecodeScratch(arch)`, the sum of `4*len` over `h, norm, sub, sub2, q, k, v, ctx, gate, up, logits` and the MoE buffers, against the
same declared sums from `layerBuffers(arch, l)`.

## CUDA: the device read (2026-11-17)

Machine: an RTX 2070 SUPER (8 GiB), NVIDIA driver 595.112.07, the exclusive timing lock held, the checkpoints read from the local `~/models`, int4,
`ResidentContext` 512, greedy, the tree at main after #1039. The ledger's `scratch` line (`r.dev.Leaser()`'s ledger, `mem.LineScratch`) after
`decoder.Load` and after two generations (a 5-token prompt, then a 300-token prompt, 4 tokens each), beside the declared rows (`Model.MetalDecodeRowWidths`:
each row at its widest layer, f32, plus the head rows) and the margin's decode allowance (`decodeAllowanceBytes`, 16 MiB, which `ctxMargin` holds for
decode's transients). The probe was a throwaway test in `backend/cuda`; it is not committed.

| checkpoint | scratch after the build (B) | after the 5-token prompt | after the 300-token prompt | declared rows, widest layer per role (B) | scratch / declared | of the 16 MiB allowance |
|---|---|---|---|---|---|---|
| tinyllama-1.1b-chat | 490,416 | 490,416 | 490,416 | 216,064 | 2.27 | 2.9% |
| qwen2.5-0.5b-instruct | 1,376,400 | 1,376,400 | 1,376,400 | 665,600 | 2.07 | 8.2% |
| qwen3-0.6b-bf16 | 1,376,536 | 1,376,536 | 1,376,536 | 669,184 | 2.06 | 8.2% |
| qwen3-1.7b-bf16 | 1,430,808 | 1,430,808 | 1,430,808 | 706,048 | 2.03 | 8.5% |
| internlm2-1_8b | 981,832 | 981,832 | 981,832 | 484,864 | 2.02 | 5.9% |

On the committed tiny fixtures (llama-tiny, qwen3moe-tiny, deepseek-tiny, nemotron-tiny, glm-tiny, olmo3-tiny; the others are not resident on CUDA at
int4 or int8) the line is 53,640 to 57,324 B, nearly the same on every one (declared rows 3.5 to 4.9 KiB): at that size a fixed part of about 50 KB
dominates.

What it says, as a slice (five dense checkpoints of 0.5 to 1.8 B parameters, one card, int4, not a 7 to 9 B model):
- CUDA's decode scratch is leased at build and **does not move** through two generations, including one off a 300-token prompt.
- It is **2.0 to 2.3 times the declared rows** (not 1.5 to 3.7, WebGPU's spread): a near-constant factor, which suggests a second copy of some rows (the
  logits are the widest) and a small fixed part, not per-dispatch buffers. The factor falls toward 2 as the model grows. That is a read, not a derived term.
- It is **3 to 9 percent of the 16 MiB decode allowance** the margin holds, so on these checkpoints the allowance is not what decode leases but
  what the margin keeps for the transients H6-R found none of; a 7 to 9 B checkpoint was not read (8 GiB card, the weights leave little room beside it).
- Nothing was charged, no margin was narrowed. Registering a CUDA list the way WebGPU's runner has one is the next step if the owner wants the figure;
  `docs/tasks/task-forward-and-memory-2026-11.md`, row MM5b, has the state.

## WebGPU on the same card (2026-11-17)

`TestMemGate_webgpu` on the RTX 2070 SUPER (Vulkan, the exclusive lock, `-tags=gpu,goinfer_testhooks`): **244 builds held, 39 options declined by
WebGPU's own build** (the same counts as the Mesa software adapter), the plan's decode scratch covers all 244 and equals the build's list on each,
and `TestMemGate_webgpuPlantsThroughTheTable` passes.
