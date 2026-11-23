# WG3-e: the Gemma 4 E-models on WebGPU over Vulkan (G-S12.1d on nobara, 2026-11-23)

WebGPU has declared `gemma4-e-model` since S12 (`docs/completed/task-multimodal-support-2026-10.md` § G-S12.1), where
G-S12.1d ran on the Mac's WebGPU over Metal only: E2B PASS (+0.74 points), E4B parked (-2.48), and the owner decided to
ship. WG3-e's gate is the same rule on nobara's RTX 2070 SUPER over Vulkan. Nothing in the E-model path changed for this
run; it reads a second adapter.

**Run.** `TestGemma4EModel_webgpuRealNonInferiority` (`backend/webgpu/gemma4_emodel_real_test.go`), unchanged, built at
`4859edaf1`, 2026-11-23 11:30-11:36 PST, under nobara's exclusive timing lock; driver 595.112.07. The int4 sidecars it
reads were written for the run by `cmd/prequant -quant int4 -target webgpu` from the quantization-aware q4_0 GGUFs of
E2B and E4B and the Qwen2.5-Coder-1.5B q4_k_m GGUF in nobara's models directory, into a scratch folder outside it. Every arm reported
`webgpu:vulkan-resident (int4)`.

| model | teacher-forced agreement | free-run passes | delta vs the reference | verdict |
|---|---|---|---|---|
| reference Qwen2.5-Coder-1.5B | 544/566 = 96.11% | 8/8 | | |
| E2B | 409/435 = 94.02% | 7/8 | -2.09 points | AMBIGUOUS (parked) |
| E4B | 401/435 = 92.18% | 8/8 | -3.93 points | AMBIGUOUS (parked) |

Beside the Mac's (Metal) reading of the same rule: reference 95.58%, E2B 96.32% (+0.74, PASS), E4B 93.10% (-2.48).
On Vulkan the reference agrees more closely with the CPU (96.11%) and both E-models less (E2B by 2.3 points, E4B by
0.9). Neither model's free-run count fails the rule.

**Decision (Francis, 2026-11-23, relayed by the coordinator ~14:00 PST): the E-models stay resident on WebGPU on every
adapter, Vulkan included**, both readings recorded here. This extends his October decision, which already shipped
E4B on WebGPU over Metal with its G-S12.1d reading parked (-2.48). A short diagnostic of why the Vulkan agreement sits
lower follows (per-layer hidden states against the CPU on both adapters); its finding is added here.
