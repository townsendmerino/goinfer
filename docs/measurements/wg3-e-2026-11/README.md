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

## The diagnostic (2026-11-23, after the decision): nothing stands out

The one mechanism on record for WebGPU's E-model gap is S12's (`docs/completed/task-multimodal-support-2026-10.md`,
"Resolved the same day: no kernel is off"): the fused RMSNorm-and-quantize of layer 0's input rounds a few int8 codes one
step differently from the CPU's float arithmetic, and the coarse codes amplify it. It was checked here on both adapters
with a throwaway test (not committed): production's widened `RmsnormQuantWGSL` on six real embedding rows (tokens 2, 105,
1000, 4000, 20000, 100000) and layer 0's attention norm, against a float64 host quantization of the same row.

| model | nobara, Vulkan | the Mac, Metal |
|---|---|---|
| E2B (1,536 wide) | 0 of 9,216 codes differ; scales equal to 1e-7 | 0 of 9,216; the same scale bits as Vulkan |
| E4B (2,560 wide) | 2 of 15,360 (both at token 2) | not run here (no plain int4 sidecar on the Mac); S12 read 2 of 2,560 at token 2 |

The first fused kernel is bit-identical across the two adapters, so it does not explain why Vulkan's E-model agreement
sits lower. Placing the first layer where Vulkan's error grows past Metal's needs a per-layer CPU capture, which
`decoder.ForwardSubCaptureLogitsForTest` does not cover for Gemma 4 ("a layer has an operation its contributions do not
cover"); that is past this diagnostic's hour. No precision bug found; nothing is owed on the evidence in hand, and the
E-models stay resident by the decision above.
