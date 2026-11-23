# WG3-temp: the attention temperature on WebGPU (Ministral 3)

Gate: `TestMinistral3ResidentParityWebGPU` (`backend/webgpu/attn_fidelity_test.go`), ministral3-tiny (original context
8, so 24 of its 32 positions carry temperatures 1.14-1.32), resident at int8int8 against the CPU int8int8. Worst relL2
2.0e-7 on mbp2's Iris Plus 645 and 2.06e-7 on nobara's RTX 2070 SUPER (Vulkan), against a 1e-4 bar; with the
temperature dropped the worst relL2 past position 7 is 0.0134, and positions 0-7 stay bit-identical. The batched
prefill's own decline is `kernels.TestRunModelToPrefill_declinesAttnTemp`.

**No real checkpoint exercises a temperature other than 1 here.** Ministral-3-3B's `original_max_position_embeddings` is
16,384, so `1 + beta·ln(1 + floor(pos/16384))` is exactly 1 below position 16,384, and WebGPU's f32 KV cache caps a
conversation at 16,384 positions. A real reading of the temperature would need the f16 cache past position 16,384,
teacher-forced against a CPU reference that long; it has not been run. The real checkpoint's gate is its composition at
temperature 1: `TestResidentCaptureParityWebGPU` per sublayer at int4 (the int8int8 load did not fit the 8 GB card at the
test's default context and declined cleanly), queued on nobara.
