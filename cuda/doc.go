// Package cuda is an OPT-IN, cgo-free native-CUDA backend for goinfer's resident decode path.
//
// The design splits in two, and the split is the whole strategic point:
//
//   - Layer A (driver plumbing) — a 1:1 shim over github.com/eitamring/gocudrv:
//     dlopen libcuda.so.1 at runtime, so `CGO_ENABLED=0` and the single-static-
//     binary property hold. All context/alloc/memcpy/module-from-PTX/launch/
//     stream/event calls are here. gocudrv covers this EXCEPT cooperative launch.
//
//   - Layer B (the compute) — the production kernel set in cuda/*.cu, each compiled to PTX by
//     `go generate` (build_ptx.sh: NVRTC, not nvcc — see that script for why; audited modules are
//     pinned to NVRTC 12.6.85) and go:embed'd for driver-side JIT (cuModuleLoadDataEx):
//     GEMV/GEMM decode (gemv_fwd, gemv_w8a8_batched,
//     gemv_w4a8_rn, gemm_w4a8_mma), prefill and attention (prefill_batched, decode_splitkv,
//     attn_block, attn_img_prefill, attn_fused, rope_mrope_prefill), MoE routing (moe,
//     router_f32), other families (deltanet, gptoss_act), and fused/misc (fused_qkv, glue,
//     argmax, lora, layernorm_quant, gelu_quant, and more). kernels.go's //go:embed list is the
//     census (28 modules on 2026-09-25; this comment said 22 and drifted — count there, not here).
//
// This backend grew out of a 2026-07 spike into a single fused decode-layer megakernel, scoped
// in docs/completed/cuda-megakernel-spec.md; the go/no-go read is
// docs/completed/task-cuda-cgofree-spike.md. The megakernel itself never shipped — the spike's
// K1/K3a option (per-op kernel fusion, not one all-in-one kernel) did, as cuda/fused_qkv.cu; a
// K2 (broader fusion) was built, measured ~0%, and reverted. The spec's scaffold
// (cuda/megakernel.cu, never go:embed'd or functional) was deleted in the 2026-09-12 closeout.
//
// Build with `-tags cuda`. BuildResident is live: it builds a resident forward when the driver
// and the arch's features allow, and declines (ok=false) otherwise, in which case the decoder
// falls back to the staged/CPU path. Blank-importing this package is safe either way.
package cuda
