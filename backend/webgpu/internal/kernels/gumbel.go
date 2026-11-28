//go:build gpu

package kernels

// Device-side temperature-only sampling by Gumbel-max on WebGPU.
//
// The token is argmax_i( l_i*invT + G_i ), G_i = -ln(E_i), E_i = -ln1p(-w_i), w_i = (h_i+0.5)/2^32, h_i a word
// of Philox4x32-10 keyed by (seed) with counter (i>>2, draw_lo, draw_hi, 0), lane i&3 — the SAME function as
// decoder.gumbelDraw (the reference) and cuda/gumbel.cu. The integer words are bit-identical everywhere; the
// float transform is f32 here, so the device and the host may pick different tokens only where the two best
// keys are within a few ulps. Measured, not assumed (gpu TestGumbelDeviceAgreesWithHost).
//
// WGSL DIFFERS FROM CUDA IN THREE WAYS, each handled here rather than hoped away:
//
//   - No 32x32->64 multiply: mulhi is built from 16-bit limbs (checked against Go's bits.Mul32 in the test).
//   - No log1p, and log's accuracy is only an ABSOLUTE 2^-21 in [0.5, 2] (WGSL spec). E = -log1p(-w) is needed to
//     relative f32 precision for tiny w, and log(1-w) near w=0 has ~no correct digits, so for w < 0.25 it is a
//     12-term polynomial (truncation 0.25^12/13 = 5e-9 relative); for w in [0.25, 0.5) E >= 0.29 and log's
//     absolute error is then a ~1e-6 RELATIVE error, which is fine. The h >= 2^31 half uses -log(v) directly
//     with v = 1-w exactly, as in the CUDA kernel.
//   - No infinities: WGSL gives no defined result for inf/NaN operands. The device path is only ever used when
//     nothing masks the row (no logit processor, bias or penalties), so real model logits are finite; the
//     sentinel "no key yet" is -FLT_MAX, and the agreement test uses a large finite negative for masked entries.
//
// TWO KERNELS: stage 1, one thread per group of 4 tokens (one Philox call, four keys), workgroup argmax, one
// (key, index) per workgroup; stage 2, one workgroup reduces those. Ties go to the LOWEST index, as on the host.
