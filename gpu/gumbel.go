//go:build gpu

package gpu

// Device-side temperature-only sampling by Gumbel-max on WebGPU (R7b, docs/tasks/red-october.md).
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

const gumbelCommonWGSL = `
fn mulhi32(a: u32, b: u32) -> u32 {
  let al = a & 0xFFFFu; let ah = a >> 16u;
  let bl = b & 0xFFFFu; let bh = b >> 16u;
  let ll = al * bl; let lh = al * bh; let hl = ah * bl; let hh = ah * bh;
  let mid = (ll >> 16u) + (lh & 0xFFFFu) + (hl & 0xFFFFu);
  return hh + (lh >> 16u) + (hl >> 16u) + (mid >> 16u);
}

// Philox4x32-10. Must match decoder.philox4x32 (Random123 known-answer vectors).
fn philox(c0_: u32, c1_: u32, c2_: u32, c3_: u32, k0_: u32, k1_: u32) -> vec4<u32> {
  var c0 = c0_; var c1 = c1_; var c2 = c2_; var c3 = c3_;
  var k0 = k0_; var k1 = k1_;
  for (var r = 0u; r < 10u; r = r + 1u) {
    if (r > 0u) { k0 = k0 + 0x9E3779B9u; k1 = k1 + 0xBB67AE85u; }
    let hi0 = mulhi32(0xD2511F53u, c0); let lo0 = 0xD2511F53u * c0;
    let hi1 = mulhi32(0xCD9E8D57u, c2); let lo1 = 0xCD9E8D57u * c2;
    let n0 = hi1 ^ c1 ^ k0; let n2 = hi0 ^ c3 ^ k1;
    c0 = n0; c1 = lo1; c2 = n2; c3 = lo0;
  }
  return vec4<u32>(c0, c1, c2, c3);
}

// log(1 - w) for w in (0, 0.5): a polynomial where log's absolute error would swamp the result.
fn log1p_neg(w: f32) -> f32 {
  if (w < 0.25) {
    var p = 1.0 / 12.0;
    p = p * w + 1.0 / 11.0;
    p = p * w + 1.0 / 10.0;
    p = p * w + 1.0 / 9.0;
    p = p * w + 1.0 / 8.0;
    p = p * w + 1.0 / 7.0;
    p = p * w + 1.0 / 6.0;
    p = p * w + 1.0 / 5.0;
    p = p * w + 1.0 / 4.0;
    p = p * w + 1.0 / 3.0;
    p = p * w + 0.5;
    p = p * w + 1.0;
    return -w * p;
  }
  return log(1.0 - w);
}

fn gumbel(h: u32) -> f32 {
  var e: f32;
  if (h < 0x80000000u) {
    let w = (f32(h) + 0.5) * 2.3283064365386963e-10;
    e = -log1p_neg(w);
  } else {
    let v = (f32(~h) + 0.5) * 2.3283064365386963e-10; // = 1 - w, exactly
    e = -log(v);
  }
  return -log(e);
}

fn better(ak: f32, ai: i32, bk: f32, bi: i32) -> bool {
  if (ai < 0) { return false; }
  if (bi < 0) { return true; }
  return ak > bk || (ak == bk && ai < bi);
}

var<workgroup> sk: array<f32, 256>;
var<workgroup> si: array<i32, 256>;
`

const gumbelStage1WGSL = gumbelCommonWGSL + `
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> prm: array<u32>;
@group(0) @binding(2) var<storage, read_write> bkey: array<f32>;
@group(0) @binding(3) var<storage, read_write> bidx: array<i32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let b = wid.x * 256u + t; // group of 4 consecutive tokens
  let v = prm[0];
  let invT = bitcast<f32>(prm[1]);
  var best = -3.4028234e38;
  var bi: i32 = -1;
  if (b * 4u < v) {
    let r = philox(b, prm[4], prm[5], 0u, prm[2], prm[3]);
    for (var lane = 0u; lane < 4u; lane = lane + 1u) {
      let i = b * 4u + lane;
      if (i < v) {
        let k = logits[i] * invT + gumbel(r[lane]);
        if (k > best) { best = k; bi = i32(i); }
      }
    }
  }
  sk[t] = best; si[t] = bi;
  workgroupBarrier();
  for (var o = 128u; o > 0u; o = o >> 1u) {
    if (t < o && better(sk[t + o], si[t + o], sk[t], si[t])) { sk[t] = sk[t + o]; si[t] = si[t + o]; }
    workgroupBarrier();
  }
  if (t == 0u) { bkey[wid.x] = sk[0]; bidx[wid.x] = si[0]; }
}
`

const gumbelStage2WGSL = gumbelCommonWGSL + `
@group(0) @binding(0) var<storage, read> prm: array<u32>;
@group(0) @binding(1) var<storage, read> bkey: array<f32>;
@group(0) @binding(2) var<storage, read> bidx: array<i32>;
@group(0) @binding(3) var<storage, read_write> outv: array<i32>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let n = prm[6];
  var best = -3.4028234e38;
  var bi: i32 = -1;
  for (var i = t; i < n; i = i + 256u) {
    if (better(bkey[i], bidx[i], best, bi)) { best = bkey[i]; bi = bidx[i]; }
  }
  sk[t] = best; si[t] = bi;
  workgroupBarrier();
  for (var o = 128u; o > 0u; o = o >> 1u) {
    if (t < o && better(sk[t + o], si[t + o], sk[t], si[t])) { sk[t] = sk[t + o]; si[t] = si[t + o]; }
    workgroupBarrier();
  }
  if (t == 0u) { outv[0] = si[0]; } // -1 when nothing was comparable: the caller falls back to the host argmax
}
`
