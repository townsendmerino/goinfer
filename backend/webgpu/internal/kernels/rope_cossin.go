//go:build gpu

package kernels

// ropeCosSinWGSL defines ropeCosSin(theta) -> vec2(cos θ, sin θ), the rotation every RoPE kernel applies, to about 1e-7 of the
// float64 cos and sin of the same f32 θ, for |θ| < 2^24. The WGSL builtins sin and cos are conformant with an absolute error of 2^-11
// inside [-π, π] and none bound outside, and adapters do return about 2^-14 there; so no kernel calls them
// (TestRopeKernels_noBuiltinSinCos). θ is reduced by π/2 in integer arithmetic (Payne-Hanek against 64 bits of 2/π), so the reduction
// is exact whatever a driver does to float expressions: a floating-point Cody-Waite split loses its precision to reassociation on some
// drivers. The remainder's polynomial runs on [-π/4, π/4], whose truncation is under 2e-9. Every shader that rotates is compiled with
// this prepended; TestRope_matchesCPUAngles holds the kernels to it.
const ropeCosSinWGSL = `
fn ropeMul32(a: u32, b: u32) -> vec2<u32> {
    let a0 = a & 0xffffu; let a1 = a >> 16u; let b0 = b & 0xffffu; let b1 = b >> 16u;
    let p00 = a0 * b0; let p01 = a0 * b1; let p10 = a1 * b0; let p11 = a1 * b1;
    let mid = (p00 >> 16u) + (p01 & 0xffffu) + (p10 & 0xffffu);
    return vec2<u32>((p00 & 0xffffu) | (mid << 16u), p11 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u));
}
fn ropeCosSin(theta: f32) -> vec2<f32> {
    let bits = bitcast<u32>(abs(theta));
    let e = i32((bits >> 23u) & 0xffu) - 150; // |θ| = m·2^e
    var r = abs(theta);
    var q = 0u;
    if (e >= -25 && e <= 0) {
        // y = |θ|·2/π: P = m·(2/π·2^64) as three 32-bit limbs; Y = the 32 bits of y·2^30 mod 2^32.
        let m = (bits & 0x7fffffu) | 0x800000u;
        let lo = ropeMul32(m, 0x4e441529u);
        let hi = ropeMul32(m, 0xa2f9836eu);
        let l1 = lo.y + hi.x;
        let l2 = hi.y + select(0u, 1u, l1 < lo.y);
        let k = u32(2 - e); // the bit of P at which y·2^30 starts, less 32
        let y = (l1 >> k) | (l2 << (32u - k));
        q = y >> 30u;
        var f = i32(y & 0x3fffffffu);
        if (f >= 0x20000000) { q = q + 1u; f = f - 0x40000000; }
        r = f32(f) * 1.4629180792671596e-09;
    } else if (e > 0) {
        // |θ| >= 2^24: the f32 angle carries no fraction of a turn; reduce as floats.
        let j = round(r * 0.63661974668502808);
        r = (r - j * 1.5703125) - j * 0.00048382679233327508;
        q = u32(i32(j) & 3);
    }
    let r2 = r * r;
    let s = r + r * r2 * (-0.16666666666666666 + r2 * (0.0083333333333333332 + r2 * (-0.00019841269841269841 + r2 * 2.7557319223985893e-06)));
    let c = 1.0 + r2 * (-0.5 + r2 * (0.041666666666666664 + r2 * (-0.0013888888888888889 + r2 * (2.4801587301587302e-05 + r2 * -2.7557319223985888e-07))));
    var cs: vec2<f32>;
    switch (q & 3u) {
        case 0u: { cs = vec2<f32>(c, s); }
        case 1u: { cs = vec2<f32>(-s, c); }
        case 2u: { cs = vec2<f32>(-c, -s); }
        default: { cs = vec2<f32>(s, -c); }
    }
    if (theta < 0.0) { cs.y = -cs.y; }
    return cs;
}
`
