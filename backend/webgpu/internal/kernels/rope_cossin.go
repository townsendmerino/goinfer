//go:build gpu

package kernels

// ropeCosSinWGSL defines ropeCosSin(theta) -> vec2(cos θ, sin θ), the rotation every RoPE kernel applies. The WGSL builtins sin and
// cos are conformant with an absolute error of 2^-11 inside [-π, π] and none bound outside, and adapters do return about 2^-14 there,
// which puts the rotated Q and K well off the CPU reference's float64 cos and sin; so no kernel calls them
// (TestRopeKernels_noBuiltinSinCos). θ is reduced by π/2 in three parts (Cody-Waite), then the quadrant's polynomial runs on
// [-π/4, π/4], whose truncation is under 2e-9. The error is about 1e-7 for θ under 8 and about ulp(θ) beyond, the resolution of the
// f32 angle itself: a driver may reassociate the reduction's subtractions. Every shader that rotates is compiled with this
// prepended; TestRope_matchesCPUAngles holds the kernels to the CPU's angles.
const ropeCosSinWGSL = `
fn ropeCosSin(theta: f32) -> vec2<f32> {
    let j = round(theta * 0.63661974668502808);
    var r = theta - j * 1.5703125;
    r = r - j * 0.00048382679233327508;
    r = r - j * 2.5632829192545614e-12;
    let r2 = r * r;
    let s = r + r * r2 * (-0.16666666666666666 + r2 * (0.0083333333333333332 + r2 * (-0.00019841269841269841 + r2 * 2.7557319223985893e-06)));
    let c = 1.0 + r2 * (-0.5 + r2 * (0.041666666666666664 + r2 * (-0.0013888888888888889 + r2 * (2.4801587301587302e-05 + r2 * -2.7557319223985888e-07))));
    let q = ((i32(j) % 4) + 4) % 4;
    if (q == 0) { return vec2<f32>(c, s); }
    if (q == 1) { return vec2<f32>(-s, c); }
    if (q == 2) { return vec2<f32>(-c, -s); }
    return vec2<f32>(s, -c);
}
`
