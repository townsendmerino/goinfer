//go:build gpu

package kernels

import (
	"fmt"
	"strings"

	"github.com/oliverbestmann/webgpu/wgpu"
)

// Per-32 ACTIVATION quantization for the resident decode runner (decoder.Options.ActQuantGroup == 32, RunModel.ActG32).
// The per-vector quantizers scale a whole activation by one max/127, and a family with massive activation outliers
// (Phi-3: max/rms about 80-90 at down_proj inputs, decoder.ActivationQuantHazard) loses nearly the whole vector to
// rounding. Here every group of 32 consecutive elements gets its own scale, scales[g], so a scale buffer holds np/32
// floats. Each quantizer is its per-vector sibling's value function (RMSNorm, SwiGLU, GELU-tanh gate, identity) with
// the max and pack done per group; the W8A8 GEMVs are their siblings with the activation scale moved inside the sum:
// each group's exact i32 dot is scaled by scales[g] in f32, and the weight's row scale multiplies the total
// (linalg.matmulW8A8GroupedRef). The bindings and launch shapes are the siblings', so the runner swaps pipelines and
// nothing else. Zero convention: an all-zero group gets scale 0 and zero codes, as the CPU's
// QuantizeActivationsGroupedInto does.
//
// Contract: np (the packed length) is a multiple of 32, and a GEMV's kp is a multiple of 32; the runner checks both.

// actG32Value is each quantizer's per-element value, val(k), for k < p.n; prelude runs before the groups (the
// RMSNorm's inverse scale into inv).
type actG32Value struct {
	structP, bindings, helpers, prelude, n string
}

// actG32Quant writes a single-workgroup, 256-lane per-32 quantizer: prelude, then lane t takes groups t, t+256, ….
func actG32Quant(v actG32Value) string {
	return v.structP + v.bindings + v.helpers + `
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
` + v.prelude + `
    let ng = p.np / 32u;
    for (var g: u32 = t; g < ng; g = g + 256u) {
        var mx: f32 = 0.0;
        for (var j: u32 = 0u; j < 32u; j = j + 1u) {
            let k = g * 32u + j;
            if (k < ` + v.n + `) { mx = max(mx, abs(val(k))); }
        }
        let s = mx / 127.0;
        var invs: f32 = 0.0;
        if (s > 0.0) { invs = 1.0 / s; }
        scales[g] = s;
        for (var wi: u32 = 0u; wi < 8u; wi = wi + 1u) {
            var word: u32 = 0u;
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let k = g * 32u + wi * 4u + j;
                var q: i32 = 0;
                if (k < ` + v.n + `) {
                    q = i32(round(val(k) * invs));
                    if (q > 127) { q = 127; } else if (q < -127) { q = -127; }
                }
                word = word | ((u32(q) & 0xffu) << (8u * j));
            }
            qout[g * 8u + wi] = word;
        }
    }
}
`
}

const actG32GatedBindings = `
@group(0) @binding(0) var<storage, read>       gate:   array<f32>;  // [n]
@group(0) @binding(1) var<storage, read>       up:     array<f32>;  // [n]
@group(0) @binding(2) var<storage, read_write> qout:   array<u32>;  // [np/4] packed int8
@group(0) @binding(3) var<storage, read_write> scales: array<f32>;  // [np/32]
@group(0) @binding(4) var<uniform>             p:      P;
`

// RmsnormQuantG32WGSL is RmsnormQuantWGSL with per-32 scales. The sum of squares runs on lanes 0-63 with the 64-lane
// stride and tree, as the widened per-vector kernel's does, so the normalized values are that kernel's bits.
var RmsnormQuantG32WGSL = actG32Quant(actG32Value{
	structP: "struct P { h: u32, eps: f32, addone: u32, np: u32 };\n",
	bindings: `
@group(0) @binding(0) var<storage, read>       src:    array<f32>;  // [h]
@group(0) @binding(1) var<storage, read>       weight: array<f32>;  // [h]
@group(0) @binding(2) var<storage, read_write> qout:   array<u32>;  // [np/4] packed int8
@group(0) @binding(3) var<storage, read_write> scales: array<f32>;  // [np/32]
@group(0) @binding(4) var<uniform>             p:      P;
var<workgroup> sh: array<f32, 64>;
var<workgroup> inv: f32;
`,
	helpers: `
fn val(k: u32) -> f32 {
    var w = weight[k];
    if (p.addone == 1u) { w = w + 1.0; }
    return src[k] * inv * w;
}
`,
	prelude: `
    var ss: f32 = 0.0;
    if (t < 64u) { for (var i: u32 = t; i < p.h; i = i + 64u) { let v = src[i]; ss = ss + v*v; } sh[t] = ss; }
    workgroupBarrier();
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { sh[t] = sh[t] + sh[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (t == 0u) { inv = 1.0 / sqrt(sh[0] / f32(p.h) + p.eps); }
    workgroupBarrier();
`,
	n: "p.h",
})

// SwigluQuantG32WGSL is SwigluQuantWGSL with per-32 scales.
var SwigluQuantG32WGSL = actG32Quant(actG32Value{
	structP:  "struct P { n: u32, np: u32, _b: u32, _c: u32 };\n",
	bindings: actG32GatedBindings,
	helpers: `
fn val(k: u32) -> f32 { let g = gate[k]; return (g / (1.0 + exp(-g))) * up[k]; }
`,
	n: "p.n",
})

// GegluQuantG32WGSL is GegluQuantWGSL with per-32 scales; the same ±15 clamp before tanh.
var GegluQuantG32WGSL = actG32Quant(actG32Value{
	structP:  "struct P { n: u32, np: u32, _b: u32, _c: u32 };\n",
	bindings: actG32GatedBindings,
	helpers: `
fn val(k: u32) -> f32 {
    let x = gate[k];
    let a = 0.7978845608028654 * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + tanh(clamp(a, -15.0, 15.0))) * up[k];
}
`,
	n: "p.n",
})

// QuantizeG32WGSL is QuantizeShaderWGSL with per-32 scales, for the single row (d.m == 1) decode quantizes.
var QuantizeG32WGSL = actG32Quant(actG32Value{
	structP: "struct QDims { m: u32, n: u32, np: u32, _p: u32 };\n",
	bindings: `
@group(0) @binding(0) var<storage, read>       src:    array<f32>;  // [n]
@group(0) @binding(1) var<storage, read_write> qout:   array<u32>;  // [np/4] packed int8
@group(0) @binding(2) var<storage, read_write> scales: array<f32>;  // [np/32]
@group(0) @binding(3) var<uniform>             p:      QDims;
`,
	helpers: "\nfn val(k: u32) -> f32 { return src[k]; }\n",
	n:       "p.n",
})

// gemvW8A8G32Body is the W8A8 GEMV's body with per-32 activation scales: lane t takes groups t, t+64, … (two vec4 words
// each), the group's exact i32 dot times aScales[g] accumulates in f32, and the f32 partials tree-reduce.
const gemvW8A8G32Body = `
fn unpack_i8x4(w: u32) -> vec4<i32> {
    return vec4<i32>(i32(w << 24u) >> 24u, i32(w << 16u) >> 24u, i32(w << 8u) >> 24u, i32(w) >> 24u);
}
fn dotw(a: u32, b: u32) -> i32 {
    let av = unpack_i8x4(a);
    let bv = unpack_i8x4(b);
    return av.x*bv.x + av.y*bv.y + av.z*bv.z + av.w*bv.w;
}
fn dot4(a: vec4<u32>, b: vec4<u32>) -> i32 {
    return dotw(a.x, b.x) + dotw(a.y, b.y) + dotw(a.z, b.z) + dotw(a.w, b.w);
}

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let n = wid.x + wid.y * 32768u;
    if (n >= dims.n) { return; }
    let t = lid.x;
    let kv = dims.kp / 16u;
    let ng = dims.kp / 32u;
    let bBase = n * kv;
    var acc: f32 = 0.0;
    for (var g: u32 = t; g < ng; g = g + 64u) {
        let isum = dot4(aq[2u * g], bq[bBase + 2u * g]) + dot4(aq[2u * g + 1u], bq[bBase + 2u * g + 1u]);
        acc = acc + f32(isum) * aScales[g];
    }
    partial[t] = acc;
    workgroupBarrier();
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { partial[t] = partial[t] + partial[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (t == 0u) {
        let r = partial[0] * bScales[n];
        EPILOGUE
    }
}
`

const gemvW8A8G32Bindings = `
@group(0) @binding(0) var<storage, read>       aq:      array<vec4<u32>>;  // [1, kp/16]
@group(0) @binding(1) var<storage, read>       bq:      array<vec4<u32>>;  // [N, kp/16]
@group(0) @binding(2) var<storage, read>       aScales: array<f32>;        // [kp/32]
@group(0) @binding(3) var<storage, read>       bScales: array<f32>;        // [N]
@group(0) @binding(4) var<storage, read_write> dst:     array<f32>;        // [N]
@group(0) @binding(5) var<uniform>             dims:    Dims;
`

// GemvW8A8G32WGSL is gemvW8A8ShaderWGSL with per-32 activation scales, the same addResidual epilogue.
var GemvW8A8G32WGSL = "struct Dims { m: u32, kp: u32, n: u32, addResidual: u32 };\n" + gemvW8A8G32Bindings +
	strings.Replace(gemvW8A8G32Body, "EPILOGUE", "if (dims.addResidual == 1u) { dst[n] = dst[n] + r; } else { dst[n] = r; }", 1)

// GemvW8A8BiasG32WGSL is gemvW8A8BiasShaderWGSL with per-32 activation scales.
var GemvW8A8BiasG32WGSL = "struct Dims { m: u32, kp: u32, n: u32, _pad: u32 };\n" + gemvW8A8G32Bindings +
	"@group(0) @binding(6) var<storage, read>       bias:    array<f32>;        // [N]\n" +
	strings.Replace(gemvW8A8G32Body, "EPILOGUE", "dst[n] = r + bias[n];", 1)

// ActG32PerVectorScaleForTest is the plant of the per-32 gates: the GEMVs read group 0's scale for every group, the
// per-vector indexing the twins replace. Read when a Context compiles them (ensureActG32).
var ActG32PerVectorScaleForTest bool

// actG32Pipes are the per-32 twins a Context compiles on first use (ensureActG32).
type actG32Pipes struct {
	rmsQuant, swigluQuant, gegluQuant, quantize, gemv, gemvBias g32Pipe
}

// g32Pipe is one compiled pipeline and its bind-group layout.
type g32Pipe struct {
	pl *wgpu.ComputePipeline
	ly *wgpu.BindGroupLayout
}

// ensureActG32 compiles the per-32 twins once per Context.
func (c *Context) ensureActG32() error {
	if c.actG32 != nil {
		return nil
	}
	gemvSrc := func(src string) string {
		if ActG32PerVectorScaleForTest {
			return strings.Replace(src, "f32(isum) * aScales[g]", "f32(isum) * aScales[0]", 1)
		}
		return src
	}
	var p actG32Pipes
	for _, k := range []struct {
		dst        *g32Pipe
		label, src string
	}{
		{&p.rmsQuant, "rmsnormQuantG32", RmsnormQuantG32WGSL},
		{&p.swigluQuant, "swigluQuantG32", SwigluQuantG32WGSL},
		{&p.gegluQuant, "gegluQuantG32", GegluQuantG32WGSL},
		{&p.quantize, "quantizeG32", QuantizeG32WGSL},
		{&p.gemv, "gemvW8A8G32", gemvSrc(GemvW8A8G32WGSL)},
		{&p.gemvBias, "gemvW8A8BiasG32", gemvSrc(GemvW8A8BiasG32WGSL)},
	} {
		_, pl, ly, err := c.mkPipeline(k.label, k.src)
		if err != nil {
			return fmt.Errorf("gpu: per-32 activations: %w", err)
		}
		*k.dst = g32Pipe{pl, ly}
	}
	c.actG32 = &p
	return nil
}
