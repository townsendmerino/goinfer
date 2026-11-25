//go:build gpu

package kernels

import "github.com/oliverbestmann/webgpu/wgpu"

// LayerNormQuantWGSL fuses a bias-free LayerNorm (decoder.FeatLayerNorm: Cohere, Command-R7B) into the int8 activation
// quantize: xn = (x - mean)·inv·weight with inv = 1/sqrt(var + eps), max-abs'd and packed in one dispatch. Its bindings
// and uniform are RmsnormQuantWGSL's (p.addone is unused), so the runner binds it in that kernel's place
// (RunModel.LayerNorm). One 64-lane workgroup; the mean and the variance are tree sums, and xn is recomputed in the pack
// pass rather than staged.
const LayerNormQuantWGSL = `
struct P { h: u32, eps: f32, addone: u32, np: u32 };
@group(0) @binding(0) var<storage, read>       src:    array<f32>;  // [h]
@group(0) @binding(1) var<storage, read>       weight: array<f32>;  // [h]
@group(0) @binding(2) var<storage, read_write> qout:   array<u32>;  // [np/4] packed int8
@group(0) @binding(3) var<storage, read_write> scales: array<f32>;  // [1]
@group(0) @binding(4) var<uniform>             p:      P;
var<workgroup> sh: array<f32, 64>;
fn treeSum(t: u32) -> f32 {
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { sh[t] = sh[t] + sh[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let r = sh[0];
    workgroupBarrier();
    return r;
}
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    var acc: f32 = 0.0;
    for (var i: u32 = t; i < p.h; i = i + 64u) { acc = acc + src[i]; }
    sh[t] = acc;
    workgroupBarrier();
    let mean = treeSum(t) / f32(p.h);
    acc = 0.0;
    for (var i: u32 = t; i < p.h; i = i + 64u) { let d = src[i] - mean; acc = acc + d * d; }
    sh[t] = acc;
    workgroupBarrier();
    let inv = 1.0 / sqrt(treeSum(t) / f32(p.h) + p.eps);
    var mx: f32 = 0.0;
    for (var i: u32 = t; i < p.h; i = i + 64u) { mx = max(mx, abs((src[i] - mean) * inv * weight[i])); }
    sh[t] = mx;
    workgroupBarrier();
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { sh[t] = max(sh[t], sh[t + stride]); }
        workgroupBarrier();
        stride = stride / 2u;
    }
    var s: f32 = sh[0] / 127.0;
    if (s == 0.0) { s = 1.0; }
    if (t == 0u) { scales[0] = s; }
    let invs = 1.0 / s;
    let nw = p.np / 4u;
    for (var wi: u32 = t; wi < nw; wi = wi + 64u) {
        var word: u32 = 0u;
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let k = wi * 4u + j;
            var q: i32 = 0;
            if (k < p.h) {
                q = i32(round((src[k] - mean) * inv * weight[k] * invs));
                if (q > 127) { q = 127; } else if (q < -127) { q = -127; }
            }
            word = word | ((u32(q) & 0xffu) << (8u * j));
        }
        qout[wi] = word;
    }
}
`

// ensureLayerNorm compiles LayerNormQuantWGSL once per Context.
func (c *Context) ensureLayerNorm() error {
	if c.layerNormQuant.pl != nil {
		return nil
	}
	_, pl, ly, err := c.mkPipeline("layerNormQuant", LayerNormQuantWGSL)
	if err != nil {
		return err
	}
	c.layerNormQuant = layerNormPipe{pl, ly}
	return nil
}

// layerNormPipe is the compiled LayerNormQuantWGSL and its bind-group layout.
type layerNormPipe struct {
	pl *wgpu.ComputePipeline
	ly *wgpu.BindGroupLayout
}
