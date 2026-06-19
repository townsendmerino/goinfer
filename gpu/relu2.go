//go:build gpu

package gpu

import "github.com/cogentcore/webgpu/wgpu"

// Nemotron-H squared-ReLU FFN (port from the granite SSM work). Nemotron-H's MLP block is
// NON-gated: down(relu²(up(x))) — no gate projection, unlike every SwiGLU family. relu2Quant
// is the fused activation+quant for it: relu²(up) → max-abs → int8, one dispatch (mirrors
// swigluQuant but unary). relu²(x)=max(0,x)² ≥ 0, so the int8 is non-negative ([0,127]); the
// down GEMV reads it like any int8 activation. The up/down GEMVs reuse the existing W8A8 path —
// this kernel is the only genuinely-new resident piece for Nemotron-H.
const relu2QuantWGSL = `
struct P { n: u32, np: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       up:     array<f32>;  // [n]
@group(0) @binding(1) var<storage, read_write> qout:   array<u32>;  // [np/4] packed int8
@group(0) @binding(2) var<storage, read_write> scales: array<f32>;  // [1]
@group(0) @binding(3) var<uniform>             p:      P;
var<workgroup> sh: array<f32, 64>;
fn relu2(x: f32) -> f32 { let r = max(0.0, x); return r * r; }
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    var mx: f32 = 0.0;
    for (var i: u32 = t; i < p.n; i = i + 64u) { mx = max(mx, relu2(up[i])); }
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
            if (k < p.n) {
                q = i32(round(relu2(up[k]) * invs));
                if (q > 127) { q = 127; } else if (q < -127) { q = -127; }
            }
            word = word | ((u32(q) & 0xffu) << (8u * j));
        }
        qout[wi] = word;
    }
}
`

func (c *Context) ensureRelu2() error {
	if c.relu2Pipeline != nil {
		return nil
	}
	sh, pl, lay, err := c.buildCompute("relu2Quant", relu2QuantWGSL)
	if err != nil {
		return err
	}
	c.relu2Shader, c.relu2Pipeline, c.relu2Layout = sh, pl, lay
	return nil
}

// relu2QuantOp records the fused relu²(up)→int8 dispatch into the resident plan, returning the
// packed int8 activation + its scale (for the down-projection W8A8 GEMV).
func (c *Context) relu2QuantBind(up, qout, scales, dims *wgpu.Buffer) *wgpu.BindGroup {
	bg, _ := c.device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: c.relu2Layout, Entries: []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: up, Size: up.GetSize()}, {Binding: 1, Buffer: qout, Size: qout.GetSize()},
		{Binding: 2, Buffer: scales, Size: scales.GetSize()}, {Binding: 3, Buffer: dims, Size: dims.GetSize()},
	}})
	return bg
}
