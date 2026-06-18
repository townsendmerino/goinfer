//go:build gpu

package gpu

import (
	"fmt"

	"github.com/cogentcore/webgpu/wgpu"
)

// Resident Mamba-2 selective-SSM decode step (Granite-4.0-H / Nemotron-H hybrids).
// Decode is a BOUNDED per-token recurrence — not the prefill scan — so it slots onto
// the resident DecodeRunner like a KV cache: persistent {conv ring, ssm} state updated
// in place per token inside the single command buffer. See docs/ssm-residency-scope.md.
//
// mambaSSM is the selective state update (mamba2.go step 4): one thread per (head, pi),
// each OWNING its state row ssm[head,pi,·] (N contiguous floats), so the in-place update
// is race-free with no scan and no position loop. Per thread:
//
//	dth = softplus(dtRaw[head] + dtBias[head]);  dA = exp(dth · Aexp[head])
//	for n: S[n] = S[n]·dA + (dth·x)·B[n];  acc += S[n]·C[n]
//	y[head·P+pi] = acc + D[head]·x
//
// conv holds [x(dInner) ‖ B(gSize) ‖ C(gSize)] contiguous (the mambaConv output layout).
// Aexp[head] = -exp(aLog[head]) is precomputed host-side in f64 (the static weight part of
// dA), matching mamba2Step's f64 A; only the per-token dth·exp stays f32. The recurrence is
// stable (dA∈(0,1) ⇒ state decays), so f32 drift over long sequences stays bounded —
// gated at 1/16/256/1k/2k tokens by TestMambaSSM_driftParity.
const mambaSSMShaderWGSL = `
struct P { nHeads: u32, hp: u32, dn: u32, ng: u32, repeat: u32, gSize: u32, dInner: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read>       conv:  array<f32>;  // [x(dInner) | B(gSize) | C(gSize)]
@group(0) @binding(1) var<storage, read>       dtRaw: array<f32>;  // [nHeads]
@group(0) @binding(2) var<storage, read>       headP: array<f32>;  // [nHeads*3]: Aexp, dtBias, D
@group(0) @binding(3) var<storage, read_write> ssm:   array<f32>;  // [nHeads*hp*dn], in place
@group(0) @binding(4) var<storage, read_write> yout:  array<f32>;  // [dInner]
@group(0) @binding(5) var<uniform>             p:     P;

fn softplus(x: f32) -> f32 {
    if (x > 20.0) { return x; }
    return log(1.0 + exp(x));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= p.nHeads * p.hp) { return; }
    let head = t / p.hp;
    let pi = t % p.hp;
    let g = head / p.repeat;
    let Aexp = headP[head * 3u + 0u];
    let dtb = headP[head * 3u + 1u];
    let Dw = headP[head * 3u + 2u];
    let dth = softplus(dtRaw[head] + dtb);
    let dA = exp(dth * Aexp);
    let xv = conv[head * p.hp + pi];
    let dx = dth * xv;
    let sBase = (head * p.hp + pi) * p.dn;
    let bBase = p.dInner + g * p.dn;
    let cBase = p.dInner + p.gSize + g * p.dn;
    var acc: f32 = 0.0;
    for (var n: u32 = 0u; n < p.dn; n = n + 1u) {
        let v = ssm[sBase + n] * dA + dx * conv[bBase + n];
        ssm[sBase + n] = v;
        acc = acc + v * conv[cBase + n];
    }
    yout[head * p.hp + pi] = acc + Dw * xv;
}
`

// mambaConv is the depthwise causal conv (mamba2.go step 2): per channel c,
// conv[c] = silu(convB[c] + Σ_{j<K-1} convW[c·K+j]·win[j][c] + convW[c·K+K-1]·xBC[c]),
// then shift the K-1 ring window (drop oldest, append xBC). One thread per channel owns
// column c, so the in-place window shift is race-free. The window starts zeroed — that
// exactly reproduces the reference's early-token zero-padding (verified by trace). win is
// stored [K-1, convDim] oldest-first; K (DConv) ≤ 8.
const mambaConvShaderWGSL = `
struct P { convDim: u32, K: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       xBC:   array<f32>;  // [convDim]
@group(0) @binding(1) var<storage, read>       convW: array<f32>;  // [convDim*K]
@group(0) @binding(2) var<storage, read>       convB: array<f32>;  // [convDim]
@group(0) @binding(3) var<storage, read_write> win:   array<f32>;  // [(K-1)*convDim], in place
@group(0) @binding(4) var<storage, read_write> conv:  array<f32>;  // [convDim]
@group(0) @binding(5) var<uniform>             p:     P;

fn silu(x: f32) -> f32 { return x / (1.0 + exp(-x)); }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= p.convDim) { return; }
    let K = p.K;
    var w: array<f32, 8>;
    var s = convB[c] + convW[c * K + (K - 1u)] * xBC[c];
    for (var j: u32 = 0u; j < K - 1u; j = j + 1u) {
        let wv = win[j * p.convDim + c];
        w[j] = wv;
        s = s + convW[c * K + j] * wv;
    }
    conv[c] = silu(s);
    for (var j: u32 = 0u; j + 1u < K - 1u; j = j + 1u) { win[j * p.convDim + c] = w[j + 1u]; }
    win[(K - 2u) * p.convDim + c] = xBC[c];
}
`

func (c *Context) ensureMambaConv() error {
	if c.mambaConvPipeline != nil {
		return nil
	}
	sh, err := c.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "mambaConv", WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: mambaConvShaderWGSL},
	})
	if err != nil {
		return fmt.Errorf("gpu: compile mambaConv: %w", err)
	}
	pl, err := c.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "mambaConv", Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return fmt.Errorf("gpu: pipeline mambaConv: %w", err)
	}
	c.mambaConvShader, c.mambaConvPipeline, c.mambaConvLayout = sh, pl, pl.GetBindGroupLayout(0)
	return nil
}

// mambaGatedNorm is the gated grouped RMSNorm (mamba2.go step 5): gated = y·silu(z), then
// each of nGroups contiguous groups is RMS-normalized independently × weight. One workgroup
// per group (groupSize = dInner/nGroups). Granite normalizes over full dInner (nGroups=1);
// Nemotron-H per group (nGroups=NGroups).
const mambaGNormShaderWGSL = `
struct P { dInner: u32, nGroups: u32, groupSize: u32, eps: f32 };
@group(0) @binding(0) var<storage, read>       yin:    array<f32>;  // [dInner] (SSM output)
@group(0) @binding(1) var<storage, read>       z:      array<f32>;  // [dInner] (gate)
@group(0) @binding(2) var<storage, read>       weight: array<f32>;  // [dInner]
@group(0) @binding(3) var<storage, read_write> gated:  array<f32>;  // [dInner] out
@group(0) @binding(4) var<uniform>             p:      P;
var<workgroup> sh: array<f32, 64>;

fn siluG(x: f32) -> f32 { return x / (1.0 + exp(-x)); }

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let g = wid.x;
    if (g >= p.nGroups) { return; }
    let t = lid.x;
    let base = g * p.groupSize;
    var ss: f32 = 0.0;
    for (var i: u32 = t; i < p.groupSize; i = i + 64u) {
        let v = yin[base + i] * siluG(z[base + i]);
        gated[base + i] = v;
        ss = ss + v * v;
    }
    sh[t] = ss;
    workgroupBarrier();
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { sh[t] = sh[t] + sh[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let inv = 1.0 / sqrt(sh[0] / f32(p.groupSize) + p.eps);
    for (var i: u32 = t; i < p.groupSize; i = i + 64u) {
        gated[base + i] = weight[base + i] * gated[base + i] * inv;
    }
}
`

func (c *Context) ensureMambaGNorm() error {
	if c.mambaGNormPipeline != nil {
		return nil
	}
	sh, err := c.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "mambaGNorm", WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: mambaGNormShaderWGSL},
	})
	if err != nil {
		return fmt.Errorf("gpu: compile mambaGNorm: %w", err)
	}
	pl, err := c.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "mambaGNorm", Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return fmt.Errorf("gpu: pipeline mambaGNorm: %w", err)
	}
	c.mambaGNormShader, c.mambaGNormPipeline, c.mambaGNormLayout = sh, pl, pl.GetBindGroupLayout(0)
	return nil
}

func (c *Context) ensureMambaSSM() error {
	if c.mambaSSMPipeline != nil {
		return nil
	}
	sh, err := c.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "mambaSSM", WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: mambaSSMShaderWGSL},
	})
	if err != nil {
		return fmt.Errorf("gpu: compile mambaSSM: %w", err)
	}
	pl, err := c.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "mambaSSM", Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return fmt.Errorf("gpu: pipeline mambaSSM: %w", err)
	}
	c.mambaSSMShader, c.mambaSSMPipeline, c.mambaSSMLayout = sh, pl, pl.GetBindGroupLayout(0)
	return nil
}
