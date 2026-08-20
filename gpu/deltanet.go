//go:build gpu

package gpu

import (
	"fmt"

	"github.com/cogentcore/webgpu/wgpu"
)

// Resident Gated-DeltaNet decode step (Qwen3.5/3.6-MoE, Qwen3-Next, Qwen3.8).
//
// This is the mixer that makes every DeltaNet hybrid CPU-only on every backend today: 48 of
// Qwen3.8's 64 layers are this recurrence, and `decodeRunnerEligible` refuses the whole
// arch.qwen35 family rather than run half a model with a silently missing mixer. See
// docs/deltanet-residency-plan.md.
//
// It slots onto the resident DecodeRunner the same way the Mamba-2 engine does — the conv window
// and the gated norm are that engine's (mambaConv / mambaGNorm, same shapes) and the persistent
// per-layer state is a storage buffer updated in place per token. Only the delta rule below is new.
//
// THE STATE IS STORED TRANSPOSED RELATIVE TO THE CPU, and that is the point of the port rather
// than an implementation detail. decoder/deltanet.go holds S as [hk, hv] and walks it COLUMN-wise:
//
//	for vd:  for kd: kv += S[kd*hv+vd]*k[kd]          // stride hv — a cache line per access
//	         for kd: S[kd*hv+vd] += ...; o += ...     // second pass over the same memory
//
// Here S is [hv, hk], so thread (headV, vd) OWNS the contiguous row S[headV][vd][0:hk] and reads it
// with stride 1. That gives the same thread-owns-its-row race-freedom mambaSSM relies on (no scan,
// no atomics, no position loop) and fixes the CPU's strided access and double pass at the same time.
//
// Per thread, for its own output element vd:
//
//	S[kd] *= gt                        // decay (gt = exp(negExpA·softplus(a+dt_bias)))
//	kv     = Σ_kd S[kd]·k[kd]
//	delta  = (v[vd] − kv)·beta         // beta = sigmoid(b)
//	S[kd] += k[kd]·delta ; o += S[kd]·q[kd]
//	out[headV*hv+vd] = o
//
// q and k arrive ALREADY l2-normalized (deltaNorm below): the norms are per KEY head, so computing
// them inside this kernel would repeat each one hv times — the same work as the recurrence itself.
const deltaRuleShaderWGSL = `
struct P { nv: u32, nk: u32, hk: u32, hv: u32, rep: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       qn:    array<f32>;  // [nk*hk] l2-normalized, scaled
@group(0) @binding(1) var<storage, read>       kn:    array<f32>;  // [nk*hk] l2-normalized
@group(0) @binding(2) var<storage, read>       v:     array<f32>;  // [nv*hv]
@group(0) @binding(3) var<storage, read>       headP: array<f32>;  // [nv*2]: beta, gt
@group(0) @binding(4) var<storage, read_write> state: array<f32>;  // [nv*hv*hk], in place, [hv,hk]
@group(0) @binding(5) var<storage, read_write> yout:  array<f32>;  // [nv*hv]
@group(0) @binding(6) var<uniform>             p:     P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= p.nv * p.hv) { return; }
    let headV = t / p.hv;
    let vd    = t % p.hv;
    let headK = headV / p.rep;          // GVA: rep value heads share one key head

    let beta = headP[headV * 2u + 0u];
    let gt   = headP[headV * 2u + 1u];

    let sBase = (headV * p.hv + vd) * p.hk;   // this thread's contiguous row
    let kBase = headK * p.hk;

    // decay + kv, in one pass over the row
    var kv = 0.0;
    for (var kd = 0u; kd < p.hk; kd = kd + 1u) {
        let s = state[sBase + kd] * gt;
        state[sBase + kd] = s;
        kv = kv + s * kn[kBase + kd];
    }
    let delta = (v[headV * p.hv + vd] - kv) * beta;

    var o = 0.0;
    for (var kd = 0u; kd < p.hk; kd = kd + 1u) {
        let s = state[sBase + kd] + kn[kBase + kd] * delta;
        state[sBase + kd] = s;
        o = o + s * qn[kBase + kd];
    }
    yout[headV * p.hv + vd] = o;
}
`

// deltaNorm l2-normalizes the per-head q and k slices of the conv output, and applies the query
// scale. One thread per KEY head: the work is nk·hk (2048 elements on the real 27.8B geometry),
// far too small to be worth splitting further, and doing it here instead of inside deltaRule keeps
// the recurrence from recomputing each norm hv times.
//
// The conv output is [q(keyDim) ‖ k(keyDim) ‖ v(valueDim)] contiguous — the same packing the CPU
// reference slices, so the layouts agree by construction rather than by comment.
const deltaNormShaderWGSL = `
struct P { nk: u32, hk: u32, keyDim: u32, _a: u32, qScale: f32, _b: f32, _c: f32, _d: f32 };
@group(0) @binding(0) var<storage, read>       conv: array<f32>;  // [q|k|v]
@group(0) @binding(1) var<storage, read_write> qn:   array<f32>;  // [nk*hk]
@group(0) @binding(2) var<storage, read_write> kn:   array<f32>;  // [nk*hk]
@group(0) @binding(3) var<uniform>             p:    P;

@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let h = gid.x;
    if (h >= p.nk) { return; }
    let base = h * p.hk;

    var qs = 0.0;
    var ks = 0.0;
    for (var i = 0u; i < p.hk; i = i + 1u) {
        let qv = conv[base + i];
        let kvv = conv[p.keyDim + base + i];
        qs = qs + qv * qv;
        ks = ks + kvv * kvv;
    }
    // sqrt(1/(ss+eps)), NOT inverseSqrt(ss). The epsilon is FLA's 1e-6, and it is NOT a precision
    // knob: on a head whose conv output is all zero — reachable, since silu(x) is exactly 0 at
    // x=0 — inverseSqrt(0) is +inf and every downstream state entry becomes NaN, where the CPU
    // reference yields a finite 1e3 scale and a zero vector. TestDeltaNorm_cpuParity gates that
    // case explicitly; at ordinary magnitudes the two forms are indistinguishable, which is why
    // the recurrence gate alone does not catch it.
    let qi = sqrt(1.0 / (qs + 1e-6)) * p.qScale;
    let ki = sqrt(1.0 / (ks + 1e-6));
    for (var i = 0u; i < p.hk; i = i + 1u) {
        qn[base + i] = conv[base + i] * qi;
        kn[base + i] = conv[p.keyDim + base + i] * ki;
    }
}
`

func (c *Context) ensureDeltaRule() error {
	if c.deltaRulePipeline != nil {
		return nil
	}
	sh, err := c.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "deltaRule", WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: deltaRuleShaderWGSL},
	})
	if err != nil {
		return fmt.Errorf("gpu: compile deltaRule: %w", err)
	}
	pl, err := c.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "deltaRule", Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return fmt.Errorf("gpu: pipeline deltaRule: %w", err)
	}
	c.track(sh.Release, pl.Release) // audit C-26: register at creation
	c.deltaRuleShader, c.deltaRulePipeline, c.deltaRuleLayout = sh, pl, c.bgl(pl)
	return nil
}

func (c *Context) ensureDeltaNorm() error {
	if c.deltaNormPipeline != nil {
		return nil
	}
	sh, err := c.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "deltaNorm", WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: deltaNormShaderWGSL},
	})
	if err != nil {
		return fmt.Errorf("gpu: compile deltaNorm: %w", err)
	}
	pl, err := c.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "deltaNorm", Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return fmt.Errorf("gpu: pipeline deltaNorm: %w", err)
	}
	c.track(sh.Release, pl.Release)
	c.deltaNormShader, c.deltaNormPipeline, c.deltaNormLayout = sh, pl, c.bgl(pl)
	return nil
}
