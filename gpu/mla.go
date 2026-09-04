//go:build gpu

package gpu

import (
	"fmt"

	"github.com/cogentcore/webgpu/wgpu"
)

// MLA residency (Lever C4) — DeepSeek / Kimi Multi-head Latent Attention on the GPU
// resident decode path. The compressed-KV latent ([kv_lora_rank | qk_rope_head_dim]
// per position) is the only cached payload; per-head K/V are never materialized.
// This file is the absorb-path core primitive (mirrors decoder.mlaAttentionAbsorb):
// the query is pre-absorbed through W_UK once per head (→ rank-space), each cached
// latent is scored by a rank+rope-dim dot, and V is collapsed to a rank-space weighted
// sum (lifted by W_UV later, on the host side of the runner). The surrounding GEMVs
// (q-LoRA, kv-down, W_UK absorb, W_UV lift, o-proj) reuse the W8A8 path; this kernel
// is the genuinely-new attention shape — a single-query attend with a KV "head" shared
// across all query heads and a score width (latDim) wider than the value width (rank).
//
// Layout: qAbs[h] = qNopeAbs_h[rank] ‖ qRope_h[qkRope] (the absorbed query, width
// latDim = rank+qkRope); lat[j] = cn_j[rank] ‖ krj_j[qkRope] (the stored latent —
// normalized + rope-keyed at store time). Score_j = scale·(qAbs[h]·lat[j]) over the
// full latDim; value = cn_j = lat[j][:rank]; output wsum[h] = Σ_j softmax_j·cn_j.
//
// One workgroup per query head, online (FlashAttention-style) softmax over keys so no
// per-key score row is materialized. The score dot and the value accumulate are both
// STRIDED over the 128-lane workgroup because rank (≤1024) and latDim exceed the lane
// count — each lane owns dims {t, t+128, …}; m/l/corr/pe are scalar (identical across
// lanes, all see the same reduced score) so no extra reduction beyond the score dot.
const mlaAttnWGSL = `
const WG: u32 = 128u;
struct P { nH: u32, latDim: u32, rank: u32, nKeys: u32, scale: f32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       qAbs: array<f32>;  // [nH*latDim] qNopeAbs ‖ qRope per head
@group(0) @binding(1) var<storage, read>       lat:  array<f32>;  // [nKeys*latDim] cn ‖ krj per key
@group(0) @binding(2) var<storage, read_write> wsum: array<f32>;  // [nH*rank] rank-space V collapse
@group(0) @binding(3) var<uniform>             p:    P;
var<workgroup> red: array<f32, WG>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let h = wid.x;
    if (h >= p.nH) { return; }
    let t = lid.x;
    let latDim = p.latDim;
    let rank = p.rank;
    let qbase = h * latDim;
    var acc: array<f32, 8>;  // per-lane value accumulators (rank ≤ 8*WG = 1024)
    for (var i: u32 = 0u; i < 8u; i = i + 1u) { acc[i] = 0.0; }
    var m: f32 = -1e30;
    var l: f32 = 0.0;
    for (var j: u32 = 0u; j < p.nKeys; j = j + 1u) {
        let kbase = j * latDim;
        // score = qAbs[h] · lat[j] over the full latDim (strided across lanes).
        var partial: f32 = 0.0;
        for (var d: u32 = t; d < latDim; d = d + WG) {
            partial = partial + qAbs[qbase + d] * lat[kbase + d];
        }
        red[t] = partial;
        workgroupBarrier();
        var stride: u32 = WG / 2u;
        loop {
            if (stride == 0u) { break; }
            if (t < stride) { red[t] = red[t] + red[t + stride]; }
            workgroupBarrier();
            stride = stride / 2u;
        }
        let x = red[0] * p.scale;
        let mnew = max(m, x);
        let corr = exp(m - mnew);
        let pe = exp(x - mnew);
        // value = cn_j = lat[j][:rank]; accumulate the rank dims this lane owns.
        var i: u32 = 0u;
        for (var d: u32 = t; d < rank; d = d + WG) {
            acc[i] = acc[i] * corr + pe * lat[kbase + d];
            i = i + 1u;
        }
        l = l * corr + pe;
        m = mnew;
        workgroupBarrier();  // all lanes consumed red[0] before next key overwrites red[t]
    }
    var i: u32 = 0u;
    for (var d: u32 = t; d < rank; d = d + WG) {
        wsum[h * rank + d] = acc[i] / l;
        i = i + 1u;
    }
}
`

// mlaLatentStore appends one token's compressed KV to the latent cache (Lever C4b),
// mirroring decoder's cache.AppendLatent feed: the kv_a_proj output kvDown =
// [latent(rank) | rope-key(qkRope)] is split, the rank latent is RMSNorm'd by
// kvALayernorm (cn, the value + score key body) and the rope-key is decoupled-RoPE'd at
// the token position (krj). Both are normalized/roped at STORE time (the position is
// fixed per entry, so this is equivalent to the CPU's attend-time recompute and saves
// re-roping every cached key each step). interleave=1 is the V3 GPT-J pairwise layout
// (de-interleave to [evens|odds] before NeoX rotate); =0 is plain NeoX. ropeScale folds
// the YaRN attention factor (1.0 when none). One workgroup; base = pos*latDim.
const mlaLatentStoreWGSL = `
struct P { rank: u32, qkRope: u32, pos: u32, eps: f32, base: u32, ropeScale: f32, interleave: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       kvDown:   array<f32>;  // [rank+qkRope] current token
@group(0) @binding(1) var<storage, read>       normW:    array<f32>;  // [rank] kvALayernorm
@group(0) @binding(2) var<storage, read>       invFreq:  array<f32>;  // [qkRope/2]
@group(0) @binding(3) var<storage, read_write> latCache: array<f32>;  // [maxLen*(rank+qkRope)]
@group(0) @binding(4) var<uniform>             p:        P;
var<workgroup> sh: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    // RMSNorm over the rank latent → cn.
    var ss: f32 = 0.0;
    for (var i: u32 = t; i < p.rank; i = i + 64u) { let v = kvDown[i]; ss = ss + v*v; }
    sh[t] = ss;
    workgroupBarrier();
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { sh[t] = sh[t] + sh[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let inv = 1.0 / sqrt(sh[0] / f32(p.rank) + p.eps);
    workgroupBarrier();
    for (var i: u32 = t; i < p.rank; i = i + 64u) {
        latCache[p.base + i] = kvDown[i] * inv * normW[i];
    }
    // Decoupled RoPE the qkRope key → krj, written after the rank block.
    let half = p.qkRope / 2u;
    for (var i: u32 = t; i < half; i = i + 64u) {
        let theta = f32(p.pos) * invFreq[i];
        let c = cos(theta) * p.ropeScale;
        let s = sin(theta) * p.ropeScale;
        var a: f32;
        var b: f32;
        if (p.interleave == 1u) {
            a = kvDown[p.rank + 2u*i];
            b = kvDown[p.rank + 2u*i + 1u];
        } else {
            a = kvDown[p.rank + i];
            b = kvDown[p.rank + half + i];
        }
        latCache[p.base + p.rank + i]        = a * c - b * s;
        latCache[p.base + p.rank + half + i] = b * c + a * s;
    }
}
`

func (c *Context) ensureMLAStore() error {
	if c.mlaStorePipeline != nil {
		return nil
	}
	sh, err := c.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "mlaLatentStore", WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: mlaLatentStoreWGSL},
	})
	if err != nil {
		return fmt.Errorf("gpu: compile mlaLatentStore: %w", err)
	}
	pl, err := c.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "mlaLatentStore", Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return fmt.Errorf("gpu: pipeline mlaLatentStore: %w", err)
	}
	c.track(sh.Release, pl.Release) // audit C-26: register at creation
	c.mlaStoreShader, c.mlaStorePipeline, c.mlaStoreLayout = sh, pl, c.bgl(pl)
	return nil
}

// MLALatentStore runs the latent append on host inputs: kvDown [rank+qkRope] → the
// normalized+roped latent entry [rank+qkRope]. Standalone op for parity (the runner
// records the device kernel with base = pos*latDim into the resident latent cache).
func (c *Context) MLALatentStore(kvDown, normW, invFreq []float32, rank, qkRope, pos int, eps, ropeScale float32, interleave bool) ([]float32, error) {
	if err := c.ensureMLAStore(); err != nil {
		return nil, err
	}
	latDim := rank + qkRope
	dBuf, _ := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "mla-kvdown", Contents: wgpu.ToBytes(kvDown), Usage: wgpu.BufferUsageStorage})
	defer dBuf.Release()
	nBuf, _ := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "mla-normw", Contents: wgpu.ToBytes(normW), Usage: wgpu.BufferUsageStorage})
	defer nBuf.Release()
	fBuf, _ := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "mla-invfreq", Contents: wgpu.ToBytes(invFreq), Usage: wgpu.BufferUsageStorage})
	defer fBuf.Release()
	lBuf, err := c.device.CreateBuffer(&wgpu.BufferDescriptor{Label: "mla-latcache", Size: uint64(latDim * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
	if err != nil {
		return nil, err
	}
	defer lBuf.Release()
	pBuf, _ := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "mla-store-p", Contents: wgpu.ToBytes([]uint32{uint32(rank), uint32(qkRope), uint32(pos), f32bits(eps), 0, f32bits(ropeScale), boolU32(interleave), 0}), Usage: wgpu.BufferUsageUniform})
	defer pBuf.Release()
	bg, err := c.device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: c.mlaStoreLayout, Entries: []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: dBuf, Size: dBuf.GetSize()},
		{Binding: 1, Buffer: nBuf, Size: nBuf.GetSize()},
		{Binding: 2, Buffer: fBuf, Size: fBuf.GetSize()},
		{Binding: 3, Buffer: lBuf, Size: lBuf.GetSize()},
		{Binding: 4, Buffer: pBuf, Size: pBuf.GetSize()},
	}})
	if err != nil {
		return nil, err
	}
	defer bg.Release()
	if err := c.submitUnary(c.mlaStorePipeline, bg, 64); err != nil {
		return nil, err
	}
	return c.readbackRaw(lBuf, latDim)
}

// mlaHeadMatvec is the per-head block-diagonal matvec behind the absorb (W_UK) and lift
// (W_UV) steps (Lever C4b). dst[h*N+n] = Σ_k a[h*aStride+k]·w[(h*N+n)*K+k] — each head
// has its OWN weight block and reads its OWN activation slice, so it's not a single dense
// GEMV. Kept f32 (parity-first, and W_UK/W_UV are small for the 16B targets; the big-V3
// case doesn't fit the 8 GB resident budget regardless). Used twice:
//   - absorb: a=q (aStride=qk_head_dim, reads the qk_nope prefix), w=W_UKᵀ, N=rank, K=qk_nope
//   - lift:   a=wsum (aStride=rank),                                w=W_UV,  N=v_head_dim, K=rank
//
// One workgroup per output element over a 2D grid (H·N can exceed the 65535/dim cap).
// outStride lets the absorb write strided into the combined qAbs buffer ([rank|qkRope]
// per head): outStride=latDim places qNopeAbs at qAbs[h*latDim..+rank]. The lift writes
// contiguously (outStride=N).
const mlaHeadMatvecWGSL = `
struct P { nH: u32, N: u32, K: u32, aStride: u32, outStride: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       a:   array<f32>;  // [nH*aStride] per-head activation
@group(0) @binding(1) var<storage, read>       w:   array<f32>;  // [nH*N*K] per-head weight, row-major
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;  // [nH*outStride]
@group(0) @binding(3) var<uniform>             p:   P;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let elem = wid.x + wid.y * 32768u;
    if (elem >= p.nH * p.N) { return; }
    let h = elem / p.N;
    let n = elem % p.N;
    let t = lid.x;
    let abase = h * p.aStride;
    let wbase = (h * p.N + n) * p.K;
    var acc: f32 = 0.0;
    for (var k: u32 = t; k < p.K; k = k + 64u) {
        acc = acc + a[abase + k] * w[wbase + k];
    }
    red[t] = acc;
    workgroupBarrier();
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { red[t] = red[t] + red[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (t == 0u) { dst[h * p.outStride + n] = red[0]; }
}
`

// mlaQRope gathers each head's qk_rope slice from the projected query q [nH*qkHead]
// (laid out [qk_nope | qk_rope] per head), decoupled-RoPEs it at the current position,
// and writes it into the combined qAbs buffer at [h*latDim + rank ..] — directly after
// the W_UK-absorbed qNopeAbs. Same de-interleave+NeoX as the key store (mlaLatentStore),
// so the q·k rope dot matches. One thread per (head, rope-pair).
const mlaQRopeWGSL = `
struct P { nH: u32, qkHead: u32, qkNope: u32, qkRope: u32, rank: u32, latDim: u32, pos: u32, interleave: u32, ropeScale: f32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       q:       array<f32>;  // [nH*qkHead]
@group(0) @binding(1) var<storage, read>       invFreq: array<f32>;  // [qkRope/2]
@group(0) @binding(2) var<storage, read_write> qAbs:    array<f32>;  // [nH*latDim], rope at [h*latDim+rank ..]
@group(0) @binding(3) var<uniform>             p:       P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half = p.qkRope / 2u;
    let idx = gid.x;
    if (idx >= p.nH * half) { return; }
    let h = idx / half;
    let i = idx % half;
    let qbase = h * p.qkHead + p.qkNope;  // rope dims of head h
    let theta = f32(p.pos) * invFreq[i];
    let c = cos(theta) * p.ropeScale;
    let s = sin(theta) * p.ropeScale;
    var a: f32;
    var b: f32;
    if (p.interleave == 1u) {
        a = q[qbase + 2u*i];
        b = q[qbase + 2u*i + 1u];
    } else {
        a = q[qbase + i];
        b = q[qbase + half + i];
    }
    let obase = h * p.latDim + p.rank;
    qAbs[obase + i]        = a * c - b * s;
    qAbs[obase + half + i] = b * c + a * s;
}
`

func (c *Context) ensureMLAHeadMV() error {
	if c.mlaHeadMVPipeline != nil {
		return nil
	}
	sh, err := c.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "mlaHeadMatvec", WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: mlaHeadMatvecWGSL},
	})
	if err != nil {
		return fmt.Errorf("gpu: compile mlaHeadMatvec: %w", err)
	}
	pl, err := c.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "mlaHeadMatvec", Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return fmt.Errorf("gpu: pipeline mlaHeadMatvec: %w", err)
	}
	c.track(sh.Release, pl.Release) // audit C-26: register at creation
	c.mlaHeadMVShader, c.mlaHeadMVPipeline, c.mlaHeadMVLayout = sh, pl, c.bgl(pl)
	return nil
}

// MLAHeadMatvec runs the per-head block-diagonal matvec on host inputs and returns dst
// [nH*N]. a is [nH*aStride], w is [nH*N*K] (per head, row-major). Standalone op for parity.
func (c *Context) MLAHeadMatvec(a, w []float32, nH, N, K, aStride int) ([]float32, error) {
	if err := c.ensureMLAHeadMV(); err != nil {
		return nil, err
	}
	aBuf, _ := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "mla-hmv-a", Contents: wgpu.ToBytes(a), Usage: wgpu.BufferUsageStorage})
	defer aBuf.Release()
	wBuf, _ := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "mla-hmv-w", Contents: wgpu.ToBytes(w), Usage: wgpu.BufferUsageStorage})
	defer wBuf.Release()
	dBuf, err := c.device.CreateBuffer(&wgpu.BufferDescriptor{Label: "mla-hmv-dst", Size: uint64(nH * N * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
	if err != nil {
		return nil, err
	}
	defer dBuf.Release()
	pBuf, _ := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "mla-hmv-p", Contents: wgpu.ToBytes([]uint32{uint32(nH), uint32(N), uint32(K), uint32(aStride), uint32(N), 0, 0, 0}), Usage: wgpu.BufferUsageUniform})
	defer pBuf.Release()
	bg, err := c.device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: c.mlaHeadMVLayout, Entries: []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: aBuf, Size: aBuf.GetSize()},
		{Binding: 1, Buffer: wBuf, Size: wBuf.GetSize()},
		{Binding: 2, Buffer: dBuf, Size: dBuf.GetSize()},
		{Binding: 3, Buffer: pBuf, Size: pBuf.GetSize()},
	}})
	if err != nil {
		return nil, err
	}
	defer bg.Release()
	enc, _ := c.device.CreateCommandEncoder(nil)
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(c.mlaHeadMVPipeline)
	pass.SetBindGroup(0, bg, nil)
	gx, gy := gemvGrid(nH * N)
	pass.DispatchWorkgroups(gx, gy, 1)
	if err := pass.End(); err != nil {
		pass.Release()
		return nil, err
	}
	pass.Release()
	cmd, _ := enc.Finish(nil)
	defer cmd.Release()
	c.queue.Submit(cmd)
	return c.readbackRaw(dBuf, nH*N)
}

func (c *Context) ensureMLAQRope() error {
	if c.mlaQRopePipeline != nil {
		return nil
	}
	sh, err := c.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "mlaQRope", WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: mlaQRopeWGSL},
	})
	if err != nil {
		return fmt.Errorf("gpu: compile mlaQRope: %w", err)
	}
	pl, err := c.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "mlaQRope", Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return fmt.Errorf("gpu: pipeline mlaQRope: %w", err)
	}
	c.track(sh.Release, pl.Release) // audit C-26: register at creation
	c.mlaQRopeShader, c.mlaQRopePipeline, c.mlaQRopeLayout = sh, pl, c.bgl(pl)
	return nil
}

func (c *Context) ensureMLAAttn() error {
	if c.mlaAttnPipeline != nil {
		return nil
	}
	sh, err := c.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "mlaAttn", WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: mlaAttnWGSL},
	})
	if err != nil {
		return fmt.Errorf("gpu: compile mlaAttn: %w", err)
	}
	pl, err := c.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "mlaAttn", Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return fmt.Errorf("gpu: pipeline mlaAttn: %w", err)
	}
	c.track(sh.Release, pl.Release) // audit C-26: register at creation
	c.mlaAttnShader, c.mlaAttnPipeline, c.mlaAttnLayout = sh, pl, c.bgl(pl)
	return nil
}

// MLAAttn runs the absorb-path rank-space attention on host inputs and returns the
// rank-space V collapse wsum [nH*rank]. qAbs is [nH*latDim] (qNopeAbs ‖ qRope per
// head), lat is [nKeys*latDim] (cn ‖ krj per key), latDim = rank + qkRope. Standalone
// op for parity (the resident runner records the device kernel directly).
func (c *Context) MLAAttn(qAbs, lat []float32, nH, latDim, rank, nKeys int, scale float32) ([]float32, error) {
	if err := c.ensureMLAAttn(); err != nil {
		return nil, err
	}
	if rank > 8*128 {
		return nil, fmt.Errorf("gpu: MLAAttn rank %d exceeds 1024 (per-lane acc cap)", rank)
	}
	qBuf, _ := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "mla-qabs", Contents: wgpu.ToBytes(qAbs), Usage: wgpu.BufferUsageStorage})
	defer qBuf.Release()
	lBuf, _ := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "mla-lat", Contents: wgpu.ToBytes(lat), Usage: wgpu.BufferUsageStorage})
	defer lBuf.Release()
	wBuf, err := c.device.CreateBuffer(&wgpu.BufferDescriptor{Label: "mla-wsum", Size: uint64(nH * rank * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
	if err != nil {
		return nil, err
	}
	defer wBuf.Release()
	pBuf, _ := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "mla-p", Contents: wgpu.ToBytes([]uint32{uint32(nH), uint32(latDim), uint32(rank), uint32(nKeys), f32bits(scale), 0, 0, 0}), Usage: wgpu.BufferUsageUniform})
	defer pBuf.Release()
	bg, err := c.device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: c.mlaAttnLayout, Entries: []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: qBuf, Size: qBuf.GetSize()},
		{Binding: 1, Buffer: lBuf, Size: lBuf.GetSize()},
		{Binding: 2, Buffer: wBuf, Size: wBuf.GetSize()},
		{Binding: 3, Buffer: pBuf, Size: pBuf.GetSize()},
	}})
	if err != nil {
		return nil, err
	}
	defer bg.Release()
	enc, _ := c.device.CreateCommandEncoder(nil)
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(c.mlaAttnPipeline)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(uint32(nH), 1, 1) // one workgroup per query head
	if err := pass.End(); err != nil {
		pass.Release()
		return nil, err
	}
	pass.Release()
	cmd, _ := enc.Finish(nil)
	defer cmd.Release()
	c.queue.Submit(cmd)
	return c.readbackRaw(wBuf, nH*rank)
}
