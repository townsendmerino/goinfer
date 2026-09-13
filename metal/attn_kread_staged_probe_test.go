//go:build darwin

package metal

import (
	"math"
	"os"
	"testing"
	"time"
)

// TestZZ_attnKReadStagedProbe — M-09's discriminating probe (audit-metal-2026-09-12.md): does
// cooperatively STAGING each key-tile through threadgroup memory (a coalesced device->threadgroup
// copy, then per-thread compute from the fast on-chip copy) beat the shipped kernel's direct
// per-thread device reads, at the shipped 12-threadgroup/128-thread decode grid and nKeys=2048?
//
// The shipped K-read has thread tid own keys tid, tid+128, ...; at any fixed loop iteration
// (a fixed dimension d, since the d-loop is the innermost and runs in lockstep across a
// simdgroup), the 32 lanes of one simdgroup read 32 DIFFERENT keys' element d — addresses
// kvDim elements (512 B at hd=128) apart, so the load is NOT coalesced: 32 lanes touch 32
// distinct cache lines per instruction (audit's "512 B-strided gather" finding).
//
// attn_staged instead loads a TILE of TILE_KEYS=64 keys into threadgroup memory using the
// natural flat stride-128 pattern (thread tid reads flat offsets tid, tid+128, ...  of the
// TILE_KEYS*hd region) — which IS coalesced, since adjacent threads read adjacent addresses —
// then each thread computes its OWN keys' dot products by reading the STAGED copy instead of
// device memory. Reduction code (softmax, PV) is untouched — only the K-read+score phase
// differs, isolating exactly what M-09 targets. TILE_KEYS=64 (not 128, matching the
// full threadgroup width) because 128*hd*2 bytes (32 KiB) alone would exceed the device's
// threadgroup-memory budget once sc[nKeys] shares it; 64 keys (16 KiB) leaves headroom.
//
// Opt-in; a timing diagnostic, not a gate. If tStaged is not materially faster than tShipped,
// the DRAM-latency reading M-09 names stands and this closes (audit's own bar); the harness
// asserts nothing either way — read the numbers.
func TestZZ_attnKReadStagedProbe(t *testing.T) {
	if os.Getenv("GOINFER_ATTN_KREAD_STAGED_PROBE") == "" {
		t.Skip("M-09 discriminating probe (timing diagnostic, not a gate); set GOINFER_ATTN_KREAD_STAGED_PROBE=1")
	}
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("device: %v", err)
	}
	lib, err := d.CompileLibrary(attnKReadStagedProbeSrc, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	shipped, err := d.NewComputePipeline(lib, "attn_shipped")
	if err != nil {
		t.Fatalf("pipeline attn_shipped: %v", err)
	}
	staged, err := d.NewComputePipeline(lib, "attn_staged")
	if err != nil {
		t.Fatalf("pipeline attn_staged: %v", err)
	}

	// qwen2.5-coder-1.5b decode-attention geometry at depth 2048 (M-09's own cited shape):
	// 28 layers, nH=12 query heads, nKV=2, hd=128 -> kvDim=256; GQA fan-out 6.
	const nL, nH, nKV, hd, nKeys = 28, 12, 2, 128, 2048
	const kvDim = nKV * hd
	scale := float32(1.0 / 11.3137) // 1/sqrt(128); value is timing-irrelevant

	q := d.NewBufferBytes(nH * hd * 4)
	outShipped := d.NewBufferBytes(nH * hd * 4)
	outStaged := d.NewBufferBytes(nH * hd * 4)
	kc, vc := d.NewBufferBytes(nKeys*kvDim*2), d.NewBufferBytes(nKeys*kvDim*2) // f16
	uNH, uNKV, uHd := NewBufferU32(d, nH), NewBufferU32(d, nKV), NewBufferU32(d, hd)
	uNKeys, uWindow := NewBufferU32(d, nKeys), NewBufferU32(d, 0)
	uScale := NewBufferFloats(d, []float32{scale})

	qu := d.NewCommandQueue()
	arp := NewARPool()
	defer arp.Drain()

	run := func(pipe Pipeline, out Buffer) float64 {
		best := time.Hour.Seconds()
		for range 20 {
			e := qu.BeginNP()
			for range nL {
				e.Dispatch(pipe, nH*128, 128, q, kc, vc, out, uNH, uNKV, uHd, uNKeys, uScale, uWindow)
			}
			e.FinishEncoding()
			e.Commit()
			e.WaitDone()
			g := e.GPUEnd() - e.GPUStart()
			if g < best {
				best = g
			}
		}
		return best * 1e3 // ms for all nL layers
	}

	// warm
	run(shipped, outShipped)
	tShipped := run(shipped, outShipped)
	tStaged := run(staged, outStaged)

	// Correctness first: a faster-but-wrong kernel proves nothing. Same tolerance class as the
	// production reduction-order contract (float add is non-associative; staging changes the
	// LOAD order, not the accumulation order, so this should be very tight, not just "close").
	sOut, gOut := outShipped.Floats(), outStaged.Floats()
	var maxAbs float64
	for i := range sOut {
		diff := math.Abs(float64(sOut[i]) - float64(gOut[i]))
		if diff > maxAbs {
			maxAbs = diff
		}
	}

	ratio := tShipped / tStaged
	t.Logf("all-%d-layer attention K-read @%d keys (M1 Pro, min GPU-busy over 20 reps):", nL, nKeys)
	t.Logf("  shipped (scattered device reads)  %.3f ms", tShipped)
	t.Logf("  staged  (coalesced->threadgroup)   %.3f ms  (%.2fx shipped)", tStaged, ratio)
	t.Logf("  correctness: maxAbs=%.6g (staged vs shipped, same output for same inputs)", maxAbs)
	if maxAbs > 1e-3 {
		t.Errorf("attn_staged disagrees with attn_shipped: maxAbs=%.6g — a staging bug, not a timing question; "+
			"the ratio above is not meaningful until this is fixed", maxAbs)
	}
	verdict := "CLOSES: staging is not materially faster (DRAM-latency reading stands, not bandwidth) — do not port"
	if ratio > 1.15 {
		verdict = "BUILDS: staging materially faster — port into attention (kernels.go), gate byte-exact against the snapshot golden at context > 128"
	}
	t.Logf("  -> %.2fx -> %s", ratio, verdict)
}

const attnKReadStagedProbeSrc = `
#include <metal_stdlib>
using namespace metal;

// attn_shipped: byte-for-byte the production attention kernel's K-read + reduction shape
// (kernels.go), narrowed to this probe's fixed nKeys=2048/hd=128 geometry.
kernel void attn_shipped(device const float* q[[buffer(0)]], device const half* kc[[buffer(1)]],
    device const half* vc[[buffer(2)]], device float* out[[buffer(3)]], constant uint& nH[[buffer(4)]],
    constant uint& nKV[[buffer(5)]], constant uint& hd[[buffer(6)]], constant uint& nKeys[[buffer(7)]],
    constant float& scale[[buffer(8)]], constant uint& window[[buffer(9)]],
    uint qh[[threadgroup_position_in_grid]], uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]]) {
    uint kvDim=nKV*hd; uint kvh=qh/(nH/nKV);
    uint winStart=(window>0u&&nKeys>window)?nKeys-window:0u;
    device const float* qr=q+qh*hd;
    device const half* kb=kc+kvh*hd; device const half* vb=vc+kvh*hd;
    threadgroup float sc[2048]; threadgroup float red[128];
    for(uint s=winStart+tid;s<nKeys;s+=tgs){ float a=0; device const half* k=kb+s*kvDim; uint dd=0;
        for(;dd<hd;dd+=4u){ half4 k4=*((device const half4*)(k+dd)); a+=qr[dd]*float(k4.x);a+=qr[dd+1u]*float(k4.y);a+=qr[dd+2u]*float(k4.z);a+=qr[dd+3u]*float(k4.w);}
        sc[s]=a*scale; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m=-INFINITY; for(uint s=winStart+tid;s<nKeys;s+=tgs)m=max(m,sc[s]);
    red[tid]=m; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint st=tgs/2;st>0;st>>=1){if(tid<st)red[tid]=max(red[tid],red[tid+st]);threadgroup_barrier(mem_flags::mem_threadgroup);}
    float mx=red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
    float ls=0; for(uint s=winStart+tid;s<nKeys;s+=tgs){float p=exp(sc[s]-mx);sc[s]=p;ls+=p;}
    red[tid]=ls; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint st=tgs/2;st>0;st>>=1){if(tid<st)red[tid]+=red[tid+st];threadgroup_barrier(mem_flags::mem_threadgroup);}
    float sum=red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint dd=tid;dd<hd;dd+=tgs){float a=0;for(uint s=winStart;s<nKeys;s++)a+=sc[s]*float(vb[s*kvDim+dd]);out[qh*hd+dd]=a/sum;}
}

// attn_staged: the SAME reduction/PV code as attn_shipped, but the K-read is tiled through
// threadgroup memory. TILE_KEYS keys are loaded per round via a coalesced flat stride-tgs copy
// (thread tid reads flat offsets tid, tid+tgs, ... of the TILE_KEYS*hd region — adjacent
// threads, adjacent addresses), then each thread computes its OWN keys' dot products by
// reading the staged (on-chip) copy at the SAME per-key access pattern the shipped kernel uses
// against device memory — only the SOURCE of the read changes.
#define TILE_KEYS 64u
kernel void attn_staged(device const float* q[[buffer(0)]], device const half* kc[[buffer(1)]],
    device const half* vc[[buffer(2)]], device float* out[[buffer(3)]], constant uint& nH[[buffer(4)]],
    constant uint& nKV[[buffer(5)]], constant uint& hd[[buffer(6)]], constant uint& nKeys[[buffer(7)]],
    constant float& scale[[buffer(8)]], constant uint& window[[buffer(9)]],
    uint qh[[threadgroup_position_in_grid]], uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]]) {
    uint kvDim=nKV*hd; uint kvh=qh/(nH/nKV);
    uint winStart=(window>0u&&nKeys>window)?nKeys-window:0u;
    device const float* qr=q+qh*hd;
    device const half* kb=kc+kvh*hd; device const half* vb=vc+kvh*hd;
    threadgroup float sc[2048]; threadgroup float red[128];
    threadgroup half kStage[TILE_KEYS*128u]; // 64*128*2B = 16 KiB
    for (uint tileBase=winStart; tileBase<nKeys; tileBase+=TILE_KEYS) {
        uint tileLen = min(TILE_KEYS, nKeys-tileBase);
        device const half* tileSrc = kb + (ulong)tileBase*kvDim;
        for (uint i=tid; i<tileLen*hd; i+=tgs) {
            uint kk = i/hd, dd = i%hd;
            kStage[kk*hd+dd] = tileSrc[(ulong)kk*kvDim+dd];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s=tileBase+tid; s<tileBase+tileLen; s+=tgs) {
            float a=0; threadgroup const half* k = kStage + (s-tileBase)*hd; uint dd=0;
            for(;dd<hd;dd+=4u){ half4 k4=*((threadgroup const half4*)(k+dd)); a+=qr[dd]*float(k4.x);a+=qr[dd+1u]*float(k4.y);a+=qr[dd+2u]*float(k4.z);a+=qr[dd+3u]*float(k4.w);}
            sc[s]=a*scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float m=-INFINITY; for(uint s=winStart+tid;s<nKeys;s+=tgs)m=max(m,sc[s]);
    red[tid]=m; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint st=tgs/2;st>0;st>>=1){if(tid<st)red[tid]=max(red[tid],red[tid+st]);threadgroup_barrier(mem_flags::mem_threadgroup);}
    float mx=red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
    float ls=0; for(uint s=winStart+tid;s<nKeys;s+=tgs){float p=exp(sc[s]-mx);sc[s]=p;ls+=p;}
    red[tid]=ls; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint st=tgs/2;st>0;st>>=1){if(tid<st)red[tid]+=red[tid+st];threadgroup_barrier(mem_flags::mem_threadgroup);}
    float sum=red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint dd=tid;dd<hd;dd+=tgs){float a=0;for(uint s=winStart;s<nKeys;s++)a+=sc[s]*float(vb[s*kvDim+dd]);out[qh*hd+dd]=a/sum;}
}
`
