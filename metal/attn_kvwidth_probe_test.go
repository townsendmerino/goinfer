//go:build darwin

package metal

import (
	"os"
	"testing"
	"time"
)

// TestZZ_attnKVWidthProbe — the P4 deciding measurement (docs/plan-still-slow.md §P4). Re-runs the
// 2026-08-04 collapse probe with a THIRD arm: half the DRAM bytes per key at the SAME element count
// (int8 KV vs the f16 baseline — q8's exact byte profile), to separate BANDWIDTH from LATENCY.
//
// The original probe pinned every K/V read to key 0 (zero distinct DRAM) and saw all-28-layer
// attention 21.5→5.3 ms — 75% of attention is distinct per-key reads. But pinning collapses BOTH
// bytes AND latency (key 0 stays cached), so it cannot say whether q8 (fewer bytes, same number of
// serial reads) helps. This probe adds `attn_q8`: hd elements per key at 1 byte each instead of 2,
// same loop / same ALU / same threadgroups, half the DRAM bytes.
//
//   - if q8 time drops ~proportionally toward the full baseline → BANDWIDTH-bound → P4 BUILDS (q8 is a
//     Metal speed lever, CUDA a reachability one).
//   - if q8 barely moves off full → the reads are latency-exposed, bytes aren't the bind → P4 is
//     CUDA-reachability-only (q8 for VRAM, not Metal speed).
//
// `attn_pin0` is the harness self-check: it must reproduce the known collapse (~4× off full), or the
// microbench geometry is wrong and the q8 number is not to be trusted. Opt-in; a timing diagnostic,
// not a gate.
func TestZZ_attnKVWidthProbe(t *testing.T) {
	if os.Getenv("GOINFER_ATTN_KVWIDTH_PROBE") == "" {
		t.Skip("P4 deciding probe (timing diagnostic, not a gate); set GOINFER_ATTN_KVWIDTH_PROBE=1")
	}
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("device: %v", err)
	}
	lib, err := d.CompileLibrary(attnProbeSrc, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	full, err := d.NewComputePipeline(lib, "attn_full")
	if err != nil {
		t.Fatalf("pipeline attn_full: %v", err)
	}
	q8, err := d.NewComputePipeline(lib, "attn_q8")
	if err != nil {
		t.Fatalf("pipeline attn_q8: %v", err)
	}
	pin0, err := d.NewComputePipeline(lib, "attn_pin0")
	if err != nil {
		t.Fatalf("pipeline attn_pin0: %v", err)
	}

	// qwen2.5-coder-1.5b decode-attention geometry at depth 2048 (the collapse-probe cell):
	// 28 layers, nH=12 query heads, nKV=2, hd=128 → kvDim=256; GQA fan-out 6.
	const nL, nH, nKV, hd, nKeys = 28, 12, 2, 128, 2048
	const kvDim = nKV * hd
	scale := float32(1.0 / 11.3137) // 1/sqrt(128); value is timing-irrelevant

	q := d.NewBufferBytes(nH * hd * 4)
	out := d.NewBufferBytes(nH * hd * 4)
	kc16, vc16 := d.NewBufferBytes(nKeys*kvDim*2), d.NewBufferBytes(nKeys*kvDim*2) // f16
	kc8, vc8 := d.NewBufferBytes(nKeys*kvDim*1), d.NewBufferBytes(nKeys*kvDim*1)   // int8 (half the bytes)
	uNH, uNKV, uHd := d.NewBufferU32(nH), d.NewBufferU32(nKV), d.NewBufferU32(hd)
	uNKeys, uWindow := d.NewBufferU32(nKeys), d.NewBufferU32(0)
	uScale := d.NewBufferFloats([]float32{scale})

	qu := d.NewCommandQueue()
	arp := NewARPool()
	defer arp.Drain()

	// One submit = all nL layers dispatched (grid = nH threadgroups × 128 threads), timed GPU-busy.
	run := func(pipe Pipeline, kc, vc Buffer) float64 {
		best := time.Hour.Seconds()
		for rep := 0; rep < 20; rep++ {
			e := qu.BeginNP()
			for l := 0; l < nL; l++ {
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
	run(full, kc16, vc16)
	tFull := run(full, kc16, vc16)
	tQ8 := run(q8, kc8, vc8)
	tPin0 := run(pin0, kc16, vc16)

	// Harness self-check: pin0 must collapse (~4× off full), matching the 2026-08-04 21.5→5.3 ms.
	pinRatio := tFull / tPin0
	q8Frac := tQ8 / tFull // fraction of full-width time the half-byte read still costs

	t.Logf("all-%d-layer attention @%d keys (M1 Pro, min GPU-busy over 20 reps):", nL, nKeys)
	t.Logf("  full (f16 KV)      %.2f ms", tFull)
	t.Logf("  q8   (int8 KV)     %.2f ms   (%.0f%% of full)", tQ8, q8Frac*100)
	t.Logf("  pin0 (key-0 reads) %.2f ms   (full/pin0 = %.1fx — collapse control)", tPin0, pinRatio)
	if pinRatio < 2.0 {
		t.Errorf("HARNESS SUSPECT: pin0 collapse only %.1fx (expected ~4x per the 2026-08-04 probe) — "+
			"geometry likely wrong; do not trust the q8 number", pinRatio)
	}
	// Verdict, stated for the report (not a pass/fail):
	verdict := "P4 CUDA-REACHABILITY-ONLY: halving KV bytes barely moved attention → latency/occupancy-bound, not bandwidth; q8 is not a Metal speed lever"
	if q8Frac < 0.75 {
		verdict = "P4 BUILDS on Metal: halving KV bytes dropped attention ~proportionally → bandwidth-bound; q8 is a Metal speed lever"
	}
	t.Logf("  → q8 costs %.0f%% of full → %s", q8Frac*100, verdict)
}

// Three attention kernels, identical to the production `attention` (kernels.go) in loop shape, ALU,
// threadgroup traffic, and reduction width — differing ONLY in the KV DRAM footprint per key:
//
//	attn_full: f16 K/V, hd elements × 2 bytes (the shipping kernel).
//	attn_q8:   int8 K/V, hd elements × 1 byte — half the bytes, SAME element count / ALU (q8's profile).
//	attn_pin0: f16 K/V but every key read pinned to key 0 — zero distinct DRAM (collapse control).
const attnProbeSrc = `
#include <metal_stdlib>
using namespace metal;

kernel void attn_full(device const float* q[[buffer(0)]], device const half* kc[[buffer(1)]],
    device const half* vc[[buffer(2)]], device float* out[[buffer(3)]], constant uint& nH[[buffer(4)]],
    constant uint& nKV[[buffer(5)]], constant uint& hd[[buffer(6)]], constant uint& nKeys[[buffer(7)]],
    constant float& scale[[buffer(8)]], constant uint& window[[buffer(9)]],
    uint qh[[threadgroup_position_in_grid]], uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]]) {
    uint kvDim=nKV*hd; uint kvh=qh/(nH/nKV);
    uint winStart=(window>0u&&nKeys>window)?nKeys-window:0u;
    device const float* qr=q+qh*hd;
    device const half* kb=kc+kvh*hd; device const half* vb=vc+kvh*hd;
    threadgroup float sc[4096]; threadgroup float red[128];
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

kernel void attn_q8(device const float* q[[buffer(0)]], device const char* kc[[buffer(1)]],
    device const char* vc[[buffer(2)]], device float* out[[buffer(3)]], constant uint& nH[[buffer(4)]],
    constant uint& nKV[[buffer(5)]], constant uint& hd[[buffer(6)]], constant uint& nKeys[[buffer(7)]],
    constant float& scale[[buffer(8)]], constant uint& window[[buffer(9)]],
    uint qh[[threadgroup_position_in_grid]], uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]]) {
    uint kvDim=nKV*hd; uint kvh=qh/(nH/nKV);
    uint winStart=(window>0u&&nKeys>window)?nKeys-window:0u;
    device const float* qr=q+qh*hd;
    device const char* kb=kc+kvh*hd; device const char* vb=vc+kvh*hd;
    threadgroup float sc[4096]; threadgroup float red[128];
    for(uint s=winStart+tid;s<nKeys;s+=tgs){ float a=0; device const char* k=kb+s*kvDim; uint dd=0;
        for(;dd<hd;dd+=4u){ char4 k4=*((device const char4*)(k+dd)); a+=qr[dd]*float(k4.x);a+=qr[dd+1u]*float(k4.y);a+=qr[dd+2u]*float(k4.z);a+=qr[dd+3u]*float(k4.w);}
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

kernel void attn_pin0(device const float* q[[buffer(0)]], device const half* kc[[buffer(1)]],
    device const half* vc[[buffer(2)]], device float* out[[buffer(3)]], constant uint& nH[[buffer(4)]],
    constant uint& nKV[[buffer(5)]], constant uint& hd[[buffer(6)]], constant uint& nKeys[[buffer(7)]],
    constant float& scale[[buffer(8)]], constant uint& window[[buffer(9)]],
    uint qh[[threadgroup_position_in_grid]], uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]]) {
    uint kvDim=nKV*hd; uint kvh=qh/(nH/nKV);
    uint winStart=(window>0u&&nKeys>window)?nKeys-window:0u;
    device const float* qr=q+qh*hd;
    device const half* kb=kc+kvh*hd; device const half* vb=vc+kvh*hd;
    threadgroup float sc[4096]; threadgroup float red[128];
    for(uint s=winStart+tid;s<nKeys;s+=tgs){ float a=0; device const half* k=kb; uint dd=0; /* pinned key 0 */
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
    for(uint dd=tid;dd<hd;dd+=tgs){float a=0;for(uint s=winStart;s<nKeys;s++)a+=sc[s]*float(vb[dd]);out[qh*hd+dd]=a/sum;}
}
`
