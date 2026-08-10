//go:build darwin

package metal

import (
	"os"
	"testing"
	"time"
)

// TestZZ_attnM3ThreadWidth — plan §M3 first data point: the Metal FA go/no-go. On M=1 decode the
// FA "don't materialize the score vector" benefit does NOT apply (the score vector is only nKeys
// floats), so the only lever an FA-style rewrite has over the shipped one-threadgroup-per-head serial
// pass is MORE IN-FLIGHT PARALLELISM to hide the DRAM-latency wall the half-width probe found (q8
// moved attention only 12% → latency-bound, not byte-bound). This sweeps the per-head threadgroup
// WIDTH (128 = shipped → 256 → 512): more threads = fewer keys/thread = more concurrent K/V loads in
// flight per head. If a wider tile materially beats 128, an occupancy/latency lever exists and M3 is
// a GO; if it is flat, the latency wall holds regardless of parallelism and the honest M3 outcome is
// a refutation joining the A2-Metal record ("the lever is elsewhere, or accept the floor").
//
// This is not the FA kernel — it is the go/no-go probe that says whether building one could help.
// Opt-in timing diagnostic, not a gate.
func TestZZ_attnM3ThreadWidth(t *testing.T) {
	if os.Getenv("GOINFER_ATTN_M3_PROBE") == "" {
		t.Skip("plan §M3 Metal go/no-go probe (timing diagnostic, not a gate); set GOINFER_ATTN_M3_PROBE=1")
	}
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("device: %v", err)
	}
	lib, err := d.CompileLibrary(attnM3Src, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "attn_tw")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}

	const nL, nH, nKV, hd, nKeys = 28, 12, 2, 128, 2048 // qwen1.5b @ depth 2048
	const kvDim = nKV * hd
	q := d.NewBufferBytes(nH * hd * 4)
	out := d.NewBufferBytes(nH * hd * 4)
	kc, vc := d.NewBufferBytes(nKeys*kvDim*2), d.NewBufferBytes(nKeys*kvDim*2)
	uNH, uNKV, uHd := d.NewBufferU32(nH), d.NewBufferU32(nKV), d.NewBufferU32(hd)
	uNKeys, uWindow := d.NewBufferU32(nKeys), d.NewBufferU32(0)
	uScale := d.NewBufferFloats([]float32{1.0 / 11.3137})

	qu := d.NewCommandQueue()
	arp := NewARPool()
	defer arp.Drain()

	run := func(tw int) float64 {
		best := time.Hour.Seconds()
		for rep := 0; rep < 20; rep++ {
			e := qu.BeginNP()
			for l := 0; l < nL; l++ {
				e.Dispatch(pipe, nH*tw, tw, q, kc, vc, out, uNH, uNKV, uHd, uNKeys, uScale, uWindow)
			}
			e.FinishEncoding()
			e.Commit()
			e.WaitDone()
			if g := e.GPUEnd() - e.GPUStart(); g < best {
				best = g
			}
		}
		return best * 1e3
	}

	run(128) // warm
	base := run(128)
	tw256 := run(256)
	tw512 := run(512)

	t.Logf("all-%d-layer attention @%d keys, per-head threadgroup width sweep (M1 Pro, min GPU-busy/20):", nL, nKeys)
	t.Logf("  tw=128 (shipped) %.2f ms", base)
	t.Logf("  tw=256           %.2f ms  (%.0f%% of shipped)", tw256, tw256/base*100)
	t.Logf("  tw=512           %.2f ms  (%.0f%% of shipped)", tw512, tw512/base*100)
	bestFrac := tw256 / base
	if tw512/base < bestFrac {
		bestFrac = tw512 / base
	}
	if bestFrac < 0.85 {
		t.Logf("  → best wide tile costs %.0f%% of shipped → M3 GO SIGNAL: more in-flight parallelism beats the latency wall; an FA-style rewrite with wider key tiling is worth prototyping", bestFrac*100)
	} else {
		t.Logf("  → best wide tile costs %.0f%% of shipped (≥85%%) → M3 REFUTATION: widening the head threadgroup does not beat the DRAM-latency wall; parallelism is not the lever (joins the A2-Metal record)", bestFrac*100)
	}
}

// attn_tw is the shipped decode `attention` kernel with the reduction scratch sized for a variable
// threadgroup width (red[512] vs the shipped red[128]) so the same kernel runs at tw ∈ {128,256,512}.
// Timing-only — the denominator reduction width is not the shipped 128, so output is not bit-exact;
// this probe measures DRAM-latency hiding, not correctness.
const attnM3Src = `
#include <metal_stdlib>
using namespace metal;
kernel void attn_tw(device const float* q[[buffer(0)]], device const half* kc[[buffer(1)]],
    device const half* vc[[buffer(2)]], device float* out[[buffer(3)]], constant uint& nH[[buffer(4)]],
    constant uint& nKV[[buffer(5)]], constant uint& hd[[buffer(6)]], constant uint& nKeys[[buffer(7)]],
    constant float& scale[[buffer(8)]], constant uint& window[[buffer(9)]],
    uint qh[[threadgroup_position_in_grid]], uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]]) {
    uint kvDim=nKV*hd; uint kvh=qh/(nH/nKV);
    uint winStart=(window>0u&&nKeys>window)?nKeys-window:0u;
    device const float* qr=q+qh*hd;
    device const half* kb=kc+kvh*hd; device const half* vb=vc+kvh*hd;
    threadgroup float sc[4096]; threadgroup float red[512];
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
`
