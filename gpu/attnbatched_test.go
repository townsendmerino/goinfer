//go:build gpu

package gpu

import (
	"math/rand"
	"testing"

	"github.com/oliverbestmann/webgpu/wgpu"
)

// TestAttnBatched_parity gates docs/completed/task-gpu-batched-prefill.md's Increment 1: does ONE
// grid-(nH,M) dispatch of attnBatchedKernel's chosen kernel match M separate
// dispatches of attnKernel's chosen kernel (the shape PrefillLastW8A8 used before
// this fix, and what decode still uses today) over the SAME fully-pre-populated
// K/V cache? No checkpoint needed — synthetic Q/K/V, a fresh cache (basePos=0), M
// rows each attending to keys [0, row].
//
// Two geometries, one per attnKernel/attnBatchedKernel branch (attention.go's
// attnKeysEligible): hd=64/kvDim=128 (both %4==0, both KEYS-eligible — the common
// dense-architecture shape, e.g. qwen2.5-coder-0.5b) and hd=48/kvDim=48 (kvDim not
// a multiple of 4 via nKV=1 group — falls to the plain kernel).
//
// NOT bit-exact, unlike TestRMSNormBatched_parity/TestRoPEBatched_parity: attention
// kernels in this file are never bit-identical to each other even for the SAME
// math (attnKeysShaderWGSL's own comment: "the denominator sums in a different
// order and the tiled rescale reassociates" vs attnShaderWGSL) — cosine/maxAbs,
// the same standard TestAttention_parity holds every attention kernel pair to.
//
//	go test -tags gpu ./gpu/ -run TestAttnBatched_parity -v
func TestAttnBatched_parity(t *testing.T) {
	c := newOrSkipHW(t)
	defer c.Close()
	for _, f := range []func() error{c.ensureAttn, c.ensurePrefillBatched} {
		if err := f(); err != nil {
			t.Fatalf("ensure: %v", err)
		}
	}

	cases := []struct {
		name         string
		nH, nKV, hd  int
		wantKeysPath bool
	}{
		{"keys-eligible(hd=64,kvDim=128)", 8, 4, 64, true},
		{"plain-fallback(hd=50,kvDim=50)", 4, 1, 50, false}, // 50 % 4 != 0 -> attnKeysEligible declines
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := attnKeysEligible(tc.hd, tc.nKV*tc.hd, false, false); got != tc.wantKeysPath {
				t.Fatalf("attnKeysEligible(hd=%d,kvDim=%d) = %v, want %v (test geometry doesn't exercise the branch it claims to)",
					tc.hd, tc.nKV*tc.hd, got, tc.wantKeysPath)
			}
			const M = 23
			kvDim := tc.nKV * tc.hd
			group := tc.nH / tc.nKV
			const scale = 0.125

			rng := rand.New(rand.NewSource(1))
			qData := make([]float32, M*tc.nH*tc.hd)
			for i := range qData {
				qData[i] = float32(rng.NormFloat64()) * 0.6
			}
			// Cache sized exactly M rows: row m (0-indexed) attends to keys [0,m], so the
			// last row's causal window is the whole cache — a fresh prefill from position 0.
			kData := make([]float32, M*kvDim)
			vData := make([]float32, M*kvDim)
			for i := range kData {
				kData[i] = float32(rng.NormFloat64()) * 0.6
				vData[i] = float32(rng.NormFloat64()) * 0.6
			}

			qBuf, err := c.device.TryCreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(qData), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
			if err != nil {
				t.Fatalf("upload q: %v", err)
			}
			defer qBuf.Release()
			kBuf, err := c.device.TryCreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(kData), Usage: wgpu.BufferUsageStorage})
			if err != nil {
				t.Fatalf("upload k: %v", err)
			}
			defer kBuf.Release()
			vBuf, err := c.device.TryCreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(vData), Usage: wgpu.BufferUsageStorage})
			if err != nil {
				t.Fatalf("upload v: %v", err)
			}
			defer vBuf.Release()

			// REFERENCE: M separate dispatches of attnKernel's chosen pipeline — the shape
			// PrefillLastW8A8 used before Increment 1, and what decode still uses today.
			refBuf, err := c.device.TryCreateBuffer(&wgpu.BufferDescriptor{Size: uint64(M * tc.nH * tc.hd * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst})
			if err != nil {
				t.Fatalf("alloc refBuf: %v", err)
			}
			defer refBuf.Release()
			noSinks, err := c.device.TryCreateBuffer(&wgpu.BufferDescriptor{Size: 4, Usage: wgpu.BufferUsageStorage})
			if err != nil {
				t.Fatalf("alloc noSinks: %v", err)
			}
			defer noSinks.Release()
			noHasSink, err := c.device.TryCreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes([]uint32{0, 0, 0, 0}), Usage: wgpu.BufferUsageUniform})
			if err != nil {
				t.Fatalf("alloc noHasSink: %v", err)
			}
			defer noHasSink.Release()
			refPl, refLy := c.attnKernel(tc.hd, kvDim, false)
			{
				enc, err := c.device.TryCreateCommandEncoder(nil)
				if err != nil {
					t.Fatalf("encoder: %v", err)
				}
				defer enc.Release()
				for row := 0; row < M; row++ {
					qRow, err := c.device.TryCreateBuffer(&wgpu.BufferDescriptor{Size: uint64(tc.nH * tc.hd * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst})
					if err != nil {
						t.Fatalf("alloc qRow: %v", err)
					}
					defer qRow.Release()
					if err := enc.TryCopyBufferToBuffer(qBuf, uint64(row*tc.nH*tc.hd*4), qRow, 0, uint64(tc.nH*tc.hd*4)); err != nil {
						t.Fatalf("copy qRow row %d: %v", row, err)
					}
					p, err := c.device.TryCreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes([]uint32{uint32(tc.nH), uint32(tc.nKV), uint32(tc.hd), uint32(row + 1), 0, uint32(group), f32bits(scale), 0}), Usage: wgpu.BufferUsageUniform})
					if err != nil {
						t.Fatalf("alloc p row %d: %v", row, err)
					}
					defer p.Release()
					cv, err := c.device.TryCreateBuffer(&wgpu.BufferDescriptor{Size: uint64(tc.nH * tc.hd * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
					if err != nil {
						t.Fatalf("alloc cv row %d: %v", row, err)
					}
					defer cv.Release()
					bg, err := c.device.TryCreateBindGroup(&wgpu.BindGroupDescriptor{Layout: refLy, Entries: []wgpu.BindGroupEntry{
						{Binding: 0, Buffer: qRow, Size: qRow.GetSize()}, {Binding: 1, Buffer: kBuf, Size: kBuf.GetSize()},
						{Binding: 2, Buffer: vBuf, Size: vBuf.GetSize()}, {Binding: 3, Buffer: cv, Size: cv.GetSize()},
						{Binding: 4, Buffer: noSinks, Size: noSinks.GetSize()}, {Binding: 5, Buffer: p, Size: p.GetSize()},
						{Binding: 6, Buffer: noHasSink, Size: noHasSink.GetSize()},
					}})
					if err != nil {
						t.Fatalf("bindgroup row %d: %v", row, err)
					}
					defer bg.Release()
					pass := enc.BeginComputePass(nil)
					pass.SetPipeline(refPl)
					pass.SetBindGroup(0, bg, nil)
					pass.DispatchWorkgroups(uint32(tc.nH), 1, 1)
					pass.TryEnd()
					pass.Release()
					if err := enc.TryCopyBufferToBuffer(cv, 0, refBuf, uint64(row*tc.nH*tc.hd*4), uint64(tc.nH*tc.hd*4)); err != nil {
						t.Fatalf("copy out row %d: %v", row, err)
					}
				}
				cmd, err := enc.TryFinish(nil)
				if err != nil {
					t.Fatalf("finish: %v", err)
				}
				defer cmd.Release()
				c.queue.Submit(cmd)
			}

			// UNDER TEST: one grid-(nH,M) dispatch of attnBatchedKernel's chosen pipeline.
			gotBuf, err := c.device.TryCreateBuffer(&wgpu.BufferDescriptor{Size: uint64(M * tc.nH * tc.hd * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
			if err != nil {
				t.Fatalf("alloc gotBuf: %v", err)
			}
			defer gotBuf.Release()
			pB, err := c.device.TryCreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes([]uint32{uint32(tc.nH), uint32(tc.nKV), uint32(tc.hd), 0, uint32(group), f32bits(scale), uint32(M), 0}), Usage: wgpu.BufferUsageUniform})
			if err != nil {
				t.Fatalf("alloc pB: %v", err)
			}
			defer pB.Release()
			batchedPl, batchedLy := c.attnBatchedKernel(tc.hd, kvDim)
			bgB, err := c.device.TryCreateBindGroup(&wgpu.BindGroupDescriptor{Layout: batchedLy, Entries: []wgpu.BindGroupEntry{
				{Binding: 0, Buffer: qBuf, Size: qBuf.GetSize()}, {Binding: 1, Buffer: kBuf, Size: kBuf.GetSize()},
				{Binding: 2, Buffer: vBuf, Size: vBuf.GetSize()}, {Binding: 3, Buffer: gotBuf, Size: gotBuf.GetSize()},
				{Binding: 4, Buffer: pB, Size: pB.GetSize()},
			}})
			if err != nil {
				t.Fatalf("bindgroup batched: %v", err)
			}
			defer bgB.Release()
			{
				enc, err := c.device.TryCreateCommandEncoder(nil)
				if err != nil {
					t.Fatalf("encoder B: %v", err)
				}
				defer enc.Release()
				pass := enc.BeginComputePass(nil)
				pass.SetPipeline(batchedPl)
				pass.SetBindGroup(0, bgB, nil)
				pass.DispatchWorkgroups(uint32(tc.nH), uint32(M), 1)
				pass.TryEnd()
				pass.Release()
				cmd, err := enc.TryFinish(nil)
				if err != nil {
					t.Fatalf("finish B: %v", err)
				}
				defer cmd.Release()
				c.queue.Submit(cmd)
			}

			ref := readF32(t, c, refBuf, M*tc.nH*tc.hd)
			got := readF32(t, c, gotBuf, M*tc.nH*tc.hd)
			cos, maxAbs := cosine(ref, got)
			t.Logf("M=%d nH=%d nKV=%d hd=%d: cosine=%.9f maxAbs=%.3e", M, tc.nH, tc.nKV, tc.hd, cos, maxAbs)
			if cos < 0.999999 || maxAbs > 1e-4 {
				t.Fatalf("batched attention diverges from M sequential attnKernel dispatches: cosine=%.9f maxAbs=%.3e (want cosine>=0.999999, maxAbs<=1e-4)", cos, maxAbs)
			}
		})
	}
}
