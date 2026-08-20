//go:build gpu

package gpu

import (
	"math"
	"math/rand"
	"strconv"
	"strings"
	"testing"

	"github.com/cogentcore/webgpu/wgpu"
)

// The WIDE single-query attention kernel vs refAttn, across head dims the narrow kernel cannot
// reach — and, deliberately, across ones it can.
//
// Why the small dims are in the list: the wide kernel's whole trick is that a lane owns a STRIDE
// of dims, so the interesting cases are the boundaries of that striding. hd=64 gives a workgroup
// where most lanes own nothing; hd=256 gives exactly one dim per lane (nper=1, the degenerate
// case where a striding bug hides); hd=257 and hd=512 give a partial and a full second stride.
// A test that only ran 256 would pass with the stride loop entirely broken.
//
// refAttn is the same Go reference TestAttnBlock_parity already uses — not a new one written to
// match this kernel.
func TestAttnWide_refParity(t *testing.T) {
	ctx := newOrSkipHW(t)
	defer ctx.Close()
	if err := ctx.ensureAttnWide(); err != nil {
		t.Skipf("wide attention unavailable: %v", err)
	}

	for _, tc := range []struct{ hd, nH, nKV, nKeys int }{
		{64, 4, 2, 7},   // far below the workgroup: most lanes idle, nper=1
		{128, 4, 4, 5},  // the narrow kernel's exact ceiling
		{256, 8, 2, 13}, // one dim per lane — nper=1, where a stride bug hides
		{257, 2, 1, 3},  // ragged second stride: lane 0 owns 2 dims, the rest own 1
		{512, 4, 2, 11}, // two full strides
		{1024, 2, 1, 6}, // four
	} {
		name := "hd" + strconv.Itoa(tc.hd)
		t.Run(name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(tc.hd)))
			rnd := func(n int) []float32 {
				s := make([]float32, n)
				for i := range s {
					s[i] = float32(rng.NormFloat64())
				}
				return s
			}
			kvDim := tc.nKV * tc.hd
			q, keys, vals := rnd(tc.nH*tc.hd), rnd(tc.nKeys*kvDim), rnd(tc.nKeys*kvDim)
			scale := float32(1 / math.Sqrt(float64(tc.hd)))
			want := refAttn(q, keys, vals, tc.nH, tc.nKV, tc.hd, tc.nKeys, scale)

			mk := func(c []float32, u wgpu.BufferUsage) *wgpu.Buffer {
				b, e := ctx.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(c), Usage: u})
				if e != nil {
					t.Fatal(e)
				}
				return b
			}
			qB, kB, vB := mk(q, wgpu.BufferUsageStorage), mk(keys, wgpu.BufferUsageStorage), mk(vals, wgpu.BufferUsageStorage)
			ctxB := mk(make([]float32, tc.nH*tc.hd), wgpu.BufferUsageStorage|wgpu.BufferUsageCopySrc)
			n := tc.nH * tc.hd
			stag, _ := ctx.device.CreateBuffer(&wgpu.BufferDescriptor{Size: uint64(n * 4), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
			uni, _ := ctx.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
				Contents: wgpu.ToBytes([]uint32{uint32(tc.nH), uint32(tc.nKV), uint32(tc.hd), uint32(tc.nKeys),
					0, uint32(tc.nH / tc.nKV), math.Float32bits(scale), 0}),
				Usage: wgpu.BufferUsageUniform})
			bg, e := ctx.device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: ctx.attnWideLayout, Entries: []wgpu.BindGroupEntry{
				{Binding: 0, Buffer: qB, Size: qB.GetSize()}, {Binding: 1, Buffer: kB, Size: kB.GetSize()},
				{Binding: 2, Buffer: vB, Size: vB.GetSize()}, {Binding: 3, Buffer: ctxB, Size: ctxB.GetSize()},
				{Binding: 4, Buffer: uni, Size: uni.GetSize()},
			}})
			if e != nil {
				t.Fatal(e)
			}
			defer func() {
				bg.Release()
				for _, b := range []*wgpu.Buffer{qB, kB, vB, ctxB, stag, uni} {
					b.Release()
				}
			}()

			enc, _ := ctx.device.CreateCommandEncoder(nil)
			pass := enc.BeginComputePass(nil)
			pass.SetPipeline(ctx.attnWidePipeline)
			pass.SetBindGroup(0, bg, nil)
			pass.DispatchWorkgroups(uint32(tc.nH), 1, 1)
			pass.End()
			pass.Release()
			enc.CopyBufferToBuffer(ctxB, 0, stag, 0, uint64(n*4))
			cmd, _ := enc.Finish(nil)
			ctx.queue.Submit(cmd)
			cmd.Release()
			enc.Release()

			st := wgpu.BufferMapAsyncStatusUnknown
			stag.MapAsync(wgpu.MapModeRead, 0, uint64(n*4), func(s wgpu.BufferMapAsyncStatus) { st = s })
			ctx.device.Poll(true, nil)
			if st != wgpu.BufferMapAsyncStatusSuccess {
				t.Fatalf("map: %v", st)
			}
			got := make([]float32, n)
			copy(got, wgpu.FromBytes[float32](stag.GetMappedRange(0, uint(n*4))))
			stag.Unmap()

			cos, maxAbs := cosSim(want, got)
			// A stride bug leaves whole dim ranges at zero, which craters the cosine — it is not
			// a tolerance question. The bound is tight because this is f32 vs float64-accumulated
			// f32 on the same summation order.
			t.Logf("  hd=%d nH=%d nKV=%d keys=%d: cosine=%.9f maxAbs=%.3g", tc.hd, tc.nH, tc.nKV, tc.nKeys, cos, maxAbs)
			if cos < 0.999999 || maxAbs > 1e-4 {
				t.Errorf("hd=%d: cosine=%.9f maxAbs=%.3g", tc.hd, cos, maxAbs)
			}
			// Every dim must have been written. A lane that owns no dims, or a stride loop that
			// stops early, leaves an exact 0 here — invisible to a cosine over a mostly-correct
			// vector when hd is large.
			zeros := 0
			for _, v := range got {
				if v == 0 {
					zeros++
				}
			}
			if zeros > 0 {
				t.Errorf("hd=%d: %d/%d output elements are exactly 0 — dims went un-accumulated", tc.hd, zeros, n)
			}
		})
	}
}

// The wide kernel is built from a template, so the failure with no symptom is a placeholder that
// survives substitution or a variant whose fetch expression silently references the wrong buffer.
// The first is caught at build time by buildWideAttnWGSL; this asserts it, plus that all three
// variants produce distinct, complete sources.
func TestBuildWideAttnWGSL(t *testing.T) {
	seen := map[string]bool{}
	for _, v := range attnWideVariants {
		src, err := buildWideAttnWGSL(v)
		if err != nil {
			t.Fatalf("%s: %v", v.name, err)
		}
		if strings.Contains(src, "__") {
			t.Errorf("%s: unsubstituted placeholder survived", v.name)
		}
		for _, want := range []string{"@workgroup_size(256)", "array<f32, 256>", "array<f32, 8>", "var stride: u32 = 128u;"} {
			if !strings.Contains(src, want) {
				t.Errorf("%s: source lacks %q", v.name, want)
			}
		}
		if seen[src] {
			t.Errorf("%s: identical source to an earlier variant — the fetch expressions did not vary", v.name)
		}
		seen[src] = true
	}
	if _, err := buildWideAttnWGSL(attnWideVariant{name: "broken", kRead: "__NOPE__"}); err == nil {
		t.Error("a variant carrying an unknown placeholder must be refused, not compiled")
	}
}
