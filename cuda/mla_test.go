//go:build cuda

package cuda

import (
	"context"
	"math"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
)

func mlaCosineMaxAbs(a, b []float32) (float64, float64) {
	if len(a) != len(b) || len(a) == 0 {
		return 0, 0
	}
	var dot, na, nb, maxAbs float64
	for i := range a {
		va, vb := float64(a[i]), float64(b[i])
		dot += va * vb
		na += va * va
		nb += vb * vb
		diff := math.Abs(va - vb)
		if diff > maxAbs {
			maxAbs = diff
		}
	}
	if na == 0 || nb == 0 {
		return 0, maxAbs
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb)), maxAbs
}

func randF32(n int, seed uint32) []float32 {
	out := make([]float32, n)
	s := seed
	for i := range out {
		s = s*1664525 + 1013904223
		out[i] = float32(int32(s>>8)%2000-1000) / 1000
	}
	return out
}

func TestMLALatentStore_cuda(t *testing.T) {
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	dev, err := gc.GetDevice(0)
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	cx, err := dev.Primary()
	if err != nil {
		t.Skipf("primary ctx: %v", err)
	}
	defer cx.Close()
	bg := context.Background()
	mod, err := cx.LoadModule(mlaPTX)
	if err != nil {
		t.Fatalf("LoadModule(mlaPTX): %v", err)
	}
	fn, err := mod.Function("mla_latent_store")
	if err != nil {
		t.Fatalf("fn(mla_latent_store): %v", err)
	}
	stream := mustStream(t, cx)

	const rank, qkRope, pos = 16, 8, 3
	const latDim = rank + qkRope
	const eps = float32(1e-6)

	kvDown := randF32(latDim, 101)
	normW := randF32(rank, 102)
	invFreq := make([]float32, qkRope/2)
	for d := range invFreq {
		invFreq[d] = float32(1.0 / math.Pow(1e4, float64(2*d)/float64(qkRope)))
	}

	for _, interleave := range []bool{false, true} {
		t.Run(map[bool]string{false: "neox", true: "gptj"}[interleave], func(t *testing.T) {
			ref := make([]float32, latDim)
			var ss float64
			for i := range rank {
				ss += float64(kvDown[i]) * float64(kvDown[i])
			}
			inv := float32(1.0 / math.Sqrt(ss/float64(rank)+float64(eps)))
			for i := range rank {
				ref[i] = kvDown[i] * inv * normW[i]
			}
			key := append([]float32(nil), kvDown[rank:]...)
			half := qkRope / 2
			if interleave {
				tmp := make([]float32, qkRope)
				for i := range half {
					tmp[i] = key[2*i]
					tmp[half+i] = key[2*i+1]
				}
				copy(key, tmp)
			}
			for d := range half {
				theta := float64(pos) * float64(invFreq[d])
				c, s := float32(math.Cos(theta)), float32(math.Sin(theta))
				x1, x2 := key[d], key[half+d]
				ref[rank+d] = x1*c - x2*s
				ref[rank+half+d] = x2*c + x1*s
			}

			dKV := mustAlloc[float32](t, cx, len(kvDown))
			dNW := mustAlloc[float32](t, cx, len(normW))
			dIF := mustAlloc[float32](t, cx, len(invFreq))
			dCache := mustAlloc[float32](t, cx, (pos+1)*latDim)
			defer dKV.Close()
			defer dNW.Close()
			defer dIF.Close()
			defer dCache.Close()

			_ = gc.CopyHtoD(bg, dKV, kvDown)
			_ = gc.CopyHtoD(bg, dNW, normW)
			_ = gc.CopyHtoD(bg, dIF, invFreq)

			base := pos * latDim
			cfg := gc.LaunchConfig{GridX: 1, GridY: 1, GridZ: 1, BlockX: 64, BlockY: 1, BlockZ: 1}
			var intl int32
			if interleave {
				intl = 1
			}
			if e := fn.LaunchOn(bg, stream, cfg,
				gc.Arg(dKV), gc.Arg(dNW), gc.Arg(dIF), gc.Arg(dCache),
				gc.ArgValue(int32(rank)), gc.ArgValue(int32(qkRope)), gc.ArgValue(int32(pos)),
				gc.ArgValue(eps), gc.ArgValue(int32(base)), gc.ArgValue(float32(1.0)), gc.ArgValue(intl)); e != nil {
				t.Fatalf("launch: %v", e)
			}
			if e := stream.Synchronize(bg); e != nil {
				t.Fatalf("sync: %v", e)
			}
			gotAll := make([]float32, (pos+1)*latDim)
			if e := gc.CopyDtoH(bg, gotAll, dCache); e != nil {
				t.Fatalf("D2H: %v", e)
			}
			got := gotAll[base : base+latDim]

			cos, maxAbs := mlaCosineMaxAbs(got, ref)
			t.Logf("interleave=%v: cosine=%.8f maxAbs=%.3e", interleave, cos, maxAbs)
			if cos < 0.9999 || maxAbs > 1e-4 {
				t.Errorf("mla_latent_store diverges: cosine=%.8f maxAbs=%.3e", cos, maxAbs)
			}
		})
	}
}

func TestMLAHeadMatvec_cuda(t *testing.T) {
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	dev, err := gc.GetDevice(0)
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	cx, err := dev.Primary()
	if err != nil {
		t.Skipf("primary ctx: %v", err)
	}
	defer cx.Close()
	bg := context.Background()
	mod, err := cx.LoadModule(mlaPTX)
	if err != nil {
		t.Fatalf("LoadModule(mlaPTX): %v", err)
	}
	fn, err := mod.Function("mla_head_matvec")
	if err != nil {
		t.Fatalf("fn(mla_head_matvec): %v", err)
	}
	stream := mustStream(t, cx)

	cases := []struct {
		name              string
		nH, N, K, aStride int
	}{
		{"absorb_WUK", 4, 16, 16, 24}, // deepseek-tiny shape: nH=4, rank=16, qkNope=16, qkHead=24
		{"lift_WUV", 4, 16, 16, 16},   // deepseek-tiny shape: nH=4, vHead=16, rank=16
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			a := randF32(tc.nH*tc.aStride, 201)
			w := randF32(tc.nH*tc.N*tc.K, 202)
			ref := make([]float32, tc.nH*tc.N)
			for h := range tc.nH {
				for n := range tc.N {
					var s float64
					for k := range tc.K {
						s += float64(a[h*tc.aStride+k]) * float64(w[(h*tc.N+n)*tc.K+k])
					}
					ref[h*tc.N+n] = float32(s)
				}
			}

			dA := mustAlloc[float32](t, cx, len(a))
			dW := mustAlloc[float32](t, cx, len(w))
			dDst := mustAlloc[float32](t, cx, tc.nH*tc.N)
			defer dA.Close()
			defer dW.Close()
			defer dDst.Close()

			_ = gc.CopyHtoD(bg, dA, a)
			_ = gc.CopyHtoD(bg, dW, w)

			elemCount := tc.nH * tc.N
			cfg := gc.LaunchConfig{GridX: uint32(elemCount), GridY: 1, GridZ: 1, BlockX: 64, BlockY: 1, BlockZ: 1}
			if e := fn.LaunchOn(bg, stream, cfg,
				gc.Arg(dA), gc.Arg(dW), gc.Arg(dDst),
				gc.ArgValue(int32(tc.nH)), gc.ArgValue(int32(tc.N)), gc.ArgValue(int32(tc.K)),
				gc.ArgValue(int32(tc.aStride)), gc.ArgValue(int32(tc.N))); e != nil {
				t.Fatalf("launch: %v", e)
			}
			if e := stream.Synchronize(bg); e != nil {
				t.Fatalf("sync: %v", e)
			}
			got := make([]float32, tc.nH*tc.N)
			if e := gc.CopyDtoH(bg, got, dDst); e != nil {
				t.Fatalf("D2H: %v", e)
			}

			cos, maxAbs := mlaCosineMaxAbs(got, ref)
			t.Logf("%s: cosine=%.8f maxAbs=%.3e", tc.name, cos, maxAbs)
			if cos < 0.9999 || maxAbs > 1e-4 {
				t.Errorf("mla_head_matvec diverges: cosine=%.8f maxAbs=%.3e", cos, maxAbs)
			}
		})
	}
}

func TestMLAAttn_cuda(t *testing.T) {
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	dev, err := gc.GetDevice(0)
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	cx, err := dev.Primary()
	if err != nil {
		t.Skipf("primary ctx: %v", err)
	}
	defer cx.Close()
	bg := context.Background()
	mod, err := cx.LoadModule(mlaPTX)
	if err != nil {
		t.Fatalf("LoadModule(mlaPTX): %v", err)
	}
	fn, err := mod.Function("mla_attn")
	if err != nil {
		t.Fatalf("fn(mla_attn): %v", err)
	}
	stream := mustStream(t, cx)

	const nH, rank, qkRope, nKeys = 4, 16, 8, 37
	const latDim = rank + qkRope
	scale := float32(1.0 / math.Sqrt(float64(rank+qkRope)))

	qAbs := randF32(nH*latDim, 301)
	lat := randF32(nKeys*latDim, 302)

	// CPU reference: exact two-pass softmax over latDim dot, value accumulate over rank prefix
	ref := make([]float32, nH*rank)
	for h := range nH {
		sc := make([]float64, nKeys)
		mx := math.Inf(-1)
		for j := range nKeys {
			var dot float64
			for d := range latDim {
				dot += float64(qAbs[h*latDim+d]) * float64(lat[j*latDim+d])
			}
			sc[j] = dot * float64(scale)
			if sc[j] > mx {
				mx = sc[j]
			}
		}
		var sum float64
		for j := range sc {
			sc[j] = math.Exp(sc[j] - mx)
			sum += sc[j]
		}
		for j := range nKeys {
			w := sc[j] / sum
			for c := range rank {
				ref[h*rank+c] += float32(w * float64(lat[j*latDim+c]))
			}
		}
	}

	dQ := mustAlloc[float32](t, cx, len(qAbs))
	dLat := mustAlloc[float32](t, cx, len(lat))
	dWsum := mustAlloc[float32](t, cx, nH*rank)
	defer dQ.Close()
	defer dLat.Close()
	defer dWsum.Close()

	_ = gc.CopyHtoD(bg, dQ, qAbs)
	_ = gc.CopyHtoD(bg, dLat, lat)

	shmem := uint32((nKeys + 128) * 4)
	cfg := gc.LaunchConfig{GridX: uint32(nH), GridY: 1, GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: shmem}
	if e := fn.LaunchOn(bg, stream, cfg,
		gc.Arg(dQ), gc.Arg(dLat), gc.Arg(dWsum),
		gc.ArgValue(int32(nH)), gc.ArgValue(int32(latDim)), gc.ArgValue(int32(rank)),
		gc.ArgValue(int32(nKeys-1)), gc.ArgValue(scale), gc.ArgValue(int32(0)), gc.ArgValue(int32(1))); e != nil {
		t.Fatalf("launch: %v", e)
	}
	if e := stream.Synchronize(bg); e != nil {
		t.Fatalf("sync: %v", e)
	}
	got := make([]float32, nH*rank)
	if e := gc.CopyDtoH(bg, got, dWsum); e != nil {
		t.Fatalf("D2H: %v", e)
	}

	cos, maxAbs := mlaCosineMaxAbs(got, ref)
	t.Logf("mla_attn: cosine=%.8f maxAbs=%.3e", cos, maxAbs)
	if cos < 0.99999 || maxAbs > 1e-4 {
		t.Errorf("mla_attn diverges: cosine=%.8f maxAbs=%.3e", cos, maxAbs)
	}
}
