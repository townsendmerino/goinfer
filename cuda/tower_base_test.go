//go:build cuda

package cuda

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"

	gpu "github.com/townsendmerino/aikit/gpu"
)

// The CUDA vision-tower base against plain Go (S4 step 1, docs/tasks/task-multimodal-support-2026-10.md): every GEMM branch the towers reach (the register
// kernel on aligned shapes, the bias kernel at any M, the tiled kernel plus add_bias at a K that is not a multiple of 16: real GLM-OCR's patch embed is K = 1176),
// the norms, and the tower_base.cu kernels. The tiny fixtures never reach these production branches, which is aikit visioncuda's failure class (tiny passes, real is
// wrong), so each branch is checked here on its own. Needs a CUDA device; skips without one.

func newTestTower(t *testing.T, maxN int) *towerOps {
	t.Helper()
	ops, err := newTowerOps(maxN)
	if err != nil {
		t.Skipf("no CUDA device for the tower base: %v", err)
	}
	t.Cleanup(ops.close)
	return ops
}

func tbRand(rng *rand.Rand, n int, scale float64) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(rng.NormFloat64() * scale)
	}
	return v
}

// run uploads nothing itself: it executes body on the executor, syncs, and fails the test on any error.
func (ops *towerOps) mustRun(t *testing.T, body func()) {
	t.Helper()
	err := ops.do(func() error {
		defer ops.releaseScratch()
		body()
		return ops.finish()
	})
	if err != nil {
		t.Fatal(err)
	}
}

func (ops *towerOps) download(t *testing.T, b Buffer, n int) []float32 {
	t.Helper()
	out := make([]float32, n)
	if err := ops.do(func() error { return gpu.Download(b, out) }); err != nil {
		t.Fatal(err)
	}
	return out
}

// alloc is a persistent float32 buffer of n elements (not released by mustRun), for outputs the test downloads afterwards.
func (ops *towerOps) alloc(t *testing.T, n int) Buffer {
	t.Helper()
	var b Buffer
	if err := ops.do(func() error { b = gpu.NewBufferLenOf[float32](ops.dev, n); return nil }); err != nil {
		t.Fatal(err)
	}
	return b
}

func (ops *towerOps) upload(t *testing.T, x []float32) Buffer {
	t.Helper()
	var b Buffer
	if err := ops.do(func() error { b = gpu.NewBufferOf(ops.dev, x); return nil }); err != nil {
		t.Fatal(err)
	}
	return b
}

// refGEMM is out[M,N] = a[M,K] . w[N,K]^T (+ bias[N]) in float64.
func refGEMM(a, w, bias []float32, M, N, K int) []float64 {
	out := make([]float64, M*N)
	for m := range M {
		for n := range N {
			s := 0.0
			for k := range K {
				s += float64(a[m*K+k]) * float64(w[n*K+k])
			}
			if bias != nil {
				s += float64(bias[n])
			}
			out[m*N+n] = s
		}
	}
	return out
}

func maxRel(t *testing.T, what string, got []float32, want []float64, tol float64) {
	t.Helper()
	worst, at := 0.0, 0
	scale := 0.0
	for _, w := range want {
		scale = math.Max(scale, math.Abs(w))
	}
	for i := range want {
		if d := math.Abs(float64(got[i])-want[i]) / (scale + 1e-12); d > worst {
			worst, at = d, i
		}
	}
	if worst > tol {
		t.Errorf("%s: worst error %.3g of the output's scale at element %d (got %v want %v), over %g", what, worst, at, got[at], want[at], tol)
	}
}

func TestTowerBase_gemmBranches(t *testing.T) {
	ops := newTestTower(t, 256)
	rng := rand.New(rand.NewSource(1))
	// {M, N, K}: aligned (register kernel), unaligned M and N at K%16==0 (bias kernel), K%16 != 0 (tiled kernel + add_bias; 1176 is GLM-OCR's patch dim), tiny.
	for _, sh := range [][3]int{{128, 128, 64}, {100, 70, 48}, {37, 50, 1176}, {5, 9, 20}, {2520, 192, 768}} {
		M, N, K := sh[0], sh[1], sh[2]
		a, w, bias := tbRand(rng, M*K, 1), tbRand(rng, N*K, 1/math.Sqrt(float64(K))), tbRand(rng, N, 1)
		resid0 := tbRand(rng, M*N, 1)
		ab, wb, bb := ops.upload(t, a), ops.upload(t, w), ops.upload(t, bias)
		var o1, o2, o3 Buffer
		ops.mustRun(t, func() {
			o1, o2 = ops.af(M*N), ops.af(M*N)
			o3 = ops.af(M * N)
			tmp := ops.af(M * N)
			if err := gpu.Upload(o3, resid0); err != nil {
				panic(err)
			}
			ops.gemm(ab, wb, o1, M, N, K)
			ops.gemmBias(ab, wb, bb, o2, M, N, K)
			ops.gemmBiasAdd(ab, wb, bb, o3, tmp, M, N, K)
			if err := ops.q.Sync(); err != nil {
				panic(err)
			}
			g1, g2, g3 := make([]float32, M*N), make([]float32, M*N), make([]float32, M*N)
			for _, p := range []struct {
				b Buffer
				v []float32
			}{{o1, g1}, {o2, g2}, {o3, g3}} {
				if err := gpu.Download(p.b, p.v); err != nil {
					panic(err)
				}
			}
			want := refGEMM(a, w, nil, M, N, K)
			wantB := refGEMM(a, w, bias, M, N, K)
			wantR := make([]float64, M*N)
			for i := range wantR {
				wantR[i] = wantB[i] + float64(resid0[i])
			}
			maxRel(t, "gemm", g1, want, 2e-5)
			maxRel(t, "gemmBias", g2, wantB, 2e-5)
			maxRel(t, "gemmBiasAdd", g3, wantR, 2e-5)
		})
		t.Logf("M=%d N=%d K=%d: gemm, gemmBias and gemmBiasAdd match float64", M, N, K)
	}
}

func TestTowerBase_normsAndActivations(t *testing.T) {
	ops := newTestTower(t, 8)
	rng := rand.New(rand.NewSource(2))
	rows, dim := 13, 64
	x, w, b := tbRand(rng, rows*dim, 2), tbRand(rng, dim, 1), tbRand(rng, dim, 1)
	xb, wb, bb := ops.upload(t, x), ops.upload(t, w), ops.upload(t, b)
	rmsO, lnO := ops.alloc(t, rows*dim), ops.alloc(t, rows*dim)
	ops.mustRun(t, func() {
		ops.rms(xb, wb, rmsO, rows, dim, 1e-6)
		ops.layerNorm(xb, wb, bb, lnO, rows, dim, 1e-6)
	})
	gotRMS, gotLN := ops.download(t, rmsO, rows*dim), ops.download(t, lnO, rows*dim)
	wantRMS, wantLN := make([]float64, rows*dim), make([]float64, rows*dim)
	for r := range rows {
		ss, mean := 0.0, 0.0
		for d := range dim {
			v := float64(x[r*dim+d])
			ss += v * v
			mean += v
		}
		inv := 1 / math.Sqrt(ss/float64(dim)+1e-6)
		mean /= float64(dim)
		vr := 0.0
		for d := range dim {
			dv := float64(x[r*dim+d]) - mean
			vr += dv * dv
		}
		invLN := 1 / math.Sqrt(vr/float64(dim)+1e-6)
		for d := range dim {
			wantRMS[r*dim+d] = float64(x[r*dim+d]) * inv * float64(w[d])
			wantLN[r*dim+d] = (float64(x[r*dim+d])-mean)*invLN*float64(w[d]) + float64(b[d])
		}
	}
	maxRel(t, "rms", gotRMS, wantRMS, 1e-5)
	maxRel(t, "layerNorm", gotLN, wantLN, 1e-5)
}

// The tower_base.cu kernels are compared bit for bit with Go's float32 arithmetic (Go does not fuse on amd64, and the kernels use explicit intrinsics).
func TestTowerBase_kernels(t *testing.T) {
	ops := newTestTower(t, 8)
	rng := rand.New(rand.NewSource(3))
	const rows, H = 11, 24
	n := rows * H

	t.Run("clamp", func(t *testing.T) {
		x := tbRand(rng, n, 3)
		lo, hi := float32(-1.5), float32(2)
		inf := float32(math.Inf(1))
		xb := ops.upload(t, x)
		cp, ci, cinf := ops.alloc(t, n), ops.alloc(t, n), ops.alloc(t, n)
		ops.mustRun(t, func() {
			ops.clampCopyTo(xb, cp, n, lo, hi)
			ops.launch(ops.clampCopy, gpu.Grid1D(n, 256), Arg(xb), Arg(ci), i64(int64(n)), f32v(-inf), f32v(inf)) // infinite bounds: a copy
			ops.clampCopyTo(xb, cinf, n, lo, inf)
		})
		got, gotI, gotHalf := ops.download(t, cp, n), ops.download(t, ci, n), ops.download(t, cinf, n)
		for i, v := range x {
			w := v
			if w < lo {
				w = lo
			}
			if w > hi {
				w = hi
			}
			wh := v
			if wh < lo {
				wh = lo
			}
			if got[i] != w || gotI[i] != v || gotHalf[i] != wh {
				t.Fatalf("element %d: clamp %v (want %v), infinite bounds %v (want %v), one-sided %v (want %v)", i, got[i], w, gotI[i], v, gotHalf[i], wh)
			}
		}
		ops.mustRun(t, func() { ops.clampInPlace(xb, n, lo, hi) })
		inPlace := ops.download(t, xb, n)
		for i := range inPlace {
			if inPlace[i] != got[i] {
				t.Fatalf("in-place clamp differs from the copy at %d", i)
			}
		}
	})

	t.Run("posAdd", func(t *testing.T) {
		const table = 9
		x, X, Y := tbRand(rng, n, 1), tbRand(rng, table*H, 1), tbRand(rng, table*H, 1)
		pos := make([]int32, 2*rows)
		for i := range rows {
			pos[2*i], pos[2*i+1] = int32(rng.Intn(table)), int32(rng.Intn(table))
		}
		xb, Xb, Yb := ops.upload(t, x), ops.upload(t, X), ops.upload(t, Y)
		var pb Buffer
		if err := ops.do(func() error { pb = gpu.NewBufferOf(ops.dev, pos); return nil }); err != nil {
			t.Fatal(err)
		}
		ops.mustRun(t, func() { ops.posAddTo(xb, Xb, Yb, pb, rows, H) })
		got := ops.download(t, xb, n)
		for i := range rows {
			for d := range H {
				want := x[i*H+d] + (X[int(pos[2*i])*H+d] + Y[int(pos[2*i+1])*H+d])
				if got[i*H+d] != want {
					t.Fatalf("row %d dim %d: %v want %v", i, d, got[i*H+d], want)
				}
			}
		}
	})

	t.Run("ropeAxial", func(t *testing.T) {
		const heads, hd = 3, 16
		T := rows
		x := tbRand(rng, T*heads*hd, 1)
		cs, sn := tbRand(rng, T*hd, 1), tbRand(rng, T*hd, 1)
		xb, cb, sb := ops.upload(t, x), ops.upload(t, cs), ops.upload(t, sn)
		ops.mustRun(t, func() { ops.ropeAxialTo(xb, cb, sb, T, heads, hd) })
		got := ops.download(t, xb, len(x))
		want := append([]float32(nil), x...)
		half, q := hd/2, hd/4
		for tt := range T {
			for h := range heads {
				for p := range half {
					base, d := (p/q)*half, p%q
					r := want[(tt*heads+h)*hd+base:]
					c, s := cs[tt*hd+base:], sn[tt*hd+base:]
					a, b := r[d], r[d+q]
					r[d] = a*c[d] - b*s[d]
					r[d+q] = b*c[d+q] + a*s[d+q]
				}
			}
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("element %d: %v want %v", i, got[i], want[i])
			}
		}
	})

	t.Run("ropeHalf", func(t *testing.T) {
		const heads, hd = 3, 16
		T := rows
		x := tbRand(rng, T*heads*hd, 1)
		cs, sn := tbRand(rng, T*hd, 1), tbRand(rng, T*hd, 1)
		xb, cb, sb := ops.upload(t, x), ops.upload(t, cs), ops.upload(t, sn)
		ops.mustRun(t, func() { ops.ropeHalfTo(xb, cb, sb, T, heads, hd) })
		got := ops.download(t, xb, len(x))
		want := append([]float32(nil), x...)
		half := hd / 2
		for tt := range T {
			for h := range heads {
				r := want[(tt*heads+h)*hd:]
				c, s := cs[tt*hd:], sn[tt*hd:]
				for d := range half {
					a, b := r[d], r[d+half]
					r[d] = a*c[d] - b*s[d]
					r[d+half] = b*c[d+half] + a*s[d+half]
				}
			}
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("element %d: %v want %v", i, got[i], want[i])
			}
		}
	})

	t.Run("mulAndScale", func(t *testing.T) {
		x, u := tbRand(rng, n, 1), tbRand(rng, n, 1)
		xb, ub := ops.upload(t, x), ops.upload(t, u)
		x2 := ops.upload(t, x)
		ops.mustRun(t, func() { ops.mulVec(xb, ub, n); ops.scaleVec(x2, n, 0.125) })
		gm, gs := ops.download(t, xb, n), ops.download(t, x2, n)
		for i := range x {
			if gm[i] != x[i]*u[i] || gs[i] != x[i]*0.125 {
				t.Fatalf("element %d: mul %v (want %v), scale %v (want %v)", i, gm[i], x[i]*u[i], gs[i], x[i]*0.125)
			}
		}
	})
}

func TestTowerBase_attention(t *testing.T) {
	ops := newTestTower(t, 8)
	rng := rand.New(rand.NewSource(4))
	for _, c := range []struct{ np, nH, hd int }{{37, 2, 16}, {200, 3, 64}, {3200, 2, 64}} { // 3200 patches takes the query-tiled kernel
		hidden := c.nH * c.hd
		q, k, v := tbRand(rng, c.np*hidden, 1), tbRand(rng, c.np*hidden, 1), tbRand(rng, c.np*hidden, 1)
		qb, kb, vb := ops.upload(t, q), ops.upload(t, k), ops.upload(t, v)
		var ob Buffer
		for _, scale := range []float32{1, float32(1 / math.Sqrt(float64(c.hd)))} {
			ops.mustRun(t, func() {
				ob = ops.af(c.np * hidden)
				if err := ops.attention(qb, kb, vb, ob, c.np, c.nH, c.hd, scale); err != nil {
					panic(err)
				}
				if err := ops.q.Sync(); err != nil {
					panic(err)
				}
				got := make([]float32, c.np*hidden)
				if err := gpu.Download(ob, got); err != nil {
					panic(err)
				}
				want := make([]float64, c.np*hidden)
				for h := range c.nH {
					for i := range c.np {
						sc := make([]float64, c.np)
						mx := math.Inf(-1)
						for j := range c.np {
							s := 0.0
							for d := range c.hd {
								s += float64(q[i*hidden+h*c.hd+d]) * float64(k[j*hidden+h*c.hd+d])
							}
							sc[j] = s * float64(scale)
							mx = math.Max(mx, sc[j])
						}
						sum := 0.0
						for j := range sc {
							sc[j] = math.Exp(sc[j] - mx)
							sum += sc[j]
						}
						for d := range c.hd {
							o := 0.0
							for j := range c.np {
								o += sc[j] / sum * float64(v[j*hidden+h*c.hd+d])
							}
							want[i*hidden+h*c.hd+d] = o
						}
					}
				}
				maxRel(t, "attention", got, want, 2e-5)
			})
		}
	}
	// An image past the untiled kernel's shared-memory row with a head dim that has no tiled kernel is refused by name (hd 64 takes the tiled kernel instead).
	if err := ops.attention(Buffer{}, Buffer{}, Buffer{}, Buffer{}, 13000, 2, 200, 1); err == nil {
		t.Error("an attention over 13000 patches at head dim 200 must be refused")
	}
}

// TestTowerOps_contextVRAM measures what a tower base's own CUDA context and kernel modules cost before any weights (a
// tower's VRAM reserve must cover them too: a second context's worth was the gap at the first image). Free VRAM through
// the driver before and after newTowerOps, and after one tiny launch. A record; asserts only that it can read the figure.
func TestTowerOps_contextVRAM(t *testing.T) {
	free := func() float64 {
		b, ok := decoder.FreeBytesFor("cuda")
		if !ok {
			t.Skip("no free-VRAM reading")
		}
		return float64(b) / (1 << 20)
	}
	f0 := free()
	ops := newTestTower(t, 64)
	f1 := free()
	_ = ops
	fmt.Fprintf(os.Stderr, "[S7 CUDA tower base VRAM] free before %.0f MiB, after newTowerOps (context + ViT module + tower_base module) %.0f MiB: the base holds %.0f MiB before any weight\n", f0, f1, f0-f1)
}
