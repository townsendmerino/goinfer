//go:build cuda

package cuda

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"testing"
	"time"

	gpu "github.com/townsendmerino/aikit/gpu"
)

// S17 lever A's kernel gate (docs/tasks/task-multimodal-support-2026-10.md, "S17's CUDA lever A"): the fused float32 attention (tower_base.cu) against a float64 reference at the production shapes
// (head dim 72 at 4096 patches, 64 at 3136, 80, a ragged length, lengths below one tile, and two segments in one buffer that must not see each other), then each planted kernel defect required red.
// Needs a CUDA device; skips without one.

// attnRef is the float64 softmax(scale q k^T) v for the given query rows over the first T rows of k and v ([T, nH*hd], heads in place).
func attnRef(q, k, v []float32, T, nH, hd int, scale float64, rows []int) map[int][]float64 {
	stride := nH * hd
	out := map[int][]float64{}
	for _, i := range rows {
		o := make([]float64, stride)
		for h := range nH {
			sc := make([]float64, T)
			mx := math.Inf(-1)
			for j := range T {
				var d float64
				for x := range hd {
					d += float64(q[i*stride+h*hd+x]) * float64(k[j*stride+h*hd+x])
				}
				sc[j] = d * scale
				mx = math.Max(mx, sc[j])
			}
			var sum float64
			for j := range sc {
				sc[j] = math.Exp(sc[j] - mx)
				sum += sc[j]
			}
			for j := range T {
				w := sc[j] / sum
				for x := range hd {
					o[h*hd+x] += w * float64(v[j*stride+h*hd+x])
				}
			}
		}
		out[i] = o
	}
	return out
}

// runAttn runs ops.attention over [T, nH*hd] q/k/v (segments are the caller's: a segment is the same call on offset views) and returns the output.
func runAttn(t *testing.T, ops *towerOps, q, k, v []float32, T, nH, hd int, scale float32, segs []int) []float32 {
	t.Helper()
	out := make([]float32, len(q))
	err := ops.do(func() error {
		defer ops.releaseScratch()
		qb, kb, vb, ob := ops.af(len(q)), ops.af(len(q)), ops.af(len(q)), ops.af(len(q))
		for _, u := range []struct {
			b Buffer
			v []float32
		}{{qb, q}, {kb, k}, {vb, v}} {
			if e := gpu.Upload(u.b, u.v); e != nil {
				return e
			}
		}
		stride := nH * hd
		for si := 1; si < len(segs); si++ {
			off, n := segs[si-1]*stride*4, segs[si]-segs[si-1]
			if e := ops.attention(qb.At(off), kb.At(off), vb.At(off), ob.At(off), n, nH, hd, scale); e != nil {
				return e
			}
		}
		if e := ops.finish(); e != nil {
			return e
		}
		return gpu.Download(ob, out)
	})
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func attnData(seed int64, n int, amp float64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(amp * rng.NormFloat64())
	}
	return x
}

func attnMaxErr(got []float32, ref map[int][]float64, stride int) float64 {
	var worst float64
	for i, r := range ref {
		for x := range r {
			worst = math.Max(worst, math.Abs(float64(got[i*stride+x])-r[x]))
		}
	}
	return worst
}

func TestTowerAttn_matchesFloat64(t *testing.T) {
	ops := newTestTower(t, 64)
	cases := []struct {
		name      string
		T, nH, hd int
		amp       float64 // q and k scale: larger sharpens the softmax
		sample    int     // query rows checked (all when 0)
		segs      []int
		wantRows  func(T int) []int
	}{
		{"siglip 4096 x 16 x 72", 4096, 16, 72, 1.5, 24, nil, nil},
		{"qwen3.5/glm 3136 x 12 x 64", 3136, 12, 64, 1.5, 24, nil, nil},
		{"80 ragged 1000 x 16", 1000, 16, 80, 1.5, 24, nil, nil},
		{"ragged 4093 x 4 x 72, last rows", 4093, 4, 72, 2.5, 0, nil, func(T int) []int { return []int{0, 1, T - 66, T - 65, T - 64, T - 2, T - 1} }},
		{"one key", 1, 2, 64, 1, 0, nil, nil},
		{"7", 7, 2, 64, 1, 0, nil, nil},
		{"33", 33, 3, 72, 2, 0, nil, nil},
		{"65", 65, 2, 80, 2, 0, nil, nil},
		{"two segments 120+180", 300, 4, 72, 2, 0, []int{0, 120, 300}, nil},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			stride := c.nH * c.hd
			q, k, v := attnData(1, c.T*stride, c.amp), attnData(2, c.T*stride, c.amp), attnData(3, c.T*stride, 1)
			scale := 1 / math.Sqrt(float64(c.hd))
			segs := c.segs
			if segs == nil {
				segs = []int{0, c.T}
			}
			got := runAttn(t, ops, q, k, v, c.T, c.nH, c.hd, float32(scale), segs)
			var worst float64
			for si := 1; si < len(segs); si++ {
				lo, hi := segs[si-1], segs[si]
				n := hi - lo
				var rows []int
				switch {
				case c.wantRows != nil:
					rows = c.wantRows(n)
				case c.sample > 0:
					rng := rand.New(rand.NewSource(int64(si)))
					for len(rows) < c.sample {
						rows = append(rows, rng.Intn(n))
					}
				default:
					for i := range n {
						rows = append(rows, i)
					}
				}
				sq, sk, sv := q[lo*stride:hi*stride], k[lo*stride:hi*stride], v[lo*stride:hi*stride]
				ref := attnRef(sq, sk, sv, n, c.nH, c.hd, scale, rows)
				worst = math.Max(worst, attnMaxErr(got[lo*stride:hi*stride], ref, stride))
			}
			fmt.Fprintf(os.Stderr, "[S17 lever A] %-34s max |err| vs float64 %.2e\n", c.name, worst)
			if worst > 5e-5 {
				t.Errorf("max abs error %.3e over 5e-5", worst)
			}
		})
	}
}

// TestTowerAttn_plantedDefects: each kernel defect alone must push the error far past the bar on a shape that can see it. The running-max defect needs several key tiles and sharp scores
// (the max moves between tiles); the key-mask defect lets one zero-score, zero-value key in, which only matters when the softmax is flat and the segment short (33 keys: it takes ~3% of the weight).
func TestTowerAttn_plantedDefects(t *testing.T) {
	ops := newTestTower(t, 64)
	defer func() { towerAttnDefect = 0 }()
	for _, c := range []struct {
		d         int
		name      string
		T, nH, hd int
		amp       float64
	}{
		{1, "no rescale when the running max moves", 700, 4, 72, 2.5},
		{2, "key mask one past the segment", 33, 3, 72, 0.2},
		{3, "V tile read one key late", 700, 4, 72, 1.5},
	} {
		stride := c.nH * c.hd
		q, k, v := attnData(1, c.T*stride, c.amp), attnData(2, c.T*stride, c.amp), attnData(3, c.T*stride, 1)
		scale := 1 / math.Sqrt(float64(c.hd))
		var rows []int
		for i := 0; i < c.T; i += 1 + c.T/12 {
			rows = append(rows, i)
		}
		ref := attnRef(q, k, v, c.T, c.nH, c.hd, scale, rows)
		clean := attnMaxErr(runAttn(t, ops, q, k, v, c.T, c.nH, c.hd, float32(scale), []int{0, c.T}), ref, stride)
		towerAttnDefect = c.d
		got := runAttn(t, ops, q, k, v, c.T, c.nH, c.hd, float32(scale), []int{0, c.T})
		towerAttnDefect = 0
		e := attnMaxErr(got, ref, stride)
		fmt.Fprintf(os.Stderr, "[S17 lever A] planted (%d) %s: max |err| %.3e (unplanted %.1e)\n", c.d, c.name, e, clean)
		if clean > 5e-5 {
			t.Errorf("(%d) the unplanted kernel already misses the bar on this shape (%.3e)", c.d, clean)
		}
		if e < 1e-3 {
			t.Errorf("planted defect (%d) %s left the error at %.3e: the test cannot see it", c.d, c.name, e)
		}
	}
}

// TestTowerAttn_speed is the kernel's own exploratory rate against aikit's on SigLIP's shape (one layer's attention): TFLOPS = 4*np^2*hd*heads / time.
func TestTowerAttn_speed(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy: set GOINFER_HEAVY_TESTS=1")
	}
	ops := newTestTower(t, 64)
	const T, nH, hd = 4096, 16, 72
	stride := nH * hd
	q, k, v := attnData(1, T*stride, 1.5), attnData(2, T*stride, 1.5), attnData(3, T*stride, 1)
	var fused, aikit time.Duration
	for _, own := range []bool{true, false} {
		towerAttnAikit = !own
		runAttn(t, ops, q, k, v, T, nH, hd, 0.118, []int{0, T}) // warm
		t0 := time.Now()
		for range 3 {
			runAttn(t, ops, q, k, v, T, nH, hd, 0.118, []int{0, T})
		}
		if own {
			fused = time.Since(t0) / 3
		} else {
			aikit = time.Since(t0) / 3
		}
	}
	towerAttnAikit = false
	fl := 4 * float64(T) * float64(T) * hd * nH
	fmt.Fprintf(os.Stderr, "[S17 lever A] SigLIP layer attention: fused %.1f ms (%.2f TFLOPS), aikit %.1f ms (%.2f TFLOPS), %.1fx (includes the upload and download; exploratory)\n",
		fused.Seconds()*1e3, fl/fused.Seconds()/1e12, aikit.Seconds()*1e3, fl/aikit.Seconds()/1e12, aikit.Seconds()/fused.Seconds())
}
