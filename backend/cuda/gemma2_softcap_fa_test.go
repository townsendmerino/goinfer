//go:build cuda && goinfer_testhooks

package cuda

import (
	"math"
	"math/rand"
	"path/filepath"
	"testing"

	gpu "github.com/townsendmerino/goinfer/backend/cuda/internal/gpudev"
	"github.com/townsendmerino/goinfer/decoder"
)

// flashRefCap is flashRef with Gemma 2's attention softcap on every score: cap·tanh(scale·q·k / cap) before the softmax, in f64.
func flashRefCap(q, k, v []float32, nH, nKV, hd, winStart, nKeys int, scale, cap float32) []float64 {
	G, kvDim := nH/nKV, nKV*hd
	out := make([]float64, nH*hd)
	sc := make([]float64, nKeys-winStart)
	for h := range nH {
		kvh := h / G
		mx := math.Inf(-1)
		for s := winStart; s < nKeys; s++ {
			var d float64
			for i := range hd {
				d += float64(q[h*hd+i]) * float64(k[s*kvDim+kvh*hd+i])
			}
			d = float64(cap) * math.Tanh(d*float64(scale)/float64(cap))
			sc[s-winStart] = d
			mx = math.Max(mx, d)
		}
		var den float64
		for i := range sc {
			sc[i] = math.Exp(sc[i] - mx)
			den += sc[i]
		}
		for s := winStart; s < nKeys; s++ {
			w := sc[s-winStart] / den
			for i := range hd {
				out[h*hd+i] += w * float64(v[s*kvDim+kvh*hd+i])
			}
		}
	}
	return out
}

// TestGemma2Softcap_flashDecodeVsF64 is OP1-o2's correctness gate (docs/completed/task-forward-architecture-2026-10.md §4.6a, "Owed,
// not in OP1"): a capped model's resident loads the flash-decode lane's capped variant (decode_fa_cap), and that lane, driven on
// synthetic K/V at head dimensions 64, 128 and 256 (the lane's), matches an f64 oracle that caps every score, within the uncapped lane's
// bound (TestFlashDecodeVsF64: 1e-4 of the largest output), at several key counts, window starts and split counts, deterministically.
// Scores are drawn large enough that the fixture's cap (5) bends most of them, and the control requires it: on most cases the uncapped
// oracle is far from the capped one, and the lane with its cap lifted to 1e30 is far from the capped oracle. The multi-row verify lane,
// loaded from the same module, equals the M=1 lane at every row bit for bit.
func TestGemma2Softcap_flashDecodeVsF64(t *testing.T) {
	requireCUDADevice(t)
	t.Setenv("GOINFER_CUDA_FLASH_DECODE", "16")
	m, err := decoder.Load(filepath.Join("../../testdata/gemma2-tiny"), decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	rf, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatalf("not resident: %s", m.ResidentDecline())
	}
	capV := rf.attnCap
	if capV == 0 || rf.faModule != "decode_fa_cap" || rf.faCombine == (Pipeline{}) || rf.faCombineRows == (Pipeline{}) {
		t.Fatalf("cap %v, lane module %q, combine loaded %v, rows loaded %v: the capped lane did not load", capV, rf.faModule,
			rf.faCombine != (Pipeline{}), rf.faCombineRows != (Pipeline{}))
	}
	const nKV = 2
	nH := rf.nH
	if nH%nKV != 0 {
		t.Fatalf("nH %d", nH)
	}
	// The lane's partial buffer is sized at load for the fixture's head dimension (32); a model the lane serves sizes it for its own. Size
	// it for the largest geometry driven here, every row of a verify batch at 16 splits, as such a load would.
	if e := rf.do(func() error { rf.faBuf = rf.tf(faMaxRows * nH * 16 * faWarps * (256 + 4)); return nil }); e != nil {
		t.Fatal(e)
	}
	rng := rand.New(rand.NewSource(5))
	worst, cases, bent, lifted := 0.0, 0, 0, 0
	for _, hd := range []int{64, 128, 256} {
		kvDim := nKV * hd
		scale := float32(1 / math.Sqrt(float64(hd)))
		for _, nKeys := range []int{1, 9, 65, 300, 1000} {
			q, k, v := make([]float32, nH*hd), make([]float32, nKeys*kvDim), make([]float32, nKeys*kvDim)
			for i := range q {
				q[i] = 3 * float32(rng.NormFloat64()) // scores of a few times the cap
			}
			for i := range k {
				k[i], v[i] = float32(rng.NormFloat64()), float32(rng.NormFloat64())
			}
			for _, winStart := range []int{0, nKeys / 3} {
				ref := flashRefCap(q, k, v, nH, nKV, hd, winStart, nKeys, scale, capV)
				plain := flashRef(q, k, v, nH, nKV, hd, winStart, nKeys, scale)
				var refMax, capGap float64
				for i, x := range ref {
					refMax = math.Max(refMax, math.Abs(x))
					capGap = math.Max(capGap, math.Abs(x-plain[i]))
				}
				refMax = math.Max(refMax, 1e-12)
				rel := func(got []float32) (r float64) {
					for i := range got {
						if math.IsNaN(float64(got[i])) {
							t.Fatalf("NaN: hd=%d nKeys=%d win=%d elem %d", hd, nKeys, winStart, i)
						}
						r = math.Max(r, math.Abs(float64(got[i])-ref[i]))
					}
					return r / refMax
				}
				for _, S := range []int{1, 4, 16} {
					rf.faSplit = S
					got := runFlash(t, rf, q, k, v, hd, nKV, winStart, nKeys, scale)
					again := runFlash(t, rf, q, k, v, hd, nKV, winStart, nKeys, scale)
					for i := range got {
						if got[i] != again[i] {
							t.Fatalf("NOT DETERMINISTIC: hd=%d nKeys=%d win=%d S=%d elem %d", hd, nKeys, winStart, S, i)
						}
					}
					e := rel(got)
					worst = math.Max(worst, e)
					cases++
					if e > 1e-4 {
						t.Errorf("hd=%d nKeys=%d win=%d S=%d: %.3g of max|ref| %.3g from the capped f64 oracle (bound 1e-4)", hd, nKeys, winStart, S, e, refMax)
					}
					if capGap/refMax > 1e-2 { // the cap moves this case's output: the lane must follow it
						bent++
						rf.attnCap = 1e30
						off := runFlash(t, rf, q, k, v, hd, nKV, winStart, nKeys, scale)
						rf.attnCap = capV
						if rel(off) > 1e-3 {
							lifted++
						}
					}
				}
			}
		}
	}
	t.Logf("%d cases, worst %.3g of max|ref| from the capped oracle; %d where the cap moves the output by over 1%%, %d of them red with the cap lifted",
		cases, worst, bent, lifted)
	if bent < cases/2 || lifted != bent {
		t.Fatalf("control: the cap moved %d of %d cases and lifting it reddened %d: the test cannot see the cap", bent, cases, lifted)
	}

	// The verify lane: each row of a multi-row launch equals the M=1 lane at its position, bit for bit, under the cap.
	const hd, maxKeys = 128, 700
	kvDim, qDim := nKV*hd, nH*hd
	scale := float32(1 / math.Sqrt(float64(hd)))
	q, k, v := make([]float32, faMaxRows*qDim), make([]float32, maxKeys*kvDim), make([]float32, maxKeys*kvDim)
	for i := range q {
		q[i] = 3 * float32(rng.NormFloat64())
	}
	for i := range k {
		k[i], v[i] = float32(rng.NormFloat64()), float32(rng.NormFloat64())
	}
	var qb, kb, vb, cb, kt Buffer
	if e := rf.do(func() error {
		qb, kb, vb, cb = rf.tf(len(q)), rf.tf(len(k)), rf.tf(len(v)), rf.tf(faMaxRows*qDim)
		kt = rf.kvIdentityTab(t, maxKeys)
		for _, u := range []struct {
			b Buffer
			x []float32
		}{{qb, q}, {kb, k}, {vb, v}} {
			if e := gpu.Upload(u.b, u.x); e != nil {
				return e
			}
		}
		return nil
	}); e != nil {
		t.Fatalf("setup: %v", e)
	}
	saved := rf.attnScale
	rf.attnScale = scale
	defer func() { rf.attnScale = saved }()
	rows := 0
	for _, S := range []int{4, 16} {
		rf.faSplit = S
		for _, n0 := range []int{1, 9, 300, 650} {
			for _, M := range []int{1, 3, 9} {
				full := make([]float32, faMaxRows*qDim)
				if e := rf.do(func() error {
					for _, run := range faRowRuns(n0, M, 0, S) {
						if e := rf.flashDecodeRowsLaunch(qb, kb, vb, kt, cb, hd, nKV, run); e != nil {
							return e
						}
					}
					if e := rf.stream.Sync(); e != nil {
						return e
					}
					return gpu.Download(cb, full)
				}); e != nil {
					t.Fatalf("multi-row launch: %v", e)
				}
				for i := range M {
					one := runFlash(t, rf, q[i*qDim:(i+1)*qDim], k, v, hd, nKV, 0, n0+i, scale)
					for j := range one {
						if math.Float32bits(one[j]) != math.Float32bits(full[i*qDim+j]) {
							t.Fatalf("S=%d n0=%d M=%d row %d elem %d: multi-row %v, M=1 %v", S, n0, M, i, j, full[i*qDim+j], one[j])
						}
					}
					rows++
				}
			}
		}
	}
	t.Logf("%d verify rows equal the M=1 capped lane bit for bit", rows)
}
