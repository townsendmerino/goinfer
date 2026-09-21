//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"os"
	"testing"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/goinfer/decoder"
)

// flashRef is the float64 oracle for one decode-attention row: softmax over scale*q.k on [winStart, nKeys),
// then the probability-weighted sum of V, per query head, GQA-mapped. f64 accumulation throughout.
func flashRef(q, k, v []float32, nH, nKV, hd, winStart, nKeys int, scale float32) []float64 {
	G := nH / nKV
	kvDim := nKV * hd
	out := make([]float64, nH*hd)
	sc := make([]float64, nKeys-winStart)
	for h := 0; h < nH; h++ {
		kvh := h / G
		mx := math.Inf(-1)
		for s := winStart; s < nKeys; s++ {
			var d float64
			for i := 0; i < hd; i++ {
				d += float64(q[h*hd+i]) * float64(k[s*kvDim+kvh*hd+i])
			}
			d *= float64(scale)
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
			for i := 0; i < hd; i++ {
				out[h*hd+i] += w * float64(v[s*kvDim+kvh*hd+i])
			}
		}
	}
	return out
}

// TestFlashDecodeVsF64 checks fa_partial/fa_combine against the f64 oracle on synthetic K/V at every supported head
// dim (0.5B hd=64, 1.5B hd=128, gemma3-1b hd=256), several key counts around the 8-key block and the split
// boundaries, sliding-window starts, and S in {1,2,4,16}; and that two runs are bit-identical (determinism).
//
//	GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks' ./cuda/ -run TestFlashDecodeVsF64 -v
func TestFlashDecodeVsF64(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads real checkpoints for their geometry)")
	}
	t.Setenv("GOINFER_CUDA_FLASH_DECODE", "16")
	for _, mc := range []struct{ name, file string }{
		{"hd64", "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"},
		{"hd128", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"},
		{"hd256", "gemma3-1b-q4_k_m.gguf"},
	} {
		t.Run(mc.name, func(t *testing.T) {
			path := modelPath(mc.file)
			if _, err := os.Stat(path); err != nil {
				t.Skipf("no fixture at %s", path)
			}
			m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			defer m.Close()
			rf, ok := m.ResidentForwardForTest().(*cudaResident)
			if !ok {
				t.Skip("resident path declined")
			}
			if rf.faSplit != 16 || rf.faCombine == (Pipeline{}) {
				t.Fatalf("flash-decode lane not loaded (faSplit=%d)", rf.faSplit)
			}
			hd, nKV, nH := rf.layers[0].hd, rf.layers[0].nKV, rf.nH
			kvDim := nKV * hd
			rng := rand.New(rand.NewSource(1))
			worst := 0.0
			for _, nKeys := range []int{1, 7, 8, 9, 63, 64, 65, 300, 1000, 4097} {
				q := make([]float32, nH*hd)
				k := make([]float32, nKeys*kvDim)
				v := make([]float32, nKeys*kvDim)
				for i := range q {
					q[i] = float32(rng.NormFloat64())
				}
				for i := range k {
					k[i] = float32(rng.NormFloat64())
					v[i] = float32(rng.NormFloat64())
				}
				scale := float32(1 / math.Sqrt(float64(hd)))
				for _, winStart := range []int{0, nKeys / 3} {
					if nKeys-winStart < 1 {
						continue
					}
					ref := flashRef(q, k, v, nH, nKV, hd, winStart, nKeys, scale)
					var refMax float64
					for _, x := range ref {
						refMax = math.Max(refMax, math.Abs(x))
					}
					for _, S := range []int{1, 2, 4, 16} {
						rf.faSplit = S
						got := runFlash(t, rf, q, k, v, hd, nKV, winStart, nKeys, scale)
						again := runFlash(t, rf, q, k, v, hd, nKV, winStart, nKeys, scale)
						for i := range got {
							if got[i] != again[i] {
								t.Fatalf("NOT DETERMINISTIC: nKeys=%d win=%d S=%d elem %d: %v vs %v", nKeys, winStart, S, i, got[i], again[i])
							}
						}
						var maxErr float64
						for i := range got {
							if math.IsNaN(float64(got[i])) {
								t.Fatalf("NaN: nKeys=%d win=%d S=%d elem %d", nKeys, winStart, S, i)
							}
							maxErr = math.Max(maxErr, math.Abs(float64(got[i])-ref[i]))
						}
						rel := maxErr / math.Max(refMax, 1e-12)
						worst = math.Max(worst, rel)
						if rel > 1e-4 {
							t.Errorf("nKeys=%d win=%d S=%d: max |err| %.3g = %.3g of max|ref| %.3g (bound 1e-4)", nKeys, winStart, S, maxErr, rel, refMax)
						}
					}
				}
			}
			t.Logf("%s (nH=%d nKV=%d hd=%d): worst relative error vs f64 = %.3g", mc.name, nH, nKV, hd, worst)
		})
	}
}

func runFlash(t *testing.T, r *cudaResident, q, k, v []float32, hd, nKV, winStart, nKeys int, scale float32) []float32 {
	t.Helper()
	out := make([]float32, r.nH*hd)
	err := r.do(func() error {
		qb, kb, vb, cb := r.af(len(q)), r.af(len(k)), r.af(len(v)), r.af(len(out))
		if e := gpu.Upload(qb, q); e != nil {
			return e
		}
		if e := gpu.Upload(kb, k); e != nil {
			return e
		}
		if e := gpu.Upload(vb, v); e != nil {
			return e
		}
		saved := r.attnScale
		r.attnScale = scale
		e := r.flashDecodeLaunch(qb, kb, vb, cb, hd, nKV, winStart, nKeys)
		r.attnScale = saved
		if e != nil {
			return e
		}
		if e := r.stream.Sync(); e != nil {
			return e
		}
		return gpu.Download(cb, out)
	})
	if err != nil {
		t.Fatalf("flash decode launch: %v", fmt.Errorf("%w", err))
	}
	return out
}

// TestFlashDecodeEndToEnd wires the lane through the real decode path: the same teacher-forced 2048-deep decode
// on the real 1.5B with the lane off (exact split-KV) and on, comparing every step's logits. It asserts the lane
// ACTUALLY ran (launch count > 0 on the lane arm, 0 on the exact arm) so it cannot pass vacuously, and that the
// logits agree closely; it is a wiring check (q layout, GQA mapping, KV addressing, window), NOT the fidelity gate.
func TestFlashDecodeEndToEnd(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a 1.5B model)")
	}
	t.Setenv("GOINFER_CUDA_FLASH_DECODE", "4")
	t.Setenv("GOINFER_CUDA_FLASH_DECODE_MIN_KEYS", "0")
	t.Setenv("GOINFER_SPLITKV_VSUM_SPLIT", "4") // the CONTROL arm: a known, accepted reduction-order change on the same input
	path := modelPath("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}
	m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	rf, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Skip("resident path declined")
	}
	if rf.faSplit != 4 {
		t.Fatalf("lane not loaded (faSplit=%d)", rf.faSplit)
	}
	_, _, _, _, _, _, vocab := m.Dims()
	const D, steps = 2048, 24
	emb := func(i int) []float32 { return m.EmbedResidentForTest((i*2654435761 + 1) % (vocab - 1)) }
	prefill := make([][]float32, D)
	var s uint32 = 987654321
	for i := range prefill {
		s = s*1664525 + 1013904223
		prefill[i] = append([]float32(nil), emb(int(s>>8)%vocab)...)
	}
	rf.skMinKeys = 0
	var runArm func(split, vsum int) [][]float32
	run := func(split int) [][]float32 { return runArm(split, 0) }
	runArm = func(split, vsum int) [][]float32 {
		rf.faSplit = split
		rf.skVsumSplit = vsum
		rf.faLaunches = 0
		if _, e := rf.PrefillLast(context.Background(), prefill, 0); e != nil {
			t.Fatalf("prefill: %v", e)
		}
		out := make([][]float32, steps)
		for i := range out {
			l, e := rf.Forward(emb(i+7), D+i) // teacher-forced: the same token every step in both arms
			if e != nil {
				t.Fatalf("decode step %d (split=%d): %v", i, split, e)
			}
			out[i] = append([]float32(nil), l...)
		}
		return out
	}
	exact := run(0)
	if rf.faLaunches != 0 {
		t.Fatalf("exact arm launched the lane %d times", rf.faLaunches)
	}
	lane := run(4)
	laneLaunches := rf.faLaunches
	if laneLaunches == 0 {
		t.Fatal("VACUOUS: the lane arm never launched fa_partial")
	}
	cmp := func(x [][]float32) (worstCos float64, differing, agree int) {
		worstCos = 1.0
		for i := range exact {
			if !sameLogits(exact[i], x[i]) {
				differing++
			}
			var dot, na, nb float64
			for j := range exact[i] {
				dot += float64(exact[i][j]) * float64(x[i][j])
				na += float64(exact[i][j]) * float64(exact[i][j])
				nb += float64(x[i][j]) * float64(x[i][j])
			}
			worstCos = math.Min(worstCos, dot/math.Sqrt(na*nb))
			if argmaxF(exact[i]) == argmaxF(x[i]) {
				agree++
			}
		}
		return
	}
	spike := runArm(0, 4)
	worstCos, differing, agree := cmp(lane)
	spCos, spDiff, spAgree := cmp(spike)
	t.Logf("CONTROL (vsum spike S=4 vs exact): %d/%d steps differ; worst logit cosine %.7f; argmax agree %d/%d", spDiff, steps, spCos, spAgree, steps)
	t.Logf("lane launches=%d; %d/%d steps differ from exact; worst logit cosine %.7f; argmax agree %d/%d", laneLaunches, differing, steps, worstCos, agree, steps)
	if differing == 0 {
		t.Fatal("lane and exact logits are bit-identical at every step: the lane did not change the attention output")
	}
	// Wiring criterion, set from the control: the lane must sit within the same order of noise as an accepted
	// reduction-order change on this chaotic synthetic input, i.e. its cosine deficit (1-cos) is at most 5x the
	// spike's. A mis-wired q/GQA/KV mapping gives cosines near 0, not a small multiple.
	if 1-worstCos > 5*(1-spCos)+1e-6 {
		t.Errorf("lane worst cosine %.7f is far worse than the control's %.7f: the lane is mis-wired", worstCos, spCos)
	}
}
