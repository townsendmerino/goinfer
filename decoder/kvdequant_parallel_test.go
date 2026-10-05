package decoder

import (
	"math"
	"math/rand"
	"os"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// Audit R-15: the int8-KV widen is split by rows across goroutines. It is elementwise, so the split must change no bit; these tests say so on random shapes and on whole
// decodes.

func TestDequantKVRows_bitIdenticalToTheSerialWiden(t *testing.T) {
	origMin, origPar := kvDequantMinFloats, kvDequantParallel
	t.Cleanup(func() { kvDequantMinFloats, kvDequantParallel = origMin, origPar })
	rng := rand.New(rand.NewSource(15))
	forks := 0
	for _, tc := range []struct{ rows, cols, minFloats int }{
		{1, 128, 1}, {2, 128, 1}, {7, 64, 1}, {64, 128, 1}, {1000, 128, 1}, {1001, 96, 1}, {4096, 128, 1},
		{600, 128, 1 << 17}, // above the real threshold, nothing forced
		{50, 128, 1 << 17},  // below it: must stay serial and still be right
	} {
		kvDequantMinFloats, kvDequantParallel = tc.minFloats, true
		n := tc.rows * tc.cols
		kq, vq := make([]int8, n), make([]int8, n)
		for i := range kq {
			kq[i], vq[i] = int8(rng.Intn(256)-128), int8(rng.Intn(256)-128)
		}
		ksc, vsc := make([]float32, tc.rows), make([]float32, tc.rows)
		for i := range ksc {
			ksc[i], vsc[i] = float32(rng.NormFloat64())*0.05, float32(rng.NormFloat64())*0.05
		}
		wantK, wantV := make([]float32, n), make([]float32, n)
		linalg.DequantizeRowsInt8Into(wantK, kq, ksc, tc.rows, tc.cols)
		linalg.DequantizeRowsInt8Into(wantV, vq, vsc, tc.rows, tc.cols)
		gotK, gotV := make([]float32, n), make([]float32, n)
		for i := range gotK { // poison: every element must be written
			gotK[i], gotV[i] = float32(math.NaN()), float32(math.NaN())
		}
		before := kvDequantFanouts.Load()
		dequantKVRows(gotK, gotV, kq, vq, ksc, vsc, tc.rows, tc.cols)
		forks += int(kvDequantFanouts.Load() - before)
		for i := range wantK {
			if math.Float32bits(gotK[i]) != math.Float32bits(wantK[i]) || math.Float32bits(gotV[i]) != math.Float32bits(wantV[i]) {
				t.Fatalf("rows=%d cols=%d: element %d differs from the serial widen", tc.rows, tc.cols, i)
			}
		}
	}
	if forks == 0 {
		t.Fatal("no case actually forked (is GOMAXPROCS 1?): the fan-out was never exercised, so this proved nothing")
	}
}

// TestKVI8_parallelWidenBitIdenticalOnWholeDecodes decodes real tiny checkpoints with int8 KV, once with the widen forced serial and once with the fan-out forced on
// even at these sizes (threshold 1), and requires every logit of every step to match. cohere2-tiny and gemma2-tiny put int8 RINGS on the same path; the others
// are global-only.
func TestKVI8_parallelWidenBitIdenticalOnWholeDecodes(t *testing.T) {
	origMin, origPar := kvDequantMinFloats, kvDequantParallel
	t.Cleanup(func() { kvDequantMinFloats, kvDequantParallel = origMin, origPar })
	for _, fx := range []string{"cohere2-tiny", "gemma2-tiny", "llama-tiny"} {
		t.Run(fx, func(t *testing.T) {
			dir := "../testdata/" + fx
			if _, err := os.Stat(dir); err != nil {
				t.Skipf("no fixture %s", dir)
			}
			m, err := Load(dir, Options{Backend: "cpu", KVQuant: "i8"})
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			defer m.Close()
			const N = 40
			toks := make([]int, N)
			rng := rand.New(rand.NewSource(8))
			for i := range toks {
				toks[i] = 1 + rng.Intn(m.w.arch.VocabSize-1)
			}
			run := func(par bool) [][]float32 {
				kvDequantParallel = par
				kvDequantMinFloats = 1
				cache := m.NewCache(N + 2)
				if cache.quant != kvI8 {
					t.Fatalf("the cache is not int8: this test would not touch the widen")
				}
				var all [][]float32
				for i, tok := range toks {
					lg, err := m.forward(tok, cache)
					if err != nil {
						t.Fatalf("par=%v step %d: %v", par, i, err)
					}
					all = append(all, append([]float32(nil), lg...))
				}
				return all
			}
			ref := run(false)
			before := kvDequantFanouts.Load()
			got := run(true)
			if kvDequantFanouts.Load() == before {
				t.Fatal("the parallel run never forked: nothing was proved")
			}
			for i := range ref {
				for j := range ref[i] {
					if math.Float32bits(ref[i][j]) != math.Float32bits(got[i][j]) {
						t.Fatalf("step %d logit %d: parallel %v != serial %v", i, j, got[i][j], ref[i][j])
					}
				}
			}
		})
	}
}
