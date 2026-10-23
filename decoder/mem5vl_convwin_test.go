package decoder

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"os"
	"testing"
	"unsafe"
)

// TestConvWin_noSlabRetained is MEM5-VL-convwin's CW-G2 (docs/tasks/task-memory-campaign-2026-10.md, "MEM5-VL-convwin gates"):
// after a deltaNetCoreN pass, no row of the conv window shares the pass's mixed slab, so the slab is free once the pass returns.
func TestConvWin_noSlabRetained(t *testing.T) {
	m, err := Load("../testdata/qwen35-tiny", Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	arch := m.w.arch
	layer := -1
	for l := range arch.NumLayers {
		if arch.isLinearLayer(l) {
			layer = l
			break
		}
	}
	p := *arch.qwen35
	nv, hk, hv, nk := p.NumValueHeads, p.KeyHeadDim, p.ValueHeadDim, p.NumKeyHeads
	valueDim, convDim := hv*nv, 2*hk*nk+hv*nv
	rnd := rand.New(rand.NewSource(3))
	vec := func(n int) []float32 {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(rnd.NormFloat64())
		}
		return x
	}
	for _, K := range []int{2, p.ConvKernel - 1, 9} {
		mixed := vec(K * convDim)
		lo := uintptr(unsafe.Pointer(&mixed[0]))
		hi := lo + uintptr(4*len(mixed))
		st := newDeltaState(p)
		deltaNetCoreN(make([]float32, K*valueDim), mixed, vec(K*nv), vec(K*nv), vec(K*valueDim), K, m.w.Layers[layer].delta, p, arch.NormEps, st)
		if len(st.convWin) != min(K, p.ConvKernel-1) {
			t.Fatalf("K=%d: the window holds %d rows", K, len(st.convWin))
		}
		for i, row := range st.convWin {
			if a := uintptr(unsafe.Pointer(&row[0])); a >= lo && a < hi {
				t.Errorf("K=%d: window row %d is a slice of the pass's slab", K, i)
			}
		}
	}
}

// TestConvWin_qwen35BitIdentical is CW-G1's model half: on each Qwen3.5 fixture, a batched 40-token prefill
// (runLayersQwen35N) and 8 greedy decode steps after give the same logits with the window's rows copied as with the old slices.
func TestConvWin_qwen35BitIdentical(t *testing.T) {
	run := func(m *Model, slices bool) [][]float32 {
		prev := convWinSliceForTest
		convWinSliceForTest = slices
		defer func() { convWinSliceForTest = prev }()
		_, _, _, _, _, _, vocab := m.Dims()
		ids := make([]int, 40)
		for i := range ids {
			ids[i] = (i*37 + 11) % vocab
		}
		cache := m.NewCache(48)
		if !m.qwen35BatchN(len(ids), cache) {
			t.Fatal("the fixture does not take the batched Qwen3.5 pass")
		}
		if cache.scr == nil {
			cache.scr = newDecodeScratch(m.w.arch)
		}
		hn, err := m.runLayersQwen35N(context.Background(), m.embedN(ids), cache)
		if err != nil {
			t.Fatal(err)
		}
		hidden := m.w.arch.HiddenDim
		lg := append([]float32(nil), m.logitsFromNormed(hn[len(hn)-hidden:], cache)...)
		out := [][]float32{lg}
		for range 8 {
			best := 0
			for j := range lg {
				if lg[j] > lg[best] {
					best = j
				}
			}
			if lg, err = m.forward(best, cache); err != nil {
				t.Fatal(err)
			}
			out = append(out, append([]float32(nil), lg...))
		}
		return out
	}
	for _, dir := range []string{"../testdata/qwen35-tiny", "testdata/qwen3_5-tiny", "testdata/qwen3_5_moe-tiny", "../testdata/qwen35vl-tiny"} {
		if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
			t.Logf("%s: no fixture", dir)
			continue
		}
		m, err := Load(dir, Options{})
		if err != nil {
			t.Fatal(err)
		}
		want, got := run(m, true), run(m, false)
		d := 0
		for s := range want {
			for j := range want[s] {
				if math.Float32bits(want[s][j]) != math.Float32bits(got[s][j]) {
					d++
				}
			}
		}
		fmt.Fprintf(os.Stderr, "[CW-G1] %s: %d of %d logits differ (prefill and 8 decode steps)\n", dir, d, len(want)*len(want[0]))
		if d != 0 {
			t.Errorf("%s: %d logits differ between the copied and the sliced window", dir, d)
		}
		m.Close()
	}
}
