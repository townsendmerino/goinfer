//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"fmt"
	"math"
	"os"
	"strconv"
	"testing"
	"time"

	gc "github.com/eitamring/gocudrv/cuda"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestMoERows_realMellumIdentityAndAB is the tier-1 reading for the grouped expert kernels on the real Mellum2.1 (int4 .giw, --moe-cache-experts, the
// shipped plan). Two parts, in one load:
//
//  1. Identity on the real checkpoint: the f32 residual of every row of a 512-row prompt after the last layer, grouped kernels against the per-pair
//     launches, must be the same bits. Real weights, real routing, 28 layers, the cache evicting.
//  2. The A/B: a cold prefill of GOINFER_ROWS_AB_M tokens (default 4096), order off, on, on, off, wall time per run and the last row's logits, which
//     must be the same bits across arms. The first run of each arm is the warm one; the pairs are read as paired differences.
//
// Exploratory by-day reading: not a published number.
//
//	GOINFER_HEAVY_TESTS=1 go test -count=1 -timeout 30m -tags 'cuda goinfer_testhooks' -run '^TestMoERows_realMellumIdentityAndAB$' -v ./cuda/
func TestMoERows_realMellumIdentityAndAB(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads Mellum2.1)")
	}
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	if _, err := gc.GetDevice(0); err != nil {
		t.Skipf("no device: %v", err)
	}
	path := modelPath("mellum2.1-thinking.int4.e4h.cuda.v17.giw")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}
	M := 4096
	if v, err := strconv.Atoi(os.Getenv("GOINFER_ROWS_AB_M")); err == nil && v > 0 {
		M = v
	}
	t0 := time.Now()
	mc, err := decoder.Load(path, decoder.Options{Backend: "cuda", MoECacheExperts: true, ResidentContext: 8192})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer mc.Close()
	fmt.Fprintf(os.Stderr, "[rows-ab] loaded in %v\n", time.Since(t0).Round(time.Millisecond))
	r, ok := mc.ResidentForwardForTest().(*cudaResident)
	if !ok || !r.cacheExperts {
		t.Fatal("the C' resident did not engage")
	}
	if !r.prefillMoERowsEligible() {
		t.Fatal("the grouped kernels are not eligible on this build")
	}
	_, _, _, _, _, _, vocab := mc.Dims()
	build := func(n int, seed uint32) [][]float32 {
		embs := make([][]float32, n)
		s := seed
		for i := range embs {
			s = s*1664525 + 1013904223
			embs[i] = append([]float32(nil), mc.EmbedResidentForTest(int(s>>8)%(vocab-1))...)
		}
		return embs
	}
	ctx := context.Background()

	// 1. identity on every residual row
	embs := build(512, 777)
	var res [2][][]float32
	for i, grouped := range []bool{false, true} {
		r.moeRowsOff = !grouped
		before := cudaMoeRowsRuns
		w := time.Now()
		rows, err := r.ResidualAll(ctx, embs, 0)
		if err != nil {
			t.Fatalf("ResidualAll (grouped=%v): %v", grouped, err)
		}
		ran := cudaMoeRowsRuns - before
		if grouped && ran == 0 || !grouped && ran != 0 {
			t.Fatalf("grouped=%v ran %d layers: the arm is not the one named", grouped, ran)
		}
		fmt.Fprintf(os.Stderr, "[rows-ab] residual 512 rows grouped=%v: %v (%d grouped layer passes)\n", grouped, time.Since(w).Round(time.Millisecond), ran)
		res[i] = rows
	}
	if d, n := residualDiff(res[0], res[1]); d != 0 {
		t.Errorf("real Mellum2.1: grouped differs from per-pair on %d of %d residual elements of 512 rows", d, n)
	} else {
		t.Logf("real Mellum2.1: grouped == per-pair on all %d residual elements of 512 rows", n)
	}

	// 2. the A/B at M
	embs = build(M, 4096)
	type run struct {
		grouped bool
		wall    time.Duration
		logits  []float32
	}
	var runs []run
	for i, grouped := range []bool{false, true, true, false} {
		r.moeRowsOff = !grouped
		w := time.Now()
		lg, err := r.PrefillLast(ctx, embs, 0)
		if err != nil {
			t.Fatalf("PrefillLast (grouped=%v): %v", grouped, err)
		}
		d := time.Since(w)
		runs = append(runs, run{grouped, d, append([]float32(nil), lg...)})
		fmt.Fprintf(os.Stderr, "[rows-ab] run %d/4 grouped=%v: %v (%.1f tok/s, %.2f ms/token)\n", i+1, grouped, d.Round(time.Millisecond), float64(M)/d.Seconds(), 1000*d.Seconds()/float64(M))
	}
	for _, x := range runs[1:] {
		for i := range x.logits {
			if math.Float32bits(x.logits[i]) != math.Float32bits(runs[0].logits[i]) {
				t.Errorf("last-row logits differ between runs at %d", i)
				break
			}
		}
	}
	off := (runs[0].wall + runs[3].wall) / 2
	on := (runs[1].wall + runs[2].wall) / 2
	t.Logf("M=%d cold prefill: per-pair %v (%.1f tok/s), grouped %v (%.1f tok/s): %.2fx", M, off.Round(time.Millisecond), float64(M)/off.Seconds(),
		on.Round(time.Millisecond), float64(M)/on.Seconds(), off.Seconds()/on.Seconds())
}
