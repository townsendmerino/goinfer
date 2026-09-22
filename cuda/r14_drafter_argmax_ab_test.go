//go:build cuda && goinfer_testhooks

package cuda

import (
	"os"
	"sort"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestR14DrafterArgmaxAB is R14's build gate (docs/measurements/r14-drafter-argmax-2026-09-22.md):
// the device argmax_rows tail against the host-loop tail it replaces, on one loaded model, the
// gate-3 code suite, arms flipped between prompts, ABBA. Correctness first: with the check mode
// armed every device call is compared row-for-row against the host loop on the SAME logits (a
// mismatch fails the test), and the emitted token sequences of the two arms must be identical
// (the lossless contract). Then the paired spec-wall speedup, LOGGED against the registered bar.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks' ./cuda/ -run TestR14DrafterArgmaxAB -v -timeout 30m
func TestR14DrafterArgmaxAB(t *testing.T) {
	requireHeavyModel(t)
	tgt := os.Getenv("GOINFER_CUDA_MODEL")
	if tgt == "" {
		tgt = os.ExpandEnv("$HOME/models/qwen3-4b")
	}
	ddir := decoder.AssetPathForTest(t, "GOINFER_DFLASH_F32")
	mc, err := decoder.Load(tgt, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer mc.Close()
	r := mc.ResidentForwardForTest().(*cudaResident)
	dr, err := decoder.LoadDFlashDrafter(ddir)
	if err != nil {
		t.Fatalf("load drafter: %v", err)
	}
	defer dr.Close()
	rd, err := r.AttachDrafter(dr)
	if err != nil {
		t.Fatalf("AttachDrafter: %v", err)
	}
	taps := dr.TargetLayerIDs()
	tk, err := decoder.LoadTokenizerForTest(tgt)
	if err != nil {
		t.Skipf("tokenizer: %v", err)
	}
	prompts := []string{
		"Write a Python function that returns the nth Fibonacci number.",
		"Write a Go function that reverses a slice of ints in place.",
	}
	const w, maxNew = 7, 96
	var ids [][]int
	for _, p := range prompts {
		id, e := decoder.EncodeChatForTest(tk, p)
		if e != nil {
			t.Fatalf("encode: %v", e)
		}
		ids = append(ids, id)
	}
	run := func(host bool) (out [][]int, wallMs float64) {
		r.SetHostArgmaxForTest(host)
		for _, id := range ids {
			got, _, ms := dflashLoop(t, mc, r, rd, taps, id, maxNew, w, dr.MaskTokenID())
			out = append(out, got)
			wallMs += ms
			rd.TruncateContext(0)
		}
		return out, wallMs
	}

	// Warm-up both arms, discarded.
	run(true)
	run(false)

	// Correctness: device arm with per-call row check against the host loop on the same logits.
	r.SetArgmaxCheckForTest(true)
	devOut, _ := run(false)
	checked := r.ArgmaxChecksForTest()
	r.SetArgmaxCheckForTest(false)
	if checked == 0 {
		t.Fatal("check mode compared nothing — the device tail did not run")
	}
	hostOut, _ := run(true)
	for i := range devOut {
		if len(devOut[i]) != len(hostOut[i]) {
			t.Fatalf("prompt %d: device arm emitted %d tokens, host arm %d", i, len(devOut[i]), len(hostOut[i]))
		}
		for k := range devOut[i] {
			if devOut[i][k] != hostOut[i][k] {
				t.Fatalf("prompt %d token %d: device %d vs host %d — NOT lossless", i, k, devOut[i][k], hostOut[i][k])
			}
		}
	}
	t.Logf("bit-identical: %d device argmax calls matched the host loop row-for-row; emitted sequences equal on %d prompts", checked, len(ids))

	// Paired perf, ABBA.
	const pairs = 6
	ratios := make([]float64, 0, pairs)
	var sumH, sumD float64
	for p := 0; p < pairs; p++ {
		var h, d float64
		if p%2 == 0 {
			_, h = run(true)
			_, d = run(false)
		} else {
			_, d = run(false)
			_, h = run(true)
		}
		ratios = append(ratios, h/d)
		sumH += h
		sumD += d
		t.Logf("pair %d: host %.0f ms | device %.0f ms | speedup %.3fx", p, h, d, h/d)
	}
	sort.Float64s(ratios)
	median := (ratios[len(ratios)/2-1] + ratios[len(ratios)/2]) / 2
	verdict := "KILL (<1%)"
	switch {
	case median >= 1.02:
		verdict = "SHIP (>=2%)"
	case median >= 1.01:
		verdict = "PARK (1-2%)"
	}
	t.Logf("paired median speedup %.3fx (min %.3f, max %.3f) over %d pairs; pooled host %.0f vs device %.0f ms — %s",
		median, ratios[0], ratios[len(ratios)-1], pairs, sumH, sumD, verdict)
}
