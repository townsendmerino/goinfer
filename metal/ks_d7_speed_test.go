//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"fmt"
	"math"
	"os"
	"slices"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// ksd7Load loads path on the Metal resident with the per-weight dequant (old) or the half2 one, and returns it with its batched
// prefill.
func ksd7Load(t *testing.T, path string, opts decoder.Options, old bool) (*decoder.Model, decoder.ResidentForward, decoder.Prefiller) {
	t.Helper()
	prev := ksd7OldDequantForTest
	ksd7OldDequantForTest = old
	defer func() { ksd7OldDequantForTest = prev }()
	m, err := decoder.Load(path, opts)
	if err != nil {
		t.Fatalf("load %s: %v", path, err)
	}
	t.Cleanup(func() { m.Close() })
	decoder.SetKnobEnvForTest(t, m, "GOINFER_METAL_FAST_PREFILL_FLOOR", "0")
	rf := m.ResidentForwardForTest()
	if rf == nil {
		t.Fatalf("%s did not go resident: %s", path, m.ResidentDecline())
	}
	pf, ok := rf.(decoder.Prefiller)
	if !ok {
		t.Fatalf("%s: no batched prefill", path)
	}
	return m, rf, pf
}

func ksd7Prompt(m *decoder.Model, n int) [][]float32 {
	_, _, _, _, _, _, vocab := m.Dims()
	embs := make([][]float32, n)
	for i := range embs {
		embs[i] = m.EmbedResidentForTest((i*7919 + 13) % vocab)
	}
	return embs
}

func ksd7Differ(a, b []float32) int {
	d := 0
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			d++
		}
	}
	return d
}

// TestKSD7_prefillIdentity is D7-G1's tiny half (docs/tasks/task-kernel-survey-2026-10.md, "KS-D7 gates"): llama-tiny at int4, a
// 100-token batched prefill's last logits with the half2 dequant against the per-weight one, byte-equal.
func TestKSD7_prefillIdentity(t *testing.T) {
	const path = "../testdata/llama-tiny"
	opts := decoder.Options{Backend: "metal", Quant: "int4"}
	mo, rfo, pfo := ksd7Load(t, path, opts, true)
	mn, rfn, pfn := ksd7Load(t, path, opts, false)
	rfo.Reset()
	rfn.Reset()
	a, err := pfo.PrefillLast(context.Background(), ksd7Prompt(mo, 100), 0)
	if err != nil {
		t.Fatal(err)
	}
	a = append([]float32(nil), a...)
	b, err := pfn.PrefillLast(context.Background(), ksd7Prompt(mn, 100), 0)
	if err != nil {
		t.Fatal(err)
	}
	d := ksd7Differ(a, b)
	fmt.Fprintf(os.Stderr, "[D7-G1] llama-tiny int4, 100-token prefill: %d of %d logits differ\n", d, len(a))
	if d != 0 {
		t.Errorf("%d logits differ between the dequant forms", d)
	}
}

// TestKSD7_prefillSpeed is D7-G2, and D7-G1's real half: Qwen2.5-Coder-1.5B (the Metal .giw), a 3,900-token prompt, the
// resident's PrefillLast wall time with the per-weight dequant (A) and the half2 one (B), two residents in one process, A B B A
// over 10 pairs after 2 warm runs each; the first runs' logits must be byte-equal. A timed run: hold the timing lock.
//
//	GOINFER_KSD7_TIMING=1 python3 scripts/timing_lock.py run --label ksd7-g2 -- go test -tags goinfer_testhooks -count=1 -run '^TestKSD7_prefillSpeed$' -v ./metal/
func TestKSD7_prefillSpeed(t *testing.T) {
	if os.Getenv("GOINFER_KSD7_TIMING") == "" {
		t.Skip("set GOINFER_KSD7_TIMING=1 (a timed run under the timing lock)")
	}
	path := os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.int4.metal.v17.giw")
	opts := decoder.Options{Backend: "metal", ResidentContext: 4096}
	mo, rfo, pfo := ksd7Load(t, path, opts, true)
	_, rfn, pfn := ksd7Load(t, path, opts, false)
	embs := ksd7Prompt(mo, 3900)
	run := func(rf decoder.ResidentForward, pf decoder.Prefiller) (float64, []float32) {
		rf.Reset()
		t0 := time.Now()
		lg, err := pf.PrefillLast(context.Background(), embs, 0)
		if err != nil {
			t.Fatal(err)
		}
		return time.Since(t0).Seconds(), append([]float32(nil), lg...)
	}
	_, la := run(rfo, pfo)
	_, lb := run(rfn, pfn)
	d := ksd7Differ(la, lb)
	fmt.Fprintf(os.Stderr, "[D7-G1] Qwen2.5-Coder-1.5B, 3,900-token prefill: %d of %d logits differ\n", d, len(la))
	if d != 0 {
		t.Errorf("%d logits differ between the dequant forms", d)
	}
	run(rfo, pfo)
	run(rfn, pfn)
	var ratios []float64
	below := 0
	for i := range 10 {
		var a, b float64
		if i%2 == 0 {
			a, _ = run(rfo, pfo)
			b, _ = run(rfn, pfn)
		} else {
			b, _ = run(rfn, pfn)
			a, _ = run(rfo, pfo)
		}
		ratios = append(ratios, b/a)
		if b < a {
			below++
		}
		fmt.Fprintf(os.Stderr, "[D7-G2] pair %d: per-weight %.1f ms, half2 %.1f ms, new/old %.4f\n", i, a*1e3, b*1e3, b/a)
	}
	slices.Sort(ratios)
	median := (ratios[4] + ratios[5]) / 2
	fmt.Fprintf(os.Stderr, "[D7-G2] median new/old %.4f, %d of 10 pairs below 1\n", median, below)
}
