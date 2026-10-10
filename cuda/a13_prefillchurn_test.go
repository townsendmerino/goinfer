//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"os"
	"strconv"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestA13_PrefillChurnPoisons measures the one shipped path that releases a large buffer inside a live
// context: the prefill's per-call M-sized scratch, freed through ReleaseBuf -> Buffer.Close ->
// cudaresult.MemFree with no pool, so every long prompt does a multi-hundred-MB free. Whether that
// poisons later launches is a measurement, not an argument; a synthetic hold-and-release sweep is not
// evidence for or against it (it proved intermittent).
//
// Two symptoms, reported separately:
//
//	(a) the prefill's own logits degrading across repetitions -> a correctness bug in shipped output
//	(b) a probe launch on the same context failing afterwards  -> narrower, still real
//
// Positive control (GOINFER_A13_CHURN_CONTROL=1): reproduce the known poisoning stimulus in this same
// process and code path and confirm it does poison. A clean result from a harness that cannot poison
// is not evidence.
//
// Record: docs/queue-engineering.md, section A13. Detail: docs/code-notes/cuda.md#TestA13_PrefillChurnPoisons.
func TestA13_PrefillChurnPoisons(t *testing.T) {
	if os.Getenv("GOINFER_A13_CHURN") == "" {
		t.Skip("set GOINFER_A13_CHURN=1 — A13 probe, deliberately not part of the tier")
	}
	const path = "../testdata/mistral-tiny-window"
	requireDeviceAndFixture(t, path)

	mc, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cuda): %v", err)
	}
	defer mc.Close()
	rf, ok := mc.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatal("resident is not *cudaResident")
	}
	if !rf.prefillReady {
		t.Fatal("batched prefill kernels did not load — PrefillLast would decline")
	}
	_, _, _, _, _, _, vocab := mc.Dims()

	// A long prompt: the scratch is sized off M, and the comment names M≈3000 as the "hundreds of
	// MB" case. Kept configurable so the churn can be scaled without editing.
	M := 3000
	if v := os.Getenv("GOINFER_A13_M"); v != "" {
		if k, e := strconv.Atoi(v); e == nil {
			M = k
		}
	}
	N := 8
	if v := os.Getenv("GOINFER_A13_N"); v != "" {
		if k, e := strconv.Atoi(v); e == nil {
			N = k
		}
	}
	prompt := make([]int, M)
	var s uint32 = 20260813
	for i := range prompt {
		s = s*1664525 + 1013904223
		prompt[i] = int(s>>8) % (vocab - 1)
	}
	embs := make([][]float32, M)
	for i, tok := range prompt {
		embs[i] = append([]float32(nil), mc.EmbedResidentForTest(tok)...)
	}

	// POSITIVE CONTROL: force the known stimulus first, so a later clean result cannot be a harness
	// that is simply incapable of showing the effect.
	if os.Getenv("GOINFER_A13_CHURN_CONTROL") != "" {
		// The control must stimulate the context under test: allocate and free through the resident's device
		// on its pinned executor thread (r.do), the only place its context is current. Detail: docs/code-notes/cuda.md#TestA13_PrefillChurnPoisons.control.
		if e := rf.do(func() error {
			free, _, _ := rf.dev.Context().MemInfo()
			var held []Buffer
			const chunk = 64 << 20
			for got := 0; got+chunk <= int(free)/2; got += chunk {
				held = append(held, rf.dev.MustBuf(chunk, chunk/4, "a13-control"))
			}
			t.Logf("CONTROL: held %d x 64 MiB = %.1f MiB in the RESIDENT's context, releasing",
				len(held), float64(len(held)*chunk)/(1<<20))
			for _, b := range held {
				rf.dev.ReleaseBuf(b)
			}
			return nil
		}); e != nil {
			t.Fatalf("control: %v", e)
		}
	}

	// (a) THE SHIPPED SYMPTOM: run the same prefill N times and compare each result to the first.
	// Identical input must give identical logits; a drift or a zero run is the bug.
	var first []float32
	for i := range N {
		lg, e := rf.PrefillLast(context.Background(), embs, 0)
		if e != nil {
			t.Fatalf("prefill %d: %v", i, e)
		}
		cp := append([]float32(nil), lg...)
		var nz int
		for _, v := range cp {
			if v != 0 {
				nz++
			}
		}
		if i == 0 {
			first = cp
			t.Logf("(a) prefill 0: %d/%d non-zero logits", nz, len(cp))
			continue
		}
		diff := 0
		for j := range cp {
			if cp[j] != first[j] {
				diff++
			}
		}
		t.Logf("(a) prefill %d: %d/%d non-zero, %d logits differ from run 0", i, nz, len(cp), diff)
		if nz == 0 {
			t.Errorf("(a) prefill %d produced ALL-ZERO logits — shipped output is silently wrong "+
				"after %d prefill scratch alloc/free cycles in a live context. See A13.", i, i)
		}
		if diff != 0 {
			t.Errorf("(a) prefill %d differs from run 0 in %d logits on IDENTICAL input — the "+
				"context degraded across prefill churn. See A13.", i, diff)
		}
	}

	// (b) THE SYNTHETIC SYMPTOM: a decode forward on the same resident, after all that churn.
	lg, e := rf.Forward(embs[0], 0)
	if e != nil {
		t.Fatalf("(b) post-churn forward: %v", e)
	}
	nz := 0
	for _, v := range lg {
		if v != 0 {
			nz++
		}
	}
	t.Logf("(b) post-churn decode forward: %d/%d non-zero logits", nz, len(lg))
	if nz == 0 {
		t.Errorf("(b) a decode forward on the same context after %d prefills produced ALL ZEROS", N)
	}
}
