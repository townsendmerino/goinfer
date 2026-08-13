//go:build cuda && goinfer_testhooks

package cuda

import (
	"os"
	"strconv"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestA13_PrefillChurnPoisons is the tag-blocking measurement.
//
// A13 established that a large hold-and-release INSIDE A LIVE CONTEXT can leave later launches
// returning success and writing nothing. Four production paths were enumerated and are clean by
// construction — admin unload destroys the context, capSlots is pure arithmetic, allocSlots discards
// the resident on failure, and no mid-life KV/expert resize exists. One is not:
//
//	cuda/prefill.go:200 allocates `scratch` per call and releases it with a deferred
//	r.dev.ReleaseBuf loop, inside the live resident context. Its own comment: "at M=3000 this is
//	hundreds of MB".
//
// And that release really does return memory to the driver: ReleaseBuf -> Buffer.Close ->
// cudaresult.MemFree. No pool, no reuse. So the stimulus occurs on the hot path on every long
// prompt, and whether it poisons is a measurement rather than an argument.
//
// SIZE IS NOT AN ARGUMENT HERE. "Hundreds of MB" lands in the band that was reliably clean, but the
// sweep is non-monotonic — 15% (~1.1 GiB) poisoned once in three while 18% and 21% were clean 3/3 —
// so nothing is safe by being under a number. Only the measurement counts.
//
// TWO SYMPTOMS, reported separately because they mean different things:
//
//	(a) the prefill's own logits degrading across repetitions -> a correctness bug in SHIPPED output
//	(b) a probe launch on the same context failing afterwards  -> narrower, still real
//
// POSITIVE CONTROL (GOINFER_A13_CHURN_CONTROL=1): reproduce the known poisoning stimulus in this same
// process and code path and confirm it DOES poison. A clean result from a harness that cannot poison
// is not evidence — the fourth time in this campaign a null needed its forcing mechanism verified
// before it meant anything.
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
		// THE CONTROL MUST STIMULATE THE CONTEXT UNDER TEST. A first version allocated through
		// dev.Primary() and showed nothing — because BuildResident creates its OWN context, so the
		// primary context is a different one and the control never touched the subject. That is the
		// "harness that cannot poison" failure, caught by running the control before believing a
		// clean result rather than after.
		//
		// So this allocates and frees through the RESIDENT's device, on the resident's pinned
		// executor thread (r.do), which is the only place its context is current.
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
		lg, e := rf.PrefillLast(embs, 0)
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
