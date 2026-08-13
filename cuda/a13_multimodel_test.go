//go:build cuda && goinfer_testhooks

package cuda

import (
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestA13_MultiModelUnloadPoisons is the shipped-path question, and the one that decides the tag.
//
// The correction that produced it: "each resident model builds its own context" is FALSE.
// CreateSystemDefaultDevice calls dev.Primary() — the device's PRIMARY context, retained by
// refcount — and Context.Close calls PrimaryCtxRelease, a decrement. gocudrv does not bind
// cuCtxCreate at all. So every model in a process shares ONE context, destroyed only when the LAST
// holder releases it.
//
// Which means POST /admin/models/unload, with another model loaded, is a multi-gigabyte free INSIDE
// A LIVE CONTEXT — the exact stimulus the A13 sweep showed can leave later launches returning
// success and writing nothing. Not synthetic, not a test artifact: a shipped feature reached by
// ordinary operation.
//
// A is the 7B at int4 (~4.9 GB, ~67% of this card) so its release lands unambiguously inside the
// poisoning range rather than in the unstable band — the sweep was reliable at >=25% and ambiguous
// at 15%. B is small, so both fit at once.
//
// NO CONTROL IS NEEDED FOR A POSITIVE. If B's output degrades after A is unloaded, that is a
// shipping correctness bug and the diagnosis stops there. A clean result is NOT clean yet — it needs
// the (LoadModule, resident-executor) cell to show that this route is poisonable at all.
func TestA13_MultiModelUnloadPoisons(t *testing.T) {
	if os.Getenv("GOINFER_A13_MULTI") == "" {
		t.Skip("set GOINFER_A13_MULTI=1 — A13 probe, deliberately not part of the tier")
	}
	big := os.ExpandEnv("$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(big); err != nil {
		t.Skipf("no 7B at %s", big)
	}
	const small = "../testdata/mistral-tiny-window"
	requireDeviceAndFixture(t, small)

	for round := range 3 {
		func() {
			// B first and kept open for the whole round: it is the holder that keeps the shared
			// primary context alive while A is torn down. That is the entire point.
			mb, err := decoder.Load(small, decoder.Options{Backend: "cuda", Quant: "int4"})
			if err != nil {
				t.Fatalf("round %d: load B: %v", round, err)
			}
			defer mb.Close()
			rb, ok := mb.ResidentForwardForTest().(*cudaResident)
			if !ok {
				t.Skip("B is not resident on cuda — nothing to protect")
			}

			// Baseline from B BEFORE A exists, so a later difference is attributable.
			emb := append([]float32(nil), mb.EmbedResidentForTest(1)...)
			before, err := rb.Forward(emb, 0)
			if err != nil {
				t.Fatalf("round %d: B baseline forward: %v", round, err)
			}
			base := append([]float32(nil), before...)
			nzBase := countNonZero(base)
			if nzBase == 0 {
				t.Fatalf("round %d: B's BASELINE is all zeros — the probe says nothing", round)
			}

			// A: big, resident, then unloaded through the shipped path. Model.Close is exactly what
			// the admin unload handler calls via its detached drain-and-close.
			ma, err := decoder.Load(big, decoder.Options{Backend: "cuda", Quant: "int4"})
			if err != nil {
				t.Fatalf("round %d: load A: %v", round, err)
			}
			if !ma.ResidentActive() {
				_ = ma.Close()
				t.Skipf("round %d: A declined residency — no large device release to make", round)
			}
			if e := ma.Close(); e != nil {
				t.Fatalf("round %d: unload A: %v", round, e)
			}

			// B again, on the context A's release just ran inside.
			after, err := rb.Forward(emb, 0)
			if err != nil {
				t.Fatalf("round %d: B post-unload forward: %v", round, err)
			}
			nzAfter := countNonZero(after)
			diff := 0
			for i := range after {
				if i < len(base) && after[i] != base[i] {
					diff++
				}
			}
			t.Logf("round %d: B non-zero %d -> %d, %d/%d logits differ from baseline",
				round, nzBase, nzAfter, diff, len(base))
			if nzAfter == 0 {
				t.Errorf("round %d: B produces ALL-ZERO logits after unloading A — SHIPPING "+
					"correctness bug: /admin/models/unload silently breaks a co-resident model, with "+
					"no error anywhere. See A13.", round)
			}
			if diff != 0 {
				t.Errorf("round %d: B's logits changed in %d places on IDENTICAL input after "+
					"unloading A — the shared primary context degraded. See A13.", round, diff)
			}
		}()
	}
}

func countNonZero(v []float32) int {
	n := 0
	for _, x := range v {
		if x != 0 {
			n++
		}
	}
	return n
}
