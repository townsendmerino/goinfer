//go:build cuda && goinfer_testhooks

package cuda

import (
	"os"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestResidentCloseFreesVRAM_7B is TestResidentCloseFreesVRAM's shape at the scale that actually
// reproduces A12: qwen2.5-7B and a real decode loop, rather than the 0.5B coder and one token.
//
// WHY A SECOND GATE RATHER THAN WIDENING THE FIRST. A12 measured TestB2DenseFlagship losing
// 1344 MiB and TestRealForwardParity 1166 MiB, each alone in its own process, each already
// deferring Close(). The existing gate is green throughout — accurately, for what it covers. It is
// not tautological and it is not exercised-but-never-triggered: it is CORRECTLY SCOPED AND SILENTLY
// NARROW, which is the variant that looks most like a working gate. Keeping both makes the scopes
// visible side by side instead of hiding one inside the other.
//
// WHAT THIS DISTINGUISHES, pre-registered before the run (A12):
//
//	loss on cycle 1, ~zero on 2 and 3   -> CONTEXT-level retention, almost certainly the
//	                                       local-memory backing store A9/A10 measured. NOT a leak:
//	                                       a one-time cost per context per kernel set. Close tears
//	                                       down a MODEL; it does not destroy the CONTEXT, and every
//	                                       kernel a 7B decode touches that a 0.5B one-token forward
//	                                       does not will have reserved backing store no model-level
//	                                       Close can return.
//	loss repeating every cycle          -> a genuine leak; hunt for what Close does not release.
//	loss shrinking but not vanishing    -> both, and the components separate before either is fixed.
//
// The differing magnitudes (-1344 vs -1166 from the same 7310 MiB start) already favour the first:
// a fixed per-model leak would repeat a fixed size, whereas different kernel sets reserving
// different backing stores would not.
func TestResidentCloseFreesVRAM_7B(t *testing.T) {
	requireHeavyModel(t)
	gguf := os.ExpandEnv("$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(gguf); err != nil {
		t.Skipf("no 7B gguf at %s", gguf)
	}
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	dev, err := gc.GetDevice(0)
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	// A context of our own purely to read MemInfo, held for the whole test so the driver keeps a
	// stable baseline underneath the cycles.
	probe, err := dev.Primary()
	if err != nil {
		t.Skipf("primary ctx: %v", err)
	}
	used := func() uint64 {
		free, total, e := probe.MemInfo()
		if e != nil {
			t.Fatalf("MemInfo: %v", e)
		}
		return (total - free) / (1 << 20)
	}

	base := used()
	t.Logf("baseline %d MiB", base)

	const cycles = 3
	var afterClose []uint64
	for i := range cycles {
		m, err := decoder.Load(gguf, decoder.Options{Backend: "cuda", Quant: "int4"})
		if err != nil {
			t.Fatalf("cycle %d: load: %v", i, err)
		}
		if !m.ResidentActive() {
			_ = m.Close()
			t.Skip("cuda declined this model — nothing resident to leak")
		}
		// A REAL DECODE LOOP, not one token. The whole point of this gate is the kernel set: a
		// single forward touches a fraction of the kernels a decode loop does, and an unreturned
		// local-memory backing store is per-kernel. Sixteen steps is enough to pull in the decode
		// path's kernels without making the gate slow.
		rf := m.ResidentForwardForTest()
		for step := range 16 {
			if _, e := rf.Forward(m.EmbedResidentForTest(1), step); e != nil {
				t.Fatalf("cycle %d step %d: forward: %v", i, step, e)
			}
		}
		loaded := used()
		if e := m.Close(); e != nil {
			t.Fatalf("cycle %d: close: %v", i, e)
		}
		c := used()
		afterClose = append(afterClose, c)
		t.Logf("cycle %d: loaded %d MiB (+%d over baseline), after Close %d MiB (+%d)",
			i, loaded, int64(loaded)-int64(base), c, int64(c)-int64(base))
	}

	// REPORTS, then asserts only what the measurement supports. The per-cycle deltas are the
	// discriminator; printing them means the next reader re-derives nothing.
	for i, c := range afterClose {
		t.Logf("  retained after cycle %d: %+d MiB over baseline", i, int64(c)-int64(base))
	}
	if len(afterClose) >= 3 {
		d12 := int64(afterClose[1]) - int64(afterClose[0])
		d23 := int64(afterClose[2]) - int64(afterClose[1])
		t.Logf("  cycle-to-cycle growth AFTER the first: %+d then %+d MiB", d12, d23)

		// THE ASSERTION IS ON STEADY STATE, NOT ON THE FIRST CYCLE. A one-time per-context cost is
		// legitimate and unreturnable by a model-level Close; unbounded growth is not. 64 MiB of
		// slack per cycle is far below a 7B model's ~4 GiB and far above driver bookkeeping.
		const perCycleSlackMiB = 64
		if d12 > perCycleSlackMiB || d23 > perCycleSlackMiB {
			t.Errorf("VRAM grows on EVERY load/close cycle (+%d then +%d MiB after the first) — this "+
				"is a genuine leak, not one-time context retention. Close is not releasing something "+
				"the 7B decode path allocates. See A12.", d12, d23)
		}
	}

	// SCOPE PRINTED WITH THE VERDICT (axis composition, applied to a resource gate). The sibling
	// gate's green is accurate and reads as supporting a claim three orders of magnitude wider;
	// saying what was covered is what stops that.
	t.Logf("SCOPE: model=qwen2.5-7b-instruct-q4_k_m quant=int4 backend=cuda workload=16-step decode "+
		"cycles=%d steady-state tolerance=64 MiB/cycle. A green here says NOTHING about other "+
		"models, longer contexts, MoE expert caches, or the staged (non-resident) path.", cycles)
}
