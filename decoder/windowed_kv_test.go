package decoder

import (
	"testing"
)

// G-W4 of docs/tasks/task-cuda-windowed-kv-2026-10.md, the decoder half: with Options.ResidentWindowedKV the CUDA price of a sliding-window
// layer's KV is min(ctx, window+WindowedKVSlack) positions, and Plan chooses a context against that price. The allocation it must equal is
// cuda's TestWindowedKV_allocation (needs a GPU).

func loadWindowFixture(t *testing.T, on bool) *Model {
	t.Helper()
	m, err := Load("../testdata/mistral-tiny-window", Options{Quant: "f32", ResidentWindowedKV: on})
	if err != nil {
		t.Skipf("no fixture: %v", err)
	}
	t.Cleanup(func() { m.Close() })
	return m
}

func TestWindowedKV_planPrices(t *testing.T) {
	off, on := loadWindowFixture(t, false), loadWindowFixture(t, true)
	window := on.SlidingWindowResident()
	if window <= 0 {
		t.Fatalf("fixture has no sliding window")
	}
	_, nLayers, _, _, _, _, _ := on.Dims()
	for _, ctx := range []int{window, window + WindowedKVSlack, window + WindowedKVSlack + 1, 4096, 100000} {
		var want int64
		for l := range nLayers {
			if !on.WindowedKVLayer(l) {
				t.Fatalf("layer %d of an all-sliding-window model is not a windowed-KV layer", l)
			}
			want += int64(on.w.arch.kvDimAt(l)) * 2 * 4 * int64(min(ctx, window+WindowedKVSlack))
		}
		if got := on.ResidentKVBytes("cuda", ctx, false, false); got != want {
			t.Errorf("ctx %d: ResidentKVBytes(cuda) = %d, want %d", ctx, got, want)
		}
		if ctx <= window+WindowedKVSlack {
			if a, b := on.ResidentKVBytes("cuda", ctx, false, false), off.ResidentKVBytes("cuda", ctx, false, false); a != b {
				t.Errorf("ctx %d at or below window+slack: windowed %d != full %d", ctx, a, b)
			}
		}
		if off.WindowedKVLayer(0) {
			t.Fatal("WindowedKVLayer is true with the option off")
		}
	}
	if full, win := off.ResidentKVBytes("cuda", 100000, false, false), on.ResidentKVBytes("cuda", 100000, false, false); win*100 >= full {
		t.Errorf("at ctx 100000 the windowed KV (%d) is not a small fraction of the full KV (%d)", win, full)
	}
}

// TestWindowedKV_planChoosesLongerContext: for the same budget, Plan keeps a context the full-KV price declines or shrinks.
func TestWindowedKV_planChoosesLongerContext(t *testing.T) {
	off, on := loadWindowFixture(t, false), loadWindowFixture(t, true)
	const ctxWant = 200000
	dense := on.ResidentDenseWeightBytes()
	budget := dense + off.ResidentKVBytes("cuda", 20000, false, false) // room for 20k positions of full KV
	pOff := off.Plan("cuda", budget, PlanRequest{Ctx: ctxWant})
	pOn := on.Plan("cuda", budget, PlanRequest{Ctx: ctxWant})
	if pOn.Placement == PlacementDecline {
		t.Fatalf("windowed Plan declined: %s", pOn.Reason)
	}
	if pOn.Ctx != ctxWant {
		t.Errorf("windowed Plan ctx = %d, want the requested %d (its KV is %d bytes, budget %d)", pOn.Ctx, ctxWant, pOn.KVBytes, budget)
	}
	if pOff.Placement != PlacementDecline && pOff.Ctx >= pOn.Ctx {
		t.Errorf("full-KV Plan ctx %d is not below the windowed %d: the test does not exercise the saving", pOff.Ctx, pOn.Ctx)
	}
	if pOn.KVBytes != on.ResidentKVBytes("cuda", pOn.Ctx, false, false) {
		t.Errorf("Plan KVBytes %d != priced %d at ctx %d", pOn.KVBytes, on.ResidentKVBytes("cuda", pOn.Ctx, false, false), pOn.Ctx)
	}
	if pOn.NeedBytes() > budget {
		t.Errorf("NeedBytes %d exceeds budget %d", pOn.NeedBytes(), budget)
	}
}

// TestWindowedKV_gatesOnLayer: a layer is windowed only when it is a sliding-window layer of an attention family.
func TestWindowedKV_gatesOnLayer(t *testing.T) {
	m, err := Load("../testdata/llama-tiny", Options{Quant: "f32", ResidentWindowedKV: true})
	if err != nil {
		t.Skipf("no fixture: %v", err)
	}
	defer m.Close()
	_, nLayers, _, _, _, _, _ := m.Dims()
	for l := range nLayers {
		if m.WindowedKVLayer(l) {
			t.Errorf("layer %d of a full-attention model is a windowed-KV layer", l)
		}
	}
}
