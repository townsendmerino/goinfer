//go:build darwin

package metal

import (
	"strconv"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestResidencySet_pinsExactlyTheLiveSlots guards the residency-set contract: the buffers pinned
// resident (r.residencyBufs) must be EXACTLY the pool's live slot buffers (r.slotBuffers()). The set
// is built once at BuildResident and attached to the queue for the resident's life with no removal —
// so if a future change reallocates slot buffers after the set is built (or adds/removes slots), the
// set would pin stale/dangling allocations. This fails loudly at build instead. Uses the small
// gemma4-moe-tiny fixture in paged mode; capability-gated (skips where MTLResidencySet is absent).
func TestResidencySet_pinsExactlyTheLiveSlots(t *testing.T) {
	if !ResidencySetsSupported() {
		t.Skip("no MTLResidencySet (needs macOS 15+)")
	}
	const ckpt = "../testdata/gemma4-moe-tiny"
	m, err := decoder.Load(ckpt, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Skipf("no fixture: %v", err)
	}
	defer m.Close()

	t.Setenv("GOINFER_GEMMA4_RESIDENT", "1")
	t.Setenv("GOINFER_METAL_MOE_SLOTS", strconv.Itoa(3)) // < nE (4) → paged
	t.Setenv("GOINFER_MOE_RESIDENCY", "1")               // default-on anyway; explicit for the gate

	r, err := BuildResident(m)
	if err != nil {
		t.Fatalf("BuildResident: %v", err)
	}
	defer r.Close()
	if r.g4moe == nil || !r.g4moe.paged {
		t.Skip("fixture did not build paged (residency set only applies to the paged path)")
	}

	live := r.slotBuffers()
	pinned := r.residencyBufs
	if len(pinned) == 0 {
		t.Fatalf("residency enabled + paged, but no buffers were pinned (r.residencyBufs empty)")
	}
	if len(pinned) != len(live) {
		t.Fatalf("pinned set (%d) != live slot buffers (%d) — the residency set no longer matches the pool "+
			"(slot reallocation would dangle)", len(pinned), len(live))
	}
	liveSet := make(map[Buffer]bool, len(live))
	for _, b := range live {
		liveSet[b] = true
	}
	for i, b := range pinned {
		if !liveSet[b] {
			t.Fatalf("pinned buffer %d is not a live pool slot — the residency set holds a stale/foreign "+
				"allocation", i)
		}
	}
	t.Logf("residency set pins exactly the %d live slot buffers", len(pinned))
}
