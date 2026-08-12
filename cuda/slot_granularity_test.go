//go:build cuda && goinfer_testhooks

package cuda

import (
	"os"
	"sort"
	"strconv"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// slotQuantum is the CUDA driver's allocation granularity on this class of device, measured rather
// than assumed: cuda/allocgran_test.go asserts it, and asserts that it is NOT next-power-of-two
// (5 MiB → 6, 6 → 6, 9 → 10). Sub-quantum requests are pool-served but not free.
const slotQuantum = 2 << 20

// TestSlotAllocation_matchesGranularityForm asserts that the expert cache's VRAM consumption is
// predicted by rounding EACH slot buffer up to the driver's allocation quantum independently.
//
// This is the gate that would have caught A1. The cap arithmetic in allocSlots sizes the cache from
// a raw byte sum — slots × bytes-per-slot × layers — and the driver charges for whole quanta, four
// times per layer. On the real 26B the shortfall put the granted cap one step past what fits: at 34
// slots all four buffers tip a quantum at once, a 4-quanta step per layer, and the forward died in
// a one-block routing kernel with 189.6 MiB still free. Every test in the suite passed throughout,
// because allocSlots runs in every MoE test and its arithmetic is never compared against what the
// driver actually took.
//
// Structure is asserted BEFORE totals, deliberately. A total that matches under a wrong structure
// is worse than a mismatch: it looks like confirmation. So the shape is pinned first — one buffer
// group per MoE layer, four buffers each, four distinct sizes in the ratio the int4/group-32 layout
// implies — and only then is the arithmetic believed.
//
// Landing requirement for A5, which replaces the division with a search over the same form.
//
// MUTATION CHECK (run before trusting this green): change roundUp's body to `return n`, i.e. sum the
// requested sizes without rounding. That is exactly the defect A1 was. At 16 slots on the scaled
// fixture the prediction drops from 226,492,416 to 214,106,112 and the test fails by 12,386,304 B.
func TestSlotAllocation_matchesGranularityForm(t *testing.T) {
	dir := os.Getenv("GOINFER_MOE_SCALED_FIXTURE")
	if dir == "" {
		dir = "../testdata/gemma4-moe-scaled"
	}
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		// Stated reason, and it names the remedy. A silent skip here is indistinguishable from a
		// pass, which is the shape this whole gate exists to reject.
		t.Skipf("no MoE fixture weights at %s/model.safetensors — run scripts/pin_gemma4_moe_scaled.py "+
			"(weights are gitignored; the fixture pins its own sha256)", dir)
	}
	slots := 16
	if v, err := strconv.Atoi(os.Getenv("GOINFER_MOE_CACHE_SLOTS")); err == nil && v > 0 {
		slots = v
	}
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "1")
	t.Setenv("GOINFER_MOE_CACHE_EXPERTS", "1")
	t.Setenv("GOINFER_MOE_CACHE_SLOTS", strconv.Itoa(slots))

	m, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load scaled MoE fixture: %v", err)
	}
	defer m.Close()
	rf := m.ResidentForwardForTest()
	if rf == nil {
		t.Fatal("cuda resident DECLINED the scaled MoE fixture — allocSlots was never reached, so " +
			"this run asserts nothing about the allocation path")
	}
	r := rf.(*cudaResident)
	if r.dbgFreeBefore == 0 || r.dbgFreeAfter == 0 {
		t.Fatal("free-VRAM readings around allocSlots are unset — the instrument did not run, which " +
			"is not the same as the allocation being free")
	}
	if len(r.dbgAllocSizes) == 0 {
		t.Fatal("no slot allocations were recorded — the expert cache did not allocate, so there is " +
			"nothing to compare the form against")
	}

	// ---- structure, before any total is believed ----
	counts := map[int]int{}
	for _, n := range r.dbgAllocSizes {
		counts[n]++
	}
	sizes := make([]int, 0, len(counts))
	for n := range counts {
		sizes = append(sizes, n)
	}
	sort.Ints(sizes)
	for _, n := range sizes {
		t.Logf("  buffer size %12d B  x%d", n, counts[n])
	}

	if len(sizes) != 4 {
		t.Fatalf("STRUCTURE: %d distinct buffer sizes, expected 4 (expGU.W, expGU.ws16, expDown.W, "+
			"expDown.ws16). The model of what allocSlots allocates is wrong, so the totals below "+
			"cannot be interpreted whether they match or not", len(sizes))
	}
	nLayers := counts[sizes[0]]
	for _, n := range sizes {
		if counts[n] != nLayers {
			t.Errorf("STRUCTURE: size %d occurs %dx, but size %d occurs %dx — buffers are allocated "+
				"once per MoE layer, so every distinct size must occur the same number of times",
				n, counts[n], sizes[0], nLayers)
		}
	}
	if got, want := len(r.dbgAllocSizes), 4*nLayers; got != want {
		t.Errorf("STRUCTURE: %d allocations recorded, expected %d (4 buffers x %d MoE layers)",
			got, want, nLayers)
	}
	// The 1:2:8:16 ratio encodes the layout, not a coincidence: expGU carries twice expDown's rows,
	// and a packed-int4 W holds 8 weights per uint32 against one f16 scale per 32 weights, so W is
	// 8x its scale buffer in bytes. If the quantization layout changes, this fails and says so
	// rather than passing on a shape nobody re-derived.
	unit := sizes[0]
	for i, mul := range []int{1, 2, 8, 16} {
		if want := unit * mul; sizes[i] != want {
			t.Errorf("STRUCTURE: buffer sizes are not in the 1:2:8:16 ratio the int4/group-32 layout "+
				"implies — sizes[%d] = %d, expected %d (unit %d)", i, sizes[i], want, unit)
		}
	}
	if t.Failed() {
		t.Fatal("structure is wrong; not proceeding to the totals")
	}

	// ---- totals ----
	roundUp := func(n int) int64 { return int64((n + slotQuantum - 1) / slotQuantum * slotQuantum) }
	var raw, rounded int64
	for _, n := range r.dbgAllocSizes {
		raw += int64(n)
		rounded += roundUp(n)
	}
	actual := int64(r.dbgFreeBefore) - int64(r.dbgFreeAfter)

	t.Logf("slots=%d, %d MoE layers, %d allocations", r.cacheSlots, nLayers, len(r.dbgAllocSizes))
	t.Logf("  free before allocSlots %13d B", r.dbgFreeBefore)
	t.Logf("  free after  allocSlots %13d B", r.dbgFreeAfter)
	t.Logf("  actual consumed        %13d B", actual)
	t.Logf("  raw sum of requests    %13d B  (what the cap arithmetic assumes)", raw)
	t.Logf("  granularity-rounded    %13d B  (sum of ceil(size/%d)*%d)", rounded, slotQuantum, slotQuantum)
	t.Logf("  shortfall of the raw sum %11d B", rounded-raw)

	if actual != rounded {
		t.Errorf("allocation does not match the granularity form: actual %d B, predicted %d B, "+
			"residual %d B (%.3f quanta over %d buffers). Either the quantum is not %d on this "+
			"device, or allocSlots allocates something this model does not know about — A5's cap "+
			"search is derived from this form and is wrong by the same amount",
			actual, rounded, actual-rounded, float64(actual-rounded)/slotQuantum,
			len(r.dbgAllocSizes), slotQuantum)
	}
	// Guard the gate against becoming vacuous. If rounding ever stops mattering for the chosen slot
	// count, the assertion above would pass against the raw sum too and would no longer be testing
	// the thing it was written for.
	if rounded == raw {
		t.Errorf("rounding is a no-op at %d slots (every buffer happens to be quantum-aligned), so "+
			"this run cannot distinguish the granularity form from the raw sum the cap arithmetic "+
			"already uses — pick a slot count where it bites", r.cacheSlots)
	}
}
