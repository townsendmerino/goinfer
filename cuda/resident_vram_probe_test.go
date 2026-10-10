//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestResidentLaunchVRAMProbe records free VRAM immediately BEFORE every launch on the real 26B at a slot count that fails
// (A1_SLOTS, default 34). describeLaunchErr's "free" reading is taken AFTER r.stream.Launch fails, so it cannot
// distinguish (a) memory released by intervening work from (b) the failed attempt itself freeing a driver-side block while
// unwinding; the pre-launch trace observes the same event from the other side, and also yields A9's first decrement (free
// at first launch to free at the failing launch).
func TestResidentLaunchVRAMProbe(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("GOINFER_HEAVY_TESTS unset — loads the real 26B (~5 min, 11.4 GB pinned host)")
	}
	dir := os.Getenv("GOINFER_GEMMA4_26B")
	if dir == "" {
		home, _ := os.UserHomeDir()
		dir = filepath.Join(home, "models", "gemma-4-26b-a4b-it")
	}
	if _, err := os.Stat(filepath.Join(dir, "config.json")); err != nil {
		t.Skipf("no 26B checkpoint at %s: %v", dir, err)
	}
	slots := os.Getenv("A1_SLOTS")
	if slots == "" {
		slots = "34"
	}
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "1")
	t.Setenv("GOINFER_MOE_CACHE_EXPERTS", "1")
	t.Setenv("GOINFER_MOE_CACHE_SLOTS", slots)

	m, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	rf := m.ResidentForwardForTest()
	if rf == nil {
		// A decline is a SKIP, not a FAIL: "this run says nothing about the probe" is could-not-evaluate, and the resident path
		// declining and falling back is designed behaviour, logged with its reason (B8's rule, applied inside a test).
		t.Skip("could not evaluate: resident DECLINED (see the [cuda] decline line above for the " +
			"KV-vs-free figures), so the launch path was never reached and this run says nothing " +
			"about the probe")
	}
	r := rf.(*cudaResident)
	// Set the probe on the resident directly: an env var read into a package-level var is initialised before t.Setenv runs and
	// records nothing. Load ordering does not matter here: allocSlots is already done and every launch is still ahead.
	r.dbgProbe = true

	tk, err := tokenizer.Load(dir)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	ids, err := tk.Encode("What is the capital of France?", true)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	out, genErr := m.Generate(context.Background(), ids, 4, decoder.SamplingParams{})
	n := 0
	for range out {
		n++
	}

	t.Logf("A1 pre-launch probe: slots requested=%s effective=%d, %d tokens generated", slots, r.cacheSlots, n)
	t.Logf("  CUDA_MODULE_LOADING=%q (empty = driver default LAZY)", os.Getenv("CUDA_MODULE_LOADING"))
	t.Logf("  free BEFORE allocSlots %13d B", r.dbgFreeBefore)
	t.Logf("  free after allocSlots  %13d B", r.dbgFreeAfter)
	t.Logf("  free at FIRST launch   %13d B", r.dbgFreeBeforeLaunch)
	t.Logf("  free before LAST launch attempted %13d B", r.dbgFreePreLaunch)
	if r.dbgFreeAfter > 0 && r.dbgFreeBeforeLaunch > 0 {
		t.Logf("  DECREMENT allocSlots→first launch  %13d B  (STRUCTURALLY ~0: lazy module materialisation\n    happens DURING the first launch, i.e. after this reading — this gap cannot contain it)",
			int64(r.dbgFreeAfter)-int64(r.dbgFreeBeforeLaunch))
	}
	if r.dbgFreeBeforeLaunch > 0 && r.dbgFreePreLaunch > 0 {
		t.Logf("  DECREMENT first launch→last launch %13d B  (the deferred first-launch reservations;\n    138,412,032 of it is moe_route — see TestMoERouteFirstLaunchReservation)",
			int64(r.dbgFreeBeforeLaunch)-int64(r.dbgFreePreLaunch))
	}
	for _, ln := range r.dbgLaunchTrace {
		t.Logf("  trace %s", ln)
	}
	if r.launchErr != nil {
		t.Logf("  launchErr: %v", r.launchErr)
	}
	if genErr != nil {
		t.Logf("  Generate: %v", genErr)
	}
	// Not an assertion on success or failure — at 34 slots this is EXPECTED to fail, and at a
	// working slot count it is expected to pass. What the run has to produce either way is the
	// pre-launch reading; a run without one has measured nothing.
	if len(r.dbgLaunchTrace) == 0 {
		// Same distinction: measuring nothing is not measuring something wrong.
		t.Skip("could not evaluate: no launches were probed — the pre-launch reading is missing, so " +
			"this run measured nothing")
	}
}

func TestResidentSlotConsumption(t *testing.T) {
	dir := os.Getenv("GOINFER_MOE_SCALED_FIXTURE")
	if dir == "" {
		t.Skip("set GOINFER_MOE_SCALED_FIXTURE")
	}
	t.Setenv("GOINFER_MOE_CACHE_EXPERTS", "1")
	t.Setenv("GOINFER_MOE_CACHE_SLOTS", os.Getenv("A1_SLOTS"))
	mc, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer mc.Close()
	rf := mc.ResidentForwardForTest()
	if rf == nil {
		t.Fatal("resident DECLINED — allocSlots not reached")
	}
	r := rf.(*cudaResident)
	if r.dbgFreeBefore == 0 || r.dbgFreeAfter == 0 {
		t.Fatal("allocSlots was NOT reached (instrument unset) — this is not a slow run, it is a run that did not get there")
	}
	slots, _ := strconv.Atoi(os.Getenv("A1_SLOTS"))
	actual := int64(r.dbgFreeBefore) - int64(r.dbgFreeAfter)
	t.Logf("A1 slots=%d (cacheSlots=%d)", slots, r.cacheSlots)
	t.Logf("  free before   %13d B  %.3f GB", r.dbgFreeBefore, float64(r.dbgFreeBefore)/1e9)
	t.Logf("  free after    %13d B  %.3f GB", r.dbgFreeAfter, float64(r.dbgFreeAfter)/1e9)
	t.Logf("  actual        %13d B  %.3f GB", actual, float64(actual)/1e9)
	t.Logf("  predicted     %13d B  %.3f GB", r.dbgPredInline, float64(r.dbgPredInline)/1e9)
	t.Logf("  DELTA         %13d B  %.3f GB  (%.1f%% of predicted)",
		actual-r.dbgPredInline, float64(actual-r.dbgPredInline)/1e9,
		100*float64(actual-r.dbgPredInline)/float64(r.dbgPredInline))
	const q = 2 << 20
	var rounded int64
	for _, n := range r.dbgAllocSizes {
		rounded += int64((n + q - 1) / q * q)
	}
	// STRUCTURE FIRST. A matching total under a wrong structure is worse than a mismatch, so the
	// shape is asserted before the arithmetic is believed: 30 identical MoE layers, four buffers
	// each, means exactly four distinct sizes occurring 30 times apiece.
	counts := map[int]int{}
	for _, n := range r.dbgAllocSizes {
		counts[n]++
	}
	sizes := make([]int, 0, len(counts))
	for n := range counts {
		sizes = append(sizes, n)
	}
	sort.Ints(sizes)
	var distinctSum int
	for _, n := range sizes {
		distinctSum += n
		t.Logf("    size %10d B  x%d", n, counts[n])
	}
	t.Logf("  buffers recorded %d (expect 120); distinct sizes %d (expect 4); distinct sum %d (expect %d)",
		len(r.dbgAllocSizes), len(sizes), distinctSum, slots*3345408)
	if len(r.dbgAllocSizes) != 120 {
		t.Errorf("STRUCTURE: %d allocations, expected 120", len(r.dbgAllocSizes))
	}
	if len(sizes) != 4 {
		t.Errorf("STRUCTURE: %d distinct sizes, expected 4 — the model (30 layers x 4 buffers) is wrong "+
			"regardless of whether the totals match", len(sizes))
	}
	for _, n := range sizes {
		if counts[n] != 30 {
			t.Errorf("STRUCTURE: size %d occurs %dx, expected 30", n, counts[n])
		}
	}
	if want := slots * 3345408; distinctSum != want {
		t.Errorf("STRUCTURE: distinct sizes sum to %d, expected %d (n x 3,345,408)", distinctSum, want)
	}
	t.Logf("  roundedPredicted %13d B   (sum of ceil(size/2MiB)*2MiB)", rounded)
	res := actual - rounded
	t.Logf("  residual vs actual %11d B  = %.3f quanta, over %d buffers; mod 30 = %d",
		res, float64(res)/q, len(r.dbgAllocSizes), res%30)
	if res != 0 {
		t.Logf("  NOTE residual nonzero: %.4f quanta per buffer", float64(res)/q/float64(len(r.dbgAllocSizes)))
	}
	t.Logf("  cap: inline chose %d, capSlots would choose %d (predInline %d, predCapSlots %d)",
		r.dbgSlotsInline, r.dbgSlotsCapSlots, r.dbgPredInline, r.dbgPredCapSlots)
	t.Logf("  free before first launch %d B", r.dbgFreeBeforeLaunch)
}
