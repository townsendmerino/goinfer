//go:build cuda && goinfer_testhooks

package cuda

import (
	"regexp"
	"sort"
	"strings"
	"testing"
	"unsafe"

	gc "github.com/eitamring/gocudrv/cuda"
)

// pipeShim mirrors aikit's `type Pipeline struct{ f *gc.Function }` so a test can reach the driver
// handle. aikit exposes no accessor and the field is unexported in another module; the layout is a
// single pointer, so the cast is well-defined for as long as that holds. If aikit ever adds a field
// this stops compiling rather than reading garbage, because the shim is a distinct type.
type pipeShim struct{ f *gc.Function }

func rawFunc(p Pipeline) *gc.Function { return (*pipeShim)(unsafe.Pointer(&p)).f }

var ptxEntry = regexp.MustCompile(`\.visible\s+\.entry\s+([A-Za-z_][A-Za-z0-9_$]*)`)

// ptxModules is every PTX blob goinfer embeds. Enumerated here rather than sampled: the whole point
// of A9's correction is that measuring two kernels and generalising is the sibling-drift shape, and
// a local-memory reservation is a per-kernel property that nothing else in the tree reports.
func ptxModules() []struct {
	name string
	ptx  []byte
} {
	return []struct {
		name string
		ptx  []byte
	}{
		{"glue.ptx", gluePTX},
		{"moe.ptx", moePTX},
		{"router_f32.ptx", routerF32PTX},
		{"argmax.ptx", argmaxPTX},
		{"gemv_fwd.ptx", gemvFwdPTX},
		{"gemv_w4a8_batched.ptx", gemvBatchedPTX},
		{"gemv_w8a8_batched.ptx", gemvW8BatchedPTX},
		{"prefill_batched.ptx", prefillBatchedPTX},
		{"decode_splitkv.ptx", decodeSplitKVPTX},
		{"gemv_w4a8_staged.ptx", gemvStagedPTX},
		{"gemv_w4a8_rn.ptx", gemvRNPTX},
		{"fused_qkv.ptx", fusedQKVPTX},
		{"deltanet.ptx", deltaNetPTX},
	}
}

// TestKernelLocalMemoryCensus reports CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES for every entry point in
// every embedded module, and checks the backing-store multiplier against a measured reservation.
//
// A9 established that moe_route's first launch reserves 138,412,032 B, and did so by measuring two
// kernels. Two kernels is a sample. Local memory per thread is a per-kernel compile-time property,
// so any kernel with per-thread arrays carries its own deferred reservation, and nothing in the tree
// reported them. This is the loop.
//
// It also settles the multiplier. "MOE_MAX_E 256 -> 512 doubled the cost from ~66 to 132 MiB"
// assumes the backing store is linear in per-thread bytes with a constant occupancy factor. That is
// an assumption about the driver, not an observation, and it is checked here against
// multiProcessorCount x maxThreadsPerMultiProcessor rather than asserted.
func TestKernelLocalMemoryCensus(t *testing.T) {
	dev, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	raw := dev.Context().Device()
	sms, err := raw.Attribute(gc.DeviceAttributeMultiprocessorCount)
	if err != nil {
		t.Fatalf("MultiprocessorCount: %v", err)
	}
	maxT, err := raw.Attribute(gc.DeviceAttributeMaxThreadsPerMultiprocessor)
	if err != nil {
		t.Fatalf("MaxThreadsPerMultiprocessor: %v", err)
	}
	t.Logf("device: %d SMs x %d max threads/SM = %d resident threads", sms, maxT, sms*maxT)

	type row struct {
		mod, fn   string
		local     int
		predicted int64
	}
	var rows []row
	var total int64
	for _, m := range ptxModules() {
		names := map[string]bool{}
		for _, mm := range ptxEntry.FindAllStringSubmatch(string(m.ptx), -1) {
			names[mm[1]] = true
		}
		if len(names) == 0 {
			t.Errorf("%s: no .visible .entry found — the census cannot be complete if a module "+
				"contributes no entries; either the regexp is wrong or the blob is empty", m.name)
			continue
		}
		mod, e := dev.CompileLibrary(m.ptx)
		if e != nil {
			t.Errorf("%s: CompileLibrary: %v", m.name, e)
			continue
		}
		sorted := make([]string, 0, len(names))
		for n := range names {
			sorted = append(sorted, n)
		}
		sort.Strings(sorted)
		for _, n := range sorted {
			p, e := dev.NewComputePipeline(mod, n)
			if e != nil {
				t.Errorf("%s/%s: NewComputePipeline: %v", m.name, n, e)
				continue
			}
			local, e := rawFunc(p).Attribute(gc.FuncAttrLocalSizeBytes)
			if e != nil {
				t.Errorf("%s/%s: LOCAL_SIZE_BYTES: %v", m.name, n, e)
				continue
			}
			pred := int64(local) * int64(sms) * int64(maxT)
			rows = append(rows, row{m.name, n, local, pred})
			total += pred
		}
	}

	sort.Slice(rows, func(i, j int) bool { return rows[i].local > rows[j].local })
	t.Logf("%-22s %-28s %10s %16s", "module", "kernel", "local/thr", "SMs*maxT*local")
	nonZero := 0
	for _, r := range rows {
		if r.local == 0 {
			continue
		}
		nonZero++
		t.Logf("%-22s %-28s %10d %16d  (%.1f MiB)", r.mod, r.fn, r.local, r.predicted,
			float64(r.predicted)/(1<<20))
	}
	t.Logf("%d entry points across %d modules; %d declare per-thread local memory",
		len(rows), len(ptxModules()), nonZero)
	// DENOMINATOR, stated every run. ptxModules() is a HAND-MAINTAINED list: a new .ptx blob that
	// nobody adds to it is simply absent, and the census stays green while covering less. Naming the
	// modules examined makes that omission visible in the log instead of invisible in the count.
	names := make([]string, 0, len(ptxModules()))
	for _, m := range ptxModules() {
		names = append(names, m.name)
	}
	t.Logf("EXAMINED: %d module(s) — %s. Any shipped PTX absent from ptxModules() is NOT in this "+
		"census; the list is hand-maintained, so this line is the audit of it.",
		len(names), strings.Join(names, " "))
	// AUDITED 2026-08-13 against the embeds: 22 .ptx blobs are go:embed-ed, 12 are here, and all 10
	// of the difference (gemv_w4a8{,_coal,_coal2,_coal3,_coal4,_fast,_v4}.ptx, gemv_w8a8.ptx,
	// addone.ptx, megakernel.ptx) are referenced ONLY from _test.go — variant-comparison blobs, no
	// production path. So the denominator is right at 12 today. It is NOT self-maintaining: a new
	// production PTX added without a ptxModules() entry would drop straight out of this census while
	// the count stayed green. Re-run the audit (`grep go:embed`, then which vars non-test files use)
	// when a module is added.
	t.Logf("SUM of per-kernel backing stores at full occupancy: %d B (%.1f MiB) — an UPPER BOUND on "+
		"the deferred cost, not a prediction: the driver may share or reuse a backing store across "+
		"kernels, and nothing here shows that it allocates them all simultaneously",
		total, float64(total)/(1<<20))

	// ---- PINNED: the per-kernel local footprint of the shipped PTX ----
	//
	// Item 6: a gate that asserts only "a reservation exists" lets a future MOE_MAX_E change double a
	// hidden 132 MiB cost while still passing — the exercised-but-never-triggered shape, inside the
	// gate written for this finding. So the BYTE FIGURES are pinned.
	//
	// This is the enumerated form, not a sample: every entry point with non-zero local memory is
	// listed, so a regeneration that gives any kernel per-thread scratch trips the gate even if
	// moe_route is untouched. LOCAL_SIZE_BYTES is read from the compiled shipped artifact, so it
	// tracks the PTX rather than the source constant — MOE_MAX_E cannot change without moving it.
	pinned := map[string]int{
		"moe_route":       4416, // two float[MOE_MAX_E] at MOE_MAX_E=512, plus the group scratch
		"rope_kv":         32,
		"rope_kv_batched": 32,
	}
	got := map[string]int{}
	for _, r := range rows {
		if r.local != 0 {
			got[r.fn] = r.local
		}
	}
	for fn, want := range pinned {
		if g, ok := got[fn]; !ok {
			t.Errorf("PINNED: %s no longer declares local memory (was %d B/thread) — its deferred "+
				"first-launch reservation has changed and the figures in docs/QUEUE.md A9 are stale", fn, want)
		} else if g != want {
			t.Errorf("PINNED: %s local memory %d B/thread, pinned at %d. A change here moves a "+
				"deferred first-launch reservation that the expert-cache sizing does not account "+
				"for; re-measure the demand threshold before updating this number", fn, g, want)
		}
	}
	for fn, g := range got {
		if _, ok := pinned[fn]; !ok {
			t.Errorf("PINNED: %s declares %d B/thread of local memory and is not in the pinned set — "+
				"a NEW deferred reservation exists that nothing has measured", fn, g)
		}
	}

	// ---- A9-FIX's precondition ----
	//
	// BuildResident forces moe_route BY NAME before sizing the expert cache, which is only sound
	// because the local-memory backing store is shared and sized by the LARGEST kernel (measured:
	// launching the whole census gives a threshold and residual identical to moe_route alone, to the
	// byte). Forcing the maximum forces the pool for everything.
	//
	// That moe_route IS the maximum is checked here rather than assumed. Without this, a new kernel
	// with deeper per-thread scratch would make the warm-up force the wrong pool, allocSlots would
	// again size against memory about to be taken, and nothing would say so — naming one member of a
	// set is the sibling-drift shape, and this is what keeps the naming honest.
	maxFn, maxLocal := "", -1
	for _, r := range rows {
		if r.local > maxLocal {
			maxFn, maxLocal = r.fn, r.local
		}
	}
	if maxFn != "moe_route" {
		t.Errorf("%s declares %d B/thread, more than moe_route's %d — cuda/backend.go forces "+
			"moe_route before allocSlots to pay the deferred local-memory reservation, and that is "+
			"sound only while moe_route is the maximum. Force %s there instead, or force both, and "+
			"re-measure the demand threshold", maxFn, maxLocal, pinned["moe_route"], maxFn)
	}

	// ---- the multiplier, checked rather than assumed ----
	//
	// moe_route's reservation was MEASURED at 138,412,032 B (RTX 2070 SUPER, driver 595.58.03,
	// 2026-08-12, both cuMemGetInfo and nvidia-smi agreeing). If the naive form
	// local x SMs x maxThreadsPerSM reproduces it, the "256 -> 512 halves it" derivation is sound.
	// If it does not, the multiplier carries something else and the derivation was doing more work
	// than it looked.
	const measuredMoERoute = 138412032
	for _, r := range rows {
		if r.fn != "moe_route" {
			continue
		}
		t.Logf("moe_route: local %d B/thread, naive form predicts %d B, MEASURED %d B (ratio %.4f)",
			r.local, r.predicted, int64(measuredMoERoute), float64(measuredMoERoute)/float64(r.predicted))
		if r.predicted != measuredMoERoute {
			t.Logf("  => the naive multiplier does NOT reproduce the measurement. The occupancy "+
				"factor is not SMs x maxThreadsPerSM (%d x %d); proportionality in local-bytes is "+
				"therefore UNVERIFIED, and any 'halving MOE_MAX_E halves the cost' claim must be "+
				"measured by building at 256 rather than derived from this form.", sms, maxT)
		} else {
			t.Logf("  => the naive multiplier reproduces the measurement exactly, so the backing " +
				"store is linear in per-thread bytes at fixed occupancy and the halving derivation " +
				"is sound.")
		}
	}
}
