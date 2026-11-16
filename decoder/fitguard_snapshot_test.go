package decoder

import (
	"strings"
	"testing"
)

// KV8-hostguard (docs/tasks/task-kv-architecture-2026-10.md): the host fit guards price the prefix-snapshot store's budget (Options.KVHostBytes), which a
// GPU-resident server holds in host memory beside what stays resident, so a load whose weights fit but whose snapshot budget does not is refused, naming
// -kv-host-bytes. Each case runs through the guard's own caller (fitCheckFor's decide, decideGIWFit), not through the arithmetic in internal/runtime/mem.

const gibSnap = int64(1) << 30

// TestFitGuard_snapshotBudgetCounts: a .gguf load on a CUDA resident takes its weights through the build and then holds only the snapshot store on
// the host. With 1 GiB of RAM available the 70% budget is 0.7 GiB: the tiny model fits it, and a 1 GiB snapshot budget does not.
func TestFitGuard_snapshotBudgetCounts(t *testing.T) {
	withFakeCUDACompiled(t)
	const gguf = "testdata/gptoss_tiny.gguf"
	defer injectHostRAMAvailable(t, gibSnap)()

	control := fitCheckFor(gguf, "int4", quantInt4, Options{Quant: "int4", Backend: "cuda"})
	if !control.known() || !control.fits() {
		t.Fatalf("VOID: the control (no snapshot budget) does not fit 1 GiB available: need %d, budget %d, known %v", control.need(), control.budget(), control.known())
	}
	if !control.deviceKV {
		t.Fatalf("VOID: the control is not a CUDA-resident load (deviceKV false), so a snapshot store is not in play")
	}

	with := fitCheckFor(gguf, "int4", quantInt4, Options{Quant: "int4", Backend: "cuda", KVHostBytes: gibSnap})
	if with.fits() {
		t.Fatalf("a 1 GiB snapshot budget on 1 GiB available fits (need %d, budget %d): the guard does not price the store", with.need(), with.budget())
	}
	_, decline := with.decide()
	if decline == nil {
		t.Fatalf("the load that does not fit once the budget counts was not refused")
	}
	if !strings.Contains(decline.Error(), "-kv-host-bytes") || !strings.Contains(decline.Error(), "snapshot") {
		t.Errorf("the refusal does not name the snapshot budget and its flag:\n%s", decline.Error())
	}

	for _, c := range []struct {
		name string
		opts Options
	}{
		{"snapshots turned off (-kv-share off)", Options{Quant: "int4", Backend: "cuda", KVHostBytes: gibSnap, KVShareOff: true}},
		{"no budget", Options{Quant: "int4", Backend: "cuda", KVHostBytes: 0}},
		{"a budget that fits", Options{Quant: "int4", Backend: "cuda", KVHostBytes: gibSnap / 4}},
		{"the CPU backend keeps no store", Options{Quant: "int4", Backend: "cpu", KVHostBytes: gibSnap}},
	} {
		f := fitCheckFor(gguf, "int4", quantInt4, c.opts)
		if _, d := f.decide(); d != nil {
			t.Errorf("%s: refused:\n%s", c.name, d.Error())
		}
	}
}

// TestDecideGIWFit_snapshotBudgetCounts is the same for a .giw load, whose guard leaves a 1 GiB margin of the available memory: 2.5 GiB available
// leaves 1.5 GiB, which a CUDA resident's priced host term (none, its KV is VRAM) fits and a 2 GiB snapshot budget does not.
func TestDecideGIWFit_snapshotBudgetCounts(t *testing.T) {
	withFakeCUDACompiled(t)
	cfg := sevenB()
	defer injectHostRAMAvailable(t, 5*gibSnap/2)()

	control := decideGIWFit(cfg, Options{Backend: "cuda"})
	if !control.known || control.err != nil {
		t.Fatalf("VOID: the control (no snapshot budget) is not admitted: known %v, err %v", control.known, control.err)
	}
	d := decideGIWFit(cfg, Options{Backend: "cuda", KVHostBytes: 2 * gibSnap})
	if d.err == nil {
		t.Fatalf("a 2 GiB snapshot budget with 1.5 GiB of budget left was admitted (need %d, budget %d)", d.need, d.budget)
	}
	if !strings.Contains(d.err.Error(), "-kv-host-bytes") {
		t.Errorf("the refusal does not name the flag that lowers the budget:\n%v", d.err)
	}
	if d.need != 2*gibSnap {
		t.Errorf("need %d, want the snapshot budget alone, %d (a CUDA resident's KV is in VRAM)", d.need, 2*gibSnap)
	}
	if d.pinCtx != 0 {
		t.Errorf("a smaller context was offered (%d): the snapshot budget does not depend on the context", d.pinCtx)
	}
	for _, c := range []struct {
		name string
		opts Options
	}{
		{"snapshots turned off", Options{Backend: "cuda", KVHostBytes: 2 * gibSnap, KVShareOff: true}},
		{"a budget that fits", Options{Backend: "cuda", KVHostBytes: gibSnap}},
	} {
		if r := decideGIWFit(cfg, c.opts); r.err != nil {
			t.Errorf("%s: refused: %v", c.name, r.err)
		}
	}
}
