package mem

import (
	"strings"
	"testing"
)

// TestSolveHost_snapshotStore: the prefix-snapshot store's budget (KV8-hostguard) joins the host need beside what stays on the host once the server is up,
// binds only when it exceeds the build's peak, names -kv-host-bytes when it does, and is not something a smaller context can pay for on a device-resident load.
func TestSolveHost_snapshotStore(t *testing.T) {
	const g = int64(1) << 30
	p := DefaultPolicy()
	m := HostMachine(10 * g)
	base := HostLoad{Name: "m", Quant: "int4", WeightBytes: 2 * g, DeviceKV: true, Ctx: 8192, MaxCtx: 32768, KV: func(int) int64 { return 0 }}

	if f := SolveHost(base, m, p); f.Decline != nil {
		t.Fatalf("VOID: the control with no store is refused: %s", f.Decline.Msg)
	}
	small := base
	small.SnapshotBytes = g
	if c := NewHostCheck(small, m, p); c.Need() != 2*g || c.snapshotBinds() {
		t.Errorf("a 1 GiB store beside a 2 GiB build peak: need %d, binds %v; want the build's 2 GiB", c.Need(), c.snapshotBinds())
	}
	big := base
	big.SnapshotBytes = 8 * g
	c := NewHostCheck(big, m, p)
	if c.Need() != 8*g || !c.snapshotBinds() || c.Fits() {
		t.Fatalf("an 8 GiB store on a device-resident load: need %d, binds %v, fits %v; want 8 GiB, true, false", c.Need(), c.snapshotBinds(), c.Fits())
	}
	f := SolveHost(big, m, p)
	if f.Decline == nil || f.PinCtx != 0 {
		t.Fatalf("the 8 GiB store: decline %v, pinned context %d; want a refusal and no smaller context", f.Decline, f.PinCtx)
	}
	if !strings.Contains(f.Decline.Msg, "-kv-host-bytes") || strings.Contains(f.Decline.Msg, "-stream-weights") {
		t.Errorf("the refusal should name -kv-host-bytes and not offer -stream-weights:\n%s", f.Decline.Msg)
	}

	// A host-resident load holds weights, KV and the store together.
	host := HostLoad{Name: "m", Quant: "int4", WeightBytes: 3 * g, KVBytes: g, SnapshotBytes: 4 * g, Ctx: 8192, MaxCtx: 8192}
	if c := NewHostCheck(host, m, p); c.Need() != 8*g {
		t.Errorf("host-resident need %d, want weights + KV + store = 8 GiB", c.Need())
	}
}
