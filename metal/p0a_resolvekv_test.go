//go:build darwin && goinfer_testhooks

package metal

import (
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestP0a_metalBuildFollowsResolveKV is P0A-G1's Metal half (docs/tasks/task-kv-precision-everywhere-2026-10.md, P0a′ gates):
// with the resolver made to deliver int8 for an auto request, the resident build allocates the int8 cache; unhooked it does not.
func TestP0a_metalBuildFollowsResolveKV(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	build := func() bool {
		m, err := decoder.Load("../testdata/llama-tiny", decoder.Options{Backend: "metal", Quant: "int4"})
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		a, ok := m.ResidentForwardForTest().(*metalResident)
		if !ok {
			t.Fatalf("not Metal-resident: %s", m.ResidentDecline())
		}
		return a.r.kvI8
	}
	if build() {
		t.Fatal("unhooked, an auto request built an int8 cache")
	}
	restore := decoder.SetResolveKVHookForTest(func(backend string, _ decoder.Options) (string, string, bool) {
		return "i8", "hooked", backend == "metal"
	})
	defer restore()
	if !build() {
		t.Error("the resolver delivered int8 for Metal and the build allocated another precision: it read the raw request")
	}
}
