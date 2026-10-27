//go:build gpu && goinfer_testhooks

package gpu

import (
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestP0a_webgpuBuildFollowsResolveKV is P0A-G1's WebGPU half (docs/tasks/task-kv-precision-everywhere-2026-10.md, P0a′ gates):
// with the resolver made to deliver f16, then int8, for an auto request, the resident runner takes that precision.
func TestP0a_webgpuBuildFollowsResolveKV(t *testing.T) {
	probe, err := New()
	if err != nil {
		t.Skipf("no GPU adapter: %v", err)
	}
	probe.Close()
	build := func() (f16, i8 bool) {
		m, err := decoder.Load("../testdata/llama-tiny", decoder.Options{Backend: "webgpu", Quant: "int8int8"})
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		if !m.ResidentActive() {
			t.Skipf("not GPU-resident: %s", m.ResidentDecline())
		}
		rd, ok := m.ResidentForwardForTest().(*residentDecoder)
		if !ok {
			t.Skipf("resident forward is not the gpu DecodeRunner (%T)", m.ResidentForwardForTest())
		}
		return rd.rm.KvF16, rd.rm.KvI8
	}
	if f16, i8 := build(); f16 || i8 {
		t.Fatalf("unhooked, an auto request built f16 %v, int8 %v; want f32", f16, i8)
	}
	for _, want := range []string{"f16", "i8"} {
		restore := decoder.SetResolveKVHookForTest(func(backend string, _ decoder.Options) (string, string, bool) {
			return want, "hooked", backend == "webgpu"
		})
		f16, i8 := build()
		restore()
		if (want == "f16") != f16 || (want == "i8") != i8 {
			t.Errorf("the resolver delivered %s and the runner built f16 %v, int8 %v: it read the raw request", want, f16, i8)
		}
	}
}
