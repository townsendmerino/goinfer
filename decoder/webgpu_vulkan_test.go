package decoder

import "testing"

// TestResidentDenseWeightBytesFor_webgpuKeepsTheUntiedTableOnTheHost is G-VP1: WebGPU gathers token embeddings on the host and
// uploads only the LM head, so an untied model's token table is not webgpu device memory, as it is not cuda's.
func TestResidentDenseWeightBytesFor_webgpuKeepsTheUntiedTableOnTheHost(t *testing.T) {
	m, err := Load("../testdata/llama-tiny", Options{Backend: "cpu", Quant: "int4"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if m.w.LMHead.Rows() == 0 {
		t.Fatal("llama-tiny's embeddings are tied: the test needs an untied fixture")
	}
	table := wmBytes(&m.w.Embed)
	if table == 0 {
		t.Fatal("llama-tiny has no token table")
	}
	for _, backend := range []string{"webgpu", "cuda"} {
		if got, want := m.ResidentDenseWeightBytesFor(backend), m.ResidentDenseWeightBytes()-table; got != want {
			t.Errorf("%s: dense device bytes %d, want %d (the dense total less the %d-byte untied token table)", backend, got, want, table)
		}
	}
}

// TestWebGPUDeviceBuffers_sumToThePlan holds webgpuDeviceBuffers to the planner's own figures: its weights are webgpu's dense bytes
// plus the routed experts, and its caches ResidentKVBytes, so the block rounding is priced over the buffers the plan counts.
func TestWebGPUDeviceBuffers_sumToThePlan(t *testing.T) {
	for _, dir := range []string{"../testdata/llama-tiny", "../testdata/glm-tiny"} {
		m, err := Load(dir, Options{Backend: "cpu", Quant: "int4"})
		if err != nil {
			t.Fatal(err)
		}
		const ctx = 512
		slot0, slotKV := m.webgpuDeviceBuffers(ctx, false, false)
		var all, kv int64
		for _, n := range slot0[:len(slot0)-1] {
			all += n
		}
		for _, n := range slotKV {
			kv += n
		}
		weights := m.ResidentDenseWeightBytesFor("webgpu") + m.ResidentWeightBytesPaged(0) - m.ResidentDenseWeightBytes()
		if all-kv != weights {
			t.Errorf("%s: the list's weights %d, the plan's %d", dir, all-kv, weights)
		}
		if want := m.ResidentKVBytes("webgpu", ctx, false, false); kv != want {
			t.Errorf("%s: the list's caches %d, ResidentKVBytes %d", dir, kv, want)
		}
		m.Close()
	}
}
