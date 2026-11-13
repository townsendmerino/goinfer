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
		want := m.ResidentDenseWeightBytes() - table
		if backend == "cuda" {
			want += m.cudaDerivedTableBytes() // CUDA's dense line also holds the RoPE and router-bias tables its build derives
		}
		if got := m.ResidentDenseWeightBytesFor(backend); got != want {
			t.Errorf("%s: dense device bytes %d, want %d (the dense total less the %d-byte untied token table)", backend, got, want, table)
		}
	}
}

// TestPlan_webgpuPricesVulkansBlocks: on a Vulkan build the webgpu plan's device total at a context is M1′'s, gpu-allocator's blocks
// over the build's buffer list plus the device and the kept host block, beside the backend's own slack and the dense line's vectors
// and biases, on a fixture with recurrent
// state, one with MLA and one with KV-shared layers as well as a plain one. On darwin it is the buffers' sum, as before.
func TestPlan_webgpuPricesVulkansBlocks(t *testing.T) {
	for _, dir := range []string{"../testdata/llama-tiny", "../testdata/qwen35-tiny", "../testdata/deepseek-tiny", "../testdata/gemma4-moe-kv-tiny"} {
		m, err := Load(dir, Options{Backend: "cpu", Quant: "int4"})
		if err != nil {
			t.Fatalf("%s: %v", dir, err)
		}
		for _, ctx := range []int{256, 4096} {
			l := m.deviceLoad("webgpu", false, false, true)
			got := l.DenseBytes + l.ExpertBytesFull + l.FixedSlack + l.KV(ctx) + l.StateBytes
			slot0, _, _ := m.webgpuDeviceBuffers(ctx, false, false)
			// The dense line also holds the norm vectors, the expert biases and the derived tables, which the list does not hold: they sit beside the blocks.
			want := webgpuVkBlocks(slot0) + webgpuVkFixed + m.planSlack("webgpu") + m.denseVectorBytes() + m.expertBiasBytes() + m.webgpuDerivedTableBytes()
			if !webgpuVulkan {
				want = m.webgpuDenseLine() + l.ExpertBytesFull + m.planSlack("webgpu") + m.ResidentKVBytes("webgpu", ctx, false, false) + l.StateBytes
			}
			if got != want {
				t.Errorf("%s at %d: the plan prices %d device bytes, M1′ over the build's list %d", dir, ctx, got, want)
			}
		}
		m.Close()
	}
}
