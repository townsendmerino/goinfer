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
