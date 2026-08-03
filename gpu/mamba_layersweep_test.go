//go:build gpu

package gpu

import (
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// Phase-A localization: the mamba path is exonerated (in_proj + mixer correct). The 93.6%→66%
// gap is the resident attn/MoE (W8A8 GEMVs) vs D3's staged matmuls. This sweeps
// GOINFER_SSM_STOP_LAYER — rebuilding the resident plan to emit logits from the hidden after
// each layer L — and compares the resident's token-0 logits to the f32 CPU reference per L. The
// first L where cosine drops names the buggy layer; a gradual drop from every layer ⇒ MoE
// (every layer), a jump at 5/15/25/35 ⇒ attention.
func TestMambaResidentLayerSweep(t *testing.T) {
	requireHeavyModel(t)
	if os.Getenv("GOINFER_SSM_QUALITY") == "" {
		t.Skip("resident layer sweep — needs granite model; set GOINFER_SSM_QUALITY=1")
	}
	if _, err := New(); err != nil {
		t.Skipf("no webgpu: %v", err)
	}
	path := os.ExpandEnv("$HOME/models/granite/granite-4.0-h-tiny-Q8_0.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no granite model: %v", err)
	}
	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	ids, _ := tk.Encode("The capital of France is Paris.", true)
	tok := ids[0]

	// f32 CPU reference (per-layer via stopLayer, cheap — re-read per forward).
	os.Unsetenv("GOINFER_SSM_RESIDENT")
	mc, err := decoder.Load(path, decoder.Options{Backend: "cpu"})
	if err != nil {
		t.Fatalf("load cpu: %v", err)
	}
	_, nLayers, _, _, _, _, _ := mc.Dims()
	cpuLogits := make([][]float32, nLayers)
	for L := 0; L < nLayers; L++ {
		os.Setenv("GOINFER_SSM_STOP_LAYER", itoa(L))
		cache := mc.NewCache(8)
		lg, e := mc.ForwardForTest(tok, cache)
		if e != nil {
			t.Fatal(e)
		}
		cpuLogits[L] = append([]float32(nil), lg...)
	}
	os.Unsetenv("GOINFER_SSM_STOP_LAYER")
	mc.Close()

	// Resident: rebuild the plan per stopLayer (shares loaded weights), token-0 from clean state.
	os.Setenv("GOINFER_SSM_RESIDENT", "1")
	defer os.Unsetenv("GOINFER_SSM_RESIDENT")
	m, err := decoder.Load(path, decoder.Options{Backend: "webgpu", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load resident: %v", err)
	}
	defer m.Close()
	rd := m.ResidentForwardForTest().(*residentDecoder)
	emb := m.EmbedResidentForTest(tok)

	prev := 1.0
	for L := 0; L < nLayers; L++ {
		os.Setenv("GOINFER_SSM_STOP_LAYER", itoa(L))
		runner, e := rd.newRunner()
		if e != nil {
			t.Fatal(e)
		}
		rd.Reset()
		lg, e := runner.Run(emb, 0)
		runner.Release()
		if e != nil {
			t.Fatal(e)
		}
		cos, _ := cosSim(cpuLogits[L], lg)
		mark := ""
		kind := "mamba"
		if !m.GraniteMambaLayer(L) {
			kind = "ATTN"
		}
		if cos < prev-0.01 {
			mark = "  <-- drop"
		}
		prev = cos
		t.Logf("  after layer %2d (%-5s): resident-vs-f32 cosine=%.6f%s", L, kind, cos, mark)
	}
	os.Unsetenv("GOINFER_SSM_STOP_LAYER")
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	b := []byte{}
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}
