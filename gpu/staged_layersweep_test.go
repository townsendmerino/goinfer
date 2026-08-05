//go:build gpu && goinfer_testhooks

package gpu

import (
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// Companion to TestMambaResidentLayerSweep: the SAME token-0 per-layer sweep but on the D3
// staged path (int8 attn/MoE via the tiled GEMM, int8 mamba via ssmQ8CPU on the CPU). Comparing
// the two declines isolates whether the resident's larger per-forward error is the resident GEMV
// path itself (staged decline gentler ⇒ yes) or shared int8 (declines match ⇒ cross-token).
func TestStagedLayerSweep(t *testing.T) {
	requireHeavyModel(t)
	if os.Getenv("GOINFER_SSM_QUALITY") == "" {
		t.Skip("staged layer sweep — needs granite model; set GOINFER_SSM_QUALITY=1")
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

	os.Unsetenv("GOINFER_SSM_RESIDENT")
	decoder.SetSSMQ8CPU(false)
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

	// Staged int8 (D3): webgpu backend, NOT resident; int8 mamba via ssmQ8CPU.
	decoder.SetSSMQ8CPU(true)
	defer decoder.SetSSMQ8CPU(false)
	ms, err := decoder.Load(path, decoder.Options{Backend: "webgpu", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load staged: %v", err)
	}
	defer ms.Close()
	if ms.ResidentActive() {
		t.Fatal("expected staged, got resident")
	}
	for L := 0; L < nLayers; L++ {
		os.Setenv("GOINFER_SSM_STOP_LAYER", itoa(L))
		cache := ms.NewCache(8)
		lg, e := ms.ForwardForTest(tok, cache)
		if e != nil {
			t.Fatal(e)
		}
		cos, _ := cosSim(cpuLogits[L], lg)
		kind := "mamba"
		if !ms.GraniteMambaLayer(L) {
			kind = "ATTN"
		}
		t.Logf("  after layer %2d (%-5s): STAGED-vs-f32 cosine=%.6f", L, kind, cos)
	}
	os.Unsetenv("GOINFER_SSM_STOP_LAYER")
}
