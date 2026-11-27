package decoder

import (
	"testing"

	"github.com/townsendmerino/goinfer/internal/runtime/memprice"
)

// TestCUDADerivedTables_matchTheDeviceRead holds the dense line's derived tables (CUDADerivedTableBytes) to what the CUDA build leased
// beyond the tensors on the checkpoints the memory gate builds resident: Flint's six layers (an attention RoPE table each and a zero
// router bias on its MoE layers) and a Mamba-2 hybrid's zero RoPE tables (the 320 B and 64 B nobara read on TestMemGate_cuda).
func TestCUDADerivedTables_matchTheDeviceRead(t *testing.T) {
	for _, c := range []struct {
		dir  string
		want int64
	}{{"flint-tiny", 320}, {"nemotron-tiny", 64}} {
		m, err := Load("../testdata/"+c.dir, Options{Backend: "cpu", Quant: "int4"})
		if err != nil {
			t.Fatalf("%s: %v", c.dir, err)
		}
		if got := memprice.CUDADerivedTableBytes(m.w, m.Program()); got != c.want {
			t.Errorf("%s: CUDADerivedTableBytes %d, the device read leased %d over the tensors", c.dir, got, c.want)
		}
		m.Close()
	}
}
