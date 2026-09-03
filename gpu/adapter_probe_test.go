//go:build gpu

package gpu

import (
	"fmt"
	"testing"
)

// TestAdapterProbe is cmd/gate's WebGPU adapter-detection vehicle (audit G-09): a fast,
// checkpoint-free check for a real adapter, run as a subprocess (`go test -tags gpu ./gpu/ -run
// TestAdapterProbe -v`) exactly the way the CUDA/Metal detection paths shell out to
// nvidia-smi/uname — cmd/gate stays free of the gpu build tag and its cgo dependency. Prints one
// parseable line to stdout; never itself a correctness gate (see TestMatmulBT_matchesNaive and the
// resident-parity tests for that).
func TestAdapterProbe(t *testing.T) {
	c, err := New()
	if err != nil {
		fmt.Printf("ADAPTER_PROBE: none (%v)\n", err)
		t.Skip("no adapter")
	}
	defer c.Close()
	fmt.Printf("ADAPTER_PROBE: backend=%s software=%v\n", c.Backend(), isSoftwareAdapter(c))
}
