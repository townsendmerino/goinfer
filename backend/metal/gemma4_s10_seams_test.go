//go:build darwin

package metal

import "os"

// GOINFER_S10_DROP re-drops S1.0's Gemma 4 fixes for a whole test run (docs/tasks/task-multimodal-support-2026-10.md,
// S1.0): "scalar" drops the dense layer scalar, "vnorm" the v_norm on non-K=V layers, "both" both. The gate reads
// each Gemma 4 parity test with and without, so the before-fix numbers are reproduced on the same binary.
func init() {
	switch os.Getenv("GOINFER_S10_DROP") {
	case "scalar":
		g4DropLayerScalarForTest = true
	case "vnorm":
		g4DropVNormForTest = true
	case "both":
		g4DropLayerScalarForTest, g4DropVNormForTest = true, true
	}
}
