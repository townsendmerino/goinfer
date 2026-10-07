//go:build cuda && goinfer_testhooks

package cuda

import "os"

// GOINFER_S10_DROP=vnorm (or both) re-drops S1.0's Gemma 4 v_norm on the non-K=V layers for a whole test run
// (docs/tasks/task-multimodal-support-2026-10.md, S1.0), so each Gemma 4 parity test can be read with and without the
// fix on the same binary. "scalar" is accepted and ignored: CUDA's dense layer scalar was already right.
func init() {
	switch os.Getenv("GOINFER_S10_DROP") {
	case "vnorm", "both":
		g4DropVNormForTest = true
	}
}
