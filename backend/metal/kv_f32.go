//go:build darwin

package metal

import "strings"

// kvF32Kernels is attention_f32, the decode attention over an f32 KV cache (--kv f32), derived at init from attention's own
// source with the cache's element type widened: every "half" in attention's text names that type (half, half4) or a comment about
// it. The two kernels therefore share one reduction order, the attention sink and the V-read schedule, so the f32 kernel cannot
// fall behind the f16 one (P3-G1, docs/tasks/task-kv-precision-everywhere-2026-10.md). Appended to allKernels when the resident's
// library is compiled.
var kvF32Kernels = deriveF32Attention()

func deriveF32Attention() string {
	k := edit(extractKernel(allKernels, "attention"), "kernel void attention(", "kernel void attention_f32(")
	return "\n// ---- attention_f32: attention over an f32 KV cache, derived from attention (kv_f32.go) ----\n" +
		strings.ReplaceAll(k, "half", "float") + "\n"
}
