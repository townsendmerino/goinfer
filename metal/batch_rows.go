//go:build darwin

package metal

import (
	"fmt"
	"strings"
)

// MC3 S3 (docs/tasks/task-concurrency-2026-09.md): multi-row forms of the per-row kernels a batched step runs once per
// sequence — rmsnorm_quant, quant_vec, swiglu_quant, rope2. Measured (TestMC3StepBreakdown, 2026-09-27): those small
// dispatches cost ~1 ms per sequence per step (4.2 of a B = 4 step's 22.6 ms at depth 128, 8.1 of B = 8's 30.1), each
// running a single threadgroup, one sequence after another. One dispatch over all rows runs them side by side.
//
// Their BODIES are production's, byte for byte: each variant is derived from allKernels' own source at init, by
// renaming the pointer / per-row parameters in the signature and adding a prologue that points them at the row this
// threadgroup (or thread) serves. Nothing in a body is retyped, so a variant cannot drift from its kernel — these
// kernels round differently when their code shape changes (see rmsnorm_quant's own notes), which is why they are not
// hand-copied — and a production signature edit panics here at init instead of silently diverging. They compile into
// the resident's main library (same fast-math setting as the originals).

// mc3RowsKernels is the derived source, appended to allKernels when the resident's library is compiled.
var mc3RowsKernels = deriveRowsKernels()

// extractKernel returns the full text of `kernel void <name>(...) { ... }` from src.
func extractKernel(src, name string) string {
	start := strings.Index(src, "kernel void "+name+"(")
	if start < 0 {
		panic(fmt.Sprintf("metal: batch_rows: kernel %s not found in allKernels", name))
	}
	open := strings.Index(src[start:], "{")
	if open < 0 {
		panic(fmt.Sprintf("metal: batch_rows: kernel %s has no body", name))
	}
	depth := 0
	for i := start + open; i < len(src); i++ {
		switch src[i] {
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return src[start : i+1]
			}
		}
	}
	panic(fmt.Sprintf("metal: batch_rows: kernel %s body not closed", name))
}

// edit applies exact replacements to a kernel's text, each of which must match exactly once.
func edit(k string, pairs ...string) string {
	for i := 0; i+1 < len(pairs); i += 2 {
		if n := strings.Count(k, pairs[i]); n != 1 {
			panic(fmt.Sprintf("metal: batch_rows: %q matches %d times (want 1) — production's signature changed", pairs[i], n))
		}
		k = strings.Replace(k, pairs[i], pairs[i+1], 1)
	}
	return k
}

const rowsTail = "uint tid[[thread_position_in_threadgroup]], uint tgs[[threads_per_threadgroup]]) {"

func deriveRowsKernels() string {
	var b strings.Builder
	b.WriteString("\n// ---- MC3 S3: multi-row forms derived from the kernels above (batch_rows.go); bodies unchanged ----\n")

	// rmsnorm_quant: one threadgroup per row; x and aq advance H, asc one.
	b.WriteString(edit(extractKernel(allKernels, "rmsnorm_quant"),
		"kernel void rmsnorm_quant(device const float* x[[buffer(0)]]",
		"kernel void mc3_rmsnorm_quant_rows(device const float* x_rows[[buffer(0)]]",
		"device char* aq[[buffer(2)]], device float* asc[[buffer(3)]]",
		"device char* aq_rows[[buffer(2)]], device float* asc_rows[[buffer(3)]]",
		rowsTail,
		"uint tid[[thread_position_in_threadgroup]], uint tgs[[threads_per_threadgroup]],\n    uint mc3_row[[threadgroup_position_in_grid]]) {\n"+
			"    device const float* x = x_rows + mc3_row*H; device char* aq = aq_rows + mc3_row*H; device float* asc = asc_rows + mc3_row;"))
	b.WriteString("\n")

	// quant_vec: one threadgroup per row; x and aq advance H (the vector length), asc one.
	b.WriteString(edit(extractKernel(allKernels, "quant_vec"),
		"kernel void quant_vec(device const float* x[[buffer(0)]], device char* aq[[buffer(1)]]",
		"kernel void mc3_quant_vec_rows(device const float* x_rows[[buffer(0)]], device char* aq_rows[[buffer(1)]]",
		"device float* asc[[buffer(2)]]",
		"device float* asc_rows[[buffer(2)]]",
		rowsTail,
		"uint tid[[thread_position_in_threadgroup]], uint tgs[[threads_per_threadgroup]],\n    uint mc3_row[[threadgroup_position_in_grid]]) {\n"+
			"    device const float* x = x_rows + mc3_row*H; device char* aq = aq_rows + mc3_row*H; device float* asc = asc_rows + mc3_row;"))
	b.WriteString("\n")

	// swiglu_quant: one threadgroup per row; a row of the gate|up buffer is [gate I | up I], so g advances 2I and u
	// is g + I (production passes the two halves of one row as g and u); dq advances I, ds one.
	b.WriteString(edit(extractKernel(allKernels, "swiglu_quant"),
		"kernel void swiglu_quant(device const float* g[[buffer(0)]], device const float* u[[buffer(1)]]",
		"kernel void mc3_swiglu_quant_rows(device const float* g_rows[[buffer(0)]], device const float* u_unused[[buffer(1)]]",
		"device char* dq[[buffer(2)]], device float* ds[[buffer(3)]]",
		"device char* dq_rows[[buffer(2)]], device float* ds_rows[[buffer(3)]]",
		rowsTail,
		"uint tid[[thread_position_in_threadgroup]], uint tgs[[threads_per_threadgroup]],\n    uint mc3_row[[threadgroup_position_in_grid]]) {\n"+
			"    device const float* g = g_rows + mc3_row*2u*I; device const float* u = g + I; device char* dq = dq_rows + mc3_row*I; device float* ds = ds_rows + mc3_row;"))
	b.WriteString("\n")

	// rope2: one thread per rotated pair, M rows of it; each row has its own position and Q temperature scale, and a
	// row of the fused qkv buffer is rowStride floats.
	b.WriteString(edit(extractKernel(allKernels, "rope2"),
		"kernel void rope2(device float* x[[buffer(0)]]",
		"kernel void mc3_rope2_rows(device float* x_rows[[buffer(0)]]",
		"constant uint& pos[[buffer(3)]]",
		"device const uint* pos_rows[[buffer(3)]]",
		"constant float& qTempScale[[buffer(9)]], uint gid[[thread_position_in_grid]]) {",
		"device const float* qts_rows[[buffer(9)]], constant uint& rowStride[[buffer(10)]],\n    constant uint& mc3_M[[buffer(11)]], uint gid_rows[[thread_position_in_grid]]) {\n"+
			"    uint mc3_row = gid_rows / (qTotal + kTotal); if (mc3_row >= mc3_M) return;\n"+
			"    uint gid = gid_rows % (qTotal + kTotal); device float* x = x_rows + mc3_row*rowStride;\n"+
			"    uint pos = pos_rows[mc3_row]; float qTempScale = qts_rows[mc3_row];"))
	b.WriteString("\n")

	// kv_store: B rows of kvDim threads; each row writes its own K/V (rows of the fused qkv buffer, rowStride floats
	// apart) into its OWN slot at its own position. A layer's slots are one allocation (kvContig), so a row's slot is
	// kc_all + slotOff[row] elements.
	b.WriteString(edit(extractKernel(allKernels, "kv_store"),
		"kernel void kv_store(device const float* k[[buffer(0)]], device const float* v[[buffer(1)]]",
		"kernel void mc3_kv_store_rows(device const float* k_rows[[buffer(0)]], device const float* v_rows[[buffer(1)]]",
		"device half* kc[[buffer(2)]], device half* vc[[buffer(3)]]",
		"device half* kc_all[[buffer(2)]], device half* vc_all[[buffer(3)]]",
		"constant uint& pos[[buffer(5)]], uint i[[thread_position_in_grid]]) {",
		"device const uint* pos_rows[[buffer(5)]], constant uint& rowStride[[buffer(6)]], constant uint& mc3_M[[buffer(7)]],\n"+
			"    device const uint* slotOff[[buffer(8)]], uint i_rows[[thread_position_in_grid]]) {\n"+
			"    uint mc3_row = i_rows / kvDim; if (mc3_row >= mc3_M) return; uint i = i_rows % kvDim;\n"+
			"    device const float* k = k_rows + mc3_row*rowStride; device const float* v = v_rows + mc3_row*rowStride;\n"+
			"    uint pos = pos_rows[mc3_row]; device half* kc = kc_all + slotOff[mc3_row]; device half* vc = vc_all + slotOff[mc3_row];"))
	b.WriteString("\n")

	// attention (the shipped per-head kernel, below attnFADepthFloor): M rows of nH threadgroups, row j serving batch row
	// rowmap[j] (the rows whose attention plan is this kernel; rows at attention_fa depth keep their own dispatch); each
	// reads its own q (a row of the fused qkv buffer: nH*hd + 2*nKV*hd floats), its own slot's K/V at its own key
	// count, and writes its own row of ctx (nH*hd floats). 15 buffers: aikit binds at most 16.
	b.WriteString(edit(extractKernel(allKernels, "attention"),
		"kernel void attention(device const float* q[[buffer(0)]], device const half* kc[[buffer(1)]],\n    device const half* vc[[buffer(2)]], device float* out[[buffer(3)]]",
		"kernel void mc3_attention_rows(device const float* q_rows[[buffer(0)]], device const half* kc_all[[buffer(1)]],\n    device const half* vc_all[[buffer(2)]], device float* out_rows[[buffer(3)]]",
		"constant uint& nKeys[[buffer(7)]]",
		"device const uint* nKeys_rows[[buffer(7)]]",
		"device const float* sinks[[buffer(10)]], constant uint& hasSink[[buffer(11)]],\n    uint qh[[threadgroup_position_in_grid]],",
		"device const float* sinks[[buffer(10)]], constant uint& hasSink[[buffer(11)]],\n"+
			"    constant uint& mc3_M[[buffer(12)]], device const uint* rowmap[[buffer(13)]], device const uint* slotOff[[buffer(14)]],\n"+
			"    uint qh_rows[[threadgroup_position_in_grid]],",
		"uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]]) {",
		"uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]]) {\n"+
			"    uint mc3_j = qh_rows / nH; if (mc3_j >= mc3_M) return; uint mc3_row = rowmap[mc3_j]; uint qh = qh_rows % nH;\n"+
			"    device const float* q = q_rows + mc3_row*(nH*hd + 2u*nKV*hd); device float* out = out_rows + mc3_row*(nH*hd);\n"+
			"    uint nKeys = nKeys_rows[mc3_row];\n"+
			"    device const half* kc = kc_all + slotOff[mc3_row]; device const half* vc = vc_all + slotOff[mc3_row];"))
	b.WriteString("\n")
	return b.String()
}
