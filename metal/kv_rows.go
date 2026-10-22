//go:build darwin

package metal

import "strings"

// kvRowsKernels are the batched step's KV store and attention for an int8 and an f32 cache (P3b-1,
// docs/tasks/task-kv-precision-everywhere-2026-10.md), derived at init the way mc3RowsKernels are: the int8 pair from
// kv_store_i8 and attention_i8 by renaming the per-row parameters and adding the same prologue, the f32 pair from the f16 rows
// kernels by the type substitution attention_f32 is made by (kv_f32.go). Bodies are never retyped, so each row computes what its
// single-token step computes, bit for bit.
//
// A row's int8 scales sit at its slot offset over head_dim: a slot holds kvDim int8s and nKV scales per position. The int8
// attention binds no sink (an attention-sink family is never batched, batchIneligible), which keeps it within aikit's 16 buffers.
var kvRowsKernels = deriveKVRowsKernels()

func deriveKVRowsKernels() string {
	var b strings.Builder
	b.WriteString("\n// ---- the batched step's int8 and f32 KV store and attention (kv_rows.go) ----\n")
	b.WriteString(edit(extractKernel(allKernels, "kv_store_i8"),
		"kernel void kv_store_i8(device const float* k[[buffer(0)]], device const float* v[[buffer(1)]],\n"+
			"    device char* kc[[buffer(2)]], device char* vc[[buffer(3)]],\n"+
			"    device float* ks[[buffer(4)]], device float* vs[[buffer(5)]],",
		"kernel void mc3_kv_store_i8_rows(device const float* k_rows[[buffer(0)]], device const float* v_rows[[buffer(1)]],\n"+
			"    device char* kc_all[[buffer(2)]], device char* vc_all[[buffer(3)]],\n"+
			"    device float* ks_all[[buffer(4)]], device float* vs_all[[buffer(5)]],",
		"constant uint& pos[[buffer(8)]], uint kvh[[thread_position_in_grid]]) {",
		"device const uint* pos_rows[[buffer(8)]], constant uint& rowStride[[buffer(9)]], constant uint& mc3_M[[buffer(10)]],\n"+
			"    device const uint* slotOff[[buffer(11)]], uint kvh_rows[[thread_position_in_grid]]) {\n"+
			"    uint mc3_row = kvh_rows / nKV; if (mc3_row >= mc3_M) return; uint kvh = kvh_rows % nKV;\n"+
			"    device const float* k = k_rows + mc3_row*rowStride; device const float* v = v_rows + mc3_row*rowStride;\n"+
			"    uint pos = pos_rows[mc3_row]; device char* kc = kc_all + slotOff[mc3_row]; device char* vc = vc_all + slotOff[mc3_row];\n"+
			"    device float* ks = ks_all + slotOff[mc3_row]/hd; device float* vs = vs_all + slotOff[mc3_row]/hd;"))
	b.WriteString("\n")
	b.WriteString(edit(extractKernel(allKernels, "attention_i8"),
		"kernel void attention_i8(device const float* q[[buffer(0)]], device const char* kc[[buffer(1)]],\n"+
			"    device const char* vc[[buffer(2)]], device const float* ks[[buffer(3)]],\n"+
			"    device const float* vs[[buffer(4)]], device float* out[[buffer(5)]],",
		"kernel void mc3_attention_i8_rows(device const float* q_rows[[buffer(0)]], device const char* kc_all[[buffer(1)]],\n"+
			"    device const char* vc_all[[buffer(2)]], device const float* ks_all[[buffer(3)]],\n"+
			"    device const float* vs_all[[buffer(4)]], device float* out_rows[[buffer(5)]],",
		"constant uint& nKeys[[buffer(9)]]", "device const uint* nKeys_rows[[buffer(9)]]",
		"constant uint& window[[buffer(11)]], device const float* sinks[[buffer(12)]],\n    constant uint& hasSink[[buffer(13)]],\n    uint qh[[threadgroup_position_in_grid]],",
		"constant uint& window[[buffer(11)]],\n"+
			"    constant uint& mc3_M[[buffer(12)]], device const uint* rowmap[[buffer(13)]], device const uint* slotOff[[buffer(14)]],\n"+
			"    uint qh_rows[[threadgroup_position_in_grid]],",
		"uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]]) {",
		"uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]]) {\n"+
			"    uint mc3_j = qh_rows / nH; if (mc3_j >= mc3_M) return; uint mc3_row = rowmap[mc3_j]; uint qh = qh_rows % nH;\n"+
			"    device const float* q = q_rows + mc3_row*(nH*hd + 2u*nKV*hd); device float* out = out_rows + mc3_row*(nH*hd);\n"+
			"    uint nKeys = nKeys_rows[mc3_row]; const uint hasSink = 0u; device const float* sinks = nullptr;\n"+
			"    device const char* kc = kc_all + slotOff[mc3_row]; device const char* vc = vc_all + slotOff[mc3_row];\n"+
			"    device const float* ks = ks_all + slotOff[mc3_row]/hd; device const float* vs = vs_all + slotOff[mc3_row]/hd;"))
	b.WriteString("\n")
	for _, name := range []string{"mc3_kv_store_rows", "mc3_attention_rows"} {
		k := extractKernel(mc3RowsKernels, name)
		k = edit(k, "kernel void "+name+"(", "kernel void "+strings.TrimSuffix(name, "_rows")+"_f32_rows(")
		b.WriteString(strings.ReplaceAll(k, "half", "float"))
		b.WriteString("\n")
	}
	return b.String()
}
