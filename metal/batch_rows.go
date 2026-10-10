//go:build darwin

package metal

import (
	"fmt"
	"strings"
)

// MC3 S3 (docs/tasks/task-concurrency-2026-09.md): multi-row forms of the per-row kernels a batched step runs once per
// sequence (rmsnorm_quant, quant_vec, swiglu_quant, rope2). Each of those small dispatches ran a single threadgroup, one
// sequence after another; one dispatch over all rows runs them side by side (cost figures:
// docs/code-notes/metal.md#mc3RowsKernels).
//
// Their BODIES are production's, byte for byte: each variant is derived from allKernels' own source at init, by renaming the
// pointer / per-row parameters in the signature and adding a prologue that points them at the row this threadgroup (or
// thread) serves. Nothing in a body is retyped, so a variant cannot drift from its kernel (these kernels round differently
// when their code shape changes, see rmsnorm_quant's own notes, which is why they are not hand-copied), and a production
// signature edit panics here at init instead of silently diverging. They compile into the resident's main library with the
// originals' fast-math setting.

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
	// qk_norm (E-P07, Qwen3's per-head Q/K RMSNorm): one threadgroup per (row, head); a row's fused qkv advances
	// qkvStride floats, and the head is the threadgroup's index within its row.
	b.WriteString(edit(extractKernel(allKernels, "qk_norm"),
		"kernel void qk_norm(device float* qkv[[buffer(0)]]",
		"kernel void mc3_qk_norm_rows(device float* qkv_rows[[buffer(0)]]",
		"constant uint& addOne[[buffer(8)]], uint head[[threadgroup_position_in_grid]],",
		"constant uint& addOne[[buffer(8)]], constant uint& qkvStride[[buffer(9)]], uint mc3_gid[[threadgroup_position_in_grid]],",
		"    threadgroup float red[128];",
		"    uint head = mc3_gid % (nH + nKV); device float* qkv = qkv_rows + (mc3_gid / (nH + nKV))*qkvStride;\n    threadgroup float red[128];"))
	b.WriteString("\n")

	// E-P05: the first pass of decode's flash attention (attention_fa, attention_fa_blk<G>, attention_fa_blk64<G>) for
	// M rows at attention_fa depth in one dispatch. Row j serves batch row rowmap[j]: its own q (a row of the fused qkv
	// buffer, qStride floats), its own slot's K/V (slotOff), its key count and split count, and its own region of the
	// partial buffer (partStride floats). The grid is M rows of nKV*maxS threadgroups; a threadgroup past its row's own
	// split count returns, and the rest see the threadgroup index the row's own dispatch gave them, so every row's
	// partials are its own dispatch's. 16 buffers: aikit binds at most 16.
	for _, fa := range []struct{ name, tmpl string }{{"attention_fa", ""}, {"attention_fa_blk", "template <uint G>\n"}, {"attention_fa_blk64", "template <uint G>\n"}} {
		b.WriteString(fa.tmpl)
		b.WriteString(edit(extractKernel(allKernels, fa.name),
			"kernel void "+fa.name+"(", "kernel void mc3_"+fa.name+"_rows(",
			"device const float* q[[buffer(0)]], device const half* kc[[buffer(1)]],\n    device const half* vc[[buffer(2)]], device float* partial[[buffer(3)]],",
			"device const float* q_rows[[buffer(0)]], device const half* kc_all[[buffer(1)]],\n    device const half* vc_all[[buffer(2)]], device float* partial_rows[[buffer(3)]],",
			"constant uint& nKeys[[buffer(6)]]", "device const uint* nKeys_rows[[buffer(6)]]",
			"constant uint& window[[buffer(8)]], constant uint& nSplit[[buffer(9)]],",
			"constant uint& window[[buffer(8)]], device const uint* nSplit_rows[[buffer(9)]],\n"+
				"    constant uint& mc3_M[[buffer(10)]], device const uint* rowmap[[buffer(11)]], device const uint* slotOff[[buffer(12)]],\n"+
				"    constant uint& qStride[[buffer(13)]], constant uint& partStride[[buffer(14)]], constant uint& maxS[[buffer(15)]],",
			"uint tgid[[threadgroup_position_in_grid]]", "uint tg_rows[[threadgroup_position_in_grid]]",
			"uint lane[[thread_index_in_simdgroup]]) {",
			"uint lane[[thread_index_in_simdgroup]]) {\n"+
				"    uint mc3_j = tg_rows / (nKV*maxS); if (mc3_j >= mc3_M) return;\n"+
				"    uint mc3_row = rowmap[mc3_j]; uint nSplit = nSplit_rows[mc3_row]; uint mc3_loc = tg_rows % (nKV*maxS);\n"+
				"    if (mc3_loc % maxS >= nSplit) return;\n"+
				"    uint tgid = (mc3_loc / maxS)*nSplit + mc3_loc % maxS; uint nKeys = nKeys_rows[mc3_row];\n"+
				"    device const float* q = q_rows + mc3_row*qStride; device float* partial = partial_rows + mc3_j*partStride;\n"+
				"    device const half* kc = kc_all + slotOff[mc3_row]; device const half* vc = vc_all + slotOff[mc3_row];"))
		b.WriteString("\n")
	}
	for g := 2; g <= 8; g++ {
		fmt.Fprintf(&b, "template [[host_name(\"mc3_attention_fa_blk_rows_g%d\")]] kernel decltype(mc3_attention_fa_blk_rows<%d>) mc3_attention_fa_blk_rows<%d>;\n", g, g, g)
	}
	b.WriteString("template [[host_name(\"mc3_attention_fa_blk64_rows_g7\")]] kernel decltype(mc3_attention_fa_blk64_rows<7>) mc3_attention_fa_blk64_rows<7>;\n")
	// Its combine: row j merges its own partial region into its own row of ctx (outStride floats), over its own splits.
	b.WriteString(edit(extractKernel(allKernels, "attention_fa_combine"),
		"kernel void attention_fa_combine(", "kernel void mc3_attention_fa_combine_rows(",
		"device const float* partial[[buffer(0)]], device float* out[[buffer(1)]],",
		"device const float* partial_rows[[buffer(0)]], device float* out_rows[[buffer(1)]],",
		"constant uint& nSplit[[buffer(4)]],",
		"device const uint* nSplit_rows[[buffer(4)]], constant uint& nH[[buffer(5)]], constant uint& mc3_M[[buffer(6)]],\n"+
			"    device const uint* rowmap[[buffer(7)]], constant uint& partStride[[buffer(8)]], constant uint& outStride[[buffer(9)]],",
		"uint qh[[threadgroup_position_in_grid]], uint d[[thread_position_in_threadgroup]]) {",
		"uint qh_rows[[threadgroup_position_in_grid]], uint d[[thread_position_in_threadgroup]]) {\n"+
			"    uint mc3_j = qh_rows / nH; if (mc3_j >= mc3_M) return; uint mc3_row = rowmap[mc3_j]; uint qh = qh_rows % nH;\n"+
			"    uint nSplit = nSplit_rows[mc3_row]; device const float* partial = partial_rows + mc3_j*partStride;\n"+
			"    device float* out = out_rows + mc3_row*outStride;"))
	b.WriteString("\n")

	// int8 slice 3b (docs/tasks/task-metal-int8-2026-10.md): the native int8 path's projection for B rows of the batched
	// step, each weight row read ONCE for all B activation rows. One simdgroup per output row keeps B integer
	// accumulators; W8A8 sums are exact, so any order gives decode's integer, and each row's epilogue is decode's own
	// statement (W8A8_SA_BODY / W8A8_COAL_BODY: float(acc)*aScale*wScale through simd_broadcast_first, then the site's
	// bias or residual). mode: 0 store (gate|up), 1 +bias (qkv), 2 += (o, down). Activation row b is K int8s at aq_rows +
	// b*K with its scale at asc_rows[b]; output row b is ldo floats from out_rows. B <= 8.
	b.WriteString(`template <uint B>
kernel void mc3_gemv_w8a8_rows(device const char* bq[[buffer(0)]], device const float* bsc[[buffer(1)]],
    device const char* aq_rows[[buffer(2)]], device const float* asc_rows[[buffer(3)]], device float* out_rows[[buffer(4)]],
    constant uint& K[[buffer(5)]], constant uint& ldo[[buffer(6)]], constant uint& Bu[[buffer(7)]],
    device const float* bias[[buffer(8)]], constant uint& mode[[buffer(9)]],
    uint gid[[threadgroup_position_in_grid]], uint lid[[thread_index_in_threadgroup]]) {
    (void)Bu;
    device const uint* brow = (device const uint*)(bq + (uint)gid*K);
    device const uint* a4 = (device const uint*)aq_rows;
    const uint G = K >> 2u;
    int acc[B];
    SA_ROWS_UNROLL for (uint b = 0u; b < B; b++) acc[b] = 0;
    for (uint g = lid; g < G; g += 32u) {
        const uint w = brow[g];
        SA_ROWS_UNROLL for (uint b = 0u; b < B; b++) acc[b] += DOT4I8(a4[b*G + g], w);
    }
    SA_ROWS_UNROLL for (uint b = 0u; b < B; b++) {
        int s = simd_sum(acc[b]);
        float y = simd_broadcast_first(float(s) * asc_rows[b] * bsc[gid]);
        if (lid == 0) {
            device float* out = out_rows + b*ldo;
            if (mode == 0u) out[gid] = y;
            else if (mode == 1u) out[gid] = y + bias[gid];
            else out[gid] += y;
        }
    }
}
`)
	for B := 2; B <= 8; B++ {
		fmt.Fprintf(&b, "template [[host_name(\"mc3_gemv_w8a8_rows%d\")]] kernel decltype(mc3_gemv_w8a8_rows<%d>) mc3_gemv_w8a8_rows<%d>;\n", B, B, B)
	}

	// D-P03: decode's k selected experts in one dispatch per stage instead of one per slot. The grid is k slots of
	// rowsPerExpert/(tgs/32) threadgroups; a threadgroup's slot is its index divided by that, and within the slot it
	// is the threadgroup production's per-slot dispatch had. Needs rowsPerExpert % (tgs/32) == 0 (encodeMoEExperts
	// checks it). Gate|up: each slot writes its own [gate | up] row of out_k (2I apart), which mc3_swiglu_quant_rows
	// reads as its rows.
	moeTail := "uint lane[[thread_index_in_simdgroup]]) {"
	moePrologue := "\n    uint dp03_per = rowsPerExpert/(tgs>>5u); uint slot = dp03_tg / dp03_per; uint tgid = dp03_tg % dp03_per;"
	b.WriteString(edit(extractKernel(allKernels, "gemv_w4a8_moe"),
		"kernel void gemv_w4a8_moe(", "kernel void dp03_gemv_w4a8_moe_k(",
		"device float* out[[buffer(4)]]", "device float* out_k[[buffer(4)]]",
		"constant uint& slot[[buffer(7)]]", "constant uint& dp03_k[[buffer(7)]]",
		"uint tgid[[threadgroup_position_in_grid]]", "uint dp03_tg[[threadgroup_position_in_grid]]",
		moeTail, moeTail+moePrologue+" device float* out = out_k + slot*rowsPerExpert;"))
	b.WriteString("\n")
	// Down: slot j reads its own activation row (dq rows K = I apart, one scale each) and stores the simdgroup sum
	// acc, unweighted, to acc_k; moe_combine_k then adds wgt[j]*acc*asc[0] into the residual in slot order, the
	// accumulation gemv_w4a8_moe_wacc does one dispatch per slot.
	b.WriteString(edit(extractKernel(allKernels, "gemv_w4a8_moe_wacc"),
		"kernel void gemv_w4a8_moe_wacc(", "kernel void dp03_gemv_w4a8_moe_down_k(",
		"device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]]",
		"device const char* aq_k[[buffer(2)]], device const float* asc_k[[buffer(3)]], device float* acc_k[[buffer(4)]]",
		"constant uint& slot[[buffer(8)]]", "constant uint& dp03_k[[buffer(8)]]",
		"uint tgid[[threadgroup_position_in_grid]]", "uint dp03_tg[[threadgroup_position_in_grid]]",
		moeTail, moeTail+moePrologue+" device const char* aq = aq_k + slot*K;",
		"if (lane==0) out[row] += wgt[slot]*acc*asc[0];", "if (lane==0) acc_k[slot*rowsPerExpert + row] = acc;"))
	b.WriteString("\n")
	// The combine: gemv_w4a8_moe_wacc's epilogue statement, verbatim, once per slot in slot order.
	b.WriteString(`kernel void moe_combine_k(device float* out[[buffer(0)]], device const float* acc_k[[buffer(1)]],
    device const float* wgt[[buffer(2)]], device const float* asc_k[[buffer(3)]], constant uint& k[[buffer(4)]],
    constant uint& H[[buffer(5)]], uint row[[thread_position_in_grid]]) {
    if (row >= H) return;
    for (uint slot=0u; slot<k; slot++) {
        float acc = acc_k[slot*H + row]; device const float* asc = asc_k + slot;
        out[row] += wgt[slot]*acc*asc[0];
    }
}
`)

	return b.String()
}
