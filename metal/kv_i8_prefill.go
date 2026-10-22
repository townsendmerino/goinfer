//go:build darwin

package metal

// kvI8PrefillKernels are the two kernels the f16 MMA prefill adds for an int8 KV cache (P3b-2, D9 (b),
// docs/tasks/task-kv-precision-everywhere-2026-10.md), appended to the prefill library so no f16 kernel changes.
//
//   - kv_store_i8_prefill stores a block's K/V rows into the int8 cache. It is kv_store_i8's source with each K/V read widened
//     from the pass's half values (one thread per row and KV head), so its codes and scales are the ones kv_store_i8 writes
//     for those values.
//   - kv_dequant_i8_f16 writes rows [0, n) of a layer's int8 cache, as half(code × scale), into the f16 scratch the pass's
//     attention kernels then read in place of the cache.
var kvI8PrefillKernels = deriveKVI8PrefillKernels()

func deriveKVI8PrefillKernels() string {
	store := edit(extractKernel(allKernels, "kv_store_i8"),
		"kernel void kv_store_i8(device const float* k[[buffer(0)]], device const float* v[[buffer(1)]],",
		"kernel void kv_store_i8_prefill(device const half* qkv[[buffer(0)]], device const uint* positions[[buffer(1)]],",
		"constant uint& pos[[buffer(8)]], uint kvh[[thread_position_in_grid]]) {",
		"constant uint& stride[[buffer(8)]], constant uint& kOff[[buffer(9)]], constant uint& vOff[[buffer(10)]],\n"+
			"    constant uint& rowsM[[buffer(11)]], uint gid[[thread_position_in_grid]]) {\n"+
			"    uint m = gid / nKV; if (m >= rowsM) return; uint kvh = gid % nKV;\n"+
			"    device const half* k = qkv + m*stride + kOff; device const half* v = qkv + m*stride + vOff; uint pos = positions[m];",
		"abs(k[base + d])", "abs(float(k[base + d]))",
		"abs(v[base + d])", "abs(float(v[base + d]))",
		"round(k[base + d] * inv_k)", "round(float(k[base + d]) * inv_k)",
		"round(v[base + d] * inv_v)", "round(float(v[base + d]) * inv_v)")
	return "\n// ---- the int8 KV cache's prefill store and dequantize (kv_i8_prefill.go) ----\n" + store + `
kernel void kv_dequant_i8_f16(device const char* kc[[buffer(0)]], device const char* vc[[buffer(1)]],
    device const float* ks[[buffer(2)]], device const float* vs[[buffer(3)]],
    device half* kOut[[buffer(4)]], device half* vOut[[buffer(5)]],
    constant uint& nKV[[buffer(6)]], constant uint& hd[[buffer(7)]], constant uint& n[[buffer(8)]],
    uint gid[[thread_position_in_grid]]) {
    uint kvDim = nKV * hd;
    if (gid >= n * kvDim) return;
    uint pos = gid / kvDim, h = (gid % kvDim) / hd;
    kOut[gid] = half(float(kc[gid]) * ks[pos * nKV + h]);
    vOut[gid] = half(float(vc[gid]) * vs[pos * nKV + h]);
}
`
}
