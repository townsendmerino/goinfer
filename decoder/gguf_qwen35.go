package decoder

// GGUF loading for the qwen3_5_moe (Qwen3.6-35B-A3B) architecture
// (general.architecture = "qwen35moe"). This is a TRANSFORM REVERSER, not a
// name-map: llama.cpp's converter bakes several layout changes into the GGUF that
// must be undone here so the existing safetensors-shaped forward runs unchanged —
// the V-head tiling (reorderVHeads/untileVHeads below), the −exp(A_log) bake
// (negExpAFromLog), and the (1+w) norm bake on the standard norms (the vnorm
// closure in buildWeightsFromGGUF; ssm_norm is exempt). The tensor loading itself
// lives in buildWeightsFromGGUF's qwen35 branch, the config map in
// internal/runtime/ggufconfig; this file holds the reorder primitive.

// reorderVHeads swaps the (outer, inner) factorization of the reordered axis of a
// [lead, outer*inner*hd, trail] row-major buffer: the element at (outer=o,
// inner=i, ...) moves to (outer=i, inner=o, ...). hd is the contiguous head block
// that rides along untouched; trail is the product of all dims after the reordered
// axis (e.g. the "in"/columns for a row reorder, or 1 for a column reorder); lead
// is the product of all dims before it (1 for a row reorder, the rows for a column
// reorder).
//
// This mirrors llama.cpp's _reorder_v_heads (gguf-py conversion/qwen.py): for
// linear-attn (Gated DeltaNet) layers where num_v_heads > num_k_heads, the
// converter takes HF's V heads stored GROUPED by K head and writes them TILED for
// ggml's broadcast. The forward (HF→GGUF) is reorderVHeads(x, lead, numKHeads,
// numVPerK, hd, trail); the loader UN-tiles (GGUF→HF) with the two factors
// swapped — see untileVHeads. reorderVHeads is its own inverse under that swap.
func reorderVHeads(src []float32, lead, outer, inner, hd, trail int) []float32 {
	mid := outer * inner * hd
	dst := make([]float32, len(src))
	for l := range lead {
		base := l * mid
		for o := range outer {
			for i := range inner {
				srcRow := base + (o*inner+i)*hd // [outer, inner, hd]
				dstRow := base + (i*outer+o)*hd // [inner, outer, hd]
				for c := range hd {
					s := (srcRow + c) * trail
					d := (dstRow + c) * trail
					copy(dst[d:d+trail], src[s:s+trail])
				}
			}
		}
	}
	return dst
}

// untileVHeads undoes the GGUF converter's V-head tiling: it takes a GGUF tensor
// block whose V heads are in tiled order and returns it in HF's grouped-by-K
// order. numVPerK = numVHeads / numKHeads. It is reorderVHeads with the converter's
// (numKHeads, numVPerK) factors swapped.
func untileVHeads(src []float32, lead, numKHeads, numVPerK, hd, trail int) []float32 {
	return reorderVHeads(src, lead, numVPerK, numKHeads, hd, trail)
}
