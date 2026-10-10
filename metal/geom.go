//go:build darwin

package metal

// attnGeom is one distinct per-layer attention geometry, shared by every layer that has it. Gemma 4's own-forward residency
// interleaves two attention shapes in one model (local: head_dim 256, kv_heads 8, full rotary; global: head_dim 512,
// kv_heads 2, partial rotary, K=V), so geometry cannot live model-level on *resident: it has no hd/nKV fields, so reading
// the wrong source is a compile error. On a uniform family every layer resolves to the same geometry and geomFor dedups by
// value, so the whole model shares one attnGeom. This mirrors the WebGPU bridge's value-keyed dedup (gpu/): one object per
// distinct {hd, nKV, half, kEqV}.
//
// nH (query heads) is deliberately NOT in the key: it is constant across a family's layers (Gemma 4 varies head_dim and
// kv_heads, not the query-head count), so it stays on *resident. A family with PER-LAYER query-head counts must move nH onto
// attnGeom and add it to the key. nHhd = nH*hd does vary with hd, so uNHhd is derived per geometry. kEqV is in the key
// because a K=V layer's V-store differs from a v_proj layer's at identical dims (V = v_norm(raw k), its own cache), so two
// such layers must not share a geom.
// History: docs/code-notes/metal.md#attnGeom.
type attnGeom struct {
	hd, nKV, kvDim, half int  // grid-sizing scalars; kvDim = nKV*hd, half = rotaryDim/2 (rotated pairs/head)
	kEqV                 bool // attention_k_eq_v: V = v_norm(raw pre-RoPE k), no v_proj (Gemma 4 globals; 9c Step 3)

	// Kernel-arg uniform buffers (Metal passes geometry as MTLBuffers, not scalars). uNHhd (= nH*hd) serves both the qk-norm and
	// the o-proj/quant sites.
	uHd, uKvDim, uNKV, uNHhd, uHalf, uQtotal, uKtotal Buffer
}

// geomFor returns the attnGeom for {hd, nKV, half, kEqV}, allocating its uniform buffers once and
// caching by value so repeated (uniform) geometries share a single object. r.nH must be set before
// the first call (BuildResident sets it on the struct literal). The cache is caller-owned (a local
// in BuildResident) — the geom's buffers are on the device ledger and freed by Close like any other.
func (r *resident) geomFor(cache map[[4]int]*attnGeom, hd, nKV, half int, kEqV bool) *attnGeom {
	key := [4]int{hd, nKV, half, b2i(kEqV)}
	if g := cache[key]; g != nil {
		return g
	}
	d := r.d
	g := &attnGeom{
		hd: hd, nKV: nKV, kvDim: nKV * hd, half: half, kEqV: kEqV,
		uHd:     NewBufferU32(d, uint32(hd)),
		uKvDim:  NewBufferU32(d, uint32(nKV*hd)),
		uNKV:    NewBufferU32(d, uint32(nKV)),
		uNHhd:   NewBufferU32(d, uint32(r.nH*hd)),
		uHalf:   NewBufferU32(d, uint32(half)),
		uQtotal: NewBufferU32(d, uint32(r.nH*half)),
		uKtotal: NewBufferU32(d, uint32(nKV*half)),
	}
	cache[key] = g
	return g
}

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}
