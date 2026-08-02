//go:build darwin

package metal

// attnGeom is one distinct per-layer attention geometry, shared by every layer that has it.
// It exists because Gemma 4's own-forward residency (9c) interleaves two attention shapes in
// one model — local (head_dim 256, kv_heads 8, full rotary) and global (head_dim 512, kv_heads
// 2, partial rotary, K=V) — so geometry can no longer live model-level on *Resident. A launch
// site that read r.hd/r.nKV/... would silently run every layer at the model-level (uniform)
// shape; those fields were REMOVED so binding the wrong source is a compile error, not a review
// catch. On a uniform family every layer resolves to the SAME geometry, so geomFor dedups by
// value and the whole model shares one attnGeom — byte-identical to the old model-level path.
//
// nH (query heads) is LOAD-BEARING as a model-level field: it is NOT part of the geom key
// because it is constant across a family's layers (Gemma 4 varies head_dim and kv_heads, not the
// query-head count — 16 in both the 12B and 26B variants), so it stays on *Resident. A future
// family with PER-LAYER query-head counts would break this: it would have to move nH onto attnGeom
// and add it to the key. This comment is the marker that stops the next person bisecting for it.
// nHhd = nH*hd DOES vary with hd, so uNHhd is derived per-geometry here. kEqV joins the key
// because a K=V layer's V-store behaviour differs from a v_proj layer's even at identical dims
// (9c Step 3: V = v_norm(raw k), its own cache) — two such layers must not share a geom.
//
// This mirrors the WebGPU bridge's value-keyed geometry dedup (gpu/, commit 9ec363f): one
// geometry object per distinct {hd, nKV, half, kEqV}, shared across the layers that request it.
type attnGeom struct {
	hd, nKV, kvDim, half int  // grid-sizing scalars; kvDim = nKV*hd, half = rotaryDim/2 (rotated pairs/head)
	kEqV                 bool // attention_k_eq_v: V = v_norm(raw pre-RoPE k), no v_proj (Gemma 4 globals; 9c Step 3)

	// Kernel-arg uniform buffers (Metal passes geometry as MTLBuffers, not scalars). uNHhd (= nH*hd)
	// serves both the qk-norm and the o-proj/quant sites that previously used the two equal-valued
	// r.uNHhd / r.uHH buffers.
	uHd, uKvDim, uNKV, uNHhd, uHalf, uQtotal, uKtotal Buffer
}

// geomFor returns the attnGeom for {hd, nKV, half, kEqV}, allocating its uniform buffers once and
// caching by value so repeated (uniform) geometries share a single object. r.nH must be set before
// the first call (BuildResident sets it on the struct literal). The cache is caller-owned (a local
// in BuildResident) — the geom's buffers are on the device ledger and freed by Close like any other.
func (r *Resident) geomFor(cache map[[4]int]*attnGeom, hd, nKV, half int, kEqV bool) *attnGeom {
	key := [4]int{hd, nKV, half, b2i(kEqV)}
	if g := cache[key]; g != nil {
		return g
	}
	d := r.d
	g := &attnGeom{
		hd: hd, nKV: nKV, kvDim: nKV * hd, half: half, kEqV: kEqV,
		uHd:     d.NewBufferU32(uint32(hd)),
		uKvDim:  d.NewBufferU32(uint32(nKV * hd)),
		uNKV:    d.NewBufferU32(uint32(nKV)),
		uNHhd:   d.NewBufferU32(uint32(r.nH * hd)),
		uHalf:   d.NewBufferU32(uint32(half)),
		uQtotal: d.NewBufferU32(uint32(r.nH * half)),
		uKtotal: d.NewBufferU32(uint32(nKV * half)),
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
