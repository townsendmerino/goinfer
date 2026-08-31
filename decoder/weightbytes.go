package decoder

import "github.com/townsendmerino/aikit/linalg"

// wmBytes is one weight matrix's in-memory footprint, read from the BACKING SLICES rather than
// computed from Rows()*Cols(). The two disagree: a quantized matrix carries per-group scales
// alongside its packed payload, and int4 packs two values per byte, so a dimension-derived
// figure is wrong in both directions at once. Asking the matrix what it is holding cannot drift
// from what it actually holds when a new kind is added.
func wmBytes(w *linalg.WeightMat) int64 {
	var n int64
	if q8, sc, _, ok := w.Int8(); ok {
		n += int64(len(q8)) + 4*int64(len(sc))
	}
	if q4, sc, _, ok := w.Int4(); ok {
		n += int64(len(q4)) + 4*int64(len(sc))
	}
	if p4, sc, ok := w.Int4Row4(); ok {
		n += int64(len(p4)) + 4*int64(len(sc))
	}
	if f, ok := w.F32(); ok {
		n += 4 * int64(len(f))
	}
	n += int64(w.SplitHalfBytes())
	return n
}

// ResidentWeightBytes is the total byte footprint of this model's weight matrices — what a
// resident backend must hold to run the whole model on-device.
//
// WHY IT EXISTS. A resident backend had no way to ask "will this model fit?" before allocating,
// and nothing else in the tree answers it: Dims() exposes hidden/layers/heads but NOT the expert
// count, so a shape-derived estimate under-reports a sparse MoE by the factor that matters most —
// gpt-oss-20b's experts ARE the model. Measured 2026-08-31: loading an 11.28 GB gpt-oss-20b on
// Metal's resident path on a 16 GB machine drove swap to 35.98 GB of 36 GB and never completed OR
// declined, because the only size guard in the tree caps the KV CONTEXT (metal/backend.go), not
// the weights.
//
// It sums the MATRICES, which is where the bytes are; the elementwise norms/biases are [hidden]-
// sized and round to nothing beside them. That makes this a LOWER BOUND on the real footprint,
// which is the safe direction for a guard: it can fail to refuse a marginal model, but it cannot
// refuse one that would have fit.
//
// This is a quantity we COMPUTE, deliberately — not the OS's account of free memory. Darwin's UBC
// reclaims under pressure, so "available" reports what survived rather than what can be asked
// for; an RSS-keyed ceiling once reported LESS memory at a known failure point than at baseline.
func (m *Model) ResidentWeightBytes() int64 {
	if m == nil || m.w == nil {
		return 0
	}
	w := m.w
	n := wmBytes(&w.Embed) + wmBytes(&w.LMHead) + wmBytes(&w.PosEmbed)
	for i := range w.Layers {
		l := &w.Layers[i]
		for _, mat := range []*linalg.WeightMat{
			&l.QProj, &l.KProj, &l.VProj, &l.OProj, &l.GProj,
			&l.GateProj, &l.UpProj, &l.DownProj,
			&l.Router, &l.SharedGate, &l.PLEGate, &l.PLEProj,
		} {
			n += wmBytes(mat)
		}
		// The experts are the whole point of this accessor — a sparse MoE is mostly experts, and
		// omitting them is the specific under-report that would let gpt-oss through the guard.
		for j := range l.Experts {
			e := &l.Experts[j]
			n += wmBytes(&e.Gate) + wmBytes(&e.Up) + wmBytes(&e.Down)
		}
		n += wmBytes(&l.SharedExpert.Gate) + wmBytes(&l.SharedExpert.Up) + wmBytes(&l.SharedExpert.Down)
	}
	return n
}
