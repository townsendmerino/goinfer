package decoder

import "github.com/townsendmerino/goinfer/internal/runtime/cpuengine"

// CPUScratchTerm is one named row of a CPU decode stream's fixed scratch and its bytes.
type CPUScratchTerm struct {
	Role  string
	Bytes int64
}

// CPUDecodeScratch is the fixed host scratch one decode stream of m allocates when its KV cache is made (newDecodeScratch), as one term
// per row, in f32: the residual, the normed row, the two sublayer outputs, q, k, v and the context, the FFN's gate and up, the logits
// and, for a mixture of experts, the router's and the experts' rows. It is what the constructor allocates from (cpuengine.DecodeScratchRows), so it
// equals the cache's own rows by construction; TestCPUDecodeScratch_equalsTheBuild holds that on every census fixture and names any
// row a stream holds that the list does not. Rows grown on demand (the attention scores, the batched logits, the per-head pool, the
// quantization scratch) are not in it. It is host memory no guard prices; reported, not charged.
func (m *Model) CPUDecodeScratch() []CPUScratchTerm {
	if m.w == nil {
		return nil
	}
	var terms []CPUScratchTerm
	for _, r := range cpuengine.DecodeScratchRows(m.w.Arch) {
		terms = append(terms, CPUScratchTerm{r.Name, 4 * int64(r.Elems)})
	}
	return terms
}

// CPUDecodeScratchBytes is the sum of CPUDecodeScratch's terms.
func (m *Model) CPUDecodeScratchBytes() int64 {
	var n int64
	for _, t := range m.CPUDecodeScratch() {
		n += t.Bytes
	}
	return n
}
