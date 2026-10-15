package decoder

// ResidentPrefillChunks is the optional capability of a Prefiller that splits a long prompt into passes itself: the decoder hands it
// the whole suffix in one PrefillLast call and the resident cuts it. PrefillChunkRows is the most rows one of its passes takes, and
// PrefillScratchBytes what a pass of that many rows allocates beyond the resident's standing buffers. Admission prices a request at
// that one pass instead of the whole-prompt figure (residentPrefillScratch). It is not ResidentPrefillBlocks: that interface makes the
// decoder split the prompt and needs PrefillTailExact, which a resident whose passes differ in kernel class by length cannot claim.
type ResidentPrefillChunks interface {
	PrefillChunkRows() int
	PrefillScratchBytes(rows int) int64
}

// ResidentPrefillTailMin is the optional capability of a resident whose last pass of a prefill must have a minimum number of rows
// to run the same kernels as the pass before it (CUDA: 16, the fused attention's floor). Every loop that cuts a prompt into
// passes reads it instead of prefillTailMin.
type ResidentPrefillTailMin interface {
	PrefillTailMinRows() int
}

// prefillTailMinRows is the shortest final pass the resident takes: its own figure when it reports one above the default.
func (m *Model) prefillTailMinRows() int {
	if tm, ok := m.resident.(ResidentPrefillTailMin); ok {
		return max(prefillTailMin, tm.PrefillTailMinRows())
	}
	return prefillTailMin
}

// residentChunkedScratch prices a request on a resident that chunks its own prefill: the device scratch of one pass of at most
// its chunk (the resident's own figure), plus the host embeddings the decoder builds for the whole suffix, which this path does
// allocate in full. ok is false for a resident that does not report chunks.
func (m *Model) residentChunkedScratch(promptTokens int) (bytes int64, ok bool) {
	sc, has := m.resident.(ResidentPrefillChunks)
	if !has || promptTokens <= 0 {
		return 0, false
	}
	C := sc.PrefillChunkRows()
	if C <= 0 {
		return 0, false
	}
	rows := min(promptTokens, C)
	return sc.PrefillScratchBytes(rows) + int64(promptTokens)*4*int64(m.ResidentEmbedLen()), true
}
