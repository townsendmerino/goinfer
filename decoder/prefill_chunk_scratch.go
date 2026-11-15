package decoder

// prefillTailMinRows is the shortest final pass the resident takes: its own figure when it reports one above the default.
func (m *Model) prefillTailMinRows() int {
	return max(prefillTailMin, residentCapsOf(m.resident).TailMinRows)
}

// residentChunkedScratch prices a request on a resident that chunks its own prefill: the device scratch of one pass of at most
// its chunk (the resident's own figure), plus the host embeddings the decoder builds for the whole suffix, which this path does
// allocate in full. ok is false for a resident that does not report chunks.
func (m *Model) residentChunkedScratch(promptTokens int) (bytes int64, ok bool) {
	pf := m.resPrefill()
	if !pf.PrefillChunksAvailable() || promptTokens <= 0 {
		return 0, false
	}
	C := pf.PrefillChunkRows()
	if C <= 0 {
		return 0, false
	}
	rows := min(promptTokens, C)
	dev := pf.PrefillScratchBytes(rows) + pf.PrefillSpanScratchBytes(rows, promptTokens)
	return dev + int64(promptTokens)*4*int64(m.BackendView().EmbedLen()), true
}
