package archparts

// KVRowPadElems is the pad, in elements, for a row of elems elements of elemBytes bytes: one 64 B line when the row is a multiple of
// align bytes, else none.
func KVRowPadElems(elems, elemBytes, align int) int {
	if elems <= 0 || align <= 0 || (elems*elemBytes)%align != 0 {
		return 0
	}
	return 64 / elemBytes
}

// KVRowPadAlignDefault is the row-pad alignment that pays on the CPU.
const KVRowPadAlignDefault = 4096
