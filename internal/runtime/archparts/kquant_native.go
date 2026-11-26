package archparts

import (
	"sync"
)

// KQuantSrc is the GGUF tensor an int8 matmul of a q4k load was re-quantized from: kind "q5k" or "q6k", and the source row
// of destination row r: start+r for a plain load or a fused split (perm nil), else perm[r] (a RoPE permutation).
type KQuantSrc struct {
	Tensor string
	Kind   string
	Start  int
	Perm   []int32
	// int8Bytes and nativeBytes are the weight's int8 copy (rows and per-row scales) and the file's blocks for it: what a lane
	// that uploads the blocks saves on the device (kquantSavings).
	Int8Bytes, NativeBytes int64
}

// Src is destination row r's source row.
func (s KQuantSrc) Src(r int) int {
	if s.Perm != nil {
		return int(s.Perm[r])
	}
	return s.Start + r
}

// KQuantRows maps rows destination rows to their source rows (rowSrc(r) / in): start for a run start, start+1, ... (perm nil),
// else every row's source in perm.
func KQuantRows(rows, in int, rowSrc func(r int) int) (start int, perm []int32) {
	start = rowSrc(0) / in
	for r := range rows {
		if rowSrc(r)/in != start+r {
			perm = make([]int32, rows)
			for i := range perm {
				perm[i] = int32(rowSrc(i) / in)
			}
			return start, perm
		}
	}
	return start, nil
}

// KQuantServes reports whether a raw accessor's answer (dims, ok; its caller has already handled an error) for a weight of rows
// destination rows over in columns, mapped by src, is one KQuantReader.Rows will serve: stored as the kind asked, two dims whose
// columns are in (positive, a multiple of 256), and every source row inside the tensor (a run, or a permutation of exactly rows
// entries).
// noteKQuant records only such a source and Rows checks the same, both with the weight's own columns and rows, so a recorded
// weight is one the lane uploads natively.
func KQuantServes(dims []int, ok bool, in int, src KQuantSrc, rows int) bool {
	if !ok || len(dims) != 2 || dims[0] != in || in <= 0 || in%256 != 0 || rows <= 0 {
		return false
	}
	if src.Perm == nil {
		return src.Start >= 0 && src.Start+rows <= dims[1]
	}
	if len(src.Perm) != rows {
		return false
	}
	for _, r := range src.Perm {
		if r < 0 || int(r) >= dims[1] {
			return false
		}
	}
	return true
}

// KQuantMu guards every Weights.kquant while a load records into it: parallelLayers builds layers on several goroutines.
// Package-level because Weights is copied by value in places, and loads are few.
var KQuantMu sync.Mutex
