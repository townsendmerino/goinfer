//go:build !arm64

package decoder

// repackRow4ForEmit is a no-op on non-arm64: RepackW4A8Row4/RepackW4A8Row4Scales
// are arm64-only in aikit (the split-half + 4-row layout is NEON-specific), so
// there is no portable way to produce these bytes on this build. The .giw
// writer falls back to kind 3 for every tensor when this always returns
// ok=false — row4 emission (docs/task-w4a8-neon-bandwidth.md's "Format
// follow-on") requires running cmd/prequant on an arm64 box.
func repackRow4ForEmit(q4 []byte, q4s []float32, rows, cols, group int) (row4 []byte, row4Scales []float32, ok bool) {
	return nil, nil, false
}
