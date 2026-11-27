package wbuild

import (
	"encoding/binary"
	"math"
	"unsafe"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/internal/runtime/giwcodec"
)

// StreamedTable is an embedding-shaped table (the token embedding, the LM head, Gemma 4's per-layer token table) that a
// streaming transcode writes from its source rows instead of building it in the heap. The build registers it with the writer under the Weights field it stands for
// (giwWriter.tables), and writeHeadGlobals writes it in two passes: every row's scales first (the format puts them before the
// codes), then the rows quantized again and written a block at a time. Each row is quantized on its own and deterministically,
// so the bytes are the ones a whole-table write of streamQuantizedEmbed's result produces.
type StreamedTable struct {
	rows, cols int
	mode       QuantMode
	newReader  func() RowReader
}

// StreamTable writes t as weightMat writes the WeightMat streamQuantizedEmbed builds for a sink (canonical, no row4 copy):
// kind 2 for int8, kind 3 for int4 (kind 5 on a cpu-arm64 target when the shape allows), kind 10 for f32.
func (w *StreamWriter) StreamTable(t *StreamedTable) {
	mode := t.mode
	if mode == QuantQ4K {
		mode = QuantInt8I8 // as streamQuantizedEmbed: a logit table has no raw Q4_K form
	}
	if mode == QuantInt4 && w.Hooks.fakeQuantScheme() != "" {
		m, err := w.Hooks.StreamQuantizedEmbed(t.rows, t.cols, mode, true, true, 1, w.TestHooks.EmbedWorkers, t.newReader)
		if err != nil {
			w.SetErr(err)
			return
		}
		w.WeightMat(&m)
		return
	}
	rows, cols := t.rows, t.cols
	header := func(kind byte, group int, w8a8 bool) {
		w.Raw([]byte{kind})
		w.U32(uint32(rows))
		w.U32(uint32(cols))
		w.U32(uint32(group))
		w.Raw([]byte{map[bool]byte{true: 1, false: 0}[w8a8]})
	}
	switch mode {
	case QuantInt8, QuantInt8I8:
		scales := make([]float32, rows)
		if !w.tablePass(t, 0, rows, func() func(int, []float32) {
			q := make([]int8, cols)
			return func(r int, row []float32) { scales[r] = linalg.QuantizeRowInt8(row, q) }
		}) {
			return
		}
		header(2, 0, mode == QuantInt8I8)
		w.AlignArray()
		w.F32Body(scales)
		w.AlignArray()
		w.U32(uint32(rows * cols))
		block := max(1, w.Hooks.streamTableBlockBytes()/cols)
		q8 := make([]int8, min(block, rows)*cols)
		for lo := 0; lo < rows && w.Err == nil; lo += block {
			hi := min(lo+block, rows)
			if !w.tablePass(t, lo, hi, func() func(int, []float32) {
				return func(r int, row []float32) { linalg.QuantizeRowInt8(row, q8[(r-lo)*cols:(r-lo+1)*cols]) }
			}) {
				return
			}
			w.Raw(unsafe.Slice((*byte)(unsafe.Pointer(&q8[0])), (hi-lo)*cols))
		}
	case QuantInt4:
		const group = Int4GroupSize
		nGroups := (cols + group - 1) / group
		bpr := (cols + 1) / 2
		q4s := make([]uint16, rows*nGroups)
		if !w.tablePass(t, 0, rows, func() func(int, []float32) {
			codes, sc := make([]byte, bpr), make([]float32, nGroups)
			return func(r int, row []float32) {
				linalg.QuantizeGroupInt4Row(row, cols, group, codes, sc)
				linalg.F32ToF16Slice(q4s[r*nGroups:(r+1)*nGroups], sc)
			}
		}) {
			return
		}
		kind, scales := byte(3), q4s
		row4 := w.Target == giwcodec.TargetCPUArm64 && giwcodec.Row4EmitEligible(rows, cols, group)
		if row4 {
			kind, scales = 5, giwcodec.RepackRow4ScalesForEmit(q4s, rows, cols, group)
		}
		header(kind, group, false)
		w.AlignArray()
		w.Scales16(scales)
		w.AlignArray()
		w.U32(uint32(rows * bpr))
		block := max(4, w.Hooks.streamTableBlockBytes()/bpr/4*4) // whole 4-row groups, the row4 layout's unit
		q4 := make([]byte, min(block, rows)*bpr)
		for lo := 0; lo < rows && w.Err == nil; lo += block {
			hi := min(lo+block, rows)
			if !w.tablePass(t, lo, hi, func() func(int, []float32) {
				sc := make([]float32, nGroups)
				return func(r int, row []float32) {
					linalg.QuantizeGroupInt4Row(row, cols, group, q4[(r-lo)*bpr:(r-lo+1)*bpr], sc)
				}
			}) {
				return
			}
			out := q4[:(hi-lo)*bpr]
			if row4 {
				out = giwcodec.RepackRow4CodesForEmit(out, hi-lo, cols, group)
			}
			w.Raw(out)
		}
	default: // f32
		if w.Hooks.GIWF32Kind1 {
			header(1, 0, false)
		} else {
			header(giwcodec.KindF32, 0, false)
			w.AlignArray()
		}
		w.U32(uint32(rows * cols))
		block := max(1, w.Hooks.streamTableBlockBytes()/(4*cols))
		buf := make([]byte, min(block, rows)*cols*4)
		for lo := 0; lo < rows && w.Err == nil; lo += block {
			hi := min(lo+block, rows)
			if !w.tablePass(t, lo, hi, func() func(int, []float32) {
				return func(r int, row []float32) {
					b := buf[(r-lo)*cols*4:]
					for i, v := range row {
						binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(v))
					}
				}
			}) {
				return
			}
			w.Raw(buf[:(hi-lo)*cols*4])
		}
	}
}

// tablePass reads source rows [lo, hi) on several goroutines (forRowRanges), each with its own do from newDo (its own
// scratch), and hands each row to it; do must write only that row's own output. It records a read error on the writer and
// reports whether the pass completed.
func (w *StreamWriter) tablePass(t *StreamedTable, lo, hi int, newDo func() func(r int, row []float32)) bool {
	if w.Err != nil {
		return false
	}
	err := ForRowRanges(hi-lo, t.cols, 1, w.TestHooks.EmbedWorkers, t.newReader, func(a, b int, read RowReader) error {
		row, do := make([]float32, t.cols), newDo()
		for r := lo + a; r < lo+b; r++ {
			if err := read(r, row); err != nil {
				return err
			}
			do(r, row)
		}
		return nil
	})
	if err != nil {
		w.SetErr(err)
		return false
	}
	return true
}
