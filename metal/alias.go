//go:build darwin

package metal

import (
	"fmt"
	"os"
	"unsafe"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/decoder"
)

// S6 (docs/tasks/task-never-swap-2026-09.md): a dense int4 weight is copied twice at build — the
// .giw's mapping → a heap []uint32 → a StorageModeShared MTLBuffer — and the MTLBuffer copy is the
// ~2.5 GB IOAccelerator dirty term of a Metal M26 load. weightAlias replaces the second copy for a
// canonical int4 array that lives in the mapping: it wraps the array's pages in a no-copy MTLBuffer
// (newBufferWithBytesNoCopy) and binds the array at its offset inside that window, so the GPU reads the
// nibbles straight out of the page cache. Nothing else changes: same bytes, same kernels, same bind
// order — only who owns the memory.
//
// One buffer per tensor, over the page-aligned window that encloses it (Model.MmapAliasWindow), rather
// than one over the whole mapping: a 16 GB .giw exceeds a 16 GB Mac's MTLBuffer size limit, and a
// window per tensor needs no page padding in the file (v12 already 16-byte-aligns every array, which is
// all the kernels' vector loads need). Scales are NOT aliased: the file stores f32 and the kernels read
// f16, so they are still converted into a small buffer (~1/8 of the nibble bytes).
//
// MEASURED CONSEQUENCE, not a free lunch: Metal wires every page of a no-copy buffer that a command
// buffer touches, and keeps them wired (memory: metal-nocopy-wiring-deadend). A decode step touches every
// dense weight, so the aliased pages are wired — file-backed and never anonymous, but not evictable. The
// win is the duplicate (anonymous MTLBuffer copy + the file's own page cache) becoming one copy, not
// weights that can be dropped under pressure.
//
// Opt-in (GOINFER_METAL_ALIAS=1) until every gate in S6's registered rule passes; off, this type is nil and
// int4Buf is byte-for-byte what it was.
type weightAlias struct {
	m    *decoder.Model
	page int

	tensors  int   // tensors wrapped in place
	aliased  int64 // nibble bytes served from the mapping
	copied   int   // eligible int4 tensors that were not in the mapping (heap-backed) and took the copy path
	copiedB  int64
	declined int // in the mapping but not 16-byte-aligned (an older .giw layout), so copied

	groups      int   // fused groups (q|k|v, gate|up) wrapped as ONE buffer over their contiguous nibbles
	groupBytes  int64 // nibble bytes those groups cover
	nonAdjacent int   // fused groups whose members are in the mapping but not back to back (a pre-v13 layout, or a K=V layer's Q|K|K)
}

// newWeightAlias returns nil unless the opt-in is set AND the model is backed by a .giw mapping.
func newWeightAlias(m *decoder.Model) *weightAlias {
	if os.Getenv("GOINFER_METAL_ALIAS") != "1" || m == nil || m.GiwPath() == "" {
		return nil
	}
	return &weightAlias{m: m, page: os.Getpagesize()}
}

// nibbles returns w's canonical packed int4 array as a Metal buffer bound at its offset in the mapping,
// or ok=false when it cannot be aliased (not a canonical group-32 int4, not in the mapping, or not
// 16-byte aligned) — the caller then takes the copy path unchanged.
func (a *weightAlias) nibbles(d *Device, w *linalg.WeightMat) (Buffer, bool) {
	if a == nil {
		return Buffer{}, false
	}
	q4, _, group, ok := w.Int4()
	if !ok || group != 32 || len(q4) == 0 {
		return Buffer{}, false
	}
	base, n, off, ok := a.m.MmapAliasWindow(q4, a.page)
	if !ok {
		a.copied++
		a.copiedB += int64(len(q4))
		return Buffer{}, false
	}
	if off%16 != 0 {
		a.declined++
		return Buffer{}, false
	}
	a.tensors++
	a.aliased += int64(len(q4))
	return d.NewBufferNoCopy(base, n).At(off), true
}

// concatNibbles is nibbles for a FUSED group: it returns ONE buffer bound over the members' nibble bytes
// when they are a single contiguous run in the mapping in the order the fused kernel wants — which is
// exactly what a v13 metal-target .giw's group block guarantees for q|k|v and gate|up (decoder/serialize.go,
// fusedGroup). Detected, not assumed: a pre-v13 file, a K=V layer's q|k|k (k twice), an int8-requantized
// member or a heap-backed one all fail the check and the caller takes the copy path unchanged. The bytes
// are the same as int4Concat's concatenation (row-major nibbles, member after member), so the kernel
// cannot tell the difference.
func (a *weightAlias) concatNibbles(d *Device, wms []*linalg.WeightMat) (Buffer, bool) {
	if a == nil || len(wms) == 0 {
		return Buffer{}, false
	}
	first := make([][]byte, len(wms))
	total := 0
	for i, w := range wms {
		q4, _, group, ok := w.Int4()
		if !ok || group != 32 || len(q4) == 0 {
			return Buffer{}, false
		}
		first[i] = q4
		total += len(q4)
	}
	for i := range first {
		if _, ok := a.m.MmapByteOffset(first[i]); !ok {
			a.copied++
			a.copiedB += int64(total)
			return Buffer{}, false
		}
		if i > 0 {
			prev := first[i-1]
			if uintptr(unsafe.Pointer(&prev[0]))+uintptr(len(prev)) != uintptr(unsafe.Pointer(&first[i][0])) {
				a.nonAdjacent++
				return Buffer{}, false
			}
		}
	}
	run := unsafe.Slice(&first[0][0], total) // one slice over the whole contiguous run
	base, n, off, ok := a.m.MmapAliasWindow(run, a.page)
	if !ok {
		a.copied++
		a.copiedB += int64(total)
		return Buffer{}, false
	}
	if off%16 != 0 {
		a.declined++
		return Buffer{}, false
	}
	a.groups++
	a.groupBytes += int64(total)
	a.aliased += int64(total)
	return d.NewBufferNoCopy(base, n).At(off), true
}

// int4ConcatA is int4Concat with the fused-group alias: when the members' nibbles are one contiguous run in
// the mapping the nibbles are bound in place and only the f16 scales (concatenated in member order, converted
// exactly as int4Concat converts them) are built; otherwise it is int4Concat unchanged.
func int4ConcatA(d *Device, a *weightAlias, wms ...*linalg.WeightMat) (Buffer, Buffer) {
	for _, w := range wms {
		if w.Cols()%32 != 0 { // int4Concat owns the K%32 panic (audit M-10)
			return int4Concat(d, wms...)
		}
	}
	nib, ok := a.concatNibbles(d, wms)
	if !ok {
		return int4Concat(d, wms...)
	}
	nScales := 0
	for _, w := range wms {
		_, q4s, _, _ := w.Int4()
		nScales += len(q4s)
	}
	scales := make([]uint16, 0, nScales)
	for _, w := range wms {
		_, q4s, _, _ := w.Int4()
		for _, s := range q4s {
			scales = append(scales, f32ToF16(s))
		}
	}
	return nib, NewBufferU16s(d, scales)
}

// summary is the banner line S6 asks for: the number a user would otherwise never see.
func (a *weightAlias) summary() string {
	if a == nil {
		return ""
	}
	return fmt.Sprintf("[metal] weights aliased from the .giw mapping (GOINFER_METAL_ALIAS=1): %.0f MB of int4 nibbles not copied "+
		"(%d single tensors + %d fused groups = %.0f MB); copied as before: %d heap-backed, %d unaligned, %d fused groups not adjacent\n",
		float64(a.aliased)/(1<<20), a.tensors, a.groups, float64(a.groupBytes)/(1<<20), a.copied, a.declined, a.nonAdjacent)
}
