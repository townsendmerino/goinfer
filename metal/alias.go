//go:build darwin

package metal

import (
	"fmt"
	"os"

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

// summary is the banner line S6 asks for: the number a user would otherwise never see.
func (a *weightAlias) summary() string {
	if a == nil {
		return ""
	}
	return fmt.Sprintf("[metal] weights aliased from the .giw mapping (GOINFER_METAL_ALIAS=1): %d tensors, %.0f MB of int4 nibbles "+
		"not copied; %d heap-backed and %d unaligned tensors copied as before\n",
		a.tensors, float64(a.aliased)/(1<<20), a.copied, a.declined)
}
