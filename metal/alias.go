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
// WHAT THE ALIASED PAGES ARE (measured, docs/measurements/s6-alias-2026-09-24.md, "MAP_SHARED"): IOKit wires a
// no-copy buffer's pages with write intent. Over a MAP_PRIVATE mapping that makes every GPU-read page a wired
// ANONYMOUS copy-on-write copy — the 1.5B served one request with +40,430 COW faults for 625 MB aliased — so
// aliasing saved nothing, and fork() then copied the whole mapping (the M26 collapse). decoder.Load therefore
// maps a .giw MAP_SHARED on darwin (decoder/giwmap_darwin.go): the GPU reads the file's own page-cache pages
// (+132 COW faults, background), wired only while a command buffer uses them, and a fork is cheap.
//
// Opt-in (GOINFER_METAL_ALIAS=1) until every gate in S6's registered rule passes; off, this type is nil and
// int4Buf is byte-for-byte what it was.
type weightAlias struct {
	m    *decoder.Model
	page int

	tensors int   // int4 tensors wrapped in place
	aliased int64 // int4 nibble bytes served from the mapping

	int8Tensors int   // int8 matrices (the LM head) whose codes are wrapped in place
	scaleBytes  int64 // f16 group-scale bytes served from the mapping (v14 metal-target files)
	int8Bytes   int64 // int8 code bytes served from the mapping
	copied      int   // eligible int4 tensors that were not in the mapping (heap-backed) and took the copy path
	copiedB     int64
	declined    int // in the mapping but not 16-byte-aligned (an older .giw layout), so copied

	groups      int   // fused groups (q|k|v, gate|up) wrapped as ONE buffer over their contiguous nibbles
	groupBytes  int64 // nibble bytes those groups cover
	nonAdjacent int   // fused groups whose members are in the mapping but not back to back (a pre-v13 layout, or a K=V layer's Q|K|K)
}

// newWeightAlias returns nil unless the opt-in is set (GOINFER_METAL_ALIAS=1; "force" is accepted as a
// synonym for scripts written while a size guard existed) AND the model is backed by a .giw mapping.
//
// There is deliberately no size limit. One existed from 2026-09-24 until the same day: on gemma4-26b the
// aliased arm paged the whole server out at its first request, 3 of 3 times. The cause was not aliasing's
// memory but fork(): a fork while Metal had any page of the MAP_PRIVATE mapping wired copied the entire
// 15 GB mapping, and the swap guard forked every 2 s. decoder.Load now marks the mapping VM_INHERIT_NONE and
// the guard reads swap with a bare sysctl; with both, four interleaved M26 arms (two aliased) ran clean with
// swap flat (docs/measurements/m26-alias-fork-collapse-2026-09-24.md).
func newWeightAlias(m *decoder.Model) *weightAlias {
	mode := os.Getenv("GOINFER_METAL_ALIAS")
	if (mode != "1" && mode != "force") || m == nil || m.GiwPath() == "" {
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
	return a.window(d, q4)
}

// int8Codes is nibbles for an int8 (W8A8) matrix: its codes are one contiguous array in the .giw, 16-byte
// aligned since weights format v12, and the kernels read them as the same signed bytes, so they can be bound
// in place exactly like int4 nibbles. On the 7B this is the LM head: 519.8 MB of the 911 MB the Metal build
// still copied after int4 aliasing (measured with a per-caller buffer ledger, 2026-09-24).
func (a *weightAlias) int8Codes(d *Device, w *linalg.WeightMat) (Buffer, bool) {
	if a == nil {
		return Buffer{}, false
	}
	q8, _, _, ok := w.Int8()
	if !ok || len(q8) == 0 {
		return Buffer{}, false
	}
	buf, ok := a.window(d, unsafe.Slice((*byte)(unsafe.Pointer(&q8[0])), len(q8)))
	if ok {
		a.int8Tensors++
		a.int8Bytes += int64(len(q8))
		a.tensors-- // counted separately from the int4 singles
		a.aliased -= int64(len(q8))
	}
	return buf, ok
}

// window binds b in place: ONE no-copy buffer over b's page-aligned window of the mapping, bound at b's
// offset in it. ok=false (and the caller copies) when b is not in the mapping or not 16-byte aligned.
func (a *weightAlias) window(d *Device, b []byte) (Buffer, bool) {
	base, n, off, ok := a.m.MmapAliasWindow(b, a.page)
	if !ok {
		a.copied++
		a.copiedB += int64(len(b))
		return Buffer{}, false
	}
	if off%16 != 0 {
		a.declined++
		return Buffer{}, false
	}
	a.tensors++
	a.aliased += int64(len(b))
	return d.NewBufferNoCopy(base, n).At(off), true
}

// int8BufA is int8Buf with the alias: the int8 codes bound in place when they are in the mapping, the
// per-row f32 scales (a few hundred KB) copied as before.
func int8BufA(d *Device, a *weightAlias, w *linalg.WeightMat) (Buffer, Buffer, error) {
	if codes, ok := a.int8Codes(d, w); ok {
		_, sc, _, _ := w.Int8()
		return codes, NewBufferFloats(d, sc), nil
	}
	return int8Buf(d, w)
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

// scales16 binds the f16 group scales of wms (one tensor, or a fused group in member order) in place, when the
// model was loaded from a v14 metal-target .giw that stores them (decoder.Model.Int4ScalesF16) AND they form
// one contiguous run in the mapping — which the file guarantees for a group and for a single tensor. The file
// stores exactly F16Bits of each f32 scale, the conversion this package applies at load (f32ToF16 IS
// F16Bits), so the kernels read the same bits either way. ok=false (and the caller converts) otherwise.
func (a *weightAlias) scales16(d *Device, wms []*linalg.WeightMat) (Buffer, bool) {
	if a == nil || len(wms) == 0 {
		return Buffer{}, false
	}
	parts := make([][]uint16, len(wms))
	total := 0
	for i, w := range wms {
		f, ok := a.m.Int4ScalesF16(w)
		if !ok || len(f) == 0 {
			return Buffer{}, false
		}
		if i > 0 {
			prev := parts[i-1]
			if uintptr(unsafe.Pointer(&prev[0]))+uintptr(2*len(prev)) != uintptr(unsafe.Pointer(&f[0])) {
				return Buffer{}, false
			}
		}
		parts[i] = f
		total += len(f)
	}
	run := unsafe.Slice((*byte)(unsafe.Pointer(&parts[0][0])), 2*total)
	base, n, off, ok := a.m.MmapAliasWindow(run, a.page)
	if !ok || off%16 != 0 {
		return Buffer{}, false
	}
	a.scaleBytes += int64(2 * total)
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
	if sc, ok := a.scales16(d, wms); ok { // v14 metal-target file: the members' f16 scales are one run too
		return nib, sc
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
		"(%d single tensors + %d fused groups = %.0f MB) + %.0f MB of int8 codes (%d matrices) + %.0f MB of f16 scales; copied as before: %d heap-backed, "+
		"%d unaligned, %d fused groups not adjacent\n",
		float64(a.aliased)/(1<<20), a.tensors, a.groups, float64(a.groupBytes)/(1<<20), float64(a.int8Bytes)/(1<<20), a.int8Tensors,
		float64(a.scaleBytes)/(1<<20), a.copied, a.declined, a.nonAdjacent)
}
