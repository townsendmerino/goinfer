// Package giwkv implements the versioned binary ".giw-kv" format for a session's CPU KV cache and token list, a sibling
// of the ".giw" weight bundle (internal/runtime/giwcodec), so a prefilled conversation can be persisted and restored
// instead of re-prefilled. decoder's Session.Snapshot and Model.LoadSession are its callers; it never imports decoder.
//
// Same discipline as the weight bundle: magic + version + a geometry guard + CRC, with a typed *SnapshotError on any
// mismatch so a stale or foreign snapshot is skipped, not fatal. Keys and values are written as plain little-endian
// payloads (the snapshot is an offline artifact, not the hot path).
//
// Format v3 (little-endian throughout, written with giwcodec.Writer and read with giwcodec.Reader):
//
//	magic     [5]byte = "GINFK"
//	version   uint32 = 3
//	id        str      (conversation/model identity, for tooling; not validated)
//	adapter   str      (the compute-time LoRA adapter the KV was built under; "" = base)
//	numLayers uint32   \
//	kvDim     uint32    |
//	window    uint32    | geometry guard — must match the loading model's cache
//	headDim   uint32    | (else the KV is for a different architecture)
//	manualPos uint8     |
//	quant     uint8    /  (0=f32, 1=int8, 2=f16)
//	pos       uint32   (stored position count)
//	tokens    u32 len + len*uint32
//	  per layer, in cache-structure order (the loader rebuilds the same rings +
//	  quant from the model, so layers carry no tag):
//	    ring layer:  count u32, then {i8 kq; i8 vq; f32 ksc; f32 vsc} if int8
//	                 else {f32 k; f32 v}   — the W-slot window
//	    global:      {i8 keysQ; i8 valsQ; f32 keyScale; f32 valScale} if int8
//	                 else {f32 keys; f32 vals}   (KV-shared layers serialize len 0)
//	crc       uint32   (CRC32-IEEE over every preceding byte)
//
// v1 (f32-global-only) and v2 (no adapter field) blobs are rejected by the version guard —
// snapshots are a regenerable cache, so a version bump just triggers a cold prefill.
package giwkv

import (
	"encoding/binary"
	"fmt"
	"hash/crc32"

	"github.com/townsendmerino/goinfer/internal/runtime/cpuengine"
	"github.com/townsendmerino/goinfer/internal/runtime/giwcodec"
)

const (
	// Magic opens every session snapshot.
	Magic = "GINFK"
	// Version is the snapshot format version this build writes and reads.
	Version = 3 // v2: ring windowed persistence × {f32 | int8} payload; v3: + the adapter name
)

// SnapshotError is returned on any magic/version/geometry/CRC mismatch — distinct so callers can skip a stale or
// foreign snapshot and fall back to a cold prefill rather than treating it as fatal.
type SnapshotError struct{ Reason string }

func (e *SnapshotError) Error() string { return "decoder: kv snapshot: " + e.Reason }

// Header is a session snapshot's header fields, as ReadHeader reads them.
type Header struct {
	ID, Adapter                       string
	NumLayers, KVDim, Window, HeadDim int
	ManualPos                         bool
	Quant                             cpuengine.KVQuant
	Pos                               int
}

// Encode serializes c and tokens to a snapshot blob. id is stored for tooling; adapter is the registry name of the
// compute-time adapter the KV was built under ("" for the base). It returns nil for a cache whose global layers do not
// all have the uniform kvDim width: the format records no per-global-layer stride, so such a cache cannot round-trip
// and the caller cold-prefills. f16ByteAsF32 writes an f16 cache's storage byte as f32's value, the planted defect the
// decoder's f16 snapshot test uses to prove the loader's geometry guard; production passes false.
func Encode(c *cpuengine.Cache, id, adapter string, tokens []int, f16ByteAsF32 bool) []byte {
	// Gemma-4's global (append-forever) layers carry per-layer KV widths (E2B/E4B give some layers a different NumKVHeads·HeadDim
	// than the cache's uniform KVDim). The loader restores every global layer to kvDim and the first TruncateTo would mis-slice
	// the odd-width ones, silently wrong on the serve session path. Ring-layer width mismatches are caught loudly by Decode (it
	// rejects a ring stride != kvDim), so only globals leak.
	for l := 0; l < c.NumLayers; l++ {
		isRing := c.Rings[l] != nil
		if !isRing && c.Stride[l] != 0 && c.Stride[l] != c.RowStride(c.KVDim) {
			return nil
		}
	}
	wr := &giwcodec.Writer{}
	wr.Raw([]byte(Magic))
	wr.U32(Version)
	wr.Str(id)
	wr.Str(adapter)
	wr.U32(uint32(c.NumLayers))
	wr.U32(uint32(c.KVDim))
	wr.U32(uint32(c.Window))
	wr.U32(uint32(c.HeadDim))
	manualPos := byte(0)
	if c.ManualPos {
		manualPos = 1
	}
	quantByte := byte(c.Quant)
	if f16ByteAsF32 && c.Quant == cpuengine.KVQF16 {
		quantByte = byte(cpuengine.KVQF32) // planted defect: the storage byte left at f32's value
	}
	wr.Raw([]byte{manualPos, quantByte})
	wr.U32(uint32(c.PosN))
	WriteInts(wr, tokens)
	// Per layer, in cache-structure order (the loader reconstructs the same rings +
	// quant from the model, so no per-layer tags are needed): ring layers store
	// their W-slot window + logical count; global layers append-forever. Each
	// payload is f32 or int8 (+ per-head scales) per the cache's quant mode. Empty
	// fields (KV-shared / never-written rings) serialize as len 0.
	for l := 0; l < c.NumLayers; l++ {
		if r := c.Rings[l]; r != nil {
			encodeRing(wr, r, c.Quant)
		} else if c.Quant == cpuengine.KVQF16 { // compact rows: the format holds kvDim-wide rows whatever the cache's pad
			n := c.StoredRows(l, c.KVDim)
			wr.Scales16(cpuengine.CompactRows(c.KeysH[l], n, c.KVDim, c.RowStride(c.KVDim)))
			wr.Scales16(cpuengine.CompactRows(c.ValsH[l], n, c.KVDim, c.RowStride(c.KVDim)))
		} else if c.Quant == cpuengine.KVI8 {
			wr.I8(c.KeysQ[l])
			wr.I8(c.ValsQ[l])
			wr.F32Body(c.KeyScale[l])
			wr.F32Body(c.ValScale[l])
		} else {
			n := c.StoredRows(l, c.KVDim)
			wr.F32Body(cpuengine.CompactRows(c.KeysBuf[l], n, c.KVDim, c.RowStride(c.KVDim)))
			wr.F32Body(cpuengine.CompactRows(c.ValsBuf[l], n, c.KVDim, c.RowStride(c.KVDim)))
		}
	}
	wr.U32(crc32.ChecksumIEEE(wr.Buf))
	return wr.Buf
}

// encodeRing writes one ring layer compactly: only the live window [count-W, count) (min(count,W) rows), unwrapped into
// absolute order — not the full W physical slots (which over-store a short, un-wrapped session). Decode re-wraps.
func encodeRing(wr *giwcodec.Writer, r *cpuengine.Ring, quant cpuengine.KVQuant) {
	lo := max(0, r.Count-r.W)
	nLive := r.Count - lo
	wr.U32(uint32(r.Count))
	wr.U32(uint32(nLive))
	wr.U32(uint32(r.Stride))
	if r.Stride == 0 || nLive == 0 {
		return // never-written ring: count/nLive/stride suffice
	}
	st := r.Stride
	switch quant {
	case cpuengine.KVQF16:
		kb, vb := make([]uint16, nLive*st), make([]uint16, nLive*st)
		for i, p := 0, lo; p < r.Count; i, p = i+1, p+1 {
			so, do := (p%r.W)*st, i*st
			copy(kb[do:do+st], r.KH[so:so+st])
			copy(vb[do:do+st], r.VH[so:so+st])
		}
		wr.Scales16(kb)
		wr.Scales16(vb)
	case cpuengine.KVI8:
		nKV := st / r.HeadDim
		kb, vb := make([]int8, nLive*st), make([]int8, nLive*st)
		ks, vs := make([]float32, nLive*nKV), make([]float32, nLive*nKV)
		for i, p := 0, lo; p < r.Count; i, p = i+1, p+1 {
			so, do := (p%r.W)*st, i*st
			copy(kb[do:do+st], r.KQ[so:so+st])
			copy(vb[do:do+st], r.VQ[so:so+st])
			sso, sdo := (p%r.W)*nKV, i*nKV
			copy(ks[sdo:sdo+nKV], r.KSC[sso:sso+nKV])
			copy(vs[sdo:sdo+nKV], r.VSC[sso:sso+nKV])
		}
		wr.I8(kb)
		wr.I8(vb)
		wr.F32Body(ks)
		wr.F32Body(vs)
	default:
		kb, vb := make([]float32, nLive*st), make([]float32, nLive*st)
		for i, p := 0, lo; p < r.Count; i, p = i+1, p+1 {
			so, do := (p%r.W)*st, i*st
			copy(kb[do:do+st], r.K[so:so+st])
			copy(vb[do:do+st], r.V[so:so+st])
		}
		wr.F32Body(kb)
		wr.F32Body(vb)
	}
}

// ReadHeader reads a session snapshot's magic, version and header, and checks the payload's CRC before any offset or
// length in it is trusted. The reader it returns is positioned after the header.
func ReadHeader(data []byte) (*giwcodec.Reader, Header, error) {
	r := &giwcodec.Reader{Data: data}
	if got := r.RawN(len(Magic)); string(got) != Magic {
		return nil, Header{}, &SnapshotError{fmt.Sprintf("bad magic %q (want %q)", got, Magic)}
	}
	if v := r.U32(); v != Version {
		return nil, Header{}, &SnapshotError{fmt.Sprintf("format version %d, this build reads %d", v, Version)}
	}
	var h Header
	h.ID = r.Str()
	h.Adapter = r.Str()
	h.NumLayers, h.KVDim, h.Window, h.HeadDim = int(r.U32()), int(r.U32()), int(r.U32()), int(r.U32())
	if !r.Need(2) {
		return nil, Header{}, &SnapshotError{"truncated header"}
	}
	h.ManualPos = r.Data[r.Off] == 1
	h.Quant = cpuengine.KVQuant(r.Data[r.Off+1])
	r.Off += 2
	h.Pos = int(r.U32())
	if r.Err != nil {
		return nil, Header{}, &SnapshotError{"truncated header"}
	}

	// CRC over the whole payload before trusting offsets/lengths: catches truncation and corruption up front.
	if len(data) < 4 {
		return nil, Header{}, &SnapshotError{"too short"}
	}
	body, want := data[:len(data)-4], binary.LittleEndian.Uint32(data[len(data)-4:])
	if got := crc32.ChecksumIEEE(body); got != want {
		return nil, Header{}, &SnapshotError{fmt.Sprintf("CRC mismatch (got %08x want %08x) — corrupt or truncated", got, want)}
	}
	return r, h, nil
}

// Decode reads the token list and per-layer KV that follow h into a cache from newCache, which must build the loading
// model's cache sized for the given position count (0 allocates no capacity). It validates every blob-controlled
// dimension against the model's geometry before allocating. On success the cache newCache last returned holds the
// restored KV with PosN set; on failure it returns a *SnapshotError and that cache must be discarded.
func Decode(r *giwcodec.Reader, h Header, newCache func(capHint int) *cpuengine.Cache) ([]int, error) {
	numLayers, kvDim, window, headDim := h.NumLayers, h.KVDim, h.Window, h.HeadDim
	manualPos, quant, pos := h.ManualPos, h.Quant, h.Pos

	// numLayers/kvDim/pos are blob-controlled and feed newCache(pos), which allocates pos × the model's KV footprint, so an
	// inflated pos would makeslice terabytes before the geometry guard below. Reject implausible header dims, zero included:
	// numLayers == 0 or kvDim == 0 makes the per-position footprint 0 and disables any bound derived from it, so a 20-byte header in
	// -session-dir could become a fatal out-of-memory at server boot.
	if numLayers <= 0 || numLayers > giwcodec.MaxSerializedLayers || kvDim <= 0 || kvDim > 1<<24 ||
		headDim <= 0 || window < 0 || pos < 0 {
		return nil, &SnapshotError{"implausible header dims"}
	}

	// Geometry first, then the bound, then the allocation. newCache(0) derives the same geometry from the arch and allocates no
	// capacity, so comparing against it is free; once it passes, blob and model geometry are equal by construction and the bound below
	// is in the units the allocation actually uses. Never allocate newCache(pos) before this comparison: the out-of-memory would
	// precede the check that rejects the blob.
	ref := newCache(0)
	if numLayers != ref.NumLayers || kvDim != ref.KVDim || window != ref.Window ||
		headDim != ref.HeadDim || manualPos != ref.ManualPos || quant != ref.Quant {
		return nil, &SnapshotError{fmt.Sprintf(
			"geometry mismatch: snapshot {layers:%d kvDim:%d window:%d headDim:%d manualPos:%v quant:%d} vs model {layers:%d kvDim:%d window:%d headDim:%d manualPos:%v quant:%d}",
			numLayers, kvDim, window, headDim, manualPos, quant, ref.NumLayers, ref.KVDim, ref.Window, ref.HeadDim, ref.ManualPos, ref.Quant)}
	}
	// Tokens before the allocation, and a ceiling on the allocation itself. Do not bound pos by the payload size
	// (len(data)/(numLayers·kvDim)): a well-formed body can carry zero KV bytes while pos > 0 (a never-written ring serialises as
	// count/nLive/stride only, a KV-shared layer stores nothing, a ring stores only min(count, W) rows), so such a bound rejects valid
	// snapshots, including any session longer than ~8·W on an all-sliding-window model. The guard exists to stop a blob-controlled
	// pos from driving a huge newCache(pos), and two checks serve that without assuming payload sizes:
	//
	//   1. pos == len(tokens), checked before the allocation. ReadInts refuses to allocate more than the body holds, so a 20-byte
	//      header yields no tokens and is rejected having allocated nothing; the format guarantees this invariant.
	//   2. an explicit ceiling on the bytes the cache would occupy (below): bound the allocation itself, not a proxy that
	//      legitimate blobs fail.
	tokens := ReadInts(r)
	// len(tokens) is blob-controlled and must equal pos. Too many tokens is the quiet failure: the decoder's rewindForReuse computes
	// matched > PosN, TruncateTo treats the out-of-range target as a no-op, and the reuse reports an exact match on a cache never
	// rewound, a session silently continuing from the wrong KV.
	if len(tokens) != pos {
		return nil, &SnapshotError{fmt.Sprintf(
			"tokens: %d ids for pos %d — a longer token list makes rewindForReuse report an "+
				"exact match it never performed", len(tokens), pos)}
	}
	// K and V, f32 or int8 plus scales; 8 B per position per kvDim is the f32 upper bound and the
	// one to budget against. Refused before makeslice ever sees it.
	if want := int64(pos) * int64(ref.NumLayers) * int64(ref.KVDim) * 8; want > giwcodec.MaxSnapshotCacheBytes {
		return nil, &SnapshotError{fmt.Sprintf(
			"snapshot would allocate %d B of KV cache (pos %d × %d layers × %d kvDim), past the "+
				"%d B ceiling — corrupt or malicious", want, pos, ref.NumLayers, ref.KVDim, giwcodec.MaxSnapshotCacheBytes)}
	}
	ref = newCache(pos)
	// Same per-layer structure order as Encode; ref already has the right rings
	// + quant from newCache, so fill its fields in place.
	for l := range numLayers {
		var err *SnapshotError
		if rr := ref.Rings[l]; rr != nil {
			err = decodeRing(r, rr, quant, kvDim)
		} else {
			err = decodeGlobal(r, ref, l, quant, pos, kvDim, headDim)
		}
		if err != nil {
			return nil, err
		}
	}
	if r.Err != nil {
		return nil, &SnapshotError{"truncated body: " + r.Err.Error()}
	}
	ref.PosN = pos
	return tokens, nil
}

// decodeRing restores one ring layer's live window into rr, re-wrapping it into the W physical slots.
func decodeRing(r *giwcodec.Reader, rr *cpuengine.Ring, quant cpuengine.KVQuant, kvDim int) *SnapshotError {
	rr.Count = int(r.U32())
	nLive, st := int(r.U32()), int(r.U32())
	rr.Stride = st
	if st == 0 || nLive == 0 {
		// A never-written ring must also have count 0: count>0 with stride/nLive 0 is a state the writer never emits (it leaves rr.Count
		// set but the k/v buffers unallocated, so the first decode reads a nil ring and panics). The return skips the geometry check
		// below, so guard it here.
		if rr.Count != 0 {
			return &SnapshotError{"inconsistent ring geometry: nonzero count on an unwritten ring"}
		}
		return nil // never-written ring
	}
	// st/nLive/count are blob-controlled and drive make([]…, rr.W·st) and the slice copies below. A ring stores exactly kvDim per
	// position; reject a mismatched stride, nLive past the window, or a count/nLive the payload cannot back, or the make OOMs or the
	// copies slice-panic. rr.HeadDim is model-derived (>0), so nKV is safe.
	if st != kvDim || rr.Count < 0 || nLive > rr.W || rr.Count < nLive {
		return &SnapshotError{"inconsistent ring geometry"}
	}
	lo := max(0, rr.Count-rr.W)
	nRows := rr.Count - lo // copy iterations = payload rows the writer emitted
	switch quant {
	case cpuengine.KVQF16:
		rr.KH, rr.VH = make([]uint16, rr.W*st), make([]uint16, rr.W*st)
		kb, vb := r.U16View(int(r.U32())), r.U16View(int(r.U32()))
		if len(kb) < nRows*st || len(vb) < nRows*st {
			return &SnapshotError{"truncated ring payload"}
		}
		for i, p := 0, lo; p < rr.Count; i, p = i+1, p+1 {
			so, do := (p%rr.W)*st, i*st
			copy(rr.KH[so:so+st], kb[do:do+st])
			copy(rr.VH[so:so+st], vb[do:do+st])
		}
	case cpuengine.KVI8:
		nKV := st / rr.HeadDim
		rr.KQ, rr.VQ = make([]int8, rr.W*st), make([]int8, rr.W*st)
		rr.KSC, rr.VSC = make([]float32, rr.W*nKV), make([]float32, rr.W*nKV)
		kb, vb, ks, vs := r.I8(), r.I8(), r.F32Body(), r.F32Body()
		if len(kb) < nRows*st || len(vb) < nRows*st || len(ks) < nRows*nKV || len(vs) < nRows*nKV {
			return &SnapshotError{"truncated ring payload"}
		}
		for i, p := 0, lo; p < rr.Count; i, p = i+1, p+1 {
			so, do := (p%rr.W)*st, i*st
			copy(rr.KQ[so:so+st], kb[do:do+st])
			copy(rr.VQ[so:so+st], vb[do:do+st])
			sso, sdo := (p%rr.W)*nKV, i*nKV
			copy(rr.KSC[sso:sso+nKV], ks[sdo:sdo+nKV])
			copy(rr.VSC[sso:sso+nKV], vs[sdo:sdo+nKV])
		}
	default:
		rr.K, rr.V = make([]float32, rr.W*st), make([]float32, rr.W*st)
		kb, vb := r.F32Body(), r.F32Body()
		if len(kb) < nRows*st || len(vb) < nRows*st {
			return &SnapshotError{"truncated ring payload"}
		}
		for i, p := 0, lo; p < rr.Count; i, p = i+1, p+1 {
			so, do := (p%rr.W)*st, i*st
			copy(rr.K[so:so+st], kb[do:do+st])
			copy(rr.V[so:so+st], vb[do:do+st])
		}
	}
	return nil
}

// decodeGlobal restores global (append-forever) layer l of c, validating every blob-controlled length.
func decodeGlobal(r *giwcodec.Reader, c *cpuengine.Cache, l int, quant cpuengine.KVQuant, pos, kvDim, headDim int) *SnapshotError {
	switch quant {
	case cpuengine.KVQF16:
		// Copied, not aliased: the cache outlives the blob it was restored from.
		c.KeysH[l] = append(c.KeysH[l][:0], r.U16View(int(r.U32()))...)
		c.ValsH[l] = append(c.ValsH[l][:0], r.U16View(int(r.U32()))...)
		if e := checkGlobalPair(l, "keysH", "valsH", len(c.KeysH[l]), len(c.ValsH[l]), pos*kvDim); e != nil {
			return e
		}
		if len(c.KeysH[l]) > 0 {
			c.Stride[l] = c.RowStride(kvDim)
			c.KeysH[l], c.ValsH[l] = cpuengine.PadRows(c.KeysH[l], pos, kvDim, c.Stride[l]), cpuengine.PadRows(c.ValsH[l], pos, kvDim, c.Stride[l])
		}
	case cpuengine.KVI8:
		c.KeysQ[l], c.ValsQ[l], c.KeyScale[l], c.ValScale[l] = r.I8(), r.I8(), r.F32Body(), r.F32Body()
		// Blob-controlled lengths must be compared, as the ring path does for its stride, nLive and payload. The forward derives
		// nKeys from `keys` and then indexes `vals` at the same positions, so a vals array one row short is an out-of-range read in the
		// generation goroutine, a panic that takes the process down; the CRC does not help, since it covers the attacker's bytes. 0 is
		// legal and means a KV-shared layer, which stores nothing of its own.
		if e := checkGlobalPair(l, "keysQ", "valsQ", len(c.KeysQ[l]), len(c.ValsQ[l]), pos*kvDim); e != nil {
			return e
		}
		nKV := kvDim / headDim
		if e := checkGlobalPair(l, "keyScale", "valScale", len(c.KeyScale[l]), len(c.ValScale[l]), pos*nKV); e != nil {
			return e
		}
		if (len(c.KeysQ[l]) == 0) != (len(c.KeyScale[l]) == 0) {
			return &SnapshotError{fmt.Sprintf("layer %d keysQ/keyScale presence mismatch (%d vs %d)", l, len(c.KeysQ[l]), len(c.KeyScale[l]))}
		}
		if len(c.KeysQ[l]) > 0 {
			c.Stride[l] = kvDim // restore the per-layer KV width
		}
	default:
		// f32 returns nil for a len-0 field (KV-shared layers); keep the cache's
		// empty-but-non-nil slice so Append/Keys stay well-formed.
		if k := r.F32Body(); k != nil {
			c.KeysBuf[l] = k
		}
		if v := r.F32Body(); v != nil {
			c.ValsBuf[l] = v
		}
		// Stride is set only by Append, which Decode bypasses; without the assignment below a restored global layer has stride 0,
		// so the first TruncateTo slices it to [:0] (or attendBatchedHeads panics) while pos stays non-zero. A global layer's width is the
		// cache's uniform kvDim (geometry-guarded above); KV-shared layers keep stride 0, which TruncateTo's min() already guards. The
		// lengths are checked as in the int8 arm above.
		if e := checkGlobalPair(l, "keys", "vals", len(c.KeysBuf[l]), len(c.ValsBuf[l]), pos*kvDim); e != nil {
			return e
		}
		if len(c.KeysBuf[l]) > 0 {
			c.Stride[l] = c.RowStride(kvDim)
			c.KeysBuf[l], c.ValsBuf[l] = cpuengine.PadRows(c.KeysBuf[l], pos, kvDim, c.Stride[l]), cpuengine.PadRows(c.ValsBuf[l], pos, kvDim, c.Stride[l])
		}
	}
	return nil
}

// WriteInts writes a length-prefixed []int as uint32s (token ids are non-negative).
func WriteInts(w *giwcodec.Writer, s []int) {
	w.U32(uint32(len(s)))
	for _, v := range s {
		w.U32(uint32(v))
	}
}

// ReadInts reads a length-prefixed []int written by WriteInts. It returns nil for a zero length, and for a length the
// remaining body cannot hold, so a forged count allocates nothing.
func ReadInts(r *giwcodec.Reader) []int {
	n := int(r.U32())
	if n == 0 || !r.Need(n*4) {
		return nil
	}
	out := make([]int, n)
	for i := range out {
		out[i] = int(r.U32())
	}
	return out
}

// checkGlobalPair validates a paired global-layer K/V (or scale) array set: both must be 0 (KV-shared) or both equal want.
func checkGlobalPair(layer int, nameA, nameB string, gotA, gotB, want int) *SnapshotError {
	if e := checkGlobalLen(layer, nameA, gotA, want); e != nil {
		return e
	}
	if e := checkGlobalLen(layer, nameB, gotB, want); e != nil {
		return e
	}
	if (gotA == 0) != (gotB == 0) {
		return &SnapshotError{fmt.Sprintf("layer %d %s/%s presence mismatch (%d vs %d)", layer, nameA, nameB, gotA, gotB)}
	}
	return nil
}

// checkGlobalLen validates one blob-controlled global-layer array against the length the header's pos and the model's geometry
// imply. 0 is legal: a KV-shared layer stores nothing of its own, and the cache keeps an empty-but-well-formed slice for it. It is
// split out rather than inlined so the two storage arms cannot drift apart.
func checkGlobalLen(layer int, name string, got, want int) *SnapshotError {
	if got != 0 && got != want {
		return &SnapshotError{fmt.Sprintf(
			"layer %d %s: %d entries, header implies %d — corrupt or malicious; the forward "+
				"derives its key count from one of these arrays and indexes the others at the "+
				"same positions", layer, name, got, want)}
	}
	return nil
}
