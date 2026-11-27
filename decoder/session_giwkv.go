package decoder

import (
	"encoding/binary"
	"fmt"
	"github.com/townsendmerino/goinfer/internal/runtime/cpuengine"
	"github.com/townsendmerino/goinfer/internal/runtime/giwcodec"
	"hash/crc32"
)

// This file defines a versioned binary format for a Session's KV cache + token
// list (a ".giw-kv" sibling of the ".giw" weight bundle in serialize.go), so a
// prefilled conversation can be persisted to disk and restored — surviving a
// server restart, or handed between processes — instead of re-prefilling the
// whole history from scratch. cmd/serve uses it to checkpoint its SessionLRU.
//
// Same discipline as SerializeWeights: magic + version + a geometry guard + CRC,
// with a typed error on any mismatch so a stale/foreign snapshot is skipped, not
// fatal. Keys/values are written as plain little-endian float32 (the snapshot is
// an offline artifact, not the hot path — gzip-wrap the bytes if size matters).
//
// Format v3 (little-endian throughout, reusing serialize.go's giwWriter/giwReader):
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

const (
	kvSnapMagic   = "GINFK"
	kvSnapVersion = 3 // v2: ring windowed persistence × {f32 | int8} payload; v3: + the adapter name
)

// SnapshotError is returned by Model.LoadSession on any magic/version/geometry/
// CRC mismatch — distinct so callers can skip a stale or foreign snapshot and
// fall back to a cold prefill rather than treating it as fatal.
type SnapshotError struct{ Reason string }

func (e *SnapshotError) Error() string { return "decoder: kv snapshot: " + e.Reason }

// Snapshot serializes the session's KV cache and token sequence to a portable
// blob. id is an opaque identity string (e.g. the model name or a conversation
// key) stored for tooling. The blob is self-describing and CRC-guarded; load it
// with Model.LoadSession on a model of the same architecture.
func (s *Session) Snapshot(id string) []byte {
	return s.snapshot(id, false)
}

func (s *Session) snapshot(id string, f16ByteAsF32 bool) []byte {
	c := s.cache
	// Refuse the state this format does not carry: every kind whose snapshot cell in the cache-state grid (cachestate.go) is
	// "refused", i.e. recurrent state (DeltaNet, Mamba-2, LFM2's conv window, KDA), the MLA latent, and the multimodal image blocks
	// and m-RoPE positions. Restoring any of them would start from zeroed or empty state and continue silently wrong, so Snapshot
	// returns nil (the caller skips and cold-prefills). The grid is read, not a list re-typed here: a hand-listed set once missed
	// LFM2 and a session was restored warm with empty conv windows. All other families serialize fully below, including ring
	// (windowed) and int8 caches.
	if c.holdsStateHandled(lcSnapshot, hRefused) {
		return nil
	}
	// Gemma-4's global (append-forever) layers carry per-layer KV widths (E2B/E4B give some layers a different NumKVHeads·HeadDim
	// than the cache's uniform c.c.KVDim). The format records no per-global-layer stride, so LoadSession restores every global layer to
	// kvDim and the first TruncateTo would mis-slice the odd-width ones, silently wrong on the serve session path. Refuse a
	// non-uniform-width cache (caller cold-prefills), as for the families above. Ring-layer width mismatches are caught loudly by
	// LoadSession (it rejects a ring stride != kvDim), so only globals leak.
	for l := 0; l < c.c.NumLayers; l++ {
		isRing := c.c.Rings[l] != nil
		if !isRing && c.c.Stride[l] != 0 && c.c.Stride[l] != c.c.RowStride(c.c.KVDim) {
			return nil
		}
	}
	// The adapter is recorded by its registry name, the only identity that survives a restart. An
	// adapter runtime without a name cannot be looked up again on load, so refuse (cold prefill).
	adapterName := ""
	if c.lora != nil {
		if c.lora.name == "" {
			return nil
		}
		adapterName = c.lora.name
	}
	wr := &giwWriter{}
	wr.Raw([]byte(kvSnapMagic))
	wr.U32(kvSnapVersion)
	wr.Str(id)
	wr.Str(adapterName)
	wr.U32(uint32(c.c.NumLayers))
	wr.U32(uint32(c.c.KVDim))
	wr.U32(uint32(c.c.Window))
	wr.U32(uint32(c.c.HeadDim))
	manualPos := byte(0)
	if c.c.ManualPos {
		manualPos = 1
	}
	quantByte := byte(c.c.Quant)
	if f16ByteAsF32 && c.c.Quant == kvqF16 {
		quantByte = byte(kvqF32) // planted defect: the storage byte left at f32's value
	}
	wr.Raw([]byte{manualPos, quantByte})
	wr.U32(uint32(c.c.PosN))
	wr.ints(s.tokens)
	// Per layer, in cache-structure order (the loader reconstructs the same rings +
	// quant from the model, so no per-layer tags are needed): ring layers store
	// their W-slot window + logical count; global layers append-forever. Each
	// payload is f32 or int8 (+ per-head scales) per the cache's quant mode. Empty
	// fields (KV-shared / never-written rings) serialize as len 0.
	for l := 0; l < c.c.NumLayers; l++ {
		if r := c.c.Rings[l]; r != nil {
			// Compact: write only the live window [count-W, count) (min(count,W)
			// rows), unwrapped into absolute order — not the full W physical slots
			// (which over-store a short, un-wrapped session). Restore re-wraps.
			lo := max(0, r.Count-r.W)
			nLive := r.Count - lo
			wr.U32(uint32(r.Count))
			wr.U32(uint32(nLive))
			wr.U32(uint32(r.Stride))
			if r.Stride == 0 || nLive == 0 {
				continue // never-written ring: count/nLive/stride suffice
			}
			st := r.Stride
			if c.c.Quant == kvqF16 {
				kb, vb := make([]uint16, nLive*st), make([]uint16, nLive*st)
				for i, p := 0, lo; p < r.Count; i, p = i+1, p+1 {
					so, do := (p%r.W)*st, i*st
					copy(kb[do:do+st], r.KH[so:so+st])
					copy(vb[do:do+st], r.VH[so:so+st])
				}
				wr.Scales16(kb)
				wr.Scales16(vb)
			} else if c.c.Quant == kvI8 {
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
			} else {
				kb, vb := make([]float32, nLive*st), make([]float32, nLive*st)
				for i, p := 0, lo; p < r.Count; i, p = i+1, p+1 {
					so, do := (p%r.W)*st, i*st
					copy(kb[do:do+st], r.K[so:so+st])
					copy(vb[do:do+st], r.V[so:so+st])
				}
				wr.F32Body(kb)
				wr.F32Body(vb)
			}
		} else if c.c.Quant == kvqF16 { // compact rows: the format holds kvDim-wide rows whatever the cache's pad
			n := c.c.StoredRows(l, c.c.KVDim)
			wr.Scales16(cpuengine.CompactRows(c.c.KeysH[l], n, c.c.KVDim, c.c.RowStride(c.c.KVDim)))
			wr.Scales16(cpuengine.CompactRows(c.c.ValsH[l], n, c.c.KVDim, c.c.RowStride(c.c.KVDim)))
		} else if c.c.Quant == kvI8 {
			wr.I8(c.c.KeysQ[l])
			wr.I8(c.c.ValsQ[l])
			wr.F32Body(c.c.KeyScale[l])
			wr.F32Body(c.c.ValScale[l])
		} else {
			n := c.c.StoredRows(l, c.c.KVDim)
			wr.F32Body(cpuengine.CompactRows(c.c.KeysBuf[l], n, c.c.KVDim, c.c.RowStride(c.c.KVDim)))
			wr.F32Body(cpuengine.CompactRows(c.c.ValsBuf[l], n, c.c.KVDim, c.c.RowStride(c.c.KVDim)))
		}
	}
	wr.U32(crc32.ChecksumIEEE(wr.Buf))
	return wr.Buf
}

// LoadSession reconstructs a *Session from a Snapshot blob, validating its KV
// geometry against this model (cache layout must match) before trusting any
// lengths. The returned session is ready to extend via Generate. On any magic/
// version/geometry/CRC mismatch it returns a *SnapshotError so the caller can
// skip it and start cold.
//
// wantID, when non-empty, must equal the id the snapshot was written with (the
// id arg to Snapshot) or the load is rejected. The geometry guard only catches a
// different *architecture*; an identity guard is what catches a same-shaped but
// different model (a different finetune, a different checkpoint file) whose KV
// would silently produce garbage. Pass "" to skip it (the id stays opaque to the
// decoder — equality is the only test).
func (m *Model) LoadSession(data []byte, wantID string) (*Session, error) {
	r, h, err := readKVSnapHeader(data)
	if err != nil {
		return nil, err
	}
	gotID, gotAdapter, numLayers, kvDim, window, headDim := h.id, h.adapter, h.numLayers, h.kvDim, h.window, h.headDim
	manualPos, quant, pos := h.manualPos, h.quant, h.pos

	// Identity guard (post-CRC, so the id is trustworthy): reject KV written for a
	// different model even if the architecture happens to match.
	if wantID != "" && gotID != wantID {
		return nil, &SnapshotError{fmt.Sprintf("model identity mismatch: snapshot %q, want %q", gotID, wantID)}
	}
	// Adapter guard (v3): the KV was projected through this adapter, so the restored session is
	// bound to the same one. If this model has no adapter by that name the KV cannot be continued
	// correctly, so it is skipped like any other stale snapshot.
	var lora *loraRuntime
	if gotAdapter != "" {
		if lora = m.adapter(gotAdapter); lora == nil {
			return nil, &SnapshotError{fmt.Sprintf("built under adapter %q, which this model has not loaded", gotAdapter)}
		}
	}

	// numLayers/kvDim/pos are blob-controlled and feed m.NewCache(pos), which allocates pos × the model's KV footprint, so an
	// inflated pos would makeslice terabytes before the geometry guard below. Reject implausible header dims, zero included:
	// numLayers == 0 or kvDim == 0 makes the per-position footprint 0 and disables any bound derived from it, so a 20-byte header in
	// -session-dir could become a fatal out-of-memory at server boot.
	if numLayers <= 0 || numLayers > maxSerializedLayers || kvDim <= 0 || kvDim > 1<<24 ||
		headDim <= 0 || window < 0 || pos < 0 {
		return nil, &SnapshotError{"implausible header dims"}
	}

	// Geometry first, then the bound, then the allocation. NewCache(0) derives the same geometry from the arch and allocates no
	// capacity, so comparing against it is free; once it passes, blob and model geometry are equal by construction and the bound below
	// is in the units the allocation actually uses. Never allocate NewCache(pos) before this comparison: the out-of-memory would
	// precede the check that rejects the blob.
	ref := m.NewCache(0)
	if numLayers != ref.c.NumLayers || kvDim != ref.c.KVDim || window != ref.c.Window ||
		headDim != ref.c.HeadDim || manualPos != ref.c.ManualPos || quant != ref.c.Quant {
		return nil, &SnapshotError{fmt.Sprintf(
			"geometry mismatch: snapshot {layers:%d kvDim:%d window:%d headDim:%d manualPos:%v quant:%d} vs model {layers:%d kvDim:%d window:%d headDim:%d manualPos:%v quant:%d}",
			numLayers, kvDim, window, headDim, manualPos, quant, ref.c.NumLayers, ref.c.KVDim, ref.c.Window, ref.c.HeadDim, ref.c.ManualPos, ref.c.Quant)}
	}
	// Tokens before the allocation, and a ceiling on the allocation itself. Do not bound pos by the payload size
	// (len(data)/(numLayers·kvDim)): a well-formed body can carry zero KV bytes while pos > 0 (a never-written ring serialises as
	// count/nLive/stride only, a KV-shared layer stores nothing, a ring stores only min(count, W) rows), so such a bound rejects valid
	// snapshots, including any session longer than ~8·W on an all-sliding-window model. The guard exists to stop a blob-controlled
	// pos from driving a huge m.NewCache(pos), and two checks serve that without assuming payload sizes:
	//
	//   1. pos == len(tokens), checked before the allocation. r.ints() refuses to allocate more than the body holds, so a 20-byte
	//      header yields no tokens and is rejected having allocated nothing; the format guarantees this invariant.
	//   2. an explicit ceiling on the bytes the cache would occupy (below): bound the allocation itself, not a proxy that
	//      legitimate blobs fail.
	tokens := r.ints()
	// len(tokens) is blob-controlled and must equal pos. Too many tokens is the quiet failure: rewindForReuse computes matched > c.c.PosN,
	// TruncateTo treats the out-of-range target as a no-op, and the reuse reports an exact match on a cache never rewound, a session
	// silently continuing from the wrong KV.
	if len(tokens) != pos {
		return nil, &SnapshotError{fmt.Sprintf(
			"tokens: %d ids for pos %d — a longer token list makes rewindForReuse report an "+
				"exact match it never performed", len(tokens), pos)}
	}
	// K and V, f32 or int8 plus scales; 8 B per position per kvDim is the f32 upper bound and the
	// one to budget against. Refused before makeslice ever sees it.
	if want := int64(pos) * int64(ref.c.NumLayers) * int64(ref.c.KVDim) * 8; want > maxSnapshotCacheBytes {
		return nil, &SnapshotError{fmt.Sprintf(
			"snapshot would allocate %d B of KV cache (pos %d × %d layers × %d kvDim), past the "+
				"%d B ceiling — corrupt or malicious", want, pos, ref.c.NumLayers, ref.c.KVDim, maxSnapshotCacheBytes)}
	}
	ref = m.NewCache(pos)
	// Same per-layer structure order as Snapshot; ref already has the right rings
	// + quant from NewCache, so fill its fields in place.
	for l := range numLayers {
		if rr := ref.c.Rings[l]; rr != nil {
			rr.Count = int(r.U32())
			nLive, st := int(r.U32()), int(r.U32())
			rr.Stride = st
			if st == 0 || nLive == 0 {
				// A never-written ring must also have count 0: count>0 with stride/nLive 0 is a state the writer never emits (it leaves rr.Count
				// set but the k/v buffers unallocated, so the first decode reads a nil ring and panics). The continue skips the geometry check
				// below, so guard it here.
				if rr.Count != 0 {
					return nil, &SnapshotError{"inconsistent ring geometry: nonzero count on an unwritten ring"}
				}
				continue // never-written ring
			}
			// st/nLive/count are blob-controlled and drive make([]…, rr.W·st) and the slice copies below. A ring stores exactly kvDim per
			// position; reject a mismatched stride, nLive past the window, or a count/nLive the payload cannot back, or the make OOMs or the
			// copies slice-panic. rr.HeadDim is model-derived (>0), so nKV is safe.
			if st != kvDim || rr.Count < 0 || nLive > rr.W || rr.Count < nLive {
				return nil, &SnapshotError{"inconsistent ring geometry"}
			}
			lo := max(0, rr.Count-rr.W)
			nRows := rr.Count - lo // copy iterations = payload rows the writer emitted
			if quant == kvqF16 {
				rr.KH, rr.VH = make([]uint16, rr.W*st), make([]uint16, rr.W*st)
				kb, vb := r.U16View(int(r.U32())), r.U16View(int(r.U32()))
				if len(kb) < nRows*st || len(vb) < nRows*st {
					return nil, &SnapshotError{"truncated ring payload"}
				}
				for i, p := 0, lo; p < rr.Count; i, p = i+1, p+1 {
					so, do := (p%rr.W)*st, i*st
					copy(rr.KH[so:so+st], kb[do:do+st])
					copy(rr.VH[so:so+st], vb[do:do+st])
				}
			} else if quant == kvI8 {
				nKV := st / rr.HeadDim
				rr.KQ, rr.VQ = make([]int8, rr.W*st), make([]int8, rr.W*st)
				rr.KSC, rr.VSC = make([]float32, rr.W*nKV), make([]float32, rr.W*nKV)
				kb, vb, ks, vs := r.I8(), r.I8(), r.F32Body(), r.F32Body()
				if len(kb) < nRows*st || len(vb) < nRows*st || len(ks) < nRows*nKV || len(vs) < nRows*nKV {
					return nil, &SnapshotError{"truncated ring payload"}
				}
				for i, p := 0, lo; p < rr.Count; i, p = i+1, p+1 {
					so, do := (p%rr.W)*st, i*st
					copy(rr.KQ[so:so+st], kb[do:do+st])
					copy(rr.VQ[so:so+st], vb[do:do+st])
					sso, sdo := (p%rr.W)*nKV, i*nKV
					copy(rr.KSC[sso:sso+nKV], ks[sdo:sdo+nKV])
					copy(rr.VSC[sso:sso+nKV], vs[sdo:sdo+nKV])
				}
			} else {
				rr.K, rr.V = make([]float32, rr.W*st), make([]float32, rr.W*st)
				kb, vb := r.F32Body(), r.F32Body()
				if len(kb) < nRows*st || len(vb) < nRows*st {
					return nil, &SnapshotError{"truncated ring payload"}
				}
				for i, p := 0, lo; p < rr.Count; i, p = i+1, p+1 {
					so, do := (p%rr.W)*st, i*st
					copy(rr.K[so:so+st], kb[do:do+st])
					copy(rr.V[so:so+st], vb[do:do+st])
				}
			}
		} else if quant == kvqF16 {
			// Copied, not aliased: the cache outlives the blob it was restored from.
			ref.c.KeysH[l] = append(ref.c.KeysH[l][:0], r.U16View(int(r.U32()))...)
			ref.c.ValsH[l] = append(ref.c.ValsH[l][:0], r.U16View(int(r.U32()))...)
			if e := checkGlobalPair(l, "keysH", "valsH", len(ref.c.KeysH[l]), len(ref.c.ValsH[l]), pos*kvDim); e != nil {
				return nil, e
			}
			if len(ref.c.KeysH[l]) > 0 {
				ref.c.Stride[l] = ref.c.RowStride(kvDim)
				ref.c.KeysH[l], ref.c.ValsH[l] = cpuengine.PadRows(ref.c.KeysH[l], pos, kvDim, ref.c.Stride[l]), cpuengine.PadRows(ref.c.ValsH[l], pos, kvDim, ref.c.Stride[l])
			}
		} else if quant == kvI8 {
			ref.c.KeysQ[l], ref.c.ValsQ[l], ref.c.KeyScale[l], ref.c.ValScale[l] = r.I8(), r.I8(), r.F32Body(), r.F32Body()
			// Blob-controlled lengths must be compared, as the ring branch above does for its stride, nLive and payload. The forward derives
			// nKeys from `keys` and then indexes `vals` at the same positions, so a vals array one row short is an out-of-range read in the
			// generation goroutine, a panic that takes the process down; the CRC does not help, since it covers the attacker's bytes. 0 is
			// legal and means a KV-shared layer, which stores nothing of its own.
			if e := checkGlobalPair(l, "keysQ", "valsQ", len(ref.c.KeysQ[l]), len(ref.c.ValsQ[l]), pos*kvDim); e != nil {
				return nil, e
			}
			nKV := kvDim / headDim
			if e := checkGlobalPair(l, "keyScale", "valScale", len(ref.c.KeyScale[l]), len(ref.c.ValScale[l]), pos*nKV); e != nil {
				return nil, e
			}
			if (len(ref.c.KeysQ[l]) == 0) != (len(ref.c.KeyScale[l]) == 0) {
				return nil, &SnapshotError{fmt.Sprintf("layer %d keysQ/keyScale presence mismatch (%d vs %d)", l, len(ref.c.KeysQ[l]), len(ref.c.KeyScale[l]))}
			}
			if len(ref.c.KeysQ[l]) > 0 {
				ref.c.Stride[l] = kvDim // restore the per-layer KV width
			}
		} else {
			// f32 returns nil for a len-0 field (KV-shared layers); keep the cache's
			// empty-but-non-nil slice so Append/Keys stay well-formed.
			if k := r.F32Body(); k != nil {
				ref.c.KeysBuf[l] = k
			}
			if v := r.F32Body(); v != nil {
				ref.c.ValsBuf[l] = v
			}
			// stride[] is set only by Append, which LoadSession bypasses; without the assignment below a restored global layer has stride 0,
			// so the first TruncateTo slices it to [:0] (or attendBatchedHeads panics) while pos stays non-zero. A global layer's width is the
			// cache's uniform kvDim (geometry-guarded above); KV-shared layers keep stride 0, which TruncateTo's min() already guards. The
			// lengths are checked as in the int8 arm above.
			if e := checkGlobalPair(l, "keys", "vals", len(ref.c.KeysBuf[l]), len(ref.c.ValsBuf[l]), pos*kvDim); e != nil {
				return nil, e
			}
			if len(ref.c.KeysBuf[l]) > 0 {
				ref.c.Stride[l] = ref.c.RowStride(kvDim)
				ref.c.KeysBuf[l], ref.c.ValsBuf[l] = cpuengine.PadRows(ref.c.KeysBuf[l], pos, kvDim, ref.c.Stride[l]), cpuengine.PadRows(ref.c.ValsBuf[l], pos, kvDim, ref.c.Stride[l])
			}
		}
	}
	if r.Err != nil {
		return nil, &SnapshotError{"truncated body: " + r.Err.Error()}
	}
	ref.c.PosN = pos
	ref.lora = lora
	return &Session{m: m, cache: ref, tokens: tokens, kvAdapter: lora}, nil
}

// kvSnapHeader is a session snapshot's header fields, as readKVSnapHeader reads them.
type kvSnapHeader struct {
	id, adapter                       string
	numLayers, kvDim, window, headDim int
	manualPos                         bool
	quant                             kvQuant
	pos                               int
}

// readKVSnapHeader reads a session snapshot's magic, version and header, and checks the payload's CRC before any offset or
// length in it is trusted. The reader it returns is positioned after the header.
func readKVSnapHeader(data []byte) (*giwReader, kvSnapHeader, error) {
	r := &giwReader{Reader: giwcodec.Reader{Data: data}}
	if got := r.RawN(len(kvSnapMagic)); string(got) != kvSnapMagic {
		return nil, kvSnapHeader{}, &SnapshotError{fmt.Sprintf("bad magic %q (want %q)", got, kvSnapMagic)}
	}
	if v := r.U32(); v != kvSnapVersion {
		return nil, kvSnapHeader{}, &SnapshotError{fmt.Sprintf("format version %d, this build reads %d", v, kvSnapVersion)}
	}
	var h kvSnapHeader
	h.id = r.Str()
	h.adapter = r.Str()
	h.numLayers, h.kvDim, h.window, h.headDim = int(r.U32()), int(r.U32()), int(r.U32()), int(r.U32())
	if !r.Need(2) {
		return nil, kvSnapHeader{}, &SnapshotError{"truncated header"}
	}
	h.manualPos = r.Data[r.Off] == 1
	h.quant = kvQuant(r.Data[r.Off+1])
	r.Off += 2
	h.pos = int(r.U32())
	if r.Err != nil {
		return nil, kvSnapHeader{}, &SnapshotError{"truncated header"}
	}

	// CRC over the whole payload before trusting offsets/lengths (serialize.go
	// idiom): catches truncation and corruption up front.
	if len(data) < 4 {
		return nil, kvSnapHeader{}, &SnapshotError{"too short"}
	}
	body, want := data[:len(data)-4], binary.LittleEndian.Uint32(data[len(data)-4:])
	if got := crc32.ChecksumIEEE(body); got != want {
		return nil, kvSnapHeader{}, &SnapshotError{fmt.Sprintf("CRC mismatch (got %08x want %08x) — corrupt or truncated", got, want)}
	}
	return r, h, nil
}

// ints writes a length-prefixed []int as uint32s (token ids are non-negative).
func (w *giwWriter) ints(s []int) {
	w.U32(uint32(len(s)))
	for _, v := range s {
		w.U32(uint32(v))
	}
}

// ints reads a length-prefixed []int written by giwWriter.ints. Returns nil for
// a zero length.
func (r *giwReader) ints() []int {
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
