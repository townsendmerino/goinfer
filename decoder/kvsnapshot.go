package decoder

import (
	"encoding/binary"
	"fmt"
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
//	quant     uint8    /  (0=f32, 1=int8)
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
	// than the cache's uniform c.kvDim). The format records no per-global-layer stride, so LoadSession restores every global layer to
	// kvDim and the first TruncateTo would mis-slice the odd-width ones, silently wrong on the serve session path. Refuse a
	// non-uniform-width cache (caller cold-prefills), as for the families above. Ring-layer width mismatches are caught loudly by
	// LoadSession (it rejects a ring stride != kvDim), so only globals leak.
	for l := 0; l < c.numLayers; l++ {
		isRing := c.rings[l] != nil
		if !isRing && c.stride[l] != 0 && c.stride[l] != c.kvDim {
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
	wr.raw([]byte(kvSnapMagic))
	wr.u32(kvSnapVersion)
	wr.str(id)
	wr.str(adapterName)
	wr.u32(uint32(c.numLayers))
	wr.u32(uint32(c.kvDim))
	wr.u32(uint32(c.window))
	wr.u32(uint32(c.headDim))
	manualPos := byte(0)
	if c.manualPos {
		manualPos = 1
	}
	wr.raw([]byte{manualPos, byte(c.quant)})
	wr.u32(uint32(c.pos))
	wr.ints(s.tokens)
	// Per layer, in cache-structure order (the loader reconstructs the same rings +
	// quant from the model, so no per-layer tags are needed): ring layers store
	// their W-slot window + logical count; global layers append-forever. Each
	// payload is f32 or int8 (+ per-head scales) per the cache's quant mode. Empty
	// fields (KV-shared / never-written rings) serialize as len 0.
	for l := 0; l < c.numLayers; l++ {
		if r := c.rings[l]; r != nil {
			// Compact: write only the live window [count-W, count) (min(count,W)
			// rows), unwrapped into absolute order — not the full W physical slots
			// (which over-store a short, un-wrapped session). Restore re-wraps.
			lo := max(0, r.count-r.w)
			nLive := r.count - lo
			wr.u32(uint32(r.count))
			wr.u32(uint32(nLive))
			wr.u32(uint32(r.stride))
			if r.stride == 0 || nLive == 0 {
				continue // never-written ring: count/nLive/stride suffice
			}
			st := r.stride
			if c.quant == kvI8 {
				nKV := st / r.headDim
				kb, vb := make([]int8, nLive*st), make([]int8, nLive*st)
				ks, vs := make([]float32, nLive*nKV), make([]float32, nLive*nKV)
				for i, p := 0, lo; p < r.count; i, p = i+1, p+1 {
					so, do := (p%r.w)*st, i*st
					copy(kb[do:do+st], r.kq[so:so+st])
					copy(vb[do:do+st], r.vq[so:so+st])
					sso, sdo := (p%r.w)*nKV, i*nKV
					copy(ks[sdo:sdo+nKV], r.ksc[sso:sso+nKV])
					copy(vs[sdo:sdo+nKV], r.vsc[sso:sso+nKV])
				}
				wr.i8(kb)
				wr.i8(vb)
				wr.f32(ks)
				wr.f32(vs)
			} else {
				kb, vb := make([]float32, nLive*st), make([]float32, nLive*st)
				for i, p := 0, lo; p < r.count; i, p = i+1, p+1 {
					so, do := (p%r.w)*st, i*st
					copy(kb[do:do+st], r.k[so:so+st])
					copy(vb[do:do+st], r.v[so:so+st])
				}
				wr.f32(kb)
				wr.f32(vb)
			}
		} else if c.quant == kvI8 {
			wr.i8(c.keysQ[l])
			wr.i8(c.valsQ[l])
			wr.f32(c.keyScale[l])
			wr.f32(c.valScale[l])
		} else {
			wr.f32(c.keys[l])
			wr.f32(c.vals[l])
		}
	}
	wr.u32(crc32.ChecksumIEEE(wr.buf))
	return wr.buf
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
	r := &giwReader{data: data}
	if got := r.rawN(len(kvSnapMagic)); string(got) != kvSnapMagic {
		return nil, &SnapshotError{fmt.Sprintf("bad magic %q (want %q)", got, kvSnapMagic)}
	}
	if v := r.u32(); v != kvSnapVersion {
		return nil, &SnapshotError{fmt.Sprintf("format version %d, this build reads %d", v, kvSnapVersion)}
	}
	gotID := r.str()
	gotAdapter := r.str()
	numLayers, kvDim, window, headDim := int(r.u32()), int(r.u32()), int(r.u32()), int(r.u32())
	if !r.need(2) {
		return nil, &SnapshotError{"truncated header"}
	}
	manualPos := r.data[r.off] == 1
	quant := kvQuant(r.data[r.off+1])
	r.off += 2
	pos := int(r.u32())
	if r.err != nil {
		return nil, &SnapshotError{"truncated header"}
	}

	// CRC over the whole payload before trusting offsets/lengths (serialize.go
	// idiom): catches truncation and corruption up front.
	if len(data) < 4 {
		return nil, &SnapshotError{"too short"}
	}
	body, want := data[:len(data)-4], binary.LittleEndian.Uint32(data[len(data)-4:])
	if got := crc32.ChecksumIEEE(body); got != want {
		return nil, &SnapshotError{fmt.Sprintf("CRC mismatch (got %08x want %08x) — corrupt or truncated", got, want)}
	}

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
	if numLayers != ref.numLayers || kvDim != ref.kvDim || window != ref.window ||
		headDim != ref.headDim || manualPos != ref.manualPos || quant != ref.quant {
		return nil, &SnapshotError{fmt.Sprintf(
			"geometry mismatch: snapshot {layers:%d kvDim:%d window:%d headDim:%d manualPos:%v quant:%d} vs model {layers:%d kvDim:%d window:%d headDim:%d manualPos:%v quant:%d}",
			numLayers, kvDim, window, headDim, manualPos, quant, ref.numLayers, ref.kvDim, ref.window, ref.headDim, ref.manualPos, ref.quant)}
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
	// len(tokens) is blob-controlled and must equal pos. Too many tokens is the quiet failure: rewindForReuse computes matched > c.pos,
	// TruncateTo treats the out-of-range target as a no-op, and the reuse reports an exact match on a cache never rewound, a session
	// silently continuing from the wrong KV.
	if len(tokens) != pos {
		return nil, &SnapshotError{fmt.Sprintf(
			"tokens: %d ids for pos %d — a longer token list makes rewindForReuse report an "+
				"exact match it never performed", len(tokens), pos)}
	}
	// K and V, f32 or int8 plus scales; 8 B per position per kvDim is the f32 upper bound and the
	// one to budget against. Refused before makeslice ever sees it.
	if want := int64(pos) * int64(ref.numLayers) * int64(ref.kvDim) * 8; want > maxSnapshotCacheBytes {
		return nil, &SnapshotError{fmt.Sprintf(
			"snapshot would allocate %d B of KV cache (pos %d × %d layers × %d kvDim), past the "+
				"%d B ceiling — corrupt or malicious", want, pos, ref.numLayers, ref.kvDim, maxSnapshotCacheBytes)}
	}
	ref = m.NewCache(pos)
	// Same per-layer structure order as Snapshot; ref already has the right rings
	// + quant from NewCache, so fill its fields in place.
	for l := range numLayers {
		if rr := ref.rings[l]; rr != nil {
			rr.count = int(r.u32())
			nLive, st := int(r.u32()), int(r.u32())
			rr.stride = st
			if st == 0 || nLive == 0 {
				// A never-written ring must also have count 0: count>0 with stride/nLive 0 is a state the writer never emits (it leaves rr.count
				// set but the k/v buffers unallocated, so the first decode reads a nil ring and panics). The continue skips the geometry check
				// below, so guard it here.
				if rr.count != 0 {
					return nil, &SnapshotError{"inconsistent ring geometry: nonzero count on an unwritten ring"}
				}
				continue // never-written ring
			}
			// st/nLive/count are blob-controlled and drive make([]…, rr.w·st) and the slice copies below. A ring stores exactly kvDim per
			// position; reject a mismatched stride, nLive past the window, or a count/nLive the payload cannot back, or the make OOMs or the
			// copies slice-panic. rr.headDim is model-derived (>0), so nKV is safe.
			if st != kvDim || rr.count < 0 || nLive > rr.w || rr.count < nLive {
				return nil, &SnapshotError{"inconsistent ring geometry"}
			}
			lo := max(0, rr.count-rr.w)
			nRows := rr.count - lo // copy iterations = payload rows the writer emitted
			if quant == kvI8 {
				nKV := st / rr.headDim
				rr.kq, rr.vq = make([]int8, rr.w*st), make([]int8, rr.w*st)
				rr.ksc, rr.vsc = make([]float32, rr.w*nKV), make([]float32, rr.w*nKV)
				kb, vb, ks, vs := r.i8(), r.i8(), r.f32(), r.f32()
				if len(kb) < nRows*st || len(vb) < nRows*st || len(ks) < nRows*nKV || len(vs) < nRows*nKV {
					return nil, &SnapshotError{"truncated ring payload"}
				}
				for i, p := 0, lo; p < rr.count; i, p = i+1, p+1 {
					so, do := (p%rr.w)*st, i*st
					copy(rr.kq[so:so+st], kb[do:do+st])
					copy(rr.vq[so:so+st], vb[do:do+st])
					sso, sdo := (p%rr.w)*nKV, i*nKV
					copy(rr.ksc[sso:sso+nKV], ks[sdo:sdo+nKV])
					copy(rr.vsc[sso:sso+nKV], vs[sdo:sdo+nKV])
				}
			} else {
				rr.k, rr.v = make([]float32, rr.w*st), make([]float32, rr.w*st)
				kb, vb := r.f32(), r.f32()
				if len(kb) < nRows*st || len(vb) < nRows*st {
					return nil, &SnapshotError{"truncated ring payload"}
				}
				for i, p := 0, lo; p < rr.count; i, p = i+1, p+1 {
					so, do := (p%rr.w)*st, i*st
					copy(rr.k[so:so+st], kb[do:do+st])
					copy(rr.v[so:so+st], vb[do:do+st])
				}
			}
		} else if quant == kvI8 {
			ref.keysQ[l], ref.valsQ[l], ref.keyScale[l], ref.valScale[l] = r.i8(), r.i8(), r.f32(), r.f32()
			// Blob-controlled lengths must be compared, as the ring branch above does for its stride, nLive and payload. The forward derives
			// nKeys from `keys` and then indexes `vals` at the same positions, so a vals array one row short is an out-of-range read in the
			// generation goroutine, a panic that takes the process down; the CRC does not help, since it covers the attacker's bytes. 0 is
			// legal and means a KV-shared layer, which stores nothing of its own.
			if e := checkGlobalLen(l, "keysQ", len(ref.keysQ[l]), pos*kvDim); e != nil {
				return nil, e
			}
			if e := checkGlobalLen(l, "valsQ", len(ref.valsQ[l]), pos*kvDim); e != nil {
				return nil, e
			}
			nKV := kvDim / headDim
			if e := checkGlobalLen(l, "keyScale", len(ref.keyScale[l]), pos*nKV); e != nil {
				return nil, e
			}
			if e := checkGlobalLen(l, "valScale", len(ref.valScale[l]), pos*nKV); e != nil {
				return nil, e
			}
			if len(ref.keysQ[l]) > 0 {
				ref.stride[l] = kvDim // restore the per-layer KV width (audit C-05)
			}
		} else {
			// f32 returns nil for a len-0 field (KV-shared layers); keep the cache's
			// empty-but-non-nil slice so Append/Keys stay well-formed.
			if k := r.f32(); k != nil {
				ref.keys[l] = k
			}
			if v := r.f32(); v != nil {
				ref.vals[l] = v
			}
			// stride[] is set only by Append, which LoadSession bypasses; without the assignment below a restored global layer has stride 0,
			// so the first TruncateTo slices it to [:0] (or attendBatchedHeads panics) while pos stays non-zero. A global layer's width is the
			// cache's uniform kvDim (geometry-guarded above); KV-shared layers keep stride 0, which TruncateTo's min() already guards. The
			// lengths are checked as in the int8 arm above.
			if e := checkGlobalLen(l, "keys", len(ref.keys[l]), pos*kvDim); e != nil {
				return nil, e
			}
			if e := checkGlobalLen(l, "vals", len(ref.vals[l]), pos*kvDim); e != nil {
				return nil, e
			}
			if len(ref.keys[l]) > 0 {
				ref.stride[l] = kvDim
			}
		}
	}
	if r.err != nil {
		return nil, &SnapshotError{"truncated body: " + r.err.Error()}
	}
	ref.pos = pos
	ref.lora = lora
	return &Session{m: m, cache: ref, tokens: tokens, kvAdapter: lora}, nil
}

// ints writes a length-prefixed []int as uint32s (token ids are non-negative).
func (w *giwWriter) ints(s []int) {
	w.u32(uint32(len(s)))
	for _, v := range s {
		w.u32(uint32(v))
	}
}

// ints reads a length-prefixed []int written by giwWriter.ints. Returns nil for
// a zero length.
func (r *giwReader) ints() []int {
	n := int(r.u32())
	if n == 0 || !r.need(n*4) {
		return nil
	}
	out := make([]int, n)
	for i := range out {
		out[i] = int(r.u32())
	}
	return out
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
