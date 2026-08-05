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
// Format v2 (little-endian throughout, reusing serialize.go's giwWriter/giwReader):
//
//	magic     [5]byte = "GINFK"
//	version   uint32 = 2
//	id        str      (conversation/model identity, for tooling; not validated)
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
// v1 (f32-global-only) blobs are rejected by the version guard — snapshots are a
// regenerable cache, so a version bump just triggers a cold prefill.

const (
	kvSnapMagic   = "GINFK"
	kvSnapVersion = 2 // v2: ring windowed persistence × {f32 | int8} payload
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
	// Some families carry recurrent / latent state this format does not persist:
	// qwen3_5_moe's DeltaNet (c.delta), Granite/Nemotron's Mamba-2 state (c.mamba), and
	// DeepSeek/Kimi's MLA compressed-KV latent (c.mlaLatent). Snapshotting any of them would
	// restore from zeroed/empty state and continue silently wrong — refuse (caller skips →
	// cold prefill). All other families serialize fully below, incl. ring (windowed) + int8 (C2).
	if c.delta != nil || len(c.mamba) > 0 || len(c.mlaLatent) > 0 {
		return nil
	}
	wr := &giwWriter{}
	wr.raw([]byte(kvSnapMagic))
	wr.u32(kvSnapVersion)
	wr.str(id)
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

	// M17: numLayers/kvDim/pos are blob-controlled and feed m.NewCache(pos), which allocates pos ×
	// the model's KV footprint — an inflated pos (up to 4B) would makeslice TBs BEFORE the geometry
	// guard below. Reject implausible header dims, and bound pos by what the body can hold (each of
	// the pos positions stores ≥1 byte across the numLayers·kvDim KV; division avoids overflow).
	if numLayers < 0 || numLayers > maxSerializedLayers || kvDim < 0 || kvDim > 1<<24 ||
		headDim < 0 || window < 0 || pos < 0 {
		return nil, &SnapshotError{"implausible header dims"}
	}
	if perPos := numLayers * kvDim; perPos > 0 && int64(pos) > int64(len(data))/int64(perPos) {
		return nil, &SnapshotError{"pos exceeds snapshot body capacity — corrupt or malicious"}
	}

	// Geometry guard: the snapshot's cache layout must match this model's, or its
	// KV is meaningless here. NewCache derives the same values from the arch.
	ref := m.NewCache(pos)
	if numLayers != ref.numLayers || kvDim != ref.kvDim || window != ref.window ||
		headDim != ref.headDim || manualPos != ref.manualPos || quant != ref.quant {
		return nil, &SnapshotError{fmt.Sprintf(
			"geometry mismatch: snapshot {layers:%d kvDim:%d window:%d headDim:%d manualPos:%v quant:%d} vs model {layers:%d kvDim:%d window:%d headDim:%d manualPos:%v quant:%d}",
			numLayers, kvDim, window, headDim, manualPos, quant, ref.numLayers, ref.kvDim, ref.window, ref.headDim, ref.manualPos, ref.quant)}
	}

	tokens := r.ints()
	// Same per-layer structure order as Snapshot; ref already has the right rings
	// + quant from NewCache, so fill its fields in place.
	for l := range numLayers {
		if rr := ref.rings[l]; rr != nil {
			rr.count = int(r.u32())
			nLive, st := int(r.u32()), int(r.u32())
			rr.stride = st
			if st == 0 || nLive == 0 {
				continue // never-written ring
			}
			// M17: st/nLive/count are blob-controlled and drive make([]…, rr.w·st) + the slice
			// copies below. A ring stores exactly kvDim per position; reject a mismatched stride,
			// nLive past the window, or a count/nLive the payload can't back — else the make OOMs
			// or the copies slice-panic. rr.headDim is model-derived (>0), so nKV is safe.
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
			// stride[] is set only by Append, which LoadSession bypasses — without this a
			// restored global layer has stride 0, so the first TruncateTo slices it to [:0]
			// (or panics in attendBatchedHeads) while pos stays non-zero (audit C-05). A
			// global layer's width is the cache's uniform kvDim (geometry-guarded above);
			// KV-shared layers keep stride 0, which TruncateTo's min() already guards.
			if len(ref.keys[l]) > 0 {
				ref.stride[l] = kvDim
			}
		}
	}
	if r.err != nil {
		return nil, &SnapshotError{"truncated body: " + r.err.Error()}
	}
	ref.pos = pos
	return &Session{m: m, cache: ref, tokens: tokens}, nil
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
