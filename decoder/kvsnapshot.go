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
// Format (little-endian throughout, reusing serialize.go's giwWriter/giwReader):
//
//	magic     [5]byte = "GINFK"
//	version   uint32
//	id        str      (conversation/model identity, for tooling; not validated)
//	numLayers uint32   \
//	kvDim     uint32    | geometry guard — must match the loading model's cache
//	window    uint32    | (else the KV is for a different architecture)
//	manualPos uint8    /
//	pos       uint32   (stored position count)
//	tokens    u32 len + len*uint32
//	  per layer: f32 keys[l] ; f32 vals[l]   (KV-shared layers serialize len 0)
//	crc       uint32   (CRC32-IEEE over every preceding byte)

const (
	kvSnapMagic   = "GINFK"
	kvSnapVersion = 1
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
	wr := &giwWriter{}
	wr.raw([]byte(kvSnapMagic))
	wr.u32(kvSnapVersion)
	wr.str(id)
	wr.u32(uint32(c.numLayers))
	wr.u32(uint32(c.kvDim))
	wr.u32(uint32(c.window))
	if c.manualPos {
		wr.buf = append(wr.buf, 1)
	} else {
		wr.buf = append(wr.buf, 0)
	}
	wr.u32(uint32(c.pos))
	wr.ints(s.tokens)
	for l := 0; l < c.numLayers; l++ {
		wr.f32(c.keys[l])
		wr.f32(c.vals[l])
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
	numLayers, kvDim, window := int(r.u32()), int(r.u32()), int(r.u32())
	if !r.need(1) {
		return nil, &SnapshotError{"truncated header"}
	}
	manualPos := r.data[r.off] == 1
	r.off++
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

	// Geometry guard: the snapshot's cache layout must match this model's, or its
	// KV is meaningless here. NewCache derives the same values from the arch.
	ref := m.NewCache(pos)
	if numLayers != ref.numLayers || kvDim != ref.kvDim || window != ref.window || manualPos != ref.manualPos {
		return nil, &SnapshotError{fmt.Sprintf(
			"geometry mismatch: snapshot {layers:%d kvDim:%d window:%d manualPos:%v} vs model {layers:%d kvDim:%d window:%d manualPos:%v}",
			numLayers, kvDim, window, manualPos, ref.numLayers, ref.kvDim, ref.window, ref.manualPos)}
	}

	tokens := r.ints()
	for l := range numLayers {
		k, v := r.f32(), r.f32()
		// f32 returns nil for a len-0 field (KV-shared layers); keep the cache's
		// empty-but-non-nil slice so Append/Keys stay well-formed.
		if k != nil {
			ref.keys[l] = k
		}
		if v != nil {
			ref.vals[l] = v
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
