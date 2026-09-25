package decoder

import (
	"bytes"
	"encoding/binary"
	"testing"
	"unsafe"

	"github.com/townsendmerino/aikit/linalg"
)

// v13's fused groups (kind 6) put the nibbles of q,k,v and of gate,up back to back so a Metal fused GEMV
// buffer can be aliased out of the mapping (S6). These pin what that relies on: the round trip is exact,
// the members really are adjacent and 16-aligned in the loaded blob, a non-Metal target's bytes are
// exactly what v12 wrote, and a group that cannot be fused degrades to plain records.

func fusedFixture(t *testing.T) *Weights {
	t.Helper()
	raw, _, _, _ := tinyNormRopeGGUF("llama")
	w := loadTinyGGUFWeights(t, raw, "llama")
	l := &w.Layers[0]
	l.QProj = synthInt4(8, 64, 32, 3)
	l.KProj = synthInt4(4, 64, 32, 5)
	l.VProj = synthInt4(4, 64, 32, 7)
	l.GateProj = synthInt4(16, 64, 32, 9)
	l.UpProj = synthInt4(16, 64, 32, 11)
	return w
}

func int4Bytes(t *testing.T, m *linalg.WeightMat) []byte {
	t.Helper()
	q4, _, _, ok := m.Int4()
	if !ok {
		t.Fatal("not a canonical int4 matrix")
	}
	return q4
}

func sameMat(t *testing.T, name string, a, b *linalg.WeightMat) {
	t.Helper()
	aq, as, ag, aok := a.Int4()
	bq, bs, bg, bok := b.Int4()
	if !aok || !bok || ag != bg || !bytes.Equal(aq, bq) || !sameF32(as, bs) || a.Rows() != b.Rows() || a.Cols() != b.Cols() {
		t.Errorf("%s: differs after the round trip", name)
	}
}

func TestGIWFused_roundTripAndAdjacency(t *testing.T) {
	src := fusedFixture(t)
	blob, err := SerializeWeightsForTarget(src, "fused", GIWTargetMetal)
	if err != nil {
		t.Fatal(err)
	}
	if v := binary.LittleEndian.Uint32(blob[len(giwMagic):]); v != giwVersion {
		t.Fatalf("a metal-target blob is version %d, want %d", v, giwVersion)
	}
	if !bytes.Contains(blob, []byte{6, 8, 0, 0, 0, 64, 0, 0, 0, 32, 0, 0, 0}) { // kind 6, rows 8 (Q), cols 64, group 32
		t.Fatal("no kind-6 record in a metal-target blob — the fused path did not engage")
	}
	view, cleanup := writeAndMap(t, blob)
	defer cleanup()
	got, err := LoadSerializedWeights(view)
	if err != nil {
		t.Fatalf("LoadSerializedWeights: %v", err)
	}
	gl, sl := &got.Layers[0], &src.Layers[0]
	for name, p := range map[string][2]*linalg.WeightMat{
		"Q": {&sl.QProj, &gl.QProj}, "K": {&sl.KProj, &gl.KProj}, "V": {&sl.VProj, &gl.VProj},
		"Gate": {&sl.GateProj, &gl.GateProj}, "Up": {&sl.UpProj, &gl.UpProj},
	} {
		sameMat(t, name, p[0], p[1])
	}
	// Adjacent: each member's nibbles end exactly where the next member's begin, in the mapping, and the
	// first starts on a 16-byte boundary — the two facts a fused buffer over the mapping needs.
	for _, grp := range [][]*linalg.WeightMat{{&gl.QProj, &gl.KProj, &gl.VProj}, {&gl.GateProj, &gl.UpProj}} {
		for i, m := range grp {
			q4 := int4Bytes(t, m)
			if !within(unsafe.Pointer(&q4[0]), view) {
				t.Errorf("member %d nibbles were copied to the heap, not aliased into the mapping", i)
			}
			if uintptr(unsafe.Pointer(&q4[0]))%16 != 0 {
				t.Errorf("member %d nibbles are not 16-byte aligned", i)
			}
			if i > 0 {
				prev := int4Bytes(t, grp[i-1])
				if uintptr(unsafe.Pointer(&prev[0]))+uintptr(len(prev)) != uintptr(unsafe.Pointer(&q4[0])) {
					t.Errorf("member %d does not start where member %d ends — the group block is not contiguous", i, i-1)
				}
			}
		}
	}
}

// A non-Metal target must be byte-for-byte what v12 wrote: nothing about fused groups may leak into a
// bundle whose reader never asked for them (and it must stay readable by a pre-v13 reader).
func TestGIWFused_nonMetalTargetIsUnchanged(t *testing.T) {
	src := fusedFixture(t)
	for _, tgt := range []GIWTarget{GIWTargetNone, GIWTargetCUDA, GIWTargetWebGPU} {
		now, err := SerializeWeightsForTarget(src, "fused", tgt)
		if err != nil {
			t.Fatal(err)
		}
		prev := giwEmitVersion
		giwEmitVersion = giwVAligned
		old, err := SerializeWeightsForTarget(src, "fused", tgt)
		giwEmitVersion = prev
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(now, old) {
			t.Errorf("target %q: bytes differ from a v12 write (%d vs %d) — a non-Metal bundle changed", tgt, len(now), len(old))
		}
		if v := binary.LittleEndian.Uint32(now[len(giwMagic):]); v != giwVAligned {
			t.Errorf("target %q: version %d, want %d (a pre-v13 reader must keep reading it)", tgt, v, giwVAligned)
		}
	}
}

// A group with a member that cannot fuse (an int8 tensor; an absent V on a K=V layer) is written as
// ordinary records — no kind 6 — and still round-trips.
func TestGIWFused_ineligibleGroupFallsBack(t *testing.T) {
	src := fusedFixture(t)
	src.Layers[0].UpProj = synthInt8(16, 64, 13) // gate int4 + up int8: not fusable
	src.Layers[0].VProj = linalg.WeightMat{}     // absent V
	blob, err := SerializeWeightsForTarget(src, "fallback", GIWTargetMetal)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(blob, []byte{6, 8, 0, 0, 0, 64, 0, 0, 0, 32, 0, 0, 0}) || bytes.Contains(blob, []byte{6, 16, 0, 0, 0, 64, 0, 0, 0, 32, 0, 0, 0}) {
		t.Error("an ineligible group was written as kind 6")
	}
	got, err := LoadSerializedWeights(blob)
	if err != nil {
		t.Fatalf("LoadSerializedWeights: %v", err)
	}
	sameMat(t, "Q", &src.Layers[0].QProj, &got.Layers[0].QProj)
	sameMat(t, "Gate", &src.Layers[0].GateProj, &got.Layers[0].GateProj)
	if got.Layers[0].VProj.Rows() != 0 {
		t.Error("an absent V came back present")
	}
}

// The group block is inside the CRC'd payload, and a truncated block is an error, not a panic or a
// short read that decodes as garbage.
func TestGIWFused_blockIsCheckedAndTruncationIsRefused(t *testing.T) {
	src := fusedFixture(t)
	blob, err := SerializeWeightsForTarget(src, "fused-crc", GIWTargetMetal)
	if err != nil {
		t.Fatal(err)
	}
	q := int4Bytes(t, &src.Layers[0].QProj)
	i := bytes.Index(blob, q)
	if i < 0 {
		t.Fatal("could not find the Q nibbles in the blob")
	}
	blob[i] ^= 0xFF
	if _, err := LoadSerializedWeights(blob); err == nil {
		t.Fatal("a corrupted nibble in the group block was not caught by the CRC")
	}
	blob[i] ^= 0xFF
	if _, err := LoadSerializedWeights(blob); err != nil {
		t.Fatalf("restoring the byte did not restore a loadable blob: %v", err)
	}
	for _, cut := range []int{len(blob) - 1, len(blob) - 40, i + len(q)/2, i} {
		if _, err := LoadSerializedWeights(blob[:cut]); err == nil {
			t.Errorf("a blob truncated to %d of %d bytes loaded", cut, len(blob))
		}
	}
}

// The group block's padding is a function of the position after the members' scale arrays, which the
// model-level fixtures (whose shapes the loader validates against the arch) always leave on a 16-byte
// boundary — so a writer or reader that dropped the pad would pass them. Drive the writer and reader
// directly, at every starting alignment and with shapes whose last scale array ends off-boundary.
func TestGIWFused_blockPaddingAtEveryAlignment(t *testing.T) {
	for _, shape := range [][]int{{6, 3, 3}, {5, 3}, {1, 1, 1}, {7, 2}, {3, 5, 9}} { // rows per member; K = 64
		for lead := range 16 { // bytes already in the blob before the group: every alignment
			ms := make([]linalg.WeightMat, len(shape))
			ptrs := make([]*linalg.WeightMat, len(shape))
			for i, rows := range shape {
				ms[i] = synthInt4(rows, 64, 32, float32(i+1))
				ptrs[i] = &ms[i]
			}
			w := &giwWriter{target: GIWTargetMetal}
			w.raw(make([]byte, lead))
			w.fusedGroup(ptrs...)
			w.raw([]byte{0xAB, 0xCD}) // a trailer: the reader must stop exactly at the group's end

			r := &giwReader{data: w.buf, off: lead, version: w.emitVersion()}
			got := make([]linalg.WeightMat, len(shape))
			gp := make([]*linalg.WeightMat, len(shape))
			for i := range got {
				gp[i] = &got[i]
			}
			r.fusedGroup(gp...)
			if r.err != nil {
				t.Fatalf("shape %v lead %d: %v", shape, lead, r.err)
			}
			if r.off != len(w.buf)-2 {
				t.Errorf("shape %v lead %d: reader stopped at %d, group ends at %d", shape, lead, r.off, len(w.buf)-2)
			}
			for i := range got {
				sameMat(t, "member", &ms[i], &got[i])
				q4 := int4Bytes(t, &got[i])
				at := int(uintptr(unsafe.Pointer(&q4[0])) - uintptr(unsafe.Pointer(&w.buf[0])))
				if at%16 != 0 {
					t.Errorf("shape %v lead %d member %d: nibbles at blob offset %d, not 16-aligned", shape, lead, i, at)
				}
				if i > 0 {
					prev := int4Bytes(t, &got[i-1])
					if uintptr(unsafe.Pointer(&prev[0]))+uintptr(len(prev)) != uintptr(unsafe.Pointer(&q4[0])) {
						t.Errorf("shape %v lead %d: members %d and %d are not adjacent", shape, lead, i-1, i)
					}
				}
				// v14: the member's f16 scales, recorded against its nibbles, equal F16Bits of its f32 scales,
				// and the members' f16 arrays are one contiguous run starting 16-aligned.
				f16 := r.f16[uintptr(unsafe.Pointer(&q4[0]))]
				_, q4s, _, _ := ms[i].Int4()
				if len(f16) != len(q4s) {
					t.Fatalf("shape %v lead %d member %d: %d f16 scales recorded, want %d", shape, lead, i, len(f16), len(q4s))
				}
				for j := range q4s {
					if f16[j] != F16Bits(q4s[j]) {
						t.Fatalf("shape %v lead %d member %d: f16[%d] = %#x, want F16Bits = %#x", shape, lead, i, j, f16[j], F16Bits(q4s[j]))
					}
				}
				at16 := int(uintptr(unsafe.Pointer(&f16[0])) - uintptr(unsafe.Pointer(&w.buf[0])))
				if i == 0 && at16%16 != 0 {
					t.Errorf("shape %v lead %d: f16 block at blob offset %d, not 16-aligned", shape, lead, at16)
				}
				if i > 0 {
					pq, _, _, _ := got[i-1].Int4()
					pf := r.f16[uintptr(unsafe.Pointer(&pq[0]))]
					if uintptr(unsafe.Pointer(&pf[0]))+uintptr(2*len(pf)) != uintptr(unsafe.Pointer(&f16[0])) {
						t.Errorf("shape %v lead %d: f16 scales of members %d and %d are not adjacent", shape, lead, i-1, i)
					}
				}
			}
		}
	}
}
