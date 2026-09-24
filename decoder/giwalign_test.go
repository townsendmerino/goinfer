package decoder

import (
	"os"
	"path/filepath"
	"testing"
	"unsafe"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/aikit/mmap"
	"github.com/townsendmerino/goinfer/internal/giw"
)

// A .giw's int4/int8 group SCALES used to be copied to the Go heap on every load (giwReader.f32
// copies because the file did not align them): 3.75 GB of a streamed M35's 5.8 GB heap, and 3.0 GB of
// the M26 Metal load's 4.4 GB. v12 pads every weight-matrix payload array to a 16-byte boundary
// (giwWriter.alignArray) inside a v3 bundle whose blob starts at file offset 64, so the reader can
// alias them out of the mapping. These are the gates: the scales really live in the mapping, an
// older layout and a misaligned blob still load (by copying), and a corrupted layout is caught.

func synthInt4(rows, cols, group int, seed float32) linalg.WeightMat {
	q4 := make([]byte, rows*((cols+1)/2))
	for i := range q4 {
		q4[i] = byte(i*7 + int(seed))
	}
	q4s := make([]float32, rows*((cols+group-1)/group))
	for i := range q4s {
		q4s[i] = seed + float32(i)*0.5
	}
	return linalg.WrapInt4(q4, q4s, rows, cols, group)
}

func synthInt8(rows, cols int, seed float32) linalg.WeightMat {
	q8 := make([]int8, rows*cols)
	for i := range q8 {
		q8[i] = int8(i*3 + int(seed))
	}
	scales := make([]float32, rows)
	for i := range scales {
		scales[i] = seed + float32(i)*0.25
	}
	return linalg.WrapInt8(q8, scales, rows, cols, false)
}

// alignFixtureWeights is the tiny f32 fixture with one int4 (kind 3), one int8 (kind 2) and — where
// the arch can emit it — one row4-eligible int4 matrix swapped into layer 0.
func alignFixtureWeights(t *testing.T) *Weights {
	t.Helper()
	raw, _, _, _ := tinyNormRopeGGUF("llama")
	w := loadTinyGGUFWeights(t, raw, "llama")
	l := &w.Layers[0]
	l.QProj = synthInt4(8, 64, 32, 3)
	l.KProj = synthInt8(4, 16, 5)
	l.GateProj = synthInt4(16, 64, 32, 9) // rows%4==0, cols%32==0: kind 5 under an arm64 target
	return w
}

// writeAndMap writes the blob as a v3 bundle in a real file and maps it exactly as Load does, so the
// alignment being tested is the alignment of a REAL mapping, not of a Go heap slice.
func writeAndMap(t *testing.T, blob []byte) (weights []byte, cleanup func()) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "align.giw")
	if err := os.WriteFile(path, giw.Write(blob, nil), 0o644); err != nil {
		t.Fatal(err)
	}
	data, err := mmap.MapReadOnly(path)
	if err != nil {
		t.Fatal(err)
	}
	blobView, _, err := giw.Read(data)
	if err != nil {
		t.Fatal(err)
	}
	return blobView, func() { _ = mmap.Unmap(data) }
}

func within(p unsafe.Pointer, region []byte) bool {
	base := uintptr(unsafe.Pointer(&region[0]))
	a := uintptr(p)
	return a >= base && a < base+uintptr(len(region))
}

func scalesOf(t *testing.T, m *linalg.WeightMat) (scales []float32, which string) {
	t.Helper()
	if _, s, _, ok := m.Int4(); ok {
		return s, "int4"
	}
	if _, s, _, ok := m.Int8(); ok {
		return s, "int8"
	}
	if _, s, ok := m.Int4Row4(); ok { // kind 5 (row4-only): no canonical view exists at all
		return s, "int4-row4"
	}
	t.Fatal("weight matrix is neither int4, int8 nor row4 int4")
	return nil, ""
}

func sameF32(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestGIWAligned_scalesAliasTheMapping(t *testing.T) {
	for _, target := range []struct {
		name string
		t    GIWTarget
	}{{"canonical", GIWTargetNone}, {"cpu-arm64 (kind 5 where eligible)", GIWTargetCPUArm64}} {
		t.Run(target.name, func(t *testing.T) {
			src := alignFixtureWeights(t)
			blob, err := SerializeWeightsForTarget(src, "align", target.t)
			if err != nil {
				t.Fatal(err)
			}
			view, cleanup := writeAndMap(t, blob)
			defer cleanup()
			got, err := LoadSerializedWeights(view)
			if err != nil {
				t.Fatalf("LoadSerializedWeights: %v", err)
			}
			for name, pair := range map[string][2]*linalg.WeightMat{
				"QProj (int4)":    {&src.Layers[0].QProj, &got.Layers[0].QProj},
				"KProj (int8)":    {&src.Layers[0].KProj, &got.Layers[0].KProj},
				"GateProj (int4)": {&src.Layers[0].GateProj, &got.Layers[0].GateProj},
			} {
				want, _ := scalesOf(t, pair[0])
				gotS, gotKind := scalesOf(t, pair[1])
				if gotKind == "int4-row4" {
					// kind 5 stores the REPACKED scales; the same repack of the source is the truth.
					q4, q4s, group, _ := pair[0].Int4()
					_ = q4
					want = linalg.RepackW4A8Row4Scales(q4s, pair[0].Rows(), pair[0].Cols(), group)
				}
				if !sameF32(want, gotS) {
					t.Errorf("%s: scales differ after the round trip (%s)", name, gotKind)
					continue
				}
				if !within(unsafe.Pointer(&gotS[0]), view) {
					t.Errorf("%s: scales were COPIED to the heap, not aliased into the mapping — "+
						"a v12 file's scale arrays must be aligned so the reader can alias them", name)
				}
				if uintptr(unsafe.Pointer(&gotS[0]))%16 != 0 {
					t.Errorf("%s: aliased scale array is not 16-byte aligned (addr %#x)", name, uintptr(unsafe.Pointer(&gotS[0])))
				}
			}
		})
	}
}

// A pre-v12 layout (no padding) must keep loading byte-for-byte correctly. giwEmitVersion is the
// writer's test seam for producing one.
func TestGIWAligned_legacyLayoutStillLoads(t *testing.T) {
	prev := giwEmitVersion
	giwEmitVersion = 11
	t.Cleanup(func() { giwEmitVersion = prev })

	src := alignFixtureWeights(t)
	blob, err := SerializeWeightsForTarget(src, "align-legacy", GIWTargetNone)
	if err != nil {
		t.Fatal(err)
	}
	giwEmitVersion = prev
	current, err := SerializeWeightsForTarget(src, "align-legacy", GIWTargetNone)
	if err != nil {
		t.Fatal(err)
	}
	if len(blob) >= len(current) {
		t.Fatalf("legacy blob (%d B) is not smaller than the padded v12 blob (%d B) — the emit seam is not producing the old layout", len(blob), len(current))
	}
	got, err := LoadSerializedWeights(blob)
	if err != nil {
		t.Fatalf("a v11-layout blob no longer loads: %v", err)
	}
	for name, pair := range map[string][2]*linalg.WeightMat{
		"QProj": {&src.Layers[0].QProj, &got.Layers[0].QProj},
		"KProj": {&src.Layers[0].KProj, &got.Layers[0].KProj},
	} {
		want, _ := scalesOf(t, pair[0])
		gotS, _ := scalesOf(t, pair[1])
		if !sameF32(want, gotS) {
			t.Errorf("%s: scales differ after loading a legacy-layout blob", name)
		}
	}
}

// A v12 blob whose mapping is NOT aligned (an embedded build, a caller-supplied []byte) must still
// load — by copying — and must never take an unaligned float32 pointer into the data (Go's checkptr,
// on under -race, forbids it).
func TestGIWAligned_misalignedBlobFallsBackToCopy(t *testing.T) {
	src := alignFixtureWeights(t)
	blob, err := SerializeWeightsForTarget(src, "align-mis", GIWTargetNone)
	if err != nil {
		t.Fatal(err)
	}
	buf := make([]byte, len(blob)+16)
	off := 0
	for uintptr(unsafe.Pointer(&buf[off]))%4 != 1 {
		off++
	}
	view := buf[off : off+len(blob)]
	copy(view, blob)
	if uintptr(unsafe.Pointer(&view[0]))%4 == 0 {
		t.Fatal("test setup: view is unexpectedly 4-byte aligned")
	}
	got, err := LoadSerializedWeights(view)
	if err != nil {
		t.Fatalf("LoadSerializedWeights on a misaligned blob: %v", err)
	}
	for name, pair := range map[string][2]*linalg.WeightMat{
		"QProj": {&src.Layers[0].QProj, &got.Layers[0].QProj},
		"KProj": {&src.Layers[0].KProj, &got.Layers[0].KProj},
	} {
		want, _ := scalesOf(t, pair[0])
		gotS, _ := scalesOf(t, pair[1])
		if !sameF32(want, gotS) {
			t.Errorf("%s: scales wrong after a misaligned load", name)
		}
		if within(unsafe.Pointer(&gotS[0]), view) {
			t.Errorf("%s: scale array aliases a MISALIGNED blob (addr %#x) — must fall back to a copy",
				name, uintptr(unsafe.Pointer(&gotS[0])))
		}
	}
}

// Every array inside a v12 blob is placed by a pure function of its offset, so flipping one padding
// byte away from zero is invisible to the reader (it skips the pad) but the CRC catches it — the
// padding is inside the checked payload, not free space an attacker or a bit-flip can use silently.
func TestGIWAligned_paddingIsCoveredByTheCRC(t *testing.T) {
	src := alignFixtureWeights(t)
	blob, err := SerializeWeightsForTarget(src, "align-crc", GIWTargetNone)
	if err != nil {
		t.Fatal(err)
	}
	// Find a pad byte: the header pad sits right after the quant label; locate the first zero run
	// that is provably padding by re-deriving the header end the way the reader does.
	r := &giwReader{data: blob}
	r.rawN(len(giwMagic))
	r.version = r.u32()
	r.u32()
	r.str()
	r.bytesField()
	if r.version >= 5 {
		r.str()
	}
	end := r.off
	if pad := int((-int64(end)) & 15); pad > 0 {
		blob[end] ^= 0xFF
		if _, err := LoadSerializedWeights(blob); err == nil {
			t.Fatal("a corrupted padding byte was not caught by the CRC")
		}
		blob[end] ^= 0xFF
	} else {
		t.Skip("header already 16-aligned in this fixture; no padding byte to corrupt")
	}
	if _, err := LoadSerializedWeights(blob); err != nil {
		t.Fatalf("restoring the byte did not restore a loadable blob: %v", err)
	}
}
