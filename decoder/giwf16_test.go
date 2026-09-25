package decoder

import (
	"testing"
	"unsafe"

	"github.com/townsendmerino/aikit/linalg"
)

// Weights format v14 (metal target): every canonical group-32 int4 tensor also carries its group scales
// pre-converted by F16Bits, so a Metal no-copy buffer can alias them instead of converting the f32 scales
// into a new buffer (~389 MB on the 7B, S6). Singles are kind 7; kind-6 groups gain an f16 block. These pin
// that the f16 arrays exist for exactly those tensors, equal the kernels' own conversion, live in the
// mapping where the Metal build can find them, and that other targets and older files are unaffected.

func f16Fixture(t *testing.T) *Weights {
	t.Helper()
	w := fusedFixture(t) // int4 Q/K/V and Gate/Up (fused groups)
	l := &w.Layers[0]
	l.OProj = synthInt4(8, 64, 32, 13)    // single → kind 7
	l.DownProj = synthInt4(8, 64, 32, 17) // single → kind 7
	return w
}

func requireF16(t *testing.T, w *Weights, view []byte, name string, m *linalg.WeightMat, wantAligned bool) {
	t.Helper()
	q4, q4s, _, ok := m.Int4()
	if !ok {
		t.Fatalf("%s: not int4", name)
	}
	f16, ok := w.int4F16[uintptr(unsafe.Pointer(&q4[0]))]
	if !ok {
		t.Fatalf("%s: no f16 scales recorded in a v14 metal-target load", name)
	}
	if len(f16) != len(q4s) {
		t.Fatalf("%s: %d f16 scales, want %d", name, len(f16), len(q4s))
	}
	for i, v := range q4s {
		if f16[i] != F16Bits(v) {
			t.Fatalf("%s: f16[%d] = %#x, want F16Bits(%g) = %#x", name, i, f16[i], v, F16Bits(v))
		}
	}
	if !within(unsafe.Pointer(&f16[0]), view) {
		t.Errorf("%s: f16 scales were copied, not aliased from the mapping", name)
	}
	if wantAligned && uintptr(unsafe.Pointer(&f16[0]))%16 != 0 {
		t.Errorf("%s: f16 scales not 16-byte aligned (%#x)", name, uintptr(unsafe.Pointer(&f16[0])))
	}
}

func TestGIWF16_metalTargetCarriesF16ScalesForEveryInt4Tensor(t *testing.T) {
	src := f16Fixture(t)
	blob, err := SerializeWeightsForTarget(src, "f16", GIWTargetMetal)
	if err != nil {
		t.Fatal(err)
	}
	view, cleanup := writeAndMap(t, blob)
	defer cleanup()
	got, err := LoadSerializedWeights(view)
	if err != nil {
		t.Fatalf("LoadSerializedWeights: %v", err)
	}
	l := &got.Layers[0]
	requireF16(t, got, view, "OProj (kind 7)", &l.OProj, true)
	requireF16(t, got, view, "DownProj (kind 7)", &l.DownProj, true)
	requireF16(t, got, view, "QProj (group head)", &l.QProj, true)
	requireF16(t, got, view, "KProj (group member)", &l.KProj, false)
	requireF16(t, got, view, "VProj (group member)", &l.VProj, false)
	requireF16(t, got, view, "GateProj (group head)", &l.GateProj, true)
	requireF16(t, got, view, "UpProj (group member)", &l.UpProj, false)
	// The round trip itself is unchanged by the extra array.
	for name, p := range map[string][2]*linalg.WeightMat{
		"OProj": {&src.Layers[0].OProj, &l.OProj}, "DownProj": {&src.Layers[0].DownProj, &l.DownProj},
	} {
		sameMat(t, name, p[0], p[1])
	}

	m := &Model{w: got}
	if f, ok := m.Int4ScalesF16(&l.OProj); !ok || len(f) == 0 {
		t.Errorf("Model.Int4ScalesF16(OProj) = %d scales, ok=%v", len(f), ok)
	}
	if _, ok := m.Int4ScalesF16(&linalg.WeightMat{}); ok {
		t.Error("Model.Int4ScalesF16 returned scales for an empty WeightMat")
	}
}

func TestGIWF16_otherTargetsAndOlderFilesCarryNone(t *testing.T) {
	src := f16Fixture(t)
	for _, tgt := range []GIWTarget{GIWTargetNone, GIWTargetCUDA} {
		blob, err := SerializeWeightsForTarget(src, "f16", tgt)
		if err != nil {
			t.Fatal(err)
		}
		got, err := LoadSerializedWeights(blob)
		if err != nil {
			t.Fatalf("%q: %v", tgt, err)
		}
		if got.int4F16 != nil {
			t.Errorf("target %q: %d f16 entries, want none", tgt, len(got.int4F16))
		}
	}
	prev := giwEmitVersion
	giwEmitVersion = giwVFused // a v13 metal-target file: fused groups, no f16
	blob, err := SerializeWeightsForTarget(src, "f16-v13", GIWTargetMetal)
	giwEmitVersion = prev
	if err != nil {
		t.Fatal(err)
	}
	got, err := LoadSerializedWeights(blob)
	if err != nil {
		t.Fatalf("a v13 metal-target blob no longer loads: %v", err)
	}
	if got.int4F16 != nil {
		t.Errorf("v13 metal-target blob: %d f16 entries, want none", len(got.int4F16))
	}
	sameMat(t, "OProj (v13)", &src.Layers[0].OProj, &got.Layers[0].OProj)
}

// F16Bits is the conversion the Metal kernels' scales were validated with (round half UP on the dropped
// bits, not ties-to-even). Pinned at the edges so a "fix" to ties-to-even — which would silently change
// every stored scale and break byte-identity with any file written before — shows up here first.
func TestF16Bits_pinned(t *testing.T) {
	for _, tc := range []struct {
		in   float32
		want uint16
	}{
		{1, 0x3C00}, {-2, 0xC000}, {0.5, 0x3800}, {0, 0}, {65504, 0x7BFF}, {65536, 0x7C00},
		{1.0 + 1.0/2048, 0x3C01}, // exactly half an f16 ULP above 1: rounds UP (ties-to-even would give 0x3C00)
		{float32(6e-8), 0x0001},  // smallest subnormal region
		{1e-9, 0},                // underflow to zero
	} {
		if got := F16Bits(tc.in); got != tc.want {
			t.Errorf("F16Bits(%g) = %#04x, want %#04x", tc.in, got, tc.want)
		}
	}
	if got := F16Bits(float32(math32NaN())); got&0x7C00 != 0x7C00 || got&0x03FF == 0 {
		t.Errorf("F16Bits(NaN) = %#04x, want a NaN", got)
	}
}

func math32NaN() float64 { var z float64; return z / z }
