package decoder

import (
	"bytes"
	"testing"
	"unsafe"

	"github.com/townsendmerino/aikit/linalg"
)

// Every int4 tensor's group scales are stored as binary16 and aliased from the mapping as the WeightMat's own storage.
// On a Metal target a fused group's members have their scales back to back in one block, so a no-copy buffer can bind
// the whole group's scales as one run; a single tensor's scales are one run by themselves (kind 3).

func f16Fixture(t *testing.T) *Weights {
	t.Helper()
	w := fusedFixture(t) // int4 Q/K/V and Gate/Up (fused groups)
	l := &w.Layers[0]
	l.OProj = synthInt4(8, 64, 32, 13)    // a single tensor: kind 3
	l.DownProj = synthInt4(8, 64, 32, 17) // a single tensor: kind 3
	return w
}

// requireScalesInMapping checks that got's binary16 scales are src's, bit for bit, and are aliased from view; with
// wantAligned, that they start 16-aligned (a single tensor, or the first member of a fused group).
func requireScalesInMapping(t *testing.T, view []byte, name string, src, got *linalg.WeightMat, wantAligned bool) []uint16 {
	t.Helper()
	want, f16 := src.Int4ScalesF16(), got.Int4ScalesF16()
	if len(f16) == 0 || len(f16) != len(want) {
		t.Fatalf("%s: %d scales, want %d", name, len(f16), len(want))
	}
	for i := range want {
		if f16[i] != want[i] {
			t.Fatalf("%s: scale %d = %#x, want %#x", name, i, f16[i], want[i])
		}
	}
	if !within(unsafe.Pointer(&f16[0]), view) {
		t.Errorf("%s: scales were copied, not aliased from the mapping", name)
	}
	if wantAligned && uintptr(unsafe.Pointer(&f16[0]))%16 != 0 {
		t.Errorf("%s: scales not 16-byte aligned (%#x)", name, uintptr(unsafe.Pointer(&f16[0])))
	}
	return f16
}

// adjacent reports whether b starts where a ends.
func adjacent(a, b []uint16) bool {
	return uintptr(unsafe.Pointer(&a[0]))+uintptr(2*len(a)) == uintptr(unsafe.Pointer(&b[0]))
}

func TestGIWF16_metalTargetScalesAreAliasedAndGroupsAreOneRun(t *testing.T) {
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
	sl, l := &src.Layers[0], &got.Layers[0]
	requireScalesInMapping(t, view, "OProj (single)", &sl.OProj, &l.OProj, true)
	requireScalesInMapping(t, view, "DownProj (single)", &sl.DownProj, &l.DownProj, true)
	q := requireScalesInMapping(t, view, "QProj (group head)", &sl.QProj, &l.QProj, true)
	k := requireScalesInMapping(t, view, "KProj (group member)", &sl.KProj, &l.KProj, false)
	v := requireScalesInMapping(t, view, "VProj (group member)", &sl.VProj, &l.VProj, false)
	g := requireScalesInMapping(t, view, "GateProj (group head)", &sl.GateProj, &l.GateProj, true)
	u := requireScalesInMapping(t, view, "UpProj (group member)", &sl.UpProj, &l.UpProj, false)
	if !adjacent(q, k) || !adjacent(k, v) {
		t.Error("the Q/K/V group's scales are not one contiguous run")
	}
	if !adjacent(g, u) {
		t.Error("the Gate/Up group's scales are not one contiguous run")
	}
	for name, p := range map[string][2]*linalg.WeightMat{
		"OProj": {&sl.OProj, &l.OProj}, "DownProj": {&sl.DownProj, &l.DownProj},
	} {
		sameMat(t, name, p[0], p[1])
	}
}

// TestGIWF16_metalSingleIsKind3: on a Metal target a single int4 tensor is written exactly as on any other target,
// so the two blobs' records for it are the same bytes. Kind 7, a Metal-only record for a single tensor, is not
// written and not read.
func TestGIWF16_metalSingleIsKind3(t *testing.T) {
	src := alignFixtureWeights(t) // int4 beside int8 and f32: no group is eligible to fuse
	src.Layers[0].OProj = synthInt4(8, 64, 32, 13)
	metal, err := SerializeWeightsForTarget(src, "single", GIWTargetMetal)
	if err != nil {
		t.Fatal(err)
	}
	plain, err := SerializeWeightsForTarget(src, "single", GIWTargetNone)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(metal, plain) {
		t.Errorf("with no fused group, a Metal-target blob (%d bytes) differs from a plain one (%d bytes)", len(metal), len(plain))
	}
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
