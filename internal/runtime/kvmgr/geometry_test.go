package kvmgr

import "testing"

func TestGroup_blocksHeld(t *testing.T) {
	full := Group{Retention: RetainFull}
	window := Group{Retention: RetainWindow, Span: 1024}
	chunk := Group{Retention: RetainChunk, Span: 8192}
	for _, tc := range []struct {
		name string
		g    Group
		n    int
		want int
	}{
		{"a full group holds every block", full, 32768, 512},
		{"a full group, a partial tail", full, 65, 2},
		{"a 1,024 window holds 17 blocks of a long sequence", window, 32768, 17},
		{"a window not yet filled holds everything", window, 700, 11},
		{"a chunk group holds the current chunk", chunk, 8192 + 100, 2},
		{"a chunk group at the start of a chunk", chunk, 16384, 0},
		{"nothing", window, 0, 0},
	} {
		if got := tc.g.BlocksHeld(tc.n); got != tc.want {
			t.Errorf("%s: %d blocks, want %d", tc.name, got, tc.want)
		}
	}
}

func TestGeometry_groupOf(t *testing.T) {
	g := Geometry{NumLayers: 4, Groups: []Group{{Layers: []int{0, 2}}, {Retention: RetainWindow, Span: 512, Layers: []int{1}}}, Shared: map[int]int{3: 1}}
	for layer, want := range map[int]int{0: 0, 1: 1, 2: 0, 3: 1} {
		if got := g.GroupOf(layer); got != want {
			t.Errorf("layer %d is in group %d, want %d", layer, got, want)
		}
	}
	if g.Recurrent() || !(Geometry{States: []State{{Kind: "deltanet", Layers: []int{0}}}}).Recurrent() {
		t.Error("Recurrent does not follow States")
	}
}
