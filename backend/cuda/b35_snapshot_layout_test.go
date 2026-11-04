//go:build cuda && goinfer_testhooks

package cuda

import (
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestB35SnapshotLayout_followsEachLayersKind: a snapshot of a mixed stack lays out K/V rows for each attention layer that owns its K/V,
// nothing for a layer that aliases another's, and the conv ring and state, not K/V, for each DeltaNet layer (whose K/V buffers do not
// exist: laying them out made SnapshotKV download from a nil buffer). Built by hand, no device.
func TestB35SnapshotLayout_followsEachLayersKind(t *testing.T) {
	const kvDim, end = 8, 64
	r := &cudaResident{
		layers: []cudaLayer{
			{mixer: decoder.MixerDeltaNet},
			{mixer: decoder.MixerAttention, kvDim: kvDim},
			{mixer: decoder.MixerDeltaNet},
			{mixer: decoder.MixerAttention, kvDim: kvDim, kvShared: true, kvSrc: 1},
		},
		dnet: &dnetParams{},
	}
	r.kc, r.vc = make([]Buffer, len(r.layers)), make([]Buffer, len(r.layers))
	layers, words := r.snapshotLayout(end, 0)
	for l, want := range []bool{false, true, false, false} {
		if layers[l].held != want {
			t.Errorf("layer %d (%v, shared %v): held %v, want %v", l, r.layers[l].mixer, r.layers[l].kvShared, layers[l].held, want)
		}
	}
	if want := end * kvDim; words != want {
		t.Errorf("the rows take %d words, want %d: only layer 1's K rows (its V buffer is empty here)", words, want)
	}
	dn, _ := r.snapshotDeltaLayout(words)
	if len(dn) != 2 || dn[0].layer != 0 || dn[1].layer != 2 {
		t.Errorf("DeltaNet state laid out for %+v, want layers 0 and 2", dn)
	}
}
