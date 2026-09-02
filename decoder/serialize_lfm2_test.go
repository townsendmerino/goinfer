package decoder

import (
	"strings"
	"testing"
)

// C-03: A CRC-VALID BUNDLE THAT LOADS CLEAN AND NIL-DEREFS AT THE FIRST FORWARD.
//
// `grep shortConv decoder/serialize.go` returned ZERO matches. cmd/prequant loaded an LFM2
// checkpoint, wrote every field except the conv mixer, appended a valid CRC, and selfCheck passed
// because selfCheck only Loads. Serving the bundle, the first token reached conv layer 0 with
// lw.shortConv == nil and panicked in the decode goroutine. That is the R3 shape: the artifact is
// well-formed by every check that runs on it, and wrong.
//
// Two halves, and BOTH are needed. The v8 tail makes a correctly-written bundle round-trip. The
// validateShapes presence check makes an INCORRECTLY-written one — a pre-v8 bundle, or a future
// writer that forgets again — fail at load instead of at the first forward, where the panic is in
// a goroutine no handler recovers.
func TestLFM2_serializeRoundTripsTheConvMixer(t *testing.T) {
	m, err := Load("../testdata/lfm2-tiny", Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()

	convLayers := 0
	for l := range m.w.Layers {
		if m.w.Layers[l].shortConv != nil {
			convLayers++
		}
	}
	if convLayers == 0 {
		t.Fatal("the premise broke: the fixture has no conv layer, so this test proves nothing")
	}

	blob, err := SerializeWeights(m.w, "lfm2-roundtrip")
	if err != nil {
		t.Fatalf("SerializeWeights: %v", err)
	}
	w2, err := LoadSerializedWeights(blob)
	if err != nil {
		t.Fatalf("LoadSerializedWeights: %v", err)
	}
	back := 0
	for l := range w2.Layers {
		got, want := w2.Layers[l].shortConv, m.w.Layers[l].shortConv
		if want == nil {
			continue
		}
		if got == nil {
			t.Fatalf("layer %d: shortConv is nil after a .giw round-trip", l)
		}
		back++
		for _, c := range []struct {
			name      string
			got, want []float32
		}{
			{"inProj", got.inProj, want.inProj},
			{"convW", got.convW, want.convW},
			{"outProj", got.outProj, want.outProj},
		} {
			if len(c.got) != len(c.want) {
				t.Fatalf("layer %d %s: len %d, want %d", l, c.name, len(c.got), len(c.want))
			}
			for i := range c.want {
				if c.got[i] != c.want[i] {
					t.Fatalf("layer %d %s[%d] = %v, want %v", l, c.name, i, c.got[i], c.want[i])
				}
			}
		}
	}
	if back != convLayers {
		t.Fatalf("%d conv mixers went in, %d came back", convLayers, back)
	}
}

// The other half: a bundle whose conv layers carry NO mixer must be refused at LOAD. Before the v8
// tail every LFM2 bundle was one of these, and the only thing that noticed was the decode goroutine
// dying. This builds one on purpose by clearing the field before serializing.
func TestLFM2_bundleWithoutConvWeightsIsRefusedAtLoad(t *testing.T) {
	m, err := Load("../testdata/lfm2-tiny", Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()

	dropped := 0
	for l := range m.w.Layers {
		if m.w.Layers[l].shortConv != nil {
			m.w.Layers[l].shortConv = nil
			dropped++
		}
	}
	if dropped == 0 {
		t.Fatal("the premise broke: nothing to drop")
	}
	blob, err := SerializeWeights(m.w, "lfm2-no-conv")
	if err != nil {
		t.Fatalf("SerializeWeights refused to write it; the point is that it writes a VALID blob "+
			"and the reader must be the one to refuse: %v", err)
	}
	_, err = LoadSerializedWeights(blob)
	if err == nil {
		t.Fatal("LoadSerializedWeights ACCEPTED a bundle with no conv mixer on any layer. It is " +
			"CRC-valid and it loads clean — and the first forward nil-derefs in the decode " +
			"goroutine, where net/http's handler recover does not reach (audit-2026-09-02 C-03)")
	}
	if !strings.Contains(err.Error(), "short-conv") {
		t.Errorf("refused, but not for the right reason: %v", err)
	}
}
