package decoder

import (
	"bytes"
	"io"
	"path/filepath"
	"testing"
	"unsafe"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
)

// TestLoadFusedExperts_perExpertSlicing exercises loadFusedExperts directly
// against a synthetic fused-expert safetensors pair (the qwen3_5_moe/Mixtral
// gate_up_proj+down_proj layout), which neither TestQwen35_forwardParity nor
// TestMixtral_forwardParity actually reaches: both tiny fixtures store
// unfused per-expert tensors, so `fusedExperts` is false for them and this
// function is never called. Each expert's data is filled with a value unique
// to (expert, position) so a stride/offset bug reads a neighboring expert's
// data instead of its own, which this test would catch.
func TestLoadFusedExperts_perExpertSlicing(t *testing.T) {
	const nExpert, inter, hidden = 3, 4, 6
	half := inter * hidden       // one of gate/up within the fused tensor
	guStride := 2 * half         // per-expert stride in gate_up_proj
	downStride := hidden * inter // per-expert stride in down_proj

	tag := func(e, base, i int) float32 { return float32(e*100000 + base + i) }

	guData := make([]float32, nExpert*guStride)
	for e := range nExpert {
		for i := range guStride {
			guData[e*guStride+i] = tag(e, 1000, i)
		}
	}
	dnData := make([]float32, nExpert*downStride)
	for e := range nExpert {
		for i := range downStride {
			dnData[e*downStride+i] = tag(e, 9000, i)
		}
	}

	dir := t.TempDir()
	path := filepath.Join(dir, "model.safetensors")
	writeSafetensors(t, path, map[string]stTensor{
		"gate_up_proj": {[]int{nExpert, 2 * inter, hidden}, guData},
		"down_proj":    {[]int{nExpert, hidden, inter}, dnData},
	})

	st, err := embed.OpenSafetensors(path)
	if err != nil {
		t.Fatalf("OpenSafetensors: %v", err)
	}
	defer st.Close()

	experts, err := loadFusedExperts(st, "gate_up_proj", "down_proj", nExpert, inter, hidden, quantNone, false, nil)
	if err != nil {
		t.Fatalf("loadFusedExperts: %v", err)
	}
	if len(experts) != nExpert {
		t.Fatalf("got %d experts, want %d", len(experts), nExpert)
	}

	for e := range nExpert {
		gate, ok := experts[e].Gate.F32()
		if !ok || len(gate) != half {
			t.Fatalf("expert %d: Gate.F32() ok=%v len=%d, want %d", e, ok, len(gate), half)
		}
		for i, v := range gate {
			if want := tag(e, 1000, i); v != want {
				t.Fatalf("expert %d Gate[%d] = %v, want %v (offset/stride bug: likely reading a neighboring expert)", e, i, v, want)
			}
		}

		up, ok := experts[e].Up.F32()
		if !ok || len(up) != half {
			t.Fatalf("expert %d: Up.F32() ok=%v len=%d, want %d", e, ok, len(up), half)
		}
		for i, v := range up {
			if want := tag(e, 1000, half+i); v != want {
				t.Fatalf("expert %d Up[%d] = %v, want %v", e, i, v, want)
			}
		}

		down, ok := experts[e].Down.F32()
		if !ok || len(down) != downStride {
			t.Fatalf("expert %d: Down.F32() ok=%v len=%d, want %d", e, ok, len(down), downStride)
		}
		for i, v := range down {
			if want := tag(e, 9000, i); v != want {
				t.Fatalf("expert %d Down[%d] = %v, want %v", e, i, v, want)
			}
		}
	}
}

// TestLoadFusedExperts_shapeMismatch confirms the Elements() validation this
// change added (loadFusedExperts used to get this for free from TensorF32's
// own length check) actually fires on a malformed tensor.
func TestLoadFusedExperts_shapeMismatch(t *testing.T) {
	const nExpert, inter, hidden = 2, 4, 6
	dir := t.TempDir()
	path := filepath.Join(dir, "model.safetensors")
	// gate_up_proj is well-formed for nExpert-1 experts; asking
	// loadFusedExperts to unpack nExpert must be rejected by its
	// Elements() check rather than silently reading past the tensor.
	writeSafetensors(t, path, map[string]stTensor{
		"gate_up_proj": {[]int{nExpert - 1, 2 * inter, hidden}, make([]float32, (nExpert-1)*2*inter*hidden)},
		"down_proj":    {[]int{nExpert, hidden, inter}, make([]float32, nExpert*hidden*inter)},
	})

	st, err := embed.OpenSafetensors(path)
	if err != nil {
		t.Fatalf("OpenSafetensors: %v", err)
	}
	defer st.Close()

	if _, err := loadFusedExperts(st, "gate_up_proj", "down_proj", nExpert, inter, hidden, quantNone, false, nil); err == nil {
		t.Fatal("loadFusedExperts should reject a gate_up_proj tensor with the wrong element count")
	}
}

// TestLoadFusedExperts_transposedLayoutGivesTheRowLayout: the transposed layout ([E, H, 2I] and [E, I, H]) holds the
// same experts as the row layout, and loadFusedExperts builds the same matrices from either, in f32 and in each
// quantized mode, with a sink and without one. The shapes are off the read's 16-wide tile edge.
func TestLoadFusedExperts_transposedLayoutGivesTheRowLayout(t *testing.T) {
	const nExpert, inter, hidden = 3, 10, 19
	guStride, downStride := 2*inter*hidden, hidden*inter
	val := func(e, base, i int) float32 { return float32((e*7919+base+i*31)%2003-1001) * 0.00390625 }

	guRows, dnRows := make([]float32, nExpert*guStride), make([]float32, nExpert*downStride)
	guCols, dnCols := make([]float32, nExpert*guStride), make([]float32, nExpert*downStride)
	for e := range nExpert {
		for r := range 2 * inter {
			for c := range hidden {
				v := val(e, 1000, r*hidden+c)
				guRows[e*guStride+r*hidden+c] = v
				guCols[e*guStride+c*2*inter+r] = v
			}
		}
		for r := range hidden {
			for c := range inter {
				v := val(e, 9000, r*inter+c)
				dnRows[e*downStride+r*inter+c] = v
				dnCols[e*downStride+c*hidden+r] = v
			}
		}
	}
	open := func(name string, gu, dn stTensor) *embed.SafetensorsFile {
		path := filepath.Join(t.TempDir(), name+".safetensors")
		writeSafetensors(t, path, map[string]stTensor{"gate_up_proj": gu, "down_proj": dn})
		st, err := embed.OpenSafetensors(path)
		if err != nil {
			t.Fatalf("OpenSafetensors: %v", err)
		}
		t.Cleanup(func() { st.Close() })
		return st
	}
	rows := open("rows", stTensor{[]int{nExpert, 2 * inter, hidden}, guRows}, stTensor{[]int{nExpert, hidden, inter}, dnRows})
	cols := open("cols", stTensor{[]int{nExpert, hidden, 2 * inter}, guCols}, stTensor{[]int{nExpert, inter, hidden}, dnCols})

	for _, quant := range []quantMode{quantNone, quantInt8, quantInt8I8, quantInt4} {
		for _, sink := range []*giwWriter{nil, {sink: io.Discard}} {
			want, err := loadFusedExperts(rows, "gate_up_proj", "down_proj", nExpert, inter, hidden, quant, false, sink)
			if err != nil {
				t.Fatalf("quant %d, row layout: %v", quant, err)
			}
			got, err := loadFusedExperts(cols, "gate_up_proj", "down_proj", nExpert, inter, hidden, quant, false, sink)
			if err != nil {
				t.Fatalf("quant %d, transposed layout: %v", quant, err)
			}
			for e := range nExpert {
				for _, m := range []struct {
					name      string
					want, got linalg.WeightMat
				}{{"gate", want[e].Gate, got[e].Gate}, {"up", want[e].Up, got[e].Up}, {"down", want[e].Down, got[e].Down}} {
					if !bytes.Equal(embedTableBytes(t, m.want), embedTableBytes(t, m.got)) {
						t.Errorf("quant %d, sink %v, expert %d %s: the transposed layout built a different matrix", quant, sink != nil, e, m.name)
					}
				}
			}
		}
	}
}

// TestLoadFusedExperts_f32ExpertsOwnTheirData: an expert kept in f32 never holds a view of the tensor, in either
// layout, so it outlives the mapping. The row layout of an F32 tensor is the case that reads a view.
func TestLoadFusedExperts_f32ExpertsOwnTheirData(t *testing.T) {
	const nExpert, inter, hidden = 2, 4, 6
	guStride, downStride := 2*inter*hidden, hidden*inter
	for _, c := range []struct {
		name   string
		gu, dn []int
	}{
		{"row layout", []int{nExpert, 2 * inter, hidden}, []int{nExpert, hidden, inter}},
		{"transposed layout", []int{nExpert, hidden, 2 * inter}, []int{nExpert, inter, hidden}},
	} {
		path := filepath.Join(t.TempDir(), "model.safetensors")
		writeSafetensors(t, path, map[string]stTensor{
			"gate_up_proj": {c.gu, make([]float32, nExpert*guStride)},
			"down_proj":    {c.dn, make([]float32, nExpert*downStride)},
		})
		st, err := embed.OpenSafetensors(path)
		if err != nil {
			t.Fatalf("OpenSafetensors: %v", err)
		}
		inTensor := func(name string, p *float32) bool {
			tn, err := st.Tensor(name)
			if err != nil {
				t.Fatal(err)
			}
			view, err := tn.SubF32(0, tn.Elements())
			if err != nil {
				t.Fatal(err)
			}
			lo, hi := uintptr(unsafe.Pointer(&view[0])), uintptr(unsafe.Pointer(&view[len(view)-1]))
			at := uintptr(unsafe.Pointer(p))
			return at >= lo && at <= hi
		}
		experts, err := loadFusedExperts(st, "gate_up_proj", "down_proj", nExpert, inter, hidden, quantNone, false, nil)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		for e := range experts {
			gate, _ := experts[e].Gate.F32()
			up, _ := experts[e].Up.F32()
			down, _ := experts[e].Down.F32()
			if len(gate) == 0 || len(up) == 0 || len(down) == 0 {
				t.Fatalf("%s, expert %d: an expert is not f32", c.name, e)
			}
			if inTensor("gate_up_proj", &gate[0]) || inTensor("gate_up_proj", &up[0]) || inTensor("down_proj", &down[0]) {
				t.Errorf("%s, expert %d: an f32 expert is a view of the tensor", c.name, e)
			}
		}
		st.Close()
	}
}
