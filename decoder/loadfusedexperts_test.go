package decoder

import (
	"path/filepath"
	"testing"

	"github.com/townsendmerino/aikit/embed"
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

	experts, err := loadFusedExperts(st, "gate_up_proj", "down_proj", nExpert, inter, hidden, quantNone)
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

	if _, err := loadFusedExperts(st, "gate_up_proj", "down_proj", nExpert, inter, hidden, quantNone); err == nil {
		t.Fatal("loadFusedExperts should reject a gate_up_proj tensor with the wrong element count")
	}
}
