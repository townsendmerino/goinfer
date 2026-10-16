package decoder

import (
	"path/filepath"
	"testing"
)

func loadTinyOps(t *testing.T, dir string) *Model {
	t.Helper()
	m, err := Load(filepath.Join("..", "testdata", dir), Options{})
	if err != nil {
		t.Fatalf("load %s: %v", dir, err)
	}
	t.Cleanup(func() { _ = m.Close() })
	return m
}

// TestResidentLayerOps_kinds: each family's layers come out as the operations its forward runs. Nemotron-H's blocks carry one
// operation each (Mamba, attention, or an FFN with no mixer); a Qwen3.5 linear layer is DeltaNet with an FFN; a dense model
// is attention with an FFN on every layer.
func TestResidentLayerOps_kinds(t *testing.T) {
	nemo := loadTinyOps(t, "nemotron-tiny")
	seen := map[ResidentMixer]int{}
	for l, k := range nemo.w.arch.nemotron.blockKind {
		mixer, ffn := nemo.ResidentLayerOps(l)
		var want ResidentMixer
		wantFFN := false
		switch k {
		case nemoMamba:
			want = MixerMamba
		case nemoAttn:
			want = MixerAttention
		default:
			want, wantFFN = MixerNone, true
		}
		if mixer != want || ffn != wantFFN {
			t.Errorf("nemotron-tiny layer %d (block kind %d): ops (%d, %v), want (%d, %v)", l, k, mixer, ffn, want, wantFFN)
		}
		seen[mixer]++
	}
	if seen[MixerMamba] == 0 || seen[MixerAttention] == 0 || seen[MixerNone] == 0 {
		t.Fatalf("nemotron-tiny does not reach every block kind (%v): the test cannot see a misclassified one", seen)
	}
	q := loadTinyOps(t, "qwen35-tiny")
	delta := 0
	for l := range q.w.arch.NumLayers {
		mixer, ffn := q.ResidentLayerOps(l)
		if want := map[bool]ResidentMixer{true: MixerDeltaNet, false: MixerAttention}[q.w.arch.isLinearLayer(l)]; mixer != want || !ffn {
			t.Errorf("qwen35-tiny layer %d: ops (%d, %v), want (%d, true)", l, mixer, ffn, want)
		}
		if mixer == MixerDeltaNet {
			delta++
		}
	}
	if delta == 0 {
		t.Fatal("qwen35-tiny has no DeltaNet layer")
	}
	ll := loadTinyOps(t, "llama-tiny")
	for l := range ll.w.arch.NumLayers {
		if mixer, ffn := ll.ResidentLayerOps(l); mixer != MixerAttention || !ffn {
			t.Errorf("llama-tiny layer %d: ops (%d, %v), want attention with an FFN", l, mixer, ffn)
		}
	}
}

// TestResidentKVBytes_attentionLayersOnly is OP2b's KV-pricing gate: on Metal a Mamba-2 layer and a mixer-less FFN block price
// at zero KV, as a DeltaNet layer always has, and an attention layer at its padded f16 K and V, as before. Each fixture's total
// must equal that sum over its own attention layers.
func TestResidentKVBytes_attentionLayersOnly(t *testing.T) {
	const ctx = 1000
	padded := int64((ctx + metalKVPad - 1) / metalKVPad * metalKVPad)
	for _, dir := range []string{"nemotron-tiny", "qwen35-tiny", "llama-tiny"} {
		m := loadTinyOps(t, dir)
		var want int64
		attn := 0
		for l := range m.w.arch.NumLayers {
			if mixer, _ := m.ResidentLayerOps(l); mixer != MixerAttention {
				continue
			}
			attn++
			want += 2 * padded * int64(m.KVHeadsAtResident(l)*m.HeadDimAtResident(l)) * 2
		}
		if got := m.ResidentKVBytes("metal", ctx, false, false); got != want {
			t.Errorf("%s: ResidentKVBytes(metal) = %d, want %d (%d attention layers of %d)", dir, got, want, attn, m.w.arch.NumLayers)
		}
		if dir != "llama-tiny" && attn == m.w.arch.NumLayers {
			t.Fatalf("%s has no non-attention layer: the gate cannot see one priced", dir)
		}
	}
}

// TestRopeInvFreqLayer_nemotronIsNoPE: Nemotron-H's attention ropes nothing on the CPU, so the table a resident reads for it
// is all zeros (the identity rotation); a roped family's table is unchanged.
func TestRopeInvFreqLayer_nemotronIsNoPE(t *testing.T) {
	nemo := loadTinyOps(t, "nemotron-tiny")
	checked := 0
	for l := range nemo.w.arch.NumLayers {
		if mixer, _ := nemo.ResidentLayerOps(l); mixer != MixerAttention {
			continue
		}
		checked++
		for j, v := range nemo.RopeInvFreqLayerResident(l) {
			if v != 0 {
				t.Fatalf("nemotron-tiny attention layer %d: inv-freq[%d] = %g, want 0 (NoPE)", l, j, v)
			}
		}
	}
	if checked == 0 {
		t.Fatal("nemotron-tiny has no attention layer")
	}
	ll := loadTinyOps(t, "llama-tiny")
	if f := ll.RopeInvFreqLayerResident(0); len(f) == 0 || f[0] == 0 {
		t.Errorf("llama-tiny layer 0's RoPE table is %v: a roped family must keep its table", f)
	}
}
