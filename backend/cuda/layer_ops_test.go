//go:build cuda && goinfer_testhooks

package cuda

import (
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// OP2b step 2a and 2d (docs/tasks/task-forward-architecture-2026-10.md §4.7): BuildResident takes each layer's mixer and feed-forward flag from
// decoder.Model.ResidentLayerOps and nothing else decides a layer's kind. This test builds a fixture of every family the builder knows and checks each
// layer's built kind against a source independent of ResidentLayerOps (the family's own layer predicate or block kind), so a builder that filled the field
// wrongly, or a ResidentLayerOps that drifted from the family, is red here.
func TestLayerOps_buildTheKindsTheFamilyHas(t *testing.T) {
	requireCUDADevice(t)
	defer nemotronFeaturesForTest()()
	type kind struct {
		mixer decoder.ResidentMixer
		ffn   bool
	}
	for _, c := range []struct {
		fixture string
		// want gives the kind layer l must have, from a source independent of ResidentLayerOps
		want func(m *decoder.Model, l int) kind
	}{
		{"llama-tiny", func(*decoder.Model, int) kind { return kind{decoder.MixerAttention, true} }},
		{"deepseek-tiny", func(*decoder.Model, int) kind { return kind{decoder.MixerMLA, true} }},
		{"qwen35-tiny", func(m *decoder.Model, l int) kind {
			if m.Qwen35LinearLayer(l) {
				return kind{decoder.MixerDeltaNet, true}
			}
			return kind{decoder.MixerAttention, true}
		}},
		{"olmo_hybrid-tiny", func(m *decoder.Model, l int) kind {
			if m.Qwen35LinearLayer(l) {
				return kind{decoder.MixerDeltaNet, true}
			}
			return kind{decoder.MixerAttention, true}
		}},
		{"nemotron-tiny", func(m *decoder.Model, l int) kind {
			switch m.NemotronBlockKind(l) {
			case 0:
				return kind{decoder.MixerMamba, false}
			case 1:
				return kind{decoder.MixerAttention, false}
			}
			return kind{decoder.MixerNone, true}
		}},
	} {
		t.Run(c.fixture, func(t *testing.T) {
			m, err := decoder.Load(filepath.Join("..", "testdata", c.fixture), decoder.Options{Backend: "cuda", Quant: "int4"})
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			defer m.Close()
			r, ok := m.ResidentForwardForTest().(*cudaResident)
			if !ok {
				t.Fatalf("not resident (decode path %q, decline %q): a CPU fallback would pass every check below trivially", m.DecodePath(), m.ResidentDecline())
			}
			kinds := map[decoder.ResidentMixer]int{}
			for l := range r.layers {
				Ly := &r.layers[l]
				if want := c.want(m, l); Ly.mixer != want.mixer || Ly.ffn != want.ffn {
					t.Errorf("layer %d: built as (%v, ffn %v), the family says (%v, ffn %v)", l, Ly.mixer, Ly.ffn, want.mixer, want.ffn)
				}
				kinds[Ly.mixer]++
			}
			t.Logf("%s: %d layers, by mixer: %v", c.fixture, len(r.layers), kinds)
			if (c.fixture == "qwen35-tiny" || c.fixture == "olmo_hybrid-tiny") && (kinds[decoder.MixerDeltaNet] == 0 || kinds[decoder.MixerAttention] == 0) {
				t.Errorf("VOID: %s no longer has both layer kinds (%v), so it cannot show the dispatch", c.fixture, kinds)
			}
			if c.fixture == "nemotron-tiny" && (kinds[decoder.MixerMamba] == 0 || kinds[decoder.MixerAttention] == 0 || kinds[decoder.MixerNone] == 0) {
				t.Errorf("VOID: nemotron-tiny no longer has all three block kinds (%v)", kinds)
			}
		})
	}
}
