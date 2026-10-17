//go:build cuda && goinfer_testhooks

package cuda

import (
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// OP2b step 2a (docs/tasks/task-forward-architecture-2026-10.md §4.7): BuildResident takes each layer's mixer and feed-forward flag from
// decoder.Model.ResidentLayerOps, and dispatches on them. The old per-family predicates (Qwen35LinearLayer, the MLA geometry) are still computed
// and BuildResident declines a layer where they disagree; this test holds the two together on a fixture of every family the builder knows, so the
// derived booleans (isDeltaNet, isMLA) and the dispatch field cannot drift apart before the predicates are retired. TRANSITIONAL: it goes with them.
func TestLayerOps_agreeWithTheFamilyPredicates(t *testing.T) {
	requireCUDADevice(t)
	for _, c := range []struct {
		fixture string
		// want gives the mixer layer l must have, from a source independent of ResidentLayerOps
		want func(m *decoder.Model, l int) decoder.ResidentMixer
	}{
		{"llama-tiny", func(*decoder.Model, int) decoder.ResidentMixer { return decoder.MixerAttention }},
		{"deepseek-tiny", func(*decoder.Model, int) decoder.ResidentMixer { return decoder.MixerMLA }},
		{"qwen35-tiny", func(m *decoder.Model, l int) decoder.ResidentMixer {
			if m.Qwen35LinearLayer(l) {
				return decoder.MixerDeltaNet
			}
			return decoder.MixerAttention
		}},
		{"olmo_hybrid-tiny", func(m *decoder.Model, l int) decoder.ResidentMixer {
			if m.Qwen35LinearLayer(l) {
				return decoder.MixerDeltaNet
			}
			return decoder.MixerAttention
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
				want := c.want(m, l)
				if Ly.mixer != want {
					t.Errorf("layer %d: mixer %v, want %v", l, Ly.mixer, want)
				}
				if !Ly.ffn {
					t.Errorf("layer %d: no feed-forward block, but every family built on CUDA today has one", l)
				}
				if Ly.isDeltaNet != (Ly.mixer == decoder.MixerDeltaNet) || Ly.isMLA != (Ly.mixer == decoder.MixerMLA) {
					t.Errorf("layer %d: the derived booleans (isDeltaNet %v, isMLA %v) disagree with mixer %v", l, Ly.isDeltaNet, Ly.isMLA, Ly.mixer)
				}
				kinds[Ly.mixer]++
			}
			t.Logf("%s: %d layers, by mixer: %v", c.fixture, len(r.layers), kinds)
			if c.fixture == "qwen35-tiny" && (kinds[decoder.MixerDeltaNet] == 0 || kinds[decoder.MixerAttention] == 0) {
				t.Errorf("VOID: %s no longer has both layer kinds (%v), so it cannot show the dispatch", c.fixture, kinds)
			}
		})
	}
}
