package decoder

import "testing"

// TestConfigSnapshot_M23 gates M-23: Model.Config() returns a snapshot, so mutating a scalar on
// the returned config cannot desync the model's precomputed caches from the config the forward
// reads. Before the fix Config() returned &m.w.Cfg, so m.Config().NumLayers = N corrupted the
// stable contract silently.
func TestConfigSnapshot_M23(t *testing.T) {
	m := &Model{w: &Weights{Cfg: Config{NumLayers: 18, HiddenDim: 640, VocabSize: 262144}}}
	c := m.Config()
	c.NumLayers = 999
	c.HiddenDim = -1
	if m.w.Cfg.NumLayers != 18 || m.w.Cfg.HiddenDim != 640 {
		t.Errorf("mutating Config() leaked into the model: NumLayers=%d HiddenDim=%d (want 18/640)",
			m.w.Cfg.NumLayers, m.w.Cfg.HiddenDim)
	}
	// A fresh call still reflects the model's real values (not the mutated copy).
	if got := m.Config(); got.NumLayers != 18 {
		t.Errorf("second Config().NumLayers = %d, want 18", got.NumLayers)
	}
}
