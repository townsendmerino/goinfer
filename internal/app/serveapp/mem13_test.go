package serveapp

import (
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// MEM13 (docs/completed/task-memory-campaign-2026-10.md): the job store keeps a capped number of jobs, each with its prompt's token
// ids, and an adapter reads its base model's token-bytes table instead of building its own.

// loadAdapters gives every adapter its base's token-bytes table, built once. Through the real caller: the adapters of
// TestServe_multiAdapter's synthetic base.
func TestLoadAdapters_adaptersShareTheBaseTokenBytes(t *testing.T) {
	base := buildSyntheticBase(t)
	a1, a2 := buildSyntheticAdapter(t, +1), buildSyntheticAdapter(t, -1)
	m, err := decoder.Load(base, decoder.Options{Backend: "cpu"})
	if err != nil {
		t.Fatalf("decoder.Load base: %v", err)
	}
	defer m.Close()
	baseLM := &loadedModel{model: m, name: "base", fp: "base", sessions: newSessionLRU(m, 4, 0, "base"), tk: scriptTokenizer(t), vocab: 40}
	srv := &server{models: map[string]*loadedModel{"base": baseLM}, cfg: config{}}
	if err := srv.loadAdapters(config{adapters: adapterFlag{{"a1", "base", a1}, {"a2", "base", a2}}, kvSessions: 4}); err != nil {
		t.Fatalf("loadAdapters: %v", err)
	}
	bt := baseLM.cachedTokenBytes()
	if len(bt) == 0 {
		t.Fatal("the base's table is empty")
	}
	for _, n := range []string{"a1", "a2"} {
		at := srv.models[n].cachedTokenBytes()
		if len(at) != len(bt) || &at[0] != &bt[0] {
			t.Errorf("adapter %s builds its own token-bytes table; it should read the base's", n)
		}
	}
}
