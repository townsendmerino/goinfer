package serveapp

import (
	"encoding/json"
	"runtime"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// MEM13 (docs/tasks/task-memory-campaign-2026-10.md): the job store keeps a capped number of jobs, each with its prompt's token
// ids, and an adapter reads its base model's token-bytes table instead of building its own.

// A job keeps the prompt ids as int32: the same JSON, half the retained bytes of the []int the request carried. The bound is on the
// heap retained by a store full of 20K-token jobs once the request slices are gone.
func TestJobStore_promptIDsRetainFourBytesEach(t *testing.T) {
	const jobs, ids = 50, 20000
	js := &jobStore{jobs: map[string]*job{}, cap: jobs}
	heap := func() uint64 {
		runtime.GC()
		var ms runtime.MemStats
		runtime.ReadMemStats(&ms)
		return ms.HeapAlloc
	}
	before := heap()
	for i := 0; i < jobs; i++ {
		req := make([]int, ids)
		for k := range req {
			req[k] = k % 150000
		}
		js.createPending(string(rune('a'+i%26))+strings.Repeat("x", i/26), "m", req)
	}
	retained := heap() - before
	runtime.KeepAlive(js)
	if want := uint64(jobs * ids * 4); retained > want+want/10 {
		t.Errorf("a store of %d jobs x %d ids retains %d bytes; int32 ids are %d", jobs, ids, retained, want)
	}
	b, err := json.Marshal(&job{ID: "j", PromptIDs: []int32{1, 150000, 7}})
	if err != nil || !strings.Contains(string(b), `"prompt_ids":[1,150000,7]`) {
		t.Errorf("the job's JSON changed shape: %s (%v)", b, err)
	}
}

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
