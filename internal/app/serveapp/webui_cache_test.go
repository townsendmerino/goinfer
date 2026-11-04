package serveapp

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

// TestWebCache_listsWhatIsOnDiskAndWhatIsLoaded is P8's page half (task-checkpoint-fetch-2026-09.md): the "On disk" route
// lists every model the pull cache holds (a GGUF, a checkpoint, an interrupted pull), and marks the one a loaded model was
// loaded FROM with its served name, through its resolved source path (here reached through a symlinked cache path).
func TestWebCache_listsWhatIsOnDiskAndWhatIsLoaded(t *testing.T) {
	_, root := fakeCache(t)
	gguf := writeCacheFile(t, filepath.Join(root, "o", "g", "m-Q4_K_M.gguf"))
	writeCacheFile(t, filepath.Join(root, "o", "g", "m-Q8_0.gguf.part"))
	writeCheckpoint(t, filepath.Join(root, "o", "ckpt"), "o/ckpt")
	link := filepath.Join(t.TempDir(), "via-link.gguf")
	if err := os.Symlink(gguf, link); err != nil {
		t.Fatal(err)
	}
	s := &server{models: map[string]*loadedModel{"m": {name: "m", source: link}, "adapter": {name: "adapter"}}}
	s.cfg.web = true
	w := httptest.NewRecorder()
	s.handleWebCache(w, httptest.NewRequest(http.MethodGet, "/web/models/cache", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("%d %s", w.Code, w.Body.String())
	}
	var j struct {
		Entries []struct {
			Repo, Path, Kind, Loaded string
			Complete                 bool
		} `json:"entries"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &j); err != nil {
		t.Fatal(err)
	}
	got := map[string]string{}
	for _, e := range j.Entries {
		rel, _ := filepath.Rel(root, e.Path)
		got[filepath.ToSlash(rel)] = e.Kind + " complete=" + map[bool]string{true: "y", false: "n"}[e.Complete] + " loaded=" + e.Loaded
	}
	want := map[string]string{
		"o/ckpt":            "checkpoint complete=y loaded=",
		"o/g/m-Q4_K_M.gguf": "gguf complete=y loaded=m",
		"o/g/m-Q8_0.gguf":   "gguf complete=n loaded=",
	}
	if len(got) != len(want) {
		t.Fatalf("entries %v, want %v", got, want)
	}
	for k, v := range want {
		if got[k] != v {
			t.Errorf("%s: %q, want %q (all: %v)", k, got[k], v, got)
		}
	}
}
