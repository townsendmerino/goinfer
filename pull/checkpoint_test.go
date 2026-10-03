package pull

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// fakeHF serves a repo's tree, its files (with Range, as HF's resolve host does) and its metadata, counts requests per
// file, and can fail one file's transfer midway.
type fakeHF struct {
	mu       sync.Mutex
	files    map[string][]byte
	gated    any
	hits     map[string]int
	failOnce string // a path whose next transfer is cut midway
}

func (h *fakeHF) serve(t *testing.T, repo string) {
	t.Helper()
	h.hits = map[string]int{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		h.mu.Lock()
		defer h.mu.Unlock()
		switch {
		case r.URL.Path == "/api/models/"+repo:
			_ = json.NewEncoder(w).Encode(map[string]any{"gated": h.gated, "private": false})
		case r.URL.Path == "/api/models/"+repo+"/tree/main":
			var out []map[string]any
			for p, b := range h.files {
				e := map[string]any{"type": "file", "path": p, "size": len(b)}
				if strings.HasSuffix(p, ".safetensors") || strings.HasSuffix(p, ".bin") || strings.HasSuffix(p, ".gguf") {
					s := sha256.Sum256(b)
					e["lfs"] = map[string]any{"oid": hex.EncodeToString(s[:])}
				}
				out = append(out, e)
			}
			_ = json.NewEncoder(w).Encode(out)
		case strings.HasPrefix(r.URL.Path, "/"+repo+"/resolve/main/"):
			p := strings.TrimPrefix(r.URL.Path, "/"+repo+"/resolve/main/")
			b, ok := h.files[p]
			if !ok {
				http.NotFound(w, r)
				return
			}
			h.hits[p]++
			if p == h.failOnce {
				h.failOnce = ""
				w.Header().Set("Content-Length", "999999999")
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write(b[:len(b)/2])
				return
			}
			http.ServeContent(w, r, p, time.Unix(0, 0), bytes.NewReader(b))
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(srv.Close)
	oa, oc := hfAPI, hfCDN
	hfAPI, hfCDN = srv.URL+"/api/models", srv.URL
	t.Cleanup(func() { hfAPI, hfCDN = oa, oc })
}

// splitSafetensors splits one safetensors file into two (tensors alternated between them) and returns the shards and an
// index naming each tensor's shard, the shape of a real sharded checkpoint.
func splitSafetensors(t *testing.T, raw []byte) (a, b, index []byte) {
	t.Helper()
	n := binary.LittleEndian.Uint64(raw[:8])
	var hdr map[string]json.RawMessage
	if err := json.Unmarshal(raw[8:8+n], &hdr); err != nil {
		t.Fatal(err)
	}
	data := raw[8+n:]
	var names []string
	for k := range hdr {
		if k != "__metadata__" {
			names = append(names, k)
		}
	}
	sort.Strings(names)
	type ent struct {
		Dtype   string  `json:"dtype"`
		Shape   []int64 `json:"shape"`
		Offsets [2]int  `json:"data_offsets"`
	}
	build := func(keep func(i int) bool) []byte {
		h := map[string]ent{}
		var body []byte
		for i, k := range names {
			if !keep(i) {
				continue
			}
			var e ent
			if err := json.Unmarshal(hdr[k], &e); err != nil {
				t.Fatal(err)
			}
			chunk := data[e.Offsets[0]:e.Offsets[1]]
			e.Offsets = [2]int{len(body), len(body) + len(chunk)}
			body = append(body, chunk...)
			h[k] = e
		}
		hj, _ := json.Marshal(h)
		for len(hj)%8 != 0 {
			hj = append(hj, ' ')
		}
		out := make([]byte, 8, 8+len(hj)+len(body))
		binary.LittleEndian.PutUint64(out, uint64(len(hj)))
		return append(append(out, hj...), body...)
	}
	a, b = build(func(i int) bool { return i%2 == 0 }), build(func(i int) bool { return i%2 == 1 })
	wm := map[string]string{}
	for i, k := range names {
		wm[k] = map[bool]string{true: "model-00001-of-00002.safetensors", false: "model-00002-of-00002.safetensors"}[i%2 == 0]
	}
	index, _ = json.Marshal(map[string]any{"metadata": map[string]any{}, "weight_map": wm})
	return a, b, index
}

// shardedTinyRepo is testdata/llama-tiny as a sharded HF repo, plus files a plan must leave out.
func shardedTinyRepo(t *testing.T) map[string][]byte {
	t.Helper()
	raw, err := os.ReadFile("../testdata/llama-tiny/model.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := os.ReadFile("../testdata/llama-tiny/config.json")
	if err != nil {
		t.Fatal(err)
	}
	a, b, idx := splitSafetensors(t, raw)
	return map[string][]byte{
		"config.json":                      cfg,
		"model.safetensors.index.json":     idx,
		"model-00001-of-00002.safetensors": a,
		"model-00002-of-00002.safetensors": b,
		"README.md":                        []byte("# a readme"),
		"pytorch_model.bin":                []byte("legacy weights a plan must not fetch"),
	}
}

func withCacheRoot(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	t.Setenv("XDG_CACHE_HOME", root)
	t.Setenv("HOME", root) // os.UserCacheDir on darwin is $HOME/Library/Caches
	return root
}

// TestCheckpoint_shardedRoundTrip is P9's first gate (docs/tasks/task-checkpoint-fetch-2026-09.md): a safetensors
// repo with a shard index goes plan -> fetch -> load -> one token. The plan takes config, index and both shards and
// leaves the README and the legacy .bin out; the fetched directory is what decoder.Load opens; a second Resolve is
// answered from the cache with no request.
func TestCheckpoint_shardedRoundTrip(t *testing.T) {
	withCacheRoot(t)
	h := &fakeHF{files: shardedTinyRepo(t), gated: false}
	h.serve(t, "o/tiny")
	p, err := PlanCheckpoint(context.Background(), "o/tiny")
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	var paths []string
	for _, f := range p.Files {
		paths = append(paths, f.Path)
	}
	want := "config.json model-00001-of-00002.safetensors model-00002-of-00002.safetensors model.safetensors.index.json"
	if strings.Join(paths, " ") != want || p.Family != "llama" {
		t.Fatalf("plan files %v family %q; want %s and llama", paths, p.Family, want)
	}
	dir, err := Resolve(context.Background(), "hf:o/tiny:safetensors", nil)
	if err != nil {
		t.Fatalf("resolve: %v", err)
	}
	if _, err := os.Stat(dir + ".partial"); !os.IsNotExist(err) {
		t.Fatalf("the staging directory survived a completed pull: %v", err)
	}
	m, err := decoder.Load(dir, decoder.Options{Backend: "cpu"})
	if err != nil {
		t.Fatalf("decoder.Load on the fetched directory: %v", err)
	}
	defer m.Close()
	ch, g := m.Generate(context.Background(), []int{1, 2, 3}, 1, decoder.SamplingParams{})
	n := 0
	for range ch {
		n++
	}
	if g.Err() != nil || n != 1 {
		t.Fatalf("one token from the fetched checkpoint: %d tokens, err %v", n, g.Err())
	}
	before := 0
	for _, c := range h.hits {
		before += c
	}
	if _, err := Resolve(context.Background(), "hf:o/tiny:safetensors", nil); err != nil {
		t.Fatalf("second resolve: %v", err)
	}
	after := 0
	for _, c := range h.hits {
		after += c
	}
	if after != before {
		t.Fatalf("a second resolve of a complete checkpoint fetched %d more files", after-before)
	}
}

// TestCheckpoint_interruptedSetNeverPublishes is the gate the doc names as mattering most: a transfer that dies inside
// the set leaves NOTHING at the final path (only the staging directory), and the re-run resumes without re-fetching
// the files already verified.
func TestCheckpoint_interruptedSetNeverPublishes(t *testing.T) {
	withCacheRoot(t)
	h := &fakeHF{files: shardedTinyRepo(t), gated: false, failOnce: "model-00002-of-00002.safetensors"}
	h.serve(t, "o/tiny")
	dir, _ := CacheDir("o/tiny")
	if _, err := Resolve(context.Background(), "hf:o/tiny:safetensors", nil); err == nil {
		t.Fatal("a transfer cut midway reported success")
	}
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Fatalf("an incomplete set published %s (stat err %v)", dir, err)
	}
	if _, err := os.Stat(dir + ".partial"); err != nil {
		t.Fatalf("the staging directory is gone, so nothing can resume: %v", err)
	}
	firstShard := h.hits["model-00001-of-00002.safetensors"]
	if _, err := Resolve(context.Background(), "hf:o/tiny:safetensors", nil); err != nil {
		t.Fatalf("resume: %v", err)
	}
	if h.hits["model-00001-of-00002.safetensors"] != firstShard {
		t.Fatalf("the resume re-fetched shard 1 (%d -> %d requests)", firstShard, h.hits["model-00001-of-00002.safetensors"])
	}
	if _, ok := CachedCheckpoint(dir); !ok {
		t.Fatal("the resumed set did not publish a complete checkpoint")
	}
}

// TestCheckpoint_unsupportedDeclinedBeforeAnyWeight: a model_type this build cannot load is refused after reading
// config.json and before any weight file is requested.
func TestCheckpoint_unsupportedDeclinedBeforeAnyWeight(t *testing.T) {
	withCacheRoot(t)
	files := shardedTinyRepo(t)
	files["config.json"] = []byte(`{"model_type": "made_up_arch", "torch_dtype": "bfloat16"}`)
	h := &fakeHF{files: files, gated: false}
	h.serve(t, "o/tiny")
	_, err := Resolve(context.Background(), "hf:o/tiny:safetensors", nil)
	if err == nil || !strings.Contains(err.Error(), "made_up_arch") {
		t.Fatalf("an unsupported model_type: err %v, want a refusal naming it", err)
	}
	for p, c := range h.hits {
		if p != "config.json" && c > 0 {
			t.Fatalf("%s was requested (%d) before the refusal", p, c)
		}
	}
}

// TestCheckpoint_gatedDeclinedBeforeAnyRequest: a gated original (the doc's §2, option (c): anonymous only) is refused
// by CheckAccess before the tree or any file is read.
func TestCheckpoint_gatedDeclinedBeforeAnyRequest(t *testing.T) {
	withCacheRoot(t)
	h := &fakeHF{files: shardedTinyRepo(t), gated: "manual"}
	h.serve(t, "o/tiny")
	if _, err := Resolve(context.Background(), "hf:o/tiny:safetensors", nil); err == nil || !strings.Contains(err.Error(), "gated") {
		t.Fatalf("a gated repo: err %v, want the gated refusal", err)
	}
	if len(h.hits) != 0 {
		t.Fatalf("a gated repo had files requested: %v", h.hits)
	}
}

// TestCheckpoint_sizeNoteOnFullPrecision: the plan states a bf16 original's download cost before the transfer.
func TestCheckpoint_sizeNoteOnFullPrecision(t *testing.T) {
	p := Plan{Files: []File{{Path: "model.safetensors", Size: 15e9}}, Bytes: 15e9, Dtype: "bfloat16"}
	if n := p.SizeNote(); !strings.Contains(n, "full-precision") || !strings.Contains(n, "quarter") {
		t.Fatalf("size note for a bf16 original: %q", n)
	}
	if n := (Plan{Files: []File{{Size: 1}}, Bytes: 1, Dtype: "int8"}).SizeNote(); strings.Contains(n, "full-precision") {
		t.Fatalf("a non-full-precision plan carries the full-precision warning: %q", n)
	}
}

// TestCheckpoint_badIndexAndPathsRefused: a shard index naming a file the repo lacks, and an unsafe path, are refused
// at plan time.
func TestCheckpoint_badIndexAndPathsRefused(t *testing.T) {
	withCacheRoot(t)
	files := shardedTinyRepo(t)
	delete(files, "model-00002-of-00002.safetensors")
	h := &fakeHF{files: files, gated: false}
	h.serve(t, "o/tiny")
	if _, err := PlanCheckpoint(context.Background(), "o/tiny"); err == nil || !strings.Contains(err.Error(), "does not have") {
		t.Fatalf("an index naming a missing shard: err %v", err)
	}
	for _, p := range []string{"../x", "/abs", "a/../b", "a\\b", ""} {
		if safeRepoPath(p) {
			t.Fatalf("safeRepoPath(%q) = true", p)
		}
	}
	if !safeRepoPath("vision/model.safetensors") {
		t.Fatal("a nested relative path was refused")
	}
}

// TestCheckpoint_existingNonCheckpointRefused: a directory at the destination that goinfer did not complete is refused,
// not overwritten.
func TestCheckpoint_existingNonCheckpointRefused(t *testing.T) {
	withCacheRoot(t)
	h := &fakeHF{files: shardedTinyRepo(t), gated: false}
	h.serve(t, "o/tiny")
	dir, _ := CacheDir("o/tiny")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "mine.txt"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Resolve(context.Background(), "hf:o/tiny:safetensors", nil); err == nil || !strings.Contains(err.Error(), "not a complete checkpoint") {
		t.Fatalf("an existing foreign directory: err %v", err)
	}
	if b, _ := os.ReadFile(filepath.Join(dir, "mine.txt")); string(b) != "x" {
		t.Fatal("the existing directory's file was touched")
	}
}
