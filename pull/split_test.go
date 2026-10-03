package pull

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// splitRepo is testdata/gguf-split (glm-tiny.gguf split into four shards by llama-gguf-split) published the way
// bartowski-style repos publish a big quant: its shards in their own folder, beside a single-file quant at the top.
func splitRepo(t *testing.T) map[string][]byte {
	t.Helper()
	files := map[string][]byte{}
	for i := 1; i <= 4; i++ {
		b, err := os.ReadFile(fmt.Sprintf("../testdata/gguf-split/glm-tiny-%05d-of-00004.gguf", i))
		if err != nil {
			t.Skipf("no split fixture: %v", err)
		}
		files[fmt.Sprintf("Q8_0/glm-tiny-Q8_0-%05d-of-00004.gguf", i)] = b
	}
	one, err := os.ReadFile("../testdata/glm-tiny.gguf")
	if err != nil {
		t.Skipf("no tiny GGUF: %v", err)
	}
	files["glm-tiny-Q4_K_M.gguf"] = one
	return files
}

// TestSplitGGUF_pullFetchesTheSetAndLoads is P4's pull half (task-checkpoint-fetch-2026-09.md): a split quant filed in
// a folder is listed (the recursive listing), selected whole by its quant, fetched shard by shard, and the path Resolve
// returns loads and decodes. A transfer cut inside shard 3 leaves a set the loader refuses by naming the missing shard,
// and the re-run fetches only what is missing.
func TestSplitGGUF_pullFetchesTheSetAndLoads(t *testing.T) {
	withCacheRoot(t)
	h := &fakeHF{files: splitRepo(t), gated: false, failOnce: "Q8_0/glm-tiny-Q8_0-00003-of-00004.gguf"}
	h.serve(t, "o/split")

	files, err := List(context.Background(), "o/split")
	if err != nil {
		t.Fatal(err)
	}
	rows := Collapse(files)
	if len(files) != 5 || len(rows) != 2 || rows[0].Shards != 4 || rows[0].Size != SetBytes(files[:4]) {
		t.Fatalf("listing %d files as %+v; want 5 files collapsing to the 4-shard set and the single file", len(files), rows)
	}

	if _, err := Resolve(context.Background(), "hf:o/split:q8_0", nil); err == nil || !strings.Contains(err.Error(), "shard 3 of 4") {
		t.Fatalf("a transfer cut in shard 3: %v, want an error naming shard 3 of 4", err)
	}
	dir, _ := CacheDir("o/split")
	first := filepath.Join(dir, "glm-tiny-Q8_0-00001-of-00004.gguf")
	if _, err := decoder.Load(first, decoder.Options{Backend: "cpu"}); err == nil || !strings.Contains(err.Error(), "shard 3 of 4") {
		t.Fatalf("loading the incomplete set: %v, want a refusal naming the missing shard", err)
	}

	path, err := Resolve(context.Background(), "hf:o/split:q8_0", nil)
	if err != nil {
		t.Fatalf("re-run: %v", err)
	}
	if path != first {
		t.Fatalf("resolved %s, want the first shard %s", path, first)
	}
	for i := 1; i <= 4; i++ {
		p := fmt.Sprintf("Q8_0/glm-tiny-Q8_0-%05d-of-00004.gguf", i)
		want := 1
		if i == 3 {
			want = 2 // cut once, fetched again
		}
		if h.hits[p] != want {
			t.Errorf("%s fetched %d times, want %d (the re-run must fetch only what is missing)", p, h.hits[p], want)
		}
	}
	m, err := decoder.Load(path, decoder.Options{Backend: "cpu", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load the fetched set: %v", err)
	}
	defer m.Close()
	ch, g := m.Generate(context.Background(), []int{1, 2, 3}, 1, decoder.SamplingParams{})
	n := 0
	for range ch {
		n++
	}
	if err := g.Err(); err != nil || n != 1 {
		t.Fatalf("the fetched set decoded %d tokens (%v), want 1", n, err)
	}
}

// TestSelectSet: a quant or any shard's exact name selects the whole set in order, a single file selects itself, and a
// set the listing holds only part of is refused rather than fetched short. Select keeps its one-file contract.
func TestSelectSet(t *testing.T) {
	files := []File{
		{Path: "Q8_0/big-Q8_0-00002-of-00003.gguf", Size: 4},
		{Path: "Q8_0/big-Q8_0-00001-of-00003.gguf", Size: 4},
		{Path: "Q8_0/big-Q8_0-00003-of-00003.gguf", Size: 2},
		{Path: "big-Q4_K_S.gguf", Size: 8},
	}
	paths := func(set []File) string {
		var p []string
		for _, f := range set {
			p = append(p, filepath.Base(f.Path))
		}
		return strings.Join(p, " ")
	}
	want := "big-Q8_0-00001-of-00003.gguf big-Q8_0-00002-of-00003.gguf big-Q8_0-00003-of-00003.gguf"
	for _, ref := range []Ref{{Repo: "a/b", Quant: "q8_0"}, {Repo: "a/b", File: "Q8_0/big-Q8_0-00002-of-00003.gguf"}} {
		set, err := SelectSet(files, ref)
		if err != nil || paths(set) != want {
			t.Errorf("SelectSet(%+v) = %s, %v; want %s", ref, paths(set), err, want)
		}
	}
	if set, err := SelectSet(files, Ref{Repo: "a/b", Quant: "q4_k_s"}); err != nil || paths(set) != "big-Q4_K_S.gguf" {
		t.Errorf("single file: %s, %v", paths(set), err)
	}
	if _, err := SelectSet(files[1:], Ref{Repo: "a/b", Quant: "q8_0"}); err == nil || !strings.Contains(err.Error(), "incomplete") {
		t.Errorf("a set missing shard 2: %v, want it refused as incomplete", err)
	}
	if _, err := Select(files, Ref{Repo: "a/b", Quant: "q8_0"}); err == nil || !strings.Contains(err.Error(), "split GGUF") {
		t.Errorf("Select on a split quant: %v, want its one-file refusal", err)
	}
}
