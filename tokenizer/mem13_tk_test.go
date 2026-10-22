package tokenizer

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"runtime"
	"slices"
	"strings"
	"testing"
)

// TK-G (docs/tasks/task-memory-campaign-2026-10.md, §4 MEM13, MEM13-tokenizer gates): the rank map keyed by ids, the typed
// metadata reader, and what they must leave unchanged.

var updateTK = flag.Bool("update-tk", false, "rewrite testdata/mem13_tk_ids.json from this tree")

// tkFixtures are the committed GGUFs that carry a tokenizer.
var tkFixtures = []string{"testdata/qwen2-0.5b-coder-meta.gguf", "testdata/granite-dbrx-meta.gguf", "testdata/chatml-tiny.gguf"}

// tkCorpus covers what a merge loop treats differently: words, code, digit runs, whitespace runs, CJK, emoji and
// accented text, and an added token.
var tkCorpus = strings.Repeat(`The quick brown fox jumps over the lazy dog. It's 2026-10-21, and 1234567 apples cost $89.99.
func main() {
	for i := 0; i < 10; i++ { fmt.Println("héllo, wörld", i) }
}
    indented   with    runs of spaces		and tabs
日本語のテキストと中文文本，还有한국어 텍스트。 Ünïcödé ñ çà 🦊🚀✨ <|im_start|>user
`, 3)

func tkLoad(t testing.TB, path string) *Tokenizer {
	t.Helper()
	tk, err := LoadGGUF(path)
	if err != nil {
		t.Fatalf("LoadGGUF(%s): %v", path, err)
	}
	return tk
}

// TestTK_idsPinned is TK-G1's id half: the corpus's token ids on each fixture equal the ones pinned from main before the
// change.
func TestTK_idsPinned(t *testing.T) {
	const golden = "testdata/mem13_tk_ids.json"
	got := map[string][]int{}
	for _, f := range tkFixtures {
		ids, err := tkLoad(t, f).Encode(tkCorpus, false)
		if err != nil {
			t.Fatalf("%s: Encode: %v", f, err)
		}
		got[f] = ids
	}
	if *updateTK {
		b, _ := json.Marshal(got)
		if err := os.WriteFile(golden, b, 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	b, err := os.ReadFile(golden)
	if err != nil {
		t.Fatal(err)
	}
	var want map[string][]int
	if err := json.Unmarshal(b, &want); err != nil {
		t.Fatal(err)
	}
	for _, f := range tkFixtures {
		if len(want[f]) == 0 {
			t.Fatalf("%s: no pinned ids", f)
		}
		if !slices.Equal(got[f], want[f]) {
			t.Errorf("%s: ids differ from the pinned ones (%d ids, pinned %d)", f, len(got[f]), len(want[f]))
		}
	}
}

// TestTK_loadMemory is TK-G2's instrument: the bytes LoadGGUF allocates and the heap the tokenizer keeps, each the median
// of 5 loads of the Qwen fixture. It reports; the gate compares the reading with main's.
func TestTK_loadMemory(t *testing.T) {
	const path = "testdata/qwen2-0.5b-coder-meta.gguf"
	var alloc, kept []uint64
	for range 5 {
		var a, b runtime.MemStats
		runtime.GC()
		runtime.ReadMemStats(&a)
		tk := tkLoad(t, path)
		runtime.ReadMemStats(&b)
		runtime.GC()
		var c runtime.MemStats
		runtime.ReadMemStats(&c)
		alloc = append(alloc, b.TotalAlloc-a.TotalAlloc)
		kept = append(kept, c.HeapAlloc-a.HeapAlloc)
		runtime.KeepAlive(tk)
	}
	slices.Sort(alloc)
	slices.Sort(kept)
	fmt.Fprintf(os.Stderr, "TK-G2 %s: allocated %.2f MiB, kept %.2f MiB (median of 5)\n", path, float64(alloc[2])/(1<<20), float64(kept[2])/(1<<20))
}

// BenchmarkEncode is TK-G3's instrument: the corpus encoded on the Qwen fixture.
func BenchmarkEncode(b *testing.B) {
	tk := tkLoad(b, "testdata/qwen2-0.5b-coder-meta.gguf")
	b.SetBytes(int64(len(tkCorpus)))
	for b.Loop() {
		if _, err := tk.Encode(tkCorpus, false); err != nil {
			b.Fatal(err)
		}
	}
}
