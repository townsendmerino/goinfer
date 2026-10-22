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

	"github.com/townsendmerino/aikit/embed"
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

// TestTK_loadMemory is TK-G2's instrument: the bytes LoadGGUF allocates and the heap the tokenizer keeps (after two GCs),
// each the median of 5 loads of the Qwen fixture. It reports; the gate compares the reading with main's.
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
		runtime.GC() // twice: an object with a finalizer (aikit's GGUFFile) and all it reaches outlive the first
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

// TestTK_typedReaderMatchesAikit is TK-G1's reader half: on each committed tokenizer GGUF, every key the tokenizer reads
// answers through ggufMeta as it does through aikit's embed.GGUFFile and the boxed-array readers ggufMeta replaced.
func TestTK_typedReaderMatchesAikit(t *testing.T) {
	for _, f := range tkFixtures {
		g, err := embed.OpenGGUF(f)
		if err != nil {
			t.Fatal(err)
		}
		m, err := readGGUFMetaFile(f)
		if err != nil {
			t.Fatal(err)
		}
		checked := 0
		for key := range ggufMetaKeys {
			_, present := g.Metadata[key]
			if m.present[key] != present {
				t.Errorf("%s %s: present %v, aikit %v", f, key, m.present[key], present)
			}
			if !present {
				continue
			}
			checked++
			s1, ok1 := m.Str(key)
			s2, ok2 := g.Str(key)
			u1, uok1 := m.Uint(key)
			u2, uok2 := g.Uint(key)
			b1, bok1 := m.Bool(key)
			b2, bok2 := g.Metadata[key].(bool)
			if s1 != s2 || ok1 != ok2 || u1 != u2 || uok1 != uok2 || b1 != b2 || bok1 != bok2 {
				t.Errorf("%s %s: scalar (%q %v, %d %v, %v %v), aikit (%q %v, %d %v, %v %v)", f, key, s1, ok1, u1, uok1, b1, bok1, s2, ok2, u2, uok2, b2, bok2)
			}
			sa1, e1 := m.stringArray(key)
			sa2, e2 := aikitStringArray(g, key)
			if (e1 == nil) != (e2 == nil) || !slices.Equal(sa1, sa2) {
				t.Errorf("%s %s: string array (%d, err %v), aikit (%d, err %v)", f, key, len(sa1), e1, len(sa2), e2)
			}
			if !slices.Equal(m.intArray(key), aikitIntArray(g, key)) {
				t.Errorf("%s %s: int array differs", f, key)
			}
			if !slices.Equal(m.floatArray(key), aikitFloatArray(g, key)) {
				t.Errorf("%s %s: float array differs", f, key)
			}
		}
		if checked < 4 {
			t.Errorf("%s: only %d of the tokenizer's keys present; the comparison checked almost nothing", f, checked)
		}
		g.Close()
	}
}

// aikitStringArray, aikitIntArray and aikitFloatArray are the readers ggufMeta replaced, over aikit's boxed metadata.
func aikitStringArray(g *embed.GGUFFile, key string) ([]string, error) {
	arr, ok := g.Metadata[key].([]any)
	if !ok {
		return nil, fmt.Errorf("%s missing or not an array", key)
	}
	out := make([]string, len(arr))
	for i, v := range arr {
		s, ok := v.(string)
		if !ok {
			return nil, fmt.Errorf("%s[%d] is %T, want string", key, i, v)
		}
		out[i] = s
	}
	return out, nil
}

func aikitIntArray(g *embed.GGUFFile, key string) []int {
	arr, ok := g.Metadata[key].([]any)
	if !ok {
		return nil
	}
	out := make([]int, len(arr))
	for i, v := range arr {
		switch n := v.(type) {
		case int8:
			out[i] = int(n)
		case int16:
			out[i] = int(n)
		case int32:
			out[i] = int(n)
		case int64:
			out[i] = int(n)
		case uint8:
			out[i] = int(n)
		case uint16:
			out[i] = int(n)
		case uint32:
			out[i] = int(n)
		case uint64:
			out[i] = int(n)
		}
	}
	return out
}

func aikitFloatArray(g *embed.GGUFFile, key string) []float32 {
	arr, ok := g.Metadata[key].([]any)
	if !ok {
		return nil
	}
	out := make([]float32, len(arr))
	for i, v := range arr {
		switch n := v.(type) {
		case float32:
			out[i] = n
		case float64:
			out[i] = float32(n)
		}
	}
	return out
}
