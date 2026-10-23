//go:build goinfer_testhooks

package decoder

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// moveRefsToSource re-keys every prompt's cached reference under another source, as a source change leaves them: the entries
// under the current source are removed, and for each prompt write(cp) gives the bytes stored under src. It returns the new paths.
func moveRefsToSource(t *testing.T, ck string, prompts [][]int, K, contN int, src string, write func(cp string) []byte) []string {
	t.Helper()
	var paths []string
	for pi := range prompts {
		parts, _ := PrefillRefKeyForTest(ck, "", prompts[pi][:K], contN)
		cp, ok := LookupPrefillRefForTest(parts)
		if !ok {
			t.Fatalf("prompt %d not stored", pi)
		}
		old := parts
		old.SourceSHA256 = src
		dst, err := prefillRefCachePath(old.Key())
		if err != nil {
			t.Fatal(err)
		}
		if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(dst, write(cp), 0o644); err != nil {
			t.Fatal(err)
		}
		b, _ := json.MarshalIndent(old, "", " ")
		if err := writeFileAtomic(dst+".key.json", b); err != nil {
			t.Fatal(err)
		}
		if err := os.Remove(cp); err != nil {
			t.Fatal(err)
		}
		if err := os.Remove(cp + ".key.json"); err != nil {
			t.Fatal(err)
		}
		paths = append(paths, dst)
	}
	return paths
}

// The generator, after a source change: one prompt is computed as the witness, and when it is byte-identical to the other
// source's the rest are adopted, not computed; when it differs every prompt is computed and the pair is never adopted.
func TestPrefillReference_adoptsAfterAWitness(t *testing.T) {
	t.Setenv("GOINFER_PREFILL_REF_CACHE", t.TempDir())
	t.Setenv("GOINFER_CPU_FAST_ATTENTION", "0")
	m, _ := tinyLlamaModel(t)
	ck := fakeCheckpoint(t, "tiny-llama-weights")
	const K, contN = 6, 3
	prompts := [][]int{{1, 2, 3, 4, 5, 6, 7}, {2, 3, 4, 5, 6, 7, 8}, {3, 4, 5, 6, 7, 8, 9}, {4, 5, 6, 7, 8, 9, 10}}
	out := filepath.Join(t.TempDir(), "prefill-ref")
	if err := os.MkdirAll(out, 0o755); err != nil {
		t.Fatal(err)
	}
	runPrefillReferenceKConcurrent(t, m, "tiny", ck, "", K, prompts, out, contN, 2)
	want := map[int][]byte{}
	for pi := range prompts {
		parts, _ := PrefillRefKeyForTest(ck, "", prompts[pi][:K], contN)
		cp, _ := LookupPrefillRefForTest(parts)
		want[pi], _ = os.ReadFile(cp)
	}
	// computedAfresh counts the prompts whose current entry is not the moved file itself (an adoption is a hard link of it).
	computedAfresh := func(moved []string) int {
		n := 0
		for pi := range prompts {
			parts, _ := PrefillRefKeyForTest(ck, "", prompts[pi][:K], contN)
			cp, ok := LookupPrefillRefForTest(parts)
			if !ok {
				t.Fatalf("prompt %d missing after the rerun", pi)
			}
			a, _ := os.Stat(cp)
			b, _ := os.Stat(moved[pi])
			if !os.SameFile(a, b) {
				n++
			}
			if got, _ := os.ReadFile(cp); !bytes.Equal(got, want[pi]) {
				t.Fatalf("prompt %d: the stored reference is not the one this source computes", pi)
			}
		}
		return n
	}

	same := moveRefsToSource(t, ck, prompts, K, contN, "agreeing-source", func(cp string) []byte {
		b, _ := os.ReadFile(cp)
		return b
	})
	runPrefillReferenceKConcurrent(t, m, "tiny", ck, "", K, prompts, out, contN, 2)
	if n := computedAfresh(same); n != 1 {
		t.Fatalf("after an agreeing witness %d prompts were computed, want 1 (the witness)", n)
	}
	parts, _ := PrefillRefKeyForTest(ck, "", prompts[0][:K], contN)
	if agree, known := PrefillRefSourcesAgreeForTest(parts, "agreeing-source"); !agree || !known {
		t.Fatalf("the witness was not recorded as agreeing (agree=%v known=%v)", agree, known)
	}

	for _, p := range same { // a cache that holds only the differing source's copies
		if err := os.Remove(p); err != nil {
			t.Fatal(err)
		}
		if err := os.Remove(p + ".key.json"); err != nil {
			t.Fatal(err)
		}
	}
	differ := moveRefsToSource(t, ck, prompts, K, contN, "differing-source", func(cp string) []byte {
		b, _ := os.ReadFile(cp)
		b = append([]byte(nil), b...)
		b[len(b)-1] ^= 1
		return b
	})
	runPrefillReferenceKConcurrent(t, m, "tiny", ck, "", K, prompts, out, contN, 2)
	if n := computedAfresh(differ); n != len(prompts) {
		t.Fatalf("after a differing witness %d prompts were computed, want all %d", n, len(prompts))
	}
	if agree, known := PrefillRefSourcesAgreeForTest(parts, "differing-source"); agree || !known {
		t.Fatalf("the differing witness was not recorded (agree=%v known=%v)", agree, known)
	}
}

// The backfill records a witness for every cross-source pair the cache already holds, once, and a pair with one differing
// witness never agrees.
func TestPrefillRefWitness_backfill(t *testing.T) {
	t.Setenv("GOINFER_PREFILL_REF_CACHE", t.TempDir())
	base := PrefillRefKeyParts{Version: 1, Checkpoint: "ck", PromptTokens: 4, Arch: "amd64", ContinuationN: 2}
	store := func(src, prompt string, last float32) {
		p := base
		p.SourceSHA256, p.PromptSHA256 = src, prompt
		if _, err := StorePrefillRefForTest(p, []float32{1, 2}, []int{3}, [][]float32{{4, last}}); err != nil {
			t.Fatal(err)
		}
	}
	store("a", "p0", 5)
	store("b", "p0", 5)
	store("a", "p1", 5)
	store("b", "p1", 6) // b agrees with a on p0 and not on p1
	store("c", "p0", 5)
	other := base
	other.ContinuationN = 3 // another scope: never a sibling
	other.SourceSHA256, other.PromptSHA256 = "d", "p0"
	if _, err := StorePrefillRefForTest(other, []float32{1, 2}, []int{3}, [][]float32{{4, 5}}); err != nil {
		t.Fatal(err)
	}
	n, differ, err := BackfillPrefillRefWitnessesForTest()
	if err != nil || n != 4 || differ != 1 {
		t.Fatalf("backfill recorded %d (%d differ), err %v; want 4 (a-b, b-c, a-c on p0; a-b on p1), 1 differing", n, differ, err)
	}
	if n, _, _ := BackfillPrefillRefWitnessesForTest(); n != 0 {
		t.Fatalf("a second backfill recorded %d, want 0", n)
	}
	q := base
	q.SourceSHA256, q.PromptSHA256 = "a", "p0"
	if agree, _ := PrefillRefSourcesAgreeForTest(q, "b"); agree {
		t.Fatal("a and b differ on p1, so they must not agree")
	}
	if agree, _ := PrefillRefSourcesAgreeForTest(q, "c"); !agree {
		t.Fatal("a and c agree on p0 and nothing differs")
	}
	q.SourceSHA256 = "e"
	if _, src, ok := PrefillRefSiblingForTest(q); !ok || src == "d" {
		t.Fatalf("sibling lookup returned source %q, ok %v: want one of a, b, c and never another scope's", src, ok)
	}
}
