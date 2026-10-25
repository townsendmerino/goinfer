//go:build goinfer_testhooks

package decoder

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
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
	if _, src, ok := PrefillRefSiblingForTest(q); !ok || src != "a" {
		t.Fatalf("sibling lookup returned source %q, ok %v: want a (most references in the scope, first by name), never another scope's d", src, ok)
	}
}

// An adopted reference is never a witness: not a sibling, not a backfill input, and a record it produced is pruned. Source b
// computes p0 and adopts p1 from a (with its marker); source d adopts p1 from a as adoptions did before the marker (a bare hard
// link, its sidecar the newer). A third source must then see only natively computed entries, so no chain a → b → c forms.
func TestPrefillRefWitness_adoptedIsNeverAWitness(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("GOINFER_PREFILL_REF_CACHE", dir)
	base := PrefillRefKeyParts{Version: 1, Checkpoint: "ck", PromptTokens: 4, Arch: "amd64", ContinuationN: 2}
	at := func(src, prompt string) PrefillRefKeyParts {
		p := base
		p.SourceSHA256, p.PromptSHA256 = src, prompt
		return p
	}
	store := func(p PrefillRefKeyParts) string {
		cp, err := StorePrefillRefForTest(p, []float32{1, 2}, []int{3}, [][]float32{{4, 5}})
		if err != nil {
			t.Fatal(err)
		}
		return cp
	}
	store(at("a", "p0"))
	a1 := store(at("a", "p1"))
	store(at("b", "p0"))
	if _, err := AdoptPrefillRefForTest(a1, at("b", "p1"), "a"); err != nil {
		t.Fatal(err)
	}
	time.Sleep(20 * time.Millisecond) // d's sidecar must be the newer
	d1, err := prefillRefCachePath(at("d", "p1").Key())
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(d1), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Link(a1, d1); err != nil {
		t.Skipf("no hard links here: %v", err)
	}
	b, _ := json.MarshalIndent(at("d", "p1"), "", " ")
	if err := writeFileAtomic(d1+".key.json", b); err != nil {
		t.Fatal(err)
	}
	// A stale record built on b's adopted p1, as a backfill before the marker could have written.
	if err := recordRefWitness(at("b", "p1"), "c", true); err != nil {
		t.Fatal(err)
	}

	for bin, e := range refEntries() {
		want := (e.parts.SourceSHA256 == "b" || e.parts.SourceSHA256 == "d") && e.parts.PromptSHA256 == "p1"
		if e.adopted != want {
			t.Errorf("%s (source %s, %s): adopted=%v, want %v", filepath.Base(bin), e.parts.SourceSHA256, e.parts.PromptSHA256, e.adopted, want)
		}
	}
	if _, src, ok := PrefillRefSiblingForTest(at("c", "p1")); !ok || src != "a" {
		t.Fatalf("c's sibling for p1 is source %q (ok %v): only a computed p1", src, ok)
	}
	n, differ, err := BackfillPrefillRefWitnessesForTest()
	if err != nil || n != 1 || differ != 0 {
		t.Fatalf("backfill recorded %d (%d differ), err %v: want 1, a-b on p0, the only pair both sides computed", n, differ, err)
	}
	for _, w := range readRefWitnesses() {
		if w.Prompt == "p1" {
			t.Fatalf("a witness on p1 survived (%s-%s): every p1 but a's is adopted", w.A, w.B)
		}
	}
}
