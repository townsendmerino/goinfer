//go:build goinfer_testhooks

package decoder

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/internal/fidelity"
)

func fakeCheckpoint(t *testing.T, content string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

// The key must move with every input a reference depends on, and with nothing else.
func TestPrefillRefKey_sensitivity(t *testing.T) {
	t.Setenv("GOINFER_PREFILL_REF_CACHE", t.TempDir())
	t.Setenv("GOINFER_CPU_FAST_ATTENTION", "0")
	ck := fakeCheckpoint(t, "weights-A")
	ids := []int{1, 2, 3, 4}
	base, err := PrefillRefKeyForTest(ck, "", ids, 64)
	if err != nil {
		t.Fatal(err)
	}
	again, _ := PrefillRefKeyForTest(ck, "", ids, 64)
	if base.Key() != again.Key() {
		t.Fatal("the key must be deterministic")
	}
	moved, _ := PrefillRefKeyForTest(fakeCheckpoint(t, "weights-A"), "", ids, 64) // same bytes, another path
	if moved.Key() != base.Key() {
		t.Fatal("the checkpoint's path must not be part of the key, only its content")
	}
	for name, other := range map[string]func() (PrefillRefKeyParts, error){
		"checkpoint bytes": func() (PrefillRefKeyParts, error) {
			return PrefillRefKeyForTest(fakeCheckpoint(t, "weights-B"), "", ids, 64)
		},
		"prompt ids":    func() (PrefillRefKeyParts, error) { return PrefillRefKeyForTest(ck, "", []int{1, 2, 3, 5}, 64) },
		"prompt length": func() (PrefillRefKeyParts, error) { return PrefillRefKeyForTest(ck, "", []int{1, 2, 3}, 64) },
		"quant":         func() (PrefillRefKeyParts, error) { return PrefillRefKeyForTest(ck, "int8", ids, 64) },
		"continuation":  func() (PrefillRefKeyParts, error) { return PrefillRefKeyForTest(ck, "", ids, 32) },
	} {
		p, err := other()
		if err != nil {
			t.Fatal(err)
		}
		if p.Key() == base.Key() {
			t.Fatalf("changing the %s did not change the key", name)
		}
	}
	t.Setenv("GOINFER_CPU_FAST_ATTENTION", "1")
	if fast, _ := PrefillRefKeyForTest(ck, "", ids, 64); fast.Key() == base.Key() {
		t.Fatal("the fast (non-reference) attention must not share a key with the exact one")
	}
	if base.SourceSHA256 == "" || base.Arch == "" {
		t.Fatal("source hash and arch must be filled")
	}
	// The sidecar's prompt hash is internal/fidelity's PromptSetHash of the one prompt: one encoding across packages.
	if base.PromptSHA256 != fidelity.PromptSetHash([][]int{ids}) {
		t.Fatal("PromptIDsSHA256 and fidelity.PromptSetHash disagree")
	}
}

func TestPrefillRefCache_storeLookupLinkIdentity(t *testing.T) {
	t.Setenv("GOINFER_PREFILL_REF_CACHE", t.TempDir())
	ck := fakeCheckpoint(t, "w")
	ids := []int{7, 8, 9}
	parts, err := PrefillRefKeyForTest(ck, "", ids, 2)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := LookupPrefillRefForTest(parts); ok {
		t.Fatal("an empty cache must miss")
	}
	seed := []float32{1, 2, 3}
	logits := [][]float32{{4, 5, 6}, {7, 8, 9}}
	cp, err := StorePrefillRefForTest(parts, seed, []int{1, 2}, logits)
	if err != nil {
		t.Fatal(err)
	}
	if got, ok := LookupPrefillRefForTest(parts); !ok || got != cp {
		t.Fatalf("a stored reference must be found (got %q, %v)", got, ok)
	}
	// An interrupted store (the .bin present, its parts record not yet written) is a miss, not a hit.
	if err := os.Remove(cp + ".key.json"); err != nil {
		t.Fatal(err)
	}
	if _, ok := LookupPrefillRefForTest(parts); ok {
		t.Fatal("a .bin without its parts record must not count as cached")
	}
	if cp, err = StorePrefillRefForTest(parts, seed, []int{1, 2}, logits); err != nil {
		t.Fatal(err)
	}
	legacy := filepath.Join(t.TempDir(), "prefill-ref", "S-K3-p0.bin")
	if err := LinkPrefillRefForTest(cp, legacy, parts); err != nil {
		t.Fatal(err)
	}
	s2, toks, l2, err := ReadPrefillReferenceForTest(legacy)
	if err != nil || len(s2) != 3 || len(toks) != 2 || l2[1][2] != 9 {
		t.Fatalf("the linked reference must read back intact: %v %v %v %v", s2, toks, l2, err)
	}
	if st, why := PrefillRefIdentityForTest(legacy, ids); st != RefIdentityVerified {
		t.Fatalf("same ids must verify: %v %s", st, why)
	}
	if st, _ := PrefillRefIdentityForTest(legacy, []int{7, 8, 10}); st != RefIdentityMismatch {
		t.Fatal("different ids must be a mismatch")
	}
	bare := filepath.Join(t.TempDir(), "old.bin")
	if err := os.WriteFile(bare, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if st, _ := PrefillRefIdentityForTest(bare, ids); st != RefIdentityNoSidecar {
		t.Fatal("a file without a sidecar must report NoSidecar so the consumer falls back to its KL check")
	}
}

// TE8(a): the generator computes a cell once. A second run finds every prompt cached, rewrites nothing, and still
// populates the consumer's path; a changed prompt is computed afresh.
func TestPrefillReference_resumesFromCache(t *testing.T) {
	t.Setenv("GOINFER_PREFILL_REF_CACHE", t.TempDir())
	t.Setenv("GOINFER_CPU_FAST_ATTENTION", "0")
	m, _ := tinyLlamaModel(t)
	ck := fakeCheckpoint(t, "tiny-llama-weights")
	const K, contN = 6, 3
	prompts := [][]int{{1, 2, 3, 4, 5, 6, 7}, {2, 3, 4, 5, 6, 7, 8}, {3, 4, 5, 6, 7, 8, 9}}
	out := filepath.Join(t.TempDir(), "prefill-ref")
	if err := os.MkdirAll(out, 0o755); err != nil {
		t.Fatal(err)
	}

	runPrefillReferenceKConcurrent(t, m, "tiny", ck, "", K, prompts, out, contN, 2)
	mtimes := map[string]time.Time{}
	for pi := range prompts {
		parts, _ := PrefillRefKeyForTest(ck, "", prompts[pi][:K], contN)
		cp, ok := LookupPrefillRefForTest(parts)
		if !ok {
			t.Fatalf("prompt %d not stored after the first run", pi)
		}
		st, _ := os.Stat(cp)
		mtimes[cp] = st.ModTime()
		legacy := filepath.Join(out, "tiny-K6-p"+string(rune('0'+pi))+".bin")
		if id, why := PrefillRefIdentityForTest(legacy, prompts[pi][:K]); id != RefIdentityVerified {
			t.Fatalf("prompt %d: consumer path not verified: %s", pi, why)
		}
	}
	time.Sleep(20 * time.Millisecond)
	runPrefillReferenceKConcurrent(t, m, "tiny", ck, "", K, prompts, out, contN, 2)
	for cp, mt := range mtimes {
		st, _ := os.Stat(cp)
		if !st.ModTime().Equal(mt) {
			t.Fatalf("%s was rewritten on a rerun; a cached reference must be reused, not recomputed", cp)
		}
	}
	// A changed prompt misses and is computed; the others stay cached.
	prompts[1] = []int{9, 9, 9, 9, 9, 9, 9}
	runPrefillReferenceKConcurrent(t, m, "tiny", ck, "", K, prompts, out, contN, 2)
	parts, _ := PrefillRefKeyForTest(ck, "", prompts[1][:K], contN)
	if _, ok := LookupPrefillRefForTest(parts); !ok {
		t.Fatal("the changed prompt must be computed and stored")
	}
	if id, _ := PrefillRefIdentityForTest(filepath.Join(out, "tiny-K6-p1.bin"), prompts[1][:K]); id != RefIdentityVerified {
		t.Fatal("the consumer path must now hold the new prompt's reference")
	}
}
