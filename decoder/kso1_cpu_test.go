//go:build darwin && arm64 && goinfer_testhooks

package decoder

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// kso1CPUFingerprint is O1-G4's digest taken before KS-O1 (docs/tasks/task-kernel-survey-2026-10.md, "Review fixes for #215").
// It is arm64's: the CPU reference is bit-identical within an arch, not across.
const kso1CPUFingerprint = "e662bdd6c6cb108e89cda98e7aa3b9554ff17df556668dc05138d1477da0bfbe"

// TestKSO1_cpuQ4KFingerprint is O1-G4 (docs/tasks/task-kernel-survey-2026-10.md, "KS-O1 on Metal: gates"): the CPU at --quant q4k on
// Qwen2.5-Coder-0.5B q4_k_m (its Q6_K layers are int8 copies there) forwards a fixed prompt and 16 greedy tokens one position at a
// time, and every position's logits bytes are hashed. The digest must equal kso1CPUFingerprint, taken before KS-O1.
//
//	GOINFER_HEAVY_TESTS=1 go test -count=1 -tags goinfer_testhooks -run '^TestKSO1_cpuQ4KFingerprint$' -v ./decoder/
func TestKSO1_cpuQ4KFingerprint(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1")
	}
	home, _ := os.UserHomeDir()
	path := filepath.Join(home, "models", "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no GGUF at %s", path)
	}
	m, err := Load(path, Options{Backend: "cpu", Quant: "q4k"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	seq := []int{785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562, 13, 151643, 3555}
	prompt := len(seq)
	const gen = 16
	cache := m.NewCache(prompt + gen)
	h := sha256.New()
	for i := 0; i < prompt+gen-1; i++ {
		lg, err := m.ForwardForTest(seq[i], cache)
		if err != nil {
			t.Fatal(err)
		}
		best := 0
		for j, v := range lg {
			_ = binary.Write(h, binary.LittleEndian, math.Float32bits(v))
			if v > lg[best] {
				best = j
			}
		}
		if i >= prompt-1 {
			seq = append(seq, best)
		}
	}
	if got := len(seq) - prompt; got != gen {
		t.Fatalf("%d greedy tokens, want %d", got, gen)
	}
	sum := fmt.Sprintf("%x", h.Sum(nil))
	fmt.Fprintf(os.Stderr, "[O1-G4] CPU q4k 0.5B: tokens %v, logits fingerprint %s\n", seq[prompt:], sum)
	if sum != kso1CPUFingerprint {
		t.Errorf("O1-G4: fingerprint %s, want %s (before KS-O1)", sum, kso1CPUFingerprint)
	}
}
