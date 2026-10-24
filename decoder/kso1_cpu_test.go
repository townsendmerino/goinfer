package decoder

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// TestKSO1_cpuQ4KFingerprint is O1-G4's instrument (docs/tasks/task-kernel-survey-2026-10.md, "KS-O1 on Metal: gates"): the CPU
// at --quant q4k on Qwen2.5-Coder-0.5B q4_k_m (its Q6_K layers are int8 copies there), 16 greedy tokens after a fixed prompt and
// the last hidden state's bytes, printed as one fingerprint. O1-G4 compares the fingerprint printed at the commit before KS-O1
// and at KS-O1's; this test only computes it. Skips with no file.
func TestKSO1_cpuQ4KFingerprint(t *testing.T) {
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
	prompt := []int{785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562, 13, 151643, 3555}
	ch, _ := m.Generate(context.Background(), prompt, 16, SamplingParams{})
	h := sha256.New()
	var toks []int
	for tok := range ch {
		toks = append(toks, tok)
		_ = binary.Write(h, binary.LittleEndian, int32(tok))
	}
	hid, err := m.HiddenLast(prompt)
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range hid {
		_ = binary.Write(h, binary.LittleEndian, math.Float32bits(v))
	}
	fmt.Fprintf(os.Stderr, "[O1-G4] CPU q4k 0.5B: tokens %v, fingerprint %x\n", toks, h.Sum(nil))
}
