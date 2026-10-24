//go:build goinfer_testhooks

package decoder

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"testing"
)

// TestKSD8_cpuReference writes D8-G3's reference (docs/tasks/task-kernel-survey-2026-10.md, "D8-G3's reference route"): the
// Hugging Face checkpoint at GOINFER_KSD8_REF_MODEL on the CPU with its bf16 weights kept bf16 and exact attention, each prompt
// of GOINFER_KSD8_REF_IDS (a JSON list of token-id lists) prefilled from an empty cache, its last-position logits appended to
// GOINFER_KSD8_REF_OUT as little-endian float32.
func TestKSD8_cpuReference(t *testing.T) {
	model, idsPath, out := os.Getenv("GOINFER_KSD8_REF_MODEL"), os.Getenv("GOINFER_KSD8_REF_IDS"), os.Getenv("GOINFER_KSD8_REF_OUT")
	if model == "" || idsPath == "" || out == "" {
		t.Skip("set GOINFER_KSD8_REF_MODEL, GOINFER_KSD8_REF_IDS and GOINFER_KSD8_REF_OUT")
	}
	t.Setenv("GOINFER_CPU_FAST_ATTENTION", "0")
	b, err := os.ReadFile(idsPath)
	if err != nil {
		t.Fatal(err)
	}
	var prompts [][]int
	if err := json.Unmarshal(b, &prompts); err != nil {
		t.Fatal(err)
	}
	m, err := Load(model, Options{Backend: "cpu", Quant: "bf16", ResidentContext: 1024})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	f, err := os.Create(out)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	for k, ids := range prompts {
		lg, err := m.PrefillLogitsForTest(context.Background(), ids, m.NewCache(len(ids)))
		if err != nil {
			t.Fatalf("prompt %d: %v", k, err)
		}
		if err := binary.Write(f, binary.LittleEndian, lg); err != nil {
			t.Fatal(err)
		}
		top := 0
		for i, v := range lg {
			if v > lg[top] {
				top = i
			}
		}
		fmt.Fprintf(os.Stderr, "[D8-G3 ref] prompt %d: %d ids, %d logits, top-1 %d\n", k, len(ids), len(lg), top)
	}
}
