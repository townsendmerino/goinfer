//go:build darwin && goinfer_testhooks

package metal

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"testing"
)

// TestALS_G3_towerHashes is MEM4(a)'s ALS-G3 (docs/completed/task-memory-campaign-2026-10.md, "MEM4(a) gates"): every tower of
// mem2Towers over its committed tiny fixture and growing inputs, the SHA-256 of each output's bits, built from aikit with and
// without the aliased tower weights (two workspaces); every line must match. The comparison across workspaces is a diff of the
// printed [ALS-G3] lines; what this test asserts itself is that each output is non-empty and finite and that a second forward of
// the same input hashes identically in this process.
func TestALS_G3_towerHashes(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	towers := mem2Towers(t)
	if len(towers) == 0 {
		t.Skip("mem2Towers built no tower: nothing was hashed")
	}
	hash := func(out []float32) string {
		h := sha256.New()
		for _, v := range out {
			_ = binary.Write(h, binary.LittleEndian, math.Float32bits(v))
		}
		return fmt.Sprintf("%x", h.Sum(nil))
	}
	for _, tw := range towers {
		inst := tw.build(t)
		for i := range tw.n {
			out, err := inst.fwd(i)
			if err != nil {
				t.Fatalf("%s input %d: %v", tw.name, i, err)
			}
			if len(out) == 0 {
				t.Fatalf("%s input %d: empty output", tw.name, i)
			}
			for k, v := range out {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Fatalf("%s input %d: output[%d] = %v", tw.name, i, k, v)
				}
			}
			first := hash(out)
			again, err := inst.fwd(i)
			if err != nil {
				t.Fatalf("%s input %d (second forward): %v", tw.name, i, err)
			}
			if second := hash(again); second != first {
				t.Fatalf("%s input %d: a second forward of the same input hashed %s, the first %s", tw.name, i, second, first)
			}
			fmt.Fprintf(os.Stderr, "[ALS-G3] %s input %d: %d values, sha256 %s\n", tw.name, i, len(out), first)
		}
		if err := inst.close(); err != nil {
			t.Fatal(err)
		}
	}
}
