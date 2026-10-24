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

// TestALS_G3_towerHashes is MEM4(a)'s ALS-G3 (docs/tasks/task-memory-campaign-2026-10.md, "MEM4(a) gates"): every tower of
// mem2Towers over its committed tiny fixture and growing inputs, the SHA-256 of each output's bits, built from aikit with and
// without the aliased tower weights (two workspaces); every line must match.
func TestALS_G3_towerHashes(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	for _, tw := range mem2Towers(t) {
		inst := tw.build(t)
		for i := range tw.n {
			out, err := inst.fwd(i)
			if err != nil {
				t.Fatalf("%s input %d: %v", tw.name, i, err)
			}
			h := sha256.New()
			for _, v := range out {
				_ = binary.Write(h, binary.LittleEndian, math.Float32bits(v))
			}
			fmt.Fprintf(os.Stderr, "[ALS-G3] %s input %d: %d values, sha256 %x\n", tw.name, i, len(out), h.Sum(nil))
		}
		if err := inst.close(); err != nil {
			t.Fatal(err)
		}
	}
}
