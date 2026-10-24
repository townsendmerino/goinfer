//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"math"
	"os"
	"os/exec"
	"slices"
	"strconv"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestMoELayerMajor_matchesSequential is prefillMoEPaged's gate, the generic twin of TestG4LayerMajor_matchesSequential:
// on Mixtral, two Qwen3-MoE shapes and the Qwen3.5 DeltaNet MoE hybrid, paged at 2 and 3 slots, prompts of 1, 5 and 9
// rows from position 0 and 6 rows continuing a 3-row prefix, through the layer-major path and through the sequential
// loop. The last row's logits and 3 decode steps after it (which read every layer's K/V and DeltaNet state) must be
// equal bit for bit, and every attention layer's K/V. Each arm starts from Reset (a DeltaNet's state zeroed). The layer-major
// arm runs KS-D9's batched phase 2, which every fixture here admits and every prompt shape must take (D9-G1), and a third arm,
// layer-major with the per-row phase 2 (moeExpertBatchOn off), must equal it bit for bit too. A resident the backend declines
// fails the test; only a slot count under the model's k (not pageable) skips. Under GOINFER_KSD9_PLANT_CHILD=1 (set by
// TestKSD9_plantSeenByTheGate is KS-D9's planted-defect check: with a defect planted in the batched phase 2 by hand, the startup
// self-test declines the backend first, so it re-runs TestMoELayerMajor_matchesSequential in a child process with the self-test
// skipped (GOINFER_KSD9_PLANT_CHILD=1, which skips it for that process only) and passes iff the child fails. Armed by
// GOINFER_KSD9_PLANT=1 only; unarmed, it skips.
func TestKSD9_plantSeenByTheGate(t *testing.T) {
	if os.Getenv("GOINFER_KSD9_PLANT") != "1" || os.Getenv("GOINFER_KSD9_PLANT_CHILD") == "1" {
		t.Skip("set GOINFER_KSD9_PLANT=1 with a defect planted in the batched phase 2")
	}
	cmd := exec.Command(os.Args[0], "-test.run", "^TestMoELayerMajor_matchesSequential$", "-test.count=1")
	cmd.Env = append(os.Environ(), "GOINFER_KSD9_PLANT_CHILD=1")
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Fatalf("the gate passed with the plant in place:\n%s", out)
	}
	t.Logf("the gate failed with the plant in place, as it must:\n%s", out)
}

// TestKSD9_scratchWithinPlan: the batched phase 2's scratch is allocated once per layer-major call, sized for its rows, and
// priced in the plan. On paged Mixtral (2 slots), one 9-row prefillMoEPaged counts exactly moeBatchBytes(9) in
// prefillScratchSeen, and prefillScratchBytes grows by moeBatchBytes of min(M, g4LayerMajorChunk) when the batched phase 2 is on.
func TestKSD9_scratchWithinPlan(t *testing.T) {
	const fx = "../testdata/mixtral-tiny"
	if _, err := os.Stat(fx + "/model.safetensors"); err != nil {
		t.Skipf("no fixture: %v", err)
	}
	t.Setenv("GOINFER_METAL_MOE_SLOTS", "2")
	m, err := decoder.Load(fx, decoder.Options{Backend: "metal", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	mr, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok || mr.r.moe == nil || !mr.r.moe.paged || !mr.r.moeExpertBatchOK() {
		t.Fatalf("not a paged generic MoE the batched phase 2 admits: %s", m.ResidentDecline())
	}
	r := mr.r
	embs := make([][]float32, 9)
	for i := range embs {
		embs[i] = m.EmbedResidentForTest(i*31 + 7)
	}
	before, groups := prefillScratchSeen.Load(), r.moeBatchGroups
	r.prefillMoEPaged(embs, 0, true)
	if err := r.takeExecErr(); err != nil {
		t.Fatal(err)
	}
	if r.moeBatchGroups == groups {
		t.Fatal("the batched phase 2 did not run")
	}
	if seen, want := prefillScratchSeen.Load()-before, r.moeBatchBytes(9); seen != want {
		t.Errorf("9 rows: the batch scratch counted %d bytes, want moeBatchBytes(9) = %d (one allocation per call)", seen, want)
	}
	for _, M := range []int{9, 300} {
		on := r.prefillScratchBytes(M)
		moeExpertBatchOn = false
		off := r.prefillScratchBytes(M)
		moeExpertBatchOn = true
		if want := r.moeBatchBytes(min(M, g4LayerMajorChunk)); on-off != want {
			t.Errorf("M=%d: the plan adds %d bytes for the batched phase 2, want %d", M, on-off, want)
		}
	}
}
