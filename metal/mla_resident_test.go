//go:build darwin && goinfer_testhooks

package metal

import (
	"errors"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestMLAResident_tinyAgainstControl is OP2a's gate M on Metal (docs/tasks/task-forward-architecture-2026-10.md §4.7, "OP2a on
// Metal", registered before any code): deepseek-tiny and kimi-tiny resident on Metal against the CPU at int4, 32 steps of
// greedy lockstep graded from position 2, against qwen3moe-tiny, a MoE already resident on Metal, through the same harness:
// each MLA fixture's min logit cosine at least the control's minus 0.001 and its argmax agreement at least the control's minus
// 2. With q's and the latent's rope and nope slices swapped in the Metal kernels only, the same comparison must fail.
func TestMLAResident_tinyAgainstControl(t *testing.T) {
	const steps, quant = 32, "int4"
	ctrl := residentLockstep(t, "qwen3moe-tiny", quant, steps)
	pass := func(s lockstep) bool { return s.minCos >= ctrl.minCos-0.001 && s.exact >= ctrl.exact-2 }
	t.Logf("control qwen3moe-tiny: min cosine %.6f, argmax %d/%d", ctrl.minCos, ctrl.exact, ctrl.steps)
	for _, fx := range []string{"deepseek-tiny", "kimi-tiny"} {
		s := residentLockstep(t, fx, quant, steps)
		t.Logf("%s: min cosine %.6f, argmax %d/%d -> %s", fx, s.minCos, s.exact, s.steps, map[bool]string{true: "pass", false: "FAIL"}[pass(s)])
		if !pass(s) {
			t.Errorf("%s misses gate M (min cosine >= %.6f, argmax >= %d)", fx, ctrl.minCos-0.001, ctrl.exact-2)
		}
	}
	plant := [][2]string{
		{"uint qbase = h * qkHead + qkNope;", "uint qbase = h * qkHead;"},
		{"if (interleave == 1u) { a = kvDown[rank + 2u*i]; b = kvDown[rank + 2u*i + 1u]; }", "if (interleave == 1u) { a = kvDown[2u*i]; b = kvDown[2u*i + 1u]; }"},
	}
	for _, p := range plant {
		if strings.Count(allKernels, p[0]) != 1 {
			t.Fatalf("the plant's target %q is not in the MLA kernels once", p[0])
		}
	}
	mlaSrcPlantForTest = plant
	defer func() { mlaSrcPlantForTest = nil }()
	s := residentLockstep(t, "deepseek-tiny", quant, steps)
	t.Logf("plant (rope and nope slices swapped, Metal only): min cosine %.6f, argmax %d/%d", s.minCos, s.exact, s.steps)
	if pass(s) {
		t.Error("with the rope and nope slices swapped the gate still passes: it cannot see the defect")
	}
}

// TestMLAResident_guards pins the paths that decline an MLA model, each with its reason: the f16 prefill, the MC3 batched
// step, KV2's slot copy (whose plant, the decline removed, must let the copy through), and int8 KV.
func TestMLAResident_guards(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	dir := filepath.Join("..", "testdata", "deepseek-tiny")
	m, err := decoder.Load(dir, decoder.Options{Quant: "int4", ResidentKVSlots: 2})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer r.Close()
	if r.mla == nil || len(r.kvSlotBufs) < 2 {
		t.Fatalf("mla %v, %d KV slots: not the resident under test", r.mla != nil, len(r.kvSlotBufs))
	}
	if r.prefillOK {
		t.Error("the f16 prefill admits an MLA model")
	}
	if why := r.batchIneligible(); why == "" {
		t.Error("the MC3 batched step admits an MLA model")
	}
	a := &metalResident{r: r, hidden: r.H}
	var dec *decoder.KVCopyDecline
	if err := a.CopyKV(0, 1, 0, 64); !errors.As(err, &dec) || !strings.Contains(err.Error(), "MLA") {
		t.Errorf("CopyKV on an MLA model = %v, want the MLA decline", err)
	}
	mlaKVCopyDeclineOffForTest = true
	err = a.CopyKV(0, 1, 0, 64)
	mlaKVCopyDeclineOffForTest = false
	if errors.As(err, &dec) && strings.Contains(err.Error(), "MLA") {
		t.Error("with the decline removed CopyKV still gives the MLA decline: the test cannot see its defect")
	}
	mi, err := decoder.Load(dir, decoder.Options{Quant: "int4", KVPrecision: "i8"})
	if err != nil {
		t.Fatal(err)
	}
	defer mi.Close()
	if _, err := buildResident(mi); err == nil || !strings.Contains(err.Error(), "int8 KV") {
		t.Errorf("an MLA model with int8 KV: buildResident = %v, want the int8-KV decline", err)
	}
}
