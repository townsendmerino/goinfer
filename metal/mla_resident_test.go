//go:build darwin && goinfer_testhooks

package metal

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestMLAResident_tinyAgainstControl is OP2a's gate M as first registered, kept as a diagnostic that logs and asserts
// nothing (docs/tasks/task-forward-architecture-2026-10.md §4.7, gate M's amendments): unforced, kimi-tiny dips at a routing
// near-tie, so gate M is graded with routing forced (TestMLAResident_routingForced); and the tiny fixtures cannot see the
// rope/nope plant, which the kernel gate TestMLAKernels_rope carries.
func TestMLAResident_tinyAgainstControl(t *testing.T) {
	const steps, quant = 32, "int4"
	ctrl := residentLockstep(t, "qwen3moe-tiny", quant, steps)
	t.Logf("control qwen3moe-tiny: min cosine %.6f, argmax %d/%d (diagnostic)", ctrl.minCos, ctrl.exact, ctrl.steps)
	for _, fx := range []string{"deepseek-tiny", "kimi-tiny"} {
		s := residentLockstep(t, fx, quant, steps)
		t.Logf("%s, routing unforced: min cosine %.6f, argmax %d/%d (diagnostic)", fx, s.minCos, s.exact, s.steps)
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

// TestMLAResident_tier1V2LiteSlice is OP2a's tier 1 (docs/tasks/task-forward-architecture-2026-10.md §4.7): the first two
// layers of DeepSeek-V2-Lite-Chat's Q4_K_M GGUF (layer 0 dense, layer 1 MoE; scripts/slice_gguf_layers.py), resident on
// Metal against the CPU at int4, by G2-R's bars with the dense 1.5B control (residentTier1).
//
//	GOINFER_HEAVY_TESTS=1 go test -tags goinfer_testhooks -count=1 -timeout 10m -run '^TestMLAResident_tier1V2LiteSlice$' -v ./metal/
func TestMLAResident_tier1V2LiteSlice(t *testing.T) {
	home, _ := os.UserHomeDir()
	residentTier1(t, filepath.Join(home, "models", "deepseek-v2-lite-gguf", "DeepSeek-V2-Lite-Chat-Q4_K_M.slice-0-1.gguf"), "DeepSeek-V2-Lite slice 0,1")
}
