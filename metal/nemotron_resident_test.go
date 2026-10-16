//go:build darwin && goinfer_testhooks

package metal

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime/debug"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// lockstep is one fixture's Metal-resident run against the CPU at the same quant: the worst logit cosine over positions
// from 2, and on how many steps the argmax agreed.
type lockstep struct {
	minCos       float64
	exact, steps int
}

// residentLockstep builds dir's resident on Metal (buildResident, so a feature not yet declared does not decline it) and
// runs it against the CPU model it was built from, in greedy lockstep off the CPU: a fixed seed, then the CPU's argmax.
func residentLockstep(t *testing.T, dir, quant string, steps int) lockstep {
	t.Helper()
	return residentLockstepAt(t, filepath.Join("..", "testdata", dir), decoder.Options{Quant: quant}, steps)
}

// residentLockstepAt is residentLockstep on the checkpoint at path, loaded with opts.
func residentLockstepAt(t *testing.T, path string, opts decoder.Options, steps int) lockstep {
	t.Helper()
	dir, quant := filepath.Base(path), opts.Quant
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	m, err := decoder.Load(path, opts)
	if err != nil {
		t.Fatalf("load %s: %v", dir, err)
	}
	defer m.Close()
	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("%s at %s: build resident: %v", dir, quant, err)
	}
	defer r.Close()
	a := &metalResident{r: r, hidden: r.H}
	_, _, _, _, _, _, vocab := m.Dims()
	seed := []int{1, 17 % vocab, 113 % vocab, 29 % vocab, 71 % vocab, 5 % vocab}
	cache := m.NewCache(0)
	st := lockstep{minCos: 1, steps: steps}
	tok := seed[0]
	for i := range steps {
		cpuL, err := m.ForwardForTest(tok, cache)
		if err != nil {
			t.Fatalf("cpu forward: %v", err)
		}
		gpuL, err := a.Forward(m.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatalf("metal forward: %v", err)
		}
		if i >= 2 {
			c, _ := mambaCos(cpuL, gpuL)
			st.minCos = min(st.minCos, c)
		}
		if argmaxF(cpuL) == argmaxF(gpuL) {
			st.exact++
		}
		if i+1 < len(seed) {
			tok = seed[i+1]
		} else {
			tok = argmaxF(cpuL)
		}
	}
	return st
}

// TestNemotronResident_tinyAgainstControl is OP2b's gate M on Metal (docs/tasks/task-forward-architecture-2026-10.md §4.7,
// "OP2b on Metal", graded at int4 by the amendment after the data): nemotron-tiny resident on Metal against the CPU at int4,
// 32 steps of greedy lockstep, graded
// from position 2. The bar is relative to llama-tiny through the same harness in the same run: nemotron-tiny's min logit
// cosine at least the control's minus 0.001, and argmax agreement on at least the control's count minus 2. With the conv
// window's taps shifted in the Metal build only (mambaSrcPlantForTest), the same comparison must fail.
func TestNemotronResident_tinyAgainstControl(t *testing.T) {
	const steps = 32
	for _, quant := range []string{"int4"} { // int4 only: the amendment after the data (§4.7); int8int8 declines, TestNemotronResident_int8Declines
		ctrl := residentLockstep(t, "llama-tiny", quant, steps)
		nemo := residentLockstep(t, "nemotron-tiny", quant, steps)
		pass := func(s lockstep) bool { return s.minCos >= ctrl.minCos-0.001 && s.exact >= ctrl.exact-2 }
		t.Logf("%s: control llama-tiny min cosine %.6f, argmax %d/%d; nemotron-tiny min cosine %.6f, argmax %d/%d -> %s",
			quant, ctrl.minCos, ctrl.exact, ctrl.steps, nemo.minCos, nemo.exact, nemo.steps, map[bool]string{true: "pass", false: "FAIL"}[pass(nemo)])
		if !pass(nemo) {
			t.Errorf("%s: nemotron-tiny on Metal is outside the control's band", quant)
		}
		mambaSrcPlantForTest = [2]string{"s = s + convW[c * K + j] * wv;", "s = s + convW[c * K + j + 1u] * wv;"}
		planted := residentLockstep(t, "nemotron-tiny", quant, steps)
		mambaSrcPlantForTest = [2]string{}
		t.Logf("%s: plant (conv taps shifted, Metal only): min cosine %.6f, argmax %d/%d -> %s", quant, planted.minCos, planted.exact,
			planted.steps, map[bool]string{true: "passes, which it must not", false: "fails, as it must"}[pass(planted)])
		if pass(planted) {
			t.Errorf("%s: the planted conv defect stayed inside the band: gate M cannot see a broken Mamba kernel", quant)
		}
	}

}

// metalGreedyReply feeds prompt to the resident from position 0 (which resets its recurrent state) and returns n greedy
// tokens after it.
func metalGreedyReply(t *testing.T, a *metalResident, m *decoder.Model, prompt []int, n int) []int {
	t.Helper()
	var out []int
	tok, pos := prompt[0], 0
	for i := 0; i < len(prompt)+n-1; i++ {
		l, err := a.Forward(m.EmbedResidentForTest(tok), pos)
		if err != nil {
			t.Fatalf("metal forward: %v", err)
		}
		pos++
		if i+1 < len(prompt) {
			tok = prompt[i+1]
			continue
		}
		tok = argmaxF(l)
		out = append(out, tok)
	}
	return out
}

// TestNemotronResident_slots is OP2b's gate S on Metal: a recurrent family gets one KV slot (so MC3 runs no concurrent
// generation on it and KV2 has no other slot to copy from) and the batched step declines it; and two conversations taking
// turns on that one slot, its state reset at each one's position 0, each give the reply they give alone. With the reset
// planted to leave the Mamba-2 state alone, the turns must differ.
func TestNemotronResident_slots(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	m, err := decoder.Load(filepath.Join("..", "testdata", "nemotron-tiny"), decoder.Options{Quant: "int4", ResidentKVSlots: 4})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if got := m.ResidentKVSlotsRequest(); got != 1 {
		t.Errorf("Nemotron-H asks for %d resident KV slots, want 1: a recurrent family's state is not part of a slot", got)
	}
	build := func() *metalResident {
		r, err := buildResident(m)
		if err != nil {
			t.Fatalf("build resident: %v", err)
		}
		t.Cleanup(func() { _ = r.Close() })
		return &metalResident{r: r, hidden: r.H}
	}
	a := build()
	if n := a.r.kvSlotCount(); n != 1 {
		t.Errorf("the resident built %d KV slots, want 1", n)
	}
	if why := a.r.batchIneligible(); why == "" {
		t.Error("the batched step admits Nemotron-H")
	}
	pA, pB := []int{1, 40, 9, 77, 3, 51}, []int{1, 12, 88, 6, 30}
	const n = 10
	aloneA := metalGreedyReply(t, build(), m, pA, n)
	aloneB := metalGreedyReply(t, build(), m, pB, n)
	turns := func(a *metalResident) (a1, b, a2 []int) {
		return metalGreedyReply(t, a, m, pA, n), metalGreedyReply(t, a, m, pB, n), metalGreedyReply(t, a, m, pA, n)
	}
	a1, b, a2 := turns(a)
	same := func(x, y []int) bool { return fmt.Sprint(x) == fmt.Sprint(y) }
	t.Logf("alone A %v, B %v; turns A %v, B %v, A %v", aloneA, aloneB, a1, b, a2)
	if !same(a1, aloneA) || !same(b, aloneB) || !same(a2, aloneA) {
		t.Error("a conversation taking turns on the slot does not give the reply it gives alone")
	}
	mambaResetSkipForTest = true
	p1, pb, p2 := turns(build())
	mambaResetSkipForTest = false
	t.Logf("plant (reset leaves the Mamba-2 state): turns A %v, B %v, A %v", p1, pb, p2)
	if same(pb, aloneB) && same(p2, aloneA) {
		t.Error("with the Mamba-2 state left between conversations the turns still match: gate S cannot see a missing reset")
	}
}

// TestNemotronResident_int8Declines: at int8int8 Metal declines Nemotron-H as resident, so the CPU runs it at the int8 it was
// loaded with, instead of re-quantizing its projections to int4 without notice (Metal's native int8 path covers no recurrent
// family). int4 builds. With the decline planted off, the int8int8 build must go through, which the gate must see.
func TestNemotronResident_int8Declines(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	build := func(quant string) error {
		m, err := decoder.Load(filepath.Join("..", "testdata", "nemotron-tiny"), decoder.Options{Quant: quant})
		if err != nil {
			t.Fatalf("load at %s: %v", quant, err)
		}
		defer m.Close()
		r, err := buildResident(m)
		if err == nil {
			_ = r.Close()
		}
		return err
	}
	if err := build("int8int8"); err == nil || !strings.Contains(err.Error(), "int4 only") {
		t.Errorf("int8int8: buildResident = %v, want the int4-only decline", err)
	}
	if err := build("int4"); err != nil {
		t.Errorf("int4: buildResident = %v, want a build", err)
	}
	// Through the backend as well: a Metal load at int8int8 runs on the CPU, not resident.
	m, err := decoder.Load(filepath.Join("..", "testdata", "nemotron-tiny"), decoder.Options{Backend: "metal", Quant: "int8int8"})
	if err != nil {
		t.Fatal(err)
	}
	if m.ResidentForwardForTest() != nil {
		t.Error("a Metal load of Nemotron-H at int8int8 went resident")
	}
	_ = m.Close()
}

// TestNemotronResident_tier1Slice is OP2b's tier 1 on Metal on a layer slice (docs/tasks/task-forward-architecture-2026-10.md
// §4.7, "Tier 1 on a layer slice", registered before any run): layers 0, 1 and 14 of NVIDIA-Nemotron-Nano-9B-v2's Q8_0 GGUF
// as an int4 .giw, resident on Metal against the CPU model it was built from, 32 steps of greedy lockstep graded from
// position 2. G2-R's bars: min logit cosine >= 0.88 and argmax agreement >= 50%. The dense Qwen2.5-Coder-1.5B through its
// int4 Metal .giw runs first through the same harness; under min cosine 0.95 the run is void. A refused load is void too.
//
//	GOINFER_HEAVY_TESTS=1 go test -count=1 -timeout 10m -run '^TestNemotronResident_tier1Slice$' -v ./metal/
func TestNemotronResident_tier1Slice(t *testing.T) {
	requireHeavyModel(t)
	const steps = 32
	home, _ := os.UserHomeDir()
	ctrlPath := filepath.Join(home, "models", "qwen2.5-coder-1.5b-instruct-q4_k_m.int4.metal.v16.giw")
	slicePath := filepath.Join(home, "models", "nemotron", "nemotron-nano-9b-v2.slice-0-1-14.int4-metal.giw")
	for _, p := range []string{ctrlPath, slicePath} {
		if _, err := os.Stat(p); err != nil {
			t.Fatalf("VOID: %s is missing", p)
		}
	}
	opts := decoder.Options{Quant: "int4", ResidentContext: 64}
	ctrl := residentLockstepAt(t, ctrlPath, opts, steps)
	debug.FreeOSMemory()
	t.Logf("control, dense 1.5B: min cosine %.6f, argmax %d/%d", ctrl.minCos, ctrl.exact, ctrl.steps)
	if ctrl.minCos < 0.95 {
		t.Fatalf("VOID: the control's min cosine %.6f is under 0.95", ctrl.minCos)
	}
	nemo := residentLockstepAt(t, slicePath, opts, steps)
	pass := nemo.minCos >= 0.88 && 2*nemo.exact >= nemo.steps
	t.Logf("Nemotron-H slice 0,1,14: min cosine %.6f, argmax %d/%d -> %s", nemo.minCos, nemo.exact, nemo.steps,
		map[bool]string{true: "PASS", false: "FAIL"}[pass])
	if !pass {
		t.Errorf("the slice misses G2-R's bars (min cosine >= 0.88, argmax >= 50%%)")
	}
}
