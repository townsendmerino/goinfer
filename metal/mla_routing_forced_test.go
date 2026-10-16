//go:build darwin && goinfer_testhooks

package metal

import (
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// residentLockstepForced is residentLockstep with every MoE layer's routing on Metal replaced, at every step, by the
// experts and weights the CPU chose for the same token (resident.moeForce, decoder's MoE trace), so a near-tie the two
// backends break differently cannot change which experts run.
func residentLockstepForced(t *testing.T, dir string, steps int) lockstep {
	t.Helper()
	m, err := decoder.Load(filepath.Join("..", "testdata", dir), decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load %s: %v", dir, err)
	}
	defer m.Close()
	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("%s: build resident: %v", dir, err)
	}
	defer r.Close()
	if r.moe == nil {
		t.Fatalf("%s has no MoE layer", dir)
	}
	k := r.moe.k
	var moeLayers []int
	for l := range r.layers {
		if r.layers[l].moe != nil {
			moeLayers = append(moeLayers, l)
		}
	}
	r.moeForce.idx = NewBufferUint32s(r.d, make([]uint32, len(r.layers)*k))
	r.moeForce.wgt = NewBufferFloats(r.d, make([]float32, len(r.layers)*k))
	decoder.StartMoESelTraceForTest()
	defer decoder.StopMoESelTraceForTest()
	a := &metalResident{r: r, hidden: r.H}
	_, _, _, _, _, _, vocab := m.Dims()
	seed := []int{1, 17 % vocab, 113 % vocab, 29 % vocab, 71 % vocab, 5 % vocab}
	cache := m.NewCache(0)
	st := lockstep{minCos: 1, steps: steps}
	tok := seed[0]
	for i := range steps {
		cpuL, err := m.ForwardForTest(tok, cache)
		if err != nil {
			t.Fatal(err)
		}
		sel, wts := decoder.TakeMoESelTraceForTest()
		if len(sel) != len(moeLayers) {
			t.Fatalf("%s step %d: the CPU routed %d MoE calls, the resident has %d MoE layers", dir, i, len(sel), len(moeLayers))
		}
		idx, wgt := r.moeForce.idx.U32s(), r.moeForce.wgt.Floats()
		for j, l := range moeLayers {
			for c := range k {
				idx[l*k+c], wgt[l*k+c] = uint32(sel[j][c]), wts[j][c]
			}
		}
		gpuL, err := a.Forward(m.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatal(err)
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

// TestMLAResident_routingForced is OP2a's diagnostic for gate M's MoE fixtures (owner, 2026-10-16): the gate's comparison
// with every MoE layer's routing forced to the CPU's choice on all three fixtures, control included. If kimi-tiny's
// cosine then holds the bar (the control's minus 0.001) on every graded step, its unforced dip at steps 8-9 was a routing
// near-tie and the MLA path is clear.
func TestMLAResident_routingForced(t *testing.T) {
	const steps = 32
	ctrl := residentLockstepForced(t, "qwen3moe-tiny", steps)
	t.Logf("control qwen3moe-tiny, routing forced: min cosine %.6f, argmax %d/%d", ctrl.minCos, ctrl.exact, ctrl.steps)
	for _, fx := range []string{"deepseek-tiny", "kimi-tiny"} {
		s := residentLockstepForced(t, fx, steps)
		ok := s.minCos >= ctrl.minCos-0.001 && s.exact >= ctrl.exact-2
		t.Logf("%s, routing forced: min cosine %.6f, argmax %d/%d -> %s", fx, s.minCos, s.exact, s.steps, map[bool]string{true: "pass", false: "FAIL"}[ok])
		if !ok {
			t.Errorf("%s with routing forced misses gate M (min cosine >= %.6f, argmax >= %d)", fx, ctrl.minCos-0.001, ctrl.exact-2)
		}
	}
}
