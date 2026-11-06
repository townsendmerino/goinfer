//go:build cuda && goinfer_testhooks

package cuda

import (
	"fmt"
	"math"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// OP2b on CUDA, step 2b (docs/tasks/task-forward-architecture-2026-10.md §4.7, "OP2b on CUDA"): Nemotron-H resident on CUDA. Gate M is the tiny model
// against the CPU in greedy lockstep with a same-run control; gate S is the slot behaviour; the int8 decline is pinned by name. CUDA's admission
// reads program.CUDA, which lists Mamba-2 and the non-gated ReLU² FFN, so these tests need no feature of their own.

// nemotronFeaturesForTest is a no-op kept for its callers: CUDA declares no feature set since OP4cu-c2, and program.CUDA, which
// lists Mamba-2 and the non-gated ReLU² FFN, admits Nemotron-H (docs/tasks/task-forward-architecture-2026-10.md §4.9a).
func nemotronFeaturesForTest() func() { return func() {} }

// lockstepReading is one fixture's CUDA-resident run against the CPU at the same quant: the worst logit cosine over positions from 2, and on how many
// steps the argmax agreed.
type lockstepReading struct {
	minCos       float64
	exact, steps int
}

// residentLockstepCUDA loads dir on CUDA and on the CPU at quant and runs them in greedy lockstep off the CPU: a fixed seed, then the CPU's argmax. The CUDA
// side must be resident (a silent CPU fallback passes every bar). Both logit rows are asserted non-zero before a cosine is taken, and a cosine where
// exactly one side is all zeros scores 0 (cosF32 would score it 1).
func residentLockstepCUDA(t *testing.T, dir, quant string, steps int) lockstepReading {
	t.Helper()
	path := filepath.Join("../../testdata", dir)
	mg, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: quant})
	if err != nil {
		t.Fatalf("load (cuda) %s at %s: %v", dir, quant, err)
	}
	defer mg.Close()
	rf := mg.ResidentForwardForTest()
	if rf == nil {
		t.Fatalf("cuda resident DECLINED %s at %s (%s): %s", dir, quant, mg.DecodePath(), mg.ResidentDecline())
	}
	mc, err := decoder.Load(path, decoder.Options{Quant: quant})
	if err != nil {
		t.Fatalf("load (cpu) %s: %v", dir, err)
	}
	defer mc.Close()
	_, _, _, _, _, _, vocab := mc.Dims()
	seed := []int{1, 17 % vocab, 113 % vocab, 29 % vocab, 71 % vocab, 5 % vocab}
	cache := mc.NewCache(steps + 2)
	rf.Reset()
	st := lockstepReading{minCos: 1, steps: steps}
	tok := seed[0]
	for i := range steps {
		cl, err := mc.ForwardForTest(tok, cache)
		if err != nil {
			t.Fatalf("cpu forward: %v", err)
		}
		gl, err := rf.Forward(mc.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatalf("cuda forward: %v", err)
		}
		if allZero(cl) || allZero(gl) {
			t.Fatalf("%s step %d: a logit row is all zeros (cpu %v, cuda %v): the reading is void", dir, i, allZero(cl), allZero(gl))
		}
		if i >= 2 { // the first positions are not a meaningful cosine (Metal's gate skips them for the same reason)
			c, _ := cosF32(cl, gl)
			if math.IsNaN(c) {
				c = 0
			}
			st.minCos = min(st.minCos, c)
		}
		if argmaxF(cl) == argmaxF(gl) {
			st.exact++
		}
		if i+1 < len(seed) {
			tok = seed[i+1]
		} else {
			tok = argmaxF(cl)
		}
	}
	return st
}

// TestNemotronResidentCUDA_tinyAgainstControl is OP2b's gate M on CUDA: nemotron-tiny resident on CUDA against the CPU at int4, 32 steps of greedy lockstep,
// graded from position 2, the bar relative to llama-tiny through the same harness in the same run: min logit cosine at least the control's minus 0.001,
// argmax agreement on at least the control's count minus 2. With mamba_conv reading its input one element off (mambaConvShiftForTest) in the CUDA build only,
// the same comparison must fail. nemotron3nano-tiny, whose MoE blocks decline by name (the decoder's admission says so before the build is asked; the build's own MoE decline is the backstop), is checked for exactly that decline.
func TestNemotronResidentCUDA_tinyAgainstControl(t *testing.T) {
	requireCUDADevice(t)
	defer nemotronFeaturesForTest()()
	const steps, quant = 32, "int4"
	ctrl := residentLockstepCUDA(t, "llama-tiny", quant, steps)
	nemo := residentLockstepCUDA(t, "nemotron-tiny", quant, steps)
	pass := func(s lockstepReading) bool { return s.minCos >= ctrl.minCos-0.001 && s.exact >= ctrl.exact-2 }
	t.Logf("%s: control llama-tiny min cosine %.6f, argmax %d/%d; nemotron-tiny min cosine %.6f, argmax %d/%d -> %s",
		quant, ctrl.minCos, ctrl.exact, ctrl.steps, nemo.minCos, nemo.exact, nemo.steps, map[bool]string{true: "pass", false: "FAIL"}[pass(nemo)])
	if !pass(nemo) {
		t.Errorf("%s: nemotron-tiny on CUDA is outside the control's band", quant)
	}
	mambaConvShiftForTest = 1
	planted := residentLockstepCUDA(t, "nemotron-tiny", quant, steps)
	mambaConvShiftForTest = 0
	t.Logf("%s: plant (mamba_conv input shifted one element, CUDA only): min cosine %.6f, argmax %d/%d -> %s", quant, planted.minCos, planted.exact,
		planted.steps, map[bool]string{true: "passes, which it must not", false: "fails, as it must"}[pass(planted)])
	if pass(planted) {
		t.Errorf("%s: the planted conv defect stayed inside the band: gate M cannot see a broken Mamba kernel", quant)
	}

	m, err := decoder.Load(filepath.Join("../../testdata/nemotron3nano-tiny"), decoder.Options{Backend: "cuda", Quant: quant})
	if err != nil {
		t.Fatalf("load nemotron3nano-tiny: %v", err)
	}
	defer m.Close()
	if m.ResidentForwardForTest() != nil {
		t.Error("nemotron3nano-tiny went resident on CUDA, whose build has no non-gated MoE")
	} else if why := m.ResidentDecline(); !strings.Contains(why, "Nemotron-H MoE FFN block has no GPU resident implementation") {
		t.Errorf("nemotron3nano-tiny declines with %q, want the decoder's Nemotron-H MoE decline by name", why)
	}
}

// greedyReplyCUDA feeds prompt to the resident from position 0 (which resets its recurrent state) and returns n greedy tokens after it.
func greedyReplyCUDA(t *testing.T, rf decoder.ResidentForward, m *decoder.Model, prompt []int, n int) []int {
	t.Helper()
	var out []int
	tok := prompt[0]
	for i := 0; i < len(prompt)+n-1; i++ {
		if i == 0 {
			rf.Reset()
		}
		l, err := rf.Forward(m.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatalf("cuda forward: %v", err)
		}
		if i+1 < len(prompt) {
			tok = prompt[i+1]
			continue
		}
		tok = argmaxF(l)
		out = append(out, tok)
	}
	return out
}

// TestNemotronResidentCUDA_slots is OP2b's gate S on CUDA: a recurrent family gets the KV slots it asks for and the batched step declines it, and two conversations
// taking turns on one slot, its state reset at each one's position 0, each give the reply they give alone. With the reset planted to leave the Mamba-2
// state alone (mambaResetSkipForTest), the turns must differ.
func TestNemotronResidentCUDA_slots(t *testing.T) {
	requireCUDADevice(t)
	defer nemotronFeaturesForTest()()
	path := filepath.Join("../../testdata/nemotron-tiny")
	load := func() (*decoder.Model, *cudaResident) {
		m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4", ResidentKVSlots: 4})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { m.Close() })
		r, ok := m.ResidentForwardForTest().(*cudaResident)
		if !ok {
			t.Fatalf("nemotron-tiny not resident (%s): %s", m.DecodePath(), m.ResidentDecline())
		}
		return m, r
	}
	m, a := load()
	if got := m.ResidentKVSlotsRequestFor("cuda"); got != 4 {
		t.Errorf("Nemotron-H asks for %d resident KV slots on CUDA, want the 4 it was given", got)
	}
	if n := a.KVSlots(); n != 4 {
		t.Errorf("the resident built %d KV slots, want 4", n)
	}
	if lo, hi := a.BatchStepRange(); hi != 0 {
		t.Errorf("the batched step admits Nemotron-H (range %d..%d)", lo, hi)
	}
	pA, pB := []int{1, 40, 9, 77, 3, 51}, []int{1, 12, 88, 6, 30}
	const n = 10
	_, ra := load()
	aloneA := greedyReplyCUDA(t, ra, m, pA, n)
	_, rb := load()
	aloneB := greedyReplyCUDA(t, rb, m, pB, n)
	turns := func(r *cudaResident) (a1, b, a2 []int) {
		return greedyReplyCUDA(t, r, m, pA, n), greedyReplyCUDA(t, r, m, pB, n), greedyReplyCUDA(t, r, m, pA, n)
	}
	a1, b, a2 := turns(a)
	same := func(x, y []int) bool { return fmt.Sprint(x) == fmt.Sprint(y) }
	t.Logf("alone A %v, B %v; turns A %v, B %v, A %v", aloneA, aloneB, a1, b, a2)
	if !same(a1, aloneA) || !same(b, aloneB) || !same(a2, aloneA) {
		t.Error("a conversation taking turns on the slot does not give the reply it gives alone")
	}
	mambaResetSkipForTest = true
	_, rp := load()
	p1, pb, p2 := turns(rp)
	mambaResetSkipForTest = false
	t.Logf("plant (reset leaves the Mamba-2 state): turns A %v, B %v, A %v", p1, pb, p2)
	if same(pb, aloneB) && same(p2, aloneA) {
		t.Error("with the Mamba-2 state left between conversations the turns still match: gate S cannot see a missing reset")
	}
}

// TestNemotronResidentCUDA_int8Declines: at int8int8 Nemotron-H does not go resident on CUDA, so the CPU runs it at the int8 it was loaded with instead of
// re-quantizing its projections to int4 without notice. Two declines hold it: the decoder's (GOINFER_SSM_RESIDENT unset) and the build's own, reached when
// that variable admits int8 (the build is the backstop, and the arm that names it). int4 builds. With the build's decline planted off, the int8int8 build
// must go through, which this test must see.
func TestNemotronResidentCUDA_int8Declines(t *testing.T) {
	requireCUDADevice(t)
	defer nemotronFeaturesForTest()()
	path := filepath.Join("../../testdata/nemotron-tiny")
	load := func(quant string) (resident bool, why string) {
		m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: quant})
		if err != nil {
			t.Fatalf("load at %s: %v", quant, err)
		}
		defer m.Close()
		return m.ResidentForwardForTest() != nil, m.ResidentDecline()
	}
	if got, why := load("int8int8"); got || !strings.Contains(why, "int4") {
		t.Errorf("int8int8 with the decoder's default: resident = %v, decline %q, want the decoder's int4 decline", got, why)
	}
	if got, why := load("int4"); !got {
		t.Errorf("int4: not resident (%s)", why)
	}
	t.Setenv("GOINFER_SSM_RESIDENT", "1")
	if got, why := load("int8int8"); got || !strings.Contains(why, "int4 only") {
		t.Errorf("int8int8 with int8 admitted by the decoder: resident = %v, decline %q, want the build's int4-only decline", got, why)
	}
	mambaInt8DeclineOffForTest = true
	got, _ := load("int8int8")
	mambaInt8DeclineOffForTest = false
	if !got {
		t.Error("with the build's int8 decline planted off, int8int8 still does not go resident: this test cannot see a missing decline")
	}
}

// TestNemotronResidentCUDA_blockKinds: each block of nemotron-tiny is built as the one operation the family's own block kind says (Mamba-2 mixer with no
// feed-forward, attention with none, feed-forward only), read from decoder.Model.NemotronBlockKind and not from ResidentLayerOps. A Mamba block owns its two
// state buffers and no KV; a feed-forward-only block owns u and d and no norm after the pre-norm.
func TestNemotronResidentCUDA_blockKinds(t *testing.T) {
	requireCUDADevice(t)
	defer nemotronFeaturesForTest()()
	m, err := decoder.Load(filepath.Join("../../testdata/nemotron-tiny"), decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	r, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatalf("not resident (%s): %s", m.DecodePath(), m.ResidentDecline())
	}
	count := map[decoder.ResidentMixer]int{}
	for l := range r.layers {
		Ly := &r.layers[l]
		var wantMixer decoder.ResidentMixer
		var wantFFN bool
		switch m.NemotronBlockKind(l) {
		case 0:
			wantMixer, wantFFN = decoder.MixerMamba, false
		case 1:
			wantMixer, wantFFN = decoder.MixerAttention, false
		case 2:
			wantMixer, wantFFN = decoder.MixerNone, true
		default:
			t.Fatalf("layer %d: block kind %d is not built in this test", l, m.NemotronBlockKind(l))
		}
		if Ly.mixer != wantMixer || Ly.ffn != wantFFN {
			t.Errorf("layer %d: built as (%v, ffn %v), the family says (%v, ffn %v)", l, Ly.mixer, Ly.ffn, wantMixer, wantFFN)
		}
		count[Ly.mixer]++
		hasState := Ly.mbWin.Len() > 0 && Ly.mbSSM.Len() > 0
		if hasState != (wantMixer == decoder.MixerMamba) {
			t.Errorf("layer %d: Mamba state buffers present = %v for a %v block", l, hasState, wantMixer)
		}
		if Ly.noKV() != (wantMixer != decoder.MixerAttention) || (Ly.kvDim == 0) != (wantMixer != decoder.MixerAttention) {
			t.Errorf("layer %d: noKV %v, kvDim %d for a %v block", l, Ly.noKV(), Ly.kvDim, wantMixer)
		}
		if wantMixer == decoder.MixerNone && (Ly.u.N == 0 || Ly.d.N == 0 || Ly.ffnI != Ly.u.N) {
			t.Errorf("layer %d: feed-forward-only block has u %d, d %d, ffnI %d", l, Ly.u.N, Ly.d.N, Ly.ffnI)
		}
	}
	t.Logf("nemotron-tiny blocks by operation: %v", count)
	if count[decoder.MixerMamba] == 0 || count[decoder.MixerAttention] == 0 || count[decoder.MixerNone] == 0 {
		t.Errorf("VOID: nemotron-tiny no longer has all three block kinds (%v)", count)
	}
}

// MEM13-copies-CUDA (docs/measurements/mem13-copies-cuda-2026-11/README.md): a Mamba-2 layer's in_proj and out_proj are planned while the build walks the
// layers and streamed to the device at upload, like every other dense projection (pack_stream.go), so the host holds no packed copy of them.
//
//	go test -count=1 -tags 'cuda goinfer_testhooks' -run '^TestMambaHostCopy_' -v ./backend/cuda/

// mambaLayersOf loads the nemotron-tiny fixture on CUDA and returns, for each Mamba-2 layer, what the build's walk makes of it.
func mambaLayersOf(t *testing.T) (m *decoder.Model, layers []int, p *mambaParams, hidden int) {
	t.Helper()
	var err error
	m, err = decoder.Load(filepath.Join("../../testdata", "nemotron-tiny"), decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load nemotron-tiny: %v", err)
	}
	t.Cleanup(func() { m.Close() })
	if p, err = buildMambaParams(m); err != nil || p == nil {
		t.Fatalf("nemotron-tiny has no Mamba-2 geometry: %v", err)
	}
	hidden, _, _, _, _, _, _ = m.Dims()
	for l, ly := range m.Program().Layers {
		if mix, _, ok := cudaMixerOf(ly); ok && mix == decoder.MixerMamba {
			layers = append(layers, l)
		}
	}
	if len(layers) == 0 {
		t.Fatal("nemotron-tiny has no Mamba-2 layer to read")
	}
	return m, layers, p, hidden
}

// packMambaLayer plans its two projections and holds no packed words; the buffers it names are the sizes packWeight would make.
func TestMambaHostCopy_planningHoldsNoPackedWords(t *testing.T) {
	m, layers, p, hidden := mambaLayersOf(t)
	for _, l := range layers {
		hm, err := packMambaLayer(m, l, hidden, p)
		if err != nil {
			t.Fatalf("layer %d: %v", l, err)
		}
		for name, h := range map[string]hostW{"in_proj": hm.in, "out_proj": hm.out} {
			if n := len(h.wpk) + len(h.ws) + len(h.ws16); n != 0 {
				t.Errorf("layer %d %s holds %d packed words before upload, want 0 (a planned weight is packed at upload)", l, name, n)
			}
			if h.src == nil {
				t.Errorf("layer %d %s is not a planned weight: it names no source for the upload to stream", l, name)
				continue
			}
			eager, err := packWeight(h.src)
			if err != nil {
				t.Fatalf("layer %d %s: packWeight: %v", l, name, err)
			}
			if h.kind != eager.kind || h.nWpk != len(eager.wpk) || h.nWs != len(eager.ws) || h.nWs16 != len(eager.ws16) {
				t.Errorf("layer %d %s: the plan names kind %s buffers %d/%d/%d, the packed weight is kind %s buffers %d/%d/%d", l, name,
					h.kind, h.nWpk, h.nWs, h.nWs16, eager.kind, len(eager.wpk), len(eager.ws), len(eager.ws16))
			}
		}
	}
}

// The device bytes a Mamba layer's projections get are the packed upload's, at one staging block and at several.
func TestMambaHostCopy_deviceBytesIdenticalToPackWeight(t *testing.T) {
	r := streamRig(t)
	m, layers, p, hidden := mambaLayersOf(t)
	for _, l := range layers {
		hm, err := packMambaLayer(m, l, hidden, p)
		if err != nil {
			t.Fatalf("layer %d: %v", l, err)
		}
		for name, h := range map[string]hostW{"in_proj": hm.in, "out_proj": hm.out} {
			if h.src == nil {
				continue // the planning test reports it
			}
			eager, err := packWeight(h.src)
			if err != nil {
				t.Fatalf("layer %d %s: packWeight: %v", l, name, err)
			}
			want := devBytes(t, r.upW(eager), h)
			for _, words := range []int{1 << 20, 64, 1} {
				withStage(t, words)
				r.pk = nil
				got := devBytes(t, r.upW(h), h)
				for k, b := range want {
					if string(got[k]) != string(b) {
						t.Errorf("layer %d %s, staging %d words: device buffer %s differs from the packed upload's", l, name, words, k)
					}
				}
			}
		}
	}
	if r.setupErr != nil {
		t.Fatalf("upload error: %v", r.setupErr)
	}
}
