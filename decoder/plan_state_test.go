package decoder

import (
	"slices"
	"strings"
	"testing"
)

// cacheStateBytes is what c's recurrent and conv state holds: the rolling windows and the matrix state, each layer's. A hybrid's scratch
// (w4a8Act, rewritten every step) is not state and is left out.
func cacheStateBytes(c *KVCache) int64 {
	var floats int
	win := func(w [][]float32) {
		for _, row := range w {
			floats += len(row)
		}
	}
	for _, s := range c.delta {
		if s != nil {
			win(s.convWin)
			floats += len(s.s)
		}
	}
	for _, s := range c.conv {
		if s != nil {
			win(s.convWin)
		}
	}
	for _, s := range c.mamba {
		if s != nil {
			win(s.ConvWin)
			floats += len(s.SSM)
		}
	}
	for _, s := range c.kda {
		if s != nil {
			win(s.convWinQ)
			win(s.convWinK)
			win(s.convWinV)
			floats += len(s.s)
		}
	}
	return int64(floats) * 4
}

// TestPlan_pricesRecurrentState holds the plan's state line to what the CPU forward holds once every window is full, on one fixture of
// each recurrent op kind, and to the plan's total. The state is read off the cache, so the figure is not this test's own arithmetic.
func TestPlan_pricesRecurrentState(t *testing.T) {
	for _, fx := range []struct{ dir, kind string }{
		{"qwen35-tiny", "Gated DeltaNet"}, {"qwen3next-tiny", "Gated DeltaNet"}, {"olmo_hybrid-tiny", "Gated DeltaNet"},
		{"nemotron-tiny", "Mamba-2"}, {"nemotron3nano-tiny", "Mamba-2"},
		{"lfm2-tiny", "ShortConv"}, {"bailing_hybrid-tiny", "KDA"},
	} {
		t.Run(fx.dir, func(t *testing.T) {
			m, err := Load("../testdata/"+fx.dir, Options{Backend: "cpu", DirectLoad: true})
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			c := m.NewCache(0)
			for i := range 16 { // past every fixture's conv kernel
				if _, err := m.forward(i%m.Config().VocabSize, c); err != nil {
					t.Fatal(err)
				}
			}
			held := cacheStateBytes(c)
			if held == 0 {
				t.Fatalf("VOID: %s (%s) holds no recurrent state after 16 tokens", fx.dir, fx.kind)
			}
			p := m.Plan("cpu", 1<<40, PlanRequest{Ctx: 64})
			if p.StateBytes != held {
				t.Errorf("%s: the plan's state line is %d bytes, the forward holds %d", fx.kind, p.StateBytes, held)
			}
			if want := p.DenseBytes + p.AllocSlackBytes + p.ExpertBytesUsed + p.KVBytes + p.HostCopyBytes + p.ExtraBytes + p.StateBytes; p.NeedBytes() != want {
				t.Errorf("NeedBytes %d, its lines sum to %d", p.NeedBytes(), want)
			}
			if tight := m.Plan("cpu", p.NeedBytes()-1, PlanRequest{Ctx: 64, CtxPinned: true}); tight.Placement == PlacementResident {
				t.Errorf("one byte under the need, including state, the plan is still resident: %s", tight.Reason)
			}
			if !strings.Contains(p.Reason, "recurrent state") {
				t.Errorf("the reason does not name the state: %q", p.Reason)
			}
		})
	}
}

// TestDeviceLoad_stateOnEachBackend: the resident backends hold the two hybrids' state as the CPU does, and a dense model has no state line
// anywhere. Every tiny buffer is under CUDA's quantum, so this reads no rounding: TestCUDAStateBytes_roundsEachBuffer does.
func TestDeviceLoad_stateOnEachBackend(t *testing.T) {
	for _, dir := range []string{"qwen3next-tiny", "nemotron-tiny"} {
		m, err := Load("../testdata/"+dir, Options{Backend: "cpu", DirectLoad: true})
		if err != nil {
			t.Fatal(err)
		}
		c := m.NewCache(0)
		for i := range 16 {
			if _, err := m.forward(i, c); err != nil {
				t.Fatal(err)
			}
		}
		for _, be := range []string{"cpu", "webgpu", "metal", "cuda"} {
			if got, held := m.deviceLoad(be, false, false, false).StateBytes, cacheStateBytes(c); got != held {
				t.Errorf("%s on %s: state line %d, the forward holds %d", dir, be, got, held)
			}
		}
		m.Close()
	}
	dense, err := Load("../testdata/llama-tiny", Options{Backend: "cpu", DirectLoad: true})
	if err != nil {
		t.Fatal(err)
	}
	defer dense.Close()
	for _, be := range []string{"cpu", "webgpu", "metal", "cuda"} {
		if got := dense.deviceLoad(be, false, false, false).StateBytes; got != 0 {
			t.Errorf("llama-tiny on %s has a state line of %d bytes", be, got)
		}
	}
}

// TestCUDAStateBytes_roundsEachBuffer: CUDA charges a buffer of 2 MiB or more in whole 2 MiB, and a smaller one as asked.
func TestCUDAStateBytes_roundsEachBuffer(t *testing.T) {
	q := BuildQuery{Options: Options{Backend: "cuda"}, StateBuffers: []int64{1 << 20, 2 << 20, 2<<20 + 1, 5 << 20}}
	if got, want := defaultCUDABuildDemand(q).StateBytes, int64(1<<20+2<<20+4<<20+6<<20); got != want {
		t.Errorf("state on CUDA: %d, want %d", got, want)
	}
}

// TestStateBuffers_granite: no tracked fixture holds Granite-4.0-H's Mamba-2 layers, so its branch is read on a hand-built architecture whose
// five dimensions all differ. Each layer has a window of (DConv-1) rows of DInner+2*NGroups*DState and an SSM of NHeads*HeadDim*DState, in
// floats, and only the Mamba layers hold them.
func TestStateBuffers_granite(t *testing.T) {
	a := &Architecture{NumLayers: 3, granite: &graniteParams{NHeads: 2, HeadDim: 8, DState: 3, NGroups: 5, DConv: 4},
		layerIsMamba: func(l int) bool { return l != 1 }}
	win, ssm := int64((4-1)*(2*8+2*5*3))*4, int64(2*8*3)*4
	if got, want := stateBuffers(a), []int64{win, ssm, win, ssm}; !slices.Equal(got, want) {
		t.Errorf("Granite's state buffers: %v, want %v", got, want)
	}
}

// TestStateBuffers_kdaPerStream: a KDA layer's three conv windows are three buffers, so a backend's allocator rounds each on its own.
func TestStateBuffers_kdaPerStream(t *testing.T) {
	a := &Architecture{NumLayers: 2, kda: &kdaParams{HeadDim: 8, NumHeads: 2, ConvKernel: 3}, layerIsLinear: func(l int) bool { return l == 1 }}
	win, s := int64((3-1)*2*8)*4, int64(2*8*8)*4
	if got, want := stateBuffers(a), []int64{win, win, win, s}; !slices.Equal(got, want) {
		t.Errorf("KDA's state buffers: %v, want %v", got, want)
	}
}
