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

// TestPlan_pricesRecurrentCheckpoint: each backend's report prices the checkpoint at what a forward's state holds where it keeps one and at none elsewhere,
// whether or not this host links the backend. On a plan: every fixture that keeps one reaches an accepted CUDA plan, which shows it; a plan with no room
// beside its need shows none at the same placement, since it stays out of the need; and a declined plan on that backend shows none.
func TestPlan_pricesRecurrentCheckpoint(t *testing.T) {
	for _, fx := range []struct {
		dir  string
		kept bool
	}{{"qwen35-tiny", true}, {"olmo_hybrid-tiny", true}, {"nemotron-tiny", true}, {"lfm2-tiny", false}, {"bailing_hybrid-tiny", false}, {"llama-tiny", false}} {
		t.Run(fx.dir, func(t *testing.T) {
			m, err := Load("../testdata/"+fx.dir, Options{Backend: "cpu", Quant: "int4", DirectLoad: true}) // CUDA's Mamba-2 takes int4, not f32
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
			if fx.kept && held == 0 {
				t.Fatalf("VOID: %s holds no state after a forward", fx.dir)
			}
			for _, b := range []string{"metal", "cuda", "webgpu", "cpu"} {
				want := int64(0)
				if fx.kept && (b == "metal" || b == "cuda") {
					want = held // tiny buffers, under CUDA's rounding quantum
				}
				got, perSlot := m.checkpointBytes(b)
				if wantPerSlot := b == "cuda" && want > 0; perSlot != wantPerSlot {
					t.Errorf("%s on %s: a checkpoint per slot %v, want %v", fx.dir, b, perSlot, wantPerSlot)
				}
				if got != want {
					t.Errorf("%s: the report's checkpoint is %d bytes, a forward holds %d", b, got, want)
				}
				p := m.Plan(b, 1<<40, PlanRequest{Ctx: 64, CtxPinned: true})
				if p.Placement == PlacementDecline {
					if p.CheckpointBytes != 0 {
						t.Errorf("%s: a declined plan shows a checkpoint of %d bytes", b, p.CheckpointBytes)
					}
					if fx.kept && b == "cuda" {
						t.Fatalf("VOID: CUDA declines %s, so no plan shows its checkpoint: %s", fx.dir, p.Reason)
					}
					continue
				}
				if p.CheckpointBytes != want {
					t.Errorf("%s: the plan's checkpoint is %d bytes, want %d", b, p.CheckpointBytes, want)
				}
				tight := m.Plan(b, p.NeedBytes(), PlanRequest{Ctx: 64, CtxPinned: true})
				if tight.Placement != p.Placement {
					t.Errorf("%s: at exactly its need, the plan reads %v: %s", b, tight.Placement, tight.Reason)
				}
				if tight.CheckpointBytes != 0 {
					t.Errorf("%s: with no room beside its need, the plan shows a checkpoint of %d bytes", b, tight.CheckpointBytes)
				}
				if b != "cuda" {
					continue
				}
				if under := m.Plan(b, p.NeedBytes()-1, PlanRequest{Ctx: 64, CtxPinned: true}); under.CheckpointBytes != 0 {
					t.Errorf("%s: a byte under its need, the plan (%v) shows a checkpoint of %d bytes", b, under.Placement, under.CheckpointBytes)
				}
				if none := m.Plan(b, 1, PlanRequest{Ctx: 64, CtxPinned: true}); none.Placement != PlacementDecline || none.CheckpointBytes != 0 {
					t.Errorf("%s: with a byte free, the plan reads %v with a checkpoint of %d bytes", b, none.Placement, none.CheckpointBytes)
				}
			}
		})
	}
}

// TestPlan_expertBiasesOnTheLineThatHoldsThem: gpt-oss's bias tables are fixed bytes on the line each backend leases them on, so an expert cap does not
// scale them; and the CPU's need agrees with its plan on an MoE model.
func TestPlan_expertBiasesOnTheLineThatHoldsThem(t *testing.T) {
	m, err := Load("testdata/gptoss_tiny.gguf", Options{Backend: "cpu", Quant: "int4", DirectLoad: true})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	bias := m.expertBiasBytes()
	if bias == 0 {
		t.Fatal("VOID: gptoss_tiny holds no expert biases")
	}
	full := m.ResidentWeightBytesPaged(0) - m.ResidentDenseWeightBytes() - bias
	for _, b := range []string{"cuda", "webgpu", "metal", "cpu"} {
		l := m.deviceLoad(b, false, false, true)
		if l.ExpertBytesFull != full {
			t.Errorf("%s: the scaled expert figure is %d, the experts alone %d", b, l.ExpertBytesFull, full)
		}
		onDense := b == "cuda" || b == "webgpu"
		if onDense == (l.ExpertBytesFixed >= bias) {
			t.Errorf("%s: %d fixed on experts with %d of biases", b, l.ExpertBytesFixed, bias)
		}
	}
	if got, want := m.ResidentDenseWeightBytesFor("webgpu")-m.ResidentDenseWeightBytes(), bias-m.kquantSavings("webgpu")-m.mambaProjSavings("webgpu"); got != want {
		t.Errorf("webgpu's dense figure holds %d past the tensors, want %d", got, want)
	}
	// Metal's autoMoESlots reads these: one more slot costs an expert alone, and what is left at no slots holds the biases.
	if perSlot := m.ResidentWeightBytesPaged(2) - m.ResidentWeightBytesPaged(1); perSlot <= 0 || perSlot*int64(m.w.arch.MoE.NumExperts) != full {
		t.Errorf("one more slot costs %d, the experts alone %d over %d experts", perSlot, full, m.w.arch.MoE.NumExperts)
	} else if got, want := m.ResidentWeightBytesPaged(1)-perSlot, m.ResidentDenseWeightBytes()+bias; got != want {
		t.Errorf("the fixed need reads %d, dense and biases %d", got, want)
	}
	// The pre-load estimate keeps them out of the expert bytes it scales.
	g, err := OpenGGUFMmap("testdata/gptoss_tiny.gguf")
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	var scaled float64
	for _, name := range g.Names() {
		if strings.Contains(name, "_exps") && !strings.HasSuffix(name, ".bias") {
			dims, _ := g.Dims(name)
			n := 1
			for _, d := range dims {
				n *= d
			}
			scaled += quantBytesPerElem(quantInt4) * float64(n)
		}
	}
	if _, experts := estimateGGUFWeightBreakdown("testdata/gptoss_tiny.gguf", quantInt4); experts != int64(scaled) || experts == 0 {
		t.Errorf("the estimate's expert bytes are %d, the expert matrices %d", experts, int64(scaled))
	}
	p := m.Plan("cpu", 1<<40, PlanRequest{Ctx: 64, CtxPinned: true})
	if got, want := m.ResidentNeedBytes("cpu", 0, p.Ctx, false, false), p.NeedBytes(); got != want {
		t.Errorf("ResidentNeedBytes %d, Plan.NeedBytes %d", got, want)
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
