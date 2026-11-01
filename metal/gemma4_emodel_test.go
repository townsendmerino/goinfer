//go:build darwin && goinfer_testhooks

package metal

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/program"
)

const (
	eModelDir    = "../testdata/gemma4-emodel-tiny"
	eModelGolden = "../testdata/gemma4_emodel_tiny_golden.json"
)

// eModelPrompt is the golden's prompt followed by its greedy continuation: 18 positions, so the KV-shared layers
// attend over history well past the sliding window (4).
func eModelPrompt(t *testing.T) []int {
	t.Helper()
	raw, err := os.ReadFile(eModelGolden)
	if err != nil {
		t.Skipf("no golden (%v) — run scripts/pin_gemma4_emodel_tiny.py", err)
	}
	var g struct {
		PromptIDs       []int `json:"prompt_ids"`
		ContinuationIDs []int `json:"continuation_ids"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	return append(append([]int(nil), g.PromptIDs...), g.ContinuationIDs...)
}

// eModelG1 is the measurement G1 grades (docs/tasks/task-multimodal-support-2026-10.md): Metal resident against the
// CPU, int4 on both sides, teacher-forced over eModelPrompt, plus the CPU's own int4-vs-f32 agreement for the
// envelope.
type eModelG1 struct {
	n, exact, gaps3        int
	worstTie               float64
	meanMetal, meanCPU     float64
	minMetal               float64
	firstHardPos, metalArg int
	metal                  [][]float32 // Metal's logits per position, for the bit-identity check on defect (5)
}

func runEModelG1(t *testing.T) eModelG1 {
	t.Helper()
	if _, err := os.Stat(eModelDir); err != nil {
		t.Skipf("no fixture (%s) — run scripts/pin_gemma4_emodel_tiny.py", eModelDir)
	}
	prompt := eModelPrompt(t)
	mg, err := decoder.Load(eModelDir, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load (metal int4): %v", err)
	}
	defer mg.Close()
	if err := program.Metal.Check(mg.Program()); err != nil || len(mg.FeaturesOutsideProgram()) > 0 {
		t.Fatalf("metal declines the tiny E-model: %v, outside the program %v", err, mg.FeaturesOutsideProgram())
	}
	r, err := buildResident(mg)
	if err != nil {
		t.Fatalf("buildResident: %v", err)
	}
	defer r.Close()
	shared := 0
	for _, L := range r.layers {
		if L.kvShared {
			shared++
		}
	}
	wantWide := 512
	if g4OneFFNWidthForTest {
		wantWide = 256 // G2 defect (4) planted: let G1 grade it rather than this shape check
	}
	if r.pleP != 32 || shared != 2 || r.layers[4].ffnI != wantWide || r.layers[0].ffnI != 256 {
		t.Fatalf("resident did not take the E-model shape: P=%d shared=%d ffn[0]=%d ffn[4]=%d", r.pleP, shared, r.layers[0].ffnI, r.layers[4].ffnI)
	}
	mc4, err := decoder.Load(eModelDir, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cpu int4): %v", err)
	}
	defer mc4.Close()
	mcF, err := decoder.Load(eModelDir, decoder.Options{Quant: "f32"})
	if err != nil {
		t.Fatalf("load (cpu f32): %v", err)
	}
	defer mcF.Close()

	c4, cF := mc4.NewCache(len(prompt)), mcF.NewCache(len(prompt))
	res := eModelG1{n: len(prompt), minMetal: 1, firstHardPos: -1}
	var sumM, sumC float64
	for i, tok := range prompt {
		cpu4, err := mc4.ForwardForTest(tok, c4)
		if err != nil {
			t.Fatalf("cpu int4 pos %d: %v", i, err)
		}
		cpuF, err := mcF.ForwardForTest(tok, cF)
		if err != nil {
			t.Fatalf("cpu f32 pos %d: %v", i, err)
		}
		gpu := append([]float32(nil), r.ForwardEmb(mg.EmbedResidentForTest(tok), i)...)
		res.metal = append(res.metal, gpu)
		cm, _ := cosMaxAbs(cpu4, gpu)
		cc, _ := cosMaxAbs(cpuF, cpu4)
		sumM += cm
		sumC += cc
		res.minMetal = math.Min(res.minMetal, cm)
		ca, ga := argmaxF(cpu4), argmaxF(gpu)
		if ca == ga {
			res.exact++
		} else {
			// The two-geometry near-tie rule: benign iff the CPU's top-1 and the index Metal chose are within 3%
			// in the CPU's logits.
			gap := (float64(cpu4[ca]) - float64(cpu4[ga])) / (math.Abs(float64(cpu4[ca])) + 1e-30)
			res.worstTie = math.Max(res.worstTie, gap)
			if gap > 0.03 {
				res.gaps3++
				if res.firstHardPos < 0 {
					res.firstHardPos, res.metalArg = i, ga
				}
			}
		}
		t.Logf("  pos %2d Metal-vs-CPUint4 %.6f | CPUint4-vs-f32 %.6f  argmax cpu=%d metal=%d", i, cm, cc, ca, ga)
	}
	res.meanMetal, res.meanCPU = sumM/float64(len(prompt)), sumC/float64(len(prompt))
	t.Logf("E-model G1: exact-argmax %d/%d gaps>3%%=%d worstNearTie=%.2f%% | mean Metal-vs-CPUint4 %.6f (min %.6f), mean CPUint4-vs-f32 %.6f",
		res.exact, res.n, res.gaps3, res.worstTie*100, res.meanMetal, res.minMetal, res.meanCPU)
	return res
}

// pass reports G1's verdict as pre-registered: every argmax identical or a near-tie, and the mean Metal-vs-CPU(int4)
// cosine at least the mean CPU(int4)-vs-CPU(f32) cosine.
func (g eModelG1) pass() (bool, string) {
	if g.gaps3 > 0 {
		return false, "argmax diverges by more than 3% (not a near-tie)"
	}
	if g.meanMetal < g.meanCPU {
		return false, "mean Metal-vs-CPUint4 cosine below the CPU's own int4-vs-f32 envelope"
	}
	return true, ""
}

// TestGemma4EModel_metalResidentParity is S1's G1 on the tiny E-model: PLE, KV-shared layers and double-wide FFNs all
// running resident on Metal against the CPU.
func TestGemma4EModel_metalResidentParity(t *testing.T) {
	g := runEModelG1(t)
	if ok, why := g.pass(); !ok {
		t.Errorf("G1 FAIL: %s (first hard divergence at pos %d)", why, g.firstHardPos)
	}
}

// TestGemma4EModel_plantedDefects is S1's G2 (docs/tasks/task-multimodal-support-2026-10.md): each planted defect,
// alone, must turn G1 red. A defect that stays green means the fixture is degenerate along that axis.
func TestGemma4EModel_plantedDefects(t *testing.T) {
	defects := []struct {
		name string
		set  func(bool)
	}{
		{"(1) PLE branch skipped", func(on bool) { g4SkipPLEForTest = on }},
		{"(2) PLE token-identity term zeroed", decoder.SetGemma4PLEDropTokenForTest},
		{"(3) shared-KV source one owning layer off", func(on bool) { g4KVSrcOffForTest = on }},
		{"(4) one FFN width for the model", func(on bool) { g4OneFFNWidthForTest = on }},
		{"(6) layer scalar dropped", func(on bool) { g4DropLayerScalarForTest = on }},
		{"(7) v_norm dropped on non-K=V layers", func(on bool) { g4DropVNormForTest = on }},
	}
	for _, d := range defects {
		t.Run(d.name, func(t *testing.T) {
			d.set(true)
			defer d.set(false)
			g := runEModelG1(t)
			ok, why := g.pass()
			t.Logf("G2 %s: G1 %s %s", d.name, map[bool]string{true: "GREEN", false: "RED"}[ok], why)
			if ok {
				t.Errorf("planted defect %s left G1 green — the fixture cannot see it", d.name)
			}
		})
	}
}

// TestGemma4EModel_sharedStoreIsNoOp records why G2's defect (5) cannot go red on any fixture (amendment,
// docs/tasks/task-multimodal-support-2026-10.md). A shared layer's Q-only GEMV writes only the Q region of the qkv
// scratch, so its K/V slots still hold what its source layer — the last owning layer of the same attention type —
// stored earlier in the same token: the other type's writes end below that slot (fixture: sliding writes floats
// 0..191, the global K slot starts at 256; E2B: 0..2559 against 4096). An unskipped store therefore re-writes the
// source's own bytes. This pins that: with the store left in, Metal's logits are bit-identical to the clean run. If a
// layout change ever breaks it, the unskipped store would corrupt the source's cache, and this goes red first.
func TestGemma4EModel_sharedStoreIsNoOp(t *testing.T) {
	clean := runEModelG1(t)
	g4KeepSharedKVStoreForTest = true
	defer func() { g4KeepSharedKVStoreForTest = false }()
	kept := runEModelG1(t)
	for p := range clean.metal {
		for i := range clean.metal[p] {
			if math.Float32bits(clean.metal[p][i]) != math.Float32bits(kept.metal[p][i]) {
				t.Fatalf("pos %d logit %d: %v with the shared store kept vs %v skipped — the qkv scratch no longer holds the source's K/V there, so an unskipped store corrupts the source cache", p, i, kept.metal[p][i], clean.metal[p][i])
			}
		}
	}
	t.Logf("shared-layer store kept: logits bit-identical to the clean run over %d positions", len(clean.metal))
}

// gemma4StepFixtures are the tracked Gemma 4 dense checkpoints the step's per-row Gemma 4 layers are checked on (OP4m-d5a,
// docs/tasks/task-forward-architecture-2026-10.md §4.9): an E-model (PLE, layers sharing another's KV), and two plain dense
// ones, one with two attention geometries.
var gemma4StepFixtures = []string{"../testdata/gemma4-emodel-tiny", "../testdata/gemma4-dense-twogeom-tiny", "../testdata/gemma4-dense-scaled"}

// gemma4StepModel loads dir at int4 with eight KV slots and builds its resident with the step's Gemma 4 blocks set to on.
func gemma4StepModel(t *testing.T, dir string, on bool) (*decoder.Model, *resident) {
	t.Helper()
	prev := gemma4StepOn.Swap(on)
	defer gemma4StepOn.Store(prev)
	m, err := decoder.Load(dir, decoder.Options{Quant: "int4", ResidentContext: 1024, ResidentKVSlots: 8})
	if err != nil {
		t.Fatalf("load %s: %v", dir, err)
	}
	r, err := buildResident(m)
	if err != nil {
		m.Close()
		t.Fatalf("build %s's resident: %v", dir, err)
	}
	t.Cleanup(func() { r.Close(); m.Close() })
	return m, r
}

// gemma4IdentityDiff counts the logits of every step's rows that differ from production's single-token forward. Each
// sequence's history is written token by token, with the full row decode takes (the PLE tail included), into its slot and a
// twin. Then 12 steps run its rows on the step and the twins on ForwardEmb.
func gemma4IdentityDiff(t *testing.T, m *decoder.Model, r *resident, depths []int) int {
	t.Helper()
	B := len(depths)
	seed := uint32(7654321)
	rnd := func() int {
		seed ^= seed << 13
		seed ^= seed >> 17
		seed ^= seed << 5
		return int(seed % uint32(min(20000, r.V)))
	}
	for s, D := range depths {
		ids := make([]int, D)
		for i := range ids {
			ids[i] = rnd()
		}
		for _, slot := range []int{s, B + s} {
			if err := r.useKVSlot(slot); err != nil {
				t.Fatal(err)
			}
			for i, id := range ids {
				r.ForwardEmb(m.EmbedResidentForTest(id), i)
			}
		}
	}
	diff := 0
	for st := range 12 {
		seqs := make([]batchSeq, B)
		ref := make([][]float32, B)
		for s := range B {
			seqs[s] = batchSeq{slot: s, pos: depths[s] + st, emb: m.EmbedResidentForTest(rnd())}
			if err := r.useKVSlot(B + s); err != nil {
				t.Fatal(err)
			}
			ref[s] = append([]float32(nil), r.ForwardEmb(seqs[s].emb, seqs[s].pos)...)
		}
		got := mc3Step(t, r, seqs)
		for s := range B {
			for i := range ref[s] {
				if i >= len(got[s]) || math.Float32bits(got[s][i]) != math.Float32bits(ref[s][i]) {
					diff++
				}
			}
		}
	}
	return diff
}

// TestMC3Step_gemma4BitIdentical is OP4m-d5a's D5a-1: with the step's Gemma 4 blocks on, each tracked Gemma 4 dense fixture
// builds the batched step, and four sequences at different depths, each on its own slot, stepped together 12 times, equal
// production's single-token forward on every logit. With the blocks off, none builds it.
func TestMC3Step_gemma4BitIdentical(t *testing.T) {
	for _, dir := range gemma4StepFixtures {
		t.Run(dir[len("../testdata/"):], func(t *testing.T) {
			if _, r := gemma4StepModel(t, dir, false); r.batch != nil {
				t.Fatal("with the Gemma 4 blocks off, a Gemma 4 model built the batched step")
			}
			m, r := gemma4StepModel(t, dir, true)
			if r.batch == nil {
				t.Fatalf("with the Gemma 4 blocks on, no batched step: %s", r.batchIneligible())
			}
			if d := gemma4IdentityDiff(t, m, r, []int{3, 7, 12, 20}); d != 0 {
				t.Errorf("%d logits differ from production's single-token forward", d)
			}
		})
	}
}

// TestMC3Step_gemma4PlantsGoRed is D5a-1's two plants, each red: every row's PLE input from row 0's (on the E-model), and
// every row's attention at row 0's position (on each fixture).
func TestMC3Step_gemma4PlantsGoRed(t *testing.T) {
	for _, c := range []struct {
		name string
		on   *bool
		dirs []string
	}{{"every row's PLE input from row 0's", &stepPLERow0PlantForTest, gemma4StepFixtures[:1]},
		{"every row's attention at row 0's position", &stepAttnRow0PlantForTest, gemma4StepFixtures}} {
		for _, dir := range c.dirs {
			t.Run(dir[len("../testdata/"):], func(t *testing.T) {
				m, r := gemma4StepModel(t, dir, true)
				if r.batch == nil {
					t.Fatalf("with the Gemma 4 blocks on, no batched step: %s", r.batchIneligible())
				}
				*c.on = true
				defer func() { *c.on = false }()
				d := gemma4IdentityDiff(t, m, r, []int{3, 7, 12, 20})
				t.Logf("%s: %d logits differ", c.name, d)
				if d == 0 {
					t.Errorf("planted %s, and the step still equals production's forward", c.name)
				}
			})
		}
	}
}

// TestOP4mD5a_gemma4StepSpeed is OP4m-d5a's D5a-3, D3-3's cell on the Gemma 4 checkpoint at GOINFER_OP4MD5A_MODEL with the
// step's Gemma 4 blocks on (GOINFER_OP4MD5A_TIMING=1, under the timing lock). Both arms take the model's full rows and return
// logits to the host, as production decodes a Gemma 4 E-model (its PLE tail comes from the host, and its logits are softcapped).
func TestOP4mD5a_gemma4StepSpeed(t *testing.T) {
	op4mStepSpeed(t, stepSpeedCell{env: "GOINFER_OP4MD5A", label: "[OP4m-d5a D5a-3]", gemma4: true,
		want: func(r *resident) bool { return r.gemma4Blocks() }, wantWhat: "a Gemma 4 resident",
		rows: func(m *decoder.Model) func(int) []float32 { return m.EmbedResidentForTest }})
}
