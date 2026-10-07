//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

const (
	eModelDir    = "../testdata/gemma4-emodel-tiny"
	eModelGolden = "../testdata/gemma4_emodel_tiny_golden.json"
)

// eModelPrompt is the golden's prompt followed by its greedy continuation: 18 positions, so the KV-shared layers attend over history well past the
// sliding window (4).
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

// eModelG1 is the measurement G1c grades (docs/tasks/task-multimodal-support-2026-10.md, "S1 on CUDA"): the CUDA resident against the CPU, int4 on
// both sides, teacher-forced over eModelPrompt, plus the CPU's own int4-vs-f32 agreement for the envelope.
type eModelG1 struct {
	n, exact, gaps3    int
	worstTie           float64
	meanCUDA, meanCPU  float64
	minCUDA            float64
	firstHardPos       int
	cuda               [][]float32 // the resident's logits per position, for the bit-identity check on defect (5)
	pleP, shared       int
	ffnFirst, ffnWide  int
	resident, cpuInt4v bool
}

func runEModelG1(t *testing.T) eModelG1 {
	t.Helper()
	if _, err := os.Stat(eModelDir); err != nil {
		t.Skipf("no fixture (%s) — run scripts/pin_gemma4_emodel_tiny.py", eModelDir)
	}
	prompt := eModelPrompt(t)
	mg, err := decoder.Load(eModelDir, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cuda int4): %v", err)
	}
	defer mg.Close()
	r, ok := mg.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatalf("the tiny E-model did not build a CUDA resident (got %T): %s", mg.ResidentForwardForTest(), mg.ResidentDecline())
	}
	res := eModelG1{n: len(prompt), minCUDA: 1, firstHardPos: -1, pleP: r.pleP}
	for _, L := range r.layers {
		if L.kvShared {
			res.shared++
		}
	}
	res.ffnFirst, res.ffnWide = r.layers[0].ffnI, r.layers[4].ffnI
	wantWide := 512
	if g4OneFFNWidthForTest {
		wantWide = 256 // G2c defect (4) planted: let G1c grade it rather than this shape check
	}
	if r.pleP != 32 || res.shared != 2 || res.ffnWide != wantWide || res.ffnFirst != 256 {
		t.Fatalf("resident did not take the E-model shape: P=%d shared=%d ffn[0]=%d ffn[4]=%d", r.pleP, res.shared, res.ffnFirst, res.ffnWide)
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
	var sumG, sumC float64
	for i, tok := range prompt {
		cpu4, err := mc4.ForwardForTest(tok, c4)
		if err != nil {
			t.Fatalf("cpu int4 pos %d: %v", i, err)
		}
		cpuF, err := mcF.ForwardForTest(tok, cF)
		if err != nil {
			t.Fatalf("cpu f32 pos %d: %v", i, err)
		}
		gpuL, err := r.Forward(mg.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatalf("cuda pos %d: %v", i, err)
		}
		gpuL = append([]float32(nil), gpuL...)
		res.cuda = append(res.cuda, gpuL)
		cm, _ := cosMaxAbs(cpu4, gpuL)
		cc, _ := cosMaxAbs(cpuF, cpu4)
		sumG += cm
		sumC += cc
		res.minCUDA = math.Min(res.minCUDA, cm)
		ca, ga := argmaxF(cpu4), argmaxF(gpuL)
		if ca == ga {
			res.exact++
		} else {
			// The two-geometry near-tie rule: benign iff the CPU's top-1 and the index the resident chose are within 3% in the CPU's logits.
			gap := (float64(cpu4[ca]) - float64(cpu4[ga])) / (math.Abs(float64(cpu4[ca])) + 1e-30)
			res.worstTie = math.Max(res.worstTie, gap)
			if gap > 0.03 {
				res.gaps3++
				if res.firstHardPos < 0 {
					res.firstHardPos = i
				}
			}
		}
		t.Logf("  pos %2d CUDA-vs-CPUint4 %.6f | CPUint4-vs-f32 %.6f  argmax cpu=%d cuda=%d", i, cm, cc, ca, ga)
	}
	res.meanCUDA, res.meanCPU = sumG/float64(len(prompt)), sumC/float64(len(prompt))
	t.Logf("E-model G1c: exact-argmax %d/%d gaps>3%%=%d worstNearTie=%.2f%% | mean CUDA-vs-CPUint4 %.6f (min %.6f), mean CPUint4-vs-f32 %.6f",
		res.exact, res.n, res.gaps3, res.worstTie*100, res.meanCUDA, res.minCUDA, res.meanCPU)
	return res
}

// pass reports G1c's verdict as pre-registered: every argmax identical or a near-tie, and the mean CUDA-vs-CPU(int4) cosine at least the mean
// CPU(int4)-vs-CPU(f32) cosine.
func (g eModelG1) pass() (bool, string) {
	if g.gaps3 > 0 {
		return false, "argmax diverges by more than 3% (not a near-tie)"
	}
	if g.meanCUDA < g.meanCPU {
		return false, "mean CUDA-vs-CPUint4 cosine below the CPU's own int4-vs-f32 envelope"
	}
	return true, ""
}

// TestGemma4EModel_cudaResidentParity is G1c: PLE, KV-shared layers and double-wide FFNs all running resident on CUDA against the CPU.
func TestGemma4EModel_cudaResidentParity(t *testing.T) {
	g := runEModelG1(t)
	if ok, why := g.pass(); !ok {
		t.Errorf("G1c FAIL: %s (first hard divergence at pos %d)", why, g.firstHardPos)
	}
}

// TestGemma4EModel_plantedDefects is G2c: each planted defect, alone, must turn G1c red. A defect that stays green means the fixture is degenerate
// along that axis (fix the fixture, not the bar). Defect (5), the K/V store not skipped on a shared layer, was registered with its outcome open
// (on Metal it is a byte-for-byte no-op and cannot go red). On CUDA it DOES go red (2026-10-07: mean cosine 0.763998 against 0.999931 clean), because
// the shared layer's K/V scratch holds another layer's data and the store writes it through the alias into the source's cache; so it is a gate here.
func TestGemma4EModel_plantedDefects(t *testing.T) {
	defects := []struct {
		name string
		set  func(bool)
	}{
		{"(1) PLE branch skipped", func(on bool) { g4SkipPLEForTest = on }},
		{"(2) PLE token-identity term zeroed", decoder.SetGemma4PLEDropTokenForTest},
		{"(3) shared-KV source one owning layer off", func(on bool) { g4KVSrcOffForTest = on }},
		{"(4) one FFN width for the model", func(on bool) { g4OneFFNWidthForTest = on }},
		{"(5) K/V store not skipped on a shared layer", func(on bool) { g4KeepSharedKVStoreForTest = on }},
		{"(6) layer scalar dropped", func(on bool) { g4DropLayerScalarForTest = on }},
		{"(7) v_norm dropped on non-K=V layers", func(on bool) { g4DropVNormForTest = on }},
		{"(8) PLE per-layer input offset off by one layer", func(on bool) {
			if on {
				pleLayerShiftForTest = 1
			} else {
				pleLayerShiftForTest = 0
			}
		}},
	}
	for _, d := range defects {
		t.Run(d.name, func(t *testing.T) {
			d.set(true)
			defer d.set(false)
			g := runEModelG1(t)
			ok, why := g.pass()
			t.Logf("G2c %s: G1c %s %s (mean cosine %.6f, exact argmax %d/%d, gaps>3%% %d)", d.name, map[bool]string{true: "GREEN", false: "RED"}[ok], why, g.meanCUDA, g.exact, g.n, g.gaps3)
			if ok {
				t.Errorf("planted defect %s left G1c green — the fixture cannot see it", d.name)
			}
		})
	}
}

// TestGemma4Graphs_bitExact_emodel is the graphs half of G2c: with CUDA graphs captured (GOINFER_CUDA_GRAPHS_UNSAFE=1 forces it on this box, whose
// DEFAULT compute mode declines graphs) the E-model's decode is byte-identical to the live launches over 40 positions, past the sliding window, with the
// PLE tail of every row non-zero. The PLE branch sits in segB, so a mis-captured offset or a stale tail shows here as a replay-vs-live divergence.
func TestGemma4Graphs_bitExact_emodel(t *testing.T) {
	t.Setenv("GOINFER_CUDA_GRAPHS_UNSAFE", "1")
	graphsBitExact(t, eModelDir, false)
}

// eModelGreedy generates n greedy tokens from prompt on the given model.
func eModelGreedy(t *testing.T, m *decoder.Model, prompt []int, n int) []int {
	t.Helper()
	ch, gen := m.Generate(context.Background(), prompt, n, decoder.SamplingParams{})
	var out []int
	for id := range ch {
		out = append(out, id)
	}
	if err := gen.Err(); err != nil {
		t.Fatalf("generate: %v", err)
	}
	return out
}

// TestGemma4EModel_declinesAndFallbacks is G2c's decline check: every path that cannot carry an E-model's per-layer embeddings, shared KV and per-layer FFN
// width refuses by NAME (never a silent truncation), and a greedy generation through the sequential fallbacks (the batched prefill declined) takes the
// same tokens as the CPU's.
func TestGemma4EModel_declinesAndFallbacks(t *testing.T) {
	if _, err := os.Stat(eModelDir); err != nil {
		t.Skipf("no fixture (%s)", eModelDir)
	}
	mg, err := decoder.Load(eModelDir, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer mg.Close()
	r, ok := mg.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatalf("no CUDA resident: %s", mg.ResidentDecline())
	}
	// The batched prefill declines, by name.
	if err := r.prefillStaticDecline(); err == nil || !errors.Is(err, errPrefillDeclined) || !strings.Contains(err.Error(), "E-model") {
		t.Errorf("prefillStaticDecline = %v, want an errPrefillDeclined naming the E-model", err)
	}
	if _, err := r.PrefillLast(context.Background(), [][]float32{mg.EmbedResidentForTest(1), mg.EmbedResidentForTest(2)}, 0); err == nil || !errors.Is(err, errPrefillDeclined) {
		t.Errorf("PrefillLast on an E-model = %v, want errPrefillDeclined", err)
	}
	// A drafter and a LoRA adapter refuse, by name.
	if _, err := r.AttachDrafter(nil); err == nil || !strings.Contains(err.Error(), "E-model") {
		t.Errorf("AttachDrafter = %v, want a refusal naming the E-model", err)
	}
	if err := r.SetAdapter([]decoder.ResidentAdapterLayer{{}}); err == nil || !strings.Contains(err.Error(), "E-model") {
		t.Errorf("SetAdapter = %v, want a refusal naming the E-model", err)
	}
	// A hidden-sized row (what every other model passes) is refused by length: its PLE tail would be stale.
	if _, err := r.Forward(make([]float32, r.hidden), 0); err == nil || !strings.Contains(err.Error(), "embedding row") {
		t.Errorf("Forward with a hidden-sized row = %v, want a row-length refusal", err)
	}
	// UploadKV to a shared layer refuses instead of overwriting its source.
	if err := r.UploadKV(4, 0, make([]float32, r.layers[4].kvDim), make([]float32, r.layers[4].kvDim)); err == nil || !strings.Contains(err.Error(), "shares layer") {
		t.Errorf("UploadKV to a shared layer = %v, want a named refusal", err)
	}
	// Greedy tokens through the sequential fallbacks equal the CPU's on the same weights.
	mc, err := decoder.Load(eModelDir, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cpu): %v", err)
	}
	defer mc.Close()
	prompt := eModelPrompt(t)[:12]
	got, want := eModelGreedy(t, mg, prompt, 8), eModelGreedy(t, mc, prompt, 8)
	t.Logf("greedy 8 tokens: cuda %v, cpu %v", got, want)
	if len(got) != len(want) {
		t.Fatalf("cuda generated %d tokens, cpu %d", len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("token %d: cuda %d, cpu %d (this fixture's logits are not near-tied at these positions: G1c had 18/18 exact argmax)", i, got[i], want[i])
		}
	}
}
