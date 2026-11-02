//go:build cuda && goinfer_testhooks

package cuda

import (
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// Olmo Hybrid on CUDA against the CPU (docs/tasks/task-olmo-hybrid-cuda-2026-11.md, O-2 to O-4). The family's DeltaNet layers
// widen the write gate to [0, 2) (NegEigval) and run the output norm at an epsilon of their own (ONormEps); the bar is the one
// the other DeltaNet families hold in TestQwen35ResidentParityCUDA: worst cosine >= 0.95, drift <= 0.02, and the replay after
// Reset reproducing the first run. Each setting is then dropped at the build (dnSettingPlantForTest) and the run is scored
// again against the same CPU logits.

const olmoParityTokens = 128

// olmoParity is one CUDA run against the CPU's logits at the same tokens.
type olmoParity struct {
	worst, first, replay float64
	agree                int
}

func (p olmoParity) red() bool {
	return p.first-p.worst > 0.02 || p.worst < 0.95 || p.replay < 0.9999999
}

func olmoCPULogits(t *testing.T, ckpt, quant string) ([][]float32, int) {
	t.Helper()
	m, err := decoder.Load(ckpt, decoder.Options{Backend: "cpu", Quant: quant})
	if err != nil {
		t.Fatalf("load cpu: %v", err)
	}
	defer m.Close()
	_, _, _, _, _, _, vocab := m.Dims()
	cache := m.NewCache(olmoParityTokens + 1)
	out := make([][]float32, olmoParityTokens)
	for i := range olmoParityTokens {
		l, err := m.ForwardForTest((i*131+7)%vocab, cache)
		if err != nil {
			t.Fatalf("cpu forward[%d]: %v", i, err)
		}
		out[i] = append([]float32(nil), l...)
	}
	return out, vocab
}

func olmoCUDARun(t *testing.T, ckpt, quant string, cpu [][]float32, vocab, plant int) olmoParity {
	t.Helper()
	dnSettingPlantForTest = plant
	defer func() { dnSettingPlantForTest = 0 }()
	m, err := decoder.Load(ckpt, decoder.Options{Backend: "cuda", Quant: quant})
	if err != nil {
		t.Fatalf("load cuda: %v", err)
	}
	defer m.Close()
	rf := m.ResidentForwardForTest()
	if rf == nil {
		t.Fatalf("olmo_hybrid did not go resident on CUDA (path %q): %s", m.DecodePath(), m.ResidentDecline())
	}
	rf.Reset()
	p := olmoParity{worst: 1, first: 1, replay: 1}
	var opening [][]float32
	for i := range olmoParityTokens {
		lr, err := rf.Forward(m.EmbedResidentForTest((i*131+7)%vocab), i)
		if err != nil {
			t.Fatalf("resident forward[%d]: %v", i, err)
		}
		if len(opening) < 16 {
			opening = append(opening, append([]float32(nil), lr...))
		}
		if argmaxF32(cpu[i]) == argmaxF32(lr) {
			p.agree++
		}
		cos, _ := cosF32(cpu[i], lr)
		if i == 0 {
			p.first = cos
		}
		p.worst = min(p.worst, cos)
	}
	rf.Reset()
	for i := range opening {
		lr, err := rf.Forward(m.EmbedResidentForTest((i*131+7)%vocab), i)
		if err != nil {
			t.Fatalf("resident replay[%d]: %v", i, err)
		}
		cos, _ := cosF32(opening[i], lr)
		p.replay = min(p.replay, cos)
	}
	return p
}

// olmoParityWithPlants runs the CPU once, the unplanted CUDA build and each plant; it returns the plants' results.
func olmoParityWithPlants(t *testing.T, ckpt, quant string) map[string]olmoParity {
	cpu, vocab := olmoCPULogits(t, ckpt, quant)
	base := olmoCUDARun(t, ckpt, quant, cpu, vocab, 0)
	t.Logf("%s at %s, %d tokens: worst cosine %.6f (tok1 %.6f, drift %.4f), argmax agree %d/%d, replay self-cosine %.9f",
		ckpt, quant, olmoParityTokens, base.worst, base.first, base.first-base.worst, base.agree, olmoParityTokens, base.replay)
	if base.red() {
		t.Errorf("olmo_hybrid on CUDA fails the DeltaNet families' bar against the CPU (worst >= 0.95, drift <= 0.02, replay >= 0.9999999)")
	}
	plants := map[string]olmoParity{}
	for _, pl := range []struct {
		name string
		kind int
	}{{"NegEigval dropped", dnSettingPlantDropNegEigval}, {"ONormEps dropped", dnSettingPlantDropONormEps}} {
		p := olmoCUDARun(t, ckpt, quant, cpu, vocab, pl.kind)
		t.Logf("plant %q: worst cosine %.6f (tok1 %.6f, drift %.4f), argmax agree %d/%d, red=%v",
			pl.name, p.worst, p.first, p.first-p.worst, p.agree, olmoParityTokens, p.red())
		plants[pl.name] = p
	}
	return plants
}

// O-2: the tiny fixture. The NegEigval plant must be red; the ONormEps plant is logged (the kernel chain carries it, O-1).
func TestOlmoHybridResidentParityCUDA_tiny(t *testing.T) {
	const ckpt = "../testdata/olmo_hybrid-tiny"
	if _, err := os.Stat(ckpt + "/model.safetensors"); err != nil {
		t.Skipf("no olmo_hybrid-tiny weights: %v", err)
	}
	plants := olmoParityWithPlants(t, ckpt, "int8int8")
	if !plants["NegEigval dropped"].red() {
		t.Errorf("the NegEigval plant is not red at the family bar on olmo_hybrid-tiny")
	}
}

// O-3: a real checkpoint, named by GOINFER_OLMO_HYBRID_CKPT (from ~/models, never the archive), at int4 on both sides. Both
// plants are logged; neither is required red.
func TestOlmoHybridResidentParityCUDA_real(t *testing.T) {
	ckpt := os.Getenv("GOINFER_OLMO_HYBRID_CKPT")
	if ckpt == "" {
		t.Skip("set GOINFER_OLMO_HYBRID_CKPT to a real Olmo Hybrid checkpoint")
	}
	olmoParityWithPlants(t, ckpt, "int4")
}

// O-4: olmo_hybrid-tiny builds resident on CUDA at each GPU KV precision, i8 among them (refused while program.CUDA did not
// claim the two settings).
func TestOlmoHybridResident_kvPrecisions(t *testing.T) {
	const ckpt = "../testdata/olmo_hybrid-tiny"
	if _, err := os.Stat(ckpt + "/model.safetensors"); err != nil {
		t.Skipf("no olmo_hybrid-tiny weights: %v", err)
	}
	for _, kv := range []string{"", "f32", "f16", "i8"} {
		m, err := decoder.Load(ckpt, decoder.Options{Backend: "cuda", Quant: "int8int8", KVPrecision: kv})
		if err != nil {
			t.Errorf("--kv %q: load: %v", kv, err)
			continue
		}
		if m.ResidentForwardForTest() == nil {
			t.Errorf("--kv %q: not resident (path %q): %s", kv, m.DecodePath(), m.ResidentDecline())
		} else {
			t.Logf("--kv %q: resident, path %s", kv, m.DecodePath())
		}
		m.Close()
	}
}
