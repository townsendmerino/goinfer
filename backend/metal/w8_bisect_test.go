//go:build darwin && goinfer_testhooks

package metal

import (
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestW8Native_perLayerBisect is the Metal int8 doc's next step after F3 and F2 failed (docs/tasks/task-metal-int8-2026-10.md,
// log 2026-10-02): name where Metal's native int8int8 leaves the CPU's int8int8, layer by layer, on the 0.5B. F1 holds
// every W8A8 GEMV bit-identical to the CPU given the same int8 inputs, so the gap enters elsewhere. The residual stream
// after each layer at the probe position (the last of a 32-token prose prompt, so attention and RoPE run over real
// history), against three references:
//   - cpu8: CPU int8int8, its f32 KV cache (F2/F3's reference);
//   - cpu8h: the same with every K and V rounded to f16 as it is stored, Metal's KV precision;
//   - the control pair: Metal int4 against CPU int4, the same path differences on the quant that passed F2.
//
// If Metal int8 matches cpu8h far better than cpu8, the f16 KV cache carries the gap. Exploratory, by day.
//
//	GOINFER_W8BISECT=1 go test -tags goinfer_testhooks -count=1 -run '^TestW8Native_perLayerBisect$' -v ./metal/
func TestW8Native_perLayerBisect(t *testing.T) {
	if os.Getenv("GOINFER_W8BISECT") != "1" {
		t.Skip("set GOINFER_W8BISECT=1 (loads the 0.5B four times)")
	}
	path := os.ExpandEnv("$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s", path)
	}
	seed := seedPrompt(t, path, "The quick brown fox jumps over the lazy dog while the old farmer watches from the porch, "+
		"counting the hours until the evening rain finally arrives over the hills and the river")
	if len(seed) > 32 {
		seed = seed[:32]
	}
	pos := len(seed) - 1

	cpuRun := func(quant string, roundKV bool) (hidden [][]float32, logits []float32) {
		m, err := decoder.Load(path, decoder.Options{Quant: quant})
		if err != nil {
			t.Fatalf("load cpu %s: %v", quant, err)
		}
		defer m.Close()
		_, nL, _, nKV, hd, _, _ := m.Dims()
		cache := decoder.NewKVCache(nL, nKV, hd, 0, 1024, nil)
		layers := make([]int, nL)
		for i := range layers {
			layers[i] = i
		}
		for i, id := range seed {
			var err error
			logits, hidden, err = m.ForwardCapture(id, cache, layers)
			if err != nil {
				t.Skipf("ForwardCapture: %v", err)
			}
			if roundKV {
				for l := range nL {
					k, v, _ := cache.LayerKVForTest(l)
					for _, s := range [][]float32{k[i*nKV*hd : (i+1)*nKV*hd], v[i*nKV*hd : (i+1)*nKV*hd]} {
						for j, x := range s {
							s[j] = f16ToF32(f32ToF16(x))
						}
					}
				}
			}
		}
		return hidden, append([]float32(nil), logits...)
	}
	metalRun := func(quant string, native bool) (hidden [][]float32, logits []float32) {
		prev := nativeInt8
		nativeInt8 = native
		defer func() { nativeInt8 = prev }()
		m, err := decoder.Load(path, decoder.Options{Quant: quant})
		if err != nil {
			t.Fatalf("load %s: %v", quant, err)
		}
		defer m.Close()
		r, err := buildResident(m)
		if err != nil {
			t.Fatalf("resident %s: %v", quant, err)
		}
		defer r.Close()
		if r.w8 != native {
			t.Fatalf("%s: native int8 = %v, want %v", quant, r.w8, native)
		}
		for i := range pos {
			r.forwardTrunkForTest(m.EmbedResidentForTest(seed[i]), i, r.nL)
		}
		for l := 1; l <= r.nL; l++ {
			hidden = append(hidden, append([]float32(nil), r.forwardTrunkForTest(m.EmbedResidentForTest(seed[pos]), pos, l)...))
		}
		return hidden, append([]float32(nil), r.ForwardEmb(m.EmbedResidentForTest(seed[pos]), pos)...)
	}
	cpu8, cpu8Lg := cpuRun("int8int8", false)
	cpu8h, cpu8hLg := cpuRun("int8int8", true)
	cpu4, cpu4Lg := cpuRun("int4", false)
	met8, met8Lg := metalRun("int8int8", true)
	met4, met4Lg := metalRun("int4", false)

	rel := func(a, b []float32) float64 {
		var d, n float64
		for i := range a {
			x := float64(a[i]) - float64(b[i])
			d += x * x
			n += float64(b[i]) * float64(b[i])
		}
		return math.Sqrt(d / n)
	}
	fmt.Fprintf(os.Stderr, "[w8-bisect] probe pos %d of a %d-token prompt; residual stream relative L2 after each layer\n", pos, len(seed))
	fmt.Fprintf(os.Stderr, "[w8-bisect] layer   met8-vs-cpu8  met8-vs-cpu8h(f16 KV)  cpu8h-vs-cpu8   met4-vs-cpu4\n")
	for l := range met8 {
		fmt.Fprintf(os.Stderr, "[w8-bisect] %5d   %11.3e  %20.3e  %12.3e  %12.3e\n", l,
			rel(met8[l], cpu8[l]), rel(met8[l], cpu8h[l]), rel(cpu8h[l], cpu8[l]), rel(met4[l], cpu4[l]))
	}
	kl := func(p, q []float32) float64 { // KL(softmax p ‖ softmax q)
		lse := func(x []float32) float64 {
			mx := math.Inf(-1)
			for _, v := range x {
				mx = math.Max(mx, float64(v))
			}
			s := 0.0
			for _, v := range x {
				s += math.Exp(float64(v) - mx)
			}
			return mx + math.Log(s)
		}
		lp, lq, k := lse(p), lse(q), 0.0
		for i := range p {
			a := float64(p[i]) - lp
			k += math.Exp(a) * (a - (float64(q[i]) - lq))
		}
		return k
	}
	fmt.Fprintf(os.Stderr, "[w8-bisect] logits at the probe: KL(cpu8 ‖ met8) %.3e, KL(cpu8h ‖ met8) %.3e, KL(cpu8 ‖ cpu8h) %.3e, KL(cpu4 ‖ met4) %.3e\n",
		kl(cpu8Lg, met8Lg), kl(cpu8hLg, met8Lg), kl(cpu8Lg, cpu8hLg), kl(cpu4Lg, met4Lg))
}

// TestW8Native_hiddenVsLogits splits F3′'s surprise (Metal int8int8 closer to f32 than the CPU's int8int8, 0.677×):
// over F3′'s 8 prompts, each position's final residual stream (before the final norm) and logits, CPU int8int8 and
// Metal native int8int8 each against CPU f32. If the hidden states are about equally far from f32 and the logits are not,
// the difference is the LM head (or the final norm), not the trunk. Exploratory, by day.
//
//	GOINFER_W8BISECT=1 go test -tags goinfer_testhooks -count=1 -run '^TestW8Native_hiddenVsLogits$' -v ./metal/
func TestW8Native_hiddenVsLogits(t *testing.T) {
	if os.Getenv("GOINFER_W8BISECT") != "1" {
		t.Skip("set GOINFER_W8BISECT=1")
	}
	path := os.ExpandEnv("$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s", path)
	}
	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatal(err)
	}
	files := decoder.PrefillGatePromptSetFor("a")
	const nP, steps = 8, 32
	prompts := make([][]int, nP)
	for i := range prompts {
		prompts[i] = decoder.PrefillGateProseIDsForTest(t, tk, files[i], steps)[:steps]
	}
	type run struct{ hidden, logits [][]float32 } // [prompt*steps+i]
	cpu := func(quant string) run {
		m, err := decoder.Load(path, decoder.Options{Quant: quant, ResidentContext: 1024})
		if err != nil {
			t.Skipf("load cpu %q: %v", quant, err)
		}
		defer m.Close()
		_, nL, _, nKV, hd, _, _ := m.Dims()
		var out run
		for p := range nP {
			cache := decoder.NewKVCache(nL, nKV, hd, 0, 1024, nil)
			for _, id := range prompts[p] {
				lg, h, err := m.ForwardCapture(id, cache, []int{nL - 1})
				if err != nil {
					t.Skipf("ForwardCapture: %v", err)
				}
				out.hidden = append(out.hidden, append([]float32(nil), h[0]...))
				out.logits = append(out.logits, append([]float32(nil), lg...))
			}
		}
		return out
	}
	f32, c8 := cpu(""), cpu("int8int8")
	prev := nativeInt8
	nativeInt8 = true
	m, err := decoder.Load(path, decoder.Options{Quant: "int8int8"})
	if err != nil {
		t.Fatal(err)
	}
	r, err := buildResident(m)
	if err != nil {
		t.Fatal(err)
	}
	nativeInt8 = prev
	var met run
	for p := range nP {
		for i, id := range prompts[p] {
			emb := m.EmbedResidentForTest(id)
			met.hidden = append(met.hidden, append([]float32(nil), r.forwardTrunkForTest(emb, i, r.nL)...))
			met.logits = append(met.logits, append([]float32(nil), r.ForwardEmb(emb, i)...))
		}
	}
	r.Close()
	m.Close()
	rel := func(a, b []float32) float64 {
		var d, n float64
		for i := range a {
			x := float64(a[i]) - float64(b[i])
			d += x * x
			n += float64(b[i]) * float64(b[i])
		}
		return math.Sqrt(d / n)
	}
	var hC, hM, kC, kM float64
	n := 0
	for p := range nP {
		for i := 2; i < steps; i++ {
			j := p*steps + i
			hC += rel(c8.hidden[j], f32.hidden[j])
			hM += rel(met.hidden[j], f32.hidden[j])
			kC += klLogits(f32.logits[j], c8.logits[j])
			kM += klLogits(f32.logits[j], met.logits[j])
			n++
		}
	}
	fmt.Fprintf(os.Stderr, "[w8-hidden] %d positions: final hidden rel L2 to f32: CPU int8 %.4f, Metal int8 %.4f; logits KL(f32 ‖ ·): CPU int8 %.5f, Metal int8 %.5f\n",
		n, hC/float64(n), hM/float64(n), kC/float64(n), kM/float64(n))
}
