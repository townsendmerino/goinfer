//go:build darwin && goinfer_testhooks

package metal

import (
	"compress/gzip"
	"encoding/gob"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestW8Native_F3amended_closerToF32 is gate F3 as the owner amended it on 2026-10-04 (docs/tasks/task-metal-int8-2026-10.md,
// "F3′"): Metal's native int8int8 (precise math, the owner's other decision that day) against the CPU's int8int8 at Metal's
// own KV precision (every K and V rounded to f16 as it is stored), over 8 prompts instead of one. Each prompt is the first
// 16 tokens of a prefill-gate set-A file, then 16 tokens of the CPU f32 model's greedy continuation; positions 2-31 are
// scored, 240 in all. Bar: the pooled mean KL(f32 ‖ Metal int8int8) is at most 1.10 × the pooled mean KL(f32 ‖ CPU
// int8int8, f16 KV). Reported beside it: the CPU int8int8 at f32 KV (F3's old reference), Metal int8 with fast math, and
// Metal int4. Each model loads once; the CPU ones one at a time.
func TestW8Native_F3amended_closerToF32(t *testing.T) {
	path := w8GateModel(t)
	// GOINFER_W8_F3_QUANT=int4mix runs the same gate for int4mix (M3, slice 4): every int8int8 arm below at int4mix,
	// the native arm with nativeInt4Mix on, and the int4 arm replaced by int4mix's re-quant.
	q8 := "int8int8"
	if os.Getenv("GOINFER_W8_F3_QUANT") == "int4mix" {
		q8 = "int4mix"
		prevMix := nativeInt4Mix
		nativeInt4Mix = true
		t.Cleanup(func() { nativeInt4Mix = prevMix })
	}
	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	files := decoder.PrefillGatePromptSetFor("a")
	const nPrompts, promptLen, steps = 8, 16, 32
	if len(files) < nPrompts {
		t.Fatalf("set A has %d files, want %d", len(files), nPrompts)
	}
	prompts := make([][]int, nPrompts)
	for i := range prompts {
		prompts[i] = decoder.PrefillGateProseIDsForTest(t, tk, files[i], promptLen)[:promptLen]
	}

	// cpu runs every prompt through one CPU model at quant: from the prompt then its own greedy continuation when forced
	// is nil, else the forced tokens; with f16KV, K and V are rounded to f16 right after the position that wrote them.
	cpu := func(quant string, f16KV bool, forced [][]int) (out [][][]float32, toks [][]int) {
		m, err := decoder.Load(path, decoder.Options{Quant: quant, ResidentContext: 1024})
		if errors.Is(err, decoder.ErrWontFitResident) {
			t.Skipf("the fit guard refused the CPU %q load on this machine now: %v", quant, err)
		}
		if err != nil {
			t.Fatalf("load (cpu, %q): %v", quant, err)
		}
		defer m.Close()
		_, nL, _, nKV, hd, _, _ := m.Dims()
		out, toks = make([][][]float32, nPrompts), make([][]int, nPrompts)
		for p := range nPrompts {
			cache := decoder.NewKVCache(nL, nKV, hd, 0, 1024, nil)
			out[p], toks[p] = make([][]float32, steps), make([]int, steps)
			tok := prompts[p][0]
			for i := range steps {
				if forced != nil {
					tok = forced[p][i]
				}
				toks[p][i] = tok
				l, err := m.ForwardForTest(tok, cache)
				if err != nil {
					t.Fatalf("cpu forward (%q) prompt %d at %d: %v", quant, p, i, err)
				}
				out[p][i] = append([]float32(nil), l...)
				if f16KV {
					for layer := range nL {
						k, v, _ := cache.LayerKVForTest(layer)
						for _, sl := range [][]float32{k[i*nKV*hd : (i+1)*nKV*hd], v[i*nKV*hd : (i+1)*nKV*hd]} {
							for j, x := range sl {
								sl[j] = f16ToF32(f32ToF16(x))
							}
						}
					}
				}
				if forced == nil {
					if i+1 < promptLen {
						tok = prompts[p][i+1]
					} else {
						tok = argmaxF(l)
					}
				}
			}
		}
		return out, toks
	}
	metal := func(quant string, native, fast bool, toks [][]int) [][][]float32 {
		prevN, prevF := nativeInt8, w8FastMath
		nativeInt8, w8FastMath = native, fast
		defer func() { nativeInt8, w8FastMath = prevN, prevF }()
		m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: quant, ResidentContext: 1024})
		if err != nil {
			t.Fatalf("load (metal, %s): %v", quant, err)
		}
		defer m.Close()
		rf := m.ResidentForwardForTest()
		if rf == nil {
			skipIfMemoryDeclined(t, m)
			t.Fatalf("metal resident declined at %s: %s", quant, m.ResidentDecline())
		}
		a, ok := rf.(*metalResident)
		if !ok || (a.r.w8 || a.r.w8Attn) != native || native && a.r.preciseMath == fast {
			t.Fatalf("%s: native %v precise %v, want native %v precise %v (%s)", quant, ok && (a.r.w8 || a.r.w8Attn), ok && a.r.preciseMath, native, !fast, m.DecodePath())
		}
		out := make([][][]float32, nPrompts)
		for p := range nPrompts {
			out[p] = make([][]float32, steps)
			for i, tok := range toks[p] {
				l, err := rf.Forward(m.EmbedResidentForTest(tok), i)
				if err != nil {
					t.Fatalf("metal forward (%s) prompt %d at %d: %v", quant, p, i, err)
				}
				out[p][i] = append([]float32(nil), l...)
			}
		}
		return out
	}

	// GOINFER_W8_F3_REF_IN: the CPU f32 reference from a file (decoder's TestW8F3Reference_write, run where the fit
	// guard admits the f32 model: nobara for the 1.5B), its prompts checked equal to the ones built here.
	var ref [][][]float32
	var toks [][]int
	if in := os.Getenv("GOINFER_W8_F3_REF_IN"); in != "" {
		ref, toks = readW8F3Ref(t, in, path, prompts, steps)
	} else {
		ref, toks = cpu("", false, nil)
	}
	cpu8h, _ := cpu(q8, true, toks)
	cpu8, _ := cpu(q8, false, toks)
	met8 := metal(q8, true, false, toks)
	met8fast := metal(q8, true, true, toks)
	var met4 [][][]float32
	if q8 == "int4mix" { // the re-quant arm: int4mix with the native path off
		prevMix := nativeInt4Mix
		nativeInt4Mix = false
		met4 = metal(q8, false, false, toks)
		nativeInt4Mix = prevMix
	} else {
		met4 = metal("int4", false, false, toks)
	}

	pooled := func(arm [][][]float32) (mean float64, perPrompt []float64) {
		n := 0
		for p := range nPrompts {
			var s float64
			for i := 2; i < steps; i++ {
				s += klLogits(ref[p][i], arm[p][i])
			}
			perPrompt = append(perPrompt, s/float64(steps-2))
			mean += s
			n += steps - 2
		}
		return mean / float64(n), perPrompt
	}
	kRef, pRef := pooled(cpu8h)
	kMet, pMet := pooled(met8)
	kCPU8, _ := pooled(cpu8)
	kFast, _ := pooled(met8fast)
	kMet4, _ := pooled(met4)
	further := 0
	for p := range nPrompts {
		for i := 2; i < steps; i++ {
			if klLogits(ref[p][i], met8[p][i]) > klLogits(ref[p][i], cpu8h[p][i]) {
				further++
			}
		}
	}
	per := ""
	for p := range nPrompts {
		per += fmt.Sprintf(" %.3f", pMet[p]/pRef[p])
	}
	t.Logf("F3′: pooled mean KL(f32 ‖ ·) over %d prompts × %d positions: CPU int8int8 f16 KV %.6f, Metal int8int8 (precise) %.6f = %.3f× (bar 1.10); further from f32 at %d of %d positions; per prompt%s",
		nPrompts, steps-2, kRef, kMet, kMet/kRef, further, nPrompts*(steps-2), per)
	t.Logf("F3′ reported: CPU int8int8 f32 KV %.6f, Metal int8int8 fast math %.6f (%.3f× the reference), Metal int4 %.6f",
		kCPU8, kFast, kFast/kRef, kMet4)
	if kMet > 1.10*kRef {
		t.Errorf("F3′ fails: KL(f32 ‖ Metal int8int8) %.6f is above 1.10 × the f16-KV CPU int8int8's %.6f", kMet, kRef)
	}
}

// readW8F3Ref reads F3′'s f32 reference written by decoder's TestW8F3Reference_write and checks it is this test's: the
// same checkpoint file name, the same prompts token for token, the same number of positions.
func readW8F3Ref(t *testing.T, in, path string, prompts [][]int, steps int) ([][][]float32, [][]int) {
	t.Helper()
	f, err := os.Open(in)
	if err != nil {
		t.Fatalf("reference: %v", err)
	}
	defer f.Close()
	zr, err := gzip.NewReader(f)
	if err != nil {
		t.Fatalf("reference: %v", err)
	}
	var r decoder.W8F3Ref
	if err := gob.NewDecoder(zr).Decode(&r); err != nil {
		t.Fatalf("reference: %v", err)
	}
	if filepath.Base(r.Model) != filepath.Base(path) || r.Pos != steps || len(r.Prompts) != len(prompts) {
		t.Fatalf("reference %s is for %s, %d positions, %d prompts; this run is %s, %d, %d", in, r.Model, r.Pos, len(r.Prompts), path, steps, len(prompts))
	}
	for p := range prompts {
		if !slices.Equal(r.Prompts[p], prompts[p]) {
			t.Fatalf("reference prompt %d differs from this run's", p)
		}
	}
	t.Logf("f32 reference read from %s (written on %s)", in, r.Arch)
	return r.Logits, r.Toks
}
